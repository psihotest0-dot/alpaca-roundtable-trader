"""One pass of the trading loop, with everything it needs handed in (task 1.7).

`TradingCycle` knows what to do in one step and nothing about when: `run_entries` turns the
newest closed bar into orders, `run_housekeeping` keeps the books and the protection in order.
`runtime.runner.Runner` decides when to call them. Because the dependencies are arguments, a
test can run a step against fakes without a loop, a thread or a sleep.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

import pandas as pd

from trading_system.broker.order_mapping import to_broker_order
from trading_system.data.live_price import UnusableLivePrice, check_live_price
from trading_system.domain.enums import OracleVerdict
from trading_system.domain.models import (
    AccountSnapshot,
    BrokerOrder,
    OracleDecision,
    PositionSnapshot,
    StrategySignal,
    TradingState,
)
from trading_system.oracle.inputs import build_oracle_request
from trading_system.risk.daily_equity import with_daily_start_equity
from trading_system.runtime.reconciliation import reconcile_orders, record_fills
from trading_system.strategy.config import StrategyConfig
from trading_system.strategy.trend import filter_signals, generate_signals

logger = logging.getLogger(__name__)

# How far back the first look for fills goes when the system starts.
FILLS_LOOKBACK = timedelta(days=2)


@dataclass
class EntriesReport:
    """What a pass over the newest bar did."""

    skipped: Optional[str] = None  # why nothing was done, if nothing was
    retry_later: bool = False  # True: nothing was decided because of a gap that may close
    signals: int = 0
    submitted: int = 0
    held: int = 0  # signals the Oracle asked to wait on; they are asked again each tick
    errors: int = 0  # signals that broke in risk or execution: no order, and the cause is a bug


class Outcome(Enum):
    """What became of one signal on one pass."""

    ORDERED = "ordered"  # an order went to the broker
    DROPPED = "dropped"  # final: vetoed, no usable price, rejected by risk, ...
    HELD = "held"  # the Oracle said "not now": try again while the signal is valid


@dataclass
class HeldSignal:
    signal: StrategySignal
    # What the signal was generated from, for asking again.
    bars_15m: dict[str, pd.DataFrame]
    bars_1h: dict[str, pd.DataFrame]
    asked: int = 1


def _amount(value: Any, what: str) -> Decimal:
    """A number Alpaca sends as text. Every field of its models can come back empty (the paper
    account leaves several empty), and an empty amount is not zero: the cycle stops with a
    message that says which one, instead of a bare error from `Decimal(str(None))`."""
    if value is None:
        raise ValueError(f"Alpaca did not return {what}")
    return Decimal(str(value))


def ensure_daily_start_equity(
    broker: Any, state_repo: Any, state: Optional[TradingState], now: datetime
) -> TradingState:
    """`state` with today's start-of-day equity, recorded once per trading day.

    Without it the risk engine rejected every signal ("Missing daily start equity"), so
    the system could never open a position. Failure to record it is logged, not fatal:
    the risk engine then rejects entries (fail closed) and the next cycle tries again."""
    state = state if state is not None else TradingState()
    try:
        account = broker.get_account()
        previous_close = getattr(account, "last_equity", None)
        updated = with_daily_start_equity(
            state,
            now=now,
            equity=Decimal(str(account.equity)),
            previous_close_equity=Decimal(str(previous_close)) if previous_close else None,
        )
        if updated is not state and updated.daily_start_equity and updated.daily_equity_date:
            state_repo.save_daily_start_equity(
                updated.daily_start_equity, updated.daily_equity_date
            )
            logger.info(
                f"Daily start equity for {updated.daily_equity_date.date()}: "
                f"{updated.daily_start_equity}"
            )
        return updated
    except Exception as e:
        logger.error(f"Could not record the daily start equity: {e}")
        return state


class TradingCycle:
    def __init__(
        self,
        *,
        settings: Any,
        broker: Any,
        data: Any,
        risk_engine: Any,
        execution: Any,
        signal_repo: Any,
        order_repo: Any,
        state_repo: Any,
        strategy_config: StrategyConfig,
        oracle: Any = None,
        risk_repo: Any = None,
        audit_repo: Any = None,
        alerts: Any = None,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.data = data
        self.risk_engine = risk_engine
        self.execution = execution
        self.signal_repo = signal_repo
        self.order_repo = order_repo
        self.state_repo = state_repo
        self.strategy_config = strategy_config
        self.oracle = oracle
        self.risk_repo = risk_repo
        self.audit_repo = audit_repo
        self.alerts = alerts
        self._fills_after: Optional[datetime] = None
        self._held: dict[str, HeldSignal] = {}

    @property
    def watchlist(self) -> list[str]:
        return list(self.settings.watchlist_symbols)

    # ------------------------------------------------------------------ entries

    def run_entries(self, now: datetime, expected_bar_start: datetime) -> EntriesReport:
        """One pass over the newest bar, recorded: the outcome goes into the journal, and an
        error in a signal raises an alert. (The pass itself is `_run_entries`.)"""
        try:
            report = self._run_entries(now, expected_bar_start)
        except Exception as e:
            self._journal(now, expected_bar_start, EntriesReport(errors=1, skipped=f"crashed: {e}"))
            self._alert("Entry look-up crashed", f"{type(e).__name__}: {e}")
            raise
        self._journal(now, expected_bar_start, report)
        if report.errors:
            self._alert(
                "Signals broke before they became orders",
                f"{report.errors} of {report.signals} signals failed in risk or execution on the "
                f"bar of {expected_bar_start:%H:%M} UTC; no order was placed for them. "
                "See the error lines in the log.",
            )
        return report

    def _journal(self, now: datetime, bar_start: datetime, report: EntriesReport) -> None:
        if self.audit_repo is None:
            return
        try:
            self.audit_repo.record(
                "entries_cycle",
                {
                    "bar_start": bar_start.isoformat(),
                    "skipped": report.skipped,
                    "retry_later": report.retry_later,
                    "signals": report.signals,
                    "submitted": report.submitted,
                    "held": report.held,
                    "errors": report.errors,
                    # The account's equity at this pass: the daily report reads the day's
                    # course from these (nothing else records it between the two baselines).
                    "equity": self._equity_now(),
                },
                when=now,
            )
        except Exception as e:  # a journal that cannot be written must not stop the trading
            logger.warning(f"Could not journal the entry look-up: {e}")

    def _equity_now(self) -> Optional[str]:
        try:
            return str(_amount(self.broker.get_account().equity, "the account's equity"))
        except Exception:  # the journal is better with a gap than with no entry
            return None

    def _alert(self, title: str, message: str) -> None:
        """A CRITICAL log line, and a journal entry that outlives the log."""
        if self.alerts is not None:
            self.alerts.send_alert(title, message)
        if self.audit_repo is not None:
            try:
                self.audit_repo.record("alert", {"title": title, "message": message})
            except Exception as e:
                logger.warning(f"Could not journal the alert: {e}")

    def _run_entries(self, now: datetime, expected_bar_start: datetime) -> EntriesReport:
        """Signals from the newest closed bar, through risk, into orders.

        `expected_bar_start` is the start of the bar that should be the newest by now. If the
        data does not have it yet, nothing is decided (a signal from the previous bar would be
        a stale one) and the report says so."""
        state = ensure_daily_start_equity(
            self.broker, self.state_repo, self.state_repo.get_state(), now
        )
        if state.is_trading_halted:
            logger.info(f"Trading is halted ({state.halt_reason}); no entries this cycle.")
            return EntriesReport(skipped="halted")

        logger.info("Fetching market data...")
        try:
            bars_15m, bars_1h = self.data.load_universe(
                self.watchlist, now, self.settings.lookback_days, self.settings.htf_lookback_days
            )
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            return EntriesReport(skipped="market data unavailable", retry_later=True)

        newest = max((df.index[-1] for df in bars_15m.values()), default=None)
        if newest is None or pd.Timestamp(newest) < pd.Timestamp(expected_bar_start):
            logger.warning(
                f"The bar that started at {expected_bar_start:%H:%M} is not available yet "
                f"(newest: {newest})."
            )
            return EntriesReport(skipped="bar not available yet", retry_later=True)

        logger.info("Generating strategy signals...")
        signals = generate_signals(bars_15m, bars_1h, self.strategy_config, now)

        try:
            positions = self.broker.get_positions()
            open_orders = self.broker.get_open_orders()
        except Exception as e:
            # Without them there is no telling which symbols are already held: no entries.
            logger.error(f"Could not read positions and orders; no entries this cycle: {e}")
            return EntriesReport(
                skipped="positions unavailable", signals=len(signals), retry_later=True
            )

        active_symbols = {p.symbol for p in positions} | {o.symbol for o in open_orders}
        signals = filter_signals(signals, active_symbols)
        logger.info(f"Generated {len(signals)} valid signals.")

        report = EntriesReport(signals=len(signals))
        for signal in signals:
            try:
                outcome = self._enter(signal, state, bars_15m, bars_1h, now)
            except Exception as e:
                logger.error(f"Error in risk evaluation / execution for {signal.symbol}: {e}")
                report.errors += 1
                continue
            if outcome is Outcome.ORDERED:
                report.submitted += 1
            elif outcome is Outcome.HELD:
                report.held += 1
                self._held[signal.signal_id or ""] = HeldSignal(signal, bars_15m, bars_1h)
        return report

    def revisit_held(self, now: datetime) -> int:
        """Asks the Oracle again about every signal it told us to hold, while the signal is
        still valid. Returns how many signals were ordered. Called on every tick of the runner
        that did not look for new entries."""
        if not self._held:
            return 0
        state = ensure_daily_start_equity(
            self.broker, self.state_repo, self.state_repo.get_state(), now
        )
        if state.is_trading_halted:
            return 0  # they stay held; a resumed system carries on with them if still valid
        ordered = 0
        for signal_id, held in list(self._held.items()):
            signal = held.signal
            if signal.expires_at is not None and signal.expires_at <= now:
                logger.info(f"Held signal {signal.symbol} expired without an answer; dropped.")
                del self._held[signal_id]
                continue
            held.asked += 1
            try:
                outcome = self._enter(signal, state, held.bars_15m, held.bars_1h, now)
            except Exception as e:
                logger.error(f"Error re-examining held signal {signal.symbol}: {e}")
                continue
            if outcome is not Outcome.HELD:
                del self._held[signal_id]
            if outcome is Outcome.ORDERED:
                ordered += 1
        return ordered

    def _enter(
        self,
        signal: StrategySignal,
        state: TradingState,
        bars_15m: dict[str, pd.DataFrame],
        bars_1h: dict[str, pd.DataFrame],
        now: datetime,
    ) -> Outcome:
        """One signal through live price, Oracle, risk and execution. The cheap check comes
        first: no Oracle call is spent on a signal that cannot be traded at a usable price."""
        logger.info(f"Processing signal: {signal.symbol} {signal.direction.value}")
        try:
            self.signal_repo.save_signal(signal)
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")

        # The entry is planned around the last trade. The bar close is up to 15 minutes old
        # and only serves as the yardstick for a bad print.
        bar_close = Decimal(str(bars_15m[signal.symbol].iloc[-1]["close"]))
        try:
            live = self.data.fetch_latest_prices([signal.symbol]).get(signal.symbol)
        except Exception as e:
            logger.error(f"Could not fetch a live price for {signal.symbol}: {e}")
            live = None
        try:
            price = check_live_price(
                live,
                reference_close=bar_close,
                now=datetime.now(timezone.utc),
                max_age_seconds=self.settings.live_price_max_age_seconds,
                max_deviation_pct=Decimal(str(self.settings.live_price_max_deviation_pct)),
            )
        except UnusableLivePrice as e:
            # Not final: a quiet symbol prints again in a minute. The signal is tried on the
            # next tick and lapses with its own expiry if the price never becomes usable.
            logger.warning(f"No entry for {signal.symbol} yet: {e}")
            return Outcome.HELD
        logger.info(f"Live price for {signal.symbol}: {price} (last bar close {bar_close})")

        if self.oracle:
            verdict = self._ask_oracle(signal, state, bars_15m, bars_1h, price, now).verdict
            if verdict is OracleVerdict.VETO:
                return Outcome.DROPPED
            if verdict is OracleVerdict.HOLD:
                return Outcome.HELD

        logger.info("Evaluating risk...")
        account, positions, active_orders = self._snapshot(now)
        decision, intent = self.risk_engine.evaluate_signal(
            signal=signal,
            account=account,
            positions=positions,
            active_orders=active_orders,
            current_price=price,
            state=state,
        )
        logger.info(f"Risk decision: {decision.approved} ({decision.reasoning})")
        self._save_risk_decision(decision)
        if not (decision.approved and intent):
            return Outcome.DROPPED

        logger.info(f"Executing order intent for {intent.symbol}...")
        return (
            Outcome.ORDERED if self.execution.submit_order(intent) is not None else Outcome.DROPPED
        )

    def _save_risk_decision(self, decision: Any) -> None:
        if self.risk_repo is None:
            return
        try:
            self.risk_repo.save(decision)
        except Exception as e:  # the decision still stands; only its record is missing
            logger.warning(f"Could not save the risk decision: {e}")

    def _ask_oracle(
        self,
        signal: StrategySignal,
        state: TradingState,
        bars_15m: dict[str, pd.DataFrame],
        bars_1h: dict[str, pd.DataFrame],
        price: Decimal,
        now: datetime,
    ) -> OracleDecision:
        """The Oracle's verdict on a signal. A provider is meant never to raise; if one does,
        that is a HOLD marked as a failure, not a veto: an Oracle that could not answer has
        said nothing about the signal."""
        logger.info("Querying Oracle...")
        account, positions, _ = self._snapshot(now)
        daily_pnl_pct = None
        if state.daily_start_equity:
            daily_pnl_pct = round(
                float((account.equity - state.daily_start_equity) / state.daily_start_equity * 100),
                3,
            )
        request = build_oracle_request(
            signal,
            bars_15m[signal.symbol],
            bars_1h[signal.symbol],
            self.strategy_config,
            live_price=price,
            context={
                "equity": float(account.equity),
                "daily_pnl_pct": daily_pnl_pct,
                "open_positions": len(positions),
                "max_open_positions": self.settings.max_open_positions,
                "stop_loss_pct": self.settings.stop_loss_pct * 100,
                "take_profit_pct": self.settings.take_profit_pct * 100,
            },
        )
        try:
            decision: OracleDecision = self.oracle.evaluate(request)
        except Exception as e:
            decision = OracleDecision(
                signal_id=signal.signal_id or "",
                verdict=OracleVerdict.HOLD,
                reason=f"The Oracle raised: {e}",
                provider=getattr(self.oracle, "name", ""),
                failed=True,
            )
        log = logger.warning if decision.failed else logger.info
        log(
            f"Oracle {decision.provider or '?'} on {signal.symbol}: {decision.verdict.value}"
            f"{' (no answer)' if decision.failed else ''} - {decision.reason}"
        )
        return decision

    def _snapshot(
        self, now: datetime
    ) -> tuple[AccountSnapshot, list[PositionSnapshot], list[BrokerOrder]]:
        """The account as it is right now. Taken per signal: an order placed for the previous
        signal of this cycle must count towards the limits of the next one."""
        account = self.broker.get_account()
        snapshot = AccountSnapshot(
            equity=_amount(account.equity, "the account's equity"),
            cash=_amount(account.cash, "the account's cash"),
            buying_power=_amount(account.buying_power, "the account's buying power"),
            day_trade_count=(
                None if account.daytrade_count is None else int(account.daytrade_count)
            ),
            updated_at=now,
        )
        positions = [
            PositionSnapshot(
                symbol=p.symbol,
                quantity=_amount(p.qty, f"the quantity of {p.symbol}"),
                market_value=_amount(p.market_value, f"the market value of {p.symbol}"),
                average_entry_price=_amount(
                    p.avg_entry_price, f"the average entry price of {p.symbol}"
                ),
                unrealized_pl=_amount(p.unrealized_pl, f"the unrealized P&L of {p.symbol}"),
            )
            for p in self.broker.get_positions()
        ]
        active_orders = [to_broker_order(o) for o in self.broker.get_open_orders()]
        return snapshot, positions, active_orders

    # ------------------------------------------------------------ housekeeping

    def run_housekeeping(self, now: datetime) -> None:
        """What has to happen whether or not anything is traded, and also while trading is
        halted: orders brought up to date from the broker, stale entries cancelled, positions
        without a stop found, fills recorded. A halt stops new entries; it must not stop
        anyone from watching the ones that exist."""
        steps = (
            ("reconcile orders", lambda: self._reconcile(now)),
            ("cancel stale entries", self.execution.check_bounded_entry_expiry),
            ("check protection", self.execution.emergency_check_unprotected_positions),
            ("record fills", lambda: self._record_fills(now)),
        )
        for name, step in steps:
            try:
                step()
            except Exception:
                logger.exception(f"Housekeeping step failed: {name}")

    def _reconcile(self, now: datetime) -> None:
        report = reconcile_orders(self.broker, self.execution, self.order_repo, now)
        if report.failed:
            logger.warning(f"Could not refresh {len(report.failed)} order(s): {report.failed}")

    def _record_fills(self, now: datetime) -> None:
        after = self._fills_after or now - FILLS_LOOKBACK
        _, self._fills_after = record_fills(self.broker, self.order_repo, after)
