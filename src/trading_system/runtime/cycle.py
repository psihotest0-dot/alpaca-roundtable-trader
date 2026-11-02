"""One pass of the trading loop, with everything it needs handed in (task 1.7).

`TradingCycle` knows what to do in one step and nothing about when: `run_entries` turns the
newest closed bar into orders, `run_housekeeping` keeps the books and the protection in order.
`runtime.runner.Runner` decides when to call them. Because the dependencies are arguments, a
test can run a step against fakes without a loop, a thread or a sleep.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

import pandas as pd

from trading_system.broker.order_mapping import to_broker_order
from trading_system.data.live_price import UnusableLivePrice, check_live_price
from trading_system.domain.models import (
    AccountSnapshot,
    BrokerOrder,
    PositionSnapshot,
    StrategySignal,
    TradingState,
)
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
        self._fills_after: Optional[datetime] = None

    @property
    def watchlist(self) -> list[str]:
        return [s.strip() for s in self.settings.watchlist.split(",") if s.strip()]

    # ------------------------------------------------------------------ entries

    def run_entries(self, now: datetime, expected_bar_start: datetime) -> EntriesReport:
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
                if self._enter(signal, state, bars_15m, now):
                    report.submitted += 1
            except Exception as e:
                logger.error(f"Error in risk evaluation / execution for {signal.symbol}: {e}")
        return report

    def _enter(
        self,
        signal: StrategySignal,
        state: TradingState,
        bars_15m: dict[str, pd.DataFrame],
        now: datetime,
    ) -> bool:
        """One signal through Oracle, live price, risk and execution. True if an order went."""
        logger.info(f"Processing signal: {signal.symbol} {signal.direction.value}")
        try:
            self.signal_repo.save_signal(signal)
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")

        if self.oracle and not self._oracle_approves(signal, bars_15m):
            return False

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
            logger.warning(f"No entry for {signal.symbol}: {e}")
            return False
        logger.info(f"Live price for {signal.symbol}: {price} (last bar close {bar_close})")

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
        if not (decision.approved and intent):
            return False

        logger.info(f"Executing order intent for {intent.symbol}...")
        return self.execution.submit_order(intent) is not None

    def _oracle_approves(self, signal: StrategySignal, bars_15m: dict[str, pd.DataFrame]) -> bool:
        logger.info("Querying Oracle...")
        technicals: dict[str, Any] = {}
        if signal.symbol in bars_15m:
            last_bar = bars_15m[signal.symbol].iloc[-1]
            for column in last_bar.index:
                value = last_bar[column]
                technicals[str(column)] = (
                    float(value) if isinstance(value, (int, float)) else str(value)
                )
        decision = self.oracle.evaluate_signal(
            signal, technicals=technicals, rationale="Trend strategy signal generated."
        )
        logger.info(f"Oracle decision: {decision.approved} ({decision.reasoning})")
        return bool(decision.approved)

    def _snapshot(
        self, now: datetime
    ) -> tuple[AccountSnapshot, list[PositionSnapshot], list[BrokerOrder]]:
        """The account as it is right now. Taken per signal: an order placed for the previous
        signal of this cycle must count towards the limits of the next one."""
        account = self.broker.get_account()
        snapshot = AccountSnapshot(
            equity=Decimal(str(account.equity)),
            cash=Decimal(str(account.cash)),
            buying_power=Decimal(str(account.buying_power)),
            day_trade_count=int(account.daytrade_count),
            updated_at=now,
        )
        positions = [
            PositionSnapshot(
                symbol=p.symbol,
                quantity=Decimal(str(p.qty)),
                market_value=Decimal(str(p.market_value)),
                average_entry_price=Decimal(str(p.avg_entry_price)),
                unrealized_pl=Decimal(str(p.unrealized_pl)),
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
