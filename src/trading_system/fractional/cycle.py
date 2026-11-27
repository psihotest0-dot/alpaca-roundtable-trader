"""The loop of the fractional paper trial (docs/fractional-paper-trial.md).

It trades the live loop's signals with REAL paper orders in the SECOND paper account, in fractional
quantities. Alpaca takes no bracket, no GTC stop, and no stop together with a target on one quantity
for a fractional order (docs/decisions.md, 2026-11-19), so protection is built differently from the
live loop's:

* the entry is a DAY limit order; the moment it fills, a DAY stop-sell for the whole quantity is placed
  from the REAL average entry price, and it is placed again whenever it is missing (the next tick,
  also while the market is closed: a DAY order placed after the close is checked at the open);
* the take-profit is software: at the target price the stop is cancelled and the position sold at
  market;
* a position with no standing stop while the market is open is "unprotected": journalled, alarmed, and
  after five minutes the entries stop (a marker file) until `fractional-resume`.

Nothing here touches the live loop's trading state (its halt flag, its orders), its Risk or
Execution. Whatever happens is journalled in `audit_events` under `fractional_*` events.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopOrderRequest

from trading_system.data.live_price import UnusableLivePrice, check_live_price
from trading_system.data.universe import MARKET_SYMBOL, observation_universe
from trading_system.fractional.plan import (
    QUANTITY_TOLERANCE,
    HardCapExceeded,
    Refused,
    live_stops_of,
    size_entry,
    stop_cover,
    stop_price_for,
    target_price_for,
    virtual_equity,
)
from trading_system.runtime.cycle import EntriesReport
from trading_system.strategy.config import StrategyConfig
from trading_system.strategy.trend import filter_signals, generate_signals

logger = logging.getLogger(__name__)

EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

# An entry that has not filled by now is cancelled (as in the live loop).
ENTRY_MAX_AGE = timedelta(minutes=15)
# A position without a standing stop for this long, with the market open, stops the entries.
UNPROTECTED_HALT_AFTER = timedelta(minutes=5)
# After Alpaca refused a stop (a 4xx answer) the next try is this much later; after a failure that may
# pass (no network, 429, 5xx) it is the shorter one.
REFUSED_RETRY = timedelta(minutes=5)
TRANSIENT_RETRY = timedelta(seconds=60)
# Waiting for the fill of an entry (to place the stop at once) and for a cancelled stop to let go.
FILL_POLLS = 10
FILL_POLL_SECONDS = 2.0
CANCEL_POLLS = 10
CANCEL_POLL_SECONDS = 1.0

BASELINE_EVENT = "fractional_baseline"
CYCLE_EVENT = "fractional_cycle"


def _text(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _decimal(value: Any, what: str) -> Decimal:
    if value is None:
        raise ValueError(f"Alpaca sent no value for {what}.")
    return Decimal(str(value))


def _when(order: Any) -> Optional[datetime]:
    moment: Optional[datetime] = getattr(order, "submitted_at", None) or getattr(
        order, "created_at", None
    )
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


class FractionalCycle:
    def __init__(
        self,
        *,
        settings: Any,
        broker: Any,
        data: Any,
        audit_repo: Any,
        strategy_config: StrategyConfig,
        alerts: Any = None,
        halt_path: Optional[Path] = None,
        sleep: Callable[[float], Any] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.data = data
        self.audit_repo = audit_repo
        self.strategy_config = strategy_config
        self.alerts = alerts
        self.halt_path = halt_path or Path(settings.fractional_halt_path)
        self._sleep = sleep
        self._clock = clock
        self._baseline: Optional[Decimal] = None
        self._unprotected_since: dict[str, datetime] = {}
        self._retry_after: dict[str, datetime] = {}
        self._alerted: set[str] = set()

    @property
    def watchlist(self) -> list[str]:
        """The symbols the trial trades (docs/fractional-paper-trial.md, section 5)."""
        if self.settings.fractional_symbols == "watchlist":
            return list(self.settings.watchlist_symbols)
        return [
            s for s in observation_universe(self.settings.watchlist_symbols) if s != MARKET_SYMBOL
        ]

    # ------------------------------------------------------------------ small things

    def _journal(self, event: str, details: dict[str, Any], now: datetime) -> None:
        """Whatever cannot be written is a gap in the journal and nothing else."""
        try:
            self.audit_repo.record(event, details, when=now)
        except Exception as e:
            logger.warning(f"Could not journal {event}: {e}")

    def _alert(self, key: str, title: str, message: str, now: datetime) -> None:
        """A CRITICAL `ALERT:` line and a journal entry; the same key is said once a day."""
        marker = f"{key}:{now.date()}"
        if marker in self._alerted:
            return
        self._alerted.add(marker)
        if self.alerts is not None:
            self.alerts.send_alert(title, message)
        self._journal("fractional_alert", {"title": title, "message": message}, now)

    def halted(self) -> bool:
        return self.halt_path.exists()

    def halt(self, reason: str, now: datetime) -> None:
        self.halt_path.parent.mkdir(parents=True, exist_ok=True)
        self.halt_path.write_text(f"{now.isoformat()} {reason}\n", encoding="utf-8")
        self._alert("halt", "Fractional entries stopped", reason, now)

    def resume(self, now: datetime) -> list[str]:
        """Lifts the halt if every position has a standing stop. Returns the symbols that still have
        none (then the halt stays)."""
        cover = stop_cover(list(self.broker.get_open_orders()))
        missing = [
            p.symbol
            for p in self.broker.get_positions()
            if _decimal(p.qty, f"the quantity of {p.symbol}") - cover.get(p.symbol, Decimal(0))
            > QUANTITY_TOLERANCE
        ]
        if not missing and self.halted():
            self.halt_path.unlink()
            self._journal("fractional_resumed", {}, now)
        return missing

    def baseline_equity(self, now: datetime) -> Decimal:
        """The account's equity when the trial began, written once to the journal and read from it
        afterwards. Written without a safety net: a baseline that is lost would be taken again at the
        next start, and the virtual capital would silently jump."""
        if self._baseline is not None:
            return self._baseline
        found = self.audit_repo.events(BASELINE_EVENT, EPOCH, now + timedelta(days=1))
        if found:
            self._baseline = Decimal(str(found[0][1]["equity"]))
        else:
            equity = _decimal(self.broker.get_account().equity, "the account's equity")
            self.audit_repo.record(
                BASELINE_EVENT,
                {"equity": str(equity), "budget": str(self.settings.fractional_budget_usd)},
                when=now,
            )
            self._baseline = equity
        return self._baseline

    def _virtual_equity(self, account: Any, now: datetime) -> Decimal:
        return virtual_equity(
            Decimal(str(self.settings.fractional_budget_usd)),
            _decimal(account.equity, "the account's equity"),
            self.baseline_equity(now),
        )

    # ------------------------------------------------------------------ entries

    def run_entries(self, now: datetime, expected_bar_start: datetime) -> EntriesReport:
        try:
            report = self._run_entries(now, expected_bar_start)
        except Exception as e:
            self._journal(
                CYCLE_EVENT,
                {
                    "bar_start": expected_bar_start.isoformat(),
                    "skipped": f"crashed: {e}",
                    "errors": 1,
                },
                now,
            )
            raise
        self._journal(
            CYCLE_EVENT,
            {
                "bar_start": expected_bar_start.isoformat(),
                "skipped": report.skipped,
                "retry_later": report.retry_later,
                "signals": report.signals,
                "submitted": report.submitted,
                "errors": report.errors,
                **self._numbers(now),
            },
            now,
        )
        return report

    def _numbers(self, now: datetime) -> dict[str, Any]:
        """The virtual capital and the gross exposure after the pass, for the report. A pass is not
        stopped by a number it could not read."""
        try:
            positions = list(self.broker.get_positions())
            equity = self._virtual_equity(self.broker.get_account(), now)
            gross = sum((_decimal(p.market_value, "market value") for p in positions), Decimal(0))
            return {
                "virtual_equity": str(equity),
                "positions": len(positions),
                "gross": str(gross),
                "gross_pct": str(gross / equity) if equity > 0 else None,
            }
        except Exception as e:
            logger.debug(f"Could not read the numbers of the pass: {e}")
            return {}

    def _run_entries(self, now: datetime, expected_bar_start: datetime) -> EntriesReport:
        if self.halted():
            logger.info("Fractional entries are halted (the marker file exists); none this pass.")
            return EntriesReport(skipped="halted")
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

        signals = generate_signals(bars_15m, bars_1h, self.strategy_config, now)
        try:
            positions = self.broker.get_positions()
            orders = self.broker.get_open_orders()
        except Exception as e:
            logger.error(f"Could not read positions and orders; no entries this pass: {e}")
            return EntriesReport(
                skipped="positions unavailable", signals=len(signals), retry_later=True
            )
        active = {p.symbol for p in positions} | {o.symbol for o in orders}
        signals = filter_signals(signals, active)

        report = EntriesReport(signals=len(signals))
        for signal in signals:
            try:
                if self._enter(signal, bars_15m, now):
                    report.submitted += 1
            except HardCapExceeded as e:
                logger.error(f"HARD CAP for {signal.symbol}: {e}")
                self._alert("hardcap", "Fractional order above the hard ceiling", str(e), now)
                report.errors += 1
            except Exception as e:  # one broken signal must not stop the others
                logger.error(f"Error entering {signal.symbol}: {e}")
                report.errors += 1
        return report

    def _enter(self, signal: Any, bars_15m: dict[str, pd.DataFrame], now: datetime) -> bool:
        """One signal to a DAY limit order. False if it was refused (and journalled why)."""
        symbol = signal.symbol

        def refuse(reason: str) -> bool:
            logger.info(f"Not entering {symbol}: {reason}")
            self._journal(
                "fractional_refused",
                {"symbol": symbol, "signal_id": signal.signal_id, "reason": reason},
                now,
            )
            return False

        if signal.expires_at is not None and signal.expires_at < now:
            return refuse("the signal has expired")

        account = self.broker.get_account()
        positions = list(self.broker.get_positions())
        orders = list(self.broker.get_open_orders())
        pending = [o for o in orders if _text(o.side) == "buy"]
        if any(o.symbol == symbol for o in pending) or any(p.symbol == symbol for p in positions):
            return refuse("a position or an entry for the symbol exists")
        asset = self.broker.get_asset(symbol)
        if not (asset.tradable and asset.fractionable):
            return refuse("the asset is not tradable or not fractionable")

        stop_pct = Decimal(str(self.settings.stop_loss_pct))
        open_gross = sum((_decimal(p.market_value, "market value") for p in positions), Decimal(0))
        open_risk = sum(
            (
                _decimal(p.avg_entry_price, "entry price") * stop_pct * _decimal(p.qty, "quantity")
                for p in positions
            ),
            Decimal(0),
        )
        for o in pending:
            price = _decimal(o.limit_price, "limit price")
            quantity = _decimal(o.qty, "quantity")
            open_gross += price * quantity
            open_risk += price * stop_pct * quantity

        reference = Decimal(str(bars_15m[symbol].iloc[-1]["close"]))
        try:
            live = self.data.fetch_latest_prices([symbol]).get(symbol)
            price = check_live_price(
                live,
                reference_close=reference,
                now=datetime.now(timezone.utc),
                max_age_seconds=self.settings.live_price_max_age_seconds,
                max_deviation_pct=Decimal(str(self.settings.live_price_max_deviation_pct)),
            )
        except UnusableLivePrice as e:
            return refuse(f"no usable live price: {e}")

        equity = self._virtual_equity(account, now)
        try:
            sized = size_entry(
                equity=equity,
                reference_price=price,
                open_positions=len(positions) + len(pending),
                open_gross=open_gross,
                open_risk=open_risk,
                max_positions=self.settings.max_open_positions,
                position_size_pct=Decimal(str(self.settings.position_size_pct)),
                max_gross_pct=Decimal(str(self.settings.max_gross_exposure_pct)),
                max_trade_risk_pct=Decimal(str(self.settings.max_planned_risk_per_trade_pct)),
                max_total_risk_pct=Decimal(str(self.settings.max_total_planned_open_risk_pct)),
                limit_buffer_pct=Decimal(str(self.settings.entry_limit_buffer_pct)),
                stop_loss_pct=stop_pct,
                take_profit_pct=Decimal(str(self.settings.take_profit_pct)),
                max_order_notional=Decimal(str(self.settings.fractional_max_order_usd)),
            )
        except HardCapExceeded:
            raise
        except Refused as e:
            return refuse(str(e))

        client_order_id = f"frac-{signal.signal_id}"[:128]
        request = LimitOrderRequest(
            symbol=symbol,
            qty=float(sized.quantity),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=float(sized.limit_price),
            client_order_id=client_order_id,
        )
        order = self.broker.submit_order(request)
        self._journal(
            "fractional_entry",
            {
                "symbol": symbol,
                "signal_id": signal.signal_id,
                "order_id": str(order.id),
                "client_order_id": client_order_id,
                "quantity": str(sized.quantity),
                "limit_price": str(sized.limit_price),
                "notional": str(sized.notional),
                "planned_risk": str(sized.planned_risk),
                "virtual_equity": str(equity),
                "reference_close": str(reference),
                "live_price": str(price),
            },
            now,
        )
        logger.info(
            f"Fractional entry {symbol}: {sized.quantity} at limit {sized.limit_price} "
            f"({sized.notional:.2f} dollars)."
        )
        if self._after_entry(order, sized.limit_price, reference, now) == "rejected":
            # Sent, but not an entry: Alpaca refused it at once (the report must not count it)
            logger.warning(f"Alpaca rejected the fractional entry {symbol} ({order.id}).")
            self._journal(
                "fractional_entry_rejected",
                {
                    "symbol": symbol,
                    "signal_id": signal.signal_id,
                    "order_id": str(order.id),
                    "client_order_id": client_order_id,
                },
                now,
            )
            return False
        return True

    def _after_entry(self, order: Any, limit: Decimal, reference: Decimal, now: datetime) -> str:
        """Waits a little for the fill, so that the stop goes on at once, and journals the fill.
        Returns the order's status when the wait ended."""
        fresh = order
        for _ in range(FILL_POLLS):
            self._sleep(FILL_POLL_SECONDS)
            fresh = self.broker.get_order_by_id(str(order.id))
            if _text(fresh.status) in ("filled", "canceled", "expired", "rejected"):
                break
        filled = _decimal(getattr(fresh, "filled_qty", None) or "0", "filled quantity")
        if filled > 0:
            average = _decimal(fresh.filled_avg_price, "fill price")
            self._journal(
                "fractional_fill",
                {
                    "symbol": order.symbol,
                    "order_id": str(order.id),
                    "status": _text(fresh.status),
                    "filled_qty": str(filled),
                    "filled_avg_price": str(average),
                    "limit_price": str(limit),
                    "reference_close": str(reference),
                    "versus_limit_pct": str((average / limit - 1) * 100),
                    "versus_reference_pct": str((average / reference - 1) * 100),
                },
                now,
            )
            filled_at = getattr(fresh, "filled_at", None)
            if filled_at is not None:
                moment = filled_at if filled_at.tzinfo else filled_at.replace(tzinfo=timezone.utc)
                self._unprotected_since.setdefault(order.symbol, min(moment, now))
            self._manage_positions(self._clock())
        return _text(fresh.status)

    # ------------------------------------------------------------------ housekeeping

    def revisit_held(self, now: datetime) -> int:
        """Nothing is held back for a second look (no Oracle here): the Runner's call is a no-op."""
        return 0

    def run_housekeeping(self, now: datetime) -> None:
        steps: tuple[tuple[str, Callable[[], None]], ...] = (
            ("cancel stale entries", lambda: self._cancel_stale_entries(now)),
            ("manage positions", lambda: self._manage_positions(now)),
        )
        for name, step in steps:
            try:
                step()
            except Exception:
                logger.exception(f"Fractional housekeeping step failed: {name}")

    def _cancel_stale_entries(self, now: datetime) -> None:
        for order in self.broker.get_open_orders():
            if _text(order.side) != "buy":
                continue
            placed = _when(order)
            if placed is None or now - placed < ENTRY_MAX_AGE:
                continue
            try:
                self.broker.cancel_order_by_id(str(order.id))
            except Exception as e:
                logger.warning(f"Could not cancel the stale entry {order.symbol}: {e}")
                continue
            self._journal(
                "fractional_entry_cancelled",
                {
                    "symbol": order.symbol,
                    "order_id": str(order.id),
                    "age_minutes": (now - placed).total_seconds() / 60,
                },
                now,
            )

    def _manage_positions(self, now: datetime) -> None:
        """Every position: sold if it is at its target, otherwise given the stop it lacks."""
        market_open = bool(self.broker.get_clock().is_open)
        positions = list(self.broker.get_positions())
        orders = list(self.broker.get_open_orders())
        cover = stop_cover(orders)
        stop_pct = Decimal(str(self.settings.stop_loss_pct))
        target_pct = Decimal(str(self.settings.take_profit_pct))
        held: set[str] = set()
        for position in positions:
            symbol = position.symbol
            held.add(symbol)
            quantity = _decimal(position.qty, f"the quantity of {symbol}")
            if quantity <= 0:
                logger.warning(f"{symbol} is not a long position ({quantity}); left alone.")
                continue
            average = _decimal(position.avg_entry_price, f"the entry price of {symbol}")
            price = _decimal(position.current_price, f"the price of {symbol}")
            if market_open and price >= target_price_for(average, target_pct):
                self._market_sell(
                    symbol,
                    quantity,
                    "fractional_target_sold",
                    {"average_entry": str(average), "price": str(price)},
                    now,
                )
                continue
            missing = quantity - cover.get(symbol, Decimal(0))
            if missing <= QUANTITY_TOLERANCE:
                self._unprotected_since.pop(symbol, None)
                continue
            self._protect(symbol, quantity, average, price, missing, stop_pct, market_open, now)
        for symbol in list(self._unprotected_since):
            if symbol not in held:
                del self._unprotected_since[symbol]

    def _protect(
        self,
        symbol: str,
        quantity: Decimal,
        average: Decimal,
        price: Decimal,
        missing: Decimal,
        stop_pct: Decimal,
        market_open: bool,
        now: datetime,
    ) -> None:
        since = self._unprotected_since.get(symbol)
        if market_open and since is None:
            since = now
            self._unprotected_since[symbol] = now
            self._journal(
                "fractional_unprotected",
                {"symbol": symbol, "quantity": str(quantity), "missing": str(missing)},
                now,
            )
            logger.warning(f"{symbol} has no standing stop for {missing} of {quantity}.")
        if (
            market_open
            and since is not None
            and now - since > UNPROTECTED_HALT_AFTER
            and not self.halted()
        ):
            self.halt(f"{symbol} has had no stop for more than 5 minutes with the market open", now)
        if now < self._retry_after.get(symbol, EPOCH):
            return
        try:
            stop = stop_price_for(average, stop_pct)
        except Refused as e:
            self._journal("fractional_stop_refused", {"symbol": symbol, "message": str(e)}, now)
            self._retry_after[symbol] = now + REFUSED_RETRY
            return
        if market_open and price <= stop:
            # The price is already through the stop: Alpaca would refuse it, and it is what the
            # stop would have done anyway.
            self._market_sell(
                symbol,
                quantity,
                "fractional_stop_sold",
                {"average_entry": str(average), "price": str(price), "stop": str(stop)},
                now,
            )
            return
        try:
            self.broker.submit_order(
                StopOrderRequest(
                    symbol=symbol,
                    qty=float(missing),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    stop_price=float(stop),
                )
            )
        except Exception as e:
            status = getattr(e, "status_code", None)
            transient = (
                not isinstance(e, APIError) or status is None or status == 429 or status >= 500
            )
            self._retry_after[symbol] = now + (TRANSIENT_RETRY if transient else REFUSED_RETRY)
            self._journal(
                "fractional_stop_refused",
                {
                    "symbol": symbol,
                    "quantity": str(missing),
                    "stop": str(stop),
                    "message": str(e),
                    "market_open": market_open,
                },
                now,
            )
            logger.error(f"Could not place the stop for {symbol}: {e}")
            self._alert(
                f"stop:{symbol}",
                "Fractional stop could not be placed",
                f"{symbol}: {e} (market {'open' if market_open else 'closed'})",
                now,
            )
            return
        self._journal(
            "fractional_stop_placed",
            {
                "symbol": symbol,
                "quantity": str(missing),
                "stop_price": str(stop),
                "market_open": market_open,
                "gap_seconds": (now - since).total_seconds() if since is not None else None,
                "covered_before": str(quantity - missing),
            },
            now,
        )
        logger.info(f"Stop for {symbol}: {missing} at {stop}.")
        self._unprotected_since.pop(symbol, None)

    def _market_sell(
        self, symbol: str, quantity: Decimal, event: str, details: dict[str, Any], now: datetime
    ) -> None:
        """Cancels the standing stops of the symbol, waits until they let go of the quantity, and
        sells all of it at market. A stop that has not let go by the end of the wait leaves the
        position as it is: the next tick tries again."""
        for stop in live_stops_of(self.broker.get_open_orders(), symbol):
            try:
                self.broker.cancel_order_by_id(str(stop.id))
            except Exception as e:
                logger.warning(f"Could not cancel the stop of {symbol}: {e}")
        for _ in range(CANCEL_POLLS):
            holding = [
                o
                for o in self.broker.get_open_orders()
                if o.symbol == symbol and _text(o.side) == "sell"
            ]
            if not holding:
                break
            self._sleep(CANCEL_POLL_SECONDS)
        else:
            self._journal("fractional_sale_waiting", {"symbol": symbol, "event": event}, now)
            logger.warning(f"The stop of {symbol} has not been cancelled yet; selling next tick.")
            return
        try:
            order = self.broker.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=float(quantity),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
        except Exception as e:
            self._journal(
                "fractional_sale_refused",
                {"symbol": symbol, "event": event, "message": str(e)},
                now,
            )
            logger.error(f"Could not sell {symbol}: {e}")
            self._alert(f"sale:{symbol}", "Fractional sale refused", f"{symbol}: {e}", now)
            return
        self._journal(
            event,
            {"symbol": symbol, "quantity": str(quantity), "order_id": str(order.id), **details},
            now,
        )
        logger.info(f"Sold {symbol} ({event}): {quantity}.")
        self._unprotected_since.pop(symbol, None)
