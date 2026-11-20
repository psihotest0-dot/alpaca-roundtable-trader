"""`paper-report`: what the real paper orders of one account have done (plan section 16.2).

The facts come from Alpaca (the account's fills: an independent source that does not depend on our own
record of the bracket's stop and target legs having succeeded) and from our journal (the order intent for
the planned stop, the signal and its `signal_context`). Read-only: it places and changes nothing.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd

from trading_system.research.anatomy import (
    MARKET_TZ,
    MIN_CLOSED_TRADES,
    OTHER,
    STOP,
    TARGET,
    Report,
    pair_round_trips,
    trades_from_trips,
)
from trading_system.research.context import SIGNAL_CONTEXT

logger = logging.getLogger(__name__)

# The clocks of paper3 on the 55-stock portfolio started on this day (docs/paper-third-account.md, 6.2).
SINCE = datetime(2026, 11, 20, tzinfo=timezone.utc)
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def exit_reason_of(order_type: Any) -> str:
    """A sell that was a stop order is a stop, one that was a limit order is the target."""
    kind = str(getattr(order_type, "value", order_type)).lower()
    if kind in ("stop", "stop_limit", "trailing_stop"):
        return STOP
    if kind == "limit":
        return TARGET
    return OTHER


def daily_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """`open` and `close` indexed by the New York date, as the research panel keeps them."""
    day = pd.DatetimeIndex(bars.index).tz_convert(MARKET_TZ).tz_localize(None).normalize()
    return pd.DataFrame(
        {"open": bars["open"].to_numpy(), "close": bars["close"].to_numpy()}, index=day
    )


def entry_lookup(
    session_factory: Any, audit: Any, now: datetime
) -> Callable[[str], dict[str, Any]]:
    """`broker order id -> {limit_price, stop_price, context}` for an entry order, from our own journal."""
    from sqlalchemy import select

    from trading_system.storage.models import (
        BrokerOrderModel,
        OrderIntentModel,
        StrategySignalModel,
    )

    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for _, details in audit.events(SIGNAL_CONTEXT, EPOCH, now + timedelta(days=1)):
        contexts.setdefault((details["symbol"], details["closed_bar"]), details)

    def facts(broker_order_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        with session_factory() as session:
            order = session.scalars(
                select(BrokerOrderModel).where(BrokerOrderModel.broker_id == broker_order_id)
            ).first()
            if order is None or not str(order.client_order_id).startswith("exec_"):
                return out
            signal_id = str(order.client_order_id)[len("exec_") :]
            intent = session.scalars(
                select(OrderIntentModel).where(OrderIntentModel.signal_id == signal_id)
            ).first()
            if intent is not None:
                out["limit_price"] = float(intent.limit_price) if intent.limit_price else None
                out["stop_price"] = float(intent.stop_price) if intent.stop_price else None
            signal = session.get(StrategySignalModel, signal_id)
            if signal is not None:
                bar = pd.Timestamp(signal.closed_bar_timestamp)
                bar = bar.tz_localize("UTC") if bar.tzinfo is None else bar.tz_convert("UTC")
                context = contexts.get((str(signal.symbol), bar.isoformat()))
                if context is not None:
                    out["context"] = context
        return out

    return facts


def gather(
    broker: Any,
    data: Any,
    session_factory: Any,
    audit: Any,
    now: datetime,
    since: datetime = SINCE,
) -> Report:
    """Builds the report for the fills since `since`. The daily bars for the nights are fetched only
    from the number of closed trades at which the result is shown."""
    fills = broker.get_fills(since)
    trips = pair_round_trips(fills)
    order_types: dict[str, str] = {}

    def exit_reason(order_id: str) -> str:
        if order_id not in order_types:
            try:
                order_types[order_id] = exit_reason_of(broker.get_order_by_id(order_id).type)
            except Exception as e:  # an order that cannot be read is simply "other"
                logger.warning(f"Could not read the exit order {order_id}: {e}")
                order_types[order_id] = OTHER
        return order_types[order_id]

    trades = trades_from_trips(trips, exit_reason, entry_lookup(session_factory, audit, now))
    entries = {str(f["order_id"]) for f in fills if str(f.get("side", "")).lower() == "buy"}
    report = Report(
        trades=trades,
        open_trades=len(list(broker.get_positions())),
        entries_filled=len(entries),
    )
    if len(trades) >= MIN_CLOSED_TRADES:
        from alpaca.data.timeframe import TimeFrame

        start = min(t.entry_time for t in trades) - timedelta(days=5)
        symbols = sorted({t.symbol for t in trades})
        frames: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), 100):
            frames.update(
                data.fetch_bars_bulk(symbols[i : i + 100], TimeFrame.Day, start, now, feed="sip")
            )
        report.daily = {symbol: daily_frame(bars) for symbol, bars in frames.items()}
    return report


def run(settings: Optional[Any] = None) -> str:
    """The report for the account of this process's keys, as text."""
    from trading_system.broker.alpaca_broker import AlpacaBroker
    from trading_system.config import Settings
    from trading_system.data.alpaca_data import AlpacaData
    from trading_system.research.anatomy import format_report
    from trading_system.storage.database import get_engine, get_session_factory
    from trading_system.storage.repositories import AuditRepository

    settings = settings or Settings()
    factory = get_session_factory(get_engine(settings.database_url))
    now = datetime.now(timezone.utc)
    report = gather(
        AlpacaBroker(settings), AlpacaData(settings), factory, AuditRepository(factory), now
    )
    return format_report(report)
