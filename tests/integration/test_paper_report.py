"""`paper-report` (research/paper_report.py): the trades are rebuilt from the account's fills, the entry
facts from our own journal, and the numbers about the result stay shut below 300 closed trades."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trading_system.domain.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    SignalDirection,
)
from trading_system.research.anatomy import STOP, TARGET, format_report
from trading_system.research.context import SIGNAL_CONTEXT
from trading_system.research.paper_report import gather
from trading_system.storage.models import (
    BrokerOrderModel,
    OrderIntentModel,
    StrategySignalModel,
)
from trading_system.storage.repositories import AuditRepository

NOW = datetime(2026, 11, 24, 21, 0, tzinfo=timezone.utc)
BAR = datetime(2026, 11, 20, 16, 45, tzinfo=timezone.utc)


class Broker:
    def __init__(self, fills: list[dict[str, Any]], types: dict[str, str], open_positions: int = 0):
        self.fills, self.types, self.open_positions = fills, types, open_positions
        self.fills_asked: list[datetime] = []

    def get_fills(self, after: datetime) -> list[dict[str, Any]]:
        self.fills_asked.append(after)
        return list(self.fills)

    def get_order_by_id(self, order_id: str) -> Any:
        return SimpleNamespace(type=self.types[order_id])

    def get_positions(self) -> list[Any]:
        return [object()] * self.open_positions


class NoNetwork:
    def fetch_bars_bulk(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the daily bars are not fetched below the number of trades")


def fill(side: str, price: float, qty: float, when: str, order: str) -> dict[str, Any]:
    return {
        "type": "fill",
        "symbol": "AXTI",
        "side": side,
        "price": str(price),
        "qty": str(qty),
        "transaction_time": when,
        "order_id": order,
    }


@pytest.fixture
def journal(session_factory: Any) -> AuditRepository:
    signal_id = "sig1"
    with session_factory.begin() as session:
        session.add(
            StrategySignalModel(
                signal_id=signal_id,
                version="1.0.0",
                symbol="AXTI",
                timeframe="15m",
                closed_bar_timestamp=BAR,
                direction=SignalDirection.LONG,
                confidence=1.0,
            )
        )
        session.flush()
        session.add(
            OrderIntentModel(
                signal_id=signal_id,
                symbol="AXTI",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                quantity=10,
                limit_price=100.1,
                stop_price=96.0,
                take_profit_price=105.0,
                intent_timestamp=BAR,
            )
        )
        session.add(
            BrokerOrderModel(
                client_order_id=f"exec_{signal_id}",
                broker_id="entry-1",
                symbol="AXTI",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                status=OrderStatus.FILLED,
                quantity=10,
                filled_quantity=10,
                limit_price=100.1,
                stop_price=None,
                created_at=BAR,
                updated_at=BAR,
            )
        )
    audit = AuditRepository(session_factory)
    audit.record(
        SIGNAL_CONTEXT,
        {
            "symbol": "AXTI",
            "closed_bar": BAR.isoformat(),
            "source": "paper3",
            "group": "risky",
            "atr_pct": 2.0,
            "close": 100.0,
        },
        when=BAR,
    )
    return audit


def test_a_trade_is_rebuilt_from_the_fills_and_our_own_journal(
    session_factory: Any, journal: AuditRepository
) -> None:
    broker = Broker(
        [
            fill("buy", 100.0, 10, "2026-11-20T17:00:00Z", "entry-1"),
            fill("sell", 104.0, 10, "2026-11-23T15:00:00Z", "leg-target"),
        ],
        {"leg-target": "limit"},
        open_positions=2,
    )

    report = gather(broker, NoNetwork(), session_factory, journal, NOW)

    (trade,) = report.trades
    assert trade.symbol == "AXTI" and trade.exit_reason == TARGET
    assert trade.limit_price == 100.1 and trade.stop_price == 96.0
    assert trade.group == "risky" and trade.signal_close == 100.0
    assert report.entries_filled == 1 and report.open_trades == 2
    assert broker.fills_asked[0].date().isoformat() == "2026-11-20"


def test_a_trade_with_no_entry_in_our_journal_is_still_a_trade_without_facts(
    session_factory: Any, journal: AuditRepository
) -> None:
    broker = Broker(
        [
            fill("buy", 50.0, 1, "2026-11-20T17:00:00Z", "foreign-entry"),
            fill("sell", 48.0, 1, "2026-11-20T18:00:00Z", "leg-stop"),
        ],
        {"leg-stop": "stop"},
    )

    report = gather(broker, NoNetwork(), session_factory, journal, NOW)

    (trade,) = report.trades
    assert trade.exit_reason == STOP and trade.limit_price is None and trade.group == "other"


def test_an_exit_order_that_cannot_be_read_is_other_and_does_not_stop_the_report(
    session_factory: Any, journal: AuditRepository
) -> None:
    broker = Broker(
        [
            fill("buy", 100.0, 10, "2026-11-20T17:00:00Z", "entry-1"),
            fill("sell", 99.0, 10, "2026-11-20T18:00:00Z", "mystery"),
        ],
        {},
    )

    report = gather(broker, NoNetwork(), session_factory, journal, NOW)

    assert report.trades[0].exit_reason == "other"


def test_the_report_below_300_trades_shows_no_result(
    session_factory: Any, journal: AuditRepository
) -> None:
    broker = Broker(
        [
            fill("buy", 100.0, 10, "2026-11-20T17:00:00Z", "entry-1"),
            fill("sell", 104.0, 10, "2026-11-23T15:00:00Z", "leg-target"),
        ],
        {"leg-target": "limit"},
    )

    text = format_report(gather(broker, NoNetwork(), session_factory, journal, NOW))

    assert "closed trades: 1" in text and "not shown until 300 closed trades" in text
    assert "mean R" not in text


def test_from_300_trades_the_daily_bars_are_fetched_for_the_nights(
    session_factory: Any, journal: AuditRepository
) -> None:
    fills = []
    for i in range(300):  # one trade after another, 15 a day, each closed before the next opens
        day, slot = 1 + i // 15, i % 15
        buy_at = f"2026-11-{day:02d}T14:{slot * 4:02d}:00Z"
        sell_at = f"2026-11-{day:02d}T14:{slot * 4 + 2:02d}:00Z"
        fills.append(fill("buy", 100.0, 1, buy_at, f"e{i}"))
        fills.append(fill("sell", 101.0, 1, sell_at, f"x{i}"))
    asked: list[Any] = []

    class Data:
        def fetch_bars_bulk(
            self, symbols: Any, timeframe: Any, start: Any, end: Any, feed: Any = None
        ) -> Any:
            asked.append((list(symbols), feed))
            index = pd.DatetimeIndex(
                pd.bdate_range("2026-10-26", "2026-11-24") + pd.Timedelta(hours=5)
            ).tz_localize("UTC")
            return {"AXTI": pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)}

    types = {f"x{i}": "limit" for i in range(300)}
    report = gather(
        Broker(fills, types),
        Data(),
        session_factory,
        journal,
        NOW,
        since=datetime(2026, 11, 1, tzinfo=timezone.utc),
    )

    assert len(report.trades) == 300 and asked == [(["AXTI"], "sip")]
    assert report.daily is not None and "AXTI" in report.daily
    assert "no closed trade has its planned stop on record" in format_report(report)
    assert "the five splits" in format_report(report)
