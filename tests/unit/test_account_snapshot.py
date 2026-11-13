"""The account as the risk engine is shown it, built from what Alpaca really returns.

The first signal on live data died on `int(account.daytrade_count)`: the paper account leaves
that field empty, and the hand-made account in the tests had a number in it. These tests use
the real alpaca-py model, filled the way the paper account fills it. Since alpaca-py 0.45 the
day-trade fields are deprecated (Alpaca stopped sending them on 2026-07-06), and the snapshot no
longer reads them; `model_dump()` looks at a field without the deprecation warning."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from alpaca.trading.enums import AccountStatus
from alpaca.trading.models import TradeAccount

from tests.fixtures.fakes import FakeBroker
from trading_system.runtime.cycle import TradingCycle

NOW = datetime(2026, 11, 3, 16, 53, tzinfo=timezone.utc)


def _account(**fields: Any) -> TradeAccount:
    values: dict[str, Any] = {
        "id": uuid.uuid4(),
        "account_number": "PA0000000000",
        "status": AccountStatus.ACTIVE,
        "equity": "102382.1",
        "cash": "102382.1",
        "buying_power": "409528.4",
        "last_equity": "101569.58",
    }
    values.update(fields)
    return TradeAccount(**values)


class _Broker:
    def __init__(self, account: TradeAccount, positions: Optional[list[Any]] = None):
        self._account = account
        self._positions = positions or []

    def get_account(self) -> TradeAccount:
        return self._account

    def get_positions(self) -> list[Any]:
        return self._positions

    def get_open_orders(self) -> list[Any]:
        return []


def _snapshot(broker: Any) -> Any:
    cycle = TradingCycle.__new__(TradingCycle)  # _snapshot reads nothing but the broker
    cycle.broker = broker
    return cycle._snapshot(NOW)


def test_the_paper_account_as_alpaca_returns_it_gives_a_snapshot() -> None:
    account = _account()
    assert account.model_dump()["daytrade_count"] is None  # what the live account does

    snapshot, positions, orders = _snapshot(_Broker(account))

    assert snapshot.equity == Decimal("102382.1") and snapshot.buying_power == Decimal("409528.4")
    assert snapshot.day_trade_count is None  # unknown stays unknown, it is not 0
    assert (positions, orders) == ([], [])


def test_the_deprecated_day_trade_count_is_not_read_even_if_a_number_is_there() -> None:
    snapshot, _, _ = _snapshot(_Broker(_account(daytrade_count=2)))

    assert snapshot.day_trade_count is None  # unknown, never a number from a dead field


def test_the_fake_broker_of_the_integration_tests_has_the_real_shape() -> None:
    account = FakeBroker().get_account()

    assert isinstance(account, TradeAccount) and account.model_dump()["daytrade_count"] is None


@pytest.mark.parametrize(
    "field, message",
    [
        ("equity", "the account's equity"),
        ("cash", "the account's cash"),
        ("buying_power", "the account's buying power"),
    ],
)
def test_an_amount_alpaca_did_not_send_stops_the_snapshot_and_says_which(
    field: str, message: str
) -> None:
    broker = _Broker(_account(**{field: None}))

    with pytest.raises(ValueError, match=f"Alpaca did not return {message}"):
        _snapshot(broker)


def test_a_position_without_a_market_value_stops_the_snapshot_and_names_the_symbol() -> None:
    position = SimpleNamespace(
        symbol="AMZN", qty="10", market_value=None, avg_entry_price="248.0", unrealized_pl="0"
    )

    with pytest.raises(ValueError, match="market value of AMZN"):
        _snapshot(_Broker(_account(), [position]))


def test_a_position_with_all_its_amounts_is_read() -> None:
    position = SimpleNamespace(
        symbol="AMZN", qty="10", market_value="2480.5", avg_entry_price="248.0", unrealized_pl="0.5"
    )

    _, positions, _ = _snapshot(_Broker(_account(), [position]))

    assert [(p.symbol, p.quantity, p.market_value) for p in positions] == [
        ("AMZN", Decimal(10), Decimal("2480.5"))
    ]
