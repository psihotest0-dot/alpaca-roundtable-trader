import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from decimal import Decimal
import datetime
import uuid

import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.models import Order, Asset
from alpaca.trading.enums import (
    TimeInForce,
    OrderSide as AlpacaOrderSide,
    OrderStatus as AlpacaOrderStatus,
    OrderType as AlpacaOrderType,
)

from trading_system.execution.engine import ExecutionEngine, find_unprotected_positions
from trading_system.broker.alpaca_broker import AlpacaBroker
from trading_system.storage.repositories import OrderRepository, StateRepository
from trading_system.domain.models import OrderIntent, TradingState
from trading_system.domain.enums import OrderSide, OrderType, OrderStatus
from trading_system.config import Settings


@pytest.fixture
def mock_broker():
    return MagicMock(spec=AlpacaBroker)


@pytest.fixture
def mock_order_repo():
    return MagicMock(spec=OrderRepository)


@pytest.fixture
def mock_state_repo():
    repo = MagicMock(spec=StateRepository)
    repo.get_state.return_value = TradingState()
    return repo


@pytest.fixture
def settings():
    # Use dummy env vars to satisfy pydantic
    import os

    os.environ["ALPACA_API_KEY"] = "test"
    os.environ["ALPACA_SECRET_KEY"] = "test"
    s = Settings()
    s.is_paper = True
    return s


@pytest.fixture
def engine(mock_broker, mock_order_repo, mock_state_repo, settings):
    return ExecutionEngine(mock_broker, mock_order_repo, mock_state_repo, settings)


@pytest.fixture
def base_intent():
    return OrderIntent(
        signal_id="test_signal_1",
        symbol="AAPL",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Decimal("10"),
        limit_price=Decimal("150.00"),
        stop_price=Decimal("140.00"),
        take_profit_price=Decimal("157.50"),
        intent_timestamp=datetime.datetime.now(datetime.timezone.utc),
    )


@pytest.fixture
def mock_asset():
    return MagicMock(
        spec=Asset,
        id="a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",  # valid uuid format
        class_="us_equity",
        exchange="NASDAQ",
        symbol="AAPL",
        name="Apple Inc.",
        status="active",
        tradable=True,
        marginable=True,
        shortable=True,
        easy_to_borrow=True,
        fractionable=True,
        price_increment=0.01,
        min_order_size=None,
        min_trade_increment=None,
        maintenance_margin_requirement=None,
        attributes=[],
    )


def test_successful_bracket_submission(
    engine, mock_broker, mock_order_repo, base_intent, mock_asset
):
    mock_broker.get_asset.return_value = mock_asset

    mock_alpaca_order = MagicMock(spec=Order)
    mock_alpaca_order.id = "test-order-id"
    mock_alpaca_order.symbol = "AAPL"

    mock_broker.submit_order.return_value = mock_alpaca_order

    result = engine.submit_order(base_intent)

    assert result is not None
    assert result.broker_id == "test-order-id"
    assert result.status == OrderStatus.SUBMITTED
    mock_order_repo.save_intent.assert_called_once_with(base_intent)
    mock_order_repo.save_broker_order.assert_called()


def test_duplicate_submission_handling(
    engine, mock_broker, mock_order_repo, base_intent, mock_asset
):
    # Simulate first submission
    mock_broker.get_asset.return_value = mock_asset
    mock_alpaca_order = MagicMock(spec=Order)
    mock_alpaca_order.id = "test-order-id"
    mock_alpaca_order.symbol = "AAPL"
    mock_broker.submit_order.return_value = mock_alpaca_order

    engine.submit_order(base_intent)

    # Simulate duplicate intent persist failure
    mock_order_repo.save_intent.side_effect = Exception("Integrity Error")

    result2 = engine.submit_order(base_intent)
    assert result2 is None
    # submit_order should not be called again
    assert mock_broker.submit_order.call_count == 1


def test_timeout_reconciliation(engine, mock_broker, mock_order_repo, base_intent, mock_asset):
    mock_broker.get_asset.return_value = mock_asset

    # Simulate network timeout during submission
    mock_broker.submit_order.side_effect = Exception("Timeout")

    # But order actually went through
    mock_reconciled_order = MagicMock(spec=Order)
    mock_reconciled_order.id = "recovered-id"
    mock_reconciled_order.symbol = "AAPL"
    mock_reconciled_order.side = "buy"
    mock_reconciled_order.type = "limit"
    mock_reconciled_order.status = AlpacaOrderStatus.NEW
    mock_reconciled_order.qty = "10"
    mock_reconciled_order.filled_qty = "0"
    mock_reconciled_order.limit_price = "150.00"
    mock_reconciled_order.stop_price = "140.00"
    mock_reconciled_order.created_at = datetime.datetime.now(datetime.timezone.utc)
    mock_reconciled_order.updated_at = datetime.datetime.now(datetime.timezone.utc)

    mock_broker.get_order_by_client_id.return_value = mock_reconciled_order

    result = engine.submit_order(base_intent)

    # It should wait, check broker, and return the recovered order
    assert result is not None
    assert result.broker_id == "recovered-id"
    assert result.status == OrderStatus.SUBMITTED


def _api_error(status: int) -> APIError:
    response = requests.Response()
    response.status_code = status
    return APIError('{"code": 1, "message": "x"}', requests.HTTPError(response=response))


def _saved_statuses(mock_order_repo) -> list[OrderStatus]:
    return [c.args[0].status for c in mock_order_repo.save_broker_order.call_args_list]


@pytest.mark.parametrize("status", [400, 403, 422])
def test_an_order_the_broker_refused_is_rejected_and_does_not_halt(
    engine, mock_broker, mock_order_repo, mock_state_repo, base_intent, mock_asset, mocker, status
):
    """A 4xx is a refusal of the order itself: nothing stands at the broker, so the signal is not
    traded and the record says REJECTED; there is no reason to stop the whole account."""
    mocker.patch("trading_system.execution.engine.time.sleep")
    mock_broker.get_asset.return_value = mock_asset
    mock_broker.submit_order.side_effect = _api_error(status)
    mock_broker.get_order_by_client_id.side_effect = _api_error(404)

    assert engine.submit_order(base_intent) is None

    mock_state_repo.save_state.assert_not_called()
    assert _saved_statuses(mock_order_repo) == [OrderStatus.SUBMITTING, OrderStatus.REJECTED]
    assert mock_broker.submit_order.call_count == 1  # never resubmitted


@pytest.mark.parametrize(
    "submit_error, lookup_error",
    [
        (Exception("Timeout"), _api_error(404)),  # perhaps sent: unknown
        (_api_error(500), _api_error(404)),  # a server error says nothing about the order
        (_api_error(429), _api_error(404)),  # a rate limit is not a refusal of the order
        (_api_error(422), _api_error(503)),  # refused, but the lookup could not confirm it
    ],
)
def test_a_submission_whose_outcome_is_unknown_still_halts(
    engine,
    mock_broker,
    mock_order_repo,
    mock_state_repo,
    base_intent,
    mock_asset,
    mocker,
    submit_error,
    lookup_error,
):
    mocker.patch("trading_system.execution.engine.time.sleep")
    mock_broker.get_asset.return_value = mock_asset
    mock_broker.submit_order.side_effect = submit_error
    mock_broker.get_order_by_client_id.side_effect = lookup_error

    assert engine.submit_order(base_intent) is None

    assert mock_state_repo.save_state.call_args[0][0].is_trading_halted is True
    assert OrderStatus.REJECTED not in _saved_statuses(mock_order_repo)


def test_a_refused_order_that_the_broker_has_anyway_is_returned(
    engine, mock_broker, mock_order_repo, mock_state_repo, base_intent, mock_asset, mocker
):
    """A 422 for a client_order_id already used means the order stands: it is ours, not refused."""
    mocker.patch("trading_system.execution.engine.time.sleep")
    mock_broker.get_asset.return_value = mock_asset
    mock_broker.submit_order.side_effect = _api_error(422)
    existing = MagicMock(spec=Order)
    existing.id = "existing-id"
    existing.symbol = "AAPL"
    existing.side = "buy"
    existing.type = "limit"
    existing.status = AlpacaOrderStatus.NEW
    existing.qty = "10"
    existing.filled_qty = "0"
    existing.limit_price = "150.00"
    existing.stop_price = "140.00"
    existing.created_at = datetime.datetime.now(datetime.timezone.utc)
    existing.updated_at = datetime.datetime.now(datetime.timezone.utc)
    mock_broker.get_order_by_client_id.return_value = existing

    result = engine.submit_order(base_intent)

    assert result is not None and result.broker_id == "existing-id"
    mock_state_repo.save_state.assert_not_called()


def test_partial_fill_tracking(engine, mock_broker, mock_order_repo, base_intent):
    # Not testing event websocket stream here, but we can test manual reconciliation mapping
    mock_alpaca_order = MagicMock(spec=Order)
    mock_alpaca_order.id = "partially-filled-id"
    mock_alpaca_order.symbol = "AAPL"
    mock_alpaca_order.side = "buy"
    mock_alpaca_order.type = "limit"
    mock_alpaca_order.status = AlpacaOrderStatus.PARTIALLY_FILLED
    mock_alpaca_order.qty = "10"
    mock_alpaca_order.filled_qty = "4"
    mock_alpaca_order.limit_price = "150.00"
    mock_alpaca_order.stop_price = "140.00"
    mock_alpaca_order.created_at = datetime.datetime.now(datetime.timezone.utc)
    mock_alpaca_order.updated_at = datetime.datetime.now(datetime.timezone.utc)

    mock_broker.get_order_by_client_id.return_value = mock_alpaca_order

    result = engine.store_broker_order(mock_alpaca_order, "exec_test_signal_1")
    assert result is not None
    assert result.status == OrderStatus.PARTIALLY_FILLED
    assert result.filled_quantity == Decimal("4")


def test_rejected_invalid_price_increments(
    engine, mock_broker, mock_order_repo, base_intent, mock_asset
):
    mock_broker.get_asset.return_value = mock_asset

    # Invalid limit price (not $0.01 increment)
    base_intent.limit_price = Decimal("150.005")

    result = engine.submit_order(base_intent)

    assert result is None
    mock_broker.submit_order.assert_not_called()


def test_bracket_is_sent_with_the_prices_of_the_intent(
    engine, mock_broker, base_intent, mock_asset
):
    """C3: execution no longer derives the target from TAKE_PROFIT_PCT; the intent decides."""
    mock_broker.get_asset.return_value = mock_asset
    mock_broker.submit_order.return_value = MagicMock(spec=Order, id="oid", symbol="AAPL")
    base_intent.take_profit_price = Decimal("163.00")  # not 5 % above the limit

    engine.submit_order(base_intent)

    (request,) = mock_broker.submit_order.call_args.args
    assert request.limit_price == 150.00
    # The legs inherit the entry's time in force; DAY would let the stop expire at the close.
    assert request.time_in_force == TimeInForce.GTC
    assert request.stop_loss.stop_price == 140.00
    assert request.take_profit.limit_price == 163.00


@pytest.mark.parametrize(
    "stop, limit, take_profit",
    [
        (None, "150.00", "157.50"),
        ("140.00", "150.00", None),
        ("150.00", "150.00", "157.50"),  # no room for the stop
        ("140.00", "150.00", "150.00"),  # no room for the target
        ("151.00", "150.00", "157.50"),  # stop above the entry
        ("140.00", "150.00", "149.00"),  # target below the entry
    ],
)
def test_incomplete_or_disordered_bracket_is_rejected(
    engine, mock_broker, base_intent, mock_asset, stop, limit, take_profit
):
    mock_broker.get_asset.return_value = mock_asset
    base_intent.stop_price = None if stop is None else Decimal(stop)
    base_intent.limit_price = Decimal(limit)
    base_intent.take_profit_price = None if take_profit is None else Decimal(take_profit)

    assert engine.submit_order(base_intent) is None
    mock_broker.submit_order.assert_not_called()


def test_take_profit_off_the_tick_is_rejected(engine, mock_broker, base_intent, mock_asset):
    mock_broker.get_asset.return_value = mock_asset
    base_intent.take_profit_price = Decimal("157.505")

    assert engine.submit_order(base_intent) is None
    mock_broker.submit_order.assert_not_called()


def test_missing_child_orders_emergency(engine, mock_state_repo):
    engine.halt_trading("Missing child orders")

    mock_state_repo.save_state.assert_called_once()
    saved_state = mock_state_repo.save_state.call_args[0][0]
    assert saved_state.is_trading_halted is True
    assert saved_state.halt_reason == "Missing child orders"


def test_cancel_fill_race_condition(engine, mock_broker):
    mock_alpaca_order = MagicMock(spec=Order)
    mock_alpaca_order.id = "filled-id"
    mock_alpaca_order.status = AlpacaOrderStatus.FILLED

    mock_broker.get_order_by_client_id.return_value = mock_alpaca_order

    engine.cancel_bracket_order("exec_test_signal_1")

    # Ensure cancel was NOT called since it's already filled
    mock_broker.cancel_order_by_id.assert_not_called()


def test_crash_after_submission_before_db_update(
    engine, mock_broker, mock_order_repo, base_intent, mock_asset
):
    mock_broker.get_asset.return_value = mock_asset

    mock_alpaca_order = MagicMock(spec=Order)
    mock_alpaca_order.id = "test-order-id"
    mock_alpaca_order.symbol = "AAPL"

    mock_broker.submit_order.return_value = mock_alpaca_order

    # Simulate crash saving to DB
    mock_order_repo.save_broker_order.side_effect = Exception("DB Crash")

    # But order exists at broker on reconciliation fallback
    mock_reconciled_order = MagicMock(spec=Order)
    mock_reconciled_order.id = "test-order-id"
    mock_reconciled_order.symbol = "AAPL"
    mock_reconciled_order.side = "buy"
    mock_reconciled_order.type = "limit"
    mock_reconciled_order.status = (
        "unknown_status"  # Alpaca API might return some unsupported string
    )
    mock_reconciled_order.qty = "10"
    mock_reconciled_order.filled_qty = "0"
    mock_reconciled_order.limit_price = "150.00"
    mock_reconciled_order.stop_price = "140.00"
    mock_reconciled_order.created_at = datetime.datetime.now(datetime.timezone.utc)
    mock_reconciled_order.updated_at = datetime.datetime.now(datetime.timezone.utc)

    mock_broker.get_order_by_client_id.return_value = mock_reconciled_order

    # Fix the side effect for the reconciliation call so it succeeds
    def save_broker_order_side_effect(order):
        if order.status == OrderStatus.SUBMITTING:
            return
        if order.status == OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION:
            return
        raise Exception("DB Crash")

    mock_order_repo.save_broker_order.side_effect = save_broker_order_side_effect

    result = engine.submit_order(base_intent)

    assert result is not None
    assert result.status == OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION


def test_dry_run_prevents_mutations(
    mock_broker, mock_order_repo, mock_state_repo, settings, base_intent
):
    # Initialize engine in dry-run mode
    engine = ExecutionEngine(
        mock_broker, mock_order_repo, mock_state_repo, settings, is_dry_run=True
    )
    mock_broker.get_asset.return_value = MagicMock(price_increment=0.01)

    # 1. Test submit_order
    result = engine.submit_order(base_intent)
    assert result is not None
    assert result.status == OrderStatus.SUBMITTED
    assert result.broker_id == f"dry_run_exec_{base_intent.signal_id}"
    mock_broker.submit_order.assert_not_called()

    # 2. Test cancel_bracket_order
    engine.cancel_bracket_order(result.client_order_id)
    mock_broker.cancel_order_by_id.assert_not_called()

    # 3. Test emergency_check_unprotected_positions
    mock_pos = MagicMock()
    mock_pos.symbol = "AAPL"
    mock_pos.qty = "10"
    mock_broker.get_positions.return_value = [mock_pos]
    mock_broker.get_open_orders.return_value = []  # No covering orders

    engine.emergency_check_unprotected_positions()
    mock_broker.submit_order.assert_not_called()


# --- C7: the unprotected-position check -------------------------------------------------


def _order(symbol, side, order_type, status, qty, *, id=None, legs=None):
    """An order as the check reads it, with the real alpaca enums."""
    return SimpleNamespace(
        id=id or f"{symbol}-{side.value}-{order_type.value}-{status.value}-{qty}",
        symbol=symbol,
        side=side,
        type=order_type,
        status=status,
        qty=str(qty),
        legs=legs,
    )


def _stop(symbol, qty, *, side=AlpacaOrderSide.SELL, status=AlpacaOrderStatus.NEW, **kwargs):
    return _order(symbol, side, AlpacaOrderType.STOP, status, qty, **kwargs)


def _position(symbol, qty):
    return SimpleNamespace(symbol=symbol, qty=str(qty))


def test_a_long_position_with_an_active_sell_stop_is_protected():
    assert find_unprotected_positions([_position("AAPL", 10)], [_stop("AAPL", 10)]) == []


def test_a_position_with_no_orders_at_all_is_unprotected():
    (gap,) = find_unprotected_positions([_position("AAPL", "91.842")], [])

    assert (gap.symbol, gap.quantity, gap.covered) == ("AAPL", Decimal("91.842"), Decimal(0))


def test_a_stop_that_covers_only_part_of_the_position_leaves_a_gap():
    (gap,) = find_unprotected_positions([_position("AAPL", 10)], [_stop("AAPL", 4)])

    assert gap.covered == Decimal(4)


def test_stops_of_another_symbol_or_the_wrong_side_do_not_protect():
    orders = [_stop("MSFT", 10), _stop("AAPL", 10, side=AlpacaOrderSide.BUY)]

    assert [g.symbol for g in find_unprotected_positions([_position("AAPL", 10)], orders)] == [
        "AAPL"
    ]


def test_a_take_profit_limit_is_not_a_stop():
    limit = _order("AAPL", AlpacaOrderSide.SELL, AlpacaOrderType.LIMIT, AlpacaOrderStatus.NEW, 10)

    assert find_unprotected_positions([_position("AAPL", 10)], [limit])


def test_a_held_stop_protects_nothing_yet():
    """The legs of a bracket stay held until the entry is completely filled."""
    held = _stop("AAPL", 10, status=AlpacaOrderStatus.HELD)

    assert find_unprotected_positions([_position("AAPL", 10)], [held])


def test_a_stop_leg_nested_under_its_parent_is_found():
    """get_open_orders asks for nested=True, so the stop of a bracket is not a top-level order."""
    parent = _order(
        "AAPL",
        AlpacaOrderSide.BUY,
        AlpacaOrderType.LIMIT,
        AlpacaOrderStatus.PARTIALLY_FILLED,
        10,
        legs=[_stop("AAPL", 10)],
    )

    assert find_unprotected_positions([_position("AAPL", 10)], [parent]) == []


def test_a_leg_listed_both_nested_and_on_its_own_is_counted_once():
    leg = _stop("AAPL", 10, id="same-leg")
    parent = _order(
        "AAPL", AlpacaOrderSide.BUY, AlpacaOrderType.LIMIT, AlpacaOrderStatus.NEW, 10, legs=[leg]
    )

    (gap,) = find_unprotected_positions([_position("AAPL", 20)], [parent, leg])

    assert gap.covered == Decimal(10)


def test_a_short_position_is_protected_by_a_buy_stop_and_unprotected_by_a_sell_stop():
    short = _position("COIN", -88)

    assert find_unprotected_positions([short], [_stop("COIN", 88, side=AlpacaOrderSide.BUY)]) == []
    (gap,) = find_unprotected_positions([short], [_stop("COIN", 88)])
    assert gap.quantity == Decimal(-88)


def _account_with_unprotected_positions(mock_broker):
    mock_broker.get_positions.return_value = [_position("AAPL", "91.842"), _position("COIN", -88)]
    mock_broker.get_open_orders.return_value = []


def test_unprotected_positions_halt_trading_once_and_are_left_alone_by_default(
    engine, mock_broker, mock_state_repo
):
    """The paper account holds positions from other tools. Raising the alarm is safe;
    selling someone's positions on a guess is not."""
    _account_with_unprotected_positions(mock_broker)

    engine.emergency_check_unprotected_positions()

    mock_state_repo.save_state.assert_called_once()
    saved = mock_state_repo.save_state.call_args[0][0]
    assert saved.is_trading_halted is True
    assert "AAPL" in saved.halt_reason and "COIN" in saved.halt_reason
    mock_broker.submit_order.assert_not_called()


def test_emergency_close_is_sent_only_when_it_is_switched_on(
    engine, mock_broker, settings, mock_state_repo
):
    _account_with_unprotected_positions(mock_broker)
    settings.emergency_close_unprotected = True

    engine.emergency_check_unprotected_positions()

    sent = {r.symbol: r for r in (c.args[0] for c in mock_broker.submit_order.call_args_list)}
    assert (sent["AAPL"].side, sent["AAPL"].qty) == (AlpacaOrderSide.SELL, 91.842)
    assert (sent["COIN"].side, sent["COIN"].qty) == (AlpacaOrderSide.BUY, 88.0)  # covers the short


def test_emergency_close_in_dry_run_sends_nothing(
    mock_broker, mock_order_repo, mock_state_repo, settings
):
    settings.emergency_close_unprotected = True
    engine = ExecutionEngine(
        mock_broker, mock_order_repo, mock_state_repo, settings, is_dry_run=True
    )
    _account_with_unprotected_positions(mock_broker)

    engine.emergency_check_unprotected_positions()

    mock_state_repo.save_state.assert_called_once()  # still halts
    mock_broker.submit_order.assert_not_called()


def test_protected_positions_do_not_halt_anything(engine, mock_broker, mock_state_repo):
    mock_broker.get_positions.return_value = [_position("AAPL", 10)]
    mock_broker.get_open_orders.return_value = [_stop("AAPL", 10)]

    engine.emergency_check_unprotected_positions()

    mock_state_repo.save_state.assert_not_called()


# --- C7: the shape Alpaca really reports for a filled bracket ---------------------------


def _real_order(order_type, side, status, *, qty="1", legs=None, **prices):
    """alpaca-py's own Order model, filled in like the paper order of 2026-11-02."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return Order(
        id=uuid.uuid4(),
        client_order_id="exec_real",
        created_at=now,
        updated_at=now,
        submitted_at=now,
        asset_id=uuid.uuid4(),
        symbol="INTC",
        asset_class="us_equity",
        order_class="bracket",
        time_in_force="day",
        extended_hours=False,
        order_type=order_type,
        type=order_type,
        side=side,
        status=status,
        qty=qty,
        filled_qty=qty if status == "filled" else "0",
        legs=legs,
        **prices,
    )


def _bracket(entry_status, stop_status, take_profit_status="new"):
    return _real_order(
        "limit",
        "buy",
        entry_status,
        limit_price="64.75",
        legs=[
            _real_order("limit", "sell", take_profit_status, limit_price="67.98"),
            _real_order("stop", "sell", stop_status, stop_price="62.16"),
        ],
    )


def test_a_filled_bracket_is_protected_by_its_held_stop_leg():
    """Seen on a real paper order: once the entry filled, the open orders held only the
    take-profit (`new`); the stop leg stayed `held` and was in no list of open orders."""
    open_orders = [_real_order("limit", "sell", "new", limit_price="67.98")]
    recent = [_bracket("filled", "held")]

    assert find_unprotected_positions([_position("INTC", 1)], open_orders) != []  # blind alone
    assert find_unprotected_positions([_position("INTC", 1)], open_orders, lambda _s: recent) == []


@pytest.mark.parametrize(
    "entry_status, stop_status",
    [
        ("new", "held"),  # the entry has not filled: legs are not active yet
        ("partially_filled", "held"),  # "active only once the first order is completely filled"
        ("filled", "canceled"),  # the stop is gone
        ("filled", "expired"),  # a DAY leg after the close
    ],
)
def test_other_bracket_states_do_not_protect(entry_status, stop_status):
    recent = [_bracket(entry_status, stop_status)]

    (gap,) = find_unprotected_positions([_position("INTC", 1)], [], lambda _s: recent)

    assert gap.covered == Decimal(0)


def test_the_history_is_asked_only_for_positions_the_open_orders_leave_uncovered():
    asked = []

    def history(symbol):
        asked.append(symbol)
        return []

    find_unprotected_positions(
        [_position("AAPL", 10), _position("INTC", 1)], [_stop("AAPL", 10)], history
    )

    assert asked == ["INTC"]


def test_a_failed_history_lookup_still_raises_the_alarm(engine, mock_broker, mock_state_repo):
    mock_broker.get_positions.return_value = [_position("INTC", 1)]
    mock_broker.get_open_orders.return_value = []
    mock_broker.get_recent_orders_for_symbol.side_effect = ConnectionError("down")

    engine.emergency_check_unprotected_positions()

    assert mock_state_repo.save_state.call_args[0][0].is_trading_halted is True


def test_the_engine_does_not_halt_for_a_filled_bracket(engine, mock_broker, mock_state_repo):
    mock_broker.get_positions.return_value = [_position("INTC", 1)]
    mock_broker.get_open_orders.return_value = [_real_order("limit", "sell", "new")]
    mock_broker.get_recent_orders_for_symbol.return_value = [_bracket("filled", "held")]

    engine.emergency_check_unprotected_positions()

    mock_state_repo.save_state.assert_not_called()
    mock_broker.get_recent_orders_for_symbol.assert_called_once_with("INTC")


# --- the protection check, for Grafana ----------------------------------------------------


def test_the_protection_check_publishes_the_count_and_when_it_completed(
    engine, mock_broker, mock_state_repo
):
    from trading_system.observability import metrics

    metrics.PROTECTION_CHECK_TIMESTAMP.set(0)
    mock_broker.get_positions.return_value = []
    mock_broker.get_open_orders.return_value = []

    engine.emergency_check_unprotected_positions()

    assert metrics.UNPROTECTED_POSITIONS._value.get() == 0
    assert metrics.PROTECTION_CHECK_TIMESTAMP._value.get() > 0

    _account_with_unprotected_positions(mock_broker)
    engine.emergency_check_unprotected_positions()

    assert metrics.UNPROTECTED_POSITIONS._value.get() == 2  # AAPL and COIN
    mock_state_repo.save_state.assert_called_once()  # the alarm and the halt are as they were


def test_a_check_that_cannot_read_the_broker_does_not_look_like_a_clean_one(engine, mock_broker):
    """A 0 is only worth something while the timestamp is fresh: a failed check stamps nothing."""
    from trading_system.observability import metrics

    metrics.PROTECTION_CHECK_TIMESTAMP.set(0)
    mock_broker.get_positions.side_effect = ConnectionError("alpaca is down")

    engine.emergency_check_unprotected_positions()  # logs the error, as before

    assert metrics.PROTECTION_CHECK_TIMESTAMP._value.get() == 0


# --- a partially filled entry (2026-11-27, code review) ------------------------------------
# While a bracket entry has filled only in part its legs wait (`held`): the shares bought have no
# stop. The rest of the entry is cancelled at once, Alpaca then works the legs for the filled
# shares, and the protection check waits for them for PARTIAL_FILL_GRACE instead of halting.

NOW = datetime.datetime(2026, 11, 25, 15, 0, tzinfo=datetime.timezone.utc)


def _entry(symbol, status, *, qty=10, filled=0, age_minutes=1, legs=None):
    order = _order(symbol, AlpacaOrderSide.BUY, AlpacaOrderType.LIMIT, status, qty, legs=legs)
    order.client_order_id = f"exec_{symbol}"
    order.created_at = NOW - datetime.timedelta(minutes=age_minutes)
    order.filled_qty = str(filled)
    return order


def _legs(symbol, stop_status, take_profit_status, qty=4):
    return [
        _order(symbol, AlpacaOrderSide.SELL, AlpacaOrderType.LIMIT, take_profit_status, qty),
        _stop(symbol, qty, status=stop_status),
    ]


@pytest.fixture
def clocked(mock_broker, mock_order_repo, mock_state_repo, settings):
    moment = {"now": NOW}
    engine = ExecutionEngine(
        mock_broker, mock_order_repo, mock_state_repo, settings, clock=lambda: moment["now"]
    )
    return engine, moment


def _halted(mock_state_repo) -> bool:
    return any(c.args[0].is_trading_halted for c in mock_state_repo.save_state.call_args_list)


def test_a_partially_filled_entry_is_cancelled_at_once_and_a_young_unfilled_one_is_not(
    clocked, mock_broker
):
    engine, _ = clocked
    mock_broker.get_open_orders.return_value = [
        _entry("AAA", AlpacaOrderStatus.PARTIALLY_FILLED, filled=4),
        _entry("BBB", AlpacaOrderStatus.NEW),
    ]
    mock_broker.get_order_by_client_id.side_effect = lambda cid: SimpleNamespace(
        id=f"id-{cid}", status="partially_filled" if cid == "exec_AAA" else "new"
    )

    engine.check_bounded_entry_expiry()

    assert [c.args[0] for c in mock_broker.cancel_order_by_id.call_args_list] == ["id-exec_AAA"]


def test_shares_of_a_partially_filled_entry_wait_for_their_legs_and_then_halt(
    clocked, mock_broker, mock_state_repo
):
    engine, moment = clocked
    parent = _entry(
        "AAA",
        AlpacaOrderStatus.PARTIALLY_FILLED,
        filled=4,
        legs=_legs("AAA", AlpacaOrderStatus.HELD, AlpacaOrderStatus.HELD),
    )
    mock_broker.get_positions.return_value = [_position("AAA", 4)]
    mock_broker.get_open_orders.return_value = [parent]
    mock_broker.get_recent_orders_for_symbol.return_value = [parent]

    engine.emergency_check_unprotected_positions()
    moment["now"] = NOW + datetime.timedelta(minutes=1)
    engine.emergency_check_unprotected_positions()
    assert not _halted(mock_state_repo)

    moment["now"] = NOW + datetime.timedelta(minutes=3)
    engine.emergency_check_unprotected_positions()
    assert _halted(mock_state_repo)


def test_after_the_cancel_a_gap_waits_for_the_legs_and_then_halts(
    clocked, mock_broker, mock_state_repo
):
    engine, moment = clocked
    mock_broker.get_open_orders.return_value = [
        _entry("AAA", AlpacaOrderStatus.PARTIALLY_FILLED, filled=4)
    ]
    mock_broker.get_order_by_client_id.return_value = SimpleNamespace(
        id="id-exec_AAA", status="partially_filled"
    )
    engine.check_bounded_entry_expiry()
    # cancelled; the legs are not working yet
    cancelled = _entry(
        "AAA",
        AlpacaOrderStatus.CANCELED,
        filled=4,
        legs=_legs("AAA", AlpacaOrderStatus.HELD, AlpacaOrderStatus.HELD),
    )
    mock_broker.get_positions.return_value = [_position("AAA", 4)]
    mock_broker.get_open_orders.return_value = []
    mock_broker.get_recent_orders_for_symbol.return_value = [cancelled]

    engine.emergency_check_unprotected_positions()
    assert not _halted(mock_state_repo)

    moment["now"] = NOW + datetime.timedelta(minutes=3)
    engine.emergency_check_unprotected_positions()
    assert _halted(mock_state_repo)


def test_an_unprotected_position_without_a_partial_entry_still_halts_at_once(
    clocked, mock_broker, mock_state_repo
):
    engine, _ = clocked
    mock_broker.get_positions.return_value = [_position("AAA", 4)]
    mock_broker.get_open_orders.return_value = []
    mock_broker.get_recent_orders_for_symbol.return_value = []

    engine.emergency_check_unprotected_positions()

    assert _halted(mock_state_repo)


@pytest.mark.parametrize(
    "stop_status, take_profit_status",
    [
        (AlpacaOrderStatus.HELD, AlpacaOrderStatus.NEW),  # the OCO pair, as for a filled entry
        (AlpacaOrderStatus.NEW, AlpacaOrderStatus.HELD),  # the stop is the working half
    ],
)
def test_an_entry_cancelled_after_a_partial_fill_is_protected_by_its_working_legs(
    stop_status, take_profit_status
):
    cancelled = _entry(
        "AAA",
        AlpacaOrderStatus.CANCELED,
        filled=4,
        legs=_legs("AAA", stop_status, take_profit_status),
    )

    assert find_unprotected_positions([_position("AAA", 4)], [], lambda _s: [cancelled]) == []


@pytest.mark.parametrize(
    "stop_status, take_profit_status, filled",
    [
        (AlpacaOrderStatus.HELD, AlpacaOrderStatus.HELD, 4),  # nothing works: no evidence
        (AlpacaOrderStatus.CANCELED, AlpacaOrderStatus.CANCELED, 4),  # the legs went with it
        (AlpacaOrderStatus.HELD, AlpacaOrderStatus.NEW, 0),  # cancelled before any fill
    ],
)
def test_other_states_of_a_cancelled_entry_do_not_protect(stop_status, take_profit_status, filled):
    cancelled = _entry(
        "AAA",
        AlpacaOrderStatus.CANCELED,
        filled=filled,
        legs=_legs("AAA", stop_status, take_profit_status),
    )

    (gap,) = find_unprotected_positions([_position("AAA", 4)], [], lambda _s: [cancelled])
    assert gap.covered == Decimal(0)
