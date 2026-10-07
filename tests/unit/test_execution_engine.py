import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

from alpaca.trading.models import Order, Asset
from alpaca.trading.enums import OrderStatus as AlpacaOrderStatus

from trading_system.execution.engine import ExecutionEngine
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
    mock_reconciled_order.order_type = "limit"
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


def test_partial_fill_tracking(engine, mock_broker, mock_order_repo, base_intent):
    # Not testing event websocket stream here, but we can test manual reconciliation mapping
    mock_alpaca_order = MagicMock(spec=Order)
    mock_alpaca_order.id = "partially-filled-id"
    mock_alpaca_order.symbol = "AAPL"
    mock_alpaca_order.side = "buy"
    mock_alpaca_order.order_type = "limit"
    mock_alpaca_order.status = AlpacaOrderStatus.PARTIALLY_FILLED
    mock_alpaca_order.qty = "10"
    mock_alpaca_order.filled_qty = "4"
    mock_alpaca_order.limit_price = "150.00"
    mock_alpaca_order.stop_price = "140.00"
    mock_alpaca_order.created_at = datetime.datetime.now(datetime.timezone.utc)
    mock_alpaca_order.updated_at = datetime.datetime.now(datetime.timezone.utc)

    mock_broker.get_order_by_client_id.return_value = mock_alpaca_order

    result = engine.reconcile_order("exec_test_signal_1")
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
    mock_reconciled_order.order_type = "limit"
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
