import os
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy import text
from alembic import command
from alembic.config import Config

from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.models import Base
from trading_system.storage.repositories import (
    SignalRepository,
    OrderRepository,
    StateRepository,
)
from trading_system.domain.models import StrategySignal, BrokerOrder, TradingState
from trading_system.domain.enums import SignalDirection, OrderSide, OrderType, OrderStatus


# Set test environment variables to satisfy Settings
os.environ["ALPACA_API_KEY"] = "test_key"
os.environ["ALPACA_SECRET_KEY"] = "test_secret"


@pytest.fixture
def engine(tmp_path):
    # Use a file-based sqlite db for alembic since in-memory is hard to share between alembic context and test engine
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    eng = get_engine(url)
    return eng, url


@pytest.fixture
def session_factory(engine):
    eng, _ = engine
    return get_session_factory(eng)


def test_migration_and_empty_db(engine, monkeypatch):
    eng, url = engine

    # Configure alembic to use test db url
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", url)

    # Run migrations
    command.upgrade(alembic_cfg, "head")

    # Verify tables are created
    with eng.connect() as conn:
        result = conn.execute(text("SELECT count(*) FROM strategy_signals")).scalar()
        assert result == 0


def test_duplicate_signal(engine, session_factory):
    eng, url = engine
    Base.metadata.create_all(eng)
    signal = StrategySignal(
        version="1.0",
        symbol="AAPL",
        timeframe="15Min",
        closed_bar_timestamp=datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc),
        direction=SignalDirection.LONG,
    )
    repo = SignalRepository(session_factory)

    # First save should succeed
    assert repo.save_signal(signal) is True

    # Second save of same signal should return False and not raise
    assert repo.save_signal(signal) is False

    # The duplicate did not disturb the stored row or the repository
    other = signal.model_copy(update={"signal_id": "another-id"})
    assert repo.save_signal(other) is True


def test_duplicate_fill(engine, session_factory):
    eng, url = engine
    Base.metadata.create_all(eng)
    repo = OrderRepository(session_factory)

    # Needs a broker order to satisfy foreign key
    broker_order = BrokerOrder(
        broker_id="broker123",
        client_order_id="client123",
        symbol="AAPL",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        quantity=Decimal("10.0"),
        filled_quantity=Decimal("10.0"),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    repo.save_broker_order(broker_order)

    # First save of fill should succeed
    assert repo.save_fill("broker123", "exec123", 150.0, 10.0, datetime.now(timezone.utc)) is True

    # Second save with same execution ID should return False and not raise
    assert repo.save_fill("broker123", "exec123", 150.0, 10.0, datetime.now(timezone.utc)) is False


def test_state_persistence(engine, session_factory):
    eng, url = engine
    Base.metadata.create_all(eng)
    repo = StateRepository(session_factory)

    # Get default state
    state = repo.get_state()
    assert state.is_trading_halted is False
    assert state.daily_start_equity is None

    # Update and save state: no commit step, the repository commits by itself
    state.is_trading_halted = True
    state.halt_reason = "Circuit Breaker"
    state.daily_start_equity = Decimal("100000.00")
    repo.save_state(state)

    # Retrieve through a brand-new engine, as a different process would
    other = StateRepository(get_session_factory(get_engine(url)))
    loaded_state = other.get_state()

    assert loaded_state.is_trading_halted is True
    assert loaded_state.halt_reason == "Circuit Breaker"
    assert loaded_state.daily_start_equity == Decimal("100000.00")


def test_get_state_is_stable_across_repeated_calls(engine, session_factory):
    eng, url = engine
    Base.metadata.create_all(eng)
    repo = StateRepository(session_factory)

    first = repo.get_state()
    second = repo.get_state()

    assert first.is_trading_halted is second.is_trading_halted is False
    assert first.daily_start_equity is second.daily_start_equity is None


def test_save_state_before_any_read_creates_the_row(engine, session_factory):
    eng, url = engine
    Base.metadata.create_all(eng)
    repo = StateRepository(session_factory)

    repo.save_state(TradingState(is_trading_halted=True, halt_reason="first write"))

    assert repo.get_state().halt_reason == "first write"
