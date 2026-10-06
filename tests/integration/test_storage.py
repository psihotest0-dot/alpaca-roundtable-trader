import os
from alembic.config import Config
from alembic import command
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trading_system.domain.models import OrderSide, SignalDirection, StrategySignal, TradingState
from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.models import Base
from trading_system.storage.repository import (
    DuplicateFillError,
    DuplicateSignalError,
    FillRepository,
    SignalRepository,
    TradingStateRepository,
    UnitOfWork,
)

@pytest.fixture
def engine():
    engine = get_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture
def session_factory(engine):
    return get_session_factory(engine)

@pytest.fixture
def uow(session_factory):
    return UnitOfWork(session_factory)

def test_duplicate_signal(uow):
    repo = SignalRepository(uow)
    dt = datetime(2023, 10, 25, 14, 30, tzinfo=timezone.utc)
    signal = StrategySignal(
        strategy_version="v1.0",
        symbol="AAPL",
        timeframe="15Min",
        closed_bar_timestamp=dt,
        direction=SignalDirection.LONG,
    )

    # Save first time should succeed
    repo.save(signal)

    # Save second time should raise error
    with pytest.raises(DuplicateSignalError):
        repo.save(signal)

def test_duplicate_fill(uow):
    repo = FillRepository(uow)
    dt = datetime(2023, 10, 25, 14, 30, tzinfo=timezone.utc)
    fill_data = {
        "broker_execution_id": "exec_123",
        "broker_order_id": "order_123",
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "qty": Decimal("10.0"),
        "price": Decimal("150.0"),
        "timestamp": dt,
    }

    # Save first time should succeed
    repo.save_fill(fill_data)

    # Save second time should raise error
    with pytest.raises(DuplicateFillError):
        repo.save_fill(fill_data)


def test_transaction_rollback(session_factory, uow):
    repo = SignalRepository(uow)
    dt = datetime(2023, 10, 25, 14, 30, tzinfo=timezone.utc)
    signal = StrategySignal(
        strategy_version="v1.0",
        symbol="AAPL",
        timeframe="15Min",
        closed_bar_timestamp=dt,
        direction=SignalDirection.LONG,
    )

    # Try to save the signal but force an error to trigger a rollback
    # We will simulate this by causing an IntegrityError intentionally
    repo.save(signal)

    # The second save will raise DuplicateSignalError, which happens AFTER session.add()
    # It will trigger the rollback in UoW
    with pytest.raises(DuplicateSignalError):
        with uow.begin():
            # Re-using the same UoW begin block here manually to see rollback
            # actually we can just call repo.save(signal) because it has its own UoW begin block.
            # But we want to test rollback explicitly:
            repo.save(signal)

    # Verify rollback by ensuring a new distinct signal CAN be inserted,
    # meaning the session isn't broken
    dt2 = datetime(2023, 10, 25, 15, 30, tzinfo=timezone.utc)
    signal2 = StrategySignal(
        strategy_version="v1.0",
        symbol="AAPL",
        timeframe="15Min",
        closed_bar_timestamp=dt2,
        direction=SignalDirection.SHORT,
    )
    repo.save(signal2)


def test_runtime_state_save_and_restore(uow):
    repo = TradingStateRepository(uow)

    # Test get_state when empty
    initial_state = repo.get_state()
    assert initial_state.is_trading_halted is False
    assert initial_state.halt_reason is None

    # Test save and retrieve
    dt = datetime(2023, 10, 25, 14, 30, tzinfo=timezone.utc)
    new_state = TradingState(
        is_trading_halted=True,
        halt_reason="Circuit breaker",
        daily_equity_base=Decimal("10000.50"),
        daily_equity_timestamp=dt,
        updated_at=dt,
    )
    repo.save_state(new_state)

    restored_state = repo.get_state()
    assert restored_state.is_trading_halted is True
    assert restored_state.halt_reason == "Circuit breaker"
    assert restored_state.daily_equity_base == Decimal("10000.50")
    assert restored_state.daily_equity_timestamp == dt
    assert restored_state.updated_at == dt

def test_alembic_migrations():
    alembic_cfg = Config("alembic.ini")
    db_path = "./var/trading.db"

    # Ensure fresh test DB
    if os.path.exists(db_path):
        os.remove(db_path)

    # Test upgrade to head
    command.upgrade(alembic_cfg, "head")
    assert os.path.exists(db_path)

    # Test downgrade
    command.downgrade(alembic_cfg, "base")

    # Cleanup test db
    os.remove(db_path)
