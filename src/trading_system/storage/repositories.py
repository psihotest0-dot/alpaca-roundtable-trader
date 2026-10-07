from contextlib import contextmanager
from typing import Generator, Optional, cast
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime
from decimal import Decimal
import os

from .database import get_engine, get_session_factory
from .models import (
    StrategySignalModel,
    OrderIntentModel,
    BrokerOrderModel,
    FillModel,
    RuntimeStateModel,
)
from ..domain.models import StrategySignal, OrderIntent, BrokerOrder, TradingState


# Ensure settings work even when environment variables aren't set during module load
_url = os.getenv("DATABASE_URL", "sqlite:///./var/trading.db")
engine = get_engine(_url)
SessionFactory = get_session_factory(engine)


@contextmanager
def transaction() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


class SignalRepository:
    def __init__(self, session: Session):
        self.session = session

    def save_signal(self, signal: StrategySignal) -> bool:
        """Saves a signal. Returns True if saved, False if it already exists (duplicate)."""
        model = StrategySignalModel(
            signal_id=signal.signal_id,
            version=signal.version,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            closed_bar_timestamp=signal.closed_bar_timestamp,
            direction=signal.direction,
            confidence=signal.confidence,
        )
        try:
            self.session.add(model)
            self.session.flush()
            return True
        except IntegrityError:
            self.session.rollback()
            return False


class OrderRepository:
    def __init__(self, session: Session):
        self.session = session

    def save_intent(self, intent: OrderIntent) -> OrderIntentModel:
        model = OrderIntentModel(
            signal_id=intent.signal_id,
            symbol=intent.symbol,
            side=intent.side,
            type=intent.type,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            intent_timestamp=intent.intent_timestamp,
        )
        self.session.add(model)
        self.session.flush()
        return model

    def save_broker_order(self, order: BrokerOrder) -> BrokerOrderModel:
        model = BrokerOrderModel(
            broker_id=order.broker_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            type=order.type,
            status=order.status,
            quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )
        self.session.merge(model)
        self.session.flush()
        return model

    def save_fill(
        self,
        broker_order_id: str,
        execution_id: str,
        price: float,
        quantity: float,
        timestamp: datetime,
    ) -> bool:
        """Saves a fill. Returns True if saved, False if it already exists."""
        model = FillModel(
            broker_order_id=broker_order_id,
            broker_execution_id=execution_id,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        )
        try:
            self.session.add(model)
            self.session.flush()
            return True
        except IntegrityError:
            self.session.rollback()
            return False


class StateRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_state(self) -> TradingState:
        model = self.session.query(RuntimeStateModel).filter_by(id=1).first()
        if not model:
            # Initialize default state
            model = RuntimeStateModel(id=1)
            self.session.add(model)
            self.session.flush()

        return TradingState(
            is_trading_halted=bool(model.is_trading_halted),
            halt_reason=str(model.halt_reason) if model.halt_reason else None,
            daily_start_equity=cast(Optional[Decimal], model.daily_start_equity),
            daily_equity_date=cast(Optional[datetime], model.daily_equity_date),
            last_updated=cast(datetime, model.last_updated),
        )

    def save_state(self, state: TradingState) -> None:
        model = self.session.query(RuntimeStateModel).filter_by(id=1).first()
        if not model:
            model = RuntimeStateModel(id=1)
            self.session.add(model)

        model.is_trading_halted = state.is_trading_halted  # type: ignore
        model.halt_reason = state.halt_reason  # type: ignore
        model.daily_start_equity = state.daily_start_equity  # type: ignore
        model.daily_equity_date = state.daily_equity_date  # type: ignore
        self.session.flush()
