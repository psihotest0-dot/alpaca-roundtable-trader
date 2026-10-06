import typing
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Generator

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from trading_system.domain.models import StrategySignal, TradingState
from trading_system.storage.models import FillModel, StrategySignalModel, RuntimeStateModel


class DuplicateSignalError(Exception):
    pass


class DuplicateFillError(Exception):
    pass


class UnitOfWork:
    def __init__(self, session_factory: typing.Callable[[], Session]) -> None:
        self.session_factory = session_factory

    @contextmanager
    def begin(self) -> Generator[Session, None, None]:
        session: Session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class SignalRepository:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    def save(self, signal: StrategySignal) -> None:
        with self.uow.begin() as session:
            model = StrategySignalModel(
                deterministic_id=signal.deterministic_id,
                strategy_version=signal.strategy_version,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                closed_bar_timestamp=signal.closed_bar_timestamp,
                direction=signal.direction,
                confidence=signal.confidence,
                reason=signal.reason,
            )
            session.add(model)
            try:
                session.flush()
            except IntegrityError as e:
                if "UNIQUE constraint failed" in str(e) or "unique constraint" in str(e).lower():
                    raise DuplicateSignalError(f"Signal {signal.deterministic_id} already exists.") from e
                raise


class FillRepository:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    def save_fill(self, fill_data: typing.Dict[str, typing.Any]) -> None:
        with self.uow.begin() as session:
            model = FillModel(**fill_data)
            session.add(model)
            try:
                session.flush()
            except IntegrityError as e:
                if "UNIQUE constraint failed" in str(e) or "unique constraint" in str(e).lower():
                    raise DuplicateFillError(f"Fill {fill_data.get('broker_execution_id')} already exists.") from e
                raise


class TradingStateRepository:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    def get_state(self) -> TradingState:
        with self.uow.begin() as session:
            model = session.query(RuntimeStateModel).order_by(RuntimeStateModel.id.desc()).first()
            if not model:
                return TradingState(updated_at=datetime.now(timezone.utc))

            # Ensure proper timezone assignment if missing in sqlite
            daily_equity_ts = model.daily_equity_timestamp
            if daily_equity_ts and daily_equity_ts.tzinfo is None:
                daily_equity_ts = daily_equity_ts.replace(tzinfo=timezone.utc)

            updated_ts = model.updated_at
            if updated_ts and updated_ts.tzinfo is None:
                updated_ts = updated_ts.replace(tzinfo=timezone.utc)

            return TradingState(
                is_trading_halted=model.is_trading_halted,
                halt_reason=model.halt_reason,
                daily_equity_base=model.daily_equity_base,
                daily_equity_timestamp=daily_equity_ts,
                updated_at=updated_ts,
            )

    def save_state(self, state: TradingState) -> None:
        with self.uow.begin() as session:
            model = RuntimeStateModel(
                is_trading_halted=state.is_trading_halted,
                halt_reason=state.halt_reason,
                daily_equity_base=state.daily_equity_base,
                daily_equity_timestamp=state.daily_equity_timestamp,
                updated_at=state.updated_at,
            )
            session.add(model)
