"""Repositories over the SQLAlchemy models.

Every public method runs in its own short transaction: it opens a session, does
its work, commits, and closes the session, even when it fails. Nothing here keeps
a session (and with it a SQLite write lock) open between calls, so a halt or an
order record is durable the moment the method returns, and one failed write can
never leave a half-broken session behind for the next caller.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional, cast

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ..domain.enums import OracleVerdict, OrderSide, OrderStatus, OrderType
from ..domain.models import (
    BrokerOrder,
    OracleDecision,
    OrderIntent,
    StrategySignal,
    TradingState,
)
from .models import (
    BrokerOrderModel,
    FillModel,
    OracleDecisionModel,
    OrderIntentModel,
    RuntimeStateModel,
    StrategySignalModel,
)

SessionFactory = sessionmaker[Session]

# runtime_state is a singleton table: the one row has this id.
_STATE_ROW_ID = 1

# An order in one of these statuses will not change any more.
_FINAL_STATUSES = (
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
)


class SignalRepository:
    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

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
            with self._session_factory.begin() as session:
                session.add(model)
        except IntegrityError:
            return False
        return True


class OrderRepository:
    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    def save_intent(self, intent: OrderIntent) -> int:
        """Saves an order intent and returns its id."""
        model = OrderIntentModel(
            signal_id=intent.signal_id,
            symbol=intent.symbol,
            side=intent.side,
            type=intent.type,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            take_profit_price=intent.take_profit_price,
            intent_timestamp=intent.intent_timestamp,
        )
        with self._session_factory.begin() as session:
            session.add(model)
            session.flush()
            return int(cast(int, model.id))

    def save_broker_order(self, order: BrokerOrder) -> None:
        """Inserts the order, or updates it when this client_order_id is already stored.

        That is how the in-flight placeholder (no broker id yet) turns into the
        confirmed order without creating a second row."""
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
        with self._session_factory.begin() as session:
            session.merge(model)

    def list_unsettled_orders(self) -> list[BrokerOrder]:
        """Orders that may still change at the broker: every status but the final ones."""
        with self._session_factory() as session:
            rows = (
                session.query(BrokerOrderModel)
                .filter(BrokerOrderModel.status.notin_(_FINAL_STATUSES))
                .order_by(BrokerOrderModel.created_at)
                .all()
            )
            return [
                BrokerOrder(
                    client_order_id=cast(str, row.client_order_id),
                    broker_id=cast(Optional[str], row.broker_id),
                    symbol=cast(str, row.symbol),
                    side=cast(OrderSide, row.side),
                    type=cast(OrderType, row.type),
                    status=cast(OrderStatus, row.status),
                    quantity=cast(Decimal, row.quantity),
                    filled_quantity=cast(Decimal, row.filled_quantity),
                    limit_price=cast(Optional[Decimal], row.limit_price),
                    stop_price=cast(Optional[Decimal], row.stop_price),
                    created_at=cast(datetime, row.created_at),
                    updated_at=cast(datetime, row.updated_at),
                )
                for row in rows
            ]

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
            with self._session_factory.begin() as session:
                session.add(model)
        except IntegrityError:
            return False
        return True


class OracleDecisionRepository:
    """What the Oracle said, answer by answer."""

    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    def save(self, decision: OracleDecision) -> None:
        model = OracleDecisionModel(
            signal_id=decision.signal_id,
            verdict=decision.verdict,
            reasoning=decision.reason,
            confidence=decision.confidence,
            provider=decision.provider,
            failed=decision.failed,
            decision_timestamp=decision.decision_timestamp,
        )
        with self._session_factory.begin() as session:
            session.add(model)

    def final_decision_for(self, signal_id: str) -> Optional[OracleDecision]:
        """The latest APPROVE or VETO for the signal. A HOLD, and an answer that never came,
        are not decisions and are never returned."""
        with self._session_factory() as session:
            row = (
                session.query(OracleDecisionModel)
                .filter(
                    OracleDecisionModel.signal_id == signal_id,
                    OracleDecisionModel.failed.is_(False),
                    OracleDecisionModel.verdict.in_((OracleVerdict.APPROVE, OracleVerdict.VETO)),
                )
                .order_by(OracleDecisionModel.id.desc())
                .first()
            )
            return None if row is None else self._decision(row)

    def replay_export(self) -> dict[str, dict[str, Any]]:
        """One entry per signal the Oracle was asked about, for the backtest's `--replay-oracle`:
        its final decision, or, if it never made one (HOLD until the signal lapsed, or no answer),
        the latest answer with `approved` false, because that is what happened: no trade."""
        with self._session_factory() as session:
            rows = session.query(OracleDecisionModel).order_by(OracleDecisionModel.id).all()
            latest: dict[str, OracleDecision] = {}
            for row in rows:
                decision = self._decision(row)
                held_before = latest.get(decision.signal_id)
                is_final = decision.verdict is not OracleVerdict.HOLD
                if is_final or held_before is None or held_before.verdict is OracleVerdict.HOLD:
                    latest[decision.signal_id] = decision
        return {
            signal_id: {
                "approved": d.approved,
                "verdict": d.verdict.value,
                "final": d.verdict is not OracleVerdict.HOLD,
                "reason": d.reason,
                "provider": d.provider,
                "decided_at": d.decision_timestamp.isoformat(),
            }
            for signal_id, d in latest.items()
        }

    @staticmethod
    def _decision(row: OracleDecisionModel) -> OracleDecision:
        moment = cast(datetime, row.decision_timestamp)
        if moment.tzinfo is None:  # SQLite hands back naive datetimes; they were stored as UTC
            moment = moment.replace(tzinfo=timezone.utc)
        return OracleDecision(
            signal_id=cast(str, row.signal_id),
            verdict=cast(OracleVerdict, row.verdict),
            reason=cast(str, row.reasoning),
            confidence=cast(Optional[float], row.confidence),
            provider=cast(str, row.provider),
            failed=cast(bool, row.failed),
            decision_timestamp=moment,
        )


class StateRepository:
    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    def get_state(self) -> TradingState:
        """Returns the runtime state, creating the default row on the very first call."""
        state = self._read()
        if state is not None:
            return state

        # First run. The row is created in its own transaction, and a concurrent
        # creator (another process) is not an error: whoever lost the race re-reads.
        try:
            with self._session_factory.begin() as session:
                session.add(RuntimeStateModel(id=_STATE_ROW_ID))
        except IntegrityError:
            pass
        state = self._read()
        if state is None:  # pragma: no cover - the row exists or the insert above raised
            raise RuntimeError("runtime_state row could not be created")
        return state

    def save_state(self, state: TradingState) -> None:
        with self._session_factory.begin() as session:
            model = session.get(RuntimeStateModel, _STATE_ROW_ID)
            if model is None:
                model = RuntimeStateModel(id=_STATE_ROW_ID)
                session.add(model)

            model.is_trading_halted = state.is_trading_halted  # type: ignore
            model.halt_reason = state.halt_reason  # type: ignore
            model.daily_start_equity = state.daily_start_equity  # type: ignore
            model.daily_equity_date = state.daily_equity_date  # type: ignore

    def save_daily_start_equity(self, equity: Decimal, equity_date: datetime) -> None:
        """Records the start-of-day equity and its trading day, touching nothing else.

        A full save_state() writes every column from a possibly stale copy. If the
        orchestrator did that while `pause-entries` ran in another process, the halt
        could be overwritten with "not halted". These two columns are all this updates."""
        with self._session_factory.begin() as session:
            model = session.get(RuntimeStateModel, _STATE_ROW_ID)
            if model is None:
                model = RuntimeStateModel(id=_STATE_ROW_ID)
                session.add(model)
            model.daily_start_equity = equity  # type: ignore
            model.daily_equity_date = equity_date  # type: ignore

    def _read(self) -> Optional[TradingState]:
        with self._session_factory.begin() as session:
            model = session.get(RuntimeStateModel, _STATE_ROW_ID)
            if model is None:
                return None
            # Build the domain object while the session is still open: after commit
            # the model's attributes are expired and the model is detached.
            return TradingState(
                is_trading_halted=bool(model.is_trading_halted),
                halt_reason=str(model.halt_reason) if model.halt_reason else None,
                daily_start_equity=cast(Optional[Decimal], model.daily_start_equity),
                daily_equity_date=cast(Optional[datetime], model.daily_equity_date),
                last_updated=cast(datetime, model.last_updated),
            )
