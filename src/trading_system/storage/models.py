from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Enum, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase

from trading_system.domain.models import OrderSide, OrderStatus, OrderType, SignalDirection


class Base(DeclarativeBase):
    pass


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class StrategySignalModel(Base):
    __tablename__ = "strategy_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    deterministic_id = Column(String(255), unique=True, nullable=False, index=True)
    strategy_version = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(20), nullable=False)
    closed_bar_timestamp = Column(DateTime(timezone=True), nullable=False)
    direction: Column[SignalDirection] = Column(Enum(SignalDirection), nullable=False)
    confidence = Column(Numeric(10, 4), nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now_utc, nullable=False)


class OracleDecisionModel(Base):
    __tablename__ = "oracle_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(255), nullable=False, index=True)
    approved = Column(Boolean, nullable=False)
    reason = Column(Text, nullable=True)
    prompt_used = Column(Text, nullable=True)
    raw_response = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now_utc, nullable=False)


class RiskDecisionModel(Base):
    __tablename__ = "risk_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(255), nullable=False, index=True)
    oracle_approved = Column(Boolean, nullable=False)
    risk_approved = Column(Boolean, nullable=False)
    position_size = Column(Numeric(20, 8), nullable=True)
    stop_loss_price = Column(Numeric(20, 8), nullable=True)
    take_profit_price = Column(Numeric(20, 8), nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now_utc, nullable=False)


class OrderIntentModel(Base):
    __tablename__ = "order_intents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(255), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    side: Column[OrderSide] = Column(Enum(OrderSide), nullable=False)
    qty = Column(Numeric(20, 8), nullable=False)
    order_type: Column[OrderType] = Column(Enum(OrderType), nullable=False)
    limit_price = Column(Numeric(20, 8), nullable=True)
    stop_price = Column(Numeric(20, 8), nullable=True)
    time_in_force = Column(String(20), nullable=False, default="day")
    created_at = Column(DateTime(timezone=True), default=_now_utc, nullable=False)

class BrokerOrderModel(Base):
    __tablename__ = "broker_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    broker_id = Column(String(255), nullable=False, unique=True, index=True)
    client_order_id = Column(String(255), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    status: Column[OrderStatus] = Column(Enum(OrderStatus), nullable=False)
    side: Column[OrderSide] = Column(Enum(OrderSide), nullable=False)
    qty = Column(Numeric(20, 8), nullable=False)
    filled_qty = Column(Numeric(20, 8), nullable=False, default=0)
    average_fill_price = Column(Numeric(20, 8), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    db_created_at = Column(DateTime(timezone=True), default=_now_utc, nullable=False)


class FillModel(Base):
    __tablename__ = "fills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    broker_execution_id = Column(String(255), nullable=False, unique=True, index=True)
    broker_order_id = Column(String(255), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    side: Column[OrderSide] = Column(Enum(OrderSide), nullable=False)
    qty = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now_utc, nullable=False)


class RuntimeStateModel(Base):
    __tablename__ = "runtime_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    is_trading_halted = Column(Boolean, nullable=False, default=False)
    halt_reason = Column(Text, nullable=True)
    daily_equity_base = Column(Numeric(20, 8), nullable=True)
    daily_equity_timestamp = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=_now_utc, nullable=False)


class AuditEventModel(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(100), nullable=False, index=True)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=_now_utc, nullable=False)


class ReconciliationCheckpointModel(Base):
    __tablename__ = "reconciliation_checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status: Column[str] = Column(String(50), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=_now_utc, nullable=False)


class DailyEquityBaseModel(Base):
    __tablename__ = "daily_equity_base"

    id = Column(Integer, primary_key=True, autoincrement=True)
    equity = Column(Numeric(20, 8), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=_now_utc, nullable=False)
