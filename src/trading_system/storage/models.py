from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Numeric, Enum, ForeignKey
from sqlalchemy.orm import DeclarativeBase

from ..domain.enums import SignalDirection, OrderSide, OrderType, OrderStatus, OracleVerdict


class Base(DeclarativeBase):
    pass


class StrategySignalModel(Base):
    __tablename__ = "strategy_signals"

    signal_id = Column(String, primary_key=True)
    version = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    closed_bar_timestamp = Column(DateTime(timezone=True), nullable=False)
    direction: Column[SignalDirection] = Column(Enum(SignalDirection), nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class OracleDecisionModel(Base):
    __tablename__ = "oracle_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Every answer is a row (a signal on HOLD is asked many times), so there is no unique key;
    # the index serves "the final decision for this signal".
    signal_id = Column(String, ForeignKey("strategy_signals.signal_id"), nullable=False, index=True)
    verdict: Column[OracleVerdict] = Column(Enum(OracleVerdict), nullable=False)
    reasoning = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    provider = Column(String, nullable=False, default="")
    failed = Column(Boolean, nullable=False, default=False)  # no answer: only ever a HOLD
    decision_timestamp = Column(DateTime(timezone=True), nullable=False)


class MarketBarModel(Base):
    """A closed candle from Alpaca (Phase 3.2). A cache, not a record: `fetch-history` can
    rebuild all of it. Prices are floats, as they are in the data frames the strategy reads."""

    __tablename__ = "market_bars"

    symbol = Column(String, primary_key=True)
    timeframe = Column(String, primary_key=True)  # "15m" or "1h"
    bar_start = Column(DateTime(timezone=True), primary_key=True)  # UTC, Alpaca's convention
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float, nullable=True)
    trade_count = Column(Integer, nullable=True)
    adjustment = Column(String, nullable=False)  # the price basis the bar was downloaded with
    fetched_at = Column(DateTime(timezone=True), nullable=False)


class RiskDecisionModel(Base):
    __tablename__ = "risk_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String, ForeignKey("strategy_signals.signal_id"), nullable=False)
    approved = Column(Boolean, nullable=False)
    reasoning = Column(String, nullable=False)
    suggested_quantity = Column(Numeric(precision=18, scale=8), nullable=True)
    suggested_limit_price = Column(Numeric(precision=18, scale=8), nullable=True)
    decision_timestamp = Column(DateTime(timezone=True), nullable=False)


class OrderIntentModel(Base):
    __tablename__ = "order_intents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String, ForeignKey("strategy_signals.signal_id"), nullable=False)
    symbol = Column(String, nullable=False)
    side: Column[OrderSide] = Column(Enum(OrderSide), nullable=False)
    type: Column[OrderType] = Column(Enum(OrderType), nullable=False)
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    limit_price = Column(Numeric(precision=18, scale=8), nullable=True)
    stop_price = Column(Numeric(precision=18, scale=8), nullable=True)
    take_profit_price = Column(Numeric(precision=18, scale=8), nullable=True)
    intent_timestamp = Column(DateTime(timezone=True), nullable=False)


class BrokerOrderModel(Base):
    __tablename__ = "broker_orders"

    # Our own id is the identity of an order: we generate it before anything is sent, so
    # it exists for the whole life of the order. The broker's id only exists once the
    # broker has answered, hence nullable. Saving the same client_order_id again updates
    # the row (the in-flight placeholder becomes the confirmed order).
    client_order_id = Column(String, primary_key=True)
    broker_id = Column(String, unique=True, nullable=True)
    symbol = Column(String, nullable=False)
    side: Column[OrderSide] = Column(Enum(OrderSide), nullable=False)
    type: Column[OrderType] = Column(Enum(OrderType), nullable=False)
    status: Column[OrderStatus] = Column(Enum(OrderStatus), nullable=False)
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    filled_quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    limit_price = Column(Numeric(precision=18, scale=8), nullable=True)
    stop_price = Column(Numeric(precision=18, scale=8), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class FillModel(Base):
    __tablename__ = "fills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    broker_order_id = Column(String, ForeignKey("broker_orders.broker_id"), nullable=False)
    broker_execution_id = Column(String, unique=True, nullable=False)
    price = Column(Numeric(precision=18, scale=8), nullable=False)
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)


class RuntimeStateModel(Base):
    __tablename__ = "runtime_state"

    id = Column(Integer, primary_key=True)  # Singleton pattern: id=1
    is_trading_halted = Column(Boolean, nullable=False, default=False)
    halt_reason = Column(String, nullable=True)
    daily_start_equity = Column(Numeric(precision=18, scale=8), nullable=True)
    daily_equity_date = Column(DateTime(timezone=True), nullable=True)
    last_updated = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class AuditEventModel(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String, nullable=False)
    details = Column(String, nullable=False)  # Storing JSON string for flexibility
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class ReconciliationCheckpointModel(Base):
    __tablename__ = "reconciliation_checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    checkpoint_time = Column(DateTime(timezone=True), nullable=False)
    status = Column(String, nullable=False)
    discrepancies = Column(String, nullable=True)  # JSON string if any issues found
