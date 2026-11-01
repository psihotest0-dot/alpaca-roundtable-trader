from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional
import hashlib
from pydantic import BaseModel, Field, model_validator

from .enums import SignalDirection, OrderSide, OrderType, OrderStatus


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Bar(BaseModel):
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class StrategySignal(BaseModel):
    version: str
    symbol: str
    timeframe: str
    closed_bar_timestamp: datetime
    direction: SignalDirection
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    signal_id: Optional[str] = None
    expires_at: Optional[datetime] = None

    # Used internally by the backtest engine; not part of the standard runtime
    _suggested_limit: Optional[Decimal] = None
    _suggested_shares: Optional[Decimal] = None

    @model_validator(mode="after")
    def compute_signal_id(self) -> "StrategySignal":
        if not self.signal_id:
            # Deterministic ID: version + symbol + timeframe + closed_bar_timestamp + direction
            unique_string = f"{self.version}_{self.symbol}_{self.timeframe}_{self.closed_bar_timestamp.isoformat()}_{self.direction.value}"
            self.signal_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
        return self


class OracleDecision(BaseModel):
    signal_id: str
    approved: bool
    reasoning: str
    decision_timestamp: datetime = Field(default_factory=_now_utc)


class RiskDecision(BaseModel):
    signal_id: str
    approved: bool
    reasoning: str
    decision_timestamp: datetime = Field(default_factory=_now_utc)
    suggested_quantity: Optional[Decimal] = None
    suggested_limit_price: Optional[Decimal] = None


class OrderIntent(BaseModel):
    signal_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    intent_timestamp: datetime = Field(default_factory=_now_utc)


class BrokerOrder(BaseModel):
    client_order_id: str
    # None until the broker has acknowledged the order (e.g. while it is SUBMITTING).
    broker_id: Optional[str] = None
    symbol: str
    side: OrderSide
    type: OrderType
    status: OrderStatus
    quantity: Decimal
    filled_quantity: Decimal
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    created_at: datetime
    updated_at: datetime


class PositionSnapshot(BaseModel):
    symbol: str
    quantity: Decimal
    market_value: Decimal
    average_entry_price: Decimal
    unrealized_pl: Decimal


class AccountSnapshot(BaseModel):
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    day_trade_count: int
    updated_at: datetime


class TradingState(BaseModel):
    is_trading_halted: bool = False
    halt_reason: Optional[str] = None
    daily_start_equity: Optional[Decimal] = None
    daily_equity_date: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=_now_utc)
