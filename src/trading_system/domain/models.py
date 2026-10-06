from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    DONE_FOR_DAY = "DONE_FOR_DAY"
    CANCELED = "CANCELED"
    REPLACED = "REPLACED"
    PENDING_CANCEL = "PENDING_CANCEL"
    PENDING_REPLACE = "PENDING_REPLACE"
    ACCEPTED = "ACCEPTED"
    PENDING_NEW = "PENDING_NEW"
    ACCEPTED_FOR_BIDDING = "ACCEPTED_FOR_BIDDING"
    STOPPED = "STOPPED"
    REJECTED = "REJECTED"
    SUSPENDED = "SUSPENDED"
    CALCULATED = "CALCULATED"
    EXPIRED = "EXPIRED"


def _validate_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(timezone.utc)


class Bar(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    @field_validator("timestamp")
    @classmethod
    def check_utc(cls, v: datetime) -> datetime:
        return _validate_utc(v)


class StrategySignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    strategy_version: str
    symbol: str
    timeframe: str
    closed_bar_timestamp: datetime
    direction: SignalDirection
    confidence: Optional[Decimal] = None
    reason: Optional[str] = None

    @field_validator("closed_bar_timestamp")
    @classmethod
    def check_utc(cls, v: datetime) -> datetime:
        return _validate_utc(v)

    @property
    def deterministic_id(self) -> str:
        ts = int(self.closed_bar_timestamp.timestamp())
        return f"{self.strategy_version}_{self.symbol}_{self.timeframe}_{ts}_{self.direction.value}"


class OracleDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    signal_id: str
    approved: bool
    reason: Optional[str] = None
    prompt_used: Optional[str] = None
    raw_response: Optional[str] = None


class RiskDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    signal_id: str
    oracle_approved: bool
    risk_approved: bool
    position_size: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    reason: Optional[str] = None

class OrderIntent(BaseModel):
    model_config = ConfigDict(frozen=True)
    signal_id: str
    symbol: str
    side: OrderSide
    qty: Decimal
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "day"


class BrokerOrder(BaseModel):
    model_config = ConfigDict(frozen=True)
    broker_id: str
    client_order_id: str
    symbol: str
    status: OrderStatus
    side: OrderSide
    qty: Decimal
    filled_qty: Decimal
    average_fill_price: Optional[Decimal] = None
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def check_utc(cls, v: datetime) -> datetime:
        return _validate_utc(v)


class PositionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    qty: Decimal
    market_value: Decimal
    unrealized_pl: Decimal
    current_price: Decimal
    avg_entry_price: Decimal


class AccountSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    initial_margin: Decimal
    maintenance_margin: Decimal
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def check_utc(cls, v: datetime) -> datetime:
        return _validate_utc(v)


class TradingState(BaseModel):
    model_config = ConfigDict(frozen=True)
    is_trading_halted: bool = False
    halt_reason: Optional[str] = None
    daily_equity_base: Optional[Decimal] = None
    daily_equity_timestamp: Optional[datetime] = None
    updated_at: datetime

    @field_validator("daily_equity_timestamp", "updated_at")
    @classmethod
    def check_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is not None:
            return _validate_utc(v)
        return v
