from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional
import hashlib
from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import SignalDirection, OrderSide, OrderType, OrderStatus, OracleVerdict


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

    @model_validator(mode="after")
    def compute_signal_id(self) -> "StrategySignal":
        if not self.signal_id:
            # Deterministic ID: version + symbol + timeframe + closed_bar_timestamp + direction
            unique_string = f"{self.version}_{self.symbol}_{self.timeframe}_{self.closed_bar_timestamp.isoformat()}_{self.direction.value}"
            self.signal_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
        return self


class OracleDecision(BaseModel):
    """What the Oracle said about one signal. It never proposes an order, a size or a price.

    `confidence` and `reason` are for the record and for later analysis; nothing is decided
    from them. `failed` marks "the Oracle did not answer" (timeout, unreadable output): that is
    not an opinion, so it can only be a HOLD and it is never cached as a VETO (H1)."""

    signal_id: str
    verdict: OracleVerdict
    reason: str = ""
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    provider: str = ""
    failed: bool = False
    decision_timestamp: datetime = Field(default_factory=_now_utc)

    @field_validator("reason")
    @classmethod
    def _short_reason(cls, value: str) -> str:
        return value[:500]

    @model_validator(mode="after")
    def _failure_is_not_an_opinion(self) -> "OracleDecision":
        if self.failed and self.verdict is not OracleVerdict.HOLD:
            raise ValueError("A failed Oracle call can only be a HOLD.")
        return self

    @property
    def approved(self) -> bool:
        return self.verdict is OracleVerdict.APPROVE


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
    take_profit_price: Optional[Decimal] = None
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
    # Alpaca leaves it empty on the paper account, so "unknown" has to be possible. It is not 0:
    # the pattern day trader rule (Phase 4) must not read a missing count as "no day trades".
    day_trade_count: Optional[int] = None
    updated_at: datetime


class TradingState(BaseModel):
    is_trading_halted: bool = False
    halt_reason: Optional[str] = None
    daily_start_equity: Optional[Decimal] = None
    daily_equity_date: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=_now_utc)
