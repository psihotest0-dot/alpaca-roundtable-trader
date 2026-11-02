from decimal import Decimal
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

from trading_system.domain.enums import OrderSide


class BacktestConfig(BaseModel):
    initial_capital: Decimal = Field(default=Decimal("100000.0"))
    fee_per_trade: Decimal = Field(
        default=Decimal("0.0"), description="Flat fee per trade execution."
    )
    fee_pct_per_trade: float = Field(
        default=0.0001, description="Percentage fee of notional value (e.g. 0.0001 for 1 bps)."
    )
    slippage_pct: float = Field(
        default=0.0005,
        description="Slippage percentage applied to execution price (e.g. 0.0005 for 5 bps).",
    )
    allow_fractional: bool = False
    replay_oracle_path: Optional[str] = None


class BacktestTrade(BaseModel):
    symbol: str
    side: OrderSide
    quantity: Decimal
    entry_price: Decimal
    entry_time: datetime
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[Decimal] = None
    pnl_pct: Optional[float] = None
    fees_paid: Decimal = Decimal("0.0")
    slippage_paid: Decimal = Decimal("0.0")


class BacktestPosition(BaseModel):
    symbol: str
    quantity: Decimal
    average_entry_price: Decimal
    entry_time: datetime
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None


class BacktestResult(BaseModel):
    initial_capital: Decimal
    final_capital: Decimal
    total_return_pct: float
    max_drawdown_pct: float
    trade_count: int
    win_rate: float
    profit_factor: float
    total_fees: Decimal
    total_slippage: Decimal
    max_exposure_pct: float
    trades: List[BacktestTrade] = Field(default_factory=list)

    assumptions_and_limitations: List[str] = Field(
        default_factory=lambda: [
            "No look-ahead bias: signals generated strictly on closed bars.",
            "Gap-aware stop model applied: filled at open if stop/target gap.",
            "Conservative intrabar collision: assumed stop-loss hit first if both SL and TP triggered in same bar.",
            "Data assumption: candles are the ones stored by `fetch-history`, adjusted for splits and dividends on the same basis as the live loop.",
            "Strategy parameters are the live loop's (RUNTIME_STRATEGY) and entries are looked for only in the regular session. Pre- and after-market bars are still in the indicators.",
            "Entry timing differs from the live system: the backtest places a limit at the signal bar's close and checks it from the bar after next; the live system sends a marketable limit (live price + ENTRY_LIMIT_BUFFER_PCT) that fills at once. Results are not comparable with live fills yet (Phase 3.1).",
            "Not the live system yet: position sizing, exposure limits and the entry plan are a simplified copy of the RiskEngine (Phase 3.1).",
            "Survivorship bias: Uses a fixed watchlist, ignoring delistings or index changes over the historical period.",
            "Results do not guarantee or promise future profitability.",
        ]
    )
