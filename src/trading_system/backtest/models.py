from decimal import Decimal
from datetime import datetime
from typing import Dict, Optional, List, Tuple
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
    # What the fills cost against the price the engine saw (entry and exit together). It is
    # already inside entry_price, exit_price and pnl: this is a breakdown, not an extra cost.
    slippage_paid: Decimal = Decimal("0.0")


class BacktestPosition(BaseModel):
    symbol: str
    quantity: Decimal
    average_entry_price: Decimal
    entry_time: datetime
    entry_fee: Decimal = Decimal("0.0")
    entry_slippage: Decimal = Decimal("0.0")
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
    # Signals the strategy produced for symbols without a position, and what happened to the
    # ones that did not become a trade (the risk engine's own reason, or why no price).
    signals: int = 0
    not_traded: Dict[str, int] = Field(default_factory=dict)
    # The equity at the end of each New York trading day (ISO date -> value), in date order.
    daily_equity: Dict[str, float] = Field(default_factory=dict)

    def daily_returns(self) -> List[Tuple[str, float]]:
        """(day, change of the equity over that day as a fraction), the first day measured from
        the initial capital."""
        returns: List[Tuple[str, float]] = []
        previous = float(self.initial_capital)
        for day, equity in sorted(self.daily_equity.items()):
            returns.append((day, equity / previous - 1 if previous else 0.0))
            previous = equity
        return returns

    assumptions_and_limitations: List[str] = Field(
        default_factory=lambda: [
            "No look-ahead bias: signals generated strictly on closed bars.",
            "Gap-aware stop model applied: filled at open if stop/target gap.",
            "Conservative intrabar collision: assumed stop-loss hit first if both SL and TP triggered in same bar.",
            "Data assumption: candles are the ones stored by `fetch-history`, adjusted for splits and dividends on the same basis as the live loop.",
            "Strategy parameters are the live loop's (RUNTIME_STRATEGY) and entries are looked for only in the regular session. Pre- and after-market bars are still in the indicators.",
            "Entries follow the live loop: the signal comes from the bar that just closed, the decision is taken ENTRY_DELAY (30 s) after the close, the price is the first print of the next bar (it has to pass the same freshness and 5 % distance check as the live price), and the entry plan, sizing and every limit come from the real RiskEngine (limit = price + ENTRY_LIMIT_BUFFER_PCT, stop and target measured from the limit). The order fills at once at price + slippage, never above the limit.",
            "The stop and the target of a new position are checked from the bar it enters in, over the whole bar (including the first 30 s before the entry).",
            "Account as the risk engine sees it: equity is cash plus positions marked at the last closed bar; buying power is the cash (no margin); the daily loss baseline is the equity at the end of the previous New York day. The pattern day trader rule is not modelled (Phase 4). An entry order that does not fill at once (the price runs away from the limit) is not modelled.",
            "Fees are counted at entry and at exit. Slippage is a fixed share of the price (BacktestConfig.slippage_pct), part of every fill price and listed on its own in the report; it is not drawn from real spreads.",
            "The strategy sees the same window of history as the live loop (LOOKBACK_DAYS of 15-minute bars, HTF_LOOKBACK_DAYS of 1-hour bars), never bars from after the decision. The first days of a replay have less history and are warm-up.",
            "Survivorship bias: Uses a fixed watchlist, ignoring delistings or index changes over the historical period.",
            "Results do not guarantee or promise future profitability.",
        ]
    )
