from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    strategy_version: str = "v1.0"

    # 15m Timeframe Configuration
    fast_ema_period: int = Field(20, gt=0, description="Fast EMA period")
    slow_ema_period: int = Field(50, gt=0, description="Slow EMA period")

    rsi_period: int = Field(14, gt=0, description="RSI period")
    rsi_lower_bound: float = Field(50.0, description="Lower bound for RSI entry")
    rsi_upper_bound: float = Field(70.0, description="Upper bound for RSI entry")

    atr_period: int = Field(14, gt=0, description="ATR period (diagnostic)")

    # 1h Timeframe Configuration
    htf_fast_ema_period: int = Field(20, gt=0, description="Higher timeframe Fast EMA period")
    htf_slow_ema_period: int = Field(50, gt=0, description="Higher timeframe Slow EMA period")

    # Safety limits
    min_warmup_bars: int = Field(
        100, gt=50, description="Minimum bars needed for stable indicators"
    )
