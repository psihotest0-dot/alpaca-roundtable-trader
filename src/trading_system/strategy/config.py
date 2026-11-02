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


# The parameters the trading loop runs with. They are not the defaults above (a 12/26 EMA
# pair and RSI 30-70 instead of 20/50 and 50-70): the loop has always used these, they used
# to be written out inside the orchestrator, and the tests copy them from here. Choosing
# the strategy's parameters is a separate decision (Phase 4); this only keeps them in one place.
RUNTIME_STRATEGY = StrategyConfig(
    strategy_version="1.0.0",
    fast_ema_period=12,
    slow_ema_period=26,
    rsi_period=14,
    rsi_lower_bound=30,
    rsi_upper_bound=70,
    atr_period=14,
    htf_fast_ema_period=12,
    htf_slow_ema_period=26,
    min_warmup_bars=51,
)
