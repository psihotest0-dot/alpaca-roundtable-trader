"""Deterministic OHLCV fixtures shaped like what Alpaca really returns.

The unit tests feed the strategy frames with a `timestamp` *column*. Alpaca's
`BarSet.df` has a (symbol, timestamp) MultiIndex and no such column, and that
mismatch went unnoticed. These helpers build both shapes from one series.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trading_system.strategy.config import StrategyConfig
from trading_system.strategy.trend import calculate_indicators

# The strategy parameters runtime/orchestrator.py hard-codes today.
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


def _ohlcv(closes: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": 1000.0,
            # Extra columns Alpaca adds; the strategy must tolerate them.
            "trade_count": 10.0,
            "vwap": closes,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def cross_up_closes(seed: int = 1, min_bars: int = 120) -> np.ndarray:
    """Random-walk closes that END on a bar where the fast EMA crosses above the
    slow EMA with RSI inside the runtime bounds (the strategy's entry condition)."""
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 0.3, 400))
    probe = pd.DataFrame(
        {"close": closes, "open": closes, "high": closes + 0.1, "low": closes - 0.1},
        index=pd.RangeIndex(len(closes)),
    )
    probe["volume"] = 1000.0
    ind = calculate_indicators(probe, RUNTIME_STRATEGY)
    crossed = (
        (ind["ema_fast"].shift(1) <= ind["ema_slow"].shift(1))
        & (ind["ema_fast"] > ind["ema_slow"])
        & ind["rsi"].between(RUNTIME_STRATEGY.rsi_lower_bound, RUNTIME_STRATEGY.rsi_upper_bound)
    )
    usable = [int(i) for i in np.flatnonzero(crossed.to_numpy()) if i >= min_bars]
    assert usable, f"seed {seed} has no usable cross-up; pick another seed"
    return closes[: usable[0] + 1]


def make_bars(now: datetime | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(15m, 1h) frames indexed by UTC timestamp. The last 15m bar is the
    cross-up bar and it closed 30 minutes before `now`, so it is the last
    closed bar however close `now` is to a bar boundary. The 1h trend is up."""
    now = now or datetime.now(timezone.utc)
    last_15m_start = pd.Timestamp(now).floor("15min") - pd.Timedelta(minutes=30)

    closes_15m = cross_up_closes()
    idx_15m = pd.date_range(end=last_15m_start, periods=len(closes_15m), freq="15min", tz="UTC")

    last_15m_close = last_15m_start + pd.Timedelta(minutes=15)
    last_1h_start = last_15m_close.floor("h") - pd.Timedelta(hours=1)
    n_1h = 120
    closes_1h = 100 + 0.5 * np.arange(n_1h)
    idx_1h = pd.date_range(end=last_1h_start, periods=n_1h, freq="1h", tz="UTC")

    return _ohlcv(closes_15m, idx_15m), _ohlcv(closes_1h, idx_1h)


def with_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    """The shape the unit tests use: a `timestamp` column AND a datetime index.
    The index is unnamed so pandas does not see an ambiguous name."""
    out = frame.copy()
    out["timestamp"] = out.index
    out.index.name = None
    return out


def alpaca_shaped(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """The shape of alpaca-py's `BarSet.df`: MultiIndex (symbol, timestamp),
    no `timestamp` column."""
    out = frame.copy()
    out.index = pd.MultiIndex.from_product([[symbol], out.index], names=["symbol", "timestamp"])
    return out
