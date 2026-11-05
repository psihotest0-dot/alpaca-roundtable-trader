"""Deterministic OHLCV fixtures shaped like what Alpaca really returns.

The unit tests feed the strategy frames with a `timestamp` *column*. Alpaca's
`BarSet.df` has a (symbol, timestamp) MultiIndex and no such column, and that
mismatch went unnoticed. These helpers build both shapes from one series.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trading_system.strategy.config import RUNTIME_STRATEGY
from trading_system.strategy.trend import calculate_indicators


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
    """(15m, 1h) frames indexed by UTC timestamp. The last 15m bar is the cross-up bar and
    it is the bar that closed at the latest 15-minute boundary, i.e. exactly the bar a run
    at `now` is expected to see (runtime.schedule.expected_bar_start). The 1h trend is up."""
    now = now or datetime.now(timezone.utc)
    last_15m_start = pd.Timestamp(now).floor("15min") - pd.Timedelta(minutes=15)

    closes_15m = cross_up_closes()
    idx_15m = pd.date_range(end=last_15m_start, periods=len(closes_15m), freq="15min", tz="UTC")

    last_15m_close = last_15m_start + pd.Timedelta(minutes=15)
    last_1h_start = last_15m_close.floor("h") - pd.Timedelta(hours=1)
    n_1h = 120
    closes_1h = 100 + 0.5 * np.arange(n_1h)
    idx_1h = pd.date_range(end=last_1h_start, periods=n_1h, freq="1h", tz="UTC")

    return _ohlcv(closes_15m, idx_15m), _ohlcv(closes_1h, idx_1h)


def daily_bars(last_start: datetime, periods: int, base: float = 100.0) -> pd.DataFrame:
    """`periods` daily bars of rising closes, the newest starting at `last_start`. Alpaca stamps a
    daily bar with the start of its New York day (05:00 UTC in winter), so pass such a time."""
    closes = base + np.arange(periods) * 0.1
    index = pd.date_range(end=last_start, periods=periods, freq="D", tz="UTC")
    return _ohlcv(closes, index)


def with_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    """The shape the unit tests use: a `timestamp` column AND a datetime index.
    The index is unnamed so pandas does not see an ambiguous name."""
    out = frame.copy()
    out["timestamp"] = out.index
    out.index.name = None
    return out


QUIET = (1.0, 1.001, 0.999, 1.0)  # (open, high, low, close) as multiples of the cross-up close
GAP_UP_6_PCT = (1.06, 1.07, 1.05, 1.06)  # opens 6 % up: past the 5 % take-profit


def make_bars_with_follow_up(
    now: datetime, follow: list[tuple[float, float, float, float]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """The cross-up fixture (`make_bars`) followed by bars shaped by `follow`, each row
    (open, high, low, close) as a multiple of the cross-up bar's close. Both frames come with
    the `timestamp` column the engines read. Returns the 15m frame, the 1h frame and the start
    times of the follow-up bars; the first of them is the bar in which the signal is acted on."""
    bars_15m, bars_1h = make_bars(now)
    cross = float(bars_15m["close"].iloc[-1])
    rows = np.array(follow, dtype=float) * cross
    extra = pd.DataFrame(
        {
            "open": rows[:, 0],
            "high": rows[:, 1],
            "low": rows[:, 2],
            "close": rows[:, 3],
            "volume": 1000.0,
        },
        index=pd.date_range(
            bars_15m.index[-1] + pd.Timedelta(minutes=15),
            periods=len(follow),
            freq="15min",
            tz="UTC",
        ),
    )
    return (
        with_timestamp_column(pd.concat([bars_15m, extra])),
        with_timestamp_column(bars_1h),
        extra.index,
    )


def random_walk_session_bars(
    seed: int, first_day: str, days: int, drift: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(15m, 1h) frames with a `timestamp` column for `days` weekdays from `first_day`: the
    regular session of a New York winter day (14:30-21:00 UTC), 26 bars a day, a random walk
    with per-bar noise of 0.4 % and a per-bar `drift`. Crossings of the EMAs are frequent, so
    the strategy has something to find. The hourly bars are made from the 15-minute ones."""
    rng = np.random.default_rng(seed)
    sessions = [
        pd.date_range(day + pd.Timedelta(hours=14, minutes=30), periods=26, freq="15min", tz="UTC")
        for day in pd.bdate_range(first_day, periods=days)
    ]
    index = sessions[0].append(sessions[1:]) if len(sessions) > 1 else sessions[0]
    closes = 100.0 * np.cumprod(1.0 + rng.normal(drift, 0.004, len(index)))
    opens = np.concatenate([[100.0], closes[:-1]])
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) * 1.001,
            "low": np.minimum(opens, closes) * 0.999,
            "close": closes,
            "volume": rng.uniform(300.0, 1700.0, len(index)),
        },
        index=index,
    )
    hourly = frame.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    hourly = hourly.dropna()
    return with_timestamp_column(frame), with_timestamp_column(hourly)


def alpaca_shaped(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """The shape of alpaca-py's `BarSet.df`: MultiIndex (symbol, timestamp),
    no `timestamp` column."""
    out = frame.copy()
    out.index = pd.MultiIndex.from_product([[symbol], out.index], names=["symbol", "timestamp"])
    return out
