"""What the Oracle is told about a signal (Phase 2.1b).

The Oracle used to receive the raw OHLCV of the last bar and a fixed sentence, which is not
an independent opinion on anything. It now gets what a trader would look at: the indicators
the strategy used (and ATR, which it does not), the last few bars, volume against the norm, the
price right now, the hourly trend, and where the account stands against its limits.
"""

import math
from decimal import Decimal
from typing import Any, Optional, Sequence

import pandas as pd

from trading_system.data.bars import normalize_bars
from trading_system.domain.models import StrategySignal
from trading_system.oracle.contract import OracleNews, OracleRequest
from trading_system.strategy.config import StrategyConfig
from trading_system.strategy.trend import calculate_indicators

RECENT_BARS = 12  # three hours of 15-minute bars
VOLUME_AVERAGE_BARS = 20  # the norm that the last bar's volume is compared with


def _number(value: Any, digits: int = 4) -> Optional[float]:
    """A plain float, or None for a missing or non-finite value (JSON has no NaN)."""
    if value is None:
        return None
    number = float(value)
    return round(number, digits) if math.isfinite(number) else None


def build_oracle_request(
    signal: StrategySignal,
    bars_15m: pd.DataFrame,
    bars_1h: pd.DataFrame,
    config: StrategyConfig,
    *,
    live_price: Decimal,
    context: dict[str, Any],
    news: Sequence[OracleNews] = (),
) -> OracleRequest:
    m15 = calculate_indicators(normalize_bars(bars_15m), config)
    h1 = calculate_indicators(normalize_bars(bars_1h), config, is_htf=True)

    last = m15.iloc[-1]
    previous = m15.iloc[-2] if len(m15) > 1 else last
    close = float(last["close"])
    technicals: dict[str, Any] = {
        "close": _number(close),
        "ema_fast": _number(last["ema_fast"]),
        "ema_slow": _number(last["ema_slow"]),
        "ema_fast_previous": _number(previous["ema_fast"]),
        "ema_slow_previous": _number(previous["ema_slow"]),
        "rsi": _number(last["rsi"], 2),
        "atr": _number(last["atr"]),
        "atr_pct_of_close": _number(float(last["atr"]) / close * 100, 3) if close else None,
        "live_price_vs_close_pct": _number((float(live_price) - close) / close * 100, 3)
        if close
        else None,
    }

    # The hourly bar the strategy itself would use: the latest one closed by the time the
    # 15-minute bar closed.
    bar_close = last["timestamp"] + pd.Timedelta(minutes=15)
    closed_hours = h1[h1["timestamp"] + pd.Timedelta(hours=1) <= bar_close]
    if not closed_hours.empty:
        hour = closed_hours.iloc[-1]
        technicals["hourly_ema_fast"] = _number(hour["ema_fast"])
        technicals["hourly_ema_slow"] = _number(hour["ema_slow"])
        technicals["hourly_trend_up"] = bool(hour["ema_fast"] > hour["ema_slow"])

    recent = m15.tail(RECENT_BARS)
    recent_bars = [
        {
            "start": row["timestamp"].isoformat(),
            "open": _number(row["open"]),
            "high": _number(row["high"]),
            "low": _number(row["low"]),
            "close": _number(row["close"]),
            "volume": _number(row["volume"], 0),
        }
        for _, row in recent.iterrows()
    ]

    norm = m15["volume"].iloc[:-1].tail(VOLUME_AVERAGE_BARS)
    volume_ratio = None
    if len(norm) >= 3 and norm.mean() > 0:
        volume_ratio = _number(float(last["volume"]) / float(norm.mean()), 2)

    return OracleRequest(
        signal=signal,
        rationale="15-minute EMA cross up with RSI in range and the hourly trend up.",
        technicals=technicals,
        recent_bars=recent_bars,
        volume_ratio=volume_ratio,
        live_price=live_price,
        context=context,
        news=list(news),
    )
