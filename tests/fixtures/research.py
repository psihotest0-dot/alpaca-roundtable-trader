"""Made-up daily prices for the research tests: random walks whose truth is known, so a test can
plant an effect and see the measurement find it, or plant none and see it find nothing."""

from typing import Optional

import numpy as np
import pandas as pd

from trading_system.data.universe import MARKET_SYMBOL, research_universe


def stock_names(count: int) -> list[str]:
    """Real tickers of the research universe (so the CLI accepts them), without the market."""
    return [s for s in research_universe() if s != MARKET_SYMBOL][:count]


def noise_returns(
    days: int = 600,
    stocks: int = 60,
    *,
    seed: int = 1,
    vol: float = 0.01,
    end: str = "2026-06-30",
) -> pd.DataFrame:
    """Daily log returns, dates down and the stocks plus SPY across; no effect anywhere."""
    rng = np.random.default_rng(seed)
    columns = [*stock_names(stocks), MARKET_SYMBOL]
    index = pd.bdate_range(end=end, periods=days)
    return pd.DataFrame(rng.normal(0.0, vol, (days, len(columns))), index=index, columns=columns)


def frames_from_returns(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """`open` and `close` per symbol for panel.build_panel. Every day opens where the last one
    closed (no gap), so a day's own return happens between its open and its close."""
    closes = 100.0 * np.exp(returns.cumsum())
    opens = closes.shift(1).fillna(100.0)
    return {
        str(symbol): pd.DataFrame({"open": opens[symbol], "close": closes[symbol]})
        for symbol in returns.columns
    }


def stored_bars(returns: pd.DataFrame, only: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
    """The same prices in the shape the candle store takes: UTC stamps at 05:00 (the start of a
    New York winter day), with high, low and volume."""
    out: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames_from_returns(returns).items():
        if only is not None and symbol not in only:
            continue
        stamped = frame.copy()
        stamped.index = (
            pd.DatetimeIndex(stamped.index).tz_localize("UTC") + pd.Timedelta(hours=5)
        ).rename("timestamp")
        stamped["high"] = stamped[["open", "close"]].max(axis=1) * 1.001
        stamped["low"] = stamped[["open", "close"]].min(axis=1) * 0.999
        stamped["volume"] = 1_000_000.0
        out[symbol] = stamped
    return out
