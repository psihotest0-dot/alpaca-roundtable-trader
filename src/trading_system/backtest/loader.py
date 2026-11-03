"""The stored candles as the backtest wants them."""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pandas as pd

from trading_system.config import Settings
from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.repositories import MarketBarRepository


class NoStoredCandles(RuntimeError):
    """The database has no candles for the watchlist in the window."""


def load_candles(
    settings: Settings, days: int, now: Optional[datetime] = None
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """(15-minute frames, 1-hour frames) of the last `days` days, for the watchlist symbols that
    have both."""
    symbols = settings.watchlist_symbols
    since = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    repo = MarketBarRepository(get_session_factory(get_engine(settings.database_url)))
    df_15m = repo.load(symbols, "15m", start=since)
    df_1h = repo.load(symbols, "1h", start=since)
    usable = [s for s in symbols if s in df_15m and s in df_1h]
    if not usable:
        raise NoStoredCandles(
            f"No stored candles for the watchlist in the last {days} days. "
            "Run `fetch-history` first."
        )
    return {s: df_15m[s] for s in usable}, {s: df_1h[s] for s in usable}
