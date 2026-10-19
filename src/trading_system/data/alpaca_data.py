import functools
import logging
import random
import time
from datetime import datetime
import pandas as pd
from typing import Any, Callable, TypeVar

from alpaca.data.historical import StockHistoricalDataClient
from pydantic import SecretStr

from trading_system.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

def retry_read_request(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            while True:
                try:
                    logger.info(f"Initiating Alpaca data read request: {func.__name__}")
                    result = func(*args, **kwargs)
                    logger.info(f"Successfully completed Alpaca data read request: {func.__name__}")
                    return result
                except Exception as e:
                    if retries >= max_retries:
                        logger.error(f"Failed {func.__name__} after {max_retries} retries: {e}")
                        raise
                    delay = min(base_delay * (2 ** retries), max_delay)
                    delay = delay * random.uniform(0.5, 1.5)
                    logger.warning(
                        f"Exception in {func.__name__} (attempt {retries + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    retries += 1
        return wrapper
    return decorator


class AlpacaData:
    """
    Market Data Adapter using Alpaca's StockHistoricalDataClient.

    If configured to use 'iex' feed, note that IEX data limitations apply (less volume,
    potentially missing trades that occurred on other exchanges). Use 'sip' for full market data.
    """
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        api_key = settings.alpaca_api_key.get_secret_value() if isinstance(settings.alpaca_api_key, SecretStr) else settings.alpaca_api_key
        secret_key = settings.alpaca_secret_key.get_secret_value() if isinstance(settings.alpaca_secret_key, SecretStr) else settings.alpaca_secret_key

        self.client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )
        self.feed = settings.data_feed.lower() # 'iex' or 'sip'


    @retry_read_request()
    def fetch_bars(
        self,
        symbol: str,
        timeframe: Any, # alpaca.data.timeframe.TimeFrame
        start: "datetime",
        end: "datetime"
    ) -> "pd.DataFrame":
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import DataFeed

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed(self.feed) if self.feed else None
        )

        bars = self.client.get_stock_bars(req)
        if not hasattr(bars, 'data') or not bars.data or symbol not in bars.data:
            import pandas as pd
            return pd.DataFrame()

        df = getattr(bars, 'df', pd.DataFrame())
        if df.empty:
            import pandas as pd
            return pd.DataFrame()

        # Alpaca returns multi-index (symbol, timestamp) for df. We drop symbol.
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)

        return df

    def clean_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures UTC timestamps, sorts, and removes duplicates.
        """
        if df.empty:
            return df

        # Ensure UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif str(df.index.tz) != 'UTC':
            df.index = df.index.tz_convert('UTC')

        # Deduplicate and sort
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        return df

    def align_and_filter(
        self,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        current_time: datetime
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies business logic:
        1. Keep only closed bars (if timestamp is the start of the bar,
           we assume bar is closed if current_time >= timestamp + timeframe).
        2. Align 15m and 1h without lookahead.
        3. Drop incomplete hourly bars.
        4. Detect missing/stale data (do not forward fill).
        """
        # Ensure UTC
        if current_time.tzinfo is None:
            from datetime import timezone
            current_time = current_time.replace(tzinfo=timezone.utc)

        # 1. Filter out unclosed bars
        # Alpaca timestamps are start of bar.
        # A 15m bar starting at 09:30 is closed at 09:45.
        if not df_15m.empty:
            df_15m = df_15m[df_15m.index + pd.Timedelta(minutes=15) <= current_time]
        if not df_1h.empty:
            df_1h = df_1h[df_1h.index + pd.Timedelta(hours=1) <= current_time]

        # If either is empty after filtering, return empty
        if df_15m.empty or df_1h.empty:
            return pd.DataFrame(), pd.DataFrame()

        # 2. Alignment
        # For a 15m bar at T, we need the latest 1h bar closed <= T.
        # So for 15m bar starting 10:15 (closes 10:30), the 1h bar starting 09:30 (closes 10:30) is available.
        # But for 15m bar starting 10:00 (closes 10:15), the 1h bar starting 09:30 is NOT YET CLOSED.
        # We must map the 15m bar to the LATEST fully closed 1h bar.

        # We can construct a 'latest_closed_1h' for each 15m bar.
        # A 1h bar at t_1h closes at t_1h + 1h.
        # A 15m bar at t_15m closes at t_15m + 15m.
        # We need t_1h + 1h <= t_15m + 15m.

        aligned_15m_idx = []
        for t_15m in df_15m.index:
            close_15m = t_15m + pd.Timedelta(minutes=15)
            # Find 1h bars that closed before or exactly at the 15m bar's close
            valid_1h = df_1h[df_1h.index + pd.Timedelta(hours=1) <= close_15m]
            if not valid_1h.empty:
                aligned_15m_idx.append(t_15m)

        df_15m_aligned = df_15m.loc[aligned_15m_idx]

        # Detect missing data (e.g. gaps in 15m or 1h). We do NOT forward fill.
        # We just return the sparse data, the strategy must handle it.

        return df_15m_aligned, df_1h
