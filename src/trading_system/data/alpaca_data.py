import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Sequence

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from pydantic import SecretStr

from trading_system.common.retry import retry_read_request
from trading_system.config import Settings
from trading_system.data.bars import normalize_bars
from trading_system.data.live_price import LivePrice

logger = logging.getLogger(__name__)

# alpaca-py 0.36 to 0.44 has no TimeFrame.__mul__, so `TimeFrame.Minute * 15` raises
# TypeError. Build the 15-minute frame explicitly.
FIFTEEN_MINUTES = TimeFrame(15, TimeFrameUnit.Minute)
ONE_HOUR = TimeFrame(1, TimeFrameUnit.Hour)

_OHLCV = ("open", "high", "low", "close", "volume")


class AlpacaData:
    """
    Market Data Adapter using Alpaca's StockHistoricalDataClient.

    If configured to use 'iex' feed, note that IEX data limitations apply (less volume,
    potentially missing trades that occurred on other exchanges). Use 'sip' for full market data.

    Every frame it returns is in the canonical shape of data.bars.normalize_bars.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        api_key = (
            settings.alpaca_api_key.get_secret_value()
            if isinstance(settings.alpaca_api_key, SecretStr)
            else settings.alpaca_api_key
        )
        secret_key = (
            settings.alpaca_secret_key.get_secret_value()
            if isinstance(settings.alpaca_secret_key, SecretStr)
            else settings.alpaca_secret_key
        )

        self.client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.feed = settings.data_feed.lower()  # 'iex' or 'sip'

    @retry_read_request()
    def fetch_bars_bulk(
        self, symbols: Sequence[str], timeframe: TimeFrame, start: datetime, end: datetime
    ) -> dict[str, pd.DataFrame]:
        """Bars for all `symbols` in ONE request (alpaca-py pages through the results).

        Symbols without any bars are left out of the result. Prices are adjusted for
        splits and dividends so that a split inside the lookback window cannot look like
        a crash to the indicators."""
        if not symbols:
            return {}

        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed(self.feed),
            adjustment=Adjustment.ALL,
        )
        # get_stock_bars returns a BarSet (or a raw dict only if raw_data=True, which we never set).
        frame = getattr(self.client.get_stock_bars(request), "df", None)
        if frame is None or frame.empty:
            return {}

        if not isinstance(frame.index, pd.MultiIndex):
            # Not what alpaca-py returns for a symbol list, but harmless to accept for one symbol.
            frame = pd.concat({symbols[0]: frame}, names=["symbol"])

        result: dict[str, pd.DataFrame] = {}
        for symbol, bars in frame.groupby(level="symbol"):
            missing = [c for c in _OHLCV if c not in bars.columns]
            if missing:
                raise ValueError(f"bars for {symbol} lack columns {missing}")
            result[str(symbol)] = normalize_bars(bars)
        return result

    @retry_read_request()
    def fetch_latest_prices(self, symbols: Sequence[str]) -> dict[str, LivePrice]:
        """The last trade of every symbol in ONE request. Symbols that have not traded are
        left out. The caller decides whether a price is fresh enough (data.live_price)."""
        if not symbols:
            return {}

        request = StockLatestTradeRequest(symbol_or_symbols=list(symbols), feed=DataFeed(self.feed))
        # get_stock_latest_trade returns {symbol: Trade} (a raw dict only if raw_data=True).
        trades = self.client.get_stock_latest_trade(request)
        return {
            str(symbol): LivePrice(price=Decimal(str(trade.price)), timestamp=trade.timestamp)
            for symbol, trade in trades.items()
        }

    def fetch_bars(
        self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Bars for one symbol (an empty frame if there are none)."""
        return self.fetch_bars_bulk([symbol], timeframe, start, end).get(symbol, pd.DataFrame())

    def load_universe(
        self,
        symbols: Sequence[str],
        now: datetime,
        lookback_days: int,
        htf_lookback_days: int,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """15-minute and 1-hour bars for every symbol, cleaned, cut to closed bars and
        aligned, ready for strategy.trend.generate_signals. Two requests in total.

        Symbols that lack bars on either timeframe are skipped (logged, not fatal)."""
        bars_15m = self.fetch_bars_bulk(
            symbols, FIFTEEN_MINUTES, now - timedelta(days=lookback_days), now
        )
        bars_1h = self.fetch_bars_bulk(
            symbols, ONE_HOUR, now - timedelta(days=htf_lookback_days), now
        )

        out_15m: dict[str, pd.DataFrame] = {}
        out_1h: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df_15m, df_1h = bars_15m.get(symbol), bars_1h.get(symbol)
            if df_15m is None or df_1h is None:
                logger.warning(
                    "No bars returned for %s (15m: %s, 1h: %s)",
                    symbol,
                    df_15m is not None,
                    df_1h is not None,
                )
                continue
            df_15m, df_1h = self.align_and_filter(
                self.clean_and_deduplicate(df_15m), self.clean_and_deduplicate(df_1h), now
            )
            if not df_15m.empty and not df_1h.empty:
                out_15m[symbol] = df_15m
                out_1h[symbol] = df_1h
        return out_15m, out_1h

    def clean_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures UTC timestamps, sorts, and removes duplicates.
        """
        if df.empty:
            return df

        # Ensure UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")

        # Deduplicate and sort
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        return df

    def align_and_filter(
        self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, current_time: datetime
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
