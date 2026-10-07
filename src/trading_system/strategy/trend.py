import pandas as pd
from typing import Dict, List
from datetime import datetime

from trading_system.strategy.config import StrategyConfig
from trading_system.domain.models import StrategySignal
from trading_system.domain.enums import SignalDirection

def calculate_indicators(df: pd.DataFrame, config: StrategyConfig, is_htf: bool = False) -> pd.DataFrame:
    """
    Pure function to calculate indicators for a dataframe of OHLCV bars.
    Returns a copy of the dataframe with added indicator columns.

    Expected df columns: ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    Assumes df is sorted by timestamp ascending.
    """
    df = df.copy()

    if df.empty:
        return df

    fast_period = config.htf_fast_ema_period if is_htf else config.fast_ema_period
    slow_period = config.htf_slow_ema_period if is_htf else config.slow_ema_period

    # EMA calculation (Standard EMA)
    df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()

    if not is_htf:
        # RSI calculation (Wilder's Smoothing)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)

        # Wilder's smoothing is equivalent to exponential moving average with alpha = 1/period
        avg_gain = gain.ewm(alpha=1/config.rsi_period, min_periods=config.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/config.rsi_period, min_periods=config.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(100) # Fallback for avg_loss == 0

        # ATR calculation (Wilder's Smoothing)
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/config.atr_period, min_periods=config.atr_period, adjust=False).mean()

    return df

def generate_signals(
    data_15m: Dict[str, pd.DataFrame],
    data_1h: Dict[str, pd.DataFrame],
    config: StrategyConfig,
    eval_timestamp: datetime
) -> List[StrategySignal]:
    """
    Pure function to generate strategy signals.
    eval_timestamp: The time the evaluation is taking place. Used to determine if a bar is fully closed.

    Criteria:
    1. 15m EMA fast crosses above EMA slow on the last CLOSED 15m bar.
    2. 15m RSI between lower and upper bounds on the last closed 15m bar.
    3. 1h EMA fast > 1h EMA slow on the MOST RECENT COMPLETED hourly bar prior to or equal to the 15m bar timestamp.
    4. Sufficient warmup bars (min_warmup_bars).
    """
    signals = []

    for symbol, df_15 in data_15m.items():
        if df_15.empty or len(df_15) < config.min_warmup_bars:
            continue

        if symbol not in data_1h:
            continue

        df_1h = data_1h[symbol]
        if df_1h.empty or len(df_1h) < config.min_warmup_bars:
            continue

        # Calculate indicators
        df_15 = calculate_indicators(df_15, config, is_htf=False)
        df_1h = calculate_indicators(df_1h, config, is_htf=True)

        # Ensure index is datetime for alignment/filtering if not already
        if not pd.api.types.is_datetime64_any_dtype(df_15['timestamp']):
            df_15['timestamp'] = pd.to_datetime(df_15['timestamp'], utc=True)
        if not pd.api.types.is_datetime64_any_dtype(df_1h['timestamp']):
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], utc=True)

        # Only use closed bars
        # A bar is considered closed if its start time + its timeframe < eval_timestamp
        df_15_closed = df_15[df_15['timestamp'] + pd.Timedelta(minutes=15) <= eval_timestamp]
        df_1h_closed = df_1h[df_1h['timestamp'] + pd.Timedelta(hours=1) <= eval_timestamp]

        if len(df_15_closed) < 2 or df_1h_closed.empty:
            continue

        # Get the last two closed 15m bars
        last_bar = df_15_closed.iloc[-1]
        prev_bar = df_15_closed.iloc[-2]

        last_time = last_bar['timestamp']

        # Check cross up condition on 15m
        prev_fast = prev_bar['ema_fast']
        prev_slow = prev_bar['ema_slow']
        last_fast = last_bar['ema_fast']
        last_slow = last_bar['ema_slow']

        if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(last_fast) or pd.isna(last_slow):
            continue

        cross_up = (prev_fast <= prev_slow) and (last_fast > last_slow)
        if not cross_up:
            continue

        # Check RSI condition on 15m
        rsi = last_bar['rsi']
        if pd.isna(rsi) or rsi < config.rsi_lower_bound or rsi > config.rsi_upper_bound:
            continue

        # Check 1h condition
        # We need the most recent completed 1h bar strictly before or exactly at the 15m bar timestamp
        # to ensure no lookahead. Actually, temporal alignment: the 1h bar timestamp must be <= last_time.
        # But wait, 1h bar starts at e.g., 10:00 and finishes at 11:00. The 15m bar finishes at 11:00 (starts 10:45).
        # We must align them so that we only look at 1h bars that closed *before or at* the 15m bar's close.
        # last_bar['timestamp'] is the open time.
        last_bar_close_time = last_time + pd.Timedelta(minutes=15)

        # 1h bars that closed before or exactly at the 15m bar's close time
        valid_1h_bars = df_1h_closed[df_1h_closed['timestamp'] + pd.Timedelta(hours=1) <= last_bar_close_time]
        if valid_1h_bars.empty:
            continue

        last_1h = valid_1h_bars.iloc[-1]
        htf_fast = last_1h['ema_fast']
        htf_slow = last_1h['ema_slow']

        if pd.isna(htf_fast) or pd.isna(htf_slow):
            continue

        if not (htf_fast > htf_slow):
            continue

        # Signal Generated
        signal = StrategySignal(
            version=config.strategy_version,
            symbol=symbol,
            timeframe="15m",
            closed_bar_timestamp=last_time, # Storing open time of closed bar as convention or close time
            direction=SignalDirection.LONG,
            expires_at=eval_timestamp + pd.Timedelta(minutes=15),
            confidence=1.0
        )
        signals.append(signal)

    return signals

def filter_signals(signals: List[StrategySignal], active_symbols: set[str]) -> List[StrategySignal]:
    """
    Filter out signals for symbols that already have an active position or active entry order.
    `active_symbols` should contain symbols that have an open position or pending order.
    """
    return [sig for sig in signals if sig.symbol not in active_symbols]
