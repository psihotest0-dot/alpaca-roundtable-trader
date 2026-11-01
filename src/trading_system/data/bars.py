"""The one shape bars have inside the system.

Two shapes were in use and nobody noticed. alpaca-py returns a (symbol, timestamp)
MultiIndex with no `timestamp` column; the strategy and the backtest read a `timestamp`
column. The unit tests only ever used the column, so the live path crashed with
KeyError('timestamp'). normalize_bars() is the single place that converts.
"""

import pandas as pd


class BarsContractError(ValueError):
    """The bars cannot be brought into the canonical shape."""


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` in the canonical shape:

    - a tz-aware UTC DatetimeIndex, sorted ascending, one row per instant (the last wins);
    - the same instants in a `timestamp` column. The index has no name, so pandas never
      sees an ambiguous `timestamp` and both `df["timestamp"]` and `df.index` work.

    Accepts alpaca-py's (symbol, timestamp) MultiIndex, a plain DatetimeIndex, or any
    index with a `timestamp` column (the shape the backtest produces). Naive times are
    taken as UTC. An empty frame is returned unchanged. The input is not modified.
    """
    if df.empty:
        return df

    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        symbol_level = "symbol" if "symbol" in out.index.names else 0
        out = out.droplevel(symbol_level)

    if "timestamp" in out.columns:
        instants = pd.DatetimeIndex(pd.to_datetime(out["timestamp"], utc=True))
        out = out.drop(columns="timestamp")
    elif isinstance(out.index, pd.DatetimeIndex):
        instants = out.index
        instants = (
            instants.tz_localize("UTC") if instants.tz is None else instants.tz_convert("UTC")
        )
    else:
        raise BarsContractError(
            "bars need a DatetimeIndex or a 'timestamp' column, got index "
            f"{type(out.index).__name__} and columns {list(out.columns)}"
        )

    out.index = instants
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["timestamp"] = out.index
    out.index.name = None
    return out
