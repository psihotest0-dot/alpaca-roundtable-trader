"""Read-only, run INSIDE the trader container (it reads var/observe.db, in the Docker volume):
`docker compose exec -T trader python - < scripts/atr_measurement/atr_observed.py`
ATR(14) of the 15-minute bars of the observed universe (docs/strategy-research-plan.md, section 13)."""

import numpy as np
import pandas as pd
from trading_system.config import Settings
from trading_system.data.universe import observed_symbols
from trading_system.storage.research import open_observe_store
from trading_system.strategy.config import RUNTIME_STRATEGY
from trading_system.strategy.session_cross import session_mask_15m
from trading_system.strategy.trend import calculate_indicators

s = Settings()
store = open_observe_store(s.observe_database_url, s.database_url)
obs = observed_symbols(s.watchlist_symbols)
bars = store.load(obs, "15m")
med, sig = [], []
for sym, df in bars.items():
    df = (
        df.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in df.columns
        else df
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    ind = calculate_indicators(df, RUNTIME_STRATEGY)
    med.append(float(np.nanmedian((ind["atr"] / ind["close"] * 100).to_numpy())))
    m = session_mask_15m(ts)
    c = df[m].assign(day=ts[m].tz_convert("America/New_York").date).groupby("day")["close"].last()
    sig.append(float(np.log(c).diff().dropna().std() * 100))
med, sig = np.array(med), np.array(sig)


def q(v, pts=(0, 10, 25, 50, 75, 90, 100)):
    return "  ".join(f"p{p} {np.percentile(v, p):.2f}" for p in pts)


print(f"{len(med)} observed symbols with bars")
print("per-symbol median ATR %:   ", q(med))
print("stop 4 % in ATRs:          ", q(4 / med))
print("target 5 % in ATRs:        ", q(5 / med))
print("daily volatility %:        ", q(sig))
print("days for 1 sd to be 4 %:   ", q((4 / sig) ** 2))
