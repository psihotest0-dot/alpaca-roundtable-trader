"""Descriptive measurement for the ATR question (docs/strategy-research-plan.md, section 13). READ-ONLY.
Run with the repository's Python: `python scripts/atr_measurement/atr_profile.py` (needs a backup in var/backups).

Works on a gunzipped COPY of the newest backup (never the live database) and on the stored candles of the
22 watchlist symbols. It tests nothing and tunes nothing: no stop or target other than the live 4 % / 5 % is
tried anywhere, and no profit is split by ATR.

A. ATR(14) of the 15-minute bars as % of the close, as the live code computes it (all stored bars, so the
   overnight gap and the thin extended-hours prints are inside it) and on regular-session bars only; the
   live stop (4 %) and target (5 %) expressed in ATRs; the symbols' daily volatility and the time a random
   walk needs to travel 4 % / 5 %.
B. The live strategy replayed by the real backtest engine over the same candles: how the trades close (stop,
   target, still open), how long they live, how many sleep overnight, the stop distance in ATRs at entry.
"""

import gzip
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
newest = sorted((ROOT / "var" / "backups").glob("trading-2*.db.gz"))[-1]
copy = Path(tempfile.mkdtemp()) / "atr_copy.db"
with gzip.open(newest, "rb") as f, open(copy, "wb") as o:
    shutil.copyfileobj(f, o)
os.environ["DATABASE_URL"] = f"sqlite:///{copy.as_posix()}"
os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")

from trading_system.backtest.engine import BacktestEngine  # noqa: E402
from trading_system.backtest.loader import load_candles  # noqa: E402
from trading_system.backtest.models import BacktestConfig  # noqa: E402
from trading_system.config import Settings  # noqa: E402
from trading_system.strategy.config import RUNTIME_STRATEGY  # noqa: E402
from trading_system.strategy.registry import get_strategy  # noqa: E402
from trading_system.strategy.session_cross import session_mask_15m  # noqa: E402
from trading_system.strategy.trend import calculate_indicators  # noqa: E402

DAYS = 180
STOP, TARGET = 4.0, 5.0  # the live percentages
NY = "America/New_York"

settings = Settings()
df_15m, df_1h = load_candles(settings, DAYS)
symbols = sorted(df_15m)
first = min(df["timestamp"].iloc[0] for df in df_15m.values())
last = max(df["timestamp"].iloc[-1] for df in df_15m.values())
print(f"backup: {newest.name}; {len(symbols)} symbols; candles {first:%Y-%m-%d} to {last:%Y-%m-%d}")


def q(values, points=(10, 25, 50, 75, 90)) -> str:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return "n/a"
    return "  ".join(f"p{p} {np.percentile(v, p):.2f}" for p in points)


# ------------------------------------------------------------------------------------------ A
print("\n=== A. ATR(14) of the 15-minute bars, % of the close ===")
pooled_full, pooled_session = [], []
per_symbol = {}
daily_sigma = {}
for sym in symbols:
    df = df_15m[sym].sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    full = calculate_indicators(df, RUNTIME_STRATEGY)
    full_pct = (full["atr"] / full["close"] * 100).to_numpy()
    mask = session_mask_15m(ts)
    sess = calculate_indicators(df[mask].reset_index(drop=True), RUNTIME_STRATEGY)
    sess_pct = (sess["atr"] / sess["close"] * 100).to_numpy()
    pooled_full += list(full_pct[np.isfinite(full_pct)])
    pooled_session += list(sess_pct[np.isfinite(sess_pct)])
    per_symbol[sym] = (np.nanmedian(full_pct), np.nanmedian(sess_pct))
    # daily volatility from the last regular-session close of each day
    c = df[mask].assign(day=ts[mask].tz_convert(NY).date).groupby("day")["close"].last()
    daily_sigma[sym] = float(np.log(c).diff().dropna().std() * 100)

print("pooled bars, ATR % (live computation, all stored bars): ", q(pooled_full))
print("pooled bars, ATR % (regular-session bars only):         ", q(pooled_session))
med_full = np.array([v[0] for v in per_symbol.values()])
med_sess = np.array([v[1] for v in per_symbol.values()])
print(
    "per-symbol median ATR %, all bars (across symbols):      ", q(med_full, (0, 25, 50, 75, 100))
)
print(
    "per-symbol median ATR %, session bars (across symbols):  ", q(med_sess, (0, 25, 50, 75, 100))
)
print("\nthe live stop (4 %) and target (5 %) in ATRs, per-symbol median ATR (live computation):")
print("  stop   = 4 / ATR%:", q(STOP / med_full, (0, 25, 50, 75, 100)))
print("  target = 5 / ATR%:", q(TARGET / med_full, (0, 25, 50, 75, 100)))
sig = np.array(list(daily_sigma.values()))
print(
    "\ndaily volatility of the symbols (std of daily log return, %):", q(sig, (0, 25, 50, 75, 100))
)
days4 = (STOP / sig) ** 2
days5 = (TARGET / sig) ** 2
print("trading days a random walk needs for one standard deviation to equal 4 % / 5 %:")
print("  4 %:", q(days4, (0, 25, 50, 75, 100)))
print("  5 %:", q(days5, (0, 25, 50, 75, 100)))

# ------------------------------------------------------------------------------------------ B
print("\n=== B. How the live strategy's trades close in the real backtest engine ===")
engine = BacktestEngine(
    settings, BacktestConfig(), df_15m, df_1h, strategy=get_strategy("trend_cross")
)
result = engine.run()
trades = result.trades
open_positions = dict(engine.positions)
print(
    f"signals {result.signals}; trades {len(trades)} closed + {len(open_positions)} still open at the "
    f"end of the data; mean closed trade {np.mean([t.pnl_pct for t in trades]) * 100:+.2f}%"
)
by_reason = pd.Series([t.exit_reason for t in trades]).value_counts()
print("closed by:", dict(by_reason))

hold_h = np.array([(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades])
entry_day = [pd.Timestamp(t.entry_time).tz_convert(NY).date() for t in trades]
exit_day = [pd.Timestamp(t.exit_time).tz_convert(NY).date() for t in trades]
same_day = np.array([a == b for a, b in zip(entry_day, exit_day)])
span = np.array([np.busday_count(a, b) for a, b in zip(entry_day, exit_day)])
print(f"calendar hours held (closed trades): {q(hold_h)}")
print(f"trading days from entry to exit:      {q(span)}")
print(
    f"closed the same New York day: {same_day.mean() * 100:.0f}%   slept at least one night: "
    f"{(~same_day).mean() * 100:.0f}%   five or more trading days: {(span >= 5).mean() * 100:.0f}%"
)

end_ts = {s: df_15m[s]["timestamp"].iloc[-1] for s in symbols}
open_age_days, open_unreal = [], []
for sym, pos in open_positions.items():
    close = float(df_15m[sym]["close"].iloc[-1])
    open_unreal.append((close / float(pos.average_entry_price) - 1) * 100)
    open_age_days.append((end_ts[sym] - pd.Timestamp(pos.entry_time)).total_seconds() / 86400)
if open_positions:
    print(
        f"still open at the end: {len(open_positions)}; age in days {q(open_age_days, (0, 50, 100))}; "
        f"unrealised % {q(open_unreal, (0, 50, 100))}"
    )

# stop distance in ATRs at the entry bar (the signal bar is the one before the entry bar)
dist = []
bucket_reason = []
for t in trades:
    df = df_15m[t.symbol].sort_values("timestamp").reset_index(drop=True)
    ind = calculate_indicators(df, RUNTIME_STRATEGY)
    ind = ind.set_index(pd.DatetimeIndex(pd.to_datetime(ind["timestamp"], utc=True)))
    signal_bar = pd.Timestamp(t.entry_time) - pd.Timedelta(minutes=15)
    atr_pct = float(ind["atr"].asof(signal_bar) / ind["close"].asof(signal_bar) * 100)
    dist.append(STOP / atr_pct if atr_pct > 0 else np.nan)
dist = np.array(dist)
print(f"\nstop distance in ATRs at entry (4 % / the ATR % of the signal bar): {q(dist)}")
print(
    "target distance in ATRs at entry (5 % / ATR %):                     " + q(dist * TARGET / STOP)
)
if len(trades) >= 30:
    cuts = np.nanpercentile(dist, [33.3, 66.7])
    terc = np.digitize(dist, cuts)
    print("\nby tercile of the stop distance in ATRs (NO profit shown, only how trades end):")
    for k, name in enumerate(["tightest third", "middle third", "widest third"]):
        sel = terc == k
        reasons = pd.Series([t.exit_reason for t, s in zip(trades, sel) if s]).value_counts()
        print(
            f"  {name}: {sel.sum()} trades, stop distance {np.nanmin(dist[sel]):.1f}-"
            f"{np.nanmax(dist[sel]):.1f} ATR; closed by {dict(reasons)}; median hours held "
            f"{np.median(hold_h[sel]):.0f}; same day {same_day[sel].mean() * 100:.0f}%"
        )
