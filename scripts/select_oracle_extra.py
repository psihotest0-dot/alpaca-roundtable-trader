"""Picks the extra stocks the Oracle observer asks about (docs/oracle-provider-task.md, section 5.2).

Run it inside the trader container, from the repo root:

    docker compose exec -T trader python - < scripts/select_oracle_extra.py > extra.json

It writes ONE JSON object on stdout (the chosen symbols and the numbers behind the choice) and a
short summary on stderr.

**The rule is fixed here, before any result, and uses no return of any kind:**

1. The pool: every active, tradable US equity listed on NYSE or Nasdaq, whose symbol is 1 to 5
   letters, and whose name does not say it is a fund, trust, index product, warrant, right, unit,
   preferred, note, depositary share or an acquisition shell, or names a fund family (ProShares,
   Direxion, iShares, SPDR, Vanguard, Invesco) or a leveraged product. (The first run, 2026-11-19
   15:14 UTC, let two leveraged ETFs through, SQQQ and TQQQ, because their names had none of the
   first words; the word list was widened and the script run again BEFORE anything was
   registered or built.)
2. Liquidity: the median of close x volume over the last 60 sessions of daily bars from SIP (the
   full-market feed), at least 55 of those 60 sessions present, median close at least 5 dollars.
3. Not already followed: the research universe, the watchlist and SPY are left out.
4. The top POOL_TOP of that ranking by median dollar volume are checked for bar completeness on the
   feed the observer reads (IEX, 15-minute bars of the last SESSIONS regular sessions): the share
   of the 26 bars a session that exist, over sessions that are over (the first run also counted
   the day in progress, which only lowered every share by the same amount). The observer's signals are computed on those bars, and a
   stock whose bars are full of holes would give signals of the holes, not of the price.
5. A stock stays if its completeness is at least the 10th percentile of the stocks already followed
   (the same number, from the same feed, in the same run), and the first TAKE of those by dollar
   volume are the answer.
"""

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from trading_system.config import Settings
from trading_system.data.alpaca_data import FIFTEEN_MINUTES, AlpacaData
from trading_system.data.universe import MARKET_SYMBOL, observation_universe

TAKE = 300
POOL_TOP = 480
SESSIONS = 10
DAILY_SESSIONS = 60
MIN_DAILY_SESSIONS = 55
MIN_PRICE = 5.0
CHUNK = 100
BARS_PER_SESSION = 26

EXCHANGES = {"NYSE", "NASDAQ"}
NOT_A_STOCK = re.compile(
    r"\b(ETF|ETN|Fund|Trust|Index|Warrants?|Rights?|Units?|Preferred|Notes?|Depositary|"
    r"Acquisition|Shares|Portfolio|Series|ProShares|UltraPro|Direxion|iShares|SPDR|Vanguard|"
    r"Invesco|Leveraged)\b",
    re.IGNORECASE,
)


def say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def chunks(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def pool(settings: Settings) -> list[str]:
    client = TradingClient(
        settings.alpaca_api_key.get_secret_value(),
        settings.alpaca_secret_key.get_secret_value(),
        paper=True,
    )
    assets = client.get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    )
    out = []
    for a in assets:
        exchange = str(getattr(a.exchange, "value", a.exchange))
        if not a.tradable or exchange not in EXCHANGES:
            continue
        if not re.fullmatch(r"[A-Z]{1,5}", a.symbol):
            continue
        if NOT_A_STOCK.search(a.name or ""):
            continue
        out.append(a.symbol)
    return sorted(set(out))


def liquidity(data: AlpacaData, symbols: list[str], now: datetime) -> pd.DataFrame:
    start = now - timedelta(days=DAILY_SESSIONS * 7 // 5 + 8)
    rows = []
    for part in chunks(symbols, CHUNK):
        frames = data.fetch_bars_bulk(part, TimeFrame.Day, start, now, feed="sip")
        for symbol, bars in frames.items():
            bars = bars.tail(DAILY_SESSIONS)
            if len(bars) < MIN_DAILY_SESSIONS:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "median_dollar_volume": float((bars["close"] * bars["volume"]).median()),
                    "median_close": float(bars["close"].median()),
                    "sessions": len(bars),
                }
            )
    return pd.DataFrame(rows)


def completeness(data: AlpacaData, symbols: list[str], now: datetime) -> dict[str, float]:
    """Share of the 26 regular-session 15-minute bars a session that exist, over the last
    SESSIONS sessions that AAPL traded (a stand-in for the calendar)."""
    start = now - timedelta(days=SESSIONS * 7 // 5 + 6)
    frames: dict[str, pd.DataFrame] = {}
    for part in chunks(sorted(set(symbols) | {"AAPL"}), 25):
        frames.update(data.fetch_bars_bulk(part, FIFTEEN_MINUTES, start, now))

    def regular(df: pd.DataFrame) -> pd.DataFrame:
        local = df.index.tz_convert("America/New_York")
        keep = (local.time >= pd.Timestamp("09:30").time()) & (
            local.time < pd.Timestamp("16:00").time()
        )
        out = df[keep]
        out.index = out.index.tz_convert("America/New_York")
        return out

    ref = regular(frames["AAPL"])
    today = now.astimezone(ZoneInfo("America/New_York")).date()
    # Only sessions that are over: the day in progress would count as a day with most bars missing.
    days = sorted({d for d in ref.index.date if d < today})[-SESSIONS:]
    out = {}
    for symbol in symbols:
        df = frames.get(symbol)
        if df is None or df.empty:
            out[symbol] = 0.0
            continue
        r = regular(df)
        n = int(pd.Index(r.index.date).isin(days).sum())
        out[symbol] = n / (BARS_PER_SESSION * len(days))
    return out


def main() -> None:
    settings = Settings()
    data = AlpacaData(settings)
    now = datetime.now(timezone.utc)

    followed = set(observation_universe(settings.watchlist_symbols)) | {MARKET_SYMBOL}
    symbols = pool(settings)
    say(f"pool after the exchange, symbol and name rules: {len(symbols)}")

    table = liquidity(data, [s for s in symbols if s not in followed], now)
    say(f"with 55+ daily sessions: {len(table)}")
    table = table[table["median_close"] >= MIN_PRICE]
    say(f"with a median close of {MIN_PRICE} or more: {len(table)}")
    table = table.sort_values("median_dollar_volume", ascending=False).head(POOL_TOP)
    say(
        f"top {len(table)} by median dollar volume: "
        f"{table['median_dollar_volume'].iloc[0] / 1e6:.0f} M down to "
        f"{table['median_dollar_volume'].iloc[-1] / 1e6:.0f} M a day"
    )

    followed_list = sorted(followed - {MARKET_SYMBOL})
    comp = completeness(data, list(table["symbol"]) + followed_list, now)
    reference = np.array([comp[s] for s in followed_list])
    bar = float(np.percentile(reference, 10))
    say(
        "completeness of the stocks already followed: "
        f"min {reference.min():.2f}, 10th percentile {bar:.2f}, median {np.median(reference):.2f}"
    )
    table["completeness"] = table["symbol"].map(comp)
    kept = table[table["completeness"] >= bar]
    say(f"kept by completeness: {len(kept)} of {len(table)}")
    chosen = kept.head(TAKE)
    say(
        f"chosen: {len(chosen)}; median dollar volume of the last one "
        f"{chosen['median_dollar_volume'].iloc[-1] / 1e6:.0f} M; "
        f"median completeness {chosen['completeness'].median():.2f}"
    )
    print(
        json.dumps(
            {
                "made_at": now.isoformat(timespec="seconds"),
                "rule": "docs/oracle-provider-task.md section 5.2",
                "completeness_bar": round(bar, 4),
                "symbols": sorted(chosen["symbol"]),
                "rows": chosen.round(4).to_dict(orient="records"),
            }
        )
    )


main()
