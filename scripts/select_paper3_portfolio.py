"""Step 1 of the paper3 portfolio (docs/paper-third-account.md, section 6): the mechanical candidates.

Run it inside a container that has the code and the Alpaca data keys, from the repo root:

    docker compose exec -T trader python - < scripts/select_paper3_portfolio.py > candidates.json

It writes ONE JSON object on stdout (the candidates and the numbers behind them) and a short summary
on stderr. It places no order and writes nothing to any database.

**The rule is registered in the document before any number, and it looks at no return of any kind:**

1. The pool: the observed stocks (the research universe and the watchlist, less SPY) and the Oracle's
   extra stocks (`data.oracle_extra`), about 450.
2. Eligible: an active, tradable plain stock on NYSE or Nasdaq (the name rule of
   `select_oracle_extra.py`: no ETF, ETN, fund, trust, warrant, right, unit, preferred, note,
   depositary share or acquisition shell), median close at least 5 dollars over the last 60 sessions,
   at least 55 of those 60 sessions present, daily bars from SIP (the full-market feed). Only sessions
   that are over count: the day of the run is left out.
3. Risk = the median daily range (high - low) / close over the last 45 of those sessions.
4. Risky candidates = the 45 eligible stocks with the largest risk. Normal candidates = the 40 eligible
   stocks whose risk is closest to the median risk of the eligible, taken only from the upper half by
   median dollar volume (so the big liquid companies come in by themselves and there is no third group).
5. Ties are broken by the symbol, so a re-run on the same bars gives the same lists.

The review (step 2) and the freeze (step 3) are `scripts/freeze_paper3_portfolio.py`.
"""

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from trading_system.config import Settings
from trading_system.data.alpaca_data import AlpacaData
from trading_system.data.universe import (
    MARKET_SYMBOL,
    observation_universe,
    oracle_extra_symbols,
)

DAILY_SESSIONS = 60
MIN_DAILY_SESSIONS = 55
RANGE_SESSIONS = 45
MIN_PRICE = 5.0
RISKY_CANDIDATES = 45
NORMAL_CANDIDATES = 40
CHUNK = 100

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


def plain_stocks(settings: Settings) -> dict[str, str]:
    """symbol -> company name for every active, tradable, plain NYSE/Nasdaq stock."""
    client = TradingClient(
        settings.alpaca_api_key.get_secret_value(),
        settings.alpaca_secret_key.get_secret_value(),
        paper=True,
    )
    assets = client.get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    )
    out: dict[str, str] = {}
    for a in assets:
        exchange = str(getattr(a.exchange, "value", a.exchange))
        if not a.tradable or exchange not in EXCHANGES:
            continue
        if not re.fullmatch(r"[A-Z]{1,5}", a.symbol):
            continue
        if NOT_A_STOCK.search(a.name or ""):
            continue
        out[a.symbol] = a.name or ""
    return out


def table(data: AlpacaData, symbols: list[str], now: datetime) -> pd.DataFrame:
    """One row a symbol: median dollar volume, median close, sessions, median daily range."""
    today = now.astimezone(ZoneInfo("America/New_York")).date()
    start = now - timedelta(days=DAILY_SESSIONS * 7 // 5 + 12)
    rows = []
    last_day = None
    for part in chunks(symbols, CHUNK):
        frames = data.fetch_bars_bulk(part, TimeFrame.Day, start, now, feed="sip")
        for symbol, bars in frames.items():
            local = bars.index.tz_convert("America/New_York")
            bars = bars[[d < today for d in local.date]].tail(DAILY_SESSIONS)
            if len(bars) < MIN_DAILY_SESSIONS:
                continue
            day = bars.index[-1].tz_convert("America/New_York").date()
            last_day = day if last_day is None else max(last_day, day)
            recent = bars.tail(RANGE_SESSIONS)
            rows.append(
                {
                    "symbol": symbol,
                    "median_dollar_volume": float((bars["close"] * bars["volume"]).median()),
                    "median_close": float(bars["close"].median()),
                    "sessions": len(bars),
                    "median_range": float(
                        ((recent["high"] - recent["low"]) / recent["close"]).median()
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["last_session"] = str(last_day)
    return out


def main() -> None:
    settings = Settings()
    data = AlpacaData(settings)
    now = datetime.now(timezone.utc)

    watchlist = settings.watchlist_symbols
    pool = sorted(
        (set(observation_universe(watchlist)) - {MARKET_SYMBOL})
        | set(oracle_extra_symbols(watchlist))
    )
    say(f"pool (observed + Oracle extra): {len(pool)}")

    names = plain_stocks(settings)
    plain = [s for s in pool if s in names]
    say(f"plain NYSE/Nasdaq stocks among them: {len(plain)}")

    frame = table(data, plain, now)
    say(f"with {MIN_DAILY_SESSIONS}+ completed daily sessions: {len(frame)}")
    frame = frame[frame["median_close"] >= MIN_PRICE].copy()
    say(f"with a median close of {MIN_PRICE:g} or more (the eligible): {len(frame)}")

    frame["name"] = frame["symbol"].map(names)
    frame = frame.sort_values(["median_range", "symbol"], ascending=[False, True]).reset_index(
        drop=True
    )
    frame["risk_rank"] = frame.index + 1

    risky = frame.head(RISKY_CANDIDATES)

    median_range = float(frame["median_range"].median())
    median_volume = float(frame["median_dollar_volume"].median())
    upper = frame[frame["median_dollar_volume"] >= median_volume].copy()
    upper["distance"] = (upper["median_range"] - median_range).abs()
    normal = upper.sort_values(["distance", "symbol"]).head(NORMAL_CANDIDATES)
    normal = normal.sort_values(["distance", "symbol"]).drop(columns="distance")

    # The full orders, for the replacements of step 2 ("the next one by the rule"): the risky order is
    # the risk order of all the eligible; the normal order is the upper half by dollar volume, nearest
    # the median risk first.
    normal_order = upper.sort_values(["distance", "symbol"])["symbol"].tolist()
    risky_order = frame["symbol"].tolist()

    overlap = set(risky["symbol"]) & set(normal["symbol"])
    if overlap:
        raise SystemExit(f"a stock is in both groups: {sorted(overlap)}")

    say(
        f"last completed session in the data: {frame.attrs.get('last_session')}; "
        f"median risk {median_range:.4f}; median dollar volume {median_volume / 1e6:.0f} M"
    )
    say(
        f"risky candidates: risk {risky['median_range'].iloc[0]:.4f} down to "
        f"{risky['median_range'].iloc[-1]:.4f}"
    )
    say(
        f"normal candidates: risk {normal['median_range'].min():.4f} to "
        f"{normal['median_range'].max():.4f}, dollar volume from "
        f"{normal['median_dollar_volume'].min() / 1e6:.0f} M"
    )

    def records(part: pd.DataFrame) -> list[dict]:
        return part.round(5).to_dict(orient="records")

    print(
        json.dumps(
            {
                "made_at": now.isoformat(timespec="seconds"),
                "rule": "docs/paper-third-account.md section 6, step 1",
                "last_session": frame.attrs.get("last_session"),
                "pool": len(pool),
                "eligible": len(frame),
                "median_range": round(median_range, 5),
                "median_dollar_volume": round(median_volume, 0),
                "risky_order": risky_order,
                "normal_order": normal_order,
                "risky_candidates": records(risky),
                "normal_candidates": records(normal),
                "eligible_rows": records(frame),
            }
        )
    )


main()
