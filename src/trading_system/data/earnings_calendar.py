"""Who reports earnings on which day (docs/strategy-research-plan.md, section 16.1).

The source is the public earnings calendar of Nasdaq (`api.nasdaq.com/api/calendar/earnings`): not an
official feed, it can be wrong or missing, and it answers one day at a time. This module is a RECORD
of what that page said, kept in a small JSON file shared by the containers (`var/earnings_calendar.json`).
It is read-only for the trading: nothing here can refuse a signal, and a page that cannot be read is
`known = False`, never "no report".

A day in the future is read once; today is read again after `REFRESH_TODAY` (the list is mostly fixed,
the refresh only catches a late addition); a day in the past is never read again.
"""

import json
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

URL = "https://api.nasdaq.com/api/calendar/earnings?date={day}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
REFRESH_TODAY = timedelta(hours=6)
RETRY_AFTER_FAILURE = timedelta(minutes=10)  # one failed day is not asked again for this long
MARKET_TZ = ZoneInfo("America/New_York")

PRE = "pre"  # before the open: the price already holds it when the session starts
AFTER = "after"
UNKNOWN = "unknown"  # the page does not say; treated like after the close
_TIMING = {"time-pre-market": PRE, "time-after-hours": AFTER, "time-not-supplied": UNKNOWN}

Fetch = Callable[[date], Mapping[str, str]]


def fetch_day(day: date) -> dict[str, str]:
    """symbol -> timing for one day, from the page. Raises on a reply that is not what it should be."""
    request = urllib.request.Request(
        URL.format(day=day.isoformat()),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=8) as reply:  # noqa: S310 (a fixed https URL)
        body = json.load(reply)
    if not isinstance(body, dict) or (body.get("status") or {}).get("rCode") != 200:
        raise ValueError("the earnings calendar did not answer with rCode 200")
    rows = (body.get("data") or {}).get("rows")
    if rows is None:  # a weekend or a holiday: a real answer, nobody reports
        return {}
    out: dict[str, str] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if symbol:
            out[symbol] = _TIMING.get(str(row.get("time", "")), UNKNOWN)
    return out


def next_weekdays(day: date, count: int) -> list[date]:
    """The next `count` weekdays after `day`. A holiday is not known here; it lists nobody."""
    out: list[date] = []
    cursor = day
    while len(out) < count:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            out.append(cursor)
    return out


@dataclass(frozen=True)
class EarningsInfo:
    known: bool  # all three days were read
    in_2_sessions: Optional[bool]  # True on any hit; False only when known; else None
    dates: list[dict[str, str]] = field(default_factory=list)  # [{"date": ..., "timing": ...}]


class EarningsCalendar:
    def __init__(
        self,
        cache_path: Path,
        fetch: Optional[Fetch] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self._fetch = fetch or fetch_day
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._failed_at: dict[date, datetime] = {}

    # ------------------------------------------------------------------ cache file

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, cache: dict[str, Any], today: date) -> None:
        # Days long gone are dropped so that the file stays small; written to a temporary file and
        # moved into place so that another container never reads half a file.
        keep = (today - timedelta(days=14)).isoformat()
        cache = {k: v for k, v in cache.items() if k >= keep}
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            handle, temp = tempfile.mkstemp(dir=self.cache_path.parent, suffix=".tmp")
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                json.dump(cache, out, sort_keys=True)
            os.replace(temp, self.cache_path)
        except OSError as e:
            logger.warning(f"Could not write the earnings calendar cache: {e}")

    # ------------------------------------------------------------------ reading

    def reports(self, day: date) -> Optional[dict[str, str]]:
        """symbol -> timing for `day`, or None when the page could not be read for it."""
        now = self._clock()
        today = now.astimezone(MARKET_TZ).date()
        cache = self._load()
        entry = cache.get(day.isoformat())
        if entry is not None:
            fetched = datetime.fromisoformat(entry["fetched_at"])
            stale = day == today and now - fetched > REFRESH_TODAY
            if not stale:
                rows = entry["rows"]
                return dict(rows) if isinstance(rows, dict) else None
        failed = self._failed_at.get(day)
        if failed is not None and now - failed < RETRY_AFTER_FAILURE:
            return dict(entry["rows"]) if entry is not None else None
        try:
            rows = dict(self._fetch(day))
        except Exception as e:  # a page that cannot be read is "not known", and the loop goes on
            self._failed_at[day] = now
            logger.warning(f"Could not read the earnings calendar for {day}: {e}")
            return dict(entry["rows"]) if entry is not None else None
        self._failed_at.pop(day, None)
        cache[day.isoformat()] = {"fetched_at": now.isoformat(), "rows": rows}
        self._save(cache, today)
        return rows

    def lookup(self, symbol: str, signal_day: date) -> EarningsInfo:
        """Whether `symbol` reports in a way that can move the price while a position bought at
        `signal_day` is held (section 16.1): after the close or unspecified on the day itself, or at any
        time on either of the next two weekdays."""
        wanted = symbol.upper()
        variants = {wanted, wanted.replace(".", "/"), wanted.replace("/", ".")}
        known = True
        found: list[dict[str, str]] = []
        for index, day in enumerate([signal_day, *next_weekdays(signal_day, 2)]):
            rows = self.reports(day)
            if rows is None:
                known = False
                continue
            timing = next((rows[v] for v in variants if v in rows), None)
            if timing is None:
                continue
            if index == 0 and timing == PRE:
                continue  # already in the price when the session opened
            found.append({"date": day.isoformat(), "timing": timing})
        if found:
            return EarningsInfo(known=known, in_2_sessions=True, dates=found)
        return EarningsInfo(known=known, in_2_sessions=False if known else None, dates=[])
