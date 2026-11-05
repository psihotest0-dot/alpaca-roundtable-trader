"""Keeps the news cache up to date from Alpaca's news API (Benzinga and fiscal.ai).

What the API gives (checked live on 2026-11-05, free plan): about 500 articles a day for the 155
symbols we follow, 40 or so on a weekend day; stamps in UTC to the second; the history reaches back
to 2015. A third of the articles are headline only (an analyst's price target changed), the rest
carry a summary and a body in HTML, up to 78,000 characters. About one in forty is changed by the
provider after it is published (`updated_at` later than `created_at`), so every update looks back a
couple of days and takes a changed article in place. A few are round-ups that name over thirty
symbols. `limit` is a limit on the whole answer, not on a page: the library follows the pages by
itself up to it.

The text is kept as plain text and cut to MAX_CONTENT_CHARS: the Oracle is to be given a headline
and at most a few hundred characters of summary, and the rest can be fetched again.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html.parser import HTMLParser
from typing import Optional, Sequence

from alpaca.data.historical.news import NewsClient
from alpaca.data.models.news import News
from alpaca.data.requests import NewsRequest
from pydantic import SecretStr

from trading_system.common.retry import retry_read_request
from trading_system.config import Settings
from trading_system.storage.news import NewsItem, NewsRepository

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 2000

# One request asks for one day. A day of our symbols is about 600 articles; the limit leaves room,
# and a day that reaches it is reported (some articles would be missing).
FETCH_LIMIT = 10_000
WINDOW = timedelta(days=1)
# Each window starts a second before the last one ended, in case `start` is exclusive; an article
# fetched twice is stored once.
WINDOW_OVERLAP = timedelta(seconds=1)

DEFAULT_BACKFILL = timedelta(days=7)  # the first run, with nothing stored
OVERLAP = timedelta(days=2)  # an update looks this far behind the newest stored article
MAX_CATCH_UP = timedelta(days=30)  # after a long stop, no further back than this


class _TextExtractor(HTMLParser):
    _BLOCKS = frozenset(
        {"p", "br", "div", "li", "ul", "ol", "tr", "table", "blockquote"}
        | {f"h{n}" for n in range(1, 7)}
    )
    _SKIPPED = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in self._SKIPPED:
            self._skipping += 1
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED:
            self._skipping = max(0, self._skipping - 1)
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)


def strip_html(text: str) -> str:
    """The text of some HTML: tags and scripts gone, entities decoded, white space collapsed, one
    paragraph per line. Plain text passes through, only tidied."""
    if not text:
        return ""
    extractor = _TextExtractor()
    extractor.feed(text)
    extractor.close()
    lines = (" ".join(line.split()) for line in "".join(extractor.parts).splitlines())
    return "\n".join(line for line in lines if line)


def to_item(article: News) -> NewsItem:
    body = strip_html(article.content)
    seen: dict[str, None] = {}
    for symbol in article.symbols:
        seen.setdefault(symbol.strip().upper())
    return NewsItem(
        id=int(article.id),
        source=article.source,
        headline=" ".join(article.headline.split()),
        summary=strip_html(article.summary),
        content=body[:MAX_CONTENT_CHARS],
        content_truncated=len(body) > MAX_CONTENT_CHARS,
        url=article.url or None,
        author=article.author,
        created_at=article.created_at,
        updated_at=article.updated_at,
        symbols=tuple(s for s in seen if s),
    )


class NewsFeed:
    """Alpaca's news API behind one read-only call."""

    def __init__(self, settings: Settings) -> None:
        def reveal(key: object) -> str:
            return key.get_secret_value() if isinstance(key, SecretStr) else str(key)

        self.client = NewsClient(
            api_key=reveal(settings.alpaca_api_key), secret_key=reveal(settings.alpaca_secret_key)
        )

    @retry_read_request()
    def fetch(self, symbols: Sequence[str], start: datetime, end: datetime) -> list[NewsItem]:
        """The articles about any of `symbols` published from `start` to `end`, oldest first."""
        if not symbols:
            return []
        request = NewsRequest(
            symbols=",".join(symbols),
            start=start,
            end=end,
            limit=FETCH_LIMIT,
            sort="asc",
            include_content=True,
        )
        result = self.client.get_news(request)
        return [to_item(article) for article in getattr(result, "data", {}).get("news", [])]


@dataclass
class NewsReport:
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    windows: int = 0
    full_windows: list[datetime] = field(default_factory=list)  # starts of windows at the limit


def first_start(latest: Optional[datetime], now: datetime, days: Optional[int] = None) -> datetime:
    """Where an update begins. `days` asked for explicitly: that far back. Nothing stored: a week.
    Otherwise a little behind the newest stored article (to take in late changes), but never
    further back than a month."""
    if days is not None:
        return now - timedelta(days=days)
    if latest is None:
        return now - DEFAULT_BACKFILL
    return max(latest - OVERLAP, now - MAX_CATCH_UP)


def update_news(
    feed: NewsFeed,
    repo: NewsRepository,
    symbols: Sequence[str],
    *,
    now: datetime,
    days: Optional[int] = None,
) -> NewsReport:
    """Brings the cache up to `now` for `symbols`, a day at a time."""
    report = NewsReport()
    start = first_start(repo.latest_created_at(), now, days)
    while start < now:
        end = min(start + WINDOW, now)
        items = feed.fetch(symbols, start, end)
        saved = repo.save(items, seen_at=now)
        report.windows += 1
        report.new += saved.new
        report.updated += saved.updated
        report.unchanged += saved.unchanged
        if len(items) >= FETCH_LIMIT:
            logger.warning("News window from %s reached the limit of %d", start, FETCH_LIMIT)
            report.full_windows.append(start)
        start = end - WINDOW_OVERLAP if end < now else now
    return report
