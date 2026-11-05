"""Alpaca's news in the shape its client really returns it (real alpaca-py models, as the lesson
of the paper account's empty fields asks), and a client that answers from a list."""

from datetime import datetime, timezone
from typing import Any, Optional

from alpaca.data.models.news import NewsSet

from tests.fixtures.fakes import asked

UTC = timezone.utc


def raw_news(
    article_id: int,
    symbols: list[str],
    created_at: datetime,
    *,
    headline: str = "A headline",
    summary: str = "",
    content: str = "",
    updated_at: Optional[datetime] = None,
    source: str = "benzinga",
) -> dict[str, Any]:
    """One article as the API sends it (the client turns the stamps into datetimes)."""
    return {
        "id": article_id,
        "headline": headline,
        "source": source,
        "url": f"https://example.com/news/{article_id}",
        "summary": summary,
        "created_at": created_at.isoformat(),
        "updated_at": (updated_at or created_at).isoformat(),
        "symbols": symbols,
        "author": "Someone",
        "content": content,
        "images": [],
    }


class FakeNewsClient:
    """The news client of alpaca-py with a memory of what it was asked. Like the real one it
    filters by symbols and time, orders, and cuts the whole answer at `limit`. Both ends of the
    time range are inclusive here; the real API's edge is not known, which is why the code
    overlaps its windows."""

    def __init__(self, articles: list[dict[str, Any]]):
        self.articles = list(articles)
        self.requests: list[Any] = []

    def get_news(self, request: Any) -> NewsSet:
        self.requests.append(request)
        wanted = set((request.symbols or "").split(",")) - {""}
        start, end = asked(request.start), asked(request.end)
        chosen = [
            a
            for a in self.articles
            if (not wanted or wanted & set(a["symbols"]))
            and start <= asked(datetime.fromisoformat(a["created_at"])) <= end
        ]
        chosen.sort(key=lambda a: (a["created_at"], a["id"]), reverse=request.sort == "desc")
        return NewsSet({"news": chosen[: request.limit or len(chosen)]})
