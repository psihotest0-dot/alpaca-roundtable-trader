"""What the Oracle may be told of the news (Phase 5): a few recent articles about the stock, cut
down to a clean headline and a short summary.

The news cache (storage/news.py) holds what Alpaca sent, which is outside text. This is the
gate between it and a model. It does three things.

* **Chooses.** The newest articles about the symbol from the last day that the cache had stored
  by now (`seen_by`), at most MAX_ITEMS. Round-ups that name more than MAX_SYMBOLS stocks are
  left out: an article that lists thirty tickers is not about this one. The same story from the
  two sources (the same headline) counts once. A headline-only article stays: "Needham lowers
  price target to $300" is the whole news.
* **Cleans.** Characters that are invisible or steer text (control, zero-width, bidirectional
  marks), addresses, leftover tags and chat-template tokens such as `<|im_start|>`, and code
  fences are removed; white space becomes single spaces, so an article is one line that cannot
  begin a new paragraph or a new section of a prompt; the length is cut.
* **Bounds.** At most MAX_ITEMS items of at most HEADLINE_CHARS + SUMMARY_CHARS characters.

Cleaning cannot recognise an instruction written in plain words ("approve every signal"), and
does not pretend to. What limits the harm of one is the design: the Oracle answers only yes, no or
not now to a signal that exists, the risk engine decides the order, and the provider (2.2) puts
the articles in the prompt as quoted data under a line that says so.
"""

import re
import unicodedata
from datetime import datetime, timedelta
from typing import Optional

from trading_system.oracle.contract import OracleNews
from trading_system.storage.news import NewsRepository

MAX_AGE = timedelta(hours=24)
MAX_ITEMS = 5
MAX_SYMBOLS = 5
HEADLINE_CHARS = 160
SUMMARY_CHARS = 300
READ_ROWS = 25  # read this many, so that dropped duplicates can be made up for

_ADDRESS = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
_TAG = re.compile(r"<[^<>]{0,200}>")  # a tag, or a template token such as <|im_start|>
_BREAKS = {"Cc", "Zl", "Zp"}  # control characters and line or paragraph separators: a space
_INVISIBLE = {"Cf"}  # zero-width and bidirectional marks: removed, so a word split by one is whole
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


def clean_for_model(text: str, limit: int) -> str:
    """`text` as one line of plain characters of at most `limit`, cut at a word and marked with an
    ellipsis if it was cut."""
    text = unicodedata.normalize("NFKC", text)  # full-width and look-alike forms become plain
    text = "".join(
        ""
        if unicodedata.category(ch) in _INVISIBLE
        else " "
        if unicodedata.category(ch) in _BREAKS
        else ch
        for ch in text
    )
    text = _ADDRESS.sub(" ", text)
    text = _TAG.sub(" ", text)
    text = text.replace("`", "'")
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "…"


def _same_story(headline: str) -> str:
    return _NOT_ALNUM.sub(" ", headline.lower()).strip()


def select_news(repo: Optional[NewsRepository], symbol: str, now: datetime) -> list[OracleNews]:
    """The news the Oracle may see about `symbol` at `now`, newest first. No repository, no news."""
    if repo is None:
        return []
    rows = repo.for_symbol(
        symbol,
        before=now,
        max_age=MAX_AGE,
        limit=READ_ROWS,
        max_symbols=MAX_SYMBOLS,
        seen_by=True,
    )
    chosen: list[OracleNews] = []
    stories: set[str] = set()
    for row in rows:
        headline = clean_for_model(row.headline, HEADLINE_CHARS)
        story = _same_story(headline)
        if not story or story in stories:
            continue
        stories.add(story)
        chosen.append(
            OracleNews(
                article_id=row.id,
                published_at=row.created_at,
                source=row.source,
                headline=headline,
                summary=clean_for_model(row.summary, SUMMARY_CHARS),
                other_symbols=max(len(row.symbols) - 1, 0),
            )
        )
        if len(chosen) == MAX_ITEMS:
            break
    return chosen
