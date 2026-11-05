"""The news cache (`news.db`, setting NEWS_DATABASE_URL): what Alpaca's news API said about our
symbols, kept so that the Oracle can be told "what was said about AMZN in the last day" without a
call to Alpaca at every decision, and so that its verdicts can be judged against what was known.

It is a cache, not a record: Alpaca keeps its news from 2015, so the file can be deleted and
filled again. What it holds that Alpaca does not give back is *when we first saw* an article
(`first_seen_at`). It has its own declarative base, so alembic does not see it and it needs no
migrations; if the shape changes, delete the file.

Two tables. `news_articles` has one row per article, with the text cut down to plain text of at
most MAX_CONTENT_CHARS. `news_symbols` has one row per article and symbol it is about, with the
publication time copied in, so that "this symbol, in this window" is one range of one index.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    aliased,
    mapped_column,
    sessionmaker,
)

from trading_system.storage.database import (
    get_engine,
    get_session_factory,
    same_database,
    sqlite_file,
)

# SQLite refuses more than 999 bound values in one statement.
_CHUNK = 400


class NewsBase(DeclarativeBase):
    pass


class NewsArticleModel(NewsBase):
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # Alpaca's own id
    source: Mapped[str] = mapped_column(String)
    headline: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(String, default="")
    content: Mapped[str] = mapped_column(String, default="")  # plain text, cut short
    content_truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    author: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # published, UTC
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # the provider changed it
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # we stored it


class NewsSymbolModel(NewsBase):
    __tablename__ = "news_symbols"

    article_id: Mapped[int] = mapped_column(ForeignKey("news_articles.id"), primary_key=True)
    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # the article's

    __table_args__ = (Index("ix_news_symbols_symbol_created", "symbol", "created_at"),)


@dataclass(frozen=True)
class NewsItem:
    """An article as the cache takes it (data/news.py builds it from Alpaca's)."""

    id: int
    source: str
    headline: str
    summary: str
    content: str
    content_truncated: bool
    url: Optional[str]
    author: str
    created_at: datetime
    updated_at: datetime
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class StoredNews:
    id: int
    source: str
    headline: str
    summary: str
    content: str
    url: Optional[str]
    created_at: datetime
    first_seen_at: datetime
    symbols: tuple[str, ...]


@dataclass
class SaveReport:
    new: int = 0
    updated: int = 0
    unchanged: int = 0


def _utc(moment: datetime) -> datetime:
    """Times go into SQLite as the fields they have, so they must be UTC first."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc)


def _as_utc(moment: datetime) -> datetime:
    """A time read back from SQLite (naive UTC) as an aware one."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


class NewsRepository:
    def __init__(self, session_factory: "sessionmaker[Session]"):
        self._session_factory = session_factory

    def save(self, items: Sequence[NewsItem], *, seen_at: datetime) -> SaveReport:
        """Stores the articles. A new one gets `first_seen_at = seen_at`. One already stored is
        replaced only if the provider changed it since (`updated_at` is later), keeping its
        `first_seen_at`; otherwise it is left alone. The same article twice in `items` counts once
        (the later copy wins)."""
        report = SaveReport()
        latest: dict[int, NewsItem] = {item.id: item for item in items}
        if not latest:
            return report
        with self._session_factory.begin() as session:
            known: dict[int, datetime] = {}
            ids = list(latest)
            for at in range(0, len(ids), _CHUNK):
                rows = session.execute(
                    select(NewsArticleModel.id, NewsArticleModel.updated_at).where(
                        NewsArticleModel.id.in_(ids[at : at + _CHUNK])
                    )
                ).all()
                known.update({row[0]: _as_utc(row[1]) for row in rows})

            for item_id, item in latest.items():
                if item_id not in known:
                    session.execute(insert(NewsArticleModel), [self._row(item, seen_at)])
                    report.new += 1
                elif _utc(item.updated_at) > known[item_id]:
                    values = self._row(item, seen_at)
                    del values["first_seen_at"]  # the first sight stays the first sight
                    session.execute(
                        update(NewsArticleModel)
                        .where(NewsArticleModel.id == item_id)
                        .values(**values)
                    )
                    session.execute(
                        delete(NewsSymbolModel).where(NewsSymbolModel.article_id == item_id)
                    )
                    report.updated += 1
                else:
                    report.unchanged += 1
                    continue
                symbols = sorted(set(item.symbols))
                if symbols:
                    session.execute(
                        insert(NewsSymbolModel),
                        [
                            {
                                "article_id": item_id,
                                "symbol": symbol,
                                "created_at": _utc(item.created_at),
                            }
                            for symbol in symbols
                        ],
                    )
        return report

    @staticmethod
    def _row(item: NewsItem, seen_at: datetime) -> dict[str, object]:
        return {
            "id": item.id,
            "source": item.source,
            "headline": item.headline,
            "summary": item.summary,
            "content": item.content,
            "content_truncated": item.content_truncated,
            "url": item.url,
            "author": item.author,
            "created_at": _utc(item.created_at),
            "updated_at": _utc(item.updated_at),
            "first_seen_at": _utc(seen_at),
        }

    def for_symbol(
        self,
        symbol: str,
        *,
        before: datetime,
        max_age: timedelta,
        limit: int,
        max_symbols: Optional[int] = None,
        seen_by: bool = False,
    ) -> list[StoredNews]:
        """The newest articles about `symbol` published at or before `before` and not older than
        `max_age`, newest first, at most `limit`.

        `max_symbols` leaves out round-ups that name more symbols than that. `seen_by` also
        requires that we had stored the article by `before`: what was known then, not what was
        published then (articles filled in afterwards have a later `first_seen_at`)."""
        upper = _utc(before)
        query = (
            select(NewsArticleModel)
            .join(NewsSymbolModel, NewsSymbolModel.article_id == NewsArticleModel.id)
            .where(
                NewsSymbolModel.symbol == symbol,
                NewsSymbolModel.created_at <= upper,
                NewsSymbolModel.created_at >= upper - max_age,
            )
            .order_by(NewsSymbolModel.created_at.desc(), NewsArticleModel.id.desc())
            .limit(limit)
        )
        if seen_by:
            query = query.where(NewsArticleModel.first_seen_at <= upper)
        if max_symbols is not None:
            tagged = aliased(NewsSymbolModel)  # not the one the query joins: its own count
            count = (
                select(func.count())
                .select_from(tagged)
                .where(tagged.article_id == NewsArticleModel.id)
                .scalar_subquery()
            )
            query = query.where(count <= max_symbols)
        with self._session_factory() as session:
            articles = session.scalars(query).all()
            by_article: dict[int, list[str]] = {}
            ids = [a.id for a in articles]
            for at in range(0, len(ids), _CHUNK):
                for article_id, name in session.execute(
                    select(NewsSymbolModel.article_id, NewsSymbolModel.symbol)
                    .where(NewsSymbolModel.article_id.in_(ids[at : at + _CHUNK]))
                    .order_by(NewsSymbolModel.symbol)
                ):
                    by_article.setdefault(article_id, []).append(name)
            return [
                StoredNews(
                    id=a.id,
                    source=a.source,
                    headline=a.headline,
                    summary=a.summary,
                    content=a.content,
                    url=a.url,
                    created_at=_as_utc(a.created_at),
                    first_seen_at=_as_utc(a.first_seen_at),
                    symbols=tuple(by_article.get(a.id, [])),
                )
                for a in articles
            ]

    def latest_created_at(self) -> Optional[datetime]:
        """The publication time of the newest stored article, or None if there is none."""
        with self._session_factory() as session:
            newest: Optional[datetime] = session.execute(
                select(func.max(NewsArticleModel.created_at))
            ).scalar_one()
        return None if newest is None else _as_utc(newest)

    def count(self) -> int:
        with self._session_factory() as session:
            return int(session.execute(select(func.count(NewsArticleModel.id))).scalar_one())


class SameNewsDatabaseError(ValueError):
    """The news file was pointed at another database that is not its own."""


def open_news_store(news_url: str, *other_urls: str) -> NewsRepository:
    """The news repository on its own file, created with its tables if it is new. Refuses to work
    if the URL names the same database as any of `other_urls` (the trading and research files)."""
    url = make_url(news_url)
    for other in other_urls:
        if same_database(url, make_url(other)):
            raise SameNewsDatabaseError(
                "NEWS_DATABASE_URL is the same database as another one of the system "
                f"({other}); the news cache has a file of its own"
            )
    news_file = sqlite_file(url)
    if news_file is not None:
        news_file.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(news_url)
    NewsBase.metadata.create_all(engine)
    return NewsRepository(get_session_factory(engine))
