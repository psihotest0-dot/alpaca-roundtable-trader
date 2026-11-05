"""The strategy research's own candle file (`research.db`, setting RESEARCH_DATABASE_URL).

It holds one table, `market_bars`, in the same shape as in trading.db, and nothing else: it is a
cache that `fetch-daily` rebuilds from Alpaca, so it needs no migrations and no backup. It is a
separate file so that a bulk write cannot hold the trading loop's lock, and so that the backups of
trading.db do not carry hundreds of thousands of research rows. If the table's shape ever changes,
delete the file and fetch again.
"""

from pathlib import Path
from typing import Optional

from sqlalchemy.engine import URL, make_url

from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.models import MarketBarModel
from trading_system.storage.repositories import MarketBarRepository


class SameDatabaseError(ValueError):
    """The research file was pointed at the trading database."""


def _sqlite_file(url: URL) -> Optional[Path]:
    """The file a SQLite URL names, or None for any other database (or one in memory)."""
    if url.get_backend_name() != "sqlite" or url.database in (None, "", ":memory:"):
        return None
    return Path(str(url.database)).resolve()


def _same_database(first: URL, second: URL) -> bool:
    file_a, file_b = _sqlite_file(first), _sqlite_file(second)
    if file_a is not None or file_b is not None:
        return file_a == file_b  # `./var/x.db` and `var/x.db` are one file
    return first.render_as_string(hide_password=True) == second.render_as_string(hide_password=True)


def open_research_store(research_url: str, trading_url: str) -> MarketBarRepository:
    """The candle repository on the research file, created with its table if it is new.

    Refuses to work if both URLs name the same database: the point of the file is to stay out of
    trading.db."""
    url = make_url(research_url)
    if _same_database(url, make_url(trading_url)):
        raise SameDatabaseError(
            "RESEARCH_DATABASE_URL is the same as DATABASE_URL; the research candles must not "
            "go into the trading database"
        )
    research_file = _sqlite_file(url)
    if research_file is not None:
        research_file.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(research_url)
    MarketBarModel.__table__.create(engine, checkfirst=True)  # type: ignore[attr-defined]
    return MarketBarRepository(get_session_factory(engine))
