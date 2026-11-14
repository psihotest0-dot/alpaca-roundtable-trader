"""The strategy research's own candle file (`research.db`, setting RESEARCH_DATABASE_URL).

It holds one table, `market_bars`, in the same shape as in trading.db, and nothing else: it is a
cache that `fetch-daily` rebuilds from Alpaca, so it needs no migrations and no backup. It is a
separate file so that a bulk write cannot hold the trading loop's lock, and so that the backups of
trading.db do not carry hundreds of thousands of research rows. If the table's shape ever changes,
delete the file and fetch again.
"""

from sqlalchemy.engine import make_url

from trading_system.storage.database import (
    get_engine,
    get_session_factory,
    same_database,
    sqlite_file,
)
from trading_system.storage.models import MarketBarModel
from trading_system.storage.repositories import MarketBarRepository


class SameDatabaseError(ValueError):
    """The research file was pointed at the trading database."""


def open_research_store(research_url: str, trading_url: str) -> MarketBarRepository:
    """The candle repository on the research file, created with its table if it is new.

    Refuses to work if both URLs name the same database: the point of the file is to stay out of
    trading.db."""
    url = make_url(research_url)
    if same_database(url, make_url(trading_url)):
        raise SameDatabaseError(
            "RESEARCH_DATABASE_URL is the same as DATABASE_URL; the research candles must not "
            "go into the trading database"
        )
    research_file = sqlite_file(url)
    if research_file is not None:
        research_file.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(research_url)
    MarketBarModel.__table__.create(engine, checkfirst=True)  # type: ignore[attr-defined]
    return MarketBarRepository(get_session_factory(engine))


def open_observe_store(observe_url: str, trading_url: str) -> MarketBarRepository:
    """The candle repository on the observation file (15-minute and 1-hour bars of ~155 symbols).
    Same shape and the same rule as the research file: never trading.db."""
    return open_research_store(observe_url, trading_url)


def open_crypto_store(
    crypto_url: str, trading_url: str, research_url: str | None = None
) -> MarketBarRepository:
    """The candle repository on the crypto file (daily bars of a few coins, plan section 15).
    Same shape as the research file, with one more rule: it must not be the research file either,
    because the stocks in research.db are the baseline of `measure-signals`."""
    if research_url is not None and same_database(make_url(crypto_url), make_url(research_url)):
        raise SameDatabaseError(
            "CRYPTO_DATABASE_URL is the same as RESEARCH_DATABASE_URL; the crypto candles must "
            "not go into the file the stock measurements read"
        )
    return open_research_store(crypto_url, trading_url)
