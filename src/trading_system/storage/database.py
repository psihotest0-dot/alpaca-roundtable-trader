from sqlalchemy import create_engine, Engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import event
from typing import Any


def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


# Two containers share the database (the trading loop and the hourly candle update). SQLite
# lets one writer in at a time; the default wait for the lock is 5 s, which a bigger candle
# write can exceed. A short wait here beats a failed cycle there.
SQLITE_BUSY_TIMEOUT_SECONDS = 30


def get_engine(database_url: str) -> Engine:
    if "sqlite" in database_url:
        connect_args: dict[str, Any] = {}
        if "timeout" not in make_url(database_url).query:  # a wait set in the URL is the caller's
            connect_args["timeout"] = SQLITE_BUSY_TIMEOUT_SECONDS
        engine = create_engine(database_url, connect_args=connect_args)
        event.listen(engine, "connect", set_sqlite_pragma)
        return engine
    return create_engine(database_url)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False)
