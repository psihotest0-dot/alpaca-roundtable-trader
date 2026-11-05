from pathlib import Path
from typing import Any, Optional

from sqlalchemy import create_engine, Engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import event


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


def sqlite_file(url: URL) -> Optional[Path]:
    """The file a SQLite URL names, or None for any other database (or one in memory)."""
    if url.get_backend_name() != "sqlite" or url.database in (None, "", ":memory:"):
        return None
    return Path(str(url.database)).resolve()


def same_database(first: URL, second: URL) -> bool:
    file_a, file_b = sqlite_file(first), sqlite_file(second)
    if file_a is not None or file_b is not None:
        return file_a == file_b  # `./var/x.db` and `var/x.db` are one file
    return first.render_as_string(hide_password=True) == second.render_as_string(hide_password=True)
