"""Two processes share the SQLite file (the trading loop and the candle update): the connection
has to wait for the other one's write lock instead of failing at once."""

from pathlib import Path

from trading_system.storage.database import SQLITE_BUSY_TIMEOUT_SECONDS, get_engine


def test_sqlite_connections_wait_for_a_busy_database(tmp_path: Path) -> None:
    engine = get_engine(f"sqlite:///{tmp_path / 'shared.db'}")

    with engine.connect() as connection:
        waits_ms = connection.exec_driver_sql("PRAGMA busy_timeout").scalar()

    assert waits_ms == SQLITE_BUSY_TIMEOUT_SECONDS * 1000
    assert SQLITE_BUSY_TIMEOUT_SECONDS >= 30  # a candle download writes for several seconds


def test_sqlite_connections_keep_wal_and_foreign_keys(tmp_path: Path) -> None:
    engine = get_engine(f"sqlite:///{tmp_path / 'shared.db'}")

    with engine.connect() as connection:
        mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar()
        keys = connection.exec_driver_sql("PRAGMA foreign_keys").scalar()

    assert (str(mode).lower(), keys) == ("wal", 1)


def test_a_wait_written_in_the_url_is_respected(tmp_path: Path) -> None:
    engine = get_engine(f"sqlite:///{tmp_path / 'shared.db'}?timeout=0.2")

    with engine.connect() as connection:
        waits_ms = connection.exec_driver_sql("PRAGMA busy_timeout").scalar()

    assert waits_ms == 200
