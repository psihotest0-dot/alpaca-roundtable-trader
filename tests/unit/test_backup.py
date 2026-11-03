"""The archive of the database is only worth having if it can be restored."""

import gzip
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_system.cli import main
from trading_system.storage.backup import BackupError, archives, backup_database, database_file

START = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def _database(path: Path, rows: int = 5) -> sqlite3.Connection:
    """A database like ours: WAL mode, a migration revision, some data. The connection stays
    open and holds the last rows only in the -wal file, as a running system does."""
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("CREATE TABLE alembic_version (version_num TEXT NOT NULL)")
    connection.execute("INSERT INTO alembic_version VALUES ('b6d1f4a82c35')")
    connection.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY, symbol TEXT)")
    connection.executemany(
        "INSERT INTO signals (symbol) VALUES (?)", [(f"S{i}",) for i in range(rows)]
    )
    connection.commit()
    return connection


def _unpack(archive: Path, into: Path) -> sqlite3.Connection:
    into.write_bytes(gzip.decompress(archive.read_bytes()))
    return sqlite3.connect(into)


def test_the_archive_holds_what_is_only_in_the_wal_file(tmp_path: Path) -> None:
    live = _database(tmp_path / "trading.db", rows=7)
    assert (tmp_path / "trading.db-wal").stat().st_size > 0  # the rows are in the WAL, not the file

    result = backup_database(f"sqlite:///{tmp_path / 'trading.db'}", tmp_path / "out", now=START)

    restored = _unpack(result.path, tmp_path / "restored.db")
    assert restored.execute("SELECT count(*) FROM signals").fetchone()[0] == 7
    assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    live.close()


def test_the_archive_is_one_self_contained_file(tmp_path: Path) -> None:
    live = _database(tmp_path / "trading.db")

    result = backup_database(f"sqlite:///{tmp_path / 'trading.db'}", tmp_path / "out", now=START)

    restored = _unpack(result.path, tmp_path / "restored.db")
    assert restored.execute("PRAGMA journal_mode").fetchone()[0] == "delete"  # not WAL
    assert result.path.name == "trading-20261103-120000.db.gz"
    assert [p.name for p in (tmp_path / "out").iterdir()] == [result.path.name]  # no leftovers
    live.close()


def test_only_the_newest_archives_are_kept_and_other_files_are_left_alone(tmp_path: Path) -> None:
    live = _database(tmp_path / "trading.db")
    out = tmp_path / "out"
    out.mkdir()
    (out / "trading-before-move.db").write_bytes(b"a snapshot taken by hand")
    (out / "notes.txt").write_text("keep me")

    for hour in range(5):
        result = backup_database(
            f"sqlite:///{tmp_path / 'trading.db'}", out, keep=3, now=START + timedelta(hours=hour)
        )

    assert [p.name for p in archives(out)] == [
        "trading-20261103-140000.db.gz",
        "trading-20261103-150000.db.gz",
        "trading-20261103-160000.db.gz",
    ]
    assert result.kept == 3 and len(result.removed) == 1
    assert (out / "trading-before-move.db").exists() and (out / "notes.txt").exists()
    live.close()


def test_a_failing_run_costs_no_existing_archive(tmp_path: Path) -> None:
    live = _database(tmp_path / "trading.db")
    out = tmp_path / "out"
    url = f"sqlite:///{tmp_path / 'trading.db'}"
    good = [
        backup_database(url, out, keep=5, now=START + timedelta(hours=hour)).path.name
        for hour in range(3)
    ]
    live.close()
    (tmp_path / "trading.db-wal").unlink(missing_ok=True)
    (tmp_path / "trading.db-shm").unlink(missing_ok=True)
    (tmp_path / "trading.db").write_bytes(b"this is not a database " * 400)  # corrupted

    with pytest.raises(BackupError):
        backup_database(url, out, keep=2, now=START + timedelta(hours=6))

    # Nothing was pruned to make room for an archive that was never made, and no half file.
    assert sorted(p.name for p in out.iterdir()) == good


def test_a_crash_while_writing_leaves_no_half_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _database(tmp_path / "trading.db")
    out = tmp_path / "out"

    def disk_full(*args: object, **kwargs: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr("trading_system.storage.backup.shutil.copyfileobj", disk_full)

    with pytest.raises(OSError):
        backup_database(f"sqlite:///{tmp_path / 'trading.db'}", out, now=START)

    assert list(out.iterdir()) == []  # not even the temporary file
    live.close()


def test_an_empty_database_is_not_archived_as_if_it_were_ours(tmp_path: Path) -> None:
    sqlite3.connect(tmp_path / "trading.db").execute("CREATE TABLE x (a)").connection.commit()
    out = tmp_path / "out"

    with pytest.raises(BackupError, match="alembic_version|migration"):
        backup_database(f"sqlite:///{tmp_path / 'trading.db'}", out, now=START)

    assert list(out.iterdir()) == []


def test_a_missing_database_is_an_error_not_an_empty_archive(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'nope.db'}"

    with pytest.raises(BackupError, match="no database"):
        backup_database(url, tmp_path / "out", now=START)

    assert not (tmp_path / "nope.db").exists()  # sqlite3.connect alone would have made one


def test_only_sqlite_files_can_be_archived() -> None:
    with pytest.raises(BackupError):
        database_file("postgresql://user@host/db")
    with pytest.raises(BackupError):
        database_file("sqlite:///:memory:")


def test_keeping_nothing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        backup_database("sqlite:///x.db", tmp_path, keep=0)


def test_the_backup_command_writes_an_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    live = _database(tmp_path / "trading.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'trading.db'}")

    code = main(["backup-db", "--out", str(tmp_path / "out"), "--keep", "2"])

    assert code == 0
    assert len(archives(tmp_path / "out")) == 1
    assert "Backup written" in capsys.readouterr().out
    live.close()


def test_the_backup_command_reports_a_failure_with_a_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'missing.db'}")

    code = main(["backup-db", "--out", str(tmp_path / "out")])

    assert code == 1
    assert "Backup FAILED" in capsys.readouterr().out
