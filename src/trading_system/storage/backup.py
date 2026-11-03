"""Archive copies of the SQLite database.

The database lives in a Docker volume, so a copy outside Docker is the only protection against
the volume being lost (Docker Desktop reset or reinstall, `docker compose down -v`, a wrong
`docker volume rm`). Copying the file would not do: the database runs in WAL mode, so
recent commits sit in the `-wal` file next to it and a file copy taken while the system runs
can be torn. SQLite's own backup API gives a consistent snapshot of a live database.

A snapshot is only worth keeping if it can be restored, so it is checked before it is kept,
it is written under a temporary name and renamed when complete (a crash never leaves a half
file that looks like a backup), and old archives are removed only after a new one is safely
there (a failing run never costs the archives that exist).
"""

import gzip
import logging
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.engine import make_url

logger = logging.getLogger(__name__)

# The only files this module creates and the only ones it ever deletes. Anything else in the
# folder (a snapshot taken by hand, notes) is left alone.
ARCHIVE_NAME = re.compile(r"^trading-\d{8}-\d{6}\.db\.gz$")


class BackupError(RuntimeError):
    """The database could not be copied, or the copy is not a usable database."""


@dataclass(frozen=True)
class BackupResult:
    path: Path
    size_bytes: int
    removed: list[Path] = field(default_factory=list)
    kept: int = 0


def database_file(database_url: str) -> Path:
    """The file behind a SQLite URL. Other databases have no single file to copy."""
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        raise BackupError(f"only a SQLite file can be backed up this way, got {database_url!r}")
    return Path(url.database)


def _snapshot(source: Path, target: Path) -> None:
    """A consistent copy of `source` in `target`, as one self-contained file."""
    reader = sqlite3.connect(source)
    try:
        writer = sqlite3.connect(target)
        try:
            reader.backup(writer)
            # The snapshot is a single file: no -wal or -shm to lose when it is moved around.
            writer.execute("PRAGMA journal_mode=DELETE")
            writer.commit()
        finally:
            writer.close()
    finally:
        reader.close()


def _check(snapshot: Path) -> None:
    """The snapshot opens, passes SQLite's integrity check and carries a migration revision
    (so it is one of our databases and not an empty file)."""
    connection = sqlite3.connect(snapshot)
    try:
        verdict = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if verdict != "ok":
            raise BackupError(f"the snapshot failed the integrity check: {verdict}")
        revisions = connection.execute("SELECT count(*) FROM alembic_version").fetchone()[0]
        if revisions != 1:
            raise BackupError("the snapshot has no migration revision; it is not our database")
    except sqlite3.DatabaseError as error:
        raise BackupError(f"the snapshot cannot be read: {error}") from error
    finally:
        connection.close()


def archives(out_dir: Path) -> list[Path]:
    """The archives in `out_dir`, oldest first (the timestamp in the name sorts as text)."""
    if not out_dir.is_dir():
        return []
    return sorted(p for p in out_dir.iterdir() if ARCHIVE_NAME.match(p.name))


def backup_database(
    database_url: str,
    out_dir: Path,
    *,
    keep: int = 28,
    now: Optional[datetime] = None,
) -> BackupResult:
    """Writes `trading-YYYYMMDD-HHMMSS.db.gz` (UTC) to `out_dir` and keeps the newest `keep`
    archives. Raises BackupError, and then changes nothing in `out_dir`."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    source = database_file(database_url)
    if not source.is_file():
        # sqlite3.connect would quietly create an empty database here and "back it up".
        raise BackupError(f"there is no database at {source}")

    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    name = f"trading-{moment:%Y%m%d-%H%M%S}.db.gz"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / name
    partial = out_dir / f".{name}.partial"

    try:
        with tempfile.TemporaryDirectory() as scratch:
            snapshot = Path(scratch) / "snapshot.db"
            try:
                _snapshot(source, snapshot)
            except sqlite3.DatabaseError as error:
                raise BackupError(f"could not copy {source}: {error}") from error
            _check(snapshot)
            # Compressed in the container's own disk, then moved onto the (possibly shared)
            # folder as one finished file.
            with open(snapshot, "rb") as raw, gzip.open(partial, "wb") as packed:
                shutil.copyfileobj(raw, packed)
        os.replace(partial, final)
    finally:
        partial.unlink(missing_ok=True)

    existing = archives(out_dir)
    removed = existing[:-keep] if len(existing) > keep else []
    for old in removed:
        old.unlink()
    logger.info(f"Database backed up to {final} ({final.stat().st_size} bytes)")
    return BackupResult(
        path=final,
        size_bytes=final.stat().st_size,
        removed=removed,
        kept=len(existing) - len(removed),
    )
