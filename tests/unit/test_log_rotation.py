"""The log file is rotated, and only the long-running loop writes it (Phase 6.3).

The tests' own conftest replaces setup_json_logger with a no-op so that tests never write the
real log; the real function is taken here, at import, before that happens."""

import logging
import logging.handlers
from pathlib import Path
from typing import Any

import pytest

from trading_system.cli import LOG_FILE, log_file_for, main
from trading_system.observability.logging import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    setup_json_logger as real_setup_json_logger,
)


@pytest.fixture
def logger_name() -> Any:
    name = "log-rotation-test"
    yield name
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def _rotating(logger: logging.Logger) -> logging.handlers.RotatingFileHandler:
    [handler] = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    return handler


def test_the_file_log_is_rotating_with_the_documented_limits(
    tmp_path: Path, logger_name: str
) -> None:
    logger = real_setup_json_logger(logger_name, log_file=str(tmp_path / "x.log"))

    handler = _rotating(logger)

    assert (handler.maxBytes, handler.backupCount) == (LOG_MAX_BYTES, LOG_BACKUP_COUNT)
    assert (LOG_MAX_BYTES, LOG_BACKUP_COUNT) == (5 * 1024 * 1024, 5)


def test_a_log_that_grows_past_its_limit_is_rotated_and_old_files_are_dropped(
    tmp_path: Path, logger_name: str
) -> None:
    path = tmp_path / "x.log"
    logger = real_setup_json_logger(logger_name, log_file=str(path), max_bytes=2000, backup_count=2)

    for n in range(300):
        logger.info(f"line {n:04d} " + "x" * 60)
    for handler in logger.handlers:
        handler.flush()

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["x.log", "x.log.1", "x.log.2"]  # no x.log.3: the oldest went
    assert all(
        p.stat().st_size <= 2000 + 400 for p in tmp_path.iterdir()
    )  # one record over at most
    newest = path.read_text(encoding="utf-8")
    assert "line 0299" in newest and "line 0000" not in "".join(
        p.read_text(encoding="utf-8") for p in tmp_path.iterdir()
    )


def test_a_log_without_a_file_has_no_file_handler(logger_name: str) -> None:
    logger = real_setup_json_logger(logger_name)

    assert not [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


def test_only_the_run_command_writes_the_log_file() -> None:
    assert log_file_for("run") == LOG_FILE
    for command in ("health", "status", "report", "backtest", "fetch-history", "backup-db", None):
        assert log_file_for(command) is None


def test_a_short_command_sets_up_the_logger_without_a_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "trading_system.observability.logging.setup_json_logger",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "nothing.json"))

    main(["health"])

    assert calls == [{"level": logging.INFO, "log_file": None}]
