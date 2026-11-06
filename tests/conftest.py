"""Fixtures every test gets."""

import pytest


@pytest.fixture(autouse=True)
def tests_do_not_write_the_real_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """cli.main() sets up the JSON logger with a file handler on var/alpaca_trading.log, and
    the handler stays on the root logger for the rest of the test session. Every test that
    called main() then wrote its own log lines into the log of the real system, between the
    lines of a running dry-run session. Tests that need log output use caplog instead."""
    monkeypatch.setattr(
        "trading_system.observability.logging.setup_json_logger", lambda *args, **kwargs: None
    )


@pytest.fixture(autouse=True)
def the_oracle_is_off_whatever_the_env_file_says(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Settings()` reads the `.env` of the working directory, and in a test run that is the
    owner's real one. Once it switches the backend Oracle on and holds its token, every test that
    builds `Settings()` (the `health` command, `doctor`) would call the real backend. An
    environment variable beats the file, so this keeps them off; a test that wants the Oracle
    passes `oracle_enabled=True` itself or sets the variable after this fixture has run."""
    monkeypatch.setenv("ORACLE_ENABLED", "false")
