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
