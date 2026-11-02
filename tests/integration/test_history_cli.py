"""`fetch-history` fills the store; `backtest` replays what is in it (Phase 3.2)."""

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.bars import make_bars
from tests.fixtures.fakes import FakeHistoryClient
from trading_system.cli import main
from trading_system.data.alpaca_data import AlpacaData
from trading_system.storage.repositories import MarketBarRepository


@pytest.fixture
def alpaca(monkeypatch: pytest.MonkeyPatch, settings: Any) -> Any:
    """Plugs a fake Alpaca into the command, built from the real (newest) fixture bars."""

    def install(client: FakeHistoryClient) -> None:
        def build(_settings: Any) -> AlpacaData:
            data = AlpacaData(settings)
            data.client = client
            return data

        monkeypatch.setattr("trading_system.data.alpaca_data.AlpacaData", build)

    return install


def _client(now: datetime) -> FakeHistoryClient:
    bars_15m, bars_1h = make_bars(now)
    return FakeHistoryClient({15: {"AAPL": bars_15m}, 60: {"AAPL": bars_1h}})


def test_fetch_history_fills_the_store_and_says_what_it_did(
    alpaca: Any, settings: Any, session_factory: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    alpaca(_client(datetime.now(timezone.utc)))

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["fetch-history", "--days", "10"]) == 0

    out = capsys.readouterr().out
    assert "Updating 1 symbols, 10 days back" in out
    assert "15m:" in out and "1h:" in out and "rows written" in out
    store = MarketBarRepository(session_factory)
    assert "AAPL" in store.load(["AAPL"], "15m") and "AAPL" in store.load(["AAPL"], "1h")


def test_fetch_history_reports_a_symbol_alpaca_does_not_know(
    alpaca: Any, settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    alpaca(_client(datetime.now(timezone.utc)))

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["fetch-history", "--days", "5", "--symbols", "aapl, zzzz"]) == 0

    assert "No data from Alpaca for: ZZZZ" in capsys.readouterr().out


def test_fetch_history_fails_cleanly_when_alpaca_does_(
    monkeypatch: pytest.MonkeyPatch, settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(_settings: Any) -> Any:
        raise RuntimeError("no network")

    monkeypatch.setattr("trading_system.data.alpaca_data.AlpacaData", broken)

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["fetch-history"]) == 1

    assert "fetch-history error: no network" in capsys.readouterr().out


def test_backtest_with_an_empty_store_says_to_fetch_history_first(
    settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["backtest"]) == 1

    assert "Run `fetch-history` first" in capsys.readouterr().out


def test_backtest_replays_the_stored_candles(
    alpaca: Any, settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    alpaca(_client(datetime.now(timezone.utc)))
    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["fetch-history", "--days", "10"]) == 0
        capsys.readouterr()

        assert main(["backtest", "--days", "10"]) == 0

    out = capsys.readouterr().out
    assert "Replaying 1 symbols" in out
    assert "=== Backtest Report ===" in out
    assert "RiskEngine" in out  # the report says plainly that the sizing is not the live one yet


def _long_history(now: datetime) -> FakeHistoryClient:
    """Twenty days of rising bars, the newest one just closed."""
    last_15m = pd.Timestamp(now).floor("15min") - pd.Timedelta(minutes=15)
    last_1h = pd.Timestamp(now).floor("h") - pd.Timedelta(hours=1)

    def bars(last: pd.Timestamp, periods: int, freq: str) -> pd.DataFrame:
        closes = 100 + np.arange(periods) * 0.01
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": closes + 0.1,
                "low": closes - 0.1,
                "close": closes,
                "volume": 1000.0,
            },
            index=pd.date_range(end=last, periods=periods, freq=freq, tz="UTC"),
        )
        frame.index.name = "timestamp"
        return frame

    return FakeHistoryClient(
        {15: {"AAPL": bars(last_15m, 20 * 96, "15min")}, 60: {"AAPL": bars(last_1h, 20 * 24, "1h")}}
    )


def test_backtest_only_replays_the_days_asked_for(
    alpaca: Any, settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    now = datetime.now(timezone.utc)
    alpaca(_long_history(now))
    with patch("trading_system.cli.Settings", return_value=settings):
        main(["fetch-history", "--days", "18"])  # the store holds 18 days
        capsys.readouterr()

        assert main(["backtest", "--days", "3"]) == 0

    replayed = re.search(r"Replaying 1 symbols, (\d{4}-\d{2}-\d{2}) to", capsys.readouterr().out)
    assert replayed is not None
    first = datetime.strptime(replayed.group(1), "%Y-%m-%d").date()
    assert first >= (now - timedelta(days=3)).date()  # not the 18 days that are stored
