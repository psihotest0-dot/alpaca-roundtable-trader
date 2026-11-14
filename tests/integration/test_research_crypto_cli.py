"""The crypto commands end to end on a made-up file: they measure, print, count apart from the stocks
and leave the stock research alone (plan section 15)."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading_system.cli import main
from trading_system.data.crypto import CRYPTO_UNIVERSE, MARKET_COIN
from trading_system.research.panel import HOLDOUT_START
from trading_system.research.trials import CRYPTO, STOCKS, trial_counts
from trading_system.storage.repositories import AuditRepository
from trading_system.storage.research import open_crypto_store

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)


def _fill(store: Any, days: int = 2000, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=days, freq="D", tz="UTC").rename("timestamp")
    for symbol in CRYPTO_UNIVERSE:
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, days)))
        frame = pd.DataFrame(
            {
                "open": np.concatenate([[100.0], closes[:-1]]),
                "close": closes,
                "volume": 1000.0,
            },
            index=index,
        )
        frame["high"] = frame[["open", "close"]].max(axis=1) * 1.001
        frame["low"] = frame[["open", "close"]].min(axis=1) * 0.999
        store.replace(symbol, "1d", frame, adjustment="none", fetched_at=NOW)


@pytest.fixture
def crypto_url(tmp_path: Any) -> str:
    return f"sqlite:///{(tmp_path / 'crypto.db').as_posix()}"


@pytest.fixture
def filled(crypto_url: str, db_url: str) -> None:
    _fill(open_crypto_store(crypto_url, db_url))


def _measure(settings: Any, *argv: str) -> int:
    with patch("trading_system.cli.Settings", return_value=settings):
        return main(["measure-crypto", *argv])


def test_the_command_measures_prints_and_counts_on_counter_c(
    filled: None,
    make_settings: Any,
    crypto_url: str,
    session_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = make_settings(crypto_database_url=crypto_url)

    assert _measure(settings, "--signal", "crypto_trend_30") == 0

    out = capsys.readouterr().out
    assert "Development part only" in out and str(HOLDOUT_START) in out
    assert "7 coins" in out and "0.50% a round trip" in out
    assert "crypto_trend_30, held 10 trading days" in out
    assert "crypto_trend_30, held 20 trading days" in out
    assert "vs BTC" in out and "BTC above its 200-day average" in out
    assert "days used:" in out and "300+ events" in out
    assert "Crypto variants tried so far (counter C): 2 (2 runs)." in out
    audit = AuditRepository(session_factory)
    assert trial_counts(audit, CRYPTO) == (2, 2)
    assert trial_counts(audit, STOCKS) == (0, 0)  # the 20 stock variants are not touched
    last_day = out.split("7 coins, ")[1].split(" to ")[1].split(" ")[0]
    assert datetime.fromisoformat(last_day).date() < HOLDOUT_START  # the hold-out stayed closed


def test_a_control_is_measured_but_not_counted(
    filled: None,
    make_settings: Any,
    crypto_url: str,
    session_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = make_settings(crypto_database_url=crypto_url)

    assert _measure(settings, "--signal", "random_half") == 0

    assert "Control, not counted as a variant." in capsys.readouterr().out
    assert trial_counts(AuditRepository(session_factory), CRYPTO) == (0, 0)


def test_the_same_variant_again_is_a_run_not_a_new_variant(
    filled: None, make_settings: Any, crypto_url: str, session_factory: Any
) -> None:
    settings = make_settings(crypto_database_url=crypto_url)
    _measure(settings, "--signal", "crypto_trend_30", "--horizons", "10")
    _measure(settings, "--signal", "crypto_trend_30", "--horizons", "10")

    assert trial_counts(AuditRepository(session_factory), CRYPTO) == (2, 1)


def test_an_unknown_signal_and_an_empty_file_are_refused(
    make_settings: Any, crypto_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(crypto_database_url=crypto_url)

    assert _measure(settings, "--signal", "momentum_12_1") == 1  # a stock signal is not offered
    assert "unknown signal" in capsys.readouterr().out
    assert _measure(settings, "--signal", "crypto_trend_30") == 1  # nothing fetched yet
    assert "fetch-crypto-daily" in capsys.readouterr().out


def test_the_crypto_file_cannot_be_the_research_file(
    make_settings: Any, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    same = f"sqlite:///{(tmp_path / 'research.db').as_posix()}"
    settings = make_settings(crypto_database_url=same, research_database_url=same)

    assert _measure(settings, "--signal", "crypto_trend_30") == 1
    assert "stock measurements" in capsys.readouterr().out


def test_fetch_crypto_daily_stores_the_coins_without_a_key(
    make_settings: Any, crypto_url: str, db_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.unit.test_research_crypto import _FakeClient, _alpaca_shaped

    settings = make_settings(crypto_database_url=crypto_url)
    from trading_system.data.crypto import CryptoDailyFeed

    class OffLine(CryptoDailyFeed):
        def __init__(self) -> None:
            super().__init__(_FakeClient(_alpaca_shaped(30)))

    with (
        patch("trading_system.cli.Settings", return_value=settings),
        patch("trading_system.data.crypto.CryptoDailyFeed", OffLine),
    ):
        assert main(["fetch-crypto-daily"]) == 0

    out = capsys.readouterr().out
    assert "no key needed" in out and "rows written" in out
    stored = open_crypto_store(crypto_url, db_url).load([MARKET_COIN], "1d")
    assert len(stored[MARKET_COIN]) > 0
