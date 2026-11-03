"""Choosing a strategy for the backtest and comparing several (Phase 3.3a)."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from tests.fixtures.bars import random_walk_session_bars
from trading_system.backtest.comparison import compare_strategies, format_comparison
from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.models import BacktestConfig
from trading_system.cli import main
from trading_system.storage.repositories import MarketBarRepository
from trading_system.strategy.registry import get_strategy


@pytest.fixture
def stored(settings: Any, session_factory: Any) -> Any:
    """Eighteen weekday sessions of a random walk for AAPL in the store."""
    bars_15m, bars_1h = random_walk_session_bars(31, "2026-10-05", 18, drift=0.0003)
    repo = MarketBarRepository(session_factory)
    now = datetime.now(timezone.utc)
    repo.replace("AAPL", "15m", bars_15m, adjustment="all", fetched_at=now)
    repo.replace("AAPL", "1h", bars_1h, adjustment="all", fetched_at=now)
    return {"AAPL": bars_15m}, {"AAPL": bars_1h}


def test_compare_runs_the_same_data_and_costs_for_every_strategy(
    settings: Any, stored: Any
) -> None:
    frames_15m, frames_1h = stored
    config = BacktestConfig()

    rows = compare_strategies(
        settings, config, frames_15m, frames_1h, ["ema_crossover", "trend_cross"]
    )

    assert [r.name for r in rows] == ["ema_crossover", "trend_cross"]  # in the order asked
    alone = BacktestEngine(
        settings, config, frames_15m, frames_1h, strategy=get_strategy("trend_cross")
    ).run()
    trend = rows[1]
    assert (trend.signals, trend.trades) == (alone.signals, alone.trade_count)
    assert trend.total_return_pct == alone.total_return_pct
    assert trend.version == get_strategy("trend_cross").version
    assert trend.costs == alone.total_fees + alone.total_slippage
    # 18 days hold far fewer than the 200 hourly bars the original's EMA 200 needs: it is silent
    assert rows[0].signals == 0 and rows[0].trades == 0


def test_an_unknown_strategy_stops_the_comparison_before_any_run(
    settings: Any, stored: Any
) -> None:
    frames_15m, frames_1h = stored

    with pytest.raises(ValueError, match="unknown strategy"):
        compare_strategies(settings, BacktestConfig(), frames_15m, frames_1h, ["trend_cross", "x"])


def test_the_table_has_a_row_per_strategy(settings: Any, stored: Any) -> None:
    frames_15m, frames_1h = stored
    rows = compare_strategies(
        settings, BacktestConfig(), frames_15m, frames_1h, ["trend_cross", "ema_crossover"]
    )

    table = format_comparison(rows).splitlines()

    assert table[0].split()[:3] == ["strategy", "signals", "trades"]
    assert [line.split()[0] for line in table[2:]] == ["trend_cross", "ema_crossover"]


def test_compare_strategies_command(
    settings: Any, stored: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("trading_system.cli.Settings", return_value=settings):
        code = main(
            ["compare-strategies", "--days", "3650", "--strategies", "trend_cross,ema_crossover"]
        )

    out = capsys.readouterr().out
    assert code == 0
    assert "trend_cross" in out and "ema_crossover" in out
    assert "lucky one" in out  # the table comes with the warning about reading it


def test_compare_strategies_command_names_the_choices_for_an_unknown_one(
    settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("trading_system.cli.Settings", return_value=settings):
        code = main(["compare-strategies", "--strategies", "nope"])

    out = capsys.readouterr().out
    assert code == 1
    assert "unknown strategy nope" in out and "strict_ema_crossover_5pct" in out


def test_backtest_replays_the_strategy_asked_for(
    settings: Any, stored: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("trading_system.cli.Settings", return_value=settings):
        code = main(["backtest", "--days", "3650", "--strategy", "strict_ema_crossover_5pct"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Strategy: strict_ema_crossover_5pct (strict-ema-crossover-20-50-5pct-v1)" in out


def test_backtest_with_an_unknown_strategy_says_which_exist(
    settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("trading_system.cli.Settings", return_value=settings):
        code = main(["backtest", "--strategy", "nope"])

    assert code == 1
    assert "unknown strategy 'nope'" in capsys.readouterr().out


def test_t_statistic_is_the_average_over_its_standard_error() -> None:
    from trading_system.backtest.comparison import _t_stat

    assert _t_stat([1.0, 2.0, 3.0]) == pytest.approx(2.0 / (1.0 / 3**0.5))
    assert _t_stat([0.5, 0.5, 0.5]) == 0.0  # no spread: no statement
    assert _t_stat([1.0]) == 0.0 and _t_stat([]) == 0.0  # too few trades to say anything
    assert _t_stat([-1.0, -2.0, -3.0]) < 0


def test_the_table_stays_aligned_for_the_longest_name(settings: Any, stored: Any) -> None:
    frames_15m, frames_1h = stored
    rows = compare_strategies(
        settings,
        BacktestConfig(),
        frames_15m,
        frames_1h,
        ["trend_cross", "strict_ema_crossover_5pct"],
    )

    lines = format_comparison(rows).splitlines()

    assert len({len(line) for line in lines}) == 1  # header, rule and rows are equally wide
    assert lines[0].rstrip().endswith("costs $") and " t " in lines[0]
