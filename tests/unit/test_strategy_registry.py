"""The strategies that can be chosen by name, and the contract every one of them keeps."""

from datetime import datetime, timezone

import pytest

from trading_system.strategy.base import BaseStrategy
from trading_system.strategy.config import RUNTIME_STRATEGY
from trading_system.strategy.registry import STRATEGIES, get_strategy, strategy_names
from trading_system.strategy.trend_cross import TrendCrossStrategy


def test_the_strategy_the_live_loop_runs_is_in_the_list_with_its_live_parameters() -> None:
    strategy = get_strategy("trend_cross")

    assert isinstance(strategy, TrendCrossStrategy)
    assert strategy.config == RUNTIME_STRATEGY


def test_the_live_one_the_two_candidates_the_two_originals_and_their_variants_are_there() -> None:
    assert strategy_names() == [
        "trend_cross",
        "trend_cross_session",
        "trend_cross_session_late",
        "ema_crossover",
        "strict_ema_crossover",
        "ema_crossover_5pct",
        "strict_ema_crossover_5pct",
    ]


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_strategy_keeps_the_contract(name: str) -> None:
    strategy = get_strategy(name)

    assert isinstance(strategy, BaseStrategy)
    assert strategy.name == name
    assert strategy.version
    # nothing to judge, nothing to say (and no crash on empty input)
    assert (
        strategy.generate_signals({}, {}, datetime(2026, 11, 2, 16, 0, tzinfo=timezone.utc)) == []
    )


def test_the_versions_tell_the_strategies_and_their_parameters_apart() -> None:
    versions = [get_strategy(name).version for name in STRATEGIES]

    assert len(set(versions)) == len(versions)


def test_an_unknown_name_lists_the_choices() -> None:
    with pytest.raises(ValueError, match="unknown strategy 'nope'.*trend_cross.*ema_crossover"):
        get_strategy("nope")


def test_a_strategy_without_the_required_parts_cannot_be_built() -> None:
    with pytest.raises(TypeError):
        BaseStrategy()  # type: ignore[abstract]
