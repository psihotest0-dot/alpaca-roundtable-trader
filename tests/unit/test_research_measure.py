"""The measurement: it finds a planted effect, finds nothing in noise, and says honestly when a
result rests on one year."""

import numpy as np
import pandas as pd

from tests.fixtures.research import frames_from_returns, noise_returns
from trading_system.research.measure import (
    CRITERIA_MIN_INDEPENDENT_EVENTS,
    Measurement,
    format_measurement,
    measure,
)
from trading_system.research.panel import Panel, build_panel
from trading_system.research.signals import random_tenth, reversal_1w


def _panel(returns: pd.DataFrame) -> Panel:
    return build_panel(frames_from_returns(returns), artifacts={})


def _planted(seed: int, bonus: float, *, days: int = 1000) -> tuple[Panel, pd.DataFrame]:
    """Noise, plus a random tenth of the stocks "on" each day, plus `bonus` added to the return
    of the NEXT day for those stocks (inside the day: it opens where the last one closed)."""
    returns = noise_returns(days=days, stocks=60, seed=seed)
    on = random_tenth(_panel(returns))
    extra = on.shift(1, fill_value=False).astype(float) * bonus
    returns = returns + extra.reindex(columns=returns.columns, fill_value=0.0)
    return _panel(returns), on


def test_a_planted_one_day_effect_is_found_with_about_the_right_size() -> None:
    panel, on = _planted(seed=1, bonus=0.01)

    m = measure(panel, on, signal="planted", horizon=1)

    # the signal stocks gain 1 % that the other ninety percent do not: 1 % x (1 - 0.1) over all
    assert 0.007 < m.vs_universe.mean < 0.011
    assert m.vs_universe.t > 10 and m.net.t > 10
    assert m.criteria.enough_events and m.criteria.t_high_enough


def test_noise_has_no_effect_and_does_not_pass() -> None:
    worst = 0.0
    for seed in range(8):
        returns = noise_returns(days=1000, stocks=60, seed=seed)
        panel = _panel(returns)
        for horizon in (1, 5, 20):
            m = measure(panel, random_tenth(panel), signal="control", horizon=horizon)
            worst = max(worst, abs(m.vs_universe.t))
            assert not m.criteria.all_met

    assert worst < 3.5  # chance alone does not reach the locked threshold of 3 here


def test_a_cost_larger_than_the_edge_leaves_nothing_net() -> None:
    panel, on = _planted(seed=2, bonus=0.002)  # an edge of about 0.18 %

    cheap = measure(panel, on, signal="planted", horizon=1, cost=0.0)
    dear = measure(panel, on, signal="planted", horizon=1, cost=0.005)

    assert cheap.net.mean > 0.0010 and dear.net.mean < 0
    assert dear.vs_universe.mean == cheap.vs_universe.mean  # the cost is only taken off the net


def test_a_result_that_rests_on_one_year_is_not_a_candidate() -> None:
    """The effect exists in one calendar year only. Over the whole span it is strong, but without that
    year nothing is left, and the criteria say so."""
    returns = noise_returns(days=1000, stocks=60, seed=3)
    on = random_tenth(_panel(returns))
    extra = on.shift(1, fill_value=False).astype(float) * 0.01
    extra[extra.index.year != 2024] = 0.0  # the effect exists in 2024 and in no other year
    panel = _panel(returns + extra.reindex(columns=returns.columns, fill_value=0.0))

    m = measure(panel, on, signal="one_year", horizon=1)

    assert m.net.t > 3 and m.criteria.enough_events
    assert not m.criteria.not_one_year
    assert not m.criteria.all_met


def test_a_small_edge_with_plenty_of_events_still_fails_on_t() -> None:
    """A real edge of 0.03 % a day, positive in most years and in plenty of events, yet too small to
    tell from chance in four years of data: the t criterion is the only one that stops it."""
    panel, on = _planted(seed=6, bonus=0.0003)

    m = measure(panel, on, signal="planted", horizon=1, cost=0.0)

    assert m.net.mean > 0 and 0.5 < m.net.t < 3
    assert m.criteria.enough_events and m.criteria.not_one_year and m.criteria.most_years_positive
    assert not m.criteria.t_high_enough and not m.criteria.all_met


def test_the_last_days_have_no_forward_return_so_they_are_never_an_observation() -> None:
    panel = _panel(noise_returns(days=300, stocks=60, seed=5))

    m = measure(panel, random_tenth(panel), signal="control", horizon=20)

    assert m.last is not None and m.last <= panel.dates[-21].date()


def test_the_regime_table_splits_the_same_days() -> None:
    returns = noise_returns(days=800, stocks=60, seed=6)
    returns["SPY"] += 0.0005 * np.sin(np.arange(len(returns)) / 60.0)  # up and down stretches
    panel = _panel(returns)

    m = measure(panel, reversal_1w(panel), signal="reversal_1w", horizon=1)

    above, below = m.by_regime
    assert above.dates > 0 and below.dates > 0
    assert above.dates + below.dates <= m.vs_universe.dates  # no day counted twice


def test_events_are_counted_per_stock_and_per_non_overlapping_window() -> None:
    panel = _panel(noise_returns(days=400, stocks=60, seed=7))
    on = random_tenth(panel)

    one = measure(panel, on, signal="control", horizon=1)
    ten = measure(panel, on, signal="control", horizon=10)

    assert one.vs_universe.events > ten.vs_universe.events * 0.9  # about the same stock-events
    assert ten.independent_events < one.independent_events / 5  # but ten days each


def test_the_report_says_what_it_measured_and_what_it_found() -> None:
    panel, on = _planted(seed=4, bonus=0.01, days=600)

    text = format_measurement(measure(panel, on, signal="planted", horizon=3))

    assert text.startswith("planted, held 3 trading days")
    for part in ("vs all stocks", "vs SPY", "net of 0.10% cost", "by year (net)", "criteria"):
        assert part in text
    assert f"{CRITERIA_MIN_INDEPENDENT_EVENTS}+ events" in text


def test_a_panel_without_enough_stocks_says_so_instead_of_inventing_a_number() -> None:
    panel = _panel(noise_returns(days=300, stocks=20, seed=8))

    m: Measurement = measure(panel, random_tenth(panel), signal="control", horizon=5)

    assert not m.usable and not m.criteria.all_met
    assert "no day has enough stocks" in format_measurement(m)
