"""The signals: they pick whom they should, look only at the past, and the control is neutral."""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.research import frames_from_returns, noise_returns
from trading_system.research.panel import Panel, build_panel
from trading_system.research.signals import (
    MIN_RANKED,
    SIGNALS,
    momentum_12_1,
    random_tenth,
    reversal_1w,
    with_regime_filter,
)


def _panel(returns: pd.DataFrame) -> Panel:
    return build_panel(frames_from_returns(returns), artifacts={})


def test_momentum_picks_the_tenth_that_rose_most_over_the_year_without_the_last_month() -> None:
    returns = noise_returns(days=400, stocks=60, seed=2, vol=0.002)
    winners = list(returns.columns[:6])
    returns.loc[returns.index[100:350], winners] += 0.004  # a strong, steady year
    losers = list(returns.columns[6:12])
    returns.loc[returns.index[100:350], losers] -= 0.004

    on = momentum_12_1(_panel(returns))

    last = on.iloc[-1]
    assert set(last[last].index) == set(winners)


def test_momentum_ignores_the_last_month() -> None:
    """A stock that jumped only in the last 20 days is not a 12-1 winner; that month is skipped."""
    returns = noise_returns(days=400, stocks=60, seed=4, vol=0.002)
    jumper = returns.columns[0]
    returns.loc[returns.index[-20:], jumper] += 0.02  # +40 % in the last month

    on = momentum_12_1(_panel(returns))

    assert not on[jumper].iloc[-1]


def test_reversal_picks_the_tenth_that_fell_most_over_the_last_five_days() -> None:
    returns = noise_returns(days=100, stocks=60, seed=3, vol=0.002)
    fallers = list(returns.columns[:6])
    returns.loc[returns.index[-5:], fallers] -= 0.03

    on = reversal_1w(_panel(returns))

    last = on.iloc[-1]
    assert set(last[last].index) == set(fallers)


def test_a_day_with_too_few_stocks_gives_no_signal_at_all() -> None:
    returns = noise_returns(days=300, stocks=MIN_RANKED - 1, seed=6)

    panel = _panel(returns)

    assert not momentum_12_1(panel).to_numpy().any()
    assert not reversal_1w(panel).to_numpy().any()


def test_the_control_takes_about_a_tenth_and_is_the_same_every_time() -> None:
    panel = _panel(noise_returns(days=300, stocks=60, seed=7))

    first, second = random_tenth(panel), random_tenth(panel)

    assert first.equals(second)
    assert 0.08 < first.to_numpy().mean() < 0.12


@pytest.mark.parametrize("name", ["momentum_12_1", "reversal_1w", "random_tenth"])
def test_a_signal_never_looks_at_the_future(name: str) -> None:
    """Two panels that are the same up to day 330 and then differ: the flags up to day 330 must
    be identical. A signal that used a later price would differ."""
    cut = 330
    base = noise_returns(days=400, stocks=60, seed=8)
    other = base.copy()
    other.iloc[cut + 1 :] = np.random.default_rng(99).normal(0.0, 0.05, other.iloc[cut + 1 :].shape)

    on_base = SIGNALS[name].build(_panel(base))
    on_other = SIGNALS[name].build(_panel(other))

    assert on_base.iloc[: cut + 1].equals(on_other.iloc[: cut + 1])
    assert not on_base.iloc[cut + 1 :].equals(on_other.iloc[cut + 1 :]) or name == "random_tenth"


def test_the_regime_filter_looks_only_at_the_past_too() -> None:
    """SPY climbs steadily, so on the cut day it is above its average. In the other panel the day
    after the cut is a 40 % crash, which would put it below. A regime that peeked at the next close
    would differ on the cut day; the real one cannot."""
    cut = 330
    base = noise_returns(days=400, stocks=60, seed=9)
    base["SPY"] = 0.001
    other = base.copy()
    other.iloc[cut + 1, other.columns.get_loc("SPY")] = -0.5

    panels = [_panel(returns) for returns in (base, other)]
    flags = [with_regime_filter(reversal_1w(p), p, "spy200") for p in panels]

    assert panels[0].market_up.iloc[cut]  # the premise: above its average on the cut day
    assert panels[0].market_up.iloc[: cut + 1].equals(panels[1].market_up.iloc[: cut + 1])
    assert flags[0].iloc[: cut + 1].equals(flags[1].iloc[: cut + 1])
    assert flags[0].iloc[cut].any()  # and the comparison was about something


def test_the_regime_filter_keeps_the_signal_only_while_spy_is_above_its_average() -> None:
    returns = noise_returns(days=400, stocks=60, seed=10)
    returns.loc[
        returns.index[250:], "SPY"
    ] = -0.01  # a steady fall: below the average after a while
    panel = _panel(returns)
    everything = pd.DataFrame(True, index=panel.close.index, columns=panel.close.columns)

    filtered = with_regime_filter(everything, panel, "spy200")

    assert filtered.iloc[:199].to_numpy().sum() == 0  # no average yet
    assert filtered.iloc[-1].to_numpy().sum() == 0  # down at the end
    assert (filtered.any(axis=1) == panel.market_up).all()
    assert filtered.dtypes.eq(bool).all()


def test_an_unknown_regime_is_an_error_not_a_silent_no_filter() -> None:
    panel = _panel(noise_returns(days=50, stocks=3))

    with pytest.raises(ValueError, match="unknown regime filter"):
        with_regime_filter(reversal_1w(panel), panel, "spy100")
