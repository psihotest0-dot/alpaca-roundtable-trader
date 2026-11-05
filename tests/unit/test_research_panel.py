"""The panel: aligned tables, the hold-out kept out, the known artifact cut out."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.research import frames_from_returns, noise_returns
from trading_system.research.measure import forward_returns
from trading_system.research.panel import ARTIFACTS, big_moves, build_panel


def _tiny_frames(closes: list[float], opens: list[float]) -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2024-01-02", periods=len(closes))
    frame = pd.DataFrame({"open": opens, "close": closes}, index=index)
    return {"AAA": frame, "SPY": frame}


def test_the_forward_return_is_from_the_next_open_to_the_close_of_the_last_day() -> None:
    #            day 0   day 1   day 2   day 3
    closes = [100.0, 102.0, 105.0, 104.0]
    opens = [100.0, 101.0, 103.0, 106.0]
    panel = build_panel(_tiny_frames(closes, opens), artifacts={})

    one_day, _ = forward_returns(panel, 1)
    two_days, _ = forward_returns(panel, 2)

    # signal at the close of day 0: in at the open of day 1 (101), out at the close of day 1 or 2
    assert one_day["AAA"].iloc[0] == pytest.approx(102.0 / 101.0 - 1)
    assert two_days["AAA"].iloc[0] == pytest.approx(105.0 / 101.0 - 1)
    assert two_days["AAA"].iloc[1] == pytest.approx(104.0 / 103.0 - 1)
    assert np.isnan(two_days["AAA"].iloc[2]) and np.isnan(two_days["AAA"].iloc[3])  # no such day


def test_a_blanked_day_blanks_every_window_that_contains_it() -> None:
    returns = noise_returns(days=40, stocks=3, seed=3)
    frames = frames_from_returns(returns)
    victim = returns.columns[0]
    day = returns.index[20].date()

    panel = build_panel(frames, artifacts={victim: (day,)})
    forward, _ = forward_returns(panel, 5)

    assert np.isnan(panel.log_return.loc[pd.Timestamp(day), victim])
    assert np.isnan(panel.overnight_gap.loc[pd.Timestamp(day), victim])
    # a window that holds day 20 (entry at the open of day k+1, exit at the close of day k+5)
    windows_with_it = forward[victim].iloc[15:20]
    assert windows_with_it.isna().all()
    assert forward[victim].iloc[:15].notna().all()  # earlier windows are untouched
    assert forward[returns.columns[1]].iloc[15:20].notna().all()  # and so are the other stocks


def test_the_known_artifact_is_registered_for_rtx() -> None:
    assert ARTIFACTS["RTX"] == (date(2020, 4, 3),)


def test_a_panel_needs_the_market() -> None:
    returns = noise_returns(days=30, stocks=3).drop(columns="SPY")

    with pytest.raises(ValueError, match="fetch-daily"):
        build_panel(frames_from_returns(returns))


def test_regime_is_unknown_until_the_200_day_average_exists() -> None:
    panel = build_panel(frames_from_returns(noise_returns(days=260, stocks=3)))

    assert not panel.market_known.iloc[:199].any() and panel.market_known.iloc[199:].all()
    assert not panel.market_up.iloc[:199].any()  # unknown counts as "not up"


def test_big_moves_lists_what_is_left_and_not_what_was_blanked() -> None:
    returns = noise_returns(days=40, stocks=3, seed=5)
    returns.iloc[10, 0] = -1.2  # -70 %
    returns.iloc[20, 1] = 0.5  # +65 %
    frames = frames_from_returns(returns)
    blanked = returns.index[10].date()

    panel = build_panel(frames, artifacts={returns.columns[0]: (blanked,)})
    found = big_moves(panel)

    assert [(s, d) for s, d, _ in found] == [(returns.columns[1], returns.index[20].date())]
