"""The signals under test: for each day and stock, "on" or not, from the prices up to that day's
close and nothing later. A signal is a table of True/False with the shape of the panel.

The first three were chosen with the owner on 2026-11-05 (docs/strategy-research-plan.md,
question 3). Every one added here, and every setting tried, is one more variant in the count
(research/trials.py): add them rarely.

Looking only at the past is not left to care: tests/unit/test_research_signals.py changes the
future of a panel and checks that no signal of an earlier day moves.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from trading_system.research.panel import Panel

# A day with fewer stocks than this that have a value gives no signal at all: a "top tenth" of a
# handful of names is not a ranking.
MIN_RANKED = 50

DECILE = 0.10
MOMENTUM_WINDOW = 231  # 252 trading days back to 21 back: a year without the last month
MOMENTUM_SKIP = 21
REVERSAL_WINDOW = 5  # one week
CONTROL_SEED = 20261105


@dataclass(frozen=True)
class Signal:
    name: str
    description: str
    horizons: tuple[int, ...]  # trading days held, measured from the next open
    build: Callable[[Panel], pd.DataFrame]
    control: bool = False  # a placebo that must show nothing; not counted as a trial


def _rank(values: pd.DataFrame) -> pd.DataFrame:
    """Each stock's place among the stocks of its day, 0 to 1, missing where it has no value or
    the day has too few stocks."""
    ranks = values.rank(axis=1, pct=True)
    thin = values.notna().sum(axis=1) < MIN_RANKED
    return ranks.where(~thin, np.nan, axis=0)


def momentum_12_1(panel: Panel) -> pd.DataFrame:
    """The tenth of the stocks that rose most over the last year, leaving out the last month."""
    formation = (
        panel.log_return.rolling(MOMENTUM_WINDOW, min_periods=MOMENTUM_WINDOW)
        .sum()
        .shift(MOMENTUM_SKIP)
    )
    return _rank(formation) > 1 - DECILE


def reversal_1w(panel: Panel) -> pd.DataFrame:
    """The tenth of the stocks that fell most over the last five trading days."""
    week = panel.log_return.rolling(REVERSAL_WINDOW, min_periods=REVERSAL_WINDOW).sum()
    return _rank(week) <= DECILE


def random_tenth(panel: Panel) -> pd.DataFrame:
    """A tenth of the stocks, drawn at random each day. The control: whatever the measurement says
    about it is what chance says, and the measurement must say about zero."""
    draw = np.random.default_rng(CONTROL_SEED).random(panel.close.shape)
    return pd.DataFrame(
        (draw < DECILE) & panel.close.notna().to_numpy(),
        index=panel.close.index,
        columns=panel.close.columns,
    )


SIGNALS: dict[str, Signal] = {
    signal.name: signal
    for signal in (
        Signal(
            "momentum_12_1",
            "top tenth by the return of the last year without the last month",
            (10, 15, 20),
            momentum_12_1,
        ),
        Signal(
            "reversal_1w",
            "bottom tenth by the return of the last five trading days",
            (1, 2, 3, 5),
            reversal_1w,
        ),
        Signal(
            "random_tenth",
            "a random tenth of the stocks each day (control)",
            (1, 5),
            random_tenth,
            control=True,
        ),
    )
}

REGIME_FILTERS = ("spy200",)


def with_regime_filter(on: pd.DataFrame, panel: Panel, name: str) -> pd.DataFrame:
    """`on` kept only on the days the market is in the named regime: "spy200" = SPY closes above its
    200-day average (known at that close)."""
    if name != "spy200":
        raise ValueError(f"unknown regime filter {name!r}; known: {', '.join(REGIME_FILTERS)}")
    in_regime = panel.market_up.reindex(on.index).fillna(False).astype(bool)
    return on.where(in_regime, False, axis=0).astype(bool)
