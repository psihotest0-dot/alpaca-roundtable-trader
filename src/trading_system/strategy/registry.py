"""The strategies that can be chosen by name (`backtest --strategy NAME`, `compare-strategies`).

The live loop does not choose from here: it runs `trend_cross` with RUNTIME_STRATEGY. Changing
what trades is a separate decision that has to be earned in the backtest first (Phase 4)."""

from typing import Callable, Dict, List

from trading_system.strategy.base import BaseStrategy
from trading_system.strategy.config import RUNTIME_STRATEGY
from trading_system.strategy.ema_crossover import (
    EMA_CROSSOVER,
    EMA_CROSSOVER_5PCT,
    STRICT_EMA_CROSSOVER,
    STRICT_EMA_CROSSOVER_5PCT,
    EmaCrossoverStrategy,
)
from trading_system.strategy.dip_cross import DipInUptrendStrategy
from trading_system.strategy.session_cross import SessionTrendCrossStrategy
from trading_system.strategy.trend_cross import TrendCrossStrategy

STRATEGIES: Dict[str, Callable[[], BaseStrategy]] = {
    # What the live loop runs.
    "trend_cross": lambda: TrendCrossStrategy(RUNTIME_STRATEGY),
    # Two candidates to repair it, registered before any result (research plan, section 12):
    # the same strategy on regular-session bars only, and the same without entries in the first
    # 30 minutes. The live loop does not run them.
    "trend_cross_session": lambda: SessionTrendCrossStrategy("trend_cross_session"),
    "trend_cross_session_late": lambda: SessionTrendCrossStrategy(
        "trend_cross_session_late", skip_open_minutes=30
    ),
    # Variant 19 (research plan, section 14): buy a sharp dip inside a rising hourly trend. It
    # runs forward in the observer only; the live loop does not run it.
    "dip_in_uptrend": lambda: DipInUptrendStrategy(),
    # The two strategies of the original Money_bot (ported, LONG only, closed bars).
    "ema_crossover": lambda: EmaCrossoverStrategy(EMA_CROSSOVER, name="ema_crossover"),
    "strict_ema_crossover": lambda: EmaCrossoverStrategy(
        STRICT_EMA_CROSSOVER, name="strict_ema_crossover"
    ),
    # The same two with the pullback distance widened from 0.5 % / 1 % to 5 %.
    "ema_crossover_5pct": lambda: EmaCrossoverStrategy(
        EMA_CROSSOVER_5PCT, name="ema_crossover_5pct"
    ),
    "strict_ema_crossover_5pct": lambda: EmaCrossoverStrategy(
        STRICT_EMA_CROSSOVER_5PCT, name="strict_ema_crossover_5pct"
    ),
}


def strategy_names() -> List[str]:
    return list(STRATEGIES)


def get_strategy(name: str) -> BaseStrategy:
    try:
        return STRATEGIES[name]()
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}; the choices are: {', '.join(STRATEGIES)}"
        ) from None
