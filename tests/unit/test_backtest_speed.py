"""The backtest asks the strategy only about symbols that could signal, and shows it the live
loop's window of history (Phase 3.3). Neither may change what is found."""

from typing import Any, Callable, Optional

import pandas as pd
import pytest

from tests.fixtures.bars import random_walk_session_bars
from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.models import BacktestConfig, BacktestResult
from trading_system.config import Settings
from trading_system.strategy.base import BaseStrategy
from trading_system.strategy.ema_crossover import EmaCrossoverConfig, EmaCrossoverStrategy
from trading_system.strategy.trend_cross import TrendCrossStrategy

# Five symbols, three weeks of sessions: enough crossings and uptrends for the strategy to fire.
SYMBOLS = {"AAA": (11, 0.0004), "BBB": (12, 0.0002), "CCC": (13, 0.0), "DDD": (14, 0.0003)}


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s")


def _frames(days: int = 18) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    pairs = {
        s: random_walk_session_bars(seed, "2026-10-05", days, drift)
        for s, (seed, drift) in SYMBOLS.items()
    }
    return {s: p[0] for s, p in pairs.items()}, {s: p[1] for s, p in pairs.items()}


def _run(
    settings: Settings, prefilter: bool, strategy: Optional[BaseStrategy] = None
) -> BacktestResult:
    frames_15m, frames_1h = _frames()
    return BacktestEngine(
        settings, BacktestConfig(), frames_15m, frames_1h, prefilter=prefilter, strategy=strategy
    ).run()


def _crossover(volume_window: Optional[int]) -> BaseStrategy:
    """The ported strategies with windows short enough for three weeks of test data (the
    originals need 200 hourly bars for their EMA 200) and a wide pullback band, so they fire."""
    return EmaCrossoverStrategy(
        EmaCrossoverConfig(
            strategy_version="test",
            short_window=5,
            long_window=12,
            pullback_pct=0.02,
            htf_ema_period=20,
            volume_window=volume_window,
        )
    )


STRATEGIES: dict[str, Callable[[], Optional[BaseStrategy]]] = {
    "trend_cross": lambda: None,  # the default: what the live loop runs
    "ema_crossover": lambda: _crossover(None),
    "strict_ema_crossover": lambda: _crossover(10),
}


@pytest.mark.parametrize("which", STRATEGIES)
def test_asking_only_about_possible_signals_finds_exactly_what_asking_about_all_finds(
    settings: Settings, which: str
) -> None:
    fast = _run(settings, prefilter=True, strategy=STRATEGIES[which]())
    exhaustive = _run(settings, prefilter=False, strategy=STRATEGIES[which]())

    assert exhaustive.signals >= 3, "the data has to give the strategy something to find"
    assert exhaustive.trade_count >= 2
    assert fast.signals == exhaustive.signals
    assert fast.not_traded == exhaustive.not_traded
    assert [
        (t.symbol, t.entry_time, t.entry_price, t.quantity, t.exit_time, t.exit_price)
        for t in fast.trades
    ] == [
        (t.symbol, t.entry_time, t.entry_price, t.quantity, t.exit_time, t.exit_price)
        for t in exhaustive.trades
    ]
    assert fast.final_capital == exhaustive.final_capital


def test_the_prefilter_asks_about_fewer_symbols(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[int] = []

    def count(self: Any, data_15m: dict[str, Any], *args: Any, **kwargs: Any) -> list[Any]:
        asked.append(len(data_15m))
        return []

    monkeypatch.setattr(TrendCrossStrategy, "generate_signals", count)
    frames_15m, frames_1h = _frames()

    BacktestEngine(settings, BacktestConfig(), frames_15m, frames_1h, prefilter=True).run()
    with_filter = sum(asked)
    asked.clear()
    BacktestEngine(settings, BacktestConfig(), frames_15m, frames_1h, prefilter=False).run()

    assert with_filter < sum(asked) / 3


def test_the_strategy_sees_the_live_window_and_nothing_from_the_future(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]] = []

    def spy(self: Any, data_15m: dict[str, Any], data_1h: dict[str, Any], when: Any) -> list[Any]:
        for symbol in data_15m:
            seen.append((pd.Timestamp(when), data_15m[symbol], data_1h[symbol]))
        return []

    monkeypatch.setattr(TrendCrossStrategy, "generate_signals", spy)
    frames_15m, frames_1h = _frames(days=40)  # longer than the 30-day window of the 15m bars

    BacktestEngine(settings, BacktestConfig(), frames_15m, frames_1h, prefilter=False).run()

    assert seen
    span_15m = pd.Timedelta(days=settings.lookback_days)
    span_1h = pd.Timedelta(days=settings.htf_lookback_days)
    for when, bars_15m, bars_1h in seen:
        if bars_15m.empty or bars_1h.empty:
            continue  # the first steps have no history yet
        assert bars_15m.index.max() < when and bars_1h.index.max() < when  # nothing from the future
        assert bars_15m.index.min() >= when - span_15m
        assert bars_1h.index.min() >= when - span_1h
    # and the window really does cut: late in a 40-day run the 15m history starts after day one
    late_when, late_15m, _ = seen[-1]
    assert late_15m.index.min() > frames_15m["AAA"]["timestamp"].iloc[0]
    assert late_when - late_15m.index.min() <= span_15m
