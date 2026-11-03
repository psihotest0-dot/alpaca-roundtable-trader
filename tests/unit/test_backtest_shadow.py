"""The Oracle's vetoes, played out (Phase 3.5): what the signals it kept us out of would have done."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.fixtures.bars import GAP_UP_6_PCT, QUIET, make_bars_with_follow_up
from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.models import BacktestConfig, BacktestResult
from trading_system.config import Settings
from trading_system.runtime.runner import ENTRY_DELAY
from trading_system.strategy.trend_cross import TrendCrossStrategy

UTC = timezone.utc
IN_SESSION = datetime(2026, 11, 2, 16, 7, tzinfo=UTC)  # the cross-up bar closed at 16:00 UTC
NO_COSTS = {"slippage_pct": 0.0, "fee_pct_per_trade": 0.0}


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s")


def _case(follow: list[tuple[float, float, float, float]]) -> tuple[Any, Any, Any, str]:
    """Bars with a cross-up signal and the follow-up `follow`; returns the frames, the start
    times of the follow-up bars and the id of the signal (found with the strategy itself)."""
    frame, hourly, follow_index = make_bars_with_follow_up(IN_SESSION, follow)
    [signal] = TrendCrossStrategy().generate_signals(
        {"AAPL": frame}, {"AAPL": hourly}, datetime(2026, 11, 2, 16, 0, 30, tzinfo=UTC)
    )
    return frame, hourly, follow_index, signal.signal_id


def _replay(tmp_path: Path, decisions: dict[str, Any]) -> BacktestConfig:
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(decisions), encoding="utf-8")
    return BacktestConfig(replay_oracle_path=str(path), **NO_COSTS)


def _run(settings: Settings, config: BacktestConfig, frame: Any, hourly: Any) -> BacktestResult:
    return BacktestEngine(settings, config, {"AAPL": frame}, {"AAPL": hourly}).run()


def test_a_vetoed_signal_is_not_traded_but_is_played_out_as_a_shadow_trade(
    settings: Settings, tmp_path: Path
) -> None:
    frame, hourly, follow_index, signal_id = _case([QUIET, QUIET, GAP_UP_6_PCT])
    config = _replay(tmp_path, {signal_id: {"approved": False, "verdict": "VETO"}})

    result = _run(settings, config, frame, hourly)

    assert result.trades == []  # nothing was bought
    assert result.not_traded == {"Oracle: no approval": 1}
    [shadow] = result.shadow_trades
    assert (shadow.symbol, shadow.signal_id, shadow.verdict) == ("AAPL", signal_id, "VETO")
    assert shadow.entry_time == follow_index[0]
    assert shadow.exit_reason == "TAKE_PROFIT" and shadow.exit_time == follow_index[2]
    assert shadow.pnl_pct == pytest.approx(0.06)  # bought at the open, sold at the next gap


def test_a_veto_that_saved_a_loss_shows_as_a_losing_shadow_trade(
    settings: Settings, tmp_path: Path
) -> None:
    frame, hourly, _, signal_id = _case([(1.0, 1.001, 0.9, 0.92), QUIET])
    config = _replay(tmp_path, {signal_id: {"approved": False, "verdict": "VETO"}})

    [shadow] = _run(settings, config, frame, hourly).shadow_trades

    assert shadow.exit_reason == "STOP_LOSS" and shadow.pnl_pct < -0.03


def test_an_approved_signal_is_traded_and_leaves_no_shadow(
    settings: Settings, tmp_path: Path
) -> None:
    frame, hourly, _, signal_id = _case([QUIET, QUIET, GAP_UP_6_PCT])
    config = _replay(tmp_path, {signal_id: {"approved": True, "verdict": "APPROVE"}})

    result = _run(settings, config, frame, hourly)

    assert result.trade_count == 1 and result.shadow_trades == []


def test_a_signal_the_replay_knows_nothing_about_is_not_a_veto(
    settings: Settings, tmp_path: Path
) -> None:
    """Not asked is not refused: the replay cannot say what the Oracle would have answered, so
    the signal is not traded, and it is not counted against the Oracle either."""
    frame, hourly, _, _ = _case([QUIET, QUIET, GAP_UP_6_PCT])
    config = _replay(tmp_path, {"some-other-signal": {"approved": True}})

    result = _run(settings, config, frame, hourly)

    assert result.trades == [] and result.shadow_trades == []
    assert result.not_traded == {"Oracle: no approval": 1}


def test_a_hold_that_never_became_an_answer_is_shadowed_with_its_own_verdict(
    settings: Settings, tmp_path: Path
) -> None:
    frame, hourly, _, signal_id = _case([QUIET, QUIET, GAP_UP_6_PCT])
    config = _replay(tmp_path, {signal_id: {"approved": False, "verdict": "HOLD", "final": False}})

    [shadow] = _run(settings, config, frame, hourly).shadow_trades

    assert shadow.verdict == "HOLD"


def test_a_replay_file_is_enough_to_switch_the_oracle_on(
    settings: Settings, tmp_path: Path
) -> None:
    assert settings.oracle_enabled is False
    frame, hourly, _, signal_id = _case([QUIET, QUIET, GAP_UP_6_PCT])

    vetoed = _run(settings, _replay(tmp_path, {signal_id: {"approved": False}}), frame, hourly)
    no_oracle = _run(settings, BacktestConfig(**NO_COSTS), frame, hourly)

    assert vetoed.trades == [] and len(vetoed.shadow_trades) == 1
    assert no_oracle.trade_count == 1  # without a replay every signal goes on


def test_no_second_shadow_position_while_the_first_in_that_symbol_is_open(
    settings: Settings, tmp_path: Path
) -> None:
    reaches_target = (1.0, 1.06, 0.999, 1.05)  # trades up to the target without a gap
    frame, hourly, follow_index, signal_id = _case([QUIET, QUIET, reaches_target, QUIET])
    config = _replay(tmp_path, {signal_id: {"approved": False, "verdict": "VETO"}})
    engine = BacktestEngine(settings, config, {"AAPL": frame}, {"AAPL": hourly})
    engine._prepare()
    [signal] = TrendCrossStrategy().generate_signals(
        {"AAPL": frame}, {"AAPL": hourly}, datetime(2026, 11, 2, 16, 0, 30, tzinfo=UTC)
    )

    def ask(moment: pd.Timestamp) -> None:
        engine._decision_time = moment + ENTRY_DELAY
        engine._shadow(signal, moment)

    ask(follow_index[0])  # opens; the target is reached on follow_index[2]
    ask(follow_index[1])  # still open: refused
    assert len(engine.shadow_trades) == 1
    ask(follow_index[2])  # the exit bar itself: it is closed by then
    assert len(engine.shadow_trades) == 2


def test_a_shadow_trade_that_cannot_be_priced_is_counted_not_hidden(
    settings: Settings, tmp_path: Path
) -> None:
    # The next bar opens 6 % away from the signal bar's close: the live price check refuses it,
    # and the shadow follows the same rule as a real entry.
    frame, hourly, _, signal_id = _case([GAP_UP_6_PCT, GAP_UP_6_PCT])
    config = _replay(tmp_path, {signal_id: {"approved": False, "verdict": "VETO"}})
    engine = BacktestEngine(settings, config, {"AAPL": frame}, {"AAPL": hourly})

    result = engine.run()

    assert result.shadow_trades == [] and engine.shadow_skipped == 1


def test_costs_are_in_the_shadow_result(settings: Settings, tmp_path: Path) -> None:
    frame, hourly, _, signal_id = _case([QUIET, QUIET, GAP_UP_6_PCT])
    path = tmp_path / "d.json"
    path.write_text(
        json.dumps({signal_id: {"approved": False, "verdict": "VETO"}}), encoding="utf-8"
    )
    costly = BacktestConfig(
        replay_oracle_path=str(path), slippage_pct=0.001, fee_pct_per_trade=0.001
    )

    [shadow] = _run(settings, costly, frame, hourly).shadow_trades

    assert shadow.pnl_pct < 0.06 - 0.003  # slippage on both sides and a fee on both
    assert shadow.entry_price > Decimal("0")
