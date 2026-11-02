from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from trading_system.config import Settings
from trading_system.domain.enums import OracleVerdict, SignalDirection
from trading_system.domain.models import OracleDecision, StrategySignal
from trading_system.oracle import NullOracle, OracleRequest, create_oracle


def _signal() -> StrategySignal:
    return StrategySignal(
        version="1",
        symbol="AAPL",
        timeframe="15m",
        closed_bar_timestamp=datetime(2026, 11, 2, 15, 30, tzinfo=timezone.utc),
        direction=SignalDirection.LONG,
    )


def test_the_oracle_can_say_exactly_three_things() -> None:
    assert [v.value for v in OracleVerdict] == ["APPROVE", "VETO", "HOLD"]


@pytest.mark.parametrize(
    "verdict, approved",
    [(OracleVerdict.APPROVE, True), (OracleVerdict.VETO, False), (OracleVerdict.HOLD, False)],
)
def test_only_an_approve_approves(verdict: OracleVerdict, approved: bool) -> None:
    assert OracleDecision(signal_id="s", verdict=verdict).approved is approved


@pytest.mark.parametrize("verdict", [OracleVerdict.APPROVE, OracleVerdict.VETO])
def test_a_call_that_failed_cannot_be_an_opinion(verdict: OracleVerdict) -> None:
    """H1: an error used to be stored as a veto. Now the type refuses to say so."""
    with pytest.raises(ValidationError, match="can only be a HOLD"):
        OracleDecision(signal_id="s", verdict=verdict, failed=True)

    assert OracleDecision(signal_id="s", verdict=OracleVerdict.HOLD, failed=True).failed


def test_confidence_is_a_fraction_and_the_reason_is_kept_short() -> None:
    with pytest.raises(ValidationError):
        OracleDecision(signal_id="s", verdict=OracleVerdict.APPROVE, confidence=1.5)

    decision = OracleDecision(signal_id="s", verdict=OracleVerdict.VETO, reason="x" * 2000)
    assert len(decision.reason) == 500  # a chatty model must not make the record fail


def test_the_null_oracle_lets_every_signal_pass() -> None:
    signal = _signal()

    decision = NullOracle().evaluate(OracleRequest(signal=signal))

    assert decision.verdict is OracleVerdict.APPROVE
    assert decision.signal_id == signal.signal_id
    assert decision.provider == "null" and decision.failed is False


def _settings(**values: object) -> Settings:
    return Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s", **values)


def test_no_oracle_when_it_is_switched_off() -> None:
    assert create_oracle(_settings(oracle_enabled=False)) is None


def test_the_factory_builds_the_provider_that_was_asked_for() -> None:
    assert isinstance(
        create_oracle(_settings(oracle_enabled=True, oracle_provider="null")), NullOracle
    )
