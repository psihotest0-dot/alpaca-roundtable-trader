"""The client of the Go backend's Oracle (task 2.2, T3), against a fake server on a real socket.

What it must do: put the signal on the wire in the shape of the contract, read the three words
back, and turn everything else (every status, a body that is not JSON, a verdict that is not one
of the three, a silence, a port with nobody behind it) into a HOLD that is marked as failed and
carries a kind and a code, never the body and never the token.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
import requests

from tests.fixtures.oracle_backend import (
    GOOD_TOKEN,
    FakeOracleBackend,
    Reply,
    answer,
    error,
)
from trading_system.config import Settings
from trading_system.domain.enums import OracleVerdict, SignalDirection
from trading_system.domain.models import StrategySignal
from trading_system.oracle.backend import (
    BAD_REPLY,
    BAD_REQUEST,
    FAILURE_COOLDOWN_SECONDS,
    INTERNAL,
    TIMEOUT,
    UNAUTHORIZED,
    UNAVAILABLE,
    BackendOracle,
)
from trading_system.oracle.contract import OracleNews, OracleRequest


def _signal() -> StrategySignal:
    return StrategySignal(
        version="trend_cross-1",
        symbol="AAPL",
        timeframe="15m",
        closed_bar_timestamp=datetime(2026, 11, 2, 15, 30, tzinfo=timezone.utc),
        direction=SignalDirection.LONG,
    )


def _request(**changes: Any) -> OracleRequest:
    values: dict[str, Any] = {
        "signal": _signal(),
        "rationale": "EMA cross up",
        "technicals": {"rsi": 61.2, "atr": None, "hourly_trend_up": True},
        "recent_bars": [{"start": "2026-11-02T15:15:00+00:00", "close": 187.1, "volume": 1200.0}],
        "volume_ratio": 1.4,
        "live_price": Decimal("187.35"),
        "context": {"equity": 100000.0, "open_positions": 1},
        "news": [
            OracleNews(
                article_id=7,
                published_at=datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc),
                age_hours=3,
                source="Benzinga",
                headline="Apple raises guidance",
                summary="Short summary.",
                other_symbols=1,
            )
        ],
    }
    values.update(changes)
    return OracleRequest(**values)


def _oracle(backend: FakeOracleBackend, **kwargs: Any) -> BackendOracle:
    kwargs.setdefault("timeout_seconds", 5.0)
    return BackendOracle(backend.url, GOOD_TOKEN, **kwargs)


# --- what goes over the wire ------------------------------------------------------------------


def test_the_signal_goes_to_the_route_with_the_bearer_token_in_the_shape_of_the_contract() -> None:
    with FakeOracleBackend() as backend:
        _oracle(backend).evaluate(_request())

    (received,) = backend.received
    assert (received.method, received.path) == ("POST", "/oracle/evaluate")
    assert received.authorization == f"Bearer {GOOD_TOKEN}"
    assert received.content_type == "application/json"
    body = received.json
    assert body["schema_version"] == 1 and body["mode"] == "shadow"
    signal = body["signal"]
    assert signal["signal_id"] == _signal().signal_id
    assert (signal["symbol"], signal["direction"], signal["timeframe"]) == ("AAPL", "LONG", "15m")
    assert signal["closed_bar_timestamp"].startswith("2026-11-02T15:30")
    assert "expires_at" not in signal  # the contract has no null for it
    assert body["live_price"] == "187.35"  # a Decimal travels as text: no float rounding
    assert body["volume_ratio"] == 1.4 and body["rationale"] == "EMA cross up"
    assert body["technicals"] == {"rsi": 61.2, "atr": None, "hourly_trend_up": True}
    (news,) = body["news"]
    assert news["headline"] == "Apple raises guidance" and news["article_id"] == 7
    assert news["age_hours"] == 3 and news["other_symbols"] == 1


def test_a_request_without_a_price_or_volume_leaves_those_keys_out() -> None:
    with FakeOracleBackend() as backend:
        _oracle(backend).evaluate(_request(live_price=None, volume_ratio=None))

    body = backend.received[0].json
    assert "live_price" not in body and "volume_ratio" not in body


def test_the_mode_travels_with_the_question() -> None:
    with FakeOracleBackend() as backend:
        decision = _oracle(backend, mode="active").evaluate(_request())

    assert backend.received[0].json["mode"] == "active" and decision.mode == "active"


# --- the three words ---------------------------------------------------------------------------


@pytest.mark.parametrize("word", ["APPROVE", "VETO", "HOLD"])
def test_each_of_the_three_verdicts_is_read_as_a_verdict_and_not_as_a_failure(word: str) -> None:
    with FakeOracleBackend(answer(word, "Причина.", request_id="req-9", model="gemma")) as backend:
        decision = _oracle(backend).evaluate(_request())

    assert decision.verdict is OracleVerdict(word)
    assert decision.failed is False and decision.error_kind is None
    assert decision.reason == "Причина."
    assert (decision.request_id, decision.model) == ("req-9", "gemma")
    assert (decision.provider, decision.mode) == ("backend", "shadow")
    assert decision.signal_id == _signal().signal_id


def test_a_verdict_from_the_backends_cache_is_the_same_verdict() -> None:
    with FakeOracleBackend(answer("VETO", cached=True, latency_ms=0)) as backend:
        decision = _oracle(backend).evaluate(_request())

    assert decision.verdict is OracleVerdict.VETO and decision.failed is False


def test_the_reason_is_one_clean_line() -> None:
    with FakeOracleBackend(answer("APPROVE", "ред едно\n‮ред   две\x00" + "я" * 400)) as b:
        decision = _oracle(b).evaluate(_request())

    assert "\n" not in decision.reason and "‮" not in decision.reason
    assert "\x00" not in decision.reason and "  " not in decision.reason
    assert decision.reason.startswith("ред едно") and len(decision.reason) <= 200


# --- everything else is a HOLD that failed -------------------------------------------------------

REFUSALS = [
    (error(401, "unauthorized"), UNAUTHORIZED),
    (error(400, "invalid_request", detail="signal.symbol"), BAD_REQUEST),
    (error(400, "unreadable_body"), BAD_REQUEST),
    (error(413, "body_too_large"), BAD_REQUEST),
    (error(405, "method_not_allowed"), BAD_REQUEST),
    (error(502, "model_unavailable"), UNAVAILABLE),
    (error(502, "unusable_reply"), BAD_REPLY),
    (error(504, "model_timeout"), TIMEOUT),
    (error(500, "internal_error"), UNAVAILABLE),
    (Reply(503, "overloaded"), UNAVAILABLE),
    (Reply(404, "not found"), UNAVAILABLE),  # the backend serves the route only with a token
    (Reply(302, "", headers={"Location": "/elsewhere"}), BAD_REPLY),
    (Reply(200, "this is not json"), BAD_REPLY),
    (Reply(200, b""), BAD_REPLY),
    (Reply(200, ["APPROVE"]), BAD_REPLY),
    (Reply(200, {"verdict": "APPROVE"}), BAD_REPLY),  # fields missing
    (Reply(200, {**answer().body, "verdict": "MAYBE"}), BAD_REPLY),
    (Reply(200, {**answer().body, "verdict": "approve"}), BAD_REPLY),  # exactly the three words
    (Reply(200, {**answer().body, "verdict": None}), BAD_REPLY),
    (Reply(200, {**answer().body, "verdict": ["APPROVE"]}), BAD_REPLY),
    (Reply(200, {**answer().body, "reason": 5}), BAD_REPLY),
    (Reply(200, {**answer().body, "cached": "false"}), BAD_REPLY),
    (Reply(200, {**answer().body, "latency_ms": True}), BAD_REPLY),
    (Reply(200, {**answer().body, "request_id": None}), BAD_REPLY),
    (Reply(200, b'{"verdict": "APPROVE", ' + b"x" * 70_000), BAD_REPLY),  # too big to be a reply
]


@pytest.mark.parametrize(
    "reply, kind", REFUSALS, ids=[f"{r.status}-{k}-{i}" for i, (r, k) in enumerate(REFUSALS)]
)
def test_anything_but_a_clean_verdict_is_a_hold_that_failed(reply: Reply, kind: str) -> None:
    with FakeOracleBackend(reply) as backend:
        decision = _oracle(backend).evaluate(_request())

    assert decision.verdict is OracleVerdict.HOLD  # never a yes, never a final no
    assert decision.failed is True and decision.error_kind == kind
    assert decision.approved is False
    assert decision.provider == "backend" and decision.signal_id == _signal().signal_id
    assert len(backend.received) == 1  # no retry inside; the cycle asks again on its tick


def test_a_redirect_is_not_followed_because_the_token_must_not_go_elsewhere() -> None:
    with FakeOracleBackend(Reply(302, "", headers={"Location": "/elsewhere"})) as backend:
        decision = _oracle(backend).evaluate(_request())

    assert decision.failed and len(backend.received) == 1


def test_the_reason_carries_the_status_and_a_known_code_and_nothing_else() -> None:
    with FakeOracleBackend(
        error(400, "invalid_request", detail="PLANTED-DETAIL"),
        Reply(500, {"error": "PLANTED free text from the body"}),
        Reply(502, "PLANTED <html>stack trace</html>"),
        Reply(200, "PLANTED not json"),
    ) as backend:
        decisions = [_oracle(backend).evaluate(_request()) for _ in range(4)]

    assert "HTTP 400, invalid_request" in decisions[0].reason
    assert "HTTP 500" in decisions[1].reason  # the code is not on the list, so it is left out
    for decision in decisions:
        assert "PLANTED" not in decision.reason


def test_a_silent_backend_is_a_timeout_after_the_time_asked_for() -> None:
    with FakeOracleBackend(Reply(200, answer().body, delay=1.5)) as backend:
        decision = _oracle(backend, timeout_seconds=0.3).evaluate(_request())

    assert decision.failed and decision.error_kind == TIMEOUT
    assert decision.verdict is OracleVerdict.HOLD and "0.3 s" in decision.reason


class _Raises(requests.Session):
    """A session whose every call fails with the given error, at once."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    def request(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise self._error


@pytest.mark.parametrize(
    "failure, name",
    [
        (requests.ConnectionError("connection refused: SECRET-HOST-DETAIL"), "ConnectionError"),
        (requests.ConnectTimeout("nobody picked up"), "ConnectTimeout"),
        (requests.exceptions.ChunkedEncodingError("cut"), "ChunkedEncodingError"),
    ],
)
def test_a_backend_that_cannot_be_reached_is_unavailable_and_not_a_timeout(
    failure: Exception, name: str
) -> None:
    oracle = BackendOracle("http://127.0.0.1:9", GOOD_TOKEN, session=_Raises(failure))

    decision = oracle.evaluate(_request())

    assert decision.failed and decision.error_kind == UNAVAILABLE
    assert name in decision.reason and "SECRET" not in decision.reason  # the class, not the text


def test_a_read_timeout_is_a_timeout() -> None:
    oracle = BackendOracle(
        "http://127.0.0.1:9", GOOD_TOKEN, session=_Raises(requests.ReadTimeout())
    )

    assert oracle.evaluate(_request()).error_kind == TIMEOUT


def test_a_port_with_nobody_behind_it_is_unavailable() -> None:
    with FakeOracleBackend() as backend:
        url = backend.url
    # On Windows a closed loopback port is refused only after about two seconds.
    decision = BackendOracle(url, GOOD_TOKEN, timeout_seconds=1.0).evaluate(_request())

    assert decision.failed and decision.error_kind == UNAVAILABLE


def test_the_backend_refusing_the_token_is_unauthorized() -> None:
    with FakeOracleBackend() as backend:
        decision = BackendOracle(backend.url, "wrong-" * 8).evaluate(_request())

    assert decision.failed and decision.error_kind == UNAUTHORIZED


@pytest.mark.parametrize("token", ["", "has space in it", "line\nbreak", "кирилица" * 5])
def test_without_a_usable_token_nothing_is_sent(token: str) -> None:
    with FakeOracleBackend() as backend:
        decision = BackendOracle(backend.url, token).evaluate(_request())

    assert decision.failed and decision.error_kind == UNAUTHORIZED
    assert backend.received == []


def test_a_request_that_cannot_be_written_as_json_is_not_sent() -> None:
    with FakeOracleBackend() as backend:
        decision = _oracle(backend).evaluate(_request(technicals={"rsi": object()}))

    assert decision.failed and decision.error_kind == BAD_REQUEST
    assert backend.received == []


def test_a_nan_in_the_numbers_goes_out_as_null_because_json_has_no_nan() -> None:
    """The Go side would refuse the word NaN: the request must hold strict JSON."""
    with FakeOracleBackend() as backend:
        decision = _oracle(backend).evaluate(
            _request(technicals={"rsi": float("nan"), "atr": float("inf")})
        )

    assert decision.verdict is OracleVerdict.APPROVE
    raw = backend.received[0].raw.decode("utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    assert backend.received[0].json["technicals"] == {"rsi": None, "atr": None}


def test_a_request_over_the_backends_limit_is_not_sent() -> None:
    bars = [{"start": str(i), "note": "x" * 1000} for i in range(80)]
    with FakeOracleBackend() as backend:
        decision = _oracle(backend).evaluate(_request(recent_bars=bars))

    assert decision.failed and decision.error_kind == BAD_REQUEST
    assert backend.received == []


def test_a_bug_in_the_client_is_a_failed_hold_and_not_an_exception() -> None:
    class Broken(requests.Session):
        def request(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            raise RuntimeError("boom")

    decision = BackendOracle("http://127.0.0.1:9", GOOD_TOKEN, session=Broken()).evaluate(
        _request()
    )

    assert decision.failed and decision.error_kind == INTERNAL
    assert decision.verdict is OracleVerdict.HOLD and "RuntimeError" in decision.reason


# --- a failure costs the cycle once --------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    "reply",
    [error(504, "model_timeout"), error(502, "model_unavailable"), error(500, "internal_error")],
)
def test_after_a_failure_the_next_calls_fail_at_once_until_the_pause_is_over(reply: Reply) -> None:
    clock = Clock()
    with FakeOracleBackend(reply, answer("APPROVE")) as backend:
        oracle = _oracle(backend, clock=clock)
        first = oracle.evaluate(_request())
        clock.now += FAILURE_COOLDOWN_SECONDS - 1
        skipped = oracle.evaluate(_request())
        assert len(backend.received) == 1  # the second question never left the machine
        clock.now += 2
        later = oracle.evaluate(_request())

    assert first.failed and skipped.failed and skipped.error_kind == UNAVAILABLE
    assert "skipped" in skipped.reason
    assert later.verdict is OracleVerdict.APPROVE and len(backend.received) == 2


def test_a_timeout_also_starts_the_pause() -> None:
    clock = Clock()
    with FakeOracleBackend(Reply(200, answer().body, delay=1.0)) as backend:
        oracle = _oracle(backend, timeout_seconds=0.2, clock=clock)
        assert oracle.evaluate(_request()).error_kind == TIMEOUT
        assert oracle.evaluate(_request()).error_kind == UNAVAILABLE  # skipped, no second wait

    assert len(backend.received) == 1


def test_a_refused_token_or_request_does_not_start_a_pause() -> None:
    """A 401 or a 400 is quick and a pause would not help: the next question asks again."""
    with FakeOracleBackend(error(401, "unauthorized"), error(400, "invalid_request")) as backend:
        oracle = _oracle(backend)
        oracle.evaluate(_request())
        oracle.evaluate(_request())
        third = oracle.evaluate(_request())

    assert len(backend.received) == 3 and third.verdict is OracleVerdict.APPROVE


def test_a_good_answer_does_not_start_a_pause() -> None:
    with FakeOracleBackend() as backend:
        oracle = _oracle(backend)
        for _ in range(3):
            oracle.evaluate(_request())

    assert len(backend.received) == 3


# --- the token stays out of everything -----------------------------------------------------------


def test_the_token_never_reaches_a_reason_or_a_log_line(caplog: pytest.LogCaptureFixture) -> None:
    replies = [r for r, _ in REFUSALS[:12]]
    with caplog.at_level(logging.DEBUG):
        with FakeOracleBackend(*replies) as backend:
            decisions = [
                BackendOracle(backend.url, GOOD_TOKEN, timeout_seconds=5).evaluate(_request())
                for _ in replies
            ]

    assert GOOD_TOKEN not in caplog.text
    assert all(GOOD_TOKEN not in d.reason for d in decisions)
    assert all(GOOD_TOKEN not in repr(d) for d in decisions)


def test_the_log_names_the_signal_and_the_backends_request_for_an_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with FakeOracleBackend(answer("VETO", request_id="req-abc")) as backend:
        with caplog.at_level(logging.INFO):
            _oracle(backend).evaluate(_request())

    line = next(r.getMessage() for r in caplog.records if "answered" in r.getMessage())
    assert f"signal_id={_signal().signal_id}" in line and "request_id=req-abc" in line


def test_proxy_settings_in_the_environment_are_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # nobody listens there
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    with FakeOracleBackend() as backend:
        decision = _oracle(backend).evaluate(_request())

    assert decision.verdict is OracleVerdict.APPROVE and len(backend.received) == 1


# --- is it there? ----------------------------------------------------------------------------------


def test_ping_asks_the_ping_route_with_the_token() -> None:
    with FakeOracleBackend(Reply(200, {"status": "ok"})) as backend:
        result = _oracle(backend).ping()

    assert result.ok and result.kind is None
    (received,) = backend.received
    assert (received.method, received.path) == ("GET", "/oracle/ping")
    assert received.authorization == f"Bearer {GOOD_TOKEN}"


@pytest.mark.parametrize(
    "reply, kind",
    [
        (error(401, "unauthorized"), UNAUTHORIZED),
        (Reply(404, "no"), UNAVAILABLE),
        (Reply(500, "no"), UNAVAILABLE),
        (Reply(200, {"status": "nope"}), BAD_REPLY),
        (Reply(200, "text"), BAD_REPLY),
    ],
)
def test_ping_says_what_is_wrong(reply: Reply, kind: str) -> None:
    with FakeOracleBackend(reply) as backend:
        result = _oracle(backend).ping()

    assert not result.ok and result.kind == kind and result.detail


def test_ping_without_a_token_or_without_a_server_does_not_raise() -> None:
    with FakeOracleBackend() as backend:
        url = backend.url
        no_token = BackendOracle(url, "").ping()
        assert backend.received == []
    gone = BackendOracle(url, GOOD_TOKEN, session=_Raises(requests.ConnectionError())).ping()

    assert (no_token.ok, no_token.kind) == (False, UNAUTHORIZED)
    assert (gone.ok, gone.kind) == (False, UNAVAILABLE)


def test_a_ping_never_starts_the_pause_of_the_questions() -> None:
    with FakeOracleBackend(Reply(500, "x"), answer("APPROVE")) as backend:
        oracle = _oracle(backend)
        assert not oracle.ping().ok
        decision = oracle.evaluate(_request())

    assert decision.verdict is OracleVerdict.APPROVE


# --- from the settings -------------------------------------------------------------------------------


def test_the_oracle_is_built_from_the_settings() -> None:
    settings = Settings(
        _env_file=None,
        alpaca_api_key="k",
        alpaca_secret_key="s",
        oracle_enabled=True,
        oracle_provider="backend",
        oracle_mode="active",
        oracle_backend_url="http://127.0.0.1:8080/",
        oracle_backend_token=GOOD_TOKEN,
        oracle_timeout_seconds=7,
    )

    oracle = BackendOracle.from_settings(settings)

    assert (oracle.url, oracle.mode, oracle.timeout_seconds) == (
        "http://127.0.0.1:8080",
        "active",
        7,
    )
    assert oracle.name == "backend"
