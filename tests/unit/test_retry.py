"""Which failures are worth retrying, and that the retry is bounded."""

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import requests
from alpaca.common.exceptions import APIError

from trading_system.common.retry import is_transient_error, retry_read_request


def _api_error(status: int | None) -> APIError:
    http_error = (
        None if status is None else SimpleNamespace(response=SimpleNamespace(status_code=status))
    )
    return APIError('{"code": 1, "message": "x"}', http_error)


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_api_error(429), True),  # rate limited
        (_api_error(500), True),
        (_api_error(503), True),
        (_api_error(401), False),  # bad credentials never fix themselves
        (_api_error(403), False),
        (_api_error(404), False),
        (_api_error(422), False),
        (_api_error(None), False),  # no HTTP response to judge by
        (requests.exceptions.ConnectionError(), True),
        (requests.exceptions.Timeout(), True),
        (ConnectionResetError(), True),
        (TimeoutError(), True),
        (TypeError("unsupported operand"), False),  # a bug
        (KeyError("timestamp"), False),
        (UnboundLocalError("pd"), False),
        (ValueError("bad"), False),
    ],
)
def test_which_errors_are_transient(exc: BaseException, expected: bool) -> None:
    assert is_transient_error(exc) is expected


class _Flaky:
    """A callable that raises the given errors one by one, then returns "ok"."""

    def __init__(self, errors: list[BaseException]):
        self.errors = list(errors)
        self.calls = 0

    def function(self) -> Callable[[], str]:
        def read_something() -> str:  # a real function, like the ones the decorator wraps
            self.calls += 1
            if self.errors:
                raise self.errors.pop(0)
            return "ok"

        return read_something


def test_transient_errors_are_retried_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    flaky = _Flaky([requests.exceptions.Timeout(), _api_error(503)])

    assert retry_read_request(max_retries=5)(flaky.function())() == "ok"
    assert flaky.calls == 3
    assert len(sleeps) == 2


def test_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    flaky = _Flaky([requests.exceptions.ConnectionError()] * 10)

    with pytest.raises(requests.exceptions.ConnectionError):
        retry_read_request(max_retries=3)(flaky.function())()

    assert flaky.calls == 4  # the first try plus three retries


@pytest.mark.parametrize("error", [TypeError("bug"), UnboundLocalError("pd"), _api_error(404)])
def test_deterministic_failures_fail_at_once_without_sleeping(
    error: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old decorator retried everything: a bug cost five attempts and ~30 s of sleeping."""

    def no_sleep(_s: float) -> None:
        raise AssertionError("must not sleep for a failure that cannot go away")

    monkeypatch.setattr("time.sleep", no_sleep)
    flaky = _Flaky([error])

    with pytest.raises(type(error)):
        retry_read_request()(flaky.function())()

    assert flaky.calls == 1


def test_backoff_grows_and_respects_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    flaky = _Flaky([requests.exceptions.Timeout()] * 6)

    with pytest.raises(requests.exceptions.Timeout):
        retry_read_request(max_retries=5, base_delay=1.0, max_delay=4.0)(flaky.function())()

    assert len(sleeps) == 5
    assert all(0.5 <= d <= 4.0 * 1.5 for d in sleeps)  # jitter is +-50 % of the capped delay
