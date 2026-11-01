"""Bounded retry for read-only calls to the broker and the data API."""

import functools
import logging
import random
import time
from typing import Any, Callable, TypeVar

import requests
from alpaca.common.exceptions import APIError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP statuses that mean "try again later": timeout, rate limit, server-side trouble.
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def is_transient_error(exc: BaseException) -> bool:
    """True for failures that can go away on their own: network errors, timeouts, rate
    limits and server errors.

    False for everything else, and that matters: a bug (TypeError, KeyError, ...) or a
    client error (401, 403, 404, 422) fails identically every time, so retrying it only
    delays the failure. The old decorator retried all of them, five times with sleeps up
    to 16 s, which hid a deterministic bug behind half a minute of waiting."""
    if isinstance(exc, APIError):
        return exc.status_code in _TRANSIENT_STATUS
    return isinstance(
        exc,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            ConnectionError,
            TimeoutError,
        ),
    )


def retry_read_request(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    is_retryable: Callable[[BaseException], bool] = is_transient_error,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator for read-only requests using bounded exponential backoff with jitter.

    Not meant for order submission, and never retries indefinitely."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            while True:
                try:
                    logger.debug("Initiating read request: %s", func.__name__)
                    return func(*args, **kwargs)
                except Exception as e:
                    if not is_retryable(e):
                        logger.error("%s failed and is not retryable: %r", func.__name__, e)
                        raise
                    if retries >= max_retries:
                        logger.error(
                            "Failed %s after %d retries: %r", func.__name__, max_retries, e
                        )
                        raise

                    # Exponential backoff capped at max_delay, with jitter of +-50 %.
                    delay = min(base_delay * (2**retries), max_delay)
                    delay = delay * random.uniform(0.5, 1.5)
                    logger.warning(
                        "Transient error in %s (attempt %d/%d): %r. Retrying in %.2fs...",
                        func.__name__,
                        retries + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    retries += 1

        return wrapper

    return decorator
