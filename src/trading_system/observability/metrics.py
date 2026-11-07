"""
Prometheus metrics of the trading system.

**Why the file looks the way it does.**

* Every name starts with `alpaca_`. The bot's metrics go into the Go backend's Prometheus and Grafana
  (job `alpaca-trader`, agreed with the backend's session on 2026-11-07), where the backend's own
  series live, so a prefix keeps the two apart.
* **Only metrics that something really sets are published.** A gauge nobody sets reads 0, and a `0` for
  "unprotected positions" or "unresolved intents" would look like all is well while meaning "not
  measured". Those are kept below in a private registry that no server exposes (`NOT_WIRED`) and move up
  the day something sets them.
* **A metric must never be able to hurt the trading.** Everything the loops call is one of the small
  functions at the end (`beat`, `entries_pass`, `rejected`, `account`, `candidate_pass`), each wrapped so
  that whatever goes wrong is swallowed. Without `METRICS_ENABLED` nothing is served; the counters still
  count in memory, which costs nothing.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# --- published: something sets every one of these --------------------------------------------

# The loop is alive: the time of its last heartbeat, one per tick (every minute with the market open,
# at least every 5 minutes closed). An alert reads `time() - alpaca_heartbeat_timestamp_seconds`.
HEARTBEAT_TIMESTAMP = Gauge(
    "alpaca_heartbeat_timestamp_seconds", "Unix time of the loop's last heartbeat"
)
MARKET_OPEN = Gauge("alpaca_market_open", "1 while the market is open, 0 while it is closed")
# When this stretch of open market began (0 while it is closed). The first look for entries of a day
# comes 15 minutes after the open, so "no look for 20 minutes" must be counted from the later of the
# last look and this time, as the healthcheck does; otherwise it would fire every morning.
MARKET_OPEN_SINCE = Gauge(
    "alpaca_market_open_since_timestamp_seconds",
    "Unix time at which the current stretch of open market began, 0 while the market is closed",
)
LAST_ENTRIES_TIMESTAMP = Gauge(
    "alpaca_last_entries_timestamp_seconds",
    "Unix time of the last completed look for entries on a new bar",
)

# One per look for entries on a bar. outcome: ok, retry_later (the bar was not there yet), skipped
# (the cycle chose not to look, for example trading is halted), crashed.
ENTRIES_PASSES = Counter(
    "alpaca_entries_passes_total", "Looks for entries on a new bar, by outcome", ["outcome"]
)
SIGNALS_FOUND = Counter("alpaca_signals_total", "Signals found by the strategy on new bars")
ENTRIES_ERRORS = Counter(
    "alpaca_entries_errors_total", "Signals that broke in risk or execution, and crashed passes"
)

# Use labels for reason, but do NOT use unique IDs like signal_id or order_id
REJECTED_SIGNALS = Counter(
    "alpaca_rejected_signals_total", "Total number of rejected trading signals", ["reason"]
)

# "Unreachable" and "says no" are different things; kind tells them apart:
# answer (the Oracle answered), cached (a final answer from the record), and for no answer
# either failed (cause unknown) or the cause: unavailable, timeout, unauthorized, bad_request,
# bad_reply, internal (see oracle/backend.py).
ORACLE_ANSWERS = Counter(
    "alpaca_oracle_answers_total",
    "Oracle answers by verdict and kind",
    ["verdict", "kind"],
)

# The account, as the live loop sees it after each look for entries.
ACCOUNT_EQUITY = Gauge("alpaca_account_equity", "Equity of the account in dollars")
ACTIVE_POSITIONS = Gauge("alpaca_active_positions", "Number of open positions")
GROSS_EXPOSURE = Gauge(
    "alpaca_gross_exposure", "Total gross exposure as a fraction of account equity"
)

# Set by oracle/recording.py on every call to the Oracle.
ORACLE_LATENCY_SECONDS = Gauge("alpaca_oracle_latency_seconds", "Latency of Oracle calls")
ORACLE_ERRORS = Counter("alpaca_oracle_errors_total", "Total number of Oracle errors")

# The protection check (execution/engine.py) and the order reconciliation (runtime/cycle.py): the
# count of open positions without a protective stop, and WHEN the checks last completed. A count of 0
# only means something while the timestamp is fresh, so alert on both.
UNPROTECTED_POSITIONS = Gauge(
    "alpaca_unprotected_positions",
    "Number of open positions lacking protective child orders (stop-loss/take-profit)",
)
PROTECTION_CHECK_TIMESTAMP = Gauge(
    "alpaca_protection_check_timestamp_seconds",
    "Unix time of the last protection check that completed (the broker could be read)",
)
RECONCILE_TIMESTAMP = Gauge(
    "alpaca_reconcile_timestamp_seconds",
    "Unix time of the last order reconciliation in which every order could be refreshed",
)

# The strategy candidates the observer runs forward in time (research plan, section 12.1). The
# control is the live strategy over the same bars: strategy="trend_cross".
CANDIDATE_SIGNALS = Counter(
    "alpaca_candidate_signals_total",
    "Signals journalled for the strategy candidates and their control",
    ["strategy"],
)
CANDIDATE_ERRORS = Counter(
    "alpaca_candidate_errors_total", "Passes in which a candidate could not be run"
)

# --- defined but set by nothing yet: NOT published ---------------------------------------------
# Exposing these at 0 would say "fine" about things nobody measures. Move one into the published
# block, with the code that sets it and a test, when it is wired.
NOT_WIRED = CollectorRegistry()

RUNTIME_HEALTH = Gauge(
    "alpaca_runtime_health",
    "Health of the trading system runtime (1=healthy, 0=unhealthy)",
    registry=NOT_WIRED,
)
MARKET_DATA_AGE_SECONDS = Gauge(
    "alpaca_market_data_age_seconds", "Age of the latest market data in seconds", registry=NOT_WIRED
)
RECONCILIATION_AGE_SECONDS = Gauge(
    "alpaca_reconciliation_age_seconds",
    "Time since last successful broker reconciliation in seconds",
    registry=NOT_WIRED,
)
RESERVED_ENTRIES = Gauge(
    "alpaca_reserved_entries", "Number of reserved entries pending execution", registry=NOT_WIRED
)
PLANNED_OPEN_RISK = Gauge(
    "alpaca_planned_open_risk",
    "Total planned open risk across all positions/intents as a fraction of equity",
    registry=NOT_WIRED,
)
DAILY_LOSS = Gauge(
    "alpaca_daily_loss",
    "Current daily loss as a fraction of start-of-day equity",
    registry=NOT_WIRED,
)
ORDER_FAILURES = Counter(
    "alpaca_order_failures_total", "Total number of failed broker orders", registry=NOT_WIRED
)
UNRESOLVED_INTENTS = Gauge(
    "alpaca_unresolved_intents",
    "Number of order intents in an unknown or unresolved state requiring reconciliation",
    registry=NOT_WIRED,
)


# --- what the loops call: never raises ---------------------------------------------------------


def _safely(function: F) -> F:
    """The wrapped function never raises: a broken metric is a missing number, not a stopped loop."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - this is the point
            logger.debug(f"A metric could not be set ({function.__name__}): {e}")
            return None

    return wrapper  # type: ignore[return-value]


@_safely
def beat(
    market_open: Optional[bool], open_since: Optional[float] = None, when: Optional[float] = None
) -> None:
    """The loop ticked. `market_open` None (the clock could not be read) leaves the gauges alone.
    `open_since`: when the current stretch of open market began."""
    HEARTBEAT_TIMESTAMP.set(time.time() if when is None else when)
    if market_open is not None:
        MARKET_OPEN.set(1 if market_open else 0)
        if not market_open:
            MARKET_OPEN_SINCE.set(0)
        elif open_since is not None:
            MARKET_OPEN_SINCE.set(open_since)


@_safely
def entries_pass(
    outcome: str, signals: int = 0, errors: int = 0, when: Optional[float] = None
) -> None:
    """One look for entries on a bar finished. A pass that did look (ok or skipped) stamps
    `alpaca_last_entries_timestamp_seconds`; one that has to be tried again or crashed does not."""
    ENTRIES_PASSES.labels(outcome=outcome).inc()
    if signals:
        SIGNALS_FOUND.inc(signals)
    if errors:
        ENTRIES_ERRORS.inc(errors)
    if outcome in ("ok", "skipped"):
        LAST_ENTRIES_TIMESTAMP.set(time.time() if when is None else when)


@_safely
def rejected(reason: str) -> None:
    """Risk said no. The reason is cut at its first colon and at 80 characters, so that a number inside
    a message cannot make a new series."""
    REJECTED_SIGNALS.labels(reason=reason.split(":")[0].strip()[:80] or "unknown").inc()


@_safely
def account(equity: float, positions: int, gross_exposure: float) -> None:
    ACCOUNT_EQUITY.set(equity)
    ACTIVE_POSITIONS.set(positions)
    GROSS_EXPOSURE.set(gross_exposure)


@_safely
def protection_checked(unprotected: int, when: Optional[float] = None) -> None:
    """The protection check completed and found `unprotected` positions without a stop."""
    UNPROTECTED_POSITIONS.set(unprotected)
    PROTECTION_CHECK_TIMESTAMP.set(time.time() if when is None else when)


@_safely
def reconciled(when: Optional[float] = None) -> None:
    """Every order could be refreshed from the broker."""
    RECONCILE_TIMESTAMP.set(time.time() if when is None else when)


@_safely
def candidate_pass(counts: dict[str, int], errors: int) -> None:
    """The observer ran the candidates on a bar. `counts`: signals per strategy (-1: that strategy broke)."""
    for strategy, count in counts.items():
        if count > 0:
            CANDIDATE_SIGNALS.labels(strategy=strategy).inc(count)
    if errors:
        CANDIDATE_ERRORS.inc(errors)


def start_metrics_server(port: int = 9100, host: str = "127.0.0.1") -> None:
    """
    Starts the Prometheus metrics scrape server.

    Security Note for Remote Deployments:
    By default, this server binds only to the loopback interface (127.0.0.1)
    to prevent unauthorized external access.

    In the Docker deployment the server must listen on 0.0.0.0 INSIDE the container (the host
    cannot reach a container's 127.0.0.1), and compose.yaml publishes the port on the host's
    loopback only (`127.0.0.1:9101:9100`), so nothing outside the PC can reach it.

    If you need to scrape metrics remotely:
    1. Do NOT expose this port directly to the public internet.
    2. Keep the bind to 127.0.0.1 and use a reverse proxy (like Nginx or Envoy)
       with authentication/authorization to expose the `/metrics` endpoint.
    3. Alternatively, bind to a private internal network interface (e.g., 10.x.x.x)
       if deployed within a secure VPC or VPN, and configure firewall rules
       (e.g., iptables, AWS Security Groups) to allow access ONLY from the IP address
       of your Prometheus scraper.
    """
    start_http_server(port, addr=host)
