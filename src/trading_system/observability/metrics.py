"""
Prometheus Metrics definitions for the trading system.
"""

from prometheus_client import Counter, Gauge, start_http_server

# Define metrics
RUNTIME_HEALTH = Gauge(
    "runtime_health", "Health of the trading system runtime (1=healthy, 0=unhealthy)"
)

MARKET_DATA_AGE_SECONDS = Gauge(
    "market_data_age_seconds", "Age of the latest market data in seconds"
)

RECONCILIATION_AGE_SECONDS = Gauge(
    "reconciliation_age_seconds", "Time since last successful broker reconciliation in seconds"
)

ACTIVE_POSITIONS = Gauge("active_positions", "Number of active trading positions")

RESERVED_ENTRIES = Gauge("reserved_entries", "Number of reserved entries pending execution")

GROSS_EXPOSURE = Gauge("gross_exposure", "Total gross exposure as a fraction of account equity")

PLANNED_OPEN_RISK = Gauge(
    "planned_open_risk",
    "Total planned open risk across all positions/intents as a fraction of equity",
)

DAILY_LOSS = Gauge("daily_loss", "Current daily loss as a fraction of start-of-day equity")

# Use labels for reason, but do NOT use unique IDs like signal_id or order_id
REJECTED_SIGNALS = Counter(
    "rejected_signals_total", "Total number of rejected trading signals", ["reason"]
)

ORACLE_LATENCY_SECONDS = Gauge("oracle_latency_seconds", "Latency of Oracle calls")

ORACLE_ERRORS = Counter("oracle_errors_total", "Total number of Oracle errors")

ORDER_FAILURES = Counter("order_failures_total", "Total number of failed broker orders")

UNRESOLVED_INTENTS = Gauge(
    "unresolved_intents",
    "Number of order intents in an unknown or unresolved state requiring reconciliation",
)

UNPROTECTED_POSITIONS = Gauge(
    "unprotected_positions",
    "Number of open positions lacking protective child orders (stop-loss/take-profit)",
)


def start_metrics_server(port: int = 9100, host: str = "127.0.0.1") -> None:
    """
    Starts the Prometheus metrics scrape server.

    Security Note for Remote Deployments:
    By default, this server binds only to the loopback interface (127.0.0.1)
    to prevent unauthorized external access.

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
