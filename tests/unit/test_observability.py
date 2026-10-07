import json
import logging
from unittest.mock import Mock

from trading_system.observability.alerts import LoggingAlertManager
from trading_system.observability.health import HealthStatus
from trading_system.observability.logging import JSONFormatter, SensitiveDataRedactor
from trading_system.observability.metrics import (
    REJECTED_SIGNALS,
    UNRESOLVED_INTENTS,
)


def test_secret_redaction_dict():
    data = {
        "normal_key": "normal_value",
        "api_key": "secret_abc123",
        "nested": {
            "SECRET": "hidden_value",
            "token": "token_123"
        },
        "environment": {"API_URL": "http://test"}
    }

    redacted = SensitiveDataRedactor.redact(data)

    assert redacted["normal_key"] == "normal_value"
    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["nested"]["SECRET"] == "***REDACTED***"
    assert redacted["nested"]["token"] == "***REDACTED***"
    assert redacted["environment"] == "***REDACTED***"


def test_secret_redaction_string():
    text = "Connecting with authorization: Bearer abcdef123 and api_key=xyz"
    redacted = SensitiveDataRedactor.redact(text)

    assert "abcdef123" not in redacted
    assert "xyz" not in redacted
    assert "***REDACTED***" in redacted


def test_json_formatter():
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message with API_KEY=secret123",
        args=(),
        exc_info=None
    )
    record.component = "TestComponent" # type: ignore
    record.signal_id = "sig_123" # type: ignore

    formatter = JSONFormatter()
    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["component"] == "TestComponent"
    assert "secret123" not in parsed["event"]
    assert "***REDACTED***" in parsed["event"]
    assert parsed["signal_id"] == "sig_123"
    assert parsed["intent_id"] is None


def test_health_readiness():
    health = HealthStatus()

    assert health.check_liveness() is True
    assert health.check_readiness() is True

    health.mark_market_data_stale(True)
    assert health.check_readiness() is False

    health.mark_market_data_stale(False)
    assert health.check_readiness() is True

    # Oracle failure should NOT affect readiness
    health.mark_oracle_failed(True)
    assert health.check_readiness() is True


def test_metrics_correctness():
    # Test that rejected signals metric correctly accepts the reason label
    before = REJECTED_SIGNALS.labels(reason="RISK_VETO")._value.get()
    REJECTED_SIGNALS.labels(reason="RISK_VETO").inc()
    after = REJECTED_SIGNALS.labels(reason="RISK_VETO")._value.get()
    assert after == before + 1


def test_unresolved_order_metrics():
    before = UNRESOLVED_INTENTS._value.get()
    UNRESOLVED_INTENTS.set(5)
    after = UNRESOLVED_INTENTS._value.get()
    assert after == 5

    # Restore to ensure no side effects
    UNRESOLVED_INTENTS.set(before)


def test_logging_alert_manager():
    mock_logger = Mock()
    manager = LoggingAlertManager(mock_logger)

    manager.send_alert("CRITICAL_ERROR", "System is down")

    mock_logger.critical.assert_called_once()
    args, kwargs = mock_logger.critical.call_args
    assert "CRITICAL_ERROR" in args[0]
    assert "System is down" in args[0]
    assert kwargs["extra"]["component"] == "AlertManager"
