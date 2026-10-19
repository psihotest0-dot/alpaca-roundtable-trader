import json
import logging
import re
from datetime import datetime
from typing import Any, Dict


class SensitiveDataRedactor:
    """Redacts sensitive information from log records."""

    # Patterns or keys to look for in dictionaries or strings
    SENSITIVE_KEYS = {"api_key", "secret", "authorization", "password", "token"}

    # Regex to catch Authorization headers, Bearer tokens, etc., in strings
    SENSITIVE_PATTERNS = [
        re.compile(r"Bearer\s+[A-Za-z0-9\-\._~]+", re.IGNORECASE),
        re.compile(r"api_key=['\"]?[A-Za-z0-9\-\._]+['\"]?", re.IGNORECASE),
        re.compile(r"secret_key=['\"]?[A-Za-z0-9\-\._]+['\"]?", re.IGNORECASE),
        re.compile(r"authorization:\s*['\"]?[A-Za-z0-9\-\._\s]+['\"]?", re.IGNORECASE),
    ]

    @classmethod
    def redact(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: cls._redact_value(k, v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.redact(v) for v in data]
        elif isinstance(data, str):
            return cls._redact_string(data)
        return data

    @classmethod
    def _redact_value(cls, key: str, value: Any) -> Any:
        key_lower = str(key).lower()
        if (
            any(sensitive_key in key_lower for sensitive_key in cls.SENSITIVE_KEYS)
            or key_lower == "env"
            or key_lower == "environment"
        ):
            return "***REDACTED***"
        return cls.redact(value)

    @classmethod
    def _redact_string(cls, text: str) -> str:
        redacted_text = text
        for pattern in cls.SENSITIVE_PATTERNS:
            redacted_text = pattern.sub("***REDACTED***", redacted_text)
        return redacted_text


class JSONFormatter(logging.Formatter):
    """
    JSON Formatter for structured logs.
    Includes explicit fields to match trading observability requirements.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Standard fields from LogRecord
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).astimezone().isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "event": record.getMessage(),
        }

        # Trading specific fields
        optional_fields = [
            "signal_id",
            "intent_id",
            "broker_order_id",
            "symbol",
            "reason_code",
        ]

        for field in optional_fields:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
            else:
                log_data[field] = None

        # Include additional extra fields
        if hasattr(record, "extra_data") and isinstance(getattr(record, "extra_data"), dict):
            for k, v in getattr(record, "extra_data").items():
                if k not in log_data:
                    log_data[k] = v

        # Add exception info if any
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Redact sensitive data
        redacted_data = SensitiveDataRedactor.redact(log_data)

        # Redact environment info specifically - never log the full environment
        if "environment" in redacted_data or "env" in redacted_data:
            if "environment" in redacted_data:
                redacted_data["environment"] = "***REDACTED***"
            if "env" in redacted_data:
                redacted_data["env"] = "***REDACTED***"

        return json.dumps(redacted_data)


def setup_json_logger(
    name: str, level: int = logging.INFO, log_file: str | None = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JSONFormatter())
    logger.addHandler(stream_handler)

    # Add file handler if log_file is provided
    if log_file:
        import os

        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger
