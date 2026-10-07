import logging
from abc import ABC, abstractmethod


class AlertManager(ABC):
    """Interface for sending critical alerts."""

    @abstractmethod
    def send_alert(self, title: str, message: str) -> None:
        """Sends an alert with the given title and message."""
        pass


class LoggingAlertManager(AlertManager):
    """
    Implements AlertManager by logging the alert.
    No undocumented external services are used.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def send_alert(self, title: str, message: str) -> None:
        self.logger.critical(f"ALERT: {title} - {message}", extra={"component": "AlertManager"})
