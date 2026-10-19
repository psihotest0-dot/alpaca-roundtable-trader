import functools
import logging
import random
import time
import datetime
from typing import Any, Callable, TypeVar

from alpaca.trading.client import TradingClient
from pydantic import SecretStr

from trading_system.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_read_request(
    max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator for read-only requests using bounded exponential backoff with jitter.
    Will not retry indefinitely, and is not meant for order submission.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            while True:
                try:
                    logger.info(f"Initiating Alpaca read request: {func.__name__}")
                    result = func(*args, **kwargs)
                    logger.info(f"Successfully completed Alpaca read request: {func.__name__}")
                    return result
                except Exception as e:
                    # In a real app we might catch specific rate limit or network exceptions here
                    # For alpaca-py this might be APIError (HTTP 429, etc)
                    if retries >= max_retries:
                        logger.error(f"Failed {func.__name__} after {max_retries} retries: {e}")
                        raise

                    # Exponential backoff: 2^retries * base_delay
                    delay = base_delay * (2**retries)
                    # Cap at max_delay
                    delay = min(delay, max_delay)
                    # Add jitter (0 to 100% of delay)
                    delay = delay * random.uniform(0.5, 1.5)

                    logger.warning(
                        f"Exception in {func.__name__} (attempt {retries + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    retries += 1

        return wrapper

    return decorator


class AlpacaBroker:
    def __init__(self, settings: Settings) -> None:
        if not settings.is_paper:
            raise ValueError("Live trading is not supported. IS_PAPER must be True.")

        self.settings = settings
        api_key = (
            settings.alpaca_api_key.get_secret_value()
            if isinstance(settings.alpaca_api_key, SecretStr)
            else settings.alpaca_api_key
        )
        secret_key = (
            settings.alpaca_secret_key.get_secret_value()
            if isinstance(settings.alpaca_secret_key, SecretStr)
            else settings.alpaca_secret_key
        )

        self.client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

    @retry_read_request()
    def get_clock(self) -> Any:
        return self.client.get_clock()

    @retry_read_request()
    def get_account(self) -> Any:
        return self.client.get_account()

    @retry_read_request()
    def get_positions(self) -> Any:
        return self.client.get_all_positions()

    @retry_read_request()
    def get_open_orders(self) -> Any:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            nested=True,  # includes child orders
        )
        return self.client.get_orders(filter=req)

    @retry_read_request()
    def get_order_by_id(self, order_id: str) -> Any:
        return self.client.get_order_by_id(order_id=order_id)

    @retry_read_request()
    def get_order_by_client_id(self, client_order_id: str) -> Any:
        return self.client.get_order_by_client_id(client_id=client_order_id)

    @retry_read_request()
    def get_asset(self, symbol: str) -> Any:
        return self.client.get_asset(symbol_or_asset_id=symbol)

    @retry_read_request()
    def get_calendar(
        self, start: "datetime.date | None" = None, end: "datetime.date | None" = None
    ) -> Any:
        from alpaca.trading.requests import GetCalendarRequest

        req = GetCalendarRequest(start=start, end=end)
        return self.client.get_calendar(filters=req)

    def submit_order(self, order_data: Any) -> Any:
        logger.info(f"Initiating Alpaca order submission")
        result = self.client.submit_order(order_data=order_data)
        logger.info(f"Successfully completed Alpaca order submission")
        return result

    def cancel_order_by_id(self, order_id: str) -> Any:
        logger.info(f"Initiating Alpaca order cancellation for order_id={order_id}")
        result = self.client.cancel_order_by_id(order_id=order_id)
        logger.info(f"Successfully completed Alpaca order cancellation for order_id={order_id}")
        return result
