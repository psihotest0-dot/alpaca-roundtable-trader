import logging
import datetime
from typing import Any

from alpaca.trading.client import TradingClient
from pydantic import SecretStr

from trading_system.common.retry import retry_read_request
from trading_system.config import Settings

logger = logging.getLogger(__name__)


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
        logger.info("Initiating Alpaca order submission")
        result = self.client.submit_order(order_data=order_data)
        logger.info("Successfully completed Alpaca order submission")
        return result

    def cancel_order_by_id(self, order_id: str) -> None:
        logger.info(f"Initiating Alpaca order cancellation for order_id={order_id}")
        self.client.cancel_order_by_id(order_id=order_id)
        logger.info(f"Successfully completed Alpaca order cancellation for order_id={order_id}")
        return None
