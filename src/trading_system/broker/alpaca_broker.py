import logging
import datetime
from typing import Any

from alpaca.trading.client import TradingClient
from pydantic import SecretStr

from trading_system.common.retry import retry_read_request
from trading_system.config import Settings

logger = logging.getLogger(__name__)

_FILLS_PAGE_SIZE = 100


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
    def get_recent_orders_for_symbol(self, symbol: str, limit: int = 50) -> Any:
        """The latest orders of any status for one symbol, legs nested under their parent.

        Unlike get_open_orders this also returns a filled bracket parent, whose stop leg stays
        `held` and is therefore missing from every list of open orders."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL, symbols=[symbol], nested=True, limit=limit
        )
        return self.client.get_orders(filter=req)

    @retry_read_request()
    def get_fills(self, after: datetime.datetime) -> list[dict[str, Any]]:
        """The executions (fills) since `after`, oldest first, as Alpaca reports them:
        dicts with id, order_id, price, qty, side, symbol, type and transaction_time.

        alpaca-py has no method for GET /account/activities/FILL, hence the raw call."""
        fills: list[dict[str, Any]] = []
        params: dict[str, Any] = {
            "after": after.astimezone(datetime.timezone.utc).isoformat(),
            "direction": "asc",
            "page_size": _FILLS_PAGE_SIZE,
        }
        while True:
            page = self.client.get("/account/activities/FILL", params)
            fills.extend(page)
            if len(page) < _FILLS_PAGE_SIZE:
                return fills
            params["page_token"] = page[-1]["id"]

    @retry_read_request()
    def get_portfolio_history(self, days: int = 30) -> Any:
        """Alpaca's own end-of-day equity of the account for the last `days` days (one point
        per trading day; see observability/equity_check.py for what its stamps mean)."""
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        return self.client.get_portfolio_history(
            GetPortfolioHistoryRequest(period=f"{days}D", timeframe="1D")
        )

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
