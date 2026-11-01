"""Hand-written fakes for the Alpaca broker and data client.

They replace the network only. Everything else in an integration test (risk,
execution, repositories, SQLite) is the real code.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pandas as pd

from tests.fixtures.bars import alpaca_shaped, with_timestamp_column
from trading_system.data.alpaca_data import AlpacaData


class FakeStockClient:
    """Stands in for alpaca's StockHistoricalDataClient. Returns BarSet-like
    objects whose `.df` has the real (symbol, timestamp) MultiIndex shape."""

    def __init__(self, frames_15m: dict[str, pd.DataFrame], frames_1h: dict[str, pd.DataFrame]):
        self._frames = {15: frames_15m, 60: frames_1h}

    def get_stock_bars(self, request: Any) -> SimpleNamespace:
        timeframe = request.timeframe
        minutes = timeframe.amount_value * (60 if timeframe.unit_value.name == "Hour" else 1)
        symbol = request.symbol_or_symbols
        frame = self._frames[minutes][symbol]
        return SimpleNamespace(data={symbol: [1]}, df=alpaca_shaped(symbol, frame))


class FakeData(AlpacaData):
    """AlpacaData with the network cut out. It keeps the real
    clean_and_deduplicate / align_and_filter. fetch_bars returns frames that
    satisfy BOTH consumers (datetime index and a `timestamp` column), which
    hides the data-shape defect so that later defects can be observed."""

    def __init__(self, frames_15m: dict[str, pd.DataFrame], frames_1h: dict[str, pd.DataFrame]):
        self._frames = {15: frames_15m, 60: frames_1h}
        self.calls: list[tuple[str, str]] = []

    def fetch_bars(self, symbol: str, timeframe: Any, start: Any, end: Any) -> pd.DataFrame:
        self.calls.append((symbol, str(timeframe)))
        minutes = timeframe.amount_value * (60 if timeframe.unit_value.name == "Hour" else 1)
        return with_timestamp_column(self._frames[minutes][symbol])


class FakeBroker:
    """The subset of AlpacaBroker that the runtime and ExecutionEngine use."""

    def __init__(self, equity: str = "100000", market_open: bool = True):
        self.equity = equity
        self.market_open = market_open
        self.positions: list[Any] = []
        self.open_orders: list[Any] = []
        self.submitted: list[SimpleNamespace] = []
        self.cancelled: list[str] = []

    def get_clock(self) -> SimpleNamespace:
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            is_open=self.market_open, timestamp=now, next_open=now + timedelta(hours=16)
        )

    def get_account(self) -> SimpleNamespace:
        return SimpleNamespace(
            equity=self.equity,
            cash=self.equity,
            buying_power=str(float(self.equity) * 4),
            daytrade_count=0,
        )

    def get_positions(self) -> list[Any]:
        return self.positions

    def get_open_orders(self) -> list[Any]:
        return self.open_orders

    def get_asset(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(symbol=symbol, tradable=True, price_increment=0.01)

    def submit_order(self, request: Any) -> SimpleNamespace:
        now = datetime.now(timezone.utc)
        order = SimpleNamespace(
            id=str(uuid.uuid4()),
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side="buy",
            order_type="limit",
            status="accepted",
            qty=str(request.qty),
            filled_qty="0",
            limit_price=str(request.limit_price),
            stop_price=None,
            created_at=now,
            updated_at=now,
            legs=[],
        )
        self.submitted.append(order)
        return order

    def get_order_by_client_id(self, client_order_id: str) -> SimpleNamespace:
        for order in self.submitted:
            if order.client_order_id == client_order_id:
                return order
        raise LookupError(client_order_id)

    def cancel_order_by_id(self, order_id: str) -> None:
        self.cancelled.append(order_id)
