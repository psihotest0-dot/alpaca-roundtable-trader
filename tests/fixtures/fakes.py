"""Hand-written fakes for the Alpaca broker and data client.

They replace the network only. Everything else in an integration test (data layer, risk,
execution, repositories, SQLite) is the real code.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pandas as pd
from alpaca.data.models import Trade

from tests.fixtures.bars import alpaca_shaped


class FakeStockClient:
    """Stands in for alpaca's StockHistoricalDataClient. Returns BarSet-like objects whose
    `.df` has the real (symbol, timestamp) MultiIndex shape, and records every request so
    tests can assert on what was asked for."""

    def __init__(
        self,
        frames_15m: dict[str, pd.DataFrame],
        frames_1h: dict[str, pd.DataFrame],
        *,
        live_prices: dict[str, float] | None = None,
        live_age_seconds: float = 0.0,
        live_error: Exception | None = None,
    ):
        self._frames = {15: frames_15m, 60: frames_1h}
        self.requests: list[Any] = []
        # Latest trades. Default: every symbol last traded, just now, at its last 15m close.
        # `live_prices` replaces that table (a symbol left out has not traded).
        self._live_prices = (
            live_prices
            if live_prices is not None
            else {symbol: float(frame["close"].iloc[-1]) for symbol, frame in frames_15m.items()}
        )
        self._live_age = timedelta(seconds=live_age_seconds)
        self._live_error = live_error
        self.trade_requests: list[Any] = []

    def get_stock_bars(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        timeframe = request.timeframe
        minutes = timeframe.amount_value * (60 if timeframe.unit_value.name == "Hour" else 1)
        symbols = request.symbol_or_symbols
        symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        shaped = [
            alpaca_shaped(symbol, self._frames[minutes][symbol])
            for symbol in symbols
            if symbol in self._frames[minutes]
        ]
        return SimpleNamespace(df=pd.concat(shaped) if shaped else pd.DataFrame())

    def get_stock_latest_trade(self, request: Any) -> dict[str, Trade]:
        """{symbol: alpaca Trade}, the real model, as StockHistoricalDataClient returns it."""
        self.trade_requests.append(request)
        if self._live_error is not None:
            raise self._live_error
        symbols = request.symbol_or_symbols
        symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        stamp = (datetime.now(timezone.utc) - self._live_age).isoformat().replace("+00:00", "Z")
        return {
            symbol: Trade(
                symbol,
                {
                    "t": stamp,
                    "x": "V",
                    "p": self._live_prices[symbol],
                    "s": 100,
                    "i": 1,
                    "c": ["@"],
                },
            )
            for symbol in symbols
            if symbol in self._live_prices
        }


class FakeBroker:
    """The subset of AlpacaBroker that the runtime and ExecutionEngine use."""

    def __init__(
        self, equity: str = "100000", market_open: bool = True, last_equity: str | None = None
    ):
        self.equity = equity
        self.last_equity = last_equity
        self.market_open = market_open
        self.positions: list[Any] = []
        self.open_orders: list[Any] = []
        self.submitted: list[SimpleNamespace] = []
        self.sent_requests: list[Any] = []  # the order requests, as the engine built them
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
            **({} if self.last_equity is None else {"last_equity": self.last_equity}),
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
        self.sent_requests.append(request)
        return order

    def get_order_by_client_id(self, client_order_id: str) -> SimpleNamespace:
        for order in self.submitted:
            if order.client_order_id == client_order_id:
                return order
        raise LookupError(client_order_id)

    def cancel_order_by_id(self, order_id: str) -> None:
        self.cancelled.append(order_id)
