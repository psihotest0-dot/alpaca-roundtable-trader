"""AlpacaData's bulk loading, with a fake Alpaca client (the network is the only fake)."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.timeframe import TimeFrame

from tests.fixtures.bars import make_bars
from tests.fixtures.fakes import FakeStockClient
from trading_system.config import Settings
from trading_system.data.alpaca_data import FIFTEEN_MINUTES, ONE_HOUR, AlpacaData


@pytest.fixture
def data() -> AlpacaData:
    return AlpacaData(Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s"))


def _client(symbols: list[str]) -> FakeStockClient:
    bars_15m, bars_1h = make_bars()
    return FakeStockClient({s: bars_15m for s in symbols}, {s: bars_1h for s in symbols})


def test_the_timeframes_are_valid_alpaca_timeframes() -> None:
    assert (FIFTEEN_MINUTES.amount_value, FIFTEEN_MINUTES.unit_value.name) == (15, "Minute")
    assert str(ONE_HOUR) == str(TimeFrame.Hour)


def test_many_symbols_cost_one_request_with_adjusted_prices_on_the_configured_feed(
    data: AlpacaData,
) -> None:
    client = _client(["AAPL", "MSFT", "NVDA"])
    data.client = client
    now = datetime.now(timezone.utc)

    result = data.fetch_bars_bulk(
        ["AAPL", "MSFT", "NVDA"], FIFTEEN_MINUTES, now - timedelta(days=5), now
    )

    assert sorted(result) == ["AAPL", "MSFT", "NVDA"]
    (request,) = client.requests
    assert request.symbol_or_symbols == ["AAPL", "MSFT", "NVDA"]
    assert request.feed == DataFeed.IEX
    assert request.adjustment == Adjustment.ALL


def test_a_request_can_name_its_own_feed_and_the_configured_one_is_left_alone(
    data: AlpacaData,
) -> None:
    client = _client(["AAPL"])
    data.client = client
    now = datetime.now(timezone.utc)

    data.fetch_bars_bulk(["AAPL"], FIFTEEN_MINUTES, now - timedelta(days=5), now, feed="sip")
    data.fetch_bars_bulk(["AAPL"], FIFTEEN_MINUTES, now - timedelta(days=5), now)

    assert [r.feed for r in client.requests] == [DataFeed.SIP, DataFeed.IEX]
    assert data.feed == "iex"


def test_a_sip_request_ends_before_the_delay_the_free_plan_allows() -> None:
    from tests.fixtures.fakes import asked

    data = AlpacaData(Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s"))
    client = _client(["AAPL"])
    data.client = client
    now = datetime.now(timezone.utc)
    long_ago = now - timedelta(days=3)

    data.fetch_bars_bulk(["AAPL"], FIFTEEN_MINUTES, now - timedelta(days=5), now, feed="sip")
    data.fetch_bars_bulk(["AAPL"], FIFTEEN_MINUTES, now - timedelta(days=9), long_ago, feed="sip")
    data.fetch_bars_bulk(["AAPL"], FIFTEEN_MINUTES, now - timedelta(days=5), now, feed="iex")

    asked_sip_to_now, asked_sip_to_the_past, asked_iex = (asked(r.end) for r in client.requests)
    assert asked_sip_to_now <= now - timedelta(minutes=15)  # held back from "now"
    assert asked_sip_to_the_past == long_ago  # an end that is old enough is not touched
    assert asked_iex >= now - timedelta(seconds=5)  # IEX has no such limit


def test_symbols_without_bars_are_left_out_not_invented(data: AlpacaData) -> None:
    data.client = _client(["AAPL"])
    now = datetime.now(timezone.utc)

    result = data.fetch_bars_bulk(["AAPL", "NOPE"], FIFTEEN_MINUTES, now - timedelta(days=5), now)

    assert list(result) == ["AAPL"]


def test_no_symbols_means_no_request(data: AlpacaData) -> None:
    client = _client(["AAPL"])
    data.client = client
    now = datetime.now(timezone.utc)

    assert data.fetch_bars_bulk([], FIFTEEN_MINUTES, now - timedelta(days=5), now) == {}
    assert client.requests == []


def test_load_universe_makes_two_requests_whatever_the_number_of_symbols(data: AlpacaData) -> None:
    client = _client(["AAPL", "MSFT"])
    data.client = client

    bars_15m, bars_1h = data.load_universe(["AAPL", "MSFT"], datetime.now(timezone.utc), 30, 90)

    assert sorted(bars_15m) == sorted(bars_1h) == ["AAPL", "MSFT"]
    assert [str(r.timeframe) for r in client.requests] == ["15Min", "1Hour"]


def test_load_universe_logs_where_the_time_of_a_pass_goes(
    data: AlpacaData, caplog: pytest.LogCaptureFixture
) -> None:
    data.client = _client(["AAPL", "MSFT"])

    with caplog.at_level("INFO", logger="trading_system.data.alpaca_data"):
        data.load_universe(["AAPL", "MSFT"], datetime.now(timezone.utc), 30, 90)

    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Market data")]
    assert line.startswith("Market data fetched: 15m ") and ", 1h " in line
    assert line.endswith("(2 symbols).") and " rows in " in line and " s, " in line


def test_load_universe_skips_a_symbol_that_lacks_one_timeframe(data: AlpacaData) -> None:
    bars_15m, bars_1h = make_bars()
    data.client = FakeStockClient({"AAPL": bars_15m, "MSFT": bars_15m}, {"AAPL": bars_1h})

    out_15m, out_1h = data.load_universe(["AAPL", "MSFT"], datetime.now(timezone.utc), 30, 90)

    assert list(out_15m) == list(out_1h) == ["AAPL"]


def test_load_universe_returns_only_closed_bars(data: AlpacaData) -> None:
    bars_15m, bars_1h = make_bars()
    data.client = FakeStockClient({"AAPL": bars_15m}, {"AAPL": bars_1h})
    now = datetime.now(timezone.utc)

    out_15m, _ = data.load_universe(["AAPL"], now, 30, 90)

    assert (out_15m["AAPL"].index + timedelta(minutes=15) <= now).all()


def test_a_programming_error_in_the_client_is_not_retried(
    data: AlpacaData, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        calls = 0

        def get_stock_bars(self, request: Any) -> Any:
            Broken.calls += 1
            raise TypeError("a bug, not a network problem")

    monkeypatch.setattr("time.sleep", lambda _s: pytest.fail("slept before giving up on a bug"))
    data.client = Broken()
    now = datetime.now(timezone.utc)

    with pytest.raises(TypeError):
        data.fetch_bars_bulk(["AAPL"], FIFTEEN_MINUTES, now - timedelta(days=5), now)

    assert Broken.calls == 1
