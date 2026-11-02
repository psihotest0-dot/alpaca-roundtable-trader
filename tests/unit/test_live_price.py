from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from tests.fixtures.fakes import FakeStockClient
from trading_system.config import Settings
from trading_system.data.alpaca_data import AlpacaData
from trading_system.data.live_price import LivePrice, UnusableLivePrice, check_live_price

NOW = datetime(2026, 11, 2, 15, 30, tzinfo=timezone.utc)
D = Decimal


def _check(
    price: str = "100.00",
    *,
    age: float = 1.0,
    close: str = "100.00",
    max_age: float = 120,
    max_deviation: str = "0.05",
) -> Decimal:
    live = LivePrice(price=D(price), timestamp=NOW - timedelta(seconds=age))
    return check_live_price(
        live,
        reference_close=D(close),
        now=NOW,
        max_age_seconds=max_age,
        max_deviation_pct=D(max_deviation),
    )


def test_a_fresh_trade_near_the_last_bar_is_the_price() -> None:
    assert _check("100.40", age=3, close="100.00") == D("100.40")


def test_the_limits_themselves_are_still_accepted() -> None:
    assert _check(age=120, max_age=120) == D("100.00")
    assert _check("105.00", close="100.00", max_deviation="0.05") == D("105.00")


def test_no_trade_means_no_price() -> None:
    with pytest.raises(UnusableLivePrice, match="no trade price"):
        check_live_price(
            None,
            reference_close=D("100"),
            now=NOW,
            max_age_seconds=120,
            max_deviation_pct=D("0.05"),
        )


def test_a_stale_trade_is_refused() -> None:
    with pytest.raises(UnusableLivePrice, match=r"601s old \(limit 120s\)"):
        _check(age=601)


def test_a_trade_from_the_future_is_refused_but_small_clock_skew_is_not() -> None:
    """This machine's clock was measured 2 s behind Alpaca's; that must not matter."""
    assert _check(age=-2) == D("100.00")
    with pytest.raises(UnusableLivePrice, match="in the future"):
        _check(age=-30)


def test_a_price_far_from_the_last_bar_is_refused_in_both_directions() -> None:
    with pytest.raises(UnusableLivePrice, match=r"10\.00% from the last bar close"):
        _check("110.00")
    with pytest.raises(UnusableLivePrice, match=r"10\.00% from the last bar close"):
        _check("90.00")


@pytest.mark.parametrize("price, close", [("0", "100"), ("-1", "100"), ("100", "0")])
def test_non_positive_prices_are_refused(price: str, close: str) -> None:
    with pytest.raises(UnusableLivePrice, match="not positive"):
        _check(price, close=close)


def test_a_timestamp_without_a_timezone_is_read_as_utc() -> None:
    naive = LivePrice(price=D("100"), timestamp=(NOW - timedelta(seconds=5)).replace(tzinfo=None))

    assert check_live_price(
        naive,
        reference_close=D("100"),
        now=NOW,
        max_age_seconds=120,
        max_deviation_pct=D("0.05"),
    ) == D("100")


def _data(client: FakeStockClient) -> AlpacaData:
    data = AlpacaData(Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s"))
    data.client = client  # type: ignore[assignment]
    return data


def test_latest_prices_come_from_one_request_for_all_symbols() -> None:
    """The adapter reads alpaca-py's real Trade model (the fake returns it) and asks once."""
    client = FakeStockClient({}, {}, live_prices={"AAPL": 150.25, "MSFT": 410.5})
    data = _data(client)

    prices = data.fetch_latest_prices(["AAPL", "MSFT", "NODATA"])

    assert [r.symbol_or_symbols for r in client.trade_requests] == [["AAPL", "MSFT", "NODATA"]]
    assert client.trade_requests[0].feed.value == "iex"
    assert set(prices) == {"AAPL", "MSFT"}  # a symbol that has not traded is left out
    assert prices["AAPL"].price == D("150.25")
    assert abs(datetime.now(timezone.utc) - prices["AAPL"].timestamp) < timedelta(seconds=5)


def test_latest_prices_for_no_symbols_makes_no_request() -> None:
    client = FakeStockClient({}, {})

    assert _data(client).fetch_latest_prices([]) == {}
    assert client.trade_requests == []
