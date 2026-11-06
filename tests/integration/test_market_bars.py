"""The candle store and its price policy (Phase 3.2)."""

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest
from alpaca.data.enums import DataFeed

from tests.fixtures.bars import daily_bars
from tests.fixtures.fakes import FakeHistoryClient, asked
from trading_system.data.alpaca_data import AlpacaData
from trading_system.data.history import (
    ADJUSTMENT,
    OVERLAP,
    prices_drifted,
    stale_symbols,
    update_history,
)
from trading_system.storage.models import MarketBarModel
from trading_system.storage.repositories import MarketBarRepository

UTC = timezone.utc
# 21:07 UTC: the 21:00 bar is still forming (15m closes at 21:15, 1h at 22:00).
NOW = datetime(2026, 11, 2, 21, 7, tzinfo=UTC)
LENGTH = {15: timedelta(minutes=15), 60: timedelta(hours=1)}


def candles(last_start: datetime, periods: int, minutes: int, base: float = 100.0) -> pd.DataFrame:
    """Rising closes, ending with the bar that starts at `last_start`, the shape alpaca-py gives
    (timestamp index, with the extra columns it adds)."""
    index = pd.date_range(end=last_start, periods=periods, freq=f"{minutes}min", tz="UTC")
    closes = base + np.arange(periods) * 0.01
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": 1000.0,
            "trade_count": 10.0,
            "vwap": closes,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def history(*symbols: str, days: int = 20) -> FakeHistoryClient:
    """`days` of bars for every symbol, the newest one still forming at NOW."""
    last_15m = datetime(2026, 11, 2, 21, 0, tzinfo=UTC)
    return FakeHistoryClient(
        {
            15: {s: candles(last_15m, days * 96, 15, base=100 + i) for i, s in enumerate(symbols)},
            60: {s: candles(last_15m, days * 24, 60, base=100 + i) for i, s in enumerate(symbols)},
        }
    )


@pytest.fixture
def store(session_factory: Any) -> MarketBarRepository:
    return MarketBarRepository(session_factory)


@pytest.fixture
def make_data(settings: Any) -> Any:
    def _make(client: FakeHistoryClient) -> AlpacaData:
        data = AlpacaData(settings)
        data.client = client
        return data

    return _make


# ------------------------------------------------------------------------ the repository


def test_bars_come_back_as_they_went_in(store: MarketBarRepository) -> None:
    bars = candles(datetime(2026, 11, 2, 15, 0, tzinfo=UTC), 10, 15)
    bars.loc[bars.index[3], ["vwap", "trade_count"]] = np.nan  # Alpaca leaves gaps here

    store.replace("AAPL", "15m", bars, adjustment="all", fetched_at=NOW)
    back = store.load(["AAPL"], "15m")["AAPL"]

    assert list(back.index) == list(bars.index) and str(back.index.tz) == "UTC"
    assert (back["timestamp"] == back.index).all()  # the canonical shape of data.bars
    assert np.allclose(
        back[["open", "high", "low", "close", "volume"]],
        bars[["open", "high", "low", "close", "volume"]],
    )
    assert np.isnan(back["vwap"].iloc[3]) and np.isnan(back["trade_count"].iloc[3])
    assert back["trade_count"].iloc[0] == 10


def test_replacing_overwrites_the_same_instants_and_keeps_the_rest(
    store: MarketBarRepository,
) -> None:
    first = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
    store.replace(
        "AAPL", "15m", candles(first + timedelta(hours=2), 9, 15), adjustment="all", fetched_at=NOW
    )
    corrected = candles(first + timedelta(hours=1), 3, 15, base=500.0)  # three bars in the middle

    store.replace("AAPL", "15m", corrected, adjustment="all", fetched_at=NOW)

    back = store.load(["AAPL"], "15m")["AAPL"]
    assert len(back) == 9  # nothing doubled, nothing lost
    assert (back.loc[corrected.index, "close"] > 400).all()  # the middle is the new copy
    assert (back.drop(corrected.index)["close"] < 200).all()  # the ends are the old one


def test_everything_replaces_the_whole_series_and_only_that_series(
    store: MarketBarRepository,
) -> None:
    end = datetime(2026, 11, 2, 16, 0, tzinfo=UTC)
    store.replace("AAPL", "15m", candles(end, 20, 15), adjustment="all", fetched_at=NOW)
    store.replace("AAPL", "1h", candles(end, 20, 60), adjustment="all", fetched_at=NOW)
    store.replace("MSFT", "15m", candles(end, 20, 15), adjustment="all", fetched_at=NOW)

    store.replace(
        "AAPL", "15m", candles(end, 5, 15), adjustment="all", fetched_at=NOW, everything=True
    )

    assert len(store.load(["AAPL"], "15m")["AAPL"]) == 5
    assert len(store.load(["AAPL"], "1h")["AAPL"]) == 20
    assert len(store.load(["MSFT"], "15m")["MSFT"]) == 20


def test_a_window_starts_inclusive_ends_exclusive_and_may_use_any_timezone(
    store: MarketBarRepository,
) -> None:
    end = datetime(2026, 11, 2, 16, 0, tzinfo=UTC)
    store.replace(
        "AAPL", "15m", candles(end, 8, 15), adjustment="all", fetched_at=NOW
    )  # 14:15 .. 16:00
    plus_two = timezone(timedelta(hours=2))

    window = store.load(
        ["AAPL", "MSFT"],
        "15m",
        start=datetime(2026, 11, 2, 16, 30, tzinfo=plus_two),  # = 14:30 UTC
        end=datetime(2026, 11, 2, 17, 45, tzinfo=plus_two),  # = 15:45 UTC
    )

    assert list(window) == ["AAPL"]  # MSFT has nothing, so it is absent
    stamps = [t.strftime("%H:%M") for t in window["AAPL"].index]
    assert stamps == ["14:30", "14:45", "15:00", "15:15", "15:30"]


def test_span_and_price_basis(store: MarketBarRepository) -> None:
    assert store.span("AAPL", "15m") is None
    end = datetime(2026, 11, 2, 16, 0, tzinfo=UTC)
    store.replace("AAPL", "15m", candles(end, 4, 15), adjustment="raw", fetched_at=NOW)

    first, last = store.span("AAPL", "15m") or (None, None)

    assert (first, last) == (datetime(2026, 11, 2, 15, 15, tzinfo=UTC), end)
    assert store.has_other_adjustment("AAPL", "15m", "all") is True
    assert store.has_other_adjustment("AAPL", "15m", "raw") is False
    assert store.has_other_adjustment("AAPL", "1h", "all") is False  # nothing stored there


# --------------------------------------------------------------------------- the update


def test_the_first_update_stores_the_closed_bars_and_not_the_one_still_forming(
    store: MarketBarRepository, make_data: Any
) -> None:
    client = history("AAPL")

    report = update_history(
        make_data(client), store, ["AAPL"], ["15m", "1h"], start=NOW - timedelta(days=10), now=NOW
    )

    bars_15m = store.load(["AAPL"], "15m")["AAPL"]
    bars_1h = store.load(["AAPL"], "1h")["AAPL"]
    assert bars_15m.index[-1] == datetime(2026, 11, 2, 20, 45, tzinfo=UTC)  # 21:00 closes at 21:15
    assert bars_1h.index[-1] == datetime(2026, 11, 2, 20, 0, tzinfo=UTC)  # 21:00 closes at 22:00
    assert report.rows[("AAPL", "15m")] == len(bars_15m)
    assert report.downloaded_again == [] and report.no_data == []
    assert len(client.requests) == 2  # one request per timeframe, however many symbols


def test_many_symbols_are_downloaded_in_chunks_of_25(
    store: MarketBarRepository, make_data: Any
) -> None:
    """One request for ~150 symbols of 15-minute bars paged for a quarter of an hour at a few KB/s."""
    names = [f"S{i:03d}" for i in range(60)]
    client = history(*names, days=3)

    update_history(
        make_data(client), store, names, ["15m"], start=NOW - timedelta(days=10), now=NOW
    )

    assert sorted(len(r.symbol_or_symbols) for r in client.requests) == [10, 25, 25]
    assert len(store.load(names, "15m")) == 60  # every symbol is in


def test_a_chunk_that_arrived_is_kept_if_a_later_one_fails(
    store: MarketBarRepository, make_data: Any
) -> None:
    names = [f"S{i:03d}" for i in range(60)]
    client = history(*names, days=3)
    real = client.get_stock_bars
    calls: list[int] = []

    def flaky(request: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("alpaca refuses")  # not a transient error: no retry
        return real(request)

    client.get_stock_bars = flaky  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        update_history(
            make_data(client), store, names, ["15m"], start=NOW - timedelta(days=10), now=NOW
        )

    assert len(store.load(names, "15m")) == 25  # the first chunk is safe


def test_the_next_update_asks_for_the_last_week_only_and_adds_what_is_new(
    store: MarketBarRepository, make_data: Any
) -> None:
    client = history("AAPL", "MSFT")
    data = make_data(client)
    start = NOW - timedelta(days=10)
    update_history(data, store, ["AAPL", "MSFT"], ["15m"], start=start, now=NOW)
    stored_last = store.span("AAPL", "15m")[1]  # type: ignore[index]
    client.requests.clear()

    later = NOW + timedelta(hours=1)
    client._frames[15]["AAPL"] = pd.concat(
        [
            client._frames[15]["AAPL"],
            candles(datetime(2026, 11, 2, 22, 0, tzinfo=UTC), 4, 15, base=101.0),
        ]
    )
    update_history(data, store, ["AAPL", "MSFT"], ["15m"], start=start, now=later)

    (request,) = client.requests  # one request for both symbols, from a week back
    assert asked(request.start) == stored_last - OVERLAP
    bars = store.load(["AAPL"], "15m")["AAPL"]
    assert bars.index[-1] > stored_last  # the new bars are in
    assert bars.index.is_unique and bars.index.is_monotonic_increasing


def test_a_dividend_makes_the_stored_prices_stale_and_that_symbol_is_downloaded_again(
    store: MarketBarRepository, make_data: Any
) -> None:
    client = history("AAPL", "MSFT")
    data = make_data(client)
    start = NOW - timedelta(days=10)
    update_history(data, store, ["AAPL", "MSFT"], ["15m"], start=start, now=NOW)
    before = store.load(["AAPL"], "15m")["AAPL"]["close"]
    client.requests.clear()

    # Alpaca re-adjusts all of AAPL's past (half a percent is a big dividend for a blue chip).
    client.readjust("AAPL", 0.995, before=NOW - timedelta(days=1))
    report = update_history(
        data, store, ["AAPL", "MSFT"], ["15m"], start=start, now=NOW + timedelta(hours=1)
    )

    assert report.downloaded_again == [("AAPL", "15m")]  # MSFT did not change
    after = store.load(["AAPL"], "15m")["AAPL"]["close"]
    old = after.index < NOW - timedelta(days=1)
    assert np.allclose(after[old].to_numpy(), before.reindex(after.index)[old].to_numpy() * 0.995)
    assert len(client.requests_for("MSFT")) == 1 and len(client.requests_for("AAPL")) == 2
    # No half-and-half: the store holds exactly what Alpaca serves now, closed bars only.
    served = client._frames[15]["AAPL"]
    served = served[
        (served.index >= start) & (served.index + LENGTH[15] <= NOW + timedelta(hours=1))
    ]
    assert list(after.index) == list(served.index)
    assert np.allclose(after.to_numpy(), served["close"].to_numpy())


def test_a_late_correction_is_picked_up_without_downloading_everything_again(
    store: MarketBarRepository, make_data: Any
) -> None:
    client = history("AAPL")
    data = make_data(client)
    start = NOW - timedelta(days=10)
    update_history(data, store, ["AAPL"], ["15m"], start=start, now=NOW)
    recent = datetime(2026, 11, 2, 20, 30, tzinfo=UTC)
    client.correct("AAPL", 15, recent, close=123.45)
    client.requests.clear()

    report = update_history(data, store, ["AAPL"], ["15m"], start=start, now=NOW)

    assert store.load(["AAPL"], "15m")["AAPL"].loc[recent, "close"] == 123.45
    assert report.downloaded_again == []
    assert len(client.requests) == 1


def test_history_that_does_not_reach_back_far_enough_is_downloaded_from_the_start(
    store: MarketBarRepository, make_data: Any
) -> None:
    client = history("AAPL", days=20)
    data = make_data(client)
    update_history(data, store, ["AAPL"], ["15m"], start=NOW - timedelta(days=5), now=NOW)
    short = len(store.load(["AAPL"], "15m")["AAPL"])

    report = update_history(data, store, ["AAPL"], ["15m"], start=NOW - timedelta(days=15), now=NOW)

    assert len(store.load(["AAPL"], "15m")["AAPL"]) > 2 * short
    assert report.downloaded_again == [("AAPL", "15m")]


def test_bars_stored_on_another_price_basis_are_replaced(
    store: MarketBarRepository, make_data: Any
) -> None:
    client = history("AAPL")
    start = NOW - timedelta(days=10)
    raw = client._frames[15]["AAPL"].loc[lambda f: f.index >= start]
    store.replace("AAPL", "15m", raw.iloc[:-1], adjustment="raw", fetched_at=NOW)

    update_history(make_data(client), store, ["AAPL"], ["15m"], start=start, now=NOW)

    assert store.has_other_adjustment("AAPL", "15m", ADJUSTMENT) is False


def test_a_symbol_alpaca_knows_nothing_about_is_reported(
    store: MarketBarRepository, make_data: Any
) -> None:
    report = update_history(
        make_data(history("AAPL")),
        store,
        ["AAPL", "ZZZZ"],
        ["15m"],
        start=NOW - timedelta(days=5),
        now=NOW,
    )

    assert report.no_data == [("ZZZZ", "15m")]
    assert "ZZZZ" not in store.load(["AAPL", "ZZZZ"], "15m")


@pytest.mark.parametrize(
    "stored_factor, expect",
    [
        (1.0, False),  # the same prices
        (1.00001, False),  # a tenth of a basis point is noise, not a dividend
        (1.002, True),  # 0.2 %, a dividend
        (2.0, True),  # a split
    ],
)
def test_prices_drifted_compares_the_oldest_bars_both_copies_share(
    stored_factor: float, expect: bool
) -> None:
    fresh = candles(datetime(2026, 11, 2, 16, 0, tzinfo=UTC), 60, 15)
    stored = fresh.copy()
    stored["close"] = stored["close"] * stored_factor

    assert prices_drifted(stored, fresh) is expect


def test_prices_drifted_looks_only_at_the_oldest_bars_so_a_recent_correction_is_not_drift() -> None:
    fresh = candles(datetime(2026, 11, 2, 16, 0, tzinfo=UTC), 60, 15)
    stored = fresh.copy()
    stored.iloc[-5:, stored.columns.get_loc("close")] *= 1.05  # only the newest bars differ

    assert prices_drifted(stored, fresh) is False


def test_nothing_to_compare_means_the_stored_copy_cannot_be_trusted() -> None:
    fresh = candles(datetime(2026, 11, 2, 16, 0, tzinfo=UTC), 10, 15)
    elsewhere = candles(datetime(2026, 10, 1, 16, 0, tzinfo=UTC), 10, 15)

    assert prices_drifted(None, fresh) is True
    assert prices_drifted(elsewhere, fresh) is True


def test_the_rows_carry_the_basis_and_the_time_they_were_fetched(
    store: MarketBarRepository, make_data: Any, session_factory: Any
) -> None:
    update_history(
        make_data(history("AAPL")), store, ["AAPL"], ["1h"], start=NOW - timedelta(days=3), now=NOW
    )

    with session_factory() as session:
        row = session.query(MarketBarModel).first()
        assert (row.adjustment, row.timeframe) == (ADJUSTMENT, "1h")
        assert row.fetched_at is not None


# ------------------------------------------------------------------------------ daily bars


def daily_history(*symbols: str, days: int = 30) -> FakeHistoryClient:
    """`days` of daily bars, the newest being the day that is still open at NOW (Alpaca stamps a
    daily bar with the start of its New York day: 05:00 UTC in winter)."""
    today = datetime(2026, 11, 2, 5, 0, tzinfo=UTC)
    return FakeHistoryClient(
        {1440: {s: daily_bars(today, days, base=100 + i) for i, s in enumerate(symbols)}}
    )


def test_daily_bars_are_stored_closed_only_and_asked_for_on_the_full_market_feed(
    store: MarketBarRepository, make_data: Any
) -> None:
    client = daily_history("AAPL")

    report = update_history(
        make_data(client), store, ["AAPL"], ["1d"], start=NOW - timedelta(days=60), now=NOW
    )

    bars = store.load(["AAPL"], "1d")["AAPL"]
    # Monday 2 November is still open at 21:07 UTC; the bar of Sunday 1 November is the newest.
    assert bars.index[-1] == datetime(2026, 11, 1, 5, 0, tzinfo=UTC)
    assert report.rows[("AAPL", "1d")] == len(bars) == 29
    (request,) = client.requests
    assert request.feed == DataFeed.SIP  # whatever DATA_FEED says: the free IEX feed is no basis


def test_the_intraday_series_stay_on_the_configured_feed(
    store: MarketBarRepository, make_data: Any
) -> None:
    client = history("AAPL")

    update_history(
        make_data(client), store, ["AAPL"], ["15m", "1h"], start=NOW - timedelta(days=3), now=NOW
    )

    assert {r.feed for r in client.requests} == {DataFeed.IEX}


def test_a_daily_bar_counts_as_closed_from_the_next_midnight_in_new_york(
    store: MarketBarRepository, make_data: Any
) -> None:
    client = daily_history("AAPL")
    data = make_data(client)
    start = NOW - timedelta(days=60)

    update_history(
        data, store, ["AAPL"], ["1d"], start=start, now=datetime(2026, 11, 3, 4, 59, tzinfo=UTC)
    )
    before = store.span("AAPL", "1d")[1]  # type: ignore[index]
    update_history(
        data, store, ["AAPL"], ["1d"], start=start, now=datetime(2026, 11, 3, 5, 0, tzinfo=UTC)
    )
    after = store.span("AAPL", "1d")[1]  # type: ignore[index]

    assert before == datetime(2026, 11, 1, 5, 0, tzinfo=UTC)
    assert after == datetime(2026, 11, 2, 5, 0, tzinfo=UTC)  # a minute later, Monday is in


def test_daily_and_intraday_bars_live_side_by_side_without_touching_each_other(
    store: MarketBarRepository, make_data: Any
) -> None:
    both = history("AAPL")
    both._frames[1440] = daily_history("AAPL")._frames[1440]
    data = make_data(both)

    update_history(data, store, ["AAPL"], ["15m", "1d"], start=NOW - timedelta(days=10), now=NOW)

    assert len(store.load(["AAPL"], "1d")["AAPL"]) == 9
    assert store.load(["AAPL"], "1h") == {}  # not asked for, not there
    assert store.span("AAPL", "15m") is not None


# ---------------------------------------------------------------------------- stale symbols


def _store_daily(store: MarketBarRepository, symbol: str, last: datetime) -> None:
    store.replace(symbol, "1d", daily_bars(last, 5), adjustment="all", fetched_at=NOW)


def test_a_symbol_whose_bars_stopped_long_before_the_others_is_flagged(
    store: MarketBarRepository,
) -> None:
    newest = datetime(2026, 11, 2, 5, 0, tzinfo=UTC)
    _store_daily(store, "AAPL", newest)
    _store_daily(store, "MSFT", newest - timedelta(days=3))  # a long weekend: not stale
    _store_daily(store, "MMC", datetime(2026, 1, 13, 5, 0, tzinfo=UTC))  # renamed in January

    flagged = stale_symbols(store, ["AAPL", "MSFT", "MMC"], "1d", tolerance=timedelta(days=5))

    assert flagged == [("MMC", datetime(2026, 1, 13, 5, 0, tzinfo=UTC))]


def test_a_symbol_with_nothing_stored_is_flagged_without_a_date(
    store: MarketBarRepository,
) -> None:
    _store_daily(store, "AAPL", datetime(2026, 11, 2, 5, 0, tzinfo=UTC))

    flagged = stale_symbols(store, ["AAPL", "ZZZZ"], "1d", tolerance=timedelta(days=5))

    assert flagged == [("ZZZZ", None)]


def test_an_empty_store_flags_every_symbol(store: MarketBarRepository) -> None:
    flagged = stale_symbols(store, ["AAPL", "MSFT"], "1d", tolerance=timedelta(days=5))

    assert flagged == [("AAPL", None), ("MSFT", None)]
