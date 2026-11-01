"""normalize_bars: every shape the system meets ends up as the one canonical shape."""

import pandas as pd
import pytest

from tests.fixtures.bars import alpaca_shaped, make_bars
from trading_system.data.bars import BarsContractError, normalize_bars


def _assert_canonical(df: pd.DataFrame) -> None:
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert df.index.name is None
    assert list(df["timestamp"]) == list(df.index)


def test_alpacas_multiindex_loses_its_symbol_level() -> None:
    bars, _ = make_bars()

    out = normalize_bars(alpaca_shaped("AAPL", bars))

    _assert_canonical(out)
    assert len(out) == len(bars)


def test_a_timestamp_column_with_a_default_index_becomes_the_index() -> None:
    bars, _ = make_bars()
    flat = bars.reset_index().rename(columns={"index": "timestamp"})
    assert "timestamp" in flat.columns and isinstance(flat.index, pd.RangeIndex)

    _assert_canonical(normalize_bars(flat))


def test_a_plain_datetime_index_gets_the_column() -> None:
    bars, _ = make_bars()
    assert "timestamp" not in bars.columns

    _assert_canonical(normalize_bars(bars))


def test_normalizing_twice_changes_nothing() -> None:
    bars, _ = make_bars()
    once = normalize_bars(alpaca_shaped("AAPL", bars))

    # check_freq=False: pandas may infer a frequency on the second pass; the data is the same.
    pd.testing.assert_frame_equal(normalize_bars(once), once, check_freq=False)


def test_naive_times_are_taken_as_utc() -> None:
    naive = pd.DataFrame(
        {"close": [1.0, 2.0]}, index=pd.to_datetime(["2023-01-02 10:00", "2023-01-02 10:15"])
    )

    out = normalize_bars(naive)

    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2023-01-02 10:00", tz="UTC")


def test_duplicates_keep_the_last_row_and_order_is_ascending() -> None:
    t1, t2 = pd.Timestamp("2023-01-02 10:00", tz="UTC"), pd.Timestamp("2023-01-02 10:15", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=[t2, t1, t2])

    out = normalize_bars(df)

    assert list(out["close"]) == [2.0, 3.0]  # t1, then the last t2
    _assert_canonical(out)


def test_the_input_is_not_modified() -> None:
    bars, _ = make_bars()
    shaped = alpaca_shaped("AAPL", bars)
    before = shaped.copy()

    normalize_bars(shaped)

    pd.testing.assert_frame_equal(shaped, before)


def test_an_empty_frame_comes_back_unchanged() -> None:
    empty = pd.DataFrame()

    assert normalize_bars(empty) is empty


def test_bars_without_any_time_axis_are_rejected() -> None:
    with pytest.raises(BarsContractError, match="DatetimeIndex"):
        normalize_bars(pd.DataFrame({"close": [1.0, 2.0]}))
