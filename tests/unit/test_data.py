from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from trading_system.config import Settings
from trading_system.data.alpaca_data import AlpacaData


@pytest.fixture
def settings():
    return Settings(
        alpaca_api_key="test_key",
        alpaca_secret_key="test_secret",
        data_feed="iex",
        is_paper=True
    )


@patch("trading_system.data.alpaca_data.StockHistoricalDataClient")
def test_data_init(mock_client, settings):
    data = AlpacaData(settings)
    mock_client.assert_called_once_with(
        api_key="test_key",
        secret_key="test_secret"
    )
    assert data.feed == "iex"


@patch("trading_system.data.alpaca_data.StockHistoricalDataClient")
def test_clean_and_deduplicate(mock_client, settings):
    data = AlpacaData(settings)

    # Create DF with duplicates and unordered
    t1 = pd.Timestamp("2023-01-01 10:00:00")
    t2 = pd.Timestamp("2023-01-01 10:15:00")

    df = pd.DataFrame(
        {"close": [100, 101, 102]},
        index=[t2, t1, t2]
    )

    cleaned = data.clean_and_deduplicate(df)

    # Assert UTC timezone
    assert str(cleaned.index.tz) == "UTC"

    # Assert ordered and deduplicated (keeps last, so t2 close is 102)
    assert len(cleaned) == 2
    assert cleaned.iloc[0]["close"] == 101  # t1
    assert cleaned.iloc[1]["close"] == 102  # t2


@patch("trading_system.data.alpaca_data.StockHistoricalDataClient")
def test_align_and_filter(mock_client, settings):
    data = AlpacaData(settings)

    # 15m bars
    # 09:30 (closes 09:45)
    # 09:45 (closes 10:00)
    # 10:00 (closes 10:15)
    # 10:15 (closes 10:30)
    # 10:30 (closes 10:45) - not closed yet if current_time = 10:30

    idx_15 = [
        pd.Timestamp("2023-01-01 09:30:00", tz="UTC"),
        pd.Timestamp("2023-01-01 09:45:00", tz="UTC"),
        pd.Timestamp("2023-01-01 10:00:00", tz="UTC"),
        pd.Timestamp("2023-01-01 10:15:00", tz="UTC"),
        pd.Timestamp("2023-01-01 10:30:00", tz="UTC"),
    ]
    df_15m = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=idx_15)

    # 1h bars
    # 09:30 (closes 10:30)

    idx_1h = [
        pd.Timestamp("2023-01-01 09:30:00", tz="UTC"),
    ]
    df_1h = pd.DataFrame({"close": [10]}, index=idx_1h)

    current_time = datetime(2023, 1, 1, 10, 35, tzinfo=timezone.utc)

    aligned_15m, filtered_1h = data.align_and_filter(df_15m, df_1h, current_time)

    # 10:30 15m bar is NOT closed since 10:30 + 15m = 10:45 > 10:35
    # 09:30 1h bar IS closed since 09:30 + 1h = 10:30 <= 10:35

    # Alignment:
    # 15m at 09:30 (closes 09:45): 1h at 09:30 closes at 10:30 > 09:45. Valid 1h = empty
    # 15m at 09:45 (closes 10:00): Valid 1h = empty
    # 15m at 10:00 (closes 10:15): Valid 1h = empty
    # 15m at 10:15 (closes 10:30): 1h at 09:30 closes at 10:30 <= 10:30. Valid 1h = 09:30 bar.

    assert len(aligned_15m) == 1
    assert aligned_15m.index[0] == pd.Timestamp("2023-01-01 10:15:00", tz="UTC")

    assert len(filtered_1h) == 1
    assert filtered_1h.index[0] == pd.Timestamp("2023-01-01 09:30:00", tz="UTC")
