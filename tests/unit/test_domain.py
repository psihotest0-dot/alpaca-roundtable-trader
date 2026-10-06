from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from trading_system.domain.models import Bar, StrategySignal, SignalDirection


def test_bar_utc_validation():
    # Valid UTC timezone
    dt = datetime.now(timezone.utc)
    bar = Bar(
        symbol="AAPL",
        timestamp=dt,
        open=Decimal("150.0"),
        high=Decimal("155.0"),
        low=Decimal("149.0"),
        close=Decimal("152.0"),
        volume=Decimal("1000000"),
    )
    assert bar.timestamp == dt
    assert bar.timestamp.tzinfo == timezone.utc

    # Invalid: no timezone
    dt_naive = datetime.now()
    with pytest.raises(ValidationError):
        Bar(
            symbol="AAPL",
            timestamp=dt_naive,
            open=Decimal("150.0"),
            high=Decimal("155.0"),
            low=Decimal("149.0"),
            close=Decimal("152.0"),
            volume=Decimal("1000000"),
        )

def test_strategy_signal_deterministic_id():
    dt = datetime(2023, 10, 25, 14, 30, tzinfo=timezone.utc)
    signal = StrategySignal(
        strategy_version="v1.0",
        symbol="AAPL",
        timeframe="15Min",
        closed_bar_timestamp=dt,
        direction=SignalDirection.LONG,
    )

    ts = int(dt.timestamp())
    expected_id = f"v1.0_AAPL_15Min_{ts}_LONG"
    assert signal.deterministic_id == expected_id

    # Test UTC validation
    dt_naive = datetime(2023, 10, 25, 14, 30)
    with pytest.raises(ValidationError):
        StrategySignal(
            strategy_version="v1.0",
            symbol="AAPL",
            timeframe="15Min",
            closed_bar_timestamp=dt_naive,
            direction=SignalDirection.LONG,
        )
