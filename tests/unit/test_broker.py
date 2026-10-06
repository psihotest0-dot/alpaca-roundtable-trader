from unittest.mock import MagicMock, patch

import pytest
from alpaca.trading.enums import QueryOrderStatus

from trading_system.broker.alpaca_broker import AlpacaBroker
from trading_system.config import Settings


@pytest.fixture
def settings():
    return Settings(
        alpaca_api_key="test_key",
        alpaca_secret_key="test_secret",
        is_paper=True
    )


@patch("trading_system.broker.alpaca_broker.TradingClient")
def test_broker_init(mock_client, settings):
    broker = AlpacaBroker(settings)
    mock_client.assert_called_once_with(
        api_key="test_key",
        secret_key="test_secret",
        paper=True
    )


@patch("trading_system.broker.alpaca_broker.TradingClient")
def test_broker_live_trading_prevented(mock_client):
    settings = Settings(
        alpaca_api_key="test_key",
        alpaca_secret_key="test_secret",
        is_paper=True
    )
    settings.is_paper = False # Pydantic settings block this on init, but simulating just in case
    with pytest.raises(ValueError, match="Live trading is not supported"):
        AlpacaBroker(settings)


@patch("trading_system.broker.alpaca_broker.TradingClient")
def test_get_account(mock_client, settings):
    mock_instance = mock_client.return_value
    mock_instance.get_account.return_value = {"id": "123"}
    broker = AlpacaBroker(settings)

    assert broker.get_account() == {"id": "123"}
    mock_instance.get_account.assert_called_once()


@patch("trading_system.broker.alpaca_broker.TradingClient")
def test_get_open_orders(mock_client, settings):
    mock_instance = mock_client.return_value
    mock_instance.get_orders.return_value = [{"id": "order1"}]
    broker = AlpacaBroker(settings)

    orders = broker.get_open_orders()
    assert orders == [{"id": "order1"}]

    # Check that it asks for open and nested
    call_args = mock_instance.get_orders.call_args
    req = call_args.kwargs["filter"]
    assert req.status == QueryOrderStatus.OPEN
    assert req.nested is True


@patch("trading_system.broker.alpaca_broker.TradingClient")
def test_get_clock(mock_client, settings):
    mock_instance = mock_client.return_value
    mock_instance.get_clock.return_value = {"is_open": True}
    broker = AlpacaBroker(settings)

    assert broker.get_clock() == {"is_open": True}
