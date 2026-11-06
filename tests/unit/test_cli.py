from unittest.mock import patch


from trading_system.cli import main


@patch("trading_system.cli.Settings")
def test_doctor_local(mock_settings, capsys):
    result = main(["doctor"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Local configuration is valid." in captured.out


@patch("trading_system.cli.Settings")
@patch("trading_system.broker.alpaca_broker.AlpacaBroker")
def test_doctor_network(mock_broker, mock_settings, capsys):
    mock_instance = mock_broker.return_value
    mock_instance.get_clock.return_value.is_open = True

    result = main(["doctor", "--network"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Running network checks..." in captured.out
    assert "Broker connection successful" in captured.out


@patch("trading_system.cli.Settings")
@patch("trading_system.broker.alpaca_broker.AlpacaBroker")
def test_doctor_network_error(mock_broker, mock_settings, capsys):
    mock_instance = mock_broker.return_value
    mock_instance.get_clock.side_effect = Exception("API error")

    result = main(["doctor", "--network"])
    assert result == 1
    captured = capsys.readouterr()
    assert "Network check failed: API error" in captured.out


@patch("trading_system.cli.Settings")
@patch("trading_system.broker.alpaca_broker.AlpacaBroker")
@patch("trading_system.data.corporate_actions.CorporateActionsFeed")
@patch(
    "trading_system.data.corporate_actions.check_holdings",
    return_value=[
        "WARNING, corporate actions due within 7 days on what we hold:",
        "  2026-11-09  AAPL",
    ],
)
def test_doctor_network_prints_a_corporate_action_on_a_holding_and_still_passes(
    mock_check, mock_feed, mock_broker, mock_settings, capsys
):
    mock_broker.return_value.get_clock.return_value.is_open = True

    assert main(["doctor", "--network"]) == 0

    out = capsys.readouterr().out
    assert "WARNING, corporate actions due within 7 days" in out
    assert "2026-11-09  AAPL" in out


@patch("trading_system.cli.Settings")
@patch("trading_system.broker.alpaca_broker.AlpacaBroker")
@patch("trading_system.data.corporate_actions.CorporateActionsFeed")
@patch("trading_system.data.corporate_actions.check_holdings", side_effect=RuntimeError("down"))
def test_doctor_network_does_not_fail_when_only_the_corporate_action_look_fails(
    mock_check, mock_feed, mock_broker, mock_settings, capsys
):
    """The look is advisory: Alpaca's market-data API being slow is not a broken system."""
    mock_broker.return_value.get_clock.return_value.is_open = True

    assert main(["doctor", "--network"]) == 0

    assert "Corporate actions could not be checked (advisory only): down" in capsys.readouterr().out


@patch("trading_system.cli.Settings")
def test_corporate_actions_lists_the_events_of_the_symbols_asked(mock_settings, capsys):
    from datetime import date

    from trading_system.data.corporate_actions import CorporateEvent

    events = [
        CorporateEvent(
            "forward_splits", ("SNDK",), date(2026, 11, 20), "split: 8 new share(s) for 1 old"
        )
    ]
    with patch(
        "trading_system.data.corporate_actions.CorporateActionsFeed.events", return_value=events
    ) as mock_events:
        assert main(["corporate-actions", "--symbols", "sndk, aapl"]) == 0

    assert mock_events.call_args.args[0] == ["AAPL", "SNDK"]
    out = capsys.readouterr().out
    assert "2026-11-20" in out and "SNDK" in out
    assert "not a promise" in out  # nothing found must never read as "nothing will happen"


@patch("trading_system.cli.Settings")
def test_corporate_actions_rejects_a_negative_window(mock_settings, capsys):
    assert main(["corporate-actions", "--symbols", "AAPL", "--ahead", "-1"]) == 1
    assert "cannot be negative" in capsys.readouterr().out


@patch("trading_system.cli.Settings")
@patch("trading_system.broker.alpaca_broker.AlpacaBroker")
def test_equity_check_with_nothing_in_alpacas_history_fails(mock_broker, mock_settings, capsys):
    from alpaca.trading.models import PortfolioHistory

    mock_broker.return_value.get_portfolio_history.return_value = PortfolioHistory(
        timestamp=[], equity=[], profit_loss=[], profit_loss_pct=[], timeframe="1D"
    )

    assert main(["equity-check"]) == 1
    assert "no equity history" in capsys.readouterr().out


def test_run_passes_the_pause_on_exit_option_on_and_leaves_the_setting_otherwise():
    from unittest.mock import MagicMock

    for argv, expected in ((["run"], None), (["run", "--pause-on-exit"], True)):
        orchestrator = MagicMock()
        with (
            patch("trading_system.cli.Settings"),
            patch("trading_system.runtime.orchestrator.Orchestrator", return_value=orchestrator),
        ):
            assert main(argv) == 0
        orchestrator.run.assert_called_once_with(False, pause_on_shutdown=expected)
