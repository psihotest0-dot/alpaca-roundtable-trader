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
