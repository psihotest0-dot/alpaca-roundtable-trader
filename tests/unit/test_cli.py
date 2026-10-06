import pytest
from trading_system.cli import create_parser

def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    parser = create_parser()
    with pytest.raises(SystemExit) as e:
        parser.parse_args(["--help"])

    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "Alpaca Roundtable Trader" in captured.out
    assert "show this help message and exit" in captured.out
