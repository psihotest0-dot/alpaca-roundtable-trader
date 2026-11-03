"""`validate-config` judges the real configuration, and the settings agree with each other
(Phase 6.4). It used to pass anywhere, because it supplied dummy keys itself."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from trading_system.cli import main
from trading_system.config import Settings
from trading_system.config_check import check_config, unknown_key_warnings

KEYS = {"ALPACA_API_KEY": "PKTESTKEY", "ALPACA_SECRET_KEY": "SECRETVALUE"}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No key from the machine's environment, no `.env` of the repository."""
    for name in list(os.environ):
        if name.upper() in {f.upper() for f in Settings.model_fields}:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def _env(tmp_path: Path, **lines: str) -> str:
    path = tmp_path / ".env"
    path.write_text("\n".join(f"{k}={v}" for k, v in lines.items()) + "\n", encoding="utf-8")
    return str(path)


# --- the check ------------------------------------------------------------------------------


def test_a_file_with_the_keys_is_valid_and_quiet(tmp_path: Path) -> None:
    report = check_config(_env(tmp_path, **KEYS, IS_PAPER="true"))

    assert report.valid and report.errors == [] and report.warnings == []


def test_missing_keys_are_an_error_and_are_named(tmp_path: Path) -> None:
    report = check_config(_env(tmp_path, IS_PAPER="true"))

    assert not report.valid
    assert any(e.startswith("ALPACA_API_KEY is not set") for e in report.errors)
    assert any(e.startswith("ALPACA_SECRET_KEY is not set") for e in report.errors)


def test_a_missing_file_with_no_keys_in_the_environment_is_invalid(tmp_path: Path) -> None:
    assert not check_config(str(tmp_path / "nothing.env")).valid


def test_keys_from_the_environment_are_enough_as_in_a_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)

    assert check_config(str(tmp_path / "nothing.env")).valid


def test_the_check_puts_nothing_into_the_environment(tmp_path: Path) -> None:
    before = dict(os.environ)

    check_config(_env(tmp_path, IS_PAPER="true"))

    assert dict(os.environ) == before  # the old command wrote dummy keys here


def test_a_setting_out_of_range_is_named(tmp_path: Path) -> None:
    report = check_config(_env(tmp_path, **KEYS, POSITION_SIZE_PCT="1.5"))

    assert not report.valid and any(e.startswith("POSITION_SIZE_PCT:") for e in report.errors)


def test_live_trading_is_refused(tmp_path: Path) -> None:
    report = check_config(_env(tmp_path, **KEYS, IS_PAPER="false"))

    assert not report.valid and any("IS_PAPER" in e for e in report.errors)


def test_settings_that_contradict_each_other_are_reported(tmp_path: Path) -> None:
    report = check_config(
        _env(tmp_path, **KEYS, POSITION_SIZE_PCT="0.3", MAX_GROSS_EXPOSURE_PCT="0.2")
    )

    assert not report.valid
    assert any("POSITION_SIZE_PCT" in e and "MAX_GROSS_EXPOSURE_PCT" in e for e in report.errors)


def test_no_value_is_ever_printed(tmp_path: Path) -> None:
    report = check_config(
        _env(
            tmp_path,
            ALPACA_API_KEY="TOPSECRET1",
            ALPACA_SECRET_KEY="TOPSECRET2",
            POSITION_SIZE_PCT="TOPSECRET3",
            LEFTOVER_KEY="TOPSECRET4",
        )
    )

    text = " ".join(report.errors + report.warnings)
    assert report.errors and report.warnings
    assert "TOPSECRET" not in text


# --- the names in the file ---------------------------------------------------------------------


def test_a_key_no_setting_reads_is_warned_about(tmp_path: Path) -> None:
    warnings = unknown_key_warnings(_env(tmp_path, **KEYS, TRADING_API_KEY="x"))

    assert warnings == [
        "TRADING_API_KEY in " + str(tmp_path / ".env") + " is not a setting and is ignored"
    ]


def test_a_misspelt_key_gets_a_suggestion(tmp_path: Path) -> None:
    [warning] = unknown_key_warnings(_env(tmp_path, ALPACA_API_KY="x"))

    assert "did you mean ALPACA_API_KEY?" in warning


def test_names_are_compared_without_regard_to_case_and_compose_keys_are_known(
    tmp_path: Path,
) -> None:
    assert unknown_key_warnings(_env(tmp_path, alpaca_api_key="x", BACKUP_KEEP="10")) == []


def test_a_warning_does_not_make_the_configuration_invalid(tmp_path: Path) -> None:
    report = check_config(_env(tmp_path, **KEYS, TRADING_API_KEY="x"))

    assert report.valid and len(report.warnings) == 1


def test_no_file_no_warnings(tmp_path: Path) -> None:
    assert unknown_key_warnings(str(tmp_path / "missing.env")) == []


# --- the command ----------------------------------------------------------------------------


def test_the_command_fails_without_keys_and_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-config"]) == 1

    out = capsys.readouterr().out
    assert "Configuration is NOT valid:" in out and "ALPACA_API_KEY is not set" in out


def test_the_command_passes_with_the_keys_in_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _env(tmp_path, **KEYS, TRADING_API_KEY="x")  # ./.env, where the command looks

    assert main(["validate-config"]) == 0

    out = capsys.readouterr().out
    assert "Configuration is valid." in out and "WARNING: TRADING_API_KEY" in out


def test_doctor_warns_about_unknown_keys_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _env(tmp_path, **KEYS, STALE_KEY="x")

    with patch("trading_system.cli.Settings"):
        assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "WARNING: STALE_KEY" in out and "Local configuration is valid." in out


# --- the settings agree with each other ---------------------------------------------------------


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s", **overrides)  # type: ignore[arg-type]


def test_the_defaults_agree() -> None:
    assert _settings().position_size_pct <= _settings().max_gross_exposure_pct


@pytest.mark.parametrize("field", ["stop_loss_pct", "take_profit_pct"])
def test_a_stop_or_a_target_of_zero_is_refused(field: str) -> None:
    with pytest.raises(ValidationError):
        _settings(**{field: 0.0})


def test_a_position_bigger_than_the_exposure_cap_is_refused() -> None:
    with pytest.raises(ValidationError, match="no position could ever be opened"):
        _settings(position_size_pct=0.3, max_gross_exposure_pct=0.2)


def test_a_trade_risk_above_the_total_risk_is_refused() -> None:
    with pytest.raises(ValidationError, match="MAX_PLANNED_RISK_PER_TRADE_PCT"):
        _settings(max_planned_risk_per_trade_pct=0.06, max_total_planned_open_risk_pct=0.05)


def test_a_chunk_overlap_that_is_not_smaller_than_the_chunk_is_refused() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        _settings(chunk_size=100, chunk_overlap=100)


@pytest.mark.parametrize("overrides", [{"timeframe_minutes": 5}, {"htf_timeframe_minutes": 30}])
def test_a_bar_size_that_nothing_reads_is_refused_instead_of_ignored(
    overrides: dict[str, int],
) -> None:
    with pytest.raises(ValidationError, match="not configurable yet"):
        _settings(**overrides)


def test_the_watchlist_is_one_list() -> None:
    assert _settings(watchlist=" aapl , MSFT, msft ").watchlist_symbols == ["AAPL", "MSFT"]
    assert _settings(watchlist="").watchlist_symbols == []
    assert len(_settings().watchlist_symbols) == 22


def test_the_file_that_is_asked_for_is_the_file_that_is_read(tmp_path: Path) -> None:
    (tmp_path / "custom.env").write_text(
        "ALPACA_API_KEY=k\nALPACA_SECRET_KEY=s\nOLD_THING=x\n", encoding="utf-8"
    )  # there is no ./.env here

    report = check_config(str(tmp_path / "custom.env"))

    assert report.valid and "OLD_THING" in report.warnings[0]


def test_a_position_exactly_as_big_as_the_exposure_cap_is_allowed() -> None:
    assert _settings(position_size_pct=0.2, max_gross_exposure_pct=0.2).position_size_pct == 0.2
    assert _settings(max_planned_risk_per_trade_pct=0.05, max_total_planned_open_risk_pct=0.05)
