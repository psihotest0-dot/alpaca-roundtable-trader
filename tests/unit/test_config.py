import pytest
from pydantic import ValidationError
from trading_system.config import Settings


def test_missing_required_keys(monkeypatch):
    # Provide _env_file=None to disable reading from .env file
    # and clear os.environ completely just in case
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ValidationError) as exc:
        Settings(alpaca_api_key="key_only", _env_file=None)
    assert "alpaca_secret_key" in str(exc.value)


def test_strict_boolean_validation():
    with pytest.raises(ValidationError):
        Settings(
            alpaca_api_key="k",
            alpaca_secret_key="s",
            oracle_enabled="invalid_bool",
        )


def test_live_mode_rejection():
    with pytest.raises(
        ValidationError, match="Live trading is not supported. IS_PAPER must be true."
    ):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", is_paper=False)


def test_invalid_percentages():
    # Test bounds
    with pytest.raises(ValidationError):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", position_size_pct=1.5)

    with pytest.raises(ValidationError):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", stop_loss_pct=-0.1)

    # Test nan/inf
    # Note: pydantic's built-in bounds check might catch 'inf'/'nan' before our validator
    # We'll use pytest.raises(ValidationError) generally.
    with pytest.raises(ValidationError):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", max_daily_loss_pct=float("inf"))

    with pytest.raises(ValidationError):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", take_profit_pct=float("nan"))

    # The entry limit buffer is a small premium over the last price, never a discount
    with pytest.raises(ValidationError):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", entry_limit_buffer_pct=-0.001)

    with pytest.raises(ValidationError):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", entry_limit_buffer_pct=0.5)

    with pytest.raises(ValidationError):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", entry_limit_buffer_pct=float("nan"))

    # The live price limits: a positive age and a finite fraction
    with pytest.raises(ValidationError):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", live_price_max_age_seconds=0)

    with pytest.raises(ValidationError):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", live_price_max_deviation_pct=-0.01)

    with pytest.raises(ValidationError):
        Settings(
            alpaca_api_key="k", alpaca_secret_key="s", live_price_max_deviation_pct=float("nan")
        )


def test_secrets_leakage():
    s = Settings(alpaca_api_key="secret123", alpaca_secret_key="secret456")
    repr_str = repr(s)
    str_str = str(s)

    assert "secret123" not in repr_str
    assert "secret456" not in repr_str
    assert "secret123" not in str_str
    assert "secret456" not in str_str

    assert "**********" in repr_str


def test_the_oracle_is_off_by_default_and_only_known_providers_are_accepted():
    default = Settings(alpaca_api_key="k", alpaca_secret_key="s", _env_file=None)
    assert default.oracle_enabled is False
    assert default.oracle_provider == "null"

    with pytest.raises(ValidationError):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", oracle_provider="anythingllm")


def _oracle_settings(**overrides):
    return Settings(alpaca_api_key="k", alpaca_secret_key="s", _env_file=None, **overrides)


def test_the_backend_oracle_defaults_are_safe():
    default = _oracle_settings()
    assert default.oracle_mode == "shadow"  # asked and recorded, the loop is told yes
    assert default.oracle_backend_url == "http://host.docker.internal:8080"
    assert default.oracle_backend_token.get_secret_value() == ""
    assert default.oracle_timeout_seconds == 10
    assert _oracle_settings(oracle_provider="backend").oracle_provider == "backend"


@pytest.mark.parametrize("mode", ["", "on", "paper", "ACTIVE "])
def test_an_unknown_oracle_mode_is_refused(mode):
    with pytest.raises(ValidationError):
        _oracle_settings(oracle_mode=mode)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "host.docker.internal:8080",  # no scheme
        "ftp://example.com",
        "http://",
        "http://user:secret@host.docker.internal:8080",  # a key in the address
        "http://host.docker.internal:8080/?token=abc",
        "http://host.docker.internal:8080/#frag",
    ],
)
def test_an_oracle_backend_url_must_be_a_plain_http_address(url):
    with pytest.raises(ValidationError, match="ORACLE_BACKEND_URL"):
        _oracle_settings(oracle_backend_url=url)


def test_the_backend_url_loses_its_trailing_slash_and_spaces():
    assert _oracle_settings(oracle_backend_url=" http://127.0.0.1:8080/ ").oracle_backend_url == (
        "http://127.0.0.1:8080"
    )
    assert _oracle_settings(oracle_backend_url="https://b.example/api/").oracle_backend_url == (
        "https://b.example/api"
    )


@pytest.mark.parametrize("seconds", [0, -1, 121])
def test_the_oracle_timeout_has_bounds(seconds):
    with pytest.raises(ValidationError):
        _oracle_settings(oracle_timeout_seconds=seconds)


def test_the_oracle_token_never_shows_in_a_repr():
    token = "T" * 40
    s = _oracle_settings(oracle_backend_token=token)
    assert token not in repr(s) and token not in str(s)
    assert s.oracle_backend_token.get_secret_value() == token


def test_watchlist_normalization_and_dedup():
    s = Settings(alpaca_api_key="k", alpaca_secret_key="s", watchlist=" aApl , MSFT, msft , nvda  ")
    assert s.watchlist == "AAPL,MSFT,NVDA"

    # Invalid symbols
    with pytest.raises(ValidationError, match="Invalid symbol"):
        Settings(alpaca_api_key="k", alpaca_secret_key="s", watchlist="AAPL, MSFT1*, NVDA")


def test_stopping_does_not_pause_entries_unless_asked_to():
    default = Settings(alpaca_api_key="k", alpaca_secret_key="s", _env_file=None)
    asked = Settings(
        alpaca_api_key="k", alpaca_secret_key="s", pause_on_shutdown=True, _env_file=None
    )
    assert default.pause_on_shutdown is False
    assert asked.pause_on_shutdown is True
