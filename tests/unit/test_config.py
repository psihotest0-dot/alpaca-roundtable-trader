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


def test_oracle_constraints():
    # Valid missing oracle
    Settings(alpaca_api_key="k", alpaca_secret_key="s", oracle_enabled=False)

    # Missing workspace slug when enabled
    with pytest.raises(ValidationError, match="ANYTHINGLLM_WORKSPACE_SLUG is required"):
        Settings(
            alpaca_api_key="k",
            alpaca_secret_key="s",
            oracle_enabled=True,
            anythingllm_api_key="llm_key",
            anythingllm_workspace_slug=None,
        )

    # Missing api key when enabled
    with pytest.raises(ValidationError, match="ANYTHINGLLM_API_KEY is required"):
        Settings(
            alpaca_api_key="k",
            alpaca_secret_key="s",
            oracle_enabled=True,
            anythingllm_workspace_slug="slug",
            anythingllm_api_key=None,
        )

    # Remote URL must be https
    with pytest.raises(ValidationError, match="Remote Oracle requires HTTPS"):
        Settings(
            alpaca_api_key="k",
            alpaca_secret_key="s",
            oracle_enabled=True,
            anythingllm_api_key="llm_key",
            anythingllm_workspace_slug="slug",
            anythingllm_base_url="http://remote-server:3001",
        )

    # Local URL can be http
    s = Settings(
        alpaca_api_key="k",
        alpaca_secret_key="s",
        oracle_enabled=True,
        anythingllm_api_key="llm_key",
        anythingllm_workspace_slug="slug",
        anythingllm_base_url="http://127.0.0.1:3001",
    )
    assert s.anythingllm_base_url == "http://127.0.0.1:3001"


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
