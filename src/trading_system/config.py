import math
from typing import Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    alpaca_api_key: SecretStr
    alpaca_secret_key: SecretStr
    data_feed: str = "iex"
    is_paper: bool = True

    oracle_enabled: bool = False
    anythingllm_api_key: Optional[SecretStr] = None
    anythingllm_base_url: str = "http://127.0.0.1:3001"
    anythingllm_workspace_slug: Optional[str] = None
    oracle_timeout_seconds: int = Field(10, gt=0)

    log_level: str = "INFO"
    database_url: str = "sqlite:///./var/trading.db"
    metrics_enabled: bool = False
    metrics_host: str = "127.0.0.1"
    metrics_port: int = 9100

    # RSS Pipeline Settings
    rss_enabled: bool = False
    rss_feeds: str = ""
    rss_update_interval_minutes: int = Field(60, gt=0)
    rss_max_concurrent_requests: int = Field(5, gt=0)
    rss_max_response_bytes: int = Field(5 * 1024 * 1024, gt=0)  # 5MB
    rss_max_entries_per_feed: int = Field(50, gt=0)
    chroma_db_path: str = "./var/chroma_db"
    chroma_collection_name: str = "news"
    embedding_provider: str = "chroma_default"
    chunk_size: int = Field(500, gt=0)
    chunk_overlap: int = Field(50, ge=0)
    retrieval_top_k: int = Field(5, gt=0)

    timeframe_minutes: int = Field(15, gt=0)
    htf_timeframe_minutes: int = Field(60, gt=0)
    lookback_days: int = Field(30, gt=0)
    htf_lookback_days: int = Field(90, gt=0)

    position_size_pct: float = Field(0.04, ge=0.0, le=1.0)
    max_open_positions: int = Field(5, gt=0)
    max_gross_exposure_pct: float = Field(0.20, ge=0.0, le=1.0)
    max_planned_risk_per_trade_pct: float = Field(0.002, ge=0.0, le=1.0)
    max_total_planned_open_risk_pct: float = Field(0.01, ge=0.0, le=1.0)
    max_daily_loss_pct: float = Field(0.02, ge=0.0, le=1.0)

    stop_loss_pct: float = Field(0.04, ge=0.0, le=1.0)
    take_profit_pct: float = Field(0.05, ge=0.0, le=1.0)
    allow_fractional: bool = False

    watchlist: str = (
        "AAPL,MSFT,NVDA,AMD,GOOGL,TSLA,INTC,META,AMZN,SOFI,PLTR,"
        "COIN,MSTR,SHOP,ORCL,MU,NKE,AAL,RIG,MARA,OPEN,PATH"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("is_paper")
    @classmethod
    def check_is_paper(cls, v: bool) -> bool:
        if v is False:
            raise ValueError("Live trading is not supported. IS_PAPER must be true.")
        return v

    @field_validator(
        "position_size_pct",
        "max_gross_exposure_pct",
        "max_planned_risk_per_trade_pct",
        "max_total_planned_open_risk_pct",
        "max_daily_loss_pct",
        "stop_loss_pct",
        "take_profit_pct",
    )
    @classmethod
    def check_finite_pct(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Percentage must be a finite number.")
        return v

    @field_validator("watchlist")
    @classmethod
    def check_watchlist(cls, v: str) -> str:
        if not v:
            return ""
        symbols = [s.strip().upper() for s in v.split(",") if s.strip()]
        valid_symbols = []
        seen = set()
        for sym in symbols:
            if not sym.isalnum():
                raise ValueError(f"Invalid symbol: {sym}")
            if sym not in seen:
                seen.add(sym)
                valid_symbols.append(sym)
        return ",".join(valid_symbols)

    @model_validator(mode="after")
    def check_oracle_settings(self) -> "Settings":
        if self.oracle_enabled:
            if not self.anythingllm_api_key or not self.anythingllm_api_key.get_secret_value():
                raise ValueError("ANYTHINGLLM_API_KEY is required when Oracle is enabled.")
            if not self.anythingllm_workspace_slug:
                raise ValueError("ANYTHINGLLM_WORKSPACE_SLUG is required when Oracle is enabled.")

        url = self.anythingllm_base_url.lower()
        if self.oracle_enabled and not (
            url.startswith("http://127.0.0.1") or url.startswith("http://localhost")
        ):
            if not url.startswith("https://"):
                raise ValueError("Remote Oracle requires HTTPS (ANYTHINGLLM_BASE_URL).")

        return self
