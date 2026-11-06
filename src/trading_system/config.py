import math
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    alpaca_api_key: SecretStr
    alpaca_secret_key: SecretStr
    data_feed: str = "iex"
    is_paper: bool = True

    oracle_enabled: bool = False
    # null: every signal passes. backend: the second opinion comes from the Go backend's
    # POST /oracle/evaluate (oracle/backend.py); the bot never calls the model itself.
    oracle_provider: Literal["null", "backend"] = "null"
    # shadow: the model is asked and its answers are recorded, but the loop is told APPROVE
    # whatever it said, so nothing about the orders changes. active: its verdict counts.
    oracle_mode: Literal["shadow", "active"] = "shadow"
    # From a container the PC is host.docker.internal (Docker Desktop forwards it to the
    # backend, which listens on the PC's loopback only). Never a place for a key in the URL.
    oracle_backend_url: str = "http://host.docker.internal:8080"
    # The backend's ORACLE_TOKEN (at least 32 characters, or the backend does not serve the
    # route). Only in `.env` and only in the trader service of compose.yaml; never in git.
    oracle_backend_token: SecretStr = SecretStr("")
    oracle_timeout_seconds: int = Field(10, gt=0, le=120)

    log_level: str = "INFO"
    database_url: str = "sqlite:///./var/trading.db"
    # The strategy research's own candle file (daily bars of the research universe, see
    # data/universe.py). A cache that `fetch-daily` rebuilds, kept out of trading.db so a big
    # write cannot hold up the trading loop and the backups of trading.db stay small.
    research_database_url: str = "sqlite:///./var/research.db"
    # The news kept for the Oracle and for judging it later (data/news.py): a cache that
    # `fetch-news` fills from Alpaca every few minutes, not a record, so also its own file.
    news_database_url: str = "sqlite:///./var/news.db"
    # The 15-minute and 1-hour bars of the whole observation universe (the research universe and the
    # watchlist, ~155 symbols), kept for the Oracle observer and for the hours research
    # (runtime/observe.py, research/intraday.py): a cache `fetch-observe` rebuilds, in a file of its own
    # so that ~1 million rows never weigh on trading.db or its backups.
    observe_database_url: str = "sqlite:///./var/observe.db"
    # Where the trading loop writes its pulse (see runtime/heartbeat.py); `health` reads it.
    heartbeat_path: str = "./var/heartbeat.json"
    metrics_enabled: bool = False
    metrics_host: str = "127.0.0.1"
    metrics_port: int = 9100

    timeframe_minutes: int = Field(15, gt=0)
    htf_timeframe_minutes: int = Field(60, gt=0)
    lookback_days: int = Field(30, gt=0)
    htf_lookback_days: int = Field(90, gt=0)

    position_size_pct: float = Field(0.04, ge=0.0, le=1.0)
    max_open_positions: int = Field(5, gt=0)
    max_gross_exposure_pct: float = Field(0.20, ge=0.0, le=1.0)
    max_planned_risk_per_trade_pct: float = Field(0.002, ge=0.0, le=1.0)
    max_total_planned_open_risk_pct: float = Field(0.05, ge=0.0, le=1.0)
    max_daily_loss_pct: float = Field(0.02, ge=0.0, le=1.0)

    # A stop of 0 is no stop, and a target of 0 is no target: both have to be above zero.
    stop_loss_pct: float = Field(0.04, gt=0.0, le=1.0)
    take_profit_pct: float = Field(0.05, gt=0.0, le=1.0)
    # How far above the last price the entry limit sits (see risk/entry_plan.py).
    entry_limit_buffer_pct: float = Field(0.001, ge=0.0, le=0.05)
    # An entry is planned around the last trade (data/live_price.py): no entry if that trade is
    # older than this, or further from the last closed bar than this. Measured on the IEX feed
    # (2026-11-02, mid-session): most watchlist symbols trade every few seconds, the quietest
    # ones (MARA, OPEN) go 1 to 2 minutes between prints.
    live_price_max_age_seconds: int = Field(120, gt=0)
    live_price_max_deviation_pct: float = Field(0.05, ge=0.0, le=1.0)
    allow_fractional: bool = False
    # A position without an active stop halts trading and raises an alarm. Closing it at market
    # is a separate, explicit choice (see ExecutionEngine.emergency_check_unprotected_positions).
    emergency_close_unprotected: bool = False
    # What stopping the system does (SIGTERM, Ctrl+C, `docker stop`). False: nothing, so a restart
    # carries on by itself. True: new entries are paused, and the next start waits for
    # `resume-entries`. `run --pause-on-exit` asks for True for one run.
    pause_on_shutdown: bool = False

    watchlist: str = (
        "AAPL,MSFT,NVDA,AMD,GOOGL,TSLA,INTC,META,AMZN,SOFI,PLTR,"
        "COIN,MSTR,SHOP,ORCL,MU,NKE,AAL,RIG,MARA,OPEN,PATH"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("oracle_backend_url")
    @classmethod
    def check_oracle_backend_url(cls, v: str) -> str:
        """An http(s) address with a host and nothing a secret could hide in. The trailing
        slash is dropped so that the path of a route can be appended."""
        v = v.strip()
        parts = urlsplit(v)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("ORACLE_BACKEND_URL must be an http:// or https:// address")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError(
                "ORACLE_BACKEND_URL must be a plain address: the token goes in "
                "ORACLE_BACKEND_TOKEN, not in the URL"
            )
        return v.rstrip("/")

    @field_validator("is_paper")
    @classmethod
    def check_is_paper(cls, v: bool) -> bool:
        if v is False:
            raise ValueError("Live trading is not supported. IS_PAPER must be true.")
        return v

    @model_validator(mode="after")
    def check_the_settings_agree(self) -> "Settings":
        """Limits that contradict each other, found at start and not on the first signal."""
        if self.position_size_pct > self.max_gross_exposure_pct:
            raise ValueError(
                f"POSITION_SIZE_PCT ({self.position_size_pct}) is larger than "
                f"MAX_GROSS_EXPOSURE_PCT ({self.max_gross_exposure_pct}): no position could ever "
                "be opened"
            )
        if self.max_planned_risk_per_trade_pct > self.max_total_planned_open_risk_pct:
            raise ValueError(
                f"MAX_PLANNED_RISK_PER_TRADE_PCT ({self.max_planned_risk_per_trade_pct}) is larger "
                f"than MAX_TOTAL_PLANNED_OPEN_RISK_PCT ({self.max_total_planned_open_risk_pct})"
            )
        # Not read yet: the 15-minute and 1-hour bars are fixed in the data layer. Another value
        # would be accepted and silently ignored, which is worse than refusing it.
        if self.timeframe_minutes != 15 or self.htf_timeframe_minutes != 60:
            raise ValueError(
                "TIMEFRAME_MINUTES must be 15 and HTF_TIMEFRAME_MINUTES 60: the bar sizes are "
                "not configurable yet"
            )
        return self

    @property
    def watchlist_symbols(self) -> list[str]:
        """The watchlist as a list (it is validated, upper-cased and de-duplicated already)."""
        return [s for s in self.watchlist.split(",") if s]

    @field_validator(
        "position_size_pct",
        "max_gross_exposure_pct",
        "max_planned_risk_per_trade_pct",
        "max_total_planned_open_risk_pct",
        "max_daily_loss_pct",
        "stop_loss_pct",
        "take_profit_pct",
        "entry_limit_buffer_pct",
        "live_price_max_deviation_pct",
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
