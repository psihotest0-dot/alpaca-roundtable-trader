from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Only paper trading is supported for MVP
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""

    # Optional AnythingLLM API config
    anythingllm_api_url: str = ""
    anythingllm_api_key: str = ""

    # SQLite local DB path
    database_url: str = "sqlite:///local.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
