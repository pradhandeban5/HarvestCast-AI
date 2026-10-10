"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with local-development defaults."""

    database_url: str = "postgresql+asyncpg://harvestcast:harvestcast@localhost:5432/harvestcast"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_origin: str = "http://localhost:5173"
    mqtt_host: str | None = None
    mqtt_port: int = 1883
    mqtt_topic: str = "harvestcast/scans"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
