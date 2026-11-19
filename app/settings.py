"""Application settings management for AIOps Copilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, HttpUrl, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file if it exists.

    Args:
        path: Path to the YAML file.

    Returns:
        A dictionary with the parsed configuration content. Empty when the file
        does not exist or contains no data.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class Settings(BaseSettings):
    """Central application configuration with environment and YAML overrides."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AIOps Copilot"
    app_env: str = Field(default="dev", alias="APP_ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")  # noqa: S104
    api_port: int = Field(default=8000, alias="API_PORT")
    dashboard_port: int = 8501

    db_url: str = Field(default="sqlite:///./data/aiops.db", alias="DB_URL")
    use_postgres: bool = Field(default=False, alias="USE_POSTGRES")
    postgres_host: str = Field(default="postgres", alias="POSTGRES_HOST")
    postgres_db: str = Field(default="aiops", alias="POSTGRES_DB")
    postgres_user: str = Field(default="aiops", alias="POSTGRES_USER")
    postgres_password: str = Field(default="aiops", alias="POSTGRES_PASSWORD")

    conformal_alpha: float = Field(default=0.1, alias="CONFORMAL_ALPHA")
    rca_alpha: float = Field(default=0.6, alias="RCA_ALPHA")
    rca_half_life_min: int = Field(default=10, alias="RCA_HALF_LIFE_MIN")

    enable_otel: bool = Field(default=False, alias="ENABLE_OTEL")
    tempo_base_url: HttpUrl | None = Field(default=None, alias="TEMPO_BASE_URL")
    prometheus_base_url: HttpUrl | None = Field(default=None, alias="PROMETHEUS_BASE_URL")

    ingest_max_points: int = 100_000
    batch_size: int = 1000
    rca_top_k: int = 5
    rca_window_minutes: int = 30
    grafana_enabled: bool = True
    enable_prometheus: bool = False

    config_app_path: Path = Path("configs/app.yaml")
    config_model_path: Path = Path("configs/model.yaml")
    config_data_path: Path = Path("configs/data.yaml")

    model_config_map: dict[str, Any] = Field(default_factory=dict)
    data_config_map: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _merge_yaml_configs(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Merge YAML configuration defaults before environment overrides.

        Args:
            values: Raw settings values prior to Pydantic validation.

        Returns:
            The values dict enriched with YAML defaults.
        """
        config_app_path = Path(values.get("config_app_path", "configs/app.yaml"))
        app_cfg = _load_yaml(config_app_path)
        app_section = app_cfg.get("app", {})
        if app_section:
            values.setdefault("app_name", app_section.get("name", "AIOps Copilot"))
            values.setdefault("app_env", app_section.get("env", "dev"))
            values.setdefault("api_host", app_section.get("api_host", "0.0.0.0"))  # noqa: S104
            values.setdefault("api_port", app_section.get("api_port", 8000))
            values.setdefault("dashboard_port", app_section.get("dashboard_port", 8501))
            values.setdefault("batch_size", app_section.get("batch_size", 1000))
            values.setdefault("ingest_max_points", app_section.get("ingest_max_points", 100_000))
            values.setdefault("use_postgres", app_section.get("use_postgres", False))
            values.setdefault("enable_otel", app_section.get("enable_otel", False))
            values.setdefault("enable_prometheus", app_section.get("enable_prometheus", False))
            values.setdefault("rca_top_k", app_section.get("rca_top_k", 5))
            values.setdefault("rca_window_minutes", app_section.get("rca_window_minutes", 30))
            values.setdefault("grafana_enabled", app_section.get("grafana_enabled", True))
        features = app_cfg.get("features", {})
        if features:
            values.setdefault("refresh_traces_interval_sec", features.get("refresh_traces_interval_sec", 60))

        values.setdefault("model_config_map", _load_yaml(Path(values.get("config_model_path", "configs/model.yaml"))))
        values.setdefault("data_config_map", _load_yaml(Path(values.get("config_data_path", "configs/data.yaml"))))
        return values

    refresh_traces_interval_sec: int = 60

    @field_validator("tempo_base_url", "prometheus_base_url")
    @classmethod
    def _validate_optional_url(cls, value: HttpUrl | None) -> HttpUrl | None:
        """Allow empty URLs in configuration.

        Args:
            value: Optional URL to validate.

        Returns:
            The original value, possibly None.
        """
        return value

    @property
    def effective_db_url(self) -> str:
        """Return the database URL taking the Postgres toggle into account.

        Returns:
            A SQLAlchemy-compatible database URL string.
        """
        if not self.use_postgres:
            return self.db_url
        credentials = f"{self.postgres_user}:{self.postgres_password}"
        return f"postgresql+psycopg2://{credentials}@{self.postgres_host}/{self.postgres_db}"


def load_settings() -> Settings:
    """Instantiate settings with validation and helpful error reporting.

    Returns:
        Validated settings instance.

    Raises:
        RuntimeError: Raised when configuration validation fails.
    """
    try:
        return Settings()
    except ValidationError as exc:
        raise RuntimeError(f"Invalid configuration: {exc}") from exc
