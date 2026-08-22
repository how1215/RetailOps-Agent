from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vllm_model: str = "google/gemma-4-31B-it-qat-w4a16-ct"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "dummy"
    database_url: str = "sqlite:///data/retailops.db"
    policy_dir: Path = Path("policies")
    trace_dir: Path = Path("traces")
    max_llm_calls: int = Field(default=8, ge=1, le=32)
    max_tool_calls: int = Field(default=12, ge=1, le=64)


@lru_cache
def get_settings() -> Settings:
    return Settings()
