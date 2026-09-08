from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    llm_model: str = Field(
        default="google/gemma-4-31B-it-qat-w4a16-ct",
        validation_alias=AliasChoices("LLM_MODEL", "VLLM_MODEL"),
    )
    llm_base_url: str = Field(
        default="http://localhost:8000/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "VLLM_BASE_URL"),
    )
    llm_api_key: str = Field(
        default="dummy",
        validation_alias=AliasChoices("LLM_API_KEY", "VLLM_API_KEY"),
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    database_url: str = "sqlite:///data/retailops.db"
    policy_dir: Path = Path("policies")
    trace_dir: Path = Path("traces")
    agent_system_prompt_path: Path = Path("prompts/retailops_system.txt")
    policy_context_limit: int = Field(default=4, ge=1, le=20)
    max_llm_calls: int = Field(default=8, ge=1, le=32)
    max_tool_calls: int = Field(default=12, ge=1, le=64)


@lru_cache
def get_settings() -> Settings:
    return Settings()
