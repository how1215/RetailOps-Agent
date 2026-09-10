from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from retailops.config import Settings


def build_chat_model(settings: Settings) -> Any:
    """Build the configured provider-specific chat model."""
    if settings.llm_provider == "google_genai":
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            api_key=SecretStr(settings.llm_api_key),
            temperature=settings.llm_temperature,
        )

    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=SecretStr(settings.llm_api_key),
        temperature=settings.llm_temperature,
    )
