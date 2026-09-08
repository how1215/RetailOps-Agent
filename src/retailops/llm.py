from langchain_openai import ChatOpenAI

from retailops.config import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Build the OpenAI-compatible chat model from centralized settings."""
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
    )
