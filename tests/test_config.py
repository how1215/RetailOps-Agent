from retailops.config import Settings


def test_legacy_vllm_setting_names_remain_supported() -> None:
    settings = Settings(
        VLLM_MODEL="legacy-model",
        VLLM_BASE_URL="http://legacy.example/v1",
        VLLM_API_KEY="legacy-key",
    )

    assert settings.llm_model == "legacy-model"
    assert settings.llm_base_url == "http://legacy.example/v1"
    assert settings.llm_api_key == "legacy-key"
