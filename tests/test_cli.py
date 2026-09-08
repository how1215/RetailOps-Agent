from conftest import ScriptedModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typer.testing import CliRunner

from retailops import cli


def test_llm_request_prints_model_response(monkeypatch) -> None:
    class RecordingModel(ScriptedModel):
        def invoke(self, messages):
            self.messages = messages
            return super().invoke(messages)

    model = RecordingModel([AIMessage(content="Gemini API OK")])
    monkeypatch.setattr(cli, "build_chat_model", lambda settings: model)

    result = CliRunner().invoke(
        cli.app,
        ["llm-request", "Test prompt", "--system", "Test system prompt"],
    )

    assert result.exit_code == 0
    assert result.stdout == "Gemini API OK\n"
    assert model.messages == [
        SystemMessage(content="Test system prompt"),
        HumanMessage(content="Test prompt"),
    ]
