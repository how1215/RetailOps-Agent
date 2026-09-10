from conftest import ScriptedModel
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from retailops.api import create_app
from retailops.container import build_runtime
from retailops.domain.models import Order


def mutation_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "cancel_order",
                "args": {"order_id": "ord_processing", "reason": "Ordered by mistake"},
                "id": "call_cancel",
                "type": "tool_call",
            }
        ],
    )


def test_session_and_message_contract(settings) -> None:
    runtime = build_runtime(
        settings,
        model=ScriptedModel([AIMessage(content="Hello from RetailOps.")]),
    )
    application = create_app(runtime)

    with TestClient(application) as client:
        session_response = client.post("/v1/sessions", json={"customer_id": "cus_001"})
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]

        message_response = client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"content": "Hello"},
        )

    assert message_response.status_code == 200
    assert message_response.json()["content"] == "Hello from RetailOps."
    assert message_response.json()["pending_action"] is None


def test_web_ui_is_served_at_root(settings) -> None:
    runtime = build_runtime(
        settings,
        model=ScriptedModel([AIMessage(content="Unused")]),
    )

    with TestClient(create_app(runtime)) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "RetailOps Agent Playground" in response.text
    assert 'action="/v1/sessions"' not in response.text


def test_approval_contract_executes_pending_action(settings) -> None:
    runtime = build_runtime(
        settings,
        model=ScriptedModel(
            [
                mutation_call(),
                AIMessage(content="The cancellation is complete."),
            ]
        ),
    )
    application = create_app(runtime)

    with TestClient(application) as client:
        session_id = client.post(
            "/v1/sessions", json={"customer_id": "cus_001"}
        ).json()["session_id"]
        proposed = client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"content": "Cancel ord_processing"},
        ).json()
        pending = proposed["pending_action"]
        assert pending["name"] == "cancel_order"

        completed = client.post(
            f"/v1/sessions/{session_id}/approvals",
            json={"action_id": pending["action_id"], "approved": True},
        )

    assert completed.status_code == 200
    assert completed.json()["content"] == "The cancellation is complete."
    with runtime.database.session_factory() as session:
        assert session.get(Order, "ord_processing").status == "cancelled"
