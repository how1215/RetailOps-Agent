from conftest import ScriptedModel
from langchain_core.messages import AIMessage, SystemMessage

from retailops.agent.harness import AgentHarness
from retailops.domain.models import Order


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def test_read_tool_loops_back_to_model(settings, tools, policies, traces) -> None:
    signed_tool_call = tool_call("get_order", {"order_id": "ord_processing"})
    signed_tool_call.additional_kwargs["__gemini_function_call_thought_signatures__"] = {
        "call_1": "test-signature"
    }
    model = ScriptedModel(
        [
            signed_tool_call,
            AIMessage(content="Your order is processing."),
        ]
    )
    harness = AgentHarness(settings, tools, policies, traces, model=model)
    session_id, _ = harness.create_session("cus_001")

    result = harness.chat(session_id, "What is the status of ord_processing?")

    assert result.content == "Your order is processing."
    assert result.pending_action is None
    assert result.llm_calls == 2
    assert result.tool_calls == 1
    replayed_call = next(
        message
        for message in model.invocations[1]
        if isinstance(message, AIMessage) and message.tool_calls
    )
    assert replayed_call.additional_kwargs[
        "__gemini_function_call_thought_signatures__"
    ] == {"call_1": "test-signature"}


def test_result_extracts_text_from_native_content_blocks(
    settings, tools, policies, traces
) -> None:
    model = ScriptedModel(
        [AIMessage(content=[{"type": "text", "text": "Native Gemini response."}])]
    )
    harness = AgentHarness(settings, tools, policies, traces, model=model)
    session_id, _ = harness.create_session("cus_001")

    result = harness.chat(session_id, "Hello")

    assert result.content == "Native Gemini response."


def test_system_prompt_and_retrieval_limit_come_from_settings(
    settings, tools, policies, traces, tmp_path
) -> None:
    prompt_path = tmp_path / "system.txt"
    prompt_path.write_text("Configured prompt\n{policy_context}", encoding="utf-8")
    configured = settings.model_copy(
        update={"agent_system_prompt_path": prompt_path, "policy_context_limit": 1}
    )
    model = ScriptedModel([AIMessage(content="Done")])
    harness = AgentHarness(configured, tools, policies, traces, model=model)
    session_id, _ = harness.create_session("cus_001")

    harness.chat(session_id, "Can I cancel an order?")

    system = model.invocations[0][0]
    assert isinstance(system, SystemMessage)
    assert str(system.content).startswith("Configured prompt")
    assert str(system.content).count("[Policy:") == 1


def test_mutation_waits_for_approval_then_executes(
    settings, tools, policies, traces, database
) -> None:
    model = ScriptedModel(
        [
            tool_call(
                "cancel_order",
                {"order_id": "ord_processing", "reason": "Ordered by mistake"},
            ),
            AIMessage(content="The order was cancelled."),
        ]
    )
    harness = AgentHarness(settings, tools, policies, traces, model=model)
    session_id, _ = harness.create_session("cus_001")

    proposed = harness.chat(session_id, "Cancel ord_processing.")
    with database.session_factory() as session:
        assert session.get(Order, "ord_processing").status == "processing"

    assert proposed.pending_action is not None
    completed = harness.resolve_approval(
        session_id, proposed.pending_action["action_id"], approved=True
    )

    with database.session_factory() as session:
        assert session.get(Order, "ord_processing").status == "cancelled"
    assert completed.content == "The order was cancelled."


def test_declined_mutation_does_not_change_database(
    settings, tools, policies, traces, database
) -> None:
    model = ScriptedModel(
        [
            tool_call(
                "update_shipping_address",
                {"order_id": "ord_processing", "new_address": "200 New Road, Taipei"},
            ),
            AIMessage(content="I left the address unchanged."),
        ]
    )
    harness = AgentHarness(settings, tools, policies, traces, model=model)
    session_id, _ = harness.create_session("cus_001")

    proposed = harness.chat(session_id, "Move my delivery to 200 New Road, Taipei.")
    assert proposed.pending_action is not None
    harness.resolve_approval(session_id, proposed.pending_action["action_id"], approved=False)

    with database.session_factory() as session:
        order = session.get(Order, "ord_processing")
        assert order.shipping_address == "100 Demo Road, Taipei"
