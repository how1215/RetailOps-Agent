import json
import time
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from retailops.agent.policies import PolicyIndex
from retailops.config import Settings
from retailops.telemetry import TraceStore
from retailops.tools.commerce import CommerceTools, ToolContext, ToolError


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str
    trace_id: str
    llm_calls: int
    tool_calls: int
    pending_action: dict[str, Any] | None


@dataclass(frozen=True)
class AgentResult:
    session_id: str
    trace_id: str
    content: str
    pending_action: dict[str, Any] | None
    llm_calls: int
    tool_calls: int


class AgentHarness:
    def __init__(
        self,
        settings: Settings,
        tools: CommerceTools,
        policies: PolicyIndex,
        traces: TraceStore,
        model: Any | None = None,
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.policies = policies
        self.traces = traces
        self.model = model or ChatOpenAI(
            model=settings.vllm_model,
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            temperature=0,
        )
        self.model_with_tools = self.model.bind_tools(tools.schemas())
        self._sessions: dict[str, tuple[str, str]] = {}
        self.graph = self._build_graph()

    def create_session(self, customer_id: str) -> tuple[str, str]:
        self.tools.assert_customer_exists(customer_id)
        session_id = f"ses_{uuid4().hex}"
        trace_id = f"trc_{uuid4().hex}"
        self._sessions[session_id] = (customer_id, trace_id)
        self.traces.record(
            trace_id,
            "session.created",
            session_id=session_id,
            customer_id=customer_id,
        )
        return session_id, trace_id

    def chat(self, session_id: str, content: str) -> AgentResult:
        customer_id, trace_id = self._session(session_id)
        snapshot = self.graph.get_state(self._config(session_id))
        if snapshot.values.get("pending_action"):
            raise ToolError("Resolve the pending action before sending another message.")
        self.traces.record(trace_id, "user.message", content=content)
        output = self.graph.invoke(
            {
                "messages": [HumanMessage(content=content)],
                "customer_id": customer_id,
                "trace_id": trace_id,
                "pending_action": None,
                "llm_calls": 0,
                "tool_calls": 0,
            },
            self._config(session_id),
        )
        return self._result(session_id, output)

    def resolve_approval(
        self, session_id: str, action_id: str, approved: bool
    ) -> AgentResult:
        customer_id, trace_id = self._session(session_id)
        snapshot = self.graph.get_state(self._config(session_id))
        state = snapshot.values
        pending = state.get("pending_action")
        if pending is None or pending.get("action_id") != action_id:
            raise ToolError("Pending action was not found or no longer matches.")

        if approved:
            started = time.perf_counter()
            try:
                result = self.tools.execute(
                    pending["name"],
                    pending["args"],
                    ToolContext(
                        session_id=session_id,
                        customer_id=customer_id,
                        idempotency_key=action_id,
                    ),
                )
                payload = {"status": "executed", "result": result}
            except (ToolError, ValidationError) as exc:
                payload = {"status": "error", "error": str(exc)}
            self.traces.record(
                trace_id,
                "approval.resolved",
                approved=True,
                action_id=action_id,
                tool=pending["name"],
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                result=payload,
            )
        else:
            payload = {"status": "declined", "action_id": action_id}
            self.traces.record(trace_id, "approval.resolved", approved=False, action_id=action_id)

        output = self.graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "[Trusted approval event] The previously proposed action was resolved. "
                            f"Execution result: {json.dumps(payload)}"
                        )
                    )
                ],
                "customer_id": customer_id,
                "trace_id": trace_id,
                "pending_action": None,
            },
            self._config(session_id),
        )
        return self._result(session_id, output)

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("assistant", self._assistant_node)
        builder.add_node("tools", self._tool_node)
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", self._route_after_assistant, ["tools", END])
        builder.add_conditional_edges("tools", self._route_after_tools, ["assistant", END])
        return builder.compile(checkpointer=MemorySaver())

    def _assistant_node(self, state: AgentState) -> dict[str, Any]:
        llm_calls = state.get("llm_calls", 0)
        if llm_calls >= self.settings.max_llm_calls:
            return {
                "messages": [
                    AIMessage(content="I reached the reasoning budget. Please escalate this case.")
                ]
            }

        last_user = next(
            (
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        policy_context = self.policies.context(str(last_user))
        system = SystemMessage(content=self._system_prompt(policy_context))
        started = time.perf_counter()
        response = self.model_with_tools.invoke([system, *state["messages"]])
        self.traces.record(
            state["trace_id"],
            "llm.completed",
            call_index=llm_calls + 1,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            tool_names=[call["name"] for call in getattr(response, "tool_calls", [])],
            usage=getattr(response, "usage_metadata", None),
        )
        return {"messages": [response], "llm_calls": llm_calls + 1}

    def _tool_node(self, state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        assert isinstance(last_message, AIMessage)
        results: list[ToolMessage] = []
        pending: dict[str, Any] | None = None
        tool_calls = state.get("tool_calls", 0)

        for call in last_message.tool_calls:
            tool_calls += 1
            if tool_calls > self.settings.max_tool_calls:
                results.append(
                    ToolMessage(
                        content=json.dumps({"status": "error", "error": "Tool budget exceeded."}),
                        tool_call_id=call["id"],
                    )
                )
                continue

            try:
                spec = self.tools.get(call["name"])
                parsed = self.tools.validate(call["name"], call["args"])
                if spec.requires_approval:
                    if pending is not None:
                        raise ToolError("Only one state-changing action may be pending at a time.")
                    action_id = f"act_{uuid4().hex}"
                    pending = {
                        "action_id": action_id,
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "args": parsed.model_dump(),
                    }
                    result = {
                        "status": "approval_required",
                        "action_id": action_id,
                        "action": call["name"],
                        "arguments": parsed.model_dump(),
                    }
                else:
                    result = self.tools.execute(
                        call["name"],
                        parsed.model_dump(),
                        ToolContext(
                            session_id=self._session_id_from_trace(state["trace_id"]),
                            customer_id=state["customer_id"],
                        ),
                    )
            except (ToolError, ValidationError) as exc:
                result = {"status": "error", "error": str(exc)}

            self.traces.record(
                state["trace_id"],
                "tool.completed" if not pending else "tool.approval_requested",
                tool=call["name"],
                result=result,
            )
            results.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))

        return {"messages": results, "tool_calls": tool_calls, "pending_action": pending}

    @staticmethod
    def _route_after_assistant(state: AgentState) -> str:
        last_message = state["messages"][-1]
        return "tools" if isinstance(last_message, AIMessage) and last_message.tool_calls else END

    @staticmethod
    def _route_after_tools(state: AgentState) -> str:
        return END if state.get("pending_action") else "assistant"

    @staticmethod
    def _system_prompt(policy_context: str) -> str:
        return f"""You are RetailOps, a careful commerce operations assistant.
The API caller has already authenticated the customer. Use tools for order facts; never invent
order state. Follow the policy excerpts below. A mutation tool returns an approval request first,
so do not claim it executed until its later tool result says executed. Do not repeat a pending
mutation. If policy forbids an action, explain why and offer escalation. Keep replies concise.

Relevant policy excerpts:
{policy_context}
"""

    def _result(self, session_id: str, state: AgentState) -> AgentResult:
        last_ai = next(
            (message for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
            AIMessage(content=""),
        )
        pending = state.get("pending_action")
        content = str(last_ai.content)
        if pending and not content:
            content = f"Approval is required for {pending['name']}."
        return AgentResult(
            session_id=session_id,
            trace_id=state["trace_id"],
            content=content,
            pending_action=pending,
            llm_calls=state.get("llm_calls", 0),
            tool_calls=state.get("tool_calls", 0),
        )

    def _session(self, session_id: str) -> tuple[str, str]:
        if session_id not in self._sessions:
            raise ToolError("Unknown session.")
        return self._sessions[session_id]

    def _session_id_from_trace(self, trace_id: str) -> str:
        for session_id, (_, stored_trace_id) in self._sessions.items():
            if stored_trace_id == trace_id:
                return session_id
        raise ToolError("Trace does not belong to an active session.")

    @staticmethod
    def _config(session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}
