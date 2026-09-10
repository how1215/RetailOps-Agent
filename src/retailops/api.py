from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from retailops.agent.harness import AgentResult
from retailops.container import Runtime, build_runtime
from retailops.seed import seed_database
from retailops.tools.commerce import ToolError

WEB_INDEX = Path(__file__).with_name("web") / "index.html"


class CreateSessionRequest(BaseModel):
    customer_id: str = Field(min_length=3, max_length=32)


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8_000)


class ApprovalRequest(BaseModel):
    action_id: str
    approved: bool


class AgentResponse(BaseModel):
    session_id: str
    trace_id: str
    content: str
    pending_action: dict[str, Any] | None
    llm_calls: int
    tool_calls: int

    @classmethod
    def from_result(cls, result: AgentResult) -> "AgentResponse":
        return cls(**result.__dict__)


def create_app(runtime: Runtime | None = None) -> FastAPI:
    selected_runtime = runtime or build_runtime()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        selected_runtime.database.create_schema()
        seed_database(selected_runtime.database)
        app.state.runtime = selected_runtime
        yield

    app = FastAPI(
        title="RetailOps Agent API",
        version="0.1.0",
        description="Local-first commerce agent with approval-gated mutations.",
        lifespan=lifespan,
    )

    @app.exception_handler(ToolError)
    async def tool_error_handler(request: Request, exc: ToolError):
        return _http_error(400, str(exc))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False, response_class=FileResponse)
    def web_ui() -> FileResponse:
        return FileResponse(WEB_INDEX)

    @app.post("/v1/sessions")
    def create_session(payload: CreateSessionRequest) -> dict[str, str]:
        session_id, trace_id = selected_runtime.harness.create_session(payload.customer_id)
        return {"session_id": session_id, "trace_id": trace_id}

    @app.post("/v1/sessions/{session_id}/messages", response_model=AgentResponse)
    def send_message(session_id: str, payload: MessageRequest) -> AgentResponse:
        return AgentResponse.from_result(selected_runtime.harness.chat(session_id, payload.content))

    @app.post("/v1/sessions/{session_id}/approvals", response_model=AgentResponse)
    def resolve_approval(session_id: str, payload: ApprovalRequest) -> AgentResponse:
        result = selected_runtime.harness.resolve_approval(
            session_id, payload.action_id, payload.approved
        )
        return AgentResponse.from_result(result)

    @app.get("/v1/traces/{trace_id}")
    def get_trace(trace_id: str) -> dict[str, Any]:
        return {"trace_id": trace_id, "events": selected_runtime.traces.get(trace_id)}

    return app


def _http_error(status_code: int, detail: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})


app = create_app()
