from dataclasses import dataclass
from typing import Any

from retailops.agent.harness import AgentHarness
from retailops.agent.policies import PolicyIndex
from retailops.config import Settings, get_settings
from retailops.db import Database
from retailops.telemetry import TraceStore
from retailops.tools.commerce import CommerceTools


@dataclass(frozen=True)
class Runtime:
    settings: Settings
    database: Database
    policies: PolicyIndex
    tools: CommerceTools
    traces: TraceStore
    harness: AgentHarness


def build_runtime(settings: Settings | None = None, model: Any | None = None) -> Runtime:
    selected = settings or get_settings()
    database = Database(selected)
    policies = PolicyIndex(selected.policy_dir)
    traces = TraceStore(selected.trace_dir)
    tools = CommerceTools(database.session_factory, policies)
    harness = AgentHarness(selected, tools, policies, traces, model=model)
    return Runtime(selected, database, policies, tools, traces, harness)
