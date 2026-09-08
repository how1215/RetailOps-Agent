from collections import deque
from pathlib import Path
from typing import Any

import pytest

from retailops.agent.policies import PolicyIndex
from retailops.config import Settings
from retailops.db import Database
from retailops.seed import seed_database
from retailops.telemetry import TraceStore
from retailops.tools.commerce import CommerceTools


class ScriptedModel:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = deque(responses)
        self.bound_tools: list[dict[str, Any]] = []
        self.invocations: list[list[Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]]):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        if not self.responses:
            raise AssertionError("Scripted model has no response left")
        return self.responses.popleft()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        policy_dir=Path("policies"),
        trace_dir=tmp_path / "traces",
    )


@pytest.fixture
def database(settings: Settings) -> Database:
    db = Database(settings)
    seed_database(db)
    return db


@pytest.fixture
def policies(settings: Settings) -> PolicyIndex:
    return PolicyIndex(settings.policy_dir)


@pytest.fixture
def tools(database: Database, policies: PolicyIndex) -> CommerceTools:
    return CommerceTools(database.session_factory, policies)


@pytest.fixture
def traces(settings: Settings) -> TraceStore:
    return TraceStore(settings.trace_dir)
