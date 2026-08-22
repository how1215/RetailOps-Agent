import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import select

from retailops.config import Settings
from retailops.container import build_runtime
from retailops.domain.models import Order
from retailops.seed import seed_database


class EvaluationRunner:
    def __init__(
        self,
        settings: Settings,
        scenario_path: Path,
        model_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.scenario_path = scenario_path
        self.model_factory = model_factory

    def run(self) -> dict[str, Any]:
        scenarios = json.loads(self.scenario_path.read_text(encoding="utf-8"))
        results = [self._run_scenario(scenario) for scenario in scenarios]
        passed = sum(result["passed"] for result in results)
        unauthorized = sum(result["unauthorized_mutations"] for result in results)
        return {
            "summary": {
                "scenario_count": len(results),
                "passed": passed,
                "task_success_rate": round(passed / len(results), 4) if results else 0,
                "unauthorized_mutations": unauthorized,
                "model": self.settings.vllm_model,
            },
            "results": results,
        }

    def _run_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="retailops-eval-") as temp_dir:
            root = Path(temp_dir)
            scenario_settings = self.settings.model_copy(
                update={
                    "database_url": f"sqlite:///{root / 'eval.db'}",
                    "trace_dir": root / "traces",
                }
            )
            model = self.model_factory(scenario) if self.model_factory else None
            runtime = build_runtime(scenario_settings, model=model)
            seed_database(runtime.database)
            session_id, trace_id = runtime.harness.create_session(scenario["customer_id"])
            started = time.perf_counter()
            error: str | None = None
            try:
                response = runtime.harness.chat(session_id, scenario["message"])
                if response.pending_action and scenario.get("approve", False):
                    response = runtime.harness.resolve_approval(
                        session_id, response.pending_action["action_id"], True
                    )
                content = response.content
            except Exception as exc:  # evaluation must retain failed trajectories
                content = ""
                error = f"{type(exc).__name__}: {exc}"

            checks = self._checks(runtime, scenario, content)
            events = runtime.traces.get(trace_id)
            return {
                "id": scenario["id"],
                "passed": all(checks.values()) and error is None,
                "checks": checks,
                "unauthorized_mutations": self._unauthorized_mutations(events, scenario),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "llm_calls": sum(event["event"] == "llm.completed" for event in events),
                "tool_events": [
                    event.get("tool") for event in events if event["event"].startswith("tool.")
                ],
                "response": content,
                "error": error,
            }

    @staticmethod
    def _checks(runtime, scenario: dict[str, Any], content: str) -> dict[str, bool]:
        checks: dict[str, bool] = {}
        expected_order = scenario.get("expected_order")
        if expected_order:
            with runtime.database.session_factory() as session:
                order = session.scalar(select(Order).where(Order.id == expected_order["id"]))
                assert order is not None
                for field, expected in expected_order.items():
                    if field == "id":
                        continue
                    actual = getattr(order, field)
                    checks[f"order.{field}"] = (
                        actual is not None if expected == "NOT_NULL" else actual == expected
                    )
        for phrase in scenario.get("response_contains", []):
            checks[f"response_contains:{phrase}"] = phrase.lower() in content.lower()
        for phrase in scenario.get("response_excludes", []):
            checks[f"response_excludes:{phrase}"] = phrase.lower() not in content.lower()
        return checks or {"completed_without_exception": True}

    @staticmethod
    def _unauthorized_mutations(events: list[dict[str, Any]], scenario: dict[str, Any]) -> int:
        forbidden = set(scenario.get("forbidden_actions", []))
        return sum(
            event.get("tool") in forbidden
            and event["event"] == "approval.resolved"
            and event.get("approved") is True
            and event.get("result", {}).get("status") == "executed"
            for event in events
        )
