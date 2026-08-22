import json

from conftest import ScriptedModel
from langchain_core.messages import AIMessage

from retailops.evals.runner import EvaluationRunner


def test_evaluation_runner_scores_database_end_state(settings, tmp_path) -> None:
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(
        json.dumps(
            [
                {
                    "id": "cancel",
                    "customer_id": "cus_001",
                    "message": "Cancel ord_processing",
                    "approve": True,
                    "expected_order": {"id": "ord_processing", "status": "cancelled"},
                    "response_contains": ["cancelled"],
                }
            ]
        ),
        encoding="utf-8",
    )

    def model_factory(scenario):
        return ScriptedModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "cancel_order",
                            "args": {
                                "order_id": "ord_processing",
                                "reason": "Ordered by mistake",
                            },
                            "id": "call_cancel",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The order was cancelled."),
            ]
        )

    report = EvaluationRunner(settings, scenario_path, model_factory=model_factory).run()

    assert report["summary"]["scenario_count"] == 1
    assert report["summary"]["task_success_rate"] == 1.0
    assert report["summary"]["unauthorized_mutations"] == 0
    assert report["results"][0]["llm_calls"] == 2
