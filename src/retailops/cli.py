import json
from pathlib import Path

import typer

from retailops.config import get_settings
from retailops.container import build_runtime
from retailops.evals.runner import EvaluationRunner
from retailops.seed import seed_database

app = typer.Typer(help="RetailOps local-first agent platform.", no_args_is_help=True)


@app.command()
def seed() -> None:
    """Create the local schema and deterministic demo records."""
    runtime = build_runtime()
    seed_database(runtime.database)
    typer.echo("Seeded customers cus_001/cus_002 and demo orders.")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Serve the FastAPI application."""
    import uvicorn

    uvicorn.run("retailops.api:app", host=host, port=port, reload=False)


@app.command()
def chat(customer_id: str = "cus_001") -> None:
    """Run an interactive terminal session against the configured local model."""
    runtime = build_runtime()
    seed_database(runtime.database)
    session_id, trace_id = runtime.harness.create_session(customer_id)
    typer.echo(f"Session {session_id} (trace {trace_id}). Type 'exit' to stop.")
    while True:
        content = typer.prompt("you")
        if content.strip().lower() in {"exit", "quit"}:
            break
        result = runtime.harness.chat(session_id, content)
        typer.echo(f"agent: {result.content}")
        if result.pending_action:
            typer.echo(json.dumps(result.pending_action, indent=2))
            approved = typer.confirm("Approve this action?", default=False)
            result = runtime.harness.resolve_approval(
                session_id, result.pending_action["action_id"], approved
            )
            typer.echo(f"agent: {result.content}")


@app.command("eval-run")
def eval_run(
    scenarios: Path = Path("src/retailops/evals/scenarios.json"),
    output: Path = Path("data/eval-results.json"),
) -> None:
    """Evaluate the configured local model using deterministic end-state checks."""
    runner = EvaluationRunner(get_settings(), scenarios)
    report = runner.run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report["summary"], indent=2))
    typer.echo(f"Full report: {output}")


if __name__ == "__main__":
    app()
