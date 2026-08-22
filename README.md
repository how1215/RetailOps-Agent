# RetailOps Agent

RetailOps is a local-first, policy-aware commerce agent built to demonstrate reliable LLM
systems engineering. It uses Gemma 4 through a vLLM OpenAI-compatible endpoint and combines
LangGraph orchestration with a custom harness for typed tools, approval gates, budgets,
telemetry, and deterministic evaluation.

## Current vertical slice

- Read-only order lookup and policy retrieval.
- Approval-gated cancellation, shipping-address changes, and return creation.
- Customer-scoped tools with Pydantic validation, idempotency, and audit events.
- In-process LangGraph checkpoints and JSONL trajectory telemetry.
- FastAPI and CLI entry points.
- Deterministic scenario format plus a local-model evaluation runner.

## Quick start

```bash
cp .env.example .env
uv sync
uv run retailops seed
uv run retailops chat --customer-id cus_001
```

Run the API:

```bash
uv run retailops serve
```

Run tests and the local evaluation suite:

```bash
uv run pytest
uv run retailops eval-run
```

The vLLM endpoint is configured with `VLLM_BASE_URL`, `VLLM_MODEL`, and `VLLM_API_KEY`.
No cloud model is required.

## Safety model

Read tools may run immediately. Every state-changing tool is converted into a pending action
and cannot execute until the caller approves its unique action ID. Execution re-checks customer
ownership and business preconditions, and an idempotency key prevents duplicate mutations.

## Architecture

```text
User/API
   │
   ▼
Context builder ── policy index
   │
   ▼
Gemma 4 / vLLM
   │ tool calls
   ▼
Harness validator ── budget ── approval gate
   │
   ▼
Commerce tool registry ── SQLite ── audit log
```

See [docs/evaluation.md](docs/evaluation.md) for metrics and experiment design.

## HTTP example

```bash
curl -s http://localhost:8080/v1/sessions \
  -H 'content-type: application/json' \
  -d '{"customer_id":"cus_001"}'

curl -s http://localhost:8080/v1/sessions/SESSION_ID/messages \
  -H 'content-type: application/json' \
  -d '{"content":"What is the status of ord_processing?"}'
```

When `pending_action` is present, approve that exact action separately:

```bash
curl -s http://localhost:8080/v1/sessions/SESSION_ID/approvals \
  -H 'content-type: application/json' \
  -d '{"action_id":"ACTION_ID","approved":true}'
```

## Verification without vLLM

The unit, workflow, API, and evaluator tests use scripted model responses and never contact a
model endpoint:

```bash
make check
```

This verifies customer isolation, approval gating, declined actions, idempotency, policy
retrieval, the HTTP contract, database end-state grading, and the complete read-tool loop.

## Containers

The API can run in Docker while vLLM remains on the host:

```bash
docker compose up --build
```

Set `DOCKER_VLLM_BASE_URL` when the model endpoint is not available at
`http://host.docker.internal:8000/v1`.

See [docs/architecture.md](docs/architecture.md) for the main design decisions and current
persistence boundary.
