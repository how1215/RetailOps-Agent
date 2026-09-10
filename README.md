# RetailOps Agent

RetailOps is a local-first, policy-aware commerce agent built to demonstrate reliable LLM
systems engineering. It uses an OpenAI-compatible endpoint (Gemma 4 through vLLM by default)
and combines LangGraph orchestration with a custom harness for typed tools, approval gates,
budgets, telemetry, and deterministic evaluation.

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

Runtime configuration is loaded from `.env` through the typed `Settings` class. Model creation is
centralized in `src/retailops/llm.py`, and the Agent system prompt lives in
`prompts/retailops_system.txt`. This keeps deployment secrets, tuning parameters, and prompt text
separate. The older `VLLM_MODEL`, `VLLM_BASE_URL`, and `VLLM_API_KEY` names are still accepted for
backward compatibility.

Common tuning settings:

| Setting | Purpose | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | Client integration (`openai_compatible` or `google_genai`) | `openai_compatible` |
| `LLM_MODEL` | Provider model name | Gemma 4 vLLM model |
| `LLM_BASE_URL` | API endpoint used by `openai_compatible` | Local vLLM |
| `LLM_API_KEY` | API credential | `dummy` for local vLLM |
| `LLM_TEMPERATURE` | Model response randomness | `0` |
| `AGENT_SYSTEM_PROMPT_PATH` | Version-controlled system prompt template | `prompts/retailops_system.txt` |
| `POLICY_CONTEXT_LIMIT` | Maximum retrieved policy chunks per LLM call | `4` |
| `MAX_LLM_CALLS` | LLM call budget per user turn | `8` |
| `MAX_TOOL_CALLS` | Tool call budget per user turn | `12` |

The prompt template must retain the `{policy_context}` placeholder. Business invariants such as
customer ownership, approval requirements, and return eligibility remain enforced in code rather
than being runtime-tunable configuration.

### Google AI Studio / Gemini API

Gemini tool calls require provider metadata such as thought signatures to survive across turns.
The native Google integration preserves that metadata. Create an API key in Google AI Studio,
then configure `.env` as follows (`LLM_BASE_URL` is ignored by this provider):

```dotenv
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-3.7-flash
LLM_API_KEY=your-google-ai-studio-api-key
LLM_TEMPERATURE=1
```

Send a prompt directly to the configured model without running the RetailOps agent or tools:

```bash
uv run retailops llm-request "Reply with exactly: Gemini API OK"
```

An optional system prompt can be supplied separately:

```bash
uv run retailops llm-request "Explain idempotency." --system "Answer in Traditional Chinese."
```

Keep the API key out of version control. The same provider settings work with Docker Compose:

```dotenv
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-3.7-flash
LLM_API_KEY=your-google-ai-studio-api-key
LLM_TEMPERATURE=1
```

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
OpenAI-compatible LLM
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

Set `DOCKER_LLM_BASE_URL` when the model endpoint is not available at
`http://host.docker.internal:8000/v1`.

See [docs/architecture.md](docs/architecture.md) for the main design decisions and current
persistence boundary.
