# Architecture decisions

## LangGraph is orchestration, not the safety boundary

LangGraph owns state transitions and checkpoints. The custom harness owns model adaptation,
context construction, call budgets, typed tool dispatch, approval state, and traces. Commerce
preconditions are enforced again inside each tool so a bad model decision cannot bypass policy.

## Mutations are proposals

A state-changing tool call first becomes a `pending_action` with an unguessable action ID. The
request ends at that boundary. A separate API call approves or declines the exact action. On
approval, the tool revalidates customer ownership and current order state, executes with the
action ID as an idempotency key, and writes an audit event.

## Retrieval starts with a measurable baseline

The first policy index uses transparent lexical overlap. This is intentionally simple: it creates
a baseline for measuring whether local dense embeddings or hybrid retrieval improve task success
enough to justify their latency and operational cost.

## Evaluation uses observable outcomes

The evaluator creates a fresh database for every scenario. It grades the final database state,
required response facts, forbidden disclosures, and executed actions. Failed calls and complete
traces remain in the output instead of being discarded.

## Current persistence boundary

Commerce data and audit events persist in SQLite. Conversation checkpoints and live session
metadata are in process for the first vertical slice. A production iteration should replace the
memory checkpointer with PostgreSQL-backed persistence before horizontal scaling.
