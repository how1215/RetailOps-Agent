# Evaluation design

RetailOps evaluates observable outcomes rather than using another LLM as a judge. Each scenario
defines initial fixtures, scripted user turns, expected final database state, required actions,
and forbidden actions.

The initial comparison will cover:

1. A direct tool-calling loop with the entire policy in context.
2. Typed tools, validation, budgets, retries, and approval gates.
3. The reliable harness plus query-specific policy retrieval and context trimming.

Primary metrics are task success, unauthorized mutations, valid tool-call rate, recovery rate,
LLM/tool calls, input/output tokens, and latency. Results must include unsuccessful trajectories
and a failure taxonomy; only reporting successful demonstrations is not acceptable.
