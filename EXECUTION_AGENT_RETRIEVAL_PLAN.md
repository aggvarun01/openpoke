# Plan: Configurable Hybrid Agent Retrieval

## Goal and Scope

Replace the full roster in Interaction Agent prompts with a bounded set of relevant Execution Agents. Combine exact matching and semantic search while preserving existing agent names, logs, and trigger ownership.

- Configure exact and semantic fields in code, not through an in-app settings page.
- Generate embeddings through an external API. Profile and query text sent to the provider leave the machine.
- Defer conversational context, pronoun resolution, and recency ranking to follow-up work.
- Do not archive agents, rewrite existing logs, or introduce a vector database in the first iteration.

## Search Tradeoffs

- **Direct exact matching:** Scan configured exact-field values in agent profiles against each user message or lookup query, alongside semantic search. For example, a stored person `Alice` matches `Did Alice reply?` on whole-phrase boundaries. This does not require the system to identify `Alice` as a name in the query, but it does require that name or alias to be present in a profile. Structured profile extraction still populates those fields.
- **No NER:** Do not add a named-entity recognition component. NER could help identify entities in a query, but it can miss unfamiliar names or misclassify text and would still require matching recognized entities to profiles. Direct matching can also produce false positives for names that are ordinary words; a match is a candidate, not proof of identity.
- **No fuzzy matching:** Misspelled names such as `Ailce` will not match `Alice` exactly. Fuzzy matching could recover typos but can confuse similar names; semantic search might help but is not a reliable spelling correction mechanism. Email addresses and other identifiers also require exact matches.
- **No search indexes:** Scan exact fields and stored semantic vectors instead of building separate exact-field or vector-search indexes. For hundreds of agents this keeps synchronization and implementation simpler, but search cost grows with the roster. Measure latency during evaluation and revisit indexing only if needed. Persisting embeddings for semantic search is still required.

## Git and PR Workflow

Use a stacked branch for each numbered feature and a separate aggregate branch for the final PR:

```text
main
	-> feature/00-plan
	-> feature/01-profiles
	-> feature/02-updates
	-> feature/03-backfill
	-> ...
	-> feature/10-evaluation
	-> feature/final
```

- Base `feature/00-plan` on `main`, the first feature branch on `feature/00-plan`, and each subsequent feature branch on its immediate predecessor. Open the plan PR against `main` and each feature PR against its parent branch, so each diff shows only its own changes. Branch and PR names may vary, but keep the ordering and bases clear.
- Open a separate final PR from `feature/final` to `main` to show the complete changes. Keep the aggregate branch synchronized with the latest stack tip and ensure the plan is included in the final diff.
- Leave all PRs open and do not merge into `main` before the presentation. The individual PRs are separately reviewable, but depend on their parent branches.
- When an earlier branch changes, coordinate propagation through its descendants and update the aggregate branch. Preserve existing work and verify each PR still has the intended base and diff; do not rewrite shared history or force-push without authorization.
- After the presentation, decide with the user whether to merge the aggregate PR and close the individual PRs or use another merge strategy. Do not merge or close PRs without approval.
- Contributors and sub-agents should follow this workflow, but create branches, commits, or PRs only when explicitly authorized for the current task.

## Feature Branches

Build these branches in order; each builds on the previous one. Keep all PRs open and do not merge into `main` before the presentation. Keep the existing full-roster prompt behavior until branch 9, so partially implemented retrieval does not silently change agent routing. Each branch includes its own focused tests and can be reviewed separately against its parent branch.

### 1. Profile Schema and Storage

- Define a typed retrieval profile and field registry under `server/services/execution/`. Set developer-maintained `EXACT_MATCH_FIELDS` to name, aliases, people, and email addresses, and `SEMANTIC_MATCH_FIELDS` to name, purpose, and topics. A field may be in both lists; validate names and types at startup and define deterministic scalar and list serialization. Adding a field also requires a population rule.
- Store profiles in a separate SQLite database under `server/data/execution_agents`, keyed by existing agent names. The roster remains authoritative. Integrate profile clearing with chat-history clearing and provide a generation or version guard against in-flight updates restoring cleared data.
- **Tests:** configuration validation, field serialization, profile persistence, name identity, clearing, and stale writes after clearing.

### 2. Incremental Profile Updates

- Populate profiles from agent names, delegated instructions, and execution responses. Extract purpose, aliases, people, email addresses, and topics as structured data; retain only values supported by the source and leave unavailable fields empty. Do not embed raw logs or tool payloads.
- Update profiles asynchronously for direct and scheduled executions without blocking dispatch. Retain the existing profile on extraction failure and prevent older concurrent updates from overwriting newer ones.
- **Tests:** grounded and missing fields, direct and scheduled updates, extraction failure, overlapping updates, and clear during an in-flight update.

### 3. Existing Agent Backfill

- Add resumable background backfill for existing roster entries using names and recorded requests and responses. Preserve existing agent names, logs, and trigger ownership; make repeated runs safe.
- **Tests:** backfill of existing agents, missing logs, restart and rerun, concurrent live updates, and clearing during backfill.

### 4. Exact Agent Search

- Scan configured exact fields against the query. Normalize Unicode case and whitespace; match whole name, alias, and person phrases on text boundaries and complete normalized email tokens. Empty fields do not match. Do not use NER, fuzzy matching, or a search index. Return deterministic, bounded, paginated results without changing prompt behavior.
- **Tests:** configured field inclusion and exclusion, normalization, similar names and substring boundaries, email tokens, empty fields, ordering, limits, and pagination.

### 5. Embedding Provider Client

- Add an async external embeddings client using the existing HTTP client style and an explicit model setting. Confirm endpoint and model availability before choosing a default. Handle timeouts, errors, and retries without blocking existing agent execution.
- Document provider configuration, data sharing, costs, and retention in `README.md`. This branch does not yet embed agent profiles or queries in production.
- **Tests:** mocked successful responses, invalid responses, model configuration, timeouts, and provider errors.

### 6. Profile Embedding Storage

- Embed labeled, deterministic serializations of only configured semantic fields, never raw logs. Store vectors with model, dimensions, and profile/configuration fingerprints in the derived SQLite store.
- Rebuild when profiles, selected fields, or the model change. Prevent stale asynchronous writes, never compare incompatible vectors, and leave profiles usable for exact search if embedding generation fails. Document configuration and rebuilding embeddings.
- **Tests:** selected fields, profile and configuration changes, model and dimension changes, embedding failures, concurrent updates, and clearing vectors with history.

### 7. Hybrid Retrieval

- Embed the latest user message or lookup query, not the conversation transcript. Combine exact matches with semantic cosine similarity over compatible vectors; rank exact matches first with deterministic tie breaking. Use a linear scan for hundreds of agents and do not assume an untested similarity threshold.
- Fall back to exact results when embeddings are unavailable, and return an explicit incomplete or degraded status. Keep profiles without vectors eligible for exact search.
- **Tests:** semantic synonyms, exact priority, unrelated and similar names, missing or incompatible vectors, provider failures, result limits, and pagination.

### 8. On-Demand Search Tool

- Add a paginated `search_agents` Interaction Agent tool backed by hybrid retrieval, and adapt the tool dispatcher for asynchronous embedding calls. Keep the existing full roster in prompts for now.
- Add guidance to search before creating an agent when unsure and to preserve existing names. Report degraded lookup without claiming that no agent exists; ask for clarification when reuse cannot be established. This reduces, but does not guarantee elimination of, duplicate creation.
- **Tests:** tool dispatch, pagination when exact matches exceed the limit, degraded results, and reuse versus creation guidance.

### 9. Bounded Interaction Prompts

- Replace full-roster injection on user turns with bounded hybrid candidates and expose incomplete or degraded lookup status. On agent-result turns, pass originating agent names explicitly from structured batch data, including multiple agents and scheduled executions. Watcher notifications have no originating Execution Agent and must not receive the full roster.
- Preserve existing execution and trigger behavior when search or embedding generation is unavailable; avoid confident no-match claims when retrieval is incomplete.
- **Tests:** user and agent turns, multi-agent batches, scheduled executions, watcher notifications, candidate limits, fallback behavior, and preservation of agent names.

### 10. Agent Evaluation and Submission Evidence

- Build a labeled synthetic dataset with hundreds of agents and human-written expected outcomes: reuse one or more agents, create a new agent, or ask for clarification. Cover exact names, semantic paraphrases, multiple agents sharing a name, unrelated requests, and older relevant agents. Include typos and pronoun references as separately reported known limitations, not as requirements for fuzzy matching or conversational context.
- Evaluate retrieval using real provider embeddings: measure recall at the candidate limit for cases with relevant agents, candidate count, and prompt size. Report results by case type and examine false candidates for no-match and ambiguous cases. Compare candidate recall and prompt size with the current full-roster baseline.
- Evaluate the Interaction Agent with controlled, fake execution tools. Compare hybrid retrieval against the full-roster baseline on the same cases: measure correct reuse, appropriate creation or clarification, and duplicate-agent creation. Repeat model-backed cases and report variability, latency, and API cost. Never send real emails or run other external actions.
- Keep a held-out set separate from ranking and candidate-limit tuning. Establish and document acceptance thresholds from a baseline and project requirements before evaluating the held-out set; report representative failures and limitations rather than hiding them in an aggregate score.
- Include fixtures, evaluator scripts, reproducible commands, configuration, results, and instructions distinguishing deterministic tests from optional live-provider evaluations. Document how the agent was tested and how its performance was assessed.
- **Tests:** repeatable fixture and evaluator calculations with fake embeddings and model responses, correct handling of multiple expected agents and no-match cases, plus regression cases for observed failures.

## Testing and Evaluation

- **Per-branch tests:** Run the focused tests listed above using fake embeddings, model responses, and execution tools. Cover profile lifecycle, matching, ranking, provider failures, concurrent updates, and direct, scheduled, batch, and watcher paths without API keys or a real inbox.
- **End-to-end tests:** Exercise the interaction flow with controlled responses and execution tools, verifying that the expected agent is reused, a new agent is created when appropriate, or the user is asked for clarification. Verify that degraded lookup is surfaced and cannot silently be treated as a definitive no-match result.
- **Model-backed evaluation:** Run the labeled benchmark against the selected embedding provider and Interaction Agent model, with synthetic data and fake execution tools. Record models, configuration, repetitions, results, and costs so the assessment can be reproduced. Live-provider evaluations require credentials and are separate from the offline test suite.
- **Submission:** Include the test cases, fixtures, evaluator scripts, commands, baseline comparison, chosen thresholds, and a concise report of results and known limitations.

## Integration Points

- `server/services/execution/roster.py`: authoritative list of agent names.
- `server/services/execution/log_store.py`: source for backfill; do not embed raw logs.
- `server/agents/interaction_agent/agent.py` and `runtime.py`: bounded candidate injection for user and agent turns.
- `server/agents/interaction_agent/tools.py` and `system_prompt.md`: lookup tool, async dispatch, and reuse guidance.
- `server/agents/execution_agent/batch_manager.py` and `server/services/trigger_scheduler.py`: direct and scheduled updates and structured agent identities.
- `server/openrouter_client/client.py` and `server/config.py`: external embeddings API and model configuration.
- `server/routes/chat.py`: clear derived retrieval data with existing state.
- `README.md`: developer configuration and privacy documentation.

## Follow-up

- Add conversational context and recency ranking, including pronoun resolution for questions such as "Did she reply?" Evaluate these separately from exact and semantic retrieval.

Stored embeddings are eventually consistent. Profile extraction or embedding-provider failures must not block existing agent execution.