# Plan: Bounded Execution Agent Retrieval (MVP)

## Goal

The Interaction Agent currently receives the full Execution Agent roster on every request. Replace that unbounded prompt content with at most five relevant agents, so the Interaction Agent can reuse persistent agents without requiring the full roster in context.

The MVP combines exact matching on agent names with semantic retrieval over short agent profiles. Keep the roster authoritative, preserve existing routing behavior when retrieval is unavailable, and make incomplete retrieval explicit so it cannot be mistaken for proof that no matching agent exists.

## MVP Scope

- Candidate limit: **5 agents** per user turn.
- Exact matching: normalized whole-name matches against the authoritative roster. Do not add NER, fuzzy matching, alias extraction, or configurable field registries.
- Semantic matching: embed a bounded profile consisting of the agent name and a capped excerpt of its original delegated instruction. Do not use an LLM to extract structured profiles or embed full logs, responses, or tool payloads.
- Profile lifecycle: create or update a profile when an agent is created; lazily backfill an existing agent from its recorded instruction when first needed. Keep only the profile data needed for retrieval.
- Existing-agent migration limitation: agents already in the roster before this feature have no stored vector initially. Lazy backfill makes their name and instruction available for exact-name matching and degraded recent-agent fallback, but does not embed them. They become semantically searchable after a later instruction updates their profile and generates a vector. Do not add a bulk backfill job in the MVP.
- Embeddings: use one configured provider and model, selected before implementation. Do not build a provider abstraction or vector database. Scan compatible vectors in-process for the current roster size.
- Degraded mode: if query embeddings or profile vectors are unavailable, include exact matches and recently active agents when available, and mark the lookup incomplete. The Interaction Agent must not claim there are no matching agents based on an incomplete lookup.
- Privacy: query text and profile instruction text sent to the embedding provider leave the machine. Document this, the selected model, configuration, and relevant provider retention/cost considerations before enabling the feature.

### Out of scope

LLM-based profile extraction, full-history backfill jobs, `search_agents` tool, fuzzy name matching, NER, pronoun resolution, recency-based ranking beyond degraded fallback, vector databases/indexes, configurable matching fields, model-backed routing A/B tests, and support for multiple embedding providers.

## Stacked PR Workflow

Keep the implementation reviewable as a stack: each feature PR targets its immediate predecessor, and an aggregate PR targets `main` and contains the complete change.

```text
main
  -> feature/00-plan
  -> feature/01-profiles
  -> feature/02-embeddings
  -> feature/03-retrieval
  -> feature/04-prompt
  -> feature/05-evaluation
  -> feature/final
```

- Keep each PR focused on its branch’s scope and include its targeted tests.
- Keep the aggregate branch synchronized with the latest stack tip so its PR shows the complete feature.
- Leave PRs open and do not merge or close them before the presentation without approval. Do not rewrite shared history or force-push without authorization.
- If an earlier branch changes, propagate the change through descendants and verify each PR still has the intended base and diff.

## Feature PRs

### 1. Agent Profiles

Add a small retrieval-profile store using the repository’s existing persistence conventions. Key profiles by the existing agent name; the roster remains authoritative. Store the name, a bounded excerpt of the original delegated instruction, and last-active time if an existing source provides it. Use one fixed excerpt limit and deterministic truncation; do not add a summarization model.

- Create/update the profile when an Execution Agent is created, without blocking its execution.
- Lazily populate profiles for agents already in the roster before this feature from their latest recorded instruction when lookup encounters a missing profile. These lazily backfilled profiles have no embedding at first: exact name matching and degraded recent-agent fallback remain available, and semantic retrieval becomes available after the agent next receives an instruction. Do not introduce a separate bulk backfill service.
- Clear derived profiles together with the existing chat-history state.
- **Tests:** profile persistence and identity, lazy backfill, missing source data, clearing, and profile-update failure not blocking execution.

### 2. Embedding Client

Add one async embeddings client using the repository’s existing HTTP and configuration patterns. Use OpenRouter's embeddings endpoint with `openai/text-embedding-3-small` by default, configurable via `EXECUTION_AGENT_EMBEDDING_MODEL` and the existing `OPENROUTER_API_KEY`.

- Embed profile text during profile creation/update and embed only the latest user message for a lookup.
- Use a short bounded timeout (target: 2 seconds) and no retry on the user-turn critical path. On failure, return an explicit unavailable result for degraded retrieval rather than blocking agent execution.
- Persist profile vectors with the model identifier and dimensions. Re-embed profiles when their source text or configured model changes. Never compare vectors from different models or dimensions.
- Document that query and profile text are sent to OpenRouter, plus configuration and current-provider pricing/retention considerations.
- **Tests:** valid and malformed responses, timeout/provider errors, model and dimension compatibility, profile re-embedding, and non-blocking profile updates.

### 3. Hybrid Retrieval

Implement a focused retrieval service that scans the current roster and compatible profile vectors.

- Return up to five candidates. Put exact normalized whole-agent-name matches first, then rank remaining candidates by cosine similarity with deterministic tie-breaking.
- Do not add an unvalidated similarity threshold. Retrieval ranks candidates; the Interaction Agent decides whether an agent is appropriate.
- If embedding lookup is unavailable, return exact matches and recently active agents when available, with an explicit incomplete/degraded status. If no such fallback data is available, return an empty but explicitly incomplete result.
- Do not represent an incomplete result as a definitive no-match.
- **Tests:** exact and semantic matches, deterministic ranking and limit, incompatible/missing vectors, provider failure, and degraded fallback status.

### 4. Bounded Interaction Prompt

Use the retrieval service to replace full-roster injection on user turns with at most five candidates and the lookup’s completeness status.

- Preserve originating agent names explicitly on agent-result turns, including multi-agent and scheduled results; do not substitute a roster lookup for known origins.
- Watcher notifications without an originating Execution Agent must not receive the full roster.
- Update the Interaction Agent instructions to reuse a relevant candidate, preserve its existing name, and avoid asserting that no agent exists when retrieval is incomplete. Keep creation and clarification decisions with the Interaction Agent.
- Do not add a separate `search_agents` tool in this MVP.
- **Tests:** bounded user-turn candidates, incomplete lookup guidance, agent-result origin names, multi-agent/scheduled results, watcher notifications, and existing execution/trigger behavior.

### 5. Evaluation

Create a reproducible synthetic benchmark whose labels are authored independently of the retrieval implementation. Keep the dataset inspectable and include its scope and limitations in the evaluation output.

- Store synthetic agent IDs, names, original delegated instructions, user queries, expected action, and relevant agent IDs in `server/evaluation/execution_agent_retrieval_benchmark.json`.
- Cover exact-name reuse, paraphrases, multiple relevant agents, genuinely new tasks, ambiguous requests, and follow-ups whose answers depend on execution outputs.
- Score recall@5, mean reciprocal rank, and mean precision@5 for labeled reuse cases. Compare hybrid retrieval against exact-name-only and keyword-overlap baselines; report misses and action/category counts.
- Treat ambiguous/new-task cases as routing evaluation inputs, not evidence that a ranked retrieval list can itself determine whether to create an agent. Exclude execution-output-dependent follow-ups from retrieval scores because this MVP indexes names and original instructions, not later outputs; report their count explicitly.
- This benchmark measures retrieval, not whether the Interaction Agent makes the right reuse/create/clarify decision from candidates. Do not claim end-to-end routing accuracy from these metrics; assess model decisions separately with controlled prompts and fake execution tools before making that claim.
- Keep the default offline run deterministic and provider-free. Its feature-hash vectors are a regression proxy only, not a semantic embedding model or evidence of real-world recall.
- Provide an opt-in real-embedding evaluation using only synthetic benchmark records and queries. Check in a compressed embedding cache keyed by model and text hash so the reported results can be reproduced without provider access or repeat charges; the cache contains only synthetic-data vectors, never credentials or raw text. Missing vectors may be fetched with the configured provider and added to the cache. Record cache/provider-fetch counts, model, and a predeclared **90% recall@5 target** in the output artifact. Do not make chat-model calls or invoke real execution tools.
- Check in the current reproducible score output at `server/evaluation/execution_agent_retrieval_results.json`; rerunning with `--live-embeddings --results-file <path>` should use cached vectors when available and refresh the requested result file.
- Measure serialized prompt characters using the same prompt renderer and same profile fields for both retrieved candidates and a full-profile-roster baseline. Also report the names-only roster baseline to make comparison to the prior prompt payload explicit. Character counts are a size proxy, not tokenizer-accurate token savings.
- **Tests:** dataset ID/label integrity, reproducible hash-vector results, cached live-vector results without provider calls, cache format/model/text-hash behavior, multi-label cases, output-dependent exclusions, candidate cap, baseline metrics, and prompt-size comparison.

## Integration Points

Confirm these against the current implementation as each PR is built:

- `server/services/execution/roster.py`: authoritative agent names.
- `server/services/execution/log_store.py`: source for lazy profile backfill; do not index or embed raw logs.
- `server/agents/interaction_agent/agent.py` and `runtime.py`: candidate injection and explicit degraded status.
- `server/agents/execution_agent/batch_manager.py` and `server/services/trigger_scheduler.py`: profile updates and structured origin names.
- `server/config.py` and the existing provider client: embedding configuration and HTTP conventions.
- `server/routes/chat.py`: clear derived profile data with existing history state.
- `README.md`: provider configuration and privacy note.

## MVP Completion Criteria

- User-turn prompts contain no more than five retrieved candidates, not the full roster.
- Exact and semantic retrieval work with deterministic tests and compatible vectors only.
- Existing agents can be lazily backfilled without a separate migration/backfill operation.
- Agents already in the roster before this feature can be used by exact-name or degraded fallback retrieval immediately after lazy profile backfill; semantic retrieval for them is deferred until their next profile update. This is an intentional MVP limitation, not a release blocker.
- Provider/profile failures do not block Execution Agent work and cannot silently become a definitive no-match.
- Known agent origins remain available on agent-result turns; watcher notifications do not receive the roster.
- Offline evaluation reports recall@5 and prompt-size reduction against the full-roster baseline.

## Follow-up

Consider aliases, recency ranking, pronoun resolution, a dedicated on-demand search tool, broader datasets, or model-backed routing evaluations only after the MVP demonstrates a retrieval gap that warrants them.
