# OpenPoke 🌴

OpenPoke is a simplified, open-source take on [Interaction Company’s](https://interaction.co/about) [Poke](https://poke.com/) assistant—built to show how a multi-agent orchestration stack can feel genuinely useful. It keeps the handful of things Poke is great at (email triage, reminders, and persistent agents) while staying easy to spin up locally.

- Multi-agent FastAPI backend that mirrors Poke's interaction/execution split, powered by [OpenRouter](https://openrouter.ai/).
- Gmail tooling via [Composio](https://composio.dev/) for drafting/replying/forwarding without leaving chat.
- Trigger scheduler and background watchers for reminders and "important email" alerts.
- Next.js web UI that proxies everything through the shared `.env`, so plugging in API keys is the only setup.

## Requirements
- Python 3.10+
- Node.js 18+
- npm 9+

## Quickstart
1. **Clone and enter the repo.**
   ```bash
   git clone https://github.com/shlokkhemani/OpenPoke
   cd OpenPoke
   ```
2. **Create a shared env file.** Copy the template and open it in your editor:
   ```bash
   cp .env.example .env
   ```
3. **Get your API keys and add them to `.env`:**
   
   **OpenRouter (Required)**
   - Create an account at [openrouter.ai](https://openrouter.ai/)
   - Generate an API key
   - Replace `your_openrouter_api_key_here` with your actual key in `.env`
   
   **Composio (Required for Gmail)**
   - Sign in at [composio.dev](https://composio.dev/)
   - Create an API key
   - Set up Gmail integration and get your auth config ID
   - Replace `your_composio_api_key_here` and `your_gmail_auth_config_id_here` in `.env`
4. **(Required) Create and activate a Python 3.10+ virtualenv:**
   ```bash
   # Ensure you're using Python 3.10+
   python3.10 -m venv .venv
   source .venv/bin/activate
   
   # Verify Python version (should show 3.10+)
   python --version
   ```
   On Windows (PowerShell):
   ```powershell
   # Use Python 3.10+ (adjust path as needed)
   python3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # Verify Python version
   python --version
   ```

5. **Install backend dependencies:**
   ```bash
   pip install -r server/requirements.txt
   ```
6. **Install frontend dependencies:**
   ```bash
   npm install --prefix web
   ```
7. **Start the FastAPI server:**
   ```bash
   python -m server.server --reload
   ```
8. **Start the Next.js app (new terminal):**
   ```bash
   npm run dev --prefix web
   ```
9. **Connect Gmail for email workflows.** With both services running, open [http://localhost:3000](http://localhost:3000), head to *Settings → Gmail*, and complete the Composio OAuth flow. This step is required for email drafting, replies, and the important-email monitor.

The web app proxies API calls to the Python server using the values in `.env`, so keeping both processes running is required for end-to-end flows.

### Execution Agent retrieval

Execution Agent retrieval uses OpenRouter's embeddings endpoint with
`openai/text-embedding-3-small` by default. Override the model with
`EXECUTION_AGENT_EMBEDDING_MODEL` if needed. The existing `OPENROUTER_API_KEY`
is used; no additional provider credential is required.

Retrieval sends the latest user message and each bounded Execution Agent profile
(agent name and delegated-instruction excerpt) to OpenRouter. Do not enable this
feature if that text should remain local. Provider availability, applicable data
retention, and current pricing depend on OpenRouter and its selected provider;
check their current policies and model pricing before use.

The synthetic benchmark contains 30 hand-authored agent profiles and 33 labeled
requests. It includes exact-name and paraphrase reuse, multi-agent requests,
ambiguous/new tasks, and output-dependent follow-ups (which are reported but
excluded from retrieval scoring because the current index does not include
execution outputs).

Run the deterministic offline regression evaluation with:

```bash
python -m server.evaluation.execution_agent_retrieval
```

The offline run uses deterministic feature-hash vectors; it makes no provider
calls and is not a measure of semantic embedding quality. It reports recall@5,
MRR, and precision@5 for hybrid retrieval and exact-name/keyword baselines, plus
prompt-character comparisons. These are retrieval metrics, not measurements of
whether the Interaction Agent chooses the right reuse/create/clarify action.

To evaluate the configured embedding model against the same synthetic-only
benchmark, run:

```bash
python -m server.evaluation.execution_agent_retrieval --live-embeddings \
  --results-file server/evaluation/execution_agent_retrieval_results.json
```

The checked-in compressed cache contains the vectors for the current synthetic
dataset, generated with `openai/text-embedding-3-small`; the command therefore
reproduces the recorded result without a key or provider calls. If the dataset
or model changes, uncached texts are sent to OpenRouter and may incur cost.
Updated cache vectors can be submitted with the results. The cache contains
only vectors derived from synthetic data, not API keys or raw text. The command
makes no chat-model calls and invokes no execution tools. The precomputed
results are in `server/evaluation/execution_agent_retrieval_results.json`.

The recorded live-embedding result achieved 100% recall@5 (30/30 relevant
agent labels), MRR 1.0, versus 90% recall@5 and MRR 0.84 for keyword matching.
This is a small synthetic benchmark result, not a production-quality claim.

## Project Layout
- `server/` – FastAPI application and agents
- `web/` – Next.js app
- `server/data/` – runtime data (ignored by git)

## License
MIT — see [LICENSE](LICENSE).
