# m45

Early implementation of a personal assistant with continuity (planned product
name: Pleiades). The project is currently version **0.3.0**.

The current increment provides PostgreSQL-backed durable source-message history,
a configurable reader for recent context from other conversations, and a
LangGraph agent served through the local Agent Server. Ollama Cloud has been
verified through Agent Chat UI, including recall across two separate threads and
the corresponding durable source rows. A single-user Discord DM adapter now calls
the same private Agent Server and has been verified with complete live turns.
Recall has also been verified in both directions between Discord and a separate
Agent Chat UI thread. Native agent middleware captures each completed turn and
supplies recent cross-conversation context without adding it to checkpointed
messages. The same model-call boundary supplies the current date and time in
India and capture times for active messages without persisting them in agent
state. The agent can use Ollama Web Search for current external information while
retaining tool exchanges in checkpoint state rather than durable source history.
It removes an echoed leading IST timestamp from final assistant replies before
they are saved; the raw prefix may briefly appear while Agent Chat UI streams.

A manual daily writer generates and saves versioned narratives from copied source
history in an isolated trial database. Revisions retain their exact input references
and can be inspected independently of live conversations. The runner commits the
revision and lexical passages together, then attempts local embedding with bounded
retries. An embedding failure leaves the saved narrative and lexical search intact.

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Docker
with Compose. Python 3.12 is selected by `.python-version`.

```sh
uv sync --locked
cp -n .env.example .env
docker compose up -d --wait postgres
docker compose exec -T postgres psql -X -U m45 -d m45 \
  -v ON_ERROR_STOP=1 \
  -c 'CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;'
uv run alembic upgrade head
```

The development database listens on `127.0.0.1:5432` by default. If that port is
occupied, change `POSTGRES_PORT` and the port in `DATABASE_URL` together in
`.env`.

To inspect or stop the development database:

```sh
docker compose exec postgres psql -U m45 -d m45
docker compose down
```

The `postgres_data` volume survives `docker compose down`. Adding `--volumes`
deletes it. PostgreSQL initialization variables only affect a fresh volume.

The retrieval database setup uses `pgvector/pgvector:0.8.7-pg17-trixie` for both
PostgreSQL services. The image supplies pgvector's files; the administrator command
above enables its SQL types and operators in the development database. Repeat that
provisioning for each new database before applying retrieval migrations. Application
tables remain Alembic-managed; installing the extension is an administrative setup
step rather than an application-startup action. `scripts/check.sh` performs the same
setup for its disposable test database. Before switching an existing container to
this image, back up its databases and retain the existing `postgres_data` volume.

Retrieval preparation has separate `EMBEDDING_MODEL` and `EMBEDDING_BASE_URL`
settings for a local Ollama embedding service. These do not select the assistant's
chat model. The manual daily writer now calls the embedding workflow; its database
and retry behavior is verified with synthetic vectors. The developer's real trial
also stored all six passages of the 1 October log's second revision as
1024-dimensional `qwen3-embedding:0.6b` vectors. Standalone semantic search is now
implemented, and the first trial query found the relevant phone-purchase passage.
Hybrid search combines lexical and semantic rankings by passage ID. The isolated
trial also retrieves original conversation passages alongside daily-log passages.
Conversational retrieval is not wired yet, so ordinary chat does not require that
local service.

### Development source-history maintenance

During local dogfooding, inspect or deliberately correct durable source rows with
the PostgreSQL client already available in the container:

```sh
docker compose exec postgres psql -U m45 -d m45
```

Inside `psql`, list the newest messages with their database IDs:

```sql
\x auto
SELECT id, source, conversation_id, message_id, role, captured_at, content
FROM source_messages
ORDER BY id DESC
LIMIT 50;
```

Make local edits inside an explicit transaction so they can be inspected before
they are committed:

```sql
BEGIN;

UPDATE source_messages
SET content = $message$replacement text$message$
WHERE id = 42
RETURNING id, source, conversation_id, message_id, role, content;

COMMIT;
```

Delete selected rows using the same pattern:

```sql
BEGIN;

DELETE FROM source_messages
WHERE id IN (42, 43)
RETURNING id, source, conversation_id, message_id, role, content;

COMMIT;
```

Use `ROLLBACK;` instead of `COMMIT;` if the returned rows are not the intended
ones. This developer-only maintenance changes source history but does not rewrite
LangGraph checkpoints. A later re-execution may recreate a deleted row or reject
an edited row whose original source identity is submitted again.

After applying the daily-log migration, saved revisions reference source rows.
Referenced rows cannot be deleted without handling those dependencies, and
`TRUNCATE source_messages` alone is no longer valid because another table references
it. A full reset requires an explicit plan for dependent records; do not add
`CASCADE` to bypass this. Direct content edits also do not revise existing logs.

### Isolated historical memory trials

Historical writer experiments use `m45_trial`, a separate database in the existing
`postgres` service. It persists in the same local volume and is independent of the
ephemeral `m45_test` database recreated by `scripts/check.sh`. Its dedicated login
owns the trial database and receives no live-table privileges. The password below
is a local-development value, like the credentials in `.env.example`.

With the development PostgreSQL service running, create the role/database once:

```sh
docker compose exec -T postgres psql -X -U m45 -d postgres \
  -v ON_ERROR_STOP=1 <<'SQL'
CREATE ROLE m45_trial LOGIN PASSWORD 'm45_trial'
    NOSUPERUSER NOCREATEDB NOCREATEROLE;
CREATE DATABASE m45_trial OWNER m45_trial;
SQL

docker compose exec -T postgres psql -X -U m45 -d m45_trial \
  -v ON_ERROR_STOP=1 \
  -c 'CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;'
```

The extension command uses the administrator login `m45`; the trial login remains
restricted and does not need permission to install extensions.

Apply the same application schema using the trial login. These command-scoped
overrides leave the live `DATABASE_URL` in `.env` unchanged:

```sh
DATABASE_URL=postgresql+psycopg://m45_trial:m45_trial@127.0.0.1:5432/m45_trial \
  uv run --no-sync alembic upgrade head

DATABASE_URL=postgresql+psycopg://m45_trial:m45_trial@127.0.0.1:5432/m45_trial \
  uv run --no-sync alembic check
```

Verify that the trial role has no write privileges on live application tables.
Every `can_write` result must be `f` before historical trials:

```sh
docker compose exec -T postgres psql -X -U m45 -d m45 \
  -v ON_ERROR_STOP=1 <<'SQL'
SELECT c.relname AS live_table,
       has_table_privilege(
           'm45_trial', c.oid, 'INSERT, UPDATE, DELETE, TRUNCATE'
       ) AS can_write
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'r'
ORDER BY c.relname;
SQL
```

The fresh trial database is empty after migration. Copy explicitly selected source
rows into it before running a historical trial, preserving their IDs, identities,
content and capture timestamps. Source copying is currently a manual preparation
step; the runner does not read from the live database. Keep private transcripts out
of committed fixtures. Trial outputs do not enter live context or advance live coverage.

With the selected day's source rows present, generate and persist a trial revision:

```sh
DATABASE_URL=postgresql+psycopg://m45_trial:m45_trial@127.0.0.1:5432/m45_trial \
LANGSMITH_TRACING=false \
uv run --no-sync python scripts/trial_daily_log.py \
  2026-10-01 --as-of 2026-10-02T05:00:00+05:30
```

`processing_day` selects the 05:00 IST processing window. `--as-of` is an exclusive,
timezone-aware review cutoff; it limits source evidence supplied to the writer.
The runner requires the `m45_trial` database/login, uses the configured model,
and loads the previous daily log before generating its replacement. It commits
the narrative, exact input references and replacement lexical passages together,
then runs the embedding graph for that saved revision ID. Local Ollama must be
running with the configured embedding model installed. Transient embedding
failures receive up to three total attempts; exhausted attempts report the saved
revision ID while retaining its narrative and lexical passages.

Each invocation that reaches the save commit creates another revision, even with
unchanged evidence or a subsequent embedding failure. Recover embedding through
`embed_saved_daily_log` with the reported `daily_log_revision_id`; rerunning the
writer generates another narrative. Recovery does not call passage replacement,
and completed vectors are skipped. There is no transparent save-retry protocol.
Begin an evolving-day trial without a later saved log for that day, so future
information cannot enter through the previous narrative.

`processing_day` identifies the daily account; `revision_number` counts its saved
versions. Each version also has a database `id`, referenced elsewhere as
`daily_log_revision_id`. Earlier narratives remain stored; search passages and
their vectors are replaced when a newer revision is indexed. Apply migrations
before using the renamed fields below.

Inspect saved revisions and their source counts:

```sh
docker compose exec -T postgres psql -X -U m45_trial -d m45_trial \
  -v ON_ERROR_STOP=1 <<'SQL'
SELECT r.processing_day, r.revision_number, count(s.source_message_id) AS sources
FROM daily_log_revisions AS r
LEFT JOIN daily_log_revision_sources AS s ON s.daily_log_revision_id = r.id
GROUP BY r.id
ORDER BY r.processing_day, r.revision_number;
SQL
```

Inspect persisted indexing for the latest revision of the selected trial day:

```sh
docker compose exec -T postgres psql -X -U m45_trial -d m45_trial \
  -v ON_ERROR_STOP=1 <<'SQL'
SELECT r.id AS daily_log_revision_id,
       r.processing_day,
       r.revision_number,
       count(p.id) AS passages,
       count(p.embedding) AS embedded,
       min(vector_dims(p.embedding)) AS dimensions,
       min(p.embedding_config ->> 'model') AS embedding_model
FROM daily_log_revisions AS r
LEFT JOIN search_passages AS p ON p.daily_log_revision_id = r.id
WHERE r.processing_day = DATE '2026-10-01'
GROUP BY r.id
ORDER BY r.revision_number DESC
LIMIT 1;
SQL
```

Successful embedding has `passages = embedded > 0`, dimensions `1024` and the
configured embedding model. This verifies storage, not semantic search quality.
Semantic and hybrid query helpers are implemented. `create_memory_search_tool`
exposes hybrid search with `all`, `daily_log` or `source_messages` scope and returns
up to five passages with references. It is intended for the dedicated retriever;
live conversational retrieval integration remains pending.

Index the eligible source conversations already copied into the trial database:

```sh
DATABASE_URL=postgresql+psycopg://m45_trial:m45_trial@127.0.0.1:5432/m45_trial \
LANGSMITH_TRACING=false \
uv run --no-sync python scripts/index_trial_sources.py
```

This runner requires the trial database/login and local Ollama. It refreshes each
conversation's derived passages, retaining unchanged vectors, then embeds only
missing vectors with up to three attempts for transient failures. A failed embedding
step leaves lexical passages available; rerunning recovers missing vectors without
re-embedding unchanged passages. Zero missing vectors filled is expected on an
unchanged rerun. The runner neither copies source messages nor enables live indexing.

Search hits retain references for explicit reading. The canonical readers return
one original message or current daily log in bounded pages, with continuation
offsets; they do not automatically expand every hit into its entire conversation.
The `read_source_message` and `read_daily_log` tools expose these readers to the
dedicated retriever and require no embedding service.
The `browse_daily_logs` tool discovers latest logs by processing-day range;
`browse_source_conversation` returns neighboring messages around a source-message
ID within its conversation. Both return up to ten 300-character previews with
continuation information. Full text uses the read tools. These capabilities are
implemented but are not yet connected to the conversational agent.

Run the standalone retriever against the same isolated trial corpus:

```sh
DATABASE_URL=postgresql+psycopg://m45_trial:m45_trial@127.0.0.1:5432/m45_trial \
LANGSMITH_TRACING=false \
uv run --no-sync python scripts/trial_retrieval.py \
  "What phone was I considering buying?"
```

The runner requires the trial database/login and the configured chat provider.
Search also needs local Ollama embeddings. It supplies current daily orientation
and starts a fresh conversation containing only the question; it does not load
Discord or Agent Chat UI checkpoints. It prints selected context notes with
canonical references and material limitations, without saving messages, logs or
checkpoints. The first real trial recovered the expected phone-purchase context;
this does not establish general retrieval accuracy.

Retrieval has six model calls, ten total tool calls including structured formatting,
and a 32,000-token estimated envelope with 4,000 reserved for output. The reserve is
budget accounting, not a provider-enforced output cap. Oversized input fails
explicitly; known call-limit exhaustion returns an incomplete-context result, and
failed tools remain visible in limitations. Live conversational integration is
the next increment; this command does not activate it.

## Local Agent Server

Set `MODEL_PROVIDER` and its corresponding credential in `.env`, then start the
agent:

```sh
./scripts/dev.sh
```

The launcher serves the `m45` graph from `langgraph.json` at
`http://127.0.0.1:2024`. It disables LangGraph CLI analytics and LangSmith
tracing. The development server supplies thread checkpointing and saves its
disposable local state under `.langgraph_api/`; application graph code does not
create a checkpointer.

The runtime supports `google_genai` and `ollama` through LangChain's native chat
model integrations. Each provider keeps its own configured model name, so changing
`MODEL_PROVIDER` selects the complete provider profile. Ollama Cloud has been
verified with two turns in Agent Chat UI, including server-owned thread history.
Both provider configurations request a sampling temperature of `0.0`; this does
not guarantee identical outputs across runs.
The Google Gemini API path is configured and type-checked but has not yet completed
a live request because the selected free-tier model was at capacity.

Web search uses the Ollama API independently of the selected chat-model provider,
so `OLLAMA_API_KEY` is required even when `MODEL_PROVIDER=google_genai`. Search
queries should minimize personal information. Retrieved content is treated as
untrusted evidence, and the agent is instructed to cite the source URLs it uses.
Each run may call search at most three times, and a failed call is retried once
before the model receives a sanitized failure message.

Automated tests use a deterministic fake chat model directly; it is not selectable
as an application provider.

## Local Discord adapter

Create a Discord bot application, add it to a private server owned by the same
Discord account, and keep all privileged Gateway intents and guild permissions
disabled. Set the bot token and the allowed human user's Discord ID in `.env`:

```dotenv
DISCORD_BOT_TOKEN=replace-with-the-bot-token
DISCORD_ALLOWED_USER_ID=123456789012345678
```

With Agent Server running in one terminal, start the outbound Discord Gateway
client in another:

```sh
uv run m45-discord
```

The adapter accepts text from that user's one-to-one DMs only. It ignores guild,
group, bot, and unauthorized-user messages; attachments and stickers receive a
text-only notice. Discord channel and message IDs provide durable source identity,
while the channel ID deterministically maps to the LangGraph thread UUID. Agent
Server remains on loopback and does not need a public inbound endpoint.

## Checks

Run the canonical check command:

```sh
./scripts/check.sh
```

It:

1. synchronizes the locked Python environment;
2. validates the Compose configuration;
3. runs Ruff formatting and lint checks;
4. runs Pyright;
5. creates a fresh ephemeral PostgreSQL test database;
6. applies all Alembic migrations;
7. checks that migrations match SQLAlchemy metadata;
8. runs pytest;
9. removes the test database container.

The check does not use the development database or require `.env`. GitHub Actions
runs the same command.

Dependency changes belong in `pyproject.toml` and `uv.lock` together. Checks use
`uv sync --locked`, so dependency resolution cannot change silently.

## Current implementation

`source_messages` stores durable text messages using the source, conversation ID,
and source message ID as its unique identity.

Capturing an exact duplicate returns the existing row. Reusing an identity with a
different role or content raises an explicit conflict. Distinct message IDs remain
distinct even when their text is identical.

Transaction ownership belongs to the caller. Source capture executes within the
provided SQLAlchemy session but does not commit it.

`daily_log.py` reads a processing day's eligible source messages across conversations
using a 05:00 IST boundary, presents their provenance, and generates a replacement
narrative using a direct model call. The writer logs reported circumstances;
interpretation and cross-day reconciliation belong to later consolidation.
Its current presentation uses Sleep and rest, Meals, and Tasks and plans sections,
followed by relevant descriptive topics and optional material Unresolved details.
Trials showed successful task/meal status updates and retained earlier context,
alongside attribution errors, omitted corrections and repeated details. The logs
remain derived accounts whose original source evidence is retained for inspection.
Daily-log revisions retain their exact input source IDs in a separate linking
table. The load/save functions preserve earlier versions, reject stale saves and
flush narrative plus references in the caller's transaction. They are not yet
connected to a scheduler or conversational context injection.

`load_recent_personal_context` selects whole user and assistant messages from
explicitly eligible sources, excludes the current source conversation, and
applies configurable message and character limits. It returns the selected
window in chronological order. Setting either limit to `0` disables that limit.

`SourceHistoryMiddleware` commits the submitted user message before model
invocation and the completed assistant message before the graph returns. It uses
the LangGraph thread ID as the source conversation identity and requires stable
message IDs. Exact re-execution remains safe through the source-history
idempotency contract. For each model call, it also labels source-backed thread
messages with their IST capture times in the model request only; neither the
checkpointed messages nor the durable source content changes.

`PersonalContextMiddleware` loads the configured window at the model-call boundary,
formats it as a role-labelled transcript with source, synthetic conversation
labels, and capture times in IST, then adds it to that model request's system
message. Capture times do not establish when the described events happened. The
injected transcript is not added to agent state or checkpoint history; raw
source IDs remain database provenance rather than prompt content.

`CurrentTimeMiddleware` computes the current date and time in `Asia/Kolkata` for
each model call and adds it transiently to the system message before personal
context. This value is neither added to agent state nor saved in checkpoints.

`InteractionInstructionsMiddleware` adds the latest standing interaction
guidance from application PostgreSQL to each model call. Revision 0 is the
code default; approved revisions are shared across threads without changing
checkpointed messages. In Agent Chat UI, `update_instructions` proposes a full
replacement when the user requests a lasting behavior change. LangChain pauses
for approval, editing, or rejection before the tool writes a revision with
source-message provenance. Discord receives the active guidance but cannot
propose an update until its adapter supports review and resume.

`StripResponseTimestampMiddleware` removes a model-echoed leading IST timestamp
from a final assistant message before it reaches checkpoint state and source
capture. The streaming UI can show the prefix until that final state update.

`web_search` calls Ollama Web Search and returns up to three title, URL, and
content blocks. LangChain's native tool-call limit and retry middleware bound its
use. Tool-call requests and results remain in LangGraph checkpoint history so a
run can be reconstructed, while `source_messages` stores only the submitted user
message and completed assistant response.

`create_application_agent` builds the real LangGraph agent with the configured
Google GenAI or Ollama chat model. `runtime.py` loads typed settings and exports
the compiled graph that Agent Server imports once. `create_deterministic_test_agent`
provides the credential-free model used by automated tests. Tests cover repeated
invocations, distinct generated assistant-message IDs, and clear failures for a
missing selected-provider credential. The middleware integration test separately
observes the actual model input, saved checkpoint messages, and durable source
rows for one complete turn.

There is no importer or consolidated long-term-memory workflow yet. Agent Chat UI and the
single-user Discord DM adapter are the currently verified interaction surfaces.

## Repository map

| Path | Responsibility | Effect of removing it |
| --- | --- | --- |
| `src/m45/` | Application package, typed configuration, database setup, source-history persistence, daily-log drafting/revision persistence, recent-context selection, provider-backed agent construction, and the Agent Server runtime export. | Application imports, persistence, context selection, model construction, and graph loading fail. |
| `migrations/` | Alembic environment and versioned PostgreSQL schema changes. | Fresh and existing databases cannot be brought to the expected schema. |
| `tests/` | Tests for source identity, daily-log boundaries/revisions, recent-context selection and formatting, middleware boundaries, deterministic agent behavior, and provider credential validation. | Persistence, context-selection, model-input, checkpoint, source-capture, and graph-runtime contracts lose automated verification. |
| `scripts/check.sh` | Canonical local and CI verification workflow. | Local and CI checks no longer share one entry point. |
| `scripts/dev.sh` | Canonical local Agent Server launcher with tracing and CLI analytics disabled. | Developers must reconstruct the correct privacy-preserving server command. |
| `scripts/trial_daily_log.py` | Manual daily writer against copied evidence in the isolated trial database. | The reusable terminal path for generating and inspecting saved trial revisions is lost. |
| `scripts/index_trial_sources.py` | Refreshes copied source-conversation passages and fills missing vectors in the trial database. | The reusable source-indexing trial and recovery command is lost. |
| `scripts/trial_retrieval.py` | Runs read-only, bounded adaptive retrieval against copied trial evidence. | The reusable terminal path for inspecting selected evidence and limitations is lost. |
| `.github/workflows/check.yml` | Runs the canonical checks on pushes and pull requests. | Automated repository checks stop. |
| `langgraph.json` | Declares the Python runtime and exposes the compiled `m45` graph to Agent Server. | The CLI cannot discover or serve the graph. |
| `pyproject.toml` | Project metadata, dependencies, and Python tool configuration. | uv and the configured development tools lose their project definition. |
| `uv.lock` | Exact dependency resolution. | Reproducible locked installation fails until regenerated. |
| `alembic.ini` | Alembic script and logging configuration. | Alembic commands lose their repository configuration. |
| `compose.yaml` | Durable development PostgreSQL and ephemeral test PostgreSQL services. | Local database setup and database-backed checks fail. |
| `.env.example` | Local configuration template, context-window defaults, and canonical test-service values. | Documented setup and Compose checks lose required values. |
| `.gitignore` | Excludes secrets, runtime state, caches, and private data. | Sensitive or generated files can appear as commit candidates. |
| `.python-version` | Selects Python 3.12 for local uv commands. | Interpreter selection falls back to the environment and project constraint. |
| `README.md` | Documents setup, checks, current behavior, and repository ownership. | Onboarding is lost and package builds lose their declared readme. |

`AGENTS.md`, `PROJECT.md`, `ARCHITECTURE.md`, and `docs/` are local project context
excluded by the repository’s existing Git policy.
