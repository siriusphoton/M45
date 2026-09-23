# m45

Early implementation of a personal assistant with continuity (planned product
name: Pleiades). The project is currently version **0.1.0**.

The current increment provides PostgreSQL-backed durable source-message history,
a bounded reader for recent context from other conversations, and a configurable
LangGraph agent served through the local Agent Server. Ollama Cloud has been
verified through Agent Chat UI, including recall across two separate threads and
the corresponding durable source rows. Native agent middleware captures each
completed turn and supplies recent cross-conversation context to the model without
adding that context to checkpointed thread messages.

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Docker
with Compose. Python 3.12 is selected by `.python-version`.

```sh
uv sync --locked
cp -n .env.example .env
docker compose up -d --wait postgres
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
The Google Gemini API path is configured and type-checked but has not yet completed
a live request because the selected free-tier model was at capacity.

Automated tests use a deterministic fake chat model directly; it is not selectable
as an application provider.

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

`load_recent_personal_context` selects whole user and assistant messages from
explicitly eligible sources, excludes the current conversation, and applies
configurable message and character limits. It returns the selected window in
chronological order.

`SourceHistoryMiddleware` commits the submitted user message before model
invocation and the completed assistant message before the graph returns. It uses
the LangGraph thread ID as the source conversation identity and requires stable
message IDs. Exact re-execution remains safe through the source-history
idempotency contract.

`PersonalContextMiddleware` loads the bounded window at the model-call boundary,
formats it as a role-labelled transcript with synthetic conversation labels, and
adds it to that model request's system message. The injected transcript is not
added to agent state or checkpoint history. Raw source IDs and capture timestamps
remain database provenance rather than prompt content.

`create_application_agent` builds the real LangGraph agent with the configured
Google GenAI or Ollama chat model. `runtime.py` loads typed settings and exports
the compiled graph that Agent Server imports once. `create_deterministic_test_agent`
provides the credential-free model used by automated tests. Tests cover repeated
invocations, distinct generated assistant-message IDs, and clear failures for a
missing selected-provider credential. The middleware integration test separately
observes the actual model input, saved checkpoint messages, and durable source
rows for one complete turn.

There is no Discord adapter, importer, or long-term-memory mechanism yet. Agent
Chat UI is the currently verified replaceable frontend.

## Repository map

| Path | Responsibility | Effect of removing it |
| --- | --- | --- |
| `src/m45/` | Application package, typed configuration, database setup, source-history persistence, recent-context selection, provider-backed agent construction, and the Agent Server runtime export. | Application imports, persistence, context selection, model construction, and graph loading fail. |
| `migrations/` | Alembic environment and versioned PostgreSQL schema changes. | Fresh and existing databases cannot be brought to the expected schema. |
| `tests/` | Tests for source identity, recent-context selection and formatting, middleware boundaries, deterministic agent behavior, and provider credential validation. | Persistence, context-selection, model-input, checkpoint, source-capture, and graph-runtime contracts lose automated verification. |
| `scripts/check.sh` | Canonical local and CI verification workflow. | Local and CI checks no longer share one entry point. |
| `scripts/dev.sh` | Canonical local Agent Server launcher with tracing and CLI analytics disabled. | Developers must reconstruct the correct privacy-preserving server command. |
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
