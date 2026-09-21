# m45

Early implementation of a personal assistant with continuity (planned product
name: Pleiades). The project is currently version **0.1.0**.

The current increment provides PostgreSQL-backed durable source-message history,
a bounded reader for recent context from other conversations, and a deterministic
LangGraph smoke agent served through the local Agent Server. Production model,
memory-injection, source-capture, and user-interface wiring are still in progress.

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

Start the credential-free smoke agent:

```sh
./scripts/dev.sh
```

The launcher serves the `m45` graph from `langgraph.json` at
`http://127.0.0.1:2024`. It disables LangGraph CLI analytics and LangSmith
tracing. The development server supplies thread checkpointing and saves its
disposable local state under `.langgraph_api/`; application graph code does not
create a checkpointer.

The smoke agent always returns `m45 smoke response`. It exists to validate graph
loading, message identity, repeated invocation, and server-owned thread state
without model credentials. It is not the product model path.

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

`create_smoke_agent` builds a real compiled LangGraph agent with LangChain's
credential-free fake chat model. The module-level `graph` export is loaded by
Agent Server. Automated tests cover repeated invocations and distinct generated
assistant-message IDs; a manual server smoke test has also confirmed that two
runs on one thread produce a checkpointed four-message history.

Prompt injection has not been implemented yet. The selected window will later be
supplied transiently through native agent middleware rather than appended to
persisted thread messages.

There is no production model integration, source-history capture in the graph,
personal-context middleware, frontend integration, Discord adapter, importer, or
long-term-memory mechanism yet.

## Repository map

| Path | Responsibility | Effect of removing it |
| --- | --- | --- |
| `src/m45/` | Application package, typed configuration, database setup, source-history persistence, recent-context selection, and the LangGraph smoke agent. | Application imports, persistence, context selection, and graph loading fail. |
| `migrations/` | Alembic environment and versioned PostgreSQL schema changes. | Fresh and existing databases cannot be brought to the expected schema. |
| `tests/` | Tests for source identity, recent-context selection, and repeatable smoke-agent behavior. | Persistence, context-selection, and graph-runtime contracts lose automated verification. |
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
