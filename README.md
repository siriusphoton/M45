# m45

Scaffold for a personal assistant with continuity (planned product name: Pleiades).
Project version: **0.1.0**. There is no application, data model, or product feature yet.

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (CI uses
0.11.2) and Docker with Compose. Python 3.12 is selected by `.python-version`;
uv can install it if needed.

```sh
uv sync --locked
cp -n .env.example .env
docker compose up -d --wait postgres
./scripts/check.sh
```

PostgreSQL 17 listens on `127.0.0.1:5432`. The example credentials are for local
development. If that port is occupied by another project, change `POSTGRES_PORT`
in `.env` and update the port in `DATABASE_URL` to match.

```sh
docker compose exec postgres psql -U m45 -d m45
docker compose down
```

The named `postgres_data` volume survives `docker compose down`. Adding `--volumes`
deletes it. PostgreSQL initialization variables apply only to a fresh volume;
changing `.env` does not change credentials in an existing database. There are no
application tables or migrations in this scaffold.

## Checks

`./scripts/check.sh` synchronizes the locked environment, validates Compose without
starting containers, checks formatting and lint with Ruff, checks types with
Pyright, and invokes pytest. GitHub Actions runs this same command. A running
database, `.env`, and model credentials are not required for checks.

There are deliberately no placeholder tests. While the repository has no Python
code, Ruff/Pyright have no code to check and pytest collects no tests. The script
explicitly accepts only pytest's "no tests collected" exit code in addition to
success; all other failures propagate. Remove that scaffold exception when the
first implementation and tests are added.

Dependency changes belong in `pyproject.toml` and `uv.lock` together. Use `uv lock`
after editing dependencies; checks use `--locked` so they cannot silently rewrite
the dependency resolution. Tool settings live in `pyproject.toml`.

## Implementation boundary

SQLAlchemy, Psycopg, Alembic, and Pydantic Settings are installed and locked.
Application settings validation, standard-library console logging, SQLAlchemy
engines/models, and Alembic's migration environment need Python code and are
intentionally not implemented yet. `DATABASE_URL` and `LOG_LEVEL` document the
initial configuration inputs; no runtime consumes them yet.

LangGraph configuration and dependencies will be added with an actual graph
entry point. There is no development server, frontend, model integration, or
background process in this scaffold. Implementation begins after scaffold review.

## Repository map

| Path | Current responsibility and dependency | Effect of removing it |
| --- | --- | --- |
| `scripts/check.sh` | Canonical checks, used locally and by CI. | Removes the shared verification entry point. |
| `.github/workflows/check.yml` | Runs the check script on pushes and pull requests. | Removes automated CI checks. |
| `pyproject.toml` | Project metadata, dependency declarations, and tool settings. | uv and checks lose their project configuration. |
| `uv.lock` | Exact dependency resolution used by `uv sync --locked`. | Locked setup and checks fail until regenerated. |
| `.python-version` | Selects Python 3.12 for uv locally and in CI. | Interpreter selection falls back to the project constraint and environment. |
| `compose.yaml` | Local PostgreSQL, health check, localhost port, and named storage. | Compose setup and configuration checks fail; the existing volume is not deleted. |
| `.env.example` | Local configuration template, also used by Compose validation in checks. | Documented setup and Compose validation fail. |
| `.gitignore` | Excludes secrets, environments, generated artifacts, and existing local notes. | Those files can appear as untracked and be accidentally committed. |
| `README.md` | Setup, scope, and this walkthrough; referenced by project metadata. | Removes onboarding and leaves the metadata reference unresolved. |

Existing `AGENTS.md`, `PROJECT.md`, `ARCHITECTURE.md`, and
`docs/archive/Thoughts.md` are local project context, kept untracked by the existing
repository policy. They have no runtime dependencies; removing them loses working
rules, current scope, architecture invariants, and historical vision respectively.
The abandoned live WhatsApp integration in the historical notes is not current scope.

Generated `.venv/`, `.ruff_cache/`, and `.pytest_cache/` are disposable local tooling
state. There are no empty `src/`, `tests/`, or migration directories. Git's `.git/`
directory holds repository history and is unrelated to application structure.
