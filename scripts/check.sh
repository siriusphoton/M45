#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

compose=(
    docker compose
    --env-file .env.example
    --profile test
)
test_database_url="postgresql+psycopg://m45:m45@127.0.0.1:5433/m45_test"

cleanup_test_database() {
    "${compose[@]}" rm --stop --force postgres_test >/dev/null 2>&1 || true
}

trap cleanup_test_database EXIT

uv sync --locked
"${compose[@]}" config --quiet
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pyright

cleanup_test_database
"${compose[@]}" up --detach --wait postgres_test

"${compose[@]}" exec -T postgres_test \
    psql -X -U m45 -d m45_test -v ON_ERROR_STOP=1 \
    -c 'CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;'

DATABASE_URL="$test_database_url" \
    uv run --no-sync alembic upgrade head

DATABASE_URL="$test_database_url" \
    uv run --no-sync alembic check

DATABASE_URL="$test_database_url" \
    uv run --no-sync pytest
