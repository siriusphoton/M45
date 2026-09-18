#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

uv sync --locked
docker compose --env-file .env.example config --quiet
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pyright

status=0
uv run --no-sync pytest || status=$?

# Temporary: remove this exception when the first implementation tests land.
# Every other pytest failure remains a failure of this script.
if [[ "$status" -eq 5 ]]; then
    echo "Pytest collected no tests yet."
    exit 0
fi

exit "$status"
