#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export LANGGRAPH_CLI_NO_ANALYTICS=1
export LANGSMITH_TRACING=false

exec uv run --locked langgraph dev --no-browser "$@"
