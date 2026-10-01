#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first; see docs/getting-started.md."
  exit 1
fi
case "${1:-check}" in
  notebook) exec uv run --locked jupyter lab ;;
  test) exec uv run --locked pytest -q ;;
  execute-notebooks) exec uv run --locked python scripts/execute_notebooks.py ;;
  *) exec uv run --locked traffic-study "${1:-check}" ;;
esac
