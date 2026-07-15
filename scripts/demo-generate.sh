#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/demo-common.sh"
demo_prepare_runtime

output="$(demo_runtime_value DEMO_PACK_HOST_DIR)"
args=(--seed "${DEMO_SEED:-42}" --rows "${DEMO_ROWS:-2500}" --output "${output}")
if [[ "${DEMO_FORCE:-false}" == "true" ]]; then
  args+=(--force)
fi

cd "${DEMO_ROOT}/backend"
uv run python scripts/seed_synthetic_data.py "${args[@]}"
