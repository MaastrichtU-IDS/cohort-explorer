#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/demo-common.sh"
demo_prepare_runtime
backend_port="$(demo_gateway_host_port 3000)"

cd "${DEMO_ROOT}/backend"
uv run python ../scripts/demo-seed.py \
  --base-url "http://127.0.0.1:${backend_port}" \
  --pack "$(demo_runtime_value DEMO_PACK_HOST_DIR)"
