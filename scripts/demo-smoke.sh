#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/demo-common.sh"
demo_prepare_runtime
DEMO_SMOKE_EVIDENCE="${DEMO_STATE_DIR}/smoke-evidence.json"
rm -f -- "${DEMO_SMOKE_EVIDENCE}"
backend_port="$(demo_gateway_host_port 3000)"
aadcr_port="$(demo_gateway_host_port 18000)"
aadcr_ui_port="$(demo_gateway_host_port 3002)"

cd "${DEMO_ROOT}/backend"
uv run python scripts/smoke_local_aadcr.py \
  --base-url "http://127.0.0.1:${backend_port}" \
  --aadcr-url "http://127.0.0.1:${aadcr_port}" \
  --aadcr-ui-url "http://127.0.0.1:${aadcr_ui_port}" \
  --pack "$(demo_runtime_value DEMO_PACK_HOST_DIR)" \
  --runtime-env "${DEMO_RUNTIME_ENV}" \
  --project-name "${DEMO_NAMESPACE}" \
  --compose-file "${DEMO_ROOT}/docker-compose.yml" \
  --compose-file "${DEMO_ROOT}/docker-compose.local-aadcr.yml" \
  --evidence "${DEMO_SMOKE_EVIDENCE}"
