#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/demo-common.sh"
demo_prepare_runtime
demo_validate_aadcr_checkout

pack="$(demo_runtime_value DEMO_PACK_HOST_DIR)"
[[ -d "${pack}" ]] || demo_die "synthetic pack is missing; run make demo-generate first"

cd "${DEMO_ROOT}/backend"
uv run python scripts/seed_synthetic_data.py --validate --output "${pack}"
cd "${DEMO_ROOT}"

demo_compose config --quiet
demo_compose up --detach --build
backend_port="$(demo_gateway_host_port 3000)"
frontend_port="$(demo_gateway_host_port 3001)"
aadcr_port="$(demo_gateway_host_port 18000)"
python3 "${DEMO_ROOT}/scripts/wait_for_demo.py" \
  --timeout "${DEMO_WAIT_TIMEOUT:-180}" \
  --probe "frontend=http://127.0.0.1:${frontend_port}/api/health" \
  --probe "backend=http://127.0.0.1:${backend_port}/health" \
  --probe "aadcrv2=http://127.0.0.1:${aadcr_port}/health"

printf 'Local demo ready at http://localhost:%s\n' "${frontend_port}"
printf 'Admin: %s\n' "$(demo_runtime_value LOCAL_AUTH_EMAIL)"
printf 'Namespace: %s\n' "${DEMO_NAMESPACE}"
