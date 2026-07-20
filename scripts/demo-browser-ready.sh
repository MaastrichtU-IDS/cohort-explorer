#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "${script_dir}/.." && pwd)"

if [[ -z "${COMPOSE_PROJECT_NAME:-}" ]]; then
  export COMPOSE_PROJECT_NAME="cohort-explorer-browser-$(date -u +%Y%m%d%H%M%S)-$$"
fi
if [[ -z "${DEMO_PACK_HOST_DIR:-}" ]]; then
  export DEMO_PACK_HOST_DIR="${root}/data/browser-${COMPOSE_PROJECT_NAME}-pack"
fi

source "${script_dir}/demo-common.sh"
demo_prepare_runtime
demo_validate_aadcr_checkout

# This command intentionally creates a fresh browser checkpoint. Only volumes
# belonging to its explicit Compose namespace are removed.
demo_compose down --remove-orphans --volumes

DEMO_FORCE=true "${script_dir}/demo-generate.sh"
"${script_dir}/demo-up.sh"
frontend_port="$(demo_gateway_host_port 3001)"
aadcr_ui_port="$(demo_gateway_host_port 3002)"
"${script_dir}/demo-seed.sh" --central-only

evidence_dir="${DEMO_STATE_DIR}/browser-evidence"
rm -rf -- "${evidence_dir}"
mkdir -p "${evidence_dir}"
chmod 700 "${evidence_dir}"

python3 - \
  "${DEMO_NAMESPACE}" \
  "$(demo_runtime_value LOCAL_AUTH_EMAIL)" \
  "$(demo_runtime_value DEMO_PACK_HOST_DIR)" \
  "${frontend_port}" \
  "${aadcr_ui_port}" \
  "${evidence_dir}" <<'PY'
import json
import sys

print(json.dumps({
    "url": f"http://localhost:{sys.argv[4]}",
    "aadcr_url": f"http://localhost:{sys.argv[5]}",
    "admin_email": sys.argv[2],
    "namespace": sys.argv[1],
    "pack": sys.argv[3],
    "evidence": sys.argv[6],
}, sort_keys=True))
PY
