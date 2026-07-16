#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "${script_dir}/.." && pwd)"

if [[ -n "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]; then
  printf 'Refusing browser evidence from a dirty Cohort Explorer checkout:\n' >&2
  git -C "${root}" status --short --untracked-files=all >&2
  exit 1
fi
ce_start_revision="$(git -C "${root}" rev-parse HEAD)"

if [[ -z "${COMPOSE_PROJECT_NAME:-}" ]]; then
  export COMPOSE_PROJECT_NAME="cohort-explorer-browser-test-$(date -u +%Y%m%d%H%M%S)-$$"
fi
if [[ -z "${DEMO_PACK_HOST_DIR:-}" ]]; then
  export DEMO_PACK_HOST_DIR="${root}/data/browser-${COMPOSE_PROJECT_NAME}-pack"
fi

source "${script_dir}/demo-common.sh"
"${script_dir}/demo-browser-ready.sh"
aadcr_root="$(demo_runtime_value AADCRV2_REPO_DIR)"
aadcr_start_revision="$(git -C "${aadcr_root}" rev-parse HEAD)"

printf 'Browser demo namespace: %s\n' "${DEMO_NAMESPACE}"
printf 'Teardown after review: COMPOSE_PROJECT_NAME=%s DEMO_PURGE=true make demo-down\n' "${DEMO_NAMESPACE}"

frontend_port="$(demo_gateway_host_port 3001)"
backend_port="$(demo_gateway_host_port 3000)"
cd "${root}/frontend"
DEMO_BROWSER_URL="http://localhost:${frontend_port}" \
DEMO_API_URL="http://localhost:${backend_port}" \
DEMO_BROWSER_PACK="$(demo_runtime_value DEMO_PACK_HOST_DIR)" \
DEMO_BROWSER_EVIDENCE="${DEMO_STATE_DIR}/browser-evidence" \
npm run test:e2e:local -- "$@"

if [[ "$(git -C "${root}" rev-parse HEAD)" != "${ce_start_revision}" ]] ||
   [[ -n "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]; then
  printf 'Cohort Explorer changed during browser acceptance; refusing passing evidence.\n' >&2
  exit 1
fi
if [[ "$(git -C "${aadcr_root}" rev-parse HEAD)" != "${aadcr_start_revision}" ]] ||
   [[ -n "$(git -C "${aadcr_root}" status --porcelain --untracked-files=all)" ]]; then
  printf 'AADCR v2 changed during browser acceptance; refusing passing evidence.\n' >&2
  exit 1
fi

python3 - \
  "${DEMO_STATE_DIR}/browser-evidence/run-summary.json" \
  "${DEMO_NAMESPACE}" \
  "http://localhost:${frontend_port}" \
  "$(demo_runtime_value LOCAL_AUTH_EMAIL)" \
  "${root}" \
  "${aadcr_root}" <<'PY'
import datetime
import json
import pathlib
import subprocess
import sys

summary_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[5])
aadcr_root = pathlib.Path(sys.argv[6])

def revision(repository: pathlib.Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

details_path = summary_path.parent / "acceptance-details.json"
if not details_path.is_file():
    raise SystemExit("Playwright passed without writing sanitized acceptance details")
details = json.loads(details_path.read_text(encoding="utf-8"))
evidence = sorted(path.name for path in summary_path.parent.glob("*.png"))
required_evidence = [f"{index:02d}-{name}.png" for index, name in enumerate((
    "login-admin",
    "dictionaries-uploaded",
    "metadata-filters",
    "manual-concept-mapping",
    "generated-mapping-table",
    "generated-mapping-graph",
    "dcr-wizard-review",
    "dcr-created",
    "my-dcrs",
    "audit-log",
    "aggregate-result",
), start=1)]
if evidence != required_evidence:
    raise SystemExit(f"Browser evidence mismatch: {evidence!r}")
summary = details | {
    "admin_email": sys.argv[4],
    "evidence": evidence,
    "namespace": sys.argv[2],
    "passed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_revisions": {
        "aadcrv2": revision(aadcr_root),
        "cohort_explorer": revision(root),
    },
    "source_tree_state": "clean",
    "status": "passed",
    "url": sys.argv[3],
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
details_path.unlink()
PY
