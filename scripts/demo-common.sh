#!/usr/bin/env bash

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_NAMESPACE="${COMPOSE_PROJECT_NAME:-cohort-explorer-aadcr-demo}"
DEMO_STATE_ROOT="${DEMO_STATE_ROOT:-${DEMO_ROOT}/.demo-state}"
DEMO_STATE_DIR="${DEMO_STATE_ROOT}/${DEMO_NAMESPACE}"
DEMO_RUNTIME_ENV="${DEMO_STATE_DIR}/runtime.env"
DEMO_DEFAULT_AADCRV2_REPO="$(cd "${DEMO_ROOT}/.." && pwd)/delta-aadcrv2"
DEMO_AADCRV2_BASE_COMMIT="f13ef54fc3f0f56dae185d4aa35c6dff01ee8839"
DEMO_AADCRV2_REQUIRED_COMMIT="08993663db8084b145d70d369309e82f7080b0f7"
DEMO_AADCRV2_BACKEND_RELATIVE="avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend"

demo_die() {
  printf 'demo error: %s\n' "$*" >&2
  return 1
}

demo_require_namespace() {
  if [[ ! "${DEMO_NAMESPACE}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    demo_die "COMPOSE_PROJECT_NAME must contain only letters, digits, underscores, and hyphens"
  fi
}

demo_abspath() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

demo_prepare_runtime() {
  demo_require_namespace || return 1
  local aadcr_repo="${AADCRV2_REPO_DIR:-${DEMO_DEFAULT_AADCRV2_REPO}}"
  local demo_pack="${DEMO_PACK_HOST_DIR:-${DEMO_ROOT}/data/synthetic-demo-pack}"
  aadcr_repo="$(demo_abspath "${aadcr_repo}")" || return 1
  demo_pack="$(demo_abspath "${demo_pack}")" || return 1

  umask 077
  mkdir -p "${DEMO_STATE_DIR}"
  if [[ ! -f "${DEMO_RUNTIME_ENV}" ]]; then
    local session_secret
    local service_secret
    session_secret="$(openssl rand -hex 32)" || return 1
    service_secret="$(openssl rand -hex 32)" || return 1
    {
      printf 'COMPOSE_PROJECT_NAME=%s\n' "${DEMO_NAMESPACE}"
      printf 'LOCAL_AUTH_EMAIL=%s\n' "${LOCAL_AUTH_EMAIL:-nikolas.molyndris@decentriq.ch}"
      printf 'AADCRV2_REPO_DIR=%s\n' "${aadcr_repo}"
      printf 'DEMO_PACK_HOST_DIR=%s\n' "${demo_pack}"
      printf 'JWT_SECRET=%s\n' "${session_secret}"
      printf 'AADCRV2_JWT_SECRET=%s\n' "${service_secret}"
    } >"${DEMO_RUNTIME_ENV}"
  fi
  chmod 600 "${DEMO_RUNTIME_ENV}"
  export DEMO_ROOT DEMO_NAMESPACE DEMO_STATE_ROOT DEMO_STATE_DIR DEMO_RUNTIME_ENV
}

demo_runtime_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "${DEMO_RUNTIME_ENV}" | tail -n 1
}

demo_validate_aadcr_checkout() {
  local aadcr_repo
  local actual_head
  local dirty_state
  aadcr_repo="$(demo_runtime_value AADCRV2_REPO_DIR)"
  [[ -e "${aadcr_repo}/.git" ]] || demo_die "AADCRV2_REPO_DIR is not a Git checkout: ${aadcr_repo}" || return 1
  [[ -f "${aadcr_repo}/${DEMO_AADCRV2_BACKEND_RELATIVE}/pyproject.toml" ]] || \
    demo_die "AADCR v2 backend was not found below ${aadcr_repo}" || return 1
  git -C "${aadcr_repo}" cat-file -e "${DEMO_AADCRV2_BASE_COMMIT}^{commit}" 2>/dev/null || \
    demo_die "required davstur/aadcrv2 base commit ${DEMO_AADCRV2_BASE_COMMIT} is missing" || return 1
  git -C "${aadcr_repo}" merge-base --is-ancestor "${DEMO_AADCRV2_BASE_COMMIT}" HEAD || \
    demo_die "AADCR checkout HEAD does not descend from ${DEMO_AADCRV2_BASE_COMMIT}" || return 1
  git -C "${aadcr_repo}" cat-file -e "${DEMO_AADCRV2_REQUIRED_COMMIT}^{commit}" 2>/dev/null || \
    demo_die "required reviewed AADCR integration commit ${DEMO_AADCRV2_REQUIRED_COMMIT} is missing" || return 1
  actual_head="$(git -C "${aadcr_repo}" rev-parse HEAD)" || return 1
  [[ "${actual_head}" == "${DEMO_AADCRV2_REQUIRED_COMMIT}" ]] || \
    demo_die "AADCR checkout HEAD ${actual_head} is not the reviewed commit ${DEMO_AADCRV2_REQUIRED_COMMIT}" || return 1
  dirty_state="$(git -C "${aadcr_repo}" status --porcelain --untracked-files=all)" || return 1
  [[ -z "${dirty_state}" ]] || \
    demo_die "AADCR checkout contains tracked or untracked changes; use the exact clean reviewed commit" || return 1
}

demo_compose() {
  env \
    -u AADCRV2_REPO_DIR \
    -u AADCRV2_JWT_SECRET \
    -u COMPOSE_PROJECT_NAME \
    -u DEMO_PACK_HOST_DIR \
    -u JWT_SECRET \
    -u LOCAL_AUTH_EMAIL \
    docker compose \
    --project-name "${DEMO_NAMESPACE}" \
    --env-file "${DEMO_RUNTIME_ENV}" \
    -f "${DEMO_ROOT}/docker-compose.yml" \
    -f "${DEMO_ROOT}/docker-compose.local-aadcr.yml" \
    "$@"
}

demo_gateway_host_port() {
  local container_port="$1"
  local binding
  local host_port
  binding="$(demo_compose port gateway "${container_port}" | tail -n 1)" || return 1
  host_port="${binding##*:}"
  if [[ ! "${host_port}" =~ ^[0-9]+$ ]]; then
    demo_die "gateway port ${container_port} is not published by the selected demo stack"
    return 1
  fi
  printf '%s\n' "${host_port}"
}
