#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/demo-common.sh"
demo_prepare_runtime

if [[ "${DEMO_PURGE:-false}" == "true" ]]; then
  demo_compose down --remove-orphans --volumes
else
  demo_compose down --remove-orphans
fi
