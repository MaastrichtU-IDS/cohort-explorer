#!/usr/bin/env bash
#
# Re-apply local iCARE4CHAIN patches after syncing icarechain/ from the
# upstream iCARE4CHAIN branch.
#
# These are the ONLY hand changes we keep on top of a pristine copy of the
# iCARE4CHAIN branch (see README.md for the running list with rationale):
#   1. api/routes/admin.py                    -> admin overview + reset routes (new file)
#   2. api/main.py                            -> register the `admin` router (2 spots)
#   3. api/services/ontology/icd10_hierarchy.json -> add C00-C97 / C00-C75 blocks
#   4. scripts/deploy.ts                      -> evm_snapshot after deploy (patches/deploy-ts-snapshot.patch)
#   5. deploy-and-generate.sh                 -> honour RPC_URL from the environment
#
# Everything else (e.g. friendlier 404 messages) lives in the frontend, so it
# never needs re-applying here.
#
# Usage:  bash icarechain-local-patches/apply.sh
#         ICARECHAIN_DIR=/some/copy bash icarechain-local-patches/apply.sh   (dry-run on a copy)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY_DIR="$REPO_ROOT/icarechain-local-patches"
ICARECHAIN_DIR="${ICARECHAIN_DIR:-$REPO_ROOT/icarechain}"

echo "==> Applying local icarechain patches"

# 1. Copy the admin route in.
mkdir -p "$ICARECHAIN_DIR/api/routes"
cp "$OVERLAY_DIR/api/routes/admin.py" "$ICARECHAIN_DIR/api/routes/admin.py"
echo "    - copied api/routes/admin.py"

# 2. Register the admin router in api/main.py (idempotent).
python3 - "$ICARECHAIN_DIR/api/main.py" <<'PY'
import re, sys

path = sys.argv[1]
with open(path, "r") as f:
    src = f.read()

changed = False

# 2a. Ensure `admin` is imported from api.routes.
import_re = re.compile(r"from api\.routes import \(\n((?:.*\n)*?)\)")
m = import_re.search(src)
if m and "admin" not in [ln.strip().rstrip(",") for ln in m.group(1).splitlines()]:
    body = m.group(1)
    new_body = "    admin,\n" + body
    src = src[:m.start(1)] + new_body + src[m.end(1):]
    changed = True

# 2b. Ensure `admin` is in the include_router loop tuple.
loop_re = re.compile(r"for r in \(([^)]*)\):")
m = loop_re.search(src)
if m and "admin" not in [x.strip() for x in m.group(1).split(",")]:
    names = m.group(1).rstrip()
    if not names.endswith(","):
        names += ","
    new_names = names + " admin"
    src = src[:m.start(1)] + new_names + src[m.end(1):]
    changed = True

if changed:
    with open(path, "w") as f:
        f.write(src)
    print("    - registered admin router in api/main.py")
else:
    print("    - api/main.py already registers admin router (no change)")
PY

# 3. Add the two malignant-neoplasm blocks to the ICD-10 hierarchy (idempotent).
python3 - "$ICARECHAIN_DIR/api/services/ontology/icd10_hierarchy.json" <<'PY'
import json, sys
path = sys.argv[1]
raw = json.load(open(path))
parents, labels = raw["parents"], raw.setdefault("labels", {})
want_parents = {
    "C00-C97": "C00-D48",
    "C00-C75": "C00-C97",
    "C50-C50": "C00-C75",
    "C60-C63": "C00-C75",
    "C81-C96": "C00-C97",
}
want_labels = {
    "C00-C97": "Malignant neoplasms",
    "C00-C75": "Malignant neoplasms, stated or presumed to be primary, of specified sites, except of lymphoid, haematopoietic and related tissue",
}
changed = False
for k, v in want_parents.items():
    if parents.get(k) != v:
        parents[k] = v; changed = True
for k, v in want_labels.items():
    if labels.get(k) != v:
        labels[k] = v; changed = True
if changed:
    with open(path, "w") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False); f.write("\n")
    print("    - added C00-C97 / C00-C75 blocks to icd10_hierarchy.json")
else:
    print("    - icd10_hierarchy.json already has the cancer blocks (no change)")
PY

# 4. Post-deploy evm_snapshot in scripts/deploy.ts (idempotent via reverse-check).
if grep -q 'evm_snapshot' "$ICARECHAIN_DIR/scripts/deploy.ts"; then
    echo "    - scripts/deploy.ts already takes a post-deploy snapshot (no change)"
else
    patch -p2 -N -d "$ICARECHAIN_DIR" < "$OVERLAY_DIR/patches/deploy-ts-snapshot.patch"
    echo "    - patched scripts/deploy.ts (evm_snapshot)"
fi

# 5. deploy-and-generate.sh: take RPC_URL from the environment (idempotent).
if grep -q 'export RPC_URL=\${RPC_URL:-' "$ICARECHAIN_DIR/deploy-and-generate.sh"; then
    echo "    - deploy-and-generate.sh already honours RPC_URL (no change)"
else
    sed -i.bak 's#^export RPC_URL=http://hardhat:8545#export RPC_URL=${RPC_URL:-http://hardhat:8545}#' "$ICARECHAIN_DIR/deploy-and-generate.sh" \
        && rm -f "$ICARECHAIN_DIR/deploy-and-generate.sh.bak"
    echo "    - patched deploy-and-generate.sh (RPC_URL from env)"
fi

echo "==> Done."
