#!/usr/bin/env bash
#
# Re-apply local iCARE4CHAIN patches after syncing icarechain/ from the
# upstream iCARE4CHAIN branch.
#
# These are the ONLY hand changes we keep on top of a pristine copy of the
# iCARE4CHAIN branch:
#   1. api/routes/admin.py   -> the admin-overview API route (new file)
#   2. api/main.py           -> register the `admin` router (2 spots)
#
# Everything else (e.g. friendlier 404 messages) lives in the frontend, so it
# never needs re-applying here.
#
# Usage:  bash icarechain-local-patches/apply.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY_DIR="$REPO_ROOT/icarechain-local-patches"
ICARECHAIN_DIR="$REPO_ROOT/icarechain"

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

echo "==> Done."
