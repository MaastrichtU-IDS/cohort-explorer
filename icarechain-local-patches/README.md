# icarechain local patches

The `icarechain/` directory is a **vendored copy of the `iCARE4CHAIN` branch**.
To keep it easy to re-sync, we hold our few local additions *here* (outside
`icarechain/`) instead of editing the vendored source. This folder is never
touched when we re-copy the upstream branch.

## What we keep

1. **`api/routes/admin.py`** — a new icarechain API route (`GET /api/admin/overview`)
   that aggregates all consent declarations, access grants and requester
   profiles from icarechain's own cache. It is a proper API endpoint: external
   callers (the `backend/src/blockchain.py` proxy → frontend) reach it **only**
   over HTTP. Nothing outside icarechain touches its internals.

2. **`api/main.py` registration** — two spots that add `admin` to the router
   imports and to the `include_router` loop. The `apply.sh` script does this
   idempotently.

## What is NOT kept here

Friendlier 404 messages (previously hand-edited in `api/routes/cohorts.py` and
`api/routes/requesters.py`) now live in the **frontend**, which maps a 404 from
those endpoints to "Cohort does not yet have usage permissions specified".
So those files stay pristine upstream copies and need no re-application.

## How to re-sync icarechain from upstream

```bash
git fetch origin iCARE4CHAIN

# Replace the tracked contents of icarechain/ with the upstream tree.
git rm -r --cached icarechain >/dev/null
rm -rf icarechain
git read-tree --prefix=icarechain/ -u origin/iCARE4CHAIN

# Re-apply our local additions.
bash icarechain-local-patches/apply.sh
```

After this, `icarechain/` == `origin/iCARE4CHAIN` **plus** the admin route.
