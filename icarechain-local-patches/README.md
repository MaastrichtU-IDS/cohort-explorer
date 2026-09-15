# icarechain local patches

The `icarechain/` directory is a **vendored copy of the `iCARE4CHAIN` branch**.
To keep it easy to re-sync, we hold our few local additions *here* (outside
`icarechain/`) instead of editing the vendored source. This folder is never
touched when we re-copy the upstream branch.

## What we keep

1. **`api/routes/admin.py`** — a new icarechain API route file with
   `GET /api/admin/overview` (aggregates all consent declarations, access grants
   and requester profiles from icarechain's own cache) and `POST /api/admin/reset`
   (dev-only: reverts the local Hardhat chain to its post-deploy snapshot and
   flushes the cache; refuses on any chain id other than 31337). External
   callers (the `backend/src/blockchain.py` proxy → frontend) reach these
   **only** over HTTP.

2. **`api/main.py` registration** — two spots that add `admin` to the router
   imports and to the `include_router` loop. The `apply.sh` script does this
   idempotently.

3. **`api/services/ontology/icd10_hierarchy.json`** — adds the `C00-C97` and
   `C00-C75` malignant-neoplasm blocks (and re-parents `C50-C50`, `C60-C63`,
   `C81-C96` under them) so the hierarchy matches the Cohort Explorer upload
   dropdown. Applied idempotently by `apply.sh`.

4. **`scripts/deploy.ts`** — calls `evm_snapshot` after deployment and writes
   `snapshotId` into `deployments.json`; this is what `/admin/reset` reverts to.
   Kept as `patches/deploy-ts-snapshot.patch`.

5. **`deploy-and-generate.sh`** — `RPC_URL` is taken from the environment
   (`${RPC_URL:-http://hardhat:8545}`) because the root compose names the node
   service `icarechain-hardhat`, not `hardhat`.

## Running list of local changes to `icarechain/`

Keep this table current: **every** hand edit inside `icarechain/` gets a row
here and an idempotent step in `apply.sh`. "Upstream?" is whether it is worth
proposing to Ankur's `iCARE4CHAIN` branch, so the row can eventually be dropped.

| Date | File | Change | Why | Upstream? |
|---|---|---|---|---|
| 2026-07-13 | `api/routes/admin.py` (new) | `GET /admin/overview` | Consent dashboard needs an all-cohorts aggregate the public routes don't expose | maybe |
| 2026-07-13 | `api/main.py` | register `admin` router | needed by the above | with the above |
| pre-2026-09 | `deploy-and-generate.sh` | `RPC_URL` from env | root compose service is `icarechain-hardhat`; hardcoded `hardhat` host never resolves, deployer waits forever | yes (harmless upstream) |
| 2026-09-15 | `api/routes/admin.py` | overview emits `disease_codes` list per consent and per grant | dashboard showed only the first code; chain and cache already store the full list | yes |
| 2026-09-15 | `api/services/ontology/icd10_hierarchy.json` | add `C00-C97`, `C00-C75` blocks | upload dropdown offers them; icarechain rejected them with 422 "not a supported ICD-10 code" | yes |
| 2026-09-15 | `api/routes/admin.py` | overview emits `reason`, `reason_detail`, `decided_at` per grant and a `rejected_access_requests` stat | dashboard needs to show why the chain rejected a request | yes |
| 2026-09-15 | `scripts/deploy.ts` | `evm_snapshot` after deploy → `deployments.json.snapshotId` | target for local chain reset | dev-only, optional |
| 2026-09-15 | `api/routes/admin.py` | `POST /admin/reset` (Hardhat 31337 only) | "Reset local chain" button on the consent dashboard; also clears the Redis cache, which a compose restart does not | dev-only, optional |

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

After this, `icarechain/` == `origin/iCARE4CHAIN` **plus** the rows in the
table above. Then rebuild the deployer image (it bakes in `deploy.ts`,
`deploy-and-generate.sh` and the hierarchy JSON):

```bash
docker compose build icarechain-deployer
```
