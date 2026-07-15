# Cohort Explorer to AADCR v2 Integration and Browser Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use the browser-use:browser skill for every visible-browser checkpoint.

**Goal:** Connect Cohort Explorer's existing DCR endpoints to the pinned local AADCR v2 backend, orchestrate a one-command synthetic demo, and prove the real upload-to-computation journey in the local browser.

**Architecture:** Cohort Explorer keeps its public HTTP contract and selects a `DcrBackend` implementation. The existing Decentriq SDK logic remains the production-default adapter; the AADCR adapter translates the same request into native DEV, merge, PROD, upload, provision, computation, result, and audit calls using signed service JWTs and a resumable local operation journal. A Cohort Explorer-owned Compose overlay builds the sibling Delta backend and exposes one isolated browser-demo namespace.

**Tech Stack:** Python 3.11, FastAPI, `httpx`, `python-jose`, JSONL, Docker Compose, Oxigraph, Next.js 14/React 18, Vitest, Playwright, pytest.

## Global Constraints

- Cohort Explorer work is in `/Users/nikolasmolyndris/projects/cohort-explorer-aadcrv2` on `codex/aadcrv2-local-integration`, based on `8aed288b18b04da3cf7d8f9c76bcb5d04467dace`.
- AADCR work is supplied by `/Users/nikolasmolyndris/projects/delta-aadcrv2` on `codex/cohort-explorer-aadcrv2`, based on `f13ef54fc3f0f56dae185d4aa35c6dff01ee8839`.
- Complete the offline-metadata and AADCR-backend plans before the composed end-to-end tasks in this plan.
- Keep `DCR_BACKEND=decentriq` as the application and production-Compose default. AADCR selection requires the explicit local overlay.
- Keep all existing Cohort Explorer route paths and successful response fields stable.
- Authenticate the browser only through guarded local login as `nikolas.molyndris@decentriq.ch`; the caller never chooses the local identity.
- Authenticate Cohort Explorer to AADCR with short-lived signed HS256 JWTs. Never forward the browser session cookie and never commit a secret.
- Row-level CSVs enter AADCR only from the generated synthetic pack and only when `AADCRV2_SYNTHETIC_DEMO=true`.
- Do not call the live Cohort Explorer, Decentriq, Athena, Qdrant, hosted models, or LLMs. Offline network checks allow only the local Compose network and loopback ports.
- Do not vendor Delta into Cohort Explorer. The overlay consumes `AADCRV2_REPO_DIR`, defaulting to the documented sibling worktree.
- Browser verification is mandatory during implementation and in the final clean acceptance run. API success alone is insufficient.
- Use TDD and commit each completed Cohort Explorer task separately. Delta changes remain separate commits in the Delta repository.

## Dependency Waves

1. Finish Tasks 1-6 of `2026-07-15-cohort-explorer-offline-metadata-demo.md` and Tasks 1-7 of `2026-07-15-aadcrv2-local-backend.md` in parallel.
2. Implement Tasks 1-6 below against mocked AADCR HTTP transport.
3. Add orchestration and run the API smoke flow against both real local services.
4. Run the browser checkpoint against API-seeded state.
5. Reset into a new isolated namespace and run the complete browser-owned acceptance journey.

## File Responsibility Map

- `backend/src/dcr_backends/`: provider-neutral contract, Decentriq adapter, AADCR adapter, HTTP client, JWT issuer, response models, and operation journal.
- `backend/src/dcr_routes.py`: existing provider-facing route paths with no backend-specific branching.
- `backend/src/upload.py`, `backend/src/main.py`, `backend/src/monitoring.py`: incremental provider routing and startup behavior.
- `backend/scripts/smoke_local_aadcr.py`: complete local API proof.
- `docker-compose.local-aadcr.yml`, `.env.local-demo.example`, `Makefile`, `scripts/demo-*`: reproducible cross-repository runtime.
- `frontend/e2e/`, `frontend/playwright.config.ts`: deterministic browser acceptance suite.
- `frontend/src/components/Nav.tsx`, `frontend/src/pages/dcrs.tsx`, related metadata screens: provider-neutral text and stable selectors only.
- `docs/local-demo.md`, `docs/local-demo-browser-checklist.md`: operator and evidence runbooks.

---

### Task 1: Define the provider-neutral DCR contract and factory

**Files:**
- Create: `backend/src/dcr_backends/__init__.py`
- Create: `backend/src/dcr_backends/contracts.py`
- Create: `backend/src/dcr_backends/models.py`
- Create: `backend/src/dcr_backends/factory.py`
- Modify: `backend/src/config.py`
- Create: `backend/tests/test_dcr_backend_factory.py`
- Create: `backend/tests/test_dcr_response_models.py`

**Interfaces:**
- Consumes: existing endpoint request dictionaries and current frontend response assumptions.
- Produces: `DcrBackend`, `DcrCapabilities`, `DcrRoom`, `DcrListResult`, `LiveCreateResult`, `ProviderError`, and `get_dcr_backend(settings)`.

- [ ] **Step 1: Write failing factory/default tests**

```python
def test_default_backend_is_decentriq(settings_factory):
    backend = get_dcr_backend(settings_factory(dcr_backend="decentriq"))
    assert backend.provider_name == "decentriq"


def test_unknown_backend_fails_closed(settings_factory):
    with pytest.raises(ValueError, match="Unsupported DCR_BACKEND"):
        get_dcr_backend(settings_factory(dcr_backend="unknown"))
```

- [ ] **Step 2: Write response-compatibility tests**

Construct `LiveCreateResult`, `DcrRoom`, and `DcrListResult`, serialize them, and assert the fields consumed by `Nav.tsx`, `dcrs.tsx`, `DcrLogPanel.tsx`, and the result panel: `message`, `dcr_id`, `dcr_url`, `dcr_title`, `cohort_ids`, `num_cohorts`, upload maps/counts, participants, owner, nodes, cohorts, provisioned datasets, provider, capabilities, and optional error. The list wrapper preserves `dcrs`, `count`, and `email` while adding top-level `provider` and `capabilities` for empty-state/refresh copy.

- [ ] **Step 3: Run tests and verify RED**

Run: `uv run --project backend pytest backend/tests/test_dcr_backend_factory.py backend/tests/test_dcr_response_models.py -q`

Expected: modules and settings are missing.

- [ ] **Step 4: Implement settings and protocol**

```python
class DcrBackend(Protocol):
    provider_name: str
    capabilities: DcrCapabilities

    async def create_provision_room(self, request: ProvisionRoomRequest, user: CurrentUser) -> dict[str, Any]: ...
    async def preview_definition(self, request: ComputeRoomRequest, user: CurrentUser) -> ResponsePayload: ...
    async def create_live_room(self, request: ComputeRoomRequest, user: CurrentUser) -> LiveCreateResult: ...
    async def list_rooms(self, user: CurrentUser, refresh: bool = False) -> DcrListResult: ...
    async def rooms_last_modified(self, user: CurrentUser) -> datetime | None: ...
    async def audit_log(self, dcr_id: str, user: CurrentUser) -> list[AuditEntry]: ...
    async def computation_output(self, dcr_id: str, user: CurrentUser) -> ResponsePayload: ...
    async def shuffle_output(self, dcr_id: str, user: CurrentUser) -> ResponsePayload: ...
```

Add explicit settings for `DCR_BACKEND`, AADCR URL/JWT/timeout/room URL template, operation-journal path, and synthetic-demo flag. Defaults select Decentriq and disable synthetic provisioning.

- [ ] **Step 5: Implement a lazy factory**

Import only the selected adapter so local AADCR mode does not initialize the Decentriq SDK and production mode does not require the AADCR client.

- [ ] **Step 6: Run tests and commit**

Run: `uv run --project backend pytest backend/tests/test_dcr_backend_factory.py backend/tests/test_dcr_response_models.py -q`

```bash
git add backend/src/config.py backend/src/dcr_backends/__init__.py \
  backend/src/dcr_backends/contracts.py backend/src/dcr_backends/models.py \
  backend/src/dcr_backends/factory.py backend/tests/test_dcr_backend_factory.py \
  backend/tests/test_dcr_response_models.py
git commit -m "refactor: define DCR backend contract"
```

### Task 2: Wrap the existing Decentriq implementation without changing defaults

**Files:**
- Create: `backend/src/dcr_backends/decentriq_backend.py`
- Create: `backend/src/dcr_routes.py`
- Modify: `backend/src/decentriq.py`
- Modify: `backend/src/upload.py`
- Modify: `backend/src/main.py`
- Modify: `backend/src/monitoring.py`
- Create: `backend/tests/test_decentriq_backend_adapter.py`
- Create: `backend/tests/test_dcr_route_contracts.py`

**Interfaces:**
- Consumes: current Decentriq helper functions and route payloads.
- Produces: `DecentriqBackend` and provider-neutral route delegation while preserving all current URLs.

- [ ] **Step 1: Freeze current route behavior in tests**

Use a fake `DcrBackend` dependency and assert each existing route delegates exactly once with the authenticated user and unchanged body/path parameters. Include create-provision, preview definition, live create, both audit aliases, compute/shuffle output, My DCRs, refresh, and last-modified.

- [ ] **Step 2: Write Decentriq adapter tests around mocked SDK helpers**

Assert the adapter returns the same JSON/file response shape and maps the provider name and capabilities without contacting Decentriq.

- [ ] **Step 3: Run tests and verify RED**

Run: `uv run --project backend pytest backend/tests/test_decentriq_backend_adapter.py backend/tests/test_dcr_route_contracts.py -q`

- [ ] **Step 4: Extract only provider-facing handlers**

Move route ownership to `dcr_routes.py`; keep the existing SDK construction/room-definition helpers in `decentriq.py`. Inject `get_dcr_backend()` through FastAPI dependencies. Replace `upload.py`'s direct `create_provision_dcr` call and startup/monitoring's unconditional SDK refresh with capability-checked provider calls.

- [ ] **Step 5: Prove production-default compatibility**

Run the contract tests with `DCR_BACKEND` unset and with `DCR_BACKEND=decentriq`. Assert that `main.py` imports without creating a Decentriq client and that production Compose explicitly sets `DCR_BACKEND=decentriq`.

- [ ] **Step 6: Run affected tests and commit**

Run: `uv run --project backend pytest backend/tests/test_decentriq_backend_adapter.py backend/tests/test_dcr_route_contracts.py -q`

```bash
git add backend/src/dcr_backends/decentriq_backend.py backend/src/dcr_routes.py \
  backend/src/decentriq.py backend/src/upload.py backend/src/main.py backend/src/monitoring.py \
  backend/tests/test_decentriq_backend_adapter.py backend/tests/test_dcr_route_contracts.py \
  docker-compose.base.yml
git commit -m "refactor: route DCR operations through providers"
```

### Task 3: Add signed AADCR authentication and bounded HTTP client

**Files:**
- Create: `backend/src/dcr_backends/aadcr_auth.py`
- Create: `backend/src/dcr_backends/aadcr_client.py`
- Create: `backend/tests/test_aadcr_auth.py`
- Create: `backend/tests/test_aadcr_client.py`

**Interfaces:**
- Consumes: AADCR URL, shared secret, issuer, audience, timeout, and authenticated Cohort Explorer user.
- Produces: `mint_aadcr_token(user, settings)` and an asynchronous typed `AadcrClient`.

- [ ] **Step 1: Write token claim/signature tests**

Decode the emitted token with the AADCR test decoder and assert HS256 plus exact `sub`, normalized `email`, `email_verified=true`, `iss`, `aud`, `iat`, and an expiry no more than five minutes later. Assert no session cookie or OAuth token appears in claims.

- [ ] **Step 2: Write HTTP-client tests with `httpx.MockTransport`**

Cover Authorization injection, explicit connect/read timeout, JSON/file upload handling, 4xx/5xx normalization, malformed response, connection failure, and redacted error/log representation.

- [ ] **Step 3: Run tests and verify RED**

Run: `uv run --project backend pytest backend/tests/test_aadcr_auth.py backend/tests/test_aadcr_client.py -q`

- [ ] **Step 4: Implement the token issuer and client**

Create one token per logical operation, never persist it, and attach `Bearer <token>` to AADCR calls. `AadcrUpstreamError` carries method, safe path, status, safe detail, and failed step but not headers, secret, cookies, or uploaded content.

- [ ] **Step 5: Run tests and commit**

Run: `uv run --project backend pytest backend/tests/test_aadcr_auth.py backend/tests/test_aadcr_client.py -q`

```bash
git add backend/src/dcr_backends/aadcr_auth.py backend/src/dcr_backends/aadcr_client.py backend/tests/test_aadcr_auth.py backend/tests/test_aadcr_client.py
git commit -m "feat: add authenticated AADCR client"
```

### Task 4: Translate native AADCR room creation and deterministic definition preview

**Files:**
- Create: `backend/src/dcr_backends/aadcr_backend.py`
- Create: `backend/src/dcr_backends/aadcr_translation.py`
- Create: `backend/src/dcr_backends/definition_archive.py`
- Create: `backend/tests/test_aadcr_translation.py`
- Create: `backend/tests/test_aadcr_backend_create.py`
- Create: `backend/tests/test_aadcr_definition_archive.py`

**Interfaces:**
- Consumes: existing Cohort Explorer cohort request, cache records, selected mapping files, dictionaries/samples, and native AADCR routes.
- Produces: native DEV→merge→PROD flow and deterministic preview ZIP with the existing endpoint content contract.

- [ ] **Step 1: Write the exact outbound-sequence test**

Mock AADCR and assert this order: create DCR; add creator/owners/service identity/analysts; add expected FILE and computation nodes; apply owner/analyst permissions; fetch DEV change IDs; create one merge request; poll to `MERGED`; fetch PROD; resolve nodes by exact expected names; upload/provision dictionaries, mappings, selected samples, and explicit-demo synthetic CSVs.

- [ ] **Step 2: Cover governance and failure branches**

Test missing cohort, invalid mapping path, unknown PROD node, pending approval, merge failure, bounded polling timeout, upload failure, and `AADCRV2_SYNTHETIC_DEMO=false` omitting all row-level uploads. Never bypass a pending approval.

- [ ] **Step 3: Write deterministic archive tests**

Assert repeated previews have identical hashes and contain `dcr_config.json`, selected dictionaries, available selected samples, mapping files, and a fixture-provenance manifest. Normalize member order, JSON order, timestamps, permissions, and paths.

- [ ] **Step 4: Run tests and verify RED**

Run: `uv run --project backend pytest backend/tests/test_aadcr_translation.py backend/tests/test_aadcr_backend_create.py backend/tests/test_aadcr_definition_archive.py -q`

- [ ] **Step 5: Implement translation and normalized response**

Resolve participant roles through the existing `build_dcr_participants()` behavior. Use provider-neutral errors containing failed step and created room ID. Construct `dcr_url` only through `AADCRV2_ROOM_URL_TEMPLATE`; label aggregate/airlock-like computation as a local simulation, not a confidential boundary.

- [ ] **Step 6: Run tests and commit**

Run: `uv run --project backend pytest backend/tests/test_aadcr_translation.py backend/tests/test_aadcr_backend_create.py backend/tests/test_aadcr_definition_archive.py -q`

```bash
git add backend/src/dcr_backends/aadcr_backend.py backend/src/dcr_backends/aadcr_translation.py \
  backend/src/dcr_backends/definition_archive.py backend/tests/test_aadcr_translation.py \
  backend/tests/test_aadcr_backend_create.py backend/tests/test_aadcr_definition_archive.py
git commit -m "feat: translate Cohort Explorer rooms to AADCR"
```

### Task 5: Make creation resumable and normalize room, audit, and result reads

**Files:**
- Create: `backend/src/dcr_backends/operation_journal.py`
- Modify: `backend/src/dcr_backends/aadcr_backend.py`
- Modify: `backend/src/dcr_backends/aadcr_translation.py`
- Create: `backend/tests/test_operation_journal.py`
- Create: `backend/tests/test_aadcr_backend_reads.py`

**Interfaces:**
- Consumes: wizard `session_id`, AADCR room/list/audit/computation responses, data-folder path.
- Produces: append-only operation state, idempotent resume, normalized My DCRs/audit/output/last-modified behavior.

- [ ] **Step 1: Write atomic journal tests**

Test session-key validation, atomic append, process-local same-session locking, corrupt trailing-record recovery, no token/row content, and deterministic replay of a completed response.

- [ ] **Step 2: Write idempotency/resume tests**

Call create twice with one completed `session_id` and assert exactly one AADCR room. Interrupt after merge and after first provision; retry and assert the adapter resumes from the first unconfirmed step without repeating confirmed mutations.

- [ ] **Step 3: Write read-normalization tests**

Mock native room/PROD/provisioned-dataset/audit/computation routes and assert `GET /my-dcrs`, refresh, last-modified, audit aliases, and computation output use existing Cohort Explorer shapes. Each AADCR room includes provider-neutral provisioning records keyed to its PROD node names. For computation, cover no prior execution, queued/running polling, completed base64 ZIP, failed execution, timeout, oversized/invalid archive, and an already-completed fetch that does not start a duplicate run. Unsupported shuffle output returns an explicit capability response rather than silently calling Decentriq.

- [ ] **Step 4: Run tests and verify RED**

Run: `uv run --project backend pytest backend/tests/test_operation_journal.py backend/tests/test_aadcr_backend_reads.py -q`

- [ ] **Step 5: Implement journal and read methods**

Journal only normalized request metadata, room ID, confirmed steps, cohort IDs, timestamps, and final response. Enrich listed rooms from PROD, native provisioned-dataset state, and journal metadata; map native audit rows to `{timestamp, user, desc}` and preserve provider/source fields as optional additions. `rooms_last_modified()` reports the journal's newest durable record.

`computation_output()` resolves the aggregate PROD computation node, first posts its ID/environment to native `/api/dcr/{dcr_id}/computation-nodes/results`, and starts native `/run` only when no execution exists. It polls `/results` to a bounded terminal state, validates and decodes the completed base64 ZIP with size/member-path limits, and returns the existing Cohort Explorer download response. It never trusts upstream archive paths or exposes native stdout containing sensitive content.

- [ ] **Step 6: Run tests and commit**

Run: `uv run --project backend pytest backend/tests/test_operation_journal.py backend/tests/test_aadcr_backend_reads.py -q`

```bash
git add backend/src/dcr_backends/operation_journal.py backend/src/dcr_backends/aadcr_backend.py \
  backend/src/dcr_backends/aadcr_translation.py backend/tests/test_operation_journal.py \
  backend/tests/test_aadcr_backend_reads.py
git commit -m "feat: resume and inspect AADCR operations"
```

### Task 6: Add provider-neutral frontend copy and DCR browser selectors

**Files:**
- Modify: `frontend/src/components/Nav.tsx`
- Modify: `frontend/src/pages/dcrs.tsx`
- Modify: `frontend/src/components/DcrLogPanel.tsx`
- Create: `frontend/src/components/DcrResultPanel.tsx`
- Create: `frontend/src/utils/dcrProvider.ts`
- Create: `frontend/src/utils/dcrProvider.test.ts`
- Create: `frontend/src/utils/dcrResult.ts`
- Create: `frontend/src/utils/dcrResult.test.ts`

**Interfaces:**
- Consumes: provider/capability fields already returned by the backend.
- Produces: correct local-vs-Decentriq copy, an inline provision/result status panel, and durable DCR wizard/My DCR locators without a page redesign.

- [ ] **Step 1: Write provider-copy and capability tests**

Assert AADCR uses “Create Data Clean Room”, “Open created room”, and “Refresh rooms”; Decentriq retains its platform-specific destination where accurate. Test the pure result projection for JSON and ZIP responses, and ensure unsupported capabilities suppress or explain their control instead of failing after click.

- [ ] **Step 2: Run tests and verify RED**

Run: `npm --prefix frontend run test -- --run frontend/src/utils/dcrProvider.test.ts frontend/src/utils/dcrResult.test.ts`

- [ ] **Step 3: Implement pure copy/capability projection**

Render from backend provider/capability data; do not infer provider from URLs and do not add a provider picker. Use each room's returned `dcr_url` instead of reconstructing a Decentriq URL in `DcrCard`.

- [ ] **Step 4: Add the inline provisioning/result panel**

Render the normalized provisioned dataset name, PROD node name, and status in each room card. When the provider advertises computation output, a button calls the existing `/compute-get-output/{dcr_id}` endpoint. JSON aggregate output is rendered as a small table; ZIP output is downloaded and the panel visibly reports filename, byte size, and ready status. The browser suite inspects the ZIP contents and hash. Do not parse or render row-level data.

- [ ] **Step 5: Add stable selectors only at DCR interaction boundaries**

Add `data-testid` values for: DCR launcher; modal; each wizard step; name/research-question inputs; participant toggles; sample/mapping toggles; mapping upload slot; previous/next/create buttons; success/error panel; created-room link; My DCR page, refresh, room card, provisioning rows, room audit, run/fetch-result control, and result-ready panel. Keep selectors semantic and independent of generated IDs, dates, CSS, and visible copy.

- [ ] **Step 6: Run frontend unit/build checks**

Run: `npm --prefix frontend run test -- --run`

Run: `npm --prefix frontend run lint && npm --prefix frontend run build`

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/Nav.tsx frontend/src/components/DcrLogPanel.tsx \
  frontend/src/components/DcrResultPanel.tsx frontend/src/pages/dcrs.tsx \
  frontend/src/utils/dcrProvider.ts frontend/src/utils/dcrProvider.test.ts \
  frontend/src/utils/dcrResult.ts frontend/src/utils/dcrResult.test.ts
git commit -m "fix: make DCR controls provider neutral"
```

### Task 7: Add the cross-repository local runtime and stable demo commands

**Files:**
- Create: `docker-compose.local-aadcr.yml`
- Create: `.env.local-demo.example`
- Modify: `.gitignore`
- Create: `Makefile`
- Create: `scripts/demo-common.sh`
- Create: `scripts/demo-generate.sh`
- Create: `scripts/demo-up.sh`
- Create: `scripts/demo-seed.py`
- Create: `scripts/demo-smoke.sh`
- Create: `scripts/demo-browser-ready.sh`
- Create: `scripts/demo-down.sh`
- Create: `scripts/wait_for_demo.py`
- Create: `backend/src/health.py`
- Modify: `backend/src/main.py`
- Create: `backend/tests/test_health.py`
- Create: `frontend/src/pages/api/health.ts`
- Create: `backend/tests/test_local_demo_configuration.py`

**Interfaces:**
- Consumes: sibling Delta backend, immutable synthetic pack, fresh mutable runtime volumes, metadata fixture providers, local login, and local readiness probes for all four services.
- Produces: `demo-generate`, `demo-up`, `demo-seed`, `demo-smoke`, `demo-browser-ready`, and `demo-down`.

- [ ] **Step 1: Write configuration-contract tests**

Parse rendered Compose config and assert: explicit AADCR build context; production default remains Decentriq; local overlay selects AADCR and all three fixture metadata providers; fixed admin is guarded by dev/local-auth flags; synthetic demo is explicit; `DEMO_PACK_DIR=/demo-pack` is a read-only bind while `DATA_FOLDER=/demo-runtime` is a project-scoped mutable volume; `INTERNAL_API_URL=http://backend:80`; Oxigraph is pinned; AADCR gets no Docker socket; secrets are environment placeholders; expected ports are 3001/3000/7878/8000.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --project backend pytest backend/tests/test_local_demo_configuration.py -q`

- [ ] **Step 3: Implement overlay and environment contract**

Set `AADCRV2_REPO_DIR` to `/Users/nikolasmolyndris/projects/delta-aadcrv2` only as the documented local default; permit override. Build the backend subdirectory inside that worktree. Mount the generated pack at `/demo-pack:ro` and keep runtime cohort files, mappings, journals, and databases in namespace-scoped volumes. Generate session and inter-service JWT secrets at runtime into an ignored state directory with mode 0600; never put a usable secret in `.env.local-demo.example`.

Add a dependency-aware Cohort Explorer backend `/health`, a frontend `/api/health`, use Oxigraph's documented liveness response, and consume AADCR `/health`. `wait_for_demo.py` reports which probe failed instead of treating an open port as ready.

- [ ] **Step 4: Implement stable commands**

- `demo-generate`: generate/validate the deterministic pack.
- `demo-up`: validate branch paths and start the explicit overlay.
- `demo-seed`: authenticate locally and upload central workbook plus both dictionaries; do not create mappings or rooms.
- `demo-smoke`: invoke the complete API smoke script.
- `demo-browser-ready`: use a unique `COMPOSE_PROJECT_NAME`, fresh mutable volumes/cookies, generate the separate immutable pack, upload only the API-only central workbook, wait for readiness, and print URL/email/pack/evidence paths. Runtime dictionaries, mappings, and rooms remain absent.
- `demo-down`: stop the selected namespace without deleting data unless `DEMO_PURGE=true` is explicitly supplied.

- [ ] **Step 5: Prove repeatability and isolation**

Render Compose with `docker compose config`. Start/stop/restart one namespace and assert its state persists as documented; then start a second distinct `COMPOSE_PROJECT_NAME` and assert it has no room, mapping, or dictionary from the first. Confirm only local endpoints are configured.

- [ ] **Step 6: Run checks and commit**

Run: `uv run --project backend pytest backend/tests/test_local_demo_configuration.py -q`

Run: `docker compose -f docker-compose.yml -f docker-compose.local-aadcr.yml config --quiet`

```bash
git add .gitignore .env.local-demo.example Makefile docker-compose.local-aadcr.yml \
  scripts/demo-common.sh scripts/demo-generate.sh scripts/demo-up.sh scripts/demo-seed.py \
  scripts/demo-smoke.sh scripts/demo-browser-ready.sh scripts/demo-down.sh \
  scripts/wait_for_demo.py backend/src/health.py backend/src/main.py \
  backend/tests/test_health.py backend/tests/test_local_demo_configuration.py \
  frontend/src/pages/api/health.ts
git commit -m "build: orchestrate local AADCR demo"
```

### Task 8: Prove the complete integration through the API lane

**Files:**
- Create: `backend/scripts/smoke_local_aadcr.py`
- Create: `backend/tests/test_local_aadcr_contract.py`
- Create: `docs/local-demo.md`

**Interfaces:**
- Consumes: running local stack and generated pack.
- Produces: machine-readable evidence for metadata, creation, provisioning, computation, result, audit, and idempotency.

- [ ] **Step 1: Write a mocked smoke-contract test**

Drive the script against an `httpx.MockTransport` and assert it stops on unexpected status/shape, records no bearer token/cookie/row contents, and emits a JSON summary with hashes and IDs only.

- [ ] **Step 2: Implement the real local sequence**

Login as the fixed admin; seed/verify metadata; create a stable wizard request with both cohorts, mapping, samples, analyst roles, and `session_id`; preview/download/inspect the deterministic definition ZIP; create the live room; assert DEV/merge/PROD state through AADCR; assert both raw CSVs and selected metadata assets were provisioned; run aggregate computation; inspect result ZIP; fetch audit and My DCRs; repeat the same `session_id` and assert the room ID/count is unchanged.

- [ ] **Step 3: Add adversarial checks**

Assert tampered token, outsider room read, cross-room node provisioning, oversized upload, and public-network egress are rejected. The egress guard allows only Compose service names and loopback ports.

- [ ] **Step 4: Run the real API smoke lane**

Run: `make demo-generate && make demo-up && make demo-seed && make demo-smoke`

Expected: one JSON summary reports both cohorts, one room, merged PROD graph, provisioned datasets, completed aggregate, retrievable result, audit entries, and idempotent retry.

- [ ] **Step 5: Run API regression tests**

Run: `uv run --project backend pytest backend/tests/test_dcr_* backend/tests/test_aadcr_* backend/tests/test_operation_journal.py backend/tests/test_local_aadcr_contract.py -q`

Document unrelated baseline failures separately; do not claim a green full suite if `backend/test_sparql_data.py` still attempts live SPARQL during collection.

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/smoke_local_aadcr.py backend/tests/test_local_aadcr_contract.py docs/local-demo.md
git commit -m "test: verify Cohort Explorer AADCR API flow"
```

### Task 9: Add automated browser acceptance and run the implementation checkpoint

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Create: `frontend/playwright.config.ts`
- Create: `frontend/e2e/local-demo.spec.ts`
- Create: `frontend/e2e/helpers/demoEvidence.ts`
- Create: `docs/local-demo-browser-checklist.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `demo-browser-ready`, generated workbook/dictionaries, stable selectors from both frontend plans.
- Produces: screenshots, downloaded definition ZIP, console/network evidence, and browser assertions against the real local UI.

- [ ] **Step 1: Add Playwright and a failing smoke assertion**

Configure Chromium, one worker, trace on first retry, video off, screenshots on failure, and output under ignored `artifacts/browser-demo/`. Do not auto-start a different stack from Playwright; consume the namespace prepared by `demo-browser-ready`.

- [ ] **Step 2: Implement the complete browser-owned journey**

From a clean context:

1. open only `http://localhost:3001` and use local login;
2. assert the signed-in email and admin state;
3. upload both dictionaries with `setInputFiles` through the visible upload/replace controls;
4. validate without mutation and verify both cohort cards and real generated counts;
5. exercise metadata search/filter/equivalent grouping, variable/category concept mapping, EDA/sample affordances, and mapping generation/table/graph views;
6. add both whole cohorts, open the six-step wizard, set name/participants/research question/samples/mapping/upload slot, and review;
7. download/inspect the preview definition, then create the live room;
8. assert the success panel contains the same room ID/title returned by the API and the local provider-neutral link;
9. open My DCRs, refresh, assert the room card/participants/nodes, audit entries, and aggregate result;
10. reload the browser and assert cohort metadata, mapping, and room remain visible.

- [ ] **Step 3: Enforce browser evidence rules**

Capture named screenshots at upload, metadata exploration, manual mapping, generated mapping, wizard review, successful room creation, My DCRs, audit, and aggregate result. Fail on unhandled page errors, failed local API responses, or console errors not present in a short checked-in allowlist. Fail if any request host is outside `localhost`, `127.0.0.1`, or the local static asset origin.

- [ ] **Step 4: Run automated checkpoint 3**

Run: `make demo-browser-ready`

Run: `npm --prefix frontend run test:e2e -- --grep "local AADCR journey"`

Expected: the complete UI flow passes against the local stack and evidence is written only to ignored artifacts.

- [ ] **Step 5: Run a visible in-app browser checkpoint**

Using the browser-use:browser skill, create a named session, inspect the DOM before every action, and repeat at minimum: local login, one dictionary upload, both cohort selection, wizard review/create, My DCR room card, audit, and aggregate result. Capture screenshots. Do not use the live lane or any non-local URL.

- [ ] **Step 6: Commit tests and checklist, not generated evidence**

```bash
git add .gitignore frontend/package.json frontend/package-lock.json frontend/playwright.config.ts \
  frontend/e2e/local-demo.spec.ts frontend/e2e/helpers/demoEvidence.ts \
  docs/local-demo-browser-checklist.md
git commit -m "test: add local browser acceptance lane"
```

### Task 10: Run clean final verification and document the cross-repository pins

**Files:**
- Modify: `docs/local-demo.md`
- Modify: `docs/local-demo-browser-checklist.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: all three implementation plans and both feature branches.
- Produces: reproducible handoff with exact commits, commands, limitations, and final browser/API evidence summary.

- [ ] **Step 1: Record exact dependency commits**

Document the final Cohort Explorer and Delta commit SHAs, their pinned bases, expected sibling paths, required Python/Node/Docker versions, and how to override `AADCRV2_REPO_DIR`. Never reference an uncommitted Delta working tree as the dependency.

- [ ] **Step 2: Start a brand-new acceptance namespace**

Stop the checkpoint namespace. Run `demo-browser-ready` with a fresh unique project name and verify via APIs that only the central workbook is seeded: zero dictionaries, zero runtime mappings, and zero rooms before browser actions.

- [ ] **Step 3: Run verification in order**

1. new and directly affected Delta unit/integration tests;
2. new and directly affected Cohort Explorer backend tests;
3. frontend unit, lint, and build checks;
4. full local API smoke flow in its own clean namespace;
5. automated Playwright journey in the clean browser namespace; and
6. visible in-app browser acceptance using browser-use:browser.

- [ ] **Step 4: Inspect final browser/runtime evidence**

Confirm all required screenshots exist; no unexplained browser-console/page errors; no public-network request; downloaded ZIP manifest/hash is correct; AADCR has one room; both CSVs are provisioned to correct PROD nodes; aggregate result and audit are visible; reload persistence passes.

- [ ] **Step 5: Inspect repository hygiene**

Run secret scanning, `git diff --check`, status, and ignored-artifact checks in both repositories. Generated data, databases, operation journals, cookies, traces, screenshots, downloads, logs, `.env` files, and test results must remain untracked.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md docs/local-demo.md docs/local-demo-browser-checklist.md
git commit -m "docs: document local AADCR demo"
```

- [ ] **Step 7: Apply verification-before-completion**

Record exact commands and fresh outputs. Report remaining unrelated baseline failures by exact test name and scope. Do not call the work complete until the clean browser-owned upload-to-result journey passes locally.

## Browser Acceptance Evidence

The final evidence package is generated, ignored, and local-only. It contains:

- `01-login-admin.png`
- `02-dictionaries-uploaded.png`
- `03-metadata-filters.png`
- `04-manual-concept-mapping.png`
- `05-generated-mapping-table.png`
- `06-generated-mapping-graph.png`
- `07-dcr-wizard-review.png`
- `08-dcr-created.png`
- `09-my-dcrs.png`
- `10-audit-log.png`
- `11-aggregate-result.png`
- Playwright trace on failure only;
- downloaded definition archive and SHA-256; and
- a sanitized JSON summary of console errors, local request failures, room ID, result hash, and relevant commit SHAs.

Evidence never contains a bearer token, browser cookie, shared secret, raw participant rows, or a live-service URL.
