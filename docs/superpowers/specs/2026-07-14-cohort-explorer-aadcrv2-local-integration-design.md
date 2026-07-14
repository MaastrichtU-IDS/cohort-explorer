# Cohort Explorer to AADCR v2 Local Integration Design

Date: 2026-07-14

Status: Architecture approved in conversation; written-spec review pending

Local admin: `nikolas.molyndris@decentriq.ch`

## 1. Objective

Build a reproducible, code-first local demonstration in which Cohort Explorer can:

1. load two synthetic iCARE4CVD cohorts and their data dictionaries;
2. authenticate a fixed development administrator without contacting the live OAuth service;
3. preserve its current DCR wizard and API response contracts;
4. create and configure an Advanced Analytics DCR v2 room through the AADCR v2 REST API;
5. upload and provision synthetic row-level datasets into the matching production data nodes;
6. run an aggregate computation and retrieve its result and audit trail; and
7. keep the existing Decentriq SDK backend available as the production/default provider.

The first implementation and verification pass is API- and test-driven. It does not inspect or seed the live Cohort Explorer, use browser state, or depend on the AADCR frontend.

## 2. Pinned source baselines and workspaces

The implementation is based on these verified commits:

- `MaastrichtU-IDS/cohort-explorer` `main` at `8aed288b18b04da3cf7d8f9c76bcb5d04467dace`.
- `decentriq/delta` `davstur/aadcrv2` at `f13ef54fc3f0f56dae185d4aa35c6dff01ee8839`.

`nik_aadcrv2` is not used as the base. It diverged 43 commits before David's final AADCR v2 branch and contains one later Copilot commit, so it is an optional patch source rather than a successor revision.

Local paths:

- canonical Cohort Explorer clone: `/Users/nikolasmolyndris/projects/cohort-explorer`
- Cohort Explorer feature worktree: `/Users/nikolasmolyndris/projects/cohort-explorer-aadcrv2`
- canonical Delta clone: `/Users/nikolasmolyndris/projects/delta`
- Delta feature worktree: `/Users/nikolasmolyndris/projects/delta-aadcrv2`

The Cohort Explorer branch is `codex/aadcrv2-local-integration`. The Delta branch is `codex/cohort-explorer-aadcrv2`.

## 3. Scope boundaries

### Included

- A provider interface inside the Cohort Explorer backend.
- An AADCR v2 HTTP provider that uses the existing DEV, merge-request, PROD, dataset, computation, and audit routes.
- The current Decentriq SDK implementation retained as the default provider.
- A development-only login using the configured fixed email.
- Signed, short-lived server-to-server JWTs.
- Local-demo hardening required to run AADCR v2 predictably and avoid trivial cross-user access.
- A deterministic synthetic-data generator grounded in the repository's schemas and committed TIME-CHF/GISSI-HF mapping artifact.
- Docker and command-line orchestration, contract tests, and a code-only end-to-end smoke test.
- Small provider-neutral text/link corrections where the current frontend hardcodes Decentriq, without redesigning the UI or using browser verification in this phase.

### Excluded

- Any read from, write to, or calibration against the live Cohort Explorer or live Decentriq platform.
- Incorporating the divergent Copilot patch.
- Claiming that the standalone AADCR v2 Python executor provides production confidential-computing or clean-room isolation.
- Production TLS, secret rotation infrastructure, malware scanning, rate limiting, tamper-evident audit storage, SQLite multi-worker guarantees, or full arbitrary-code sandboxing.
- Rebuilding the AADCR v2 frontend.
- Vendoring the Delta feature into the Cohort Explorer repository.

## 4. Chosen architecture

Cohort Explorer remains the only client-facing application. Its browser/API contract does not change based on the selected DCR provider.

```text
Cohort Explorer frontend
          |
          v
Cohort Explorer FastAPI routes
          |
          v
      DcrBackend
       /      \
      v        v
Decentriq SDK  AadcrV2Backend --signed HTTP--> AADCR v2 FastAPI
                                             |       |       |
                                           SQLite  files  executor
```

The provider is selected with `DCR_BACKEND=decentriq` or `DCR_BACKEND=aadcrv2`; the default is `decentriq`. Production Compose explicitly selects `decentriq`, so enabling local AADCR can never happen implicitly.

The Cohort Explorer backend gains a small provider package with:

- a protocol and provider-neutral request/result models;
- a factory selected from settings;
- a Decentriq SDK adapter around the existing implementation; and
- an asynchronous AADCR v2 HTTP adapter built on the already installed `httpx` dependency.

The large existing `backend/src/decentriq.py` is not rewritten wholesale. Provider-facing route handlers are moved or wrapped incrementally while the SDK functions remain available behind `DecentriqBackend`. `backend/src/upload.py`, `backend/src/main.py`, and `backend/src/monitoring.py` must stop importing SDK functions as the only possible implementation.

### Rejected alternatives

1. A dedicated AADCR bulk/bootstrap endpoint was rejected because it would duplicate the native room-change and merge workflow and tightly couple Delta to one client.
2. Replacing the SDK implementation directly was rejected because it would remove production behavior and force the two backends to share one implementation shape.
3. A Decentriq-SDK compatibility proxy was rejected because emulating the SDK is larger and less testable than translating the small Cohort Explorer contract.

## 5. Provider contract and compatibility

The provider contract covers these behaviors:

- create a single-cohort provision room;
- preview/download a multi-cohort room definition;
- create a live multi-cohort compute room;
- list rooms visible to the current user;
- refresh room data and expose last-modified time;
- return an audit log;
- run/fetch computation results when supported; and
- expose a capability map for provider-specific features.

Existing HTTP paths remain stable, including:

- `POST /create-provision-dcr`
- `POST /get-compute-dcr-definition`
- `POST /create-live-compute-dcr`
- `GET /dcr-log/{dcr_id}`
- `GET /dcr-log-main/{dcr_id}`
- `GET /compute-get-output/{dcr_id}`
- `GET /shuffle-get-output/{dcr_id}`
- `GET /my-dcrs`
- `POST /my-dcrs/refresh`
- `GET /my-dcrs/last-modified`

The provider-neutral room record preserves the fields already consumed by Cohort Explorer: `id`, `title`, `description`, `createdAt`, `dcr_url`, `provider`, `owner.email`, participants, nodes, cohorts, and optional error information. The live-create response preserves `message`, `dcr_id`, `dcr_url`, `dcr_title`, `cohort_ids`, `num_cohorts`, all upload result maps/counts, and `participants`. `dcr_url` is built from an `AADCRV2_ROOM_URL_TEMPLATE`; its local default points to the AADCR API room resource and can later be changed to an AADCR frontend route without changing the response contract.

The existing Decentriq-specific refresh endpoint remains as a compatibility alias. New internal naming and documentation use provider-neutral terms.

When AADCR v2 is selected, the definition-download endpoint creates a deterministic ZIP containing `dcr_config.json`, selected metadata dictionaries, available shuffled samples, and selected mapping files. It does not instantiate or contact the Decentriq SDK.

## 6. AADCR v2 translation flow

For `POST /create-live-compute-dcr`, the adapter performs the following native API sequence:

1. Validate the wizard request and resolve cohort owners, selected variables, analysts, exclusions, mapping files, and available synthetic/demo assets.
2. Check the local operation journal by `session_id`. A completed operation returns the same result; an incomplete operation resumes from its last confirmed step.
3. `POST /api/dcr/` with the room name.
4. Add the creator, cohort owners, service identity, and requested analysts through the DEV participant route.
5. Add DEV FILE data nodes:
   - `<cohort>` for row-level data;
   - `<cohort>_metadata_dictionary` for its dictionary;
   - `<cohort>_shuffled_sample` when selected and available;
   - one node per selected mapping file; and
   - `CrossStudyMappings` when the upload slot is requested.
6. Add DEV computation nodes for deterministic metadata preview and aggregate summary statistics. Any local airlock-like computation is labelled as a simulation and emits aggregates only; it is not presented as a security boundary.
7. Add `DATA_OWNER` and `DATA_ANALYST` permissions from the resolved participant map.
8. Read the DEV view and collect every new `changeId`.
9. Submit one initial merge request using those change IDs.
10. Poll the merge request until it is `MERGED` or a bounded timeout expires. The current initial-room behavior should merge without an approval bypass because no prior PROD approval paths exist. If the backend returns pending approval, the adapter returns that structured state and never circumvents governance.
11. Read the PROD view and resolve immutable PROD node IDs by their expected names.
12. Upload metadata dictionaries, mapping files, and selected samples through `POST /api/upload`, then provision them to the matching PROD nodes.
13. In explicit synthetic-demo mode only, upload and provision the generated row-level CSV for each cohort. Normal operation continues to require the data owner to upload row-level data separately.
14. Persist the normalized response and return the existing Cohort Explorer live-create shape.

The same primitives implement the single-cohort provision-room route. The AADCR provider lists rooms with `GET /api/dcr/`, enriches each accessible room from its PROD view and the local operation journal, and converts audit entries to `{timestamp, user, desc}`.

### Idempotency and partial failure

A local JSONL operation journal under the configured Cohort Explorer data folder is keyed by `session_id`. It contains no tokens or row data. Each record tracks the AADCR room ID, completed steps, selected cohort IDs, normalized request metadata, and final response. A process-local lock plus atomic append prevents two requests for the same `session_id` from advancing concurrently in the supported single-worker demo configuration.

All outbound requests have explicit connect/read timeouts. A failure returns a provider-neutral error containing the failed step, upstream status/message, and created room ID when one exists. The adapter does not automatically delete a partially created room because deletion would destroy its audit trail and make debugging harder. Retrying the same session resumes instead of creating a duplicate.

## 7. Authentication, administration, and authorization

### Cohort Explorer local login

Settings gain:

- `LOCAL_AUTH_ENABLED=false`
- `LOCAL_AUTH_EMAIL=nikolas.molyndris@decentriq.ch`
- `SESSION_COOKIE_SECURE=true`

When and only when `DEV_MODE=true` and `LOCAL_AUTH_ENABLED=true`, `GET /login` issues the existing Cohort Explorer session JWT for the fixed configured email instead of redirecting to OAuth, sets the normal HTTP-only cookie, and redirects to `FRONTEND_URL`. The caller cannot supply a different email. Local HTTP sets `SESSION_COOKIE_SECURE=false`; production remains secure. The configured development email is appended to `admins_list` only in this guarded mode, so all existing admin checks recognize it without hard-coded addresses in application logic.

### Cohort Explorer to AADCR authentication

The browser session cookie is never forwarded to AADCR v2. The Cohort Explorer backend mints a short-lived signed JWT containing `sub`, `email`, `email_verified`, `iss`, `aud`, `iat`, and `exp`. Both services share a local inter-service secret supplied through environment variables. AADCR rejects unsigned tokens, `alg=none`, bad signatures, expired tokens, and wrong issuer/audience.

Relevant settings are:

- `AADCRV2_API_URL=http://aadcrv2:8000`
- `AADCRV2_JWT_SECRET`
- `AADCRV2_JWT_ISSUER=cohort-explorer-local`
- `AADCRV2_JWT_AUDIENCE=aadcrv2-local`
- `AADCRV2_TIMEOUT_SECONDS`
- `AADCRV2_ROOM_URL_TEMPLATE=http://localhost:8000/api/dcr/{dcr_id}`

No secret is committed. `.env.example` contains generated-placeholder instructions only.

### AADCR authorization

AADCR v2 gains a central access policy:

- creator or PROD participant may read the room and its views/audit information;
- only the creator may rename or delete the room;
- DEV, merge, provisioning, computation, and explanation operations require room membership plus the appropriate owner/analyst role;
- every node lookup verifies that the node belongs to the DCR named in the URL; and
- test/reset endpoints are registered only when explicitly enabled for tests.

This closes the current trivial cross-room access paths needed for a credible local multi-user demonstration. It does not constitute a complete production security review.

## 8. AADCR v2 local runtime hardening

The Delta branch receives these scoped corrections:

- central settings in `aadcrv2/config.py`;
- Python constrained to `>=3.11,<3.14`, plus a valid backend README/package configuration so `poetry install` succeeds;
- environment-configurable `DATABASE_URL` and results directory;
- public `/health` liveness/readiness response;
- explicit CORS origins rather than wildcard-with-credentials;
- signed JWT verification and centralized room/node access checks;
- upload filename, non-empty, UTF-8/CSV, and configurable size validation with `413` for oversized files;
- opt-in, idempotent sample seeding that never deletes and recreates a room on startup;
- a packaged computation script instead of the developer-specific absolute path;
- deterministic behavior when no Gemini key is configured: the core API starts normally and explanation endpoints return HTTP `503` with a clear disabled response;
- removal of the committed Azure configuration credential from the local setup guide, with rotation called out as an external owner action;
- a non-root backend Dockerfile, `.dockerignore`, safe `.env.example`, persistent data/result mounts, and healthcheck; and
- no Authorization header or secret-bearing configuration in request logs.

The computation executor still runs Python subprocesses and is not a confidential-computing sandbox. The local Compose network, non-root container, isolated volumes, absence of the Docker socket, bounded execution time, and explicit synthetic-only data reduce demo risk but do not change that limitation.

## 9. Synthetic-data contract

The generator uses only checked-in code, schemas, and variable mappings. It never reads live pages, live APIs, or patient rows.

Source grounding:

- dictionary validation in `backend/src/upload.py`;
- the 28-column Decentriq dictionary schema in `backend/src/decentriq.py`;
- the `Descriptions` parser in `backend/src/cohort_cache.py`; and
- `backend/CohortVarLinker/mapping_output/time-chf_gissi-hf_full.csv`.

The generator is `backend/scripts/seed_synthetic_data.py` and is invoked as:

```bash
uv run --project backend python backend/scripts/seed_synthetic_data.py \
  --seed 42 --rows 2500 --output data/synthetic-demo
```

It refuses a non-empty output directory unless `--force` is supplied. The demo backend runs with `DATA_FOLDER=data/synthetic-demo`, preventing collisions with normal local or live data.

Generated outputs are:

- `data/synthetic-demo/iCARE4CVD_Cohorts.xlsx`
- `data/synthetic-demo/cohorts/TIME-CHF/TIME-CHF_datadictionary.csv`
- `data/synthetic-demo/cohorts/GISSI-HF/GISSI-HF_datadictionary.csv`
- `data/synthetic-demo/dcr-input/TIME-CHF.csv`
- `data/synthetic-demo/dcr-input/GISSI-HF.csv`
- `data/synthetic-demo/manifest.json`

The manifest records the seed, generator version, row counts, source commit, mapping source, selected mapping rows, and SHA-256 hashes. Generated demo data is ignored by Git; the generator, parameter profile, schemas, and small test fixtures are committed.

### Dictionary schema

Each dictionary is UTF-8 without BOM and contains exactly these 28 columns in this order:

```text
VARIABLENAME,VARIABLELABEL,VARTYPE,UNITS,CATEGORICAL,MISSING,COUNT,NA,MIN,MAX,Formula,Categorical Value Concept Code,Categorical Value Concept Name,Categorical Value OMOP ID,Variable Concept Code,Variable Concept Name,Variable OMOP ID,Additional Context Concept Name,Additional Context Concept Code,Additional Context OMOP ID,Unit Concept Name,Unit Concept Code,Unit OMOP ID,Domain,Visits,Visit OMOP ID,Visit Concept Name,Visit Concept Code
```

Required row fields are `VARIABLENAME`, `VARIABLELABEL`, `VARTYPE`, `Domain`, and `Variable OMOP ID`. Types are `STR`, `FLOAT`, `INT`, or `DATETIME`. Domains use one accepted value. Concept IDs/codes are scalar; any pipe-separated concept triples have matching lengths. `COUNT`, `NA`, `MIN`, and `MAX` are calculated from the emitted rows.

The `Descriptions` workbook includes the parser-required study fields plus institute, population, duration, sex/age distribution, administrator/contact emails, location, language, dataset format, and coding system. Both cohorts use the configured local administrator for demo permissions.

### Variables and plausibility

The baseline uses the real mapped TIME-CHF/GISSI-HF variable pairs for patient ID, age, sex, diabetes, hypertension, smoking, systolic/diastolic pressure, heart rate, weight, height, ejection fraction, NT-proBNP, creatinine, haemoglobin, NYHA class, furosemide, spironolactone, and heart-failure hospitalization. It also includes mapped three-month and one-year measurements for pressure, weight, creatinine, NYHA, and furosemide dose.

Generation uses `numpy.random.Generator(PCG64(seed))` and stable SHA-256-derived sub-seeds, never Python's process-dependent `hash()`. Patient-level latent severity creates coherent directional relationships: higher severity raises NYHA, NT-proBNP, heart rate, diuretic exposure/dose, and hospitalization risk while lowering ejection fraction. Height and BMI determine weight; systolic and diastolic pressure share a latent component; follow-ups use patient-level autoregressive trajectories and monotone dropout; drug dose is zero exactly when exposure is absent. These are fixture rules, not epidemiological claims.

Categorical encodings are stable and documented: binary no/yes, female/male, never/former/current smoking, and NYHA I-IV. IDs are synthetic and unique. Float formatting, row/column order, line endings, XLSX properties, and ZIP timestamps are normalized so a fixed seed produces fixed hashes.

## 10. Local orchestration

Cohort Explorer owns the integration orchestration without copying Delta source. A Compose overlay accepts the sibling AADCR backend path through an environment variable and adds the `aadcrv2` service to the existing backend, frontend, and Oxigraph services.

Expected ports:

- Cohort Explorer frontend: `3001`
- Cohort Explorer backend: `3000`
- Oxigraph: `7878`
- AADCR v2 backend: `8000`

Convenience commands are exposed through a Makefile or scripts with these stable semantics:

- `demo-generate`: generate and validate the synthetic pack;
- `demo-up`: build/start the local stack with AADCR selected;
- `demo-seed`: upload the central workbook and both dictionaries through Cohort Explorer APIs;
- `demo-smoke`: exercise the complete API flow and verify results; and
- `demo-down`: stop services without deleting persisted data.

The production Compose file explicitly retains `DCR_BACKEND=decentriq`. AADCR local mode requires an intentional overlay/profile and cannot be selected by a missing environment value.

## 11. Testing and verification

### Cohort Explorer tests

Add focused tests under `backend/tests/` for:

- provider selection and default behavior;
- AADCR request translation with a mocked HTTP transport;
- operation-journal idempotency and resume;
- create/provision response compatibility;
- My DCRs and audit-log normalization;
- local login guards, fixed email, cookie security, and admin membership;
- deterministic synthetic generation, exact workbook/dictionary schema, production validator success, raw/dictionary column equality, type/category/range rules, mapping coverage, hashes, correlation directions, and longitudinal invariants; and
- upstream timeouts, partial errors, and pending-approval behavior.

The existing `backend/test_sparql_data.py` performs network I/O at import time and is not a safe unit-test entry point. New tests run with:

```bash
cd backend
uv sync --frozen --extra test
uv run pytest tests -q
uv run ruff check src tests
uv run ruff format --check src tests
```

### AADCR v2 tests

Add or update tests for:

- signed, tampered, expired, wrong-issuer/audience, and `alg=none` tokens;
- creator/participant/outsider authorization and cross-DCR node rejection;
- DEV graph creation, merge, PROD node resolution, upload, provisioning, execution, result retrieval, and audit logs;
- health, CORS, environment configuration, idempotent seed, and disabled explanation provider;
- upload filename/type/size/encoding boundaries; and
- test-route registration only in test mode.

Live-server computation tests are marked separately from hermetic TestClient tests. Baseline and integration commands use Python 3.11.

### Code-only end-to-end smoke test

The smoke test uses HTTP clients and generated data, not a browser:

1. wait for Oxigraph, Cohort Explorer, and AADCR health checks;
2. call local login and retain the Cohort Explorer cookie;
3. upload `iCARE4CVD_Cohorts.xlsx`;
4. upload both dictionaries through `POST /upload-cohort`;
5. submit the existing six-step wizard payload to `POST /create-live-compute-dcr`;
6. assert the normalized create response and My DCRs entry;
7. assert AADCR PROD participants, nodes, permissions, and provisioned synthetic datasets;
8. run the aggregate computation and validate the returned result ZIP; and
9. assert audit entries for creation, merge, provisioning, and execution.

No command contacts the live OAuth, Cohort Explorer, Decentriq, Athena, Gemini, or OpenAI endpoints. External concept validation is stubbed from checked-in mappings.

## 12. Known baseline debt

Baseline verification on the pinned commits found:

- Cohort Explorer dependency setup succeeds on Python 3.10, but bare `pytest` collects the network-active `test_sparql_data.py` and fails because no SPARQL service is running. The repository currently has no hermetic auth/DCR suite.
- AADCR v2 Poetry initially selects Python 3.14, which its locked `pydantic-core` cannot build; Python 3.11 succeeds.
- AADCR v2 `poetry install` then fails to install the root package because the declared backend README is absent; `poetry install --no-root` installs dependencies.
- AADCR v2's current full suite reports 148 passed and 24 failed. Failures include unmarked live-server tests on `127.0.0.1:8765`, missing audit-log tables in fixtures, dependency-view drift, dataset permission/cleanup cases, and explanation tests that require an unconfigured Gemini key.

The implementation must make all new and directly affected tests pass and classify the remaining unrelated baseline failures explicitly. It must not claim a green full baseline until the evidence supports that claim.

## 13. Acceptance criteria

The work is complete when all of the following are demonstrated from clean local checkouts:

1. Cohort Explorer defaults to its existing Decentriq provider and production Compose states that choice explicitly.
2. Local AADCR mode starts through documented commands without live credentials or committed secrets.
3. `nikolas.molyndris@decentriq.ch` receives a valid local session and administrator permissions only in guarded development mode.
4. The synthetic generator produces both cohorts deterministically and passes the production dictionary validator and plausibility tests.
5. The existing Cohort Explorer upload and DCR-creation endpoints work without a browser.
6. One wizard request creates a native AADCR DEV graph, initial merge request, PROD graph, participants, nodes, and permissions.
7. Both synthetic row-level datasets are uploaded and provisioned to the correct PROD nodes only in explicit demo mode.
8. An aggregate computation completes, its result is retrievable, and the audit trail records the flow.
9. Repeating the same `session_id` does not create a duplicate room.
10. Invalid/tampered tokens, outsider access, cross-DCR node use, and oversized uploads are rejected by tests.
11. No live/browser lane was used, no real patient rows were included, and no secret was committed.
12. Cohort Explorer and Delta changes are committed separately with their pinned cross-repository dependency documented.

## 14. Repository delivery

The canonical adapter, synthetic generator, orchestration, and user documentation live in the MaastrichtU-IDS Cohort Explorer branch and are suitable for an IDS pull request. AADCR runtime fixes remain a separate Delta branch based on `davstur/aadcrv2`; they are not vendored. If Delta push permission is unavailable, the local branch and its commit series remain the reproducible dependency, and the Cohort Explorer documentation records the exact base and required commits.

Live seeding and browser walkthrough are a later, explicit phase after the code-only local acceptance criteria pass.
