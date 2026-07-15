# Cohort Explorer to AADCR v2 Local Integration Design

Date: 2026-07-14

Revised: 2026-07-15

Status: Approved in conversation; local browser lane required during implementation and final verification

Local admin: `nikolas.molyndris@decentriq.ch`

## 1. Objective

Build a reproducible, code-first local demonstration in which Cohort Explorer can:

1. load two synthetic iCARE4CVD cohorts and their data dictionaries;
2. authenticate a fixed development administrator without contacting the live OAuth service;
3. exercise its current metadata lifecycle: validation, exploration, search/filter data, manual and generated mappings, mapping cache/logs, EDA assets, downloads, selection, and re-upload behavior;
4. replace only the external concept/model services needed by that lifecycle with deterministic local fixtures;
5. preserve its current DCR wizard and API response contracts;
6. create and configure an Advanced Analytics DCR v2 room through the AADCR v2 REST API;
7. upload and provision synthetic row-level datasets into the matching production data nodes;
8. run an aggregate computation and retrieve its result and audit trail; and
9. keep the existing live metadata providers and Decentriq SDK backend available as production defaults.

Implementation is test-first: hermetic unit/contract tests and the local API smoke flow establish deterministic state before UI assertions. A required browser lane then runs at integration checkpoints and as the final acceptance pass against the local Cohort Explorer at `http://localhost:3001`. It uses only the generated synthetic pack and local services, never the live Cohort Explorer or live Decentriq platform, and does not depend on the AADCR frontend.

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
- The current Cohort Explorer metadata lifecycle, backed by real local Oxigraph/filesystem state and deterministic replacements for external concept and model services.
- Fixture providers for concept search, concept validation, and automatic cross-study mapping that preserve the existing route and artifact contracts without model downloads, hosted vector databases, LLMs, or credentials.
- Deterministic local EDA and shuffled-sample assets derived from the synthetic rows.
- Docker and command-line orchestration, contract tests, a local API end-to-end smoke test, browser checkpoints during implementation, and a complete local browser acceptance run.
- Small provider-neutral text/link, provisioning/result-status, and stable-selector corrections where the current frontend hardcodes Decentriq or lacks a visible local-browser proof, without redesigning the UI.

### Excluded

- Any read from, write to, or calibration against the live Cohort Explorer or live Decentriq platform.
- Incorporating the divergent Copilot patch.
- Claiming that the standalone AADCR v2 Python executor provides production confidential-computing or clean-room isolation.
- Production TLS, secret rotation infrastructure, malware scanning, rate limiting, tamper-evident audit storage, SQLite multi-worker guarantees, or full arbitrary-code sandboxing.
- Rebuilding the AADCR v2 frontend.
- Redesigning Cohort Explorer metadata screens or adding metadata capabilities that do not exist on the pinned baseline.
- Treating Cohort Explorer's currently unused `cohort_data` form field as a row-level upload path. In demo mode, row-level synthetic CSVs are provisioned to AADCR separately from Cohort Explorer's metadata upload.
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

The provider-neutral room record preserves the fields already consumed by Cohort Explorer: `id`, `title`, `description`, `createdAt`, `dcr_url`, `provider`, `owner.email`, participants, nodes, cohorts, and optional error information. It adds optional provider-neutral capabilities and provisioned-dataset status so My DCRs can prove the local flow. The live-create response preserves `message`, `dcr_id`, `dcr_url`, `dcr_title`, `cohort_ids`, `num_cohorts`, all upload result maps/counts, and `participants`. `dcr_url` is built from an `AADCRV2_ROOM_URL_TEMPLATE`; its local default points to the AADCR API room resource and can later be changed to an AADCR frontend route without changing the response contract.

My DCRs receives a compact provider-neutral result panel rather than a new workflow. It shows provisioned dataset/node status and invokes the existing `GET /compute-get-output/{dcr_id}` contract when the selected provider advertises computation support. JSON aggregates render inline; ZIP results remain downloads and the panel reports their ready state, filename, and size. The browser suite validates the downloaded ZIP contents and hash.

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

The existing `/debug/permissions` route is registered only in development mode and requires the authenticated local administrator; it is not left as an unauthenticated production-readable permission dump.

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
  --seed 42 --rows 2500 --output data/synthetic-demo-pack
```

It refuses a non-empty output directory unless `--force` is supplied. Generated inputs and mutable application state are deliberately separate: the ignored, immutable pack is mounted read-only at `DEMO_PACK_DIR=/demo-pack`, while cohort files, Oxigraph-derived cache, mapping outputs, journals, and rooms use the browser/API namespace's fresh `DATA_FOLDER=/demo-runtime`. This prevents generated dictionaries from masquerading as UI uploads and makes `demo-browser-ready` capable of starting with zero runtime dictionaries.

Generated outputs are:

- `data/synthetic-demo-pack/iCARE4CVD_Cohorts.xlsx`
- `data/synthetic-demo-pack/cohorts/TIME-CHF/TIME-CHF_datadictionary.csv`
- `data/synthetic-demo-pack/cohorts/GISSI-HF/GISSI-HF_datadictionary.csv`
- `data/synthetic-demo-pack/dcr-input/TIME-CHF.csv`
- `data/synthetic-demo-pack/dcr-input/GISSI-HF.csv`
- `data/synthetic-demo-pack/dcr_output_TIME-CHF/eda_output_TIME-CHF.json`, per-variable PNGs, `shuffled_sample.csv`, and `shuffle_summary.txt`
- `data/synthetic-demo-pack/dcr_output_GISSI-HF/eda_output_GISSI-HF.json`, per-variable PNGs, `shuffled_sample.csv`, and `shuffle_summary.txt`
- `data/synthetic-demo-pack/manifest.json`

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

## 10. Metadata feature fidelity and offline providers

The local demonstration must exercise the metadata behavior already present on the pinned Cohort Explorer baseline. It is not enough to seed two dictionaries and jump directly to room creation. Oxigraph, the cohort cache, metadata files, mapping files, and activity logs remain real local state. Only dependencies that would otherwise leave the machine or download/run hosted models are replaced.

### Capability matrix

| Current capability | Existing contract/state | Local demonstration evidence |
| --- | --- | --- |
| Central cohort inventory | `POST /upload-cohorts-metadata`, `GET /cohorts-metadata`, and `GET /cohorts-metadata-sparql` | Upload the synthetic `Descriptions` workbook through the local API, assert full/summary views from Oxigraph/cache, and verify both cohort cards and their real synthetic counts in the browser. |
| Dictionary upload and re-upload | `POST /upload-cohort`; cohort files, base graph, separate mapping graph, and the existing upload page | Upload both valid dictionaries. Re-upload one with a changed label but stable variable names and prove its manual mapping survives. Submit one invalid re-upload and prove restoration. Exercise the visible upload/replace flow in a browser after correcting its current destructive “Validate” action. |
| Syntax and concept validation | `POST /metadata-syntax-issues-report`, `POST /validate-athena-codes/{cohort_id}`, and `POST /validate-athena-codes-all-summary` | Show a clean report for the valid pack and deterministic failures for a checked-in malformed dictionary and unknown concept. Concept validation uses the fixture provider in offline mode. |
| Metadata exploration | `GET /cohorts-metadata`, cohort/variable/category metadata, and the current client-side search/filter/equivalent-name utilities | Verify payloads through HTTP and predicates through unit tests, then exercise OR/AND/exact search, study/institute, OMOP-domain, type, category-count, visit, outcome, source, and equivalent-variable grouping in the browser. |
| Manual concept mapping | `GET /api/search-concepts` and `POST /insert-triples` | Search the local fixture catalog, map one variable and one categorical value, assert fresh metadata and local `used_by`, then repeat the same operations from the visible concept controls in the browser. Re-selecting a concept exercises overwrite behavior. |
| Automatic cross-study mapping | `/api/check-mapping-cache`, `/api/generate-mapping`, `/api/get-available-mapping-files`, `/api/get-cached-mapping-file/{filename}`, and `/api/mapping-activity-log` | Generate/retrieve/list/cache-check the fixture mapping through the API, verify its log, test table/graph projections, and use the mapping page in the browser to generate or load the same artifact and switch between table and graph views. |
| EDA and sample assets | `GET /cohort-eda-output/{cohort_name}`, `GET /api/compare-eda/{source_cohort}/{source_var}/{target_cohort}/{target_var}`, `GET /get-cohorts-with-shuffled-samples`, and `GET /get-shuffled-sample/{cohort_name}` | Generate deterministic EDA JSON/PNGs and shuffled samples, retrieve/compare them through HTTP, and open the corresponding local variable/EDA view in the browser. |
| Metadata export and cache | `GET /cohort-spreadsheet/{cohort_id}`, `GET /download-cohorts-metadata-spreadsheet`, `GET /api/download-cohorts-cache`, `POST /api/compare-cohorts-cache`, `POST /refresh-cache`, and `POST /clear-cache` | Download each cohort dictionary and the central cohort-inventory workbook, round-trip the cache comparison, and prove cache refresh reconstructs the same synthetic inventory. Cache clearing is tested against isolated fixture state rather than during the presentation path. |
| Selection and DCR handoff | Existing whole-cohort selection state and six-step DCR request contract | Add both cohorts, select the generated mapping, preview participants, and hand the request to AADCR through API tests and the complete visible browser wizard. The baseline contains unrendered per-variable helpers, so subset selection is not presented as an existing UI capability. |
| Guarded metadata administration | `POST /normalize-all-dictionary-headers`, `POST /get-logs`, and `POST /delete-cohort` | Preserve the routes and add authorization/regression tests. Destructive deletion uses an isolated scratch cohort and is not part of the two-cohort happy path. |

The search and filter behavior is currently client-side. The API smoke test proves that all required fields are present, frontend pure-function tests prove the predicates, and the browser lane proves that the real controls render and apply those predicates to the generated metadata.

### Baseline fidelity repairs included

Code inspection found several baseline defects that would make a successful-looking demo misleading. The implementation includes only the repairs needed to make the existing contracts behave consistently:

- use one canonical studies-metadata graph identifier and make central-workbook replacement transactional, preventing stale triples or a corrupt replacement from becoming the active source;
- extract dictionary validation from persistence and make the upload page's “Validate” action call a non-mutating validation route before its one intentional upload/replace request;
- hydrate variable and category manual mappings from the separate mapping graph into fresh metadata/cache responses, and make category subjects use the same URI scheme during dictionary import and manual mapping;
- update concept-search `used_by` enrichment to query the CMEO study/data-element vocabulary actually emitted by the uploader, rather than the legacy iCARE cohort/variable vocabulary;
- preserve the current mapping upsert/overwrite behavior; no mapping delete/history feature is added;
- centralize the configured mapping-output directory so the committed fixture, generation, cache check, list, retrieval, and DCR selection read the same location;
- make freshness authoritative: an outdated mapping cannot be reused merely because a filename exists, and available-file filtering honors the requested cohort IDs;
- make `GET /cohort-spreadsheet/{cohort_id}` select the canonical current `_datadictionary.csv`, never a timestamped backup or report;
- make concept-validation provider failures explicit and fail closed instead of treating an unavailable OMOP/Athena dependency as a successful validation;
- replace the hard-coded startup HTTP self-call for the syntax report with the underlying service function;
- avoid importing or refreshing Decentriq/model clients during startup when their providers are not selected; and
- make generated EDA links use a configured local/public API base URL instead of the hard-coded live Explorer hostname.

These are regression-tested repairs to routes and screens that already exist. The non-mutating validation route corrects the existing upload-screen contract; the work does not add mapping deletion, mapping history, per-variable DCR controls, or a new metadata workflow.

### Provider seams and safe defaults

External metadata behavior moves behind narrow backend interfaces rather than being monkeypatched at the route level:

- `ConceptSearchProvider.search(query, domains)` for Athena concept lookup;
- `ConceptValidationProvider.validate(dictionary)` for Athena/OMOP code validation; and
- `MappingGenerationProvider.generate(source, target, options)` for CohortVarLinker's embeddings, Qdrant, and optional LLM pipeline.

Settings are explicit and independently selectable:

```text
CONCEPT_SEARCH_BACKEND=athena
CONCEPT_VALIDATION_BACKEND=cohortvarlinker
MAPPING_GENERATION_BACKEND=cohortvarlinker
OFFLINE_DEMO=false
```

Those values preserve today's live behavior by default. The local Compose overlay sets the three backends to `fixture` and `OFFLINE_DEMO=true`. Offline startup fails fast if any external metadata backend is not a fixture backend. Live implementations are imported lazily; provider factories neither import modules with live-client side effects nor instantiate the Athena HTTP client, Hugging Face models, Qdrant client, or an LLM client in fixture mode. A network-denial test fails if a metadata route attempts outbound HTTP while the guard is active.

The live provider adapters are thin wrappers around the existing Athena and CohortVarLinker behavior. The route request/response schemas and normal cache/output locations do not change when a provider changes.

### Deterministic fixture contracts

Committed runtime fixtures live under `backend/demo/metadata-fixtures/`, separate from generated data and test-only fixtures.

The concept catalog is a small JSON document containing every concept used by the two synthetic dictionaries plus a few deterministic alternatives needed to demonstrate search and overwrite. It is derived from the committed mapping artifact and dictionary profile, not from live Athena. Search is case-insensitive, token-based, deterministically ranked, and respects the existing domain filters. Results preserve the current `id`, `uri`, `label`, `domain`, `vocabulary`, and `used_by` shape; `used_by` is still calculated from local Oxigraph. An unknown query returns an empty result.

The validation fixture indexes the same catalog by concept code and OMOP ID. A small checked-in OMOP relationship CSV is loaded through the real local graph-validation code; no Python-version-sensitive graph pickle is committed and no missing concept falls through to Athena. The provider runs the existing local syntax validation first, then returns the same validation result columns and status/reason semantics as the live validator. It accepts the generated dictionaries and rejects checked-in invalid-code, mismatched-code/ID, and malformed-category examples.

The mapping fixture is intentionally limited to the generated TIME-CHF/GISSI-HF pair. It materializes the known rows from `time-chf_gissi-hf_full.csv` through the existing `_full.csv`, JSON, cache, activity-log, and frontend table/graph parser contracts. It never guesses a new mapping. An unsupported cohort pair returns a clear demo-provider error while live mode remains unchanged. A sidecar records `provider=fixture`, `synthetic=true`, source commit, source-artifact hash, request parameters, and output hashes so a fixture result cannot be mistaken for model output.

The synthetic generator also writes deterministic EDA JSON, per-variable PNGs, combined comparison inputs, and shuffled samples into the filenames/directories already consumed by the current endpoints. These are computed locally from the generated synthetic rows; no EDA, ontology, embedding, vector-database, or LLM service is contacted.

### Deterministic metadata/API sequence

The repeatable foundation flow is:

1. authenticate the guarded local administrator and upload the central workbook;
2. upload the valid TIME-CHF and GISSI-HF dictionaries and assert full/summary inventory state;
3. run syntax plus single/all-cohort concept validation against the fixture catalog;
4. exercise the checked-in invalid upload and verify transactional restoration;
5. search fixture concepts, add a variable mapping and category mapping, and verify them from freshly read metadata;
6. re-upload the valid dictionary with one non-key label change and verify that its separate mapping graph is preserved;
7. generate, list, retrieve, and cache-check the fixture cross-study mapping and inspect its activity/sidecar provenance;
8. retrieve both cohort dictionaries, the central inventory workbook, cache export/comparison, shuffled samples, EDA JSON, and one cross-cohort EDA image;
9. run the frontend search/filter and equivalent-variable grouping suite against the same fixture metadata; and
10. add both cohorts through the current whole-cohort selection semantics, select the mapping artifact, preview the existing six-step request, and continue into the AADCR v2 creation/provision/computation flow.

The browser lane uses the same generated synthetic pack and contracts rather than a different fixture or live seed. Incremental checkpoints may reuse API-seeded local state, but the final browser run uses an isolated clean demo namespace so prior API-smoke mappings, rooms, or cookies cannot mask a UI defect.

## 11. Local orchestration

Cohort Explorer owns the integration orchestration without copying Delta source. A Compose overlay accepts the sibling AADCR backend path through an environment variable and adds the `aadcrv2` service to the existing backend, frontend, and Oxigraph services. The overlay pins the Oxigraph image by version/digest rather than using `latest` and selects all three fixture metadata providers explicitly. It mounts the generated pack read-only at `/demo-pack` and Compose-project-scoped mutable volumes at `/demo-runtime`; `DATA_FOLDER`, mapping output, cache, journals, and Oxigraph state never point into the pack. Server-side Next.js proxy calls use `INTERNAL_API_URL=http://backend:80`, while browser calls retain `http://localhost:3000`. Readiness waits for Cohort Explorer backend/frontend, Oxigraph, and AADCR probes rather than open ports alone.

Expected ports:

- Cohort Explorer frontend: `3001`
- Cohort Explorer backend: `3000`
- Oxigraph: `7878`
- AADCR v2 backend: `8000`

Convenience commands are exposed through a Makefile or scripts with these stable semantics:

- `demo-generate`: generate and validate the synthetic pack;
- `demo-up`: build/start the local stack with AADCR and offline metadata fixtures selected;
- `demo-seed`: upload the central workbook and both dictionaries through Cohort Explorer APIs and refresh their metadata/cache state; it does not pre-populate the runtime mapping cache;
- `demo-smoke`: exercise the complete API flow and verify results; and
- `demo-browser-ready`: reset/start an isolated browser-demo namespace, generate the pack, upload only the API-only central inventory workbook, wait for local health, and print the frontend URL plus fixed administrator identity; dictionaries, mappings, and rooms remain unseeded for the browser journey; and
- `demo-down`: stop services without deleting persisted data.

The production Compose file explicitly retains `DCR_BACKEND=decentriq`. AADCR local mode requires an intentional overlay/profile and cannot be selected by a missing environment value.

## 12. Testing and verification

### Cohort Explorer tests

Add focused tests under `backend/tests/` for:

- provider selection and default behavior;
- live-default and offline-fixture selection for all three metadata provider seams;
- fixture concept search ranking/domain filters/empty results and CMEO-backed local `used_by` enrichment;
- fixture concept validation success and invalid-code, mismatched-code/ID, and malformed-category failures;
- fixture mapping artifact, cache invalidation, sidecar hashes, activity log, and unsupported-pair error;
- valid upload, invalid re-upload rollback, canonical study-graph replacement, mapping projection/preservation across re-upload, metadata summaries, cache refresh, exact dictionary downloads, EDA, and shuffled samples;
- non-mutating dictionary validation followed by exactly one upload/replace mutation;
- mapping-directory consistency, cohort filtering, freshness, canonical dictionary selection, category-URI alignment, explicit provider failures, and local EDA URLs;
- a no-socket unit/import suite for the fixture providers and an external-egress denial assertion covering the composed metadata flow;
- AADCR request translation with a mocked HTTP transport;
- operation-journal idempotency and resume;
- create/provision response compatibility;
- My DCRs and audit-log normalization;
- local login guards, fixed email, cookie security, admin membership, and development-only authenticated permission debugging;
- deterministic synthetic generation, exact workbook/dictionary schema, local schema/content plus fixture-concept validation success, raw/dictionary column equality, type/category/range rules, mapping coverage, hashes, correlation directions, and longitudinal invariants; and
- upstream timeouts, partial errors, and pending-approval behavior.

Add focused frontend tests for the existing search modes, cohort/variable/category filter predicates, equivalent-variable grouping by standard code/OMOP ID, and mapping table/graph projections using the generated metadata/mapping fixtures. If a predicate, grouping, or projection function is currently embedded in a component, extract it into a pure utility without changing UI behavior. The frontend production build must still pass.

Fixture-provider unit and import tests allow no socket at all. Compose integration tests allow only the explicitly named local services (`backend`, `db`, `aadcrv2`, and loopback equivalents) and fail any public-network connection. The demo environment also sets Hugging Face/Transformers offline flags and leaves every model/provider credential unset.

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

### API end-to-end smoke test

The API smoke test uses HTTP clients and generated data so backend failures are diagnosed independently of UI state:

1. wait for Oxigraph, Cohort Explorer, and AADCR health checks;
2. call local login and retain the Cohort Explorer cookie;
3. upload `iCARE4CVD_Cohorts.xlsx`;
4. upload both dictionaries through `POST /upload-cohort`;
5. assert full/summary metadata, syntax validation, fixture concept validation, and download/cache contracts;
6. exercise concept search, variable/category mapping, mapping preservation on re-upload, fixture mapping generation/cache/logs, shuffled samples, and EDA retrieval;
7. submit the existing six-step wizard payload with both whole-cohort selections and the mapping file to `POST /create-live-compute-dcr`;
8. assert the normalized create response and My DCRs entry;
9. assert AADCR PROD participants, nodes, permissions, and provisioned synthetic datasets;
10. run the aggregate computation and validate the returned result ZIP; and
11. assert audit entries for creation, merge, provisioning, and execution.

The runtime smoke flow contacts none of the live OAuth, Cohort Explorer, Decentriq, Athena, Hugging Face, Qdrant, Together, OpenRouter, Gemini, Anthropic, Groq, Ollama, or OpenAI endpoints. External concept search, concept validation, and model-based mapping are served by the checked-in deterministic providers. Initial installation may still use ordinary package/image registries; runtime application behavior requires no hosted model or service credential.

### Required browser lane

Browser verification is a required implementation lane, not a post-project optional walkthrough. It targets only `http://localhost:3001` and the local services prepared by `demo-browser-ready`. The live Explorer and live Decentriq platform are never opened or mutated.

Run these browser checkpoints as their dependencies land:

1. after local auth, synthetic generation, and metadata seeding: log in as `nikolas.molyndris@decentriq.ch`, confirm administrator state, cohort cards/counts, dictionary details, and the upload/replace validation flow;
2. after metadata providers and fidelity repairs: exercise cohort/variable search modes and filters, equivalent-variable grouping, variable/category concept mapping, generated-mapping cache/table/graph/log views, EDA, and one metadata download;
3. after the AADCR adapter: add both cohorts, complete every visible step of the six-step wizard, create the room, and confirm the normalized success state, My DCRs entry, participants, audit log, provisioned status, and aggregate result from Cohort Explorer; and
4. after all automated suites pass: reset to clean synthetic demo volumes and repeat the complete browser journey without API shortcuts beyond the API-only central-workbook seed.

The upload page receives stable `data-testid` attributes only where semantic labels are not unique enough for durable automation. Browser actions use DOM-backed locators, assert the authoritative visible state after every mutation, and inspect console errors after each checkpoint. Final evidence includes screenshots of the Explorer inventory, a mapped variable/category, mapping graph/table, wizard review, creation success, My DCRs, and result/audit views. Generated browser evidence lives under an ignored `artifacts/browser/` directory and is summarized in the verification handoff; it is not committed unless explicitly requested.

The committed browser runbook distinguishes API-only setup from browser behavior. Central workbook upload remains API-only because the baseline has no corresponding screen. Dictionary upload/re-upload, concept mapping, exploration, mapping views, DCR wizard, and My DCRs are browser-verified. The browser lane supplements rather than replaces backend, frontend, and API tests.

## 13. Known baseline debt

Baseline verification on the pinned commits found:

- Cohort Explorer dependency setup succeeds on Python 3.10, but bare `pytest` collects the network-active `test_sparql_data.py` and fails because no SPARQL service is running. The repository currently has no hermetic auth/DCR suite.
- The upload screen's “Validate Dictionary” action calls the mutating upload route, and final submission calls it again; there is no dry-run per-file endpoint. Section 10 scopes the non-mutating validation repair required before browser acceptance.
- Manual mappings are not reliably projected back into refreshed metadata, category mapping subjects use a different URI shape from imported categories, the tracked sample mapping is outside the default runtime mapping directory, and mapping freshness/list filtering is inconsistent. Section 10 scopes the corresponding fidelity repairs.
- Per-variable DCR helper functions exist but are not rendered in the current variable list; the current visible selection path is whole-cohort selection. The demo does not overstate this boundary.
- AADCR v2 Poetry initially selects Python 3.14, which its locked `pydantic-core` cannot build; Python 3.11 succeeds.
- AADCR v2 `poetry install` then fails to install the root package because the declared backend README is absent; `poetry install --no-root` installs dependencies.
- AADCR v2's current full suite reports 148 passed and 24 failed. Failures include unmarked live-server tests on `127.0.0.1:8765`, missing audit-log tables in fixtures, dependency-view drift, dataset permission/cleanup cases, and explanation tests that require an unconfigured Gemini key.

The implementation must make all new and directly affected tests pass and classify the remaining unrelated baseline failures explicitly. It must not claim a green full baseline until the evidence supports that claim.

## 14. Acceptance criteria

The work is complete when all of the following are demonstrated from clean local checkouts:

1. Cohort Explorer defaults to its existing Decentriq provider and production Compose states that choice explicitly.
2. Local AADCR mode starts through documented commands without live credentials or committed secrets.
3. `nikolas.molyndris@decentriq.ch` receives a valid local session and administrator permissions only in guarded development mode.
4. The synthetic generator produces both cohorts deterministically and passes the existing local schema/content validator, the explicit fixture concept validator, and plausibility tests.
5. The full metadata capability matrix is demonstrated through existing APIs, focused frontend tests, and the required local browser checkpoints without introducing a new metadata UI.
6. Concept search, concept validation, and generated mapping work locally through deterministic fixtures with exact route/artifact compatibility and explicit fixture provenance.
7. Manual variable/category mappings are visible after a fresh metadata read; a valid dictionary re-upload preserves them, and an invalid re-upload restores the prior dictionary and graph.
8. EDA, shuffled samples, spreadsheets, cache export/comparison, mapping cache/retrieval/logs/table/graph projections, and metadata selection all work from the synthetic pack.
9. Offline mode starts only with fixture metadata providers; fixture unit/import tests allow no socket, and the composed test proves that its metadata flow makes no public-network request or model download.
10. The existing Cohort Explorer upload and DCR-creation endpoints work independently in the API lane and through their corresponding local browser controls.
11. One wizard request creates a native AADCR DEV graph, initial merge request, PROD graph, participants, nodes, and permissions.
12. Both synthetic row-level datasets are uploaded and provisioned to the correct PROD nodes only in explicit demo mode; Cohort Explorer does not claim to ingest raw participant rows through its unused form field.
13. An aggregate computation completes, its result is retrievable, and the audit trail records the flow.
14. Repeating the same `session_id` does not create a duplicate room.
15. Invalid/tampered tokens, outsider access, cross-DCR node use, and oversized uploads are rejected by tests.
16. The complete local browser journey passes after a clean synthetic reset, with the required screenshots and no unexplained browser-console errors.
17. No live Cohort Explorer or live Decentriq state was read or written, no real patient rows were included, and no secret was committed.
18. Cohort Explorer and Delta changes are committed separately with their pinned cross-repository dependency documented.

## 15. Repository delivery

The canonical adapter, synthetic generator, orchestration, and user documentation live in the MaastrichtU-IDS Cohort Explorer branch and are suitable for an IDS pull request. AADCR runtime fixes remain a separate Delta branch based on `davstur/aadcrv2`; they are not vendored. If Delta push permission is unavailable, the local branch and its commit series remain the reproducible dependency, and the Cohort Explorer documentation records the exact base and required commits.

The local browser lane is part of implementation and acceptance. Any future live seeding or live-environment walkthrough remains a separate, explicit phase after all local API and browser criteria pass.
