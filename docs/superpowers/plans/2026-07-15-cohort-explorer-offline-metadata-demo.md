# Cohort Explorer Offline Metadata Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Cohort Explorer's existing metadata lifecycle reproducible locally with a fixed administrator, deterministic synthetic cohorts, and fixture-backed external concept/mapping services while preserving live defaults.

**Architecture:** Local Oxigraph, files, cache, mapping artifacts, EDA, and shuffled samples remain real. Narrow provider interfaces replace Athena, OMOP fallback, Qdrant, embeddings, and LLMs only when the local demo selects fixture backends. Existing routes and UI behavior are preserved, with fidelity repairs where the current implementation contradicts its own visible contract.

**Tech Stack:** Python 3.11, FastAPI, pandas/openpyxl, RDF/Oxigraph, pytest/ruff, Next.js 14, React 18, TypeScript, Vitest.

## Global Constraints

- Work only in `/Users/nikolasmolyndris/projects/cohort-explorer-aadcrv2` on `codex/aadcrv2-local-integration`.
- Baseline is MaastrichtU-IDS Cohort Explorer `8aed288b18b04da3cf7d8f9c76bcb5d04467dace`.
- Local administrator is exactly `nikolas.molyndris@decentriq.ch`; no request parameter may choose another local identity.
- Production defaults remain `DCR_BACKEND=decentriq`, `CONCEPT_SEARCH_BACKEND=athena`, and both CohortVarLinker live providers.
- `OFFLINE_DEMO=true` must fail startup unless all external metadata providers are fixtures.
- Cohort Explorer remains metadata-only; generated participant rows are not accepted through its unused `cohort_data` form field.
- The local demo must not contact live OAuth, Cohort Explorer, Decentriq, Athena, model, vector-database, or LLM endpoints.
- Generated data lives under an isolated demo data folder and is ignored by Git; fixture definitions, tests, and source mapping provenance are committed.
- Browser checkpoints target only `http://localhost:3001`; live browser state is out of scope.
- Do not run bare `pytest` from `backend`; `backend/test_sparql_data.py` performs SPARQL I/O at collection time.
- Use TDD for every behavior change and commit each completed task separately.

## File Responsibility Map

- `backend/src/config.py`: environment-backed application settings and safe defaults.
- `backend/src/auth.py`, `backend/src/admin.py`, `backend/src/main.py`: guarded local login and admin/debug registration.
- `backend/src/metadata_providers/`: live/fixture provider protocols, factories, and implementations.
- `backend/src/metadata_paths.py`, `metadata_reports.py`, `metadata_mappings.py`, `concept_usage.py`, `mapping_artifacts.py`, `triplestore.py`: focused metadata fidelity services.
- `backend/src/upload.py`, `mapping.py`, `cohort_cache.py`, `explore.py`: existing route adapters that delegate to those services.
- `backend/src/demo/` and `backend/scripts/seed_synthetic_data.py`: deterministic synthetic pack.
- `backend/demo/metadata-fixtures/`: checked-in concept, OMOP relationship, and mapping fixture definitions.
- `frontend/src/utils/`: extracted search/filter/equivalence/mapping projection functions.
- `frontend/src/pages/` and `frontend/src/components/`: existing UI plus stable selectors and corrected validation request.
- `backend/tests/` and `frontend/src/utils/__tests__/`: hermetic behavior tests.

---

### Task 1: Settings, test foundation, and guarded local administrator

**Files:**
- Modify: `backend/src/config.py`
- Modify: `backend/src/auth.py`
- Modify: `backend/src/admin.py`
- Modify: `backend/src/main.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_config.py`
- Create: `backend/tests/test_auth_local.py`

**Interfaces:**
- Consumes: existing `Settings`, `create_access_token()`, `get_current_user()`, and `admins_list`.
- Produces: `Settings.validate_runtime()`, guarded local `/login`, and development-only authenticated `/debug/permissions`.

- [ ] **Step 1: Write failing settings and local-login tests**

```python
def test_offline_demo_rejects_live_metadata_provider(settings_factory):
    settings = settings_factory(
        dev_mode=True,
        offline_demo=True,
        concept_search_backend="athena",
        concept_validation_backend="fixture",
        mapping_generation_backend="fixture",
    )
    with pytest.raises(ValueError, match="fixture metadata providers"):
        settings.validate_runtime()


def test_local_login_uses_only_configured_admin(client, local_settings):
    response = client.get("/login", follow_redirects=False)
    assert response.status_code == 307
    token = response.cookies["token"]
    payload = decode_test_session(token, local_settings.jwt_secret)
    assert payload["email"] == "nikolas.molyndris@decentriq.ch"
    assert "nikolas.molyndris@decentriq.ch" in local_settings.admins_list
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cd backend && uv run pytest tests/test_config.py tests/test_auth_local.py -q`

Expected: failures for missing settings, `validate_runtime()`, and guarded local-login behavior.

- [ ] **Step 3: Add explicit settings and runtime validation**

```python
@dataclass
class Settings:
    local_auth_enabled: bool = field(default_factory=lambda: env_bool("LOCAL_AUTH_ENABLED", False))
    local_auth_email: str = field(
        default_factory=lambda: os.getenv("LOCAL_AUTH_EMAIL", "nikolas.molyndris@decentriq.ch").strip().lower()
    )
    session_cookie_secure: bool = field(default_factory=lambda: env_bool("SESSION_COOKIE_SECURE", True))
    concept_search_backend: str = field(default_factory=lambda: os.getenv("CONCEPT_SEARCH_BACKEND", "athena"))
    concept_validation_backend: str = field(
        default_factory=lambda: os.getenv("CONCEPT_VALIDATION_BACKEND", "cohortvarlinker")
    )
    mapping_generation_backend: str = field(
        default_factory=lambda: os.getenv("MAPPING_GENERATION_BACKEND", "cohortvarlinker")
    )
    offline_demo: bool = field(default_factory=lambda: env_bool("OFFLINE_DEMO", False))
    mapping_output_dir: str = field(
        default_factory=lambda: os.getenv(
            "MAPPING_OUTPUT_DIR",
            str(Path(__file__).resolve().parents[1] / "CohortVarLinker" / "data" / "mapping_output"),
        )
    )
    demo_pack_dir: str = field(
        default_factory=lambda: os.getenv("DEMO_PACK_DIR", "../data/synthetic-demo-pack")
    )
    public_api_url: str = field(default_factory=lambda: os.getenv("PUBLIC_API_URL", "http://localhost:3000"))

    def validate_runtime(self) -> None:
        providers = (
            self.concept_search_backend,
            self.concept_validation_backend,
            self.mapping_generation_backend,
        )
        if self.offline_demo and any(provider != "fixture" for provider in providers):
            raise ValueError("OFFLINE_DEMO requires fixture metadata providers")
        if self.local_auth_enabled and not self.dev_mode:
            raise ValueError("LOCAL_AUTH_ENABLED requires DEV_MODE=true")
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET is required")
```

Remove the committed usable `JWT_SECRET` fallback from `Settings`; tests inject an ephemeral value and local orchestration generates one into ignored runtime state.

- [ ] **Step 4: Implement guarded login and debug-route registration**

```python
@router.get("/login")
async def login() -> Response:
    if settings.dev_mode and settings.local_auth_enabled:
        expires = datetime.now(timezone.utc) + timedelta(hours=8)
        token = create_access_token(
            {"email": settings.local_auth_email, "access_token": "local-demo"},
            int(expires.timestamp()),
        )
        response = RedirectResponse(settings.frontend_url)
        response.set_cookie(
            "token",
            token,
            httponly=True,
            secure=settings.session_cookie_secure,
            samesite="lax",
        )
        return response
    return RedirectResponse(build_authorization_url())
```

Register `debug_router` only when `settings.dev_mode`; its handler depends on `get_current_user` and `_require_admin`.

- [ ] **Step 5: Run auth/config tests and formatting**

Run: `cd backend && uv run pytest tests/test_config.py tests/test_auth_local.py -q && uv run ruff check src tests`

Expected: all focused tests pass; Ruff reports no errors in touched files.

- [ ] **Step 6: Commit**

```bash
git add backend/src/config.py backend/src/auth.py backend/src/admin.py backend/src/main.py \
  backend/tests/conftest.py backend/tests/test_config.py backend/tests/test_auth_local.py
git commit -m "feat: add guarded local admin login"
```

### Task 2: Metadata provider contracts and fixture concept search

**Files:**
- Create: `backend/src/metadata_providers/__init__.py`
- Create: `backend/src/metadata_providers/contracts.py`
- Create: `backend/src/metadata_providers/factory.py`
- Create: `backend/src/metadata_providers/athena_search.py`
- Create: `backend/src/metadata_providers/cohortvarlinker_validation.py`
- Create: `backend/src/metadata_providers/cohortvarlinker_mapping.py`
- Create: `backend/src/metadata_providers/fixture_catalog.py`
- Create: `backend/src/concept_usage.py`
- Create: `backend/demo/metadata-fixtures/concepts.json`
- Create: `backend/tests/test_metadata_provider_factory.py`
- Create: `backend/tests/test_fixture_concept_search.py`
- Modify: `backend/src/mapping.py`

**Interfaces:**
- Consumes: provider selector settings and local `run_query()`.
- Produces: `ConceptResult`, `ConceptSearchProvider`, `ConceptValidationProvider`, `MappingGenerationProvider`, and lazy provider factories.

- [ ] **Step 1: Write failing provider/factory tests**

```python
@pytest.mark.anyio
async def test_fixture_search_is_ranked_filtered_and_enriched(fixture_search_provider):
    results = await fixture_search_provider.search("heart rate", ["Measurement"])
    assert [item.id for item in results][:1] == ["snomedct:364075005"]
    assert results[0].domain == "Measurement"
    assert results[0].used_by == [{
        "cohort_id": "TIME-CHF",
        "var_name": "heart_rate",
        "var_label": "Heart rate",
        "omop_domain": "Measurement",
    }]


def test_live_search_provider_is_not_imported_in_fixture_mode(monkeypatch, fixture_settings):
    monkeypatch.setitem(sys.modules, "src.metadata_providers.athena_search", ImportBomb())
    provider = get_concept_search_provider(fixture_settings)
    assert provider.__class__.__name__ == "FixtureConceptSearchProvider"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd backend && uv run pytest tests/test_metadata_provider_factory.py tests/test_fixture_concept_search.py -q`

Expected: import errors for the not-yet-created contracts/factory.

- [ ] **Step 3: Implement typed contracts and lazy factories**

```python
@dataclass(frozen=True)
class ConceptResult:
    id: str
    uri: str
    label: str
    domain: str
    vocabulary: str
    used_by: list[dict[str, str]]


class ConceptSearchProvider(Protocol):
    async def search(self, query: str, domains: Sequence[str]) -> list[ConceptResult]: ...


class ConceptValidationProvider(Protocol):
    def validate(self, dictionary_path: Path, report_path: Path) -> bool: ...


class MappingGenerationProvider(Protocol):
    def generate(
        self, source_study: str, target_studies: Sequence[tuple[str, bool]]
    ) -> MappingGenerationResult: ...
```

Factories import the live Athena/CohortVarLinker adapters only inside the selected live branch.

- [ ] **Step 4: Implement deterministic search and CMEO `used_by` enrichment**

Load `concepts.json` once, score exact phrase before all-token before any-token matches, then sort by `(rank, label.casefold(), id)`. Query the CMEO study/data-element graphs emitted by `upload.py`, not legacy `icare:Cohort` triples. Preserve route output keys `id`, `uri`, `label`, `domain`, `vocabulary`, and `used_by`.

- [ ] **Step 5: Route `/api/search-concepts` through the provider**

```python
@router.get("/search-concepts")
async def search_concepts(
    query: str,
    domain: list[str] | None = Query(default=None),
    user: Any = Depends(get_current_user),
):
    provider = get_concept_search_provider(settings)
    return [asdict(item) for item in await provider.search(query, domain or [])]
```

- [ ] **Step 6: Run focused tests and the no-network import test**

Run: `cd backend && uv run pytest tests/test_metadata_provider_factory.py tests/test_fixture_concept_search.py -q`

Expected: all pass without constructing `requests.Session`, `QdrantClient`, `AutoModel`, or `AutoTokenizer`.

- [ ] **Step 7: Commit**

```bash
git add backend/src/metadata_providers/__init__.py backend/src/metadata_providers/contracts.py \
  backend/src/metadata_providers/factory.py backend/src/metadata_providers/athena_search.py \
  backend/src/metadata_providers/cohortvarlinker_validation.py \
  backend/src/metadata_providers/cohortvarlinker_mapping.py \
  backend/src/metadata_providers/fixture_catalog.py backend/src/concept_usage.py \
  backend/src/mapping.py backend/demo/metadata-fixtures/concepts.json \
  backend/tests/test_metadata_provider_factory.py backend/tests/test_fixture_concept_search.py
git commit -m "feat: add offline metadata provider boundary"
```

### Task 3: Non-mutating validation and transactional metadata replacement

**Files:**
- Create: `backend/src/dictionary_validation.py`
- Create: `backend/src/metadata_paths.py`
- Create: `backend/src/metadata_reports.py`
- Create: `backend/src/triplestore.py`
- Create: `backend/src/metadata_providers/fixture_validation.py`
- Create: `backend/demo/metadata-fixtures/concept_relationship_enriched.csv`
- Modify: `backend/src/upload.py`
- Modify: `backend/src/explore.py`
- Modify: `backend/src/main.py`
- Modify: `frontend/src/pages/upload.tsx`
- Create: `backend/tests/test_fixture_concept_validation.py`
- Create: `backend/tests/test_metadata_fidelity.py`

**Interfaces:**
- Consumes: `ConceptValidationProvider`, current dictionary parser, graph publisher, cache builder.
- Produces: `validate_dictionary_upload()`, `replace_metadata_transactionally()`, `POST /validate-cohort-dictionary`, and canonical dictionary lookup.

- [ ] **Step 1: Write failing validation/rollback tests**

```python
def test_validate_route_does_not_mutate_any_metadata_state(
    auth_client, valid_dictionary, metadata_state_snapshot
):
    before = metadata_state_snapshot()
    response = auth_client.post(
        "/validate-cohort-dictionary",
        data={"cohort_id": "TIME-CHF"},
        files={"cohort_dictionary": ("dictionary.csv", valid_dictionary, "text/csv")},
    )
    assert response.status_code == 200
    assert metadata_state_snapshot() == before


def test_invalid_reupload_restores_file_graph_and_cache(seed_cohort, auth_client, invalid_dictionary):
    before = seed_cohort.snapshot()
    response = auth_client.post("/upload-cohort", data={"cohort_id": "TIME-CHF"}, files=invalid_dictionary)
    assert response.status_code == 422
    assert seed_cohort.snapshot() == before


def test_invalid_central_workbook_restores_file_graph_and_cache(
    seeded_inventory, auth_client, invalid_workbook
):
    before = seeded_inventory.snapshot()
    response = auth_client.post("/upload-cohorts-metadata", files=invalid_workbook)
    assert response.status_code == 422
    assert seeded_inventory.snapshot() == before
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cd backend && uv run pytest tests/test_fixture_concept_validation.py tests/test_metadata_fidelity.py -q`

Expected: missing route/service failures and proof that current validation mutates state.

- [ ] **Step 3: Implement the fixture concept validator and route both existing reports**

Load the committed concept catalog and OMOP relationship CSV, run the existing local syntax checks first, and validate variable, category, context, unit, and visit code/OMOP-ID pairs without a fallback network request. Route both `POST /validate-athena-codes/{cohort_id}` and `POST /validate-athena-codes-all-summary` through `get_concept_validation_provider(settings)` while preserving their report columns/status/reason shapes. Missing concepts and provider failures fail closed.

- [ ] **Step 4: Extract upload validation into a pure service**

```python
@dataclass(frozen=True)
class DictionaryValidationResult:
    cohort_id: str
    normalized_csv: bytes
    syntax_issues: tuple[str, ...]
    concepts_valid: bool


def validate_dictionary_upload(
    cohort_id: str,
    content: bytes,
    concept_provider: ConceptValidationProvider,
    work_dir: Path,
) -> DictionaryValidationResult:
    text = content.decode("utf-8")
    normalized = normalize_dictionary_headers(text)
    issues = tuple(validate_dictionary_schema(normalized))
    if issues:
        raise InvalidDictionary(issues)
    concepts_valid = concept_provider.validate(write_temp_csv(normalized, work_dir), work_dir / "report.csv")
    if not concepts_valid:
        raise InvalidDictionary(("Concept validation failed",))
    return DictionaryValidationResult(cohort_id, normalized.encode(), (), True)
```

- [ ] **Step 5: Make upload replacement transactional**

Stage the new file and RDF graph in a temporary directory. Only after validation succeeds, atomically replace the canonical `<cohort>_datadictionary.csv`, canonical named graph, and cache. On any exception, restore the prior file bytes, graph serialization, and cache snapshot.

- [ ] **Step 6: Correct the upload page contract**

Change `validateDictionary()` to call `/validate-cohort-dictionary`; keep `uploadCohort()` as the single call to `/upload-cohort`. Add stable IDs `upload-cohort-select`, `metadata-file`, `validate-dictionary`, `validation-results`, and `upload-dictionary`.

- [ ] **Step 7: Use canonical graph/path/report services everywhere**

Use only `https://w3id.org/CMEO/graph/studies_metadata`. `GET /cohort-spreadsheet/{cohort_id}` returns the exact canonical dictionary. Startup calls `metadata_reports.build_syntax_report()` directly instead of posting to localhost.

- [ ] **Step 8: Run backend tests and frontend build**

Run: `cd backend && uv run pytest tests/test_fixture_concept_validation.py tests/test_metadata_fidelity.py -q`

Run: `cd frontend && npm ci && npm run lint && npm run build`

Expected: validation is non-mutating, rollback snapshots match, and the frontend builds.

- [ ] **Step 9: Commit**

```bash
git add backend/src/dictionary_validation.py backend/src/metadata_paths.py \
  backend/src/metadata_reports.py backend/src/triplestore.py \
  backend/src/metadata_providers/fixture_validation.py backend/src/upload.py \
  backend/src/explore.py backend/src/main.py \
  backend/demo/metadata-fixtures/concept_relationship_enriched.csv \
  backend/tests/test_fixture_concept_validation.py backend/tests/test_metadata_fidelity.py \
  frontend/src/pages/upload.tsx
git commit -m "fix: make metadata validation and replacement transactional"
```

### Task 4: Manual mapping hydration and mapping artifact fidelity

**Files:**
- Create: `backend/src/metadata_mappings.py`
- Create: `backend/src/mapping_artifacts.py`
- Create: `backend/src/metadata_providers/fixture_mapping.py`
- Create: `backend/demo/metadata-fixtures/mapping-profile.json`
- Modify: `backend/src/upload.py`
- Modify: `backend/src/cohort_cache.py`
- Modify: `backend/src/mapping.py`
- Create: `backend/tests/test_mapping_hydration.py`
- Create: `backend/tests/test_mapping_artifacts.py`
- Create: `backend/tests/test_fixture_mapping.py`

**Interfaces:**
- Consumes: canonical dictionary paths, mapping graph, tracked `time-chf_gissi-hf_full.csv`.
- Produces: `MappingArtifactStore`, fixture generation result, and mapping-aware cache hydration.

- [ ] **Step 1: Write failing mapping persistence/cache tests**

```python
def test_variable_and_category_mappings_survive_cache_rebuild(mapped_cohort, rebuild_cache):
    rebuilt = rebuild_cache("TIME-CHF")
    assert rebuilt.variables["heart_rate"].mapped_id == "snomedct:364075005"
    assert rebuilt.variables["sex"].categories[0].mapped_id == "snomedct:248153007"


def test_outdated_mapping_is_not_reused(mapping_store, source_dictionary):
    artifact = mapping_store.materialize_fixture("TIME-CHF", "GISSI-HF")
    touch_newer(source_dictionary)
    assert mapping_store.cache_status(artifact).fresh is False
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd backend && uv run pytest tests/test_mapping_hydration.py tests/test_mapping_artifacts.py tests/test_fixture_mapping.py -q`

- [ ] **Step 3: Align category URIs and hydrate the cache**

Resolve `category_id` to the imported category value/label, build the same normalized `categorical_value_specification/<label>` URI, and query the separate mapping graph after CSV parsing. Never create orphan `/category/<index>` subjects.

- [ ] **Step 4: Implement the artifact store**

```python
@dataclass(frozen=True)
class MappingCacheStatus:
    filename: str
    fresh: bool
    source_mtime_ns: int
    target_mtime_ns: int


class MappingArtifactStore:
    def __init__(self, output_dir: Path): ...
    def list_for_cohorts(self, cohort_ids: set[str]) -> list[MappingArtifact]: ...
    def cache_status(self, artifact: MappingArtifact) -> MappingCacheStatus: ...
    def safe_path(self, filename: str) -> Path: ...
```

Reject traversal, `.meta.json` downloads, unrelated cohorts, and stale reuse.

- [ ] **Step 5: Implement deterministic fixture mapping generation**

For only the TIME-CHF/GISSI-HF pair, read the tracked `_full.csv`, write the runtime `_full.csv`, combined JSON, `.meta.json`, and activity entry atomically. Record `provider=fixture`, `synthetic=true`, source commit, source hash, parameters, and output hashes. Unsupported pairs return a clear 422-compatible provider error.

- [ ] **Step 6: Route all mapping APIs through the store/provider**

Update check/list/get/generate/log endpoints to use one configured directory and return their existing shapes.

- [ ] **Step 7: Run focused mapping tests**

Run: `cd backend && uv run pytest tests/test_mapping_hydration.py tests/test_mapping_artifacts.py tests/test_fixture_mapping.py -q`

Expected: persistence, filtering, freshness, safe retrieval, provenance, and logs pass.

- [ ] **Step 8: Commit**

```bash
git add backend/src/metadata_mappings.py backend/src/mapping_artifacts.py \
  backend/src/metadata_providers/fixture_mapping.py backend/src/upload.py \
  backend/src/cohort_cache.py backend/src/mapping.py \
  backend/demo/metadata-fixtures/mapping-profile.json \
  backend/tests/test_mapping_hydration.py backend/tests/test_mapping_artifacts.py \
  backend/tests/test_fixture_mapping.py
git commit -m "feat: add deterministic metadata mapping artifacts"
```

### Task 5: Deterministic synthetic cohort, EDA, and shuffled-sample pack

**Files:**
- Create: `backend/src/demo/__init__.py`
- Create: `backend/src/demo/profiles.py`
- Create: `backend/src/demo/generator.py`
- Create: `backend/src/demo/eda.py`
- Create: `backend/src/demo/manifest.py`
- Create: `backend/src/demo/assets.py`
- Create: `backend/scripts/seed_synthetic_data.py`
- Create: `backend/tests/test_synthetic_demo.py`
- Modify: `backend/src/explore.py`
- Modify: `backend/src/mapping.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 28-column schema and tracked mapping rows.
- Produces: `generate_demo_pack(output_dir: Path, seed: int, rows: int, force: bool) -> DemoManifest`.

- [ ] **Step 1: Write failing deterministic generation tests**

```python
def test_same_seed_produces_identical_manifest_hashes(tmp_path):
    first = generate_demo_pack(tmp_path / "one", seed=42, rows=2500, force=False)
    second = generate_demo_pack(tmp_path / "two", seed=42, rows=2500, force=False)
    assert first.files == second.files


def test_dictionary_columns_match_rows_and_validate(tmp_path):
    manifest = generate_demo_pack(tmp_path / "demo", seed=42, rows=2500, force=False)
    for cohort in ("TIME-CHF", "GISSI-HF"):
        dictionary = read_dictionary(manifest.dictionary(cohort))
        rows = pd.read_csv(manifest.rows(cohort))
        assert list(rows.columns) == list(dictionary["VARIABLENAME"])
        assert validate_fixture_dictionary(dictionary).all_pass


@pytest.mark.parametrize("cohort", ["TIME-CHF", "GISSI-HF"])
def test_generated_clinical_directions_and_exposure_invariants(generated_pack, cohort, profiles):
    rows = generated_pack.rows_frame(cohort)
    profile = profiles[cohort]
    nyha = profile.column("nyha_class")
    assert rows[nyha].corr(rows[profile.column("nt_pro_bnp")], method="spearman") > 0
    assert rows[nyha].corr(rows[profile.column("ejection_fraction")], method="spearman") < 0
    assert rows[nyha].corr(rows[profile.column("heart_failure_hospitalization")], method="spearman") > 0
    assert (rows[profile.column("furosemide_dose")] == 0).equals(
        rows[profile.column("furosemide_exposed")] == 0
    )


def test_followup_dropout_is_monotone_and_trajectories_are_bounded(generated_pack, profiles):
    rows = generated_pack.rows_frame("TIME-CHF")
    profile = profiles["TIME-CHF"]
    assert not (rows[profile.column("weight", "1y")].notna() & rows[profile.column("weight", "3m")].isna()).any()
    assert not (
        rows[profile.column("creatinine", "1y")].notna()
        & rows[profile.column("creatinine", "3m")].isna()
    ).any()
    assert rows[profile.column("patient_id")].is_unique
    weight_columns = [profile.column("weight", visit) for visit in ("baseline", "3m", "1y")]
    assert rows[weight_columns].stack().between(35, 220).all()


def test_selected_mapping_rows_resolve_to_both_emitted_dictionaries(generated_pack):
    source = set(generated_pack.dictionary_frame("TIME-CHF")["VARIABLENAME"])
    target = set(generated_pack.dictionary_frame("GISSI-HF")["VARIABLENAME"])
    mappings = generated_pack.selected_mapping_frame()
    assert {"nbnp", "nyha_class"} <= set(mappings["source"])
    assert {"v1_nt_probnp", "nyha"} <= set(mappings["target"])
    assert set(mappings["source"]) <= source
    assert set(mappings["target"]) <= target
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd backend && uv run pytest tests/test_synthetic_demo.py -q`

- [ ] **Step 3: Implement stable cohort profiles and row generation**

Use `numpy.random.Generator(PCG64(seed))` and SHA-256-derived cohort sub-seeds. Define semantic variables once but give each cohort an explicit real mapping-backed column profile (for example `nyha_class`→`nyha` and `nbnp`→`v1_nt_probnp`); never flatten both dictionaries to one canonical naming scheme. Encode the documented patient severity, correlated vitals, medications, outcomes, longitudinal trajectories, dropout, and categorical values. Never use Python `hash()`.

- [ ] **Step 4: Emit dictionaries/workbook and local assets into the immutable pack**

Write the exact 28 dictionary columns, `Descriptions` sheet, lowercase variable PNG filenames, `eda_output_<cohort>.json`, `shuffled_sample.csv`, and `shuffle_summary.txt` under the current endpoint paths. In offline mode, EDA/shuffled/compare endpoints read through `DEMO_PACK_DIR`; normal mode continues to read `DATA_FOLDER`. Runtime uploads, mappings, cache, and journals never write into the pack.

- [ ] **Step 5: Normalize and hash every artifact**

Normalize CSV line endings/float formatting, workbook properties/ZIP timestamps, row/column order, and JSON ordering. Manifest entries include seed, generator version, source commit, mapping source hash, row counts, and SHA-256 per file.

- [ ] **Step 6: Implement CLI overwrite protection**

```python
def main() -> int:
    args = parse_args()
    generate_demo_pack(args.output, args.seed, args.rows, args.force)
    return 0
```

Refuse a non-empty output directory unless `--force` is explicitly supplied.

- [ ] **Step 7: Run generator and plausibility tests**

Run: `cd backend && uv run pytest tests/test_synthetic_demo.py -q`

Run: `uv run python scripts/seed_synthetic_data.py --seed 42 --rows 2500 --output ../data/synthetic-demo-pack`

Expected: tests pass; manifest lists both cohorts, EDA, samples, and row files; generated data remains ignored.

- [ ] **Step 8: Commit**

```bash
git add .gitignore backend/src/demo/__init__.py backend/src/demo/profiles.py \
  backend/src/demo/generator.py backend/src/demo/eda.py backend/src/demo/manifest.py \
  backend/src/demo/assets.py backend/src/explore.py backend/src/mapping.py \
  backend/scripts/seed_synthetic_data.py backend/tests/test_synthetic_demo.py
git commit -m "feat: generate deterministic synthetic cohort demo"
```

### Task 6: Frontend metadata predicates, mapping projections, and stable browser selectors

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/src/utils/cohortFiltering.ts`
- Create: `frontend/src/utils/variableFiltering.ts`
- Create: `frontend/src/utils/equivalentVariables.ts`
- Create: `frontend/src/utils/mappingPreview.ts`
- Create: `frontend/src/utils/mappingGraph.ts`
- Create: `frontend/src/utils/apiUrls.ts`
- Create: `frontend/src/utils/__tests__/cohortFiltering.test.ts`
- Create: `frontend/src/utils/__tests__/variableFiltering.test.ts`
- Create: `frontend/src/utils/__tests__/equivalentVariables.test.ts`
- Create: `frontend/src/utils/__tests__/mappingPreview.test.ts`
- Create: `frontend/src/utils/__tests__/apiUrls.test.ts`
- Modify: `frontend/src/pages/cohorts.tsx`
- Modify: `frontend/src/components/VariablesList.tsx`
- Modify: `frontend/src/components/AutocompleteConcept.tsx`
- Modify: `frontend/src/pages/mapping.tsx`
- Modify: `frontend/src/pages/api/compare-eda/[sourceCohort]/[sourceVar]/[targetCohort]/[targetVar].js`

**Interfaces:**
- Consumes: existing page/component algorithms and synthetic metadata fixture.
- Produces: pure predicate/projection functions and stable `data-testid` contracts.

- [ ] **Step 1: Add Vitest and write failing parity tests**

```typescript
it('groups equivalent variables by concept code before OMOP id', () => {
  const grouped = groupEquivalentVariables(metadata, ['heart', 'rate'], 'and', 'variables');
  expect(grouped.conceptGroups[0].code).toBe('snomedct:364075005');
});

it('projects the fixture mapping into stable graph nodes and edges', () => {
  const preview = parseMappingPreview(mappingCsv);
  const graph = buildMappingGraph(preview.rows);
  expect(graph.edges.length).toBeGreaterThan(0);
  expect(graph.nodes.map(node => node.id)).toContain('TIME-CHF:heart_rate');
});

it('uses the container-internal backend URL for server-side EDA proxying', () => {
  expect(resolveServerApiUrl({ INTERNAL_API_URL: 'http://backend:80', NEXT_PUBLIC_API_URL: 'http://localhost:3000' }))
    .toBe('http://backend:80');
});
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd frontend && npm ci && npm run test -- --run`

Expected: missing Vitest script/modules.

- [ ] **Step 3: Extract pure functions without changing results**

Move the current inline algorithms into the utility files, export typed inputs/results, and make the existing components call them. Preserve OR, AND, exact, source, visit, category-count, outcome, domain/type, equivalent-code/OMOP, mapping table, and mapping graph semantics. The server-side compare-EDA proxy uses `INTERNAL_API_URL` first and the browser-facing URL only as a local-process fallback; the Compose overlay supplies `http://backend:80`.

- [ ] **Step 4: Add only stable selectors needed by the browser lane**

Add IDs such as `cohort-TIME-CHF`, `variable-TIME-CHF-heart_rate`, `concept-map-TIME-CHF-heart_rate`, `mapping-source`, `mapping-target-GISSI-HF`, `generate-mapping`, `mapping-view-table`, `mapping-view-graph`, and `mapping-preview`. Keep visible copy/layout unchanged.

- [ ] **Step 5: Run frontend tests, lint, and production build**

Run: `cd frontend && npm run test -- --run && npm run lint && npm run build`

Expected: all pass; extraction produces no rendering/type regression.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vitest.config.ts \
  frontend/src/utils/cohortFiltering.ts frontend/src/utils/variableFiltering.ts \
  frontend/src/utils/equivalentVariables.ts frontend/src/utils/mappingPreview.ts \
  frontend/src/utils/mappingGraph.ts frontend/src/utils/apiUrls.ts \
  frontend/src/utils/__tests__/cohortFiltering.test.ts \
  frontend/src/utils/__tests__/variableFiltering.test.ts \
  frontend/src/utils/__tests__/equivalentVariables.test.ts \
  frontend/src/utils/__tests__/mappingPreview.test.ts frontend/src/utils/__tests__/apiUrls.test.ts \
  frontend/src/pages/cohorts.tsx \
  frontend/src/components/VariablesList.tsx frontend/src/components/AutocompleteConcept.tsx \
  frontend/src/pages/mapping.tsx \
  'frontend/src/pages/api/compare-eda/[sourceCohort]/[sourceVar]/[targetCohort]/[targetVar].js'
git commit -m "test: cover metadata UI predicates and projections"
```

### Task 7: Cohort Explorer verification and metadata browser checkpoints

**Files:**
- Create: `backend/tests/test_metadata_routes.py`
- Create: `backend/tests/test_offline_metadata.py`
- Create: `docs/local-demo-metadata-browser-checklist.md`

**Interfaces:**
- Consumes: Tasks 1-6 and integration-plan `demo-browser-ready` command.
- Produces: complete metadata API proof and executable browser checkpoint checklist.

- [ ] **Step 1: Write the API lifecycle test**

Cover local login/admin, central workbook plus invalid-workbook rollback, valid/invalid/re-upload, full/summary metadata, validation reports, concept search, variable/category mapping and persistence, fixture mapping cache/list/get/log, EDA, shuffled samples, dictionaries/workbook, cache export/compare/refresh, and an outbound-network denial assertion. Exercise guarded `normalize-all-dictionary-headers` and `get-logs`; use an isolated scratch cohort for `delete-cohort` and isolated cache state for `clear-cache`, never the two-cohort presentation state.

- [ ] **Step 2: Run the complete focused backend/frontend suites**

Run: `cd backend && uv run pytest tests -q && uv run ruff check src tests scripts`

Run: `cd frontend && npm run test -- --run && npm run lint && npm run build`

Expected: all new/directly affected tests pass; unrelated `test_sparql_data.py` is not collected.

- [ ] **Step 3: Execute browser checkpoint 1**

Run `demo-browser-ready`, open only `http://localhost:3001`, log in as the configured admin, upload both dictionaries through `/upload`, and assert cohort cards/counts, dictionary variables, successful non-mutating validation, and replacement persistence. Capture screenshots and inspect console errors.

- [ ] **Step 4: Execute browser checkpoint 2**

On `/cohorts` and `/mapping`, exercise all search modes, representative cohort/variable/source filters, equivalent groups, variable/category concept mapping followed by reload, fixture mapping generation, table/graph switch, mapping log, EDA/variable graph, and one download. Assert no public-network request and no unexplained console error.

- [ ] **Step 5: Commit the tests and runbook**

```bash
git add backend/tests/test_metadata_routes.py backend/tests/test_offline_metadata.py \
  docs/local-demo-metadata-browser-checklist.md
git commit -m "test: verify offline metadata demo lifecycle"
```

Plan 3 owns the full DCR browser checkpoint and clean final journey.
