# AADCR v2 Local Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pinned AADCR v2 backend safe and reproducible enough for the synthetic local Cohort Explorer integration while preserving its native DEV, merge, PROD, provisioning, computation, and audit workflow.

**Architecture:** The backend receives injectable settings/database runtime, verifies Cohort Explorer's short-lived signed JWTs, and applies one central room/node access policy across existing routes. Local storage remains SQLite plus mounted files; computation remains an explicitly non-confidential subprocess demo with bounded execution.

**Tech Stack:** Python 3.11, Poetry, FastAPI, Pydantic Settings, SQLAlchemy/SQLite, pytest, Docker.

## Global Constraints

- Work only in `/Users/nikolasmolyndris/projects/delta-aadcrv2` on `codex/cohort-explorer-aadcrv2`.
- Baseline is `decentriq/delta` `davstur/aadcrv2` at `f13ef54fc3f0f56dae185d4aa35c6dff01ee8839`.
- Backend root is `avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend`.
- Execute every `Run:` command from that backend root. Execute every Git snippet from `/Users/nikolasmolyndris/projects/delta-aadcrv2`; its paths are Delta-root-relative.
- Python is constrained to `>=3.11,<3.14`; use Python 3.11 for every command.
- This backend is a local synthetic-data demo, not a confidential-computing or arbitrary-code sandbox.
- JWTs require HS256 signature plus `sub`, verified email, `iss`, `aud`, `iat`, and `exp`; reject `alg=none`.
- No secret, token, cloud credential, row data, or Authorization header may be committed or logged.
- Sample seeding is opt-in and idempotent; startup never deletes/recreates a room.
- Test/reset routes are absent unless `ENABLE_TEST_ROUTES=true`.
- Explanation endpoints return HTTP 503 when Gemini is disabled; the core API still starts.
- Delta provides the backend image; Cohort Explorer owns the cross-repository Compose topology.
- Use TDD and commit each completed task separately on the Delta branch.

## File Responsibility Map

- `aadcrv2/config.py`: typed environment configuration.
- `aadcrv2/database.py`: injectable engine/session runtime.
- `aadcrv2/auth.py`, `aadcrv2/middleware.py`: signed-token verification and request identity.
- `aadcrv2/services/access_policy.py`: room membership, creator, role, and node containment enforcement.
- `aadcrv2/services/upload_validation.py`: bounded safe CSV upload.
- Existing `routes/` and `services/`: native AADCR workflow, delegated to central policy/runtime.
- `aadcrv2/computations/aggregate_summary.py`: packaged deterministic aggregate computation.
- `Dockerfile`, `.dockerignore`, `.env.example`, `README.md`: local runtime packaging.
- `tests/unit/`, `tests/integration/`: hermetic and complete native-flow coverage.

---

### Task 1: Packaging, settings, and injectable database runtime

**Files:**
- Create: `README.md`
- Create: `.env.example`
- Create: `aadcrv2/config.py`
- Modify: `pyproject.toml`
- Modify: `poetry.lock`
- Modify: `aadcrv2/database.py`
- Modify: `aadcrv2/main.py`
- Modify: `aadcrv2/middleware.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_database_runtime.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: current SQLAlchemy models and `get_db` dependency.
- Produces: `Settings`, `DatabaseRuntime`, and injectable `create_app(settings, database)` used by request dependencies, audit logging, and background execution.

- [ ] **Step 1: Write failing settings/runtime tests**

```python
def test_settings_parse_paths_origins_and_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-only-secret-with-adequate-length")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'aadcr.db'}")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3001")
    settings = Settings()
    assert settings.results_dir == tmp_path / "results"
    assert settings.cors_origins == ["http://localhost:3001"]
    assert settings.seed_sample_data is False


def test_create_app_uses_injected_database(tmp_path, settings_factory):
    runtime = DatabaseRuntime(f"sqlite:///{tmp_path / 'test.db'}")
    app = create_app(settings_factory(), runtime)
    assert app.state.database is runtime


def test_audit_logger_writes_to_injected_database(client, runtime, signed_headers, room):
    response = client.post(
        f"/api/dcr/{room.id}/dev/participants",
        json={"userEmail": "owner@example.test"},
        headers=signed_headers,
    )
    assert response.status_code == 200
    with runtime.session_factory() as db:
        assert db.query(AuditLog).filter(AuditLog.dcr_id == room.id).count() == 1
```

- [ ] **Step 2: Run tests and verify RED**

Run: `poetry run pytest tests/unit/test_config.py tests/unit/test_database_runtime.py -q`

Expected: missing modules/interfaces.

- [ ] **Step 3: Add typed settings**

```python
class Settings(BaseSettings):
    database_url: str = "sqlite:///./aadcrv2.db"
    results_dir: Path = Path("./results")
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3001"]
    jwt_secret: SecretStr
    jwt_issuer: str = "cohort-explorer-local"
    jwt_audience: str = "aadcrv2-local"
    upload_max_bytes: int = 25 * 1024 * 1024
    seed_sample_data: bool = False
    seed_owner_email: EmailStr = "nikolas.molyndris@decentriq.ch"
    enable_test_routes: bool = False
    computation_timeout_seconds: int = 30
    google_ai_api_key: SecretStr | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

Import `NoDecode` from `pydantic_settings` and parse `CORS_ORIGINS` from a comma-separated environment value with a `mode="before"` field validator. This prevents Pydantic Settings from attempting JSON decoding before validation. Direct `Settings()` tests always inject a test-only `JWT_SECRET`; production/local Compose supplies an environment secret.

- [ ] **Step 4: Add `DatabaseRuntime` and inject it into the app**

```python
class DatabaseRuntime:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, connect_args=sqlite_args(database_url))
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def reset(self) -> None:
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def session(self) -> Iterator[Session]:
        db = self.session_factory()
        try:
            yield db
        finally:
            db.close()
```

`create_app(settings=None, database=None)` stores both on `app.state`; lifespan creates tables and seeds only when configured. `get_db` resolves the request app's runtime, the audit dependency uses `request.app.state.database.session_factory`, and no request/background path imports a global `SessionLocal`. Task 5 injects the same session factory into the execution worker.

- [ ] **Step 5: Repair Poetry package metadata**

Set `python = ">=3.11,<3.14"`, add PyJWT with cryptography support, keep `packages = [{include = "aadcrv2"}]`, create the declared README, and regenerate the lock file.

- [ ] **Step 6: Verify package/runtime tests**

Run: `poetry lock && poetry install && poetry check && poetry run pytest tests/unit/test_config.py tests/unit/test_database_runtime.py -q`

Expected: Poetry metadata is valid and tests pass.

- [ ] **Step 7: Commit**

```bash
git add avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/README.md \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/.env.example \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/pyproject.toml \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/poetry.lock \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/config.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/database.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/main.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/middleware.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/conftest.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/unit/test_config.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/unit/test_database_runtime.py
git commit -m "refactor: configure AADCR v2 runtime"
```

### Task 2: Signed JWT authentication

**Files:**
- Create: `aadcrv2/auth.py`
- Modify: `aadcrv2/middleware.py`
- Modify: `aadcrv2/main.py`
- Modify: `tests/client.py`
- Create: `tests/unit/test_auth.py`
- Modify: `tests/integration/test_authentication.py`

**Interfaces:**
- Consumes: `Settings.jwt_*`.
- Produces: `AuthenticatedUser`, `decode_access_token()`, and signed test-token factory.

- [ ] **Step 1: Write failing token tests**

```python
@pytest.mark.parametrize("token_kind", ["alg_none", "tampered", "expired", "wrong_issuer", "wrong_audience"])
def test_invalid_tokens_are_rejected(client, token_factory, token_kind):
    response = client.get("/api/dcr/", headers={"Authorization": f"Bearer {token_factory(token_kind)}"})
    assert response.status_code == 401


def test_valid_signed_token_sets_verified_user(client, token_factory):
    response = client.get("/api/dcr/", headers={"Authorization": f"Bearer {token_factory('valid')}"})
    assert response.status_code == 200
```

- [ ] **Step 2: Run and verify current `alg=none` test fails**

Run: `poetry run pytest tests/unit/test_auth.py tests/integration/test_authentication.py -q`

Expected: unsigned token is currently accepted or decoder interface is missing.

- [ ] **Step 3: Implement verified decoding**

```python
@dataclass(frozen=True)
class AuthenticatedUser:
    subject: str
    email: str
    email_verified: bool


def decode_access_token(token: str, settings: Settings) -> AuthenticatedUser:
    payload = jwt.decode(
        token,
        settings.jwt_secret.get_secret_value(),
        algorithms=["HS256"],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        options={"require": ["sub", "email", "email_verified", "iss", "aud", "iat", "exp"]},
    )
    if payload["email_verified"] is not True:
        raise InvalidToken("email is not verified")
    return AuthenticatedUser(payload["sub"], payload["email"].strip().lower(), True)
```

Replace unsafe payload parsing in middleware; exclude only health/openapi/docs and conditionally registered test routes.

- [ ] **Step 4: Add signed token factory to tests**

The factory emits the same claims Cohort Explorer will mint and never uses a committed secret.

- [ ] **Step 5: Run auth tests**

Run: `poetry run pytest tests/unit/test_auth.py tests/integration/test_authentication.py -q`

Expected: valid token passes; all invalid variants return 401.

- [ ] **Step 6: Commit**

```bash
git add avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/auth.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/middleware.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/main.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/client.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/unit/test_auth.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_authentication.py
git commit -m "fix: verify AADCR v2 access tokens"
```

### Task 3: Central room, role, and cross-DCR access policy

**Files:**
- Create: `aadcrv2/services/access_policy.py`
- Modify: `aadcrv2/routes/dcr.py`
- Modify: `aadcrv2/routes/dev_env.py`
- Modify: `aadcrv2/routes/prod_env.py`
- Modify: `aadcrv2/routes/datasets.py`
- Modify: `aadcrv2/routes/computation.py`
- Modify: `aadcrv2/routes/merge_requests.py`
- Modify: `aadcrv2/routes/explain.py`
- Modify: `aadcrv2/routes/audit_log.py`
- Modify: `aadcrv2/services/dcr_service.py`
- Modify: `aadcrv2/services/dev_env_service.py`
- Modify: `aadcrv2/services/prod_env_service.py`
- Modify: `aadcrv2/services/datasets_service.py`
- Modify: `aadcrv2/services/execution_service.py`
- Modify: `aadcrv2/services/merge_request_service.py`
- Modify: `aadcrv2/services/views_service.py`
- Modify: `aadcrv2/services/explain_service.py`
- Create: `tests/integration/test_access_policy.py`

**Interfaces:**
- Consumes: authenticated request user, SQLAlchemy session, current participant/permission models.
- Produces: `AccessPolicy` dependency used by every room-scoped route.

- [ ] **Step 1: Write creator/member/outsider and cross-room tests**

```python
def test_outsider_cannot_read_room(client, room, outsider_headers):
    assert client.get(f"/api/dcr/{room.id}", headers=outsider_headers).status_code == 403


def test_room_creation_writes_audit_after_id_exists(client, creator_headers, db):
    response = client.post("/api/dcr/", json={"name": "audited-room"}, headers=creator_headers)
    assert response.status_code == 200
    room_id = response.json()["id"]
    assert db.query(AuditLog).filter(AuditLog.dcr_id == room_id).count() == 1


def test_node_from_other_room_is_rejected(client, room_a, dataset, node_b, owner_headers):
    response = client.post(
        f"/api/dcr/{room_a.id}/provision-dataset",
        json={
            "dataset_id": dataset.id,
            "dataset_node_id": node_b.id,
            "provision_type": "PROD",
        },
        headers=owner_headers,
    )
    assert response.status_code in {403, 404}
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `poetry run pytest tests/integration/test_access_policy.py -q`

- [ ] **Step 3: Implement the central policy**

```python
class AccessPolicy:
    def __init__(self, db: Session, user: AuthenticatedUser): ...
    def require_room_member(self, dcr_id: str) -> AdvancedAnalyticsDCR: ...
    def require_room_creator(self, dcr_id: str) -> AdvancedAnalyticsDCR: ...
    def require_node_in_room(self, dcr_id: str, node_id: str, model: type[T]) -> T: ...
    def require_data_owner(self, dcr_id: str, node_id: str) -> None: ...
    def require_data_analyst(self, dcr_id: str, node_id: str) -> None: ...
```

Creator or PROD participant may read; creator alone renames/deletes; mutations require membership and owner/analyst roles. Every node lookup filters by both node ID and URL DCR ID.

- [ ] **Step 4: Replace ad-hoc route checks with policy calls**

Apply to room, DEV, merge, PROD, dataset provisioning, computation, explanation, and audit routes without changing their HTTP paths or successful response bodies. The standalone authenticated `/api/upload` route has no room/node identifiers; containment is enforced when its returned dataset ID is provisioned to a DCR node. Because generic route audit middleware cannot know a DCR ID before creation, `DCRService.create_dcr()` flushes the new ID and writes the creation `AuditLog` in the same transaction before commit.

- [ ] **Step 5: Run policy and affected route suites**

Run: `poetry run pytest tests/integration/test_access_policy.py tests/integration/test_dcr_routes.py tests/integration/test_dev_env_* tests/integration/test_datasets_routes.py -q`

Expected: new policy tests and directly affected baseline tests pass.

- [ ] **Step 6: Commit**

```bash
git add avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/access_policy.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/dcr.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/dev_env.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/prod_env.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/datasets.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/computation.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/merge_requests.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/explain.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/audit_log.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/dcr_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/dev_env_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/prod_env_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/datasets_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/execution_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/merge_request_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/views_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/explain_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_access_policy.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_dcr_routes.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_dev_env_computenode_routes.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_dev_env_datanode_routes.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_dev_env_general.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_dev_env_participant_routes.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_dev_env_permission_routes.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_datasets_routes.py
git commit -m "fix: enforce AADCR room authorization"
```

### Task 4: Bounded safe dataset upload

**Files:**
- Create: `aadcrv2/services/upload_validation.py`
- Modify: `aadcrv2/routes/datasets.py`
- Modify: `aadcrv2/services/datasets_service.py`
- Create: `tests/unit/test_upload_validation.py`
- Modify: `tests/integration/test_datasets_routes.py`

**Interfaces:**
- Consumes: `Settings.upload_max_bytes` and the authenticated uploader.
- Produces: `ValidatedCsvUpload` and 400/413 error mapping.

- [ ] **Step 1: Write boundary tests**

```python
@pytest.mark.parametrize("filename", ["../patients.csv", "/tmp/patients.csv", "patients.exe"])
def test_bad_filename_or_extension_is_rejected(client, filename, owner_headers, upload_url):
    response = client.post(
        upload_url,
        data={"dataset_name": "synthetic-patients"},
        files={"file": (filename, b"a,b\n1,2\n")},
        headers=owner_headers,
    )
    assert response.status_code == 400


def test_oversized_upload_is_413(client, owner_headers, upload_url, tiny_limit):
    response = client.post(
        upload_url,
        data={"dataset_name": "synthetic-patients"},
        files={"file": ("patients.csv", b"x" * 1025)},
        headers=owner_headers,
    )
    assert response.status_code == 413
```

- [ ] **Step 2: Run tests and verify RED**

Run: `poetry run pytest tests/unit/test_upload_validation.py tests/integration/test_datasets_routes.py -q`

- [ ] **Step 3: Implement streaming validation**

Sanitize with `Path(filename).name == filename`, require `.csv`, read at most `upload_max_bytes + 1`, reject empty/oversize, decode strict UTF-8, and parse a header plus one data row with `csv.reader`. Return bounded validated text to the existing `DatasetsService`, which preserves its current non-null database-backed `Datasets.data` storage; do not introduce an unplanned filesystem/migration path.

- [ ] **Step 4: Authenticate upload and enforce room/node policy at provisioning**

`POST /api/upload` requires a verified authenticated user and validates content before database persistence; it cannot perform a room check because its contract has no DCR/node identifier. `POST /api/dcr/{dcr_id}/provision-dataset` then validates room membership, target-node containment, dataset ownership, and the required PROD data-owner role before creating provisioning state. Never log content, bearer token, or configured secret.

- [ ] **Step 5: Run upload tests and commit**

Run: `poetry run pytest tests/unit/test_upload_validation.py tests/integration/test_datasets_routes.py -q`

```bash
git add avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/upload_validation.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/datasets.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/datasets_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/unit/test_upload_validation.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_datasets_routes.py
git commit -m "fix: validate AADCR dataset uploads"
```

### Task 5: Idempotent seed, packaged computation, and disabled explanation mode

**Files:**
- Modify: `aadcrv2/sample_data.py`
- Create: `aadcrv2/computations/__init__.py`
- Create: `aadcrv2/computations/aggregate_summary.py`
- Modify: `aadcrv2/services/computation_executor.py`
- Modify: `aadcrv2/services/execution_service.py`
- Modify: `aadcrv2/workers/execution_worker.py`
- Modify: `aadcrv2/routes/computation.py`
- Modify: `aadcrv2/services/llm.py`
- Modify: `aadcrv2/services/explain_service.py`
- Modify: `aadcrv2/routes/explain.py`
- Create: `tests/unit/test_seed.py`
- Create: `tests/unit/test_computation_runtime.py`
- Modify: `tests/integration/test_explain_endpoint.py`

**Interfaces:**
- Consumes: settings paths/timeouts/seed flag.
- Produces: idempotent `ensure_sample_room()`, packaged aggregate script, bounded executor, and explicit 503 explanation behavior.

- [ ] **Step 1: Write failing seed/computation/disabled-provider tests**

```python
def test_seed_is_opt_in_and_idempotent(app, db):
    assert count_rooms(db) == 0
    ensure_sample_room(db, enabled=False)
    assert count_rooms(db) == 0
    ensure_sample_room(db, enabled=True)
    ensure_sample_room(db, enabled=True)
    assert count_rooms(db) == 1


def test_explain_without_key_is_503(client, member_headers, room):
    response = client.get(f"/api/dcr/{room.id}/prod/explain", headers=member_headers)
    assert response.status_code == 503
    assert response.json()["detail"] == "Explanation provider is disabled"


def test_results_route_reads_configured_results_dir(client, settings, completed_execution, analyst_headers):
    write_result(settings.results_dir, completed_execution, "summary.json", b"{}")
    response = client.post(completed_execution.results_url, json=completed_execution.request, headers=analyst_headers)
    assert response.status_code == 200
    assert response.json()["files"] == ["summary.json"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `poetry run pytest tests/unit/test_seed.py tests/unit/test_computation_runtime.py tests/integration/test_explain_endpoint.py -q`

- [ ] **Step 3: Make sample seeding opt-in/idempotent**

Query by stable demo room ID/name; create only missing entities; never delete existing state; log no secret or row data.

- [ ] **Step 4: Package the aggregate computation**

Implement a script that reads provisioned CSV inputs, emits only row counts, numeric count/mean/min/max, and categorical counts above the demo threshold, and writes a deterministic result ZIP. Resolve it with `importlib.resources`, not a developer absolute path.

- [ ] **Step 5: Bound execution**

Run the subprocess as the container's non-root user, with configured timeout, results directory, minimal environment, no Docker socket, and captured size-limited stdout/stderr. Timeout becomes a structured failed computation state. Inject the app's `DatabaseRuntime.session_factory` into background execution/worker code; remove its global `SessionLocal` imports. Both `ExecutionService` dependency-copy logic and `routes/computation.py` result retrieval resolve paths from `Settings.results_dir`, never `/tmp/aadcrv2_results`.

- [ ] **Step 6: Disable explanations cleanly without a key**

Do not construct Gemini on startup. Explanation routes check provider availability and return 503; core routes remain usable.

- [ ] **Step 7: Run tests and commit**

Run: `poetry run pytest tests/unit/test_seed.py tests/unit/test_computation_runtime.py tests/integration/test_explain_endpoint.py -q`

```bash
git add avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/sample_data.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/computations/__init__.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/computations/aggregate_summary.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/computation_executor.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/execution_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/workers/execution_worker.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/computation.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/llm.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/services/explain_service.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/explain.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/unit/test_seed.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/unit/test_computation_runtime.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_explain_endpoint.py
git commit -m "feat: harden local AADCR execution"
```

### Task 6: Health, CORS, test-route gating, Docker, and documentation

**Files:**
- Create: `aadcrv2/routes/health.py`
- Create: `aadcrv2/routes/test_system.py`
- Modify: `aadcrv2/routes/system.py`
- Modify: `aadcrv2/main.py`
- Create: `Dockerfile`
- Create: `.dockerignore`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `../install-and-run-locally.md`
- Create: `tests/unit/test_app_runtime.py`

**Interfaces:**
- Consumes: settings/database runtime and prior tasks.
- Produces: public `/health`, explicit CORS, gated reset routes, non-root image.

- [ ] **Step 1: Write failing app-runtime tests**

```python
def test_health_is_public_and_reports_database(client_without_auth):
    response = client_without_auth.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ready"}


@pytest.mark.parametrize("path", ["/api/test/reset-database", "/api/test/clear-llm-cache"])
def test_destructive_test_routes_are_absent_outside_tests(client, path):
    assert client.delete(path).status_code == 404


def test_test_mode_registers_only_the_real_compatibility_paths(settings_factory, runtime):
    app = create_app(settings_factory(enable_test_routes=True), runtime)
    paths = {route.path for route in app.routes}
    assert {"/api/test/reset-database", "/api/test/clear-llm-cache"} <= paths
```

- [ ] **Step 2: Run tests and verify RED**

Run: `poetry run pytest tests/unit/test_app_runtime.py -q`

- [ ] **Step 3: Implement health, CORS, and conditional route registration**

Only `/health`, docs, and OpenAPI are public. Configure `allow_origins=settings.cors_origins`; never combine wildcard origins and credentials. Move the existing `DELETE /api/test/reset-database` and `DELETE /api/test/clear-llm-cache` handlers from `system.py` to `test_system.py`, preserve those paths, and include that router only when `enable_test_routes` is true. Keep non-destructive `/api/system/info` separately registered and authenticated.

- [ ] **Step 4: Build a non-root image**

Use a Python 3.11 slim multi-stage image, install Poetry dependencies without dev packages into a virtual environment, copy only the backend, create mounted data/results directories, switch to a numeric non-root UID, expose 8000, and healthcheck `/health`. Do not copy `.env`, databases, results, caches, or Git data.

- [ ] **Step 5: Clean documentation and credential material**

Document generated local secrets/placeholders only. Remove the committed Azure configuration credential from `install-and-run-locally.md` and record that its owner must rotate it externally; never reproduce the value in commits or test output.

- [ ] **Step 6: Verify tests and image**

Run: `poetry run pytest tests/unit/test_app_runtime.py -q`

Run: `docker build -t aadcrv2-local .`

Expected: tests pass; image runs as non-root and reports healthy with environment-provided secret/database paths.

- [ ] **Step 7: Commit**

```bash
git add avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/.dockerignore \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/.env.example \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/Dockerfile \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/README.md \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/main.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/health.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/system.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/aadcrv2/routes/test_system.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/unit/test_app_runtime.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/install-and-run-locally.md
git commit -m "build: package local AADCR v2 backend"
```

### Task 7: Native AADCR local-flow verification and baseline classification

**Files:**
- Create: `tests/integration/test_local_flow.py`
- Modify: `pytest.ini`
- Modify: `tests/conftest.py`
- Modify: `tests/integration/test_computation_execution_routes.py`

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: one hermetic native create-to-audit proof and explicit live-test markers.

- [ ] **Step 1: Write the complete native flow test**

Using signed creator/owner/analyst tokens: create a room; add participants, FILE nodes, computation, and permissions; create/observe initial merged request; resolve PROD nodes; upload and provision both synthetic CSV fixtures; run aggregate computation; fetch/inspect result ZIP; assert creation, merge, provision, and execution audit entries.

- [ ] **Step 2: Mark true live-server tests explicitly**

Add `live` marker and ensure `-m "not live"` excludes tests requiring `127.0.0.1:8765` or external providers. Repair directly affected fixture-table/permission failures; list unrelated remaining failures rather than masking them.

- [ ] **Step 3: Run the local flow and hermetic suite**

Run: `poetry run pytest tests/integration/test_local_flow.py -q`

Run: `poetry run pytest -q -m "not live"`

Expected: local flow passes; all new/directly affected tests pass; any unrelated baseline failures are reported by exact test name.

- [ ] **Step 4: Run formatting/type/package checks**

Run: `poetry check && poetry run black --check aadcrv2 tests && poetry run isort --check-only aadcrv2 tests`

- [ ] **Step 5: Commit**

```bash
git add avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/pytest.ini \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/conftest.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_computation_execution_routes.py \
  avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend/tests/integration/test_local_flow.py
git commit -m "test: verify native AADCR local flow"
```

Plan 3 consumes this branch through the sibling-repository Docker build and verifies it through Cohort Explorer's browser workflow.
