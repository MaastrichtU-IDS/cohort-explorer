import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
DELTA_ROOT = Path(
    os.environ.get(
        "AADCRV2_TEST_REPO_DIR",
        str(ROOT.parent / "delta-aadcrv2"),
    )
).resolve()
DOCKER = shutil.which("docker") or "/usr/local/bin/docker"
GIT = shutil.which("git") or "/usr/bin/git"
DELTA_BACKEND = "avato-backend/frontend/decentriq-platform/src/features/aadcrv2/backend"
REQUIRED_AADCR_COMMIT = "03cefacf1a70cbe67189821d5739a7abd581d48e"


def _load_demo_seed_module():
    path = ROOT / "scripts" / "demo-seed.py"
    spec = importlib.util.spec_from_file_location("demo_seed", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compose_config() -> dict:
    env = {
        **os.environ,
        "AADCRV2_REPO_DIR": str(DELTA_ROOT),
        "DEMO_PACK_HOST_DIR": str(ROOT / "data" / "synthetic-demo-pack"),
        "JWT_SECRET": "configuration-test-session-secret",
        "AADCRV2_JWT_SECRET": "configuration-test-service-secret",
        "COMPOSE_PROJECT_NAME": "cohort-explorer-config-test",
    }
    result = subprocess.run(  # noqa: S603 - fixed local Docker command
        [
            DOCKER,
            "compose",
            "-f",
            str(ROOT / "docker-compose.yml"),
            "-f",
            str(ROOT / "docker-compose.local-aadcr.yml"),
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _published_ports(service: dict) -> set[int]:
    return {int(port["published"]) for port in service.get("ports", [])}


def _published_host_ips(service: dict) -> set[str]:
    return {port["host_ip"] for port in service.get("ports", [])}


def _volume_for(service: dict, target: str) -> dict:
    return next(volume for volume in service.get("volumes", []) if volume["target"] == target)


def test_local_overlay_pins_services_ports_and_local_build(compose_config):
    services = compose_config["services"]

    assert set(services) == {"backend", "frontend", "db", "aadcrv2", "aadcrv2-ui", "gateway"}
    for name in ("backend", "frontend", "db", "aadcrv2", "aadcrv2-ui"):
        assert services[name].get("ports", []) == []
    assert _published_ports(services["gateway"]) == {3000, 3001, 3002, 18000}
    assert _published_host_ips(services["gateway"]) == {"127.0.0.1"}
    assert services["aadcrv2"]["build"]["context"] == str(DELTA_ROOT / DELTA_BACKEND)
    assert services["aadcrv2-ui"]["build"]["context"] == str(DELTA_ROOT)
    assert services["aadcrv2-ui"]["build"]["dockerfile"] == (
        "avato-backend/frontend/decentriq-platform/Dockerfile.aadcr-local"
    )
    assert services["db"]["image"] == "ghcr.io/oxigraph/oxigraph:0.5.9"


def test_local_overlay_selects_offline_metadata_and_guarded_admin(compose_config):
    environment = compose_config["services"]["backend"]["environment"]

    assert environment["DCR_BACKEND"] == "aadcrv2"
    assert environment["CONCEPT_SEARCH_BACKEND"] == "fixture"
    assert environment["CONCEPT_VALIDATION_BACKEND"] == "fixture"
    assert environment["MAPPING_GENERATION_BACKEND"] == "fixture"
    assert environment["OFFLINE_DEMO"] == "true"
    assert environment["AADCRV2_SYNTHETIC_DEMO"] == "true"
    assert environment["AADCRV2_HANDOFF_MODE"] == "bootstrap"
    assert environment["DEV_MODE"] == "true"
    assert environment["LOCAL_AUTH_ENABLED"] == "true"
    assert environment["LOCAL_AUTH_EMAIL"] == "nikolas.molyndris@decentriq.ch"
    assert environment["SESSION_COOKIE_SECURE"] == "false"
    assert environment["AADCRV2_ROOM_URL_TEMPLATE"] == (
        "http://localhost:3002/aadcrv2/dcr/{dcr_id}"
    )
    assert compose_config["services"]["aadcrv2"]["environment"]["LOCAL_DEMO_AUTH_ENABLED"] == "true"
    assert compose_config["services"]["aadcrv2"]["environment"]["SEED_OWNER_EMAIL"] == (
        "nikolas.molyndris@decentriq.ch"
    )
    assert compose_config["services"]["frontend"]["environment"]["INTERNAL_API_URL"] == "http://backend:80"


def test_gateway_rejects_unknown_hosts_and_strips_cross_service_cookies():
    config = (ROOT / "config" / "local-demo-gateway.conf").read_text(encoding="utf-8")

    assert "listen 7878" not in config
    assert config.count("default_server") == 4
    assert config.count("return 444") == 4
    assert config.count('proxy_set_header Cookie "";') >= 3
    assert "proxy_pass http://aadcrv2-ui:8080" in config


def test_local_backend_never_attempts_an_online_uv_sync(compose_config):
    entrypoint = compose_config["services"]["backend"]["entrypoint"]

    assert entrypoint[:3] == ["uv", "run", "--no-sync"]


def test_local_frontend_runs_the_built_app_without_runtime_compilation(compose_config):
    assert compose_config["services"]["frontend"]["entrypoint"] == [
        "npm",
        "run",
        "start",
        "--",
        "-p",
        "3001",
    ]


def test_local_overlay_separates_immutable_pack_and_mutable_state(compose_config):
    backend = compose_config["services"]["backend"]
    pack = _volume_for(backend, "/demo-pack")
    runtime = _volume_for(backend, "/demo-runtime")
    frontend_pack = _volume_for(compose_config["services"]["frontend"], "/data")

    assert pack["type"] == "bind"
    assert pack["read_only"] is True
    assert frontend_pack["type"] == "bind"
    assert frontend_pack["source"] == pack["source"]
    assert frontend_pack["read_only"] is True
    assert runtime["type"] == "volume"
    assert backend["environment"]["DEMO_PACK_DIR"] == "/demo-pack"
    assert backend["environment"]["DATA_FOLDER"] == "/demo-runtime"
    assert backend["environment"]["MAPPING_OUTPUT_DIR"].startswith("/demo-runtime/")


def test_local_overlay_is_internal_and_never_mounts_docker_socket(compose_config):
    assert compose_config["networks"]["demo_internal"]["internal"] is True
    assert compose_config["networks"]["demo_ingress"].get("internal", False) is False
    for name in ("backend", "frontend", "db", "aadcrv2", "aadcrv2-ui"):
        assert set(compose_config["services"][name]["networks"]) == {"demo_internal"}
    assert set(compose_config["services"]["gateway"]["networks"]) == {
        "demo_ingress",
        "demo_internal",
    }
    aadcr_volumes = compose_config["services"]["aadcrv2"].get("volumes", [])
    assert all(volume["source"] != "/var/run/docker.sock" for volume in aadcr_volumes)
    gateway = compose_config["services"]["gateway"]
    assert gateway["read_only"] is True
    assert gateway.get("environment", {}) == {}
    assert all(volume["type"] == "bind" and volume["read_only"] for volume in gateway["volumes"])


def test_example_environment_contains_no_usable_secret():
    example = (ROOT / ".env.local-demo.example").read_text()

    assert "JWT_SECRET=" not in example
    assert "AADCRV2_JWT_SECRET=" not in example
    assert "nikolas.molyndris@decentriq.ch" in example


def test_example_environment_is_tracked_demo_documentation():
    result = subprocess.run(  # noqa: S603 - fixed local Git command
        [GIT, "check-ignore", "--quiet", ".env.local-demo.example"],
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 1


def test_tracked_demo_configuration_does_not_embed_a_developer_home_path():
    portable_files = (
        ROOT / ".env.local-demo.example",
        ROOT / "LOCAL_DEMO.md",
        ROOT / "docker-compose.local-aadcr.yml",
        ROOT / "scripts" / "demo-common.sh",
    )

    for path in portable_files:
        assert "/Users/nikolasmolyndris" not in path.read_text(), path

    common = (ROOT / "scripts" / "demo-common.sh").read_text()
    assert '$(cd "${DEMO_ROOT}/.." && pwd)/delta-aadcrv2' in common


def test_demo_common_generates_private_distinct_runtime_secrets(tmp_path):
    env = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": "configuration-secret-test",
        "DEMO_STATE_ROOT": str(tmp_path),
        "AADCRV2_REPO_DIR": str(DELTA_ROOT),
        "DEMO_PACK_HOST_DIR": str(ROOT / "data" / "synthetic-demo-pack"),
    }
    result = subprocess.run(  # noqa: S603 - fixed local Bash command
        [
            "/bin/bash",
            "-c",
            "source scripts/demo-common.sh && demo_prepare_runtime && printf '%s' \"$DEMO_RUNTIME_ENV\"",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    runtime_env = Path(result.stdout)
    values = dict(
        line.split("=", 1) for line in runtime_env.read_text().splitlines() if line and not line.startswith("#")
    )
    assert runtime_env.stat().st_mode & 0o777 == 0o600
    assert len(values["JWT_SECRET"]) >= 64
    assert len(values["AADCRV2_JWT_SECRET"]) >= 64
    assert values["JWT_SECRET"] != values["AADCRV2_JWT_SECRET"]


def test_demo_compose_uses_the_validated_runtime_paths_not_later_shell_overrides(tmp_path):
    env = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": "configuration-path-pin-test",
        "DEMO_STATE_ROOT": str(tmp_path),
        "AADCRV2_REPO_DIR": str(DELTA_ROOT),
        "DEMO_PACK_HOST_DIR": str(ROOT / "data" / "synthetic-demo-pack"),
    }
    result = subprocess.run(  # noqa: S603 - fixed local Bash command
        [
            "/bin/bash",
            "-c",
            (
                "source scripts/demo-common.sh && demo_prepare_runtime && "
                "export AADCRV2_REPO_DIR=/tmp/unvalidated-aadcr && "
                "export DEMO_PACK_HOST_DIR=/tmp/unvalidated-pack && "
                "demo_compose config --format json"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["services"]["aadcrv2"]["build"]["context"] == str(DELTA_ROOT / DELTA_BACKEND)
    assert _volume_for(rendered["services"]["backend"], "/demo-pack")["source"] == str(
        ROOT / "data" / "synthetic-demo-pack"
    )


def test_demo_common_requires_the_exact_clean_reviewed_aadcr_checkout():
    script = (ROOT / "scripts" / "demo-common.sh").read_text(encoding="utf-8")

    assert f'DEMO_AADCRV2_REQUIRED_COMMIT="{REQUIRED_AADCR_COMMIT}"' in script
    assert 'actual_head="$(git -C "${aadcr_repo}" rev-parse HEAD)"' in script
    assert '[[ "${actual_head}" == "${DEMO_AADCRV2_REQUIRED_COMMIT}" ]]' in script
    assert "status --porcelain --untracked-files=all" in script


def test_wait_script_names_the_probe_that_never_becomes_ready():
    result = subprocess.run(  # noqa: S603 - fixed local Python command
        [
            sys.executable,
            str(ROOT / "scripts" / "wait_for_demo.py"),
            "--timeout",
            "0.05",
            "--interval",
            "0.01",
            "--probe",
            "aadcrv2=http://127.0.0.1:1/health",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "aadcrv2" in result.stderr


def test_makefile_exposes_stable_demo_commands():
    makefile = (ROOT / "Makefile").read_text()

    for target in (
        "demo-generate",
        "demo-up",
        "demo-seed",
        "demo-smoke",
        "demo-browser-ready",
        "demo-browser-install",
        "demo-browser-test",
        "demo-down",
    ):
        assert f"{target}:" in makefile


def test_makefile_demo_entrypoints_are_executable():
    for script in (
        "demo-generate.sh",
        "demo-up.sh",
        "demo-seed.sh",
        "demo-smoke.sh",
        "demo-browser-ready.sh",
        "demo-browser-test.sh",
        "demo-down.sh",
    ):
        assert os.access(ROOT / "scripts" / script, os.X_OK), script


def test_committed_browser_acceptance_covers_metadata_handoff_and_real_aadcr_ui():
    package = json.loads((ROOT / "frontend" / "package.json").read_text())
    config = (ROOT / "frontend" / "playwright.config.ts").read_text()
    spec = (ROOT / "frontend" / "e2e" / "local-aadcr-demo.spec.ts").read_text()

    assert "test:e2e:local" in package["scripts"]
    assert "workers: 1" in config
    assert "timeout: 900_000" in config
    assert "webServer" not in config
    assert "demo-browser-install" in (ROOT / "Makefile").read_text()
    assert "npm ci && npm exec -- playwright install chromium" in (ROOT / "Makefile").read_text()
    runner = (ROOT / "scripts" / "demo-browser-test.sh").read_text()
    assert "run-summary.json" in runner
    assert "acceptance-details.json" in runner
    assert '"source_revisions"' in runner
    assert '"source_tree_state": "clean"' in runner
    assert "status --porcelain --untracked-files=all" in runner
    assert "ce_start_revision=" in runner
    assert "aadcr_start_revision=" in runner
    assert "changed during browser acceptance" in runner
    assert runner.index("npm run test:e2e:local") < runner.index("run-summary.json")
    for selector in (
        "upload-cohort-select",
        "validate-dictionary",
        "upload-dictionary",
        "mapping-source",
        "generate-mapping",
        "dcr-launcher",
        "dcr-create",
        "dcr-handoff-boundary",
        "dcr-bootstrap-next-steps",
        "concept-map-TIME-CHF-age",
        "concept-map-TIME-CHF-gender-category-0",
        "mapping-view-table",
        "mapping-view-graph",
        "dcr-name-input",
        "dcr-mapping-toggle",
        "dcr-mapping-upload-slot",
        "eda-cv-ranking",
        "eda-original-graph",
        "metadata-filter-study_design",
        "metadata-filter-institution",
        "metadata-filter-${filterId}",
        "source-tab-TIME-CHF-EHR",
        "source-tab-TIME-CHF-all",
    ):
        assert selector in spec
    for metadata_control in (
        "OR Search",
        "AND Search",
        "Exact Phrase",
        "measurement",
        "4+ categories",
        "follow-up 1 year",
        "Show Outcome Variables",
        "age at enrollment",
        "TIME-CHF_invalid_datadictionary.csv",
        "approvedLocalFailures",
        "Open Advanced Analytics DCR",
        "Add dataset",
        "Add computation node",
        "Add participant node",
        "Create change request",
    ):
        assert metadata_control in spec
    for audit_label in ("Create merge request", "Provision dataset"):
        assert audit_label in spec
    for screenshot in (
        "01-login-admin.png",
        "02-dictionaries-uploaded.png",
        "03-metadata-filters.png",
        "04-manual-concept-mapping.png",
        "05-generated-mapping-table.png",
        "06-generated-mapping-graph.png",
        "07-dcr-handoff-review.png",
        "08-dcr-handoff-created.png",
        "09-aadcr-original-production.png",
        "10-aadcr-synthetic-upload.png",
        "11-aadcr-computation-editor.png",
        "12-aadcr-change-request.png",
        "13-aadcr-audit-log.png",
    ):
        assert screenshot in spec
    assert "DEMO_AADCR_UI_URL" in spec
    assert "aadcr-data-node-${nodeId}" in spec
    assert "handoff_mode: 'bootstrap'" in spec
    assert "not.toHaveProperty('merge_request_id')" in spec
    assert "not.toHaveProperty('aggregate_computation_node_id')" in spec
    assert "acceptance-details.json" in spec
    assert "page.context().on('request'" in spec
    assert "page.on('websocket'" in spec


def test_demo_up_validates_the_immutable_pack_before_starting_compose():
    script = (ROOT / "scripts" / "demo-up.sh").read_text()

    validation = 'scripts/seed_synthetic_data.py --validate --output "${pack}"'
    assert validation in script
    assert script.index(validation) < script.index("demo_compose up")


def test_demo_smoke_invalidates_previous_success_evidence_before_running():
    script = (ROOT / "scripts" / "demo-smoke.sh").read_text()

    invalidate = 'rm -f -- "${DEMO_SMOKE_EVIDENCE}"'
    assert invalidate in script
    assert script.index(invalidate) < script.index("smoke_local_aadcr.py")
    assert "DEMO_BASE_URL" not in script
    assert "DEMO_AADCR_URL" not in script
    assert "demo_gateway_host_port 3000" in script
    assert "demo_gateway_host_port 18000" in script


def test_demo_seed_runs_with_the_backend_uv_environment():
    script = (ROOT / "scripts" / "demo-seed.sh").read_text()

    assert 'cd "${DEMO_ROOT}/backend"' in script
    assert "uv run python ../scripts/demo-seed.py" in script
    assert '"$@"' in script
    assert '"${DEMO_ROOT}/scripts/demo-seed.py"' not in script


def test_browser_checkpoint_seeds_only_the_central_workbook():
    script = (ROOT / "scripts" / "demo-browser-ready.sh").read_text()

    assert '"${script_dir}/demo-seed.sh" --central-only' in script
    assert "demo-seed.py" not in script
    assert "down --remove-orphans --volumes" in script
    assert "|| true" not in script
    assert 'rm -rf -- "${evidence_dir}"' in script


def test_readiness_does_not_publish_or_probe_oxigraph():
    wait_script = (ROOT / "scripts" / "wait_for_demo.py").read_text()
    up_script = (ROOT / "scripts" / "demo-up.sh").read_text()

    assert "7878" not in wait_script
    assert "oxigraph=" not in up_script


def test_demo_seed_authenticates_and_uploads_only_requested_metadata(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    (pack / "cohorts" / "TIME-CHF").mkdir(parents=True)
    (pack / "cohorts" / "GISSI-HF").mkdir(parents=True)
    (pack / "iCARE4CVD_Cohorts.xlsx").write_bytes(b"workbook")
    (pack / "cohorts" / "TIME-CHF" / "TIME-CHF_datadictionary.csv").write_text("A,B\n1,2\n")
    (pack / "cohorts" / "GISSI-HF" / "GISSI-HF_datadictionary.csv").write_text("A,B\n1,2\n")
    requests: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        requests.append((request.url.path, body))
        if request.url.path == "/login":
            return httpx.Response(
                307,
                headers={"set-cookie": "token=opaque-session; Path=/; HttpOnly"},
            )
        return httpx.Response(200, json={"message": "ok"})

    module = _load_demo_seed_module()
    client_options = {}
    real_client = httpx.Client

    def local_client(*args, **kwargs):
        client_options.update(kwargs)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "Client", local_client)
    summary = module.seed_demo(
        pack,
        base_url="http://127.0.0.1:3000",
        transport=httpx.MockTransport(handler),
    )

    assert [path for path, _body in requests] == [
        "/login",
        "/upload-cohorts-metadata",
        "/upload-cohort",
        "/upload-cohort",
    ]
    assert b'name="cohorts_metadata"' in requests[1][1]
    assert b'name="cohort_dictionary"' in requests[2][1]
    assert b'name="cohort_id"' in requests[2][1]
    assert client_options["trust_env"] is False
    assert summary == {
        "central_workbook": "iCARE4CVD_Cohorts.xlsx",
        "dictionaries": ["GISSI-HF", "TIME-CHF"],
    }


def test_demo_seed_central_only_does_not_upload_dictionaries(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "iCARE4CVD_Cohorts.xlsx").write_bytes(b"workbook")
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/login":
            return httpx.Response(307, headers={"set-cookie": "token=x; Path=/"})
        return httpx.Response(200, json={"message": "ok"})

    module = _load_demo_seed_module()
    module.seed_demo(
        pack,
        base_url="http://127.0.0.1:3000",
        central_only=True,
        transport=httpx.MockTransport(handler),
    )

    assert paths == ["/login", "/upload-cohorts-metadata"]


def test_demo_seed_defaults_to_the_namespace_runtime_pack(tmp_path, monkeypatch):
    pack = tmp_path / "custom-pack"
    state = tmp_path / "state" / "custom-namespace"
    state.mkdir(parents=True)
    (state / "runtime.env").write_text(f"DEMO_PACK_HOST_DIR={pack}\n")
    monkeypatch.setenv("DEMO_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "custom-namespace")
    monkeypatch.delenv("DEMO_PACK_HOST_DIR", raising=False)

    module = _load_demo_seed_module()

    assert module.default_pack_path(ROOT) == pack


def test_offline_demo_can_start_before_the_central_workbook_is_seeded(
    tmp_path,
    monkeypatch,
):
    from src import upload

    runtime = tmp_path / "empty-runtime"
    workbook = runtime / "iCARE4CVD_Cohorts.xlsx"
    calls: list[str] = []
    monkeypatch.setattr(upload.settings, "offline_demo", True)
    monkeypatch.setattr(upload.settings, "data_folder", str(runtime))
    monkeypatch.setattr(upload, "COHORTS_METADATA_FILEPATH", str(workbook))
    monkeypatch.setattr(upload, "clear_cache", lambda: calls.append("clear-cache"))
    monkeypatch.setattr(
        upload,
        "_perform_triplestore_initialization",
        lambda: pytest.fail("an empty offline runtime must wait for the API seed"),
    )

    upload.init_triplestore()

    assert calls == ["clear-cache"]
    assert runtime.is_dir()
    assert not workbook.exists()
    assert not (runtime / "triplestore_init.lock").exists()
