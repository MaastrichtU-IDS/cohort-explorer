import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _transport(*, aadcr_status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "db" and request.url.path == "/":
            return httpx.Response(200, text="<!doctype html><title>Oxigraph</title>")
        if request.url.host == "aadcrv2" and request.url.path == "/health":
            return httpx.Response(aadcr_status, json={"status": "ok"})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _client(settings, transport: httpx.MockTransport) -> TestClient:
    from src.health import build_health_router

    app = FastAPI()
    app.include_router(build_health_router(settings, transport=transport))
    return TestClient(app)


def test_health_reports_all_local_dependencies_ready(settings_factory):
    settings = settings_factory(
        dcr_backend="aadcrv2",
        sparql_endpoint="http://db:7878",
        aadcrv2_url="http://aadcrv2:8000",
        aadcrv2_jwt_secret="not-returned",  # noqa: S106 - synthetic test value
    )

    response = _client(settings, _transport()).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "dependencies": {
            "sparql": {"status": "ok"},
            "dcr_backend": {"status": "ok", "provider": "aadcrv2"},
        },
    }
    assert "not-returned" not in response.text


def test_health_fails_closed_without_leaking_dependency_details(settings_factory):
    settings = settings_factory(
        dcr_backend="aadcrv2",
        sparql_endpoint="http://db:7878",
        aadcrv2_url="http://aadcrv2:8000",
        aadcrv2_jwt_secret="secret-must-not-leak",  # noqa: S106 - synthetic test value
    )

    response = _client(settings, _transport(aadcr_status=503)).get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "dependencies": {
            "sparql": {"status": "ok"},
            "dcr_backend": {"status": "unavailable", "provider": "aadcrv2"},
        },
    }
    assert "secret-must-not-leak" not in response.text
    assert "http://" not in response.text


def test_health_skips_aadcr_probe_for_the_production_provider(settings_factory):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text="Oxigraph")

    settings = settings_factory(
        dcr_backend="decentriq",
        sparql_endpoint="http://db:7878",
    )

    response = _client(settings, httpx.MockTransport(handler)).get("/health")

    assert response.status_code == 200
    assert response.json()["dependencies"]["dcr_backend"] == {
        "status": "configured",
        "provider": "decentriq",
    }
    assert seen == ["http://db:7878/"]


def test_main_exposes_health_route(local_settings, import_main_with_stubs):
    module = import_main_with_stubs(local_settings)

    assert any(route.path == "/health" for route in module.app.routes)
