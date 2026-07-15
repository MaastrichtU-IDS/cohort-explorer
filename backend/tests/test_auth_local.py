from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_local_login_uses_only_configured_admin(client, local_settings, decode_test_session):
    response = client.get("/login?email=attacker@example.test", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == local_settings.frontend_url
    token = response.cookies["token"]
    payload = decode_test_session(token, local_settings.jwt_secret)
    assert payload["email"] == "nikolas.molyndris@decentriq.ch"
    assert payload["access_token"] == "local-demo"
    assert "nikolas.molyndris@decentriq.ch" in local_settings.admins_list
    assert "Secure" not in response.headers["set-cookie"]


def test_local_login_is_disabled_outside_development(settings_factory, monkeypatch):
    from src import auth

    production_settings = settings_factory(
        dev_mode=False,
        local_auth_enabled=True,
    )
    monkeypatch.setattr(auth, "settings", production_settings)
    app = FastAPI()
    app.include_router(auth.router)

    with TestClient(app) as production_client:
        response = production_client.get("/login", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"].startswith("https://auth.example.test/authorize?")
    assert "token" not in response.cookies


@pytest.mark.parametrize(("dev_mode", "route_is_registered"), [(False, False), (True, True)])
def test_debug_permissions_registration_matches_development_mode(
    settings_factory,
    import_main_with_stubs,
    dev_mode,
    route_is_registered,
):
    settings = settings_factory(dev_mode=dev_mode)
    main = import_main_with_stubs(settings)

    paths = {route.path for route in main.app.routes}
    assert ("/debug/permissions" in paths) is route_is_registered


def test_debug_permissions_requires_authentication(debug_client):
    response = debug_client.get("/debug/permissions")

    assert response.status_code == 401


def test_debug_permissions_rejects_non_admin(
    debug_client,
    local_settings,
    monkeypatch,
):
    from src import auth

    monkeypatch.setattr(auth, "settings", local_settings)
    expires = datetime.now(timezone.utc) + timedelta(minutes=5)
    token = auth.create_access_token(
        {"email": "attacker@example.test", "access_token": "test"},
        int(expires.timestamp()),
    )
    debug_client.cookies.set("token", token)

    response = debug_client.get("/debug/permissions")

    assert response.status_code == 403
    assert response.json() == {"detail": "Admin access required"}


def test_local_admin_can_access_debug_permissions(debug_client, local_settings):
    login_response = debug_client.get("/login", follow_redirects=False)
    assert login_response.status_code == 307

    response = debug_client.get("/debug/permissions")

    assert response.status_code == 200
    assert response.json() == {
        "admins": ["nikolas.molyndris@decentriq.ch"],
        "emails": {},
    }
