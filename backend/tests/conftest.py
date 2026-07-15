import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from jose import jwt

TEST_JWT_SECRET = "ephemeral-test-jwt-secret"
LOCAL_ADMIN_EMAIL = "nikolas.molyndris@decentriq.ch"


@pytest.fixture
def settings_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    monkeypatch.setenv("JWT_SECRET", TEST_JWT_SECRET)

    from src.config import Settings

    def factory(**overrides: object) -> Settings:
        values: dict[str, object] = {
            "auth_endpoint": "https://auth.example.test",
            "client_id": "test-client",
            "data_folder": str(tmp_path),
            "jwt_secret": TEST_JWT_SECRET,
        }
        values.update(overrides)
        return Settings(**values)

    return factory


@pytest.fixture
def local_settings(settings_factory):
    return settings_factory(
        admins="",
        dev_mode=True,
        local_auth_email=LOCAL_ADMIN_EMAIL,
        local_auth_enabled=True,
        session_cookie_secure=False,
    )


@pytest.fixture
def decode_test_session():
    def decode(token: str, secret: str) -> dict[str, object]:
        return jwt.decode(token, secret, algorithms=["HS256"])

    return decode


@pytest.fixture
def client(local_settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from src import auth

    monkeypatch.setattr(auth, "settings", local_settings)
    app = FastAPI()
    app.include_router(auth.router)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def debug_client(local_settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from src import admin, auth, cohort_cache

    monkeypatch.setattr(auth, "settings", local_settings)
    monkeypatch.setattr(admin, "settings", local_settings)
    monkeypatch.setattr(cohort_cache, "get_cohorts_from_cache", lambda _email: {})

    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(admin.debug_router)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def import_main_with_stubs(monkeypatch: pytest.MonkeyPatch):
    loaded_main_modules: list[ModuleType] = []

    def stub_module(name: str, **attributes: object) -> None:
        module = ModuleType(name)
        module.router = APIRouter()
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    def load(test_settings):
        from src import config

        monkeypatch.setattr(config, "settings", test_settings)
        stub_module("src.data_analysis")
        stub_module(
            "src.decentriq",
            refresh_all_dcrs_via_decentriq_api=lambda: None,
        )
        stub_module("src.explore")
        stub_module("src.mapping")
        stub_module("src.upload", init_triplestore=lambda: None)
        stub_module("src.monitoring", run_periodic_monitoring=lambda: None)
        stub_module("src.docs")

        sys.modules.pop("src.main", None)
        module = importlib.import_module("src.main")
        loaded_main_modules.append(module)
        return module

    yield load

    for module in loaded_main_modules:
        if sys.modules.get("src.main") is module:
            sys.modules.pop("src.main", None)
