import asyncio
import importlib
from types import SimpleNamespace

import pytest

from src.dcr_backends.factory import get_dcr_backend


def test_default_backend_is_decentriq(settings_factory):
    backend = get_dcr_backend(settings_factory(dcr_backend="decentriq"))

    assert backend.provider_name == "decentriq"
    assert backend.capabilities.supports_live_creation is True
    assert backend.capabilities.local_simulation is False


def test_aadcr_backend_advertises_local_capabilities(settings_factory):
    backend = get_dcr_backend(settings_factory(dcr_backend="aadcrv2"))

    assert backend.provider_name == "aadcrv2"
    assert backend.capabilities.supports_computation_output is True
    assert backend.capabilities.supports_shuffle_output is False
    assert backend.capabilities.synthetic_data_only is True


def test_unknown_backend_fails_closed(settings_factory):
    settings = settings_factory(dcr_backend="unknown")

    with pytest.raises(ValueError, match="Unsupported DCR_BACKEND"):
        settings.validate_runtime()
    with pytest.raises(ValueError, match="Unsupported DCR_BACKEND"):
        get_dcr_backend(settings)


def test_factory_imports_only_the_selected_adapter(settings_factory, monkeypatch):
    imported: list[str] = []

    class FakeAadcrBackend:
        def __init__(self, settings):
            self.settings = settings

        async def list_rooms(self, user, refresh=False):
            return {"email": user["email"], "refresh": refresh}

    def fake_import(name: str):
        imported.append(name)
        if name == "src.dcr_backends.aadcr_backend":
            return SimpleNamespace(AadcrBackend=FakeAadcrBackend)
        raise AssertionError(f"unexpected adapter import: {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    backend = get_dcr_backend(settings_factory(dcr_backend="aadcrv2"))

    assert imported == []
    result = asyncio.run(backend.list_rooms({"email": "owner@example.test"}, refresh=True))
    assert result == {"email": "owner@example.test", "refresh": True}
    assert imported == ["src.dcr_backends.aadcr_backend"]


def test_aadcr_runtime_requires_shared_secret(settings_factory):
    settings = settings_factory(dcr_backend="aadcrv2", aadcrv2_jwt_secret="")

    with pytest.raises(ValueError, match="AADCRV2_JWT_SECRET is required"):
        settings.validate_runtime()
