from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.auth import get_current_user
from src.dcr_backends.models import DcrCapabilities, DcrListResult, LiveCreateResult
from src.dcr_routes import get_backend, router


def capabilities() -> DcrCapabilities:
    return DcrCapabilities(
        supports_provisioning=True,
        supports_definition_preview=True,
        supports_live_creation=True,
        supports_room_refresh=True,
        supports_audit_log=True,
        supports_computation_output=True,
        supports_shuffle_output=False,
        synthetic_data_only=True,
        local_simulation=True,
    )


class FakeBackend:
    provider_name = "aadcrv2"
    capabilities = capabilities()

    def __init__(self):
        self.calls: list[tuple[Any, ...]] = []

    async def create_provision_room(self, request, user):
        self.calls.append(("provision", request, user))
        return {"message": "provisioned", "cohort_id": request["cohort_id"]}

    async def preview_definition(self, request, user):
        self.calls.append(("preview", request, user))
        return {"kind": "definition", "cohorts": request["cohorts"]}

    async def create_live_room(self, request, user):
        self.calls.append(("live", request, user))
        return LiveCreateResult(
            message="created",
            dcr_id="room-123",
            dcr_url="http://localhost:8001/api/dcr/room-123",
            dcr_title=request["dcr_name"],
            cohort_ids=list(request["cohorts"]),
            num_cohorts=len(request["cohorts"]),
            provider=self.provider_name,
            capabilities=self.capabilities,
        )

    async def list_rooms(self, user, refresh=False):
        self.calls.append(("list", user, refresh))
        return DcrListResult(
            dcrs=[],
            count=0,
            email=user["email"],
            provider=self.provider_name,
            capabilities=self.capabilities,
            refresh_summary={"refreshed": refresh},
        )

    async def rooms_last_modified(self, user):
        self.calls.append(("last-modified", user))
        return datetime(2026, 7, 15, 12, 30, tzinfo=timezone.utc)

    async def audit_log(self, dcr_id, user, main_only=True):
        self.calls.append(("audit", dcr_id, user, main_only))
        description = "Created room" if main_only else "log has been retrieved"
        return [{"timestamp": "2026-07-15T12:00:00Z", "user": user["email"], "desc": description}]

    async def computation_output(self, dcr_id, user):
        self.calls.append(("computation", dcr_id, user))
        return {"status": "success", "dcr_id": dcr_id}

    async def shuffle_output(self, dcr_id, user):
        self.calls.append(("shuffle", dcr_id, user))
        return {"status": "unsupported", "dcr_id": dcr_id}


@pytest.fixture
def fake_backend():
    return FakeBackend()


@pytest.fixture
def route_client(fake_backend):
    user = {"email": "nikolas.molyndris@decentriq.ch"}
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_backend] = lambda: fake_backend
    with TestClient(app) as client:
        yield client, fake_backend, user


def test_creation_routes_delegate_unchanged_requests(route_client):
    client, backend, user = route_client
    request = {
        "cohorts": {"TIME-CHF": ["heart_rate"]},
        "dcr_name": "Local heart failure analysis",
        "session_id": "session-123",
    }

    provision = client.post("/create-provision-dcr", data={"cohort_id": "TIME-CHF"})
    preview = client.post("/get-compute-dcr-definition", json=request)
    live = client.post("/create-live-compute-dcr", json=request)

    assert provision.status_code == 200
    assert provision.json() == {"message": "provisioned", "cohort_id": "TIME-CHF"}
    assert preview.json() == {"kind": "definition", "cohorts": {"TIME-CHF": ["heart_rate"]}}
    assert live.json()["dcr_id"] == "room-123"
    assert live.json()["provider"] == "aadcrv2"
    assert backend.calls == [
        ("provision", {"cohort_id": "TIME-CHF"}, user),
        ("preview", request, user),
        ("live", request, user),
    ]


def test_audit_and_output_aliases_delegate_path_parameters(route_client):
    client, backend, user = route_client

    audit = client.get("/dcr-log/room-123")
    main_audit = client.get("/dcr-log-main/room-123")
    computation = client.get("/compute-get-output/room-123")
    shuffle = client.get("/shuffle-get-output/room-123")

    assert audit.status_code == main_audit.status_code == 200
    assert audit.json()[0]["desc"] == "log has been retrieved"
    assert main_audit.json()[0]["desc"] == "Created room"
    assert computation.json() == {"status": "success", "dcr_id": "room-123"}
    assert shuffle.json() == {"status": "unsupported", "dcr_id": "room-123"}
    assert backend.calls == [
        ("audit", "room-123", user, False),
        ("audit", "room-123", user, True),
        ("computation", "room-123", user),
        ("shuffle", "room-123", user),
    ]


def test_room_list_refresh_and_timestamp_routes_preserve_wrappers(route_client):
    client, backend, user = route_client

    rooms = client.get("/my-dcrs")
    refreshed = client.post("/my-dcrs/refresh")
    modified = client.get("/my-dcrs/last-modified")

    assert rooms.json() == {
        "dcrs": [],
        "count": 0,
        "email": user["email"],
        "provider": "aadcrv2",
        "capabilities": capabilities().to_dict(),
        "refresh_summary": {"refreshed": False},
    }
    assert refreshed.json()["refresh_summary"] == {"refreshed": True}
    assert modified.json() == {"last_modified": "2026-07-15T12:30:00+00:00"}
    assert backend.calls == [
        ("list", user, False),
        ("list", user, True),
        ("last-modified", user),
    ]
