import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from src.dcr_backends.decentriq_backend import DecentriqBackend


def run(coroutine):
    return asyncio.run(coroutine)


def test_adapter_does_not_create_client_until_an_operation(settings_factory, monkeypatch):
    from src import decentriq as legacy

    def fail_create_client(*_args, **_kwargs):
        raise AssertionError("Decentriq client initialized during adapter construction")

    monkeypatch.setattr(legacy.dq, "create_client", fail_create_client)

    backend = DecentriqBackend(settings_factory(dcr_backend="decentriq"))

    assert backend.provider_name == "decentriq"
    assert backend.capabilities.supports_shuffle_output is True


def test_adapter_wraps_creation_without_changing_legacy_payloads(settings_factory, monkeypatch):
    from src import cohort_cache
    from src import decentriq as legacy

    cohort = SimpleNamespace(cohort_id="TIME-CHF", can_edit=True)
    monkeypatch.setattr(cohort_cache, "get_cohorts_from_cache", lambda _email: {"TIME-CHF": cohort})
    monkeypatch.setattr(legacy, "create_provision_dcr", lambda user, selected: {"owner": user["email"], "cohort": selected.cohort_id})

    async def preview(request, user):
        return {"request": request, "email": user["email"]}

    async def create_live(request, user):
        return {
            "message": "created",
            "dcr_id": "room-123",
            "dcr_url": "https://platform.decentriq.com/datarooms/p/room-123",
            "dcr_title": request["dcr_name"],
            "cohort_ids": list(request["cohorts"]),
            "num_cohorts": len(request["cohorts"]),
            "metadata_upload_results": {"TIME-CHF": "success"},
            "metadata_uploads_successful": 1,
            "shuffled_upload_results": {},
            "shuffled_uploads_successful": 0,
            "mapping_upload_results": {},
            "mapping_uploads_successful": 0,
            "participants": {},
        }

    monkeypatch.setattr(legacy, "api_get_compute_dcr_definition", preview)
    monkeypatch.setattr(legacy, "api_create_live_compute_dcr", create_live)
    backend = DecentriqBackend(settings_factory(dcr_backend="decentriq"))
    user = {"email": "owner@example.test"}
    request = {"cohorts": {"TIME-CHF": ["heart_rate"]}, "dcr_name": "Legacy room"}

    assert run(backend.create_provision_room({"cohort_id": "TIME-CHF"}, user)) == {
        "owner": "owner@example.test",
        "cohort": "TIME-CHF",
    }
    assert run(backend.preview_definition(request, user)) == {"request": request, "email": user["email"]}
    created = run(backend.create_live_room(request, user)).to_dict()
    assert created["dcr_id"] == "room-123"
    assert created["provider"] == "decentriq"
    assert created["metadata_upload_results"] == {"TIME-CHF": "success"}


def test_adapter_normalizes_list_audit_outputs_and_last_modified(tmp_path, settings_factory, monkeypatch):
    from src import decentriq as legacy

    refresh_calls: list[str] = []
    history = tmp_path / "dcr_history.jsonl"
    history.write_text("{}\n", encoding="utf-8")
    fixed_timestamp = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc).timestamp()
    history.touch()
    monkeypatch.setattr(legacy, "_dcr_history_path", lambda: str(history))
    monkeypatch.setattr(legacy.os.path, "getmtime", lambda _path: fixed_timestamp)
    monkeypatch.setattr(legacy, "load_dcr_history_from_disk", lambda: refresh_calls.append("load") or 1)
    monkeypatch.setattr(
        legacy,
        "refresh_dcrs_in_memory_only",
        lambda: refresh_calls.append("refresh") or {"processed": 1},
    )
    monkeypatch.setattr(
        legacy,
        "get_dcrs_for_participant",
        lambda _email: [{"id": "room-123", "title": "Existing room", "participants": [], "nodes": []}],
    )
    monkeypatch.setattr(
        legacy,
        "get_dcr_log_main",
        lambda dcr_id, user: [{"timestamp": "now", "user": user["email"], "desc": dcr_id}],
    )
    monkeypatch.setattr(
        legacy,
        "get_dcr_log",
        lambda dcr_id, user: [{"timestamp": "now", "user": user["email"], "desc": f"full:{dcr_id}"}],
    )
    backend = DecentriqBackend(settings_factory(dcr_backend="decentriq"))
    user = {"email": "owner@example.test"}

    listed = run(backend.list_rooms(user, refresh=True)).to_dict()
    assert refresh_calls == ["load", "refresh"]
    assert listed["refresh_summary"] == {"processed": 1}
    assert listed["dcrs"][0]["provider"] == "decentriq"
    assert listed["dcrs"][0]["dcr_url"].endswith("/room-123")
    assert run(backend.audit_log("room-123", user))[0]["desc"] == "room-123"
    assert run(backend.audit_log("room-123", user, main_only=False))[0]["desc"] == "full:room-123"
    assert run(backend.rooms_last_modified(user)) == datetime.fromtimestamp(fixed_timestamp, timezone.utc)
