import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from src.dcr_backends.operation_journal import (
    JournalCorruptionError,
    OperationJournal,
    normalize_request_metadata,
    request_fingerprint,
    validate_session_id,
)


def _metadata(name: str = "Synthetic study") -> dict:
    return {
        "dcr_name": name,
        "research_question": "What do aggregate fixtures show?",
        "selected_variables": {"TIME-CHF": ["AGE"]},
        "include_shuffled_samples": {"TIME-CHF": True},
        "include_mapping_upload_slot": False,
        "selected_mapping_files": ["time-gissi.csv"],
        "synthetic_demo": True,
        "access_token": "must-not-be-persisted",
        "rows": [{"subject_id": "patient-secret"}],
    }


def _append(journal: OperationJournal, session_id: str, **overrides):
    values = {
        "request_metadata": _metadata(),
        "request_fingerprint": "f" * 64,
        "cohort_ids": ["TIME-CHF"],
        "dcr_id": None,
        "confirmed_steps": [],
        "final_response": None,
    }
    values.update(overrides)
    return journal.append(session_id, **values)


@pytest.mark.parametrize(
    "session_id",
    ["", "../session", "session/child", "session\nchild", " session", "session ", "x" * 129],
)
def test_session_keys_fail_closed_before_touching_disk(tmp_path, session_id):
    journal_path = tmp_path / "operations.jsonl"

    with pytest.raises(ValueError, match="session_id"):
        validate_session_id(session_id)
    with pytest.raises(ValueError, match="session_id"):
        _append(OperationJournal(journal_path), session_id)

    assert not journal_path.exists()


def test_atomic_append_and_deterministic_completed_replay_omit_sensitive_content(tmp_path):
    journal_path = tmp_path / "operations.jsonl"
    journal = OperationJournal(journal_path)
    completed_response = {
        "message": "created",
        "dcr_id": "room-123",
        "dcr_url": "http://rooms.test/dcr/room-123",
        "dcr_title": "Synthetic study",
        "cohort_ids": ["TIME-CHF"],
        "num_cohorts": 1,
        "metadata_upload_results": {"TIME-CHF": "success"},
        "metadata_uploads_successful": 1,
        "shuffled_upload_results": {},
        "shuffled_uploads_successful": 0,
        "mapping_upload_results": {},
        "mapping_uploads_successful": 0,
        "row_upload_results": {"TIME-CHF": "success"},
        "row_uploads_successful": 1,
        "participants": {
            "creator@example.test": {
                "data_owner_of": ["TIME-CHF"],
                "analyst_of": ["aggregate-summary-local-simulation"],
            }
        },
        "provider": "aadcrv2",
        "capabilities": {"local_simulation": True},
        "merge_request_id": "merge-1",
        "aggregate_computation_node_id": "prod-aggregate",
        "authorization": "Bearer journal-secret",
        "row_content": [{"subject_id": "patient-secret"}],
    }

    first = _append(journal, "session-123")
    completed = _append(
        journal,
        "session-123",
        dcr_id="room-123",
        confirmed_steps=["room_created", "merged", "completed"],
        final_response=completed_response,
    )
    replayed = journal.load("session-123")

    assert first.timestamp <= completed.timestamp
    assert replayed == completed
    assert replayed.final_response["dcr_id"] == "room-123"
    assert replayed.final_response["participants"] == completed_response["participants"]
    raw = journal_path.read_text(encoding="utf-8")
    assert raw.count("\n") == 2
    assert "must-not-be-persisted" not in raw
    assert "journal-secret" not in raw
    assert "patient-secret" not in raw
    assert '"rows"' not in raw
    assert '"row_content"' not in raw


def test_concurrent_appends_preserve_one_valid_json_record_per_call(tmp_path):
    journal = OperationJournal(tmp_path / "operations.jsonl")

    def append(index: int):
        _append(journal, f"session-{index}", request_fingerprint=f"{index:064x}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(40)))

    lines = journal.path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 40
    assert {record["session_id"] for record in records} == {f"session-{index}" for index in range(40)}


def test_same_session_lock_is_process_local_across_journal_instances(tmp_path):
    path = tmp_path / "operations.jsonl"
    first_journal = OperationJournal(path)
    second_journal = OperationJournal(path)
    active = 0
    maximum_active = 0

    async def worker(journal: OperationJournal):
        nonlocal active, maximum_active
        async with journal.session_lock("session-locked"):
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1

    async def exercise():
        await asyncio.gather(worker(first_journal), worker(second_journal))

    asyncio.run(exercise())

    assert maximum_active == 1


def test_corrupt_trailing_record_is_ignored_then_removed_before_next_append(tmp_path):
    journal = OperationJournal(tmp_path / "operations.jsonl")
    valid = _append(journal, "session-recover")
    with journal.path.open("ab") as handle:
        handle.write(b'{"session_id":"session-recover","timestamp":')

    assert journal.load("session-recover") == valid

    recovered = _append(
        journal,
        "session-recover",
        dcr_id="room-recovered",
        confirmed_steps=["room_created"],
    )
    lines = journal.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line) for line in lines)
    assert journal.load("session-recover") == recovered


def test_newline_terminated_invalid_final_record_fails_closed(tmp_path):
    journal = OperationJournal(tmp_path / "operations.jsonl")
    _append(journal, "session-good")
    with journal.path.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")
    corrupted = journal.path.read_bytes()

    with pytest.raises(JournalCorruptionError, match="line 2"):
        journal.load("session-good")
    with pytest.raises(JournalCorruptionError, match="line 2"):
        _append(journal, "session-later")

    assert journal.path.read_bytes() == corrupted


def test_corruption_before_the_trailing_record_fails_closed(tmp_path):
    journal = OperationJournal(tmp_path / "operations.jsonl")
    _append(journal, "session-good")
    with journal.path.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")
        handle.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": "session-later",
                    "timestamp": "2026-07-15T10:00:00+00:00",
                    "request_metadata": {},
                    "request_fingerprint": "a" * 64,
                    "cohort_ids": [],
                    "dcr_id": None,
                    "confirmed_steps": [],
                    "final_response": None,
                }
            )
            + "\n"
        )

    with pytest.raises(JournalCorruptionError, match="line 2"):
        journal.load("session-good")


def test_latest_timestamp_and_room_lookup_use_durable_records_not_file_mtime(tmp_path):
    journal = OperationJournal(tmp_path / "operations.jsonl")
    first = _append(journal, "session-first", dcr_id="room-first")
    second = _append(journal, "session-second", dcr_id="room-second")
    ancient_mtime = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
    journal.path.touch()
    journal.path.chmod(0o600)
    os.utime(journal.path, (ancient_mtime, ancient_mtime))

    assert journal.latest_timestamp() == second.timestamp
    assert journal.find_by_dcr_id("room-first") == first
    assert journal.find_by_dcr_id("missing") is None


def test_retry_fingerprint_covers_participants_and_mapping_semantics_without_paths():
    metadata = _metadata() | {
        "additional_analysts": ["ANALYST@example.test"],
        "excluded_data_owners": ["OWNER@example.test"],
        "selected_mapping_files": [
            {
                "filename": "time-gissi.csv",
                "filepath": "/private/source/time-gissi.csv",
                "cohorts": ["TIME-CHF", "GISSI-HF"],
            }
        ],
    }

    normalized = normalize_request_metadata(metadata)
    baseline = request_fingerprint(metadata, ["TIME-CHF"])
    changed_participant = request_fingerprint(
        metadata | {"additional_analysts": ["different@example.test"]},
        ["TIME-CHF"],
    )
    changed_mapping = request_fingerprint(
        metadata
        | {
            "selected_mapping_files": [
                {
                    "filename": "time-gissi.csv",
                    "filepath": "/another/private/path/time-gissi.csv",
                    "cohorts": ["GISSI-HF", "TIME-CHF"],
                }
            ]
        },
        ["TIME-CHF"],
    )

    assert normalized["additional_analysts"] == ["analyst@example.test"]
    assert normalized["excluded_data_owners"] == ["owner@example.test"]
    assert normalized["selected_mapping_files"] == [{"filename": "time-gissi.csv", "cohorts": ["TIME-CHF", "GISSI-HF"]}]
    assert "/private/" not in repr(normalized)
    assert baseline != changed_participant
    assert baseline != changed_mapping


def test_retry_fingerprint_binds_stable_asset_identity_and_content_without_paths():
    metadata = _metadata()
    assets = [
        {
            "kind": "metadata",
            "key": "TIME-CHF",
            "node_name": "TIME-CHF_metadata_dictionary",
            "sha256": "a" * 64,
            "path": "/private/first/TIME-CHF_datadictionary.csv",
        },
        {
            "kind": "mapping",
            "key": "time-gissi.csv",
            "node_name": "TIME-CHF_GISSI-HF_mapping",
            "sha256": "b" * 64,
        },
    ]

    baseline = request_fingerprint(metadata, ["TIME-CHF"], asset_fingerprints=assets)
    reordered = request_fingerprint(metadata, ["TIME-CHF"], asset_fingerprints=reversed(assets))
    relocated = request_fingerprint(
        metadata,
        ["TIME-CHF"],
        asset_fingerprints=[assets[0] | {"path": "/another/private/location.csv"}, assets[1]],
    )
    changed_content = request_fingerprint(
        metadata,
        ["TIME-CHF"],
        asset_fingerprints=[assets[0] | {"sha256": "c" * 64}, assets[1]],
    )
    changed_identity = request_fingerprint(
        metadata,
        ["TIME-CHF"],
        asset_fingerprints=[assets[0] | {"node_name": "different_node"}, assets[1]],
    )

    assert baseline == reordered == relocated
    assert baseline != changed_content
    assert baseline != changed_identity
