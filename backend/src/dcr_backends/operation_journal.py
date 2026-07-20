"""Append-only, privacy-bounded journal for resumable AADCR operations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_SESSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_FINAL_RESPONSE_FIELDS = frozenset(
    {
        "aggregate_computation_node_id",
        "capabilities",
        "cohort_ids",
        "data_node_ids",
        "dcr_id",
        "dcr_title",
        "dcr_url",
        "environment",
        "handoff_mode",
        "mapping_upload_results",
        "mapping_uploads_successful",
        "merge_request_id",
        "message",
        "metadata_upload_results",
        "metadata_uploads_successful",
        "num_cohorts",
        "participants",
        "provider",
        "row_upload_results",
        "row_uploads_successful",
        "shuffled_upload_results",
        "shuffled_uploads_successful",
    }
)
_FORBIDDEN_KEYS = frozenset({"content", "payload", "record", "records", "row", "row_content", "rows"})
_FORBIDDEN_KEY_PARTS = ("authorization", "cookie", "password", "secret", "token")
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "session_id",
        "timestamp",
        "request_metadata",
        "request_fingerprint",
        "cohort_ids",
        "dcr_id",
        "confirmed_steps",
        "final_response",
    }
)

_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[str, threading.Lock] = {}
_SESSION_LOCKS: dict[tuple[str, str], threading.Lock] = {}


class JournalCorruptionError(ValueError):
    """Raised when a non-trailing journal record cannot be trusted."""


@dataclass(frozen=True)
class JournalState:
    """The latest durable snapshot for one wizard session."""

    session_id: str
    timestamp: datetime
    request_metadata: dict[str, Any]
    request_fingerprint: str
    cohort_ids: tuple[str, ...]
    dcr_id: str | None
    confirmed_steps: tuple[str, ...]
    final_response: dict[str, Any] | None


def validate_session_id(session_id: object) -> str:
    """Return a safe journal key or reject it before touching the filesystem."""
    if not isinstance(session_id, str) or _SESSION_PATTERN.fullmatch(session_id) is None:
        raise ValueError("session_id must be 1-128 URL-safe characters")
    return session_id


def normalize_request_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only request metadata needed to detect safe deterministic retries."""
    normalized: dict[str, Any] = {}
    for field_name in ("dcr_name", "research_question", "creator_email"):
        value = metadata.get(field_name)
        if isinstance(value, str):
            normalized[field_name] = value.strip()[:2000]

    selected_variables = metadata.get("selected_variables")
    if isinstance(selected_variables, dict):
        normalized["selected_variables"] = {
            str(cohort_id): sorted(str(variable) for variable in variables)
            for cohort_id, variables in sorted(selected_variables.items(), key=lambda item: str(item[0]))
            if isinstance(variables, (list, tuple))
        }

    include_samples = metadata.get("include_shuffled_samples")
    if isinstance(include_samples, bool):
        normalized["include_shuffled_samples"] = include_samples
    elif isinstance(include_samples, dict):
        normalized["include_shuffled_samples"] = {
            str(cohort_id): bool(selected)
            for cohort_id, selected in sorted(include_samples.items(), key=lambda item: str(item[0]))
        }

    for field_name in ("include_mapping_upload_slot", "synthetic_demo"):
        value = metadata.get(field_name)
        if isinstance(value, bool):
            normalized[field_name] = value

    for field_name in ("additional_analysts", "excluded_data_owners"):
        values = metadata.get(field_name)
        if isinstance(values, (list, tuple, set)):
            normalized[field_name] = sorted({str(value).strip().lower() for value in values if str(value).strip()})

    raw_mapping_files = metadata.get("selected_mapping_files")
    if isinstance(raw_mapping_files, (list, tuple)):
        mapping_descriptors: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        for raw_mapping in raw_mapping_files:
            if isinstance(raw_mapping, dict):
                raw_name = raw_mapping.get("filename") or raw_mapping.get("filepath")
                raw_cohorts = raw_mapping.get("cohorts")
            else:
                raw_name = raw_mapping
                raw_cohorts = None
            if isinstance(raw_name, str) and raw_name.strip():
                filename = Path(raw_name).name
                normalized_cohorts: list[str] = []
                if isinstance(raw_cohorts, (list, tuple, set)):
                    for raw_cohort in raw_cohorts:
                        cohort = str(raw_cohort).strip()
                        if cohort and cohort not in normalized_cohorts:
                            normalized_cohorts.append(cohort)
                cohorts = tuple(normalized_cohorts)
                mapping_descriptors[(filename, cohorts)] = {
                    "filename": filename,
                    "cohorts": list(cohorts),
                }
        normalized["selected_mapping_files"] = [mapping_descriptors[key] for key in sorted(mapping_descriptors)]

    return normalized


def request_fingerprint(
    metadata: dict[str, Any],
    cohort_ids: Iterable[str],
    *,
    asset_fingerprints: Iterable[dict[str, Any]] = (),
) -> str:
    """Hash normalized request identity plus path-free asset digests, never raw contents."""
    normalized_assets: list[dict[str, str]] = []
    for asset in asset_fingerprints:
        if not isinstance(asset, dict):
            raise ValueError("asset fingerprint must be an object")
        kind = asset.get("kind")
        key = asset.get("key")
        node_name = asset.get("node_name")
        sha256 = asset.get("sha256")
        if not all(isinstance(value, str) and value for value in (kind, key, node_name)):
            raise ValueError("asset fingerprint identity fields must be non-empty strings")
        if not isinstance(sha256, str) or _FINGERPRINT_PATTERN.fullmatch(sha256) is None:
            raise ValueError("asset fingerprint sha256 must be a lowercase SHA-256 digest")
        normalized_assets.append(
            {
                "key": key,
                "kind": kind,
                "node_name": node_name,
                "sha256": sha256,
            }
        )

    payload = {
        "asset_fingerprints": sorted(
            normalized_assets,
            key=lambda asset: (asset["kind"], asset["key"], asset["node_name"], asset["sha256"]),
        ),
        "cohort_ids": sorted({str(cohort_id) for cohort_id in cohort_ids}),
        "request_metadata": normalize_request_metadata(metadata),
    }
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_forbidden_key(key: object) -> bool:
    normalized = str(key).strip().lower()
    return normalized in _FORBIDDEN_KEYS or any(part in normalized for part in _FORBIDDEN_KEY_PARTS)


def _sanitize_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items() if not _is_forbidden_key(key)}
    return str(value)


def _sanitize_final_response(response: dict[str, Any] | None) -> dict[str, Any] | None:
    if response is None:
        return None
    return {key: _sanitize_json(response[key]) for key in sorted(_FINAL_RESPONSE_FIELDS & response.keys())}


def _normalize_steps(steps: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_step in steps:
        step = str(raw_step)
        if not step or len(step) > 300 or any(ord(character) < 32 for character in step):
            raise ValueError("confirmed steps must be non-empty printable strings")
        if step not in normalized:
            normalized.append(step)
    return tuple(normalized)


def _lock_for(registry: dict[Any, threading.Lock], key: Any) -> threading.Lock:
    with _LOCKS_GUARD:
        return registry.setdefault(key, threading.Lock())


class OperationJournal:
    """Persist replayable snapshots without credentials, paths, or row data."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._path_key = str(self.path.resolve())

    @asynccontextmanager
    async def session_lock(self, session_id: object) -> AsyncIterator[None]:
        """Serialize one session across journal instances without blocking the event loop."""
        safe_session_id = validate_session_id(session_id)
        lock = _lock_for(_SESSION_LOCKS, (self._path_key, safe_session_id))
        while not lock.acquire(blocking=False):
            await asyncio.sleep(0.01)
        try:
            yield
        finally:
            lock.release()

    def append(
        self,
        session_id: object,
        *,
        request_metadata: dict[str, Any],
        request_fingerprint: str,
        cohort_ids: Iterable[str],
        dcr_id: str | None,
        confirmed_steps: Iterable[str],
        final_response: dict[str, Any] | None,
    ) -> JournalState:
        """Append one complete snapshot and fsync it before returning."""
        safe_session_id = validate_session_id(session_id)
        if _FINGERPRINT_PATTERN.fullmatch(request_fingerprint) is None:
            raise ValueError("request_fingerprint must be a lowercase SHA-256 digest")
        safe_dcr_id = None if dcr_id is None else str(dcr_id).strip()
        if safe_dcr_id is not None and (
            not safe_dcr_id or len(safe_dcr_id) > 255 or any(ord(character) < 32 for character in safe_dcr_id)
        ):
            raise ValueError("dcr_id must be a printable identifier")

        state = JournalState(
            session_id=safe_session_id,
            timestamp=datetime.now(timezone.utc),
            request_metadata=normalize_request_metadata(request_metadata),
            request_fingerprint=request_fingerprint,
            cohort_ids=tuple(sorted({str(cohort_id) for cohort_id in cohort_ids})),
            dcr_id=safe_dcr_id,
            confirmed_steps=_normalize_steps(confirmed_steps),
            final_response=_sanitize_final_response(final_response),
        )
        record = self._state_record(state)
        encoded = (json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")

        file_lock = _lock_for(_FILE_LOCKS, self._path_key)
        with file_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._read_records_locked(recover_trailing=True)
            needs_separator = self.path.exists() and self.path.stat().st_size > 0
            if needs_separator:
                with self.path.open("rb") as handle:
                    handle.seek(-1, os.SEEK_END)
                    needs_separator = handle.read(1) != b"\n"
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
                if needs_separator:
                    os.write(descriptor, b"\n")
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("failed to append operation journal record")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return state

    def load(self, session_id: object) -> JournalState | None:
        safe_session_id = validate_session_id(session_id)
        states = self._read_records()
        return next(
            (state for state in reversed(states) if state.session_id == safe_session_id),
            None,
        )

    def find_by_dcr_id(self, dcr_id: str) -> JournalState | None:
        target = str(dcr_id)
        return next(
            (state for state in reversed(self._read_records()) if state.dcr_id == target),
            None,
        )

    def latest_timestamp(self) -> datetime | None:
        states = self._read_records()
        return max((state.timestamp for state in states), default=None)

    def _read_records(self) -> list[JournalState]:
        file_lock = _lock_for(_FILE_LOCKS, self._path_key)
        with file_lock:
            return self._read_records_locked(recover_trailing=False)

    def _read_records_locked(self, *, recover_trailing: bool) -> list[JournalState]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
        trailing_index = nonempty_indices[-1] if nonempty_indices else None
        states: list[JournalState] = []
        offset = 0
        for index, line in enumerate(lines):
            line_start = offset
            offset += len(line)
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                state = self._record_state(record)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                is_unterminated_trailing_fragment = index == trailing_index and not line.endswith((b"\n", b"\r"))
                if is_unterminated_trailing_fragment:
                    if recover_trailing:
                        with self.path.open("r+b") as handle:
                            handle.truncate(line_start)
                    break
                raise JournalCorruptionError(f"operation journal is corrupt at line {index + 1}") from exc
            states.append(state)
        return states

    @staticmethod
    def _state_record(state: JournalState) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "session_id": state.session_id,
            "timestamp": state.timestamp.isoformat(),
            "request_metadata": state.request_metadata,
            "request_fingerprint": state.request_fingerprint,
            "cohort_ids": list(state.cohort_ids),
            "dcr_id": state.dcr_id,
            "confirmed_steps": list(state.confirmed_steps),
            "final_response": state.final_response,
        }

    @staticmethod
    def _record_state(record: Any) -> JournalState:
        if not isinstance(record, dict) or set(record) - _RECORD_FIELDS:
            raise ValueError("journal record contains unexpected fields")
        if record.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("unsupported journal schema version")
        session_id = validate_session_id(record.get("session_id"))
        timestamp = datetime.fromisoformat(str(record.get("timestamp")))
        if timestamp.tzinfo is None:
            raise ValueError("journal timestamp must include a timezone")
        metadata = record.get("request_metadata")
        fingerprint = record.get("request_fingerprint")
        cohort_ids = record.get("cohort_ids")
        confirmed_steps = record.get("confirmed_steps")
        final_response = record.get("final_response")
        if not isinstance(metadata, dict):
            raise ValueError("journal request metadata must be an object")
        if not isinstance(fingerprint, str) or _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
            raise ValueError("journal request fingerprint is invalid")
        if not isinstance(cohort_ids, list) or not isinstance(confirmed_steps, list):
            raise ValueError("journal list fields are invalid")
        if final_response is not None and not isinstance(final_response, dict):
            raise ValueError("journal final response must be an object")
        dcr_id = record.get("dcr_id")
        if dcr_id is not None and not isinstance(dcr_id, str):
            raise ValueError("journal room ID is invalid")
        return JournalState(
            session_id=session_id,
            timestamp=timestamp,
            request_metadata=normalize_request_metadata(metadata),
            request_fingerprint=fingerprint,
            cohort_ids=tuple(str(cohort_id) for cohort_id in cohort_ids),
            dcr_id=dcr_id,
            confirmed_steps=_normalize_steps(confirmed_steps),
            final_response=_sanitize_final_response(final_response),
        )
