#!/usr/bin/env python3
"""Verify the fully local Cohort Explorer to AADCR v2 contract."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import secrets
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx
from jose import jwt

from src.demo.manifest import DemoManifest, DemoPackError, validate_demo_pack

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
COHORTS = ("GISSI-HF", "TIME-CHF")
EXPECTED_DEFINITION_ASSETS = {
    "mapping_files/time-chf_gissi-hf_full.csv": "mapping",
    "metadata_dictionaries/GISSI-HF_datadictionary.csv": "metadata",
    "metadata_dictionaries/TIME-CHF_datadictionary.csv": "metadata",
    "shuffled_samples/GISSI-HF_shuffled_sample.csv": "shuffled",
    "shuffled_samples/TIME-CHF_shuffled_sample.csv": "shuffled",
}
EXPECTED_DATA_NODES = {
    "CrossStudyMappings",
    "GISSI-HF",
    "GISSI-HF_metadata_dictionary",
    "GISSI-HF_shuffled_sample",
    "TIME-CHF",
    "TIME-CHF_metadata_dictionary",
    "TIME-CHF_shuffled_sample",
    "time-chf_gissi-hf_mapping",
}
EXPECTED_COMPUTATION_NODES = {
    "aggregate-summary-local-simulation",
    "metadata-preview-local-simulation",
}
JWT_ALGORITHM = "HS256"
MAX_DEFINITION_BYTES = 20 * 1024 * 1024
MAX_RESULT_BYTES = 20 * 1024 * 1024
MAX_ZIP_MEMBERS = 128
MAX_ZIP_EXPANDED_BYTES = 50 * 1024 * 1024
FORBIDDEN_EVIDENCE_TERMS = {
    "authorization",
    "bearer",
    "cookie",
    "patient",
    "row",
    "secret",
    "token",
}
ALLOWED_EVIDENCE_KEYS = {
    "adversarial_check_count",
    "audit_count",
    "cohort_count",
    "data_node_count",
    "definition_member_count",
    "environment",
    "handoff_mode",
    "idempotent",
    "preview_sha256",
    "room_count",
    "room_id",
    "schema_version",
}


class SmokeError(RuntimeError):
    """One safe, response-body-free local smoke failure."""


class SmokeHttp:
    """Fail-closed HTTP boundary restricted to one loopback origin."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in LOOPBACK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise SmokeError("local smoke requests require an HTTP loopback origin")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            transport=transport,
            trust_env=False,
        )

    def __enter__(self) -> SmokeHttp:
        self._client.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._client.__exit__(exc_type, exc_value, traceback)

    @property
    def cookies(self) -> httpx.Cookies:
        return self._client.cookies

    def request(
        self,
        method: str,
        path: str,
        *,
        step: str,
        expected_status: set[int],
        **kwargs: Any,
    ) -> httpx.Response:
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
            raise SmokeError(f"{step} requires a relative local path")
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as error:
            raise SmokeError(f"{step} could not reach the local service") from error
        if response.status_code not in expected_status:
            raise SmokeError(f"{step} failed with HTTP {response.status_code}")
        return response

    def json(
        self,
        method: str,
        path: str,
        *,
        step: str,
        expected_status: set[int],
        required: Mapping[str, type] | None = None,
        **kwargs: Any,
    ) -> Any:
        response = self.request(
            method,
            path,
            step=step,
            expected_status=expected_status,
            **kwargs,
        )
        try:
            payload = response.json()
        except ValueError as error:
            raise SmokeError(f"{step} returned invalid JSON") from error
        if required is not None:
            if not isinstance(payload, dict):
                raise SmokeError(f"{step} returned an invalid object")
            for key, expected_type in required.items():
                value = payload.get(key)
                if not isinstance(value, expected_type) or (expected_type is str and not value):
                    raise SmokeError(f"{step} returned an invalid {key}")
        return payload


def _validate_evidence(value: Any, *, key: str = "evidence") -> None:
    lowered = key.casefold()
    if any(term in lowered for term in FORBIDDEN_EVIDENCE_TERMS):
        raise SmokeError(f"unsafe evidence field: {key}")
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                raise SmokeError("unsafe evidence key type")
            _validate_evidence(nested_value, key=nested_key)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _validate_evidence(item, key=key)
        return
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise SmokeError(f"unsafe evidence value: {key}")


def write_evidence(path: Path, evidence: Mapping[str, Any]) -> None:
    """Atomically write a private, flat, allowlisted evidence summary."""
    unexpected = sorted(set(evidence) - ALLOWED_EVIDENCE_KEYS)
    if unexpected:
        raise SmokeError(f"unsafe evidence field: {unexpected[0]}")
    _validate_evidence(evidence)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(evidence), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _runtime_values(path: Path) -> dict[str, str]:
    """Read the private generated runtime file without evaluating shell syntax."""
    try:
        lines = path.expanduser().resolve(strict=True).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SmokeError("local demo runtime configuration is unavailable") from error
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or "\x00" in value or "\n" in value:
            raise SmokeError("local demo runtime configuration is malformed")
        values[key] = value
    required = {
        "AADCRV2_JWT_SECRET",
        "COMPOSE_PROJECT_NAME",
        "JWT_SECRET",
        "LOCAL_AUTH_EMAIL",
    }
    if not required <= values.keys():
        raise SmokeError("local demo runtime configuration is incomplete")
    for secret_key in ("AADCRV2_JWT_SECRET", "JWT_SECRET"):
        secret = values[secret_key]
        if len(secret) < 64 or any(character not in "0123456789abcdef" for character in secret):
            raise SmokeError("local demo runtime secrets are invalid")
    return values


def _mint_aadcr_token(email: str, secret: str) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "sub": email,
        "email": email,
        "email_verified": True,
        "iss": "cohort-explorer-local",
        "aud": "aadcrv2-local",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
    }
    return jwt.encode(claims, secret, algorithm=JWT_ALGORITHM)


def _safe_zip_members(
    content: bytes,
    *,
    step: str,
    max_bytes: int,
) -> tuple[zipfile.ZipFile, io.BytesIO, tuple[str, ...]]:
    if not content or len(content) > max_bytes:
        raise SmokeError(f"{step} exceeded its archive size limit")
    buffer = io.BytesIO(content)
    try:
        archive = zipfile.ZipFile(buffer)
        members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise SmokeError(f"{step} did not return a valid ZIP archive") from error
    if len(members) > MAX_ZIP_MEMBERS:
        archive.close()
        raise SmokeError(f"{step} returned too many ZIP members")
    names: list[str] = []
    expanded = 0
    for member in members:
        name = member.filename
        path = Path(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or name in names
            or (((member.external_attr >> 16) & 0o170000) == 0o120000)
        ):
            archive.close()
            raise SmokeError(f"{step} returned an unsafe ZIP member")
        expanded += member.file_size
        if expanded > MAX_ZIP_EXPANDED_BYTES:
            archive.close()
            raise SmokeError(f"{step} exceeded its expanded archive size limit")
        names.append(name)
    return archive, buffer, tuple(names)


def _inspect_definition(
    content: bytes,
    *,
    expected_asset_hashes: Mapping[str, str],
    owner_email: str,
    expected_selected_variables: Mapping[str, list[str]],
) -> tuple[str, int]:
    archive, _buffer, names = _safe_zip_members(
        content,
        step="definition preview",
        max_bytes=MAX_DEFINITION_BYTES,
    )
    try:
        expected_members = {
            "dcr_config.json",
            "fixture-provenance.json",
            *EXPECTED_DEFINITION_ASSETS,
        }
        if set(names) != expected_members or set(expected_asset_hashes) != set(EXPECTED_DEFINITION_ASSETS):
            raise SmokeError("definition preview did not contain the exact metadata fixture")
        config_bytes = archive.read("dcr_config.json")
        if b"/Users/" in config_bytes or b"/demo-pack" in config_bytes:
            raise SmokeError("definition preview exposed a host or runtime path")
        try:
            config = json.loads(config_bytes)
            room = config["dataScienceDataRoom"]
        except (KeyError, TypeError, ValueError) as error:
            raise SmokeError("definition preview returned an invalid DCR configuration") from error
        if (
            not isinstance(room, dict)
            or room.get("provider") != "aadcrv2"
            or room.get("synthetic_demo") is not True
            or room.get("local_simulation") is not True
            or room.get("confidential_boundary") is not False
        ):
            raise SmokeError("definition preview returned an unexpected provider contract")
        data_nodes = room.get("data_nodes")
        computation_nodes = room.get("computation_nodes")
        participants = room.get("participants")
        permissions = room.get("permissions")
        expected_permissions = {(owner_email, "DATA_OWNER", name) for name in EXPECTED_DATA_NODES} | {
            (owner_email, "DATA_ANALYST", name) for name in EXPECTED_COMPUTATION_NODES
        }
        actual_permissions = {
            (item.get("email"), item.get("type"), item.get("node"))
            for item in permissions or []
            if isinstance(item, dict)
        }
        if (
            set(room.get("cohorts") or []) != set(COHORTS)
            or not isinstance(data_nodes, list)
            or len(data_nodes) != 8
            or {(node.get("name"), node.get("type")) for node in data_nodes if isinstance(node, dict)}
            != {(name, "FILE") for name in EXPECTED_DATA_NODES}
            or not isinstance(computation_nodes, list)
            or len(computation_nodes) != 2
            or {node.get("name") for node in computation_nodes if isinstance(node, dict)} != EXPECTED_COMPUTATION_NODES
            or not isinstance(participants, list)
            or len(participants) != 1
            or not isinstance(participants[0], dict)
            or participants[0].get("email") != owner_email
            or len(participants[0].get("data_owner_of") or []) != 8
            or set(participants[0].get("data_owner_of") or []) != EXPECTED_DATA_NODES
            or len(participants[0].get("analyst_of") or []) != 2
            or set(participants[0].get("analyst_of") or []) != EXPECTED_COMPUTATION_NODES
            or not isinstance(permissions, list)
            or len(permissions) != 10
            or actual_permissions != expected_permissions
            or room.get("selected_variables")
            != {cohort: sorted(expected_selected_variables[cohort]) for cohort in COHORTS}
        ):
            raise SmokeError("definition preview returned unexpected room semantics")
        try:
            provenance = json.loads(archive.read("fixture-provenance.json"))
        except (TypeError, ValueError) as error:
            raise SmokeError("definition preview returned invalid fixture provenance") from error
        expected_provenance = {
            "files": [
                {
                    "archive_path": path,
                    "kind": EXPECTED_DEFINITION_ASSETS[path],
                    "sha256": expected_asset_hashes[path],
                }
                for path in sorted(EXPECTED_DEFINITION_ASSETS)
            ],
            "format_version": 1,
            "provider": "aadcrv2",
            "synthetic_fixture": True,
        }
        if provenance != expected_provenance:
            raise SmokeError("definition preview did not contain the exact metadata fixture provenance")
        if any(_sha256(archive.read(path)) != expected_asset_hashes[path] for path in EXPECTED_DEFINITION_ASSETS):
            raise SmokeError("definition preview did not contain the expected metadata fixture bytes")
    finally:
        archive.close()
    return _sha256(content), len(names)


def _inspect_result(
    content: bytes,
    *,
    manifest: DemoManifest,
) -> tuple[str, int]:
    archive, _buffer, names = _safe_zip_members(
        content,
        step="aggregate result",
        max_bytes=MAX_RESULT_BYTES,
    )
    try:
        if set(names) != {"aggregate-summary.json"}:
            raise SmokeError("aggregate result violated the aggregate-only archive contract")
        try:
            summary = json.loads(archive.read("aggregate-summary.json"))
            nodes = summary["nodes"]
        except (KeyError, TypeError, ValueError) as error:
            raise SmokeError("aggregate result returned invalid JSON") from error
        expected = {
            "confidential_boundary": False,
            "local_simulation": True,
            "nodes": _expected_aggregate_nodes(manifest),
        }
        if not isinstance(nodes, dict) or summary != expected:
            raise SmokeError("aggregate result did not match the exact fixture aggregate")
    finally:
        archive.close()
    return _sha256(content), len(names)


def _expected_aggregate_nodes(manifest: DemoManifest) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        try:
            with manifest.rows(cohort).open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                if not fieldnames or any(not name for name in fieldnames) or len(set(fieldnames)) != len(fieldnames):
                    raise SmokeError("synthetic row fixture has an invalid CSV header")
                row_count = 0
                non_empty = {column: 0 for column in fieldnames}
                numeric = {column: {"count": 0, "maximum": None, "minimum": None, "sum": 0.0} for column in fieldnames}
                for row in reader:
                    row_count += 1
                    if set(row) != set(fieldnames):
                        raise SmokeError("synthetic row fixture has an invalid CSV row")
                    for column, value in row.items():
                        if value is None or not value.strip():
                            continue
                        non_empty[column] += 1
                        try:
                            numeric_value = float(value)
                        except ValueError:
                            continue
                        state = numeric[column]
                        state["count"] += 1
                        state["sum"] += numeric_value
                        if state["minimum"] is None or numeric_value < state["minimum"]:
                            state["minimum"] = numeric_value
                        if state["maximum"] is None or numeric_value > state["maximum"]:
                            state["maximum"] = numeric_value
        except (OSError, csv.Error) as error:
            raise SmokeError("synthetic row fixture could not be aggregated") from error
        numeric_summary = {
            column: {
                "count": state["count"],
                "maximum": state["maximum"],
                "mean": state["sum"] / state["count"],
                "minimum": state["minimum"],
            }
            for column, state in numeric.items()
            if state["count"]
        }
        nodes[cohort] = {
            "non_empty": non_empty,
            "numeric": numeric_summary,
            "rows": row_count,
        }
    return nodes


def _expected_definition_asset_hashes(manifest: DemoManifest) -> dict[str, str]:
    return {
        "mapping_files/time-chf_gissi-hf_full.csv": manifest.mapping_source["sha256"],
        "metadata_dictionaries/GISSI-HF_datadictionary.csv": manifest.files[
            manifest.cohorts["GISSI-HF"].dictionary
        ].sha256,
        "metadata_dictionaries/TIME-CHF_datadictionary.csv": manifest.files[
            manifest.cohorts["TIME-CHF"].dictionary
        ].sha256,
        "shuffled_samples/GISSI-HF_shuffled_sample.csv": manifest.files[
            manifest.cohorts["GISSI-HF"].shuffled_sample
        ].sha256,
        "shuffled_samples/TIME-CHF_shuffled_sample.csv": manifest.files[
            manifest.cohorts["TIME-CHF"].shuffled_sample
        ].sha256,
    }


def _selected_variables(manifest: DemoManifest) -> dict[str, list[str]]:
    source: list[str] = []
    target: list[str] = []
    for row in manifest.selected_mapping_rows:
        source_name = row["source"]
        target_name = row["target"]
        if source_name not in source:
            source.append(source_name)
        if target_name not in target:
            target.append(target_name)
        if len(source) >= 6 and len(target) >= 6:
            break
    if not source or not target:
        raise SmokeError("synthetic pack does not contain selected mapping variables")
    return {"TIME-CHF": source[:6], "GISSI-HF": target[:6]}


def _login_and_seed(client: SmokeHttp, manifest: DemoManifest) -> dict[str, Any]:
    login = client.request(
        "GET",
        "/login",
        step="local admin login",
        expected_status={302, 303, 307, 308},
    )
    if not login.headers.get("location") or not client.cookies.get("token"):
        raise SmokeError("local admin login did not establish a session")

    client.json(
        "POST",
        "/upload-cohorts-metadata",
        step="central workbook upload",
        expected_status={200},
        required={"message": str},
        files={
            "cohorts_metadata": (
                manifest.workbook.name,
                manifest.workbook.read_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    for cohort in COHORTS:
        dictionary = manifest.dictionary(cohort)
        upload = (dictionary.name, dictionary.read_bytes(), "text/csv")
        client.json(
            "POST",
            "/validate-cohort-dictionary",
            step=f"{cohort} dictionary validation",
            expected_status={200},
            required={"identifier": str, "concepts_valid": bool},
            data={"cohort_id": cohort},
            files={"cohort_dictionary": upload},
        )
        client.json(
            "POST",
            "/upload-cohort",
            step=f"{cohort} dictionary upload",
            expected_status={200},
            required={"identifier": str, "message": str},
            data={"cohort_id": cohort},
            files={"cohort_dictionary": upload},
        )

    metadata = client.json(
        "GET",
        "/cohorts-metadata",
        step="metadata read",
        expected_status={200},
    )
    if not isinstance(metadata, dict):
        raise SmokeError("metadata read returned an invalid object")
    for cohort in COHORTS:
        cohort_metadata = metadata.get(cohort)
        if not isinstance(cohort_metadata, dict) or not isinstance(cohort_metadata.get("variables"), dict):
            raise SmokeError(f"metadata read omitted {cohort} variables")
    return metadata


def _generate_mapping(client: SmokeHttp) -> dict[str, Any]:
    before = client.json(
        "POST",
        "/api/get-available-mapping-files",
        step="fresh mapping baseline",
        expected_status={200},
        required={"available_mappings": list, "cohort_count": int},
        json=["TIME-CHF", "GISSI-HF"],
    )
    if before["available_mappings"]:
        raise SmokeError("local smoke requires a fresh mapping runtime")
    client.json(
        "POST",
        "/api/generate-mapping",
        step="fixture mapping generation",
        expected_status={200},
        required={"filename": str, "file_content": str},
        json={
            "source_study": "TIME-CHF",
            "target_studies": [["GISSI-HF", False]],
        },
    )
    listed = client.json(
        "POST",
        "/api/get-available-mapping-files",
        step="mapping artifact listing",
        expected_status={200},
        required={"available_mappings": list, "cohort_count": int},
        json=["TIME-CHF", "GISSI-HF"],
    )
    mappings = listed["available_mappings"]
    if len(mappings) != 1 or not isinstance(mappings[0], dict):
        raise SmokeError("mapping artifact listing did not return exactly one mapping")
    mapping = mappings[0]
    for key in ("cohorts", "display_name", "filename", "filepath"):
        if key not in mapping:
            raise SmokeError(f"mapping artifact listing omitted {key}")
    if mapping["filename"] != "time-chf_gissi-hf_full.csv":
        raise SmokeError("mapping artifact listing returned an unexpected mapping")
    return mapping


def _build_room_request(
    manifest: DemoManifest,
    mapping: Mapping[str, Any],
    *,
    run_nonce: str | None = None,
) -> dict[str, Any]:
    """Build the exact deterministic request exercised by the local wizard lane."""
    selected = _selected_variables(manifest)
    mapping_fields: dict[str, Any] = {}
    for key in ("filename", "filepath", "display_name", "cohorts"):
        value = mapping.get(key)
        if key == "cohorts":
            if (
                not isinstance(value, list)
                or len(value) < 2
                or not all(isinstance(cohort, str) and cohort for cohort in value)
            ):
                raise SmokeError("mapping artifact has invalid cohort semantics")
            mapping_fields[key] = list(value)
        elif not isinstance(value, str) or not value:
            raise SmokeError(f"mapping artifact has invalid {key}")
        else:
            mapping_fields[key] = value

    try:
        manifest_digest = _sha256((manifest.root / "manifest.json").read_bytes())
    except OSError as error:
        raise SmokeError("synthetic pack manifest could not be fingerprinted") from error
    if run_nonce is not None and (
        len(run_nonce) != 24 or any(character not in "0123456789abcdef" for character in run_nonce)
    ):
        raise SmokeError("smoke run nonce is invalid")
    session_suffix = run_nonce or manifest_digest[:24]
    return {
        "cohorts": {
            "GISSI-HF": selected["GISSI-HF"],
            "TIME-CHF": selected["TIME-CHF"],
        },
        "include_shuffled_samples": {cohort: True for cohort in COHORTS},
        "additional_analysts": [],
        "excluded_data_owners": [],
        "airlock_settings": {cohort: 0 for cohort in COHORTS},
        "dcr_name": "Synthetic cardiovascular cohort aggregate demo",
        "research_question": (
            "What aggregate completeness and numeric summaries are available "
            "across the two synthetic cardiovascular cohorts?"
        ),
        "session_id": f"local-demo-{session_suffix}",
        "selected_mapping_files": [mapping_fields],
        "include_mapping_upload_slot": True,
    }


def _require_string(payload: Mapping[str, Any], key: str, *, step: str) -> str:
    value = payload.get(key)
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 255
        or any(character in value for character in "/\\?#")
        or any(ord(character) < 32 for character in value)
    ):
        raise SmokeError(f"{step} returned an invalid {key}")
    return value


def _inspect_live_creation(
    payload: Any,
    *,
    owner_email: str,
    frontend_url: str,
) -> tuple[str, str, str]:
    if not isinstance(payload, dict):
        raise SmokeError("live room creation returned an invalid object")
    room_id = _require_string(payload, "dcr_id", step="live room creation")
    merge_id = _require_string(payload, "merge_request_id", step="live room creation")
    computation_id = _require_string(
        payload,
        "aggregate_computation_node_id",
        step="live room creation",
    )
    expected_counts = {
        "metadata_uploads_successful": 2,
        "row_uploads_successful": 2,
        "shuffled_uploads_successful": 2,
        "mapping_uploads_successful": 1,
        "num_cohorts": 2,
    }
    for key, expected in expected_counts.items():
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected:
            raise SmokeError(f"live room creation returned an invalid {key}")
    success_by_cohort = {cohort: "success" for cohort in COHORTS}
    expected_participants = {
        owner_email: {
            "data_owner_of": sorted(EXPECTED_DATA_NODES),
            "analyst_of": sorted(EXPECTED_COMPUTATION_NODES),
        }
    }
    dcr_url = urlsplit(str(payload.get("dcr_url") or ""))
    frontend_origin = urlsplit(frontend_url)
    if (
        payload.get("provider") != "aadcrv2"
        or len(payload.get("cohort_ids") or []) != 2
        or set(payload.get("cohort_ids") or []) != set(COHORTS)
        or payload.get("metadata_upload_results") != success_by_cohort
        or payload.get("row_upload_results") != success_by_cohort
        or payload.get("shuffled_upload_results") != success_by_cohort
        or payload.get("mapping_upload_results") != {"time-chf_gissi-hf_full.csv": "success"}
        or payload.get("participants") != expected_participants
        or dcr_url.scheme != "http"
        or dcr_url.hostname not in LOOPBACK_HOSTS
        or dcr_url.port != frontend_origin.port
        or dcr_url.path != "/dcrs"
        or parse_qsl(dcr_url.query, keep_blank_values=True) != [("room", room_id)]
        or dcr_url.fragment
        or dcr_url.username is not None
        or dcr_url.password is not None
        or not isinstance(payload.get("capabilities"), dict)
        or payload["capabilities"].get("local_simulation") is not True
        or payload["capabilities"].get("synthetic_data_only") is not True
    ):
        raise SmokeError("live room creation returned an unexpected provider contract")
    return room_id, merge_id, computation_id


def _inspect_bootstrap_creation(
    payload: Any,
    *,
    aadcr_ui_url: str,
) -> tuple[str, dict[str, str]]:
    """Validate that Cohort Explorer stops at the explicit AADCR UI handoff."""
    if not isinstance(payload, dict):
        raise SmokeError("live room creation returned an invalid object")
    room_id = _require_string(payload, "dcr_id", step="live room creation")
    raw_node_ids = payload.get("data_node_ids")
    if not isinstance(raw_node_ids, dict) or set(raw_node_ids) != EXPECTED_DATA_NODES:
        raise SmokeError("live room creation returned unexpected DEV data slots")
    node_ids = {
        name: _require_string(raw_node_ids, name, step="live room creation")
        for name in sorted(EXPECTED_DATA_NODES)
    }
    if len(set(node_ids.values())) != len(node_ids):
        raise SmokeError("live room creation returned duplicate DEV data slot IDs")

    handoff = urlsplit(str(payload.get("dcr_url") or ""))
    expected_origin = urlsplit(aadcr_ui_url)
    empty_results = (
        "metadata_upload_results",
        "row_upload_results",
        "shuffled_upload_results",
        "mapping_upload_results",
    )
    zero_counts = (
        "metadata_uploads_successful",
        "row_uploads_successful",
        "shuffled_uploads_successful",
        "mapping_uploads_successful",
    )
    if (
        payload.get("provider") != "aadcrv2"
        or payload.get("handoff_mode") != "bootstrap"
        or payload.get("environment") != "DEV"
        or set(payload.get("cohort_ids") or []) != set(COHORTS)
        or payload.get("num_cohorts") != len(COHORTS)
        or payload.get("participants") != {}
        or payload.get("merge_request_id") is not None
        or payload.get("aggregate_computation_node_id") is not None
        or any(payload.get(key) != {} for key in empty_results)
        or any(payload.get(key) != 0 for key in zero_counts)
        or handoff.scheme != "http"
        or handoff.hostname not in LOOPBACK_HOSTS
        or handoff.port != expected_origin.port
        or handoff.path != f"/aadcrv2/dcr/{room_id}"
        or handoff.query
        or handoff.fragment
        or handoff.username is not None
        or handoff.password is not None
        or not isinstance(payload.get("capabilities"), dict)
        or payload["capabilities"].get("local_simulation") is not True
        or payload["capabilities"].get("synthetic_data_only") is not True
    ):
        raise SmokeError("live room creation returned an unexpected bootstrap contract")
    return room_id, node_ids


def _bearer_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _native_room_count(
    client: SmokeHttp,
    *,
    headers: Mapping[str, str],
    step: str,
) -> int:
    rooms = client.json(
        "GET",
        "/api/dcr/",
        step=step,
        expected_status={200},
        headers=headers,
    )
    room_ids = [room.get("id") for room in rooms if isinstance(room, dict)] if isinstance(rooms, list) else []
    if (
        not isinstance(rooms, list)
        or len(room_ids) != len(rooms)
        or any(not isinstance(room_id, str) or not room_id for room_id in room_ids)
        or len(set(room_ids)) != len(room_ids)
    ):
        raise SmokeError(f"{step} returned an invalid room list")
    return len(rooms)


def _inspect_native_bootstrap(
    client: SmokeHttp,
    *,
    headers: Mapping[str, str],
    room_id: str,
    expected_node_ids: Mapping[str, str],
) -> int:
    """Prove the native room contains only CE-created DEV data-slot changes."""
    room = client.json(
        "GET",
        f"/api/dcr/{room_id}",
        step="native room read",
        expected_status={200},
        required={"id": str, "name": str},
        headers=headers,
    )
    if room["id"] != room_id:
        raise SmokeError("native room read returned the wrong room")

    dev = client.json(
        "GET",
        f"/api/dcr/{room_id}/dev/view",
        step="native DEV read",
        expected_status={200},
        headers=headers,
    )
    if not isinstance(dev, dict):
        raise SmokeError("native DEV read returned an invalid object")
    data_nodes = dev.get("data_nodes")
    actual_node_ids = {
        node.get("name"): node.get("id")
        for node in data_nodes or []
        if isinstance(node, dict)
        and isinstance(node.get("name"), str)
        and isinstance(node.get("id"), str)
    }
    if (
        not isinstance(data_nodes, list)
        or len(data_nodes) != len(EXPECTED_DATA_NODES)
        or actual_node_ids != dict(expected_node_ids)
        or dev.get("participants") != []
        or dev.get("computation_nodes") != []
        or dev.get("permissions") != []
    ):
        raise SmokeError("native DEV graph crossed the metadata-only handoff boundary")

    prod = client.json(
        "GET",
        f"/api/dcr/{room_id}/prod/view",
        step="native PROD read",
        expected_status={200},
        headers=headers,
    )
    if not isinstance(prod, dict) or any(
        prod.get(collection) != []
        for collection in ("participants", "data_nodes", "computation_nodes", "permissions")
    ):
        raise SmokeError("native PROD graph was unexpectedly populated before handoff")

    provisions = client.json(
        "GET",
        f"/api/dcr/{room_id}/provisioned-datasets",
        step="native provisioning read",
        expected_status={200},
        required={"provisioned_datasets": list},
        headers=headers,
    )
    if provisions["provisioned_datasets"]:
        raise SmokeError("native room was unexpectedly provisioned before handoff")

    merge_requests = client.json(
        "GET",
        f"/api/dcr/{room_id}/merge-requests/",
        step="native merge-request read",
        expected_status={200},
        required={"mergeRequests": list},
        headers=headers,
    )
    if merge_requests["mergeRequests"]:
        raise SmokeError("native room had a merge request before handoff")

    audit = client.json(
        "GET",
        f"/api/dcr/{room_id}/audit-logs",
        step="native audit read",
        expected_status={200},
        required={"audit_logs": list},
        headers=headers,
    )["audit_logs"]
    if not audit:
        raise SmokeError("native audit read returned no entries")
    return len(audit)


def _inspect_native_room(
    client: SmokeHttp,
    *,
    headers: Mapping[str, str],
    room_id: str,
    merge_id: str,
    computation_id: str,
    owner_email: str,
) -> tuple[str, int, int, tuple[dict[str, str], ...]]:
    room = client.json(
        "GET",
        f"/api/dcr/{room_id}",
        step="native room read",
        expected_status={200},
        required={"id": str, "name": str},
        headers=headers,
    )
    if room["id"] != room_id:
        raise SmokeError("native room read returned the wrong room")
    merge = client.json(
        "GET",
        f"/api/dcr/{room_id}/merge-requests/{merge_id}",
        step="native merge read",
        expected_status={200},
        required={"id": str, "status": str},
        headers=headers,
    )
    if merge["id"] != merge_id or str(merge["status"]).upper() != "MERGED":
        raise SmokeError("native merge did not reach MERGED")

    dev = client.json(
        "GET",
        f"/api/dcr/{room_id}/dev/view",
        step="native DEV read",
        expected_status={200},
        headers=headers,
    )
    if not isinstance(dev, dict):
        raise SmokeError("native DEV read returned an invalid object")
    for collection_name in (
        "participants",
        "data_nodes",
        "computation_nodes",
        "permissions",
    ):
        collection = dev.get(collection_name)
        if not isinstance(collection, list):
            raise SmokeError(f"native DEV read omitted {collection_name}")
        if any(
            isinstance(item, dict) and str(item.get("nodeStatus") or "").upper() in {"DEV_ADDED", "DEV_PROD_REMOVED"}
            for item in collection
        ):
            raise SmokeError("native DEV graph retained an unmerged change")

    prod = client.json(
        "GET",
        f"/api/dcr/{room_id}/prod/view",
        step="native PROD read",
        expected_status={200},
        headers=headers,
    )
    if not isinstance(prod, dict):
        raise SmokeError("native PROD read returned an invalid object")
    for collection_name in ("participants", "data_nodes", "computation_nodes", "permissions"):
        collection = prod.get(collection_name)
        if not isinstance(collection, list):
            raise SmokeError(f"native PROD read omitted {collection_name}")

    participants = prod["participants"]
    if (
        len(participants) != 1
        or not isinstance(participants[0], dict)
        or participants[0].get("userEmail") != owner_email
    ):
        raise SmokeError("native PROD read returned unexpected participants")

    data_node_by_id = {
        node.get("id"): node.get("name")
        for node in prod["data_nodes"]
        if isinstance(node, dict) and isinstance(node.get("id"), str) and isinstance(node.get("name"), str)
    }
    if (
        len(prod["data_nodes"]) != 8
        or len(data_node_by_id) != 8
        or set(data_node_by_id.values()) != EXPECTED_DATA_NODES
    ):
        raise SmokeError("native PROD read returned an unexpected data-node graph")

    computation_by_name = {
        node.get("name"): node.get("id")
        for node in prod["computation_nodes"]
        if isinstance(node, dict) and isinstance(node.get("id"), str) and isinstance(node.get("name"), str)
    }
    if (
        len(prod["computation_nodes"]) != 2
        or len(computation_by_name) != 2
        or set(computation_by_name) != EXPECTED_COMPUTATION_NODES
        or computation_by_name.get("aggregate-summary-local-simulation") != computation_id
    ):
        raise SmokeError("native PROD read did not preserve the aggregate node")

    expected_permissions = {(owner_email, "DATA_OWNER", resource_name) for resource_name in EXPECTED_DATA_NODES} | {
        (owner_email, "DATA_ANALYST", resource_name) for resource_name in EXPECTED_COMPUTATION_NODES
    }
    actual_permissions = {
        (
            permission.get("userEmail"),
            permission.get("permissionType"),
            permission.get("resourceName"),
        )
        for permission in prod["permissions"]
        if isinstance(permission, dict)
    }
    if len(prod["permissions"]) != 10 or actual_permissions != expected_permissions:
        raise SmokeError("native PROD read returned unexpected permission semantics")

    provisioned = client.json(
        "GET",
        f"/api/dcr/{room_id}/provisioned-datasets",
        step="native provisioning read",
        expected_status={200},
        required={"provisioned_datasets": list},
        headers=headers,
    )["provisioned_datasets"]
    expected_asset_names = EXPECTED_DATA_NODES - {"CrossStudyMappings"}
    dataset_ids: set[str] = set()
    provisioned_names: set[str] = set()
    provisioned_node_ids: set[str] = set()
    normalized_provisions: list[dict[str, str]] = []
    for item in provisioned:
        if not isinstance(item, dict):
            raise SmokeError("native provisioning read returned invalid asset coverage")
        dataset_id = item.get("dataset_id")
        node_id = item.get("dataset_node_id")
        dataset_name = item.get("dataset_name")
        if (
            not isinstance(dataset_id, str)
            or not dataset_id
            or not isinstance(node_id, str)
            or data_node_by_id.get(node_id) != dataset_name
            or dataset_name not in expected_asset_names
            or item.get("provision_type") != "PROD"
        ):
            raise SmokeError("native provisioning read returned invalid asset coverage")
        dataset_ids.add(dataset_id)
        provisioned_names.add(dataset_name)
        provisioned_node_ids.add(node_id)
        normalized_provisions.append(
            {
                "dataset_id": dataset_id,
                "dataset_node_id": node_id,
                "dataset_name": dataset_name,
                "provision_type": "PROD",
            }
        )
    if (
        len(provisioned) != 7
        or len(dataset_ids) != 7
        or len(provisioned_node_ids) != 7
        or provisioned_names != expected_asset_names
    ):
        raise SmokeError("native provisioning read returned invalid asset coverage")

    audit = client.json(
        "GET",
        f"/api/dcr/{room_id}/audit-logs",
        step="native audit read",
        expected_status={200},
        required={"audit_logs": list},
        headers=headers,
    )["audit_logs"]
    if not audit:
        raise SmokeError("native audit read returned no entries")
    return (
        sorted(dataset_ids)[0],
        len(provisioned),
        len(audit),
        tuple(sorted(normalized_provisions, key=lambda item: item["dataset_name"])),
    )


def _inspect_ce_reads(
    client: SmokeHttp,
    *,
    room_id: str,
    owner_email: str,
    native_provisions: Sequence[Mapping[str, str]],
    expected_room_count: int,
    frontend_url: str,
) -> tuple[int, int]:
    rooms = client.json(
        "GET",
        "/my-dcrs",
        step="Cohort Explorer room read",
        expected_status={200},
        required={"dcrs": list, "count": int, "provider": str},
    )
    if rooms["provider"] != "aadcrv2" or rooms["count"] != len(rooms["dcrs"]) or rooms["count"] != expected_room_count:
        raise SmokeError("Cohort Explorer room read returned an invalid provider contract")
    matching = [room for room in rooms["dcrs"] if isinstance(room, dict) and room.get("id") == room_id]
    if len(matching) != 1:
        raise SmokeError("Cohort Explorer room read omitted the created room")
    room = matching[0]
    room_url = urlsplit(str(room.get("dcr_url") or ""))
    frontend_origin = urlsplit(frontend_url)
    capabilities = room.get("capabilities")
    if (
        set(room.get("cohorts") or []) != set(COHORTS)
        or room.get("owner") != {"email": owner_email}
        or room.get("provider") != "aadcrv2"
        or not isinstance(room.get("title"), str)
        or not room["title"]
        or room_url.scheme != "http"
        or room_url.hostname not in LOOPBACK_HOSTS
        or room_url.port != frontend_origin.port
        or room_url.path != "/dcrs"
        or parse_qsl(room_url.query, keep_blank_values=True) != [("room", room_id)]
        or not isinstance(capabilities, dict)
        or capabilities.get("local_simulation") is not True
        or capabilities.get("synthetic_data_only") is not True
    ):
        raise SmokeError("Cohort Explorer room read returned unexpected wrapper semantics")
    participants = room.get("participants")
    if (
        not isinstance(participants, list)
        or len(participants) != 1
        or not isinstance(participants[0], dict)
        or participants[0].get("email") != owner_email
        or len(participants[0].get("data_owner_of") or []) != 8
        or set(participants[0].get("data_owner_of") or []) != EXPECTED_DATA_NODES
        or len(participants[0].get("analyst_of") or []) != 2
        or set(participants[0].get("analyst_of") or []) != EXPECTED_COMPUTATION_NODES
    ):
        raise SmokeError("Cohort Explorer room read returned unexpected participant roles")
    nodes = room.get("nodes")
    if (
        not isinstance(nodes, list)
        or len(nodes) != 10
        or {(node.get("name"), node.get("type")) for node in nodes if isinstance(node, dict)}
        != {
            *((name, "RawDataNodeDefinition") for name in EXPECTED_DATA_NODES),
            *((name, "PythonComputeNodeDefinition") for name in EXPECTED_COMPUTATION_NODES),
        }
    ):
        raise SmokeError("Cohort Explorer room read returned unexpected node semantics")
    ce_provisions = room.get("provisioned_datasets")
    if not isinstance(ce_provisions, list) or len(ce_provisions) != 7:
        raise SmokeError("Cohort Explorer room read omitted provisioned datasets")
    normalized_ce_provisions: list[dict[str, str]] = []
    for provision in ce_provisions:
        if not isinstance(provision, dict):
            raise SmokeError("Cohort Explorer room read returned invalid provisioning semantics")
        normalized = {
            key: provision.get(key) for key in ("dataset_id", "dataset_node_id", "dataset_name", "provision_type")
        }
        if (
            not all(isinstance(value, str) and value for value in normalized.values())
            or provision.get("node_name") != normalized["dataset_name"]
            or provision.get("status") != "provisioned"
        ):
            raise SmokeError("Cohort Explorer room read returned invalid provisioning semantics")
        normalized_ce_provisions.append(normalized)  # type: ignore[arg-type]
    expected_provisions = sorted(native_provisions, key=lambda item: item["dataset_name"])
    if sorted(normalized_ce_provisions, key=lambda item: item["dataset_name"]) != expected_provisions:
        raise SmokeError("Cohort Explorer room read changed native provisioning semantics")
    audit = client.json(
        "GET",
        f"/dcr-log/{room_id}",
        step="Cohort Explorer audit read",
        expected_status={200},
    )
    if not isinstance(audit, list) or not audit:
        raise SmokeError("Cohort Explorer audit read returned no entries")
    for entry in audit:
        if not isinstance(entry, dict) or set(entry) not in (
            {"timestamp", "user", "desc", "provider"},
            {"timestamp", "user", "desc", "provider", "source"},
        ):
            raise SmokeError("Cohort Explorer audit read returned unsafe fields")
        if (
            entry.get("provider") != "aadcrv2"
            or entry.get("user") != owner_email
            or any(not isinstance(entry.get(key), str) for key in ("timestamp", "user", "desc", "provider"))
            or len(entry["desc"]) > 300
        ):
            raise SmokeError("Cohort Explorer audit read returned an invalid provider")
        source = entry.get("source")
        if source is not None and (
            not isinstance(source, dict)
            or not set(source) <= {"id", "method", "path"}
            or any(not isinstance(value, str) for value in source.values())
        ):
            raise SmokeError("Cohort Explorer audit read returned unsafe source fields")
    return rooms["count"], len(audit)


def _run_adversarial_checks(
    native: SmokeHttp,
    *,
    headers: Mapping[str, str],
    token: str,
    secret: str,
    room_id: str,
    dataset_id: str,
    main_data_node_id: str,
    upload_limit_bytes: int,
    expected_room_count: int,
) -> int:
    tampered = _tamper_token_signature(token)
    native.request(
        "GET",
        "/api/dcr/",
        step="tampered token rejection",
        expected_status={401},
        headers=_bearer_headers(tampered),
    )
    outsider = _mint_aadcr_token("outsider@example.test", secret)
    native.request(
        "GET",
        f"/api/dcr/{room_id}",
        step="outsider room rejection",
        expected_status={403},
        headers=_bearer_headers(outsider),
    )

    with tempfile.TemporaryFile() as oversized:
        oversized.write(b"value\n1\n")
        oversized.seek(upload_limit_bytes)
        oversized.write(b"x")
        oversized.seek(0)
        native.request(
            "POST",
            "/api/upload",
            step="oversized upload rejection",
            expected_status={413},
            headers=headers,
            data={"dataset_name": "oversized-adversarial-check"},
            files={"file": ("oversized.csv", oversized, "text/csv")},
        )

    temporary_room_id: str | None = None
    try:
        temporary_room = native.json(
            "POST",
            "/api/dcr/",
            step="cross-room fixture creation",
            expected_status={200},
            required={"id": str},
            headers=headers,
            json={"name": "Cross-room containment smoke fixture"},
        )
        temporary_room_id = _require_string(
            temporary_room,
            "id",
            step="cross-room fixture creation",
        )
        native.request(
            "POST",
            f"/api/dcr/{temporary_room_id}/provision-dataset",
            step="cross-room node rejection",
            expected_status={404},
            headers=headers,
            json={
                "dataset_id": dataset_id,
                "dataset_node_id": main_data_node_id,
                "provision_type": "PROD",
            },
        )
    finally:
        if temporary_room_id is not None:
            native.request(
                "DELETE",
                f"/api/dcr/{temporary_room_id}",
                step="cross-room fixture cleanup",
                expected_status={200},
                headers=headers,
            )
            native.request(
                "GET",
                f"/api/dcr/{temporary_room_id}",
                step="cross-room fixture deletion verification",
                expected_status={404},
                headers=headers,
            )
            if (
                _native_room_count(
                    native,
                    headers=headers,
                    step="cross-room fixture room-count restoration",
                )
                != expected_room_count
            ):
                raise SmokeError("cross-room fixture cleanup did not restore the room count")
    return 4


def _run_bootstrap_adversarial_checks(
    native: SmokeHttp,
    *,
    headers: Mapping[str, str],
    token: str,
    secret: str,
    room_id: str,
    upload_limit_bytes: int,
) -> int:
    """Exercise auth and upload limits without mutating the bootstrapped room."""
    native.request(
        "GET",
        "/api/dcr/",
        step="tampered token rejection",
        expected_status={401},
        headers=_bearer_headers(_tamper_token_signature(token)),
    )
    outsider = _mint_aadcr_token("outsider@example.test", secret)
    native.request(
        "GET",
        f"/api/dcr/{room_id}",
        step="outsider room rejection",
        expected_status={403},
        headers=_bearer_headers(outsider),
    )
    with tempfile.TemporaryFile() as oversized:
        oversized.write(b"value\n1\n")
        oversized.seek(upload_limit_bytes)
        oversized.write(b"x")
        oversized.seek(0)
        native.request(
            "POST",
            "/api/upload",
            step="oversized upload rejection",
            expected_status={413},
            headers=headers,
            data={"dataset_name": "oversized-adversarial-check"},
            files={"file": ("oversized.csv", oversized, "text/csv")},
        )
    return 3


def _tamper_token_signature(token: str) -> str:
    """Change signed bytes, avoiding no-op base64url padding-bit mutations."""
    parts = token.split(".")
    if len(parts) != 3 or len(parts[2]) < 2:
        raise SmokeError("cannot tamper malformed local smoke token")
    signature = parts[2]
    replacement = "A" if signature[0] != "A" else "B"
    parts[2] = f"{replacement}{signature[1:]}"
    return ".".join(parts)


def _run_command(command: list[str], *, step: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603 - bounded argv is assembled internally
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SmokeError(f"{step} could not run") from error


def _assert_requested_origins_match_gateway(
    bindings: Mapping[str, Any],
    *,
    base_url: str,
    aadcr_url: str,
    aadcr_ui_url: str,
) -> dict[str, str]:
    expected_ports = {"18000/tcp", "3000/tcp", "3001/tcp", "3002/tcp"}
    if set(bindings) != expected_ports:
        raise SmokeError("selected gateway has unexpected published ports")
    published: dict[str, int] = {}
    for container_port in expected_ports:
        targets = bindings.get(container_port)
        if (
            not isinstance(targets, list)
            or len(targets) != 1
            or not isinstance(targets[0], dict)
            or targets[0].get("HostIp") != "127.0.0.1"
        ):
            raise SmokeError("selected gateway is not bound exclusively to loopback")
        try:
            published[container_port] = int(targets[0]["HostPort"])
        except (KeyError, TypeError, ValueError) as error:
            raise SmokeError("selected gateway has an invalid host port") from error

    def canonical(url: str, *, expected_port: int, label: str) -> tuple[str, str]:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in LOOPBACK_HOSTS
            or parsed.port != expected_port
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise SmokeError(f"{label} origin does not match the selected gateway")
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        return f"http://{host}:{expected_port}", parsed.hostname

    backend, backend_host = canonical(
        base_url,
        expected_port=published["3000/tcp"],
        label="Cohort Explorer",
    )
    aadcr, _aadcr_host = canonical(
        aadcr_url,
        expected_port=published["18000/tcp"],
        label="AADCR",
    )
    aadcr_ui, _aadcr_ui_host = canonical(
        aadcr_ui_url,
        expected_port=published["3002/tcp"],
        label="AADCR UI",
    )
    frontend_host = f"[{backend_host}]" if ":" in backend_host else backend_host
    return {
        "aadcr": aadcr,
        "aadcr_ui": aadcr_ui,
        "backend": backend,
        "frontend": f"http://{frontend_host}:{published['3001/tcp']}",
    }


def _assert_internal_network_and_no_public_egress(
    *,
    project_name: str,
    runtime_env: Path,
    compose_files: Sequence[Path],
    base_url: str,
    aadcr_url: str,
    aadcr_ui_url: str,
) -> dict[str, str]:
    compose = [
        "docker",
        "compose",
        "--project-name",
        project_name,
        "--env-file",
        str(runtime_env),
    ]
    for compose_file in compose_files:
        compose.extend(("-f", str(compose_file)))
    container_ids: dict[str, str] = {}
    for service in ("backend", "frontend", "db", "aadcrv2", "aadcrv2-ui", "gateway"):
        lookup = _run_command(
            [*compose, "ps", "-q", service],
            step=f"{service} container lookup",
        )
        container_id = lookup.stdout.strip()
        if lookup.returncode != 0 or not container_id or "\n" in container_id:
            raise SmokeError(f"{service} container lookup returned an invalid container")
        container_ids[service] = container_id

    network_internal: dict[str, bool] = {}
    gateway_bindings: Mapping[str, Any] | None = None
    for service, container_id in container_ids.items():
        inspection = _run_command(
            [
                "docker",
                "inspect",
                container_id,
                "--format",
                "{{json .NetworkSettings.Networks}}",
            ],
            step=f"{service} network inspection",
        )
        try:
            networks = json.loads(inspection.stdout)
        except (TypeError, ValueError) as error:
            raise SmokeError(f"{service} network inspection returned invalid JSON") from error
        if inspection.returncode != 0 or not isinstance(networks, dict) or not networks:
            raise SmokeError(f"{service} network inspection returned no networks")
        for network_name in networks:
            if network_name not in network_internal:
                internal = _run_command(
                    ["docker", "network", "inspect", network_name, "--format", "{{.Internal}}"],
                    step="demo network inspection",
                )
                if internal.returncode != 0:
                    raise SmokeError("demo network inspection failed")
                network_internal[network_name] = internal.stdout.strip().casefold() == "true"
        if service != "gateway" and (
            len(networks) != 1 or not all(network_internal[network_name] for network_name in networks)
        ):
            raise SmokeError(f"{service} is not isolated on the internal demo network")

    gateway_id = container_ids["gateway"]
    gateway_networks_result = _run_command(
        [
            "docker",
            "inspect",
            gateway_id,
            "--format",
            "{{json .NetworkSettings.Networks}}",
        ],
        step="gateway network boundary inspection",
    )
    try:
        gateway_networks = json.loads(gateway_networks_result.stdout)
    except (TypeError, ValueError) as error:
        raise SmokeError("gateway network boundary inspection returned invalid JSON") from error
    expected_gateway_networks = {
        f"{project_name}_demo_ingress": False,
        f"{project_name}_demo_internal": True,
    }
    if (
        gateway_networks_result.returncode != 0
        or not isinstance(gateway_networks, dict)
        or {name: network_internal.get(name) for name in gateway_networks} != expected_gateway_networks
    ):
        raise SmokeError("gateway did not preserve the ingress/internal network boundary")

    for service, container_id in container_ids.items():
        bindings_result = _run_command(
            [
                "docker",
                "inspect",
                container_id,
                "--format",
                "{{json .HostConfig.PortBindings}}",
            ],
            step=f"{service} port-binding inspection",
        )
        try:
            bindings = json.loads(bindings_result.stdout)
        except (TypeError, ValueError) as error:
            raise SmokeError(f"{service} port-binding inspection returned invalid JSON") from error
        if bindings_result.returncode != 0 or not isinstance(bindings, dict):
            raise SmokeError(f"{service} port-binding inspection failed")
        if service != "gateway" and bindings:
            raise SmokeError(f"{service} unexpectedly published a direct host port")
        if service == "gateway":
            gateway_bindings = bindings

    if gateway_bindings is None:
        raise SmokeError("selected gateway did not expose port bindings")
    origins = _assert_requested_origins_match_gateway(
        gateway_bindings,
        base_url=base_url,
        aadcr_url=aadcr_url,
        aadcr_ui_url=aadcr_ui_url,
    )
    for label, path in {
        "aadcr": "/health",
        "aadcr_ui": "/healthz",
        "backend": "/health",
        "frontend": "/api/health",
    }.items():
        try:
            response = httpx.get(
                f"{origins[label]}{path}",
                headers={"Host": "attacker.example"},
                timeout=5.0,
                trust_env=False,
            )
        except httpx.HTTPError:
            continue
        if response.status_code < 400:
            raise SmokeError("gateway accepted a non-local Host header")

    egress_probe = _run_command(
        [
            "docker",
            "exec",
            container_ids["aadcrv2"],
            "python",
            "-c",
            (
                "import sys,urllib.request\n"
                "try:\n"
                " urllib.request.urlopen('https://example.com',timeout=3).read(1)\n"
                "except Exception:\n"
                " sys.exit(0)\n"
                "sys.exit(1)"
            ),
        ],
        step="public egress rejection",
        timeout=10.0,
    )
    if egress_probe.returncode != 0:
        raise SmokeError("AADCR container unexpectedly reached a public network")
    return origins


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    try:
        manifest = validate_demo_pack(args.pack)
    except DemoPackError as error:
        raise SmokeError("synthetic demo pack validation failed") from error
    runtime = _runtime_values(args.runtime_env)
    email = runtime["LOCAL_AUTH_EMAIL"].strip().lower()
    if email != "nikolas.molyndris@decentriq.ch":
        raise SmokeError("local demo admin is not the required user")
    origins = _assert_internal_network_and_no_public_egress(
        project_name=args.project_name,
        runtime_env=args.runtime_env,
        compose_files=args.compose_file,
        base_url=args.base_url,
        aadcr_url=args.aadcr_url,
        aadcr_ui_url=args.aadcr_ui_url,
    )
    token = _mint_aadcr_token(email, runtime["AADCRV2_JWT_SECRET"])
    headers = _bearer_headers(token)
    with SmokeHttp(origins["aadcr"], timeout=args.timeout) as baseline_native:
        if (
            _native_room_count(
                baseline_native,
                headers=headers,
                step="fresh native room baseline",
            )
            != 0
        ):
            raise SmokeError("local smoke requires fresh mutable runtime volumes")

    with SmokeHttp(origins["backend"], timeout=args.timeout) as cohort_explorer:
        _login_and_seed(cohort_explorer, manifest)
        mapping = _generate_mapping(cohort_explorer)
        room_request = _build_room_request(
            manifest,
            mapping,
            run_nonce=secrets.token_hex(12),
        )
        preview = cohort_explorer.request(
            "POST",
            "/get-compute-dcr-definition",
            step="definition preview",
            expected_status={200},
            json=room_request,
        )
        preview_replay = cohort_explorer.request(
            "POST",
            "/get-compute-dcr-definition",
            step="deterministic definition replay",
            expected_status={200},
            json=room_request,
        )
        if preview_replay.content != preview.content:
            raise SmokeError("definition preview was not byte-for-byte deterministic")
        preview_sha256, definition_member_count = _inspect_definition(
            preview.content,
            expected_asset_hashes=_expected_definition_asset_hashes(manifest),
            owner_email=email,
            expected_selected_variables={cohort: list(room_request["cohorts"][cohort]) for cohort in COHORTS},
        )
        created = cohort_explorer.json(
            "POST",
            "/create-live-compute-dcr",
            step="live room creation",
            expected_status={200},
            json=room_request,
        )
        room_id, data_node_ids = _inspect_bootstrap_creation(
            created,
            aadcr_ui_url=origins["aadcr_ui"],
        )

        with SmokeHttp(origins["aadcr"], timeout=args.timeout) as native:
            native_audit_count = _inspect_native_bootstrap(
                native,
                headers=headers,
                room_id=room_id,
                expected_node_ids=data_node_ids,
            )
            replay = cohort_explorer.json(
                "POST",
                "/create-live-compute-dcr",
                step="idempotent room replay",
                expected_status={200},
                json=room_request,
            )
            replay_room_id, replay_data_node_ids = _inspect_bootstrap_creation(
                replay,
                aadcr_ui_url=origins["aadcr_ui"],
            )
            room_count = _native_room_count(
                native,
                headers=headers,
                step="idempotent room count",
            )
            if room_count != 1 or (replay_room_id, replay_data_node_ids) != (
                room_id,
                data_node_ids,
            ):
                raise SmokeError("idempotent room replay duplicated or changed the room")
            adversarial_check_count = 1 + _run_bootstrap_adversarial_checks(
                native,
                headers=headers,
                token=token,
                secret=runtime["AADCRV2_JWT_SECRET"],
                room_id=room_id,
                upload_limit_bytes=args.upload_limit_bytes,
            )

    evidence = {
        "schema_version": 1,
        "room_id": room_id,
        "preview_sha256": preview_sha256,
        "cohort_count": len(COHORTS),
        "data_node_count": len(data_node_ids),
        "room_count": room_count,
        "audit_count": native_audit_count,
        "definition_member_count": definition_member_count,
        "handoff_mode": "bootstrap",
        "environment": "DEV",
        "idempotent": True,
        "adversarial_check_count": adversarial_check_count,
    }
    write_evidence(args.evidence, evidence)
    return evidence


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--aadcr-url", default="http://127.0.0.1:18000")
    parser.add_argument("--aadcr-ui-url", default="http://127.0.0.1:3002")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument(
        "--compose-file",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--upload-limit-bytes", type=int, default=25 * 1024 * 1024)
    arguments = parser.parse_args()
    if not arguments.compose_file:
        arguments.compose_file = [
            root / "docker-compose.yml",
            root / "docker-compose.local-aadcr.yml",
        ]
    return arguments


def main() -> int:
    try:
        evidence = run_smoke(build_parser())
    except SmokeError as error:
        print(f"smoke error: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
