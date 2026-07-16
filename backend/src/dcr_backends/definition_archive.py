"""Build byte-for-byte deterministic AADCR definition preview archives."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

from src.dcr_backends.aadcr_translation import DcrOperationError, RoomPlan

_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_definition_archive(plan: RoomPlan) -> bytes:
    """Return a stable ZIP with no source-machine paths or mutable timestamps."""
    members: dict[str, bytes] = {"dcr_config.json": _canonical_json(plan.definition_config())}
    provenance_files: list[dict[str, str]] = []
    for asset in plan.assets:
        if asset.archive_path is None:
            continue
        if asset.archive_path in members:
            raise DcrOperationError(
                detail=f"Definition archive path is duplicated: {asset.archive_path}",
                failed_step="build definition archive",
            )
        try:
            content = asset.path.read_bytes()
        except OSError:
            raise DcrOperationError(
                detail=f"Definition asset {asset.path.name} is no longer available",
                failed_step="build definition archive",
            ) from None
        members[asset.archive_path] = content
        provenance_files.append(
            {
                "archive_path": asset.archive_path,
                "kind": asset.kind,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    provenance = {
        "files": sorted(provenance_files, key=lambda item: item["archive_path"]),
        "format_version": 1,
        "provider": "aadcrv2",
        "synthetic_fixture": bool(plan.synthetic_demo),
    }
    members["fixture-provenance.json"] = _canonical_json(provenance)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for path in sorted(members):
            package.writestr(_zip_info(path), members[path])
    return buffer.getvalue()
