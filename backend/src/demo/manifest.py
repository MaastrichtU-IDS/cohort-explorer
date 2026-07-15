"""Manifest model and corruption checks for immutable synthetic demo packs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOWER_HEX = frozenset("0123456789abcdef")


class DemoPackError(RuntimeError):
    """The requested pack operation would be unsafe or the pack is invalid."""


@dataclass(frozen=True)
class FileDigest:
    sha256: str
    size_bytes: int

    @property
    def bytes(self) -> int:
        return self.size_bytes

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "bytes": self.size_bytes}


@dataclass(frozen=True)
class CohortPackPaths:
    row_count: int
    dictionary: str
    rows: str
    eda: str
    shuffled_sample: str
    shuffle_summary: str
    images: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "dictionary": self.dictionary,
            "rows": self.rows,
            "eda": self.eda,
            "shuffled_sample": self.shuffled_sample,
            "shuffle_summary": self.shuffle_summary,
            "images": list(self.images),
        }


@dataclass(frozen=True)
class DemoManifest:
    schema_version: int
    generator_version: str
    seed: int
    source_commit: str
    mapping_source: dict[str, str]
    workbook_relative: str
    cohorts: dict[str, CohortPackPaths]
    selected_mapping_rows: tuple[dict[str, str], ...]
    files: dict[str, FileDigest]
    root: Path = field(compare=False, repr=False)

    @property
    def workbook(self) -> Path:
        return self.root / self.workbook_relative

    def dictionary(self, cohort_id: str) -> Path:
        return self.root / self.cohorts[cohort_id].dictionary

    def rows(self, cohort_id: str) -> Path:
        return self.root / self.cohorts[cohort_id].rows

    def shuffled_sample(self, cohort_id: str) -> Path:
        return self.root / self.cohorts[cohort_id].shuffled_sample

    def dictionary_frame(self, cohort_id: str):
        import pandas as pd

        return pd.read_csv(self.dictionary(cohort_id), dtype=str, keep_default_na=False)

    def rows_frame(self, cohort_id: str):
        import pandas as pd

        return pd.read_csv(self.rows(cohort_id))

    def selected_mapping_frame(self):
        import pandas as pd

        return pd.DataFrame(self.selected_mapping_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "source_commit": self.source_commit,
            "mapping_source": dict(self.mapping_source),
            "workbook": self.workbook_relative,
            "cohorts": {
                cohort_id: record.to_dict()
                for cohort_id, record in self.cohorts.items()
            },
            "selected_mapping_rows": [dict(row) for row in self.selected_mapping_rows],
            "files": {
                relative_path: digest.to_dict()
                for relative_path, digest in sorted(self.files.items())
            },
        }


def manifest_bytes(manifest: DemoManifest) -> bytes:
    return (
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _safe_member(root: Path, relative_path: str) -> Path:
    if not relative_path or "\\" in relative_path:
        raise DemoPackError(f"Invalid manifest path: {relative_path!r}")
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise DemoPackError(f"Invalid manifest path: {relative_path!r}")
    resolved_root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise DemoPackError(f"Manifest path escapes the pack: {relative_path!r}")
    return candidate


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise DemoPackError(f"Manifest field {key!r} must be a non-empty string")
    return value


def _cohort_record(cohort_id: str, value: Any) -> CohortPackPaths:
    if not isinstance(value, dict):
        raise DemoPackError(f"Manifest cohort {cohort_id!r} must be an object")
    images = value.get("images")
    if not isinstance(images, list) or not all(isinstance(item, str) for item in images):
        raise DemoPackError(f"Manifest cohort {cohort_id!r} has invalid images")
    row_count = value.get("row_count")
    if not isinstance(row_count, int) or row_count <= 0:
        raise DemoPackError(f"Manifest cohort {cohort_id!r} has invalid row_count")
    return CohortPackPaths(
        row_count=row_count,
        dictionary=_string(value, "dictionary"),
        rows=_string(value, "rows"),
        eda=_string(value, "eda"),
        shuffled_sample=_string(value, "shuffled_sample"),
        shuffle_summary=_string(value, "shuffle_summary"),
        images=tuple(images),
    )


def validate_demo_pack(pack_dir: Path) -> DemoManifest:
    """Load a pack and verify its schema, safe paths, sizes, and SHA-256 hashes."""
    root = Path(pack_dir)
    manifest_path = root / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DemoPackError(f"Demo manifest is missing: {manifest_path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise DemoPackError(f"Demo manifest is unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise DemoPackError("Demo manifest must contain a JSON object")

    if payload.get("schema_version") != 1:
        raise DemoPackError("Unsupported demo manifest schema_version")
    seed = payload.get("seed")
    if not isinstance(seed, int):
        raise DemoPackError("Manifest seed must be an integer")
    source_commit = _string(payload, "source_commit")
    if len(source_commit) != 40 or not set(source_commit) <= LOWER_HEX:
        raise DemoPackError("Manifest source_commit must be a lowercase 40-character SHA")
    mapping_source = payload.get("mapping_source")
    if not isinstance(mapping_source, dict):
        raise DemoPackError("Manifest mapping_source must be an object")
    for key in ("repository_path", "source_commit", "sha256"):
        _string(mapping_source, key)
    mapping_commit = str(mapping_source["source_commit"])
    mapping_sha256 = str(mapping_source["sha256"])
    if len(mapping_commit) != 40 or not set(mapping_commit) <= LOWER_HEX:
        raise DemoPackError(
            "Manifest mapping_source source_commit must be a lowercase 40-character SHA"
        )
    if len(mapping_sha256) != 64 or not set(mapping_sha256) <= LOWER_HEX:
        raise DemoPackError("Manifest mapping_source sha256 must be lowercase hexadecimal")

    raw_cohorts = payload.get("cohorts")
    if not isinstance(raw_cohorts, dict) or set(raw_cohorts) != {"TIME-CHF", "GISSI-HF"}:
        raise DemoPackError("Manifest must describe TIME-CHF and GISSI-HF")
    cohorts = {
        cohort_id: _cohort_record(cohort_id, raw_cohorts[cohort_id])
        for cohort_id in ("TIME-CHF", "GISSI-HF")
    }

    mapping_rows = payload.get("selected_mapping_rows")
    if not isinstance(mapping_rows, list) or not mapping_rows:
        raise DemoPackError("Manifest selected_mapping_rows must be a non-empty list")
    required_mapping_keys = (
        "source",
        "target",
        "mapping_type",
        "source_visit",
        "target_visit",
    )
    if any(
        not isinstance(row, dict) or set(row) != set(required_mapping_keys)
        for row in mapping_rows
    ):
        raise DemoPackError("Manifest selected_mapping_rows has an invalid entry")

    raw_files = payload.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise DemoPackError("Manifest files must be a non-empty object")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    unlisted_files = sorted(actual_files - set(raw_files))
    if unlisted_files:
        raise DemoPackError(
            "Demo pack contains files not listed in the manifest: "
            f"{', '.join(unlisted_files)}"
        )
    files: dict[str, FileDigest] = {}
    for relative_path, raw_digest in raw_files.items():
        if not isinstance(relative_path, str) or not isinstance(raw_digest, dict):
            raise DemoPackError("Manifest file entry is invalid")
        sha256 = raw_digest.get("sha256")
        size_bytes = raw_digest.get("bytes")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise DemoPackError(f"Manifest digest is invalid for {relative_path}")
        path = _safe_member(root, relative_path)
        try:
            content = path.read_bytes()
        except OSError as error:
            raise DemoPackError(f"Manifest file is missing: {relative_path}") from error
        if len(content) != size_bytes:
            raise DemoPackError(f"File size mismatch for {relative_path}")
        if hashlib.sha256(content).hexdigest() != sha256:
            raise DemoPackError(f"SHA-256 mismatch for {relative_path}")
        files[relative_path] = FileDigest(sha256=sha256, size_bytes=size_bytes)

    required_paths = {_string(payload, "workbook")}
    for record in cohorts.values():
        required_paths.update(
            {
                record.dictionary,
                record.rows,
                record.eda,
                record.shuffled_sample,
                record.shuffle_summary,
                *record.images,
            }
        )
    missing_entries = sorted(required_paths - set(files))
    if missing_entries:
        raise DemoPackError(
            f"Manifest files omit required artifacts: {', '.join(missing_entries)}"
        )

    return DemoManifest(
        schema_version=1,
        generator_version=_string(payload, "generator_version"),
        seed=seed,
        source_commit=source_commit,
        mapping_source={key: str(value) for key, value in mapping_source.items()},
        workbook_relative=_string(payload, "workbook"),
        cohorts=cohorts,
        selected_mapping_rows=tuple(
            {key: str(row[key]) for key in required_mapping_keys} for row in mapping_rows
        ),
        files=files,
        root=root,
    )
