"""Canonical runtime storage for cross-cohort mapping artifacts."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MappingCacheStatus:
    filename: str
    fresh: bool
    source_mtime_ns: int
    target_mtime_ns: int


@dataclass(frozen=True)
class MappingArtifact:
    filename: str
    cohorts: tuple[str, ...]
    path: Path
    file_size: int
    timestamp: float
    stats: dict[str, int]

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "cohorts": list(self.cohorts),
            "filename": self.filename,
            "filepath": str(self.path),
            "file_size": self.file_size,
            "timestamp": self.timestamp,
            "display_name": " → ".join(self.cohorts),
            "stats": dict(self.stats),
        }


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Durably replace one artifact without exposing partial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_mapping_json(
    path: Path,
    value: object,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Atomically publish JSON, then its hash-attested completion sidecar."""
    path = Path(path)
    content = (
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
    ).encode("utf-8")
    metadata = dict(provenance)
    output_hashes = metadata.get("output_sha256", {})
    metadata["output_sha256"] = {
        **(output_hashes if isinstance(output_hashes, dict) else {}),
        path.name: hashlib.sha256(content).hexdigest(),
    }
    sidecar_content = (
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n"
    ).encode("utf-8")

    # The sidecar is the completion marker. A reader rejects the JSON until
    # this second atomic replacement attests the exact bytes now at ``path``.
    atomic_write_bytes(path, content)
    atomic_write_bytes(path.with_name(f"{path.name}.meta.json"), sidecar_content)
    return metadata


class MappingArtifactStore:
    """List, validate, retrieve, and freshness-check one configured directory."""

    def __init__(self, output_dir: Path, *, cohorts_root: Path | None = None) -> None:
        if cohorts_root is None:
            from src.config import settings

            cohorts_root = Path(settings.cohort_folder)
        self.output_dir = Path(output_dir)
        self.cohorts_root = Path(cohorts_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def activity_log_path(self) -> Path:
        return self.output_dir / "mapping_activity.jsonl"

    def safe_path(self, filename: str) -> Path:
        """Resolve a downloadable mapping filename without traversal or sidecars."""
        if not filename or "\x00" in filename or "\\" in filename:
            raise ValueError("Invalid mapping filename")
        candidate_name = Path(filename)
        if candidate_name.is_absolute() or candidate_name.name != filename:
            raise ValueError("Mapping filename must not contain a path")
        lower_name = filename.casefold()
        if lower_name.endswith(".meta.json"):
            raise ValueError("Cannot download mapping sidecar files directly")
        if not lower_name.endswith((".csv", ".json")):
            raise ValueError("Unsupported mapping artifact type")
        candidate = (self.output_dir / filename).resolve()
        if candidate.parent != self.output_dir.resolve():
            raise ValueError("Mapping path escapes the configured output directory")
        return candidate

    def _sidecar(self, path: Path) -> Path:
        return path.with_name(f"{path.name}.meta.json")

    def _read_sidecar(self, path: Path) -> dict[str, Any] | None:
        sidecar = self._sidecar(path)
        if not sidecar.is_file():
            return None
        try:
            value = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _verified_result_sidecar(self, path: Path) -> dict[str, Any] | None:
        metadata = self._read_sidecar(path)
        if metadata is None:
            return None
        provider = str(metadata.get("provider", "")).strip().casefold()
        synthetic = metadata.get("synthetic")
        if not (
            (provider == "fixture" and synthetic is True)
            or (provider == "cohortvarlinker" and synthetic is False)
        ):
            return None
        output_hashes = metadata.get("output_sha256", {})
        expected = output_hashes.get(path.name) if isinstance(output_hashes, dict) else None
        if not isinstance(expected, str):
            return None
        try:
            content = path.read_bytes()
            parsed = json.loads(content)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(parsed, dict):
            return None
        actual = hashlib.sha256(content).hexdigest()
        return metadata if actual == expected.casefold() else None

    @staticmethod
    def _normal_target(value: Any) -> str:
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        return str(value).strip().casefold()

    def _cohorts_from_sidecar(self, path: Path) -> tuple[str, ...]:
        sidecar = self._read_sidecar(path)
        parameters = sidecar.get("parameters", {}) if sidecar else {}
        if not isinstance(parameters, dict):
            return ()
        source = str(parameters.get("source_study", "")).strip().casefold()
        targets = parameters.get("target_studies", [])
        if not source or not isinstance(targets, list):
            return ()
        normalized_targets = tuple(
            target for target in (self._normal_target(item) for item in targets) if target
        )
        return (source, *normalized_targets) if normalized_targets else ()

    def _parse_cohorts(self, path: Path) -> tuple[str, ...]:
        from_sidecar = self._cohorts_from_sidecar(path)
        if from_sidecar:
            return from_sidecar

        name = path.name
        if name.casefold().endswith("_full.csv"):
            stem = name[: -len("_full.csv")]
        elif name.casefold().endswith(".json"):
            stem = name[:-len(".json")]
        else:
            return ()

        plus_index = stem.find("+")
        if plus_index >= 0:
            before_plus = stem[:plus_index]
            cohort_part, separator, _model = before_plus.rpartition("_")
            if not separator:
                return ()
        else:
            cohort_part = stem
        cohorts = tuple(part.strip().casefold() for part in cohort_part.split("_") if part.strip())
        return cohorts if len(cohorts) >= 2 else ()

    def _canonical_dictionary(self, cohort_id: str) -> Path | None:
        wanted = cohort_id.casefold()
        if not self.cohorts_root.is_dir():
            return None
        cohort_folder = next(
            (
                item
                for item in self.cohorts_root.iterdir()
                if item.is_dir() and item.name.casefold() == wanted
            ),
            None,
        )
        if cohort_folder is None:
            return None
        expected = cohort_folder / f"{cohort_folder.name}_datadictionary.csv"
        if expected.is_file():
            return expected
        expected_name = expected.name.casefold()
        return next(
            (
                item
                for item in cohort_folder.iterdir()
                if item.is_file() and item.name.casefold() == expected_name
            ),
            None,
        )

    def dictionary_mtime_ns(self, cohort_id: str) -> int:
        path = self._canonical_dictionary(cohort_id)
        return path.stat().st_mtime_ns if path is not None else 0

    @staticmethod
    def _csv_row_count(path: Path) -> int:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                return sum(1 for row in reader if any(cell.strip() for cell in row))
        except OSError:
            return 0

    def _stats(self, path: Path) -> dict[str, int]:
        sidecar = self._read_sidecar(path)
        if sidecar and isinstance(sidecar.get("total_mappings"), int):
            return {"total_mappings": sidecar["total_mappings"]}
        if path.suffix.casefold() == ".csv":
            return {"total_mappings": self._csv_row_count(path)}
        return {"total_mappings": 0}

    def artifact_for(self, path: Path) -> MappingArtifact | None:
        if not path.is_file() or path.name.casefold().endswith(".meta.json"):
            return None
        cohorts = self._parse_cohorts(path)
        if len(cohorts) < 2:
            return None
        stat = path.stat()
        return MappingArtifact(
            filename=path.name,
            cohorts=cohorts,
            path=path,
            file_size=stat.st_size,
            timestamp=stat.st_mtime,
            stats=self._stats(path),
        )

    def cache_status(self, artifact: MappingArtifact) -> MappingCacheStatus:
        dictionary_mtimes = [self.dictionary_mtime_ns(cohort) for cohort in artifact.cohorts]
        all_sources_exist = bool(dictionary_mtimes) and all(dictionary_mtimes)
        source_mtime = max(dictionary_mtimes, default=0)
        try:
            target_mtime = artifact.path.stat().st_mtime_ns
        except FileNotFoundError:
            target_mtime = 0
        return MappingCacheStatus(
            filename=artifact.filename,
            fresh=all_sources_exist and target_mtime >= source_mtime,
            source_mtime_ns=source_mtime,
            target_mtime_ns=target_mtime,
        )

    def find_pair(
        self,
        source_study: str,
        target_study: str,
        *,
        include_stale: bool = True,
    ) -> MappingArtifact | None:
        pair = (source_study.strip().casefold(), target_study.strip().casefold())
        matches: list[MappingArtifact] = []
        for path in self.output_dir.glob("*_full.csv"):
            artifact = self.artifact_for(path)
            if artifact is None or artifact.cohorts != pair:
                continue
            if not include_stale and not self.cache_status(artifact).fresh:
                continue
            matches.append(artifact)
        return max(matches, key=lambda item: item.timestamp, default=None)

    def result_for_request(
        self,
        source_study: str,
        target_studies: list[str],
    ) -> MappingArtifact | None:
        source = source_study.strip().casefold()
        targets = {target.strip().casefold() for target in target_studies}
        matches: list[MappingArtifact] = []
        for path in self.output_dir.glob("*.json"):
            if path.name.casefold().endswith(".meta.json"):
                continue
            if self._verified_result_sidecar(path) is None:
                continue
            artifact = self.artifact_for(path)
            if artifact is None or artifact.cohorts[0] != source:
                continue
            # The live provider may expand one requested target into member
            # studies.  Preserve that contract while still requiring every
            # explicitly requested target to be present.
            if not targets.issubset(set(artifact.cohorts[1:])):
                continue
            if not self.cache_status(artifact).fresh:
                continue
            matches.append(artifact)
        return max(matches, key=lambda item: item.timestamp, default=None)

    def list_for_cohorts(self, cohort_ids: set[str]) -> list[MappingArtifact]:
        selected = {cohort.strip().casefold() for cohort in cohort_ids if cohort.strip()}
        candidates: list[MappingArtifact] = []
        for path in self.output_dir.glob("*_full.csv"):
            artifact = self.artifact_for(path)
            if artifact is None or not set(artifact.cohorts).issubset(selected):
                continue
            if artifact.stats.get("total_mappings", 0) <= 0:
                continue
            if not self.cache_status(artifact).fresh:
                continue
            candidates.append(artifact)

        candidates.sort(key=lambda artifact: artifact.timestamp, reverse=True)
        deduplicated: dict[tuple[str, ...], MappingArtifact] = {}
        for artifact in candidates:
            key = tuple(sorted(artifact.cohorts))
            deduplicated.setdefault(key, artifact)
        return list(deduplicated.values())

    def record_activity(
        self,
        event: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        level: str = "MAIN",
        process: str = "cohort_var_linker",
        depth: int = 0,
    ) -> dict[str, Any]:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "process": process,
            "event": event,
            "msg": message,
            "ctx": context or {},
            "depth": depth,
        }
        line = json.dumps(entry, default=str, ensure_ascii=False) + "\n"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.activity_log_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return entry

    def read_activity(
        self,
        *,
        limit: int,
        level: str | None = None,
        process: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self.activity_log_path.is_file():
            return [], 0
        entries: list[dict[str, Any]] = []
        with self.activity_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                if level and entry.get("level") != level:
                    continue
                if process and entry.get("process") != process:
                    continue
                entries.append(entry)
        return entries[-limit:], len(entries)
