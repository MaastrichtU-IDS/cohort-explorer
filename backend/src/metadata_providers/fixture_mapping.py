"""Deterministic offline mapping provider backed by the tracked mapping CSV."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.mapping_artifacts import MappingArtifactStore, atomic_write_bytes
from src.metadata_providers.contracts import MappingGenerationResult


class MappingProviderError(RuntimeError):
    """Provider failure carrying an HTTP-compatible status and detail."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    ).encode("utf-8")


class FixtureMappingGenerationProvider:
    """Materialize only the reviewed TIME-CHF to GISSI-HF mapping fixture."""

    def __init__(
        self,
        *,
        output_dir: Path | None = None,
        cohorts_root: Path | None = None,
        tracked_mapping_path: Path | None = None,
        profile_path: Path | None = None,
    ) -> None:
        from src.config import settings

        backend_root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir or settings.mapping_output_dir)
        self.tracked_mapping_path = Path(
            tracked_mapping_path
            or backend_root
            / "CohortVarLinker"
            / "mapping_output"
            / "time-chf_gissi-hf_full.csv"
        )
        self.profile_path = Path(
            profile_path or backend_root / "demo" / "metadata-fixtures" / "mapping-profile.json"
        )
        self.store = MappingArtifactStore(self.output_dir, cohorts_root=cohorts_root)

    def _load_profile(self) -> dict[str, Any]:
        try:
            profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MappingProviderError(500, f"Invalid fixture mapping profile: {error}") from error
        if not isinstance(profile, dict):
            raise MappingProviderError(500, "Invalid fixture mapping profile")
        return profile

    @staticmethod
    def _normalize_targets(
        target_studies: Sequence[tuple[str, bool]],
    ) -> list[tuple[str, bool]]:
        normalized: list[tuple[str, bool]] = []
        for target in target_studies:
            if not isinstance(target, (tuple, list)) or len(target) < 1:
                raise MappingProviderError(422, "Invalid target study request")
            normalized.append((str(target[0]).strip().casefold(), bool(target[1]) if len(target) > 1 else False))
        return normalized

    @staticmethod
    def _validate_request(
        profile: dict[str, Any],
        source_study: str,
        targets: list[tuple[str, bool]],
    ) -> tuple[str, str]:
        request = profile.get("supported_request", {})
        expected_source = str(request.get("source_study", "")).casefold()
        expected_target = str(request.get("target_study", "")).casefold()
        source = source_study.strip().casefold()
        if source != expected_source or len(targets) != 1 or targets[0][0] != expected_target:
            requested_targets = ", ".join(target.upper() for target, _constraint in targets) or "none"
            raise MappingProviderError(
                422,
                "Unsupported fixture mapping pair: "
                f"{source_study.upper()} -> {requested_targets}. "
                f"Only {expected_source.upper()} -> {expected_target.upper()} is available.",
            )
        return source, expected_target

    @staticmethod
    def _mapping_json(
        csv_content: bytes,
        source_study: str,
        target_study: str,
    ) -> tuple[dict[str, Any], Counter[str], Counter[str]]:
        text = csv_content.decode("utf-8-sig")
        mappings: dict[str, list[dict[str, Any]]] = defaultdict(list)
        relation_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        for row in csv.DictReader(io.StringIO(text, newline="")):
            source_variable = str(row.pop("source", "")).strip()
            if not source_variable:
                continue
            relation = str(row.pop("mapping type", row.pop("mapping_type", ""))).strip()
            mapping = dict(row)
            mapping["source_study"] = source_study
            mapping["target_study"] = target_study
            mapping["mapping_relation"] = relation
            mapping["harmonization_status"] = "pending"
            mappings[source_variable].append(mapping)
            relation_counts[relation] += 1
            status_counts["pending"] += 1
        combined = {
            source: {"from": source_study, "mappings": values}
            for source, values in sorted(mappings.items())
        }
        return combined, relation_counts, status_counts

    def _cached_outputs_are_valid(
        self,
        source_study: str,
        target_study: str,
        csv_name: str,
        json_name: str,
        source_sha256: str,
    ) -> bool:
        csv_artifact = self.store.find_pair(source_study, target_study, include_stale=False)
        result_artifact = self.store.result_for_request(source_study, [target_study])
        if (
            csv_artifact is None
            or csv_artifact.filename != csv_name
            or result_artifact is None
            or result_artifact.filename != json_name
        ):
            return False
        sidecar_path = self.output_dir / f"{json_name}.meta.json"
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        expected_hashes = sidecar.get("output_sha256", {})
        return (
            sidecar.get("provider") == "fixture"
            and sidecar.get("synthetic") is True
            and sidecar.get("source_sha256") == source_sha256
            and expected_hashes.get(csv_name) == _sha256(csv_artifact.path.read_bytes())
            and expected_hashes.get(json_name) == _sha256(result_artifact.path.read_bytes())
        )

    def _cache_info(self, source_study: str, target_study: str) -> dict[str, Any]:
        artifact = self.store.find_pair(source_study, target_study, include_stale=False)
        return {
            "cached_pairs": [
                {
                    "source": source_study,
                    "target": target_study,
                    "timestamp": artifact.timestamp,
                }
            ]
            if artifact
            else [],
            "uncached_pairs": [] if artifact else [{"source": source_study, "target": target_study}],
        }

    def _mark_outputs_newer_than_dictionaries(self, paths: Sequence[Path], cohorts: Sequence[str]) -> None:
        source_mtime = max((self.store.dictionary_mtime_ns(cohort) for cohort in cohorts), default=0)
        generated_mtime = max(time.time_ns(), source_mtime + 1)
        for path in paths:
            os.utime(path, ns=(generated_mtime, generated_mtime))

    def generate(
        self,
        source_study: str,
        target_studies: Sequence[tuple[str, bool]],
    ) -> MappingGenerationResult:
        profile = self._load_profile()
        targets = self._normalize_targets(target_studies)
        source, target = self._validate_request(profile, source_study, targets)

        try:
            csv_content = self.tracked_mapping_path.read_bytes()
        except OSError as error:
            raise MappingProviderError(500, f"Fixture mapping artifact is unavailable: {error}") from error
        source_sha256 = _sha256(csv_content)
        if source_sha256 != profile.get("source_sha256"):
            raise MappingProviderError(500, "Fixture mapping artifact hash does not match its profile")

        outputs = profile.get("outputs", {})
        csv_name = str(outputs.get("csv", ""))
        json_name = str(outputs.get("json", ""))
        try:
            csv_path = self.store.safe_path(csv_name)
            json_path = self.store.safe_path(json_name)
        except ValueError as error:
            raise MappingProviderError(500, f"Invalid fixture output name: {error}") from error
        meta_path = self.output_dir / f"{json_name}.meta.json"

        if self._cached_outputs_are_valid(source, target, csv_name, json_name, source_sha256):
            self.store.record_activity(
                "fixture_cache_hit",
                f"Reused fresh fixture mapping {source} -> {target}",
                context={"provider": "fixture", "synthetic": True, "source": source, "target": target},
            )
            return MappingGenerationResult(cache_info=self._cache_info(source, target))

        combined, relation_counts, status_counts = self._mapping_json(csv_content, source, target)
        json_content = _json_bytes(combined)
        total_mappings = sum(len(entry["mappings"]) for entry in combined.values())
        parameters = {
            "source_study": source,
            "target_studies": [[name, constraint] for name, constraint in targets],
        }
        meta = {
            "provider": "fixture",
            "synthetic": True,
            "source_commit": profile.get("source_commit"),
            "source_artifact": profile.get("source_artifact"),
            "source_sha256": source_sha256,
            "parameters": parameters,
            "output_sha256": {
                csv_name: _sha256(csv_content),
                json_name: _sha256(json_content),
            },
            "total_mappings": total_mappings,
            "harmonization_status": dict(sorted(status_counts.items())),
            "mapping_relation": dict(sorted(relation_counts.items())),
        }
        meta_content = _json_bytes(meta)

        atomic_write_bytes(csv_path, csv_content)
        atomic_write_bytes(json_path, json_content)
        atomic_write_bytes(meta_path, meta_content)
        self._mark_outputs_newer_than_dictionaries(
            (csv_path, json_path, meta_path),
            (source, target),
        )
        self.store.record_activity(
            "fixture_materialized",
            f"Materialized fixture mapping {source} -> {target}",
            context={
                "provider": "fixture",
                "synthetic": True,
                "source": source,
                "target": target,
                "filename": json_name,
                "total_mappings": total_mappings,
                "source_sha256": source_sha256,
                "output_sha256": meta["output_sha256"],
            },
        )
        return MappingGenerationResult(cache_info=self._cache_info(source, target))
