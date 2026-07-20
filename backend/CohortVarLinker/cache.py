"""Cache eligibility for CohortVarLinker's normal mapping pipeline."""

import hashlib
import json
from pathlib import Path
from collections.abc import Sequence


def _canonical_dictionary(cohorts_root: Path, cohort_id: str) -> Path | None:
    wanted = cohort_id.strip().casefold()
    if not cohorts_root.is_dir():
        return None
    cohort_dir = next(
        (
            candidate
            for candidate in cohorts_root.iterdir()
            if candidate.is_dir() and candidate.name.casefold() == wanted
        ),
        None,
    )
    if cohort_dir is None:
        return None
    expected_name = f"{cohort_dir.name}_datadictionary.csv".casefold()
    return next(
        (
            candidate
            for candidate in cohort_dir.iterdir()
            if candidate.is_file() and candidate.name.casefold() == expected_name
        ),
        None,
    )


def select_cached_mapping_csv(
    source_study: str,
    target_study: str,
    output_dir: str | Path,
    *,
    cohorts_root: str | Path,
    expected_suffix: str | None = None,
) -> Path | None:
    """Return only a current normal-provider CSV for the requested pair."""
    source = source_study.strip().casefold()
    target = target_study.strip().casefold()
    suffix = expected_suffix.strip() if expected_suffix is not None else None
    if not source or not target or (suffix is not None and Path(suffix).name != suffix):
        return None

    dictionaries = (
        _canonical_dictionary(Path(cohorts_root), source),
        _canonical_dictionary(Path(cohorts_root), target),
    )
    if any(dictionary is None for dictionary in dictionaries):
        return None
    newest_dictionary_mtime = max(
        dictionary.stat().st_mtime_ns
        for dictionary in dictionaries
        if dictionary is not None
    )

    output = Path(output_dir)
    prefix = f"{source}_{target}_"
    expected_name = f"{prefix}{suffix}".casefold() if suffix is not None else None

    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def attestation(path: Path) -> str | None:
        path_digest: str | None = None
        normal_attested = False
        for sidecar in output.glob("*.meta.json"):
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            output_hashes = metadata.get("output_sha256", {})
            expected_hash = output_hashes.get(path.name) if isinstance(output_hashes, dict) else None
            if not isinstance(expected_hash, str):
                continue
            path_digest = path_digest or digest(path)
            if path_digest != expected_hash:
                continue
            provider = str(metadata.get("provider", "")).strip().casefold()
            if metadata.get("synthetic") is True or provider == "fixture":
                return "synthetic"
            if provider == "cohortvarlinker" and metadata.get("synthetic") is False:
                normal_attested = True
        return "normal" if normal_attested else None

    def is_normal_candidate(candidate: Path) -> bool:
        name = candidate.name.casefold()
        if not candidate.is_file() or not name.startswith(prefix) or not name.endswith("_full.csv"):
            return False
        if expected_name is not None and name != expected_name:
            return False
        provenance = attestation(candidate)
        if provenance == "synthetic":
            return False
        tag = name[len(prefix) : -len("_full.csv")]
        model, separator, configuration = tag.partition("+")
        model_tagged = bool(model and separator and "_" in configuration)
        return model_tagged or provenance == "normal"

    candidates = (
        candidate
        for candidate in output.glob(f"{source}_{target}_*.csv")
        if is_normal_candidate(candidate)
    )
    fresh = [
        candidate
        for candidate in candidates
        if candidate.stat().st_mtime_ns >= newest_dictionary_mtime
    ]
    return max(fresh, key=lambda candidate: candidate.stat().st_mtime_ns, default=None)


def require_cached_mapping_csvs(
    source_study: str,
    target_studies: Sequence[str],
    output_dir: str | Path,
    *,
    cohorts_root: str | Path,
) -> dict[str, Path]:
    """Select every requested input before a combiner is allowed to write."""
    selected: dict[str, Path] = {}
    missing: list[str] = []
    for target_study in target_studies:
        target = target_study.strip().casefold()
        candidate = select_cached_mapping_csv(
            source_study,
            target,
            output_dir,
            cohorts_root=cohorts_root,
        )
        if candidate is None:
            missing.append(target)
        else:
            selected[target] = candidate
    if missing:
        raise FileNotFoundError(
            "No fresh normal-provider mapping CSV for: " + ", ".join(sorted(missing))
        )
    return selected
