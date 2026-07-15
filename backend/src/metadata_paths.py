import re
from pathlib import Path

STUDIES_METADATA_GRAPH_URI = "https://w3id.org/CMEO/graph/studies_metadata"
_COHORT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_cohort_id(cohort_id: str) -> str:
    canonical = cohort_id.strip()
    if not _COHORT_ID.fullmatch(canonical):
        raise ValueError(f"Invalid cohort ID: {cohort_id!r}")
    return canonical


def canonical_dictionary_path(cohorts_root: str | Path, cohort_id: str) -> Path:
    canonical_id = validate_cohort_id(cohort_id)
    return Path(cohorts_root) / canonical_id / f"{canonical_id}_datadictionary.csv"
