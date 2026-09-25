"""Per-variable observation counts of every profiled cohort, read from the EDA
output files in ``{data_folder}/dcr_output_{cohort_id}/``.

Only cohorts that went through EDA (variable profiling) have counts; the
dictionaries' own COUNT / NA columns are declared by the data owners and are
not used here. The result is small (a few numbers per variable) so the
cross-cohort views can load every profiled cohort at once instead of the full
EDA files (several MB each).

Two EDA generations exist:

- v2 (``eda_output_v2_{id}.json``): ``variables[name].n`` is the count of valid
  values; ``completeness`` holds ``n_rows``, ``n_empty``, ``n_coded_missing``.
- v1 (``eda_output_{id}.json``): a flat ``{name: entry}`` dict. Its
  ``count of observations (ex. missing/empty)`` is, despite the label, the
  non-null count for numeric variables but the TOTAL row count for categorical
  ones. The dataset's row count is taken as the most common value of
  count + empty + missing over its variables, and the non-null count as
  min(count, rows - empty - missing), which is right for both readings.

v2 is preferred when both files exist (as in ``detect_eda_status``).
"""

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

from src.config import settings

_LEADING_INT = re.compile(r"^\s*(-?\d+)")

# cohort_id -> (file path, mtime, parsed counts); re-read when the file changes.
_cache: dict[str, tuple[str, float, dict[str, Any]]] = {}


def _leading_int(raw: Any) -> int:
    """The count in "188 (30.23%)", or a bare number; 0 when absent."""
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw) if raw == raw else 0  # NaN check
    m = _LEADING_INT.match(str(raw))
    return int(m.group(1)) if m else 0


def _as_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return int(value)


def eda_file_for(cohort_id: str) -> tuple[Optional[str], Optional[str]]:
    """(path, version) of the cohort's EDA output, v2 first; (None, None) if none."""
    dcr_dir = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}")
    v2 = os.path.join(dcr_dir, f"eda_output_v2_{cohort_id}.json")
    if os.path.exists(v2):
        return v2, "v2"
    v1 = os.path.join(dcr_dir, f"eda_output_{cohort_id}.json")
    if os.path.exists(v1):
        return v1, "v1"
    return None, None


def counts_from_v2(raw: dict[str, Any]) -> dict[str, Any]:
    variables: dict[str, Any] = {}
    for name, entry in (raw.get("variables") or {}).items():
        if not isinstance(entry, dict):
            continue
        comp = entry.get("completeness") if isinstance(entry.get("completeness"), dict) else {}
        n = _as_int(entry.get("n"))
        if n is None:
            n = _as_int(comp.get("n_valid"))
        if n is None:
            continue
        variables[str(name).strip().lower()] = {
            "n": n,
            "rows": _as_int(comp.get("n_rows")),
            "empty": _as_int(comp.get("n_empty")) or 0,
            "missing": _as_int(comp.get("n_coded_missing")) or 0,
        }
    n_rows = _as_int(raw.get("n_rows"))
    if n_rows is None:
        rows = Counter(v["rows"] for v in variables.values() if v["rows"])
        n_rows = rows.most_common(1)[0][0] if rows else None
    for v in variables.values():
        v.pop("rows", None)
    return {"n_rows": n_rows, "variables": variables}


def counts_from_v1(raw: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, tuple[int, int, int]] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        count = _as_int(entry.get("count of observations (ex. missing/empty)"))
        if count is None:
            # Listed in the dictionary but never profiled (no column in the data).
            continue
        parsed[str(name).strip().lower()] = (
            count,
            _leading_int(entry.get("count empty")),
            _leading_int(entry.get("count missing")),
        )
    sums = Counter(c + e + m for c, e, m in parsed.values())
    n_rows = sums.most_common(1)[0][0] if sums else None
    variables: dict[str, Any] = {}
    for name, (count, empty, missing) in parsed.items():
        n = count if n_rows is None else min(count, max(0, n_rows - empty - missing))
        variables[name] = {"n": n, "empty": empty, "missing": missing}
    return {"n_rows": n_rows, "variables": variables}


def read_observation_counts(cohort_id: str) -> Optional[dict[str, Any]]:
    """Counts of one cohort, or None when it has no (readable) EDA output.

    Shape: {eda_version, n_rows, n_listed, generated_at (ISO date of the EDA file),
    variables: {lowercased var name: {n, empty, missing}}}.
    """
    path, version = eda_file_for(cohort_id)
    if not path:
        _cache.pop(cohort_id, None)
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _cache.get(cohort_id)
    if cached and cached[0] == path and cached[1] == mtime:
        return cached[2]
    try:
        with open(path, encoding="utf-8") as f:
            # Python wrote NaN / Infinity, which are not JSON: read them as null.
            raw = json.load(f, parse_constant=lambda _constant: None)
    except Exception as e:
        logging.warning(f"observation counts: cannot read {path}: {e}")
        return None
    if not isinstance(raw, dict):
        return None
    is_v2 = isinstance(raw.get("variables"), dict)
    result = counts_from_v2(raw) if is_v2 else counts_from_v1(raw)
    result["eda_version"] = "v2" if is_v2 else version
    # Entries listed in the EDA file (profiled or not) and the file's date,
    # shown as "profiled on" (the file is rewritten by each EDA run).
    result["n_listed"] = len(raw["variables"]) if is_v2 else sum(1 for e in raw.values() if isinstance(e, dict))
    result["generated_at"] = datetime.fromtimestamp(mtime, tz=timezone.utc).date().isoformat()
    _cache[cohort_id] = (path, mtime, result)
    return result


def all_observation_counts(cohort_ids: list[str]) -> dict[str, Any]:
    """Counts of every profiled cohort among ``cohort_ids``, keyed by cohort id."""
    out: dict[str, Any] = {}
    for cohort_id in cohort_ids:
        counts = read_observation_counts(cohort_id)
        if counts is not None:
            out[cohort_id] = counts
    return out
