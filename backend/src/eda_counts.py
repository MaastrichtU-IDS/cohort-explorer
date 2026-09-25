"""Minimal per-variable non-null observation counts for the concept coverage
heatmap, read from the EDA output files written by the DCR enclave scripts
(c3 in eda_scripts.py) into ``{data_folder}/dcr_output_{cohort_id}/``.

Two EDA generations exist, with different shapes:

- v2 (``eda_output_v2_{id}.json``, preferred): ``variables[name].n`` is the
  count of valid observations (blank, coded-missing and invalid cells all
  excluded); ``variables[name].completeness.n_valid`` holds the same number
  and the dataset row total is the top-level ``n_rows``.
- v1 (``eda_output_{id}.json``, legacy flat dict): the
  ``count of observations (ex. missing/empty)`` key is, despite its label,
  the non-null count for numeric variables but the TOTAL row count for
  categorical ones. The row total is taken as the most common value of
  count + empty + missing over the variables, and the non-null count as
  min(count, rows - empty - missing), which is right for both readings.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Any, Optional

from fastapi import APIRouter, Depends

from src.auth import get_current_user
from src.config import settings

router = APIRouter()

_LEADING_INT = re.compile(r"^\s*(-?\d+)")


def _as_int(raw: Any) -> Optional[int]:
    """Leading integer of a value like 42, 42.0 or "188 (30.23%)"; None when absent."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw) if raw == raw else None  # NaN check
    m = _LEADING_INT.match(str(raw))
    return int(m.group(1)) if m else None


def _counts_from_v2(raw: dict[str, Any]) -> dict[str, Any]:
    variables: dict[str, int] = {}
    for name, entry in (raw.get("variables") or {}).items():
        if not isinstance(entry, dict):
            continue
        comp = entry.get("completeness") if isinstance(entry.get("completeness"), dict) else {}
        n = _as_int(entry.get("n"))
        if n is None:
            n = _as_int(comp.get("n_valid"))
        if n is not None:
            variables[str(name).strip().lower()] = n
    return {"eda_version": "v2", "n_rows": _as_int(raw.get("n_rows")), "variables": variables}


def _counts_from_v1(raw: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, tuple[int, int, int]] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        count = _as_int(entry.get("count of observations (ex. missing/empty)"))
        if count is None:
            continue
        parsed[str(name).strip().lower()] = (
            count,
            _as_int(entry.get("count empty")) or 0,
            _as_int(entry.get("count missing")) or 0,
        )
    sums = Counter(c + e + m for c, e, m in parsed.values())
    n_rows = sums.most_common(1)[0][0] if sums else None
    variables = {
        name: (count if n_rows is None else min(count, max(0, n_rows - empty - missing)))
        for name, (count, empty, missing) in parsed.items()
    }
    return {"eda_version": "v1", "n_rows": n_rows, "variables": variables}


def _read_cohort_counts(cohort_id: str) -> Optional[dict[str, Any]]:
    dcr_dir = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}")
    v2 = os.path.join(dcr_dir, f"eda_output_v2_{cohort_id}.json")
    v1 = os.path.join(dcr_dir, f"eda_output_{cohort_id}.json")
    path = v2 if os.path.exists(v2) else v1 if os.path.exists(v1) else None
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            # The enclave scripts write NaN / Infinity, which are not JSON: read them as null.
            raw = json.load(f, parse_constant=lambda _constant: None)
    except Exception as e:
        logging.warning(f"eda counts: cannot read {path}: {e}")
        return None
    if not isinstance(raw, dict):
        return None
    return _counts_from_v2(raw) if isinstance(raw.get("variables"), dict) else _counts_from_v1(raw)


@router.get("/eda-observation-counts")
def get_eda_observation_counts(user: Any = Depends(get_current_user)) -> dict:
    """Non-null observation counts of every cohort with EDA output.

    {cohort_id: {eda_version, n_rows, variables: {lowercased var name: n}}},
    where n is the count of valid (non-empty, non-coded-missing) values.
    Cohorts without EDA output are absent.
    """
    out: dict[str, Any] = {}
    prefix = "dcr_output_"
    try:
        entries = sorted(os.listdir(settings.data_folder))
    except OSError as e:
        logging.warning(f"eda counts: cannot list data folder: {e}")
        return out
    for name in entries:
        if not name.startswith(prefix):
            continue
        cohort_id = name[len(prefix):]
        counts = _read_cohort_counts(cohort_id)
        if counts is not None:
            out[cohort_id] = counts
    return out
