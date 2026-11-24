"""Suspect values: numeric variables whose profiling statistics look corrupted
by one or a few values that are probably not real measurements.

The case this was written for: a data owner used 999 as a stand-in for missing
without declaring it in the dictionary, while 9999 WAS declared. The profiling
script excludes declared codes and keeps 999, so it lands in the statistics as
a real observation: for one TIME-CHF variable the mean became 113.85 against a
median of 5.60, the standard deviation 312 against an MAD of 1.05, and the
chart became a single bar at the left edge of a 0-1000 axis.

Nothing here reads patient data. Everything is derived from the aggregate EDA
output the enclave already wrote to {data_folder}/dcr_output_{cohort}/, in
either generation:

- v2 (eda_output_v2_{id}.json): variables[name] with mean, median, q1, q3,
  iqr, mad, trimmed_mean_10, min, max, std, skewness, n_unique and a metadata
  block; completeness.missing_codes_declared holds the declared codes.
- v1 (eda_output_{id}.json): a flat dict per variable with the same statistics
  under spelled-out keys ("std dev", "number of unique values/categories"),
  the dictionary fields suffixed "(metadata dictionary)", and no trimmed mean,
  MAD or percentiles.

Accuracy is deliberately not the goal: the page this feeds is a worklist for
an admin, false positives are expected and cheap, a missed column is not. Each
flagged variable carries the signals that fired, so the reader can dismiss it
in a glance. The fix for a true hit is to add the code to the dictionary's
MISSING column, which the profiling script already honours and which accepts
several codes separated by "|".

Endpoint (admin): GET /api/suspect-values
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from src.auth import get_current_user
from src.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# A variable needs at least this many valid observations before its shape says
# anything. Below it the quartiles are meaningless and everything looks extreme.
MIN_OBSERVATIONS = 8

# Default cut-off for the worklist. Every signal below contributes points; a
# single weak signal scores 1 or 2, so 4 means "at least one strong signal, or
# several weak ones agreeing".
DEFAULT_MIN_SCORE = 4

_LEADING_NUMBER = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _num(raw: Any) -> Optional[float]:
    """A float out of a profiling value, which may be a number, None, or a v1
    string like "2 (20.00%)" or "non-normal"."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        f = float(raw)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    m = _LEADING_NUMBER.match(str(raw))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _digits(value: Optional[float]) -> Optional[str]:
    """Digit string of a value, sign and decimal point dropped: 999.0 and 99.9
    both give "999". Used only to recognise sentinel shapes."""
    if value is None:
        return None
    text = f"{abs(float(value)):.10f}".rstrip("0").rstrip(".")
    return text.replace(".", "") or None


def _is_repdigit(digits: Optional[str], digit: str, min_len: int) -> bool:
    return bool(digits) and len(digits) >= min_len and set(digits) == {digit}


def _sentinel_shape(value: Optional[float]) -> Optional[str]:
    """A short description when the literal looks like a stand-in for missing,
    else None. Nines are the common convention; sevens and eights appear in
    survey-derived data and are only read as sentinels when whole."""
    if value is None:
        return None
    digits = _digits(value)
    if _is_repdigit(digits, "9", 2):
        return f"{'a negative ' if value < 0 else 'an '}all-nines literal"
    whole = float(value) == int(value)
    if whole and (_is_repdigit(digits, "8", 3) or _is_repdigit(digits, "7", 3)):
        return "a repeated-digit literal"
    if whole and value in (-1.0, -9.0, -99.0, -999.0, -9999.0):
        return "a negative sentinel literal"
    return None


def _same_code_family(declared: list[str], value: Optional[float]) -> Optional[str]:
    """The declared missing code this value looks like a sibling of: same
    repeated digit, different width. 9999 declared makes 999 a sibling."""
    vd = _digits(value)
    if not vd or len(set(vd)) != 1:
        return None
    for code in declared:
        cd = _digits(_num(code))
        if cd and len(set(cd)) == 1 and cd[0] == vd[0] and cd != vd:
            return str(code)
    return None


# ---------------------------------------------------------------------------
# Normalising the two profiling generations into one record
# ---------------------------------------------------------------------------

# Placeholders that appear in the dictionary cell where no code is declared.
_NOT_A_CODE = {"", "none", "nan", "na", "n/a", "null", "-", "nil"}


def _declared_codes(raw: Any) -> list[str]:
    """Declared missing codes as strings, from a list or a "999 |9999" cell.
    Placeholder spellings of "nothing declared" are dropped."""
    if raw is None:
        return []
    parts = [str(c) for c in raw] if isinstance(raw, list) else str(raw).split("|")
    return [p.strip() for p in parts if p.strip().lower() not in _NOT_A_CODE]


def _from_v2(name: str, entry: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(entry, dict) or entry.get("median") is None or entry.get("iqr") is None:
        return None  # not a profiled numeric variable
    meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    comp = entry.get("completeness") if isinstance(entry.get("completeness"), dict) else {}
    return {
        "variable": name,
        "label": meta.get("label"),
        "units": meta.get("units"),
        "concept_name": meta.get("concept_name"),
        "domain": meta.get("domain"),
        "var_type": meta.get("var_type"),
        # Passed to the chart modal, which uses them to recognise a patient-id
        # variable. The stored graph_url is not: it carries a hardcoded
        # production host, so the page builds the chart path itself.
        "omop_id": meta.get("omop_id"),
        "concept_code": meta.get("concept_code"),
        "visits": meta.get("visits"),
        "n": _num(entry.get("n")),
        "n_unique": _num(entry.get("n_unique")),
        "zero_fraction": _num(entry.get("zero_fraction")),
        "mean": _num(entry.get("mean")),
        "median": _num(entry.get("median")),
        "q1": _num(entry.get("q1")),
        "q3": _num(entry.get("q3")),
        "iqr": _num(entry.get("iqr")),
        "minimum": _num(entry.get("min")),
        "maximum": _num(entry.get("max")),
        "std": _num(entry.get("std")),
        "mad": _num(entry.get("mad")),
        "trimmed_mean": _num(entry.get("trimmed_mean_10")),
        "skewness": _num(entry.get("skewness")),
        "declared_missing": _declared_codes(comp.get("missing_codes_declared")),
        "declared_max": None,
        "declared_min": None,
    }


def _from_v1(name: str, entry: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(entry, dict) or entry.get("median") is None or entry.get("iqr") is None:
        return None
    return {
        "variable": name,
        "label": entry.get("variablelabel (metadata dictionary)") or entry.get("label"),
        "units": entry.get("units (metadata dictionary)"),
        "concept_name": entry.get("variable concept name (metadata dictionary)"),
        "domain": entry.get("domain (metadata dictionary)"),
        "var_type": entry.get("vartype (metadata dictionary)"),
        "omop_id": entry.get("variable omop id (metadata dictionary)"),
        "concept_code": entry.get("variable concept code (metadata dictionary)"),
        "visits": entry.get("visits (metadata dictionary)"),
        "n": _num(entry.get("count of observations (ex. missing/empty)")),
        "n_unique": _num(entry.get("number of unique values/categories")),
        "zero_fraction": None,  # v1 does not record it; the median rule stands in
        "mean": _num(entry.get("mean")),
        "median": _num(entry.get("median")),
        "q1": _num(entry.get("q1")),
        "q3": _num(entry.get("q3")),
        "iqr": _num(entry.get("iqr")),
        "minimum": _num(entry.get("min")),
        "maximum": _num(entry.get("max")),
        "std": _num(entry.get("std dev")),
        "mad": None,          # v1 carries no MAD
        "trimmed_mean": None,  # nor a trimmed mean
        "skewness": _num(entry.get("skewness")),
        "declared_missing": _declared_codes(
            entry.get("missing (metadata dictionary)") or entry.get("code for missing value")
        ),
        "declared_max": _num(entry.get("max (metadata dictionary)")),
        "declared_min": _num(entry.get("min (metadata dictionary)")),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _observations_clustered_at(median: Optional[float], low: Optional[float],
                               high: Optional[float], zero_fraction: Optional[float]) -> Optional[str]:
    """Where most of a column's observations sit, when they pile up on one value
    at the edge of its range: "the maximum (730)", "the minimum (0)", "zero".
    None for a column whose median sits inside its range.

    The median equal to the minimum or maximum means at least half of the
    observations are that one value. A quarter of zeros is enough on its own,
    since it already pulls Q1 onto zero.
    """
    if median is not None and high is not None and median == high:
        return f"the maximum ({_fmt(high)})"
    if median is not None and low is not None and median == low:
        return f"the minimum ({_fmt(low)})"
    if median == 0 or (zero_fraction is not None and zero_fraction >= 0.25):
        return "zero"
    return None


def _tiered(value: Optional[float], tiers: list[tuple[float, int]]) -> int:
    """Points for a ratio, from the highest tier it clears."""
    if value is None:
        return 0
    points = 0
    for threshold, pts in tiers:
        if value >= threshold:
            points = pts
    return points


def score_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Points, reasons and derived ratios for one numeric variable.

    The signals are deliberately redundant: each one alone has false positives,
    and a value that is really a stand-in for missing trips most of them at once.
    """
    reasons: list[str] = []
    score = 0
    metrics: dict[str, Any] = {}

    median, q1, q3 = rec["median"], rec["q1"], rec["q3"]
    low, high = rec["minimum"], rec["maximum"]
    # Cluster of observations at the minimum or maximum. When most values sit
    # on one bound the quartiles collapse onto it and every distance-in-IQRs
    # test fires on ordinary values: when most patients take no nitrate the
    # median is 0 and the largest real dose looks hundreds of spreads away;
    # when most answer the time trade-off at its 730-day cap, a real answer of
    # 1 day does. Such a column is only judged by the shape of its extreme
    # literals and by the sign of its minimum.
    zero_fraction = _num(rec.get("zero_fraction"))
    clustered_at = _observations_clustered_at(median, low, high, zero_fraction)
    degenerate = clustered_at is not None
    # Robust width of the body of the distribution. The IQR is in both
    # generations; the MAD stands in when the quartiles coincide, which happens
    # in sparse columns where most values are equal.
    spread = rec["iqr"] if (rec["iqr"] or 0) > 0 else (
        (rec["mad"] or 0) * 1.4826 if (rec["mad"] or 0) > 0 else None)
    metrics["spread"] = spread
    metrics["observations_clustered_at"] = clustered_at

    # (1) How far the extremes sit beyond the body, in spread units. A genuine
    # extreme is a few units out; a sentinel is hundreds.
    gap_high = (high - q3) / spread if spread and high is not None and q3 is not None else None
    gap_low = (q1 - low) / spread if spread and low is not None and q1 is not None else None
    metrics["gap_high"] = gap_high
    metrics["gap_low"] = gap_low
    gap = max([g for g in (gap_high, gap_low) if g is not None], default=None)
    gap_points = 0 if degenerate else _tiered(gap, [(10, 2), (50, 3), (200, 4)])
    if gap_points:
        side = "above Q3" if (gap_high or 0) >= (gap_low or 0) else "below Q1"
        reasons.append(f"extreme sits {gap:.0f} interquartile ranges {side}")
        score += gap_points

    # (2) Mean against the trimmed mean, the sharpest signal when it is there:
    # trimming 10% from each tail removes the sentinel and the mean collapses.
    # The distance is scored in interquartile ranges, not as a share of the
    # trimmed mean: a lab value with a naturally long right tail, such as
    # troponin or CRP, has a mean 30% to 150% above its trimmed mean with no
    # bad value in it, but that gap is only about one IQR. A stand-in for
    # missing puts it tens of IQRs away. The share is still printed, since it
    # is the easier number to read.
    if not degenerate and spread and rec["trimmed_mean"] is not None and rec["mean"] is not None:
        diff = rec["mean"] - rec["trimmed_mean"]
        in_iqrs = abs(diff) / spread
        share = abs(diff) / max(abs(rec["trimmed_mean"]), 1e-9)
        metrics["mean_vs_trimmed"] = share
        metrics["mean_vs_trimmed_iqrs"] = in_iqrs
        pts = _tiered(in_iqrs, [(3, 1), (10, 2), (40, 3)])
        if pts:
            if diff > 0 and share >= 1:
                phrase = f"mean is {share + 1:.1f}x the trimmed mean"
            else:
                phrase = f"mean is {share:.0%} {'above' if diff > 0 else 'below'} the trimmed mean"
            reasons.append(f"{phrase}, {in_iqrs:.0f} interquartile ranges away")
            score += pts

    # (3) Mean against the median, the same idea with what v1 also carries.
    if not degenerate and spread and rec["mean"] is not None and median is not None:
        shift = abs(rec["mean"] - median) / spread
        metrics["mean_vs_median"] = shift
        pts = _tiered(shift, [(3, 1), (10, 2), (40, 3)])
        if pts:
            reasons.append(f"mean sits {shift:.0f} interquartile ranges from the median")
            score += pts

    # (4) Standard deviation against the robust spread: one far value inflates
    # the first and leaves the second untouched.
    if not degenerate and spread and rec["std"]:
        ratio = rec["std"] / spread
        metrics["std_vs_spread"] = ratio
        pts = _tiered(ratio, [(10, 1), (50, 2)])
        if pts:
            reasons.append(f"standard deviation is {ratio:.0f}x the robust spread")
            score += pts

    # (5) and (6) The literal itself. Checked on whichever extreme is further
    # out, so a column with a sentinel at both ends still reports one row.
    candidates = [v for v in (high, low) if v is not None]
    if (gap_low or 0) > (gap_high or 0):
        candidates.reverse()
    suspect = None
    likely_cause = "extreme outlier"
    for value in candidates:
        shape = _sentinel_shape(value)
        sibling = _same_code_family(rec["declared_missing"], value)
        if shape or sibling:
            suspect = value
            likely_cause = "undeclared missing code"
            if shape:
                reasons.append(f"{_fmt(value)} is {shape}")
                score += 3
            if sibling:
                reasons.append(f"{_fmt(value)} is a sibling of the declared code {sibling}")
                score += 4
            break
    # (7) A negative minimum in a column whose maximum is positive. Most
    # measurements cannot go below zero, so a negative value among positive ones
    # is either a code (-1 for "not applicable") or an entry error, such as a
    # visit dated 272 days before inclusion. Some columns go negative for real
    # (an electrical axis, days to an event recorded before baseline), which is
    # why this weighs 2 and needs another signal to reach the default cut-off.
    negative_min = low is not None and high is not None and low < 0 < high
    if negative_min:
        reasons.append(f"minimum is negative ({_fmt(low)}) while the maximum is positive")
        score += 2

    if suspect is None and candidates:
        # With no literal to point at, the suspect is the far extreme; when no
        # distance signal fired and only the sign did, it is the minimum.
        if negative_min and not gap_points:
            suspect = low
        else:
            suspect = candidates[0]
        if negative_min and suspect == low:
            likely_cause = "negative minimum"

    if clustered_at is not None and score:
        # Said only on a row that is flagged anyway, so the reader knows why the
        # distance signals are silent on it.
        reasons.append(
            f"most observations sit at {clustered_at}, so distances from the quartiles are not scored")

    # (8) The dictionary's own declared range, where v1 recorded it. The
    # comparison needs a margin: those cells are stored at single precision, so
    # a declared maximum of 358.0899963 sits just under an observed 358.09.
    if _exceeds(high, rec["declared_max"]):
        reasons.append(f"maximum exceeds the declared maximum {_fmt(rec['declared_max'])}")
        score += 1
    if _exceeds(rec["declared_min"], low):
        reasons.append(f"minimum is below the declared minimum {_fmt(rec['declared_min'])}")
        score += 1

    rec = dict(rec)
    rec.update({
        "score": score,
        "reasons": reasons,
        "metrics": metrics,
        "suspect_value": suspect,
        "likely_cause": likely_cause,
    })
    return rec


# Relative margin for the declared-range comparison, wide enough to absorb the
# single-precision dictionary cells and any hand-rounded bound.
_RANGE_TOLERANCE = 0.01


def _exceeds(value: Optional[float], bound: Optional[float]) -> bool:
    """True when `value` is meaningfully larger than `bound`."""
    if value is None or bound is None:
        return False
    return (value - bound) > max(abs(bound), abs(value), 1.0) * _RANGE_TOLERANCE


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "?"
    return str(int(value)) if float(value) == int(value) else f"{value:g}"


def _apply_cohort_convention(rows: list[dict[str, Any]]) -> None:
    """Second pass over one cohort: a literal that is the suspect value of
    several unrelated variables is a house convention, not a measurement."""
    counts: dict[str, int] = {}
    for row in rows:
        if row["likely_cause"] == "undeclared missing code" and row["suspect_value"] is not None:
            key = _fmt(row["suspect_value"])
            counts[key] = counts.get(key, 0) + 1
    for row in rows:
        if row["suspect_value"] is None:
            continue
        key = _fmt(row["suspect_value"])
        if counts.get(key, 0) >= 3:
            row["reasons"].append(f"{key} is the suspect value in {counts[key]} variables of this cohort")
            row["score"] += 2


# ---------------------------------------------------------------------------
# Reading the profiling output, with a cache keyed on the files themselves
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"signature": None, "rows": [], "scanned": (0, 0), "built_at": None}


def _profiling_files() -> list[tuple[str, str, str]]:
    """(cohort_id, path, version) for every cohort with profiling output."""
    out: list[tuple[str, str, str]] = []
    prefix = "dcr_output_"
    try:
        entries = sorted(os.listdir(settings.data_folder))
    except OSError as e:
        logger.warning("suspect values: cannot list the data folder: %s", e)
        return out
    for name in entries:
        if not name.startswith(prefix):
            continue
        cohort_id = name[len(prefix):]
        folder = os.path.join(settings.data_folder, name)
        v2 = os.path.join(folder, f"eda_output_v2_{cohort_id}.json")
        v1 = os.path.join(folder, f"eda_output_{cohort_id}.json")
        if os.path.exists(v2):
            out.append((cohort_id, v2, "v2"))
        elif os.path.exists(v1):
            out.append((cohort_id, v1, "v1"))
    return out


def _signature(files: list[tuple[str, str, str]]) -> str:
    parts = []
    for cohort_id, path, _version in files:
        try:
            st = os.stat(path)
            parts.append(f"{cohort_id}:{int(st.st_mtime)}:{st.st_size}")
        except OSError:
            parts.append(f"{cohort_id}:missing")
    return "|".join(parts)


def _measure_key(rec: dict[str, Any]) -> str:
    """What makes two variables of one cohort the same measure taken at different
    visits: the variable concept and the units. hscrp_bl and hscrp_v18 share
    both, so the page shows them as one measure with its visits listed.

    A variable with no concept stays on its own, keyed by its name, rather than
    being merged with others on a guess from the label.
    """
    concept = rec.get("omop_id") or rec.get("concept_code")
    concept = str(concept).strip().lower() if concept not in (None, "") else ""
    if concept in _NOT_A_CODE:
        return f"variable:{rec['variable']}"
    units = str(rec.get("units") or "").strip().lower()
    return f"concept:{concept}|units:{units}"


def _scan_cohort(cohort_id: str, path: str, version: str) -> tuple[list[dict[str, Any]], int]:
    """Scored rows for one cohort, and how many numeric variables were read."""
    try:
        with open(path, encoding="utf-8") as f:
            # The enclave writes NaN and Infinity, which are not valid JSON.
            raw = json.load(f, parse_constant=lambda _c: None)
    except Exception as e:
        logger.warning("suspect values: cannot read %s: %s", path, e)
        return [], 0
    if not isinstance(raw, dict):
        return [], 0
    entries = raw.get("variables") if version == "v2" else raw
    if not isinstance(entries, dict):
        return [], 0

    rows: list[dict[str, Any]] = []
    for name, entry in entries.items():
        rec = _from_v2(name, entry) if version == "v2" else _from_v1(name, entry)
        if rec is None:
            continue
        if (rec["n"] or 0) < MIN_OBSERVATIONS:
            continue
        rec["cohort"] = cohort_id
        rec["eda_version"] = version
        rec["measure"] = _measure_key(rec)
        rows.append(score_record(rec))
    _split_shared_visits(rows)
    _apply_cohort_convention(rows)
    return rows, len(rows)


def _split_shared_visits(rows: list[dict[str, Any]]) -> None:
    """Keep a variable out of its concept's group when another variable of the
    same concept and units is recorded at the same visit.

    The same measure at different visits has one variable per visit. Two at one
    visit means the concept is broader than the measure: in TIME-CHF the days to
    a first heart failure event and the days to a first event of any kind carry
    the same concept, both at baseline. Those stay as separate rows, and the
    variables of that concept with a visit of their own still group.
    """
    def visit_of(row: dict[str, Any]) -> str:
        return str(row.get("visits") or "").strip().lower()

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["measure"], visit_of(row))
        counts[key] = counts.get(key, 0) + 1
    for row in rows:
        if counts[(row["measure"], visit_of(row))] > 1:
            row["measure"] = f"variable:{row['variable']}"


def _all_rows() -> dict[str, Any]:
    """Every scored numeric variable, rebuilt only when a profiling file changed."""
    files = _profiling_files()
    signature = _signature(files)
    with _cache_lock:
        if _cache["signature"] == signature:
            return {"rows": _cache["rows"], "scanned": _cache["scanned"], "built_at": _cache["built_at"]}
    rows: list[dict[str, Any]] = []
    variables = 0
    for cohort_id, path, version in files:
        cohort_rows, n = _scan_cohort(cohort_id, path, version)
        rows.extend(cohort_rows)
        variables += n
    built_at = datetime.now().isoformat()
    with _cache_lock:
        _cache.update({"signature": signature, "rows": rows,
                       "scanned": (len(files), variables), "built_at": built_at})
    logger.info("suspect values: scored %d numeric variables across %d cohorts", variables, len(files))
    return {"rows": rows, "scanned": (len(files), variables), "built_at": built_at}


def _require_admin(user: dict[str, Any]) -> None:
    if user.get("email") not in settings.admins_list:
        raise HTTPException(status_code=403, detail="Admins only.")


@router.get("/suspect-values")
def get_suspect_values(
    min_score: int = DEFAULT_MIN_SCORE,
    cohort: Optional[str] = None,
    limit: int = 1000,
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Numeric variables whose profiling statistics suggest one or a few values
    that are not real measurements, ranked by how much evidence there is.

    Read from the EDA output already on disk, so it costs no enclave run. False
    positives are expected: every row carries the signals that fired.
    """
    _require_admin(user)
    data = _all_rows()
    cohorts_scanned, variables_scanned = data["scanned"]
    rows = data["rows"]
    if cohort:
        wanted = cohort.strip().lower()
        rows = [r for r in rows if r["cohort"].lower() == wanted]
    flagged = [r for r in rows if r["score"] >= min_score]
    # Sentinel-shaped cases first: they are the actionable ones, since the fix
    # is a line in the dictionary. Pure shape outliers follow, ranked by score.
    flagged.sort(key=lambda r: (r["likely_cause"] != "undeclared missing code", -r["score"],
                                -(r["metrics"].get("gap_high") or 0), r["cohort"], r["variable"]))
    return {
        "generated_at": data["built_at"],
        "min_score": min_score,
        "cohorts_scanned": cohorts_scanned,
        "variables_scanned": variables_scanned,
        "n_flagged": len(flagged),
        "cohorts": sorted({r["cohort"] for r in rows}),
        "by_cohort": {c: sum(1 for r in flagged if r["cohort"] == c)
                      for c in sorted({r["cohort"] for r in flagged})},
        "items": flagged[:max(1, limit)],
    }
