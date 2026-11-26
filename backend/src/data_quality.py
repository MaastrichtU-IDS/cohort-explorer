"""All data issues of the catalogue in one list, for the /data-quality page.

Three sources, each turned into the same issue record:

- Dictionary syntax: the validation every dictionary upload goes through, run
  over each cohort's current dictionary (collect_metadata_syntax_issues). It
  is cheap, so it runs live, cached until a dictionary file changes. Its
  messages are plain strings; _classify_syntax sorts them into types from the
  wording, which is all written in upload.py.
- Concept codes: the OMOP/Athena check of the variable, category, unit, visit
  and additional-context concepts. It queries the Athena API for every
  concept, so it is never run by a page load: an admin starts it (for one
  cohort or all), it runs in the background, and its findings are kept in
  {data_folder}/data_quality/concept_check.json with the dictionary they were
  computed from. A cohort whose dictionary changed since is marked stale.
- Data values: the suspect values worklist, read from the variable profiling.

Endpoints (admin):
  GET  /api/data-quality/issues
  POST /api/data-quality/concept-check            ?cohort= for one cohort
  GET  /api/data-quality/concept-check/status
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.auth import get_current_user
from src.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue types
# ---------------------------------------------------------------------------

CATEGORIES: list[dict[str, str]] = [
    {"key": "syntax", "label": "Dictionary syntax",
     "description": "Problems in how the data dictionary is written. Most block a dictionary upload; the declared range and missing-code checks (S6, S7 and invalid MIN or MAX) are found only here."},
    {"key": "concepts", "label": "Concept codes",
     "description": "Concept codes that do not match OMOP: wrong code, wrong name for the code, or a code that does not exist."},
    {"key": "values", "label": "Data values",
     "description": "Values in the profiled data that are probably not real measurements."},
]

# key → (category, label, description, specification id). The order is the
# order on the page. The specification ids (S1 to S14) are those of the data
# quality specification; a type outside it has none.
ISSUE_TYPES: dict[str, tuple[str, str, str, Optional[str]]] = {
    "syntax.missing_columns": ("syntax", "Missing required columns", "Columns every dictionary must have are absent.", None),
    "syntax.invalid_vartype": ("syntax", "Invalid VARTYPE", "VARTYPE is not one of STR, INT, FLOAT, DATETIME.", "S1"),
    "syntax.invalid_domain": ("syntax", "Invalid DOMAIN", "DOMAIN is not one of the OMOP tables it may name.", "S1"),
    "syntax.concept_code_format": ("syntax", "Malformed concept code", "A concept code lacks its prefix or colon, uses an unknown prefix, or several codes share one cell.", "S1"),
    "syntax.omop_id_format": ("syntax", "Malformed OMOP id", "An OMOP id is not an integer, or several share one cell.", "S1"),
    "syntax.missing_code_format": ("syntax", "Malformed MISSING code", "A code in the MISSING column contains a space.", "S1"),
    "syntax.invalid_range_bound": ("syntax", "Invalid MIN or MAX", "MIN or MAX of a numeric or date variable is not a number or a date.", "S1"),
    "syntax.missing_required_field": ("syntax", "Required field missing", "The variable name, label or VARTYPE of a row is blank.", "S2"),
    "syntax.duplicate_variable": ("syntax", "Duplicate variable name", "The same VARIABLENAME appears on several rows.", "S3"),
    "syntax.category_order": ("syntax", "Category value and label swapped", "Value and label appear reversed for the declared VARTYPE.", "S4"),
    "syntax.category_code_type": ("syntax", "Category code contradicts VARTYPE", "A category code is not of the declared VARTYPE, such as a non-integer code on an INT variable.", "S4"),
    "syntax.category_format": ("syntax", "Malformed categories", "Categories without '=', without the '|' separator, or not parseable.", "S5"),
    "syntax.repeated_category_code": ("syntax", "Repeated category code", "One code is declared twice in the same categorical cell.", "S5"),
    "syntax.empty_category_label": ("syntax", "Empty category label", "A category has a code but no label.", "S5"),
    "syntax.impossible_range": ("syntax", "Impossible declared range", "MIN is above MAX, or the range goes below zero for a quantity that cannot, such as a concentration or a length.", "S6"),
    "syntax.missing_code_collision": ("syntax", "Missing code collides", "A declared missing code is also a category code, or lies inside MIN to MAX, so it cannot be told apart from a real value.", "S7"),
    "syntax.orphan_concept": ("syntax", "Orphan concept annotation", "Category, unit, visit or additional-context concepts on a row without the categories, unit, visit or variable concept they describe.", "S8"),
    "syntax.concept_count_mismatch": ("syntax", "Concept count mismatch", "Concept names, codes, OMOP ids and categories differ in number.", "S9"),
    "syntax.other": ("syntax", "Other syntax issue", "A dictionary problem of another kind.", None),
    "concepts.code_id_mismatch": ("concepts", "Concept code and ID disagree", "The concept code and the OMOP id refer to different concepts, or the id does not exist.", "S10"),
    "concepts.wrong_vocabulary": ("concepts", "Wrong vocabulary", "A concept from a vocabulary not accepted for its field, such as a unit that is not UCUM.", "S11"),
    "concepts.wrong_domain": ("concepts", "Wrong domain", "A concept whose OMOP domain does not fit its field or the row's DOMAIN.", "S11"),
    "concepts.deprecated": ("concepts", "Deprecated concept", "Athena marks the concept deprecated or replaced.", "S12"),
    "concepts.non_standard": ("concepts", "Non-standard concept", "Athena marks the concept non-standard or a classification concept.", "S12"),
    "concepts.name_mismatch": ("concepts", "Concept name mismatch", "The concept name differs from the vocabulary's name and synonyms.", "S13"),
    "concepts.vartype_implausible": ("concepts", "VARTYPE implausible for concept", "VARTYPE conflicts with the LOINC scale of the variable concept.", "S14"),
    "concepts.unit_implausible": ("concepts", "Unit implausible for concept", "The unit's dimension conflicts with the LOINC property of the variable concept.", "S14"),
    "concepts.structure": ("concepts", "Dictionary structure", "The concept check could not read the dictionary's columns.", None),
    "concepts.other": ("concepts", "Other concept issue", "A concept problem of another kind.", None),
    "concepts.check_failed": ("concepts", "Check incomplete", "The concept check stopped with an error, or Athena could not be reached for some concepts.", None),
    "values.undeclared_missing_code": ("values", "Undeclared missing code", "A value shaped like a missing code (999, -1) that the dictionary does not declare.", None),
    "values.negative_minimum": ("values", "Negative minimum", "A negative value in a column that is otherwise positive.", None),
    "values.extreme_outlier": ("values", "Extreme outlier", "A value far beyond the rest of its column.", None),
}


# Dictionary problems found only here, not by the upload validation, so they do
# not block an upload.
NON_BLOCKING_TYPES = {"syntax.invalid_range_bound", "syntax.impossible_range", "syntax.missing_code_collision"}


# ---------------------------------------------------------------------------
# Dictionary syntax
# ---------------------------------------------------------------------------

def _syntax_rules() -> list[tuple[str, re.Pattern[str]]]:
    # Imported here: upload.py is large and imports much of the backend.
    from src.upload import _TAG_EMPTY_LABEL, _TAG_REPEATED_CODE
    rules = [
        ("syntax.repeated_category_code", re.escape(_TAG_REPEATED_CODE)),
        ("syntax.empty_category_label", re.escape(_TAG_EMPTY_LABEL)),
        ("syntax.missing_columns", r"^Missing required columns"),
        ("syntax.duplicate_variable", r"^Duplicate VARIABLENAME"),
        ("syntax.missing_required_field", r"is missing value for the required field"),
        ("syntax.invalid_vartype", r"has an invalid data type"),
        ("syntax.category_order", r"in the wrong order for the declared vartype"),
        ("syntax.category_code_type", r"have a code that contradicts the declared vartype"),
        ("syntax.missing_code_format", r"in the MISSING column contains a space"),
        ("syntax.category_format", r"without the '\|' separator|is missing '='|has an invalid category"),
        ("syntax.concept_count_mismatch", r"The number of "),
        ("syntax.orphan_concept", r"fields are provided|CONCEPT NAME is provided"),
        ("syntax.omop_id_format", r"OMOP IDs? |must be an integer"),
        ("syntax.concept_code_format", r"concept code|CURIE"),
    ]
    return [(key, re.compile(pattern)) for key, pattern in rules]


_ROW_PREFIX = re.compile(r"^Row (\d+) \(Variable: '([^']*)'(?:, Category: '([^']*)')?\):?\s*")


def _classify_syntax(message: str, rules: list[tuple[str, re.Pattern[str]]]) -> dict[str, Any]:
    """Type, variable, dictionary row and the message without its row prefix."""
    issue_type = next((key for key, pattern in rules if pattern.search(message)), "syntax.other")
    variable = row = None
    text = message
    m = _ROW_PREFIX.match(message)
    if m:
        row = int(m.group(1))
        variable = m.group(2) or None
        text = message[m.end():]
        if m.group(3):
            text = f"Category '{m.group(3)}': {text}"
        elif text[:1].islower():
            text = f"This row {text}"
    # The tag the syntax report marks these two classes with is the type badge here.
    text = re.sub(r"^\[(?:repeated category code|empty category label)\]\s*", "", text)
    return {"type": issue_type, "variable": variable, "row": row, "message": text}


_syntax_lock = threading.Lock()
_syntax_cache: dict[str, Any] = {"signature": None, "data": None}


def _dictionary_signature() -> str:
    """Changes whenever any cohort's current dictionary does."""
    from src.upload import get_latest_datadictionary
    parts = []
    try:
        names = sorted(os.listdir(settings.cohort_folder))
    except OSError:
        return ""
    for name in names:
        folder = os.path.join(settings.cohort_folder, name)
        if not os.path.isdir(folder):
            continue
        path = get_latest_datadictionary(folder)
        if path:
            st = os.stat(path)
            parts.append(f"{name}:{os.path.basename(path)}:{st.st_mtime_ns}:{st.st_size}")
    return "|".join(parts)


def _syntax_report() -> dict[str, Any]:
    from src.upload import collect_metadata_syntax_issues
    signature = _dictionary_signature()
    with _syntax_lock:
        if _syntax_cache["signature"] != signature or _syntax_cache["data"] is None:
            _syntax_cache["data"] = collect_metadata_syntax_issues()
            _syntax_cache["signature"] = signature
        return _syntax_cache["data"]


def _syntax_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    rules = _syntax_rules()
    issues: list[dict[str, Any]] = []
    for c in report.get("cohorts", []):
        cohort = c["cohort_id"]
        # A cohort without a dictionary has nothing to be wrong yet, and a file
        # that cannot be parsed is turned away at upload: neither is listed.
        if c["status"] in ("no_dictionary", "no_folder", "failed"):
            continue
        for message in c.get("errors") or []:
            issues.append({"cohort": cohort, **_classify_syntax(message, rules)})
    return issues


# ---------------------------------------------------------------------------
# Declared range and missing codes (S1, S6, S7), which the upload does not check
# ---------------------------------------------------------------------------

def _number(text: str) -> Optional[float]:
    try:
        value = float(str(text).strip().strip("'\"").replace(",", "."))
    except ValueError:
        return None
    return value if value == value and abs(value) != float("inf") else None


def _date(text: str) -> Optional[datetime]:
    """A date or a bare year. Python's datetime rather than pandas', which
    stops at 2262 and would call a year of 9999 no date at all."""
    from dateutil import parser as date_parser
    text = str(text).strip()
    if re.fullmatch(r"\d{4}", text):
        return datetime(int(text), 1, 1) if int(text) >= 1 else None
    try:
        return date_parser.parse(text, default=datetime(2000, 1, 1))
    except (ValueError, OverflowError, TypeError):
        return None


def _fmt_number(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:g}"


def _cannot_be_negative(units: str) -> Optional[str]:
    """What kind of quantity a unit measures when that quantity has no negative
    values (a mass, an amount of substance, a length, area or volume, or any
    concentration of these), else None. Time, temperature, fractions and
    unitless scores can go below zero, so they are not judged."""
    if not units:
        return None
    from CohortVarLinker.src.units import parse_unit
    parsed = parse_unit(units)
    if parsed is None:
        return None
    dims = dict(parsed[1])
    if "[mass]" in dims and dims["[mass]"] > 0 and set(dims) <= {"[mass]", "[length]"}:
        return "a mass or mass concentration"
    if "[substance]" in dims and dims["[substance]"] > 0 and set(dims) <= {"[substance]", "[length]", "[time]"}:
        return "an amount or concentration of substance"
    if dims and set(dims) == {"[length]"}:
        return "a length, area, volume or count per volume"
    return None


def _content_issues_for(cohort: str, path: str) -> list[dict[str, Any]]:
    from src.upload import _CAT_PAIR, _category_code_key, _cat_strip, normalize_column_name
    df = pd.read_csv(path, na_values=[""], keep_default_na=False, dtype=str).fillna("")
    df.columns = [normalize_column_name(c) for c in df.columns]

    def cell(row: pd.Series, col: str) -> str:
        value = str(row.get(col, "") if col in df.columns else "").strip()
        return "" if value.lower() in ("na", "nan", "none") else value

    issues: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        row_no = int(i) + 2
        variable = cell(row, "VARIABLENAME") or None
        vartype = cell(row, "VARTYPE").upper()
        raw_min, raw_max = cell(row, "MIN"), cell(row, "MAX")

        def add(issue_type: str, message: str) -> None:
            issues.append({"cohort": cohort, "type": issue_type, "variable": variable, "row": row_no,
                           "message": message})

        # S1 and S6: the declared range.
        low = high = None
        if vartype in ("INT", "FLOAT"):
            low, high = (_number(raw_min) if raw_min else None), (_number(raw_max) if raw_max else None)
            for label, raw, value in (("MIN", raw_min, low), ("MAX", raw_max, high)):
                if raw and value is None:
                    add("syntax.invalid_range_bound", f"{label} '{raw}' is not a number, but VARTYPE is {vartype}.")
            if low is not None and high is not None and low > high:
                add("syntax.impossible_range", f"MIN {_fmt_number(low)} is above MAX {_fmt_number(high)}.")
            kind = _cannot_be_negative(cell(row, "UNITS"))
            if kind and low is not None and low < 0:
                add("syntax.impossible_range",
                    f"MIN is {_fmt_number(low)}, but '{cell(row, 'UNITS')}' measures {kind}, which cannot be negative.")
        d_low = d_high = None
        # A bare number shorter than a year (a month or day count typed
        # DATETIME) is no date to compare: the parser would read 1 as a day
        # and 99 as 1999. Such bounds are left unjudged.
        short_numbers = any(re.fullmatch(r"\d{1,3}", r) for r in (raw_min, raw_max) if r)
        if vartype == "DATETIME" and not short_numbers:
            d_low, d_high = (_date(raw_min) if raw_min else None), (_date(raw_max) if raw_max else None)
            for label, raw, value in (("MIN", raw_min, d_low), ("MAX", raw_max, d_high)):
                if raw and value is None:
                    add("syntax.invalid_range_bound", f"{label} '{raw}' is not a date, but VARTYPE is DATETIME.")
            if d_low is not None and d_high is not None and d_low > d_high:
                add("syntax.impossible_range", f"MIN {raw_min} is after MAX {raw_max}.")

        # S7: missing codes that are also a category, or a value inside the range.
        missing = [c.strip() for c in cell(row, "MISSING").split("|") if c.strip()]
        if not missing:
            continue
        categories: dict[str, str] = {}
        for entry in cell(row, "CATEGORICAL").split("|"):
            parts = _CAT_PAIR.split(entry.strip(), maxsplit=1)
            if len(parts) == 2 and _cat_strip(parts[0]):
                categories.setdefault(_category_code_key(parts[0]), entry.strip())
        for code in missing:
            entry = categories.get(_category_code_key(code))
            if entry:
                add("syntax.missing_code_collision",
                    f"Missing code '{code}' is also the category '{entry}': a {code} in the data cannot be "
                    f"told apart as missing or as that category.")
                continue
            value = _number(code)
            if value is not None and low is not None and high is not None and low <= value <= high:
                add("syntax.missing_code_collision",
                    f"Missing code '{code}' lies inside the declared range {_fmt_number(low)} to "
                    f"{_fmt_number(high)}, so it cannot be told apart from a real value.")
                continue
            as_date = _date(code) if d_low is not None and d_high is not None else None
            if as_date is not None and d_low <= as_date <= d_high:
                add("syntax.missing_code_collision",
                    f"Missing code '{code}' lies inside the declared range {raw_min} to {raw_max}, "
                    f"so it cannot be told apart from a real date.")
    return issues


_content_cache: dict[str, Any] = {"signature": None, "issues": None}


def _content_issues() -> list[dict[str, Any]]:
    """S1 (range bounds), S6 and S7 for every cohort's current dictionary,
    cached until a dictionary file changes."""
    from src.upload import get_latest_datadictionary
    signature = _dictionary_signature()
    with _syntax_lock:
        if _content_cache["signature"] == signature and _content_cache["issues"] is not None:
            return [dict(i) for i in _content_cache["issues"]]
    issues: list[dict[str, Any]] = []
    try:
        names = sorted(os.listdir(settings.cohort_folder))
    except OSError:
        names = []
    for name in names:
        folder = os.path.join(settings.cohort_folder, name)
        if not os.path.isdir(folder):
            continue
        path = get_latest_datadictionary(folder)
        if not path:
            continue
        try:
            issues += _content_issues_for(name, path)
        except Exception as e:
            logger.warning("range and missing-code check failed for %s: %s", name, e)
    with _syntax_lock:
        _content_cache.update(signature=signature, issues=issues)
    return [dict(i) for i in issues]


# ---------------------------------------------------------------------------
# Concept codes, checked in the background and kept on disk
# ---------------------------------------------------------------------------

# The concept check's components, by the columns of its result CSV.
_CONCEPT_COMPONENTS = [
    ("variable", "Variable concept", "variable_status", "variable_reason"),
    ("categorical", "Category concepts", "categorical_value_status", "categorical_value_reason"),
    ("unit", "Unit concept", "unit_status", "unit_reason"),
    ("visit", "Visit concept", "visit_status", "visit_reason"),
]

# Messages of the concept check that repeat what the syntax check already
# reports (S1, S4, S8, S9): dropped, so one problem is listed once.
_CDE_DUPLICATES = re.compile(
    r"exist but|is empty but|Mismatch in number|Multiple .* found|must be strings|"
    r"Invalid data type for visit|depends on|Additional context depends", re.I)
_CDE_TYPES = [
    ("concepts.code_id_mismatch", re.compile(
        r"do not match|only in capitalization|matched but|Neither concept_id nor concept_code", re.I)),
    ("concepts.wrong_vocabulary", re.compile(r"Vocabulary prefix missing/unknown", re.I)),
]
# "label= FAIL: description" items joined by "; " for a list component. A
# description may itself contain "; ", so items are split only before a label.
_CDE_ITEM = re.compile(r";\s(?=[^;=]*=\s(?:PASS|FAIL|N/A):)")


def _cde_failures(description: str) -> list[tuple[Optional[str], str]]:
    """(item label, reason) for each failing part of one component's verdict."""
    text = (description or "").strip()
    # "Invalid categorical format ... Additionally <items>": the format part is
    # S4, reported by the syntax check; keep the per-category verdicts.
    if text.lower().startswith("invalid categorical format"):
        _head, _sep, rest = text.partition("Additionally ")
        text = rest.strip()
        if not re.search(r"=\s(?:PASS|FAIL|N/A):", text):
            return []
    if re.search(r"=\s(?:PASS|FAIL|N/A):", text):
        out = []
        for item in _CDE_ITEM.split(text):
            m = re.match(r"\s*(.*?)=\s(PASS|FAIL|N/A):\s*(.*)$", item, re.S)
            if m and m.group(2) == "FAIL":
                out.append((m.group(1).strip() or None, m.group(3).strip().rstrip(";").strip()))
        return out
    return [(None, text)]


def _cde_issue_type(reason: str) -> Optional[str]:
    """The issue type of one failing verdict, or None when the syntax check
    already reports it."""
    if _CDE_DUPLICATES.search(reason):
        return None
    for issue_type, pattern in _CDE_TYPES:
        if pattern.search(reason):
            return issue_type
    return "concepts.other"


def _concept_results_path() -> str:
    return os.path.join(settings.data_folder, "data_quality", "concept_check.json")


_concept_lock = threading.Lock()
_concept_job: dict[str, Any] = {
    "running": False, "started_at": None, "finished_at": None,
    "done": 0, "total": 0, "current": None, "error": None,
}


def _load_concept_results() -> dict[str, Any]:
    try:
        with open(_concept_results_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_concept_result(cohort_id: str, record: dict[str, Any]) -> None:
    path = _concept_results_path()
    with _concept_lock:
        data = _load_concept_results()
        data.setdefault("cohorts", {})[cohort_id] = record
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, default=str)
        os.replace(tmp, path)


def _additional_context_reason(raw: str) -> Optional[str]:
    """The CSV keeps the additional-context verdict as a ValidationLog repr."""
    if "status='FAIL'" not in raw and 'status="FAIL"' not in raw:
        return None
    m = re.search(r"description='([^']*)'", raw) or re.search(r'description="([^"]*)"', raw)
    return m.group(1) if m else raw


def _check_cohort_concepts(cohort_id: str) -> dict[str, Any]:
    """Run the concept check on one cohort's current dictionary."""
    from CohortVarLinker.validate_cde import validate_dictionary
    from src.upload import get_latest_datadictionary

    folder = os.path.join(settings.cohort_folder, cohort_id)
    path = get_latest_datadictionary(folder) if os.path.isdir(folder) else None
    record: dict[str, Any] = {"checked_at": datetime.now().isoformat(timespec="seconds"), "issues": []}
    if not path:
        record["status"] = "no_dictionary"
        return record
    st = os.stat(path)
    record["dictionary_file"] = os.path.basename(path)
    record["dictionary_mtime_ns"] = st.st_mtime_ns

    out_csv = os.path.join(os.path.dirname(_concept_results_path()), f"_{cohort_id}_{os.getpid()}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    try:
        verdict = validate_dictionary(path, out_csv)
        result = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
        if verdict is None:
            # The structure check failed before any concept was looked at; its
            # CSV holds one row per problem with a status and a description.
            record["status"] = "structure_failed"
            for _, r in result.iterrows():
                text = str(r.get("description") or "Structure check failed")
                issue_type = "syntax.invalid_domain" if "column 'domain'" in text else "concepts.structure"
                record["issues"].append({"type": issue_type, "variable": None, "row": None, "message": text})
            return record
        record["status"] = "passed" if verdict else "failed"
        for _, r in result.iterrows():
            if str(r.get("overall_status", "")).strip() != "FAIL":
                continue
            variable = str(r.get("variable name") or "").strip() or None
            verdicts = [(field, label, str(r.get(reason_col) or ""))
                        for field, label, status_col, reason_col in _CONCEPT_COMPONENTS
                        if str(r.get(status_col, "")).strip() == "FAIL"]
            ac = _additional_context_reason(str(r.get("additional_context_status", "")))
            if ac:
                verdicts.append(("additional_context", "Additional context concept", ac))
            for field, label, description in verdicts:
                for item, reason in _cde_failures(description):
                    issue_type = _cde_issue_type(reason)
                    if issue_type is None or not reason or reason.lower() == "nan":
                        continue
                    what = f"{label} '{item}'" if item else label
                    record["issues"].append({"type": issue_type, "variable": variable, "row": None,
                                             "field": field, "message": f"{what}: {reason}"})

        # S10 to S14 beyond what the validator checks (src/concept_semantics.py).
        from src.concept_semantics import check_dictionary
        semantics = check_dictionary(path)
        reported = {(i["variable"], i.get("field")) for i in record["issues"]
                    if i["type"] == "concepts.code_id_mismatch"}
        for issue in semantics["issues"]:
            if issue["type"] == "concepts.code_id_mismatch" and (issue["variable"], issue.get("field")) in reported:
                continue  # the validator already named this disagreement
            record["issues"].append(issue)
        if semantics["athena_failed"]:
            record["issues"].append({
                "type": "concepts.check_failed", "variable": None, "row": None,
                "message": f"Athena could not be reached for {semantics['athena_failed']} of "
                           f"{semantics['concepts_checked']} concepts, so whether they are standard and "
                           f"valid (S12) was not checked, and ids missing from the local OMOP graph "
                           f"were not confirmed (S10). Run the check again later."})
        return record
    except Exception as e:
        logger.warning("concept check failed for %s: %s", cohort_id, e)
        record["status"] = "error"
        record["issues"].append({"type": "concepts.check_failed", "variable": None, "row": None,
                                 "message": f"The concept check stopped with an error: {e}"})
        return record
    finally:
        if os.path.exists(out_csv):
            os.remove(out_csv)


def _run_concept_check(cohort_ids: list[str]) -> None:
    try:
        for cohort_id in cohort_ids:
            _concept_job["current"] = cohort_id
            _save_concept_result(cohort_id, _check_cohort_concepts(cohort_id))
            _concept_job["done"] += 1
    except Exception as e:
        logger.exception("concept check run failed")
        _concept_job["error"] = str(e)
    finally:
        _concept_job.update(running=False, current=None,
                            finished_at=datetime.now().isoformat(timespec="seconds"))


def _concept_issues(cohort_ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The stored concept findings, and per cohort when they were computed and
    whether the dictionary has changed since."""
    from src.upload import get_latest_datadictionary
    stored = _load_concept_results().get("cohorts", {})
    issues: list[dict[str, Any]] = []
    status: dict[str, Any] = {}
    for cohort_id in cohort_ids:
        record = stored.get(cohort_id)
        if not record:
            status[cohort_id] = {"checked_at": None, "stale": False}
            continue
        folder = os.path.join(settings.cohort_folder, cohort_id)
        current = get_latest_datadictionary(folder) if os.path.isdir(folder) else None
        stale = bool(current) and (
            os.path.basename(current) != record.get("dictionary_file")
            or os.stat(current).st_mtime_ns != record.get("dictionary_mtime_ns"))
        status[cohort_id] = {"checked_at": record.get("checked_at"), "stale": stale,
                             "status": record.get("status")}
        for issue in record.get("issues", []):
            issues.append({"cohort": cohort_id, **issue, "stale": stale})
    return issues, status


# ---------------------------------------------------------------------------
# Data values
# ---------------------------------------------------------------------------

_CAUSE_TYPE = {
    "undeclared missing code": "values.undeclared_missing_code",
    "negative minimum": "values.negative_minimum",
    "extreme outlier": "values.extreme_outlier",
}


def _value_issues() -> tuple[list[dict[str, Any]], Optional[str]]:
    from src.suspect_values import _all_rows
    data = _all_rows()
    issues = []
    for r in data["rows"]:
        if r["score"] < 1:
            continue
        issues.append({
            "cohort": r["cohort"],
            "type": _CAUSE_TYPE.get(r["likely_cause"], "values.extreme_outlier"),
            "variable": r["variable"],
            "row": None,
            "message": "; ".join(r["reasons"]),
            "reasons": r["reasons"],
            "level": r["level"],
            "level_rank": r["level_rank"],
            "suspect_value": r.get("suspect_value"),
            "median": r.get("median"),
            "minimum": r.get("minimum"),
            "maximum": r.get("maximum"),
            "declared_missing": r.get("declared_missing") or [],
            "measure": r.get("measure"),
            "label": r.get("label"),
            "units": r.get("units"),
            "visits": r.get("visits"),
            "omop_id": r.get("omop_id"),
            "concept_code": r.get("concept_code"),
        })
    return issues, data.get("built_at")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _require_admin(user: dict[str, Any]) -> None:
    if user.get("email") not in settings.admins_list:
        raise HTTPException(status_code=403, detail="Admins only.")


def _known_cohorts() -> list[str]:
    from src.cohort_cache import get_cohorts_from_cache
    return sorted(get_cohorts_from_cache(settings.admins_list[0]).keys())


def _attach_variable_details(issues: list[dict[str, Any]]) -> None:
    """Label, units and visit from the cohort cache, for issues that name a
    variable but did not come with them (syntax and concept issues)."""
    from src.cohort_cache import get_cohorts_from_cache
    cohorts = get_cohorts_from_cache(settings.admins_list[0])
    lookup: dict[str, dict[str, Any]] = {}
    for issue in issues:
        if not issue.get("variable") or issue.get("label"):
            continue
        cohort = cohorts.get(issue["cohort"])
        if cohort is None:
            continue
        if issue["cohort"] not in lookup:
            lookup[issue["cohort"]] = {name.lower(): v for name, v in (cohort.variables or {}).items()}
        var = lookup[issue["cohort"]].get(issue["variable"].lower())
        if var is not None:
            issue["label"] = getattr(var, "var_label", None)
            issue["units"] = getattr(var, "units", None)
            issue["visits"] = getattr(var, "visits", None)
            issue["omop_id"] = getattr(var, "omop_id", None)
            issue["concept_code"] = getattr(var, "concept_code", None)


@router.get("/data-quality/issues")
def get_data_quality_issues(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Every known issue of every cohort, from the three sources, each issue
    carrying its cohort, type, variable where there is one, and message."""
    _require_admin(user)
    from src.suspect_values import _profiling_files

    cohorts = _known_cohorts()
    syntax = _syntax_report()
    issues = _syntax_issues(syntax) + _content_issues()
    concept_issues, concept_status = _concept_issues(cohorts)
    issues += concept_issues
    value_issues, values_read_at = _value_issues()
    issues += value_issues
    _attach_variable_details(issues)

    order = {key: i for i, key in enumerate(ISSUE_TYPES)}
    for issue in issues:
        if issue["type"] not in ISSUE_TYPES:
            # A finding stored under a type that has since been renamed.
            issue["type"] = "concepts.other" if issue["type"].startswith("concepts.") else "syntax.other"
        category, _label, _desc, spec = ISSUE_TYPES[issue["type"]]
        issue["spec"] = spec
        issue["blocking"] = category == "syntax" and issue["type"] not in NON_BLOCKING_TYPES
        issue["category"] = category
        issue.setdefault("level", None)
        issue.setdefault("level_rank", None)
    issues.sort(key=lambda i: (order[i["type"]], i["level_rank"] if i["level_rank"] is not None else 0,
                               i["cohort"], (i.get("variable") or "").lower(), i.get("row") or 0))

    profiled = {cohort_id for cohort_id, _p, _v in _profiling_files()}
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cohorts": sorted(set(cohorts) | profiled | {i["cohort"] for i in issues}),
        "cohorts_with_profiling": sorted(profiled),
        "categories": CATEGORIES,
        "types": [{"key": k, "category": c, "label": l, "description": d, "spec": sp}
                  for k, (c, l, d, sp) in ISSUE_TYPES.items()],
        "issues": issues,
        "sources": {
            "syntax": {"checked_at": syntax.get("generated_at")},
            "concepts": {"cohorts": concept_status, "job": dict(_concept_job)},
            "values": {"read_at": values_read_at},
        },
    }


@router.post("/data-quality/concept-check")
def start_concept_check(cohort: Optional[str] = None, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Start the concept check in the background, for one cohort or all."""
    _require_admin(user)
    cohorts = _known_cohorts()
    if cohort:
        match = next((c for c in cohorts if c.lower() == cohort.strip().lower()), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Unknown cohort {cohort!r}.")
        targets = [match]
    else:
        targets = cohorts
    with _concept_lock:
        if _concept_job["running"]:
            raise HTTPException(status_code=409, detail="A concept check is already running.")
        _concept_job.update(running=True, started_at=datetime.now().isoformat(timespec="seconds"),
                            finished_at=None, done=0, total=len(targets), current=None, error=None,
                            cohorts=targets)
    threading.Thread(target=_run_concept_check, args=(targets,), daemon=True).start()
    return dict(_concept_job)


@router.get("/data-quality/concept-check/status")
def concept_check_status(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    _require_admin(user)
    return dict(_concept_job)
