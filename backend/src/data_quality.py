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
from fastapi.responses import FileResponse

from src.auth import get_current_user
from src.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue types
# ---------------------------------------------------------------------------

CATEGORIES: list[dict[str, str]] = [
    {"key": "syntax", "label": "Dictionary syntax",
     "description": "Problems in how the data dictionary is written. Each one blocks a dictionary upload."},
    {"key": "concepts", "label": "Concept codes",
     "description": "Concept codes that do not match OMOP: wrong code, wrong name for the code, or a code that does not exist."},
    {"key": "integrity", "label": "Dictionary vs data",
     "description": "Whether the dictionary and the dataset describe the same thing, from the last variable profiling."},
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
    "integrity.variable_not_in_data": ("integrity", "Variable not in data", "A dictionary variable with no matching column in the data.", "I1"),
    "integrity.column_not_in_dictionary": ("integrity", "Column not in dictionary", "A data column with no dictionary entry.", "I2"),
    "integrity.type_contradicted": ("integrity", "Type contradicted", "The values of a variable are not of its VARTYPE, such as numbers in a DATETIME variable.", "I3"),
    "integrity.unreadable_values": ("integrity", "Values not readable", "Some values of a variable could not be read as its type.", "I3"),
    "integrity.mixed_formats": ("integrity", "Mixed value formats", "A variable mixes date formats, day-first and month-first dates, or decimal points and decimal commas.", "I4"),
    "integrity.duplicate_rows": ("integrity", "Duplicate records", "Rows that are exact copies of an earlier row.", "I5"),
    "integrity.duplicate_ids": ("integrity", "Duplicate participant IDs", "A few participants appear on more than one row where each should have one.", "I5"),
    "integrity.record_structure": ("integrity", "Record structure", "Rows per participant do not fit the layout the dictionary describes.", "I6"),
    "values.undeclared_missing_code": ("values", "Undeclared missing code", "A value shaped like a missing code (999, -1) that the dictionary does not declare.", None),
    "values.negative_minimum": ("values", "Negative minimum", "A negative value in a column that is otherwise positive.", None),
    "values.extreme_outlier": ("values", "Extreme outlier", "A value far beyond the rest of its column.", None),
}


# A short example of each type, for the page's explanation on hover: what the
# problem looks like and, where it helps, what it should be. Generic, not taken
# from any cohort.
EXAMPLES: dict[str, str] = {
    "syntax.missing_columns": "The dictionary has no VARTYPE column.",
    "syntax.invalid_vartype": "VARTYPE 'NUMBER' → use one of STR, INT, FLOAT, DATETIME.",
    "syntax.invalid_domain": "DOMAIN 'labs' → name an OMOP table, such as 'measurement'.",
    "syntax.concept_code_format": "'loinc 2160-0' or 'xyz:123' → 'loinc:2160-0'.",
    "syntax.omop_id_format": "OMOP id '3016723.5' or '3016723, 3020564' → one integer, '3016723'.",
    "syntax.missing_code_format": "MISSING '999 000' → '999000' or '999.000'.",
    "syntax.invalid_range_bound": "INT variable with MAX 'unknown'; DATETIME variable with MIN 'early'.",
    "syntax.missing_required_field": "A row with a variable name but no VARTYPE.",
    "syntax.duplicate_variable": "Two rows both named 'age'.",
    "syntax.category_order": "INT variable with 'yes=1|no=0' → '1=yes|0=no'.",
    "syntax.category_code_type": "INT variable with category '1.5=mild' or 'abc=yes'.",
    "syntax.category_format": "'0=no,1=yes' → '0=no|1=yes'; 'yes|1=no' (one entry without '=').",
    "syntax.repeated_category_code": "'1=yes|1=no' → '1=yes|2=no'.",
    "syntax.empty_category_label": "'1=yes|2=' → '1=yes|2=no'.",
    "syntax.impossible_range": "MIN 10 and MAX 5; a concentration in mg/L with MIN -0.06.",
    "syntax.missing_code_collision": "MISSING 9 with categories '1=yes|2=no|9=unknown'; MISSING 999 with range 0 to 1000.",
    "syntax.orphan_concept": "UNIT CONCEPT CODE 'ucum:mg/dL' on a row whose UNITS is empty.",
    "syntax.concept_count_mismatch": "Three categories but two CATEGORICAL VALUE OMOP IDs.",
    "syntax.other": "A dictionary message that fits none of the types above.",
    "concepts.code_id_mismatch": "Code 'loinc:2160-0' (creatinine) with OMOP id 3004249 (systolic blood pressure).",
    "concepts.wrong_vocabulary": "A unit coded 'loinc:2345-7'; a variable coded with a UCUM unit.",
    "concepts.wrong_domain": "The dictionary declares a heart failure concept to be in domain \"Observation\". Correct domain: \"Condition\".",
    "concepts.deprecated": "A SNOMED code that Athena marks deprecated or replaced by another.",
    "concepts.non_standard": "An ICD-10 code or an ATC class where OMOP expects a standard SNOMED or RxNorm concept.",
    "concepts.name_mismatch": "The dictionary associates loinc:2345-7 with the name 'Blood sugar'; LOINC associates it with 'Glucose [Mass/volume] in Serum or Plasma'.",
    "concepts.vartype_implausible": "A quantitative LOINC test declared STR; a nominal one declared FLOAT.",
    "concepts.unit_implausible": "A LOINC mass concentration ([Mass/volume]) with UNITS 'mmol/L'.",
    "concepts.structure": "A DOMAIN value the concept check does not accept, or columns it cannot find.",
    "concepts.other": "A concept message that fits none of the types above.",
    "concepts.check_failed": "Athena could not be reached for some concepts, so they went unchecked.",
    "integrity.variable_not_in_data": "The dictionary lists 'bmi_v12', but the data has no such column.",
    "integrity.column_not_in_dictionary": "The data has a column 'bmi_v12' that the dictionary does not list.",
    "integrity.type_contradicted": "VARTYPE DATETIME, but the column holds plain numbers such as 1998.",
    "integrity.unreadable_values": "A FLOAT column holding '<40' or 'n/a' in a few cells.",
    "integrity.mixed_formats": "Dates as '2020-01-05' and '05/01/2020' in one column; '12.5' and '12,5'.",
    "integrity.duplicate_rows": "Two rows identical in every column.",
    "integrity.duplicate_ids": "A participant on two rows in data that has one row per participant.",
    "integrity.record_structure": "A third of participants on several rows, while the dictionary has one column per visit.",
    "values.undeclared_missing_code": "999 in a column where only 9999 is declared missing.",
    "values.negative_minimum": "-1 in a column of days that is otherwise positive.",
    "values.extreme_outlier": "QT interval 4060 ms in a column whose median is 412.",
}

# The level of every issue type outside the data values, whose level comes
# from their suspicion score. Three levels, as for the values: high for what is
# plainly wrong or blocks use of the data, moderate for what needs checking,
# low for what is usually cosmetic or a limit of the check itself. Every
# dictionary syntax issue is high: it would block a new upload of the dictionary.
TYPE_LEVELS: dict[str, str] = {
    **{k: "high" for k in ISSUE_TYPES if k.startswith("syntax.")},
    "concepts.code_id_mismatch": "high",
    "concepts.deprecated": "high",
    "concepts.structure": "high",
    "concepts.wrong_vocabulary": "moderate",
    "concepts.wrong_domain": "moderate",
    "concepts.non_standard": "moderate",
    "concepts.vartype_implausible": "moderate",
    "concepts.unit_implausible": "moderate",
    "concepts.other": "moderate",
    "concepts.name_mismatch": "low",
    "concepts.check_failed": "low",
    "integrity.duplicate_rows": "high",
    "integrity.duplicate_ids": "high",
    "integrity.variable_not_in_data": "moderate",
    "integrity.column_not_in_dictionary": "moderate",
    "integrity.type_contradicted": "moderate",
    "integrity.mixed_formats": "moderate",
    "integrity.record_structure": "moderate",
    "integrity.unreadable_values": "low",
}
LEVEL_RANK = {"high": 0, "moderate": 1, "low": 2}

# Shown on the page only when a cohort has some: catch-alls and problems with
# the check itself rather than kinds of issue.
OCCASIONAL_TYPES = {"syntax.other", "concepts.other", "concepts.structure", "concepts.check_failed"}


def _day_first(iso: Optional[str]) -> str:
    try:
        return datetime.fromisoformat(str(iso)).strftime("%d.%m.%Y")
    except (TypeError, ValueError):
        return str(iso or "")


def _check_status(cohorts: list[str], syntax_report: dict[str, Any],
                  concept_status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """For every cohort and issue type, whether the type was checked, and if
    not, why: {cohort: {type: {"checked": bool, "note": str}}}. A count of
    zero means nothing only when the type was checked."""
    has_dictionary = {c["cohort_id"]: c["status"] in ("ok", "errors")
                      for c in syntax_report.get("cohorts", [])}
    facts = _profiling_facts()
    out: dict[str, dict[str, Any]] = {}
    for cohort in cohorts:
        f = facts.get(cohort)
        concept = concept_status.get(cohort) or {}
        per: dict[str, Any] = {}
        for key in ISSUE_TYPES:
            category = ISSUE_TYPES[key][0]
            checked, note = True, ""
            if category == "syntax":
                if not has_dictionary.get(cohort):
                    checked, note = False, "no dictionary on file"
                else:
                    note = "checked against the current dictionary"
            elif category == "concepts":
                if not concept.get("checked_at"):
                    checked, note = False, "the concept check has not been run for this cohort"
                else:
                    note = f"checked {_day_first(concept['checked_at'])}" + (
                        "; the dictionary changed since" if concept.get("stale") else "")
            elif category in ("integrity", "values"):
                if not f:
                    checked, note = False, "the cohort has no variable profiling yet"
                elif key == "integrity.column_not_in_dictionary" and not f.get("c1_lists"):
                    checked, note = False, "needs the name lists collected at the next profiling download"
                elif key == "integrity.unreadable_values" and f.get("version") != "v2":
                    checked, note = False, "needs a profiling in the newer (v2) format"
                elif key in ("integrity.mixed_formats", "integrity.duplicate_rows",
                             "integrity.duplicate_ids", "integrity.record_structure") and not f.get("measures"):
                    checked, note = False, "needs a profiling by a clean room set up after 26.11.2026"
                elif key in ("integrity.duplicate_ids", "integrity.record_structure") and not f.get("patient_id"):
                    checked, note = False, "no column is annotated as the patient identifier"
                else:
                    note = f"profiling of {_day_first(f.get('profiled_at'))}"
            per[key] = {"checked": checked, "note": note}
        out[cohort] = per
    return out


# ---------------------------------------------------------------------------
# Dictionary syntax
# ---------------------------------------------------------------------------

def _syntax_rules() -> list[tuple[str, re.Pattern[str]]]:
    # Imported here: upload.py is large and imports much of the backend.
    from src.upload import _TAG_EMPTY_LABEL, _TAG_REPEATED_CODE
    rules = [
        ("syntax.repeated_category_code", re.escape(_TAG_REPEATED_CODE)),
        ("syntax.empty_category_label", re.escape(_TAG_EMPTY_LABEL)),
        ("syntax.invalid_range_bound", r"(?:MIN|MAX) '.*' is not a (?:number|date), but VARTYPE"),
        ("syntax.impossible_range", r"MIN .* is (?:above|after) MAX|which cannot be negative"),
        ("syntax.missing_code_collision", r"Missing code '.*' (?:is also the category|lies inside the declared range)"),
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
# Dictionary vs data (I1 to I3), from the variable profiling
# ---------------------------------------------------------------------------

def _profiled_kind(entry: dict[str, Any], version: str) -> Optional[str]:
    """numeric, categorical, date or text: how the profiling read the column."""
    raw = str(entry.get("type") or "").lower()
    for kind in ("numeric", "categorical", "date", "text"):
        if raw.startswith(kind) or (kind == "date" and raw.startswith("datetime")):
            return kind
    return None


def _declared_type(entry: dict[str, Any], version: str) -> str:
    if version == "v2":
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        return str(meta.get("var_type") or "").strip().upper()
    return str(entry.get("vartype (metadata dictionary)") or "").strip().upper()


# VARTYPE → the profiled kinds its values may come out as. A coded INT or FLOAT
# variable comes out categorical, and a STR variable with few distinct values
# does too, so categorical fits every type except DATETIME.
_KINDS_FITTING = {
    "INT": {"numeric", "categorical"},
    "FLOAT": {"numeric", "categorical"},
    "STR": {"categorical", "text"},
    "DATETIME": {"date"},
}

_KIND_WORDS = {"numeric": "numbers", "categorical": "a few distinct codes",
               "date": "dates", "text": "free text"}


def _c1_names(folder: str, filename: str, column: str) -> Optional[list[str]]:
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return None
    return [str(v).strip() for v in df.get(column, pd.Series(dtype=str)).tolist() if str(v).strip()]


def _integrity_issues_for(cohort: str, path: str, version: str,
                          facts: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    from src.upload import get_latest_datadictionary, normalize_column_name
    with open(path, encoding="utf-8") as f:
        raw = json.load(f, parse_constant=lambda _c: None)
    entries = raw.get("variables") if version == "v2" else raw
    if not isinstance(entries, dict):
        return []
    profiled_at = datetime.fromtimestamp(os.path.getmtime(path))
    when = profiled_at.strftime("%d.%m.%Y")
    folder = os.path.dirname(path)

    dictionary_path = get_latest_datadictionary(os.path.join(settings.cohort_folder, cohort))
    stale = bool(dictionary_path) and datetime.fromtimestamp(os.path.getmtime(dictionary_path)) > profiled_at
    issues: list[dict[str, Any]] = []

    def add(issue_type: str, variable: Optional[str], message: str) -> None:
        issues.append({"cohort": cohort, "type": issue_type, "variable": variable, "row": None,
                       "message": message, "stale": stale})

    # I1 and I2. c1's own comparison where the server has it (collected with the
    # profiling since it was added); it is case-sensitive while the profiling is
    # not, so a name differing only in case is taken as a match.
    not_in_data = _c1_names(folder, "in_dictionary_not_in_dataset.csv", "In Dictionary Not in Dataset")
    not_in_dict = _c1_names(folder, "in_dataset_not_in_dictionary.csv", "In Dataset Not in Dictionary")
    if not_in_data is not None and not_in_dict is not None:
        case_only = {n.lower() for n in not_in_data} & {n.lower() for n in not_in_dict}
        for name in sorted(not_in_data, key=str.lower):
            if name.lower() not in case_only:
                add("integrity.variable_not_in_data", name,
                    f"No column of this name in the data (profiling of {when}).")
        for name in sorted(not_in_dict, key=str.lower):
            if name.lower() not in case_only:
                add("integrity.column_not_in_dictionary", name,
                    f"A data column with no entry in the dictionary (profiling of {when}).")
    elif dictionary_path:
        # Without c1's lists: the profiling covers every column that is in both
        # the data and the dictionary, so a dictionary variable it has no entry
        # for is missing from the data, unless the profiling says it was a
        # patient identifier or failed. Columns missing from the dictionary
        # leave no trace, so I2 waits for the next profiling download.
        notes = [str(n) for n in (raw.get("data_issues") or [])] if version == "v2" else []
        skipped = set()
        for note in notes:
            m = re.match(r"Variable (\S+) is a patient identifier", note) or \
                re.match(r"Failed to perform EDA on column (\S+?)\.", note)
            if m:
                skipped.add(m.group(1).strip().lower())
        profiled = {str(k).strip().lower() for k in entries}
        df = pd.read_csv(dictionary_path, na_values=[""], keep_default_na=False, dtype=str).fillna("")
        df.columns = [normalize_column_name(c) for c in df.columns]
        for name in df.get("VARIABLENAME", pd.Series(dtype=str)).tolist():
            name = str(name).strip()
            if name and name.lower() not in profiled and name.lower() not in skipped:
                add("integrity.variable_not_in_data", name,
                    f"The profiling of {when} has no column of this name.")

    # I4 to I6, measured in the enclave (v2 profiling from the version that
    # added them on; an older profiling has no "integrity" block).
    measures = raw.get("integrity") if version == "v2" and isinstance(raw.get("integrity"), dict) else None
    if facts is not None:
        facts.update({
            "version": version,
            "profiled_at": profiled_at.isoformat(timespec="seconds"),
            "c1_lists": not_in_data is not None and not_in_dict is not None,
            "measures": bool(measures) and "error" not in (measures or {}),
            "patient_id": bool((measures or {}).get("patient_id_column")),
        })
    if measures:
        issues += _measured_integrity_issues(cohort, measures, dictionary_path, when, stale)

    # I3: the profiled kind of each variable against its VARTYPE.
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        declared = _declared_type(entry, version)
        kind = _profiled_kind(entry, version)
        if declared in _KINDS_FITTING and kind and kind not in _KINDS_FITTING[declared]:
            add("integrity.type_contradicted", name,
                f"VARTYPE is {declared}, but the values came out as {_KIND_WORDS[kind]} (profiling of {when}).")
        comp = entry.get("completeness") if isinstance(entry.get("completeness"), dict) else {}
        n_invalid = comp.get("n_invalid") or 0
        if isinstance(n_invalid, (int, float)) and n_invalid > 0:
            n = int(n_invalid)
            add("integrity.unreadable_values", name,
                f"{n} value{'s' if n != 1 else ''} could not be read as {_KIND_WORDS.get(kind or '', kind or 'its type')}"
                f"{f' (VARTYPE {declared})' if declared else ''}; profiling of {when}.")
    return issues


# Share of participants with more than one row from which the data is taken to
# be in long layout rather than to hold a few duplicated participants.
_LONG_LAYOUT_SHARE = 0.05


def _dictionary_is_wide(dictionary_path: Optional[str]) -> Optional[bool]:
    """Whether the dictionary puts visits in separate columns: one concept
    recorded as several variables at different visits (hscrp_bl, hscrp_v12).
    None when the dictionary cannot be read."""
    from src.upload import normalize_column_name
    if not dictionary_path:
        return None
    try:
        df = pd.read_csv(dictionary_path, na_values=[""], keep_default_na=False, dtype=str).fillna("")
    except Exception:
        return None
    df.columns = [normalize_column_name(c) for c in df.columns]
    if "VISITS" not in df.columns:
        return False
    visits_by_concept: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        concept = str(row.get("VARIABLE OMOP ID") or row.get("VARIABLE CONCEPT CODE") or "").strip().lower()
        visit = str(row.get("VISITS") or "").strip().lower()
        if concept and concept not in ("na", "nan") and visit and visit not in ("na", "nan"):
            visits_by_concept.setdefault(concept, set()).add(visit)
    return any(len(v) >= 2 for v in visits_by_concept.values())


def _measured_integrity_issues(cohort: str, m: dict[str, Any], dictionary_path: Optional[str],
                               when: str, stale: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(issue_type: str, variable: Optional[str], message: str) -> None:
        out.append({"cohort": cohort, "type": issue_type, "variable": variable, "row": None,
                    "message": message, "stale": stale})

    # I4
    for variable, f in sorted((m.get("mixed_formats") or {}).items()):
        if f.get("kind") == "decimal":
            extra = (f"; {f['comma_three_digits']} more have three digits after a comma, which may be "
                     f"a thousands separator" if f.get("comma_three_digits") else "")
            add("integrity.mixed_formats", variable,
                f"{f.get('decimal_point', 0)} values use a decimal point and {f.get('decimal_comma', 0)} a "
                f"decimal comma{extra} (profiling of {when}).")
        elif f.get("kind") == "date":
            shapes = ", ".join(f"{shape} ({n})" for shape, n in (f.get("shapes") or {}).items())
            order = ""
            if f.get("day_first") and f.get("month_first"):
                order = (f"; {f['day_first']} dates can only be day-first and {f['month_first']} only "
                         f"month-first")
            add("integrity.mixed_formats", variable,
                f"Dates are written in several layouts: {shapes}{order} (Y is a four-digit year, n a "
                f"day or month; profiling of {when}).")

    # I5: exact copies, whatever the layout.
    n_rows = m.get("n_rows") or 0
    dup = m.get("exact_duplicate_rows") or 0
    if dup:
        add("integrity.duplicate_rows", None,
            f"{dup} of {n_rows} rows are exact copies of an earlier row (profiling of {when}).")

    # I5 and I6 from rows per participant, judged against the dictionary's layout.
    rpp = m.get("rows_per_participant") or {}
    participants = rpp.get("participants") or 0
    multi = rpp.get("with_multiple_rows") or 0
    if participants and multi:
        wide = _dictionary_is_wide(dictionary_path)
        share = multi / participants
        id_col = m.get("patient_id_column") or "the patient identifier"
        if wide:
            extra_rows = (rpp.get("rows_with_id") or 0) - participants
            if share < _LONG_LAYOUT_SHARE:
                add("integrity.duplicate_ids", id_col,
                    f"{multi} participants appear on more than one row ({extra_rows} extra rows), but the "
                    f"dictionary puts each visit in its own column, so each participant should have one "
                    f"row (profiling of {when}).")
            else:
                add("integrity.record_structure", id_col,
                    f"{multi} of {participants} participants ({share:.0%}) have more than one row (median "
                    f"{rpp.get('median', 0):g}, up to {rpp.get('max', 0)}), so the data looks long, but the "
                    f"dictionary puts visits in separate columns, which is a wide layout (profiling of {when}).")
    if m.get("rows_without_id"):
        add("integrity.record_structure", m.get("patient_id_column"),
            f"{m['rows_without_id']} rows have no participant identifier (profiling of {when}).")
    return out


_integrity_lock = threading.Lock()
_integrity_cache: dict[str, Any] = {"signature": None, "issues": None}


def _integrity_issues() -> list[dict[str, Any]]:
    from src.suspect_values import _profiling_files, _signature
    files = _profiling_files()
    c1 = []
    for cohort_id, path, _v in files:
        for name in ("in_dictionary_not_in_dataset.csv", "in_dataset_not_in_dictionary.csv"):
            p = os.path.join(os.path.dirname(path), name)
            c1.append(f"{name}:{os.path.getmtime(p) if os.path.exists(p) else 0}")
    signature = _signature(files) + "|" + "|".join(c1) + "|" + _dictionary_signature()
    with _integrity_lock:
        if _integrity_cache["signature"] == signature and _integrity_cache["issues"] is not None:
            return [dict(i) for i in _integrity_cache["issues"]]
    issues: list[dict[str, Any]] = []
    facts: dict[str, dict[str, Any]] = {}
    for cohort_id, path, version in files:
        facts[cohort_id] = {}
        try:
            issues += _integrity_issues_for(cohort_id, path, version, facts[cohort_id])
        except Exception as e:
            logger.warning("integrity check failed for %s: %s", cohort_id, e)
    with _integrity_lock:
        _integrity_cache.update(signature=signature, issues=issues, facts=facts)
    return [dict(i) for i in issues]


def _profiling_facts() -> dict[str, dict[str, Any]]:
    """Per profiled cohort: version, date, whether c1's name lists and the
    enclave's integrity measures are there, whether it has a patient id column.
    Filled by _integrity_issues, which runs first."""
    with _integrity_lock:
        return dict(_integrity_cache.get("facts") or {})


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

_dictionary_lock = threading.Lock()
_dictionary_cache: dict[str, Any] = {}


def _dictionary_index(cohort_id: str) -> Optional[dict[str, Any]]:
    """The cohort's current dictionary (the file its card dates too), read once
    per version of the file: the path, when it was uploaded (the file's date),
    the columns as written, and each row by lower-cased variable name, with
    the row number as a spreadsheet shows it (header is row 1)."""
    from src.upload import current_datadictionary, normalize_column_name
    folder = os.path.join(settings.cohort_folder, cohort_id)
    path = current_datadictionary(folder) if os.path.isdir(folder) else None
    if not path:
        return None
    st = os.stat(path)
    key = f"{path}:{st.st_mtime_ns}:{st.st_size}"
    with _dictionary_lock:
        hit = _dictionary_cache.get(cohort_id)
        if hit and hit["key"] == key:
            return hit
    try:
        df = pd.read_csv(path, na_values=[""], keep_default_na=False, dtype=str).fillna("")
    except Exception as e:
        logger.warning("cannot read the dictionary of %s: %s", cohort_id, e)
        return None
    columns = [str(c) for c in df.columns]
    normalized = [normalize_column_name(c) for c in columns]
    rows: dict[str, dict[str, Any]] = {}
    for i, values in enumerate(df.itertuples(index=False, name=None)):
        raw = dict(zip(columns, (str(v) for v in values)))
        norm = dict(zip(normalized, (str(v).strip() for v in values)))
        name = norm.get("VARIABLENAME", "").strip()
        if name and name.lower() not in rows:
            rows[name.lower()] = {"row": i + 2, "raw": raw, "norm": norm}
    index = {
        "key": key, "path": path, "file": os.path.basename(path),
        "uploaded_at": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "columns": columns, "rows": rows, "n_variables": len(rows),
    }
    with _dictionary_lock:
        _dictionary_cache[cohort_id] = index
    return index


def _blank(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return None if text.lower() in ("", "na", "nan", "none") else text


def _charted_variables(cohort_id: str) -> set[str]:
    """Lower-cased names of the variables that have a distribution chart: the
    {name}.png files the profiling left in the cohort's results folder, which
    is where /api/variable-graph serves them from."""
    folder = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}")
    try:
        return {n[:-4].lower() for n in os.listdir(folder) if n.lower().endswith(".png")}
    except OSError:
        return set()


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
    issues = _syntax_issues(syntax)
    concept_issues, concept_status = _concept_issues(cohorts)
    issues += concept_issues
    issues += _integrity_issues()
    value_issues, values_read_at = _value_issues()
    issues += value_issues
    _attach_variable_details(issues)
    # Label, standard concept name and visit from the dictionary itself, which
    # holds them for every cohort whatever the state of the cache.
    dictionaries = {c: _dictionary_index(c) for c in sorted(set(cohorts) | {i["cohort"] for i in issues})}
    for issue in issues:
        if not issue.get("variable"):
            continue
        index = dictionaries.get(issue["cohort"])
        entry = (index or {}).get("rows", {}).get(issue["variable"].strip().lower())
        issue["in_dictionary"] = entry is not None
        if entry is None:
            continue
        norm = entry["norm"]
        issue["label"] = _blank(norm.get("VARIABLELABEL")) or issue.get("label")
        issue["concept_name"] = _blank(norm.get("VARIABLE CONCEPT NAME")) or issue.get("concept_name")
        issue["visit_name"] = _blank(norm.get("VISIT CONCEPT NAME")) or _blank(norm.get("VISITS"))
        issue["visits"] = _blank(norm.get("VISITS")) or issue.get("visits")
        issue["units"] = _blank(norm.get("UNITS")) or issue.get("units")
        issue["omop_id"] = _blank(norm.get("VARIABLE OMOP ID")) or issue.get("omop_id")
        issue["concept_code"] = _blank(norm.get("VARIABLE CONCEPT CODE")) or issue.get("concept_code")
    charts: dict[str, set[str]] = {}
    for issue in issues:
        if issue.get("variable"):
            if issue["cohort"] not in charts:
                charts[issue["cohort"]] = _charted_variables(issue["cohort"])
            issue["has_chart"] = issue["variable"].strip().lower() in charts[issue["cohort"]]

    order = {key: i for i, key in enumerate(ISSUE_TYPES)}
    for issue in issues:
        if issue["type"] not in ISSUE_TYPES:
            # A finding stored under a type that has since been renamed.
            issue["type"] = "concepts.other" if issue["type"].startswith("concepts.") else "syntax.other"
        category, _label, _desc, spec = ISSUE_TYPES[issue["type"]]
        issue["spec"] = spec
        issue["category"] = category
        if category != "values":
            issue["level"] = TYPE_LEVELS.get(issue["type"], "moderate")
            issue["level_rank"] = LEVEL_RANK[issue["level"]]
        # A dictionary syntax issue is one the upload validation rejects.
        issue["blocks_upload"] = category == "syntax"
    issues.sort(key=lambda i: (order[i["type"]], i["level_rank"] if i["level_rank"] is not None else 0,
                               i["cohort"], (i.get("variable") or "").lower(), i.get("row") or 0))

    profiled = {cohort_id for cohort_id, _p, _v in _profiling_files()}
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cohorts": sorted(set(cohorts) | profiled | {i["cohort"] for i in issues}),
        # Per cohort: its dictionary (file, when uploaded, how many variables),
        # how many issues, and how many distinct variables they concern.
        "cohort_summary": _cohort_summary(sorted(set(cohorts) | profiled | {i["cohort"] for i in issues}),
                                          issues, dictionaries),
        "cohorts_with_profiling": sorted(profiled),
        # The profiling notes files on disk, per cohort, for the page's buttons.
        "data_issues_files": {c: files for c in sorted(set(cohorts) | profiled)
                              if (files := _data_issues_files(c))},
        "categories": CATEGORIES,
        "types": [{"key": k, "category": c, "label": l, "description": d, "spec": sp,
                   "example": EXAMPLES.get(k), "occasional": k in OCCASIONAL_TYPES}
                  for k, (c, l, d, sp) in ISSUE_TYPES.items()],
        "check_status": _check_status(sorted(set(cohorts) | profiled), syntax, concept_status),
        "issues": issues,
        "sources": {
            "syntax": {"checked_at": syntax.get("generated_at")},
            "concepts": {"cohorts": concept_status, "job": dict(_concept_job)},
            "values": {"read_at": values_read_at},
        },
    }


# The notes the profiling enclave writes next to its results, by file name:
# label for the page. Only these names are ever served.
DATA_ISSUES_FILES = {
    "data_issues.json": "data_issues.txt (generated during EDA)",
    "data_issues_longitudinal.json": "data_issues_longitudinal.txt (generated during the longitudinal analysis)",
}


def _data_issues_files(cohort_id: str) -> list[dict[str, Any]]:
    folder = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}")
    out = []
    for name, label in DATA_ISSUES_FILES.items():
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            st = os.stat(path)
            out.append({"name": name, "label": label, "size": st.st_size,
                        "generated_at": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
    return out


def _known_cohort(cohort: str) -> str:
    """The cohort id as the catalog or the profiling folders spell it; 404 otherwise.
    It becomes part of a path, so only a known id is accepted."""
    from src.suspect_values import _profiling_files
    known = set(_known_cohorts()) | {c for c, _p, _v in _profiling_files()}
    match = next((c for c in known if c.lower() == cohort.strip().lower()), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Unknown cohort {cohort!r}.")
    return match


def _cohort_summary(cohorts: list[str], issues: list[dict[str, Any]],
                    dictionaries: dict[str, Optional[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cohort in cohorts:
        mine = [i for i in issues if i["cohort"] == cohort]
        index = dictionaries.get(cohort) or _dictionary_index(cohort)
        out[cohort] = {
            "issues": len(mine),
            "variables_with_issues": len({i["variable"].strip().lower() for i in mine if i.get("variable")}),
            "variables_total": (index or {}).get("n_variables"),
            "dictionary_file": (index or {}).get("file"),
            "dictionary_uploaded_at": (index or {}).get("uploaded_at"),
        }
    return out


@router.get("/data-quality/dictionary/{cohort}")
def download_dictionary(cohort: str, user: Any = Depends(get_current_user)) -> FileResponse:
    """The cohort's current dictionary file, as uploaded."""
    _require_admin(user)
    cohort_id = _known_cohort(cohort)
    index = _dictionary_index(cohort_id)
    if not index:
        raise HTTPException(status_code=404, detail=f"{cohort_id} has no dictionary.")
    return FileResponse(index["path"], media_type="text/csv", filename=index["file"])


@router.get("/data-quality/dictionary-row/{cohort}/{variable}")
def dictionary_row(cohort: str, variable: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """One variable's row of the cohort's current dictionary, every column as written."""
    _require_admin(user)
    cohort_id = _known_cohort(cohort)
    index = _dictionary_index(cohort_id)
    if not index:
        raise HTTPException(status_code=404, detail=f"{cohort_id} has no dictionary.")
    entry = index["rows"].get(variable.strip().lower())
    if entry is None:
        raise HTTPException(status_code=404, detail=f"{variable!r} is not in the dictionary of {cohort_id}.")
    return {"cohort": cohort_id, "variable": variable, "file": index["file"],
            "uploaded_at": index["uploaded_at"], "row_number": entry["row"],
            "columns": index["columns"], "values": entry["raw"]}


@router.get("/data-quality/data-issues-files/{cohort}/{name}")
def get_data_issues_file(cohort: str, name: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """The text of one of a cohort's profiling notes files."""
    _require_admin(user)
    if name not in DATA_ISSUES_FILES:
        raise HTTPException(status_code=404, detail="No such file.")
    cohort_id = _known_cohort(cohort)
    entry = next((f for f in _data_issues_files(cohort_id) if f["name"] == name), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"{cohort_id} has no {name}.")
    path = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}", name)
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    return {**entry, "cohort": cohort_id, "content": content}


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
