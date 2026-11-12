"""Derived variables: rows for variables a study can compute but does not have.

A derived variable (settings.DERIVED_VARIABLES_LIST, e.g. BMI from height and
weight) is added for a pair of studies when:

    one study has it and the other can compute it   -> real variable <-> derived_<name>
    neither has it but both can compute it          -> derived_<name> <-> derived_<name>
    both have it                                    -> nothing; ordinary matching pairs them

"Has it" and "has an input" use the OMOP graph, not only the configured id: a
study variable counts when it is the same concept (OmopGraphNX.check_concept_match),
e.g. TIME-CHF body surface area coded as LOINC 3005424 instead of the configured
SNOMED 4201235, or triglycerides in mmol/L (3025839) instead of mg/dL (3022192).

A study computes a variable only at visits where all its inputs occur. A
computed side has no min, max or data type (those need the data); a side with
the real variable keeps its own.

An optional "condition" on a definition restricts when the formula is valid,
e.g. Friedewald LDL only for triglycerides < 400 mg/dL:

    "condition": {"input": 3022192, "op": "<", "value": 400, "unit": "mg/dL",
                  "converted": {"mmol/L": 4.52}, "note": "fasting sample"}

It is checked per computing study against the input's min and max:
    all      max (or min) is on the valid side: every patient qualifies
    none     no patient qualifies: the study does not get the derived variable
    some     the range straddles the threshold: derive, and filter per patient
             when the formula is applied to the data
    unknown  no min/max for the input, or its unit has no threshold
The result goes into <side>_derivation_condition and the condition text into
transformation_rule.
"""

from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

from .data_model import ContextMatchType, MappingRelation
from .fuzz_match import FuzzyMatcher

_OPS = {"<", "<=", ">", ">="}


def _as_int(value) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _visit(value) -> str:
    return str(value or "").strip().lower()


def _norm_unit(unit) -> str:
    return str(unit or "").strip().lower().removeprefix("ucum:")


def same_concept_ids(graph, concept_id: int, study_ids: Iterable[int]) -> Set[int]:
    """concept_id plus the study concept ids the OMOP graph matches to it."""
    ids = {int(concept_id)}
    if graph is None:
        return ids
    for i in study_ids:
        if i not in ids and graph.check_concept_match(i, concept_id)[0]:
            ids.add(i)
    return ids


def _rows_with_ids(rows: list, ids: Set[int], key: str = "omop_id") -> list:
    return [r for r in rows if _as_int(r.get(key)) in ids]


def _input_visits(context: dict, input_ids: List[Set[int]], side: str) -> Set[str]:
    """Visits at which every input occurs on `side` (empty: cannot compute)."""
    id_key = "somop_id" if side == "source" else "tomop_id"
    visit_key = "source_visit" if side == "source" else "target_visit"
    per_input: List[Set[str]] = []
    for ids in input_ids:
        rows = _rows_with_ids(context[side], ids)
        if rows:
            visits = {_visit(r.get("visit")) for r in rows}
        else:
            visits = {_visit(r.get(visit_key)) for r in _rows_with_ids(context["mapped"], ids, id_key)}
        visits.discard("")
        # An input with no visit-resolved occurrence cannot anchor a shared-visit
        # derivation, so the side cannot compute the variable.
        if not visits:
            return set()
        per_input.append(visits)
    return set.intersection(*per_input) if per_input else set()


def condition_text(condition: Optional[dict]) -> str:
    if not condition:
        return ""
    text = f"valid only where concept {condition['input']} {condition['op']} {condition['value']} {condition['unit']}"
    for unit, value in (condition.get("converted") or {}).items():
        text += f" ({value} {unit})"
    if condition.get("note"):
        text += f"; {condition['note']}"
    return text


def check_condition(condition: dict, input_ids: Set[int], nodes: Iterable[Any], visit: str) -> str:
    """all / some / none / unknown for one study at one visit, from the input's min and max."""
    op = condition["op"]
    if op not in _OPS:
        raise ValueError(f"unknown condition operator {op!r}")
    thresholds = {_norm_unit(condition["unit"]): float(condition["value"])}
    thresholds.update({_norm_unit(u): float(v) for u, v in (condition.get("converted") or {}).items()})

    lows, highs = [], []
    for node in nodes:
        if _as_int(node.main_id) not in input_ids or _visit(node.visit) != visit:
            continue
        threshold = thresholds.get(_norm_unit(node.unit))
        stats = node.statistics[0] if isinstance(node.statistics, tuple) else node.statistics
        lo = getattr(stats, "min_val", None)
        hi = getattr(stats, "max_val", None)
        if threshold is None or lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
            return "unknown"
        # Shift each range so the threshold is 0; ranges in different units then combine.
        lows.append(float(lo) - threshold)
        highs.append(float(hi) - threshold)
    if not lows:
        return "unknown"
    lo, hi = min(lows), max(highs)
    if op == "<":
        return "all" if hi < 0 else "none" if lo >= 0 else "some"
    if op == "<=":
        return "all" if hi <= 0 else "none" if lo > 0 else "some"
    if op == ">":
        return "all" if lo > 0 else "none" if hi <= 0 else "some"
    return "all" if lo >= 0 else "none" if hi < 0 else "some"


def _name_for_visit(rows: list, visit: str, default: str, key: str) -> str:
    for row in rows:
        if _visit(row.get("visit")) == visit:
            return row.get(key) or default
    return default


def derived_variable_rows(
    context: dict,
    definition: dict,
    derived_ids: Set[int],
    input_ids: List[Set[int]],
    condition_ids: Set[int],
    nodes: Dict[str, list],
) -> List[Dict[str, Any]]:
    """Rows for one derived variable between two studies (empty if none apply)."""
    name = definition["name"]
    code, label = definition["code"], definition["label"]
    omop_id = int(definition["omop_id"])
    condition = definition.get("condition")

    source_has = _rows_with_ids(context["source"], derived_ids)
    target_has = _rows_with_ids(context["target"], derived_ids)
    if source_has and target_has:
        return []
    for row in context["mapped"]:
        if _as_int(row.get("somop_id")) in derived_ids and _as_int(row.get("tomop_id")) in derived_ids:
            return []

    source_visits = _input_visits(context, input_ids, "source")
    target_visits = _input_visits(context, input_ids, "target")
    source_visits |= {_visit(r.get("visit")) for r in source_has} - {""}
    target_visits |= {_visit(r.get("visit")) for r in target_has} - {""}
    if not source_visits or not target_visits:
        return []

    aligned = [(s, t) for s in source_visits for t in target_visits
               if FuzzyMatcher.check_visit_string(s, t)]
    if not aligned:
        return []

    stats_type = "continuous_variable"
    has_rows = source_has + target_has
    if has_rows and has_rows[0].get("stats_type"):
        stats_type = has_rows[0].get("stats_type")
    src_unit = source_has[0].get("unit_label") if source_has else ""
    tgt_unit = target_has[0].get("unit_label") if target_has and not src_unit else ""
    unit = definition.get("unit", "")

    rule = f"Derived variable {name} using parameter columns {definition['required_omops']}."
    if condition:
        rule += f" Condition: {condition_text(condition)}."

    rows = []
    for src_visit, tgt_visit in aligned:
        derived_name = f"derived_{name}"
        source_name = _name_for_visit(source_has, src_visit, derived_name, "source")
        target_name = _name_for_visit(target_has, tgt_visit, derived_name, "target")

        checks = {}
        for side, var, visit in (("source", source_name, src_visit), ("target", target_name, tgt_visit)):
            computed = var == derived_name
            checks[side] = (check_condition(condition, condition_ids, nodes[side], visit)
                            if condition and computed else "")
        if "none" in checks.values():
            continue

        row = {
            "source": source_name, "target": target_name,
            # A side that only computes the variable has no dictionary label;
            # it falls back to the standard label.
            "source_label": _name_for_visit(source_has, src_visit, label, "description"),
            "target_label": _name_for_visit(target_has, tgt_visit, label, "description"),
            "somop_id": omop_id, "tomop_id": omop_id,
            "scode": code, "slabel": label, "tcode": code, "tlabel": label,
            "source_visit": src_visit, "target_visit": tgt_visit,
            "category": definition["category"],
            "source_type": stats_type, "target_type": stats_type,
            "source_unit": src_unit or unit, "target_unit": tgt_unit or unit,
            "source_composite_code_labels": label, "source_composite_code_omop_ids": f"{omop_id}",
            "target_composite_code_labels": label, "target_composite_code_omop_ids": f"{omop_id}",
            "mapping_relation": MappingRelation.SymbolicCloseMatch.value,
            "context_match_type": ContextMatchType.PARTIAL.value,
            "sim_score": 0.8,
            "derived_variable_name": name,
            "transformation_rule": rule,
            "source_derivation_condition": checks["source"],
            "target_derivation_condition": checks["target"],
        }
        rows.append(row)
    return rows


def _real_variable_stats(row: dict, nodes_by_name: Dict[str, Dict[str, Any]]) -> None:
    """A side with the real variable keeps its own min, max and data type."""
    for side in ("source", "target"):
        node = nodes_by_name[side].get(row.get(side))
        if node is None:
            continue
        stats = node.statistics[0] if isinstance(node.statistics, tuple) else node.statistics
        row[f"{side}_min_val"] = getattr(stats, "min_val", None)
        row[f"{side}_max_val"] = getattr(stats, "max_val", None)
        row[f"{side}_data_type"] = node.data_type or ""


def compute_derived_variables(df: pd.DataFrame, src_collection, tgt_collection,
                              graph, definitions: List[dict]) -> pd.DataFrame:
    """Append derived-variable rows to the mapped DataFrame."""
    nodes = {"source": list(src_collection.variables), "target": list(tgt_collection.variables)}
    context = {
        "source": [n.to_element_dict("source") for n in nodes["source"]],
        "target": [n.to_element_dict("target") for n in nodes["target"]],
        "mapped": df.to_dict("records"),
    }
    study_ids = {i for n in nodes["source"] + nodes["target"] if (i := _as_int(n.main_id)) is not None}

    new_rows = []
    for definition in definitions:
        condition = definition.get("condition")
        new_rows.extend(derived_variable_rows(
            context, definition,
            derived_ids=same_concept_ids(graph, definition["omop_id"], study_ids),
            input_ids=[same_concept_ids(graph, i, study_ids) for i in definition["required_omops"]],
            condition_ids=same_concept_ids(graph, condition["input"], study_ids) if condition else set(),
            nodes=nodes,
        ))

    nodes_by_name = {side: {n.name: n for n in nodes[side]} for side in nodes}
    for row in new_rows:
        _real_variable_stats(row, nodes_by_name)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df
