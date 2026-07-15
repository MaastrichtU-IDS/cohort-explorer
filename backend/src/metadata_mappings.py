"""Manual concept-mapping subjects and cache hydration.

Dictionary metadata and user-created mappings intentionally live in separate
named graphs.  The cache is rebuilt from the canonical CSV first, then this
module overlays the manual mappings from the cohort's mapping graph.
"""

from __future__ import annotations

from collections.abc import Mapping

from rdflib import URIRef

from src.models import CohortVariable, VariableCategory
from src.utils import curie_converter, normalize_text, run_query

ICARE_BASE = "https://w3id.org/icare4cvd/"


def mapping_graph_uri(cohort_id: str) -> URIRef:
    return URIRef(f"{ICARE_BASE}cohort/{cohort_id.strip().replace(' ', '_')}/mappings")


def variable_subject_uri(cohort_id: str, var_name: str) -> URIRef:
    cohort = cohort_id.strip().replace(" ", "_")
    variable = var_name.strip().replace(" ", "_")
    if not variable:
        raise ValueError("Variable ID cannot be empty")
    return URIRef(f"{ICARE_BASE}cohort/{cohort}/{variable}")


def category_subject_uri(cohort_id: str, var_name: str, category_label: str) -> URIRef:
    normalized_label = normalize_text(category_label)
    if not normalized_label:
        raise ValueError("Category label cannot be empty")
    return URIRef(
        f"{variable_subject_uri(cohort_id, var_name)}/"
        f"categorical_value_specification/{normalized_label}"
    )


def _binding_value(row: dict, key: str) -> str:
    binding = row.get(key, {})
    if not isinstance(binding, dict):
        return ""
    return str(binding.get("value", ""))


def _compact_uri(uri: str) -> str:
    return curie_converter.compress(uri) or uri


def hydrate_manual_mappings(
    cohort_id: str,
    variables: Mapping[str, CohortVariable],
) -> None:
    """Overlay manual variable/category mappings onto freshly parsed CSV data."""
    targets: dict[str, CohortVariable | VariableCategory] = {}
    for var_name, variable in variables.items():
        targets[str(variable_subject_uri(cohort_id, var_name))] = variable
        for category in variable.categories:
            try:
                subject = category_subject_uri(cohort_id, var_name, category.label)
            except ValueError:
                continue
            targets[str(subject)] = category

    if not targets:
        return

    subjects = " ".join(f"<{subject}>" for subject in sorted(targets))
    query = f"""
    PREFIX icare: <https://w3id.org/icare4cvd/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?subject ?mappedId ?mappedLabel
    WHERE {{
        GRAPH <{mapping_graph_uri(cohort_id)}> {{
            VALUES ?subject {{ {subjects} }}
            ?subject icare:mappedId ?mappedId .
            OPTIONAL {{ ?mappedId rdfs:label ?mappedLabel . }}
        }}
    }}
    """
    bindings = run_query(query).get("results", {}).get("bindings", [])
    for row in bindings:
        target = targets.get(_binding_value(row, "subject"))
        mapped_id = _binding_value(row, "mappedId")
        if target is None or not mapped_id:
            continue
        target.mapped_id = _compact_uri(mapped_id)
        target.mapped_label = _binding_value(row, "mappedLabel") or None
