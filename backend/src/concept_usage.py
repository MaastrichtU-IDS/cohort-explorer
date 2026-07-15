from collections.abc import Sequence

from src.utils import run_query


def _binding_value(row: dict, name: str) -> str:
    value = row.get(name, {})
    return str(value.get("value", "")) if isinstance(value, dict) else ""


def get_concept_usage(concept_uris: Sequence[str]) -> dict[str, list[dict[str, str]]]:
    unique_uris = sorted(set(concept_uris))
    usage_by_uri: dict[str, list[dict[str, str]]] = {uri: [] for uri in unique_uris}
    if not unique_uris:
        return usage_by_uri

    values = " ".join(f"<{uri}>" for uri in unique_uris)
    query = f"""
    PREFIX cmeo: <https://w3id.org/CMEO/>
    PREFIX icare: <https://w3id.org/icare4cvd/>
    PREFIX obi: <http://purl.obolibrary.org/obo/obi.owl/>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?cohortId ?varName ?varLabel ?omopDomain ?mappedId
    WHERE {{
        VALUES ?mappedId {{ {values} }}
        GRAPH ?cohortVarGraph {{
            ?cohort a cmeo:study ;
                dc:identifier ?cohortId .
            ?variable a cmeo:data_element ;
                dc:identifier ?varName ;
                rdfs:label ?varLabel .
            OPTIONAL {{ ?variable cmeo:domain ?omopDomain }}
        }}
        {{
            GRAPH ?cohortMappingsGraph {{
                ?variable icare:mappedId ?mappedId .
            }}
        }} UNION {{
            GRAPH ?cohortVarGraph {{
                ?category a obi:categorical_value_specification ;
                    obi:specifies_value_of ?variable .
            }}
            GRAPH ?cohortMappingsGraph {{
                ?category icare:mappedId ?mappedId .
            }}
        }}
    }}
    ORDER BY ?cohortId ?varName
    """

    bindings = run_query(query).get("results", {}).get("bindings", [])
    for row in bindings:
        mapped_id = _binding_value(row, "mappedId")
        if mapped_id not in usage_by_uri:
            continue
        entry = {
            "cohort_id": _binding_value(row, "cohortId"),
            "var_name": _binding_value(row, "varName"),
            "var_label": _binding_value(row, "varLabel"),
            "omop_domain": _binding_value(row, "omopDomain"),
        }
        if entry not in usage_by_uri[mapped_id]:
            usage_by_uri[mapped_id].append(entry)

    for entries in usage_by_uri.values():
        entries.sort(key=lambda entry: (entry["cohort_id"], entry["var_name"]))
    return usage_by_uri
