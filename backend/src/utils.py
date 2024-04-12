from typing import Any

import curies
from rdflib import Dataset, Namespace
from rdflib.namespace import RDF, RDFS
from SPARQLWrapper import JSON, SPARQLWrapper

from src.config import settings
from src.models import Cohort, CohortVariable, VariableCategory

# Define the namespaces
ICARE = Namespace("https://w3id.org/icare4cvd/")

query_endpoint = SPARQLWrapper(settings.query_endpoint)
query_endpoint.setReturnFormat(JSON)

converter = curies.get_bioregistry_converter()


def init_graph(default_graph: str | None = None) -> Dataset:
    """Initialize a new RDF graph for nquads with the iCARE4CVD namespace bindings."""
    g = Dataset(store="Oxigraph", default_graph_base=default_graph)
    g.bind("icare", ICARE)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    return g


def run_query(query: str) -> dict[str, Any]:
    """Function to run a SPARQL query against a remote endpoint"""
    query_endpoint.setQuery(query)
    return query_endpoint.query().convert()


get_variables_query = """PREFIX icare: <https://w3id.org/icare4cvd/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX dcterms: <http://purl.org/dc/terms/>

SELECT DISTINCT ?cohortId ?cohortInstitution ?cohortType ?cohortEmail ?study_type ?study_participants
    ?study_duration ?study_ongoing ?study_population ?study_objective ?airlock
    ?variable ?varName ?varLabel ?varType ?index ?count ?na ?max ?min ?units ?formula ?definition
    ?omopDomain ?conceptId ?mappedId ?mappedLabel ?visits ?categoryValue ?categoryLabel ?categoryConceptId ?categoryMappedId ?categoryMappedLabel
WHERE {
    GRAPH ?cohortMetadataGraph {
        ?cohort a icare:Cohort ;
            dc:identifier ?cohortId ;
            icare:institution ?cohortInstitution .
        OPTIONAL { ?cohort icare:cohortType ?cohortType . }
        OPTIONAL { ?cohort icare:email ?cohortEmail . }
        OPTIONAL { ?cohort icare:studyType ?study_type . }
        OPTIONAL { ?cohort icare:studyParticipants ?study_participants . }
        OPTIONAL { ?cohort icare:studyDuration ?study_duration . }
        OPTIONAL { ?cohort icare:studyOngoing ?study_ongoing . }
        OPTIONAL { ?cohort icare:studyPopulation ?study_population . }
        OPTIONAL { ?cohort icare:studyObjective ?study_objective . }
    }

    OPTIONAL {
        GRAPH ?cohortVarGraph {
            ?cohort icare:hasVariable ?variable .
            ?variable a icare:Variable ;
                dc:identifier ?varName ;
                rdfs:label ?varLabel ;
                icare:varType ?varType ;
                icare:index ?index .
            OPTIONAL { ?variable icare:count ?count }
            OPTIONAL { ?variable icare:na ?na }
            OPTIONAL { ?variable icare:max ?max }
            OPTIONAL { ?variable icare:min ?min }
            OPTIONAL { ?variable icare:units ?units }
            OPTIONAL { ?variable icare:formula ?formula }
            OPTIONAL { ?variable icare:definition ?definition }
            OPTIONAL { ?variable icare:conceptId ?conceptId }
            OPTIONAL { ?variable icare:omop ?omopDomain }
            OPTIONAL { ?variable icare:visits ?visits }
            OPTIONAL {
                ?variable icare:categories ?category.
                ?category rdfs:label ?categoryLabel ;
                    rdf:value ?categoryValue .
                OPTIONAL { ?category icare:conceptId ?categoryConceptId }
            }
        }
    }

    OPTIONAL {
        GRAPH ?cohortMappingsGraph {
            OPTIONAL {
                ?variable icare:mappedId ?mappedId .
                OPTIONAL { ?mappedId rdfs:label ?mappedLabel }
            }
            OPTIONAL {
                ?category icare:mappedId ?categoryMappedId .
                OPTIONAL { ?categoryMappedId rdfs:label ?categoryMappedLabel }
            }
            OPTIONAL { ?cohort icare:previewEnabled ?airlock . }
        }
    }
} ORDER BY ?cohort ?index
"""


# TODO: Utility to get value or None if key is missing or value is empty string
def get_value(key: str, row: dict[str, Any]) -> str | None:
    return str(row[key]["value"]) if key in row and row[key]["value"] else None


def get_int_value(key: str, row: dict[str, Any]) -> int | None:
    return int(row[key]["value"]) if key in row and row[key]["value"] else None


def get_bool_value(key: str, row: dict[str, Any]) -> bool:
    return str(row[key]["value"]).lower() == "true" if key in row and row[key]["value"] else False


def get_curie_value(key: str, row: dict[str, Any]) -> int | None:
    return converter.compress(get_value(key, row)) if get_value(key, row) else None


def retrieve_cohorts_metadata(user_email: str) -> dict[str, Cohort]:
    """Get all cohorts metadata from the SPARQL endpoint (infos, variables)"""
    results = run_query(get_variables_query)["results"]["bindings"]
    cohorts_with_variables = {}
    cohorts_without_variables = {}
    print(f"Get cohorts metadata query results: {len(results)}")
    for row in results:
        cohort_id = str(row["cohortId"]["value"])
        var_id = str(row["varName"]["value"]) if "varName" in row else None
        # Determine which dictionary to use
        target_dict = cohorts_with_variables if var_id else cohorts_without_variables

        # Initialize cohort data structure if not exists
        if cohort_id and cohort_id not in target_dict:
            target_dict[cohort_id] = Cohort(
                cohort_id=row["cohortId"]["value"],
                cohort_type=get_value("cohortType", row),
                cohort_email=[get_value("cohortEmail", row)] if get_value("cohortEmail", row) else [],
                # owner=get_value("owner", row),
                institution=get_value("cohortInstitution", row),
                study_type=get_value("study_type", row),
                study_participants=get_value("study_participants", row),
                study_duration=get_value("study_duration", row),
                study_ongoing=get_value("study_ongoing", row),
                study_population=get_value("study_population", row),
                study_objective=get_value("study_objective", row),
                variables={},
                airlock=get_bool_value("airlock", row),
                can_edit=user_email in [*settings.admins_list, get_value("cohortEmail", row)],
            )
        elif get_value("cohortEmail", row) not in target_dict[cohort_id].cohort_email:
            # Handle multiple emails for the same cohort
            target_dict[cohort_id].cohort_email.append(get_value("cohortEmail", row))
            if user_email == get_value("cohortEmail", row):
                target_dict[cohort_id].can_edit = True

        # Process variables
        if "varName" in row and var_id not in target_dict[cohort_id].variables:
            target_dict[cohort_id].variables[var_id] = CohortVariable(
                var_name=row["varName"]["value"],
                var_label=row["varLabel"]["value"],
                var_type=row["varType"]["value"],
                count=int(row["count"]["value"]),
                max=get_value("max", row),
                min=get_value("min", row),
                units=get_value("units", row),
                visits=get_value("visits", row),
                formula=get_value("formula", row),
                definition=get_value("definition", row),
                concept_id=get_curie_value("conceptId", row),
                mapped_id=get_curie_value("mappedId", row),
                mapped_label=get_value("mappedLabel", row),
                omop_domain=get_value("omopDomain", row),
                index=get_int_value("index", row),
                na=get_int_value("na", row) or 0,
            )

        # Process categories of variables
        if "varName" in row and "categoryLabel" in row and "categoryValue" in row:
            target_dict[cohort_id].variables[var_id].categories.append(
                VariableCategory(
                    value=str(row["categoryValue"]["value"]),
                    label=str(row["categoryLabel"]["value"]),
                    concept_id=get_curie_value("categoryConceptId", row),
                    mapped_id=get_curie_value("categoryMappedId", row),
                    mapped_label=get_value("categoryMappedLabel", row),
                )
            )

    # Merge dictionaries, cohorts with variables first
    return {**cohorts_with_variables, **cohorts_without_variables}
