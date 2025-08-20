import logging
from typing import Any, Dict, Tuple

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

# CURIES converter docs: https://curies.readthedocs.io/en/latest/
# Using URIs from from https://bioregistry.io/
prefix_map = [
    {
        "prefix": "snomedct",
        "uri_prefix": "http://snomed.info/id/",
        "prefix_synonyms": ["snomed", "SNOMED"],
    },
    {
        "prefix": "loinc",
        "uri_prefix": "http://loinc.org/rdf/",
        "prefix_synonyms": ["LOINC"],
    },
    {
        "prefix": "icare",
        "uri_prefix": str(ICARE),
    },
    {
        "prefix": "icd10",
        "uri_prefix": "https://icd.who.int/browse10/2019/en#/",
    },
    {
        "prefix": "atc",
        "uri_prefix": "http://www.whocc.no/atc_ddd_index/?code=",
    },
    {   
        "prefix": "rxnorm",
        "uri_prefix": "https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm=",
        "prefix_synonyms": ["RXNORM"]
    },
    {
        "prefix": "cdisc",
        "uri_prefix": "https://www.cdisc.org/search?search_api_fulltext="
    },
    {
        "prefix": "omop",
        "uri_prefix": "https://athena.ohdsi.org/search-terms/terms/",
        "prefix_synonyms": ["OMOP"]
    },
    {
        "prefix": "ucum",
        "uri_prefix": "http://unitsofmeasure.org/ucum/",
        "prefix_synonyms": ["UCUM"]
    }
]
curie_converter = curies.load_extended_prefix_map(prefix_map)

# Old way to do it using the predefined bioregistry:
# curie_converter = curies.get_bioregistry_converter()
# curie_converter.add_prefix("icare", str(ICARE))

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
    ?study_duration ?study_ongoing ?study_population ?study_objective ?primary_outcome_spec ?secondary_outcome_spec
    ?male_percentage ?female_percentage ?study_start ?study_end
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
        OPTIONAL { ?cohort icare:primaryOutcomeSpec ?primary_outcome_spec . }
        OPTIONAL { ?cohort icare:secondaryOutcomeSpec ?secondary_outcome_spec . }
        OPTIONAL { ?cohort icare:malePercentage ?male_percentage . }
        OPTIONAL { ?cohort icare:femalePercentage ?female_percentage . }
        OPTIONAL { ?cohort icare:studyStart ?study_start . }
        OPTIONAL { ?cohort icare:studyEnd ?study_end . }
        
        # We'll handle inclusion and exclusion criteria separately
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
            # OPTIONAL { ?cohort icare:previewEnabled ?airlock . }
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
    return curie_converter.compress(get_value(key, row)) if get_value(key, row) else None


def get_cohort_criteria(cohort_id: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Retrieve inclusion and exclusion criteria for a specific cohort.
    
    Args:
        cohort_id: The cohort identifier.
        
    Returns:
        Tuple[Dict[str, str], Dict[str, str]]: Tuple containing inclusion and exclusion criteria dictionaries.
    """
    inclusion_criteria = {}
    exclusion_criteria = {}
    
    # Query for inclusion criteria
    inclusion_query = """
    PREFIX icare: <https://w3id.org/icare4cvd/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT ?criterionLabel ?criterionValue
    WHERE {
        GRAPH ?cohortMetadataGraph {
            ?cohort a icare:Cohort ;
                dc:identifier ?cohortId .
            ?cohort ?predicate ?criterionValue .
            ?predicate rdf:type icare:InclusionCriterion .
            ?predicate rdfs:label ?criterionLabel .
            FILTER(?cohortId = "%s")
        }
    }
    """ % cohort_id
    
    # Query for exclusion criteria
    exclusion_query = """
    PREFIX icare: <https://w3id.org/icare4cvd/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT ?criterionLabel ?criterionValue
    WHERE {
        GRAPH ?cohortMetadataGraph {
            ?cohort a icare:Cohort ;
                dc:identifier ?cohortId .
            ?cohort ?predicate ?criterionValue .
            ?predicate rdf:type icare:ExclusionCriterion .
            ?predicate rdfs:label ?criterionLabel .
            FILTER(?cohortId = "%s")
        }
    }
    """ % cohort_id
    
    try:
        # Get inclusion criteria
        inclusion_results = run_sparql_query(inclusion_query)
        for row in inclusion_results["results"]["bindings"]:
            label = row.get("criterionLabel", {}).get("value")
            value = row.get("criterionValue", {}).get("value")
            if label and value:
                inclusion_criteria[label] = value
                
        # Get exclusion criteria
        exclusion_results = run_sparql_query(exclusion_query)
        for row in exclusion_results["results"]["bindings"]:
            label = row.get("criterionLabel", {}).get("value")
            value = row.get("criterionValue", {}).get("value")
            if label and value:
                exclusion_criteria[label] = value
    except Exception as e:
        logger.error(f"Error retrieving criteria for cohort {cohort_id}: {str(e)}")
    
    return inclusion_criteria, exclusion_criteria


def retrieve_cohorts_metadata(user_email: str) -> dict[str, Cohort]:
    """Get all cohorts metadata from the SPARQL endpoint (infos, variables)"""
    results = run_query(get_variables_query)["results"]["bindings"]
    cohorts_with_variables = {}
    cohorts_without_variables = {}
    logging.info(f"Get cohorts metadata query results: {len(results)}")
    for row in results:
        try:
            cohort_id = str(row["cohortId"]["value"])
            var_id = str(row["varName"]["value"]) if "varName" in row else None
            # Determine which dictionary to use
            target_dict = cohorts_with_variables if var_id else cohorts_without_variables

            # Initialize cohort data structure if not exists
            if cohort_id and cohort_id not in target_dict:
                # Get inclusion and exclusion criteria for this cohort
                inclusion_criteria, exclusion_criteria = get_cohort_criteria(cohort_id)
                
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
                    primary_outcome_spec=get_value("primary_outcome_spec", row),
                    secondary_outcome_spec=get_value("secondary_outcome_spec", row),
                    male_percentage=float(get_value("male_percentage", row)) if get_value("male_percentage", row) else None,
                    female_percentage=float(get_value("female_percentage", row)) if get_value("female_percentage", row) else None,
                    inclusion_criteria=inclusion_criteria,
                    exclusion_criteria=exclusion_criteria,
                    study_start=get_value("study_start", row),
                    study_end=get_value("study_end", row),
                    variables={},
                    # airlock=get_bool_value("airlock", row),
                    can_edit=user_email in [*settings.admins_list, get_value("cohortEmail", row)],
                    physical_dictionary_exists=False # Initialize here, will attempt to set below
                )
                # Attempt to determine if a physical dictionary file exists
                try:
                    if target_dict[cohort_id].metadata_filepath: # Accessing the property
                        target_dict[cohort_id].physical_dictionary_exists = True
                except FileNotFoundError:
                    target_dict[cohort_id].physical_dictionary_exists = False

            elif get_value("cohortEmail", row) not in target_dict[cohort_id].cohort_email:
                # Handle multiple emails for the same cohort
                target_dict[cohort_id].cohort_email.append(get_value("cohortEmail", row))
                if user_email == get_value("cohortEmail", row):
                    target_dict[cohort_id].can_edit = True
                    
            # Inclusion and exclusion criteria are now handled separately by get_cohort_criteria()

            # Process variables
            if "varName" in row and var_id not in target_dict[cohort_id].variables:
                target_dict[cohort_id].variables[var_id] = CohortVariable(
                    var_name=row["varName"]["value"],
                    var_label=row["varLabel"]["value"],
                    var_type=row["varType"]["value"],
                    count=get_int_value("count", row) or 0,
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
            # raise Exception(f"OLALALA")

            # Process categories of variables
            if "varName" in row and "categoryLabel" in row and "categoryValue" in row:
                new_category = VariableCategory(
                    value=str(row["categoryValue"]["value"]),
                    label=str(row["categoryLabel"]["value"]),
                    concept_id=get_curie_value("categoryConceptId", row),
                    mapped_id=get_curie_value("categoryMappedId", row),
                    mapped_label=get_value("categoryMappedLabel", row),
                )
                # Check for duplicates before appending
                if new_category not in target_dict[cohort_id].variables[var_id].categories:
                    target_dict[cohort_id].variables[var_id].categories.append(new_category)
        except Exception as e:
            logging.warning(f"Error processing row {row}: {e}")
    # Merge dictionaries, cohorts with variables first
    return {**cohorts_with_variables, **cohorts_without_variables}
