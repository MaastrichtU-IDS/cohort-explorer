import logging
from typing import Any

import curies
from rdflib import Dataset, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, DC
from enum import Enum
import urllib.parse
from SPARQLWrapper import JSON, SPARQLWrapper

from src.config import settings
from src.models import Cohort, CohortVariable, VariableCategory

# Define the namespaces
ICARE = Namespace("https://w3id.org/icare4cvd/")

class OntologyNamespaces(Enum):
    CMEO = Namespace("https://w3id.org/CMEO/")
    OMOP = Namespace("http://omop.org/OMOP/")
    ATC = Namespace("http://purl.bioontology.org/ontology/ATC/")
    RXNORM = Namespace("http://purl.bioontology.org/ontology/RXNORM/")
    UCUM = Namespace("http://unitsofmeasure.org/")
    OMOP_EXT = Namespace("http://omop.org/omopextension/")
    OWL = Namespace("http://www.w3.org/2002/07/owl#")
    OBI = Namespace("http://purl.obolibrary.org/obo/obi.owl/")
    BFO = Namespace("http://purl.obolibrary.org/obo/bfo.owl/")
    STATO = Namespace("http://purl.obolibrary.org/obo/stato.owl/")
    DEFAULT_VALUE = 'Unmapped'
    SNOMEDCT = Namespace("http://purl.bioontology.org/ontology/SNOMEDCT/")
    LOINC = Namespace("http://purl.bioontology.org/ontology/LNC/") 
    RO = Namespace("http://purl.obolibrary.org/obo/ro.owl/")
    IAO = Namespace("http://purl.obolibrary.org/obo/iao.owl/")
    TIME = Namespace("http://www.w3.org/2006/time#")
    SIO = Namespace("http://semanticscience.org/ontology/sio/v1.59/sio-release.owl#")

query_endpoint = SPARQLWrapper(settings.query_endpoint)
query_endpoint.setReturnFormat(JSON)
# Set timeout to 300 seconds (5 minutes) for large queries
query_endpoint.setTimeout(300)

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

# def init_graph(default_graph: str | None = None) -> Dataset:
#     """Initialize a new RDF graph for nquads with the iCARE4CVD namespace bindings."""
#     g = Dataset(store="Oxigraph", default_graph_base=default_graph)
#     g.bind("icare", ICARE)
#     g.bind("rdf", RDF)
#     g.bind("rdfs", RDFS)
#     return g

def normalize_text(text: str) -> str:
    if text is None or text == "nan" or text == "":
        return None
    text = str(text).lower().strip().replace(" ", "_").replace("/", "_").replace(":", "_").replace('[','').replace(']','')
    return urllib.parse.quote(text, safe='_-')

def init_graph(default_graph_identifier: str | None = "https://w3id.org/CMEO/graph/studies_metadata") -> Dataset:
    """Initialize a new RDF graph for nquads with the voc namespace bindings."""
    g = Dataset(store="Oxigraph")
    g.bind("cmeo", OntologyNamespaces.CMEO.value)
    g.bind("bfo", OntologyNamespaces.BFO.value)
    g.bind("obi", OntologyNamespaces.OBI.value)
    g.bind("stato", OntologyNamespaces.STATO.value)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    # g.bind("omop", OMOP)
    g.bind("dc", DC)
    # g.bind("snomed", SNOMED)
    # do we need individual bindings for each ontology?
         # g.bind("snomed", SNOMED)
        # g.bind("loinc", LOINC)
        # g.bind("atc", ATC)
        # g.bind("rxnorm", RXNORM)
        # g.bind("ucum", UCUM)
        # g.bind("mesh", MESH)
        # g.bind("omop_ext", OMOP_EXT)
    g.graph(identifier=URIRef(default_graph_identifier))
    return g


def run_query(query: str) -> dict[str, Any]:
    """Function to run a SPARQL query against a remote endpoint"""
    query_endpoint.setQuery(query)
    return query_endpoint.query().convert()


def get_cohorts_metadata_query() -> str:
    """Get SPARQL query for retrieving cohorts metadata using new CMEO-based structure."""
    query = f"""
PREFIX cmeo: <https://w3id.org/CMEO/>
PREFIX obi: <http://purl.obolibrary.org/obo/obi.owl/>
PREFIX bfo: <http://purl.obolibrary.org/obo/bfo.owl/>
PREFIX ro: <http://purl.obolibrary.org/obo/ro.owl/>
PREFIX iao: <http://purl.obolibrary.org/obo/iao.owl/>
PREFIX sio: <http://semanticscience.org/ontology/sio/v1.59/sio-release.owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX icare: <{ICARE}>

SELECT DISTINCT ?cohortId ?cohortInstitution ?cohortEmail ?study_type ?study_participants ?study_duration ?study_ongoing ?study_population ?study_objective ?primary_outcome_spec ?secondary_outcome_spec ?morbidity ?study_start ?study_end ?male_percentage ?female_percentage ?administrator ?administrator_email ?study_contact_person ?study_contact_person_email ?references ?population_location ?language ?data_collection_frequency ?interventions ?sex_inclusion ?health_status_inclusion ?clinically_relevant_exposure_inclusion ?age_group_inclusion ?bmi_range_inclusion ?ethnicity_inclusion ?family_status_inclusion ?hospital_patient_inclusion ?use_of_medication_inclusion ?health_status_exclusion ?bmi_range_exclusion ?limited_life_expectancy_exclusion ?need_for_surgery_exclusion ?surgical_procedure_history_exclusion ?clinically_relevant_exposure_exclusion
    ?variable ?varName ?varLabel ?varType ?index ?count ?na ?max ?min ?units ?formula ?definition
    ?omopDomain ?conceptId ?conceptCode ?conceptName ?omopId ?mappedId ?mappedLabel ?visits ?categoryValue ?categoryLabel ?categoryConceptId ?categoryMappedId ?categoryMappedLabel
WHERE {{
    GRAPH <https://w3id.org/CMEO/graph/studies_metadata> {{
        ?study_design_execution a obi:study_design_execution ;
            dc:identifier ?cohortId .
        
        # Get institute/organization information  
        OPTIONAL {{
            ?study_design_execution ro:has_participant ?institute_entity .
            ?institute_entity a obi:organization ;
                cmeo:has_value ?cohortInstitution .
        }}
        
        # Get study type from descriptor
        OPTIONAL {{
            ?study_design_execution ro:concretizes ?study_design .
            ?study_design sio:is_described_by ?study_type_entity .
            ?study_type_entity a sio:study_descriptor ;
                cmeo:has_value ?study_type .
        }}
        
        # Get study objective
        OPTIONAL {{
            ?study_design_execution ro:concretizes ?study_design .
            ?study_design ro:has_part ?protocol .
            ?protocol ro:has_part ?study_objective_entity .
            ?study_objective_entity a obi:objective_specification ;
                cmeo:has_value ?study_objective .
        }}
        
        # Get number of participants
        OPTIONAL {{
            ?protocol ro:has_part ?number_of_participants_entity .
            ?number_of_participants_entity a cmeo:number_of_participants ;
                cmeo:has_value ?study_participants .
        }}
        
        # Get timing information
        OPTIONAL {{
            ?study_design_execution iao:has_time_stamp ?start_date_entity .
            ?start_date_entity a cmeo:start_time ;
                cmeo:has_value ?study_start .
        }}
        
        OPTIONAL {{
            ?study_design_execution iao:has_time_stamp ?end_date_entity .
            ?end_date_entity a cmeo:end_time ;
                cmeo:has_value ?study_end .
        }}
        
        OPTIONAL {{
            ?study_design_execution ro:has_characteristic ?ongoing_entity .
            ?ongoing_entity a cmeo:ongoing ;
                cmeo:has_value ?study_ongoing .
        }}
        
        # Get outcome specifications (direct from protocol, not nested)
        OPTIONAL {{
            ?protocol ro:has_part ?primary_outcome_spec_entity .
            ?primary_outcome_spec_entity a cmeo:primary_outcome_specification ;
                cmeo:has_value ?primary_outcome_spec .
        }}
        
        OPTIONAL {{
            ?protocol ro:has_part ?secondary_outcome_spec_entity .
            ?secondary_outcome_spec_entity a cmeo:secondary_outcome_specification ;
                cmeo:has_value ?secondary_outcome_spec .
        }}
        
        # Get morbidity information
        OPTIONAL {{
            ?protocol ro:has_part ?morbidity_entity .
            ?morbidity_entity a obi:morbidity ;
                cmeo:has_value ?morbidity .
        }}
        
        # Get eligibility criteria and demographics
        OPTIONAL {{
            ?protocol ro:has_part ?eligibility .
            ?eligibility a obi:eligibility_criterion .
            
            # Male/female percentages
            OPTIONAL {{
                ?eligibility ro:has_part ?male_percentage_entity .
                ?male_percentage_entity a cmeo:male_percentage ;
                    cmeo:has_value ?male_percentage .
            }}
            
            OPTIONAL {{
                ?eligibility ro:has_part ?female_percentage_entity .
                ?female_percentage_entity a cmeo:female_percentage ;
                    cmeo:has_value ?female_percentage .
            }}
        }}
        
        # Get age distribution
        OPTIONAL {{
            ?study_design_execution ro:has_characteristic ?age_distribution_entity .
            ?age_distribution_entity a obi:age_distribution ;
                cmeo:has_value ?age_distribution .
        }}
        
        # Get population location
        OPTIONAL {{
            ?study_design_execution ro:has_characteristic ?population_location_entity .
            ?population_location_entity a bfo:site ;
                cmeo:has_value ?population_location .
        }}
        
        # Get language (direct property)
        OPTIONAL {{
            ?study_design_execution dc:language ?language .
        }}
        
        # Get contact information (simplified - direct participants)
        OPTIONAL {{
            ?study_design_execution ro:has_participant ?study_contact_person_entity .
            ?study_contact_person_entity a cmeo:homo_sapiens ;
                cmeo:has_value ?study_contact_person .
        }}
        
        OPTIONAL {{
            ?study_design_execution ro:has_participant ?administrator_entity .
            ?administrator_entity a cmeo:homo_sapiens ;
                cmeo:has_value ?administrator .
        }}
    }}
    
    OPTIONAL {{
        GRAPH ?cohortGraph {{
            ?cohort icare:hasVariable ?variable .
            ?variable dc:identifier ?varName ;
                rdfs:label ?varLabel ;
                icare:varType ?varType ;
                icare:index ?index .
            OPTIONAL {{ ?variable icare:count ?count . }}
            OPTIONAL {{ ?variable icare:na ?na . }}
            OPTIONAL {{ ?variable icare:max ?max . }}
            OPTIONAL {{ ?variable icare:min ?min . }}
            OPTIONAL {{ ?variable icare:units ?units . }}
            OPTIONAL {{ ?variable icare:formula ?formula . }}
            OPTIONAL {{ ?variable icare:definition ?definition . }}
            OPTIONAL {{ ?variable icare:omopDomain ?omopDomain . }}
            OPTIONAL {{ ?variable icare:conceptId ?conceptId . }}
            OPTIONAL {{ ?variable icare:conceptCode ?conceptCode . }}
            OPTIONAL {{ ?variable icare:conceptName ?conceptName . }}
            OPTIONAL {{ ?variable icare:omopId ?omopId . }}
            OPTIONAL {{ ?variable icare:mappedId ?mappedId . }}
            OPTIONAL {{ ?variable icare:mappedLabel ?mappedLabel . }}
            OPTIONAL {{ ?variable icare:visits ?visits . }}
            
            OPTIONAL {{
                ?variable icare:categories ?category .
                ?category rdf:value ?categoryValue ;
                    rdfs:label ?categoryLabel .
                OPTIONAL {{ ?category icare:conceptId ?categoryConceptId . }}
                OPTIONAL {{ ?category icare:mappedId ?categoryMappedId . }}
                OPTIONAL {{ ?category icare:mappedLabel ?categoryMappedLabel . }}
            }}
        }}
    }}
}}
ORDER BY ?cohortId ?index ?categoryValue"""
    return query


# TODO: Utility to get value or None if key is missing or value is empty string
def get_value(key: str, row: dict[str, Any]) -> str:
    """Safely get a value from a SPARQL result row.
    
    Args:
        key: The key to look for in the row
        row: The SPARQL result row dictionary
        
    Returns:
        The string value if found, empty string otherwise
    """
    try:
        if key in row and row[key]["value"]:
            return str(row[key]["value"])
        return ""
    except (KeyError, TypeError):
        return ""


def get_int_value(key: str, row: dict[str, Any]) -> int | None:
    """Safely get an integer value from a SPARQL result row.
    
    Args:
        key: The key to look for in the row
        row: The SPARQL result row dictionary
        
    Returns:
        The integer value if found and valid, None otherwise
    """
    try:
        if key in row and row[key]["value"]:
            return int(row[key]["value"])
        return None
    except (KeyError, TypeError, ValueError):
        return None


def get_bool_value(key: str, row: dict[str, Any]) -> bool:
    """Safely get a boolean value from a SPARQL result row.
    
    Args:
        key: The key to look for in the row
        row: The SPARQL result row dictionary
        
    Returns:
        True if the value is 'true' (case insensitive), False otherwise
    """
    try:
        if key in row and row[key]["value"]:
            return str(row[key]["value"]).lower() == "true"
        return False
    except (KeyError, TypeError):
        return False


def get_curie_value(key: str, row: dict[str, Any]) -> str | None:
    """Safely get a CURIE value from a SPARQL result row.
    
    Args:
        key: The key to look for in the row
        row: The SPARQL result row dictionary
        
    Returns:
        The compressed CURIE value if found and valid, None otherwise
    """
    try:
        value = get_value(key, row)
        if value:
            return curie_converter.compress(value)
        return None
    except Exception:
        return None


def retrieve_cohorts_metadata(user_email: str, include_sparql_metadata: bool = False) -> dict[str, Cohort] | dict:
    """Get all cohorts metadata from the SPARQL endpoint (infos, variables)
    
    Args:
        user_email: Email of the user requesting the data
        include_sparql_metadata: If True, returns dict with cohorts and metadata
    """
    import time
    start_time = time.time()
    
    # Execute SPARQL query and measure its execution time
    query_start_time = time.time()
    results = run_query(get_cohorts_metadata_query())["results"]["bindings"]
    query_end_time = time.time()
    query_duration = query_end_time - query_start_time
    
    cohorts_with_variables = {}
    cohorts_without_variables = {}
    logging.info(f"Get cohorts metadata query execution time: {query_duration:.2f} seconds, results: {len(results)})")
    for row in results:
        try:
            cohort_id = str(row["cohortId"]["value"])
            var_id = str(row["varName"]["value"]) if "varName" in row else None
            # Determine which dictionary to use
            target_dict = cohorts_with_variables if var_id else cohorts_without_variables

            # Initialize cohort data structure if not exists
            if cohort_id and cohort_id not in target_dict:
                cohort = Cohort(
                    cohort_id=get_value("cohortId", row),
                    institution=get_value("cohortInstitution", row),
                    study_type=get_value("study_type", row),
                    study_participants=get_value("study_participants", row),
                    study_duration=get_value("study_duration", row),
                    study_ongoing=get_value("study_ongoing", row),
                    study_population=get_value("study_population", row),
                    study_objective=get_value("study_objective", row),
                    primary_outcome_spec=get_value("primary_outcome_spec", row),
                    secondary_outcome_spec=get_value("secondary_outcome_spec", row),
                    morbidity=get_value("morbidity", row),
                    study_start=get_value("study_start", row),
                    study_end=get_value("study_end", row),
                    male_percentage=float(get_value("male_percentage", row)) if get_value("male_percentage", row) else None,
                    female_percentage=float(get_value("female_percentage", row)) if get_value("female_percentage", row) else None,
                    # Contact information fields
                    administrator=get_value("administrator", row),
                    administrator_email=get_value("administrator_email", row),
                    study_contact_person=get_value("study_contact_person", row),
                    study_contact_person_email=get_value("study_contact_person_email", row),
                    references=[],
                    # Additional metadata fields
                    population_location=get_value("population_location", row),
                    language=get_value("language", row),
                    data_collection_frequency=get_value("data_collection_frequency", row),
                    interventions=get_value("interventions", row),
                    # Inclusion criteria fields
                    sex_inclusion=get_value("sex_inclusion", row),
                    health_status_inclusion=get_value("health_status_inclusion", row),
                    clinically_relevant_exposure_inclusion=get_value("clinically_relevant_exposure_inclusion", row),
                    age_group_inclusion=get_value("age_group_inclusion", row),
                    bmi_range_inclusion=get_value("bmi_range_inclusion", row),
                    ethnicity_inclusion=get_value("ethnicity_inclusion", row),
                    family_status_inclusion=get_value("family_status_inclusion", row),
                    hospital_patient_inclusion=get_value("hospital_patient_inclusion", row),
                    use_of_medication_inclusion=get_value("use_of_medication_inclusion", row),
                    # Exclusion criteria fields
                    health_status_exclusion=get_value("health_status_exclusion", row),
                    bmi_range_exclusion=get_value("bmi_range_exclusion", row),
                    limited_life_expectancy_exclusion=get_value("limited_life_expectancy_exclusion", row),
                    need_for_surgery_exclusion=get_value("need_for_surgery_exclusion", row),
                    surgical_procedure_history_exclusion=get_value("surgical_procedure_history_exclusion", row),
                    clinically_relevant_exposure_exclusion=get_value("clinically_relevant_exposure_exclusion", row),
                    variables={},
                    can_edit=user_email in [*settings.admins_list, get_value("cohortEmail", row)],
                    physical_dictionary_exists=False
                )
                target_dict[cohort_id] = cohort
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
            
            # Handle references - process independently of other conditions
            if get_value("references", row) and get_value("references", row) not in target_dict[cohort_id].references:
                # Add reference to the list
                target_dict[cohort_id].references.append(get_value("references", row))

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
                    concept_code=get_value("conceptCode", row),
                    concept_name=get_value("conceptName", row),
                    omop_id=get_value("omopId", row),
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
    # Merge cohorts with and without variables
    # Put cohorts with variables first so they appear at the top of the list
    cohorts = {**cohorts_with_variables, **cohorts_without_variables}
    
    # Log total function execution time
    end_time = time.time()
    total_duration = end_time - start_time
    processing_duration = total_duration - query_duration
    logging.info(f"Total cohorts metadata retrieval time: {total_duration:.2f} seconds (Query: {query_duration:.2f}s, Processing: {processing_duration:.2f}s)")
    
    # Return with metadata if requested
    if include_sparql_metadata:
        return {
            "cohorts": cohorts,
            "sparql_metadata": {
                "row_count": len(results),
                "query_duration_ms": round(query_duration * 1000),
                "processing_duration_ms": round(processing_duration * 1000),
                "total_duration_ms": round(total_duration * 1000)
            }
        }
    
    return cohorts
