import logging
import re
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
    SIO = Namespace("http://semanticscience.org/ontology/sio.owl/")
    NCBI = Namespace("http://purl.bioontology.org/ontology/NCBITAXON/")

query_endpoint = SPARQLWrapper(settings.query_endpoint)
query_endpoint.setReturnFormat(JSON)
# Set timeout to 300 seconds (5 minutes) for large queries
query_endpoint.setTimeout(300)
# Use POST method for large queries (avoids header size limits)
query_endpoint.setMethod('POST')
# Enable HTTP connection keep-alive for better performance
query_endpoint.addCustomHttpHeader("Connection", "keep-alive")

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


def extract_age_range(text):
    """
    Extract minimum and maximum age values from text descriptions.
    Matches patterns like: "age >= 18", "between 18 and 65 years", etc.
    Returns tuple (min_age, max_age) or None if no age range found.
    """
    # Normalize Unicode comparison symbols
    text = text.strip().replace("≥", ">=").replace("≤", "<=")

    # Patterns for extracting min and max age
    age_conditions = re.findall(r'(?:age\s*)?(>=|<=|>|<)\s*(\d+(?:\.\d+)?)\s*(?:years\s*old|years)?', text, flags=re.IGNORECASE)

    min_age = None
    max_age = None

    for operator, value in age_conditions:
        value = float(value)
        if operator in ('>=', '>'):
            if min_age is None or value > min_age:
                min_age = value if operator == '>' else value  # can adjust to value + epsilon if needed
        elif operator in ('<=', '<'):
            if max_age is None or value < max_age:
                max_age = value if operator == '<' else value  # can adjust to value - epsilon if needed

    # Also handle "between X and Y years" separately
    match = re.search(r'between\s+(\d+(?:\.\d+)?)\s*(?:and|[-–])\s*(\d+(?:\.\d+)?)\s*years?', text, flags=re.IGNORECASE)
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        if min_age is None or min_val > min_age:
            min_age = min_val
        if max_age is None or max_val < max_age:
            max_age = max_val

    if min_age is not None or max_age is not None:
        return min_age, max_age

    return None


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
    import time

    start = time.time()
    query_endpoint.setQuery(query)

    # We already configured HTTP method as POST globally; SPARQLWrapper
    # will detect SELECT vs CONSTRUCT from the query text.
    result = query_endpoint.query().convert()
    duration = time.time() - start
    if duration > 1.0:  # Log queries taking more than 1 second
        logging.warning(f"[TIMING] SPARQL query took {duration:.2f}s")
    return result


def get_cohorts_metadata_query() -> str:
    """DEPRECATED: This old combined query has been replaced by two separate queries.
    Use get_studies_metadata_query() and get_variables_metadata_query() instead.
    This function is kept for backwards compatibility but may be removed in future versions."""
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


def get_studies_metadata_query() -> str:
    """Get SPARQL query for retrieving studies/cohorts metadata (from sparql_queries.txt)."""
    # Load the first query from CohortVarLinker/queries/sparql_queries.txt (lines 3-290)
    import os
    query_file = os.path.join(os.path.dirname(__file__), '..', 'CohortVarLinker', 'queries', 'sparql_queries.txt')
    with open(query_file, 'r') as f:
        lines = f.readlines()
        # Extract lines 3-290 (study metadata query including final GROUP BY)
        # Lines 1-2 are comments, so we skip them
        query = ''.join(lines[2:291])  # 0-indexed, so lines[2:291] = lines 3-290 (inclusive)
    return query


def get_variables_metadata_query() -> str:
    """Get SPARQL query for retrieving variables metadata (from sparql_queries.txt).
    
    The query is extracted from the second query in the file (lines 294-418).
    Note: Line numbers are 1-indexed in the file, but 0-indexed in the list.
    """
    import os
    query_file = os.path.join(os.path.dirname(__file__), '..', 'CohortVarLinker', 'queries', 'sparql_queries.txt')
    
    with open(query_file, 'r') as f:
        lines = f.readlines()
    
    # Extract only the SPARQL lines for Query 2 (prefixes + SELECT + WHERE)
    # Line 294 (index 293): comment "Query 2: All variables from each study graph"
    # Line 295 (index 294): first PREFIX (stato)
    # Line 419 (index 418): closing brace of Query 2
    # Skip the comment line, start from first PREFIX
    query = ''.join(lines[294:419]).strip()
    return query


def retrieve_cohorts_metadata(user_email: str, include_sparql_metadata: bool = False) -> dict[str, Cohort] | dict:
    """Get all cohorts metadata from the SPARQL endpoint using two separate queries.
    
    This new implementation uses two separate queries instead of one large combined query:
    1. First query: Retrieves study/cohort metadata from studies_metadata graph
    2. Second query: Retrieves variables metadata from individual study graphs
    3. Merges results to create complete Cohort objects
    
    Args:
        user_email: Email of the user requesting the data
        include_sparql_metadata: If True, returns dict with cohorts and metadata
    """
    import time
    start_time = time.time()
    
    # ===== STEP 1: Execute Studies Metadata Query =====
    logging.info("🔍 Starting studies metadata query...")
    studies_query_start = time.time()
    studies_results = run_query(get_studies_metadata_query())["results"]["bindings"]
    studies_query_duration = time.time() - studies_query_start
    logging.info(f"✅ Studies metadata query completed: {studies_query_duration:.2f}s, {len(studies_results)} results")
    
    # ===== STEP 2: Execute Variables Metadata Query =====
    logging.info("🔍 Starting variables metadata query...")
    variables_query_start = time.time()
    variables_results = run_query(get_variables_metadata_query())["results"]["bindings"]
    variables_query_duration = time.time() - variables_query_start
    logging.info(f"✅ Variables metadata query completed: {variables_query_duration:.2f}s, {len(variables_results)} results")
    
    total_query_duration = studies_query_duration + variables_query_duration
    
    # ===== STEP 3: Process Studies Metadata =====
    logging.info(f"📊 Processing {len(studies_results)} study metadata rows...")
    cohorts = {}
    
    for row in studies_results:
        try:
            cohort_id = str(row["cohortId"]["value"])
            
            if cohort_id not in cohorts:
                cohort = Cohort(
                    cohort_id=get_value("cohortId", row),
                    institution=get_value("cohortInstitution", row),
                    study_type=get_value("study_type", row),
                    study_participants=get_value("study_participants", row),
                    study_duration=get_value("duration", row),  # Note: new query uses "duration" not "study_duration"
                    study_ongoing=get_value("study_ongoing", row),
                    study_population=get_value("study_population", row),  # May not be in new query
                    study_objective=get_value("study_objective", row),
                    primary_outcome_spec=get_value("primary_outcome_spec", row),
                    secondary_outcome_spec=get_value("secondary_outcome_spec", row),
                    morbidity=get_value("morbidity", row),
                    study_start=get_value("study_start", row),
                    study_end=get_value("study_end", row),
                    male_percentage=None,  # Not in new query structure
                    female_percentage=None,  # Not in new query structure
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
                    # Inclusion/Exclusion criteria - new query structure
                    sex_inclusion=get_value("inclusion_labels", row),  # Aggregated in new query
                    health_status_inclusion=get_value("inclusion_values", row),  # Aggregated
                    clinically_relevant_exposure_inclusion="",
                    age_group_inclusion="",
                    bmi_range_inclusion="",
                    ethnicity_inclusion="",
                    family_status_inclusion="",
                    hospital_patient_inclusion="",
                    use_of_medication_inclusion="",
                    # Exclusion criteria fields
                    health_status_exclusion=get_value("exclusion_labels", row),  # Aggregated
                    bmi_range_exclusion=get_value("exclusion_values", row),  # Aggregated
                    limited_life_expectancy_exclusion="",
                    need_for_surgery_exclusion="",
                    surgical_procedure_history_exclusion="",
                    clinically_relevant_exposure_exclusion="",
                    variables={},
                    can_edit=user_email in settings.admins_list,
                    physical_dictionary_exists=False
                )
                cohorts[cohort_id] = cohort
                
                # Check if physical dictionary file exists
                try:
                    if cohorts[cohort_id].metadata_filepath:
                        cohorts[cohort_id].physical_dictionary_exists = True
                except FileNotFoundError:
                    cohorts[cohort_id].physical_dictionary_exists = False
                    
        except Exception as e:
            logging.warning(f"Error processing study row: {e}")
    
    logging.info(f"✅ Studies processing completed: {len(cohorts)} cohorts created")
    
    # ===== STEP 4: Process Variables Metadata =====
    logging.info(f"📊 Processing {len(variables_results)} variable metadata rows...")
    
    # Create a mapping for faster lookup (case-insensitive)
    cohort_id_map = {cid.lower(): cid for cid in cohorts.keys()}
    
    variables_processed = 0
    for row in variables_results:
        try:
            # Extract study name from the query result
            study_name = get_value("study_name", row)
            var_name = get_value("var_name", row)
            
            # Fast lookup using pre-built map
            cohort_id = cohort_id_map.get(study_name.lower())
            
            if not cohort_id:
                # Try partial match as fallback
                for lower_cid, actual_cid in cohort_id_map.items():
                    if study_name.lower() in lower_cid or lower_cid in study_name.lower():
                        cohort_id = actual_cid
                        break
            
            if not cohort_id:
                # If we can't find a match, use study_name as cohort_id
                cohort_id = study_name
            if cohort_id not in cohorts:
                logging.warning(f"Variable for unknown cohort: {cohort_id}")
                continue
            
            # Add variable to cohort
            if var_name and var_name not in cohorts[cohort_id].variables:
                # Parse categorical values (format: "value1=label1|value2=label2")
                categories = []
                categorical_values_str = get_value("categorical_values", row)
                if categorical_values_str:
                    pairs = categorical_values_str.split("|")
                    category_codes_list = get_value("category_concept_codes", row).split("|") if get_value("category_concept_codes", row) else []
                    category_labels_list = get_value("category_concept_label", row).split("|") if get_value("category_concept_label", row) else []
                    category_omop_list = get_value("category_omop_id", row).split("|") if get_value("category_omop_id", row) else []
                    
                    for i, pair in enumerate(pairs):
                        if "=" in pair:
                            value, label = pair.split("=", 1)
                            categories.append(VariableCategory(
                                value=value.strip(),
                                label=label.strip(),
                                concept_id=category_codes_list[i] if i < len(category_codes_list) else None,
                                mapped_id=None,
                                mapped_label=category_labels_list[i] if i < len(category_labels_list) else None,
                            ))
                
                # Parse concept codes and labels (format: "code1 || code2 || code3")
                concept_codes_str = get_value("concept_codes", row)
                concept_labels_str = get_value("concept_labels", row)
                concept_codes_list = concept_codes_str.split(" || ") if concept_codes_str else []
                concept_labels_list = concept_labels_str.split(" || ") if concept_labels_str else []
                
                cohorts[cohort_id].variables[var_name] = CohortVariable(
                    var_name=var_name,
                    var_label=get_value("var_label", row),
                    var_type=get_value("varType", row),
                    count=get_int_value("count_value", row) or 0,
                    max=get_value("maximum_value", row),
                    min=get_value("minimum_value", row),
                    units=get_value("unit_value", row),
                    visits=get_value("visit", row),
                    formula=get_value("formula_value", row),
                    definition="",  # Not in new query
                    concept_id=concept_codes_list[0] if concept_codes_list else None,
                    concept_code=concept_codes_list[0] if concept_codes_list else None,
                    concept_name=concept_labels_list[0] if concept_labels_list else None,
                    omop_id=get_value("concept_omop_id", row),
                    mapped_id=None,  # Not clear in new query
                    mapped_label=None,  # Not clear in new query
                    omop_domain=get_value("omopDomain", row),
                    index=None,  # Not in new query
                    na=0,  # Not in new query
                    categories=categories
                )
                variables_processed += 1
                
        except Exception as e:
            logging.warning(f"Error processing variable row: {e}")
    
    logging.info(f"✅ Variables processing completed: {variables_processed} variables added")
    
    # ===== STEP 5: Separate cohorts with and without variables =====
    cohorts_with_variables = {k: v for k, v in cohorts.items() if v.variables}
    cohorts_without_variables = {k: v for k, v in cohorts.items() if not v.variables}
    
    # Merge with variables first (so they appear at top)
    cohorts = {**cohorts_with_variables, **cohorts_without_variables}
    
    # Log timing information
    end_time = time.time()
    total_duration = end_time - start_time
    processing_duration = total_duration - total_query_duration
    logging.info(f"Total retrieval time: {total_duration:.2f}s (Queries: {total_query_duration:.2f}s, Processing: {processing_duration:.2f}s)")
    
    # Return with metadata if requested
    if include_sparql_metadata:
        return {
            "cohorts": cohorts,
            "sparql_metadata": {
                "row_count": len(studies_results) + len(variables_results),
                "studies_count": len(studies_results),
                "variables_count": len(variables_results),
                "query_duration_ms": round(total_query_duration * 1000),
                "processing_duration_ms": round(processing_duration * 1000),
                "total_duration_ms": round(total_duration * 1000)
            }
        }
    
    return cohorts

