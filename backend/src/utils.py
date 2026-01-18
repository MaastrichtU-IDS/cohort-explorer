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
# Set timeout to 600 seconds (10 minutes) for large queries
query_endpoint.setTimeout(600)
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
        "prefix_synonyms": ["icd9proc"]
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
        "prefix_synonyms": ["OMOP", "omopgenomic", "rxnorm extension", "rxnorm_extension", "RXNORM EXTENSION", "RxNorm Extension", "rxnormextension" ]
    },
    {
        "prefix": "ucum",
        "uri_prefix": "http://unitsofmeasure.org/ucum/",
        "prefix_synonyms": ["UCUM"]
    },
    {
        "prefix":"ukbiobank",
        "uri_prefix": "https://biobank.ndph.ox.ac.uk/showcase/",
        

    },
    {
        "prefix":"mesh",
        "uri_prefix": "https://id.nlm.nih.gov/mesh/"
    },
    {
        "prefix": "cpt4",
        "uri_prefix": "https://www.ama-assn.org/practice-management/cpt/",
        "prefix_synonyms": ["CPT4", "cpt", "CPT"]
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
    query = """
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
    ?omopDomain ?conceptId ?conceptCode ?conceptName ?omopId ?mappedId ?mappedLabel ?visits ?visitConceptName ?categoryValue ?categoryLabel ?categoryConceptId ?categoryMappedId ?categoryMappedLabel
WHERE {
    GRAPH ?cohortMetadatXaGraph {
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
        OPTIONAL { ?cohort icare:morbidity ?morbidity . }
        OPTIONAL { ?cohort icare:studyStart ?study_start . }
        OPTIONAL { ?cohort icare:studyEnd ?study_end . }
        OPTIONAL { ?cohort icare:malePercentage ?male_percentage . }
        OPTIONAL { ?cohort icare:femalePercentage ?female_percentage . }
        
        # Get institute/organization information  
        OPTIONAL {
            ?study_design_execution ro:has_participant ?institute_entity .
            ?institute_entity a obi:organization ;
                cmeo:has_value ?cohortInstitution .
        }
        
        # Get study type from descriptor
        OPTIONAL {
            ?study_design_execution ro:concretizes ?study_design .
            ?study_design sio:is_described_by ?study_type_entity .
            ?study_type_entity a sio:study_descriptor ;
                cmeo:has_value ?study_type .
        }
        
        # Get study objective
        OPTIONAL {
            ?study_design_execution ro:concretizes ?study_design .
            ?study_design ro:has_part ?protocol .
            ?protocol ro:has_part ?study_objective_entity .
            ?study_objective_entity a obi:objective_specification ;
                cmeo:has_value ?study_objective .
        }
        
        # Get number of participants
        OPTIONAL {
            ?protocol ro:has_part ?number_of_participants_entity .
            ?number_of_participants_entity a cmeo:number_of_participants ;
                cmeo:has_value ?study_participants .
        }
        
        # Get timing information
        OPTIONAL {
            ?study_design_execution iao:has_time_stamp ?start_date_entity .
            ?start_date_entity a cmeo:start_time ;
                cmeo:has_value ?study_start .
        }
        
        OPTIONAL {
            ?study_design_execution iao:has_time_stamp ?end_date_entity .
            ?end_date_entity a cmeo:end_time ;
                cmeo:has_value ?study_end .
        }
        
        OPTIONAL {
            ?study_design_execution ro:has_characteristic ?ongoing_entity .
            ?ongoing_entity a cmeo:ongoing ;
                cmeo:has_value ?study_ongoing .
        }
        
        # Get outcome specifications (direct from protocol, not nested)
        OPTIONAL {
            ?protocol ro:has_part ?primary_outcome_spec_entity .
            ?primary_outcome_spec_entity a cmeo:primary_outcome_specification ;
                cmeo:has_value ?primary_outcome_spec .
        }
        
        OPTIONAL {
            ?protocol ro:has_part ?secondary_outcome_spec_entity .
            ?secondary_outcome_spec_entity a cmeo:secondary_outcome_specification ;
                cmeo:has_value ?secondary_outcome_spec .
        }
        
        # Get morbidity information
        OPTIONAL {
            ?protocol ro:has_part ?morbidity_entity .
            ?morbidity_entity a obi:morbidity ;
                cmeo:has_value ?morbidity .
        }
        
        # Get eligibility criteria and demographics
        OPTIONAL {
            ?protocol ro:has_part ?eligibility .
            ?eligibility a obi:eligibility_criterion .
            
            # Male/female percentages
            OPTIONAL {
                ?eligibility ro:has_part ?male_percentage_entity .
                ?male_percentage_entity a cmeo:male_percentage ;
                    cmeo:has_value ?male_percentage .
            }
            
            OPTIONAL {
                ?eligibility ro:has_part ?female_percentage_entity .
                ?female_percentage_entity a cmeo:female_percentage ;
                    cmeo:has_value ?female_percentage .
            }
        }
        
        # Get age distribution
        OPTIONAL {
            ?study_design_execution ro:has_characteristic ?age_distribution_entity .
            ?age_distribution_entity a obi:age_distribution ;
                cmeo:has_value ?age_distribution .
        }
        
        # Get population location
        OPTIONAL {
            ?study_design_execution ro:has_characteristic ?population_location_entity .
            ?population_location_entity a bfo:site ;
                cmeo:has_value ?population_location .
        }
        
        # Get language (direct property)
        OPTIONAL {
            ?study_design_execution dc:language ?language .
        }
        
        # Get contact information (simplified - direct participants)
        OPTIONAL {
            ?study_design_execution ro:has_participant ?study_contact_person_entity .
            ?study_contact_person_entity a cmeo:homo_sapiens ;
                cmeo:has_value ?study_contact_person .
        }
        
        OPTIONAL {
            ?study_design_execution ro:has_participant ?administrator_entity .
            ?administrator_entity a cmeo:homo_sapiens ;
                cmeo:has_value ?administrator .
        }
    }
    
    OPTIONAL {
        GRAPH ?cohortGraph {
            ?cohort icare:hasVariable ?variable .
            ?variable dc:identifier ?varName ;
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
            OPTIONAL { ?variable icare:conceptCode ?conceptCode }
            OPTIONAL { ?variable icare:conceptName ?conceptName }
            OPTIONAL { ?variable icare:omopId ?omopId }
            OPTIONAL { ?variable icare:domain ?omopDomain }
            OPTIONAL { ?variable icare:visits ?visits }
            OPTIONAL { ?variable icare:visitConceptName ?visitConceptName }
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
        }
    }
} ORDER BY ?cohort ?index
""".format(ICARE=str(ICARE))
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


def get_cohorts_metadata_query() -> str:
    """Get SPARQL query for retrieving cohort metadata from studies_metadata graph (Query 1).
    
    The query is extracted from the first query in the file (lines 1-293).
    Note: Line numbers are 1-indexed in the file, but 0-indexed in the list.
    """
    import os
    query_file = os.path.join(os.path.dirname(__file__), '..', 'CohortVarLinker', 'queries', 'sparql_queries.txt')
    
    with open(query_file, 'r') as f:
        lines = f.readlines()
    
    # Extract Query 1 (cohort metadata from studies_metadata graph)
    # Line 1 (index 0): comment "# Query to Retrieve Studies Metadata"
    # Line 2 (index 1): first PREFIX
    # Line 293 (index 292): GROUP BY ?study ?cohortId ?study_type (last line of Query 1)
    # Skip the comment line, start from first PREFIX
    query = ''.join(lines[1:293]).strip()
    return query


def get_variables_metadata_query() -> str:
    """Get SPARQL query for retrieving variables metadata (from sparql_queries.txt).
    
    The query is extracted from the second query in the file (lines 296+).
    Note: Line numbers are 1-indexed in the file, but 0-indexed in the list.
    """
    import os
    query_file = os.path.join(os.path.dirname(__file__), '..', 'CohortVarLinker', 'queries', 'sparql_queries.txt')
    
    with open(query_file, 'r') as f:
        lines = f.readlines()
    
    # Extract only the SPARQL lines for Query 2 (prefixes + SELECT + WHERE)
    # Line 296 (index 295): comment "Query 2: All variables from each study graph"
    # Line 297 (index 296): first PREFIX (stato)
    # Skip the comment line, start from first PREFIX, go to end of file
    query = ''.join(lines[296:]).strip()
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
    
    # ========== STEP 1: Get cohort metadata from studies_metadata graph ==========
    logging.info("Step 1: Executing SPARQL query to retrieve cohort metadata from studies_metadata graph...")
    metadata_query_start = time.time()
    
    try:
        metadata_results = run_query(get_cohorts_metadata_query())["results"]["bindings"]
    except Exception as e:
        logging.error(f"Cohort metadata query failed: {e}")
        raise
    
    metadata_query_end = time.time()
    metadata_duration = metadata_query_end - metadata_query_start
    logging.info(f"Cohort metadata query completed in {metadata_duration:.2f}s, returned {len(metadata_results)} cohorts")
    
    # Create cohorts dictionary with metadata
    cohorts_with_variables = {}
    cohorts_without_variables = {}
    
    for row in metadata_results:
        cohort_id = get_value("cohortId", row)
        if not cohort_id:
            continue
            
        cohort = Cohort(
            cohort_id=cohort_id,
            institution=get_value("cohortInstitution", row),
            study_type=get_value("study_type", row),
            study_participants=get_value("study_participants", row),
            study_duration=get_value("duration", row),
            study_ongoing=get_value("study_ongoing", row),
            study_population=get_value("study_population", row),
            study_objective=get_value("study_objective", row),
            primary_outcome_spec=get_value("primary_outcome_spec", row),
            secondary_outcome_spec=get_value("secondary_outcome_spec", row),
            morbidity=get_value("morbidity", row),
            study_start=get_value("study_start", row),
            study_end=get_value("study_end", row),
            male_percentage=None,
            female_percentage=None,
            administrator=get_value("administrator", row),
            administrator_email=get_value("administrator_email", row).lower() if get_value("administrator_email", row) else "",
            study_contact_person=get_value("study_contact_person", row),
            study_contact_person_email=get_value("study_contact_person_email", row).lower() if get_value("study_contact_person_email", row) else "",
            references=[],
            population_location=get_value("population_location", row),
            language=get_value("language", row),
            data_collection_frequency=get_value("data_collection_frequency", row),
            interventions=get_value("interventions", row),
            sex_inclusion=get_value("inclusion_labels", row),
            health_status_inclusion=get_value("inclusion_values", row),
            clinically_relevant_exposure_inclusion="",
            age_group_inclusion="",
            bmi_range_inclusion="",
            ethnicity_inclusion="",
            family_status_inclusion="",
            hospital_patient_inclusion="",
            use_of_medication_inclusion="",
            health_status_exclusion=get_value("exclusion_labels", row),
            bmi_range_exclusion=get_value("exclusion_values", row),
            limited_life_expectancy_exclusion="",
            need_for_surgery_exclusion="",
            surgical_procedure_history_exclusion="",
            clinically_relevant_exposure_exclusion="",
            variables={},
            physical_dictionary_exists=False
        )
        
        # Store in cohorts_without_variables initially (will move to with_variables if variables found)
        cohorts_without_variables[cohort_id] = cohort
    
    logging.info(f"Created {len(cohorts_without_variables)} cohort objects with metadata")
    
    # ========== STEP 2: Get variables metadata from individual study graphs ==========
    logging.info("Step 2: Executing SPARQL query to retrieve variables metadata...")
    variables_query_start = time.time()
    
    try:
        results = run_query(get_variables_metadata_query())["results"]["bindings"]
    except Exception as e:
        logging.error(f"Variables query failed: {e}")
        raise
    
    query_end_time = time.time()
    query_duration = query_end_time - variables_query_start
    
    total_rows = len(results)
    logging.info(f"Variables query completed in {query_duration:.2f}s, returned {total_rows} rows. Starting processing...")
    
    # Track progress during row processing
    rows_processed = 0
    variables_processed = 0
    last_progress_log = time.time()
    progress_interval = 30  # Log progress every 30 seconds
    
    for row in results:
        rows_processed += 1
        
        # Log progress every 30 seconds
        current_time = time.time()
        if current_time - last_progress_log >= progress_interval:
            elapsed = current_time - query_end_time
            progress_pct = (rows_processed / total_rows * 100) if total_rows > 0 else 0
            logging.info(f"Processing progress: {rows_processed}/{total_rows} rows ({progress_pct:.1f}%) - {elapsed:.1f}s elapsed")
            last_progress_log = current_time
        try:
            # CMEO query returns 'study_name' and 'var_name' instead of 'cohortId' and 'varName'
            cohort_id = str(row["study_name"]["value"]) if "study_name" in row else None
            var_id = str(row["var_name"]["value"]) if "var_name" in row else None
            
            if not cohort_id:
                logging.warning(f"Skipping row with missing study_name: {list(row.keys())}")
                continue
            
            # Check if cohort exists (should exist from Query 1 metadata)
            if cohort_id not in cohorts_with_variables and cohort_id not in cohorts_without_variables:
                logging.warning(f"Cohort {cohort_id} found in variables query but not in metadata query, skipping")
                continue
            
            # Determine which dictionary the cohort is in
            if cohort_id in cohorts_with_variables:
                target_dict = cohorts_with_variables
            elif cohort_id in cohorts_without_variables and var_id:
                # Move cohort from without_variables to with_variables if we now have a variable
                target_dict = cohorts_with_variables
                cohorts_with_variables[cohort_id] = cohorts_without_variables.pop(cohort_id)
            elif cohort_id in cohorts_without_variables:
                target_dict = cohorts_without_variables
            else:
                continue  # Should not reach here

            # Process variables
            if "var_name" in row and var_id and var_id not in target_dict[cohort_id].variables:
                # Parse categories from CMEO query format
                categories = []
                categorical_values_str = get_value("categorical_values", row)
                if categorical_values_str:
                    # CMEO query returns "value=label|value=label" format
                    cat_pairs = categorical_values_str.split("|")
                    category_codes_str = get_value("category_concept_codes", row) or ""
                    category_labels_str = get_value("category_concept_label", row) or ""
                    category_omop_str = get_value("category_omop_id", row) or ""
                    
                    cat_codes = category_codes_str.split("|") if category_codes_str else []
                    cat_labels = category_labels_str.split("|") if category_labels_str else []
                    cat_omops = category_omop_str.split("|") if category_omop_str else []
                    
                    for idx, pair in enumerate(cat_pairs):
                        if "=" in pair:
                            value, label = pair.split("=", 1)
                            categories.append({
                                "value": value.strip(),
                                "label": label.strip(),
                                "concept_id": cat_codes[idx].strip() if idx < len(cat_codes) else "",
                                "mapped_label": cat_labels[idx].strip() if idx < len(cat_labels) else "",
                                "mapped_id": cat_omops[idx].strip() if idx < len(cat_omops) else ""
                            })
                

                target_dict[cohort_id].variables[var_id] = CohortVariable(
                    var_name=row["var_name"]["value"],
                    var_label=row["var_label"]["value"] if "var_label" in row else "",
                    var_type=row["varType"]["value"] if "varType" in row else "",
                    count=get_int_value("count_value", row) or 0,
                    max=get_value("maximum_value", row),
                    min=get_value("minimum_value", row),
                    units=get_value("unit_value", row),
                    visits=get_value("visit", row),
                    visit_concept_name=get_value("visit", row),
                    formula=get_value("formula_value", row),
                    definition=get_value("definition", row),
                    concept_id=get_curie_value("concept_codes", row),
                    concept_code=get_value("concept_codes", row),
                    concept_name=get_value("concept_labels", row),
                    mapped_id=get_curie_value("concept_omop_id", row),
                    mapped_label=get_value("concept_labels", row),
                    omop_domain=get_value("omopDomain", row),
                    index=None,  # Not in new query
                    na=0,  # Not in new query
                    categories=categories
                )
                variables_processed += 1
                
        except Exception as e:
            var_name = None
            try:
                var_name = row.get("var_name", {}).get("value") if isinstance(row.get("var_name"), dict) else None
            except Exception:
                pass
            exc_type = type(e).__name__
            # Avoid logging duplicate errors for the same cohort/var/error combo
            # (since we process many rows per cohort)
            cohort_id_str = str(cohort_id) if cohort_id else "UNKNOWN"
            var_name_str = str(var_name) if var_name else "UNKNOWN"
            sig = (cohort_id_str, var_name_str, exc_type)
            _sig_set = globals().setdefault("_SPARQL_ROW_ERROR_SIGNATURES", set())
            if len(_sig_set) > 5000:
                _sig_set.clear()
            if sig not in _sig_set:
                _sig_set.add(sig)
                logging.exception(
                    f"Error processing SPARQL row for cohort_id={cohort_id}, var_name={var_name}, error={exc_type}. Raw row keys={list(row.keys())}"
                )
            else:
                logging.debug(
                    f"Suppressed duplicate error for cohort_id={cohort_id}, var_name={var_name}, error={exc_type}"
                )
    # Merge cohorts with and without variables
    # Put cohorts with variables first so they appear at the top of the list
    logging.info(f"Merging cohorts: {len(cohorts_with_variables)} with variables, {len(cohorts_without_variables)} without")
    cohorts = {**cohorts_with_variables, **cohorts_without_variables}
    
    # Count total variables and categories
    total_variables = sum(len(c.variables) for c in cohorts.values())
    total_categories = sum(
        len(var.categories) 
        for cohort in cohorts.values() 
        for var in cohort.variables.values()
    )
    
    # Log total function execution time
    end_time = time.time()
    total_duration = end_time - start_time
    processing_duration = total_duration - query_duration
    logging.info(
        f"Cohorts metadata retrieval completed: {len(cohorts)} cohorts, {total_variables} variables, {total_categories} categories. "
        f"Time: {total_duration:.2f}s (Query: {query_duration:.2f}s, Processing: {processing_duration:.2f}s)"
    )
    
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

