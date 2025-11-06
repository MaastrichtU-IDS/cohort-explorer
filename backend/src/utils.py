import logging
from typing import Any

import curies
from rdflib import Dataset, Namespace
from rdflib.namespace import RDF, RDFS, DC
from SPARQLWrapper import JSON, SPARQLWrapper

from src.config import settings
from src.models import Cohort, CohortVariable, VariableCategory

# Define the namespaces
ICARE = Namespace("https://w3id.org/icare4cvd/")

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
        "prefix_synonyms": ["OMOP", "rxnorm extension", "RXNORM EXTENSION", "RxNorm Extension", "rxnorm_extension"]
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
    import time
    start = time.time()
    query_endpoint.setQuery(query)
    result = query_endpoint.query().convert()
    duration = time.time() - start
    if duration > 1.0:  # Log queries taking more than 1 second
        logging.warning(f"[TIMING] SPARQL query took {duration:.2f}s")
    return result


get_variables_query = """PREFIX icare: <https://w3id.org/icare4cvd/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX dcterms: <http://purl.org/dc/terms/>

SELECT DISTINCT ?cohortId ?cohortInstitution ?cohortEmail ?study_type ?study_participants ?study_duration ?study_ongoing ?study_population ?study_objective ?primary_outcome_spec ?secondary_outcome_spec ?morbidity ?study_start ?study_end ?male_percentage ?female_percentage ?administrator ?administrator_email ?study_contact_person ?study_contact_person_email ?references ?population_location ?language ?data_collection_frequency ?interventions ?sex_inclusion ?health_status_inclusion ?clinically_relevant_exposure_inclusion ?age_group_inclusion ?bmi_range_inclusion ?ethnicity_inclusion ?family_status_inclusion ?hospital_patient_inclusion ?use_of_medication_inclusion ?health_status_exclusion ?bmi_range_exclusion ?limited_life_expectancy_exclusion ?need_for_surgery_exclusion ?surgical_procedure_history_exclusion ?clinically_relevant_exposure_exclusion
    ?variable ?varName ?varLabel ?varType ?index ?count ?na ?max ?min ?units ?formula ?definition
    ?omopDomain ?conceptId ?conceptCode ?conceptName ?omopId ?mappedId ?mappedLabel ?visits ?visitConceptName ?categoryValue ?categoryLabel ?categoryConceptId ?categoryMappedId ?categoryMappedLabel
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
        OPTIONAL { ?cohort icare:morbidity ?morbidity . }
        OPTIONAL { ?cohort icare:studyStart ?study_start . }
        OPTIONAL { ?cohort icare:studyEnd ?study_end . }
        OPTIONAL { ?cohort icare:malePercentage ?male_percentage . }
        OPTIONAL { ?cohort icare:femalePercentage ?female_percentage . }
        
        # Contact information fields
        OPTIONAL { ?cohort icare:administrator ?administrator . }
        OPTIONAL { ?cohort icare:administratorEmail ?administrator_email . }
        OPTIONAL { ?cohort dc:creator ?study_contact_person . }
        OPTIONAL { ?cohort icare:email ?study_contact_person_email . }
        OPTIONAL { ?cohort icare:references ?references . }
        
        # Additional metadata fields
        OPTIONAL { ?cohort icare:populationLocation ?population_location . }
        OPTIONAL { ?cohort icare:language ?language . }
        OPTIONAL { ?cohort icare:dataCollectionFrequency ?data_collection_frequency . }
        OPTIONAL { ?cohort icare:interventions ?interventions . }
        
        # Inclusion criteria fields
        OPTIONAL { ?cohort icare:sexInclusion ?sex_inclusion . }
        OPTIONAL { ?cohort icare:healthStatusInclusion ?health_status_inclusion . }
        OPTIONAL { ?cohort icare:clinicallyRelevantExposureInclusion ?clinically_relevant_exposure_inclusion . }
        OPTIONAL { ?cohort icare:ageGroupInclusion ?age_group_inclusion . }
        OPTIONAL { ?cohort icare:bmiRangeInclusion ?bmi_range_inclusion . }
        OPTIONAL { ?cohort icare:ethnicityInclusion ?ethnicity_inclusion . }
        OPTIONAL { ?cohort icare:familyStatusInclusion ?family_status_inclusion . }
        OPTIONAL { ?cohort icare:hospitalPatientInclusion ?hospital_patient_inclusion . }
        OPTIONAL { ?cohort icare:useOfMedicationInclusion ?use_of_medication_inclusion . }
        
        # Exclusion criteria fields
        OPTIONAL { ?cohort icare:healthStatusExclusion ?health_status_exclusion . }
        OPTIONAL { ?cohort icare:bmiRangeExclusion ?bmi_range_exclusion . }
        OPTIONAL { ?cohort icare:limitedLifeExpectancyExclusion ?limited_life_expectancy_exclusion . }
        OPTIONAL { ?cohort icare:needForSurgeryExclusion ?need_for_surgery_exclusion . }
        OPTIONAL { ?cohort icare:surgicalProcedureHistoryExclusion ?surgical_procedure_history_exclusion . }
        OPTIONAL { ?cohort icare:clinicallyRelevantExposureExclusion ?clinically_relevant_exposure_exclusion . }
    }

    OPTIONAL {
        # Bind the cohortVarGraph to be the same as the cohort URI
        BIND(?cohort AS ?cohortVarGraph)
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
            # OPTIONAL { ?cohort icare:previewEnabled ?airlock . }
        }
    }
} ORDER BY ?cohort ?index
"""


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
    logging.info("Executing SPARQL query to retrieve cohorts metadata...")
    query_start_time = time.time()
    
    try:
        results = run_query(get_variables_query)["results"]["bindings"]
    except Exception as e:
        logging.error(f"SPARQL query failed: {e}")
        raise
    
    query_end_time = time.time()
    query_duration = query_end_time - query_start_time
    
    cohorts_with_variables = {}
    cohorts_without_variables = {}
    total_rows = len(results)
    logging.info(f"SPARQL query completed in {query_duration:.2f}s, returned {total_rows} rows. Starting processing...")
    
    # Track progress during row processing
    rows_processed = 0
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
            cohort_id = str(row["cohortId"]["value"])
            var_id = str(row["varName"]["value"]) if "varName" in row else None
            
            # Check if cohort exists in EITHER dictionary to avoid duplicates
            cohort_exists = cohort_id in cohorts_with_variables or cohort_id in cohorts_without_variables
            
            # Determine which dictionary to use (prefer cohorts_with_variables if it has variables)
            if cohort_id in cohorts_with_variables:
                target_dict = cohorts_with_variables
            elif cohort_id in cohorts_without_variables and var_id:
                # Move cohort from without_variables to with_variables if we now have a variable
                target_dict = cohorts_with_variables
                cohorts_with_variables[cohort_id] = cohorts_without_variables.pop(cohort_id)
            elif cohort_id in cohorts_without_variables:
                target_dict = cohorts_without_variables
            else:
                # New cohort: put in appropriate dict based on whether it has variables
                target_dict = cohorts_with_variables if var_id else cohorts_without_variables

            # Initialize cohort data structure if not exists
            if cohort_id and not cohort_exists:
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
                    administrator_email=get_value("administrator_email", row).lower() if get_value("administrator_email", row) else "",
                    study_contact_person=get_value("study_contact_person", row),
                    study_contact_person_email=get_value("study_contact_person_email", row).lower() if get_value("study_contact_person_email", row) else "",
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
                    physical_dictionary_exists=False
                )
                
                # Make the cohort accessible via target_dict before any access
                target_dict[cohort_id] = cohort
                
                # Debug logging for cohort creation
                cohort_email = get_value("cohortEmail", row).lower() if get_value("cohortEmail", row) else ""
                is_admin = user_email in settings.admins_list
                is_cohort_owner = user_email == cohort_email
                can_edit = is_admin or is_cohort_owner
                
                logging.debug(
                    f"Cohort {get_value('cohortId', row)} - "
                    f"User: {user_email}, "
                    f"Cohort Email: {cohort_email}, "
                    f"Is Admin: {is_admin}, "
                    f"Is Owner: {is_cohort_owner}, "
                    f"Can Edit: {can_edit}"
                )
                
                # Update the can_edit flag after initialization
                cohort.can_edit = can_edit
                
                # Attempt to determine if a physical dictionary file exists
                try:
                    meta_path = getattr(target_dict[cohort_id], "metadata_filepath", None)
                    target_dict[cohort_id].physical_dictionary_exists = bool(meta_path)
                except Exception as _e:
                    logging.debug(f"Skipping physical dictionary check for cohort_id={cohort_id}: {_e}")
                    target_dict[cohort_id].physical_dictionary_exists = False

            elif get_value("cohortEmail", row).lower() not in target_dict[cohort_id].cohort_email:
                # Handle multiple emails for the same cohort
                cohort_email = get_value("cohortEmail", row).lower() if get_value("cohortEmail", row) else ""
                if cohort_email:
                    target_dict[cohort_id].cohort_email.append(cohort_email)
                    if user_email == cohort_email:
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
                    visit_concept_name=get_value("visitConceptName", row),
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
            var_name = None
            try:
                var_name = row.get("varName", {}).get("value") if isinstance(row.get("varName"), dict) else None
            except Exception:
                pass
            exc_type = type(e).__name__
            sig = (str(cohort_id), str(var_name), exc_type)
            _sig_set = globals().setdefault("_SPARQL_ROW_ERROR_SIGNATURES", set())
            if len(_sig_set) > 5000:
                _sig_set.clear()
            if sig not in _sig_set:
                _sig_set.add(sig)
                logging.exception(
                    f"Error processing SPARQL row for cohort_id={cohort_id}, varName={var_name}, error={exc_type}. Raw row keys={list(row.keys())}"
                )
            else:
                logging.debug(
                    f"Suppressed duplicate error for cohort_id={cohort_id}, varName={var_name}, error={exc_type}"
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
