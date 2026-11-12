import logging
import re
from typing import Any

import curies
from rdflib import Dataset, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, DC
import urllib.parse
from SPARQLWrapper import JSON, POST, SPARQLWrapper

# The vocabulary of the knowledge graph belongs to CohortVarLinker (its graph
# writers and mapping queries use it): OntologyNamespaces.<NS>.value.<name>
# resolves a friendly term name to the ontology IRI through term_iri, whichever
# namespace it is called on; names without an ontology class resolve to the
# KG-local base CMEO_KG (https://w3id.org/CMEO/). The explorer writes and reads
# the same IRIs by importing them from there instead of keeping its own copy.
from CohortVarLinker.src.utils import CMEO_KG, OBO, SIO_NS, OntologyNamespaces, study_design_class, term_iri  # noqa: F401
from src.config import settings
from src.kg_queries import cohorts_metadata_query, parse_category_records, variables_metadata_query
from src.models import Cohort, CohortVariable, VariableCategory

# Define the namespaces
ICARE = Namespace("https://w3id.org/icare4cvd/")
# KG-local predicates the explorer itself puts on the cohort node (email,
# institution, studyType, availableDataTypes, ...): plain names under the KG
# base, with no ontology class behind them. Written and read through this
# namespace so the intent is explicit and term_iri does not warn about them.
CMEO_LOCAL = Namespace(CMEO_KG)


def split_emails(*values: Any) -> list[str]:
    """Individual, lowercased email addresses out of raw metadata email values.

    A value may be a single address, a ';'/','-joined string of several (a
    common spreadsheet entry: "person1@org.com;person2@org.com"), or a list of
    such strings. Order is kept, duplicates and non-address junk are dropped.
    Every place that turns cohort metadata emails into participants or
    permissions must go through this, or a combined cell becomes one bogus
    "person1@org.com;person2@org.com" participant."""
    out: list[str] = []

    def _one(v: Any) -> None:
        if not v:
            return
        if isinstance(v, (list, tuple, set)):
            for x in v:
                _one(x)
            return
        for part in re.split(r"[;,\s]+", str(v)):
            e = part.strip().lower()
            if e and "@" in e and e not in out:
                out.append(e)

    for v in values:
        _one(v)
    return out

query_endpoint = SPARQLWrapper(settings.query_endpoint)
query_endpoint.setReturnFormat(JSON)
# Use POST method for large queries (GET has 8KB header limit)
query_endpoint.setMethod(POST)
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
        "prefix_synonyms": ["snomed", "SNOMED", "snomed-ct", "SNOMED-CT"],
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
    if text is None:
        return None
    # Convert to string and strip whitespace first
    text = str(text).strip()
    # Check for empty or nan after stripping
    if text == "" or text.lower() == "nan":
        return None
    text = text.lower().replace(" ", "_").replace("/", "_").replace(":", "_").replace('[','').replace(']','')
    # Final check - if after all replacements we have empty string, return None
    if not text:
        return None
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
    g.bind("cmeo", CMEO_LOCAL)
    g.bind("obo", Namespace(OBO))
    g.bind("sio", Namespace(SIO_NS))
    g.bind("skos", OntologyNamespaces.SKOS.value)
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


def get_cohorts_metadata_query() -> str:
    """SPARQL for one row per cohort from the studies_metadata graph (src.kg_queries)."""
    return cohorts_metadata_query()


def get_variables_metadata_query() -> str:
    """SPARQL for one row per variable from every cohort graph (src.kg_queries)."""
    return variables_metadata_query()


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
            
        # Get both email fields for data owners
        admin_email = get_value("administrator_email", row).lower() if get_value("administrator_email", row) else ""
        contact_email = get_value("study_contact_person_email", row).lower() if get_value("study_contact_person_email", row) else ""

        # Build list of data owner emails (both administrator and study contact
        # person). A field may hold several ';'-joined addresses: split them so
        # each person becomes their own owner entry.
        data_owner_emails = split_emails(admin_email, contact_email)
        
        cohort = Cohort(
            cohort_id=cohort_id,
            cohort_email=data_owner_emails,  # Both emails are data owners
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
            administrator_email=admin_email,
            study_contact_person=get_value("study_contact_person", row),
            study_contact_person_email=contact_email,
            # The graph only knows the legacy roles: PI = administrator, data provider = study contact person.
            pi=get_value("administrator", row),
            pi_email=admin_email,
            data_provider=get_value("study_contact_person", row),
            data_provider_email=contact_email,
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
                # Categories: one record per category, each carrying its own
                # concept (kg_queries.parse_category_records), so a category
                # without a concept no longer shifts the others' codes.
                categories = [
                    VariableCategory(
                        value=rec["value"],
                        label=rec["label"] or rec["value"],
                        concept_id=rec["code"],
                        mapped_label=rec["code_label"],
                        mapped_id=rec["omop_id"],
                    )
                    for rec in parse_category_records(get_value("categorical_values", row))
                ]

                target_dict[cohort_id].variables[var_id] = CohortVariable(
                    var_name=row["var_name"]["value"],
                    var_label=row["var_label"]["value"] if "var_label" in row else "",
                    var_type=row["varType"]["value"] if "varType" in row else "",
                    count=get_int_value("count_value", row) or 0,
                    max=get_value("maximum_value", row),
                    min=get_value("minimum_value", row),
                    units=get_value("unit_value", row),
                    visits=get_value("visit", row),
                    visit_concept_name=get_value("visit_concept_name", row) or get_value("visit", row),
                    formula=get_value("formula_value", row),
                    definition=get_value("definition", row),
                    concept_id=get_curie_value("concept_codes", row),
                    concept_code=get_value("concept_codes", row),
                    concept_name=get_value("concept_labels", row),
                    omop_id=get_value("concept_omop_id", row),
                    mapped_id=None,  # Only set when user manually maps a concept
                    mapped_label=None,  # Only set when user manually maps a concept
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
    # Merge cohorts with and without variables.
    # Within each group, sort alphabetically (case-insensitive) by cohort_id so
    # the order is deterministic and matches the cache-backed endpoint.
    logging.info(f"Merging cohorts: {len(cohorts_with_variables)} with variables, {len(cohorts_without_variables)} without")
    with_vars_sorted = sorted(cohorts_with_variables.keys(), key=lambda cid: cid.lower())
    without_vars_sorted = sorted(cohorts_without_variables.keys(), key=lambda cid: cid.lower())
    cohorts = {
        **{cid: cohorts_with_variables[cid] for cid in with_vars_sorted},
        **{cid: cohorts_without_variables[cid] for cid in without_vars_sorted},
    }
    
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

