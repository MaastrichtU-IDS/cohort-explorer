"""
Module for managing a cohort cache to speed up cohort loading by bypassing the triple store.
This provides a parallel structure to the RDF graph for faster data access.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from src.config import settings
from src.models import Cohort, CohortVariable, VariableCategory
from rdflib import Dataset, URIRef

# Global cache for cohorts data
_cohorts_cache: Dict[str, Dict[str, Any]] = {}
_cache_initialized = False

def get_cache_file_path() -> Path:
    """Get the path to the cache file."""
    cache_dir = Path(settings.data_folder) / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "cohorts_cache.json"

def save_cache_to_disk() -> None:
    """Save the current cache to disk."""
    if not _cohorts_cache:
        logging.warning("Attempted to save empty cache to disk")
        return
    
    try:
        cache_file = get_cache_file_path()
        
        # Convert Cohort objects to serializable dictionaries
        serializable_cache = {}
        for cohort_id, cohort_data in _cohorts_cache.items():
            # Convert each cohort to a serializable dictionary
            serializable_cache[cohort_id] = cohort_to_dict(cohort_data)
        
        with open(cache_file, 'w') as f:
            json.dump(serializable_cache, f)
        
        logging.info(f"Saved cohorts cache to {cache_file}")
    except Exception as e:
        logging.error(f"Error saving cohorts cache to disk: {e}")

def load_cache_from_disk() -> bool:
    """Load the cache from disk if it exists."""
    global _cohorts_cache, _cache_initialized
    
    cache_file = get_cache_file_path()
    if not cache_file.exists():
        logging.info("No cohorts cache file found")
        return False
    
    try:
        with open(cache_file, 'r') as f:
            serialized_cache = json.load(f)
        
        # Convert serialized dictionaries back to Cohort objects
        for cohort_id, cohort_data in serialized_cache.items():
            _cohorts_cache[cohort_id] = dict_to_cohort(cohort_data)
        
        _cache_initialized = True
        logging.info(f"Loaded cohorts cache from {cache_file} with {len(_cohorts_cache)} cohorts")
        return True
    except Exception as e:
        logging.error(f"Error loading cohorts cache from disk: {e}")
        return False

def cohort_to_dict(cohort: Cohort) -> Dict[str, Any]:
    """Convert a Cohort object to a serializable dictionary."""
    cohort_dict = {
        "cohort_id": cohort.cohort_id,
        "cohort_type": cohort.cohort_type,
        "cohort_email": cohort.cohort_email,
        "institution": cohort.institution,
        "study_type": cohort.study_type,
        "study_participants": cohort.study_participants,
        "study_duration": cohort.study_duration,
        "study_ongoing": cohort.study_ongoing,
        "study_population": cohort.study_population,
        "study_objective": cohort.study_objective,
        "primary_outcome_spec": cohort.primary_outcome_spec,
        "secondary_outcome_spec": cohort.secondary_outcome_spec,
        "morbidity": cohort.morbidity,
        "study_start": cohort.study_start,
        "study_end": cohort.study_end,
        "male_percentage": cohort.male_percentage,
        "female_percentage": cohort.female_percentage,
        "administrator": cohort.administrator,
        "administrator_email": cohort.administrator_email,
        "study_contact_person": cohort.study_contact_person,
        "study_contact_person_email": cohort.study_contact_person_email,
        "references": cohort.references,
        "population_location": cohort.population_location,
        "language": cohort.language,
        "data_collection_frequency": cohort.data_collection_frequency,
        "interventions": cohort.interventions,
        "sex_inclusion": cohort.sex_inclusion,
        "health_status_inclusion": cohort.health_status_inclusion,
        "clinically_relevant_exposure_inclusion": cohort.clinically_relevant_exposure_inclusion,
        "age_group_inclusion": cohort.age_group_inclusion,
        "bmi_range_inclusion": cohort.bmi_range_inclusion,
        "ethnicity_inclusion": cohort.ethnicity_inclusion,
        "family_status_inclusion": cohort.family_status_inclusion,
        "hospital_patient_inclusion": cohort.hospital_patient_inclusion,
        "use_of_medication_inclusion": cohort.use_of_medication_inclusion,
        "health_status_exclusion": cohort.health_status_exclusion,
        "bmi_range_exclusion": cohort.bmi_range_exclusion,
        "limited_life_expectancy_exclusion": cohort.limited_life_expectancy_exclusion,
        "need_for_surgery_exclusion": cohort.need_for_surgery_exclusion,
        "surgical_procedure_history_exclusion": cohort.surgical_procedure_history_exclusion,
        "clinically_relevant_exposure_exclusion": cohort.clinically_relevant_exposure_exclusion,
        "can_edit": cohort.can_edit,
        "physical_dictionary_exists": cohort.physical_dictionary_exists,
        "variables": {}
    }
    
    # Convert variables
    for var_id, variable in cohort.variables.items():
        cohort_dict["variables"][var_id] = {
            "var_name": variable.var_name,
            "var_label": variable.var_label,
            "var_type": variable.var_type,
            "count": variable.count,
            "max": variable.max,
            "min": variable.min,
            "units": variable.units,
            "visits": variable.visits,
            "formula": variable.formula,
            "definition": variable.definition,
            "concept_id": variable.concept_id,
            "concept_code": variable.concept_code,
            "concept_name": variable.concept_name,
            "omop_id": variable.omop_id,
            "mapped_id": variable.mapped_id,
            "mapped_label": variable.mapped_label,
            "omop_domain": variable.omop_domain,
            "index": variable.index,
            "na": variable.na,
            "categories": []
        }
        
        # Convert categories
        for category in variable.categories:
            cohort_dict["variables"][var_id]["categories"].append({
                "value": category.value,
                "label": category.label,
                "concept_id": category.concept_id,
                "mapped_id": category.mapped_id,
                "mapped_label": category.mapped_label
            })
    
    return cohort_dict

def dict_to_cohort(cohort_dict: Dict[str, Any]) -> Cohort:
    """Convert a dictionary to a Cohort object."""
    # Create the cohort without variables first
    variables_dict = cohort_dict.pop("variables", {})
    cohort = Cohort(**cohort_dict)
    
    # Add variables
    for var_id, var_dict in variables_dict.items():
        categories_list = var_dict.pop("categories", [])
        variable = CohortVariable(**var_dict)
        
        # Add categories
        for cat_dict in categories_list:
            category = VariableCategory(**cat_dict)
            variable.categories.append(category)
        
        cohort.variables[var_id] = variable
    
    return cohort

def add_cohort_to_cache(cohort: Cohort) -> None:
    """Add or update a cohort in the cache."""
    global _cohorts_cache, _cache_initialized
    
    _cohorts_cache[cohort.cohort_id] = cohort
    _cache_initialized = True
    
    # Save the updated cache to disk
    save_cache_to_disk()

def remove_cohort_from_cache(cohort_id: str) -> None:
    """Remove a cohort from the cache."""
    global _cohorts_cache, _cache_initialized
    
    if cohort_id in _cohorts_cache:
        del _cohorts_cache[cohort_id]
        _cache_initialized = True
    
    # Save the updated cache to disk
    save_cache_to_disk()


def create_cohort_from_metadata_graph(cohort_id: str, cohort_uri: URIRef, g: Dataset) -> Optional[Cohort]:
    """
    Create a basic Cohort object from the metadata graph (without variables).
    This is used to add cohort metadata from the spreadsheet to the cache.
    
    Args:
        cohort_id: The ID of the cohort
        cohort_uri: The URI of the cohort in the RDF graph
        g: The RDF graph containing the cohort metadata
        
    Returns:
        A Cohort object if successful, None otherwise
    """
    try:
        # Create a basic Cohort object with the ID
        cohort = Cohort(cohort_id=cohort_id)
        
        # Extract cohort metadata from the graph
        cohorts_graph = URIRef("https://w3id.org/icare4cvd/graph/metadata")
        
        # Helper function to get a literal value from the graph
        def get_literal_value(predicate: URIRef) -> Optional[str]:
            for _, _, o, _ in g.quads((cohort_uri, predicate, None, None)):
                return str(o)
            return None
        
        # Helper function to get multiple literal values from the graph
        def get_literal_values(predicate: URIRef) -> List[str]:
            values = []
            for _, _, o, _ in g.quads((cohort_uri, predicate, None, None)):
                values.append(str(o))
            return values
        
        # Extract basic cohort metadata
        from src.utils import ICARE, DC
        
        # Institution
        cohort.institution = get_literal_value(ICARE.institution) or ""
        
        # Study type and cohort type
        cohort.study_type = get_literal_value(ICARE.studyType)
        cohort_types = get_literal_values(ICARE.cohortType)
        if cohort_types:
            cohort.cohort_type = cohort_types[0]  # Take the first one if multiple
        
        # Study details
        cohort.study_participants = get_literal_value(ICARE.studyParticipants)
        cohort.study_duration = get_literal_value(ICARE.studyDuration)
        cohort.study_ongoing = get_literal_value(ICARE.studyOngoing)
        cohort.study_population = get_literal_value(ICARE.studyPopulation)
        cohort.study_objective = get_literal_value(ICARE.studyObjective)
        cohort.primary_outcome_spec = get_literal_value(ICARE.primaryOutcomeSpec)
        cohort.secondary_outcome_spec = get_literal_value(ICARE.secondaryOutcomeSpec)
        cohort.morbidity = get_literal_value(ICARE.morbidity)
        
        # Study dates
        cohort.study_start = get_literal_value(ICARE.studyStart)
        cohort.study_end = get_literal_value(ICARE.studyEnd)
        
        # Sex distribution (added in memory 75cce55b-6774-4556-82cf-41522282e61f)
        cohort.male_percentage = None
        cohort.female_percentage = None
        mixed_sex = get_literal_value(ICARE.mixedSex)
        if mixed_sex:
            import re
            # Try to extract male and female percentages
            male_match = re.search(r'male[\s:]*([0-9.]+)\s*%', mixed_sex, re.IGNORECASE)
            female_match = re.search(r'female[\s:]*([0-9.]+)\s*%', mixed_sex, re.IGNORECASE)
            
            if male_match:
                cohort.male_percentage = float(male_match.group(1))
            if female_match:
                cohort.female_percentage = float(female_match.group(1))
        
        # Contact information
        cohort.administrator = get_literal_value(ICARE.administrator)
        cohort.administrator_email = get_literal_value(ICARE.administratorEmail)
        cohort.study_contact_person = get_literal_value(DC.creator)
        
        # Get all emails
        emails = get_literal_values(ICARE.email)
        if emails:
            cohort.cohort_email = emails
            cohort.study_contact_person_email = emails[0]  # Use the first email as contact
        
        # References
        cohort.references = get_literal_values(ICARE.references)
        
        # Additional metadata
        cohort.population_location = get_literal_value(ICARE.populationLocation)
        cohort.language = get_literal_value(ICARE.language)
        cohort.data_collection_frequency = get_literal_value(ICARE.dataCollectionFrequency)
        cohort.interventions = get_literal_value(ICARE.interventions)
        
        # Inclusion criteria
        cohort.sex_inclusion = get_literal_value(ICARE.sexInclusion)
        cohort.health_status_inclusion = get_literal_value(ICARE.healthStatusInclusion)
        cohort.clinically_relevant_exposure_inclusion = get_literal_value(ICARE.clinicallyRelevantExposureInclusion)
        cohort.age_group_inclusion = get_literal_value(ICARE.ageGroupInclusion)
        cohort.bmi_range_inclusion = get_literal_value(ICARE.bmiRangeInclusion)
        cohort.ethnicity_inclusion = get_literal_value(ICARE.ethnicityInclusion)
        cohort.family_status_inclusion = get_literal_value(ICARE.familyStatusInclusion)
        cohort.hospital_patient_inclusion = get_literal_value(ICARE.hospitalPatientInclusion)
        cohort.use_of_medication_inclusion = get_literal_value(ICARE.useOfMedicationInclusion)
        
        # Exclusion criteria
        cohort.health_status_exclusion = get_literal_value(ICARE.healthStatusExclusion)
        cohort.bmi_range_exclusion = get_literal_value(ICARE.bmiRangeExclusion)
        cohort.limited_life_expectancy_exclusion = get_literal_value(ICARE.limitedLifeExpectancyExclusion)
        cohort.need_for_surgery_exclusion = get_literal_value(ICARE.needForSurgeryExclusion)
        cohort.surgical_procedure_history_exclusion = get_literal_value(ICARE.surgicalProcedureHistoryExclusion)
        cohort.clinically_relevant_exposure_exclusion = get_literal_value(ICARE.clinicallyRelevantExposureExclusion)
        
        # Set folder path and check if physical dictionary exists
        cohort_folder_path = os.path.join(settings.data_folder, "cohorts", cohort_id)
        cohort.physical_dictionary_exists = os.path.exists(cohort_folder_path) and any(
            f.endswith("_datadictionary.csv") for f in os.listdir(cohort_folder_path)
        ) if os.path.exists(cohort_folder_path) else False
        
        # Initialize empty variables dictionary
        cohort.variables = {}
        
        # Add cohort to cache
        add_cohort_to_cache(cohort)
        
        return cohort
    except Exception as e:
        logging.error(f"Error creating cohort from metadata graph: {e}")
        return None


def create_cohort_from_dict_file(cohort_id: str, cohort_uri: URIRef, g: Dataset) -> Optional[Cohort]:
    """
    Create a Cohort object directly from the data dictionary and graph data.
    This allows us to update the cache without querying the triplestore.
    
    Args:
        cohort_id: The ID of the cohort
        cohort_uri: The URI of the cohort in the RDF graph
        g: The RDF graph containing the cohort data
        
    Returns:
        A Cohort object if successful, None otherwise
    """
    try:
        logging.info(f"Creating cohort {cohort_id} from dictionary file")
        
        # Create a basic Cohort object with the ID
        cohort = Cohort(cohort_id=cohort_id)
        
        # Set the folder path
        cohort_folder_path = os.path.join(settings.data_folder, "cohorts", cohort_id)
        
        # Check if the physical dictionary exists
        cohort.physical_dictionary_exists = os.path.exists(cohort_folder_path) and any(
            f.endswith("_datadictionary.csv") for f in os.listdir(cohort_folder_path)
        ) if os.path.exists(cohort_folder_path) else False
        
        # Extract variables from the graph - use a more efficient approach
        variables = {}
        var_count = 0
        
        # Define URIs once to avoid repeated creation
        has_variable_uri = URIRef("https://w3id.org/icare4cvd/hasVariable")
        dc_identifier_uri = URIRef("http://purl.org/dc/elements/1.1/identifier")
        rdfs_label_uri = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
        var_type_uri = URIRef("https://w3id.org/icare4cvd/varType")
        categories_uri = URIRef("https://w3id.org/icare4cvd/categories")
        rdf_value_uri = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#value")
        
        # Get all variable URIs first - use cohort_uri as graph context
        variable_uris = [o for _, _, o, g_ctx in g.quads((cohort_uri, has_variable_uri, None, cohort_uri))]
        logging.info(f"Found {len(variable_uris)} variables for cohort {cohort_id}")
        
        # Process each variable
        for var_uri in variable_uris:
            var_count += 1
            if var_count % 50 == 0:
                logging.info(f"Processing variable {var_count}/{len(variable_uris)} for cohort {cohort_id}")
                
            # Get variable properties using specific queries instead of iterating all properties
            var_name = None
            var_label = None
            var_type = None
            
            # Get variable name - use cohort_uri as graph context
            for _, _, vo, _ in g.quads((var_uri, dc_identifier_uri, None, cohort_uri)):
                var_name = str(vo)
                break
                
            # Get variable label - use cohort_uri as graph context
            for _, _, vo, _ in g.quads((var_uri, rdfs_label_uri, None, cohort_uri)):
                var_label = str(vo)
                break
                
            # Get variable type - use cohort_uri as graph context
            for _, _, vo, _ in g.quads((var_uri, var_type_uri, None, cohort_uri)):
                var_type = str(vo)
                break
            
            if not var_name or not var_label or not var_type:
                logging.warning(f"Skipping variable with incomplete data: name={var_name}, label={var_label}, type={var_type}")
                continue
                
            # Create a variable with required fields
            variable = CohortVariable(
                var_name=var_name,
                var_label=var_label,
                var_type=var_type,
                count=0  # Default count, will be updated if available
            )
            
            # Add variable to the cohort
            variables[var_name] = variable
            
            # Get category URIs for this variable - use cohort_uri as graph context
            category_uris = [co for _, _, co, _ in g.quads((var_uri, categories_uri, None, cohort_uri))]
            
            # Process each category
            for cat_uri in category_uris:
                cat_value = None
                cat_label = None
                
                # Get category value - use cohort_uri as graph context
                for _, _, cato, _ in g.quads((cat_uri, rdf_value_uri, None, cohort_uri)):
                    cat_value = str(cato)
                    break
                    
                # Get category label - use cohort_uri as graph context
                for _, _, cato, _ in g.quads((cat_uri, rdfs_label_uri, None, cohort_uri)):
                    cat_label = str(cato)
                    break
                
                if cat_value and cat_label:
                    # Create a category
                    category = VariableCategory(value=cat_value, label=cat_label)
                    variable.categories.append(category)
        
        # Add variables to the cohort
        cohort.variables = variables
        logging.info(f"Successfully processed {var_count} variables for cohort {cohort_id}")
        
        # Add the cohort to the cache
        add_cohort_to_cache(cohort)
        logging.info(f"Added cohort {cohort_id} to cache with {len(variables)} variables")
        
        return cohort
    except Exception as e:
        logging.error(f"Error creating cohort from dictionary file: {e}", exc_info=True)
        return None

def initialize_cache_from_triplestore() -> None:
    """
    Initialize the cohort cache from the triplestore.
    This is called during application startup if the triplestore is already initialized.
    """
    from src.utils import retrieve_cohorts_metadata
    
    global _cache_initialized
    
    if _cache_initialized:
        logging.info("Cache already initialized, skipping initialization")
        return
    
    logging.info("Initializing cohort cache from triplestore...")
    
    # Get all cohorts from the triplestore - use settings.admins_list[0] to ensure we get all cohorts
    admin_email = settings.admins_list[0] if settings.admins_list else "admin@example.com"
    cohorts = retrieve_cohorts_metadata(admin_email)
    
    # Add each cohort to the cache
    for cohort_id, cohort in cohorts.items():
        print(f"\nAdding cohort {cohort_id} to cache at {time.time()}")
        add_cohort_to_cache(cohort)
        print(f"Finished adding cohort {cohort_id} to cache at {time.time()}")
    
    _cache_initialized = True
    logging.info(f"Cache initialized with {len(cohorts)} cohorts from triplestore")


def initialize_cache_from_metadata_file(filepath: str) -> None:
    """
    Initialize the cohort cache directly from the Excel metadata file.
    This bypasses the SPARQL query and is more efficient for initial cache population.
    
    Args:
        filepath: Path to the Excel metadata file
    """
    import pandas as pd
    import time
    from src.models import Cohort
    
    global _cache_initialized
    
    if _cache_initialized:
        logging.info("Cache already initialized, skipping initialization from metadata file")
        return
    
    logging.info(f"Initializing cohort cache from metadata file: {filepath}")
    
    try:
        # Read the Excel file
        df = pd.read_excel(filepath, sheet_name="Descriptions")
        df = df.fillna("")
        
        # Process each row (cohort)
        cohort_count = 0
        for _i, row in df.iterrows():
            start_time = time.time()
            cohort_id = str(row["Study Name"]).strip()
            if not cohort_id:  # Skip empty rows
                continue
                
            logging.info(f"Processing cohort metadata for: {cohort_id}")
            
            # Create a basic Cohort object
            cohort = Cohort(cohort_id=cohort_id)
            
            # Set the folder path
            cohort_folder_path = os.path.join(settings.data_folder, "cohorts", cohort_id)
            
            # Check if the physical dictionary exists
            cohort.physical_dictionary_exists = os.path.exists(cohort_folder_path) and any(
                f.endswith("_datadictionary.csv") for f in os.listdir(cohort_folder_path)
            ) if os.path.exists(cohort_folder_path) else False
            
            # Add metadata fields from the Excel file
            if row.get("Institute", ""):
                cohort.institution = str(row["Institute"])
                
            if row.get("Administrator", ""):
                cohort.administrator = str(row["Administrator"])
                
            if row.get("Administrator Email Address", ""):
                cohort.administrator_email = str(row["Administrator Email Address"])
                
            if row.get("Study Contact Person", ""):
                cohort.creator = str(row["Study Contact Person"])
                
            if row.get("Study Contact Person Email Address", ""):
                cohort.emails = [email.strip() for email in str(row["Study Contact Person Email Address"]).split(";")]
                
            if row.get("References", ""):
                cohort.references = [ref.strip() for ref in str(row["References"]).split(";")]
                
            if row.get("Population Location", ""):
                cohort.population_location = str(row["Population Location"])
                
            if row.get("Language", ""):
                cohort.language = str(row["Language"])
                
            if row.get("Frequency of data collection", ""):
                cohort.data_collection_frequency = str(row["Frequency of data collection"])
                
            if row.get("Interventions", ""):
                cohort.interventions = str(row["Interventions"])
                
            if row.get("Study Type", ""):
                cohort.cohort_types = [st.strip() for st in str(row["Study Type"]).split("/")]
                
            if row.get("Study Design", ""):
                cohort.study_type = str(row["Study Design"])
                
            if row.get("Start date", "") and row.get("End date", ""):
                cohort.study_start = str(row["Start date"])
                cohort.study_end = str(row["End date"])
                
            if row.get("Number of Participants", ""):
                cohort.study_participants = str(row["Number of Participants"])
                
            if row.get("Ongoing", ""):
                cohort.study_ongoing = str(row["Ongoing"])
                
            if row.get("Study Objective", ""):
                cohort.study_objective = str(row["Study Objective"])
                
            # Handle primary and secondary outcome specifications
            primary_outcome_keys = ["primary outcome specification", "Primary outcome specification", "primary_outcome_specification"]
            for key in primary_outcome_keys:
                if key in row and row[key]:
                    cohort.primary_outcome_spec = str(row[key])
                    break
                    
            secondary_outcome_keys = ["secondary outcome specification", "Secondary outcome specification", "secondary_outcome_specification"]
            for key in secondary_outcome_keys:
                if key in row and row[key]:
                    cohort.secondary_outcome_spec = str(row[key])
                    break
                    
            # Handle Mixed Sex field for male/female percentages
            mixed_sex_keys = ["Mixed Sex", "mixed sex", "mixed_sex"]
            for key in mixed_sex_keys:
                if key in row and row[key]:
                    mixed_sex_value = str(row[key])
                    
                    # Split the string by common separators
                    parts = []
                    if ";" in mixed_sex_value:
                        parts = mixed_sex_value.split(";")
                    elif "and" in mixed_sex_value:
                        parts = mixed_sex_value.split("and")
                    else:
                        parts = [mixed_sex_value]
                    
                    # Process each part to find male and female percentages
                    for part in parts:
                        part = part.strip().lower().replace(",", ".")
                        if "male" in part and "female" not in part:  # Ensure we're not catching 'female' in 'male'
                            # Extract only digits and period for the percentage
                            digits_only = ''.join(c for c in part if c.isdigit() or c == '.')
                            if digits_only:
                                try:
                                    cohort.male_percentage = float(digits_only)
                                except ValueError:
                                    logging.warning(f"Could not convert '{digits_only}' to float for male percentage")
                        
                        if "female" in part:
                            # Extract only digits and period for the percentage
                            digits_only = ''.join(c for c in part if c.isdigit() or c == '.')
                            if digits_only:
                                try:
                                    cohort.female_percentage = float(digits_only)
                                except ValueError:
                                    logging.warning(f"Could not convert '{digits_only}' to float for female percentage")
                    break
            
            # Add the cohort to the cache
            add_cohort_to_cache(cohort)
            cohort_count += 1
            
            end_time = time.time()
            logging.info(f"Added cohort {cohort_id} to cache in {end_time - start_time:.2f} seconds")
        
        _cache_initialized = True
        logging.info(f"Cache initialized with {cohort_count} cohorts from metadata file")
        
    except Exception as e:
        logging.error(f"Error initializing cache from metadata file: {e}", exc_info=True)
        # Don't set _cache_initialized to True if there was an error


def get_cohorts_from_cache(user_email: str) -> Dict[str, Cohort]:
    """
    Get all cohorts from the cache, updating the can_edit field based on user email."""
    global _cohorts_cache, _cache_initialized
    
    if not _cache_initialized:
        # Cache not initialized, load from disk if available
        load_cache_from_disk()
        
        # If still not initialized or empty, try to initialize from triplestore
        if not _cohorts_cache:
            initialize_cache_from_triplestore()
    
    # If still not initialized or empty, return empty dict
    if not _cohorts_cache:
        return {}
    
    # Create a copy of the cache with updated can_edit fields
    result = {}
    for cohort_id, cohort in _cohorts_cache.items():
        # Create a copy of the cohort
        cohort_copy = dict_to_cohort(cohort_to_dict(cohort))
        
        # Update can_edit based on user email (admin or cohort owner)
        cohort_copy.can_edit = user_email in [*settings.admins_list, *cohort_copy.cohort_email]
        
        result[cohort_id] = cohort_copy
    
    return result

def is_cache_initialized() -> bool:
    """Check if the cache has been initialized."""
    global _cache_initialized
    return _cache_initialized

def clear_cache() -> None:
    """Clear the cache."""
    global _cohorts_cache, _cache_initialized
    _cohorts_cache = {}
    _cache_initialized = False
    
    # Remove the cache file if it exists
    cache_file = get_cache_file_path()
    if cache_file.exists():
        os.remove(cache_file)
        logging.info(f"Removed cohorts cache file {cache_file}")
