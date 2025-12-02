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

def get_cache_timestamp_file() -> Path:
    """Get the path to the cache timestamp file (marks when service started)."""
    cache_dir = Path(settings.data_folder) / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / ".cache_timestamp"

def save_cache_to_disk() -> None:
    """Save the current cache to disk using atomic write with file locking."""
    if not _cohorts_cache:
        logging.warning("Attempted to save empty cache to disk")
        return
    
    import fcntl
    import tempfile
    
    try:
        cache_file = get_cache_file_path()
        lock_file_path = cache_file.parent / ".cache_write.lock"
        
        # Acquire lock to prevent concurrent writes
        with open(lock_file_path, 'w') as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                
                # Convert Cohort objects to serializable dictionaries
                serializable_cache = {}
                for cohort_id, cohort_data in _cohorts_cache.items():
                    # Convert each cohort to a serializable dictionary
                    serializable_cache[cohort_id] = cohort_to_dict(cohort_data)
                
                # Use atomic write: write to temp file then rename
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=cache_file.parent,
                    prefix='.cohorts_cache_',
                    suffix='.tmp'
                )
                
                try:
                    with os.fdopen(temp_fd, 'w') as f:
                        json.dump(serializable_cache, f, indent=2)
                    
                    # Atomic rename (overwrites existing file)
                    os.replace(temp_path, cache_file)
                    logging.info(f"Saved cohorts cache to {cache_file} ({len(_cohorts_cache)} cohorts)")
                except Exception as write_error:
                    # Clean up temp file on error
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise write_error
                    
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                
    except Exception as e:
        logging.error(f"Error saving cohorts cache to disk: {e}")

def load_cache_from_disk() -> bool:
    """Load the cache from disk if it exists, with file locking to prevent reading partial writes."""
    global _cohorts_cache, _cache_initialized
    
    import fcntl
    
    cache_file = get_cache_file_path()
    if not cache_file.exists():
        logging.info("No cohorts cache file found")
        return False
    
    lock_file_path = cache_file.parent / ".cache_write.lock"
    
    try:
        # Try to acquire read lock (will wait if someone is writing)
        with open(lock_file_path, 'w') as lock_file:
            try:
                # Use shared lock for reading (multiple readers OK, blocks if writer active)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
                
                with open(cache_file, 'r') as f:
                    serialized_cache = json.load(f)
                
                # Convert serialized dictionaries back to Cohort objects
                for cohort_id, cohort_data in serialized_cache.items():
                    _cohorts_cache[cohort_id] = dict_to_cohort(cohort_data)
                
                _cache_initialized = True
                logging.info(f"Loaded cohorts cache from {cache_file} with {len(_cohorts_cache)} cohorts")
                return True
                
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                
    except json.JSONDecodeError as json_err:
        logging.error(f"Error loading cohorts cache from disk: Invalid JSON - {json_err}. Cache file may be corrupted, will re-initialize.")
        # Delete corrupted cache file
        try:
            cache_file.unlink()
            logging.info("Deleted corrupted cache file")
        except:
            pass
        return False
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
            "visit_concept_name": variable.visit_concept_name,
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

def add_cohort_to_cache(cohort: Cohort, save_to_disk: bool = True) -> None:
    """Add or update a cohort in the cache.
    
    Args:
        cohort: The cohort to add
        save_to_disk: If True, saves cache to disk after adding. Set to False during
                     bulk operations to avoid race conditions and improve performance.
    """
    global _cohorts_cache, _cache_initialized
    
    _cohorts_cache[cohort.cohort_id] = cohort
    _cache_initialized = True
    
    # Save the updated cache to disk (unless explicitly disabled for bulk operations)
    if save_to_disk:
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
        admin_email = get_literal_value(ICARE.administratorEmail)
        cohort.administrator_email = admin_email.lower() if admin_email else None
        cohort.study_contact_person = get_literal_value(DC.creator)
        
        # Get all emails (normalize to lowercase)
        emails = get_literal_values(ICARE.email)
        if emails:
            cohort.cohort_email = [email.lower() for email in emails if email]
            cohort.study_contact_person_email = cohort.cohort_email[0]  # Use the first email as contact
        
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
    Create or update a Cohort object with variables from the data dictionary and graph data.
    This preserves existing cohort metadata if the cohort is already in the cache.
    
    Args:
        cohort_id: The ID of the cohort
        cohort_uri: The URI of the cohort in the RDF graph
        g: The RDF graph containing the cohort data
        
    Returns:
        A Cohort object if successful, None otherwise
    """
    try:
        global _cohorts_cache
        
        # Check if cohort already exists in cache (with metadata)
        if cohort_id in _cohorts_cache:
            # Use existing cohort to preserve metadata
            cohort = _cohorts_cache[cohort_id]
            logging.info(f"Updating existing cohort {cohort_id} in cache with variables from dictionary")
        else:
            # Create a new basic Cohort object with the ID
            cohort = Cohort(cohort_id=cohort_id)
            logging.info(f"Creating new cohort {cohort_id} in cache from dictionary file")
        
        # Set the folder path
        cohort_folder_path = os.path.join(settings.data_folder, "cohorts", cohort_id)
        
        # Check if the physical dictionary exists
        cohort.physical_dictionary_exists = os.path.exists(cohort_folder_path) and any(
            f.endswith("_datadictionary.csv") for f in os.listdir(cohort_folder_path)
        ) if os.path.exists(cohort_folder_path) else False
        
        # Extract variables from the graph
        variables = {}
        
        # Find all variables for this cohort
        for s, p, o, _ in g.quads((cohort_uri, URIRef("https://w3id.org/icare4cvd/hasVariable"), None, None)):
            var_uri = o
            var_name = None
            var_label = None
            var_type = None
            
            # Get variable properties
            for vs, vp, vo, _ in g.quads((var_uri, None, None, None)):
                if vp == URIRef("http://purl.org/dc/elements/1.1/identifier"):
                    var_name = str(vo)
                elif vp == URIRef("http://www.w3.org/2000/01/rdf-schema#label"):
                    var_label = str(vo)
                elif vp == URIRef("https://w3id.org/icare4cvd/varType"):
                    var_type = str(vo)
            
            if var_name and var_label and var_type:
                # Create a variable with required fields
                variable = CohortVariable(
                    var_name=var_name,
                    var_label=var_label,
                    var_type=var_type,
                    count=0  # Default count, will be updated if available
                )
                
                # Add variable to the cohort
                variables[var_name] = variable
                
                # Get categories for this variable
                for cs, cp, co, _ in g.quads((var_uri, URIRef("https://w3id.org/icare4cvd/categories"), None, None)):
                    cat_uri = co
                    cat_value = None
                    cat_label = None
                    
                    # Get category properties
                    for cats, catp, cato, _ in g.quads((cat_uri, None, None, None)):
                        if catp == URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#value"):
                            cat_value = str(cato)
                        elif catp == URIRef("http://www.w3.org/2000/01/rdf-schema#label"):
                            cat_label = str(cato)
                    
                    if cat_value and cat_label:
                        # Create a category
                        category = VariableCategory(value=cat_value, label=cat_label)
                        variable.categories.append(category)
        
        # Update the cohort's variables (preserve existing metadata)
        cohort.variables = variables
        
        # Add the cohort to the cache
        add_cohort_to_cache(cohort)
        
        return cohort
    except Exception as e:
        logging.error(f"Error creating cohort from dictionary file: {e}")
        return None

def initialize_cache_from_triplestore(admin_email: str | None = None, force_refresh: bool = False) -> None:
    """Initialize the cache from the triplestore.
    
    This function can be called independently of the triplestore initialization
    to ensure the cache is built even when the triplestore is not empty.
    Uses file-based locking to prevent multiple workers from initializing simultaneously.
    
    Args:
        admin_email: Email to use for retrieving cohorts from the triplestore. 
                     If None, uses first admin from settings.
        force_refresh: If True, forces re-initialization even if cache exists
    """
    global _cohorts_cache, _cache_initialized
    
    # If already initialized and not forcing refresh, skip
    if _cache_initialized and not force_refresh:
        logging.info("Cache already initialized, skipping")
        return
    
    import fcntl
    import time
    from src.config import settings
    
    # Get admin email from settings if not provided
    if admin_email is None:
        admin_email = settings.admins_list[0] if settings.admins_list else ""
    
    lock_file_path = os.path.join(settings.data_folder, ".cache_init.lock")
    timestamp_file = get_cache_timestamp_file()
    
    # Check for and remove stale lock files
    max_lock_age = 600  # 10 minutes in seconds (same as max_wait)
    if os.path.exists(lock_file_path):
        try:
            lock_age = time.time() - os.path.getmtime(lock_file_path)
            if lock_age > max_lock_age:
                logging.warning(f"Found stale lock file (age: {lock_age:.0f}s), removing it...")
                os.remove(lock_file_path)
                logging.info("Stale lock file removed successfully")
        except Exception as e:
            logging.warning(f"Could not check/remove stale lock file: {e}")
    
    # Try to acquire lock with timeout
    lock_file = None
    try:
        lock_file = open(lock_file_path, 'w')
        # Try to acquire exclusive lock with timeout (non-blocking)
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            logging.info("Acquired cache initialization lock, starting initialization...")
        except IOError:
            # Another process is initializing, wait for it to finish
            logging.info("Another worker is initializing cache, waiting...")
            max_wait = 600  # Wait up to 10 minutes
            start_wait = time.time()
            while time.time() - start_wait < max_wait:
                time.sleep(2)
                if _cache_initialized or is_cache_initialized():
                    logging.info("Cache initialized by another worker")
                    return
            logging.warning("Timeout waiting for cache initialization by another worker")
            return
        
        # Clear the cache before initialization
        clear_cache()
        
        # Import here to avoid circular imports
        from src.utils import retrieve_cohorts_metadata
        
        # Retrieve cohorts from the triplestore
        cohorts = retrieve_cohorts_metadata(admin_email)
        
        # Add each cohort to the cache (without saving to disk after each one)
        for cohort_id, cohort in cohorts.items():
            add_cohort_to_cache(cohort, save_to_disk=False)
        
        _cache_initialized = True
        
        # Save all cohorts to disk at once (prevents race conditions)
        save_cache_to_disk()
        
        # Write timestamp file to mark this initialization session
        with open(timestamp_file, 'w') as f:
            f.write(str(time.time()))
        
        logging.info(f"Cache initialized with {len(cohorts)} cohorts from triplestore")
        
    except Exception as e:
        logging.error(f"Error initializing cache from triplestore: {e}")
    finally:
        # Release lock and clean up lock file
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                # Delete the lock file to prevent stale locks and signal completion
                if os.path.exists(lock_file_path):
                    os.remove(lock_file_path)
                    logging.debug("Lock file removed after releasing lock")
            except Exception as e:
                logging.warning(f"Error cleaning up lock file: {e}")


def get_cohorts_from_cache(user_email: str) -> Dict[str, Cohort]:
    """Get all cohorts from the cache, updating the can_edit field based on user email."""
    global _cohorts_cache, _cache_initialized
    
    import time
    
    # If cache is empty in this worker's memory
    if not _cohorts_cache:
        cache_file = get_cache_file_path()
        timestamp_file = get_cache_timestamp_file()
        
        # Check if we need to refresh cache (timestamp file doesn't exist = new service start)
        if not timestamp_file.exists():
            # New service start - force fresh initialization from triplestore
            logging.info("New service start detected, initializing fresh cache from triplestore")
            initialize_cache_from_triplestore(force_refresh=True)
        elif cache_file.exists():
            # Timestamp exists and cache file exists - load from disk (same session)
            logging.info("Loading cache from disk for this worker")
            load_cache_from_disk()
        else:
            # Timestamp exists but no cache file - initialize
            logging.info("No cache file found, initializing from triplestore")
            initialize_cache_from_triplestore()
    
    # If still empty, return empty dict
    if not _cohorts_cache:
        logging.warning("Cache is empty after initialization attempts")
        return {}
    
    # Create a copy of the cache with updated can_edit fields
    result = {}
    # Normalize user email to lowercase for case-insensitive comparison
    user_email_lower = user_email.lower() if user_email else ""
    
    for cohort_id, cohort in _cohorts_cache.items():
        # Create a copy of the cohort
        cohort_copy = dict_to_cohort(cohort_to_dict(cohort))
        
        # Check permissions - user can edit if they are:
        # 1. Global admin
        # 2. In cohort_email list (data owners)
        # 3. Administrator email
        # 4. Study contact person email
        is_admin = user_email_lower in settings.admins_list
        is_cohort_owner = user_email_lower in cohort_copy.cohort_email
        is_administrator = cohort_copy.administrator_email and user_email_lower == cohort_copy.administrator_email.lower()
        is_contact_person = cohort_copy.study_contact_person_email and user_email_lower == cohort_copy.study_contact_person_email.lower()
        
        cohort_copy.can_edit = is_admin or is_cohort_owner or is_administrator or is_contact_person
        
        # Debug logging if user should have access but doesn't
        should_have_access = user_email_lower in [*settings.admins_list, *cohort_copy.cohort_email] or \
                            (cohort_copy.administrator_email and user_email_lower == cohort_copy.administrator_email.lower()) or \
                            (cohort_copy.study_contact_person_email and user_email_lower == cohort_copy.study_contact_person_email.lower())
        
        if not cohort_copy.can_edit and should_have_access:
            logging.warning(
                f"Permission issue for user {user_email_lower} on cohort {cohort_id}. "
                f"is_admin: {is_admin}, is_cohort_owner: {is_cohort_owner}, "
                f"is_administrator: {is_administrator}, is_contact_person: {is_contact_person}, "
                f"cohort_emails: {cohort_copy.cohort_email}, "
                f"administrator_email: {cohort_copy.administrator_email}, "
                f"study_contact_person_email: {cohort_copy.study_contact_person_email}"
            )
        
        result[cohort_id] = cohort_copy
    
    return result

def is_cache_initialized() -> bool:
    """Check if the cache has been initialized.
    
    Checks both in-memory flag and disk cache file to handle multi-worker scenarios.
    """
    global _cache_initialized, _cohorts_cache
    
    # Check in-memory first
    if _cache_initialized and _cohorts_cache:
        return True
    
    # Check if cache file exists on disk (for multi-worker scenarios)
    cache_file = get_cache_file_path()
    if cache_file.exists():
        # Try to load from disk if not already loaded
        if not _cohorts_cache:
            load_cache_from_disk()
        return len(_cohorts_cache) > 0
    
    return False

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
