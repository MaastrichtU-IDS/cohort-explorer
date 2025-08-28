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
        
        # Get all variable URIs first
        variable_uris = [o for _, _, o, _ in g.quads((cohort_uri, has_variable_uri, None, None))]
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
            
            # Get variable name
            for _, _, vo, _ in g.quads((var_uri, dc_identifier_uri, None, None)):
                var_name = str(vo)
                break
                
            # Get variable label
            for _, _, vo, _ in g.quads((var_uri, rdfs_label_uri, None, None)):
                var_label = str(vo)
                break
                
            # Get variable type
            for _, _, vo, _ in g.quads((var_uri, var_type_uri, None, None)):
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
            
            # Get category URIs for this variable
            category_uris = [co for _, _, co, _ in g.quads((var_uri, categories_uri, None, None))]
            
            # Process each category
            for cat_uri in category_uris:
                cat_value = None
                cat_label = None
                
                # Get category value
                for _, _, cato, _ in g.quads((cat_uri, rdf_value_uri, None, None)):
                    cat_value = str(cato)
                    break
                    
                # Get category label
                for _, _, cato, _ in g.quads((cat_uri, rdfs_label_uri, None, None)):
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

def initialize_cache_from_triplestore(admin_email: str = "admin@example.com") -> None:
    """Initialize the cache from the triplestore.
    
    This function can be called independently of the triplestore initialization
    to ensure the cache is built even when the triplestore is not empty.
    
    Args:
        admin_email: Email to use for retrieving cohorts from the triplestore
    """
    global _cohorts_cache, _cache_initialized
    
    # Clear the cache before initialization
    clear_cache()
    
    # Import here to avoid circular imports
    from src.utils import retrieve_cohorts_metadata
    
    try:
        # Retrieve cohorts from the triplestore
        cohorts = retrieve_cohorts_metadata(admin_email)
        
        # Add each cohort to the cache
        for cohort_id, cohort in cohorts.items():
            add_cohort_to_cache(cohort)
        
        _cache_initialized = True
        logging.info(f"Cache initialized with {len(cohorts)} cohorts from triplestore")
    except Exception as e:
        logging.error(f"Error initializing cache from triplestore: {e}")


def get_cohorts_from_cache(user_email: str) -> Dict[str, Cohort]:
    """Get all cohorts from the cache, updating the can_edit field based on user email."""
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
