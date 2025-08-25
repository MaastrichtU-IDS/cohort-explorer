import os
import re # Import re for regex matching in metadata_filepath
from dataclasses import asdict, dataclass, field
from typing import Optional

from src.config import settings


@dataclass
class VariableCategory:
    """Metadata about a category (possible values) of a cohort variable"""

    value: str
    label: str
    concept_id: Optional[str] = None
    mapped_id: Optional[str] = None
    mapped_label: Optional[str] = None


@dataclass
class CohortVariable:
    """Metadata about a variable (aka. column) in a cohort"""

    var_name: str
    var_label: str
    var_type: str
    count: int
    max: Optional[str] = None
    min: Optional[str] = None
    units: Optional[str] = None
    visits: Optional[str] = None
    formula: Optional[str] = None
    definition: Optional[str] = None
    concept_id: Optional[str] = None
    mapped_id: Optional[str] = None
    mapped_label: Optional[str] = None
    omop_domain: Optional[str] = None
    index: Optional[int] = None
    na: int = 0
    categories: list[VariableCategory] = field(default_factory=list)


@dataclass
class Cohort:
    """Metadata about a cohort (1 tabular data file)"""

    cohort_id: str
    cohort_type: Optional[str] = None
    cohort_email: list[str] = field(default_factory=list)
    institution: str = ""
    study_type: Optional[str] = None
    study_participants: Optional[str] = None
    study_duration: Optional[str] = None
    study_ongoing: Optional[str] = None
    study_population: Optional[str] = None
    study_objective: Optional[str] = None
    primary_outcome_spec: Optional[str] = None
    secondary_outcome_spec: Optional[str] = None
    morbidity: Optional[str] = None
    study_start: Optional[str] = None
    study_end: Optional[str] = None
    male_percentage: Optional[float] = None
    female_percentage: Optional[float] = None
    # Contact information fields
    administrator: Optional[str] = None
    administrator_email: Optional[str] = None
    study_contact_person: Optional[str] = None
    study_contact_person_email: Optional[str] = None
    references: list[str] = field(default_factory=list)
    # Additional metadata fields
    population_location: Optional[str] = None
    language: Optional[str] = None
    data_collection_frequency: Optional[str] = None
    # Inclusion criteria fields
    sex_inclusion: Optional[str] = None
    health_status_inclusion: Optional[str] = None
    clinically_relevant_exposure_inclusion: Optional[str] = None
    age_group_inclusion: Optional[str] = None
    bmi_range_inclusion: Optional[str] = None
    ethnicity_inclusion: Optional[str] = None
    family_status_inclusion: Optional[str] = None
    hospital_patient_inclusion: Optional[str] = None
    use_of_medication_inclusion: Optional[str] = None
    
    # Exclusion criteria fields
    health_status_exclusion: Optional[str] = None
    bmi_range_exclusion: Optional[str] = None
    limited_life_expectancy_exclusion: Optional[str] = None
    need_for_surgery_exclusion: Optional[str] = None
    surgical_procedure_history_exclusion: Optional[str] = None
    clinically_relevant_exposure_exclusion: Optional[str] = None
    variables: dict[str, CohortVariable] = field(default_factory=dict)
    can_edit: bool = False
    physical_dictionary_exists: bool = False # New field

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

    @property
    def folder_path(self) -> str:
        return os.path.join(settings.data_folder, "cohorts", self.cohort_id)

    @property
    def metadata_filepath(self) -> str:
        if not os.path.exists(self.folder_path) or not os.path.isdir(self.folder_path):
            # If folder doesn't exist, no file can be found.
            raise FileNotFoundError(f"Cohort data folder not found for {self.cohort_id} at {self.folder_path}")

        candidate_files = []
        for filename in os.listdir(self.folder_path):
            filepath = os.path.join(self.folder_path, filename)
            if not os.path.isfile(filepath): # Skip directories
                continue

            # Check for the suffix and exclude known non-primary files
            if filename.endswith("_datadictionary.csv") and \
               not "_noHeader" in filename and \
               not re.match(r'.*_\d{8}_\d{6}\.csv$', filename): # Exclude timestamped backups
                candidate_files.append(filepath)
        
        if not candidate_files:
            raise FileNotFoundError(f"No suitable metadata dictionary file found for cohort {self.cohort_id} in {self.folder_path}")
        
        # If multiple candidates, pick the most recently modified one
        if len(candidate_files) > 1:
            # Sort by modification time, most recent first
            candidate_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            # Potentially log a warning if multiple candidates are found, as it might indicate a cleanup issue
            # logging.warning(f"Multiple candidate dictionary files found for cohort {self.cohort_id}, selecting most recent: {candidate_files[0]}")
        
        return candidate_files[0]
        # return os.path.join(self.folder_path, f"{self.cohort_id}_datadictionary.csv")
        # return os.path.join(self.folder_path, f"{self.cohort_id}-metadata.csv")
