from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class StatisticalVariableType (Enum):
    CATEGERICAL = "categorical_variable"
    Dischothomous = "binary_class_variable"
    POLYCHOTHOMOUS = "multi_class_variable"


@dataclass
class Concept:
    code: Optional[str] = None
    omop_id: Optional[int] = None
    standard_label: Optional[str] = None
    terminology: Optional[str] = None
@dataclass
class DataElement:
    """Metadata about a category (possible values) of a cohort variable"""

    name: Optional[str] = None
    label:Optional[str] = None
    concept: Optional[list[Concept]] = None



@dataclass
class Contextual_Factors:
    """Metadata about a category (possible values) of a cohort variable"""

    label:Optional[str] = None
    concept: Optional[Concept] = None




@dataclass
class MeasurementUnit:
    """Metadata about a category (possible values) of a cohort variable"""

    value: str
    concept: Optional[Concept] = None

@dataclass
class Statistic:

    minimum_value: Optional[int] = None
    maximum_value: Optional[int] = None
    missing_value_count : Optional[int] = None
    total_data_items : Optional[int] = None


@dataclass
class VisitMeasurementDatum:

    label: Optional[int] = None
    concept: Optional[list[Concept]] = None

@dataclass
class VariableCategory:
    """Metadata about a category (possible values) of a cohort variable"""

    label: Optional[str] = None
    code: Optional[str] = None
    omop_id: Optional[int] = None
    representative_value: Optional[str] = None
    standard_label: Optional[str] = None

@dataclass
class ValueSpecification:
    """Metadata about a category (possible values) of a cohort variable"""

    value: Optional[str] = None

@dataclass
class DataElementSummary:
    """Metadata about a variable (aka. column) in a cohort"""

    data_element: Optional[DataElement] = None
    # context: List[Contextual_Factors] = field(default_factory=list)
    measuement_unit: Optional[MeasurementUnit] = None
    broader_category: Optional[str] = None
    has_statistica_variable: Optional[StatisticalVariableType] = None
    permissable_values: List[VariableCategory] = field(default_factory=list)
    statistic : Optional[Statistic] = None
    contextul_factors: Optional[Contextual_Factors] = None
    has_formula: Optional[str] = None


@dataclass
class StudyDesignSpecification:
    primary_outcome_specification : Optional[str] = None
    secondary_outcome_specification : Optional[str] = None
    inclusion_criteria : Optional[str] = None
    exclusion_criteria : Optional[str] = None
    organization : Optional[str] = None
    organization_contact_personnel : Optional[str] = None
    organization_contact_personnel_email : Optional[str] = None
    study_primary_purpose_specification : Optional[str] = None
    number_of_participants : Optional[int] = None

@dataclass
class Study:

    study_type : Optional[str] = None   # study classifies as the type of study
    study_design : Optional[StudyDesignSpecification] = None  # study design conforms to study type 
    study_design_execution : Optional[str] = None  # study design execution realizes the study plan
    study_plan : Optional[str] = None  # study plan concretizes the study design
    study_duration : Optional[str] = None  # study duration is the duration of the study design execution (process)
    start_date : Optional[str] = None  # start date is the date when the study design execution starts
    study_completion_date : Optional[str] = None  # end date is the date when the study design execution ends

    




