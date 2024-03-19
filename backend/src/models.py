from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class VariableCategory:
    """Metadata about a category (possible values) of a cohort variable"""

    value: str
    label: str
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
    categories: List[VariableCategory] = field(default_factory=list)


@dataclass
class Cohort:
    """Metadata about a cohort (1 tabular data file)"""

    cohort_id: str
    cohort_type: Optional[str] = None
    cohort_email: list[str] = field(default_factory=list)
    owner: Optional[str] = None
    institution: str = ""
    study_type: Optional[str] = None
    study_participants: Optional[str] = None
    study_duration: Optional[str] = None
    study_ongoing: Optional[str] = None
    study_population: Optional[str] = None
    study_objective: Optional[str] = None
    variables: Dict[str, CohortVariable] = field(default_factory=dict)
    can_edit: bool = False

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
