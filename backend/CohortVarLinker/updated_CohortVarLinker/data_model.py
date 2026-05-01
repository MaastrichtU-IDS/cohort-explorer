from __future__ import annotations
import math
from enum import Enum, IntEnum
from typing import Any, ClassVar, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field
# import re
# from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class VDGResult:
    """Result of comparing two Variable Descriptor Graphs."""
    
    main_overlap: float          # 0.0–1.0: do expanded MAIN concept neighborhoods intersect?
    context_overlap: float       # 0.0–1.0: do expanded CONTEXT neighborhoods intersect?
    value_overlap: float         # 0.0–1.0: do expanded VALUE SET neighborhoods intersect?
    cross_role_signal: bool      # src MAIN found in tgt VALUES or vice versa (binary extraction)
    context_subsumption: bool    # one context is subset of the other
    context_asymmetry: bool      # one has context the other doesn't
    resolved: bool               # did the graph find ANY evidence? (False → LLM fallback)



# =============================================================================
# ENUMS
# =============================================================================


class EmbeddingType(str, Enum):
    ED = "embedding(description)"
    EC = "embedding(concept)"
    EH = "embedding(hybrid)"

class MappingType(str, Enum):
    OO = "OO" #ontology_only
    OEC = "OEC" #ontology+embedding(concept)
    OEH = "OEH" #ontology+embedding(hybrid)
    NE = "NE" #neural matching only (no ontology)

# class MappingType(str, Enum):
#     OO = "OO" #ontology_only
#     NS = "NS" #neural symbolic matching
#     NO = "NO" #neural only matching


class ContextMatchType(IntEnum):
    PENDING = 0
    EXACT = 1
    COMPATIBLE = 2
    SUBSUMED = 3
    PARTIAL = 4
    NOT_APPLICABLE = 5

    def to_str(self) -> str:
        return {0:"pending", 1:"exact match", 2:"compatible match",
                3:"subsumed", 4:"partial match", 5:"not applicable"}[self]
class MatchLevel(IntEnum):
    """Hierarchy: Lower = Better Match"""
    IDENTICAL = 1
    COMPATIBLE = 2
    PARTIAL = 3
    # PARTIAL_PROXIMATE = 5
    # PARTIAL_TENTATIVE = 6
    NOT_APPLICABLE = 4

    def to_str(self) -> str:
        mapping = {
            1: "Identical Match",
            2: "Compatible Match",
            3: "Partial Match",
            4: "Not Applicable"
        }
        return mapping[self]


class DATA_TYPE(str, Enum):
    """Kept for backward compatibility."""
    STRING = "str"
    FLOAT = "float"
    INTEGER = "int"
    DATETIME = "datetime"

class TransformationType(str, Enum):
    """Type of transformation needed for harmonization.
    
    Using str Enum so values work as plain strings everywhere (Pydantic, dicts, etc.).
    """
    NONE = "None"
    UNIT_CONVERSION = "Require Transformation forUnit Conversion"
    UNIT_ALIGNMENT = "Require Transformation for Unit Alignment"  # one has unit, other doesn't
    VALUE_NORMALIZATION = "Require Transformation for Categorical Value Normalization"
    AGGREGATION_OR_EXPANSION = "Require Transformation for Categorical Aggregation/Expansion"
    BINARY_EXTRACTION = "Require Transformation to compute binary variblable "
    TIMEPOINT_ALIGNMENT = "Require Transformation for Timepoints Alignment"
    DERIVATION = "Require Transformation to compute derived variable"
    MANUAL_REVIEW = "Require Manual Review before harmonization"


class StatisticalType(str, Enum):
    """Statistical classification of clinical variables."""
    CONTINUOUS = "continuous_variable"
    BINARY = "binary_class_variable"
    MULTI_CLASS = "multi_class_variable"
    QUALITATIVE = "qualitative_variable"
    DERIVED = "derived_variable"
    
    @classmethod
    def from_string(cls, value: Optional[str]) -> Optional["StatisticalType"]:
        """Parse statistical type from string, handling common variations."""
        if not value:
            return None
        value_lower = value.lower().strip()
        
        # Direct enum value match
        for member in cls:
            if member.value == value_lower:
                return member
        
        # Common aliases
        aliases = {
            "continuous": cls.CONTINUOUS,
            "binary": cls.BINARY,
            "categorical": cls.MULTI_CLASS,
            "multiclass": cls.MULTI_CLASS,
            "multi-class": cls.MULTI_CLASS,
            "qualitative": cls.QUALITATIVE,
            "derived": cls.DERIVED,
        }
        return aliases.get(value_lower)


# # =============================================================================
# # STATISTICS MODEL
# # =============================================================================

class Statistics(BaseModel):
    """Statistical profile of a variable.
    
    Maps to STATO ontology concepts where available:
    - mean: stato:mean
    - stddev: stato:standard_deviation
    - min_val/max_val: stato:minimum_value/stato:maximum_value
    - median: stato:median
    - mode: stato:mode
    - iqr: stato:interquartile_range
    """
    
    mean: Optional[float] = Field(default=None, description="Arithmetic mean (stato:mean)")
    stddev: Optional[float] = Field(default=None, description="Standard deviation (stato:standard_deviation)")
    min_val: Optional[float] = Field(default=None, description="Minimum value (stato:minimum_value)")
    max_val: Optional[float] = Field(default=None, description="Maximum value (stato:maximum_value)")
    median: Optional[float] = Field(default=None, description="Median value (stato:median)")
    mode: Optional[Any] = Field(default=None, description="Mode value (stato:mode)")
    iqr: Optional[float] = Field(default=None, description="Interquartile range (stato:interquartile_range)")
    z_score: Optional[float] = Field(default=None, description="Z-score (stato:z_score)")
    
    class Config:
        extra = "allow"  # Allow additional statistical fields
    
    @model_validator(mode='before')
    @classmethod
    def clean_nan_values(cls, data):
        """Convert NaN/Inf from DataFrame to None."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    data[key] = None
        return data
    
    @property
    def has_range(self) -> bool:
        """Check if min/max range is available."""
        return self.min_val is not None and self.max_val is not None
    
    @property
    def range_tuple(self) -> Optional[tuple[float, float]]:
        """Return (min, max) tuple if both available."""
        if self.has_range:
            return (self.min_val, self.max_val)
        return None
    
    def is_compatible_range(self, other: "Statistics", tolerance: float = 0.1) -> bool:
        """Check if ranges overlap with tolerance for unit conversion detection."""
        if not (self.has_range and other.has_range):
            return True  # Cannot determine incompatibility
        
        # Check for overlap
        return not (self.max_val < other.min_val or other.max_val < self.min_val)


# # =============================================================================
# # VARIABLE NODE MODEL
# # =============================================================================

class VariableNode(BaseModel):
    """
    Pydantic model representing a clinical variable as a typed subgraph 
    with connections to OMOP concepts.
    
    This model captures:
    - Variable identification (name, study context)
    - Original variable label/description (non-standardized)
    - Primary OMOP concept alignment (main_id, main_label)
    - Contextual OMOP concepts (context_ids, context_labels) - composite codes
    - Categorical value mappings (category_ids, category_labels)
    - Statistical properties (statistical_type, statistics)
    - Temporal context (visit/timepoint)
    - Unit information
    
    Designed for bidirectional conversion with Dict[str, Any] used in:
    - CandidateContext in constraints.py
    - Element dicts in neuro_matcher.py
    - DataFrame rows in run.py
    """
    
    # --- Identification ---
    name: str = Field(..., description="Variable identifier (dc:identifier)")
    study: Optional[str] = Field(default=None, description="Source study name")
    role: Optional[str] = Field(default=None, description="Role in matching (source or target)")
    # either source or target 
    
    description: str = Field(default="", description="Original variable label/description (non-standardized)")
    # --- Primary OMOP Concept (skos:hasCloseMatch) ---
    main_id: Optional[int] = Field(default=None, description="Primary OMOP concept ID (skos:hasCloseMatch)")
    main_label: str = Field(default="", description="Primary concept label (rdfs:label)")
    main_code: Optional[str] = Field(default=None, description="Primary concept code (e.g., SNOMED, LOINC code)")
    
    # --- Context Concepts (composite codes) ---
    context_ids: List[int] = Field(default_factory=list, description="Context OMOP concept IDs")
    context_labels: List[str] = Field(default_factory=list, description="Context concept labels")
    context_codes: List[str] = Field(default_factory=list, description="Context concept codes")
    # --- Categorical Values (for categorical variables) ---
    category_ids: List[int] = Field(default_factory=list, description="Categorical value OMOP IDs")
    category_labels: List[str] = Field(default_factory=list, description="Categorical value labels")
    category_codes: List[str] = Field(default_factory=list, description="Categorical value codes")
    original_categories: List[str] = Field(default_factory=list, description="Original categorical values from source")
    
    # --- Statistical Properties ---
    statistical_type: Optional[StatisticalType] = Field(
        default=None, 
        description="Statistical classification"
    )
    statistics: Statistics = Field(default_factory=Statistics, description="Statistical profile")
    
    # --- Unit Information ---
    unit: str = Field(default="", description="UCUM unit ID or unit label")
    
    # --- Temporal Context ---
    visit: str = Field(
        default="baseline", 
        description="Timepoint/visit context. Use 'undetermined' for event_date/visit_date patterns"
    )
    data_type: Optional[str] = Field(default=None, description="Data type (int, float, str, datetime)")
    # --- Similarity Score (for neural matching) ---
    sim_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Similarity score from embedding match")
    
    # --- Category ---
    category: Optional[str] = Field(default=None, description="Variable category/domain (e.g., 'demographics', 'labs')")
    
    class Config:
        use_enum_values = True  # Serialize enums as their values
        extra = "allow"  # Allow additional fields for extensibility
    
    # =========================================================================
    # VALIDATORS
    # =========================================================================
    
    @model_validator(mode='before')
    @classmethod
    def clean_nan_values(cls, data):
        """Convert NaN/Inf/None from DataFrame rows to safe defaults for proper validation."""
        if isinstance(data, dict):
            # Non-optional str fields that need "" instead of None
            str_default_fields = {'unit', 'main_label'}
            for key, value in data.items():
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    data[key] = "" if key in str_default_fields else None
                elif value is None and key in str_default_fields:
                    data[key] = ""
        return data
    
    @field_validator('statistical_type', mode='before')
    @classmethod
    def parse_statistical_type(cls, v):
        """Parse statistical type from string or enum."""
        if v is None:
            return None
        if isinstance(v, StatisticalType):
            return v
        if isinstance(v, str):
            return StatisticalType.from_string(v)
        return None
    
    @field_validator('context_labels', 'category_labels', 'original_categories', mode='before')
    @classmethod
    def parse_pipe_separated(cls, v):
        """Parse pipe-separated string into list."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if x]
        if isinstance(v, str):
            if '||' in v:
                return [x.strip() for x in v.split('||') if x.strip()]
            return [v.strip()] if v.strip() else []
        return []
    
    @field_validator('context_ids', 'category_ids', mode='before')
    @classmethod
    def parse_id_list(cls, v):
        """Parse list of IDs from various formats."""
        if v is None:
            return []
        if isinstance(v, list):
            result = []
            for x in v:
                try:
                    result.append(int(x))
                except (ValueError, TypeError):
                    continue
            return result
        if isinstance(v, str):
            result = []
            for x in v.replace('||', ';').split(';'):
                try:
                    result.append(int(x.strip()))
                except (ValueError, TypeError):
                    continue
            return result
        return []
    
    @field_validator('role', mode='before')
    @classmethod
    def validate_role(cls, v):
        """Validate role is either 'source' or 'target'."""
        if v is None:
            return None
        v_str = str(v).lower().strip()
        if v_str in ('source', 'target'):
            return v_str
        raise ValueError("Role must be either 'source' or 'target'")
    @field_validator('visit', mode='before')
    @classmethod
    def normalize_visit(cls, v):
        """Normalize visit/timepoint string."""
        if v is None:
            return "baseline"
        v_str = str(v).lower().strip()
        if not v_str or v_str in ('none', 'nan', 'null'):
            return "baseline"
        return v_str
    
    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================
    
    @computed_field
    @property
    def is_categorical(self) -> bool:
        """Check if variable is categorical (binary or multi-class)."""
        if self.statistical_type is None:
            return bool(self.category_labels or self.original_categories)
        return self.statistical_type in (
            StatisticalType.BINARY, 
            StatisticalType.MULTI_CLASS
        )
    
    @computed_field
    @property
    def is_continuous(self) -> bool:
        """Check if variable is continuous."""
        if self.statistical_type is None:
            return not self.is_categorical
        return self.statistical_type == StatisticalType.CONTINUOUS
    
    @computed_field
    @property
    def is_derived(self) -> bool:
        """Check if variable is derived/computed."""
        # if self.statistical_type == StatisticalType.DERIVED:
        #     return True
        return self.name.startswith("derived_") or self.name.endswith("_derived")
    
    @computed_field
    @property
    def has_omop_mapping(self) -> bool:
        """Check if variable has primary OMOP concept mapping."""
        return self.main_id is not None and self.main_id > 0
    
    @computed_field
    @property
    def composite_code_labels(self) -> str:
        """Return pipe-joined context labels for compatibility with existing code."""
        if self.context_labels:
            return '|'.join(self.context_labels)
        return self.main_label
    
    @computed_field
    @property
    def is_visit_undetermined(self) -> bool:
        """Check if visit/timepoint is undetermined."""
        undetermined_hints = {'event_date', 'visit_date', 'undetermined', 'unknown'}
        return any(hint in self.visit.lower() for hint in undetermined_hints)
    
    # =========================================================================
    # CONVERSION METHODS
    # =========================================================================
    
    def to_source_dict(self) -> Dict[str, Any]:
        """
        Convert to source-side dict format used by CandidateContext in constraints.py.
        
        Maps to keys: study, slabel, somop_id, stats_type, unit, visit, 
                      original_categories, categories_labels, categories_omop_ids,
                      composite_code_labels, sim_score, min_value, max_value
        """
        return {
            "study": self.study or "",
            "slabel": self.main_label,
            "somop_id": self.main_id,
            "scode": self.main_code,
            "stats_type": self.statistical_type.value if self.statistical_type else "",
            "unit": self.unit,
            "visit": self.visit,
            "original_categories": '|'.join(self.original_categories) if self.original_categories else "",
            "categories_labels": '|'.join(self.category_labels) if self.category_labels else "",
            "categories_omop_ids": '|'.join(str(x) for x in self.category_ids) if self.category_ids else "",
            "composite_code_labels": self.composite_code_labels,
            "sim_score": self.sim_score or 0.0,
            "min_value": self.statistics.min_val,
            "max_value": self.statistics.max_val,
        }
    
    def to_target_dict(self) -> Dict[str, Any]:
        """
        Convert to target-side dict format used by CandidateContext in constraints.py.
        
        Maps to keys: study, tlabel, tomop_id, stats_type, unit, visit, etc.
        """
        return {
            "study": self.study or "",
            "source": self.name,
            "tlabel": self.main_label,
            "tomop_id": self.main_id,
            "tcode": self.main_code,
            "stats_type": self.statistical_type.value if self.statistical_type else "",
            "unit": self.unit,
            "visit": self.visit,
            "original_categories": '|'.join(self.original_categories) if self.original_categories else "",
            "categories_labels": '|'.join(self.category_labels) if self.category_labels else "",
            "categories_omop_ids": '|'.join(str(x) for x in self.category_ids) if self.category_ids else "",
            "composite_code_labels": self.composite_code_labels,
            "sim_score": self.sim_score or 0.0,
            "min_value": self.statistics.min_val,
            "max_value": self.statistics.max_val,
        }
    
    def to_element_dict(self, role: str = "source") -> Dict[str, Any]:
        """
        Convert to element dict format used by NeuroSymbolicMatcher.
        
        Args:
            role: Either "source" or "target" to set the variable name key
        """
        return {
            "omop_id": self.main_id,
            "source": self.name,
            "code": self.main_code,
            "code_label": self.main_label,
            "category": self.category,
            "visit": self.visit,
            role: self.name,  # "source": name or "target": name
            "stats_type": self.statistical_type.value if self.statistical_type else "",
            "unit_label": self.unit,
        }
    
    def to_match_row_dict(self, target: "VariableNode", relation: str = "Symbolic Match") -> Dict[str, Any]:
        """
        Create a match row dict combining source and target for DataFrame creation.
        
        Used when building match results in neuro_matcher.py and run.py.
        """
        return {
            "source": self.name,
            "target": target.name,
            "somop_id": self.main_id,
            "tomop_id": target.main_id,
            "scode": self.main_code,
            "slabel": self.main_label,
            "tcode": target.main_code,
            "tlabel": target.main_label,
            "source_visit": self.visit,
            "target_visit": target.visit,
            "category": self.category or target.category,
            "mapping_relation": relation,
        }
    
    @classmethod
    def from_source_row(cls, row: Dict[str, Any], study: Optional[str] = None) -> "VariableNode":
        """
        Create VariableNode from a DataFrame row (source-side columns).
        
        Parses columns: source, slabel, somop_id, source_type, source_unit, 
                       source_visit, source_original_categories, etc.
        """
        stats = Statistics(
            min_val=row.get("source_min_val"),
            max_val=row.get("source_max_val"),
        )
        
        return cls(
            name=row.get("source", ""),
            study=study or row.get("study"),
            main_id=row.get("somop_id"),
            main_label=row.get("slabel", ""),
            main_code=row.get("scode"),
            context_labels=row.get("source_composite_code_labels", ""),
            category_labels=row.get("source_categories_labels", ""),
            category_ids=row.get("source_categories_omop_ids", ""),
            original_categories=row.get("source_original_categories", ""),
            statistical_type=row.get("source_type"),
            statistics=stats,
            unit=row.get("source_unit", ""),
            visit=row.get("source_visit", "baseline"),
            sim_score=row.get("sim_score"),
            category=row.get("category"),
            context_match_type = row.get("context_match_type"),
            data_type=row.get("source_data_type"),
        )
    
    @classmethod
    def from_target_row(cls, row: Dict[str, Any], study: Optional[str] = None) -> "VariableNode":
        """
        Create VariableNode from a DataFrame row (target-side columns).
        
        Parses columns: target, tlabel, tomop_id, target_type, target_unit,
                       target_visit, target_original_categories, etc.
        """
        stats = Statistics(
            min_val=row.get("target_min_val"),
            max_val=row.get("target_max_val"),
        )
        
        return cls(
            name=row.get("target", ""),
            study=study or row.get("study"),
            main_id=row.get("tomop_id"),
            main_label=row.get("tlabel", ""),
            main_code=row.get("tcode"),
            context_labels=row.get("target_composite_code_labels", ""),
            category_labels=row.get("target_categories_labels", ""),
            category_ids=row.get("target_categories_omop_ids", ""),
            original_categories=row.get("target_original_categories", ""),
            statistical_type=row.get("target_type"),
            statistics=stats,
            unit=row.get("target_unit", ""),
            visit=row.get("target_visit", "baseline"),
            sim_score=row.get("sim_score"),
            category=row.get("category"),
            context_match_type = row.get("context_match_type"),
            data_type=row.get("target_data_type"),
        )
    
    @classmethod
    def from_element_dict(cls, elem: Dict[str, Any], role: str = "source") -> "VariableNode":
        """
        Create VariableNode from element dict used in NeuroSymbolicMatcher.
        """
        return cls(
            name=elem.get(role, ""),
            main_id=elem.get("omop_id"),
            main_label=elem.get("code_label", ""),
            main_code=elem.get("code"),
            category=elem.get("category"),
            visit=elem.get("visit", "baseline"),
            statistical_type=elem.get("stats_type"),
            unit=elem.get("unit_label", ""),
        )
    
    @classmethod
    def from_sparql_binding(
        cls, 
        binding: Dict[str, Any],
        var_name: str,
        visit: str,
        role: str = "source"
    ) -> "VariableNode":
        """
        Create VariableNode from SPARQL query binding.
        
        Used when parsing SPARQL results in run.py._parse_sparql_bindings().
        """
        return cls(
            name=var_name,
            main_id=int(binding.get("omop_id", {}).get("value", 0)),
            main_label=binding.get("code_label", {}).get("value", ""),
            main_code=binding.get("code_value", {}).get("value"),
            visit=visit,
            category=binding.get(f"{role}_domain", {}).get("value", "").strip().lower(),
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def visits_align_with(self, other: "VariableNode") -> bool:
        """
        Check if visits/timepoints align using the same logic as ConstraintSolver.
        
        Two visits align if:
        1. They are exactly equal, OR
        2. Neither contains date hints that conflict
        """
        date_hints = {'event_date', 'visit_date', 'index_date', 'date'}
        
        s_low = self.visit.lower()
        t_low = other.visit.lower()
        
        # Exact match
        if s_low == t_low:
            return True
        
        # Check for conflicting date hints
        for hint in date_hints:
            if hint in s_low and hint in t_low:
                return True
            elif hint in s_low:
                return False
            elif hint in t_low:
                return False
        
        return s_low == t_low
    
    def category_overlap_with(self, other: "VariableNode") -> Set[str]:
        """Get overlapping category labels between two variables."""
        self_cats = {c.lower().strip() for c in self.category_labels}
        other_cats = {c.lower().strip() for c in other.category_labels}
        return self_cats & other_cats
    
    def __repr__(self) -> str:
        """Readable string representation."""
        type_str = self.statistical_type.value if self.statistical_type else "unknown"
        return f"VariableNode(name='{self.name}', main_id={self.main_id}, type={type_str}, visit='{self.visit}')"
    
    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash((self.name, self.main_id, self.visit, self.study))
    
    def __eq__(self, other: object) -> bool:
        """Equality check based on identifying fields."""
        if not isinstance(other, VariableNode):
            return False
        return (
            self.name == other.name and 
            self.main_id == other.main_id and 
            self.visit == other.visit and
            self.study == other.study
        )


# =============================================================================
# VARIABLE PAIR MODEL (for match results)
# =============================================================================

class VariablePair(BaseModel):
    """
    Represents a candidate or confirmed variable pair for harmonization.
    
    Used to encapsulate source-target pairs with their relationship metadata,
    useful for structured results from the constraint solver.
    """
    
    source: VariableNode
    target: VariableNode
    mapping_relation: str = Field(default="unknown", description="Type of mapping (Symbolic, Neural, etc.)")
    sim_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    explanation: Optional[str] = Field(default=None, description="Explanation for the mapping decision")    
    harmonization_type: Optional[TransformationType] = Field(
        default=None, 
        description="Type of harmonization needed (unit conversion, value normalization, etc.)"
    )
    
    class Config:
        extra = "allow"
    
    def to_match_dict(self) -> Dict[str, Any]:
        """Convert to match row dict for DataFrame creation."""
        base = self.source.to_match_row_dict(self.target, self.mapping_relation)
        if self.sim_score is not None:
            base["sim_score"] = self.sim_score
        return base
    
    # @property
    # def visits_aligned(self) -> bool:
    #     """Check if source and target visits align."""
    #     return self.source.visits_align_with(self.target)
    
    @property
    def omop_ids_match(self) -> bool:
        """Check if source and target have same OMOP ID."""
        return (
            self.source.main_id is not None and 
            self.source.main_id == self.target.main_id
        )


# =============================================================================
# MATCH RESULT MODEL
# =============================================================================

class MatchResult(BaseModel):
    """
    Complete match result combining source, target, and constraint evaluation.
    
    This is the final output structure after constraint solving.
    """
    source: VariableNode
    target: VariableNode
    mapping_relation: str = Field(default="unknown")
    sim_score: float = Field(default=0.0, ge=0.0, le=1.0)
    harmonization_status: str = Field(default="")
    transformation_rule: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
    
    def to_dataframe_row(self) -> Dict[str, Any]:
        """Convert to flat dict for DataFrame creation."""
        return {
            # Identifiers
            "source": self.source.name,
            "target": self.target.name,
            
            # OMOP mappings
            "somop_id": self.source.main_id,
            "tomop_id": self.target.main_id,
            "scode": self.source.main_code,
            "tcode": self.target.main_code,
            "slabel": self.source.main_label,
            "tlabel": self.target.main_label,
            
            # Visits
            "source_visit": self.source.visit,
            "target_visit": self.target.visit,
            
            # Types and units
            "source_type": self.source.statistical_type.value if self.source.statistical_type else "",
            "target_type": self.target.statistical_type.value if self.target.statistical_type else "",
            "source_unit": self.source.unit,
            "target_unit": self.target.unit,
            
            # Categories
            "source_categories_labels": "|".join(self.source.category_labels),
            "target_categories_labels": "|".join(self.target.category_labels),
            "source_categories_omop_ids": "|".join(str(x) for x in self.source.category_ids),
            "target_categories_omop_ids": "|".join(str(x) for x in self.target.category_ids),
            "source_original_categories": "|".join(self.source.original_categories),
            "target_original_categories": "|".join(self.target.original_categories),
            
            # Composite codes
            "source_composite_code_labels": self.source.composite_code_labels,
            "target_composite_code_labels": self.target.composite_code_labels,
            "source_composite_code_omop_ids": "|".join(str(x) for x in self.source.context_ids),
            "target_composite_code_omop_ids": "|".join(str(x) for x in self.target.context_ids),
            
            # Statistics
            "source_min_val": self.source.statistics.min_val,
            "source_max_val": self.source.statistics.max_val,
            "target_min_val": self.target.statistics.min_val,
            "target_max_val": self.target.statistics.max_val,
            
            # Match metadata
            "category": self.source.category or self.target.category,
            "mapping_relation": self.mapping_relation,
            "sim_score": self.sim_score,
            "harmonization_status": self.harmonization_status,
            "transformation_rule": self.transformation_rule,
        }


# =============================================================================
# VARIABLE COLLECTION (for study-level operations)
# =============================================================================

class VariableCollection(BaseModel):
    """
    Collection of variables from a single study.
    
    Provides efficient lookups by OMOP ID, name, and visit for matching operations.
    """
    study: str
    variables: List[VariableNode] = Field(default_factory=list)
    
    # Indexes built on first access
    _by_omop_id: Optional[Dict[int, List[VariableNode]]] = None
    _by_name: Optional[Dict[str, VariableNode]] = None
    _by_visit:Optional[Dict[str, List[VariableNode]]] = None
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
    
    def model_post_init(self, __context) -> None:
        """Build indexes after initialization."""
        self._build_indexes()
    
    def _build_indexes(self) -> None:
        """Build lookup indexes."""
        self._by_omop_id = {}
        self._by_name = {}
        self._by_visit = {}
        for var in self.variables:
            var.study = self.study  # Ensure study is set
            
            # Index by OMOP ID (multiple vars can share same OMOP)
            if var.main_id:
                if var.main_id not in self._by_omop_id:
                    self._by_omop_id[var.main_id] = []
                self._by_omop_id[var.main_id].append(var)
            
            # Index by name (unique)
            self._by_name[var.name] = var
            # we need to combine all visits and append them as str for a variable ???
    
    def get_by_omop_id(self, omop_id: int) -> List[VariableNode]:
        """Get all variables with given OMOP ID."""
        if self._by_omop_id is None:
            self._build_indexes()
        return self._by_omop_id.get(omop_id, [])
    
    def get_by_name(self, name: str) -> Optional[VariableNode]:
        """Get variable by name."""
        if self._by_name is None:
            self._build_indexes()
        return self._by_name.get(name)
    
    @property
    def omop_ids(self) -> Set[int]:
        """Get all unique OMOP IDs in collection."""
        if self._by_omop_id is None:
            self._build_indexes()
        return set(self._by_omop_id.keys())
    
    def __len__(self) -> int:
        return len(self.variables)
    
    def __iter__(self):
        return iter(self.variables)
    
    @classmethod
    def from_sparql_bindings(
        cls, 
        study: str,
        alignment_bindings: List[Dict[str, Any]],
        stats_bindings: List[Dict[str, Any]],
        role: str = "source"
    ) -> "VariableCollection":
        """
        Create VariableCollection from SPARQL query results.
        
        Args:
            study: Study name
            alignment_bindings: Results from build_alignment_query
            stats_bindings: Results from build_statistic_query
            role: "source" or "target" to determine which columns to parse
        """
        # Build stats lookup by identifier
        stats_map: Dict[str, Dict] = {}
        for binding in stats_bindings:
            identifier = binding.get("identifier", {}).get("value", "")
            if identifier:
                stats_map[identifier] = binding
        
        variables: List[VariableNode] = []
        seen: Set[tuple] = set()  # (name, visit) to deduplicate
        
        # Parse alignment bindings
        data_key = f"{role}_data"
        domain_key = f"{role}_domain"
        
        for binding in alignment_bindings:
            omop_id = int(binding.get("omop_id", {}).get("value", 0))
            code_label = binding.get("code_label", {}).get("value", "")
            code_value = binding.get("code_value", {}).get("value", "")
            category = binding.get(domain_key, {}).get("value", "").strip().lower()
            
            # Parse variable names and visits from combined data
            raw_data = binding.get(data_key, {}).get("value", "")
            if not raw_data:
                continue
            
            for item in raw_data.split("||"):
                item = item.strip()
                if not item:
                    continue
                
                # Parse "varname[visit]" format
                if "[" in item and item.endswith("]"):
                    parts = item.rsplit("[", 1)
                    var_name = parts[0].strip()
                    visit = parts[1].strip(" ]")
                else:
                    var_name = item
                    visit = "baseline"
                
                # Deduplicate
                key = (var_name, visit)
                if key in seen:
                    continue
                seen.add(key)
                
                # Get stats for this variable
                stats_data = stats_map.get(var_name, {})
                
                # Build Statistics
                stats = Statistics(
                    min_val=_safe_float(stats_data.get("min_val", {}).get("value")),
                    max_val=_safe_float(stats_data.get("max_val", {}).get("value")),
                )
                
                # Parse categories
                cat_labels = _safe_str(stats_data.get("all_cat_labels", {}).get("value", ""))
                cat_omop_ids = _safe_str(stats_data.get("cat_omop_ids", {}).get("value", ""))
                original_cats = _safe_str(stats_data.get("all_original_cat_values", {}).get("value", ""))
                
                # Parse composite codes
                composite_labels = _safe_str(stats_data.get("code_label", {}).get("value", ""))
                
                var = VariableNode(
                    name=var_name,
                    study=study,
                    main_id=omop_id,
                    main_label=code_label,
                    main_code=code_value,
                    context_labels=composite_labels,
                    category_labels=cat_labels,
                    category_ids=cat_omop_ids,
                    original_categories=original_cats,
                    statistical_type=_safe_str(stats_data.get("stat_label", {}).get("value", "")),
                    statistics=stats,
                    unit=_safe_str(stats_data.get("unit_label", {}).get("value", "")),
                    visit=visit,
                    category=category,
                )
                variables.append(var)
        
        return cls(study=study, variables=variables)


# =============================================================================
# VARIABLE PROFILE ROW (for SPARQL fetch validation)
# =============================================================================

class VariableProfileRow(BaseModel):
    """Pydantic model for a single variable profile fetched from the KG.
    
    Validates and normalizes the raw SPARQL result before it becomes
    DataFrame columns. Used by VariableProfile._fetch_chunk().
    
    Also serves as single source of truth for the column rename mappings
    used when merging profile data into the alignment DataFrame.
    """
    identifier: str = Field(..., description="Variable identifier (dc:identifier)")
    stat_label: Optional[str] = Field(default=None, description="Statistical type label")
    unit_label: Optional[str] = Field(default=None, description="Unit label (UCUM)")
    data_type: Optional[str] = Field(default=None, description="Data type (int, float, str, datetime)")
    categories_labels: Optional[str] = Field(default=None, description="Pipe-separated category labels")
    categories_omop_ids: Optional[str] = Field(default=None, description="Pipe-separated category OMOP IDs")
    original_categories: Optional[str] = Field(default=None, description="Pipe-separated original category values")
    composite_code_labels: Optional[str] = Field(default=None, description="Pipe-separated composite code labels")
    composite_code_values: Optional[str] = Field(default=None, description="Pipe-separated composite code values")
    composite_code_omop_ids: Optional[str] = Field(default=None, description="Pipe-separated composite OMOP IDs")
    min_val: Optional[str] = Field(default=None, description="Minimum value (as string from SPARQL)")
    max_val: Optional[str] = Field(default=None, description="Maximum value (as string from SPARQL)")
    
    class Config:
        extra = "allow"

    # Maps profile field names → prefixed DataFrame column names.
    # "identifier" becomes the merge key ("source" or "target").
    _COLUMN_MAP: ClassVar[Dict[str, str]] = {
        "identifier":            "{side}",
        "stat_label":            "{side}_type",
        "unit_label":            "{side}_unit",
        "data_type":             "{side}_data_type",
        "original_categories":   "{side}_original_categories",
        "categories_labels":     "{side}_categories_labels",
        "categories_omop_ids":   "{side}_categories_omop_ids",
        "composite_code_labels": "{side}_composite_code_labels",
        "composite_code_omop_ids": "{side}_composite_code_omop_ids",
        "min_val":               "{side}_min_val",
        "max_val":               "{side}_max_val",
    }

    @classmethod
    def column_map(cls, side: str) -> Dict[str, str]:
        """Build the column rename dict for a given side ('source' or 'target').
        
        >>> VariableProfileRow.column_map("source")
        {'identifier': 'source', 'stat_label': 'source_type', ...}
        """
        return {k: v.format(side=side) for k, v in cls._COLUMN_MAP.items()}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_float(val: Any) -> Optional[float]:
    """Safely convert value to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_str(val: Any) -> str:
    """Safely convert value to string."""
    if val is None:
        return ""
    return str(val).strip()
