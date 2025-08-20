export interface Cohort {
  cohort_id: string;
  cohort_type: string;
  cohort_email: string[];
  institution: string;
  study_type: string;
  study_participants: number;
  study_population: string;
  study_duration: string;
  study_ongoing: string;
  study_objective: string;
  primary_outcome_spec: string;
  secondary_outcome_spec: string;
  study_start: string;
  study_end: string;
  variables: {[key: string]: Variable};
}

export interface Variable {
  var_name: string;
  var_label: string;
  var_type: string;
  count: number;
  na: number;
  max: string;
  min: string;
  units: string;
  visits: string;
  formula: string;
  definition: string;
  omop_domain: string;
  index: number;
  concept_id: string;
  mapped_id: string | null;
  mapped_label: string | null;
  categories: Category[];
  [key: string]: any;
}

export interface Category {
  value: string;
  label: string;
  concept_id: string | null;
  mapped_id: string | null;
  mapped_label: string | null;
}

export interface Concept {
  label: string;
  domain: string;
  id: string;
  vocabulary: string;
}

export interface AutocompleteConceptProps {
  onSelect: (suggestion: Concept) => void;
  query?: string;
  value?: string;
  domain?: string;
  index?: string;
  cohortId?: string;
  tooltip?: string;
  canEdit?: boolean;
}
