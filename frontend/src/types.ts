export interface Cohort {
  cohort_id: string;
  cohort_type: string;
  cohort_email: string;
  institution: string;
  variables: { [key: string]: Variable };
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
  concept_id: string;
  omop_domain: string;
  index: number;
  categories: Category[];
  [key: string]: any;
}

export interface Category {
  value: string;
  label: string;
}