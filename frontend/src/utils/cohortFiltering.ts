import type {Cohort} from '@/types';

export type SearchMode = 'or' | 'and' | 'exact';
export type SearchScope = 'cohorts' | 'variables' | 'all';

const FIELD_TO_SECTION: Record<string, string> = {
  cohort_id: 'cohort name',
  institution: 'institution',
  study_type: 'study type',
  study_design: 'study design',
  study_objective: 'study objective',
  morbidity: 'morbidity',
  study_participants: 'study participants',
  study_population: 'study population',
  administrator: 'administrator',
  population_location: 'population location',
  primary_outcome_spec: 'primary outcome specification',
  secondary_outcome_spec: 'secondary outcome specification',
  interventions: 'interventions',
  comparator: 'comparator',
  race_ethnicity: 'race/ethnicity',
  part_of_study: 'part of study'
};

const SEARCHABLE_COHORT_FIELDS = Object.keys(FIELD_TO_SECTION);
const SEARCHABLE_VARIABLE_FIELDS = [
  'var_name',
  'var_label',
  'concept_name',
  'mapped_label',
  'omop_domain',
  'concept_code',
  'omop_id'
];
const SEARCHABLE_CATEGORY_FIELDS = ['value', 'label', 'mapped_label'];

export interface CohortSearchResults {
  matchedCohorts: {cohortId: string; sections: string[]}[];
  variablesByCohort: Record<string, string[]>;
  totalVariables: number;
}

export interface CohortFilterOptions {
  selectedStudyTypes: ReadonlySet<string>;
  selectedInstitutes: ReadonlySet<string>;
  searchTerms: string[];
  searchMode: SearchMode;
  searchScope: SearchScope;
}

export const cohortElementId = (cohortId: string): string => `cohort-${cohortId}`;

export const normalizeSearchText = (text: string): string =>
  text
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .replace(/[_\-.,—]/g, ' ');

export const matchesSearchTerms = (
  text: string | number | null | undefined,
  searchTerms: string[],
  searchMode: SearchMode
): boolean => {
  if (!text || searchTerms.length === 0) return false;
  const normalizedText = normalizeSearchText(String(text).toLowerCase());

  if (searchMode === 'exact') {
    return normalizedText.includes(normalizeSearchText(searchTerms.join(' ').toLowerCase()));
  }
  if (searchMode === 'and') {
    return searchTerms.every(term => normalizedText.includes(normalizeSearchText(term.toLowerCase())));
  }
  return searchTerms.some(term => normalizedText.includes(normalizeSearchText(term.toLowerCase())));
};

const cohortMetadataMatches = (
  cohortId: string,
  cohort: Cohort,
  searchTerms: string[],
  searchMode: SearchMode
): boolean => {
  const cohortWithId = {...cohort, cohort_id: cohortId};
  return SEARCHABLE_COHORT_FIELDS.some(field =>
    matchesSearchTerms((cohortWithId as Record<string, unknown>)[field] as string | number | null, searchTerms, searchMode)
  );
};

const matchingVariableNames = (cohort: Cohort, searchTerms: string[], searchMode: SearchMode): string[] =>
  Object.entries(cohort.variables || {})
    .filter(([variableName, variableData]) => {
      const variableWithName = {...variableData, var_name: variableName} as Record<string, unknown>;
      const variableMatches = SEARCHABLE_VARIABLE_FIELDS.some(field =>
        matchesSearchTerms(variableWithName[field] as string | number | null, searchTerms, searchMode)
      );
      if (variableMatches) return true;
      return variableData.categories?.some(category =>
        SEARCHABLE_CATEGORY_FIELDS.some(field =>
          matchesSearchTerms(category[field as keyof typeof category] as string | number | null, searchTerms, searchMode)
        )
      );
    })
    .map(([variableName]) => variableName);

export const collectCohortSearchResults = (
  cohortsData: Record<string, Cohort>,
  searchTerms: string[],
  searchMode: SearchMode,
  searchScope: SearchScope
): CohortSearchResults => {
  const results: CohortSearchResults = {
    matchedCohorts: [],
    variablesByCohort: {},
    totalVariables: 0
  };
  if (searchTerms.length === 0) return results;

  Object.entries(cohortsData).forEach(([cohortId, cohort]) => {
    if (searchScope === 'cohorts' || searchScope === 'all') {
      const cohortWithId = {...cohort, cohort_id: cohortId};
      const sections = SEARCHABLE_COHORT_FIELDS.filter(field =>
        matchesSearchTerms(
          (cohortWithId as Record<string, unknown>)[field] as string | number | null,
          searchTerms,
          searchMode
        )
      ).map(field => FIELD_TO_SECTION[field]);
      if (sections.length > 0) results.matchedCohorts.push({cohortId, sections});
    }

    if (searchScope === 'variables' || searchScope === 'all') {
      const variables = matchingVariableNames(cohort, searchTerms, searchMode);
      if (variables.length > 0) {
        results.variablesByCohort[cohortId] = variables;
        results.totalVariables += variables.length;
      }
    }
  });

  return results;
};

export const filterCohorts = (
  cohortsData: Record<string, Cohort>,
  options: CohortFilterOptions
): Cohort[] => {
  const {
    selectedStudyTypes,
    selectedInstitutes,
    searchTerms,
    searchMode,
    searchScope
  } = options;

  return Object.entries(cohortsData)
    .filter(([cohortId, cohort]) => {
      if (selectedStudyTypes.size > 0 && !selectedStudyTypes.has(cohort.study_design || '')) return false;
      if (selectedInstitutes.size > 0 && !selectedInstitutes.has(cohort.institution)) return false;
      if (searchTerms.length === 0) return true;

      const metadataMatches = cohortMetadataMatches(cohortId, cohort, searchTerms, searchMode);
      const variablesMatch = matchingVariableNames(cohort, searchTerms, searchMode).length > 0;
      if (searchScope === 'cohorts') return metadataMatches;
      if (searchScope === 'variables') return variablesMatch;
      return metadataMatches || variablesMatch;
    })
    .map(([, cohort]) => cohort);
};
