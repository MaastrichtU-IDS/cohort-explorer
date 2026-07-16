import type {SearchMode, SearchScope} from '@/utils/cohortFiltering';

export type VariableRecord = Record<string, any>;
export type FilteredVariable = VariableRecord & {var_name: string};

export interface VariableFilterOptions {
  selectedOMOPDomains: ReadonlySet<string>;
  selectedDataTypes: ReadonlySet<string>;
  selectedCategoryTypes: ReadonlySet<string>;
  selectedVisitTypes: ReadonlySet<string>;
  showOnlyOutcomes: boolean;
  searchScope?: SearchScope;
  searchTerms: string[];
  searchMode: SearchMode;
}

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
const OUTCOME_KEYWORDS = [
  'outcome',
  'endpoint',
  'end point',
  'hospitalization',
  'hospitalisation',
  'hospital admission'
];

export const variableElementId = (cohortId: string, variableName: string): string =>
  `variable-${cohortId}-${variableName}`;

export const conceptMapElementId = (cohortId: string, variableName: string): string =>
  `concept-map-${cohortId}-${variableName}`;

const fieldMatches = (value: unknown, searchTerms: string[], searchMode: SearchMode): boolean => {
  if (value == null) return false;
  const text = String(value).toLowerCase();
  if (searchMode === 'exact') return text.includes(searchTerms.join(' ').toLowerCase());
  if (searchMode === 'and') return searchTerms.every(term => text.includes(term.toLowerCase()));
  return searchTerms.some(term => text.includes(term.toLowerCase()));
};

export const resolveVariableSearchTerms = (searchTerms?: string[], searchQuery?: string): string[] => {
  if (searchTerms) return searchTerms;
  if (!searchQuery) return [];
  return searchQuery
    .split(' ')
    .map(term => term.trim())
    .filter(Boolean);
};

export const variableMatchesSearch = (
  variableName: string,
  variableData: VariableRecord,
  searchTerms: string[],
  searchMode: SearchMode
): boolean => {
  if (searchTerms.length === 0) return true;
  const variableWithName = {...variableData, var_name: variableName} as VariableRecord;

  if (SEARCHABLE_VARIABLE_FIELDS.some(field => fieldMatches(variableWithName[field], searchTerms, searchMode))) {
    return true;
  }

  return (variableData.categories || []).some((category: VariableRecord) =>
    SEARCHABLE_CATEGORY_FIELDS.some(field => fieldMatches(category[field], searchTerms, searchMode))
  );
};

export const variableMatchesOutcome = (variableName: string, variableData: VariableRecord): boolean => {
  const variableWithName = {...variableData, var_name: variableName} as VariableRecord;
  return SEARCHABLE_VARIABLE_FIELDS.some(field => {
    const value = variableWithName[field];
    if (value == null) return false;
    const text = String(value).toLowerCase();
    return OUTCOME_KEYWORDS.some(keyword => text.includes(keyword));
  });
};

const matchesCategoryCount = (categoryCount: number, selectedCategoryTypes: ReadonlySet<string>): boolean => {
  if (selectedCategoryTypes.size === 0) return true;
  if (selectedCategoryTypes.has('Non-categorical') && categoryCount === 0) return true;
  if (selectedCategoryTypes.has('All categorical') && categoryCount > 0) return true;
  if (selectedCategoryTypes.has('2 categories') && categoryCount === 2) return true;
  if (selectedCategoryTypes.has('3 categories') && categoryCount === 3) return true;
  return selectedCategoryTypes.has('4+ categories') && categoryCount >= 4;
};

export const filterVariables = (
  variables: Record<string, VariableRecord>,
  options: VariableFilterOptions
): FilteredVariable[] =>
  Object.entries(variables)
    .filter(([variableName, variableData]) => {
      if (
        options.searchScope === 'variables' &&
        options.searchTerms.length > 0 &&
        !variableMatchesSearch(variableName, variableData, options.searchTerms, options.searchMode)
      ) {
        return false;
      }
      if (options.showOnlyOutcomes && !variableMatchesOutcome(variableName, variableData)) return false;
      if (options.selectedOMOPDomains.size > 0 && !options.selectedOMOPDomains.has(variableData.omop_domain))
        return false;
      if (options.selectedDataTypes.size > 0 && !options.selectedDataTypes.has(variableData.var_type)) return false;
      if (options.selectedVisitTypes.size > 0 && !options.selectedVisitTypes.has(variableData.visits)) return false;
      return matchesCategoryCount((variableData.categories || []).length, options.selectedCategoryTypes);
    })
    .map(([variableName, variableData]) => ({...variableData, var_name: variableName}));

export const parseVariableSources = (sourceName: string | null | undefined): string[] => {
  if (!sourceName) return [];
  return sourceName
    .split('|')
    .map(source => source.trim().toUpperCase())
    .filter(Boolean);
};

export const buildSourceDisplayMap = (variables: FilteredVariable[]): Record<string, string> => {
  const displayBySource: Record<string, string> = {};
  variables.forEach(variable => {
    const sources = parseVariableSources(variable.source_name);
    const labels = variable.source_label
      ? String(variable.source_label)
          .split('|')
          .map(label => label.trim())
          .filter(Boolean)
      : [];
    sources.forEach((source, index) => {
      if (!displayBySource[source]) displayBySource[source] = labels[index] || source;
    });
  });
  return displayBySource;
};

export const getSourceTabs = (variables: FilteredVariable[]): string[] => {
  const sources = new Set<string>();
  variables.forEach(variable => parseVariableSources(variable.source_name).forEach(source => sources.add(source)));
  return sources.size >= 2 ? Array.from(sources).sort() : [];
};

export const filterVariablesBySource = (
  variables: FilteredVariable[],
  sourceTabs: string[],
  activeSourceTab: string | null
): FilteredVariable[] => {
  if (sourceTabs.length === 0 || !activeSourceTab || activeSourceTab === '__all__') return variables;
  return variables.filter(variable => parseVariableSources(variable.source_name).includes(activeSourceTab));
};
