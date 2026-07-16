import {describe, expect, it} from 'vitest';
import {
  buildSourceDisplayMap,
  conceptMapElementId,
  filterVariables,
  filterVariablesBySource,
  getSourceTabs,
  variableElementId,
  variableMatchesSearch
} from '@/utils/variableFiltering';

const variables = {
  heart_rate: {
    var_label: 'Primary outcome heart rate',
    var_type: 'integer',
    visits: 'baseline',
    omop_domain: 'Measurement',
    source_name: 'ehr | crf',
    source_label: 'Electronic record | Case report form',
    categories: []
  },
  nyha_class: {
    var_label: 'Functional class',
    var_type: 'string',
    visits: 'month 3',
    omop_domain: 'Observation',
    source_name: 'crf',
    source_label: 'Case report form',
    categories: [
      {value: '1', label: 'Class I'},
      {value: '2', label: 'Class II'}
    ]
  },
  status: {
    var_label: 'Clinical status',
    var_type: 'string',
    visits: 'baseline',
    omop_domain: 'Observation',
    source_name: 'registry',
    source_label: 'Registry',
    categories: [
      {value: '0', label: 'Stable'},
      {value: '1', label: 'Endpoint reached'},
      {value: '2', label: 'Unknown'},
      {value: '3', label: 'Other'}
    ]
  }
} as any;

const noFilters = {
  selectedOMOPDomains: new Set<string>(),
  selectedDataTypes: new Set<string>(),
  selectedCategoryTypes: new Set<string>(),
  selectedVisitTypes: new Set<string>(),
  showOnlyOutcomes: false,
  searchScope: 'variables' as const,
  searchTerms: [] as string[],
  searchMode: 'or' as const
};

describe('variable predicates', () => {
  it('builds stable variable and concept-map selectors', () => {
    expect(variableElementId('TIME-CHF', 'heart_rate')).toBe('variable-TIME-CHF-heart_rate');
    expect(conceptMapElementId('TIME-CHF', 'heart_rate')).toBe('concept-map-TIME-CHF-heart_rate');
  });

  it('matches OR, AND, exact, and category fields', () => {
    expect(variableMatchesSearch('heart_rate', variables.heart_rate, ['primary', 'rate'], 'and')).toBe(true);
    expect(variableMatchesSearch('heart_rate', variables.heart_rate, ['missing', 'heart'], 'or')).toBe(true);
    expect(variableMatchesSearch('heart_rate', variables.heart_rate, ['primary', 'outcome'], 'exact')).toBe(true);
    expect(variableMatchesSearch('nyha_class', variables.nyha_class, ['class', 'ii'], 'and')).toBe(true);
  });

  it('combines search, outcome, domain, type, visit, and category-count filters', () => {
    const result = filterVariables(variables, {
      ...noFilters,
      selectedOMOPDomains: new Set(['Measurement']),
      selectedDataTypes: new Set(['integer']),
      selectedCategoryTypes: new Set(['Non-categorical']),
      selectedVisitTypes: new Set(['baseline']),
      showOnlyOutcomes: true,
      searchTerms: ['heart', 'rate'],
      searchMode: 'and'
    });

    expect(result.map(variable => variable.var_name)).toEqual(['heart_rate']);
    expect(
      filterVariables(variables, {
        ...noFilters,
        selectedCategoryTypes: new Set(['2 categories', '4+ categories'])
      }).map(variable => variable.var_name)
    ).toEqual(['nyha_class', 'status']);
  });

  it('recognizes hospitalization metadata as an outcome without rewriting its source label', () => {
    const hospitalization = {
      hf_hosp: {
        var_label: 'heart failure hospitalization',
        var_type: 'INT',
        visits: 'baseline time',
        omop_domain: 'observation',
        categories: [
          {value: '0', label: 'No'},
          {value: '1', label: 'Yes'}
        ]
      }
    } as any;

    expect(
      filterVariables(hospitalization, {
        ...noFilters,
        showOnlyOutcomes: true
      }).map(variable => variable.var_name)
    ).toEqual(['hf_hosp']);

    hospitalization.hf_hosp.var_label = 'emergency hospital admission for heart failure';
    expect(
      filterVariables(hospitalization, {
        ...noFilters,
        showOnlyOutcomes: true
      }).map(variable => variable.var_name)
    ).toEqual(['hf_hosp']);
  });

  it('keeps all variables for all-scope highlighting and projects source tabs', () => {
    const filtered = filterVariables(variables, {
      ...noFilters,
      searchScope: 'all',
      searchTerms: ['not-present']
    });
    const tabs = getSourceTabs(filtered);

    expect(filtered).toHaveLength(3);
    expect(tabs).toEqual(['CRF', 'EHR', 'REGISTRY']);
    expect(buildSourceDisplayMap(filtered)).toEqual({
      EHR: 'Electronic record',
      CRF: 'Case report form',
      REGISTRY: 'Registry'
    });
    expect(filterVariablesBySource(filtered, tabs, 'CRF').map(variable => variable.var_name)).toEqual([
      'heart_rate',
      'nyha_class'
    ]);
  });
});
