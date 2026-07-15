import {describe, expect, it} from 'vitest';
import {
  cohortElementId,
  collectCohortSearchResults,
  filterCohorts,
  matchesSearchTerms
} from '@/utils/cohortFiltering';

const metadata = {
  'TIME-CHF': {
    cohort_id: 'TIME-CHF',
    institution: 'University Hospital Zurich',
    study_type: 'Observational',
    study_design: 'Cohort study',
    study_objective: 'Follow heart failure outcomes',
    variables: {
      heart_rate: {
        var_label: 'Resting heart rate',
        concept_name: 'Heart rate',
        categories: []
      },
      nyha_class: {
        var_label: 'NYHA class',
        categories: [{value: '4', label: 'Severe heart failure'}]
      }
    }
  },
  'GISSI-HF': {
    cohort_id: 'GISSI-HF',
    institution: 'Italian Cardiology Network',
    study_type: 'Interventional',
    study_design: 'Randomized trial',
    study_objective: 'Evaluate therapy',
    variables: {
      HR: {
        var_label: 'Pulse',
        concept_name: 'Heart rate',
        categories: []
      }
    }
  }
} as any;

describe('metadata search predicates', () => {
  it('builds a stable cohort selector', () => {
    expect(cohortElementId('TIME-CHF')).toBe('cohort-TIME-CHF');
  });

  it('preserves OR, AND, and exact matching with normalized separators', () => {
    expect(matchesSearchTerms('resting_heart-rate', ['heart', 'weight'], 'or')).toBe(true);
    expect(matchesSearchTerms('resting_heart-rate', ['resting', 'rate'], 'and')).toBe(true);
    expect(matchesSearchTerms('resting_heart-rate', ['resting', 'heart'], 'exact')).toBe(true);
    expect(matchesSearchTerms('resting_heart-rate', ['heart', 'resting'], 'exact')).toBe(false);
  });

  it('filters cohort, variable, and all scopes without bypassing metadata filters', () => {
    expect(
      filterCohorts(metadata, {
        selectedStudyTypes: new Set(['Cohort study']),
        selectedInstitutes: new Set<string>(),
        searchTerms: ['heart', 'rate'],
        searchMode: 'and',
        searchScope: 'variables'
      }).map(cohort => cohort.cohort_id)
    ).toEqual(['TIME-CHF']);

    expect(
      filterCohorts(metadata, {
        selectedStudyTypes: new Set<string>(),
        selectedInstitutes: new Set<string>(),
        searchTerms: ['italian'],
        searchMode: 'or',
        searchScope: 'cohorts'
      }).map(cohort => cohort.cohort_id)
    ).toEqual(['GISSI-HF']);

    expect(
      filterCohorts(metadata, {
        selectedStudyTypes: new Set<string>(),
        selectedInstitutes: new Set<string>(),
        searchTerms: ['severe'],
        searchMode: 'or',
        searchScope: 'all'
      }).map(cohort => cohort.cohort_id)
    ).toEqual(['TIME-CHF']);
  });

  it('projects matched cohort sections and matching variable names', () => {
    const results = collectCohortSearchResults(metadata, ['heart', 'failure'], 'and', 'all');

    expect(results.matchedCohorts).toEqual([
      {cohortId: 'TIME-CHF', sections: ['study objective']}
    ]);
    expect(results.variablesByCohort).toEqual({'TIME-CHF': ['nyha_class']});
    expect(results.totalVariables).toBe(1);
  });
});
