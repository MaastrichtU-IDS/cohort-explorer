import {describe, expect, it} from 'vitest';

import {calculateCohortStatistics} from '@/utils/cohortStatistics';

describe('cohort statistics', () => {
  it('calculates the complete dashboard inventory from the supplied metadata snapshot', async () => {
    const aggregateChecks: string[] = [];
    const cohorts = {
      'TIME-CHF': {
        cohort_id: 'TIME-CHF',
        study_participants: '2,500 participants',
        variables: {age: {}, sex: {}}
      },
      'GISSI-HF': {
        cohort_id: 'GISSI-HF',
        study_participants: 2500,
        variables: {age: {}, sex: {}}
      }
    } as any;

    const result = await calculateCohortStatistics(cohorts, async cohortId => {
      aggregateChecks.push(cohortId);
      return true;
    });

    expect(result).toEqual({
      totalCohorts: 2,
      cohortsWithMetadata: 2,
      cohortsWithAggregateAnalysis: 2,
      totalPatients: 5000,
      patientsInCohortsWithMetadata: 5000,
      totalVariables: 4
    });
    expect(aggregateChecks).toEqual(['TIME-CHF', 'GISSI-HF']);
  });

  it('excludes cohorts without variables from metadata-specific totals', async () => {
    const cohorts = {
      complete: {cohort_id: 'complete', study_participants: '42', variables: {age: {}}},
      empty: {cohort_id: 'empty', study_participants: 'unknown', variables: {}}
    } as any;

    expect(await calculateCohortStatistics(cohorts, async cohortId => cohortId === 'complete')).toEqual({
      totalCohorts: 2,
      cohortsWithMetadata: 1,
      cohortsWithAggregateAnalysis: 1,
      totalPatients: 42,
      patientsInCohortsWithMetadata: 42,
      totalVariables: 1
    });
  });
});
