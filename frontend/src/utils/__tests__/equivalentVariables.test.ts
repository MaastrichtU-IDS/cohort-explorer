import {describe, expect, it} from 'vitest';
import {groupEquivalentVariables} from '@/utils/equivalentVariables';

const metadata = {
  'TIME-CHF': {
    variables: {
      heart_rate: {
        concept_name: 'Heart rate',
        concept_code: 'snomedct:364075005',
        omop_id: '3027018'
      },
      pulse_baseline: {
        concept_name: 'Pulse finding',
        omop_id: '3027018'
      }
    }
  },
  'GISSI-HF': {
    variables: {
      HR: {
        concept_name: 'Heart rate',
        concept_code: 'SNOMEDCT:364075005',
        omop_id: '3027018'
      },
      pulse: {
        concept_name: 'Pulse finding',
        omop_id: '3027018'
      }
    }
  }
} as any;

describe('equivalent-variable projections', () => {
  it('groups equivalent variables by concept code before OMOP id', () => {
    const grouped = groupEquivalentVariables(metadata, ['heart', 'rate'], 'and', 'variables');

    expect(grouped?.conceptGroups[0].code).toBe('snomedct:364075005');
    expect(grouped?.conceptGroups[0].namesByCohort).toEqual([
      {cohortId: 'GISSI-HF', names: ['HR'], isMatched: true},
      {cohortId: 'TIME-CHF', names: ['heart_rate'], isMatched: true}
    ]);
    expect(grouped?.omopGroups[0].code).toBe('3027018');
  });

  it('uses OMOP ids for matches without a concept code and skips cohort-only searches', () => {
    const grouped = groupEquivalentVariables(metadata, ['pulse'], 'or', 'variables');

    expect(grouped?.conceptGroups).toEqual([]);
    expect(grouped?.omopGroups[0].namesByCohort.flatMap(entry => entry.names)).toEqual([
      'HR',
      'pulse',
      'heart_rate',
      'pulse_baseline'
    ]);
    expect(groupEquivalentVariables(metadata, ['pulse'], 'and', 'cohorts')).toBeNull();
  });
});
