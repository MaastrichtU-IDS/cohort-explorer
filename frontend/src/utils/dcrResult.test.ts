import {describe, expect, it} from 'vitest';
import {projectArchiveResult, projectJsonResult} from '@/utils/dcrResult';

describe('DCR result projection', () => {
  it('projects aggregate JSON objects into a compact metric table', () => {
    expect(projectJsonResult({participant_count: 240, mean_age: 68.4})).toEqual({
      kind: 'json',
      status: 'ready',
      columns: ['metric', 'value'],
      rows: [
        {metric: 'participant_count', value: '240'},
        {metric: 'mean_age', value: '68.4'}
      ]
    });
  });

  it('preserves tabular JSON rows with deterministic columns', () => {
    expect(
      projectJsonResult([
        {cohort: 'TIME-CHF', count: 120},
        {cohort: 'GISSI-HF', count: 140, status: 'complete'}
      ])
    ).toEqual({
      kind: 'json',
      status: 'ready',
      columns: ['cohort', 'count', 'status'],
      rows: [
        {cohort: 'TIME-CHF', count: '120', status: ''},
        {cohort: 'GISSI-HF', count: '140', status: 'complete'}
      ]
    });
  });

  it('projects a ZIP result without parsing row-level content', () => {
    expect(projectArchiveResult('aggregate-result.zip', 4096)).toEqual({
      kind: 'archive',
      status: 'ready',
      filename: 'aggregate-result.zip',
      byteSize: 4096
    });
  });
});
