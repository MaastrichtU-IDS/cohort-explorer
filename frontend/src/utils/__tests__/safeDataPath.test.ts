import {mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';

import {describe, expect, it} from 'vitest';

import {
  assertAssetComponent,
  assertCohortIdentifier,
  resolveContainedPath,
  resolveCohortOutputDirectory,
  resolveEdaOutputPath,
  resolveVariableGraphPath
} from '@/utils/safeDataPath';

describe('safe /data paths', () => {
  it('resolves supported cohort and variable identifiers beneath /data', () => {
    expect(resolveCohortOutputDirectory('TIME-CHF')).toBe('/data/dcr_output_TIME-CHF');
    expect(resolveEdaOutputPath('GISSI-HF')).toBe('/data/dcr_output_GISSI-HF/eda_output_GISSI-HF.json');
    expect(resolveVariableGraphPath('TIME-CHF', 'F4_PAS')).toBe('/data/dcr_output_TIME-CHF/f4_pas.png');
    expect(resolveVariableGraphPath('TIME-CHF', 'Systolic BP.v1 (%)')).toBe(
      '/data/dcr_output_TIME-CHF/systolic bp.v1 (%).png'
    );
  });

  it.each([
    '../package',
    '..',
    '.',
    'cohort/name',
    'cohort\\name',
    'white space',
    '',
    ['TIME-CHF']
  ])('rejects unsafe or ambiguous cohort identifier %j', value => {
    expect(() => assertCohortIdentifier(value, 'cohortId')).toThrow('Invalid cohortId');
  });

  it.each(['../package', '..', '.', 'variable/name', 'variable\\name', 'line\nbreak', '', ['age']])(
    'rejects unsafe or ambiguous asset component %j',
    value => {
      expect(() => assertAssetComponent(value, 'variableName')).toThrow('Invalid variableName');
    }
  );

  it('preserves the existing variable component contract', () => {
    expect(assertAssetComponent('Systolic BP.v1 (%)', 'variableName')).toBe('Systolic BP.v1 (%)');
    expect(assertAssetComponent('κρεατινίνη', 'variableName')).toBe('κρεατινίνη');
  });

  it('rejects an existing symlink that resolves outside the allowed root', () => {
    const workspace = mkdtempSync(path.join(tmpdir(), 'cohort-path-'));
    const root = path.join(workspace, 'data');
    const outside = path.join(workspace, 'outside');
    mkdirSync(root);
    mkdirSync(outside);
    writeFileSync(path.join(outside, 'secret.json'), '{}');
    symlinkSync(outside, path.join(root, 'escaped'));

    try {
      expect(() => resolveContainedPath(root, 'escaped', 'secret.json')).toThrow('Invalid data path');
    } finally {
      rmSync(workspace, {recursive: true, force: true});
    }
  });
});
