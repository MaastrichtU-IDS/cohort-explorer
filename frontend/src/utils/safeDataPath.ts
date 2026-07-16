import fs from 'node:fs';
import path from 'node:path';

const DATA_ROOT = path.resolve('/data');
const COHORT_IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;
const CONTROL_CHARACTER = new RegExp('\\p{C}', 'u');

export class InvalidDataIdentifierError extends Error {}

export const assertCohortIdentifier = (value: unknown, label: string): string => {
  if (typeof value !== 'string' || !COHORT_IDENTIFIER.test(value)) {
    throw new InvalidDataIdentifierError(`Invalid ${label}`);
  }
  return value;
};

export const assertAssetComponent = (value: unknown, label: string): string => {
  const invalid =
    typeof value !== 'string' ||
    value.length === 0 ||
    value === '.' ||
    value === '..' ||
    value.includes('/') ||
    value.includes('\\') ||
    CONTROL_CHARACTER.test(value);
  if (invalid) throw new InvalidDataIdentifierError(`Invalid ${label}`);
  return value;
};

const assertContained = (root: string, candidate: string): void => {
  const relative = path.relative(root, candidate);
  if (relative.length === 0 || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new InvalidDataIdentifierError('Invalid data path');
  }
};

export const resolveContainedPath = (root: string, ...segments: string[]): string => {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(resolvedRoot, ...segments);
  assertContained(resolvedRoot, resolved);

  if (fs.existsSync(resolved) && fs.existsSync(resolvedRoot)) {
    try {
      const realRoot = fs.realpathSync(resolvedRoot);
      const realCandidate = fs.realpathSync(resolved);
      assertContained(realRoot, realCandidate);
      return realCandidate;
    } catch (error) {
      if (error instanceof InvalidDataIdentifierError) throw error;
      throw new InvalidDataIdentifierError('Invalid data path');
    }
  }
  return resolved;
};

const resolveInsideData = (...segments: string[]): string => resolveContainedPath(DATA_ROOT, ...segments);

export const resolveCohortOutputDirectory = (cohortId: unknown): string => {
  const safeCohortId = assertCohortIdentifier(cohortId, 'cohortId');
  return resolveInsideData(`dcr_output_${safeCohortId}`);
};

export const resolveEdaOutputPath = (cohortName: unknown): string => {
  const safeCohortName = assertCohortIdentifier(cohortName, 'cohortName');
  return resolveInsideData(`dcr_output_${safeCohortName}`, `eda_output_${safeCohortName}.json`);
};

export const resolveVariableGraphPath = (cohortId: unknown, variableName: unknown): string => {
  const safeCohortId = assertCohortIdentifier(cohortId, 'cohortId');
  const safeVariableName = assertAssetComponent(variableName, 'variableName');
  return resolveInsideData(`dcr_output_${safeCohortId}`, `${safeVariableName.toLowerCase()}.png`);
};
