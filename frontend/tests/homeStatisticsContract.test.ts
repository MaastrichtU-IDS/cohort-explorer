import {readFileSync} from 'node:fs';

import {describe, expect, it} from 'vitest';

const homeSource = readFileSync(new URL('../src/pages/index.tsx', import.meta.url), 'utf8');
const contextSource = readFileSync(new URL('../src/components/CohortsContext.tsx', import.meta.url), 'utf8');

describe('home statistics authority contract', () => {
  it('renders the context snapshot instead of racing it with cached API reads', () => {
    expect(homeSource).toContain('cohortStatistics');
    expect(homeSource).toContain('statisticsStatus');
    expect(homeSource).not.toContain('/api/get-statistics');
    expect(homeSource).not.toContain('calculateStatistics');
    expect(homeSource).not.toContain('setTimeout');
    expect(contextSource).not.toContain('/api/save-statistics');
  });

  it('calculates and publishes empty snapshots so stale non-empty results are invalidated', () => {
    expect(contextSource).not.toContain("if (Object.keys(snapshot).length === 0) return");
    expect(contextSource).toContain("statisticsStatus");
    expect(contextSource).toContain("setStatisticsStatus('loaded')");
  });

  it('invalidates an in-flight calculation before changing metadata sources', () => {
    expect(contextSource).toMatch(
      /useEffect\(\(\) => \{\s*metadataRequestGeneration\.current \+= 1;\s*statisticsGeneration\.current \+= 1;\s*setDataCleanRoom/
    );
  });
});
