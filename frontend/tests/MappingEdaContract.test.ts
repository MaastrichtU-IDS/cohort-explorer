import {readFileSync} from 'node:fs';
import {describe, expect, it} from 'vitest';

const mappingSource = readFileSync(
  new URL('../src/pages/mapping.tsx', import.meta.url),
  'utf8'
);

describe('mapping EDA route contract', () => {
  it('canonicalizes both cohort IDs at every compare endpoint call site', () => {
    const compareUrls = [...mappingSource.matchAll(
      /const imageUrl = `\/api\/compare-eda\/([^`]+)`;/g
    )].map(match => match[1]);

    expect(compareUrls).toHaveLength(3);
    for (const url of compareUrls) {
      expect(url).toMatch(/canonicalSource/);
      expect(url).toMatch(/canonicalT(?:arget|gtStudy)/);
    }
  });
});
