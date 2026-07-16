import {readFileSync} from 'node:fs';

import {describe, expect, it} from 'vitest';

const contextSource = readFileSync(new URL('../src/components/CohortsContext.tsx', import.meta.url), 'utf8');
const uploadSource = readFileSync(new URL('../src/pages/upload.tsx', import.meta.url), 'utf8');
const cacheWorkerSource = readFileSync(new URL('../public/cohortsWorker.js', import.meta.url), 'utf8');
const sparqlWorkerSource = readFileSync(new URL('../public/cohortsSparqlWorker.js', import.meta.url), 'utf8');

describe('cohort metadata worker contract', () => {
  it('tags requests and ignores responses older than the latest issued refresh', () => {
    expect(contextSource).toContain('metadataRequestGeneration');
    expect(contextSource).toContain('isLatestMetadataResponse(requestId, metadataRequestGeneration.current)');
    expect(contextSource).toContain('requestId: ++metadataRequestGeneration.current');
  });

  it.each([cacheWorkerSource, sparqlWorkerSource])('echoes request ids in worker response envelopes', workerSource => {
    expect(workerSource).toContain('requestId');
    expect(workerSource).toContain('self.postMessage({requestId, payload:');
    expect(workerSource).toContain('self.postMessage({requestId, error:');
  });

  it('does not refresh unchanged metadata after read-only dictionary validation', () => {
    const validationHandler = uploadSource.slice(
      uploadSource.indexOf('const handleValidateDictionary'),
      uploadSource.indexOf('const handleMetadataSubmit')
    );
    expect(validationHandler).not.toContain('fetchCohortsData');
  });
});
