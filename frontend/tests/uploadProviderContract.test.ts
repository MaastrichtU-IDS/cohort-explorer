import {readFileSync} from 'node:fs';

import {describe, expect, it} from 'vitest';

const uploadSource = readFileSync(new URL('../src/pages/upload.tsx', import.meta.url), 'utf8');

describe('upload page DCR provider contract', () => {
  it('loads and renders the configured provider before room creation', () => {
    expect(uploadSource).toContain('/api/dcr/provider');
    expect(uploadSource).toContain('projectDcrUpload');
    expect(uploadSource).toContain('data-testid="upload-dcr-provider-copy"');
    expect(uploadSource).toContain('data-testid="upload-dcr-step-title"');
  });

  it('surfaces provider load failure and keeps room creation disabled', () => {
    expect(uploadSource).toContain(
      'Unable to load the configured Data Clean Room provider. Room creation is disabled.'
    );
    expect(uploadSource).toContain('data-testid="upload-dcr-provider-load-error"');
    expect(uploadSource).toContain('!dcrUploadUi.resolved');
  });

  it('does not hard-code production security claims in the page component', () => {
    expect(uploadSource).not.toContain('external Decentriq platform');
    expect(uploadSource).not.toContain('secure confines of the DCR');
    expect(uploadSource).not.toContain('separately upload the actual patient-level data');
  });

  it('aborts provider discovery before a full-page navigation discards the document', () => {
    expect(uploadSource).toContain('abortOnPageExit(controller');
    expect(uploadSource).toContain('requestController.signal.aborted');
  });

  it('restarts provider discovery when a persisted page is restored', () => {
    expect(uploadSource).toContain('abortOnPageExit(controller, window, loadDcrProvider)');
    expect(uploadSource).toContain('const loadDcrProvider = () =>');
  });
});
