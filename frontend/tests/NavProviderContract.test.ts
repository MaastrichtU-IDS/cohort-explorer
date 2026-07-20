import {readFileSync} from 'node:fs';

import {describe, expect, it} from 'vitest';

const navSource = readFileSync(new URL('../src/components/Nav.tsx', import.meta.url), 'utf8');

describe('navigation DCR provider contract', () => {
  it('resolves the wizard provider independently before exposing creation', () => {
    expect(navSource).toContain('/api/dcr/provider');
    expect(navSource).not.toContain('fetch(`${apiUrl}/my-dcrs`');
    expect(navSource).toContain('dcrWizardUi.resolved');
  });

  it('renders a visible provider failure and gates wizard interactions', () => {
    expect(navSource).toContain('data-testid="dcr-provider-error"');
    expect(navSource).toContain('disabled={!dcrWizardUi.resolved}');
    expect(navSource).toMatch(/!dcrWizardUi\.resolved\s*\|\|\s*isLoading/);
  });

  it('uses one complete wizard request projection for preview and creation', () => {
    expect(navSource.match(/JSON\.stringify\(buildCurrentDcrWizardRequest\(\)\)/g)).toHaveLength(2);
    expect(navSource).toContain('buildDcrWizardRequest({');
  });

  it('aborts provider discovery before a full-page navigation discards the document', () => {
    expect(navSource).toContain('abortOnPageExit(controller');
    expect(navSource).toContain('requestController.signal.aborted');
  });

  it('restarts provider discovery when a persisted page is restored', () => {
    expect(navSource).toContain('abortOnPageExit(controller, window, loadDcrProvider)');
    expect(navSource).toContain('const loadDcrProvider = () =>');
  });
});
