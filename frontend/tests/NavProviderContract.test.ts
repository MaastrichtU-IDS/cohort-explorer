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
    expect(navSource).toContain('!dcrWizardUi.resolved || isLoading');
  });
});
