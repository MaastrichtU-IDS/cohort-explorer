import {describe, expect, it} from 'vitest';
import {projectDcrAirlockSettings, projectDcrProvider, projectDcrWizard} from '@/utils/dcrProvider';

describe('DCR provider UI projection', () => {
  it('uses local provider-neutral copy and capabilities for AADCR v2', () => {
    expect(
      projectDcrProvider('aadcrv2', {
        supports_computation_output: true,
        supports_shuffle_output: false,
        local_simulation: true,
        synthetic_data_only: true
      })
    ).toEqual({
      provider: 'aadcrv2',
      createLabel: 'Create Data Clean Room',
      openLabel: 'Open in My DCRs',
      refreshLabel: 'Refresh rooms',
      loadingLabel: 'Creating the local Data Clean Room. This may take a few seconds...',
      canRunResult: true,
      canShuffle: false,
      localSimulation: true,
      syntheticDataOnly: true
    });
  });

  it('preserves Decentriq copy and defaults for legacy responses', () => {
    expect(projectDcrProvider(undefined, undefined)).toEqual(
      expect.objectContaining({
        provider: 'decentriq',
        openLabel: 'Open on Decentriq',
        refreshLabel: 'Refresh from Decentriq',
        canRunResult: false,
        localSimulation: false
      })
    );
  });

  it('does not infer a provider from a room URL', () => {
    const projection = projectDcrProvider(undefined, {
      supports_computation_output: true
    });

    expect(projection.provider).toBe('decentriq');
    expect(projection.canRunResult).toBe(true);
  });

  it('removes Airlock promises and warns before creating a local simulation', () => {
    const wizard = projectDcrWizard(
      projectDcrProvider('aadcrv2', {
        local_simulation: true,
        synthetic_data_only: true
      })
    );

    expect(wizard.supportsAirlock).toBe(false);
    expect(wizard.creationWarning).toBe(
      'Local synthetic-data simulation only. This does not provide a confidential-computing or production security boundary. Do not use real or confidential data.'
    );
    expect(wizard.steps.find(step => step.id === 'data-samples')?.title).toBe('Synthetic Samples');
  });

  it('fails closed for AADCR v2 when a legacy capability response is incomplete', () => {
    const wizard = projectDcrWizard(projectDcrProvider('aadcrv2'));

    expect(wizard.supportsAirlock).toBe(false);
    expect(wizard.creationWarning).not.toBeNull();
  });

  it('preserves the Airlock workflow for production Decentriq rooms', () => {
    const wizard = projectDcrWizard(projectDcrProvider('decentriq'));

    expect(wizard.supportsAirlock).toBe(true);
    expect(wizard.creationWarning).toBeNull();
    expect(wizard.steps.find(step => step.id === 'data-samples')?.title).toBe('Data Samples');
  });

  it('omits Airlock configuration from local-simulation creation requests', () => {
    const wizard = projectDcrWizard(projectDcrProvider('aadcrv2', {local_simulation: true}));

    expect(projectDcrAirlockSettings(wizard, {'TIME-CHF': true})).toBeUndefined();
  });

  it('converts Decentriq Airlock selections to percentages', () => {
    const wizard = projectDcrWizard(projectDcrProvider('decentriq'));

    expect(projectDcrAirlockSettings(wizard, {'TIME-CHF': true, 'GISSI-HF': false})).toEqual({
      'TIME-CHF': 20,
      'GISSI-HF': 0
    });
  });
});
