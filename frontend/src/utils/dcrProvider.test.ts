import {describe, expect, it} from 'vitest';
import {projectDcrProvider} from '@/utils/dcrProvider';

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
      openLabel: 'Open created room',
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
});
