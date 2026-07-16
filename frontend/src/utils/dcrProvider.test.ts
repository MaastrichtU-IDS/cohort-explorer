import {describe, expect, it} from 'vitest';
import {
  projectConfiguredDcrProvider,
  projectDcrAirlockSettings,
  projectDcrProvider,
  projectDcrUpload,
  projectDcrWizard
} from '@/utils/dcrProvider';
import * as dcrProviderModule from '@/utils/dcrProvider';

describe('DCR provider UI projection', () => {
  it('exposes a provider-projected upload contract', () => {
    const projectDcrUpload = (dcrProviderModule as Record<string, unknown>).projectDcrUpload;

    expect(typeof projectDcrUpload).toBe('function');
  });

  it('describes AADCR provisioning as a non-confidential synthetic flow', () => {
    const projection = projectDcrUpload(
      projectDcrProvider('aadcrv2', {
        local_simulation: true,
        synthetic_data_only: true,
        supports_provisioning: true
      })
    );
    const renderedCopy = JSON.stringify(projection);

    expect(projection).toMatchObject({
      resolved: true,
      localSimulation: true,
      heading: 'Step 2: Create Local Synthetic AADCR Simulation',
      warning:
        'Local synthetic-data simulation only. This does not provide a confidential-computing or production security boundary. Do not use real or confidential data.'
    });
    expect(renderedCopy).toContain('generated synthetic CSV');
    expect(renderedCopy).toContain('Cohort Explorer provisions');
    expect(renderedCopy).not.toContain('external Decentriq platform');
    expect(renderedCopy).not.toContain('secure confines');
    expect(renderedCopy).not.toContain('separately upload the actual patient-level data');
  });

  it('preserves the existing Decentriq upload and external-provisioning copy', () => {
    expect(projectDcrUpload(projectDcrProvider('decentriq'))).toEqual({
      resolved: true,
      localSimulation: false,
      heading: 'Step 2: Initiate Data Clean Room (DCR) Creation',
      metadataPurpose:
        'Providing accurate metadata is crucial for enabling data scientists to understand and effectively utilize the data within the secure Decentriq platform later.',
      creationIntro:
        'The next step is to initiate the creation of its secure Data Clean Room (DCR) on the external Decentriq platform.',
      provisioningIntro:
        'This DCR will be configured based on the variables defined in your metadata. Once the DCR is provisioned on Decentriq:',
      provisioningSteps: [
        'You (or the designated data custodian) will need to separately upload the actual patient-level data directly and securely into the Decentriq DCR.',
        'Patient data never passes through or is stored by this Cohort Explorer application.',
        'Data scientists can then request access to perform analysis within the secure confines of the DCR.'
      ],
      warning: null
    });
  });

  it('uses neutral copy until the configured provider is known', () => {
    const projection = projectDcrUpload();
    const renderedCopy = JSON.stringify(projection);

    expect(projection.resolved).toBe(false);
    expect(projection.heading).toBe('Step 2: Load Data Clean Room Provider');
    expect(projection.warning).toContain('Provider details must load');
    expect(renderedCopy).not.toContain('Decentriq');
    expect(renderedCopy).not.toContain('confidential-computing');
  });

  it('surfaces a provider-load failure while keeping upload creation unresolved', () => {
    const loadError =
      'Unable to load the configured Data Clean Room provider. Room creation is disabled.';
    const projection = projectDcrUpload(undefined, loadError);

    expect(projection).toMatchObject({
      resolved: false,
      heading: 'Step 2: Data Clean Room Provider Unavailable',
      warning: loadError
    });
  });

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

  it('accepts only an explicitly supported provider for pre-creation security copy', () => {
    expect(projectConfiguredDcrProvider('decentriq')).toMatchObject({provider: 'decentriq'});
    expect(projectConfiguredDcrProvider('aadcrv2', {local_simulation: true})).toMatchObject({
      provider: 'aadcrv2',
      localSimulation: true
    });
    expect(() => projectConfiguredDcrProvider(undefined)).toThrow('did not identify a supported provider');
    expect(() => projectConfiguredDcrProvider('unknown-provider')).toThrow(
      'did not identify a supported provider'
    );
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

  it('keeps the wizard neutral and disabled until its provider is resolved', () => {
    const wizard = projectDcrWizard();

    expect(wizard).toMatchObject({
      resolved: false,
      supportsAirlock: false,
      creationWarning: 'Loading the configured Data Clean Room provider...'
    });
  });

  it('projects a visible fail-closed wizard error without Decentriq semantics', () => {
    const providerError =
      'Unable to load the configured Data Clean Room provider. Wizard creation is disabled.';
    const wizard = projectDcrWizard(undefined, providerError);

    expect(wizard).toMatchObject({
      resolved: false,
      supportsAirlock: false,
      creationWarning: providerError
    });
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
