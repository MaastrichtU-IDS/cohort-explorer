export interface DcrCapabilities {
  supports_provisioning?: boolean;
  supports_definition_preview?: boolean;
  supports_live_creation?: boolean;
  supports_room_refresh?: boolean;
  supports_audit_log?: boolean;
  supports_computation_output?: boolean;
  supports_shuffle_output?: boolean;
  synthetic_data_only?: boolean;
  local_simulation?: boolean;
}

export type DcrProvider = 'decentriq' | 'aadcrv2' | string;

export interface DcrProviderProjection {
  provider: DcrProvider;
  createLabel: string;
  openLabel: string;
  refreshLabel: string;
  loadingLabel: string;
  canRunResult: boolean;
  canShuffle: boolean;
  localSimulation: boolean;
  syntheticDataOnly: boolean;
}

export interface DcrWizardStep {
  id: 'name' | 'participants' | 'research-goals' | 'data-samples' | 'mapping' | 'review';
  title: string;
}

export interface DcrWizardProjection {
  resolved: boolean;
  steps: DcrWizardStep[];
  supportsAirlock: boolean;
  creationWarning: string | null;
}

export interface DcrUploadProjection {
  resolved: boolean;
  localSimulation: boolean;
  heading: string;
  metadataPurpose: string;
  creationIntro: string;
  provisioningIntro: string;
  provisioningSteps: string[];
  warning: string | null;
}

const DCR_WIZARD_STEPS: DcrWizardStep[] = [
  {id: 'name', title: 'DCR Name & Cohorts'},
  {id: 'participants', title: 'Participants'},
  {id: 'research-goals', title: 'Research Goals'},
  {id: 'data-samples', title: 'Data Samples'},
  {id: 'mapping', title: 'Mapping Files'},
  {id: 'review', title: 'Review & Create'}
];

export function projectDcrWizard(
  provider?: DcrProviderProjection,
  loadError?: string | null
): DcrWizardProjection {
  if (!provider) {
    return {
      resolved: false,
      steps: DCR_WIZARD_STEPS.map(step => ({...step})),
      supportsAirlock: false,
      creationWarning: loadError || 'Loading the configured Data Clean Room provider...'
    };
  }

  const isLocalSimulation = provider.localSimulation || provider.provider === 'aadcrv2';

  return {
    resolved: true,
    steps: DCR_WIZARD_STEPS.map(step =>
      isLocalSimulation && step.id === 'data-samples' ? {...step, title: 'Synthetic Samples'} : {...step}
    ),
    supportsAirlock: !isLocalSimulation,
    creationWarning: isLocalSimulation
      ? 'Local synthetic-data simulation only. This does not provide a confidential-computing or production security boundary. Do not use real or confidential data.'
      : null
  };
}

export function projectDcrAirlockSettings(
  wizard: DcrWizardProjection,
  selections: Record<string, boolean>
): Record<string, number> | undefined {
  if (!wizard.supportsAirlock) return undefined;

  return Object.fromEntries(Object.entries(selections).map(([cohortId, isEnabled]) => [cohortId, isEnabled ? 20 : 0]));
}


export function projectDcrUpload(
  provider?: DcrProviderProjection,
  loadError?: string | null
): DcrUploadProjection {
  if (!provider) {
    return {
      resolved: false,
      localSimulation: false,
      heading: loadError
        ? 'Step 2: Data Clean Room Provider Unavailable'
        : 'Step 2: Load Data Clean Room Provider',
      metadataPurpose: 'Providing accurate metadata enables the configured Data Clean Room workflow.',
      creationIntro: 'Loading the configured Data Clean Room provider details...',
      provisioningIntro: '',
      provisioningSteps: [],
      warning: loadError || 'Provider details must load before a room can be created.'
    };
  }

  const isLocalSimulation = provider.localSimulation || provider.provider === 'aadcrv2';
  if (isLocalSimulation) {
    return {
      resolved: true,
      localSimulation: true,
      heading: 'Step 2: Create Local Synthetic AADCR Simulation',
      metadataPurpose:
        'In the local AADCR v2 demo, this metadata is used only to configure a synthetic-data simulation. It is not a confidential-computing or production security boundary.',
      creationIntro:
        'The next step creates a local AADCR v2 Data Clean Room simulation for the generated synthetic cohort.',
      provisioningIntro: 'Cohort Explorer creates and provisions this local demo automatically:',
      provisioningSteps: [
        'Cohort Explorer provisions the generated synthetic CSV from the immutable demo pack into AADCR v2 for aggregate computation.',
        'Do not use or upload real, patient-level, or confidential data in this local simulation.',
        'This flow demonstrates local integration behavior only; it does not provide production Data Clean Room security.'
      ],
      warning: projectDcrWizard(provider).creationWarning
    };
  }

  return {
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
  };
}


export function projectConfiguredDcrProvider(
  provider: unknown,
  capabilities?: unknown
): DcrProviderProjection {
  const normalizedProvider = typeof provider === 'string' ? provider.trim().toLowerCase() : '';
  if (normalizedProvider !== 'decentriq' && normalizedProvider !== 'aadcrv2') {
    throw new Error('Provider response did not identify a supported provider');
  }
  if (
    capabilities !== undefined &&
    capabilities !== null &&
    (typeof capabilities !== 'object' || Array.isArray(capabilities))
  ) {
    throw new Error('Provider response contained invalid capabilities');
  }

  return projectDcrProvider(
    normalizedProvider,
    (capabilities ?? undefined) as DcrCapabilities | undefined
  );
}


export function projectDcrProvider(
  provider?: string,
  capabilities?: DcrCapabilities
): DcrProviderProjection {
  const normalizedProvider = provider?.trim().toLowerCase() || 'decentriq';
  const isLocal = normalizedProvider === 'aadcrv2';

  return {
    provider: normalizedProvider,
    createLabel: 'Create Data Clean Room',
    openLabel: isLocal ? 'Open in My DCRs' : 'Open on Decentriq',
    refreshLabel: isLocal ? 'Refresh rooms' : 'Refresh from Decentriq',
    loadingLabel: isLocal
      ? 'Creating the local Data Clean Room. This may take a few seconds...'
      : 'Creating the Data Clean Room on Decentriq Platform. Will take a few seconds...',
    canRunResult: capabilities?.supports_computation_output === true,
    canShuffle: capabilities?.supports_shuffle_output === true,
    localSimulation: capabilities?.local_simulation === true,
    syntheticDataOnly: capabilities?.synthetic_data_only === true
  };
}
