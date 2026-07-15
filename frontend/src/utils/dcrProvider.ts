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
  steps: DcrWizardStep[];
  supportsAirlock: boolean;
  creationWarning: string | null;
}

const DCR_WIZARD_STEPS: DcrWizardStep[] = [
  {id: 'name', title: 'DCR Name & Cohorts'},
  {id: 'participants', title: 'Participants'},
  {id: 'research-goals', title: 'Research Goals'},
  {id: 'data-samples', title: 'Data Samples'},
  {id: 'mapping', title: 'Mapping Files'},
  {id: 'review', title: 'Review & Create'}
];

export function projectDcrWizard(provider: DcrProviderProjection): DcrWizardProjection {
  const isLocalSimulation = provider.localSimulation || provider.provider === 'aadcrv2';

  return {
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
