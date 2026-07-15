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

export function projectDcrProvider(
  provider?: string,
  capabilities?: DcrCapabilities
): DcrProviderProjection {
  const normalizedProvider = provider?.trim().toLowerCase() || 'decentriq';
  const isLocal = normalizedProvider === 'aadcrv2';

  return {
    provider: normalizedProvider,
    createLabel: 'Create Data Clean Room',
    openLabel: isLocal ? 'Open created room' : 'Open on Decentriq',
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
