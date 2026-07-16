export interface DcrMappingFileSelection {
  filename: string;
  filepath: string;
  display_name: string;
  cohorts: string[];
}

export interface BuildDcrWizardRequestInput {
  dataCleanRoom: Record<string, unknown>;
  shuffledSampleSettings: Record<string, boolean>;
  additionalAnalysts: string[];
  excludedDataOwners: string[];
  airlockSettings: Record<string, number> | undefined;
  dcrName: string;
  researchQuestion: string;
  sessionId: string | null;
  availableMappingFiles: DcrMappingFileSelection[];
  selectedMappingFiles: Record<string, boolean>;
  includeMappingUploadSlot: boolean;
}

export const buildDcrWizardRequest = ({
  dataCleanRoom,
  shuffledSampleSettings,
  additionalAnalysts,
  excludedDataOwners,
  airlockSettings,
  dcrName,
  researchQuestion,
  sessionId,
  availableMappingFiles,
  selectedMappingFiles,
  includeMappingUploadSlot
}: BuildDcrWizardRequestInput) => ({
  ...dataCleanRoom,
  include_shuffled_samples: shuffledSampleSettings,
  additional_analysts: additionalAnalysts,
  excluded_data_owners: excludedDataOwners,
  airlock_settings: airlockSettings,
  dcr_name: dcrName,
  research_question: researchQuestion,
  session_id: sessionId,
  selected_mapping_files: availableMappingFiles
    .filter(mapping => selectedMappingFiles[mapping.filename] !== false)
    .map(mapping => ({
      filename: mapping.filename,
      filepath: mapping.filepath,
      display_name: mapping.display_name,
      cohorts: mapping.cohorts
    })),
  include_mapping_upload_slot: includeMappingUploadSlot
});

export const sortedEnabledKeys = (settings: Record<string, boolean>): string[] =>
  Object.entries(settings)
    .filter(([, enabled]) => enabled !== false)
    .map(([key]) => key)
    .sort();
