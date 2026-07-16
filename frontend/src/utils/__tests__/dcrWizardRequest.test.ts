import {describe, expect, it} from 'vitest';

import {buildDcrWizardRequest, sortedEnabledKeys} from '@/utils/dcrWizardRequest';

describe('DCR wizard request contract', () => {
  it('projects the same complete payload for preview and creation', () => {
    const request = buildDcrWizardRequest({
      dataCleanRoom: {cohorts: {'TIME-CHF': ['age'], 'GISSI-HF': ['age']}},
      shuffledSampleSettings: {'TIME-CHF': true, 'GISSI-HF': true},
      additionalAnalysts: ['analyst@example.test'],
      excludedDataOwners: ['excluded@example.test'],
      airlockSettings: undefined,
      dcrName: 'Local integration room',
      researchQuestion: 'Can the synthetic cohorts be aggregated?',
      sessionId: 'session-123',
      availableMappingFiles: [
        {
          filename: 'selected.csv',
          filepath: '/safe/selected.csv',
          display_name: 'TIME-CHF → GISSI-HF',
          cohorts: ['TIME-CHF', 'GISSI-HF']
        },
        {
          filename: 'excluded.csv',
          filepath: '/safe/excluded.csv',
          display_name: 'Excluded mapping',
          cohorts: ['A', 'B']
        }
      ],
      selectedMappingFiles: {'excluded.csv': false},
      includeMappingUploadSlot: true
    });

    expect(request).toEqual({
      cohorts: {'TIME-CHF': ['age'], 'GISSI-HF': ['age']},
      include_shuffled_samples: {'TIME-CHF': true, 'GISSI-HF': true},
      additional_analysts: ['analyst@example.test'],
      excluded_data_owners: ['excluded@example.test'],
      airlock_settings: undefined,
      dcr_name: 'Local integration room',
      research_question: 'Can the synthetic cohorts be aggregated?',
      session_id: 'session-123',
      selected_mapping_files: [
        {
          filename: 'selected.csv',
          filepath: '/safe/selected.csv',
          display_name: 'TIME-CHF → GISSI-HF',
          cohorts: ['TIME-CHF', 'GISSI-HF']
        }
      ],
      include_mapping_upload_slot: true
    });
  });

  it('orders enabled review selections deterministically', () => {
    expect(sortedEnabledKeys({'TIME-CHF': true, 'GISSI-HF': true, disabled: false})).toEqual([
      'GISSI-HF',
      'TIME-CHF'
    ]);
  });
});
