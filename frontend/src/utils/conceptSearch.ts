export const ACCEPTED_CONCEPT_DOMAINS = [
  'Condition',
  'Device',
  'Drug',
  'Geography',
  'Meas Value',
  'Measurement',
  'Metadata',
  'Observation',
  'Procedure',
  'Spec Anatomic Site',
  'Specimen',
  'Condition Status',
  'Condition/Device',
  'Condition/Meas',
  'Condition/Obs',
  'Condition/Procedure',
  'Cost',
  'Currency',
  'Device/Drug',
  'Device/Procedure',
  'Drug/Procedure',
  'Episode',
  'Ethnicity',
  'Gender',
  'Language',
  'Meas Value Operator',
  'Meas/Procedure',
  'Note',
  'Obs/Procedure',
  'Payer',
  'Person',
  'Place of Service',
  'Plan',
  'Plan Stop Reason',
  'Provider',
  'Race',
  'Regimen',
  'Relationship',
  'Revenue Code',
  'Route',
  'Spec Disease Status',
  'Sponsor',
  'Type Concept',
  'Unit',
  'Visit'
] as const;

export const resolveInitialConceptDomains = (domain: string): string[] => {
  const normalized = domain.trim().toLocaleLowerCase();
  const compatibleDomain = ACCEPTED_CONCEPT_DOMAINS.find(candidate => candidate.toLocaleLowerCase() === normalized);
  return compatibleDomain ? [compatibleDomain] : [...ACCEPTED_CONCEPT_DOMAINS];
};

export type ConceptSuggestionState = 'idle' | 'loading' | 'empty' | 'error' | 'results';

export const resolveConceptSuggestionState = ({
  isLoading,
  hasSearched,
  errorMsg,
  resultCount
}: {
  isLoading: boolean;
  hasSearched: boolean;
  errorMsg: string;
  resultCount: number;
}): ConceptSuggestionState => {
  if (isLoading) return 'loading';
  if (errorMsg) return 'error';
  if (resultCount > 0) return 'results';
  return hasSearched ? 'empty' : 'idle';
};
