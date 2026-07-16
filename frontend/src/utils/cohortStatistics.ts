import type {Cohort} from '@/types';

export interface CohortStatistics {
  totalCohorts: number;
  cohortsWithMetadata: number;
  cohortsWithAggregateAnalysis: number;
  totalPatients: number;
  patientsInCohortsWithMetadata: number;
  totalVariables: number;
}

type CohortSnapshot = Omit<Cohort, 'study_participants'> & {
  study_participants: string | number | null | undefined;
};

const parseParticipants = (participants: CohortSnapshot['study_participants']): number => {
  if (participants === undefined || participants === null) return 0;
  if (typeof participants === 'number') return participants;
  const numericPart = participants.split(' ')[0].replace(/[^0-9,]/g, '');
  const parsedValue = Number.parseInt(numericPart.replace(/,/g, ''), 10);
  return Number.isNaN(parsedValue) ? 0 : parsedValue;
};

export const calculateCohortStatistics = async (
  cohortsData: Record<string, CohortSnapshot>,
  hasAggregateAnalysis: (cohortId: string) => Promise<boolean>
): Promise<CohortStatistics> => {
  const cohorts = Object.values(cohortsData);
  const cohortsWithMetadata = cohorts.filter(cohort => Object.keys(cohort.variables || {}).length > 0);
  const aggregateResults = await Promise.all(cohorts.map(cohort => hasAggregateAnalysis(cohort.cohort_id)));

  return {
    totalCohorts: cohorts.length,
    cohortsWithMetadata: cohortsWithMetadata.length,
    cohortsWithAggregateAnalysis: aggregateResults.filter(Boolean).length,
    totalPatients: cohorts.reduce((sum, cohort) => sum + parseParticipants(cohort.study_participants), 0),
    patientsInCohortsWithMetadata: cohortsWithMetadata.reduce(
      (sum, cohort) => sum + parseParticipants(cohort.study_participants),
      0
    ),
    totalVariables: cohorts.reduce((sum, cohort) => sum + Object.keys(cohort.variables || {}).length, 0)
  };
};
