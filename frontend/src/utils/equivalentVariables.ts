import type {SearchMode, SearchScope} from '@/utils/cohortFiltering';
import {matchesSearchTerms} from '@/utils/cohortFiltering';

interface EquivalentVariable {
  concept_name?: string | null;
  concept_code?: string | null;
  omop_id?: string | number | null;
}

interface EquivalentCohort {
  variables?: Record<string, EquivalentVariable>;
}

export interface EquivalentNamesByCohort {
  cohortId: string;
  names: string[];
  isMatched: boolean;
}

export interface EquivalentVariableGroup {
  code: string;
  conceptName: string | null;
  namesByCohort: EquivalentNamesByCohort[];
}

export interface EquivalentVariableGroups {
  conceptGroups: EquivalentVariableGroup[];
  omopGroups: EquivalentVariableGroup[];
  uncoded: {cohortId: string; names: string[]}[];
}

export const groupEquivalentVariables = (
  cohortsData: Record<string, EquivalentCohort>,
  searchTerms: string[],
  _searchMode: SearchMode,
  searchScope: SearchScope
): EquivalentVariableGroups | null => {
  if (searchTerms.length === 0 || searchScope === 'cohorts') return null;

  const matchedConceptCodes = new Map<string, {displayCode: string; names: Set<string>}>();
  const matchedOmopIds = new Map<string, Set<string>>();
  const uncodedByCohort = new Map<string, string[]>();

  Object.entries(cohortsData).forEach(([cohortId, cohort]) => {
    Object.entries(cohort.variables || {}).forEach(([variableName, variable]) => {
      const nameMatches =
        matchesSearchTerms(variableName, searchTerms, 'and') ||
        matchesSearchTerms(variable.concept_name, searchTerms, 'and');
      if (!nameMatches) return;

      const rawCode = variable.concept_code?.trim() || '';
      const rawOmop = variable.omop_id ? String(variable.omop_id).trim() : '';
      if (rawCode) {
        const key = rawCode.toUpperCase();
        if (!matchedConceptCodes.has(key)) {
          matchedConceptCodes.set(key, {displayCode: rawCode, names: new Set()});
        }
        matchedConceptCodes.get(key)!.names.add(variableName);
      }
      if (rawOmop) {
        if (!matchedOmopIds.has(rawOmop)) matchedOmopIds.set(rawOmop, new Set());
        matchedOmopIds.get(rawOmop)!.add(variableName);
      }
      if (!rawCode && !rawOmop) {
        if (!uncodedByCohort.has(cohortId)) uncodedByCohort.set(cohortId, []);
        uncodedByCohort.get(cohortId)!.push(variableName);
      }
    });
  });

  const buildEntries = (
    keyType: 'concept_code' | 'omop_id',
    key: string,
    matchedVariableNames: Set<string>
  ): {conceptName: string | null; cohortEntries: EquivalentNamesByCohort[]} => {
    let conceptName: string | null = null;
    const cohortEntries: EquivalentNamesByCohort[] = [];

    Object.entries(cohortsData).forEach(([cohortId, cohort]) => {
      const names = Object.entries(cohort.variables || {})
        .filter(([, variable]) =>
          keyType === 'concept_code'
            ? Boolean(variable.concept_code && variable.concept_code.trim().toUpperCase() === key)
            : Boolean(variable.omop_id && String(variable.omop_id).trim() === key)
        )
        .map(([variableName, variable]) => {
          if (!conceptName && variable.concept_name) conceptName = variable.concept_name;
          return variableName;
        })
        .sort();

      if (names.length > 0) {
        cohortEntries.push({
          cohortId,
          names,
          isMatched: names.some(name => matchedVariableNames.has(name))
        });
      }
    });

    cohortEntries.sort((left, right) => {
      if (left.isMatched === right.isMatched) return left.cohortId.localeCompare(right.cohortId);
      return left.isMatched ? 1 : -1;
    });
    return {conceptName, cohortEntries};
  };

  const conceptGroups = Array.from(matchedConceptCodes.entries()).map(([key, match]) => {
    const {conceptName, cohortEntries} = buildEntries('concept_code', key, match.names);
    return {code: match.displayCode, conceptName, namesByCohort: cohortEntries};
  });
  const omopGroups = Array.from(matchedOmopIds.entries()).map(([omopId, matchedVariableNames]) => {
    const {conceptName, cohortEntries} = buildEntries('omop_id', omopId, matchedVariableNames);
    return {code: omopId, conceptName, namesByCohort: cohortEntries};
  });
  const countNames = (group: EquivalentVariableGroup): number =>
    group.namesByCohort.reduce((total, entry) => total + entry.names.length, 0);
  conceptGroups.sort((left, right) => countNames(right) - countNames(left));
  omopGroups.sort((left, right) => countNames(right) - countNames(left));

  const uncoded = Array.from(uncodedByCohort.entries())
    .map(([cohortId, names]) => ({cohortId, names: names.slice().sort()}))
    .sort((left, right) => left.cohortId.localeCompare(right.cohortId));

  if (conceptGroups.length === 0 && omopGroups.length === 0 && uncoded.length === 0) return null;
  return {conceptGroups, omopGroups, uncoded};
};
