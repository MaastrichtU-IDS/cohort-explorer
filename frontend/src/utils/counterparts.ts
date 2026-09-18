import {Cohort, Variable} from '@/types';

// Cross-cohort counterparts: a variable's counterparts are the variables of
// OTHER cohorts that share one of its standard identifiers (concept code or
// OMOP ID). Same-cohort siblings (e.g. the same concept at several visits) are
// not counterparts. Matching follows the Concept Clusters page: identifiers
// may be pipe-separated lists, are compared case-insensitively, and empty /
// "NA" values are ignored. "0" is skipped as well: OMOP concept id 0 is the
// "no matching concept" sentinel and would link every unmapped variable.

export type CounterpartIdentifier = 'concept_code' | 'omop_id';

export const IDENTIFIER_LABELS: Record<CounterpartIdentifier, string> = {
  concept_code: 'concept code',
  omop_id: 'OMOP ID',
};

export interface Counterpart {
  cohortId: string;
  varName: string;
  variable: Variable;
  // Which of the identifiers this counterpart shares with the variable.
  matchedOn: CounterpartIdentifier[];
}

export interface VariableCounterparts {
  // Sorted by cohort id, then variable name.
  counterparts: Counterpart[];
  // Distinct other cohorts holding a counterpart, sorted.
  cohortIds: string[];
  // The variable's own identifier values that found a match (as written in
  // its dictionary), per identifier type.
  matchedValues: Record<CounterpartIdentifier, string[]>;
}

export interface CounterpartIndex {
  byVariable: Map<string, VariableCounterparts>;
}

export const counterpartKey = (cohortId: string, varName: string): string => `${cohortId}::${varName}`;

export const EMPTY_COUNTERPART_INDEX: CounterpartIndex = {byVariable: new Map()};

export function splitIdentifierValues(raw: unknown): string[] {
  if (raw === null || raw === undefined) return [];
  return String(raw)
    .split('|')
    .map(v => v.trim())
    .filter(v => v && v.toLowerCase() !== 'na' && v !== '0');
}

const normalize = (v: string): string => v.trim().toLowerCase();

interface IndexedVariable {
  cohortId: string;
  varName: string;
  variable: Variable;
}

export function buildCounterpartIndex(cohortsData: Record<string, Cohort> | null | undefined): CounterpartIndex {
  const byVariable = new Map<string, VariableCounterparts>();
  if (!cohortsData) return {byVariable};

  const identifiers: CounterpartIdentifier[] = ['concept_code', 'omop_id'];
  const byValue: Record<CounterpartIdentifier, Map<string, IndexedVariable[]>> = {
    concept_code: new Map(),
    omop_id: new Map(),
  };

  const all: IndexedVariable[] = [];
  for (const [cohortId, cohort] of Object.entries(cohortsData)) {
    if (!cohort?.variables) continue;
    for (const [varName, variable] of Object.entries(cohort.variables)) {
      const entry: IndexedVariable = {cohortId, varName, variable};
      all.push(entry);
      for (const id of identifiers) {
        for (const value of splitIdentifierValues(variable[id])) {
          const key = normalize(value);
          const bucket = byValue[id].get(key);
          if (bucket) bucket.push(entry);
          else byValue[id].set(key, [entry]);
        }
      }
    }
  }

  for (const {cohortId, varName, variable} of all) {
    const found = new Map<string, Counterpart>();
    const matchedValues: Record<CounterpartIdentifier, string[]> = {concept_code: [], omop_id: []};
    for (const id of identifiers) {
      for (const value of splitIdentifierValues(variable[id])) {
        const hits = byValue[id].get(normalize(value));
        if (!hits) continue;
        let matched = false;
        for (const hit of hits) {
          if (hit.cohortId === cohortId) continue;
          matched = true;
          const key = counterpartKey(hit.cohortId, hit.varName);
          const existing = found.get(key);
          if (existing) {
            if (!existing.matchedOn.includes(id)) existing.matchedOn.push(id);
          } else {
            found.set(key, {cohortId: hit.cohortId, varName: hit.varName, variable: hit.variable, matchedOn: [id]});
          }
        }
        if (matched && !matchedValues[id].includes(value)) matchedValues[id].push(value);
      }
    }
    if (found.size === 0) continue;
    const counterparts = Array.from(found.values()).sort(
      (a, b) => a.cohortId.localeCompare(b.cohortId) || a.varName.localeCompare(b.varName)
    );
    const cohortIds = Array.from(new Set(counterparts.map(c => c.cohortId))).sort();
    byVariable.set(counterpartKey(cohortId, varName), {counterparts, cohortIds, matchedValues});
  }

  return {byVariable};
}
