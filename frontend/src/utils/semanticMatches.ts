import {Cohort, Variable} from '@/types';

// Semantic matches: a variable's semantic matches are the variables of OTHER
// cohorts that share one of its standard identifiers (concept code or OMOP
// ID). Same-cohort siblings (e.g. the same concept at several visits) are not
// matches. Identifiers may be pipe-separated lists, are compared
// case-insensitively, and empty /
// "NA" values are ignored. "0" is skipped as well: OMOP concept id 0 is the
// "no matching concept" sentinel and would link every unmapped variable.

export type MatchIdentifier = 'concept_code' | 'omop_id';

export const IDENTIFIER_LABELS: Record<MatchIdentifier, string> = {
  concept_code: 'concept code',
  omop_id: 'OMOP ID',
};

export interface SemanticMatch {
  cohortId: string;
  varName: string;
  variable: Variable;
  // Which of the identifiers this match shares with the variable.
  matchedOn: MatchIdentifier[];
}

export interface VariableSemanticMatches {
  // Sorted by cohort id, then variable name.
  matches: SemanticMatch[];
  // Distinct other cohorts holding a match, sorted.
  cohortIds: string[];
  // The variable's own identifier values that found a match (as written in
  // its dictionary), per identifier type.
  matchedValues: Record<MatchIdentifier, string[]>;
}

export interface SemanticMatchIndex {
  byVariable: Map<string, VariableSemanticMatches>;
}

export const semanticMatchKey = (cohortId: string, varName: string): string => `${cohortId}::${varName}`;

export const EMPTY_SEMANTIC_MATCH_INDEX: SemanticMatchIndex = {byVariable: new Map()};

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

export function buildSemanticMatchIndex(cohortsData: Record<string, Cohort> | null | undefined): SemanticMatchIndex {
  const byVariable = new Map<string, VariableSemanticMatches>();
  if (!cohortsData) return {byVariable};

  const identifiers: MatchIdentifier[] = ['concept_code', 'omop_id'];
  const byValue: Record<MatchIdentifier, Map<string, IndexedVariable[]>> = {
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
    const found = new Map<string, SemanticMatch>();
    const matchedValues: Record<MatchIdentifier, string[]> = {concept_code: [], omop_id: []};
    for (const id of identifiers) {
      for (const value of splitIdentifierValues(variable[id])) {
        const hits = byValue[id].get(normalize(value));
        if (!hits) continue;
        let matched = false;
        for (const hit of hits) {
          if (hit.cohortId === cohortId) continue;
          matched = true;
          const key = semanticMatchKey(hit.cohortId, hit.varName);
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
    const matches = Array.from(found.values()).sort(
      (a, b) => a.cohortId.localeCompare(b.cohortId) || a.varName.localeCompare(b.varName)
    );
    const cohortIds = Array.from(new Set(matches.map(c => c.cohortId))).sort();
    byVariable.set(semanticMatchKey(cohortId, varName), {matches, cohortIds, matchedValues});
  }

  return {byVariable};
}
