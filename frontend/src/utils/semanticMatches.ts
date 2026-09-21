import {Cohort, Variable} from '@/types';

// Semantic matches: a variable's semantic matches are the variables of OTHER
// cohorts that share one of its standard identifiers (concept code or OMOP
// ID); they decide which variables are marked and how they rank. Same-cohort
// siblings (e.g. the same concept at several visits) are recorded alongside,
// for display only. Identifiers may be pipe-separated lists, are compared
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
  // Matches in OTHER cohorts, sorted by cohort id, then variable name.
  matches: SemanticMatch[];
  // Distinct other cohorts holding a match, sorted.
  cohortIds: string[];
  // Other variables of the SAME cohort sharing an identifier, sorted by name.
  sameCohort: SemanticMatch[];
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
    const foundSame = new Map<string, SemanticMatch>();
    const matchedValues: Record<MatchIdentifier, string[]> = {concept_code: [], omop_id: []};
    for (const id of identifiers) {
      for (const value of splitIdentifierValues(variable[id])) {
        const hits = byValue[id].get(normalize(value));
        if (!hits) continue;
        let matched = false;
        for (const hit of hits) {
          const same = hit.cohortId === cohortId;
          if (same && hit.varName === varName) continue;
          if (!same) matched = true;
          const bucket = same ? foundSame : found;
          const key = semanticMatchKey(hit.cohortId, hit.varName);
          const existing = bucket.get(key);
          if (existing) {
            if (!existing.matchedOn.includes(id)) existing.matchedOn.push(id);
          } else {
            bucket.set(key, {cohortId: hit.cohortId, varName: hit.varName, variable: hit.variable, matchedOn: [id]});
          }
        }
        if (matched && !matchedValues[id].includes(value)) matchedValues[id].push(value);
      }
    }
    // Only cross-cohort matches mark a variable; same-cohort siblings ride along.
    if (found.size === 0) continue;
    const matches = Array.from(found.values()).sort(
      (a, b) => a.cohortId.localeCompare(b.cohortId) || a.varName.localeCompare(b.varName)
    );
    const cohortIds = Array.from(new Set(matches.map(c => c.cohortId))).sort();
    const sameCohort = Array.from(foundSame.values()).sort((a, b) => a.varName.localeCompare(b.varName));
    byVariable.set(semanticMatchKey(cohortId, varName), {matches, cohortIds, sameCohort, matchedValues});
  }

  return {byVariable};
}
