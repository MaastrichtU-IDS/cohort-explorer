import {Variable} from '@/types';
import {parseEdaJson} from '@/utils/edaParsing';

// Summary statistics of one variable, from the cohort's EDA output.
export interface SummaryStats {
  median?: number;
  min?: number;
  max?: number;
}

// Keyed by lowercased, trimmed variable name.
export type SummaryStatsByName = Record<string, SummaryStats>;

// Compact number for a card's numeric summary line ("120", "2.5", "3.14").
export const fmtStat = (x: any): string => {
  if (x === null || x === undefined || x === '') return '';
  const n = Number(x);
  if (!Number.isFinite(n)) return String(x);
  return Number.isInteger(n) ? String(n) : n.toFixed(2).replace(/\.?0+$/, '');
};

export const hasValue = (x: any): boolean => x !== null && x !== undefined && x !== '';

export const isNumericVariable = (v: Variable): boolean =>
  !(v.categories?.length > 0) && ['INT', 'FLOAT'].includes(String(v.var_type || '').toUpperCase());

// Coarse kind of a variable for display: what a reader wants to know before
// the raw dictionary type (INT / FLOAT / STR / DATETIME).
export const variableKind = (v: Variable): string => {
  if (v.categories?.length > 0) return 'Categorical';
  const t = String(v.var_type || '').toUpperCase();
  if (t === 'INT' || t === 'FLOAT') return 'Numeric';
  if (t === 'DATETIME' || t === 'DATE') return 'Date';
  if (t === 'STR' || t === 'STRING' || t === 'TEXT') return 'Text';
  return v.var_type || '—';
};

// Summary statistics of a cohort's variables (median; min/max as a fallback
// when the dictionary has none). {} when the cohort has no summary statistics
// or they cannot be read, so callers never wait on an error.
export async function loadEdaSummaryStats(cohortId: string): Promise<SummaryStatsByName> {
  try {
    const res = await fetch(`/api/cohort-eda-output/${encodeURIComponent(cohortId)}`);
    if (!res.ok) return {};
    const raw = await res.json();
    if (!raw) return {};
    const data = parseEdaJson(raw);
    const map: SummaryStatsByName = {};
    (data?.variables || []).forEach(v => {
      map[v.name.toLowerCase().trim()] = {median: v.median, min: v.min, max: v.max};
    });
    return map;
  } catch {
    return {};
  }
}

export const statsFor = (stats: SummaryStatsByName | null | undefined, varName: string): SummaryStats | undefined =>
  stats?.[String(varName).toLowerCase().trim()];
