// Shared client for the study x theme view of the metadata clustering
// (backend: /api/metadata-clustering/themes), used by /metadata-heatmap and
// /metadata-constellation.
import {apiUrl} from '@/utils';

export interface ThemeParent {
  id: string;
  label: string;
}

export interface Theme {
  id: string;
  label: string;
  parent: string;
  raw: string[];
  studies: number;
  // Inverse-frequency weight, log(N/n)/log(N): 0 when every study carries it; rarer is heavier.
  specificity: number;
  // Binary entropy of the carrier fraction, in bits: 1 when half the studies carry it, 0 when all or
  // (almost) none do. The default weight: it penalises universal and one-pair themes alike.
  balance: number;
  excluded: boolean;
}

export type ThemeWeight = 'balance' | 'specificity';

export function themeWeight(t: {balance: number; specificity: number}, mode: ThemeWeight): number {
  return mode === 'balance' ? t.balance : t.specificity;
}

export interface ThemeExcerpt {
  field: string;
  text: string;
  with: string;
  with_text: string;
  strength: 'strong' | 'partial' | null;
}

export interface ThemeCell {
  strength: number;
  count: number;
  excerpts: ThemeExcerpt[];
}

export interface SharedTheme {
  theme: string;
  parent: string;
  w: number;
  specificity: number;
  balance: number;
}

export interface StudyPair {
  a: string;
  b: string;
  shared: SharedTheme[];
  n: number;
  // Sum of min(strength) x balance over shared themes (the server's default weight).
  strength: number;
  strength_specificity: number;
  raw_strength: number;
}

export interface ThemeView {
  finished: string | null;
  model: string | null;
  studies: string[];
  n_studies: number;
  parents: ThemeParent[];
  themes: Theme[];
  table: Record<string, Record<string, ThemeCell>>;
  pairs: StudyPair[];
  excluded: string[];
}

export async function fetchThemeView(): Promise<ThemeView> {
  const r = await fetch(`${apiUrl}/api/metadata-clustering/themes`, {credentials: 'include'});
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      detail = (await r.json()).detail || detail;
    } catch {
      /* not JSON */
    }
    const err = new Error(detail) as Error & {status?: number};
    err.status = r.status;
    throw err;
  }
  return r.json();
}

export async function setThemeExcluded(themeId: string, excluded: boolean): Promise<string[]> {
  const r = await fetch(`${apiUrl}/api/metadata-clustering/themes/exclude`, {
    method: 'POST',
    credentials: 'include',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({theme_id: themeId, excluded})
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()).excluded || [];
}

// Validated categorical palette (dataviz skill reference): slots 1, 2, 3, 4, 7, 5, 6, 8.
// Parents take hues in the fixed order they come from the backend; a 9th folds into grey.
const PARENT_HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e87ba4', '#008300', '#e34948'];

export function parentColor(parents: ThemeParent[], parentId: string): string {
  const i = parents.findIndex(p => p.id === parentId);
  return i >= 0 && i < PARENT_HUES.length ? PARENT_HUES[i] : '#94a3b8';
}

/** Sequential shade for a cell: one hue, light to dark, by strength in [0, 1]. */
export function strengthColor(t: number): string {
  const v = Math.min(1, Math.max(0, t));
  return `hsl(215, 70%, ${Math.round(94 - 48 * v)}%)`;
}

/** Themes that survive the admin exclusions and the weight floor. */
export function activeThemes(view: ThemeView, minWeight: number, mode: ThemeWeight = 'balance'): Theme[] {
  return view.themes.filter(t => !t.excluded && t.studies > 0 && themeWeight(t, mode) >= minWeight);
}

/** Pairs recomputed for the chosen weight and floor (the backend already applied exclusions). */
export function pairsAbove(view: ThemeView, minWeight: number, mode: ThemeWeight = 'balance'): StudyPair[] {
  return view.pairs
    .map(p => {
      const shared = p.shared.filter(s => themeWeight(s, mode) >= minWeight);
      const weighted = shared.reduce((acc, s) => acc + s.w * themeWeight(s, mode), 0);
      return {...p, shared, n: shared.length, strength: weighted, strength_specificity: weighted, raw_strength: shared.reduce((acc, s) => acc + s.w, 0)};
    })
    .filter(p => p.n > 0);
}

export function dominantParent(shared: SharedTheme[], mode: ThemeWeight = 'balance'): string {
  const sums: Record<string, number> = {};
  shared.forEach(s => {
    sums[s.parent] = (sums[s.parent] || 0) + s.w * (0.2 + themeWeight(s, mode));
  });
  return Object.entries(sums).sort((x, y) => y[1] - x[1])[0]?.[0] || '';
}
