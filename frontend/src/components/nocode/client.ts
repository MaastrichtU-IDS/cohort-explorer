// API client + shared types for the no-code DCR wizard.
import {apiUrl} from '@/utils';

export type Kind = 'stratified' | 'correlation' | 'crosstab' | 'compare';

export interface KindMeta {
  label: string;
  explain?: string;
  roles: string[];
  optional_roles?: string[];
  min_cohorts: number;
  max_cohorts: number;
  blurb: string;
}

export interface CategoryInfo {
  value: string;
  label: string;
  concept_code?: string | null;
  omop_id?: string | null;
  concept_name?: string | null;
}

export interface VarInfo {
  cohort_id: string;
  var_name: string;
  var_label: string;
  var_type: string;
  // From the dictionary: categorical = categories declared; numeric = numeric VARTYPE;
  // other = free text / identifier / date, not usable in an analysis role.
  kind: 'categorical' | 'numeric' | 'other';
  units: string;
  unit_concept_name?: string;
  concept_code: string;
  concept_name: string;
  omop_id: string;
  omop_domain?: string;
  visits?: string;
  count?: number | null;
  categories: CategoryInfo[];
  // From the cohort's EDA profiling output, when it exists (normalized over both
  // EDA formats). Lets the user spot scale/unit differences across cohorts.
  eda?: EdaStats | null;
  score?: number;
  equivalents?: {cohort_id: string; var_name: string; var_label: string}[];
}

export interface EdaStats {
  n?: number | null;
  missing_pct?: number | null;
  mean?: number | null;
  std?: number | null;
  median?: number | null;
  min?: number | null;
  max?: number | null;
  q1?: number | null;
  q3?: number | null;
  n_unique?: number | null;
  type?: string | null;
  distribution?: {value: string; label: string; count?: number | null; pct?: number | null}[] | null;
}

const fmtNum = (x: number | null | undefined): string | null => {
  if (x === null || x === undefined || Number.isNaN(x)) return null;
  const abs = Math.abs(x);
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
  return x.toLocaleString(undefined, {maximumFractionDigits: digits});
};

// One muted line summarising a variable's scale: "n 1,245 · mean 78.4 (sd 12.3) · 38 to 160".
export function edaLine(v: {kind: string; eda?: EdaStats | null}): string {
  const e = v.eda;
  if (!e) return '';
  const parts: string[] = [];
  if (e.n != null) parts.push(`n ${fmtNum(e.n)}`);
  if (e.missing_pct != null && e.missing_pct > 0) parts.push(`${fmtNum(e.missing_pct)}% missing`);
  if (v.kind === 'numeric') {
    if (e.mean != null) parts.push(`mean ${fmtNum(e.mean)}${e.std != null ? ` (sd ${fmtNum(e.std)})` : ''}`);
    if (e.median != null) parts.push(`median ${fmtNum(e.median)}`);
    if (e.min != null && e.max != null) parts.push(`${fmtNum(e.min)} to ${fmtNum(e.max)}`);
  } else if (e.n_unique != null) {
    parts.push(`${fmtNum(e.n_unique)} distinct values`);
  }
  return parts.join(' · ');
}

// Categorical frequencies as "I 12% · II 45% · III 35% · IV 8%" (up to 8 shown).
export function categoryLine(v: {kind: string; eda?: EdaStats | null}): string {
  const d = v.eda?.distribution;
  if (!d || d.length === 0) return '';
  const shown = d.slice(0, 8).map(x => `${x.label && x.label !== x.value ? `${x.value} ${x.label}` : x.value}${x.pct != null ? ` ${fmtNum(x.pct)}%` : ''}`);
  return shown.join(' · ') + (d.length > 8 ? ` · +${d.length - 8} more` : '');
}

// Explicit "unit: kg" / "visit: baseline time" lines (only the ones that are known).
export function unitVisitLines(v: {units?: string; visits?: string}): string[] {
  const lines: string[] = [];
  if (v.units) lines.push(`unit: ${v.units}`);
  if (v.visits && v.visits.toLowerCase() !== 'none') lines.push(`visit: ${v.visits}`);
  return lines;
}

export interface Evidence {
  type: 'code' | 'text' | 'cache' | 'ai' | 'manual' | 'warning' | 'label';
  detail?: string;
  score?: number;
  file?: string;
  status?: string;
  relation?: string;
  value_mapping?: {source?: Record<string, string>; target?: Record<string, string>};
  harmonized_variable?: string;
}

export interface Candidate extends VarInfo {
  evidence: Evidence[];
}

// One harmonized variable: which variable plays this role in each cohort,
// and how their values/units are aligned.
export interface HVar {
  harmonized_name: string;
  label: string;
  type: 'categorical' | 'numeric';
  unit?: string;
  members: Record<string, {var_name: string; var_label?: string; unit?: string; kind?: string}>;
  value_map: Record<string, Record<string, string>>; // cohort -> raw value -> harmonized value
  unit_conversion: Record<string, {factor: number | null; from: string; to: string}>;
  evidence: Evidence[];
  notes?: string;
}

export interface MappingSpec {
  id?: string;
  name: string;
  cohorts: string[];
  variables: HVar[];
  created_by?: string;
  created_at?: string;
  sources?: string[]; // cached mapping files consulted
}

export interface AnalysisSpec {
  analysis: {
    kind: Kind;
    title: string;
    suppression_k: number;
    bins: number;
    roles: Record<string, string>;
  };
  cohorts: string[];
  // 'full' = the cohorts' real data; 'shuffled' = the platform-uploaded shuffled
  // samples (code testing only — every figure carries a notice).
  data_source?: 'full' | 'shuffled';
  mapping: MappingSpec;
}

export interface ValueCluster {
  harmonized: string;
  sources: Record<string, string[]>;
  evidence: Evidence[];
  cohorts_covered: string[];
  complete: boolean;
}

async function post<T>(path: string, body: any): Promise<T> {
  const res = await fetch(`${apiUrl}${path}`, {
    method: 'POST',
    credentials: 'include',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${apiUrl}${path}`, {credentials: 'include'});
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch {
      /* not json */
    }
    const err: any = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export const fetchKinds = () => get<{kinds: Record<Kind, KindMeta>}>('/api/nocode/kinds');

export const fetchAllVariables = (cohortIds: string[]) =>
  post<{variables: VarInfo[]}>('/api/nocode/variables', {cohort_ids: cohortIds});

export const searchVariables = (cohortIds: string[], query: string, mode: 'any' | 'all' | 'exact' = 'any') =>
  post<{results: VarInfo[]; total: number}>('/api/nocode/search', {cohort_ids: cohortIds, query, mode});

export const suggestMatches = (anchor: {cohort_id: string; var_name: string}, targets: string[], cachedFiles: string[]) =>
  post<{anchor: VarInfo; candidates: Record<string, Candidate[]>}>('/api/nocode/suggest', {
    anchor,
    targets,
    cached_files: cachedFiles
  });

export const suggestValues = (members: Record<string, string>, cachedFiles: string[]) =>
  post<{clusters: ValueCluster[]; members: Record<string, {var_name: string; categories: CategoryInfo[]}>}>(
    '/api/nocode/suggest-values',
    {members, cached_files: cachedFiles}
  );

export const aiSuggest = (body: any) => post<{task: string; result: any; model: string}>('/api/nocode/ai-suggest', body);

export const fetchCachedMappings = (cohortIds: string[]) =>
  get<{pairs: any[]; files: {filename: string; source: string; target: string; generated_at: string; size_kb: number}[]}>(
    `/api/nocode/cached-mappings?cohort_ids=${encodeURIComponent(cohortIds.join(','))}`
  );

export const saveMapping = (spec: MappingSpec) => post<{ok: boolean; id: string}>('/api/nocode/mappings', spec);
export const listMappings = (cohortIds: string[]) =>
  get<{mappings: {id: string; name: string; cohorts: string[]; variables: number; created_by: string; updated_at: string}[]}>(
    `/api/nocode/mappings?cohort_ids=${encodeURIComponent(cohortIds.join(','))}`
  );
export const loadMapping = (id: string) => get<MappingSpec>(`/api/nocode/mappings/${encodeURIComponent(id)}`);

export const describeSpec = (spec: AnalysisSpec) => post<{description: string; script: string}>('/api/nocode/describe', spec);

export const createNocodeDcr = (body: any) => post<any>('/create-live-compute-dcr', body);

export const runNocode = (dcrId: string, nodeName: string) =>
  post<{ok: boolean; summary: any}>(`/api/nocode/run/${encodeURIComponent(dcrId)}`, {node_name: nodeName});
export const fetchNocodeResults = (dcrId: string, nodeName: string) =>
  get<any>(`/api/nocode/results/${encodeURIComponent(dcrId)}/${encodeURIComponent(nodeName)}`);
export const resultFileUrl = (dcrId: string, nodeName: string, path: string) =>
  `${apiUrl}/api/nocode/results/${encodeURIComponent(dcrId)}/${encodeURIComponent(nodeName)}/file/${path}`;

export async function fetchResultBlob(dcrId: string, nodeName: string, path: string): Promise<string> {
  const res = await fetch(resultFileUrl(dcrId, nodeName, path), {credentials: 'include'});
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return URL.createObjectURL(await res.blob());
}

export async function fetchResultText(dcrId: string, nodeName: string, path: string): Promise<string> {
  const res = await fetch(resultFileUrl(dcrId, nodeName, path), {credentials: 'include'});
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return res.text();
}

// ---- helpers shared by the UI -------------------------------------------------

export function slugName(text: string): string {
  return (
    text
      .normalize('NFKD')
      .replace(/[̀-ͯ]/g, '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '') || 'variable'
  );
}

export function newHVar(anchor: VarInfo): HVar {
  const base = anchor.concept_name || anchor.var_label || anchor.var_name;
  return {
    harmonized_name: slugName(base).slice(0, 40),
    label: anchor.concept_name || anchor.var_label || anchor.var_name,
    type: anchor.kind === 'categorical' ? 'categorical' : 'numeric',
    unit: anchor.units || '',
    members: {[anchor.cohort_id]: {var_name: anchor.var_name, var_label: anchor.var_label, unit: anchor.units, kind: anchor.kind}},
    value_map: {},
    unit_conversion: {},
    evidence: [{type: 'manual', detail: `anchor ${anchor.var_name} [${anchor.cohort_id}]`}]
  };
}

// Human-readable provenance line, mirroring what the enclave script prints
// under every figure.
export function provenanceLine(hv: HVar): string {
  const members = Object.entries(hv.members)
    .filter(([, m]) => m && m.var_name)
    .map(([cohort, m]) => {
      let piece = `${m.var_name} [${cohort}]`;
      const vm = hv.value_map?.[cohort] || {};
      const pairs = Object.entries(vm);
      if (pairs.length) {
        piece += ` (${pairs.slice(0, 6).map(([k, v]) => `${k}→${v}`).join(', ')}${pairs.length > 6 ? ', …' : ''})`;
      }
      const conv = hv.unit_conversion?.[cohort];
      if (conv && conv.factor && conv.factor !== 1) piece += ` (×${conv.factor} ${conv.from || '?'}→${conv.to || '?'})`;
      return piece;
    });
  return `${hv.harmonized_name} := ${members.join(' | ')}`;
}
