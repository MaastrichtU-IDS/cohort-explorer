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

// Pseudo raw value standing for empty cells, NaN and declared missing codes.
// Pseudo-key in value_map carrying the policy for empty / coded-missing values:
// "" = those patients are excluded, MISSING_LABEL = kept as one category.
export const MISSING_KEY = '__MISSING__';
export const MISSING_LABEL = '<missing>';
export const displayRaw = (raw: string) => (raw === MISSING_KEY ? '(missing)' : raw);

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
  system?: string; // for code evidence: 'SNOMED', 'LOINC', 'OMOP ID', ...
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
  // True once the user moved a raw value between rows (or to/from missing):
  // the value alignment is then a judgement, not just what the codes dictate.
  value_map_edited?: boolean;
}

// "_pooled" when the alignment is natural (numeric without unit conversion;
// categorical with every row matched by a standard code and untouched),
// "_harmonized" when a real harmonization decision was made.
export function nameSuffix(hv: HVar, rowsMatchedByCode: boolean): '_pooled' | '_harmonized' {
  if (hv.type === 'numeric') {
    const converted = Object.values(hv.unit_conversion || {}).some(c => c && c.factor !== null && c.factor !== undefined && c.factor !== 1);
    return converted ? '_harmonized' : '_pooled';
  }
  return hv.value_map_edited || !rowsMatchedByCode ? '_harmonized' : '_pooled';
}

export function withSuffix(name: string, suffix: '_pooled' | '_harmonized'): string {
  return name.replace(/_(pooled|harmonized)$/, '') + suffix;
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

export const aiName = (variables: {cohort_id: string; var_name: string; var_label?: string}[]) =>
  post<{name: string; label: string; source: string}>('/api/nocode/ai-name', {variables});

export type AiListingMode = {mode: 'top' | 'all'; listed: number; total: number};
export const aiSuggest = (body: any) =>
  post<{task: string; result: any; model: string; modes?: Record<string, AiListingMode>}>('/api/nocode/ai-suggest', body);

// Name of the room following NoCode-<Type>-<Variables>-<Cohorts|Ncohorts>-<MonDD>,
// with short forms of the variables/cohorts from the local LLM when available.
export const aiDcrName = (body: {kind: Kind; roles: Record<string, string>; cohorts: string[]; variables: HVar[]}) =>
  post<{name: string; source: string}>('/api/nocode/ai-dcr-name', body);

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
// 'TIME-CHF' -> 'TIME': a two-part cohort name keeps its first part.
export const shortCohort = (c: string) => {
  const parts = c.split('-');
  return parts.length === 2 && parts[0] ? parts[0] : c;
};

// The compact line printed under the figures (mirrors provenance_lines in the
// enclave script): "hname: TIME::BNP1 -- CHECK::NTBNP (same LOINC 33762-6)".
// Value maps and the other evidence go to provenance.md, not the figures.
export function provenanceLine(hv: HVar): string {
  const members = Object.entries(hv.members)
    .filter(([, m]) => m && m.var_name)
    .map(([cohort, m]) => {
      let piece = `${shortCohort(cohort)}::${m.var_name}`;
      const conv = hv.unit_conversion?.[cohort];
      if (conv && conv.factor && conv.factor !== 1) piece += ` (x${conv.factor})`;
      return piece;
    });
  const codes = Array.from(new Set(hv.evidence.filter(e => e.type === 'code').map(e => `same ${e.system || 'code'} ${e.detail || ''}`.trim())));
  return `${hv.harmonized_name}: ${members.join(' -- ')}${codes.length ? ` (${codes.join('; ')})` : ''}`;
}
