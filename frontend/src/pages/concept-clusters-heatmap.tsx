'use client';

import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useCohorts } from '@/components/CohortsContext';
import LoginPrompt from '@/components/LoginPrompt';
import { apiUrl, parseParticipantCount } from '@/utils';
import { Cohort, Variable } from '@/types';
import { Grid, ChevronLeft, ChevronRight, ChevronDown, ChevronUp, Search, Download, ZoomIn, X, EyeOff } from 'react-feather';

const NO_VISIT_KEY = '__no_visit__';
const NO_VISIT_LABEL = '(no visit)';

function splitValues(raw: string | null | undefined): string[] {
  if (!raw) return [];
  return String(raw)
    .split('|')
    .map(v => v.trim())
    .filter(v => v && v.toLowerCase() !== 'na');
}

function normalizeValue(v: string): string {
  return v.trim().toLowerCase();
}

// Standard variable names are matched lowercased; shown in title case
// ("hemoglobin [mass/volume] in blood" -> "Hemoglobin [Mass/Volume] In Blood").
function titleCase(s: string): string {
  return s.replace(/(^|[\s([/\-"])([a-z])/g, (_m, sep: string, ch: string) => sep + ch.toUpperCase());
}

const parseParticipants = (raw: string | null | undefined): number | null => parseParticipantCount(raw);

// --- Visit ordering, ported from the longitudinal analysis logic
// (_visit_sort_key in backend/src/longitudinal_audit.py) ---

const VISIT_UNIT_DAYS: Record<string, number> = {
  d: 1, day: 1, days: 1,
  w: 7, wk: 7, wks: 7, week: 7, weeks: 7,
  m: 30.44, mo: 30.44, mos: 30.44, mon: 30.44, month: 30.44, months: 30.44,
  y: 365.25, yr: 365.25, yrs: 365.25, year: 365.25, years: 365.25,
};
const BASELINE_RE = /(baseline|screening|enrol{1,2}ment|day\s*0\b|week\s*0\b|visit\s*0\b)/i;
const END_RE = /(end\s*of\s*(study|trial)|\bfinal\b|\blast\b|\beos\b|study\s*end|follow[- ]?up\s*end)/i;
const TOKEN_RE = /[a-z]+|\d+(?:\.\d+)?/g;

function visitOffsetDays(text: string): number | null {
  const tokens = text.match(TOKEN_RE) || [];
  for (let i = 0; i < tokens.length; i++) {
    if (!/^\d/.test(tokens[i])) continue;
    for (const neighbour of [i + 1, i - 1]) {
      if (neighbour >= 0 && neighbour < tokens.length && VISIT_UNIT_DAYS[tokens[neighbour]] !== undefined) {
        return parseFloat(tokens[i]) * VISIT_UNIT_DAYS[tokens[neighbour]];
      }
    }
  }
  return null;
}

function visitOrdinal(text: string): number | null {
  const tokens = text.match(TOKEN_RE) || [];
  for (const tok of tokens) {
    if (/^\d/.test(tok)) return parseFloat(tok);
  }
  return null;
}

function visitSortKey(label: string | null | undefined): [number, number, string] {
  if (label === NO_VISIT_KEY) return [5, 0, ''];
  if (!label) return [3, 0, ''];
  const s = String(label).trim().toLowerCase();
  if (BASELINE_RE.test(s)) return [0, 0, s];
  if (END_RE.test(s)) return [4, 0, s];
  const offset = visitOffsetDays(s);
  if (offset !== null) return [1, offset, s];
  const ordinal = visitOrdinal(s);
  if (ordinal !== null) return [2, ordinal, s];
  return [3, 0, s];
}

function compareVisitKeys(a: string, b: string): number {
  const ka = visitSortKey(a);
  const kb = visitSortKey(b);
  if (ka[0] !== kb[0]) return ka[0] - kb[0];
  if (ka[1] !== kb[1]) return ka[1] - kb[1];
  return ka[2].localeCompare(kb[2]);
}

// Non-null observation counts from the EDA output files (variable profiling),
// via GET /eda-observation-counts. The dictionary COUNT column is only a
// fallback: it is declared by the data owners and often 0/empty.
type EdaCounts = Record<
  string,
  { eda_version: string; n_rows: number | null; variables: Record<string, number> }
>;

function edaCountFor(counts: EdaCounts | null, cohortId: string, varName: string): number | undefined {
  return counts?.[cohortId]?.variables?.[String(varName).trim().toLowerCase()];
}

type CountSource = 'eda' | 'dict';

// A variable's non-null count: from the EDA profiling when available, else
// from the metadata dictionary's COUNT column (empty/0 counts as absent there,
// since owners leave the column blank far more often than a variable is truly
// all-missing).
function variableCount(
  counts: EdaCounts | null,
  cohortId: string,
  variable: Variable
): { n: number | null; source: CountSource | null } {
  const eda = edaCountFor(counts, cohortId, variable.var_name);
  if (eda !== undefined) return { n: eda, source: 'eda' };
  const dict = Number(variable.count);
  if (Number.isFinite(dict) && dict > 0) return { n: dict, source: 'dict' };
  return { n: null, source: null };
}

// --- Cohort typing: wide format / long format / no variable profiling ---

type CohortType = 'wide' | 'long' | 'none';

const TYPE_ORDER: Record<CohortType, number> = { wide: 0, long: 1, none: 2 };

const TYPE_LABELS: Record<CohortType, string> = {
  wide: 'wide format',
  long: 'long format',
  none: 'no profiling',
};

const TYPE_CHIP: Record<CohortType, string> = {
  wide: 'bg-sky-100 text-sky-900 border-sky-300',
  long: 'bg-amber-100 text-amber-900 border-amber-300',
  none: 'bg-gray-200 text-gray-600 border-gray-300',
};

const CSV_SUFFIX: Record<CohortType, string> = {
  wide: '',
  long: ' (long data format)',
  none: ' (no profiling yet. counts from metadata dictionary)',
};

// Patient-id concept identifiers, same as the c4 longitudinal script.
const PATIENT_OMOP_IDS = new Set(['4086934', '40757164']);
const PATIENT_CONCEPT_CODE = '184107009';

function isPatientIdOrGenderVar(v: Variable): boolean {
  if (splitValues(v.omop_id as string).some(id => PATIENT_OMOP_IDS.has(normalizeValue(id)))) return true;
  if (splitValues(v.concept_code as string).some(c => normalizeValue(c).includes(PATIENT_CONCEPT_CODE))) return true;
  const text = `${v.var_name || ''} ${v.var_label || ''} ${v.concept_name || ''}`.toLowerCase();
  if (/\b(gender|sex)\b/.test(text)) return true;
  if (/\b(patient|subject|record|study)[\s_-]*id\b/.test(text)) return true;
  if (String(v.var_name || '').trim().toLowerCase() === 'id') return true;
  return false;
}

function isPatientIdVar(v: Variable): boolean {
  if (splitValues(v.omop_id as string).some(id => PATIENT_OMOP_IDS.has(normalizeValue(id)))) return true;
  if (splitValues(v.concept_code as string).some(c => normalizeValue(c).includes(PATIENT_CONCEPT_CODE))) return true;
  const text = `${v.var_name || ''} ${v.var_label || ''} ${v.concept_name || ''}`.toLowerCase();
  if (/\b(patient|subject|record|study)[\s_-]*id\b/.test(text)) return true;
  return String(v.var_name || '').trim().toLowerCase() === 'id';
}

// The cohort's count of patient IDs: the highest non-null count among its
// patient-id variables (null when it has none, or none with a count).
function patientIdCount(cohort: Cohort, counts: EdaCounts | null): number | null {
  let max: number | null = null;
  for (const variable of Object.values(cohort.variables || {}) as Variable[]) {
    if (!isPatientIdVar(variable)) continue;
    const { n } = variableCount(counts, cohort.cohort_id, variable);
    if (n !== null && (max === null || n > max)) max = n;
  }
  return max;
}

// Long format = profiled, but the observation count of patient id / gender
// exceeds the number of participants (several rows per subject).
function classifyCohort(cohort: Cohort, participants: number | null, counts: EdaCounts | null): CohortType {
  if (!cohort.eda_version && !counts?.[cohort.cohort_id]) return 'none';
  if (!participants || !cohort.variables) return 'wide';
  let maxIdCount = 0;
  for (const variable of Object.values(cohort.variables) as Variable[]) {
    if (!isPatientIdOrGenderVar(variable)) continue;
    const { n } = variableCount(counts, cohort.cohort_id, variable);
    if (n !== null && n > maxIdCount) maxIdCount = n;
  }
  return maxIdCount > participants ? 'long' : 'wide';
}

// --- Clustering: inclusive OR over OMOP ID and concept code (union-find) ---

class UnionFind {
  private parent = new Map<string, string>();

  find(x: string): string {
    let root = this.parent.get(x);
    if (root === undefined) {
      this.parent.set(x, x);
      return x;
    }
    if (root !== x) {
      root = this.find(root);
      this.parent.set(x, root);
    }
    return root;
  }

  union(a: string, b: string): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent.set(rb, ra);
  }
}

interface CellEntry {
  varName: string;
  varLabel: string;
  count: number | null;
  source: CountSource | null;
}

interface HeatCluster {
  key: string;
  label: string;
  identifiers: string[];
  cohortIds: Set<string>;
  variableCount: number;
}

interface HeatModel {
  clusters: HeatCluster[];
  visitsByCohort: Record<string, { key: string; label: string }[]>;
  // `${cohortId}||${visitKey}||${clusterKey}` -> entries
  cells: Map<string, CellEntry[]>;
  cohortIds: string[];
}

function variableIdentifiers(variable: Variable): string[] {
  return [
    ...splitValues(variable.omop_id as string).map(v => `omop:${normalizeValue(v)}`),
    ...splitValues(variable.concept_code as string).map(v => `code:${normalizeValue(v)}`),
  ];
}

function buildHeatModel(cohortsData: Record<string, Cohort>, counts: EdaCounts | null): HeatModel {
  const uf = new UnionFind();
  const members: { cohortId: string; variable: Variable; visitKeys: string[]; ids: string[] }[] = [];
  const visitRegistry: Record<string, Map<string, string>> = {};

  for (const [cohortId, cohort] of Object.entries(cohortsData)) {
    if (!cohort.variables) continue;
    const registry = new Map<string, string>();
    visitRegistry[cohortId] = registry;
    for (const variable of Object.values(cohort.variables) as Variable[]) {
      const visits = splitValues(variable.visits);
      const visitKeys = visits.length > 0 ? visits.map(normalizeValue) : [NO_VISIT_KEY];
      visits.forEach(v => {
        const k = normalizeValue(v);
        if (!registry.has(k)) registry.set(k, v);
      });
      if (visits.length === 0 && !registry.has(NO_VISIT_KEY)) registry.set(NO_VISIT_KEY, NO_VISIT_LABEL);

      const ids = variableIdentifiers(variable);
      if (ids.length === 0) continue;
      for (let i = 1; i < ids.length; i++) uf.union(ids[0], ids[i]);
      members.push({ cohortId, variable, visitKeys, ids });
    }
  }

  // Canonical key per component: the smallest identifier in it.
  const keyByRoot = new Map<string, string>();
  const idsByRoot = new Map<string, Set<string>>();
  for (const m of members) {
    const root = uf.find(m.ids[0]);
    if (!idsByRoot.has(root)) idsByRoot.set(root, new Set());
    m.ids.forEach(id => idsByRoot.get(root)!.add(id));
  }
  for (const [root, ids] of idsByRoot.entries()) {
    keyByRoot.set(root, [...ids].sort()[0]);
  }

  const groups = new Map<string, { cohortId: string; variable: Variable; visitKeys: string[] }[]>();
  for (const m of members) {
    const key = keyByRoot.get(uf.find(m.ids[0]))!;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(m);
  }

  const clusters: HeatCluster[] = [];
  const cells = new Map<string, CellEntry[]>();
  const cohortsSeen = new Set<string>();

  for (const [key, group] of groups.entries()) {
    const cohortIds = new Set(group.map(m => m.cohortId));
    cohortIds.forEach(id => cohortsSeen.add(id));

    // Majority concept name among members, as the readable column label
    const nameCounts: Record<string, number> = {};
    for (const m of group) {
      for (const n of splitValues(m.variable.concept_name)) {
        const nn = normalizeValue(n);
        nameCounts[nn] = (nameCounts[nn] || 0) + 1;
      }
    }
    const majorityName = Object.entries(nameCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || '';
    const root = uf.find(variableIdentifiers(group[0].variable)[0]);
    const identifiers = [...(idsByRoot.get(root) || [])].sort();

    clusters.push({
      key,
      label: majorityName ? titleCase(majorityName) : key.replace(/^(omop|code):/, ''),
      identifiers,
      cohortIds,
      variableCount: group.length,
    });

    for (const m of group) {
      const { n, source } = variableCount(counts, m.cohortId, m.variable);
      for (const visitKey of m.visitKeys) {
        const cellKey = `${m.cohortId}||${visitKey}||${key}`;
        if (!cells.has(cellKey)) cells.set(cellKey, []);
        cells.get(cellKey)!.push({
          varName: m.variable.var_name,
          varLabel: m.variable.var_label || m.variable.var_name,
          count: n,
          source,
        });
      }
    }
  }

  clusters.sort(
    (a, b) => b.cohortIds.size - a.cohortIds.size || b.variableCount - a.variableCount || a.label.localeCompare(b.label)
  );

  const visitsByCohort: Record<string, { key: string; label: string }[]> = {};
  for (const [cohortId, registry] of Object.entries(visitRegistry)) {
    visitsByCohort[cohortId] = [...registry.entries()]
      .sort((a, b) => compareVisitKeys(a[0], b[0]))
      .map(([k, label]) => ({ key: k, label }));
  }

  return { clusters, visitsByCohort, cells, cohortIds: [...cohortsSeen] };
}

// The cell's value is the highest count among its variables (EDA counts win
// ties over dictionary ones); `source` is where that winning count came from.
function cellValue(entries: CellEntry[] | undefined): {
  n: number | null;
  source: CountSource | null;
  entries: CellEntry[];
} {
  if (!entries || entries.length === 0) return { n: null, source: null, entries: [] };
  let best: CellEntry | null = null;
  for (const e of entries) {
    if (e.count === null) continue;
    if (
      best === null ||
      e.count > best.count! ||
      (e.count === best.count && e.source === 'eda' && best.source !== 'eda')
    ) {
      best = e;
    }
  }
  return { n: best ? best.count : null, source: best ? best.source : null, entries };
}

function formatPct(pct: number): string {
  if (pct >= 0.995) return `${Math.round(pct * 100)}%`;
  return `${(pct * 100).toFixed(1)}%`;
}

// --- Colour scales ---
// Profiled wide-format cohorts get the red -> amber -> green spectrum: their
// numbers are comparable (one row per subject), so a share of the cohort
// means something. Long-format and non-profiled cohorts get two gentler
// single-hue scales that follow the counts only (never percentages).

type ShadeBy = 'pct' | 'count';
type PctMode = 'none' | 'ids';

const PCT_MODE_LABELS: Record<PctMode, string> = {
  none: 'No percentages',
  ids: '% of patient IDs counted',
};

// t in [0, 1] -> red (0) .. amber .. green (1).
const wideColor = (t: number): string => `hsl(${Math.round(120 * Math.min(1, Math.max(0, t)))}, 70%, 74%)`;
// t in [0, 1] -> very light .. medium, one soft hue.
const softColor = (hue: number, sat: number) => (t: number): string =>
  `hsl(${hue}, ${sat}%, ${Math.round(95 - 25 * Math.min(1, Math.max(0, t)))}%)`;
const longColor = softColor(212, 45); // slate blue
const noneColor = softColor(275, 30); // muted violet

const SCALE_COLOR: Record<CohortType, (t: number) => string> = {
  wide: wideColor,
  long: longColor,
  none: noneColor,
};

// Counts span 1 .. 100k: shade on a log scale so small cohorts stay visible.
const countShade = (n: number, max: number): number => (max > 1 && n > 0 ? Math.log(n) / Math.log(max) : n > 0 ? 1 : 0);

// The percentage of a cell under the chosen basis; never for long-format
// cohorts, whose counts aggregate several rows per subject.
function cellPct(n: number | null, row: HeatRow, mode: PctMode): number | null {
  if (n === null || mode === 'none' || row.type === 'long') return null;
  return row.idCount ? n / row.idCount : null;
}

// --- CSV export ---

interface HeatRow {
  cohortId: string;
  type: CohortType;
  visitKey: string;
  visitLabel: string;
  firstOfCohort: boolean;
  participants: number | null;
  idCount: number | null;
}

function csvField(value: string | number): string {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

// What each exported cell holds; any combination, at least one.
interface ExportContents {
  numbers: boolean;
  percentages: boolean;
  names: boolean;
}

// The export's percentage basis: patient IDs counted for wide format,
// falling back to the declared number of participants when the cohort has no
// patient-id count; declared participants for cohorts without profiling; none
// for long format (several rows per subject). No basis -> "NA".
function exportPctBase(row: HeatRow): number | null {
  if (row.type === 'long') return null;
  if (row.type === 'wide' && row.idCount) return row.idCount;
  return row.participants;
}

function buildCsv(
  rows: HeatRow[],
  clusters: HeatCluster[],
  cells: Map<string, CellEntry[]>,
  contents: ExportContents
): { csv: string; cohortCount: number; conceptCount: number } {
  const lines: string[] = [];
  lines.push(['Cohort', 'Visit', ...clusters.map(c => c.label)].map(csvField).join(','));
  for (const row of rows) {
    const cohortName = `${row.cohortId}${CSV_SUFFIX[row.type]}`;
    const values = clusters.map(cluster => {
      const { n, entries } = cellValue(cells.get(`${row.cohortId}||${row.visitKey}||${cluster.key}`));
      if (entries.length === 0) return '';
      const parts: string[] = [];
      // No EDA and no dictionary count: non-profiled cohorts fall back to the
      // overall participant count.
      const number = n ?? (row.type === 'none' ? row.participants : null);
      if (contents.numbers && number !== null) parts.push(String(number));
      if (contents.percentages) {
        const base = exportPctBase(row);
        const pctText =
          row.type === 'long' ? 'n/a' : n !== null && base ? `${Number(((n / base) * 100).toFixed(1))}%` : 'NA';
        parts.push(parts.length ? `(${pctText})` : pctText);
      }
      if (contents.names) {
        const names = entries.map(e => e.varName).join(', ');
        parts.push(parts.length ? `[${names}]` : names);
      }
      return parts.join(' ');
    });
    lines.push([cohortName, row.visitLabel, ...values].map(csvField).join(','));
  }
  return {
    csv: lines.join('\n'),
    cohortCount: new Set(rows.map(r => r.cohortId)).size,
    conceptCount: clusters.length,
  };
}

function downloadCsv(csv: string, filename: string): void {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

type ExportStep = 'options' | 'warn_mix';

const rowKey = (cohortId: string, visitKey: string): string => `${cohortId}||${visitKey}`;

interface HiddenItem {
  key: string;
  label: string;
}

type CellInfo =
  | { kind: 'empty' }
  | { kind: 'participants'; entries: CellEntry[]; participants: number | null; bg: string; tooltip: string }
  | {
      kind: 'value';
      n: number | null;
      source: CountSource | null;
      entries: CellEntry[];
      pct: number | null;
      bg: string;
      tooltip: string;
    };

type ZoomTarget = { kind: 'row'; row: HeatRow } | { kind: 'column'; cluster: HeatCluster };

// --- Hide animation: a label flies from the hidden row / column into the box ---

interface Ghost {
  id: number;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

function FlyingGhost({ ghost, target, onDone }: { ghost: Ghost; target: HTMLElement | null; onDone: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !target) {
      onDone();
      return;
    }
    const t = target.getBoundingClientRect();
    const frame = requestAnimationFrame(() => {
      el.style.left = `${t.left + 12}px`;
      el.style.top = `${t.top + 8}px`;
      el.style.width = `${Math.min(t.width - 24, 220)}px`;
      el.style.height = '22px';
      el.style.opacity = '0.2';
    });
    const timer = setTimeout(onDone, 650);
    return () => {
      cancelAnimationFrame(frame);
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <div
      ref={ref}
      aria-hidden="true"
      className="fixed z-[60] pointer-events-none overflow-hidden rounded-md border border-base-300 bg-base-100 px-2 text-xs font-semibold shadow-lg flex items-center"
      style={{
        left: ghost.x,
        top: ghost.y,
        width: ghost.w,
        height: Math.max(22, Math.min(ghost.h, 60)),
        opacity: 1,
        transition: 'left 550ms cubic-bezier(.2,.8,.2,1), top 550ms cubic-bezier(.2,.8,.2,1), width 550ms ease, height 550ms ease, opacity 550ms ease',
      }}
    >
      <span className="truncate">{ghost.label}</span>
    </div>
  );
}

// --- Zoom overlay ---
// A zoomed COLUMN lists its cells (one per cohort + visit) top to bottom and
// wraps into the next column when the window is full, like reading a
// newspaper; a zoomed ROW lists its cells (one per concept) left to right and
// wraps onto the next line, like reading text. Tiles are fixed-size so the
// wrapping is regular; each carries its label and the cell's hover text.

const TILE_W = 200;
const TILE_H = 104;
const TILE_GAP = 8;

interface ZoomTile {
  key: string;
  label: string;
  sublabel?: string;
  info: CellInfo;
}

function ZoomOverlay({
  title,
  subtitle,
  direction,
  tiles,
  renderBody,
  onClose,
}: {
  title: string;
  subtitle: string;
  direction: 'column' | 'row';
  tiles: ZoomTile[];
  renderBody: (info: CellInfo) => React.ReactNode;
  onClose: () => void;
}) {
  const [includeEmpty, setIncludeEmpty] = useState(true);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [bodyH, setBodyH] = useState(0);
  const shown = includeEmpty ? tiles : tiles.filter(t => t.info.kind !== 'empty');
  const withData = tiles.filter(t => t.info.kind !== 'empty').length;

  useLayoutEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const measure = () => setBodyH(el.clientHeight);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Tiles per column for the vertical flow (column zoom).
  const perColumn = Math.max(1, Math.floor((bodyH - 24 + TILE_GAP) / (TILE_H + TILE_GAP)));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-6" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="flex h-[88vh] w-[94vw] flex-col rounded-xl bg-base-100 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-base-300 px-5 py-3">
          <div className="min-w-0 flex-1">
            <div className="text-xs uppercase tracking-wide text-base-content/50">
              {direction === 'column' ? 'Column (concept cluster)' : 'Row (cohort + visit)'}
            </div>
            <h3 className="truncate text-lg font-bold" title={title}>
              {title}
            </h3>
            <div className="truncate text-xs text-base-content/60" title={subtitle}>
              {subtitle}
            </div>
          </div>
          <div className="text-sm text-base-content/60">
            {withData} with data · {tiles.length - withData} empty
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="toggle toggle-sm"
              checked={includeEmpty}
              onChange={e => setIncludeEmpty(e.target.checked)}
            />
            Include empty cells
          </label>
          <span className="text-xs text-base-content/50">
            {direction === 'column' ? 'Reads top to bottom, then left to right ↓ →' : 'Reads left to right, then down → ↓'}
          </span>
          <button className="btn btn-sm btn-ghost" onClick={onClose} aria-label="Close zoom">
            <X size={18} />
          </button>
        </div>
        <div ref={bodyRef} className={`min-h-0 flex-1 p-3 ${direction === 'column' ? 'overflow-x-auto overflow-y-hidden' : 'overflow-y-auto'}`}>
          {shown.length === 0 ? (
            <p className="py-10 text-center text-sm text-base-content/50">No cells with data.</p>
          ) : (
            <div
              className={direction === 'column' ? 'grid' : 'flex flex-wrap'}
              style={
                direction === 'column'
                  ? {
                      gridAutoFlow: 'column',
                      gridTemplateRows: `repeat(${perColumn}, ${TILE_H}px)`,
                      gridAutoColumns: `${TILE_W}px`,
                      gap: TILE_GAP,
                    }
                  : { gap: TILE_GAP }
              }
            >
              {shown.map(t => {
                const empty = t.info.kind === 'empty';
                return (
                  <div
                    key={t.key}
                    className={`flex flex-col overflow-hidden rounded-md border px-2 py-1.5 text-xs ${
                      empty ? 'border-dashed border-base-300 text-base-content/40' : 'border-base-300'
                    }`}
                    style={{
                      width: TILE_W,
                      height: TILE_H,
                      backgroundColor: empty ? undefined : t.info.kind === 'empty' ? undefined : t.info.bg,
                    }}
                    title={t.info.kind === 'empty' ? `${t.label}: no variable here` : `${t.label}\n${t.info.tooltip}`}
                  >
                    <div className="truncate font-semibold text-base-content" title={t.label}>
                      {t.label}
                    </div>
                    {t.sublabel && <div className="truncate text-[10px] text-base-content/60">{t.sublabel}</div>}
                    <div className="mt-1 min-h-0 flex-1 overflow-hidden text-center">
                      {empty ? <div className="pt-3">·</div> : renderBody(t.info)}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ConceptCoverageHeatmapPage() {
  const { cohortsData, isLoading, userEmail } = useCohorts();
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [cohortFilter, setCohortFilter] = useState<Record<string, boolean>>({});
  const [clusterFilter, setClusterFilter] = useState<Record<string, boolean>>({});
  const [clusterSearch, setClusterSearch] = useState('');
  const [threshold, setThreshold] = useState(2);
  const [thresholdConfirm, setThresholdConfirm] = useState<{ columns: number; rows: number } | null>(null);
  const [exportStep, setExportStep] = useState<ExportStep | null>(null);
  const [exportContents, setExportContents] = useState<ExportContents>({
    numbers: true,
    percentages: false,
    names: false,
  });
  // Wide-format cells shade by percentage or by count; percentages shown under
  // the counts (off by default), and relative to what.
  const [shadeBy, setShadeBy] = useState<ShadeBy>('pct');
  const [pctMode, setPctMode] = useState<PctMode>('none');
  // Variable names inside the cells (the cells grow to fit them).
  const [showVarNames, setShowVarNames] = useState(false);
  // Percentages need a basis even when none are shown (shading, export).
  const pctBasis: Exclude<PctMode, 'none'> = 'ids';
  const [exportExcludeNonWide, setExportExcludeNonWide] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // null while loading; {} when the request failed (dictionary counts as fallback)
  const [edaCounts, setEdaCounts] = useState<EdaCounts | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(`${apiUrl}/eda-observation-counts`, { credentials: 'include' })
      .then(res => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then(data => {
        if (alive) setEdaCounts(data && typeof data === 'object' ? (data as EdaCounts) : {});
      })
      .catch(err => {
        console.error('eda observation counts:', err);
        if (alive) setEdaCounts({});
      });
    return () => {
      alive = false;
    };
  }, []);

  const showToast = (message: string) => {
    setToast(message);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 3500);
  };

  const model = useMemo(() => {
    if (!cohortsData || Object.keys(cohortsData).length === 0) {
      return { clusters: [], visitsByCohort: {}, cells: new Map(), cohortIds: [] } as HeatModel;
    }
    return buildHeatModel(cohortsData, edaCounts);
  }, [cohortsData, edaCounts]);

  const cohortInfo = useMemo(() => {
    const out: Record<string, { type: CohortType; participants: number | null; idCount: number | null }> = {};
    for (const id of model.cohortIds) {
      const cohort = cohortsData?.[id];
      const participants = parseParticipants(cohort?.study_participants);
      out[id] = {
        type: cohort ? classifyCohort(cohort, participants, edaCounts) : 'none',
        participants,
        idCount: cohort ? patientIdCount(cohort, edaCounts) : null,
      };
    }
    return out;
  }, [model, cohortsData, edaCounts]);

  // Wide format first, then long format, then no profiling; alphabetical within each.
  const allCohortIds = useMemo(
    () =>
      [...model.cohortIds].sort(
        (a, b) => TYPE_ORDER[cohortInfo[a].type] - TYPE_ORDER[cohortInfo[b].type] || a.localeCompare(b)
      ),
    [model, cohortInfo]
  );

  const isCohortSelected = (id: string) => cohortFilter[id] !== false;
  const isClusterSelected = (key: string) => clusterFilter[key] !== false;

  const selectedCohorts = useMemo(() => allCohortIds.filter(isCohortSelected), [allCohortIds, cohortFilter]);

  // Clusters passing the cohort threshold (counted over ALL cohorts)
  const thresholdClusters = useMemo(
    () => model.clusters.filter(c => c.cohortIds.size >= threshold),
    [model, threshold]
  );

  // Rows (cohort + visit) and columns (clusters) hidden with their "x":
  // kept apart from the filters, so the threshold slider and the filter
  // checkboxes never bring them back; only Restore does.
  const [hiddenRows, setHiddenRows] = useState<HiddenItem[]>([]);
  const [hiddenCols, setHiddenCols] = useState<HiddenItem[]>([]);
  const hiddenRowKeys = useMemo(() => new Set(hiddenRows.map(h => h.key)), [hiddenRows]);
  const hiddenColKeys = useMemo(() => new Set(hiddenCols.map(h => h.key)), [hiddenCols]);

  const visibleClusters = useMemo(
    () => thresholdClusters.filter(c => isClusterSelected(c.key) && !hiddenColKeys.has(c.key)),
    [thresholdClusters, clusterFilter, hiddenColKeys]
  );

  const rows: HeatRow[] = useMemo(() => {
    const out: HeatRow[] = [];
    for (const cohortId of selectedCohorts) {
      const visits = model.visitsByCohort[cohortId] || [];
      const { type, participants, idCount } = cohortInfo[cohortId];
      let first = true;
      for (const v of visits) {
        if (hiddenRowKeys.has(rowKey(cohortId, v.key))) continue;
        out.push({ cohortId, type, visitKey: v.key, visitLabel: v.label, firstOfCohort: first, participants, idCount });
        first = false;
      }
    }
    return out;
  }, [selectedCohorts, model, cohortInfo, hiddenRowKeys]);

  // Highest count per cohort type in the matrix shown: each colour scale
  // spans its own range.
  const maxByType = useMemo(() => {
    const max: Record<CohortType, number> = { wide: 0, long: 0, none: 0 };
    for (const row of rows) {
      for (const cluster of visibleClusters) {
        const { n } = cellValue(model.cells.get(`${row.cohortId}||${row.visitKey}||${cluster.key}`));
        if (n !== null && n > max[row.type]) max[row.type] = n;
      }
    }
    return max;
  }, [rows, visibleClusters, model]);

  // One cell (row x cluster): what it shows, its colour and its hover text.
  // Shared by the matrix and the zoom overlay so both read the same.
  const cellInfo = (row: HeatRow, cluster: HeatCluster): CellInfo => {
    const { n, source, entries } = cellValue(model.cells.get(`${row.cohortId}||${row.visitKey}||${cluster.key}`));
    if (entries.length === 0) return { kind: 'empty' };
    if (n === null && row.type === 'none') {
      // Non-profiled cohort AND no dictionary count: the participant count
      // stands in for the cell value.
      return {
        kind: 'participants',
        entries,
        participants: row.participants,
        bg: 'rgba(148, 163, 184, 0.25)',
        tooltip: `No profiling and no dictionary count — overall participant count shown\n${entries
          .map(e => `${e.varName} — ${e.varLabel}`)
          .join('\n')}`,
      };
    }
    const pct = cellPct(n, row, pctMode);
    // Wide format: by percentage (or count, per the toggle); long format and
    // no profiling: by count, on their own scales.
    const sharePct = row.type === 'wide' && shadeBy === 'pct' ? cellPct(n, row, pctBasis) : null;
    const shade = n === null ? 0 : sharePct !== null ? Math.min(sharePct, 1) : countShade(n, maxByType[row.type]);
    const sourceNote = source === 'dict' ? 'count from the metadata dictionary, not EDA profiling\n' : '';
    return {
      kind: 'value',
      n,
      source,
      entries,
      pct,
      bg: n === null ? 'transparent' : SCALE_COLOR[row.type](shade),
      tooltip:
        sourceNote +
        entries
          .map(
            e =>
              `${e.varName}: ${e.count === null ? 'count n/a' : e.count}${e.source === 'dict' ? ' (dict)' : ''} — ${
                e.varLabel
              }`
          )
          .join('\n'),
    };
  };

  const cellBody = (info: CellInfo, withNames: boolean) => {
    if (info.kind === 'empty') return null;
    const names = withNames && (
      <div className="mt-0.5 text-[10px] leading-tight font-mono text-base-content/80 break-all">
        {info.entries.map(e => e.varName).join(', ')}
      </div>
    );
    if (info.kind === 'participants') {
      return (
        <>
          <div className="font-mono font-semibold">{info.participants ?? '?'}</div>
          <div className="text-[10px] text-base-content/60">participants</div>
          {names}
        </>
      );
    }
    const { n, source, entries, pct } = info;
    return (
      <>
        <div className="font-mono font-semibold">
          {n === null ? '?' : n}
          {source === 'dict' && <span className="font-sans font-normal text-base-content/50">*</span>}
        </div>
        {(pct !== null || source === 'dict' || entries.length > 1) && (
          <div className="text-[10px] text-base-content/70">
            {[pct !== null ? formatPct(pct) : '', source === 'dict' ? 'dict' : '', entries.length > 1 ? `${entries.length} vars` : '']
              .filter(Boolean)
              .join(' · ')}
          </div>
        )}
        {names}
      </>
    );
  };

  // --- Hide / restore ---
  // A hidden row or column flies into the "Hidden items" box (top right).
  const hiddenBoxRef = useRef<HTMLDivElement>(null);
  const [ghosts, setGhosts] = useState<Ghost[]>([]);
  const ghostId = useRef(0);
  const [hiddenBoxOpen, setHiddenBoxOpen] = useState(true);

  const flyToHiddenBox = (label: string, from: Element | null) => {
    if (!from || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const r = from.getBoundingClientRect();
    ghostId.current += 1;
    setGhosts(g => [...g, { id: ghostId.current, label, x: r.left, y: r.top, w: r.width, h: r.height }]);
  };

  const hideRow = (row: HeatRow, from: Element | null) => {
    const label = `${row.cohortId} · ${row.visitLabel}`;
    flyToHiddenBox(label, from);
    setHiddenBoxOpen(true);
    setHiddenRows(prev => [...prev, { key: rowKey(row.cohortId, row.visitKey), label }]);
  };

  const hideColumn = (cluster: HeatCluster, from: Element | null) => {
    flyToHiddenBox(cluster.label, from);
    setHiddenBoxOpen(true);
    setHiddenCols(prev => [...prev, { key: cluster.key, label: cluster.label }]);
  };

  // --- Zoom ---
  const [zoom, setZoom] = useState<ZoomTarget | null>(null);

  const clusterSearchLower = clusterSearch.trim().toLowerCase();
  const clusterListForFilter = useMemo(
    () =>
      clusterSearchLower
        ? thresholdClusters.filter(
            c => c.label.toLowerCase().includes(clusterSearchLower) || c.identifiers.some(id => id.includes(clusterSearchLower))
          )
        : thresholdClusters,
    [thresholdClusters, clusterSearchLower]
  );

  const setAllCohorts = (value: boolean) => {
    const next: Record<string, boolean> = {};
    for (const id of allCohortIds) next[id] = value;
    setCohortFilter(next);
  };

  const setAllClusters = (value: boolean) => {
    setClusterFilter(prev => {
      const next = { ...prev };
      for (const c of clusterListForFilter) next[c.key] = value;
      return next;
    });
  };

  // Moving the slider re-sets every other filter, so the estimate and the
  // resulting matrix are always over the full cohort/cluster sets.
  const applyThreshold = (value: number) => {
    setThreshold(value);
    setCohortFilter({});
    setClusterFilter({});
    setClusterSearch('');
    showToast(`Threshold set to ${value}: cohort and variable-cluster filters were reset.`);
  };

  const onThresholdChange = (value: number) => {
    if (value === threshold) return;
    if (value === 1) {
      // Warn before including single-cohort concepts: the matrix explodes.
      const totalRows = allCohortIds.reduce((sum, id) => sum + (model.visitsByCohort[id] || []).length, 0);
      setThresholdConfirm({ columns: model.clusters.length, rows: totalRows });
      return;
    }
    applyThreshold(value);
  };

  // --- Export flow ---

  const exportRows = useMemo(
    () => (exportExcludeNonWide ? rows.filter(r => r.type === 'wide') : rows),
    [rows, exportExcludeNonWide]
  );

  const exportTypes = useMemo(() => new Set(exportRows.map(r => r.type)), [exportRows]);
  const exportHasMixedTypes = exportTypes.size > 1 && (exportTypes.has('long') || exportTypes.has('none'));

  const openExportDialog = () => {
    setExportContents({ numbers: true, percentages: false, names: false });
    setExportExcludeNonWide(false);
    setExportStep('options');
  };

  const runExport = (excludeNonWide: boolean) => {
    const finalRows = excludeNonWide ? rows.filter(r => r.type === 'wide') : rows;
    const { csv, cohortCount, conceptCount } = buildCsv(finalRows, visibleClusters, model.cells, exportContents);
    downloadCsv(csv, `icare4cvd-concept-coverage-${cohortCount}cohorts-${conceptCount}concepts.csv`);
    setExportStep(null);
    setExportExcludeNonWide(false);
  };

  const advanceExport = (fromStep: ExportStep, excludeNonWide: boolean) => {
    setExportExcludeNonWide(excludeNonWide);
    if (fromStep === 'options' && !excludeNonWide && exportHasMixedTypes) {
      setExportStep('warn_mix');
      return;
    }
    runExport(excludeNonWide);
  };

  const exportButton = (
    <button className="btn btn-sm btn-outline gap-1" onClick={openExportDialog} disabled={rows.length === 0 || visibleClusters.length === 0}>
      <Download size={14} />
      Export to CSV
    </button>
  );

  if (userEmail === null && apiUrl !== 'mock') {
    return (
      <main className="flex flex-col items-center justify-center p-4">
        <LoginPrompt />
      </main>
    );
  }

  return (
    <div className="min-h-screen bg-base-100">
      <div className="px-4 py-6">
        <div className="mb-4">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Grid size={26} />
            Concept Coverage Heatmap
          </h1>
          <p className="text-base-content/60 mt-1 text-sm">
            Rows are cohort + visit; columns are variable clusters (variables sharing an OMOP ID or a concept code,
            inclusive). Each cell shows the number of non-null observations (from the EDA variable profiling, falling
            back to the dictionary COUNT column when a cohort has no EDA output), optionally with a percentage.
          </p>
        </div>

        {isLoading ? (
          <div className="flex justify-center items-center py-20">
            <span className="loading loading-spinner loading-lg"></span>
          </div>
        ) : (
          <>
            <div className="flex flex-wrap gap-3 mb-4 items-center">
              <button onClick={() => setFiltersOpen(o => !o)} className="btn btn-sm btn-outline gap-1">
                {filtersOpen ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
                {filtersOpen ? 'Hide filters' : 'Show filters'}
              </button>
              <div className="divider divider-horizontal mx-0"></div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-base-content/60 whitespace-nowrap">
                  Min cohorts per concept: <span className="font-semibold text-base-content">{threshold}</span>
                </span>
                <input
                  type="range"
                  min={1}
                  max={10}
                  step={1}
                  value={threshold}
                  onChange={e => onThresholdChange(Number(e.target.value))}
                  className="range range-sm range-primary w-48"
                />
              </div>
              <div className="divider divider-horizontal mx-0"></div>
              <div className="flex items-center gap-2" title="What the red-to-green shading of profiled wide-format cohorts follows">
                <span className="text-sm text-base-content/60 whitespace-nowrap">Shade wide-format cells by</span>
                <div className="join">
                  {(['pct', 'count'] as ShadeBy[]).map(v => (
                    <button
                      key={v}
                      className={`btn btn-xs join-item ${shadeBy === v ? 'btn-active' : ''}`}
                      onClick={() => setShadeBy(v)}
                      aria-pressed={shadeBy === v}
                      title={
                        v === 'pct'
                          ? 'Share of the cohort (of its patient IDs counted)'
                          : 'Number of non-null observations (log scale)'
                      }
                    >
                      {v === 'pct' ? 'percentage' : 'count'}
                    </button>
                  ))}
                </div>
              </div>
              <div className="divider divider-horizontal mx-0"></div>
              <div className="flex items-center gap-3" role="radiogroup" aria-label="Percentages shown under the counts">
                {(Object.keys(PCT_MODE_LABELS) as PctMode[]).map(m => (
                  <label key={m} className="flex items-center gap-1.5 cursor-pointer text-sm">
                    <input
                      type="radio"
                      name="pct-mode"
                      className="radio radio-xs"
                      checked={pctMode === m}
                      onChange={() => setPctMode(m)}
                    />
                    <span className={pctMode === m ? 'text-base-content' : 'text-base-content/60'}>{PCT_MODE_LABELS[m]}</span>
                  </label>
                ))}
              </div>
              <div className="divider divider-horizontal mx-0"></div>
              <label className="flex items-center gap-2 cursor-pointer text-sm" title="Show the names of the variables behind each cell (cells get wider)">
                <input
                  type="checkbox"
                  className="toggle toggle-sm"
                  checked={showVarNames}
                  onChange={e => setShowVarNames(e.target.checked)}
                />
                <span className={showVarNames ? 'text-base-content' : 'text-base-content/60'}>Variable names in cells</span>
              </label>
              <div className="divider divider-horizontal mx-0"></div>
              {exportButton}
              <span className="text-sm text-base-content/50">
                {visibleClusters.length} columns (concept clusters) · {rows.length} rows
              </span>
            </div>

            <div className="flex gap-4 items-start">
              {filtersOpen && (
                <aside className="w-72 shrink-0 space-y-4">
                  <div className="collapse collapse-arrow bg-base-200 border border-base-300">
                    <input type="checkbox" defaultChecked />
                    <div className="collapse-title font-semibold text-sm">
                      Cohorts ({selectedCohorts.length}/{allCohortIds.length})
                    </div>
                    <div className="collapse-content">
                      <div className="flex flex-wrap gap-1 mb-2 text-[10px]">
                        {(Object.keys(TYPE_LABELS) as CohortType[]).map(t => (
                          <span key={t} className={`px-2 py-0.5 rounded-full border ${TYPE_CHIP[t]}`}>
                            {TYPE_LABELS[t]}
                          </span>
                        ))}
                      </div>
                      <div className="flex gap-2 mb-2">
                        <button className="btn btn-xs btn-outline" onClick={() => setAllCohorts(true)}>
                          All
                        </button>
                        <button className="btn btn-xs btn-outline" onClick={() => setAllCohorts(false)}>
                          None
                        </button>
                      </div>
                      <div className="max-h-64 overflow-y-auto space-y-1">
                        {allCohortIds.map(id => (
                          <label key={id} className="flex items-center gap-2 text-sm cursor-pointer">
                            <input
                              type="checkbox"
                              className="checkbox checkbox-xs"
                              checked={isCohortSelected(id)}
                              onChange={e => setCohortFilter(prev => ({ ...prev, [id]: e.target.checked }))}
                            />
                            <span
                              className={`truncate px-2 py-0.5 rounded-full border text-xs ${TYPE_CHIP[cohortInfo[id].type]}`}
                              title={`${id} — ${TYPE_LABELS[cohortInfo[id].type]}`}
                            >
                              {id}
                            </span>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="collapse collapse-arrow bg-base-200 border border-base-300">
                    <input type="checkbox" defaultChecked />
                    <div className="collapse-title font-semibold text-sm">
                      Variable clusters ({thresholdClusters.length})
                    </div>
                    <div className="collapse-content">
                      <label className="input input-sm input-bordered flex items-center gap-2 mb-2">
                        <Search size={14} className="text-base-content/40" />
                        <input
                          type="text"
                          className="grow"
                          placeholder="Search clusters"
                          value={clusterSearch}
                          onChange={e => setClusterSearch(e.target.value)}
                        />
                      </label>
                      <div className="flex gap-2 mb-2 items-center">
                        <button className="btn btn-xs btn-outline" onClick={() => setAllClusters(true)}>
                          All
                        </button>
                        <button className="btn btn-xs btn-outline" onClick={() => setAllClusters(false)}>
                          None
                        </button>
                        {clusterSearchLower && (
                          <span className="text-xs text-base-content/50">(applies to matches)</span>
                        )}
                      </div>
                      <div className="max-h-96 overflow-y-auto space-y-1">
                        {clusterListForFilter.map(c => (
                          <label key={c.key} className="flex items-start gap-2 text-sm cursor-pointer">
                            <input
                              type="checkbox"
                              className="checkbox checkbox-xs mt-0.5"
                              checked={isClusterSelected(c.key)}
                              onChange={e => setClusterFilter(prev => ({ ...prev, [c.key]: e.target.checked }))}
                            />
                            <span className="min-w-0">
                              <span className="block whitespace-normal break-words" title={c.label}>
                                {c.label}
                              </span>
                              <span className="block whitespace-normal break-all text-xs text-base-content/50" title={c.identifiers.join(', ')}>
                                {c.identifiers.join(', ')}
                              </span>
                              <span className="block text-xs text-base-content/40">
                                {c.cohortIds.size} cohorts · {c.variableCount} vars
                              </span>
                            </span>
                          </label>
                        ))}
                        {clusterListForFilter.length === 0 && (
                          <p className="text-xs text-base-content/40">No clusters match the search.</p>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="bg-base-200 border border-base-300 rounded-lg p-3">{exportButton}</div>
                </aside>
              )}

              <main className="flex-1 min-w-0">
                {rows.length === 0 || visibleClusters.length === 0 ? (
                  <div className="text-center py-20 text-base-content/40">
                    Nothing to show. Select at least one cohort and one cluster, or lower the cohort threshold.
                  </div>
                ) : (
                  <div className="overflow-auto border border-base-300 rounded-lg" style={{ maxHeight: '97vh' }}>
                    <table className="border-separate border-spacing-0 text-xs">
                      <thead>
                        <tr>
                          <th className="sticky top-0 left-0 z-30 bg-base-200 border-b border-r border-base-300 px-2 py-1 text-left min-w-[10rem]">
                            Cohort
                          </th>
                          <th className="sticky top-0 z-20 bg-base-200 border-b border-r border-base-300 px-2 py-1 text-left min-w-[8rem]" style={{ left: '10rem' }}>
                            Visit
                          </th>
                          {visibleClusters.map(c => (
                            <th
                              key={c.key}
                              className="group/col sticky top-0 z-10 bg-base-200 border-b border-r border-base-300 px-1 align-bottom"
                              title={`${c.label} — ${c.identifiers.join(', ')}`}
                            >
                              <div className="mb-1 flex justify-center gap-0.5 opacity-40 transition-opacity group-hover/col:opacity-100">
                                <button
                                  type="button"
                                  className="rounded p-0.5 hover:bg-base-300"
                                  title={`Zoom in on this column: every cohort + visit for ${c.label}`}
                                  aria-label={`Zoom in on column ${c.label}`}
                                  onClick={() => setZoom({ kind: 'column', cluster: c })}
                                >
                                  <ZoomIn size={13} />
                                </button>
                                <button
                                  type="button"
                                  className="rounded p-0.5 hover:bg-base-300"
                                  title="Hide this column (it goes to Hidden items, top right)"
                                  aria-label={`Hide column ${c.label}`}
                                  onClick={e => hideColumn(c, e.currentTarget.closest('th'))}
                                >
                                  <X size={13} />
                                </button>
                              </div>
                              <div
                                className="mx-auto overflow-hidden text-ellipsis whitespace-nowrap font-medium"
                                style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', maxHeight: '9rem', minHeight: '9rem' }}
                              >
                                {c.label}
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map(row => (
                          <tr key={`${row.cohortId}||${row.visitKey}`} className="group/row">
                            <td
                              className={`sticky left-0 z-20 bg-base-100 border-r border-base-300 px-2 py-1 font-semibold min-w-[10rem] max-w-[10rem] ${
                                row.firstOfCohort ? 'border-t' : ''
                              }`}
                              title={`${row.cohortId} — ${TYPE_LABELS[row.type]}`}
                            >
                              {row.firstOfCohort && (
                                <span className={`inline-block max-w-full truncate px-2 py-0.5 rounded-full border ${TYPE_CHIP[row.type]}`}>
                                  {row.cohortId}
                                </span>
                              )}
                            </td>
                            <td
                              className={`sticky z-10 bg-base-100 border-r border-base-300 px-2 py-1 min-w-[8rem] max-w-[8rem] truncate ${
                                row.firstOfCohort ? 'border-t' : ''
                              }`}
                              style={{ left: '10rem' }}
                              title={row.visitLabel}
                            >
                              <div className="flex items-center gap-1">
                                <span className="min-w-0 flex-1 truncate">{row.visitLabel}</span>
                                <span className="flex flex-shrink-0 opacity-30 transition-opacity group-hover/row:opacity-100">
                                  <button
                                    type="button"
                                    className="rounded p-0.5 hover:bg-base-300"
                                    title={`Zoom in on this row: every concept for ${row.cohortId} · ${row.visitLabel}`}
                                    aria-label={`Zoom in on row ${row.cohortId} ${row.visitLabel}`}
                                    onClick={() => setZoom({ kind: 'row', row })}
                                  >
                                    <ZoomIn size={13} />
                                  </button>
                                  <button
                                    type="button"
                                    className="rounded p-0.5 hover:bg-base-300"
                                    title="Hide this row (it goes to Hidden items, top right)"
                                    aria-label={`Hide row ${row.cohortId} ${row.visitLabel}`}
                                    onClick={e => hideRow(row, e.currentTarget.closest('tr'))}
                                  >
                                    <X size={13} />
                                  </button>
                                </span>
                              </div>
                            </td>
                            {visibleClusters.map(cluster => {
                              const info = cellInfo(row, cluster);
                              const border = row.firstOfCohort ? 'border-t border-t-base-300' : '';
                              if (info.kind === 'empty') {
                                return (
                                  <td
                                    key={cluster.key}
                                    className={`border-r border-base-200 px-1 py-1 text-center text-base-content/20 ${border}`}
                                  >
                                    ·
                                  </td>
                                );
                              }
                              return (
                                <td
                                  key={cluster.key}
                                  className={`border-r border-base-200 px-1 py-1 text-center ${
                                    showVarNames ? 'min-w-[7rem] max-w-[11rem] align-top' : 'whitespace-nowrap'
                                  } ${border}`}
                                  style={{ backgroundColor: info.bg }}
                                  title={info.tooltip}
                                >
                                  {cellBody(info, showVarNames)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <p className="mt-2 text-xs text-base-content/40">
                  Colours: profiled wide-format cohorts run red (low) to green (high), by percentage or count as chosen
                  above; long-format cohorts (blue) and cohorts without profiling (violet) follow the counts only, on a
                  log scale of their own. Percentages are never shown for long-format cohorts. When several variables of a cohort fall in the same cluster and visit, the highest count is shown (hover
                  for all). Counts marked with * (&quot;dict&quot;) come from the metadata dictionary&apos;s COUNT column
                  instead of the EDA profiling. Gray cells have neither: the overall participant count is shown.
                  Long-format cohorts have no fixed visit columns, so their counts aggregate over all visits.
                </p>
              </main>
            </div>
          </>
        )}
      </div>

      {/* Hidden items: only while something is hidden */}
      {(hiddenRows.length > 0 || hiddenCols.length > 0) && (
        <div
          ref={hiddenBoxRef}
          className="fixed right-4 top-20 z-40 w-72 rounded-lg border border-base-300 bg-base-100 shadow-lg"
          aria-label="Hidden items"
        >
          <div className="flex items-center gap-2 border-b border-base-300 px-3 py-2">
            <EyeOff size={14} className="text-base-content/60" />
            <span className="flex-1 text-sm font-semibold">
              Hidden items ({hiddenRows.length + hiddenCols.length})
            </span>
            <button
              className="btn btn-xs btn-ghost font-normal"
              onClick={() => {
                setHiddenRows([]);
                setHiddenCols([]);
              }}
              title="Bring every hidden row and column back into the matrix"
            >
              Restore all
            </button>
            <button
              className="btn btn-xs btn-ghost px-1"
              onClick={() => setHiddenBoxOpen(o => !o)}
              aria-label={hiddenBoxOpen ? 'Collapse hidden items' : 'Expand hidden items'}
              aria-expanded={hiddenBoxOpen}
            >
              {hiddenBoxOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
          </div>
          {hiddenBoxOpen && (
            <div className="max-h-72 space-y-3 overflow-y-auto px-3 py-2 text-sm">
              {(
                [
                  ['Rows (cohort · visit)', hiddenRows, setHiddenRows],
                  ['Columns (concept clusters)', hiddenCols, setHiddenCols],
                ] as [string, HiddenItem[], React.Dispatch<React.SetStateAction<HiddenItem[]>>][]
              )
                .filter(([, items]) => items.length > 0)
                .map(([heading, items, setItems]) => (
                  <div key={heading}>
                    <div className="mb-1 text-[11px] uppercase tracking-wide text-base-content/50">{heading}</div>
                    <ul className="space-y-1">
                      {items.map(item => (
                        <li key={item.key} className="flex items-start gap-2">
                          <span className="min-w-0 flex-1 break-words text-xs" title={item.label}>
                            {item.label}
                          </span>
                          <button
                            className="btn btn-xs btn-outline flex-shrink-0 font-normal"
                            onClick={() => setItems(prev => prev.filter(h => h.key !== item.key))}
                          >
                            Restore
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
      {ghosts.map(g => (
        <FlyingGhost
          key={g.id}
          ghost={g}
          target={hiddenBoxRef.current}
          onDone={() => setGhosts(prev => prev.filter(x => x.id !== g.id))}
        />
      ))}

      {/* Zoom on one row or one column */}
      {zoom?.kind === 'column' && (
        <ZoomOverlay
          direction="column"
          title={zoom.cluster.label}
          subtitle={`${zoom.cluster.identifiers.join(', ')} · ${zoom.cluster.cohortIds.size} cohorts · ${zoom.cluster.variableCount} variables`}
          tiles={rows.map(row => ({
            key: rowKey(row.cohortId, row.visitKey),
            label: row.cohortId,
            sublabel: `${row.visitLabel} · ${TYPE_LABELS[row.type]}`,
            info: cellInfo(row, zoom.cluster),
          }))}
          renderBody={info => cellBody(info, true)}
          onClose={() => setZoom(null)}
        />
      )}
      {zoom?.kind === 'row' && (
        <ZoomOverlay
          direction="row"
          title={`${zoom.row.cohortId} · ${zoom.row.visitLabel}`}
          subtitle={`${TYPE_LABELS[zoom.row.type]} · ${visibleClusters.length} concept clusters`}
          tiles={visibleClusters.map(cluster => ({
            key: cluster.key,
            label: cluster.label,
            info: cellInfo(zoom.row, cluster),
          }))}
          renderBody={info => cellBody(info, true)}
          onClose={() => setZoom(null)}
        />
      )}

      {/* Transient notice (e.g. filters reset by the threshold slider) */}
      {toast && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50">
          <div className="alert alert-info shadow-lg py-2 px-4 text-sm">{toast}</div>
        </div>
      )}

      {/* Threshold = 1 confirmation */}
      {thresholdConfirm && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg">Include single-cohort concepts?</h3>
            <p className="py-3 text-sm">
              Lowering the threshold to 1 includes every concept that appears in only one cohort. This will increase the
              matrix size to{' '}
              <span className="font-semibold">{thresholdConfirm.columns.toLocaleString()} columns</span> (concept
              clusters, over {thresholdConfirm.rows.toLocaleString()} rows), which can make the page slow. All other
              filters will be reset.
            </p>
            <div className="modal-action">
              <button className="btn btn-sm" onClick={() => setThresholdConfirm(null)}>
                Cancel
              </button>
              <button
                className="btn btn-sm btn-primary"
                onClick={() => {
                  applyThreshold(1);
                  setThresholdConfirm(null);
                }}
              >
                Proceed
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export dialog */}
      {exportStep === 'options' && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg">Export to CSV</h3>
            <p className="py-2 text-sm text-base-content/60">
              Exports the matrix as currently shown ({rows.length} rows × {visibleClusters.length} concept clusters).
            </p>
            <p className="text-sm font-semibold">Include in each cell:</p>
            <div className="space-y-3 py-2">
              {(
                [
                  ['numbers', 'Raw numbers', 'Non-null observation counts'],
                  [
                    'percentages',
                    'Percentages',
                    'Wide format: of patient IDs counted (else of declared participants) · long format: n/a · no profiling: of declared participants · NA when neither is known',
                  ],
                  ['names', 'Variable names', "The cohort's own variable name(s) behind the cell"],
                ] as [keyof ExportContents, string, string][]
              ).map(([key, label, detail]) => (
                <label key={key} className="flex items-start gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    className="checkbox checkbox-sm mt-0.5"
                    checked={exportContents[key]}
                    onChange={e => setExportContents(prev => ({ ...prev, [key]: e.target.checked }))}
                  />
                  <span className="text-sm">
                    {label}
                    <span className="block text-xs text-base-content/50">{detail}</span>
                  </span>
                </label>
              ))}
              <p className="text-xs text-base-content/50">
                Several parts share one cell, e.g. &quot;437 (70.3%) [Hb6, Hb12]&quot;.
              </p>
            </div>
            <div className="modal-action">
              <button className="btn btn-sm" onClick={() => setExportStep(null)}>
                Cancel
              </button>
              <button
                className="btn btn-sm btn-primary"
                onClick={() => advanceExport('options', false)}
                disabled={!exportContents.numbers && !exportContents.percentages && !exportContents.names}
                title={
                  !exportContents.numbers && !exportContents.percentages && !exportContents.names
                    ? 'Tick at least one thing to include'
                    : undefined
                }
              >
                Export
              </button>
            </div>
          </div>
        </div>
      )}

      {exportStep === 'warn_mix' && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg text-warning">Mixing different types of numbers</h3>
            <div className="py-3 text-sm space-y-2">
              <p>
                The matrix you are exporting mixes cohorts of different types, so the numbers are{' '}
                <span className="font-semibold">not directly comparable</span>:
              </p>
              <ul className="list-disc list-inside space-y-1">
                {exportTypes.has('long') && (
                  <li>
                    <span className="font-semibold">Long-format cohorts</span> have no fixed visit types, so their
                    numbers are <span className="font-semibold">aggregated over many visits</span> (one subject can be
                    counted several times).
                  </li>
                )}
                {exportTypes.has('none') && (
                  <li>
                    <span className="font-semibold">Cohorts without variable profiling</span> have no verified
                    observation counts yet: their numbers come from the{' '}
                    <span className="font-semibold">metadata dictionary&apos;s COUNT column</span> as declared by the
                    data owners (or the overall participant count where even that is missing).
                  </li>
                )}
              </ul>
              <p>These are different kinds of numbers sitting side by side in the same file.</p>
            </div>
            <div className="modal-action flex-wrap">
              <button className="btn btn-sm" onClick={() => setExportStep(null)}>
                Cancel
              </button>
              <button className="btn btn-sm btn-outline" onClick={() => advanceExport('warn_mix', true)}>
                Exclude long-format &amp; no-profiling cohorts
              </button>
              <button className="btn btn-sm btn-warning" onClick={() => advanceExport('warn_mix', false)}>
                Proceed with all
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
