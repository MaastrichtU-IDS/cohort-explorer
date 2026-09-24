'use client';

import React, { useMemo, useRef, useState } from 'react';
import { useCohorts } from '@/components/CohortsContext';
import { Cohort, Variable } from '@/types';
import { Grid, ChevronLeft, ChevronRight, Search, Download } from 'react-feather';

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

function parseParticipants(raw: string | null | undefined): number | null {
  if (!raw) return null;
  const match = String(raw).replace(/[,\s]/g, '').match(/(\d+)/);
  if (!match) return null;
  const n = parseInt(match[1], 10);
  return n > 0 ? n : null;
}

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
  none: ' (no variable profiling yet - number of participants shown)',
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

// Long format = profiled, but the observation count of patient id / gender
// exceeds the number of participants (several rows per subject).
function classifyCohort(cohort: Cohort, participants: number | null): CohortType {
  if (!cohort.eda_version) return 'none';
  if (!participants || !cohort.variables) return 'wide';
  let maxIdCount = 0;
  for (const variable of Object.values(cohort.variables) as Variable[]) {
    if (!isPatientIdOrGenderVar(variable)) continue;
    const count = Number(variable.count);
    if (Number.isFinite(count) && count > maxIdCount) maxIdCount = count;
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

function buildHeatModel(cohortsData: Record<string, Cohort>): HeatModel {
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
      label: majorityName || key.replace(/^(omop|code):/, ''),
      identifiers,
      cohortIds,
      variableCount: group.length,
    });

    for (const m of group) {
      const rawCount = m.variable.count;
      const count = rawCount === null || rawCount === undefined || Number.isNaN(Number(rawCount)) ? null : Number(rawCount);
      for (const visitKey of m.visitKeys) {
        const cellKey = `${m.cohortId}||${visitKey}||${key}`;
        if (!cells.has(cellKey)) cells.set(cellKey, []);
        cells.get(cellKey)!.push({
          varName: m.variable.var_name,
          varLabel: m.variable.var_label || m.variable.var_name,
          count,
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

function cellValue(entries: CellEntry[] | undefined): { n: number | null; entries: CellEntry[] } {
  if (!entries || entries.length === 0) return { n: null, entries: [] };
  const counts = entries.map(e => e.count).filter((c): c is number => c !== null);
  return { n: counts.length > 0 ? Math.max(...counts) : null, entries };
}

function formatPct(pct: number): string {
  if (pct >= 0.995) return `${Math.round(pct * 100)}%`;
  return `${(pct * 100).toFixed(1)}%`;
}

// --- CSV export ---

interface HeatRow {
  cohortId: string;
  type: CohortType;
  visitKey: string;
  visitLabel: string;
  firstOfCohort: boolean;
  participants: number | null;
}

function csvField(value: string | number): string {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function buildCsv(
  rows: HeatRow[],
  clusters: HeatCluster[],
  cells: Map<string, CellEntry[]>,
  numberFormat: 'raw' | 'pct'
): { csv: string; cohortCount: number; conceptCount: number } {
  const lines: string[] = [];
  lines.push(['Cohort', 'Visit', ...clusters.map(c => c.label)].map(csvField).join(','));
  for (const row of rows) {
    const cohortName = `${row.cohortId}${CSV_SUFFIX[row.type]}`;
    const values = clusters.map(cluster => {
      const { n, entries } = cellValue(cells.get(`${row.cohortId}||${row.visitKey}||${cluster.key}`));
      if (entries.length === 0) return '';
      if (row.type === 'none') {
        return numberFormat === 'raw' ? (row.participants ?? '') : '';
      }
      if (n === null) return '';
      if (numberFormat === 'raw') return n;
      return row.participants ? Number(((n / row.participants) * 100).toFixed(1)) : '';
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

type ExportStep = 'options' | 'warn_mix' | 'warn_pct';

export default function ConceptCoverageHeatmapPage() {
  const { cohortsData, isLoading } = useCohorts();
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [cohortFilter, setCohortFilter] = useState<Record<string, boolean>>({});
  const [clusterFilter, setClusterFilter] = useState<Record<string, boolean>>({});
  const [clusterSearch, setClusterSearch] = useState('');
  const [threshold, setThreshold] = useState(2);
  const [thresholdConfirm, setThresholdConfirm] = useState<{ cellCount: number } | null>(null);
  const [exportStep, setExportStep] = useState<ExportStep | null>(null);
  const [exportFormat, setExportFormat] = useState<'raw' | 'pct'>('raw');
  const [exportExcludeNonWide, setExportExcludeNonWide] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showToast = (message: string) => {
    setToast(message);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 3500);
  };

  const model = useMemo(() => {
    if (!cohortsData || Object.keys(cohortsData).length === 0) {
      return { clusters: [], visitsByCohort: {}, cells: new Map(), cohortIds: [] } as HeatModel;
    }
    return buildHeatModel(cohortsData);
  }, [cohortsData]);

  const cohortInfo = useMemo(() => {
    const out: Record<string, { type: CohortType; participants: number | null }> = {};
    for (const id of model.cohortIds) {
      const cohort = cohortsData?.[id];
      const participants = parseParticipants(cohort?.study_participants);
      out[id] = { type: cohort ? classifyCohort(cohort, participants) : 'none', participants };
    }
    return out;
  }, [model, cohortsData]);

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

  const visibleClusters = useMemo(
    () => thresholdClusters.filter(c => isClusterSelected(c.key)),
    [thresholdClusters, clusterFilter]
  );

  const rows: HeatRow[] = useMemo(() => {
    const out: HeatRow[] = [];
    for (const cohortId of selectedCohorts) {
      const visits = model.visitsByCohort[cohortId] || [];
      const { type, participants } = cohortInfo[cohortId];
      visits.forEach((v, i) => {
        out.push({ cohortId, type, visitKey: v.key, visitLabel: v.label, firstOfCohort: i === 0, participants });
      });
    }
    return out;
  }, [selectedCohorts, model, cohortInfo]);

  const maxN = useMemo(() => {
    let max = 0;
    for (const row of rows) {
      if (row.type === 'none') continue;
      for (const cluster of visibleClusters) {
        const { n } = cellValue(model.cells.get(`${row.cohortId}||${row.visitKey}||${cluster.key}`));
        if (n !== null && n > max) max = n;
      }
    }
    return max;
  }, [rows, visibleClusters, model]);

  const clusterSearchLower = clusterSearch.trim().toLowerCase();
  const clusterListForFilter = useMemo(
    () =>
      clusterSearchLower
        ? thresholdClusters.filter(
            c => c.label.includes(clusterSearchLower) || c.identifiers.some(id => id.includes(clusterSearchLower))
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
      setThresholdConfirm({ cellCount: totalRows * model.clusters.length });
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
  const exportMissingTotals = useMemo(
    () => [...new Set(exportRows.filter(r => r.participants === null).map(r => r.cohortId))],
    [exportRows]
  );

  const openExportDialog = () => {
    setExportFormat('raw');
    setExportExcludeNonWide(false);
    setExportStep('options');
  };

  const runExport = (excludeNonWide: boolean) => {
    const finalRows = excludeNonWide ? rows.filter(r => r.type === 'wide') : rows;
    const { csv, cohortCount, conceptCount } = buildCsv(finalRows, visibleClusters, model.cells, exportFormat);
    downloadCsv(csv, `icare4cvd-concept-coverage-${cohortCount}cohorts-${conceptCount}concepts.csv`);
    setExportStep(null);
    setExportExcludeNonWide(false);
  };

  const advanceExport = (fromStep: ExportStep, excludeNonWide: boolean) => {
    setExportExcludeNonWide(excludeNonWide);
    const missingTotals = excludeNonWide
      ? [...new Set(rows.filter(r => r.type === 'wide' && r.participants === null).map(r => r.cohortId))]
      : exportMissingTotals;
    if (fromStep === 'options' && !excludeNonWide && exportHasMixedTypes) {
      setExportStep('warn_mix');
      return;
    }
    if (exportFormat === 'pct' && missingTotals.length > 0 && fromStep !== 'warn_pct') {
      setExportStep('warn_pct');
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
            inclusive). Each cell shows the number of non-null observations and its percentage of the cohort&apos;s
            participants.
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
              {exportButton}
              <span className="text-sm text-base-content/50">
                {visibleClusters.length} concept clusters · {rows.length} rows
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
                              <span className="block truncate" title={c.label}>
                                {c.label}
                              </span>
                              <span className="block truncate text-xs text-base-content/50" title={c.identifiers.join(', ')}>
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
                  <div className="overflow-auto border border-base-300 rounded-lg" style={{ maxHeight: '75vh' }}>
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
                              className="sticky top-0 z-10 bg-base-200 border-b border-r border-base-300 px-1 align-bottom"
                              title={`${c.label} — ${c.identifiers.join(', ')}`}
                            >
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
                          <tr key={`${row.cohortId}||${row.visitKey}`}>
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
                              {row.visitLabel}
                            </td>
                            {visibleClusters.map(cluster => {
                              const { n, entries } = cellValue(
                                model.cells.get(`${row.cohortId}||${row.visitKey}||${cluster.key}`)
                              );
                              if (entries.length === 0) {
                                return (
                                  <td
                                    key={cluster.key}
                                    className={`border-r border-base-200 px-1 py-1 text-center text-base-content/20 ${
                                      row.firstOfCohort ? 'border-t border-t-base-300' : ''
                                    }`}
                                  >
                                    ·
                                  </td>
                                );
                              }
                              if (row.type === 'none') {
                                // No variable profiling yet: only presence is known;
                                // the participant count stands in for the cell value.
                                return (
                                  <td
                                    key={cluster.key}
                                    className={`border-r border-base-200 px-1 py-1 text-center whitespace-nowrap ${
                                      row.firstOfCohort ? 'border-t border-t-base-300' : ''
                                    }`}
                                    style={{ backgroundColor: 'rgba(148, 163, 184, 0.25)' }}
                                    title={`No variable profiling yet — participant count shown\n${entries
                                      .map(e => `${e.varName} — ${e.varLabel}`)
                                      .join('\n')}`}
                                  >
                                    <div className="font-mono font-semibold">{row.participants ?? '?'}</div>
                                    <div className="text-[10px] text-base-content/60">participants</div>
                                  </td>
                                );
                              }
                              const pct = n !== null && row.participants ? n / row.participants : null;
                              const intensity =
                                pct !== null ? Math.min(pct, 1) : n !== null && maxN > 0 ? Math.min(n / maxN, 1) : 0.15;
                              const bg = `rgba(16, 185, 129, ${0.08 + 0.62 * intensity})`;
                              const tooltip = entries
                                .map(e => `${e.varName}: ${e.count === null ? 'count n/a' : e.count} — ${e.varLabel}`)
                                .join('\n');
                              return (
                                <td
                                  key={cluster.key}
                                  className={`border-r border-base-200 px-1 py-1 text-center whitespace-nowrap ${
                                    row.firstOfCohort ? 'border-t border-t-base-300' : ''
                                  }`}
                                  style={{ backgroundColor: bg }}
                                  title={tooltip}
                                >
                                  <div className="font-mono font-semibold">{n === null ? '?' : n}</div>
                                  <div className="text-[10px] text-base-content/60">
                                    {pct !== null ? formatPct(pct) : '—'}
                                    {entries.length > 1 ? ` · ${entries.length} vars` : ''}
                                  </div>
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
                  Cell color scales with the percentage of non-null observations over the cohort&apos;s participant count.
                  When several variables of a cohort fall in the same cluster and visit, the highest count is shown (hover
                  for all). Gray cells belong to cohorts without variable profiling: only the participant count is known.
                  Long-format cohorts have no fixed visit columns, so their counts aggregate over all visits.
                </p>
              </main>
            </div>
          </>
        )}
      </div>

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
              <span className="font-semibold">{thresholdConfirm.cellCount.toLocaleString()} cells</span>, which can make
              the page slow. All other filters will be reset.
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
            <div className="space-y-2 py-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="export-format"
                  className="radio radio-sm"
                  checked={exportFormat === 'raw'}
                  onChange={() => setExportFormat('raw')}
                />
                <span className="text-sm">Raw numbers (non-null observation counts)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="export-format"
                  className="radio radio-sm"
                  checked={exportFormat === 'pct'}
                  onChange={() => setExportFormat('pct')}
                />
                <span className="text-sm">Percentages (of the cohort&apos;s participant count)</span>
              </label>
            </div>
            <div className="modal-action">
              <button className="btn btn-sm" onClick={() => setExportStep(null)}>
                Cancel
              </button>
              <button className="btn btn-sm btn-primary" onClick={() => advanceExport('options', false)}>
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
                    <span className="font-semibold">Cohorts without variable profiling</span> have no observation counts
                    yet, so their numbers are the <span className="font-semibold">overall number of participants</span>,
                    not observation counts.
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

      {exportStep === 'warn_pct' && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg text-warning">Unknown participant totals</h3>
            <div className="py-3 text-sm space-y-2">
              <p>
                Percentages cannot be computed for the following cohorts because their participant count is unknown;
                their cells will be left empty:
              </p>
              <p className="font-mono text-xs">
                {(exportExcludeNonWide
                  ? [...new Set(rows.filter(r => r.type === 'wide' && r.participants === null).map(r => r.cohortId))]
                  : exportMissingTotals
                ).join(', ')}
              </p>
            </div>
            <div className="modal-action">
              <button className="btn btn-sm" onClick={() => setExportStep(null)}>
                Cancel
              </button>
              <button className="btn btn-sm btn-warning" onClick={() => runExport(exportExcludeNonWide)}>
                Proceed
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
