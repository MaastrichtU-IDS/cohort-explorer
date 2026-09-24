'use client';

import React, { useMemo, useState } from 'react';
import { useCohorts } from '@/components/CohortsContext';
import { Cohort, Variable } from '@/types';
import { Grid, ChevronLeft, ChevronRight, Search } from 'react-feather';

type ClusterMode = 'omop_id' | 'concept_name' | 'concept_code';

const MODE_LABELS: Record<ClusterMode, string> = {
  omop_id: 'OMOP ID',
  concept_name: 'Concept Name',
  concept_code: 'Concept Code',
};

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

// --- Cluster + cell model ---

interface CellEntry {
  varName: string;
  varLabel: string;
  count: number | null;
}

interface HeatCluster {
  key: string;
  label: string;
  conceptHint: string;
  cohortIds: Set<string>;
  variableCount: number;
}

interface HeatModel {
  clusters: HeatCluster[];
  // cohortId -> ordered visit keys (normalized); display labels alongside
  visitsByCohort: Record<string, { key: string; label: string }[]>;
  // `${cohortId}||${visitKey}||${clusterKey}` -> entries
  cells: Map<string, CellEntry[]>;
}

function buildHeatModel(cohortsData: Record<string, Cohort>, mode: ClusterMode): HeatModel {
  const groups: Record<string, { members: { cohortId: string; variable: Variable; visitKeys: string[] }[] }> = {};
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

      const values = splitValues(variable[mode] as string);
      if (values.length === 0) continue;
      for (const val of values) {
        const key = normalizeValue(val);
        if (!groups[key]) groups[key] = { members: [] };
        groups[key].members.push({ cohortId, variable, visitKeys });
      }
    }
  }

  const clusters: HeatCluster[] = [];
  const cells = new Map<string, CellEntry[]>();

  for (const [key, group] of Object.entries(groups)) {
    const cohortIds = new Set(group.members.map(m => m.cohortId));
    if (cohortIds.size < 2) continue;

    // Majority concept name among members, as a readable hint for id/code keys
    const nameCounts: Record<string, number> = {};
    for (const m of group.members) {
      for (const n of splitValues(m.variable.concept_name)) {
        const nn = normalizeValue(n);
        nameCounts[nn] = (nameCounts[nn] || 0) + 1;
      }
    }
    const majorityName = Object.entries(nameCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || '';

    clusters.push({
      key,
      label: key,
      conceptHint: mode === 'concept_name' ? '' : majorityName,
      cohortIds,
      variableCount: group.members.length,
    });

    for (const m of group.members) {
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

  clusters.sort((a, b) => b.cohortIds.size - a.cohortIds.size || b.variableCount - a.variableCount || a.key.localeCompare(b.key));

  const visitsByCohort: Record<string, { key: string; label: string }[]> = {};
  for (const [cohortId, registry] of Object.entries(visitRegistry)) {
    visitsByCohort[cohortId] = [...registry.entries()]
      .sort((a, b) => compareVisitKeys(a[0], b[0]))
      .map(([k, label]) => ({ key: k, label }));
  }

  return { clusters, visitsByCohort, cells };
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

const COLUMN_LIMITS = [25, 50, 100, 0];

export default function ConceptClustersHeatmapPage() {
  const { cohortsData, isLoading } = useCohorts();
  const [mode, setMode] = useState<ClusterMode>('omop_id');
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [cohortFilter, setCohortFilter] = useState<Record<string, boolean>>({});
  const [clusterFilter, setClusterFilter] = useState<Record<string, boolean>>({});
  const [clusterSearch, setClusterSearch] = useState('');
  const [columnLimit, setColumnLimit] = useState(50);

  const model = useMemo(() => {
    if (!cohortsData || Object.keys(cohortsData).length === 0) {
      return { clusters: [], visitsByCohort: {}, cells: new Map() } as HeatModel;
    }
    return buildHeatModel(cohortsData, mode);
  }, [cohortsData, mode]);

  const allCohortIds = useMemo(() => {
    const withClusters = new Set<string>();
    for (const c of model.clusters) c.cohortIds.forEach(id => withClusters.add(id));
    return [...withClusters].sort((a, b) => a.localeCompare(b));
  }, [model]);

  const isCohortSelected = (id: string) => cohortFilter[id] !== false;
  const isClusterSelected = (key: string) => clusterFilter[key] !== false;

  const selectedCohorts = useMemo(() => allCohortIds.filter(isCohortSelected), [allCohortIds, cohortFilter]);

  const visibleClusters = useMemo(() => {
    const selected = model.clusters.filter(
      c => isClusterSelected(c.key) && [...c.cohortIds].some(id => isCohortSelected(id))
    );
    return columnLimit > 0 ? selected.slice(0, columnLimit) : selected;
  }, [model, clusterFilter, cohortFilter, columnLimit, selectedCohorts]);

  const rows = useMemo(() => {
    const out: { cohortId: string; visitKey: string; visitLabel: string; firstOfCohort: boolean; participants: number | null }[] = [];
    for (const cohortId of selectedCohorts) {
      const visits = model.visitsByCohort[cohortId] || [];
      const participants = parseParticipants(cohortsData?.[cohortId]?.study_participants);
      visits.forEach((v, i) => {
        out.push({ cohortId, visitKey: v.key, visitLabel: v.label, firstOfCohort: i === 0, participants });
      });
    }
    return out;
  }, [selectedCohorts, model, cohortsData]);

  const maxN = useMemo(() => {
    let max = 0;
    for (const row of rows) {
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
        ? model.clusters.filter(
            c => c.key.includes(clusterSearchLower) || c.conceptHint.includes(clusterSearchLower)
          )
        : model.clusters,
    [model, clusterSearchLower]
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

  const changeMode = (m: ClusterMode) => {
    setMode(m);
    setClusterFilter({});
    setClusterSearch('');
  };

  return (
    <div className="min-h-screen bg-base-100">
      <div className="px-4 py-6">
        <div className="mb-4">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Grid size={26} />
            Concept Clusters Heatmap
          </h1>
          <p className="text-base-content/60 mt-1 text-sm">
            Rows are cohort + visit; columns are variable clusters (2+ cohorts sharing the same {MODE_LABELS[mode]}). Each
            cell shows the number of non-null observations (from the data dictionary COUNT) and its percentage of the
            cohort&apos;s participants.
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
              <span className="text-sm text-base-content/60">Cluster by:</span>
              <div className="flex gap-2">
                {(Object.keys(MODE_LABELS) as ClusterMode[]).map(m => (
                  <button
                    key={m}
                    onClick={() => changeMode(m)}
                    className={`btn btn-sm ${mode === m ? 'btn-primary' : 'btn-outline'}`}
                  >
                    {MODE_LABELS[m]}
                  </button>
                ))}
              </div>
              <div className="divider divider-horizontal mx-0"></div>
              <span className="text-sm text-base-content/60">Columns:</span>
              <select
                className="select select-sm select-bordered"
                value={columnLimit}
                onChange={e => setColumnLimit(Number(e.target.value))}
              >
                {COLUMN_LIMITS.map(l => (
                  <option key={l} value={l}>
                    {l === 0 ? 'All' : `Top ${l}`}
                  </option>
                ))}
              </select>
              <span className="text-sm text-base-content/50">
                {visibleClusters.length} clusters · {rows.length} rows
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
                            <span className="truncate" title={id}>
                              {id}
                            </span>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="collapse collapse-arrow bg-base-200 border border-base-300">
                    <input type="checkbox" defaultChecked />
                    <div className="collapse-title font-semibold text-sm">Variable clusters ({model.clusters.length})</div>
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
                              <span className="block truncate" title={c.key}>
                                {c.key}
                              </span>
                              {c.conceptHint && (
                                <span className="block truncate text-xs text-base-content/50" title={c.conceptHint}>
                                  {c.conceptHint}
                                </span>
                              )}
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
                </aside>
              )}

              <main className="flex-1 min-w-0">
                {rows.length === 0 || visibleClusters.length === 0 ? (
                  <div className="text-center py-20 text-base-content/40">
                    Nothing to show. Select at least one cohort and one cluster.
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
                              title={c.conceptHint ? `${c.key} — ${c.conceptHint}` : c.key}
                            >
                              <div
                                className="mx-auto overflow-hidden text-ellipsis whitespace-nowrap font-medium"
                                style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', maxHeight: '9rem', minHeight: '9rem' }}
                              >
                                {c.conceptHint || c.key}
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map(row => (
                          <tr key={`${row.cohortId}||${row.visitKey}`}>
                            <td
                              className={`sticky left-0 z-20 bg-base-100 border-r border-base-300 px-2 py-1 font-semibold min-w-[10rem] max-w-[10rem] truncate ${
                                row.firstOfCohort ? 'border-t' : ''
                              }`}
                              title={row.cohortId}
                            >
                              {row.firstOfCohort ? row.cohortId : ''}
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
                  Cell color scales with the percentage of non-null observations over the cohort&apos;s participant count
                  (falls back to the max observation count in view when the participant count is unknown). When several
                  variables of a cohort fall in the same cluster and visit, the highest count is shown (hover for all).
                </p>
              </main>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
