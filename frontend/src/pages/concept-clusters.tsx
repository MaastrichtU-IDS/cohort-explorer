'use client';

import React, { useMemo, useState } from 'react';
import { useCohorts } from '@/components/CohortsContext';
import { Cohort, Variable } from '@/types';
import { Layers, ArrowUp, ArrowDown, AlertTriangle, Maximize2, X } from 'react-feather';

type ClusterMode = 'concept_name' | 'concept_code' | 'omop_id';
type SortKey = 'cohorts' | 'variables';
type SortDir = 'asc' | 'desc';

const ALL_MODES: ClusterMode[] = ['concept_name', 'concept_code', 'omop_id'];

const MODE_LABELS: Record<ClusterMode, string> = {
  concept_name: 'Variable Concept Name',
  concept_code: 'Variable Concept Code',
  omop_id: 'Variable OMOP ID',
};

const MODE_SHORT: Record<ClusterMode, string> = {
  concept_name: 'concept name',
  concept_code: 'concept code',
  omop_id: 'OMOP ID',
};

// The three cluster types are told apart by color: a left border on the card
// and a chip naming the grouping. Essential in the discrepancy view, where
// cards from all three groupings are shown together.
const MODE_STYLE: Record<ClusterMode, { border: string; chip: string }> = {
  concept_name: { border: 'border-l-sky-500', chip: 'bg-sky-100 text-sky-900 border-sky-300' },
  concept_code: { border: 'border-l-violet-500', chip: 'bg-violet-100 text-violet-900 border-violet-300' },
  omop_id: { border: 'border-l-emerald-500', chip: 'bg-emerald-100 text-emerald-900 border-emerald-300' },
};

interface ClusterMember {
  cohortId: string;
  varName: string;
  varLabel: string;
  visitConceptName: string;
  visits: string;
  additionalContext: string;
  unitConceptName: string;
  concept_name: string;
  concept_code: string;
  omop_id: string;
}

interface Cluster {
  key: string;
  mode: ClusterMode;
  members: ClusterMember[];
  cohortCount: number;
  variableCount: number;
  // Correspondence counts to the other two dimensions
  // e.g. if mode is 'concept_name', correspondences has 'concept_code' and 'omop_id'
  correspondences: Partial<Record<ClusterMode, Record<string, number>>>;
  // Most common value per other dimension (the "majority mapping").
  majority: Partial<Record<ClusterMode, string>>;
  // Members disagree on at least one of the other identifiers.
  hasDiscrepancy: boolean;
}

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

function buildClusters(
  cohortsData: Record<string, Cohort>,
  mode: ClusterMode
): Cluster[] {
  const groups: Record<string, ClusterMember[]> = {};

  for (const [cohortId, cohort] of Object.entries(cohortsData)) {
    if (!cohort.variables) continue;
    for (const variable of Object.values(cohort.variables) as Variable[]) {
      const values = splitValues(variable[mode] as string);
      if (values.length === 0) continue;

      const member: ClusterMember = {
        cohortId,
        varName: variable.var_name,
        varLabel: variable.var_label || variable.var_name,
        visitConceptName: variable.visit_concept_name || '',
        visits: variable.visits || '',
        additionalContext: variable.additional_context || '',
        unitConceptName: variable.unit_concept_name || '',
        concept_name: variable.concept_name || '',
        concept_code: variable.concept_code || '',
        omop_id: variable.omop_id ? String(variable.omop_id) : '',
      };

      for (const val of values) {
        const normKey = normalizeValue(val);
        if (!groups[normKey]) groups[normKey] = [];
        groups[normKey].push(member);
      }
    }
  }

  const otherModes = ALL_MODES.filter(m => m !== mode);

  const clusters: Cluster[] = [];
  for (const [key, members] of Object.entries(groups)) {
    if (members.length < 2) continue;
    const uniqueCohorts = new Set(members.map(m => m.cohortId));

    // Build correspondence counts for each other mode
    const correspondences: Partial<Record<ClusterMode, Record<string, number>>> = {};
    const majority: Partial<Record<ClusterMode, string>> = {};
    let hasDiscrepancy = false;
    for (const otherMode of otherModes) {
      const counts: Record<string, number> = {};
      for (const m of members) {
        const vals = splitValues(m[otherMode]);
        for (const v of vals) {
          const nv = normalizeValue(v);
          counts[nv] = (counts[nv] || 0) + 1;
        }
      }
      correspondences[otherMode] = counts;
      const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
      if (sorted.length > 0) majority[otherMode] = sorted[0][0];
      // Two or more distinct values for the same identifier = the members
      // are not standardized the same way (missing values are not conflicts).
      if (sorted.length > 1) hasDiscrepancy = true;
    }

    clusters.push({
      key,
      mode,
      members,
      cohortCount: uniqueCohorts.size,
      variableCount: members.length,
      correspondences,
      majority,
      hasDiscrepancy,
    });
  }

  return clusters;
}

function sortClusters(
  clusters: Cluster[],
  sortKey: SortKey,
  sortDir: SortDir
): Cluster[] {
  const sorted = [...clusters].sort((a, b) => {
    const aVal = sortKey === 'cohorts' ? a.cohortCount : a.variableCount;
    const bVal = sortKey === 'cohorts' ? b.cohortCount : b.variableCount;
    return aVal - bVal;
  });
  return sortDir === 'desc' ? sorted.reverse() : sorted;
}

// Which of a member's other-identifier values disagree with the cluster's
// majority mapping. Empty result = in sync.
function outOfSyncFields(member: ClusterMember, cluster: Cluster): { mode: ClusterMode; value: string; majority: string }[] {
  const out: { mode: ClusterMode; value: string; majority: string }[] = [];
  for (const otherMode of ALL_MODES) {
    if (otherMode === cluster.mode) continue;
    const maj = cluster.majority[otherMode];
    if (!maj) continue;
    const vals = splitValues(member[otherMode]).map(normalizeValue);
    if (vals.length === 0) continue;
    if (!vals.includes(maj) || vals.length > 1) {
      out.push({ mode: otherMode, value: splitValues(member[otherMode]).join(' | '), majority: maj });
    }
  }
  return out;
}

export default function ConceptClustersPage() {
  const { cohortsData, isLoading } = useCohorts();
  const [mode, setMode] = useState<ClusterMode>('concept_name');
  const [sortKey, setSortKey] = useState<SortKey>('cohorts');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  // Discrepancy view: every cluster with a standardization discrepancy, from
  // ALL three groupings at once (the mode buttons are then not applicable).
  const [onlyDiscrepancies, setOnlyDiscrepancies] = useState(false);

  const clustersByMode = useMemo(() => {
    const out: Record<ClusterMode, Cluster[]> = { concept_name: [], concept_code: [], omop_id: [] };
    if (!cohortsData || Object.keys(cohortsData).length === 0) return out;
    for (const m of ALL_MODES) out[m] = buildClusters(cohortsData, m);
    return out;
  }, [cohortsData]);

  const clusters = useMemo(() => {
    if (onlyDiscrepancies) {
      return ALL_MODES.flatMap(m => clustersByMode[m].filter(c => c.hasDiscrepancy));
    }
    return clustersByMode[mode];
  }, [clustersByMode, mode, onlyDiscrepancies]);

  const discrepancyTotal = useMemo(
    () => ALL_MODES.reduce((sum, m) => sum + clustersByMode[m].filter(c => c.hasDiscrepancy).length, 0),
    [clustersByMode]
  );

  const sortedClusters = useMemo(() => {
    return sortClusters(clusters, sortKey, sortDir);
  }, [clusters, sortKey, sortDir]);

  const toggleSortDir = () => {
    setSortDir(prev => (prev === 'asc' ? 'desc' : 'asc'));
  };

  return (
    <div className="min-h-screen bg-base-100">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Layers size={28} />
            Concept Clusters
          </h1>
          <p className="text-base-content/60 mt-1">
            Variables grouped by shared concept metadata across cohorts. A cluster is any 2+ variables sharing the same value.
          </p>
        </div>

        {isLoading ? (
          <div className="flex justify-center items-center py-20">
            <span className="loading loading-spinner loading-lg"></span>
          </div>
        ) : (
          <>
            {/* Controls */}
            <div className="flex flex-wrap gap-4 mb-6 items-center">
              {/* Mode selection */}
              <div className="flex gap-2">
                {(Object.keys(MODE_LABELS) as ClusterMode[]).map(m => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    disabled={onlyDiscrepancies}
                    className={`btn btn-sm ${mode === m && !onlyDiscrepancies ? 'btn-primary' : 'btn-outline'}`}
                    title={onlyDiscrepancies ? 'The discrepancy view shows all three groupings at once' : undefined}
                  >
                    {MODE_LABELS[m]}
                  </button>
                ))}
              </div>

              {/* Divider */}
              <div className="divider divider-horizontal mx-0"></div>

              {/* Discrepancy filter */}
              <button
                onClick={() => setOnlyDiscrepancies(prev => !prev)}
                className={`btn btn-sm gap-1 ${onlyDiscrepancies ? 'btn-warning' : 'btn-outline btn-warning'}`}
                title="Show only clusters whose members are not standardized the same way (different concept names, concept codes or OMOP IDs), across all three groupings"
              >
                <AlertTriangle size={14} />
                {onlyDiscrepancies ? 'Showing discrepancies only' : 'Show only discrepancies'}
                <span className="badge badge-sm">{discrepancyTotal}</span>
              </button>

              {/* Divider */}
              <div className="divider divider-horizontal mx-0"></div>

              {/* Sort key selection */}
              <div className="flex gap-2 items-center">
                <span className="text-sm text-base-content/60">Sort by:</span>
                <button
                  onClick={() => setSortKey('cohorts')}
                  className={`btn btn-sm ${sortKey === 'cohorts' ? 'btn-primary' : 'btn-outline'}`}
                >
                  Cohorts
                </button>
                <button
                  onClick={() => setSortKey('variables')}
                  className={`btn btn-sm ${sortKey === 'variables' ? 'btn-primary' : 'btn-outline'}`}
                >
                  Variables
                </button>
              </div>

              {/* Sort direction */}
              <button
                onClick={toggleSortDir}
                className="btn btn-sm btn-ghost gap-1"
                title={sortDir === 'asc' ? 'Ascending' : 'Descending'}
              >
                {sortDir === 'asc' ? <ArrowUp size={16} /> : <ArrowDown size={16} />}
                {sortDir === 'asc' ? 'Ascending' : 'Descending'}
              </button>
            </div>

            {/* Summary + legend */}
            <div className="mb-4 text-sm text-base-content/70 flex flex-wrap items-center gap-x-4 gap-y-1">
              <span>
                <span className="font-semibold">{sortedClusters.length}</span> clusters
                {onlyDiscrepancies ? (
                  <span className="text-base-content/40"> with a standardization discrepancy (all three groupings)</span>
                ) : (
                  <span className="text-base-content/40"> (grouped by {MODE_LABELS[mode]})</span>
                )}
              </span>
              <span className="flex items-center gap-2 text-xs">
                {ALL_MODES.map(m => (
                  <span key={m} className={`px-2 py-0.5 rounded-full border ${MODE_STYLE[m].chip}`}>
                    grouped by {MODE_SHORT[m]}
                  </span>
                ))}
              </span>
            </div>

            {/* Clusters */}
            {sortedClusters.length === 0 ? (
              <div className="text-center py-20 text-base-content/40">
                {onlyDiscrepancies
                  ? 'No cluster has a standardization discrepancy.'
                  : `No clusters found. This means no two variables share the same ${MODE_LABELS[mode]}.`}
              </div>
            ) : (
              <div className="space-y-4">
                {sortedClusters.map((cluster, idx) => (
                  <ClusterCard key={`${cluster.mode}-${cluster.key}-${idx}`} cluster={cluster} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function CorrespondenceBadges({ cluster }: { cluster: Cluster }) {
  const otherModes = ALL_MODES.filter(m => m !== cluster.mode);

  return (
    <div className="mt-2 space-y-1">
      {otherModes.map(otherMode => {
        const counts = cluster.correspondences[otherMode];
        if (!counts || Object.keys(counts).length === 0) return null;
        const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        const total = sorted.reduce((sum, [, c]) => sum + c, 0);
        const conflicting = sorted.length > 1;

        return (
          <div key={otherMode} className="text-xs">
            <span className="font-semibold text-base-content/60">{MODE_LABELS[otherMode]}s:</span>{' '}
            {sorted.map(([val, count], i) => (
              <span key={val}>
                {i > 0 && ', '}
                <span
                  className={
                    count === total
                      ? 'font-semibold'
                      : conflicting && i === 0
                        ? 'font-semibold'
                        : conflicting
                          ? 'text-amber-700 font-semibold'
                          : 'text-base-content/50'
                  }
                >
                  {val}
                </span>
                <span className="text-base-content/40"> ({count})</span>
              </span>
            ))}
          </div>
        );
      })}
    </div>
  );
}

type AggColumn = 'visits' | 'visitConceptName' | 'additionalContext' | 'unitConceptName';

const AGG_COLUMNS: AggColumn[] = ['visits', 'visitConceptName', 'additionalContext', 'unitConceptName'];

const AGG_LABELS: Record<AggColumn, string> = {
  visits: 'Visits',
  visitConceptName: 'Visit Concept',
  additionalContext: 'Additional Context',
  unitConceptName: 'Unit Concept',
};

// Rows shown in the compact (small) graph; the enlarged graph shows them all.
const SMALL_GRAPH_ROWS = 6;

interface Histogram {
  sorted: [string, number][];
  maxCount: number;
}

function buildHistogram(members: ClusterMember[], col: AggColumn): Histogram {
  const counts: Record<string, number> = {};
  for (const m of members) {
    const raw = m[col];
    if (!raw || raw.trim().toLowerCase() === 'na') continue;
    const vals = raw.split('|').map(v => v.trim()).filter(v => v && v.toLowerCase() !== 'na');
    for (const v of vals) {
      counts[v] = (counts[v] || 0) + 1;
    }
  }
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return { sorted, maxCount: sorted.length > 0 ? sorted[0][1] : 0 };
}

function AggGraph({
  col,
  histogram,
  enlarged,
  onToggle,
}: {
  col: AggColumn;
  histogram: Histogram;
  enlarged: boolean;
  onToggle: () => void;
}) {
  const rows = enlarged ? histogram.sorted : histogram.sorted.slice(0, SMALL_GRAPH_ROWS);
  const hidden = histogram.sorted.length - rows.length;
  return (
    <div
      className={`bg-base-100 rounded-lg border border-base-200 relative ${enlarged ? 'col-span-2 p-3' : 'p-2'}`}
      onClick={e => e.stopPropagation()}
    >
      <div className="flex items-center justify-between mb-1">
        <div className={`font-medium ${enlarged ? 'text-sm' : 'text-xs'}`}>{AGG_LABELS[col]}</div>
        <button
          type="button"
          onClick={onToggle}
          className="btn btn-ghost btn-xs px-1"
          title={enlarged ? 'Shrink' : 'Enlarge'}
          aria-label={enlarged ? 'Shrink graph' : 'Enlarge graph'}
        >
          {enlarged ? <X size={14} /> : <Maximize2 size={12} />}
        </button>
      </div>
      {histogram.sorted.length === 0 ? (
        <p className="text-[11px] text-base-content/40">No values.</p>
      ) : (
        <div className={enlarged ? 'space-y-1' : 'space-y-0.5'}>
          {rows.map(([val, count]) => (
            <div key={val} className={`flex items-center gap-2 ${enlarged ? 'text-xs' : 'text-[11px]'}`}>
              <div className={`flex-shrink-0 truncate ${enlarged ? 'w-64' : 'w-28'}`} title={val}>{val}</div>
              <div className={`flex-1 bg-base-200 rounded-full overflow-hidden ${enlarged ? 'h-3.5' : 'h-2'}`}>
                <div
                  className="bg-primary h-full rounded-full"
                  style={{ width: `${(count / histogram.maxCount) * 100}%` }}
                />
              </div>
              <div className="flex-shrink-0 w-7 text-right font-mono">{count}</div>
            </div>
          ))}
          {hidden > 0 && (
            <div className="text-[11px] text-base-content/40 italic">+{hidden} more (enlarge to see all)</div>
          )}
        </div>
      )}
    </div>
  );
}

function ClusterCard({ cluster }: { cluster: Cluster }) {
  const [expanded, setExpanded] = useState(false);
  const [enlarged, setEnlarged] = useState<AggColumn | null>(null);
  const style = MODE_STYLE[cluster.mode];

  // Group members by cohort
  const byCohort = useMemo(() => {
    const groups: Record<string, ClusterMember[]> = {};
    for (const member of cluster.members) {
      if (!groups[member.cohortId]) groups[member.cohortId] = [];
      groups[member.cohortId].push(member);
    }
    return groups;
  }, [cluster.members]);

  // All four aggregation graphs are shown at once
  const histograms = useMemo(() => {
    const out = {} as Record<AggColumn, Histogram>;
    for (const col of AGG_COLUMNS) out[col] = buildHistogram(cluster.members, col);
    return out;
  }, [cluster.members]);

  // Members that disagree with the majority mapping on another identifier
  const outOfSync = useMemo(
    () =>
      cluster.members
        .map(m => ({ member: m, diffs: outOfSyncFields(m, cluster) }))
        .filter(x => x.diffs.length > 0),
    [cluster]
  );
  const outOfSyncKeys = useMemo(
    () => new Set(outOfSync.map(x => `${x.member.cohortId}::${x.member.varName}`)),
    [outOfSync]
  );

  const otherModes = ALL_MODES.filter(m => m !== cluster.mode);

  return (
    <div
      className={`card bg-base-200 shadow-sm border border-base-300 border-l-4 ${style.border} ${
        cluster.hasDiscrepancy ? 'ring-1 ring-amber-300' : ''
      }`}
    >
      <div className="card-body p-4">
        {/* Cluster header */}
        <div
          className="flex items-center justify-between gap-4 cursor-pointer"
          onClick={() => setExpanded(prev => !prev)}
        >
          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="font-semibold text-lg truncate" title={cluster.key}>
                {cluster.key}
              </h3>
              <span className={`px-2 py-0.5 rounded-full border text-[11px] uppercase tracking-wide whitespace-nowrap ${style.chip}`}>
                grouped by {MODE_SHORT[cluster.mode]}
              </span>
              {cluster.hasDiscrepancy && (
                <span
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-400 text-xs font-semibold whitespace-nowrap"
                  title="Members of this cluster are not standardized the same way: they carry different values for at least one of the other identifiers"
                >
                  <AlertTriangle size={12} /> Standardization discrepancy
                </span>
              )}
            </div>
            <div className="flex gap-3 mt-1 text-sm">
              <span className="badge badge-sm badge-primary">
                {cluster.cohortCount} {cluster.cohortCount === 1 ? 'cohort' : 'cohorts'}
              </span>
              <span className="badge badge-sm badge-secondary">
                {cluster.variableCount} {cluster.variableCount === 1 ? 'variable' : 'variables'}
              </span>
              {outOfSync.length > 0 && (
                <span className="badge badge-sm badge-warning">
                  {outOfSync.length} out of sync
                </span>
              )}
            </div>

            {/* Correspondence counts */}
            <CorrespondenceBadges cluster={cluster} />
          </div>
          <button className="btn btn-ghost btn-sm">
            {expanded ? '▲ Collapse' : '▼ Expand'}
          </button>
        </div>

        {/* Aggregation graphs: all of them, compact, two per row; one can be
            enlarged to full width (the arrow turns into a close button). */}
        {expanded && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-3">
            {AGG_COLUMNS.map(col => (
              <AggGraph
                key={col}
                col={col}
                histogram={histograms[col]}
                enlarged={enlarged === col}
                onToggle={() => setEnlarged(prev => (prev === col ? null : col))}
              />
            ))}
          </div>
        )}

        {/* Out-of-sync members first: they differ from the majority mapping */}
        {expanded && outOfSync.length > 0 && (
          <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-2 font-medium text-sm text-amber-900 mb-2">
              <AlertTriangle size={14} />
              Out of sync with the majority mapping ({outOfSync.length})
            </div>
            <div className="overflow-x-auto">
              <table className="table table-xs">
                <thead>
                  <tr>
                    <th>Cohort</th>
                    <th>Variable</th>
                    <th>Label</th>
                    <th>Differs on</th>
                  </tr>
                </thead>
                <tbody>
                  {outOfSync.map(({ member, diffs }, i) => (
                    <tr key={i} className="bg-amber-100/60">
                      <td className="text-xs whitespace-nowrap">{member.cohortId}</td>
                      <td className="font-mono text-xs">{member.varName}</td>
                      <td className="text-xs">{member.varLabel}</td>
                      <td className="text-xs">
                        {diffs.map(d => (
                          <div key={d.mode}>
                            <span className="font-semibold">{MODE_SHORT[d.mode]}:</span>{' '}
                            <span className="text-amber-800 font-mono">{d.value}</span>
                            <span className="text-base-content/50"> (majority: {d.majority})</span>
                          </div>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Cluster details */}
        {expanded && (
          <div className="mt-4 space-y-3" onClick={e => e.stopPropagation()}>
            {Object.entries(byCohort).map(([cohortId, members]) => (
              <div key={cohortId} className="bg-base-100 rounded-lg p-3 border border-base-200">
                <div className="font-medium text-sm text-base-content/80 mb-2">
                  {cohortId}
                </div>
                <div className="overflow-x-auto">
                  <table className="table table-xs">
                    <thead>
                      <tr>
                        <th>Variable</th>
                        <th>Label</th>
                        {otherModes.map(m => (
                          <th key={m}>{MODE_LABELS[m]}</th>
                        ))}
                        <th>Visit Concept</th>
                        <th>Visits</th>
                        <th>Additional Context</th>
                        <th>Unit Concept</th>
                      </tr>
                    </thead>
                    <tbody>
                      {members.map((m, i) => {
                        const isOff = outOfSyncKeys.has(`${m.cohortId}::${m.varName}`);
                        const diffModes = new Set(isOff ? outOfSyncFields(m, cluster).map(d => d.mode) : []);
                        return (
                          <tr key={i} className={isOff ? 'bg-amber-100/60' : ''}>
                            <td className="font-mono text-xs">
                              {isOff && <AlertTriangle size={11} className="inline mr-1 text-amber-700" />}
                              {m.varName}
                            </td>
                            <td>{m.varLabel}</td>
                            {otherModes.map(om => (
                              <td
                                key={om}
                                className={`text-xs font-mono ${diffModes.has(om) ? 'text-amber-800 font-semibold' : ''}`}
                              >
                                {m[om] || '—'}
                              </td>
                            ))}
                            <td className="text-xs">{m.visitConceptName || '—'}</td>
                            <td className="text-xs">{m.visits || '—'}</td>
                            <td className="text-xs">{m.additionalContext || '—'}</td>
                            <td className="text-xs">{m.unitConceptName || '—'}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
