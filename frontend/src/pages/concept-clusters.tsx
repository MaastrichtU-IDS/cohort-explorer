'use client';

import React, { useMemo, useState } from 'react';
import { useCohorts } from '@/components/CohortsContext';
import { Cohort, Variable } from '@/types';
import { Layers, ArrowUp, ArrowDown } from 'react-feather';

type ClusterMode = 'concept_name' | 'concept_code' | 'omop_id';
type SortKey = 'cohorts' | 'variables';
type SortDir = 'asc' | 'desc';

const ALL_MODES: ClusterMode[] = ['concept_name', 'concept_code', 'omop_id'];

const MODE_LABELS: Record<ClusterMode, string> = {
  concept_name: 'Variable Concept Name',
  concept_code: 'Variable Concept Code',
  omop_id: 'Variable OMOP ID',
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
  members: ClusterMember[];
  cohortCount: number;
  variableCount: number;
  // Correspondence counts to the other two dimensions
  // e.g. if mode is 'concept_name', correspondences has 'concept_code' and 'omop_id'
  correspondences: Partial<Record<ClusterMode, Record<string, number>>>;
}

function splitValues(raw: string | null | undefined): string[] {
  if (!raw) return [];
  return String(raw)
    .split('|')
    .map(v => v.trim())
    .filter(v => v && v.toLowerCase() !== 'na');
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
        if (!groups[val]) groups[val] = [];
        groups[val].push(member);
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
    for (const otherMode of otherModes) {
      const counts: Record<string, number> = {};
      for (const m of members) {
        const vals = splitValues(m[otherMode]);
        for (const v of vals) {
          counts[v] = (counts[v] || 0) + 1;
        }
      }
      correspondences[otherMode] = counts;
    }

    clusters.push({
      key,
      members,
      cohortCount: uniqueCohorts.size,
      variableCount: members.length,
      correspondences,
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

export default function ConceptClustersPage() {
  const { cohortsData, isLoading } = useCohorts();
  const [mode, setMode] = useState<ClusterMode>('concept_name');
  const [sortKey, setSortKey] = useState<SortKey>('cohorts');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const clusters = useMemo(() => {
    if (!cohortsData || Object.keys(cohortsData).length === 0) return [];
    return buildClusters(cohortsData, mode);
  }, [cohortsData, mode]);

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
                    className={`btn btn-sm ${mode === m ? 'btn-primary' : 'btn-outline'}`}
                  >
                    {MODE_LABELS[m]}
                  </button>
                ))}
              </div>

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

            {/* Summary */}
            <div className="mb-4 text-sm text-base-content/70">
              <span className="font-semibold">{sortedClusters.length}</span> clusters found
              {' '}
              <span className="text-base-content/40">
                (grouped by {MODE_LABELS[mode]})
              </span>
            </div>

            {/* Clusters */}
            {sortedClusters.length === 0 ? (
              <div className="text-center py-20 text-base-content/40">
                No clusters found. This means no two variables share the same {MODE_LABELS[mode]}.
              </div>
            ) : (
              <div className="space-y-4">
                {sortedClusters.map((cluster, idx) => (
                  <ClusterCard key={`${mode}-${cluster.key}-${idx}`} cluster={cluster} mode={mode} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function CorrespondenceBadges({ cluster, mode }: { cluster: Cluster; mode: ClusterMode }) {
  const otherModes = ALL_MODES.filter(m => m !== mode);

  return (
    <div className="mt-2 space-y-1">
      {otherModes.map(otherMode => {
        const counts = cluster.correspondences[otherMode];
        if (!counts || Object.keys(counts).length === 0) return null;
        const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        const total = sorted.reduce((sum, [, c]) => sum + c, 0);

        return (
          <div key={otherMode} className="text-xs">
            <span className="font-semibold text-base-content/60">{MODE_LABELS[otherMode]}s:</span>{' '}
            {sorted.map(([val, count], i) => (
              <span key={val}>
                {i > 0 && ', '}
                <span className={count === total ? 'font-semibold' : 'text-base-content/50'}>
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

const AGG_LABELS: Record<AggColumn, string> = {
  visits: 'Visits',
  visitConceptName: 'Visit Concept',
  additionalContext: 'Additional Context',
  unitConceptName: 'Unit Concept',
};

function ClusterCard({ cluster, mode }: { cluster: Cluster; mode: ClusterMode }) {
  const [expanded, setExpanded] = useState(false);
  const [aggColumn, setAggColumn] = useState<AggColumn | null>(null);

  // Group members by cohort
  const byCohort = useMemo(() => {
    const groups: Record<string, ClusterMember[]> = {};
    for (const member of cluster.members) {
      if (!groups[member.cohortId]) groups[member.cohortId] = [];
      groups[member.cohortId].push(member);
    }
    return groups;
  }, [cluster.members]);

  // Build histogram for the selected aggregation column
  const aggHistogram = useMemo(() => {
    if (!aggColumn) return null;
    const counts: Record<string, number> = {};
    for (const m of cluster.members) {
      const raw = m[aggColumn];
      if (!raw || raw.trim().toLowerCase() === 'na') continue;
      // Split pipe-separated values
      const vals = raw.split('|').map(v => v.trim()).filter(v => v && v.toLowerCase() !== 'na');
      for (const v of vals) {
        counts[v] = (counts[v] || 0) + 1;
      }
    }
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const maxCount = sorted.length > 0 ? sorted[0][1] : 0;
    return { sorted, maxCount, total: cluster.members.length };
  }, [aggColumn, cluster.members]);

  return (
    <div className="card bg-base-200 shadow-sm border border-base-300">
      <div className="card-body p-4">
        {/* Cluster header */}
        <div
          className="flex items-center justify-between gap-4 cursor-pointer"
          onClick={() => setExpanded(prev => !expanded)}
        >
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-lg truncate" title={cluster.key}>
              {cluster.key}
            </h3>
            <div className="flex gap-3 mt-1 text-sm">
              <span className="badge badge-sm badge-primary">
                {cluster.cohortCount} {cluster.cohortCount === 1 ? 'cohort' : 'cohorts'}
              </span>
              <span className="badge badge-sm badge-secondary">
                {cluster.variableCount} {cluster.variableCount === 1 ? 'variable' : 'variables'}
              </span>
            </div>

            {/* Correspondence counts */}
            <CorrespondenceBadges cluster={cluster} mode={mode} />
          </div>
          <button className="btn btn-ghost btn-sm">
            {expanded ? '▲ Collapse' : '▼ Expand'}
          </button>
        </div>

        {/* Aggregation controls */}
        {expanded && (
          <div className="flex flex-wrap gap-2 items-center mt-2" onClick={e => e.stopPropagation()}>
            <span className="text-xs text-base-content/60">Aggregate:</span>
            {(Object.keys(AGG_LABELS) as AggColumn[]).map(col => (
              <button
                key={col}
                onClick={() => setAggColumn(prev => prev === col ? null : col)}
                className={`btn btn-xs ${aggColumn === col ? 'btn-primary' : 'btn-outline'}`}
              >
                {AGG_LABELS[col]}
              </button>
            ))}
          </div>
        )}

        {/* Aggregation histogram */}
        {expanded && aggHistogram && (
          <div className="bg-base-100 rounded-lg p-3 border border-base-200 mt-2" onClick={e => e.stopPropagation()}>
            <div className="font-medium text-sm mb-2">
              {AGG_LABELS[aggColumn!]} distribution
            </div>
            {aggHistogram.sorted.length === 0 ? (
              <p className="text-xs text-base-content/40">No values for this column.</p>
            ) : (
              <div className="space-y-1">
                {aggHistogram.sorted.map(([val, count]) => (
                  <div key={val} className="flex items-center gap-2 text-xs">
                    <div className="flex-shrink-0 w-40 truncate" title={val}>{val}</div>
                    <div className="flex-1 bg-base-200 rounded-full h-4 overflow-hidden">
                      <div
                        className="bg-primary h-full rounded-full"
                        style={{ width: `${(count / aggHistogram.maxCount) * 100}%` }}
                      />
                    </div>
                    <div className="flex-shrink-0 w-8 text-right font-mono">{count}</div>
                  </div>
                ))}
              </div>
            )}
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
                        <th>Visit Concept</th>
                        <th>Visits</th>
                        <th>Additional Context</th>
                        <th>Unit Concept</th>
                      </tr>
                    </thead>
                    <tbody>
                      {members.map((m, i) => (
                        <tr key={i}>
                          <td className="font-mono text-xs">{m.varName}</td>
                          <td>{m.varLabel}</td>
                          <td className="text-xs">{m.visitConceptName || '—'}</td>
                          <td className="text-xs">{m.visits || '—'}</td>
                          <td className="text-xs">{m.additionalContext || '—'}</td>
                          <td className="text-xs">{m.unitConceptName || '—'}</td>
                        </tr>
                      ))}
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
