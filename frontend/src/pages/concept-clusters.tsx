'use client';

import React, { useMemo, useState } from 'react';
import { useCohorts } from '@/components/CohortsContext';
import { Cohort, Variable } from '@/types';
import { Layers, ArrowUp, ArrowDown } from 'react-feather';

type ClusterMode = 'concept_name' | 'concept_code' | 'omop_id';
type SortKey = 'cohorts' | 'variables';
type SortDir = 'asc' | 'desc';

interface ClusterMember {
  cohortId: string;
  cohortName: string;
  varName: string;
  varLabel: string;
  visitConceptName: string;
  additionalContext: string;
  unitConceptName: string;
}

interface Cluster {
  key: string;
  members: ClusterMember[];
  cohortCount: number;
  variableCount: number;
}

const MODE_LABELS: Record<ClusterMode, string> = {
  concept_name: 'Variable Concept Name',
  concept_code: 'Variable Concept Code',
  omop_id: 'Variable OMOP ID',
};

function buildClusters(
  cohortsData: Record<string, Cohort>,
  mode: ClusterMode
): Cluster[] {
  const groups: Record<string, ClusterMember[]> = {};

  for (const [cohortId, cohort] of Object.entries(cohortsData)) {
    if (!cohort.variables) continue;
    for (const variable of Object.values(cohort.variables) as Variable[]) {
      const rawValue = variable[mode];
      if (!rawValue || String(rawValue).trim().toLowerCase() === 'na') continue;

      // Some fields may be pipe-separated (e.g. multiple concept names)
      const values = String(rawValue).split('|').map(v => v.trim()).filter(v => v && v.toLowerCase() !== 'na');

      for (const val of values) {
        const key = val;
        if (!groups[key]) groups[key] = [];
        groups[key].push({
          cohortId,
          cohortName: cohortId,
          varName: variable.var_name,
          varLabel: variable.var_label || variable.var_name,
          visitConceptName: variable.visit_concept_name || '',
          additionalContext: variable.additional_context || '',
          unitConceptName: variable.unit_concept_name || '',
        });
      }
    }
  }

  const clusters: Cluster[] = [];
  for (const [key, members] of Object.entries(groups)) {
    if (members.length < 2) continue;
    const uniqueCohorts = new Set(members.map(m => m.cohortId));
    clusters.push({
      key,
      members,
      cohortCount: uniqueCohorts.size,
      variableCount: members.length,
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

function ClusterCard({ cluster, mode }: { cluster: Cluster; mode: ClusterMode }) {
  const [expanded, setExpanded] = useState(false);

  // Group members by cohort
  const byCohort = useMemo(() => {
    const groups: Record<string, ClusterMember[]> = {};
    for (const member of cluster.members) {
      if (!groups[member.cohortId]) groups[member.cohortId] = [];
      groups[member.cohortId].push(member);
    }
    return groups;
  }, [cluster.members]);

  return (
    <div className="card bg-base-200 shadow-sm border border-base-300">
      <div
        className="card-body p-4 cursor-pointer"
        onClick={() => setExpanded(prev => !expanded)}
      >
        {/* Cluster header */}
        <div className="flex items-center justify-between gap-4">
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
          </div>
          <button className="btn btn-ghost btn-sm">
            {expanded ? '▲ Collapse' : '▼ Expand'}
          </button>
        </div>

        {/* Cluster details */}
        {expanded && (
          <div className="mt-4 space-y-3">
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
