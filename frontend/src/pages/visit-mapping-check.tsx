'use client';

import React, { useEffect, useState } from 'react';
import { useCohorts } from '@/components/CohortsContext';
import { apiUrl } from '@/utils';
import LoginPrompt from '@/components/LoginPrompt';
import { AlertTriangle, CheckCircle, Activity } from 'react-feather';

interface SuspectVariable {
  visit_concept_name: string;
  variable_count: number;
  variables: [string, string][];
}

interface SuspectMapping {
  visits_value: string;
  majority: {
    visit_concept_name: string;
    variable_count: number;
    variables: [string, string][];
  };
  minorities: SuspectVariable[];
  total_variables: number;
  distinct_concept_names: string[];
}

interface VisitMappingResult {
  total_visits_values: number;
  suspect_count: number;
  suspect_mappings: SuspectMapping[];
}

export default function VisitMappingCheckPage() {
  const { userEmail } = useCohorts();
  const [data, setData] = useState<VisitMappingResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (userEmail === null) return;
    setLoading(true);
    fetch(`${apiUrl}/api/check-visit-mapping`, { credentials: 'include' })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((d: VisitMappingResult) => {
        setData(d);
        setError(null);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [userEmail]);

  if (userEmail === null) {
    return (
      <LoginPrompt message="Authenticate to access this page" />
    );
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center py-20">
        <span className="loading loading-spinner loading-lg"></span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="alert alert-error">
          <AlertTriangle size={20} />
          <span>Error: {error}</span>
        </div>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="min-h-screen bg-base-100">
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Activity size={28} />
            Visit Mapping Consistency Check
          </h1>
          <p className="text-base-content/60 mt-1">
            Detects inconsistencies where the same visit value is mapped to different visit concept names across cohorts.
          </p>
        </div>

        {/* Summary cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
          <div className="stat bg-base-200 rounded-lg border border-base-300">
            <div className="stat-title">Total distinct visit values</div>
            <div className="stat-value text-primary">{data.total_visits_values}</div>
          </div>
          <div className="stat bg-base-200 rounded-lg border border-base-300">
            <div className="stat-title">Suspect mappings</div>
            <div className={`stat-value ${data.suspect_count > 0 ? 'text-warning' : 'text-success'}`}>
              {data.suspect_count}
            </div>
          </div>
        </div>

        {/* Results */}
        {data.suspect_count === 0 ? (
          <div className="flex items-center gap-3 p-6 bg-success/10 rounded-lg border border-success/20">
            <CheckCircle size={28} className="text-success" />
            <div>
              <h3 className="font-semibold text-lg">All visit mappings are consistent</h3>
              <p className="text-sm text-base-content/60">
                Every visit value maps to exactly one visit concept name across all cohorts.
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-warning">
              <AlertTriangle size={20} />
              <span className="font-semibold">
                {data.suspect_count} visit {data.suspect_count === 1 ? 'value' : 'values'} mapped to multiple concept names
              </span>
            </div>

            {data.suspect_mappings.map((sm, idx) => (
              <SuspectCard key={idx} suspect={sm} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SuspectCard({ suspect }: { suspect: SuspectMapping }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="card bg-base-200 shadow-sm border border-base-300">
      <div className="card-body p-4">
        {/* Header */}
        <div
          className="flex items-center justify-between gap-4 cursor-pointer"
          onClick={() => setExpanded(prev => !prev)}
        >
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-lg truncate" title={suspect.visits_value}>
              {suspect.visits_value}
            </h3>
            <div className="flex flex-wrap gap-2 mt-1">
              <span className="badge badge-sm badge-secondary">
                {suspect.total_variables} variables
              </span>
              <span className="badge badge-sm badge-warning">
                {suspect.distinct_concept_names.length} distinct concept names
              </span>
            </div>
            <div className="text-xs mt-1 text-base-content/60">
              <span className="font-semibold">Majority:</span>{' '}
              <span className="text-success">{suspect.majority.visit_concept_name}</span>
              {' '}({suspect.majority.variable_count})
              {suspect.minorities.map(m => (
                <span key={m.visit_concept_name}>
                  {' · '}
                  <span className="text-warning">{m.visit_concept_name}</span>
                  {' '}({m.variable_count})
                </span>
              ))}
            </div>
          </div>
          <button className="btn btn-ghost btn-sm">
            {expanded ? '▲ Collapse' : '▼ Expand'}
          </button>
        </div>

        {/* Inconsistency description */}
        <div className="mt-2 bg-warning/10 border border-warning/20 rounded-lg p-3 text-sm">
          <div className="flex items-start gap-2">
            <AlertTriangle size={18} className="text-warning flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-semibold">
                Inconsistency detected for visits value: <span className="font-mono">{suspect.visits_value}</span>
              </p>
              <p className="text-base-content/70">
                This visits value is mapped to <span className="font-semibold text-warning">{suspect.distinct_concept_names.length} different visit concept names</span> across {suspect.total_variables} variables:
              </p>
              <ul className="list-disc list-inside space-y-0.5 text-base-content/70">
                <li>
                  <span className="text-success font-semibold">{suspect.majority.visit_concept_name}</span>
                  {' — '}
                  {suspect.majority.variable_count} {suspect.majority.variable_count === 1 ? 'variable' : 'variables'}
                  {' (majority mapping)'}
                </li>
                {suspect.minorities.map((m, i) => (
                  <li key={i}>
                    <span className="text-warning font-semibold">{m.visit_concept_name}</span>
                    {' — '}
                    {m.variable_count} {m.variable_count === 1 ? 'variable' : 'variables'}
                    {' (suspect mapping)'}
                  </li>
                ))}
              </ul>
              <p className="text-base-content/50 text-xs mt-1">
                The majority mapping is assumed correct. Variables in the suspect mapping(s) should be reviewed and remapped to <span className="font-semibold">{suspect.majority.visit_concept_name}</span>.
              </p>
            </div>
          </div>
        </div>

        {/* Details */}
        {expanded && (
          <div className="mt-4 space-y-3" onClick={e => e.stopPropagation()}>
            {/* Majority mapping */}
            <div className="bg-success/10 rounded-lg p-3 border border-success/20">
              <div className="font-medium text-sm mb-2 flex items-center gap-2">
                <CheckCircle size={16} className="text-success" />
                Majority: {suspect.majority.visit_concept_name} ({suspect.majority.variable_count} variables)
              </div>
              <VariableList variables={suspect.majority.variables} visitsValue={suspect.visits_value} visitConceptName={suspect.majority.visit_concept_name} />
            </div>

            {/* Minority mappings */}
            {suspect.minorities.map((m, i) => (
              <div key={i} className="bg-warning/10 rounded-lg p-3 border border-warning/20">
                <div className="font-medium text-sm mb-2 flex items-center gap-2">
                  <AlertTriangle size={16} className="text-warning" />
                  Suspect: {m.visit_concept_name} ({m.variable_count} variables)
                </div>
                <VariableList variables={m.variables} visitsValue={suspect.visits_value} visitConceptName={m.visit_concept_name} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function VariableList({ variables, visitsValue, visitConceptName }: { variables: [string, string][]; visitsValue: string; visitConceptName: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="table table-xs">
        <thead>
          <tr>
            <th>Variable</th>
            <th>Cohort</th>
            <th>Visits</th>
            <th>Visit Concept Name</th>
          </tr>
        </thead>
        <tbody>
          {variables.map(([varName, cohortId], i) => (
            <tr key={i}>
              <td className="font-mono text-xs">{varName}</td>
              <td className="text-xs">{cohortId}</td>
              <td className="text-xs">{visitsValue}</td>
              <td className="text-xs">{visitConceptName}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
