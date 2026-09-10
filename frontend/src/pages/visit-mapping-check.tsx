'use client';

import React, { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useCohorts } from '@/components/CohortsContext';
import { apiUrl } from '@/utils';
import LoginPrompt from '@/components/LoginPrompt';
import { AlertTriangle, CheckCircle, Activity, HelpCircle, Hash, BookOpen } from 'react-feather';

type VarPair = [string, string]; // [variable name, cohort id]
type Severity = 'real' | 'trivial';

interface SuspectVariable {
  visit_concept_name: string;
  variable_count: number;
  variables: VarPair[];
}

interface SuspectMapping {
  visits_value: string;
  severity?: Severity;
  cohort_count?: number;
  majority: {
    visit_concept_name: string;
    variable_count: number;
    variables: VarPair[];
  };
  minorities: SuspectVariable[];
  total_variables: number;
  distinct_concept_names: string[];
}

interface ConsistentMapping {
  visits_value: string;
  visit_concept_name: string;
  variable_count: number;
  cohort_count: number;
  variables: VarPair[];
}

interface UnmappedMapping {
  visits_value: string;
  variable_count: number;
  cohort_count: number;
  variables: VarPair[];
}

interface ConflictValue {
  value: string;
  variable_count: number;
  cohort_count: number;
  variables: VarPair[];
}

interface CodeConflict {
  kind: 'name_to_codes' | 'name_to_omop_ids' | 'code_to_names';
  key: string;
  // 'trivial' when one code sits under names that differ only by spelling
  severity?: Severity;
  values: ConflictValue[];
}

interface ConceptVocabulary {
  visit_concept_name: string;
  normalized: string;
  variable_count: number;
  cohort_count: number;
  codes: string[];
  omop_ids: string[];
  raw_values: { value: string; variable_count: number; cohort_count: number }[];
}

interface VisitMappingResult {
  total_visits_values: number;
  suspect_count: number;
  suspect_real_count?: number;
  suspect_trivial_count?: number;
  suspect_mappings: SuspectMapping[];
  consistent_count?: number;
  consistent_mappings?: ConsistentMapping[];
  unmapped_count?: number;
  unmapped_mappings?: UnmappedMapping[];
  codes_available?: boolean;
  code_conflict_count?: number;
  code_conflicts?: CodeConflict[];
  concept_vocabulary?: ConceptVocabulary[];
  near_duplicate_groups?: string[][];
}

type View = 'suspect' | 'consistent' | 'unmapped' | 'codes' | 'concepts';
type SeverityFilter = 'all' | 'real' | 'trivial';

const CONFLICT_LABELS: Record<CodeConflict['kind'], { title: string; valueHeader: string }> = {
  name_to_codes: { title: 'Concept name carrying several visit concept codes', valueHeader: 'Visit concept code' },
  name_to_omop_ids: { title: 'Concept name carrying several visit OMOP IDs', valueHeader: 'Visit OMOP ID' },
  code_to_names: { title: 'Visit concept code used under several concept names', valueHeader: 'Visit concept name' },
};

// Does a mapping mention the filter text in its visit value, concept name or
// any of its variables/cohorts?
function matchesFilter(needle: string, texts: string[], pairs: VarPair[]): boolean {
  if (!needle) return true;
  const n = needle.toLowerCase();
  return (
    texts.some(t => (t || '').toLowerCase().includes(n)) ||
    pairs.some(([v, c]) => v.toLowerCase().includes(n) || c.toLowerCase().includes(n))
  );
}

export default function VisitMappingCheckPage() {
  const { userEmail } = useCohorts();
  const [data, setData] = useState<VisitMappingResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>('suspect');
  const [severity, setSeverity] = useState<SeverityFilter>('all');
  const [filter, setFilter] = useState('');

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

  const consistent = useMemo(
    () => (data?.consistent_mappings || []).filter(m => matchesFilter(filter, [m.visits_value, m.visit_concept_name], m.variables)),
    [data, filter]
  );
  const suspects = useMemo(
    () =>
      (data?.suspect_mappings || [])
        .filter(m => severity === 'all' || (m.severity || 'real') === severity)
        .filter(m =>
          matchesFilter(
            filter,
            [m.visits_value, ...m.distinct_concept_names],
            [...m.majority.variables, ...m.minorities.flatMap(x => x.variables)]
          )
        ),
    [data, filter, severity]
  );
  const unmapped = useMemo(
    () => (data?.unmapped_mappings || []).filter(m => matchesFilter(filter, [m.visits_value], m.variables)),
    [data, filter]
  );
  const conflicts = useMemo(
    () =>
      (data?.code_conflicts || []).filter(c =>
        matchesFilter(filter, [c.key, ...c.values.map(v => v.value)], c.values.flatMap(v => v.variables))
      ),
    [data, filter]
  );
  const vocabulary = useMemo(
    () =>
      (data?.concept_vocabulary || []).filter(e =>
        matchesFilter(filter, [e.visit_concept_name, ...e.raw_values.map(r => r.value), ...e.codes, ...e.omop_ids], [])
      ),
    [data, filter]
  );
  // concept name -> the other names that differ from it only by spelling
  const nearDuplicateOf = useMemo(() => {
    const map: Record<string, string[]> = {};
    (data?.near_duplicate_groups || []).forEach(group =>
      group.forEach(name => {
        map[name] = group.filter(n => n !== name);
      })
    );
    return map;
  }, [data]);

  if (userEmail === null) {
    return <LoginPrompt message="Authenticate to access this page" />;
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

  const realCount = data.suspect_real_count ?? data.suspect_count;
  const trivialCount = data.suspect_trivial_count ?? 0;
  const consistentCount = data.consistent_count ?? data.consistent_mappings?.length ?? 0;
  const unmappedCount = data.unmapped_count ?? data.unmapped_mappings?.length ?? 0;
  const conflictCount = data.code_conflict_count ?? data.code_conflicts?.length ?? 0;
  const conceptCount = data.concept_vocabulary?.length ?? 0;
  const nearDupCount = (data.near_duplicate_groups || []).length;

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
            Detects inconsistencies in how visit values are mapped to visit concept names and codes across cohorts.
          </p>
        </div>

        {/* Summary cards */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-6">
          <div className="stat bg-base-200 rounded-lg border border-base-300 p-3">
            <div className="stat-title text-xs">Distinct visit values</div>
            <div className="stat-value text-primary text-2xl">{data.total_visits_values}</div>
          </div>
          <div className="stat bg-base-200 rounded-lg border border-base-300 p-3">
            <div className="stat-title text-xs">Real conflicts</div>
            <div className={`stat-value text-2xl ${realCount > 0 ? 'text-error' : 'text-success'}`}>{realCount}</div>
          </div>
          <div className="stat bg-base-200 rounded-lg border border-base-300 p-3">
            <div className="stat-title text-xs">Spelling-only</div>
            <div className={`stat-value text-2xl ${trivialCount > 0 ? 'text-warning' : 'text-success'}`}>{trivialCount}</div>
          </div>
          <div className="stat bg-base-200 rounded-lg border border-base-300 p-3">
            <div className="stat-title text-xs">Code conflicts</div>
            <div className={`stat-value text-2xl ${conflictCount > 0 ? 'text-error' : 'text-success'}`}>{conflictCount}</div>
          </div>
          <div className="stat bg-base-200 rounded-lg border border-base-300 p-3">
            <div className="stat-title text-xs">Unmapped</div>
            <div className={`stat-value text-2xl ${unmappedCount > 0 ? 'text-base-content/60' : 'text-success'}`}>{unmappedCount}</div>
          </div>
        </div>

        {/* View switch + filter */}
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <div className="join flex-wrap">
            <button
              className={`btn btn-sm join-item gap-1 ${view === 'suspect' ? 'btn-warning' : 'btn-outline'}`}
              onClick={() => setView('suspect')}
            >
              <AlertTriangle size={14} /> Suspect
              <span className="badge badge-sm">{data.suspect_count}</span>
            </button>
            <button
              className={`btn btn-sm join-item gap-1 ${view === 'codes' ? 'btn-error' : 'btn-outline'}`}
              onClick={() => setView('codes')}
              title="Concept names carrying several codes / OMOP ids, or one code used under several names"
            >
              <Hash size={14} /> Codes
              <span className="badge badge-sm">{conflictCount}</span>
            </button>
            <button
              className={`btn btn-sm join-item gap-1 ${view === 'consistent' ? 'btn-success' : 'btn-outline'}`}
              onClick={() => setView('consistent')}
            >
              <CheckCircle size={14} /> Consistent
              <span className="badge badge-sm">{consistentCount}</span>
            </button>
            <button
              className={`btn btn-sm join-item gap-1 ${view === 'unmapped' ? 'btn-neutral' : 'btn-outline'}`}
              onClick={() => setView('unmapped')}
              title="Visit values used by variables that have no visit concept name at all"
            >
              <HelpCircle size={14} /> Unmapped
              <span className="badge badge-sm">{unmappedCount}</span>
            </button>
            <button
              className={`btn btn-sm join-item gap-1 ${view === 'concepts' ? 'btn-info' : 'btn-outline'}`}
              onClick={() => setView('concepts')}
              title="Informational: which raw visit values feed each visit concept name"
            >
              <BookOpen size={14} /> By concept
              <span className="badge badge-sm">{conceptCount}</span>
            </button>
          </div>
          <input
            type="text"
            className="input input-bordered input-sm w-72"
            placeholder="Filter by visit value, concept, code, variable or cohort…"
            value={filter}
            onChange={e => setFilter(e.target.value)}
          />
          {filter && (
            <button className="btn btn-ghost btn-xs" onClick={() => setFilter('')}>
              clear
            </button>
          )}
        </div>

        {/* Suspect view */}
        {view === 'suspect' && (
          <>
            <div className="flex flex-wrap items-center gap-2 mb-3 text-xs">
              <span className="text-base-content/60">Severity:</span>
              {(['all', 'real', 'trivial'] as SeverityFilter[]).map(s => (
                <button
                  key={s}
                  className={`btn btn-xs ${severity === s ? (s === 'real' ? 'btn-error' : s === 'trivial' ? 'btn-warning' : 'btn-primary') : 'btn-outline'}`}
                  onClick={() => setSeverity(s)}
                >
                  {s === 'all' ? `All (${data.suspect_count})` : s === 'real' ? `Real conflicts (${realCount})` : `Spelling-only (${trivialCount})`}
                </button>
              ))}
              <span className="text-base-content/50 ml-2">
                Spelling-only = the competing concept names differ just by case, spacing or punctuation.
              </span>
            </div>
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
                    {suspects.length} visit {suspects.length === 1 ? 'value' : 'values'} mapped to multiple concept names
                    {suspects.length !== data.suspect_count && (
                      <span className="text-base-content/50 font-normal"> (of {data.suspect_count}, filtered)</span>
                    )}
                  </span>
                </div>
                {suspects.map((sm, idx) => (
                  <SuspectCard key={idx} suspect={sm} />
                ))}
              </div>
            )}
          </>
        )}

        {/* Codes view */}
        {view === 'codes' && (
          <div className="space-y-4">
            {!data.codes_available ? (
              <div className="flex items-center gap-3 p-6 bg-base-200 rounded-lg border border-base-300">
                <HelpCircle size={28} className="text-base-content/50" />
                <div>
                  <h3 className="font-semibold">No visit concept codes in the catalog data</h3>
                  <p className="text-sm text-base-content/60">
                    None of the loaded variables carries a VISIT CONCEPT CODE or VISIT OMOP ID, so the name/code
                    consistency cannot be checked. Codes are read from the data dictionaries when the cache is
                    built from the source files.
                  </p>
                </div>
              </div>
            ) : conflictCount === 0 ? (
              <div className="flex items-center gap-3 p-6 bg-success/10 rounded-lg border border-success/20">
                <CheckCircle size={28} className="text-success" />
                <div>
                  <h3 className="font-semibold text-lg">Visit concept names and codes agree</h3>
                  <p className="text-sm text-base-content/60">
                    Every concept name carries a single code / OMOP ID, and every code is used under a single name.
                  </p>
                </div>
              </div>
            ) : (
              <>
                <div className="flex items-center gap-2 text-error">
                  <Hash size={20} />
                  <span className="font-semibold">
                    {conflicts.length} code {conflicts.length === 1 ? 'conflict' : 'conflicts'}
                    {conflicts.length !== conflictCount && (
                      <span className="text-base-content/50 font-normal"> (of {conflictCount}, filtered)</span>
                    )}
                  </span>
                </div>
                {conflicts.map((c, i) => (
                  <CodeConflictCard key={i} conflict={c} />
                ))}
              </>
            )}
          </div>
        )}

        {/* Consistent view */}
        {view === 'consistent' && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-success mb-2">
              <CheckCircle size={20} />
              <span className="font-semibold">
                {consistent.length} visit {consistent.length === 1 ? 'value maps' : 'values map'} to exactly one concept name
                {filter && consistent.length !== consistentCount && (
                  <span className="text-base-content/50 font-normal"> (of {consistentCount}, filtered)</span>
                )}
              </span>
            </div>
            {consistent.length === 0 ? (
              <p className="text-sm text-base-content/50">Nothing matches.</p>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-base-300">
                <table className="table table-sm">
                  <thead>
                    <tr>
                      <th>Visit value</th>
                      <th>Visit concept name</th>
                      <th className="text-right">Variables</th>
                      <th className="text-right">Cohorts</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {consistent.map(m => (
                      <ConsistentRow key={m.visits_value} mapping={m} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Unmapped view */}
        {view === 'unmapped' && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-base-content/70 mb-2">
              <HelpCircle size={20} />
              <span className="font-semibold">
                {unmapped.length} visit {unmapped.length === 1 ? 'value' : 'values'} used without any visit concept name
              </span>
            </div>
            {unmapped.length === 0 ? (
              <p className="text-sm text-base-content/50">Every visit value has a concept name.</p>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-base-300">
                <table className="table table-sm">
                  <thead>
                    <tr>
                      <th>Visit value</th>
                      <th className="text-right">Variables</th>
                      <th className="text-right">Cohorts</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {unmapped.map(m => (
                      <UnmappedRow key={m.visits_value} mapping={m} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* By concept view (informational) */}
        {view === 'concepts' && (
          <div className="space-y-3">
            <div className="p-3 rounded-lg bg-info/10 border border-info/20 text-sm">
              <span className="font-semibold">Informational.</span> For each visit concept name, the raw visit values that
              map to it. Many raw spellings feeding one concept is the normal, intended outcome of mapping - not an error.
              {nearDupCount > 0 ? (
                <>
                  {' '}The one thing worth a look: <span className="font-semibold">{nearDupCount}</span>{' '}
                  {nearDupCount === 1 ? 'group' : 'groups'} of concept names differ only by spelling (highlighted below) -
                  those are probably meant to be one concept.
                </>
              ) : (
                <> No two concept names differ only by spelling.</>
              )}
            </div>
            {vocabulary.length === 0 ? (
              <p className="text-sm text-base-content/50">Nothing matches.</p>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-base-300">
                <table className="table table-sm">
                  <thead>
                    <tr>
                      <th>Visit concept name</th>
                      <th>Code / OMOP ID</th>
                      <th className="text-right">Variables</th>
                      <th className="text-right">Cohorts</th>
                      <th>Raw visit values feeding it</th>
                    </tr>
                  </thead>
                  <tbody>
                    {vocabulary.map(e => {
                      const dups = nearDuplicateOf[e.visit_concept_name] || [];
                      return (
                        <tr key={e.visit_concept_name} className={dups.length > 0 ? 'bg-warning/10' : ''}>
                          <td className="text-sm align-top">
                            {e.visit_concept_name}
                            {dups.length > 0 && (
                              <div className="text-xs text-warning mt-0.5" title="These concept names differ only by spelling">
                                <AlertTriangle size={11} className="inline mr-1" />
                                spelling variant of: {dups.join(', ')}
                              </div>
                            )}
                          </td>
                          <td className="text-xs font-mono align-top">
                            {[...e.codes, ...e.omop_ids].join(', ') || <span className="text-base-content/40">—</span>}
                          </td>
                          <td className="text-right font-mono text-xs align-top">{e.variable_count}</td>
                          <td className="text-right font-mono text-xs align-top">{e.cohort_count}</td>
                          <td className="text-xs align-top">
                            <div className="flex flex-wrap gap-1">
                              {e.raw_values.map(r => (
                                <span
                                  key={r.value}
                                  className="badge badge-sm badge-ghost font-mono"
                                  title={`${r.variable_count} variables in ${r.cohort_count} cohort(s)`}
                                >
                                  {r.value} <span className="ml-1 text-base-content/50">{r.variable_count}</span>
                                </span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function CohortLink({ cohortId }: { cohortId: string }) {
  return (
    <Link
      href={{ pathname: '/cohorts', query: { cohort: cohortId } }}
      className="text-purple-700 dark:text-purple-400 underline underline-offset-2 hover:text-purple-900"
      title={`Open ${cohortId} on the explore page`}
    >
      {cohortId}
    </Link>
  );
}

function SeverityTag({ severity }: { severity?: Severity }) {
  if (severity === 'trivial') {
    return (
      <span
        className="badge badge-sm badge-warning"
        title="The competing concept names differ only by case, spacing or punctuation"
      >
        spelling-only
      </span>
    );
  }
  return (
    <span className="badge badge-sm badge-error" title="The competing concept names are genuinely different concepts">
      real conflict
    </span>
  );
}

function ConsistentRow({ mapping }: { mapping: ConsistentMapping }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr className="hover cursor-pointer" onClick={() => setOpen(o => !o)}>
        <td className="font-mono text-xs">{mapping.visits_value}</td>
        <td className="text-sm">{mapping.visit_concept_name}</td>
        <td className="text-right font-mono text-xs">{mapping.variable_count}</td>
        <td className="text-right font-mono text-xs">{mapping.cohort_count}</td>
        <td className="text-right text-xs text-base-content/50">{open ? '▲' : '▼'}</td>
      </tr>
      {open && (
        <tr>
          <td colSpan={5} className="bg-base-100">
            <VariableList variables={mapping.variables} visitsValue={mapping.visits_value} visitConceptName={mapping.visit_concept_name} />
          </td>
        </tr>
      )}
    </>
  );
}

function UnmappedRow({ mapping }: { mapping: UnmappedMapping }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr className="hover cursor-pointer" onClick={() => setOpen(o => !o)}>
        <td className="font-mono text-xs">{mapping.visits_value}</td>
        <td className="text-right font-mono text-xs">{mapping.variable_count}</td>
        <td className="text-right font-mono text-xs">{mapping.cohort_count}</td>
        <td className="text-right text-xs text-base-content/50">{open ? '▲' : '▼'}</td>
      </tr>
      {open && (
        <tr>
          <td colSpan={4} className="bg-base-100">
            <VariableList variables={mapping.variables} visitsValue={mapping.visits_value} visitConceptName="—" />
          </td>
        </tr>
      )}
    </>
  );
}

function CodeConflictCard({ conflict }: { conflict: CodeConflict }) {
  const [expanded, setExpanded] = useState(false);
  const meta = CONFLICT_LABELS[conflict.kind];
  const total = conflict.values.reduce((s, v) => s + v.variable_count, 0);
  const trivial = conflict.severity === 'trivial';
  return (
    <div className={`card bg-base-200 shadow-sm border ${trivial ? 'border-base-300' : 'border-error/30'}`}>
      <div className="card-body p-4">
        <div className="flex items-center justify-between gap-4 cursor-pointer" onClick={() => setExpanded(p => !p)}>
          <div className="flex-1 min-w-0">
            <div className={`text-xs uppercase tracking-wide font-semibold ${trivial ? 'text-warning' : 'text-error'}`}>{meta.title}</div>
            <div className="flex flex-wrap items-center gap-2 mt-0.5">
              <h3 className="font-semibold text-lg truncate" title={conflict.key}>
                {conflict.key}
              </h3>
              <SeverityTag severity={conflict.severity} />
            </div>
            <div className="text-xs mt-1 text-base-content/70">
              {conflict.values.map((v, i) => (
                <span key={v.value}>
                  {i > 0 && ' · '}
                  <span className={i === 0 ? 'text-success font-semibold' : 'text-error font-semibold'}>{v.value}</span>
                  {' '}({v.variable_count} var{v.variable_count === 1 ? '' : 's'}, {v.cohort_count} cohort{v.cohort_count === 1 ? '' : 's'})
                </span>
              ))}
              <span className="text-base-content/40"> · {total} variables in total; the first is the majority</span>
            </div>
          </div>
          <button className="btn btn-ghost btn-sm">{expanded ? '▲ Collapse' : '▼ Expand'}</button>
        </div>
        {expanded && (
          <div className="mt-3 space-y-3" onClick={e => e.stopPropagation()}>
            {conflict.values.map((v, i) => (
              <div key={v.value} className={`rounded-lg p-3 border ${i === 0 ? 'bg-success/10 border-success/20' : 'bg-error/10 border-error/20'}`}>
                <div className="font-medium text-sm mb-2">
                  {meta.valueHeader}: <span className="font-mono">{v.value}</span> ({v.variable_count} variables)
                </div>
                <div className="overflow-x-auto">
                  <table className="table table-xs">
                    <thead>
                      <tr>
                        <th>Variable</th>
                        <th>Cohort</th>
                      </tr>
                    </thead>
                    <tbody>
                      {v.variables.map(([varName, cohortId], k) => (
                        <tr key={k}>
                          <td className="font-mono text-xs">{varName}</td>
                          <td className="text-xs"><CohortLink cohortId={cohortId} /></td>
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

function SuspectCard({ suspect }: { suspect: SuspectMapping }) {
  const [expanded, setExpanded] = useState(false);
  const trivial = suspect.severity === 'trivial';

  return (
    <div className={`card bg-base-200 shadow-sm border ${trivial ? 'border-base-300' : 'border-error/30'}`}>
      <div className="card-body p-4">
        {/* Header */}
        <div
          className="flex items-center justify-between gap-4 cursor-pointer"
          onClick={() => setExpanded(prev => !prev)}
        >
          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="font-semibold text-lg truncate" title={suspect.visits_value}>
                {suspect.visits_value}
              </h3>
              <SeverityTag severity={suspect.severity} />
            </div>
            <div className="flex flex-wrap gap-2 mt-1">
              <span className="badge badge-sm badge-secondary">
                {suspect.total_variables} variables
              </span>
              {suspect.cohort_count !== undefined && (
                <span className="badge badge-sm badge-primary">
                  {suspect.cohort_count} {suspect.cohort_count === 1 ? 'cohort' : 'cohorts'}
                </span>
              )}
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
        <div className={`mt-2 rounded-lg p-3 text-sm border ${trivial ? 'bg-warning/10 border-warning/20' : 'bg-error/10 border-error/20'}`}>
          <div className="flex items-start gap-2">
            <AlertTriangle size={18} className={`flex-shrink-0 mt-0.5 ${trivial ? 'text-warning' : 'text-error'}`} />
            <div className="space-y-1">
              <p className="font-semibold">
                {trivial ? 'Spelling variants' : 'Inconsistency'} detected for visits value: <span className="font-mono">{suspect.visits_value}</span>
              </p>
              <p className="text-base-content/70">
                This visits value is mapped to <span className="font-semibold">{suspect.distinct_concept_names.length} different visit concept names</span> across {suspect.total_variables} variables:
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
                {trivial
                  ? <>These names differ only by case, spacing or punctuation: a bulk rename to <span className="font-semibold">{suspect.majority.visit_concept_name}</span> resolves it.</>
                  : <>The majority mapping is assumed correct. Variables in the suspect mapping(s) should be reviewed and remapped to <span className="font-semibold">{suspect.majority.visit_concept_name}</span>.</>}
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

function VariableList({ variables, visitsValue, visitConceptName }: { variables: VarPair[]; visitsValue: string; visitConceptName: string }) {
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
              <td className="text-xs"><CohortLink cohortId={cohortId} /></td>
              <td className="text-xs">{visitsValue}</td>
              <td className="text-xs">{visitConceptName}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
