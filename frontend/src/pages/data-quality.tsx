'use client';

// Data quality (/data-quality): every known issue of a cohort in one list, from
// three sources: the dictionary syntax check every upload goes through, the
// OMOP concept-code check, and the suspect values found in the variable
// profiling. Pick a cohort at the top; its issue types appear as tags, each
// with a checkbox that shows or hides that type. The links to the individual
// checks are on the data quality hub (/data-quality-hub).

import React, {useCallback, useEffect, useMemo, useState} from 'react';
import Link from 'next/link';
import {AlertTriangle, Download, Play, RefreshCw, Search} from 'react-feather';
import {apiUrl} from '@/utils';
import AdminOnly from '@/components/AdminOnly';
import VariableGraphModal from '@/components/VariableGraphModal';

interface Issue {
  cohort: string;
  category: string;
  type: string;
  variable?: string | null;
  row?: number | null;
  message: string;
  reasons?: string[];
  level?: string | null;
  level_rank?: number | null;
  suspect_value?: number | null;
  median?: number | null;
  minimum?: number | null;
  maximum?: number | null;
  declared_missing?: string[];
  label?: string | null;
  units?: string | null;
  visits?: string | null;
  omop_id?: string | number | null;
  concept_code?: string | null;
  stale?: boolean;
}

interface IssueType {
  key: string;
  category: string;
  label: string;
  description: string;
  // Id in the data quality specification (S1 to S14), where the type has one.
  spec?: string | null;
}

const typeLabel = (t?: IssueType, fallback = '') =>
  t ? (t.spec ? `${t.label} (${t.spec})` : t.label) : fallback;

interface Category {
  key: string;
  label: string;
  description: string;
}

interface ConceptJob {
  running: boolean;
  started_at?: string | null;
  finished_at?: string | null;
  done: number;
  total: number;
  current?: string | null;
  error?: string | null;
}

interface Payload {
  generated_at: string;
  cohorts: string[];
  cohorts_with_profiling: string[];
  categories: Category[];
  types: IssueType[];
  issues: Issue[];
  sources: {
    syntax: {checked_at?: string | null};
    concepts: {cohorts: Record<string, {checked_at?: string | null; stale?: boolean; status?: string}>; job: ConceptJob};
    values: {read_at?: string | null};
  };
}

const CATEGORY_BADGE: Record<string, string> = {
  syntax: 'badge-secondary',
  concepts: 'badge-accent',
  values: 'badge-primary',
};

// What the severity column says for issues that have no suspicion level.
const CATEGORY_SEVERITY: Record<string, {label: string; badge: string; title: string}> = {
  syntax: {label: 'Blocks upload', badge: 'badge-error badge-outline', title: 'A dictionary with this issue is rejected on upload'},
  concepts: {label: 'Fails check', badge: 'badge-warning badge-outline', title: 'The concept does not match OMOP'},
};

const LEVEL_BADGE: Record<string, string> = {
  'very high': 'badge-error',
  high: 'badge-warning',
  moderate: 'badge-info',
  low: 'badge-ghost',
};

const capitalise = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const num = (v?: number | null): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
};

const fmtDate = (iso?: string | null): string => {
  if (!iso) return '';
  const d = new Date(iso.replace(' ', 'T'));
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

type Arrange = 'type' | 'variable';

function DataQualityPage() {
  const [data, setData] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [cohort, setCohort] = useState('');
  // Types the reader has unticked. Kept as the hidden set, so a type that first
  // appears after a refresh is shown.
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState('');
  const [arrange, setArrange] = useState<Arrange>('type');
  const [chartOf, setChartOf] = useState<Issue | null>(null);

  const [job, setJob] = useState<ConceptJob | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${apiUrl}/api/data-quality/issues`, {credentials: 'include'});
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
      const payload: Payload = await r.json();
      setData(payload);
      setJob(payload.sources.concepts.job);
    } catch (e: any) {
      setError(e?.message || 'Could not load the data issues');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // While a concept check runs, follow it, and reload the issues when it ends.
  useEffect(() => {
    if (!job?.running) return;
    const timer = setInterval(async () => {
      try {
        const r = await fetch(`${apiUrl}/api/data-quality/concept-check/status`, {credentials: 'include'});
        if (!r.ok) return;
        const next: ConceptJob = await r.json();
        setJob(next);
        if (!next.running) load();
      } catch {
        // a missed poll is retried on the next tick
      }
    }, 4000);
    return () => clearInterval(timer);
  }, [job?.running, load]);

  const runConceptCheck = async (forCohort: string) => {
    setJobError(null);
    try {
      const qs = forCohort ? `?cohort=${encodeURIComponent(forCohort)}` : '';
      const r = await fetch(`${apiUrl}/api/data-quality/concept-check${qs}`, {method: 'POST', credentials: 'include'});
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
      setJob(await r.json());
    } catch (e: any) {
      setJobError(e?.message || 'Could not start the concept check');
    }
  };

  const typeByKey = useMemo(() => {
    const m: Record<string, IssueType> = {};
    for (const t of data?.types || []) m[t.key] = t;
    return m;
  }, [data]);

  const typeOrder = useMemo(() => {
    const m: Record<string, number> = {};
    (data?.types || []).forEach((t, i) => { m[t.key] = i; });
    return m;
  }, [data]);

  const byCohort = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const i of data?.issues || []) counts[i.cohort] = (counts[i.cohort] || 0) + 1;
    return counts;
  }, [data]);

  const cohortIssues = useMemo(() => {
    const items = data?.issues || [];
    return cohort ? items.filter(i => i.cohort === cohort) : items;
  }, [data, cohort]);

  // The tags: only the types this cohort actually has, with their counts.
  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const i of cohortIssues) counts[i.type] = (counts[i.type] || 0) + 1;
    return counts;
  }, [cohortIssues]);

  const visible = useMemo(() => {
    let items = cohortIssues.filter(i => !hiddenTypes.has(i.type));
    const q = query.trim().toLowerCase();
    if (q) {
      items = items.filter(i =>
        (i.variable || '').toLowerCase().includes(q) ||
        (i.label || '').toLowerCase().includes(q) ||
        i.message.toLowerCase().includes(q)
      );
    }
    if (arrange === 'variable') {
      items = [...items].sort((a, b) =>
        a.cohort.localeCompare(b.cohort) ||
        (a.variable || '￿').toLowerCase().localeCompare((b.variable || '￿').toLowerCase()) ||
        (typeOrder[a.type] ?? 0) - (typeOrder[b.type] ?? 0));
    }
    return items;
  }, [cohortIssues, hiddenTypes, query, arrange, typeOrder]);

  const toggleType = (key: string) => setHiddenTypes(prev => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  const setCategoryShown = (category: string, shown: boolean) => setHiddenTypes(prev => {
    const next = new Set(prev);
    for (const t of data?.types || []) {
      if (t.category !== category || !typeCounts[t.key]) continue;
      if (shown) next.delete(t.key); else next.add(t.key);
    }
    return next;
  });

  const profiled = useMemo(() => new Set(data?.cohorts_with_profiling || []), [data]);

  const conceptStatus = cohort ? data?.sources.concepts.cohorts[cohort] : undefined;
  const conceptCheckedCount = useMemo(
    () => Object.values(data?.sources.concepts.cohorts || {}).filter(c => c.checked_at).length,
    [data]);

  const downloadCsv = () => {
    const header = ['cohort', 'category', 'type', 'variable', 'dictionary_row', 'visit', 'label', 'severity', 'message'];
    const esc = (v: any) => `"${String(v ?? '').replace(/"/g, '""')}"`;
    const lines = [header.join(',')].concat(visible.map(i => [
      i.cohort, i.category, typeLabel(typeByKey[i.type], i.type), i.variable, i.row, i.visits, i.label,
      i.level || CATEGORY_SEVERITY[i.category]?.label, i.message,
    ].map(esc).join(',')));
    const blob = new Blob([lines.join('\n')], {type: 'text/csv'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `data-issues-${cohort || 'all-cohorts'}-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="container mx-auto p-6 max-w-[90rem]">
      <div className="flex items-center gap-3 mb-2">
        <AlertTriangle size={28} />
        <h1 className="text-3xl font-bold">Data quality</h1>
        <Link href="/data-quality-hub" className="btn btn-ghost btn-sm ml-auto">Data quality hub</Link>
        <button className="btn btn-ghost btn-sm" onClick={load} title="Refresh">
          <RefreshCw size={16} />
        </button>
        <button className="btn btn-ghost btn-sm" onClick={downloadCsv} disabled={!visible.length} title="Download the list as CSV">
          <Download size={16} />
        </button>
      </div>
      <p className="text-sm text-base-content/70 mb-5 max-w-3xl">
        Every known issue of a cohort in one list: mistakes in how the data dictionary is written,
        concept codes that do not match OMOP, and values in the data that are probably not real
        measurements.
      </p>

      <div className="card bg-base-200 mb-5">
        <div className="card-body p-5 gap-4">
          <label className="form-control w-full max-w-xl mx-auto">
            <span className="label-text font-semibold mb-1">Cohort</span>
            <select className="select select-bordered select-lg w-full" value={cohort}
                    onChange={e => setCohort(e.target.value)}>
              <option value="">All cohorts</option>
              {(data?.cohorts || []).map(c => (
                <option key={c} value={c}>{c}{byCohort[c] ? ` (${byCohort[c]})` : ' (no issues)'}</option>
              ))}
            </select>
          </label>

          {data && (
            <div className="flex flex-col gap-3">
              {data.categories.map(cat => {
                const types = data.types.filter(t => t.category === cat.key && typeCounts[t.key]);
                if (!types.length) return null;
                const allShown = types.every(t => !hiddenTypes.has(t.key));
                return (
                  <div key={cat.key} className="flex flex-wrap items-center justify-center gap-2">
                    <button className="text-sm font-semibold mr-1 hover:underline" title={`${cat.description} Click to ${allShown ? 'hide' : 'show'} all.`}
                            onClick={() => setCategoryShown(cat.key, !allShown)}>
                      {cat.label}
                    </button>
                    {types.map(t => {
                      const on = !hiddenTypes.has(t.key);
                      return (
                        <label key={t.key} title={t.description}
                               className={`flex items-center gap-2 rounded-full border px-3 py-1 cursor-pointer text-sm transition-opacity ${
                                 on ? 'bg-base-100 border-base-300' : 'border-transparent opacity-50'}`}>
                          <input type="checkbox" className="checkbox checkbox-xs" checked={on}
                                 onChange={() => toggleType(t.key)} />
                          <span>{typeLabel(t)}</span>
                          <span className={`badge badge-sm ${CATEGORY_BADGE[cat.key]}`}>{typeCounts[t.key]}</span>
                        </label>
                      );
                    })}
                  </div>
                );
              })}
              {cohort && !cohortIssues.length && !loading && (
                <div className="text-center text-sm text-base-content/70">No known issues for {cohort}.</div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Concept codes are checked on request: the check queries Athena for every concept. */}
      {data && (
        <div className="flex flex-wrap items-center gap-3 mb-4 text-sm">
          <span className="text-base-content/70">
            Concept codes:{' '}
            {job?.running
              ? `checking ${job.current || '…'} (${job.done} of ${job.total} cohorts)`
              : cohort
                ? conceptStatus?.checked_at
                  ? <>checked {fmtDate(conceptStatus.checked_at)}{conceptStatus.stale && <span className="badge badge-warning badge-sm ml-2">dictionary changed since</span>}</>
                  : 'not checked yet for this cohort'
                : `checked for ${conceptCheckedCount} of ${data.cohorts.length} cohorts`}
          </span>
          {job?.running
            ? <span className="loading loading-spinner loading-sm" />
            : (
              <>
                {cohort && (
                  <button className="btn btn-outline btn-xs gap-1" onClick={() => runConceptCheck(cohort)}
                          title="A minute or two for one cohort">
                    <Play size={12} /> Check {cohort} now
                  </button>
                )}
                <button className="btn btn-outline btn-xs gap-1" onClick={() => runConceptCheck('')}
                        title="Takes a while across all cohorts">
                  <Play size={12} /> Check all cohorts now
                </button>
              </>
            )}
          {!job?.running && (
            <span className="text-xs text-base-content/50">Runs in the background.</span>
          )}
          {jobError && <span className="text-error text-xs">{jobError}</span>}
          {job?.error && !job.running && <span className="text-error text-xs">Last run stopped: {job.error}</span>}
        </div>
      )}

      <div className="flex flex-wrap items-end gap-3 mb-4">
        <label className="form-control flex-1 min-w-60">
          <span className="label-text text-xs">Search</span>
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 opacity-50" />
            <input value={query} onChange={e => setQuery(e.target.value)}
                   placeholder="variable, label or message"
                   className="input input-bordered input-sm w-full pl-8" />
          </div>
        </label>
        <label className="form-control">
          <span className="label-text text-xs">Arrange by</span>
          <select className="select select-bordered select-sm w-44" value={arrange}
                  onChange={e => setArrange(e.target.value as Arrange)}>
            <option value="type">Type of issue</option>
            <option value="variable">Variable</option>
          </select>
        </label>
      </div>

      {error && <div className="alert alert-error mb-4 text-sm">{error}</div>}

      {data && !loading && (
        <p className="text-xs text-base-content/60 mb-2">
          Showing {visible.length} of {cohortIssues.length} issues{cohort ? ` for ${cohort}` : ' across all cohorts'}.
          {data.sources.syntax.checked_at && ` Dictionaries checked ${fmtDate(data.sources.syntax.checked_at)}.`}
          {data.sources.values.read_at && ` Profiling read ${fmtDate(data.sources.values.read_at)}.`}
        </p>
      )}

      {loading ? (
        <div className="flex items-center gap-3 text-base-content/70 py-8">
          <span className="loading loading-spinner loading-md" /> <span>Gathering the issues…</span>
        </div>
      ) : visible.length === 0 ? (
        data && cohortIssues.length > 0 && <div className="alert">Nothing to show with these filters.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="table table-sm table-zebra">
            <thead>
              <tr>
                <th>Issue</th>
                {!cohort && <th>Cohort</th>}
                <th>Variable</th>
                <th>What is wrong</th>
                <th>Severity</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((i, idx) => {
                const t = typeByKey[i.type];
                const severity = CATEGORY_SEVERITY[i.category];
                return (
                  <tr key={`${i.cohort}/${i.type}/${i.variable || ''}/${i.row || ''}/${idx}`}>
                    <td className="whitespace-nowrap align-top">
                      <span className={`badge badge-sm ${CATEGORY_BADGE[i.category]}`} title={t?.description}>
                        {typeLabel(t, i.type)}
                      </span>
                    </td>
                    {!cohort && <td className="whitespace-nowrap align-top">{i.cohort}</td>}
                    <td className="max-w-72 align-top">
                      {i.variable ? (
                        <>
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="font-mono text-xs">{i.variable}</span>
                            {i.visits && <span className="badge badge-ghost badge-xs whitespace-nowrap">{i.visits}</span>}
                            {i.row && <span className="text-xs text-base-content/50">row {i.row}</span>}
                          </div>
                          {i.label && <div className="text-xs text-base-content/60 truncate" title={i.label}>{i.label}</div>}
                          {i.units && <div className="text-xs text-base-content/40">{i.units}</div>}
                          {profiled.has(i.cohort) && (
                            <div className="mt-1">
                              <button
                                className="btn-sm rounded-lg bg-teal-50 border border-teal-200 hover:bg-teal-100 dark:bg-teal-900/30 dark:border-teal-700 inline-flex items-center gap-1.5 px-2"
                                title="Summary statistics — click to view"
                                onClick={() => setChartOf(i)}
                              >
                                <span className="text-xl leading-none">📊</span>
                                <span className="text-xs font-medium text-teal-800 dark:text-teal-300">summary statistics</span>
                              </button>
                            </div>
                          )}
                        </>
                      ) : (
                        <span className="text-xs text-base-content/50">{i.row ? `row ${i.row}` : 'whole dictionary'}</span>
                      )}
                    </td>
                    <td className="max-w-2xl align-top text-xs">
                      {i.reasons && i.reasons.length ? (
                        <>
                          {i.suspect_value !== null && i.suspect_value !== undefined && (
                            <div className="mb-1">
                              Suspect value <span className="font-mono font-semibold">{num(i.suspect_value)}</span>
                              <span className="text-base-content/50">
                                {' '}· median {num(i.median)} · range {num(i.minimum)} to {num(i.maximum)}
                                {i.declared_missing && i.declared_missing.length > 0 && ` · declared missing ${i.declared_missing.join(' | ')}`}
                              </span>
                            </div>
                          )}
                          <ul className="list-disc list-inside space-y-0.5">
                            {i.reasons.map((r, k) => <li key={k}>{r}</li>)}
                          </ul>
                        </>
                      ) : (
                        <span className="whitespace-pre-wrap">{i.message}</span>
                      )}
                      {i.stale && (
                        <div className="mt-1">
                          <span className="badge badge-warning badge-xs">dictionary changed since this was checked</span>
                        </div>
                      )}
                    </td>
                    <td className="whitespace-nowrap align-top">
                      {i.level ? (
                        <span className={`badge badge-sm ${LEVEL_BADGE[i.level] || 'badge-ghost'}`} title="Suspicion that the value is not a real measurement">
                          {capitalise(i.level)}
                        </span>
                      ) : severity ? (
                        <span className={`badge badge-sm ${severity.badge}`} title={severity.title}>{severity.label}</span>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {chartOf && chartOf.variable && (
        <VariableGraphModal
          isOpen={true}
          cohortId={chartOf.cohort}
          variableName={chartOf.variable}
          variableLabel={chartOf.label || undefined}
          omopId={chartOf.omop_id !== null && chartOf.omop_id !== undefined ? String(chartOf.omop_id) : undefined}
          conceptCode={chartOf.concept_code || undefined}
          onClose={() => setChartOf(null)}
        />
      )}
    </div>
  );
}

export default function DataQualityPageAdminOnly() {
  return <AdminOnly><DataQualityPage /></AdminOnly>;
}
