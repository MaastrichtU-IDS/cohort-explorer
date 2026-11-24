'use client';

import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {AlertTriangle, BarChart2, Download, RefreshCw, Search} from 'react-feather';
import {apiUrl} from '@/utils';
import LoginPrompt from '@/components/LoginPrompt';
import VariableGraphModal from '@/components/VariableGraphModal';

interface SuspectRow {
  cohort: string;
  variable: string;
  label?: string | null;
  units?: string | null;
  concept_name?: string | null;
  domain?: string | null;
  eda_version: string;
  n?: number | null;
  n_unique?: number | null;
  mean?: number | null;
  median?: number | null;
  q1?: number | null;
  q3?: number | null;
  iqr?: number | null;
  minimum?: number | null;
  maximum?: number | null;
  std?: number | null;
  trimmed_mean?: number | null;
  declared_missing: string[];
  omop_id?: string | number | null;
  concept_code?: string | null;
  visits?: string | null;
  // Same value for the variables of one cohort that are the same measure at
  // different visits (shared concept and units); "variable:<name>" otherwise.
  measure: string;
  suspect_value?: number | null;
  likely_cause: string;
  score: number;
  reasons: string[];
}

interface Payload {
  generated_at?: string | null;
  min_score: number;
  cohorts_scanned: number;
  variables_scanned: number;
  n_flagged: number;
  cohorts: string[];
  by_cohort: Record<string, number>;
  items: SuspectRow[];
}

const num = (v?: number | null, digits = 2): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  return Number.isInteger(v) ? String(v) : v.toFixed(digits);
};

const CAUSES = ['undeclared missing code', 'negative minimum', 'extreme outlier'] as const;

interface MeasureGroup {
  key: string;
  // Highest-ranked first: the order the backend sent them in.
  members: SuspectRow[];
}

const visitOf = (r: SuspectRow) => (r.visits || '').trim();

// One variable. The first row of a measure carries the fold toggle and names
// the visits folded under it; the rows it unfolds are drawn indented.
const SuspectTableRow = ({row: r, nested, others, isOpen, onToggle, onChart}: {
  row: SuspectRow;
  nested: boolean;
  others: SuspectRow[];
  isOpen: boolean;
  onToggle: () => void;
  onChart: () => void;
}) => (
  <tr className={nested ? 'bg-base-200/60' : undefined}>
    <td>
      <span className={`badge badge-sm ${r.score >= 10 ? 'badge-error' : r.score >= 6 ? 'badge-warning' : 'badge-ghost'}`}>
        {r.score}
      </span>
    </td>
    <td className="whitespace-nowrap">{nested ? '' : r.cohort}</td>
    <td className={`max-w-72 ${nested ? 'pl-6' : ''}`}>
      <div className="flex items-center gap-1.5">
        <button className="btn btn-ghost btn-xs px-1" onClick={onChart} title="Show the distribution chart">
          <BarChart2 size={14} />
        </button>
        <span className="font-mono text-xs">{r.variable}</span>
        {visitOf(r) && <span className="badge badge-ghost badge-xs whitespace-nowrap">{visitOf(r)}</span>}
      </div>
      {r.label && <div className="text-xs text-base-content/60 truncate" title={r.label}>{r.label}</div>}
      {r.units && <div className="text-xs text-base-content/40">{r.units}</div>}
      {others.length > 0 && (
        <button className="btn btn-link btn-xs px-0 h-auto min-h-0 no-underline text-left" onClick={onToggle}
                title={others.map(o => `${o.variable}${visitOf(o) ? ` (${visitOf(o)})` : ''}`).join(', ')}>
          {isOpen
            ? 'hide the other visits'
            : `+ ${others.length} more ${others.length === 1 ? 'visit' : 'visits'} flagged`
              + (others.some(o => visitOf(o)) ? `: ${others.map(visitOf).filter(Boolean).join(', ')}` : '')}
        </button>
      )}
    </td>
    <td className="text-right font-mono">{num(r.suspect_value)}</td>
    <td className="whitespace-nowrap text-xs">{r.likely_cause}</td>
    <td className="text-right">{num(r.n, 0)}</td>
    <td className="text-right font-mono text-xs">{num(r.median)}</td>
    <td className="text-right font-mono text-xs">{num(r.mean)}</td>
    <td className="text-right font-mono text-xs">{num(r.minimum)}</td>
    <td className="text-right font-mono text-xs">{num(r.maximum)}</td>
    <td className="font-mono text-xs">{(r.declared_missing || []).join(' | ') || '—'}</td>
    <td className="max-w-96">
      <ul className="text-xs list-disc list-inside space-y-0.5">
        {(r.reasons || []).map((reason, i) => <li key={i}>{reason}</li>)}
      </ul>
    </td>
  </tr>
);

export default function SuspectValuesPage() {
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [needsLogin, setNeedsLogin] = useState(false);
  const [data, setData] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // The variable whose distribution chart is open, if any. Same modal the
  // variable cards on the explore page use, so the chart is the familiar one.
  const [chartOf, setChartOf] = useState<SuspectRow | null>(null);

  const [minScore, setMinScore] = useState(4);
  const [cohort, setCohort] = useState('');
  const [cause, setCause] = useState('');
  const [query, setQuery] = useState('');
  // One row per measure, its other visits folded under it. On by default:
  // without it a cohort lists the same lab value five or six times.
  const [groupVisits, setGroupVisits] = useState(true);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetch(`${apiUrl}/admin/check`, {credentials: 'include'})
      .then(r => {
        if (r.status === 401) {
          setNeedsLogin(true);
          return {is_admin: false};
        }
        return r.ok ? r.json() : {is_admin: false};
      })
      .then(d => setIsAdmin(!!d.is_admin))
      .catch(() => setIsAdmin(false));
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({min_score: String(minScore)});
      if (cohort) params.set('cohort', cohort);
      const r = await fetch(`${apiUrl}/api/suspect-values?${params.toString()}`, {credentials: 'include'});
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
      setData(await r.json());
    } catch (e: any) {
      setError(e?.message || 'Could not load the suspect values');
    } finally {
      setLoading(false);
    }
  }, [minScore, cohort]);

  useEffect(() => {
    if (isAdmin) load();
  }, [isAdmin, load]);

  const rows = useMemo(() => {
    let items = data?.items || [];
    if (cause) items = items.filter(r => r.likely_cause === cause);
    const q = query.trim().toLowerCase();
    if (q) {
      items = items.filter(r =>
        r.variable.toLowerCase().includes(q) ||
        (r.label || '').toLowerCase().includes(q) ||
        (r.concept_name || '').toLowerCase().includes(q) ||
        r.cohort.toLowerCase().includes(q)
      );
    }
    return items;
  }, [data, cause, query]);

  const groups = useMemo<MeasureGroup[]>(() => {
    const byKey = new Map<string, MeasureGroup>();
    const order: MeasureGroup[] = [];
    for (const r of rows) {
      const key = groupVisits ? `${r.cohort}|${r.measure}` : `${r.cohort}|variable:${r.variable}`;
      let g = byKey.get(key);
      if (!g) {
        g = {key, members: []};
        byKey.set(key, g);
        order.push(g);
      }
      g.members.push(r);
    }
    return order;
  }, [rows, groupVisits]);

  const toggleExpanded = (key: string) => setExpanded(prev => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  const downloadCsv = () => {
    const header = ['cohort', 'measure', 'variable', 'visit', 'label', 'units', 'n', 'median', 'mean', 'min', 'max',
      'suspect_value', 'likely_cause', 'score', 'declared_missing_codes', 'reasons'];
    const esc = (v: any) => `"${String(v ?? '').replace(/"/g, '""')}"`;
    const lines = [header.join(',')].concat(rows.map(r => [
      r.cohort, r.measure, r.variable, r.visits, r.label, r.units, r.n, r.median, r.mean, r.minimum, r.maximum,
      r.suspect_value, r.likely_cause, r.score, (r.declared_missing || []).join(' | '),
      (r.reasons || []).join('; '),
    ].map(esc).join(',')));
    const blob = new Blob([lines.join('\n')], {type: 'text/csv'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `suspect-values-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  if (needsLogin) return <LoginPrompt message="Authenticate to access this page (administrators only)" />;
  if (isAdmin === false) {
    return <div className="container mx-auto p-6"><div className="alert alert-warning">This page is for administrators only.</div></div>;
  }
  if (isAdmin === null) {
    return (
      <div className="container mx-auto p-6 flex items-center gap-3 text-base-content/70">
        <span className="loading loading-spinner loading-md" /> <span>Loading…</span>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-[90rem]">
      <div className="flex items-center gap-3 mb-2">
        <AlertTriangle size={28} />
        <h1 className="text-3xl font-bold">Suspect values</h1>
        <button className="btn btn-ghost btn-sm ml-auto" onClick={load} title="Refresh">
          <RefreshCw size={16} />
        </button>
        <button className="btn btn-ghost btn-sm" onClick={downloadCsv} disabled={!rows.length} title="Download the table as CSV">
          <Download size={16} />
        </button>
      </div>

      <p className="text-sm text-base-content/70 mb-4 max-w-4xl">
        Numeric variables whose profiling statistics look distorted by one or a few values that are
        probably not real measurements. The common cause is a stand-in for missing that the data
        owner never declared, such as 999 in a column where 9999 is declared. Everything here is
        derived from the variable profiling output already on disk, so the list costs no enclave
        run and nothing patient-level is read. False positives are expected: each row shows the
        signals that fired, so judge it on those. A confirmed case is fixed by adding the code to
        the MISSING column of the cohort&apos;s data dictionary, which accepts several codes
        separated by a pipe.
      </p>

      <div className="flex flex-wrap items-end gap-3 mb-4">
        <label className="form-control">
          <span className="label-text text-xs">Minimum score</span>
          <input type="number" min={1} max={30} value={minScore}
                 onChange={e => setMinScore(Math.max(1, Number(e.target.value) || 1))}
                 className="input input-bordered input-sm w-28" />
        </label>
        <label className="form-control">
          <span className="label-text text-xs">Cohort</span>
          <select className="select select-bordered select-sm w-56" value={cohort}
                  onChange={e => setCohort(e.target.value)}>
            <option value="">All cohorts</option>
            {(data?.cohorts || []).map(c => (
              <option key={c} value={c}>{c}{data?.by_cohort?.[c] ? ` (${data.by_cohort[c]})` : ''}</option>
            ))}
          </select>
        </label>
        <label className="form-control">
          <span className="label-text text-xs">Likely cause</span>
          <select className="select select-bordered select-sm w-60" value={cause}
                  onChange={e => setCause(e.target.value)}>
            <option value="">Any cause</option>
            {CAUSES.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <label className="form-control flex-1 min-w-60">
          <span className="label-text text-xs">Search</span>
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 opacity-50" />
            <input value={query} onChange={e => setQuery(e.target.value)}
                   placeholder="variable, label or concept"
                   className="input input-bordered input-sm w-full pl-8" />
          </div>
        </label>
        <label className="label cursor-pointer gap-2 pb-1">
          <input type="checkbox" className="checkbox checkbox-sm" checked={groupVisits}
                 onChange={e => setGroupVisits(e.target.checked)} />
          <span className="label-text text-sm">Group visits of one measure</span>
        </label>
      </div>

      {error && <div className="alert alert-error mb-4 text-sm">{error}</div>}

      {data && !loading && (
        <p className="text-xs text-base-content/60 mb-2">
          {data.n_flagged} flagged at score {data.min_score} or above, out of {data.variables_scanned} profiled
          numeric variables across {data.cohorts_scanned} cohorts. Showing {rows.length}
          {groupVisits && groups.length !== rows.length && ` as ${groups.length} measures`}.
          {data.generated_at && ` Profiling read at ${new Date(data.generated_at).toLocaleString()}.`}
        </p>
      )}

      {loading ? (
        <div className="flex items-center gap-3 text-base-content/70 py-8">
          <span className="loading loading-spinner loading-md" /> <span>Reading the profiling output…</span>
        </div>
      ) : rows.length === 0 ? (
        <div className="alert">Nothing flagged at this score. Lower the minimum score to widen the net.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="table table-sm table-zebra">
            <thead>
              <tr>
                <th>Score</th>
                <th>Cohort</th>
                <th>Variable</th>
                <th className="text-right">Suspect</th>
                <th>Likely cause</th>
                <th className="text-right">n</th>
                <th className="text-right">Median</th>
                <th className="text-right">Mean</th>
                <th className="text-right">Min</th>
                <th className="text-right">Max</th>
                <th>Declared missing</th>
                <th>Why it is flagged</th>
              </tr>
            </thead>
            <tbody>
              {groups.map(g => {
                const isOpen = expanded.has(g.key);
                const shown = isOpen ? g.members : g.members.slice(0, 1);
                return shown.map((r, i) => (
                  <SuspectTableRow
                    key={`${r.cohort}/${r.variable}`}
                    row={r}
                    nested={i > 0}
                    others={i === 0 ? g.members.slice(1) : []}
                    isOpen={isOpen}
                    onToggle={() => toggleExpanded(g.key)}
                    onChart={() => setChartOf(r)}
                  />
                ));
              })}
            </tbody>
          </table>
        </div>
      )}

      {chartOf && (
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
