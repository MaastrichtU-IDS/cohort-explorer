'use client';

// Data Quality Dashboard (/data-quality): every known issue of a cohort, from
// the dictionary syntax check every upload goes through, the OMOP concept-code
// check, the dictionary-vs-data checks on the variable profiling, and the
// suspect values. Without a cohort chosen the page shows one card per cohort;
// with one (?cohort=…) it shows that cohort's issues (components/
// DataQualityIssues), its concept check, and whether its issues are shared on
// the explore page with the cohort's owners, and which types.

import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {Play, RefreshCw} from 'react-feather';
import {useRouter} from 'next/router';
import {apiUrl} from '@/utils';
import AdminOnly from '@/components/AdminOnly';
import {BLUE, CohortIssuesView, ConceptJob, Payload, fmtDay, fmtDayFirst, typeLabel} from '@/components/DataQualityIssues';

type Sharing = {enabled: boolean; types: string[]; updated_at?: string; updated_by?: string};

// Whether a cohort's issues show on its explore-page card (to the platform's
// admins and the cohort's owners), and which issue types. Saved at each change.
const SharingControl = ({cohort, data, sharing, onChange}: {
  cohort: string;
  data: Payload;
  sharing: Sharing;
  onChange: (next: Sharing) => void;
}) => {
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const save = async (patch: Partial<Sharing>) => {
    setSaving(true);
    setError(null);
    try {
      const r = await fetch(`${apiUrl}/api/data-quality/sharing/${encodeURIComponent(cohort)}`, {
        method: 'PUT', credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(patch),
      });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
      const d = await r.json();
      onChange({enabled: !!d.enabled, types: d.types || [], updated_at: d.updated_at, updated_by: d.updated_by});
    } catch (e: any) {
      setError(e?.message || 'Could not save');
    } finally {
      setSaving(false);
    }
  };

  return (
    <label className="flex items-center gap-2 cursor-pointer text-sm" title="Shows a data quality tab on this cohort's card on the explore page, to the platform admins and this cohort's owners">
      <input type="checkbox" className="checkbox checkbox-sm" checked={sharing.enabled} disabled={saving}
             onChange={e => save({enabled: e.target.checked})} />
      <span>Show on the explore page <span className="text-base-content/60">(admins and this cohort&apos;s owners)</span></span>
      {saving && <span className="loading loading-spinner loading-xs" />}
      {error && <span className="text-error text-xs">{error}</span>}
    </label>
  );
};

// The issue types shared on the explore page, by group, each with a checkbox.
const SharedTypesPanel = ({cohort, data, sharing, onChange}: {
  cohort: string;
  data: Payload;
  sharing: Sharing;
  onChange: (next: Sharing) => void;
}) => {
  const [error, setError] = useState<string | null>(null);
  const selected = new Set(sharing.types);

  const save = async (types: string[]) => {
    setError(null);
    const previous = sharing;
    onChange({...sharing, types});
    try {
      const r = await fetch(`${apiUrl}/api/data-quality/sharing/${encodeURIComponent(cohort)}`, {
        method: 'PUT', credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({types}),
      });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
      const d = await r.json();
      onChange({enabled: !!d.enabled, types: d.types || [], updated_at: d.updated_at, updated_by: d.updated_by});
    } catch (e: any) {
      onChange(previous);
      setError(e?.message || 'Could not save');
    }
  };

  const toggle = (key: string) => {
    const next = new Set(selected);
    if (next.has(key)) next.delete(key); else next.add(key);
    save(data.types.filter(t => next.has(t.key)).map(t => t.key));
  };

  const setGroup = (category: string, on: boolean) => {
    const next = new Set(selected);
    for (const t of data.types) if (t.category === category) { if (on) next.add(t.key); else next.delete(t.key); }
    save(data.types.filter(t => next.has(t.key)).map(t => t.key));
  };

  return (
    <div className="rounded-lg border border-base-300 bg-base-100 p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-3">
        <div className="text-sm font-semibold">Issue types shown on the explore page</div>
        <div className="text-xs text-base-content/60">
          {selected.size} of {data.types.length} types · saved as you change them
          {sharing.updated_at && ` · last changed ${fmtDayFirst(sharing.updated_at)}${sharing.updated_by ? ` by ${sharing.updated_by}` : ''}`}
          {error && <span className="text-error ml-2">{error}</span>}
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {data.categories.map(cat => {
          const types = data.types.filter(t => t.category === cat.key);
          return (
            <div key={cat.key}>
              <div className="flex items-baseline gap-2 mb-1">
                <span className="text-sm font-semibold">{cat.label}</span>
                <button className="link link-hover text-xs" onClick={() => setGroup(cat.key, true)}>all</button>
                <button className="link link-hover text-xs" onClick={() => setGroup(cat.key, false)}>none</button>
              </div>
              <div className="flex flex-col gap-0.5">
                {types.map(t => (
                  <label key={t.key} className="flex items-center gap-2 text-xs cursor-pointer" title={t.description}>
                    <input type="checkbox" className="checkbox checkbox-xs" checked={selected.has(t.key)}
                           onChange={() => toggle(t.key)} />
                    <span>{typeLabel(t)}</span>
                  </label>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

function DataQualityPage() {
  const router = useRouter();
  const [data, setData] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<ConceptJob | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [sharing, setSharing] = useState<Record<string, Sharing>>({});

  // The cohort lives in the URL, so a card opens it, the back button returns to
  // the cards, and a cohort's view can be linked to.
  const cohort = typeof router.query.cohort === 'string' ? router.query.cohort : '';
  const setCohort = (next: string) =>
    router.push({pathname: router.pathname, query: next ? {cohort: next} : {}}, undefined, {shallow: true});

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${apiUrl}/api/data-quality/issues`, {credentials: 'include'});
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
      const payload: Payload = await r.json();
      setData(payload);
      setJob(payload.sources.concepts.job);
      const s: Record<string, Sharing> = {};
      for (const [c, v] of Object.entries(payload.sharing || {})) {
        s[c] = {enabled: !!v.enabled, types: v.types || payload.types.map(t => t.key),
                updated_at: v.updated_at, updated_by: v.updated_by};
      }
      setSharing(s);
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
      const r = await fetch(`${apiUrl}/api/data-quality/concept-check?cohort=${encodeURIComponent(forCohort)}`,
                            {method: 'POST', credentials: 'include'});
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
      setJob(await r.json());
    } catch (e: any) {
      setJobError(e?.message || 'Could not start the concept check');
    }
  };

  const summaries = data?.cohort_summary || {};

  // The cards: every cohort with a dictionary or with issues.
  const cardCohorts = useMemo(
    () => (data?.cohorts || []).filter(c => summaries[c]?.dictionary_file || summaries[c]?.issues),
    [data, summaries]);

  const cohortSharing: Sharing = sharing[cohort] || {enabled: false, types: (data?.types || []).map(t => t.key)};
  const conceptStatus = cohort ? data?.sources.concepts.cohorts[cohort] : undefined;

  const picker = (
    <label className="form-control w-full max-w-xl mx-auto">
      <span className="label-text font-semibold mb-1">Cohort</span>
      <select className="select select-bordered select-lg w-full" value={cohort}
              onChange={e => setCohort(e.target.value)}>
        {!cohort && <option value="" disabled>Choose a cohort…</option>}
        {(data?.cohorts || []).filter(c => summaries[c]?.dictionary_file).map(c => {
          const n = summaries[c]?.variables_with_issues || 0;
          return <option key={c} value={c}>{c} ({n} {n === 1 ? 'variable' : 'variables'} with issues)</option>;
        })}
      </select>
    </label>
  );

  return (
    <div className="container mx-auto p-6 max-w-[90rem]">
      <div className="relative mb-4">
        <h1 className="text-3xl font-bold text-center">Data Quality Dashboard</h1>
        <div className="absolute right-0 top-1/2 -translate-y-1/2">
          <button className="btn btn-ghost btn-sm" onClick={load} title="Refresh">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error mb-4 text-sm">{error}</div>}

      {loading && !data ? (
        <div className="flex items-center justify-center gap-3 text-base-content/70 py-8">
          <span className="loading loading-spinner loading-md" /> <span>Gathering the issues…</span>
        </div>
      ) : !data ? null : !cohort ? (
        // --- landing: the picker and one card per cohort ---
        <>
          <div className="mb-6">{picker}</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 max-w-6xl mx-auto">
            {cardCohorts.map(c => {
              const s = summaries[c];
              return (
                <button key={c} onClick={() => setCohort(c)} style={BLUE}
                        className="rounded-xl border p-4 text-left shadow-sm hover:shadow-md hover:brightness-[0.97] transition">
                  <div className="text-lg font-bold leading-tight mb-2 break-words">{c}</div>
                  <div className="text-sm">
                    <span className="text-2xl font-semibold">{s.variables_with_issues}</span> variables with issues
                  </div>
                  <div className="text-sm opacity-80">
                    of {s.variables_total ?? '—'} variables in the dictionary
                  </div>
                  <div className="text-xs opacity-70 mt-2">
                    {s.dictionary_uploaded_at ? `Dictionary updated ${fmtDay(s.dictionary_uploaded_at)}` : 'No dictionary on file'}
                  </div>
                  {sharing[c]?.enabled && (
                    <div className="text-xs mt-1 font-medium">Shown on the explore page</div>
                  )}
                </button>
              );
            })}
          </div>
          <div className="flex flex-wrap items-center justify-center gap-3 mt-8 text-sm text-base-content/70">
            Concept codes:{' '}
            {job?.running
              ? `checking ${job.current || '…'} (${job.done} of ${job.total} cohorts)`
              : `checked for ${Object.values(data.sources.concepts.cohorts || {}).filter(c => c.checked_at).length} of ${data.cohorts.length} cohorts`}
            {job?.running && <span className="loading loading-spinner loading-sm" />}
          </div>
        </>
      ) : (
        // --- one cohort: the shared view, with the dashboard's own controls in its slots ---
        <CohortIssuesView
          key={cohort}
          data={data}
          cohort={cohort}
          picker={picker}
          topRight={
            <SharingControl cohort={cohort} data={data} sharing={cohortSharing}
                            onChange={next => setSharing(prev => ({...prev, [cohort]: next}))} />
          }
          belowTop={cohortSharing.enabled ? (
            <SharedTypesPanel cohort={cohort} data={data} sharing={cohortSharing}
                              onChange={next => setSharing(prev => ({...prev, [cohort]: next}))} />
          ) : null}
          afterCard={
            <div className="flex flex-wrap items-center gap-3 mb-4 text-sm">
              <span className="text-base-content/70">
                Concept codes:{' '}
                {job?.running
                  ? `checking ${job.current || '…'} (${job.done} of ${job.total} cohorts)`
                  : conceptStatus?.checked_at
                    ? <>checked {fmtDayFirst(conceptStatus.checked_at)}{conceptStatus.stale && <span className="badge badge-warning badge-sm ml-2">dictionary changed since</span>}</>
                    : 'not checked yet for this cohort'}
              </span>
              {job?.running ? <span className="loading loading-spinner loading-sm" /> : (
                <>
                  <button className="btn btn-outline btn-xs gap-1" onClick={() => runConceptCheck(cohort)} title="A minute or two for one cohort">
                    <Play size={12} /> Check {cohort} now
                  </button>
                  <span className="text-xs text-base-content/50">Runs in the background.</span>
                </>
              )}
              {jobError && <span className="text-error text-xs">{jobError}</span>}
              {job?.error && !job.running && <span className="text-error text-xs">Last run stopped: {job.error}</span>}
            </div>
          }
        />
      )}
    </div>
  );
}

export default function DataQualityPageAdminOnly() {
  return <AdminOnly><DataQualityPage /></AdminOnly>;
}
