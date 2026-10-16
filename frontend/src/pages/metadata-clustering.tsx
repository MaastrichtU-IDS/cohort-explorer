'use client';

import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {Layers, Play, RefreshCw} from 'react-feather';
import {apiUrl} from '@/utils';
import {MatchList, MetadataElement, MetadataMatch, SourceBadge} from '@/components/MetadataMatches';

interface FieldGroup {
  group: string;
  fields: [string, string][];
}

interface JobState {
  running?: boolean;
  phase?: string;
  started?: string;
  finished?: string;
  error?: string | null;
  started_by?: string;
  instructions?: string;
  elements?: number;
  cohorts?: number;
  lexical_matches?: number;
  ai_done?: number;
  ai_failed?: number;
  ai_available?: boolean;
  model?: string;
  has_results?: boolean;
  is_admin?: boolean;
  fields?: FieldGroup[];
  counts?: Record<string, number>;
  duration_seconds?: number;
}

interface Results {
  started: string;
  finished: string;
  duration_seconds: number;
  model: string | null;
  ai_available: boolean;
  instructions: string;
  fields: FieldGroup[];
  counts: Record<string, number>;
  ai_failed: Record<string, string>;
  elements: MetadataElement[];
  matches: Record<string, MetadataMatch[]>;
}

type View = 'study' | 'field' | 'ai';

export default function MetadataClusteringPage() {
  const [job, setJob] = useState<JobState | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [instructions, setInstructions] = useState('');
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>('study');
  const [cohort, setCohort] = useState<string>('');
  const [field, setField] = useState<string>('');
  const [aiOnly, setAiOnly] = useState(false);

  const loadStatus = useCallback(async () => {
    try {
      const r = await fetch(`${apiUrl}/api/metadata-clustering/status`, {credentials: 'include'});
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const s: JobState = await r.json();
      setJob(s);
      return s;
    } catch (e: any) {
      setError(e.message);
      return null;
    }
  }, []);

  const loadResults = useCallback(async () => {
    const r = await fetch(`${apiUrl}/api/metadata-clustering/results`, {credentials: 'include'});
    if (r.status === 404) { setResults(null); return; }
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data: Results = await r.json();
    setResults(data);
    if (data.instructions && !instructions) setInstructions(data.instructions);
  }, [instructions]);

  // Initial load, then poll while a run is in progress.
  // Status needs only the session cookie: fetch at once rather than waiting for
  // the cohorts context (its user email arrives with the full cohorts payload).
  useEffect(() => {
    let timer: any;
    const tick = async () => {
      const s = await loadStatus();
      if (s?.running) timer = setTimeout(tick, 4000);
      else if (s?.has_results) loadResults().catch(e => setError(e.message));
    };
    tick();
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.running]);

  const startRun = async () => {
    setStarting(true);
    setError(null);
    try {
      const r = await fetch(`${apiUrl}/api/metadata-clustering/run`, {
        method: 'POST',
        credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({instructions}),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${r.status}`);
      }
      await loadStatus();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setStarting(false);
    }
  };

  const cohorts = useMemo(
    () => Array.from(new Set((results?.elements || []).map(e => e.cohort))).sort((a, b) => a.localeCompare(b)),
    [results]
  );
  const fieldGroups: FieldGroup[] = results?.fields || job?.fields || [];
  const fieldLabel = useMemo(() => {
    const m: Record<string, string> = {};
    fieldGroups.forEach(g => g.fields.forEach(([f, l]) => (m[f] = l)));
    return m;
  }, [fieldGroups]);

  useEffect(() => {
    if (cohorts.length && !cohort) setCohort(cohorts[0]);
    if (fieldGroups.length && !field) setField(fieldGroups[0].fields[0][0]);
  }, [cohorts, fieldGroups, cohort, field]);

  const elementsOf = (c: string) => (results?.elements || []).filter(e => e.cohort === c);
  const matchesOf = (id: string) => results?.matches[id] || [];

  // Pairs listed once (not twice) for the field and AI-only views.
  const pairsFor = (filter: (e: MetadataElement, m: MetadataMatch) => boolean) => {
    const seen = new Set<string>();
    const out: {a: MetadataElement; m: MetadataMatch}[] = [];
    (results?.elements || []).forEach(a => {
      matchesOf(a.id).forEach(m => {
        if (!filter(a, m)) return;
        const key = [a.id, m.element_id].sort().join('|');
        if (seen.has(key)) return;
        seen.add(key);
        out.push({a, m});
      });
    });
    return out;
  };

  if (job && job.is_admin === false) {
    return (
      <div className="container mx-auto p-6">
        <div className="alert alert-warning">This page is for administrators only.</div>
      </div>
    );
  }
  if (!job && !error) {
    return (
      <div className="container mx-auto p-6 flex items-center gap-3 text-base-content/70">
        <span className="loading loading-spinner loading-md" /> <span>Loading…</span>
      </div>
    );
  }

  const aiCount = results?.counts?.ai ?? 0;

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="flex items-center gap-3 mb-2">
        <Layers size={28} />
        <h1 className="text-3xl font-bold">Metadata clustering</h1>
      </div>
      <p className="text-base-content/70 mb-6">
        For every study-level metadata element (objective, population, outcomes, inclusion and exclusion criteria)
        the closest matching elements in the other studies. Matches come from a lexical pass (text similarity)
        and an AI pass; matches the AI found that the lexical pass missed are tagged <SourceBadge source="ai" />.
        Matching is symmetric: if A matches B, B matches A.
      </p>

      {/* Run controls */}
      <div className="card bg-base-200 shadow mb-6">
        <div className="card-body">
          <h2 className="card-title text-lg">Run the matching</h2>
          <label className="form-control">
            <span className="label-text mb-1">
              Additional AI instructions <span className="text-base-content/50">(optional)</span>
            </span>
            <textarea
              className="textarea textarea-bordered h-24 font-mono text-sm"
              placeholder={'e.g. Treat "heart failure with preserved ejection fraction" and "HFpEF" as the same population. Ignore differences in age thresholds of 5 years or less.'}
              value={instructions}
              onChange={e => setInstructions(e.target.value)}
              disabled={!!job?.running}
            />
          </label>
          <div className="flex flex-wrap items-center gap-3 mt-3">
            <button className="btn btn-primary" onClick={startRun} disabled={starting || !!job?.running}>
              {job?.running ? <span className="loading loading-spinner loading-sm" /> : <Play size={16} />}
              {job?.running ? 'Running…' : results ? 'Run again' : 'Run'}
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => loadStatus()} title="Refresh status">
              <RefreshCw size={14} />
            </button>
            {job?.ai_available === false && (
              <span className="text-warning text-sm">
                LITELLM_BASE_URL is not configured: only the lexical pass will run.
              </span>
            )}
            {job?.model && job?.ai_available !== false && (
              <span className="text-sm text-base-content/60">AI model: {job.model}</span>
            )}
          </div>

          {job?.running && (
            <div className="mt-3 text-sm">
              <div className="flex items-center gap-2">
                <span className="loading loading-dots loading-xs" />
                <span>
                  Phase: <strong>{job.phase}</strong>
                  {job.elements ? ` · ${job.elements} elements from ${job.cohorts} studies` : ''}
                  {job.lexical_matches !== undefined ? ` · ${job.lexical_matches} lexical pairs` : ''}
                  {job.phase === 'ai' && job.elements ? ` · AI ${job.ai_done ?? 0}/${job.elements}` : ''}
                  {job.ai_failed ? ` (${job.ai_failed} failed)` : ''}
                </span>
              </div>
              {job.phase === 'ai' && job.elements ? (
                <progress className="progress progress-primary w-full mt-2" value={job.ai_done ?? 0} max={job.elements} />
              ) : null}
              <p className="text-xs text-base-content/50 mt-1">Started {job.started ? new Date(job.started).toLocaleString() : ''} by {job.started_by}</p>
            </div>
          )}
          {job?.phase === 'failed' && job.error && (
            <div className="alert alert-error mt-3 text-sm">Last run failed: {job.error}</div>
          )}
          {error && <div className="alert alert-error mt-3 text-sm">{error}</div>}
        </div>
      </div>

      {/* Results */}
      {!results && !job?.running && (
        <div className="alert">No results yet. Run the matching to review matches.</div>
      )}

      {results && (
        <>
          <div className="stats stats-vertical sm:stats-horizontal shadow mb-4 w-full">
            <div className="stat py-3">
              <div className="stat-title">Studies</div>
              <div className="stat-value text-2xl">{results.counts.cohorts}</div>
            </div>
            <div className="stat py-3">
              <div className="stat-title">Elements</div>
              <div className="stat-value text-2xl">{results.counts.elements}</div>
            </div>
            <div className="stat py-3">
              <div className="stat-title">Matched pairs</div>
              <div className="stat-value text-2xl">{results.counts.pairs}</div>
              <div className="stat-desc">
                {results.counts.both} both · {results.counts.lexical} lexical only · {results.counts.ai} AI only
              </div>
            </div>
            <div className="stat py-3">
              <div className="stat-title">Last run</div>
              <div className="stat-value text-lg">{new Date(results.finished).toLocaleString()}</div>
              <div className="stat-desc">
                {results.duration_seconds}s · {results.model || 'no AI'}
                {results.counts.ai_failed ? ` · ${results.counts.ai_failed} AI calls failed` : ''}
              </div>
            </div>
          </div>
          {results.instructions && (
            <p className="text-xs text-base-content/60 mb-4">
              <span className="font-medium">AI instructions used:</span> {results.instructions}
            </p>
          )}

          <div className="flex flex-wrap items-center gap-3 mb-4">
            <div role="tablist" className="tabs tabs-boxed">
              <button role="tab" className={`tab ${view === 'study' ? 'tab-active' : ''}`} onClick={() => setView('study')}>By study</button>
              <button role="tab" className={`tab ${view === 'field' ? 'tab-active' : ''}`} onClick={() => setView('field')}>By field</button>
              <button role="tab" className={`tab ${view === 'ai' ? 'tab-active' : ''}`} onClick={() => setView('ai')}>
                AI-only matches <span className="badge badge-secondary badge-sm ml-1">{aiCount}</span>
              </button>
            </div>
            {view === 'study' && (
              <select className="select select-bordered select-sm" value={cohort} onChange={e => setCohort(e.target.value)}>
                {cohorts.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            )}
            {view === 'field' && (
              <select className="select select-bordered select-sm" value={field} onChange={e => setField(e.target.value)}>
                {fieldGroups.map(g => (
                  <optgroup key={g.group} label={g.group}>
                    {g.fields.map(([f, l]) => <option key={f} value={f}>{l}</option>)}
                  </optgroup>
                ))}
              </select>
            )}
            {view !== 'ai' && (
              <label className="label cursor-pointer gap-2">
                <input type="checkbox" className="checkbox checkbox-sm checkbox-secondary" checked={aiOnly} onChange={e => setAiOnly(e.target.checked)} />
                <span className="label-text">AI-only matches</span>
              </label>
            )}
          </div>

          {view === 'study' && cohort && (
            <div className="space-y-4">
              {elementsOf(cohort).length === 0 && <p className="italic text-base-content/60">This study has no metadata elements to compare.</p>}
              {elementsOf(cohort).map(e => (
                <div key={e.id} className="card bg-base-100 border border-base-300">
                  <div className="card-body p-4">
                    <div className="flex items-baseline gap-2">
                      <h3 className="font-semibold">{e.label}</h3>
                      <span className="text-xs text-base-content/50">{e.group}</span>
                      <span className="badge badge-sm ml-auto">{matchesOf(e.id).filter(m => !aiOnly || m.source === 'ai').length}</span>
                    </div>
                    <p className="text-sm whitespace-pre-wrap bg-base-200 rounded p-2 mb-2">{e.text}</p>
                    <MatchList matches={matchesOf(e.id)} aiOnly={aiOnly} emptyText="No matches in other studies." />
                  </div>
                </div>
              ))}
            </div>
          )}

          {view === 'field' && field && (
            <PairTable
              title={fieldLabel[field] || field}
              pairs={pairsFor((a, m) => (a.field === field || m.field === field) && (!aiOnly || m.source === 'ai'))}
            />
          )}

          {view === 'ai' && (
            <PairTable
              title="Matches found by the AI pass only"
              pairs={pairsFor((_a, m) => m.source === 'ai')}
            />
          )}

          {Object.keys(results.ai_failed || {}).length > 0 && (
            <details className="mt-6 text-sm">
              <summary className="cursor-pointer text-warning">
                {Object.keys(results.ai_failed).length} elements got no AI verdict (call failed)
              </summary>
              <ul className="list-disc ml-6 mt-2 text-xs">
                {Object.entries(results.ai_failed).map(([id, err]) => <li key={id}><code>{id}</code>: {err}</li>)}
              </ul>
            </details>
          )}
        </>
      )}
    </div>
  );
}

function PairTable({title, pairs}: {title: string; pairs: {a: MetadataElement; m: MetadataMatch}[]}) {
  if (pairs.length === 0) return <p className="italic text-base-content/60">No pairs for {title}.</p>;
  return (
    <div>
      <h3 className="font-semibold mb-2">{title} <span className="badge badge-sm">{pairs.length} pairs</span></h3>
      <div className="overflow-x-auto">
        <table className="table table-sm">
          <thead>
            <tr>
              <th className="w-1/2">Study A</th>
              <th className="w-1/2">Study B</th>
              <th>How</th>
            </tr>
          </thead>
          <tbody>
            {pairs.map(({a, m}) => (
              <tr key={`${a.id}|${m.element_id}`} className="align-top">
                <td>
                  <div className="font-semibold">{a.cohort} <span className="text-xs font-normal text-base-content/60">{a.label}</span></div>
                  <div className="text-sm whitespace-pre-wrap">{a.text}</div>
                </td>
                <td>
                  <div className="font-semibold">{m.cohort} <span className={`text-xs font-normal ${m.same_field ? 'text-base-content/60' : 'text-warning'}`}>{m.label}</span></div>
                  <div className="text-sm whitespace-pre-wrap">{m.text}</div>
                </td>
                <td className="min-w-[10rem]">
                  <SourceBadge source={m.source} strength={m.ai_strength} />
                  <div className="text-xs text-base-content/50 mt-1">lexical {Math.round(m.score * 100)}%</div>
                  {m.ai_reason && <div className="text-xs mt-1">{m.ai_reason}</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
