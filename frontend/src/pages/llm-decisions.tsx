'use client';

import React, {useCallback, useEffect, useState} from 'react';
import {Database, Download, RefreshCw, Search} from 'react-feather';
import {apiUrl} from '@/utils';
import LoginPrompt from '@/components/LoginPrompt';

interface Run {
  run_id: string;
  started: string;
  finished?: string | null;
  status?: string;
  source_study?: string;
  target_studies: string[];
  embed_model?: string;
  mapping_mode?: string;
  llm_model?: string;
  requested_by?: string | null;
  n_decisions: number;
  n_cached: number;
  n_failed: number;
  error?: string | null;
}

interface Decision {
  id: number;
  run_id: string | null;
  created_at: string;
  model: string;
  mapping_mode: string;
  source_study: string | null;
  target_study: string | null;
  pair_key: string;
  src_desc: string; src_concepts: string; src_cats: string; src_unit: string;
  tgt_desc: string; tgt_concepts: string; tgt_cats: string; tgt_unit: string;
  matched: number | null;
  ctx_type: string;
  confidence: number;
  status: string;
  reason: string | null;
  transform: string | null;
  transform_direction: string | null;
  cached: number;
  attempts: number | null;
  duration_ms: number | null;
  round_idx: number;
  error: string | null;
  // detail only
  evidence?: string | null;
  system_prompt?: string;
  prompt?: string;
  raw_response?: string | null;
  verdict_json?: string;
}

interface ConsistencyItem {
  pair_key: string;
  n: number;
  n_runs: number;
  n_models: number;
  n_status: number;
  statuses: string[];
  conflict: boolean;
  src_desc: string;
  tgt_desc: string;
  source_study: string;
  target_study: string;
  min_conf: number;
  max_conf: number;
  first_seen: string;
  last_seen: string;
}

type Tab = 'runs' | 'decisions' | 'consistency';
const STATUSES = ['COMPLETE', 'COMPATIBLE', 'PARTIAL', 'IMPOSSIBLE', 'PENDING', 'PARSE_ERROR'];
const PAGE = 100;

function statusClass(s: string) {
  switch (s) {
    case 'COMPLETE': return 'badge-success';
    case 'COMPATIBLE': return 'badge-info';
    case 'PARTIAL': return 'badge-warning';
    case 'IMPOSSIBLE': return 'badge-ghost';
    default: return 'badge-error';
  }
}

function StatusBadge({s}: {s: string}) {
  return <span className={`badge badge-sm ${statusClass(s)} whitespace-nowrap`}>{s}</span>;
}

function fmt(ts?: string | null) {
  return ts ? new Date(ts).toLocaleString() : '';
}

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${apiUrl}${path}`, {credentials: 'include'});
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

export default function LlmDecisionsPage() {
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [needsLogin, setNeedsLogin] = useState(false);
  // True until the first runs/stats load has completed (or failed).
  const [loadingOverview, setLoadingOverview] = useState(true);
  const [tab, setTab] = useState<Tab>('runs');
  const [stats, setStats] = useState<any>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState<string | null>(null);

  // decisions tab
  const [filters, setFilters] = useState({run_id: '', status: '', source: '', target: '', model: '', q: '', cached: ''});
  const [decisions, setDecisions] = useState<{total: number; items: Decision[]} | null>(null);
  const [offset, setOffset] = useState(0);
  const [loadingDecisions, setLoadingDecisions] = useState(false);
  const [detail, setDetail] = useState<Record<number, Decision | 'loading'>>({});

  // consistency tab
  const [onlyConflicts, setOnlyConflicts] = useState(true);
  const [consistency, setConsistency] = useState<{repeated_pairs: number; conflicting_pairs: number; items: ConsistencyItem[]} | null>(null);
  const [pairHistory, setPairHistory] = useState<Record<string, Decision[] | 'loading'>>({});

  // The admin check needs only the session cookie: do it at once, instead of
  // waiting for the cohorts context (whose user email arrives with the full
  // cohorts payload, several seconds later).
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

  const loadOverview = useCallback(async () => {
    setError(null);
    setLoadingOverview(true);
    try {
      const [s, r] = await Promise.all([getJson<any>('/api/llm-decisions/stats'), getJson<{items: Run[]}>('/api/llm-decisions/runs')]);
      setStats(s);
      setRuns(r.items);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoadingOverview(false);
    }
  }, []);

  const loadDecisions = useCallback(async (off = 0) => {
    setLoadingDecisions(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([k, v]) => { if (v) params.set(k, v); });
      params.set('limit', String(PAGE));
      params.set('offset', String(off));
      setDecisions(await getJson(`/api/llm-decisions/decisions?${params.toString()}`));
      setOffset(off);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoadingDecisions(false);
    }
  }, [filters]);

  const loadConsistency = useCallback(async () => {
    setError(null);
    try {
      setConsistency(await getJson(`/api/llm-decisions/consistency?only_conflicts=${onlyConflicts}`));
    } catch (e: any) {
      setError(e.message);
    }
  }, [onlyConflicts]);

  useEffect(() => { if (isAdmin) loadOverview(); }, [isAdmin, loadOverview]);
  useEffect(() => { if (isAdmin && tab === 'decisions') loadDecisions(0); }, [isAdmin, tab, filters.run_id, filters.status, filters.cached, loadDecisions]);
  useEffect(() => { if (isAdmin && tab === 'consistency') loadConsistency(); }, [isAdmin, tab, onlyConflicts, loadConsistency]);

  const openDetail = async (id: number) => {
    if (detail[id]) { setDetail(d => { const n = {...d}; delete n[id]; return n; }); return; }
    setDetail(d => ({...d, [id]: 'loading'}));
    try {
      const full = await getJson<Decision>(`/api/llm-decisions/decision/${id}`);
      setDetail(d => ({...d, [id]: full}));
    } catch (e: any) {
      setError(e.message);
      setDetail(d => { const n = {...d}; delete n[id]; return n; });
    }
  };

  const openPair = async (key: string) => {
    if (pairHistory[key]) { setPairHistory(p => { const n = {...p}; delete n[key]; return n; }); return; }
    setPairHistory(p => ({...p, [key]: 'loading'}));
    try {
      const r = await getJson<{items: Decision[]}>(`/api/llm-decisions/pair/${key}`);
      setPairHistory(p => ({...p, [key]: r.items}));
    } catch (e: any) {
      setError(e.message);
    }
  };

  const [exporting, setExporting] = useState<string | null>(null);
  // Fetch with the session cookie, then hand the file to the browser under the server's file name.
  const exportRun = async (runId: string) => {
    setExporting(runId);
    setError(null);
    try {
      const r = await fetch(`${apiUrl}/api/llm-decisions/runs/${encodeURIComponent(runId)}/export.csv`, {credentials: 'include'});
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || `Export failed (${r.status})`);
      }
      const disposition = r.headers.get('Content-Disposition') || '';
      const m = /filename="([^"]+)"/.exec(disposition);
      const name = r.headers.get('X-Export-Filename') || (m ? m[1] : `llm-decisions_${runId}.csv`);
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setExporting(null);
    }
  };

  const showRunDecisions = (runId: string) => {
    setFilters(f => ({...f, run_id: runId}));
    setTab('decisions');
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
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="flex items-center gap-3 mb-2">
        <Database size={28} />
        <h1 className="text-3xl font-bold">LLM mapping decisions</h1>
        <button className="btn btn-ghost btn-sm ml-auto" onClick={() => { loadOverview(); if (tab === 'decisions') loadDecisions(offset); if (tab === 'consistency') loadConsistency(); }} title="Refresh">
          <RefreshCw size={14} />
        </button>
      </div>
      <p className="text-base-content/70 mb-4">
        Every question the cross-study mapping asked the LLM about a (source variable, target variable) pair, with the full
        input (the pair as the model saw it, system prompt, prompt) and output (raw reply, parsed verdict, confidence).
        Recorded from the moment the store was deployed; earlier runs are not included.
      </p>

      {stats && (
        <div className="stats stats-vertical sm:stats-horizontal shadow mb-4 w-full">
          <div className="stat py-3"><div className="stat-title">Runs</div><div className="stat-value text-2xl">{stats.runs}</div></div>
          <div className="stat py-3"><div className="stat-title">Decisions</div><div className="stat-value text-2xl">{stats.decisions}</div><div className="stat-desc">{stats.cached} served from the response cache</div></div>
          <div className="stat py-3"><div className="stat-title">Distinct pairs</div><div className="stat-value text-2xl">{stats.distinct_pairs}</div></div>
          <div className="stat py-3">
            <div className="stat-title">Verdicts</div>
            <div className="flex flex-wrap gap-1 mt-1">
              {Object.entries(stats.by_status || {}).map(([s, n]) => (
                <span key={s} className={`badge badge-sm ${statusClass(s)}`}>{s} {String(n)}</span>
              ))}
            </div>
          </div>
        </div>
      )}
      {error && <div className="alert alert-error text-sm mb-4">{error}</div>}

      <div role="tablist" className="tabs tabs-boxed mb-4 w-fit">
        <button role="tab" className={`tab ${tab === 'runs' ? 'tab-active' : ''}`} onClick={() => setTab('runs')}>Runs</button>
        <button role="tab" className={`tab ${tab === 'decisions' ? 'tab-active' : ''}`} onClick={() => setTab('decisions')}>Decisions</button>
        <button role="tab" className={`tab ${tab === 'consistency' ? 'tab-active' : ''}`} onClick={() => setTab('consistency')}>Consistency across runs</button>
      </div>

      {/* ---------------- Runs ---------------- */}
      {tab === 'runs' && (
        <div className="overflow-x-auto">
          {loadingOverview && (
            <div className="flex items-center gap-3 py-6 text-base-content/70">
              <span className="loading loading-spinner loading-md" /> <span>Loading runs…</span>
            </div>
          )}
          {!loadingOverview && runs.length === 0 && <p className="italic text-base-content/60">No mapping runs recorded yet.</p>}
          {!loadingOverview && runs.length > 0 && (
            <table className="table table-sm">
              <thead><tr><th>Started</th><th>Source → targets</th><th>Model</th><th>Mode</th><th>Decisions</th><th>Cached</th><th>Failed</th><th>Status</th><th>By</th><th></th></tr></thead>
              <tbody>
                {runs.map(r => (
                  <tr key={r.run_id}>
                    <td className="whitespace-nowrap">{fmt(r.started)}<div className="text-xs text-base-content/50">{r.finished ? `${Math.round((new Date(r.finished).getTime() - new Date(r.started).getTime()) / 1000)}s` : ''}</div></td>
                    <td><span className="font-semibold">{r.source_study}</span> → {r.target_studies.join(', ')}</td>
                    <td className="text-xs">{r.llm_model}</td>
                    <td className="text-xs">{r.mapping_mode}</td>
                    <td>{r.n_decisions}</td>
                    <td>{r.n_cached}</td>
                    <td className={r.n_failed ? 'text-error' : ''}>{r.n_failed}</td>
                    <td><span className={`badge badge-sm ${r.status === 'done' ? 'badge-success' : r.status === 'running' ? 'badge-info' : r.status === 'crashed' ? 'badge-warning' : 'badge-error'}`} title={r.status === 'crashed' ? 'No completion was recorded within 3 hours of the start: the worker crashed or was restarted' : undefined}>{r.status}</span>{r.error && <div className="text-xs text-error">{r.error}</div>}</td>
                    <td className="text-xs">{r.requested_by || ''}</td>
                    <td className="whitespace-nowrap">
                      <button className="btn btn-xs" onClick={() => showRunDecisions(r.run_id)}>decisions</button>{' '}
                      <button className="btn btn-xs btn-outline gap-1" onClick={() => exportRun(r.run_id)} disabled={exporting === r.run_id || !r.n_decisions} title="Download every decision of this run, with full prompts and replies, as CSV">
                        {exporting === r.run_id ? <span className="loading loading-spinner loading-xs" /> : <Download size={12} />} CSV
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* ---------------- Decisions ---------------- */}
      {tab === 'decisions' && (
        <>
          <div className="flex flex-wrap gap-2 mb-3 items-end">
            <select className="select select-bordered select-sm" value={filters.run_id} onChange={e => setFilters(f => ({...f, run_id: e.target.value}))}>
              <option value="">All runs</option>
              {runs.map(r => <option key={r.run_id} value={r.run_id}>{fmt(r.started)} · {r.source_study} → {r.target_studies.join(', ')}</option>)}
            </select>
            <select className="select select-bordered select-sm" value={filters.status} onChange={e => setFilters(f => ({...f, status: e.target.value}))}>
              <option value="">All verdicts</option>
              {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <select className="select select-bordered select-sm" value={filters.cached} onChange={e => setFilters(f => ({...f, cached: e.target.value}))}>
              <option value="">Cached + live</option>
              <option value="false">Live calls only</option>
              <option value="true">Cache hits only</option>
            </select>
            <input className="input input-bordered input-sm w-32" placeholder="source study" value={filters.source} onChange={e => setFilters(f => ({...f, source: e.target.value}))} />
            <input className="input input-bordered input-sm w-32" placeholder="target study" value={filters.target} onChange={e => setFilters(f => ({...f, target: e.target.value}))} />
            <input className="input input-bordered input-sm w-64" placeholder="search descriptions, concepts, reason, pair key" value={filters.q} onChange={e => setFilters(f => ({...f, q: e.target.value}))} onKeyDown={e => { if (e.key === 'Enter') loadDecisions(0); }} />
            <button className="btn btn-sm btn-primary" onClick={() => loadDecisions(0)}><Search size={14} /> Search</button>
            {loadingDecisions && <span className="loading loading-spinner loading-sm" />}
          </div>

          {decisions && (
            <>
              <div className="text-sm text-base-content/60 mb-2">
                {decisions.total} decisions{decisions.total > PAGE ? `, showing ${offset + 1}–${Math.min(offset + PAGE, decisions.total)}` : ''}
              </div>
              <div className="overflow-x-auto">
                <table className="table table-sm">
                  <thead><tr><th>When</th><th>Pair</th><th>Source variable</th><th>Target variable</th><th>Verdict</th><th>Conf.</th><th>Reason</th><th></th></tr></thead>
                  <tbody>
                    {decisions.items.map(d => (
                      <React.Fragment key={d.id}>
                        <tr className="align-top hover">
                          <td className="whitespace-nowrap text-xs">{fmt(d.created_at)}<div className="text-base-content/50">{d.cached ? 'cache' : `${d.attempts ?? '?'} call${d.attempts === 1 ? '' : 's'}${d.duration_ms != null ? `, ${(d.duration_ms / 1000).toFixed(1)}s` : ''}`}{d.round_idx ? ` · retry ${d.round_idx}` : ''}</div></td>
                          <td className="text-xs whitespace-nowrap">{d.source_study} → {d.target_study}<div><button className="link text-base-content/50" title="all decisions for this pair" onClick={() => { setFilters(f => ({...f, q: d.pair_key, run_id: ''})); setTimeout(() => loadDecisions(0), 0); }}>{d.pair_key}</button></div></td>
                          <td className="text-sm max-w-xs"><div>{d.src_desc}</div><div className="text-xs text-base-content/50">{[d.src_concepts, d.src_cats, d.src_unit].filter(Boolean).join(' · ')}</div></td>
                          <td className="text-sm max-w-xs"><div>{d.tgt_desc}</div><div className="text-xs text-base-content/50">{[d.tgt_concepts, d.tgt_cats, d.tgt_unit].filter(Boolean).join(' · ')}</div></td>
                          <td><StatusBadge s={d.status} />{d.transform && <div className="text-xs mt-1 font-mono break-all">{d.transform}{d.transform_direction ? ` (${d.transform_direction})` : ''}</div>}</td>
                          <td>{d.confidence?.toFixed(2)}</td>
                          <td className="text-xs max-w-sm">{d.reason}{d.error && <div className="text-error">{d.error}</div>}</td>
                          <td><button className="btn btn-xs" onClick={() => openDetail(d.id)}>{detail[d.id] ? 'hide' : 'full I/O'}</button></td>
                        </tr>
                        {detail[d.id] && (
                          <tr><td colSpan={8} className="bg-base-200">
                            {detail[d.id] === 'loading' ? <span className="loading loading-spinner loading-sm" /> : <DecisionDetail d={detail[d.id] as Decision} />}
                          </td></tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="join mt-3">
                <button className="join-item btn btn-sm" disabled={offset === 0} onClick={() => loadDecisions(Math.max(0, offset - PAGE))}>«</button>
                <button className="join-item btn btn-sm btn-disabled">page {Math.floor(offset / PAGE) + 1} / {Math.max(1, Math.ceil(decisions.total / PAGE))}</button>
                <button className="join-item btn btn-sm" disabled={offset + PAGE >= decisions.total} onClick={() => loadDecisions(offset + PAGE)}>»</button>
              </div>
            </>
          )}
        </>
      )}

      {/* ---------------- Consistency ---------------- */}
      {tab === 'consistency' && (
        <>
          <p className="text-sm text-base-content/70 mb-2">
            Pairs the LLM was asked about more than once (other runs, other models, retries), grouped by the content of the
            question. A <span className="badge badge-sm badge-error">conflict</span> is a pair that received different verdicts
            (COMPLETE / COMPATIBLE / PARTIAL / IMPOSSIBLE); failed calls do not count as opinions.
          </p>
          <label className="label cursor-pointer gap-2 w-fit mb-2">
            <input type="checkbox" className="checkbox checkbox-sm" checked={onlyConflicts} onChange={e => setOnlyConflicts(e.target.checked)} />
            <span className="label-text">Conflicts only</span>
          </label>
          {consistency && (
            <>
              <div className="text-sm mb-2">
                {consistency.repeated_pairs} pairs asked more than once · <span className={consistency.conflicting_pairs ? 'text-error font-semibold' : 'text-success'}>{consistency.conflicting_pairs} with conflicting verdicts</span>
              </div>
              {consistency.items.length === 0 && <p className="italic text-base-content/60">Nothing to show.</p>}
              {consistency.items.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="table table-sm">
                    <thead><tr><th>Pair</th><th>Source variable</th><th>Target variable</th><th>Verdicts</th><th>Asked</th><th>Conf. range</th><th></th></tr></thead>
                    <tbody>
                      {consistency.items.map(c => (
                        <React.Fragment key={c.pair_key}>
                          <tr className={`align-top ${c.conflict ? 'bg-error/5' : ''}`}>
                            <td className="text-xs whitespace-nowrap">{c.source_study} → {c.target_study}<div className="text-base-content/50">{c.pair_key}</div></td>
                            <td className="text-sm max-w-xs">{c.src_desc}</td>
                            <td className="text-sm max-w-xs">{c.tgt_desc}</td>
                            <td><div className="flex flex-wrap gap-1">{c.statuses.map(s => <StatusBadge key={s} s={s} />)}</div>{c.conflict && <span className="badge badge-sm badge-error mt-1">conflict</span>}</td>
                            <td className="text-xs whitespace-nowrap">{c.n}× in {c.n_runs} run{c.n_runs === 1 ? '' : 's'}, {c.n_models} model{c.n_models === 1 ? '' : 's'}<div className="text-base-content/50">{fmt(c.first_seen)} → {fmt(c.last_seen)}</div></td>
                            <td className="text-xs">{c.min_conf?.toFixed(2)} – {c.max_conf?.toFixed(2)}</td>
                            <td><button className="btn btn-xs" onClick={() => openPair(c.pair_key)}>{pairHistory[c.pair_key] ? 'hide' : 'history'}</button></td>
                          </tr>
                          {pairHistory[c.pair_key] && (
                            <tr><td colSpan={7} className="bg-base-200">
                              {pairHistory[c.pair_key] === 'loading' ? <span className="loading loading-spinner loading-sm" /> : (
                                <table className="table table-xs">
                                  <thead><tr><th>When</th><th>Run</th><th>Model</th><th>Verdict</th><th>Conf.</th><th>Reason</th><th>Transform</th><th>Source</th><th></th></tr></thead>
                                  <tbody>
                                    {(pairHistory[c.pair_key] as Decision[]).map(h => (
                                      <React.Fragment key={h.id}>
                                        <tr>
                                          <td className="whitespace-nowrap">{fmt(h.created_at)}</td>
                                          <td className="text-xs">{h.run_id}</td>
                                          <td className="text-xs">{h.model}</td>
                                          <td><StatusBadge s={h.status} /></td>
                                          <td>{h.confidence?.toFixed(2)}</td>
                                          <td className="text-xs max-w-sm">{h.reason}{h.error && <div className="text-error">{h.error}</div>}</td>
                                          <td className="text-xs font-mono">{h.transform}</td>
                                          <td className="text-xs">{h.cached ? 'cache' : 'live'}</td>
                                          <td><button className="btn btn-xs" onClick={() => openDetail(h.id)}>{detail[h.id] ? 'hide' : 'full I/O'}</button></td>
                                        </tr>
                                        {detail[h.id] && <tr><td colSpan={9}>{detail[h.id] === 'loading' ? <span className="loading loading-spinner loading-sm" /> : <DecisionDetail d={detail[h.id] as Decision} />}</td></tr>}
                                      </React.Fragment>
                                    ))}
                                  </tbody>
                                </table>
                              )}
                            </td></tr>
                          )}
                        </React.Fragment>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}

/** The complete input and output of one LLM call. */
function DecisionDetail({d}: {d: Decision}) {
  const Block = ({title, text}: {title: string; text?: string | null}) => (
    <div className="mb-2">
      <div className="text-xs font-semibold uppercase text-base-content/60 mb-1">{title}</div>
      <pre className="text-xs whitespace-pre-wrap break-words bg-base-100 rounded p-2 max-h-80 overflow-auto">{text || <span className="italic text-base-content/40">empty</span>}</pre>
    </div>
  );
  return (
    <div className="p-2">
      <div className="grid md:grid-cols-2 gap-3 mb-2 text-sm">
        <div className="bg-base-100 rounded p-2">
          <div className="text-xs font-semibold uppercase text-base-content/60">Source ({d.source_study})</div>
          <div>{d.src_desc}</div>
          <div className="text-xs text-base-content/60">concepts: {d.src_concepts || '-'} · categories: {d.src_cats || '-'} · unit: {d.src_unit || '-'}</div>
        </div>
        <div className="bg-base-100 rounded p-2">
          <div className="text-xs font-semibold uppercase text-base-content/60">Target ({d.target_study})</div>
          <div>{d.tgt_desc}</div>
          <div className="text-xs text-base-content/60">concepts: {d.tgt_concepts || '-'} · categories: {d.tgt_cats || '-'} · unit: {d.tgt_unit || '-'}</div>
        </div>
      </div>
      <div className="text-xs mb-2">
        model <code>{d.model}</code> · mode <code>{d.mapping_mode}</code> · run <code>{d.run_id}</code> · {d.cached ? 'served from the response cache' : `${d.attempts ?? '?'} call(s), ${d.duration_ms != null ? (d.duration_ms / 1000).toFixed(1) : '?'}s`}
        {d.error && <span className="text-error"> · {d.error}</span>}
      </div>
      {d.evidence && <Block title="Structural evidence given to the model" text={d.evidence} />}
      <Block title="System prompt" text={d.system_prompt} />
      <Block title="Prompt" text={d.prompt} />
      <Block title="Raw response" text={d.raw_response} />
      <Block title="Parsed verdict" text={d.verdict_json} />
    </div>
  );
}
