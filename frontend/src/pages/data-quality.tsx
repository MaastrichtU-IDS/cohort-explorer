'use client';

// Data quality (/data-quality): every known issue of a cohort in one list, from
// three sources: the dictionary syntax check every upload goes through, the
// OMOP concept-code check, and the suspect values found in the variable
// profiling. Pick a cohort at the top; its issue types appear as tags, each
// with a checkbox that shows or hides that type. The links to the individual
// checks are on the data quality hub (/data-quality-hub).

import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {AlertTriangle, Download, FileText, Play, RefreshCw, Search, X} from 'react-feather';
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
  // Whether the variable has a distribution chart from the profiling.
  has_chart?: boolean;
  // A dictionary syntax issue: the upload validation would reject the dictionary.
  blocks_upload?: boolean;
}

interface IssueType {
  key: string;
  category: string;
  label: string;
  description: string;
  // Id in the data quality specification (S1 to S14, I1 to I6), where the type has one.
  spec?: string | null;
  // A short generic example, for the explanation on hover.
  example?: string | null;
  // Catch-alls and check failures: shown only when a cohort has some.
  occasional?: boolean;
}

const typeLabel = (t?: IssueType, fallback = '') =>
  t ? (t.spec ? `${t.spec}: ${t.label}` : t.label) : fallback;

// A faint shade per type, spread around the colour wheel so neighbouring types
// differ. Translucent, so it sits on the light and the dark theme alike.
const typeShade = (index: number): React.CSSProperties => {
  const hue = Math.round((index * 137.508) % 360);
  return {
    backgroundColor: `hsla(${hue}, 65%, 55%, 0.14)`,
    borderColor: `hsla(${hue}, 55%, 50%, 0.45)`,
  };
};

type TagState = 'found' | 'clean' | 'unchecked';

// One issue type: its count, a checkbox when there is something to show or
// hide, and on hover or focus a card with the definition and an example.
const TypeTag = ({type, index, state, count, note, shown, onToggle}: {
  type: IssueType;
  index: number;
  state: TagState;
  count: number;
  note: string;
  shown: boolean;
  onToggle: () => void;
}) => {
  const [hover, setHover] = useState(false);
  const found = state === 'found';
  const style: React.CSSProperties = found ? typeShade(index) : {};
  const look = found
    ? `cursor-pointer ${shown ? '' : 'opacity-50'}`
    : state === 'clean'
      ? 'cursor-default bg-base-200 border-base-300 text-base-content/50'
      : 'cursor-default border-dashed border-base-content/30 text-base-content/45';
  return (
    <div className="relative" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      <label
        tabIndex={found ? undefined : 0}
        aria-disabled={!found}
        onFocus={() => setHover(true)}
        onBlur={() => setHover(false)}
        style={style}
        className={`flex items-center gap-2 rounded-full border px-3 py-1 text-sm transition-opacity ${look}`}
      >
        <input type="checkbox" className="checkbox checkbox-xs" checked={found && shown} disabled={!found}
               onChange={() => { if (found) onToggle(); }}
               onFocus={() => setHover(true)} onBlur={() => setHover(false)} />
        <span>{typeLabel(type)}</span>
        <span className={`font-mono text-xs ${found ? 'font-semibold' : ''}`}>{state === 'unchecked' ? '—' : count}</span>
      </label>
      {hover && (
        <div role="tooltip"
             className="absolute z-40 left-1/2 -translate-x-1/2 top-full mt-2 w-80 max-w-[85vw] rounded-lg border border-base-300 bg-base-100 p-3 shadow-lg text-left text-xs font-normal text-base-content">
          <div className="font-semibold text-sm mb-1">{typeLabel(type)}</div>
          <div className="mb-2">{type.description}</div>
          {type.example && (
            <div className="mb-2">
              <span className="text-base-content/60">Example: </span>
              <span className="font-mono">{type.example}</span>
            </div>
          )}
          <div className="text-base-content/70">
            {state === 'found' && `${count} found${note ? ` (${note})` : ''}.`}
            {state === 'clean' && `None found${note ? ` (${note})` : ''}.`}
            {state === 'unchecked' && `Not checked yet: ${note || 'no data to check against'}.`}
          </div>
        </div>
      )}
    </div>
  );
};

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
  // The profiling notes files on disk, per cohort (only cohorts that have any).
  data_issues_files?: Record<string, DataIssuesFile[]>;
  categories: Category[];
  types: IssueType[];
  issues: Issue[];
  // cohort → type → whether it was checked, and a note (when, or why not).
  check_status?: Record<string, Record<string, {checked: boolean; note: string}>>;
  sources: {
    syntax: {checked_at?: string | null};
    concepts: {cohorts: Record<string, {checked_at?: string | null; stale?: boolean; status?: string}>; job: ConceptJob};
    values: {read_at?: string | null};
  };
}

// Three levels for every issue: the suspicion level for data values, a level
// per type for the rest (set in the backend).
const LEVEL_BADGE: Record<string, string> = {
  high: 'badge-error',
  moderate: 'badge-warning',
  low: 'badge-ghost',
};

const LEVEL_TITLE: Record<string, string> = {
  high: 'Plainly wrong, or stands in the way of using the data',
  moderate: 'Worth checking against the source data or the dictionary',
  low: 'Usually minor, or a natural extreme',
};

const capitalise = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const num = (v?: number | null): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
};

interface DataIssuesFile {
  name: string;
  label: string;
  size: number;
  generated_at: string;
}

// dd.mm.yyyy hh:mm, the date style used across the explorer.
const fmtDayFirst = (iso?: string | null): string => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(d.getDate())}.${p(d.getMonth() + 1)}.${d.getFullYear()} ${p(d.getHours())}:${p(d.getMinutes())}`;
};

// A file's text in an overlay. Closes on Escape, on the backdrop or on the button.
const FileOverlay = ({cohort, file, onClose}: {cohort: string; file: DataIssuesFile; onClose: () => void}) => {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    fetch(`${apiUrl}/api/data-quality/data-issues-files/${encodeURIComponent(cohort)}/${encodeURIComponent(file.name)}`,
          {credentials: 'include'})
      .then(async r => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
        return r.json();
      })
      .then(d => { if (live) setContent(d.content ?? ''); })
      .catch(e => { if (live) setError(e?.message || 'Could not load the file'); });
    return () => { live = false; };
  }, [cohort, file.name]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={onClose}>
      <div className="bg-base-100 rounded-lg shadow-xl w-full max-w-5xl max-h-[85vh] flex flex-col"
           onClick={e => e.stopPropagation()} role="dialog" aria-modal="true">
        <div className="flex items-start gap-3 px-5 py-3 border-b border-base-300">
          <div className="flex-1 min-w-0">
            <div className="font-semibold">{cohort}: {file.label}</div>
            <div className="text-xs text-base-content/60">
              generated {fmtDayFirst(file.generated_at)}
            </div>
          </div>
          <button className="btn btn-ghost btn-sm btn-square" onClick={onClose} title="Close (Esc)">
            <X size={18} />
          </button>
        </div>
        <div className="overflow-auto px-5 py-4">
          {error ? (
            <div className="alert alert-error text-sm">{error}</div>
          ) : content === null ? (
            <div className="flex items-center gap-2 text-base-content/70">
              <span className="loading loading-spinner loading-sm" /> Loading…
            </div>
          ) : (
            <pre className="font-mono text-sm whitespace-pre-wrap break-words">{content || '(empty file)'}</pre>
          )}
        </div>
      </div>
    </div>
  );
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
  // The profiling notes file open in the overlay, if any.
  const [openFile, setOpenFile] = useState<DataIssuesFile | null>(null);

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

  // Per type, for the cohort in view: found, checked and clean, or not checked.
  // With all cohorts in view a type counts as checked where any cohort was.
  const tagInfo = useMemo(() => {
    const info: Record<string, {state: TagState; count: number; note: string}> = {};
    const status = data?.check_status || {};
    const cohorts = cohort ? [cohort] : Object.keys(status);
    for (const t of data?.types || []) {
      const count = typeCounts[t.key] || 0;
      const checkedIn = cohorts.filter(c => status[c]?.[t.key]?.checked).length;
      let note = '';
      if (cohort) note = status[cohort]?.[t.key]?.note || '';
      else if (cohorts.length) note = `checked in ${checkedIn} of ${cohorts.length} cohorts`;
      const state: TagState = count > 0 ? 'found' : checkedIn > 0 ? 'clean' : 'unchecked';
      info[t.key] = {state, count, note};
    }
    return info;
  }, [data, cohort, typeCounts]);

  const stateRank: Record<TagState, number> = {found: 0, clean: 1, unchecked: 2};

  // The tags of a category: every regular type, the occasional ones only when
  // they have findings; found first by count, then clean, then not checked.
  const tagsOf = useCallback((category: string) => (data?.types || [])
    .filter(t => t.category === category && (!t.occasional || (typeCounts[t.key] || 0) > 0))
    .sort((a, b) => {
      const ia = tagInfo[a.key], ib = tagInfo[b.key];
      return stateRank[ia.state] - stateRank[ib.state] || ib.count - ia.count
        || (typeOrder[a.key] ?? 0) - (typeOrder[b.key] ?? 0);
    }), [data, typeCounts, tagInfo, typeOrder]);

  const summary = useMemo(() => {
    const regular = (data?.types || []).filter(t => !t.occasional);
    const run = regular.filter(t => tagInfo[t.key]?.state !== 'unchecked').length;
    const found = (data?.types || []).filter(t => tagInfo[t.key]?.state === 'found').length;
    return {run, found, notRun: regular.length - run};
  }, [data, tagInfo]);

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


  const conceptStatus = cohort ? data?.sources.concepts.cohorts[cohort] : undefined;
  const conceptCheckedCount = useMemo(
    () => Object.values(data?.sources.concepts.cohorts || {}).filter(c => c.checked_at).length,
    [data]);

  const downloadCsv = () => {
    const header = ['cohort', 'category', 'type', 'variable', 'dictionary_row', 'visit', 'label', 'level', 'blocks_dictionary_upload', 'message'];
    const esc = (v: any) => `"${String(v ?? '').replace(/"/g, '""')}"`;
    const lines = [header.join(',')].concat(visible.map(i => [
      i.cohort, i.category, typeLabel(typeByKey[i.type], i.type), i.variable, i.row, i.visits, i.label,
      i.level, i.blocks_upload ? 'yes' : '', i.message,
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
        <button className="btn btn-ghost btn-sm ml-auto" onClick={load} title="Refresh">
          <RefreshCw size={16} />
        </button>
        <button className="btn btn-ghost btn-sm" onClick={downloadCsv} disabled={!visible.length} title="Download the list as CSV">
          <Download size={16} />
        </button>
      </div>
      <div className="card bg-base-200 mb-5 mt-3">
        <div className="card-body p-5 gap-4">
          {/* The chosen cohort's profiling notes files, when the server has them. */}
          {cohort && (data?.data_issues_files?.[cohort] || []).length > 0 && (
            <div className="flex flex-wrap items-center gap-2 self-start">
              {(data?.data_issues_files?.[cohort] || []).map(f => (
                <button key={f.name} className="btn btn-sm btn-outline gap-2 h-auto py-1.5" onClick={() => setOpenFile(f)}>
                  <FileText size={14} />
                  <span className="flex flex-col items-start leading-tight">
                    <span>{f.label}</span>
                    <span className="text-[11px] font-normal opacity-70">generated {fmtDayFirst(f.generated_at)}</span>
                  </span>
                </button>
              ))}
            </div>
          )}
          <label className="form-control w-full max-w-xl mx-auto">
            <span className="label-text font-semibold mb-1">Cohort</span>
            <select className="select select-bordered select-lg w-full" value={cohort}
                    onChange={e => { setCohort(e.target.value); setOpenFile(null); }}>
              <option value="">All cohorts</option>
              {(data?.cohorts || []).map(c => (
                <option key={c} value={c}>{c}{byCohort[c] ? ` (${byCohort[c]})` : ' (no issues)'}</option>
              ))}
            </select>
          </label>

          {data && (
            <div className="flex flex-col gap-3">
              <div className="text-center text-sm text-base-content/70">
                <span className="font-semibold text-base-content">{summary.run}</span> checks run
                {' · '}<span className="font-semibold text-base-content">{summary.found}</span> found issues
                {' · '}<span className="font-semibold text-base-content">{summary.notRun}</span> not run yet
                <span className="text-xs text-base-content/50"> (hover a tag for what it checks)</span>
              </div>
              {data.categories.map(cat => {
                const types = tagsOf(cat.key);
                if (!types.length) return null;
                const foundTypes = types.filter(t => tagInfo[t.key]?.state === 'found');
                const allShown = foundTypes.every(t => !hiddenTypes.has(t.key));
                return (
                  <div key={cat.key} className="flex flex-wrap items-center justify-center gap-2">
                    <button className="text-sm font-semibold mr-1 hover:underline disabled:no-underline disabled:cursor-default"
                            disabled={!foundTypes.length}
                            title={`${cat.description}${foundTypes.length ? ` Click to ${allShown ? 'hide' : 'show'} all found.` : ''}`}
                            onClick={() => setCategoryShown(cat.key, !allShown)}>
                      {cat.label}
                    </button>
                    {types.map(t => (
                      <TypeTag key={t.key} type={t} index={typeOrder[t.key] ?? 0}
                               state={tagInfo[t.key].state} count={tagInfo[t.key].count} note={tagInfo[t.key].note}
                               shown={!hiddenTypes.has(t.key)} onToggle={() => toggleType(t.key)} />
                    ))}
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
                <th>Level</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((i, idx) => {
                const t = typeByKey[i.type];
                return (
                  <tr key={`${i.cohort}/${i.type}/${i.variable || ''}/${i.row || ''}/${idx}`}>
                    <td className="whitespace-nowrap align-top">
                      <span className="badge badge-sm border" style={typeShade(typeOrder[i.type] ?? 0)} title={t?.description}>
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
                          {i.has_chart ? (
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
                          ) : (
                            <div className="mt-1 text-[11px] text-base-content/50">This variable has no EDA information.</div>
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
                      {i.level && (
                        <span className={`badge badge-sm ${LEVEL_BADGE[i.level] || 'badge-ghost'}`} title={LEVEL_TITLE[i.level]}>
                          {capitalise(i.level)}
                        </span>
                      )}
                      {i.blocks_upload && (
                        <div className="text-[11px] text-base-content/60 mt-1 whitespace-normal max-w-40">
                          Would block a new upload of the dictionary
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {openFile && cohort && (
        <FileOverlay cohort={cohort} file={openFile} onClose={() => setOpenFile(null)} />
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
