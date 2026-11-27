'use client';

// The data quality issues of one cohort, as the Data Quality Dashboard
// (/data-quality) shows them and as a cohort's explore-page tab shows them:
// the cohort's files, how many variables have issues, a tag per issue type
// (found, checked and clean, or not checked), and the issues grouped by type,
// grouped by variable, or listed in full. The dashboard adds its cohort picker,
// sharing control and concept check through the slots of CohortIssuesView.

import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {Download, FileText, Search, Table, X} from 'react-feather';
import {apiUrl} from '@/utils';
import VariableGraphModal from '@/components/VariableGraphModal';

export interface Issue {
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
  // The variable's standard concept name and its visit, from the dictionary.
  concept_name?: string | null;
  visit_name?: string | null;
  in_dictionary?: boolean;
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

export interface IssueType {
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

export const typeLabel = (t?: IssueType, fallback = '') =>
  t ? (t.spec ? `${t.spec}: ${t.label}` : t.label) : fallback;

// A faint shade per type, spread around the colour wheel so neighbouring types
// differ. Translucent, so it sits on the light and the dark theme alike.
export const typeShade = (index: number): React.CSSProperties => {
  const hue = Math.round((index * 137.508) % 360);
  return {
    backgroundColor: `hsla(${hue}, 65%, 55%, 0.14)`,
    borderColor: `hsla(${hue}, 55%, 50%, 0.45)`,
  };
};

export type TagState = 'found' | 'clean' | 'unchecked';

// One issue type: its count, a checkbox when there is something to show or
// hide, and on hover or focus a card with the definition and an example.
export const TypeTag = ({type, index, state, count, note, shown, onToggle, quiet = false}: {
  type: IssueType;
  index: number;
  state: TagState;
  count: number;
  note: string;
  shown: boolean;
  onToggle: () => void;
  // Smaller and fainter, for the explore page's tab.
  quiet?: boolean;
}) => {
  const [hover, setHover] = useState(false);
  const found = state === 'found';
  const style: React.CSSProperties = found ? {...typeShade(index), ...(quiet ? {backgroundColor: 'transparent'} : {})} : {};
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
        className={`flex items-center rounded-full border transition-opacity ${quiet ? 'gap-1.5 px-2 py-0.5 text-xs' : 'gap-2 px-3 py-1 text-sm'} ${look}`}
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

export interface Category {
  key: string;
  label: string;
  description: string;
}

export interface ConceptJob {
  running: boolean;
  started_at?: string | null;
  finished_at?: string | null;
  done: number;
  total: number;
  current?: string | null;
  error?: string | null;
}

export interface Payload {
  generated_at: string;
  // The explore-page payload names its one cohort.
  cohort?: string;
  cohorts: string[];
  cohorts_with_profiling: string[];
  // Per cohort: its dictionary, how many issues, how many variables they concern.
  cohort_summary?: Record<string, CohortSummary>;
  // The profiling notes files on disk, per cohort (only cohorts that have any).
  data_issues_files?: Record<string, DataIssuesFile[]>;
  categories: Category[];
  types: IssueType[];
  issues: Issue[];
  // cohort → type → whether it was checked, and a note (when, or why not).
  check_status?: Record<string, Record<string, {checked: boolean; note: string}>>;
  // Dashboard only: per cohort, whether its issues show on the explore page,
  // and which types.
  sharing?: Record<string, {enabled?: boolean; types?: string[]; updated_at?: string; updated_by?: string}>;
  sources: {
    syntax: {checked_at?: string | null};
    concepts: {cohorts: Record<string, {checked_at?: string | null; stale?: boolean; status?: string}>; job: ConceptJob};
    values: {read_at?: string | null};
  };
}

// Three levels for every issue: the suspicion level for data values, a level
// per type for the rest (set in the backend).
export const LEVEL_BADGE: Record<string, string> = {
  high: 'badge-error',
  moderate: 'badge-warning',
  low: 'badge-ghost',
};

export const LEVEL_TITLE: Record<string, string> = {
  high: 'Plainly wrong, or stands in the way of using the data',
  moderate: 'Worth checking against the source data or the dictionary',
  low: 'Usually minor, or a natural extreme',
};

export const capitalise = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export const num = (v?: number | null): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
};

export interface CohortSummary {
  issues: number;
  variables_with_issues: number;
  variables_total?: number | null;
  dictionary_file?: string | null;
  dictionary_uploaded_at?: string | null;
}

// dd.mm.yyyy, the date style used across the explorer.
export const fmtDay = (iso?: string | null): string => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(d.getDate())}.${p(d.getMonth() + 1)}.${d.getFullYear()}`;
};

// The blue of the explorer's tags (cohort cards, profiling markers).
export const BLUE = {backgroundColor: '#dbeafe', color: '#1e3a8a', borderColor: '#bfdbfe'};

export interface DataIssuesFile {
  name: string;
  label: string;
  size: number;
  generated_at: string;
}

// dd.mm.yyyy hh:mm, the date style used across the explorer.
export const fmtDayFirst = (iso?: string | null): string => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(d.getDate())}.${p(d.getMonth() + 1)}.${d.getFullYear()} ${p(d.getHours())}:${p(d.getMinutes())}`;
};

// A file's text in an overlay. Closes on Escape, on the backdrop or on the button.
export const FileOverlay = ({cohort, file, onClose}: {cohort: string; file: DataIssuesFile; onClose: () => void}) => {
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

// One variable's row of the current dictionary, every column as written, in an
// overlay. Closes on Escape, on the backdrop or on the button.
export const DictionaryRowOverlay = ({cohort, variable, onClose}: {cohort: string; variable: string; onClose: () => void}) => {
  const [row, setRow] = useState<{file: string; uploaded_at: string; row_number: number; columns: string[];
                                   values: Record<string, string>} | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    fetch(`${apiUrl}/api/data-quality/dictionary-row/${encodeURIComponent(cohort)}/${encodeURIComponent(variable)}`,
          {credentials: 'include'})
      .then(async r => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
        return r.json();
      })
      .then(d => { if (live) setRow(d); })
      .catch(e => { if (live) setError(e?.message || 'Could not load the dictionary row'); });
    return () => { live = false; };
  }, [cohort, variable]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={onClose}>
      <div className="bg-base-100 rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] flex flex-col"
           onClick={e => e.stopPropagation()} role="dialog" aria-modal="true">
        <div className="flex items-start gap-3 px-5 py-3 border-b border-base-300">
          <div className="flex-1 min-w-0">
            <div className="font-semibold">{cohort}: dictionary row of <span className="font-mono">{variable}</span></div>
            {row && (
              <div className="text-xs text-base-content/60">
                row {row.row_number} of {row.file} · uploaded {fmtDay(row.uploaded_at)}
              </div>
            )}
          </div>
          <button className="btn btn-ghost btn-sm btn-square" onClick={onClose} title="Close (Esc)">
            <X size={18} />
          </button>
        </div>
        <div className="overflow-auto px-5 py-3">
          {error ? (
            <div className="alert alert-error text-sm">{error}</div>
          ) : !row ? (
            <div className="flex items-center gap-2 text-base-content/70">
              <span className="loading loading-spinner loading-sm" /> Loading…
            </div>
          ) : (
            <table className="table table-sm">
              <tbody>
                {row.columns.map(col => {
                  const value = row.values[col] ?? '';
                  return (
                    <tr key={col}>
                      <th className="align-top whitespace-nowrap text-xs font-semibold w-1/3">{col}</th>
                      <td className={`align-top text-sm whitespace-pre-wrap break-words ${value ? '' : 'text-base-content/40'}`}>
                        {value || '(empty)'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
};

export type DisplayMode = 'by-type' | 'by-variable' | 'list';

export const DISPLAY_MODES: {key: DisplayMode; label: string}[] = [
  {key: 'by-type', label: 'Group by issue type'},
  {key: 'by-variable', label: 'Group by variable'},
  {key: 'list', label: 'Expanded list'},
];


export function CohortIssuesView({data, cohort, quietTags = false, topRight, belowTop, picker, afterCard}: {
  data: Payload;
  cohort: string;
  // Smaller, fainter tags (the explore page).
  quietTags?: boolean;
  // Slots for the dashboard: beside the file buttons, under them, between them
  // and the count, and under the card.
  topRight?: React.ReactNode;
  belowTop?: React.ReactNode;
  picker?: React.ReactNode;
  afterCard?: React.ReactNode;
}) {
  // Types the reader has unticked. Kept as the hidden set, so a type that first
  // appears after a refresh is shown.
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<DisplayMode>('by-type');
  const [chartOf, setChartOf] = useState<Issue | null>(null);
  const [rowOf, setRowOf] = useState<string | null>(null);
  const [openFile, setOpenFile] = useState<DataIssuesFile | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const current = data.cohort_summary?.[cohort];

  const typeByKey = useMemo(() => {
    const m: Record<string, IssueType> = {};
    for (const t of data.types || []) m[t.key] = t;
    return m;
  }, [data]);

  const typeOrder = useMemo(() => {
    const m: Record<string, number> = {};
    (data.types || []).forEach((t, i) => { m[t.key] = i; });
    return m;
  }, [data]);

  const cohortIssues = useMemo(() => (data.issues || []).filter(i => i.cohort === cohort), [data, cohort]);

  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const i of cohortIssues) counts[i.type] = (counts[i.type] || 0) + 1;
    return counts;
  }, [cohortIssues]);

  // Per type: found, checked and clean, or not checked.
  const tagInfo = useMemo(() => {
    const info: Record<string, {state: TagState; count: number; note: string}> = {};
    const status = data.check_status?.[cohort] || {};
    for (const t of data.types || []) {
      const count = typeCounts[t.key] || 0;
      const checked = !!status[t.key]?.checked;
      const state: TagState = count > 0 ? 'found' : checked ? 'clean' : 'unchecked';
      info[t.key] = {state, count, note: status[t.key]?.note || ''};
    }
    return info;
  }, [data, cohort, typeCounts]);

  const stateRank: Record<TagState, number> = {found: 0, clean: 1, unchecked: 2};

  // The tags of a category: every regular type, the occasional ones only when
  // they have findings; found first by count, then clean, then not checked.
  const tagsOf = useCallback((category: string) => (data.types || [])
    .filter(t => t.category === category && (!t.occasional || (typeCounts[t.key] || 0) > 0))
    .sort((a, b) => {
      const ia = tagInfo[a.key], ib = tagInfo[b.key];
      return stateRank[ia.state] - stateRank[ib.state] || ib.count - ia.count
        || (typeOrder[a.key] ?? 0) - (typeOrder[b.key] ?? 0);
    }), [data, typeCounts, tagInfo, typeOrder]);

  const summary = useMemo(() => {
    const regular = (data.types || []).filter(t => !t.occasional);
    const run = regular.filter(t => tagInfo[t.key]?.state !== 'unchecked').length;
    const found = (data.types || []).filter(t => tagInfo[t.key]?.state === 'found').length;
    return {run, found, notRun: regular.length - run};
  }, [data, tagInfo]);

  const visible = useMemo(() => {
    let items = cohortIssues.filter(i => !hiddenTypes.has(i.type));
    const q = query.trim().toLowerCase();
    if (q) {
      items = items.filter(i =>
        (i.variable || '').toLowerCase().includes(q) ||
        (i.label || '').toLowerCase().includes(q) ||
        (i.concept_name || '').toLowerCase().includes(q) ||
        i.message.toLowerCase().includes(q)
      );
    }
    return items;
  }, [cohortIssues, hiddenTypes, query]);

  // Group by issue type: the most frequent first, then in specification order;
  // within a type the highest level first, then by variable.
  const byType = useMemo(() => {
    const groups = new Map<string, Issue[]>();
    for (const i of visible) {
      if (!groups.has(i.type)) groups.set(i.type, []);
      groups.get(i.type)!.push(i);
    }
    return [...groups.entries()]
      .sort((a, b) => b[1].length - a[1].length || (typeOrder[a[0]] ?? 0) - (typeOrder[b[0]] ?? 0))
      .map(([type, items]) => ({
        type,
        items: [...items].sort((x, y) => (x.level_rank ?? 9) - (y.level_rank ?? 9)
          || (x.variable || '').toLowerCase().localeCompare((y.variable || '').toLowerCase())),
      }));
  }, [visible, typeOrder]);

  // Group by variable: the variables with the most serious issues first, then
  // the most issues; issues that name no variable come last, as one group.
  const byVariable = useMemo(() => {
    const groups = new Map<string, Issue[]>();
    for (const i of visible) {
      const key = i.variable ? i.variable.toLowerCase() : '';
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(i);
    }
    return [...groups.entries()]
      .map(([key, items]) => ({
        key,
        items: [...items].sort((x, y) => (x.level_rank ?? 9) - (y.level_rank ?? 9)
          || (typeOrder[x.type] ?? 0) - (typeOrder[y.type] ?? 0)),
      }))
      .sort((a, b) => (a.key === '' ? 1 : 0) - (b.key === '' ? 1 : 0)
        || Math.min(...a.items.map(i => i.level_rank ?? 9)) - Math.min(...b.items.map(i => i.level_rank ?? 9))
        || b.items.length - a.items.length || a.key.localeCompare(b.key));
  }, [visible, typeOrder]);

  const toggleType = (key: string) => setHiddenTypes(prev => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  const setCategoryShown = (category: string, shown: boolean) => setHiddenTypes(prev => {
    const next = new Set(prev);
    for (const t of data.types || []) {
      if (t.category !== category || !typeCounts[t.key]) continue;
      if (shown) next.delete(t.key); else next.add(t.key);
    }
    return next;
  });

  const downloadDictionary = async () => {
    setDownloadError(null);
    try {
      const r = await fetch(`${apiUrl}/api/data-quality/dictionary/${encodeURIComponent(cohort)}`, {credentials: 'include'});
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
      const blob = await r.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = current?.dictionary_file || `${cohort}_datadictionary.csv`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e: any) {
      setDownloadError(e?.message || 'Could not download the dictionary');
    }
  };

  const downloadCsv = () => {
    const header = ['cohort', 'category', 'type', 'variable', 'dictionary_row', 'label', 'concept_name', 'visit',
      'level', 'blocks_dictionary_upload', 'message'];
    const esc = (v: any) => `"${String(v ?? '').replace(/"/g, '""')}"`;
    const lines = [header.join(',')].concat(visible.map(i => [
      i.cohort, i.category, typeLabel(typeByKey[i.type], i.type), i.variable, i.row, i.label, i.concept_name,
      i.visit_name || i.visits, i.level, i.blocks_upload ? 'yes' : '', i.message,
    ].map(esc).join(',')));
    const blob = new Blob([lines.join('\n')], {type: 'text/csv'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `data-issues-${cohort}-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  // --- pieces of a row ------------------------------------------------------

  const typeBadge = (type: string) => {
    const t = typeByKey[type];
    return (
      <span className="badge badge-sm border whitespace-nowrap" style={typeShade(typeOrder[type] ?? 0)} title={t?.description}>
        {typeLabel(t, type)}
      </span>
    );
  };

  const levelCell = (i: Issue) => (
    <>
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
    </>
  );

  const whatCell = (i: Issue) => (
    <>
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
    </>
  );

  // A variable: name and row, its label, standard concept name, visit and
  // units, the chart button (or a note that it has none), and its dictionary row.
  const variableCell = (i: Issue, withRow = true, withVisitAndUnits = true) => {
    if (!i.variable) {
      return <span className="text-xs text-base-content/50">{i.row ? `row ${i.row}` : 'whole dictionary or dataset'}</span>;
    }
    return (
      <div className="space-y-0.5">
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="font-mono text-xs font-semibold">{i.variable}</span>
          {withRow && i.row && <span className="text-xs text-base-content/50">row {i.row}</span>}
        </div>
        {i.label && <div className="text-xs" title={i.label}>{i.label}</div>}
        {i.concept_name && (
          <div className="text-xs text-base-content/60"><span className="text-base-content/45">Concept: </span>{i.concept_name}</div>
        )}
        {withVisitAndUnits && (i.visit_name || i.visits) && (
          <div className="text-xs text-base-content/60"><span className="text-base-content/45">Visit: </span>{i.visit_name || i.visits}</div>
        )}
        {withVisitAndUnits && i.units && <div className="text-xs text-base-content/45">{i.units}</div>}
        <div className="flex flex-nowrap items-center gap-1.5 pt-1 whitespace-nowrap">
          {i.has_chart ? (
            <button
              className="btn-sm rounded-lg bg-teal-50 border border-teal-200 hover:bg-teal-100 dark:bg-teal-900/30 dark:border-teal-700 inline-flex items-center gap-1.5 px-2"
              title="Summary statistics — click to view"
              onClick={() => setChartOf(i)}
            >
              <span className="text-xl leading-none">📊</span>
              <span className="text-xs font-medium text-teal-800 dark:text-teal-300">summary statistics</span>
            </button>
          ) : (
            <span className="text-[11px] text-base-content/50">This variable has no EDA information.</span>
          )}
          {i.in_dictionary && (
            <button className="btn btn-xs btn-outline gap-1 shrink-0" onClick={() => setRowOf(i.variable!)}
                    title="Show this variable's row of the metadata dictionary">
              <Table size={12} /> dictionary row
            </button>
          )}
        </div>
      </div>
    );
  };

  return (
    <>
      <div className="card bg-base-200 mb-5">
        <div className="card-body p-5 gap-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            {/* The cohort's files: the profiling notes and its dictionary. */}
            <div className="flex flex-wrap items-center gap-2">
              {(data.data_issues_files?.[cohort] || []).map(f => (
                <button key={f.name} className="btn btn-sm btn-outline gap-2 h-auto py-1.5" onClick={() => setOpenFile(f)}>
                  <FileText size={14} />
                  <span className="flex flex-col items-start leading-tight">
                    <span>{f.label}</span>
                    <span className="text-[11px] font-normal opacity-70">generated {fmtDayFirst(f.generated_at)}</span>
                  </span>
                </button>
              ))}
              {current?.dictionary_file && (
                <button className="btn btn-sm btn-outline gap-2 h-auto py-1.5" onClick={downloadDictionary}
                        title={current.dictionary_file}>
                  <Download size={14} />
                  <span className="flex flex-col items-start leading-tight">
                    <span>Download {cohort.length > 12 ? `${cohort.slice(0, 10)}...` : cohort}&apos;s Dictionary</span>
                    <span className="text-[11px] font-normal opacity-70">uploaded {fmtDayFirst(current.dictionary_uploaded_at)}</span>
                  </span>
                </button>
              )}
              {downloadError && <span className="text-error text-xs">{downloadError}</span>}
            </div>
            {topRight}
          </div>

          {belowTop}
          {picker}

          {current && (
            <div className="text-center text-base text-base-content/80">
              <span className="font-semibold text-base-content">{current.variables_with_issues}</span>
              {' '}of {current.variables_total ?? '—'} variables have issues
              <span className="text-base-content/60"> · {current.issues} issues in all</span>
            </div>
          )}

          <div className={`flex flex-col ${quietTags ? 'gap-2' : 'gap-3'}`}>
            <div className={`text-center text-base-content/70 ${quietTags ? 'text-xs' : 'text-sm'}`}>
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
                <div key={cat.key} className={`flex flex-col items-center ${quietTags ? 'gap-1' : 'gap-2'}`}>
                  <button className={`font-semibold hover:underline disabled:no-underline disabled:cursor-default ${quietTags ? 'text-xs text-base-content/70' : 'text-sm'}`}
                          disabled={!foundTypes.length}
                          title={`${cat.description}${foundTypes.length ? ` Click to ${allShown ? 'hide' : 'show'} all found.` : ''}`}
                          onClick={() => setCategoryShown(cat.key, !allShown)}>
                    {cat.label}
                  </button>
                  <div className={`flex flex-wrap items-center justify-center ${quietTags ? 'gap-1.5' : 'gap-2'}`}>
                    {types.map(t => (
                      <TypeTag key={t.key} type={t} index={typeOrder[t.key] ?? 0} quiet={quietTags}
                               state={tagInfo[t.key].state} count={tagInfo[t.key].count} note={tagInfo[t.key].note}
                               shown={!hiddenTypes.has(t.key)} onToggle={() => toggleType(t.key)} />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {afterCard}

      <div className="flex flex-wrap items-end gap-3 mb-4">
        <div className="join">
          {DISPLAY_MODES.map(m => (
            <button key={m.key} onClick={() => setMode(m.key)}
                    className={`btn btn-sm join-item ${mode === m.key ? 'btn-active btn-neutral' : ''}`}>
              {m.label}
            </button>
          ))}
        </div>
        <label className="form-control flex-1 min-w-60">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 opacity-50" />
            <input value={query} onChange={e => setQuery(e.target.value)}
                   placeholder="Search variable, label, concept or message"
                   className="input input-bordered input-sm w-full pl-8" />
          </div>
        </label>
        <button className="btn btn-ghost btn-sm" onClick={downloadCsv} disabled={!visible.length} title="Download the list as CSV">
          <Download size={16} />
        </button>
      </div>

      <p className="text-xs text-base-content/60 mb-2">
        Showing {visible.length} of {cohortIssues.length} issues for {cohort}.
        {data.sources?.syntax?.checked_at && ` Dictionary checked ${fmtDayFirst(data.sources.syntax.checked_at.replace(' ', 'T'))}.`}
        {data.sources?.values?.read_at && ` Profiling read ${fmtDayFirst(data.sources.values.read_at)}.`}
      </p>

      {!cohortIssues.length ? (
        <div className="alert">No known issues for {cohort}.</div>
      ) : !visible.length ? (
        <div className="alert">Nothing to show with these filters.</div>
      ) : mode === 'by-type' ? (
        <div className="overflow-x-auto">
          <table className="table table-sm">
            <tbody>
              {byType.map(g => {
                const nVars = new Set(g.items.filter(i => i.variable).map(i => i.variable!.toLowerCase())).size;
                return (
                  <React.Fragment key={g.type}>
                    <tr className="bg-base-200 border-t-2 border-base-300">
                      <td colSpan={3} className="py-3">
                        <div className="flex flex-wrap items-center gap-3">
                          <span className="badge badge-lg border text-base font-semibold px-3 py-3 whitespace-nowrap"
                                style={typeShade(typeOrder[g.type] ?? 0)} title={typeByKey[g.type]?.description}>
                            {typeLabel(typeByKey[g.type], g.type)}
                          </span>
                          <span className="text-sm text-base-content/70">
                            {g.items.length} {g.items.length === 1 ? 'issue' : 'issues'}
                            {nVars ? ` · ${nVars} ${nVars === 1 ? 'variable' : 'variables'}` : ''}
                          </span>
                        </div>
                      </td>
                    </tr>
                    {g.items.map((i, idx) => (
                      <tr key={`${g.type}/${i.variable || ''}/${i.row || ''}/${idx}`}>
                        <td className="align-top pl-10 min-w-80">{variableCell(i, true, false)}</td>
                        <td className="align-top text-xs max-w-2xl">{whatCell(i)}</td>
                        <td className="align-top whitespace-nowrap">{levelCell(i)}</td>
                      </tr>
                    ))}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : mode === 'by-variable' ? (
        <div className="overflow-x-auto">
          <table className="table table-sm">
            <tbody>
              {byVariable.map(g => (
                <React.Fragment key={g.key || '(none)'}>
                  <tr className="bg-base-200">
                    <td colSpan={3} className="py-2">
                      {g.key ? (
                        <div className="flex flex-wrap items-start gap-4">
                          <div className="flex-1 min-w-64">{variableCell(g.items[0], false)}</div>
                          <span className="text-xs text-base-content/70">
                            {g.items.length} {g.items.length === 1 ? 'issue' : 'issues'}
                          </span>
                        </div>
                      ) : (
                        <span className="text-sm font-semibold">Whole dictionary or dataset
                          <span className="text-xs font-normal text-base-content/70"> · {g.items.length} {g.items.length === 1 ? 'issue' : 'issues'}</span>
                        </span>
                      )}
                    </td>
                  </tr>
                  {g.items.map((i, idx) => (
                    <tr key={`${g.key}/${i.type}/${idx}`}>
                      <td className="align-top pl-10 whitespace-nowrap">
                        {typeBadge(i.type)}
                        {i.row && <div className="text-xs text-base-content/50 mt-1">row {i.row}</div>}
                      </td>
                      <td className="align-top text-xs max-w-2xl">{whatCell(i)}</td>
                      <td className="align-top whitespace-nowrap">{levelCell(i)}</td>
                    </tr>
                  ))}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="table table-sm table-zebra">
            <thead>
              <tr>
                <th>Issue</th>
                <th>Variable</th>
                <th>What is wrong</th>
                <th>Level</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((i, idx) => (
                <tr key={`${i.type}/${i.variable || ''}/${i.row || ''}/${idx}`}>
                  <td className="whitespace-nowrap align-top">{typeBadge(i.type)}</td>
                  <td className="min-w-80 align-top">{variableCell(i)}</td>
                  <td className="max-w-2xl align-top text-xs">{whatCell(i)}</td>
                  <td className="whitespace-nowrap align-top">{levelCell(i)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {openFile && <FileOverlay cohort={cohort} file={openFile} onClose={() => setOpenFile(null)} />}
      {rowOf && <DictionaryRowOverlay cohort={cohort} variable={rowOf} onClose={() => setRowOf(null)} />}
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
    </>
  );
}

// A cohort's data quality tab on its explore-page card: the issue types an
// admin chose to share, for the platform admins and the cohort's owners only
// (the backend refuses everyone else), with quieter tags.
export function ExploreDataQualityTab({cohort}: {cohort: string}) {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    fetch(`${apiUrl}/api/cohort-data-quality/${encodeURIComponent(cohort)}`, {credentials: 'include'})
      .then(async r => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
        return r.json();
      })
      .then(d => { if (live) setData(d); })
      .catch(e => { if (live) setError(e?.message || 'Could not load the data quality issues'); });
    return () => { live = false; };
  }, [cohort]);

  return (
    <div className="px-2">
      <div className="alert border border-red-200 bg-red-50 text-red-900 dark:bg-red-950/30 dark:text-red-200 dark:border-red-900 mb-4 text-sm">
        Only the platform admins and the owners of {cohort} can see this page.
      </div>
      {error ? (
        <div className="alert alert-error text-sm">{error}</div>
      ) : !data ? (
        <div className="flex items-center gap-3 text-base-content/70 py-6">
          <span className="loading loading-spinner loading-md" /> <span>Gathering the issues…</span>
        </div>
      ) : (
        <CohortIssuesView data={data} cohort={data.cohort || cohort} quietTags />
      )}
    </div>
  );
}
