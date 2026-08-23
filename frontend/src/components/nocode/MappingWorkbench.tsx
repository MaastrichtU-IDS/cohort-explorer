'use client';

// Mapping Workbench — the harmonization board of the no-code DCR wizard.
//
// For every role the chosen analysis needs (e.g. "variable of interest",
// "break down by"), the user builds one HARMONIZED VARIABLE directly in the
// role field: a searchable dropdown over every variable of the selected
// cohorts picks the anchor; for multi-cohort analyses one more dropdown per
// other cohort picks the member, with evidence-ranked suggestions (shared
// codes, text similarity, cached mapping files, AI) listed first; then a value
// map (categorical) or unit conversion (numeric). Everything the user decides
// ends up in the mapping spec, which the enclave script prints under every
// figure as provenance.
import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {AlertTriangle, Check, ChevronDown, Database, Trash2, X, Zap} from 'react-feather';
import {SparklesIcon as Sparkles} from '@/components/Icons';
import {
  Candidate,
  Evidence,
  HVar,
  MappingSpec,
  ValueCluster,
  VarInfo,
  aiSuggest,
  categoryLine,
  edaLine,
  unitVisitLines,
  fetchCachedMappings,
  fetchAllVariables,
  newHVar,
  provenanceLine,
  suggestMatches,
  suggestValues
} from './client';

export interface RoleDef {
  key: string;
  label: string;
  hint?: string;
  optional?: boolean;
  kind?: 'numeric' | 'categorical';
}

interface Props {
  cohorts: string[];
  roles: RoleDef[];
  mapping: MappingSpec;
  roleAssignments: Record<string, string>; // role key -> harmonized_name
  onMappingChange: (m: MappingSpec) => void;
  onRolesChange: (r: Record<string, string>) => void;
  userEmail?: string | null;
}

// Distinct, stable colour per cohort (full class strings for Tailwind's JIT).
const COHORT_COLORS = [
  'bg-sky-100 text-sky-900 border-sky-300',
  'bg-emerald-100 text-emerald-900 border-emerald-300',
  'bg-amber-100 text-amber-900 border-amber-300',
  'bg-rose-100 text-rose-900 border-rose-300',
  'bg-violet-100 text-violet-900 border-violet-300',
  'bg-teal-100 text-teal-900 border-teal-300'
];
export const cohortColor = (cohorts: string[], id: string) => COHORT_COLORS[Math.max(0, cohorts.indexOf(id)) % COHORT_COLORS.length];

const EVIDENCE_STYLE: Record<string, string> = {
  code: 'bg-emerald-600 text-white',
  text: 'bg-sky-600 text-white',
  cache: 'bg-violet-600 text-white',
  ai: 'bg-amber-500 text-white',
  manual: 'bg-slate-500 text-white',
  label: 'bg-slate-300 text-slate-800',
  warning: 'bg-rose-600 text-white'
};

export function EvidenceBadge({e}: {e: Evidence}) {
  let text = e.type.toUpperCase();
  let title = '';
  if (e.type === 'code') {
    const sys = e.system || 'code';
    text = sys === 'OMOP ID' ? 'SAME OMOP ID' : `SAME ${sys.toUpperCase()} CODE`;
    title = `Both variables carry ${sys} ${e.detail}`;
  } else if (e.type === 'text') {
    text = `TEXT MATCH ${Math.round((e.score ?? 0) * 100)}%`;
    title = 'Similarity of names, labels and standard names (normalized tokens)';
  } else if (e.type === 'cache') {
    text = `COMPUTED · ${e.status || '?'}`;
    title = `From the computed mapping file ${e.file}`;
  } else if (e.type === 'ai') {
    text = 'AI';
    title = e.detail || 'Suggested by iCARE-AI';
  } else if (e.type === 'manual') {
    text = 'MANUAL';
    title = e.detail || 'Chosen by you';
  } else if (e.type === 'warning') {
    text = '! ' + (e.detail || '');
    title = e.detail || '';
  } else if (e.type === 'label') {
    text = 'LABEL';
    title = 'Placed by its own label (no counterpart found)';
  }
  return (
    <span title={title} className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold tracking-wide ${EVIDENCE_STYLE[e.type] || EVIDENCE_STYLE.manual}`}>
      {text}
    </span>
  );
}

// ---- Searchable dropdown over variables ---------------------------------------

function matches(v: VarInfo, q: string): boolean {
  if (!q) return true;
  const hay = `${v.var_name} ${v.var_label} ${v.concept_name} ${v.concept_code} ${v.omop_id} ${v.visits || ''}`.toLowerCase();
  return q
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every(t => hay.includes(t));
}

function VariableRow({v, cohorts, showCohort, evidence, score}: {v: VarInfo; cohorts: string[]; showCohort: boolean; evidence?: Evidence[]; score?: number}) {
  return (
    <div className="flex items-start gap-2 w-full">
      {showCohort && <span className={`mt-0.5 shrink-0 px-1.5 py-0.5 rounded text-[10px] font-semibold border ${cohortColor(cohorts, v.cohort_id)}`}>{v.cohort_id}</span>}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="font-mono font-semibold text-sm truncate">{v.var_name}</span>
          <span className="text-[10px] uppercase tracking-wide text-base-content/50">{v.kind}</span>
          {unitVisitLines(v).length > 0 && <span className="text-[10px] text-base-content/60 bg-base-200 rounded px-1">{unitVisitLines(v).join(' · ')}</span>}
        </div>
        <div className="text-xs text-base-content/80 truncate">{v.var_label}</div>
        {v.concept_name && <div className="text-[11px] text-base-content/50 truncate">{v.concept_name}</div>}
        {edaLine(v) && <div className="text-[11px] text-base-content/50 truncate tabular-nums">{edaLine(v)}</div>}
        {categoryLine(v) && <div className="text-[11px] text-base-content/50 truncate tabular-nums">{categoryLine(v)}</div>}
        {v.equivalents && v.equivalents.length > 0 && (
          <div className="text-[11px] text-emerald-800 truncate">≈ {v.equivalents.map(e => `${e.var_name} [${e.cohort_id}]`).join(', ')}</div>
        )}
      </div>
      {evidence && (
        <span className="flex gap-1 shrink-0 flex-wrap justify-end max-w-[180px]">
          {evidence.slice(0, 3).map((e, i) => (
            <EvidenceBadge key={i} e={e} />
          ))}
        </span>
      )}
      {score !== undefined && <span className="text-xs tabular-nums text-base-content/50 w-9 text-right shrink-0">{score.toFixed(2)}</span>}
    </div>
  );
}

function VariableCombobox({
  cohorts,
  variables,
  value,
  onPick,
  placeholder,
  suggestions,
  restrictCohort,
  kindFilter,
  onClear
}: {
  cohorts: string[];
  variables: VarInfo[];
  value?: {cohort_id: string; var_name: string; var_label?: string} | null;
  onPick: (v: VarInfo, fromSuggestion?: Candidate) => void;
  placeholder: string;
  suggestions?: Candidate[];
  restrictCohort?: string;
  kindFilter?: 'numeric' | 'categorical';
  onClear?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState('');
  const box = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    setTimeout(() => input.current?.focus(), 0);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const pool = useMemo(() => {
    let list = restrictCohort ? variables.filter(v => v.cohort_id === restrictCohort) : variables;
    list = list.filter(v => v.kind !== 'other');
    if (kindFilter) list = list.filter(v => v.kind === kindFilter);
    return list.filter(v => matches(v, q));
  }, [variables, restrictCohort, kindFilter, q]);
  const suggested = (suggestions || []).filter(c => matches(c, q));
  const suggestedNames = new Set(suggested.map(c => c.var_name));
  const rest = pool.filter(v => !(restrictCohort && suggestedNames.has(v.var_name)));

  const grouped = useMemo(() => {
    const g: Record<string, VarInfo[]> = {};
    rest.forEach(v => (g[v.cohort_id] = g[v.cohort_id] || []).push(v));
    return g;
  }, [rest]);

  return (
    <div ref={box} className="relative">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className={`w-full text-left input input-bordered h-auto min-h-[2.75rem] py-1.5 flex items-center gap-2 ${value ? '' : 'text-base-content/50'}`}
      >
        {value ? (
          <span className="flex items-center gap-2 min-w-0 flex-1">
            {!restrictCohort && <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-semibold border ${cohortColor(cohorts, value.cohort_id)}`}>{value.cohort_id}</span>}
            <span className="font-mono font-semibold text-sm">{value.var_name}</span>
            {value.var_label && <span className="text-xs text-base-content/70 truncate">{value.var_label}</span>}
          </span>
        ) : (
          <span className="flex-1">{placeholder}</span>
        )}
        {value && onClear && (
          <span
            role="button"
            className="opacity-50 hover:opacity-100"
            onClick={e => {
              e.stopPropagation();
              onClear();
            }}
            title="Clear"
          >
            <X size={14} />
          </span>
        )}
        <ChevronDown size={16} className="opacity-60 shrink-0" />
      </button>
      {open && (
        <div className="absolute z-30 mt-1 w-full min-w-[520px] max-w-[720px] bg-base-100 border border-base-300 rounded-xl shadow-xl">
          <div className="p-2 border-b border-base-200">
            <input ref={input} className="input input-sm input-bordered w-full" placeholder="Type to filter: name, label, standard name, code, visit…" value={q} onChange={e => setQ(e.target.value)} />
          </div>
          <div className="max-h-80 overflow-y-auto">
            {suggested.length > 0 && (
              <div>
                <div className="px-3 py-1 text-[10px] uppercase tracking-wide text-base-content/50 bg-base-200/60">Suggested matches</div>
                {suggested.map(c => (
                  <button
                    key={`s-${c.cohort_id}-${c.var_name}`}
                    type="button"
                    className="w-full text-left px-3 py-2 hover:bg-base-200 border-b border-base-200/60"
                    onClick={() => {
                      onPick(c, c);
                      setOpen(false);
                      setQ('');
                    }}
                  >
                    <VariableRow v={c} cohorts={cohorts} showCohort={false} evidence={c.evidence} score={c.score} />
                  </button>
                ))}
              </div>
            )}
            {Object.entries(grouped).map(([cid, list]) => (
              <div key={cid}>
                <div className="px-3 py-1 text-[10px] uppercase tracking-wide text-base-content/50 bg-base-200/60">
                  {restrictCohort ? 'All variables' : cid} <span className="normal-case tracking-normal">({list.length})</span>
                </div>
                {list.slice(0, 400).map(v => (
                  <button
                    key={`${v.cohort_id}-${v.var_name}`}
                    type="button"
                    className="w-full text-left px-3 py-2 hover:bg-base-200 border-b border-base-200/60"
                    onClick={() => {
                      onPick(v);
                      setOpen(false);
                      setQ('');
                    }}
                  >
                    <VariableRow v={v} cohorts={cohorts} showCohort={false} />
                  </button>
                ))}
                {list.length > 400 && <div className="px-3 py-1 text-xs text-base-content/50">… {list.length - 400} more. Type to narrow down</div>}
              </div>
            ))}
            {pool.length === 0 && suggested.length === 0 && <div className="p-3 text-sm text-base-content/50">No variables match.</div>}
            <div className="px-3 py-1.5 text-[11px] text-base-content/50 border-t border-base-200">
              {kindFilter === 'categorical'
                ? 'Only categorical variables are listed (categories declared in the dictionary).'
                : kindFilter === 'numeric'
                  ? 'Only numeric variables are listed (numeric type in the dictionary).'
                  : 'Free-text, identifier and date variables are not listed: they cannot be analysed here.'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---- Value map editor (categorical) --------------------------------------------

function ValueMapEditor({
  hv,
  cohorts,
  categories,
  clusters,
  onChange,
  onSuggest,
  onAi,
  busy
}: {
  hv: HVar;
  cohorts: string[];
  categories: Record<string, {value: string; label: string}[]>;
  clusters: ValueCluster[] | null;
  onChange: (vm: Record<string, Record<string, string>>) => void;
  onSuggest: () => void;
  onAi: () => void;
  busy: string | null;
}) {
  const memberCohorts = cohorts.filter(c => hv.members[c]?.var_name);
  const harmonized = useMemo(() => {
    const seen: string[] = [];
    memberCohorts.forEach(c =>
      Object.values(hv.value_map[c] || {}).forEach(h => {
        if (h && !seen.includes(h)) seen.push(h);
      })
    );
    return seen;
  }, [hv.value_map, memberCohorts]);
  const [newValue, setNewValue] = useState('');

  const rawsFor = (c: string, h: string) => Object.entries(hv.value_map[c] || {}).filter(([, v]) => v === h).map(([k]) => k);
  const unmapped = (c: string) => (categories[c] || []).filter(cat => !(hv.value_map[c] || {})[cat.value]);

  const setRaw = (c: string, raw: string, h: string | null) => {
    const vm = {...hv.value_map, [c]: {...(hv.value_map[c] || {})}};
    if (h) vm[c][raw] = h;
    else delete vm[c][raw];
    onChange(vm);
  };
  const renameHarmonized = (from: string, to: string) => {
    if (!to.trim() || from === to) return;
    const vm: Record<string, Record<string, string>> = {};
    memberCohorts.forEach(c => {
      vm[c] = {};
      Object.entries(hv.value_map[c] || {}).forEach(([k, v]) => (vm[c][k] = v === from ? to : v));
    });
    onChange(vm);
  };
  const removeHarmonized = (h: string) => {
    const vm: Record<string, Record<string, string>> = {};
    memberCohorts.forEach(c => {
      vm[c] = {};
      Object.entries(hv.value_map[c] || {}).forEach(([k, v]) => {
        if (v !== h) vm[c][k] = v;
      });
    });
    onChange(vm);
  };
  const clusterEvidence = (h: string) => clusters?.find(cl => cl.harmonized.toLowerCase() === h.toLowerCase())?.evidence || [];

  return (
    <div className="mt-4 ml-3 pl-4 border-l-2 border-base-300 max-w-4xl">
      <div className="flex items-center gap-2 flex-wrap">
        <h4 className="font-semibold text-sm">Value mapping</h4>
        <span className="text-xs text-base-content/50">one row per harmonized value; assign each cohort&rsquo;s raw values to it</span>
        <div className="ml-auto flex gap-2">
          <button className="btn btn-xs btn-outline gap-1" onClick={onSuggest} disabled={busy !== null}>
            <Zap size={12} /> {busy === 'values' ? 'Suggesting…' : 'Suggest from codes, labels & computed mappings'}
          </button>
          <button className="btn btn-xs btn-outline btn-warning gap-1" onClick={onAi} disabled={busy !== null}>
            <Sparkles size={12} /> {busy === 'ai-values' ? 'Asking…' : 'Ask iCARE-AI'}
          </button>
        </div>
      </div>
      <div className="overflow-x-auto mt-2 rounded-lg border border-base-300">
        <table className="table table-sm">
          <thead>
            <tr>
              <th className="w-56">Harmonized value</th>
              {memberCohorts.map(c => (
                <th key={c}>
                  <span className={`px-1.5 py-0.5 rounded border text-[10px] ${cohortColor(cohorts, c)}`}>{c}</span> <span className="font-mono text-xs">{hv.members[c].var_name}</span>
                </th>
              ))}
              <th className="w-8"></th>
            </tr>
          </thead>
          <tbody>
            {harmonized.map(h => (
              <tr key={h}>
                <td className="align-top">
                  <input className="input input-xs input-bordered w-full font-semibold" defaultValue={h} onBlur={e => renameHarmonized(h, e.target.value)} />
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {clusterEvidence(h)
                      .slice(0, 3)
                      .map((e, i) => (
                        <EvidenceBadge key={i} e={e} />
                      ))}
                  </div>
                </td>
                {memberCohorts.map(c => (
                  <td key={c} className="align-top">
                    <div className="flex flex-wrap gap-1">
                      {rawsFor(c, h).map(raw => {
                        const cat = (categories[c] || []).find(x => x.value === raw);
                        return (
                          <span key={raw} className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-base-200 text-xs" title={cat?.label}>
                            <span className="font-mono">{raw}</span>
                            {cat?.label && cat.label !== raw && <span className="opacity-60">{cat.label}</span>}
                            <button onClick={() => setRaw(c, raw, null)} className="opacity-50 hover:opacity-100">
                              <X size={10} />
                            </button>
                          </span>
                        );
                      })}
                      {unmapped(c).length > 0 && (
                        <select className="select select-xs select-bordered max-w-[160px]" value="" onChange={e => e.target.value && setRaw(c, e.target.value, h)}>
                          <option value="">+ add value</option>
                          {unmapped(c).map(cat => (
                            <option key={cat.value} value={cat.value}>
                              {cat.value}
                              {cat.label && cat.label !== cat.value ? ` (${cat.label})` : ''}
                            </option>
                          ))}
                        </select>
                      )}
                      {rawsFor(c, h).length === 0 && unmapped(c).length === 0 && <span className="text-xs text-base-content/40">none</span>}
                    </div>
                  </td>
                ))}
                <td className="align-top">
                  <button className="btn btn-ghost btn-xs" onClick={() => removeHarmonized(h)} title="Remove this harmonized value">
                    <Trash2 size={12} />
                  </button>
                </td>
              </tr>
            ))}
            <tr>
              <td colSpan={memberCohorts.length + 2}>
                <div className="flex items-center gap-2">
                  <input
                    className="input input-xs input-bordered w-56"
                    placeholder="new harmonized value…"
                    value={newValue}
                    onChange={e => setNewValue(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter' && newValue.trim()) {
                        const c = memberCohorts.find(cc => unmapped(cc).length > 0);
                        if (c) setRaw(c, unmapped(c)[0].value, newValue.trim());
                        setNewValue('');
                      }
                    }}
                  />
                  <span className="text-xs text-base-content/50">Enter to add (starts with the first unassigned raw value; adjust afterwards)</span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex flex-wrap gap-4 text-xs">
        {memberCohorts.map(c => (
          <div key={c}>
            <span className="font-semibold">{c} unassigned → missing:</span>{' '}
            {unmapped(c).length === 0 ? <span className="text-emerald-700">none</span> : unmapped(c).map(cat => cat.value).join(', ')}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---- Unit editor (numeric) ---------------------------------------------------

function UnitEditor({hv, cohorts, onChange}: {hv: HVar; cohorts: string[]; onChange: (hv: HVar) => void}) {
  const memberCohorts = cohorts.filter(c => hv.members[c]?.var_name);
  const units = memberCohorts.map(c => (hv.members[c].unit || '').trim()).filter(Boolean);
  const mismatch = new Set(units.map(u => u.toLowerCase())).size > 1;
  return (
    <div className="mt-4 ml-3 pl-4 border-l-2 border-base-300 max-w-3xl">
      <div className="flex items-center gap-2">
        <h4 className="font-semibold text-sm">Units</h4>
        {mismatch ? (
          <span className="text-xs text-rose-700 inline-flex items-center gap-1">
            <AlertTriangle size={12} /> the cohorts declare different units: enter a conversion factor
          </span>
        ) : (
          <span className="text-xs text-base-content/50">declared units agree (or are missing); factors are optional</span>
        )}
      </div>
      <div className="mt-2 grid sm:grid-cols-2 gap-2">
        <label className="text-xs">
          Harmonized unit
          <input className="input input-xs input-bordered w-full" value={hv.unit || ''} placeholder="e.g. kg" onChange={e => onChange({...hv, unit: e.target.value})} />
        </label>
        {memberCohorts.map(c => {
          const conv = hv.unit_conversion[c] || {factor: null, from: hv.members[c].unit || '', to: hv.unit || ''};
          return (
            <div key={c} className="text-xs">
              <span className={`px-1.5 py-0.5 rounded border text-[10px] ${cohortColor(cohorts, c)}`}>{c}</span> <span className="font-mono">{hv.members[c].var_name}</span> in{' '}
              <b>{hv.members[c].unit || 'unknown unit'}</b>
              <div className="flex items-center gap-1 mt-1">
                × factor
                <input
                  type="number"
                  step="any"
                  className="input input-xs input-bordered w-28"
                  value={conv.factor ?? ''}
                  placeholder="1"
                  onChange={e =>
                    onChange({
                      ...hv,
                      unit_conversion: {
                        ...hv.unit_conversion,
                        [c]: {factor: e.target.value === '' ? null : parseFloat(e.target.value), from: hv.members[c].unit || '', to: hv.unit || ''}
                      }
                    })
                  }
                />
                <span className="text-base-content/50">→ {hv.unit || '?'}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---- The workbench ------------------------------------------------------------

export default function MappingWorkbench({cohorts, roles, mapping, roleAssignments, onMappingChange, onRolesChange, userEmail}: Props) {
  const multi = cohorts.length > 1;
  const [variables, setVariables] = useState<VarInfo[]>([]);
  const [loadingVars, setLoadingVars] = useState(false);
  const [suggestions, setSuggestions] = useState<Record<string, Record<string, Candidate[]>>>({}); // hvar -> cohort -> candidates
  const [clusters, setClusters] = useState<Record<string, ValueCluster[]>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [forceOpen, setForceOpen] = useState<Record<string, boolean>>({}); // harmonized_name -> show editor despite type mismatch
  const [error, setError] = useState<string | null>(null);
  const [cacheFiles, setCacheFiles] = useState<{filename: string; source: string; target: string; generated_at: string; size_kb: number}[]>([]);
  const [useCache, setUseCache] = useState<string[]>(mapping.sources || []);
  const [cacheOpen, setCacheOpen] = useState(false);
  const cacheBox = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!cacheOpen) return;
    const onDoc = (e: MouseEvent) => {
      if (cacheBox.current && !cacheBox.current.contains(e.target as Node)) setCacheOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [cacheOpen]);
  const [showProvenance, setShowProvenance] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setLoadingVars(true);
    fetchAllVariables(cohorts)
      .then(r => setVariables(r.variables))
      .catch(e => setError(e.message))
      .finally(() => setLoadingVars(false));
    if (multi) {
      fetchCachedMappings(cohorts).then(r => setCacheFiles(r.files)).catch(() => setCacheFiles([]));
    }
  }, [cohorts, multi]);

  const categoriesOf = useCallback(
    (cohort: string, varName: string) => {
      const v = variables.find(x => x.cohort_id === cohort && x.var_name === varName);
      return (v?.categories || []).map(c => ({value: c.value, label: c.label}));
    },
    [variables]
  );

  const replaceHVar = useCallback(
    (oldName: string | undefined, next: HVar, roleKey: string) => {
      const vars = mapping.variables.filter(x => x.harmonized_name !== oldName && x.harmonized_name !== next.harmonized_name);
      vars.push(next);
      onMappingChange({...mapping, variables: vars, sources: useCache});
      onRolesChange({...roleAssignments, [roleKey]: next.harmonized_name});
    },
    [mapping, onMappingChange, onRolesChange, roleAssignments, useCache]
  );

  const fetchSuggestions = (hv: HVar, files: string[]) => {
    if (!multi) return;
    const anchorCohort = Object.keys(hv.members)[0];
    const targets = cohorts.filter(c => c !== anchorCohort);
    setBusy(`suggest:${hv.harmonized_name}`);
    suggestMatches({cohort_id: anchorCohort, var_name: hv.members[anchorCohort].var_name}, targets, files)
      .then(r => setSuggestions(prev => ({...prev, [hv.harmonized_name]: r.candidates})))
      .catch(e => setError(e.message))
      .finally(() => setBusy(null));
  };

  // Pick the anchor for a role: (re)creates the harmonized variable.
  const pickAnchor = (roleKey: string, v: VarInfo) => {
    const role = roles.find(r => r.key === roleKey);
    if (role?.kind && v.kind !== role.kind) {
      setError(`"${role.label}" needs a ${role.kind} variable; ${v.var_name} is ${v.kind}.`);
      return;
    }
    setError(null);
    const old = roleAssignments[roleKey];
    let created = newHVar(v);
    const taken = new Set(mapping.variables.filter(x => x.harmonized_name !== old).map(x => x.harmonized_name));
    let name = created.harmonized_name;
    let n = 2;
    while (taken.has(name)) name = `${created.harmonized_name}_${n++}`;
    created = {...created, harmonized_name: name};
    replaceHVar(old, created, roleKey);
    fetchSuggestions(created, useCache);
  };

  const clearRole = (roleKey: string) => {
    const old = roleAssignments[roleKey];
    onMappingChange({...mapping, variables: mapping.variables.filter(x => x.harmonized_name !== old), sources: useCache});
    const next = {...roleAssignments};
    delete next[roleKey];
    onRolesChange(next);
  };

  const updateHVar = (roleKey: string, next: HVar) => replaceHVar(roleAssignments[roleKey], next, roleKey);

  const linkMember = (roleKey: string, hv: HVar, cohortId: string, v: VarInfo, cand?: Candidate) => {
    const next: HVar = {
      ...hv,
      members: {...hv.members, [cohortId]: {var_name: v.var_name, var_label: v.var_label, unit: v.units, kind: v.kind}},
      evidence: [
        ...hv.evidence,
        ...(cand
          ? cand.evidence.filter(e => e.type !== 'warning').map(e => ({...e, detail: e.detail ? `${e.detail} (${cohortId})` : undefined}))
          : [{type: 'manual' as const, detail: `${v.var_name} [${cohortId}] picked from the list`}])
      ]
    };
    // cached value_mapping arrives with the candidate: pre-fill
    const cacheEv = cand?.evidence.find(e => e.type === 'cache' && e.value_mapping);
    if (cacheEv?.value_mapping && hv.type === 'categorical') {
      const vm = {...next.value_map};
      const anchorCohort = Object.keys(hv.members)[0];
      const anchorCats = new Set(categoriesOf(anchorCohort, hv.members[anchorCohort].var_name).map(x => x.value));
      const candCats = new Set(v.categories.map(x => x.value));
      const src = cacheEv.value_mapping.source || {};
      const tgt = cacheEv.value_mapping.target || {};
      const assign = (cohort: string, m: Record<string, string>, cats: Set<string>) => {
        const out: Record<string, string> = {...(vm[cohort] || {})};
        Object.entries(m).forEach(([k, val]) => {
          if (cats.has(k) && val && val !== 'unknown') out[k] = val;
        });
        vm[cohort] = out;
      };
      if (Object.keys(src).some(k => anchorCats.has(k))) {
        assign(anchorCohort, src, anchorCats);
        assign(cohortId, tgt, candCats);
      } else {
        assign(anchorCohort, tgt, anchorCats);
        assign(cohortId, src, candCats);
      }
      next.value_map = vm;
    }
    updateHVar(roleKey, next);
  };

  const unlinkMember = (roleKey: string, hv: HVar, cohortId: string) => {
    const members = {...hv.members};
    delete members[cohortId];
    const vm = {...hv.value_map};
    delete vm[cohortId];
    updateHVar(roleKey, {...hv, members, value_map: vm});
  };

  const doSuggestValues = (roleKey: string, hv: HVar) => {
    const members: Record<string, string> = {};
    cohorts.forEach(c => {
      if (hv.members[c]?.var_name) members[c] = hv.members[c].var_name;
    });
    setBusy('values');
    suggestValues(members, useCache)
      .then(r => {
        setClusters(prev => ({...prev, [hv.harmonized_name]: r.clusters}));
        const vm: Record<string, Record<string, string>> = {};
        r.clusters.forEach(cl => {
          Object.entries(cl.sources).forEach(([c, raws]) => {
            vm[c] = vm[c] || {};
            raws.forEach(raw => (vm[c][raw] = cl.harmonized));
          });
        });
        updateHVar(roleKey, {...hv, value_map: vm, evidence: [...hv.evidence, {type: 'manual', detail: 'value map suggested from codes, labels and computed mappings'}]});
      })
      .catch(e => setError(e.message))
      .finally(() => setBusy(null));
  };

  const doAiValues = (roleKey: string, hv: HVar) => {
    const vars: Record<string, any> = {};
    cohorts.forEach(c => {
      if (hv.members[c]?.var_name) vars[c] = {var_name: hv.members[c].var_name, var_label: hv.members[c].var_label, categories: categoriesOf(c, hv.members[c].var_name)};
    });
    setBusy('ai-values');
    aiSuggest({task: 'values', variables: vars})
      .then(r => {
        const vmap = r.result?.value_map || {};
        const vm: Record<string, Record<string, string>> = {};
        Object.entries(vmap).forEach(([c, m]: [string, any]) => {
          vm[c] = {};
          Object.entries(m || {}).forEach(([k, val]) => {
            if (val) vm[c][k] = String(val);
          });
        });
        updateHVar(roleKey, {...hv, value_map: vm, evidence: [...hv.evidence, {type: 'ai', detail: `value map proposed by ${r.model}`}]});
      })
      .catch(e => setError(e.message))
      .finally(() => setBusy(null));
  };

  const doAiMatch = (hv: HVar) => {
    const anchorCohort = Object.keys(hv.members)[0];
    const current = suggestions[hv.harmonized_name] || {};
    setBusy('ai-match');
    aiSuggest({
      task: 'match',
      anchor: {cohort_id: anchorCohort, ...hv.members[anchorCohort]},
      candidates: Object.fromEntries(Object.entries(current).map(([c, list]) => [c, list.slice(0, 12).map(x => ({var_name: x.var_name, var_label: x.var_label, concept_name: x.concept_name, units: x.units}))]))
    })
      .then(r => {
        const m = r.result?.matches || {};
        const next = {...current};
        Object.entries(m).forEach(([c, pick]: [string, any]) => {
          if (!pick?.var_name || !next[c]) return;
          next[c] = next[c]
            .map(cand => (cand.var_name === pick.var_name ? {...cand, evidence: [...cand.evidence, {type: 'ai' as const, detail: pick.reason || 'AI suggestion'}], score: Math.max(cand.score || 0, 0.9)} : cand))
            .sort((a, b) => (b.score || 0) - (a.score || 0));
        });
        setSuggestions(prev => ({...prev, [hv.harmonized_name]: next}));
      })
      .catch(e => setError(e.message))
      .finally(() => setBusy(null));
  };

  const toggleCache = (f: string) => {
    const next = useCache.includes(f) ? useCache.filter(x => x !== f) : [...useCache, f];
    setUseCache(next);
    onMappingChange({...mapping, sources: next});
    mapping.variables.forEach(hv => fetchSuggestions(hv, next));
  };

  return (
    <div className="flex flex-col gap-4">
      {error && (
        <div className="alert alert-error text-sm py-2">
          <span>{error}</span>
          <button className="btn btn-ghost btn-xs" onClick={() => setError(null)}>✕</button>
        </div>
      )}
      {loadingVars && <div className="text-sm text-base-content/50">Loading the variables of {cohorts.join(', ')}…</div>}

      {/* Computed mappings toolbar (multi-cohort only) */}
      {multi && (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <div className="relative" ref={cacheBox}>
            <button className="btn btn-sm btn-outline gap-1" onClick={() => setCacheOpen(o => !o)}>
              <Database size={14} /> Computed mappings {useCache.length > 0 && <span className="badge badge-sm badge-primary">{useCache.length} in use</span>}
              <ChevronDown size={14} />
            </button>
            {cacheOpen && (
              <div className="absolute z-20 mt-1 w-[520px] max-h-80 overflow-y-auto bg-base-100 border border-base-300 rounded-lg shadow-xl p-2">
                <div className="flex items-center justify-between px-1 pb-1 mb-1 border-b border-base-200">
                  <span className="text-xs font-semibold">Computed mappings for these cohorts</span>
                  <button type="button" className="btn btn-xs btn-ghost gap-1" onClick={() => setCacheOpen(false)}>
                    <X size={12} /> Close
                  </button>
                </div>
                {cacheFiles.length === 0 && <div className="text-xs text-base-content/50 p-2">No computed mappings for these cohorts. Generate them from the Mapping page.</div>}
                {cacheFiles.map(f => (
                  <label key={f.filename} className="flex items-start gap-2 p-1.5 hover:bg-base-200 rounded cursor-pointer">
                    <input type="checkbox" className="checkbox checkbox-xs mt-0.5" checked={useCache.includes(f.filename)} onChange={() => toggleCache(f.filename)} />
                    <span className="text-xs">
                      <span className="font-semibold">
                        {f.source} → {f.target}
                      </span>{' '}
                      · {f.generated_at.slice(0, 10)}
                      <div className="font-mono text-[10px] text-base-content/50 break-all">{f.filename}</div>
                    </span>
                  </label>
                ))}
                <div className="text-[11px] text-base-content/50 p-2 border-t border-base-200 mt-1">
                  Ticked computed mappings become an evidence source: their rows appear as <span className="px-1 rounded bg-violet-600 text-white">COMPUTED</span> badges in the suggestions and their value mappings pre-fill the value table.
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* One card per role */}
      {roles.map(role => {
        const name = roleAssignments[role.key];
        const hv = mapping.variables.find(x => x.harmonized_name === name) || null;
        const anchorCohort = hv ? Object.keys(hv.members)[0] : null;
        const complete = !!hv && cohorts.every(c => hv.members[c]?.var_name);
        const roleSugg = hv ? suggestions[hv.harmonized_name] || {} : {};
        const suggesting = !!hv && busy === `suggest:${hv.harmonized_name}`;
        return (
          <section key={role.key} className="rounded-xl border border-base-300 bg-base-100 p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[11px] uppercase tracking-wide text-base-content/60 font-semibold">{role.label}</span>
              {role.optional && <span className="text-[11px] text-base-content/50">(optional)</span>}
              {role.kind && <span className="text-[11px] text-base-content/50">· {role.kind} variable</span>}
              {role.hint && !hv && <span className="text-xs text-base-content/50 ml-2">{role.hint}</span>}
              {hv && (complete ? <Check size={14} className="text-emerald-600 ml-auto" /> : <AlertTriangle size={14} className="text-amber-600 ml-auto" />)}
            </div>

            <div className="grid gap-3" style={{gridTemplateColumns: multi && hv ? `repeat(${Math.min(cohorts.length, 3)}, minmax(0, 1fr))` : '1fr'}}>
              {!hv ? (
                <VariableCombobox cohorts={cohorts} variables={variables} value={null} onPick={v => pickAnchor(role.key, v)} placeholder={multi ? 'Choose a variable in any cohort to start from…' : 'Choose a variable…'} kindFilter={role.kind} />
              ) : (
                cohorts.map(c => {
                  const m = hv.members[c];
                  const isAnchor = c === anchorCohort;
                  return (
                    <div key={c}>
                      {multi && (
                        <div className="flex items-center gap-1 mb-1">
                          <span className={`px-1.5 py-0.5 rounded border text-[10px] font-semibold ${cohortColor(cohorts, c)}`}>{c}</span>
                          {isAnchor && <span className="text-[10px] uppercase tracking-wide text-base-content/40">anchor</span>}
                          {!isAnchor && !m?.var_name && (roleSugg[c]?.length || 0) > 0 && (
                            <button className="btn btn-ghost btn-xs gap-1 ml-auto" onClick={() => doAiMatch(hv)} disabled={busy !== null} title="Ask the local model which candidate matches best">
                              <Sparkles size={11} /> {busy === 'ai-match' ? 'Asking…' : 'Ask iCARE-AI'}
                            </button>
                          )}
                        </div>
                      )}
                      <VariableCombobox
                        cohorts={cohorts}
                        variables={variables}
                        value={m?.var_name ? {cohort_id: c, var_name: m.var_name, var_label: m.var_label} : null}
                        onPick={(v, cand) => (isAnchor ? pickAnchor(role.key, v) : linkMember(role.key, hv, c, v, cand))}
                        placeholder={suggesting ? 'Looking for matches…' : (roleSugg[c]?.length || 0) > 0 ? `${roleSugg[c].length} suggested, choose…` : 'Choose the matching variable…'}
                        suggestions={isAnchor ? undefined : roleSugg[c]}
                        restrictCohort={c}
                        kindFilter={isAnchor ? role.kind : undefined}
                        onClear={isAnchor ? () => clearRole(role.key) : m?.var_name ? () => unlinkMember(role.key, hv, c) : undefined}
                      />
                      {m?.var_name &&
                        (() => {
                          const info = variables.find(x => x.cohort_id === c && x.var_name === m.var_name);
                          if (!info) return null;
                          const lines = [...unitVisitLines(info), edaLine(info), categoryLine(info)].filter(Boolean);
                          if (lines.length === 0) return null;
                          return (
                            <div className="mt-1 px-1 text-[11px] text-base-content/60 tabular-nums leading-snug space-y-0.5">
                              {lines.map((l, i) => (
                                <div key={i} className={i < unitVisitLines(info).length ? 'text-base-content/70' : ''}>{l}</div>
                              ))}
                            </div>
                          );
                        })()}
                    </div>
                  );
                })
              )}
            </div>

            {hv && (
              <div className="mt-3">

                {(() => {
                  // Members whose dictionary type differs from the anchor's: a mapping
                  // between a categorical and a numeric variable is unlikely to be right.
                  const kindOf = (c: string) =>
                    (hv.members[c]?.kind as string | undefined) || variables.find(x => x.cohort_id === c && x.var_name === hv.members[c]?.var_name)?.kind || '';
                  const withKind = cohorts.filter(c => hv.members[c]?.var_name && kindOf(c));
                  const anchorKind = anchorCohort ? kindOf(anchorCohort) : '';
                  const mismatched = withKind.filter(c => c !== anchorCohort && anchorKind && kindOf(c) !== anchorKind);
                  if (mismatched.length === 0) return null;
                  return (
                    <div className="mt-3 rounded-xl border border-rose-300 bg-rose-50 text-rose-900 p-3 text-sm">
                      <div className="font-semibold mb-1">This mapping is unlikely to be correct due to a type difference</div>
                      <ul className="list-disc ml-5 space-y-0.5">
                        {mismatched.map(c => (
                          <li key={c}>
                            <span className="font-mono">{hv.members[anchorCohort!].var_name}</span> [{anchorCohort}] is {anchorKind} while{' '}
                            <span className="font-mono">{hv.members[c].var_name}</span> [{c}] is {kindOf(c)}
                          </li>
                        ))}
                      </ul>
                      {!forceOpen[hv.harmonized_name] && (
                        <button className="btn btn-xs btn-outline btn-error mt-2" onClick={() => setForceOpen(prev => ({...prev, [hv.harmonized_name]: true}))}>
                          Show the {hv.type === 'categorical' ? 'value mapping' : 'unit settings'} anyway
                        </button>
                      )}
                    </div>
                  );
                })()}
                {(() => {
                  const kindOf = (c: string) =>
                    (hv.members[c]?.kind as string | undefined) || variables.find(x => x.cohort_id === c && x.var_name === hv.members[c]?.var_name)?.kind || '';
                  const anchorKind = anchorCohort ? kindOf(anchorCohort) : '';
                  const hasMismatch = cohorts.some(c => hv.members[c]?.var_name && c !== anchorCohort && anchorKind && kindOf(c) && kindOf(c) !== anchorKind);
                  const editorsVisible = !hasMismatch || !!forceOpen[hv.harmonized_name];
                  if (!editorsVisible) return null;
                  return (
                    <>
                      {complete && multi && hv.type === 'categorical' && (
                        <ValueMapEditor
                          hv={hv}
                          cohorts={cohorts}
                          categories={Object.fromEntries(cohorts.map(c => [c, hv.members[c]?.var_name ? categoriesOf(c, hv.members[c].var_name) : []]))}
                          clusters={clusters[hv.harmonized_name] || null}
                          onChange={vm => updateHVar(role.key, {...hv, value_map: vm})}
                          onSuggest={() => doSuggestValues(role.key, hv)}
                          onAi={() => doAiValues(role.key, hv)}
                          busy={busy}
                        />
                      )}
                      {complete && multi && hv.type === 'numeric' && <UnitEditor hv={hv} cohorts={cohorts} onChange={next => updateHVar(role.key, next)} />}
                    </>
                  );
                })()}
                {!multi && hv.type === 'categorical' && anchorCohort && (
                  <div className="mt-2 text-xs text-base-content/60">
                    Categories:{' '}
                    {categoriesOf(anchorCohort, hv.members[anchorCohort].var_name)
                      .map(c => `${c.value}${c.label && c.label !== c.value ? ` (${c.label})` : ''}`)
                      .join(', ') || 'none'}
                  </div>
                )}

                <div className="mt-3">
                  <button
                    type="button"
                    className="text-xs text-base-content/60 hover:text-base-content inline-flex items-center gap-1"
                    onClick={() => setShowProvenance(prev => ({...prev, [hv.harmonized_name]: !prev[hv.harmonized_name]}))}
                  >
                    <ChevronDown size={12} className={showProvenance[hv.harmonized_name] ? 'rotate-180 transition-transform' : 'transition-transform'} />
                    {showProvenance[hv.harmonized_name] ? 'Hide' : 'Show'} how this mapping will be printed under the figures
                  </button>
                  {showProvenance[hv.harmonized_name] && (
                    <div className="mt-2 rounded-lg bg-base-200 p-2.5">
                      <div className="font-mono text-xs break-words">{provenanceLine(hv)}</div>
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {hv.evidence.slice(-6).map((e, i) => (
                          <EvidenceBadge key={i} e={e} />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}
