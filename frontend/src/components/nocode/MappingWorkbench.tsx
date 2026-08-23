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
import {AlertTriangle, Check, ChevronDown, Database, Edit2, HelpCircle, Trash2, X, Zap} from 'react-feather';
import {SparklesIcon as Sparkles} from '@/components/Icons';
import {
  Candidate,
  Evidence,
  HVar,
  MappingSpec,
  ValueCluster,
  VarInfo,
  aiName,
  aiSuggest,
  categoryLine,
  MISSING_KEY,
  MISSING_LABEL,
  displayRaw,
  nameSuffix,
  withSuffix,
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

// One hue per cohort, by its position in the selection. Kept away from the
// platform's own colours: no light blues (the accent), no indigo/violet (the
// primary buttons), no greens (the completed steps), no black.
const COHORT_HUES = [
  'bg-amber-100 text-amber-900 border-amber-300',
  'bg-rose-100 text-rose-900 border-rose-300',
  'bg-fuchsia-100 text-fuchsia-900 border-fuchsia-300',
  'bg-orange-100 text-orange-900 border-orange-300',
  'bg-stone-200 text-stone-800 border-stone-400',
  'bg-red-100 text-red-900 border-red-300'
];
export const cohortColor = (cohorts: string[], id: string) => COHORT_HUES[Math.max(0, cohorts.indexOf(id)) % COHORT_HUES.length];

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
  busy,
  missingPct
}: {
  hv: HVar;
  cohorts: string[];
  categories: Record<string, {value: string; label: string}[]>;
  clusters: ValueCluster[] | null;
  onChange: (vm: Record<string, Record<string, string>>, manualMove?: boolean) => void;
  onSuggest: () => void;
  onAi: () => void;
  busy: string | null;
  missingPct?: Record<string, number | null | undefined>;
}) {
  const memberCohorts = cohorts.filter(c => hv.members[c]?.var_name);
  const [infoOpen, setInfoOpen] = useState(false);
  // Cells with an open "+ add" picker (key: cohort|row). A filled cell hides
  // its picker behind a small + so the table stays readable.
  const [openAdd, setOpenAdd] = useState<Record<string, boolean>>({});
  // Has the mapping been worked on (a suggestion run, iCARE-AI asked, a
  // computed mapping applied, or a manual move)? Until then the two buttons
  // shimmer to invite a click.
  const touched = !!hv.value_map_edited || hv.evidence.some(e => (e.detail || '').startsWith('value map') || (e.type === 'cache' && !!e.value_mapping));

  // Model: every recorded value of every cohort is always present in value_map,
  // either assigned to a harmonized value (row) or mapped to "" = excluded.
  // Raw values that are not in the map yet get their own row, named after their
  // label. The MISSING_KEY entry is not a value of the table: it carries the
  // policy for empty/coded-missing values ("" = exclude, MISSING_LABEL = keep
  // as one category), see MissingPolicy below.
  useEffect(() => {
    let changed = false;
    const vm: Record<string, Record<string, string>> = {};
    memberCohorts.forEach(c => {
      vm[c] = {...(hv.value_map[c] || {})};
      (categories[c] || []).forEach(cat => {
        if (!(cat.value in vm[c])) {
          vm[c][cat.value] = (cat.label && cat.label.trim()) || cat.value;
          changed = true;
        }
      });
      if (!(MISSING_KEY in vm[c])) {
        vm[c][MISSING_KEY] = '';
        changed = true;
      }
    });
    if (changed) onChange(vm);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hv.members, categories]);

  const recorded = (c: string) => Object.entries(hv.value_map[c] || {}).filter(([k]) => k !== MISSING_KEY);
  const harmonized = useMemo(() => {
    const seen: string[] = [];
    memberCohorts.forEach(c =>
      recorded(c).forEach(([, h]) => {
        if (h && !seen.includes(h)) seen.push(h);
      })
    );
    return seen;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hv.value_map, memberCohorts]);

  const labelOf = (c: string, raw: string) => (categories[c] || []).find(x => x.value === raw)?.label;
  const rawsFor = (c: string, h: string) => recorded(c).filter(([, v]) => v === h).map(([k]) => k);
  const missingOf = (c: string) => recorded(c).filter(([, v]) => v === '').map(([k]) => k);
  const elsewhere = (c: string, h: string) => recorded(c).filter(([, v]) => v !== h).map(([k, v]) => ({raw: k, where: v}));

  const setRaw = (c: string, raw: string, h: string) => {
    const vm = {...hv.value_map, [c]: {...(hv.value_map[c] || {}), [raw]: h}};
    onChange(vm, true);
  };
  const renameHarmonized = (from: string, to: string) => {
    const name = to.trim();
    if (!name || from === name) return;
    const vm: Record<string, Record<string, string>> = {};
    memberCohorts.forEach(c => {
      vm[c] = {};
      Object.entries(hv.value_map[c] || {}).forEach(([k, v]) => (vm[c][k] = v === from ? name : v));
    });
    onChange(vm);
  };
  const dropRow = (h: string) => {
    // Every raw value of the row is then excluded from the analysis.
    const vm: Record<string, Record<string, string>> = {};
    memberCohorts.forEach(c => {
      vm[c] = {};
      Object.entries(hv.value_map[c] || {}).forEach(([k, v]) => (vm[c][k] = v === h ? '' : v));
    });
    onChange(vm, true);
  };
  const splitOut = (c: string, raw: string) => setRaw(c, raw, (labelOf(c, raw) && labelOf(c, raw)!.trim()) || raw);

  // Evidence is attached to the VALUES of a row (via the suggestion clusters),
  // so renaming the harmonized value does not lose it.
  const rowEvidence = (h: string): Evidence[] => {
    if (!clusters) return [];
    const out: Evidence[] = [];
    clusters.forEach(cl => {
      const overlaps = memberCohorts.some(c => (cl.sources[c] || []).some(raw => rawsFor(c, h).includes(raw)));
      if (overlaps) cl.evidence.forEach(e => {
        if (e.type !== 'label' && !out.some(x => x.type === e.type && x.detail === e.detail && x.file === e.file)) out.push(e);
      });
    });
    return out;
  };

  const anyExcluded = memberCohorts.some(c => missingOf(c).length > 0);

  return (
    <div className="mt-4 ml-3 pl-4 border-l-2 border-base-300 max-w-4xl">
      <div className="flex items-center gap-2 flex-wrap">
        <h4 className="font-semibold text-sm">Value mapping</h4>
        <span className="text-xs text-base-content/50">every raw value of every cohort is listed; move values between rows to align them</span>
        <div className="ml-auto flex gap-2">
          <button className={`btn btn-sm btn-outline gap-1.5 ${!touched && busy === null ? 'shimmer-warm' : ''}`} onClick={onSuggest} disabled={busy !== null}>
            <Zap size={14} /> {busy === 'values' ? 'Suggesting…' : 'Suggest from codes, labels & computed mappings'}
          </button>
          <button className={`btn btn-sm btn-outline btn-warning gap-1.5 ${!touched && busy === null ? 'shimmer-warm' : ''}`} onClick={onAi} disabled={busy !== null}>
            <Sparkles size={14} /> {busy === 'ai-values' ? 'Asking…' : 'Ask iCARE-AI'}
          </button>
        </div>
      </div>
      <div className="overflow-x-auto mt-2 rounded-lg border border-base-300">
        <table className="table table-sm">
          <thead>
            <tr>
              <th className="w-56">
                <span className="inline-flex items-center gap-1">
                  Harmonized value
                  <button type="button" className="text-base-content/50 hover:text-base-content" onClick={() => setInfoOpen(o => !o)} title="What is this?">
                    <HelpCircle size={13} />
                  </button>
                </span>
                {infoOpen && (
                  <div className="mt-1 font-normal normal-case text-[11px] leading-snug text-base-content/70 bg-base-200 rounded p-2 whitespace-normal">
                    The name the merged variable will carry for the values of this row, in the figures and tables. It is only a label: changing it does not change
                    which raw values are aligned together. Any name you like works.
                  </div>
                )}
              </th>
              {memberCohorts.map(c => (
                <th key={c}>
                  <span className={`px-1.5 py-0.5 rounded border text-[10px] ${cohortColor(cohorts, c)}`}>{c}</span>{' '}
                  <span className="font-mono text-xs">{hv.members[c].var_name}</span>
                </th>
              ))}
              <th className="w-8"></th>
            </tr>
          </thead>
          <tbody>
            {harmonized.map(h => {
              const ev = rowEvidence(h);
              return (
                <React.Fragment key={h}>
                  <tr className={ev.length ? 'border-b-0' : ''}>
                    <td className="align-top">
                      <input className="input input-xs input-bordered w-full font-semibold" defaultValue={h} onBlur={e => renameHarmonized(h, e.target.value)} />
                    </td>
                    {memberCohorts.map(c => {
                      const here = rawsFor(c, h);
                      const others = elsewhere(c, h);
                      return (
                        <td key={c} className="align-top">
                          <div className="flex flex-wrap gap-1">
                            {here.map(raw => (
                              <span key={raw} className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs bg-base-200" title={labelOf(c, raw)}>
                                <span className="font-mono">{displayRaw(raw)}</span>
                                {labelOf(c, raw) && labelOf(c, raw) !== raw && <span className="opacity-60">{labelOf(c, raw)}</span>}
                                {here.length > 1 && (
                                  <button onClick={() => splitOut(c, raw)} className="opacity-50 hover:opacity-100" title="Move to its own row">
                                    ↗
                                  </button>
                                )}
                                <button onClick={() => setRaw(c, raw, '')} className="opacity-50 hover:opacity-100" title="Exclude this value from the analysis">
                                  <X size={10} />
                                </button>
                              </span>
                            ))}
                            {others.length > 0 && (here.length === 0 || openAdd[`${c}|${h}`]) && (
                              <select
                                className="select select-xs select-bordered max-w-[180px]"
                                value=""
                                autoFocus={!!openAdd[`${c}|${h}`]}
                                onBlur={() => setOpenAdd(prev => ({...prev, [`${c}|${h}`]: false}))}
                                onChange={e => {
                                  if (e.target.value) setRaw(c, e.target.value, h);
                                  setOpenAdd(prev => ({...prev, [`${c}|${h}`]: false}));
                                }}
                                title="Move a value of this cohort into this row"
                              >
                                <option value="">{here.length ? 'move here…' : 'choose a value…'}</option>
                                {others.map(o => (
                                  <option key={o.raw} value={o.raw}>
                                    {displayRaw(o.raw)}
                                    {labelOf(c, o.raw) && labelOf(c, o.raw) !== o.raw ? ` (${labelOf(c, o.raw)})` : ''}
                                    {o.where ? ` (now in "${o.where}")` : ' (excluded)'}
                                  </option>
                                ))}
                              </select>
                            )}
                            {others.length > 0 && here.length > 0 && !openAdd[`${c}|${h}`] && (
                              <button
                                type="button"
                                className="text-base-content/30 hover:text-base-content text-sm leading-none px-1"
                                title="Move another value of this cohort into this row"
                                onClick={() => setOpenAdd(prev => ({...prev, [`${c}|${h}`]: true}))}
                              >
                                +
                              </button>
                            )}
                            {here.length === 0 && others.length === 0 && <span className="text-xs text-base-content/40">no values</span>}
                          </div>
                        </td>
                      );
                    })}
                    <td className="align-top">
                      <button className="btn btn-ghost btn-xs" onClick={() => dropRow(h)} title="Remove this row (its values are excluded from the analysis)">
                        <Trash2 size={12} />
                      </button>
                    </td>
                  </tr>
                  {ev.length > 0 && (
                    <tr>
                      <td></td>
                      <td colSpan={memberCohorts.length + 1} className="pt-0">
                        <div className="flex items-center gap-1 flex-wrap text-[10px] text-base-content/50">
                          matched by
                          {ev.slice(0, 4).map((e, i) => (
                            <EvidenceBadge key={i} e={e} />
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-2 text-xs space-y-1">
        {!anyExcluded && memberCohorts.every(c => (categories[c] || []).length > 0) && (
          <div className="text-emerald-700 inline-flex items-center gap-1">
            <Check size={12} /> All recorded values of every cohort are mapped.
          </div>
        )}
        {memberCohorts.map(c =>
          missingOf(c).length === 0 ? null : (
            <div key={c} className="flex flex-wrap items-center gap-1">
              <span className="font-semibold">{c} values excluded from the analysis:</span>
              {missingOf(c).map(raw => (
                <span key={raw} className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-rose-50 border border-rose-200 text-rose-900">
                  <span className="font-mono">{raw}</span>
                  {labelOf(c, raw) && labelOf(c, raw) !== raw && <span className="opacity-60">{labelOf(c, raw)}</span>}
                  <button onClick={() => splitOut(c, raw)} className="opacity-60 hover:opacity-100" title="Restore as its own row">
                    restore
                  </button>
                </span>
              ))}
            </div>
          )
        )}
      </div>
      <MissingPolicy hv={hv} cohorts={memberCohorts} missingPct={missingPct} onChange={vm => onChange(vm)} />
    </div>
  );
}

// ---- Empty / coded-missing values: one policy, applied to every cohort ---------

function MissingPolicy({
  hv,
  cohorts,
  missingPct,
  onChange
}: {
  hv: HVar;
  cohorts: string[];
  missingPct?: Record<string, number | null | undefined>;
  onChange: (vm: Record<string, Record<string, string>>) => void;
}) {
  // Two choices only: exclude those patients, or keep them as one category
  // named MISSING_LABEL in every cohort (empty cells and the dictionary's
  // declared missing codes alike), so the category is harmonized by construction.
  const included = cohorts.length > 0 && cohorts.every(c => (hv.value_map[c] || {})[MISSING_KEY] === MISSING_LABEL);
  const setPolicy = (include: boolean) => {
    const vm: Record<string, Record<string, string>> = {...hv.value_map};
    cohorts.forEach(c => {
      vm[c] = {...(vm[c] || {}), [MISSING_KEY]: include ? MISSING_LABEL : ''};
    });
    onChange(vm);
  };
  const pcts = cohorts.filter(c => missingPct?.[c] != null).map(c => `${c} ${Math.round(missingPct![c]!)}%`);
  // Nothing to decide when the EDA reports no missing values anywhere: say so
  // instead of offering the choice (the box stays, as a statement of fact).
  const noneMissing = cohorts.length > 0 && cohorts.every(c => missingPct?.[c] != null && Math.abs(missingPct![c]!) < 0.05);
  if (noneMissing) {
    return (
      <div className="mt-3 text-xs rounded-lg border border-base-300 bg-base-100 px-3 py-2">
        <div className="font-semibold inline-flex items-center gap-1">
          <Check size={12} className="text-emerald-700" /> No empty or missing values
          <span className="font-normal text-base-content/60">
            {' '}
            (0% of rows in {cohorts.length === 1 ? cohorts[0] : `${cohorts.slice(0, -1).join(', ')} and ${cohorts[cohorts.length - 1]}`}, per the EDA)
          </span>
        </div>
        <div className="text-[11px] text-base-content/60">Nothing to decide for this variable: every patient has a recorded value.</div>
      </div>
    );
  }
  return (
    <div className="mt-3 text-xs rounded-lg border border-base-300 bg-base-100 px-3 py-2">
      <div className="font-semibold">
        Patients with an empty or coded-missing value
        {pcts.length > 0 && <span className="font-normal text-base-content/60"> ({pcts.join(', ')} of rows)</span>}
      </div>
      <div className="text-[11px] text-base-content/60 mb-1.5">Empty cells and the dictionary&rsquo;s declared missing codes are treated alike, identically in every cohort.</div>
      <div className="flex flex-wrap gap-x-5 gap-y-1">
        <label className="inline-flex items-center gap-1.5 cursor-pointer">
          <input type="radio" className="radio radio-xs" checked={!included} onChange={() => setPolicy(false)} />
          <span>Exclude them from the analysis</span>
        </label>
        <label className="inline-flex items-center gap-1.5 cursor-pointer">
          <input type="radio" className="radio radio-xs" checked={included} onChange={() => setPolicy(true)} />
          <span>
            Include them as the category <span className="font-mono">{MISSING_LABEL}</span>
          </span>
        </label>
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
  // Short harmonized names: asked from iCARE-AI (or a heuristic) once per
  // completed harmonized variable, keyed by the set of member variables.
  const namedFor = useRef<Record<string, string>>({}); // role key -> member signature already named
  const [editingName, setEditingName] = useState<string | null>(null); // role key being renamed
  const [nameDraft, setNameDraft] = useState('');
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

  // Cohorts ordered by how many variables they expose (most first), so the
  // first field offers the most options.
  const orderedCohorts = useMemo(() => {
    const counts: Record<string, number> = {};
    variables.forEach(v => (counts[v.cohort_id] = (counts[v.cohort_id] || 0) + 1));
    return [...cohorts].sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
  }, [cohorts, variables]);

  const categoriesOf = useCallback(
    (cohort: string, varName: string) => {
      const v = variables.find(x => x.cohort_id === cohort && x.var_name === varName);
      return (v?.categories || []).map(c => ({value: c.value, label: c.label}));
    },
    [variables]
  );

  // Every row of the value map rests on a shared standard code (from the
  // value-suggestion clusters) — the "natural" alignment that earns "_pooled".
  const rowsMatchedByCode = (hv: HVar): boolean => {
    if (hv.type !== 'categorical') return true;
    const cl = clusters[hv.harmonized_name] || [];
    if (cl.length === 0) return false;
    const memberCohorts = cohorts.filter(c => hv.members[c]?.var_name);
    const rows = new Set<string>();
    // The MISSING_KEY entry carries the empty/missing policy, not a row of the table.
    memberCohorts.forEach(c => Object.entries(hv.value_map[c] || {}).forEach(([k, h]) => k !== MISSING_KEY && h && rows.add(h)));
    return Array.from(rows).every(h =>
      cl.some(x => x.evidence.some(e => e.type === 'code') && memberCohorts.some(c => (x.sources[c] || []).some(raw => (hv.value_map[c] || {})[raw] === h)))
    );
  };

  const rekey = <T,>(setter: React.Dispatch<React.SetStateAction<Record<string, T>>>, from: string, to: string) =>
    setter(prev => {
      if (!(from in prev) || from === to) return prev;
      const {[from]: moved, ...rest} = prev;
      return {...rest, [to]: moved};
    });

  const replaceHVar = useCallback(
    (oldName: string | undefined, next: HVar, roleKey: string) => {
      const vars = mapping.variables.filter(x => x.harmonized_name !== oldName && x.harmonized_name !== next.harmonized_name);
      vars.push(next);
      onMappingChange({...mapping, variables: vars, sources: useCache});
      onRolesChange({...roleAssignments, [roleKey]: next.harmonized_name});
      // Per-variable UI state is keyed by the harmonized name: follow a rename.
      if (oldName && oldName !== next.harmonized_name) {
        rekey(setSuggestions, oldName, next.harmonized_name);
        rekey(setClusters, oldName, next.harmonized_name);
        rekey(setForceOpen, oldName, next.harmonized_name);
        rekey(setShowProvenance, oldName, next.harmonized_name);
      }
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
        const next: HVar = {...hv, value_map: vm, value_map_edited: false, evidence: [...hv.evidence, {type: 'manual', detail: 'value map suggested from codes, labels and computed mappings'}]};
        const allByCode = r.clusters.length > 0 && r.clusters.every(cl => cl.evidence.some(e => e.type === 'code'));
        updateHVar(roleKey, {...next, harmonized_name: withSuffix(next.harmonized_name, nameSuffix(next, allByCode))});
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
    // The server decides what the model sees per cohort: the top-ranked
    // suggestions when a strong one exists (codes, high text similarity,
    // computed mappings), otherwise the cohort's whole variable list, each
    // variable with its categories or observed range.
    aiSuggest({
      task: 'match',
      anchor: {cohort_id: anchorCohort, var_name: hv.members[anchorCohort].var_name},
      targets: cohorts.filter(c => c !== anchorCohort),
      cached_files: useCache
    })
      .then(r => {
        const m = r.result?.matches || {};
        const next = {...current};
        Object.entries(m).forEach(([c, pick]: [string, any]) => {
          if (!pick?.var_name) return;
          const list = next[c] ? [...next[c]] : [];
          const mode = r.modes?.[c];
          const looked = mode ? (mode.mode === 'all' ? ` (looked at ${mode.listed === mode.total ? 'all' : `${mode.listed} of`} ${mode.total} variables)` : ` (looked at the top ${mode.listed} suggestions)`) : '';
          const aiEvidence = {type: 'ai' as const, detail: (pick.reason || 'AI suggestion') + looked};
          const existing = list.find(cand => cand.var_name === pick.var_name);
          if (existing) {
            next[c] = list
              .map(cand => (cand.var_name === pick.var_name ? {...cand, evidence: [...cand.evidence, aiEvidence], score: Math.max(cand.score || 0, 0.9)} : cand))
              .sort((a, b) => (b.score || 0) - (a.score || 0));
          } else {
            // Not among the scorer's candidates: add it from the cohort's variable list.
            const v = variables.find(x => x.cohort_id === c && x.var_name === pick.var_name);
            if (v) next[c] = [{...v, evidence: [aiEvidence], score: Math.max(0.9, Number(pick.confidence) || 0)}, ...list];
          }
        });
        setSuggestions(prev => ({...prev, [hv.harmonized_name]: next}));
      })
      .catch(e => setError(e.message))
      .finally(() => setBusy(null));
  };

  useEffect(() => {
    if (!multi) return;
    roles.forEach(r => {
      const name = roleAssignments[r.key];
      const hv = mapping.variables.find(x => x.harmonized_name === name);
      if (!hv) return;
      const members = cohorts.filter(c => hv.members[c]?.var_name).map(c => ({cohort_id: c, var_name: hv.members[c].var_name, var_label: hv.members[c].var_label}));
      if (members.length < cohorts.length) return;
      const signature = members.map(m => `${m.cohort_id}:${m.var_name}`).join('|');
      if (namedFor.current[r.key] === signature) return;
      namedFor.current[r.key] = signature;
      aiName(members)
        .then(res => {
          // the user may have moved on: apply to the current state of this role
          const current = mapping.variables.find(x => x.harmonized_name === roleAssignments[r.key]);
          if (!current) return;
          const taken = new Set(mapping.variables.filter(x => x.harmonized_name !== current.harmonized_name).map(x => x.harmonized_name));
          const base = withSuffix(res.name, nameSuffix(current, rowsMatchedByCode(current)));
          let newName = base;
          let n = 2;
          while (taken.has(newName)) newName = `${base}_${n++}`;
          replaceHVar(current.harmonized_name, {...current, harmonized_name: newName, label: res.label || current.label}, r.key);
        })
        .catch(() => null);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapping.variables, roleAssignments, cohorts, multi]);

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
                      </span>
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

            <div className="grid gap-3" style={{gridTemplateColumns: multi ? `repeat(${Math.min(cohorts.length, 3)}, minmax(0, 1fr))` : '1fr'}}>
              {!hv ? (
                // One field per cohort from the start; picking in any of them makes
                // that variable the anchor and suggests matches for the other cohorts.
                orderedCohorts.map(c => (
                  <div key={c}>
                    <div className="flex items-center gap-1 mb-1">
                      <span className={`px-1.5 py-0.5 rounded border text-[10px] font-semibold ${cohortColor(cohorts, c)}`}>{c}</span>
                    </div>
                    <VariableCombobox cohorts={cohorts} variables={variables} value={null} onPick={v => pickAnchor(role.key, v)} placeholder="Choose a variable…" restrictCohort={c} kindFilter={role.kind} />
                  </div>
                ))
              ) : (
                orderedCohorts.map(c => {
                  const m = hv.members[c];
                  const isAnchor = c === anchorCohort;
                  return (
                    <div key={c}>
                      {multi && (
                        <div className="flex items-center gap-1 mb-1">
                          <span className={`px-1.5 py-0.5 rounded border text-[10px] font-semibold ${cohortColor(cohorts, c)}`}>{c}</span>
                          {isAnchor && <span className="text-[10px] uppercase tracking-wide text-base-content/40">anchor</span>}
                          {!isAnchor && !m?.var_name && (roleSugg[c]?.length || 0) > 0 && (
                            <button className={`btn btn-sm btn-outline btn-warning gap-1 ml-auto ${busy === null ? 'shimmer-warm' : ''}`} onClick={() => doAiMatch(hv)} disabled={busy !== null} title="Ask the local model which candidate matches best">
                              <Sparkles size={13} /> {busy === 'ai-match' ? 'Asking…' : 'Ask iCARE-AI'}
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
                <div className="flex items-center gap-2 text-xs text-base-content/70">
                  <span>Harmonized variable:</span>
                  {editingName === role.key ? (
                    <>
                      <input
                        className="input input-xs input-bordered font-mono w-64"
                        value={nameDraft}
                        autoFocus
                        onChange={e => setNameDraft(e.target.value.replace(/[^a-zA-Z0-9_]/g, '_'))}
                        onKeyDown={e => {
                          if (e.key === 'Enter') {
                            if (nameDraft.trim()) updateHVar(role.key, {...hv, harmonized_name: nameDraft.trim()});
                            setEditingName(null);
                          }
                          if (e.key === 'Escape') setEditingName(null);
                        }}
                      />
                      <button
                        className="btn btn-xs"
                        onClick={() => {
                          if (nameDraft.trim()) updateHVar(role.key, {...hv, harmonized_name: nameDraft.trim()});
                          setEditingName(null);
                        }}
                      >
                        Done
                      </button>
                    </>
                  ) : (
                    <>
                      <span className="font-mono text-base-content">{hv.harmonized_name}</span>
                      <button
                        className="btn btn-ghost btn-xs gap-1"
                        title="Rename the harmonized variable (a label only; it does not change the mapping)"
                        onClick={() => {
                          setNameDraft(hv.harmonized_name);
                          setEditingName(role.key);
                        }}
                      >
                        <Edit2 size={12} /> rename
                      </button>
                    </>
                  )}
                </div>

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
                          onChange={(vm, manualMove) => {
                            const next: HVar = {...hv, value_map: vm, value_map_edited: hv.value_map_edited || !!manualMove};
                            const byCode = rowsMatchedByCode(next);
                            updateHVar(role.key, {...next, harmonized_name: withSuffix(next.harmonized_name, nameSuffix(next, byCode))});
                          }}
                          onSuggest={() => doSuggestValues(role.key, hv)}
                          onAi={() => doAiValues(role.key, hv)}
                          busy={busy}
                          missingPct={Object.fromEntries(cohorts.map(c => [c, variables.find(x => x.cohort_id === c && x.var_name === hv.members[c]?.var_name)?.eda?.missing_pct]))}
                        />
                      )}
                      {complete && multi && hv.type === 'numeric' && (
                        <UnitEditor hv={hv} cohorts={cohorts} onChange={next => updateHVar(role.key, {...next, harmonized_name: withSuffix(next.harmonized_name, nameSuffix(next, true))})} />
                      )}
                    </>
                  );
                })()}
                {!multi && hv.type === 'categorical' && anchorCohort && (
                  <div className="mt-2 text-xs text-base-content/60 space-y-1">
                    <div>
                      Categories:{' '}
                      {categoriesOf(anchorCohort, hv.members[anchorCohort].var_name)
                        .map(c => `${c.value}${c.label && c.label !== c.value ? ` (${c.label})` : ''}`)
                        .join(', ') || 'none'}
                    </div>
                    <MissingPolicy
                      hv={hv}
                      cohorts={[anchorCohort]}
                      missingPct={{[anchorCohort]: variables.find(x => x.cohort_id === anchorCohort && x.var_name === hv.members[anchorCohort].var_name)?.eda?.missing_pct}}
                      onChange={vm => updateHVar(role.key, {...hv, value_map: vm})}
                    />
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
