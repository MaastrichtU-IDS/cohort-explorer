'use client';

// Mapping Workbench — the harmonization board of the guided (no-code) wizard.
//
// For every role the chosen analysis needs (e.g. "variable of interest",
// "break down by"), the user builds one HARMONIZED VARIABLE: an anchor variable
// found by search, plus one member variable per other cohort, linked from
// evidence-ranked suggestions (shared codes, text similarity, cached mapping
// files, AI), then a value map (categorical) or unit conversion (numeric).
// Everything the user decides ends up in the mapping spec, which the enclave
// script prints under every figure as provenance.
import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {AlertTriangle, Check, ChevronDown, Database, Link2, Search, Trash2, X, Zap} from 'react-feather';
import {SparklesIcon as Sparkles} from '@/components/Icons';
import {
  Candidate,
  Evidence,
  HVar,
  MappingSpec,
  ValueCluster,
  VarInfo,
  aiSuggest,
  fetchCachedMappings,
  listMappings,
  loadMapping,
  newHVar,
  provenanceLine,
  saveMapping,
  searchVariables,
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
    text = 'CODE';
    title = `Shared standard code: ${e.detail}`;
  } else if (e.type === 'text') {
    text = `TEXT ${(e.score ?? 0).toFixed(2)}`;
    title = 'Name/label similarity (after synonym folding)';
  } else if (e.type === 'cache') {
    text = `CACHE · ${e.status || '?'}`;
    title = `From cached mapping file ${e.file}`;
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

function MemberChip({cohortId, cohorts, member, onRemove}: {cohortId: string; cohorts: string[]; member: {var_name: string; var_label?: string}; onRemove?: () => void}) {
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-lg border text-sm ${cohortColor(cohorts, cohortId)}`} title={member.var_label}>
      <span className="font-mono font-semibold">{member.var_name}</span>
      {member.var_label && <span className="opacity-70 truncate max-w-[220px]">{member.var_label}</span>}
      {onRemove && (
        <button onClick={onRemove} className="opacity-60 hover:opacity-100" title="Unlink">
          <X size={12} />
        </button>
      )}
    </span>
  );
}

// ---- Search panel ----------------------------------------------------------

function SearchPanel({cohorts, onPick, kindFilter}: {cohorts: string[]; onPick: (v: VarInfo) => void; kindFilter?: 'numeric' | 'categorical'}) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'any' | 'all' | 'exact'>('any');
  const [cohortFilter, setCohortFilter] = useState<string>('');
  const [results, setResults] = useState<VarInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const timer = useRef<any>(null);

  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      if (!query.trim()) {
        setResults([]);
        return;
      }
      setLoading(true);
      searchVariables(cohortFilter ? [cohortFilter] : cohorts, query, mode)
        .then(r => setResults(r.results))
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(timer.current);
  }, [query, mode, cohortFilter, cohorts]);

  const shown = kindFilter ? results.filter(r => r.kind === kindFilter) : results;

  return (
    <div className="flex flex-col h-full">
      <div className="relative">
        <Search size={15} className="absolute left-3 top-3 text-base-content/40" />
        <input
          className="input input-bordered w-full pl-9"
          placeholder="Search variables: name, label, standard name, code…"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
      </div>
      <div className="flex flex-wrap items-center gap-1.5 mt-2 text-xs">
        {(['any', 'all', 'exact'] as const).map(m => (
          <button key={m} onClick={() => setMode(m)} className={`px-2 py-0.5 rounded-full border ${mode === m ? 'bg-base-content text-base-100 border-base-content' : 'border-base-300 hover:bg-base-200'}`}>
            {m === 'any' ? 'any word' : m === 'all' ? 'all words' : 'exact phrase'}
          </button>
        ))}
        <span className="mx-1 text-base-content/30">|</span>
        <button onClick={() => setCohortFilter('')} className={`px-2 py-0.5 rounded-full border ${!cohortFilter ? 'bg-base-content text-base-100 border-base-content' : 'border-base-300'}`}>
          all cohorts
        </button>
        {cohorts.map(c => (
          <button key={c} onClick={() => setCohortFilter(c)} className={`px-2 py-0.5 rounded-full border ${cohortFilter === c ? 'ring-2 ring-offset-1 ring-base-content ' : ''}${cohortColor(cohorts, c)}`}>
            {c}
          </button>
        ))}
        {kindFilter && <span className="ml-auto text-base-content/50">showing {kindFilter} variables only</span>}
      </div>
      <div className="mt-2 flex-1 overflow-y-auto rounded-lg border border-base-300 bg-base-100 min-h-[200px]">
        {loading && <div className="p-3 text-sm text-base-content/50">Searching…</div>}
        {!loading && query && shown.length === 0 && <div className="p-3 text-sm text-base-content/50">No variables match.</div>}
        {!query && (
          <div className="p-4 text-sm text-base-content/50 leading-relaxed">
            Type to search across the selected cohorts. Matches are found on the variable name, its label, the
            standard concept name and codes. Variables in other cohorts that share a standard code are listed as{' '}
            <em>equivalents</em> under each result.
          </div>
        )}
        {shown.map(v => (
          <button
            key={`${v.cohort_id}::${v.var_name}`}
            onClick={() => onPick(v)}
            className="w-full text-left px-3 py-2 border-b border-base-200 hover:bg-base-200/60 transition-colors"
          >
            <div className="flex items-center gap-2">
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border ${cohortColor(cohorts, v.cohort_id)}`}>{v.cohort_id}</span>
              <span className="font-mono font-semibold text-sm">{v.var_name}</span>
              <span className="text-[10px] uppercase tracking-wide text-base-content/50">{v.kind}</span>
              {v.units && <span className="text-[10px] text-base-content/50">{v.units}</span>}
            </div>
            <div className="text-sm text-base-content/80 truncate">{v.var_label}</div>
            <div className="text-xs text-base-content/50 truncate">
              {v.concept_name && <span>{v.concept_name}</span>}
              {v.concept_code && <span className="ml-2 font-mono">{v.concept_code}</span>}
            </div>
            {v.equivalents && v.equivalents.length > 0 && (
              <div className="text-xs mt-0.5 text-emerald-800">
                ≈ {v.equivalents.map(e => `${e.var_name} [${e.cohort_id}]`).join(', ')}
              </div>
            )}
          </button>
        ))}
      </div>
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
  // harmonized values = union over value_map targets (ordered by first appearance)
  const harmonized = useMemo(() => {
    const seen: string[] = [];
    memberCohorts.forEach(c => Object.values(hv.value_map[c] || {}).forEach(h => {
      if (h && !seen.includes(h)) seen.push(h);
    }));
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
    <div className="mt-3">
      <div className="flex items-center gap-2 flex-wrap">
        <h4 className="font-semibold text-sm">Value mapping</h4>
        <span className="text-xs text-base-content/50">each row is one harmonized value; assign the raw values of every cohort to it</span>
        <div className="ml-auto flex gap-2">
          <button className="btn btn-xs btn-outline gap-1" onClick={onSuggest} disabled={busy !== null}>
            <Zap size={12} /> {busy === 'values' ? 'Suggesting…' : 'Suggest from codes, labels & cache'}
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
                  <span className={`px-1.5 py-0.5 rounded border text-[10px] ${cohortColor(cohorts, c)}`}>{c}</span>{' '}
                  <span className="font-mono text-xs">{hv.members[c].var_name}</span>
                </th>
              ))}
              <th className="w-8"></th>
            </tr>
          </thead>
          <tbody>
            {harmonized.map(h => (
              <tr key={h}>
                <td className="align-top">
                  <input
                    className="input input-xs input-bordered w-full font-semibold"
                    defaultValue={h}
                    onBlur={e => renameHarmonized(h, e.target.value)}
                  />
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {clusterEvidence(h).slice(0, 3).map((e, i) => (
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
                        <select
                          className="select select-xs select-bordered max-w-[160px]"
                          value=""
                          onChange={e => e.target.value && setRaw(c, e.target.value, h)}
                        >
                          <option value="">+ add value</option>
                          {unmapped(c).map(cat => (
                            <option key={cat.value} value={cat.value}>
                              {cat.value}
                              {cat.label && cat.label !== cat.value ? ` — ${cat.label}` : ''}
                            </option>
                          ))}
                        </select>
                      )}
                      {rawsFor(c, h).length === 0 && unmapped(c).length === 0 && <span className="text-xs text-base-content/40">—</span>}
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
                      if (e.key === 'Enter' && newValue.trim() && memberCohorts[0]) {
                        // create the row by assigning a placeholder: rows exist only through assignments,
                        // so we add the value to the first unmapped raw of the first cohort if any.
                        const c = memberCohorts.find(cc => unmapped(cc).length > 0);
                        if (c) setRaw(c, unmapped(c)[0].value, newValue.trim());
                        setNewValue('');
                      }
                    }}
                  />
                  <span className="text-xs text-base-content/50">Enter to add (it is created with the first unassigned raw value; adjust afterwards)</span>
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
    <div className="mt-3">
      <div className="flex items-center gap-2">
        <h4 className="font-semibold text-sm">Units</h4>
        {mismatch ? (
          <span className="text-xs text-rose-700 inline-flex items-center gap-1">
            <AlertTriangle size={12} /> the cohorts declare different units — enter a conversion factor
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
              <span className={`px-1.5 py-0.5 rounded border text-[10px] ${cohortColor(cohorts, c)}`}>{c}</span>{' '}
              <span className="font-mono">{hv.members[c].var_name}</span> in <b>{hv.members[c].unit || 'unknown unit'}</b>
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
  const [activeRole, setActiveRole] = useState<string>(roles[0]?.key || '');
  const [suggestions, setSuggestions] = useState<Record<string, Candidate[]>>({});
  const [clusters, setClusters] = useState<ValueCluster[] | null>(null);
  const [categories, setCategories] = useState<Record<string, {value: string; label: string}[]>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cacheFiles, setCacheFiles] = useState<{filename: string; source: string; target: string; generated_at: string; size_kb: number}[]>([]);
  const [useCache, setUseCache] = useState<string[]>([]);
  const [cacheOpen, setCacheOpen] = useState(false);
  const [saved, setSaved] = useState<{id: string; name: string; cohorts: string[]; variables: number; updated_at: string}[]>([]);
  const [saveName, setSaveName] = useState(mapping.name || '');
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  const activeName = roleAssignments[activeRole];
  const hvIndex = mapping.variables.findIndex(v => v.harmonized_name === activeName);
  const hv = hvIndex >= 0 ? mapping.variables[hvIndex] : null;
  const roleDef = roles.find(r => r.key === activeRole);

  useEffect(() => {
    if (!multi) return;
    fetchCachedMappings(cohorts).then(r => setCacheFiles(r.files)).catch(() => setCacheFiles([]));
    listMappings(cohorts).then(r => setSaved(r.mappings)).catch(() => setSaved([]));
  }, [cohorts, multi]);

  const updateHVar = useCallback(
    (next: HVar) => {
      const vars = mapping.variables.slice();
      const i = vars.findIndex(v => v.harmonized_name === activeName);
      if (i >= 0) vars[i] = next;
      else vars.push(next);
      onMappingChange({...mapping, variables: vars, sources: useCache});
      if (next.harmonized_name !== activeName) onRolesChange({...roleAssignments, [activeRole]: next.harmonized_name});
    },
    [mapping, activeName, activeRole, onMappingChange, onRolesChange, roleAssignments, useCache]
  );

  // Pick an anchor from search: creates the harmonized variable for the active role.
  const pickAnchor = (v: VarInfo) => {
    if (roleDef?.kind && v.kind !== roleDef.kind) {
      setError(`"${roleDef.label}" needs a ${roleDef.kind} variable; ${v.var_name} is ${v.kind}.`);
      return;
    }
    setError(null);
    let created = newHVar(v);
    // keep harmonized names unique
    const taken = new Set(mapping.variables.filter(x => x.harmonized_name !== activeName).map(x => x.harmonized_name));
    let name = created.harmonized_name;
    let n = 2;
    while (taken.has(name)) name = `${created.harmonized_name}_${n++}`;
    created = {...created, harmonized_name: name};
    const vars = mapping.variables.filter(x => x.harmonized_name !== activeName);
    vars.push(created);
    onMappingChange({...mapping, variables: vars, sources: useCache});
    onRolesChange({...roleAssignments, [activeRole]: name});
    setSuggestions({});
    setClusters(null);
    setCategories({[v.cohort_id]: v.categories.map(c => ({value: c.value, label: c.label}))});
    if (multi) {
      setBusy('suggest');
      suggestMatches({cohort_id: v.cohort_id, var_name: v.var_name}, cohorts.filter(c => c !== v.cohort_id), useCache)
        .then(r => setSuggestions(r.candidates))
        .catch(e => setError(e.message))
        .finally(() => setBusy(null));
    }
  };

  const linkCandidate = (c: Candidate) => {
    if (!hv) return;
    const next: HVar = {
      ...hv,
      members: {...hv.members, [c.cohort_id]: {var_name: c.var_name, var_label: c.var_label, unit: c.units, kind: c.kind}},
      evidence: [...hv.evidence, ...c.evidence.filter(e => e.type !== 'warning').map(e => ({...e, detail: e.detail ? `${e.detail} (${c.cohort_id})` : undefined}))]
    };
    // cached value_mapping arrives with the candidate: pre-fill
    const cacheEv = c.evidence.find(e => e.type === 'cache' && e.value_mapping);
    if (cacheEv && cacheEv.value_mapping && hv.type === 'categorical') {
      const vm = {...next.value_map};
      const anchorCohort = Object.keys(hv.members)[0];
      const src = cacheEv.value_mapping.source || {};
      const tgt = cacheEv.value_mapping.target || {};
      // direction unknown here; assign whichever side's keys exist among the categories
      const anchorCats = new Set((categories[anchorCohort] || []).map(x => x.value));
      const candCats = new Set(c.categories.map(x => x.value));
      const assign = (cohort: string, m: Record<string, string>, cats: Set<string>) => {
        const out: Record<string, string> = {...(vm[cohort] || {})};
        Object.entries(m).forEach(([k, v]) => {
          if (cats.has(k) && v && v !== 'unknown') out[k] = v;
        });
        vm[cohort] = out;
      };
      const srcKeys = Object.keys(src);
      if (srcKeys.some(k => anchorCats.has(k))) {
        assign(anchorCohort, src, anchorCats);
        assign(c.cohort_id, tgt, candCats);
      } else {
        assign(anchorCohort, tgt, anchorCats);
        assign(c.cohort_id, src, candCats);
      }
      next.value_map = vm;
    }
    updateHVar(next);
    setCategories(prev => ({...prev, [c.cohort_id]: c.categories.map(x => ({value: x.value, label: x.label}))}));
  };

  const unlink = (cohortId: string) => {
    if (!hv) return;
    const members = {...hv.members};
    delete members[cohortId];
    const vm = {...hv.value_map};
    delete vm[cohortId];
    updateHVar({...hv, members, value_map: vm});
  };

  const doSuggestValues = () => {
    if (!hv) return;
    const members: Record<string, string> = {};
    cohorts.forEach(c => {
      if (hv.members[c]?.var_name) members[c] = hv.members[c].var_name;
    });
    setBusy('values');
    suggestValues(members, useCache)
      .then(r => {
        setClusters(r.clusters);
        const cats: Record<string, {value: string; label: string}[]> = {};
        Object.entries(r.members).forEach(([c, m]) => (cats[c] = m.categories.map(x => ({value: x.value, label: x.label}))));
        setCategories(prev => ({...prev, ...cats}));
        // apply clusters as the value map (user can edit afterwards)
        const vm: Record<string, Record<string, string>> = {};
        r.clusters.forEach(cl => {
          Object.entries(cl.sources).forEach(([c, raws]) => {
            vm[c] = vm[c] || {};
            raws.forEach(raw => (vm[c][raw] = cl.harmonized));
          });
        });
        updateHVar({...hv, value_map: vm, evidence: [...hv.evidence, {type: 'manual', detail: 'value map suggested from codes/labels/cache'}]});
      })
      .catch(e => setError(e.message))
      .finally(() => setBusy(null));
  };

  const doAiValues = () => {
    if (!hv) return;
    const variables: Record<string, any> = {};
    cohorts.forEach(c => {
      if (hv.members[c]?.var_name) variables[c] = {var_name: hv.members[c].var_name, var_label: hv.members[c].var_label, categories: categories[c] || []};
    });
    setBusy('ai-values');
    aiSuggest({task: 'values', variables})
      .then(r => {
        const vmap = r.result?.value_map || {};
        const vm: Record<string, Record<string, string>> = {};
        Object.entries(vmap).forEach(([c, m]: [string, any]) => {
          vm[c] = {};
          Object.entries(m || {}).forEach(([k, v]) => {
            if (v) vm[c][k] = String(v);
          });
        });
        updateHVar({...hv, value_map: vm, evidence: [...hv.evidence, {type: 'ai', detail: `value map proposed by ${r.model}`}]});
      })
      .catch(e => setError(e.message))
      .finally(() => setBusy(null));
  };

  const doAiMatch = () => {
    if (!hv) return;
    const anchorCohort = Object.keys(hv.members)[0];
    setBusy('ai-match');
    aiSuggest({
      task: 'match',
      anchor: {cohort_id: anchorCohort, ...hv.members[anchorCohort]},
      candidates: Object.fromEntries(Object.entries(suggestions).map(([c, list]) => [c, list.slice(0, 12).map(x => ({var_name: x.var_name, var_label: x.var_label, concept_name: x.concept_name, units: x.units}))]))
    })
      .then(r => {
        const matches = r.result?.matches || {};
        const next = {...suggestions};
        Object.entries(matches).forEach(([c, m]: [string, any]) => {
          if (!m || !m.var_name || !next[c]) return;
          next[c] = next[c].map(cand =>
            cand.var_name === m.var_name ? {...cand, evidence: [...cand.evidence, {type: 'ai', detail: m.reason || 'AI suggestion'}], score: Math.max(cand.score || 0, 0.9)} : cand
          );
          next[c].sort((a, b) => (b.score || 0) - (a.score || 0));
        });
        setSuggestions(next);
      })
      .catch(e => setError(e.message))
      .finally(() => setBusy(null));
  };

  const refreshSuggestions = (files: string[]) => {
    if (!hv || !multi) return;
    const anchorCohort = Object.keys(hv.members)[0];
    setBusy('suggest');
    suggestMatches({cohort_id: anchorCohort, var_name: hv.members[anchorCohort].var_name}, cohorts.filter(c => c !== anchorCohort), files)
      .then(r => setSuggestions(r.candidates))
      .catch(e => setError(e.message))
      .finally(() => setBusy(null));
  };

  const toggleCache = (f: string) => {
    const next = useCache.includes(f) ? useCache.filter(x => x !== f) : [...useCache, f];
    setUseCache(next);
    onMappingChange({...mapping, sources: next});
    refreshSuggestions(next);
  };

  const doSave = () => {
    const name = saveName.trim() || `mapping ${cohorts.join(' + ')}`;
    saveMapping({...mapping, name, cohorts, sources: useCache, created_by: mapping.created_by || userEmail || undefined})
      .then(r => {
        onMappingChange({...mapping, name, id: r.id, sources: useCache});
        setSaveMsg(`Saved as "${name}"`);
        listMappings(cohorts).then(x => setSaved(x.mappings)).catch(() => null);
      })
      .catch(e => setSaveMsg(e.message));
  };

  const doLoad = (id: string) => {
    loadMapping(id)
      .then(spec => {
        onMappingChange({...spec, cohorts});
        setSaveName(spec.name || '');
        // assign roles by order where names exist
        const assign: Record<string, string> = {...roleAssignments};
        roles.forEach((r, i) => {
          if (!assign[r.key] && spec.variables[i]) assign[r.key] = spec.variables[i].harmonized_name;
        });
        onRolesChange(assign);
      })
      .catch(e => setError(e.message));
  };

  const missingCohorts = hv ? cohorts.filter(c => !hv.members[c]?.var_name) : cohorts;

  return (
    <div className="flex flex-col gap-4">
      {/* Role slots */}
      <div className="grid gap-2" style={{gridTemplateColumns: `repeat(${Math.min(roles.length, 3)}, minmax(0, 1fr))`}}>
        {roles.map(r => {
          const name = roleAssignments[r.key];
          const v = mapping.variables.find(x => x.harmonized_name === name);
          const complete = v && cohorts.every(c => v.members[c]?.var_name);
          return (
            <button
              key={r.key}
              onClick={() => setActiveRole(r.key)}
              className={`text-left rounded-xl border-2 p-3 transition-colors ${activeRole === r.key ? 'border-base-content bg-base-100' : 'border-base-300 bg-base-100/60 hover:border-base-content/40'}`}
            >
              <div className="text-[11px] uppercase tracking-wide text-base-content/50 flex items-center gap-2">
                {r.label}
                {r.optional && <span className="normal-case tracking-normal">(optional)</span>}
                {r.kind && <span className="ml-auto normal-case tracking-normal">{r.kind}</span>}
              </div>
              {v ? (
                <div className="mt-1">
                  <div className="font-semibold flex items-center gap-2">
                    {v.label || v.harmonized_name}
                    {complete ? <Check size={14} className="text-emerald-600" /> : <AlertTriangle size={14} className="text-amber-600" />}
                  </div>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {cohorts.map(c => (
                      <span key={c} className={`px-1.5 py-0.5 rounded border text-[10px] ${v.members[c]?.var_name ? cohortColor(cohorts, c) : 'bg-base-200 text-base-content/40 border-base-300 line-through'}`}>
                        {c}: {v.members[c]?.var_name || '—'}
                      </span>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="mt-1 text-sm text-base-content/50">{r.hint || 'Pick a variable from the search'}</div>
              )}
            </button>
          );
        })}
      </div>

      {error && (
        <div className="alert alert-error text-sm py-2">
          <span>{error}</span>
          <button className="btn btn-ghost btn-xs" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* Cached mappings + saved mappings toolbar (multi-cohort only) */}
      {multi && (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <div className="relative">
            <button className="btn btn-sm btn-outline gap-1" onClick={() => setCacheOpen(o => !o)}>
              <Database size={14} /> Cached mapping files {useCache.length > 0 && <span className="badge badge-sm badge-primary">{useCache.length} in use</span>}
              <ChevronDown size={14} />
            </button>
            {cacheOpen && (
              <div className="absolute z-20 mt-1 w-[520px] max-h-72 overflow-y-auto bg-base-100 border border-base-300 rounded-lg shadow-xl p-2">
                {cacheFiles.length === 0 && <div className="text-xs text-base-content/50 p-2">No cached mapping files for these cohorts. Generate them from the Mapping page.</div>}
                {cacheFiles.map(f => (
                  <label key={f.filename} className="flex items-start gap-2 p-1.5 hover:bg-base-200 rounded cursor-pointer">
                    <input type="checkbox" className="checkbox checkbox-xs mt-0.5" checked={useCache.includes(f.filename)} onChange={() => toggleCache(f.filename)} />
                    <span className="text-xs">
                      <span className="font-semibold">{f.source} → {f.target}</span> · {f.generated_at.slice(0, 10)} · {f.size_kb} KB
                      <div className="font-mono text-[10px] text-base-content/50 break-all">{f.filename}</div>
                    </span>
                  </label>
                ))}
                <div className="text-[11px] text-base-content/50 p-2 border-t border-base-200 mt-1">
                  Files you tick become an evidence source: their rows appear as <span className="px-1 rounded bg-violet-600 text-white">CACHE</span> badges and their value mappings pre-fill the value table.
                </div>
              </div>
            )}
          </div>
          {saved.length > 0 && (
            <select className="select select-sm select-bordered" value="" onChange={e => e.target.value && doLoad(e.target.value)}>
              <option value="">Load a saved mapping…</option>
              {saved.map(m => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.variables} vars, {m.cohorts.join('+')}, {m.updated_at?.slice(0, 10)})
                </option>
              ))}
            </select>
          )}
          <div className="ml-auto flex items-center gap-2">
            <input className="input input-sm input-bordered w-56" placeholder="name this mapping to save it" value={saveName} onChange={e => setSaveName(e.target.value)} />
            <button className="btn btn-sm btn-outline" onClick={doSave} disabled={mapping.variables.length === 0}>
              Save mapping
            </button>
            {saveMsg && <span className="text-xs text-base-content/60">{saveMsg}</span>}
          </div>
        </div>
      )}

      {/* Main board */}
      <div className="grid lg:grid-cols-[360px_1fr] gap-4 items-start">
        <div className="rounded-xl border border-base-300 bg-base-100 p-3 h-[620px]">
          <div className="text-[11px] uppercase tracking-wide text-base-content/50 mb-2">
            {hv ? 'Replace the anchor variable' : `Find the variable for "${roleDef?.label}"`}
          </div>
          <SearchPanel cohorts={cohorts} onPick={pickAnchor} kindFilter={roleDef?.kind} />
        </div>

        <div className="rounded-xl border border-base-300 bg-base-100 p-4 min-h-[620px]">
          {!hv ? (
            <div className="h-full flex flex-col items-center justify-center text-center text-base-content/50 gap-2 py-16">
              <Link2 size={28} />
              <div className="max-w-md">
                Search on the left and click a variable to make it the <b>anchor</b> for "{roleDef?.label}".
                {multi && ' Matching variables in the other cohorts are then suggested with their evidence.'}
              </div>
            </div>
          ) : (
            <div>
              <div className="flex flex-wrap items-end gap-3">
                <label className="text-xs">
                  Harmonized name
                  <input
                    className="input input-sm input-bordered w-52 font-mono"
                    value={hv.harmonized_name}
                    onChange={e => updateHVar({...hv, harmonized_name: e.target.value.replace(/[^a-zA-Z0-9_]/g, '_')})}
                  />
                </label>
                <label className="text-xs flex-1 min-w-[200px]">
                  Label shown on figures
                  <input className="input input-sm input-bordered w-full" value={hv.label} onChange={e => updateHVar({...hv, label: e.target.value})} />
                </label>
                <label className="text-xs">
                  Type
                  <select className="select select-sm select-bordered" value={hv.type} onChange={e => updateHVar({...hv, type: e.target.value as any})}>
                    <option value="categorical">categorical</option>
                    <option value="numeric">numeric</option>
                  </select>
                </label>
              </div>

              {/* Members per cohort */}
              <div className="mt-4 space-y-3">
                {cohorts.map(c => {
                  const m = hv.members[c];
                  if (m?.var_name) {
                    return (
                      <div key={c} className="flex items-center gap-2">
                        <span className={`w-28 shrink-0 px-1.5 py-0.5 rounded border text-[10px] text-center ${cohortColor(cohorts, c)}`}>{c}</span>
                        <MemberChip cohortId={c} cohorts={cohorts} member={m} onRemove={Object.keys(hv.members).length > 1 ? () => unlink(c) : undefined} />
                        {Object.keys(hv.members)[0] === c && <span className="text-[10px] uppercase tracking-wide text-base-content/40">anchor</span>}
                      </div>
                    );
                  }
                  const cands = suggestions[c] || [];
                  return (
                    <div key={c} className="rounded-lg border border-dashed border-base-300 p-2">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`w-28 shrink-0 px-1.5 py-0.5 rounded border text-[10px] text-center ${cohortColor(cohorts, c)}`}>{c}</span>
                        <span className="text-xs text-base-content/60">
                          {busy === 'suggest' ? 'looking for matches…' : cands.length ? `${cands.length} suggestion${cands.length > 1 ? 's' : ''} — click to link` : 'no suggestion found; search on the left and pick one'}
                        </span>
                        {cands.length > 0 && (
                          <button className="btn btn-xs btn-ghost gap-1 ml-auto" onClick={doAiMatch} disabled={busy !== null} title="Ask the local model which candidate matches best">
                            <Sparkles size={12} /> {busy === 'ai-match' ? 'Asking…' : 'Ask iCARE-AI'}
                          </button>
                        )}
                      </div>
                      <div className="flex flex-col gap-1 max-h-56 overflow-y-auto">
                        {cands.slice(0, 8).map(cand => (
                          <button
                            key={cand.var_name}
                            onClick={() => linkCandidate(cand)}
                            className="text-left flex items-start gap-2 px-2 py-1.5 rounded hover:bg-base-200 transition-colors"
                          >
                            <span className="font-mono text-sm font-semibold shrink-0">{cand.var_name}</span>
                            <span className="text-xs text-base-content/70 truncate flex-1">
                              {cand.var_label}
                              {cand.units ? ` · ${cand.units}` : ''}
                              {cand.kind !== hv.type ? ` · ${cand.kind}` : ''}
                            </span>
                            <span className="flex gap-1 shrink-0">
                              {cand.evidence.slice(0, 3).map((e, i) => (
                                <EvidenceBadge key={i} e={e} />
                              ))}
                            </span>
                            <span className="text-xs tabular-nums text-base-content/50 w-10 text-right">{(cand.score || 0).toFixed(2)}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Values / units */}
              {missingCohorts.length === 0 && hv.type === 'categorical' && multi && (
                <ValueMapEditor
                  hv={hv}
                  cohorts={cohorts}
                  categories={categories}
                  clusters={clusters}
                  onChange={vm => updateHVar({...hv, value_map: vm})}
                  onSuggest={doSuggestValues}
                  onAi={doAiValues}
                  busy={busy}
                />
              )}
              {missingCohorts.length === 0 && hv.type === 'numeric' && multi && <UnitEditor hv={hv} cohorts={cohorts} onChange={updateHVar} />}
              {!multi && hv.type === 'categorical' && (
                <div className="mt-3 text-xs text-base-content/60">
                  Categories: {(categories[cohorts[0]] || []).map(c => `${c.value}${c.label && c.label !== c.value ? ` (${c.label})` : ''}`).join(', ') || '—'}
                </div>
              )}

              {/* Provenance preview */}
              <div className="mt-4 rounded-lg bg-base-200 p-3">
                <div className="text-[11px] uppercase tracking-wide text-base-content/50 mb-1">Provenance line printed under every figure</div>
                <div className="font-mono text-xs break-words">{provenanceLine(hv)}</div>
                <div className="flex flex-wrap gap-1 mt-2">
                  {hv.evidence.slice(-6).map((e, i) => (
                    <EvidenceBadge key={i} e={e} />
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
