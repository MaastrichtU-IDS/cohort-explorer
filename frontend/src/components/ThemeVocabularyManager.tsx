'use client';

// Admin editing of the metadata-clustering theme vocabulary: two levels, parents
// (groups) and themes (tags). Pick a theme, then act on it from the bar: rename,
// move to a group, merge into another theme, nest under another theme (which
// becomes the name of a group holding both), hide or unhide. Groups can be renamed
// or deleted (their themes go to "unsorted themes"). Every edit is applied to the
// stored results at once, recorded as a rule that later runs re-apply, and the
// last one can be undone.
import React, {useEffect, useMemo, useState} from 'react';
import {apiUrl} from '@/utils';
import {setThemeExcluded} from '@/components/metadataThemes';

interface VocabParent {
  id: string;
  label: string;
}

interface VocabTheme {
  id: string;
  label: string;
  parent: string;
  raw: string[];
  studies: number;
  excluded: boolean;
}

interface VocabRule {
  op: string;
  at?: string;
  by?: string;
  [k: string]: any;
}

interface VocabView {
  parents: VocabParent[];
  themes: VocabTheme[];
  n_studies: number;
  rules: VocabRule[];
  undo_available: boolean;
  built?: string;
  rebuilt?: string;
}

type Mode = 'rename' | 'move' | 'merge' | 'nest' | null;
type Tab = 'vocabulary' | 'terms';

interface PairSide {
  cohort: string;
  field: string;
  label: string;
  text: string;
}

/** One matched pair as /vocabulary/theme/{id}/pairs and /terms/pairs return it. */
export interface PairView {
  a: PairSide;
  b: PairSide;
  source: 'lexical' | 'ai' | 'both';
  ai_strength: 'strong' | 'partial' | null;
  score: number;
  shared: string | null;
  differs: string | null;
  basis: string | null;
  useful_terms: string[];
  dropped_terms: string[];
  terms_judged: boolean;
  raw_tags: string[];
  outcome_kind: string | null;
  theme_ids: string[];
  term_kept?: boolean;
}

interface PairsPayload {
  pairs: PairView[];
  n_pairs: number;
  studies: string[];
  theme_labels: Record<string, string>;
  theme?: {id: string; label: string; parent: string; raw: string[]};
  term?: string;
  kept?: number;
}

interface TermEntry {
  term: string;
  words: number;
  pairs: number;
  useful: number;
  dropped: number;
  theme: string | null;
  theme_label: string | null;
}

/** The pairs behind a theme or a common term, each with its explanation, the
 *  common terms the model kept and dropped, and the other themes on it. */
export function PairList({payload, focusTheme}: {payload: PairsPayload; focusTheme?: string}) {
  const [showText, setShowText] = useState(false);
  if (payload.pairs.length === 0) return <p className="text-sm italic text-base-content/60">No pair carries this.</p>;
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 text-xs text-base-content/60">
        <span>{payload.n_pairs} pairs · {payload.studies.length} studies</span>
        <label className="label cursor-pointer gap-1 py-0">
          <input type="checkbox" className="checkbox checkbox-xs" checked={showText} onChange={e => setShowText(e.target.checked)} />
          <span className="label-text text-xs">show the element texts</span>
        </label>
      </div>
      <div className="max-h-[32rem] overflow-y-auto space-y-2 pr-1">
        {payload.pairs.map((p, i) => (
          <div key={i} className="rounded-md border border-base-300 bg-base-100 p-2 text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-semibold">{p.a.cohort}</span>
              <span className="text-xs text-base-content/60">{p.a.label}</span>
              <span className="text-base-content/40">↔</span>
              <span className="font-semibold">{p.b.cohort}</span>
              <span className={`text-xs ${p.a.field === p.b.field ? 'text-base-content/60' : 'text-warning'}`}>{p.b.label}</span>
              <span className={`badge badge-xs ${p.source === 'ai' ? 'badge-secondary' : p.source === 'both' ? 'badge-primary' : 'badge-ghost'}`}>
                {p.source === 'ai' ? 'AI only' : p.source === 'both' ? 'Lexical + AI' : 'Lexical'}{p.ai_strength ? ` · ${p.ai_strength}` : ''}
              </span>
              {p.outcome_kind && <span className="badge badge-xs badge-info badge-outline">{p.outcome_kind}</span>}
              {p.term_kept !== undefined && (
                <span className={`badge badge-xs ${p.term_kept ? 'badge-success' : 'badge-ghost line-through'}`} title={p.term_kept ? 'The model kept this term as informative for this pair' : 'The model dropped this term for this pair'}>
                  {p.term_kept ? 'term kept' : 'term dropped'}
                </span>
              )}
            </div>
            {showText && (
              <div className="grid md:grid-cols-2 gap-2 mt-1 text-xs text-base-content/70">
                <div className="whitespace-pre-wrap">{p.a.text}</div>
                <div className="whitespace-pre-wrap">{p.b.text}</div>
              </div>
            )}
            {p.shared && <p className="text-xs mt-1"><span className="font-medium">Shared:</span> {p.shared}</p>}
            {p.differs && !/^(nothing material|nothing|none|n\/a)\.?$/i.test(p.differs.trim()) && (
              <p className="text-xs text-base-content/60"><span className="font-medium">Differs:</span> {p.differs}</p>
            )}
            <div className="flex flex-wrap gap-1 mt-1 items-center">
              {p.theme_ids.map(t => (
                <span key={t} className={`badge badge-xs ${t === focusTheme ? 'badge-primary' : 'badge-outline'}`}>{payload.theme_labels[t] || t}</span>
              ))}
              {p.raw_tags.length > 0 && <span className="text-[11px] text-base-content/40" title="The model's raw tags for this pair">raw: {p.raw_tags.join(' · ')}</span>}
            </div>
            {(p.useful_terms.length > 0 || p.dropped_terms.length > 0) && (
              <div className="text-[11px] mt-0.5">
                <span className="text-base-content/50">in both texts: </span>
                {p.useful_terms.map(t => <span key={'u' + t} className="mr-1.5 text-base-content/80">{t}</span>)}
                {p.dropped_terms.map(t => <span key={'d' + t} className="mr-1.5 text-base-content/35 line-through" title="dropped by the model as non-informative">{t}</span>)}
                {!p.terms_judged && <span className="text-warning"> (not yet judged by the model: rebuild the themes)</span>}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function describeRule(r: VocabRule): string {
  switch (r.op) {
    case 'merge': return `merged “${r.source}” into “${r.target}”`;
    case 'move': return `moved “${r.theme}” to group “${r.parent_label}”`;
    case 'nest': return `nested “${r.theme}” under “${r.under}”`;
    case 'rename': return r.theme ? `renamed theme “${r.theme}” to “${r.label}”` : `renamed group “${r.parent}” to “${r.label}”`;
    case 'delete_parent': return `deleted group “${r.parent}”`;
    default: return r.op;
  }
}

export default function ThemeVocabularyManager({onChanged}: {onChanged?: () => void}) {
  const [view, setView] = useState<VocabView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState('');
  const [picked, setPicked] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>(null);
  const [text, setText] = useState('');
  const [parentEdit, setParentEdit] = useState<{id: string; label: string} | null>(null);
  const [tab, setTab] = useState<Tab>('vocabulary');
  // Pairs behind the picked theme (vocabulary tab) or the picked term (terms tab)
  const [pairs, setPairs] = useState<PairsPayload | null>(null);
  const [pairsFor, setPairsFor] = useState<string | null>(null);
  const [pairsBusy, setPairsBusy] = useState(false);
  const [terms, setTerms] = useState<{terms: TermEntry[]; n_pairs: number; judged_pairs: number} | null>(null);
  const [termFilter, setTermFilter] = useState('');
  const [termSort, setTermSort] = useState<'pairs' | 'useful' | 'dropped' | 'term'>('pairs');

  const getJson = async (path: string) => {
    const r = await fetch(`${apiUrl}/api/metadata-clustering/${path}`, {credentials: 'include'});
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
    return d;
  };
  const loadPairs = async (path: string, key: string) => {
    setPairsBusy(true);
    setError(null);
    try {
      setPairs(await getJson(path));
      setPairsFor(key);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setPairsBusy(false);
    }
  };
  useEffect(() => {
    if (tab === 'terms' && !terms) getJson('terms').then(setTerms).catch(e => setError(e.message));
  }, [tab, terms]);
  const visibleTerms = useMemo(() => {
    const f = termFilter.trim().toLowerCase();
    const list = (terms?.terms || []).filter(t => !f || t.term.includes(f) || (t.theme_label || '').includes(f));
    list.sort((a, b) => termSort === 'term' ? a.term.localeCompare(b.term) : (b[termSort] - a[termSort]) || b.pairs - a.pairs);
    return list.slice(0, 400);
  }, [terms, termFilter, termSort]);

  const load = async () => {
    try {
      const r = await fetch(`${apiUrl}/api/metadata-clustering/vocabulary`, {credentials: 'include'});
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${r.status}`);
      }
      setView(await r.json());
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
  };
  useEffect(() => { load(); }, []);

  const post = async (path: string, body?: any) => {
    setBusy(true);
    setError(null);
    try {
      const r = await fetch(`${apiUrl}/api/metadata-clustering/vocabulary/${path}`, {
        method: 'POST',
        credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: body ? JSON.stringify(body) : undefined,
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setView(data);
      setMode(null);
      setText('');
      onChanged?.();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const themeById = useMemo(() => Object.fromEntries((view?.themes || []).map(t => [t.id, t])), [view]);
  const pickedTheme = picked ? themeById[picked] : null;
  const filterLower = filter.trim().toLowerCase();
  const byParent = useMemo(() => {
    const out: Record<string, VocabTheme[]> = {};
    for (const t of view?.themes || []) {
      if (filterLower && !t.label.includes(filterLower) && !t.raw.some(r => r.includes(filterLower))) continue;
      (out[t.parent] ||= []).push(t);
    }
    for (const lst of Object.values(out)) lst.sort((a, b) => b.studies - a.studies || a.label.localeCompare(b.label));
    return out;
  }, [view, filterLower]);

  // The second theme of a merge or nest: typed label, matched exactly against the vocabulary.
  const target = useMemo(() => {
    const l = text.trim().toLowerCase();
    return l ? (view?.themes || []).find(t => t.label === l && t.id !== picked) || null : null;
  }, [text, view, picked]);

  const act = () => {
    if (!pickedTheme) return;
    if (mode === 'rename') post('edit', {op: 'rename', theme: pickedTheme.id, label: text});
    else if (mode === 'move') {
      const existing = (view?.parents || []).find(p => p.label === text.trim().toLowerCase());
      post('edit', existing ? {op: 'move', theme: pickedTheme.id, parent: existing.id} : {op: 'move', theme: pickedTheme.id, parent_label: text});
    } else if (mode === 'merge' && target) post('edit', {op: 'merge', source: pickedTheme.id, target: target.id});
    else if (mode === 'nest' && target) post('edit', {op: 'nest', theme: pickedTheme.id, under: target.id});
  };

  const toggleHidden = async (t: VocabTheme) => {
    setBusy(true);
    try {
      await setThemeExcluded(t.id, !t.excluded);
      await load();
      onChanged?.();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  if (!view && !error) return <p className="text-sm text-base-content/60">Loading the vocabulary…</p>;

  const canAct = !!pickedTheme && !busy && (
    mode === 'rename' ? text.trim().length > 0 && text.trim().toLowerCase() !== pickedTheme.label
    : mode === 'move' ? text.trim().length > 0
    : (mode === 'merge' || mode === 'nest') ? !!target
    : false);

  return (
    <div className="space-y-3">
      {error && <div className="alert alert-error text-sm">{error}</div>}
      <div className="tabs tabs-boxed w-fit">
        <button className={`tab ${tab === 'vocabulary' ? 'tab-active' : ''}`} onClick={() => setTab('vocabulary')}>Vocabulary (themes)</button>
        <button className={`tab ${tab === 'terms' ? 'tab-active' : ''}`} onClick={() => setTab('terms')} title="The words and phrases both texts of a pair contain, found before the model runs, with the model's verdict on each">Common terms (n-grams)</button>
      </div>

      {tab === 'terms' && (
        <div className="space-y-3">
          {!terms ? <p className="text-sm text-base-content/60">Loading the common terms…</p> : (
            <>
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <input className="input input-bordered input-sm w-56" placeholder="Find a term or theme…" value={termFilter} onChange={e => setTermFilter(e.target.value)} />
                <span className="text-base-content/60">{terms.terms.length} distinct terms across {terms.n_pairs} pairs · {terms.judged_pairs} pairs judged by the model</span>
                <span className="ml-auto text-xs text-base-content/50">sort by</span>
                <div className="join">
                  {(['pairs', 'useful', 'dropped', 'term'] as const).map(k => (
                    <button key={k} className={`btn btn-xs join-item ${termSort === k ? 'btn-active' : ''}`} onClick={() => setTermSort(k)}>{k}</button>
                  ))}
                </div>
              </div>
              {terms.terms.length === 0 && (
                <p className="text-sm text-base-content/60">No common terms stored yet: run &quot;Rebuild themes only&quot; (it computes them and asks the model about each).</p>
              )}
              <div className="grid lg:grid-cols-2 gap-3 items-start">
                <div className="overflow-auto max-h-[32rem] rounded border border-base-300 bg-base-100">
                  <table className="table table-xs table-pin-rows">
                    <thead><tr><th>Term</th><th className="text-right" title="pairs whose both texts contain it">pairs</th><th className="text-right" title="pairs where the model kept it as informative">kept</th><th className="text-right" title="pairs where the model dropped it">dropped</th><th>theme</th></tr></thead>
                    <tbody>
                      {visibleTerms.map(t => (
                        <tr key={t.term} className={`cursor-pointer hover:bg-primary/5 ${pairsFor === 'term:' + t.term ? 'bg-primary/10' : ''}`}
                            onClick={() => loadPairs(`terms/pairs?term=${encodeURIComponent(t.term)}`, 'term:' + t.term)}>
                          <td className="font-mono text-xs">{t.term}</td>
                          <td className="text-right">{t.pairs}</td>
                          <td className="text-right text-success">{t.useful || ''}</td>
                          <td className="text-right text-base-content/50">{t.dropped || ''}</td>
                          <td className="text-xs">{t.theme_label ? <span className="badge badge-xs badge-outline">{t.theme_label}</span> : <span className="text-base-content/30">—</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  {pairsBusy ? <span className="loading loading-spinner loading-sm" /> : pairs && pairsFor?.startsWith('term:') ? (
                    <>
                      <h4 className="font-semibold text-sm mb-1">Pairs containing “{pairs.term}” <span className="font-normal text-base-content/60">({pairs.kept} kept by the model, {(pairs.n_pairs || 0) - (pairs.kept || 0)} dropped)</span></h4>
                      <PairList payload={pairs} />
                    </>
                  ) : <p className="text-sm text-base-content/50">Click a term to list the pairs that share it, with the model&apos;s verdict on each.</p>}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {tab === 'vocabulary' && view && (
        <>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <input className="input input-bordered input-sm w-56" placeholder="Find a theme or raw tag…" value={filter} onChange={e => setFilter(e.target.value)} />
            <span className="text-base-content/60">
              {view.themes.length} themes in {view.parents.length} groups · {view.n_studies} studies
            </span>
            <button className="btn btn-xs btn-outline ml-auto" onClick={() => post('undo')} disabled={busy || !view.undo_available}
                    title="Restore the vocabulary from before the last edit (one step)">
              Undo last edit
            </button>
          </div>

          {/* Action bar for the picked theme */}
          <div className={`rounded-md border p-2 text-sm ${pickedTheme ? 'border-primary bg-primary/5' : 'border-base-300 bg-base-100'}`}>
            {pickedTheme ? (
              <div className="flex flex-wrap items-center gap-2">
                <span>Selected: <strong>{pickedTheme.label}</strong> <span className="text-base-content/50">({pickedTheme.studies} studies)</span></span>
                <button className={`btn btn-xs ${mode === 'rename' ? 'btn-primary' : 'btn-outline'}`} onClick={() => { setMode('rename'); setText(pickedTheme.label); }}>Rename</button>
                <button className={`btn btn-xs ${mode === 'move' ? 'btn-primary' : 'btn-outline'}`} onClick={() => { setMode('move'); setText(''); }}>Move to group</button>
                <button className={`btn btn-xs ${mode === 'merge' ? 'btn-primary' : 'btn-outline'}`} onClick={() => { setMode('merge'); setText(''); }}
                        title="Fold this theme into another one: its studies and raw tags move there, and it disappears">Merge into…</button>
                <button className={`btn btn-xs ${mode === 'nest' ? 'btn-primary' : 'btn-outline'}`} onClick={() => { setMode('nest'); setText(''); }}
                        title="Make another theme its super-tag: a group named after that theme holds both (two levels only)">Nest under…</button>
                <button className="btn btn-xs btn-outline" onClick={() => toggleHidden(pickedTheme)} disabled={busy}>
                  {pickedTheme.excluded ? 'Unhide' : 'Hide from views'}
                </button>
                <button className={`btn btn-xs ${pairsFor === 'theme:' + pickedTheme.id ? 'btn-primary' : 'btn-outline'}`} disabled={pairsBusy}
                        onClick={() => loadPairs(`vocabulary/theme/${encodeURIComponent(pickedTheme.id)}/pairs`, 'theme:' + pickedTheme.id)}
                        title="Every matched pair that carries this theme, with its explanation and common terms">
                  {pairsBusy ? <span className="loading loading-spinner loading-xs" /> : 'Pairs'}
                </button>
                <button className="btn btn-xs btn-ghost" onClick={() => { setPicked(null); setMode(null); setText(''); }}>Clear</button>
                {mode && (
                  <div className="flex items-center gap-2 w-full mt-1">
                    <span className="text-base-content/70">
                      {mode === 'rename' ? 'New name:' : mode === 'move' ? 'Group (existing or new):' : mode === 'merge' ? 'Merge into theme:' : 'Super-tag theme:'}
                    </span>
                    <input
                      className="input input-bordered input-sm w-72"
                      list={mode === 'move' ? 'vocab-parents' : mode === 'rename' ? undefined : 'vocab-themes'}
                      value={text}
                      onChange={e => setText(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter' && canAct) act(); }}
                      autoFocus
                    />
                    <button className="btn btn-sm btn-primary" onClick={act} disabled={!canAct}>
                      {busy ? <span className="loading loading-spinner loading-xs" /> : 'Apply'}
                    </button>
                    {(mode === 'merge' || mode === 'nest') && text.trim() && !target && (
                      <span className="text-xs text-warning">pick an existing theme from the list</span>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <span className="text-base-content/60">Click a theme below to rename it, move it to a group, merge it into another theme, nest it under another theme, or hide it. Click a group name to rename or delete the group.</span>
            )}
          </div>
          <datalist id="vocab-themes">
            {view.themes.filter(t => t.id !== picked).map(t => <option key={t.id} value={t.label} />)}
          </datalist>
          <datalist id="vocab-parents">
            {view.parents.map(p => <option key={p.id} value={p.label} />)}
          </datalist>

          {pairs && pairsFor === 'theme:' + picked && pairs.theme && (
            <div className="rounded-md border border-primary/40 bg-base-100 p-3">
              <div className="flex items-center gap-2 mb-2">
                <h4 className="font-semibold text-sm">Pairs tagged “{pairs.theme.label}”</h4>
                {pairs.theme.raw.length > 0 && <span className="text-xs text-base-content/50" title="raw tags merged into this theme">raw: {pairs.theme.raw.slice(0, 8).join(' · ')}{pairs.theme.raw.length > 8 ? ' …' : ''}</span>}
                <button className="btn btn-ghost btn-xs ml-auto" onClick={() => { setPairs(null); setPairsFor(null); }}>×</button>
              </div>
              <PairList payload={pairs} focusTheme={pairs.theme.id} />
            </div>
          )}

          {/* Groups and their themes */}
          <div className="grid gap-3 md:grid-cols-2">
            {view.parents.map(p => (
              <div key={p.id} className="rounded-md border border-base-300 bg-base-100 p-2">
                <div className="flex items-center gap-2 mb-1">
                  {parentEdit?.id === p.id ? (
                    <>
                      <input className="input input-bordered input-xs w-56" value={parentEdit.label} autoFocus
                             onChange={e => setParentEdit({id: p.id, label: e.target.value})}
                             onKeyDown={e => {
                               if (e.key === 'Enter' && parentEdit.label.trim()) { post('edit', {op: 'rename', parent: p.id, label: parentEdit.label}); setParentEdit(null); }
                               if (e.key === 'Escape') setParentEdit(null);
                             }} />
                      <button className="btn btn-xs btn-primary" disabled={!parentEdit.label.trim() || busy}
                              onClick={() => { post('edit', {op: 'rename', parent: p.id, label: parentEdit.label}); setParentEdit(null); }}>Save</button>
                      <button className="btn btn-xs btn-ghost" onClick={() => setParentEdit(null)}>Cancel</button>
                    </>
                  ) : (
                    <>
                      <button className="font-semibold text-sm link link-hover" onClick={() => setParentEdit({id: p.id, label: p.label})} title="Rename this group">
                        {p.label}
                      </button>
                      <span className="text-xs text-base-content/50">{(byParent[p.id] || []).length} themes</span>
                      {p.id !== 'p-unsorted' && (
                        <button className="btn btn-xs btn-ghost text-error ml-auto" disabled={busy}
                                onClick={() => { if (window.confirm(`Delete the group “${p.label}”? Its themes move to “unsorted themes”.`)) post('edit', {op: 'delete_parent', parent: p.id}); }}>
                          Delete group
                        </button>
                      )}
                    </>
                  )}
                </div>
                <div className="flex flex-wrap gap-1">
                  {(byParent[p.id] || []).map(t => (
                    <button
                      key={t.id}
                      className={`badge badge-sm cursor-pointer ${picked === t.id ? 'badge-primary' : t.excluded ? 'badge-ghost line-through opacity-60' : 'badge-outline'}`}
                      title={`${t.studies} of ${view.n_studies} studies${t.raw.length ? `\nraw tags: ${t.raw.join(' · ')}` : ''}${t.excluded ? '\nhidden from the views' : ''}`}
                      onClick={() => { setPicked(picked === t.id ? null : t.id); setMode(null); setText(''); }}
                    >
                      {t.label} <span className="opacity-60 ml-1">{t.studies}</span>
                    </button>
                  ))}
                  {(byParent[p.id] || []).length === 0 && <span className="text-xs text-base-content/40 italic">no theme matches the filter</span>}
                </div>
              </div>
            ))}
          </div>

          {view.rules.length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-base-content/70">Edits applied ({view.rules.length}), re-applied after every run or rebuild</summary>
              <ol className="list-decimal ml-5 mt-1 space-y-0.5">
                {view.rules.map((r, i) => (
                  <li key={i}>{describeRule(r)}{r.by ? ` — ${r.by}` : ''}{r.at ? `, ${new Date(r.at).toLocaleString()}` : ''}</li>
                ))}
              </ol>
            </details>
          )}
        </>
      )}
    </div>
  );
}
