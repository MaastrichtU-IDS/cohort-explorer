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
      {view && (
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
