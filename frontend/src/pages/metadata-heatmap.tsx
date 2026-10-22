'use client';

// Metadata heatmap: studies (rows) x themes (columns grouped by parent theme),
// from the metadata clustering's theme view. Same frame as the concept
// coverage heatmap: sticky headers, shaded cells, a details drawer.
import React, {useEffect, useMemo, useState} from 'react';
import Link from 'next/link';
import {Grid, X} from 'react-feather';
import {apiUrl} from '@/utils';
import {
  Theme,
  ThemeView,
  ThemeWeight,
  activeThemes,
  fetchThemeView,
  parentColor,
  setThemeExcluded,
  strengthColor,
  themeWeight
} from '@/components/metadataThemes';

const WEIGHT_HELP: Record<ThemeWeight, string> = {
  balance: 'Balance: how evenly a theme splits the studies (1 when half carry it, 0 when all or almost none do). Penalises universal and one-pair themes alike.',
  specificity: 'Specificity: rarer is heavier (log(N/n)/log(N)); 0 when every study carries the theme.'
};

export default function MetadataHeatmapPage() {
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [view, setView] = useState<ThemeView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [minSpec, setMinSpec] = useState(0);
  const [weight, setWeight] = useState<ThemeWeight>('balance');
  const [showExcluded, setShowExcluded] = useState(false);
  const [picks, setPicks] = useState<string[]>([]);
  const [cell, setCell] = useState<{study: string; theme: Theme} | null>(null);

  useEffect(() => {
    fetch(`${apiUrl}/admin/check`, {credentials: 'include'})
      .then(r => (r.ok ? r.json() : {is_admin: false}))
      .then(d => setIsAdmin(!!d.is_admin))
      .catch(() => setIsAdmin(false));
  }, []);

  const load = () =>
    fetchThemeView()
      .then(v => {
        setView(v);
        setError(null);
      })
      .catch(e => setError(e.message));
  useEffect(() => {
    if (isAdmin) load();
  }, [isAdmin]);

  const columns = useMemo(() => {
    if (!view) return [] as {parent: {id: string; label: string}; themes: Theme[]}[];
    const active = activeThemes(view, minSpec, weight);
    const pool = showExcluded ? view.themes.filter(t => t.studies > 0 && (t.excluded || themeWeight(t, weight) >= minSpec)) : active;
    return view.parents
      .map(p => ({parent: p, themes: pool.filter(t => t.parent === p.id).sort((a, b) => b.studies - a.studies || a.label.localeCompare(b.label))}))
      .filter(g => g.themes.length > 0);
  }, [view, minSpec, showExcluded, weight]);

  const toggleExclude = async (t: Theme) => {
    try {
      await setThemeExcluded(t.id, !t.excluded);
      await load();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const togglePick = (s: string) =>
    setPicks(p => {
      const i = p.indexOf(s);
      if (i >= 0) return p.filter(x => x !== s);
      const next = [...p, s];
      return next.length > 2 ? next.slice(1) : next;
    });

  if (isAdmin === false) {
    return <div className="container mx-auto p-6"><div className="alert alert-warning">This page is for administrators only.</div></div>;
  }

  const compare = picks.length === 2 && view ? (() => {
    const [a, b] = picks;
    const shared = columns.flatMap(g => g.themes).filter(t => view.table[a]?.[t.id] && view.table[b]?.[t.id]);
    return {a, b, shared: shared.sort((x, y) => themeWeight(y, weight) - themeWeight(x, weight))};
  })() : null;

  return (
    <div className="min-h-screen bg-base-100">
      <div className="px-4 py-6">
        <div className="mb-4">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Grid size={26} />
            Metadata Heatmap
          </h1>
          <p className="text-base-content/60 mt-1 text-sm">
            Rows are studies; columns are the themes the metadata matching found in common between studies, grouped under
            their parent themes. A cell is shaded by how strongly the study carries the theme (darker is stronger). Click a
            cell for the matched excerpts; click two study names to see exactly what they share. A theme carried by every
            study tells nothing apart, and one carried by a single pair tells little; the weight and its floor below hide both ends.
          </p>
        </div>

        {isAdmin === null || (!view && !error) ? (
          <div className="flex justify-center items-center py-20">
            <span className="loading loading-spinner loading-lg"></span>
          </div>
        ) : error ? (
          <div className="alert alert-warning text-sm">
            <span>
              {error}{' '}
              <Link href="/metadata-clustering" className="link">Open the metadata clustering page</Link> to run the matching.
            </span>
          </div>
        ) : view && (
          <>
            <div className="flex flex-wrap gap-3 mb-4 items-center">
              <div className="flex items-center gap-2" title={WEIGHT_HELP[weight]}>
                <span className="text-sm text-base-content/60 whitespace-nowrap">Weight themes by</span>
                <div className="join">
                  {(['balance', 'specificity'] as ThemeWeight[]).map(w => (
                    <button key={w} className={`btn btn-xs join-item ${weight === w ? 'btn-active' : ''}`} onClick={() => setWeight(w)}>{w}</button>
                  ))}
                </div>
              </div>
              <div className="divider divider-horizontal mx-0"></div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-base-content/60 whitespace-nowrap">
                  Min {weight}: <span className="font-semibold text-base-content">{minSpec.toFixed(2)}</span>
                </span>
                <input id="minspec" type="range" min={0} max={0.9} step={0.05} value={minSpec} onChange={e => setMinSpec(Number(e.target.value))} className="range range-sm range-primary w-48" />
              </div>
              <div className="divider divider-horizontal mx-0"></div>
              <label className="label cursor-pointer gap-2">
                <input id="showexcl" type="checkbox" className="checkbox checkbox-sm" checked={showExcluded} onChange={e => setShowExcluded(e.target.checked)} />
                <span className="label-text text-sm">Show excluded themes ({view.excluded.length})</span>
              </label>
              <div className="divider divider-horizontal mx-0"></div>
              <span className="text-sm text-base-content/60">
                {view.studies.length} studies · {columns.reduce((n, g) => n + g.themes.length, 0)} themes · computed {view.finished ? new Date(view.finished).toLocaleString() : ''}{view.model ? ` with ${view.model}` : ''}
              </span>
              {picks.length > 0 && (
                <button className="btn btn-xs btn-ghost ml-auto" onClick={() => setPicks([])}>clear selection</button>
              )}
            </div>

            <div className="flex gap-4 items-start">
              <div className="min-w-0 flex-1">
                <div className="overflow-auto border border-base-300 rounded-lg" style={{maxHeight: '85vh'}}>
                  <table className="border-separate border-spacing-0 text-xs">
                    <thead>
                      <tr>
                        <th className="sticky top-0 left-0 z-30 bg-base-200 border-b border-r border-base-300 px-2 py-1 text-left min-w-[10rem]" rowSpan={2}>Study</th>
                        {columns.map(g => (
                          <th key={g.parent.id} colSpan={g.themes.length} className="sticky top-0 z-10 bg-base-200 border-b border-r border-base-300 px-2 py-1 text-left whitespace-nowrap">
                            <span className="inline-flex items-center gap-1.5">
                              <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{background: parentColor(view.parents, g.parent.id)}} />
                              <span className="font-semibold uppercase tracking-wide text-[10px]">{g.parent.label}</span>
                            </span>
                          </th>
                        ))}
                      </tr>
                      <tr>
                        {columns.flatMap(g =>
                          g.themes.map(t => (
                            <th key={t.id} className={`group/col sticky z-10 bg-base-200 border-b border-r border-base-300 px-1 align-bottom ${t.excluded ? 'opacity-50' : ''}`} style={{top: '1.75rem'}}>
                              <div className="flex items-end justify-center gap-0.5">
                                <span
                                  className="overflow-hidden text-ellipsis whitespace-nowrap font-medium"
                                  style={{writingMode: 'vertical-rl', transform: 'rotate(180deg)', maxHeight: '9rem', minHeight: '9rem'}}
                                  title={`${t.label}\n${t.studies} of ${view.n_studies} studies · balance ${t.balance.toFixed(2)} · specificity ${t.specificity.toFixed(2)}\nraw terms: ${t.raw.join(', ')}`}
                                >
                                  {t.label}
                                </span>
                                <button
                                  type="button"
                                  className="rounded p-0.5 opacity-40 transition-opacity hover:bg-base-300 group-hover/col:opacity-100"
                                  title={t.excluded ? 'Include this theme again' : 'Exclude this theme from the views (results are kept)'}
                                  aria-label={`${t.excluded ? 'Include' : 'Exclude'} theme ${t.label}`}
                                  onClick={() => toggleExclude(t)}
                                >
                                  <X size={13} />
                                </button>
                              </div>
                            </th>
                          ))
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {view.studies.map(s => (
                        <tr key={s} className="group/row">
                          <td className={`sticky left-0 z-20 bg-base-100 border-r border-b border-base-200 px-2 py-1 min-w-[10rem] max-w-[12rem] ${picks.includes(s) ? 'font-bold' : 'font-semibold'}`}>
                            <button type="button" className="truncate text-left hover:underline max-w-full" title="Click to select; select two studies to compare them" onClick={() => togglePick(s)}>
                              {s}
                            </button>
                          </td>
                          {columns.flatMap(g =>
                            g.themes.map(t => {
                              const c = view.table[s]?.[t.id];
                              const dim = compare ? !(compare.shared.some(x => x.id === t.id) && picks.includes(s)) && picks.includes(s) : false;
                              return (
                                <td
                                  key={t.id}
                                  className={`border-r border-b border-base-200 px-1 py-1 text-center whitespace-nowrap ${c ? 'cursor-pointer hover:outline hover:outline-1 hover:outline-primary' : ''} ${picks.includes(s) ? 'ring-inset ring-1 ring-base-content/40' : ''}`}
                                  style={{backgroundColor: c ? strengthColor(c.strength) : undefined, opacity: dim ? 0.3 : 1, minWidth: '2.4rem'}}
                                  title={c ? `${s} · ${t.label}\nstrength ${c.strength.toFixed(1)} · ${c.count} matching pair${c.count === 1 ? '' : 's'}` : `${s} · ${t.label}: not shared with another study`}
                                  onClick={() => c && setCell({study: s, theme: t})}
                                >
                                  {c ? c.count : ''}
                                </td>
                              );
                            })
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-2 text-xs text-base-content/40">
                  Shading follows the strength of the best match that carried the theme for that study (strong 1.0, partial 0.6); the number is how many matched pairs carried it.
                  Balance is the binary entropy of n / N (n studies carrying the theme out of N), 1 bit when half carry it and 0 when all do; specificity is log(N / n) / log(N). Themes come from pairs, so n is at least 2.
                </p>
              </div>

              {(cell || compare) && (
                <aside className="w-80 shrink-0 border border-base-300 rounded-lg bg-base-100 p-3 text-sm space-y-2 sticky top-4">
                  <div className="flex items-start justify-between gap-2">
                    <h3 className="font-semibold">{cell ? `${cell.study} · ${cell.theme.label}` : `${compare!.a} · ${compare!.b}`}</h3>
                    <button className="btn btn-ghost btn-xs btn-circle" onClick={() => { setCell(null); setPicks([]); }} aria-label="Close"><X size={14} /></button>
                  </div>
                  {cell && (
                    <>
                      <div className="text-xs text-base-content/60">
                        {view.parents.find(p => p.id === cell.theme.parent)?.label} · {cell.theme.studies} of {view.n_studies} studies · balance {cell.theme.balance.toFixed(2)} · specificity {cell.theme.specificity.toFixed(2)}
                      </div>
                      {(view.table[cell.study]?.[cell.theme.id]?.excerpts || []).map((x, i) => (
                        <div key={i} className="border-t border-base-200 pt-2">
                          <div className="text-[11px] uppercase tracking-wide text-base-content/50">{x.field} · matched with {x.with}{x.strength ? ` (${x.strength})` : ''}</div>
                          <div className="mt-0.5">“{x.text}”</div>
                          <div className="mt-0.5 text-base-content/70">{x.with}: “{x.with_text}”</div>
                        </div>
                      ))}
                      <div className="text-xs text-base-content/60 pt-2 border-t border-base-200">
                        Also in: {view.studies.filter(s => s !== cell.study && view.table[s]?.[cell.theme.id]).join(', ') || 'no other study'}
                      </div>
                    </>
                  )}
                  {!cell && compare && (
                    <>
                      <div className="text-xs text-base-content/60">{compare.shared.length} shared themes, heaviest first by {weight}.</div>
                      {compare.shared.length === 0 && <p className="italic text-base-content/50">Nothing in common above this floor.</p>}
                      {compare.shared.map(t => (
                        <div key={t.id} className="border-t border-base-200 pt-2">
                          <div className="flex items-center gap-1.5">
                            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{background: parentColor(view.parents, t.parent)}} />
                            <span className="font-medium">{t.label}</span>
                            <span className="text-xs text-base-content/50 ml-auto">{weight} {themeWeight(t, weight).toFixed(2)}</span>
                          </div>
                          <div className="mt-0.5 text-xs text-base-content/70">{compare.a}: “{view.table[compare.a][t.id].excerpts[0]?.text}”</div>
                          <div className="text-xs text-base-content/70">{compare.b}: “{view.table[compare.b][t.id].excerpts[0]?.text}”</div>
                        </div>
                      ))}
                    </>
                  )}
                </aside>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
