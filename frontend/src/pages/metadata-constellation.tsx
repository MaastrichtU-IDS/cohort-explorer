'use client';

// Metadata constellation: studies as nodes, linked when they share themes.
// Edge width is the number of shared themes, edge colour the dominant parent
// theme, and studies that share more sit closer (a small force layout, no
// library). Data: the metadata clustering's theme view.
import React, {useEffect, useMemo, useRef, useState} from 'react';
import Link from 'next/link';
import {Share2, X} from 'react-feather';
import {apiUrl} from '@/utils';
import LoginPrompt from '@/components/LoginPrompt';
import {
  StudyPair,
  ThemeView,
  ThemeWeight,
  dominantParent,
  fetchThemeView,
  pairsAbove,
  parentColor,
  themeWeight
} from '@/components/metadataThemes';

const WEIGHT_HELP: Record<ThemeWeight, string> = {
  balance: 'Balance: how evenly a theme splits the studies (1 when half carry it, 0 when all or almost none do). Penalises universal and one-pair themes alike.',
  specificity: 'Specificity: rarer is heavier (log(N/n)/log(N)); 0 when every study carries the theme.'
};

const W = 960;
const H = 600;

type Pt = {id: string; x: number; y: number; vx: number; vy: number; r: number};

/** A few hundred steps of a spring/repulsion layout, run once per data change. */
function layout(studies: string[], pairs: StudyPair[], themeCount: Record<string, number>): Pt[] {
  const pts: Pt[] = studies.map((id, i) => {
    const a = (i / studies.length) * Math.PI * 2;
    return {id, x: W / 2 + Math.cos(a) * 200, y: H / 2 + Math.sin(a) * 180, vx: 0, vy: 0, r: 10 + Math.min(14, (themeCount[id] || 0) * 1.2)};
  });
  const idx = Object.fromEntries(pts.map((p, i) => [p.id, i]));
  const maxS = Math.max(0.01, ...pairs.map(p => p.strength));
  for (let step = 0; step < 400; step++) {
    const t = 1 - step / 400;
    // repulsion
    for (let i = 0; i < pts.length; i++)
      for (let j = i + 1; j < pts.length; j++) {
        const a = pts[i], b = pts[j];
        let dx = b.x - a.x, dy = b.y - a.y;
        let d2 = dx * dx + dy * dy || 1;
        const f = 9000 / d2;
        const d = Math.sqrt(d2);
        dx /= d; dy /= d;
        a.vx -= dx * f; a.vy -= dy * f; b.vx += dx * f; b.vy += dy * f;
      }
    // springs: rest length shrinks with strength
    pairs.forEach(p => {
      const a = pts[idx[p.a]], b = pts[idx[p.b]];
      if (!a || !b) return;
      const rest = 300 - 220 * (p.strength / maxS);
      let dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (d - rest) * 0.02 * (0.3 + p.strength / maxS);
      dx /= d; dy /= d;
      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    });
    pts.forEach(p => {
      // gentle pull to the centre, damping, bounds
      p.vx += (W / 2 - p.x) * 0.002; p.vy += (H / 2 - p.y) * 0.002;
      p.x += p.vx * t * 0.9; p.y += p.vy * t * 0.9;
      p.vx *= 0.6; p.vy *= 0.6;
      p.x = Math.max(p.r + 60, Math.min(W - p.r - 60, p.x));
      p.y = Math.max(p.r + 24, Math.min(H - p.r - 30, p.y));
    });
  }
  return pts;
}

export default function MetadataConstellationPage() {
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [needsLogin, setNeedsLogin] = useState(false);
  const [view, setView] = useState<ThemeView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [minSpec, setMinSpec] = useState(0.2);
  const [weight, setWeight] = useState<ThemeWeight>('balance');
  const [hiddenParents, setHiddenParents] = useState<Set<string>>(new Set());
  const [picked, setPicked] = useState<string | null>(null);
  const [hoverEdge, setHoverEdge] = useState<{pair: StudyPair; x: number; y: number} | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

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
  useEffect(() => {
    if (isAdmin) fetchThemeView().then(setView).catch(e => setError(e.message));
  }, [isAdmin]);

  const themeById = useMemo(() => Object.fromEntries((view?.themes || []).map(t => [t.id, t])), [view]);
  const pairs = useMemo(() => {
    if (!view) return [] as (StudyPair & {parent: string})[];
    return pairsAbove(view, minSpec, weight)
      .map(p => ({...p, parent: dominantParent(p.shared, weight)}))
      .filter(p => !hiddenParents.has(p.parent));
  }, [view, minSpec, hiddenParents, weight]);
  const themeCount = useMemo(() => {
    const out: Record<string, number> = {};
    if (view) view.studies.forEach(s => { out[s] = Object.keys(view.table[s] || {}).filter(t => themeById[t] && !themeById[t].excluded && themeWeight(themeById[t], weight) >= minSpec).length; });
    return out;
  }, [view, themeById, minSpec, weight]);
  const pts = useMemo(() => (view ? layout(view.studies, pairs, themeCount) : []), [view, pairs, themeCount]);
  const pos = useMemo(() => Object.fromEntries(pts.map(p => [p.id, p])), [pts]);
  const maxN = Math.max(1, ...pairs.map(p => p.n));

  const neighbours = useMemo(() => {
    if (!picked) return null;
    const s = new Set<string>([picked]);
    pairs.forEach(p => { if (p.a === picked) s.add(p.b); if (p.b === picked) s.add(p.a); });
    return s;
  }, [picked, pairs]);

  if (needsLogin) return <LoginPrompt message="Authenticate to access this page (administrators only)" />;
  if (isAdmin === false) {
    return <div className="container mx-auto p-6"><div className="alert alert-warning">This page is for administrators only.</div></div>;
  }

  const pickedPairs = picked ? pairs.filter(p => p.a === picked || p.b === picked).sort((x, y) => y.strength - x.strength) : [];

  return (
    <div className="min-h-screen bg-base-100">
      <div className="px-4 py-6">
        <div className="mb-4">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Share2 size={26} />
            Metadata Constellation
          </h1>
          <p className="text-base-content/60 mt-1 text-sm">
            Studies linked by the themes their metadata share. Edge thickness is the number of shared themes, edge colour the
            parent theme that dominates between the two studies, and studies that share more sit closer together. Hover an
            edge for its themes; click a study to list what it shares with each neighbour. Themes carried by every study, or
            by a single pair, count little; the weight and its floor keep the ones that divide the catalogue.
          </p>
        </div>

        {isAdmin === null || (!view && !error) ? (
          <div className="flex justify-center items-center py-20"><span className="loading loading-spinner loading-lg"></span></div>
        ) : error ? (
          <div className="alert alert-warning text-sm"><span>{error} <Link href="/metadata-clustering" className="link">Open the metadata clustering page</Link> to run the matching.</span></div>
        ) : view && (
          <>
            <div className="flex flex-wrap gap-3 mb-3 items-center">
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
                <span className="text-sm text-base-content/60 whitespace-nowrap">Min {weight}: <span className="font-semibold text-base-content">{minSpec.toFixed(2)}</span></span>
                <input id="minspec" type="range" min={0} max={0.9} step={0.05} value={minSpec} onChange={e => setMinSpec(Number(e.target.value))} className="range range-sm range-primary w-48" />
              </div>
              <div className="divider divider-horizontal mx-0"></div>
              <div className="flex flex-wrap gap-1.5">
                {view.parents.map(p => {
                  const off = hiddenParents.has(p.id);
                  return (
                    <button
                      key={p.id}
                      className={`btn btn-xs gap-1.5 ${off ? 'btn-ghost opacity-50' : 'btn-outline'}`}
                      onClick={() => setHiddenParents(h => { const n = new Set(h); if (n.has(p.id)) n.delete(p.id); else n.add(p.id); return n; })}
                      title={off ? 'Show edges dominated by this theme' : 'Hide edges dominated by this theme'}
                    >
                      <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{background: parentColor(view.parents, p.id)}} />
                      {p.label}
                    </button>
                  );
                })}
              </div>
              <span className="text-xs text-base-content/50 ml-auto">{view.studies.length} studies · {pairs.length} links · computed {view.finished ? new Date(view.finished).toLocaleString() : ''}</span>
            </div>

            <div className="flex gap-4 items-start">
              <div className="min-w-0 flex-1 border border-base-300 rounded-lg bg-base-100 overflow-hidden relative">
                <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="block w-full h-auto" role="img" aria-label="Studies linked by shared metadata themes">
                  {pairs.map(p => {
                    const a = pos[p.a], b = pos[p.b];
                    if (!a || !b) return null;
                    const on = neighbours ? p.a === picked || p.b === picked : true;
                    return (
                      <line
                        key={`${p.a}|${p.b}`}
                        x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                        stroke={parentColor(view.parents, p.parent)}
                        strokeWidth={1 + (p.n / maxN) * 7}
                        strokeOpacity={on ? 0.6 : 0.08}
                        strokeLinecap="round"
                        style={{cursor: 'pointer'}}
                        onMouseMove={e => {
                          const rect = svgRef.current?.getBoundingClientRect();
                          setHoverEdge({pair: p, x: e.clientX - (rect?.left || 0), y: e.clientY - (rect?.top || 0)});
                        }}
                        onMouseLeave={() => setHoverEdge(null)}
                      />
                    );
                  })}
                  {pts.map(p => (
                    <g key={p.id} transform={`translate(${p.x},${p.y})`} style={{cursor: 'pointer'}} opacity={neighbours && !neighbours.has(p.id) ? 0.3 : 1} onClick={() => setPicked(picked === p.id ? null : p.id)}>
                      <title>{`${p.id}\n${themeCount[p.id] || 0} themes at this specificity\n(click to see what it shares with its neighbours)`}</title>
                      <circle r={p.r} fill="#e5edfa" stroke={picked === p.id ? '#1d4ed8' : '#2a5db0'} strokeWidth={picked === p.id ? 3 : 1.5} />
                      <text y={p.r + 13} textAnchor="middle" fontSize={11} fontFamily="ui-monospace, monospace" fill="currentColor" style={{paintOrder: 'stroke', stroke: 'hsl(var(--b1))', strokeWidth: 3, strokeLinejoin: 'round'}}>
                        {p.id}
                      </text>
                    </g>
                  ))}
                </svg>
                {hoverEdge && (
                  <div className="absolute pointer-events-none z-10 rounded-md border border-base-300 bg-base-100 px-2.5 py-2 text-xs shadow-lg max-w-xs" style={{left: Math.min(hoverEdge.x + 12, (svgRef.current?.clientWidth || 600) - 260), top: hoverEdge.y + 12}}>
                    <div className="font-semibold mb-1">{hoverEdge.pair.a} · {hoverEdge.pair.b}</div>
                    <div className="flex flex-wrap gap-1">
                      {hoverEdge.pair.shared.slice().sort((x, y) => themeWeight(y, weight) - themeWeight(x, weight)).map(s => (
                        <span key={s.theme} className="inline-flex items-center gap-1 rounded-full bg-base-200 px-2 py-0.5">
                          <span className="inline-block w-2 h-2 rounded-sm" style={{background: parentColor(view.parents, s.parent)}} />
                          {themeById[s.theme]?.label || s.theme}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {picked && (
                <aside className="w-80 shrink-0 border border-base-300 rounded-lg bg-base-100 p-3 text-sm space-y-2 sticky top-4">
                  <div className="flex items-start justify-between gap-2">
                    <h3 className="font-semibold">{picked}</h3>
                    <button className="btn btn-ghost btn-xs btn-circle" onClick={() => setPicked(null)} aria-label="Close"><X size={14} /></button>
                  </div>
                  <div className="text-xs text-base-content/60">{pickedPairs.length} linked studies, strongest first. Bars are the shared strength weighted by {weight}.</div>
                  {pickedPairs.map(p => {
                    const other = p.a === picked ? p.b : p.a;
                    const max = Math.max(0.01, ...pickedPairs.map(q => q.strength));
                    return (
                      <div key={other} className="border-t border-base-200 pt-2">
                        <div className="font-mono text-xs">{other}</div>
                        <div className="h-1.5 rounded bg-base-200 my-1 overflow-hidden"><div className="h-full bg-primary" style={{width: `${Math.round((p.strength / max) * 100)}%`}} /></div>
                        <div className="flex flex-wrap gap-1">
                          {p.shared.slice().sort((x, y) => themeWeight(y, weight) - themeWeight(x, weight)).map(s => (
                            <span key={s.theme} className="inline-flex items-center gap-1 rounded-full bg-base-200 px-2 py-0.5 text-xs">
                              <span className="inline-block w-2 h-2 rounded-sm" style={{background: parentColor(view.parents, s.parent)}} />
                              {themeById[s.theme]?.label || s.theme}
                            </span>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </aside>
              )}
            </div>
            <p className="mt-2 text-xs text-base-content/40">
              Link strength = sum over shared themes of min(strength in A, strength in B) x weight. Balance = binary entropy of n / N (1 bit when half the studies carry the theme, 0 when all do); specificity = log(N / n) / log(N). Excluded themes (set on the <Link href="/metadata-heatmap" className="link">heatmap</Link>) never count.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
