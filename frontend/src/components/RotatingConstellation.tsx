'use client';

// The rotating metadata constellation: studies re-cluster every 30 seconds by
// the category of metadata that carried their shared themes (objective,
// population, outcomes, inclusion, exclusion). Clusters are the theme groups
// of the vocabulary (general tags), placed as fixed anchors; within a cluster,
// studies sharing the same specific themes attract each other. A study carried
// by several groups is shown either as one node pulled between them ("edges")
// or as one node per group, the copies tethered and marked ("copies"); the
// default comes from the admin setting, and both can be tried here. Node
// colour is the study type, node size equal or the log of participants or
// variables; a solid outline marks a dictionary, a centre dot an EDA profile.
import React, {useEffect, useMemo, useRef, useState} from 'react';
import {Pause, Play, SkipBack, SkipForward} from 'react-feather';
import {parentColor, ThemeView} from '@/components/metadataThemes';

export type ConstellationMode = 'edges' | 'copies';
type SizeBy = 'equal' | 'participants' | 'variables';

const W = 1100;
const H = 720;
const FRAME_SECONDS = 30;
const TOP_EDGES = 4;          // strongest links kept per node: the rest only tangle
const COPY_SHARE = 0.25;      // copies mode: a group gets a copy when its share is at least this fraction of the study's largest
const ANCHOR_R = Math.min(W, H) * 0.34;

interface Anchor {
  id: string;
  label: string;
  x: number;
  y: number;
  color: string;
}

interface NodeSpec {
  key: string;        // study, or study|parent in copies mode
  study: string;
  parent: string | null;   // copies mode: the group this copy belongs to
  copies: number;     // how many nodes this study has in this frame
  share: Record<string, number>;   // group id -> share of the study's weight in this category
  themes: string[];   // theme ids carried in this category
  connected: boolean;
}

interface EdgeSpec {
  a: string;          // node keys
  b: string;
  w: number;
  tether?: boolean;   // copies of one study
}

interface Frame {
  category: string;
  anchors: Anchor[];
  nodes: NodeSpec[];
  edges: EdgeSpec[];
  connected: number;
  unconnected: number;
  hasCategoryInfo: boolean;
}

interface Body {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

const studyOf = (key: string): string => key.split('|')[0];

const TYPE_PALETTE = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2', '#db2777', '#65a30d', '#9333ea', '#ea580c'];

function studyTypeColor(types: string[], t: string | null | undefined): string {
  if (!t) return '#9ca3af';
  const i = types.indexOf(t);
  return i < 0 ? '#9ca3af' : TYPE_PALETTE[i % TYPE_PALETTE.length];
}

function buildFrame(view: ThemeView, category: string, mode: ConstellationMode): Frame {
  const themeById = Object.fromEntries(view.themes.map(t => [t.id, t]));
  const parentIndex = Object.fromEntries(view.parents.map((p, i) => [p.id, i]));
  // What each study carries in this category: theme -> weight (min strength x balance)
  const carried: Record<string, Record<string, number>> = {};
  let anyGroups = false;
  for (const study of view.studies) {
    const row = view.table[study] || {};
    const out: Record<string, number> = {};
    for (const [tid, cell] of Object.entries(row)) {
      const t = themeById[tid];
      if (!t || t.excluded || !t.studies) continue;
      const groups = cell.groups;
      if (groups && Object.keys(groups).length) anyGroups = true;
      // Results without per-category counts (older runs) show the same picture in every frame.
      const inCategory = groups && Object.keys(groups).length ? (groups[category] || 0) > 0 : true;
      if (!inCategory) continue;
      out[tid] = cell.strength * t.balance;
    }
    carried[study] = out;
  }
  // Group shares per study
  const share: Record<string, Record<string, number>> = {};
  for (const study of view.studies) {
    const s: Record<string, number> = {};
    for (const [tid, w] of Object.entries(carried[study])) {
      const p = themeById[tid].parent;
      s[p] = (s[p] || 0) + w;
    }
    share[study] = s;
  }
  // Anchors: the groups present in this category, on a circle at angles fixed by their index
  const present = view.parents.filter(p => view.studies.some(s => (share[s][p.id] || 0) > 0));
  const anchors: Anchor[] = present.map(p => {
    const angle = (parentIndex[p.id] / Math.max(1, view.parents.length)) * Math.PI * 2 - Math.PI / 2;
    return {id: p.id, label: p.label, x: W / 2 + ANCHOR_R * Math.cos(angle), y: H / 2 + ANCHOR_R * Math.sin(angle),
            color: parentColor(view.parents, p.id)};
  });
  // Pair weights in this category
  const pairW: Record<string, Record<string, number>> = {};
  for (let i = 0; i < view.studies.length; i++) {
    for (let j = i + 1; j < view.studies.length; j++) {
      const a = view.studies[i], b = view.studies[j];
      let w = 0;
      for (const [tid, wa] of Object.entries(carried[a])) {
        const wb = carried[b][tid];
        if (wb) w += Math.min(wa, wb);
      }
      if (w > 0) {
        (pairW[a] ||= {})[b] = w;
        (pairW[b] ||= {})[a] = w;
      }
    }
  }
  const topEdges = (a: string): [string, number][] =>
    Object.entries(pairW[a] || {}).sort((x, y) => y[1] - x[1]).slice(0, TOP_EDGES);

  const nodes: NodeSpec[] = [];
  const edges: EdgeSpec[] = [];
  const edgeSeen = new Set<string>();
  if (mode === 'edges') {
    for (const study of view.studies) {
      const themes = Object.keys(carried[study]);
      nodes.push({key: study, study, parent: null, copies: 1, share: share[study], themes, connected: Object.keys(pairW[study] || {}).length > 0});
    }
    for (const a of view.studies) {
      for (const [b, w] of topEdges(a)) {
        const k = a < b ? `${a}|${b}` : `${b}|${a}`;
        if (edgeSeen.has(k)) continue;
        edgeSeen.add(k);
        edges.push({a, b, w});
      }
    }
  } else {
    // One node per (study, group) whose share is a real part of the study's weight
    const copiesOf: Record<string, string[]> = {};
    for (const study of view.studies) {
      const s = share[study];
      const max = Math.max(0, ...Object.values(s));
      const groups = max > 0 ? Object.keys(s).filter(p => s[p] >= COPY_SHARE * max) : [];
      copiesOf[study] = groups;
      if (groups.length === 0) {
        nodes.push({key: study, study, parent: null, copies: 1, share: s, themes: [], connected: false});
        continue;
      }
      for (const p of groups) {
        const themes = Object.keys(carried[study]).filter(t => themeById[t].parent === p);
        nodes.push({key: `${study}|${p}`, study, parent: p, copies: groups.length, share: s, themes, connected: Object.keys(pairW[study] || {}).length > 0});
      }
      // tethers between the copies of one study
      for (let i = 0; i < groups.length; i++) {
        for (let j = i + 1; j < groups.length; j++) edges.push({a: `${study}|${groups[i]}`, b: `${study}|${groups[j]}`, w: 0, tether: true});
      }
    }
    // links within a group: shared themes of that group, strongest few per copy
    for (const a of view.studies) {
      for (const p of copiesOf[a]) {
        const cands: [string, number][] = [];
        for (const b of view.studies) {
          if (b === a || !copiesOf[b].includes(p)) continue;
          let w = 0;
          for (const [tid, wa] of Object.entries(carried[a])) {
            if (themeById[tid].parent !== p) continue;
            const wb = carried[b][tid];
            if (wb) w += Math.min(wa, wb);
          }
          if (w > 0) cands.push([b, w]);
        }
        cands.sort((x, y) => y[1] - x[1]);
        for (const [b, w] of cands.slice(0, TOP_EDGES)) {
          const ka = `${a}|${p}`, kb = `${b}|${p}`;
          const k = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
          if (edgeSeen.has(k)) continue;
          edgeSeen.add(k);
          edges.push({a: ka, b: kb, w});
        }
      }
    }
  }
  const connectedStudies = new Set(nodes.filter(n => n.connected).map(n => n.study));
  return {category, anchors, nodes, edges, connected: connectedStudies.size,
          unconnected: view.studies.length - connectedStudies.size, hasCategoryInfo: anyGroups};
}

export default function RotatingConstellation({view}: {view: ThemeView}) {
  const categories = view.categories || [];
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const [left, setLeft] = useState(FRAME_SECONDS);
  const [mode, setMode] = useState<ConstellationMode>((view.constellation_mode as ConstellationMode) || 'edges');
  const [sizeBy, setSizeBy] = useState<SizeBy>('equal');
  const [picked, setPicked] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const category = categories[index] || '';
  const frame = useMemo(() => buildFrame(view, category, mode), [view, category, mode]);
  const types = useMemo(() => Array.from(new Set(Object.values(view.study_info || {}).map(i => i.type).filter(Boolean))).sort() as string[], [view]);

  // Countdown and rotation
  useEffect(() => {
    if (paused || categories.length < 2) return;
    const t = setInterval(() => {
      setLeft(l => {
        if (l <= 1) {
          setIndex(i => (i + 1) % categories.length);
          return FRAME_SECONDS;
        }
        return l - 1;
      });
    }, 1000);
    return () => clearInterval(t);
  }, [paused, categories.length]);
  const goTo = (i: number) => { setIndex(((i % categories.length) + categories.length) % categories.length); setLeft(FRAME_SECONDS); };

  // Simulation state lives in refs; positions persist across frames so a study can be followed.
  const bodies = useRef<Record<string, Body>>({});
  const alpha = useRef(1);
  const frameRef = useRef(frame);
  const sizeRef = useRef<Record<string, number>>({});

  const radius = (study: string): number => {
    const info = view.study_info?.[study];
    if (sizeBy === 'participants') {
      const n = info?.participants || 0;
      return n > 0 ? 6 + Math.min(14, 4 * Math.log10(n)) : 6;
    }
    if (sizeBy === 'variables') {
      const n = info?.n_variables || 0;
      return n > 0 ? 6 + Math.min(14, 5 * Math.log10(n)) : 6;
    }
    return 9;
  };

  useEffect(() => {
    frameRef.current = frame;
    alpha.current = 1;
    const anchorOf = Object.fromEntries(frame.anchors.map(a => [a.id, a]));
    for (const n of frame.nodes) {
      if (bodies.current[n.key]) continue;
      // A new node starts near its anchor (copies) or near the study's other position, else the centre.
      const base = n.parent && anchorOf[n.parent] ? anchorOf[n.parent] : bodies.current[n.study] || {x: W / 2, y: H / 2};
      bodies.current[n.key] = {x: base.x + (Math.random() - 0.5) * 60, y: base.y + (Math.random() - 0.5) * 60, vx: 0, vy: 0};
    }
  }, [frame]);

  useEffect(() => {
    const r: Record<string, number> = {};
    for (const s of view.studies) r[s] = radius(s);
    sizeRef.current = r;
    alpha.current = Math.max(alpha.current, 0.3);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sizeBy, view]);

  useEffect(() => {
    let raf = 0;
    const step = () => {
      const f = frameRef.current;
      const b = bodies.current;
      const a = alpha.current;
      if (a > 0.005) {
        const anchorOf = Object.fromEntries(f.anchors.map(x => [x.id, x]));
        const keys = f.nodes.map(n => n.key);
        // repulsion
        for (let i = 0; i < keys.length; i++) {
          for (let j = i + 1; j < keys.length; j++) {
            const p = b[keys[i]], q = b[keys[j]];
            if (!p || !q) continue;
            let dx = q.x - p.x, dy = q.y - p.y;
            let d2 = dx * dx + dy * dy;
            if (d2 < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 1; }
            const d = Math.sqrt(d2);
            const minD = (sizeRef.current[f.nodes[i].study] || 9) + (sizeRef.current[f.nodes[j].study] || 9) + 6;
            const force = (900 / d2 + (d < minD ? (minD - d) * 0.5 : 0)) * a;
            p.vx -= (dx / d) * force; p.vy -= (dy / d) * force;
            q.vx += (dx / d) * force; q.vy += (dy / d) * force;
          }
        }
        // springs
        for (const e of f.edges) {
          const p = b[e.a], q = b[e.b];
          if (!p || !q) continue;
          const dx = q.x - p.x, dy = q.y - p.y;
          const d = Math.max(1, Math.sqrt(dx * dx + dy * dy));
          const rest = e.tether ? 160 : Math.max(40, 120 - 60 * Math.min(1, e.w));
          const k = e.tether ? 0.004 : 0.02 * Math.min(1, 0.3 + e.w);
          const force = (d - rest) * k * a;
          p.vx += (dx / d) * force; p.vy += (dy / d) * force;
          q.vx -= (dx / d) * force; q.vy -= (dy / d) * force;
        }
        // anchors and the outer ring for the unconnected
        f.nodes.forEach((n, i) => {
          const p = b[n.key];
          if (!p) return;
          if (n.parent && anchorOf[n.parent]) {
            const an = anchorOf[n.parent];
            p.vx += (an.x - p.x) * 0.05 * a; p.vy += (an.y - p.y) * 0.05 * a;
          } else if (n.themes.length) {
            const total = Object.values(n.share).reduce((s, v) => s + v, 0) || 1;
            for (const [pid, w] of Object.entries(n.share)) {
              const an = anchorOf[pid];
              if (!an) continue;
              const k = 0.045 * (w / total);
              p.vx += (an.x - p.x) * k * a; p.vy += (an.y - p.y) * k * a;
            }
          } else {
            const angle = (i / Math.max(1, f.nodes.length)) * Math.PI * 2;
            const tx = W / 2 + (ANCHOR_R + 110) * Math.cos(angle), ty = H / 2 + (ANCHOR_R + 110) * Math.sin(angle);
            p.vx += (tx - p.x) * 0.03 * a; p.vy += (ty - p.y) * 0.03 * a;
          }
          // keep inside
          p.vx += (W / 2 - p.x) * 0.002 * a; p.vy += (H / 2 - p.y) * 0.002 * a;
        });
        for (const k of keys) {
          const p = b[k];
          if (!p) continue;
          p.vx *= 0.6; p.vy *= 0.6;
          p.x = Math.max(20, Math.min(W - 20, p.x + p.vx));
          p.y = Math.max(20, Math.min(H - 20, p.y + p.vy));
        }
        alpha.current = a * 0.97;
        setTick(t => t + 1);
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);

  const pos = bodies.current;
  const pickedNode = picked ? frame.nodes.find(n => n.study === picked) : null;
  const themeById = useMemo(() => Object.fromEntries(view.themes.map(t => [t.id, t])), [view]);
  const parentLabel = useMemo(() => Object.fromEntries(view.parents.map(p => [p.id, p.label])), [view]);
  const pickedThemes = useMemo(() => {
    if (!picked) return [] as {parent: string; themes: string[]}[];
    const ids = new Set<string>();
    for (const n of frame.nodes) if (n.study === picked) n.themes.forEach(t => ids.add(t));
    const byParent: Record<string, string[]> = {};
    for (const id of ids) (byParent[themeById[id].parent] ||= []).push(themeById[id].label);
    return Object.entries(byParent).map(([parent, themes]) => ({parent, themes: themes.sort()}));
  }, [picked, frame, themeById]);
  const info = (s: string) => view.study_info?.[s];
  const progress = 1 - left / FRAME_SECONDS;

  return (
    <div className="space-y-3" data-tick={tick}>
      {/* Banner: what the clustering is on, and the countdown */}
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-primary/40 bg-primary/5 px-4 py-2">
        <div className="text-xs uppercase tracking-wide text-base-content/60">Clustered on</div>
        <div className="text-2xl font-bold">{category || '—'}</div>
        <div className="flex items-center gap-1 ml-2">
          {categories.map((c, i) => (
            <button key={c} className={`btn btn-xs ${i === index ? 'btn-primary' : 'btn-ghost'}`} onClick={() => goTo(i)} title={`Show ${c} now`}>{c}</button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <svg width="34" height="34" viewBox="0 0 34 34" aria-label={`${left} seconds to the next category`}>
            <circle cx="17" cy="17" r="14" fill="none" stroke="#e5e7eb" strokeWidth="4" />
            <circle cx="17" cy="17" r="14" fill="none" stroke="#2563eb" strokeWidth="4" strokeDasharray={`${2 * Math.PI * 14}`}
                    strokeDashoffset={`${2 * Math.PI * 14 * progress}`} transform="rotate(-90 17 17)" style={{transition: 'stroke-dashoffset 1s linear'}} />
            <text x="17" y="21" textAnchor="middle" fontSize="11" fontWeight="600" fill="currentColor">{left}</text>
          </svg>
          <span className="text-sm text-base-content/70 whitespace-nowrap">{paused ? 'paused' : `next in ${left} s`}</span>
          <button className="btn btn-xs btn-outline" onClick={() => goTo(index - 1)} title="Previous category"><SkipBack size={12} /></button>
          <button className="btn btn-xs btn-outline" onClick={() => setPaused(p => !p)} title={paused ? 'Resume the rotation' : 'Pause the rotation'}>
            {paused ? <Play size={12} /> : <Pause size={12} />}
          </button>
          <button className="btn btn-xs btn-outline" onClick={() => goTo(index + 1)} title="Next category"><SkipForward size={12} /></button>
        </div>
      </div>
      <div className="text-xs text-base-content/60">
        {frame.anchors.length} groups · {frame.connected} studies linked · {frame.unconnected} without a shared theme in this category
        {!frame.hasCategoryInfo && ' · these results predate per-category counts: every frame shows all themes (rebuild the themes to fix)'}
      </div>

      <div className="flex gap-4 items-start">
        <div className="min-w-0 flex-1 border border-base-300 rounded-lg bg-base-100 overflow-hidden relative">
          <svg viewBox={`0 0 ${W} ${H}`} className="block w-full h-auto" role="img" aria-label={`Studies clustered by ${category}`}>
            {frame.anchors.map(a => (
              <g key={a.id}>
                <circle cx={a.x} cy={a.y} r={90} fill={a.color} opacity={0.07} />
                <text x={a.x} y={a.y - 98} textAnchor="middle" fontSize="13" fontWeight="600" fill={a.color}>{a.label}</text>
              </g>
            ))}
            {frame.edges.map(e => {
              const p = pos[e.a], q = pos[e.b];
              if (!p || !q) return null;
              const on = !picked || studyOf(e.a) === picked || studyOf(e.b) === picked;
              return e.tether ? (
                <line key={`${e.a}~${e.b}`} x1={p.x} y1={p.y} x2={q.x} y2={q.y} stroke="#6b7280" strokeWidth={1} strokeDasharray="3 4" opacity={on ? 0.5 : 0.15} />
              ) : (
                <line key={`${e.a}~${e.b}`} x1={p.x} y1={p.y} x2={q.x} y2={q.y} stroke="#64748b" strokeWidth={0.8 + 2 * Math.min(1, e.w)} opacity={on ? (picked ? 0.6 : 0.18) : 0.05} />
              );
            })}
            {frame.nodes.map(n => {
              const p = pos[n.key];
              if (!p) return null;
              const i = info(n.study);
              const r = sizeRef.current[n.study] || 9;
              const color = studyTypeColor(types, i?.type);
              const isPicked = picked === n.study;
              const dim = picked && !isPicked && !frame.edges.some(e => (studyOf(e.a) === picked && e.b === n.key) || (studyOf(e.b) === picked && e.a === n.key));
              return (
                <g key={n.key} transform={`translate(${p.x},${p.y})`} style={{cursor: 'pointer'}} opacity={dim ? 0.3 : 1}
                   onClick={() => { setPicked(isPicked ? null : n.study); if (!isPicked) setPaused(true); }}
                   onMouseEnter={() => setHover(n.key)} onMouseLeave={() => setHover(null)}>
                  <title>{`${n.study}${i?.type ? ` · ${i.type}` : ''}${i?.participants ? ` · ${i.participants.toLocaleString()} participants` : ''}${i?.n_variables ? ` · ${i.n_variables} variables` : ''}\n${n.themes.length} themes in ${category}${n.copies > 1 ? `\nshown ${n.copies} times (one per group)` : ''}${i?.has_dictionary ? '' : '\nno data dictionary uploaded'}${i?.has_eda ? '\nEDA profile available' : ''}`}</title>
                  <circle r={r} fill={color} fillOpacity={i?.has_dictionary ? 0.85 : 0.35}
                          stroke={isPicked ? '#111827' : color} strokeWidth={isPicked ? 3 : 1.5}
                          strokeDasharray={n.copies > 1 ? '3 2' : i?.has_dictionary ? undefined : '2 2'} />
                  {i?.has_eda && <circle r={Math.max(1.8, r * 0.22)} fill="#ffffff" />}
                  {n.copies > 1 && <text x={r + 2} y={-r + 4} fontSize="9" fill="#6b7280">×{n.copies}</text>}
                  {(hover === n.key || isPicked || r >= 12) && (
                    <text y={r + 11} textAnchor="middle" fontSize="10" fill="#374151">{n.study}</text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>

        <aside className="w-72 shrink-0 space-y-3 text-sm">
          <div className="rounded-lg border border-base-300 bg-base-100 p-3 space-y-2">
            <div>
              <div className="text-xs uppercase tracking-wide text-base-content/50 mb-1">Node size</div>
              <div className="join">
                {([['equal', 'equal'], ['participants', 'log participants'], ['variables', 'log variables']] as [SizeBy, string][]).map(([k, l]) => (
                  <button key={k} className={`btn btn-xs join-item ${sizeBy === k ? 'btn-active' : ''}`} onClick={() => setSizeBy(k)}>{l}</button>
                ))}
              </div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-base-content/50 mb-1">Multi-group studies</div>
              <div className="join">
                {(['edges', 'copies'] as ConstellationMode[]).map(m => (
                  <button key={m} className={`btn btn-xs join-item ${mode === m ? 'btn-active' : ''}`} onClick={() => setMode(m)}
                          title={m === 'edges' ? 'One node per study, pulled between the groups it belongs to' : 'One node per group, the copies tethered and marked ×n'}>
                    {m}{view.constellation_mode === m ? ' (default)' : ''}
                  </button>
                ))}
              </div>
              <div className="text-xs text-base-content/50 mt-1">The default is set on the admin settings page.</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-base-content/50 mb-1">Colour: study type</div>
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
                {types.map(t => (
                  <span key={t} className="inline-flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-full" style={{background: studyTypeColor(types, t)}} />{t}</span>
                ))}
                <span className="inline-flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-full bg-gray-400" />unknown</span>
              </div>
            </div>
            <div className="text-xs text-base-content/60 space-y-0.5">
              <div><span className="inline-block w-2.5 h-2.5 rounded-full border-2 border-gray-500 align-middle mr-1" />solid outline: data dictionary uploaded; dashed and pale: none</div>
              <div><span className="inline-block w-2.5 h-2.5 rounded-full bg-gray-500 align-middle mr-1 relative"><span className="absolute inset-[3px] rounded-full bg-white" /></span>centre dot: EDA profile available</div>
            </div>
          </div>

          {pickedNode && (
            <div className="rounded-lg border border-base-300 bg-base-100 p-3 space-y-2">
              <div className="flex items-start justify-between gap-2">
                <h3 className="font-semibold">{picked}</h3>
                <button className="btn btn-ghost btn-xs" onClick={() => setPicked(null)}>×</button>
              </div>
              <div className="text-xs text-base-content/60">
                {info(picked!)?.type || 'type unknown'}{info(picked!)?.participants ? ` · ${info(picked!)!.participants!.toLocaleString()} participants` : ''}{info(picked!)?.n_variables ? ` · ${info(picked!)!.n_variables} variables` : ''}
              </div>
              <div className="text-xs text-base-content/60">Themes in {category}:</div>
              {pickedThemes.length === 0 && <div className="text-xs italic text-base-content/50">none in this category</div>}
              {pickedThemes.map(g => (
                <div key={g.parent}>
                  <div className="text-xs font-semibold" style={{color: parentColor(view.parents, g.parent)}}>{parentLabel[g.parent] || g.parent}</div>
                  <div className="flex flex-wrap gap-1 mt-0.5">{g.themes.map(t => <span key={t} className="badge badge-xs badge-outline">{t}</span>)}</div>
                </div>
              ))}
              <div className="text-xs text-base-content/50">The rotation is paused while a study is selected.</div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
