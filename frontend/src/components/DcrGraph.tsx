'use client';

// Dependency graph of a Data Clean Room: data nodes first, the compute nodes
// that read them next, airlocks and the nodes analysts run last. Layout is a
// plain longest-path layering (a node sits one stage after the last of its
// dependencies), drawn as SVG, left-to-right by default or top-to-bottom.
import React, {useMemo, useState} from 'react';
import {RoomNode, RoomNodeKind, RoomOverview} from '@/components/ai/chatClient';

// Clearly distinct hues per kind: the four data kinds are the ones readers
// must tell apart at a glance (blue / green / purple / orange).
export const KIND_STYLE: Record<RoomNodeKind, {fill: string; stroke: string; label: string}> = {
  cohort_data: {fill: '#dbeafe', stroke: '#1d4ed8', label: 'cohort data (full, patient-level)'},
  dictionary: {fill: '#dcfce7', stroke: '#15803d', label: 'data dictionary'},
  shuffled: {fill: '#ede9fe', stroke: '#6d28d9', label: 'shuffled sample (synthetic rows)'},
  mapping: {fill: '#ffedd5', stroke: '#c2410c', label: 'cross-study mapping file'},
  python: {fill: '#fefce8', stroke: '#a16207', label: 'python compute node'},
  airlock: {fill: '#fee2e2', stroke: '#f87171', label: 'airlock (controlled exit for row-level data)'},
  environment: {fill: '#f5f5f4', stroke: '#78716c', label: 'custom python environment'},
  other: {fill: '#f5f5f4', stroke: '#78716c', label: 'other'}
};

export type GraphOrientation = 'horizontal' | 'vertical';

const BOX_W = 220;
const BOX_H = 30; // data nodes, airlocks, environments
const BOX_H_TALL = 46; // python nodes: a second line lists what they write
const PAD = 16;
// Distance between stages and between nodes in a stage, per orientation.
const STAGE_GAP = {horizontal: 250, vertical: 90};
const NODE_GAP = {horizontal: 58, vertical: 236};

function boxHeight(n: RoomNode): number {
  return n.kind === 'python' ? BOX_H_TALL : BOX_H;
}

/** Second line of a python node: its output files, abbreviated. */
function outputsLabel(n: RoomNode): string {
  const outs = n.outputs || [];
  const kinds = n.output_kinds || [];
  let text: string;
  if (outs.length > 0) {
    const more = outs.length > 2 ? ` +${outs.length - 2}` : '';
    text = `→ ${outs.slice(0, 2).join(', ')}${more}`;
  } else if (kinds.length > 0) {
    text = `→ ${kinds.map(k => k.replace(/ \(.*\)$/, '')).join(', ')}`;
  } else {
    text = n.writes ? '→ writes files' : '→ no file output found';
  }
  if (n.external_writes) text += ' + package files';
  return text.length > 36 ? text.slice(0, 35) + '…' : text;
}

function layer(nodes: RoomNode[]): Map<string, number> {
  const byName = new Map(nodes.map(n => [n.name, n]));
  const memo = new Map<string, number>();
  const visiting = new Set<string>();
  const depth = (name: string): number => {
    if (memo.has(name)) return memo.get(name)!;
    if (visiting.has(name)) return 0; // cycle guard; a DCR graph is acyclic
    visiting.add(name);
    const n = byName.get(name);
    const d = n && n.dependencies.length ? 1 + Math.max(...n.dependencies.map(depth)) : 0;
    visiting.delete(name);
    memo.set(name, d);
    return d;
  };
  nodes.forEach(n => depth(n.name));
  return memo;
}

/** The part of a room around one node: the node, what it reads and what reads it. */
export function neighbourhood(room: RoomOverview, name: string): RoomOverview {
  const keep = new Set<string>([name]);
  room.edges.forEach(e => {
    if (e.from === name) keep.add(e.to);
    if (e.to === name) keep.add(e.from);
  });
  return {
    ...room,
    nodes: room.nodes.filter(n => keep.has(n.name)),
    edges: room.edges.filter(e => keep.has(e.from) && keep.has(e.to)),
    exits: room.exits.filter(e => keep.has(e.node))
  };
}

export function DcrGraph({
  room,
  selected,
  onSelect,
  onHover,
  selectableKinds = 'python',
  compact = false
}: {
  room: RoomOverview;
  selected?: string | null;
  // Clicking a selectable node asks the parent to open it.
  onSelect?: (name: string, node: RoomNode) => void;
  // The node under the pointer (null when none), for a details box beside the graph.
  onHover?: (name: string | null, node: RoomNode | null) => void;
  // Which nodes react to a click: python nodes only (the explain views) or every node (the graph overlay).
  selectableKinds?: 'python' | 'all';
  // Smaller chrome (no legend text), for the neighbourhood view of one node.
  compact?: boolean;
}) {
  const [hover, setHoverState] = useState<string | null>(null);
  const setHover = (name: string | null) => {
    setHoverState(name);
    onHover?.(name, name ? room.nodes.find(n => n.name === name) || null : null);
  };
  const [orientation, setOrientation] = useState<GraphOrientation>('horizontal');
  const horizontal = orientation === 'horizontal';

  const {positions, width, height, stages} = useMemo(() => {
    const depths = layer(room.nodes);
    const cols = new Map<number, RoomNode[]>();
    room.nodes.forEach(n => {
      const d = depths.get(n.name) || 0;
      if (!cols.has(d)) cols.set(d, []);
      cols.get(d)!.push(n);
    });
    // Within a stage: data kinds first, then by name, so each cohort's nodes stay together.
    const order: Record<RoomNodeKind, number> = {cohort_data: 0, dictionary: 1, shuffled: 2, mapping: 3, environment: 4, python: 5, airlock: 6, other: 7};
    const positions = new Map<string, {x: number; y: number}>();
    let maxPerStage = 0;
    const stageGap = STAGE_GAP[orientation];
    const nodeGap = NODE_GAP[orientation];
    Array.from(cols.keys()).sort((a, b) => a - b).forEach(c => {
      const list = cols.get(c)!.sort((a, b) => order[a.kind] - order[b.kind] || a.name.localeCompare(b.name));
      maxPerStage = Math.max(maxPerStage, list.length);
      list.forEach((n, i) => {
        if (horizontal) positions.set(n.name, {x: PAD + c * stageGap, y: PAD + i * nodeGap});
        else positions.set(n.name, {x: PAD + i * nodeGap, y: PAD + c * stageGap});
      });
    });
    const nStages = Math.max(1, cols.size);
    const width = horizontal ? PAD * 2 + (nStages - 1) * stageGap + BOX_W : PAD * 2 + Math.max(1, maxPerStage - 1) * nodeGap + BOX_W;
    const height = horizontal ? PAD * 2 + Math.max(1, maxPerStage - 1) * nodeGap + BOX_H_TALL : PAD * 2 + (nStages - 1) * stageGap + BOX_H_TALL;
    return {positions, width, height, stages: nStages};
  }, [room, orientation, horizontal]);

  const byName = useMemo(() => new Map(room.nodes.map(n => [n.name, n])), [room]);
  const focus = hover || selected || null;
  const related = useMemo(() => {
    if (!focus) return null;
    const s = new Set<string>([focus]);
    room.edges.forEach(e => {
      if (e.from === focus) s.add(e.to);
      if (e.to === focus) s.add(e.from);
    });
    return s;
  }, [focus, room]);

  const exitNodes = useMemo(() => new Set(room.exits.map(e => e.node)), [room]);
  const unprovisioned = room.nodes.filter(n => n.provisioned === false).length;

  return (
    <div className="h-full flex flex-col">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2 text-[11px] text-base-content/70 border-b border-base-300">
        {(Object.keys(KIND_STYLE) as RoomNodeKind[])
          .filter(k => k !== 'other' && room.nodes.some(n => n.kind === k))
          .map(k => (
            <span key={k} className="inline-flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm border" style={{background: KIND_STYLE[k].fill, borderColor: KIND_STYLE[k].stroke}} />
              {KIND_STYLE[k].label}
            </span>
          ))}
        <span className="inline-flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm border-2 border-dashed border-base-content/60" /> exit: analysts see its output
        </span>
        {room.provisioning?.known && (
          <span className="inline-flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-sm border border-dashed" style={{borderColor: '#9ca3af', background: '#f9fafb'}} /> no data provisioned yet
          </span>
        )}
        {/* Orientation: deliberately unobtrusive. */}
        <button
          type="button"
          className="ml-auto btn btn-ghost btn-xs text-base-content/40 hover:text-base-content"
          onClick={() => setOrientation(o => (o === 'horizontal' ? 'vertical' : 'horizontal'))}
          title={horizontal ? 'Show the flow top-to-bottom' : 'Show the flow left-to-right'}
          aria-label="Toggle graph orientation"
        >
          {horizontal ? '⇅' : '⇆'}
        </button>
        {!compact && (
          <span className="basis-full text-base-content/50">
            Arrows point from a node to the nodes that read it. The second line of a compute node lists the files it writes.
            Hover a node for details; click a compute node to open it.
            {room.provisioning?.known
              ? unprovisioned > 0
                ? ` ${unprovisioned} of ${room.provisioning.data_nodes} data nodes have no data yet; nodes reading them cannot run.`
                : ' All data nodes hold data.'
              : ' Provisioning status could not be retrieved.'}
          </span>
        )}
      </div>
      <div className="flex-1 overflow-auto bg-base-200/40">
        <svg width={width} height={height} className="block" style={{minWidth: '100%'}}>
          <defs>
            <marker id="dcr-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
            </marker>
            <marker id="dcr-arrow-hi" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#1d4ed8" />
            </marker>
          </defs>
          {room.edges.map((e, i) => {
            const a = positions.get(e.from);
            const b = positions.get(e.to);
            if (!a || !b) return null;
            const na = byName.get(e.from);
            const nb = byName.get(e.to);
            const ha = na ? boxHeight(na) : BOX_H;
            const hb = nb ? boxHeight(nb) : BOX_H;
            let d: string;
            if (horizontal) {
              const x1 = a.x + BOX_W, y1 = a.y + ha / 2, x2 = b.x, y2 = b.y + hb / 2;
              const dx = Math.max(30, (x2 - x1) / 2);
              d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
            } else {
              const x1 = a.x + BOX_W / 2, y1 = a.y + ha, x2 = b.x + BOX_W / 2, y2 = b.y;
              const dy = Math.max(20, (y2 - y1) / 2);
              d = `M ${x1} ${y1} C ${x1} ${y1 + dy}, ${x2} ${y2 - dy}, ${x2} ${y2}`;
            }
            const hi = related ? related.has(e.from) && related.has(e.to) && (e.from === focus || e.to === focus) : false;
            const dim = related && !hi;
            return (
              <path
                key={i}
                d={d}
                fill="none"
                stroke={hi ? '#1d4ed8' : '#94a3b8'}
                strokeWidth={hi ? 2 : 1}
                opacity={dim ? 0.15 : 0.9}
                markerEnd={hi ? 'url(#dcr-arrow-hi)' : 'url(#dcr-arrow)'}
              />
            );
          })}
          {room.nodes.map(n => {
            const p = positions.get(n.name);
            if (!p) return null;
            const st = KIND_STYLE[n.kind];
            const isExit = exitNodes.has(n.name);
            const noData = n.provisioned === false;
            const dim = related && !related.has(n.name);
            const clickable = !!onSelect && (selectableKinds === 'all' || n.kind === 'python');
            const h = boxHeight(n);
            const maxChars = noData ? 24 : 30;
            const label = n.name.length > maxChars ? n.name.slice(0, maxChars - 1) + '…' : n.name;
            const title =
              `${n.name}\n${st.label}` +
              (noData ? '\nno data provisioned yet' : n.provisioned === true ? `\ndata provisioned${n.provisioned_by ? ` by ${n.provisioned_by}` : ''}` : '') +
              (n.dependencies.length ? `\nreads: ${n.dependencies.join(', ')}` : '') +
              (n.kind === 'python' ? `\nrunnable by ${n.runners} participant(s)` : '') +
              (n.outputs?.length ? `\nwrites: ${n.outputs.join(', ')}` : n.output_kinds?.length ? `\nwrites: ${n.output_kinds.join(', ')}` : '') +
              (n.external_writes ? `\nalso the files of ${(n.external_packages || []).join(', ') || 'its package'}: see the output folder` : '') +
              (n.prints ? '\nprints or logs text' : '') +
              (n.kind === 'airlock' && n.quota_bytes ? `\nquota ${Math.round(n.quota_bytes / 1e6)} MB` : '') +
              (n.external_packages?.length ? `\nexternal code: ${n.external_packages.join(', ')}` : '') +
              (clickable ? (selectableKinds === 'all' ? '\n(click to pin its details)' : '\n(click to open this node)') : '');
            return (
              <g
                key={n.name}
                transform={`translate(${p.x}, ${p.y})`}
                opacity={dim ? 0.3 : 1}
                style={{cursor: clickable ? 'pointer' : 'default'}}
                onMouseEnter={() => setHover(n.name)}
                onMouseLeave={() => setHover(null)}
                onClick={() => clickable && onSelect!(n.name, n)}
              >
                <title>{title}</title>
                <rect
                  width={BOX_W}
                  height={h}
                  rx={6}
                  fill={noData ? '#f9fafb' : st.fill}
                  stroke={selected === n.name ? '#1d4ed8' : noData ? '#9ca3af' : st.stroke}
                  strokeWidth={selected === n.name ? 2.5 : isExit ? 2 : 1.2}
                  strokeDasharray={noData ? '4 3' : isExit ? '5 3' : undefined}
                />
                {/* Unprovisioned data node: dashed grey box, an empty-tray icon and a quiet "no data yet" tag; a normal state until the owner uploads. Provisioned ones carry no mark. */}
                {noData && (
                  <g transform="translate(8, 9)" fill="none" stroke="#9ca3af" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M 1 6 L 1 11 L 13 11 L 13 6" />
                    <path d="M 1 6 L 3.5 1 L 10.5 1 L 13 6" />
                    <path d="M 1 6 L 4.5 6 L 5.5 8 L 8.5 8 L 9.5 6 L 13 6" />
                  </g>
                )}
                <text
                  x={noData ? 26 : 10}
                  y={BOX_H / 2 + 4}
                  fontSize={11}
                  fontFamily="ui-monospace, monospace"
                  fill={noData ? '#6b7280' : '#111827'}
                >
                  {label}
                </text>
                {noData && (
                  <text x={BOX_W - 8} y={BOX_H / 2 + 4} fontSize={9} fill="#9ca3af" textAnchor="end">
                    no data yet
                  </text>
                )}
                {n.kind === 'python' && (
                  <text x={10} y={BOX_H + 10} fontSize={9.5} fill={n.outputs?.length ? '#374151' : '#6b7280'} fontStyle={n.outputs?.length ? 'normal' : 'italic'}>
                    {outputsLabel(n)}
                  </text>
                )}
                {n.external_packages && n.external_packages.length > 0 && (
                  <text x={BOX_W - 12} y={BOX_H / 2 + 4} fontSize={11} fill="#b91c1c" fontWeight={700}>
                    !
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      {!compact && (
        <div className="px-4 py-1.5 text-[11px] text-base-content/50 border-t border-base-300">
          {room.nodes.length} nodes in {stages} stages · {room.exits.length} exit{room.exits.length === 1 ? '' : 's'} where analysts see output
        </div>
      )}
    </div>
  );
}
