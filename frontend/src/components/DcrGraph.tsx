'use client';

// Dependency graph of a Data Clean Room: data nodes on the left, the compute
// nodes that read them to the right, airlocks and the nodes analysts run at
// the end. Layout is a plain longest-path layering (a node sits one column
// after the last of its dependencies), drawn as SVG; no library needed.
import React, {useMemo, useState} from 'react';
import {RoomNode, RoomNodeKind, RoomOverview} from '@/components/ai/chatClient';

const KIND_STYLE: Record<RoomNodeKind, {fill: string; stroke: string; label: string}> = {
  cohort_data: {fill: '#dbeafe', stroke: '#1d4ed8', label: 'cohort data (full, patient-level)'},
  dictionary: {fill: '#e0f2fe', stroke: '#0369a1', label: 'data dictionary'},
  shuffled: {fill: '#ede9fe', stroke: '#6d28d9', label: 'shuffled sample (synthetic rows)'},
  mapping: {fill: '#f1f5f9', stroke: '#475569', label: 'cross-study mapping file'},
  python: {fill: '#fef3c7', stroke: '#b45309', label: 'python compute node'},
  airlock: {fill: '#fee2e2', stroke: '#b91c1c', label: 'airlock (controlled exit for row-level data)'},
  environment: {fill: '#f5f5f4', stroke: '#78716c', label: 'custom python environment'},
  other: {fill: '#f5f5f4', stroke: '#78716c', label: 'other'}
};

const COL_W = 250;
const ROW_H = 58;
const BOX_W = 220;
const BOX_H = 30; // data nodes, airlocks, environments
const BOX_H_TALL = 46; // python nodes: a second line lists what they write
const PAD = 16;

function boxHeight(n: RoomNode): number {
  return n.kind === 'python' ? BOX_H_TALL : BOX_H;
}

/** Second line of a python node: its output files, abbreviated. */
function outputsLabel(n: RoomNode): string {
  const outs = n.outputs || [];
  if (outs.length === 0) return n.external_writes ? '→ files written by external code' : n.writes ? '→ writes files (names not literal)' : '→ no file output found';
  const shown = outs.slice(0, 2).join(', ');
  const more = outs.length > 2 ? ` +${outs.length - 2}` : '';
  const ext = n.external_writes ? ' +external' : '';
  const text = `→ ${shown}${more}${ext}`;
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
  // Environments carry no data: keep them in the first column, dimmed.
  return memo;
}

export function DcrGraph({
  room,
  selected,
  onSelect
}: {
  room: RoomOverview;
  selected?: string | null;
  // Clicking a python node asks the parent to open its script.
  onSelect?: (name: string, node: RoomNode) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);

  const {positions, width, height, columns} = useMemo(() => {
    const depths = layer(room.nodes);
    const cols = new Map<number, RoomNode[]>();
    room.nodes.forEach(n => {
      const d = depths.get(n.name) || 0;
      if (!cols.has(d)) cols.set(d, []);
      cols.get(d)!.push(n);
    });
    // Within a column: data kinds first, then by name, so each cohort's nodes stay together.
    const order: Record<RoomNodeKind, number> = {cohort_data: 0, dictionary: 1, shuffled: 2, mapping: 3, environment: 4, python: 5, airlock: 6, other: 7};
    const positions = new Map<string, {x: number; y: number}>();
    let maxRows = 0;
    Array.from(cols.keys()).sort((a, b) => a - b).forEach(c => {
      const list = cols.get(c)!.sort((a, b) => order[a.kind] - order[b.kind] || a.name.localeCompare(b.name));
      maxRows = Math.max(maxRows, list.length);
      list.forEach((n, i) => positions.set(n.name, {x: PAD + c * COL_W, y: PAD + i * ROW_H}));
    });
    const nCols = cols.size;
    return {positions, width: PAD * 2 + Math.max(1, nCols) * COL_W - (COL_W - BOX_W), height: PAD * 2 + maxRows * ROW_H + (BOX_H_TALL - BOX_H), columns: nCols};
  }, [room]);

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

  return (
    <div className="h-full flex flex-col">
      <div className="flex flex-wrap gap-x-4 gap-y-1 px-4 py-2 text-[11px] text-base-content/70 border-b border-base-300">
        {(Object.keys(KIND_STYLE) as RoomNodeKind[])
          .filter(k => k !== 'other' && room.nodes.some(n => n.kind === k))
          .map(k => (
            <span key={k} className="inline-flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm border" style={{background: KIND_STYLE[k].fill, borderColor: KIND_STYLE[k].stroke}} />
              {KIND_STYLE[k].label}
            </span>
          ))}
        <span className="inline-flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm border-2 border-dashed border-base-content/60" /> analysts can see its output (exit)
        </span>
        {room.provisioning?.known && (
          <>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-full" style={{background: '#16a34a'}} /> data provisioned
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-full border-2" style={{borderColor: '#9ca3af'}} /> no data yet
            </span>
          </>
        )}
        <span className="basis-full text-base-content/50">
          Arrows point from a node to the nodes that read it. The second line of a compute node lists the files it writes.
          Hover a node for details; click a python node to open its script.
          {room.provisioning?.known
            ? ` ${room.provisioning.provisioned} of ${room.provisioning.data_nodes} data nodes hold data.`
            : ' Provisioning status could not be retrieved.'}
        </span>
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
            const x1 = a.x + BOX_W;
            const y1 = a.y + (na ? boxHeight(na) : BOX_H) / 2;
            const x2 = b.x;
            const y2 = b.y + (nb ? boxHeight(nb) : BOX_H) / 2;
            const dx = Math.max(30, (x2 - x1) / 2);
            const hi = related ? related.has(e.from) && related.has(e.to) && (e.from === focus || e.to === focus) : false;
            const dim = related && !hi;
            return (
              <path
                key={i}
                d={`M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`}
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
            const dim = related && !related.has(n.name);
            const clickable = n.kind === 'python' && !!onSelect;
            const label = n.name.length > 30 ? n.name.slice(0, 29) + '…' : n.name;
            const h = boxHeight(n);
            const title =
              `${n.name}\n${st.label}` +
              (n.provisioned === true ? `\ndata provisioned${n.provisioned_by ? ` by ${n.provisioned_by}` : ''}` : n.provisioned === false ? '\nno data provisioned yet' : '') +
              (n.dependencies.length ? `\nreads: ${n.dependencies.join(', ')}` : '') +
              (n.kind === 'python' ? `\nrunnable by ${n.runners} participant(s)` : '') +
              (n.outputs?.length ? `\nwrites: ${n.outputs.join(', ')}` : '') +
              (n.external_writes ? '\nexternal code may write further files' : '') +
              (n.prints ? '\nprints or logs text' : '') +
              (n.kind === 'airlock' && n.quota_bytes ? `\nquota ${Math.round(n.quota_bytes / 1e6)} MB` : '') +
              (n.external_packages?.length ? `\nexternal code: ${n.external_packages.join(', ')}` : '') +
              (clickable ? '\n(click to open the script)' : '');
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
                  fill={st.fill}
                  stroke={selected === n.name ? '#1d4ed8' : st.stroke}
                  strokeWidth={selected === n.name ? 2.5 : isExit ? 2 : 1.2}
                  strokeDasharray={isExit ? '5 3' : undefined}
                  opacity={n.provisioned === false ? 0.55 : 1}
                />
                {/* Data nodes: filled dot = dataset published, hollow = nothing provisioned yet. */}
                {n.provisioned !== undefined && n.provisioned !== null && (
                  <circle cx={10} cy={BOX_H / 2} r={4} fill={n.provisioned ? '#16a34a' : 'none'} stroke={n.provisioned ? '#16a34a' : '#9ca3af'} strokeWidth={2} />
                )}
                <text
                  x={n.provisioned !== undefined && n.provisioned !== null ? 20 : 10}
                  y={BOX_H / 2 + 4}
                  fontSize={11}
                  fontFamily="ui-monospace, monospace"
                  fill="#111827"
                >
                  {n.provisioned !== undefined && n.provisioned !== null && n.name.length > 27 ? n.name.slice(0, 26) + '…' : label}
                </text>
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
      <div className="px-4 py-1.5 text-[11px] text-base-content/50 border-t border-base-300">
        {room.nodes.length} nodes in {columns} stages · {room.exits.length} exit{room.exits.length === 1 ? '' : 's'} where analysts see output
        {byName.size !== room.nodes.length ? ' · duplicate node names' : ''}
      </div>
    </div>
  );
}
