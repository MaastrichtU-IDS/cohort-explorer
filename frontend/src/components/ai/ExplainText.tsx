'use client';

// Explanation text with inline graph snippets. The model marks a local piece
// of the room's flow with a fenced block:
//
//   ```dcrgraph
//   TIME-CHF -> create-airlock-without-outliers-TIME-CHF -> preview-airlock-TIME-CHF
//   TIME-CHF_metadata_dictionary -> create-airlock-without-outliers-TIME-CHF
//   ```
//
// Each line is a chain of exact node names joined by "->"; a comma-separated
// list on either side fans in or out. The block is drawn as a small
// left-to-right graph, the rest of the text is rendered as usual.
import React, {useMemo} from 'react';
import {RichText} from '@/components/ai/ui';
import {RoomNode, RoomNodeKind, RoomOverview} from '@/components/ai/chatClient';
import {KIND_STYLE} from '@/components/DcrGraph';

const FENCE_RE = /```dcrgraph\s*\n([\s\S]*?)```/g;

interface Parsed {
  nodes: string[];
  edges: {from: string; to: string}[];
}

export function parseDcrGraph(code: string): Parsed {
  const nodes: string[] = [];
  const edges: {from: string; to: string}[] = [];
  const add = (n: string) => {
    if (n && !nodes.includes(n)) nodes.push(n);
  };
  code.split('\n').forEach(raw => {
    const line = raw.trim().replace(/^[-*]\s+/, '');
    if (!line || line.startsWith('#')) return;
    const groups = line.split(/\s*(?:->|→|=>)\s*/).map(g => g.split(/\s*,\s*/).map(s => s.replace(/^[`"']|[`"']$/g, '').trim()).filter(Boolean));
    groups.forEach(g => g.forEach(add));
    for (let i = 0; i + 1 < groups.length; i++) {
      groups[i].forEach(a => groups[i + 1].forEach(b => {
        if (a !== b && !edges.some(e => e.from === a && e.to === b)) edges.push({from: a, to: b});
      }));
    }
  });
  return {nodes, edges};
}

function layer(parsed: Parsed): Map<string, number> {
  const memo = new Map<string, number>();
  const visiting = new Set<string>();
  const preds = (n: string) => parsed.edges.filter(e => e.to === n).map(e => e.from);
  const depth = (n: string): number => {
    if (memo.has(n)) return memo.get(n)!;
    if (visiting.has(n)) return 0;
    visiting.add(n);
    const p = preds(n);
    const d = p.length ? 1 + Math.max(...p.map(depth)) : 0;
    visiting.delete(n);
    memo.set(n, d);
    return d;
  };
  parsed.nodes.forEach(depth);
  return memo;
}

const BW = 150;
const BH = 24;
const SG = 190;
const NG = 34;
const PAD = 8;

/** A small left-to-right drawing of a few nodes; kinds come from the room when known. */
export function InlineDcrGraph({code, room, onOpenNode}: {code: string; room: RoomOverview | null; onOpenNode?: (name: string) => void}) {
  const parsed = useMemo(() => parseDcrGraph(code), [code]);
  const byName = useMemo(() => new Map((room?.nodes || []).map(n => [n.name, n])), [room]);
  const resolve = (name: string): RoomNode | undefined => {
    if (byName.has(name)) return byName.get(name);
    // tolerate case and space/hyphen differences
    const k = name.toLowerCase().replace(/[\s_-]+/g, '-');
    return (room?.nodes || []).find(n => n.name.toLowerCase().replace(/[\s_-]+/g, '-') === k);
  };
  const {pos, width, height} = useMemo(() => {
    const depths = layer(parsed);
    const cols = new Map<number, string[]>();
    parsed.nodes.forEach(n => {
      const d = depths.get(n) || 0;
      if (!cols.has(d)) cols.set(d, []);
      cols.get(d)!.push(n);
    });
    const pos = new Map<string, {x: number; y: number}>();
    let maxRows = 1;
    Array.from(cols.keys()).sort((a, b) => a - b).forEach(c => {
      const list = cols.get(c)!;
      maxRows = Math.max(maxRows, list.length);
      list.forEach((n, i) => pos.set(n, {x: PAD + c * SG, y: PAD + i * NG}));
    });
    return {pos, width: PAD * 2 + (cols.size - 1) * SG + BW, height: PAD * 2 + (maxRows - 1) * NG + BH};
  }, [parsed]);

  if (parsed.nodes.length === 0) return null;
  return (
    <div className="my-2 rounded-md border border-base-300 bg-base-200/40 overflow-x-auto not-prose">
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} className="block max-w-full h-auto" role="img" aria-label="Part of the room's flow">
        <defs>
          <marker id="inl-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
          </marker>
        </defs>
        {parsed.edges.map((e, i) => {
          const a = pos.get(e.from), b = pos.get(e.to);
          if (!a || !b) return null;
          const x1 = a.x + BW, y1 = a.y + BH / 2, x2 = b.x, y2 = b.y + BH / 2;
          const dx = Math.max(20, (x2 - x1) / 2);
          return <path key={i} d={`M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`} fill="none" stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#inl-arrow)" />;
        })}
        {parsed.nodes.map(name => {
          const p = pos.get(name)!;
          const n = resolve(name);
          const kind: RoomNodeKind = n?.kind || 'other';
          const st = KIND_STYLE[kind];
          const label = name.length > 22 ? name.slice(0, 21) + '…' : name;
          const clickable = !!onOpenNode && n?.kind === 'python';
          return (
            <g key={name} transform={`translate(${p.x},${p.y})`} style={{cursor: clickable ? 'pointer' : 'default'}} onClick={() => clickable && onOpenNode!(n!.name)}>
              <title>{`${name}\n${n ? st.label : 'not a node of this room (as named by the model)'}${clickable ? '\n(click to open this node)' : ''}`}</title>
              <rect width={BW} height={BH} rx={5} fill={st.fill} stroke={n ? st.stroke : '#9ca3af'} strokeWidth={1.2} strokeDasharray={n ? undefined : '3 2'} />
              <text x={8} y={BH / 2 + 4} fontSize={10} fontFamily="ui-monospace, monospace" fill="#111827">{label}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Markdown-ish text with ```dcrgraph blocks drawn inline. */
export function ExplainText({text, room, onOpenNode}: {text: string; room: RoomOverview | null; onOpenNode?: (name: string) => void}) {
  const parts = useMemo(() => {
    const out: {kind: 'text' | 'graph'; body: string}[] = [];
    let last = 0;
    for (const m of text.matchAll(FENCE_RE)) {
      if (m.index! > last) out.push({kind: 'text', body: text.slice(last, m.index)});
      out.push({kind: 'graph', body: m[1]});
      last = m.index! + m[0].length;
    }
    if (last < text.length) out.push({kind: 'text', body: text.slice(last)});
    return out;
  }, [text]);
  return (
    <>
      {parts.map((p, i) => (p.kind === 'graph' ? <InlineDcrGraph key={i} code={p.body} room={room} onOpenNode={onOpenNode} /> : <RichText key={i} text={p.body} />))}
    </>
  );
}
