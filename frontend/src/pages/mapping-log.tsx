'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { useCohorts } from '@/components/CohortsContext';
import { apiUrl } from '@/utils';

interface LogEntry {
  ts: string;
  level: 'MAIN' | 'DETAIL';
  process: string;
  event: string;
  msg: string;
  ctx: Record<string, any>;
  depth: number;
}

function formatTs(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleString(undefined, {
      month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch {
    return ts;
  }
}

function pairKey(source: string, targets: string[]): string {
  return `${source} → ${[...targets].sort().join(', ')}`;
}

function extractPairsFromEntries(entries: LogEntry[]): string[] {
  const seen = new Set<string>();
  for (const e of entries) {
    const source = e.ctx?.source as string | undefined;
    const targets = e.ctx?.targets as string[] | undefined;
    const target = e.ctx?.target as string | undefined;
    if (source) {
      const tList = targets && targets.length > 0
        ? targets
        : target ? [target] : [];
      if (tList.length > 0) seen.add(pairKey(source, tList));
    }
  }
  return Array.from(seen).sort();
}

function entryMatchesPair(entry: LogEntry, pair: string): boolean {
  const source = entry.ctx?.source as string | undefined;
  const targets = entry.ctx?.targets as string[] | undefined;
  const target = entry.ctx?.target as string | undefined;
  if (!source) return false;
  const tList = targets && targets.length > 0
    ? targets
    : target ? [target] : [];
  if (tList.length === 0) return false;
  return pairKey(source, tList) === pair;
}

function processLabel(process: string): string {
  if (process === 'cohort_var_linker') return 'CohortVarLinker';
  if (process === 'standard_code_mapping') return 'Code Mapping';
  return process;
}

function CtxCell({ ctx }: { ctx: Record<string, any> }) {
  const [expanded, setExpanded] = useState(false);
  const keys = Object.keys(ctx).filter(k => k !== 'source' && k !== 'targets' && k !== 'target');
  if (keys.length === 0) return <span className="text-base-content/30 text-xs">—</span>;
  const preview = keys.slice(0, 2).map(k => {
    const v = ctx[k];
    const vs = typeof v === 'object' ? JSON.stringify(v) : String(v);
    return `${k}: ${vs.length > 30 ? vs.slice(0, 30) + '…' : vs}`;
  }).join(' · ');
  return (
    <span>
      {expanded
        ? <pre className="text-xs whitespace-pre-wrap max-w-xs">{JSON.stringify(ctx, null, 2)}</pre>
        : <span className="text-xs text-base-content/60">{preview}{keys.length > 2 ? ' …' : ''}</span>
      }
      <button
        className="ml-1 link link-primary text-xs"
        onClick={() => setExpanded(e => !e)}
      >
        {expanded ? 'less' : 'more'}
      </button>
    </span>
  );
}

export default function MappingLogPage() {
  const { userEmail } = useCohorts();
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [levelFilter, setLevelFilter] = useState<'ALL' | 'MAIN' | 'DETAIL'>('ALL');
  const [pairFilter, setPairFilter] = useState<string | null>(null);

  useEffect(() => {
    if (userEmail === null) return;
    const fetchLog = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiUrl}/api/mapping-activity-log?limit=2000`, {
          credentials: 'include',
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setEntries((data.entries as LogEntry[]).reverse());
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    fetchLog();
  }, [userEmail]);

  const availablePairs = useMemo(() => extractPairsFromEntries(entries), [entries]);

  const filtered = useMemo(() => {
    return entries.filter(e => {
      if (levelFilter !== 'ALL' && e.level !== levelFilter) return false;
      if (pairFilter !== null) {
        // Keep entries that belong to this pair OR have no pair context (sub-events)
        if (e.ctx?.source) {
          if (!entryMatchesPair(e, pairFilter)) return false;
        }
      }
      return true;
    });
  }, [entries, levelFilter, pairFilter]);

  if (userEmail === null) {
    return <p className="text-red-500 text-center mt-[20%]">Authenticate to access the mapping log</p>;
  }

  return (
    <div className="container mx-auto px-4 py-6 max-w-screen-xl">
      <h1 className="text-2xl font-bold mb-1">Mapping Activity Log</h1>
      <p className="text-base-content/60 text-sm mb-6">
        {loading ? 'Loading…' : `${filtered.length} of ${entries.length} entries`}
      </p>

      {/* ── Filter section ─────────────────────────────────────────── */}
      <div className="space-y-4 mb-8">

        {/* Level of detail */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-base-content/50 mb-2">Level of detail</p>
          <div className="flex flex-wrap gap-3">
            {(['ALL', 'MAIN', 'DETAIL'] as const).map(lvl => (
              <button
                key={lvl}
                className={`btn btn-lg rounded-xl font-semibold ${
                  levelFilter === lvl
                    ? lvl === 'MAIN'
                      ? 'btn-primary'
                      : lvl === 'DETAIL'
                        ? 'btn-secondary'
                        : 'btn-neutral'
                    : 'btn-outline'
                }`}
                onClick={() => setLevelFilter(lvl)}
              >
                {lvl === 'ALL' ? 'All levels' : lvl === 'MAIN' ? 'Main only' : 'Detail only'}
              </button>
            ))}
          </div>
        </div>

        {/* Mapping pair */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-base-content/50 mb-2">Mapping pair</p>
          <div className="flex flex-wrap gap-3">
            <button
              className={`btn btn-lg rounded-xl font-semibold ${pairFilter === null ? 'btn-neutral' : 'btn-outline'}`}
              onClick={() => setPairFilter(null)}
            >
              All pairs
            </button>
            {availablePairs.map(pair => (
              <button
                key={pair}
                className={`btn btn-lg rounded-xl font-mono font-semibold ${pairFilter === pair ? 'btn-accent' : 'btn-outline'}`}
                onClick={() => setPairFilter(p => p === pair ? null : pair)}
              >
                {pair}
              </button>
            ))}
            {availablePairs.length === 0 && !loading && (
              <span className="text-base-content/40 text-sm self-center">No pairs found in log</span>
            )}
          </div>
        </div>
      </div>

      {/* ── Log table ──────────────────────────────────────────────── */}
      {error && (
        <div className="alert alert-error mb-4">{error}</div>
      )}

      {loading ? (
        <div className="flex items-center gap-3 py-16 justify-center opacity-60">
          <span className="loading loading-spinner loading-lg"></span>
          <span>Loading log…</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-16 text-base-content/40">No log entries match the current filters.</div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-base-300">
          <table className="table table-sm table-zebra w-full text-sm">
            <thead>
              <tr className="bg-base-200 text-xs uppercase tracking-wide">
                <th className="whitespace-nowrap">Time</th>
                <th>Level</th>
                <th>Process</th>
                <th>Event</th>
                <th className="min-w-[260px]">Message</th>
                <th>Context</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((entry, i) => (
                <tr
                  key={i}
                  style={{ paddingLeft: entry.depth > 0 ? `${entry.depth * 1}rem` : undefined }}
                  className={entry.level === 'MAIN' ? 'font-medium' : 'opacity-80'}
                >
                  <td className="whitespace-nowrap text-xs text-base-content/60 font-mono">
                    {formatTs(entry.ts)}
                  </td>
                  <td>
                    <span className={`badge badge-sm font-semibold ${
                      entry.level === 'MAIN' ? 'badge-primary' : 'badge-ghost'
                    }`}>
                      {entry.level}
                    </span>
                  </td>
                  <td className="whitespace-nowrap text-xs text-base-content/70">
                    {processLabel(entry.process)}
                  </td>
                  <td className="font-mono text-xs text-base-content/80">
                    {entry.event}
                  </td>
                  <td style={{ paddingLeft: entry.depth > 0 ? `${entry.depth * 1.25}rem` : undefined }}>
                    {entry.msg}
                  </td>
                  <td>
                    <CtxCell ctx={entry.ctx} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
