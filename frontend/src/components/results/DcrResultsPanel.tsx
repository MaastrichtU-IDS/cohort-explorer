'use client';

// "Compute nodes & results" section of a My DCRs card: run or re-run each
// compute node the user is an analyst of, then view, download or share each
// file of its latest output.
import React, {useCallback, useEffect, useState} from 'react';
import Link from 'next/link';
import {AlertTriangle, ChevronDown, ChevronRight, Download, Eye, Play, RefreshCw, Share2} from 'react-feather';
import {apiUrl} from '@/utils';
import {ResultFileModal, downloadFile, formatBytes, formatDateTime, isViewable} from './ResultFileView';
import {ShareResultDialog, ShareTarget} from './ShareResultDialog';

interface NodeResult {
  name: string;
  status: 'never_run' | 'running' | 'succeeded' | 'failed';
  started_at?: string | null;
  started_by?: string | null;
  finished_at?: string | null;
  error?: string | null;
  result_generated_at?: string | null;
  result_run_by?: string | null;
  files: {path: string; size: number}[];
}

// Decentriq offers no way to list a node's earlier runs, so status is shown
// only for runs started from the explorer; 'never_run' gets no badge.
const STATUS_BADGE: Partial<Record<NodeResult['status'], string>> = {
  running: 'badge-info',
  succeeded: 'badge-success',
  failed: 'badge-error',
};
const STATUS_LABEL: Partial<Record<NodeResult['status'], string>> = {
  running: 'running…',
  succeeded: 'last run succeeded',
  failed: 'last run failed',
};

const fileKey = (node: string, path: string) => `${node}\u0000${path}`;

async function errorDetail(res: Response): Promise<string> {
  try {
    return (await res.json())?.detail || `${res.status} ${res.statusText}`;
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

export function DcrResultsPanel({dcrId, dcrTitle}: {dcrId: string; dcrTitle: string}) {
  const [open, setOpen] = useState(false);
  const [nodes, setNodes] = useState<NodeResult[] | null>(null);
  const [cohorts, setCohorts] = useState<string[]>([]);
  const [sharerName, setSharerName] = useState('');
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runErrors, setRunErrors] = useState<Record<string, string>>({});
  const [viewing, setViewing] = useState<{node: string; path: string} | null>(null);
  const [sharing, setSharing] = useState<ShareTarget | null>(null);
  const [sharedMsg, setSharedMsg] = useState<string | null>(null);

  const base = `${apiUrl}/my-dcrs/${encodeURIComponent(dcrId)}`;
  const fileUrl = (node: string, path: string) =>
    `${base}/nodes/${encodeURIComponent(node)}/files/${path.split('/').map(encodeURIComponent).join('/')}`;

  const load = useCallback(
    async (quiet = false) => {
      if (!quiet) setLoading(true);
      try {
        const res = await fetch(`${base}/results`, {credentials: 'include'});
        if (!res.ok) throw new Error(await errorDetail(res));
        const data = await res.json();
        const loaded: NodeResult[] = Array.isArray(data?.nodes) ? data.nodes : [];
        setNodes(loaded);
        setCohorts(Array.isArray(data?.cohorts) ? data.cohorts : []);
        setSharerName(data?.sharer_name || '');
        // Drop selections whose file is gone after a re-run.
        const present = new Set(loaded.flatMap(n => n.files.map(f => fileKey(n.name, f.path))));
        setSelected(prev => prev.filter(k => present.has(k)));
        setError(null);
      } catch (e: any) {
        setError(e?.message || 'Failed to load compute nodes');
      } finally {
        if (!quiet) setLoading(false);
      }
    },
    [base]
  );

  useEffect(() => {
    if (open && nodes === null) load();
  }, [open, nodes, load]);

  // Poll while any node is running.
  const anyRunning = !!nodes?.some(n => n.status === 'running');
  useEffect(() => {
    if (!open || !anyRunning) return;
    const t = setInterval(() => load(true), 5000);
    return () => clearInterval(t);
  }, [open, anyRunning, load]);

  const toggleSelected = (key: string) =>
    setSelected(prev => (prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]));

  const share = (files: {nodeName: string; filePath: string}[]) =>
    setSharing({dcrId, dcrTitle, files, cohorts, sharerName});

  const shareSelected = () => {
    const files = (nodes || []).flatMap(n =>
      n.files.filter(f => selected.includes(fileKey(n.name, f.path))).map(f => ({nodeName: n.name, filePath: f.path}))
    );
    if (files.length) share(files);
  };

  const run = async (node: string) => {
    setRunErrors(prev => ({...prev, [node]: ''}));
    try {
      const res = await fetch(`${base}/nodes/${encodeURIComponent(node)}/run`, {method: 'POST', credentials: 'include'});
      if (!res.ok) throw new Error(await errorDetail(res));
      setNodes(prev => prev?.map(n => (n.name === node ? {...n, status: 'running', error: null} : n)) ?? prev);
    } catch (e: any) {
      setRunErrors(prev => ({...prev, [node]: e?.message || 'Could not start the run'}));
    }
  };

  return (
    <div className="mt-3 pt-3 border-t border-base-300">
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn btn-sm btn-outline gap-1" onClick={() => setOpen(o => !o)}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />} Compute nodes &amp; results
        </button>
        {open && (
          <button className="btn btn-sm btn-ghost gap-1" onClick={() => load()} disabled={loading} title="Reload run status">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        )}
        {open && selected.length > 0 && (
          <>
            <button className="btn btn-sm btn-primary gap-1" onClick={shareSelected}>
              <Share2 size={14} /> Share selected ({selected.length})
            </button>
            <button className="btn btn-sm btn-ghost" onClick={() => setSelected([])}>Clear selection</button>
          </>
        )}
        {sharedMsg && (
          <span className="text-sm text-success">
            {sharedMsg}{' '}
            <Link href="/shared-results" className="link">Open Shared Results</Link>
          </span>
        )}
      </div>

      {open && (
        <div className="mt-3 space-y-3">
          {loading && nodes === null && <span className="loading loading-spinner loading-sm" />}
          {error && (
            <div className="alert alert-error text-sm">
              <AlertTriangle size={16} /> {error}
            </div>
          )}
          {nodes && nodes.length === 0 && (
            <div className="text-sm opacity-70">You are not an analyst of any runnable compute node in this DCR.</div>
          )}
          {nodes?.map(node => (
            <div key={node.name} className="rounded border border-base-300 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-sm font-semibold break-all">{node.name}</span>
                {STATUS_LABEL[node.status] && (
                  <span className={`badge badge-sm ${STATUS_BADGE[node.status]}`}>{STATUS_LABEL[node.status]}</span>
                )}
                <div className="flex-1" />
                <button className="btn btn-xs btn-primary gap-1" onClick={() => run(node.name)} disabled={node.status === 'running'}>
                  {node.status === 'running' ? (
                    <span className="loading loading-spinner loading-xs" />
                  ) : node.status === 'never_run' ? (
                    <Play size={12} />
                  ) : (
                    <RefreshCw size={12} />
                  )}
                  {node.status === 'never_run' ? 'Run' : 'Re-run'}
                </button>
              </div>
              <div className="text-xs opacity-70 mt-1 space-y-0.5">
                {node.status === 'running' && node.started_at && (
                  <div>
                    Started {formatDateTime(node.started_at)}
                    {node.started_by && <> by {node.started_by}</>}. Runs that pull data from several cohorts can take a few minutes.
                  </div>
                )}
                {node.result_generated_at && (
                  <div>
                    Results shown below are from {formatDateTime(node.result_generated_at)}
                    {node.result_run_by && <> (run by {node.result_run_by})</>}.
                  </div>
                )}
              </div>
              {node.status === 'failed' && node.error && (
                <div className="text-xs text-error mt-1 break-words">Last run failed: {node.error}</div>
              )}
              {runErrors[node.name] && <div className="text-xs text-error mt-1">{runErrors[node.name]}</div>}

              {node.files.length > 0 && (
                <div className="mt-2 divide-y divide-base-200">
                  {node.files.length > 1 && (
                    <label className="flex items-center gap-2 py-1 text-xs opacity-70 cursor-pointer w-fit">
                      <input
                        type="checkbox"
                        className="checkbox checkbox-xs"
                        checked={node.files.every(f => selected.includes(fileKey(node.name, f.path)))}
                        onChange={e => {
                          const keys = node.files.map(f => fileKey(node.name, f.path));
                          setSelected(prev =>
                            e.target.checked ? Array.from(new Set([...prev, ...keys])) : prev.filter(k => !keys.includes(k))
                          );
                        }}
                      />
                      Select all files of this node
                    </label>
                  )}
                  {node.files.map(f => {
                    const key = fileKey(node.name, f.path);
                    return (
                      <div key={f.path} className="flex flex-wrap sm:flex-nowrap items-center gap-x-3 gap-y-1 py-1">
                        <input type="checkbox" className="checkbox checkbox-xs shrink-0" checked={selected.includes(key)}
                          onChange={() => toggleSelected(key)} aria-label={`Select ${f.path}`} />
                        <span className="font-mono text-xs break-all flex-1 min-w-0">{f.path}</span>
                        <span className="text-xs opacity-70 whitespace-nowrap shrink-0 w-20 text-right">{formatBytes(f.size)}</span>
                        <div className="flex gap-1 shrink-0">
                          {isViewable(f.path) ? (
                            <button className="btn btn-xs btn-ghost gap-1" onClick={() => setViewing({node: node.name, path: f.path})}>
                              <Eye size={12} /> View
                            </button>
                          ) : (
                            <span className="btn btn-xs btn-ghost invisible gap-1" aria-hidden><Eye size={12} /> View</span>
                          )}
                          <button
                            className="btn btn-xs btn-ghost gap-1"
                            onClick={() =>
                              downloadFile(fileUrl(node.name, f.path), f.path.split('/').pop() || f.path).catch(e =>
                                setRunErrors(prev => ({...prev, [node.name]: e?.message || 'Download failed'}))
                              )
                            }
                          >
                            <Download size={12} /> Download
                          </button>
                          <button className="btn btn-xs btn-ghost gap-1" onClick={() => share([{nodeName: node.name, filePath: f.path}])}>
                            <Share2 size={12} /> Share
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              {node.status === 'succeeded' && node.files.length === 0 && (
                <div className="text-xs opacity-70 mt-1">The run produced no output files.</div>
              )}
            </div>
          ))}
        </div>
      )}

      {viewing && (
        <ResultFileModal
          url={fileUrl(viewing.node, viewing.path)}
          fileName={viewing.path.split('/').pop() || viewing.path}
          subtitle={`${viewing.node} · ${viewing.path}`}
          onClose={() => setViewing(null)}
        />
      )}
      {sharing && (
        <ShareResultDialog
          target={sharing}
          onClose={() => setSharing(null)}
          onShared={() => {
            const n = sharing.files.length;
            setSharedMsg(n === 1 ? `Shared “${sharing.files[0].filePath.split('/').pop()}”.` : `Shared ${n} files.`);
            setSelected(prev => prev.filter(k => !sharing.files.some(f => fileKey(f.nodeName, f.filePath) === k)));
            setSharing(null);
          }}
        />
      )}
    </div>
  );
}
