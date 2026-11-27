'use client';

// Shared Results: results any logged-in user shared with every logged-in user,
// newest first, filterable by cohort and searchable. A share is a title, a
// description, optional details, the cohorts it concerns and its files: either
// uploaded here ("Share results"), or output files of a DCR, shared from My DCRs.
import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {AlertTriangle, Box, Download, Eye, Plus, Search, Trash2, Upload, User, X} from 'react-feather';
import {apiUrl} from '@/utils';
import {useCohorts} from '@/components/CohortsContext';
import LoginPrompt from '@/components/LoginPrompt';
import {ResultFileModal, downloadFile, fileExt, formatBytes, formatDateTime, isViewable} from '@/components/results/ResultFileView';

interface SharedFile {
  node_name: string;
  file_path: string;
  file_name: string;
  size: number;
  result_generated_at?: string | null;
}

interface SharedItem {
  id: string;
  source?: string;
  title: string;
  description: string;
  details: string;
  shared_by: string;
  shared_at: string;
  dcr_id: string;
  dcr_title: string;
  dcr_created_at?: string | null;
  cohorts: string[];
  files: SharedFile[];
  can_delete?: boolean;
}

const IMAGE_EXT = ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'bmp'];
const isImage = (name: string) => IMAGE_EXT.includes(fileExt(name));

// The limits the backend enforces, checked here first to fail early.
const MAX_FILES = 100;
const MAX_FILE_MB = 200;
const MAX_TOTAL_MB = 1024;

const fileUrl = (id: string, index: number) => `${apiUrl}/shared-results/${encodeURIComponent(id)}/files/${index}`;

export default function SharedResultsPage() {
  const [items, setItems] = useState<SharedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [needsLogin, setNeedsLogin] = useState(false);
  const [selectedCohorts, setSelectedCohorts] = useState<string[]>([]);
  const [query, setQuery] = useState('');
  const [viewing, setViewing] = useState<{item: SharedItem; index: number} | null>(null);
  const [sharing, setSharing] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/shared-results`, {credentials: 'include'});
      if (res.status === 401 || res.status === 403) {
        setNeedsLogin(true);
        return;
      }
      setNeedsLogin(false);
      if (!res.ok) throw new Error(`Failed to load Shared Results: ${res.status} ${res.statusText}`);
      const data = await res.json();
      setItems(Array.isArray(data?.items) ? data.items : []);
    } catch (e: any) {
      setError(e?.message || 'Failed to load Shared Results');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const allCohorts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const i of items) for (const c of i.cohorts || []) counts.set(c, (counts.get(c) || 0) + 1);
    return Array.from(counts.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [items]);

  // Newest first (the backend's order), narrowed by cohort and search.
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return items.filter(i => {
      if (selectedCohorts.length > 0 && !(i.cohorts || []).some(c => selectedCohorts.includes(c))) return false;
      if (!q) return true;
      return [i.title, i.description, i.details, i.shared_by, i.dcr_title, ...(i.cohorts || []), ...(i.files || []).map(f => f.file_name)]
        .some(s => (s || '').toLowerCase().includes(q));
    });
  }, [items, selectedCohorts, query]);

  const toggleCohort = (c: string) =>
    setSelectedCohorts(prev => (prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]));

  const remove = async (item: SharedItem) => {
    if (!window.confirm(`Remove “${item.title}” from Shared Results?`)) return;
    const res = await fetch(`${apiUrl}/shared-results/${encodeURIComponent(item.id)}`, {method: 'DELETE', credentials: 'include'});
    if (!res.ok) {
      setError(`Could not remove the result: ${res.status} ${res.statusText}`);
      return;
    }
    setItems(prev => prev.filter(i => i.id !== item.id));
  };

  return (
    <main className="flex flex-col items-center justify-start p-6 min-h-screen bg-base-200">
      <div className="w-full max-w-6xl space-y-6">
        <header className="text-center">
          <h1 className="text-3xl font-bold">Shared Results</h1>
          <p className="text-lg text-base-content/70 mt-1">Results shared with every user of the explorer.</p>
          {!needsLogin && (
            <button className="btn btn-primary mt-4 gap-2" onClick={() => setSharing(true)}>
              <Plus size={16} /> Share results
            </button>
          )}
        </header>

        {!loading && !error && items.length > 0 && (
          <div className="card bg-base-100 shadow-sm">
            <div className="card-body p-4 space-y-3">
              <label className="input input-bordered input-sm flex items-center gap-2 max-w-md">
                <Search size={14} className="opacity-60" />
                <input className="grow" placeholder="Search titles, descriptions, cohorts, files, sharers…" value={query} onChange={e => setQuery(e.target.value)} />
              </label>
              {allCohorts.length > 0 && (
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-semibold mr-1">Cohorts:</span>
                  <button
                    className={`badge badge-lg cursor-pointer ${selectedCohorts.length === 0 ? 'badge-primary' : 'badge-outline'}`}
                    onClick={() => setSelectedCohorts([])}
                  >
                    All
                  </button>
                  {allCohorts.map(([c, n]) => (
                    <button
                      key={c}
                      className={`badge badge-lg cursor-pointer ${selectedCohorts.includes(c) ? 'badge-primary' : 'badge-outline'}`}
                      onClick={() => toggleCohort(c)}
                    >
                      {c} <span className="opacity-60 ml-1">{n}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {loading && (
          <div className="flex justify-center py-16">
            <span className="loading loading-spinner loading-lg" />
          </div>
        )}
        {needsLogin && <LoginPrompt />}
        {error && (
          <div className="alert alert-error">
            <AlertTriangle size={20} /> <span>{error}</span>
          </div>
        )}
        {!loading && !error && !needsLogin && items.length === 0 && (
          <div className="text-center text-base-content/60 py-16">No results have been shared yet.</div>
        )}
        {!loading && !error && items.length > 0 && visible.length === 0 && (
          <div className="text-center text-base-content/60 py-16">No shared results match these filters.</div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {visible.map(item => (
            <SharedCard key={item.id} item={item} onView={index => setViewing({item, index})} onRemove={() => remove(item)} onError={setError} />
          ))}
        </div>
      </div>

      {viewing && (
        <ResultFileModal
          url={fileUrl(viewing.item.id, viewing.index)}
          fileName={viewing.item.files[viewing.index].file_name}
          subtitle={`${viewing.item.title} · shared by ${viewing.item.shared_by}`}
          onClose={() => setViewing(null)}
        />
      )}

      {sharing && (
        <ShareDialog
          onClose={() => setSharing(false)}
          onShared={item => {
            setItems(prev => [{...item, can_delete: true}, ...prev]);
            setSharing(false);
          }}
        />
      )}
    </main>
  );
}

function SharedCard({item, onView, onRemove, onError}: {item: SharedItem; onView: (index: number) => void; onRemove: () => void; onError: (m: string) => void}) {
  const [showDetails, setShowDetails] = useState(false);
  const images = item.files.map((f, index) => ({f, index})).filter(({f}) => isImage(f.file_name));
  return (
    <div className="card bg-base-100 shadow-sm border border-base-300">
      <div className="card-body p-4 gap-2">
        <div className="flex items-start gap-2">
          <h3 className="font-semibold text-lg flex-1">{item.title}</h3>
          {item.can_delete && (
            <button className="btn btn-xs btn-ghost text-error" onClick={onRemove} title="Remove from Shared Results">
              <Trash2 size={12} />
            </button>
          )}
        </div>
        <div className="text-xs text-base-content/70 flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="flex items-center gap-1">
            <User size={12} /> {item.shared_by}
          </span>
          <span>· shared {formatDateTime(item.shared_at)}</span>
          {item.dcr_title && (
            <span className="flex items-center gap-1" title={item.dcr_id}>
              · <Box size={12} /> from DCR {item.dcr_title}
            </span>
          )}
        </div>
        {item.cohorts.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {item.cohorts.map(c => <span key={c} className="badge badge-primary badge-sm">{c}</span>)}
          </div>
        )}
        <p className="text-sm whitespace-pre-wrap">{item.description}</p>
        {item.details && (
          <div>
            <button className="link text-xs" onClick={() => setShowDetails(s => !s)}>
              {showDetails ? 'Hide details' : 'Show details'}
            </button>
            {showDetails && <p className="text-sm whitespace-pre-wrap mt-1 bg-base-200 rounded p-2">{item.details}</p>}
          </div>
        )}
        {images.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {images.slice(0, 6).map(({f, index}) => (
              <button key={index} onClick={() => onView(index)} title={f.file_name}
                      className="w-24 h-24 rounded border border-base-300 overflow-hidden bg-base-200 hover:ring-2 hover:ring-primary">
                <img src={fileUrl(item.id, index)} alt={f.file_name} crossOrigin="use-credentials"
                     className="w-full h-full object-cover" loading="lazy" />
              </button>
            ))}
            {images.length > 6 && <span className="text-xs text-base-content/60 self-end">+{images.length - 6} more images</span>}
          </div>
        )}
        <div className="mt-auto pt-2 border-t border-base-200 divide-y divide-base-200">
          {item.files.map((f, index) => (
            <div key={index} className="flex flex-wrap sm:flex-nowrap items-center gap-x-2 gap-y-1 py-1">
              <span className="font-mono text-xs break-all flex-1 min-w-0" title={f.node_name ? `node ${f.node_name}: ${f.file_path}` : f.file_name}>
                {f.file_name} <span className="opacity-60 whitespace-nowrap">({formatBytes(f.size)})</span>
              </span>
              <div className="flex gap-1 shrink-0">
                {isViewable(f.file_name) && (
                  <button className="btn btn-xs btn-outline gap-1" onClick={() => onView(index)}>
                    <Eye size={12} /> View
                  </button>
                )}
                <button
                  className="btn btn-xs btn-outline gap-1"
                  onClick={() => downloadFile(fileUrl(item.id, index), f.file_name).catch(e => onError(e?.message || 'Download failed'))}
                >
                  <Download size={12} /> Download
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// The share dialogue: title, description, optional details, the cohorts the
// results concern, and the files (any type; drop them or browse).
function ShareDialog({onClose, onShared}: {onClose: () => void; onShared: (item: SharedItem) => void}) {
  const {cohortsData} = useCohorts();
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [details, setDetails] = useState('');
  const [cohorts, setCohorts] = useState<string[]>([]);
  const [cohortQuery, setCohortQuery] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const catalogue = useMemo(() => Object.keys(cohortsData || {}).sort((a, b) => a.localeCompare(b)), [cohortsData]);
  const matching = useMemo(() => {
    const q = cohortQuery.trim().toLowerCase();
    return q ? catalogue.filter(c => c.toLowerCase().includes(q)) : catalogue;
  }, [catalogue, cohortQuery]);

  const totalBytes = files.reduce((n, f) => n + f.size, 0);
  const tooBig = files.find(f => f.size > MAX_FILE_MB * 1024 * 1024);
  const problem =
    !title.trim() ? 'Give the results a title.'
      : !description.trim() ? 'Describe the results.'
        : !files.length ? 'Add at least one file.'
          : files.length > MAX_FILES ? `At most ${MAX_FILES} files at once.`
            : tooBig ? `${tooBig.name} is larger than ${MAX_FILE_MB} MB.`
              : totalBytes > MAX_TOTAL_MB * 1024 * 1024 ? `The files together are larger than ${MAX_TOTAL_MB} MB.`
                : null;

  const addFiles = (list: FileList | null) => {
    if (!list) return;
    setFiles(prev => {
      const seen = new Set(prev.map(f => `${f.name}/${f.size}/${f.lastModified}`));
      return [...prev, ...Array.from(list).filter(f => !seen.has(`${f.name}/${f.size}/${f.lastModified}`))];
    });
  };

  const toggleCohort = (c: string) =>
    setCohorts(prev => (prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && progress === null) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, progress]);

  // XMLHttpRequest rather than fetch, for the upload's progress.
  const submit = () => {
    if (problem) return;
    setError(null);
    const form = new FormData();
    form.append('title', title.trim());
    form.append('description', description.trim());
    form.append('details', details.trim());
    form.append('cohorts', JSON.stringify(cohorts));
    for (const f of files) form.append('files', f, f.name);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${apiUrl}/shared-results/upload`);
    xhr.withCredentials = true;
    xhr.upload.onprogress = e => { if (e.lengthComputable) setProgress(e.loaded / e.total); };
    xhr.onload = () => {
      setProgress(null);
      let body: any = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) onShared(body);
      else setError(body?.detail || `Upload failed: HTTP ${xhr.status}`);
    };
    xhr.onerror = () => { setProgress(null); setError('Upload failed: the server could not be reached.'); };
    setProgress(0);
    xhr.send(form);
  };

  const busy = progress !== null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={() => { if (!busy) onClose(); }}>
      <div className="bg-base-100 rounded-lg shadow-xl w-full max-w-3xl max-h-[92vh] flex flex-col"
           onClick={e => e.stopPropagation()} role="dialog" aria-modal="true">
        <div className="flex items-center gap-3 px-5 py-3 border-b border-base-300">
          <h2 className="font-semibold text-lg flex-1">Share results</h2>
          <button className="btn btn-ghost btn-sm btn-square" onClick={onClose} disabled={busy} title="Close (Esc)">
            <X size={18} />
          </button>
        </div>

        <div className="overflow-auto px-5 py-4 space-y-4">
          <label className="form-control">
            <span className="label-text font-semibold mb-1">Title</span>
            <input className="input input-bordered" value={title} maxLength={200} onChange={e => setTitle(e.target.value)}
                   placeholder="e.g. Kaplan-Meier curves for heart failure hospitalisation" disabled={busy} />
          </label>
          <label className="form-control">
            <span className="label-text font-semibold mb-1">Description</span>
            <textarea className="textarea textarea-bordered min-h-24" value={description} maxLength={2000}
                      onChange={e => setDescription(e.target.value)} placeholder="What the results show and how they were produced" disabled={busy} />
          </label>
          <label className="form-control">
            <span className="label-text font-semibold mb-1">Details <span className="font-normal text-base-content/60">(optional)</span></span>
            <textarea className="textarea textarea-bordered min-h-16" value={details} maxLength={10000}
                      onChange={e => setDetails(e.target.value)} placeholder="Methods, parameters, caveats…" disabled={busy} />
          </label>

          <div>
            <div className="flex flex-wrap items-baseline gap-2 mb-1">
              <span className="label-text font-semibold">Cohorts these results concern</span>
              <span className="text-xs text-base-content/60">{cohorts.length} selected</span>
            </div>
            {cohorts.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-2">
                {cohorts.map(c => (
                  <button key={c} className="badge badge-primary gap-1" onClick={() => toggleCohort(c)} disabled={busy} title="Remove">
                    {c} <X size={10} />
                  </button>
                ))}
              </div>
            )}
            <label className="input input-bordered input-sm flex items-center gap-2 mb-2">
              <Search size={14} className="opacity-60" />
              <input className="grow" placeholder="Find a cohort" value={cohortQuery} onChange={e => setCohortQuery(e.target.value)} disabled={busy} />
            </label>
            <div className="max-h-40 overflow-auto rounded border border-base-300 p-2 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-x-3 gap-y-1">
              {matching.map(c => (
                <label key={c} className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="checkbox" className="checkbox checkbox-xs" checked={cohorts.includes(c)}
                         onChange={() => toggleCohort(c)} disabled={busy} />
                  <span className="truncate" title={c}>{c}</span>
                </label>
              ))}
              {!matching.length && <span className="text-xs text-base-content/50">No cohort matches.</span>}
            </div>
          </div>

          <div>
            <span className="label-text font-semibold">Files and images</span>
            <div
              className={`mt-1 rounded-lg border-2 border-dashed p-5 text-center transition-colors ${dragOver ? 'border-primary bg-primary/5' : 'border-base-300'}`}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={e => { e.preventDefault(); setDragOver(false); if (!busy) addFiles(e.dataTransfer.files); }}
            >
              <Upload size={22} className="mx-auto opacity-60" />
              <div className="text-sm mt-1">
                Drop files here, or{' '}
                <button className="link link-primary" onClick={() => inputRef.current?.click()} disabled={busy}>browse</button>
              </div>
              <div className="text-xs text-base-content/50 mt-1">
                Any type · up to {MAX_FILES} files, {MAX_FILE_MB} MB each, {MAX_TOTAL_MB} MB together
              </div>
              <input ref={inputRef} type="file" multiple className="hidden"
                     onChange={e => { addFiles(e.target.files); e.target.value = ''; }} />
            </div>
            {files.length > 0 && (
              <div className="mt-2 divide-y divide-base-200 rounded border border-base-300">
                {files.map((f, i) => (
                  <div key={`${f.name}/${i}`} className="flex items-center gap-2 px-2 py-1 text-xs">
                    <span className="font-mono break-all flex-1">{f.name}</span>
                    <span className={`whitespace-nowrap ${f.size > MAX_FILE_MB * 1024 * 1024 ? 'text-error' : 'opacity-60'}`}>{formatBytes(f.size)}</span>
                    <button className="btn btn-ghost btn-xs btn-square" onClick={() => setFiles(prev => prev.filter((_, j) => j !== i))}
                            disabled={busy} title="Remove">
                      <X size={12} />
                    </button>
                  </div>
                ))}
                <div className="px-2 py-1 text-xs text-base-content/60">{files.length} files · {formatBytes(totalBytes)}</div>
              </div>
            )}
          </div>

          {error && <div className="alert alert-error text-sm">{error}</div>}
        </div>

        <div className="flex flex-wrap items-center gap-3 px-5 py-3 border-t border-base-300">
          <span className="text-xs text-base-content/60 flex-1">
            {busy
              ? `Uploading… ${Math.round((progress || 0) * 100)}%`
              : problem || 'Everyone who can sign in to the explorer will be able to see and download these results.'}
          </span>
          {busy && <progress className="progress progress-primary w-32" value={(progress || 0) * 100} max={100} />}
          <button className="btn btn-ghost" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn-primary gap-2" onClick={submit} disabled={!!problem || busy}>
            <Upload size={16} /> Share
          </button>
        </div>
      </div>
    </div>
  );
}
