'use client';

// Shared Results: DCR result files that participants shared with every
// logged-in user, grouped by the DCR they came from and filterable by cohort.
import React, {useCallback, useEffect, useMemo, useState} from 'react';
import Link from 'next/link';
import {AlertTriangle, Box, Download, Eye, Search, Trash2, User} from 'react-feather';
import {apiUrl} from '@/utils';
import {ResultFileModal, downloadFile, formatBytes, formatDateTime, isViewable} from '@/components/results/ResultFileView';

interface GalleryFile {
  node_name: string;
  file_path: string;
  file_name: string;
  size: number;
  result_generated_at?: string | null;
}

interface GalleryItem {
  id: string;
  title: string;
  description: string;
  details: string;
  shared_by: string;
  shared_at: string;
  dcr_id: string;
  dcr_title: string;
  dcr_created_at?: string | null;
  cohorts: string[];
  files: GalleryFile[];
  can_delete?: boolean;
}

interface DcrGroup {
  dcr_id: string;
  dcr_title: string;
  dcr_created_at?: string | null;
  cohorts: string[];
  items: GalleryItem[];
}

const fileUrl = (id: string, index: number) => `${apiUrl}/shared-results/${encodeURIComponent(id)}/files/${index}`;

export default function SharedResultsPage() {
  const [items, setItems] = useState<GalleryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCohorts, setSelectedCohorts] = useState<string[]>([]);
  const [query, setQuery] = useState('');
  const [viewing, setViewing] = useState<{item: GalleryItem; index: number} | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/shared-results`, {credentials: 'include'});
      if (res.status === 401 || res.status === 403) throw new Error('You must be signed in to view Shared Results.');
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

  const groups = useMemo<DcrGroup[]>(() => {
    const q = query.trim().toLowerCase();
    const filtered = items.filter(i => {
      if (selectedCohorts.length > 0 && !(i.cohorts || []).some(c => selectedCohorts.includes(c))) return false;
      if (!q) return true;
      return [i.title, i.description, i.details, i.shared_by, i.dcr_title, ...(i.files || []).map(f => f.file_path)]
        .some(s => (s || '').toLowerCase().includes(q));
    });
    const byDcr = new Map<string, DcrGroup>();
    for (const i of filtered) {
      let g = byDcr.get(i.dcr_id);
      if (!g) {
        g = {dcr_id: i.dcr_id, dcr_title: i.dcr_title, dcr_created_at: i.dcr_created_at, cohorts: [], items: []};
        byDcr.set(i.dcr_id, g);
      }
      g.items.push(i);
      for (const c of i.cohorts || []) if (!g.cohorts.includes(c)) g.cohorts.push(c);
    }
    // Items arrive newest first, so each group's first item is its latest share.
    return Array.from(byDcr.values()).sort((a, b) => b.items[0].shared_at.localeCompare(a.items[0].shared_at));
  }, [items, selectedCohorts, query]);

  const toggleCohort = (c: string) =>
    setSelectedCohorts(prev => (prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]));

  const remove = async (item: GalleryItem) => {
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
          <p className="text-lg text-base-content/70 mt-1">
            Results from Data Clean Rooms, shared by their participants. Share your own from{' '}
            <Link href="/dcrs" className="link">My DCRs</Link>.
          </p>
        </header>

        {!loading && !error && items.length > 0 && (
          <div className="card bg-base-100 shadow-sm">
            <div className="card-body p-4 space-y-3">
              <label className="input input-bordered input-sm flex items-center gap-2 max-w-md">
                <Search size={14} className="opacity-60" />
                <input className="grow" placeholder="Search titles, descriptions, sharers…" value={query} onChange={e => setQuery(e.target.value)} />
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
        {error && (
          <div className="alert alert-error">
            <AlertTriangle size={20} /> <span>{error}</span>
          </div>
        )}
        {!loading && !error && items.length === 0 && (
          <div className="text-center text-base-content/60 py-16">No results have been shared yet.</div>
        )}
        {!loading && !error && items.length > 0 && groups.length === 0 && (
          <div className="text-center text-base-content/60 py-16">No shared results match these filters.</div>
        )}

        {groups.map(g => (
          <section key={g.dcr_id} className="card bg-base-100 shadow-sm border border-base-300">
            <div className="card-body p-4">
              <div className="flex flex-wrap items-center gap-2">
                <Box size={18} className="opacity-70" />
                <h2 className="font-semibold text-xl">{g.dcr_title || 'Untitled DCR'}</h2>
                <span className="badge badge-ghost">{g.items.length} result{g.items.length === 1 ? '' : 's'}</span>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-sm text-base-content/70">
                {g.dcr_created_at && <span>DCR created {formatDateTime(g.dcr_created_at)}</span>}
                {g.cohorts.map(c => (
                  <span key={c} className="badge badge-primary">{c}</span>
                ))}
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mt-3">
                {g.items.map(item => (
                  <GalleryCard key={item.id} item={item} onView={index => setViewing({item, index})} onRemove={() => remove(item)} onError={setError} />
                ))}
              </div>
            </div>
          </section>
        ))}
      </div>

      {viewing && (
        <ResultFileModal
          url={fileUrl(viewing.item.id, viewing.index)}
          fileName={viewing.item.files[viewing.index].file_name}
          subtitle={`${viewing.item.title} · shared by ${viewing.item.shared_by}`}
          onClose={() => setViewing(null)}
        />
      )}
    </main>
  );
}

function GalleryCard({item, onView, onRemove, onError}: {item: GalleryItem; onView: (index: number) => void; onRemove: () => void; onError: (m: string) => void}) {
  const [showDetails, setShowDetails] = useState(false);
  return (
    <div className="rounded-lg border border-base-300 p-3 flex flex-col gap-2">
      <div className="flex items-start gap-2">
        <h3 className="font-semibold flex-1">{item.title}</h3>
        {item.can_delete && (
          <button className="btn btn-xs btn-ghost text-error" onClick={onRemove} title="Remove from Shared Results">
            <Trash2 size={12} />
          </button>
        )}
      </div>
      <div className="text-xs text-base-content/70 flex flex-wrap items-center gap-x-2">
        <span className="flex items-center gap-1">
          <User size={12} /> {item.shared_by}
        </span>
        <span>· shared {formatDateTime(item.shared_at)}</span>
        {item.cohorts.length > 0 && <span>· {item.cohorts.join(', ')}</span>}
      </div>
      <p className="text-sm whitespace-pre-wrap">{item.description}</p>
      {item.details && (
        <div>
          <button className="link text-xs" onClick={() => setShowDetails(s => !s)}>
            {showDetails ? 'Hide details' : 'Show details'}
          </button>
          {showDetails && <p className="text-sm whitespace-pre-wrap mt-1 bg-base-200 rounded p-2">{item.details}</p>}
        </div>
      )}
      <div className="mt-auto pt-2 border-t border-base-200 divide-y divide-base-200">
        {item.files.map((f, index) => (
          <div key={index} className="flex flex-wrap sm:flex-nowrap items-center gap-x-2 gap-y-1 py-1">
            <span className="font-mono text-xs break-all flex-1 min-w-0" title={`node ${f.node_name}: ${f.file_path}`}>
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
  );
}
