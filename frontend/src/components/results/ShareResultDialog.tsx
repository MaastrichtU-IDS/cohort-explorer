'use client';

// Dialog for sharing one or more DCR result files to Shared Results as
// one entry (title, description, optional details, cohorts).
import React, {useState} from 'react';
import {createPortal} from 'react-dom';
import {Plus, Share2} from 'react-feather';
import {apiUrl} from '@/utils';

export interface ShareTarget {
  dcrId: string;
  dcrTitle: string;
  files: {nodeName: string; filePath: string}[];
  cohorts: string[];
  sharerName: string;
}

export function ShareResultDialog({target, onClose, onShared}: {target: ShareTarget; onClose: () => void; onShared: () => void}) {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [details, setDetails] = useState('');
  const [showDetails, setShowDetails] = useState(false);
  const [cohorts, setCohorts] = useState<string[]>(target.cohorts);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = title.trim() && description.trim() && !submitting;

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/shared-results`, {
        method: 'POST',
        credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          dcr_id: target.dcrId,
          files: target.files.map(f => ({node_name: f.nodeName, file_path: f.filePath})),
          title,
          description,
          details: showDetails ? details : '',
          cohorts,
        }),
      });
      if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`;
        try {
          detail = (await res.json())?.detail || detail;
        } catch {}
        throw new Error(detail);
      }
      onShared();
    } catch (e: any) {
      setError(e?.message || 'Sharing failed');
      setSubmitting(false);
    }
  };

  const toggleCohort = (c: string) => setCohorts(prev => (prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]));

  const modal = (
    <div className="modal modal-open z-[10000]" onMouseDown={onClose}>
      <div className="modal-box max-w-2xl flex flex-col max-h-[90vh]" onMouseDown={e => e.stopPropagation()}>
        <h3 className="font-bold text-lg flex items-center gap-2 shrink-0">
          <Share2 size={18} /> Share to Shared Results
        </h3>
        <div className="text-sm opacity-70 mt-1 shrink-0">
          {target.files.length === 1 ? '1 file' : `${target.files.length} files`} from{' '}
          <span className="font-semibold">{target.dcrTitle || target.dcrId}</span>
          {target.sharerName && <>, shared as <span className="font-semibold">{target.sharerName}</span></>}
        </div>
        <ul className="text-xs font-mono mt-2 max-h-24 overflow-auto bg-base-200 rounded px-3 py-2 shrink-0">
          {target.files.map(f => (
            <li key={`${f.nodeName}/${f.filePath}`} className="break-all">{f.filePath}</li>
          ))}
        </ul>

        <div className="overflow-auto mt-4 space-y-3 pr-1">
          <label className="form-control">
            <span className="label-text font-semibold">Title *</span>
            <input className="input input-bordered input-sm" maxLength={200} value={title} onChange={e => setTitle(e.target.value)} autoFocus />
          </label>
          <label className="form-control">
            <span className="label-text font-semibold">Description *</span>
            <textarea className="textarea textarea-bordered textarea-sm h-20" maxLength={2000} value={description}
              onChange={e => setDescription(e.target.value)} placeholder="What these results show, in a few sentences." />
          </label>
          {showDetails ? (
            <label className="form-control">
              <span className="label-text font-semibold">Details</span>
              <textarea className="textarea textarea-bordered textarea-sm h-32" maxLength={10000} value={details}
                onChange={e => setDetails(e.target.value)} autoFocus
                placeholder="Research question, methods, variables and filters used, how to read the results, caveats." />
            </label>
          ) : (
            <button className="btn btn-xs btn-ghost gap-1" onClick={() => setShowDetails(true)}>
              <Plus size={12} /> Add details
            </button>
          )}
          {target.cohorts.length > 0 && (
            <div>
              <div className="label-text font-semibold mb-1">Cohorts</div>
              <div className="flex flex-wrap gap-x-4 gap-y-1">
                {target.cohorts.map(c => (
                  <label key={c} className="label cursor-pointer gap-2 py-0.5">
                    <input type="checkbox" className="checkbox checkbox-sm" checked={cohorts.includes(c)} onChange={() => toggleCohort(c)} />
                    <span className="label-text">{c}</span>
                  </label>
                ))}
              </div>
            </div>
          )}
          {error && <div className="alert alert-error text-sm">{error}</div>}
        </div>

        <div className="modal-action shrink-0">
          <button className="btn btn-sm btn-ghost" onClick={onClose} disabled={submitting}>Cancel</button>
          <button className="btn btn-sm btn-primary" onClick={submit} disabled={!canSubmit}>
            {submitting && <span className="loading loading-spinner loading-xs" />} Share
          </button>
        </div>
      </div>
    </div>
  );
  return typeof document === 'undefined' ? modal : createPortal(modal, document.body);
}
