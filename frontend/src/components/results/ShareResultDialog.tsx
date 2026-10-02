'use client';

// Dialog for sharing one DCR result file to the Results Gallery.
import React, {useState} from 'react';
import {createPortal} from 'react-dom';
import {Share2} from 'react-feather';
import {apiUrl} from '@/utils';

const NAME_KEY = 'resultsGallery.sharerName';

function readStoredName(): string {
  try {
    return localStorage.getItem(NAME_KEY) || '';
  } catch {
    return '';
  }
}

export interface ShareTarget {
  dcrId: string;
  dcrTitle: string;
  nodeName: string;
  filePath: string;
  cohorts: string[];
}

export function ShareResultDialog({target, onClose, onShared}: {target: ShareTarget; onClose: () => void; onShared: () => void}) {
  const fileName = target.filePath.split('/').pop() || target.filePath;
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [details, setDetails] = useState('');
  const [sharerName, setSharerName] = useState(readStoredName);
  const [cohorts, setCohorts] = useState<string[]>(target.cohorts);
  const [confirmed, setConfirmed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = title.trim() && description.trim() && sharerName.trim() && confirmed && !submitting;

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/results-gallery`, {
        method: 'POST',
        credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          dcr_id: target.dcrId,
          node_name: target.nodeName,
          file_path: target.filePath,
          title,
          description,
          details,
          sharer_name: sharerName,
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
      try {
        localStorage.setItem(NAME_KEY, sharerName.trim());
      } catch {}
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
          <Share2 size={18} /> Share to the Results Gallery
        </h3>
        <div className="text-sm opacity-70 mt-1 break-all shrink-0">
          <span className="font-mono">{fileName}</span> from <span className="font-mono">{target.nodeName}</span> in{' '}
          <span className="font-semibold">{target.dcrTitle || target.dcrId}</span>
        </div>

        <div className="overflow-auto mt-4 space-y-3 pr-1">
          <label className="form-control">
            <span className="label-text font-semibold">Title *</span>
            <input className="input input-bordered input-sm" maxLength={200} value={title} onChange={e => setTitle(e.target.value)}
              placeholder="e.g. Age distribution by sex across the pooled cohorts" />
          </label>
          <label className="form-control">
            <span className="label-text font-semibold">Description *</span>
            <textarea className="textarea textarea-bordered textarea-sm h-20" maxLength={2000} value={description}
              onChange={e => setDescription(e.target.value)} placeholder="What this result shows, in a few sentences." />
          </label>
          <label className="form-control">
            <span className="label-text font-semibold">Details</span>
            <span className="label-text-alt opacity-70 mb-1">
              Research question, methods, variables and filters used, how to read the result, caveats and limitations.
            </span>
            <textarea className="textarea textarea-bordered textarea-sm h-32" maxLength={10000} value={details}
              onChange={e => setDetails(e.target.value)} />
          </label>
          <label className="form-control">
            <span className="label-text font-semibold">Your name (shown with the result) *</span>
            <input className="input input-bordered input-sm" maxLength={120} value={sharerName} onChange={e => setSharerName(e.target.value)} />
          </label>
          {target.cohorts.length > 0 && (
            <div>
              <div className="label-text font-semibold mb-1">Cohorts this result concerns</div>
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
          <label className="label cursor-pointer justify-start gap-2 items-start bg-base-200 rounded p-2">
            <input type="checkbox" className="checkbox checkbox-sm mt-0.5" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />
            <span className="label-text text-sm">
              I confirm this file contains only aggregate results that may be shown to every logged-in Cohort Explorer user, and
              that sharing it is in line with the data owners&apos; terms for this DCR.
            </span>
          </label>
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
