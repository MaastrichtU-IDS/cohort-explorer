'use client';

// "Deactivate DCR" control on a My DCRs card. Deactivation stops the DCR on
// Decentriq: no new computation can run in it, and it cannot be undone. Once
// done, the button is replaced by a notice.
import React, {useState} from 'react';
import {createPortal} from 'react-dom';
import {AlertOctagon, AlertTriangle, Power} from 'react-feather';
import {apiUrl} from '@/utils';

interface Props {
  dcrId: string;
  dcrTitle: string;
  deactivated: boolean;
  deactivatedAt?: string | null;
  deactivatedBy?: string | null;
}

export function DeactivateDcr({dcrId, dcrTitle, deactivated, deactivatedAt, deactivatedBy}: Props) {
  const [done, setDone] = useState<{at?: string | null; by?: string | null} | null>(
    deactivated ? {at: deactivatedAt, by: deactivatedBy} : null
  );
  const [confirming, setConfirming] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const deactivate = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/my-dcrs/${encodeURIComponent(dcrId)}/deactivate`, {method: 'POST', credentials: 'include'});
      if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`;
        try {
          detail = (await res.json())?.detail || detail;
        } catch {}
        throw new Error(detail);
      }
      const data = await res.json();
      setDone({at: data?.deactivated_at, by: data?.deactivated_by});
      setConfirming(false);
    } catch (e: any) {
      setError(e?.message || 'Deactivation failed');
    } finally {
      setSubmitting(false);
    }
  };

  if (done) {
    return (
      <div className="mt-3 alert alert-error font-semibold">
        <AlertOctagon size={20} />
        <span>
          This DCR has been deactivated. No new computations can be run in it.
          {(done.at || done.by) && (
            <span className="block text-xs font-normal opacity-90">
              Deactivated{done.at && <> on {new Date(done.at).toLocaleString()}</>}
              {done.by && <> by {done.by}</>}.
            </span>
          )}
        </span>
      </div>
    );
  }

  const modal = confirming && (
    <div className="modal modal-open z-[10000]" onMouseDown={() => !submitting && setConfirming(false)}>
      <div className="modal-box" onMouseDown={e => e.stopPropagation()}>
        <h3 className="font-bold text-lg flex items-center gap-2 text-error">
          <AlertTriangle size={20} /> Deactivate this DCR?
        </h3>
        <p className="mt-3 text-sm">
          <span className="font-semibold">{dcrTitle || dcrId}</span> will be stopped on Decentriq.
        </p>
        <ul className="list-disc ml-5 mt-2 text-sm space-y-1">
          <li>No new computations can be run in it, by anyone.</li>
          <li className="font-semibold text-error">This action is not reversible.</li>
        </ul>
        {error && <div className="alert alert-error text-sm mt-3">{error}</div>}
        <div className="modal-action">
          <button className="btn btn-sm btn-ghost" onClick={() => setConfirming(false)} disabled={submitting}>Cancel</button>
          <button className="btn btn-sm btn-error" onClick={deactivate} disabled={submitting}>
            {submitting && <span className="loading loading-spinner loading-xs" />} Deactivate DCR
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="mt-3">
      <button className="btn btn-error btn-sm gap-2 text-white" onClick={() => setConfirming(true)}>
        <Power size={14} /> Deactivate DCR
      </button>
      {modal && (typeof document === 'undefined' ? modal : createPortal(modal, document.body))}
    </div>
  );
}
