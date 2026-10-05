'use client';

// "Deactivate DCR" control on a My DCRs card, shown only to the DCR's creator
// (its owner on Decentriq). Deactivation stops the DCR: no new computation can
// run in it, and it cannot be undone. Decentriq lets only the owner stop it:
//   - owned by the explorer's service account: stopped right here (inApp);
//   - owned by a user: the dialog sends them to the DCR on the Decentriq
//     platform, then asks the backend whether Decentriq reports it stopped.
// Once it is, the button is replaced by a notice (shown to every participant).
import React, {useState} from 'react';
import {createPortal} from 'react-dom';
import {AlertOctagon, AlertTriangle, ExternalLink, Power} from 'react-feather';
import {apiUrl} from '@/utils';

interface Props {
  dcrId: string;
  dcrTitle: string;
  deactivated: boolean;
  canDeactivate: boolean;
  inApp: boolean;
}

// The backend also records who deactivated the DCR and when (deactivated_by /
// deactivated_at / deactivated_source on the DCR record); not shown for now.
export function DeactivateDcr({dcrId, dcrTitle, deactivated, canDeactivate, inApp}: Props) {
  const [done, setDone] = useState(deactivated);
  const [confirming, setConfirming] = useState(false);
  const [checking, setChecking] = useState(false);
  const [message, setMessage] = useState<{kind: 'error' | 'info'; text: string} | null>(null);
  const decentriqUrl = `https://platform.decentriq.com/datarooms/p/${dcrId}`;

  // inApp: stop it now; otherwise: ask whether the owner has stopped it on Decentriq.
  const callBackend = async () => {
    setChecking(true);
    setMessage(null);
    try {
      const action = inApp ? 'deactivate' : 'check-deactivated';
      const res = await fetch(`${apiUrl}/my-dcrs/${encodeURIComponent(dcrId)}/${action}`, {
        method: 'POST',
        credentials: 'include',
      });
      if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`;
        try {
          detail = (await res.json())?.detail || detail;
        } catch {}
        throw new Error(detail);
      }
      const data = await res.json();
      if (data?.deactivated) {
        setDone(true);
        setConfirming(false);
      } else {
        setMessage({kind: 'info', text: 'Decentriq still reports this DCR as active.'});
      }
    } catch (e: any) {
      setMessage({kind: 'error', text: e?.message || (inApp ? 'Deactivation failed' : 'Could not check the status')});
    } finally {
      setChecking(false);
    }
  };

  if (done) {
    return (
      <div className="mt-3 alert bg-base-200 border-base-300 text-base-content font-semibold">
        <AlertOctagon size={20} />
        <span>
          This DCR has been deactivated. No new computations can be run in it.
        </span>
      </div>
    );
  }

  if (!canDeactivate) return null;

  const modal = confirming && (
    <div className="modal modal-open z-[10000]" onMouseDown={() => !checking && setConfirming(false)}>
      <div className="modal-box" onMouseDown={e => e.stopPropagation()}>
        <h3 className="font-bold text-lg flex items-center gap-2">
          <AlertTriangle size={20} /> Deactivate this DCR?
        </h3>
        <p className="mt-3 text-sm">
          Deactivating <span className="font-semibold">{dcrTitle || dcrId}</span> stops it on Decentriq:
        </p>
        <ul className="list-disc ml-5 mt-2 text-sm space-y-1">
          <li>No new computations can be run in it, by anyone.</li>
          <li>This action is not reversible.</li>
        </ul>
        {!inApp && (
          <p className="mt-3 text-sm">
            Decentriq only lets the creator of a DCR stop it, so this is done by you on the Decentriq platform: open
            the DCR there and stop it. Then come back and check its status here.
          </p>
        )}
        {message && (
          <div className="alert bg-base-200 border-base-300 text-base-content text-sm mt-3">{message.text}</div>
        )}
        <div className="modal-action flex-wrap">
          <button className="btn btn-sm btn-ghost" onClick={() => setConfirming(false)} disabled={checking}>
            Cancel
          </button>
          {inApp ? (
            <button className="btn btn-sm btn-error text-white" onClick={callBackend} disabled={checking}>
              {checking && <span className="loading loading-spinner loading-xs" />} Deactivate DCR
            </button>
          ) : (
            <>
              <a className="btn btn-sm btn-error text-white gap-1" href={decentriqUrl} target="_blank" rel="noopener noreferrer">
                <ExternalLink size={14} /> Open DCR on Decentriq
              </a>
              <button className="btn btn-sm btn-outline" onClick={callBackend} disabled={checking}>
                {checking && <span className="loading loading-spinner loading-xs" />} I&apos;ve deactivated it - check status
              </button>
            </>
          )}
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
