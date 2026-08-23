import React, {useEffect, useRef} from 'react';
import {createPortal} from 'react-dom';
import {X} from 'react-feather';

export type DataOwner = { email: string; cohorts: string[] };

// Data owners are INCLUDED by default: whenever an owner email appears for the
// first time it is added to the included list. Owners the user unticks stay
// unticked (they have been seen already). Shared by every participants list
// (flexible DCR wizard, no-code DCR wizard, provision/upload flow).
export function useOwnersIncludedByDefault(
  dataOwners: DataOwner[],
  included: string[],
  setIncluded: (emails: string[]) => void
) {
  const seen = useRef<Set<string>>(new Set());
  useEffect(() => {
    const fresh = dataOwners.map(o => o.email).filter(e => !seen.current.has(e));
    if (fresh.length === 0) return;
    fresh.forEach(e => seen.current.add(e));
    setIncluded(Array.from(new Set([...included, ...fresh])));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataOwners]);
}

type Props = {
  dataOwners: DataOwner[];
  userEmail: string | null;
  additionalAnalysts: string[];
  newAnalystEmail: string;
  setNewAnalystEmail: (email: string) => void;
  addAnalyst: () => void;
  removeAnalyst: (email: string) => void;
  manuallyIncludedOwners: string[];
  setManuallyIncludedOwners: (emails: string[]) => void;
  onClose: () => void;
  isLoading: boolean;
};

export const ParticipantsModal = React.memo(({
  dataOwners,
  userEmail,
  additionalAnalysts,
  newAnalystEmail,
  setNewAnalystEmail,
  addAnalyst,
  removeAnalyst,
  manuallyIncludedOwners,
  setManuallyIncludedOwners,
  onClose,
  isLoading
}: Props) => {
  const isExcluded = (email: string) => !manuallyIncludedOwners.includes(email);
  // Keep the invite field in view: on open, and again after every add/remove.
  const inviteRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    inviteRef.current?.scrollIntoView({block: 'end'});
  }, [isLoading, dataOwners.length, additionalAnalysts.length]);
  const toggleDataOwner = (email: string) => {
    if (manuallyIncludedOwners.includes(email)) {
      setManuallyIncludedOwners(manuallyIncludedOwners.filter(e => e !== email));
    } else {
      setManuallyIncludedOwners([...manuallyIncludedOwners, email]);
    }
  };
  // Rendered through a portal on <body>: when this list is opened from a wizard
  // that is itself a modal, the wizard's transformed box would otherwise become
  // the containing block of this fixed-position overlay and clip it.
  const modal = (
    <div className="modal modal-open z-[10000]">
      <div className="modal-box flex flex-col max-h-[90vh]">
        <div className="flex items-center justify-between mb-4 shrink-0">
          <h3 className="font-bold text-lg">DCR Participants</h3>
          <button type="button" className="btn btn-sm btn-circle btn-ghost" onClick={onClose} aria-label="Close">
            <X size={20} />
          </button>
        </div>

        <div className="space-y-4 overflow-y-auto flex-1">
          {/* Data owners */}
          <div>
            <h4 className="font-semibold mb-2">Data Owners</h4>
            {isLoading ? (
              <div className="bg-base-200 p-3 rounded-lg mb-2">
                <p className="text-sm text-gray-500">
                  Retrieving list of data owners for the selected cohorts...
                </p>
              </div>
            ) : dataOwners.length === 0 ? (
              <div className="bg-base-200 p-3 rounded-lg mb-2">
                <p className="text-sm text-base-content/60">
                  No cohorts have been selected.
                </p>
              </div>
            ) : (
              dataOwners.map((owner) => (
                <div key={owner.email} className={`p-3 rounded-lg mb-2 flex items-start gap-3 ${isExcluded(owner.email) ? 'bg-base-200 opacity-50' : 'bg-base-200'}`}>
                  <input
                    type="checkbox"
                    checked={!isExcluded(owner.email)}
                    onChange={() => toggleDataOwner(owner.email)}
                    className="checkbox checkbox-primary mt-1"
                  />
                  <div className="flex-1">
                    <p className={`font-semibold ${isExcluded(owner.email) ? 'line-through' : ''}`}>
                      {owner.email}
                      {owner.email === userEmail && <span className="ml-2 text-xs badge badge-primary">You</span>}
                    </p>
                    <p className="text-sm text-gray-500">
                      Data Owner for: {owner.cohorts.join(', ')}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Analysts */}
          <div>
            <h4 className="font-semibold mb-2">Analysts</h4>
            {/* Current user */}
            <div className="bg-base-200 p-3 rounded-lg mb-2">
              <div>
                <p className="font-semibold">{userEmail}</p>
                <p className="text-sm text-gray-500">
                  Analyst (You)
                  {dataOwners.some(owner => owner.email === userEmail) && ' • Also Data Owner'}
                </p>
              </div>
            </div>

            {/* Additional analysts */}
            {additionalAnalysts.map((email) => (
              <div key={email} className="bg-base-200 p-3 rounded-lg mb-2 flex justify-between items-center">
                <div>
                  <p className="font-semibold">{email}</p>
                  <p className="text-sm text-gray-500">Analyst</p>
                </div>
                <button
                  className="btn btn-sm btn-error btn-outline"
                  onClick={() => removeAnalyst(email)}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>

          {/* Add new analyst */}
          <div className="divider">Add Analyst</div>
          <div className="flex gap-2" ref={inviteRef}>
            <input
              type="text"
              placeholder="Enter email address"
              className="input input-bordered flex-1"
              value={newAnalystEmail}
              onChange={(e) => setNewAnalystEmail(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && addAnalyst()}
            />
            <button
              className="btn btn-primary"
              onClick={addAnalyst}
              disabled={!newAnalystEmail.trim()}
            >
              Add Analyst
            </button>
          </div>
        </div>

        <div className="modal-action shrink-0">
          <button className="btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
  return typeof document === 'undefined' ? modal : createPortal(modal, document.body);
});

ParticipantsModal.displayName = 'ParticipantsModal';
