import React from 'react';

export type DataOwner = { email: string; cohorts: string[] };

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
  const toggleDataOwner = (email: string) => {
    if (manuallyIncludedOwners.includes(email)) {
      setManuallyIncludedOwners(manuallyIncludedOwners.filter(e => e !== email));
    } else {
      setManuallyIncludedOwners([...manuallyIncludedOwners, email]);
    }
  };
  return (
    <div className="modal modal-open">
      <div className="modal-box">
        <h3 className="font-bold text-lg mb-4">DCR Participants</h3>

        <div className="space-y-4">
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
          <div className="flex gap-2">
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

        <div className="modal-action">
          <button className="btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
});

ParticipantsModal.displayName = 'ParticipantsModal';
