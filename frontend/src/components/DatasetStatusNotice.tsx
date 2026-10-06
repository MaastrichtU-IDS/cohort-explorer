'use client';

// Notice on a My DCRs card when some of the DCR's main cohort datasets (the
// data nodes named exactly after a cohort) are not provisioned on Decentriq.
// Styled like the "This DCR has been deactivated" notice. Shows nothing while
// loading, when every dataset is there, or when the status cannot be read.
import React, {useEffect, useState} from 'react';
import {AlertCircle} from 'react-feather';
import {apiUrl} from '@/utils';

/** "A", "A and B", "A, B and C", with each name in bold. */
function NameList({names}: {names: string[]}) {
  return (
    <>
      {names.map((name, i) => (
        <React.Fragment key={name}>
          {i > 0 && (i === names.length - 1 ? ' and ' : ', ')}
          <span className="font-bold">{name}</span>
        </React.Fragment>
      ))}
    </>
  );
}

export function DatasetStatusNotice({dcrId}: {dcrId: string}) {
  const [missing, setMissing] = useState<string[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetch(`${apiUrl}/my-dcrs/${encodeURIComponent(dcrId)}/dataset-status`, {credentials: 'include'})
      .then(res => (res.ok ? res.json() : null))
      .then(data => {
        if (!cancelled && Array.isArray(data?.missing)) setMissing(data.missing);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [dcrId]);

  if (missing.length === 0) return null;
  const one = missing.length === 1;
  return (
    <div className="mt-3 alert bg-base-200 border-base-300 text-base-content font-semibold">
      <AlertCircle size={20} />
      {/* Normal weight, so the cohort names can stand out in bold. */}
      <span className="font-normal">
        {one ? 'Cohort dataset' : 'Cohort datasets'} <NameList names={missing} /> {one ? 'has' : 'have'} not yet
        been provisioned, or {one ? 'has' : 'have'} been de-provisioned. Computations that use the data cannot run.
      </span>
    </div>
  );
}
