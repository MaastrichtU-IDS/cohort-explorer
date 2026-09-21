import React from 'react';
import {Link2} from 'react-feather';
import {VariableSemanticMatches} from '@/utils/semanticMatches';

// Frame of a variable card that has semantic matches in other cohorts: a
// light emerald border plus a left accent whose depth grows with the number of
// other cohorts holding a match (1 / 2 / 3+), so a list sorted "matches first"
// fades from dark to light as you scroll. Dark mode brightens instead.
export const matchedFrameClass = (otherCohorts: number): string => {
  const accent =
    otherCohorts >= 3
      ? 'border-l-emerald-700 dark:border-l-emerald-300'
      : otherCohorts === 2
        ? 'border-l-emerald-500 dark:border-l-emerald-500'
        : 'border-l-emerald-300 dark:border-l-emerald-700';
  return `border-emerald-200 dark:border-emerald-800 border-l-4 ${accent}`;
};

// The "N semantic matches in M cohorts" badge, centered on a card's bottom
// line; clicking it opens the SemanticMatchesModal. Shared by the variable
// cards of the explore page and of the per-cohort search results.
export default function SemanticMatchesBadge({
  shared,
  onClick,
}: {
  shared: VariableSemanticMatches;
  onClick: () => void;
}) {
  const n = shared.matches.length;
  const c = shared.cohortIds.length;
  const same = shared.sameCohort.length;
  return (
    <div className="mt-2 flex justify-center">
      <button
        type="button"
        className="badge gap-1 border cursor-pointer bg-emerald-50 text-emerald-800 border-emerald-300 hover:bg-emerald-100 dark:bg-emerald-900/30 dark:text-emerald-200 dark:border-emerald-700 dark:hover:bg-emerald-900/50"
        title={`${n} variable${n === 1 ? '' : 's'} in ${shared.cohortIds.join(', ')} match${n === 1 ? 'es' : ''} this variable on concept code or OMOP ID${
          same > 0 ? `, plus ${same} in this cohort` : ''
        }. Click to see them.`}
        onClick={onClick}
      >
        <Link2 size={12} />
        {n} semantic {n === 1 ? 'match' : 'matches'} in {c} {c === 1 ? 'cohort' : 'cohorts'}
        {same > 0 && <span className="opacity-70 font-normal">· {same} in this cohort</span>}
      </button>
    </div>
  );
}
