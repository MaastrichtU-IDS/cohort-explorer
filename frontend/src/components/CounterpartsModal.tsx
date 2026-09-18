import React, {useEffect, useMemo, useRef, useState} from 'react';
import Link from 'next/link';
import {Link2, ExternalLink} from 'react-feather';
import {Variable} from '@/types';
import {
  Counterpart,
  CounterpartIdentifier,
  IDENTIFIER_LABELS,
  VariableCounterparts,
  splitIdentifierValues,
} from '@/utils/counterparts';

// Rows shown per cohort before the "show all" button.
const ROWS_PER_COHORT = 15;

const IDENTIFIER_ORDER: CounterpartIdentifier[] = ['concept_code', 'omop_id'];

// Same blue as the concept code / OMOP ID badges on the variable card.
const idBadgeStyle = {backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe'};

const sameValues = (a: unknown, b: unknown): boolean => {
  const norm = (x: unknown) => splitIdentifierValues(x).map(v => v.toLowerCase()).sort().join('|');
  return norm(a) === norm(b);
};

function CohortGroup({
  cohortId,
  rows,
  variable,
  onNavigate,
}: {
  cohortId: string;
  rows: Counterpart[];
  variable: Variable;
  onNavigate: () => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? rows : rows.slice(0, ROWS_PER_COHORT);
  const hidden = rows.length - visible.length;

  return (
    <div className="rounded-lg border border-base-300 bg-base-100">
      <div className="flex items-center justify-between gap-3 px-3 py-2 border-b border-base-200">
        <Link
          href={{pathname: '/cohorts', query: {cohort: cohortId}}}
          onClick={onNavigate}
          className="inline-flex items-center gap-1.5 font-semibold link link-hover"
          title={`Open ${cohortId} on the explore page`}
        >
          {cohortId}
          <ExternalLink size={13} className="opacity-60" />
        </Link>
        <span className="badge badge-sm badge-ghost">
          {rows.length} {rows.length === 1 ? 'variable' : 'variables'}
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="table table-xs">
          <thead>
            <tr>
              <th>Variable</th>
              <th>Label</th>
              <th>Concept name</th>
              <th>Concept code</th>
              <th>OMOP ID</th>
              <th>Visit</th>
              <th>Shared</th>
            </tr>
          </thead>
          <tbody>
            {visible.map(c => {
              const v = c.variable;
              // An identifier the counterpart does NOT share, while both sides
              // carry one, points at a standardization discrepancy (the same
              // convention as the Concept Clusters page): tint it amber.
              const differs = (id: CounterpartIdentifier) =>
                !c.matchedOn.includes(id) &&
                splitIdentifierValues(v[id]).length > 0 &&
                splitIdentifierValues(variable[id]).length > 0 &&
                !sameValues(v[id], variable[id]);
              return (
                <tr key={c.varName}>
                  <td className="font-mono text-xs whitespace-nowrap">{c.varName}</td>
                  <td className="text-xs">{v.var_label || '—'}</td>
                  <td className="text-xs">{v.concept_name || '—'}</td>
                  <td className={`text-xs font-mono ${differs('concept_code') ? 'text-amber-700 dark:text-amber-400 font-semibold' : ''}`}>
                    {v.concept_code || '—'}
                  </td>
                  <td className={`text-xs font-mono ${differs('omop_id') ? 'text-amber-700 dark:text-amber-400 font-semibold' : ''}`}>
                    {v.omop_id || '—'}
                  </td>
                  <td className="text-xs">{v.visits || v.visit_concept_name || '—'}</td>
                  <td className="text-xs whitespace-nowrap">
                    {IDENTIFIER_ORDER.filter(id => c.matchedOn.includes(id)).map(id => (
                      <span key={id} className="badge badge-xs mr-1" style={idBadgeStyle}>
                        {IDENTIFIER_LABELS[id]}
                      </span>
                    ))}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {hidden > 0 && (
        <div className="px-3 py-1.5 border-t border-base-200">
          <button type="button" className="btn btn-ghost btn-xs" onClick={() => setShowAll(true)}>
            Show all {rows.length} ({hidden} more)
          </button>
        </div>
      )}
    </div>
  );
}

export default function CounterpartsModal({
  cohortId,
  variable,
  counterparts,
  onClose,
}: {
  cohortId: string;
  variable: Variable;
  counterparts: VariableCounterparts;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (!dialog.open) dialog.showModal();
    const handleClose = () => onClose();
    dialog.addEventListener('close', handleClose);
    return () => dialog.removeEventListener('close', handleClose);
  }, [onClose]);

  const byCohort = useMemo(() => {
    const groups = new Map<string, Counterpart[]>();
    for (const c of counterparts.counterparts) {
      const list = groups.get(c.cohortId);
      if (list) list.push(c);
      else groups.set(c.cohortId, [c]);
    }
    return Array.from(groups.entries());
  }, [counterparts]);

  const total = counterparts.counterparts.length;
  const nCohorts = counterparts.cohortIds.length;
  const close = () => ref.current?.close();

  return (
    <dialog ref={ref} className="modal">
      <div className="modal-box max-w-5xl space-y-3">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h3 className="font-bold text-lg flex items-center gap-2">
              <Link2 size={18} className="text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
              <span className="truncate">Counterparts of {variable.var_name}</span>
            </h3>
            {variable.var_label && <p className="text-sm text-base-content/70 mt-0.5">{variable.var_label}</p>}
            <p className="text-sm mt-2">
              <span className="font-semibold">{total}</span> {total === 1 ? 'variable' : 'variables'} in{' '}
              <span className="font-semibold">{nCohorts}</span> other {nCohorts === 1 ? 'cohort' : 'cohorts'} share
              {' '}
              {IDENTIFIER_ORDER.filter(id => counterparts.matchedValues[id].length > 0).map((id, i, arr) => (
                <span key={id}>
                  {i > 0 && (i === arr.length - 1 ? ' or ' : ', ')}
                  the {IDENTIFIER_LABELS[id]}{' '}
                  {counterparts.matchedValues[id].map(val => (
                    <span key={val} className="badge badge-sm mr-1 font-mono" style={idBadgeStyle}>
                      {val}
                    </span>
                  ))}
                </span>
              ))}
              with this {cohortId} variable.
            </p>
          </div>
          <button type="button" className="btn btn-sm btn-circle btn-ghost flex-shrink-0" onClick={close} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="space-y-3">
          {byCohort.map(([otherCohortId, rows]) => (
            <CohortGroup key={otherCohortId} cohortId={otherCohortId} rows={rows} variable={variable} onNavigate={close} />
          ))}
        </div>

        <p className="text-xs text-base-content/50">
          Amber identifiers differ from this variable&apos;s: the two cohorts standardized the same concept differently.
          The full picture across all cohorts is on the{' '}
          <Link href="/concept-clusters" className="link" onClick={close}>
            Concept Clusters
          </Link>{' '}
          page.
        </p>
      </div>
      <form method="dialog" className="modal-backdrop">
        <button>close</button>
      </form>
    </dialog>
  );
}
