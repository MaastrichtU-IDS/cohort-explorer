import React, {useEffect, useMemo, useRef, useState} from 'react';
import Link from 'next/link';
import {Link2, ExternalLink} from 'react-feather';
import {Variable} from '@/types';
import {
  SemanticMatch,
  MatchIdentifier,
  IDENTIFIER_LABELS,
  VariableSemanticMatches,
  splitIdentifierValues,
} from '@/utils/semanticMatches';
import {
  SummaryStats,
  SummaryStatsByName,
  fmtStat,
  hasValue,
  isNumericVariable,
  loadEdaSummaryStats,
  statsFor,
  variableKind,
} from '@/utils/variableStats';

// Rows shown per cohort before the "show all" button.
const ROWS_PER_COHORT = 15;
// Category encodings shown in the Values column before "+n more".
const CATEGORIES_SHOWN = 6;

const IDENTIFIER_ORDER: MatchIdentifier[] = ['concept_code', 'omop_id'];

// Same blue as the concept code / OMOP ID badges on the variable card.
const idBadgeStyle = {backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe'};

const sameValues = (a: unknown, b: unknown): boolean => {
  const norm = (x: unknown) => splitIdentifierValues(x).map(v => v.toLowerCase()).sort().join('|');
  return norm(a) === norm(b);
};

const KIND_CLASS: Record<string, string> = {
  Categorical: 'bg-amber-50 text-amber-900 border-amber-200 dark:bg-amber-900/30 dark:text-amber-100 dark:border-amber-800',
  Numeric: 'bg-sky-50 text-sky-900 border-sky-200 dark:bg-sky-900/30 dark:text-sky-100 dark:border-sky-800',
  Date: 'bg-violet-50 text-violet-900 border-violet-200 dark:bg-violet-900/30 dark:text-violet-100 dark:border-violet-800',
  Text: 'bg-base-200 text-base-content/70 border-base-300',
};

function TypeCell({v}: {v: Variable}) {
  const kind = variableKind(v);
  const detail =
    kind === 'Categorical'
      ? `${v.categories.length} ${v.categories.length === 1 ? 'category' : 'categories'}${v.var_type ? ` · ${v.var_type}` : ''}`
      : v.var_type && v.var_type !== kind
        ? v.var_type
        : '';
  return (
    <div className="whitespace-nowrap">
      <span className={`badge badge-xs border ${KIND_CLASS[kind] || KIND_CLASS.Text}`}>{kind}</span>
      {detail && <div className="text-[11px] text-base-content/50 mt-0.5">{detail}</div>}
    </div>
  );
}

// Numeric: min, max, median and unit (dictionary first, summary statistics
// as fallback). Categorical: the encodings (value → meaning).
function ValuesCell({v, stats}: {v: Variable; stats?: SummaryStats}) {
  if (v.categories?.length > 0) {
    const shown = v.categories.slice(0, CATEGORIES_SHOWN);
    const hidden = v.categories.length - shown.length;
    return (
      <div className="space-y-0.5 min-w-[10rem]">
        {shown.map((c, i) => (
          <div key={i} className="leading-tight">
            <span className="badge badge-xs badge-ghost font-mono mr-1">{c.value}</span>
            <span>{c.label}</span>
          </div>
        ))}
        {hidden > 0 && (
          <div
            className="text-[11px] text-base-content/50 italic cursor-help"
            title={v.categories.map(c => `${c.value} = ${c.label}`).join('\n')}
          >
            +{hidden} more (hover for all)
          </div>
        )}
      </div>
    );
  }
  const min = hasValue(v.min) ? v.min : stats?.min;
  const max = hasValue(v.max) ? v.max : stats?.max;
  const median = stats?.median;
  const unit = v.units || v.unit_concept_name;
  const parts: [string, string][] = [];
  if (hasValue(min)) parts.push(['Min', fmtStat(min)]);
  if (hasValue(max)) parts.push(['Max', fmtStat(max)]);
  if (hasValue(median)) parts.push(['Median', fmtStat(median)]);
  if (parts.length === 0 && !unit) return <span className="text-base-content/40">—</span>;
  return (
    <div className="whitespace-nowrap">
      {parts.map(([label, value], i) => (
        <span key={label}>
          {i > 0 && <span className="text-base-content/30"> · </span>}
          <span className="text-base-content/50">{label}</span> <span className="font-mono">{value}</span>
        </span>
      ))}
      {unit && (
        <span>
          {parts.length > 0 && <span className="text-base-content/30"> · </span>}
          <span className="text-base-content/50">Unit</span> {unit}
        </span>
      )}
      {isNumericVariable(v) && parts.length === 0 && unit && (
        <div className="text-[11px] text-base-content/40 italic">no range in dictionary or summary statistics</div>
      )}
    </div>
  );
}

function VisitCell({v}: {v: Variable}) {
  const visits = v.visits || '';
  const concept = v.visit_concept_name || '';
  if (!visits && !concept) return <span className="text-base-content/40">—</span>;
  return (
    <div>
      <div>{visits || concept}</div>
      {visits && concept && concept.toLowerCase() !== visits.toLowerCase() && (
        <div className="text-[11px] text-base-content/50">{concept}</div>
      )}
    </div>
  );
}

function HeaderRow({last}: {last: string}) {
  return (
    <thead>
      <tr>
        <th>Variable</th>
        <th>Label</th>
        <th>Concept name</th>
        <th>Concept code</th>
        <th>OMOP ID</th>
        <th>Visit</th>
        <th>Type</th>
        <th>Values</th>
        <th>{last}</th>
      </tr>
    </thead>
  );
}

// The cells shared by the reference row (the variable itself) and each match.
// `reference` is the variable the row is compared with: an identifier the row
// does NOT share, while both sides carry one, points at a standardization
// discrepancy (the same convention as the Concept Clusters page) and is tinted
// amber.
function VariableCells({
  v,
  stats,
  reference,
  matchedOn,
}: {
  v: Variable;
  stats?: SummaryStats;
  reference?: Variable;
  matchedOn?: MatchIdentifier[];
}) {
  const differs = (id: MatchIdentifier) =>
    !!reference &&
    !!matchedOn &&
    !matchedOn.includes(id) &&
    splitIdentifierValues(v[id]).length > 0 &&
    splitIdentifierValues(reference[id]).length > 0 &&
    !sameValues(v[id], reference[id]);
  const idClass = (id: MatchIdentifier) =>
    `text-xs font-mono ${differs(id) ? 'text-amber-700 dark:text-amber-400 font-semibold' : ''}`;
  return (
    <>
      <td className="text-xs">{v.var_label || '—'}</td>
      <td className="text-xs">{v.concept_name || '—'}</td>
      <td className={idClass('concept_code')}>{v.concept_code || '—'}</td>
      <td className={idClass('omop_id')}>{v.omop_id || '—'}</td>
      <td className="text-xs">
        <VisitCell v={v} />
      </td>
      <td className="text-xs">
        <TypeCell v={v} />
      </td>
      <td className="text-xs">
        <ValuesCell v={v} stats={stats} />
      </td>
    </>
  );
}

function CohortGroup({
  cohortId,
  rows,
  variable,
  stats,
  onNavigate,
}: {
  cohortId: string;
  rows: SemanticMatch[];
  variable: Variable;
  stats?: SummaryStatsByName;
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
          <HeaderRow last="Matched on" />
          <tbody>
            {visible.map(c => (
              <tr key={c.varName} className="align-top">
                <td className="font-mono text-xs whitespace-nowrap">{c.varName}</td>
                <VariableCells v={c.variable} stats={statsFor(stats, c.varName)} reference={variable} matchedOn={c.matchedOn} />
                <td className="text-xs whitespace-nowrap">
                  {IDENTIFIER_ORDER.filter(id => c.matchedOn.includes(id)).map(id => (
                    <span key={id} className="badge badge-xs mr-1" style={idBadgeStyle}>
                      {IDENTIFIER_LABELS[id]}
                    </span>
                  ))}
                </td>
              </tr>
            ))}
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

export default function SemanticMatchesModal({
  cohortId,
  variable,
  semanticMatches,
  onClose,
}: {
  cohortId: string;
  variable: Variable;
  semanticMatches: VariableSemanticMatches;
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

  // Summary statistics (median; min/max fallback) of every cohort on screen:
  // this variable's own cohort and each cohort holding a match. One fetch per
  // cohort; a cohort without statistics simply shows no median.
  const [statsByCohort, setStatsByCohort] = useState<Record<string, SummaryStatsByName>>({});
  useEffect(() => {
    let alive = true;
    const ids = Array.from(new Set([cohortId, ...semanticMatches.cohortIds]));
    ids.forEach(id => {
      loadEdaSummaryStats(id).then(map => {
        if (alive) setStatsByCohort(prev => ({...prev, [id]: map}));
      });
    });
    return () => {
      alive = false;
    };
  }, [cohortId, semanticMatches]);

  const byCohort = useMemo(() => {
    const groups = new Map<string, SemanticMatch[]>();
    for (const c of semanticMatches.matches) {
      const list = groups.get(c.cohortId);
      if (list) list.push(c);
      else groups.set(c.cohortId, [c]);
    }
    return Array.from(groups.entries());
  }, [semanticMatches]);

  const total = semanticMatches.matches.length;
  const nCohorts = semanticMatches.cohortIds.length;
  const close = () => ref.current?.close();

  return (
    <dialog ref={ref} className="modal">
      <div className="modal-box max-w-7xl w-[95vw] space-y-3">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h3 className="font-bold text-lg flex items-center gap-2">
              <Link2 size={18} className="text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
              <span className="truncate">Semantic matches of {variable.var_name}</span>
            </h3>
            <p className="text-sm mt-1">
              <span className="font-semibold">{total}</span> {total === 1 ? 'variable' : 'variables'} in{' '}
              <span className="font-semibold">{nCohorts}</span> other {nCohorts === 1 ? 'cohort' : 'cohorts'}{' '}
              {total === 1 ? 'matches' : 'match'} on{' '}
              {IDENTIFIER_ORDER.filter(id => semanticMatches.matchedValues[id].length > 0).map((id, i, arr) => (
                <span key={id}>
                  {i > 0 && (i === arr.length - 1 ? ' or ' : ', ')}
                  the {IDENTIFIER_LABELS[id]}{' '}
                  {semanticMatches.matchedValues[id].map(val => (
                    <span key={val} className="badge badge-sm mr-1 font-mono" style={idBadgeStyle}>
                      {val}
                    </span>
                  ))}
                </span>
              ))}
              .
            </p>
          </div>
          <button type="button" className="btn btn-sm btn-circle btn-ghost flex-shrink-0" onClick={close} aria-label="Close">
            ✕
          </button>
        </div>

        {/* The variable itself, for side-by-side comparison with its matches */}
        <div className="rounded-lg border border-emerald-200 dark:border-emerald-800 border-l-4 border-l-emerald-500 bg-base-100">
          <div className="flex items-center justify-between gap-3 px-3 py-2 border-b border-base-200">
            <span className="font-semibold">
              This variable <span className="text-base-content/50 font-normal">({cohortId})</span>
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="table table-xs">
              <HeaderRow last="" />
              <tbody>
                <tr className="align-top">
                  <td className="font-mono text-xs whitespace-nowrap">{variable.var_name}</td>
                  <VariableCells v={variable} stats={statsFor(statsByCohort[cohortId], variable.var_name)} />
                  <td />
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="space-y-3">
          {byCohort.map(([otherCohortId, rows]) => (
            <CohortGroup
              key={otherCohortId}
              cohortId={otherCohortId}
              rows={rows}
              variable={variable}
              stats={statsByCohort[otherCohortId]}
              onNavigate={close}
            />
          ))}
        </div>

        <p className="text-xs text-base-content/50">
          Values: min / max from the data dictionary, or from the cohort&apos;s summary statistics when the dictionary has
          none; median from the summary statistics. Amber identifiers differ from this variable&apos;s: the two cohorts
          standardized the same concept differently. The full picture across all cohorts is on the{' '}
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
