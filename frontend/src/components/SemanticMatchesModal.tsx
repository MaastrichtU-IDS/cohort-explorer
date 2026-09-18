import React, {useEffect, useMemo, useRef, useState} from 'react';
import Link from 'next/link';
import {Link2, ExternalLink, AlertTriangle, ChevronDown, ChevronRight, X} from 'react-feather';
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
  valueRange,
  variableKind,
} from '@/utils/variableStats';

// Rows shown per cohort before the "show all" button.
const ROWS_PER_COHORT = 15;

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

// Numeric: min, max, median and unit (summary statistics first, the
// dictionary's min / max as fallback - see valueRange). Categorical: every
// encoding (value -> meaning) on one wrapping line.
function ValuesCell({v, stats}: {v: Variable; stats?: SummaryStats}) {
  if (v.categories?.length > 0) {
    return (
      <div className="min-w-[12rem] leading-relaxed">
        {v.categories.map((c, i) => (
          <span key={i} className="whitespace-nowrap">
            {i > 0 && <span className="text-base-content/30">, </span>}
            <span className="badge badge-xs badge-ghost font-mono mr-1">{c.value}</span>
            <span>{c.label}</span>
          </span>
        ))}
      </div>
    );
  }
  const {min, max, median} = valueRange(v, stats);
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

function HeaderRow() {
  return (
    <thead>
      <tr>
        <th>Variable</th>
        <th>Label</th>
        <th>Visit</th>
        <th>Type</th>
        <th>Values</th>
      </tr>
    </thead>
  );
}

// Identifiers on which a match does NOT agree with the reference variable
// while both sides carry one: the two cohorts standardized the same concept
// differently. Shown as an amber marker next to the variable name.
function discrepancies(v: Variable, reference: Variable, matchedOn: MatchIdentifier[]): string[] {
  return IDENTIFIER_ORDER.filter(
    id =>
      !matchedOn.includes(id) &&
      splitIdentifierValues(v[id]).length > 0 &&
      splitIdentifierValues(reference[id]).length > 0 &&
      !sameValues(v[id], reference[id])
  ).map(id => `${IDENTIFIER_LABELS[id]}: ${v[id]} (this variable: ${reference[id]})`);
}

function VariableRow({
  name,
  v,
  stats,
  discrepancy,
}: {
  name: string;
  v: Variable;
  stats?: SummaryStats;
  discrepancy?: string[];
}) {
  return (
    <tr className="align-top">
      <td className="font-mono text-xs whitespace-nowrap">
        {discrepancy && discrepancy.length > 0 && (
          <span title={`Standardized differently - ${discrepancy.join('; ')}`} className="cursor-help">
            <AlertTriangle size={11} className="inline mr-1 text-amber-700 dark:text-amber-400" />
          </span>
        )}
        {name}
      </td>
      <td className="text-xs">{v.var_label || '—'}</td>
      <td className="text-xs">
        <VisitCell v={v} />
      </td>
      <td className="text-xs">
        <TypeCell v={v} />
      </td>
      <td className="text-xs">
        <ValuesCell v={v} stats={stats} />
      </td>
    </tr>
  );
}

function CohortGroup({
  cohortId,
  rows,
  variable,
  stats,
  collapsed,
  onToggle,
  onNavigate,
  innerRef,
}: {
  cohortId: string;
  rows: SemanticMatch[];
  variable: Variable;
  stats?: SummaryStatsByName;
  collapsed: boolean;
  onToggle: () => void;
  onNavigate: () => void;
  innerRef: (el: HTMLDivElement | null) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? rows : rows.slice(0, ROWS_PER_COHORT);
  const hidden = rows.length - visible.length;

  return (
    <div ref={innerRef} className="rounded-lg border border-base-300 bg-base-100 scroll-mt-2">
      <div
        className="flex items-center gap-2 px-3 py-2 cursor-pointer select-none hover:bg-base-200/60 rounded-t-lg"
        onClick={onToggle}
        role="button"
        aria-expanded={!collapsed}
      >
        {collapsed ? <ChevronRight size={14} className="opacity-60" /> : <ChevronDown size={14} className="opacity-60" />}
        <span className="font-semibold">{cohortId}</span>
        <span className="badge badge-sm badge-ghost">
          {rows.length} {rows.length === 1 ? 'variable' : 'variables'}
        </span>
        <span className="flex-1" />
        <Link
          href={{pathname: '/cohorts', query: {cohort: cohortId, tab: 'list'}}}
          onClick={e => {
            e.stopPropagation();
            onNavigate();
          }}
          className="inline-flex items-center gap-1 text-xs link link-hover text-base-content/60"
          title={`Open the variables list of ${cohortId}`}
        >
          open study <ExternalLink size={12} />
        </Link>
      </div>
      {!collapsed && (
        <>
          <div className="overflow-x-auto border-t border-base-200">
            <table className="table table-xs">
              <HeaderRow />
              <tbody>
                {visible.map(c => (
                  <VariableRow
                    key={c.varName}
                    name={c.varName}
                    v={c.variable}
                    stats={statsFor(stats, c.varName)}
                    discrepancy={discrepancies(c.variable, variable, c.matchedOn)}
                  />
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
        </>
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
  const groupRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

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

  // A cohort chip expands its group (if collapsed) and scrolls the modal to it.
  const jumpTo = (id: string) => {
    setCollapsed(prev => ({...prev, [id]: false}));
    requestAnimationFrame(() => {
      groupRefs.current[id]?.scrollIntoView({behavior: 'smooth', block: 'start'});
    });
  };

  return (
    <dialog ref={ref} className="modal">
      <div className="modal-box max-w-6xl w-[95vw] space-y-3">
        {/* Close button pinned to the top-right corner while the modal scrolls:
            a zero-height sticky row at the very top of the scroll container. */}
        <div className="sticky top-0 z-20 h-0 flex justify-end pointer-events-none">
          <button
            type="button"
            className="pointer-events-auto btn btn-circle btn-ghost bg-base-100/90 shadow-sm -mt-2 -mr-2"
            onClick={close}
            aria-label="Close"
            title="Close"
          >
            <X size={24} />
          </button>
        </div>
        <div className="flex items-start justify-between gap-4 pr-12 !mt-0">
          <div className="min-w-0 flex-1">
            <h3 className="font-bold text-lg flex items-center gap-2">
              <Link2 size={18} className="text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
              <span className="truncate">Semantic matches of {variable.var_name}</span>
              <span className="text-sm font-normal text-base-content/50 whitespace-nowrap">({cohortId})</span>
            </h3>
            {/* The concept behind the matches: its name and identifiers, prominent */}
            <div className="mt-2 rounded-lg bg-base-200/60 px-4 py-3">
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="text-xs uppercase tracking-wide text-base-content/70 whitespace-nowrap">
                  Standard concept name
                </span>
                {variable.concept_name ? (
                  <span className="text-xl font-semibold leading-snug text-red-400 dark:text-red-300">
                    {variable.concept_name}
                  </span>
                ) : (
                  <span className="text-base text-base-content/50 italic">none in the dictionary</span>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2 mt-2">
                {variable.concept_code && (
                  <span className="badge badge-lg font-mono text-base" style={idBadgeStyle}>
                    {variable.concept_code}
                  </span>
                )}
                {variable.omop_id && (
                  <span className="badge badge-lg font-mono text-base" style={idBadgeStyle}>
                    OMOP ID: {variable.omop_id}
                  </span>
                )}
              </div>
            </div>
            <p className="text-sm mt-2">
              <span className="font-semibold">{total}</span> {total === 1 ? 'variable' : 'variables'} in{' '}
              <span className="font-semibold">{nCohorts}</span> other {nCohorts === 1 ? 'cohort' : 'cohorts'}{' '}
              {total === 1 ? 'matches' : 'match'} on{' '}
              {IDENTIFIER_ORDER.filter(id => semanticMatches.matchedValues[id].length > 0)
                .map(id => `the ${IDENTIFIER_LABELS[id]}`)
                .join(' or ')}
              .
            </p>
            {/* Cohort chips: one per cohort with matches, click to jump to its group */}
            <div className="flex flex-wrap items-center gap-1.5 mt-2">
              <span className="text-xs text-base-content/50">In:</span>
              {byCohort.map(([id, rows]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => jumpTo(id)}
                  className="badge badge-sm gap-1 border cursor-pointer bg-emerald-50 text-emerald-800 border-emerald-300 hover:bg-emerald-100 dark:bg-emerald-900/30 dark:text-emerald-200 dark:border-emerald-700 dark:hover:bg-emerald-900/50"
                  title={`Jump to the ${rows.length} matching ${rows.length === 1 ? 'variable' : 'variables'} of ${id}`}
                >
                  {id}
                  <span className="font-mono text-[10px] opacity-70">{rows.length}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* The variable itself, for side-by-side comparison with its matches */}
        <div className="rounded-lg border border-emerald-200 dark:border-emerald-800 border-l-4 border-l-emerald-500 bg-base-100">
          <div className="px-3 py-2 border-b border-base-200 font-semibold">
            This variable <span className="text-base-content/50 font-normal">({cohortId})</span>
          </div>
          <div className="overflow-x-auto">
            <table className="table table-xs">
              <HeaderRow />
              <tbody>
                <VariableRow name={variable.var_name} v={variable} stats={statsFor(statsByCohort[cohortId], variable.var_name)} />
              </tbody>
            </table>
          </div>
        </div>

        <div className="space-y-8 pt-2">
          {byCohort.map(([otherCohortId, rows]) => (
            <CohortGroup
              key={otherCohortId}
              cohortId={otherCohortId}
              rows={rows}
              variable={variable}
              stats={statsByCohort[otherCohortId]}
              collapsed={!!collapsed[otherCohortId]}
              onToggle={() => setCollapsed(prev => ({...prev, [otherCohortId]: !prev[otherCohortId]}))}
              onNavigate={close}
              innerRef={el => {
                groupRefs.current[otherCohortId] = el;
              }}
            />
          ))}
        </div>

        <p className="text-xs text-base-content/50">
          Values: min / max / median from the cohort&apos;s summary statistics; the data dictionary&apos;s min / max only
          when no statistics are available. An amber marker next to a variable means the other cohort
          standardized the same concept with a different concept code or OMOP ID (hover for details).
        </p>
        <div className="modal-action justify-center">
          <button type="button" className="btn btn-sm btn-ghost border border-base-300 px-6" onClick={close}>
            Close
          </button>
        </div>
      </div>
      <form method="dialog" className="modal-backdrop">
        <button>close</button>
      </form>
    </dialog>
  );
}
