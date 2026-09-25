// Concept Coverage (/concept-coverage): every concept, threaded through every
// cohort. Rows are concepts (variables of any cohort sharing a concept code or
// OMOP ID), columns are all registered cohorts grouped by what we know about
// them, and cells show the observation counts of the cohorts that went
// through EDA. The page owns the URL-backed view state and derives everything
// once per change (see utils/conceptCoverage.ts); the components below read it
// through LoomContext.

import React, {useSyncExternalStore, createContext, useContext, useMemo, useState, forwardRef, useImperativeHandle, useRef, useCallback, useEffect, useLayoutEffect, useId, useDeferredValue, memo, useReducer} from 'react';
import Link from 'next/link';
import {Eye, EyeOff, Search as SearchIcon, X, ChevronDown, ChevronRight, Minus, Plus, Layout as LayoutIcon, CornerDownLeft, ChevronLeft, AlertCircle, Bookmark, Grid, RefreshCw, RotateCcw, Sliders, Table, ChevronsLeft, ChevronsRight, ArrowLeft, HelpCircle, Share2, Download, Info, ChevronUp, AlertTriangle} from 'react-feather';
import Head from 'next/head';
import {useRouter} from 'next/router';
import {useCohorts} from '@/components/CohortsContext';
import LoginPrompt from '@/components/LoginPrompt';
import SemanticMatchesModal from '@/components/SemanticMatchesModal';
import VariableGraphModal from '@/components/VariableGraphModal';
import {semanticMatchKey} from '@/utils/semanticMatches';
import {type PointerTarget, BasketPreview, type ColumnLayout, type DensitySpec, type FilterResult, type LensCells, LoomAction, LoomModel, LoomUiState, LoomViewState, type PoolResult, type RowLayout, type RowStats, type SearchQuery, guestMembers, rowBridgePairs, lacksVocabularyPrefix, tokenLabel, fmtInt, type Domain, type Tier, DOMAIN_LABEL, DOMAIN_MONOGRAM, TIER_LABEL, fmtCompact, type LoomCohort, type PinKind, CellState, type FlatItem, type Metric, binFor, type LoomTokens, type LoomRow, SLOT_LONG, CellFlag, type Density, type VisitSlot, itemAtY, ALL_SLOTS_MASK, BAND_H, SLOT_BIT, VISIT_SLOTS, type VisitLens, computeRowStats, lensMembers, resolveCell, readLoomTokens, cellReadout, cohortReadout, rowReadout, columnAtX, ALL_LENS, jaccard, GROUP_GAP, type GridColumn, METRIC_LABEL, lensLabel, type ColGroup, type ColSort, type RowGroup, type RowSort, applySuggestion, parseSearch, searchSuggestions, toggleChipNegation, type SearchSuggestion, DOMAINS, SLOT_SHORT, FOLLOW_UP_MASK, METRIC_EXPLAINER, activePreset, VIEW_PRESETS, type CellStateValue, type ResolvedCell, CAVEATS, fmtPct, isAllLens, lensIncludes, cellMembers, type PoolCohort, type PoolStatus, BASELINE_LENS, sameLens, poolBasketPreview, cohortPasses, filterRows, DESIGN_LABEL, FAMILY_LABEL, STATUS_LABEL, DEFAULT_VIEW_STATE, SIZE_MAX, SIZE_MIN, COHORT_STATUSES, DESIGN_FAMILIES, STUDY_FAMILIES, type MatchMode, type TypeClass, BIN_LABELS, type CountsStatus, InspectorTarget, type LoomVariable, homeMembers, TIER_NOTE_LABEL, slotsOfMask, slotRowKeys, countAlreadyIn, useObservationCounts, reduceViewState, type FilterChip, type SetMode, cellSummary, stateLabel, type Toast, buildModel, resolveLens, buildColumns, buildRowLayout, orderRows, computePool, decodeViewState, encodeViewState, runSelfCheck, applyPreview, DcrBasket, persistBasket, DENSITY, buildAxes, cohortSlotMasks, type AxisItem, type CoverageAxes} from '@/utils/conceptCoverage';

// ============================================================================
// pointerStore
// ============================================================================

// Hover and keyboard focus of the Loom grid, kept OUTSIDE React state so a
// mousemove never re-renders the page: the grid writes, and only the few
// components that show the pointed-at mark (tooltip, FOCUS bar, header
// highlight) subscribe.

type Listener = () => void;

interface PointerStore {
  get: () => PointerSnapshot;
  setHover: (t: PointerTarget | null) => void;
  setFocus: (t: PointerTarget | null) => void;
  subscribe: (l: Listener) => () => void;
}

interface PointerSnapshot {
  hover: PointerTarget | null;
  focus: PointerTarget | null;
  // Screen position of the pointer (tooltip placement), client coordinates.
  clientX: number;
  clientY: number;
}

const same = (a: PointerTarget | null, b: PointerTarget | null): boolean =>
  a === b ||
  (!!a && !!b && a.kind === b.kind && a.row === b.row && a.cohort === b.cohort && a.item === b.item && a.slot === b.slot && a.source === b.source);

function createPointerStore(): PointerStore & {setClient: (x: number, y: number) => void} {
  let snap: PointerSnapshot = {hover: null, focus: null, clientX: 0, clientY: 0};
  const listeners = new Set<Listener>();
  const emit = () => listeners.forEach(l => l());
  return {
    get: () => snap,
    setHover: t => {
      if (same(snap.hover, t)) return;
      snap = {...snap, hover: t};
      emit();
    },
    setFocus: t => {
      if (same(snap.focus, t)) return;
      snap = {...snap, focus: t};
      emit();
    },
    // Updates the position without notifying: readers pick it up with the
    // next hover change, or read it imperatively (tooltip follows the pointer
    // through a ref, not through React).
    setClient: (x, y) => {
      snap.clientX = x;
      snap.clientY = y;
    },
    subscribe: l => {
      listeners.add(l);
      return () => listeners.delete(l);
    }
  };
}

function usePointer(store: PointerStore): PointerSnapshot {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}

// The mark the readouts describe: keyboard focus wins over hover once the
// grid has focus; otherwise the hovered mark.
const activeTarget = (s: PointerSnapshot): PointerTarget | null => s.hover ?? s.focus;

// ============================================================================
// LoomContext
// ============================================================================

// Everything a Loom component needs, computed once per change by the page
// (pages/loom.tsx) and shared through context so no component re-derives it.
interface LoomContextValue {
  model: LoomModel;
  cells: LensCells; // under state.visit / state.floor / state.guests
  stats: RowStats; // over the visible columns
  columns: ColumnLayout;
  search: SearchQuery;
  filter: FilterResult;
  layout: RowLayout; // visible rows, ordered, with bands / tallies / sub-rows
  pool: PoolResult | null; // null when the Pool is empty
  state: LoomViewState; // URL-backed
  dispatch: (action: LoomAction) => void;
  ui: LoomUiState;
  setUi: (patch: Partial<LoomUiState> | ((prev: LoomUiState) => Partial<LoomUiState>)) => void;
  density: DensitySpec;
  pointer: PointerStore;
  // DCR basket: dataCleanRoom.cohorts (cohort id -> variable names).
  basket: Record<string, string[]>;
  // Opens the basket preview popover (DcrAddPreview commits it).
  previewBasket: (preview: BasketPreview) => void;
  // Commits a preview to the basket (with an Undo toast).
  commitBasket: (preview: BasketPreview) => void;
  retryCounts: () => void;
  theme: 'light' | 'dark';
  // Scroll the grid so a row / a cohort column is in view (and flash it).
  revealRow: (row: number) => void;
  revealCohort: (cohort: number) => void;
  // Opens the Explore page's semantic matches modal / variable graph for a variable.
  openSemanticMatches: (variable: number) => void;
  openDistribution: (variable: number) => void;
  // True during the first-load "shuttle reveal" (rings fill column by column).
  revealProgress: number | null; // 0..1 while animating, null otherwise
  // The matrix axes: concepts and cohort visits, and which runs down the rows.
  axes: CoverageAxes;
}

const LoomContext = createContext<LoomContextValue | null>(null);

function useLoom(): LoomContextValue {
  const ctx = useContext(LoomContext);
  if (!ctx) throw new Error('useLoom() outside <LoomContext.Provider>');
  return ctx;
}

// ============================================================================
// BridgeExplainer
// ============================================================================

// The bridge explainer (spec §5 step 6): why these variables are one row.
// A row is a connected component of identifiers: a variable carrying both a
// concept code and an OMOP ID ties them together, and a chain of such pairs
// across cohorts merges into one concept. Listing every distinct
// (code <-> ID) pair with who uses it makes each link auditable.

const PAIRS_SHOWN = 8;

const plural = (n: number, one: string, many: string) => `${fmtInt(n)} ${n === 1 ? one : many}`;

function BridgeExplainer({row}: {row: number}) {
  const {model} = useLoom();
  const r = model.rows[row];
  const [showAll, setShowAll] = useState(false);

  const {pairs, guests, links} = useMemo(() => {
    const list = rowBridgePairs(model, row);
    let guestCount = 0;
    for (let c = 0; c < model.K; c++) guestCount += guestMembers(model, row, c).length;
    // Identifiers that occur in two or more pairs are the links that tie the
    // pairs into one concept.
    const seen = new Map<string, number>();
    for (const p of list) for (const t of [p.cc, p.oi]) if (t) seen.set(t, (seen.get(t) ?? 0) + 1);
    return {
      pairs: list,
      guests: guestCount,
      links: new Set(Array.from(seen.entries()).flatMap(([t, n]) => (n > 1 ? [t] : [])))
    };
  }, [model, row]);

  const ccHeads = new Set(pairs.flatMap(p => (p.cc ? [p.cc] : []))).size;
  const oiHeads = new Set(pairs.flatMap(p => (p.oi ? [p.oi] : []))).size;
  const visible = showAll ? pairs : pairs.slice(0, PAIRS_SHOWN);

  const ident = (t: string | null, kind: 'code' | 'ID') => {
    if (!t)
      return (
        <span
          className="text-base-content/40"
          title={`No ${kind === 'code' ? 'concept code' : 'OMOP ID'} on these variables`}
        >
          —
        </span>
      );
    const link = links.has(t);
    const noPrefix = kind === 'code' && lacksVocabularyPrefix(t);
    return (
      <span
        className={`font-mono ${link ? 'font-semibold underline decoration-dotted underline-offset-2' : ''}`}
        title={
          link
            ? `${tokenLabel(t)} is shared by several pairs: it ties them into one concept`
            : noPrefix
              ? `${tokenLabel(t)}: no vocabulary prefix (kept as written; no vocabulary is inferred)`
              : tokenLabel(t)
        }
      >
        {tokenLabel(t)}
        {noPrefix && <span className="font-sans font-normal text-base-content/50"> (no prefix)</span>}
      </span>
    );
  };

  return (
    <div className="space-y-2 text-xs">
      <p className="leading-snug text-base-content/80">
        {pairs.length <= 1 ? (
          'Every variable of this concept carries the same identifiers.'
        ) : (
          <>
            This concept joins <span className="font-semibold loom-tabular">{plural(ccHeads, 'code', 'codes')}</span>{' '}
            and <span className="font-semibold loom-tabular">{plural(oiHeads, 'OMOP ID', 'OMOP IDs')}</span> in{' '}
            {pairs.length} distinct pairs. A variable carrying both a code and an ID links them; a chain of such pairs
            across cohorts becomes one row.{links.size > 0 && ' Underlined identifiers are the links.'}
          </>
        )}
      </p>
      {r.flags.suspect && (
        <p className="leading-snug">
          <span className="font-semibold">⚠ Suspect bridge.</span> More than six distinct codes or IDs are joined: one
          mis-mapped variable can glue two unrelated concepts together. Check the pairs used by a single variable.
        </p>
      )}
      <table className="w-full table-fixed text-left">
        <caption className="sr-only">Distinct concept code and OMOP ID pairs of this concept</caption>
        <colgroup>
          <col className="w-[34%]" />
          <col className="w-[24%]" />
          <col />
          <col className="w-9" />
        </colgroup>
        <thead>
          <tr className="text-[10px] uppercase tracking-wide text-base-content/50">
            <th scope="col" className="py-1 pr-2 font-medium">
              Concept code
            </th>
            <th scope="col" className="py-1 pr-2 font-medium">
              OMOP ID
            </th>
            <th scope="col" className="py-1 pr-2 font-medium">
              Cohorts
            </th>
            <th scope="col" className="py-1 text-right font-medium">
              Vars
            </th>
          </tr>
        </thead>
        <tbody>
          {visible.map(p => (
            <tr key={`${p.cc}|${p.oi}`} className="border-t border-[color:var(--loom-hairline)] align-top">
              <td className="break-all py-1 pr-2">{ident(p.cc, 'code')}</td>
              <td className="break-all py-1 pr-2">{ident(p.oi, 'ID')}</td>
              <td className="py-1 pr-2 text-base-content/70" title={p.cohorts.join(', ')}>
                <span className="loom-tabular">{p.cohorts.length}</span>
                <span className="text-base-content/50">
                  {' · '}
                  {p.cohorts.slice(0, 3).join(', ')}
                  {p.cohorts.length > 3 && ` +${p.cohorts.length - 3}`}
                </span>
              </td>
              <td className="py-1 text-right loom-tabular">{fmtInt(p.variables)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {pairs.length > PAIRS_SHOWN && (
        <button
          type="button"
          className="btn btn-ghost btn-xs font-normal"
          onClick={() => setShowAll(s => !s)}
          title={showAll ? 'Show only the most used pairs' : 'Show every distinct pair'}
        >
          {showAll ? 'Show fewer' : `Show all ${pairs.length} pairs`}
        </button>
      )}
      {guests > 0 && (
        <p className="leading-snug text-base-content/60">
          {plural(guests, 'more variable joins', 'more variables join')} as a guest through a secondary code (a legacy
          pipe-separated list); guests never link concepts together.
        </p>
      )}
      <Link
        href="/concept-clusters"
        className="link link-hover inline-block text-base-content/70"
        title="Compare how cohorts group variables by concept name, code and OMOP ID"
      >
        Compare groupings in Concept clusters →
      </Link>
    </div>
  );
}

// ============================================================================
// WovenGlyph
// ============================================================================

// The Loom's small SVG marks: the woven wordmark, and the cell-grammar glyphs
// that the legend, the help sheet, the knowledge bar and the cohort list draw
// so they read exactly like the canvas cells. Every colour is a --loom-* token
// (set on .loom-root), so light / dark follow the theme with no re-render.

const SURFACE = 'oklch(var(--b1))';

// ---------------------------------------------------------------------------
// Wordmark: 3 ink warp threads over and under 3 accent weft threads in a
// plain weave (4px threads, 4px apart, filling the 20px square). Where a
// thread passes under another it is cut by the matrix's own 2px surface gap
// on each side, so the thread on top reads as lying over it.
// ---------------------------------------------------------------------------

const BAR = 4;
const BARS = [0, 8, 16];
const CUT = 2;

function WovenGlyph({size = 20, className = ''}: {size?: number; className?: string}) {
  const pieces: React.ReactNode[] = [];
  // Weft (horizontal, accent) under the warp where (i + j) is even, over it otherwise.
  BARS.forEach((y, j) => {
    const cuts = BARS.filter((_, i) => (i + j) % 2 === 0);
    segments(0, 20, cuts).forEach(([a, b], k) =>
      pieces.push(
        <rect key={`w${j}-${k}`} x={a} y={y} width={b - a} height={BAR} style={{fill: 'var(--loom-accent)'}} />
      )
    );
  });
  BARS.forEach((x, i) => {
    const cuts = BARS.filter((_, j) => (i + j) % 2 === 1);
    segments(0, 20, cuts).forEach(([a, b], k) =>
      pieces.push(<rect key={`p${i}-${k}`} x={x} y={a} width={BAR} height={b - a} style={{fill: 'var(--loom-ink)'}} />)
    );
  });
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" className={className} aria-hidden="true" focusable="false">
      {pieces}
    </svg>
  );
}

// [start, end) spans of a thread that passes UNDER the crossing threads at
// `cuts`, leaving the surface gap on each side of every one of them.
function segments(start: number, end: number, cuts: number[]): [number, number][] {
  const out: [number, number][] = [];
  let at = start;
  for (const c of cuts) {
    if (c - CUT > at) out.push([at, c - CUT]);
    at = c + BAR + CUT;
  }
  if (end > at) out.push([at, end]);
  return out;
}

// ---------------------------------------------------------------------------
// Cell glyphs (spec §4.3): "the outline says the dictionary lists it; what's
// inside says what profiling found".
// ---------------------------------------------------------------------------

type GlyphKind =
  | 'fill' // profiled, values counted: ramp bin
  | 'presence' // counted, drawn in ink (Presence metric, or dataset size unknown)
  | 'ring' // dictionary only: listed, count unknown
  | 'pending' // counts loading / unavailable: the same plain ring as the grid draws
  | 'zero' // profiled: no values
  | 'notInEda' // listed, not found in the profiled data
  | 'dot' // coded at another visit (visit lens active)
  | 'fog' // no dictionary: unknown, not absent
  | 'blank'; // not coded with this concept

interface CellGlyphProps {
  kind: GlyphKind;
  bin?: number; // 1..5 for 'fill'
  dogEar?: boolean; // several variables in the cell
  floor?: boolean; // counted, below the usable floor: a 3px bar
  guest?: boolean; // guest-only member: 60% size
  width?: number;
  height?: number;
  className?: string;
}

function CellGlyph({
  kind,
  bin = 3,
  dogEar,
  floor,
  guest,
  width = 14,
  height = 12,
  className = ''
}: CellGlyphProps) {
  const W = width;
  const H = height;
  if (kind === 'fog') {
    // The hatch is the grid's own CSS layer, so the legend can never drift from it.
    return (
      <span
        className={`loom-fog inline-block shrink-0 align-middle ${className}`}
        style={{width: W, height: H}}
        aria-hidden="true"
      />
    );
  }
  const r = W >= 18 ? 3 : W >= 10 ? 2 : 0;
  const mark = {fill: 'none', stroke: 'var(--loom-mark)', strokeWidth: 1};
  const parts: React.ReactNode[] = [];
  if (kind === 'fill' || kind === 'presence') {
    const color = kind === 'fill' ? `var(--loom-bin-${Math.max(1, Math.min(5, bin))})` : 'var(--loom-ink-fill)';
    if (floor) {
      parts.push(<rect key="f" x={0} y={H - 3} width={W} height={3} rx={1} style={{fill: color}} />);
    } else {
      parts.push(<rect key="f" x={0} y={0} width={W} height={H} rx={r} style={{fill: color}} />);
      if (dogEar) parts.push(<path key="d" d={`M${W - 3.5} 0H${W}V3.5Z`} style={{fill: SURFACE}} />);
    }
  } else if (kind === 'ring' || kind === 'pending' || kind === 'zero' || kind === 'notInEda') {
    parts.push(
      <rect
        key="r"
        x={0.5}
        y={0.5}
        width={W - 1}
        height={H - 1}
        rx={Math.max(0, r - 0.5)}
        style={mark}
        vectorEffect="non-scaling-stroke"
      />
    );
    if (kind === 'zero') {
      const half = Math.min(3, (W - 4) / 2);
      parts.push(
        <line
          key="z"
          x1={W / 2 - half}
          x2={W / 2 + half}
          y1={H / 2}
          y2={H / 2}
          style={mark}
          vectorEffect="non-scaling-stroke"
        />
      );
    }
    if (kind === 'notInEda') {
      parts.push(<line key="s" x1={2} y1={2} x2={W - 2} y2={H - 2} style={mark} vectorEffect="non-scaling-stroke" />);
    }
    if (dogEar) parts.push(<path key="d" d={`M${W - 3.5} 0H${W}V3.5Z`} style={{fill: 'var(--loom-mark)'}} />);
  } else if (kind === 'dot') {
    parts.push(<circle key="c" cx={W / 2} cy={H / 2} r={1.5} style={{fill: 'var(--loom-mark)'}} />);
  }
  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      className={`inline-block shrink-0 overflow-visible align-middle ${className}`}
      aria-hidden="true"
      focusable="false"
    >
      {guest ? <g transform={`translate(${W * 0.2} ${H * 0.2}) scale(0.6)`}>{parts}</g> : parts}
    </svg>
  );
}

// Tier mark in neutral ink, same grammar as the cells: ■ profiled, □
// dictionary only, ▨ no dictionary.
function TierGlyph({
  tier,
  width = 9,
  height = 9,
  className = ''
}: {
  tier: Tier;
  width?: number;
  height?: number;
  className?: string;
}) {
  const kind: GlyphKind = tier === 'profiled' ? 'presence' : tier === 'dictionary' ? 'ring' : 'fog';
  return <CellGlyph kind={kind} width={width} height={height} className={className} />;
}

// 14px domain monogram in a hairline box (row headers use the same mark).
function DomainMark({domain, size = 14, className = ''}: {domain: Domain; size?: number; className?: string}) {
  return (
    <span
      className={`inline-flex shrink-0 items-center justify-center rounded-[3px] border font-semibold leading-none text-base-content/70 ${className}`}
      style={{borderColor: 'var(--loom-hairline)', width: size, height: size, fontSize: Math.round(size * 0.64)}}
      title={DOMAIN_LABEL[domain]}
      aria-hidden="true"
    >
      {DOMAIN_MONOGRAM[domain]}
    </span>
  );
}

// Set-pin marks: ✓ coded, ✓✓ with data, ✕ not coded, ○ none.
const PIN_MARK = {coded: '✓', data: '✓✓', not: '✕', none: '○'} as const;

// ============================================================================
// CohortList
// ============================================================================

// Every registered cohort, searchable: tier glyph, name, size, and per-row
// hide / only / pin. A cohort the current filters remove stays listed
// (dimmed), so the list is also the answer to "where did X go?".

const PIN_TITLE: Record<PinKind | 'none', string> = {
  none: 'No pin. Click to pin ✓ (coded)',
  coded: '✓ coded: concepts must be coded in this cohort. Click for the next pin state',
  data: '✓✓ with data: concepts must be measured here (n > 0, above the usable floor). Click for ✕',
  not: '✕ not coded: concepts must NOT be coded here. Click to clear'
};

function sizeText(c: LoomCohort): {text: string; title: string} {
  if (c.nRows !== null) return {text: fmtCompact(c.nRows), title: `${fmtInt(c.nRows)} dataset rows (EDA)`};
  if (c.declared !== null)
    return {text: `~${fmtCompact(c.declared)}`, title: `~${fmtInt(c.declared)} declared participants`};
  return {text: '?', title: 'Size unknown'};
}

function CohortList() {
  const {model, state, dispatch, columns, setUi, revealCohort} = useLoom();
  const [query, setQuery] = useState('');

  const fog = useMemo(() => new Set(columns.visibleFog), [columns.visibleFog]);
  const pins = useMemo(() => new Map(state.pins.map(p => [p.cohortId, p.kind])), [state.pins]);
  const hidden = useMemo(() => new Set(state.hiddenCohorts), [state.hiddenCohorts]);
  const only = useMemo(() => new Set(state.onlyCohorts), [state.onlyCohorts]);

  const q = query.trim().toLowerCase();
  const list = model.cohorts.filter(
    c => !q || c.id.toLowerCase().includes(q) || c.institution.toLowerCase().includes(q)
  );
  const groups: {tier: Tier; cohorts: LoomCohort[]}[] = (['profiled', 'dictionary', 'none'] as Tier[])
    .map(tier => ({tier, cohorts: list.filter(c => c.tier === tier)}))
    .filter(g => g.cohorts.length > 0);

  const shown = (c: LoomCohort) => columns.visibleMask[c.i] === 1 || fog.has(c.i);

  const open = (c: LoomCohort, e: React.MouseEvent) => {
    if (e.altKey) {
      dispatch({type: 'soloCohort', cohortId: c.id});
      return;
    }
    setUi({inspector: {kind: 'cohort', cohort: c.i}});
    if (shown(c)) revealCohort(c.i);
  };

  // Width is set per button (never two width utilities on one element).
  const iconButton =
    'flex h-5 shrink-0 items-center justify-center rounded text-[10px] leading-none hover:bg-base-content/[0.08] hover:text-base-content disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-transparent';

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex h-7 items-center rounded-md border border-base-content/15 px-1.5 focus-within:border-base-content/40">
        <SearchIcon size={12} className="shrink-0 text-base-content/40" aria-hidden="true" />
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Escape' && query) {
              e.stopPropagation();
              setQuery('');
            }
          }}
          placeholder={`Find among ${model.cohorts.length} cohorts`}
          aria-label="Find a cohort"
          title="Filter this list by cohort name or institution (the matrix is not affected)"
          className="min-w-0 flex-1 bg-transparent px-1.5 text-[12px] outline-none placeholder:text-base-content/40"
        />
        {query && (
          <button
            type="button"
            className={`${iconButton} w-5`}
            title="Clear"
            aria-label="Clear the cohort search"
            onClick={() => setQuery('')}
          >
            <X size={11} />
          </button>
        )}
      </div>
      {(state.hiddenCohorts.length > 0 || state.onlyCohorts.length > 0) && (
        <div className="flex flex-wrap gap-1 text-[10.5px]">
          {state.onlyCohorts.length > 0 && (
            <button
              type="button"
              className="rounded border border-base-content/20 px-1.5 py-[1px] text-base-content/70 hover:border-base-content/40 hover:text-base-content"
              title={`Showing only ${state.onlyCohorts.join(', ')}. Click to show every cohort again.`}
              onClick={() => dispatch({type: 'patch', patch: {onlyCohorts: []}})}
            >
              only {state.onlyCohorts.length} ×
            </button>
          )}
          {state.hiddenCohorts.length > 0 && (
            <button
              type="button"
              className="rounded border border-base-content/20 px-1.5 py-[1px] text-base-content/70 hover:border-base-content/40 hover:text-base-content"
              title={`Hidden: ${state.hiddenCohorts.join(', ')}. Click to show them again.`}
              onClick={() => dispatch({type: 'patch', patch: {hiddenCohorts: []}})}
            >
              {state.hiddenCohorts.length} hidden ×
            </button>
          )}
        </div>
      )}
      <div className="max-h-[320px] overflow-y-auto pr-0.5" role="list" aria-label="Cohorts">
        {groups.map(g => (
          <div key={g.tier} role="presentation">
            <div className="sticky top-0 z-[1] flex items-center gap-1.5 bg-base-100 py-1 text-[10px] font-semibold uppercase tracking-[0.09em] text-base-content/45">
              <TierGlyph tier={g.tier} width={8} height={8} />
              {TIER_LABEL[g.tier]} · {g.cohorts.length}
            </div>
            {g.cohorts.map(c => {
              const pin = pins.get(c.id) ?? null;
              const isHidden = hidden.has(c.id);
              const isOnly = only.has(c.id);
              const size = sizeText(c);
              const visible = shown(c);
              return (
                <div
                  key={c.id}
                  role="listitem"
                  className={`group flex h-[22px] items-center gap-1 rounded px-0.5 hover:bg-base-content/[0.04] ${visible ? '' : 'opacity-45'}`}
                >
                  <TierGlyph tier={c.tier} width={8} height={8} className="mx-0.5" />
                  <button
                    type="button"
                    className={`min-w-0 flex-1 truncate text-left text-[12px] ${isHidden ? 'line-through decoration-base-content/40' : ''} ${
                      isOnly ? 'font-semibold' : ''
                    }`}
                    title={`${c.id}${c.institution ? ` · ${c.institution}` : ''} · ${TIER_LABEL[c.tier]}${
                      visible ? '' : ' · not shown with the current filters'
                    }\nClick: open in the inspector${visible ? ' and scroll to its column' : ''} · Alt-click: show only this cohort`}
                    onClick={e => open(c, e)}
                  >
                    {c.id}
                  </button>
                  <span className="loom-tabular shrink-0 text-[10.5px] text-base-content/45" title={size.title}>
                    {size.text}
                  </span>
                  <span
                    className={`flex shrink-0 items-center ${isHidden || isOnly || pin ? '' : 'opacity-0 group-focus-within:opacity-100 group-hover:opacity-100'}`}
                  >
                    <button
                      type="button"
                      className={`${iconButton} w-5 ${isHidden ? 'text-base-content' : 'text-base-content/50'}`}
                      aria-pressed={isHidden}
                      aria-label={`Hide ${c.id}`}
                      title={isHidden ? `Show ${c.id} again` : `Hide ${c.id} (its column leaves the matrix)`}
                      onClick={() => dispatch({type: 'toggleHiddenCohort', cohortId: c.id})}
                    >
                      {isHidden ? <EyeOff size={11} /> : <Eye size={11} />}
                    </button>
                    <button
                      type="button"
                      className={`${iconButton} px-1 text-[9.5px] font-semibold uppercase tracking-wide ${
                        isOnly ? 'bg-base-content/[0.1] text-base-content' : 'text-base-content/50'
                      }`}
                      aria-pressed={isOnly}
                      aria-label={`Only ${c.id}`}
                      title={isOnly ? 'Show every cohort again' : `Show only ${c.id} (alt-click a name does the same)`}
                      onClick={() => dispatch({type: 'soloCohort', cohortId: c.id})}
                    >
                      only
                    </button>
                    <button
                      type="button"
                      className={`${iconButton} w-6 font-semibold ${pin ? 'text-base-content' : 'text-base-content/40'}`}
                      disabled={c.tier === 'none'}
                      aria-label={`Pin ${c.id}: ${pin ?? 'none'}`}
                      title={
                        c.tier === 'none'
                          ? 'Nothing to pin: no dictionary'
                          : `${PIN_TITLE[pin ?? 'none']}${c.tier === 'profiled' ? '' : ' (✓✓ needs a profiled cohort)'}. Shift-click clears.`
                      }
                      onClick={e =>
                        e.shiftKey
                          ? dispatch({type: 'setPin', cohortId: c.id, kind: null})
                          : dispatch({type: 'cyclePin', cohortId: c.id, profiled: c.tier === 'profiled'})
                      }
                    >
                      {PIN_MARK[pin ?? 'none']}
                    </button>
                  </span>
                </div>
              );
            })}
          </div>
        ))}
        {groups.length === 0 && <p className="py-2 text-[11px] text-base-content/50">No cohort matches “{query}”.</p>}
      </div>
    </div>
  );
}

// ============================================================================
// drawCells
// ============================================================================

// The cell grammar of the Loom (spec §4.3), drawn on a canvas in DEVICE
// pixels so every ring and gap lands on the pixel grid at any devicePixelRatio:
// the outline says the dictionary lists it; what is inside says what
// profiling found. Pure drawing, no React: paintGlyph draws one glyph,
// GlyphAtlas caches them; CoverageGrid blits them.
//
// A glyph is identified by a small integer key (state, bin and the modifiers
// that change its drawing); the grid blits glyphs from a pre-rendered atlas,
// so a frame of ~2k cells is ~2k drawImage calls with no path work.

// Basket notch variants (Comfortable only).
const BASKET_SOME = 1;
const BASKET_ALL = 2;

// Key layout: state (3 bits) | bin (3) | multi (1) | floor (1) | guest (1) | basket (2).
function glyphKey(state: number, bin: number, flags: number, density: Density): number {
  const counted = state === CellState.Counted;
  const multi = flags & CellFlag.Multi && state !== CellState.OtherVisit ? 1 : 0;
  const floor = counted && flags & CellFlag.BelowFloor ? 1 : 0;
  const guest = flags & CellFlag.GuestOnly ? 1 : 0;
  // The notch is a second micro-modifier: only Comfortable has room for two.
  const basket =
    density !== 'comfortable'
      ? 0
      : flags & CellFlag.InBasketAll
        ? BASKET_ALL
        : flags & CellFlag.InBasket
          ? BASKET_SOME
          : 0;
  return state | ((counted ? bin : 0) << 3) | (multi << 6) | (floor << 7) | (guest << 8) | (basket << 9);
}

// Whether a key draws anything at all (not coded and fog draw nothing: the
// surface, respectively the CSS hatch layer, carries the meaning).
const drawsGlyph = (state: number): boolean => state !== CellState.NotCoded && state !== CellState.Fog;

interface GlyphPaint {
  density: Density;
  spec: DensitySpec;
  dpr: number;
  tokens: LoomTokens;
}

// Rounded-rectangle path (own helper: CanvasRenderingContext2D.roundRect is
// missing from older Safari/Firefox builds still in use at partner sites).
function roundRectPath(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  if (rr === 0) {
    ctx.rect(x, y, w, h);
    return;
  }
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

// Draws one glyph into the w x h device-pixel box at (x, y). The box is
// transparent outside the glyph; "surface-coloured" cuts are punched out with
// destination-out so they show whatever surface is behind the canvas.
function paintGlyph(
  ctx: CanvasRenderingContext2D,
  key: number,
  x: number,
  y: number,
  w: number,
  h: number,
  p: GlyphPaint
): void {
  const state = key & 7;
  if (!drawsGlyph(state)) return;
  const bin = (key >> 3) & 7;
  const multi = (key >> 6) & 1;
  const floor = (key >> 7) & 1;
  const guest = (key >> 8) & 1;
  const basket = (key >> 9) & 3;
  const {dpr, tokens, density} = p;
  const overview = density === 'overview';
  const px = (css: number) => Math.max(1, Math.round(css * dpr));
  let r = p.spec.radius * dpr;

  // Guest (legacy pipe-list secondary code): the same glyph at 60%, centred.
  if (guest) {
    const gw = Math.max(1, Math.round(w * 0.6));
    const gh = Math.max(1, Math.round(h * 0.6));
    x += Math.floor((w - gw) / 2);
    y += Math.floor((h - gh) / 2);
    w = gw;
    h = gh;
    r *= 0.6;
  }

  const lw = px(1);
  ctx.save();

  const markCorner = () => {
    // Dog-ear on a hollow glyph: a mark-filled corner.
    ctx.fillStyle = tokens.mark;
    if (overview) {
      ctx.fillRect(x + w - px(2), y, px(2), px(1));
      return;
    }
    const d = px(3);
    ctx.beginPath();
    ctx.moveTo(x + w - d, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + d);
    ctx.closePath();
    ctx.fill();
  };

  if (state === CellState.Counted) {
    const colour = bin > 0 ? tokens.bins[bin - 1] : tokens.inkFill;
    ctx.fillStyle = colour;
    if (floor) {
      // Below the usable floor: the fill collapses to a bar along the bottom,
      // same hue so it never reads as a lower bin.
      const barH = Math.min(h, overview ? px(1) : px(3));
      roundRectPath(ctx, x, y + h - barH, w, barH, Math.min(r, barH / 2));
      ctx.fill();
      if (multi) markCorner();
    } else {
      roundRectPath(ctx, x, y, w, h, r);
      ctx.fill();
      if (multi) {
        // Surface-coloured corner cut on a fill.
        ctx.globalCompositeOperation = 'destination-out';
        ctx.fillStyle = '#000';
        if (overview) {
          ctx.fillRect(x + w - px(2), y, px(2), px(1));
        } else {
          const d = px(3);
          ctx.beginPath();
          ctx.moveTo(x + w - d, y);
          ctx.lineTo(x + w, y);
          ctx.lineTo(x + w, y + d);
          ctx.closePath();
          ctx.fill();
        }
        ctx.globalCompositeOperation = 'source-over';
      }
    }
  } else if (state === CellState.OtherVisit) {
    // Coded only at another visit: a small centred dot, no outline.
    ctx.fillStyle = tokens.mark;
    const d = overview ? px(1) : px(3);
    const cx = x + Math.floor((w - d) / 2);
    const cy = y + Math.floor((h - d) / 2);
    if (d <= 2) {
      ctx.fillRect(cx, cy, d, d);
    } else {
      ctx.beginPath();
      ctx.arc(cx + d / 2, cy + d / 2, d / 2, 0, Math.PI * 2);
      ctx.fill();
    }
  } else if (overview) {
    // Overview merges every hollow state into a 1px line through the middle.
    ctx.fillStyle = tokens.mark;
    ctx.fillRect(x, y + Math.floor((h - lw) / 2), w, lw);
    if (multi) markCorner();
  } else {
    // Hollow states: 1px ring (dictionary only, pending), plus a dash (zero)
    // or a single "\" stroke (not in the profiled data) - the opposite
    // direction of the "/" fog hatch, so the two never read alike.
    ctx.strokeStyle = tokens.mark;
    ctx.lineWidth = lw;
    roundRectPath(ctx, x + lw / 2, y + lw / 2, w - lw, h - lw, Math.max(0, r - lw / 2));
    ctx.stroke();
    if (state === CellState.Zero) {
      const len = Math.min(w - 2 * lw - px(2), px(density === 'comfortable' ? 8 : 6));
      ctx.fillStyle = tokens.mark;
      ctx.fillRect(x + Math.round((w - len) / 2), y + Math.floor((h - lw) / 2), len, lw);
    } else if (state === CellState.NotInEda) {
      ctx.beginPath();
      ctx.moveTo(x + lw, y + lw);
      ctx.lineTo(x + w - lw, y + h - lw);
      ctx.lineCap = 'butt';
      ctx.stroke();
    }
    if (multi) markCorner();
  }

  if (basket) {
    // In the DCR basket: a 4x4 ink notch bottom-left with a 1px surface ring;
    // hollow when only some of the cell's variables are in the basket.
    const s = px(4);
    const ring = px(1);
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = '#000';
    ctx.fillRect(x, y + h - s - ring, s + ring, s + ring);
    ctx.globalCompositeOperation = 'source-over';
    if (basket === BASKET_ALL) {
      ctx.fillStyle = tokens.ink;
      ctx.fillRect(x, y + h - s, s, s);
    } else {
      ctx.strokeStyle = tokens.ink;
      ctx.lineWidth = lw;
      ctx.strokeRect(x + lw / 2, y + h - s + lw / 2, s - lw, s - lw);
    }
  }
  ctx.restore();
}

// Pre-rendered glyphs for one (tokens, density, dpr) combination. Glyphs are
// rendered lazily on first use into fixed slots of one canvas and blitted
// 1:1 (no scaling, integer device coordinates: crisp).
class GlyphAtlas {
  readonly gw: number;
  readonly gh: number;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly slots = new Map<number, number>();
  private readonly perRow = 32;
  private readonly rows = 24;
  private readonly pad = 2;
  private next = 0;

  constructor(readonly paint: GlyphPaint) {
    this.gw = Math.max(1, Math.round(paint.spec.cellW * paint.dpr));
    this.gh = Math.max(1, Math.round(paint.spec.cellH * paint.dpr));
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.perRow * (this.gw + this.pad);
    this.canvas.height = this.rows * (this.gh + this.pad);
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Loom: 2D canvas unavailable');
    this.ctx = ctx;
  }

  matches(p: GlyphPaint): boolean {
    return p.tokens === this.paint.tokens && p.density === this.paint.density && p.dpr === this.paint.dpr;
  }

  private slotOf(key: number): number {
    let slot = this.slots.get(key);
    if (slot !== undefined) return slot;
    if (this.next >= this.perRow * this.rows) {
      // Full (never in practice: ~200 distinct glyphs exist): start over.
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.slots.clear();
      this.next = 0;
    }
    slot = this.next++;
    this.slots.set(key, slot);
    const {sx, sy} = this.origin(slot);
    paintGlyph(this.ctx, key, sx, sy, this.gw, this.gh, this.paint);
    return slot;
  }

  private origin(slot: number): {sx: number; sy: number} {
    return {sx: (slot % this.perRow) * (this.gw + this.pad), sy: Math.floor(slot / this.perRow) * (this.gh + this.pad)};
  }

  blit(target: CanvasRenderingContext2D, key: number, dx: number, dy: number): void {
    const {sx, sy} = this.origin(this.slotOf(key));
    target.drawImage(this.canvas, sx, sy, this.gw, this.gh, dx, dy, this.gw, this.gh);
  }
}

// ---------------------------------------------------------------------------
// Legend-as-filter (ui.legendDim): the cells it names stay, the rest dim to 20%.
// ---------------------------------------------------------------------------

type DimRule = {kind: 'bin'; bin: number} | {kind: 'state'; state: number} | {kind: 'flag'; mask: number} | null;

const DIM_FLAGS: Record<string, number> = {
  multi: CellFlag.Multi,
  floor: CellFlag.BelowFloor,
  guest: CellFlag.GuestOnly,
  basket: CellFlag.InBasket | CellFlag.InBasketAll,
  broad: CellFlag.Broad
};

// 'bin:<0..5>' | 'state:<CellState>' | 'flag:multi|floor|guest|basket|broad'.
function parseDim(raw: string | null): DimRule {
  if (!raw) return null;
  const [kind, value] = raw.split(':');
  if (kind === 'bin' && /^\d+$/.test(value)) return {kind: 'bin', bin: Number(value)};
  if (kind === 'state' && /^\d+$/.test(value)) return {kind: 'state', state: Number(value)};
  if (kind === 'flag' && DIM_FLAGS[value]) return {kind: 'flag', mask: DIM_FLAGS[value]};
  return null;
}

const dimKeeps = (d: NonNullable<DimRule>, state: number, bin: number, flags: number): boolean =>
  d.kind === 'bin'
    ? state === CellState.Counted && bin === d.bin
    : d.kind === 'state'
      ? state === d.state
      : (flags & d.mask) !== 0;

// ---------------------------------------------------------------------------
// A frame of the matrix body: cells, band hairlines, tallies, the shuttle,
// and the COVER / POOLED marginals.
// ---------------------------------------------------------------------------

// ============================================================================
// CoverageGrid
// ============================================================================

// The matrix. One cell = one concept in one cohort at one visit, drawn with
// the cell grammar (drawCells) at a size large enough to print its value.
// By default the rows are cohort visits (a cohort with six visits is six
// rows) and the columns are concepts; state.transpose swaps them. Headers
// and cells are purely visual layers over one scroll container, which does
// all hit testing, so scrolling, hover and clicks share one coordinate system.

interface CellSize {
  w: number;
  h: number;
  gap: number;
  radius: number;
  text: boolean;
}

const CELL_SIZE: Record<Density, CellSize> = {
  overview: {w: 10, h: 8, gap: 2, radius: 1, text: false},
  compact: {w: 22, h: 16, gap: 3, radius: 2, text: false},
  comfortable: {w: 46, h: 22, gap: 4, radius: 3, text: true}
};
const AXIS_GROUP_GAP = 8;
const HEADER_H = 168;
const ROW_HEAD_W: Record<'concept' | 'visit', number> = {concept: 300, visit: 236};
const GRID_OVERSCAN = 6;

export interface GridApi {
  revealRow(row: number): void;
  revealCohort(cohort: number): void;
  focus(): void;
}

interface AxisGeometry {
  pos: Float64Array;
  total: number;
  pitch: number;
  size: number;
}

function axisGeometry(items: AxisItem[], size: number, gap: number): AxisGeometry {
  const pos = new Float64Array(items.length);
  let p = 0;
  for (let i = 0; i < items.length; i++) {
    if (i > 0 && items[i].group !== items[i - 1].group) p += AXIS_GROUP_GAP;
    pos[i] = p;
    p += size + gap;
  }
  return {pos, total: p, pitch: size + gap, size};
}

// Index of the item whose pitch contains v, -1 in a group gap or outside.
function axisIndexAt(g: AxisGeometry, v: number): number {
  let lo = 0;
  let hi = g.pos.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (g.pos[mid] <= v) {
      found = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return found >= 0 && v < g.pos[found] + g.pitch ? found : -1;
}

// First index whose item ends after v (for visible ranges).
function axisFirstAfter(g: AxisGeometry, v: number): number {
  let lo = 0;
  let hi = g.pos.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (g.pos[mid] + g.pitch <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

const singleSlot = (slot: VisitSlot): VisitLens => ({kind: 'slots', mask: SLOT_BIT[slot]});

// Readable text on a ramp colour: dark ink on light steps, white on dark ones.
function inkOn(colour: string): string {
  const m = colour.match(/^#([0-9a-f]{6})$/i);
  if (!m) return '#ffffff';
  const n = parseInt(m[1], 16);
  const lin = (c: number) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  };
  const L = 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
  return L > 0.3 ? '#0b1b33' : '#ffffff';
}

function cellValueText(metric: Metric, lo: number, N: number | null): string {
  if (metric === 'n' || N === null || N === 0) return fmtCompact(lo);
  const pct = (100 * lo) / N;
  return pct >= 99.5 ? '100%' : pct < 1 ? '<1%' : `${Math.round(pct)}%`;
}

type GridHit =
  | {zone: 'cell'; i: number; j: number}
  | {zone: 'rowHead'; i: number}
  | {zone: 'colHead'; j: number}
  | null;

function CoverageGrid({apiRef}: {apiRef: React.MutableRefObject<GridApi | null>}) {
  const loom = useLoom();
  const {model, axes, filter, state, dispatch, ui, setUi, pointer, theme, stats} = loom;
  const size = CELL_SIZE[state.density];
  const metric: Metric = model.countsStatus === 'ok' ? state.metric : 'pres';
  const rowKind = axes.rows.length && axes.rows[0].kind === 'concept' ? 'concept' : 'visit';
  const headW = ROW_HEAD_W[rowKind];

  const rootRef = useRef<HTMLDivElement>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const colInnerRef = useRef<HTMLDivElement>(null);
  const rowInnerRef = useRef<HTMLDivElement>(null);
  const rowWashRef = useRef<HTMLDivElement>(null);
  const colWashRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const tipTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const frame = useRef(0);
  const scroll = useRef({top: 0, left: 0});
  const hover = useRef<GridHit>(null);
  const focusCell = useRef<{i: number; j: number} | null>(null);
  const atlasRef = useRef<GlyphAtlas | null>(null);

  const [view, setView] = useState({w: 0, h: 0});
  const [tokens, setTokens] = useState<LoomTokens | null>(null);
  const [font, setFont] = useState('system-ui, sans-serif');
  const [range, setRange] = useState({r0: 0, r1: -1, c0: 0, c1: -1});
  const dpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1;

  const rowGeo = useMemo(() => axisGeometry(axes.rows, size.h, size.gap), [axes.rows, size]);
  const colGeo = useMemo(() => axisGeometry(axes.cols, size.w, size.gap), [axes.cols, size]);
  const bodyW = Math.max(0, view.w - headW);
  const bodyH = Math.max(0, view.h - HEADER_H);

  // Resolved cells are cached per (concept, cohort, visit) until an input of
  // the cell rule changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps -- a fresh cache whenever these change
  const cellCache = useMemo(() => new Map<string, ResolvedCell>(), [model, state.floor, state.guests]);

  const resolve = useCallback(
    (row: number, cohort: number, slot: VisitSlot): ResolvedCell => {
      const key = `${row}|${cohort}|${slot}`;
      let cell = cellCache.get(key);
      if (!cell) {
        cell = resolveCell(model, row, cohort, singleSlot(slot), state.floor, state.guests);
        cellCache.set(key, cell);
      }
      return cell;
    },
    [model, state.floor, state.guests, cellCache]
  );

  // (concept row, visit item) of a cell, whichever axis each runs along.
  const cellItems = useCallback(
    (i: number, j: number): {concept: number; visit: AxisItem} | null => {
      const a = axes.rows[i];
      const b = axes.cols[j];
      if (!a || !b) return null;
      if (a.kind === 'concept') return {concept: a.row, visit: b};
      if (b.kind === 'concept') return {concept: b.row, visit: a};
      return null;
    },
    [axes]
  );

  // ------------------------------------------------------------- theme, size
  const readTokens = useCallback(() => {
    const el = rootRef.current;
    if (!el) return;
    setTokens(readLoomTokens(el.closest('.loom-root') ?? el));
    setFont(getComputedStyle(el).fontFamily || 'system-ui, sans-serif');
  }, []);
  useLayoutEffect(readTokens, [readTokens, theme]);

  useLayoutEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const measure = () =>
      setView(prev => (prev.w === el.clientWidth && prev.h === el.clientHeight ? prev : {w: el.clientWidth, h: el.clientHeight}));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ------------------------------------------------------------------ paint
  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !tokens) return;
    const W = Math.round(bodyW * dpr);
    const H = Math.round(bodyH * dpr);
    if (canvas.width !== W) canvas.width = W;
    if (canvas.height !== H) canvas.height = H;
    const ctx = canvas.getContext('2d');
    if (!ctx || !W || !H) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const spec: DensitySpec = {cellW: size.w, cellH: size.h, pitchX: size.w + size.gap, pitchY: size.h + size.gap, radius: size.radius, gap: size.gap};
    const glyph = {density: state.density, spec, dpr, tokens};
    let atlas = atlasRef.current;
    if (!atlas || !atlas.matches(glyph) || atlas.gw !== Math.round(size.w * dpr) || atlas.gh !== Math.round(size.h * dpr))
      atlas = atlasRef.current = new GlyphAtlas(glyph);
    const {top, left} = scroll.current;
    const r0 = axisFirstAfter(rowGeo, top);
    const c0 = axisFirstAfter(colGeo, left);
    const dim = parseDim(ui.legendDim);
    const highlight = filter.highlight;
    const lw = Math.max(1, Math.round(dpr));

    // Cohorts without a dictionary: one continuous hatch per band ("unknown, not absent").
    const hatch = (x: number, y: number, w: number, h: number) => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      ctx.strokeStyle = tokens.fog;
      ctx.lineWidth = lw;
      const step = Math.round(6 * dpr);
      for (let d = -h; d < w + h; d += step) {
        ctx.beginPath();
        ctx.moveTo(x + d, y + h);
        ctx.lineTo(x + d + h, y);
        ctx.stroke();
      }
      ctx.restore();
    };
    for (let i = r0; i < axes.rows.length && rowGeo.pos[i] - top < bodyH; i++) {
      if (axes.rows[i].kind === 'fog')
        hatch(0, Math.round((rowGeo.pos[i] - top) * dpr), W, Math.round(size.h * dpr));
    }
    for (let j = c0; j < axes.cols.length && colGeo.pos[j] - left < bodyW; j++) {
      if (axes.cols[j].kind === 'fog')
        hatch(Math.round((colGeo.pos[j] - left) * dpr), 0, Math.round(size.w * dpr), H);
    }

    ctx.font = `600 ${Math.round(11 * dpr)}px ${font}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = r0; i < axes.rows.length; i++) {
      const y = rowGeo.pos[i] - top;
      if (y >= bodyH) break;
      for (let j = c0; j < axes.cols.length; j++) {
        const x = colGeo.pos[j] - left;
        if (x >= bodyW) break;
        const it = cellItems(i, j);
        if (!it || it.visit.kind !== 'visit') continue;
        const cell = resolve(it.concept, it.visit.cohort, it.visit.slot);
        // In this layout the row / column IS the visit: "coded at another
        // visit" is simply not coded here.
        const state_ = cell.state === CellState.OtherVisit ? CellState.NotCoded : cell.state;
        if (state_ === CellState.NotCoded) continue;
        const bin = state_ === CellState.Counted ? binFor(metric, cell.lo, cell.N) : 0;
        let alpha = highlight && !highlight[it.concept] ? 0.25 : 1;
        if (dim) {
          const keep =
            dim.kind === 'bin'
              ? state_ === CellState.Counted && bin === dim.bin
              : dim.kind === 'state'
                ? state_ === dim.state
                : (cell.flags & dim.mask) !== 0;
          if (!keep) alpha = Math.min(alpha, 0.2);
        }
        ctx.globalAlpha = alpha;
        const dx = Math.round(x * dpr);
        const dy = Math.round(y * dpr);
        atlas.blit(ctx, glyphKey(state_, bin, cell.flags, state.density), dx, dy);
        if (size.text && state_ === CellState.Counted && metric !== 'pres' && !(cell.flags & CellFlag.BelowFloor)) {
          ctx.fillStyle = inkOn(bin > 0 ? tokens.bins[bin - 1] : tokens.inkFill);
          ctx.fillText(cellValueText(metric, cell.lo, cell.N), dx + atlas.gw / 2, dy + atlas.gh / 2 + lw / 2);
        }
      }
    }
    ctx.globalAlpha = 1;
  }, [tokens, bodyW, bodyH, dpr, size, state.density, rowGeo, colGeo, axes, cellItems, resolve, metric, ui.legendDim, filter.highlight, font]);

  // Visible header range (with overscan so scrolling rarely re-renders).
  const syncRange = useCallback(() => {
    const {top, left} = scroll.current;
    const r0 = Math.max(0, axisFirstAfter(rowGeo, top) - GRID_OVERSCAN);
    const r1 = Math.min(axes.rows.length - 1, axisFirstAfter(rowGeo, top + bodyH) + GRID_OVERSCAN);
    const c0 = Math.max(0, axisFirstAfter(colGeo, left) - GRID_OVERSCAN);
    const c1 = Math.min(axes.cols.length - 1, axisFirstAfter(colGeo, left + bodyW) + GRID_OVERSCAN);
    setRange(prev =>
      prev.r0 <= r0 + GRID_OVERSCAN / 2 &&
      prev.r0 >= r0 - GRID_OVERSCAN &&
      prev.r1 >= r1 - GRID_OVERSCAN / 2 &&
      prev.r1 <= r1 + GRID_OVERSCAN &&
      prev.c0 <= c0 + GRID_OVERSCAN / 2 &&
      prev.c0 >= c0 - GRID_OVERSCAN &&
      prev.c1 >= c1 - GRID_OVERSCAN / 2 &&
      prev.c1 <= c1 + GRID_OVERSCAN
        ? prev
        : {r0, r1, c0, c1}
    );
  }, [rowGeo, colGeo, axes, bodyH, bodyW]);

  const placeOverlays = useCallback(() => {
    const {top, left} = scroll.current;
    if (colInnerRef.current) colInnerRef.current.style.transform = `translate3d(${-left}px,0,0)`;
    if (rowInnerRef.current) rowInnerRef.current.style.transform = `translate3d(0,${-top}px,0)`;
    const h = hover.current;
    const f = focusCell.current;
    const at = h && h.zone === 'cell' ? {i: h.i, j: h.j} : f;
    const iRow = h ? (h.zone === 'cell' || h.zone === 'rowHead' ? h.i : -1) : f ? f.i : -1;
    const jCol = h ? (h.zone === 'cell' || h.zone === 'colHead' ? h.j : -1) : f ? f.j : -1;
    const show = (el: HTMLDivElement | null, on: boolean, css: Partial<CSSStyleDeclaration>) => {
      if (!el) return;
      el.style.display = on ? 'block' : 'none';
      if (on) Object.assign(el.style, css);
    };
    show(rowWashRef.current, iRow >= 0, {top: `${HEADER_H + rowGeo.pos[iRow] - top}px`, height: `${size.h}px`});
    show(colWashRef.current, jCol >= 0, {left: `${headW + colGeo.pos[jCol] - left}px`, width: `${size.w}px`});
    show(ringRef.current, !!at, at ? {left: `${headW + colGeo.pos[at.j] - left - 2}px`, top: `${HEADER_H + rowGeo.pos[at.i] - top - 2}px`, width: `${size.w + 4}px`, height: `${size.h + 4}px`} : {});
  }, [rowGeo, colGeo, size, headW]);

  const schedule = useCallback(() => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      const el = scrollerRef.current;
      if (el) scroll.current = {top: el.scrollTop, left: el.scrollLeft};
      syncRange();
      placeOverlays();
      paint();
    });
  }, [syncRange, placeOverlays, paint]);

  useEffect(() => {
    schedule();
  });
  useEffect(
    () => () => {
      cancelAnimationFrame(frame.current);
      frame.current = 0;
      if (tipTimer.current) clearTimeout(tipTimer.current);
    },
    []
  );

  // ------------------------------------------------------------- hit testing
  const hitAt = useCallback(
    (clientX: number, clientY: number): GridHit => {
      const el = scrollerRef.current;
      if (!el) return null;
      const r = el.getBoundingClientRect();
      const x = clientX - r.left;
      const y = clientY - r.top;
      const {top, left} = scroll.current;
      if (x < headW && y >= HEADER_H) {
        const i = axisIndexAt(rowGeo, y - HEADER_H + top);
        return i >= 0 ? {zone: 'rowHead', i} : null;
      }
      if (y < HEADER_H && x >= headW) {
        const j = axisIndexAt(colGeo, x - headW + left);
        return j >= 0 ? {zone: 'colHead', j} : null;
      }
      if (x >= headW && y >= HEADER_H) {
        const i = axisIndexAt(rowGeo, y - HEADER_H + top);
        const j = axisIndexAt(colGeo, x - headW + left);
        return i >= 0 && j >= 0 ? {zone: 'cell', i, j} : null;
      }
      return null;
    },
    [rowGeo, colGeo, headW]
  );

  const itemTarget = useCallback(
    (item: AxisItem, source: 'mouse' | 'keyboard'): PointerTarget | null =>
      item.kind === 'concept'
        ? {kind: 'row', row: item.row, cohort: -1, item: -1, source}
        : item.kind === 'visit'
          ? {kind: 'column', row: -1, cohort: item.cohort, item: -1, source}
          : item.cohorts.length === 1
            ? {kind: 'column', row: -1, cohort: item.cohorts[0], item: -1, source}
            : null,
    []
  );

  const targetOf = useCallback(
    (h: GridHit, source: 'mouse' | 'keyboard'): PointerTarget | null => {
      if (!h) return null;
      if (h.zone === 'rowHead') return itemTarget(axes.rows[h.i], source);
      if (h.zone === 'colHead') return itemTarget(axes.cols[h.j], source);
      const it = cellItems(h.i, h.j);
      if (!it) return null;
      if (it.visit.kind !== 'visit') return itemTarget(it.visit, source);
      return {kind: 'cell', row: it.concept, cohort: it.visit.cohort, item: -1, slot: it.visit.slot, source};
    },
    [axes, cellItems, itemTarget]
  );

  // Tooltip text (value first), built with textContent: labels are data.
  const tooltipLines = useCallback(
    (h: GridHit): {head: string; body: string[]; foot: string[]} | null => {
      if (!h) return null;
      const describe = (item: AxisItem) => {
        if (item.kind === 'concept') {
          const r = rowReadout(model, stats, item.row);
          return {head: r.headline, body: r.lines, foot: ['Click for the concept inspector']};
        }
        if (item.kind === 'visit') {
          const r = cohortReadout(model, item.cohort);
          return {head: `${r.headline} · ${SLOT_LONG[item.slot]}`, body: r.lines, foot: ['Click for the cohort inspector']};
        }
        return {
          head: `${fmtInt(item.cohorts.length)} ${item.cohorts.length === 1 ? 'cohort' : 'cohorts'} without a dictionary`,
          body: [item.cohorts.map(c => model.cohorts[c].id).join(', '), CAVEATS.fog],
          foot: item.cohorts.length > 1 ? ['Click to show one row per cohort'] : []
        };
      };
      if (h.zone === 'rowHead') return describe(axes.rows[h.i]);
      if (h.zone === 'colHead') return describe(axes.cols[h.j]);
      const it = cellItems(h.i, h.j);
      if (!it) return null;
      if (it.visit.kind !== 'visit') return describe(it.visit);
      const r = cellReadout(model, it.concept, it.visit.cohort, singleSlot(it.visit.slot), metric, state.floor, state.guests);
      return {head: r.headline, body: [r.context, ...r.lines], foot: r.footer};
    },
    [axes, cellItems, model, stats, metric, state.floor, state.guests]
  );

  const hideTip = useCallback(() => {
    if (tipTimer.current) clearTimeout(tipTimer.current);
    tipTimer.current = null;
    if (tipRef.current) tipRef.current.style.display = 'none';
  }, []);

  const showTip = useCallback(
    (h: GridHit, clientX: number, clientY: number) => {
      const el = tipRef.current;
      const content = tooltipLines(h);
      if (!el || !content) return hideTip();
      el.replaceChildren();
      const head = document.createElement('div');
      head.className = 'font-semibold text-[13px] loom-tabular';
      head.textContent = content.head;
      el.appendChild(head);
      for (const line of content.body.filter(Boolean)) {
        const d = document.createElement('div');
        d.className = 'mt-1 text-[12px] text-base-content/80';
        d.textContent = line;
        el.appendChild(d);
      }
      for (const line of content.foot.filter(Boolean)) {
        const d = document.createElement('div');
        d.className = 'mt-1.5 text-[11px] text-base-content/55';
        d.textContent = line;
        el.appendChild(d);
      }
      el.style.display = 'block';
      const {width, height} = el.getBoundingClientRect();
      const x = clientX + 14 + width > window.innerWidth ? clientX - 14 - width : clientX + 14;
      const y = clientY + 14 + height > window.innerHeight ? clientY - 14 - height : clientY + 14;
      el.style.left = `${Math.max(4, x)}px`;
      el.style.top = `${Math.max(4, y)}px`;
    },
    [tooltipLines, hideTip]
  );

  const onMove = (e: React.PointerEvent) => {
    const h = hitAt(e.clientX, e.clientY);
    const prev = hover.current;
    const same =
      (h === null && prev === null) ||
      (!!h && !!prev && h.zone === prev.zone && ('i' in h ? h.i : -1) === ('i' in prev ? prev.i : -1) && ('j' in h ? h.j : -1) === ('j' in prev ? prev.j : -1));
    if (same) return;
    hover.current = h;
    pointer.setHover(targetOf(h, 'mouse'));
    placeOverlays();
    hideTip();
    if (h) {
      const {clientX, clientY} = e;
      tipTimer.current = setTimeout(() => showTip(h, clientX, clientY), 80);
    }
    if (scrollerRef.current) scrollerRef.current.style.cursor = h ? 'pointer' : '';
  };

  const onLeave = () => {
    hover.current = null;
    pointer.setHover(null);
    hideTip();
    placeOverlays();
  };

  const activate = (h: GridHit) => {
    if (!h) return;
    const item = h.zone === 'rowHead' ? axes.rows[h.i] : h.zone === 'colHead' ? axes.cols[h.j] : null;
    if (item) {
      if (item.kind === 'concept') setUi({inspector: {kind: 'concept', row: item.row}});
      else if (item.kind === 'visit') setUi({inspector: {kind: 'cohort', cohort: item.cohort}});
      else if (item.cohorts.length > 1) dispatch({type: 'patch', patch: {fog: 'cols'}});
      else setUi({inspector: {kind: 'cohort', cohort: item.cohorts[0]}});
      return;
    }
    if (h.zone !== 'cell') return;
    const it = cellItems(h.i, h.j);
    if (!it) return;
    if (it.visit.kind === 'visit') setUi({inspector: {kind: 'cell', row: it.concept, cohort: it.visit.cohort}});
    else activate(h.zone === 'cell' && axes.rows[h.i].kind === 'fog' ? {zone: 'rowHead', i: h.i} : {zone: 'colHead', j: h.j});
  };

  const onClick = (e: React.MouseEvent) => {
    const h = hitAt(e.clientX, e.clientY);
    if (h && h.zone === 'cell') focusCell.current = {i: h.i, j: h.j};
    hideTip();
    activate(h);
  };

  // ------------------------------------------------------------- scrolling
  const scrollToCell = useCallback(
    (i: number | null, j: number | null, smooth: boolean) => {
      const el = scrollerRef.current;
      if (!el) return;
      let {top, left} = scroll.current;
      if (i !== null && i >= 0) {
        const y = rowGeo.pos[i];
        if (y < top || y + size.h > top + bodyH) top = Math.max(0, y - bodyH / 3);
      }
      if (j !== null && j >= 0) {
        const x = colGeo.pos[j];
        if (x < left || x + size.w > left + bodyW) left = Math.max(0, x - bodyW / 3);
      }
      el.scrollTo({top, left, behavior: smooth ? 'smooth' : 'auto'});
    },
    [rowGeo, colGeo, size, bodyH, bodyW]
  );

  useImperativeHandle(
    apiRef,
    () => ({
      revealRow: (row: number) => {
        const k = axes.concepts.findIndex(a => a.kind === 'concept' && a.row === row);
        if (k < 0) return;
        if (state.transpose) scrollToCell(k, null, true);
        else scrollToCell(null, k, true);
      },
      revealCohort: (cohort: number) => {
        const k = axes.visits.findIndex(a => (a.kind === 'visit' && a.cohort === cohort) || (a.kind === 'fog' && a.cohorts.includes(cohort)));
        if (k < 0) return;
        if (state.transpose) scrollToCell(null, k, true);
        else scrollToCell(k, null, true);
      },
      focus: () => scrollerRef.current?.focus()
    }),
    [axes, state.transpose, scrollToCell]
  );

  // ------------------------------------------------------------- keyboard
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const f = focusCell.current ?? {i: 0, j: 0};
    const move: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
      PageUp: [-20, 0],
      PageDown: [20, 0]
    };
    if (e.key in move) {
      e.preventDefault();
      const [di, dj] = move[e.key];
      const i = Math.max(0, Math.min(axes.rows.length - 1, f.i + di));
      const j = Math.max(0, Math.min(axes.cols.length - 1, f.j + dj));
      focusCell.current = {i, j};
      hover.current = null;
      pointer.setFocus(targetOf({zone: 'cell', i, j}, 'keyboard'));
      scrollToCell(i, j, false);
      placeOverlays();
    } else if (e.key === 'Enter' && focusCell.current) {
      e.preventDefault();
      activate({zone: 'cell', ...focusCell.current});
    } else if (e.key === 'Escape' && ui.inspector) {
      e.preventDefault();
      setUi({inspector: null});
    }
  };

  // ------------------------------------------------------------- headers
  const colItems = axes.cols.slice(range.c0, range.c1 + 1);
  const rowItems = axes.rows.slice(range.r0, range.r1 + 1);
  const conceptCount = (row: number) => model.rows[row].dictCoverage;

  const groupBrackets = useMemo(() => {
    // Column groups: a hairline bracket and a label over each run of one group.
    const out: {label: string; x: number; w: number}[] = [];
    let start = 0;
    for (let j = 1; j <= axes.cols.length; j++) {
      if (j === axes.cols.length || axes.cols[j].group !== axes.cols[start].group) {
        const a = axes.cols[start];
        const label =
          a.kind === 'concept'
            ? DOMAIN_LABEL[model.rows[a.row].domain]
            : a.kind === 'visit'
              ? model.cohorts[a.cohort].id
              : a.cohorts.length > 1
                ? 'No dictionary'
                : model.cohorts[a.cohorts[0]].id;
        out.push({label, x: colGeo.pos[start], w: colGeo.pos[j - 1] + size.w - colGeo.pos[start]});
        start = j;
      }
    }
    return out;
  }, [axes.cols, colGeo, size.w, model]);

  const colLabel = (a: AxisItem): {text: string; strong?: string} =>
    a.kind === 'concept'
      ? {text: clipText(model.rows[a.row].label, 34)}
      : a.kind === 'visit'
        ? a.first
          ? {strong: clipText(model.cohorts[a.cohort].id, 20), text: ` · ${SLOT_SHORT[a.slot]}`}
          : {text: SLOT_SHORT[a.slot]}
        : {text: a.cohorts.length > 1 ? `${a.cohorts.length} without a dictionary` : `${clipText(model.cohorts[a.cohorts[0]].id, 20)} · no dictionary`};

  const rowHeader = (a: AxisItem) => {
    if (a.kind === 'concept') {
      const r = model.rows[a.row];
      return (
        <div className="flex h-full items-center gap-2 pl-3 pr-2">
          <span className="inline-flex h-[15px] w-[15px] flex-shrink-0 items-center justify-center border border-[color:var(--loom-hairline)] text-[9px] text-base-content/60">
            {DOMAIN_MONOGRAM[r.domain]}
          </span>
          <span className="min-w-0 flex-1 truncate text-[12px] text-base-content">{r.label}</span>
          <span className="flex-shrink-0 text-[10px] text-base-content/45 loom-tabular" title="Cohorts coding this concept">
            {conceptCount(a.row)}
          </span>
        </div>
      );
    }
    if (a.kind === 'visit') {
      const c = model.cohorts[a.cohort];
      return (
        <div className={`flex h-full items-center gap-2 pl-3 pr-3 ${a.first ? '' : 'border-l-2 border-[color:var(--loom-hairline)] ml-3'}`}>
          {a.first ? (
            <>
              <CubeFaceTierGlyph tier={c.tier} />
              <span className="min-w-0 flex-1 truncate text-[12px] font-semibold text-base-content">{c.id}</span>
            </>
          ) : (
            <span className="flex-1" />
          )}
          <span className="flex-shrink-0 text-[11px] text-base-content/65">{SLOT_LONG[a.slot]}</span>
        </div>
      );
    }
    return (
      <div className="flex h-full items-center gap-2 pl-3 pr-3">
        <span className="loom-fog inline-block h-[12px] w-[12px] flex-shrink-0" />
        <span className="min-w-0 flex-1 truncate text-[12px] text-base-content/70">
          {a.cohorts.length > 1 ? `${a.cohorts.length} cohorts without a dictionary` : model.cohorts[a.cohorts[0]].id}
        </span>
        <span className="flex-shrink-0 text-[11px] italic text-base-content/50">unknown</span>
      </div>
    );
  };

  const empty = axes.rows.length === 0 || axes.cols.length === 0;
  const topChips = useMemo(() => [...filter.chips].sort((a, b) => b.cost - a.cost).slice(0, 3), [filter.chips]);

  return (
    <div ref={rootRef} className="loom-grid absolute inset-0 overflow-hidden">
      <div
        ref={scrollerRef}
        className="absolute inset-0 overflow-auto outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[color:var(--loom-ink)]"
        tabIndex={0}
        role="grid"
        aria-label="Concept coverage: one cell per concept, cohort and visit. Arrow keys move, Enter inspects."
        aria-rowcount={axes.rows.length}
        aria-colcount={axes.cols.length}
        onScroll={schedule}
        onPointerMove={onMove}
        onPointerLeave={onLeave}
        onClick={onClick}
        onKeyDown={onKeyDown}
      >
        <div style={{width: headW + colGeo.total + 24, height: HEADER_H + rowGeo.total + 24}} />
      </div>

      {/* Cells */}
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute"
        style={{left: headW, top: HEADER_H, width: bodyW, height: bodyH}}
        aria-hidden="true"
      />
      <div ref={rowWashRef} className="pointer-events-none absolute left-0 right-0 hidden" style={{background: 'var(--loom-wash)'}} />
      <div ref={colWashRef} className="pointer-events-none absolute bottom-0 top-0 hidden" style={{background: 'var(--loom-wash)'}} />
      <div
        ref={ringRef}
        className="pointer-events-none absolute hidden rounded-[4px]"
        style={{boxShadow: '0 0 0 2px var(--loom-ink), 0 0 0 3px oklch(var(--b1))'}}
      />

      {/* Column headers */}
      <div
        className="pointer-events-none absolute top-0 overflow-hidden border-b border-[color:var(--loom-hairline)] bg-base-100"
        style={{left: headW, width: bodyW, height: HEADER_H}}
        aria-hidden="true"
      >
        <div ref={colInnerRef} className="absolute left-0 top-0 will-change-transform" style={{width: colGeo.total, height: HEADER_H}}>
          {/* Names lean right at -55deg: clipped below the group labels. */}
          <div className="absolute inset-x-0 bottom-0 overflow-hidden" style={{top: 30}}>
            {colItems.map((a, k) => {
              const j = range.c0 + k;
              const label = colLabel(a);
              return (
                <div
                  key={j}
                  className={`absolute whitespace-nowrap text-[11px] leading-[14px] ${a.kind === 'fog' ? 'italic text-base-content/50' : 'text-base-content/75'}`}
                  style={{
                    left: colGeo.pos[j] + size.w / 2 - 4,
                    top: HEADER_H - 30 - 16,
                    transformOrigin: '0 50%',
                    transform: 'rotate(-55deg)'
                  }}
                >
                  {label.strong && <span className="font-semibold text-base-content">{label.strong}</span>}
                  {label.text}
                </div>
              );
            })}
          </div>
          {groupBrackets.map(g => (
            <div key={`${g.label}-${g.x}`} className="absolute top-1" style={{left: g.x, width: g.w}}>
              <div className="truncate text-[10px] font-semibold uppercase tracking-wide text-base-content/60">{g.label}</div>
              <div className="mt-0.5 h-[4px] border-x border-t border-[color:var(--loom-hairline)]" />
            </div>
          ))}
        </div>
      </div>

      {/* Row headers */}
      <div
        className="pointer-events-none absolute left-0 overflow-hidden border-r border-[color:var(--loom-hairline)] bg-base-100"
        style={{top: HEADER_H, width: headW, height: bodyH}}
        aria-hidden="true"
      >
        <div ref={rowInnerRef} className="absolute left-0 top-0 w-full will-change-transform" style={{height: rowGeo.total}}>
          {rowItems.map((a, k) => {
            const i = range.r0 + k;
            return (
              <div key={i} className="absolute left-0 right-0" style={{top: rowGeo.pos[i], height: size.h}}>
                {rowHeader(a)}
              </div>
            );
          })}
        </div>
      </div>

      {/* Corner */}
      <div
        className="pointer-events-none absolute left-0 top-0 flex flex-col justify-end border-b border-r border-[color:var(--loom-hairline)] bg-base-100 px-3 pb-3"
        style={{width: headW, height: HEADER_H}}
      >
        <div className="text-[17px] font-semibold leading-none loom-tabular">
          {fmtInt(axes.rows.length)} <span className="font-normal text-base-content/40">×</span> {fmtInt(axes.cols.length)}
        </div>
        <div className="mt-1 text-[11px] leading-tight text-base-content/60">
          {state.transpose ? 'concepts × cohort visits' : 'cohort visits × concepts'}
        </div>
        <div className="mt-1 text-[11px] leading-tight text-base-content/60">
          {METRIC_LABEL[metric]} · one cell = one variable at one visit
        </div>
      </div>

      {empty && (
        <div className="absolute inset-x-0 flex flex-col items-center gap-2 text-sm text-base-content/60" style={{top: HEADER_H + 40}}>
          <p>No concept passes every filter.</p>
          {topChips.map(chip => (
            <button
              key={chip.id}
              type="button"
              className="btn btn-xs btn-outline font-normal normal-case"
              title={`Remove this filter: ${fmtInt(chip.cost)} concepts come back`}
              onClick={() => dispatch(chip.clear)}
            >
              Remove “{chip.label}” · +{fmtInt(chip.cost)}
            </button>
          ))}
        </div>
      )}

      <div
        ref={tipRef}
        role="tooltip"
        className="pointer-events-none fixed z-[70] hidden max-w-[380px] rounded-md border border-base-300 bg-base-100 px-3 py-2 shadow-lg"
      />
    </div>
  );
}

const clipText = (s: string, n: number): string => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

// ============================================================================
// ViewBar
// ============================================================================

// The two controls that shape the matrix itself, always in view: how many
// cohorts a concept must be found in, and which way round the matrix runs.

const MIN_COHORTS_MAX = 10;

function ViewBar() {
  const {state, dispatch, axes, ui, setUi, filter} = useLoom();
  const k = Math.min(state.minCohorts, MIN_COHORTS_MAX);
  const setK = (v: number) => dispatch({type: 'patch', patch: {minCohorts: Math.max(1, Math.min(MIN_COHORTS_MAX, v))}});
  const otherFilters = filter.chips.filter(c => c.id !== 'min').length;
  return (
    <div className="flex flex-shrink-0 flex-wrap items-center gap-x-5 gap-y-2 border-b border-[color:var(--loom-hairline)] px-4 py-2 text-[12px]">
      <button
        type="button"
        className={`btn btn-sm gap-1.5 font-normal ${ui.railOpen ? 'btn-active' : 'btn-ghost border border-base-300'}`}
        onClick={() => setUi({railOpen: !ui.railOpen})}
        aria-pressed={ui.railOpen}
        title={ui.railOpen ? 'Hide the filters' : 'Show the filters (cohorts by type, status, name; concepts by domain, name, ...)'}
      >
        <Sliders size={14} /> Filters{otherFilters > 0 ? ` (${otherFilters})` : ''}
      </button>

      <label className="flex items-center gap-3" title="Show only concepts found in at least this many cohorts">
        <span className="whitespace-nowrap text-base-content/70">
          Concepts found in at least{' '}
          <span className="font-semibold text-base-content loom-tabular">
            {k >= MIN_COHORTS_MAX ? `${MIN_COHORTS_MAX}+` : k}
          </span>{' '}
          {k === 1 ? 'cohort' : 'cohorts'}
        </span>
        <input
          type="range"
          min={1}
          max={MIN_COHORTS_MAX}
          step={1}
          value={k}
          onChange={e => setK(Number(e.target.value))}
          className="range range-xs w-44"
          aria-label="Minimum number of cohorts a concept is found in"
        />
      </label>

      <button
        type="button"
        className={`btn btn-sm gap-1.5 font-normal ${state.transpose ? 'btn-active' : 'btn-ghost border border-base-300'}`}
        onClick={() => dispatch({type: 'patch', patch: {transpose: !state.transpose}})}
        aria-pressed={state.transpose}
        title="Swap rows and columns"
      >
        <RefreshCw size={14} className="rotate-45" /> Transpose
      </button>

      {k === 1 && (
        <div role="status" className="flex items-center gap-2 rounded-md border border-warning/60 bg-warning/10 px-3 py-1.5 text-base-content">
          <AlertTriangle size={14} className="flex-shrink-0 text-warning" />
          <span>
            Every concept is shown, including those found in only one cohort: the table will be very sparse, with{' '}
            <span className="font-semibold loom-tabular">{fmtInt(axes.rows.length)}</span> rows and{' '}
            <span className="font-semibold loom-tabular">{fmtInt(axes.cols.length)}</span> columns.
          </span>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// LayoutMenu
// ============================================================================

// The Layout menu of the command bar, plus the few control primitives the
// Loom chrome shares (popover, segmented control, option list, switch row,
// caps label) so every panel looks and behaves the same.

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

// Section title: small caps with tracking. Tone and size are props (not
// className) so a caller never stacks two colour or size utilities.
function Caps({
  children,
  className = '',
  tone = 'text-base-content/50',
  size = 'text-[10px]'
}: {
  children: React.ReactNode;
  className?: string;
  tone?: string;
  size?: string;
}) {
  return (
    <div className={`${size} ${tone} font-semibold uppercase leading-none tracking-[0.09em] ${className}`}>
      {children}
    </div>
  );
}

// Single-key shortcuts are ignored while the reader types (spec §8).
function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== 'string') return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

const FOCUSABLE =
  'button:not([disabled]),[href],input:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])';

// Keeps Tab inside a modal layer (drawer, sheet): wraps from the last
// focusable element to the first and back.
function trapTab(e: React.KeyboardEvent, container: HTMLElement | null) {
  if (e.key !== 'Tab' || !container) return;
  const items = Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(el => el.offsetParent !== null);
  if (!items.length) return;
  const first = items[0];
  const last = items[items.length - 1];
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}

interface PopoverProps {
  label: React.ReactNode;
  title: string; // tooltip of the trigger: what the menu does
  ariaLabel?: string;
  align?: 'left' | 'right';
  direction?: 'down' | 'up';
  buttonClassName?: string;
  panelClassName?: string;
  // Controlled mode (e.g. ui.ledgerOpen); uncontrolled when omitted.
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: (close: () => void) => React.ReactNode;
}

// A button with an anchored, non-modal panel: outside click and Esc close it
// (Esc returns focus to the button), Tab moves through its controls.
function Popover({
  label,
  title,
  ariaLabel,
  align = 'left',
  direction = 'down',
  buttonClassName,
  panelClassName = '',
  open: openProp,
  onOpenChange,
  children
}: PopoverProps) {
  const [innerOpen, setInnerOpen] = useState(false);
  const open = openProp ?? innerOpen;
  const setOpen = useCallback(
    (v: boolean) => {
      if (onOpenChange) onOpenChange(v);
      if (openProp === undefined) setInnerOpen(v);
    },
    [onOpenChange, openProp]
  );
  const close = useCallback(() => setOpen(false), [setOpen]);
  const rootRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open, setOpen]);

  useEffect(() => {
    if (open) panelRef.current?.querySelector<HTMLElement>(FOCUSABLE)?.focus({preventScroll: true});
  }, [open]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== 'Escape' || !open) return;
    e.preventDefault();
    e.stopPropagation();
    setOpen(false);
    buttonRef.current?.focus();
  };

  return (
    <div ref={rootRef} className="relative" onKeyDown={onKeyDown}>
      <button
        ref={buttonRef}
        type="button"
        className={
          buttonClassName ??
          `inline-flex h-7 items-center gap-1 rounded-md border px-2 text-[12px] transition-colors ${
            open
              ? 'border-base-content/30 bg-base-content/[0.06] text-base-content'
              : 'border-base-content/15 text-base-content/80 hover:border-base-content/30 hover:text-base-content'
          }`
        }
        title={title}
        aria-label={ariaLabel}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={() => setOpen(!open)}
      >
        {label}
      </button>
      {open && (
        <div
          ref={panelRef}
          id={panelId}
          role="dialog"
          aria-label={ariaLabel ?? title}
          className={`absolute z-50 rounded-lg border border-base-300 bg-base-100 text-[12px] shadow-xl ${
            direction === 'down' ? 'top-full mt-1' : 'bottom-full mb-1'
          } ${align === 'left' ? 'left-0' : 'right-0'} ${panelClassName}`}
        >
          {children(close)}
        </div>
      )}
    </div>
  );
}

interface SegmentOption<T extends string> {
  value: T;
  label: React.ReactNode;
  title: string;
  ariaLabel?: string;
  disabled?: boolean;
}

// Radio group drawn as a segmented control: one tab stop, arrows move the
// selection (skipping disabled options).
function Segmented<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  size = 'sm',
  bare = false,
  className = ''
}: {
  value: T | null;
  options: SegmentOption<T>[];
  onChange: (v: T) => void;
  ariaLabel: string;
  size?: 'xs' | 'sm';
  bare?: boolean; // no outline (inside another field)
  className?: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const current = options.findIndex(o => o.value === value);
  const tabStop = current >= 0 ? current : options.findIndex(o => !o.disabled);
  const move = (from: number, dir: 1 | -1) => {
    for (let k = 1; k <= options.length; k++) {
      const i = (from + dir * k + options.length) % options.length;
      if (!options[i].disabled) {
        onChange(options[i].value);
        refs.current[i]?.focus();
        return;
      }
    }
  };
  const h = size === 'xs' ? 'h-5 px-1.5 text-[10px]' : 'h-6 px-2 text-[11px]';
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={`inline-flex items-center gap-[2px] rounded-md p-[2px] ${bare ? '' : 'border border-base-content/15'} ${className}`}
    >
      {options.map((o, i) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            ref={el => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={on}
            aria-label={o.ariaLabel}
            disabled={o.disabled}
            tabIndex={i === tabStop ? 0 : -1}
            title={o.title}
            className={`${h} whitespace-nowrap rounded leading-none transition-colors disabled:cursor-not-allowed disabled:opacity-35 ${
              on
                ? 'bg-base-content/[0.09] font-semibold text-base-content'
                : 'text-base-content/60 hover:bg-base-content/[0.04] hover:text-base-content'
            }`}
            onClick={() => onChange(o.value)}
            onKeyDown={e => {
              if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                move(i, 1);
              } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                move(i, -1);
              }
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

interface ListOption<T extends string> {
  value: T;
  label: React.ReactNode;
  hint?: string; // one-line explanation under the label
  title?: string;
  disabled?: boolean;
  disabledReason?: string;
}

// Vertical radio list for menus: a check mark on the chosen option.
function OptionList<T extends string>({
  value,
  options,
  onChange,
  ariaLabel
}: {
  value: T;
  options: ListOption<T>[];
  onChange: (v: T) => void;
  ariaLabel: string;
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="flex flex-col">
      {options.map(o => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            disabled={o.disabled}
            title={o.disabled ? o.disabledReason : (o.title ?? o.hint)}
            className="group flex items-start gap-2 rounded px-2 py-1 text-left hover:bg-base-content/[0.05] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent"
            onClick={() => onChange(o.value)}
          >
            <span
              className={`mt-[1px] w-3 shrink-0 text-[11px] ${on ? 'text-base-content' : 'text-transparent'}`}
              aria-hidden="true"
            >
              ✓
            </span>
            <span className="min-w-0">
              <span className={`block text-[12px] ${on ? 'font-semibold text-base-content' : 'text-base-content/80'}`}>
                {o.label}
              </span>
              {o.hint && <span className="block text-[10.5px] leading-snug text-base-content/50">{o.hint}</span>}
            </span>
          </button>
        );
      })}
    </div>
  );
}

// A labelled on/off switch row (role="switch").
function SwitchRow({
  checked,
  onChange,
  label,
  hint,
  title,
  disabled
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: React.ReactNode;
  hint?: string;
  title: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      title={title}
      className="flex w-full items-start gap-2 rounded px-2 py-1 text-left hover:bg-base-content/[0.05] disabled:cursor-not-allowed disabled:opacity-40"
      onClick={() => onChange(!checked)}
    >
      <span
        className={`relative mt-[2px] inline-block h-3 w-5 shrink-0 rounded-full border transition-colors ${
          checked ? 'border-base-content/70 bg-base-content/70' : 'border-base-content/30 bg-transparent'
        }`}
        aria-hidden="true"
      >
        <span
          className={`absolute top-[1px] h-2 w-2 rounded-full transition-all ${
            checked ? 'left-[9px] bg-base-100' : 'left-[1px] bg-base-content/40'
          }`}
        />
      </span>
      <span className="min-w-0">
        <span className="block text-[12px] text-base-content/85">{label}</span>
        {hint && <span className="block text-[10.5px] leading-snug text-base-content/50">{hint}</span>}
      </span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Layout menu
// ---------------------------------------------------------------------------

const ROW_GROUPS: ListOption<RowGroup>[] = [
  {value: 'domain', label: 'Domain bands', hint: 'Person, Measurement, Condition, Drug, …'},
  {value: 'coverage', label: 'Coverage bucket', hint: 'Coded in all / most / some / two of the shown cohorts'},
  {value: 'none', label: 'None', hint: 'One continuous list'}
];

const COL_GROUPS: ListOption<ColGroup>[] = [
  {value: 'tier', label: 'Knowledge tier', hint: 'Profiled · Dictionary only · No dictionary'},
  {value: 'type', label: 'Study type', hint: 'Population → … → Heart failure'},
  {value: 'design', label: 'Design', hint: 'RCT, observational, retrospective, …'},
  {value: 'status', label: 'Status', hint: 'Ongoing · Completed · Unknown'},
  {value: 'none', label: 'None', hint: 'One block, sorted as below'}
];

const selectClass =
  'h-6 w-full rounded border border-base-content/15 bg-base-100 px-1 text-[11px] text-base-content focus:border-base-content/40 focus:outline-none';

function LayoutMenu({compact = false}: {compact?: boolean}) {
  const {model, state, dispatch, layout, revealRow} = useLoom();
  const patch = (p: Partial<LoomViewState>) => dispatch({type: 'patch', patch: p});
  const dictCohorts = model.cohorts.filter(c => c.tier !== 'none');
  const anchorName = state.anchor;
  const sortCohortName = state.sortCohort;

  const rowSorts: ListOption<RowSort>[] = [
    {value: 'coverage', label: 'Coverage', hint: 'Cohorts coding it, then measured coverage, then pooled, then A–Z'},
    {value: 'pooled', label: 'Pooled values', hint: 'Sum of each profiled cohort’s best variable (a lower bound)'},
    {value: 'az', label: 'A–Z', hint: 'By concept label'},
    {
      value: 'cohort',
      label: sortCohortName ? `By ${sortCohortName}` : 'By cohort…',
      hint: 'Its measured cells by completeness, then its outlined ones; choose the cohort below',
      disabled: !sortCohortName,
      disabledReason: 'Choose a cohort in “Sort rows by cohort” below first'
    },
    {
      value: 'anchor',
      label: 'Anchor completeness',
      hint: anchorName ? `Completeness in ${anchorName}` : 'Needs an anchor cohort',
      disabled: !anchorName,
      disabledReason: 'Set an anchor cohort below first'
    }
  ];
  const colSorts: ListOption<ColSort>[] = [
    {value: 'size', label: 'Size', hint: 'Dataset rows, else declared participants'},
    {value: 'name', label: 'Name', hint: 'Alphabetical'},
    {value: 'coverage', label: 'Coverage', hint: 'How many concepts the cohort codes'},
    {
      value: 'anchor',
      label: 'Similarity to anchor',
      hint: anchorName ? `Concepts shared with ${anchorName} (Jaccard)` : 'Needs an anchor cohort',
      disabled: !anchorName,
      disabledReason: 'Set an anchor cohort below first'
    }
  ];

  const bands = layout.items.filter(
    (it): it is Extract<FlatItem, {kind: 'band'}> & {domain: Domain} => it.kind === 'band' && it.domain !== null
  );
  const jumpTo = (domain: Domain, collapsed: boolean) => {
    const row = layout.order.find(r => model.rows[r].domain === domain);
    if (row === undefined) return;
    if (collapsed) {
      dispatch({type: 'toggleBand', domain});
      window.setTimeout(() => revealRow(row), 60);
    } else revealRow(row);
  };

  const setAnchor = (id: string) => {
    const anchor = id || null;
    const p: Partial<LoomViewState> = {anchor};
    if (!anchor && state.rowSort === 'anchor') p.rowSort = 'coverage';
    if (!anchor && state.colSort === 'anchor') p.colSort = 'size';
    if (anchor && !state.anchor) p.colSort = 'anchor';
    patch(p);
  };

  return (
    <Popover
      label={
        <>
          <LayoutIcon size={13} aria-hidden="true" />
          {!compact && <span>Layout</span>}
          <ChevronDown size={12} className="opacity-60" aria-hidden="true" />
        </>
      }
      ariaLabel="Layout"
      title="Layout: how rows and columns are grouped and sorted, order lock, anchor cohort, minimap"
      align="right"
      panelClassName="w-[540px] max-w-[calc(100vw-2rem)] p-3"
    >
      {() => (
        <div className="grid grid-cols-2 gap-x-4 gap-y-3">
          <section className="flex flex-col gap-1" aria-label="Rows">
            <Caps className="px-2">Group rows</Caps>
            <OptionList
              ariaLabel="Group rows"
              value={state.rowGroup}
              options={ROW_GROUPS}
              onChange={v => patch({rowGroup: v})}
            />
            <Caps className="mt-2 px-2">Sort rows</Caps>
            <OptionList
              ariaLabel="Sort rows"
              value={state.rowSort}
              options={rowSorts}
              onChange={v => patch({rowSort: v})}
            />
            <label
              className="mt-1 flex flex-col gap-1 px-2"
              title="Sort rows by the completeness of their best variable in this cohort"
            >
              <span className="text-[10.5px] text-base-content/55">Sort rows by cohort</span>
              <select
                className={selectClass}
                value={state.sortCohort ?? ''}
                aria-label="Sort rows by cohort"
                onChange={e => {
                  const id = e.target.value || null;
                  patch(
                    id
                      ? {sortCohort: id, rowSort: 'cohort'}
                      : {sortCohort: null, rowSort: state.rowSort === 'cohort' ? 'coverage' : state.rowSort}
                  );
                }}
              >
                <option value="">None</option>
                {dictCohorts.map(c => (
                  <option key={c.id} value={c.id}>
                    {c.id}
                    {c.tier === 'dictionary' ? ' (dictionary only)' : ''}
                  </option>
                ))}
              </select>
            </label>
            <div className="mt-2">
              <SwitchRow
                checked={state.lock}
                onChange={v => patch({lock: v})}
                label="Lock row order"
                hint="Filters hide rows without reshuffling the rest"
                title="When on, the row order is computed once and filters only hide rows, so nothing moves under your eyes; a Re-sort button appears when the order is stale"
              />
            </div>
          </section>
          <section className="flex flex-col gap-1" aria-label="Columns">
            <Caps className="px-2">Group columns</Caps>
            <OptionList
              ariaLabel="Group columns"
              value={state.colGroup}
              options={COL_GROUPS}
              onChange={v => patch({colGroup: v})}
            />
            <Caps className="mt-2 px-2">Sort columns within groups</Caps>
            <OptionList
              ariaLabel="Sort columns"
              value={state.colSort}
              options={colSorts}
              onChange={v => patch({colSort: v})}
            />
            <label
              className="mt-1 flex flex-col gap-1 px-2"
              title="Anchor: columns re-sort by the concepts they share with this cohort, and the size bar shows “shares N”"
            >
              <span className="text-[10.5px] text-base-content/55">Anchor cohort</span>
              <select
                className={selectClass}
                value={state.anchor ?? ''}
                aria-label="Anchor cohort"
                onChange={e => setAnchor(e.target.value)}
              >
                <option value="">None</option>
                {dictCohorts.map(c => (
                  <option key={c.id} value={c.id}>
                    {c.id}
                  </option>
                ))}
              </select>
            </label>
            <div className="mt-2">
              <SwitchRow
                checked={state.minimap}
                onChange={v => patch({minimap: v})}
                label="Minimap"
                hint="The whole filtered matrix at the right edge; drag to scroll"
                title="Show the minimap of the whole filtered matrix; click or drag it to jump"
              />
            </div>
          </section>
          {state.rowGroup === 'domain' && bands.length > 1 && (
            <section className="col-span-2 border-t border-base-300 pt-2" aria-label="Jump to band">
              <Caps className="mb-1 px-2">Jump to band</Caps>
              <div className="flex flex-wrap gap-1 px-1">
                {bands.map(b => (
                  <button
                    key={b.domain}
                    type="button"
                    className="rounded border border-base-content/15 px-1.5 py-[2px] text-[11px] text-base-content/75 hover:border-base-content/35 hover:text-base-content"
                    title={`Scroll to the ${DOMAIN_LABEL[b.domain]} band (${fmtInt(b.count)} concepts)${b.collapsed ? '; it is collapsed and will be expanded' : ''}`}
                    onClick={() => jumpTo(b.domain, b.collapsed)}
                  >
                    {DOMAIN_LABEL[b.domain]}{' '}
                    <span className="loom-tabular text-base-content/45">{fmtInt(b.count)}</span>
                  </button>
                ))}
              </div>
            </section>
          )}
          <p className="col-span-2 px-2 text-[10.5px] text-base-content/45">
            Grouping and sorting never change a cell’s colour: bins are fixed and absolute.
          </p>
        </div>
      )}
    </Popover>
  );
}

// ============================================================================
// SearchBox
// ============================================================================

type SearchChip = SearchQuery['chips'][number];

// The command bar's search: free text plus tokens (spec §7.4). The input keeps
// the raw query; a mirror layer behind it draws the recognised tokens as pills
// in place, so the syntax is visible while typing without ever changing what
// the reader typed. The dropdown lists the tokens (click = negate, × = remove)
// and the autocomplete; Enter on a concept pulls it into the Pool.

const DEBOUNCE_MS = 120;
const MAX_SUGGESTIONS = 16;

const KIND_HEADING: Record<SearchSuggestion['kind'], string> = {
  key: 'Filters',
  domain: 'Domains',
  cohort: 'Cohorts',
  concept: 'Concepts · Enter pulls into the Pool'
};

// [start, end) of the whitespace-separated token under the cursor; quotes keep
// spaces inside one token (in:"TheBox (myocardial infarction)").
function tokenSpan(raw: string, cursor: number): [number, number] {
  let start = 0;
  let quoted = false;
  for (let i = 0; i <= raw.length; i++) {
    const ch = raw[i];
    if (i === raw.length || (!quoted && /\s/.test(ch))) {
      if (cursor >= start && cursor <= i) return [start, i];
      start = i + 1;
    } else if (ch === '"') quoted = !quoted;
  }
  return [cursor, cursor];
}

const tidy = (s: string) => s.replace(/\s{2,}/g, ' ').replace(/^\s+/, '');

function suggestionDomain(sg: SearchSuggestion): Domain | null {
  const value = (sg.insert.split(':').pop() ?? '').replace(/^-/, '').toLowerCase();
  const label = sg.label.toLowerCase();
  return (
    DOMAINS.find(
      d => d.toLowerCase() === value || d.toLowerCase() === label || DOMAIN_LABEL[d].toLowerCase() === label
    ) ?? null
  );
}

function SearchBox() {
  const {model, state, dispatch, filter, layout, revealRow, setUi} = useLoom();
  const inputRef = useRef<HTMLInputElement>(null);
  const mirrorRef = useRef<HTMLSpanElement>(null);
  const [text, setText] = useState(state.q);
  const [cursor, setCursor] = useState(state.q.length);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const sent = useRef(state.q);
  const jumpRow = useRef(-1);
  const listId = useId();

  // Outside changes (Reset, a preset, a recipe chip) replace the text; our own
  // debounced writes do not (compared trimmed, so a trailing space survives).
  useEffect(() => {
    if (state.q.trim() === sent.current.trim()) return;
    sent.current = state.q;
    setText(state.q);
    setCursor(state.q.length);
  }, [state.q]);

  const commit = useCallback(
    (value: string) => {
      sent.current = value;
      dispatch({type: 'patch', patch: {q: value}});
    },
    [dispatch]
  );

  useEffect(() => {
    if (text === sent.current) return;
    const timer = window.setTimeout(() => commit(text), DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [text, commit]);

  // "/" focuses the search from anywhere.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== '/' || e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTypingTarget(e.target)) return;
      e.preventDefault();
      inputRef.current?.focus();
      inputRef.current?.select();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const parsed = useMemo(() => parseSearch(text, model), [text, model]);
  const chips = useMemo(() => [...parsed.chips].sort((a, b) => a.start - b.start), [parsed]);
  const suggestions = useMemo(
    () => (open ? searchSuggestions(model, text, cursor).slice(0, MAX_SUGGESTIONS) : []),
    [open, model, text, cursor]
  );
  const [spanStart, spanEnd] = tokenSpan(text, cursor);
  const typingToken = spanEnd > spanStart;

  // Filter mode pre-selects the first suggestion, so "type, Enter" pulls the
  // concept; Highlight mode keeps Enter for "next match".
  const preselect = state.qMode === 'filter' && typingToken && suggestions.length > 0;
  useEffect(() => setActive(preselect ? 0 : -1), [suggestions, preselect]);

  const syncScroll = useCallback(() => {
    const input = inputRef.current;
    if (input && mirrorRef.current) mirrorRef.current.style.transform = `translateX(${-input.scrollLeft}px)`;
  }, []);
  useLayoutEffect(syncScroll, [text, syncScroll]);

  const place = (value: string, pos: number) => {
    setText(value);
    setCursor(pos);
    requestAnimationFrame(() => {
      inputRef.current?.setSelectionRange(pos, pos);
      syncScroll();
    });
  };

  const readCursor = () => {
    const input = inputRef.current;
    if (input) setCursor(input.selectionStart ?? input.value.length);
    syncScroll();
  };

  const pulled = useMemo(() => new Set(state.pool.flatMap(k => k.split('~'))), [state.pool]);

  const accept = (sg: SearchSuggestion) => {
    if (sg.kind === 'concept' && sg.row !== undefined) {
      // The typed words found the concept: they leave the query as it goes into the Pool.
      const [a, b] = tokenSpan(text, cursor);
      const row = model.rows[sg.row];
      const before = text;
      place(tidy(text.slice(0, a) + text.slice(b)), a);
      if (pulled.has(row.key)) {
        setUi({toast: {id: Date.now(), text: `“${row.label}” is already in the Pool`}});
        return;
      }
      dispatch({type: 'pull', keys: [row.key]});
      setUi({
        toast: {
          id: Date.now(),
          text: `Pulled “${row.label}” into the Pool`,
          undo: () => {
            dispatch({type: 'release', key: row.key});
            place(before, before.length);
          }
        }
      });
      return;
    }
    const next = applySuggestion(text, cursor, sg);
    place(next.value, next.cursor);
  };

  const showConcept = (row: number) => {
    setOpen(false);
    setUi({inspector: {kind: 'concept', row}});
    revealRow(row);
  };

  const clear = () => {
    place('', 0);
    commit('');
  };

  const toggleNegation = (chip: SearchChip) => {
    const value = toggleChipNegation(text, chip);
    place(value, Math.max(0, cursor + (value.length - text.length) * (cursor > chip.start ? 1 : 0)));
  };
  const removeChip = (chip: SearchChip) => {
    const value = tidy(text.slice(0, chip.start) + text.slice(chip.end));
    place(value, Math.min(chip.start, value.length));
  };

  // Highlight mode: Enter walks the matches in display order.
  const jump = (dir: 1 | -1) => {
    const hl = filter.highlight;
    const order = layout.order;
    if (!hl || order.length === 0) return;
    const n = order.length;
    // From the last match we jumped to; the first jump starts at the top (or the bottom going back).
    const from = order.indexOf(jumpRow.current);
    for (let k = 1; k <= n; k++) {
      const i = from < 0 ? (dir > 0 ? k - 1 : n - k) : (((from + dir * k) % n) + n) % n;
      if (hl[order[i]]) {
        jumpRow.current = order[i];
        revealRow(order[i]);
        return;
      }
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    const n = suggestions.length;
    if (e.key === 'ArrowDown' && n > 0) {
      e.preventDefault();
      setOpen(true);
      setActive(a => (a + 1) % n);
    } else if (e.key === 'ArrowUp' && n > 0) {
      e.preventDefault();
      setActive(a => (a <= 0 ? n - 1 : a - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (open && active >= 0 && active < n) accept(suggestions[active]);
      else {
        if (text !== sent.current) commit(text);
        if (state.qMode === 'highlight') jump(e.shiftKey ? -1 : 1);
        else setOpen(false);
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      if (open && n > 0) setOpen(false);
      else if (text) clear();
      else inputRef.current?.blur();
    }
  };

  // Mirror segments: key tokens (d:, in:, …) as pills, negated terms struck,
  // plain words as typed. Same font, no padding or border on the pills (a
  // ring is a box-shadow), so the mirror lines up with the input's own glyphs
  // to the pixel.
  const mirror: React.ReactNode[] = [];
  let at = 0;
  chips.forEach((c, i) => {
    if (c.start < at || c.end > text.length) return;
    if (c.start > at) mirror.push(<span key={`t${i}`}>{text.slice(at, c.start)}</span>);
    const raw = text.slice(c.start, c.end);
    const key = /^-?[A-Za-z]+:/.exec(raw);
    const pill = key
      ? c.negated
        ? 'rounded-[3px] text-base-content/55 line-through decoration-base-content/40 ring-1 ring-base-content/25'
        : 'rounded-[3px] bg-base-content/[0.08] ring-1 ring-base-content/[0.12]'
      : c.negated
        ? 'text-base-content/55 line-through decoration-base-content/40'
        : '';
    mirror.push(
      <span key={`c${i}`} className={pill}>
        {key ? (
          <>
            <span className="text-base-content/50">{key[0]}</span>
            {raw.slice(key[0].length)}
          </>
        ) : (
          raw
        )}
      </span>
    );
    at = c.end;
  });
  if (at < text.length) mirror.push(<span key="tail">{text.slice(at)}</span>);

  const activeId = open && active >= 0 && active < suggestions.length ? `${listId}-o${active}` : undefined;
  const showPanel = open && (suggestions.length > 0 || chips.length > 0 || !text);
  const qActive = state.q.trim() !== '';

  const modeOption = (value: 'filter' | 'highlight', label: string, title: string) => {
    const on = state.qMode === value;
    return (
      <button
        type="button"
        role="radio"
        aria-checked={on}
        tabIndex={on ? 0 : -1}
        title={title}
        className={`inline-flex items-center gap-[3px] rounded-sm px-0.5 hover:text-base-content ${on ? 'text-base-content/80' : 'text-base-content/45'}`}
        onMouseDown={e => e.preventDefault()}
        onClick={() => dispatch({type: 'patch', patch: {qMode: value}})}
        onKeyDown={e => {
          if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
            e.preventDefault();
            const other = value === 'filter' ? 'highlight' : 'filter';
            dispatch({type: 'patch', patch: {qMode: other}});
            (e.currentTarget.parentElement?.querySelector(`[data-mode="${other}"]`) as HTMLElement | null)?.focus();
          }
        }}
        data-mode={value}
      >
        <span
          className={`inline-block h-[7px] w-[7px] rounded-full border ${on ? 'border-base-content/70 bg-base-content/70' : 'border-base-content/40'}`}
          aria-hidden="true"
        />
        {label}
      </button>
    );
  };

  return (
    <div className="relative flex min-w-[200px] max-w-[460px] flex-1 flex-col justify-center gap-[2px]">
      <div className="flex h-[26px] items-center rounded-md border border-base-content/15 bg-base-100 transition-colors focus-within:border-base-content/45 hover:border-base-content/30">
        <SearchIcon size={12} className="ml-2 shrink-0 text-base-content/45" aria-hidden="true" />
        <div className="relative mx-1.5 h-full min-w-0 flex-1 overflow-hidden">
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 whitespace-pre text-[12px] leading-[24px] text-base-content"
          >
            <span ref={mirrorRef} className="inline-block will-change-transform">
              {mirror}
            </span>
          </div>
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded={showPanel}
            aria-controls={showPanel ? listId : undefined}
            aria-autocomplete="list"
            aria-activedescendant={activeId}
            aria-label="Search concepts"
            title={
              'Search concepts (press / from anywhere). Tokens: d: in: notin: code: omop: v: t: is: unit: cov:>=k n:>=k, "phrase", a | b, -term'
            }
            placeholder="Search concepts…   d:measurement  in:TIME-CHF  -v:un"
            spellCheck={false}
            autoComplete="off"
            className="absolute inset-0 h-full w-full bg-transparent text-[12px] leading-[24px] text-transparent caret-base-content outline-none placeholder:text-base-content/40 selection:bg-base-content/20"
            value={text}
            onChange={e => {
              setText(e.target.value);
              setCursor(e.target.selectionStart ?? e.target.value.length);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onBlur={() => setOpen(false)}
            onKeyDown={onKeyDown}
            onKeyUp={readCursor}
            onClick={readCursor}
            onSelect={readCursor}
            onScroll={syncScroll}
          />
        </div>
        {text && (
          <button
            type="button"
            className="mr-1 flex h-5 w-5 shrink-0 items-center justify-center rounded text-base-content/45 hover:bg-base-content/[0.06] hover:text-base-content"
            title="Clear the search (Esc)"
            aria-label="Clear the search"
            onMouseDown={e => e.preventDefault()}
            onClick={clear}
          >
            <X size={12} />
          </button>
        )}
      </div>
      <div className="flex h-3 items-center justify-between px-1 text-[10px] leading-none">
        <span role="radiogroup" aria-label="Search mode" className="flex items-center gap-2">
          {modeOption('filter', 'Filter', 'Filter: hide the concepts that do not match')}
          {modeOption(
            'highlight',
            'Highlight',
            'Highlight: keep every concept in place and dim the non-matches to 25%; Enter jumps to the next match'
          )}
        </span>
        {qActive && (
          <span
            className="loom-tabular whitespace-nowrap text-base-content/50"
            title={`${fmtInt(filter.matchCount)} concepts match the search`}
            aria-live="polite"
          >
            {fmtInt(filter.matchCount)} {filter.matchCount === 1 ? 'match' : 'matches'}
            {state.qMode === 'highlight' && filter.matchCount > 0 ? ' · ↵ next' : ''}
          </span>
        )}
      </div>

      {showPanel && (
        <div
          className="absolute left-0 top-full z-50 mt-1 w-[max(100%,440px)] max-w-[calc(100vw-2rem)] overflow-hidden rounded-lg border border-base-300 bg-base-100 text-[12px] shadow-xl"
          onMouseDown={e => e.preventDefault()}
        >
          {chips.length > 0 && (
            <div className="flex flex-wrap items-center gap-1 border-b border-base-300 px-2.5 py-2">
              <span className="mr-1 text-[10px] font-semibold uppercase tracking-[0.09em] text-base-content/45">
                Tokens
              </span>
              {chips.map((c, i) => (
                <span
                  key={`${c.start}-${i}`}
                  className={`inline-flex items-center rounded border text-[11px] ${
                    c.negated
                      ? 'border-dashed border-base-content/30 text-base-content/60'
                      : 'border-base-content/20 bg-base-content/[0.05]'
                  }`}
                >
                  <button
                    type="button"
                    className={`px-1.5 py-[1px] ${c.negated ? 'line-through decoration-base-content/40' : ''}`}
                    title={
                      c.negated
                        ? `Excluded: ${c.text}. Click to include instead.`
                        : `${c.text}. Click to exclude instead (negate).`
                    }
                    aria-label={`${c.negated ? 'Include' : 'Exclude'} ${c.text}`}
                    onClick={() => toggleNegation(c)}
                  >
                    {text.slice(c.start, c.end) || c.text}
                  </button>
                  <button
                    type="button"
                    className="border-l border-base-content/15 px-1 text-base-content/45 hover:text-base-content"
                    title={`Remove ${c.text}`}
                    aria-label={`Remove ${c.text}`}
                    onClick={() => removeChip(c)}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
          <ul id={listId} role="listbox" aria-label="Suggestions" className="max-h-[320px] overflow-y-auto py-1">
            {suggestions.map((sg, i) => {
              const heading = i === 0 || suggestions[i - 1].kind !== sg.kind;
              const dom =
                sg.kind === 'concept' && sg.row !== undefined
                  ? model.rows[sg.row].domain
                  : sg.kind === 'domain'
                    ? suggestionDomain(sg)
                    : null;
              const isPulled = sg.kind === 'concept' && sg.row !== undefined && pulled.has(model.rows[sg.row].key);
              return (
                <React.Fragment key={`${i}:${sg.kind}:${sg.insert}`}>
                  {heading && (
                    <li
                      role="presentation"
                      className="px-2.5 pb-0.5 pt-1.5 text-[10px] font-semibold uppercase tracking-[0.09em] text-base-content/40"
                    >
                      {KIND_HEADING[sg.kind]}
                    </li>
                  )}
                  <li
                    id={`${listId}-o${i}`}
                    role="option"
                    aria-selected={i === active}
                    className={`group flex cursor-pointer items-center gap-2 px-2.5 py-[5px] ${
                      i === active ? 'bg-base-content/[0.08]' : 'hover:bg-base-content/[0.04]'
                    }`}
                    onMouseEnter={() => setActive(i)}
                    onClick={() => accept(sg)}
                    title={
                      sg.kind === 'concept'
                        ? isPulled
                          ? 'Already in the Pool'
                          : 'Click or Enter: pull this concept into the Pool'
                        : `Insert ${sg.insert}`
                    }
                  >
                    <span className="flex w-4 shrink-0 justify-center">
                      {dom ? (
                        <DomainMark domain={dom} />
                      ) : sg.kind === 'cohort' && sg.tier ? (
                        <TierGlyph tier={sg.tier} />
                      ) : (
                        <span className="font-mono text-[10px] text-base-content/45">:</span>
                      )}
                    </span>
                    <span className={`min-w-0 truncate ${sg.kind === 'key' ? 'font-mono text-[11.5px]' : ''}`}>
                      {sg.label}
                    </span>
                    {sg.detail && (
                      <span className="ml-auto shrink-0 pl-2 text-[10.5px] text-base-content/45">{sg.detail}</span>
                    )}
                    {sg.kind === 'concept' && sg.row !== undefined && (
                      <span className={`flex shrink-0 items-center gap-1 ${sg.detail ? '' : 'ml-auto'}`}>
                        <button
                          type="button"
                          className="rounded px-1 text-[10.5px] text-base-content/50 opacity-0 hover:bg-base-content/[0.08] hover:text-base-content focus:opacity-100 group-hover:opacity-100"
                          title="Show this concept: scroll to its row and open it in the inspector (does not pull it)"
                          aria-label={`Show ${sg.label}`}
                          onClick={e => {
                            e.stopPropagation();
                            showConcept(sg.row as number);
                          }}
                        >
                          show
                        </button>
                        <span
                          className={`inline-flex items-center gap-0.5 rounded border px-1 text-[10px] ${
                            i === active
                              ? 'border-base-content/30 text-base-content/70'
                              : 'border-transparent text-transparent'
                          }`}
                          aria-hidden="true"
                        >
                          <CornerDownLeft size={9} /> {isPulled ? 'pulled' : 'pull'}
                        </span>
                      </span>
                    )}
                  </li>
                </React.Fragment>
              );
            })}
          </ul>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 border-t border-base-300 bg-base-content/[0.02] px-2.5 py-1.5 text-[10.5px] text-base-content/50">
            <span>↑↓ choose</span>
            <span>↵ insert · on a concept: pull into the Pool</span>
            {state.qMode === 'highlight' && <span>↵ with nothing chosen: next match (⇧↵ previous)</span>}
            <span>{text ? 'Esc closes, again clears' : 'Esc closes'}</span>
            {!text && (
              <span className="basis-full font-mono text-[10.5px] text-base-content/45">
                a | b · &quot;phrase&quot; · -term · d: in: notin: code:loinc:* omop: v:bl t:num is:bridged unit:
                cov:&gt;=3 n:&gt;=1000
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// VisitScrubber
// ============================================================================

// The visit lens: slices the whole matrix to one visit slot, a range, a set,
// "any follow-up" or the unanchored variables. Only slots that occur in the
// data get a stop; each stop carries a neutral histogram of the variables
// recorded at it in the shown cohorts, so the reader sees where the data is
// before choosing. `[` / `]` step through the stops from anywhere on the page.

const UN_TITLE =
  'Unanchored: not tied to a study timepoint (for example "visit date" or "days of visit"). Never counted as baseline.';

const idx = (s: VisitSlot) => VISIT_SLOTS.indexOf(s);

// Mask of every anchored slot between a and b (inclusive), so a range reads
// as one contiguous span ("Baseline–6 months") even across slots with no data.
function rangeMask(a: VisitSlot, b: VisitSlot): number {
  const [lo, hi] = idx(a) <= idx(b) ? [idx(a), idx(b)] : [idx(b), idx(a)];
  let mask = 0;
  for (let i = lo; i <= hi; i++) if (VISIT_SLOTS[i] !== 'UN') mask |= SLOT_BIT[VISIT_SLOTS[i]];
  return mask;
}

const single = (s: VisitSlot): VisitLens => ({kind: 'slots', mask: SLOT_BIT[s]});

// The stop sequence `[` / `]` walk through: All, each present anchored slot, Unanchored.
type StepStop = 'all' | VisitSlot;

function stepLens(model: LoomModel, lens: VisitLens, dir: 1 | -1): VisitLens | null {
  const seq: StepStop[] = ['all', ...VISIT_SLOTS.filter(s => (model.slotsPresent & SLOT_BIT[s]) !== 0)];
  let at: number;
  if (lens.kind === 'all') at = 0;
  else {
    const inLens = seq.map((s, i) => (s !== 'all' && (lens.mask & SLOT_BIT[s]) !== 0 ? i : -1)).filter(i => i >= 0);
    if (inLens.length === 0) at = 0;
    else at = dir > 0 ? inLens[inLens.length - 1] : inLens[0];
  }
  const next = at + dir;
  if (next < 0 || next >= seq.length) return null;
  const stop = seq[next];
  return stop === 'all' ? ALL_LENS : single(stop);
}

// Width of the full scrubber for a given data set (the command bar picks the
// compact form when it would not fit).
function scrubberWidth(model: LoomModel): number {
  const present = VISIT_SLOTS.filter(s => (model.slotsPresent & SLOT_BIT[s]) !== 0);
  const anchored = present.filter(s => s !== 'UN').length;
  return (
    40 + anchored * 35 + ((FOLLOW_UP_MASK & model.slotsPresent) !== 0 ? 70 : 0) + (present.includes('UN') ? 46 : 0)
  );
}

function VisitScrubber({compact = false}: {compact?: boolean}) {
  const {model, state, dispatch, columns} = useLoom();
  const lens = state.visit;
  const setLens = useCallback((l: VisitLens) => dispatch({type: 'patch', patch: {visit: l}}), [dispatch]);
  const anchorRef = useRef<VisitSlot | null>(null);

  const present = VISIT_SLOTS.filter(s => (model.slotsPresent & SLOT_BIT[s]) !== 0);
  const anchored = present.filter(s => s !== 'UN');
  const hasUn = present.includes('UN');
  const presentFu = FOLLOW_UP_MASK & model.slotsPresent;

  // Variables per slot among the SHOWN cohorts (all rows, any lens): it moves
  // with the cohort filters but stays put while scrubbing, so the histogram
  // is a map of the data rather than of the current slice.
  const counts = useMemo(() => {
    const out = {} as Record<VisitSlot, number>;
    VISIT_SLOTS.forEach(s => (out[s] = 0));
    for (const v of model.variables) if (v.home >= 0 && columns.visibleMask[v.cohort]) out[v.slot]++;
    return out;
  }, [model, columns.visibleMask]);
  const max = Math.max(1, ...present.map(s => counts[s]));

  const inLens = (s: VisitSlot) => lens.kind === 'slots' && (lens.mask & SLOT_BIT[s]) !== 0;
  const fuActive =
    lens.kind === 'slots' &&
    presentFu !== 0 &&
    (lens.mask & ~FOLLOW_UP_MASK) === 0 &&
    (lens.mask & presentFu) === presentFu;
  const isSingle = (s: VisitSlot) => lens.kind === 'slots' && lens.mask === SLOT_BIT[s];

  const onStop = (s: VisitSlot, e: React.MouseEvent) => {
    if (e.metaKey || e.ctrlKey) {
      // Add or remove one slot (a set such as "Baseline, 12 months").
      const base = lens.kind === 'slots' ? lens.mask : 0;
      const mask = base ^ SLOT_BIT[s];
      setLens(mask === 0 ? ALL_LENS : {kind: 'slots', mask});
      anchorRef.current = s;
      return;
    }
    if (e.shiftKey && s !== 'UN' && lens.kind === 'slots') {
      const from =
        anchorRef.current && anchorRef.current !== 'UN' && inLens(anchorRef.current)
          ? anchorRef.current
          : anchored.find(a => inLens(a));
      if (from) {
        setLens({kind: 'slots', mask: rangeMask(from, s)});
        return;
      }
    }
    anchorRef.current = s;
    setLens(isSingle(s) ? ALL_LENS : single(s));
  };

  // [ and ] step the lens from anywhere, unless the reader is typing or a
  // focused widget (the grid) already handled the key.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTypingTarget(e.target)) return;
      if (e.key !== '[' && e.key !== ']') return;
      const next = stepLens(model, lens, e.key === ']' ? 1 : -1);
      e.preventDefault();
      if (next) setLens(next);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [model, lens, setLens]);

  const stopTitle = (s: VisitSlot) =>
    `${s === 'UN' ? UN_TITLE : SLOT_LONG[s]} · ${fmtInt(counts[s])} variable${counts[s] === 1 ? '' : 's'} in the shown cohorts.\n` +
    'Click: only this visit (again: all visits) · Shift-click: range · ⌘/Ctrl-click: add or remove · [ ] step';

  const bar = (s: VisitSlot, strong: boolean) => {
    const h = counts[s] > 0 ? Math.max(1, Math.round((counts[s] / max) * 8)) : 0;
    return (
      <span className="flex h-2 items-end" aria-hidden="true">
        <span
          className={`block w-3 rounded-[1px] ${strong ? 'bg-base-content/60' : 'bg-base-content/20'}`}
          style={{height: h}}
        />
      </span>
    );
  };

  const stopClass = (on: boolean, within: boolean) =>
    `flex h-8 min-w-[26px] flex-col items-center justify-end gap-[3px] rounded px-1 pb-[5px] text-[10.5px] leading-none transition-colors ${
      on
        ? 'bg-base-content/[0.1] font-semibold text-base-content'
        : within
          ? 'bg-base-content/[0.05] text-base-content'
          : 'text-base-content/60 hover:bg-base-content/[0.04] hover:text-base-content'
    }`;

  // Compact form (narrow screens): a select with step buttons.
  const selectValue =
    lens.kind === 'all' ? 'all' : fuActive ? 'fu' : (present.find(s => lens.mask === SLOT_BIT[s]) ?? 'custom');
  const onSelect = (v: string) => {
    if (v === 'all') setLens(ALL_LENS);
    else if (v === 'fu') setLens({kind: 'slots', mask: FOLLOW_UP_MASK});
    else if (v !== 'custom') setLens(single(v as VisitSlot));
  };
  const prev = stepLens(model, lens, -1);
  const next = stepLens(model, lens, 1);

  if (!compact)
    return (
      <div
        role="group"
        aria-label={`Visit lens: ${lensLabel(lens)}`}
        className="flex items-end gap-[1px] rounded-md border border-base-content/15 px-[2px] py-[1px]"
      >
        <button
          type="button"
          aria-pressed={lens.kind === 'all'}
          className={stopClass(lens.kind === 'all', false)}
          title="All visits: each cell shows its best visit. [ ] step through the visits."
          onClick={() => setLens(ALL_LENS)}
        >
          <span className="flex h-2 items-end" aria-hidden="true" />
          All
        </button>
        {anchored.map(s => (
          <button
            key={s}
            type="button"
            aria-pressed={inLens(s)}
            aria-label={`${SLOT_LONG[s]}, ${fmtInt(counts[s])} variables`}
            className={stopClass(isSingle(s), inLens(s))}
            title={stopTitle(s)}
            onClick={e => onStop(s, e)}
          >
            {bar(s, inLens(s))}
            {SLOT_SHORT[s]}
          </button>
        ))}
        {presentFu !== 0 && (
          <button
            type="button"
            aria-pressed={fuActive}
            className={stopClass(fuActive, false)}
            title="Any follow-up: every visit from 1 month to end of study. Completeness is of dataset rows, not of patients still enrolled; attrition and death lower it."
            onClick={() => setLens(fuActive ? ALL_LENS : {kind: 'slots', mask: FOLLOW_UP_MASK})}
          >
            <span className="flex h-2 items-end" aria-hidden="true" />
            Follow-up
          </button>
        )}
        {hasUn && (
          <>
            <span
              className="mx-[3px] mb-[6px] h-5 w-0 self-end border-l border-dashed border-base-content/30"
              aria-hidden="true"
            />
            <button
              type="button"
              aria-pressed={inLens('UN')}
              aria-label={`Unanchored, ${fmtInt(counts.UN)} variables`}
              className={stopClass(isSingle('UN'), inLens('UN'))}
              title={stopTitle('UN')}
              onClick={e => onStop('UN', e)}
            >
              {bar('UN', inLens('UN'))}
              {SLOT_SHORT.UN}
            </button>
          </>
        )}
      </div>
    );

  return (
    <div className="flex items-center gap-[2px]" role="group" aria-label="Visit lens">
      <button
        type="button"
        className="flex h-7 w-6 items-center justify-center rounded text-base-content/60 hover:bg-base-content/[0.05] hover:text-base-content disabled:opacity-30"
        title="Previous visit ( [ )"
        aria-label="Previous visit"
        disabled={!prev}
        onClick={() => prev && setLens(prev)}
      >
        <ChevronLeft size={14} />
      </button>
      <select
        className="h-7 max-w-[11rem] rounded-md border border-base-content/15 bg-base-100 px-1.5 text-[12px] text-base-content focus:border-base-content/40 focus:outline-none"
        value={selectValue}
        aria-label="Visit lens"
        title="Visit lens: slice the matrix to one visit. [ ] step through the visits."
        onChange={e => onSelect(e.target.value)}
      >
        <option value="all">All visits</option>
        {anchored.map(s => (
          <option key={s} value={s}>
            {SLOT_LONG[s]} · {fmtInt(counts[s])}
          </option>
        ))}
        {presentFu !== 0 && <option value="fu">Any follow-up</option>}
        {hasUn && <option value="UN">Unanchored · {fmtInt(counts.UN)}</option>}
        {selectValue === 'custom' && <option value="custom">{lensLabel(lens)}</option>}
      </select>
      <button
        type="button"
        className="flex h-7 w-6 items-center justify-center rounded text-base-content/60 hover:bg-base-content/[0.05] hover:text-base-content disabled:opacity-30"
        title="Next visit ( ] )"
        aria-label="Next visit"
        disabled={!next}
        onClick={() => next && setLens(next)}
      >
        <ChevronRight size={14} />
      </button>
    </div>
  );
}

// ============================================================================
// CommandBar
// ============================================================================

// The 44px command bar: search, metric, visit lens, density, layout, views,
// Loom / Table and Reset. Below 1280px the rail becomes a drawer and its
// "Filters (n)" button lives here. The bar measures itself and steps down
// (full scrubber -> select with ◂ ▸, full labels -> short ones) instead of
// overflowing, because how many visit stops exist depends on the data.

// Widths (px) of the bar's parts as rendered, for choosing a fit level.
const W = {
  search: 220, // the search box's comfortable minimum
  metric: 262,
  scrubberCompact: 210,
  right: 505, // density, layout, views, Loom / Table, reset, all labelled
  rightIcons: 370, // views and Loom / Table as icons
  gaps: 72, // padding and gaps
  filters: 90 // "Filters (n)", shown below 1280px
};

const UNAVAILABLE = 'Counts could not be loaded; showing presence only';

const DENSITIES: {value: Density; label: string; hint: string}[] = [
  {
    value: 'overview',
    label: 'Overview',
    hint: '6 × 4 px cells: the whole cloth at once; outlined states merge into a line'
  },
  {value: 'compact', label: 'Compact', hint: '16 × 14 px pitch: about 40 rows on a laptop screen'},
  {value: 'comfortable', label: 'Comfortable', hint: '22 × 20 px pitch, codes under each label, basket notches'}
];

function CommandBar() {
  const {model, state, dispatch, ui, setUi, filter, retryCounts} = useLoom();
  const unavailable = model.countsStatus === 'unavailable';
  // When counts fail the cells fall back to presence, whatever the URL asks for.
  const metric: Metric = unavailable ? 'pres' : state.metric;
  const preset = activePreset(state);
  const density = DENSITIES.find(d => d.value === state.density) ?? DENSITIES[1];

  const barRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(1440);
  useLayoutEffect(() => {
    const el = barRef.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // 3: everything spelled out; 2: Views and Loom / Table as icons; 1: the
  // visit lens as a select with ◂ ▸; 0: short metric labels, icon-only Layout.
  const scrubber = scrubberWidth(model);
  const base = W.search + W.gaps + (width < 1280 ? W.filters : 0);
  const level =
    width >= base + W.metric + scrubber + W.right
      ? 3
      : width >= base + W.metric + scrubber + W.rightIcons
        ? 2
        : width >= base + W.metric + W.scrubberCompact + W.rightIcons
          ? 1
          : 0;
  const short = level === 0;
  const iconsOnly = level <= 2;

  const reset = () => {
    const before = state;
    dispatch({type: 'reset'});
    setUi({
      toast: {
        id: Date.now(),
        text: 'View reset to the defaults (the Pool is kept)',
        undo: () => dispatch({type: 'patch', patch: before})
      }
    });
  };

  const metricOptions = (['comp', 'n', 'pres'] as Metric[]).map(m => ({
    value: m,
    disabled: unavailable && m !== 'pres',
    title:
      unavailable && m !== 'pres'
        ? UNAVAILABLE
        : METRIC_EXPLAINER[m] + (m === 'pres' ? ': every counted cell in ink, no ramp' : ''),
    ariaLabel: m === 'comp' ? 'Completeness' : m === 'n' ? 'Count' : 'Presence',
    label: short
      ? m === 'comp'
        ? 'C'
        : m === 'n'
          ? 'n'
          : 'P'
      : m === 'comp'
        ? 'Completeness'
        : m === 'n'
          ? 'Count'
          : 'Presence'
  }));

  return (
    <div ref={barRef} className="relative z-[45] flex h-11 shrink-0 items-center gap-2 border-b border-base-300 px-4">
      <button
        type="button"
        className="inline-flex h-7 shrink-0 items-center gap-1.5 rounded-md border border-base-content/15 px-2 text-[12px] text-base-content/80 hover:border-base-content/30 hover:text-base-content xl:hidden"
        title="Open the filters drawer (cohorts and concepts)"
        aria-label={`Filters, ${filter.chips.length} active`}
        aria-expanded={ui.railOpen}
        onClick={() => setUi({railOpen: true})}
      >
        <Sliders size={13} aria-hidden="true" />
        Filters
        {filter.chips.length > 0 && <span className="loom-tabular text-base-content/60">({filter.chips.length})</span>}
      </button>

      <SearchBox />

      <div className="flex shrink-0 items-center gap-1">
        <Segmented
          ariaLabel="Metric"
          value={metric}
          options={metricOptions}
          onChange={m => dispatch({type: 'patch', patch: {metric: m}})}
        />
        {unavailable && (
          <button
            type="button"
            className="inline-flex h-7 items-center gap-1 rounded-md border border-base-content/15 px-1.5 text-[11px] text-base-content/75 hover:border-base-content/30 hover:text-base-content"
            title={`${UNAVAILABLE}. Retry loading the observation counts.`}
            onClick={retryCounts}
          >
            <AlertCircle size={12} aria-hidden="true" />
            <RefreshCw size={11} aria-hidden="true" />
            Retry
          </button>
        )}
      </div>

      <div className="flex shrink-0 items-center">
        <VisitScrubber compact={level <= 1} />
      </div>

      <div className="ml-auto flex shrink-0 items-center gap-1.5">
        <Popover
          ariaLabel="Density"
          title={`Density: ${density.label}. ${density.hint}`}
          align="right"
          panelClassName="w-[300px] p-2"
          label={
            <>
              <span>{density.label}</span>
              <ChevronDown size={12} className="opacity-60" aria-hidden="true" />
            </>
          }
        >
          {close => (
            <>
              <Caps className="mb-1 px-2 pt-1">Density</Caps>
              <OptionList
                ariaLabel="Density"
                value={state.density}
                options={DENSITIES}
                onChange={v => {
                  dispatch({type: 'patch', patch: {density: v}});
                  close();
                }}
              />
            </>
          )}
        </Popover>

        <LayoutMenu compact={short} />

        <Popover
          ariaLabel="Views"
          title={`Views: ready-made settings for common questions${preset ? ` (current: ${preset.label})` : ''}`}
          align="right"
          panelClassName="w-[340px] p-2"
          label={
            <>
              <Bookmark size={13} aria-hidden="true" />
              {!iconsOnly && <span className="max-w-[9rem] truncate">{preset ? preset.label : 'Views'}</span>}
              <ChevronDown size={12} className="opacity-60" aria-hidden="true" />
            </>
          }
        >
          {close => (
            <>
              <Caps className="mb-1 px-2 pt-1">Views</Caps>
              <OptionList
                ariaLabel="Views"
                value={preset?.id ?? ''}
                options={VIEW_PRESETS.map(p => ({value: p.id, label: p.label, hint: p.description}))}
                onChange={id => {
                  const p = VIEW_PRESETS.find(x => x.id === id);
                  if (p) dispatch({type: 'patch', patch: p.patch});
                  close();
                }}
              />
              <p className="mt-1 border-t border-base-300 px-2 pt-1.5 text-[10.5px] leading-snug text-base-content/50">
                A view sets the concept filters, the visit and the set mode; your cohort filters, pins, search and Pool
                stay.
              </p>
            </>
          )}
        </Popover>

        <Segmented
          ariaLabel="View as"
          value={state.view}
          onChange={v => dispatch({type: 'patch', patch: {view: v}})}
          options={[
            {
              value: 'loom',
              title: 'Matrix: the concept × cohort grid',
              ariaLabel: 'Matrix',
              label: (
                <span className="inline-flex items-center gap-1">
                  <Grid size={11} aria-hidden="true" />
                  {!iconsOnly && <span>Matrix</span>}
                </span>
              )
            },
            {
              value: 'table',
              title: 'Table: the same rows and cohorts as a table (n / N per cell), with a CSV download',
              ariaLabel: 'Table',
              label: (
                <span className="inline-flex items-center gap-1">
                  <Table size={11} aria-hidden="true" />
                  {!iconsOnly && <span>Table</span>}
                </span>
              )
            }
          ]}
        />

        <button
          type="button"
          className="flex h-7 w-7 items-center justify-center rounded-md text-base-content/60 hover:bg-base-content/[0.06] hover:text-base-content"
          title="Reset: restore the default view (filters, pins, visit, metric, layout). The Pool is kept; Undo is offered."
          aria-label="Reset the view"
          onClick={reset}
        >
          <RotateCcw size={14} />
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// CubeFace
// ============================================================================

// ---------------------------------------------------------------------------
// The cell grammar (spec §4.3) as SVG, shared by the inspector panels: the
// outline says the dictionary lists it; what's inside says what profiling
// found. Same rules as the canvas grid, so a glyph means the same thing
// wherever it appears.
// ---------------------------------------------------------------------------

// The theme's base-100: dog-ears are cut out of fills with the surface colour.
const CUBE_FACE_SURFACE = 'oklch(var(--b1))';

// Fill of a counted mark: its ramp bin, or the neutral presence fill when the
// metric has no bin (Presence, or Completeness with an unknown dataset size).
const binFill = (bin: number): string => (bin > 0 ? `var(--loom-bin-${bin})` : 'var(--loom-ink-fill)');

const fillFor = (metric: Metric, lo: number, N: number | null): string => binFill(binFor(metric, lo, N));

interface GlyphProps {
  x: number;
  y: number;
  w: number;
  h: number;
  state: CellStateValue;
  flags: number;
  lo: number;
  N: number | null;
  metric: Metric;
  r?: number;
}

function CubeFaceCellGlyph({x, y, w, h, state, flags, lo, N, metric, r = 2}: GlyphProps) {
  // Guest-only cells (legacy secondary codes): the same glyph at 60%, centred.
  if (flags & CellFlag.GuestOnly) {
    const gw = Math.round(w * 0.6);
    const gh = Math.round(h * 0.6);
    x += Math.round((w - gw) / 2);
    y += Math.round((h - gh) / 2);
    w = gw;
    h = gh;
    r = Math.min(r, 1);
  }
  const multi = (flags & CellFlag.Multi) !== 0;
  const dogEar = (fill: string) => <path d={`M${x + w - 3} ${y}H${x + w}V${y + 3}Z`} style={{fill}} />;
  const ring = (
    <rect
      x={x + 0.5}
      y={y + 0.5}
      width={w - 1}
      height={h - 1}
      rx={r}
      style={{fill: 'none', stroke: 'var(--loom-mark)'}}
      strokeWidth={1}
    />
  );
  const cx = x + w / 2;
  const cy = Math.round(y + h / 2) + 0.5;
  switch (state) {
    case CellState.Counted: {
      const fill = fillFor(metric, lo, N);
      // Below the usable floor: the fill collapses to a 3px bar in the same
      // bin colour, so it never reads as a lower bin.
      if (flags & CellFlag.BelowFloor) return <rect x={x} y={y + h - 3} width={w} height={3} style={{fill}} />;
      return (
        <g>
          <rect x={x} y={y} width={w} height={h} rx={r} style={{fill}} />
          {multi && dogEar(CUBE_FACE_SURFACE)}
        </g>
      );
    }
    case CellState.Zero: {
      const half = Math.min(3, (w - 4) / 2);
      return (
        <g>
          {ring}
          <line x1={cx - half} x2={cx + half} y1={cy} y2={cy} style={{stroke: 'var(--loom-mark)'}} strokeWidth={1} />
          {multi && dogEar('var(--loom-mark)')}
        </g>
      );
    }
    case CellState.NotInEda:
      return (
        <g>
          {ring}
          <line
            x1={x + 2}
            y1={y + 2}
            x2={x + w - 2}
            y2={y + h - 2}
            style={{stroke: 'var(--loom-mark)'}}
            strokeWidth={1}
          />
          {multi && dogEar('var(--loom-mark)')}
        </g>
      );
    case CellState.Dictionary:
    case CellState.Pending:
      return (
        <g>
          {ring}
          {multi && dogEar('var(--loom-mark)')}
        </g>
      );
    case CellState.OtherVisit:
      return <circle cx={cx} cy={y + h / 2} r={1.5} style={{fill: 'var(--loom-mark)'}} />;
    default:
      return null;
  }
}

// One glyph as an inline icon (lists, legends inside the panels).
function GlyphIcon({
  state,
  flags = 0,
  lo = 0,
  N = null,
  metric = 'comp',
  className = ''
}: {
  state: CellStateValue;
  flags?: number;
  lo?: number;
  N?: number | null;
  metric?: Metric;
  className?: string;
}) {
  return (
    <svg
      width={14}
      height={12}
      viewBox="0 0 14 12"
      className={`inline-block flex-shrink-0 ${className}`}
      aria-hidden="true"
    >
      <CubeFaceCellGlyph x={0} y={0} w={14} h={12} state={state} flags={flags} lo={lo} N={N} metric={metric} />
    </svg>
  );
}

// Tier glyph of a cohort, in neutral ink (same grammar as its column header):
// profiled = filled, dictionary only = ring, no dictionary = hatch.
function CubeFaceTierGlyph({tier, className = ''}: {tier: Tier; className?: string}) {
  if (tier === 'none')
    return (
      <span
        className={`loom-fog inline-block h-[12px] w-[10px] flex-shrink-0 align-[-1px] ${className}`}
        aria-hidden="true"
      />
    );
  return (
    <span
      className={`inline-block h-[12px] w-[10px] flex-shrink-0 rounded-[2px] align-[-1px] ${className}`}
      style={
        tier === 'profiled' ? {background: 'var(--loom-ink-fill)'} : {boxShadow: 'inset 0 0 0 1px var(--loom-mark)'}
      }
      aria-hidden="true"
    />
  );
}

// ---------------------------------------------------------------------------
// The cube face: one concept as a cohort x visit-slot matrix, every cell
// resolved with a single-slot lens by THE cell rule (resolveCell). For a
// longitudinal measurement it shows the attrition staircase.
// ---------------------------------------------------------------------------

const LABEL_W = 124;
const CELL_W = 16;
const CELL_H = 14;
const PITCH_X = 18;
const PITCH_Y = 16;
const HEAD_H = 18;
const DIVIDER = 8; // extra gap before the Unanchored column (it is never a timepoint)

const cubeFaceSlotLens = (slot: VisitSlot): VisitLens => ({kind: 'slots', mask: SLOT_BIT[slot]});

const ellipsize = (s: string, max: number) => (s.length > max ? `${s.slice(0, max - 1)}…` : s);

function CubeFace({row, onOpenCell}: {row: number; onOpenCell: (cohort: number) => void}) {
  const {model, columns, state} = useLoom();
  const {metric, floor, guests, visit} = state;

  const face = useMemo(() => {
    // Visible dictionary columns, in grid order, that hold a member of the row.
    const cohorts: number[] = [];
    let slotMask = 0;
    let hiddenCoding = 0;
    const shown = new Set<number>();
    for (const col of columns.columns) {
      if (col.kind !== 'cohort' || !columns.visibleMask[col.cohort]) continue;
      shown.add(col.cohort);
      const members = cellMembers(model, row, col.cohort);
      const home = model.cellHomeCount[row * model.K + col.cohort];
      const counted = guests ? members.length : home;
      if (counted === 0) continue;
      cohorts.push(col.cohort);
      for (let j = 0; j < counted; j++) slotMask |= SLOT_BIT[model.variables[members[j]].slot];
    }
    const rowCohorts = model.rows[row].cohortMask;
    for (let c = 0; c < model.K; c++) if (rowCohorts[c] && !shown.has(c)) hiddenCoding++;
    const slots: VisitSlot[] = VISIT_SLOTS.filter(s => slotMask & SLOT_BIT[s]);
    const grid: ResolvedCell[][] = cohorts.map(c =>
      slots.map(s => resolveCell(model, row, c, cubeFaceSlotLens(s), floor, guests))
    );
    return {cohorts, slots, grid, hiddenCoding};
  }, [model, columns, row, floor, guests]);

  const {cohorts, slots, grid, hiddenCoding} = face;
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [kbd, setKbd] = useState<[number, number] | null>(null);
  // Positions from before a data or filter change may point past the face.
  const inFace = (p: [number, number] | null) => (p && p[0] < cohorts.length && p[1] < slots.length ? p : null);
  const focusAt = inFace(kbd);
  const active = inFace(hover) ?? focusAt;

  if (cohorts.length === 0 || slots.length === 0)
    return <p className="text-xs text-base-content/60">No visible cohort codes this concept at any visit.</p>;

  const unIdx = slots.indexOf('UN');
  const xOf = (si: number) => LABEL_W + si * PITCH_X + (unIdx > 0 && si >= unIdx ? DIVIDER : 0);
  const width = xOf(slots.length - 1) + CELL_W + 2;
  const height = HEAD_H + cohorts.length * PITCH_Y;
  const yOf = (ci: number) => HEAD_H + ci * PITCH_Y;
  const lensActive = !isAllLens(visit);

  const move = (dc: number, ds: number) => {
    const [ci, si] = focusAt ?? [0, 0];
    setHover(null);
    setKbd([Math.max(0, Math.min(cohorts.length - 1, ci + dc)), Math.max(0, Math.min(slots.length - 1, si + ds))]);
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    const steps: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1]
    };
    const step = steps[e.key];
    if (step) {
      e.preventDefault();
      move(step[0], step[1]);
    } else if (e.key === 'Home') {
      e.preventDefault();
      setKbd([focusAt?.[0] ?? 0, 0]);
    } else if (e.key === 'End') {
      e.preventDefault();
      setKbd([focusAt?.[0] ?? 0, slots.length - 1]);
    } else if (e.key === 'Enter' && focusAt) {
      e.preventDefault();
      onOpenCell(cohorts[focusAt[0]]);
    }
  };

  let readout: {title: string; headline: string; stair: string | null} | null = null;
  if (active) {
    const [ci, si] = active;
    const c = cohorts[ci];
    const cohort = model.cohorts[c];
    const r = cellReadout(model, row, c, cubeFaceSlotLens(slots[si]), metric, floor, guests);
    // The cohort's whole row as text: the staircase, readable without hover.
    const stair =
      cohort.tier === 'profiled'
        ? slots
            .map((s, j) => {
              const cell = grid[ci][j];
              let v = '·';
              if (cell.state === CellState.Counted)
                v = metric === 'n' || !cell.N ? fmtCompact(cell.lo) : fmtPct(cell.lo / cell.N);
              else if (cell.state === CellState.Zero) v = '0';
              else if (cell.state === CellState.NotInEda) v = 'not in data';
              else if (cell.state === CellState.Pending) v = '…';
              return `${SLOT_SHORT[s]} ${v}`;
            })
            .join(' · ')
        : null;
    readout = {title: `${cohort.id} · ${SLOT_LONG[slots[si]]}`, headline: r.headline, stair};
  }

  return (
    <div>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-roledescription="cube face"
        tabIndex={0}
        aria-label={`Cube face: ${cohorts.length} cohorts by ${slots.length} visits. Arrow keys move, Enter opens the cell.`}
        aria-describedby={`loom-cube-readout-${row}`}
        className="max-w-full outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--loom-ink)] rounded-sm"
        onKeyDown={onKeyDown}
        onFocus={() => !focusAt && setKbd([0, 0])}
        onBlur={() => setKbd(null)}
        onMouseLeave={() => setHover(null)}
      >
        {/* The current visit lens, as a column wash: the grid shows these slots. */}
        {lensActive &&
          slots.map((s, si) =>
            lensIncludes(visit, s) ? (
              <rect
                key={`lens-${s}`}
                x={xOf(si) - 1}
                y={HEAD_H - 2}
                width={CELL_W + 2}
                height={height - HEAD_H + 2}
                style={{fill: 'var(--loom-wash)'}}
              />
            ) : null
          )}
        {slots.map((s, si) => (
          <text
            key={s}
            x={xOf(si) + CELL_W / 2}
            y={11}
            textAnchor="middle"
            className="loom-tabular"
            style={{
              fill: 'currentColor',
              fontSize: 10,
              opacity: lensActive && !lensIncludes(visit, s) ? 0.5 : 0.8,
              fontWeight: lensActive && lensIncludes(visit, s) ? 600 : 400
            }}
          >
            <title>{SLOT_LONG[s]}</title>
            {SLOT_SHORT[s]}
          </text>
        ))}
        {unIdx > 0 && (
          <line
            x1={xOf(unIdx) - DIVIDER / 2 - 1}
            x2={xOf(unIdx) - DIVIDER / 2 - 1}
            y1={2}
            y2={height}
            strokeDasharray="2 2"
            style={{stroke: 'var(--loom-hairline)'}}
          />
        )}
        {cohorts.map((c, ci) => {
          const cohort = model.cohorts[c];
          return (
            <g key={cohort.id}>
              <text
                x={0}
                y={yOf(ci) + CELL_H - 3}
                style={{
                  fill: 'currentColor',
                  fontSize: 11,
                  opacity: active?.[0] === ci ? 1 : 0.75,
                  fontWeight: active?.[0] === ci ? 600 : 400
                }}
              >
                <title>{cohort.id}</title>
                {ellipsize(cohort.id, 19)}
              </text>
              {slots.map((s, si) => {
                const cell = grid[ci][si];
                return (
                  <g key={s}>
                    <CubeFaceCellGlyph
                      x={xOf(si)}
                      y={yOf(ci)}
                      w={CELL_W}
                      h={CELL_H}
                      state={cell.state === CellState.OtherVisit ? CellState.NotCoded : cell.state}
                      flags={cell.flags}
                      lo={cell.lo}
                      N={cell.N}
                      metric={metric}
                    />
                    <rect
                      x={xOf(si) - 1}
                      y={yOf(ci) - 1}
                      width={PITCH_X}
                      height={PITCH_Y}
                      fill="transparent"
                      className="cursor-pointer"
                      onMouseEnter={() => setHover([ci, si])}
                      onClick={() => onOpenCell(c)}
                    />
                  </g>
                );
              })}
            </g>
          );
        })}
        {active && (
          <rect
            x={xOf(active[1]) - 1.5}
            y={yOf(active[0]) - 1.5}
            width={CELL_W + 3}
            height={CELL_H + 3}
            rx={3}
            pointerEvents="none"
            style={{fill: 'none', stroke: 'var(--loom-ink)'}}
            strokeWidth={2}
          />
        )}
      </svg>
      <div
        id={`loom-cube-readout-${row}`}
        aria-live="polite"
        className="mt-1.5 min-h-[2.5rem] rounded border border-[color:var(--loom-hairline)] px-2 py-1 text-[11px] leading-snug"
      >
        {readout ? (
          <>
            <div>
              <span className="font-semibold">{readout.title}</span>
              <span className="text-base-content/50"> · </span>
              <span className="loom-tabular">{readout.headline}</span>
            </div>
            {readout.stair && (
              <div className="loom-tabular text-base-content/60 truncate" title={readout.stair}>
                {readout.stair}
              </div>
            )}
          </>
        ) : (
          <span className="text-base-content/50">
            Point at a cell, or focus the face and use the arrow keys, to read it. Click or Enter opens the cell.
          </span>
        )}
      </div>
      {metric !== 'pres' && slots.some(s => SLOT_BIT[s] & FOLLOW_UP_MASK) && (
        <p className="mt-1 text-[11px] leading-snug text-base-content/50">{CAVEATS.followUp}</p>
      )}
      {hiddenCoding > 0 && (
        <p className="mt-1 text-[11px] text-base-content/50">
          {hiddenCoding} more {hiddenCoding === 1 ? 'cohort codes' : 'cohorts code'} this concept but{' '}
          {hiddenCoding === 1 ? 'is' : 'are'} hidden by the cohort filters.
        </p>
      )}
    </div>
  );
}

// ============================================================================
// PoolGauge
// ============================================================================

// The Pool gauge (spec §6 "Pooled joint bounds"): how many dataset rows could
// have every pulled concept, as bounds. Solid from 0 to the lower bound
// ("certain", Fréchet), lighter from the lower to the upper bound
// ("possible"). They are counts, so they wear the ramp (bin 3 / bin 1).
// Dictionary-only cohorts and the fog are text lines below, never on the
// gauge: an unknown quantity gets no length.

const poolGaugePlural = (n: number, one: string, many: string) => `${fmtInt(n)} ${n === 1 ? one : many}`;

type Scale = 'linear' | 'log';

function PoolGauge({pool}: {pool: PoolResult}) {
  const [scale, setScale] = useState<Scale>('linear');
  const qualifying = pool.cohorts.filter(c => c.status === 'qualifies');
  // The ceiling: the dataset rows of the qualifying cohorts (their upper
  // bound when the dataset size is unknown).
  const ceiling = qualifying.reduce((s, c) => s + (c.N ?? c.upper), 0);
  const lower = pool.pooledLower;
  const upper = pool.pooledUpper;
  const k = pool.slots.length;
  const allWord = k === 1 ? 'the concept' : k === 2 ? 'both concepts' : `all ${k} concepts`;

  const top = scale === 'log' ? Math.pow(10, Math.max(1, Math.ceil(Math.log10(Math.max(10, ceiling))))) : ceiling;
  const at = (v: number): number => {
    if (top <= 0) return 0;
    if (scale === 'linear') return Math.max(0, Math.min(1, v / top));
    return v <= 1 ? 0 : Math.max(0, Math.min(1, Math.log10(v) / Math.log10(top)));
  };
  const ticks: number[] = [];
  if (scale === 'log') for (let t = 1; t <= top; t *= 10) ticks.push(t);
  else if (ceiling > 0) ticks.push(0, ceiling / 2, ceiling);

  const certainW = at(lower) * 100;
  const possibleW = Math.max(0, at(upper) * 100 - certainW);

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm loom-tabular">
          {qualifying.length === 0 ? (
            <span className="text-base-content/70">
              No profiled cohort has {allWord} at {lensLabel(pool.visit)}.
            </span>
          ) : (
            <>
              <span className="font-semibold">
                {lower === upper ? fmtInt(lower) : `${fmtInt(lower)}–${fmtInt(upper)}`}
              </span>{' '}
              <span className="text-base-content/70">
                dataset rows with {allWord}, in {poolGaugePlural(qualifying.length, 'profiled cohort', 'profiled cohorts')}
              </span>
            </>
          )}
        </p>
        {qualifying.length > 0 && (
          <div className="join flex-shrink-0" role="group" aria-label="Gauge scale">
            {(['linear', 'log'] as Scale[]).map(s => (
              <button
                key={s}
                type="button"
                className={`btn join-item btn-xs font-normal ${scale === s ? 'btn-active' : 'btn-ghost border border-base-300'}`}
                aria-pressed={scale === s}
                onClick={() => setScale(s)}
                title={
                  s === 'linear'
                    ? 'Linear scale from 0'
                    : 'Logarithmic scale (decades), for small bounds next to large cohorts'
                }
              >
                {s === 'linear' ? 'Linear' : 'Log'}
              </button>
            ))}
          </div>
        )}
      </div>

      {qualifying.length > 0 && (
        <div>
          <div
            className="relative h-3.5 w-full overflow-hidden rounded-[2px] border border-[color:var(--loom-hairline)]"
            role="img"
            aria-label={`At least ${fmtInt(lower)} and at most ${fmtInt(upper)} of ${fmtInt(ceiling)} dataset rows have ${allWord}`}
          >
            <div
              className="absolute inset-y-0 left-0"
              style={{width: `${certainW}%`, background: 'var(--loom-bin-3)'}}
            />
            <div
              className="absolute inset-y-0"
              style={{left: `${certainW}%`, width: `${possibleW}%`, background: 'var(--loom-bin-1)'}}
            />
          </div>
          <div className="relative mt-0.5 h-3.5 text-[10px] text-base-content/50 loom-tabular" aria-hidden="true">
            {ticks.map((t, i) => {
              const x = at(t) * 100;
              const edge =
                i === 0 ? 'translate-x-0' : i === ticks.length - 1 ? '-translate-x-full' : '-translate-x-1/2';
              return (
                <span key={t} className={`absolute top-0 ${edge}`} style={{left: `${x}%`}}>
                  {fmtCompact(Math.round(t))}
                </span>
              );
            })}
          </div>
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-[11px] text-base-content/70 loom-tabular">
            <span
              className="inline-flex items-center gap-1.5"
              title="Fréchet lower bound: at least this many rows have every concept"
            >
              <span className="inline-block h-2.5 w-2.5" style={{background: 'var(--loom-bin-3)'}} aria-hidden="true" />
              certain ≥ {fmtInt(lower)}
            </span>
            <span
              className="inline-flex items-center gap-1.5"
              title="Upper bound: the concept with the fewest values caps each cohort"
            >
              <span className="inline-block h-2.5 w-2.5" style={{background: 'var(--loom-bin-1)'}} aria-hidden="true" />
              possible ≤ {fmtInt(upper)}
            </span>
            <span className="text-base-content/50">of {fmtInt(ceiling)} dataset rows in those cohorts</span>
          </div>
        </div>
      )}

      <div className="space-y-0.5 text-[11px] leading-snug text-base-content/70">
        {pool.dictionaryOnly.count > 0 && (
          <p>
            <span className="loom-tabular">
              +{poolGaugePlural(pool.dictionaryOnly.count, 'dictionary-only cohort', 'dictionary-only cohorts')}
            </span>{' '}
            {pool.dictionaryOnly.count === 1 ? 'codes' : 'code'} {allWord}
            {pool.dictionaryOnly.declared > 0 && (
              <span className="loom-tabular"> (~{fmtInt(pool.dictionaryOnly.declared)} declared participants)</span>
            )}
            : counts unknown, not on the gauge.
          </p>
        )}
        {pool.fog.count > 0 && (
          <p>
            <span className="loom-tabular">? {poolGaugePlural(pool.fog.count, 'cohort', 'cohorts')}</span> without a dictionary
            {pool.fog.declared > 0 && (
              <span className="loom-tabular"> (~{fmtCompact(pool.fog.declared)} declared participants)</span>
            )}
            : unknown, not absent.
          </p>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// PoolDock
// ============================================================================

// The Pool (spec §6, §7.5): pulled concepts, and how many patients could be
// pooled across cohorts, as bounds. A 48px dock under the matrix while
// anything is pulled; "Open" grows it into a 60vh sheet (the grid shrinks,
// nothing is covered).

// Basket previews opened from the Pool carry this source prefix, so the
// preview (DcrAddPreview) can offer the Representative / All choice.
const POOL_SOURCE_PREFIX = 'Pool · ';
const isPoolSource = (source: string): boolean => source.startsWith(POOL_SOURCE_PREFIX);

const poolDockPlural = (n: number, one: string, many: string) => `${fmtInt(n)} ${n === 1 ? one : many}`;

// The basket preview of the Pool under the current handoff settings.
function poolPreview(
  model: LoomModel,
  pool: PoolResult,
  state: LoomViewState,
  basket: Record<string, string[]>
): BasketPreview {
  const preview = poolBasketPreview(model, pool, state.poolHandoff, state.poolIncludeDict, basket);
  return {
    ...preview,
    source: `${POOL_SOURCE_PREFIX}${poolDockPlural(pool.slots.length, 'concept', 'concepts')} · ${lensLabel(pool.visit)}`
  };
}

// "has it" / "has both" / "has all 3"
const allWord = (k: number) => (k === 1 ? 'it' : k === 2 ? 'both' : `all ${k}`);

const STATUS_WORD: Record<PoolStatus, string> = {
  qualifies: 'has all',
  nearMiss: 'near miss',
  unknown: 'unknown',
  dictionary: 'dictionary only',
  lacks: 'lacks'
};
// Neutral text glyphs (no icons, no colour): status is read from the word.
const STATUS_GLYPH: Record<PoolStatus, string> = {
  qualifies: '●',
  nearMiss: '◐',
  unknown: '?',
  dictionary: '□',
  lacks: '○'
};
const STATUS_ORDER: PoolStatus[] = ['qualifies', 'nearMiss', 'unknown', 'dictionary', 'lacks'];
const STATUS_GROUP: Record<PoolStatus, string> = {
  qualifies: 'Has every concept',
  nearMiss: 'Near miss: one concept short',
  unknown: 'Unknown: a concept is listed but not in the profiled data',
  dictionary: 'Dictionary only: counts unknown',
  lacks: 'Lacks two or more concepts'
};

const lensValue = (l: VisitLens) => `${l.kind}:${l.mask}`;
const FOLLOW_UP: VisitLens = {kind: 'slots', mask: FOLLOW_UP_MASK};

const BTN = 'btn btn-xs btn-ghost border border-base-300 font-normal';

function CohortLine({pc, pool}: {pc: PoolCohort; pool: PoolResult}) {
  const {model, state, setUi} = useLoom();
  const cohort = model.cohorts[pc.cohort];
  const k = pool.slots.length;
  const at = lensLabel(pool.visit);

  // Why a concept is missing, in words: never shown as 0 when it is unknown.
  const why = (label: string): string => {
    const i = pool.slots.findIndex(s => s.label === label);
    const st = i >= 0 ? pc.perSlot[i]?.state : undefined;
    if (st === 'notInEda') return `not in profiled data at ${at}`;
    if (st === 'zero') return `no values at ${at}`;
    if (st === 'dictionary') return 'count unknown';
    if (st === 'counted') return `below ${state.poolThreshold}% at ${at}`;
    return `not coded at ${at}`;
  };

  let detail: string;
  if (pc.status === 'qualifies') {
    const parts = pool.slots.map((s, i) => {
      const ps = pc.perSlot[i];
      const names = ps.repVars.map(v => model.variables[v].name);
      const vars = names.length > 3 ? `${names.slice(0, 2).join(', ')} +${names.length - 2}` : names.join(', ');
      return `${s.label} ${fmtInt(ps.lo)}${vars ? ` (${vars})` : ''}`;
    });
    detail = parts.join(' · ');
    if (k > 1 && pc.limitingSlot) detail = `limited by ${pc.limitingSlot} · ${detail}`;
  } else if (pc.status === 'nearMiss') detail = pc.missing.map(l => `missing ${l} (${why(l)})`).join('; ');
  else if (pc.status === 'unknown') detail = pc.missing.map(l => `${l} ${why(l)}`).join('; ');
  else if (pc.status === 'dictionary')
    detail =
      pc.missing.length > 0
        ? `count unknown · lacks ${pc.missing.join(', ')}`
        : `codes ${allWord(k)}; not profiled, so counts are unknown${cohort.declared != null ? ` · ~${fmtInt(cohort.declared)} declared` : ''}`;
  else detail = `lacks ${pc.missing.join(', ')}`;

  const counted = pc.status === 'qualifies';
  const N = pc.N;
  return (
    <li className="py-1">
      <div className="grid grid-cols-[14px_minmax(0,11rem)_minmax(0,1fr)_auto] items-center gap-x-2 text-xs">
        <span className="text-center text-base-content/60" aria-hidden="true">
          {STATUS_GLYPH[pc.status]}
        </span>
        <button
          type="button"
          className="truncate text-left hover:underline"
          onClick={() => setUi({inspector: {kind: 'cohort', cohort: pc.cohort}})}
          title={`Open ${cohort.id} in the inspector`}
        >
          {cohort.id}
        </button>
        {counted && N != null && N > 0 ? (
          <span
            className="relative h-2 overflow-hidden rounded-[1px] border border-[color:var(--loom-hairline)]"
            aria-hidden="true"
            title={`${fmtInt(pc.lower)}–${fmtInt(pc.upper)} of ${fmtInt(N)} dataset rows`}
          >
            <span
              className="absolute inset-y-0 left-0"
              style={{width: `${(pc.lower / N) * 100}%`, background: 'var(--loom-bin-3)'}}
            />
            <span
              className="absolute inset-y-0"
              style={{
                left: `${(pc.lower / N) * 100}%`,
                width: `${(Math.max(0, pc.upper - pc.lower) / N) * 100}%`,
                background: 'var(--loom-bin-1)'
              }}
            />
          </span>
        ) : (
          <span />
        )}
        <span className="whitespace-nowrap text-right text-[11px] loom-tabular">
          <span className="text-base-content/60">{STATUS_WORD[pc.status]}</span>
          {counted && (
            <>
              {' · '}
              <span className="font-semibold">
                {pc.lower === pc.upper ? fmtInt(pc.upper) : `${fmtInt(pc.lower)}–${fmtInt(pc.upper)}`}
              </span>
              {N != null && <span className="text-base-content/60"> of {fmtInt(N)}</span>}
            </>
          )}
        </span>
      </div>
      <p className="ml-[22px] truncate text-[11px] text-base-content/60 loom-tabular" title={detail}>
        {detail}
      </p>
    </li>
  );
}

function PoolSheet({pool}: {pool: PoolResult}) {
  const {model, state, dispatch, setUi, basket, previewBasket, revealRow} = useLoom();
  const [showLacks, setShowLacks] = useState(false);

  // Visit choices: Baseline, any visit, any follow-up, then every slot where
  // a pulled concept is coded.
  const slotChoices = useMemo(() => {
    let mask = 0;
    for (const s of pool.slots) for (const r of s.rows) mask |= model.rows[r].slotMask;
    return VISIT_SLOTS.filter(s => s !== 'BL' && mask & SLOT_BIT[s]);
  }, [pool.slots, model]);
  const lensChoices: {value: string; lens: VisitLens; label: string}[] = [
    {value: lensValue(BASELINE_LENS), lens: BASELINE_LENS, label: 'Baseline'},
    {value: lensValue(ALL_LENS), lens: ALL_LENS, label: 'Any visit (best per concept)'},
    {value: lensValue(FOLLOW_UP), lens: FOLLOW_UP, label: 'Any follow-up'}
  ];
  const specific = slotChoices.map(s => {
    const lens: VisitLens = {kind: 'slots', mask: SLOT_BIT[s]};
    return {value: lensValue(lens), lens, label: SLOT_LONG[s]};
  });
  const choices = [...lensChoices, ...specific];
  const chosen = choices.find(c => sameLens(c.lens, state.poolVisit));
  const current = chosen ? chosen.value : lensValue(state.poolVisit);

  const groups = STATUS_ORDER.map(st => ({st, list: pool.cohorts.filter(c => c.status === st)})).filter(
    g => g.list.length > 0
  );

  const copy = () => {
    const done = (text: string) => setUi({toast: {id: Date.now(), text}});
    if (!navigator.clipboard) {
      done('Could not copy: the clipboard is not available here');
      return;
    }
    navigator.clipboard.writeText(pool.summary).then(
      () => done('Pool summary copied'),
      () => done('Could not copy the Pool summary')
    );
  };

  const gains = pool.dropToGain.flatMap(d => {
    const slot = pool.slots.find(s => s.key === d.slot || s.label === d.slot);
    return slot && (d.plusCohorts > 0 || d.plusUpper > 0)
      ? [{slot, plusCohorts: d.plusCohorts, plusUpper: d.plusUpper}]
      : [];
  });

  return (
    <div className="grid gap-x-6 gap-y-4 px-4 py-3 md:grid-cols-[minmax(240px,300px)_minmax(0,1fr)]">
      <div className="space-y-4">
        <section>
          <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-base-content/60">
            Concepts · {pool.slots.length}
          </h3>
          <ul className="space-y-0.5">
            {pool.slots.map(s => (
              <li key={s.key} className="flex items-center gap-1 text-xs">
                <button
                  type="button"
                  className="min-w-0 flex-1 truncate rounded px-1 py-0.5 text-left hover:bg-[color:var(--loom-wash)]"
                  onClick={() => {
                    revealRow(s.rows[0]);
                    setUi({inspector: {kind: 'concept', row: s.rows[0]}});
                  }}
                  title={`Show ${s.label} in the matrix and the inspector`}
                >
                  {s.label}
                  {s.rows.length > 1 && <span className="text-base-content/50"> · {s.rows.length} rows braided</span>}
                </button>
                <button
                  type="button"
                  className="btn btn-ghost btn-xs btn-square"
                  onClick={() => dispatch({type: 'release', key: s.key})}
                  aria-label={`Release ${s.label} from the Pool`}
                  title={`Release ${s.label} from the Pool`}
                >
                  <X size={12} />
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="space-y-2 text-xs">
          <label className="flex items-center justify-between gap-2">
            <span className="text-base-content/70">At visit</span>
            <select
              className="select select-bordered select-xs max-w-[12rem]"
              value={current}
              onChange={e => {
                const pick = choices.find(c => c.value === e.target.value);
                if (pick) dispatch({type: 'patch', patch: {poolVisit: pick.lens}});
              }}
              aria-label="Visit at which every concept must be present"
              title="Visit at which every concept must be present for a cohort to count"
            >
              {lensChoices.map(c => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
              {(specific.length > 0 || !chosen) && (
                <optgroup label="One visit">
                  {specific.map(c => (
                    <option key={c.value} value={c.value}>
                      {c.label}
                    </option>
                  ))}
                  {!chosen && <option value={current}>{lensLabel(state.poolVisit)}</option>}
                </optgroup>
              )}
            </select>
          </label>
          <label className="block">
            <span className="flex items-baseline justify-between">
              <span className="text-base-content/70">Completeness ≥</span>
              <span className="loom-tabular">{state.poolThreshold}%</span>
            </span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={state.poolThreshold}
              onChange={e => dispatch({type: 'patch', patch: {poolThreshold: Number(e.target.value)}})}
              className="range range-xs mt-1"
              aria-label="Minimum completeness for a cohort to count as having a concept"
              title="A cohort counts as having a concept only if its best variable reaches this completeness"
            />
          </label>
        </section>

        <section className="space-y-1.5 text-xs">
          <h3 className="text-[11px] font-semibold uppercase tracking-wide text-base-content/60">Handoff</h3>
          {(['rep', 'all'] as const).map(h => (
            <label key={h} className="flex cursor-pointer items-start gap-2">
              <input
                type="radio"
                name="loom-pool-handoff"
                className="radio radio-xs mt-0.5"
                checked={state.poolHandoff === h}
                onChange={() => dispatch({type: 'patch', patch: {poolHandoff: h}})}
                title={
                  h === 'rep'
                    ? 'Hand off one variable per concept per visit: the one each cell shows'
                    : 'Hand off every variable of these cells'
                }
              />
              <span>
                {h === 'rep' ? 'Representative variables' : 'All variables in these cells'}
                <span className="block text-[11px] text-base-content/50">
                  {h === 'rep' ? '1 per concept per visit' : 'every visit and duplicate under the lens'}
                </span>
              </span>
            </label>
          ))}
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              className="checkbox checkbox-xs"
              checked={state.poolIncludeDict}
              onChange={e => dispatch({type: 'patch', patch: {poolIncludeDict: e.target.checked}})}
              title="Dictionary-only cohorts have known variables but unknown counts; the chain decides access"
            />
            <span>Include dictionary-only cohorts</span>
          </label>
          <div className="flex flex-wrap gap-1.5 pt-1">
            <button
              type="button"
              className="btn btn-sm btn-neutral"
              onClick={() => previewBasket(poolPreview(model, pool, state, basket))}
              title="Preview the variables this Pool would add to your Data Clean Room basket"
            >
              Add to Data Clean Room…
            </button>
            <button
              type="button"
              className={`${BTN} btn-sm`}
              onClick={copy}
              title="Copy a plain-text summary of this Pool (concepts, visit, bounds, cohorts)"
            >
              Copy summary
            </button>
          </div>
        </section>
      </div>

      <div className="min-w-0 space-y-4">
        <p className="text-[11px] leading-snug text-base-content/60">{CAVEATS.pool}</p>
        <PoolGauge pool={pool} />
        {gains.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-base-content/60">Drop to gain:</span>
            {gains.map(({slot, plusCohorts, plusUpper}) => (
              <button
                key={slot.key}
                type="button"
                className="badge badge-sm h-auto gap-1 border border-base-300 bg-base-100 py-0.5 font-normal hover:bg-[color:var(--loom-wash)]"
                onClick={() => dispatch({type: 'release', key: slot.key})}
                title={`Release ${slot.label}: ${poolDockPlural(plusCohorts, 'more cohort', 'more cohorts')} would have every remaining concept, and the upper bound grows by ${fmtInt(plusUpper)}`}
              >
                without {slot.label}
                <span className="loom-tabular text-base-content/60">
                  +{plusCohorts} {plusCohorts === 1 ? 'cohort' : 'cohorts'}, +{fmtCompact(plusUpper)}
                </span>
              </button>
            ))}
          </div>
        )}
        <div className="space-y-3">
          {groups.map(({st, list}) => {
            const collapsed = st === 'lacks' && !showLacks;
            return (
              <section key={st}>
                <h3 className="flex items-baseline gap-2 text-[11px] font-semibold uppercase tracking-wide text-base-content/60">
                  <span>{STATUS_GROUP[st]}</span>
                  <span className="font-normal loom-tabular">{list.length}</span>
                  {st === 'lacks' && (
                    <button
                      type="button"
                      className="link link-hover font-normal normal-case tracking-normal"
                      aria-expanded={showLacks}
                      onClick={() => setShowLacks(s => !s)}
                      title={showLacks ? 'Hide the cohorts lacking concepts' : 'List the cohorts lacking concepts'}
                    >
                      {showLacks ? 'hide' : 'show'}
                    </button>
                  )}
                </h3>
                {!collapsed && (
                  <ul className="divide-y divide-[color:var(--loom-hairline)]">
                    {list.map(pc => (
                      <CohortLine key={pc.cohort} pc={pc} pool={pool} />
                    ))}
                  </ul>
                )}
              </section>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function PoolDock() {
  const {pool, ui, setUi, dispatch, model, state, basket, previewBasket} = useLoom();
  if (!pool) return null;
  const open = ui.poolOpen;
  const k = pool.slots.length;
  const q = pool.qualifying;
  const parts: string[] = [
    poolDockPlural(k, 'concept', 'concepts'),
    lensLabel(pool.visit),
    q === 0
      ? `no profiled cohort has ${allWord(k)}`
      : `${poolDockPlural(q, 'profiled cohort has', 'profiled cohorts have')} ${allWord(k)}`
  ];
  if (q > 0)
    parts.push(
      pool.pooledLower === pool.pooledUpper
        ? fmtInt(pool.pooledUpper)
        : `${fmtInt(pool.pooledLower)}–${fmtInt(pool.pooledUpper)}`
    );
  if (pool.dictionaryOnly.count > 0) parts.push(`+${pool.dictionaryOnly.count} dictionary-only (unknown)`);
  if (pool.fog.count > 0) parts.push(`? ${pool.fog.count} unknown`);
  const line = parts.join(' · ');

  const onKeyDown = (e: React.KeyboardEvent) => {
    // Esc folds the sheet back into the dock (the topmost layer first).
    if (e.key === 'Escape' && open) {
      e.preventDefault();
      setUi({poolOpen: false});
    }
  };

  return (
    <div
      className={`flex flex-shrink-0 flex-col border-t border-base-300 bg-base-100 ${open ? 'h-[60vh]' : ''}`}
      onKeyDown={onKeyDown}
      role="region"
      aria-label="Pool"
    >
      <div className="flex h-12 flex-shrink-0 items-center gap-3 px-4">
        <span className="text-[11px] font-semibold uppercase tracking-wide">Pool</span>
        <p className="min-w-0 flex-1 truncate text-xs text-base-content/80 loom-tabular" title={line}>
          {line}
        </p>
        <button
          type="button"
          className={BTN}
          aria-expanded={open}
          onClick={() => setUi({poolOpen: !open})}
          title={open ? 'Fold the Pool back into the dock (Esc)' : 'Open the Pool: bounds, cohorts, handoff'}
        >
          {open ? 'Close ▾' : 'Open ▴'}
        </button>
        <button
          type="button"
          className="btn btn-xs btn-neutral hidden sm:inline-flex"
          onClick={() => previewBasket(poolPreview(model, pool, state, basket))}
          title="Preview the variables this Pool would add to your Data Clean Room basket"
        >
          Add to Data Clean Room…
        </button>
        <button
          type="button"
          className={BTN}
          onClick={() => {
            dispatch({type: 'clearPool'});
            setUi({poolOpen: false});
          }}
          title="Release every concept from the Pool"
        >
          Clear
        </button>
      </div>
      {open && (
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain border-t border-[color:var(--loom-hairline)]">
          <PoolSheet pool={pool} />
        </div>
      )}
    </div>
  );
}

// ============================================================================
// DcrAddPreview
// ============================================================================

// The basket preview (spec §8 "DCR basket"): every add goes through it. It
// says what will be added, what is already in the basket and where counts
// are unknown; per-cohort checkboxes trim it. It informs, it never gates:
// access is decided downstream, when the Data Clean Room is set up.

const dcrAddPreviewPlural = (n: number, one: string, many: string) => `${fmtInt(n)} ${n === 1 ? one : many}`;

const NAMES_SHOWN = 8;

type Entry = BasketPreview['byCohort'][number];

function CohortChoice({
  entry,
  inBasket,
  checked,
  onToggle
}: {
  entry: Entry;
  inBasket: Set<string>;
  checked: boolean;
  onToggle: () => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const names = showAll ? entry.names : entry.names.slice(0, NAMES_SHOWN);
  const already = entry.names.filter(n => inBasket.has(n)).length;
  return (
    <li className="py-1.5">
      <label className="flex cursor-pointer items-center gap-2 text-sm">
        <input
          type="checkbox"
          className="checkbox checkbox-xs"
          checked={checked}
          onChange={onToggle}
          title={checked ? `Leave ${entry.cohortId} out of this add` : `Include ${entry.cohortId} in this add`}
        />
        <TierGlyph tier={entry.tier} />
        <span className="min-w-0 flex-1 truncate font-medium">{entry.cohortId}</span>
        <span className="whitespace-nowrap text-xs text-base-content/60 loom-tabular">
          {dcrAddPreviewPlural(entry.names.length, 'variable', 'variables')}
          {already > 0 && ` · ${fmtInt(already)} already in`}
          {entry.tier === 'dictionary' && ' · counts unknown'}
        </span>
      </label>
      <p
        className={`ml-6 mt-0.5 text-[11px] leading-relaxed ${checked ? 'text-base-content/70' : 'text-base-content/40'}`}
      >
        {names.map((n, i) => (
          <span key={n}>
            {i > 0 && <span className="text-base-content/30">, </span>}
            <span
              className={`font-mono ${inBasket.has(n) ? 'text-base-content/40' : ''}`}
              title={inBasket.has(n) ? `${n} is already in your basket` : undefined}
            >
              {n}
              {inBasket.has(n) && ' ✓'}
            </span>
          </span>
        ))}
        {entry.names.length > NAMES_SHOWN && (
          <button
            type="button"
            className="link link-hover ml-1"
            onClick={() => setShowAll(s => !s)}
            title={showAll ? 'Show fewer names' : 'Show every variable name'}
          >
            {showAll ? 'fewer' : `+${entry.names.length - NAMES_SHOWN} more`}
          </button>
        )}
        {already === entry.names.length && <span className="ml-1 text-base-content/50">· nothing new</span>}
      </p>
    </li>
  );
}

function PreviewDialog({opened}: {opened: BasketPreview}) {
  const {setUi, model, pool, state, basket, dispatch, commitBasket} = useLoom();
  const ref = useRef<HTMLDialogElement>(null);
  const [excluded, setExcluded] = useState<Set<string>>(() => new Set());

  // A new preview starts with every cohort checked.
  useEffect(() => {
    setExcluded(new Set());
  }, [opened]);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (!dialog.open) dialog.showModal();
    const onClose = () => setUi({basketPreview: null});
    dialog.addEventListener('close', onClose);
    return () => dialog.removeEventListener('close', onClose);
  }, [setUi]);

  // From the Pool, the handoff choices recompute the preview live.
  const fromPool = isPoolSource(opened.source) && pool != null;
  const preview = useMemo(
    () => (fromPool && pool ? poolPreview(model, pool, state, basket) : opened),
    [fromPool, pool, model, state, basket, opened]
  );

  // What is already in the basket is read from the basket itself, so the
  // sentence, the checkboxes and the commit always agree with it.
  const inBasket = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const e of preview.byCohort) map.set(e.cohortId, new Set(basket[e.cohortId] ?? []));
    return map;
  }, [preview, basket]);
  const have = (e: Entry) => inBasket.get(e.cohortId) ?? new Set<string>();
  const chosen = preview.byCohort.filter(c => !excluded.has(c.cohortId) && c.names.length > 0);
  const total = chosen.reduce((s, c) => s + c.names.length, 0);
  const newByCohort: Entry[] = chosen.flatMap(c => {
    const names = c.names.filter(n => !have(c).has(n));
    return names.length ? [{...c, names, alreadyIn: 0}] : [];
  });
  const fresh = newByCohort.reduce((s, c) => s + c.names.length, 0);
  const already = total - fresh;
  const dictCount = chosen.filter(c => c.tier === 'dictionary').length;

  const close = () => ref.current?.close();
  // Only the new variables are committed, so the Undo toast counts what
  // actually changed.
  const commit = () => {
    commitBasket({byCohort: newByCohort, source: preview.source});
    setUi({basketPreview: null});
  };
  const toggle = (id: string) =>
    setExcluded(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <dialog ref={ref} className="modal" aria-labelledby="loom-basket-title">
      <div className="modal-box max-w-lg p-0">
        <div className="flex items-start gap-2 border-b border-[color:var(--loom-hairline)] px-5 pb-3 pt-4">
          <div className="min-w-0 flex-1">
            <h3 id="loom-basket-title" className="text-base font-semibold">
              Add to your Data Clean Room basket
            </h3>
            <p className="truncate text-xs text-base-content/60" title={preview.source}>
              {preview.source}
            </p>
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-sm btn-square"
            onClick={close}
            aria-label="Close"
            title="Close (Esc)"
          >
            <X size={18} />
          </button>
        </div>

        <div className="max-h-[60vh] overflow-y-auto px-5 py-3">
          <p className="text-sm leading-snug" aria-live="polite">
            {total === 0 ? (
              'Nothing selected to add.'
            ) : (
              <>
                Adds <span className="font-semibold loom-tabular">{dcrAddPreviewPlural(total, 'variable', 'variables')}</span> from{' '}
                {dcrAddPreviewPlural(chosen.length, 'cohort', 'cohorts')}
                {already > 0 && (
                  <span className="loom-tabular">
                    {' '}
                    ({fmtInt(already)} already in your basket, {fmtInt(fresh)} new)
                  </span>
                )}
                .
                {dictCount > 0 && (
                  <>
                    {' '}
                    {dcrAddPreviewPlural(dictCount, 'cohort is', 'cohorts are')} dictionary-only: {dictCount === 1 ? 'its' : 'their'}{' '}
                    counts are unknown.
                  </>
                )}
              </>
            )}
          </p>

          {fromPool && (
            <fieldset className="mt-3 space-y-1 rounded border border-[color:var(--loom-hairline)] px-3 py-2 text-xs">
              <legend className="px-1 text-[11px] font-semibold uppercase tracking-wide text-base-content/60">
                From the Pool
              </legend>
              {(['rep', 'all'] as const).map(h => (
                <label key={h} className="flex cursor-pointer items-center gap-2">
                  <input
                    type="radio"
                    name="loom-basket-handoff"
                    className="radio radio-xs"
                    checked={state.poolHandoff === h}
                    onChange={() => dispatch({type: 'patch', patch: {poolHandoff: h}})}
                    title={
                      h === 'rep'
                        ? 'One variable per concept per visit: the one each cell shows'
                        : 'Every variable of these cells under the Pool visit'
                    }
                  />
                  {h === 'rep' ? 'Representative variables (1 per concept per visit)' : 'All variables in these cells'}
                </label>
              ))}
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  className="checkbox checkbox-xs"
                  checked={state.poolIncludeDict}
                  onChange={e => dispatch({type: 'patch', patch: {poolIncludeDict: e.target.checked}})}
                  title="Dictionary-only cohorts have known variables but unknown counts"
                />
                Include dictionary-only cohorts
              </label>
            </fieldset>
          )}

          {preview.byCohort.length > 0 ? (
            <ul className="mt-3 divide-y divide-[color:var(--loom-hairline)]">
              {preview.byCohort.map(entry => (
                <CohortChoice
                  key={entry.cohortId}
                  entry={entry}
                  inBasket={have(entry)}
                  checked={!excluded.has(entry.cohortId)}
                  onToggle={() => toggle(entry.cohortId)}
                />
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-xs text-base-content/60">
              No cohort has variables to add{fromPool ? ' under these Pool settings' : ''}.
            </p>
          )}

          <p className="mt-3 text-[11px] leading-snug text-base-content/50">
            Adding only fills your basket. Access and consent are decided when the Data Clean Room is set up, not here.
            Cohorts without a dictionary cannot be added: none of their variables are known.
          </p>
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-[color:var(--loom-hairline)] px-5 py-3">
          <button type="button" className="btn btn-ghost btn-sm" onClick={close} title="Close without adding">
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-neutral btn-sm"
            onClick={commit}
            disabled={fresh <= 0}
            title={
              fresh > 0
                ? 'Add these variables to your basket (you can undo it)'
                : total > 0
                  ? 'Every selected variable is already in your basket'
                  : 'Select at least one cohort'
            }
          >
            {fresh > 0 ? `Add ${dcrAddPreviewPlural(fresh, 'variable', 'variables')}` : 'Nothing new to add'}
          </button>
        </div>
      </div>
      <form method="dialog" className="modal-backdrop">
        <button aria-label="Close" title="Close">
          close
        </button>
      </form>
    </dialog>
  );
}

function DcrAddPreview() {
  const {ui} = useLoom();
  if (!ui.basketPreview) return null;
  return <PreviewDialog opened={ui.basketPreview} />;
}

// ============================================================================
// LogRangeSlider
// ============================================================================

// Dual-handle slider on a LOG scale (cohort sizes run from 36 to 73,883, so a
// linear track would crush every small cohort into one pixel), with a dot
// strip of every value underneath so the reader sees where the cohorts are
// before dragging. Dragging previews locally and commits on release; the
// arrow keys step through 1-2-5 values and commit at once.

interface RangeDot {
  key: string;
  value: number;
  title: string; // tooltip: cohort name and what the size is
  solid: boolean; // measured (EDA dataset rows) vs declared (outlined)
}

const STOPS = (() => {
  const out: number[] = [];
  for (let e = 1; e <= 5; e++) for (const m of [1, 2, 5]) out.push(m * 10 ** e);
  return out.filter(v => v <= 100000);
})(); // 10, 20, 50, 100, …, 50k, 100k

const TICKS = [10, 100, 1000, 10000, 100000];

// Two significant digits: 1,234 -> 1,200; 56,789 -> 57,000.
const round2 = (v: number) => {
  const p = 10 ** Math.max(0, Math.floor(Math.log10(v)) - 1);
  return Math.round(v / p) * p;
};

function LogRangeSlider({
  value,
  min,
  max,
  onChange,
  dots,
  ariaLabel,
  unit = 'participants'
}: {
  value: [number, number] | null; // null = full range
  min: number;
  max: number;
  onChange: (v: [number, number] | null) => void;
  dots: RangeDot[];
  ariaLabel: string;
  unit?: string;
}) {
  const lmin = Math.log10(min);
  const lspan = Math.log10(max) - lmin;
  const pos = (v: number) => (Math.log10(Math.min(max, Math.max(min, v))) - lmin) / lspan; // 0..1
  const at = (p: number) => 10 ** (lmin + Math.min(1, Math.max(0, p)) * lspan);

  const committed: [number, number] = value ?? [min, max];
  const [draft, setDraft] = useState<[number, number] | null>(null);
  const [lo, hi] = draft ?? committed;
  const trackRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{handle: 0 | 1} | null>(null);
  const handles = useRef<(HTMLDivElement | null)[]>([]);

  // A new value from outside (Reset, URL) drops any stale preview.
  useEffect(() => setDraft(null), [value]);

  const emit = (v: [number, number]) => onChange(v[0] <= min && v[1] >= max ? null : v);

  const valueAt = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return min;
    const v = at((clientX - rect.left) / rect.width);
    return v <= min * 1.02 ? min : v >= max * 0.98 ? max : round2(v);
  };

  const setHandle = (h: 0 | 1, v: number, base: [number, number]): [number, number] =>
    h === 0 ? [Math.min(v, base[1]), base[1]] : [base[0], Math.max(v, base[0])];

  const onPointerDown = (e: React.PointerEvent, handle?: 0 | 1) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const v = valueAt(e.clientX);
    // A press on the track grabs the nearer handle (in log space).
    const h: 0 | 1 = handle ?? (Math.abs(pos(v) - pos(lo)) <= Math.abs(pos(v) - pos(hi)) ? 0 : 1);
    drag.current = {handle: h};
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    handles.current[h]?.focus({preventScroll: true});
    if (handle === undefined) setDraft(setHandle(h, v, [lo, hi]));
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    setDraft(setHandle(drag.current.handle, valueAt(e.clientX), [lo, hi]));
  };
  const onPointerUp = () => {
    if (!drag.current) return;
    drag.current = null;
    if (draft) emit(draft);
  };

  const onKeyDown = (h: 0 | 1) => (e: React.KeyboardEvent) => {
    const cur = h === 0 ? lo : hi;
    let v: number | null = null;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') v = STOPS.find(s => s > cur * 1.001) ?? max;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') v = [...STOPS].reverse().find(s => s < cur * 0.999) ?? min;
    else if (e.key === 'PageUp') v = Math.min(max, cur * 10);
    else if (e.key === 'PageDown') v = Math.max(min, cur / 10);
    else if (e.key === 'Home') v = min;
    else if (e.key === 'End') v = max;
    if (v === null) return;
    e.preventDefault();
    const next = setHandle(h, v, [lo, hi]);
    setDraft(null);
    emit(next);
  };

  // Dots in up to three lanes so near-equal sizes stay individually hoverable.
  const placed = useMemo(() => {
    const sorted = [...dots].sort((a, b) => a.value - b.value);
    const laneEnd: number[] = [];
    return sorted.map(d => {
      const p = pos(d.value);
      let lane = laneEnd.findIndex(end => p - end >= 0.028);
      if (lane < 0) lane = laneEnd.length < 3 ? laneEnd.length : laneEnd.indexOf(Math.min(...laneEnd));
      laneEnd[lane] = p;
      return {...d, p, lane};
    });
    // pos depends only on min / max
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dots, min, max]);

  const label = (v: number, side: 0 | 1) =>
    side === 0
      ? v <= min
        ? 'no minimum'
        : `≥ ${fmtInt(v)} ${unit}`
      : v >= max
        ? 'no maximum'
        : `≤ ${fmtInt(v)} ${unit}`;
  const inRange = (v: number) => (lo <= min || v >= lo) && (hi >= max || v <= hi);
  const shown = dots.filter(d => inRange(d.value)).length;
  const rangeText =
    lo <= min && hi >= max
      ? 'Any size'
      : lo <= min
        ? `≤ ${fmtCompact(hi)}`
        : hi >= max
          ? `≥ ${fmtCompact(lo)}`
          : `${fmtCompact(lo)} – ${fmtCompact(hi)}`;

  return (
    <div className="select-none">
      <div className="mb-1 flex items-baseline justify-between text-[11px]">
        <span className="loom-tabular text-base-content/80">{rangeText}</span>
        <span
          className="loom-tabular text-[10.5px] text-base-content/45"
          title="Cohorts of known size inside the range"
        >
          {shown} of {dots.length}
        </span>
      </div>
      <div
        ref={trackRef}
        className="relative h-5 cursor-pointer touch-none"
        onPointerDown={e => onPointerDown(e)}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <div className="absolute left-0 right-0 top-1/2 h-[2px] -translate-y-1/2 rounded bg-base-content/15" />
        <div
          className="absolute top-1/2 h-[2px] -translate-y-1/2 rounded bg-base-content/60"
          style={{left: `${pos(lo) * 100}%`, width: `${(pos(hi) - pos(lo)) * 100}%`}}
        />
        {([0, 1] as const).map(h => {
          const v = h === 0 ? lo : hi;
          return (
            <div
              key={h}
              ref={el => {
                handles.current[h] = el;
              }}
              role="slider"
              tabIndex={0}
              aria-label={`${ariaLabel}, ${h === 0 ? 'minimum' : 'maximum'}`}
              aria-valuemin={min}
              aria-valuemax={max}
              aria-valuenow={Math.round(v)}
              aria-valuetext={label(v, h)}
              title={`${label(v, h)}. Drag, or use the arrow keys (Page Up / Down: ×10, Home / End: open)`}
              className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 cursor-grab rounded-full border-2 border-base-content/70 bg-base-100 shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-base-content/40 active:cursor-grabbing"
              style={{left: `${pos(v) * 100}%`, zIndex: h === 0 && pos(lo) > 0.9 ? 2 : 1}}
              onPointerDown={e => {
                e.stopPropagation();
                onPointerDown(e, h);
              }}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onKeyDown={onKeyDown(h)}
            />
          );
        })}
      </div>
      <div className="relative mt-0.5 h-[17px]" aria-hidden="true">
        {placed.map(d => (
          <span
            key={d.key}
            title={d.title}
            className={`absolute h-[5px] w-[5px] -translate-x-1/2 rounded-full border transition-opacity ${
              inRange(d.value) ? 'opacity-100' : 'opacity-25'
            }`}
            style={{
              left: `${d.p * 100}%`,
              top: d.lane * 6,
              borderColor: 'var(--loom-ink-fill)',
              backgroundColor: d.solid ? 'var(--loom-ink-fill)' : 'transparent'
            }}
          />
        ))}
      </div>
      <div className="relative h-3 text-[9.5px] text-base-content/40" aria-hidden="true">
        {TICKS.map(t => (
          <span
            key={t}
            className="loom-tabular absolute"
            style={{left: `${pos(t) * 100}%`, transform: `translateX(${t <= min ? 0 : t >= max ? -100 : -50}%)`}}
          >
            {fmtCompact(t)}
          </span>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// FilterRail
// ============================================================================

// The left rail: every cohort (column) and concept (row) filter, each chip
// with a LIVE count of what it would show given the other filters, so the
// reader sees the consequence before clicking. 248px, folds to a 48px strip;
// below 1280px it is a drawer opened from the command bar ("Filters (n)").

const WIDE = '(min-width: 1280px)';

function useIsWide(): boolean {
  const [wide, setWide] = useState(() => typeof window === 'undefined' || window.matchMedia(WIDE).matches);
  useEffect(() => {
    const mq = window.matchMedia(WIDE);
    const on = () => setWide(mq.matches);
    on();
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return wide;
}

const TIERS: Tier[] = ['profiled', 'dictionary', 'none'];
const TYPES: {value: TypeClass; label: string}[] = [
  {value: 'num', label: 'Numeric'},
  {value: 'cat', label: 'Categorical'},
  {value: 'text', label: 'Text'},
  {value: 'date', label: 'Date'}
];
const NO_FLAGS = DEFAULT_VIEW_STATE.flagFilter;

// Toggle one value of a facet list (null = all). Alt-click = only this one (again = all).
function toggleFacet<T extends string>(current: T[] | null, all: T[], value: T, only: boolean): T[] | null {
  if (only) return current && current.length === 1 && current[0] === value ? null : [value];
  const on = new Set(current ?? all);
  if (on.has(value)) on.delete(value);
  else on.add(value);
  const next = all.filter(v => on.has(v));
  return next.length === all.length ? null : next;
}

const COHORT_KEYS = [
  'tiers',
  'fog',
  'eda',
  'families',
  'designs',
  'statuses',
  'size',
  'sizeUnknown',
  'hiddenCohorts',
  'onlyCohorts'
] as const;
const CONCEPT_KEYS = [
  'domains',
  'minCohorts',
  'minProfiledOnly',
  'types',
  'floor',
  'flagFilter',
  'hiddenRows',
  'guests',
  'match',
  'pooledMin'
] as const;

const differs = (state: LoomViewState, keys: readonly (keyof LoomViewState)[]) =>
  keys.some(k => JSON.stringify(state[k]) !== JSON.stringify(DEFAULT_VIEW_STATE[k]));
const defaultsOf = (keys: readonly (keyof LoomViewState)[]): Partial<LoomViewState> =>
  Object.fromEntries(keys.map(k => [k, DEFAULT_VIEW_STATE[k]])) as Partial<LoomViewState>;

// ---------------------------------------------------------------------------
// Small parts
// ---------------------------------------------------------------------------

function Chip({
  on,
  label,
  count,
  title,
  onClick,
  lead
}: {
  on: boolean;
  label: React.ReactNode;
  count?: number;
  title: string;
  onClick: (e: React.MouseEvent) => void;
  lead?: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-pressed={on}
      title={title}
      onClick={onClick}
      className={`inline-flex h-[22px] items-center gap-1 rounded border px-1.5 text-[11px] leading-none transition-colors ${
        on
          ? 'border-base-content/25 bg-base-content/[0.07] text-base-content'
          : 'border-dashed border-base-content/20 text-base-content/45 hover:border-base-content/35 hover:text-base-content/80'
      }`}
    >
      {lead}
      <span className="whitespace-nowrap">{label}</span>
      {count !== undefined && (
        <span className={`loom-tabular text-[10px] ${count === 0 ? 'text-base-content/30' : 'text-base-content/50'}`}>
          {fmtInt(count)}
        </span>
      )}
    </button>
  );
}

function Field({
  label,
  hint,
  right,
  children
}: {
  label: string;
  hint?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-3.5">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[11px] font-medium text-base-content/70" title={hint}>
          {label}
        </span>
        {right}
      </div>
      {children}
    </div>
  );
}

function SectionHead({
  title,
  meta,
  onClear,
  clearTitle
}: {
  title: string;
  meta: string;
  onClear?: () => void;
  clearTitle: string;
}) {
  return (
    <div className="sticky top-0 z-10 -mx-3 flex h-8 items-center justify-between gap-2 border-b border-base-300 bg-base-100 px-3">
      <div className="flex items-baseline gap-2">
        <Caps tone="text-base-content/65">{title}</Caps>
        <span className="loom-tabular text-[10.5px] text-base-content/45">{meta}</span>
      </div>
      {onClear && (
        <button
          type="button"
          className="rounded px-1 text-[10.5px] text-base-content/55 underline decoration-base-content/25 underline-offset-2 hover:text-base-content"
          title={clearTitle}
          onClick={onClear}
        >
          clear
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cohorts
// ---------------------------------------------------------------------------

function CohortSection() {
  const {model, state, dispatch, columns} = useLoom();
  const patch = (p: Partial<LoomViewState>) => dispatch({type: 'patch', patch: p});

  // Leave-one-out counts: cohorts each chip would show given every OTHER cohort filter.
  const counts = useMemo(() => {
    const tally = <K extends string>(without: Partial<LoomViewState>, key: (c: LoomCohort) => K | null) => {
      const s = {...state, ...without};
      const out = new Map<K, number>();
      for (const c of model.cohorts) {
        if (!cohortPasses(model, s, c.i)) continue;
        const k = key(c);
        if (k !== null) out.set(k, (out.get(k) ?? 0) + 1);
      }
      return out;
    };
    return {
      tiers: tally(
        {tiers: {profiled: true, dictionary: true, none: true}, fog: state.fog === 'hide' ? 'fold' : state.fog},
        c => c.tier
      ),
      eda: tally({eda: {v1: true, v2: true}}, c =>
        c.tier === 'profiled' && (c.edaVersion === 'v1' || c.edaVersion === 'v2') ? c.edaVersion : null
      ),
      families: tally({families: null}, c => c.family),
      designs: tally({designs: null}, c => c.design),
      statuses: tally({statuses: null}, c => c.status)
    };
  }, [model, state]);

  const totals = useMemo(() => {
    const by = <K extends string>(key: (c: LoomCohort) => K) => {
      const m = new Map<K, number>();
      model.cohorts.forEach(c => m.set(key(c), (m.get(key(c)) ?? 0) + 1));
      return m;
    };
    return {
      tiers: by(c => c.tier),
      families: by(c => c.family),
      designs: by(c => c.design),
      statuses: by(c => c.status)
    };
  }, [model]);

  const familiesPresent = STUDY_FAMILIES.filter(f => totals.families.has(f));
  const designsPresent = DESIGN_FAMILIES.filter(d => totals.designs.has(d));
  const statusesPresent = COHORT_STATUSES.filter(s => totals.statuses.has(s));
  const rawStudyTypes = useMemo(() => {
    const m = new Map<string, {n: number; family: string}>();
    model.cohorts.forEach(c => {
      const raw = c.studyTypeRaw.trim() || '(empty)';
      const e = m.get(raw);
      if (e) e.n++;
      else m.set(raw, {n: 1, family: FAMILY_LABEL[c.family]});
    });
    return [...m.entries()].sort((a, b) => b[1].n - a[1].n || a[0].localeCompare(b[0]));
  }, [model]);

  const dots: RangeDot[] = useMemo(
    () =>
      model.cohorts
        .filter(c => c.size !== null)
        .map(c => ({
          key: c.id,
          value: c.size as number,
          solid: c.nRows !== null,
          title:
            c.nRows !== null
              ? `${c.id} · ${fmtInt(c.nRows)} dataset rows`
              : `${c.id} · ~${fmtInt(c.declared ?? 0)} declared participants`
        })),
    [model]
  );
  const unknownSize = model.cohorts.length - dots.length;

  const fogShown = state.tiers.none && state.fog !== 'hide';
  const shownCount = columns.visibleMask.reduce((a, b) => a + b, 0) + columns.visibleFog.length;
  const profiledEda = (['v1', 'v2'] as const).filter(v =>
    model.cohorts.some(c => c.tier === 'profiled' && c.edaVersion === v)
  );

  const setTier = (t: Tier, on: boolean) => {
    if (t === 'none') patch({tiers: {...state.tiers, none: on}, fog: on && state.fog === 'hide' ? 'fold' : state.fog});
    else patch({tiers: {...state.tiers, [t]: on}});
  };

  const tierHint: Record<Tier, string> = {
    profiled: 'Dictionary and EDA counts: cells are filled with the metric',
    dictionary: 'Dictionary only: cells are rings, counts unknown',
    none: 'No dictionary: a hatched column, unknown, not absent'
  };

  return (
    <section aria-label="Cohort filters" className="pb-2">
      <SectionHead
        title="Cohorts"
        meta={`${fmtInt(shownCount)} of ${fmtInt(model.cohorts.length)} shown`}
        onClear={differs(state, COHORT_KEYS) ? () => patch(defaultsOf(COHORT_KEYS)) : undefined}
        clearTitle="Clear every cohort filter (tiers, study type, design, status, size, hidden / only)"
      />

      <Field label="Knowledge" hint="What we know about each cohort's variables">
        <div className="flex flex-col gap-0.5">
          {TIERS.map(t => {
            const on = t === 'none' ? fogShown : state.tiers[t];
            const live = counts.tiers.get(t) ?? 0;
            const total = totals.tiers.get(t) ?? 0;
            return (
              <div key={t}>
                <button
                  type="button"
                  role="switch"
                  aria-checked={on}
                  title={`${tierHint[t]}. ${fmtInt(live)} of ${fmtInt(total)} pass the other cohort filters. Click to ${on ? 'hide' : 'show'} them.`}
                  onClick={() => setTier(t, !on)}
                  className="flex h-6 w-full items-center gap-2 rounded px-1 text-left hover:bg-base-content/[0.04]"
                >
                  <TierGlyph tier={t} width={10} height={10} />
                  <span
                    className={`flex-1 text-[12px] ${on ? 'text-base-content' : 'text-base-content/40 line-through decoration-base-content/30'}`}
                  >
                    {TIER_LABEL[t]}
                  </span>
                  <span className="loom-tabular text-[10.5px] text-base-content/50">{fmtInt(live)}</span>
                  <span
                    className={`relative inline-block h-3 w-5 shrink-0 rounded-full border ${
                      on ? 'border-base-content/70 bg-base-content/70' : 'border-base-content/30'
                    }`}
                    aria-hidden="true"
                  >
                    <span
                      className={`absolute top-[1px] h-2 w-2 rounded-full ${on ? 'left-[9px] bg-base-100' : 'left-[1px] bg-base-content/40'}`}
                    />
                  </span>
                </button>
                {t === 'profiled' && state.tiers.profiled && profiledEda.length > 1 && (
                  <div className="mb-1 ml-[22px] flex items-center gap-1">
                    <span className="mr-0.5 text-[10.5px] text-base-content/45">EDA</span>
                    {profiledEda.map(v => (
                      <Chip
                        key={v}
                        on={state.eda[v]}
                        label={v}
                        count={counts.eda.get(v) ?? 0}
                        title={`Profiled cohorts whose EDA output is ${v}${v === 'v1' ? ' (categorical counts derived as rows − empty − coded missing)' : ''}. Alt-click: only ${v}.`}
                        onClick={e =>
                          patch({
                            eda: e.altKey ? {v1: v === 'v1', v2: v === 'v2'} : {...state.eda, [v]: !state.eda[v]}
                          })
                        }
                      />
                    ))}
                  </div>
                )}
                {t === 'none' && fogShown && total > 0 && (
                  <div className="mb-1 ml-[22px]">
                    <Segmented
                      size="xs"
                      ariaLabel="No-dictionary columns"
                      value={state.fog === 'cols' ? 'cols' : 'fold'}
                      onChange={v => patch({fog: v})}
                      options={[
                        {
                          value: 'fold',
                          label: 'Folded',
                          title: 'One 28px hatched column for every cohort without a dictionary'
                        },
                        {
                          value: 'cols',
                          label: 'Columns',
                          title: 'One 8px hatched column per cohort without a dictionary'
                        }
                      ]}
                    />
                  </div>
                )}
                {t === 'none' && !fogShown && total > 0 && (
                  <p className="mb-1 ml-[22px] text-[10.5px] italic text-base-content/45">
                    Hidden: unknown, not absent
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </Field>

      {familiesPresent.length > 1 && (
        <Field label="Study type" hint="Families of the study type field, ordered along the disease continuum">
          <div className="flex flex-wrap gap-1">
            {familiesPresent.map(f => (
              <Chip
                key={f}
                on={!state.families || state.families.includes(f)}
                label={FAMILY_LABEL[f]}
                count={counts.families.get(f) ?? 0}
                title={`${FAMILY_LABEL[f]}: ${fmtInt(counts.families.get(f) ?? 0)} cohorts pass the other filters (${fmtInt(
                  totals.families.get(f) ?? 0
                )} registered). Click to toggle; Alt-click: only this.`}
                onClick={e => patch({families: toggleFacet(state.families, familiesPresent, f, e.altKey)})}
              />
            ))}
          </div>
          <details className="group mt-1">
            <summary
              className="cursor-pointer list-none text-[10.5px] text-base-content/45 hover:text-base-content/70 [&::-webkit-details-marker]:hidden"
              title="The original study-type strings and the family each one falls into"
            >
              <span className="inline-block transition-transform group-open:rotate-90">▸</span> Raw values
            </summary>
            <ul className="mt-1 max-h-40 space-y-0.5 overflow-y-auto text-[10.5px]">
              {rawStudyTypes.map(([raw, e]) => (
                <li key={raw} className="flex gap-2">
                  <span className="loom-tabular w-4 shrink-0 text-right text-base-content/45">{e.n}</span>
                  <span className="min-w-0 flex-1 text-base-content/75">{raw}</span>
                  <span className="shrink-0 text-base-content/40">{e.family}</span>
                </li>
              ))}
            </ul>
          </details>
        </Field>
      )}

      {designsPresent.length > 1 && (
        <Field label="Design">
          <div className="flex flex-wrap gap-1">
            {designsPresent.map(d => (
              <Chip
                key={d}
                on={!state.designs || state.designs.includes(d)}
                label={DESIGN_LABEL[d]}
                count={counts.designs.get(d) ?? 0}
                title={`${DESIGN_LABEL[d]}: ${fmtInt(counts.designs.get(d) ?? 0)} cohorts pass the other filters. Click to toggle; Alt-click: only this.`}
                onClick={e => patch({designs: toggleFacet(state.designs, designsPresent, d, e.altKey)})}
              />
            ))}
          </div>
        </Field>
      )}

      {statusesPresent.length > 1 && (
        <Field label="Status">
          <div className="flex flex-wrap gap-1">
            {statusesPresent.map(s => (
              <Chip
                key={s}
                on={!state.statuses || state.statuses.includes(s)}
                label={STATUS_LABEL[s]}
                count={counts.statuses.get(s) ?? 0}
                title={`${STATUS_LABEL[s]}: ${fmtInt(counts.statuses.get(s) ?? 0)} cohorts pass the other filters. Click to toggle; Alt-click: only this.`}
                onClick={e => patch({statuses: toggleFacet(state.statuses, statusesPresent, s, e.altKey)})}
              />
            ))}
          </div>
        </Field>
      )}

      <Field
        label="Size"
        hint="Dataset rows for profiled cohorts, else the declared participants; log scale"
        right={<span className="text-[10px] text-base-content/40">● rows ○ declared</span>}
      >
        <LogRangeSlider
          ariaLabel="Cohort size"
          min={SIZE_MIN}
          max={SIZE_MAX}
          value={state.size}
          dots={dots}
          onChange={v => patch({size: v})}
        />
        {unknownSize > 0 && (
          <label
            className="mt-1 flex cursor-pointer items-center gap-1.5 text-[11px] text-base-content/70"
            title="Cohorts with neither dataset rows nor a declared participant count"
          >
            <input
              type="checkbox"
              className="checkbox checkbox-xs rounded-[3px]"
              checked={state.sizeUnknown}
              onChange={e => patch({sizeUnknown: e.target.checked})}
            />
            Include {fmtInt(unknownSize)} of unknown size
          </label>
        )}
      </Field>

      <Field label="Cohort list" hint="Hide, show only, or pin a cohort (pins set which concepts are shown)">
        <CohortList />
      </Field>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Concepts
// ---------------------------------------------------------------------------

function ConceptSection() {
  const {model, cells, stats, columns, state, search, filter, dispatch} = useLoom();
  const patch = (p: Partial<LoomViewState>) => dispatch({type: 'patch', patch: p});

  // Leave-one-out row counts, computed on a deferred snapshot so the rail never
  // delays the matrix: each facet re-runs the row filter without itself.
  const snap = useMemo(
    () => ({model, cells, stats, columns, state, search, filter}),
    [model, cells, stats, columns, state, search, filter]
  );
  const d = useDeferredValue(snap);
  const live = useMemo(() => {
    const pulled = new Set(d.state.pool.flatMap(k => k.split('~')));
    const without = (p: Partial<LoomViewState>, active: boolean): Uint8Array =>
      active
        ? filterRows(d.model, d.cells, d.stats, d.columns, {...d.state, ...p}, d.search, pulled).visible
        : d.filter.visible;
    const R = d.model.rows.length;
    const K = d.model.K;

    const domains = new Map<Domain, number>();
    const domVis = without({domains: null}, d.state.domains !== null);
    for (let r = 0; r < R; r++)
      if (domVis[r]) domains.set(d.model.rows[r].domain, (domains.get(d.model.rows[r].domain) ?? 0) + 1);

    // A row counts under every type one of its home members has (in a shown cohort).
    const types = new Map<TypeClass, number>();
    const typeVis = without({types: null}, d.state.types !== null);
    const {cellStart, cellItems, cellHomeCount, variables} = d.model;
    for (let r = 0; r < R; r++) {
      if (!typeVis[r]) continue;
      const seen = new Set<TypeClass>();
      for (let c = 0; c < K; c++) {
        if (!d.columns.visibleMask[c]) continue;
        const key = r * K + c;
        const start = cellStart[key];
        for (let i = start; i < start + cellHomeCount[key]; i++) seen.add(variables[cellItems[i]].type);
      }
      seen.forEach(t => types.set(t, (types.get(t) ?? 0) + 1));
    }

    const flagVis = without({flagFilter: NO_FLAGS}, Object.values(d.state.flagFilter).some(Boolean));
    const flags = {bridged: 0, broad: 0, units: 0, all: 0};
    for (let r = 0; r < R; r++) {
      if (!flagVis[r]) continue;
      const f = d.model.rows[r].flags;
      flags.all++;
      if (f.bridged) flags.bridged++;
      if (f.broad) flags.broad++;
      if (f.unitsDiffer) flags.units++;
    }
    return {domains, types, flags};
  }, [d]);
  const stale = d !== snap;

  const domainsPresent = useMemo(() => DOMAINS.filter(dm => model.rows.some(r => r.domain === dm)), [model]);
  const typesPresent = useMemo(
    () => TYPES.filter(t => model.variables.some(v => v.home >= 0 && v.type === t.value)),
    [model]
  );
  const visibleDict = columns.visibleMask.reduce((a, b) => a + b, 0);
  const guestRows = useMemo(() => model.rows.filter(r => r.flags.hasGuests).length, [model]);

  // Concepts by the number of shown cohorts coding them (1, 2, … k): the "In ≥ k" stepper's histogram.
  // With "count profiled with data only" the filter counts measured coverage, and so does the histogram.
  const coverage = useMemo(() => {
    const cov = state.minProfiledOnly ? stats.covMeasured : stats.covDict;
    const max = Math.max(1, visibleDict);
    const hist = new Array<number>(max + 1).fill(0);
    for (let r = 0; r < cov.length; r++) {
      const k = Math.min(max, cov[r]);
      if (k > 0) hist[k]++;
    }
    return hist;
  }, [stats, visibleDict, state.minProfiledOnly]);
  const histMax = Math.max(1, ...coverage.slice(1));

  const [floor, setFloor] = useState(state.floor);
  useEffect(() => setFloor(state.floor), [state.floor]);
  const floorTimer = useRef<number | undefined>(undefined);
  const onFloor = (v: number) => {
    setFloor(v);
    window.clearTimeout(floorTimer.current);
    floorTimer.current = window.setTimeout(() => patch({floor: v}), 150);
  };
  useEffect(() => () => window.clearTimeout(floorTimer.current), []);

  const setMin = (k: number) => patch({minCohorts: Math.max(1, Math.min(Math.max(1, visibleDict), k))});
  const flagToggle = (key: 'bridged' | 'broad' | 'units' | 'hideBroad') => {
    const next = {...state.flagFilter, [key]: !state.flagFilter[key]};
    // "only broad" and "hide broad" exclude each other.
    if (key === 'broad' && next.broad) next.hideBroad = false;
    if (key === 'hideBroad' && next.hideBroad) next.broad = false;
    patch({flagFilter: next});
  };

  return (
    <section aria-label="Concept filters" className="pb-2">
      <SectionHead
        title="Concepts"
        meta={`${fmtInt(filter.visibleCount)} of ${fmtInt(filter.ledger.totalRows)} shown`}
        onClear={differs(state, CONCEPT_KEYS) ? () => patch(defaultsOf(CONCEPT_KEYS)) : undefined}
        clearTitle="Clear every concept filter (domain, In ≥ k, type, usable floor, flags, hidden concepts, guests, match, pooled)"
      />

      <Field label="Domain" hint="OMOP domain of the concept (each cohort votes once)">
        <div className={`flex flex-wrap gap-1 transition-opacity ${stale ? 'opacity-70' : ''}`}>
          {domainsPresent.map(dm => (
            <Chip
              key={dm}
              lead={<DomainMark domain={dm} size={12} />}
              on={!state.domains || state.domains.includes(dm)}
              label={DOMAIN_LABEL[dm]}
              count={live.domains.get(dm) ?? 0}
              title={`${DOMAIN_LABEL[dm]}: ${fmtInt(live.domains.get(dm) ?? 0)} concepts with the other filters. Click to toggle; Alt-click: only this.`}
              onClick={e => patch({domains: toggleFacet(state.domains, domainsPresent, dm, e.altKey)})}
            />
          ))}
        </div>
      </Field>

      <Field
        label={`In ≥ ${state.minCohorts} ${state.minCohorts === 1 ? 'cohort' : 'cohorts'}`}
        hint="Concepts coded in at least k of the shown dictionary cohorts (under the visit lens)"
        right={
          <span className="flex items-center rounded-md border border-base-content/15">
            <button
              type="button"
              className="h-5 w-5 text-[12px] text-base-content/60 hover:text-base-content disabled:opacity-30"
              aria-label="Fewer cohorts"
              title="Lower k"
              disabled={state.minCohorts <= 1}
              onClick={() => setMin(state.minCohorts - 1)}
            >
              −
            </button>
            <input
              type="number"
              min={1}
              max={Math.max(1, visibleDict)}
              value={state.minCohorts}
              onChange={e => {
                const v = Number(e.target.value);
                if (Number.isFinite(v)) setMin(Math.round(v));
              }}
              aria-label="Minimum number of cohorts"
              title={`Coded in at least this many of the ${visibleDict} shown dictionary cohorts`}
              className="loom-tabular h-5 w-8 border-x border-base-content/15 bg-transparent text-center text-[11px] outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
            />
            <button
              type="button"
              className="h-5 w-5 text-[12px] text-base-content/60 hover:text-base-content disabled:opacity-30"
              aria-label="More cohorts"
              title="Raise k"
              disabled={state.minCohorts >= visibleDict}
              onClick={() => setMin(state.minCohorts + 1)}
            >
              +
            </button>
          </span>
        }
      >
        <div
          className="flex h-5 items-end gap-[1px]"
          role="group"
          aria-label="Concepts by number of cohorts coding them"
        >
          {coverage.slice(1).map((n, i) => {
            const k = i + 1;
            const kept = k >= state.minCohorts;
            return (
              <button
                key={k}
                type="button"
                className="flex h-full min-w-[3px] flex-1 items-end"
                title={`${fmtInt(n)} concept${n === 1 ? '' : 's'} coded in exactly ${k} of the shown cohorts (bar height ∝ √count). Click: In ≥ ${k}`}
                aria-label={`In at least ${k}: ${fmtInt(n)} concepts coded in exactly ${k}`}
                onClick={() => setMin(k)}
              >
                <span
                  className={`block w-full rounded-t-[1px] ${kept ? 'bg-base-content/55' : 'bg-base-content/15'}`}
                  style={{height: n > 0 ? Math.max(1, Math.round(Math.sqrt(n / histMax) * 20)) : 0}}
                />
              </button>
            );
          })}
        </div>
        <div className="mt-0.5 flex justify-between text-[9.5px] text-base-content/40" aria-hidden="true">
          <span>1</span>
          <span>{Math.max(1, visibleDict)} cohorts</span>
        </div>
        <label
          className="mt-1 flex cursor-pointer items-center gap-1.5 text-[11px] text-base-content/70"
          title="Count only profiled cohorts where the concept is measured (n > 0, above the usable floor)"
        >
          <input
            type="checkbox"
            className="checkbox checkbox-xs rounded-[3px]"
            checked={state.minProfiledOnly}
            onChange={e => patch({minProfiledOnly: e.target.checked})}
          />
          Count profiled with data only
        </label>
      </Field>

      {typesPresent.length > 1 && (
        <Field label="Data type" hint="A concept is kept if any of its variables has the type">
          <div className={`flex flex-wrap gap-1 transition-opacity ${stale ? 'opacity-70' : ''}`}>
            {typesPresent.map(t => (
              <Chip
                key={t.value}
                on={!state.types || state.types.includes(t.value)}
                label={t.label}
                count={live.types.get(t.value) ?? 0}
                title={`${t.label}: ${fmtInt(live.types.get(t.value) ?? 0)} concepts have a ${t.label.toLowerCase()} variable. Click to toggle; Alt-click: only this.`}
                onClick={e =>
                  patch({
                    types: toggleFacet(
                      state.types,
                      typesPresent.map(x => x.value),
                      t.value,
                      e.altKey
                    )
                  })
                }
              />
            ))}
          </div>
        </Field>
      )}

      <Field
        label={`Usable ≥ ${floor}%`}
        hint="Cells whose completeness is below this become a 3px floor bar and stop counting toward coverage and the Pool"
      >
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={floor}
          onChange={e => onFloor(Number(e.target.value))}
          className="range range-xs"
          aria-label="Usable completeness floor, percent"
          aria-valuetext={`${floor}%`}
          title="Usable floor: cells below it keep their colour as a 3px bar and stop counting"
        />
        <div className="flex justify-between text-[9.5px] text-base-content/40" aria-hidden="true">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </Field>

      <Field label="Flags" hint="Integrity flags of the concept grouping">
        <div className={`flex flex-wrap gap-1 transition-opacity ${stale ? 'opacity-70' : ''}`}>
          <Chip
            on={state.flagFilter.bridged}
            label="≡ only bridged"
            count={live.flags.bridged}
            title="Only concepts joined through more than one code or OMOP ID (code ↔ ID pairs of the dictionaries)"
            onClick={() => flagToggle('bridged')}
          />
          <Chip
            on={state.flagFilter.broad}
            label="◇ only broad"
            count={live.flags.broad}
            title="Only concepts a cohort codes in 3+ variables at one visit, or 12+ in total"
            onClick={() => flagToggle('broad')}
          />
          <Chip
            on={state.flagFilter.units}
            label="≠u only units differ"
            count={live.flags.units}
            title="Only concepts whose units differ across cohorts"
            onClick={() => flagToggle('units')}
          />
          <Chip
            on={state.flagFilter.hideBroad}
            label="hide ◇ broad"
            count={live.flags.all - live.flags.broad}
            title="Hide broad concepts (the count is what remains)"
            onClick={() => flagToggle('hideBroad')}
          />
        </div>
      </Field>

      {state.hiddenRows.length > 0 && (
        <Field label="Hidden concepts">
          <button
            type="button"
            className="rounded border border-base-content/20 px-1.5 py-[2px] text-[11px] text-base-content/75 hover:border-base-content/40 hover:text-base-content"
            title="Concepts hidden with the eye on their row. Click to show them all again."
            onClick={() => patch({hiddenRows: []})}
          >
            {fmtInt(state.hiddenRows.length)} hidden · show ×
          </button>
        </Field>
      )}

      <Field label="Match on" hint="Which identifiers join variables into one concept">
        <Segmented<MatchMode>
          size="xs"
          ariaLabel="Match on"
          value={state.match}
          onChange={v => patch({match: v})}
          options={[
            {
              value: 'both',
              label: 'Code + OMOP ID',
              title: 'Join on concept codes and OMOP IDs, linked through the code ↔ ID pairs'
            },
            {value: 'code', label: 'Code', title: 'Join on concept codes only'},
            {value: 'omop', label: 'OMOP ID', title: 'Join on OMOP concept IDs only'}
          ]}
        />
      </Field>

      <Field
        label="Pooled ≥"
        hint="Sum over the shown profiled cohorts of each cohort's best variable (the lower bound L)"
      >
        <Segmented<string>
          size="xs"
          ariaLabel="Pooled at least"
          value={String(state.pooledMin)}
          onChange={v => patch({pooledMin: Number(v)})}
          options={[0, 100, 1000, 10000].map(n => ({
            value: String(n),
            label: n === 0 ? 'any' : fmtCompact(n),
            title:
              n === 0
                ? 'No minimum'
                : `Only concepts with at least ${fmtInt(n)} pooled values (a lower bound; not unique participants)`
          }))}
        />
      </Field>

      <div className="mt-3">
        <SwitchRow
          checked={state.guests}
          onChange={v => patch({guests: v})}
          disabled={guestRows === 0}
          label="Show guest variables"
          hint={
            guestRows === 0
              ? 'None in these dictionaries (no legacy code lists)'
              : `${fmtInt(guestRows)} concepts reach extra variables through secondary codes of legacy lists; drawn at 60%, never counted`
          }
          title="Guests: variables that join a concept only through a secondary code of a legacy pipe-separated list"
        />
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Rail
// ---------------------------------------------------------------------------

function RailBody({drawer, onClose}: {drawer: boolean; onClose: () => void}) {
  const {filter} = useLoom();
  return (
    <>
      <div className="flex h-9 shrink-0 items-center justify-between border-b border-base-300 px-3">
        <div className="flex items-baseline gap-2">
          <Caps tone="text-base-content/65">Filters</Caps>
          {filter.chips.length > 0 && (
            <span className="loom-tabular text-[10.5px] text-base-content/45">{filter.chips.length} active</span>
          )}
        </div>
        <button
          type="button"
          className="flex h-6 w-6 items-center justify-center rounded text-base-content/55 hover:bg-base-content/[0.06] hover:text-base-content"
          title={drawer ? 'Close the filters (Esc)' : 'Fold the rail to a 48px strip'}
          aria-label={drawer ? 'Close the filters' : 'Fold the filters rail'}
          onClick={onClose}
        >
          {drawer ? <X size={14} /> : <ChevronsLeft size={14} />}
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 pb-8">
        <CohortSection />
        <ConceptSection />
      </div>
    </>
  );
}

function FilterRail() {
  const {ui, setUi, filter} = useLoom();
  const wide = useIsWide();
  const drawerRef = useRef<HTMLDivElement>(null);

  // Crossing the breakpoint: the inline rail opens, the drawer closes.
  const lastWide = useRef<boolean | null>(null);
  useEffect(() => {
    if (lastWide.current !== null && lastWide.current !== wide) setUi({railOpen: wide});
    lastWide.current = wide;
  }, [wide, setUi]);

  const drawerOpen = !wide && ui.railOpen;
  useEffect(() => {
    if (!drawerOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    drawerRef.current?.querySelector<HTMLElement>('button')?.focus();
    return () => previous?.focus?.();
  }, [drawerOpen]);

  if (!wide) {
    if (!drawerOpen) return null;
    return (
      <div
        className="fixed inset-0 z-[60] flex"
        role="dialog"
        aria-modal="true"
        aria-label="Filters"
        onKeyDown={e => {
          if (e.key === 'Escape') {
            e.stopPropagation();
            setUi({railOpen: false});
          }
          trapTab(e, drawerRef.current);
        }}
      >
        <div ref={drawerRef} className="relative flex h-full w-[320px] max-w-[88vw] flex-col bg-base-100 shadow-2xl">
          <RailBody drawer onClose={() => setUi({railOpen: false})} />
        </div>
        <div className="flex-1 bg-black/30" onClick={() => setUi({railOpen: false})} aria-hidden="true" />
      </div>
    );
  }

  // Folded, the rail takes no room: the view bar's Filters button unfolds it.
  if (!ui.railOpen) return null;

  return (
    <aside className="flex h-full w-[248px] shrink-0 flex-col border-r border-base-300" aria-label="Filters">
      <RailBody drawer={false} onClose={() => setUi({railOpen: false})} />
    </aside>
  );
}

// ============================================================================
// HelpSheet
// ============================================================================

// The "?" sheet: how to read the Loom (the cell grammar drawn with the same
// glyphs as the matrix and the legend), the metrics and their caveats, the
// search syntax and the keyboard. Opens with "?" from anywhere.

const GRAMMAR: {glyph: CellGlyphProps; name: string; meaning: string}[] = [
  {
    glyph: {kind: 'fill', bin: 4},
    name: 'Filled',
    meaning:
      'Profiled, values counted. The blue is the metric’s bin: fixed and absolute, so a cell never changes colour because of a filter.'
  },
  {
    glyph: {kind: 'presence'},
    name: 'Ink fill',
    meaning:
      'Profiled with values, drawn without the ramp: the Presence metric, or completeness with an unknown dataset size.'
  },
  {
    glyph: {kind: 'ring'},
    name: 'Ring',
    meaning: 'In the dictionary, count unknown: the cohort has not been profiled (or counts are still loading).'
  },
  {
    glyph: {kind: 'zero'},
    name: 'Ring and dash',
    meaning: 'Profiled: no values (every row empty or coded missing). A real zero, never drawn blue.'
  },
  {
    glyph: {kind: 'notInEda'},
    name: 'Ring and slash',
    meaning: 'In the dictionary, but not found in the profiled data (renamed, dropped, or added after profiling).'
  },
  {
    glyph: {kind: 'dot'},
    name: 'Dot',
    meaning: 'Only while a visit is selected: not coded at that visit, but coded at another one.'
  },
  {
    glyph: {kind: 'fog'},
    name: 'Hatch',
    meaning:
      'A whole column: the cohort has no dictionary, so nothing is known about its variables (unknown, not absent). The hatch is the only texture on the page and means only “unknown”.'
  },
  {glyph: {kind: 'blank'}, name: 'Blank', meaning: CAVEATS.blank}
];

const MODIFIERS: {glyph: CellGlyphProps; name: string; meaning: string}[] = [
  {
    glyph: {kind: 'fill', bin: 3, dogEar: true},
    name: 'Corner cut',
    meaning: 'Several variables in the cell. The cell shows the best one (the representative), never a sum.'
  },
  {
    glyph: {kind: 'fill', bin: 3, floor: true},
    name: 'Floor bar',
    meaning:
      'Counted, but below “Usable ≥ x%”: same colour, collapsed to a bar, no longer counted toward coverage or the Pool.'
  },
  {glyph: {kind: 'fill', bin: 3, guest: true}, name: 'Small glyph', meaning: CAVEATS.guest}
];

const SEARCH: [string, string][] = [
  ['hemoglobin creatinine', 'every word must match (label, codes, variable names)'],
  ['bnp | natriuretic', 'either word'],
  ['"ejection fraction"', 'the exact phrase'],
  ['-urine', 'exclude a word; any token can be negated'],
  ['d:measurement', 'domain'],
  ['in:TIME-CHF · notin:Believe', 'coded / not coded in a cohort'],
  ['code:loinc:* · omop:3000963', 'concept code (* wildcard) · OMOP ID'],
  ['v:bl · v:6m · v:fu · v:un', 'coded at a visit'],
  ['t:num · t:cat · t:date', 'data type'],
  ['is:bridged · is:broad · is:units · is:suspect', 'integrity flags'],
  ['is:pulled', 'concepts pulled into the Pool'],
  ['unit:mg · cov:>=3 · n:>=1000', 'units · coded in ≥ k shown cohorts · pooled values ≥ n']
];

const KEYS: [string, string][] = [
  ['/', 'Search'],
  ['?', 'This sheet'],
  ['[  ]', 'Previous / next visit'],
  ['Arrows', 'Move one cell (grid focused)'],
  ['Home / End', 'Row start / end'],
  ['PgUp / PgDn', 'Move 20 rows'],
  ['Ctrl+Home', 'Top-left cell'],
  ['Enter', 'Open the inspector'],
  ['Space', 'Pull the row into the Pool, or release it'],
  ['U', 'Unfold visit sub-rows'],
  ['M', 'Cycle the metric'],
  ['P', 'Cycle the pin of the focused column'],
  ['T', 'Table view'],
  ['Esc', 'Close the topmost layer'],
  ['Enter (search)', 'Pull the chosen concept; in Highlight mode, next match (⇧ previous)'],
  ['Shift-click a visit', 'A range of visits'],
  ['⌘ / Ctrl-click a visit', 'Add or remove one visit'],
  ['Alt-click', 'On a chip or cohort: only this one']
];

function Row({glyph, name, meaning}: {glyph: CellGlyphProps; name: string; meaning: string}) {
  return (
    <li className="flex items-start gap-3 py-1">
      <span className="flex h-5 w-9 shrink-0 items-center justify-center">
        <CellGlyph {...glyph} width={20} height={18} />
      </span>
      <span className="text-[12px] leading-snug">
        <span className="font-semibold text-base-content">{name}.</span>{' '}
        <span className="text-base-content/75">{meaning}</span>
      </span>
    </li>
  );
}

function HelpSheet() {
  const {ui, setUi} = useLoom();
  const closeRef = useRef<HTMLButtonElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const open = ui.helpOpen;

  // "?" opens the sheet from anywhere (not while typing).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== '?' || e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTypingTarget(e.target)) return;
      e.preventDefault();
      setUi(prev => ({helpOpen: !prev.helpOpen}));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setUi]);

  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => previous?.focus?.();
  }, [open]);

  if (!open) return null;
  const close = () => setUi({helpOpen: false});

  return (
    <div
      className="modal modal-open z-[1000]"
      role="dialog"
      aria-modal="true"
      aria-labelledby="loom-help-title"
      onMouseDown={close}
      onKeyDown={e => {
        if (e.key === 'Escape') {
          e.stopPropagation();
          close();
        }
        trapTab(e, boxRef.current);
      }}
    >
      <div
        ref={boxRef}
        className="modal-box flex max-h-[88vh] w-[min(920px,96vw)] max-w-none flex-col p-0"
        onMouseDown={e => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center gap-3 border-b border-base-300 px-5 py-3">
          <WovenGlyph />
          <div className="min-w-0 flex-1">
            <h2 id="loom-help-title" className="text-[15px] font-semibold">
              How to read Concept Coverage
            </h2>
            <p className="text-[11.5px] text-base-content/60">
              Rows are concepts (variables of any cohort sharing a concept code or OMOP ID). Columns are cohorts. Each
              cell shows one representative variable.
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            className="btn btn-circle btn-ghost btn-sm"
            onClick={close}
            aria-label="Close"
            title="Close (Esc)"
          >
            <X size={18} />
          </button>
        </div>

        <div className="grid min-h-0 flex-1 grid-cols-1 gap-x-8 gap-y-5 overflow-y-auto px-5 py-4 md:grid-cols-2">
          <section aria-labelledby="loom-help-grammar">
            <Caps>
              <span id="loom-help-grammar">The cell</span>
            </Caps>
            <p className="mt-1.5 text-[12px] italic text-base-content/70">
              The outline says the dictionary lists it; what is inside says what profiling found.
            </p>
            <ul className="mt-1">
              {GRAMMAR.map(g => (
                <Row key={g.name} {...g} />
              ))}
            </ul>
            <Caps className="mt-4">Modifiers</Caps>
            <ul className="mt-1">
              {MODIFIERS.map(g => (
                <Row key={g.name} {...g} />
              ))}
            </ul>
          </section>

          <section className="flex flex-col gap-4" aria-label="Scales and columns">
            <div>
              <Caps>Colour</Caps>
              <p className="mt-1.5 text-[12px] text-base-content/75">
                Blue means only “observations were counted here”. Five fixed bins:
              </p>
              {(['comp', 'n'] as const).map(m => (
                <div key={m} className="mt-2">
                  <div className="text-[11px] text-base-content/60">{METRIC_EXPLAINER[m]}</div>
                  <div className="mt-1 flex gap-[2px]">
                    {BIN_LABELS[m].map((label, i) => (
                      <div key={label} className="flex w-[60px] flex-col gap-0.5">
                        <span className="h-3 rounded-[2px]" style={{backgroundColor: `var(--loom-bin-${i + 1})`}} />
                        <span className="loom-tabular text-[10px] text-base-content/60">{label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
              <p className="mt-2 text-[11.5px] text-base-content/60">
                {METRIC_EXPLAINER.pres}: every counted cell in ink, no ramp; the automatic fallback when counts cannot
                be loaded.
              </p>
            </div>

            <div>
              <Caps>Columns</Caps>
              <ul className="mt-1.5 space-y-1 text-[12px] text-base-content/75">
                <li className="flex items-center gap-2">
                  <TierGlyph tier="profiled" width={10} height={10} />{' '}
                  <b className="font-semibold text-base-content">Profiled</b>: dictionary and EDA counts
                </li>
                <li className="flex items-center gap-2">
                  <TierGlyph tier="dictionary" width={10} height={10} />{' '}
                  <b className="font-semibold text-base-content">Dictionary only</b>: variables known, counts unknown
                </li>
                <li className="flex items-center gap-2">
                  <TierGlyph tier="none" width={10} height={10} />{' '}
                  <b className="font-semibold text-base-content">No dictionary</b>: unknown, not absent
                </li>
                <li>
                  Pins on a column set which concepts are shown:{' '}
                  <b className="font-semibold text-base-content">{PIN_MARK.coded}</b> coded ·{' '}
                  <b className="font-semibold text-base-content">{PIN_MARK.data}</b> with data ·{' '}
                  <b className="font-semibold text-base-content">{PIN_MARK.not}</b> not coded; ALL / ANY / ONLY / ALL
                  BUT ONE in the recipe bar.
                </li>
              </ul>
            </div>

            <div>
              <Caps>Margins</Caps>
              <ul className="mt-1.5 space-y-1 text-[12px] text-base-content/75">
                <li>
                  <b className="font-semibold text-base-content">Cover</b>: solid = cohorts where the concept is
                  measured; outlined = coded but not measured. Scale: the shown dictionary cohorts.
                </li>
                <li>
                  <b className="font-semibold text-base-content">Pooled</b>: solid dot = lower bound, hollow dot = upper
                  bound, log scale. “+?” = dictionary-only cohorts also code it (they add no length). {CAVEATS.pooled}
                </li>
              </ul>
            </div>

            <div>
              <Caps>Domains</Caps>
              <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[11.5px] text-base-content/70">
                {DOMAINS.map(d => (
                  <span key={d} className="inline-flex items-center gap-1">
                    <DomainMark domain={d} /> {DOMAIN_LABEL[d]}
                  </span>
                ))}
              </div>
            </div>
          </section>

          <section className="md:col-span-2" aria-label="Caveats">
            <Caps>What the numbers mean</Caps>
            <ul className="mt-1.5 grid list-disc gap-x-8 gap-y-1 pl-4 text-[12px] text-base-content/75 md:grid-cols-2">
              {[
                CAVEATS.rows,
                CAVEATS.best,
                CAVEATS.pool,
                CAVEATS.followUp,
                CAVEATS.edaV1,
                `Unanchored visits are ${CAVEATS.unanchored}; they are never counted as baseline.`
              ].map(t => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </section>

          <section aria-label="Search syntax">
            <Caps>Search</Caps>
            <table className="mt-1.5 w-full text-[11.5px]">
              <tbody>
                {SEARCH.map(([q, what]) => (
                  <tr key={q} className="align-top">
                    <td className="whitespace-nowrap py-[3px] pr-3 font-mono text-[11px] text-base-content">{q}</td>
                    <td className="py-[3px] text-base-content/65">{what}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-1 text-[11px] text-base-content/55">
              Enter on a suggested concept pulls it into the Pool. Every token has an equivalent control.
            </p>
          </section>

          <section aria-label="Keyboard">
            <Caps>Keyboard</Caps>
            <table className="mt-1.5 w-full text-[11.5px]">
              <tbody>
                {KEYS.map(([k, what]) => (
                  <tr key={k}>
                    <td className="whitespace-nowrap py-[3px] pr-3">
                      <kbd className="kbd kbd-xs font-mono">{k}</kbd>
                    </td>
                    <td className="py-[3px] text-base-content/70">{what}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-1 text-[11px] text-base-content/55">Single keys are ignored while typing in a field.</p>
          </section>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Inspector
// ============================================================================

// The inspector: where a researcher reads the truth behind a mark. A
// non-modal right drawer (400px from 1280px; below that an overlay of
// min(420px, 100vw) with a scrim) in three modes: cell, concept, cohort.
// The page renders it inside a `relative` container spanning the grid area.

// ---------------------------------------------------------------------------
// Small shared pieces
// ---------------------------------------------------------------------------

const TYPE_LABEL: Record<TypeClass, string> = {num: 'Numeric', cat: 'Categorical', text: 'Text', date: 'Date'};

const INSPECTOR_PIN_GLYPH: Record<PinKind | 'none', string> = {none: '○', coded: '✓', data: '✓✓', not: '✕'};
const PIN_WORD: Record<PinKind | 'none', string> = {
  none: 'not pinned',
  coded: 'coded',
  data: 'coded with data',
  not: 'not coded'
};

const INSPECTOR_BTN = 'btn btn-xs btn-ghost border border-base-300 font-normal';

const UNIT_PLACEHOLDERS = new Set(['', 'na', 'n/a', 'nan', 'null', 'none', '-']);

const inspectorPlural = (n: number, one: string, many: string) => `${fmtInt(n)} ${n === 1 ? one : many}`;

const exploreHref = (cohortId: string, tab: 'list' | 'eda') => ({pathname: '/cohorts', query: {cohort: cohortId, tab}});

function Monogram({domain}: {domain: Domain}) {
  return (
    <span
      className="inline-flex h-[14px] w-[14px] flex-shrink-0 items-center justify-center rounded-[2px] border border-[color:var(--loom-hairline)] text-[9px] font-semibold leading-none text-base-content/70"
      title={DOMAIN_LABEL[domain]}
      aria-hidden="true"
    >
      {DOMAIN_MONOGRAM[domain]}
    </span>
  );
}

function Section({title, note, children}: {title: string; note?: React.ReactNode; children: React.ReactNode}) {
  return (
    <section className="border-t border-[color:var(--loom-hairline)] px-4 py-3">
      <h3 className="mb-2 flex flex-wrap items-baseline gap-x-2 text-[11px] font-semibold uppercase tracking-wide text-base-content/60">
        <span>{title}</span>
        {note && <span className="font-normal normal-case tracking-normal text-base-content/50">{note}</span>}
      </h3>
      {children}
    </section>
  );
}

// The words for a count that is not a number: never drawn as zero.
function uncountedWords(v: LoomVariable, status: CountsStatus): string | null {
  if (v.n === -1) return 'not in profiled data';
  if (v.n === -2) return 'not profiled';
  if (v.n === -3) return status === 'loading' ? 'counts loading…' : 'counts unavailable';
  return null;
}

// Completeness of one variable as a thin bar over the dataset rows: values in
// the bin colour, coded missing as an outlined segment, empty as surface. No
// bar when the dataset size is unknown (an unknown quantity gets no length).
function CompletenessBar({v, N, metric}: {v: LoomVariable; N: number | null; metric: Metric}) {
  if (N == null || N <= 0 || v.n < 0) return null;
  const frac = (x: number) => Math.max(0, Math.min(1, x / N));
  const nw = frac(v.n);
  const mw = Math.min(frac(v.missing), 1 - nw);
  return (
    <div
      className="relative mt-1 ml-auto h-[6px] w-[76px] overflow-hidden rounded-[1px] border border-[color:var(--loom-hairline)]"
      title={`${fmtInt(v.n)} with a value · ${fmtInt(v.missing)} coded missing · ${fmtInt(v.empty)} empty · of ${fmtInt(N)} dataset rows`}
      aria-hidden="true"
    >
      <div className="absolute inset-y-0 left-0" style={{width: `${nw * 100}%`, background: fillFor(metric, v.n, N)}} />
      {mw > 0 && (
        <div
          className="absolute inset-y-0"
          style={{left: `${nw * 100}%`, width: `${mw * 100}%`, boxShadow: 'inset 0 0 0 1px var(--loom-mark)'}}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cell mode
// ---------------------------------------------------------------------------

function MemberRows({
  v,
  row,
  cohort,
  isLensRep,
  isSlotRep,
  slotBestRank,
  inLens,
  guest,
  open,
  onToggle
}: {
  v: LoomVariable;
  row: LoomRow;
  cohort: LoomCohort;
  isLensRep: boolean;
  isSlotRep: boolean;
  slotBestRank: number;
  inLens: boolean;
  guest: boolean;
  open: boolean;
  onToggle: () => void;
}) {
  const {model, state, basket, previewBasket, openSemanticMatches, openDistribution} = useLoom();
  const N = cohort.nRows;
  const words = uncountedWords(v, model.countsStatus);
  const inBasket = (basket[cohort.id] ?? []).includes(v.name);
  const star = isLensRep ? '★' : isSlotRep ? '☆' : '';
  const starTitle = isLensRep
    ? 'Representative: the variable this cell shows under the current visit lens'
    : isSlotRep
      ? `Best variable at ${SLOT_LONG[v.slot]}`
      : undefined;

  // Step 9: within a visit only the best eligibility rank present may
  // represent it, so a flag or a date never outranks the measurement.
  const notCompared = !guest && v.eligibleRank > slotBestRank;
  const notes: string[] = [];
  if (guest)
    notes.push(
      'Guest: carries this concept only as a secondary code (a legacy pipe-separated list). Drawn smaller; excluded from coverage and pooled numbers.'
    );
  if (notCompared) {
    if (v.eligibleRank === 2)
      notes.push('Not compared: a date represents its visit only when every variable there is a date.');
    else
      notes.push(
        `Not compared: its type or domain differs from the concept's (${TYPE_LABEL[row.type]} · ${DOMAIN_LABEL[row.domain]}), and a matching variable exists at this visit.`
      );
  }
  if (v.n === -1)
    notes.push('In the dictionary, but not found in the profiled data (renamed, dropped, or added after profiling).');
  if (v.caseCollision) notes.push(`Name differs only by case from ${v.caseCollision}; the count may belong to either.`);
  if (v.declaredCount && cohort.tier !== 'profiled')
    notes.push(`Declared by the data owner: count ${v.declaredCount} (unverified; not used for colour).`);
  if (cohort.edaVersion === 'v1' && v.type === 'cat' && v.n >= 0) notes.push(CAVEATS.edaV1);

  // Raw visit, units, type and the missing-value split, readable without
  // opening the row.
  const meta = [
    v.slotRaw || 'no visit given',
    v.units,
    TYPE_LABEL[v.type],
    v.n >= 0 ? `${fmtInt(v.empty)} empty · ${fmtInt(v.missing)} coded missing` : ''
  ]
    .filter(Boolean)
    .join(' · ');

  const addThis = () =>
    previewBasket({
      byCohort: [{cohortId: cohort.id, names: [v.name], alreadyIn: inBasket ? 1 : 0, tier: cohort.tier}],
      source: `${v.name} · ${cohort.id}`
    });

  return (
    <>
      <tr
        className={`border-t border-[color:var(--loom-hairline)] align-top ${open ? 'bg-[color:var(--loom-wash)]' : ''} ${
          inLens ? '' : 'text-base-content/60'
        }`}
      >
        <td className="w-5 pt-1.5 text-center text-[11px] leading-none" title={starTitle}>
          {star && <span aria-label={starTitle}>{star}</span>}
        </td>
        <td className="max-w-0 py-1 pr-2">
          <button
            type="button"
            className="block w-full min-w-0 text-left"
            aria-expanded={open}
            onClick={onToggle}
            title={open ? `Hide the details of ${v.name}` : `Show the details and actions of ${v.name}`}
          >
            <span className="block truncate font-mono text-xs">
              {v.name}
              {guest && <span className="ml-1 font-sans text-[10px] text-base-content/50">guest</span>}
              {notCompared && <span className="ml-1 font-sans text-[10px] text-base-content/50">not compared</span>}
              {inBasket && <span className="ml-1 font-sans text-[10px] text-base-content/50">· in basket</span>}
            </span>
            <span className="block truncate text-[11px] text-base-content/60">{v.label || '—'}</span>
            <span className="block truncate text-[10px] text-base-content/50 loom-tabular">{meta}</span>
          </button>
        </td>
        <td className="whitespace-nowrap py-1 text-right loom-tabular">
          {words ? (
            <span className="whitespace-normal text-[11px] italic text-base-content/60">{words}</span>
          ) : (
            <>
              <div className="text-xs">
                {fmtInt(v.n)}
                {N != null && <span className="text-base-content/50"> / {fmtInt(N)}</span>}
              </div>
              <div className="text-[11px] text-base-content/60">
                {N ? (v.n === 0 ? 'no values' : fmtPct(v.n / N)) : 'dataset size unknown'}
              </div>
              <CompletenessBar v={v} N={N} metric={state.metric} />
            </>
          )}
        </td>
      </tr>
      {open && (
        <tr className="bg-[color:var(--loom-wash)]">
          <td />
          <td colSpan={2} className="pb-2 pr-1">
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-[11px]">
              <dt className="text-base-content/50">Visit</dt>
              <dd>
                {v.slotRaw || <span className="text-base-content/40">none given</span>}
                <span className="text-base-content/50"> → {SLOT_LONG[v.slot]}</span>
              </dd>
              <dt className="text-base-content/50">Domain</dt>
              <dd>{v.domainRaw || DOMAIN_LABEL[v.domain]}</dd>
              {v.n >= 0 && N != null && (
                <>
                  <dt className="text-base-content/50">Dataset rows</dt>
                  <dd className="loom-tabular">
                    {fmtInt(v.n)} with a value · {fmtInt(v.missing)} coded missing · {fmtInt(v.empty)} empty
                    {v.n + v.missing + v.empty !== N && (
                      <span className="text-base-content/50"> · {fmtInt(N)} in all</span>
                    )}
                  </dd>
                </>
              )}
              {v.conceptCode && (
                <>
                  <dt className="text-base-content/50">Concept code</dt>
                  <dd className="break-all font-mono">{v.conceptCode}</dd>
                </>
              )}
              {v.omopId && (
                <>
                  <dt className="text-base-content/50">OMOP ID</dt>
                  <dd className="break-all font-mono">{v.omopId}</dd>
                </>
              )}
            </dl>
            {notes.length > 0 && (
              <ul className="mt-1.5 space-y-0.5 text-[11px] leading-snug text-base-content/70">
                {notes.map(n => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            )}
            <div className="mt-2 flex flex-wrap gap-1">
              <button
                type="button"
                className={INSPECTOR_BTN}
                onClick={addThis}
                title={
                  inBasket
                    ? `${v.name} is already in your Data Clean Room basket`
                    : `Preview adding ${v.name} of ${cohort.id} to the Data Clean Room basket`
                }
              >
                {inBasket ? 'In basket ✓' : 'Add to Data Clean Room…'}
              </button>
              <button
                type="button"
                className={INSPECTOR_BTN}
                onClick={() => openSemanticMatches(v.i)}
                title={`Variables of other cohorts sharing ${v.name}'s concept code or OMOP ID`}
              >
                Semantic matches…
              </button>
              {cohort.tier === 'profiled' && (
                <button
                  type="button"
                  className={INSPECTOR_BTN}
                  onClick={() => openDistribution(v.i)}
                  disabled={v.n < 0}
                  title={
                    v.n < 0 ? `${v.name} is not in the profiled data: no distribution` : `Distribution of ${v.name}`
                  }
                >
                  Distribution
                </button>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function CellMode({row, cohort, go}: {row: number; cohort: number; go: (t: InspectorTarget) => void}) {
  const {model, cells, state, basket, previewBasket} = useLoom();
  const r = model.rows[row];
  const c = model.cohorts[cohort];
  const key = row * model.K + cohort;
  const {visit, metric, floor, guests} = state;

  const readout = useMemo(
    () => cellReadout(model, row, cohort, visit, metric, floor, guests),
    [model, row, cohort, visit, metric, floor, guests]
  );
  const members = useMemo(() => cellMembers(model, row, cohort), [model, row, cohort]);
  const homeCount = model.cellHomeCount[key];
  const lensRep = cells.rep[key];

  // Home members grouped by visit slot (slot order), each slot with its own
  // representative (the same rule under a single-slot lens); guests last.
  const groups = useMemo(() => {
    const bySlot = new Map<VisitSlot, number[]>();
    for (let j = 0; j < homeCount; j++) {
      const v = members[j];
      const s = model.variables[v].slot;
      const list = bySlot.get(s);
      if (list) list.push(v);
      else bySlot.set(s, [v]);
    }
    return VISIT_SLOTS.filter(s => bySlot.has(s)).map(s => {
      const rep = resolveCell(model, row, cohort, {kind: 'slots', mask: SLOT_BIT[s]}, floor, guests).rep;
      // The visit's representative first, then the variables that could
      // represent it, then those that are not compared.
      const vars = (bySlot.get(s) as number[]).sort((a, b) => {
        const va = model.variables[a];
        const vb = model.variables[b];
        return +(b === rep) - +(a === rep) || va.eligibleRank - vb.eligibleRank || vb.n - va.n;
      });
      return {slot: s, vars, rep, bestRank: Math.min(...vars.map(v => model.variables[v].eligibleRank))};
    });
  }, [model, row, cohort, members, homeCount, floor, guests]);
  const guestVars = members.slice(homeCount);

  const [open, setOpen] = useState<number | null>(lensRep >= 0 ? lensRep : (members[0] ?? null));

  // "All variables in the cell" = the members that decide it under the lens
  // (home members at the lens slots, else guests), as the grid draws it.
  const lensNames = useMemo(
    () => Array.from(new Set(lensMembers(model, row, cohort, visit, guests).members.map(v => model.variables[v].name))),
    [model, row, cohort, visit, guests]
  );
  const lensWord = isAllLens(visit) ? '' : ` at ${lensLabel(visit)}`;
  const addAll = () =>
    previewBasket({
      byCohort: [
        {cohortId: c.id, names: lensNames, alreadyIn: countAlreadyIn({cohorts: basket}, c.id, lensNames), tier: c.tier}
      ],
      source: `${r.label} · ${c.id}${lensWord}`
    });

  const renderMember = (v: number, slotRep: number, bestRank: number, guest: boolean) => {
    const variable = model.variables[v];
    return (
      <MemberRows
        key={v}
        v={variable}
        row={r}
        cohort={c}
        isLensRep={v === lensRep}
        isSlotRep={v === slotRep}
        slotBestRank={bestRank}
        inLens={lensIncludes(visit, variable.slot)}
        guest={guest}
        open={open === v}
        onToggle={() => setOpen(o => (o === v ? null : v))}
      />
    );
  };

  return (
    <>
      <div className="px-4 pb-3 pt-2">
        <div className="mb-1.5 flex flex-wrap items-center gap-1 text-[11px]">
          <button
            type="button"
            className="inline-flex max-w-[60%] items-center gap-1 rounded border border-[color:var(--loom-hairline)] px-1.5 py-0.5 hover:bg-[color:var(--loom-wash)]"
            onClick={() => go({kind: 'concept', row})}
            title={`Open the concept ${r.label}`}
          >
            <Monogram domain={r.domain} />
            <span className="truncate">{r.label}</span>
            <span aria-hidden="true">›</span>
          </button>
          <button
            type="button"
            className="inline-flex max-w-[40%] items-center gap-1 rounded border border-[color:var(--loom-hairline)] px-1.5 py-0.5 hover:bg-[color:var(--loom-wash)]"
            onClick={() => go({kind: 'cohort', cohort})}
            title={`Open the cohort ${c.id}`}
          >
            <TierGlyph tier={c.tier} />
            <span className="truncate">{c.id}</span>
            <span aria-hidden="true">›</span>
          </button>
          {!isAllLens(visit) && (
            <span className="rounded bg-[color:var(--loom-wash)] px-1.5 py-0.5 text-base-content/70" title="Visit lens">
              {lensLabel(visit)}
            </span>
          )}
        </div>
        <p className="text-sm font-semibold leading-snug loom-tabular">{readout.headline}</p>
        {readout.lines.length > 0 && (
          <ul className="mt-1.5 space-y-0.5 text-[11px] leading-snug text-base-content/70 loom-tabular">
            {readout.lines.map((l: string, i: number) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        )}
        {readout.footer.length > 0 && (
          <div className="mt-2 space-y-0.5 text-[10px] leading-snug text-base-content/50">
            {readout.footer.map((f: string, i: number) => (
              <p key={i}>{f}</p>
            ))}
          </div>
        )}
      </div>

      {members.length === 0 ? (
        c.nUnmapped > 0 && (
          <Section title="Where it might be">
            <p className="text-xs leading-snug text-base-content/70">
              {inspectorPlural(c.nUnmapped, 'variable', 'variables')} of {c.id} {c.nUnmapped === 1 ? 'carries' : 'carry'} no
              valid concept code or OMOP ID: the concept may be among them.{' '}
              <Link
                href={exploreHref(c.id, 'list')}
                className="link link-hover"
                title={`Review the variables of ${c.id} on the Explore page`}
              >
                Review in Explore
              </Link>
            </p>
          </Section>
        )
      ) : (
        <Section
          title="Variables"
          note={
            <>
              {inspectorPlural(homeCount, 'variable', 'variables')}
              {guestVars.length > 0 && ` + ${guestVars.length} guest${guestVars.length === 1 ? '' : 's'}`}
              {lensRep >= 0 && ' · ★ shown in the cell'}
              {groups.length > 1 && ' · ☆ best at its visit'}
            </>
          }
        >
          <table className="w-full table-fixed text-xs">
            <caption className="sr-only">
              Variables of {c.id} coded with {r.label}, grouped by visit
            </caption>
            <colgroup>
              <col className="w-5" />
              <col />
              <col className="w-[92px]" />
            </colgroup>
            <thead className="sr-only">
              <tr>
                <th scope="col">Representative</th>
                <th scope="col">Variable</th>
                <th scope="col">Values of dataset rows</th>
              </tr>
            </thead>
            {groups.map(g => (
              <tbody key={g.slot}>
                <tr>
                  <th
                    scope="rowgroup"
                    colSpan={3}
                    className="pb-0.5 pt-2 text-left text-[11px] font-semibold text-base-content/70"
                  >
                    {SLOT_LONG[g.slot]}
                    <span className="font-normal text-base-content/50">
                      {' · '}
                      {inspectorPlural(g.vars.length, 'variable', 'variables')}
                      {!lensIncludes(visit, g.slot) && ` · outside the ${lensLabel(visit)} lens`}
                    </span>
                  </th>
                </tr>
                {g.vars.map(v => renderMember(v, g.rep, g.bestRank, false))}
              </tbody>
            ))}
            {guestVars.length > 0 && (
              <tbody>
                <tr>
                  <th
                    scope="rowgroup"
                    colSpan={3}
                    className="pb-0.5 pt-2 text-left text-[11px] font-semibold text-base-content/70"
                  >
                    Guests
                    <span className="font-normal text-base-content/50">
                      {' · secondary code'}
                      {!guests && ' · hidden in the grid (Guests off)'}
                    </span>
                  </th>
                </tr>
                {guestVars.map(v => renderMember(v, -1, 0, true))}
              </tbody>
            )}
          </table>
        </Section>
      )}

      <Section title="Actions">
        <div className="flex flex-wrap gap-1.5">
          {c.tier === 'none' && (
            <p className="w-full text-xs text-base-content/60">
              No variables are known for this cohort, so nothing can be added to a Data Clean Room.
            </p>
          )}
          {lensNames.length > 0 && (
            <button
              type="button"
              className="btn btn-sm btn-neutral"
              onClick={addAll}
              title={`Preview adding the ${lensNames.length} variable${lensNames.length === 1 ? '' : 's'} of this cell${lensWord} to the Data Clean Room basket`}
            >
              {lensNames.length === 1
                ? `Add ${lensNames[0]} to Data Clean Room…`
                : `Add all ${lensNames.length}${lensWord} to Data Clean Room…`}
            </button>
          )}
          <Link
            href={exploreHref(c.id, 'list')}
            className={`${INSPECTOR_BTN} btn-sm`}
            title={`Open the variables list of ${c.id} on the Explore page`}
          >
            Open in Explore
          </Link>
          {c.tier === 'profiled' && (
            <Link
              href={exploreHref(c.id, 'eda')}
              className={`${INSPECTOR_BTN} btn-sm`}
              title={`Open the profiling (EDA) of ${c.id} on the Explore page`}
            >
              Open EDA
            </Link>
          )}
        </div>
      </Section>
    </>
  );
}

// ---------------------------------------------------------------------------
// Concept mode
// ---------------------------------------------------------------------------

interface Contribution {
  cohort: number;
  state: CellStateValue;
  lo: number;
  hi: number;
  N: number | null;
  flags: number;
  rep: number;
  otherSlots: number;
}

function contributions(model: LoomModel, cells: LensCells, columns: ColumnLayout, row: number) {
  const out = {
    counted: [] as Contribution[],
    zero: [] as Contribution[],
    notInEda: [] as Contribution[],
    pending: [] as Contribution[],
    otherVisit: [] as Contribution[],
    dictionary: [] as Contribution[],
    notCoded: [] as number[]
  };
  for (const col of columns.columns) {
    if (col.kind !== 'cohort' || !columns.visibleMask[col.cohort]) continue;
    const c = col.cohort;
    const key = row * model.K + c;
    const item: Contribution = {
      cohort: c,
      state: cells.state[key] as CellStateValue,
      lo: cells.lo[key],
      hi: cells.hi[key],
      N: model.cohorts[c].nRows,
      flags: cells.flags[key],
      rep: cells.rep[key],
      otherSlots: cells.otherSlots[key]
    };
    if (item.state === CellState.Counted) out.counted.push(item);
    else if (item.state === CellState.Zero) out.zero.push(item);
    else if (item.state === CellState.NotInEda) out.notInEda.push(item);
    else if (item.state === CellState.Pending) out.pending.push(item);
    else if (item.state === CellState.OtherVisit) out.otherVisit.push(item);
    else if (item.state === CellState.Dictionary) out.dictionary.push(item);
    else if (item.state === CellState.NotCoded) out.notCoded.push(c);
  }
  out.counted.sort((a, b) => b.lo - a.lo || model.cohorts[a.cohort].id.localeCompare(model.cohorts[b.cohort].id));
  return out;
}

function ContributionLine({
  item,
  maxLo,
  onOpen,
  children
}: {
  item: Contribution;
  maxLo: number;
  onOpen: () => void;
  children?: React.ReactNode;
}) {
  const {model, state} = useLoom();
  const cohort = model.cohorts[item.cohort];
  const rep = item.rep >= 0 ? model.variables[item.rep] : null;
  const counted = item.state === CellState.Counted;
  const below = (item.flags & CellFlag.BelowFloor) !== 0;
  const via = rep ? `via ${rep.name} (${SLOT_LONG[rep.slot]})` : '';
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className="grid w-full grid-cols-[14px_minmax(0,1fr)_88px_auto] items-center gap-x-2 rounded px-1 py-0.5 text-left hover:bg-[color:var(--loom-wash)]"
        title={`Open the cell of ${cohort.id}${via ? ` · ${via}` : ''}${
          counted && item.hi > item.lo ? ` · at most ${fmtInt(item.hi)} rows have any value` : ''
        }`}
      >
        <GlyphIcon state={item.state} flags={item.flags} lo={item.lo} N={item.N} metric={state.metric} />
        <span className="truncate text-xs">{cohort.id}</span>
        {counted ? (
          <span className="relative h-[8px]" aria-hidden="true">
            <span
              className="absolute left-0"
              style={{
                width: `${Math.max(2, (item.lo / Math.max(1, maxLo)) * 88)}px`,
                background: fillFor(state.metric, item.lo, item.N),
                top: below ? 5 : 0,
                height: below ? 3 : 8
              }}
            />
          </span>
        ) : (
          <span />
        )}
        <span className="whitespace-nowrap text-right text-[11px] text-base-content/70 loom-tabular">
          {counted ? (
            <>
              {fmtInt(item.lo)}
              {item.N != null && (
                <span className="text-base-content/50">
                  {' / '}
                  {fmtCompact(item.N)} · {fmtPct(item.lo / item.N)}
                </span>
              )}
              {below && <span className="text-base-content/50"> · below floor</span>}
            </>
          ) : (
            children
          )}
        </span>
      </button>
    </li>
  );
}

function ConceptMode({row, go}: {row: number; go: (t: InspectorTarget) => void}) {
  const {model, cells, stats, state, columns, dispatch, revealRow} = useLoom();
  const r = model.rows[row];
  const readout = useMemo(() => rowReadout(model, stats, row), [model, stats, row]);
  const contrib = useMemo(() => contributions(model, cells, columns, row), [model, cells, columns, row]);
  const [showNotCoded, setShowNotCoded] = useState(false);
  const [showAlts, setShowAlts] = useState(false);
  const pulledKey = state.pool.find(k => slotRowKeys(k).includes(r.key)) ?? null;
  const nSlots = slotsOfMask(r.slotMask).length;
  const unfolded = state.unfolded.includes(r.key);
  const openCell = (cohort: number) => go({kind: 'cell', row, cohort});

  // Facts behind the identifiers and units sections, from the home members:
  // variables per token, and each distinct unit with the cohorts using it.
  const facts = useMemo(() => {
    const tokenVars = new Map<string, number>();
    const units = new Map<string, {label: string; cohorts: number[]}>();
    for (let c = 0; c < model.K; c++) {
      if (!r.cohortMask[c]) continue;
      for (const i of homeMembers(model, row, c)) {
        const v = model.variables[i];
        for (const t of v.tokens) if (r.tokens.includes(t)) tokenVars.set(t, (tokenVars.get(t) ?? 0) + 1);
        const u = v.units.trim();
        // The model's unit rule (the ≠u flag): case, spaces and µ/u ignored.
        const k = u.toLowerCase().replace(/\s+/g, '').replace(/[µμ]/g, 'u');
        if (UNIT_PLACEHOLDERS.has(k)) continue;
        const entry = units.get(k) ?? {label: u, cohorts: []};
        if (!entry.cohorts.includes(c)) entry.cohorts.push(c);
        units.set(k, entry);
      }
    }
    return {
      tokens: r.tokens.map(t => ({t, n: tokenVars.get(t) ?? 0})),
      units: Array.from(units.values()).sort((a, b) => b.cohorts.length - a.cohorts.length)
    };
  }, [model, r, row]);

  const maxLo = contrib.counted.reduce((m, x) => Math.max(m, x.lo), 0);
  const fogCount = columns.visibleFog.length;
  const fogDeclared = columns.visibleFog.reduce((s, c) => s + (model.cohorts[c].declared ?? 0), 0);
  const fogTotal = model.cohorts.filter(c => c.tier === 'none').length;
  const alts = showAlts ? r.altLabels : r.altLabels.slice(0, 4);
  const hidden = state.hiddenRows.includes(r.key);

  return (
    <>
      <div className="px-4 pb-3 pt-2">
        <div className="flex items-start gap-2">
          <span className="mt-1">
            <Monogram domain={r.domain} />
          </span>
          <div className="min-w-0">
            <h2 className="text-base font-semibold leading-snug">{readout.headline}</h2>
            {r.altLabels.length > 0 && (
              <p className="mt-0.5 text-[11px] leading-snug text-base-content/60">
                <span className="text-base-content/50">Also named </span>
                {alts.join(' · ')}
                {r.altLabels.length > 4 && (
                  <button
                    type="button"
                    className="link link-hover ml-1"
                    onClick={() => setShowAlts(s => !s)}
                    title={showAlts ? 'Show fewer names' : 'Show every name the cohorts use'}
                  >
                    {showAlts ? 'fewer' : `+${r.altLabels.length - 4} more`}
                  </button>
                )}
              </p>
            )}
          </div>
        </div>
        {readout.lines.length > 0 && (
          <ul className="mt-2 space-y-0.5 text-[11px] leading-snug text-base-content/70 loom-tabular">
            {readout.lines.map((l: string, k: number) => (
              <li key={k}>{l}</li>
            ))}
          </ul>
        )}
        <div className="mt-3 flex flex-wrap gap-1.5">
          <button
            type="button"
            className="btn btn-sm btn-neutral"
            onClick={() => {
              if (pulledKey) dispatch({type: 'release', key: pulledKey});
              else dispatch({type: 'pull', keys: [r.key]});
            }}
            title={
              pulledKey
                ? 'Release this concept from the Pool'
                : 'Pull this concept into the Pool: how many patients could have it together with the other pulled concepts'
            }
          >
            {pulledKey ? 'Release from Pool' : 'Pull into Pool'}
          </button>
          <button
            type="button"
            className={`${INSPECTOR_BTN} btn-sm`}
            onClick={() => revealRow(row)}
            title="Scroll the matrix to this row"
          >
            Reveal in matrix
          </button>
          {nSlots > 1 && (
            <button
              type="button"
              className={`${INSPECTOR_BTN} btn-sm`}
              onClick={() => dispatch({type: 'toggleUnfold', key: r.key})}
              title={
                unfolded ? 'Fold the visit sub-rows of this concept' : 'Show one sub-row per visit under this concept'
              }
            >
              {unfolded ? 'Fold visits' : 'Unfold visits'}
            </button>
          )}
          <button
            type="button"
            className={`${INSPECTOR_BTN} btn-sm`}
            onClick={() => dispatch({type: 'toggleHiddenRow', key: r.key})}
            title={hidden ? 'Show this concept again' : 'Hide this concept (a chip in the recipe bar restores it)'}
          >
            {hidden ? 'Unhide' : 'Hide'}
          </button>
        </div>
      </div>

      <Section
        title="Contributions"
        note={isAllLens(state.visit) ? 'best visit per cohort' : `at ${lensLabel(state.visit)}`}
      >
        <ul className="space-y-px">
          {contrib.counted.map(item => (
            <ContributionLine key={item.cohort} item={item} maxLo={maxLo} onOpen={() => openCell(item.cohort)} />
          ))}
          {contrib.zero.map(item => (
            <ContributionLine key={item.cohort} item={item} maxLo={maxLo} onOpen={() => openCell(item.cohort)}>
              no values
            </ContributionLine>
          ))}
          {contrib.notInEda.map(item => (
            <ContributionLine key={item.cohort} item={item} maxLo={maxLo} onOpen={() => openCell(item.cohort)}>
              not in profiled data
            </ContributionLine>
          ))}
          {contrib.pending.map(item => (
            <ContributionLine key={item.cohort} item={item} maxLo={maxLo} onOpen={() => openCell(item.cohort)}>
              {model.countsStatus === 'loading' ? 'counts loading…' : 'counts unavailable'}
            </ContributionLine>
          ))}
          {contrib.dictionary.map(item => {
            const declared = model.cohorts[item.cohort].declared;
            return (
              <ContributionLine key={item.cohort} item={item} maxLo={maxLo} onOpen={() => openCell(item.cohort)}>
                count unknown
                {declared != null && <span className="text-base-content/50"> · ~{fmtCompact(declared)} declared</span>}
              </ContributionLine>
            );
          })}
          {contrib.otherVisit.map(item => (
            <ContributionLine key={item.cohort} item={item} maxLo={maxLo} onOpen={() => openCell(item.cohort)}>
              at{' '}
              {slotsOfMask(item.otherSlots)
                .map(s => SLOT_SHORT[s])
                .join(', ')}
            </ContributionLine>
          ))}
        </ul>
        {contrib.notCoded.length > 0 && (
          <div className="mt-2 text-[11px] text-base-content/60">
            <button
              type="button"
              className="link link-hover"
              aria-expanded={showNotCoded}
              onClick={() => setShowNotCoded(s => !s)}
              title="List the visible cohorts whose dictionary does not code this concept"
            >
              Not coded in {inspectorPlural(contrib.notCoded.length, 'visible cohort', 'visible cohorts')}
            </button>
            {showNotCoded && (
              <p className="mt-0.5 leading-snug">
                {contrib.notCoded.map(c => model.cohorts[c].id).join(', ')}. It may exist there uncoded or under another
                code.
              </p>
            )}
          </div>
        )}
        {fogTotal > 0 && (
          <p className="mt-1 flex items-center gap-1.5 text-[11px] text-base-content/60">
            <TierGlyph tier="none" />
            {fogCount > 0 ? (
              <span>
                ? {inspectorPlural(fogCount, 'cohort', 'cohorts')} without a dictionary
                {fogDeclared > 0 && <> (~{fmtCompact(fogDeclared)} declared participants)</>}: unknown, not absent.
              </span>
            ) : (
              <span>
                ? {inspectorPlural(fogTotal, 'cohort', 'cohorts')} without a dictionary {fogTotal === 1 ? 'is' : 'are'} hidden:
                unknown, not absent.
              </span>
            )}
          </p>
        )}
      </Section>

      <Section title="Cohort × visit" note="the same cell grammar, one visit per column">
        <CubeFace row={row} onOpenCell={openCell} />
      </Section>

      {facts.units.length > 0 && (
        <Section title="Units" note={r.flags.unitsDiffer ? '≠u differ across cohorts' : 'the same in every cohort'}>
          <ul className="space-y-0.5 text-xs">
            {facts.units.map(u => {
              const names = u.cohorts.map(c => model.cohorts[c].id);
              return (
                <li key={u.label} className="flex items-baseline gap-2" title={names.join(', ')}>
                  <span className="font-mono">{u.label}</span>
                  <span className="min-w-0 truncate text-[11px] text-base-content/60">
                    {inspectorPlural(names.length, 'cohort', 'cohorts')} · {names.slice(0, 4).join(', ')}
                    {names.length > 4 && ` +${names.length - 4}`}
                  </span>
                </li>
              );
            })}
          </ul>
        </Section>
      )}

      <Section title="Identifiers" note={`${inspectorPlural(r.tokens.length, 'token', 'tokens')} · key ${tokenLabel(r.key)}`}>
        <ul className="mb-3 flex flex-wrap gap-1">
          {facts.tokens.map(({t, n}) => (
            <li
              key={t}
              className="rounded border border-[color:var(--loom-hairline)] px-1.5 py-0.5 font-mono text-[11px]"
              title={`${tokenLabel(t)}: carried by ${inspectorPlural(n, 'variable', 'variables')} of this concept`}
            >
              {tokenLabel(t)}
              <span className="ml-1 font-sans text-base-content/50 loom-tabular">{n}</span>
            </li>
          ))}
        </ul>
        <BridgeExplainer row={row} />
      </Section>
    </>
  );
}

// ---------------------------------------------------------------------------
// Cohort mode
// ---------------------------------------------------------------------------

function fmtDate(iso: string | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? null
    : d.toLocaleDateString(undefined, {year: 'numeric', month: 'short', day: 'numeric'});
}

function CohortMode({cohort, go}: {cohort: number; go: (t: InspectorTarget) => void}) {
  const {model, state, dispatch, columns, filter, revealCohort} = useLoom();
  const c = model.cohorts[cohort];
  // The profiling date and listed entries live in the counts response (one
  // request per session, shared with the page).
  const {counts} = useObservationCounts(c.tier === 'profiled');
  const edaMeta = counts?.[c.id];
  const profiledOn = fmtDate(edaMeta?.generated_at);
  const pin = state.pins.find(p => p.cohortId === c.id)?.kind ?? 'none';
  const anchored = state.anchor === c.id;
  const hidden = state.hiddenCohorts.includes(c.id);
  const solo = state.onlyCohorts.length === 1 && state.onlyCohorts[0] === c.id;
  const hasDict = c.tier !== 'none';

  const perDomain = useMemo(() => {
    const all = new Map<Domain, number>();
    const shown = new Map<Domain, number>();
    for (const r of model.rows) {
      if (!r.cohortMask[cohort]) continue;
      all.set(r.domain, (all.get(r.domain) ?? 0) + 1);
      if (filter.visible[r.i]) shown.set(r.domain, (shown.get(r.domain) ?? 0) + 1);
    }
    return DOMAINS.filter(d => all.has(d)).map(d => ({d, all: all.get(d) as number, shown: shown.get(d) ?? 0}));
  }, [model, cohort, filter]);
  const maxDomain = perDomain.reduce((m, x) => Math.max(m, x.all), 0);
  const totalConcepts = perDomain.reduce((s, x) => s + x.all, 0);

  // Nearest overlap among the visible dictionary cohorts (Jaccard of the
  // concepts they code).
  const nearest = useMemo(() => {
    if (!hasDict) return [];
    const list: {cohort: number; shares: number; j: number}[] = [];
    for (const col of columns.columns) {
      if (col.kind !== 'cohort' || col.cohort === cohort || !columns.visibleMask[col.cohort]) continue;
      const o = jaccard(model, cohort, col.cohort);
      if (o.shares > 0) list.push({cohort: col.cohort, shares: o.shares, j: o.j});
    }
    return list.sort((a, b) => b.j - a.j || b.shares - a.shares).slice(0, 3);
  }, [model, columns, cohort, hasDict]);

  const found = c.nVars - c.nNotInEda;
  const foundFrac = c.nVars > 0 ? found / c.nVars : 1;
  const matchedW = c.nVars > 0 ? (c.nMatched / c.nVars) * 100 : 0;
  const singleW = c.nVars > 0 ? (c.nSingle / c.nVars) * 100 : 0;
  const unmappedW = c.nVars > 0 ? (c.nUnmapped / c.nVars) * 100 : 0;

  const setAnchor = () =>
    dispatch({
      type: 'patch',
      patch: anchored
        ? {anchor: null, colSort: state.colSort === 'anchor' ? DEFAULT_VIEW_STATE.colSort : state.colSort}
        : {anchor: c.id, colSort: 'anchor'}
    });

  return (
    <>
      <div className="px-4 pb-3 pt-2">
        <div className="flex items-start gap-2">
          <TierGlyph tier={c.tier} className="mt-1.5" />
          <div className="min-w-0">
            <h2 className="break-words text-base font-semibold leading-snug">{c.id}</h2>
            {c.institution && <p className="text-[11px] text-base-content/60">{c.institution}</p>}
            <p className="text-[11px] text-base-content/60">
              {TIER_LABEL[c.tier]}
              {c.edaVersion && c.tier === 'profiled' && ` · EDA ${c.edaVersion}`}
              {profiledOn && ` · profiled ${profiledOn}`}
              {c.note && ` · ${TIER_NOTE_LABEL[c.note]}`}
            </p>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          <button
            type="button"
            className={`${INSPECTOR_BTN} btn-sm`}
            disabled={!hasDict}
            onClick={() => dispatch({type: 'cyclePin', cohortId: c.id, profiled: c.tier === 'profiled'})}
            title={
              hasDict
                ? `Pin: ${PIN_WORD[pin]}. Click to cycle ○ → ✓ coded → ${c.tier === 'profiled' ? '✓✓ with data → ' : ''}✕ not coded → ○`
                : 'Nothing to pin: no dictionary'
            }
          >
            <span className="loom-tabular">{INSPECTOR_PIN_GLYPH[pin]}</span> Pin{pin !== 'none' && `: ${PIN_WORD[pin]}`}
          </button>
          <button
            type="button"
            className={`${INSPECTOR_BTN} btn-sm`}
            onClick={() => dispatch({type: 'toggleHiddenCohort', cohortId: c.id})}
            title={hidden ? `Show ${c.id} again` : `Hide the column of ${c.id}`}
          >
            {hidden ? 'Unhide' : 'Hide'}
          </button>
          <button
            type="button"
            className={`${INSPECTOR_BTN} btn-sm`}
            onClick={() => dispatch({type: 'soloCohort', cohortId: c.id})}
            title={solo ? 'Show every cohort again' : `Show only ${c.id}`}
          >
            {solo ? 'Show all' : 'Only this'}
          </button>
          <button
            type="button"
            className={`${INSPECTOR_BTN} btn-sm`}
            disabled={!hasDict}
            onClick={setAnchor}
            title={
              !hasDict
                ? 'No dictionary: nothing to compare'
                : anchored
                  ? 'Clear the anchor'
                  : `Anchor: sort the columns by how many concepts they share with ${c.id}`
            }
          >
            {anchored ? 'Clear anchor' : 'Anchor'}
          </button>
          <button
            type="button"
            className={`${INSPECTOR_BTN} btn-sm`}
            onClick={() => revealCohort(cohort)}
            disabled={columns.cohortToColumn[cohort] < 0}
            title={
              columns.cohortToColumn[cohort] < 0
                ? `${c.id} is not shown by the current filters`
                : 'Scroll the matrix to this column'
            }
          >
            Reveal
          </button>
          <Link href={exploreHref(c.id, 'list')} className={`${INSPECTOR_BTN} btn-sm`} title={`Open ${c.id} on the Explore page`}>
            Open in Explore
          </Link>
        </div>
      </div>

      <Section title="Study">
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs">
          <dt className="text-base-content/50">Study type</dt>
          <dd>
            {c.studyTypeRaw || <span className="text-base-content/40">not given</span>}
            <span className="text-base-content/50"> · {FAMILY_LABEL[c.family]}</span>
          </dd>
          <dt className="text-base-content/50">Design</dt>
          <dd>
            {c.designRaw || <span className="text-base-content/40">not given</span>}
            <span className="text-base-content/50"> · {DESIGN_LABEL[c.design]}</span>
          </dd>
          <dt className="text-base-content/50">Status</dt>
          <dd>{STATUS_LABEL[c.status]}</dd>
          <dt className="text-base-content/50">Participants</dt>
          <dd className="loom-tabular">
            {c.declared != null ? (
              `~${fmtInt(c.declared)} declared`
            ) : (
              <span className="text-base-content/40">none declared</span>
            )}
            {c.nRows != null && (
              <>
                <span className="text-base-content/50"> · </span>
                {fmtInt(c.nRows)} dataset rows (EDA)
              </>
            )}
          </dd>
          {c.tier === 'profiled' && (
            <>
              <dt className="text-base-content/50">Profiling</dt>
              <dd className="loom-tabular">
                EDA {c.edaVersion ?? '?'}
                {profiledOn && ` · ${profiledOn}`}
                {edaMeta?.n_listed != null && ` · ${inspectorPlural(edaMeta.n_listed, 'entry', 'entries')} listed`}
              </dd>
            </>
          )}
        </dl>
        {c.tier === 'profiled' && c.nVars > 0 && (
          <div className="mt-2 text-xs">
            <p className="loom-tabular">
              Profiled <span className="font-semibold">{fmtInt(found)}</span> of {fmtInt(c.nVars)} dictionary variables
              <span className="text-base-content/60"> ({fmtPct(foundFrac)})</span>
            </p>
            {foundFrac < 0.8 && (
              <p className="mt-0.5 leading-snug text-base-content/70">
                Caution: {inspectorPlural(c.nNotInEda, 'dictionary variable was', 'dictionary variables were')} not found in the
                profiled data (renamed, dropped, or added after profiling). Their cells show ⊘, never zero.
              </p>
            )}
            {c.edaVersion === 'v1' && <p className="mt-0.5 leading-snug text-base-content/60">{CAVEATS.edaV1}</p>}
          </div>
        )}
        {c.nRows == null && c.tier === 'profiled' && (
          <p className="mt-1 text-[11px] leading-snug text-base-content/60">
            {CAVEATS.sizeUnknown.charAt(0).toUpperCase() + CAVEATS.sizeUnknown.slice(1)}.
          </p>
        )}
      </Section>

      {hasDict ? (
        <>
          <Section title="Dictionary" note={inspectorPlural(c.nVars, 'variable', 'variables')}>
            <div
              className="flex h-2.5 w-full gap-[2px] overflow-hidden"
              role="img"
              aria-label={`${c.nMatched} matched in two or more cohorts, ${c.nSingle} in this cohort only, ${c.nUnmapped} without a valid identifier`}
            >
              {matchedW > 0 && <span style={{width: `${matchedW}%`, background: 'var(--loom-ink-fill)'}} />}
              {singleW > 0 && <span style={{width: `${singleW}%`, background: 'var(--loom-soft-fill)'}} />}
              {unmappedW > 0 && (
                <span style={{width: `${unmappedW}%`, boxShadow: 'inset 0 0 0 1px var(--loom-mark)'}} />
              )}
            </div>
            <dl className="mt-1.5 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-xs loom-tabular">
              <dt className="flex items-center gap-1.5">
                <span
                  className="inline-block h-2 w-2"
                  style={{background: 'var(--loom-ink-fill)'}}
                  aria-hidden="true"
                />
                {fmtInt(c.nMatched)}
              </dt>
              <dd className="text-base-content/70">share a concept with another cohort</dd>
              <dt className="flex items-center gap-1.5">
                <span
                  className="inline-block h-2 w-2"
                  style={{background: 'var(--loom-soft-fill)'}}
                  aria-hidden="true"
                />
                {fmtInt(c.nSingle)}
              </dt>
              <dd className="text-base-content/70">coded, but no other cohort has the concept</dd>
              <dt className="flex items-center gap-1.5">
                <span
                  className="inline-block h-2 w-2"
                  style={{boxShadow: 'inset 0 0 0 1px var(--loom-mark)'}}
                  aria-hidden="true"
                />
                {fmtInt(c.nUnmapped)}
              </dt>
              <dd className="text-base-content/70">
                no valid concept code or OMOP ID ·{' '}
                <Link
                  href={exploreHref(c.id, 'list')}
                  className="link link-hover"
                  title={`Review the variables of ${c.id} on the Explore page`}
                >
                  review in Explore
                </Link>
              </dd>
            </dl>
          </Section>

          <Section title="Concepts per domain" note={`${fmtInt(totalConcepts)} coded (outline) · shown now (fill)`}>
            <ul className="space-y-0.5">
              {perDomain.map(({d, all, shown}) => (
                <li key={d} className="grid grid-cols-[14px_110px_minmax(0,1fr)_auto] items-center gap-x-2 text-xs">
                  <Monogram domain={d} />
                  <span className="truncate">{DOMAIN_LABEL[d]}</span>
                  <span className="relative h-[6px]" aria-hidden="true">
                    <span
                      className="absolute inset-y-0 left-0"
                      style={{
                        width: `${(all / Math.max(1, maxDomain)) * 100}%`,
                        boxShadow: 'inset 0 0 0 1px var(--loom-mark)'
                      }}
                    />
                    <span
                      className="absolute inset-y-0 left-0"
                      style={{width: `${(shown / Math.max(1, maxDomain)) * 100}%`, background: 'var(--loom-ink-fill)'}}
                    />
                  </span>
                  <span className="whitespace-nowrap text-right text-[11px] text-base-content/70 loom-tabular">
                    {fmtInt(all)}
                    {shown !== all && <span className="text-base-content/50"> · {fmtInt(shown)} shown</span>}
                  </span>
                </li>
              ))}
            </ul>
          </Section>

          {nearest.length > 0 && (
            <Section title="Nearest overlap" note="concepts coded by both ÷ coded by either">
              <ul className="space-y-0.5">
                {nearest.map(o => (
                  <li key={o.cohort}>
                    <button
                      type="button"
                      className="flex w-full items-center gap-2 rounded px-1 py-0.5 text-left text-xs hover:bg-[color:var(--loom-wash)]"
                      onClick={() => go({kind: 'cohort', cohort: o.cohort})}
                      title={`Open ${model.cohorts[o.cohort].id}`}
                    >
                      <TierGlyph tier={model.cohorts[o.cohort].tier} />
                      <span className="min-w-0 flex-1 truncate">{model.cohorts[o.cohort].id}</span>
                      <span className="whitespace-nowrap text-[11px] text-base-content/70 loom-tabular">
                        shares {fmtInt(o.shares)} · J {o.j.toFixed(2)}
                        {o.j >= 0.95 && ' ≡'}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </Section>
          )}
        </>
      ) : (
        <Section title="No dictionary">
          <p className="text-xs leading-snug text-base-content/70">
            No dictionary was uploaded for {c.id}
            {c.declared != null && <> ({fmtInt(c.declared)} declared participants)</>}. Nothing is known about its
            variables: unknown, not absent. Its column is hatched in every row, and it cannot be added to a Data Clean
            Room from here because none of its variables are known.
          </p>
        </Section>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// The drawer
// ---------------------------------------------------------------------------

const valid = (t: InspectorTarget, model: LoomModel): boolean => {
  const rowOk = (r: number) => r >= 0 && r < model.rows.length;
  const cohortOk = (c: number) => c >= 0 && c < model.cohorts.length;
  if (t.kind === 'cell') return rowOk(t.row) && cohortOk(t.cohort);
  if (t.kind === 'concept') return rowOk(t.row);
  return cohortOk(t.cohort);
};

const targetTitle = (t: InspectorTarget, model: LoomModel): string =>
  t.kind === 'cell'
    ? `${model.rows[t.row].label} · ${model.cohorts[t.cohort].id}`
    : t.kind === 'concept'
      ? model.rows[t.row].label
      : model.cohorts[t.cohort].id;

const MODE_LABEL: Record<InspectorTarget['kind'], string> = {cell: 'Cell', concept: 'Concept', cohort: 'Cohort'};

function Drawer({
  target,
  canGoBack,
  go,
  back,
  close
}: {
  target: InspectorTarget;
  canGoBack: boolean;
  go: (t: InspectorTarget) => void;
  back: () => void;
  close: () => void;
}) {
  const {model, ui} = useLoom();
  const asideRef = useRef<HTMLElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [entered, setEntered] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => setEntered(true));
    return () => cancelAnimationFrame(id);
  }, []);

  // A new target starts at the top.
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }, [target]);

  // Esc closes the drawer unless a layer above it (a modal, the help sheet)
  // takes the key, or the key belongs to an editable field elsewhere. The
  // grid refocuses itself when the drawer closes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return;
      if (ui.basketPreview || ui.helpOpen) return;
      const el = e.target instanceof HTMLElement ? e.target : null;
      if (el?.closest('dialog')) return;
      const inside = !!el && !!asideRef.current?.contains(el);
      if (!inside && el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;
      close();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [close, ui.basketPreview, ui.helpOpen]);

  const title = targetTitle(target, model);
  return (
    <>
      <div className="absolute inset-0 z-30 bg-base-content/10 xl:hidden" onClick={close} aria-hidden="true" />
      <aside
        ref={asideRef}
        aria-label={`Inspector: ${title}`}
        className={`absolute inset-y-0 right-0 z-40 flex w-[min(420px,100vw)] flex-col border-l border-base-300 bg-base-100 shadow-xl transition-transform duration-200 ease-out motion-reduce:transition-none xl:w-[400px] ${
          entered ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <header className="flex h-10 flex-shrink-0 items-center gap-1 border-b border-[color:var(--loom-hairline)] px-2">
          {canGoBack && (
            <button
              type="button"
              className="btn btn-ghost btn-xs btn-square"
              onClick={back}
              aria-label="Back to the previous inspector view"
              title="Back"
            >
              <ArrowLeft size={14} />
            </button>
          )}
          <span className="px-1 text-[11px] font-semibold uppercase tracking-wide text-base-content/60">
            {MODE_LABEL[target.kind]}
          </span>
          <span className="min-w-0 flex-1 truncate text-xs text-base-content/70" title={title}>
            {title}
          </span>
          <button
            type="button"
            className="btn btn-ghost btn-xs btn-square"
            onClick={close}
            aria-label="Close the inspector"
            title="Close (Esc)"
          >
            <X size={16} />
          </button>
        </header>
        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto overscroll-contain pb-6">
          {target.kind === 'cell' ? (
            <CellMode key={`${target.row}:${target.cohort}`} row={target.row} cohort={target.cohort} go={go} />
          ) : target.kind === 'concept' ? (
            <ConceptMode key={target.row} row={target.row} go={go} />
          ) : (
            <CohortMode key={target.cohort} cohort={target.cohort} go={go} />
          )}
        </div>
      </aside>
    </>
  );
}

function Inspector() {
  const {ui, setUi, model} = useLoom();
  const target = ui.inspector;
  // Navigation inside the drawer (concept -> cell -> cohort ...) keeps a
  // short history for Back; a target set from outside (grid, table) starts
  // a new one.
  const [history, setHistory] = useState<InspectorTarget[]>([]);
  const internal = useRef(false);

  useEffect(() => {
    if (internal.current) internal.current = false;
    else setHistory([]);
  }, [target]);

  const go = useCallback(
    (next: InspectorTarget) => {
      if (target) {
        internal.current = true;
        setHistory(h => [...h.slice(-19), target]);
      }
      setUi({inspector: next});
    },
    [target, setUi]
  );
  const back = useCallback(() => {
    const prev = history[history.length - 1];
    if (!prev) return;
    internal.current = true;
    setHistory(h => h.slice(0, -1));
    setUi({inspector: prev});
  }, [history, setUi]);
  const close = useCallback(() => setUi({inspector: null}), [setUi]);

  if (!target || !valid(target, model)) return null;
  return <Drawer target={target} canGoBack={history.length > 0} go={go} back={back} close={close} />;
}

// ============================================================================
// NotShownLedger
// ============================================================================

// "What is not on screen, and why" (spec §9.4): every concept the view hides,
// attributed to the ONE filter that hides it (or to several), plus what never
// became a row at all. Each line with a meaningful way back has a "show".

function Line({
  label,
  value,
  unit,
  hint,
  action,
  actionTitle,
  indent
}: {
  label: React.ReactNode;
  value: number;
  unit?: string;
  hint?: string;
  action?: () => void;
  actionTitle?: string;
  indent?: boolean;
}) {
  return (
    <li className={`flex items-baseline gap-2 py-[3px] ${indent ? 'pl-3' : ''}`} title={hint}>
      <span className="min-w-0 flex-1 text-[12px] text-base-content/80">{label}</span>
      <span className="loom-tabular shrink-0 text-[12px] text-base-content">
        {fmtInt(value)}
        {unit && <span className="ml-1 text-[10.5px] text-base-content/45">{unit}</span>}
      </span>
      <span className="w-10 shrink-0 text-right">
        {action && (
          <button
            type="button"
            className="text-[11px] text-base-content/60 underline decoration-base-content/25 underline-offset-2 hover:text-base-content"
            title={actionTitle}
            onClick={action}
          >
            show
          </button>
        )}
      </span>
    </li>
  );
}

function NotShownLedger({onDone}: {onDone?: () => void}) {
  const {filter, state, dispatch} = useLoom();
  const ledger = filter.ledger;
  const chipById = new Map(filter.chips.map(c => [c.id, c]));
  const hiddenRows = ledger.totalRows - filter.visibleCount;

  const run = (action: LoomAction) => {
    dispatch(action);
    onDone?.();
  };
  // Every chip cleared in ONE state change (a single URL write, a single recompute).
  const clearAll = () => {
    const next = filter.chips.reduce((s, c) => reduceViewState(s, c.clear), state);
    run({type: 'patch', patch: next});
  };

  const fogHidden = !state.tiers.none || state.fog === 'hide';
  // A filter that hides nothing on its own has nothing to account for here.
  const hiddenByFilters = ledger.byFilter.filter(f => f.rows > 0);

  return (
    <div className="flex max-h-[70vh] flex-col">
      <div className="border-b border-base-300 px-3 pb-2 pt-3">
        <Caps>What is not shown</Caps>
        <p className="mt-1.5 text-[12px] text-base-content/80">
          <span className="loom-tabular font-semibold text-base-content">{fmtInt(filter.visibleCount)}</span> of{' '}
          <span className="loom-tabular">{fmtInt(ledger.totalRows)}</span> concepts shown
          {hiddenRows > 0 && (
            <>
              {' '}
              · <span className="loom-tabular">{fmtInt(hiddenRows)}</span> hidden
            </>
          )}
        </p>
      </div>
      <div className="overflow-y-auto px-3 py-2">
        {(hiddenByFilters.length > 0 || ledger.multiCause > 0) && (
          <>
            <Caps className="mb-1 mt-1">Hidden by the filters</Caps>
            <ul>
              {hiddenByFilters.map(f => {
                const chip = chipById.get(f.id);
                return (
                  <Line
                    key={f.id}
                    label={f.label}
                    value={f.rows}
                    hint={`${fmtInt(f.rows)} concepts are hidden by this filter alone: removing it brings exactly these back`}
                    action={chip ? () => run(chip.clear) : undefined}
                    actionTitle={chip ? `Remove “${chip.label}”` : undefined}
                  />
                );
              })}
              {ledger.multiCause > 0 && (
                <Line
                  label="By two or more filters"
                  value={ledger.multiCause}
                  hint="Concepts that more than one filter hides: removing any single filter does not bring them back"
                  action={filter.chips.length > 0 ? clearAll : undefined}
                  actionTitle="Remove every filter chip"
                />
              )}
            </ul>
          </>
        )}

        <Caps className="mb-1 mt-3">Coded in one cohort only</Caps>
        <ul>
          <Line
            label="Single-cohort concepts"
            value={ledger.singleCohortRows}
            hint="Real concepts that only one cohort's dictionary codes; hidden by the default “In ≥ 2 cohorts”"
            action={
              state.minCohorts > 1 && ledger.singleCohortRows > 0
                ? () => run({type: 'patch', patch: {minCohorts: 1}})
                : undefined
            }
            actionTitle="Set “In ≥ 1 cohort”"
          />
        </ul>

        <Caps className="mb-1 mt-3">Never rows</Caps>
        <ul>
          <Line
            label="Variables without a valid code or OMOP ID"
            unit="variables"
            value={ledger.unmappedVariables}
            hint="Unmapped variables cannot be matched across cohorts; each cohort's inspector links to them in Explore"
          />
          {ledger.malformedIdentifiers > 0 && (
            <Line
              label="Malformed identifiers (dropped)"
              unit="values"
              value={ledger.malformedIdentifiers}
              hint="OMOP IDs that are not whole numbers, dropped while matching"
              indent
            />
          )}
          {ledger.notInEdaVariables > 0 && (
            <Line
              label={
                <>
                  <CellGlyph kind="notInEda" width={12} height={10} className="mr-1.5" />
                  Listed, not in the profiled data
                </>
              }
              unit="variables"
              value={ledger.notInEdaVariables}
              hint="Dictionary variables of profiled cohorts that the EDA output does not contain (renamed, dropped, or added after profiling). They stay in their rows as ⊘."
            />
          )}
        </ul>

        {ledger.fogCohorts.count > 0 && (
          <>
            <Caps className="mb-1 mt-3">Cohorts</Caps>
            <ul>
              <Line
                label={
                  <>
                    <CellGlyph kind="fog" width={10} height={10} className="mr-1.5" />
                    Without a dictionary
                    {ledger.fogCohorts.declared > 0 && (
                      <span className="text-base-content/50">
                        {' '}
                        (~{fmtInt(ledger.fogCohorts.declared)} declared participants)
                      </span>
                    )}
                  </>
                }
                unit="cohorts"
                value={ledger.fogCohorts.count}
                hint="No dictionary uploaded: nothing is known about their variables. Unknown, not absent."
                action={
                  fogHidden
                    ? () => run({type: 'patch', patch: {tiers: {...state.tiers, none: true}, fog: 'fold'}})
                    : undefined
                }
                actionTitle="Show the cohorts without a dictionary (folded)"
              />
            </ul>
          </>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// LoomHeader
// ============================================================================

// Page header (56px): the woven wordmark, the knowledge bar (one pip per
// registered cohort, grouped by what we know about it), the concept count
// that opens the Not-shown ledger, the counts status, Help and Share.

const HEADER_TIERS: Tier[] = ['profiled', 'dictionary', 'none'];
const TIER_WORD: Record<Tier, string> = {profiled: 'profiled', dictionary: 'dictionary only', none: 'no dictionary'};

function pipStyle(tier: Tier): React.CSSProperties {
  if (tier === 'profiled') return {backgroundColor: 'var(--loom-ink-fill)'};
  if (tier === 'dictionary') return {boxShadow: 'inset 0 0 0 1px var(--loom-mark)'};
  return {};
}

function sizeWords(c: LoomCohort): string {
  if (c.nRows !== null) return `${fmtInt(c.nRows)} dataset rows`;
  if (c.declared !== null) return `~${fmtInt(c.declared)} declared participants`;
  return 'size unknown';
}

// Copies the address of the current view. The URL already carries every
// setting (defaults omitted), so the link reproduces this exact view.
async function copyViewLink(): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(window.location.href);
    return true;
  } catch {
    return false;
  }
}

function KnowledgeBar() {
  const {model, columns, state, dispatch, revealCohort, setUi} = useLoom();
  const fog = useMemo(() => new Set(columns.visibleFog), [columns.visibleFog]);
  const shown = (c: LoomCohort) => columns.visibleMask[c.i] === 1 || fog.has(c.i);
  const groups = HEADER_TIERS.map(t => ({tier: t, cohorts: model.cohorts.filter(c => c.tier === t)})).filter(
    g => g.cohorts.length > 0
  );
  const flat = groups.flatMap(g => g.cohorts);
  const [focusIdx, setFocusIdx] = useState(0);
  const pips = useRef<(HTMLButtonElement | null)[]>([]);

  const totals = useMemo(() => {
    const sum = (t: Tier, f: (c: LoomCohort) => number | null) =>
      model.cohorts.filter(c => c.tier === t).reduce((s, c) => s + (f(c) ?? 0), 0);
    return {
      profiled: `${fmtCompact(sum('profiled', c => c.nRows))} dataset rows`,
      dictionary: `~${fmtCompact(sum('dictionary', c => c.declared))} declared participants`,
      none: `~${fmtCompact(sum('none', c => c.declared))} declared participants`
    } as Record<Tier, string>;
  }, [model]);

  const activate = (c: LoomCohort) => {
    if (shown(c)) revealCohort(c.i);
    else setUi({inspector: {kind: 'cohort', cohort: c.i}});
  };

  const onKeyDown = (e: React.KeyboardEvent, i: number) => {
    let next = -1;
    if (e.key === 'ArrowRight') next = Math.min(flat.length - 1, i + 1);
    else if (e.key === 'ArrowLeft') next = Math.max(0, i - 1);
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = flat.length - 1;
    if (next < 0) return;
    e.preventDefault();
    setFocusIdx(next);
    pips.current[next]?.focus();
  };

  const toggleTier = (t: Tier) => {
    const on = t === 'none' ? state.tiers.none && state.fog !== 'hide' : state.tiers[t];
    if (t === 'none')
      dispatch({
        type: 'patch',
        patch: {tiers: {...state.tiers, none: !on}, fog: !on && state.fog === 'hide' ? 'fold' : state.fog}
      });
    else dispatch({type: 'patch', patch: {tiers: {...state.tiers, [t]: !on}}});
  };

  const tabStop = Math.min(focusIdx, flat.length - 1);
  let idx = -1;
  return (
    <div className="flex min-w-0 flex-1 flex-col justify-center gap-[5px]">
      <div className="flex min-w-0 items-center gap-2" role="toolbar" aria-label="Registered cohorts by knowledge tier">
        {groups.map(g => (
          <div key={g.tier} className="flex min-w-0 gap-[2px]" style={{flexShrink: g.cohorts.length}}>
            {g.cohorts.map(c => {
              idx++;
              const i = idx;
              const isShown = shown(c);
              return (
                <button
                  key={c.id}
                  ref={el => {
                    pips.current[i] = el;
                  }}
                  type="button"
                  tabIndex={i === tabStop ? 0 : -1}
                  aria-label={`${c.id}, ${TIER_LABEL[c.tier]}${isShown ? '' : ', hidden by the filters'}`}
                  title={`${c.id} · ${TIER_LABEL[c.tier]} · ${sizeWords(c)}${
                    isShown
                      ? '\nClick: scroll to its column'
                      : '\nHidden by the current filters. Click: open it in the inspector'
                  }`}
                  className={`h-3 w-2 min-w-[3px] shrink rounded-[1px] outline-none transition-opacity hover:ring-1 hover:ring-base-content/50 focus-visible:ring-2 focus-visible:ring-base-content/60 ${
                    c.tier === 'none' ? 'loom-fog' : ''
                  } ${isShown ? '' : 'opacity-25'}`}
                  style={pipStyle(c.tier)}
                  onFocus={() => setFocusIdx(i)}
                  onKeyDown={e => onKeyDown(e, i)}
                  onClick={() => activate(c)}
                />
              );
            })}
          </div>
        ))}
      </div>
      <div className="flex items-center gap-1 text-[11px] leading-none text-base-content/60">
        {groups.map((g, gi) => {
          const on = g.tier === 'none' ? state.tiers.none && state.fog !== 'hide' : state.tiers[g.tier];
          return (
            <React.Fragment key={g.tier}>
              {gi > 0 && <span className="text-base-content/30">·</span>}
              <button
                type="button"
                aria-pressed={on}
                className={`loom-tabular rounded px-0.5 hover:text-base-content ${on ? '' : 'text-base-content/35 line-through decoration-base-content/30'}`}
                title={`${fmtInt(g.cohorts.length)} ${TIER_LABEL[g.tier].toLowerCase()} cohorts · ${totals[g.tier]}. Click to ${on ? 'hide' : 'show'} them.`}
                onClick={() => toggleTier(g.tier)}
              >
                <span className="font-semibold text-base-content/80">{fmtInt(g.cohorts.length)}</span>
                <span className="hidden xl:inline"> {TIER_WORD[g.tier]}</span>
              </button>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}

function LoomHeader() {
  const {model, filter, ui, setUi, retryCounts} = useLoom();

  const share = async () => {
    const ok = await copyViewLink();
    setUi({
      toast: {
        id: Date.now(),
        text: ok ? 'Link copied: it opens this exact view' : 'Could not copy: the link is in the address bar'
      }
    });
  };

  const iconButton =
    'flex h-7 w-7 items-center justify-center rounded-md text-base-content/60 hover:bg-base-content/[0.06] hover:text-base-content';

  return (
    <header className="relative z-50 flex h-12 shrink-0 items-center gap-5 border-b border-base-300 px-4 xl:h-14">
      <div className="flex shrink-0 items-center gap-2.5">
        <WovenGlyph />
        <div className="leading-tight">
          <h1 className="text-[17px] font-semibold tracking-[-0.01em] text-base-content">Concept Coverage</h1>
          <p
            className="hidden text-[11px] text-base-content/55 xl:block"
            title="Which cohorts record which concepts, at which visits, and how many values back them."
          >
            Every concept, threaded through every cohort.
          </p>
        </div>
      </div>

      <KnowledgeBar />

      <div className="flex shrink-0 items-center gap-1.5">
        {model.countsStatus === 'loading' && (
          <span className="flex items-center gap-1.5 text-[11px] text-base-content/55" role="status">
            <span className="loading loading-spinner loading-xs" aria-hidden="true" />
            loading counts…
          </span>
        )}
        {model.countsStatus === 'unavailable' && (
          <span className="flex items-center gap-1 text-[11px] text-base-content/70" role="alert">
            <AlertCircle size={13} aria-hidden="true" />
            counts unavailable ·
            <button
              type="button"
              className="inline-flex items-center gap-0.5 underline decoration-base-content/30 underline-offset-2 hover:text-base-content"
              title="Counts could not be loaded; showing presence only. Retry loading the observation counts."
              onClick={retryCounts}
            >
              <RefreshCw size={11} aria-hidden="true" /> Retry
            </button>
          </span>
        )}

        <Popover
          open={ui.ledgerOpen}
          onOpenChange={open => setUi({ledgerOpen: open})}
          align="right"
          ariaLabel="What is not shown"
          title={`${fmtInt(filter.visibleCount)} of ${fmtInt(filter.ledger.totalRows)} concepts shown. Click to see what is not shown, and why.`}
          panelClassName="w-[380px] max-w-[calc(100vw-2rem)]"
          buttonClassName="inline-flex h-7 items-center gap-1 rounded-md px-2 text-[12px] text-base-content/80 hover:bg-base-content/[0.06] hover:text-base-content"
          label={
            <>
              <span className="loom-tabular font-semibold text-base-content">{fmtInt(filter.visibleCount)}</span>
              <span>concepts</span>
              <ChevronDown size={12} className="opacity-60" aria-hidden="true" />
            </>
          }
        >
          {close => <NotShownLedger onDone={close} />}
        </Popover>

        <button
          type="button"
          className={iconButton}
          title="How to read Concept Coverage, and keyboard shortcuts (?)"
          aria-label="Help"
          onClick={() => setUi({helpOpen: true})}
        >
          <HelpCircle size={16} />
        </button>
        <button
          type="button"
          className={iconButton}
          title="Share: copy a link to this exact view"
          aria-label="Copy a link to this view"
          onClick={share}
        >
          <Share2 size={15} />
        </button>
      </div>
    </header>
  );
}

// ============================================================================
// LoomTable
// ============================================================================

// The Table twin (spec §10.3): the same rows, order and visible cohorts as
// the Loom, as a real <table> with text cells, so every mark is readable
// without colour or hover. Rows are virtualized (spacer rows keep the scroll
// height); the header row and the concept column stay put.

const ROW_H = 30;
const TABLE_BAND_H = 26;
const TABLE_OVERSCAN = 12;
const QUANT = (TABLE_OVERSCAN / 2) * ROW_H;
const COL_W = 132;
// The concept column never takes more than about half of a phone screen.
const HEAD_W = 'min(280px, 45vw)';

type Item = {kind: 'band'; label: string; count: number; y: number} | {kind: 'row'; row: number; y: number};

// Word for a cell that is not a count. Blank = not coded (as in the Loom).
function stateWord(state: CellStateValue, status: CountsStatus, otherSlots: number): string {
  switch (state) {
    case CellState.Zero:
      return 'no values';
    case CellState.NotInEda:
      return 'not in profiled data';
    case CellState.Dictionary:
      return 'not profiled';
    case CellState.Pending:
      return status === 'loading' ? 'counts loading' : 'counts unavailable';
    case CellState.OtherVisit:
      return `other visit: ${slotsOfMask(otherSlots)
        .map(s => SLOT_SHORT[s])
        .join(', ')}`;
    default:
      return '';
  }
}

interface CellText {
  text: string;
  counted: boolean;
  title: string;
}

function cellText(model: LoomModel, cells: LensCells, row: number, cohort: number): CellText {
  const key = row * model.K + cohort;
  const state = cells.state[key] as CellStateValue;
  const flags = cells.flags[key];
  const N = model.cohorts[cohort].nRows;
  const rep = cells.rep[key];
  const repVar = rep >= 0 ? model.variables[rep] : null;
  const via = repVar ? `via ${repVar.name} (${SLOT_LONG[repVar.slot]})` : '';
  const k = cells.k[key];
  const notes: string[] = [];
  if (k > 1) notes.push(`best of ${k} variables (not a sum)`);
  if (flags & CellFlag.BelowFloor) notes.push('below the usable floor');
  if (flags & CellFlag.GuestOnly) notes.push('guest (secondary code) only');
  if (flags & CellFlag.InBasketAll) notes.push('all in your basket');
  else if (flags & CellFlag.InBasket) notes.push('partly in your basket');
  if (state === CellState.Counted) {
    const lo = cells.lo[key];
    // N is in the column header; the cell keeps the count and its share.
    const text = N ? `${fmtInt(lo)} · ${fmtPct(lo / N)}` : fmtInt(lo);
    const full = N ? `${fmtInt(lo)} / ${fmtInt(N)} · ${fmtPct(lo / N)}` : fmtInt(lo);
    return {text, counted: true, title: [full, via, ...notes].filter(Boolean).join(' · ')};
  }
  const word = stateWord(state, model.countsStatus, cells.otherSlots[key]);
  return {
    text: word,
    counted: false,
    title: [word || 'not coded with this concept', via, ...notes].filter(Boolean).join(' · ')
  };
}

// ---------------------------------------------------------------------------
// CSV (wide format, RFC 4180): one line per concept, four columns per cohort.
// ---------------------------------------------------------------------------

const CSV_STATE: Record<CellStateValue, string> = {
  [CellState.NotCoded]: 'not coded',
  [CellState.Counted]: 'counted',
  [CellState.Zero]: 'no values',
  [CellState.NotInEda]: 'not in profiled data',
  [CellState.Dictionary]: 'not profiled',
  [CellState.Pending]: 'counts unavailable',
  [CellState.OtherVisit]: 'other visit',
  [CellState.Fog]: 'no dictionary'
};

function csvField(value: string | number | null): string {
  if (value == null) return '';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '';
  // Text from the data never runs as a spreadsheet formula.
  const safe = /^[=+\-@\t\r]/.test(value) ? `'${value}` : value;
  return /[",\r\n]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
}

function buildCsv(
  model: LoomModel,
  cells: LensCells,
  stats: RowStats,
  rows: number[],
  cohorts: number[]
): string {
  const header: string[] = [
    'concept_key',
    'concept',
    'domain',
    'cohorts_coding',
    'cohorts_measured',
    'pooled_lower',
    'pooled_upper',
    'dictionary_only_cohorts'
  ];
  for (const c of cohorts) {
    const id = model.cohorts[c].id;
    header.push(`${id} n`, `${id} N`, `${id} status`, `${id} variable`);
  }
  const lines = [header.map(csvField).join(',')];
  for (const r of rows) {
    const row = model.rows[r];
    const fields: (string | number | null)[] = [
      row.key,
      row.label,
      DOMAIN_LABEL[row.domain],
      stats.covDict[r],
      stats.covMeasured[r],
      stats.pooledLo[r],
      stats.pooledHi[r],
      stats.dictOnly[r]
    ];
    for (const c of cohorts) {
      const key = r * model.K + c;
      const state = cells.state[key] as CellStateValue;
      const rep = cells.rep[key];
      const counted = state === CellState.Counted || state === CellState.Zero;
      fields.push(
        counted ? cells.lo[key] : null,
        model.cohorts[c].nRows,
        state === CellState.Pending && model.countsStatus === 'loading' ? 'counts loading' : CSV_STATE[state],
        rep >= 0 ? model.variables[rep].name : null
      );
    }
    lines.push(fields.map(csvField).join(','));
  }
  return lines.join('\r\n') + '\r\n';
}

function download(filename: string, text: string) {
  // The BOM makes spreadsheet apps read the file as UTF-8.
  const blob = new Blob(['\uFEFF', text], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// One concept row of the table (memoized: scrolling re-renders only the rows
// that enter the window).
const TableRow = memo(function TableRow({
  row,
  rowIndex,
  cohorts,
  model,
  cells,
  stats,
  dim,
  setUi
}: {
  row: number;
  rowIndex: number;
  cohorts: number[];
  model: LoomModel;
  cells: LensCells;
  stats: RowStats;
  dim: boolean;
  setUi: (patch: Partial<LoomUiState>) => void;
}) {
  const r = model.rows[row];
  const measured = stats.covMeasured[row];
  const coded = stats.covDict[row];
  const lo = stats.pooledLo[row];
  const hi = stats.pooledHi[row];
  const td = 'whitespace-nowrap border-b border-[color:var(--loom-hairline)] px-2 group-hover:bg-base-200';
  return (
    <tr className={`group${dim ? ' opacity-25' : ''}`} style={{height: ROW_H}} aria-rowindex={rowIndex}>
      <th
        scope="row"
        className="sticky left-0 z-10 border-b border-r border-[color:var(--loom-hairline)] bg-base-100 px-3 text-left font-normal group-hover:bg-base-200"
      >
        <button
          type="button"
          className="flex w-full items-center gap-2 text-left hover:underline"
          onClick={() => setUi({inspector: {kind: 'concept', row}})}
          title={`${r.label} · ${DOMAIN_LABEL[r.domain]}: open in the inspector`}
        >
          <span className="inline-flex h-[14px] w-[14px] flex-shrink-0 items-center justify-center border border-[color:var(--loom-hairline)] text-[9px] text-base-content/60">
            {DOMAIN_MONOGRAM[r.domain]}
          </span>
          <span className="truncate">{r.label}</span>
        </button>
      </th>
      {cohorts.map(c => {
        const cell = cellText(model, cells, row, c);
        return (
          <td key={c} className={`${td} ${cell.counted ? 'loom-tabular' : 'text-[11px] italic text-base-content/60'}`}>
            {cell.text && (
              <button
                type="button"
                className="w-full truncate text-left hover:underline"
                onClick={() => setUi({inspector: {kind: 'cell', row, cohort: c}})}
                title={`${model.cohorts[c].id}: ${cell.title}`}
              >
                {cell.text}
              </button>
            )}
          </td>
        );
      })}
      <td className={`${td} text-right loom-tabular`} title={`${measured} with measured data, ${coded - measured} coded without a count`}>
        {measured}
        <span className="text-base-content/50"> + {coded - measured}</span>
      </td>
      <td className={`${td} text-right loom-tabular`} title={CAVEATS.pooled}>
        {lo > 0 ? `${fmtCompact(lo)}–${fmtCompact(hi)}` : '—'}
      </td>
    </tr>
  );
});

// ---------------------------------------------------------------------------

function LoomTable() {
  const {model, cells, stats, columns, layout, filter, state, setUi} = useLoom();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({top: 0, height: 800});

  // Visible dictionary columns in Loom order (the fog is a note, not columns).
  const cohorts = useMemo(
    () =>
      columns.columns.filter(col => col.kind === 'cohort' && columns.visibleMask[col.cohort]).map(col => col.cohort),
    [columns]
  );

  // Band headers and rows in Loom order; a collapsed band still lists its rows
  // (collapsing is a Loom gesture), visit sub-rows are the Loom's alone.
  const {items, height, rowCount} = useMemo(() => {
    const list: Item[] = [];
    let y = 0;
    let n = 0;
    for (const it of layout.items) {
      if (it.kind === 'band') {
        list.push({kind: 'band', label: it.label, count: it.count, y});
        y += TABLE_BAND_H;
      } else if (it.kind === 'row') {
        list.push({kind: 'row', row: it.row, y});
        y += ROW_H;
        n++;
      } else if (it.kind === 'tally') {
        for (const r of it.rows) {
          list.push({kind: 'row', row: r, y});
          y += ROW_H;
          n++;
        }
      }
    }
    return {items: list, height: y, rowCount: n};
  }, [layout]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // The window moves in steps of half the overscan, so scrolling re-renders
    // only when rows actually enter or leave it.
    const measure = () => {
      const top = Math.floor(el.scrollTop / QUANT) * QUANT;
      const height = el.clientHeight;
      setView(v => (v.top === top && v.height === height ? v : {top, height}));
    };
    measure();
    let frame = 0;
    const onScroll = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        measure();
      });
    };
    el.addEventListener('scroll', onScroll, {passive: true});
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => {
      el.removeEventListener('scroll', onScroll);
      ro.disconnect();
      if (frame) cancelAnimationFrame(frame);
    };
  }, []);

  // The window of items to render: binary search on y.
  const first = useMemo(() => {
    let lo = 0;
    let hi = items.length - 1;
    const y = Math.max(0, view.top - TABLE_OVERSCAN * ROW_H);
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (items[mid].y <= y) lo = mid;
      else hi = mid - 1;
    }
    return Math.max(0, lo);
  }, [items, view.top]);
  const bottom = view.top + view.height + TABLE_OVERSCAN * ROW_H;
  let last = first;
  while (last < items.length && items[last].y < bottom) last++;
  const windowItems = items.slice(first, last);
  const padTop = items.length > 0 ? items[first].y : 0;
  const endY = last < items.length ? items[last].y : height;
  const padBottom = height - endY;

  const colSpan = cohorts.length + 3;
  const fogCount = columns.visibleFog.length;
  const fogDeclared = columns.visibleFog.reduce((s, c) => s + (model.cohorts[c].declared ?? 0), 0);
  const highlight = filter.highlight;

  const downloadCsv = () => {
    const rows = items.flatMap(it => (it.kind === 'row' ? [it.row] : []));
    const date = new Date().toISOString().slice(0, 10);
    const lens = lensLabel(state.visit)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-');
    download(`loom_${lens}_${date}.csv`, buildCsv(model, cells, stats, rows, cohorts));
  };

  return (
    <div className="flex h-full w-full flex-col">
      <div className="flex flex-shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-[color:var(--loom-hairline)] px-4 py-2 text-xs">
        <span className="loom-tabular">
          <span className="font-semibold">{fmtInt(rowCount)}</span> concepts ×{' '}
          <span className="font-semibold">{fmtInt(cohorts.length)}</span> cohorts · {lensLabel(state.visit)}
        </span>
        <span className="text-base-content/60" title={`${CAVEATS.best} ${CAVEATS.rows}`}>
          Cells: values of the best single variable / dataset rows · completeness. Blank = not coded.
        </span>
        {fogCount > 0 && (
          <span className="text-base-content/60">
            {fogCount} {fogCount === 1 ? 'cohort' : 'cohorts'} without a dictionary
            {fogDeclared > 0 && ` (~${fmtCompact(fogDeclared)} declared)`} are not columns: unknown, not absent.
          </span>
        )}
        <span className="flex-1" />
        <button
          type="button"
          className="btn btn-xs btn-ghost border border-base-300 font-normal"
          onClick={downloadCsv}
          disabled={rowCount === 0}
          title="Download these concepts and cohorts as CSV (wide: n, N, status and variable per cohort)"
        >
          <Download size={12} /> Download CSV
        </button>
      </div>

      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto">
        <table
          className="table-fixed border-separate border-spacing-0 text-xs"
          style={{width: `calc(${HEAD_W} + ${cohorts.length * COL_W + 200}px)`}}
          aria-rowcount={items.length + 1}
          aria-colcount={colSpan}
        >
          <caption className="sr-only">
            Concepts by cohort at {lensLabel(state.visit)}: values with a count over dataset rows, or the reason there
            is no count
          </caption>
          <colgroup>
            <col style={{width: HEAD_W}} />
            {cohorts.map(c => (
              <col key={c} style={{width: COL_W}} />
            ))}
            <col style={{width: 88}} />
            <col style={{width: 112}} />
          </colgroup>
          <thead>
            <tr aria-rowindex={1}>
              <th
                scope="col"
                className="sticky left-0 top-0 z-30 border-b border-r border-[color:var(--loom-hairline)] bg-base-100 px-3 py-2 text-left align-bottom font-semibold"
              >
                Concept
              </th>
              {cohorts.map(c => {
                const cohort = model.cohorts[c];
                return (
                  <th
                    key={c}
                    scope="col"
                    className="sticky top-0 z-20 border-b border-[color:var(--loom-hairline)] bg-base-100 px-2 py-2 text-left align-bottom font-normal"
                  >
                    <button
                      type="button"
                      className="block w-full text-left hover:underline"
                      onClick={() => setUi({inspector: {kind: 'cohort', cohort: c}})}
                      title={`${cohort.id} · ${TIER_LABEL[cohort.tier]}${cohort.nRows != null ? ` · ${fmtInt(cohort.nRows)} dataset rows` : ''}: open in the inspector`}
                    >
                      <span className="line-clamp-2 break-words font-semibold leading-tight">{cohort.id}</span>
                      <span className="mt-0.5 flex items-center gap-1 text-[10px] text-base-content/60">
                        <TierGlyph tier={cohort.tier} />
                        {cohort.tier === 'profiled' && cohort.nRows != null
                          ? `N ${fmtInt(cohort.nRows)}`
                          : TIER_LABEL[cohort.tier]}
                      </span>
                    </button>
                  </th>
                );
              })}
              <th
                scope="col"
                className="sticky top-0 z-20 border-b border-[color:var(--loom-hairline)] bg-base-100 px-2 py-2 text-right align-bottom font-semibold"
                title="Visible cohorts coding the concept: with measured data + coded without a count"
              >
                Coverage
              </th>
              <th
                scope="col"
                className="sticky top-0 z-20 border-b border-[color:var(--loom-hairline)] bg-base-100 px-2 py-2 text-right align-bottom font-semibold"
                title={`Lower–upper bound. ${CAVEATS.pooled}`}
              >
                Pooled
              </th>
            </tr>
          </thead>
          <tbody>
            {padTop > 0 && (
              <tr aria-hidden="true">
                <td colSpan={colSpan} style={{height: padTop, padding: 0}} />
              </tr>
            )}
            {windowItems.map((it, j) => {
              if (it.kind === 'band')
                return (
                  <tr key={`band-${first + j}`} style={{height: TABLE_BAND_H}} aria-rowindex={first + j + 2}>
                    <th
                      scope="rowgroup"
                      colSpan={colSpan}
                      className="border-b border-[color:var(--loom-hairline)] bg-base-100 px-3 text-left text-[11px] font-semibold uppercase tracking-wide text-base-content/60"
                    >
                      <span className="sticky left-3">
                        {it.label} · {fmtInt(it.count)}
                      </span>
                    </th>
                  </tr>
                );
              return (
                <TableRow
                  key={it.row}
                  row={it.row}
                  rowIndex={first + j + 2}
                  cohorts={cohorts}
                  model={model}
                  cells={cells}
                  stats={stats}
                  dim={highlight != null && !highlight[it.row]}
                  setUi={setUi}
                />
              );
            })}
            {padBottom > 0 && (
              <tr aria-hidden="true">
                <td colSpan={colSpan} style={{height: padBottom, padding: 0}} />
              </tr>
            )}
          </tbody>
        </table>
        {rowCount === 0 && (
          <p className="px-4 py-6 text-sm text-base-content/60">No concept passes the current filters.</p>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// RecipeBar
// ============================================================================

// The recipe (spec §7.3): what the view shows, as a sentence of removable
// clauses. Each clause is a chip with its cost (−N = concepts that come back
// when this clause alone is removed). Wraps to two lines, then "+k more".
// Rendered only while something narrows the view.

const SET_MODES: {value: SetMode; label: string; title: string}[] = [
  {value: 'all', label: 'ALL', title: 'ALL: concepts coded in every ✓ pinned cohort'},
  {value: 'any', label: 'ANY', title: 'ANY: concepts coded in at least one ✓ pinned cohort'},
  {value: 'only', label: 'ONLY', title: 'ONLY: coded in every pinned cohort and in no other shown cohort'},
  {value: 'abo', label: 'ALL BUT ONE', title: 'ALL BUT ONE: exactly one pinned cohort lacks the concept (a gap finder)'}
];

const TWO_LINES = 50; // px: two 22px chip lines and their gap

function Clause({chip, onClear, strong}: {chip: FilterChip; onClear: () => void; strong: boolean}) {
  return (
    <span
      data-clause
      className={`inline-flex h-[22px] max-w-full items-center rounded border text-[11px] ${
        strong ? 'border-base-content/45 bg-base-content/[0.08]' : 'border-base-content/20 bg-base-content/[0.04]'
      }`}
    >
      <span
        className="truncate pl-1.5 pr-1 text-base-content/85"
        title={`${chip.label}. ${
          chip.cost > 0
            ? `Removing it alone brings back ${fmtInt(chip.cost)} concept${chip.cost === 1 ? '' : 's'}.`
            : 'Removing it alone brings back no concept (another filter also hides them).'
        }`}
      >
        {chip.label}
      </span>
      {chip.cost > 0 && (
        <span className="loom-tabular pr-1 text-[10.5px] text-base-content/50">−{fmtInt(chip.cost)}</span>
      )}
      <button
        type="button"
        className="flex h-full items-center border-l border-base-content/15 px-1 text-base-content/45 hover:bg-base-content/[0.06] hover:text-base-content"
        title={`Remove “${chip.label}”`}
        aria-label={`Remove ${chip.label}`}
        onClick={onClear}
      >
        <X size={11} />
      </button>
    </span>
  );
}

function RecipeBar() {
  const {filter, state, dispatch} = useLoom();
  const wrapRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  const [overflow, setOverflow] = useState(0);

  const lensActive = !isAllLens(state.visit);
  const positivePins = state.pins.some(p => p.kind !== 'not');
  const notice = filter.singleCohortShownBySearch;
  const active = filter.chips.length > 0 || lensActive || state.pins.length > 0 || notice > 0;
  const chipKey = filter.chips.map(c => `${c.id}:${c.label}`).join('|');
  const empty = filter.visibleCount === 0;

  // Count the clauses pushed past the second line, for "+k more".
  useLayoutEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const measure = () => {
      const items = Array.from(wrap.querySelectorAll<HTMLElement>('[data-clause]'));
      if (!items.length) return setOverflow(0);
      const top = items[0].offsetTop;
      setOverflow(items.filter(el => el.offsetTop - top >= TWO_LINES - 22).length);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [chipKey, lensActive, notice, active, empty]);

  if (!active) return null;

  const run = (a: LoomAction) => dispatch(a);
  // Everything in one state change: one URL write, one recompute.
  const clearAll = () => {
    let next = filter.chips.reduce((s, c) => reduceViewState(s, c.clear), state);
    next = {...next, pins: [], visit: ALL_LENS};
    dispatch({type: 'patch', patch: next});
  };
  const costliest = filter.chips.reduce<FilterChip | null>((best, c) => (c.cost > (best?.cost ?? 0) ? c : best), null);

  return (
    <div
      className="flex shrink-0 items-start gap-2 border-b border-base-300 px-4 py-[3px]"
      role="region"
      aria-label={filter.recipe ? `View recipe: ${filter.recipe}` : 'Active filters'}
      title={filter.recipe || undefined}
    >
      <Caps className="mt-[8px] shrink-0" tone="text-base-content/45">
        Showing
      </Caps>
      <div
        ref={wrapRef}
        className={`flex min-w-0 flex-1 flex-wrap items-center gap-1 ${expanded ? '' : 'max-h-[50px] overflow-hidden'}`}
      >
        {/* First, so the two-line fold never hides it. */}
        {empty && costliest && (
          <span data-clause className="text-[11px] text-base-content/70" role="status">
            No concept passes every clause: removing <span className="font-semibold">{costliest.label}</span> brings
            back <span className="loom-tabular font-semibold">{fmtInt(costliest.cost)}</span>.
          </span>
        )}
        {empty && !costliest && (
          <span data-clause className="text-[11px] text-base-content/70" role="status">
            No concept passes every clause, and no single clause is the cause: several hide them together. Clear removes
            them all.
          </span>
        )}
        {filter.chips.map(chip => (
          <Clause key={chip.id} chip={chip} strong={empty && chip === costliest} onClear={() => run(chip.clear)} />
        ))}
        {lensActive && (
          <span
            data-clause
            className="inline-flex h-[22px] items-center rounded border border-base-content/20 bg-base-content/[0.04] text-[11px]"
          >
            <span
              className="pl-1.5 pr-1 text-base-content/85"
              title="The visit lens: every cell shows only the variables recorded at this visit; concepts not coded at it in any shown cohort are hidden"
            >
              at {lensLabel(state.visit)}
            </span>
            <button
              type="button"
              className="flex h-full items-center border-l border-base-content/15 px-1 text-base-content/45 hover:bg-base-content/[0.06] hover:text-base-content"
              title="Back to all visits"
              aria-label="Back to all visits"
              onClick={() => run({type: 'patch', patch: {visit: ALL_LENS}})}
            >
              <X size={11} />
            </button>
          </span>
        )}
        {notice > 0 && (
          <span
            data-clause
            className="inline-flex h-[22px] items-center gap-1 rounded border border-dashed border-base-content/25 px-1.5 text-[11px] text-base-content/65"
            title="Single-cohort concepts are normally hidden by “In ≥ 2 cohorts”; the search shows the ones it matches"
          >
            <Info size={11} aria-hidden="true" /> includes {fmtInt(notice)} single-cohort concept
            {notice === 1 ? '' : 's'}
          </span>
        )}
      </div>
      {(overflow > 0 || expanded) && (
        <button
          type="button"
          className="mt-[4px] h-[22px] shrink-0 rounded px-1.5 text-[11px] text-base-content/60 hover:bg-base-content/[0.06] hover:text-base-content"
          title={expanded ? 'Fold the clauses back to two lines' : 'Show every clause'}
          aria-expanded={expanded}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'less' : `+${overflow} more`}
        </button>
      )}
      {positivePins && (
        <div className="mt-[3px] flex shrink-0 items-center gap-1">
          <span className="text-[10.5px] text-base-content/50">Pins</span>
          <Segmented
            size="xs"
            ariaLabel="Set mode of the ✓ pins"
            value={state.setMode}
            options={SET_MODES}
            onChange={v => run({type: 'patch', patch: {setMode: v}})}
          />
        </div>
      )}
      <button
        type="button"
        className="mt-[4px] flex h-[22px] shrink-0 items-center gap-1 rounded px-1.5 text-[11px] text-base-content/60 hover:bg-base-content/[0.06] hover:text-base-content"
        title="Remove every clause: filters, pins and the visit lens (metric, layout and the Pool stay)"
        onClick={clearAll}
      >
        <RotateCcw size={11} aria-hidden="true" /> Clear
      </button>
    </div>
  );
}

// ============================================================================
// StatusBar
// ============================================================================

// The 32px status bar. Left: the FOCUS readout, the same sentence as the
// tooltip for whatever the pointer or the keyboard is on, so every fact is on
// screen without hovering (announced politely for keyboard moves only).
// Right: the legend, listing only the glyphs present in the current view; a
// click on an entry dims every other cell (selective reading, not a filter).

interface Readout {
  text: string; // the one line shown
  full: string; // every line (tooltip)
  spoken: string; // what a screen reader hears
}

interface LegendEntry {
  key: string | null; // ui.legendDim key; null = explanation only
  glyph: {kind: GlyphKind; dogEar?: boolean; floor?: boolean; guest?: boolean};
  label: string;
  title: string;
}

const STATE_ENTRY: Partial<Record<CellStateValue, {kind: GlyphKind; label: string}>> = {
  [CellState.Zero]: {kind: 'zero', label: 'no values'},
  [CellState.NotInEda]: {kind: 'notInEda', label: 'not in EDA'},
  [CellState.Dictionary]: {kind: 'ring', label: 'dictionary only'},
  [CellState.Pending]: {kind: 'pending', label: 'counts pending'},
  [CellState.OtherVisit]: {kind: 'dot', label: 'other visit'},
  [CellState.Fog]: {kind: 'fog', label: 'no dictionary'}
};

function useReadout(target: PointerTarget | null, metric: Metric): Readout | null {
  const {model, stats, state, layout} = useLoom();
  return useMemo(() => {
    if (!target) return null;
    const {row, cohort} = target;
    const rowOk = row >= 0 && row < model.rows.length;
    const cohortOk = cohort >= 0 && cohort < model.cohorts.length;
    if (target.kind === 'cell' && rowOk && cohortOk) {
      const lens: VisitLens = target.slot ? {kind: 'slots', mask: SLOT_BIT[target.slot]} : state.visit;
      const text = cellSummary(model, row, cohort, lens, metric, state.floor, state.guests);
      const r = cellReadout(model, row, cohort, lens, metric, state.floor, state.guests);
      return {
        text,
        full: [r.headline, r.context, ...r.lines, ...r.footer].join('\n'),
        spoken: `${r.context}: ${r.headline}`
      };
    }
    if (target.kind === 'row' && rowOk) {
      const r = rowReadout(model, stats, row);
      return {
        text: [r.headline, ...r.lines.slice(1, 3)].join(' · '),
        full: [r.headline, ...r.lines].join('\n'),
        spoken: `${r.headline}. ${r.lines.slice(1, 3).join('. ')}`
      };
    }
    if (target.kind === 'column' && cohortOk) {
      const r = cohortReadout(model, cohort);
      return {
        text: [r.headline, ...r.lines].join(' · '),
        full: [r.headline, ...r.lines].join('\n'),
        spoken: `${r.headline}. ${r.lines[0]}`
      };
    }
    const item = target.item >= 0 ? layout.items[target.item] : undefined;
    if (target.kind === 'band' && item && item.kind === 'band') {
      const text = `${item.label} · ${fmtInt(item.count)} concept${item.count === 1 ? '' : 's'}${item.collapsed ? ' · collapsed' : ''}`;
      return {text, full: text, spoken: text};
    }
    return null;
  }, [target, metric, model, stats, state.visit, state.floor, state.guests, layout]);
}

interface LegendData {
  entries: LegendEntry[];
  bins: Set<number>;
  hollow: boolean;
}

// Approximate inline widths (px) of the legend's parts, and the least room the
// FOCUS readout keeps before the legend folds into its popover.
const RAMP_W = 255;
const ENTRY_W = 104;
const FOCUS_MIN = 600;

function useLegend(metric: Metric): LegendData {
  const {model, cells, columns, layout, state} = useLoom();
  return useMemo(() => {
    const K = model.K;
    const cols: number[] = [];
    for (const col of columns.columns)
      if (col.kind === 'cohort' && col.cohort >= 0 && columns.visibleMask[col.cohort]) cols.push(col.cohort);
    const states = new Set<number>();
    const bins = new Set<number>();
    let presence = false;
    let multi = false;
    let floor = false;
    let guest = false;
    const note = (st: number, fl: number, lo: number, c: number) => {
      states.add(st);
      if (st === CellState.NotCoded) return;
      if (fl & CellFlag.Multi) multi = true;
      if (fl & CellFlag.GuestOnly) guest = true;
      if (st === CellState.Counted) {
        if (fl & CellFlag.BelowFloor) floor = true;
        const b = binFor(metric, lo, model.cohorts[c].nRows);
        if (b > 0) bins.add(b);
        else presence = true;
      }
    };
    for (const r of layout.order) {
      const base = r * K;
      for (const c of cols) note(cells.state[base + c], cells.flags[base + c], cells.lo[base + c], c);
    }
    // Unfolded visit sub-rows show per-slot cells that can differ from their row.
    for (const it of layout.items) {
      if (it.kind !== 'sub') continue;
      const lens: VisitLens = {kind: 'slots', mask: SLOT_BIT[it.slot]};
      for (const c of cols) {
        const cell = resolveCell(model, it.row, c, lens, state.floor, state.guests);
        note(cell.state, cell.flags, cell.lo, c);
      }
    }
    if (columns.visibleFog.length > 0) states.add(CellState.Fog);

    const dimTitle = ' Click to dim every other cell; click again to undo.';
    const entries: LegendEntry[] = [];
    if (presence)
      entries.push({
        key: `state:${CellState.Counted}`,
        glyph: {kind: 'presence'},
        label: metric === 'pres' ? 'values counted' : 'values, size unknown',
        title:
          (metric === 'pres'
            ? 'Presence: profiled, at least one value (no ramp).'
            : `Profiled with values, but the ${CAVEATS.sizeUnknown}.`) + dimTitle
      });
    (
      [
        CellState.Dictionary,
        CellState.Pending,
        CellState.Zero,
        CellState.NotInEda,
        CellState.OtherVisit,
        CellState.Fog
      ] as CellStateValue[]
    ).forEach(st => {
      const e = STATE_ENTRY[st];
      if (!e || !states.has(st)) return;
      const label = st === CellState.Pending && model.countsStatus === 'unavailable' ? 'counts unavailable' : e.label;
      entries.push({key: `state:${st}`, glyph: {kind: e.kind}, label, title: `${stateLabel(st)}.${dimTitle}`});
    });
    if (multi)
      entries.push({
        key: 'flag:multi',
        glyph: {kind: bins.size ? 'fill' : 'ring', dogEar: true},
        label: 'several vars',
        title: `Corner cut: several variables in the cell; it shows the best one. ${CAVEATS.best}${dimTitle}`
      });
    if (floor)
      entries.push({
        key: null,
        glyph: {kind: 'fill', floor: true},
        label: `below ${state.floor}%`,
        title: `Below the usable floor (${state.floor}%): the fill collapses to a bar in the same colour and stops counting toward coverage and the Pool.`
      });
    if (guest)
      entries.push({
        key: null,
        glyph: {kind: bins.size ? 'fill' : 'ring', guest: true},
        label: 'guest',
        title: CAVEATS.guest
      });
    const hollow = states.has(CellState.Dictionary) || states.has(CellState.Zero) || states.has(CellState.NotInEda);
    return {entries, bins, hollow};
  }, [model, cells, columns, layout, metric, state.floor, state.guests]);
}

function Legend({metric, data, stacked}: {metric: Metric; data: LegendData; stacked?: boolean}) {
  const {ui, setUi, state} = useLoom();
  const {entries, bins, hollow} = data;
  const labels = metric === 'n' ? BIN_LABELS.n : BIN_LABELS.comp;
  // The lower edge of bins 2..5 ('25–50%' -> '25', '≥ 95%' -> '95%').
  const edges = labels.slice(1).map(l => l.replace(/^[≥>]\s*/, '').split('–')[0]);
  const dim = ui.legendDim;
  const toggle = (key: string) => setUi({legendDim: dim === key ? null : key});
  const faded = (key: string | null) => dim !== null && key !== dim;

  return (
    <div className={`flex ${stacked ? 'flex-col items-start gap-2' : 'items-center gap-2.5'}`}>
      {bins.size > 0 && metric !== 'pres' && (
        <div
          className="flex items-center gap-1.5"
          role="group"
          aria-label={`Legend: ${METRIC_EXPLAINER[metric]}`}
          title={METRIC_EXPLAINER[metric]}
        >
          <Caps size="text-[9.5px]">{METRIC_LABEL[metric]}</Caps>
          <div className="flex flex-col gap-[2px]">
            <div className="flex gap-[1px]">
              {labels.map((label, i) => {
                const key = `bin:${i + 1}`;
                const here = bins.has(i + 1);
                return (
                  <button
                    key={key}
                    type="button"
                    aria-pressed={dim === key}
                    aria-label={`${METRIC_LABEL[metric]} ${label}`}
                    title={`${METRIC_LABEL[metric]} ${label}${here ? '' : ' (none in this view)'}. Click to dim every other cell; click again to undo.`}
                    className={`h-[9px] w-[24px] rounded-[1px] transition-opacity ${
                      dim === key ? 'ring-2 ring-base-content/60 ring-offset-1 ring-offset-base-100' : ''
                    } ${faded(key) ? 'opacity-30' : here ? '' : 'opacity-40'}`}
                    style={{backgroundColor: `var(--loom-bin-${i + 1})`}}
                    onClick={() => toggle(key)}
                  />
                );
              })}
            </div>
            {/* Tick labels on the bin edges: "25 50 80 95%" / "50 250 1k 5k". */}
            <div className="relative h-[8px]" aria-hidden="true">
              {edges.map((edge, i) => (
                <span
                  key={edge}
                  className="loom-tabular absolute -translate-x-1/2 whitespace-nowrap text-[8.5px] leading-none text-base-content/55"
                  style={{left: (i + 1) * 25 - 0.5}}
                >
                  {edge}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
      <div className={`flex ${stacked ? 'flex-col items-start gap-1' : 'items-center gap-1'}`}>
        {entries.map(e => {
          const body = (
            <>
              <CellGlyph
                kind={e.glyph.kind}
                dogEar={e.glyph.dogEar}
                floor={e.glyph.floor}
                guest={e.glyph.guest}
                bin={3}
              />
              <span className="whitespace-nowrap text-[10.5px] text-base-content/65">{e.label}</span>
            </>
          );
          return e.key ? (
            <button
              key={e.label}
              type="button"
              aria-pressed={dim === e.key}
              title={e.title}
              className={`flex h-6 items-center gap-1 rounded px-1 transition-opacity ${
                dim === e.key ? 'bg-base-content/[0.08] ring-1 ring-base-content/30' : 'hover:bg-base-content/[0.05]'
              } ${faded(e.key) ? 'opacity-35' : ''}`}
              onClick={() => toggle(e.key as string)}
            >
              {body}
            </button>
          ) : (
            <span
              key={e.label}
              title={e.title}
              className={`flex h-6 items-center gap-1 px-1 ${dim !== null ? 'opacity-35' : ''}`}
            >
              {body}
            </span>
          );
        })}
      </div>
      {state.density === 'overview' && hollow && (
        <span
          className="whitespace-nowrap text-[10px] italic text-base-content/45"
          title="At Overview density every hollow state is drawn as a 1px line"
        >
          {CAVEATS.overviewMerged}
        </span>
      )}
    </div>
  );
}

function StatusBar() {
  const {model, state, pointer} = useLoom();
  const snap = usePointer(pointer);
  const target = activeTarget(snap);
  const metric: Metric = model.countsStatus === 'unavailable' ? 'pres' : state.metric;
  const readout = useReadout(target, metric);
  const spoken = target?.source === 'keyboard' && readout ? readout.spoken : '';
  const legend = useLegend(metric);

  // The legend stays inline while the readout keeps enough room; otherwise it
  // folds into "Legend ▴".
  const barRef = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(1440);
  useLayoutEffect(() => {
    const el = barRef.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const legendW = (legend.bins.size > 0 && metric !== 'pres' ? RAMP_W : 0) + legend.entries.length * ENTRY_W;
  const inline = width - legendW >= FOCUS_MIN;

  return (
    <footer
      ref={barRef}
      className="relative z-[45] flex h-8 shrink-0 items-center gap-3 border-t border-base-300 px-4 text-[11px]"
      aria-label="Focus readout and legend"
    >
      <Caps className="shrink-0" tone="text-base-content/45">
        Focus
      </Caps>
      <p className="loom-tabular min-w-0 flex-1 truncate text-base-content/85" title={readout?.full || undefined}>
        {readout ? (
          readout.text
        ) : (
          <span className="text-base-content/45">
            Point at or focus a cell to read it here · a blank cell is not coded with the concept · ? explains every
            glyph
          </span>
        )}
      </p>
      <span className="sr-only" aria-live="polite" aria-atomic="true">
        {spoken}
      </span>
      {inline ? (
        <div className="shrink-0">
          <Legend metric={metric} data={legend} />
        </div>
      ) : (
        <div className="shrink-0">
          <Popover
            direction="up"
            align="right"
            ariaLabel="Legend"
            title={`Legend of this view: ${[
              ...(legend.bins.size > 0 && metric !== 'pres' ? [METRIC_LABEL[metric]] : []),
              ...legend.entries.map(e => e.label)
            ].join(', ')}. Click an entry to dim the rest.`}
            panelClassName="w-[260px] p-3"
            buttonClassName="inline-flex h-6 items-center gap-1 rounded border border-base-content/15 px-1.5 text-[11px] text-base-content/70 hover:border-base-content/30 hover:text-base-content"
            label={
              <>
                Legend <ChevronUp size={11} aria-hidden="true" />
              </>
            }
          >
            {() => <Legend metric={metric} data={legend} stacked />}
          </Popover>
        </div>
      )}
    </footer>
  );
}

// ============================================================================
// ToastHost
// ============================================================================

// Bottom-centre toast (ui.toast), with Undo when the action can be undone.
// It stays 6 s (the timer pauses while the pointer or focus is on it), fades
// in and out in 180 ms, and does not move under reduced motion.

const LIFETIME_MS = 6000;
const FADE_MS = 180;

function ToastHost() {
  const {ui, setUi} = useLoom();
  const toast = ui.toast;
  // The toast on screen: kept through the fade-out after ui.toast clears.
  const [shown, setShown] = useState<Toast | null>(toast);
  const [visible, setVisible] = useState(false);
  const [paused, setPaused] = useState(false);
  const remaining = useRef(LIFETIME_MS);

  useEffect(() => {
    if (toast) {
      setShown(toast);
      remaining.current = LIFETIME_MS;
      const id = requestAnimationFrame(() => setVisible(true));
      return () => cancelAnimationFrame(id);
    }
    setVisible(false);
    const id = window.setTimeout(() => setShown(null), FADE_MS);
    return () => window.clearTimeout(id);
  }, [toast]);

  // Auto-dismiss: only this toast (a newer one keeps its own timer).
  useEffect(() => {
    if (!toast || paused) return;
    const started = Date.now();
    const id = window.setTimeout(
      () => setUi(prev => (prev.toast?.id === toast.id ? {toast: null} : {})),
      remaining.current
    );
    return () => {
      window.clearTimeout(id);
      remaining.current = Math.max(1000, remaining.current - (Date.now() - started));
    };
  }, [toast, paused, setUi]);

  if (!shown) return null;
  const dismiss = () => setUi(prev => (prev.toast?.id === shown.id ? {toast: null} : {}));
  const undo = () => {
    shown.undo?.();
    dismiss();
  };

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-[92px] z-[60] flex justify-center px-4">
      <div
        role="status"
        aria-live="polite"
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
        onFocus={() => setPaused(true)}
        onBlur={() => setPaused(false)}
        className={`pointer-events-auto flex max-w-[min(560px,100%)] items-center gap-3 rounded-lg bg-neutral py-2 pl-4 pr-2 text-sm text-neutral-content shadow-lg transition duration-[180ms] ease-out motion-reduce:translate-y-0 motion-reduce:transition-none ${
          visible ? 'translate-y-0 opacity-100' : 'translate-y-2 opacity-0'
        }`}
      >
        <span className="min-w-0 flex-1">{shown.text}</span>
        {shown.undo && (
          <button
            type="button"
            className="btn btn-ghost btn-xs text-neutral-content underline-offset-2 hover:underline"
            onClick={undo}
            title="Undo this change"
          >
            Undo
          </button>
        )}
        <button
          type="button"
          className="btn btn-ghost btn-xs btn-square text-neutral-content"
          onClick={dismiss}
          aria-label="Dismiss"
          title="Dismiss"
        >
          <X size={14} />
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// concept-coverage
// ============================================================================

// Loom: every concept, threaded through every cohort. Rows are concepts
// (variables of any cohort sharing a concept code or OMOP ID), columns are all
// registered cohorts grouped by what we know about them, and cells show the
// observation counts of the cohorts that went through EDA. The page owns the
// URL-backed view state and derives everything once per change; the
// components under components/loom/ only read it (LoomContext).

type PageAction = LoomAction | {type: 'load'; state: LoomViewState};

const pageReducer = (state: LoomViewState, action: PageAction): LoomViewState =>
  action.type === 'load' ? action.state : reduceViewState(state, action);

const INITIAL_UI: LoomUiState = {
  inspector: null,
  legendDim: null,
  railOpen: false,
  poolOpen: false,
  ledgerOpen: false,
  helpOpen: false,
  toast: null,
  basketPreview: null
};

// Canonical query string of a state, to tell our own URL writes from
// back / forward navigation.
const canonicalQuery = (state: LoomViewState): string => new URLSearchParams(encodeViewState(state)).toString();

const REVEAL_FLAG = 'loom-revealed';
const REVEAL_MS = 520;

// useLayoutEffect warns during server rendering; the page is server-rendered.
const useIsoLayoutEffect = typeof window === 'undefined' ? useEffect : useLayoutEffect;

// The model is rebuilt when the counts settle (or fail, or are retried) and
// when the Match setting changes; row, cohort and variable indices differ
// between builds. These carry what the reader has open over to the new build
// by stable identity (row tokens, cohort id, cohort id + variable name), or
// return -1 when it no longer exists.
function remapRow(from: LoomModel, to: LoomModel, row: number): number {
  const r = from.rows[row];
  if (!r) return -1;
  for (const t of [r.key, ...r.tokens]) {
    const next = to.tokenToRow.get(t);
    if (next !== undefined) return next;
  }
  return -1;
}

function remapCohort(from: LoomModel, to: LoomModel, cohort: number): number {
  const c = from.cohorts[cohort];
  return c ? to.cohortIndex.get(c.id) ?? -1 : -1;
}

function remapVariable(from: LoomModel, to: LoomModel, variable: number | null): number | null {
  if (variable === null) return null;
  const v = from.variables[variable];
  const cohort = v ? remapCohort(from, to, v.cohort) : -1;
  if (cohort < 0) return null;
  const next = to.variables.findIndex(w => w.cohort === cohort && w.name === v.name);
  return next >= 0 ? next : null;
}

function remapTarget(from: LoomModel, to: LoomModel, t: InspectorTarget): InspectorTarget | null {
  if (t.kind === 'cohort') {
    const cohort = remapCohort(from, to, t.cohort);
    return cohort >= 0 ? {kind: 'cohort', cohort} : null;
  }
  const row = remapRow(from, to, t.row);
  if (row < 0) return null;
  if (t.kind === 'concept') return {kind: 'concept', row};
  const cohort = remapCohort(from, to, t.cohort);
  return cohort >= 0 ? {kind: 'cell', row, cohort} : null;
}

function readTheme(): 'light' | 'dark' {
  if (typeof document === 'undefined') return 'light';
  const html = document.documentElement;
  return html.getAttribute('data-theme') === 'dark' || html.classList.contains('dark') ? 'dark' : 'light';
}

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

export default function LoomPage() {
  const router = useRouter();
  const {cohortsData, userEmail, isLoading, dataCleanRoom, setDataCleanRoom, semanticMatchIndex} = useCohorts();
  const hasData = !!cohortsData && Object.keys(cohortsData).length > 0;
  const {counts, status: countsStatus, retry: retryCounts} = useObservationCounts(hasData);

  const [state, dispatchPage] = useReducer(pageReducer, DEFAULT_VIEW_STATE);
  const dispatch = useCallback((action: LoomAction) => dispatchPage(action), []);
  const [ui, setUiState] = useState<LoomUiState>(INITIAL_UI);
  const setUi = useCallback(
    (patch: Partial<LoomUiState> | ((prev: LoomUiState) => Partial<LoomUiState>)) =>
      setUiState(prev => ({...prev, ...(typeof patch === 'function' ? patch(prev) : patch)})),
    []
  );
  const [pointer] = useState(createPointerStore);
  const gridApi = useRef<GridApi | null>(null);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [semanticVar, setSemanticVar] = useState<number | null>(null);
  const [distributionVar, setDistributionVar] = useState<number | null>(null);
  const [revealProgress, setRevealProgress] = useState<number | null>(null);

  // --- URL <-> state -------------------------------------------------------
  const loaded = useRef(false);
  const lastQuery = useRef<string | null>(null);

  useEffect(() => {
    if (!router.isReady) return;
    const decoded = decodeViewState(router.query);
    const canonical = canonicalQuery(decoded);
    if (loaded.current && canonical === lastQuery.current) return; // our own write
    loaded.current = true;
    lastQuery.current = canonical;
    dispatchPage({type: 'load', state: decoded});
  }, [router.isReady, router.query]);

  useEffect(() => {
    if (!loaded.current) return;
    const canonical = canonicalQuery(state);
    if (canonical === lastQuery.current) return;
    const timer = window.setTimeout(() => {
      lastQuery.current = canonical;
      router.replace({pathname: router.pathname, query: encodeViewState(state)}, undefined, {shallow: true, scroll: false});
    }, 300);
    return () => window.clearTimeout(timer);
    // router is stable enough for replace; depending on it would re-run on every navigation
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  // --- Theme and layout ----------------------------------------------------
  useEffect(() => {
    setTheme(readTheme());
    const observer = new MutationObserver(() => setTheme(readTheme()));
    observer.observe(document.documentElement, {attributes: true, attributeFilter: ['class', 'data-theme']});
    return () => observer.disconnect();
  }, []);

  // The page fills the viewport below the nav; the matrix scrolls inside.
  const rootRef = useRef<HTMLDivElement>(null);
  const [top, setTop] = useState(64);
  useIsoLayoutEffect(() => {
    const measure = () => {
      const el = rootRef.current;
      if (el) setTop(Math.max(0, Math.round(el.getBoundingClientRect().top + window.scrollY)));
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [hasData]);

  // --- Derived model -------------------------------------------------------
  const basket: Record<string, string[]> = useMemo(() => dataCleanRoom?.cohorts || {}, [dataCleanRoom]);

  const model: LoomModel | null = useMemo(
    () => (hasData ? buildModel(cohortsData, counts, countsStatus, state.match) : null),
    [hasData, cohortsData, counts, countsStatus, state.match]
  );

  // Carry the open inspector target and modals over to a rebuilt model before
  // the new build paints (their indices belong to the previous build).
  const prevModel = useRef<LoomModel | null>(null);
  useIsoLayoutEffect(() => {
    const from = prevModel.current;
    prevModel.current = model;
    if (!from || !model || from === model) return;
    setUi(prev => {
      if (!prev.inspector) return {};
      return {inspector: remapTarget(from, model, prev.inspector)};
    });
    setSemanticVar(v => remapVariable(from, model, v));
    setDistributionVar(v => remapVariable(from, model, v));
  }, [model, setUi]);

  const cells = useMemo(
    () => (model ? resolveLens(model, state.visit, state.floor, state.guests, basket) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [model, state.visit.kind, state.visit.mask, state.floor, state.guests, basket]
  );

  const density = DENSITY[state.density];
  const columnKey = JSON.stringify([
    state.tiers,
    state.fog,
    state.eda,
    state.families,
    state.designs,
    state.statuses,
    state.size,
    state.sizeUnknown,
    state.hiddenCohorts,
    state.onlyCohorts,
    state.colGroup,
    state.colSort,
    state.anchor,
    state.density
  ]);
  const columns = useMemo(
    () => (model ? buildColumns(model, state, density) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [model, columnKey]
  );

  const stats = useMemo(
    () => (model && cells && columns ? computeRowStats(model, cells, columns.visibleMask) : null),
    [model, cells, columns]
  );

  const search = useMemo(() => (model ? parseSearch(state.q, model) : null), [model, state.q]);
  const pulledKeys = useMemo(() => new Set(state.pool.flatMap(k => k.split('~'))), [state.pool]);

  const filter = useMemo(
    () =>
      model && cells && stats && columns && search
        ? filterRows(model, cells, stats, columns, state, search, pulledKeys)
        : null,
    [model, cells, stats, columns, state, search, pulledKeys]
  );

  // Row order. With "lock order" on (the default), rows are sorted once per
  // sort setting over ALL rows and filters only hide rows from that order, so
  // nothing reshuffles under the reader's eyes; "Re-sort" refreshes it after
  // the lens or the visible cohorts changed what the sort keys mean.
  const sortKey = `${state.rowSort}|${state.sortCohort}|${state.anchor}|${state.rowGroup}`;
  // What the sort keys are computed over: when it changes, a locked order is stale.
  const basisKey = `${state.visit.kind}:${state.visit.mask}|${state.floor}|${state.guests}|${columnKey}`;
  const locked = useRef<{model: LoomModel | null; key: string; basis: string; order: number[]}>({
    model: null,
    key: '',
    basis: '',
    order: []
  });
  const [resortTick, setResortTick] = useState(0);
  const lastResort = useRef(0);
  const order = useMemo(() => {
    if (!model || !cells || !stats || !filter) return [];
    if (state.lock) {
      const lock = locked.current;
      if (lock.model !== model || lock.key !== sortKey || lastResort.current !== resortTick) {
        lastResort.current = resortTick;
        const all = new Uint8Array(model.rows.length).fill(1);
        locked.current = {model, key: sortKey, basis: basisKey, order: orderRows(model, cells, stats, state, all, null)};
      }
      return locked.current.order.filter(r => filter.visible[r] === 1);
    }
    return orderRows(model, cells, stats, state, filter.visible, null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, cells, stats, filter, state.lock, sortKey, resortTick]);
  const orderStale = state.lock && locked.current.model === model && locked.current.basis !== basisKey;
  const resort = useCallback(() => setResortTick(t => t + 1), []);

  const layout = useMemo(
    () => (model && stats ? buildRowLayout(model, stats, order, state, density) : null),
    [model, stats, order, state, density]
  );

  const pool = useMemo(() => (model && columns ? computePool(model, state, columns) : null), [model, state, columns]);

  // The matrix axes (one cell = one concept in one cohort at one visit).
  const slotMasks = useMemo(() => (model ? cohortSlotMasks(model) : null), [model]);
  const axes = useMemo(
    () => (model && columns && slotMasks ? buildAxes(model, columns, order, state, slotMasks) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [model, columns, order, slotMasks, state.visit, state.transpose]
  );

  // --- Deep links: r=<token> scrolls to a row, i=<token>@<cohort> opens a cell
  useEffect(() => {
    if (!model || !layout || (!state.focusRowToken && !state.inspectToken)) return;
    const rowOf = (token: string | null) => (token ? model.tokenToRow.get(token.toLowerCase()) ?? -1 : -1);
    const focusRow = rowOf(state.focusRowToken);
    if (state.inspectToken) {
      const at = state.inspectToken.lastIndexOf('@');
      const row = rowOf(at > 0 ? state.inspectToken.slice(0, at) : state.inspectToken);
      const cohort = at > 0 ? model.cohortIndex.get(state.inspectToken.slice(at + 1)) ?? -1 : -1;
      if (row >= 0) setUi({inspector: cohort >= 0 ? {kind: 'cell', row, cohort} : {kind: 'concept', row}});
      if (row >= 0) window.setTimeout(() => gridApi.current?.revealRow(row), 60);
    } else if (focusRow >= 0) {
      window.setTimeout(() => gridApi.current?.revealRow(focusRow), 60);
    }
    dispatch({type: 'patch', patch: {focusRowToken: null, inspectToken: null}});
  }, [model, layout, state.focusRowToken, state.inspectToken, dispatch, setUi]);

  // --- Shuttle reveal: once per session, dictionaries show as rings until the
  // counts arrive, then the profiled columns fill left to right.
  const sawLoading = useRef(false);
  useEffect(() => {
    if (countsStatus === 'loading') sawLoading.current = true;
    if (countsStatus !== 'ok' || !sawLoading.current || prefersReducedMotion()) return;
    try {
      if (sessionStorage.getItem(REVEAL_FLAG)) return;
      sessionStorage.setItem(REVEAL_FLAG, '1');
    } catch {
      return;
    }
    let frame = 0;
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / REVEAL_MS);
      setRevealProgress(t);
      if (t < 1) frame = requestAnimationFrame(step);
      else setRevealProgress(null);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [countsStatus]);

  // --- ?debug=1: the invariants of the concept grouping -------------------
  useEffect(() => {
    if (!model || !state.debug || countsStatus === 'loading') return;
    const result = runSelfCheck(model, semanticMatchIndex);
    console.info(`[loom] model built in ${model.stats.buildMs.toFixed(1)} ms · ${result.summary}`);
    if (result.violations.length) console.warn('[loom] self-check violations', result.violations);
  }, [model, state.debug, semanticMatchIndex, countsStatus]);

  // --- DCR basket ----------------------------------------------------------
  const previewBasket = useCallback((preview: BasketPreview) => setUi({basketPreview: preview}), [setUi]);

  const commitBasket = useCallback(
    (preview: BasketPreview) => {
      // Other pages mutate the basket object in place: keep a deep copy to undo to.
      const before: DcrBasket = JSON.parse(JSON.stringify(dataCleanRoom || {cohorts: {}}));
      const after = applyPreview(before, preview);
      setDataCleanRoom(after);
      persistBasket(after);
      const variables = preview.byCohort.reduce((sum, c) => sum + c.names.length, 0);
      const cohorts = preview.byCohort.filter(c => c.names.length > 0).length;
      setUi({
        basketPreview: null,
        toast: {
          id: Date.now(),
          text: `Added ${variables} variable${variables === 1 ? '' : 's'} from ${cohorts} cohort${cohorts === 1 ? '' : 's'} to the Data Clean Room basket`,
          undo: () => {
            setDataCleanRoom(before);
            persistBasket(before);
          }
        }
      });
    },
    [dataCleanRoom, setDataCleanRoom, setUi]
  );

  // --- Links out to the Explore page's modals --------------------------------
  const semanticModal = useMemo(() => {
    if (semanticVar === null || !model) return null;
    const v = model.variables[semanticVar];
    const cohortId = model.cohorts[v.cohort].id;
    const variable = cohortsData?.[cohortId]?.variables?.[v.name];
    const matches = semanticMatchIndex?.byVariable?.get(semanticMatchKey(cohortId, v.name));
    return variable && matches ? {cohortId, variable, matches} : null;
  }, [semanticVar, model, cohortsData, semanticMatchIndex]);

  const openSemanticMatches = useCallback(
    (variable: number) => {
      if (!model) return;
      const v = model.variables[variable];
      const cohortId = model.cohorts[v.cohort].id;
      if (semanticMatchIndex?.byVariable?.get(semanticMatchKey(cohortId, v.name))) setSemanticVar(variable);
      else
        setUi({
          toast: {
            id: Date.now(),
            text: `${v.name} has no semantic matches on the Explore page (Concept Coverage also joins codes that differ only in spacing or format)`
          }
        });
    },
    [model, semanticMatchIndex, setUi]
  );

  const revealRow = useCallback((row: number) => gridApi.current?.revealRow(row), []);
  const revealCohort = useCallback((cohort: number) => gridApi.current?.revealCohort(cohort), []);

  const ctx: LoomContextValue | null = useMemo(
    () =>
      model && cells && stats && columns && search && filter && layout && axes
        ? {
            model,
            cells,
            stats,
            columns,
            search,
            filter,
            layout,
            pool,
            state,
            dispatch,
            ui,
            setUi,
            density,
            pointer,
            basket,
            previewBasket,
            commitBasket,
            retryCounts,
            theme,
            revealRow,
            revealCohort,
            openSemanticMatches,
            openDistribution: setDistributionVar,
            revealProgress,
            axes
          }
        : null,
    [
      model,
      cells,
      stats,
      columns,
      search,
      filter,
      layout,
      pool,
      state,
      dispatch,
      ui,
      setUi,
      density,
      pointer,
      basket,
      previewBasket,
      commitBasket,
      retryCounts,
      theme,
      revealRow,
      revealCohort,
      openSemanticMatches,
      revealProgress,
      axes
    ]
  );

  const distribution = distributionVar !== null && model ? model.variables[distributionVar] : null;

  let body: React.ReactNode;
  if (userEmail === null) {
    body = <LoginPrompt message="Authenticate to see Concept Coverage" />;
  } else if (!ctx) {
    body = (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 text-base-content/60" role="status">
        <span className="loading loading-dots loading-md" />
        <span className="text-sm">{isLoading || !hasData ? 'Stringing the loom: loading the data dictionaries…' : 'Weaving…'}</span>
      </div>
    );
  } else {
    body = (
      <LoomContext.Provider value={ctx}>
        <LoomHeader />
        <CommandBar />
        <ViewBar />
        <RecipeBar />
        <div className="relative flex min-h-0 flex-1">
          <FilterRail />
          <main className="relative min-w-0 flex-1" aria-label="Concept by cohort matrix">
            {state.view === 'table' ? <LoomTable /> : <CoverageGrid apiRef={gridApi} />}
            {orderStale && state.view !== 'table' && (
              <button
                type="button"
                className="btn btn-xs absolute right-4 top-2 z-20 gap-1 border-base-300 bg-base-100 font-normal shadow-sm"
                title="The row order is locked so that filtering never reshuffles rows. The visit lens or the visible cohorts changed since it was computed: click to sort again."
                onClick={resort}
              >
                Order kept · Re-sort
              </button>
            )}
          </main>
          <Inspector />
        </div>
        <PoolDock />
        <StatusBar />
        <DcrAddPreview />
        <HelpSheet />
        <ToastHost />
      </LoomContext.Provider>
    );
  }

  return (
    <>
      <Head>
        <title>Concept Coverage · Cohort Explorer</title>
      </Head>
      <div
        ref={rootRef}
        className="loom-root flex flex-col overflow-hidden"
        style={{height: `calc(100vh - ${top}px)`}}
        data-theme-mode={theme}
      >
        {body}
      </div>
      {semanticModal && (
        <SemanticMatchesModal
          cohortId={semanticModal.cohortId}
          variable={semanticModal.variable}
          semanticMatches={semanticModal.matches}
          onClose={() => setSemanticVar(null)}
        />
      )}
      {distribution && model && (
        <VariableGraphModal
          isOpen
          cohortId={model.cohorts[distribution.cohort].id}
          variableName={distribution.name}
          variableLabel={distribution.label}
          omopId={distribution.omopId}
          conceptCode={distribution.conceptCode}
          onClose={() => setDistributionVar(null)}
        />
      )}
    </>
  );
}
