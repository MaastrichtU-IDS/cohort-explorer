// Concept Coverage (page /concept-coverage): the logic behind the cross-cohort
// concept x cohort matrix of observation counts - the derived model, cell
// rules, filters, search, ordering, layout, URL state, the Pool and the DCR
// basket helpers. Pure TypeScript apart from the counts hook; the page
// (pages/concept-coverage.tsx) renders it.

import {type Variable, type Cohort} from '@/types';
import {useCallback, useEffect, useState} from 'react';
import {apiUrl} from '@/utils';
import {type SemanticMatchIndex} from '@/utils/semanticMatches';

// ============================================================================
// types
// ============================================================================

// Loom: the cross-cohort concept x cohort matrix (page /loom).
//
// This file is the CONTRACT shared by every Loom module. The design is in the
// build spec (sections referenced as "spec §n"). Rules that every module
// follows:
//
// 1. Unknown is never drawn as absent; uncounted is never drawn as zero; an
//    unknown quantity never gets length.
// 2. Only data wears colour: the blue ramp means "observations were counted".
// 3. Colour follows the entity: bins are fixed and absolute, filters never
//    repaint the cells that remain.
// 4. Never sum the variables inside a cell: each cell shows one
//    representative variable; pooled figures are bounds.
// 5. Every mark is reachable without hover (FOCUS bar, keyboard, table).
// 6. The page informs, it never gates (no policy checks before the basket).

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

// Knowledge tier of a cohort. 'profiled' = dictionary + EDA counts,
// 'dictionary' = dictionary only (counts unknown), 'none' = no dictionary
// (nothing known about its variables).
export type Tier = 'profiled' | 'dictionary' | 'none';

export type CountsStatus = 'loading' | 'ok' | 'unavailable';

// Why a cohort's tier is qualified (tooltips / column notes).
export type TierNote = 'loading' | 'eda-unreadable' | 'counts-unavailable' | 'counts-without-dictionary';

export const VISIT_SLOTS = ['PRE', 'BL', 'M1', 'M3', 'M6', 'M9', 'M12', 'M18', 'M24', 'M36', 'END', 'UN'] as const;
export type VisitSlot = (typeof VISIT_SLOTS)[number];
// Bit of a slot in a slot mask: 1 << slotIndex(slot).
export const SLOT_BIT: Record<VisitSlot, number> = VISIT_SLOTS.reduce(
  (acc, s, i) => ({...acc, [s]: 1 << i}),
  {} as Record<VisitSlot, number>
);
export const ALL_SLOTS_MASK = (1 << VISIT_SLOTS.length) - 1;
// Follow-up slots (M1..END), used by the "Any follow-up" lens.
export const FOLLOW_UP_MASK = ['M1', 'M3', 'M6', 'M9', 'M12', 'M18', 'M24', 'M36', 'END'].reduce(
  (m, s) => m | SLOT_BIT[s as VisitSlot],
  0
);

// Domain bands, in display order (spec §4.4).
export const DOMAINS = [
  'Person',
  'Measurement',
  'Condition',
  'Drug',
  'Observation',
  'Procedure',
  'Death',
  'Visit',
  'Device',
  'Other'
] as const;
export type Domain = (typeof DOMAINS)[number];

export type TypeClass = 'num' | 'cat' | 'text' | 'date';

// Study-type families in continuum order (spec §7.2).
export const STUDY_FAMILIES = ['pop', 'risk', 'cvdckd', 'mi', 'prehf', 'hf', 'other', 'unspec'] as const;
export type StudyFamily = (typeof STUDY_FAMILIES)[number];
export const DESIGN_FAMILIES = ['rct', 'obs', 'retro', 'single', 'unspec'] as const;
export type DesignFamily = (typeof DESIGN_FAMILIES)[number];
export const COHORT_STATUSES = ['yes', 'no', 'unknown'] as const;
export type CohortStatus = (typeof COHORT_STATUSES)[number];

// Cell knowledge state (spec §4.3). Plain numeric constants (not a const
// enum: isolatedModules). Stored in LensCells.state.
export const CellState = {
  NotCoded: 0, // cohort has a dictionary but no member in this row (under the lens)
  Counted: 1, // profiled, representative n > 0
  Zero: 2, // profiled, every member has n = 0
  NotInEda: 3, // profiled cohort, no member found in the EDA output
  Dictionary: 4, // dictionary-only cohort, member(s) exist, count unknown
  Pending: 5, // counts loading / unavailable (cohort has members)
  OtherVisit: 6, // lens active: not coded at the lens, but coded at another slot
  Fog: 7 // no-dictionary cohort: unknown (drawn as a hatched column layer)
} as const;
export type CellStateValue = (typeof CellState)[keyof typeof CellState];

// Cell flag bits, stored in LensCells.flags.
export const CellFlag = {
  Multi: 1, // >= 2 home members under the lens (dog-ear)
  BelowFloor: 2, // counted, but completeness below the "usable" floor (drawn as a 3px bar)
  GuestOnly: 4, // only guest members (legacy pipe-list secondary codes): drawn at 60%
  InBasket: 8, // at least one member is in the DCR basket
  InBasketAll: 16, // every member is in the DCR basket
  Broad: 32 // >= 3 home members in one slot, or >= 12 members in total
} as const;

export type Metric = 'comp' | 'n' | 'pres';
export type Density = 'overview' | 'compact' | 'comfortable';
export type RowGroup = 'domain' | 'coverage' | 'none';
export type RowSort = 'coverage' | 'pooled' | 'az' | 'cohort' | 'anchor';
export type ColGroup = 'tier' | 'type' | 'design' | 'status' | 'none';
export type ColSort = 'size' | 'name' | 'coverage' | 'anchor';
export type FogMode = 'fold' | 'cols' | 'hide';
export type SetMode = 'all' | 'any' | 'only' | 'abo';
export type PinKind = 'coded' | 'data' | 'not';
export type MatchMode = 'both' | 'code' | 'omop';

// ---------------------------------------------------------------------------
// Derived model (built once per data load by utils/loom/buildModel.ts)
// ---------------------------------------------------------------------------

export interface LoomCohort {
  i: number; // index in model.cohorts
  id: string;
  tier: Tier;
  note?: TierNote;
  edaVersion: string | null; // from the counts response ('v1' | 'v2'), else cohort.eda_version
  nRows: number | null; // EDA dataset rows (profiled only)
  declared: number | null; // leading integer of study_participants
  size: number | null; // nRows ?? declared
  studyTypeRaw: string;
  family: StudyFamily;
  designRaw: string;
  design: DesignFamily;
  status: CohortStatus;
  institution: string;
  nVars: number; // dictionary variables
  nMatched: number; // variables whose home row spans >= 2 dictionary cohorts
  nSingle: number; // variables whose home row spans only this cohort
  nUnmapped: number; // variables with no valid identifier
  nNotInEda: number; // profiled: dictionary variables absent from the EDA output
}

export interface LoomVariable {
  i: number; // index in model.variables
  cohort: number; // index in model.cohorts
  name: string; // var_name as in the dictionary
  label: string; // var_label
  type: TypeClass;
  domain: Domain;
  domainRaw: string;
  slot: VisitSlot;
  slotRaw: string; // visit_concept_name, else visits
  units: string;
  conceptName: string;
  conceptCode: string; // raw
  omopId: string; // raw
  tokens: string[]; // every normalized token ('cc:loinc:718-7', 'oi:3000963')
  head: string[]; // head tokens (cc[0], oi[0])
  home: number; // row index, -1 = unmapped
  guestRows: number[]; // rows it is a guest of (legacy pipe-list secondary tokens)
  // n >= 0 counted; -1 not in the EDA output (cohort profiled); -2 cohort not
  // profiled (dictionary only); -3 counts loading / unavailable.
  n: number;
  empty: number;
  missing: number;
  caseCollision: string | null; // another var of the same cohort equal after lowercasing
  declaredCount: string | null; // dictionary COUNT column, shown only as "declared (unverified)"
  eligibleRank: 0 | 1 | 2; // 0 = type & domain match the row's; 1 = other non-date; 2 = date
}

export interface RowFlags {
  bridged: boolean; // > 1 distinct cc head or > 1 distinct oi head
  suspect: boolean; // > 6 distinct cc tokens or > 6 distinct oi tokens
  broad: boolean; // some cell has >= 3 home members in one slot or >= 12 members
  unitsDiffer: boolean; // > 1 distinct normalized non-empty units across cohorts
  hasGuests: boolean;
  mixedDomain: Domain[]; // other domains voted by some cohorts
}

export interface LoomRow {
  i: number;
  key: string; // canonical token: 'oi:<smallest id>' else 'cc:<smallest code>'
  tokens: string[]; // every token of the component (sorted)
  label: string; // winning concept name (original casing)
  altLabels: string[];
  domain: Domain;
  type: TypeClass;
  flags: RowFlags;
  slotMask: number; // slots present among home members
  cohortMask: Uint8Array; // length K: 1 if the cohort has >= 1 home member (any slot)
  dictCoverage: number; // number of cohorts with >= 1 home member (any slot)
  nVars: number; // home members
  haystack: string; // lowercased: label, alt labels, tokens, member names/labels (for search)
}

export interface LoomModel {
  cohorts: LoomCohort[]; // ALL registered cohorts
  cohortIndex: Map<string, number>; // cohort id -> index
  rows: LoomRow[];
  variables: LoomVariable[];
  unmapped: number[]; // variable indices with no valid identifier
  tokenToRow: Map<string, number>;
  // Members per (row, cohort), CSR over key row * K + cohort (K = cohorts.length):
  // cellItems[cellStart[key] .. cellStart[key + 1]) are variable indices,
  // home members first (by slot order, then n desc), then guests.
  K: number;
  cellStart: Uint32Array; // length R * K + 1
  cellItems: Uint32Array;
  cellHomeCount: Uint16Array; // length R * K: home members per cell
  slotsPresent: number; // mask of slots occurring among mapped variables
  slotVarCounts: Record<VisitSlot, number>; // mapped variables per slot (scrubber histogram base)
  countsStatus: CountsStatus;
  match: MatchMode;
  stats: {buildMs: number; variables: number; mapped: number; malformed: number};
}

// ---------------------------------------------------------------------------
// Visit lens and per-cell resolution (utils/loom/cells.ts)
// ---------------------------------------------------------------------------

// kind 'all' = every slot; 'slots' = the slots in mask.
export interface VisitLens {
  kind: 'all' | 'slots';
  mask: number;
}

export interface LensCells {
  lensKey: string; // `${kind}:${mask}:${floor}` for memo checks
  R: number;
  K: number;
  // All arrays have length R * K, indexed row * K + cohort.
  state: Uint8Array; // CellState
  flags: Uint8Array; // CellFlag bits (InBasket bits are filled by applyBasket)
  k: Uint16Array; // home members under the lens
  lo: Uint32Array; // representative n (max eligible n) under the lens; 0 when not counted
  hi: Uint32Array; // min(N, sum of eligible n) under the lens
  rep: Int32Array; // representative variable index, -1 none
  otherSlots: Uint16Array; // OtherVisit: slot mask where the concept IS coded in this cohort
}

// Per-row statistics over the VISIBLE columns (utils/loom/cells.ts
// computeRowStats). Unknown quantities never add length (spec §6).
export interface RowStats {
  covDict: Uint16Array; // visible dictionary/profiled cohorts with >= 1 home member under the lens
  covMeasured: Uint16Array; // visible profiled cohorts with lo > 0 and not BelowFloor
  pooledLo: Float64Array; // sum of lo over visible profiled cohorts (not BelowFloor)
  pooledHi: Float64Array; // sum of hi over the same cohorts
  dictOnly: Uint16Array; // visible dictionary-only cohorts coding the row
  dictOnlyDeclared: Float64Array; // their declared participants (tooltip text only)
}

// ---------------------------------------------------------------------------
// View state (URL-backed; utils/loom/urlState.ts encodes / decodes it)
// ---------------------------------------------------------------------------

export interface PinState {
  cohortId: string;
  kind: PinKind;
}

// A Pool slot: one concept, or a braid (several rows joined by '~').
export type PoolSlotKey = string; // row keys joined by '~'

export interface LoomViewState {
  // Cohorts (columns)
  tiers: Record<Tier, boolean>; // tier=p,d,n  (default all true)
  fog: FogMode; // fog=cols|hide (default fold)
  eda: {v1: boolean; v2: boolean}; // eda=v1 | v2 (default both)
  families: StudyFamily[] | null; // st= (null = all)
  designs: DesignFamily[] | null; // des=
  statuses: CohortStatus[] | null; // on=
  size: [number, number] | null; // sz=100-100000 (null = full range)
  sizeUnknown: boolean; // szu=0 (default true: include unknown size)
  hiddenCohorts: string[]; // hide=
  onlyCohorts: string[]; // only=
  // Concepts (rows)
  domains: Domain[] | null; // dom=
  minCohorts: number; // min= (default 2)
  minProfiledOnly: boolean; // minp=1
  types: TypeClass[] | null; // typ=
  floor: number; // floor= 0..100 (default 0)
  flagFilter: {bridged: boolean; broad: boolean; units: boolean; hideBroad: boolean}; // fl=b,w,u,hw
  hiddenRows: string[]; // hr= (row keys)
  pins: PinState[]; // pin=TIME-CHF,*BIOSTAT-CHF,!Believe
  setMode: SetMode; // set= (default all)
  q: string; // q=
  qMode: 'filter' | 'highlight'; // qm=h
  guests: boolean; // gst=0 (default true)
  match: MatchMode; // match= (default both)
  pooledMin: number; // pn= (default 0)
  // Encoding / layout
  metric: Metric; // m= (default comp)
  visit: VisitLens; // v=all|bl|m6|bl-m6|fu|un|... (default all)
  density: Density; // d=o|c|l (default l: cells large enough to print their count)
  rowGroup: RowGroup; // rows= (default domain)
  rowSort: RowSort; // rs= (default coverage)
  sortCohort: string | null; // rsc= cohort for rowSort 'cohort'
  colGroup: ColGroup; // cols= (default tier)
  colSort: ColSort; // cs= (default size)
  lock: boolean; // lock=0 (default true)
  collapsedBands: Domain[]; // cb=
  unfolded: string[]; // uf= row keys with visit sub-rows open
  view: 'loom' | 'table'; // view=table
  minimap: boolean; // mm=0 (default true)
  transpose: boolean; // tr=1: concepts as rows, cohort visits as columns (default: the other way round)
  anchor: string | null; // anc=
  // Pool
  pool: PoolSlotKey[]; // pool=oi:1;oi:2~cc:x
  poolVisit: VisitLens; // pv= (default BL)
  poolThreshold: number; // pt= 0..100
  poolHandoff: 'rep' | 'all'; // ph=all
  poolIncludeDict: boolean; // pd=0
  // Deep link targets (consumed once, then cleared)
  focusRowToken: string | null; // r=
  inspectToken: string | null; // i=<token>@<cohortId>
  debug: boolean; // debug=1
}

export type LoomAction =
  | {type: 'patch'; patch: Partial<LoomViewState>}
  | {type: 'reset'}
  | {type: 'cyclePin'; cohortId: string; profiled: boolean} // ○ → ✓ → ✓✓ (profiled only) → ✕ → ○
  | {type: 'setPin'; cohortId: string; kind: PinKind | null}
  | {type: 'toggleHiddenRow'; key: string}
  | {type: 'toggleHiddenCohort'; cohortId: string}
  | {type: 'soloCohort'; cohortId: string} // only this one (toggle)
  | {type: 'toggleBand'; domain: Domain}
  | {type: 'toggleUnfold'; key: string}
  | {type: 'pull'; keys: string[]} // add Pool slots (row keys)
  | {type: 'release'; key: PoolSlotKey}
  | {type: 'clearPool'};

// ---------------------------------------------------------------------------
// Columns (utils/loom/layout.ts buildColumns)
// ---------------------------------------------------------------------------

export interface DensitySpec {
  cellW: number;
  cellH: number;
  pitchX: number;
  pitchY: number;
  radius: number;
  gap: number;
}
export const DENSITY: Record<Density, DensitySpec> = {
  overview: {cellW: 5, cellH: 3, pitchX: 6, pitchY: 4, radius: 0, gap: 1},
  compact: {cellW: 14, cellH: 12, pitchX: 16, pitchY: 14, radius: 2, gap: 2},
  comfortable: {cellW: 20, cellH: 18, pitchX: 22, pitchY: 20, radius: 3, gap: 2}
};
export const GROUP_GAP = 8; // px between column groups
export const FOG_FOLD_W = 28; // folded no-dictionary column
export const FOG_COL_W = 8; // one expanded no-dictionary column
export const BAND_H = 24; // domain band header row
export const TALLY_H = 16; // collapsed band tally row

export interface GridColumn {
  kind: 'cohort' | 'fogFold';
  cohort: number; // cohort index; -1 for the fold
  x: number; // left edge in grid coordinates
  w: number;
  group: number; // index into ColumnLayout.groups
}

export interface ColumnLayout {
  columns: GridColumn[]; // left to right
  groups: {key: string; label: string; count: number; x: number; w: number}[];
  width: number; // total grid width in px
  visibleMask: Uint8Array; // per cohort index: 1 if shown as a DICTIONARY/PROFILED column (fog never counts)
  visibleFog: number[]; // no-dictionary cohorts passing the cohort filters (fold or columns)
  cohortToColumn: Int32Array; // per cohort index: index into columns, -1 if not shown
  hiddenByFilter: number; // cohorts removed by the cohort filters
}

// ---------------------------------------------------------------------------
// Row filtering (utils/loom/filters.ts) and search (utils/loom/search.ts)
// ---------------------------------------------------------------------------

export interface SearchQuery {
  raw: string;
  // Free terms: every group must match (AND); a group matches if any of its
  // alternatives appears in the haystack (OR, from 'a | b'). Phrases are
  // alternatives with spaces.
  groups: string[][];
  excludes: string[]; // -term
  domains: Domain[]; // d: / domain:
  inCohorts: string[]; // in:
  notInCohorts: string[]; // notin:
  codes: string[]; // code: ('*' wildcard) - matched against row tokens (cc:...)
  omops: string[]; // omop:
  visits: VisitSlot[]; // v: / visit:
  types: TypeClass[]; // t: / type:
  is: ('bridged' | 'broad' | 'units' | 'pulled')[];
  chips: {text: string; negated: boolean; start: number; end: number}[]; // for rendering tokens
  empty: boolean;
}

export interface FilterChip {
  id: string;
  label: string; // e.g. "Domain: Measurement, Condition"
  cost: number; // rows whose ONLY failing filter is this one (-N)
  clear: LoomAction; // dispatching it removes the filter
}

export interface FilterResult {
  visible: Uint8Array; // per row: 1 if shown
  visibleCount: number;
  highlight: Uint8Array | null; // search Highlight mode: 1 = match (others dim to 25%)
  matchCount: number; // rows matching the search (either mode)
  chips: FilterChip[];
  recipe: string; // "Concepts coded in ALL of TIME-CHF, BIOSTAT-CHF (with data) · not coded in Believe · at Baseline"
  singleCohortShownBySearch: number; // single-cohort rows shown because the search matched them
  ledger: NotShownLedger;
}

export interface NotShownLedger {
  totalRows: number;
  byFilter: {id: string; label: string; rows: number}[]; // rows hidden by each chip (sole cause)
  multiCause: number; // rows hidden by 2+ filters
  singleCohortRows: number;
  unmappedVariables: number;
  malformedIdentifiers: number;
  fogCohorts: {count: number; declared: number};
  notInEdaVariables: number;
}

// ---------------------------------------------------------------------------
// Row layout (utils/loom/layout.ts buildRowLayout)
// ---------------------------------------------------------------------------

export type FlatItem =
  | {kind: 'band'; domain: Domain | null; label: string; count: number; collapsed: boolean; y: number; h: number}
  | {kind: 'tally'; domain: Domain | null; rows: number[]; y: number; h: number}
  | {kind: 'row'; row: number; y: number; h: number}
  | {kind: 'sub'; row: number; slot: VisitSlot; y: number; h: number};

export interface RowLayout {
  items: FlatItem[];
  totalHeight: number;
  rowToItem: Int32Array; // per row index: index into items, -1 if not laid out
  order: number[]; // visible rows in display order
}

// ---------------------------------------------------------------------------
// Pool (utils/loom/pool.ts)
// ---------------------------------------------------------------------------

export interface PoolSlot {
  key: PoolSlotKey;
  rows: number[]; // rows.length > 1 = braid
  label: string;
}

export type PoolStatus = 'qualifies' | 'nearMiss' | 'unknown' | 'dictionary' | 'lacks';

export interface PoolCohort {
  cohort: number;
  status: PoolStatus;
  lower: number; // Fréchet lower bound on rows with all slots
  upper: number; // min over slots of hi
  N: number | null;
  limitingSlot: string | null; // label of the slot that sets `upper`
  missing: string[]; // labels of slots lacking (lacks / nearMiss) or unknown
  perSlot: {lo: number; hi: number; repVars: number[]; allVars: number[]; state: 'counted' | 'zero' | 'notInEda' | 'dictionary' | 'absent'}[];
}

export interface PoolResult {
  slots: PoolSlot[];
  visit: VisitLens;
  cohorts: PoolCohort[]; // visible dictionary cohorts, sorted: qualifies (by upper desc), nearMiss, unknown, dictionary, lacks
  qualifying: number;
  pooledLower: number;
  pooledUpper: number;
  dictionaryOnly: {count: number; declared: number}; // dictionary-only cohorts coding every slot
  fog: {count: number; declared: number};
  dropToGain: {slot: string; plusCohorts: number; plusUpper: number}[];
  summary: string; // one paragraph, plain text, for "Copy summary"
}

// ---------------------------------------------------------------------------
// Ephemeral UI state (not in the URL)
// ---------------------------------------------------------------------------

export type InspectorTarget =
  | {kind: 'cell'; row: number; cohort: number}
  | {kind: 'concept'; row: number}
  | {kind: 'cohort'; cohort: number};

export interface Toast {
  id: number;
  text: string;
  undo?: () => void;
}

export interface BasketPreview {
  // variables to add, per cohort id
  byCohort: {cohortId: string; names: string[]; alreadyIn: number; tier: Tier}[];
  source: string; // e.g. "Hemoglobin · TIME-CHF" or "Pool (3 concepts)"
}

export interface LoomUiState {
  inspector: InspectorTarget | null;
  legendDim: string | null; // 'bin:<1..5>' | 'state:<CellState>' | 'flag:multi' | null
  railOpen: boolean;
  poolOpen: boolean;
  ledgerOpen: boolean;
  helpOpen: boolean;
  toast: Toast | null;
  basketPreview: BasketPreview | null;
}

// Hover / keyboard focus live OUTSIDE React state (no re-render per
// mousemove): a tiny external store (components/loom/pointerStore.ts) that the
// grid writes and the StatusBar / HoverLayer read with useSyncExternalStore.
export interface PointerTarget {
  kind: 'cell' | 'row' | 'column' | 'band';
  row: number; // -1 when not applicable
  cohort: number; // -1 when not applicable
  item: number; // index into RowLayout.items, -1 when not applicable
  slot?: VisitSlot; // sub-rows
  source: 'mouse' | 'keyboard';
}

// ============================================================================
// bins
// ============================================================================

// The sequential ramp (spec §4.2): one blue hue, five FIXED absolute bins, so
// a cell's colour follows the entity and never changes when filters change.
// Same values as the --loom-bin-* tokens of .loom-root in globals.css (the
// canvas reads the tokens; these tables serve non-DOM code and fallbacks).
export const RAMP_LIGHT: string[] = ['#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
// Dark mode is designed, not flipped: more is brighter.
export const RAMP_DARK: string[] = ['#1c5cab', '#2a78d6', '#5598e7', '#86b6ef', '#cde2fb'];

// Bin 1..5 of a counted cell, 0 = no ramp colour: the Presence metric, or
// Completeness when the dataset size is unknown (drawn as the presence fill).
// Integer comparisons, so 95% is exactly the edge of bin 5.
export function binFor(metric: Metric, lo: number, N: number | null): number {
  if (metric === 'pres' || lo <= 0) return 0;
  if (metric === 'n') {
    if (lo < 50) return 1;
    if (lo < 250) return 2;
    if (lo < 1000) return 3;
    if (lo < 5000) return 4;
    return 5;
  }
  if (N === null || N <= 0) return 0;
  const pct = lo * 100;
  if (pct < 25 * N) return 1;
  if (pct < 50 * N) return 2;
  if (pct < 80 * N) return 3;
  if (pct < 95 * N) return 4;
  return 5;
}

export const BIN_LABELS: {comp: string[]; n: string[]} = {
  comp: ['< 25%', '25–50%', '50–80%', '80–95%', '≥ 95%'],
  n: ['1–49', '50–249', '250–999', '1k–5k', '≥ 5k']
};

export const METRIC_LABEL: Record<Metric, string> = {
  comp: 'Completeness',
  n: 'Count',
  pres: 'Presence'
};

// Legend titles (spec §4.5).
export const METRIC_EXPLAINER: Record<Metric, string> = {
  comp: 'Completeness = rows with a value ÷ dataset rows',
  n: 'Values counted (absolute)',
  pres: 'Presence'
};

export interface LoomTokens {
  bins: string[];
  mark: string;
  inkFill: string;
  softFill: string;
  fog: string;
  hairline: string;
  ink: string;
  wash: string;
  accent: string;
  surface: string;
  text: string;
}

// Light-theme values of the tokens, used when an element is not (yet) inside
// .loom-root and the custom properties resolve to nothing.
const LIGHT_FALLBACK: Omit<LoomTokens, 'bins'> = {
  mark: '#6b7280',
  inkFill: '#4b5563',
  softFill: '#9ca3af',
  fog: '#d1d5db',
  hairline: '#e5e7eb',
  ink: '#1f2937',
  wash: 'rgba(31, 41, 55, 0.06)',
  accent: '#2a78d6',
  surface: '#ffffff',
  text: '#1f2937'
};

// The canvas's colours, read from the CSS tokens of the .loom-root element
// (re-read on theme change). surface / text are the element's computed
// background and text colour, i.e. the active theme's base-100 / base-content.
export function readLoomTokens(el: Element): LoomTokens {
  const cs = getComputedStyle(el);
  const token = (name: string, fallback: string): string => cs.getPropertyValue(name).trim() || fallback;
  const background = cs.backgroundColor;
  const transparent = !background || background === 'transparent' || background === 'rgba(0, 0, 0, 0)';
  return {
    bins: RAMP_LIGHT.map((hex, i) => token(`--loom-bin-${i + 1}`, hex)),
    mark: token('--loom-mark', LIGHT_FALLBACK.mark),
    inkFill: token('--loom-ink-fill', LIGHT_FALLBACK.inkFill),
    softFill: token('--loom-soft-fill', LIGHT_FALLBACK.softFill),
    fog: token('--loom-fog', LIGHT_FALLBACK.fog),
    hairline: token('--loom-hairline', LIGHT_FALLBACK.hairline),
    ink: token('--loom-ink', LIGHT_FALLBACK.ink),
    wash: token('--loom-wash', LIGHT_FALLBACK.wash),
    accent: token('--loom-accent', LIGHT_FALLBACK.accent),
    surface: transparent ? LIGHT_FALLBACK.surface : background,
    text: cs.color || LIGHT_FALLBACK.text
  };
}

// ============================================================================
// cohortFacets
// ============================================================================

// Cohort facets (spec §7.2): the size used by the size filter, and the ordered
// rules that fold the free-text study type / design / status fields into a
// few families. Kept in one reviewable file: each rule is first-match-wins on
// the lowercased value, and the typo-tolerant patterns are deliberate.

// Leading integer of study_participants: "73883 total participants; ..." ->
// 73883, "4,256" / "4 256" / "4.256" -> 4256. A separator only counts when it
// groups thousands, so "1480 2020" reads 1480, not 14802020. 0 or no number
// -> null (unknown).
export function parseDeclared(raw: unknown): number | null {
  if (raw === null || raw === undefined) return null;
  if (typeof raw === 'number') return Number.isFinite(raw) && raw > 0 ? Math.round(raw) : null;
  const m = /^\s*(\d{1,3}(?:[,.\s]\d{3})+(?!\d)|\d+)/.exec(String(raw));
  if (!m) return null;
  const n = Number(m[1].replace(/[,.\s]/g, ''));
  return Number.isFinite(n) && n > 0 ? n : null;
}

// Continuum order: Population -> Early risk -> CVD/CKD -> Post-MI/CAD ->
// Pre-HF -> Heart failure -> Other -> Unspecified. The order of the rules is
// NOT the continuum order: "CVD/CKD but no structural/functional heart
// disease" must be caught before the structural rule, and "structural heart
// disease but no heart failure" before the heart-failure rule.
const FAMILY_RULES: [RegExp, StudyFamily][] = [
  [/(cardio|cadio)vascular dis|chronic kidney/, 'cvdckd'], // "Cadiovascular" typo
  [/structural|functional|ventricular dysfunction/, 'prehf'],
  [/heart failure|\bhf\b/, 'hf'],
  [/myocardial infarction|coronary|\bcad\b/, 'mi'],
  [/early risk/, 'risk'],
  [/population|house\w*hold/, 'pop'] // "houselhold" typo (Believe)
];

export function studyFamily(raw: string | null | undefined): StudyFamily {
  const s = (raw ?? '').trim().toLowerCase();
  if (!s) return 'unspec';
  for (const [re, family] of FAMILY_RULES) if (re.test(s)) return family;
  return 'other';
}

const DESIGN_RULES: [RegExp, DesignFamily][] = [
  [/ran?d?omi[sz]ed|\brct\b/, 'rct'], // "ranomized" typo
  [/single.?arm|cross.?over/, 'single'],
  [/retrospect/, 'retro'],
  [/observ/, 'obs']
];

export function designFamily(raw: string | null | undefined): DesignFamily {
  const s = (raw ?? '').trim().toLowerCase();
  if (!s) return 'unspec';
  for (const [re, design] of DESIGN_RULES) if (re.test(s)) return design;
  return 'unspec';
}

// study_ongoing: 'yes' / 'no' / '' in the metadata sheet.
export function cohortStatus(raw: string | null | undefined): CohortStatus {
  const s = String(raw ?? '')
    .trim()
    .toLowerCase();
  if (s === 'yes' || s === 'y' || s === 'true' || s === 'ongoing') return 'yes';
  if (s === 'no' || s === 'n' || s === 'false' || s === 'completed') return 'no';
  return 'unknown';
}

export const FAMILY_LABEL: Record<StudyFamily, string> = {
  pop: 'Population',
  risk: 'Early risk',
  cvdckd: 'CVD/CKD',
  mi: 'Post-MI/CAD',
  prehf: 'Pre-HF',
  hf: 'Heart failure',
  other: 'Other',
  unspec: 'Unspecified'
};

export const DESIGN_LABEL: Record<DesignFamily, string> = {
  rct: 'RCT',
  obs: 'Observational',
  retro: 'Retrospective',
  single: 'Single-arm / crossover',
  unspec: 'Unspecified'
};

export const STATUS_LABEL: Record<CohortStatus, string> = {
  yes: 'Ongoing',
  no: 'Completed',
  unknown: 'Unknown'
};

export const TIER_LABEL: Record<Tier, string> = {
  profiled: 'Profiled',
  dictionary: 'Dictionary only',
  none: 'No dictionary'
};

// ============================================================================
// domains
// ============================================================================

// OMOP domain of a dictionary variable -> Loom domain band (spec §5 step 7).
// The raw value is lowercased and runs of spaces / underscores become one '_'
// ('Condition_occurrence', 'condition occurrence' -> 'condition_occurrence').
const DOMAIN_MAP: Record<string, Domain> = {
  measurement: 'Measurement',
  condition_occurrence: 'Condition',
  condition_era: 'Condition',
  condition: 'Condition',
  drug_exposure: 'Drug',
  drug_era: 'Drug',
  drug: 'Drug',
  observation: 'Observation',
  procedure_occurrence: 'Procedure',
  procedure: 'Procedure',
  person: 'Person',
  visit_occurrence: 'Visit',
  visit_detail: 'Visit',
  visit: 'Visit',
  observation_period: 'Visit',
  death: 'Death',
  device_exposure: 'Device',
  device: 'Device'
};

export function normalizeDomain(raw: string | null | undefined): Domain {
  const key = String(raw ?? '')
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, '_');
  return DOMAIN_MAP[key] ?? 'Other';
}

export const DOMAIN_MONOGRAM: Record<Domain, string> = {
  Person: 'P',
  Measurement: 'M',
  Condition: 'C',
  Drug: 'D',
  Observation: 'O',
  Procedure: 'R',
  Death: '†',
  Visit: 'V',
  Device: 'E',
  Other: '·'
};

export const DOMAIN_LABEL: Record<Domain, string> = {
  Person: 'Person',
  Measurement: 'Measurement',
  Condition: 'Condition',
  Drug: 'Drug',
  Observation: 'Observation',
  Procedure: 'Procedure',
  Death: 'Death',
  Visit: 'Visit & period',
  Device: 'Device',
  Other: 'Other'
};

// ============================================================================
// identifiers
// ============================================================================

// Identifier normalization for Loom (spec §5 steps 1-2). Stricter than the
// Explore badges (utils/semanticMatches.ts): inner whitespace is removed
// ('loinc: 2276-4' = 'loinc:2276-4'), a trailing '.0' is stripped from OMOP
// IDs, and SNOMED prefix spellings are aliased. Every rule only ever MERGES
// values the Explore page keeps apart, never splits values it joins, so Loom
// rows are a superset of Explore's semantic matches (verified by selfCheck.ts).

// Placeholders that mean "no identifier". '0' is OMOP's "no matching concept"
// sentinel: kept, it would link every unmapped variable.
const EMPTY_VALUES = new Set(['', 'na', 'n/a', 'nan', 'null', 'none', '-', '0']);

// Vocabulary prefixes written several ways in the dictionaries.
const VOCABULARY_ALIASES: Record<string, string> = {snomedct: 'snomed', 'snomed-ct': 'snomed', sct: 'snomed'};

const normalizeValue = (raw: string): string => raw.replace(/\s+/g, '').toLowerCase();

// Written positions are kept (null = empty placeholder) so that two parallel
// lists still pair by index when one of them has a gap.
function splitPositions(raw: unknown): (string | null)[] {
  if (raw === null || raw === undefined) return [];
  const text = String(raw);
  if (!text.trim()) return [];
  return text.split('|').map(part => {
    const v = normalizeValue(part);
    return EMPTY_VALUES.has(v) ? null : v;
  });
}

export function splitIdentifierTokens(raw: unknown): string[] {
  return splitPositions(raw).filter((v): v is string => v !== null);
}

export function ccToken(value: string): string | null {
  const v = normalizeValue(value);
  if (EMPTY_VALUES.has(v)) return null;
  const colon = v.indexOf(':');
  // An unprefixed code is kept as written ("no vocabulary prefix" in the
  // inspector): a prefix is never inferred.
  if (colon < 0) return `cc:${v}`;
  const prefix = v.slice(0, colon);
  const code = v.slice(colon + 1);
  // 'loinc:' or 'loinc:na' carries no code.
  if (EMPTY_VALUES.has(code)) return null;
  return `cc:${VOCABULARY_ALIASES[prefix] ?? prefix}:${code}`;
}

export function oiToken(value: string): string | null {
  const v = normalizeValue(value).replace(/\.0+$/, '');
  if (!/^\d+$/.test(v)) return null;
  // OMOP IDs are integers: '03000963' is 3000963.
  const id = v.replace(/^0+/, '');
  return id ? `oi:${id}` : null;
}

export interface VariableIdentity {
  cc: string[]; // concept code tokens, written order (duplicates kept)
  oi: string[]; // OMOP ID tokens, written order (duplicates kept)
  head: string[]; // cc head and/or oi head: they decide the home row
  // Code <-> ID pairs to union: the heads, or every index of two parallel lists.
  pairs: [string, string][];
  malformed: number; // values present but not a valid identifier (e.g. OMOP 'abc')
}

// Spec §5 step 2. Pipe lists are legacy (upload rejects '|' now). Two lists of
// equal written length k >= 2 are PARALLEL: cc[i] pairs with oi[i]. Otherwise
// only the heads (first valid value of each list) pair, and every other token
// is secondary: the variable is a guest in that token's row, and tokens of one
// list are never paired with each other, so a composite variable cannot glue
// two concept families together.
export function variableIdentity(
  v: Pick<Variable, 'concept_code' | 'omop_id'>,
  match: MatchMode = 'both'
): VariableIdentity {
  let malformed = 0;
  const ccPos: (string | null)[] = [];
  const oiPos: (string | null)[] = [];
  if (match !== 'omop') {
    for (const raw of splitPositions(v.concept_code)) {
      if (raw === null) ccPos.push(null);
      else {
        const t = ccToken(raw);
        if (!t) malformed++;
        ccPos.push(t);
      }
    }
  }
  if (match !== 'code') {
    for (const raw of splitPositions(v.omop_id)) {
      if (raw === null) oiPos.push(null);
      else {
        const t = oiToken(raw);
        if (!t) malformed++;
        oiPos.push(t);
      }
    }
  }
  const cc = ccPos.filter((t): t is string => t !== null);
  const oi = oiPos.filter((t): t is string => t !== null);
  const head: string[] = [];
  const pairs: [string, string][] = [];
  if (ccPos.length >= 2 && ccPos.length === oiPos.length) {
    // Head = the first written index holding a valid value.
    for (let i = 0; i < ccPos.length; i++) {
      const c = ccPos[i];
      const o = oiPos[i];
      if (!head.length && (c || o)) {
        if (c) head.push(c);
        if (o) head.push(o);
      }
      if (c && o) pairs.push([c, o]);
    }
  } else {
    if (cc.length) head.push(cc[0]);
    if (oi.length) head.push(oi[0]);
    if (cc.length && oi.length) pairs.push([cc[0], oi[0]]);
  }
  return {cc, oi, head, pairs, malformed};
}

export function variableTokens(v: Variable, match: MatchMode): {cc: string[]; oi: string[]; malformed: number} {
  const {cc, oi, malformed} = variableIdentity(v, match);
  return {cc, oi, malformed};
}

export function tokenLabel(token: string): string {
  if (token.startsWith('oi:')) return `OMOP ${token.slice(3)}`;
  if (token.startsWith('cc:')) return token.slice(3);
  return token;
}

// A concept code written without 'vocabulary:' (shown as a caution in the inspector).
export const lacksVocabularyPrefix = (token: string): boolean => token.startsWith('cc:') && !token.includes(':', 3);

// Numeric order of OMOP tokens ('oi:9' < 'oi:10'), lexicographic for codes.
export function compareTokens(a: string, b: string): number {
  const ao = a.startsWith('oi:');
  const bo = b.startsWith('oi:');
  if (ao !== bo) return ao ? -1 : 1;
  if (ao && a.length !== b.length) return a.length - b.length;
  return a < b ? -1 : a > b ? 1 : 0;
}

// ============================================================================
// visits
// ============================================================================

// Visit slots (spec §5 step 8): the dictionary's visit_concept_name (else the
// visits column) folded onto a small timeline. The first matching rule wins
// and the raw value is always kept next to the slot. Visits are never parsed
// from variable names.

const PRE_RE = /prior to baseline|pre-?baseline|before baseline|screening/;
const BL_RE = /baseline|month 0\b|inclusion|enrol|randomi[sz]ation/;
// "18 months", "1-year", "12 weeks" ...
const DURATION_RE = /(\d+(?:\.\d+)?)\s*-?\s*(day|week|month|year)s?/;
// ... and the unit-first spelling of the visits column ("month 6", "week 12").
const DURATION_UNIT_FIRST_RE = /\b(day|week|month|year)s?\s*-?\s*(\d+(?:\.\d+)?)/;
const END_RE = /end of (study|follow)|final visit|close-?out|last visit/;

const MONTH_STOPS: [number, VisitSlot][] = [
  [1, 'M1'],
  [3, 'M3'],
  [6, 'M6'],
  [9, 'M9'],
  [12, 'M12'],
  [18, 'M18'],
  [24, 'M24']
];

const toMonths = (value: number, unit: string): number =>
  unit === 'day' ? value / 30.44 : unit === 'week' ? value / 4.35 : unit === 'year' ? value * 12 : value;

// Nearest stop; a tie goes to the earlier stop. >= 30 months is "36+".
// Under half a month ("0 days") is the baseline itself.
function snapMonths(months: number): VisitSlot {
  if (months >= 30) return 'M36';
  if (months < 0.5) return 'BL';
  let best = MONTH_STOPS[0];
  for (const stop of MONTH_STOPS) if (Math.abs(stop[0] - months) < Math.abs(best[0] - months)) best = stop;
  return best[1];
}

export function visitSlot(visitConceptName: unknown, visits: unknown): VisitSlot {
  const primary = String(visitConceptName ?? '')
    .trim()
    .toLowerCase();
  const s =
    primary ||
    String(visits ?? '')
      .trim()
      .toLowerCase();
  if (!s) return 'UN';
  if (PRE_RE.test(s)) return 'PRE';
  if (BL_RE.test(s)) return 'BL';
  const d = DURATION_RE.exec(s);
  if (d) return snapMonths(toMonths(Number(d[1]), d[2]));
  const u = DURATION_UNIT_FIRST_RE.exec(s);
  if (u) return snapMonths(toMonths(Number(u[2]), u[1]));
  if (END_RE.test(s)) return 'END';
  // "visit date", "days of visit", empty: not tied to a study timepoint.
  return 'UN';
}

export const SLOT_SHORT: Record<VisitSlot, string> = {
  PRE: 'Pre',
  BL: 'BL',
  M1: '1m',
  M3: '3m',
  M6: '6m',
  M9: '9m',
  M12: '12m',
  M18: '18m',
  M24: '24m',
  M36: '36+',
  END: 'End',
  UN: 'Un'
};

export const SLOT_LONG: Record<VisitSlot, string> = {
  PRE: 'Before baseline',
  BL: 'Baseline',
  M1: '1 month',
  M3: '3 months',
  M6: '6 months',
  M9: '9 months',
  M12: '12 months',
  M18: '18 months',
  M24: '24 months',
  M36: '36+ months',
  END: 'End of study',
  UN: 'Unanchored'
};

// Month value of the month slots (for "1–6 months" range labels).
const SLOT_MONTHS: Partial<Record<VisitSlot, string>> = {
  M1: '1',
  M3: '3',
  M6: '6',
  M9: '9',
  M12: '12',
  M18: '18',
  M24: '24',
  M36: '36+'
};

const lowerFirst = (s: string): string => s.charAt(0).toLowerCase() + s.slice(1);

// In-sentence form: "6 m", "baseline", "end of study" ("coded at 6 m, 12 m").
export function slotPhrase(s: VisitSlot): string {
  const m = SLOT_MONTHS[s];
  return m ? `${m} m` : lowerFirst(SLOT_LONG[s]);
}

const SLOT_INDEX = VISIT_SLOTS.reduce((acc, s, i) => ({...acc, [s]: i}), {} as Record<VisitSlot, number>);

export function slotIndex(s: VisitSlot): number {
  return SLOT_INDEX[s];
}

export function slotsOfMask(mask: number): VisitSlot[] {
  return VISIT_SLOTS.filter(s => (mask & SLOT_BIT[s]) !== 0);
}

export const ALL_LENS: VisitLens = {kind: 'all', mask: ALL_SLOTS_MASK};

export function lensIncludes(lens: VisitLens, slot: VisitSlot): boolean {
  return lens.kind === 'all' || (lens.mask & SLOT_BIT[slot]) !== 0;
}

// True when the lens is the whole timeline (kind 'all', or every slot picked).
export const isAllLens = (lens: VisitLens): boolean =>
  lens.kind === 'all' || (lens.mask & ALL_SLOTS_MASK) === ALL_SLOTS_MASK;

export function lensLabel(lens: VisitLens): string {
  if (isAllLens(lens)) return 'All visits';
  const mask = lens.mask & ALL_SLOTS_MASK;
  if (mask === 0) return 'No visit';
  if (mask === FOLLOW_UP_MASK) return 'Any follow-up';
  const slots = slotsOfMask(mask);
  if (slots.length === 1) return SLOT_LONG[slots[0]];
  // A contiguous stretch of the timeline reads as a range. Unanchored is off
  // the timeline (after the divider), so it never joins a range.
  const first = slotIndex(slots[0]);
  const last = slotIndex(slots[slots.length - 1]);
  const contiguous = last - first === slots.length - 1 && !slots.includes('UN');
  if (contiguous) {
    const a = slots[0];
    const b = slots[slots.length - 1];
    const am = SLOT_MONTHS[a];
    const bm = SLOT_MONTHS[b];
    if (am && bm) return `${am}–${bm} months`;
    return `${SLOT_LONG[a]}–${lowerFirst(SLOT_LONG[b])}`;
  }
  return slots.map((s, i) => (i === 0 ? SLOT_LONG[s] : lowerFirst(SLOT_LONG[s]))).join(', ');
}

// ============================================================================
// observationCounts
// ============================================================================

// Observation counts of the cohorts that went through EDA (variable
// profiling), from GET /observation-counts. Only those cohorts have real
// counts; the dictionaries' own COUNT / NA columns are declared by the data
// owners and are not used for them.

export interface VariableObservationCount {
  // Non-empty, non-coded-missing values.
  n: number;
  // Blank cells.
  empty: number;
  // Cells holding a declared missing-value code.
  missing: number;
}

export interface CohortObservationCounts {
  eda_version: 'v1' | 'v2' | string;
  // Rows of the profiled dataset (null when the EDA file does not say).
  n_rows: number | null;
  // Entries listed in the EDA file, profiled or not.
  n_listed?: number;
  // ISO date of the EDA file on the server ("profiled on").
  generated_at?: string;
  // Keyed by lowercased, trimmed variable name.
  variables: Record<string, VariableObservationCount>;
}

export type ObservationCounts = Record<string, CohortObservationCounts>;

export async function fetchObservationCounts(): Promise<ObservationCounts> {
  const res = await fetch(`${apiUrl}/observation-counts`, {credentials: 'include'});
  if (!res.ok) throw new Error(`observation counts: HTTP ${res.status}`);
  const data = await res.json();
  return data && typeof data === 'object' ? (data as ObservationCounts) : {};
}

export const countFor = (
  counts: ObservationCounts | null | undefined,
  cohortId: string,
  varName: string
): VariableObservationCount | undefined => counts?.[cohortId]?.variables[String(varName).trim().toLowerCase()];

// One request per page session, shared by every caller; a failed request is
// forgotten so retry() asks again.
let pending: Promise<ObservationCounts> | null = null;

export type ObservationCountsStatus = 'loading' | 'ok' | 'unavailable';

export function useObservationCounts(enabled = true): {
  counts: ObservationCounts | null;
  status: ObservationCountsStatus;
  retry: () => void;
} {
  const [counts, setCounts] = useState<ObservationCounts | null>(null);
  const [status, setStatus] = useState<ObservationCountsStatus>('loading');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    setStatus('loading');
    if (!pending) pending = fetchObservationCounts();
    const mine = pending;
    mine
      .then(data => {
        if (!alive) return;
        setCounts(data);
        setStatus('ok');
      })
      .catch(err => {
        if (pending === mine) pending = null;
        console.error(err);
        if (!alive) return;
        setCounts(null);
        setStatus('unavailable');
      });
    return () => {
      alive = false;
    };
  }, [enabled, attempt]);

  const retry = useCallback(() => {
    pending = null;
    setAttempt(a => a + 1);
  }, []);

  return {counts, status, retry};
}

// ============================================================================
// buildModel
// ============================================================================

// The Loom model (spec §5): rows are the connected components of a union-find
// over normalized identifier tokens, joined only through code <-> ID pairs.
// Built once per data load (and once more when the counts arrive); every view
// (lens, filters, order) is derived from it without touching this structure.

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

const str = (v: unknown): string => (v === null || v === undefined ? '' : String(v).trim());

const TIER_RANK: Record<Tier, number> = {profiled: 0, dictionary: 1, none: 2};
const DOMAIN_RANK = DOMAINS.reduce((acc, d, i) => ({...acc, [d]: i}), {} as Record<Domain, number>);
// Tie order of the type vote.
const TYPE_CLASSES: readonly TypeClass[] = ['num', 'cat', 'text', 'date'];
const TYPE_RANK: Record<TypeClass, number> = {num: 0, cat: 1, text: 2, date: 3};
const SLOT_RANK = VISIT_SLOTS.reduce((acc, s) => ({...acc, [s]: slotIndex(s)}), {} as Record<VisitSlot, number>);

// Entries of the cohorts payload that are not cohorts (e.g. 'userEmail') are skipped.
function isCohortLike(c: unknown): c is Cohort {
  if (!c || typeof c !== 'object' || Array.isArray(c)) return false;
  return 'cohort_id' in c || 'variables' in c;
}

function typeClass(v: Variable): TypeClass {
  const t = str(v.var_type).toUpperCase();
  if (t.startsWith('DATE')) return 'date';
  if (Array.isArray(v.categories) && v.categories.length > 0) return 'cat';
  if (t === 'INT' || t === 'INTEGER' || t === 'FLOAT' || t === 'DOUBLE' || t === 'NUMERIC' || t === 'DECIMAL')
    return 'num';
  return 'text';
}

// Label vote key: lowercase, quotes stripped, punctuation and whitespace runs
// collapsed, so "Hemoglobin [Mass/volume] in Blood" and "hemoglobin mass/volume
// in blood" vote together.
const normLabel = (s: string): string =>
  s
    .toLowerCase()
    .replace(/["'`´‘’“”]/g, '')
    .replace(/[\s!#-/:-@[-^_{-~–—]+/g, ' ')
    .trim();

const UNIT_PLACEHOLDERS = new Set(['', 'na', 'n/a', 'nan', 'null', 'none', '-']);
const normUnits = (s: string): string => {
  const u = s.toLowerCase().replace(/\s+/g, '').replace(/[µμ]/g, 'u');
  return UNIT_PLACEHOLDERS.has(u) ? '' : u;
};

const validCount = (x: unknown): number | null => {
  const n = typeof x === 'number' ? x : typeof x === 'string' && x.trim() ? Number(x) : NaN;
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : null;
};

// "count 248 · NA 2": the data owner's own figures, shown only as declared (unverified).
function declaredCountOf(v: Variable): string | null {
  const count = validCount(v.count);
  if (count === null) return null;
  const na = validCount(v.na);
  return na === null ? `count ${count}` : `count ${count} · NA ${na}`;
}

// Most frequent key of a tally; ties broken by `tie` (negative = a first).
function argmax<K>(tally: Map<K, number>, tie: (a: K, b: K) => number): K | undefined {
  let best: K | undefined;
  let bestN = -1;
  tally.forEach((n, k) => {
    if (n > bestN || (n === bestN && best !== undefined && tie(k, best) < 0)) {
      best = k;
      bestN = n;
    }
  });
  return best;
}

function memo<T>(fn: (key: string) => T): (key: string) => T {
  const cache = new Map<string, T>();
  return key => {
    let value = cache.get(key);
    if (value === undefined) {
      value = fn(key);
      cache.set(key, value);
    }
    return value;
  };
}

// A variable's distinct tokens, codes first, in written order.
function uniqueTokens(cc: string[], oi: string[]): string[] {
  const out: string[] = [];
  for (const t of cc) if (!out.includes(t)) out.push(t);
  for (const t of oi) if (!out.includes(t)) out.push(t);
  return out;
}

// Index of the largest tally; ties go to the lower index (band / type order).
function argmaxIndex(tally: Uint16Array): number {
  let best = 0;
  for (let i = 1; i < tally.length; i++) if (tally[i] > tally[best]) best = i;
  return best;
}

const bump = <K>(m: Map<K, number>, k: K, by = 1): void => {
  m.set(k, (m.get(k) ?? 0) + by);
};

const byLengthThenAlpha = (a: string, b: string): number => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0);

interface Draft {
  v: LoomVariable;
  secondary: string[]; // tokens that are not heads (legacy pipe lists)
}

export function buildModel(
  cohortsData: Record<string, Cohort>,
  counts: ObservationCounts | null,
  status: CountsStatus,
  match: MatchMode = 'both'
): LoomModel {
  const t0 = now();

  // --- Cohorts and tiers (spec §5 step 11: tiers come from the counts response).
  const response: ObservationCounts | null = status === 'ok' ? (counts ?? {}) : null;
  const cohortEntries: {cohort: LoomCohort; vars: [string, Variable][]}[] = [];
  for (const [id, raw] of Object.entries(cohortsData ?? {})) {
    if (!isCohortLike(raw)) continue;
    const vars: [string, Variable][] =
      raw.variables && typeof raw.variables === 'object'
        ? Object.entries(raw.variables).filter(
            (e): e is [string, Variable] => !!e[1] && typeof e[1] === 'object' && !Array.isArray(e[1])
          )
        : [];
    const own = response && Object.prototype.hasOwnProperty.call(response, id) ? response[id] : undefined;
    const entry =
      own && typeof own === 'object' && own.variables && typeof own.variables === 'object' ? own : undefined;
    let tier: Tier;
    let note: TierNote | undefined;
    if (!vars.length) {
      tier = 'none';
      if (counts && Object.prototype.hasOwnProperty.call(counts, id)) note = 'counts-without-dictionary';
    } else if (status === 'loading') {
      // Provisional until the response says: the EDA marker only places the
      // column (so it does not jump when counts arrive); every cell is Pending.
      tier = raw.eda_version ? 'profiled' : 'dictionary';
      note = 'loading';
    } else if (status === 'unavailable') {
      tier = 'dictionary';
      note = 'counts-unavailable';
    } else if (entry) {
      tier = 'profiled';
    } else {
      tier = 'dictionary';
      if (raw.eda_version) note = 'eda-unreadable';
    }
    const nRows = tier === 'profiled' && entry ? validCount(entry.n_rows) || null : null;
    const declared = parseDeclared(raw.study_participants);
    const cohort: LoomCohort = {
      i: -1,
      id,
      tier,
      note,
      edaVersion: entry?.eda_version ? String(entry.eda_version) : (raw.eda_version ?? null),
      nRows,
      declared,
      size: nRows ?? declared,
      studyTypeRaw: str(raw.study_type),
      family: studyFamily(raw.study_type),
      designRaw: str(raw.study_design),
      design: designFamily(raw.study_design),
      status: cohortStatus(raw.study_ongoing),
      institution: str(raw.institution),
      nVars: vars.length,
      nMatched: 0,
      nSingle: 0,
      nUnmapped: 0,
      nNotInEda: 0
    };
    cohortEntries.push({cohort, vars});
  }
  // Stable base order: tier, then size (unknown last), then id.
  cohortEntries.sort((a, b) => {
    const x = a.cohort;
    const y = b.cohort;
    return (
      TIER_RANK[x.tier] - TIER_RANK[y.tier] ||
      (x.size === null ? 1 : 0) - (y.size === null ? 1 : 0) ||
      (y.size ?? 0) - (x.size ?? 0) ||
      (x.id < y.id ? -1 : x.id > y.id ? 1 : 0)
    );
  });
  const cohorts = cohortEntries.map((e, i) => {
    e.cohort.i = i;
    return e.cohort;
  });
  const K = cohorts.length;
  const cohortIndex = new Map(cohorts.map(c => [c.id, c.i] as [string, number]));

  // --- Variables, the counts join (step 10) and the token universe.
  const tokenId = new Map<string, number>();
  const tokenList: string[] = [];
  const parent: number[] = [];
  const idOf = (t: string): number => {
    let i = tokenId.get(t);
    if (i === undefined) {
      i = tokenList.length;
      tokenId.set(t, i);
      tokenList.push(t);
      parent.push(i);
    }
    return i;
  };
  const find = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  const union = (a: number, b: number): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  // The normalizers are pure and their inputs repeat heavily (a handful of
  // visit, domain and unit spellings, a few hundred codes and names across
  // thousands of variables), so each distinct input is normalized once per build.
  const slotOf = memo(key => visitSlot(key.slice(0, key.indexOf('\u0000')), key.slice(key.indexOf('\u0000') + 1)));
  const domainOf = memo(normalizeDomain);
  const labelKey = memo(normLabel);
  const unitKey = memo(normUnits);
  const identityOf = memo(key => {
    const sep = key.indexOf('\u0000');
    const identity = variableIdentity({concept_code: key.slice(0, sep), omop_id: key.slice(sep + 1)}, match);
    const tokens = uniqueTokens(identity.cc, identity.oi);
    return {identity, tokens, secondary: tokens.filter(t => !identity.head.includes(t))};
  });

  const drafts: Draft[] = [];
  let malformed = 0;
  for (const {cohort, vars} of cohortEntries) {
    const entry = response && cohort.tier === 'profiled' ? response[cohort.id] : undefined;
    // Names equal after lowercasing: the EDA key cannot tell them apart.
    const byLower = new Map<string, string[]>();
    for (const [name] of vars) {
      const key = name.trim().toLowerCase();
      const list = byLower.get(key);
      if (list) list.push(name);
      else byLower.set(key, [name]);
    }
    for (const [name, raw] of vars) {
      const conceptCode = str(raw.concept_code);
      const omopId = str(raw.omop_id);
      const {identity, tokens, secondary} = identityOf(`${conceptCode}\u0000${omopId}`);
      malformed += identity.malformed;
      const key = name.trim().toLowerCase();
      let n: number;
      let empty = 0;
      let missing = 0;
      if (status !== 'ok') n = -3;
      else if (cohort.tier !== 'profiled') n = -2;
      else {
        const c = entry?.variables[key];
        const cn = c ? validCount(c.n) : null;
        if (c && cn !== null) {
          n = cn;
          empty = validCount(c.empty) ?? 0;
          missing = validCount(c.missing) ?? 0;
        } else n = -1;
      }
      const same = byLower.get(key) ?? [];
      const visitName = str(raw.visit_concept_name);
      const visitsRaw = str(raw.visits);
      const domainRaw = str(raw.omop_domain);
      const v: LoomVariable = {
        i: drafts.length,
        cohort: cohort.i,
        name,
        label: str(raw.var_label),
        type: typeClass(raw),
        domain: domainOf(domainRaw),
        domainRaw,
        slot: slotOf(`${visitName}\u0000${visitsRaw}`),
        slotRaw: visitName || visitsRaw,
        units: str(raw.units),
        conceptName: str(raw.concept_name),
        conceptCode,
        omopId,
        // Shared between variables with the same identifiers: read-only.
        tokens,
        head: identity.head,
        home: -1,
        guestRows: [],
        n,
        empty,
        missing,
        caseCollision: same.length > 1 ? (same.find(other => other !== name) ?? null) : null,
        declaredCount: declaredCountOf(raw),
        eligibleRank: 1
      };
      drafts.push({v, secondary});
      for (const t of tokens) idOf(t);
      // Step 3: union only code <-> ID pairs, never two tokens of one list.
      for (const [c, o] of identity.pairs) union(idOf(c), idOf(o));
    }
  }
  const variables = drafts.map(d => d.v);

  // --- Rows = connected components (step 4), with a canonical key.
  const rootRow = new Int32Array(tokenList.length).fill(-1);
  const componentTokens = new Map<number, string[]>();
  for (let t = 0; t < tokenList.length; t++) {
    const r = find(t);
    const list = componentTokens.get(r);
    if (list) list.push(tokenList[t]);
    else componentTokens.set(r, [tokenList[t]]);
  }
  const components = Array.from(componentTokens.entries()).map(([root, tokens]) => {
    tokens.sort(compareTokens);
    // compareTokens puts OMOP IDs first, smallest numeric ID first; else the smallest code.
    return {root, tokens, key: tokens[0]};
  });
  components.sort((a, b) => compareTokens(a.key, b.key));
  const R = components.length;
  const tokenToRow = new Map<string, number>();
  components.forEach((c, r) => {
    rootRow[c.root] = r;
    for (const t of c.tokens) tokenToRow.set(t, r);
  });

  // --- Membership (step 5): home = the heads' component; guest = any other
  // component holding one of its secondary tokens.
  const unmapped: number[] = [];
  const cellCount = new Uint32Array(R * K);
  const homeCount = new Uint16Array(R * K);
  for (const d of drafts) {
    const v = d.v;
    if (!v.head.length) {
      unmapped.push(v.i);
      continue;
    }
    v.home = rootRow[find(tokenId.get(v.head[0]) as number)];
    for (const t of d.secondary) {
      const r = rootRow[find(tokenId.get(t) as number)];
      if (r !== v.home && !v.guestRows.includes(r)) v.guestRows.push(r);
    }
    v.guestRows.sort((a, b) => a - b);
    const key = v.home * K + v.cohort;
    cellCount[key]++;
    homeCount[key]++;
    for (const g of v.guestRows) cellCount[g * K + v.cohort]++;
  }
  const cellStart = new Uint32Array(R * K + 1);
  for (let key = 0; key < R * K; key++) cellStart[key + 1] = cellStart[key] + cellCount[key];
  const cellItems = new Uint32Array(cellStart[R * K]);
  const fill = cellStart.slice(0, R * K);
  // Homes first, then guests, so each cell starts with its home members.
  for (const v of variables) if (v.home >= 0) cellItems[fill[v.home * K + v.cohort]++] = v.i;
  for (const v of variables) for (const g of v.guestRows) cellItems[fill[g * K + v.cohort]++] = v.i;
  // Within a cell: home before guest, then slot order, then n desc, then name.
  for (let key = 0; key < R * K; key++) {
    const a = cellStart[key];
    const b = cellStart[key + 1];
    if (b - a < 2) continue;
    const row = Math.floor(key / K);
    const slice = Array.from(cellItems.subarray(a, b)).sort((x, y) => {
      const vx = variables[x];
      const vy = variables[y];
      return (
        (vx.home === row ? 0 : 1) - (vy.home === row ? 0 : 1) ||
        SLOT_RANK[vx.slot] - SLOT_RANK[vy.slot] ||
        vy.n - vx.n ||
        (vx.name < vy.name ? -1 : vx.name > vy.name ? 1 : 0)
      );
    });
    cellItems.set(slice, a);
  }

  // --- Row label, domain and type (step 7: per-cohort votes, each cohort
  // weighted once) and integrity flags (step 6). This loop visits every
  // (row, cohort) pair, so its tallies are scratch structures reused across
  // rows rather than allocated per row.
  const perSlot = new Uint16Array(VISIT_SLOTS.length);
  const cohortDomains = new Uint16Array(DOMAINS.length);
  const cohortTypes = new Uint16Array(TYPE_CLASSES.length);
  const domainVotes = new Uint16Array(DOMAINS.length);
  const typeVotes = new Uint16Array(TYPE_CLASSES.length);
  const cohortLabels = new Map<string, number>();
  const labelVotes = new Map<string, number>();
  const labelMembers = new Map<string, number>();
  const casings = new Map<string, number>(); // `${normalized}\u0000${as written}` -> members
  const units = new Set<string>();
  const hay: string[] = [];
  const masks = new Uint8Array(R * K);
  // Most common original casing of a normalized label; ties alphabetical.
  const display = (norm: string): string => {
    const prefix = `${norm}\u0000`;
    let best = norm;
    let bestN = 0;
    casings.forEach((n, key) => {
      if (!key.startsWith(prefix)) return;
      const text = key.slice(prefix.length);
      if (n > bestN || (n === bestN && text < best)) {
        best = text;
        bestN = n;
      }
    });
    return best;
  };
  const rows: LoomRow[] = components.map((comp, r) => {
    const tokens = comp.tokens;
    const cohortMask = masks.subarray(r * K, (r + 1) * K);
    let nVars = 0;
    let slotMask = 0;
    let hasGuests = false;
    let broad = false;
    let anyConceptName = false;
    let dictCoverage = 0;
    for (let c = 0; c < K; c++) {
      const key = r * K + c;
      const h = homeCount[key];
      if (cellStart[key + 1] - cellStart[key] > h) hasGuests = true;
      if (!h) continue;
      cohortMask[c] = 1;
      dictCoverage++;
      nVars += h;
      if (h >= 12) broad = true;
      perSlot.fill(0);
      for (let p = cellStart[key]; p < cellStart[key] + h; p++) {
        const v = variables[cellItems[p]];
        slotMask |= SLOT_BIT[v.slot];
        if (++perSlot[SLOT_RANK[v.slot]] >= 3) broad = true;
        if (v.conceptName) anyConceptName = true;
      }
    }
    // Labels vote with concept names; only a row without any falls back to
    // variable labels. A row reached only through secondary tokens (no home
    // member) votes with its guests.
    labelVotes.clear();
    labelMembers.clear();
    casings.clear();
    units.clear();
    domainVotes.fill(0);
    typeVotes.fill(0);
    hay.length = 0;
    for (let c = 0; c < K; c++) {
      const key = r * K + c;
      const a = cellStart[key];
      const b = cellStart[key + 1];
      if (a === b) continue;
      // Every member (home and guest) is searchable by name and label.
      for (let p = a; p < b; p++) hay.push(variables[cellItems[p]].name, variables[cellItems[p]].label);
      const end = nVars ? a + homeCount[key] : b;
      if (end === a) continue;
      cohortLabels.clear();
      cohortDomains.fill(0);
      cohortTypes.fill(0);
      for (let p = a; p < end; p++) {
        const v = variables[cellItems[p]];
        const text = anyConceptName ? v.conceptName : v.label;
        const norm = text ? labelKey(text) : '';
        if (norm) {
          bump(cohortLabels, norm);
          bump(labelMembers, norm);
          bump(casings, `${norm}\u0000${text}`);
        }
        cohortDomains[DOMAIN_RANK[v.domain]]++;
        cohortTypes[TYPE_RANK[v.type]]++;
        const u = unitKey(v.units);
        if (u) units.add(u);
      }
      const label = argmax(cohortLabels, byLengthThenAlpha);
      if (label !== undefined) bump(labelVotes, label);
      // Ties go to the earlier domain band / type class.
      domainVotes[argmaxIndex(cohortDomains)]++;
      typeVotes[argmaxIndex(cohortTypes)]++;
    }

    const winner = argmax(
      labelVotes,
      (a, b) => (labelMembers.get(b) ?? 0) - (labelMembers.get(a) ?? 0) || byLengthThenAlpha(a, b)
    );
    const label = winner !== undefined ? display(winner) : tokenLabel(comp.key);
    const altLabels = Array.from(labelMembers.keys())
      .filter(k => k !== winner)
      .sort((a, b) => (labelMembers.get(b) ?? 0) - (labelMembers.get(a) ?? 0) || (a < b ? -1 : a > b ? 1 : 0))
      .map(display);
    const domain = DOMAINS[argmaxIndex(domainVotes)];
    const mixedDomain = DOMAINS.filter((d, i) => d !== domain && domainVotes[i] > 0).sort(
      (a, b) => domainVotes[DOMAIN_RANK[b]] - domainVotes[DOMAIN_RANK[a]] || DOMAIN_RANK[a] - DOMAIN_RANK[b]
    );
    const type = TYPE_CLASSES[argmaxIndex(typeVotes)];

    let ccCount = 0;
    let oiCount = 0;
    for (const t of tokens) {
      if (t.startsWith('oi:')) oiCount++;
      else ccCount++;
    }
    const flags: RowFlags = {
      // Two codes (or two IDs) in one component can only be joined through a
      // code <-> ID pair of some variable: a bridge (step 6).
      bridged: ccCount > 1 || oiCount > 1,
      suspect: ccCount > 6 || oiCount > 6,
      broad,
      unitsDiffer: units.size > 1,
      hasGuests,
      mixedDomain
    };
    const haystack = [label, ...altLabels, ...tokens, ...tokens.map(tokenLabel), ...hay]
      .filter(Boolean)
      .join('\n')
      .toLowerCase();
    return {
      i: r,
      key: comp.key,
      tokens,
      label,
      altLabels,
      domain,
      type,
      flags,
      slotMask,
      cohortMask,
      dictCoverage,
      nVars,
      haystack
    };
  });

  // --- Eligibility (step 9) against the home row, and per-cohort tallies.
  let slotsPresent = 0;
  const slotVarCounts = VISIT_SLOTS.reduce((acc, s) => ({...acc, [s]: 0}), {} as Record<VisitSlot, number>);
  for (const v of variables) {
    const cohort = cohorts[v.cohort];
    if (v.n === -1) cohort.nNotInEda++;
    if (v.home < 0) {
      cohort.nUnmapped++;
      v.eligibleRank = v.type === 'date' ? 2 : 1;
      continue;
    }
    const row = rows[v.home];
    v.eligibleRank = eligibleRankIn(v, row);
    if (row.dictCoverage >= 2) cohort.nMatched++;
    else cohort.nSingle++;
    slotsPresent |= SLOT_BIT[v.slot];
    slotVarCounts[v.slot]++;
  }

  const mapped = variables.length - unmapped.length;
  return {
    cohorts,
    cohortIndex,
    rows,
    variables,
    unmapped,
    tokenToRow,
    K,
    cellStart,
    cellItems,
    cellHomeCount: homeCount,
    slotsPresent,
    slotVarCounts,
    countsStatus: status,
    match,
    stats: {buildMs: now() - t0, variables: variables.length, mapped, malformed}
  };
}

// 0 = type class and domain match the row's (the preferred representatives;
// for a date concept that includes its dates), 1 = any other non-date, 2 = a
// date. Within a visit slot only the best rank present is eligible (step 9).
export function eligibleRankIn(v: LoomVariable, row: LoomRow): 0 | 1 | 2 {
  if (v.type === row.type && v.domain === row.domain) return 0;
  return v.type === 'date' ? 2 : 1;
}

export function cellKey(model: LoomModel, row: number, cohort: number): number {
  return row * model.K + cohort;
}

export function cellMembers(model: LoomModel, row: number, cohort: number): number[] {
  const key = row * model.K + cohort;
  return Array.from(model.cellItems.subarray(model.cellStart[key], model.cellStart[key + 1]));
}

export function homeMembers(model: LoomModel, row: number, cohort: number): number[] {
  const key = row * model.K + cohort;
  const a = model.cellStart[key];
  return Array.from(model.cellItems.subarray(a, a + model.cellHomeCount[key]));
}

export function guestMembers(model: LoomModel, row: number, cohort: number): number[] {
  const key = row * model.K + cohort;
  return Array.from(
    model.cellItems.subarray(model.cellStart[key] + model.cellHomeCount[key], model.cellStart[key + 1])
  );
}

// Per-cohort domain votes of a row, most votes first ("Condition in 3
// cohorts, Observation in 1"): the numbers behind RowFlags.mixedDomain.
export function rowDomainVotes(model: LoomModel, row: number): {domain: Domain; cohorts: number}[] {
  const votes = new Uint16Array(DOMAINS.length);
  const tally = new Uint16Array(DOMAINS.length);
  // The same voters as buildModel: home members, or guests for a row without any.
  const members = model.rows[row].nVars ? homeMembers : cellMembers;
  for (let c = 0; c < model.K; c++) {
    const list = members(model, row, c);
    if (!list.length) continue;
    tally.fill(0);
    for (const i of list) tally[DOMAIN_RANK[model.variables[i].domain]]++;
    votes[argmaxIndex(tally)]++;
  }
  return DOMAINS.map((domain, i) => ({domain, cohorts: votes[i]}))
    .filter(d => d.cohorts > 0)
    .sort((a, b) => b.cohorts - a.cohorts || DOMAIN_RANK[a.domain] - DOMAIN_RANK[b.domain]);
}

export interface BridgePair {
  cc: string | null; // head concept code token
  oi: string | null; // head OMOP ID token
  cohorts: string[]; // cohort ids using this pair, base order
  variables: number; // home members using it
}

// The bridge explainer (spec §5 step 6): every distinct (code <-> ID) head
// pair among the row's home members, with who uses it. On current data (no
// pipe lists) a row is bridged exactly when this lists more than one code or
// more than one ID; legacy parallel lists can add non-head pairs as well.
export function rowBridgePairs(model: LoomModel, row: number): BridgePair[] {
  const pairs = new Map<string, {cc: string | null; oi: string | null; cohorts: Set<number>; variables: number}>();
  for (let c = 0; c < model.K; c++) {
    for (const i of homeMembers(model, row, c)) {
      const head = model.variables[i].head;
      const cc = head.find(t => t.startsWith('cc:')) ?? null;
      const oi = head.find(t => t.startsWith('oi:')) ?? null;
      const key = `${cc ?? ''}\u0000${oi ?? ''}`;
      let p = pairs.get(key);
      if (!p) pairs.set(key, (p = {cc, oi, cohorts: new Set(), variables: 0}));
      p.cohorts.add(c);
      p.variables++;
    }
  }
  return Array.from(pairs.values())
    .map(p => ({
      cc: p.cc,
      oi: p.oi,
      cohorts: Array.from(p.cohorts)
        .sort((a, b) => a - b)
        .map(c => model.cohorts[c].id),
      variables: p.variables
    }))
    .sort((a, b) => b.cohorts.length - a.cohorts.length || b.variables - a.variables);
}

// ============================================================================
// cells
// ============================================================================

// THE single rule that turns a cell's members into what the Loom draws (spec
// §4.3 precedence, §5 step 9 representative, §6 lo / hi). Every surface (the
// canvas, tooltip, FOCUS bar, inspector, table, Pool) reads it from here, so
// they can never disagree about a cell.

export interface ResolvedCell {
  state: CellStateValue;
  flags: number;
  k: number; // home members under the lens (0 for guest-only cells)
  lo: number; // representative n; 0 unless Counted
  hi: number; // min(N, sum of eligible n); 0 unless Counted
  rep: number; // representative variable index, -1 none
  otherSlots: number; // OtherVisit: slot mask where the concept is coded in this cohort
  N: number | null; // the cohort's dataset rows (profiled), else null
}

const CELLS_SLOT_RANK = VISIT_SLOTS.reduce((acc, s) => ({...acc, [s]: slotIndex(s)}), {} as Record<VisitSlot, number>);

const isAll = (lens: VisitLens): boolean => lens.kind === 'all' || (lens.mask & ALL_SLOTS_MASK) === ALL_SLOTS_MASK;

const shorterName = (a: string, b: string): boolean => a.length < b.length || (a.length === b.length && a < b);

// Which members of a cell decide it (cellItems[from..to)):
// - home members at a lens slot;
// - else guests at a lens slot (legacy pipe-list secondary codes stand in
//   only where the cohort has no home member at the lens);
// - else (atLens false) the members coded at other slots: OtherVisit.
interface MemberRange {
  from: number;
  to: number;
  guest: boolean;
  atLens: boolean;
  slots: number; // slots of the range's members (all of them, lens or not)
}

function pickRange(model: LoomModel, key: number, lens: VisitLens, guests: boolean): MemberRange | null {
  const a = model.cellStart[key];
  const h = a + model.cellHomeCount[key];
  const b = model.cellStart[key + 1];
  if (a === b) return null;
  const all = isAll(lens);
  const {variables, cellItems} = model;
  let homeSlots = 0;
  for (let p = a; p < h; p++) homeSlots |= SLOT_BIT[variables[cellItems[p]].slot];
  if (homeSlots && (all || homeSlots & lens.mask))
    return {from: a, to: h, guest: false, atLens: true, slots: homeSlots};
  let guestSlots = 0;
  if (guests) for (let p = h; p < b; p++) guestSlots |= SLOT_BIT[variables[cellItems[p]].slot];
  if (guestSlots && (all || guestSlots & lens.mask))
    return {from: h, to: b, guest: true, atLens: true, slots: guestSlots};
  if (homeSlots) return {from: a, to: h, guest: false, atLens: false, slots: homeSlots};
  if (guestSlots) return {from: h, to: b, guest: true, atLens: false, slots: guestSlots};
  return null;
}

// Walks the members of a range that sit at a lens slot, slot by slot (members
// are stored by slot, so each slot is one run). Within a run only one
// eligibility rank is eligible (step 9): a date, or a differently typed flag
// such as TIME-CHF's BNP1_reached, never represents a slot that has a proper
// measurement. In a profiled cohort with counts the rank is chosen among the
// members that have values first (n > 0, else n = 0, else all), so a member
// missing from the EDA output never hides a counted one (spec §4.3: ⊘ means no
// member was found in the profiled data). Rank then decides only between
// members that have a count: BNP1 (557) still beats BNP1_reached (564).
function forEachAtLens(
  model: LoomModel,
  row: number,
  range: MemberRange,
  lens: VisitLens,
  visit: (i: number, eligible: boolean) => void
): void {
  const {variables, cellItems} = model;
  const rowObj = model.rows[row];
  const all = isAll(lens);
  const rank = (i: number): number => (range.guest ? eligibleRankIn(variables[i], rowObj) : variables[i].eligibleRank);
  const cohort = range.from < range.to ? model.cohorts[variables[cellItems[range.from]].cohort] : null;
  const counted = model.countsStatus === 'ok' && cohort !== null && cohort.tier === 'profiled';
  let p = range.from;
  while (p < range.to) {
    const slot = variables[cellItems[p]].slot;
    let end = p;
    while (end < range.to && variables[cellItems[end]].slot === slot) end++;
    if (all || (lens.mask & SLOT_BIT[slot]) !== 0) {
      let best = 3;
      if (counted) {
        let bestZero = 3;
        for (let q = p; q < end; q++) {
          const i = cellItems[q];
          const n = variables[i].n;
          if (n > 0) best = Math.min(best, rank(i));
          else if (n === 0) bestZero = Math.min(bestZero, rank(i));
        }
        if (best === 3) best = bestZero;
      }
      if (best === 3) for (let q = p; q < end; q++) best = Math.min(best, rank(cellItems[q]));
      for (let q = p; q < end; q++) visit(cellItems[q], rank(cellItems[q]) === best);
    }
    p = end;
  }
}

// The eligibility rank that represents a slot of a cell (home members at that
// slot), or null when the cell has none there. A better-ranked member missing
// from the profiled data, or without values, is passed over (see above).
export function slotEligibleRank(model: LoomModel, row: number, cohort: number, slot: VisitSlot): number | null {
  const lm = lensMembers(model, row, cohort, {kind: 'slots', mask: SLOT_BIT[slot]}, false);
  const j = lm.eligible.indexOf(true);
  return j < 0 ? null : model.variables[lm.members[j]].eligibleRank;
}

function resolveRange(
  model: LoomModel,
  row: number,
  cohort: number,
  range: MemberRange | null,
  lens: VisitLens,
  floorPct: number
): ResolvedCell {
  const c = model.cohorts[cohort];
  const out: ResolvedCell = {
    state: CellState.NotCoded,
    flags: 0,
    k: 0,
    lo: 0,
    hi: 0,
    rep: -1,
    otherSlots: 0,
    N: c.nRows
  };
  if (c.tier === 'none') {
    out.state = CellState.Fog;
    return out;
  }
  if (!range) return out;
  if (!range.atLens) {
    out.state = CellState.OtherVisit;
    out.otherSlots = range.slots;
    if (range.guest) out.flags = CellFlag.GuestOnly;
    return out;
  }
  const {variables} = model;
  const counted = model.countsStatus === 'ok' && c.tier === 'profiled';
  const perSlot = new Uint16Array(VISIT_SLOTS.length);
  let members = 0;
  let maxN = -1;
  let sum = 0;
  let rep = -1;
  let zeroRep = -1;
  let firstByName = -1;
  forEachAtLens(model, row, range, lens, (i, eligible) => {
    const v = variables[i];
    members++;
    perSlot[CELLS_SLOT_RANK[v.slot]]++;
    if (!eligible) return;
    if (firstByName < 0 || v.name < variables[firstByName].name) firstByName = i;
    if (!counted) return;
    if (v.n > 0) {
      sum += v.n;
      if (v.n > maxN || (v.n === maxN && shorterName(v.name, variables[rep].name))) {
        maxN = v.n;
        rep = i;
      }
    } else if (v.n === 0 && (zeroRep < 0 || shorterName(v.name, variables[zeroRep].name))) zeroRep = i;
  });

  out.k = range.guest ? 0 : members;
  let flags = range.guest ? CellFlag.GuestOnly : 0;
  if (members >= 2) flags |= CellFlag.Multi;
  if (!range.guest) {
    let broad = members >= 12;
    for (let s = 0; s < perSlot.length && !broad; s++) if (perSlot[s] >= 3) broad = true;
    if (broad) flags |= CellFlag.Broad;
  }
  // Precedence among the eligible members: n > 0 wins over zero, which wins
  // over "not in the profiled data". Counts pending / unavailable and
  // dictionary-only cohorts never show a count.
  if (model.countsStatus !== 'ok') {
    out.state = CellState.Pending;
    out.rep = firstByName;
  } else if (c.tier !== 'profiled') {
    out.state = CellState.Dictionary;
    out.rep = firstByName;
  } else if (maxN > 0) {
    out.state = CellState.Counted;
    out.rep = rep;
    out.lo = maxN;
    // Never below lo, even if a dataset reports more values than rows.
    out.hi = Math.max(maxN, c.nRows !== null ? Math.min(c.nRows, sum) : sum);
    if (c.nRows !== null && maxN * 100 < floorPct * c.nRows) flags |= CellFlag.BelowFloor;
  } else if (zeroRep >= 0) {
    out.state = CellState.Zero;
    out.rep = zeroRep;
  } else {
    out.state = CellState.NotInEda;
    out.rep = firstByName;
  }
  out.flags = flags;
  return out;
}

export function resolveCell(
  model: LoomModel,
  row: number,
  cohort: number,
  lens: VisitLens,
  floorPct: number,
  guests: boolean
): ResolvedCell {
  const range = model.cohorts[cohort].tier === 'none' ? null : pickRange(model, row * model.K + cohort, lens, guests);
  return resolveRange(model, row, cohort, range, lens, floorPct);
}

export interface LensMembers {
  members: number[]; // variable indices at the lens that decide the cell (slot order, then n desc)
  eligible: boolean[]; // parallel to members: may represent its slot (step 9)
  guest: boolean; // the members are guests (the cell is GuestOnly)
}

// The members behind a resolved cell, for the readouts, the inspector's ★ and
// the handoff. Empty for NotCoded / OtherVisit / Fog cells.
export function lensMembers(
  model: LoomModel,
  row: number,
  cohort: number,
  lens: VisitLens,
  guests: boolean
): LensMembers {
  const out: LensMembers = {members: [], eligible: [], guest: false};
  if (model.cohorts[cohort].tier === 'none') return out;
  const range = pickRange(model, row * model.K + cohort, lens, guests);
  if (!range || !range.atLens) return out;
  out.guest = range.guest;
  forEachAtLens(model, row, range, lens, (i, eligible) => {
    out.members.push(i);
    out.eligible.push(eligible);
  });
  return out;
}

export function resolveLens(
  model: LoomModel,
  lens: VisitLens,
  floorPct: number,
  guests: boolean,
  basket: Record<string, string[]>
): LensCells {
  const R = model.rows.length;
  const K = model.K;
  const n = R * K;
  const state = new Uint8Array(n);
  const flags = new Uint8Array(n);
  const k = new Uint16Array(n);
  const lo = new Uint32Array(n);
  const hi = new Uint32Array(n);
  const rep = new Int32Array(n).fill(-1);
  const otherSlots = new Uint16Array(n);
  const inBasket: (Set<string> | null)[] = model.cohorts.map(c => {
    const names = basket?.[c.id];
    return Array.isArray(names) && names.length ? new Set(names) : null;
  });
  const fog = model.cohorts.map(c => c.tier === 'none');
  const all = isAll(lens);
  const {cellStart, cellItems, variables} = model;
  for (let r = 0; r < R; r++) {
    for (let c = 0; c < K; c++) {
      const key = r * K + c;
      if (fog[c]) {
        state[key] = CellState.Fog;
        continue;
      }
      if (cellStart[key] === cellStart[key + 1]) continue; // NotCoded: the arrays' zeros
      const range = pickRange(model, key, lens, guests);
      const cell = resolveRange(model, r, c, range, lens, floorPct);
      let f = cell.flags;
      const names = inBasket[c];
      if (names && range) {
        // Basket marks describe the members the cell shows (those at the lens,
        // or those at other visits for an OtherVisit dot).
        let seen = 0;
        let hits = 0;
        for (let p = range.from; p < range.to; p++) {
          const v = variables[cellItems[p]];
          if (range.atLens && !all && (lens.mask & SLOT_BIT[v.slot]) === 0) continue;
          seen++;
          if (names.has(v.name)) hits++;
        }
        if (hits) f |= CellFlag.InBasket;
        if (hits && hits === seen) f |= CellFlag.InBasketAll;
      }
      state[key] = cell.state;
      flags[key] = f;
      k[key] = cell.k;
      lo[key] = cell.lo;
      hi[key] = cell.hi;
      rep[key] = cell.rep;
      otherSlots[key] = cell.otherSlots;
    }
  }
  return {
    lensKey: `${lens.kind}:${lens.mask}:${floorPct}:${guests ? 'g' : 'ng'}`,
    R,
    K,
    state,
    flags,
    k,
    lo,
    hi,
    rep,
    otherSlots
  };
}

// Row statistics over the VISIBLE dictionary columns (spec §6). Only counted
// cells above the floor add to measured coverage and the pooled bounds;
// dictionary-only cohorts are tallied apart (the "+?"), never as length.
export function computeRowStats(model: LoomModel, cells: LensCells, visibleMask: Uint8Array): RowStats {
  const R = cells.R;
  const K = cells.K;
  const covDict = new Uint16Array(R);
  const covMeasured = new Uint16Array(R);
  const pooledLo = new Float64Array(R);
  const pooledHi = new Float64Array(R);
  const dictOnly = new Uint16Array(R);
  const dictOnlyDeclared = new Float64Array(R);
  const visible: number[] = [];
  for (let c = 0; c < K; c++) if (visibleMask[c] && model.cohorts[c].tier !== 'none') visible.push(c);
  const dictionaryTier = model.cohorts.map(c => c.tier === 'dictionary');
  for (let r = 0; r < R; r++) {
    const base = r * K;
    for (const c of visible) {
      const key = base + c;
      if (!cells.k[key]) continue; // no home member under the lens (guest-only cells never count)
      covDict[r]++;
      const st = cells.state[key];
      if (st === CellState.Counted) {
        if (cells.flags[key] & CellFlag.BelowFloor) continue;
        covMeasured[r]++;
        pooledLo[r] += cells.lo[key];
        pooledHi[r] += cells.hi[key];
      } else if (st === CellState.Dictionary || (st === CellState.Pending && dictionaryTier[c])) {
        // Counts unknown: tallied for the "+?" text, never added as length.
        dictOnly[r]++;
        dictOnlyDeclared[r] += model.cohorts[c].declared ?? 0;
      }
    }
  }
  return {covDict, covMeasured, pooledLo, pooledHi, dictOnly, dictOnlyDeclared};
}

// ============================================================================
// format
// ============================================================================

// Numbers and sentences of the Loom. The tooltip, the FOCUS bar, the
// inspector header and the table title all print these, so one fact is worded
// one way everywhere. Numbers are text only: cells never print them.

export function fmtInt(n: number): string {
  if (!Number.isFinite(n)) return '–';
  const s = String(Math.round(Math.abs(n))).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return n < 0 ? `-${s}` : s;
}

// Three significant digits: '980', '1.23k', '18.2k', '22k', '212k', '1.2M'.
export function fmtCompact(n: number): string {
  if (!Number.isFinite(n)) return '–';
  const sign = n < 0 ? '-' : '';
  const a = Math.abs(n);
  if (a < 999.5) return sign + String(Math.round(a));
  const units: [number, string][] = [
    [1e3, 'k'],
    [1e6, 'M'],
    [1e9, 'B']
  ];
  for (let u = 0; u < units.length; u++) {
    const v = a / units[u][0];
    const r = v >= 100 ? Math.round(v) : v >= 10 ? Math.round(v * 10) / 10 : Math.round(v * 100) / 100;
    if (r >= 1000 && u < units.length - 1) continue;
    return `${sign}${r}${units[u][1]}`;
  }
  return sign + String(Math.round(a));
}

// One decimal. Never rounds a partial value to 0% or 100%: that would claim
// "none" or "all" when the data says otherwise.
export function fmtPct(frac: number): string {
  if (!Number.isFinite(frac)) return '–';
  const p = frac * 100;
  if (p === 0) return '0%';
  if (p === 100) return '100%';
  if (p > 0 && p < 0.05) return '<0.1%';
  if (p > 99.95 && p < 100) return '>99.9%';
  return `${p.toFixed(1)}%`;
}

// The verbatim caveats of spec §6 (plus the fixed notes that share their voice).
export const CAVEATS: Record<string, string> = {
  rows: 'Counts are values in the profiled dataset. They equal patients only if the dataset has one row per patient.',
  best: 'Best single variable (not a sum): a lower bound on rows with at least one value.',
  pooled:
    "Sum over profiled cohorts of each cohort's best variable. Not a count of unique participants. Dictionary-only and no-dictionary cohorts are not included.",
  pool: 'Patients with all selected concepts can only be counted inside a Data Clean Room. These are the bounds implied by per-variable counts.',
  followUp: 'Completeness is of dataset rows, not of patients still enrolled; attrition and death lower it.',
  edaV1: 'EDA v1: categorical counts are derived as rows − empty − coded missing.',
  blank: 'Not coded with this concept. It may exist uncoded or under another code.',
  fog: 'No dictionary: unknown, not absent.',
  presenceOnly: 'Counts could not be loaded; showing presence only',
  sizeUnknown: 'dataset size unknown: completeness unavailable; switch to Count',
  unanchored: 'not tied to a study timepoint',
  guest: 'Only through a secondary code of a legacy code list: not counted in coverage or pooled numbers.',
  overviewMerged: 'outlined states merged at this density'
};

// Column notes of a qualified tier (spec §5 step 11).
export const TIER_NOTE_LABEL: Record<TierNote, string> = {
  loading: 'loading counts…',
  'eda-unreadable': 'EDA unreadable: an EDA run is recorded, but its output could not be read',
  'counts-unavailable': 'counts unavailable',
  'counts-without-dictionary': 'profiling exists, but no dictionary'
};

const TYPE_WORD: Record<TypeClass, string> = {num: 'numeric', cat: 'categorical', text: 'text', date: 'date'};

const formatLowerFirst = (s: string): string => s.charAt(0).toLowerCase() + s.slice(1);
const slotWords = (v: LoomVariable): string => formatLowerFirst(SLOT_LONG[v.slot]);
const plural = (n: number, one: string, many = `${one}s`): string => `${fmtInt(n)} ${n === 1 ? one : many}`;

export function stateLabel(state: CellStateValue): string {
  switch (state) {
    case CellState.Counted:
      return 'Profiled: values counted';
    case CellState.Zero:
      return 'Profiled: no values';
    case CellState.NotInEda:
      return 'In the dictionary, not in the profiled data';
    case CellState.Dictionary:
      return 'In the dictionary: count unknown';
    case CellState.Pending:
      return 'Counts loading or unavailable';
    case CellState.OtherVisit:
      return 'Coded only at another visit';
    case CellState.Fog:
      return 'No dictionary: unknown, not absent';
    default:
      return 'Not coded with this concept';
  }
}

export interface CellReadout {
  headline: string;
  context: string;
  lines: string[];
  footer: string[];
}

// Why a member does not represent its slot (step 9), in a few words.
function notComparable(model: LoomModel, row: number, v: LoomVariable): string {
  const r = model.rows[row];
  const why: string[] = [];
  if (v.type !== r.type) why.push(TYPE_WORD[v.type]);
  if (v.domain !== r.domain) why.push(DOMAIN_LABEL[v.domain]);
  return `${why.join(', ') || 'other kind'}: not compared`;
}

const MAX_MEMBERS = 5;

export function cellReadout(
  model: LoomModel,
  row: number,
  cohort: number,
  lens: VisitLens,
  metric: Metric,
  floorPct: number,
  guests: boolean
): CellReadout {
  const r = model.rows[row];
  const c = model.cohorts[cohort];
  const cell = resolveCell(model, row, cohort, lens, floorPct, guests);
  const allLens = isAllLens(lens);
  const context = allLens ? `${r.label} · ${c.id}` : `${r.label} · ${c.id} · ${lensLabel(lens)}`;
  const lines: string[] = [];
  const footer: string[] = [];
  const {variables} = model;
  const rep = cell.rep >= 0 ? variables[cell.rep] : null;
  const shown = lensMembers(model, row, cohort, lens, guests);

  switch (cell.state) {
    case CellState.Fog: {
      const size = c.declared !== null ? ` (${fmtInt(c.declared)} declared participants)` : '';
      if (c.note === 'counts-without-dictionary')
        lines.push('Profiling exists, but no dictionary: its variables cannot be matched.');
      footer.push(CAVEATS.fog);
      return {
        headline: `No dictionary uploaded for ${c.id}${size}. Nothing is known about its variables: unknown, not absent.`,
        context: `${r.label} · ${c.id}`,
        lines,
        footer
      };
    }
    case CellState.NotCoded:
      return {
        headline: `Not coded with this concept in ${c.id}'s dictionary. It may exist uncoded or under another code.`,
        context,
        lines,
        footer
      };
    case CellState.OtherVisit: {
      const at = slotsOfMask(cell.otherSlots).map(slotPhrase).join(', ');
      if (cell.flags & CellFlag.GuestOnly) lines.push(CAVEATS.guest);
      return {headline: `Not at ${lensLabel(lens)}; coded at ${at}`, context, lines, footer};
    }
  }

  // Member line: eligible counted members by n, then zeros, then members that
  // do not represent their slot, then those absent from the profiled data.
  const order = shown.members
    .map((i, j) => ({v: variables[i], eligible: shown.eligible[j]}))
    .sort((a, b) => {
      const group = (m: {v: LoomVariable; eligible: boolean}): number =>
        m.v.n > 0 ? (m.eligible ? 0 : 2) : m.v.n === 0 ? (m.eligible ? 1 : 3) : 4;
      return group(a) - group(b) || b.v.n - a.v.n || (a.v.name < b.v.name ? -1 : 1);
    });
  const memberText = (m: {v: LoomVariable; eligible: boolean}): string => {
    const v = m.v;
    if (v.n === -1) return `${v.name} (${slotWords(v)}): not in profiled data`;
    if (v.n < 0) return `${v.name} (${slotWords(v)})`;
    return m.eligible ? `${v.name} ${fmtInt(v.n)}` : `${v.name} ${fmtInt(v.n)} (${notComparable(model, row, v)})`;
  };
  const memberLine = (): string | null => {
    if (order.length < 2) return null;
    const head = order.slice(0, MAX_MEMBERS).map(memberText);
    if (order.length > MAX_MEMBERS) head.push(`+${order.length - MAX_MEMBERS} more`);
    return head.join(' · ');
  };
  const edaTag = c.edaVersion ? ` · EDA ${c.edaVersion}` : '';
  const guestOnly = (cell.flags & CellFlag.GuestOnly) !== 0;

  let headline: string;
  if (cell.state === CellState.Counted && rep) {
    const N = cell.N;
    if (metric === 'n')
      headline = `${fmtInt(cell.lo)} values counted${N ? ` · ${fmtPct(cell.lo / N)} of ${fmtInt(N)} rows` : ''}`;
    else if (N) headline = `${fmtInt(cell.lo)} of ${fmtInt(N)} rows have a value · ${fmtPct(cell.lo / N)}`;
    else headline = `${fmtInt(cell.lo)} rows have a value · ${CAVEATS.sizeUnknown}`;

    const profiledSlots = new Set(
      shown.members.filter((i, j) => shown.eligible[j] && variables[i].n >= 0).map(i => variables[i].slot)
    );
    const k = shown.members.length;
    const via = `${rep.name} (${slotWords(rep)})`;
    if (cell.flags & CellFlag.Broad) lines.push(`Best of ${plural(k, 'variable')}: ${via}`);
    else if (profiledSlots.size >= 2) lines.push(`Best of ${profiledSlots.size} profiled visits: ${via}`);
    else if (k >= 2) lines.push(`Best of ${plural(k, 'variable')}: ${via}`);
    else lines.push(`Via ${via}`);
    const members = memberLine();
    if (members) lines.push(members);
    const tail = [`Profiled${edaTag}`, N ? `${fmtInt(N)} dataset rows` : 'dataset size unknown'];
    if (k >= 2) tail.push(`at most ${fmtInt(cell.hi)} rows have any value`);
    lines.push(tail.join(' · '));
    if (cell.flags & CellFlag.BelowFloor)
      lines.push(`Below the usable floor (${floorPct}%): not counted in coverage or pooled numbers`);
    if (rep.caseCollision)
      lines.push(`${rep.name}: name differs only by case from ${rep.caseCollision}; count may belong to either`);
    if (guestOnly) lines.push(CAVEATS.guest);
    footer.push(CAVEATS.rows);
    if (k >= 2) footer.push(CAVEATS.best);
    if (SLOT_BIT[rep.slot] & FOLLOW_UP_MASK) footer.push(CAVEATS.followUp);
    if (c.edaVersion === 'v1' && rep.type === 'cat') footer.push(CAVEATS.edaV1);
    return {headline, context, lines, footer};
  }

  if (cell.state === CellState.Zero) {
    headline = 'Profiled: no values (every row empty or coded missing)';
    if (rep && rep.empty + rep.missing > 0)
      lines.push(`${rep.name}: ${fmtInt(rep.empty)} empty · ${fmtInt(rep.missing)} coded missing`);
    const members = memberLine();
    if (members) lines.push(members);
    lines.push(`Profiled${edaTag}${cell.N ? ` · ${fmtInt(cell.N)} dataset rows` : ''}`);
    if (guestOnly) lines.push(CAVEATS.guest);
    footer.push(CAVEATS.rows);
    return {headline, context, lines, footer};
  }

  if (cell.state === CellState.NotInEda) {
    headline = 'In the dictionary, but not found in the profiled data (renamed, dropped, or added after profiling)';
    lines.push(memberLine() ?? `${rep ? rep.name : ''} (${rep ? slotWords(rep) : ''})`);
    lines.push(`Profiled${edaTag}${cell.N ? ` · ${fmtInt(cell.N)} dataset rows` : ''}`);
    if (guestOnly) lines.push(CAVEATS.guest);
    return {headline, context, lines, footer};
  }

  // Dictionary only, or counts pending / unavailable: members are listed, no count is claimed.
  const names = order.slice(0, MAX_MEMBERS).map(m => `${m.v.name} (${slotWords(m.v)})`);
  if (order.length > MAX_MEMBERS) names.push(`+${order.length - MAX_MEMBERS} more`);
  if (cell.state === CellState.Pending) {
    headline = model.countsStatus === 'loading' ? 'Loading counts…' : 'Counts could not be loaded';
    if (names.length) lines.push(names.join(' · '));
    if (model.countsStatus !== 'loading') lines.push(CAVEATS.presenceOnly);
  } else {
    headline = 'In the dictionary · count unknown: this cohort has not been profiled';
    if (names.length) lines.push(names.join(' · '));
    if (rep?.declaredCount)
      lines.push(`Declared by the data owner: ${rep.declaredCount} (unverified; not used for colour)`);
    if (c.note === 'eda-unreadable') lines.push(TIER_NOTE_LABEL['eda-unreadable']);
    lines.push(
      `${TIER_LABEL.dictionary}${c.declared !== null ? ` · ~${fmtInt(c.declared)} declared participants` : ''}`
    );
  }
  if (guestOnly) lines.push(CAVEATS.guest);
  return {headline, context, lines, footer};
}

// The FOCUS bar's one-liner (spec §3.1): "NT-proBNP · TIME-CHF · 486 of 622
// (78.1%) at 6 months via BNP6 · 11 variables". Other states print their headline.
export function cellSummary(
  model: LoomModel,
  row: number,
  cohort: number,
  lens: VisitLens,
  metric: Metric,
  floorPct: number,
  guests: boolean
): string {
  const r = model.rows[row];
  const c = model.cohorts[cohort];
  const cell = resolveCell(model, row, cohort, lens, floorPct, guests);
  const total = homeMembers(model, row, cohort).length;
  const vars = total >= 2 ? ` · ${plural(total, 'variable')}` : '';
  if (cell.state === CellState.Counted && cell.rep >= 0) {
    const v = model.variables[cell.rep];
    const value =
      metric === 'n' || !cell.N
        ? `${fmtInt(cell.lo)} values${cell.N ? ` of ${fmtInt(cell.N)} rows (${fmtPct(cell.lo / cell.N)})` : ''}`
        : `${fmtInt(cell.lo)} of ${fmtInt(cell.N)} (${fmtPct(cell.lo / cell.N)})`;
    return `${r.label} · ${c.id} · ${value} at ${slotWords(v)} via ${v.name}${vars}`;
  }
  const {headline} = cellReadout(model, row, cohort, lens, metric, floorPct, guests);
  return `${r.label} · ${c.id} · ${headline}${cell.state === CellState.Fog || cell.state === CellState.NotCoded ? '' : vars}`;
}

export function rowReadout(model: LoomModel, stats: RowStats | null, row: number): {headline: string; lines: string[]} {
  const r = model.rows[row];
  const lines: string[] = [];
  const shownTokens = r.tokens.slice(0, 6).map(tokenLabel);
  if (r.tokens.length > 6) shownTokens.push(`+${r.tokens.length - 6} more`);
  lines.push(shownTokens.join(' · '));
  if (r.flags.mixedDomain.length) {
    const votes = rowDomainVotes(model, row).map(
      (d, i) => `${DOMAIN_LABEL[d.domain]} in ${d.cohorts}${i === 0 ? (d.cohorts === 1 ? ' cohort' : ' cohorts') : ''}`
    );
    lines.push(`Domain: ${votes.join(', ')}`);
  } else lines.push(`Domain: ${DOMAIN_LABEL[r.domain]} · ${TYPE_WORD[r.type]}`);
  const slots = slotsOfMask(r.slotMask);
  lines.push(
    `${plural(r.nVars, 'variable')} in ${plural(r.dictCoverage, 'cohort')}${slots.length > 1 ? ` · at ${slots.length} visit slots` : ''}`
  );
  if (stats) {
    const cov = [
      `Coded in ${plural(stats.covDict[row], 'shown cohort')}`,
      `measured in ${fmtInt(stats.covMeasured[row])}`
    ];
    if (stats.dictOnly[row]) {
      const declared = stats.dictOnlyDeclared[row];
      cov.push(
        `+${plural(stats.dictOnly[row], 'dictionary-only cohort')}${declared ? ` (~${fmtInt(declared)} declared participants)` : ''}: counts unknown`
      );
    }
    lines.push(cov.join(' · '));
    if (stats.pooledLo[row] > 0)
      lines.push(
        `Pooled: ≥ ${fmtCompact(stats.pooledLo[row])} values, at most ${fmtCompact(stats.pooledHi[row])} (sum of each profiled cohort's best variable; not unique participants)`
      );
  }
  const f = r.flags;
  if (f.bridged) {
    const cc = r.tokens.filter(t => t.startsWith('cc:')).length;
    const oi = r.tokens.length - cc;
    lines.push(
      `≡ Bridged: ${plural(cc, 'code')} and ${plural(oi, 'OMOP ID')} joined through code ↔ ID pairs of the dictionaries`
    );
  }
  if (f.suspect) lines.push('⚠ Suspect bridge: more than 6 codes or IDs joined; check the pairs before pooling');
  if (f.broad) lines.push('◇ Broad: a cohort codes it in 3+ variables at one visit, or 12+ in total');
  if (f.unitsDiffer) lines.push('≠u Units differ across cohorts');
  if (f.hasGuests) lines.push('Includes variables reached only through a secondary code (legacy code lists)');
  return {headline: r.label, lines};
}

function sizeLine(c: LoomCohort): string {
  const parts: string[] = [];
  if (c.nRows !== null) parts.push(`${fmtInt(c.nRows)} dataset rows`);
  if (c.declared !== null) parts.push(`${c.nRows !== null ? '' : '~'}${fmtInt(c.declared)} declared participants`);
  return parts.join(' · ') || 'Size unknown';
}

export function cohortReadout(model: LoomModel, cohort: number): {headline: string; lines: string[]} {
  const c = model.cohorts[cohort];
  const lines: string[] = [];
  const tier = [TIER_LABEL[c.tier]];
  if (c.edaVersion && c.tier === 'profiled') tier.push(`EDA ${c.edaVersion}`);
  if (c.note) tier.push(TIER_NOTE_LABEL[c.note]);
  lines.push(tier.join(' · '));
  if (c.institution) lines.push(c.institution);
  lines.push(
    [c.studyTypeRaw || 'Study type unspecified', c.designRaw || 'design unspecified', STATUS_LABEL[c.status]].join(
      ' · '
    )
  );
  lines.push(sizeLine(c));
  if (c.tier === 'none') {
    lines.push('No dictionary uploaded: nothing is known about its variables (unknown, not absent)');
    return {headline: c.id, lines};
  }
  lines.push(
    `${plural(c.nVars, 'variable')} · ${fmtInt(c.nMatched)} matched in ≥ 2 cohorts · ${fmtInt(c.nSingle)} in this cohort only · ${fmtInt(c.nUnmapped)} unmapped`
  );
  if (c.tier === 'profiled' && model.countsStatus === 'ok' && c.nVars > 0) {
    const found = c.nVars - c.nNotInEda;
    const frac = found / c.nVars;
    lines.push(
      `Profiled ${fmtInt(found)} of ${fmtInt(c.nVars)} dictionary variables (${fmtPct(frac)})${
        frac < 0.8 ? ': caution, under 80% of the dictionary was found in the profiled data' : ''
      }`
    );
  }
  return {headline: c.id, lines};
}

// ============================================================================
// urlState
// ============================================================================

// The Loom view lives in the URL (spec §7): every control has a short query
// parameter, defaults are omitted (the default link is simply /loom), list
// elements are joined with ',' and, inside an element, ', ; ~ %' are escaped
// as %2C %3B %7E %25 (the router percent-encodes everything else, e.g. the
// spaces and parentheses of "TheBox (myocardial infarction)"). Decoding never
// throws: a garbled parameter falls back to its default.

// Size filter domain: the rail's log slider runs 10 -> 100k. A handle at
// either end leaves that side open, so a 212k-participant cohort still passes
// at the top end and a cohort of 8 at the bottom end.
export const SIZE_MIN = 10;
export const SIZE_MAX = 100000;

export const BASELINE_LENS: VisitLens = {kind: 'slots', mask: SLOT_BIT.BL};

export function defaultViewState(): LoomViewState {
  return {
    tiers: {profiled: true, dictionary: true, none: true},
    fog: 'fold',
    eda: {v1: true, v2: true},
    families: null,
    designs: null,
    statuses: null,
    size: null,
    sizeUnknown: true,
    hiddenCohorts: [],
    onlyCohorts: [],
    domains: null,
    minCohorts: 2,
    minProfiledOnly: false,
    types: null,
    floor: 0,
    flagFilter: {bridged: false, broad: false, units: false, hideBroad: false},
    hiddenRows: [],
    pins: [],
    setMode: 'all',
    q: '',
    qMode: 'filter',
    guests: true,
    match: 'both',
    pooledMin: 0,
    metric: 'comp',
    visit: ALL_LENS,
    density: 'comfortable',
    rowGroup: 'domain',
    rowSort: 'coverage',
    sortCohort: null,
    colGroup: 'tier',
    colSort: 'size',
    lock: true,
    collapsedBands: [],
    unfolded: [],
    view: 'loom',
    minimap: true,
    transpose: false,
    anchor: null,
    pool: [],
    poolVisit: BASELINE_LENS,
    poolThreshold: 0,
    poolHandoff: 'rep',
    poolIncludeDict: true,
    focusRowToken: null,
    inspectToken: null,
    debug: false
  };
}

export const DEFAULT_VIEW_STATE: LoomViewState = defaultViewState();

// Query parameter of every state field (the page keeps any other query key
// it does not own untouched).
export const URL_PARAMS: Record<keyof LoomViewState, string> = {
  tiers: 'tier',
  fog: 'fog',
  eda: 'eda',
  families: 'st',
  designs: 'des',
  statuses: 'on',
  size: 'sz',
  sizeUnknown: 'szu',
  hiddenCohorts: 'hide',
  onlyCohorts: 'only',
  domains: 'dom',
  minCohorts: 'min',
  minProfiledOnly: 'minp',
  types: 'typ',
  floor: 'floor',
  flagFilter: 'fl',
  hiddenRows: 'hr',
  pins: 'pin',
  setMode: 'set',
  q: 'q',
  qMode: 'qm',
  guests: 'gst',
  match: 'match',
  pooledMin: 'pn',
  metric: 'm',
  visit: 'v',
  density: 'd',
  rowGroup: 'rows',
  rowSort: 'rs',
  sortCohort: 'rsc',
  colGroup: 'cols',
  colSort: 'cs',
  lock: 'lock',
  collapsedBands: 'cb',
  unfolded: 'uf',
  view: 'view',
  minimap: 'mm',
  transpose: 'tr',
  anchor: 'anc',
  pool: 'pool',
  poolVisit: 'pv',
  poolThreshold: 'pt',
  poolHandoff: 'ph',
  poolIncludeDict: 'pd',
  focusRowToken: 'r',
  inspectToken: 'i',
  debug: 'debug'
};

// ---------------------------------------------------------------------------
// Element escaping
// ---------------------------------------------------------------------------

const ESCAPES: Record<string, string> = {',': '%2C', ';': '%3B', '~': '%7E', '%': '%25'};

export const escapeElement = (s: string): string => s.replace(/[,;~%]/g, ch => ESCAPES[ch]);

// One pass, so an escaped '%' ('%25') is never read as the start of another
// escape. %2A / %21 only occur at the start of a pin element (a cohort id that
// itself begins with the '*' or '!' pin prefix).
export const unescapeElement = (s: string): string =>
  s.replace(/%(2C|3B|7E|25|2A|21)/gi, (_, hex: string) => String.fromCharCode(parseInt(hex, 16)));

const joinList = (items: string[]): string => items.map(escapeElement).join(',');

const splitList = (raw: string): string[] =>
  raw === ''
    ? []
    : raw
        .split(',')
        .map(unescapeElement)
        .filter(s => s !== '');

const unique = <T>(items: T[]): T[] => Array.from(new Set(items));

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

type Query = Record<string, string | string[] | undefined>;

const one = (v: string | string[] | undefined): string | undefined => {
  const first = Array.isArray(v) ? v[0] : v;
  return typeof first === 'string' ? first : undefined;
};

const pick = <T extends string>(raw: string | undefined, allowed: readonly T[], fallback: T): T => {
  if (raw === undefined) return fallback;
  const lower = raw.trim().toLowerCase();
  return allowed.find(a => a.toLowerCase() === lower) ?? fallback;
};

const TRUE_WORDS = ['1', 'true', 'yes', 'on'];
const FALSE_WORDS = ['0', 'false', 'no', 'off'];
const bool = (raw: string | undefined, fallback: boolean): boolean => {
  if (raw === undefined) return fallback;
  const lower = raw.trim().toLowerCase();
  if (TRUE_WORDS.includes(lower)) return true;
  if (FALSE_WORDS.includes(lower)) return false;
  return fallback;
};

// '1000', '1k', '1.5k', '2M' -> number; anything else -> null.
export function parseAmount(raw: string | undefined): number | null {
  if (raw === undefined) return null;
  const m = /^\s*(\d+(?:\.\d+)?)\s*([km])?\s*$/i.exec(raw);
  if (!m) return null;
  const scale = !m[2] ? 1 : m[2].toLowerCase() === 'k' ? 1e3 : 1e6;
  const value = Number(m[1]) * scale;
  return Number.isFinite(value) ? value : null;
}

const clampInt = (raw: string | undefined, lo: number, hi: number, fallback: number): number => {
  const n = parseAmount(raw);
  if (n === null) return fallback;
  return Math.min(hi, Math.max(lo, Math.round(n)));
};

const intText = (n: number): string => String(Math.round(n));

// Enum lists: '-' (or an empty value) is the empty selection; a list with no
// valid element at all is garbage and falls back to the default.
function enumList<T extends string>(raw: string | undefined, allowed: readonly T[]): T[] | null | undefined {
  if (raw === undefined) return undefined;
  const trimmed = raw.trim();
  if (trimmed === '' || trimmed === '-') return [];
  const out: T[] = [];
  for (const part of trimmed.split(',')) {
    const lower = part.trim().toLowerCase();
    const hit = allowed.find(a => a.toLowerCase() === lower);
    if (hit && !out.includes(hit)) out.push(hit);
  }
  return out.length ? out : null;
}

const enumListText = (items: string[]): string => (items.length ? items.map(s => s.toLowerCase()).join(',') : '-');

// ---------------------------------------------------------------------------
// Visit lens: all | bl | m6 | bl-m6 (a stretch of the timeline) | fu (any
// follow-up) | un | comma lists of those ('bl,m12', 'end,un')
// ---------------------------------------------------------------------------

const SLOT_CODE: Record<VisitSlot, string> = {
  PRE: 'pre',
  BL: 'bl',
  M1: 'm1',
  M3: 'm3',
  M6: 'm6',
  M9: 'm9',
  M12: 'm12',
  M18: 'm18',
  M24: 'm24',
  M36: 'm36',
  END: 'end',
  UN: 'un'
};

// Hand-typed spellings accepted on the way in (never written).
const SLOT_ALIASES: Record<string, VisitSlot> = {
  baseline: 'BL',
  '1m': 'M1',
  '3m': 'M3',
  '6m': 'M6',
  '9m': 'M9',
  '12m': 'M12',
  '18m': 'M18',
  '24m': 'M24',
  '36m': 'M36',
  unanchored: 'UN'
};

const slotFromCode = (code: string): VisitSlot | null => {
  const c = code.trim().toLowerCase();
  const direct = VISIT_SLOTS.find(s => SLOT_CODE[s] === c);
  return direct ?? SLOT_ALIASES[c] ?? null;
};

export const sameLens = (a: VisitLens, b: VisitLens): boolean =>
  a.kind === b.kind && (a.kind === 'all' || (a.mask & ALL_SLOTS_MASK) === (b.mask & ALL_SLOTS_MASK));

export function encodeLens(lens: VisitLens): string {
  const mask = lens.mask & ALL_SLOTS_MASK;
  // An empty slot set is not a lens anyone can pick; it reads as "all".
  if (lens.kind === 'all' || mask === 0) return 'all';
  if (mask === FOLLOW_UP_MASK) return 'fu';
  const parts: string[] = [];
  let i = 0;
  while (i < VISIT_SLOTS.length) {
    if (!(mask & SLOT_BIT[VISIT_SLOTS[i]])) {
      i++;
      continue;
    }
    let j = i;
    while (j + 1 < VISIT_SLOTS.length && mask & SLOT_BIT[VISIT_SLOTS[j + 1]]) j++;
    parts.push(j === i ? SLOT_CODE[VISIT_SLOTS[i]] : `${SLOT_CODE[VISIT_SLOTS[i]]}-${SLOT_CODE[VISIT_SLOTS[j]]}`);
    i = j + 1;
  }
  return parts.join(',');
}

export function decodeLens(raw: string | undefined, fallback: VisitLens): VisitLens {
  if (raw === undefined) return fallback;
  const text = raw.trim().toLowerCase();
  if (text === '') return fallback;
  let mask = 0;
  let all = false;
  for (const part of text.split(',')) {
    const token = part.trim();
    if (token === 'all') {
      all = true;
    } else if (token === 'fu') {
      mask |= FOLLOW_UP_MASK;
    } else if (token.includes('-')) {
      const [a, b, extra] = token.split('-');
      const sa = slotFromCode(a);
      const sb = slotFromCode(b ?? '');
      if (!sa || !sb || extra !== undefined) continue;
      let ia = VISIT_SLOTS.indexOf(sa);
      let ib = VISIT_SLOTS.indexOf(sb);
      if (ia > ib) [ia, ib] = [ib, ia];
      for (let k = ia; k <= ib; k++) mask |= SLOT_BIT[VISIT_SLOTS[k]];
    } else {
      const s = slotFromCode(token);
      if (s) mask |= SLOT_BIT[s];
    }
  }
  if (all) return ALL_LENS;
  return mask ? {kind: 'slots', mask} : fallback;
}

// ---------------------------------------------------------------------------
// Pins: 'TIME-CHF,*BIOSTAT-CHF,!Believe' ('*' = with data, '!' = not coded)
// ---------------------------------------------------------------------------

const PIN_PREFIX: Record<PinKind, string> = {coded: '', data: '*', not: '!'};

const encodePin = (p: PinState): string => {
  // A cohort id that itself starts with a pin prefix keeps it escaped.
  const id = escapeElement(p.cohortId).replace(/^[*!]/, ch => (ch === '*' ? '%2A' : '%21'));
  return PIN_PREFIX[p.kind] + id;
};

function decodePins(raw: string | undefined): PinState[] {
  if (raw === undefined) return [];
  const out: PinState[] = [];
  for (const part of raw.split(',')) {
    if (!part) continue;
    const kind: PinKind = part[0] === '*' ? 'data' : part[0] === '!' ? 'not' : 'coded';
    const cohortId = unescapeElement(kind === 'coded' ? part : part.slice(1));
    if (cohortId && !out.some(p => p.cohortId === cohortId)) out.push({cohortId, kind});
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pool: 'oi:3029187;oi:3000963~cc:loinc:2160-0' (';' separates slots, '~'
// joins the rows of a braid)
// ---------------------------------------------------------------------------

const encodePool = (pool: string[]): string => pool.map(slot => slot.split('~').map(escapeElement).join('~')).join(';');

function decodePool(raw: string | undefined): string[] {
  if (raw === undefined) return [];
  const slots = raw
    .split(';')
    .map(slot =>
      slot
        .split('~')
        .map(unescapeElement)
        .filter(k => k !== '')
        .join('~')
    )
    .filter(slot => slot !== '');
  return unique(slots);
}

// ---------------------------------------------------------------------------
// Encode / decode
// ---------------------------------------------------------------------------

const TIER_CODE: Record<Tier, string> = {profiled: 'p', dictionary: 'd', none: 'n'};
const TIERS: Tier[] = ['profiled', 'dictionary', 'none'];
const DENSITY_CODE: Record<Density, string> = {overview: 'o', compact: 'c', comfortable: 'l'};
const DENSITIES: Density[] = ['overview', 'compact', 'comfortable'];
const FLAG_CODES = {bridged: 'b', broad: 'w', units: 'u', hideBroad: 'hw'} as const;
type FlagName = keyof typeof FLAG_CODES;
const FLAG_NAMES = Object.keys(FLAG_CODES) as FlagName[];
const URL_STATE_TYPE_CLASSES: TypeClass[] = ['num', 'cat', 'text', 'date'];
const METRICS: Metric[] = ['comp', 'n', 'pres'];
const FOG_MODES: FogMode[] = ['fold', 'cols', 'hide'];
const SET_MODES: SetMode[] = ['all', 'any', 'only', 'abo'];
const MATCH_MODES: MatchMode[] = ['both', 'code', 'omop'];
const ROW_GROUPS: RowGroup[] = ['domain', 'coverage', 'none'];
const ROW_SORTS: RowSort[] = ['coverage', 'pooled', 'az', 'cohort', 'anchor'];
const COL_GROUPS: ColGroup[] = ['tier', 'type', 'design', 'status', 'none'];
const COL_SORTS: ColSort[] = ['size', 'name', 'coverage', 'anchor'];

export function encodeViewState(state: LoomViewState): Record<string, string> {
  const d = DEFAULT_VIEW_STATE;
  const out: Record<string, string> = {};
  const P = URL_PARAMS;

  if (TIERS.some(t => state.tiers[t] !== d.tiers[t])) {
    const on = TIERS.filter(t => state.tiers[t]).map(t => TIER_CODE[t]);
    out[P.tiers] = on.length ? on.join(',') : '-';
  }
  if (state.fog !== d.fog) out[P.fog] = state.fog;
  if (state.eda.v1 !== d.eda.v1 || state.eda.v2 !== d.eda.v2) {
    const on = (['v1', 'v2'] as const).filter(v => state.eda[v]);
    out[P.eda] = on.length ? on.join(',') : '-';
  }
  if (state.families) out[P.families] = enumListText(state.families);
  if (state.designs) out[P.designs] = enumListText(state.designs);
  if (state.statuses) out[P.statuses] = enumListText(state.statuses);
  if (state.size) out[P.size] = `${intText(state.size[0])}-${intText(state.size[1])}`;
  if (state.sizeUnknown !== d.sizeUnknown) out[P.sizeUnknown] = state.sizeUnknown ? '1' : '0';
  if (state.hiddenCohorts.length) out[P.hiddenCohorts] = joinList(state.hiddenCohorts);
  if (state.onlyCohorts.length) out[P.onlyCohorts] = joinList(state.onlyCohorts);

  if (state.domains) out[P.domains] = enumListText(state.domains);
  if (state.minCohorts !== d.minCohorts) out[P.minCohorts] = intText(state.minCohorts);
  if (state.minProfiledOnly !== d.minProfiledOnly) out[P.minProfiledOnly] = state.minProfiledOnly ? '1' : '0';
  if (state.types) out[P.types] = enumListText(state.types);
  if (state.floor !== d.floor) out[P.floor] = intText(state.floor);
  const flags = FLAG_NAMES.filter(f => state.flagFilter[f]).map(f => FLAG_CODES[f]);
  if (flags.length) out[P.flagFilter] = flags.join(',');
  if (state.hiddenRows.length) out[P.hiddenRows] = joinList(state.hiddenRows);
  if (state.pins.length) out[P.pins] = state.pins.map(encodePin).join(',');
  if (state.setMode !== d.setMode) out[P.setMode] = state.setMode;
  if (state.q !== d.q) out[P.q] = state.q;
  if (state.qMode !== d.qMode) out[P.qMode] = state.qMode === 'highlight' ? 'h' : 'f';
  if (state.guests !== d.guests) out[P.guests] = state.guests ? '1' : '0';
  if (state.match !== d.match) out[P.match] = state.match;
  if (state.pooledMin !== d.pooledMin) out[P.pooledMin] = intText(state.pooledMin);

  if (state.metric !== d.metric) out[P.metric] = state.metric;
  if (!sameLens(state.visit, d.visit)) out[P.visit] = encodeLens(state.visit);
  if (state.density !== d.density) out[P.density] = DENSITY_CODE[state.density];
  if (state.rowGroup !== d.rowGroup) out[P.rowGroup] = state.rowGroup;
  if (state.rowSort !== d.rowSort) out[P.rowSort] = state.rowSort;
  if (state.sortCohort) out[P.sortCohort] = state.sortCohort;
  if (state.colGroup !== d.colGroup) out[P.colGroup] = state.colGroup;
  if (state.colSort !== d.colSort) out[P.colSort] = state.colSort;
  if (state.lock !== d.lock) out[P.lock] = state.lock ? '1' : '0';
  if (state.collapsedBands.length) out[P.collapsedBands] = state.collapsedBands.map(b => b.toLowerCase()).join(',');
  if (state.unfolded.length) out[P.unfolded] = joinList(state.unfolded);
  if (state.view !== d.view) out[P.view] = state.view;
  if (state.minimap !== d.minimap) out[P.minimap] = state.minimap ? '1' : '0';
  if (state.transpose !== d.transpose) out[P.transpose] = state.transpose ? '1' : '0';
  if (state.anchor) out[P.anchor] = state.anchor;

  if (state.pool.length) out[P.pool] = encodePool(state.pool);
  if (!sameLens(state.poolVisit, d.poolVisit)) out[P.poolVisit] = encodeLens(state.poolVisit);
  if (state.poolThreshold !== d.poolThreshold) out[P.poolThreshold] = intText(state.poolThreshold);
  if (state.poolHandoff !== d.poolHandoff) out[P.poolHandoff] = state.poolHandoff;
  if (state.poolIncludeDict !== d.poolIncludeDict) out[P.poolIncludeDict] = state.poolIncludeDict ? '1' : '0';

  if (state.focusRowToken) out[P.focusRowToken] = state.focusRowToken;
  if (state.inspectToken) out[P.inspectToken] = state.inspectToken;
  if (state.debug) out[P.debug] = '1';
  return out;
}

export function decodeViewState(query: Query): LoomViewState {
  const d = defaultViewState();
  const P = URL_PARAMS;
  const get = (key: string): string | undefined => (query ? one(query[key]) : undefined);

  const tiersRaw = get(P.tiers);
  let tiers = d.tiers;
  if (tiersRaw !== undefined) {
    const text = tiersRaw.trim().toLowerCase();
    const codes = text === '-' || text === '' ? [] : text.split(',');
    const on = TIERS.filter(t => codes.includes(TIER_CODE[t]));
    // A list of nothing but garbage keeps the default; '-' is "none".
    if (on.length || codes.length === 0) {
      tiers = {profiled: on.includes('profiled'), dictionary: on.includes('dictionary'), none: on.includes('none')};
    }
  }

  const edaRaw = get(P.eda);
  let eda = d.eda;
  if (edaRaw !== undefined) {
    const on = enumList(edaRaw, ['v1', 'v2'] as const);
    if (on) eda = {v1: on.includes('v1'), v2: on.includes('v2')};
  }

  const sizeRaw = get(P.size);
  let size: [number, number] | null = null;
  if (sizeRaw !== undefined) {
    const [a, b, extra] = sizeRaw.split('-');
    const lo = parseAmount(a);
    const hi = parseAmount(b);
    if (lo !== null && hi !== null && extra === undefined) {
      const clamp = (n: number) => Math.min(SIZE_MAX, Math.max(SIZE_MIN, Math.round(n)));
      size = lo <= hi ? [clamp(lo), clamp(hi)] : [clamp(hi), clamp(lo)];
    }
  }

  const flagsRaw = get(P.flagFilter);
  const flagCodes = flagsRaw ? flagsRaw.toLowerCase().split(',') : [];
  const flagFilter = {
    bridged: flagCodes.includes(FLAG_CODES.bridged),
    broad: flagCodes.includes(FLAG_CODES.broad),
    units: flagCodes.includes(FLAG_CODES.units),
    hideBroad: flagCodes.includes(FLAG_CODES.hideBroad)
  };

  const list = (key: string): string[] => {
    const raw = get(key);
    return raw === undefined ? [] : unique(splitList(raw));
  };
  const nullable = (key: string): string | null => {
    const raw = get(key);
    return raw && raw.trim() ? raw : null;
  };
  const orDefault = <T>(v: T[] | null | undefined, fallback: T[] | null): T[] | null => v ?? fallback;

  const qm = get(P.qMode);
  const densityRaw = get(P.density)?.trim().toLowerCase();
  return {
    tiers,
    fog: pick(get(P.fog), FOG_MODES, d.fog),
    eda,
    families: orDefault(enumList<StudyFamily>(get(P.families), STUDY_FAMILIES), d.families),
    designs: orDefault(enumList<DesignFamily>(get(P.designs), DESIGN_FAMILIES), d.designs),
    statuses: orDefault(enumList<CohortStatus>(get(P.statuses), COHORT_STATUSES), d.statuses),
    size,
    sizeUnknown: bool(get(P.sizeUnknown), d.sizeUnknown),
    hiddenCohorts: list(P.hiddenCohorts),
    onlyCohorts: list(P.onlyCohorts),
    domains: orDefault(enumList<Domain>(get(P.domains), DOMAINS), d.domains),
    minCohorts: clampInt(get(P.minCohorts), 1, 999, d.minCohorts),
    minProfiledOnly: bool(get(P.minProfiledOnly), d.minProfiledOnly),
    types: orDefault(enumList<TypeClass>(get(P.types), URL_STATE_TYPE_CLASSES), d.types),
    floor: clampInt(get(P.floor), 0, 100, d.floor),
    flagFilter,
    hiddenRows: list(P.hiddenRows),
    pins: decodePins(get(P.pins)),
    setMode: pick(get(P.setMode), SET_MODES, d.setMode),
    q: get(P.q) ?? d.q,
    qMode: qm === 'h' || qm === 'highlight' ? 'highlight' : 'filter',
    guests: bool(get(P.guests), d.guests),
    match: pick(get(P.match), MATCH_MODES, d.match),
    pooledMin: clampInt(get(P.pooledMin), 0, 1e9, d.pooledMin),
    metric: pick(get(P.metric), METRICS, d.metric),
    visit: decodeLens(get(P.visit), d.visit),
    density: DENSITIES.find(x => DENSITY_CODE[x] === densityRaw || x === densityRaw) ?? d.density,
    rowGroup: pick(get(P.rowGroup), ROW_GROUPS, d.rowGroup),
    rowSort: pick(get(P.rowSort), ROW_SORTS, d.rowSort),
    sortCohort: nullable(P.sortCohort),
    colGroup: pick(get(P.colGroup), COL_GROUPS, d.colGroup),
    colSort: pick(get(P.colSort), COL_SORTS, d.colSort),
    lock: bool(get(P.lock), d.lock),
    collapsedBands: enumList<Domain>(get(P.collapsedBands), DOMAINS) ?? [],
    unfolded: list(P.unfolded),
    view: get(P.view)?.trim().toLowerCase() === 'table' ? 'table' : 'loom',
    minimap: bool(get(P.minimap), d.minimap),
    transpose: bool(get(P.transpose), d.transpose),
    anchor: nullable(P.anchor),
    pool: decodePool(get(P.pool)),
    poolVisit: decodeLens(get(P.poolVisit), d.poolVisit),
    poolThreshold: clampInt(get(P.poolThreshold), 0, 100, d.poolThreshold),
    poolHandoff: get(P.poolHandoff)?.trim().toLowerCase() === 'all' ? 'all' : 'rep',
    poolIncludeDict: bool(get(P.poolIncludeDict), d.poolIncludeDict),
    focusRowToken: nullable(P.focusRowToken),
    inspectToken: nullable(P.inspectToken),
    debug: bool(get(P.debug), d.debug)
  };
}

// ---------------------------------------------------------------------------
// View presets (spec §11 Phase 2). Each preset sets the same concept-side
// fields, so picking one after another never leaves a mix of both behind;
// cohort filters, pins, the search and the Pool are left alone.
// ---------------------------------------------------------------------------

export interface ViewPreset {
  id: string;
  label: string;
  description: string;
  patch: Partial<LoomViewState>;
}

const NO_FLAGS = {bridged: false, broad: false, units: false, hideBroad: false};

const conceptView = (patch: Partial<LoomViewState>): Partial<LoomViewState> => ({
  domains: null,
  minCohorts: 2,
  minProfiledOnly: false,
  types: null,
  floor: 0,
  flagFilter: NO_FLAGS,
  setMode: 'all',
  pooledMin: 0,
  visit: ALL_LENS,
  ...patch
});

export const VIEW_PRESETS: ViewPreset[] = [
  {
    id: 'everything',
    label: 'Everything we know',
    description: 'Every concept at every visit, including those coded in a single cohort.',
    patch: conceptView({minCohorts: 1})
  },
  {
    id: 'baseline-labs',
    label: 'Poolable baseline labs',
    description: 'Measurements at baseline with at least 50% of rows filled, in 2 or more profiled cohorts.',
    patch: conceptView({domains: ['Measurement'], visit: BASELINE_LENS, floor: 50, minProfiledOnly: true})
  },
  {
    id: 'medications',
    label: 'Medications',
    description: 'Drug concepts shared by 2 or more cohorts.',
    patch: conceptView({domains: ['Drug']})
  },
  {
    id: 'outcomes',
    label: 'Outcomes & events',
    description: 'Deaths and conditions recorded at the end of study or not tied to a study timepoint.',
    patch: conceptView({domains: ['Death', 'Condition'], visit: {kind: 'slots', mask: SLOT_BIT.END | SLOT_BIT.UN}})
  },
  {
    id: 'gaps',
    label: 'Harmonization gaps',
    description:
      'Concepts joined through different codes (≡ bridged), with the ALL BUT ONE set mode: pin cohorts to list the concepts exactly one of them lacks.',
    patch: conceptView({setMode: 'abo', flagFilter: {...NO_FLAGS, bridged: true}})
  }
];

// The preset whose settings the state currently matches, if any (menu check mark).
export function activePreset(state: LoomViewState): ViewPreset | null {
  const same = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b);
  return (
    VIEW_PRESETS.find(p =>
      (Object.keys(p.patch) as (keyof LoomViewState)[]).every(k =>
        k === 'visit' ? sameLens(state.visit, p.patch.visit as VisitLens) : same(state[k], p.patch[k])
      )
    ) ?? null
  );
}

// ============================================================================
// search
// ============================================================================

// The Loom search language (spec §7.4). Free terms are AND-ed, 'a | b' is OR,
// "a phrase" keeps its spaces, -term excludes, and key:value tokens restrict
// by structure:
//   d: / domain:   measurement, drug, visit & period ... (comma = any of)
//   in: / notin:   coded (not coded) in a cohort, under the visit lens
//   code:          concept code; '*' is a wildcard (code:loinc:*, code:2160*)
//   omop:          OMOP concept ID (omop:3000963, omop:3000*)
//   v: / visit:    coded at a visit (bl, 6m, m12, end, un, fu, bl-m6)
//   t: / type:     num, cat, text, date (any member of the concept)
//   is:            bridged, broad, units, suspect, pulled
//   unit:          a member's units contain the value
//   cov:>=k        coded in >= k shown cohorts (>, >=, <, <=, =; bare = >=)
//   n:>=k          pooled lower bound >= k (1k, 10k accepted)
// Every token has an equivalent control, so the syntax is never required.

type KeyName = 'domain' | 'in' | 'notin' | 'code' | 'omop' | 'visit' | 'type' | 'is' | 'unit' | 'cov' | 'n';

const KEYS: Record<string, KeyName> = {
  d: 'domain',
  domain: 'domain',
  in: 'in',
  notin: 'notin',
  code: 'code',
  omop: 'omop',
  v: 'visit',
  visit: 'visit',
  t: 'type',
  type: 'type',
  is: 'is',
  unit: 'unit',
  cov: 'cov',
  n: 'n'
};

// ---------------------------------------------------------------------------
// Tokenizer (positions kept for the chips drawn inside the input)
// ---------------------------------------------------------------------------

interface RawToken {
  start: number; // includes a leading '-'
  end: number;
  or: boolean; // the '|' operator
  negated: boolean;
  key: KeyName | null;
  keyText: string; // the key as typed ('d', 'domain')
  value: string; // unquoted
  quoted: boolean;
  text: string; // as typed, without the leading '-'
}

const isSpace = (ch: string): boolean => /\s/.test(ch);

function tokenize(raw: string): RawToken[] {
  const out: RawToken[] = [];
  let i = 0;
  const n = raw.length;
  while (i < n) {
    const ch = raw[i];
    if (isSpace(ch)) {
      i++;
      continue;
    }
    if (ch === '|') {
      out.push({
        start: i,
        end: i + 1,
        or: true,
        negated: false,
        key: null,
        keyText: '',
        value: '',
        quoted: false,
        text: '|'
      });
      i++;
      continue;
    }
    const start = i;
    let negated = false;
    if (ch === '-') {
      // A lone '-' (or '-|') is not a negation of anything.
      if (i + 1 >= n || isSpace(raw[i + 1]) || raw[i + 1] === '|') {
        i++;
        continue;
      }
      negated = true;
      i++;
    }
    const bodyStart = i;
    const readWord = () => {
      let inQuote = false;
      while (i < n) {
        const c = raw[i];
        if (c === '"') inQuote = !inQuote;
        else if (!inQuote && (isSpace(c) || c === '|')) break;
        i++;
      }
    };
    readWord();
    // Codes are written with a space after the vocabulary in some
    // dictionaries ("loinc: 2276-4"): a pasted "code:loinc: 2276-4" stays one
    // token.
    if (/^code:[^:\s"]+:$/i.test(raw.slice(bodyStart, i))) {
      let j = i;
      while (j < n && isSpace(raw[j])) j++;
      if (j > i && j < n && raw[j] !== '|' && raw[j] !== '-' && raw[j] !== '"' && !/^[A-Za-z]+:/.test(raw.slice(j))) {
        i = j;
        readWord();
      }
    }
    const body = raw.slice(bodyStart, i);
    const m = /^([A-Za-z]+):([\s\S]*)$/.exec(body);
    const key = m ? (KEYS[m[1].toLowerCase()] ?? null) : null;
    const valueRaw = key && m ? m[2] : body;
    out.push({
      start,
      end: i,
      or: false,
      negated,
      key,
      keyText: key && m ? m[1] : '',
      value: valueRaw.replace(/"/g, '').trim(),
      quoted: valueRaw.includes('"'),
      text: body
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Compiled query: groups of atoms (AND of ORs); every atom may be negated.
// ---------------------------------------------------------------------------

type Cmp = '>=' | '>' | '<=' | '<' | '=';

type Atom =
  | {kind: 'text'; term: string}
  | {kind: 'domain'; set: Set<Domain>}
  | {kind: 'in'; cohorts: number[]}
  | {kind: 'notin'; cohorts: number[]}
  | {kind: 'token'; prefix: 'cc:' | 'oi:'; exact: string | null; re: RegExp | null}
  | {kind: 'visit'; mask: number}
  | {kind: 'type'; bits: number}
  | {kind: 'is'; flags: IsFlag[]}
  | {kind: 'unit'; term: string}
  | {kind: 'cov' | 'n'; cmp: Cmp; value: number}
  | {kind: 'never'}; // a well-formed token nothing can satisfy (in:<unknown>, is:braided)

interface CompiledAtom {
  atom: Atom;
  negated: boolean;
}

interface Compiled {
  model: LoomModel;
  groups: CompiledAtom[][];
  needsStats: boolean;
}

export const TYPE_BIT: Record<TypeClass, number> = {num: 1, cat: 2, text: 4, date: 8};

const TYPE_WORDS: Record<string, TypeClass> = {
  num: 'num',
  numeric: 'num',
  number: 'num',
  int: 'num',
  integer: 'num',
  float: 'num',
  cat: 'cat',
  categorical: 'cat',
  category: 'cat',
  text: 'text',
  string: 'text',
  str: 'text',
  date: 'date',
  datetime: 'date',
  time: 'date'
};

const IS_FLAGS = ['bridged', 'broad', 'units', 'suspect', 'pulled'] as const;
type IsFlag = (typeof IS_FLAGS)[number];
// The flags SearchQuery.is reports (suspect is searchable, not summarized).
const isSummaryFlag = (f: string): f is SearchQuery['is'][number] =>
  f === 'bridged' || f === 'broad' || f === 'units' || f === 'pulled';

const escapeRe = (s: string): string => s.replace(/[.+?^${}()|[\]\\]/g, '\\$&');

// Commas list alternatives ("d:drug,condition") unless the value was quoted.
const alternatives = (value: string, quoted: boolean): string[] =>
  (quoted ? [value] : value.split(',')).map(v => v.trim()).filter(Boolean);

// in: / notin: resolve to cohorts with a dictionary: an exact id (any case)
// wins, else every id starting with the value, else every id containing it.
export function resolveCohorts(model: LoomModel, value: string): number[] {
  const v = value.trim().toLowerCase();
  if (!v) return [];
  const known = model.cohorts.filter(c => c.tier !== 'none');
  const exact = known.filter(c => c.id.toLowerCase() === v);
  if (exact.length) return exact.map(c => c.i);
  const prefix = known.filter(c => c.id.toLowerCase().startsWith(v));
  if (prefix.length) return prefix.map(c => c.i);
  return known.filter(c => c.id.toLowerCase().includes(v)).map(c => c.i);
}

function resolveDomains(values: string[]): Set<Domain> {
  const set = new Set<Domain>();
  for (const value of values) {
    const v = value.toLowerCase();
    const exact = DOMAINS.filter(d => d.toLowerCase() === v || DOMAIN_LABEL[d].toLowerCase() === v);
    const hits = exact.length
      ? exact
      : DOMAINS.filter(d => d.toLowerCase().startsWith(v) || DOMAIN_LABEL[d].toLowerCase().startsWith(v));
    hits.forEach(d => set.add(d));
  }
  return set;
}

function visitMask(values: string[]): number {
  let mask = 0;
  const none: {kind: 'slots'; mask: number} = {kind: 'slots', mask: 0};
  for (const value of values) {
    const v = value.toLowerCase().replace(/^36\+$/, 'm36');
    if (v === 'follow-up' || v === 'followup') mask |= FOLLOW_UP_MASK;
    else mask |= decodeLens(v, none).mask & ALL_SLOTS_MASK;
  }
  return mask;
}

// code: / omop: values are normalized like the dictionaries' identifiers, so
// 'LOINC: 718-7' finds 'cc:loinc:718-7' and 'snomedct:*' reads as 'snomed:*'.
// Without a vocabulary, a code matches the code part of any vocabulary
// ('code:718-7'). Patterns test the token without its 'cc:' / 'oi:' prefix.
function tokenAtom(prefix: 'cc:' | 'oi:', value: string): Atom {
  const v = value.toLowerCase().replace(/\s+/g, '');
  const glob = (pattern: string, anyVocabulary: boolean): Atom => {
    const body = pattern.split('*').map(escapeRe).join('.*');
    return {kind: 'token', prefix, exact: null, re: new RegExp(anyVocabulary ? `^(?:[^:]*:)?${body}$` : `^${body}$`)};
  };
  if (prefix === 'oi:') {
    if (v.includes('*')) return glob(v, false);
    const tok = oiToken(v);
    return tok ? {kind: 'token', prefix, exact: tok, re: null} : {kind: 'never'};
  }
  const code = ccToken(v)?.slice(3);
  if (!code) return {kind: 'never'};
  if (code.includes('*') || !code.includes(':')) return glob(code, !code.includes(':'));
  return {kind: 'token', prefix, exact: `cc:${code}`, re: null};
}

function compareAtom(kind: 'cov' | 'n', value: string): Atom {
  const m = /^(>=|<=|>|<|=)?\s*(.+)$/.exec(value.replace(/\s+/g, ''));
  const amount = m ? parseAmount(m[2]) : null;
  if (!m || amount === null) return {kind: 'never'};
  return {kind, cmp: (m[1] as Cmp | undefined) ?? '>=', value: amount};
}

function compileToken(model: LoomModel, t: RawToken): Atom | null {
  if (!t.value) return null; // "d:" while typing: not a restriction yet
  if (!t.key) return {kind: 'text', term: t.value.toLowerCase().replace(/\s+/g, ' ')};
  const values = alternatives(t.value, t.quoted);
  switch (t.key) {
    case 'domain': {
      const set = resolveDomains(values);
      return set.size ? {kind: 'domain', set} : {kind: 'never'};
    }
    case 'in':
    case 'notin': {
      const cohorts = Array.from(new Set(values.flatMap(v => resolveCohorts(model, v))));
      // in:<unknown> can match nothing; notin:<unknown> excludes nothing.
      return {kind: t.key, cohorts};
    }
    case 'code':
    case 'omop': {
      const atoms = values.map(v => tokenAtom(t.key === 'code' ? 'cc:' : 'oi:', v));
      // A comma list of codes becomes one regex of alternatives.
      if (atoms.length === 1) return atoms[0];
      const live = atoms.filter((a): a is Extract<Atom, {kind: 'token'}> => a.kind === 'token');
      if (!live.length) return {kind: 'never'};
      const source = live
        .map(a => (a.re ? a.re.source : `^${escapeRe((a.exact ?? '').slice(a.prefix.length))}$`))
        .join('|');
      return {kind: 'token', prefix: live[0].prefix, exact: null, re: new RegExp(source)};
    }
    case 'visit': {
      const mask = visitMask(values);
      return mask ? {kind: 'visit', mask} : {kind: 'never'};
    }
    case 'type': {
      let bits = 0;
      for (const v of values) {
        const tc = TYPE_WORDS[v.toLowerCase()];
        if (tc) bits |= TYPE_BIT[tc];
      }
      return bits ? {kind: 'type', bits} : {kind: 'never'};
    }
    case 'is': {
      // Braids (is:braided) do not exist in this data: like any unknown flag,
      // it matches nothing.
      const flags = values.map(v => IS_FLAGS.find(f => f === v.toLowerCase()));
      return flags.every(Boolean) ? {kind: 'is', flags: flags as IsFlag[]} : {kind: 'never'};
    }
    case 'unit':
      return {kind: 'unit', term: t.value.toLowerCase().replace(/\s+/g, ' ')};
    case 'cov':
    case 'n':
      return compareAtom(t.key, t.value);
  }
}

// AND of OR-groups: 'a | b c' = (a OR b) AND c.
function groupTokens(tokens: RawToken[]): RawToken[][] {
  const groups: RawToken[][] = [];
  let joinNext = false;
  for (const t of tokens) {
    if (t.or) {
      joinNext = groups.length > 0;
      continue;
    }
    if (joinNext) groups[groups.length - 1].push(t);
    else groups.push([t]);
    joinNext = false;
  }
  return groups;
}

function compile(model: LoomModel, raw: string): Compiled {
  const groups: CompiledAtom[][] = [];
  let needsStats = false;
  for (const g of groupTokens(tokenize(raw))) {
    const atoms: CompiledAtom[] = [];
    for (const t of g) {
      const atom = compileToken(model, t);
      if (!atom) continue;
      if (atom.kind === 'cov' || atom.kind === 'n') needsStats = true;
      atoms.push({atom, negated: t.negated});
    }
    if (atoms.length) groups.push(atoms);
  }
  return {model, groups, needsStats};
}

// parseSearch output -> compiled form (keyed by the query object; recompiled
// if the same query meets a rebuilt model).
const compiledCache = new WeakMap<SearchQuery, Compiled>();

function compiledFor(model: LoomModel, q: SearchQuery): Compiled {
  const hit = compiledCache.get(q);
  if (hit && hit.model === model) return hit;
  const fresh = compile(model, q.raw);
  compiledCache.set(q, fresh);
  return fresh;
}

export function parseSearch(raw: string, model: LoomModel): SearchQuery {
  const text = typeof raw === 'string' ? raw : '';
  const tokens = tokenize(text);
  const q: SearchQuery = {
    raw: text,
    groups: [],
    excludes: [],
    domains: [],
    inCohorts: [],
    notInCohorts: [],
    codes: [],
    omops: [],
    visits: [],
    types: [],
    is: [],
    chips: [],
    empty: true
  };
  for (const t of tokens) {
    if (!t.or && t.value) q.chips.push({text: t.text, negated: t.negated, start: t.start, end: t.end});
  }
  const compiled = compile(model, text);
  q.empty = compiled.groups.length === 0;
  const push = <T>(list: T[], items: T[]) => items.forEach(x => !list.includes(x) && list.push(x));
  // The summary fields describe the plain (single-token, un-negated) clauses;
  // OR-groups that mix kinds live only in the compiled form.
  for (const g of groupTokens(tokens)) {
    const live = g.filter(t => t.value);
    if (!live.length) continue;
    if (live.every(t => !t.key && !t.negated)) {
      q.groups.push(live.map(t => t.value.toLowerCase()));
      continue;
    }
    if (live.length !== 1) continue;
    const t = live[0];
    if (!t.key) {
      if (t.negated) push(q.excludes, [t.value.toLowerCase()]);
      continue;
    }
    const values = alternatives(t.value, t.quoted);
    const ids = (vals: string[]) => vals.flatMap(v => resolveCohorts(model, v)).map(c => model.cohorts[c].id);
    if (t.key === 'in') push(t.negated ? q.notInCohorts : q.inCohorts, ids(values));
    if (t.key === 'notin') push(t.negated ? q.inCohorts : q.notInCohorts, ids(values));
    if (t.negated) continue;
    const lower = values.map(v => v.toLowerCase().replace(/\s+/g, ''));
    switch (t.key) {
      case 'domain':
        push(q.domains, Array.from(resolveDomains(values)));
        break;
      case 'code':
        push(q.codes, lower);
        break;
      case 'omop':
        push(q.omops, lower);
        break;
      case 'visit':
        push(q.visits, slotsOfMask(visitMask(values)));
        break;
      case 'type': {
        const types: TypeClass[] = [];
        lower.forEach(v => TYPE_WORDS[v] && types.push(TYPE_WORDS[v]));
        push(q.types, types);
        break;
      }
      case 'is':
        push(q.is, lower.filter(isSummaryFlag));
        break;
    }
  }
  compiledCache.set(q, compiled);
  return q;
}

// ---------------------------------------------------------------------------
// Row facts the search needs beyond LoomRow (built once per model)
// ---------------------------------------------------------------------------

interface RowFacts {
  typeMask: Uint8Array; // TYPE_BIT of every home member's type
  units: string[]; // distinct normalized units of the home members, '\u0001'-joined
}

const factsCache = new WeakMap<LoomModel, RowFacts>();

const searchNormUnits = (u: string): string => u.trim().toLowerCase().replace(/\s+/g, ' ');

function rowFacts(model: LoomModel): RowFacts {
  const hit = factsCache.get(model);
  if (hit) return hit;
  const R = model.rows.length;
  const typeMask = new Uint8Array(R);
  const unitSets: (Set<string> | null)[] = new Array(R).fill(null);
  for (const v of model.variables) {
    if (v.home < 0) continue;
    typeMask[v.home] |= TYPE_BIT[v.type];
    const u = v.units ? searchNormUnits(v.units) : '';
    if (u) (unitSets[v.home] ??= new Set()).add(u);
  }
  const facts: RowFacts = {typeMask, units: unitSets.map(s => (s ? Array.from(s).join('\u0001') : ''))};
  factsCache.set(model, facts);
  return facts;
}

// Without the page's RowStats, cov: / n: read the stats over every cohort
// with a dictionary (built once per LensCells).
const fallbackStats = new WeakMap<LensCells, RowStats>();

function statsFor(model: LoomModel, cells: LensCells): RowStats {
  const hit = fallbackStats.get(cells);
  if (hit) return hit;
  const mask = new Uint8Array(model.K);
  model.cohorts.forEach(c => (mask[c.i] = c.tier === 'none' ? 0 : 1));
  const stats = computeRowStats(model, cells, mask);
  fallbackStats.set(cells, stats);
  return stats;
}

const compare = (a: number, cmp: Cmp, b: number): boolean =>
  cmp === '>=' ? a >= b : cmp === '>' ? a > b : cmp === '<=' ? a <= b : cmp === '<' ? a < b : a === b;

function atomMatches(
  model: LoomModel,
  cells: LensCells,
  row: number,
  atom: Atom,
  pulledKeys: Set<string>,
  stats: RowStats | null
): boolean {
  const r = model.rows[row];
  switch (atom.kind) {
    case 'text':
      return r.haystack.includes(atom.term);
    case 'domain':
      return atom.set.has(r.domain);
    case 'in':
      return atom.cohorts.some(c => cells.k[row * cells.K + c] > 0);
    case 'notin':
      return atom.cohorts.every(c => cells.k[row * cells.K + c] === 0);
    case 'token':
      return r.tokens.some(
        t =>
          t.startsWith(atom.prefix) &&
          (atom.exact !== null ? t === atom.exact : !!atom.re && atom.re.test(t.slice(atom.prefix.length)))
      );
    case 'visit':
      return (r.slotMask & atom.mask) !== 0;
    case 'type':
      return (rowFacts(model).typeMask[row] & atom.bits) !== 0;
    case 'is':
      return atom.flags.some(f =>
        // A Pool slot may name the row by any of its tokens.
        f === 'pulled' ? r.tokens.some(t => pulledKeys.has(t)) : f === 'units' ? r.flags.unitsDiffer : r.flags[f]
      );
    case 'unit':
      return rowFacts(model).units[row].includes(atom.term);
    case 'cov':
      return !!stats && compare(stats.covDict[row], atom.cmp, atom.value);
    case 'n':
      return !!stats && compare(stats.pooledLo[row], atom.cmp, atom.value);
    case 'never':
      return false;
  }
}

// `stats` (optional, beyond the contract) lets cov: / n: read the coverage of
// the SHOWN columns; filterRows always passes it.
export function rowMatchesSearch(
  model: LoomModel,
  cells: LensCells,
  row: number,
  q: SearchQuery,
  pulledKeys: Set<string>,
  stats?: RowStats
): boolean {
  if (q.empty) return true;
  const compiled = compiledFor(model, q);
  const s = compiled.needsStats ? (stats ?? statsFor(model, cells)) : null;
  for (const group of compiled.groups) {
    let any = false;
    for (const {atom, negated} of group) {
      if (atomMatches(model, cells, row, atom, pulledKeys, s) !== negated) {
        any = true;
        break;
      }
    }
    if (!any) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Chips and autocomplete
// ---------------------------------------------------------------------------

// Clicking a chip negates it (or lifts the negation): the new query text.
export function toggleChipNegation(raw: string, chip: SearchQuery['chips'][number]): string {
  return chip.negated
    ? raw.slice(0, chip.start) + raw.slice(chip.start + 1)
    : `${raw.slice(0, chip.start)}-${raw.slice(chip.start)}`;
}

export interface SearchSuggestion {
  kind: 'key' | 'domain' | 'cohort' | 'concept';
  label: string;
  insert: string; // replaces the whole token under the cursor (applySuggestion)
  detail?: string;
  row?: number; // concept suggestions: Enter pulls this row into the Pool
  tier?: Tier; // cohort suggestions: the tier glyph
}

interface KeyHelp {
  key: string;
  primary: boolean;
  insert: string;
  detail: string;
}

const KEY_HELP: KeyHelp[] = [
  {key: 'd', primary: true, insert: 'd:', detail: 'Domain: measurement, drug, condition…'},
  {key: 'domain', primary: false, insert: 'domain:', detail: 'Domain: measurement, drug, condition…'},
  {key: 'in', primary: true, insert: 'in:', detail: 'Coded in a cohort'},
  {key: 'notin', primary: true, insert: 'notin:', detail: 'Not coded in a cohort'},
  {key: 'code', primary: true, insert: 'code:', detail: 'Concept code; * is a wildcard (code:loinc:*)'},
  {key: 'omop', primary: true, insert: 'omop:', detail: 'OMOP concept ID'},
  {key: 'v', primary: true, insert: 'v:', detail: 'Coded at a visit: bl, 6m, end, un, fu'},
  {key: 'visit', primary: false, insert: 'visit:', detail: 'Coded at a visit: bl, 6m, end, un, fu'},
  {key: 't', primary: true, insert: 't:', detail: 'Data type: num, cat, text, date'},
  {key: 'type', primary: false, insert: 'type:', detail: 'Data type: num, cat, text, date'},
  {key: 'is', primary: true, insert: 'is:', detail: 'bridged, broad, units, suspect, pulled'},
  {key: 'unit', primary: true, insert: 'unit:', detail: 'Units contain'},
  {key: 'cov', primary: true, insert: 'cov:>=', detail: 'Coded in at least k shown cohorts'},
  {key: 'n', primary: true, insert: 'n:>=', detail: 'Pooled values (lower bound) at least n'}
];

const IS_HELP: Record<IsFlag, string> = {
  bridged: '≡ joined through more than one code or ID',
  broad: '◇ many variables in one cell',
  units: '≠u units differ between cohorts',
  suspect: '⚠ suspect bridge: many distinct codes',
  pulled: 'Pulled into the Pool'
};

const TYPE_HELP: [string, string][] = [
  ['num', 'Numeric'],
  ['cat', 'Categorical'],
  ['text', 'Text'],
  ['date', 'Date']
];

const SLOT_CODE_OUT: Record<VisitSlot, string> = {
  PRE: 'pre',
  BL: 'bl',
  M1: '1m',
  M3: '3m',
  M6: '6m',
  M9: '9m',
  M12: '12m',
  M18: '18m',
  M24: '24m',
  M36: 'm36',
  END: 'end',
  UN: 'un'
};

const quoteIfNeeded = (v: string): string => (/[\s"|,]/.test(v) || v.startsWith('-') ? `"${v.replace(/"/g, '')}"` : v);

// The token under the cursor ({start, end} of the whole token, '-' included).
function tokenAtCursor(raw: string, cursor: number): {start: number; end: number} {
  const at = Math.max(0, Math.min(cursor, raw.length));
  const hit = tokenize(raw).find(t => !t.or && t.start <= at && at <= t.end);
  return hit ? {start: hit.start, end: hit.end} : {start: at, end: at};
}

export function applySuggestion(raw: string, cursor: number, s: SearchSuggestion): {value: string; cursor: number} {
  const {start, end} = tokenAtCursor(raw, cursor);
  // Keys stay open for their value; finished tokens get a trailing space.
  const open = /[:=>]$/.test(s.insert);
  const after = raw.slice(end);
  const space = open || after.startsWith(' ') ? '' : ' ';
  const value = raw.slice(0, start) + s.insert + space + after;
  return {value, cursor: start + s.insert.length + (open ? 0 : 1)};
}

interface ModelIndex {
  domainRows: Record<Domain, number>;
  vocabularies: [string, number][]; // cc vocabulary prefix, tokens
  units: [string, number][]; // normalized units, variables
  tokens: string[]; // every token of every row, sorted
}

const indexCache = new WeakMap<LoomModel, ModelIndex>();

function modelIndex(model: LoomModel): ModelIndex {
  const hit = indexCache.get(model);
  if (hit) return hit;
  const domainRows = DOMAINS.reduce((acc, d) => ({...acc, [d]: 0}), {} as Record<Domain, number>);
  model.rows.forEach(r => domainRows[r.domain]++);
  const vocab = new Map<string, number>();
  const tokens = Array.from(model.tokenToRow.keys()).sort();
  for (const t of tokens) {
    if (!t.startsWith('cc:')) continue;
    const colon = t.indexOf(':', 3);
    if (colon > 3) vocab.set(t.slice(3, colon), (vocab.get(t.slice(3, colon)) ?? 0) + 1);
  }
  const units = new Map<string, number>();
  for (const v of model.variables) {
    const u = v.home >= 0 && v.units ? searchNormUnits(v.units) : '';
    if (u) units.set(u, (units.get(u) ?? 0) + 1);
  }
  const byCount = (m: Map<string, number>) =>
    Array.from(m.entries()).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  const index: ModelIndex = {domainRows, vocabularies: byCount(vocab), units: byCount(units), tokens};
  indexCache.set(model, index);
  return index;
}

const TIER_ORDER: Record<Tier, number> = {profiled: 0, dictionary: 1, none: 2};

function cohortSuggestions(
  model: LoomModel,
  key: string,
  prefix: string,
  v: string,
  limit: number
): SearchSuggestion[] {
  const needle = v.toLowerCase();
  return model.cohorts
    .filter(c => c.tier !== 'none' && (!needle || c.id.toLowerCase().includes(needle)))
    .map(c => ({c, starts: !needle || c.id.toLowerCase().startsWith(needle) ? 0 : 1}))
    .sort((a, b) => a.starts - b.starts || TIER_ORDER[a.c.tier] - TIER_ORDER[b.c.tier] || a.c.id.localeCompare(b.c.id))
    .slice(0, limit)
    .map(({c}) => ({
      kind: 'cohort' as const,
      label: c.id,
      insert: `${prefix}${key}:${quoteIfNeeded(c.id)}`,
      detail: `${TIER_LABEL[c.tier]}${c.size !== null ? ` · ${fmtCompact(c.size)}` : ''}`,
      tier: c.tier
    }));
}

function conceptSuggestions(model: LoomModel, text: string, limit: number): SearchSuggestion[] {
  const p = text.toLowerCase().trim();
  if (p.length < 2) return [];
  const scored: {row: number; score: number}[] = [];
  for (const r of model.rows) {
    const label = r.label.toLowerCase();
    let score: number;
    if (label.startsWith(p)) score = 0;
    else if (label.includes(` ${p}`) || label.includes(`(${p}`) || label.includes(`[${p}`)) score = 1;
    else if (label.includes(p)) score = 2;
    else if (r.haystack.includes(p)) score = 3;
    else continue;
    scored.push({row: r.i, score});
  }
  const rows = model.rows;
  scored.sort(
    (a, b) =>
      a.score - b.score ||
      rows[b.row].dictCoverage - rows[a.row].dictCoverage ||
      rows[a.row].label.length - rows[b.row].label.length ||
      a.row - b.row
  );
  return scored.slice(0, limit).map(({row}) => {
    const r = rows[row];
    return {
      kind: 'concept' as const,
      label: r.label,
      insert: quoteIfNeeded(r.label),
      detail: `${DOMAIN_LABEL[r.domain]} · in ${fmtInt(r.dictCoverage)} ${r.dictCoverage === 1 ? 'cohort' : 'cohorts'}`,
      row
    };
  });
}

function valueSuggestions(
  model: LoomModel,
  key: KeyName,
  keyText: string,
  neg: string,
  rawValue: string
): SearchSuggestion[] {
  const v = rawValue.replace(/"/g, '').trim().toLowerCase();
  const head = `${neg}${keyText}:`;
  const idx = modelIndex(model);
  const make = (label: string, value: string, detail?: string): SearchSuggestion => ({
    kind: 'key',
    label,
    insert: head + value,
    detail
  });
  const conceptOf = (token: string): string | undefined => {
    const r = model.tokenToRow.get(token);
    return r === undefined ? undefined : model.rows[r].label;
  };
  switch (key) {
    case 'domain':
      return DOMAINS.filter(
        d => !v || d.toLowerCase().startsWith(v) || DOMAIN_LABEL[d].toLowerCase().startsWith(v)
      ).map(d => ({
        kind: 'domain',
        label: DOMAIN_LABEL[d],
        insert: head + d.toLowerCase(),
        detail: `${fmtInt(idx.domainRows[d])} concepts`
      }));
    case 'in':
    case 'notin':
      return cohortSuggestions(model, keyText, neg, v, 10);
    case 'visit': {
      const present = VISIT_SLOTS.filter(s => model.slotsPresent & SLOT_BIT[s]);
      const out = present
        .filter(s => !v || SLOT_CODE_OUT[s].startsWith(v) || SLOT_LONG[s].toLowerCase().startsWith(v))
        .map(s => make(SLOT_LONG[s], SLOT_CODE_OUT[s], `${fmtInt(model.slotVarCounts[s])} variables`));
      if (!v || 'fu'.startsWith(v) || 'follow-up'.startsWith(v))
        out.push(make('Any follow-up', 'fu', '1 month to end of study'));
      return out;
    }
    case 'type':
      return TYPE_HELP.filter(([code, label]) => !v || code.startsWith(v) || label.toLowerCase().startsWith(v)).map(
        ([code, label]) => make(label, code)
      );
    case 'is':
      return IS_FLAGS.filter(f => !v || f.startsWith(v)).map(f => make(f, f, IS_HELP[f]));
    case 'code': {
      if (!v.includes(':')) {
        return idx.vocabularies
          .filter(([voc]) => !v || voc.startsWith(v))
          .slice(0, 8)
          .map(([voc, n]) => make(`${voc}:*`, `${voc}:*`, `${fmtInt(n)} codes`));
      }
      const needle = `cc:${v.replace(/\*+$/, '')}`;
      return idx.tokens
        .filter(t => t.startsWith(needle))
        .slice(0, 8)
        .map(t => make(t.slice(3), quoteIfNeeded(t.slice(3)), conceptOf(t)));
    }
    case 'omop': {
      const needle = `oi:${v.replace(/\*+$/, '')}`;
      if (needle === 'oi:') return [];
      return idx.tokens
        .filter(t => t.startsWith(needle))
        .slice(0, 8)
        .map(t => make(`OMOP ${t.slice(3)}`, t.slice(3), conceptOf(t)));
    }
    case 'unit':
      return idx.units
        .filter(([u]) => !v || u.includes(v))
        .slice(0, 8)
        .map(([u, n]) => make(u, quoteIfNeeded(u), `${fmtInt(n)} variables`));
    case 'cov':
      return [2, 3, 5, 10].map(k => make(`coded in ≥ ${k} shown cohorts`, `>=${k}`));
    case 'n':
      return ['100', '1k', '10k'].map(text => make(`pooled ≥ ${text}`, `>=${text}`));
  }
}

// Autocomplete for the token under the cursor: keys, domains, cohorts (with
// their tier) and the top 8 matching concepts.
export function searchSuggestions(model: LoomModel, raw: string, cursor: number): SearchSuggestion[] {
  const text = typeof raw === 'string' ? raw : '';
  const {start} = tokenAtCursor(text, cursor);
  const partial = text.slice(start, Math.max(start, Math.min(cursor, text.length)));
  const neg = partial.startsWith('-') ? '-' : '';
  const body = partial.slice(neg.length);
  const m = /^([A-Za-z]+):([\s\S]*)$/.exec(body);
  const key = m ? KEYS[m[1].toLowerCase()] : undefined;
  if (m && key) return valueSuggestions(model, key, m[1], neg, m[2]);

  const lower = body.replace(/"/g, '').toLowerCase();
  const out: SearchSuggestion[] = [];
  const keys = KEY_HELP.filter(k => (lower ? k.key.startsWith(lower) : k.primary));
  keys.forEach(k => out.push({kind: 'key', label: k.insert, insert: neg + k.insert, detail: k.detail}));
  if (lower) {
    DOMAINS.filter(d => d.toLowerCase().startsWith(lower) || DOMAIN_LABEL[d].toLowerCase().startsWith(lower))
      .slice(0, 3)
      .forEach(d =>
        out.push({kind: 'domain', label: DOMAIN_LABEL[d], insert: `${neg}d:${d.toLowerCase()}`, detail: 'Domain'})
      );
    out.push(...cohortSuggestions(model, 'in', neg, lower, 4));
    if (!neg) out.push(...conceptSuggestions(model, body.replace(/"/g, ''), 8));
  }
  return out;
}

// ============================================================================
// filters
// ============================================================================

// Which cohorts are columns and which concepts are rows (spec §6 "Filter
// cost", §7.2, §7.3, §9.4). Every active filter is a chip with its cost: the
// number of rows that come back when that chip alone is removed. Row chips
// are costed in the single filter pass (a row failing exactly one chip is
// that chip's cost); column chips are costed by re-running the pass with the
// cohorts only that chip removes put back.

// ---------------------------------------------------------------------------
// Cohorts (columns)
// ---------------------------------------------------------------------------

const CF = {
  tier: 1, // a profiled / dictionary-only tier toggled off
  fog: 2, // no-dictionary cohorts off (tier toggle or fog=hide)
  eda: 4,
  family: 8,
  design: 16,
  status: 32,
  size: 64,
  hidden: 128,
  only: 256
} as const;

// Bit set of the cohort filters a cohort fails (0 = shown).
export function cohortFailures(model: LoomModel, state: LoomViewState, cohort: number): number {
  const c = model.cohorts[cohort];
  let f = 0;
  if (c.tier === 'none') {
    if (!state.tiers.none || state.fog === 'hide') f |= CF.fog;
  } else if (!state.tiers[c.tier]) f |= CF.tier;
  if (c.tier === 'profiled' && !(state.eda.v1 && state.eda.v2)) {
    // EDA version sits under "Profiled": it only ever removes profiled cohorts.
    const v = c.edaVersion === 'v1' || c.edaVersion === 'v2' ? c.edaVersion : null;
    if (!v || !state.eda[v]) f |= CF.eda;
  }
  if (state.families && !state.families.includes(c.family)) f |= CF.family;
  if (state.designs && !state.designs.includes(c.design)) f |= CF.design;
  if (state.statuses && !state.statuses.includes(c.status)) f |= CF.status;
  if (c.size === null) {
    if (!state.sizeUnknown) f |= CF.size;
  } else if (state.size) {
    const [lo, hi] = state.size;
    // A handle parked at either end of the slider leaves that side open.
    if ((lo > SIZE_MIN && c.size < lo) || (hi < SIZE_MAX && c.size > hi)) f |= CF.size;
  }
  if (state.hiddenCohorts.includes(c.id)) f |= CF.hidden;
  if (state.onlyCohorts.length && !state.onlyCohorts.includes(c.id)) f |= CF.only;
  return f;
}

export function cohortPasses(model: LoomModel, state: LoomViewState, cohort: number): boolean {
  return cohortFailures(model, state, cohort) === 0;
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

const RF = {
  pins: 1,
  notPins: 2,
  domains: 4,
  min: 8,
  types: 16,
  bridged: 32,
  broad: 64,
  units: 128,
  hideBroad: 256,
  hidden: 512,
  search: 1024,
  pooled: 2048
} as const;
type RowFilter = keyof typeof RF;

const TYPE_LABEL: Record<TypeClass, string> = {num: 'Numeric', cat: 'Categorical', text: 'Text', date: 'Date'};

const FLAG_CHIPS = [
  ['bridged', 'Only bridged ≡'],
  ['broad', 'Only broad ◇'],
  ['units', 'Only units differ ≠u'],
  ['hideBroad', 'Broad ◇ hidden']
] as const;

interface Pin {
  c: number;
  data: boolean; // ✓✓ with data
}

// What a filter pass needs besides the visible columns and their stats.
interface Ctx {
  model: LoomModel;
  cells: LensCells;
  state: LoomViewState;
  search: SearchQuery;
  pulledKeys: Set<string>;
  pos: Pin[]; // ✓ / ✓✓ pins (cohorts with a dictionary)
  neg: number[]; // ✕ pins
  setActive: boolean; // the ✓ clause filters (≥ 1 positive pin)
  setBypass: boolean; // ONLY / ALL BUT ONE: single-cohort rows are in scope
  hiddenRows: Uint8Array;
  typeBits: number; // 0 = no type filter
  domains: Set<string> | null;
  minActive: boolean;
  searchActive: boolean;
  searchFilters: boolean; // filter mode (highlight mode only dims)
}

// Per-row outcome of a pass.
interface Pass {
  visible: Uint8Array;
  present: Uint8Array; // ≥ 1 member in a shown dictionary column under the lens
  fail: Uint16Array; // RF bits
  searchMatch: Uint8Array;
  minBypassedBySearch: Uint8Array; // single-cohort row shown only because the search matched it
  minSavedBySet: Uint8Array; // passes "In ≥ k" only through the ONLY / ALL BUT ONE bypass
}

// Type classes of a row's home members in the shown columns (any visit): the
// Data type filter keeps a row if any of them matches, as the rail counts it.
function memberTypes(model: LoomModel, base: number, visibleDict: number[]): number {
  let bits = 0;
  for (const c of visibleDict) {
    const start = model.cellStart[base + c];
    const end = start + model.cellHomeCount[base + c];
    for (let p = start; p < end; p++) bits |= TYPE_BIT[model.variables[model.cellItems[p]].type];
  }
  return bits;
}

const CODED_STATES = new Set<number>([
  CellState.Counted,
  CellState.Zero,
  CellState.NotInEda,
  CellState.Dictionary,
  CellState.Pending
]);

function runPass(ctx: Ctx, mask: Uint8Array, stats: RowStats): Pass {
  const {model, cells, state, pos, neg} = ctx;
  const R = model.rows.length;
  const K = model.K;
  const visibleDict: number[] = [];
  for (let c = 0; c < K; c++) if (mask[c] && model.cohorts[c].tier !== 'none') visibleDict.push(c);
  const pinned = new Set(pos.map(p => p.c));
  const others = visibleDict.filter(c => !pinned.has(c)); // ONLY: nobody else may code the row

  const pass: Pass = {
    visible: new Uint8Array(R),
    present: new Uint8Array(R),
    fail: new Uint16Array(R),
    searchMatch: new Uint8Array(R),
    minBypassedBySearch: new Uint8Array(R),
    minSavedBySet: new Uint8Array(R)
  };
  const {flagFilter} = state;
  const dataMask = CellFlag.BelowFloor | CellFlag.GuestOnly;

  for (let r = 0; r < R; r++) {
    const base = r * K;
    let present = false;
    for (const c of visibleDict) {
      if (CODED_STATES.has(cells.state[base + c])) {
        present = true;
        break;
      }
    }
    if (!present) continue;
    pass.present[r] = 1;
    const row = model.rows[r];
    let fail = 0;

    if (ctx.setActive) {
      let satisfied = 0;
      for (const p of pos) {
        const key = base + p.c;
        const met = p.data
          ? cells.state[key] === CellState.Counted && (cells.flags[key] & dataMask) === 0
          : cells.k[key] > 0;
        if (met) satisfied++;
      }
      let ok: boolean;
      if (state.setMode === 'any') ok = satisfied > 0;
      else if (state.setMode === 'abo') ok = satisfied === pos.length - 1;
      else if (state.setMode === 'only') ok = satisfied === pos.length && others.every(c => cells.k[base + c] === 0);
      else ok = satisfied === pos.length;
      if (!ok) fail |= RF.pins;
    }
    if (neg.length && neg.some(c => cells.k[base + c] > 0)) fail |= RF.notPins;
    if (ctx.domains && !ctx.domains.has(row.domain)) fail |= RF.domains;
    if (ctx.typeBits && !(memberTypes(model, base, visibleDict) & ctx.typeBits)) fail |= RF.types;
    if (flagFilter.bridged && !row.flags.bridged) fail |= RF.bridged;
    if (flagFilter.broad && !row.flags.broad) fail |= RF.broad;
    if (flagFilter.units && !row.flags.unitsDiffer) fail |= RF.units;
    if (flagFilter.hideBroad && row.flags.broad) fail |= RF.hideBroad;
    if (ctx.hiddenRows[r]) fail |= RF.hidden;
    if (state.pooledMin > 0 && stats.pooledLo[r] < state.pooledMin) fail |= RF.pooled;

    const match = ctx.searchActive && rowMatchesSearch(model, cells, r, ctx.search, ctx.pulledKeys, stats);
    if (match) pass.searchMatch[r] = 1;
    if (ctx.searchFilters && !match) fail |= RF.search;

    if (ctx.minActive) {
      const have = state.minProfiledOnly ? stats.covMeasured[r] : stats.covDict[r];
      if (have < state.minCohorts) {
        // Single-cohort concepts come back when the search finds them or a
        // set mode asks about them (spec §5 step 13).
        const single = stats.covDict[r] === 1;
        const bySearch = single && match;
        const bySet = single && ctx.setBypass;
        if (!bySearch && !bySet) fail |= RF.min;
        else if (bySearch) pass.minBypassedBySearch[r] = 1;
        else pass.minSavedBySet[r] = 1;
      }
    }
    pass.fail[r] = fail;
    if (!fail) pass.visible[r] = 1;
  }
  return pass;
}

// ---------------------------------------------------------------------------
// Wording
// ---------------------------------------------------------------------------

const capitalize = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);

const shortList = (names: string[], noun: string): string =>
  names.length <= 2 ? names.join(', ') : `${fmtInt(names.length)} ${noun}`;

function lensPhrase(state: LoomViewState): string {
  const lens = state.visit;
  const slots = lens.mask & ALL_SLOTS_MASK;
  if (lens.kind === 'all' || slots === ALL_SLOTS_MASK || slots === 0) return '';
  if (slots === FOLLOW_UP_MASK) return 'at any follow-up visit';
  if (slots === SLOT_BIT.UN) return 'at unanchored visits';
  return `at ${lensLabel(lens)}`;
}

function positiveClause(model: LoomModel, state: LoomViewState, pos: Pin[], mask: Uint8Array): string {
  const id = (p: Pin) => `${model.cohorts[p.c].id}${mask[p.c] ? '' : ' (hidden)'}`;
  // ✓✓ on a cohort without counts (a stale link) can never be met: say so.
  const data = (p: Pin) => (model.cohorts[p.c].tier === 'profiled' ? ' (with data)' : ' (with data; not profiled)');
  const list = pos.map(p => (p.data ? `${id(p)}${data(p)}` : id(p))).join(', ');
  if (pos.length === 1) {
    if (state.setMode === 'only') return `coded ONLY in ${list}`;
    // ALL BUT ONE of a single cohort: the concept is missing there.
    if (state.setMode === 'abo') return pos[0].data ? `without data in ${id(pos[0])}` : `not coded in ${id(pos[0])}`;
    return `coded in ${list}`;
  }
  switch (state.setMode) {
    case 'any':
      return `coded in ANY of ${list}`;
    case 'only':
      return `coded ONLY in ${list}`;
    case 'abo':
      return `coded in ALL BUT ONE of ${list}`;
    default:
      return `coded in ALL of ${list}`;
  }
}

// ---------------------------------------------------------------------------
// filterRows
// ---------------------------------------------------------------------------

interface ColumnChip {
  id: string;
  label: string;
  clear: LoomAction;
  bit: number;
  tier?: Tier; // tier chips remove one tier only
}

function columnChips(model: LoomModel, state: LoomViewState): ColumnChip[] {
  const out: ColumnChip[] = [];
  const patch = (p: Partial<LoomViewState>): LoomAction => ({type: 'patch', patch: p});
  const count = (tier: Tier) => model.cohorts.filter(c => c.tier === tier).length;
  const plural = (n: number, one: string, many: string) => `${fmtInt(n)} ${n === 1 ? one : many}`;
  if (!state.tiers.profiled) {
    out.push({
      id: 'c:tier:profiled',
      label: `${plural(count('profiled'), 'profiled cohort', 'profiled cohorts')} hidden`,
      clear: patch({tiers: {...state.tiers, profiled: true}}),
      bit: CF.tier,
      tier: 'profiled'
    });
  }
  if (!state.tiers.dictionary) {
    out.push({
      id: 'c:tier:dictionary',
      label: `${plural(count('dictionary'), 'dictionary-only cohort', 'dictionary-only cohorts')} hidden`,
      clear: patch({tiers: {...state.tiers, dictionary: true}}),
      bit: CF.tier,
      tier: 'dictionary'
    });
  }
  const fog = count('none');
  if (fog && (!state.tiers.none || state.fog === 'hide')) {
    out.push({
      id: 'c:fog',
      label: `${plural(fog, 'cohort', 'cohorts')} without a dictionary ${fog === 1 ? 'is' : 'are'} hidden: unknown, not absent`,
      clear: patch({tiers: {...state.tiers, none: true}, fog: state.fog === 'hide' ? 'fold' : state.fog}),
      bit: CF.fog
    });
  }
  if (!(state.eda.v1 && state.eda.v2)) {
    const label = state.eda.v1 ? 'EDA v1 only' : state.eda.v2 ? 'EDA v2 only' : 'No EDA version';
    out.push({id: 'c:eda', label, clear: patch({eda: {v1: true, v2: true}}), bit: CF.eda});
  }
  if (state.families) {
    const names = state.families.map(f => FAMILY_LABEL[f]);
    out.push({
      id: 'c:st',
      label: `Study type: ${names.length ? names.join(', ') : 'none'}`,
      clear: patch({families: null}),
      bit: CF.family
    });
  }
  if (state.designs) {
    const names = state.designs.map(d => DESIGN_LABEL[d]);
    out.push({
      id: 'c:des',
      label: `Design: ${names.length ? names.join(', ') : 'none'}`,
      clear: patch({designs: null}),
      bit: CF.design
    });
  }
  if (state.statuses) {
    const names = state.statuses.map(s => STATUS_LABEL[s]);
    out.push({
      id: 'c:on',
      label: `Status: ${names.length ? names.join(', ') : 'none'}`,
      clear: patch({statuses: null}),
      bit: CF.status
    });
  }
  if (state.size || !state.sizeUnknown) {
    const parts: string[] = [];
    if (state.size) {
      const [lo, hi] = state.size;
      parts.push(
        lo <= SIZE_MIN
          ? `Size ≤ ${fmtCompact(hi)}`
          : hi >= SIZE_MAX
            ? `Size ≥ ${fmtCompact(lo)}`
            : `Size ${fmtCompact(lo)}–${fmtCompact(hi)}`
      );
    }
    if (!state.sizeUnknown) parts.push(state.size ? 'unknown size excluded' : 'Unknown size excluded');
    out.push({id: 'c:sz', label: parts.join(', '), clear: patch({size: null, sizeUnknown: true}), bit: CF.size});
  }
  if (state.hiddenCohorts.length) {
    out.push({
      id: 'c:hide',
      label: `Hidden: ${shortList(state.hiddenCohorts, 'cohorts')}`,
      clear: patch({hiddenCohorts: []}),
      bit: CF.hidden
    });
  }
  if (state.onlyCohorts.length) {
    out.push({
      id: 'c:only',
      label: `Only ${shortList(state.onlyCohorts, 'cohorts')}`,
      clear: patch({onlyCohorts: []}),
      bit: CF.only
    });
  }
  return out;
}

export function filterRows(
  model: LoomModel,
  cells: LensCells,
  stats: RowStats,
  columns: ColumnLayout,
  state: LoomViewState,
  search: SearchQuery,
  pulledKeys: Set<string>
): FilterResult {
  const R = model.rows.length;
  const mask = columns.visibleMask;

  // Pins name cohorts with a dictionary; stale ids and no-dictionary cohorts
  // (nothing to pin) drop out. Pins stay in force on hidden cohorts: they are
  // explicit, and the recipe marks them "(hidden)".
  const seen = new Set<number>();
  const pos: Pin[] = [];
  const neg: number[] = [];
  for (const p of state.pins) {
    const c = model.cohortIndex.get(p.cohortId);
    if (c === undefined || seen.has(c) || model.cohorts[c].tier === 'none') continue;
    seen.add(c);
    if (p.kind === 'not') neg.push(c);
    else pos.push({c, data: p.kind === 'data'});
  }

  const hiddenRows = new Uint8Array(R);
  for (const key of state.hiddenRows) {
    const r = model.tokenToRow.get(key);
    if (r !== undefined) hiddenRows[r] = 1;
  }
  const typeBits = state.types ? state.types.reduce((b, t) => b | TYPE_BIT[t], 0) : 0;
  const setActive = pos.length > 0;
  const ctx: Ctx = {
    model,
    cells,
    state,
    search,
    pulledKeys,
    pos,
    neg,
    setActive,
    setBypass: setActive && (state.setMode === 'only' || state.setMode === 'abo'),
    hiddenRows,
    typeBits,
    domains: state.domains ? new Set<string>(state.domains) : null,
    minActive: state.minCohorts > 1 || state.minProfiledOnly,
    searchActive: !search.empty,
    searchFilters: !search.empty && state.qMode === 'filter'
  };
  const pass = runPass(ctx, mask, stats);

  // --- Row chips and their single-pass costs.
  const chipDefs: {id: RowFilter; label: string; clear: LoomAction}[] = [];
  const patch = (p: Partial<LoomViewState>): LoomAction => ({type: 'patch', patch: p});
  const posClause = setActive ? positiveClause(model, state, pos, mask) : '';
  const negNames = neg.map(c => `${model.cohorts[c].id}${mask[c] ? '' : ' (hidden)'}`);
  const negClause = neg.length ? `not coded in ${negNames.join(', ')}` : '';
  if (setActive) {
    chipDefs.push({
      id: 'pins',
      label: capitalize(posClause),
      clear: patch({pins: state.pins.filter(p => p.kind === 'not')})
    });
  }
  if (neg.length) {
    chipDefs.push({
      id: 'notPins',
      label: capitalize(negClause),
      clear: patch({pins: state.pins.filter(p => p.kind !== 'not')})
    });
  }
  if (state.domains) {
    const names = state.domains.map(d => DOMAIN_LABEL[d]);
    chipDefs.push({
      id: 'domains',
      label: `Domain: ${names.length ? names.join(', ') : 'none'}`,
      clear: patch({domains: null})
    });
  }
  if (ctx.minActive) {
    const k = state.minCohorts;
    const label = state.minProfiledOnly
      ? `In ≥ ${k} profiled ${k === 1 ? 'cohort' : 'cohorts'} with data`
      : `In ≥ ${k} cohorts`;
    chipDefs.push({id: 'min', label, clear: patch({minCohorts: 1, minProfiledOnly: false})});
  }
  if (state.types) {
    const names = state.types.map(t => TYPE_LABEL[t]);
    chipDefs.push({
      id: 'types',
      label: `Type: ${names.length ? names.join(', ') : 'none'}`,
      clear: patch({types: null})
    });
  }
  const fl = state.flagFilter;
  for (const [id, label] of FLAG_CHIPS) {
    if (fl[id]) chipDefs.push({id, label, clear: patch({flagFilter: {...fl, [id]: false}})});
  }
  if (state.pooledMin > 0) {
    chipDefs.push({id: 'pooled', label: `Pooled ≥ ${fmtCompact(state.pooledMin)}`, clear: patch({pooledMin: 0})});
  }
  if (ctx.searchFilters) {
    const q = search.raw.trim();
    chipDefs.push({id: 'search', label: `Search: ${q.length > 40 ? `${q.slice(0, 39)}…` : q}`, clear: patch({q: ''})});
  }
  const hiddenCount = hiddenRows.reduce((n, v) => n + v, 0);
  if (hiddenCount) {
    chipDefs.push({
      id: 'hidden',
      label: `${fmtInt(hiddenCount)} hidden ${hiddenCount === 1 ? 'concept' : 'concepts'}`,
      clear: patch({hiddenRows: []})
    });
  }

  // A row comes back when a chip alone is removed iff that chip is its only
  // failure; for the ✓ clause, also only if "In ≥ k" does not then catch it
  // (its ONLY / ALL BUT ONE bypass leaves with the clause).
  const cost = new Map<number, number>();
  let visibleCount = 0;
  let absent = 0;
  let multiCause = 0;
  let singleCohortRows = 0;
  let singleCohortShownBySearch = 0;
  let matchCount = 0;
  for (let r = 0; r < R; r++) {
    if (pass.visible[r]) {
      visibleCount++;
      if (pass.searchMatch[r]) matchCount++;
      if (pass.minBypassedBySearch[r]) singleCohortShownBySearch++;
      continue;
    }
    if (!pass.present[r]) {
      absent++;
      continue;
    }
    const f = pass.fail[r];
    const sole = (f & (f - 1)) === 0 && !(f === RF.pins && pass.minSavedBySet[r]);
    if (!sole) {
      multiCause++;
      continue;
    }
    cost.set(f, (cost.get(f) ?? 0) + 1);
    if (f === RF.min && stats.covDict[r] === 1) singleCohortRows++;
  }
  const rowChips: FilterChip[] = chipDefs.map(d => ({
    id: d.id,
    label: d.label,
    cost: cost.get(RF[d.id]) ?? 0,
    clear: d.clear
  }));

  // --- Column chips: put back the cohorts only this chip removes and count
  // the rows that the full filter then shows. Each costs one more pass, so it
  // runs on first read of `cost` (callers that only want `visible`, such as
  // the rail's live counts, never pay for it).
  const failures = model.cohorts.map(c => cohortFailures(model, state, c.i));
  const colChips: FilterChip[] = columnChips(model, state).map(chip => {
    let memo: number | null = null;
    const measure = (): number => {
      // No-dictionary cohorts never add a row: their chip costs nothing.
      if (chip.bit === CF.fog) return 0;
      const next = mask.slice();
      let added = 0;
      for (const c of model.cohorts) {
        if (c.tier === 'none' || failures[c.i] !== chip.bit || (chip.tier && c.tier !== chip.tier)) continue;
        next[c.i] = 1;
        added++;
      }
      if (!added) return 0;
      const alt = runPass(ctx, next, computeRowStats(model, cells, next));
      let rows = 0;
      for (let r = 0; r < R; r++) if (alt.visible[r] && !pass.visible[r]) rows++;
      return rows;
    };
    return {
      id: chip.id,
      label: chip.label,
      clear: chip.clear,
      get cost(): number {
        if (memo === null) memo = measure();
        return memo;
      }
    };
  });

  // --- Recipe sentence (spec §7.3).
  const lens = lensPhrase(state);
  const clauses = [posClause, negClause, lens].filter(Boolean);
  const recipe = !clauses.length
    ? ''
    : clauses.length === 1 && lens
      ? `Concepts coded ${lens}`
      : `Concepts ${clauses.join(' · ')}`;

  // --- Not-shown ledger (spec §9.4): hidden rows = absent + Σ sole causes + multi-cause.
  const byFilter = rowChips.map(c => ({id: c.id, label: c.label, rows: c.cost}));
  if (absent) {
    byFilter.push({
      id: 'absent',
      label: !shownDictionaryCohorts(model, mask)
        ? 'No cohort with a dictionary is shown'
        : lens
          ? `Not coded ${lens} in any shown cohort`
          : 'Not coded in any shown cohort',
      rows: absent
    });
  }
  const fog = model.cohorts.filter(c => c.tier === 'none');
  const ledger: NotShownLedger = {
    totalRows: R,
    byFilter,
    multiCause,
    singleCohortRows,
    unmappedVariables: model.unmapped.length,
    malformedIdentifiers: model.stats.malformed,
    fogCohorts: {count: fog.length, declared: fog.reduce((s, c) => s + (c.declared ?? 0), 0)},
    notInEdaVariables: model.cohorts.reduce((s, c) => s + (c.tier === 'profiled' ? c.nNotInEda : 0), 0)
  };

  let highlight: Uint8Array | null = null;
  if (ctx.searchActive && state.qMode === 'highlight') highlight = pass.searchMatch;

  return {
    visible: pass.visible,
    visibleCount,
    highlight,
    matchCount: ctx.searchActive ? matchCount : 0,
    chips: [...rowChips, ...colChips],
    recipe,
    singleCohortShownBySearch,
    ledger
  };
}

// Shown columns with a dictionary: the "In ≥ k" stepper's upper bound.
export function shownDictionaryCohorts(model: LoomModel, visibleMask: Uint8Array): number {
  let n = 0;
  for (let c = 0; c < model.K; c++) if (visibleMask[c] && model.cohorts[c].tier !== 'none') n++;
  return n;
}

// ============================================================================
// order
// ============================================================================

// Row and column order (spec §7.1 Layout, §6 "Anchor overlap"). Order is
// locked by default: filters hide rows without reshuffling the ones that
// remain (law 3); new rows slot in at their sorted place among neighbours.

const collator = new Intl.Collator(undefined, {numeric: true, sensitivity: 'base'});

// A–Z rank of every row label (ties by row index), built once per model, so
// sorts compare integers instead of strings.
const labelRankCache = new WeakMap<LoomModel, Int32Array>();

function labelRank(model: LoomModel): Int32Array {
  const hit = labelRankCache.get(model);
  if (hit) return hit;
  const order = model.rows
    .map(r => r.i)
    .sort((a, b) => collator.compare(model.rows[a].label, model.rows[b].label) || a - b);
  const rank = new Int32Array(model.rows.length);
  order.forEach((row, i) => (rank[row] = i));
  labelRankCache.set(model, rank);
  return rank;
}

// Knowledge rank of one cell for "sort rows by this cohort": measured data
// first (by completeness), then the outlined states, then nothing coded.
function cellRank(cells: LensCells, key: number): number {
  const st = cells.state[key];
  const f = cells.flags[key];
  if (f & CellFlag.GuestOnly) return 6;
  switch (st) {
    case CellState.Counted:
      return f & CellFlag.BelowFloor ? 1 : 0;
    case CellState.Zero:
      return 2;
    case CellState.NotInEda:
      return 3;
    case CellState.Dictionary:
    case CellState.Pending:
      return 4;
    case CellState.OtherVisit:
      return 5;
    default:
      return 7;
  }
}

// A fresh sort of the visible rows (no lock).
export function sortRows(
  model: LoomModel,
  cells: LensCells,
  stats: RowStats,
  state: LoomViewState,
  visible: Uint8Array
): number[] {
  const rank = labelRank(model);
  const rows: number[] = [];
  for (let r = 0; r < visible.length; r++) if (visible[r]) rows.push(r);

  // Coverage (then measured coverage, then pooled L, then A–Z): the default.
  const byCoverage = (a: number, b: number): number =>
    stats.covDict[b] - stats.covDict[a] ||
    stats.covMeasured[b] - stats.covMeasured[a] ||
    stats.pooledLo[b] - stats.pooledLo[a] ||
    rank[a] - rank[b];

  const cohortId = state.rowSort === 'cohort' ? state.sortCohort : state.rowSort === 'anchor' ? state.anchor : null;
  const cohort = cohortId ? model.cohortIndex.get(cohortId) : undefined;

  if (state.rowSort === 'az') return rows.sort((a, b) => rank[a] - rank[b]);
  if (state.rowSort === 'pooled') {
    return rows.sort(
      (a, b) => stats.pooledLo[b] - stats.pooledLo[a] || stats.pooledHi[b] - stats.pooledHi[a] || byCoverage(a, b)
    );
  }
  if (cohort !== undefined && model.cohorts[cohort].tier !== 'none') {
    // By one cohort (or the anchor): its measured cells by completeness (count
    // when the dataset size is unknown), then its outlined cells, then the rest.
    const K = cells.K;
    const N = model.cohorts[cohort].nRows;
    const value = (r: number): number => {
      const lo = cells.lo[r * K + cohort];
      return N ? lo / N : lo;
    };
    return rows.sort(
      (a, b) =>
        cellRank(cells, a * K + cohort) - cellRank(cells, b * K + cohort) || value(b) - value(a) || byCoverage(a, b)
    );
  }
  return rows.sort(byCoverage);
}

// Keeps the relative order of the rows already on screen; rows not in the
// locked order follow the sorted row they come after (a stable merge), so a
// row that reappears lands next to its sorted neighbours.
export function lockMerge(sorted: number[], locked: number[]): number[] {
  const inSorted = new Set(sorted);
  const kept = locked.filter(r => inSorted.has(r));
  const keptSet = new Set(kept);
  const lead: number[] = [];
  const after = new Map<number, number[]>();
  let anchor = -1;
  for (const r of sorted) {
    if (keptSet.has(r)) {
      anchor = r;
      continue;
    }
    if (anchor < 0) lead.push(r);
    else {
      const list = after.get(anchor);
      if (list) list.push(r);
      else after.set(anchor, [r]);
    }
  }
  const out = lead;
  for (const r of kept) {
    out.push(r);
    const list = after.get(r);
    if (list) out.push(...list);
  }
  return out;
}

// The page passes the order it showed last (null to re-sort: after "Re-sort",
// or when the sort / grouping settings change).
export function orderRows(
  model: LoomModel,
  cells: LensCells,
  stats: RowStats,
  state: LoomViewState,
  visible: Uint8Array,
  lockedOrder: number[] | null
): number[] {
  const sorted = sortRows(model, cells, stats, state, visible);
  return state.lock && lockedOrder ? lockMerge(sorted, lockedOrder) : sorted;
}

// ---------------------------------------------------------------------------
// Columns
// ---------------------------------------------------------------------------

// Concepts each cohort codes (home members at any visit), once per model.
const rowsCodedCache = new WeakMap<LoomModel, Uint32Array>();

export function rowsCoded(model: LoomModel): Uint32Array {
  const hit = rowsCodedCache.get(model);
  if (hit) return hit;
  const out = new Uint32Array(model.K);
  for (const row of model.rows) for (let c = 0; c < model.K; c++) out[c] += row.cohortMask[c];
  rowsCodedCache.set(model, out);
  return out;
}

// Anchor overlap: shares = concepts coded in both, J = shares / concepts coded
// in either. By default over every concept row (all visits); pass `rowMask`
// to restrict it to a row set (e.g. the rows shown before set pins).
const jaccardCache = new WeakMap<LoomModel, Map<number, {shares: number; union: number; j: number}>>();

export function jaccard(
  model: LoomModel,
  a: number,
  b: number,
  rowMask?: Uint8Array
): {shares: number; union: number; j: number} {
  const key = Math.min(a, b) * model.K + Math.max(a, b);
  let memo = rowMask ? undefined : jaccardCache.get(model);
  const hit = memo?.get(key);
  if (hit) return hit;
  let shares = 0;
  let union = 0;
  for (const row of model.rows) {
    if (rowMask && !rowMask[row.i]) continue;
    const x = row.cohortMask[a];
    const y = row.cohortMask[b];
    if (x && y) shares++;
    if (x || y) union++;
  }
  const out = {shares, union, j: union ? shares / union : 0};
  if (!rowMask) {
    if (!memo) {
      memo = new Map();
      jaccardCache.set(model, memo);
    }
    memo.set(key, out);
  }
  return out;
}

// Order of the cohorts inside one column group.
export function orderCohorts(model: LoomModel, cohorts: number[], state: LoomViewState): number[] {
  const cs = model.cohorts;
  const bySize = (a: number, b: number): number => {
    const sa = cs[a].size;
    const sb = cs[b].size;
    if (sa !== sb) {
      if (sa === null) return 1; // unknown size last
      if (sb === null) return -1;
      return sb - sa;
    }
    return collator.compare(cs[a].id, cs[b].id);
  };
  const out = cohorts.slice();
  if (state.colSort === 'name') return out.sort((a, b) => collator.compare(cs[a].id, cs[b].id) || a - b);
  if (state.colSort === 'coverage') {
    const coded = rowsCoded(model);
    return out.sort((a, b) => coded[b] - coded[a] || bySize(a, b));
  }
  // An anchor re-sorts the columns by similarity unless another sort was
  // picked on purpose (size is the default), so a bare ?anc= link works.
  const anchored = state.anchor && (state.colSort === 'anchor' || state.colSort === 'size');
  const anchor = anchored && state.anchor ? model.cohortIndex.get(state.anchor) : undefined;
  if (anchor !== undefined) {
    const j = new Float64Array(model.K);
    const shares = new Float64Array(model.K);
    for (const c of out) {
      const o = jaccard(model, anchor, c);
      j[c] = o.j;
      shares[c] = o.shares;
    }
    return out.sort((a, b) => {
      if (a === anchor || b === anchor) return a === anchor ? -1 : 1;
      return j[b] - j[a] || shares[b] - shares[a] || bySize(a, b);
    });
  }
  return out.sort(bySize);
}

// ============================================================================
// layout
// ============================================================================

// Geometry of the matrix (spec §3, §4.4, §10.3): column x prefix sums with
// the group gaps and the no-dictionary fold, and the flat row model (bands,
// rows, visit sub-rows, tallies) with y prefix sums. Hit tests are binary
// searches over those sums.

// ---------------------------------------------------------------------------
// Columns
// ---------------------------------------------------------------------------

interface Group {
  key: string;
  label: string;
  cohorts: number[];
}

function dictionaryGroups(model: LoomModel, state: LoomViewState, shown: number[]): Group[] {
  const cs = model.cohorts;
  const make = <K extends string>(keys: readonly K[], of: (c: number) => K, label: (k: K) => string): Group[] =>
    keys.map(k => ({key: k, label: label(k), cohorts: shown.filter(c => of(c) === k)}));
  switch (state.colGroup) {
    case 'tier':
      return make(
        ['profiled', 'dictionary'] as const,
        c => cs[c].tier as 'profiled' | 'dictionary',
        k => TIER_LABEL[k]
      );
    case 'type':
      return make(
        STUDY_FAMILIES,
        c => cs[c].family,
        k => FAMILY_LABEL[k]
      );
    case 'design':
      return make(
        DESIGN_FAMILIES,
        c => cs[c].design,
        k => DESIGN_LABEL[k]
      );
    case 'status':
      return make(
        COHORT_STATUSES,
        c => cs[c].status,
        k => STATUS_LABEL[k]
      );
    default:
      return [{key: 'all', label: 'Cohorts', cohorts: shown}];
  }
}

export function buildColumns(model: LoomModel, state: LoomViewState, density: DensitySpec): ColumnLayout {
  const K = model.K;
  const visibleMask = new Uint8Array(K);
  const cohortToColumn = new Int32Array(K).fill(-1);
  const shown: number[] = [];
  const fog: number[] = [];
  let hiddenByFilter = 0;
  for (const c of model.cohorts) {
    if (!cohortPasses(model, state, c.i)) {
      hiddenByFilter++;
      continue;
    }
    if (c.tier === 'none') fog.push(c.i);
    else {
      shown.push(c.i);
      visibleMask[c.i] = 1;
    }
  }

  // No-dictionary cohorts always form the last group: folded into one hatched
  // column, or one narrow hatched column each.
  const groups = dictionaryGroups(model, state, shown)
    .filter(g => g.cohorts.length)
    .map(g => ({...g, cohorts: orderCohorts(model, g.cohorts, state)}));
  const visibleFog = orderCohorts(model, fog, state);
  if (visibleFog.length) groups.push({key: 'none', label: TIER_LABEL.none, cohorts: visibleFog});

  const columns: GridColumn[] = [];
  const outGroups: ColumnLayout['groups'] = [];
  let x = 0;
  groups.forEach((g, gi) => {
    if (gi > 0) x += GROUP_GAP;
    const x0 = x;
    if (g.key === 'none' && state.fog === 'fold') {
      columns.push({kind: 'fogFold', cohort: -1, x, w: FOG_FOLD_W, group: gi});
      // Every folded cohort points at the fold (a knowledge-bar pip scrolls there).
      for (const c of g.cohorts) cohortToColumn[c] = columns.length - 1;
      x += FOG_FOLD_W;
    } else {
      const w = g.key === 'none' ? FOG_COL_W : density.pitchX;
      for (const c of g.cohorts) {
        cohortToColumn[c] = columns.length;
        columns.push({kind: 'cohort', cohort: c, x, w, group: gi});
        x += w;
      }
    }
    outGroups.push({key: g.key, label: g.label, count: g.cohorts.length, x: x0, w: x - x0});
  });

  return {columns, groups: outGroups, width: x, visibleMask, visibleFog, cohortToColumn, hiddenByFilter};
}

// Index into columns.columns of the column under x (grid coordinates), -1 in
// a group gap or outside.
export function columnAtX(columns: ColumnLayout, x: number): number {
  const cols = columns.columns;
  let lo = 0;
  let hi = cols.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const col = cols[mid];
    if (x < col.x) hi = mid - 1;
    else if (x >= col.x + col.w) lo = mid + 1;
    else return mid;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

type Bucket = 'all' | 'most' | 'some' | 'two' | 'one';
const BUCKETS: Bucket[] = ['all', 'most', 'some', 'two', 'one'];

// Coverage buckets over the D shown dictionary cohorts: all (= D), two
// (exactly 2), most (at least half), some (3 up to half), one.
function bucketOf(cov: number, D: number, half: number): Bucket {
  if (cov >= D && D > 0) return 'all';
  if (cov === 2) return 'two';
  if (cov >= half && cov >= 3) return 'most';
  if (cov >= 3) return 'some';
  return 'one';
}

function bucketLabel(b: Bucket, D: number, half: number): string {
  const range = (lo: number, hi: number) => (lo === hi ? `${lo}` : `${lo}–${hi}`);
  switch (b) {
    case 'all':
      return D === 1 ? 'The one shown cohort' : `All ${D} shown cohorts`;
    case 'most':
      return `Most cohorts (${range(Math.max(3, half), D - 1)})`;
    case 'some':
      return `Some cohorts (${range(3, Math.max(3, half - 1))})`;
    case 'two':
      return 'Two cohorts';
    case 'one':
      return 'One cohort';
  }
}

export function buildRowLayout(
  model: LoomModel,
  stats: RowStats,
  order: number[],
  state: LoomViewState,
  density: DensitySpec
): RowLayout {
  const items: FlatItem[] = [];
  const rowToItem = new Int32Array(model.rows.length).fill(-1);
  const displayOrder: number[] = [];
  const h = density.pitchY;
  let y = 0;

  const unfolded = new Set<number>();
  for (const key of state.unfolded) {
    const r = model.tokenToRow.get(key);
    if (r !== undefined) unfolded.add(r);
  }
  // Sub-rows follow the visit lens: a slice at Baseline unfolds nothing else.
  const lensMask = state.visit.kind === 'all' ? ALL_SLOTS_MASK : state.visit.mask & ALL_SLOTS_MASK;

  const pushRow = (r: number) => {
    rowToItem[r] = items.length;
    displayOrder.push(r);
    items.push({kind: 'row', row: r, y, h});
    y += h;
    if (!unfolded.has(r)) return;
    const slots = model.rows[r].slotMask & lensMask;
    for (const slot of VISIT_SLOTS) {
      if (!(slots & SLOT_BIT[slot])) continue;
      items.push({kind: 'sub', row: r, slot, y, h});
      y += h;
    }
  };

  if (state.rowGroup === 'none') {
    order.forEach(pushRow);
    return {items, totalHeight: y, rowToItem, order: displayOrder};
  }

  if (state.rowGroup === 'domain') {
    const byDomain = new Map<Domain, number[]>(DOMAINS.map(d => [d, []]));
    for (const r of order) byDomain.get(model.rows[r].domain)?.push(r);
    const collapsed = new Set(state.collapsedBands);
    for (const d of DOMAINS) {
      const rows = byDomain.get(d) ?? [];
      if (!rows.length) continue;
      const isCollapsed = collapsed.has(d);
      items.push({
        kind: 'band',
        domain: d,
        label: DOMAIN_LABEL[d],
        count: rows.length,
        collapsed: isCollapsed,
        y,
        h: BAND_H
      });
      y += BAND_H;
      if (isCollapsed) {
        // One tally row stands for the band; its rows point at it (reveal).
        for (const r of rows) {
          rowToItem[r] = items.length;
          displayOrder.push(r);
        }
        items.push({kind: 'tally', domain: d, rows, y, h: TALLY_H});
        y += TALLY_H;
      } else rows.forEach(pushRow);
    }
    return {items, totalHeight: y, rowToItem, order: displayOrder};
  }

  // Coverage buckets (not collapsible: collapsed bands are domains).
  let D = 0;
  for (const c of model.cohorts) if (c.tier !== 'none' && cohortPasses(model, state, c.i)) D++;
  const half = Math.ceil(D / 2);
  const byBucket = new Map<Bucket, number[]>(BUCKETS.map(b => [b, []]));
  for (const r of order) byBucket.get(bucketOf(stats.covDict[r], D, half))?.push(r);
  for (const b of BUCKETS) {
    const rows = byBucket.get(b) ?? [];
    if (!rows.length) continue;
    items.push({
      kind: 'band',
      domain: null,
      label: bucketLabel(b, D, half),
      count: rows.length,
      collapsed: false,
      y,
      h: BAND_H
    });
    y += BAND_H;
    rows.forEach(pushRow);
  }
  return {items, totalHeight: y, rowToItem, order: displayOrder};
}

// Index into layout.items of the item under y (grid body coordinates), -1
// outside.
export function itemAtY(layout: RowLayout, y: number): number {
  const items = layout.items;
  if (y < 0 || y >= layout.totalHeight) return -1;
  let lo = 0;
  let hi = items.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const it = items[mid];
    if (y < it.y) hi = mid - 1;
    else if (y >= it.y + it.h) lo = mid + 1;
    else return mid;
  }
  return -1;
}

// ============================================================================
// state
// ============================================================================

// Pools have at most 12 slots: drop-to-gain recomputes the Pool once per slot
// and the dock lists every slot, so the cap keeps both instant and legible.
export const POOL_MAX_SLOTS = 12;

const toggle = <T>(list: T[], item: T): T[] => (list.includes(item) ? list.filter(x => x !== item) : [...list, item]);

const pinOf = (pins: PinState[], cohortId: string): PinKind | null =>
  pins.find(p => p.cohortId === cohortId)?.kind ?? null;

// Replaces a cohort's pin in place (keeping the recipe's clause order), adds
// it at the end, or removes it (kind null).
function withPin(pins: PinState[], cohortId: string, kind: PinKind | null): PinState[] {
  if (kind === null) return pins.filter(p => p.cohortId !== cohortId);
  if (pins.some(p => p.cohortId === cohortId)) return pins.map(p => (p.cohortId === cohortId ? {cohortId, kind} : p));
  return [...pins, {cohortId, kind}];
}

// ○ → ✓ → ✓✓ → ✕ → ○; ✓✓ ("with data") only exists for profiled cohorts.
export function nextPin(current: PinKind | null, profiled: boolean): PinKind | null {
  switch (current) {
    case null:
      return 'coded';
    case 'coded':
      return profiled ? 'data' : 'not';
    case 'data':
      return 'not';
    case 'not':
      return null;
  }
}

// The row keys a Pool slot holds (a braid joins several with '~').
export const slotRowKeys = (slot: string): string[] => slot.split('~').filter(Boolean);

export function reduceViewState(state: LoomViewState, action: LoomAction): LoomViewState {
  switch (action.type) {
    case 'patch':
      return {...state, ...action.patch};
    case 'reset': {
      // Reset restores the default VIEW. The Pool is the researcher's work in
      // progress, not a view setting, so it survives (it has its own Clear).
      const d = defaultViewState();
      return {
        ...d,
        pool: state.pool,
        poolVisit: state.poolVisit,
        poolThreshold: state.poolThreshold,
        poolHandoff: state.poolHandoff,
        poolIncludeDict: state.poolIncludeDict,
        debug: state.debug
      };
    }
    case 'cyclePin':
      return {
        ...state,
        pins: withPin(state.pins, action.cohortId, nextPin(pinOf(state.pins, action.cohortId), action.profiled))
      };
    case 'setPin':
      return {...state, pins: withPin(state.pins, action.cohortId, action.kind)};
    case 'toggleHiddenRow':
      return {...state, hiddenRows: toggle(state.hiddenRows, action.key)};
    case 'toggleHiddenCohort': {
      const hiding = !state.hiddenCohorts.includes(action.cohortId);
      return {
        ...state,
        hiddenCohorts: toggle(state.hiddenCohorts, action.cohortId),
        // Hiding a cohort that the "only" list names would leave the two lists
        // fighting; hide wins.
        onlyCohorts: hiding ? state.onlyCohorts.filter(id => id !== action.cohortId) : state.onlyCohorts
      };
    }
    case 'soloCohort': {
      const solo = state.onlyCohorts.length === 1 && state.onlyCohorts[0] === action.cohortId;
      return {
        ...state,
        onlyCohorts: solo ? [] : [action.cohortId],
        hiddenCohorts: solo ? state.hiddenCohorts : state.hiddenCohorts.filter(id => id !== action.cohortId)
      };
    }
    case 'toggleBand':
      return {...state, collapsedBands: toggle(state.collapsedBands, action.domain)};
    case 'toggleUnfold':
      return {...state, unfolded: toggle(state.unfolded, action.key)};
    case 'pull': {
      const held = new Set(state.pool.flatMap(slotRowKeys));
      const pool = [...state.pool];
      for (const key of action.keys) {
        if (!key || held.has(key) || pool.length >= POOL_MAX_SLOTS) continue;
        held.add(key);
        pool.push(key);
      }
      return pool.length === state.pool.length ? state : {...state, pool};
    }
    case 'release':
      return state.pool.includes(action.key) ? {...state, pool: state.pool.filter(k => k !== action.key)} : state;
    case 'clearPool':
      return state.pool.length ? {...state, pool: []} : state;
  }
}

// ============================================================================
// pool
// ============================================================================

// The Pool (spec §6 "Pool rows", §7.5, §9.3): which shown cohorts have every
// pulled concept at the Pool visit, and how many rows could hold all of them.
// Only bounds are claimed:
//   per slot s in cohort c   lo_s = max n(v), hi_s = min(N, Σ n(v)) over the
//                            eligible members of the slot's rows at the visit
//   joint ("has all k")      upper = min_s hi_s
//                            lower = max(0, Σ_s lo_s − (k − 1)·N)   (Fréchet)
// A cohort qualifies only when every slot has a counted member (at or above
// the Pool's completeness threshold). Unknown is never shown as 0, and
// dictionary-only and no-dictionary cohorts are text lines, never length.

// What one slot finds in one cohort.
type SlotFinding = 'ok' | 'below' | 'zero' | 'notInEda' | 'pending' | 'dictionary' | 'absent';

interface SlotCell {
  finding: SlotFinding;
  lo: number;
  hi: number;
  repVars: number[];
  allVars: number[];
}

const PERSLOT_STATE: Record<SlotFinding, PoolCohort['perSlot'][number]['state']> = {
  ok: 'counted',
  below: 'counted',
  zero: 'zero',
  notInEda: 'notInEda',
  pending: 'dictionary',
  dictionary: 'dictionary',
  absent: 'absent'
};

const LACKING = new Set<SlotFinding>(['below', 'zero', 'absent']);
const UNKNOWN = new Set<SlotFinding>(['notInEda', 'pending']);

const STATUS_ORDER: Record<PoolStatus, number> = {qualifies: 0, nearMiss: 1, unknown: 2, dictionary: 3, lacks: 4};

const uniq = (xs: number[]): number[] => Array.from(new Set(xs));

export function poolSlots(model: LoomModel, keys: PoolSlotKey[]): PoolSlot[] {
  const out: PoolSlot[] = [];
  const seen = new Set<string>();
  for (const key of keys) {
    // Any token of a row resolves it, so a pulled concept survives a rebuild
    // that changes its canonical key; keys that match nothing are dropped.
    const rows = uniq(
      slotRowKeys(key)
        .map(k => model.tokenToRow.get(k))
        .filter((r): r is number => r !== undefined)
    ).sort((a, b) => a - b);
    const sig = rows.join(',');
    if (!rows.length || seen.has(sig)) continue;
    seen.add(sig);
    out.push({key, rows, label: rows.map(r => model.rows[r].label).join(' + ')});
  }
  return out;
}

function lensSlotMask(lens: VisitLens): number {
  return lens.kind === 'all' ? ALL_SLOTS_MASK : lens.mask & ALL_SLOTS_MASK;
}

// One slot (its rows) in one cohort at the Pool visit.
function slotCell(model: LoomModel, slot: PoolSlot, c: LoomCohort, lens: VisitLens, threshold: number): SlotCell {
  let best: SlotFinding = 'absent';
  const rank: Record<SlotFinding, number> = {
    ok: 0,
    below: 1,
    zero: 2,
    notInEda: 3,
    pending: 4,
    dictionary: 5,
    absent: 6
  };
  let lo = 0;
  let sum = 0;
  const allVars: number[] = [];
  // One representative per visit: the best across the slot's rows.
  const repByVisit = new Map<number, {rep: number; lo: number}>();
  const mask = lensSlotMask(lens);
  for (const r of slot.rows) {
    const cell = resolveCell(model, r, c.i, lens, threshold, false);
    let finding: SlotFinding;
    switch (cell.state) {
      case CellState.Counted:
        finding = cell.flags & CellFlag.BelowFloor ? 'below' : 'ok';
        break;
      case CellState.Zero:
        finding = 'zero';
        break;
      case CellState.NotInEda:
        finding = 'notInEda';
        break;
      case CellState.Pending:
        finding = 'pending';
        break;
      case CellState.Dictionary:
        finding = 'dictionary';
        break;
      default:
        finding = 'absent';
    }
    if (rank[finding] < rank[best]) best = finding;
    if (finding === 'absent') continue;
    if (cell.state === CellState.Counted) {
      lo = Math.max(lo, cell.lo);
      sum += cell.hi;
    }
    allVars.push(...lensMembers(model, r, c.i, lens, false).members);
    const visits = model.rows[r].slotMask & mask;
    for (let s = 0; s < VISIT_SLOTS.length; s++) {
      const bit = SLOT_BIT[VISIT_SLOTS[s]];
      if (!(visits & bit)) continue;
      const at = resolveCell(model, r, c.i, {kind: 'slots', mask: bit}, 0, false);
      if (at.rep < 0) continue;
      const held = repByVisit.get(s);
      if (!held || at.lo > held.lo) repByVisit.set(s, {rep: at.rep, lo: at.lo});
    }
  }
  const counted = best === 'ok' || best === 'below';
  const hi = counted ? Math.max(lo, c.nRows !== null ? Math.min(c.nRows, sum) : sum) : 0;
  const repVars = Array.from(repByVisit.keys())
    .sort((a, b) => a - b)
    .map(s => repByVisit.get(s)?.rep ?? -1)
    .filter(v => v >= 0);
  return {finding: best, lo: counted ? lo : 0, hi, repVars: uniq(repVars), allVars: uniq(allVars)};
}

interface Verdict {
  status: PoolStatus;
  lower: number;
  upper: number;
  limiting: string | null;
  missing: string[];
}

// The status of one cohort over a subset of the slots (all of them, or all
// but one for drop-to-gain).
function judge(c: LoomCohort, row: SlotCell[], slots: PoolSlot[], use: number[], threshold: number): Verdict {
  // A lacking slot says why when it is not simply absent.
  const why = (i: number): string => {
    const f = row[i].finding;
    if (f === 'below') return `${slots[i].label} (below ${threshold}%)`;
    if (f === 'zero') return `${slots[i].label} (no values)`;
    return slots[i].label;
  };
  if (c.tier !== 'profiled') {
    // Dictionary only (or counts unavailable): members listed, counts unknown.
    const absent = use.filter(i => row[i].finding === 'absent');
    return absent.length
      ? {status: 'lacks', lower: 0, upper: 0, limiting: null, missing: absent.map(why)}
      : {status: 'dictionary', lower: 0, upper: 0, limiting: null, missing: []};
  }
  const lacking = use.filter(i => LACKING.has(row[i].finding));
  const unknown = use.filter(i => UNKNOWN.has(row[i].finding));
  if (lacking.length) {
    const near = lacking.length === 1 && !unknown.length && use.length >= 2;
    return {status: near ? 'nearMiss' : 'lacks', lower: 0, upper: 0, limiting: null, missing: lacking.map(why)};
  }
  if (unknown.length) return {status: 'unknown', lower: 0, upper: 0, limiting: null, missing: unknown.map(why)};

  const k = use.length;
  const N = c.nRows;
  let upper = Infinity;
  let sumLo = 0;
  for (const i of use) {
    upper = Math.min(upper, row[i].hi);
    sumLo += row[i].lo;
  }
  // Fréchet needs N; with the dataset size unknown only one slot has a floor.
  const lower = N !== null ? Math.max(0, sumLo - (k - 1) * N) : k === 1 ? row[use[0]].lo : 0;
  // The slot(s) with the smallest upper bound limit the cohort, unless the
  // bound is the dataset size itself.
  const limiting =
    k >= 2 && (N === null || upper < N)
      ? use
          .filter(i => row[i].hi === upper)
          .map(i => slots[i].label)
          .join(' and ')
      : null;
  return {status: 'qualifies', lower, upper, limiting, missing: []};
}

function visitPhrase(lens: VisitLens): string {
  const mask = lensSlotMask(lens);
  if (mask === ALL_SLOTS_MASK) return 'across all visits';
  if (mask === FOLLOW_UP_MASK) return 'at any follow-up visit';
  if (mask === SLOT_BIT.UN) return 'at unanchored visits';
  return `at ${lensLabel(lens)}`;
}

const poolPlural = (n: number, one: string, many = `${one}s`): string => `${fmtInt(n)} ${n === 1 ? one : many}`;

const listText = (items: string[]): string =>
  items.length <= 1 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;

const boundsText = (lower: number, upper: number): string =>
  lower === upper ? fmtInt(upper) : `${fmtInt(lower)}–${fmtInt(upper)}`;

function summarize(
  model: LoomModel,
  state: LoomViewState,
  slots: PoolSlot[],
  cohorts: PoolCohort[],
  pooledLower: number,
  pooledUpper: number,
  dictionaryOnly: {count: number; declared: number},
  fog: {count: number; declared: number}
): string {
  const k = slots.length;
  const parts: string[] = [];
  const at = visitPhrase(state.poolVisit);
  const threshold = state.poolThreshold > 0 ? ` (completeness ≥ ${state.poolThreshold}%)` : '';
  parts.push(`Pool: ${listText(slots.map(s => s.label))} ${at}${threshold}.`);
  if (model.countsStatus === 'loading') parts.push('Observation counts are still loading.');
  if (model.countsStatus === 'unavailable') {
    parts.push('Observation counts could not be loaded: no cohort can be counted.');
  }

  const all = k === 1 ? 'it' : k === 2 ? 'both' : `all ${k}`;
  const profiled = cohorts.filter(c => model.cohorts[c.cohort].tier === 'profiled').length;
  const qualifying = cohorts.filter(c => c.status === 'qualifies');
  const name = (c: PoolCohort) => model.cohorts[c.cohort].id;
  if (qualifying.length) {
    const each = qualifying.map(c => {
      const of = c.N !== null ? ` of ${fmtInt(c.N)}` : ' (dataset size unknown)';
      const lim = c.limitingSlot ? `, limited by ${c.limitingSlot}` : '';
      return `${name(c)} ${boundsText(c.lower, c.upper)}${of}${lim}`;
    });
    const has = qualifying.length === 1 ? 'has' : 'have';
    parts.push(
      `${fmtInt(qualifying.length)} of ${poolPlural(profiled, 'shown profiled cohort')} ${has} ${all}: ${each.join('; ')}.`
    );
    const rows = k === 1 ? 'a value' : k === 2 ? 'both concepts' : 'every concept';
    parts.push(
      `Together: ${boundsText(pooledLower, pooledUpper)} rows with ${rows} (bounds from per-variable counts, not unique patients).`
    );
  } else {
    parts.push(`No shown profiled cohort has ${all} with counted values.`);
  }
  const near = cohorts.filter(c => c.status === 'nearMiss');
  if (near.length) parts.push(`Near misses: ${near.map(c => `${name(c)} lacks ${c.missing[0]}`).join('; ')}.`);
  const unknown = cohorts.filter(c => c.status === 'unknown');
  if (unknown.length && model.countsStatus === 'loading') {
    parts.push(`Unknown until the counts load: ${listText(unknown.map(name))}.`);
  } else if (unknown.length) {
    const each = unknown.map(c => `${name(c)} (${listText(c.missing)} not in profiled data ${at})`);
    parts.push(`Unknown: ${each.join('; ')}.`);
  }
  const lacks = cohorts.filter(c => c.status === 'lacks');
  if (lacks.length) {
    const verb = lacks.length === 1 ? 'lacks' : 'lack';
    parts.push(`${poolPlural(lacks.length, 'shown cohort')} ${verb} ${k === 1 ? 'it' : 'at least one concept'}.`);
  }
  if (dictionaryOnly.count) {
    const declared = dictionaryOnly.declared ? ` (~${fmtCompact(dictionaryOnly.declared)} declared participants)` : '';
    parts.push(
      `+? ${poolPlural(dictionaryOnly.count, 'dictionary-only cohort')} ${dictionaryOnly.count === 1 ? 'codes' : 'code'} ${k === 1 ? 'it' : 'every concept'}${declared}: counts unknown.`
    );
  }
  if (fog.count) {
    const declared = fog.declared ? ` (~${fmtCompact(fog.declared)} declared participants)` : '';
    parts.push(`? ${poolPlural(fog.count, 'cohort')} without a dictionary${declared}: unknown, not absent.`);
  }
  parts.push(CAVEATS.pool);
  return parts.join(' ');
}

export function computePool(model: LoomModel, state: LoomViewState, columns: ColumnLayout): PoolResult | null {
  if (!state.pool.length) return null;
  const slots = poolSlots(model, state.pool);
  const lens = state.poolVisit;
  const fogCohorts = columns.visibleFog.map(c => model.cohorts[c]);
  const fog = {count: fogCohorts.length, declared: fogCohorts.reduce((s, c) => s + (c.declared ?? 0), 0)};
  if (!slots.length) {
    return {
      slots,
      visit: lens,
      cohorts: [],
      qualifying: 0,
      pooledLower: 0,
      pooledUpper: 0,
      dictionaryOnly: {count: 0, declared: 0},
      fog,
      dropToGain: [],
      summary: 'None of the pulled concepts exists in the current data.'
    };
  }

  const shown = model.cohorts.filter(c => columns.visibleMask[c.i] && c.tier !== 'none');
  const grid = shown.map(c => slots.map(s => slotCell(model, s, c, lens, state.poolThreshold)));
  const everySlot = slots.map((_, i) => i);
  const tally = (use: number[]) => {
    let count = 0;
    let upper = 0;
    shown.forEach((c, ci) => {
      const v = judge(c, grid[ci], slots, use, state.poolThreshold);
      if (v.status === 'qualifies') {
        count++;
        upper += v.upper;
      }
    });
    return {count, upper};
  };

  const cohorts: PoolCohort[] = shown.map((c, ci) => {
    const v = judge(c, grid[ci], slots, everySlot, state.poolThreshold);
    return {
      cohort: c.i,
      status: v.status,
      lower: v.lower,
      upper: v.upper,
      N: c.nRows,
      limitingSlot: v.limiting,
      missing: v.missing,
      perSlot: grid[ci].map(cell => ({
        lo: cell.lo,
        hi: cell.hi,
        repVars: cell.repVars,
        allVars: cell.allVars,
        state: PERSLOT_STATE[cell.finding]
      }))
    };
  });
  cohorts.sort(
    (a, b) =>
      STATUS_ORDER[a.status] - STATUS_ORDER[b.status] ||
      b.upper - a.upper ||
      b.lower - a.lower ||
      a.missing.length - b.missing.length ||
      model.cohorts[a.cohort].id.localeCompare(model.cohorts[b.cohort].id)
  );

  const qualifying = cohorts.filter(c => c.status === 'qualifies');
  const pooledLower = qualifying.reduce((s, c) => s + c.lower, 0);
  const pooledUpper = qualifying.reduce((s, c) => s + c.upper, 0);
  const dict = cohorts.filter(c => c.status === 'dictionary').map(c => model.cohorts[c.cohort]);
  const dictionaryOnly = {count: dict.length, declared: dict.reduce((s, c) => s + (c.declared ?? 0), 0)};

  // Drop-to-gain: what each slot costs (≤ 12 slots x ≤ 60 cohorts: instant).
  const dropToGain =
    slots.length < 2
      ? []
      : slots.map((s, i) => {
          const without = tally(everySlot.filter(j => j !== i));
          return {
            slot: s.label,
            plusCohorts: without.count - qualifying.length,
            plusUpper: without.upper - pooledUpper
          };
        });

  return {
    slots,
    visit: lens,
    cohorts,
    qualifying: qualifying.length,
    pooledLower,
    pooledUpper,
    dictionaryOnly,
    fog,
    dropToGain,
    summary: summarize(model, state, slots, cohorts, pooledLower, pooledUpper, dictionaryOnly, fog)
  };
}

// The Pool's handoff to the DCR basket: the cohorts that have every concept
// (qualifies) or might (unknown; dictionary-only when included), with one
// representative per concept per visit, or every variable in those cells.
// Cohorts whose counts show a concept missing stay out; among the others the
// preview's per-cohort checkboxes decide (the page informs, it never gates).
export function poolBasketPreview(
  model: LoomModel,
  pool: PoolResult,
  handoff: 'rep' | 'all',
  includeDict: boolean,
  basket: Record<string, string[]>
): BasketPreview {
  const byCohort: BasketPreview['byCohort'] = [];
  for (const pc of pool.cohorts) {
    const take = pc.status === 'qualifies' || pc.status === 'unknown' || (includeDict && pc.status === 'dictionary');
    if (!take) continue;
    const c = model.cohorts[pc.cohort];
    const vars = pc.perSlot.flatMap(s => (handoff === 'rep' ? s.repVars : s.allVars));
    const names = Array.from(new Set(vars.map(v => model.variables[v].name)));
    if (!names.length) continue;
    const have = new Set(basket?.[c.id] ?? []);
    byCohort.push({cohortId: c.id, names, alreadyIn: names.filter(n => have.has(n)).length, tier: c.tier});
  }
  const k = pool.slots.length;
  return {
    byCohort,
    source: `Pool (${poolPlural(k, 'concept')} · ${lensLabel(pool.visit)})`
  };
}

// ============================================================================
// selfCheck
// ============================================================================

// The invariants of spec §5 step 14, run on /loom?debug=1:
// 1. Explore superset: every semantic match the Explore badges report is a
//    pair of members of one Loom row (home or guest), so no cohort the badge
//    names can be missing from the row.
// 2. Rows are exactly the connected components of the code <-> ID pair graph:
//    no other union happened, so no secondary token ever glued two components.
// 3. Home memberships + unmapped = total variables, and the member arrays
//    agree with every variable's home and guest rows.

const MAX_LISTED = 200;

// Explore's own normalization (utils/semanticMatches.ts): split on '|', trim,
// lowercase, drop 'na' and '0'.
function exploreValues(raw: string): Set<string> {
  const out = new Set<string>();
  for (const part of raw.split('|')) {
    const v = part.trim();
    if (v && v.toLowerCase() !== 'na' && v !== '0') out.add(v.toLowerCase());
  }
  return out;
}

export function runSelfCheck(
  model: LoomModel,
  index: SemanticMatchIndex
): {ok: boolean; violations: string[]; summary: string} {
  const violations: string[] = [];
  let total = 0;
  const fail = (msg: string): void => {
    total++;
    if (violations.length < MAX_LISTED) violations.push(msg);
  };
  const {variables, rows, cohorts} = model;
  const isMember = (i: number, row: number): boolean =>
    variables[i].home === row || variables[i].guestRows.includes(row);
  const inCell = (i: number, row: number): boolean => {
    const key = row * model.K + variables[i].cohort;
    for (let p = model.cellStart[key]; p < model.cellStart[key + 1]; p++) if (model.cellItems[p] === i) return true;
    return false;
  };
  const describe = (i: number): string => `${cohorts[variables[i].cohort].id}::${variables[i].name}`;

  // --- 1. Explore superset.
  const byKey = new Map<string, number>();
  for (const v of variables) byKey.set(`${cohorts[v.cohort].id}::${v.name}`, v.i);
  let verified = 0;
  let rejected = 0;
  let outsideMode = 0;
  index.byVariable.forEach((entry, key) => {
    const vi = byKey.get(key);
    if (vi === undefined) {
      fail(`Explore variable ${key} is not in the Loom model`);
      return;
    }
    for (const m of entry.matches) {
      const mi = byKey.get(`${m.cohortId}::${m.varName}`);
      if (mi === undefined) {
        fail(`Explore match ${m.cohortId}::${m.varName} (of ${key}) is not in the Loom model`);
        continue;
      }
      const ids = m.matchedOn.filter(id => (id === 'concept_code' ? model.match !== 'omop' : model.match !== 'code'));
      if (!ids.length) {
        outsideMode++;
        continue;
      }
      let tokens = 0;
      for (const id of ids) {
        const mine = exploreValues(id === 'concept_code' ? variables[vi].conceptCode : variables[vi].omopId);
        const theirs = exploreValues(id === 'concept_code' ? variables[mi].conceptCode : variables[mi].omopId);
        mine.forEach(value => {
          if (!theirs.has(value)) return;
          const token = id === 'concept_code' ? ccToken(value) : oiToken(value);
          if (!token) return;
          tokens++;
          const row = model.tokenToRow.get(token);
          if (row === undefined) {
            fail(`${key} ↔ ${describe(mi)}: token ${token} has no row`);
            return;
          }
          if (!isMember(vi, row) || !inCell(vi, row))
            fail(`${key} is not a member of row ${rows[row].key} that holds its token ${token}`);
          if (!isMember(mi, row) || !inCell(mi, row))
            fail(`${key} ↔ ${describe(mi)}: ${m.cohortId} missing from row ${rows[row].key} (token ${token})`);
        });
      }
      // Explore also matches on placeholders ('N/A', 'none') and malformed
      // OMOP IDs, which Loom rejects on purpose: a divergence, not a violation.
      if (tokens) verified++;
      else rejected++;
    }
  });

  // --- 2. Rows = components of the pair graph.
  const adjacency = new Map<string, string[]>();
  const link = (a: string, b: string): void => {
    const list = adjacency.get(a);
    if (list) list.push(b);
    else adjacency.set(a, [b]);
  };
  for (const v of variables) {
    for (const [c, o] of variableIdentity({concept_code: v.conceptCode, omop_id: v.omopId}, model.match).pairs) {
      link(c, o);
      link(o, c);
    }
  }
  let rowTokens = 0;
  for (const row of rows) {
    rowTokens += row.tokens.length;
    const seen = new Set<string>([row.tokens[0]]);
    const queue = [row.tokens[0]];
    while (queue.length) {
      const t = queue.pop() as string;
      for (const u of adjacency.get(t) ?? []) {
        if (!seen.has(u)) {
          seen.add(u);
          queue.push(u);
        }
      }
    }
    const own = new Set(row.tokens);
    if (seen.size !== own.size || row.tokens.some(t => !seen.has(t)))
      fail(`Row ${row.key}: its ${own.size} tokens are not one pair-graph component (${seen.size} reachable)`);
    for (const t of row.tokens)
      if (model.tokenToRow.get(t) !== row.i) fail(`Token ${t} does not resolve to its row ${row.key}`);
  }
  if (rowTokens !== model.tokenToRow.size)
    fail(`Rows hold ${rowTokens} tokens but tokenToRow has ${model.tokenToRow.size}: a token sits in two rows`);
  for (const v of variables) {
    for (const t of v.tokens) {
      const row = model.tokenToRow.get(t);
      if (row === undefined) fail(`${describe(v.i)}: token ${t} has no row`);
      else if (v.head.includes(t) ? row !== v.home : row !== v.home && !v.guestRows.includes(row))
        fail(`${describe(v.i)}: token ${t} is in row ${rows[row].key}, which is neither its home nor a guest row`);
    }
  }

  // --- 3. Home memberships + unmapped = total variables.
  let homes = 0;
  for (const v of variables) {
    if (v.home < 0) {
      if (v.tokens.length) fail(`${describe(v.i)} has identifiers but no home row`);
      continue;
    }
    homes++;
    if (!inCell(v.i, v.home)) fail(`${describe(v.i)} is missing from its home cell`);
    for (const g of v.guestRows)
      if (!inCell(v.i, g)) fail(`${describe(v.i)} is missing from its guest cell in ${rows[g].key}`);
  }
  let cellHomes = 0;
  for (let key = 0; key < model.cellHomeCount.length; key++) cellHomes += model.cellHomeCount[key];
  if (homes + model.unmapped.length !== variables.length)
    fail(`${homes} home memberships + ${model.unmapped.length} unmapped ≠ ${variables.length} variables`);
  if (cellHomes !== homes) fail(`The cells hold ${cellHomes} home members but ${homes} variables have a home row`);

  if (total > violations.length) violations.push(`… and ${total - violations.length} more`);
  const summary = [
    `Loom self-check: ${total === 0 ? 'OK' : `${total} violation${total === 1 ? '' : 's'}`}`,
    `${verified} Explore matches found in Loom rows` +
      (rejected ? ` (${rejected} more only via placeholder or malformed values Loom rejects)` : '') +
      (outsideMode ? ` (${outsideMode} outside match mode '${model.match}')` : ''),
    `${rows.length} rows · ${homes} home memberships + ${model.unmapped.length} unmapped = ${variables.length} variables`
  ].join(' · ');
  return {ok: total === 0, violations, summary};
}

// ============================================================================
// dcrBasket
// ============================================================================

// The Data Clean Room basket (sessionStorage 'dataCleanRoom', mirrored in
// CohortsContext): cohort id -> variable names. Every write goes through
// here: updates are immutable, never add a name twice, and drop a cohort once
// its list is empty. The functions keep any other field of the stored object.

export type DcrBasket = {cohorts: Record<string, string[]>};

const STORAGE_KEY = 'dataCleanRoom';
const PULSE_CLASS = 'dcr-highlight-effect';
const PULSE_MS = 2000;

const cohortsOf = (dcr: DcrBasket | null | undefined): Record<string, string[]> =>
  dcr && dcr.cohorts && typeof dcr.cohorts === 'object' ? dcr.cohorts : {};

const dcrBasketUnique = (names: string[]): string[] => Array.from(new Set(names.filter(n => typeof n === 'string' && n)));

// Returns the same object when nothing changes, so a second identical add is
// a no-op for React and for the Undo toast.
export function addVariables<T extends DcrBasket>(dcr: T, cohortId: string, names: string[]): T {
  const cohorts = cohortsOf(dcr);
  const current = Array.isArray(cohorts[cohortId]) ? cohorts[cohortId] : [];
  const have = new Set(current);
  const fresh = dcrBasketUnique(names).filter(n => !have.has(n));
  if (!fresh.length) return dcr;
  return {...dcr, cohorts: {...cohorts, [cohortId]: [...unique(current), ...fresh]}};
}

export function removeVariables<T extends DcrBasket>(dcr: T, cohortId: string, names: string[]): T {
  const cohorts = cohortsOf(dcr);
  const current = cohorts[cohortId];
  if (!Array.isArray(current)) return dcr;
  const drop = new Set(names);
  const kept = dcrBasketUnique(current).filter(n => !drop.has(n));
  if (kept.length === current.length) return dcr;
  const next = {...cohorts};
  if (kept.length) next[cohortId] = kept;
  else delete next[cohortId];
  return {...dcr, cohorts: next};
}

export function applyPreview<T extends DcrBasket>(dcr: T, preview: BasketPreview): T {
  return preview.byCohort.reduce((acc, entry) => addVariables(acc, entry.cohortId, entry.names), dcr);
}

export function countAlreadyIn(dcr: DcrBasket, cohortId: string, names: string[]): number {
  const current = cohortsOf(dcr)[cohortId];
  if (!Array.isArray(current)) return 0;
  const have = new Set(current);
  return dcrBasketUnique(names).filter(n => have.has(n)).length;
}

let pulseTimer: ReturnType<typeof setTimeout> | null = null;

// Saves the basket for the session and pulses the Nav's "Create a Data Clean
// Room" button (skipped under reduced motion). Storage can be unavailable
// (private windows, blocked site data): the in-memory basket still works.
export function persistBasket(dcr: DcrBasket): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(dcr));
  } catch {
    // Not persisted: the basket lives in CohortsContext for this page view.
  }
  if (typeof document === 'undefined') return;
  const reduced = typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  const button = document.getElementById('dcr-button');
  if (!button || reduced) return;
  // Restart the animation when pulsed again before it ended.
  button.classList.remove(PULSE_CLASS);
  void button.offsetWidth;
  button.classList.add(PULSE_CLASS);
  if (pulseTimer) clearTimeout(pulseTimer);
  pulseTimer = setTimeout(() => {
    button.classList.remove(PULSE_CLASS);
    pulseTimer = null;
  }, PULSE_MS);
}

// ============================================================================
// axes
// ============================================================================

// The two axes of the coverage matrix. One cell = one concept in one cohort
// at one visit (the cohort's representative variable for that concept at that
// visit). Cohort axis: one entry per (cohort, visit) the cohort's dictionary
// has mapped variables at, in the column order of the cohort filters; the
// cohorts without a dictionary are one hatched entry each (fog 'cols') or a
// single folded entry (fog 'fold'), since nothing is known about their visits.
// Concept axis: the visible concepts in display order. `transpose` swaps
// which axis runs down the rows.

export type AxisItem =
  | {kind: 'concept'; row: number; group: string}
  | {kind: 'visit'; cohort: number; slot: VisitSlot; group: string; first: boolean}
  | {kind: 'fog'; cohorts: number[]; group: string; first: boolean};

export interface CoverageAxes {
  concepts: AxisItem[];
  visits: AxisItem[];
  rows: AxisItem[];
  cols: AxisItem[];
}

// Slots at which each cohort has at least one mapped variable.
export function cohortSlotMasks(model: LoomModel): Uint16Array {
  const masks = new Uint16Array(model.K);
  for (const v of model.variables) if (v.home >= 0) masks[v.cohort] |= SLOT_BIT[v.slot];
  return masks;
}

export function buildAxes(
  model: LoomModel,
  columns: ColumnLayout,
  order: number[],
  state: LoomViewState,
  slotMasks: Uint16Array
): CoverageAxes {
  const lensMask = state.visit.kind === 'all' ? ALL_SLOTS_MASK : state.visit.mask;
  const visits: AxisItem[] = [];
  for (const col of columns.columns) {
    if (col.kind === 'fogFold') {
      if (columns.visibleFog.length)
        visits.push({kind: 'fog', cohorts: columns.visibleFog.slice(), group: 'fog', first: true});
      continue;
    }
    const c = model.cohorts[col.cohort];
    if (c.tier === 'none') {
      visits.push({kind: 'fog', cohorts: [col.cohort], group: c.id, first: true});
      continue;
    }
    let first = true;
    for (const slot of VISIT_SLOTS) {
      if (!(slotMasks[col.cohort] & SLOT_BIT[slot] & lensMask)) continue;
      visits.push({kind: 'visit', cohort: col.cohort, slot, group: c.id, first});
      first = false;
    }
  }
  // Concepts run in display order, kept together by domain when rows are
  // grouped by domain (stable: the order within a domain is kept).
  const rank = new Map(DOMAINS.map((d, i) => [d, i]));
  const ordered =
    state.rowGroup === 'domain'
      ? order
          .map((row, i) => ({row, i}))
          .sort((a, b) => rank.get(model.rows[a.row].domain)! - rank.get(model.rows[b.row].domain)! || a.i - b.i)
          .map(x => x.row)
      : order;
  const concepts: AxisItem[] = ordered.map(row => ({
    kind: 'concept',
    row,
    group: state.rowGroup === 'domain' ? model.rows[row].domain : ''
  }));
  return state.transpose
    ? {concepts, visits, rows: concepts, cols: visits}
    : {concepts, visits, rows: visits, cols: concepts};
}
