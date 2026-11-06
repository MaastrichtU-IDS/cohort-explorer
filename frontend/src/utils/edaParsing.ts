/**
 * Parsing utilities for EDA output JSON files.
 *
 * The EDA JSON produced by eda_scripts.py stores summary statistics and
 * stringified axis-tick / class-balance information. These helpers convert
 * those strings into structured objects the frontend charting layer can use.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type EdaVarType = 'numeric' | 'categorical' | 'date' | 'text' | 'unknown';

export interface EdaVariable {
  name: string;
  label: string;
  type: EdaVarType;
  rawType: string;

  // Metadata from dictionary
  metadataLabel?: string;
  metadataVarType?: string;
  units?: string;
  conceptCode?: string;
  conceptName?: string;
  omopId?: number;
  domain?: string;
  visit?: string;
  visitConceptName?: string;

  // Observation counts. totalObservations is the number of valid values
  // (the v2 formats say so; the pre-July-2026 v1 file put the row count here).
  totalObservations: number;
  totalRows?: number;
  countEmpty: number;
  countEmptyPct: number;
  // Coded missing values (the dictionary's missing codes), as in v1's "count missing".
  countMissing: number;
  countMissingPct: number;
  // Since July 2026: empty + coded missing together, values that failed to parse,
  // and the missing codes the dictionary declared.
  countMissingTotal?: number;
  countMissingTotalPct?: number;
  countInvalid?: number;
  countInvalidPct?: number;
  missingCodes?: string;
  uniqueValues: number;

  // Numeric stats (only for numeric type)
  mean?: number;
  median?: number;
  mode?: number;
  stdDev?: number;
  variance?: number;
  min?: number;
  max?: number;
  range?: number;
  q1?: number;
  q3?: number;
  iqr?: number;
  skewness?: number;
  kurtosis?: number;
  normalityTest?: string;
  isNormal?: boolean;
  wTest?: number;
  outliersIqr?: number;
  outliersIqrPct?: number;
  outliersZ?: number;
  outliersZPct?: number;
  // Since July 2026 (numeric): robust and tail measures.
  p5?: number;
  p95?: number;
  trimmedMean10?: number;
  cv?: number;                 // std / mean (a ratio, not a percentage)
  mad?: number;                // median absolute deviation
  zeroFractionPct?: number;    // share of valid values equal to zero
  outliersMad?: number;        // |modified z| > 3.5
  outliersMadPct?: number;
  normalityPValue?: number;    // D'Agostino K² p-value (Shapiro when absent)

  // Categorical stats
  classBalance?: { label: string; percentage: number; count: number }[];
  chiSquare?: number;
  chiSquarePValue?: number;
  chiSquareDof?: number;
  mostFrequentCategory?: string;
  // Since July 2026 (categorical): spread of the categories.
  entropyBits?: number;
  normalizedEntropy?: number;  // 0 = one category, 1 = uniform
  giniImpurity?: number;       // 1 - sum p²
  imbalanceRatio?: number;     // largest / smallest category count

  // Date stats (since July 2026): dates as the file prints them
  dateMin?: string;
  dateMax?: string;
  dateMedian?: string;
  dateGranularity?: string;
  rangeDays?: number;
  iqrDays?: number;
  futureDates?: number;

  // Text stats (since July 2026)
  textLengthMin?: number;
  textLengthMean?: number;
  textLengthMax?: number;
  pctContainingDigit?: number;

  // Tick data (parsed)
  xTicks?: string[];
  xTicksNumeric?: number[];
  yTicks?: number[];
  yTickMax?: number;

  // Graph URL
  graphUrl?: string;

  // Raw JSON entry for anything we missed
  raw: Record<string, any>;
}

export interface EdaData {
  variables: EdaVariable[];
  numericVars: EdaVariable[];
  categoricalVars: EdaVariable[];
  dateVars: EdaVariable[];
  textVars: EdaVariable[];
  timePointGroups: TimePointGroup[];
}

export interface TimePointGroup {
  baseName: string;
  label: string;
  variables: { visit: string; variable: EdaVariable }[];
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/** Parse "166 (26.69%)" → { count: 166, pct: 26.69 } */
export function parseCountPct(s: string | undefined | null): { count: number; pct: number } {
  if (!s) return { count: 0, pct: 0 };
  const m = String(s).match(/^(\d+)\s*\((\d+\.?\d*)%\)/);
  if (m) return { count: parseInt(m[1], 10), pct: parseFloat(m[2]) };
  // If it's just a number
  const n = parseFloat(String(s));
  return { count: isNaN(n) ? 0 : n, pct: 0 };
}

/** A number from a raw value, or undefined for anything that is not one ("n/a", "", null). */
export function num(x: any): number | undefined {
  if (x === null || x === undefined || x === '') return undefined;
  if (typeof x === 'number') return isNaN(x) ? undefined : x;
  const n = parseFloat(String(x).replace('%', ''));
  return isNaN(n) ? undefined : n;
}

/** The first key of `keys` present in the entry: the flat file's keys follow the
 *  chart-panel wording, which changed in July 2026, so a field is read under its
 *  old name or its new one. */
export function pick(entry: Record<string, any>, keys: string[]): any {
  for (const k of keys) {
    if (entry[k] !== undefined && entry[k] !== null && entry[k] !== '') return entry[k];
  }
  return undefined;
}

/** Parse "434 of 622 rows" → { valid: 434, rows: 622 }; a bare number is a valid count. */
export function parseValidOfRows(s: any): { valid: number; rows?: number } {
  if (s === undefined || s === null) return { valid: 0 };
  const m = String(s).match(/^(\d+)\s+of\s+(\d+)/);
  if (m) return { valid: parseInt(m[1], 10), rows: parseInt(m[2], 10) };
  return { valid: num(s) ?? 0 };
}

/** Parse "101 / 160" → [101, 160]; "n/a / n/a" → [undefined, undefined]. */
export function parsePair(s: any): [number | undefined, number | undefined] {
  if (s === undefined || s === null) return [undefined, undefined];
  const parts = String(s).split('/').map(x => num(x.trim()));
  return [parts[0], parts[1]];
}

/**
 * Parse the class balance of the files written since July 2026:
 * "0 -> 343 (55%); 20 -> 119 (19% / 18%); ... and 3 more categories (see json output)"
 * Each entry carries the count and the share of valid values (the second share,
 * of all rows, is dropped); the trailing "... and n more" note is skipped.
 */
export function parseClassBalanceFlat(s: any): { label: string; percentage: number; count: number }[] {
  if (!s) return [];
  return String(s).split(/;\s*/).map(e => e.trim()).filter(e => e && !e.startsWith('...')).map(entry => {
    const m = entry.match(/^(.*?)\s*->\s*(\d+)\s*\(([\d.]+)%/);
    if (!m) return { label: entry, percentage: 0, count: 0 };
    return { label: m[1], count: parseInt(m[2], 10), percentage: parseFloat(m[3]) || 0 };
  });
}

/** Parse "8 (1.29%)" → { count: 8, pct: 1.29 } (for outliers) */
export function parseOutliers(s: string | undefined | null): { count: number; pct: number } {
  return parseCountPct(s);
}

/**
 * Parse class balance string:
 * "<na> -> 99.84%\n\t33 -> 0.16%"
 * Returns array of { label, percentage }.
 *
 * The percentages (the '<na>' bucket included) are computed over the value of
 * 'count of observations (ex. missing/empty)' - which, despite its label,
 * holds the dataset's TOTAL row count in v1 EDA outputs (verified against the
 * rendered graphs: empty-row share == the '<na>' percentage of that count).
 * Pass exactly that field here. (Counts are derived from 2-decimal
 * percentages: approximate.)
 */
export function parseClassBalance(
  s: string | undefined | null,
  totalRows: number
): { label: string; percentage: number; count: number }[] {
  if (!s) return [];
  const entries = String(s)
    .split(/\n\t?/)
    .map(e => e.trim())
    .filter(Boolean);

  return entries.map(entry => {
    const parts = entry.split('->').map(p => p.trim());
    if (parts.length < 2) return { label: entry, percentage: 0, count: 0 };
    const label = parts[0];
    const pctStr = parts[1].replace('%', '');
    const percentage = parseFloat(pctStr) || 0;
    const count = Math.round((percentage / 100) * totalRows);
    return { label, percentage, count };
  });
}

/**
 * Parse x-ticks string:
 * "text(2.5, 0, '2.5') - text(5.0, 0, '5.0')"
 * Returns array of tick label strings.
 */
export function parseXTicks(s: string | undefined | null): string[] {
  if (!s) return [];
  const matches = String(s).matchAll(/text\([^)]*,\s*[^)]*,\s*'([^']*)'\)/g);
  return Array.from(matches, m => m[1]);
}

/**
 * Parse y-ticks string:
 * "text(0, 0.0, '0') - text(0, 20.0, '20')"
 * Returns array of numeric tick values.
 */
export function parseYTicks(s: string | undefined | null): number[] {
  if (!s) return [];
  const matches = String(s).matchAll(/text\([^,]*,\s*([^,)]+)/g);
  return Array.from(matches, m => parseFloat(m[1])).filter(n => !isNaN(n));
}

/**
 * Determine variable type from the raw "type" field.
 * Examples: "numeric (encoded as int64)", "categorical (encoded as object)", "datetime ..."
 */
export function parseVarType(rawType: string | undefined): EdaVarType {
  if (!rawType) return 'unknown';
  const lower = rawType.toLowerCase();
  if (lower.startsWith('numeric')) return 'numeric';
  if (lower.startsWith('categorical')) return 'categorical';
  if (lower.startsWith('date') || lower.includes('datetime')) return 'date';
  if (lower.startsWith('text')) return 'text';
  return 'unknown';
}

/**
 * Parse normality test string: "p-value=0.0000 => non-normal"
 * Returns { isNormal, raw }
 */
export function parseNormality(s: string | undefined | null): { isNormal: boolean; raw: string } {
  if (!s) return { isNormal: false, raw: '' };
  const str = String(s);
  const isNormal = str.toLowerCase().includes('=> normal') && !str.toLowerCase().includes('non-normal');
  return { isNormal, raw: str };
}

// ---------------------------------------------------------------------------
// Main parser: convert raw EDA JSON object → structured EdaData
// ---------------------------------------------------------------------------

// One v2 entry (from eda_output_v2_<id>.json) normalized to the EdaVariable
// shape the dashboard and the detail modal were built on.
function parseV2Entry(varName: string, entry: Record<string, any>): EdaVariable {
  const meta = entry['metadata'] || {};
  const comp = entry['completeness'] || {};
  const n = Number(entry['n']) || 0;
  const nRows = Number(comp['n_rows']) || n + (Number(comp['n_empty']) || 0) + (Number(comp['n_coded_missing']) || 0);
  const empty = Number(comp['n_empty']) || 0;
  const missing = Number(comp['n_coded_missing']) || 0;
  const type: EdaVarType = entry['type'] === 'numeric' ? 'numeric' : entry['type'] === 'categorical' ? 'categorical'
    : entry['type'] === 'text' ? 'text' : parseVarType(entry['type']);
  const v: EdaVariable = {
    name: varName,
    label: entry['label'] || meta['label'] || varName,
    type,
    rawType: String(meta['var_type'] || entry['type'] || ''),
    metadataLabel: meta['label'] || undefined,
    metadataVarType: meta['var_type'] || undefined,
    units: meta['units'] || undefined,
    conceptCode: meta['concept_code'] || undefined,
    conceptName: meta['concept_name'] || undefined,
    omopId: meta['omop_id'] || undefined,
    domain: meta['domain'] || undefined,
    visit: meta['visits'] || undefined,
    totalObservations: n,
    totalRows: nRows || undefined,
    countEmpty: empty,
    countEmptyPct: nRows ? (100 * empty) / nRows : 0,
    countMissing: missing,
    countMissingPct: nRows ? (100 * missing) / nRows : 0,
    countMissingTotal: empty + missing,
    countMissingTotalPct: nRows ? (100 * (empty + missing)) / nRows : 0,
    countInvalid: num(comp['n_invalid']),
    countInvalidPct: num(comp['pct_invalid']),
    missingCodes: Array.isArray(comp['missing_codes_declared']) && comp['missing_codes_declared'].length ? comp['missing_codes_declared'].join(', ') : undefined,
    uniqueValues: Number(entry['n_unique']) || 0,
    graphUrl: entry['graph_url'],
    raw: entry
  };
  if (type === 'numeric') {
    v.mean = entry['mean'];
    v.median = entry['median'];
    v.stdDev = entry['std'];
    v.variance = entry['variance'];
    v.min = entry['min'];
    v.max = entry['max'];
    v.range = entry['range'];
    v.q1 = entry['q1'];
    v.q3 = entry['q3'];
    v.iqr = entry['iqr'];
    v.skewness = entry['skewness'];
    v.kurtosis = entry['kurtosis'];
    v.p5 = num(entry['p5']);
    v.p95 = num(entry['p95']);
    v.trimmedMean10 = num(entry['trimmed_mean_10']);
    v.cv = num(entry['cv']);
    v.mad = num(entry['mad']);
    const zf = num(entry['zero_fraction']);
    v.zeroFractionPct = zf === undefined ? undefined : zf * 100;
    const norm = entry['normality'] || {};
    const pNorm = num(norm['dagostino_p']) ?? num(norm['shapiro_p']);
    v.wTest = num(norm['shapiro_w']);
    if (pNorm !== undefined) {
      v.normalityPValue = pNorm;
      v.isNormal = norm['is_normal'] !== undefined && norm['is_normal'] !== null ? !!norm['is_normal'] : pNorm > 0.05;
      v.normalityTest = `${norm['dagostino_p'] !== undefined ? "D'Agostino" : 'Shapiro'} p=${pNorm}`;
    }
    const oi = entry['outliers_iqr'];
    if (oi && typeof oi === 'object') {
      v.outliersIqr = num(oi['count']) ?? num(oi['n']) ?? 0;
      v.outliersIqrPct = num(oi['pct']) ?? (n ? (100 * v.outliersIqr) / n : 0);
    }
    const oz = entry['outliers_z'];
    if (oz && typeof oz === 'object') {
      v.outliersZ = num(oz['count']) ?? num(oz['n']) ?? 0;
      v.outliersZPct = num(oz['pct']);
    }
    const om = entry['outliers_mad'];
    if (om && typeof om === 'object') {
      v.outliersMad = num(om['count']) ?? 0;
      v.outliersMadPct = num(om['pct']);
    }
  }
  if (type === 'categorical') {
    const dist = Array.isArray(entry['distribution']) ? entry['distribution'] : [];
    v.classBalance = dist.map((d: any) => ({
      label: String(d?.label ?? d?.value ?? ''),
      percentage: Number(d?.pct) || 0,
      count: Number(d?.count) || 0
    }));
    if (v.classBalance && v.classBalance.length > 0) v.mostFrequentCategory = v.classBalance[0].label;
    v.entropyBits = num(entry['entropy_bits']);
    v.normalizedEntropy = num(entry['normalized_entropy']);
    v.giniImpurity = num(entry['gini_impurity']);
    v.imbalanceRatio = num(entry['imbalance_ratio']);
    const chi = entry['chi_square_uniform'];
    if (chi && typeof chi === 'object') {
      v.chiSquare = num(chi['statistic']);
      v.chiSquarePValue = num(chi['p_value']);
      v.chiSquareDof = num(chi['dof']);
    }
  }
  if (type === 'date') {
    v.dateMin = entry['min'] ? String(entry['min']) : undefined;
    v.dateMax = entry['max'] ? String(entry['max']) : undefined;
    v.dateMedian = entry['median'] ? String(entry['median']) : undefined;
    v.dateGranularity = entry['granularity'] ? String(entry['granularity']) : undefined;
    v.rangeDays = num(entry['range_days']);
    v.iqrDays = num(entry['iqr_days']);
    v.futureDates = num(entry['future_dates']);
  }
  if (type === 'text') {
    const len = entry['length'] || {};
    v.textLengthMin = num(len['min']);
    v.textLengthMean = num(len['mean']);
    v.textLengthMax = num(len['max']);
    v.pctContainingDigit = num(entry['pct_containing_digit']);
  }
  return v;
}

export function parseEdaJson(raw: Record<string, any>): EdaData {
  const variables: EdaVariable[] = [];

  // v2 shape: {schema_version, variables: {name: entry}, ...}; v1 is the flat
  // {name: entry} dict.
  if (raw && typeof raw === 'object' && raw['variables'] && typeof raw['variables'] === 'object' && !Array.isArray(raw['variables'])) {
    for (const [varName, entry] of Object.entries(raw['variables'] as Record<string, any>)) {
      if (!entry || typeof entry !== 'object') continue;
      variables.push(parseV2Entry(varName, entry));
    }
    const numericVars = variables.filter(v => v.type === 'numeric');
    const categoricalVars = variables.filter(v => v.type === 'categorical');
    const dateVars = variables.filter(v => v.type === 'date');
    const textVars = variables.filter(v => v.type === 'text');
    return { variables, numericVars, categoricalVars, dateVars, textVars, timePointGroups: detectTimePointGroups(variables) };
  }

  for (const [varName, entry] of Object.entries(raw)) {
    if (!entry || typeof entry !== 'object') continue;

    // Keys are the chart-panel wording: the pre-July-2026 file (first name) and
    // the file written since (second name) differ for the counts and a few stats.
    const varType = parseVarType(entry['type']);
    const validOf = parseValidOfRows(pick(entry, ['count of observations (ex. missing/empty)', 'count of valid observations']));
    const totalObs = validOf.valid;
    const { count: emptyCount, pct: emptyPct } = parseCountPct(pick(entry, ['count empty', 'count empty (blank cells, % of all rows)']));
    const { count: missingCount, pct: missingPct } = parseCountPct(pick(entry, ['count missing', 'count coded missing (% of all rows)']));
    const missingTotal = entry['count missing total (% of all rows)'] !== undefined ? parseCountPct(entry['count missing total (% of all rows)']) : undefined;
    const invalid = entry['count invalid (% of all rows)'] !== undefined ? parseCountPct(entry['count invalid (% of all rows)']) : undefined;
    const normality = parseNormality(entry['normality test']);
    const normalityP = num(entry["normality (d'agostino) p-value"]);

    const v: EdaVariable = {
      name: varName,
      label: entry['label'] || entry['variablelabel (metadata dictionary)'] || varName,
      type: varType,
      rawType: entry['type'] || '',

      metadataLabel: entry['variablelabel (metadata dictionary)'],
      metadataVarType: entry['vartype (metadata dictionary)'],
      units: entry['units (metadata dictionary)'],
      conceptCode: entry['variable concept code (metadata dictionary)'],
      conceptName: entry['variable concept name (metadata dictionary)'],
      omopId: entry['variable omop id (metadata dictionary)'],
      domain: entry['domain (metadata dictionary)'],
      visit: entry['visits (metadata dictionary)'],
      visitConceptName: entry['visit concept name (metadata dictionary)'],

      totalObservations: totalObs,
      totalRows: validOf.rows,
      countEmpty: emptyCount,
      countEmptyPct: emptyPct,
      countMissing: missingCount,
      countMissingPct: missingPct,
      countMissingTotal: missingTotal?.count,
      countMissingTotalPct: missingTotal?.pct,
      countInvalid: invalid?.count,
      countInvalidPct: invalid?.pct,
      missingCodes: (() => { const c = pick(entry, ['missing code(s) declared', 'code for missing value']); return c && String(c) !== 'none' ? String(c) : undefined; })(),
      uniqueValues: num(pick(entry, ['number of unique values/categories', 'number of unique categories', 'number of unique values'])) ?? 0,

      graphUrl: entry['url'],
      xTicks: parseXTicks(entry['x-ticks']),
      xTicksNumeric: parseXTicks(entry['x-ticks']).map(Number).filter(n => !isNaN(n)),
      yTicks: parseYTicks(entry['y-ticks']),
      yTickMax: (() => { const yt = parseYTicks(entry['y-ticks']); return yt.length > 0 ? Math.max(...yt) : undefined; })(),

      raw: entry,
    };

    // Numeric-specific fields
    if (varType === 'numeric') {
      v.mean = entry['mean'];
      v.median = entry['median'];
      v.mode = entry['mode'];
      v.stdDev = entry['std dev'];
      v.variance = entry['variance'];
      v.min = entry['min'];
      v.max = entry['max'];
      v.range = entry['range'];
      v.q1 = entry['q1'];
      v.q3 = entry['q3'];
      v.iqr = entry['iqr'];
      v.skewness = entry['skewness'];
      v.kurtosis = entry['kurtosis'];
      v.normalityTest = normality.raw;
      // D'Agostino's p-value (files since July 2026) decides when present; else the Shapiro line.
      v.normalityPValue = normalityP;
      v.isNormal = normalityP !== undefined ? normalityP > 0.05 : normality.isNormal;
      v.wTest = num(entry['w_test']);
      const iqrOutliers = parseOutliers(pick(entry, ['outliers (iqr)', 'outliers (iqr, % of valid)']));
      v.outliersIqr = iqrOutliers.count;
      v.outliersIqrPct = iqrOutliers.pct;
      const zOutliers = parseOutliers(entry['outliers (z)']);
      v.outliersZ = zOutliers.count;
      v.outliersZPct = zOutliers.pct || undefined;
      if (entry['outliers (mad, % of valid)'] !== undefined) {
        const madOutliers = parseOutliers(entry['outliers (mad, % of valid)']);
        v.outliersMad = madOutliers.count;
        v.outliersMadPct = madOutliers.pct;
      }
      v.trimmedMean10 = num(entry['trimmed mean (10%)']);
      v.cv = num(entry['cv']);
      v.mad = num(entry['mad']);
      v.zeroFractionPct = num(entry['zero fraction (% of valid)']);
      [v.p5, v.p95] = parsePair(entry['p5 / p95']);
    }

    // Categorical-specific fields
    if (varType === 'categorical') {
      const flat = entry['class balance (n, % of valid / % of all rows)'];
      v.classBalance = flat !== undefined ? parseClassBalanceFlat(flat) : parseClassBalance(entry['class balance'], totalObs);
      v.mostFrequentCategory = entry['most frequent category'];
      if (entry['chi-square vs uniform'] !== undefined) {
        // "stat=961.62, df=6, p=0.0000" (possibly followed by a note)
        const m = String(entry['chi-square vs uniform']).match(/stat=([\d.e+-]+|n\/a).*?df=([\d]+|n\/a).*?p=([\d.e+-]+|n\/a)/);
        if (m) {
          v.chiSquare = num(m[1]);
          v.chiSquareDof = num(m[2]);
          v.chiSquarePValue = num(m[3]);
        }
      } else {
        v.chiSquare = num(entry['chi-square test statistic']);
      }
      v.entropyBits = num(entry['entropy (bits)']);
      v.normalizedEntropy = num(entry['normalized entropy']);
      v.giniImpurity = num(entry['gini impurity']);
      v.imbalanceRatio = num(entry['imbalance ratio']);
    }

    // Date-specific fields (files since July 2026 print the dates as text)
    if (varType === 'date') {
      v.dateMin = entry['min'] !== undefined ? String(entry['min']) : undefined;
      v.dateMax = entry['max'] !== undefined ? String(entry['max']) : undefined;
      v.dateMedian = entry['median'] !== undefined ? String(entry['median']) : undefined;
      v.dateGranularity = entry['value granularity'] !== undefined ? String(entry['value granularity']) : undefined;
      v.rangeDays = num(entry['range (days)']);
      v.iqrDays = num(entry['iqr (days)']);
      v.futureDates = num(entry['dates in the future']);
      v.uniqueValues = num(pick(entry, ['number of unique values', 'number of unique values/categories'])) ?? v.uniqueValues;
    }

    // Text-specific fields (a type that exists since July 2026)
    if (varType === 'text') {
      const [lmin, lmean, lmax] = String(entry['value length (min / mean / max)'] ?? '').split('/').map(x => num(x.trim()));
      v.textLengthMin = lmin;
      v.textLengthMean = lmean;
      v.textLengthMax = lmax;
      v.pctContainingDigit = num(entry['values containing a digit (% of valid)']);
      v.uniqueValues = num(entry['number of unique values']) ?? v.uniqueValues;
    }

    variables.push(v);
  }

  const numericVars = variables.filter(v => v.type === 'numeric');
  const categoricalVars = variables.filter(v => v.type === 'categorical');
  const dateVars = variables.filter(v => v.type === 'date');
  const textVars = variables.filter(v => v.type === 'text');

  // Detect time-point groups
  const timePointGroups = detectTimePointGroups(variables);

  return { variables, numericVars, categoricalVars, dateVars, textVars, timePointGroups };
}

// ---------------------------------------------------------------------------
// Time-point group detection
// ---------------------------------------------------------------------------

/**
 * Detect groups of variables that represent the same measurement at different
 * time points (e.g., HWdiast, HWdiast_12, HWdiast_18).
 *
 * Heuristic: strip common suffixes like _6, _12, _18, _BL and group by base name.
 * Only keep groups with 2+ members.
 */
export function detectTimePointGroups(variables: EdaVariable[]): TimePointGroup[] {
  const VISIT_SUFFIX_RE = /^(.+?)(?:_(BL|\d+))?$/i;
  const groups: Record<string, { visit: string; variable: EdaVariable }[]> = {};

  for (const v of variables) {
    const match = v.name.match(VISIT_SUFFIX_RE);
    if (!match) continue;
    const baseName = match[1];
    const suffix = match[2] || 'baseline';
    const visitLabel = v.visit || (suffix === 'baseline' ? 'Baseline' : `Month ${suffix}`);

    if (!groups[baseName]) groups[baseName] = [];
    groups[baseName].push({ visit: visitLabel, variable: v });
  }

  return Object.entries(groups)
    .filter(([, members]) => members.length >= 2)
    .map(([baseName, members]) => ({
      baseName,
      label: members[0].variable.label.replace(/\s+at\s+(baseline|visit\d+)/i, ''),
      variables: members.sort((a, b) => {
        // Sort: baseline first, then by numeric suffix
        const aNum = a.variable.name.match(/_(\d+)$/);
        const bNum = b.variable.name.match(/_(\d+)$/);
        if (!aNum && bNum) return -1;
        if (aNum && !bNum) return 1;
        if (aNum && bNum) return parseInt(aNum[1]) - parseInt(bNum[1]);
        return 0;
      }),
    }));
}

// ---------------------------------------------------------------------------
// Utility: compute completeness score
// ---------------------------------------------------------------------------
export function completenessScore(v: EdaVariable): number {
  if (v.totalObservations === 0) return 0;
  return 100 - v.countEmptyPct - v.countMissingPct;
}
