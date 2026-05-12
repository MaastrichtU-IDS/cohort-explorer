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

export interface EdaVariable {
  name: string;
  label: string;
  type: 'numeric' | 'categorical' | 'date' | 'unknown';
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

  // Observation counts
  totalObservations: number;
  countEmpty: number;
  countEmptyPct: number;
  countMissing: number;
  countMissingPct: number;
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

  // Categorical stats
  classBalance?: { label: string; percentage: number; count: number }[];
  chiSquare?: number;
  mostFrequentCategory?: string;

  // Tick data (parsed)
  xTicks?: string[];
  yTicks?: number[];

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

/** Parse "8 (1.29%)" → { count: 8, pct: 1.29 } (for outliers) */
export function parseOutliers(s: string | undefined | null): { count: number; pct: number } {
  return parseCountPct(s);
}

/**
 * Parse class balance string:
 * "<na> -> 99.84%\n\t33 -> 0.16%"
 * Returns array of { label, percentage }.
 */
export function parseClassBalance(
  s: string | undefined | null,
  totalObs: number
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
    const count = Math.round((percentage / 100) * totalObs);
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
export function parseVarType(rawType: string | undefined): 'numeric' | 'categorical' | 'date' | 'unknown' {
  if (!rawType) return 'unknown';
  const lower = rawType.toLowerCase();
  if (lower.startsWith('numeric')) return 'numeric';
  if (lower.startsWith('categorical')) return 'categorical';
  if (lower.startsWith('date') || lower.includes('datetime')) return 'date';
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

export function parseEdaJson(raw: Record<string, any>): EdaData {
  const variables: EdaVariable[] = [];

  for (const [varName, entry] of Object.entries(raw)) {
    if (!entry || typeof entry !== 'object') continue;

    const varType = parseVarType(entry['type']);
    const totalObs = Number(entry['count of observations (ex. missing/empty)']) || 0;
    const { count: emptyCount, pct: emptyPct } = parseCountPct(entry['count empty']);
    const { count: missingCount, pct: missingPct } = parseCountPct(entry['count missing']);
    const normality = parseNormality(entry['normality test']);

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
      countEmpty: emptyCount,
      countEmptyPct: emptyPct,
      countMissing: missingCount,
      countMissingPct: missingPct,
      uniqueValues: Number(entry['number of unique values/categories']) || 0,

      graphUrl: entry['url'],
      xTicks: parseXTicks(entry['x-ticks']),
      yTicks: parseYTicks(entry['y-ticks']),

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
      v.isNormal = normality.isNormal;
      v.wTest = entry['w_test'];
      const iqrOutliers = parseOutliers(entry['outliers (iqr)']);
      v.outliersIqr = iqrOutliers.count;
      v.outliersIqrPct = iqrOutliers.pct;
      v.outliersZ = typeof entry['outliers (z)'] === 'number' ? entry['outliers (z)'] : parseFloat(entry['outliers (z)']);
    }

    // Categorical-specific fields
    if (varType === 'categorical') {
      v.classBalance = parseClassBalance(entry['class balance'], totalObs);
      v.chiSquare = entry['chi-square test statistic'];
      v.mostFrequentCategory = entry['most frequent category'];
    }

    variables.push(v);
  }

  const numericVars = variables.filter(v => v.type === 'numeric');
  const categoricalVars = variables.filter(v => v.type === 'categorical');
  const dateVars = variables.filter(v => v.type === 'date');

  // Detect time-point groups
  const timePointGroups = detectTimePointGroups(variables);

  return { variables, numericVars, categoricalVars, dateVars, timePointGroups };
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
