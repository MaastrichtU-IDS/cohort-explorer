/**
 * Parse free-text age-range strings from the iCARE4CVD Cohorts spreadsheet
 * ("age group inclusion criterion" column).
 *
 * The column is heavily free-text and carries a handful of recurring shapes:
 *   - Explicit bracketed range, e.g. "<= 84.5 and >= 69.3 years old"
 *   - Single-sided lower bound, e.g. ">= 18 years old" / ">18 years"
 *   - Single-sided upper bound, e.g. "<= 65 years old"
 *   - Median +/- delta, e.g. "55 +/- 10" or "55 ± 10"
 *   - Simple dash range, e.g. "18-65" or "18 to 65"
 *   - "Not Applicable" / empty  -> no information, return null
 *
 * Quirks handled:
 *   - Unicode comparators "≥" / "≤" are normalised to ">=" / "<=".
 *   - Unicode "±" is normalised to "+-".
 *   - Arbitrary whitespace (including double spaces like "<= 69.5 and  >= 55.1").
 *   - Strict vs non-strict comparators ('>' vs '>=') are treated identically --
 *     the visualisation is soft-edged so the difference is visually invisible.
 *
 * The returned shape:
 *   - `min` / `max` are inclusive axis bounds (in years).
 *   - `median` is populated only for the "X +/- Y" form, where it is known.
 *   - A missing `min` means the cohort has no stated lower bound
 *     (consumer typically renders an arrow off the left edge of the axis).
 *     Likewise for `max`.
 *
 * Returns null when the string is empty, "Not Applicable", or has no numeric
 * information the parser can latch onto.
 */
export interface ParsedAgeRange {
  min?: number;
  max?: number;
  median?: number;
}

export function parseAgeRange(raw: string | null | undefined): ParsedAgeRange | null {
  if (!raw) return null;
  const trimmed = String(raw).trim();
  if (!trimmed) return null;

  // Normalise: lower-case, Unicode -> ASCII, collapse whitespace.
  const normalised = trimmed
    .toLowerCase()
    .replace(/≥/g, '>=')
    .replace(/≤/g, '<=')
    .replace(/±/g, '+-')
    .replace(/\s+/g, ' ');

  if (!normalised || normalised === 'not applicable' || normalised === 'n/a' || normalised === 'na') {
    return null;
  }

  // Shape 1: median +/- delta ("55 +- 10" after normalisation).
  const pmMatch = normalised.match(/(-?\d+(?:\.\d+)?)\s*\+-\s*(-?\d+(?:\.\d+)?)/);
  if (pmMatch) {
    const median = parseFloat(pmMatch[1]);
    const delta = parseFloat(pmMatch[2]);
    if (Number.isFinite(median) && Number.isFinite(delta)) {
      return { min: median - delta, max: median + delta, median };
    }
  }

  // Shape 2: plain dash/en-dash/em-dash/"to" range ("18-65", "18 to 65"). We
  // guard with a comparator-free check so we don't mis-parse something like
  // ">= 18 - 65" -- in practice the spreadsheet never combines the two.
  if (!/[<>]/.test(normalised)) {
    const rangeMatch = normalised.match(/(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)/);
    if (rangeMatch) {
      const a = parseFloat(rangeMatch[1]);
      const b = parseFloat(rangeMatch[2]);
      if (Number.isFinite(a) && Number.isFinite(b)) {
        return { min: Math.min(a, b), max: Math.max(a, b) };
      }
    }
  }

  // Shape 3: one or more comparator clauses (">=", "<=", ">", "<"). "<=" /
  // "<" constrains the upper bound; ">=" / ">" constrains the lower bound.
  const opRegex = /(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)/g;
  const result: ParsedAgeRange = {};
  let match: RegExpExecArray | null;
  while ((match = opRegex.exec(normalised)) !== null) {
    const op = match[1];
    const num = parseFloat(match[2]);
    if (!Number.isFinite(num)) continue;
    if (op === '>=' || op === '>') {
      // Later occurrences overwrite earlier ones, which is fine since the
      // sheet never carries more than one lower bound per cell.
      result.min = num;
    } else {
      result.max = num;
    }
  }

  if (result.min === undefined && result.max === undefined) return null;
  return result;
}
