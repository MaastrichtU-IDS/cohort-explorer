import React from 'react';
import { parseAgeRange } from '@/utils/parseAgeRange';

interface AgeRangeWhiskerProps {
  /** Raw "age group inclusion criterion" string straight from the spreadsheet. */
  ageRangeRaw: string | null | undefined;
  width?: number;
  height?: number;
  axisMin?: number;
  axisMax?: number;
}

/**
 * Compact (~60x24 px) boxplot-style glyph that visualises the age range of a
 * cohort on a fixed 0-100 axis so cohorts can be compared at a glance.
 *
 * Rendered elements:
 *   - Medium-grey background rail spanning the full [axisMin, axisMax] range.
 *   - Blue whisker spanning [min, max] of the cohort's reported age range.
 *   - Vertical end-caps only on explicit bounds (e.g. "<= 84 and >= 65").
 *     When a side is unbounded (e.g. ">= 18 years old") we draw no marker at
 *     all -- the rail continuing past the whisker visually conveys "no upper
 *     limit" without cluttering the glyph.
 *   - A small dark-blue dot at the median when it is known (only produced by
 *     the "X +/- Y" form; explicit ranges leave the dot out).
 *
 * Tooltip: handled by the parent wrapper in cohorts.tsx so that gender + age
 * glyphs share a single combined tooltip. No `title` attribute is set here.
 *
 * Returns null when parseAgeRange cannot extract any numeric information --
 * matching the behaviour of GenderPieChart / AgeDistributionBar so empty cells
 * simply render nothing instead of showing a misleading empty glyph.
 */
const AgeRangeWhisker: React.FC<AgeRangeWhiskerProps> = ({
  ageRangeRaw,
  width = 60,
  height = 24,
  axisMin = 0,
  axisMax = 100,
}) => {
  const parsed = parseAgeRange(ageRangeRaw);
  if (!parsed) return null;

  const { min, max, median } = parsed;
  const pad = 4;
  const innerW = width - pad * 2;
  const yMid = height / 2;

  const xFromAge = (age: number): number => {
    const clamped = Math.max(axisMin, Math.min(axisMax, age));
    return pad + ((clamped - axisMin) / (axisMax - axisMin)) * innerW;
  };

  const hasMin = min !== undefined;
  const hasMax = max !== undefined;
  // When a bound is open, anchor the whisker to the padded edge so the
  // rail continues past the coloured segment without any end marker.
  const x1 = hasMin ? xFromAge(min as number) : pad;
  const x2 = hasMax ? xFromAge(max as number) : width - pad;

  // Rail uses a mid-grey so the axis is legible against both white and the
  // zebra-striped cohort list background. It was previously gray-200
  // (#e5e7eb) which blended into the row backgrounds.
  const railColour = '#9ca3af';
  const barColour = '#2563eb';
  const medianColour = '#1e3a8a';

  return (
    <div className="inline-flex items-center" style={{ width, height }}>
      <svg width={width} height={height} aria-hidden="true">
        {/* Background rail */}
        <line
          x1={pad}
          x2={width - pad}
          y1={yMid}
          y2={yMid}
          stroke={railColour}
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        {/* Whisker body */}
        <line
          x1={x1}
          x2={x2}
          y1={yMid}
          y2={yMid}
          stroke={barColour}
          strokeWidth={3}
          strokeLinecap="round"
        />
        {/* Explicit-bound end caps -- only rendered on the side(s) where the
            spreadsheet gave an explicit numeric bound. Open sides are left
            unmarked so the continuing rail signals "no stated limit". */}
        {hasMin && (
          <line
            x1={x1}
            x2={x1}
            y1={yMid - 5}
            y2={yMid + 5}
            stroke={barColour}
            strokeWidth={1.5}
            strokeLinecap="round"
          />
        )}
        {hasMax && (
          <line
            x1={x2}
            x2={x2}
            y1={yMid - 5}
            y2={yMid + 5}
            stroke={barColour}
            strokeWidth={1.5}
            strokeLinecap="round"
          />
        )}
        {/* Median dot (only present for "X +/- Y" form) */}
        {median !== undefined && (
          <circle
            cx={xFromAge(median)}
            cy={yMid}
            r={2.5}
            fill={medianColour}
          />
        )}
      </svg>
    </div>
  );
};

export default AgeRangeWhisker;
