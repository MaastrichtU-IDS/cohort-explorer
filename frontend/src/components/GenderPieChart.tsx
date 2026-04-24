import React from 'react';

interface GenderPieChartProps {
  malePercentage: number | null;
  femalePercentage: number | null;
  size?: number;
}

const GenderPieChart: React.FC<GenderPieChartProps> = ({ 
  malePercentage, 
  femalePercentage,
  size = 28 
}) => {
  // Don't render if we don't have both percentages
  if (malePercentage === null || femalePercentage === null) {
    return null;
  }

  const radius = size / 2;
  const strokeWidth = size / 5;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = normalizedRadius * 2 * Math.PI;

  // Calculate the stroke dash offset for the male percentage
  const maleOffset = circumference - (malePercentage / 100) * circumference;

  // Tooltip is intentionally omitted: cohorts.tsx wraps this glyph together
  // with the age-range whisker in a shared hover area so a single tooltip
  // can show both demographic summaries at once.
  return (
    <div className="flex items-center">
      <svg
        height={size}
        width={size}
        style={{ transform: 'rotate(-90deg)' }}
      >
        {/* Female portion (pink) - full circle as background */}
        <circle
          stroke="#ec4899"
          fill="transparent"
          strokeWidth={strokeWidth}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
        {/* Male portion (blue) - overlay on top */}
        <circle
          stroke="#3b82f6"
          fill="transparent"
          strokeWidth={strokeWidth}
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={maleOffset}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
      </svg>
    </div>
  );
};

export default GenderPieChart;
