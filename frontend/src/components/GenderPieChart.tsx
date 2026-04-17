import React from 'react';

interface GenderPieChartProps {
  malePercentage: number | null;
  femalePercentage: number | null;
  size?: number;
}

const GenderPieChart: React.FC<GenderPieChartProps> = ({ 
  malePercentage, 
  femalePercentage,
  size = 40 
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

  return (
    <div className="flex flex-col items-center gap-1">
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
      <div className="text-xs text-gray-600 dark:text-gray-400 flex gap-2">
        <span className="flex items-center gap-0.5">
          <span style={{ color: '#3b82f6' }}>M</span>
          {malePercentage}%
        </span>
        <span className="flex items-center gap-0.5">
          <span style={{ color: '#ec4899' }}>F</span>
          {femalePercentage}%
        </span>
      </div>
    </div>
  );
};

export default GenderPieChart;
