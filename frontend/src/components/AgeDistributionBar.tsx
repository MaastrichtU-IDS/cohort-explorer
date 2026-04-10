import React from 'react';

interface AgeDistributionBarProps {
  ageDistribution: {[key: string]: number} | undefined;
  height?: number;
}

const AgeDistributionBar: React.FC<AgeDistributionBarProps> = ({ 
  ageDistribution,
  height = 50 
}) => {
  // Don't render if we don't have age distribution data
  if (!ageDistribution || Object.keys(ageDistribution).length === 0) {
    return null;
  }

  // Sort age groups for consistent display (youngest to oldest)
  const sortedGroups = Object.entries(ageDistribution).sort((a, b) => {
    const aStart = parseInt(a[0].split('-')[0]);
    const bStart = parseInt(b[0].split('-')[0]);
    return aStart - bStart;
  });

  // Color palette for age groups (gradient from light to dark)
  const colors = [
    '#93c5fd', // light blue
    '#60a5fa', // medium blue
    '#3b82f6', // blue
    '#2563eb', // darker blue
    '#1d4ed8', // darkest blue
  ];

  const barWidth = 60;
  const barHeight = 30;

  return (
    <div className="flex flex-col items-center gap-1">
      {/* Tiny horizontal stacked bar */}
      <div 
        className="flex rounded overflow-hidden" 
        style={{ 
          width: `${barWidth}px`, 
          height: `${barHeight}px`,
          border: '1px solid #e5e7eb'
        }}
      >
        {sortedGroups.map(([ageRange, percentage], index) => (
          <div
            key={ageRange}
            style={{
              width: `${percentage}%`,
              backgroundColor: colors[index % colors.length],
              height: '100%'
            }}
            title={`${ageRange}: ${percentage}%`}
          />
        ))}
      </div>
      
      {/* Age ranges and percentages below */}
      <div className="text-xs text-gray-600 dark:text-gray-400 flex flex-col gap-0.5">
        {sortedGroups.map(([ageRange, percentage], index) => (
          <div key={ageRange} className="flex items-center gap-1">
            <span 
              className="inline-block w-2 h-2 rounded-sm" 
              style={{ backgroundColor: colors[index % colors.length] }}
            />
            <span className="whitespace-nowrap">{ageRange}: {percentage}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgeDistributionBar;
