import React, { useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { Variable } from '@/types';

interface CohortSummaryGraphsProps {
  variables: { [key: string]: Variable };
}

// Color palettes for the pie charts
const DOMAIN_COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
  '#84cc16', // lime
  '#6366f1', // indigo
];

const TYPE_COLORS = [
  '#0ea5e9', // sky
  '#22c55e', // green
  '#eab308', // yellow
  '#f43f5e', // rose
  '#a855f7', // purple
  '#14b8a6', // teal
  '#fb923c', // orange
  '#4ade80', // green
  '#818cf8', // indigo
  '#fb7185', // rose
];

export default function CohortSummaryGraphs({ variables }: CohortSummaryGraphsProps) {
  // Calculate OMOP domain distribution
  const domainData = useMemo(() => {
    const domainCounts: { [key: string]: number } = {};
    
    Object.values(variables).forEach((variable) => {
      const domain = variable.omop_domain || 'Unmapped';
      domainCounts[domain] = (domainCounts[domain] || 0) + 1;
    });
    
    return Object.entries(domainCounts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value); // Sort by count descending
  }, [variables]);

  // Calculate data type distribution
  const typeData = useMemo(() => {
    const typeCounts: { [key: string]: number } = {};
    
    Object.values(variables).forEach((variable) => {
      const type = variable.var_type || 'Unknown';
      typeCounts[type] = (typeCounts[type] || 0) + 1;
    });
    
    return Object.entries(typeCounts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value); // Sort by count descending
  }, [variables]);

  // Custom label renderer to show percentage
  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * (Math.PI / 180));
    const y = cy + radius * Math.sin(-midAngle * (Math.PI / 180));

    if (percent < 0.05) return null; // Don't show label if slice is too small

    return (
      <text
        x={x}
        y={y}
        fill="white"
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
        className="text-xs font-semibold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  if (Object.keys(variables).length === 0) {
    return (
      <div className="bg-white shadow-md rounded-lg p-4 mb-4">
        <h3 className="text-lg font-semibold mb-3 border-b pb-2">Summary Graphs</h3>
        <p className="text-gray-500"><em>No variables available to display graphs</em></p>
      </div>
    );
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-4 mb-4">
      <h3 className="text-lg font-semibold mb-3 border-b pb-2">Summary Graphs</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* OMOP Domains Pie Chart */}
        <div>
          <h4 className="text-md font-medium mb-2 text-center">OMOP Domains Distribution</h4>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={domainData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={renderCustomLabel}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {domainData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={DOMAIN_COLORS[index % DOMAIN_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value: number) => [`${value} variables`, 'Count']}
              />
              <Legend 
                verticalAlign="bottom" 
                height={36}
                formatter={(value) => {
                  const item = domainData.find(d => d.name === value);
                  return `${value} (${item?.value || 0})`;
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Data Types Pie Chart */}
        <div>
          <h4 className="text-md font-medium mb-2 text-center">Data Types Distribution</h4>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={typeData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={renderCustomLabel}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {typeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={TYPE_COLORS[index % TYPE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value: number) => [`${value} variables`, 'Count']}
              />
              <Legend 
                verticalAlign="bottom" 
                height={36}
                formatter={(value) => {
                  const item = typeData.find(d => d.name === value);
                  return `${value} (${item?.value || 0})`;
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
