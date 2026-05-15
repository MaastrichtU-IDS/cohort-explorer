import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable, TimePointGroup } from '@/utils/edaParsing';
import DomainFilterBar from './DomainFilterBar';

interface Props {
  variables: EdaVariable[];
  timePointGroups: TimePointGroup[];
  onVariableClick: (v: EdaVariable) => void;
}

type ColorMode = 'type' | 'longitudinal';

const EdaSkewnessIqrScatter: React.FC<Props> = ({ variables, timePointGroups, onVariableClick }) => {
  const [colorMode, setColorMode] = useState<ColorMode>('type');
  const [showLongitudinalLines, setShowLongitudinalLines] = useState(true);
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);

  // Filter to numeric vars with both skewness and iqr
  const eligible = useMemo(() => {
    let list = variables.filter(v => v.type === 'numeric' && v.skewness !== undefined && v.iqr !== undefined);
    if (selectedDomain) list = list.filter(v => v.domain === selectedDomain);
    return list;
  }, [variables, selectedDomain]);

  // Build a map of variable -> longitudinal group for coloring
  const varToGroup = useMemo(() => {
    const map = new Map<string, string>();
    for (const g of timePointGroups) {
      for (const m of g.variables) {
        map.set(m.variable.name, g.baseName);
      }
    }
    return map;
  }, [timePointGroups]);

  // Unique groups for color assignment
  const uniqueGroups = useMemo(() => {
    const groups = new Set<string>();
    for (const v of eligible) {
      const g = varToGroup.get(v.name);
      if (g) groups.add(g);
    }
    return Array.from(groups);
  }, [eligible, varToGroup]);

  const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'];

  const scatterOption = useMemo(() => {
    if (eligible.length === 0) return null;

    // Build series based on color mode
    if (colorMode === 'type') {
      // Single series, all numeric vars
      const data = eligible.map(v => ({
        value: [v.iqr!, v.skewness!],
        name: v.name,
        variable: v,
      }));

      return {
        tooltip: {
          trigger: 'item' as const,
          formatter: (p: any) => {
            const v = p.data.variable as EdaVariable;
            const group = varToGroup.get(v.name);
            return `<strong>${v.name}</strong>${group ? ` (${group})` : ''}<br/>` +
              `${v.label}<br/>` +
              `IQR: ${v.iqr} · Skewness: ${v.skewness?.toFixed(3)}<br/>` +
              `Mean: ${v.mean?.toFixed(2)} · Median: ${v.median}<br/>` +
              `n=${v.totalObservations}`;
          },
        },
        grid: { left: 70, right: 30, top: 50, bottom: 60 },
        xAxis: {
          type: 'value' as const,
          name: 'IQR (Interquartile Range)',
          nameLocation: 'center' as const,
          nameGap: 40,
          axisLabel: { fontSize: 10 },
        },
        yAxis: {
          type: 'value' as const,
          name: 'Skewness',
          nameLocation: 'center' as const,
          nameGap: 50,
          axisLabel: { fontSize: 10 },
        },
        series: [{
          type: 'scatter' as const,
          data,
          symbolSize: 10,
          itemStyle: { color: '#3b82f6', opacity: 0.7 },
          emphasis: {
            itemStyle: { borderColor: '#000', borderWidth: 2, shadowBlur: 4 },
          },
        }],
      };
    }

    // Longitudinal mode: one series per group, with connecting lines
    const nonGrouped = eligible.filter(v => !varToGroup.has(v.name));
    const series: any[] = [];

    // Non-grouped variables
    if (nonGrouped.length > 0) {
      series.push({
        name: 'Non-longitudinal',
        type: 'scatter' as const,
        data: nonGrouped.map(v => ({
          value: [v.iqr!, v.skewness!],
          name: v.name,
          variable: v,
        })),
        symbolSize: 8,
        itemStyle: { color: '#94a3b8', opacity: 0.5 },
      });
    }

    // Grouped variables
    uniqueGroups.forEach((groupName, gi) => {
      const group = timePointGroups.find(g => g.baseName === groupName);
      if (!group) return;

      const members = group.variables
        .filter(m => m.variable.skewness !== undefined && m.variable.iqr !== undefined)
        .sort((a, b) => {
          // Sort by visit time
          const aOrder = visitSortOrder(a.visit);
          const bOrder = visitSortOrder(b.visit);
          return aOrder - bOrder;
        });

      if (members.length === 0) return;

      const color = COLORS[gi % COLORS.length];
      const data = members.map(m => ({
        value: [m.variable.iqr!, m.variable.skewness!],
        name: m.variable.name,
        variable: m.variable,
      }));

      series.push({
        name: groupName,
        type: 'scatter' as const,
        data,
        symbolSize: 10,
        itemStyle: { color, opacity: 0.85 },
        emphasis: {
          itemStyle: { borderColor: '#000', borderWidth: 2 },
        },
      });

      // Connecting line (trajectory)
      if (showLongitudinalLines && members.length >= 2) {
        series.push({
          name: `${groupName} (trend)`,
          type: 'line' as const,
          data: data.map(d => d.value),
          lineStyle: { color, width: 1.5, type: 'dashed' as const, opacity: 0.6 },
          symbol: 'none',
          silent: true,
          z: 1,
        });
      }
    });

    return {
      tooltip: {
        trigger: 'item' as const,
        formatter: (p: any) => {
          if (!p.data?.variable) return '';
          const v = p.data.variable as EdaVariable;
          const group = varToGroup.get(v.name);
          return `<strong>${v.name}</strong>${group ? ` (${group})` : ''}<br/>` +
            `${v.label}<br/>` +
            `IQR: ${v.iqr} · Skewness: ${v.skewness?.toFixed(3)}<br/>` +
            `Mean: ${v.mean?.toFixed(2)} · Median: ${v.median}<br/>` +
            `Visit: ${v.visit || '—'} · n=${v.totalObservations}`;
        },
      },
      legend: {
        type: 'scroll' as const,
        top: 0,
        textStyle: { fontSize: 9 },
        data: series.filter(s => !s.name.includes('(trend)')).map(s => s.name),
      },
      grid: { left: 70, right: 30, top: 50, bottom: 60 },
      xAxis: {
        type: 'value' as const,
        name: 'IQR (Interquartile Range)',
        nameLocation: 'center' as const,
        nameGap: 40,
        axisLabel: { fontSize: 10 },
      },
      yAxis: {
        type: 'value' as const,
        name: 'Skewness',
        nameLocation: 'center' as const,
        nameGap: 50,
        axisLabel: { fontSize: 10 },
      },
      series,
    };
  }, [eligible, colorMode, varToGroup, uniqueGroups, timePointGroups, showLongitudinalLines]);

  if (eligible.length === 0) {
    return (
      <div className="alert alert-info max-w-xl mx-auto my-8">
        <span>No numeric variables with both skewness and IQR data available.</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-lg font-bold">Skewness vs IQR</h2>
        <p className="text-sm text-gray-500 mt-1">
          Each point is a numeric variable. X = spread (IQR), Y = asymmetry (skewness).
          {colorMode === 'longitudinal' && ' Dashed lines connect time points of the same variable.'}
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-center gap-3">
        <div className="join">
          <button
            className={`btn btn-sm join-item ${colorMode === 'type' ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => setColorMode('type')}
          >
            All Variables
          </button>
          <button
            className={`btn btn-sm join-item ${colorMode === 'longitudinal' ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => setColorMode('longitudinal')}
          >
            By Longitudinal Group
          </button>
        </div>
        {colorMode === 'longitudinal' && (
          <label className="label cursor-pointer gap-2">
            <input
              type="checkbox"
              className="checkbox checkbox-sm"
              checked={showLongitudinalLines}
              onChange={e => setShowLongitudinalLines(e.target.checked)}
            />
            <span className="label-text text-sm">Show trajectories</span>
          </label>
        )}
        <DomainFilterBar variables={variables} selectedDomain={selectedDomain} onChange={setSelectedDomain} />
        <span className="text-sm text-gray-500">{eligible.length} variables</span>
      </div>

      {/* Scatter chart */}
      {scatterOption && (
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts
            option={scatterOption}
            style={{ height: 500 }}
            onEvents={{
              click: (params: any) => {
                if (params.data?.variable) {
                  onVariableClick(params.data.variable);
                }
              },
            }}
          />
        </div>
      )}

      {/* Interpretation guide */}
      <div className="card bg-base-200 p-4 text-sm">
        <h3 className="font-semibold mb-2">How to read this chart</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div><strong>Top-right:</strong> Wide spread + right-skewed — consider log transform</div>
          <div><strong>Top-left:</strong> Tight spread + right-skewed — floor effect or discrete values</div>
          <div><strong>Bottom-right:</strong> Wide spread + left-skewed — ceiling effect</div>
          <div><strong>Center (y≈0):</strong> Symmetric distributions — likely normal-ish</div>
        </div>
      </div>
    </div>
  );
};

function visitSortOrder(visit: string): number {
  const lower = visit.toLowerCase().trim();
  if (lower.includes('prior')) return -1;
  if (lower.includes('baseline') && !lower.includes('month')) return 0;
  const m = lower.match(/month\s*(\d+)/);
  if (m) return parseInt(m[1], 10);
  if (lower.includes('end of study')) return 9999;
  return 5000;
}

export default EdaSkewnessIqrScatter;
