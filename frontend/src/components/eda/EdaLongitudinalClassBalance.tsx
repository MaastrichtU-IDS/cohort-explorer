import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable, TimePointGroup } from '@/utils/edaParsing';
import DomainFilterBar from './DomainFilterBar';

interface Props {
  groups: TimePointGroup[];
  onVariableClick: (v: EdaVariable) => void;
}

/**
 * Calculate Gini coefficient for class imbalance
 * Gini = 1 - sum(p_i^2) where p_i is the proportion of each class
 * Range: 0 (perfectly balanced) to 1 (completely imbalanced)
 */
function calculateGini(classBalance: { percentage: number }[]): number {
  if (!classBalance || classBalance.length === 0) return 0;
  const sumSquares = classBalance.reduce((sum, c) => sum + Math.pow(c.percentage / 100, 2), 0);
  return 1 - sumSquares;
}

interface RankedGroup {
  baseName: string;
  label: string;
  baselineGini: number;
  latestGini: number;
  absoluteChange: number;
  percentChange: number;
  baselineVisit: string;
  latestVisit: string;
  timePoints: number;
  allValues: { visit: string; value: number; sortOrder: number }[];
  variables: { visit: string; variable: EdaVariable }[];
}

type SortKey = 'absChange' | 'pctChange' | 'name' | 'timePoints';
type ViewMode = 'table' | 'heatmap';

/** Extract a numeric sort order from a visit label like "month 12", "month 0/ baseline", "prior baseline", etc. */
function visitSortOrder(visit: string): number {
  const lower = visit.toLowerCase().trim();
  if (lower.includes('prior')) return -1;
  if (lower.includes('baseline') && !lower.includes('month')) return 0;
  const m = lower.match(/month\s*(\d+)/);
  if (m) return parseInt(m[1], 10);
  if (lower.includes('end of study')) return 9999;
  return 5000; // unknown — sort near the end
}

const EdaLongitudinalClassBalance: React.FC<Props> = ({ groups, onVariableClick }) => {
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [sortKey, setSortKey] = useState<SortKey>('absChange');
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);
  const [expandedGroup, setExpandedGroup] = useState<RankedGroup | null>(null);

  // Compute ranked groups — only categorical groups with class balance data at 2+ time points
  const ranked: RankedGroup[] = useMemo(() => {
    return groups
      .map(g => {
        const withValue = g.variables.filter(m => m.variable.classBalance && m.variable.classBalance.length > 1 && m.variable.type === 'categorical');
        if (withValue.length < 2) return null;

        // Sort members by visit time (numeric extraction)
        const sorted = [...withValue].sort((a, b) => visitSortOrder(a.visit) - visitSortOrder(b.visit));

        const allValues = sorted.map(m => ({ visit: m.visit, value: calculateGini(m.variable.classBalance || []), sortOrder: visitSortOrder(m.visit) }));
        const baseline = sorted[0];
        const latest = sorted[sorted.length - 1];

        const baselineGini = calculateGini(baseline.variable.classBalance || []);
        const latestGini = calculateGini(latest.variable.classBalance || []);
        const absoluteChange = latestGini - baselineGini;
        const percentChange = baselineGini > 0 ? (absoluteChange / baselineGini) * 100 : 0;

        return {
          baseName: g.baseName,
          label: g.label,
          baselineGini,
          latestGini,
          absoluteChange,
          percentChange,
          baselineVisit: baseline.visit,
          latestVisit: latest.visit,
          timePoints: sorted.length,
          allValues,
          variables: sorted,
        };
      })
      .filter((g): g is RankedGroup => g !== null && (selectedDomain ? g.variables[0].variable.domain === selectedDomain : true))
      .sort((a, b) => {
        switch (sortKey) {
          case 'absChange': return Math.abs(b.absoluteChange) - Math.abs(a.absoluteChange);
          case 'pctChange': return Math.abs(b.percentChange) - Math.abs(a.percentChange);
          case 'timePoints': return b.timePoints - a.timePoints;
          case 'name': return a.baseName.localeCompare(b.baseName);
        }
      });
  }, [groups, sortKey, selectedDomain]);

  // Filter by search
  const filtered = useMemo(() => {
    if (!searchFilter) return ranked;
    const lower = searchFilter.toLowerCase();
    return ranked.filter(g => 
      g.baseName.toLowerCase().includes(lower) || 
      g.label.toLowerCase().includes(lower)
    );
  }, [ranked, searchFilter]);

  // Heatmap option
  const heatmapOption = useMemo(() => {
    if (filtered.length === 0) return null;

    const display = filtered.slice(0, 50);
    const cols = 10;
    const rows = Math.ceil(display.length / cols);

    const data: [number, number, number, any][] = [];
    const xLabels: string[] = [];
    const yLabels: string[] = [];

    for (let i = 0; i < display.length; i++) {
      const g = display[i];
      const x = i % cols;
      const y = Math.floor(i / cols);
      data.push([x, y, Math.abs(g.absoluteChange), { group: g }]);
      if (x === 0) yLabels.push(`Row ${y + 1}`);
    }
    for (let i = 0; i < cols; i++) xLabels.push(`Col ${i + 1}`);

    return {
      title: { text: 'Change in Class Balance Heatmap', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        formatter: (params: any) => {
          const g = params.data[3].group;
          return `<strong>${g.baseName}</strong><br/>Label: ${g.label}<br/>Baseline: ${g.baselineGini.toFixed(3)} (${g.baselineVisit})<br/>Latest: ${g.latestGini.toFixed(3)} (${g.latestVisit})<br/>Abs Change: ${g.absoluteChange.toFixed(3)}<br/>% Change: ${g.percentChange.toFixed(1)}%<br/>Time Points: ${g.timePoints}`;
        },
      },
      grid: { left: 60, right: 20, top: 40, bottom: 40 },
      xAxis: {
        type: 'category' as const,
        data: xLabels,
        show: false,
      },
      yAxis: {
        type: 'category' as const,
        data: yLabels,
        show: false,
      },
      visualMap: {
        min: 0,
        max: Math.max(...filtered.map(g => Math.abs(g.absoluteChange))),
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 10,
        dimension: 2,
        inRange: {
          color: ['#90EE90', '#FFD700', '#FF6347', '#8B0000'],
        },
      },
      series: [{
        type: 'heatmap',
        data,
        label: { show: false },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)',
          },
        },
      }],
    };
  }, [filtered]);

  if (filtered.length === 0) {
    return (
      <div className="card bg-base-100 shadow-md p-6">
        <p className="text-center text-gray-500">No categorical longitudinal variables with class balance data available.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <DomainFilterBar variables={groups.flatMap(g => g.variables.map(m => m.variable))} selectedDomain={selectedDomain} onChange={setSelectedDomain} />
        <input
          type="text"
          className="input input-sm input-bordered flex-1 min-w-[200px]"
          placeholder="Search variables..."
          value={searchFilter}
          onChange={e => setSearchFilter(e.target.value)}
        />
        <select className="select select-sm select-bordered" value={sortKey} onChange={e => setSortKey(e.target.value as any)}>
          <option value="absChange">Sort by Abs Change</option>
          <option value="pctChange">Sort by % Change</option>
          <option value="timePoints">Sort by Time Points</option>
          <option value="name">Sort by Name</option>
        </select>
        <div className="join">
          <button className={`btn btn-sm ${viewMode === 'table' ? 'btn-active' : ''}`} onClick={() => setViewMode('table')}>Table</button>
          <button className={`btn btn-sm ${viewMode === 'heatmap' ? 'btn-active' : ''}`} onClick={() => setViewMode('heatmap')}>Heatmap</button>
        </div>
      </div>

      {/* Summary stats */}
      <div className="stats shadow w-full bg-base-100">
        <div className="stat">
          <div className="stat-title">Variables Ranked</div>
          <div className="stat-value text-amber-500">{filtered.length}</div>
        </div>
        <div className="stat">
          <div className="stat-title">Avg Abs Change</div>
          <div className="stat-value text-amber-500">{(filtered.reduce((s, g) => s + Math.abs(g.absoluteChange), 0) / filtered.length).toFixed(3)}</div>
        </div>
        <div className="stat">
          <div className="stat-title">Max Abs Change</div>
          <div className="stat-value text-amber-500">{Math.max(...filtered.map(g => Math.abs(g.absoluteChange))).toFixed(3)}</div>
        </div>
      </div>

      {/* View */}
      {viewMode === 'heatmap' ? (
        heatmapOption && (
          <div className="card bg-base-100 shadow-md p-4">
            <ReactECharts
              option={heatmapOption}
              style={{ height: Math.max(300, Math.min(filtered.length, 50) * 20 + 80) }}
              onEvents={{
                click: (params: any) => {
                  if (params.data && params.data[3]) {
                    setExpandedGroup(params.data[3].group);
                  }
                },
              }}
            />
          </div>
        )
      ) : (
        <div className="card bg-base-100 shadow-md">
          <div className="overflow-x-auto">
            <table className="table table-sm table-zebra">
              <thead>
                <tr>
                  <th>Variable</th>
                  <th>Label</th>
                  <th>Baseline Gini</th>
                  <th>Latest Gini</th>
                  <th>Abs Change</th>
                  <th>% Change</th>
                  <th>Time Points</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 50).map(g => (
                  <tr key={g.baseName} className="cursor-pointer hover:bg-primary/5" onClick={() => setExpandedGroup(g)}>
                    <td className="font-semibold">{g.baseName}</td>
                    <td>{g.label}</td>
                    <td>{g.baselineGini.toFixed(3)}</td>
                    <td>{g.latestGini.toFixed(3)}</td>
                    <td className={`font-medium ${g.absoluteChange > 0 ? 'text-red-500' : g.absoluteChange < 0 ? 'text-blue-500' : ''}`}>
                      {g.absoluteChange > 0 ? '+' : ''}{g.absoluteChange.toFixed(3)}
                    </td>
                    <td className={`font-medium ${g.percentChange > 0 ? 'text-red-500' : g.percentChange < 0 ? 'text-blue-500' : ''}`}>
                      {g.percentChange > 0 ? '+' : ''}{g.percentChange.toFixed(1)}%
                    </td>
                    <td>{g.timePoints}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Detail modal */}
      {expandedGroup && (
        <GroupDetailModal group={expandedGroup} onClose={() => setExpandedGroup(null)} />
      )}
    </div>
  );
};

const GroupDetailModal: React.FC<{ group: RankedGroup; onClose: () => void }> = ({ group, onClose }) => {
  const chartOption = useMemo(() => {
    const categories = group.allValues.map(v => v.visit);
    const giniValues = group.allValues.map(v => v.value);

    return {
      title: { text: `Class Balance Trend: ${group.baseName}`, left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const p = params[0];
          return `<strong>${categories[p.dataIndex]}</strong><br/>Gini: ${giniValues[p.dataIndex].toFixed(3)}`;
        },
      },
      xAxis: {
        type: 'category' as const,
        data: categories,
        axisLabel: { fontSize: 11, rotate: 30 },
      },
      yAxis: {
        type: 'value' as const,
        name: 'Gini Coefficient',
        nameLocation: 'center',
        nameGap: 30,
        min: 0,
        max: 1,
      },
      series: [{
        type: 'line',
        data: giniValues,
        smooth: true,
        lineStyle: { color: '#3b82f6', width: 2 },
        itemStyle: { color: '#3b82f6' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
              { offset: 1, color: 'rgba(59, 130, 246, 0.05)' },
            ],
          },
        },
      }],
      grid: { left: 80, right: 30, top: 50, bottom: 60 },
    };
  }, [group]);

  return (
    <div className="modal modal-open" onClick={onClose}>
      <div className="modal-box max-w-4xl max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <button className="btn btn-sm btn-circle btn-ghost absolute right-2 top-2" onClick={onClose}>✕</button>
        
        <h2 className="text-xl font-bold mb-4">{group.baseName}</h2>
        <p className="text-gray-600 mb-4">{group.label}</p>

        {/* Summary */}
        <div className="stats shadow w-full bg-base-100 mb-4">
          <div className="stat">
            <div className="stat-title">Baseline Gini</div>
            <div className="stat-value text-amber-500">{group.baselineGini.toFixed(3)}</div>
            <div className="stat-desc">{group.baselineVisit}</div>
          </div>
          <div className="stat">
            <div className="stat-title">Latest Gini</div>
            <div className="stat-value text-amber-500">{group.latestGini.toFixed(3)}</div>
            <div className="stat-desc">{group.latestVisit}</div>
          </div>
          <div className="stat">
            <div className="stat-title">Abs Change</div>
            <div className={`stat-value ${group.absoluteChange > 0 ? 'text-red-500' : group.absoluteChange < 0 ? 'text-blue-500' : 'text-green-500'}`}>
              {group.absoluteChange > 0 ? '+' : ''}{group.absoluteChange.toFixed(3)}
            </div>
            <div className="stat-desc">{group.percentChange.toFixed(1)}%</div>
          </div>
        </div>

        {/* Chart */}
        <div className="card bg-base-200 p-3 mb-4">
          <h3 className="font-semibold text-sm mb-2">Gini Coefficient Over Time</h3>
          <ReactECharts option={chartOption} style={{ height: 350 }} />
        </div>

        {/* Class balance table */}
        <div className="card bg-base-200 p-3">
          <h3 className="font-semibold text-sm mb-2">Class Balance by Time Point</h3>
          <div className="overflow-x-auto">
            <table className="table table-xs table-zebra">
              <thead>
                <tr>
                  <th>Visit</th>
                  <th>Variable</th>
                  <th>Gini</th>
                  <th>Num Classes</th>
                  <th>Most Frequent</th>
                </tr>
              </thead>
              <tbody>
                {group.variables.map(({ visit, variable: v }) => (
                  <tr key={v.name}>
                    <td className="font-medium">{visit}</td>
                    <td className="text-xs">{v.name}</td>
                    <td>{calculateGini(v.classBalance || []).toFixed(3)}</td>
                    <td>{v.classBalance?.length || 0}</td>
                    <td>{v.mostFrequentCategory || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EdaLongitudinalClassBalance;
