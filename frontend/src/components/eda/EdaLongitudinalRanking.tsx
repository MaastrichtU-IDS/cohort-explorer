import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { TimePointGroup, EdaVariable, completenessScore } from '@/utils/edaParsing';

export type RankingMetric = 'mean' | 'outliersZ' | 'outliersIqr' | 'iqr';

interface Props {
  groups: TimePointGroup[];
  onVariableClick: (v: EdaVariable) => void;
  metric: RankingMetric;
}

interface RankedGroup {
  baseName: string;
  label: string;
  baselineValue: number;
  latestValue: number;
  absoluteChange: number;
  percentChange: number;
  baselineVisit: string;
  latestVisit: string;
  timePoints: number;
  units?: string;
  allValues: { visit: string; value: number; sortOrder: number }[];
  variables: { visit: string; variable: EdaVariable }[];
}

type SortKey = 'absChange' | 'pctChange' | 'name' | 'timePoints';
type ViewMode = 'table' | 'heatmap';

const METRIC_CONFIG: Record<RankingMetric, {
  title: string;
  description: string;
  valueLabel: string;
  extractor: (v: EdaVariable) => number | undefined;
}> = {
  mean: {
    title: 'Rank Longitudinal Variables by Change in Mean',
    description: 'ranked by absolute change in mean from baseline to latest measurement.',
    valueLabel: 'Mean',
    extractor: v => v.mean,
  },
  outliersZ: {
    title: 'Rank Longitudinal Variables by Change in Outliers (Z-Score)',
    description: 'ranked by absolute change in Z-score outlier count from baseline to latest.',
    valueLabel: 'Outliers (Z)',
    extractor: v => v.outliersZ,
  },
  outliersIqr: {
    title: 'Rank Longitudinal Variables by Change in Outliers (IQR)',
    description: 'ranked by absolute change in IQR outlier count from baseline to latest.',
    valueLabel: 'Outliers (IQR)',
    extractor: v => v.outliersIqr,
  },
  iqr: {
    title: 'Rank Longitudinal Variables by Change in IQR',
    description: 'ranked by absolute change in interquartile range from baseline to latest.',
    valueLabel: 'IQR',
    extractor: v => v.iqr,
  },
};

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

const EdaLongitudinalRanking: React.FC<Props> = ({ groups, onVariableClick, metric }) => {
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [sortKey, setSortKey] = useState<SortKey>('absChange');
  const [searchFilter, setSearchFilter] = useState('');
  const [expandedGroup, setExpandedGroup] = useState<RankedGroup | null>(null);

  const config = METRIC_CONFIG[metric];

  // Compute ranked groups — only numeric groups with the metric available at 2+ time points
  const ranked: RankedGroup[] = useMemo(() => {
    return groups
      .map(g => {
        const withValue = g.variables.filter(m => config.extractor(m.variable) !== undefined && m.variable.type === 'numeric');
        if (withValue.length < 2) return null;

        // Sort members by visit time (numeric extraction)
        const sorted = [...withValue].sort((a, b) => visitSortOrder(a.visit) - visitSortOrder(b.visit));

        const allValues = sorted.map(m => ({ visit: m.visit, value: config.extractor(m.variable)!, sortOrder: visitSortOrder(m.visit) }));
        const baseline = sorted[0];
        const latest = sorted[sorted.length - 1];
        const baselineValue = config.extractor(baseline.variable)!;
        const latestValue = config.extractor(latest.variable)!;
        const absoluteChange = latestValue - baselineValue;
        const percentChange = baselineValue !== 0 ? ((latestValue - baselineValue) / Math.abs(baselineValue)) * 100 : 0;

        return {
          baseName: g.baseName,
          label: g.label,
          baselineValue,
          latestValue,
          absoluteChange,
          percentChange,
          baselineVisit: baseline.visit,
          latestVisit: latest.visit,
          timePoints: sorted.length,
          units: baseline.variable.units,
          allValues,
          variables: sorted,
        } as RankedGroup;
      })
      .filter((g): g is RankedGroup => g !== null);
  }, [groups, config]);

  // Filter and sort — always descending (biggest absolute changes first)
  const sorted = useMemo(() => {
    let list = [...ranked];
    if (searchFilter) {
      const q = searchFilter.toLowerCase();
      list = list.filter(g => g.baseName.toLowerCase().includes(q) || g.label.toLowerCase().includes(q));
    }
    list.sort((a, b) => {
      switch (sortKey) {
        case 'absChange': return Math.abs(b.absoluteChange) - Math.abs(a.absoluteChange);
        case 'pctChange': return Math.abs(b.percentChange) - Math.abs(a.percentChange);
        case 'name': return a.baseName.localeCompare(b.baseName);
        case 'timePoints': return b.timePoints - a.timePoints;
        default: return 0;
      }
    });
    return list;
  }, [ranked, searchFilter, sortKey, config]);

  // Heatmap option
  const heatmapOption = useMemo(() => {
    if (sorted.length === 0) return null;

    // Collect all unique visits and sort by numeric time
    const visitMap = new Map<string, number>();
    for (const g of sorted) {
      for (const m of g.allValues) {
        if (!visitMap.has(m.visit)) {
          visitMap.set(m.visit, m.sortOrder);
        }
      }
    }
    const allVisits = Array.from(visitMap.entries())
      .sort((a, b) => a[1] - b[1])
      .map(([visit]) => visit);

    // Build heatmap data: each cell = percent change from baseline for that group at that visit
    const heatData: [number, number, number | null][] = [];
    const varNames = sorted.map(g => g.baseName);

    for (let gi = 0; gi < sorted.length; gi++) {
      const g = sorted[gi];
      const baseline = g.allValues[0].value;
      for (let vi = 0; vi < allVisits.length; vi++) {
        const entry = g.allValues.find(m => m.visit === allVisits[vi]);
        if (entry && baseline !== 0) {
          const pctFromBaseline = ((entry.value - baseline) / Math.abs(baseline)) * 100;
          heatData.push([vi, gi, parseFloat(pctFromBaseline.toFixed(2))]);
        } else if (entry) {
          heatData.push([vi, gi, parseFloat((entry.value - baseline).toFixed(2))]);
        } else {
          heatData.push([vi, gi, null]);
        }
      }
    }

    const maxAbs = Math.max(...heatData.filter(d => d[2] !== null).map(d => Math.abs(d[2] as number)), 1);

    return {
      tooltip: {
        formatter: (p: any) => {
          const g = sorted[p.data[1]];
          const visit = allVisits[p.data[0]];
          const val = p.data[2];
          const entry = g.allValues.find(m => m.visit === visit);
          return `<strong>${g.baseName}</strong><br/>${g.label}<br/>` +
            `Visit: ${visit}<br/>` +
            `${config.valueLabel}: ${entry?.value.toFixed(2) ?? 'N/A'}${g.units ? ' ' + g.units : ''}<br/>` +
            `Change from baseline: ${val !== null ? val.toFixed(2) + '%' : 'N/A'}`;
        },
      },
      grid: { left: 160, right: 80, top: 30, bottom: 60 },
      xAxis: {
        type: 'category' as const,
        data: allVisits,
        axisLabel: { rotate: 30, fontSize: 10 },
        position: 'bottom' as const,
      },
      yAxis: {
        type: 'category' as const,
        data: varNames,
        axisLabel: { fontSize: 9, width: 150, overflow: 'truncate' as const },
      },
      visualMap: {
        min: -maxAbs,
        max: maxAbs,
        calculable: true,
        orient: 'vertical' as const,
        right: 0,
        top: 'center',
        inRange: {
          color: ['#2563eb', '#93c5fd', '#f3f4f6', '#fca5a5', '#dc2626'],
        },
        text: [`+${maxAbs.toFixed(0)}%`, `-${maxAbs.toFixed(0)}%`],
        textStyle: { fontSize: 10 },
      },
      series: [{
        type: 'heatmap' as const,
        data: heatData.filter(d => d[2] !== null),
        label: {
          show: sorted.length <= 25,
          fontSize: 9,
          formatter: (p: any) => `${p.data[2] > 0 ? '+' : ''}${p.data[2]}%`,
        },
        emphasis: {
          itemStyle: { shadowBlur: 6, shadowColor: 'rgba(0,0,0,0.3)' },
        },
      }],
    };
  }, [sorted, config]);

  if (ranked.length === 0) {
    return (
      <div className="alert alert-info max-w-xl mx-auto my-8">
        <span>No longitudinal variable groups with {config.valueLabel.toLowerCase()} data found at 2+ time points.</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-lg font-bold">{config.title}</h2>
        <p className="text-sm text-gray-500 mt-1">
          {ranked.length} variable groups measured across multiple time points, {config.description}
        </p>
      </div>

      {/* View mode switch */}
      <div className="flex justify-center">
        <div className="join">
          <button
            className={`btn btn-sm join-item ${viewMode === 'table' ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => setViewMode('table')}
          >
            Table
          </button>
          <button
            className={`btn btn-sm join-item ${viewMode === 'heatmap' ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => setViewMode('heatmap')}
          >
            Heatmap
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-center gap-3">
        <input
          type="text"
          placeholder="Filter variables..."
          className="input input-sm input-bordered w-60"
          value={searchFilter}
          onChange={e => setSearchFilter(e.target.value)}
        />
        {viewMode === 'table' && (
          <select
            className="select select-sm select-bordered"
            value={sortKey}
            onChange={e => setSortKey(e.target.value as SortKey)}
          >
            <option value="absChange">Sort by Absolute Change</option>
            <option value="pctChange">Sort by % Change</option>
            <option value="name">Sort by Name</option>
            <option value="timePoints">Sort by # Time Points</option>
          </select>
        )}
        <span className="text-sm text-gray-500">{sorted.length} groups</span>
      </div>

      {/* Table view */}
      {viewMode === 'table' && (
        <div className="card bg-base-100 shadow-md">
          <div className="overflow-x-auto">
            <table className="table table-sm table-zebra table-pin-rows">
              <thead>
                <tr>
                  <th className="text-center w-12">#</th>
                  <th className="cursor-pointer select-none" onClick={() => setSortKey('name')}>
                    Variable {sortKey === 'name' ? '▼' : ''}
                  </th>
                  <th>Label</th>
                  <th>Units</th>
                  <th className="cursor-pointer select-none" onClick={() => setSortKey('timePoints')}>
                    Points {sortKey === 'timePoints' ? '▼' : ''}
                  </th>
                  <th>Baseline {config.valueLabel}</th>
                  <th>Latest {config.valueLabel}</th>
                  <th className="cursor-pointer select-none" onClick={() => setSortKey('absChange')}>
                    Abs Change {sortKey === 'absChange' ? '▼' : ''}
                  </th>
                  <th className="cursor-pointer select-none" onClick={() => setSortKey('pctChange')}>
                    % Change {sortKey === 'pctChange' ? '▼' : ''}
                  </th>
                  <th>Trend</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((g, i) => (
                  <tr
                    key={g.baseName}
                    className="cursor-pointer hover:bg-primary/5"
                    onClick={() => setExpandedGroup(g)}
                  >
                    <td className="text-center font-bold text-gray-400">{i + 1}</td>
                    <td className="font-medium text-xs max-w-[150px] truncate" title={g.baseName}>{g.baseName}</td>
                    <td className="text-xs max-w-[200px] truncate" title={g.label}>{g.label}</td>
                    <td className="text-xs">{g.units || '—'}</td>
                    <td className="text-center">{g.timePoints}</td>
                    <td className="text-xs">
                      {g.baselineValue.toFixed(2)}
                      <span className="text-gray-400 text-[10px] ml-1">({g.baselineVisit})</span>
                    </td>
                    <td className="text-xs">
                      {g.latestValue.toFixed(2)}
                      <span className="text-gray-400 text-[10px] ml-1">({g.latestVisit})</span>
                    </td>
                    <td className={`font-medium ${g.absoluteChange > 0 ? 'text-red-500' : g.absoluteChange < 0 ? 'text-blue-500' : ''}`}>
                      {g.absoluteChange > 0 ? '+' : ''}{g.absoluteChange.toFixed(2)}
                    </td>
                    <td className={`font-medium ${g.percentChange > 0 ? 'text-red-500' : g.percentChange < 0 ? 'text-blue-500' : ''}`}>
                      {g.percentChange > 0 ? '+' : ''}{g.percentChange.toFixed(1)}%
                    </td>
                    <td>
                      {/* Mini sparkline showing mean trajectory */}
                      <div className="flex items-end gap-[2px] h-5">
                        {g.allValues.map((m, mi) => {
                          const minM = Math.min(...g.allValues.map(x => x.value));
                          const maxM = Math.max(...g.allValues.map(x => x.value));
                          const range = maxM - minM || 1;
                          const h = ((m.value - minM) / range) * 16 + 4;
                          return (
                            <div
                              key={mi}
                              className="rounded-sm"
                              style={{
                                width: 4,
                                height: h,
                                backgroundColor: mi === 0 ? '#3b82f6' : mi === g.allValues.length - 1 ? (g.absoluteChange > 0 ? '#ef4444' : '#10b981') : '#94a3b8',
                              }}
                              title={`${m.visit}: ${m.value.toFixed(2)}`}
                            />
                          );
                        })}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Heatmap view */}
      {viewMode === 'heatmap' && heatmapOption && (
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts
            option={heatmapOption}
            style={{ height: Math.max(400, sorted.length * 24 + 100) }}
            onEvents={{
              click: (params: any) => {
                if (params.data && params.data[1] !== undefined && sorted[params.data[1]]) {
                  setExpandedGroup(sorted[params.data[1]]);
                }
              },
            }}
          />
        </div>
      )}

      {/* Group detail modal — box plots for all time points */}
      {expandedGroup && (
        <GroupDetailModal group={expandedGroup} onClose={() => setExpandedGroup(null)} />
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Group detail modal: shows box plots for each time point of a longitudinal group
// ---------------------------------------------------------------------------

interface GroupDetailModalProps {
  group: RankedGroup;
  onClose: () => void;
}

const GroupDetailModal: React.FC<GroupDetailModalProps> = ({ group: g, onClose }) => {
  const boxPlotOption = useMemo(() => {
    const numericVars = g.variables
      .filter(m => m.variable.type === 'numeric' && m.variable.mean !== undefined)
      .sort((a, b) => visitSortOrder(a.visit) - visitSortOrder(b.visit));

    if (numericVars.length === 0) return null;

    const visits = numericVars.map(m => m.visit);
    const boxData = numericVars.map(m => {
      const v = m.variable;
      const min = v.min ?? 0;
      const q1 = v.q1 ?? 0;
      const median = v.median ?? 0;
      const q3 = v.q3 ?? 0;
      const max = v.max ?? 0;
      const iqr = v.iqr ?? (q3 - q1);
      const lowerWhisker = Math.max(min, q1 - 1.5 * iqr);
      const upperWhisker = Math.min(max, q3 + 1.5 * iqr);
      return [lowerWhisker, q1, median, q3, upperWhisker];
    });
    const meanData = numericVars.map((m, i) => [i, m.variable.mean ?? 0]);

    // Outlier points (min/max beyond whiskers)
    const outlierData: [number, number][] = [];
    numericVars.forEach((m, i) => {
      const v = m.variable;
      const iqr = v.iqr ?? ((v.q3 ?? 0) - (v.q1 ?? 0));
      const lw = Math.max(v.min ?? 0, (v.q1 ?? 0) - 1.5 * iqr);
      const uw = Math.min(v.max ?? 0, (v.q3 ?? 0) + 1.5 * iqr);
      if ((v.min ?? 0) < lw) outlierData.push([i, v.min ?? 0]);
      if ((v.max ?? 0) > uw) outlierData.push([i, v.max ?? 0]);
    });

    return {
      tooltip: {
        trigger: 'item' as const,
        formatter: (params: any) => {
          if (params.seriesType === 'boxplot') {
            const v = numericVars[params.dataIndex].variable;
            return `<strong>${numericVars[params.dataIndex].visit}</strong> (${v.name})<br/>` +
              `Min: ${v.min} · Q1: ${v.q1}<br/>` +
              `Median: ${v.median} · Mean: ${v.mean?.toFixed(2)}<br/>` +
              `Q3: ${v.q3} · Max: ${v.max}<br/>` +
              `Std Dev: ${v.stdDev?.toFixed(2)} · IQR: ${v.iqr}<br/>` +
              `n=${v.totalObservations} · ${v.isNormal ? 'Normal' : 'Non-normal'}`;
          }
          if (params.seriesType === 'scatter' && params.seriesName === 'Mean') {
            const v = numericVars[params.data[0]].variable;
            return `Mean: ${v.mean?.toFixed(2)}`;
          }
          return '';
        },
      },
      grid: { left: 80, right: 30, top: 40, bottom: 50 },
      xAxis: {
        type: 'category' as const,
        data: visits,
        axisLabel: { fontSize: 11 },
        name: 'Visit',
        nameLocation: 'center' as const,
        nameGap: 35,
      },
      yAxis: {
        type: 'value' as const,
        name: g.units || 'Value',
      },
      series: [
        {
          name: 'Box Plot',
          type: 'boxplot' as const,
          data: boxData,
          itemStyle: { color: '#dbeafe', borderColor: '#3b82f6', borderWidth: 2 },
        },
        {
          name: 'Mean',
          type: 'scatter' as const,
          data: meanData,
          symbolSize: 10,
          symbol: 'diamond',
          itemStyle: { color: '#ef4444', borderColor: '#fff', borderWidth: 1 },
          z: 10,
        },
        // Mean trend line
        {
          name: 'Mean Trend',
          type: 'line' as const,
          data: meanData.map(d => d[1]),
          lineStyle: { color: '#ef4444', type: 'dashed' as const, width: 1.5 },
          symbol: 'none',
          z: 5,
        },
        ...(outlierData.length > 0 ? [{
          name: 'Outliers',
          type: 'scatter' as const,
          data: outlierData,
          symbolSize: 6,
          itemStyle: { color: '#f59e0b' },
        }] : []),
      ],
    };
  }, [g]);

  // Summary stats table
  const statsRows = useMemo(() => {
    return g.variables
      .filter(m => m.variable.type === 'numeric' && m.variable.mean !== undefined)
      .sort((a, b) => visitSortOrder(a.visit) - visitSortOrder(b.visit));
  }, [g]);

  return (
    <div className="modal modal-open" onClick={onClose}>
      <div className="modal-box max-w-5xl max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-xl font-bold">{g.baseName}</h2>
            <p className="text-gray-500">{g.label}</p>
            <div className="flex gap-2 mt-2">
              <span className="badge badge-primary">{g.timePoints} time points</span>
              {g.units && <span className="badge badge-outline">{g.units}</span>}
              <span className={`badge ${Math.abs(g.percentChange) > 10 ? 'badge-error' : 'badge-info'}`}>
                {g.percentChange > 0 ? '+' : ''}{g.percentChange.toFixed(1)}% change
              </span>
            </div>
          </div>
          <button onClick={onClose} className="btn btn-sm btn-circle">✕</button>
        </div>

        {/* Box plots chart */}
        {boxPlotOption && (
          <div className="card bg-base-200 p-3 mb-4">
            <h3 className="font-semibold text-sm mb-1">Distribution Across Time Points</h3>
            <p className="text-xs text-gray-500 mb-2">
              Box = IQR (Q1–Q3), line = median, <span className="text-red-500">◆</span> = mean (dashed line = mean trend), <span className="text-amber-500">●</span> = outliers
            </p>
            <ReactECharts option={boxPlotOption} style={{ height: 350 }} />
          </div>
        )}

        {/* Stats comparison table */}
        <div className="card bg-base-200 p-3">
          <h3 className="font-semibold text-sm mb-2">Statistics by Time Point</h3>
          <div className="overflow-x-auto">
            <table className="table table-xs table-zebra">
              <thead>
                <tr>
                  <th>Visit</th>
                  <th>Variable</th>
                  <th>n</th>
                  <th>Mean</th>
                  <th>Median</th>
                  <th>Std Dev</th>
                  <th>Min</th>
                  <th>Q1</th>
                  <th>Q3</th>
                  <th>Max</th>
                  <th>IQR</th>
                  <th>Skewness</th>
                  <th>Outliers (IQR)</th>
                  <th>Completeness</th>
                </tr>
              </thead>
              <tbody>
                {statsRows.map(({ visit, variable: v }) => (
                  <tr key={v.name}>
                    <td className="font-medium">{visit}</td>
                    <td className="text-xs">{v.name}</td>
                    <td>{v.totalObservations}</td>
                    <td>{v.mean?.toFixed(2)}</td>
                    <td>{v.median}</td>
                    <td>{v.stdDev?.toFixed(2)}</td>
                    <td>{v.min}</td>
                    <td>{v.q1}</td>
                    <td>{v.q3}</td>
                    <td>{v.max}</td>
                    <td>{v.iqr}</td>
                    <td>{v.skewness?.toFixed(2)}</td>
                    <td>{v.outliersIqr ?? 0} ({v.outliersIqrPct?.toFixed(1) ?? 0}%)</td>
                    <td>{completenessScore(v).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="modal-backdrop" onClick={onClose}></div>
    </div>
  );
};

export default EdaLongitudinalRanking;
