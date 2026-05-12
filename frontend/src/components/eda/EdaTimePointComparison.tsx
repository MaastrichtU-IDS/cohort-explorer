import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable, TimePointGroup } from '@/utils/edaParsing';

interface Props {
  groups: TimePointGroup[];
  onVariableClick: (v: EdaVariable) => void;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

const EdaTimePointComparison: React.FC<Props> = ({ groups, onVariableClick }) => {
  const [selectedGroup, setSelectedGroup] = useState<string>(groups[0]?.baseName || '');
  const [searchFilter, setSearchFilter] = useState('');
  const [metric, setMetric] = useState<'mean' | 'median' | 'stdDev' | 'completeness' | 'observations'>('mean');

  const filteredGroups = useMemo(() => {
    if (!searchFilter) return groups;
    const q = searchFilter.toLowerCase();
    return groups.filter(g =>
      g.baseName.toLowerCase().includes(q) ||
      g.label.toLowerCase().includes(q) ||
      g.variables.some(v => v.variable.label.toLowerCase().includes(q))
    );
  }, [groups, searchFilter]);

  const activeGroup = useMemo(
    () => filteredGroups.find(g => g.baseName === selectedGroup) || filteredGroups[0],
    [filteredGroups, selectedGroup]
  );

  // Multi-metric line chart for selected group
  const lineChartOption = useMemo(() => {
    if (!activeGroup) return null;
    const visits = activeGroup.variables.map(v => v.visit);
    const numericMembers = activeGroup.variables.filter(v => v.variable.type === 'numeric');

    if (numericMembers.length === 0) return null;

    return {
      title: {
        text: `${activeGroup.label || activeGroup.baseName} — Across Time Points`,
        left: 'center',
        textStyle: { fontSize: 14 },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          let html = `<strong>${params[0].name}</strong><br/>`;
          params.forEach((p: any) => {
            if (p.value !== null && p.value !== undefined)
              html += `${p.marker} ${p.seriesName}: ${typeof p.value === 'number' ? p.value.toFixed(2) : p.value}<br/>`;
          });
          return html;
        },
      },
      legend: { bottom: 0 },
      grid: { left: 60, right: 30, top: 50, bottom: 60 },
      xAxis: {
        type: 'category',
        data: numericMembers.map(v => v.visit),
        axisLabel: { fontSize: 11 },
      },
      yAxis: [
        { type: 'value', name: 'Value', position: 'left' },
        { type: 'value', name: 'Count', position: 'right' },
      ],
      series: [
        {
          name: 'Mean',
          type: 'line',
          data: numericMembers.map(v => v.variable.mean ?? null),
          lineStyle: { width: 3, color: '#3b82f6' },
          itemStyle: { color: '#3b82f6' },
          symbol: 'circle',
          symbolSize: 8,
        },
        {
          name: 'Median',
          type: 'line',
          data: numericMembers.map(v => v.variable.median ?? null),
          lineStyle: { width: 2, color: '#10b981', type: 'dashed' },
          itemStyle: { color: '#10b981' },
          symbol: 'diamond',
          symbolSize: 8,
        },
        {
          name: 'Std Dev',
          type: 'line',
          data: numericMembers.map(v => v.variable.stdDev ?? null),
          lineStyle: { width: 2, color: '#f59e0b', type: 'dotted' },
          itemStyle: { color: '#f59e0b' },
          symbol: 'triangle',
          symbolSize: 8,
        },
        {
          name: 'Observations',
          type: 'bar',
          yAxisIndex: 1,
          data: numericMembers.map(v => v.variable.totalObservations),
          itemStyle: { color: 'rgba(139, 92, 246, 0.3)' },
          barWidth: '30%',
        },
      ],
    };
  }, [activeGroup]);

  // Box plot comparison across time points
  const boxPlotOption = useMemo(() => {
    if (!activeGroup) return null;
    const numericMembers = activeGroup.variables.filter(v => v.variable.type === 'numeric');
    if (numericMembers.length === 0) return null;

    return {
      title: { text: 'Distribution Comparison', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (params.seriesType === 'boxplot') {
            const v = numericMembers[params.dataIndex]?.variable;
            if (!v) return '';
            return `<strong>${params.name}</strong><br/>
              Min: ${v.min} | Q1: ${v.q1}<br/>
              Median: ${v.median} | Mean: ${v.mean?.toFixed(2)}<br/>
              Q3: ${v.q3} | Max: ${v.max}<br/>
              n=${v.totalObservations}`;
          }
          return '';
        },
      },
      grid: { left: 60, right: 30, top: 40, bottom: 30 },
      xAxis: { type: 'category', data: numericMembers.map(v => v.visit) },
      yAxis: { type: 'value' },
      series: [
        {
          type: 'boxplot',
          data: numericMembers.map(v => [
            v.variable.min ?? 0,
            v.variable.q1 ?? 0,
            v.variable.median ?? 0,
            v.variable.q3 ?? 0,
            v.variable.max ?? 0,
          ]),
          itemStyle: { color: '#e0ecff', borderColor: '#3b82f6' },
        },
        {
          type: 'scatter',
          data: numericMembers.map((v, i) => [i, v.variable.mean ?? 0]),
          symbolSize: 10,
          itemStyle: { color: '#ef4444' },
        },
      ],
    };
  }, [activeGroup]);

  // Overview heatmap: all groups × metric across time points
  const overviewOption = useMemo(() => {
    const numericGroups = filteredGroups.filter(g =>
      g.variables.some(v => v.variable.type === 'numeric')
    ).slice(0, 30);

    if (numericGroups.length === 0) return null;

    // Collect all unique visit labels
    const allVisits = new Set<string>();
    numericGroups.forEach(g => g.variables.forEach(v => allVisits.add(v.visit)));
    const visitArr = Array.from(allVisits).sort();

    const getValue = (v: EdaVariable): number | null => {
      switch (metric) {
        case 'mean': return v.mean ?? null;
        case 'median': return v.median ?? null;
        case 'stdDev': return v.stdDev ?? null;
        case 'completeness': return 100 - v.countEmptyPct;
        case 'observations': return v.totalObservations;
      }
    };

    const data: [number, number, number | null][] = [];
    let minVal = Infinity;
    let maxVal = -Infinity;
    numericGroups.forEach((g, gi) => {
      visitArr.forEach((visit, vi) => {
        const member = g.variables.find(v => v.visit === visit);
        const val = member ? getValue(member.variable) : null;
        if (val !== null) {
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        }
        data.push([vi, gi, val]);
      });
    });

    return {
      title: { text: `Time-Point Overview: ${metric}`, left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        formatter: (p: any) => {
          const g = numericGroups[p.data[1]];
          return `<strong>${g.baseName}</strong><br/>${g.label}<br/>Visit: ${visitArr[p.data[0]]}<br/>${metric}: ${p.data[2] !== null ? p.data[2].toFixed(2) : 'N/A'}`;
        },
      },
      grid: { left: 180, right: 80, top: 40, bottom: 30 },
      xAxis: { type: 'category', data: visitArr, position: 'top', axisLabel: { fontSize: 10 } },
      yAxis: {
        type: 'category',
        data: numericGroups.map(g => g.baseName),
        axisLabel: { fontSize: 9, width: 160, overflow: 'truncate' },
      },
      visualMap: {
        min: isFinite(minVal) ? minVal : 0,
        max: isFinite(maxVal) ? maxVal : 100,
        calculable: true,
        orient: 'vertical',
        right: 10,
        top: 'center',
        inRange: { color: ['#eff6ff', '#3b82f6', '#1e3a8a'] },
      },
      series: [{
        type: 'heatmap',
        data: data.filter(d => d[2] !== null),
        label: { show: numericGroups.length <= 15 && visitArr.length <= 5, fontSize: 9, formatter: (p: any) => p.data[2]?.toFixed(1) },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } },
      }],
    };
  }, [filteredGroups, metric]);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Filter groups..."
          className="input input-sm input-bordered w-60"
          value={searchFilter}
          onChange={e => setSearchFilter(e.target.value)}
        />
        <select
          className="select select-sm select-bordered"
          value={metric}
          onChange={e => setMetric(e.target.value as any)}
        >
          <option value="mean">Mean</option>
          <option value="median">Median</option>
          <option value="stdDev">Std Dev</option>
          <option value="completeness">Completeness</option>
          <option value="observations">Observations</option>
        </select>
        <span className="text-sm text-gray-500">{filteredGroups.length} time-point groups</span>
      </div>

      {/* Overview heatmap */}
      {overviewOption && (
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts option={overviewOption} style={{ height: Math.max(300, Math.min(filteredGroups.length, 30) * 25 + 100) }} />
        </div>
      )}

      {/* Group selector + detail charts */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Group list */}
        <div className="card bg-base-100 shadow-md p-3 max-h-[600px] overflow-y-auto">
          <h3 className="font-bold text-sm mb-2">Time-Point Groups ({filteredGroups.length})</h3>
          <div className="space-y-1">
            {filteredGroups.map(g => (
              <button
                key={g.baseName}
                onClick={() => setSelectedGroup(g.baseName)}
                className={`w-full text-left p-2 rounded text-sm hover:bg-base-200 transition-colors ${
                  g.baseName === activeGroup?.baseName ? 'bg-primary/10 border border-primary font-medium' : ''
                }`}
              >
                <div className="font-medium truncate">{g.baseName}</div>
                <div className="text-xs text-gray-500">{g.variables.length} time points</div>
              </button>
            ))}
          </div>
        </div>

        {/* Detail charts */}
        <div className="lg:col-span-3 space-y-4">
          {lineChartOption && (
            <div className="card bg-base-100 shadow-md p-4">
              <ReactECharts option={lineChartOption} style={{ height: 350 }} />
            </div>
          )}
          {boxPlotOption && (
            <div className="card bg-base-100 shadow-md p-4">
              <ReactECharts option={boxPlotOption} style={{ height: 300 }} />
            </div>
          )}
          {activeGroup && (
            <div className="card bg-base-100 shadow-md p-4">
              <h3 className="font-bold mb-2">Variable Details</h3>
              <div className="overflow-x-auto">
                <table className="table table-sm table-zebra">
                  <thead>
                    <tr>
                      <th>Visit</th>
                      <th>Variable</th>
                      <th>Type</th>
                      <th>Obs</th>
                      <th>Empty %</th>
                      <th>Mean</th>
                      <th>Median</th>
                      <th>Std Dev</th>
                      <th>Min</th>
                      <th>Max</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeGroup.variables.map(({ visit, variable: v }) => (
                      <tr
                        key={v.name}
                        className="cursor-pointer hover:bg-primary/5"
                        onClick={() => onVariableClick(v)}
                      >
                        <td className="text-xs">{visit}</td>
                        <td className="font-medium text-xs">{v.name}</td>
                        <td><span className="badge badge-xs">{v.type}</span></td>
                        <td>{v.totalObservations}</td>
                        <td className={v.countEmptyPct > 50 ? 'text-red-500 font-medium' : ''}>{v.countEmptyPct.toFixed(1)}%</td>
                        <td>{v.mean?.toFixed(2) ?? '—'}</td>
                        <td>{v.median ?? '—'}</td>
                        <td>{v.stdDev?.toFixed(2) ?? '—'}</td>
                        <td>{v.min ?? '—'}</td>
                        <td>{v.max ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EdaTimePointComparison;
