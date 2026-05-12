import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable, completenessScore } from '@/utils/edaParsing';

interface Props {
  variables: EdaVariable[];
  onVariableClick: (v: EdaVariable) => void;
}

const EdaDataQualityMatrix: React.FC<Props> = ({ variables, onVariableClick }) => {
  const [sortBy, setSortBy] = useState<'name' | 'empty' | 'missing' | 'completeness'>('completeness');
  const [filterType, setFilterType] = useState<'all' | 'numeric' | 'categorical'>('all');

  const filtered = useMemo(() => {
    let list = [...variables];
    if (filterType !== 'all') list = list.filter(v => v.type === filterType);
    list.sort((a, b) => {
      switch (sortBy) {
        case 'empty': return b.countEmptyPct - a.countEmptyPct;
        case 'missing': return b.countMissingPct - a.countMissingPct;
        case 'completeness': return completenessScore(a) - completenessScore(b);
        default: return a.name.localeCompare(b.name);
      }
    });
    return list;
  }, [variables, sortBy, filterType]);

  // Heatmap: variables (y) vs quality metrics (x)
  const heatmapOption = useMemo(() => {
    const display = filtered.slice(0, 80); // limit for readability
    const names = display.map(v => v.name);
    const metrics = ['Empty %', 'Missing %', 'Completeness %'];

    const data: [number, number, number][] = [];
    display.forEach((v, yi) => {
      data.push([0, yi, v.countEmptyPct]);
      data.push([1, yi, v.countMissingPct]);
      data.push([2, yi, completenessScore(v)]);
    });

    return {
      title: { text: 'Data Quality Heatmap', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        formatter: (params: any) => {
          const v = display[params.data[1]];
          const metric = metrics[params.data[0]];
          return `<strong>${v.name}</strong><br/>${v.label}<br/>${metric}: ${params.data[2].toFixed(1)}%`;
        },
      },
      grid: { left: 180, right: 80, top: 40, bottom: 30 },
      xAxis: { type: 'category', data: metrics, position: 'top', axisLabel: { fontSize: 11 } },
      yAxis: {
        type: 'category',
        data: names,
        axisLabel: { fontSize: 9, width: 160, overflow: 'truncate' },
      },
      visualMap: {
        min: 0,
        max: 100,
        calculable: true,
        orient: 'vertical',
        right: 10,
        top: 'center',
        inRange: {
          color: ['#ef4444', '#f59e0b', '#fbbf24', '#a3e635', '#22c55e'],
        },
      },
      series: [{
        type: 'heatmap',
        data,
        label: { show: display.length <= 30, fontSize: 9, formatter: (p: any) => `${p.data[2].toFixed(0)}%` },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } },
      }],
    };
  }, [filtered]);

  // Completeness waterfall — sorted bar chart
  const waterfallOption = useMemo(() => {
    const display = filtered.slice(0, 50);
    return {
      title: { text: 'Variable Completeness (sorted)', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const p = params[0];
          const v = display[p.dataIndex];
          return `<strong>${v.name}</strong><br/>${v.label}<br/>
            Complete: ${completenessScore(v).toFixed(1)}%<br/>
            Empty: ${v.countEmptyPct.toFixed(1)}%<br/>
            Missing: ${v.countMissingPct.toFixed(1)}%<br/>
            Observations: ${v.totalObservations}`;
        },
      },
      grid: { left: 160, right: 20, top: 40, bottom: 30 },
      xAxis: { type: 'value', max: 100, name: '% Complete' },
      yAxis: {
        type: 'category',
        data: display.map(v => v.name),
        axisLabel: { fontSize: 9, width: 140, overflow: 'truncate' },
      },
      series: [{
        type: 'bar',
        data: display.map(v => {
          const score = completenessScore(v);
          return {
            value: score,
            itemStyle: {
              color: score >= 80 ? '#22c55e' : score >= 50 ? '#f59e0b' : '#ef4444',
            },
          };
        }),
        barMaxWidth: 20,
      }],
    };
  }, [filtered]);

  // Summary statistics
  const summary = useMemo(() => {
    const scores = filtered.map(v => completenessScore(v));
    const avg = scores.reduce((a, b) => a + b, 0) / (scores.length || 1);
    const gt80 = scores.filter(s => s >= 80).length;
    const lt50 = scores.filter(s => s < 50).length;
    return { avg, gt80, lt50, total: filtered.length };
  }, [filtered]);

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="stats shadow w-full bg-base-100">
        <div className="stat">
          <div className="stat-title">Avg Completeness</div>
          <div className={`stat-value ${summary.avg >= 80 ? 'text-green-500' : summary.avg >= 50 ? 'text-amber-500' : 'text-red-500'}`}>
            {summary.avg.toFixed(1)}%
          </div>
        </div>
        <div className="stat">
          <div className="stat-title">≥80% Complete</div>
          <div className="stat-value text-green-500">{summary.gt80}</div>
          <div className="stat-desc">{((summary.gt80 / summary.total) * 100).toFixed(0)}% of variables</div>
        </div>
        <div className="stat">
          <div className="stat-title">&lt;50% Complete</div>
          <div className="stat-value text-red-500">{summary.lt50}</div>
          <div className="stat-desc">{((summary.lt50 / summary.total) * 100).toFixed(0)}% of variables</div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <select className="select select-sm select-bordered" value={sortBy} onChange={e => setSortBy(e.target.value as any)}>
          <option value="completeness">Sort by Completeness</option>
          <option value="empty">Sort by Empty %</option>
          <option value="missing">Sort by Missing %</option>
          <option value="name">Sort by Name</option>
        </select>
        <div className="join">
          {(['all', 'numeric', 'categorical'] as const).map(t => (
            <button
              key={t}
              className={`btn btn-sm join-item ${filterType === t ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setFilterType(t)}
            >
              {t === 'all' ? 'All' : t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts option={heatmapOption} style={{ height: Math.max(400, Math.min(filtered.length, 80) * 18 + 80) }} />
        </div>
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts option={waterfallOption} style={{ height: Math.max(400, Math.min(filtered.length, 50) * 18 + 80) }} />
        </div>
      </div>

      {/* Clickable variable tiles */}
      <div className="card bg-base-100 shadow-md p-4">
        <h3 className="font-bold text-lg mb-3">Variables — click for details</h3>
        <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-8 gap-2 max-h-[400px] overflow-y-auto">
          {filtered.map(v => {
            const score = completenessScore(v);
            const r = Math.round(255 * (1 - score / 100));
            const g = Math.round(200 * (score / 100));
            return (
              <button
                key={v.name}
                onClick={() => onVariableClick(v)}
                className="rounded p-1 text-center hover:ring-2 ring-primary transition-all cursor-pointer"
                style={{ backgroundColor: `rgba(${r}, ${g}, 80, 0.2)`, border: `1px solid rgba(${r}, ${g}, 80, 0.4)` }}
                title={`${v.name}\n${v.label}\nComplete: ${score.toFixed(1)}%`}
              >
                <div className="text-[10px] font-medium truncate">{v.name}</div>
                <div className="text-[10px]">{score.toFixed(0)}%</div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default EdaDataQualityMatrix;
