import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable } from '@/utils/edaParsing';

interface Props {
  variables: EdaVariable[];
  onVariableClick: (v: EdaVariable) => void;
}

const PAGE_SIZE = 20;

const EdaNumericBoxPlots: React.FC<Props> = ({ variables, onVariableClick }) => {
  const [page, setPage] = useState(0);
  const [sortBy, setSortBy] = useState<'name' | 'mean' | 'spread' | 'skew' | 'completeness'>('name');
  const [searchFilter, setSearchFilter] = useState('');

  const filtered = useMemo(() => {
    let list = [...variables];
    if (searchFilter) {
      const q = searchFilter.toLowerCase();
      list = list.filter(v => v.name.toLowerCase().includes(q) || v.label.toLowerCase().includes(q));
    }
    list.sort((a, b) => {
      switch (sortBy) {
        case 'mean': return (b.mean || 0) - (a.mean || 0);
        case 'spread': return (b.stdDev || 0) - (a.stdDev || 0);
        case 'skew': return Math.abs(b.skewness || 0) - Math.abs(a.skewness || 0);
        case 'completeness': return (100 - (b.countEmptyPct || 0)) - (100 - (a.countEmptyPct || 0));
        default: return a.name.localeCompare(b.name);
      }
    });
    return list;
  }, [variables, sortBy, searchFilter]);

  const pageCount = Math.ceil(filtered.length / PAGE_SIZE);
  const pageVars = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  // Combined box plot chart for current page
  const boxPlotOption = useMemo(() => {
    if (pageVars.length === 0) return null;

    const names = pageVars.map(v => v.name);
    const boxData = pageVars.map(v => [
      v.min ?? 0,
      v.q1 ?? 0,
      v.median ?? 0,
      v.q3 ?? 0,
      v.max ?? 0,
    ]);
    const meanData = pageVars.map((v, i) => [i, v.mean ?? 0]);

    return {
      title: {
        text: `Numeric Variable Box Plots (${page * PAGE_SIZE + 1}–${Math.min((page + 1) * PAGE_SIZE, filtered.length)} of ${filtered.length})`,
        left: 'center',
        textStyle: { fontSize: 14 },
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (params.seriesType === 'boxplot') {
            const v = pageVars[params.dataIndex];
            return `<strong>${v.name}</strong><br/>
              ${v.label}<br/>
              Min: ${v.min} | Q1: ${v.q1}<br/>
              Median: ${v.median} | Mean: ${v.mean?.toFixed(2)}<br/>
              Q3: ${v.q3} | Max: ${v.max}<br/>
              Std Dev: ${v.stdDev?.toFixed(2)} | IQR: ${v.iqr}<br/>
              Skewness: ${v.skewness?.toFixed(2)} | Kurtosis: ${v.kurtosis?.toFixed(2)}<br/>
              ${v.units ? `Units: ${v.units}` : ''}`;
          }
          if (params.seriesType === 'scatter') {
            const v = pageVars[params.data[0]];
            return `<strong>${v.name}</strong><br/>Mean: ${v.mean?.toFixed(2)}`;
          }
          return '';
        },
      },
      grid: { left: 160, right: 30, top: 40, bottom: 30 },
      xAxis: { type: 'value', name: 'Value' },
      yAxis: {
        type: 'category',
        data: names,
        axisLabel: { fontSize: 10, width: 140, overflow: 'truncate' },
      },
      series: [
        {
          name: 'Box Plot',
          type: 'boxplot',
          data: boxData,
          itemStyle: { color: '#e0ecff', borderColor: '#3b82f6' },
          emphasis: { itemStyle: { borderColor: '#1d4ed8', borderWidth: 2 } },
        },
        {
          name: 'Mean',
          type: 'scatter',
          data: meanData,
          symbolSize: 8,
          itemStyle: { color: '#ef4444' },
          z: 10,
        },
      ],
    };
  }, [pageVars, page, filtered.length]);

  // Individual mini box plots for more detail
  const miniChartOptions = useMemo(() => {
    return pageVars.map(v => {
      const min = v.min ?? 0;
      const q1 = v.q1 ?? 0;
      const median = v.median ?? 0;
      const q3 = v.q3 ?? 0;
      const max = v.max ?? 0;
      const mean = v.mean ?? 0;
      const iqr = v.iqr ?? (q3 - q1);
      const lowerWhisker = Math.max(min, q1 - 1.5 * iqr);
      const upperWhisker = Math.min(max, q3 + 1.5 * iqr);

      const outlierPoints: number[][] = [];
      if (min < lowerWhisker) outlierPoints.push([min, 0]);
      if (max > upperWhisker) outlierPoints.push([max, 0]);

      const xTickNums = v.xTicksNumeric ?? [];
      const axisMin = xTickNums.length > 0 ? Math.min(...xTickNums) : min - (max - min) * 0.05;
      const axisMax = xTickNums.length > 0 ? Math.max(...xTickNums) : max + (max - min) * 0.05;

      return {
        variable: v,
        option: {
          tooltip: {
            trigger: 'item' as const,
            formatter: () =>
              `<strong>${v.name}</strong><br/>Min: ${min} · Q1: ${q1}<br/>Med: ${median} · Mean: ${mean.toFixed(1)}<br/>Q3: ${q3} · Max: ${max}`,
          },
          xAxis: {
            type: 'value' as const,
            min: axisMin,
            max: axisMax,
            axisLabel: { fontSize: 8 },
            splitLine: { show: false },
          },
          yAxis: { type: 'category' as const, data: [''], show: false },
          series: [
            {
              type: 'boxplot' as const,
              data: [[lowerWhisker, q1, median, q3, upperWhisker]],
              itemStyle: { color: '#dbeafe', borderColor: '#3b82f6', borderWidth: 1.5 },
              boxWidth: ['50%', '50%'],
            },
            {
              type: 'scatter' as const,
              data: [[mean, 0]],
              symbolSize: 8,
              symbol: 'diamond',
              itemStyle: { color: '#ef4444' },
              z: 10,
            },
            ...(outlierPoints.length > 0 ? [{
              type: 'scatter' as const,
              data: outlierPoints,
              symbolSize: 5,
              itemStyle: { color: '#f59e0b' },
            }] : []),
          ],
          grid: { left: 5, right: 5, top: 8, bottom: 18 },
        },
      };
    });
  }, [pageVars]);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Filter variables..."
          className="input input-sm input-bordered w-60"
          value={searchFilter}
          onChange={e => { setSearchFilter(e.target.value); setPage(0); }}
        />
        <select
          className="select select-sm select-bordered"
          value={sortBy}
          onChange={e => { setSortBy(e.target.value as any); setPage(0); }}
        >
          <option value="name">Sort by Name</option>
          <option value="mean">Sort by Mean</option>
          <option value="spread">Sort by Spread (Std Dev)</option>
          <option value="skew">Sort by Skewness</option>
          <option value="completeness">Sort by Completeness</option>
        </select>
        <span className="text-sm text-gray-500">{filtered.length} variables</span>
      </div>

      {/* Combined box plot */}
      {boxPlotOption && (
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts
            option={boxPlotOption}
            style={{ height: Math.max(300, pageVars.length * 35 + 80) }}
            onEvents={{
              click: (params: any) => {
                if (params.dataIndex !== undefined && pageVars[params.dataIndex]) {
                  onVariableClick(pageVars[params.dataIndex]);
                }
              },
            }}
          />
        </div>
      )}

      {/* Mini box plots grid */}
      <div className="card bg-base-100 shadow-md p-4">
        <h3 className="font-bold text-lg mb-3">Individual Box Plots</h3>
        <p className="text-sm text-gray-500 mb-3">
          Box = IQR (Q1–Q3), line = median, <span className="text-red-500">◆</span> = mean, <span className="text-amber-500">●</span> = outliers (min/max beyond whiskers).
          Click for details.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {miniChartOptions.map(({ variable: v, option }) => (
            <div
              key={v.name}
              className="border rounded-lg p-2 hover:shadow-md transition-shadow cursor-pointer bg-white"
              onClick={() => onVariableClick(v)}
            >
              <div className="font-medium text-xs truncate" title={v.name}>{v.name}</div>
              <div className="text-[10px] text-gray-500 truncate mb-1" title={v.label}>{v.label}</div>
              <ReactECharts option={option} style={{ height: 100 }} opts={{ renderer: 'svg' }} />
              <div className="flex justify-between text-[10px] text-gray-600 mt-1">
                <span>n={v.totalObservations}</span>
                <span>{v.isNormal ? '✓ Normal' : '✗ Non-normal'}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Pagination */}
      {pageCount > 1 && (
        <div className="flex justify-center gap-2">
          <button className="btn btn-sm" disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Prev</button>
          <span className="flex items-center text-sm">Page {page + 1} of {pageCount}</span>
          <button className="btn btn-sm" disabled={page >= pageCount - 1} onClick={() => setPage(p => p + 1)}>Next →</button>
        </div>
      )}
    </div>
  );
};

export default EdaNumericBoxPlots;
