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

  // Individual mini-charts for more detail
  const miniChartOptions = useMemo(() => {
    return pageVars.map(v => {
      // Build a simplified histogram approximation from x-ticks / y-ticks
      // We use a gaussian approximation centered on mean with stdDev
      const nPoints = 50;
      const mean = v.mean ?? 0;
      const sd = v.stdDev ?? 1;
      const xMin = v.min ?? (mean - 3 * sd);
      const xMax = v.max ?? (mean + 3 * sd);
      const step = (xMax - xMin) / nPoints;
      const curveData: [number, number][] = [];
      for (let i = 0; i <= nPoints; i++) {
        const x = xMin + i * step;
        const y = Math.exp(-0.5 * Math.pow((x - mean) / sd, 2)) / (sd * Math.sqrt(2 * Math.PI));
        curveData.push([x, y]);
      }

      return {
        variable: v,
        option: {
          tooltip: { trigger: 'axis' },
          xAxis: {
            type: 'value',
            min: xMin,
            max: xMax,
            axisLabel: { fontSize: 9 },
            splitLine: { show: false },
          },
          yAxis: { type: 'value', show: false },
          series: [
            {
              type: 'line',
              data: curveData,
              smooth: true,
              areaStyle: { color: 'rgba(59, 130, 246, 0.15)' },
              lineStyle: { color: '#3b82f6', width: 2 },
              symbol: 'none',
            },
            // Mark lines for key stats
            {
              type: 'line',
              markLine: {
                silent: true,
                symbol: 'none',
                data: [
                  { xAxis: v.median, lineStyle: { color: '#10b981', type: 'solid', width: 2 }, label: { formatter: 'Med', fontSize: 9 } },
                  { xAxis: v.mean, lineStyle: { color: '#ef4444', type: 'dashed', width: 1.5 }, label: { formatter: 'Mean', fontSize: 9 } },
                  ...(v.q1 !== undefined ? [{ xAxis: v.q1, lineStyle: { color: '#94a3b8', type: 'dotted', width: 1 }, label: { formatter: 'Q1', fontSize: 8 } }] : []),
                  ...(v.q3 !== undefined ? [{ xAxis: v.q3, lineStyle: { color: '#94a3b8', type: 'dotted', width: 1 }, label: { formatter: 'Q3', fontSize: 8 } }] : []),
                ],
              },
              data: [],
            },
          ],
          grid: { left: 10, right: 10, top: 10, bottom: 20 },
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

      {/* Mini distribution curves grid */}
      <div className="card bg-base-100 shadow-md p-4">
        <h3 className="font-bold text-lg mb-3">Approximate Distribution Curves</h3>
        <p className="text-sm text-gray-500 mb-3">
          Gaussian approximation based on mean and std dev. Green = median, Red dashed = mean, Gray dotted = Q1/Q3.
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
