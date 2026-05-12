import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable } from '@/utils/edaParsing';

interface Props {
  variables: EdaVariable[];
  onVariableClick: (v: EdaVariable) => void;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#84cc16', '#6366f1'];
const PAGE_SIZE = 12;

const EdaCategoricalCharts: React.FC<Props> = ({ variables, onVariableClick }) => {
  const [page, setPage] = useState(0);
  const [sortBy, setSortBy] = useState<'name' | 'categories' | 'completeness'>('name');
  const [searchFilter, setSearchFilter] = useState('');
  const [chartType, setChartType] = useState<'bar' | 'pie' | 'treemap'>('bar');

  // Filter out variables that are essentially all empty (>99% NA with ≤1 real category)
  const meaningfulVars = useMemo(() => {
    return variables.filter(v => {
      if (!v.classBalance || v.classBalance.length === 0) return false;
      const realCategories = v.classBalance.filter(c => c.label !== '<na>' && c.percentage > 0);
      return realCategories.length > 0;
    });
  }, [variables]);

  const filtered = useMemo(() => {
    let list = [...meaningfulVars];
    if (searchFilter) {
      const q = searchFilter.toLowerCase();
      list = list.filter(v => v.name.toLowerCase().includes(q) || v.label.toLowerCase().includes(q));
    }
    list.sort((a, b) => {
      switch (sortBy) {
        case 'categories': return (b.uniqueValues || 0) - (a.uniqueValues || 0);
        case 'completeness': return (100 - (b.countEmptyPct || 0)) - (100 - (a.countEmptyPct || 0));
        default: return a.name.localeCompare(b.name);
      }
    });
    return list;
  }, [meaningfulVars, sortBy, searchFilter]);

  const pageCount = Math.ceil(filtered.length / PAGE_SIZE);
  const pageVars = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  // Generate chart options per variable
  const chartOptions = useMemo(() => {
    return pageVars.map(v => {
      const categories = v.classBalance || [];
      const labels = categories.map(c => c.label);
      const counts = categories.map(c => c.count);
      const pcts = categories.map(c => c.percentage);

      if (chartType === 'pie') {
        return {
          variable: v,
          option: {
            tooltip: {
              trigger: 'item',
              formatter: (p: any) => `<strong>${p.name}</strong><br/>${p.data.count} obs (${p.data.pct.toFixed(1)}%)`,
            },
            series: [{
              type: 'pie',
              radius: ['25%', '60%'],
              itemStyle: { borderRadius: 4, borderColor: '#fff', borderWidth: 1 },
              label: { show: categories.length <= 8, fontSize: 9, formatter: '{b}' },
              data: categories.map((c, i) => ({
                value: c.count,
                name: c.label,
                count: c.count,
                pct: c.percentage,
                itemStyle: { color: COLORS[i % COLORS.length] },
              })),
            }],
          },
        };
      }

      if (chartType === 'treemap') {
        return {
          variable: v,
          option: {
            tooltip: {
              formatter: (p: any) => `<strong>${p.name}</strong><br/>${p.data.count} obs (${p.data.pct.toFixed(1)}%)`,
            },
            series: [{
              type: 'treemap',
              roam: false,
              nodeClick: false,
              breadcrumb: { show: false },
              label: { show: true, formatter: '{b}', fontSize: 10 },
              data: categories.map((c, i) => ({
                name: c.label,
                value: c.count,
                count: c.count,
                pct: c.percentage,
                itemStyle: { color: COLORS[i % COLORS.length] },
              })),
            }],
          },
        };
      }

      // Default: bar chart
      return {
        variable: v,
        option: {
          tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            formatter: (params: any) => {
              const p = params[0];
              const idx = p.dataIndex;
              return `<strong>${labels[idx]}</strong><br/>${counts[idx]} observations (${pcts[idx].toFixed(1)}%)`;
            },
          },
          xAxis: {
            type: 'category',
            data: labels,
            axisLabel: { rotate: labels.length > 5 ? 45 : 0, fontSize: 9, interval: 0 },
          },
          yAxis: { type: 'value', name: 'Count', axisLabel: { fontSize: 9 } },
          series: [{
            type: 'bar',
            data: counts.map((c, i) => ({
              value: c,
              itemStyle: { color: COLORS[i % COLORS.length] },
            })),
            barMaxWidth: 40,
            label: { show: categories.length <= 10, position: 'top', fontSize: 9, formatter: (p: any) => `${pcts[p.dataIndex].toFixed(0)}%` },
          }],
          grid: { left: 40, right: 10, top: 10, bottom: labels.some(l => l.length > 5) ? 60 : 30 },
        },
      };
    });
  }, [pageVars, chartType]);

  // Stacked overview bar chart — all categorical vars in one view
  const stackedOverviewOption = useMemo(() => {
    const top20 = filtered.slice(0, 20);
    if (top20.length === 0) return null;

    // For each variable, show proportional stacked bars
    const varNames = top20.map(v => v.name);
    // Collect all unique category labels
    const allLabels = new Set<string>();
    top20.forEach(v => v.classBalance?.forEach(c => allLabels.add(c.label)));
    const labelArr = Array.from(allLabels);

    const series = labelArr.map((label, i) => ({
      name: label,
      type: 'bar' as const,
      stack: 'total',
      emphasis: { focus: 'series' as const },
      data: top20.map(v => {
        const cat = v.classBalance?.find(c => c.label === label);
        return cat ? cat.percentage : 0;
      }),
      itemStyle: { color: COLORS[i % COLORS.length] },
    }));

    return {
      title: { text: 'Category Distribution Overview (top 20)', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: (params: any) => {
        let html = `<strong>${params[0].name}</strong><br/>`;
        params.forEach((p: any) => {
          if (p.value > 0) html += `${p.marker} ${p.seriesName}: ${p.value.toFixed(1)}%<br/>`;
        });
        return html;
      }},
      legend: { bottom: 0, type: 'scroll' },
      grid: { left: 160, right: 20, top: 40, bottom: 60 },
      xAxis: { type: 'value', max: 100, name: '%' },
      yAxis: { type: 'category', data: varNames, axisLabel: { fontSize: 10, width: 140, overflow: 'truncate' } },
      series,
    };
  }, [filtered]);

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
        <select className="select select-sm select-bordered" value={sortBy} onChange={e => { setSortBy(e.target.value as any); setPage(0); }}>
          <option value="name">Sort by Name</option>
          <option value="categories">Sort by # Categories</option>
          <option value="completeness">Sort by Completeness</option>
        </select>
        <div className="join">
          {(['bar', 'pie', 'treemap'] as const).map(t => (
            <button
              key={t}
              className={`btn btn-sm join-item ${chartType === t ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setChartType(t)}
            >
              {t === 'bar' ? '📊 Bar' : t === 'pie' ? '🥧 Pie' : '🗂 Treemap'}
            </button>
          ))}
        </div>
        <span className="text-sm text-gray-500">{filtered.length} of {variables.length} variables (hiding all-empty)</span>
      </div>

      {/* Stacked overview */}
      {stackedOverviewOption && (
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts option={stackedOverviewOption} style={{ height: Math.max(300, Math.min(filtered.length, 20) * 30 + 120) }} />
        </div>
      )}

      {/* Individual charts grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {chartOptions.map(({ variable: v, option }) => (
          <div
            key={v.name}
            className="card bg-base-100 shadow-md p-3 hover:shadow-lg transition-shadow cursor-pointer"
            onClick={() => onVariableClick(v)}
          >
            <div className="font-medium text-sm truncate" title={v.name}>{v.name}</div>
            <div className="text-xs text-gray-500 truncate mb-2" title={v.label}>{v.label}</div>
            <ReactECharts option={option} style={{ height: 180 }} opts={{ renderer: 'svg' }} />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>{v.uniqueValues} categories</span>
              <span>n={v.totalObservations}</span>
            </div>
          </div>
        ))}
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

export default EdaCategoricalCharts;
