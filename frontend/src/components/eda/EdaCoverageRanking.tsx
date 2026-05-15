import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable, completenessScore } from '@/utils/edaParsing';
import DomainFilterBar from './DomainFilterBar';

interface Props {
  variables: EdaVariable[];
  onVariableClick: (v: EdaVariable) => void;
}

type ViewMode = 'table' | 'chart';

const EdaCoverageRanking: React.FC<Props> = ({ variables, onVariableClick }) => {
  const [viewMode, setViewMode] = useState<ViewMode>('chart');
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);
  const [typeFilter, setTypeFilter] = useState<'all' | 'numeric' | 'categorical'>('all');

  const ranked = useMemo(() => {
    let list = variables.map(v => ({
      variable: v,
      completeness: completenessScore(v),
      missingPct: v.countMissingPct + v.countEmptyPct,
      n: v.totalObservations,
    }));

    if (typeFilter !== 'all') {
      list = list.filter(d => d.variable.type === typeFilter);
    }
    if (selectedDomain) list = list.filter(d => d.variable.domain === selectedDomain);
    if (searchFilter) {
      const q = searchFilter.toLowerCase();
      list = list.filter(d => d.variable.name.toLowerCase().includes(q) || d.variable.label.toLowerCase().includes(q) || (d.variable.conceptName && d.variable.conceptName.toLowerCase().includes(q)));
    }

    // Sort descending by completeness
    list.sort((a, b) => b.completeness - a.completeness);
    return list;
  }, [variables, searchFilter, typeFilter, selectedDomain]);

  const chartOption = useMemo(() => {
    if (ranked.length === 0) return null;
    const allVars = ranked;

    // Arrange in a grid (10 columns)
    const cols = 10;
    const rows = Math.ceil(allVars.length / cols);

    const data: [number, number, number, any][] = [];
    const xLabels: string[] = [];
    const yLabels: string[] = [];

    for (let i = 0; i < allVars.length; i++) {
      const d = allVars[i];
      const x = i % cols;
      const y = Math.floor(i / cols);
      data.push([x, y, d.completeness, { variable: d.variable, completeness: d.completeness, missingPct: d.missingPct, n: d.n }]);
      if (x === 0) yLabels.push(`Row ${y + 1}`);
    }
    for (let i = 0; i < cols; i++) xLabels.push(`Col ${i + 1}`);

    return {
      tooltip: {
        formatter: (params: any) => {
          const meta = params.data[3];
          return `<strong>${meta.variable.name}</strong><br/>` +
            `Label: ${meta.variable.label}<br/>` +
            `Concept Name: ${meta.variable.conceptName || '—'}<br/>` +
            `Domain: ${meta.variable.domain || '—'}<br/>` +
            `Type: ${meta.variable.type}<br/>` +
            `Completeness: ${meta.completeness.toFixed(1)}%<br/>` +
            `Missing: ${meta.missingPct.toFixed(1)}% · n=${meta.n}`;
        },
      },
      grid: { left: 60, right: 20, top: 30, bottom: 40 },
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
        max: 100,
        calculable: true,
        orient: 'horizontal' as const,
        left: 'center',
        bottom: 0,
        inRange: {
          color: ['#ef4444', '#f59e0b', '#10b981'],
        },
        text: ['100%', '0%'],
        textStyle: { fontSize: 10 },
        dimension: 2,
      },
      series: [{
        type: 'heatmap' as const,
        data,
        itemStyle: {
          borderWidth: 1,
          borderColor: '#fff',
        },
        label: {
          show: true,
          fontSize: 8,
          formatter: (p: any) => {
            const meta = p.data[3];
            return meta.variable.name.length > 12 ? meta.variable.name.substring(0, 10) + '...' : meta.variable.name;
          },
        },
        emphasis: {
          itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' },
        },
      }],
    };
  }, [ranked]);

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-lg font-bold">Variables Ranked by Data Completeness</h2>
        <p className="text-sm text-gray-500 mt-1">
          {ranked.length} variables sorted by coverage (highest first). Green ≥90%, yellow ≥70%, red &lt;70%.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-gray-500">{ranked.length} variables</span>
        <div className="join">
          <button className={`btn btn-sm join-item ${viewMode === 'table' ? 'btn-primary' : 'btn-outline'}`} onClick={() => setViewMode('table')}>Table</button>
          <button className={`btn btn-sm join-item ${viewMode === 'chart' ? 'btn-primary' : 'btn-outline'}`} onClick={() => setViewMode('chart')}>Heatmap</button>
        </div>
        <input
          type="text"
          placeholder="Filter..."
          className="input input-sm input-bordered w-48"
          value={searchFilter}
          onChange={e => setSearchFilter(e.target.value)}
        />
        <DomainFilterBar variables={variables} selectedDomain={selectedDomain} onChange={setSelectedDomain} />
        <select className="select select-sm select-bordered" value={typeFilter} onChange={e => setTypeFilter(e.target.value as any)}>
          <option value="all">All types</option>
          <option value="numeric">Numeric only</option>
          <option value="categorical">Categorical only</option>
        </select>
      </div>

      {/* Table view */}
      {viewMode === 'table' && (
        <div className="card bg-base-100 shadow-md">
          <div className="overflow-x-auto">
            <table className="table table-sm table-zebra table-pin-rows">
              <thead>
                <tr>
                  <th className="text-center w-12">#</th>
                  <th>Variable</th>
                  <th>Label</th>
                  <th>Type</th>
                  <th>n</th>
                  <th>Completeness</th>
                  <th>Missing %</th>
                  <th>Empty %</th>
                  <th className="w-32">Bar</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((d, i) => (
                  <tr key={d.variable.name} className="cursor-pointer hover:bg-primary/5" onClick={() => onVariableClick(d.variable)}>
                    <td className="text-center font-bold text-gray-400">{i + 1}</td>
                    <td className="font-medium text-xs max-w-[150px] truncate" title={d.variable.name}>{d.variable.name}</td>
                    <td className="text-xs max-w-[200px] truncate" title={d.variable.label}>{d.variable.label}</td>
                    <td className="text-xs">{d.variable.type}</td>
                    <td>{d.n}</td>
                    <td className={`font-medium ${d.completeness >= 90 ? 'text-green-600' : d.completeness >= 70 ? 'text-amber-600' : 'text-red-600'}`}>
                      {d.completeness.toFixed(1)}%
                    </td>
                    <td className="text-xs">{d.variable.countMissingPct.toFixed(1)}%</td>
                    <td className="text-xs">{d.variable.countEmptyPct.toFixed(1)}%</td>
                    <td>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                          className={`h-2.5 rounded-full ${d.completeness >= 90 ? 'bg-green-500' : d.completeness >= 70 ? 'bg-amber-500' : 'bg-red-500'}`}
                          style={{ width: `${d.completeness}%` }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Chart view */}
      {viewMode === 'chart' && chartOption && (
        <div className="card bg-base-100 shadow-md p-4">
          <p className="text-xs text-gray-500 mb-2">{ranked.length} variables arranged in grid. Color = completeness.</p>
          <ReactECharts
            option={chartOption}
            style={{ height: Math.max(400, Math.ceil(ranked.length / 10) * 50 + 80) }}
            onEvents={{
              click: (params: any) => {
                if (params.data && params.data[3]) {
                  onVariableClick(params.data[3].variable);
                }
              },
            }}
          />
        </div>
      )}
    </div>
  );
};

export default EdaCoverageRanking;
