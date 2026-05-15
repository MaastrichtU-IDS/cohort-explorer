import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable } from '@/utils/edaParsing';
import DomainFilterBar from './DomainFilterBar';

interface Props {
  variables: EdaVariable[];
  onVariableClick: (v: EdaVariable) => void;
}

type ViewMode = 'table' | 'chart';

const EdaCVRanking: React.FC<Props> = ({ variables, onVariableClick }) => {
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);

  const ranked = useMemo(() => {
    let list = variables
      .filter(v => v.type === 'numeric' && v.mean !== undefined && v.mean !== 0 && v.stdDev !== undefined)
      .map(v => {
        const stdDev = v.stdDev!;
        const mean = v.mean!;
        const cv = (stdDev / Math.abs(mean)) * 100;
        return {
          variable: v,
          cv,
          mean,
          stdDev,
          variance: v.variance,
        };
      });

    if (selectedDomain) list = list.filter(d => d.variable.domain === selectedDomain);
    if (searchFilter) {
      const q = searchFilter.toLowerCase();
      list = list.filter(d => d.variable.name.toLowerCase().includes(q) || d.variable.label.toLowerCase().includes(q) || (d.variable.conceptName && d.variable.conceptName.toLowerCase().includes(q)));
    }

    // Sort descending by CV
    list.sort((a, b) => b.cv - a.cv);
    return list;
  }, [variables, searchFilter, selectedDomain]);

  const chartOption = useMemo(() => {
    if (ranked.length === 0) return null;
    const allVars = ranked;
    const maxCv = Math.max(...allVars.map(d => d.cv), 1);

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
      data.push([x, y, d.cv, { variable: d.variable, cv: d.cv, mean: d.mean, stdDev: d.stdDev }]);
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
            `CV: ${meta.cv.toFixed(1)}%<br/>` +
            `Mean: ${meta.mean.toFixed(2)} · Std Dev: ${meta.stdDev.toFixed(2)}<br/>` +
            `n=${meta.variable.totalObservations}`;
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
        max: maxCv,
        calculable: true,
        orient: 'horizontal' as const,
        left: 'center',
        bottom: 0,
        inRange: {
          color: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'],
        },
        text: [`${maxCv.toFixed(0)}%`, '0%'],
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

  if (ranked.length === 0) {
    return (
      <div className="alert alert-info max-w-xl mx-auto my-8">
        <span>No numeric variables with both mean and stdDev available for CV calculation.</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-lg font-bold">Variables Ranked by Coefficient of Variation</h2>
        <p className="text-sm text-gray-500 mt-1">
          {ranked.length} numeric variables sorted by CV (highest first). CV = StdDev/|Mean| × 100%.
          Higher CV = more relative variability.
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
                  <th>n</th>
                  <th>Mean</th>
                  <th>Std Dev</th>
                  <th>Variance</th>
                  <th>CV %</th>
                  <th>Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((d, i) => (
                  <tr key={d.variable.name} className="cursor-pointer hover:bg-primary/5" onClick={() => onVariableClick(d.variable)}>
                    <td className="text-center font-bold text-gray-400">{i + 1}</td>
                    <td className="font-medium text-xs max-w-[150px] truncate" title={d.variable.name}>{d.variable.name}</td>
                    <td className="text-xs max-w-[180px] truncate" title={d.variable.label}>{d.variable.label}</td>
                    <td>{d.variable.totalObservations}</td>
                    <td className="text-xs">{d.mean.toFixed(2)}</td>
                    <td className="text-xs">{d.stdDev.toFixed(2)}</td>
                    <td className="text-xs">{d.variance?.toFixed(2) ?? '—'}</td>
                    <td className={`font-medium ${d.cv >= 50 ? 'text-red-600' : d.cv >= 30 ? 'text-amber-600' : d.cv >= 15 ? 'text-blue-600' : 'text-green-600'}`}>
                      {d.cv.toFixed(1)}%
                    </td>
                    <td className="text-xs">
                      {d.cv >= 50 ? 'Very high variability' : d.cv >= 30 ? 'High variability' : d.cv >= 15 ? 'Moderate variability' : 'Low variability'}
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
          <p className="text-xs text-gray-500 mb-2">{ranked.length} variables arranged in grid. Color = CV.</p>
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

      {/* Explanation */}
      <div className="card bg-base-200 p-4 text-sm">
        <h3 className="font-semibold mb-2">What is Coefficient of Variation?</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div><strong>CV &lt; 15%:</strong> Low relative variability — data is tightly clustered around the mean</div>
          <div><strong>CV 15-30%:</strong> Moderate variability — typical for many biological measurements</div>
          <div><strong>CV 30-50%:</strong> High variability — considerable spread relative to the mean</div>
          <div><strong>CV &gt; 50%:</strong> Very high variability — mean may not be representative; consider transformation</div>
          <div><strong>Formula:</strong> CV = (StdDev / |Mean|) × 100%</div>
          <div><strong>Use case:</strong> Unit-free measure of relative spread; enables comparison across variables with different units</div>
        </div>
      </div>
    </div>
  );
};

export default EdaCVRanking;
