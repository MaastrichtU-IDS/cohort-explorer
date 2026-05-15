import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable } from '@/utils/edaParsing';
import DomainFilterBar from './DomainFilterBar';

interface Props {
  variables: EdaVariable[];
  onVariableClick: (v: EdaVariable) => void;
}

type OutlierType = 'iqr' | 'zscore';
type ViewMode = 'table' | 'chart';

const EdaOutlierRanking: React.FC<Props> = ({ variables, onVariableClick }) => {
  const [outlierType, setOutlierType] = useState<OutlierType>('iqr');
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);

  const ranked = useMemo(() => {
    let list = variables
      .filter(v => v.type === 'numeric')
      .map(v => {
        const iqrCount = v.outliersIqr ?? 0;
        const iqrPct = v.outliersIqrPct ?? (v.totalObservations > 0 ? (iqrCount / v.totalObservations) * 100 : 0);
        const zCount = v.outliersZ ?? 0;
        const zPct = v.totalObservations > 0 ? (zCount / v.totalObservations) * 100 : 0;
        const cv = (v.mean && v.mean !== 0 && v.stdDev !== undefined) ? (v.stdDev / Math.abs(v.mean)) * 100 : undefined;

        return {
          variable: v,
          iqrCount,
          iqrPct,
          zCount,
          zPct,
          cv,
        };
      });

    if (selectedDomain) list = list.filter(d => d.variable.domain === selectedDomain);
    if (searchFilter) {
      const q = searchFilter.toLowerCase();
      list = list.filter(d => d.variable.name.toLowerCase().includes(q) || d.variable.label.toLowerCase().includes(q) || (d.variable.conceptName && d.variable.conceptName.toLowerCase().includes(q)));
    }

    // Sort descending by selected outlier percentage
    list.sort((a, b) => {
      if (outlierType === 'iqr') return b.iqrPct - a.iqrPct;
      return b.zPct - a.zPct;
    });

    return list;
  }, [variables, searchFilter, outlierType, selectedDomain]);

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
      const value = outlierType === 'iqr' ? d.iqrPct : d.zPct;
      data.push([x, y, value, { variable: d.variable, iqrCount: d.iqrCount, iqrPct: d.iqrPct, zCount: d.zCount, zPct: d.zPct, cv: d.cv }]);
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
            `Outliers (IQR): ${meta.iqrCount} (${meta.iqrPct.toFixed(1)}%)<br/>` +
            `Outliers (Z): ${meta.zCount} (${meta.zPct.toFixed(1)}%)<br/>` +
            `CV: ${meta.cv?.toFixed(1) ?? 'N/A'}%<br/>` +
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
        max: 15,
        calculable: true,
        orient: 'horizontal' as const,
        left: 'center',
        bottom: 0,
        inRange: {
          color: ['#10b981', '#f59e0b', '#ef4444'],
        },
        text: ['15%', '0%'],
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
  }, [ranked, outlierType]);

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-lg font-bold">Variables Ranked by Outlier Percentage</h2>
        <p className="text-sm text-gray-500 mt-1">
          {ranked.length} numeric variables sorted by outlier % (highest first). Uses percentage for fair cross-variable comparison.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-gray-500">{ranked.length} variables</span>
        <div className="join">
          <button className={`btn btn-sm join-item ${viewMode === 'table' ? 'btn-primary' : 'btn-outline'}`} onClick={() => setViewMode('table')}>Table</button>
          <button className={`btn btn-sm join-item ${viewMode === 'chart' ? 'btn-primary' : 'btn-outline'}`} onClick={() => setViewMode('chart')}>Heatmap</button>
        </div>
        <div className="join">
          <button className={`btn btn-sm join-item ${outlierType === 'iqr' ? 'btn-secondary' : 'btn-outline'}`} onClick={() => setOutlierType('iqr')}>IQR Method</button>
          <button className={`btn btn-sm join-item ${outlierType === 'zscore' ? 'btn-secondary' : 'btn-outline'}`} onClick={() => setOutlierType('zscore')}>Z-Score Method</button>
        </div>
        <input
          type="text"
          placeholder="Filter..."
          className="input input-sm input-bordered w-48"
          value={searchFilter}
          onChange={e => setSearchFilter(e.target.value)}
        />
        <DomainFilterBar variables={variables} selectedDomain={selectedDomain} onChange={d => setSelectedDomain(d)} />
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
                  <th>Outliers (IQR)</th>
                  <th>IQR %</th>
                  <th>Outliers (Z)</th>
                  <th>Z %</th>
                  <th>CV %</th>
                  <th>IQR</th>
                  <th>Std Dev</th>
                  <th>Skewness</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((d, i) => (
                  <tr key={d.variable.name} className="cursor-pointer hover:bg-primary/5" onClick={() => onVariableClick(d.variable)}>
                    <td className="text-center font-bold text-gray-400">{i + 1}</td>
                    <td className="font-medium text-xs max-w-[150px] truncate" title={d.variable.name}>{d.variable.name}</td>
                    <td className="text-xs max-w-[180px] truncate" title={d.variable.label}>{d.variable.label}</td>
                    <td>{d.variable.totalObservations}</td>
                    <td className="font-medium">{d.iqrCount}</td>
                    <td className={`font-medium ${d.iqrPct >= 10 ? 'text-red-600' : d.iqrPct >= 5 ? 'text-amber-600' : ''}`}>
                      {d.iqrPct.toFixed(1)}%
                    </td>
                    <td className="font-medium">{d.zCount}</td>
                    <td className={`font-medium ${d.zPct >= 10 ? 'text-red-600' : d.zPct >= 5 ? 'text-amber-600' : ''}`}>
                      {d.zPct.toFixed(1)}%
                    </td>
                    <td className="text-xs">{d.cv?.toFixed(1) ?? '—'}</td>
                    <td className="text-xs">{d.variable.iqr ?? '—'}</td>
                    <td className="text-xs">{d.variable.stdDev?.toFixed(2) ?? '—'}</td>
                    <td className="text-xs">{d.variable.skewness?.toFixed(2) ?? '—'}</td>
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
          <p className="text-xs text-gray-500 mb-2">{ranked.length} variables arranged in grid. Color = outlier %.</p>
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
        <h3 className="font-semibold mb-2">Metrics Explained</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div><strong>Outliers (IQR):</strong> Values below Q1−1.5×IQR or above Q3+1.5×IQR</div>
          <div><strong>Outliers (Z-Score):</strong> Values with |z| &gt; 3 (more than 3 SD from mean)</div>
          <div><strong>CV (Coefficient of Variation):</strong> StdDev/|Mean| × 100% — unit-free relative spread</div>
          <div><strong>Why %?</strong> Raw counts aren&apos;t comparable across variables with different n</div>
        </div>
      </div>
    </div>
  );
};

export default EdaOutlierRanking;
