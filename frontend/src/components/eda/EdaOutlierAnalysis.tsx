import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable } from '@/utils/edaParsing';

interface Props {
  variables: EdaVariable[];
  onVariableClick: (v: EdaVariable) => void;
}

const EdaOutlierAnalysis: React.FC<Props> = ({ variables, onVariableClick }) => {
  const [sortBy, setSortBy] = useState<'iqr_count' | 'iqr_pct' | 'z_count' | 'name'>('iqr_pct');

  // Only vars with outlier data
  const withOutliers = useMemo(() => {
    return variables
      .filter(v => (v.outliersIqr !== undefined && v.outliersIqr > 0) || (v.outliersZ !== undefined && v.outliersZ > 0))
      .sort((a, b) => {
        switch (sortBy) {
          case 'iqr_count': return (b.outliersIqr ?? 0) - (a.outliersIqr ?? 0);
          case 'iqr_pct': return (b.outliersIqrPct ?? 0) - (a.outliersIqrPct ?? 0);
          case 'z_count': return (b.outliersZ ?? 0) - (a.outliersZ ?? 0);
          default: return a.name.localeCompare(b.name);
        }
      });
  }, [variables, sortBy]);

  // Summary stats
  const summary = useMemo(() => {
    const totalVars = variables.length;
    const withAny = withOutliers.length;
    const avgIqrPct = withOutliers.reduce((s, v) => s + (v.outliersIqrPct ?? 0), 0) / (withOutliers.length || 1);
    const maxIqrPct = Math.max(...withOutliers.map(v => v.outliersIqrPct ?? 0), 0);
    return { totalVars, withAny, avgIqrPct, maxIqrPct };
  }, [variables, withOutliers]);

  // Bar chart: IQR outlier % per variable
  const outlierBarOption = useMemo(() => {
    const display = withOutliers.slice(0, 40);
    if (display.length === 0) return null;

    return {
      title: { text: 'Outlier Percentage by Variable (IQR method)', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const p = params[0];
          const v = display[p.dataIndex];
          return `<strong>${v.name}</strong><br/>${v.label}<br/>
            IQR Outliers: ${v.outliersIqr} (${v.outliersIqrPct?.toFixed(2)}%)<br/>
            Z-score Outliers: ${v.outliersZ ?? 'N/A'}<br/>
            Mean: ${v.mean?.toFixed(2)} | Std Dev: ${v.stdDev?.toFixed(2)}<br/>
            IQR: ${v.iqr} | Range: ${v.range}`;
        },
      },
      grid: { left: 160, right: 30, top: 40, bottom: 30 },
      xAxis: { type: 'value', name: 'Outlier %', axisLabel: { formatter: '{value}%' } },
      yAxis: {
        type: 'category',
        data: display.map(v => v.name),
        axisLabel: { fontSize: 9, width: 140, overflow: 'truncate' },
      },
      series: [{
        type: 'bar',
        data: display.map(v => ({
          value: v.outliersIqrPct ?? 0,
          itemStyle: {
            color: (v.outliersIqrPct ?? 0) > 5 ? '#ef4444' :
                   (v.outliersIqrPct ?? 0) > 2 ? '#f59e0b' : '#3b82f6',
          },
        })),
        barMaxWidth: 20,
      }],
    };
  }, [withOutliers]);

  // Scatter: IQR outlier count vs Z-score outlier count
  const scatterOption = useMemo(() => {
    const data = withOutliers
      .filter(v => v.outliersIqr !== undefined && v.outliersZ !== undefined)
      .map(v => ({
        value: [v.outliersIqr, v.outliersZ],
        name: v.name,
        label: v.label,
        iqrPct: v.outliersIqrPct,
      }));

    if (data.length === 0) return null;

    return {
      title: { text: 'IQR vs Z-Score Outlier Counts', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        formatter: (p: any) => `<strong>${p.data.name}</strong><br/>${p.data.label}<br/>IQR: ${p.data.value[0]} (${p.data.iqrPct?.toFixed(1)}%)<br/>Z-score: ${p.data.value[1]}`,
      },
      xAxis: { type: 'value', name: 'IQR Outliers', nameLocation: 'center', nameGap: 30 },
      yAxis: { type: 'value', name: 'Z-Score Outliers' },
      series: [{
        type: 'scatter',
        symbolSize: (d: any) => Math.max(8, Math.min(30, (d[0] + d[1]) / 2)),
        data,
        itemStyle: { color: '#8b5cf6', opacity: 0.7 },
        emphasis: { itemStyle: { color: '#6d28d9', borderColor: '#000', borderWidth: 1, opacity: 1 } },
      }],
      grid: { left: 60, right: 30, top: 40, bottom: 50 },
    };
  }, [withOutliers]);

  // Skewness vs outlier % bubble chart
  const skewOutlierOption = useMemo(() => {
    const data = variables
      .filter(v => v.skewness !== undefined && v.outliersIqrPct !== undefined && v.outliersIqrPct > 0)
      .map(v => ({
        value: [v.skewness, v.outliersIqrPct, v.totalObservations],
        name: v.name,
        label: v.label,
      }));

    if (data.length < 3) return null;

    return {
      title: { text: 'Skewness vs Outlier % (bubble = sample size)', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        formatter: (p: any) => `<strong>${p.data.name}</strong><br/>${p.data.label}<br/>Skewness: ${p.data.value[0]?.toFixed(2)}<br/>Outlier %: ${p.data.value[1]?.toFixed(2)}%<br/>n=${p.data.value[2]}`,
      },
      xAxis: { type: 'value', name: 'Skewness', nameLocation: 'center', nameGap: 30 },
      yAxis: { type: 'value', name: 'Outlier %', axisLabel: { formatter: '{value}%' } },
      series: [{
        type: 'scatter',
        symbolSize: (d: any) => Math.max(6, Math.min(25, Math.sqrt(d[2]) / 3)),
        data,
        itemStyle: { color: '#f59e0b', opacity: 0.6 },
        emphasis: { itemStyle: { color: '#d97706', opacity: 1 } },
      }],
      grid: { left: 60, right: 30, top: 40, bottom: 50 },
    };
  }, [variables]);

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="stats shadow w-full bg-base-100">
        <div className="stat">
          <div className="stat-title">Variables with Outliers</div>
          <div className="stat-value text-amber-500">{summary.withAny}</div>
          <div className="stat-desc">of {summary.totalVars} total numeric</div>
        </div>
        <div className="stat">
          <div className="stat-title">Avg Outlier %</div>
          <div className="stat-value text-blue-500">{summary.avgIqrPct.toFixed(2)}%</div>
          <div className="stat-desc">IQR method</div>
        </div>
        <div className="stat">
          <div className="stat-title">Max Outlier %</div>
          <div className="stat-value text-red-500">{summary.maxIqrPct.toFixed(2)}%</div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <select className="select select-sm select-bordered" value={sortBy} onChange={e => setSortBy(e.target.value as any)}>
          <option value="iqr_pct">Sort by IQR %</option>
          <option value="iqr_count">Sort by IQR Count</option>
          <option value="z_count">Sort by Z-Score Count</option>
          <option value="name">Sort by Name</option>
        </select>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {outlierBarOption && (
          <div className="card bg-base-100 shadow-md p-4">
            <ReactECharts
              option={outlierBarOption}
              style={{ height: Math.max(300, Math.min(withOutliers.length, 40) * 20 + 80) }}
              onEvents={{
                click: (params: any) => {
                  const v = withOutliers[params.dataIndex];
                  if (v) onVariableClick(v);
                },
              }}
            />
          </div>
        )}
        {scatterOption && (
          <div className="card bg-base-100 shadow-md p-4">
            <ReactECharts option={scatterOption} style={{ height: 400 }} />
          </div>
        )}
      </div>

      {skewOutlierOption && (
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts option={skewOutlierOption} style={{ height: 400 }} />
        </div>
      )}

      {/* Table of outlier details */}
      <div className="card bg-base-100 shadow-md">
        <div className="overflow-x-auto">
          <table className="table table-sm table-zebra">
            <thead>
              <tr>
                <th>Variable</th>
                <th>Label</th>
                <th>IQR Outliers</th>
                <th>IQR %</th>
                <th>Z Outliers</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>IQR</th>
                <th>Skewness</th>
              </tr>
            </thead>
            <tbody>
              {withOutliers.slice(0, 50).map(v => (
                <tr key={v.name} className="cursor-pointer hover:bg-primary/5" onClick={() => onVariableClick(v)}>
                  <td className="font-medium text-xs max-w-[120px] truncate">{v.name}</td>
                  <td className="text-xs max-w-[180px] truncate">{v.label}</td>
                  <td>{v.outliersIqr ?? '—'}</td>
                  <td className={`font-medium ${(v.outliersIqrPct ?? 0) > 5 ? 'text-red-500' : (v.outliersIqrPct ?? 0) > 2 ? 'text-amber-500' : ''}`}>
                    {v.outliersIqrPct?.toFixed(2) ?? '—'}%
                  </td>
                  <td>{v.outliersZ ?? '—'}</td>
                  <td className="text-xs">{v.mean?.toFixed(2) ?? '—'}</td>
                  <td className="text-xs">{v.stdDev?.toFixed(2) ?? '—'}</td>
                  <td>{v.iqr ?? '—'}</td>
                  <td className="text-xs">{v.skewness?.toFixed(2) ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default EdaOutlierAnalysis;
