import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable } from '@/utils/edaParsing';
import DomainFilterBar from './DomainFilterBar';

interface Props {
  variables: EdaVariable[];
  onVariableClick: (v: EdaVariable) => void;
  metric?: 'iqr' | 'z';
}

const EdaOutlierAnalysis: React.FC<Props> = ({ variables, onVariableClick, metric = 'iqr' }) => {
  const [sortBy, setSortBy] = useState<'count' | 'pct' | 'name'>(metric === 'iqr' ? 'pct' : 'count');
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);

  // Only vars with outlier data for the selected metric
  const withOutliers = useMemo(() => {
    let list = variables
      .filter(v => metric === 'iqr' ? (v.outliersIqr !== undefined && v.outliersIqr > 0) : (v.outliersZ !== undefined && v.outliersZ > 0));
    if (selectedDomain) list = list.filter(v => v.domain === selectedDomain);
    return list.sort((a, b) => {
      switch (sortBy) {
        case 'count':
          if (metric === 'iqr') return (b.outliersIqr ?? 0) - (a.outliersIqr ?? 0);
          return (b.outliersZ ?? 0) - (a.outliersZ ?? 0);
        case 'pct':
          if (metric === 'iqr') return (b.outliersIqrPct ?? 0) - (a.outliersIqrPct ?? 0);
          // Calculate Z-score percentage from count and totalObservations
          const aZPct = a.totalObservations ? (a.outliersZ ?? 0) / a.totalObservations * 100 : 0;
          const bZPct = b.totalObservations ? (b.outliersZ ?? 0) / b.totalObservations * 100 : 0;
          return bZPct - aZPct;
        default: return a.name.localeCompare(b.name);
      }
    });
  }, [variables, sortBy, selectedDomain, metric]);

  // Summary stats
  const summary = useMemo(() => {
    const totalVars = variables.length;
    const withAny = withOutliers.length;
    if (metric === 'iqr') {
      const avgPct = withOutliers.reduce((s, v) => s + (v.outliersIqrPct ?? 0), 0) / (withOutliers.length || 1);
      const maxPct = Math.max(...withOutliers.map(v => v.outliersIqrPct ?? 0), 0);
      return { totalVars, withAny, avgPct, maxPct };
    } else {
      const avgPct = withOutliers.reduce((s, v) => s + (v.totalObservations ? (v.outliersZ ?? 0) / v.totalObservations * 100 : 0), 0) / (withOutliers.length || 1);
      const maxPct = Math.max(...withOutliers.map(v => v.totalObservations ? (v.outliersZ ?? 0) / v.totalObservations * 100 : 0), 0);
      return { totalVars, withAny, avgPct, maxPct };
    }
  }, [variables, withOutliers, metric]);

  // Bar chart: outlier % per variable
  const outlierBarOption = useMemo(() => {
    const display = withOutliers.slice(0, 40);
    if (display.length === 0) return null;

    const title = metric === 'iqr' ? 'Outlier Percentage by Variable (IQR method)' : 'Outlier Percentage by Variable (Z-score method)';
    const getPct = (v: any) => metric === 'iqr' ? v.outliersIqrPct : (v.totalObservations ? (v.outliersZ ?? 0) / v.totalObservations * 100 : 0);
    const getCount = (v: any) => metric === 'iqr' ? v.outliersIqr : v.outliersZ;

    return {
      title: { text: title, left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const p = params[0];
          const v = display[p.dataIndex];
          const pct = getPct(v);
          const count = getCount(v);
          const methodName = metric === 'iqr' ? 'IQR' : 'Z-score';
          return `<strong>${v.name}</strong><br/>${v.label}<br/>
            ${methodName} Outliers: ${count} (${pct.toFixed(2)}%)<br/>
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
        data: display.map(v => {
          const pct = getPct(v);
          return {
            value: pct,
            itemStyle: {
              color: pct > 5 ? '#ef4444' : pct > 2 ? '#f59e0b' : '#3b82f6',
            },
          };
        }),
        barMaxWidth: 20,
      }],
    };
  }, [withOutliers, metric]);

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="stats shadow w-full bg-base-100">
        <DomainFilterBar variables={variables} selectedDomain={selectedDomain} onChange={setSelectedDomain} />
        <div className="stat">
          <div className="stat-title">Variables with Outliers</div>
          <div className="stat-value text-amber-500">{summary.withAny}</div>
          <div className="stat-desc">of {summary.totalVars} total numeric</div>
        </div>
        <div className="stat">
          <div className="stat-title">Avg Outlier %</div>
          <div className="stat-value text-amber-500">{summary.avgPct.toFixed(1)}%</div>
        </div>
        <div className="stat">
          <div className="stat-title">Max Outlier %</div>
          <div className="stat-value text-amber-500">{summary.maxPct.toFixed(1)}%</div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <select className="select select-sm select-bordered" value={sortBy} onChange={e => setSortBy(e.target.value as any)}>
          <option value="pct">Sort by %</option>
          <option value="count">Sort by Count</option>
          <option value="name">Sort by Name</option>
        </select>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-1 gap-4">
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
      </div>

      {/* Table of outlier details */}
      <div className="card bg-base-100 shadow-md">
        <div className="overflow-x-auto">
          <table className="table table-sm table-zebra">
            <thead>
              <tr>
                <th>Variable</th>
                <th>Label</th>
                <th>{metric === 'iqr' ? 'IQR Outliers' : 'Z-Score Outliers'}</th>
                <th>{metric === 'iqr' ? 'IQR %' : 'Z-Score %'}</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>IQR</th>
                <th>Skewness</th>
              </tr>
            </thead>
            <tbody>
              {withOutliers.slice(0, 50).map(v => (
                <tr key={v.name} className="cursor-pointer hover:bg-primary/5" onClick={() => onVariableClick(v)}>
                  <td className="font-semibold">{v.name}</td>
                  <td>{v.label}</td>
                  <td>{metric === 'iqr' ? v.outliersIqr : v.outliersZ}</td>
                  <td>{metric === 'iqr' ? v.outliersIqrPct?.toFixed(2) : (v.totalObservations ? (v.outliersZ ?? 0) / v.totalObservations * 100 : 0).toFixed(2)}%</td>
                  <td>{v.mean?.toFixed(2)}</td>
                  <td>{v.stdDev?.toFixed(2)}</td>
                  <td>{v.iqr}</td>
                  <td className="text-xs">{v.skewness?.toFixed(2)}</td>
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
