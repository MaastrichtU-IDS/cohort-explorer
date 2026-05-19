import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable } from '@/utils/edaParsing';
import DomainFilterBar from './DomainFilterBar';

interface Props {
  variables: EdaVariable[];
  onVariableClick: (v: EdaVariable) => void;
}

/**
 * Calculate Gini coefficient for class imbalance
 * Gini = 1 - sum(p_i^2) where p_i is the proportion of each class
 * Range: 0 (perfectly balanced) to 1 (completely imbalanced)
 */
function calculateGini(classBalance: { percentage: number }[]): number {
  if (!classBalance || classBalance.length === 0) return 0;
  const sumSquares = classBalance.reduce((sum, c) => sum + Math.pow(c.percentage / 100, 2), 0);
  return 1 - sumSquares;
}

const EdaImbalanceRanking: React.FC<Props> = ({ variables, onVariableClick }) => {
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);

  // Filter categorical variables with class balance data
  const categoricalVars = useMemo(() => {
    let list = variables
      .filter(v => v.type === 'categorical' && v.classBalance && v.classBalance.length > 1);
    if (selectedDomain) list = list.filter(v => v.domain === selectedDomain);
    return list;
  }, [variables, selectedDomain]);

  // Calculate Gini coefficient for each variable and sort
  const rankedVars = useMemo(() => {
    return categoricalVars
      .map(v => ({
        variable: v,
        gini: calculateGini(v.classBalance || []),
        maxClassPct: Math.max(...(v.classBalance || []).map(c => c.percentage)),
        minClassPct: Math.min(...(v.classBalance || []).map(c => c.percentage)),
        numClasses: v.classBalance?.length || 0,
      }))
      .sort((a, b) => b.gini - a.gini);
  }, [categoricalVars]);

  // Heatmap option
  const chartOption = useMemo(() => {
    if (rankedVars.length === 0) return null;

    const display = rankedVars.slice(0, 50);
    const cols = 10;
    const rows = Math.ceil(display.length / cols);

    const data: [number, number, number, any][] = [];
    const xLabels: string[] = [];
    const yLabels: string[] = [];

    for (let i = 0; i < display.length; i++) {
      const d = display[i];
      const x = i % cols;
      const y = Math.floor(i / cols);
      data.push([x, y, d.gini, { variable: d.variable, gini: d.gini, maxClassPct: d.maxClassPct, minClassPct: d.minClassPct, numClasses: d.numClasses }]);
      if (x === 0) yLabels.push(`Row ${y + 1}`);
    }
    for (let i = 0; i < cols; i++) xLabels.push(`Col ${i + 1}`);

    return {
      title: { text: 'Class Imbalance Heatmap (Gini Coefficient)', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        formatter: (params: any) => {
          const meta = params.data[3];
          return `<strong>${meta.variable.name}</strong><br/>Label: ${meta.variable.label}<br/>Concept Name: ${meta.variable.conceptName || '—'}<br/>Domain: ${meta.variable.domain || '—'}<br/>Type: ${meta.variable.type}<br/>Gini: ${meta.gini.toFixed(3)}<br/>Max Class: ${meta.maxClassPct.toFixed(1)}%<br/>Min Class: ${meta.minClassPct.toFixed(1)}%<br/>Num Classes: ${meta.numClasses}`;
        },
      },
      grid: { left: 60, right: 20, top: 40, bottom: 40 },
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
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 10,
        dimension: 2,
        inRange: {
          color: ['#90EE90', '#FFD700', '#FF6347', '#8B0000'],
        },
      },
      series: [{
        type: 'heatmap',
        data,
        label: { show: false },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)',
          },
        },
      }],
    };
  }, [rankedVars]);

  if (categoricalVars.length === 0) {
    return (
      <div className="card bg-base-100 shadow-md p-6">
        <p className="text-center text-gray-500">No categorical variables with class balance data available.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Domain filter */}
      <DomainFilterBar variables={categoricalVars} selectedDomain={selectedDomain} onChange={setSelectedDomain} />

      {/* Summary stats */}
      <div className="stats shadow w-full bg-base-100">
        <div className="stat">
          <div className="stat-title">Categorical Variables</div>
          <div className="stat-value text-amber-500">{categoricalVars.length}</div>
        </div>
        <div className="stat">
          <div className="stat-title">Avg Gini</div>
          <div className="stat-value text-amber-500">{(rankedVars.reduce((s, v) => s + v.gini, 0) / rankedVars.length).toFixed(3)}</div>
        </div>
        <div className="stat">
          <div className="stat-title">Max Gini</div>
          <div className="stat-value text-amber-500">{rankedVars[0]?.gini.toFixed(3) || '0'}</div>
        </div>
      </div>

      {/* Heatmap */}
      {chartOption && (
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts
            option={chartOption}
            style={{ height: Math.max(300, Math.min(rankedVars.length, 50) * 20 + 80) }}
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

      {/* Table of imbalance details */}
      <div className="card bg-base-100 shadow-md">
        <div className="overflow-x-auto">
          <table className="table table-sm table-zebra">
            <thead>
              <tr>
                <th>Variable</th>
                <th>Label</th>
                <th>Gini</th>
                <th>Max Class %</th>
                <th>Min Class %</th>
                <th>Num Classes</th>
                <th>Most Frequent</th>
              </tr>
            </thead>
            <tbody>
              {rankedVars.slice(0, 50).map(v => (
                <tr key={v.variable.name} className="cursor-pointer hover:bg-primary/5" onClick={() => onVariableClick(v.variable)}>
                  <td className="font-semibold">{v.variable.name}</td>
                  <td>{v.variable.label}</td>
                  <td className={v.gini > 0.7 ? 'text-red-500 font-bold' : v.gini > 0.4 ? 'text-amber-500' : 'text-green-500'}>{v.gini.toFixed(3)}</td>
                  <td>{v.maxClassPct.toFixed(1)}%</td>
                  <td>{v.minClassPct.toFixed(1)}%</td>
                  <td>{v.numClasses}</td>
                  <td>{v.variable.mostFrequentCategory || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default EdaImbalanceRanking;
