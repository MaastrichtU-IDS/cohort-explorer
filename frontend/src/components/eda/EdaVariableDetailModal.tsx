import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaVariable, completenessScore } from '@/utils/edaParsing';

interface Props {
  variable: EdaVariable;
  onClose: () => void;
  cohortId: string;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];

const EdaVariableDetailModal: React.FC<Props> = ({ variable: v, onClose, cohortId }) => {
  const [imageState, setImageState] = useState<'loading' | 'loaded' | 'error'>('loading');
  const score = completenessScore(v);

  // Box plot chart for numeric variables
  const boxPlotOption = useMemo(() => {
    if (v.type !== 'numeric') return null;

    const min = v.min ?? 0;
    const q1 = v.q1 ?? 0;
    const median = v.median ?? 0;
    const q3 = v.q3 ?? 0;
    const max = v.max ?? 0;
    const mean = v.mean ?? 0;
    const iqr = v.iqr ?? (q3 - q1);

    // Whisker bounds (1.5 * IQR)
    const lowerWhisker = Math.max(min, q1 - 1.5 * iqr);
    const upperWhisker = Math.min(max, q3 + 1.5 * iqr);

    // Outlier points (approximate: we know count but not exact values, so show min/max if outside whiskers)
    const outlierPoints: number[][] = [];
    if (min < lowerWhisker) outlierPoints.push([min, 0]);
    if (max > upperWhisker) outlierPoints.push([max, 0]);

    // Use x-tick range for axis if available
    const xTickNums = v.xTicksNumeric ?? [];
    const axisMin = xTickNums.length > 0 ? Math.min(...xTickNums) : Math.min(min, lowerWhisker) - (max - min) * 0.05;
    const axisMax = xTickNums.length > 0 ? Math.max(...xTickNums) : Math.max(max, upperWhisker) + (max - min) * 0.05;

    return {
      tooltip: {
        trigger: 'item',
        formatter: () =>
          `<strong>${v.name}</strong><br/>` +
          `Min: ${min} &nbsp; Q1: ${q1}<br/>` +
          `Median: <strong>${median}</strong> &nbsp; Mean: <strong>${mean.toFixed(2)}</strong><br/>` +
          `Q3: ${q3} &nbsp; Max: ${max}<br/>` +
          `IQR: ${iqr} &nbsp; Range: ${v.range}<br/>` +
          `Std Dev: ${v.stdDev?.toFixed(2)} &nbsp; Skewness: ${v.skewness?.toFixed(2)}<br/>` +
          (v.yTickMax ? `Peak histogram count: ~${v.yTickMax}` : ''),
      },
      xAxis: { type: 'value' as const, min: axisMin, max: axisMax, name: v.units || '', nameLocation: 'center' as const, nameGap: 25 },
      yAxis: { type: 'category' as const, data: [''], axisLine: { show: false }, axisTick: { show: false }, axisLabel: { show: false } },
      series: [
        {
          type: 'boxplot' as const,
          data: [[lowerWhisker, q1, median, q3, upperWhisker]],
          itemStyle: { color: '#dbeafe', borderColor: '#3b82f6', borderWidth: 2 },
          boxWidth: ['40%', '40%'],
        },
        // Mean marker
        {
          type: 'scatter' as const,
          data: [[mean, 0]],
          symbolSize: 12,
          symbol: 'diamond',
          itemStyle: { color: '#ef4444', borderColor: '#fff', borderWidth: 1 },
          z: 10,
          tooltip: { formatter: () => `Mean: ${mean.toFixed(2)}` },
        },
        // Outlier points
        ...(outlierPoints.length > 0 ? [{
          type: 'scatter' as const,
          data: outlierPoints,
          symbolSize: 8,
          itemStyle: { color: '#f59e0b' },
          tooltip: { formatter: (p: any) => `Outlier: ${p.data[0]}` },
        }] : []),
      ],
      grid: { left: 20, right: 30, top: 15, bottom: 35 },
    };
  }, [v]);

  // Bar chart for categorical variables
  const categoricalChartOption = useMemo(() => {
    if (v.type !== 'categorical' || !v.classBalance) return null;
    const categories = v.classBalance;
    return {
      tooltip: {
        trigger: 'axis' as const,
        axisPointer: { type: 'shadow' as const },
        formatter: (params: any) => {
          const p = params[0];
          const cat = categories[p.dataIndex];
          return `<strong>${cat.label}</strong><br/>${cat.count} observations (${cat.percentage.toFixed(1)}%)`;
        },
      },
      xAxis: {
        type: 'category' as const,
        data: categories.map(c => c.label),
        axisLabel: { rotate: categories.length > 6 ? 45 : 0, fontSize: 11, interval: 0 },
      },
      yAxis: { type: 'value' as const, name: 'Count' },
      series: [{
        type: 'bar' as const,
        data: categories.map((c, i) => ({
          value: c.count,
          itemStyle: { color: COLORS[i % COLORS.length] },
        })),
        barMaxWidth: 60,
        label: { show: true, position: 'top' as const, fontSize: 10, formatter: (p: any) => `${categories[p.dataIndex].percentage.toFixed(1)}%` },
      }],
      grid: { left: 50, right: 20, top: 20, bottom: categories.some(c => c.label.length > 5) ? 80 : 40 },
    };
  }, [v]);

  // Donut chart for completeness
  const completenessOption = useMemo(() => ({
    series: [{
      type: 'pie',
      radius: ['55%', '75%'],
      label: { show: false },
      data: [
        { value: score, name: 'Complete', itemStyle: { color: score >= 80 ? '#10b981' : score >= 50 ? '#f59e0b' : '#ef4444' } },
        { value: 100 - score, name: 'Incomplete', itemStyle: { color: '#e5e7eb' } },
      ],
    }],
    graphic: [{
      type: 'text',
      left: 'center',
      top: 'center',
      style: {
        text: `${score.toFixed(0)}%`,
        fontSize: 18,
        fontWeight: 'bold',
        fill: score >= 80 ? '#10b981' : score >= 50 ? '#f59e0b' : '#ef4444',
      },
    }],
  }), [score]);

  const imageUrl = `/api/variable-graph/${encodeURIComponent(cohortId)}/${encodeURIComponent(v.name.toLowerCase())}`;

  return (
    <div className="modal modal-open" onClick={onClose}>
      <div className="modal-box max-w-5xl max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-xl font-bold">{v.name}</h2>
            <p className="text-gray-500">{v.label}</p>
            <div className="flex gap-2 mt-2">
              <span className={`badge ${v.type === 'numeric' ? 'badge-primary' : v.type === 'categorical' ? 'badge-success' : 'badge-warning'}`}>
                {v.rawType}
              </span>
              {v.units && <span className="badge badge-outline">{v.units}</span>}
              {v.domain && <span className="badge badge-ghost">{v.domain}</span>}
              {v.visit && <span className="badge badge-info badge-outline">{v.visit}</span>}
            </div>
          </div>
          <button onClick={onClose} className="btn btn-sm btn-circle">✕</button>
        </div>

        {/* Main content: chart + stats side by side */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Charts */}
          <div className="lg:col-span-2 space-y-3">
            {/* Box plot for numeric */}
            {boxPlotOption && (
              <div className="card bg-base-200 p-3">
                <h3 className="font-semibold text-sm mb-1">Box Plot</h3>
                <p className="text-xs text-gray-500 mb-1">
                  Box = IQR (Q1–Q3), line = median, <span className="text-red-500">◆</span> = mean
                  {v.outliersIqr ? `, ${v.outliersIqr} outliers (${v.outliersIqrPct?.toFixed(1)}%)` : ''}
                  {v.yTickMax ? ` · Original histogram peak: ~${v.yTickMax} observations` : ''}
                </p>
                <ReactECharts option={boxPlotOption} style={{ height: 120 }} />
              </div>
            )}

            {/* Bar chart for categorical */}
            {categoricalChartOption && (
              <div className="card bg-base-200 p-3">
                <h3 className="font-semibold text-sm mb-2">Category Distribution</h3>
                <ReactECharts option={categoricalChartOption} style={{ height: 280 }} />
              </div>
            )}

            {/* Original EDA graph — shown inline */}
            <div className="card bg-base-200 p-3">
              <h3 className="font-semibold text-sm mb-2">Original EDA Graph</h3>
              <div className="flex justify-center min-h-[80px] bg-white rounded-lg p-2">
                {imageState === 'loading' && <span className="loading loading-spinner loading-md"></span>}
                <img
                  src={imageUrl}
                  alt={`${v.name} EDA graph`}
                  className={`max-w-full ${imageState === 'loaded' ? '' : 'hidden'}`}
                  onLoad={() => setImageState('loaded')}
                  onError={() => setImageState('error')}
                  crossOrigin="use-credentials"
                />
                {imageState === 'error' && <p className="text-gray-500 text-sm self-center">Original graph not available</p>}
              </div>
            </div>
          </div>

          {/* Stats panel */}
          <div className="space-y-3">
            {/* Completeness ring */}
            <div className="card bg-base-200 p-3">
              <h4 className="font-semibold text-sm mb-1">Data Completeness</h4>
              <ReactECharts option={completenessOption} style={{ height: 120 }} />
              <div className="text-center text-xs space-y-0.5">
                <div>Empty: {v.countEmpty} ({v.countEmptyPct.toFixed(1)}%)</div>
                <div>Missing: {v.countMissing} ({v.countMissingPct.toFixed(1)}%)</div>
                <div>Total observations: {v.totalObservations}</div>
              </div>
            </div>

            {/* Numeric stats */}
            {v.type === 'numeric' && (
              <div className="card bg-base-200 p-3">
                <h4 className="font-semibold text-sm mb-2">Descriptive Statistics</h4>
                <table className="table table-xs">
                  <tbody>
                    {[
                      ['Mean', v.mean?.toFixed(2)],
                      ['Median', v.median],
                      ['Mode', v.mode],
                      ['Std Dev', v.stdDev?.toFixed(2)],
                      ['Variance', v.variance?.toFixed(2)],
                      ['Min', v.min],
                      ['Max', v.max],
                      ['Range', v.range],
                      ['Q1', v.q1],
                      ['Q3', v.q3],
                      ['IQR', v.iqr],
                      ['Skewness', v.skewness?.toFixed(3)],
                      ['Kurtosis', v.kurtosis?.toFixed(3)],
                      ['W-test', v.wTest?.toFixed(4)],
                      ['Normality', v.isNormal ? 'Normal' : 'Non-normal'],
                      ['Outliers (IQR)', `${v.outliersIqr ?? 0} (${v.outliersIqrPct?.toFixed(2) ?? 0}%)`],
                      ['Outliers (Z)', v.outliersZ],
                    ].map(([label, value]) => (
                      <tr key={label as string}>
                        <td className="font-medium">{label}</td>
                        <td className="text-right">{value ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Categorical stats */}
            {v.type === 'categorical' && (
              <div className="card bg-base-200 p-3">
                <h4 className="font-semibold text-sm mb-2">Categorical Statistics</h4>
                <table className="table table-xs">
                  <tbody>
                    <tr><td className="font-medium">Categories</td><td className="text-right">{v.uniqueValues}</td></tr>
                    <tr><td className="font-medium">Most Frequent</td><td className="text-right">{v.mostFrequentCategory || '—'}</td></tr>
                    <tr><td className="font-medium">Chi-Square</td><td className="text-right">{v.chiSquare?.toFixed(2) ?? '—'}</td></tr>
                  </tbody>
                </table>
                {v.classBalance && v.classBalance.length > 0 && (
                  <>
                    <h4 className="font-semibold text-sm mt-3 mb-1">Class Balance</h4>
                    <div className="space-y-1 max-h-[200px] overflow-y-auto">
                      {v.classBalance.map((c, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                          <span className="flex-1 truncate" title={c.label}>{c.label}</span>
                          <span className="font-medium">{c.count}</span>
                          <span className="text-gray-500">({c.percentage.toFixed(1)}%)</span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Metadata */}
            <div className="card bg-base-200 p-3">
              <h4 className="font-semibold text-sm mb-2">Metadata</h4>
              <table className="table table-xs">
                <tbody>
                  {[
                    ['Concept Code', v.conceptCode],
                    ['Concept Name', v.conceptName],
                    ['OMOP ID', v.omopId],
                    ['Domain', v.domain],
                    ['Visit', v.visit],
                    ['Visit Concept', v.visitConceptName],
                    ['Units', v.units],
                    ['Dict Type', v.metadataVarType],
                    ['Unique Values', v.uniqueValues],
                  ].filter(([, val]) => val !== undefined && val !== null && val !== '').map(([label, value]) => (
                    <tr key={label as string}>
                      <td className="font-medium">{label}</td>
                      <td className="text-right text-xs">{String(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      <div className="modal-backdrop" onClick={onClose}></div>
    </div>
  );
};

export default EdaVariableDetailModal;
