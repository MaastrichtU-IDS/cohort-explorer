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
  const [showImage, setShowImage] = useState(false);
  const [imageState, setImageState] = useState<'loading' | 'loaded' | 'error'>('loading');
  const score = completenessScore(v);

  // Main chart based on type
  const mainChartOption = useMemo(() => {
    if (v.type === 'numeric') {
      // Gaussian approximation + box plot markers
      const mean = v.mean ?? 0;
      const sd = v.stdDev ?? 1;
      const xMin = v.min ?? (mean - 3.5 * sd);
      const xMax = v.max ?? (mean + 3.5 * sd);
      const nPoints = 80;
      const step = (xMax - xMin) / nPoints;
      const curveData: [number, number][] = [];
      for (let i = 0; i <= nPoints; i++) {
        const x = xMin + i * step;
        const y = Math.exp(-0.5 * Math.pow((x - mean) / sd, 2)) / (sd * Math.sqrt(2 * Math.PI));
        curveData.push([x, y]);
      }

      return {
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'value', min: xMin, max: xMax },
        yAxis: { type: 'value', show: false },
        series: [
          {
            type: 'line',
            data: curveData,
            smooth: true,
            areaStyle: { color: 'rgba(59, 130, 246, 0.15)' },
            lineStyle: { color: '#3b82f6', width: 2.5 },
            symbol: 'none',
          },
          {
            type: 'line',
            markLine: {
              silent: true,
              symbol: 'none',
              data: [
                { xAxis: v.median, lineStyle: { color: '#10b981', type: 'solid', width: 2.5 }, label: { formatter: `Median: ${v.median}`, fontSize: 11 } },
                { xAxis: v.mean, lineStyle: { color: '#ef4444', type: 'dashed', width: 2 }, label: { formatter: `Mean: ${v.mean?.toFixed(2)}`, fontSize: 11 } },
                { xAxis: v.q1, lineStyle: { color: '#94a3b8', type: 'dotted', width: 1.5 }, label: { formatter: `Q1: ${v.q1}`, fontSize: 10 } },
                { xAxis: v.q3, lineStyle: { color: '#94a3b8', type: 'dotted', width: 1.5 }, label: { formatter: `Q3: ${v.q3}`, fontSize: 10 } },
              ].filter(d => d.xAxis !== undefined),
            },
            data: [],
          },
        ],
        grid: { left: 30, right: 30, top: 20, bottom: 30 },
      };
    }

    if (v.type === 'categorical' && v.classBalance) {
      const categories = v.classBalance;
      return {
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          formatter: (params: any) => {
            const p = params[0];
            const cat = categories[p.dataIndex];
            return `<strong>${cat.label}</strong><br/>${cat.count} observations (${cat.percentage.toFixed(1)}%)`;
          },
        },
        xAxis: {
          type: 'category',
          data: categories.map(c => c.label),
          axisLabel: { rotate: categories.length > 6 ? 45 : 0, fontSize: 11, interval: 0 },
        },
        yAxis: { type: 'value', name: 'Count' },
        series: [{
          type: 'bar',
          data: categories.map((c, i) => ({
            value: c.count,
            itemStyle: { color: COLORS[i % COLORS.length] },
          })),
          barMaxWidth: 60,
          label: { show: true, position: 'top', fontSize: 10, formatter: (p: any) => `${categories[p.dataIndex].percentage.toFixed(1)}%` },
        }],
        grid: { left: 50, right: 20, top: 20, bottom: categories.some(c => c.label.length > 5) ? 80 : 40 },
      };
    }

    return null;
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
          {/* Chart */}
          <div className="lg:col-span-2">
            {mainChartOption && (
              <div className="card bg-base-200 p-3">
                <h3 className="font-semibold text-sm mb-2">
                  {v.type === 'numeric' ? 'Distribution (Gaussian approximation)' : 'Category Distribution'}
                </h3>
                <ReactECharts option={mainChartOption} style={{ height: 280 }} />
              </div>
            )}

            {/* Original graph toggle */}
            <div className="mt-3">
              <button
                className="btn btn-sm btn-outline"
                onClick={() => { setShowImage(!showImage); setImageState('loading'); }}
              >
                {showImage ? 'Hide' : 'Show'} Original EDA Graph (PNG)
              </button>
              {showImage && (
                <div className="mt-2 flex justify-center min-h-[100px] bg-white rounded-lg p-2">
                  {imageState === 'loading' && <span className="loading loading-spinner loading-md"></span>}
                  <img
                    src={imageUrl}
                    alt={`${v.name} EDA graph`}
                    className={`max-w-full ${imageState === 'loaded' ? '' : 'hidden'}`}
                    onLoad={() => setImageState('loaded')}
                    onError={() => setImageState('error')}
                    crossOrigin="use-credentials"
                  />
                  {imageState === 'error' && <p className="text-gray-500 text-sm">Original graph not available</p>}
                </div>
              )}
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
