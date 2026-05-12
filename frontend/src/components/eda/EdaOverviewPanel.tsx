import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { EdaData, EdaVariable, completenessScore } from '@/utils/edaParsing';

interface EdaOverviewPanelProps {
  edaData: EdaData;
  onVariableClick: (v: EdaVariable) => void;
}

const EdaOverviewPanel: React.FC<EdaOverviewPanelProps> = ({ edaData, onVariableClick }) => {
  const { variables, numericVars, categoricalVars, dateVars } = edaData;

  // 1. Variable type distribution pie
  const typePieOption = useMemo(() => ({
    title: { text: 'Variable Types', left: 'center', textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { bottom: 0, type: 'scroll' },
    series: [{
      type: 'pie',
      radius: ['35%', '65%'],
      avoidLabelOverlap: true,
      itemStyle: { borderRadius: 6, borderColor: '#fff', borderWidth: 2 },
      label: { show: true, formatter: '{b}\n{c}' },
      data: [
        { value: numericVars.length, name: 'Numeric', itemStyle: { color: '#3b82f6' } },
        { value: categoricalVars.length, name: 'Categorical', itemStyle: { color: '#10b981' } },
        ...(dateVars.length > 0 ? [{ value: dateVars.length, name: 'Date', itemStyle: { color: '#f59e0b' } }] : []),
      ].filter(d => d.value > 0),
    }],
  }), [numericVars, categoricalVars, dateVars]);

  // 2. Completeness distribution histogram
  const completenessHistOption = useMemo(() => {
    const scores = variables.map(v => completenessScore(v));
    const buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    const counts = new Array(buckets.length - 1).fill(0);
    scores.forEach(s => {
      const idx = Math.min(Math.floor(s / 10), 9);
      counts[idx]++;
    });
    const labels = buckets.slice(0, -1).map((b, i) => `${b}-${buckets[i + 1]}%`);

    return {
      title: { text: 'Data Completeness Distribution', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: labels, axisLabel: { rotate: 45, fontSize: 10 } },
      yAxis: { type: 'value', name: 'Variables' },
      series: [{
        type: 'bar',
        data: counts.map((c, i) => ({
          value: c,
          itemStyle: {
            color: i >= 7 ? '#10b981' : i >= 4 ? '#f59e0b' : '#ef4444',
          },
        })),
        barWidth: '60%',
      }],
      grid: { bottom: 60, left: 50, right: 20, top: 40 },
    };
  }, [variables]);

  // 3. Normality test summary (numeric vars only)
  const normalitySummary = useMemo(() => {
    const normal = numericVars.filter(v => v.isNormal).length;
    const nonNormal = numericVars.filter(v => !v.isNormal).length;
    return { normal, nonNormal };
  }, [numericVars]);

  const normalityPieOption = useMemo(() => ({
    title: { text: 'Normality Test Results', left: 'center', textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { bottom: 0 },
    series: [{
      type: 'pie',
      radius: ['35%', '65%'],
      itemStyle: { borderRadius: 6, borderColor: '#fff', borderWidth: 2 },
      label: { show: true, formatter: '{b}\n{c}' },
      data: [
        { value: normalitySummary.normal, name: 'Normal', itemStyle: { color: '#10b981' } },
        { value: normalitySummary.nonNormal, name: 'Non-Normal', itemStyle: { color: '#f59e0b' } },
      ].filter(d => d.value > 0),
    }],
  }), [normalitySummary]);

  // 4. Top/bottom variables by completeness
  const topComplete = useMemo(() =>
    [...variables].sort((a, b) => completenessScore(b) - completenessScore(a)).slice(0, 10),
    [variables]
  );
  const bottomComplete = useMemo(() =>
    [...variables].sort((a, b) => completenessScore(a) - completenessScore(b)).slice(0, 10),
    [variables]
  );

  const completenessBarOption = useMemo(() => ({
    title: { text: 'Most & Least Complete Variables', left: 'center', textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { bottom: 0, data: ['Most Complete', 'Least Complete'] },
    grid: { left: 180, right: 30, top: 40, bottom: 50 },
    xAxis: { type: 'value', max: 100, name: '%' },
    yAxis: {
      type: 'category',
      data: [
        ...bottomComplete.map(v => v.name).reverse(),
        ...topComplete.map(v => v.name).reverse(),
      ],
      axisLabel: { fontSize: 10, width: 160, overflow: 'truncate' },
    },
    series: [
      {
        name: 'Most Complete',
        type: 'bar',
        data: [
          ...new Array(bottomComplete.length).fill(null),
          ...topComplete.map(v => completenessScore(v)).reverse(),
        ],
        itemStyle: { color: '#10b981' },
        barWidth: '60%',
      },
      {
        name: 'Least Complete',
        type: 'bar',
        data: [
          ...bottomComplete.map(v => completenessScore(v)).reverse(),
          ...new Array(topComplete.length).fill(null),
        ],
        itemStyle: { color: '#ef4444' },
        barWidth: '60%',
      },
    ],
  }), [topComplete, bottomComplete]);

  // 5. Skewness distribution scatter (numeric)
  const skewnessScatterOption = useMemo(() => {
    if (numericVars.length === 0) return null;
    const data = numericVars
      .filter(v => v.skewness !== undefined && v.kurtosis !== undefined)
      .map(v => ({
        value: [v.skewness, v.kurtosis],
        name: v.name,
        label: v.label,
      }));

    return {
      title: { text: 'Skewness vs Kurtosis', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: {
        formatter: (p: any) => `<strong>${p.data.name}</strong><br/>${p.data.label}<br/>Skewness: ${p.data.value[0]}<br/>Kurtosis: ${p.data.value[1]}`,
      },
      xAxis: { type: 'value', name: 'Skewness', nameLocation: 'center', nameGap: 30 },
      yAxis: { type: 'value', name: 'Kurtosis' },
      series: [{
        type: 'scatter',
        symbolSize: 10,
        data,
        itemStyle: { color: '#6366f1', opacity: 0.7 },
        emphasis: { itemStyle: { color: '#4f46e5', borderColor: '#000', borderWidth: 1 } },
      }],
      grid: { left: 60, right: 30, top: 40, bottom: 50 },
    };
  }, [numericVars]);

  // 6. Domain distribution (from metadata)
  const domainDistOption = useMemo(() => {
    const domainCounts: Record<string, number> = {};
    variables.forEach(v => {
      const d = v.domain || 'Unknown';
      domainCounts[d] = (domainCounts[d] || 0) + 1;
    });
    const sorted = Object.entries(domainCounts).sort((a, b) => b[1] - a[1]);
    if (sorted.length === 0) return null;

    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];
    return {
      title: { text: 'Variable Domains', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: { bottom: 0, type: 'scroll' },
      series: [{
        type: 'pie',
        radius: ['35%', '65%'],
        itemStyle: { borderRadius: 6, borderColor: '#fff', borderWidth: 2 },
        label: { show: true, formatter: '{b}\n{c}' },
        data: sorted.map(([name, value], i) => ({
          name, value, itemStyle: { color: colors[i % colors.length] },
        })),
      }],
    };
  }, [variables]);

  return (
    <div className="space-y-6">
      {/* Row 1: Type pie + Completeness histogram + Normality pie */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts option={typePieOption} style={{ height: 300 }} />
        </div>
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts option={completenessHistOption} style={{ height: 300 }} />
        </div>
        {numericVars.length > 0 ? (
          <div className="card bg-base-100 shadow-md p-4">
            <ReactECharts option={normalityPieOption} style={{ height: 300 }} />
          </div>
        ) : (
          domainDistOption && (
            <div className="card bg-base-100 shadow-md p-4">
              <ReactECharts option={domainDistOption} style={{ height: 300 }} />
            </div>
          )
        )}
      </div>

      {/* Row 2: Domain distribution (if normality shown above) + Completeness bars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {numericVars.length > 0 && domainDistOption && (
          <div className="card bg-base-100 shadow-md p-4">
            <ReactECharts option={domainDistOption} style={{ height: 350 }} />
          </div>
        )}
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts option={completenessBarOption} style={{ height: Math.max(350, (topComplete.length + bottomComplete.length) * 22 + 100) }} />
        </div>
      </div>

      {/* Row 3: Skewness vs Kurtosis scatter */}
      {skewnessScatterOption && (
        <div className="card bg-base-100 shadow-md p-4">
          <ReactECharts option={skewnessScatterOption} style={{ height: 400 }} />
        </div>
      )}

      {/* Row 4: Quick-access variable cards */}
      <div className="card bg-base-100 shadow-md p-4">
        <h3 className="font-bold text-lg mb-3">Quick Stats — Click any variable for details</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2 max-h-[500px] overflow-y-auto">
          {variables.slice(0, 60).map(v => {
            const score = completenessScore(v);
            const bgColor = score >= 80 ? 'bg-green-50 border-green-200' :
              score >= 50 ? 'bg-amber-50 border-amber-200' : 'bg-red-50 border-red-200';
            return (
              <button
                key={v.name}
                onClick={() => onVariableClick(v)}
                className={`border rounded-lg p-2 text-left hover:shadow-md transition-shadow cursor-pointer ${bgColor}`}
              >
                <div className="font-medium text-xs truncate" title={v.name}>{v.name}</div>
                <div className="text-[10px] text-gray-500 truncate" title={v.label}>{v.label}</div>
                <div className="flex items-center gap-1 mt-1">
                  <span className={`badge badge-xs ${v.type === 'numeric' ? 'badge-primary' : v.type === 'categorical' ? 'badge-success' : 'badge-warning'}`}>
                    {v.type}
                  </span>
                  <span className="text-[10px]">{score.toFixed(0)}%</span>
                </div>
              </button>
            );
          })}
        </div>
        {variables.length > 60 && (
          <p className="text-sm text-gray-500 mt-2">Showing first 60 of {variables.length} variables</p>
        )}
      </div>
    </div>
  );
};

export default EdaOverviewPanel;
