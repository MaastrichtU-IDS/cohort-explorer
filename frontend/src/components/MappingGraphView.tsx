import React, { useState, useMemo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';

interface RowData {
  [key: string]: string | number | boolean | null | undefined;
}

interface MappingGraphViewProps {
  data: RowData[];
  sourceCohort: string;
}

// Amplify edge width: use a power curve so differences between 0.4 and 0.8 are dramatic
function amplifiedWidth(score: number | null): number {
  if (score == null || isNaN(score)) return 1;
  // Clamp to [0, 1]
  const s = Math.max(0, Math.min(1, score));
  // Power curve: raises small differences significantly
  // score=0.4 → ~1.6, score=0.6 → ~3.6, score=0.8 → ~7.2, score=0.95 → ~11.4
  return 1 + Math.pow(s, 3) * 14;
}

// Color palette for cohorts
const COHORT_COLORS = [
  '#6366f1', // indigo
  '#f59e0b', // amber
  '#10b981', // emerald
  '#ef4444', // red
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#f97316', // orange
  '#84cc16', // lime
  '#ec4899', // pink
  '#14b8a6', // teal
];

// Harmonization status colors for edges
const STATUS_COLORS: Record<string, string> = {
  'pending': '#94a3b8',       // slate
  'confirmed': '#22c55e',     // green
  'rejected': '#ef4444',      // red
  'needs_review': '#f59e0b',  // amber
};

function getStatusColor(status: string): string {
  return STATUS_COLORS[status] || '#94a3b8';
}

export default function MappingGraphView({ data, sourceCohort }: MappingGraphViewProps) {
  const [selectedStatuses, setSelectedStatuses] = useState<string[]>([]);
  const [selectedEdge, setSelectedEdge] = useState<RowData | null>(null);
  const [edaImage, setEdaImage] = useState<string | null>(null);
  const [edaError, setEdaError] = useState<string | null>(null);
  const [edaLoading, setEdaLoading] = useState(false);

  // Get unique harmonization statuses with counts
  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    data.forEach(row => {
      const status = (row.harmonization_status?.toString() || 'pending');
      counts[status] = (counts[status] || 0) + 1;
    });
    return counts;
  }, [data]);

  // Compute degree and role from UNFILTERED data (stays fixed regardless of filters)
  const { degree, nodeRole } = useMemo(() => {
    const deg: Record<string, number> = {};
    const role: Record<string, 'source' | 'target' | 'both'> = {};
    data.forEach((row) => {
      const sourceVar = row.s_source as string;
      const targetVar = row.target as string;
      const sourceStudy = (row.source_study as string) || sourceCohort;
      const targetStudy = row.target_study as string;
      if (!sourceVar || !targetVar) return;
      const sourceId = `${sourceStudy}::${sourceVar}`;
      const targetId = `${targetStudy}::${targetVar}`;
      deg[sourceId] = (deg[sourceId] || 0) + 1;
      deg[targetId] = (deg[targetId] || 0) + 1;
      const prevSrc = role[sourceId];
      role[sourceId] = prevSrc === 'target' ? 'both' : 'source';
      const prevTgt = role[targetId];
      role[targetId] = prevTgt === 'source' ? 'both' : 'target';
    });
    return { degree: deg, nodeRole: role };
  }, [data, sourceCohort]);

  // Filter data by selected harmonization statuses
  const filteredData = useMemo(() => {
    if (selectedStatuses.length === 0) return data;
    return data.filter(row => {
      const status = (row.harmonization_status?.toString() || 'pending');
      return selectedStatuses.includes(status);
    });
  }, [data, selectedStatuses]);

  // Build graph nodes and edges from filtered data
  const { nodes, edges, cohortColorMap } = useMemo(() => {
    const nodeMap = new Map<string, { id: string; name: string; cohort: string; category: string }>();
    const edgeList: any[] = [];
    const cohorts = new Set<string>();

    filteredData.forEach((row) => {
      const sourceVar = row.s_source as string;
      const targetVar = row.target as string;
      const sourceStudy = (row.source_study as string) || sourceCohort;
      const targetStudy = row.target_study as string;
      const simScore = row.sim_score as number | null;
      const status = (row.harmonization_status?.toString() || 'pending');

      if (!sourceVar || !targetVar) return;

      cohorts.add(sourceStudy);
      cohorts.add(targetStudy);

      const sourceId = `${sourceStudy}::${sourceVar}`;
      const targetId = `${targetStudy}::${targetVar}`;

      if (!nodeMap.has(sourceId)) {
        nodeMap.set(sourceId, {
          id: sourceId,
          name: sourceVar,
          cohort: sourceStudy,
          category: sourceStudy,
        });
      }
      if (!nodeMap.has(targetId)) {
        nodeMap.set(targetId, {
          id: targetId,
          name: targetVar,
          cohort: targetStudy,
          category: targetStudy,
        });
      }

      edgeList.push({
        source: sourceId,
        target: targetId,
        lineStyle: {
          width: amplifiedWidth(simScore),
          color: getStatusColor(status),
          opacity: 0.7,
          curveness: 0.2,
        },
        // Attach row data for edge click
        rowData: row,
      });
    });

    // Assign colors to cohorts
    const cohortArr = Array.from(cohorts);
    const colorMap: Record<string, string> = {};
    cohortArr.forEach((c, i) => {
      colorMap[c] = COHORT_COLORS[i % COHORT_COLORS.length];
    });

    const nodeList = Array.from(nodeMap.values()).map(n => {
      const deg = degree[n.id] || 0;
      const role = nodeRole[n.id] || 'source';
      // Badge on outer edge: right side for source, left side for target
      const isSource = role === 'source' || role === 'both';

      return {
        id: n.id,
        name: n.name,
        category: n.cohort,
        symbolSize: 18,
        itemStyle: { color: colorMap[n.cohort] },
        label: {
          show: true,
          fontSize: 9,
          position: (isSource ? 'right' : 'left') as 'right' | 'left',
          formatter: `{name|${n.name}}  {badge|${deg}}`,
          rich: {
            name: {
              fontSize: 9,
              color: '#333',
            },
            badge: {
              fontSize: 9,
              fontWeight: 'bold' as const,
              color: '#fff',
              backgroundColor: colorMap[n.cohort] || '#6366f1',
              borderRadius: 7,
              padding: [1, 4, 1, 4],
            },
          },
        },
      };
    });

    return { nodes: nodeList, edges: edgeList, cohortColorMap: colorMap };
  }, [filteredData, sourceCohort, degree, nodeRole]);

  // ECharts option
  const option = useMemo(() => ({
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        if (params.dataType === 'edge') {
          const row = params.data?.rowData;
          if (!row) return '';
          const score = row.sim_score != null ? Number(row.sim_score).toFixed(4) : '--';
          return `<div style="max-width:300px">
            <strong>${row.s_source}</strong> → <strong>${row.target}</strong><br/>
            Score: <strong>${score}</strong><br/>
            Status: ${row.harmonization_status || 'pending'}<br/>
            Relation: ${row.mapping_relation || '--'}<br/>
            <em style="color:#6366f1">Click edge to compare EDAs</em>
          </div>`;
        }
        if (params.dataType === 'node') {
          return `<strong>${params.data.name}</strong><br/>${params.data.category}`;
        }
        return '';
      },
    },
    legend: {
      data: Object.keys(cohortColorMap).map(c => ({
        name: c,
        itemStyle: { color: cohortColorMap[c] },
      })),
      top: 10,
      textStyle: { fontSize: 11 },
    },
    animationDuration: 800,
    series: [{
      type: 'graph',
      layout: 'force',
      roam: true,
      draggable: true,
      force: {
        repulsion: 200,
        gravity: 0.1,
        edgeLength: [80, 200],
        layoutAnimation: true,
      },
      categories: Object.keys(cohortColorMap).map(c => ({
        name: c,
        itemStyle: { color: cohortColorMap[c] },
      })),
      data: nodes,
      links: edges,
      emphasis: {
        focus: 'adjacency',
        lineStyle: { width: 6, opacity: 1 },
      },
      lineStyle: {
        curveness: 0.2,
      },
    }],
  }), [nodes, edges, cohortColorMap]);

  // Handle edge click — show detail panel
  const handleChartClick = useCallback((params: any) => {
    if (params.dataType === 'edge' && params.data?.rowData) {
      setSelectedEdge(params.data.rowData);
      setEdaImage(null);
      setEdaError(null);
    }
  }, []);

  // Handle EDA comparison
  const handleCompareEda = async () => {
    if (!selectedEdge) return;
    const sourceVar = selectedEdge.s_source as string;
    const targetVar = selectedEdge.target as string;
    const targetStudy = selectedEdge.target_study as string;

    if (!sourceCohort || !sourceVar || !targetStudy || !targetVar) return;

    setEdaLoading(true);
    setEdaError(null);
    setEdaImage(null);

    const imageUrl = `/api/compare-eda/${encodeURIComponent(sourceCohort)}/${encodeURIComponent(sourceVar)}/${encodeURIComponent(targetStudy)}/${encodeURIComponent(targetVar)}`;

    try {
      const response = await fetch(imageUrl);
      if (!response.ok) {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const errorData = await response.json();
          setEdaError(errorData.details || errorData.error || 'Failed to load image');
        } else {
          const errorText = await response.text();
          setEdaError(errorText || 'Failed to load image');
        }
      } else {
        setEdaImage(imageUrl);
      }
    } catch (error) {
      setEdaError('Failed to fetch image: ' + (error as Error).message);
    } finally {
      setEdaLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Harmonization Status Filter */}
      <div className="flex items-center gap-4 flex-wrap">
        <span className="text-sm font-semibold">Filter by status:</span>
        {Object.entries(statusCounts).map(([status, count]) => (
          <label key={status} className="flex items-center gap-1.5 cursor-pointer text-sm">
            <input
              type="checkbox"
              className="checkbox checkbox-xs"
              checked={selectedStatuses.includes(status)}
              onChange={(e) => {
                if (e.target.checked) {
                  setSelectedStatuses(prev => [...prev, status]);
                } else {
                  setSelectedStatuses(prev => prev.filter(s => s !== status));
                }
              }}
            />
            <span
              className="inline-block w-3 h-3 rounded-full mr-0.5"
              style={{ backgroundColor: getStatusColor(status) }}
            />
            {status} ({count})
          </label>
        ))}
        {selectedStatuses.length > 0 && (
          <button className="btn btn-xs btn-ghost" onClick={() => setSelectedStatuses([])}>
            Clear
          </button>
        )}
      </div>

      {/* Edge width legend */}
      <div className="flex items-center gap-3 text-xs text-gray-500">
        <span>Edge width = similarity score:</span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 20, height: 2, background: '#94a3b8' }} /> 0.4
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 20, height: 5, background: '#94a3b8' }} /> 0.6
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 20, height: 9, background: '#94a3b8' }} /> 0.8
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 20, height: 13, background: '#94a3b8' }} /> 0.95
        </span>
      </div>

      {/* Graph */}
      <div className="border rounded-lg bg-white relative" style={{ height: '600px' }}>
        {nodes.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400">
            No mappings to display with current filters.
          </div>
        ) : (
          <ReactECharts
            option={option}
            style={{ height: '100%', width: '100%' }}
            onEvents={{ click: handleChartClick }}
            opts={{ renderer: 'canvas' }}
          />
        )}
      </div>

      {/* Edge Detail Panel — slides in when an edge is clicked */}
      {selectedEdge && (
        <div className="border rounded-lg bg-base-100 shadow-lg p-5 animate-[fadeIn_0.2s_ease-out]">
          <div className="flex justify-between items-start">
            <div>
              <h4 className="font-bold text-base mb-2">Mapping Detail</h4>
              <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-sm">
                <div><span className="text-gray-500">Source:</span> <strong>{selectedEdge.s_source as string}</strong></div>
                <div><span className="text-gray-500">Target:</span> <strong>{selectedEdge.target as string}</strong></div>
                <div><span className="text-gray-500">Source Cohort:</span> {(selectedEdge.source_study as string) || sourceCohort}</div>
                <div><span className="text-gray-500">Target Cohort:</span> {selectedEdge.target_study as string}</div>
                <div><span className="text-gray-500">Score:</span> {selectedEdge.sim_score != null ? Number(selectedEdge.sim_score).toFixed(4) : '--'}</div>
                <div><span className="text-gray-500">Status:</span> {selectedEdge.harmonization_status as string}</div>
                <div><span className="text-gray-500">Relation:</span> {selectedEdge.mapping_relation as string || '--'}</div>
              </div>
            </div>
            <div className="flex gap-2 items-center">
              <button
                className="btn btn-primary btn-sm gap-1"
                onClick={handleCompareEda}
                disabled={edaLoading}
              >
                {edaLoading ? (
                  <span className="loading loading-spinner loading-xs"></span>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="M7 16l4-8 4 4 4-8"/></svg>
                )}
                Compare EDA
              </button>
              <button className="btn btn-ghost btn-sm btn-circle" onClick={() => { setSelectedEdge(null); setEdaImage(null); setEdaError(null); }}>✕</button>
            </div>
          </div>

          {/* EDA Comparison Result */}
          {(edaImage || edaError) && (
            <div className="mt-4 pt-4 border-t">
              {edaError ? (
                <div className="alert alert-error text-sm">
                  <div className="whitespace-pre-wrap">{edaError}</div>
                </div>
              ) : edaImage ? (
                <div className="text-center">
                  <img
                    src={edaImage}
                    alt="EDA Comparison"
                    style={{ maxWidth: '75%', height: 'auto' }}
                    className="mx-auto rounded shadow"
                  />
                  <a
                    href={edaImage}
                    download={`comparison_${selectedEdge.s_source}_vs_${selectedEdge.target}.png`}
                    className="btn btn-sm btn-outline mt-3"
                  >
                    Save Image
                  </a>
                </div>
              ) : null}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
