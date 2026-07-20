import type {MappingRow} from '@/utils/mappingPreview';

export interface MappingGraphNode {
  id: string;
  varName: string;
  label: string;
  domain: string;
  cohortId: string;
  side: 'source' | 'target';
  omopCode?: string;
  uncovered?: boolean;
  categories?: string;
}

export interface MappingGraphEdge {
  srcId: string;
  tgtId: string;
  sourceStudy: string;
  targetStudy: string;
  sourceVar: string;
  targetVar: string;
  relation: string;
  status: string;
  sim: number;
}

export interface MappingGraph {
  nodes: MappingGraphNode[];
  edges: MappingGraphEdge[];
  sourceNodes: MappingGraphNode[];
  targetNodes: MappingGraphNode[];
  sourceDomains: string[];
  targetDomains: string[];
  relations: string[];
  sourceEdgeCounts: Map<string, number>;
  targetEdgeCounts: Map<string, number>;
  sourceMaxMappings: number;
  targetMaxMappings: number;
  relationCounts: Record<string, number>;
  statusCounts: Record<string, number>;
}

export interface BuildMappingGraphOptions {
  sourceCohort?: string;
  targetCohort?: string;
}

const normalizedDomain = (raw: unknown): string =>
  String(raw || '')
    .split('||')[0]
    .trim()
    .toLowerCase()
    .replace(/ /g, '_') || 'uncategorized';

const modeDomain = (domains: string[]): string => {
  const counts: Record<string, number> = {};
  domains.forEach(domain => {
    counts[domain] = (counts[domain] || 0) + 1;
  });
  return Object.entries(counts).sort((left, right) => right[1] - left[1])[0]?.[0] || '';
};

const stableNodeId = (cohortId: string, variableName: string): string =>
  cohortId ? `${cohortId}:${variableName}` : variableName;

export const buildMappingGraph = (
  rows: MappingRow[],
  options: BuildMappingGraphOptions = {}
): MappingGraph => {
  const selectedRows = options.targetCohort
    ? rows.filter(row => row.target_study === options.targetCohort)
    : rows;
  const sourceLabels = new Map<string, string>();
  const targetLabels = new Map<string, string>();
  const sourceDomainsById = new Map<string, string[]>();
  const targetDomainsById = new Map<string, string[]>();
  const sourceCategories = new Map<string, string>();
  const targetCategories = new Map<string, string>();
  const sourceCodes = new Map<string, string>();
  const targetCodes = new Map<string, string>();
  const sourceMetadata = new Map<string, {cohortId: string; variableName: string}>();
  const targetMetadata = new Map<string, {cohortId: string; variableName: string}>();
  const edges: MappingGraphEdge[] = [];

  selectedRows.forEach(row => {
    const sourceVar = String(row.s_source || '');
    const targetVar = String(row.target || '');
    const sourceStudy = String(row.source_study || options.sourceCohort || '');
    const targetStudy = String(row.target_study || options.targetCohort || '');
    const sourceId = stableNodeId(sourceStudy, sourceVar);
    const targetId = stableNodeId(targetStudy, targetVar);
    const domain = normalizedDomain(row.omop_domain);

    if (!sourceLabels.has(sourceId)) sourceLabels.set(sourceId, String(row.s_label || ''));
    if (!targetLabels.has(targetId)) targetLabels.set(targetId, String(row.target_label || targetVar));
    if (!sourceDomainsById.has(sourceId)) sourceDomainsById.set(sourceId, []);
    if (!targetDomainsById.has(targetId)) targetDomainsById.set(targetId, []);
    sourceDomainsById.get(sourceId)!.push(domain);
    targetDomainsById.get(targetId)!.push(domain);
    sourceMetadata.set(sourceId, {cohortId: sourceStudy, variableName: sourceVar});
    targetMetadata.set(targetId, {cohortId: targetStudy, variableName: targetVar});
    if (!sourceCategories.has(sourceId) && row.source_categories_labels) {
      sourceCategories.set(sourceId, String(row.source_categories_labels));
    }
    if (!targetCategories.has(targetId) && row.target_categories_labels) {
      targetCategories.set(targetId, String(row.target_categories_labels));
    }
    const sourceCode = String(row.scode || '');
    const targetCode = String(row.tcode || '');
    if (!sourceCodes.has(sourceId) && sourceCode) sourceCodes.set(sourceId, sourceCode);
    if (!targetCodes.has(targetId) && targetCode) targetCodes.set(targetId, targetCode);
    edges.push({
      srcId: sourceId,
      tgtId: targetId,
      sourceStudy,
      targetStudy,
      sourceVar,
      targetVar,
      relation: String(row.mapping_relation || ''),
      status: String(row.harmonization_status || 'pending'),
      sim: Number(row.sim_score) || 0.5
    });
  });

  const sourceNodes = Array.from(sourceLabels.keys()).map(id => ({
    id,
    varName: sourceMetadata.get(id)!.variableName,
    label: sourceLabels.get(id)!,
    domain: modeDomain(sourceDomainsById.get(id) || []),
    cohortId: sourceMetadata.get(id)!.cohortId,
    side: 'source' as const,
    omopCode: sourceCodes.get(id),
    categories: sourceCategories.get(id)
  }));
  const targetNodes = Array.from(targetLabels.keys()).map(id => ({
    id,
    varName: targetMetadata.get(id)!.variableName,
    label: targetLabels.get(id)!,
    domain: modeDomain(targetDomainsById.get(id) || []),
    cohortId: targetMetadata.get(id)!.cohortId,
    side: 'target' as const,
    omopCode: targetCodes.get(id),
    categories: targetCategories.get(id)
  }));
  const sourceEdgeCounts = new Map<string, number>();
  const targetEdgeCounts = new Map<string, number>();
  const relationCounts: Record<string, number> = {};
  const statusCounts: Record<string, number> = {};
  edges.forEach(edge => {
    sourceEdgeCounts.set(edge.srcId, (sourceEdgeCounts.get(edge.srcId) || 0) + 1);
    targetEdgeCounts.set(edge.tgtId, (targetEdgeCounts.get(edge.tgtId) || 0) + 1);
    relationCounts[edge.relation] = (relationCounts[edge.relation] || 0) + 1;
    statusCounts[edge.status] = (statusCounts[edge.status] || 0) + 1;
  });
  const sourceCounts = Array.from(sourceEdgeCounts.values());
  const targetCounts = Array.from(targetEdgeCounts.values());

  return {
    nodes: [...sourceNodes, ...targetNodes],
    edges,
    sourceNodes,
    targetNodes,
    sourceDomains: Array.from(new Set(sourceNodes.map(node => node.domain))).filter(Boolean).sort(),
    targetDomains: Array.from(new Set(targetNodes.map(node => node.domain))).filter(Boolean).sort(),
    relations: Array.from(new Set(edges.map(edge => edge.relation))).filter(Boolean).sort(),
    sourceEdgeCounts,
    targetEdgeCounts,
    sourceMaxMappings: sourceCounts.length ? Math.max(...sourceCounts) : 0,
    targetMaxMappings: targetCounts.length ? Math.max(...targetCounts) : 0,
    relationCounts,
    statusCounts
  };
};
