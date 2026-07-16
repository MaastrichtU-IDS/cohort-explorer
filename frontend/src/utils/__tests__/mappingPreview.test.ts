import {describe, expect, it} from 'vitest';
import {buildMappingGraph} from '@/utils/mappingGraph';
import {
  canonicalCohortId,
  mappingTargetElementId,
  parseMappingPreview,
  projectMappingJson
} from '@/utils/mappingPreview';

const mappingCsv = [
  'source_study,target_study,source,target,slabel,tlabel,category,mapping_relation,harmonization_status,sim_score,source_original_categories,source_categories_labels,target_original_categories,target_categories_labels',
  'TIME-CHF,GISSI-HF,heart_rate,HR,"Heart rate, resting",Pulse,measurement,exact match,Identical Match,0.98,1||2,low||high,A||B,low||high',
  'TIME-CHF,GISSI-HF,nyha,NYHA,NYHA class,Functional class,observation,compatible match,Compatible Match,0.8,,,,',
].join('\n');

describe('mapping projections', () => {
  it('builds a stable target-cohort selector', () => {
    expect(mappingTargetElementId('GISSI-HF')).toBe('mapping-target-GISSI-HF');
  });

  it('resolves fixture study identifiers to canonical metadata keys', () => {
    const cohorts = {'TIME-CHF': {}, 'GISSI-HF': {}};

    expect(canonicalCohortId('time-chf', cohorts)).toBe('TIME-CHF');
    expect(canonicalCohortId('gissi-hf', cohorts)).toBe('GISSI-HF');
    expect(canonicalCohortId('unknown', cohorts)).toBe('unknown');
  });

  it('parses CSV rows and category labels into the stable table shape', () => {
    const preview = parseMappingPreview(mappingCsv);

    expect(preview.rows[0]).toMatchObject({
      source_study: 'TIME-CHF',
      s_source: 'heart_rate',
      s_label: 'Heart rate, resting',
      target_study: 'GISSI-HF',
      target: 'HR',
      target_label: 'Pulse',
      mapping_relation: 'exact match',
      sim_score: 0.98,
      source_categories_codes_labels: '1||2 (low||high)',
      target_categories_codes_labels: 'A||B (low||high)'
    });
  });

  it('projects the fixture mapping into stable graph nodes and edges', () => {
    const preview = parseMappingPreview(mappingCsv);
    const graph = buildMappingGraph(preview.rows);

    expect(graph.edges.length).toBeGreaterThan(0);
    expect(graph.nodes.map(node => node.id)).toContain('TIME-CHF:heart_rate');
    expect(graph.nodes.map(node => node.id)).toContain('GISSI-HF:HR');
    expect(graph.sourceDomains).toEqual(['measurement', 'observation']);
    expect(graph.edges[0]).toMatchObject({
      sourceVar: 'heart_rate',
      targetVar: 'HR',
      status: 'Identical Match'
    });
  });

  it('keeps mappings without an OMOP domain visible as uncategorized', () => {
    const graph = buildMappingGraph([
      {
        source_study: 'time-chf',
        target_study: 'gissi-hf',
        s_source: 'bmi',
        target: 'bmi (derived)',
        omop_domain: '',
        harmonization_status: 'pending'
      }
    ]);

    expect(graph.edges).toHaveLength(1);
    expect(graph.sourceNodes[0].domain).toBe('uncategorized');
    expect(graph.targetNodes[0].domain).toBe('uncategorized');
    expect(graph.sourceDomains).toEqual(['uncategorized']);
    expect(graph.targetDomains).toEqual(['uncategorized']);
  });

  it('preserves old prefixed JSON mapping fields', () => {
    const rows = projectMappingJson({
      heart_rate: {
        mappings: [{
          s_source: 'heart_rate',
          s_slabel: 'Heart rate',
          target_study: 'GISSI-HF',
          'gissi-hf_target': 'HR',
          'gissi-hf_tlabel': 'Pulse',
          category: 'measurement'
        }]
      }
    });

    expect(rows).toEqual([
      expect.objectContaining({
        s_source: 'heart_rate',
        s_label: 'Heart rate',
        target_study: 'GISSI-HF',
        target: 'HR',
        target_label: 'Pulse',
        harmonization_status: 'pending'
      })
    ]);
  });
});
