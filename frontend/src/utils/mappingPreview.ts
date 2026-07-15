export interface MappingRow {
  [key: string]: string | number | boolean | null | undefined;
}

export interface MappingPreview {
  rows: MappingRow[];
}

export const mappingTargetElementId = (cohortId: string): string => `mapping-target-${cohortId}`;

export const parseCsvLine = (line: string): string[] => {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      inQuotes = !inQuotes;
    } else if (character === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += character;
    }
  }
  result.push(current);
  return result;
};

const categoriesWithLabels = (codes: string, labels: string): string => {
  if (codes && labels) return `${codes} (${labels})`;
  return codes || labels || '';
};

export const parseMappingPreview = (csvText: string, cohorts: string[] = []): MappingPreview => {
  const lines = csvText
    .trim()
    .split('\n')
    .filter(line => line.trim() !== '');
  if (lines.length < 2) return {rows: []};

  const headers = parseCsvLine(lines[0]).map(header => header.trim());
  const rows = lines
    .slice(1)
    .map(line => {
      const values = parseCsvLine(line);
      const csvRow: Record<string, string> = {};
      headers.forEach((header, index) => {
        csvRow[header] = (values[index] || '').trim();
      });

      const row: MappingRow = {...csvRow};
      row.s_source = csvRow.source || '';
      row.s_label = csvRow.slabel || csvRow.source_label || '';
      row.target_study = csvRow.target_study || cohorts[1] || '';
      row.target = csvRow.target || '';
      row.target_label = csvRow.tlabel || csvRow.target_label || '';
      row.mapping_relation = csvRow.mapping_relation || csvRow['mapping type'] || '';
      row.harmonization_status = csvRow.harmonization_status || '';
      row.sim_score = csvRow.sim_score ? Number(csvRow.sim_score) : null;
      row.omop_domain = csvRow.category || '';
      row.source_categories_codes_labels = categoriesWithLabels(
        csvRow.source_original_categories || '',
        csvRow.source_categories_labels || ''
      );
      row.target_categories_codes_labels = categoriesWithLabels(
        csvRow.target_original_categories || '',
        csvRow.target_categories_labels || ''
      );
      return row;
    })
    .filter(row => row.s_source || row.target);
  return {rows};
};

export const projectMappingJson = (jsonData: unknown): MappingRow[] => {
  if (typeof jsonData !== 'object' || jsonData === null) return [];
  const rows: MappingRow[] = [];

  Object.entries(jsonData).forEach(([sourceVariable, value]) => {
    if (!value || typeof value !== 'object' || !Array.isArray((value as {mappings?: unknown[]}).mappings)) return;
    (value as {mappings: Record<string, any>[]}).mappings.forEach(mapping => {
      const sourceLabels = mapping.s_source_categories_labels || mapping.source_categories_labels || '';
      const sourceCodes = mapping.s_source_original_categories || mapping.source_original_categories || '';
      const row: MappingRow = {
        s_source: mapping.s_source || mapping.source || sourceVariable,
        s_label: mapping.s_slabel || mapping.slabel || mapping.source_label || '',
        target_study: mapping.target_study,
        harmonization_status: mapping.harmonization_status || 'pending',
        source_categories_codes_labels: categoriesWithLabels(sourceCodes, sourceLabels),
        mapping_relation: mapping.mapping_relation || '',
        target: mapping.target || '',
        target_label: mapping.tlabel || mapping.target_label || ''
      };
      if (mapping.source_study) row.source_study = mapping.source_study;
      let targetLabels = mapping.target_categories_labels || '';
      let targetCodes = mapping.target_original_categories || '';

      Object.keys(mapping).forEach(key => {
        if (!row.target && key.endsWith('_target')) row.target = mapping[key];
        else if (!row.target_label && key.endsWith('_tlabel')) row.target_label = mapping[key];
        else if (!targetLabels && key.endsWith('_target_categories_labels')) targetLabels = mapping[key] || '';
        else if (!targetCodes && key.endsWith('_target_original_categories')) targetCodes = mapping[key] || '';
      });

      row.target_categories_codes_labels = categoriesWithLabels(targetCodes, targetLabels);
      row.omop_domain = mapping.category || '';
      row.sim_score = mapping.sim_score != null ? Number(mapping.sim_score) : null;
      rows.push(row);
    });
  });

  return rows;
};
