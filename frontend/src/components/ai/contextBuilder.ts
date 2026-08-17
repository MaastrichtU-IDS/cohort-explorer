// Client-side cohort-context assembly.
//
// The default chat backend builds its own context server-side (see
// backend/src/chat.py). The Glass Box layout instead builds the context here,
// in the browser, from the user's explicit choices — which metadata groups to
// share, how much variable detail — and sends it as an override. What the user
// previews is byte-for-byte what the model receives.
import {Cohort, Variable} from '@/types';

// ---- System prompt presets -------------------------------------------------

// Mirrors SYSTEM_PROMPT in backend/src/chat.py — keep in sync.
export const DEFAULT_SYSTEM_PROMPT =
  'You are the iCARE4CVD Cohort Explorer assistant. You help researchers ' +
  'understand and compare cardiovascular research cohorts and their variables. ' +
  'Answer using ONLY the cohort context provided in this conversation. If the ' +
  'context does not contain the answer, say so plainly and suggest what the user ' +
  'could select or ask instead. Be concise, use short paragraphs and bullet ' +
  'points, reference cohorts and variables by name, and never invent variables, ' +
  'values, or statistics that are not present in the context.';

export interface PromptPreset {
  id: string;
  label: string;
  blurb: string;
  text: string;
}

export const promptPresets: PromptPreset[] = [
  {
    id: 'default',
    label: 'Default assistant',
    blurb: 'The standard grounded cohort assistant.',
    text: DEFAULT_SYSTEM_PROMPT
  },
  {
    id: 'auditor',
    label: 'Strict auditor',
    blurb: 'Verbatim facts only, every claim attributed.',
    text:
      'You are a meticulous data auditor for the iCARE4CVD cohort catalog. Answer only with facts ' +
      'that appear verbatim in the cohort context. After every claim, name the cohort and metadata ' +
      'field it came from in parentheses, e.g. (TIME-CHF, study design). If the context does not ' +
      'contain the answer, reply exactly: "Not in the shared context." Do not speculate, do not ' +
      'generalize beyond the context, and never estimate numbers.'
  },
  {
    id: 'plain',
    label: 'Plain-language explainer',
    blurb: 'For non-specialists, no jargon.',
    text:
      'You explain the iCARE4CVD cohort catalog to readers without a medical or statistical ' +
      'background. Use only the cohort context provided. Use short sentences and everyday words; ' +
      'define any acronym or clinical term the first time it appears. Prefer concrete examples over ' +
      'abstractions. If the context does not contain the answer, say so simply.'
  },
  {
    id: 'methods',
    label: 'Methods consultant',
    blurb: 'Design critique and analysis ideas.',
    text:
      'You are an epidemiology methods consultant reviewing cohorts from the iCARE4CVD catalog. ' +
      'Ground every factual statement in the cohort context provided, and clearly separate FACTS ' +
      '(from the context) from SUGGESTIONS (your methodological input). Discuss design strengths ' +
      'and limitations, potential biases, and analysis strategies the data could support. Be ' +
      'specific about which variables or metadata motivate each suggestion.'
  }
];

// ---- Metadata groups -------------------------------------------------------

const clean = (v: any): string => {
  if (v === undefined || v === null) return '';
  const t = String(v).trim();
  return ['', 'na', 'n/a', 'nan', 'none', 'null', '-', '--'].includes(t.toLowerCase()) ? '' : t;
};

export interface FieldSpec {
  label: string;
  get: (c: Cohort) => string;
}

export interface GroupSpec {
  key: string;
  label: string;
  hint: string;
  fields: FieldSpec[];
}

const f = (label: string, get: (c: Cohort) => any): FieldSpec => ({label, get: c => clean(get(c))});

const sexSplit = (c: Cohort): string => {
  if (c.male_percentage == null && c.female_percentage == null) return '';
  const parts = [];
  if (c.male_percentage != null) parts.push(`${c.male_percentage}% male`);
  if (c.female_percentage != null) parts.push(`${c.female_percentage}% female`);
  return parts.join(' / ');
};

const ageDist = (c: Cohort): string => {
  const dist = c.age_distribution;
  if (!dist || Object.keys(dist).length === 0) return '';
  return Object.entries(dist)
    .map(([band, pct]) => `${band}: ${pct}%`)
    .join(', ');
};

export const metadataGroups: GroupSpec[] = [
  {
    key: 'design',
    label: 'Study design & timeline',
    hint: 'Type, design, start/end, cadence',
    fields: [
      f('Study type', c => c.study_type),
      f('Study design', c => c.study_design),
      f('Study start', c => c.study_start),
      f('Study end', c => c.study_end),
      f('Duration', c => c.study_duration),
      f('Ongoing', c => c.study_ongoing),
      f('Collection frequency', c => c.data_collection_frequency)
    ]
  },
  {
    key: 'population',
    label: 'Population & demographics',
    hint: 'Who is in the cohort',
    fields: [
      f('Participants', c => c.study_participants),
      f('Population', c => c.study_population),
      f('Location', c => c.population_location),
      f('Race / ethnicity', c => c.race_ethnicity),
      {label: 'Sex split', get: sexSplit},
      {label: 'Age distribution', get: ageDist}
    ]
  },
  {
    key: 'clinical',
    label: 'Objectives & outcomes',
    hint: 'Aims, outcomes, morbidity, interventions',
    fields: [
      f('Objective', c => c.study_objective),
      f('Primary outcome', c => c.primary_outcome_spec),
      f('Secondary outcome', c => c.secondary_outcome_spec),
      f('Morbidity', c => c.morbidity),
      f('Interventions', c => c.interventions),
      f('Enrolled with diabetes', c => c.enrolled_with_diabetes),
      f('Enrolled with CVD', c => c.enrolled_with_cvd)
    ]
  },
  {
    key: 'criteria',
    label: 'Inclusion & exclusion criteria',
    hint: 'Eligibility rules',
    fields: [
      f('Inclusion: sex', c => c.sex_inclusion),
      f('Inclusion: health status', c => c.health_status_inclusion),
      f('Inclusion: relevant exposure', c => c.clinically_relevant_exposure_inclusion),
      f('Inclusion: age group', c => c.age_group_inclusion),
      f('Inclusion: BMI range', c => c.bmi_range_inclusion),
      f('Inclusion: ethnicity', c => c.ethnicity_inclusion),
      f('Inclusion: family status', c => c.family_status_inclusion),
      f('Inclusion: hospital patient', c => c.hospital_patient_inclusion),
      f('Inclusion: medication use', c => c.use_of_medication_inclusion),
      f('Exclusion: health status', c => c.health_status_exclusion),
      f('Exclusion: BMI range', c => c.bmi_range_exclusion),
      f('Exclusion: limited life expectancy', c => c.limited_life_expectancy_exclusion),
      f('Exclusion: need for surgery', c => c.need_for_surgery_exclusion),
      f('Exclusion: surgical history', c => c.surgical_procedure_history_exclusion),
      f('Exclusion: relevant exposure', c => c.clinically_relevant_exposure_exclusion)
    ]
  },
  {
    key: 'provenance',
    label: 'Provenance & data format',
    hint: 'Institution, formats, coding systems',
    fields: [
      f('Institution', c => c.institution),
      f('Language', c => c.language),
      f('Dataset format', c => c.dataset_format),
      f('Coding system', c => c.coding_system),
      f('Anonymisation', c => c.anonymisation_technique),
      f('Part of study', c => c.part_of_study)
    ]
  }
];

// ---- Variables -------------------------------------------------------------

export type VariableDetail = 'off' | 'names' | 'detailed' | 'omop';

export const variableDetailOptions: {id: VariableDetail; label: string; blurb: string}[] = [
  {id: 'off', label: 'None', blurb: 'Share no variables'},
  {id: 'names', label: 'Names', blurb: 'Variable names only'},
  {id: 'detailed', label: 'Detailed', blurb: 'Names, labels, types, units'},
  {id: 'omop', label: 'Detailed + OMOP', blurb: 'Adds OMOP domain and mapped concept'}
];

function describeVariable(v: Variable, detail: VariableDetail): string {
  const name = clean(v.var_name) || '?';
  if (detail === 'names') return name;
  const bits: string[] = [];
  const label = clean(v.var_label);
  if (label && label.toLowerCase() !== name.toLowerCase()) bits.push(label);
  const meta = [clean(v.var_type), clean(v.units)].filter(Boolean);
  if (meta.length) bits.push(`[${meta.join(', ')}]`);
  if (detail === 'omop') {
    const omop = [clean(v.omop_domain), clean(v.concept_name)].filter(Boolean).join(': ');
    if (omop) bits.push(`(OMOP ${omop})`);
  }
  return bits.length ? `${name} — ${bits.join(' ')}` : name;
}

// ---- Context assembly ------------------------------------------------------

export interface ContextOptions {
  groups: {[key: string]: boolean};
  variableDetail: VariableDetail;
  maxVars: number;
}

export const defaultContextOptions: ContextOptions = {
  groups: Object.fromEntries(metadataGroups.map(g => [g.key, true])),
  variableDetail: 'detailed',
  maxVars: 40
};

export function buildClientContext(
  selected: Cohort[],
  allCohorts: Cohort[],
  opts: ContextOptions
): string {
  if (selected.length === 0) {
    const withVars = allCohorts.filter(c => Object.keys(c.variables || {}).length > 0);
    const lines = [
      `No specific cohort is selected. Catalog of ${allCohorts.length} cohorts (${withVars.length} with uploaded variables):`
    ];
    for (const c of allCohorts) {
      const stype = clean(c.study_type);
      lines.push(`- ${c.cohort_id}${stype ? ` (${stype})` : ''}: ${Object.keys(c.variables || {}).length} variables`);
    }
    return lines.join('\n');
  }

  const parts: string[] = [`The user is focusing on ${selected.length} cohort(s):`];
  for (const c of selected) {
    const lines = [`### Cohort: ${c.cohort_id}`];
    for (const group of metadataGroups) {
      if (!opts.groups[group.key]) continue;
      for (const field of group.fields) {
        const value = field.get(c);
        if (value) lines.push(`- ${field.label}: ${value}`);
      }
    }
    const variables = Object.values(c.variables || {});
    lines.push(`- Variable count: ${variables.length}`);
    if (opts.variableDetail !== 'off' && variables.length > 0) {
      const sample = variables.slice(0, opts.maxVars);
      lines.push('- Variables (sample):');
      for (const v of sample) {
        lines.push(`    - ${describeVariable(v, opts.variableDetail)}`);
      }
      if (variables.length > opts.maxVars) {
        lines.push(`    - …and ${variables.length - opts.maxVars} more variables`);
      }
    }
    parts.push(lines.join('\n'));
  }
  return parts.join('\n\n');
}

// Rough but honest: ~4 characters per token for English prose.
export const estimateTokens = (text: string): number => Math.round(text.length / 4);
