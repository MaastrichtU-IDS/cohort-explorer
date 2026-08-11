// Shared building blocks for guided prompt assembly.
// Used by Prompt Studio (full canvas) and the Copilot's Guide tab.
import type {ComponentType} from 'react';
import {Activity, Columns, FileText, Star} from 'react-feather';

export interface Intent {
  id: string;
  label: string;
  icon: ComponentType<any>;
  template: (cohort: string, topic: string) => string;
  blurb: string;
}

export const intents: Intent[] = [
  {
    id: 'summarize',
    label: 'Summarize',
    icon: FileText,
    blurb: 'Get a concise overview of a cohort.',
    template: (c, t) =>
      c
        ? `Summarize the ${c} cohort${t ? ` with a focus on ${t}` : ''}. Include study design, population, and key variables.`
        : `Give me an overview of the cohort catalog${t ? ` with a focus on ${t}` : ''}.`
  },
  {
    id: 'compare',
    label: 'Compare',
    icon: Columns,
    blurb: 'Compare two or more cohorts side by side.',
    template: (c, t) =>
      c
        ? `Compare the ${c} cohort with other similar cohorts${t ? ` in terms of ${t}` : ''}.`
        : `Compare the cohorts in this catalog${t ? ` focusing on ${t}` : ''}.`
  },
  {
    id: 'explore',
    label: 'Explore variables',
    icon: Activity,
    blurb: 'Find variables and measurements of interest.',
    template: (c, t) =>
      c
        ? `What variables related to ${t || 'the main outcomes'} are available in the ${c} cohort?`
        : `Which cohorts have variables related to ${t || 'cardiovascular disease'}?`
  },
  {
    id: 'research',
    label: 'Research question',
    icon: Star,
    blurb: 'Generate a research question from the data.',
    template: (c, t) =>
      c
        ? `Suggest a research question that the ${c} cohort could answer${t ? ` about ${t}` : ''}.`
        : `Suggest a research question that these cohorts could answer${t ? ` about ${t}` : ''}.`
  }
];

export const topicBank = [
  'blood pressure',
  'cholesterol',
  'diabetes',
  'cardiovascular disease',
  'mortality',
  'medication use',
  'biomarkers',
  'lifestyle factors',
  'comorbidities',
  'study design',
  'population demographics',
  'longitudinal measurements'
];

// "A", "A and B", "A, B and C" — for splicing cohort ids into templates.
export function joinCohortLabel(ids: string[]): string {
  if (ids.length === 0) return '';
  if (ids.length === 1) return ids[0];
  return `${ids.slice(0, -1).join(', ')} and ${ids[ids.length - 1]}`;
}

export function assemblePrompt(intent: Intent, cohortIds: string[], topic: string): string {
  return intent.template(joinCohortLabel(cohortIds), topic.trim());
}
