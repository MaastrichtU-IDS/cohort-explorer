// Shared building blocks for guided prompt assembly.
// `intents` powers the Prompt Studio (alternatives); `guidedIntents` powers the
// Guided Exploration mode of the main iCARE-AI page.
import type {ComponentType} from 'react';
import {Activity, Columns, FileText, Filter, HelpCircle, Star} from 'react-feather';

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

// ---- Guided Exploration intents (main iCARE-AI page) -----------------------

export interface GuidedIntent {
  id: string;
  label: string;
  icon: ComponentType<any>;
  blurb: string;
  // Label + placeholder for the topic input; some intents reinterpret "topic"
  // (e.g. as criteria, or as the hypothesis to explore).
  topicLabel: string;
  topicPlaceholder: string;
  template: (cohort: string, topic: string) => string;
}

export const guidedIntents: GuidedIntent[] = [
  {
    id: 'identify',
    label: 'Identify Cohorts Based on Criteria',
    icon: Filter,
    blurb: 'Find the cohorts that match your inclusion criteria.',
    topicLabel: 'Your criteria',
    topicPlaceholder: 'e.g. elderly patients with heart failure and medication data',
    template: (c, t) =>
      t
        ? `Identify ${c ? `which of ${c}` : 'cohorts in the catalog that'} match the following criteria: ${t}. For each candidate, explain briefly why it matches and note any caveats.`
        : `What kinds of populations and data does this catalog cover? Help me figure out which criteria I could use to identify suitable cohorts.`
  },
  {
    id: 'compare',
    label: 'Compare Cohorts',
    icon: Columns,
    blurb: 'Put two or more cohorts side by side.',
    topicLabel: 'Focus on a topic (optional)',
    topicPlaceholder: '…or type your own topic',
    template: (c, t) =>
      c
        ? `Compare ${c}: study design, population, size and what they measure${t ? `, focusing on ${t}` : ''}. Highlight the most important differences.`
        : `Compare the cohorts in this catalog${t ? `, focusing on ${t}` : ''}. Highlight the most important differences and suggest which are most similar.`
  },
  {
    id: 'hypothesis',
    label: 'Explore a Hypothesis',
    icon: HelpCircle,
    blurb: 'Check whether the data could support your hypothesis.',
    topicLabel: 'Your hypothesis',
    topicPlaceholder: 'e.g. beta-blocker use is associated with slower cognitive decline in the elderly',
    template: (c, t) =>
      t
        ? `I want to explore this hypothesis: "${t}". ${c ? `Focusing on ${c}, assess` : 'Assess'} which cohorts and variables could support investigating it, suggest a study design, and note limitations.`
        : `Help me formulate a hypothesis that ${c ? `${c} could` : 'the cohorts in this catalog could'} realistically support investigating.`
  },
  {
    id: 'research',
    label: 'Formulate Research Questions',
    icon: Star,
    blurb: 'Generate research questions the data could answer.',
    topicLabel: 'Focus on a topic (optional)',
    topicPlaceholder: '…or type your own topic',
    template: (c, t) =>
      `Suggest research questions that ${c ? `the ${c} cohort(s)` : 'these cohorts'} could answer${t ? ` about ${t}` : ''}, and for each note which variables would be involved.`
  },
  {
    id: 'variables',
    label: 'Explore Variables & Measurements',
    icon: Activity,
    blurb: 'Find out what is measured, where, and how.',
    topicLabel: 'Focus on a topic (optional)',
    topicPlaceholder: '…or type your own topic',
    template: (c, t) =>
      c
        ? `What variables related to ${t || 'the main outcomes'} are available in ${c}? Include how they are measured (units, visits) where known.`
        : `Which cohorts have variables related to ${t || 'cardiovascular disease'}, and what exactly do they measure?`
  },
  {
    id: 'summarize',
    label: 'Summarize a Cohort',
    icon: FileText,
    blurb: 'Get a concise overview of a cohort or the whole catalog.',
    topicLabel: 'Focus on a topic (optional)',
    topicPlaceholder: '…or type your own topic',
    template: (c, t) =>
      c
        ? `Summarize ${c}${t ? ` with a focus on ${t}` : ''}: study design, population, and key variables.`
        : `Give me an overview of the cohort catalog${t ? ` with a focus on ${t}` : ''}.`
  }
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
