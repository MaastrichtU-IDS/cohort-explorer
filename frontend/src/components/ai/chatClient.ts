// Shared client + helpers for the experimental AI chat pages.
import {apiUrl} from '@/utils';
import {Cohort} from '@/types';

export type Role = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface ChatConfig {
  enabled: boolean;
  model: string;
}

export interface SendOptions {
  messages: ChatMessage[];
  cohortIds: string[];
  focus?: string;
  onChunk: (delta: string) => void;
  signal?: AbortSignal;
}

// Fetch whether the backend chat proxy is configured + the default model.
export async function fetchChatConfig(): Promise<ChatConfig> {
  try {
    const res = await fetch(`${apiUrl}/api/chat/config`, {credentials: 'include'});
    if (!res.ok) return {enabled: false, model: ''};
    return await res.json();
  } catch {
    return {enabled: false, model: ''};
  }
}

// Stream a completion. Reads the plain-text chunked body and forwards deltas.
export async function streamChat(opts: SendOptions): Promise<void> {
  const res = await fetch(`${apiUrl}/api/chat/stream`, {
    method: 'POST',
    credentials: 'include',
    headers: {'Content-Type': 'application/json'},
    signal: opts.signal,
    body: JSON.stringify({
      messages: opts.messages,
      cohort_ids: opts.cohortIds,
      focus: opts.focus || null
    })
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch {
      /* body was not JSON */
    }
    throw new Error(detail);
  }
  if (!res.body) {
    const text = await res.text();
    opts.onChunk(text);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    opts.onChunk(decoder.decode(value, {stream: true}));
  }
}

// ---- Context helpers driven by the client-side cohort cache ----------------

export interface CohortBrief {
  id: string;
  variableCount: number;
  studyType?: string;
  participants?: string;
  population?: string;
}

const isMeaningful = (v: any): boolean => {
  if (v === undefined || v === null) return false;
  const t = String(v).trim().toLowerCase();
  return t !== '' && !['na', 'n/a', 'nan', 'none', 'null', '-', '--'].includes(t);
};

export function toBriefs(cohortsData: {[id: string]: Cohort}): CohortBrief[] {
  return Object.values(cohortsData || {})
    .map(c => ({
      id: c.cohort_id,
      variableCount: Object.keys(c.variables || {}).length,
      studyType: isMeaningful(c.study_type) ? c.study_type : undefined,
      participants: isMeaningful(c.study_participants) ? c.study_participants : undefined,
      population: isMeaningful(c.study_population) ? c.study_population : undefined
    }))
    .sort((a, b) => b.variableCount - a.variableCount || a.id.localeCompare(b.id));
}

// Build a set of guided suggestion prompts from the current selection.
export function buildSuggestions(cohortsData: {[id: string]: Cohort}, selected: string[]): string[] {
  const ids = selected.filter(id => cohortsData[id]);
  if (ids.length === 0) {
    return [
      'Which cohorts have the most variables available?',
      'Which cohorts focus on diabetes or cardiovascular disease?',
      'Help me choose cohorts to compare blood pressure over time.',
      'Give me an overview of what this cohort catalog contains.'
    ];
  }
  if (ids.length === 1) {
    const c = cohortsData[ids[0]];
    return [
      `Summarize the ${c.cohort_id} cohort in a few bullet points.`,
      `What kinds of variables does ${c.cohort_id} collect?`,
      `List the measurements in ${c.cohort_id} related to blood pressure or cholesterol.`,
      `What is the study population and design of ${c.cohort_id}?`
    ];
  }
  const [a, b] = ids;
  return [
    `Compare the ${a} and ${b} cohorts: design, population and size.`,
    `Which variables are shared across the selected cohorts?`,
    `Where do the selected cohorts differ most in what they measure?`,
    `Suggest a research question these cohorts could answer together.`
  ];
}
