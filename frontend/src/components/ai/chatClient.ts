// Shared client + helpers for the experimental AI chat pages.
import {apiUrl} from '@/utils';
import {Cohort} from '@/types';

export type Role = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  role: Role;
  content: string;
  // Assistant turns are answered twice — once briefly, once in depth — and the
  // bubble lets the user toggle between the two variants.
  summary?: string;
  detailed?: string;
}

export type AnswerStyle = 'summary' | 'detailed';

export interface ChatConfig {
  enabled: boolean;
  model: string;
}

export interface SendOptions {
  messages: ChatMessage[];
  cohortIds: string[];
  focus?: string;
  // Optional overrides: replace the server's default instructions and/or the
  // server-built cohort context with client-supplied text. Used by layouts
  // that give the user direct control over what the model sees.
  systemPrompt?: string;
  contextOverride?: string;
  // Answer style: the backend appends a matching instruction (short summary vs
  // in-depth). Omit for the default, unconstrained style.
  style?: AnswerStyle;
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
      focus: opts.focus || null,
      system_prompt: opts.systemPrompt || null,
      context: opts.contextOverride || null,
      style: opts.style || null
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

// ---- Conversation starters (model-generated pool, served by the backend) ----
// Admin-managed via /ai/starters; the chat landing page shows a random sample.

export interface ConversationStarter {
  text: string;
  kind: 'interesting' | 'basic';
  // Up to 3 keyword themes this starter belongs to (from the grouping pass).
  keywords?: string[];
}

export async function fetchConversationStarters(n = 6): Promise<ConversationStarter[]> {
  try {
    const res = await fetch(`${apiUrl}/api/chat/conversation-starters?n=${n}`, {credentials: 'include'});
    if (!res.ok) return [];
    const j = await res.json();
    return Array.isArray(j.starters) ? j.starters : [];
  } catch {
    return [];
  }
}

// Thematic keyword groups derived from the starter pool. Shown as the next
// selection after "Formulate Research Questions" in Guided Exploration.
export interface StarterKeyword {
  keyword: string;
  count: number;
  questions: string[];
}

export async function fetchStarterKeywords(): Promise<StarterKeyword[]> {
  try {
    const res = await fetch(`${apiUrl}/api/chat/starter-keywords`, {credentials: 'include'});
    if (!res.ok) return [];
    const j = await res.json();
    return Array.isArray(j.keywords) ? j.keywords : [];
  } catch {
    return [];
  }
}

// ---- Admin management of the starter pool (used by /ai/starters) ------------

export interface StarterPoolEntry extends ConversationStarter {
  generated_at?: string;
  model?: string;
  direction?: string;
}

export interface StarterManageData {
  chat_enabled: boolean;
  model: string;
  starters: StarterPoolEntry[];
  keywords: StarterKeyword[];
  keywords_meta: {generated_at?: string; model?: string; pool_size?: number};
}

async function adminPost(path: string, body?: object): Promise<any> {
  const res = await fetch(`${apiUrl}${path}`, {
    method: 'POST',
    credentials: 'include',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {})
  });
  if (!res.ok) {
    const j = await res.json().catch(() => ({}));
    throw new Error(j.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export async function adminFetchStarterPool(): Promise<StarterManageData> {
  const res = await fetch(`${apiUrl}/api/chat/starters/manage`, {credentials: 'include'});
  if (!res.ok) {
    const j = await res.json().catch(() => ({}));
    throw new Error(j.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export const adminGenerateStarters = (direction: string) =>
  adminPost('/api/chat/starters/generate', {direction: direction.trim() || null});

export const adminRegroupStarters = () => adminPost('/api/chat/starters/regroup');

export const adminDeleteStarters = (texts: string[]) =>
  adminPost('/api/chat/starters/delete', {texts});

export interface ContextDiagnostics {
  sizes: {
    n_cohorts: number;
    n_variables: number;
    n_distinct_concepts: number;
    current_catalog_context_tokens: number;
    concept_index_tokens: number;
    full_detail_tokens: number;
  };
  model_info: {max_input_tokens?: number; max_tokens?: number; max_output_tokens?: number} | null;
  model_info_error?: string;
  window_probe?: {approx_tokens: number; ok: boolean; error?: string}[];
}

export const adminContextDiagnostics = (probeWindow: boolean): Promise<ContextDiagnostics> =>
  adminPost('/api/chat/starters/context-diagnostics', {probe_window: probeWindow});

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
