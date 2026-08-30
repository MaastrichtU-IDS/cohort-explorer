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
  // Catalog searches run for this turn (planning round): rendered as a
  // dedicated search panel above the assistant's answer.
  searches?: SearchRun[];
  searchTerms?: string[];
  searchConcepts?: SearchConcept[];
  searchIntersection?: IntersectionRow[] | null;
  // The user opened this answer's Summary tab at least once (Detailed is the
  // default view); stored with the conversation history.
  summaryViewed?: boolean;
  // The planning round failed (endpoint error or server-side search error):
  // shown as a small note so failures are visible instead of silent.
  searchError?: string;
  // Disambiguation turn: the planner found the question ambiguous and this
  // reply only asks which reading is meant (shown with its own pink tag).
  clarify?: boolean;
}

// ---- Catalog search (the chat's search tool) --------------------------------

export interface SearchVariable {
  var_name: string;
  var_label?: string;
  concept_name?: string;
  omop_domain?: string;
  var_type?: string;
  units?: string;
  visits?: string;
  categorical?: boolean;
  // The variable has an EDA entry (and usually a distribution graph): the UI
  // shows a clickable chart icon opening the EDA overlay.
  has_eda?: boolean;
  // Matched through a shared standard code rather than the search text itself
  // (e.g. BB_3M counts as a beta blocker via ATC:C07A).
  via_code?: boolean;
  matched_code?: string;
  equivalents?: {cohort_id: string; var_name: string}[];
}

export interface SearchCohort {
  cohort_id: string;
  matches: number;
  text_matches?: number;
  code_matches?: number;
  in_selection?: boolean;
  // Top cohort whose variable list goes into the model's context; the panel
  // shows every cohort's (capped) list on click regardless.
  detailed?: boolean;
  variables: SearchVariable[];
}

export interface SearchRun {
  term: string;
  total_matches: number;
  cohorts_matched: number;
  // Standard codes that pulled equivalent variables into the results.
  codes?: {code: string; display: string}[];
  cohorts: SearchCohort[];
}

// A concept groups the terms that stand for one criterion of the question
// (e.g. "beta blockers" = beta blocker, bisoprolol, ...). When a question has
// several criteria, the server also computes which cohorts match EVERY concept.
export interface SearchConcept {
  name: string;
  terms: string[];
  cohorts: Record<string, number>;
}

export interface IntersectionRow {
  cohort_id: string;
  per_concept: Record<string, number>;
}

export interface SearchPayload {
  runs: SearchRun[];
  concepts?: SearchConcept[];
  intersection?: IntersectionRow[] | null;
}

// Planning round: the model proposes search terms for the question; the server
// runs them through the catalog search and returns the structured results.
export async function planSearch(
  question: string,
  cohortIds: string[],
  history: ChatMessage[]
): Promise<{
  needed: boolean;
  terms: string[];
  searches: SearchRun[];
  concepts?: SearchConcept[];
  intersection?: IntersectionRow[] | null;
  interpretations?: string[];
}> {
  const res = await fetch(`${apiUrl}/api/chat/plan-search`, {
    method: 'POST',
    credentials: 'include',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      question,
      cohort_ids: cohortIds,
      history: history.map(m => ({role: m.role, content: m.role === 'assistant' ? m.detailed || m.content : m.content}))
    })
  });
  if (!res.ok) throw new Error(`Search planning failed (${res.status})`);
  const j = await res.json();
  if (j.error) throw new Error(String(j.error));
  return {
    needed: !!j.needed,
    terms: j.terms || [],
    searches: Array.isArray(j.searches) ? j.searches : [],
    concepts: Array.isArray(j.concepts) ? j.concepts : undefined,
    intersection: Array.isArray(j.intersection) ? j.intersection : null,
    interpretations: Array.isArray(j.interpretations) ? j.interpretations : []
  };
}

// One retry on failure: the first call after a deploy can hit a cold model.
export async function planSearchWithRetry(question: string, cohortIds: string[], history: ChatMessage[]) {
  try {
    return await planSearch(question, cohortIds, history);
  } catch {
    return await planSearch(question, cohortIds, history);
  }
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
  // Structured results of the planning round's catalog searches (runs plus the
  // concept grouping and cross-concept intersection): injected into the model's
  // context server-side, identical to what the search panel shows.
  searchResults?: SearchPayload;
  // Disambiguation turn: the readings the planner found. The server then asks
  // for a short clarifying reply instead of a full answer.
  clarifyInterpretations?: string[];
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
      style: opts.style || null,
      search_results: opts.searchResults && opts.searchResults.runs.length > 0 ? opts.searchResults : null,
      clarify_interpretations: opts.clarifyInterpretations && opts.clarifyInterpretations.length >= 2 ? opts.clarifyInterpretations : null
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

export const adminAddStarter = (text: string, kind: 'interesting' | 'basic') =>
  adminPost('/api/chat/starters/add', {text, kind});

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

// ---- Cross-cohort mapping availability --------------------------------------
// For each pair of selected cohorts, the backend reports whether a cached
// mapping file exists (the assistant grounds cross-cohort answers in it) or
// not (the chat shows a "generate the mapping" button).

export interface MappingPairStatus {
  source: string;
  target: string;
  cached: boolean;
  filename: string | null;
  generated_at?: string;
}

export async function fetchMappingStatus(cohortIds: string[]): Promise<MappingPairStatus[]> {
  if (cohortIds.length < 2) return [];
  const res = await fetch(
    `${apiUrl}/api/chat/mapping-status?cohort_ids=${encodeURIComponent(cohortIds.join(','))}`,
    {credentials: 'include'}
  );
  if (!res.ok) throw new Error(`Failed to check mapping status (${res.status})`);
  const data = await res.json();
  return data.pairs || [];
}

// Generate the mapping for one pair via the same endpoint the mapping page
// uses. Slow (can take minutes): callers should show a progress state.
export async function generateMappingPair(source: string, target: string): Promise<void> {
  const res = await fetch(`${apiUrl}/api/generate-mapping`, {
    method: 'POST',
    credentials: 'include',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source_study: source, target_studies: [[target, false]]})
  });
  if (!res.ok) {
    let detail = `Mapping generation failed (${res.status})`;
    try {
      const j = await res.json();
      detail = j.detail || j.error || detail;
    } catch {
      /* body was not JSON */
    }
    throw new Error(detail);
  }
}

// ---- Conversation history ---------------------------------------------------
// The client upserts the full transcript after each completed turn; the backend
// (src/ai_history.py) derives usage metrics. Each user sees their own history;
// admins can request scope='all'.

export type ArrivalPath = 'chat' | 'intention_cards';

// A conversation as returned by the list endpoint (metrics + preview, no full
// transcript). The detail endpoint returns the same shape plus `messages`.
export interface ConversationSummary {
  id: string;
  user_id: string;
  arrival_path: ArrivalPath | string;
  model: string;
  started_at: string;
  created_at: string;
  updated_at: string;
  duration_seconds: number | null;
  message_count: number;
  user_message_count: number;
  assistant_message_count: number;
  user_chars: number;
  assistant_chars: number;
  preview: string;
  entry_context: Record<string, any>;
}

export interface ConversationDetail extends ConversationSummary {
  messages: ChatMessage[];
}

export interface HistoryPage {
  total: number;
  limit: number;
  offset: number;
  scope: 'own' | 'all';
  items: ConversationSummary[];
}

export interface UsageSummary {
  scope: 'own' | 'all';
  conversations: number;
  messages: number;
  user_messages: number;
  assistant_messages: number;
  users: number;
  avg_messages: number | null;
  avg_duration_seconds: number | null;
  user_chars: number;
  assistant_chars: number;
  by_path: Record<string, number>;
  by_day: {day: string; count: number}[];
  top_users: {user_id: string; conversations: number}[];
}

export interface SaveConversationPayload {
  conversationId: string;
  startedAt: string;
  arrivalPath: ArrivalPath;
  entryContext?: Record<string, any>;
  model?: string;
  messages: ChatMessage[];
}

// Persist (upsert) a conversation. Fire-and-forget from the caller's view:
// failures are logged, never surfaced, so history never breaks the chat.
export async function saveConversation(payload: SaveConversationPayload): Promise<void> {
  try {
    await fetch(`${apiUrl}/api/chat/history`, {
      method: 'POST',
      credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        conversation_id: payload.conversationId,
        started_at: payload.startedAt,
        arrival_path: payload.arrivalPath,
        entry_context: payload.entryContext || {},
        model: payload.model || null,
        messages: payload.messages,
        summary_clicked: payload.messages.some(m => (m as ChatMessage).summaryViewed) || false
      })
    });
  } catch (err) {
    // Non-fatal: the conversation continues even if history can't be saved.
    // eslint-disable-next-line no-console
    console.warn('Failed to save conversation history', err);
  }
}

export async function fetchHistory(params?: {
  scope?: 'own' | 'all';
  path?: string;
  search?: string;
  minMessages?: number;
  maxMessages?: number;
  limit?: number;
  offset?: number;
}): Promise<HistoryPage> {
  const q = new URLSearchParams();
  if (params?.scope) q.set('scope', params.scope);
  if (params?.path) q.set('path', params.path);
  if (params?.search) q.set('search', params.search);
  if (params?.minMessages != null) q.set('min_messages', String(params.minMessages));
  if (params?.maxMessages != null) q.set('max_messages', String(params.maxMessages));
  if (params?.limit != null) q.set('limit', String(params.limit));
  if (params?.offset != null) q.set('offset', String(params.offset));
  const res = await fetch(`${apiUrl}/api/chat/history?${q.toString()}`, {credentials: 'include'});
  if (!res.ok) throw new Error(`Failed to load history (${res.status})`);
  return await res.json();
}

export async function fetchConversation(id: string): Promise<ConversationDetail> {
  const res = await fetch(`${apiUrl}/api/chat/history/${encodeURIComponent(id)}`, {
    credentials: 'include'
  });
  if (!res.ok) throw new Error(`Failed to load conversation (${res.status})`);
  return await res.json();
}

export async function fetchUsageSummary(scope: 'own' | 'all' = 'own'): Promise<UsageSummary> {
  const res = await fetch(`${apiUrl}/api/chat/history/summary?scope=${scope}`, {
    credentials: 'include'
  });
  if (!res.ok) throw new Error(`Failed to load usage summary (${res.status})`);
  return await res.json();
}
