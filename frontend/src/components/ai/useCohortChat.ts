// Shared conversation state + streaming logic reused by every AI layout.
import {useCallback, useEffect, useRef, useState} from 'react';
import {
  ArrivalPath,
  ChatMessage,
  ContextInfo,
  ConversationDetail,
  ProgressStep,
  SearchPayload,
  fetchChatConfig,
  fetchEdaFollowup,
  planSearchWithRetry,
  saveConversation,
  streamChat
} from '@/components/ai/chatClient';

export interface SendOverrides {
  systemPrompt?: string;
  contextOverride?: string;
  // Start a fresh conversation for this turn, discarding prior messages (used
  // when Guided Exploration sends its assembled question).
  startNew?: boolean;
  // How this conversation was entered — recorded in history. Defaults to 'chat';
  // Guided Exploration passes 'intention_cards'.
  arrivalPath?: ArrivalPath;
  // Extra context to store with the conversation (intent, topics, starter…).
  entryContext?: Record<string, any>;
}

// One-line summary of what the backend put into a request's context.
export function describeContext(info: ContextInfo): string {
  const bits: string[] = [];
  if (info.cohorts) {
    const detail =
      info.detail === 'labels' ? 'listed with labels' : info.detail === 'names' ? 'listed by name' : 'not listed';
    bits.push(`${info.cohorts} cohort${info.cohorts === 1 ? '' : 's'} (metadata)`);
    if (info.variables) bits.push(`${info.variables.toLocaleString()} variables (${detail})`);
  }
  if (info.search_cohorts)
    bits.push(`search results for ${info.search_cohorts} cohort${info.search_cohorts === 1 ? '' : 's'}`);
  if (info.related_variables) bits.push(`${info.related_variables} related variables`);
  if (info.equivalent_clusters)
    bits.push(`${info.equivalent_clusters} cross-cohort concept${info.equivalent_clusters === 1 ? '' : 's'}`);
  if (info.approx_tokens) {
    const t = info.approx_tokens;
    bits.push(`~${t >= 1000 ? `${Math.round(t / 1000)}k` : t} tokens`);
  }
  return bits.join(' · ');
}

// A stable per-conversation id, best-effort (crypto.randomUUID where available).
function newConversationId(): string {
  try {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  } catch {
    /* fall through */
  }
  return `conv-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export interface UseCohortChat {
  messages: ChatMessage[];
  input: string;
  setInput: (v: string) => void;
  selected: string[];
  toggleCohort: (id: string) => void;
  clearSelection: () => void;
  focus: string;
  setFocus: (v: string) => void;
  isStreaming: boolean;
  enabled: boolean;
  model: string;
  configLoaded: boolean;
  error: string | null;
  send: (text?: string, overrides?: SendOverrides) => Promise<void>;
  markSummaryViewed: (index: number) => void;
  stop: () => void;
  reset: () => void;
  // Resume a stored conversation: restores the transcript and the conversation
  // identity, so follow-up turns keep updating the same history record.
  loadConversation: (detail: ConversationDetail) => void;
}

export function useCohortChat(): UseCohortChat {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [selected, setSelected] = useState<string[]>([]);
  const [focus, setFocus] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const [model, setModel] = useState('');
  const [configLoaded, setConfigLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Conversation identity for history persistence. Set when a conversation
  // begins (first turn or an explicit startNew) and reused across follow-ups.
  const conversationIdRef = useRef<string | null>(null);
  const startedAtRef = useRef<string | null>(null);
  const arrivalPathRef = useRef<ArrivalPath>('chat');
  const entryContextRef = useRef<Record<string, any>>({});

  useEffect(() => {
    fetchChatConfig().then(cfg => {
      setEnabled(cfg.enabled);
      setModel(cfg.model);
      setConfigLoaded(true);
    });
  }, []);

  const toggleCohort = useCallback((id: string) => {
    setSelected(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]));
  }, []);

  const clearSelection = useCallback(() => setSelected([]), []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
  }, []);

  // The user opened an answer's Summary tab: record it on the message and
  // re-save the conversation so the history carries the fact.
  const markSummaryViewed = useCallback(
    (index: number) => {
      setMessages(prev => {
        const target = prev[index];
        if (!target || target.role !== 'assistant' || target.summaryViewed) return prev;
        const next = [...prev];
        next[index] = {...target, summaryViewed: true};
        if (conversationIdRef.current) {
          void saveConversation({
            conversationId: conversationIdRef.current,
            startedAt: startedAtRef.current || new Date().toISOString(),
            arrivalPath: arrivalPathRef.current,
            entryContext: entryContextRef.current,
            model,
            messages: next.map(({progress, ...m}) => m)
          });
        }
        return next;
      });
    },
    [model]
  );

  const reset = useCallback(() => {
    stop();
    setMessages([]);
    setError(null);
    // Next send starts a brand-new conversation record.
    conversationIdRef.current = null;
  }, [stop]);

  const loadConversation = useCallback(
    (detail: ConversationDetail) => {
      stop();
      setError(null);
      setMessages(Array.isArray(detail.messages) ? detail.messages : []);
      conversationIdRef.current = detail.id;
      startedAtRef.current = detail.started_at || null;
      arrivalPathRef.current = (detail.arrival_path as ArrivalPath) || 'chat';
      entryContextRef.current = detail.entry_context || {};
      // Restore the pinned cohort scope the conversation was started with, so
      // follow-up turns keep the same context as the original exchange.
      const cohortIds = detail.entry_context?.cohortIds;
      setSelected(Array.isArray(cohortIds) ? cohortIds.filter((c: any) => typeof c === 'string') : []);
      if (typeof detail.entry_context?.focus === 'string') setFocus(detail.entry_context.focus);
    },
    [stop]
  );

  const send = useCallback(
    async (text?: string, overrides?: SendOverrides) => {
      const content = (text ?? input).trim();
      if (!content || isStreaming) return;
      setError(null);
      setInput('');

      // A new conversation starts from an empty history.
      const base = overrides?.startNew ? [] : messages;

      // Establish conversation identity for history. A fresh record begins on an
      // explicit startNew, or on the first turn of an otherwise-empty chat.
      const startingNew = overrides?.startNew || !conversationIdRef.current || base.length === 0;
      if (startingNew) {
        conversationIdRef.current = newConversationId();
        startedAtRef.current = new Date().toISOString();
        arrivalPathRef.current = overrides?.arrivalPath || 'chat';
        entryContextRef.current = overrides?.entryContext || {cohortIds: selected, focus};
      }

      // For follow-up turns the model sees the DETAILED variant of earlier
      // answers (that is the fuller record of what was said).
      const historyForModel: ChatMessage[] = [...base, {role: 'user' as const, content}].map(m =>
        m.role === 'assistant' ? {role: m.role, content: m.detailed || m.content} : {role: m.role, content: m.content}
      );

      // Live progress of this turn, shown inside the assistant bubble while it
      // is answered (the first words can take a while with a large context).
      // Kept on the message at mainIdx and removed again when the turn ends.
      const mainIdx = base.length + 1;
      const step = (key: string, label: string): ProgressStep => ({key, label, state: 'pending'});
      const initialProgress: ProgressStep[] = [
        ...(overrides?.contextOverride ? [] : [step('search', 'Planning and running the catalog search')]),
        step('read', 'Reading the catalog data'),
        step('detailed', 'Writing the detailed answer'),
        step('summary', 'Condensing it into a short summary')
      ];
      const updateProgress = (fn: (steps: ProgressStep[]) => ProgressStep[] | undefined) =>
        setMessages(prev => {
          const m = prev[mainIdx];
          if (!m || m.role !== 'assistant' || !m.progress) return prev;
          const next = [...prev];
          next[mainIdx] = {...m, progress: fn(m.progress)};
          return next;
        });
      const patchStep = (key: string, patch: Partial<ProgressStep>) =>
        updateProgress(steps => steps.map(s => (s.key === key ? {...s, ...patch} : s)));
      const startStep = (key: string, detail?: string) =>
        patchStep(key, {state: 'active', startedAt: Date.now(), ...(detail !== undefined ? {detail} : {})});
      const finishStep = (key: string, detail?: string, state: 'done' | 'failed' = 'done') =>
        patchStep(key, {state, endedAt: Date.now(), ...(detail !== undefined ? {detail} : {})});
      const clearProgress = () => updateProgress(() => undefined);

      // Add the user turn plus an empty assistant turn holding both variants,
      // each streamed by its own request.
      setMessages([
        ...base,
        {role: 'user', content},
        {role: 'assistant', content: '', summary: '', detailed: '', progress: initialProgress}
      ]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      // Planning round: does this question involve finding cohorts/variables?
      // If so the model proposes terms, the server runs the catalog search, the
      // results appear in the search panel and go into both answers' context.
      let payload: SearchPayload | undefined;
      let searchTerms: string[] = [];
      let interpretations: string[] = [];
      if (!overrides?.contextOverride) {
        startStep('search', 'Choosing search terms for your question…');
        try {
          const plan = await planSearchWithRetry(content, selected, base);
          if (plan.needed && plan.searches.length > 0) {
            const matched = new Set(plan.searches.flatMap(r => r.cohorts.map(c => c.cohort_id)));
            finishStep(
              'search',
              `${plan.terms.length} search term${plan.terms.length === 1 ? '' : 's'} (${plan.terms.join(', ')}) → ` +
                `${matched.size} cohort${matched.size === 1 ? '' : 's'} matched`
            );
            payload = {runs: plan.searches, concepts: plan.concepts, intersection: plan.intersection};
            searchTerms = plan.terms;
            interpretations = plan.interpretations || [];
            // A disambiguation turn shows no search panel: the short clarifying
            // reply carries the preliminary numbers itself.
            if (interpretations.length < 2) {
              setMessages(prev => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last && last.role === 'assistant')
                  next[next.length - 1] = {...last, searches: plan.searches, searchTerms, searchConcepts: plan.concepts, searchIntersection: plan.intersection};
                return next;
              });
              // Search-based answers are followed by a check of the recorded
              // summary statistics (see the EDA follow-up below).
              updateProgress(steps => [...steps, step('stats', 'Checking the summary statistics for numbers')]);
            }
          } else {
            finishStep('search', 'No catalog search needed for this question');
          }
        } catch (e: any) {
          // Planning is best-effort (the answer falls back to single-round
          // retrieval), but the failure is shown, not swallowed.
          const searchError = e?.message || 'catalog search failed';
          finishStep('search', 'The search could not run; answering from the catalog data only', 'failed');
          setMessages(prev => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last && last.role === 'assistant') next[next.length - 1] = {...last, searchError};
            return next;
          });
        }
        if (controller.signal.aborted) {
          clearProgress();
          setIsStreaming(false);
          abortRef.current = null;
          return;
        }
      }

      // Disambiguation turn: ONE short clarifying reply (no summary/detailed
      // pair, no search panel) that sketches each reading and asks which one
      // is meant. The user's next message re-plans from scratch.
      if (payload && interpretations.length >= 2) {
        setMessages(prev => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === 'assistant')
            next[next.length - 1] = {
              role: 'assistant',
              content: '',
              clarify: true,
              progress: [
                ...(last.progress || []).filter(s => s.key === 'search'),
                step('read', 'Reading the catalog data'),
                step('clarify', 'Writing a clarifying question')
              ]
            };
          return next;
        });
        let clarifyText = '';
        let clarifyStarted = false;
        startStep('read', 'Waiting for the model to take in the catalog data…');
        try {
          await streamChat({
            messages: historyForModel,
            cohortIds: selected,
            focus,
            signal: controller.signal,
            searchResults: payload,
            clarifyInterpretations: interpretations,
            onContext: info => patchStep('read', {detail: describeContext(info)}),
            onChunk: delta => {
              if (!clarifyStarted) {
                clarifyStarted = true;
                finishStep('read');
                startStep('clarify');
              }
              clarifyText += delta;
              setMessages(prev => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last && last.role === 'assistant') next[next.length - 1] = {...last, content: (last.content || '') + delta};
                return next;
              });
            }
          });
          if (conversationIdRef.current) {
            void saveConversation({
              conversationId: conversationIdRef.current,
              startedAt: startedAtRef.current || new Date().toISOString(),
              arrivalPath: arrivalPathRef.current,
              entryContext: entryContextRef.current,
              model,
              messages: [...base, {role: 'user', content}, {role: 'assistant', content: clarifyText, clarify: true}]
            });
          }
        } catch (e: any) {
          if (e?.name !== 'AbortError') setError(e?.message || 'Something went wrong contacting the model.');
        }
        clearProgress();
        setIsStreaming(false);
        abortRef.current = null;
        return;
      }

      // Accumulate each variant locally too, so we can persist the final
      // transcript without reading React state back out.
      const acc: {summary: string; detailed: string} = {summary: '', detailed: ''};

      const streamVariant = async (style: 'summary' | 'detailed', summarizeText?: string) => {
        // The detailed request carries the big context: its wait before the
        // first words is shown as "reading", then "writing".
        let started = false;
        if (style === 'detailed') startStep('read', 'Sending the catalog data to the model…');
        else startStep('summary');
        try {
          await streamChat({
            messages: historyForModel,
            cohortIds: selected,
            focus,
            systemPrompt: overrides?.systemPrompt,
            contextOverride: overrides?.contextOverride,
            style,
            // In summarize mode the server uses the search results only for the
            // list of cohorts the summary must keep.
            searchResults: payload,
            summarizeText,
            signal: controller.signal,
            onContext: info => {
              if (style === 'detailed') patchStep('read', {detail: describeContext(info)});
            },
            onChunk: delta => {
              if (!started && style === 'detailed') {
                finishStep('read');
                startStep('detailed');
              }
              started = true;
              acc[style] += delta;
              setMessages(prev => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last && last.role === 'assistant') {
                  next[next.length - 1] = {...last, [style]: (last[style] || '') + delta};
                }
                return next;
              });
            }
          });
          if (style === 'detailed' && !started) finishStep('read');
          finishStep(style);
        } catch (e) {
          if (style === 'detailed' && !started) finishStep('read', undefined, 'failed');
          finishStep(style, undefined, 'failed');
          throw e;
        }
      };

      // Detailed first (it is the default view); the summary is then produced
      // BY SUMMARIZING the finished detailed answer, so the two variants can
      // never diverge. If the detailed stream produced nothing, the summary
      // falls back to answering from the context on its own.
      const results: PromiseSettledResult<void>[] = [];
      try {
        await streamVariant('detailed');
        results.push({status: 'fulfilled', value: undefined});
      } catch (e: any) {
        results.push({status: 'rejected', reason: e});
      }
      if (!controller.signal.aborted) {
        try {
          await streamVariant('summary', acc.detailed.trim() || undefined);
          results.push({status: 'fulfilled', value: undefined});
        } catch (e: any) {
          results.push({status: 'rejected', reason: e});
        }
      }
      const failures = results.filter(
        (r): r is PromiseRejectedResult => r.status === 'rejected' && r.reason?.name !== 'AbortError'
      );
      if (failures.length === results.length) {
        // Both variants failed: surface the error and drop the empty bubble.
        setError(failures[0].reason?.message || 'Something went wrong contacting the model.');
        setMessages(prev => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === 'assistant' && !last.summary && !last.detailed && !last.content) {
            next.pop();
          }
          return next;
        });
      } else {
        if (failures.length > 0) {
          setError(failures[0].reason?.message || 'One of the answer variants failed.');
        }
        // Persist the completed turn (best-effort; never blocks the UI).
        const assistant: ChatMessage = {
          role: 'assistant',
          content: acc.detailed || acc.summary,
          summary: acc.summary,
          detailed: acc.detailed,
          ...(payload ? {searches: payload.runs, searchTerms, searchConcepts: payload.concepts, searchIntersection: payload.intersection} : {})
        };
        const transcript: ChatMessage[] = [...base, {role: 'user', content}, assistant];
        if (conversationIdRef.current) {
          void saveConversation({
            conversationId: conversationIdRef.current,
            startedAt: startedAtRef.current || new Date().toISOString(),
            arrivalPath: arrivalPathRef.current,
            entryContext: entryContextRef.current,
            model,
            messages: transcript
          });
        }

        // EDA follow-up: when the question asks for numbers the matched
        // variables' profiles hold (per-category patient counts, numeric
        // summaries), the server selects the relevant profiles and ONE extra
        // answer is streamed, grounded in them; it shows as the turn's
        // "Summary statistics" tab. Best-effort: failures only show in the
        // progress panel and the main answer stands.
        const dropEmptyFollowup = () =>
          setMessages(prev => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last && last.role === 'assistant' && last.followup && !last.content) next.pop();
            return next;
          });
        if (payload && !controller.signal.aborted) {
          // The check shows as a progress step; the follow-up message stays
          // empty (no tab) unless the selection round finds numbers to add.
          setMessages(prev => [...prev, {role: 'assistant', content: '', followup: true}]);
          startStep('stats', 'Deciding whether the recorded summary statistics answer the question…');
          try {
            const fu = await fetchEdaFollowup(content, payload, selected, controller.signal);
            if (fu.needed && fu.context && !controller.signal.aborted) {
              const nVars = fu.variables?.length || 0;
              patchStep('stats', {
                detail: `Writing a follow-up from the statistics of ${nVars} variable${nVars === 1 ? '' : 's'}…`
              });
              let fuText = '';
              await streamChat({
                messages: historyForModel,
                cohortIds: selected,
                focus,
                searchResults: payload,
                edaContext: fu.context,
                signal: controller.signal,
                onChunk: delta => {
                  fuText += delta;
                  setMessages(prev => {
                    const next = [...prev];
                    const last = next[next.length - 1];
                    if (last && last.role === 'assistant' && last.followup) {
                      next[next.length - 1] = {...last, content: (last.content || '') + delta};
                    }
                    return next;
                  });
                }
              });
              if (fuText && conversationIdRef.current) {
                void saveConversation({
                  conversationId: conversationIdRef.current,
                  startedAt: startedAtRef.current || new Date().toISOString(),
                  arrivalPath: arrivalPathRef.current,
                  entryContext: entryContextRef.current,
                  model,
                  messages: [
                    ...base,
                    {role: 'user', content},
                    assistant,
                    {role: 'assistant', content: fuText, followup: true}
                  ]
                });
              }
              if (fuText) finishStep('stats', 'Numbers added: see the Summary statistics tab');
              else {
                finishStep('stats', 'No follow-up was produced');
                dropEmptyFollowup();
              }
            } else {
              // The statistics add nothing for this question: no follow-up.
              finishStep('stats', 'Not needed for this question');
              dropEmptyFollowup();
            }
          } catch {
            finishStep('stats', 'Could not check the summary statistics', 'failed');
            dropEmptyFollowup();
          }
        }
      }
      clearProgress();
      setIsStreaming(false);
      abortRef.current = null;
    },
    [input, isStreaming, messages, selected, focus, model]
  );

  return {
    messages,
    input,
    setInput,
    selected,
    toggleCohort,
    clearSelection,
    focus,
    setFocus,
    isStreaming,
    enabled,
    model,
    configLoaded,
    error,
    send,
    markSummaryViewed,
    stop,
    reset,
    loadConversation
  };
}
