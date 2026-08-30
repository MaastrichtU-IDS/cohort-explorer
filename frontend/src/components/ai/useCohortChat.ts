// Shared conversation state + streaming logic reused by every AI layout.
import {useCallback, useEffect, useRef, useState} from 'react';
import {
  ArrivalPath,
  ChatMessage,
  SearchPayload,
  fetchChatConfig,
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
  stop: () => void;
  reset: () => void;
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

  const reset = useCallback(() => {
    stop();
    setMessages([]);
    setError(null);
    // Next send starts a brand-new conversation record.
    conversationIdRef.current = null;
  }, [stop]);

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

      // Add the user turn plus an empty assistant turn holding both variants,
      // each streamed by its own request.
      setMessages([...base, {role: 'user', content}, {role: 'assistant', content: '', summary: '', detailed: ''}]);
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
        try {
          const plan = await planSearchWithRetry(content, selected, base);
          if (plan.needed && plan.searches.length > 0) {
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
            }
          }
        } catch (e: any) {
          // Planning is best-effort (the answer falls back to single-round
          // retrieval), but the failure is shown, not swallowed.
          const searchError = e?.message || 'catalog search failed';
          setMessages(prev => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last && last.role === 'assistant') next[next.length - 1] = {...last, searchError};
            return next;
          });
        }
        if (controller.signal.aborted) {
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
          if (next[next.length - 1]?.role === 'assistant') next[next.length - 1] = {role: 'assistant', content: ''};
          return next;
        });
        let clarifyText = '';
        try {
          await streamChat({
            messages: historyForModel,
            cohortIds: selected,
            focus,
            signal: controller.signal,
            searchResults: payload,
            clarifyInterpretations: interpretations,
            onChunk: delta => {
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
              messages: [...base, {role: 'user', content}, {role: 'assistant', content: clarifyText}]
            });
          }
        } catch (e: any) {
          if (e?.name !== 'AbortError') setError(e?.message || 'Something went wrong contacting the model.');
        }
        setIsStreaming(false);
        abortRef.current = null;
        return;
      }

      // Accumulate each variant locally too, so we can persist the final
      // transcript without reading React state back out.
      const acc: {summary: string; detailed: string} = {summary: '', detailed: ''};

      const streamVariant = (style: 'summary' | 'detailed') =>
        streamChat({
          messages: historyForModel,
          cohortIds: selected,
          focus,
          systemPrompt: overrides?.systemPrompt,
          contextOverride: overrides?.contextOverride,
          style,
          searchResults: payload,
          signal: controller.signal,
          onChunk: delta => {
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

      const results = await Promise.allSettled([streamVariant('summary'), streamVariant('detailed')]);
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
      }
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
    stop,
    reset
  };
}
