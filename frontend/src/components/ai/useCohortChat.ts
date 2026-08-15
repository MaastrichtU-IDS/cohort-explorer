// Shared conversation state + streaming logic reused by every AI layout.
import {useCallback, useEffect, useRef, useState} from 'react';
import {ChatMessage, fetchChatConfig, streamChat} from '@/components/ai/chatClient';

export interface SendOverrides {
  systemPrompt?: string;
  contextOverride?: string;
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
  }, [stop]);

  const send = useCallback(
    async (text?: string, overrides?: SendOverrides) => {
      const content = (text ?? input).trim();
      if (!content || isStreaming) return;
      setError(null);
      setInput('');

      // For follow-up turns the model sees the DETAILED variant of earlier
      // answers (that is the fuller record of what was said).
      const historyForModel: ChatMessage[] = [...messages, {role: 'user' as const, content}].map(m =>
        m.role === 'assistant' ? {role: m.role, content: m.detailed || m.content} : {role: m.role, content: m.content}
      );

      // Add the user turn plus an empty assistant turn holding both variants,
      // each streamed by its own request.
      setMessages([...messages, {role: 'user', content}, {role: 'assistant', content: '', summary: '', detailed: ''}]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      const streamVariant = (style: 'summary' | 'detailed') =>
        streamChat({
          messages: historyForModel,
          cohortIds: selected,
          focus,
          systemPrompt: overrides?.systemPrompt,
          contextOverride: overrides?.contextOverride,
          style,
          signal: controller.signal,
          onChunk: delta => {
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
      } else if (failures.length > 0) {
        setError(failures[0].reason?.message || 'One of the answer variants failed.');
      }
      setIsStreaming(false);
      abortRef.current = null;
    },
    [input, isStreaming, messages, selected, focus]
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
