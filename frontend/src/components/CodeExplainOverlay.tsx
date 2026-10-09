'use client';

// Explain-the-code overlay for a compute node of a DCR (My DCRs page).
//
// Left: the node's Python script, one clickable row per line (click selects a
// line, shift-click extends the selection). Right: the conversation with the
// local model, which explains the script and checks it for row-level data
// leaving the room (backend: src/code_explain.py). The conversation is saved to
// the AI history as a 'code_explanation' session that records where it came
// from (this DCR and node), and can be reopened from there.
import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {AlertTriangle, Send, X} from 'react-feather';
import {LocalModelNote, RichText, TypingDots} from '@/components/ai/ui';
import {
  ChatMessage,
  NodeScript,
  fetchChatConfig,
  fetchConversation,
  fetchNodeScript,
  saveConversation,
  streamCodeExplanation
} from '@/components/ai/chatClient';

type Range = [number, number];

const OVERVIEW_QUESTION = 'Explain this whole script to me, step by step, and check whether any row-level data could leave the DCR.';
const LEAK_QUESTION = 'Could any individual-level rows leave the DCR through this script? Check every output.';

function newConversationId(): string {
  try {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  } catch {
    /* fall through */
  }
  return `conv-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

const rangeLabel = (r: Range) => (r[0] === r[1] ? `line ${r[0]}` : `lines ${r[0]}–${r[1]}`);

const CodeRow = React.memo(function CodeRow({
  n,
  text,
  selected,
  hint,
  onClick
}: {
  n: number;
  text: string;
  selected: boolean;
  hint?: {text: string; cls: string};
  onClick: (n: number, shift: boolean) => void;
}) {
  const isComment = text.trim().startsWith('#');
  return (
    <div
      data-line={n}
      onMouseDown={e => e.shiftKey && e.preventDefault()}
      onClick={e => onClick(n, e.shiftKey)}
      className={`flex cursor-pointer min-w-max hover:bg-base-300/60 ${selected ? 'bg-primary/20' : ''}`}
    >
      <span className={`w-5 shrink-0 text-center select-none ${hint?.cls || ''}`} title={hint?.text}>
        {hint ? '●' : ''}
      </span>
      <span className="w-10 shrink-0 pr-3 text-right text-base-content/40 select-none tabular-nums">{n}</span>
      <span className={`whitespace-pre pr-4 ${isComment ? 'text-base-content/50 italic' : ''}`}>{text || ' '}</span>
    </div>
  );
});

// The two panels for one node. Mounted with key=<node>, so switching nodes
// starts a fresh conversation; the parent's cache hands a node its earlier
// conversation back when the reader returns to it.
type Session = {convId: string; startedAt: string; messages: ChatMessage[]};

function NodeExplainer({
  dcrId,
  dcrTitle,
  nodeName,
  resumeConversationId,
  session,
  onSession
}: {
  dcrId: string;
  dcrTitle: string;
  nodeName: string;
  resumeConversationId?: string | null;
  session?: Session;
  onSession: (s: Session) => void;
}) {
  const [script, setScript] = useState<NodeScript | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>(session?.messages || []);
  const [streaming, setStreaming] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [selection, setSelection] = useState<Range | null>(null);
  const [input, setInput] = useState('');
  const anchor = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const convIdRef = useRef<string>(session?.convId || resumeConversationId || newConversationId());
  const startedAtRef = useRef<string>(session?.startedAt || new Date().toISOString());
  const transcriptRef = useRef<HTMLDivElement>(null);
  const follow = useRef(true);

  const lines = useMemo(() => (script ? script.script.split('\n') : []), [script]);
  const hintByLine = useMemo(() => {
    // Strongest marker wins when a line has several: external call > patient-level name > file write.
    const m = new Map<number, {text: string; cls: string}>();
    for (const h of script?.hints || []) {
      if (h.kind === 'output') m.set(h.line, {text: 'This line writes a file (an output of the node)', cls: 'text-warning'});
    }
    for (const sn of script?.external?.sensitive_names || []) {
      m.set(sn.line, {text: `Names a file that sounds like patient-level data: ${sn.name}. Check the node's output file to confirm what it contains.`, cls: 'text-secondary'});
    }
    for (const n of script?.external?.external_call_lines || []) {
      m.set(n, {text: 'Calls external code whose source is not shown: it may write files you cannot see', cls: 'text-error'});
    }
    return m;
  }, [script]);

  // Remember this node's conversation for when the reader comes back to it.
  useEffect(() => {
    onSession({convId: convIdRef.current, startedAt: startedAtRef.current, messages});
  }, [messages, onSession]);

  // Load the script and, when reopening a stored session, its transcript.
  useEffect(() => {
    let cancelled = false;
    fetchNodeScript(dcrId, nodeName)
      .then(s => {
        if (!cancelled) setScript(s);
      })
      .catch(err => {
        if (!cancelled) setLoadError(err?.message || 'Could not load the script.');
      });
    if (resumeConversationId && !session) {
      fetchConversation(resumeConversationId)
        .then(c => {
          if (cancelled) return;
          setMessages(c.messages || []);
          if (c.started_at) startedAtRef.current = c.started_at;
        })
        .catch(() => {
          /* start a fresh conversation under the same id */
        });
    }
    return () => {
      cancelled = true;
    };
  }, [dcrId, nodeName, resumeConversationId]);

  // Stop a running answer when the node is switched or the overlay closed.
  useEffect(() => () => abortRef.current?.abort(), []);

  // Keep the newest text in view while it streams, unless the reader scrolled up.
  useEffect(() => {
    const el = transcriptRef.current;
    if (el && follow.current) el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  const onRowClick = useCallback((n: number, shift: boolean) => {
    if (shift && anchor.current != null) {
      const a = anchor.current;
      setSelection([Math.min(a, n), Math.max(a, n)]);
    } else {
      anchor.current = n;
      setSelection(prev => (prev && prev[0] === n && prev[1] === n ? null : [n, n]));
    }
  }, []);

  const persist = useCallback(
    (msgs: ChatMessage[]) =>
      saveConversation({
        conversationId: convIdRef.current,
        startedAt: startedAtRef.current,
        arrivalPath: 'code_explanation',
        entryContext: {source: 'my_dcrs', page: '/dcrs', dcr_id: dcrId, dcr_title: dcrTitle, node_name: nodeName},
        messages: msgs
      }),
    [dcrId, dcrTitle, nodeName]
  );

  const ask = useCallback(
    async (question: string, opts?: {overview?: boolean}) => {
      const text = question.trim();
      if (!text || streaming || !script) return;
      const sel: Range | undefined = selection ? [selection[0], selection[1]] : undefined;
      const userMsg: ChatMessage = {role: 'user', content: text, ...(sel ? {codeLines: sel} : {})};
      const history = [...messages, userMsg];
      setMessages([...history, {role: 'assistant', content: ''}]);
      setInput('');
      setChatError(null);
      setStreaming(true);
      follow.current = true;
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      let answer = '';
      try {
        await streamCodeExplanation({
          dcrId,
          nodeName,
          messages: history,
          selectedLines: sel || null,
          overview: opts?.overview,
          signal: ctrl.signal,
          onChunk: d => {
            answer += d;
            setMessages([...history, {role: 'assistant', content: answer}]);
          }
        });
        const done: ChatMessage[] = [...history, {role: 'assistant', content: answer.trim()}];
        setMessages(done);
        persist(done);
      } catch (err: any) {
        if (err?.name === 'AbortError') return;
        setChatError(err?.message || 'The model could not be reached.');
        // Keep the question; drop the empty answer.
        setMessages(answer ? [...history, {role: 'assistant', content: answer}] : history);
      } finally {
        setStreaming(false);
      }
    },
    [messages, streaming, script, selection, dcrId, nodeName, persist]
  );

  const send = () => {
    if (input.trim()) ask(input);
    else if (selection) ask(`Explain ${rangeLabel(selection)}.`);
  };

  return (
        <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-base-300">
          {/* ---- code ---- */}
          <section className="min-h-0 flex flex-col">
            <div className="px-4 py-2 text-xs text-base-content/60 border-b border-base-300 flex flex-wrap gap-x-4">
              <span className="basis-full text-base-content/80">
                <b>Selecting code:</b> click a line to select it. To select a block, click its{' '}
                <b>first line</b>, then hold <kbd className="kbd kbd-xs">Shift</kbd> and click its <b>last line</b>.
                Click the selected line again to unselect.
              </span>
              <span>
                <span className="text-warning">●</span> writes a file
              </span>
              <span>
                <span className="text-error">●</span> external code
              </span>
              <span>
                <span className="text-secondary">●</span> patient-level file name
              </span>
              {script && script.dependencies.length > 0 && (
                <span className="truncate">Reads: {script.dependencies.join(', ')}</span>
              )}
            </div>
            {script?.external_notice && (
              <div className="px-4 py-3 text-sm bg-error/10 border-b border-error/30 max-h-48 overflow-y-auto">
                <RichText text={script.external_notice} />
              </div>
            )}
            <div className="flex-1 overflow-auto bg-base-200/60 font-mono text-[12.5px] leading-6 py-2">
              {!script && !loadError && (
                <div className="flex justify-center py-12">
                  <span className="loading loading-spinner loading-md" />
                </div>
              )}
              {loadError && (
                <div className="alert alert-error text-sm m-4 w-auto font-sans">
                  <AlertTriangle size={16} /> <span>{loadError}</span>
                </div>
              )}
              {lines.map((text, i) => (
                <CodeRow
                  key={i}
                  n={i + 1}
                  text={text}
                  selected={!!selection && i + 1 >= selection[0] && i + 1 <= selection[1]}
                  hint={hintByLine.get(i + 1)}
                  onClick={onRowClick}
                />
              ))}
            </div>
          </section>

          {/* ---- explanations ---- */}
          <section className="min-h-0 flex flex-col">
            <div
              ref={transcriptRef}
              className="flex-1 overflow-y-auto px-5 py-4 space-y-4"
              onScroll={e => {
                const el = e.currentTarget;
                follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
              }}
            >
              {messages.length === 0 && (
                <div className="text-sm text-base-content/70 space-y-3">
                  <p>
                    I can explain what this script does to the data, line by line or as a whole, and check whether any
                    row-level data could leave the DCR. Select one line, or a block (click its first line, then
                    Shift-click its last line), on the left to ask about it, or start here:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <button
                      className="btn btn-sm btn-outline"
                      disabled={!script || streaming}
                      onClick={() => ask(OVERVIEW_QUESTION, {overview: true})}
                    >
                      Explain the whole script
                    </button>
                    <button className="btn btn-sm btn-outline" disabled={!script || streaming} onClick={() => ask(LEAK_QUESTION)}>
                      Check for data leaks
                    </button>
                  </div>
                  <p className="text-xs text-base-content/50">
                    Answers come from a local language model and can be wrong: treat the privacy check as a second
                    pair of eyes, not as an approval.
                  </p>
                </div>
              )}
              {messages.map((m, i) =>
                m.role === 'user' ? (
                  <div key={i} className="flex justify-end">
                    <div className="rounded-2xl px-4 py-2 max-w-[85%] text-sm whitespace-pre-wrap bg-primary text-primary-content">
                      {m.codeLines && (
                        <div className="text-[10px] uppercase tracking-wide opacity-70 mb-0.5">
                          {rangeLabel(m.codeLines)}
                        </div>
                      )}
                      {m.content}
                    </div>
                  </div>
                ) : (
                  <div key={i} className="flex justify-start">
                    <div className="rounded-2xl px-4 py-2 max-w-[92%] text-sm bg-base-200">
                      {m.content ? <RichText text={m.content} /> : <TypingDots />}
                    </div>
                  </div>
                )
              )}
              {chatError && (
                <div className="alert alert-error text-sm">
                  <AlertTriangle size={16} /> <span>{chatError}</span>
                </div>
              )}
            </div>

            {/* ---- dialogue box ---- */}
            <div className="border-t border-base-300 px-4 py-3 space-y-2">
              {selection && (
                <div className="flex items-center gap-2 text-xs">
                  <span className="badge badge-primary badge-outline">{rangeLabel(selection)} selected</span>
                  <button className="link text-base-content/60" onClick={() => setSelection(null)}>
                    clear
                  </button>
                </div>
              )}
              <div className="flex items-end gap-2">
                <textarea
                  className="textarea textarea-bordered flex-1 text-sm leading-snug min-h-[2.75rem]"
                  rows={2}
                  placeholder={selection ? `Ask about ${rangeLabel(selection)}… (empty = explain it)` : 'Ask a question about this script…'}
                  value={input}
                  disabled={!script}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      send();
                    }
                  }}
                />
                <button
                  className="btn btn-primary"
                  disabled={streaming || !script || (!input.trim() && !selection)}
                  onClick={send}
                  title={input.trim() ? 'Send' : selection ? 'Explain the selected lines' : 'Type a question'}
                >
                  <Send size={16} />
                  {!input.trim() && selection ? 'Explain' : 'Send'}
                </button>
              </div>
            </div>
          </section>
        </div>
  );
}

export function CodeExplainOverlay({
  dcrId,
  dcrTitle,
  nodes,
  initialNode,
  resumeConversationId,
  onClose
}: {
  dcrId: string;
  dcrTitle: string;
  // The DCR's compute nodes that have code to explain.
  nodes: string[];
  initialNode?: string | null;
  // A stored conversation to continue; it belongs to initialNode.
  resumeConversationId?: string | null;
  onClose: () => void;
}) {
  const [node, setNode] = useState<string>(initialNode && nodes.includes(initialNode) ? initialNode : nodes[0] || '');
  const [model, setModel] = useState('');
  const sessions = useRef<Record<string, Session>>({});
  const saveSession = useCallback(
    (s: Session) => {
      sessions.current[node] = s;
    },
    [node]
  );

  useEffect(() => {
    fetchChatConfig().then(c => setModel(c.model || ''));
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-3 md:p-6" onClick={onClose}>
      <div
        className="bg-base-100 rounded-2xl shadow-2xl w-full h-full max-w-[1500px] flex flex-col overflow-hidden"
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-label="Explain the code of the compute nodes"
      >
        <div className="px-5 py-3 border-b border-base-300 space-y-2">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <h3 className="font-bold">Explain the code</h3>
              <div className="text-xs text-base-content/60 truncate">{dcrTitle || 'Untitled DCR'}</div>
              <div className="text-[11px] text-base-content/50 truncate">
                Model: <span className="font-mono">{model || '…'}</span> ·{' '}
                <LocalModelNote className="text-[11px] text-base-content/50" />
              </div>
            </div>
            <button className="btn btn-sm btn-ghost btn-circle" onClick={onClose} aria-label="Close">
              <X size={18} />
            </button>
          </div>
          <label className="block">
            <span className="text-xs font-semibold text-base-content/70">Compute node</span>
            <select
              className="select select-bordered select-lg w-full font-mono text-base"
              value={node}
              onChange={e => setNode(e.target.value)}
              aria-label="Compute node to explain"
            >
              {nodes.map(n => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>
        {node && (
          <NodeExplainer
            key={node}
            dcrId={dcrId}
            dcrTitle={dcrTitle}
            nodeName={node}
            resumeConversationId={node === initialNode ? resumeConversationId : null}
            session={sessions.current[node]}
            onSession={saveSession}
          />
        )}
      </div>
    </div>
  );
}
