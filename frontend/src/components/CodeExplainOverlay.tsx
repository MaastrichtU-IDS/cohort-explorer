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
import {AlertTriangle, ArrowLeft, Send, X} from 'react-feather';
import {LocalModelNote, RichText, TypingDots} from '@/components/ai/ui';
import {
  ChatMessage,
  NodeScript,
  RoomOverview,
  fetchChatConfig,
  fetchConversation,
  fetchNodeScript,
  fetchRoomOverview,
  saveConversation,
  streamCodeExplanation,
  streamRoomExplanation
} from '@/components/ai/chatClient';
import {DcrGraph, KIND_STYLE, neighbourhood} from '@/components/DcrGraph';
import {DcrGraphOverlay, DcrParticipantLite} from '@/components/DcrGraphOverlay';
import {ExplainText} from '@/components/ai/ExplainText';
import type {RoomNode} from '@/components/ai/chatClient';

// The picker value for the whole-room view (the default when the overlay opens).
export const ROOM_VIEW = '__room__';

type Range = [number, number];

const OVERVIEW_QUESTION = 'Explain this whole script to me, step by step, and check whether any row-level data could leave the DCR.';
const ROOM_OVERVIEW_QUESTION =
  'Explain what happens in this data clean room overall, how the data flows between the nodes, and confirm whether any individual-level data reaches an output that analysts can see.';
const NODE_IO_QUESTION = (name: string) =>
  `What does the node "${name}" read, what does it write, and who can see its output?`;

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
                      {m.content ? <ExplainText text={m.content} room={null} /> : <TypingDots />}
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

// The whole-room view: dependency graph on the left, the room-level explanation
// and its follow-up chat on the right. The overview is requested automatically
// the first time the room is opened (backend: src/code_explain_room.py).
function RoomExplainer({
  dcrId,
  dcrTitle,
  room,
  loadError,
  resumeConversationId,
  session,
  onSession,
  onOpenNode,
  onOpenFullGraph
}: {
  dcrId: string;
  dcrTitle: string;
  // The room structure, loaded once by the overlay (null while loading).
  room: RoomOverview | null;
  loadError: string | null;
  // A stored room conversation to continue (from the AI history).
  resumeConversationId?: string | null;
  session?: Session;
  onSession: (s: Session) => void;
  onOpenNode: (name: string) => void;
  onOpenFullGraph: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>(session?.messages || []);
  const [streaming, setStreaming] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const convIdRef = useRef<string>(session?.convId || resumeConversationId || newConversationId());
  const startedAtRef = useRef<string>(session?.startedAt || new Date().toISOString());
  const transcriptRef = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  // A resumed conversation must not trigger a fresh overview while its transcript loads.
  const autoStarted = useRef(!!resumeConversationId && !session);

  useEffect(() => {
    onSession({convId: convIdRef.current, startedAt: startedAtRef.current, messages});
  }, [messages, onSession]);

  useEffect(() => {
    let cancelled = false;
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
  }, [dcrId, resumeConversationId]);

  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    const el = transcriptRef.current;
    if (el && follow.current) el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  const persist = useCallback(
    (msgs: ChatMessage[]) =>
      saveConversation({
        conversationId: convIdRef.current,
        startedAt: startedAtRef.current,
        arrivalPath: 'code_explanation',
        entryContext: {source: 'my_dcrs', page: '/dcrs', dcr_id: dcrId, dcr_title: dcrTitle, node_name: ROOM_VIEW},
        messages: msgs
      }),
    [dcrId, dcrTitle]
  );

  const ask = useCallback(
    async (question: string, opts?: {overview?: boolean}) => {
      const text = question.trim();
      if (!text || streaming) return;
      const userMsg: ChatMessage = {role: 'user', content: text};
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
        await streamRoomExplanation({
          dcrId,
          messages: history,
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
        setMessages(answer ? [...history, {role: 'assistant', content: answer}] : history);
      } finally {
        setStreaming(false);
      }
    },
    [messages, streaming, dcrId, persist]
  );

  // Start the overview on first open; a resumed session keeps its transcript.
  useEffect(() => {
    if (room && messages.length === 0 && !autoStarted.current) {
      autoStarted.current = true;
      ask(ROOM_OVERVIEW_QUESTION, {overview: true});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [room]);

  const exits = room?.exits || [];

  return (
    <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] divide-y md:divide-y-0 md:divide-x divide-base-300">
      {/* ---- graph ---- */}
      <section className="min-h-0 flex flex-col">
        {!room && !loadError && (
          <div className="flex justify-center py-12">
            <span className="loading loading-spinner loading-md" />
          </div>
        )}
        {loadError && (
          <div className="alert alert-error text-sm m-4 w-auto">
            <AlertTriangle size={16} /> <span>{loadError}</span>
          </div>
        )}
        {room && (
          <>
            <div className="min-h-0 flex-1 flex flex-col"><DcrGraph room={room} onSelect={name => onOpenNode(name)} mini /></div>
            <div className="px-3 py-2 text-[11px] text-base-content/60 border-t border-base-300 flex items-center gap-2 flex-wrap">
              <span>Map of the room: hover a shape for its name, click a compute node to open it.</span>
              <button className="btn btn-xs btn-outline ml-auto" onClick={onOpenFullGraph}>Open the full node graph</button>
            </div>
          </>
        )}
      </section>

      {/* ---- explanation + chat ---- */}
      <section className="min-h-0 flex flex-col">
        <div
          ref={transcriptRef}
          className="flex-1 overflow-y-auto px-5 py-4 space-y-4"
          onScroll={e => {
            const el = e.currentTarget;
            follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
          }}
        >
          {room && (
            <div className="text-xs text-base-content/60 rounded-lg bg-base-200 px-3 py-2">
              <b>{room.nodes.filter(n => n.kind === 'cohort_data').length}</b> cohort data node{room.nodes.filter(n => n.kind === 'cohort_data').length === 1 ? '' : 's'},{' '}
              <b>{room.nodes.filter(n => n.kind === 'python').length}</b> compute nodes,{' '}
              <b>{exits.filter(e => e.kind === 'airlock').length}</b> airlock{exits.filter(e => e.kind === 'airlock').length === 1 ? '' : 's'},{' '}
              <b>{exits.filter(e => e.kind === 'runnable').length}</b> node{exits.filter(e => e.kind === 'runnable').length === 1 ? '' : 's'} that participants can run directly.
              These exits are the only places where output becomes visible to analysts.
              {room.provisioning?.known && (
                <>
                  {' '}
                  <b>{room.provisioning.provisioned}</b> of <b>{room.provisioning.data_nodes}</b> data nodes currently hold data
                  {room.provisioning.provisioned < room.provisioning.data_nodes ? ' (nodes reading an empty data node cannot run yet).' : '.'}
                </>
              )}
            </div>
          )}
          {messages.map((m, i) =>
            m.role === 'user' ? (
              <div key={i} className="flex justify-end">
                <div className="rounded-2xl px-4 py-2 max-w-[85%] text-sm whitespace-pre-wrap bg-primary text-primary-content">{m.content}</div>
              </div>
            ) : (
              <div key={i} className="flex justify-start">
                <div className="rounded-2xl px-4 py-2 max-w-[92%] text-sm bg-base-200">
                  {m.content ? <ExplainText text={m.content} room={room} onOpenNode={onOpenNode} /> : <TypingDots />}
                </div>
              </div>
            )
          )}
          {messages.length > 0 && !streaming && (
            <p className="text-xs text-base-content/50">
              Answers come from a local language model and can be wrong: treat the safeguards section as a second pair of
              eyes, not as an approval. For the full script of a node, pick it in the selector above or click it in the graph.
            </p>
          )}
          {chatError && (
            <div className="alert alert-error text-sm">
              <AlertTriangle size={16} /> <span>{chatError}</span>
            </div>
          )}
        </div>
        <div className="border-t border-base-300 px-4 py-3">
          <div className="flex items-end gap-2">
            <textarea
              className="textarea textarea-bordered flex-1 text-sm leading-snug min-h-[2.75rem]"
              rows={2}
              placeholder="Ask a follow-up question about this room… (e.g. what exactly can analysts download?)"
              value={input}
              disabled={!room}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (input.trim()) ask(input);
                }
              }}
            />
            <button className="btn btn-primary" disabled={streaming || !room || !input.trim()} onClick={() => ask(input)} title="Send">
              <Send size={16} /> Send
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

// The "inputs and outputs" view of one compute node: on the left the facts
// from the room structure (what it reads, what it writes, who sees it) and the
// node's neighbourhood in the graph; on the right the model's I/O-focused
// explanation with follow-up chat. No code shown; that is the other mode.
function NodeIoExplainer({
  dcrId,
  dcrTitle,
  nodeName,
  room,
  loadError,
  session,
  onSession,
  onOpenNode
}: {
  dcrId: string;
  dcrTitle: string;
  nodeName: string;
  room: RoomOverview | null;
  loadError: string | null;
  session?: Session;
  onSession: (s: Session) => void;
  onOpenNode: (name: string) => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>(session?.messages || []);
  const [streaming, setStreaming] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const convIdRef = useRef<string>(session?.convId || newConversationId());
  const startedAtRef = useRef<string>(session?.startedAt || new Date().toISOString());
  const transcriptRef = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  const autoStarted = useRef(false);

  const node: RoomNode | undefined = useMemo(() => room?.nodes.find(n => n.name === nodeName), [room, nodeName]);
  const inputs = useMemo(() => (node ? node.dependencies.map(d => room!.nodes.find(n => n.name === d)).filter(Boolean) as RoomNode[] : []), [node, room]);
  const readers = useMemo(() => (room ? room.nodes.filter(n => n.dependencies.includes(nodeName)) : []), [room, nodeName]);
  const exit = useMemo(() => room?.exits.find(e => e.node === nodeName), [room, nodeName]);
  const airlocksFed = readers.filter(r => r.kind === 'airlock');
  const sub = useMemo(() => (room ? neighbourhood(room, nodeName) : null), [room, nodeName]);

  useEffect(() => {
    onSession({convId: convIdRef.current, startedAt: startedAtRef.current, messages});
  }, [messages, onSession]);
  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    const el = transcriptRef.current;
    if (el && follow.current) el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  const persist = useCallback(
    (msgs: ChatMessage[]) =>
      saveConversation({
        conversationId: convIdRef.current,
        startedAt: startedAtRef.current,
        arrivalPath: 'code_explanation',
        entryContext: {source: 'my_dcrs', page: '/dcrs', dcr_id: dcrId, dcr_title: dcrTitle, node_name: nodeName, view: 'io'},
        messages: msgs
      }),
    [dcrId, dcrTitle, nodeName]
  );

  const ask = useCallback(
    async (question: string, opts?: {overview?: boolean}) => {
      const text = question.trim();
      if (!text || streaming) return;
      const history = [...messages, {role: 'user', content: text} as ChatMessage];
      setMessages([...history, {role: 'assistant', content: ''}]);
      setInput('');
      setChatError(null);
      setStreaming(true);
      follow.current = true;
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      let answer = '';
      try {
        await streamRoomExplanation({
          dcrId,
          messages: history,
          overview: opts?.overview,
          focusNode: nodeName,
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
        setMessages(answer ? [...history, {role: 'assistant', content: answer}] : history);
      } finally {
        setStreaming(false);
      }
    },
    [messages, streaming, dcrId, nodeName, persist]
  );

  useEffect(() => {
    if (room && messages.length === 0 && !autoStarted.current) {
      autoStarted.current = true;
      ask(NODE_IO_QUESTION(nodeName), {overview: true});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [room]);

  const kindLabel = (n: RoomNode) => KIND_STYLE[n.kind].label.replace(/ \(.*\)$/, '');
  const Chip = ({n}: {n: RoomNode}) => (
    <li className="flex flex-wrap items-center gap-2 text-sm">
      <span className="inline-block w-3 h-3 rounded-sm border shrink-0" style={{background: KIND_STYLE[n.kind].fill, borderColor: KIND_STYLE[n.kind].stroke}} />
      {n.kind === 'python' ? (
        <button className="link link-hover font-mono text-xs" onClick={() => onOpenNode(n.name)} title="Open this node">
          {n.name}
        </button>
      ) : (
        <span className="font-mono text-xs">{n.name}</span>
      )}
      <span className="text-xs text-base-content/60">{kindLabel(n)}</span>
      {n.provisioned === false && <span className="badge badge-ghost badge-xs">no data yet</span>}
      {n.kind === 'airlock' && n.quota_bytes ? <span className="text-xs text-base-content/60">quota {Math.round(n.quota_bytes / 1e6)} MB</span> : null}
    </li>
  );

  return (
    <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-base-300">
      {/* ---- facts + neighbourhood ---- */}
      <section className="min-h-0 flex flex-col">
        {!room && !loadError && (
          <div className="flex justify-center py-12">
            <span className="loading loading-spinner loading-md" />
          </div>
        )}
        {loadError && (
          <div className="alert alert-error text-sm m-4 w-auto">
            <AlertTriangle size={16} /> <span>{loadError}</span>
          </div>
        )}
        {room && node && (
          <>
            <div className="px-5 py-3 space-y-3 overflow-y-auto border-b border-base-300 max-h-[55%]">
              <div>
                <div className="text-xs font-semibold uppercase text-base-content/60 mb-1">Reads ({inputs.length})</div>
                {inputs.length === 0 ? <p className="text-sm text-base-content/50 italic">No inputs.</p> : <ul className="space-y-1">{inputs.map(n => <Chip key={n.name} n={n} />)}</ul>}
                {inputs.some(n => n.provisioned === false) && (
                  <p className="text-xs text-base-content/60 mt-1">An input has no data yet; the node can run once it is provisioned.</p>
                )}
              </div>
              <div>
                <div className="text-xs font-semibold uppercase text-base-content/60 mb-1">Writes</div>
                {node.outputs && node.outputs.length > 0 && (
                  <ul className="flex flex-wrap gap-1 mb-1">
                    {node.outputs.map(o => (
                      <li key={o} className="badge badge-outline font-mono text-[11px]">{o}</li>
                    ))}
                  </ul>
                )}
                {node.output_kinds && node.output_kinds.length > 0 ? (
                  <p className="text-xs text-base-content/60">
                    Writes {node.output_kinds.join(', ')}{node.outputs && node.outputs.length > 0 ? ' (file names above; some may be computed at run time)' : ' (file names are computed at run time)'}.
                  </p>
                ) : (
                  !(node.outputs && node.outputs.length > 0) && <p className="text-sm text-base-content/50 italic">No file output found in the script.</p>
                )}
                {node.external_writes && (
                  <p className="text-xs text-base-content/60 mt-1">
                    Also uses the package {(node.external_packages || []).join(', ')}, which writes its own files into the output folder; they are not listed here, so check the node&apos;s output folder for the full set.
                  </p>
                )}
                {node.prints && <p className="text-xs text-base-content/60 mt-1">Also prints or logs text{node.logs.length ? ` (logs shown ${node.logs.join(' and ').replace(/_/g, ' ')})` : ''}.</p>}
              </div>
              <div>
                <div className="text-xs font-semibold uppercase text-base-content/60 mb-1">Who sees the output</div>
                <ul className="text-sm space-y-1">
                  <li>
                    {exit && exit.kind === 'runnable'
                      ? <><b>{exit.runners}</b> participant{exit.runners === 1 ? '' : 's'} can run this node directly and download its output files.</>
                      : <>No participant can run it directly: it only runs as a dependency of other nodes.</>}
                  </li>
                  {airlocksFed.length > 0 && <li>Its output is shown through the airlock{airlocksFed.length === 1 ? '' : 's'} {airlocksFed.map(a => <code key={a.name} className="text-xs">{a.name}</code>)}.</li>}
                  {readers.filter(r => r.kind !== 'airlock').length > 0 && (
                    <li>Read by: {readers.filter(r => r.kind !== 'airlock').map(r => <button key={r.name} className="link link-hover font-mono text-xs mr-2" onClick={() => onOpenNode(r.name)}>{r.name}</button>)}</li>
                  )}
                  {readers.length === 0 && !(exit && exit.kind === 'runnable') && <li className="text-base-content/50 italic">Nothing reads its output.</li>}
                </ul>
              </div>
            </div>
            <div className="flex-1 min-h-0">
              {sub && <DcrGraph room={sub} selected={nodeName} compact onSelect={name => name !== nodeName && onOpenNode(name)} />}
            </div>
          </>
        )}
        {room && !node && <div className="alert alert-warning text-sm m-4 w-auto">This node is not in the room structure.</div>}
      </section>

      {/* ---- explanation + chat ---- */}
      <section className="min-h-0 flex flex-col">
        <div
          ref={transcriptRef}
          className="flex-1 overflow-y-auto px-5 py-4 space-y-4"
          onScroll={e => {
            const el = e.currentTarget;
            follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
          }}
        >
          {messages.map((m, i) =>
            m.role === 'user' ? (
              <div key={i} className="flex justify-end">
                <div className="rounded-2xl px-4 py-2 max-w-[85%] text-sm whitespace-pre-wrap bg-primary text-primary-content">{m.content}</div>
              </div>
            ) : (
              <div key={i} className="flex justify-start">
                <div className="rounded-2xl px-4 py-2 max-w-[92%] text-sm bg-base-200">
                  {m.content ? <ExplainText text={m.content} room={room} onOpenNode={onOpenNode} /> : <TypingDots />}
                </div>
              </div>
            )
          )}
          {messages.length > 0 && !streaming && (
            <p className="text-xs text-base-content/50">
              Answers come from a local language model and can be wrong. Switch to &quot;Code &amp; explanation&quot; above for the step-by-step view of the script.
            </p>
          )}
          {chatError && (
            <div className="alert alert-error text-sm">
              <AlertTriangle size={16} /> <span>{chatError}</span>
            </div>
          )}
        </div>
        <div className="border-t border-base-300 px-4 py-3">
          <div className="flex items-end gap-2">
            <textarea
              className="textarea textarea-bordered flex-1 text-sm leading-snug min-h-[2.75rem]"
              rows={2}
              placeholder={`Ask about the inputs or outputs of ${nodeName}…`}
              value={input}
              disabled={!room}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (input.trim()) ask(input);
                }
              }}
            />
            <button className="btn btn-primary" disabled={streaming || !room || !input.trim()} onClick={() => ask(input)} title="Send">
              <Send size={16} /> Send
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
  participants,
  onClose
}: {
  dcrId: string;
  dcrTitle: string;
  // The DCR's compute nodes that have code to explain.
  nodes: string[];
  initialNode?: string | null;
  // A stored conversation to continue; it belongs to initialNode.
  resumeConversationId?: string | null;
  // The room's participants with their per-node permissions, for the full node graph.
  participants?: DcrParticipantLite[];
  onClose: () => void;
}) {
  const [fullGraph, setFullGraph] = useState(false);
  // Opens on the whole-room view unless a node (or a stored node conversation) was asked for.
  const [node, setNode] = useState<string>(initialNode && nodes.includes(initialNode) ? initialNode : ROOM_VIEW);
  // For a node: 'io' (inputs and outputs, the default) or 'code' (the script, explained line by line).
  const [mode, setMode] = useState<'io' | 'code'>(resumeConversationId && initialNode && nodes.includes(initialNode) ? 'code' : 'io');
  const [model, setModel] = useState('');
  const [room, setRoom] = useState<RoomOverview | null>(null);
  const [roomError, setRoomError] = useState<string | null>(null);
  const sessions = useRef<Record<string, Session>>({});
  const sessionKey = node === ROOM_VIEW ? ROOM_VIEW : `${node}::${mode}`;
  const saveSession = useCallback(
    (s: Session) => {
      sessions.current[sessionKey] = s;
    },
    [sessionKey]
  );
  const openNode = useCallback(
    (name: string) => {
      if (!nodes.includes(name)) return;
      setNode(name);
      setMode('io');
    },
    [nodes]
  );

  useEffect(() => {
    fetchChatConfig().then(c => setModel(c.model || ''));
    let cancelled = false;
    fetchRoomOverview(dcrId)
      .then(r => {
        if (!cancelled) setRoom(r);
      })
      .catch(err => {
        if (!cancelled) setRoomError(err?.message || 'Could not load the room structure.');
      });
    return () => {
      cancelled = true;
    };
  }, [dcrId]);

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
              <h3 className="font-bold">AI - Explain this data clean room</h3>
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
          <div className="flex flex-wrap items-end justify-center gap-3">
            {node !== ROOM_VIEW && (
              <button className="btn btn-sm btn-primary gap-1" onClick={() => setNode(ROOM_VIEW)} title="Back to the whole room: overview, graph and safeguards">
                <ArrowLeft size={14} /> Room graph
              </button>
            )}
          <label className="block w-auto min-w-[16rem] max-w-full">
            <span className="block text-center text-xs font-semibold text-base-content/70 mb-1">{node === ROOM_VIEW ? 'View' : 'Compute node'}</span>
            {/* Same blue as the concept-code tags on the explore page; custom, bold chevron. */}
            <select
              className="w-full appearance-none rounded-xl py-3 pl-4 pr-12 font-mono text-base font-semibold cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-300"
              style={{
                backgroundColor: '#dbeafe',
                color: '#1e3a8a',
                border: '1px solid #bfdbfe',
                backgroundImage:
                  "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%231e3a8a' stroke-width='3.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 12 15 18 9'/></svg>\")",
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'right 0.75rem center',
                backgroundSize: '1.5rem'
              }}
              value={node}
              onChange={e => (e.target.value === ROOM_VIEW ? setNode(ROOM_VIEW) : openNode(e.target.value))}
              aria-label="Compute node to explain"
            >
              <option value={ROOM_VIEW}>Whole room: overview, data flow and safeguards</option>
              {nodes.map(n => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
            {node !== ROOM_VIEW && (
              <div role="tablist" className="tabs tabs-boxed tabs-sm">
                <button role="tab" className={`tab ${mode === 'io' ? 'tab-active' : ''}`} onClick={() => setMode('io')} title="What this node reads and writes, and who sees it">
                  Inputs &amp; outputs
                </button>
                <button role="tab" className={`tab ${mode === 'code' ? 'tab-active' : ''}`} onClick={() => setMode('code')} title="The script, explained step by step">
                  Code &amp; explanation
                </button>
              </div>
            )}
          </div>
        </div>
        {node === ROOM_VIEW && (
          <RoomExplainer
            key={ROOM_VIEW}
            dcrId={dcrId}
            dcrTitle={dcrTitle}
            room={room}
            loadError={roomError}
            resumeConversationId={initialNode && !nodes.includes(initialNode) ? resumeConversationId : null}
            session={sessions.current[ROOM_VIEW]}
            onSession={saveSession}
            onOpenNode={openNode}
            onOpenFullGraph={() => setFullGraph(true)}
          />
        )}
        {fullGraph && (
          <DcrGraphOverlay dcrId={dcrId} dcrTitle={dcrTitle} participants={participants || []} onClose={() => setFullGraph(false)} />
        )}
        {node && node !== ROOM_VIEW && mode === 'io' && (
          <NodeIoExplainer
            key={`${node}::io`}
            dcrId={dcrId}
            dcrTitle={dcrTitle}
            nodeName={node}
            room={room}
            loadError={roomError}
            session={sessions.current[`${node}::io`]}
            onSession={saveSession}
            onOpenNode={openNode}
          />
        )}
        {node && node !== ROOM_VIEW && mode === 'code' && (
          <NodeExplainer
            key={`${node}::code`}
            dcrId={dcrId}
            dcrTitle={dcrTitle}
            nodeName={node}
            resumeConversationId={node === initialNode ? resumeConversationId : null}
            session={sessions.current[`${node}::code`]}
            onSession={saveSession}
          />
        )}
      </div>
    </div>
  );
}
