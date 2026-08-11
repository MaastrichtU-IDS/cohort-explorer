'use client';

// Layout F — "Field Notebook": ask in sequence, build a document.
// Each question becomes a cell — question, streamed answer, and the cohort
// scope it was asked under. The result is a living research brief you can
// re-run cell by cell and export as Markdown.
import React, {useEffect, useMemo, useRef, useState} from 'react';
import Link from 'next/link';
import {ArrowLeft, BookOpen, Download, RefreshCw, Search, Trash2, X} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {
  ChatMessage,
  buildSuggestions,
  fetchChatConfig,
  streamChat,
  toBriefs
} from '@/components/ai/chatClient';
import {Composer, DisabledNotice, ExperimentBadge, LoginNotice, RichText, TypingDots} from '@/components/ai/ui';

interface Cell {
  id: number;
  question: string;
  answer: string;
  scope: string[];
  status: 'streaming' | 'done' | 'error';
  error?: string;
}

let nextCellId = 1;

export default function FieldNotebook() {
  const {cohortsData, userEmail} = useCohorts();
  const [title, setTitle] = useState('Untitled brief');
  const [cells, setCells] = useState<Cell[]>([]);
  const [input, setInput] = useState('');
  const [scope, setScope] = useState<string[]>([]);
  const [scopeQuery, setScopeQuery] = useState('');
  const [scopeOpen, setScopeOpen] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const [model, setModel] = useState('');
  const [configLoaded, setConfigLoaded] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchChatConfig().then(cfg => {
      setEnabled(cfg.enabled);
      setModel(cfg.model);
      setConfigLoaded(true);
    });
    return () => abortRef.current?.abort();
  }, []);

  const briefs = useMemo(() => toBriefs(cohortsData || {}), [cohortsData]);
  const scopeCandidates = useMemo(() => {
    const q = scopeQuery.trim().toLowerCase();
    const pool = briefs.filter(b => !scope.includes(b.id));
    return (q ? pool.filter(b => b.id.toLowerCase().includes(q)) : pool).slice(0, 10);
  }, [briefs, scope, scopeQuery]);

  const isStreaming = cells.some(c => c.status === 'streaming');
  const blocked = !enabled || !userEmail;

  const suggestions = useMemo(
    () => buildSuggestions(cohortsData || {}, scope).slice(0, 3),
    [cohortsData, scope]
  );

  // History for the model: all completed cells before `beforeId` (or all).
  const historyUpTo = (list: Cell[], beforeId?: number): ChatMessage[] => {
    const msgs: ChatMessage[] = [];
    for (const cell of list) {
      if (beforeId !== undefined && cell.id === beforeId) break;
      if (cell.status !== 'done' || !cell.answer) continue;
      msgs.push({role: 'user', content: cell.question});
      msgs.push({role: 'assistant', content: cell.answer});
    }
    return msgs;
  };

  // Latest cells, for history assembly inside async runs.
  const cellsRef = useRef<Cell[]>(cells);
  useEffect(() => {
    cellsRef.current = cells;
  }, [cells]);

  const runCell = async (cellId: number, question: string, cellScope: string[]) => {
    const controller = new AbortController();
    abortRef.current = controller;
    const patch = (p: Partial<Cell>) =>
      setCells(prev => prev.map(c => (c.id === cellId ? {...c, ...p} : c)));
    try {
      await streamChat({
        messages: [...historyUpTo(cellsRef.current, cellId), {role: 'user', content: question}],
        cohortIds: cellScope,
        signal: controller.signal,
        onChunk: delta =>
          setCells(prev => prev.map(c => (c.id === cellId ? {...c, answer: c.answer + delta} : c)))
      });
      patch({status: 'done'});
    } catch (e: any) {
      if (e?.name === 'AbortError') {
        patch({status: 'done'});
      } else {
        patch({status: 'error', error: e?.message || 'Something went wrong contacting the model.'});
      }
    } finally {
      abortRef.current = null;
    }
  };

  const ask = async () => {
    const question = input.trim();
    if (!question || isStreaming || blocked) return;
    setInput('');
    const cell: Cell = {id: nextCellId++, question, answer: '', scope: [...scope], status: 'streaming'};
    setCells(prev => [...prev, cell]);
    setTimeout(() => endRef.current?.scrollIntoView({behavior: 'smooth', block: 'end'}), 50);
    await runCell(cell.id, question, cell.scope);
  };

  const rerun = async (cell: Cell) => {
    if (isStreaming) return;
    setCells(prev => prev.map(c => (c.id === cell.id ? {...c, answer: '', status: 'streaming', error: undefined, scope: [...scope]} : c)));
    await runCell(cell.id, cell.question, scope);
  };

  const removeCell = (id: number) => setCells(prev => prev.filter(c => c.id !== id));

  const stop = () => abortRef.current?.abort();

  const exportMarkdown = () => {
    const lines: string[] = [`# ${title.trim() || 'Untitled brief'}`, ''];
    lines.push(`_Generated with the Cohort Explorer Field Notebook${model ? ` (model: ${model})` : ''}._`, '');
    for (const cell of cells) {
      if (cell.status !== 'done' || !cell.answer) continue;
      lines.push(`## ${cell.question}`, '');
      if (cell.scope.length) lines.push(`> Scope: ${cell.scope.join(', ')}`, '');
      lines.push(cell.answer.trim(), '');
    }
    const blob = new Blob([lines.join('\n')], {type: 'text/markdown'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(title.trim() || 'brief').toLowerCase().replace(/[^a-z0-9]+/g, '-')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const answeredCount = cells.filter(c => c.status === 'done' && c.answer).length;

  return (
    <main className="h-[calc(100vh-8rem)] bg-base-200 flex flex-col">
      {/* Top bar */}
      <div className="border-b border-base-300 bg-base-100 px-6 py-3 flex items-center gap-3">
        <Link href="/ai" className="btn btn-ghost btn-sm gap-1">
          <ArrowLeft size={16} /> Hub
        </Link>
        <h1 className="font-bold text-lg">Field Notebook</h1>
        <ExperimentBadge />
        <div className="ml-auto flex items-center gap-2">
          {model && <span className="text-xs text-base-content/50">model: {model}</span>}
          <button
            className="btn btn-outline btn-sm gap-1"
            onClick={exportMarkdown}
            disabled={answeredCount === 0}
          >
            <Download size={14} /> Export .md
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-8">
          {configLoaded && !enabled && <DisabledNotice />}
          {!userEmail && (
            <div className="mt-3 mb-3">
              <LoginNotice />
            </div>
          )}

          {/* The document */}
          <div className="bg-base-100 rounded-2xl border border-base-300 shadow-sm px-6 md:px-10 py-8 min-h-[60vh]">
            <input
              className="w-full text-3xl font-bold bg-transparent focus:outline-none placeholder:text-base-content/30"
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="Untitled brief"
            />

            {/* Scope control */}
            <div className="flex items-center gap-1.5 flex-wrap mt-3 pb-6 border-b border-base-200">
              <span className="text-xs text-base-content/50">Scope for new questions:</span>
              {scope.map(id => (
                <span key={id} className="badge badge-primary badge-sm gap-1">
                  {id}
                  <button onClick={() => setScope(prev => prev.filter(x => x !== id))}>
                    <X size={11} />
                  </button>
                </span>
              ))}
              {scopeOpen ? (
                <>
                  <label className="input input-xs input-bordered flex items-center gap-1 w-40">
                    <Search size={11} className="opacity-50" />
                    <input
                      autoFocus
                      className="grow"
                      placeholder="Find cohort…"
                      value={scopeQuery}
                      onChange={e => setScopeQuery(e.target.value)}
                      onBlur={() => setTimeout(() => setScopeOpen(false), 200)}
                    />
                  </label>
                  {scopeCandidates.map(b => (
                    <button
                      key={b.id}
                      onClick={() => {
                        setScope(prev => [...prev, b.id]);
                        setScopeQuery('');
                      }}
                      className="badge badge-outline badge-sm hover:badge-primary transition-all"
                    >
                      {b.id}
                    </button>
                  ))}
                </>
              ) : (
                <button className="btn btn-ghost btn-xs" onClick={() => setScopeOpen(true)}>
                  + add cohort
                </button>
              )}
            </div>

            {/* Cells */}
            {cells.length === 0 && (
              <div className="text-center py-14">
                <div className="inline-flex p-4 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-600 text-white shadow-lg mb-4">
                  <BookOpen size={28} />
                </div>
                <div className="text-xl font-bold mb-2">A brief, built question by question</div>
                <p className="text-base-content/60 max-w-md mx-auto text-sm">
                  Every question you ask becomes a cell: your question, the model&apos;s answer, and the
                  cohorts it was scoped to. Re-run cells as your scope evolves, then export the whole
                  thing as Markdown.
                </p>
              </div>
            )}

            <div className="divide-y divide-base-200">
              {cells.map((cell, idx) => (
                <div key={cell.id} className="py-6 group">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-baseline gap-3 min-w-0">
                      <span className="text-xs font-bold text-base-content/30 tabular-nums shrink-0">
                        {String(idx + 1).padStart(2, '0')}
                      </span>
                      <h2 className="font-semibold leading-snug">{cell.question}</h2>
                    </div>
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                      <button
                        className="btn btn-ghost btn-xs"
                        title="Re-run with the current scope"
                        disabled={isStreaming}
                        onClick={() => rerun(cell)}
                      >
                        <RefreshCw size={12} />
                      </button>
                      <button
                        className="btn btn-ghost btn-xs text-error"
                        title="Delete cell"
                        disabled={cell.status === 'streaming'}
                        onClick={() => removeCell(cell.id)}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                  {cell.scope.length > 0 && (
                    <div className="flex gap-1 mt-1.5 ml-8 flex-wrap">
                      {cell.scope.map(id => (
                        <span key={id} className="badge badge-ghost badge-xs">
                          {id}
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="mt-3 ml-8">
                    {cell.answer ? (
                      <RichText text={cell.answer} />
                    ) : cell.status === 'streaming' ? (
                      <TypingDots />
                    ) : null}
                    {cell.status === 'error' && (
                      <div className="alert alert-error mt-2 text-xs py-2">
                        <span>{cell.error}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Next-question suggestions */}
            {cells.length > 0 && !isStreaming && (
              <div className="flex gap-1.5 mt-2 flex-wrap">
                {suggestions.map(s => (
                  <button
                    key={s}
                    disabled={blocked}
                    onClick={() => {
                      setInput(s);
                    }}
                    className="px-2.5 py-1 rounded-full border border-base-300 text-xs text-base-content/60 hover:border-primary hover:text-primary transition-all disabled:opacity-50"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
            <div ref={endRef} />
          </div>
        </div>
      </div>

      {/* Composer */}
      <div className="border-t border-base-300 bg-base-100 px-4 py-3">
        <div className="max-w-3xl mx-auto">
          <Composer
            value={input}
            onChange={setInput}
            onSend={ask}
            onStop={stop}
            isStreaming={isStreaming}
            disabled={blocked}
            placeholder={
              scope.length
                ? `Ask about ${scope.slice(0, 2).join(', ')}${scope.length > 2 ? '…' : ''} — becomes cell ${cells.length + 1}`
                : `Ask the catalog — becomes cell ${cells.length + 1}`
            }
          />
        </div>
      </div>
    </main>
  );
}
