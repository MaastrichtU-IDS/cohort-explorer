'use client';

// Layout A — "Copilot": a centered conversation with a right-hand context rail.
import React, {useMemo, useState} from 'react';
import Link from 'next/link';
import {ArrowLeft, X, Search} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {useCohortChat} from '@/components/ai/useCohortChat';
import {buildSuggestions, toBriefs} from '@/components/ai/chatClient';
import {Composer, DisabledNotice, ExperimentBadge, LoginNotice, MessageList} from '@/components/ai/ui';

export default function CopilotChat() {
  const {cohortsData, userEmail} = useCohorts();
  const chat = useCohortChat();
  const [query, setQuery] = useState('');

  const briefs = useMemo(() => toBriefs(cohortsData || {}), [cohortsData]);
  const filtered = useMemo(
    () => briefs.filter(b => b.id.toLowerCase().includes(query.toLowerCase())).slice(0, 60),
    [briefs, query]
  );
  const suggestions = useMemo(
    () => buildSuggestions(cohortsData || {}, chat.selected),
    [cohortsData, chat.selected]
  );

  const blocked = !chat.enabled || !userEmail;

  return (
    <main className="h-[calc(100vh-8rem)] bg-base-200 flex flex-col">
      <div className="border-b border-base-300 bg-base-100 px-6 py-3 flex items-center gap-3">
        <Link href="/ai" className="btn btn-ghost btn-sm gap-1">
          <ArrowLeft size={16} /> Hub
        </Link>
        <h1 className="font-bold text-lg">Copilot</h1>
        <ExperimentBadge />
        {chat.model && <span className="text-xs text-base-content/50 ml-auto">model: {chat.model}</span>}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Conversation column */}
        <section className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6">
            <div className="max-w-3xl mx-auto">
              {chat.configLoaded && !chat.enabled && <DisabledNotice />}
              {!userEmail && (
                <div className="mt-3">
                  <LoginNotice />
                </div>
              )}

              {chat.messages.length === 0 ? (
                <div className="text-center mt-10">
                  <div className="text-2xl font-bold mb-2">Ask about the cohorts</div>
                  <p className="text-base-content/60 mb-6">
                    Pin cohorts from the right to ground your questions, or just start typing.
                  </p>
                  <div className="grid sm:grid-cols-2 gap-3 max-w-2xl mx-auto">
                    {suggestions.map(s => (
                      <button
                        key={s}
                        disabled={blocked}
                        onClick={() => chat.send(s)}
                        className="text-left rounded-xl border border-base-300 bg-base-100 p-3 hover:border-primary hover:shadow-sm transition-all text-sm disabled:opacity-50"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <MessageList messages={chat.messages} streaming={chat.isStreaming} />
              )}

              {chat.error && (
                <div className="alert alert-error mt-4 text-sm">
                  <span>{chat.error}</span>
                </div>
              )}
            </div>
          </div>

          <div className="border-t border-base-300 bg-base-100 px-4 md:px-8 py-3">
            <div className="max-w-3xl mx-auto">
              {chat.selected.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {chat.selected.map(id => (
                    <span key={id} className="badge badge-primary badge-sm gap-1">
                      {id}
                      <button onClick={() => chat.toggleCohort(id)}>
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                </div>
              )}
              <Composer
                value={chat.input}
                onChange={chat.setInput}
                onSend={() => chat.send()}
                onStop={chat.stop}
                isStreaming={chat.isStreaming}
                disabled={blocked}
                placeholder={
                  chat.selected.length
                    ? `Ask about ${chat.selected.slice(0, 2).join(', ')}${chat.selected.length > 2 ? '…' : ''}`
                    : 'Ask about the cohorts…'
                }
              />
            </div>
          </div>
        </section>

        {/* Context rail */}
        <aside className="w-80 border-l border-base-300 bg-base-100 hidden lg:flex flex-col">
          <div className="p-3 border-b border-base-300">
            <div className="font-semibold text-sm mb-2">Context cohorts</div>
            <label className="input input-sm input-bordered flex items-center gap-2">
              <Search size={14} className="opacity-50" />
              <input
                className="grow"
                placeholder="Filter cohorts…"
                value={query}
                onChange={e => setQuery(e.target.value)}
              />
            </label>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {filtered.map(b => {
              const active = chat.selected.includes(b.id);
              return (
                <button
                  key={b.id}
                  onClick={() => chat.toggleCohort(b.id)}
                  className={`w-full text-left rounded-lg px-3 py-2 border transition-all ${
                    active ? 'border-primary bg-primary/10' : 'border-transparent hover:bg-base-200'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-sm truncate">{b.id}</span>
                    <span className="badge badge-ghost badge-sm">{b.variableCount}</span>
                  </div>
                  {b.studyType && <div className="text-xs text-base-content/50 truncate">{b.studyType}</div>}
                </button>
              );
            })}
          </div>
          {chat.selected.length > 0 && (
            <div className="p-3 border-t border-base-300">
              <button className="btn btn-ghost btn-sm w-full" onClick={chat.clearSelection}>
                Clear {chat.selected.length} selected
              </button>
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}
