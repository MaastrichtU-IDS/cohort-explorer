'use client';

// Layout A — "Copilot": a centered conversation with a right-hand context rail.
// The rail has two tabs: pin cohorts as context, or use the Guide — the Prompt
// Studio's intent/topic building blocks, embedded — to assemble a question
// directly into the composer.
import React, {useEffect, useMemo, useState} from 'react';
import Link from 'next/link';
import {ArrowLeft, X, Search, Layers, Compass, Edit3} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {useCohortChat} from '@/components/ai/useCohortChat';
import {buildSuggestions, toBriefs} from '@/components/ai/chatClient';
import {intents, joinCohortLabel, topicBank} from '@/components/ai/promptKit';
import {Composer, DisabledNotice, ExperimentBadge, LoginNotice, MessageList} from '@/components/ai/ui';

// ---- Guide tab: compact prompt builder -------------------------------------

function GuidePanel({
  selected,
  onPrompt
}: {
  selected: string[];
  onPrompt: (text: string) => void;
}) {
  const [intentId, setIntentId] = useState<string | null>(null);
  const [topic, setTopic] = useState('');
  const [customTopic, setCustomTopic] = useState('');

  const activeIntent = intents.find(i => i.id === intentId) || null;
  const effectiveTopic = customTopic.trim() || topic;

  const assembled = useMemo(
    () => (activeIntent ? activeIntent.template(joinCohortLabel(selected), effectiveTopic) : ''),
    [activeIntent, selected, effectiveTopic]
  );

  // Push each newly assembled question into the composer, where it stays editable.
  useEffect(() => {
    if (assembled) onPrompt(assembled);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assembled]);

  return (
    <div className="p-3 space-y-4">
      <div>
        <div className="text-xs font-semibold text-base-content/60 uppercase tracking-wide mb-2">
          What do you want to do?
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          {intents.map(intent => {
            const Icon = intent.icon;
            const active = intentId === intent.id;
            return (
              <button
                key={intent.id}
                onClick={() => setIntentId(active ? null : intent.id)}
                className={`rounded-lg border p-2 text-left transition-all ${
                  active
                    ? 'border-primary bg-primary/10'
                    : 'border-base-300 hover:border-primary/40 hover:bg-base-200'
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <Icon size={13} className={active ? 'text-primary' : 'opacity-60'} />
                  <span className="text-xs font-semibold">{intent.label}</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <div className="text-xs font-semibold text-base-content/60 uppercase tracking-wide mb-2">
          Focus on a topic <span className="normal-case font-normal">(optional)</span>
        </div>
        <div className="flex flex-wrap gap-1">
          {topicBank.slice(0, 9).map(t => (
            <button
              key={t}
              onClick={() => {
                setTopic(topic === t ? '' : t);
                setCustomTopic('');
              }}
              className={`px-2 py-0.5 rounded-full text-xs border transition-all ${
                topic === t
                  ? 'border-primary bg-primary/10 text-primary font-medium'
                  : 'border-base-300 hover:border-primary/40 text-base-content/70'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        <input
          className="input input-xs input-bordered w-full mt-2"
          placeholder="…or type your own topic"
          value={customTopic}
          onChange={e => {
            setCustomTopic(e.target.value);
            if (e.target.value.trim()) setTopic('');
          }}
        />
      </div>

      {assembled ? (
        <div className="rounded-lg border border-primary/30 bg-primary/5 p-2.5">
          <div className="text-[11px] font-semibold text-primary mb-1 flex items-center gap-1">
            <Edit3 size={11} /> In your composer — edit or send
          </div>
          <p className="text-xs text-base-content/80 leading-relaxed">{assembled}</p>
        </div>
      ) : (
        <p className="text-xs text-base-content/50 leading-relaxed">
          Pick an intent above and the question appears in the composer, tailored to your pinned
          cohorts. You can always edit it before sending.
        </p>
      )}
    </div>
  );
}

// ---- Main layout -----------------------------------------------------------

export default function CopilotChat() {
  const {cohortsData, userEmail} = useCohorts();
  const chat = useCohortChat();
  const [query, setQuery] = useState('');
  const [railTab, setRailTab] = useState<'cohorts' | 'guide'>('cohorts');

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
                    Pin cohorts from the right to ground your questions, use the Guide to build one,
                    or just start typing.
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
              {/* Follow-up suggestions once a conversation is under way */}
              {chat.messages.length > 0 && !chat.isStreaming && (
                <div className="flex gap-1.5 mb-2 overflow-x-auto pb-0.5">
                  {suggestions.slice(0, 3).map(s => (
                    <button
                      key={s}
                      disabled={blocked}
                      onClick={() => chat.send(s)}
                      className="shrink-0 px-2.5 py-1 rounded-full border border-base-300 bg-base-100 text-xs text-base-content/70 hover:border-primary hover:text-primary transition-all disabled:opacity-50"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
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
          <div className="grid grid-cols-2 border-b border-base-300">
            <button
              onClick={() => setRailTab('cohorts')}
              className={`flex items-center justify-center gap-1.5 py-2.5 text-sm font-semibold border-b-2 transition-colors ${
                railTab === 'cohorts'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-base-content/50 hover:text-base-content'
              }`}
            >
              <Layers size={14} /> Cohorts
              {chat.selected.length > 0 && (
                <span className="badge badge-primary badge-xs">{chat.selected.length}</span>
              )}
            </button>
            <button
              onClick={() => setRailTab('guide')}
              className={`flex items-center justify-center gap-1.5 py-2.5 text-sm font-semibold border-b-2 transition-colors ${
                railTab === 'guide'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-base-content/50 hover:text-base-content'
              }`}
            >
              <Compass size={14} /> Guide
            </button>
          </div>

          {railTab === 'cohorts' ? (
            <>
              <div className="p-3 border-b border-base-300">
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
            </>
          ) : (
            <div className="flex-1 overflow-y-auto">
              <GuidePanel selected={chat.selected} onPrompt={chat.setInput} />
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}
