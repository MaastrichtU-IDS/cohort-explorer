'use client';

// Layout C — "Prompt Studio": a guided, card-driven canvas.
// Pick an intent, a cohort, and a topic to assemble a well-formed question,
// or free-type. Great for discovering what to ask.
import React, {useEffect, useMemo, useState} from 'react';
import Link from 'next/link';
import {ArrowLeft, Zap, Target, Layers, Send, X} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {useCohortChat} from '@/components/ai/useCohortChat';
import {toBriefs} from '@/components/ai/chatClient';
import {Intent, intents, joinCohortLabel, topicBank} from '@/components/ai/promptKit';
import {DisabledNotice, ExperimentBadge, LoginNotice, MessageList, Composer} from '@/components/ai/ui';
import {withAiAccess} from '@/components/ai/guards';

// ---- Card components -------------------------------------------------------

function IntentCard({
  intent,
  active,
  onClick
}: {
  intent: Intent;
  active: boolean;
  onClick: () => void;
}) {
  const Icon = intent.icon;
  return (
    <button
      onClick={onClick}
      className={`relative rounded-xl border p-4 text-left transition-all overflow-hidden ${
        active
          ? 'border-primary bg-primary/5 shadow-sm'
          : 'border-base-300 bg-base-100 hover:border-primary/40 hover:shadow-sm'
      }`}
    >
      {active && <div className="absolute inset-x-0 top-0 h-0.5 bg-primary" />}
      <div className="flex items-center gap-2 mb-1">
        <div className={`inline-flex p-1.5 rounded-lg ${active ? 'bg-primary text-primary-content' : 'bg-base-200'}`}>
          <Icon size={16} />
        </div>
        <span className="font-semibold text-sm">{intent.label}</span>
      </div>
      <p className="text-xs text-base-content/60 leading-relaxed">{intent.blurb}</p>
    </button>
  );
}

function CohortChip({
  id,
  active,
  onClick
}: {
  id: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`badge badge-lg gap-1 cursor-pointer transition-all ${
        active ? 'badge-primary' : 'badge-outline hover:badge-primary'
      }`}
    >
      {id}
    </button>
  );
}

function TopicChip({topic, active, onClick}: {topic: string; active: boolean; onClick: () => void}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-full text-sm border transition-all ${
        active
          ? 'border-primary bg-primary/10 text-primary font-medium'
          : 'border-base-300 hover:border-primary/40 text-base-content/70'
      }`}
    >
      {topic}
    </button>
  );
}

// ---- Main layout -----------------------------------------------------------

function PromptStudio() {
  const {cohortsData, userEmail} = useCohorts();
  const chat = useCohortChat();
  const [intentId, setIntentId] = useState<string | null>(null);
  const [topic, setTopic] = useState('');
  const [customTopic, setCustomTopic] = useState('');
  const [cohortQuery, setCohortQuery] = useState('');

  const briefs = useMemo(() => toBriefs(cohortsData || {}), [cohortsData]);
  const filteredCohorts = useMemo(() => {
    if (!cohortQuery.trim()) return briefs.slice(0, 30);
    const q = cohortQuery.toLowerCase();
    return briefs.filter(b => b.id.toLowerCase().includes(q)).slice(0, 30);
  }, [briefs, cohortQuery]);

  const activeIntent = intents.find(i => i.id === intentId) || null;
  const effectiveTopic = customTopic.trim() || topic;

  const assembledPrompt = useMemo(() => {
    if (!activeIntent) return '';
    return activeIntent.template(joinCohortLabel(chat.selected), effectiveTopic);
  }, [activeIntent, chat.selected, effectiveTopic]);

  const blocked = !chat.enabled || !userEmail;

  const sendAssembled = () => {
    if (assembledPrompt.trim()) {
      chat.send(assembledPrompt);
    }
  };

  // Sync assembled prompt into the chat input so the user can edit before sending.
  useEffect(() => {
    if (assembledPrompt) {
      chat.setInput(assembledPrompt);
    }
  }, [assembledPrompt]);

  return (
    <main className="h-[calc(100vh-8rem)] bg-base-200 flex flex-col">
      {/* Top bar */}
      <div className="border-b border-base-300 bg-base-100 px-6 py-3 flex items-center gap-3">
        <Link href="/ai/alternatives" className="btn btn-ghost btn-sm gap-1">
          <ArrowLeft size={16} /> Alternatives
        </Link>
        <h1 className="font-bold text-lg">Prompt Studio</h1>
        <ExperimentBadge />
        {chat.model && <span className="text-xs text-base-content/50 ml-auto">model: {chat.model}</span>}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: guided builder */}
        <aside className="w-[28rem] border-r border-base-300 bg-base-100 hidden lg:flex flex-col overflow-y-auto">
          <div className="p-4 space-y-5">
            {chat.configLoaded && !chat.enabled && <DisabledNotice />}
            {!userEmail && <LoginNotice />}

            {/* Step 1: Intent */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="inline-flex p-1.5 rounded-lg bg-rose-100 text-rose-700">
                  <Target size={15} />
                </div>
                <span className="font-semibold text-sm">1 · Choose an intent</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {intents.map(intent => (
                  <IntentCard
                    key={intent.id}
                    intent={intent}
                    active={intentId === intent.id}
                    onClick={() => setIntentId(intent.id === intentId ? null : intent.id)}
                  />
                ))}
              </div>
            </div>

            {/* Step 2: Cohort */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="inline-flex p-1.5 rounded-lg bg-emerald-100 text-emerald-700">
                  <Layers size={15} />
                </div>
                <span className="font-semibold text-sm">2 · Pick cohort context</span>
              </div>
              <input
                className="input input-sm input-bordered w-full mb-2"
                placeholder="Filter cohorts…"
                value={cohortQuery}
                onChange={e => setCohortQuery(e.target.value)}
              />
              <div className="flex flex-wrap gap-1.5">
                {filteredCohorts.map(b => (
                  <CohortChip
                    key={b.id}
                    id={b.id}
                    active={chat.selected.includes(b.id)}
                    onClick={() => chat.toggleCohort(b.id)}
                  />
                ))}
              </div>
              {chat.selected.length > 0 && (
                <button
                  className="btn btn-ghost btn-xs mt-2"
                  onClick={chat.clearSelection}
                >
                  Clear selection
                </button>
              )}
            </div>

            {/* Step 3: Topic */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="inline-flex p-1.5 rounded-lg bg-indigo-100 text-indigo-700">
                  <Zap size={15} />
                </div>
                <span className="font-semibold text-sm">3 · Pick a topic (optional)</span>
              </div>
              <div className="flex flex-wrap gap-1.5 mb-2">
                {topicBank.map(t => (
                  <TopicChip
                    key={t}
                    topic={t}
                    active={topic === t}
                    onClick={() => setTopic(topic === t ? '' : t)}
                  />
                ))}
              </div>
              <input
                className="input input-sm input-bordered w-full"
                placeholder="…or type your own topic"
                value={customTopic}
                onChange={e => {
                  setCustomTopic(e.target.value);
                  if (e.target.value.trim()) setTopic('');
                }}
              />
            </div>

            {/* Assembled preview */}
            {assembledPrompt && (
              <div className="rounded-xl border border-primary/30 bg-primary/5 p-3">
                <div className="text-xs font-semibold text-primary mb-1 flex items-center gap-1">
                  <Send size={12} /> Assembled prompt
                </div>
                <p className="text-sm text-base-content/80 leading-relaxed">{assembledPrompt}</p>
                <button
                  className="btn btn-primary btn-sm w-full mt-3 gap-1"
                  onClick={sendAssembled}
                  disabled={blocked || chat.isStreaming}
                >
                  <Send size={14} /> Send to model
                </button>
              </div>
            )}
          </div>
        </aside>

        {/* Right: conversation */}
        <section className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6">
            <div className="max-w-3xl mx-auto">
              {chat.messages.length === 0 ? (
                <div className="text-center mt-10">
                  <div className="inline-flex p-4 rounded-2xl bg-gradient-to-br from-rose-500 to-orange-500 text-white shadow-lg mb-4">
                    <Zap size={28} />
                  </div>
                  <div className="text-2xl font-bold mb-2">Build a question, step by step</div>
                  <p className="text-base-content/60 max-w-md mx-auto">
                    Use the panel on the left to assemble a well-formed question from building blocks —
                    or just type freely below.
                  </p>
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
                placeholder="Type your question, or assemble one from the left…"
              />
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

export default withAiAccess(PromptStudio, {requireAdmin: true});
