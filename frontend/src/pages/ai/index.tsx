'use client';

// iCARE-AI — the main AI interface. Two modes, switched by two big buttons at
// the top:
//   - Guided Exploration: pick what you want to do (identify / compare /
//     hypothesis / research questions / …), optionally focus on a topic and on
//     specific cohorts, then send the assembled question.
//   - Chat: a landing-style centered chat box with conversation starters drawn
//     at random from an admin-managed, model-generated pool (see /ai/starters).
// Sending from either mode lands in the same conversation.
// Requires login. Alternative experimental layouts live under
// /ai/alternatives (admins only, not linked from here on purpose).
import React, {useEffect, useMemo, useState} from 'react';
import {ChevronDown, ChevronLeft, ChevronUp, Compass, Home, MessageCircle, Search, Send, X} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {useCohortChat} from '@/components/ai/useCohortChat';
import {withAiAccess} from '@/components/ai/guards';
import {
  StarterKeyword,
  ConversationStarter,
  buildSuggestions,
  fetchStarterKeywords,
  fetchConversationStarters,
  toBriefs
} from '@/components/ai/chatClient';
import {guidedIntents, joinCohortLabel, topicBank} from '@/components/ai/promptKit';
import {
  Composer,
  DisabledNotice,
  ExperimentBadge,
  LocalModelNote,
  MessageList
} from '@/components/ai/ui';

// ---- Cohort focus: collapsible picker ---------------------------------------

function CohortFocus({
  selected,
  onToggle,
  onClear,
  open,
  onOpenChange
}: {
  selected: string[];
  onToggle: (id: string) => void;
  onClear: () => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const {cohortsData} = useCohorts();
  const [query, setQuery] = useState('');

  const briefs = useMemo(() => toBriefs(cohortsData || {}), [cohortsData]);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? briefs.filter(b => b.id.toLowerCase().includes(q)) : briefs;
  }, [briefs, query]);

  return (
    <div>
      <button
        onClick={() => onOpenChange(!open)}
        className="w-full flex items-center justify-center gap-2 rounded-xl border border-base-300 bg-base-100 py-2 text-sm text-base-content/70 hover:border-blue-300 transition-all"
      >
        {open ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        <span>
          [optional] Focus on specific cohorts…
          {selected.length > 0 && (
            <span className="badge badge-sm ml-2 bg-blue-100 text-blue-900 border-blue-300">{selected.length} selected</span>
          )}
        </span>
      </button>

      {open && (
        <div className="mt-2 rounded-xl border border-base-300 bg-base-100 p-3">
          <label className="input input-sm input-bordered flex items-center gap-2 mb-3">
            <Search size={14} className="opacity-50" />
            <input
              className="grow"
              placeholder="Search cohorts…"
              value={query}
              onChange={e => setQuery(e.target.value)}
            />
          </label>
          <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto">
            {filtered.map(b => {
              const active = selected.includes(b.id);
              return (
                <button
                  key={b.id}
                  onClick={() => onToggle(b.id)}
                  title={b.studyType || undefined}
                  className={`badge gap-1 cursor-pointer transition-all ${
                    active ? 'border-blue-300 bg-blue-100 text-blue-900' : 'badge-outline hover:border-blue-300'
                  }`}
                >
                  {b.id}
                  {active && <X size={11} />}
                </button>
              );
            })}
            {filtered.length === 0 && (
              <span className="text-xs text-base-content/40 py-2">No cohorts match “{query}”</span>
            )}
          </div>
          {selected.length > 0 && (
            <button className="btn btn-ghost btn-xs mt-2" onClick={onClear}>
              Clear selection
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ---- Guided Exploration mode ------------------------------------------------

// Per-intent flow rules: which intents open the cohort picker on selection,
// which replace the generic topic chips (keywords for research, free text only
// for criteria/hypothesis), and when a question counts as "well-specified"
// (readiness drives the shimmer nudge on the Ask button).
const COHORT_CENTRIC_INTENTS = new Set(['compare', 'summarize']);
const FREE_TEXT_INTENTS = new Set(['identify', 'hypothesis']);

function intentReadiness(
  intentId: string | null,
  topic: string,
  nCohorts: number
): {ready: boolean; hint?: string} {
  switch (intentId) {
    case 'identify':
      return topic
        ? {ready: true}
        : {ready: false, hint: 'Describe your criteria above to search for matching cohorts.'};
    case 'compare':
      return nCohorts >= 2
        ? {ready: true}
        : {
            ready: false,
            hint: 'Tip: select at least two cohorts above — as is, the comparison will span the whole catalog.'
          };
    case 'summarize':
      return nCohorts >= 1
        ? {ready: true}
        : {ready: false, hint: 'Pick a cohort above — or send as is for an overview of the whole catalog.'};
    case 'hypothesis':
      return topic
        ? {ready: true}
        : {ready: false, hint: 'State your hypothesis above — or send as is to get help formulating one.'};
    case 'research':
      return topic || nCohorts > 0
        ? {ready: true}
        : {ready: false, hint: 'Pick a keyword or a cohort to sharpen the questions — or send as is.'};
    default:
      return {ready: true};
  }
}

function GuidedExploration({
  selected,
  onToggleCohort,
  onClearCohorts,
  onAsk,
  blocked
}: {
  selected: string[];
  onToggleCohort: (id: string) => void;
  onClearCohorts: () => void;
  onAsk: (text: string) => void;
  blocked: boolean;
}) {
  const [intentId, setIntentId] = useState<string | null>(null);
  const [topic, setTopic] = useState('');
  const [customTopic, setCustomTopic] = useState('');
  const [cohortsOpen, setCohortsOpen] = useState(false);
  const [keywords, setKeywords] = useState<StarterKeyword[]>([]);

  // Thematic keywords derived from the conversation-starter pool.
  useEffect(() => {
    fetchStarterKeywords().then(setKeywords);
  }, []);

  const selectIntent = (id: string) => {
    setIntentId(id);
    setTopic('');
    setCustomTopic('');
    // Cohort-centric intents make the cohort picker the natural next step.
    setCohortsOpen(COHORT_CENTRIC_INTENTS.has(id));
  };

  const goBack = () => {
    setIntentId(null);
    setTopic('');
    setCustomTopic('');
    setCohortsOpen(false);
    onClearCohorts();
  };

  const activeIntent = guidedIntents.find(i => i.id === intentId) || null;
  const effectiveTopic = customTopic.trim() || topic;
  const assembled = activeIntent
    ? activeIntent.template(joinCohortLabel(selected), effectiveTopic)
    : '';

  const isResearch = intentId === 'research';
  const freeTextOnly = intentId !== null && FREE_TEXT_INTENTS.has(intentId);
  // Keyword chips for research (fallback to the generic topics until the
  // starter pool has been grouped); generic topic chips otherwise.
  const chips = isResearch && keywords.length > 0 ? keywords.map(k => k.keyword) : topicBank;
  const {ready, hint} = intentReadiness(intentId, effectiveTopic, selected.length);

  // Step 1 — pick an intent. The cards are the only thing on screen.
  if (!activeIntent) {
    return (
      <div className="max-w-3xl mx-auto px-4 pb-10">
        <div className="grid sm:grid-cols-2 gap-3">
          {guidedIntents.map(intent => {
            const Icon = intent.icon;
            return (
              <button
                key={intent.id}
                onClick={() => selectIntent(intent.id)}
                className="rounded-xl border border-base-300 bg-base-100 p-4 text-left transition-all hover:border-blue-300 hover:shadow-sm"
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="inline-flex p-1.5 rounded-lg bg-base-200">
                    <Icon size={16} />
                  </span>
                  <span className="font-semibold text-sm">{intent.label}</span>
                </div>
                <p className="text-xs text-base-content/60 leading-relaxed">{intent.blurb}</p>
              </button>
            );
          })}
        </div>
        <p className="text-center mt-6">
          <LocalModelNote />
        </p>
      </div>
    );
  }

  // Step 2 — the options relevant to the chosen intent, plus a back button.
  const ActiveIcon = activeIntent.icon;
  return (
    <div className="max-w-3xl mx-auto px-4 pb-10 space-y-6">
      <div className="flex items-center gap-3">
        <button className="btn btn-ghost btn-sm gap-1" onClick={goBack}>
          <ChevronLeft size={16} /> Back
        </button>
        <div className="flex items-center gap-2">
          <span className="inline-flex p-1.5 rounded-lg bg-blue-100 text-blue-900">
            <ActiveIcon size={16} />
          </span>
          <span className="font-semibold">{activeIntent.label}</span>
        </div>
      </div>

      {/* Topic / criteria / hypothesis / keywords */}
      <div className={freeTextOnly ? 'rounded-xl ring-2 ring-blue-300 p-4 bg-base-100' : undefined}>
        <div className="text-sm font-semibold text-base-content/60 uppercase tracking-wide mb-3 text-center">
          {activeIntent.topicLabel}
          {isResearch && keywords.length > 0 && (
            <span className="normal-case font-normal text-base-content/40"> — suggested themes</span>
          )}
        </div>
        {/* Criteria/hypothesis are free text of the user's own; chips only get in the way there. */}
        {!freeTextOnly && (
          <div className="flex flex-wrap justify-center gap-1.5 mb-2">
            {chips.map(t => (
              <button
                key={t}
                onClick={() => {
                  setTopic(topic === t ? '' : t);
                  setCustomTopic('');
                }}
                className={`px-3 py-1 rounded-full text-sm border transition-all ${
                  topic === t
                    ? 'border-blue-300 bg-blue-100 text-blue-900 font-medium'
                    : 'border-base-300 hover:border-blue-300 text-base-content/70'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        )}
        {intentId === 'hypothesis' ? (
          <textarea
            key={intentId}
            autoFocus
            rows={2}
            className="textarea textarea-bordered w-full"
            placeholder={activeIntent.topicPlaceholder}
            value={customTopic}
            onChange={e => {
              setCustomTopic(e.target.value);
              if (e.target.value.trim()) setTopic('');
            }}
          />
        ) : (
          <input
            key={intentId || 'none'}
            autoFocus={freeTextOnly}
            className="input input-bordered w-full"
            placeholder={activeIntent.topicPlaceholder}
            value={customTopic}
            onChange={e => {
              setCustomTopic(e.target.value);
              if (e.target.value.trim()) setTopic('');
            }}
          />
        )}
      </div>

      {/* Cohort focus */}
      <CohortFocus
        selected={selected}
        onToggle={onToggleCohort}
        onClear={onClearCohorts}
        open={cohortsOpen}
        onOpenChange={setCohortsOpen}
      />

      {/* Assembled question */}
      <div className="rounded-xl border border-blue-200 bg-blue-50 p-4">
        <div className="text-xs font-semibold text-blue-900 mb-1.5">Your question</div>
        <p className="text-sm text-base-content/80 leading-relaxed mb-3">{assembled}</p>
        <button
          className={`btn w-full gap-2 bg-blue-100 text-blue-900 hover:bg-blue-200 border-blue-300 ${ready && !blocked ? 'shimmer-nudge' : ''}`}
          disabled={blocked}
          onClick={() => onAsk(assembled)}
        >
          <Send size={15} /> Ask iCARE-AI
        </button>
        {hint && <p className="text-xs text-base-content/50 text-center mt-2">{hint}</p>}
      </div>

      <p className="text-center">
        <LocalModelNote />
      </p>
    </div>
  );
}

// ---- Main layout -----------------------------------------------------------

function ICareAI() {
  const {cohortsData, userEmail} = useCohorts();
  const chat = useCohortChat();
  const [mode, setMode] = useState<'guided' | 'chat'>('chat');
  const [starters, setStarters] = useState<ConversationStarter[]>([]);

  // Random selection of conversation starters (fresh per visit).
  useEffect(() => {
    fetchConversationStarters(6).then(setStarters);
  }, []);

  const followUps = useMemo(
    () => buildSuggestions(cohortsData || {}, chat.selected).slice(0, 3),
    [cohortsData, chat.selected]
  );

  const blocked = !chat.enabled || !userEmail;

  const ask = (text: string) => {
    setMode('chat');
    chat.send(text);
  };

  const modeButton = (m: 'guided' | 'chat', label: string, Icon: any) => (
    <button
      onClick={() => setMode(m)}
      className={`flex items-center gap-2.5 px-8 py-3.5 rounded-2xl border-2 text-base font-semibold transition-all ${
        mode === m
          ? 'border-blue-300 bg-blue-100 text-blue-900 shadow-md'
          : 'border-base-300 bg-base-100 text-base-content/70 hover:border-blue-300 hover:text-base-content'
      }`}
    >
      <Icon size={19} /> {label}
    </button>
  );

  return (
    <main className="h-[calc(100vh-8rem)] bg-base-200 flex flex-col">
      {/* Top bar */}
      <div className="border-b border-base-300 bg-base-100 px-6 py-3 flex items-center gap-3">
        {(chat.messages.length > 0 || mode === 'guided') && (
          <button
            className="btn btn-ghost btn-sm gap-1.5"
            title="Back to the iCARE-AI landing page"
            onClick={() => {
              chat.reset();
              setMode('chat');
            }}
          >
            <Home size={15} /> Home
          </button>
        )}
        <h1 className="font-bold text-lg">iCARE-AI</h1>
        <ExperimentBadge />
        {chat.model && (
          <div className="ml-auto text-right leading-tight">
            <div className="text-xs text-base-content/50">model: {chat.model}</div>
            <LocalModelNote className="text-[10px] text-base-content/40" />
          </div>
        )}
      </div>

      {/* Mode switcher */}
      <div className="flex justify-center gap-4 py-5">
        {modeButton('chat', 'Chat', MessageCircle)}
        {modeButton('guided', 'Guided Exploration', Compass)}
      </div>

      <div className="flex-1 overflow-y-auto flex flex-col">
        {chat.configLoaded && !chat.enabled && (
          <div className="max-w-2xl mx-auto px-4 w-full mb-4">
            <DisabledNotice />
          </div>
        )}

        {mode === 'guided' ? (
          <GuidedExploration
            selected={chat.selected}
            onToggleCohort={chat.toggleCohort}
            onClearCohorts={chat.clearSelection}
            onAsk={ask}
            blocked={blocked}
          />
        ) : chat.messages.length === 0 ? (
          /* Chat landing: big centered box + suggested questions */
          <div className="flex-1 flex flex-col justify-center max-w-3xl mx-auto w-full px-4 pb-16">
            <h2 className="text-3xl font-bold text-center mb-5">What would you like to know?</h2>
            <div className="shadow-lg rounded-2xl">
              <Composer
                value={chat.input}
                onChange={chat.setInput}
                onSend={() => chat.send()}
                onStop={chat.stop}
                isStreaming={chat.isStreaming}
                disabled={blocked}
                placeholder="Ask about the studies…"
                large
              />
            </div>
            <p className="text-center mt-2 mb-7">
              <LocalModelNote />
            </p>
            {starters.length > 0 && (
              <div className="grid sm:grid-cols-2 gap-4">
                {starters.map(q => (
                  <button
                    key={q.text}
                    disabled={blocked}
                    onClick={() => chat.send(q.text)}
                    className="flex flex-col gap-2 text-left rounded-2xl border border-base-300 bg-base-100 px-5 py-4 hover:border-blue-300 hover:shadow-md transition-all disabled:opacity-50"
                  >
                    <span className="text-[15px] leading-snug text-base-content/80">{q.text}</span>
                    {q.keywords && q.keywords.length > 0 && (
                      <span className="flex flex-wrap gap-1">
                        {q.keywords.map(kw => (
                          <span
                            key={kw}
                            className="px-2 py-0.5 rounded-full text-[11px] bg-blue-100 text-blue-900 border border-blue-200"
                          >
                            {kw}
                          </span>
                        ))}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
            {chat.error && (
              <div className="alert alert-error mt-4 text-sm">
                <span>{chat.error}</span>
              </div>
            )}
          </div>
        ) : (
          /* Ongoing conversation */
          <div className="flex-1 flex flex-col min-h-0">
            <div className="flex-1 overflow-y-auto px-4 md:px-8 py-4">
              <div className="max-w-3xl mx-auto">
                <MessageList messages={chat.messages} streaming={chat.isStreaming} />
                {chat.error && (
                  <div className="alert alert-error mt-4 text-sm">
                    <span>{chat.error}</span>
                  </div>
                )}
              </div>
            </div>
            <div className="border-t border-base-300 bg-base-100 px-4 md:px-8 py-3">
              <div className="max-w-3xl mx-auto">
                {!chat.isStreaming && (
                  <div className="flex gap-1.5 mb-2 overflow-x-auto pb-0.5">
                    {followUps.map(s => (
                      <button
                        key={s}
                        disabled={blocked}
                        onClick={() => chat.send(s)}
                        className="shrink-0 px-2.5 py-1 rounded-full border border-base-300 bg-base-100 text-xs text-base-content/70 hover:border-blue-300 hover:text-blue-900 transition-all disabled:opacity-50"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                )}
                {chat.selected.length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {chat.selected.map(id => (
                      <span key={id} className="badge badge-sm gap-1 bg-blue-100 text-blue-900 border-blue-300">
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
                <div className="text-center mt-1.5">
                  <LocalModelNote />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

export default withAiAccess(ICareAI);
