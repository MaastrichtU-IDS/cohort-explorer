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
import {ChevronDown, ChevronLeft, ChevronUp, Compass, Home, MessageCircle, RefreshCw, Search, Send, X} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {useCohortChat} from '@/components/ai/useCohortChat';
import {withAiAccess} from '@/components/ai/guards';
import {
  StarterKeyword,
  ConversationStarter,
  MappingPairStatus,
  fetchStarterKeywords,
  fetchConversationStarters,
  fetchMappingStatus,
  generateMappingPair,
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

// Distinct color per keyword theme (deterministic: same keyword -> same color).
// Full class strings so Tailwind's JIT keeps them.
const KEYWORD_COLORS = [
  'bg-blue-100 text-blue-900 border-blue-200',
  'bg-emerald-100 text-emerald-900 border-emerald-200',
  'bg-amber-100 text-amber-900 border-amber-200',
  'bg-rose-100 text-rose-900 border-rose-200',
  'bg-violet-100 text-violet-900 border-violet-200',
  'bg-cyan-100 text-cyan-900 border-cyan-200',
  'bg-orange-100 text-orange-900 border-orange-200',
  'bg-teal-100 text-teal-900 border-teal-200',
  'bg-fuchsia-100 text-fuchsia-900 border-fuchsia-200',
  'bg-lime-100 text-lime-900 border-lime-200'
];
function keywordColor(kw: string): string {
  let h = 0;
  for (let i = 0; i < kw.length; i++) h = (h * 31 + kw.charCodeAt(i)) >>> 0;
  return KEYWORD_COLORS[h % KEYWORD_COLORS.length];
}

// ---- Cohort focus: collapsible picker ---------------------------------------

function CohortFocus({
  selected,
  onToggle,
  onClear,
  open,
  onOpenChange,
  maxCohorts
}: {
  selected: string[];
  onToggle: (id: string) => void;
  onClear: () => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  maxCohorts?: number;
}) {
  const {cohortsData} = useCohorts();
  const [query, setQuery] = useState('');
  const atMax = maxCohorts !== undefined && selected.length >= maxCohorts;

  const briefs = useMemo(() => toBriefs(cohortsData || {}), [cohortsData]);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    // Without a search, show only cohorts that have variables uploaded — those
    // are the ones with data to explore. Searching reaches the whole catalog.
    return q ? briefs.filter(b => b.id.toLowerCase().includes(q)) : briefs.filter(b => b.variableCount > 0);
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
              // When at the cap, non-selected cohorts can't be added (only removals).
              const disabled = !active && atMax;
              return (
                <button
                  key={b.id}
                  onClick={() => !disabled && onToggle(b.id)}
                  disabled={disabled}
                  title={disabled ? `Maximum ${maxCohorts} cohorts` : b.studyType || undefined}
                  className={`badge gap-1 transition-all ${
                    active
                      ? 'border-blue-300 bg-blue-100 text-blue-900 cursor-pointer'
                      : disabled
                        ? 'badge-outline opacity-40 cursor-not-allowed'
                        : 'badge-outline hover:border-blue-300 cursor-pointer'
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
          {atMax && (
            <p className="text-[11px] text-blue-900/70 mt-2">Maximum {maxCohorts} cohorts selected.</p>
          )}
          {!query.trim() && (
            <p className="text-[11px] text-base-content/40 mt-2">
              Showing cohorts with uploaded variables. Search to find any cohort in the catalog.
            </p>
          )}
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

// ---- Cross-cohort mapping availability ---------------------------------------
//
// When 2+ cohorts are selected, show for each pair whether a cached mapping
// file is available to the assistant, and offer to generate missing ones via
// the same pipeline as the mapping page. Generation is slow (minutes), so the
// button turns into a progress state while it runs.
function MappingAvailability({selected}: {selected: string[]}) {
  const [pairs, setPairs] = useState<MappingPairStatus[]>([]);
  const [generating, setGenerating] = useState<string | null>(null); // "src|tgt"
  const [genError, setGenError] = useState<string | null>(null);

  const refresh = () => {
    if (selected.length < 2) {
      setPairs([]);
      return;
    }
    fetchMappingStatus(selected)
      .then(setPairs)
      .catch(() => setPairs([]));
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(refresh, [selected.join('|')]);

  if (selected.length < 2 || pairs.length === 0) return null;

  const generate = async (source: string, target: string) => {
    const key = `${source}|${target}`;
    setGenerating(key);
    setGenError(null);
    try {
      await generateMappingPair(source, target);
      refresh();
    } catch (err: any) {
      setGenError(err?.message || 'Mapping generation failed.');
    } finally {
      setGenerating(null);
    }
  };

  return (
    <div className="mb-2 space-y-1">
      {pairs.map(p => {
        const key = `${p.source}|${p.target}`;
        if (p.cached) {
          return (
            <div key={key} className="text-[11px] text-base-content/60 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />
              Mapping {p.source} → {p.target} available to the assistant
              <span className="font-mono text-[10px] bg-base-300/60 px-1.5 py-0.5 rounded">{p.filename}</span>
            </div>
          );
        }
        const busy = generating === key;
        return (
          <div key={key} className="flex items-center gap-2">
            <button
              className="btn btn-xs btn-outline btn-primary normal-case"
              disabled={generating !== null}
              onClick={() => generate(p.source, p.target)}
            >
              {busy ? (
                <>
                  <span className="loading loading-spinner loading-xs" />
                  Generating mapping {p.source} → {p.target}… (this can take several minutes)
                </>
              ) : (
                <>Generate the mapping between {p.source} and {p.target} and let the assistant use it</>
              )}
            </button>
          </div>
        );
      })}
      {genError && <div className="text-[11px] text-error">{genError}</div>}
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
// Intents whose question needs no topic/criteria input at all.
const NO_TOPIC_INTENTS = new Set(['compare', 'summarize']);
// Compare requires between 2 and 4 cohorts.
const COMPARE_MIN = 2;
const COMPARE_MAX = 4;
// Topic chips are multi-select, capped like the cohort picker.
const MAX_TOPICS = 5;

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
      return nCohorts >= COMPARE_MIN
        ? {ready: true}
        : {ready: false, hint: `Select ${COMPARE_MIN} to ${COMPARE_MAX} cohorts to compare.`};
    case 'summarize':
      return nCohorts >= 1
        ? {ready: true}
        : {ready: false, hint: 'Pick a cohort above, or send as is for an overview of the whole catalog.'};
    case 'hypothesis':
      return topic
        ? {ready: true}
        : {ready: false, hint: 'State your hypothesis above, or send as is to get help formulating one.'};
    case 'research':
      return topic || nCohorts > 0
        ? {ready: true}
        : {ready: false, hint: 'Pick a keyword or a cohort to sharpen the questions, or send as is.'};
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
  onAsk: (text: string, meta?: {intent?: string | null; topics?: string}) => void;
  blocked: boolean;
}) {
  const [intentId, setIntentId] = useState<string | null>(null);
  // Up to MAX_TOPICS chip selections, plus an optional free-text topic.
  const [topics, setTopics] = useState<string[]>([]);
  const [customTopic, setCustomTopic] = useState('');
  const [cohortsOpen, setCohortsOpen] = useState(false);
  const [keywords, setKeywords] = useState<StarterKeyword[]>([]);

  const toggleTopic = (t: string) =>
    setTopics(prev => (prev.includes(t) ? prev.filter(x => x !== t) : prev.length >= MAX_TOPICS ? prev : [...prev, t]));

  // Thematic keywords derived from the conversation-starter pool.
  useEffect(() => {
    fetchStarterKeywords().then(setKeywords);
  }, []);

  const selectIntent = (id: string) => {
    setIntentId(id);
    setTopics([]);
    setCustomTopic('');
    // Cohort-centric intents make the cohort picker the natural next step.
    setCohortsOpen(COHORT_CENTRIC_INTENTS.has(id));
  };

  const goBack = () => {
    setIntentId(null);
    setTopics([]);
    setCustomTopic('');
    setCohortsOpen(false);
    onClearCohorts();
  };

  const activeIntent = guidedIntents.find(i => i.id === intentId) || null;
  // Selected chip topics plus any free-text topic, as one comma-joined string.
  const effectiveTopic = [...topics, customTopic.trim()].filter(Boolean).join(', ');
  const atTopicMax = topics.length >= MAX_TOPICS;
  const assembled = activeIntent
    ? activeIntent.template(joinCohortLabel(selected), effectiveTopic)
    : '';

  const isResearch = intentId === 'research';
  const freeTextOnly = intentId !== null && FREE_TEXT_INTENTS.has(intentId);
  const noTopic = intentId !== null && NO_TOPIC_INTENTS.has(intentId);
  // Keyword chips for research (fallback to the generic topics until the
  // starter pool has been grouped); generic topic chips otherwise.
  const chips = isResearch && keywords.length > 0 ? keywords.map(k => k.keyword) : topicBank;
  const {ready, hint} = intentReadiness(intentId, effectiveTopic, selected.length);
  // Hard gate (button not clickable), as opposed to the soft readiness hint:
  // Compare requires at least COMPARE_MIN cohorts.
  const hardBlocked = intentId === 'compare' && selected.length < COMPARE_MIN;

  const submit = () => {
    if (blocked || hardBlocked || !assembled.trim()) return;
    onAsk(assembled, {intent: intentId, topics: effectiveTopic});
  };
  // Enter (without Shift) in a guided input submits, like clicking Ask.
  const onInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  // Step 1 — pick an intent. The cards are the only thing on screen.
  if (!activeIntent) {
    return (
      <div className="max-w-4xl mx-auto px-4 pb-10">
        <div className="grid sm:grid-cols-2 gap-5">
          {guidedIntents.map(intent => {
            const Icon = intent.icon;
            return (
              <button
                key={intent.id}
                onClick={() => selectIntent(intent.id)}
                className="rounded-2xl border border-base-300 bg-base-100 p-6 text-left transition-all hover:border-blue-300 hover:shadow-md"
              >
                <div className="flex items-center gap-3 mb-2">
                  <span className="inline-flex p-2.5 rounded-xl bg-base-200">
                    <Icon size={22} />
                  </span>
                  <span className="font-semibold text-lg">{intent.label}</span>
                </div>
                <p className="text-sm text-base-content/60 leading-relaxed">{intent.blurb}</p>
              </button>
            );
          })}
        </div>
        <p className="text-center mt-8">
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

      {/* Topic / criteria / hypothesis / keywords — omitted for intents that
          need no topic (Compare, Summarize). */}
      {!noTopic && (
        <div className={freeTextOnly ? 'rounded-xl ring-2 ring-blue-300 p-4 bg-base-100' : undefined}>
          <div className="text-sm font-semibold text-base-content/60 uppercase tracking-wide mb-3 text-center">
            {activeIntent.topicLabel}
            {isResearch && keywords.length > 0 && (
              <span className="normal-case font-normal text-base-content/40"> · suggested themes</span>
            )}
            {!freeTextOnly && (
              <span className="normal-case font-normal text-base-content/40"> · pick up to {MAX_TOPICS}</span>
            )}
          </div>
          {/* Criteria/hypothesis are free text of the user's own; chips only get in the way there. */}
          {!freeTextOnly && (
            <div className="flex flex-wrap justify-center gap-1.5 mb-2">
              {chips.map(t => {
                const active = topics.includes(t);
                const disabled = !active && atTopicMax;
                return (
                  <button
                    key={t}
                    onClick={() => toggleTopic(t)}
                    disabled={disabled}
                    title={disabled ? `Up to ${MAX_TOPICS} topics` : undefined}
                    className={`px-3 py-1 rounded-full text-sm border transition-all ${
                      active
                        ? 'border-blue-300 bg-blue-100 text-blue-900 font-medium'
                        : disabled
                          ? 'border-base-300 text-base-content/30 cursor-not-allowed'
                          : 'border-base-300 hover:border-blue-300 text-base-content/70'
                    }`}
                  >
                    {t}
                  </button>
                );
              })}
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
              onChange={e => setCustomTopic(e.target.value)}
              onKeyDown={onInputKeyDown}
            />
          ) : (
            <input
              key={intentId || 'none'}
              autoFocus={freeTextOnly}
              className="input input-bordered w-full"
              placeholder={activeIntent.topicPlaceholder}
              value={customTopic}
              onChange={e => setCustomTopic(e.target.value)}
              onKeyDown={onInputKeyDown}
            />
          )}
        </div>
      )}

      {/* Cohort focus */}
      <CohortFocus
        selected={selected}
        onToggle={onToggleCohort}
        onClear={onClearCohorts}
        open={cohortsOpen}
        onOpenChange={setCohortsOpen}
        maxCohorts={intentId === 'compare' ? COMPARE_MAX : undefined}
      />

      {/* Assembled question */}
      <div className="rounded-xl border border-blue-200 bg-blue-50 p-4">
        <div className="text-xs font-semibold text-blue-900 mb-1.5">Your question</div>
        <p className="text-sm text-base-content/80 leading-relaxed mb-3">{assembled}</p>
        <button
          className={`btn w-full gap-2 bg-blue-100 text-blue-900 hover:bg-blue-200 border-blue-300 ${ready && !blocked && !hardBlocked ? 'shimmer-nudge' : ''}`}
          disabled={blocked || hardBlocked}
          onClick={submit}
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
  const {userEmail} = useCohorts();
  const chat = useCohortChat();
  const [mode, setMode] = useState<'guided' | 'chat'>('chat');
  const [starters, setStarters] = useState<ConversationStarter[]>([]);
  // Bumped to remount GuidedExploration back to its landing (step 1).
  const [guidedKey, setGuidedKey] = useState(0);

  // Random selection of conversation starters (fresh per visit). "More
  // starters" re-fetches for a new random 4.
  const shuffleStarters = () => fetchConversationStarters(4).then(setStarters);
  useEffect(() => {
    shuffleStarters();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const blocked = !chat.enabled || !userEmail;

  // Guided Exploration sends its assembled question as a fresh conversation.
  const ask = (text: string, meta?: {intent?: string | null; topics?: string}) => {
    setMode('chat');
    chat.send(text, {
      startNew: true,
      arrivalPath: 'intention_cards',
      entryContext: {
        intent: meta?.intent || null,
        topics: meta?.topics || '',
        cohortIds: chat.selected,
        focus: chat.focus
      }
    });
  };

  // Return to the chat landing, clearing any ongoing conversation. Used by both
  // the Home button and the Chat mode button (they behave identically).
  const goHome = () => {
    chat.reset();
    setMode('chat');
  };

  // Jump to the Guided Exploration landing (intent cards), resetting any
  // in-progress wizard step and cohort selection.
  const goGuided = () => {
    chat.clearSelection();
    setGuidedKey(k => k + 1);
    setMode('guided');
  };

  const modeButton = (m: 'guided' | 'chat', label: string, Icon: any) => (
    <button
      onClick={() => (m === 'chat' ? goHome() : goGuided())}
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
            onClick={goHome}
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
            key={guidedKey}
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
            <div className="flex justify-center">
              <MappingAvailability selected={chat.selected} />
            </div>
            {starters.length > 0 && (
              <p className="text-center text-xs text-base-content/50 mb-2">
                Some example questions (auto generated)
              </p>
            )}
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
                          <span key={kw} className={`px-2 py-0.5 rounded-full text-[11px] border ${keywordColor(kw)}`}>
                            {kw}
                          </span>
                        ))}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
            {starters.length > 0 && (
              <div className="text-center mt-3">
                <button
                  onClick={shuffleStarters}
                  className="inline-flex items-center gap-1 text-xs text-base-content/50 hover:text-blue-900 transition-colors"
                >
                  <RefreshCw size={12} /> More starters
                </button>
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
              <div className="max-w-5xl mx-auto">
                <MessageList messages={chat.messages} streaming={chat.isStreaming} onSummaryViewed={chat.markSummaryViewed} />
                {chat.error && (
                  <div className="alert alert-error mt-4 text-sm">
                    <span>{chat.error}</span>
                  </div>
                )}
              </div>
            </div>
            <div className="bg-base-200 px-4 md:px-8 pb-3">
              <div className="max-w-5xl mx-auto">
                {/* In an ongoing conversation this is a follow-up field: no
                    placeholder, no suggestion chips — just continue the thread.
                    Any pinned cohort scope stays visible as context. */}
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
                <MappingAvailability selected={chat.selected} />
                <Composer
                  value={chat.input}
                  onChange={chat.setInput}
                  onSend={() => chat.send()}
                  onStop={chat.stop}
                  isStreaming={chat.isStreaming}
                  disabled={blocked}
                  placeholder=""
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

export default withAiAccess(ICareAI);
