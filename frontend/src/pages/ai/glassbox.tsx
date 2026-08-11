'use client';

// Layout D — "Glass Box": nothing hidden, everything adjustable.
// The user edits the model's instructions, chooses exactly which metadata
// groups and how much variable detail are shared, and previews the precise
// payload before sending. The context is built client-side and sent as an
// override, so the preview is byte-for-byte what the model receives.
import React, {useMemo, useState} from 'react';
import Link from 'next/link';
import {ArrowLeft, Check, Copy, Eye, RotateCcw, Search, X} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {useCohortChat} from '@/components/ai/useCohortChat';
import {toBriefs} from '@/components/ai/chatClient';
import {
  buildClientContext,
  defaultContextOptions,
  estimateTokens,
  metadataGroups,
  promptPresets,
  variableDetailOptions,
  VariableDetail
} from '@/components/ai/contextBuilder';
import {Composer, DisabledNotice, ExperimentBadge, LoginNotice, MessageList} from '@/components/ai/ui';
import {Cohort} from '@/types';

function SectionTitle({step, children}: {step: number; children: React.ReactNode}) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <span className="w-5 h-5 rounded-full bg-sky-100 text-sky-700 text-xs font-bold inline-flex items-center justify-center">
        {step}
      </span>
      <span className="font-semibold text-sm">{children}</span>
    </div>
  );
}

export default function GlassBox() {
  const {cohortsData, userEmail} = useCohorts();
  const chat = useCohortChat();

  // -- user-controlled knobs -------------------------------------------------
  const [presetId, setPresetId] = useState<string | null>('default');
  const [systemPrompt, setSystemPrompt] = useState(promptPresets[0].text);
  const [groups, setGroups] = useState<{[key: string]: boolean}>(defaultContextOptions.groups);
  const [variableDetail, setVariableDetail] = useState<VariableDetail>('detailed');
  const [maxVars, setMaxVars] = useState(40);
  const [cohortQuery, setCohortQuery] = useState('');
  const [copied, setCopied] = useState(false);

  const briefs = useMemo(() => toBriefs(cohortsData || {}), [cohortsData]);
  const filteredCohorts = useMemo(() => {
    const q = cohortQuery.trim().toLowerCase();
    const list = q ? briefs.filter(b => b.id.toLowerCase().includes(q)) : briefs;
    // Selected first so the scope is always visible.
    return [...list].sort(
      (a, b) => Number(chat.selected.includes(b.id)) - Number(chat.selected.includes(a.id))
    ).slice(0, 40);
  }, [briefs, cohortQuery, chat.selected]);

  // -- the payload, exactly as the model will see it -------------------------
  const context = useMemo(() => {
    const all = Object.values(cohortsData || {}) as Cohort[];
    const selected = chat.selected.map(id => cohortsData?.[id]).filter(Boolean) as Cohort[];
    return buildClientContext(selected, all, {groups, variableDetail, maxVars});
  }, [cohortsData, chat.selected, groups, variableDetail, maxVars]);

  const contextTokens = estimateTokens(context);
  const promptTokens = estimateTokens(systemPrompt);
  const blocked = !chat.enabled || !userEmail;

  const applyPreset = (id: string) => {
    const preset = promptPresets.find(p => p.id === id);
    if (!preset) return;
    setPresetId(id);
    setSystemPrompt(preset.text);
  };

  const sharedGroupCount = metadataGroups.filter(g => groups[g.key]).length;

  const send = (text?: string) =>
    chat.send(text, {systemPrompt: systemPrompt.trim() || undefined, contextOverride: context});

  const copyPayload = async () => {
    try {
      await navigator.clipboard.writeText(`SYSTEM INSTRUCTIONS\n\n${systemPrompt}\n\nCOHORT CONTEXT\n\n${context}`);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <main className="h-[calc(100vh-8rem)] bg-base-200 flex flex-col">
      {/* Top bar */}
      <div className="border-b border-base-300 bg-base-100 px-6 py-3 flex items-center gap-3">
        <Link href="/ai" className="btn btn-ghost btn-sm gap-1">
          <ArrowLeft size={16} /> Hub
        </Link>
        <h1 className="font-bold text-lg">Glass Box</h1>
        <ExperimentBadge />
        <span className="hidden md:inline text-xs text-base-content/50">
          you control exactly what the model sees
        </span>
        {chat.model && <span className="text-xs text-base-content/50 ml-auto">model: {chat.model}</span>}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: the controls */}
        <aside className="w-[27rem] border-r border-base-300 bg-base-100 hidden lg:flex flex-col overflow-y-auto">
          <div className="p-4 space-y-5">
            {chat.configLoaded && !chat.enabled && <DisabledNotice />}
            {!userEmail && <LoginNotice />}

            {/* 1 · Scope */}
            <div>
              <SectionTitle step={1}>Cohorts in scope</SectionTitle>
              <label className="input input-sm input-bordered flex items-center gap-2 mb-2">
                <Search size={14} className="opacity-50" />
                <input
                  className="grow"
                  placeholder="Filter cohorts…"
                  value={cohortQuery}
                  onChange={e => setCohortQuery(e.target.value)}
                />
              </label>
              <div className="flex flex-wrap gap-1.5">
                {filteredCohorts.map(b => {
                  const active = chat.selected.includes(b.id);
                  return (
                    <button
                      key={b.id}
                      onClick={() => chat.toggleCohort(b.id)}
                      className={`badge gap-1 cursor-pointer transition-all ${
                        active ? 'badge-primary' : 'badge-outline hover:badge-primary'
                      }`}
                    >
                      {b.id}
                      {active && <X size={11} />}
                    </button>
                  );
                })}
              </div>
              {chat.selected.length === 0 && (
                <p className="text-xs text-base-content/50 mt-2">
                  Nothing selected — the model gets a thin catalog listing (ids and variable counts only).
                </p>
              )}
            </div>

            {/* 2 · Instructions */}
            <div>
              <SectionTitle step={2}>Model instructions</SectionTitle>
              <div className="flex flex-wrap gap-1 mb-2">
                {promptPresets.map(p => (
                  <button
                    key={p.id}
                    title={p.blurb}
                    onClick={() => applyPreset(p.id)}
                    className={`px-2 py-0.5 rounded-full text-xs border transition-all ${
                      presetId === p.id
                        ? 'border-sky-600 bg-sky-50 text-sky-700 font-medium'
                        : 'border-base-300 hover:border-sky-400 text-base-content/70'
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
                {presetId === null && (
                  <span className="px-2 py-0.5 rounded-full text-xs border border-sky-600 bg-sky-50 text-sky-700 font-medium">
                    custom
                  </span>
                )}
              </div>
              <textarea
                className="textarea textarea-bordered w-full text-xs leading-relaxed font-mono"
                rows={6}
                value={systemPrompt}
                onChange={e => {
                  setSystemPrompt(e.target.value);
                  setPresetId(null);
                }}
              />
              <div className="flex items-center justify-between mt-1">
                <span className="text-[11px] text-base-content/40">≈ {promptTokens} tokens</span>
                <button className="btn btn-ghost btn-xs gap-1" onClick={() => applyPreset('default')}>
                  <RotateCcw size={11} /> Reset to default
                </button>
              </div>
            </div>

            {/* 3 · Metadata sharing */}
            <div>
              <SectionTitle step={3}>Shared metadata</SectionTitle>
              <div className="space-y-1">
                {metadataGroups.map(g => (
                  <label
                    key={g.key}
                    className={`flex items-start gap-2 rounded-lg border px-3 py-2 cursor-pointer transition-all ${
                      groups[g.key] ? 'border-sky-300 bg-sky-50/50' : 'border-base-300 opacity-70'
                    }`}
                  >
                    <input
                      type="checkbox"
                      className="checkbox checkbox-xs checkbox-primary mt-0.5"
                      checked={!!groups[g.key]}
                      onChange={() => setGroups(prev => ({...prev, [g.key]: !prev[g.key]}))}
                    />
                    <span className="min-w-0">
                      <span className="block text-sm font-medium">{g.label}</span>
                      <span className="block text-[11px] text-base-content/50">{g.hint}</span>
                    </span>
                  </label>
                ))}
              </div>

              <div className="mt-3">
                <div className="text-xs font-semibold text-base-content/60 mb-1.5">Variable detail</div>
                <div className="join w-full">
                  {variableDetailOptions.map(opt => (
                    <button
                      key={opt.id}
                      title={opt.blurb}
                      onClick={() => setVariableDetail(opt.id)}
                      className={`join-item btn btn-xs flex-1 ${
                        variableDetail === opt.id ? 'btn-primary' : 'btn-ghost border border-base-300'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
                {variableDetail !== 'off' && (
                  <div className="mt-2">
                    <div className="flex justify-between text-[11px] text-base-content/50 mb-1">
                      <span>Max variables per cohort</span>
                      <span className="font-semibold">{maxVars}</span>
                    </div>
                    <input
                      type="range"
                      min={10}
                      max={150}
                      step={10}
                      value={maxVars}
                      onChange={e => setMaxVars(Number(e.target.value))}
                      className="range range-primary range-xs"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* 4 · Payload preview */}
            <div>
              <SectionTitle step={4}>Exactly what gets sent</SectionTitle>
              <div className="rounded-xl border border-base-300 overflow-hidden">
                <div className="flex items-center gap-2 px-3 py-2 bg-base-200 border-b border-base-300">
                  <Eye size={13} className="opacity-60" />
                  <span className="text-xs font-medium">
                    {sharedGroupCount}/{metadataGroups.length} metadata groups · ≈ {contextTokens} tokens
                  </span>
                  <button className="btn btn-ghost btn-xs ml-auto gap-1" onClick={copyPayload}>
                    {copied ? <Check size={11} /> : <Copy size={11} />}
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <pre className="text-[11px] leading-relaxed p-3 max-h-56 overflow-auto whitespace-pre-wrap font-mono bg-base-100">
                  {context}
                </pre>
              </div>
            </div>
          </div>
        </aside>

        {/* Right: conversation */}
        <section className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6">
            <div className="max-w-3xl mx-auto">
              {chat.messages.length === 0 ? (
                <div className="text-center mt-10">
                  <div className="inline-flex p-4 rounded-2xl bg-gradient-to-br from-cyan-500 to-sky-600 text-white shadow-lg mb-4">
                    <Eye size={28} />
                  </div>
                  <div className="text-2xl font-bold mb-2">Nothing hidden</div>
                  <p className="text-base-content/60 max-w-md mx-auto">
                    Tune the instructions, choose what metadata is shared, and check the payload preview
                    — then ask. What you see on the left is exactly what the model sees.
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
              <div className="flex items-center gap-2 mb-2 text-[11px] text-base-content/50">
                <span className="badge badge-ghost badge-xs">
                  {chat.selected.length ? `${chat.selected.length} cohort(s)` : 'catalog only'}
                </span>
                <span className="badge badge-ghost badge-xs">
                  {presetId ? promptPresets.find(p => p.id === presetId)?.label : 'custom instructions'}
                </span>
                <span className="badge badge-ghost badge-xs">≈ {contextTokens + promptTokens} tokens shared</span>
              </div>
              <Composer
                value={chat.input}
                onChange={chat.setInput}
                onSend={() => send()}
                onStop={chat.stop}
                isStreaming={chat.isStreaming}
                disabled={blocked}
                placeholder="Ask — with full control over what the model knows…"
              />
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
