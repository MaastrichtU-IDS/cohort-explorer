'use client';

// Layout G — "Hypothesis Lab": research brainstorming over the catalog.
// Instead of a conversation, the model's output becomes an idea board: pick a
// brainstorming mode (hypotheses, opportunities, gaps — or feasibility-check
// your own thesis), and each generated idea lands as a card you can keep,
// refine into an analysis plan, or discard. Kept ideas export as Markdown.
import React, {useEffect, useMemo, useRef, useState} from 'react';
import Link from 'next/link';
import {
  ArrowLeft,
  Download,
  GitBranch,
  HelpCircle,
  Plus,
  Search,
  Star,
  Trash2,
  TrendingUp,
  X,
  Zap
} from 'react-feather';
import type {ComponentType} from 'react';
import {useCohorts} from '@/components/CohortsContext';
import {fetchChatConfig, streamChat, toBriefs} from '@/components/ai/chatClient';
import {DisabledNotice, ExperimentBadge, LoginNotice, RichText, TypingDots} from '@/components/ai/ui';
import {withAiAccess} from '@/components/ai/guards';

// ---- Brainstorming modes ---------------------------------------------------

const IDEATION_SYSTEM =
  'You are a research strategist for the iCARE4CVD cohort catalog. You help analysts discover what ' +
  'they can DO with the data they actually have. Ground every suggestion strictly in the cohort ' +
  'metadata and variables provided in the context — always name the specific cohorts and variables ' +
  'involved, and never invent data that is not there. Format your answer ONLY as a series of ideas, ' +
  "each starting on its own line with '### ' followed by a short title, then the bullet points " +
  'requested. No introduction, no conclusion — just the ideas.';

interface Mode {
  id: string;
  label: string;
  icon: ComponentType<any>;
  blurb: string;
  needsThesis?: boolean;
  prompt: (topic: string) => string;
}

const modes: Mode[] = [
  {
    id: 'hypotheses',
    label: 'Hypotheses',
    icon: Zap,
    blurb: 'Testable theses the selected data could support.',
    prompt: topic =>
      'Propose 5 distinct, testable research hypotheses that could be investigated with the selected ' +
      `cohorts${topic ? `, focused on ${topic}` : ''}. For each idea use '### <short title>' followed by bullets: ` +
      '**Hypothesis** (one precise, falsifiable sentence), **Rationale** (why the metadata makes it plausible), ' +
      '**Data to use** (cohorts and specific variables), **Suggested design & analysis**, **Feasibility & limitations**.'
  },
  {
    id: 'opportunities',
    label: 'Opportunities',
    icon: TrendingUp,
    blurb: 'What an analyst could realistically do with this data.',
    prompt: topic =>
      'Map the concrete analysis opportunities in the selected cohorts — what could an analyst ' +
      `realistically do with this data${topic ? `, especially around ${topic}` : ''}? For each use '### <short title>' ` +
      'followed by bullets: **What** (the analysis in one sentence), **Using** (cohorts and specific variables), ' +
      '**Why it matters**, **Effort** (quick win / moderate / ambitious, with a one-line justification).'
  },
  {
    id: 'gaps',
    label: 'Gaps',
    icon: GitBranch,
    blurb: 'Underexplored angles and what would unlock them.',
    prompt: topic =>
      'Identify gaps and underexplored angles in the selected cohorts' +
      `${topic ? `, with particular attention to ${topic}` : ''}: measurements that exist but seem underused, ` +
      "combinations of cohorts nobody would think to join, and missing pieces. For each use '### <short title>' " +
      'followed by bullets: **The gap or angle**, **Evidence in the metadata** (which cohorts/variables show it), ' +
      '**What it would take** (data, linkage, or design needed to act on it).'
  },
  {
    id: 'feasibility',
    label: 'Check a thesis',
    icon: HelpCircle,
    blurb: 'Bring your own thesis — can this data test it?',
    needsThesis: true,
    prompt: () => '' // built from the thesis at send time
  }
];

const feasibilityPrompt = (thesis: string): string =>
  `An analyst proposes the following thesis:\n\n"${thesis}"\n\n` +
  "Assess whether the selected cohorts can test it. Answer with a single idea block: '### Feasibility: <short restatement>' " +
  'followed by bullets: **Verdict** (feasible / partially feasible / not with this data — and why), ' +
  '**Supporting data** (cohorts and specific variables that would be used), **Missing pieces**, ' +
  '**Suggested approach** (design and analysis if at least partially feasible), **Stronger alternative formulations** of the thesis.';

// ---- Idea parsing ----------------------------------------------------------

interface Idea {
  id: number;
  title: string;
  body: string;
  mode: string;
  kept: boolean;
  refinement: string;
  refining: boolean;
}

let nextIdeaId = 1;

function parseIdeas(text: string, mode: string): Idea[] {
  return text
    .split(/^###\s+/m)
    .map(chunk => chunk.trim())
    .filter(Boolean)
    .map(chunk => {
      const nl = chunk.indexOf('\n');
      const title = (nl === -1 ? chunk : chunk.slice(0, nl)).trim();
      const body = nl === -1 ? '' : chunk.slice(nl + 1).trim();
      return {id: nextIdeaId++, title, body, mode, kept: false, refinement: '', refining: false};
    })
    .filter(idea => idea.body); // drop stray preambles without content
}

// ---- Main layout -----------------------------------------------------------

function HypothesisLab() {
  const {cohortsData, userEmail} = useCohorts();
  const [scope, setScope] = useState<string[]>([]);
  const [scopeQuery, setScopeQuery] = useState('');
  const [modeId, setModeId] = useState('hypotheses');
  const [topic, setTopic] = useState('');
  const [thesis, setThesis] = useState('');
  const [ideas, setIdeas] = useState<Idea[]>([]);
  const [draft, setDraft] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [model, setModel] = useState('');
  const [configLoaded, setConfigLoaded] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

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
    return (q ? pool.filter(b => b.id.toLowerCase().includes(q)) : pool).slice(0, 8);
  }, [briefs, scope, scopeQuery]);

  const mode = modes.find(m => m.id === modeId)!;
  const blocked = !enabled || !userEmail;
  const canBrainstorm = !blocked && !isStreaming && (!mode.needsThesis || thesis.trim().length > 0);

  const brainstorm = async () => {
    if (!canBrainstorm) return;
    const prompt = mode.needsThesis ? feasibilityPrompt(thesis.trim()) : mode.prompt(topic.trim());
    setError(null);
    setDraft('');
    setIsStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    let buffer = '';
    try {
      await streamChat({
        messages: [{role: 'user', content: prompt}],
        cohortIds: scope,
        focus: topic.trim() || undefined,
        systemPrompt: IDEATION_SYSTEM,
        signal: controller.signal,
        onChunk: delta => {
          buffer += delta;
          setDraft(buffer);
        }
      });
      setIdeas(prev => [...parseIdeas(buffer, mode.id), ...prev]);
      setDraft('');
    } catch (e: any) {
      if (e?.name !== 'AbortError') {
        setError(e?.message || 'Something went wrong contacting the model.');
      } else if (buffer) {
        setIdeas(prev => [...parseIdeas(buffer, mode.id), ...prev]);
        setDraft('');
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  };

  const refine = async (idea: Idea) => {
    if (blocked || isStreaming || idea.refining) return;
    const patch = (p: Partial<Idea>) =>
      setIdeas(prev => prev.map(i => (i.id === idea.id ? {...i, ...p} : i)));
    patch({refining: true, refinement: ''});
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat({
        messages: [
          {
            role: 'user',
            content:
              'Develop the following research idea into a concrete analysis plan: refined hypothesis, ' +
              'study population, exposure and outcome variables (by name), statistical approach, key ' +
              'confounders to adjust for, and one sensitivity analysis. Stay grounded in the cohorts in context.\n\n' +
              `### ${idea.title}\n${idea.body}`
          }
        ],
        cohortIds: scope,
        systemPrompt: IDEATION_SYSTEM.replace('ONLY as a series of ideas', 'as a compact plan') ,
        signal: controller.signal,
        onChunk: delta =>
          setIdeas(prev => prev.map(i => (i.id === idea.id ? {...i, refinement: i.refinement + delta} : i)))
      });
    } catch (e: any) {
      if (e?.name !== 'AbortError') setError(e?.message || 'Refinement failed.');
    } finally {
      patch({refining: false});
      abortRef.current = null;
    }
  };

  const stop = () => abortRef.current?.abort();

  const keptCount = ideas.filter(i => i.kept).length;

  const exportKept = () => {
    const kept = ideas.filter(i => i.kept);
    const lines: string[] = ['# Research ideas', ''];
    lines.push(
      `_Generated with the Cohort Explorer Hypothesis Lab${model ? ` (model: ${model})` : ''}` +
        `${scope.length ? `, scoped to ${scope.join(', ')}` : ''}._`,
      ''
    );
    for (const idea of kept) {
      lines.push(`## ${idea.title}`, '', idea.body, '');
      if (idea.refinement) lines.push('### Analysis plan', '', idea.refinement, '');
    }
    const blob = new Blob([lines.join('\n')], {type: 'text/markdown'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'research-ideas.md';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="h-[calc(100vh-8rem)] bg-base-200 flex flex-col">
      {/* Top bar */}
      <div className="border-b border-base-300 bg-base-100 px-6 py-3 flex items-center gap-3">
        <Link href="/ai/alternatives" className="btn btn-ghost btn-sm gap-1">
          <ArrowLeft size={16} /> Alternatives
        </Link>
        <h1 className="font-bold text-lg">Hypothesis Lab</h1>
        <ExperimentBadge />
        <span className="hidden md:inline text-xs text-base-content/50">
          from data you have to theses you can test
        </span>
        <div className="ml-auto flex items-center gap-2">
          {model && <span className="text-xs text-base-content/50">model: {model}</span>}
          <button className="btn btn-outline btn-sm gap-1" onClick={exportKept} disabled={keptCount === 0}>
            <Download size={14} /> Export {keptCount > 0 ? `${keptCount} kept` : 'kept'}
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: the bench */}
        <aside className="w-[24rem] border-r border-base-300 bg-base-100 hidden lg:flex flex-col overflow-y-auto">
          <div className="p-4 space-y-5">
            {configLoaded && !enabled && <DisabledNotice />}
            {!userEmail && <LoginNotice />}

            {/* Scope */}
            <div>
              <div className="font-semibold text-sm mb-2">Data on the bench</div>
              <div className="flex flex-wrap gap-1.5 mb-2">
                {scope.map(id => (
                  <span key={id} className="badge badge-primary gap-1">
                    {id}
                    <button onClick={() => setScope(prev => prev.filter(x => x !== id))}>
                      <X size={11} />
                    </button>
                  </span>
                ))}
              </div>
              <label className="input input-sm input-bordered flex items-center gap-2 mb-2">
                <Search size={14} className="opacity-50" />
                <input
                  className="grow"
                  placeholder="Add cohorts…"
                  value={scopeQuery}
                  onChange={e => setScopeQuery(e.target.value)}
                />
              </label>
              <div className="flex flex-wrap gap-1.5">
                {scopeCandidates.map(b => (
                  <button
                    key={b.id}
                    onClick={() => {
                      setScope(prev => [...prev, b.id]);
                      setScopeQuery('');
                    }}
                    className="badge badge-outline gap-1 hover:badge-primary transition-all"
                  >
                    <Plus size={11} /> {b.id}
                  </button>
                ))}
              </div>
              <p className="text-xs text-base-content/50 mt-2">
                {scope.length === 0
                  ? 'No cohorts selected — ideas will draw on the whole catalog at a glance.'
                  : `Ideas will be grounded in ${scope.length} cohort(s), metadata and variables included.`}
              </p>
            </div>

            {/* Mode */}
            <div>
              <div className="font-semibold text-sm mb-2">What are we brainstorming?</div>
              <div className="space-y-1.5">
                {modes.map(m => {
                  const Icon = m.icon;
                  const active = modeId === m.id;
                  return (
                    <button
                      key={m.id}
                      onClick={() => setModeId(m.id)}
                      className={`w-full flex items-start gap-2.5 rounded-lg border px-3 py-2 text-left transition-all ${
                        active ? 'border-primary bg-primary/5' : 'border-base-300 hover:border-primary/40'
                      }`}
                    >
                      <span
                        className={`inline-flex p-1.5 rounded-lg mt-0.5 ${
                          active ? 'bg-primary text-primary-content' : 'bg-base-200'
                        }`}
                      >
                        <Icon size={14} />
                      </span>
                      <span className="min-w-0">
                        <span className="block text-sm font-semibold">{m.label}</span>
                        <span className="block text-xs text-base-content/50">{m.blurb}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Focus / thesis */}
            {mode.needsThesis ? (
              <div>
                <div className="font-semibold text-sm mb-2">Your thesis</div>
                <textarea
                  className="textarea textarea-bordered w-full text-sm"
                  rows={3}
                  placeholder="e.g. Long-term beta-blocker use is associated with slower cognitive decline in elderly heart-failure patients."
                  value={thesis}
                  onChange={e => setThesis(e.target.value)}
                />
              </div>
            ) : (
              <div>
                <div className="font-semibold text-sm mb-2">
                  Focus <span className="font-normal text-base-content/50">(optional)</span>
                </div>
                <input
                  className="input input-sm input-bordered w-full"
                  placeholder="e.g. hypertension, frailty, medication adherence…"
                  value={topic}
                  onChange={e => setTopic(e.target.value)}
                />
              </div>
            )}

            {isStreaming ? (
              <button className="btn btn-error w-full gap-2" onClick={stop}>
                Stop
              </button>
            ) : (
              <button className="btn btn-primary w-full gap-2" onClick={brainstorm} disabled={!canBrainstorm}>
                <Zap size={16} /> Brainstorm
              </button>
            )}

            {error && (
              <div className="alert alert-error text-xs py-2">
                <span>{error}</span>
              </div>
            )}
          </div>
        </aside>

        {/* Right: the idea board */}
        <section className="flex-1 overflow-y-auto p-6">
          {/* Live draft while the model thinks */}
          {isStreaming && (
            <div className="max-w-3xl mx-auto mb-5 rounded-xl border border-primary/30 bg-primary/5 p-4">
              <div className="text-xs font-semibold text-primary mb-2 flex items-center gap-2">
                <TypingDots /> drafting ideas…
              </div>
              {draft ? (
                <RichText text={draft} className="prose prose-sm max-w-none leading-relaxed [&_*]:my-0 opacity-70" />
              ) : null}
            </div>
          )}

          {ideas.length === 0 && !isStreaming ? (
            <div className="text-center mt-14 max-w-md mx-auto">
              <div className="inline-flex p-4 rounded-2xl bg-gradient-to-br from-lime-500 to-emerald-600 text-white shadow-lg mb-4">
                <Zap size={28} />
              </div>
              <div className="text-2xl font-bold mb-2">What could this data prove?</div>
              <p className="text-base-content/60">
                Put cohorts on the bench, pick a brainstorming mode, and generate. Each idea arrives as a
                card — keep the promising ones, refine them into analysis plans, discard the rest, and
                export what survives.
              </p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-4">
              {ideas.map(idea => {
                const ideaMode = modes.find(m => m.id === idea.mode);
                return (
                  <div
                    key={idea.id}
                    className={`rounded-xl border bg-base-100 p-4 shadow-sm transition-all ${
                      idea.kept ? 'border-primary ring-1 ring-primary/30' : 'border-base-300'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          {ideaMode && (
                            <span className="badge badge-ghost badge-xs gap-1">
                              <ideaMode.icon size={10} /> {ideaMode.label}
                            </span>
                          )}
                        </div>
                        <h2 className="font-bold mt-1">{idea.title}</h2>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          className={`btn btn-xs gap-1 ${idea.kept ? 'btn-primary' : 'btn-ghost'}`}
                          title={idea.kept ? 'Kept — click to unkeep' : 'Keep this idea'}
                          onClick={() =>
                            setIdeas(prev => prev.map(i => (i.id === idea.id ? {...i, kept: !i.kept} : i)))
                          }
                        >
                          <Star size={12} /> {idea.kept ? 'Kept' : 'Keep'}
                        </button>
                        <button
                          className="btn btn-ghost btn-xs"
                          title="Develop into an analysis plan"
                          disabled={blocked || isStreaming || idea.refining}
                          onClick={() => refine(idea)}
                        >
                          <TrendingUp size={12} /> Refine
                        </button>
                        <button
                          className="btn btn-ghost btn-xs text-error"
                          title="Discard"
                          onClick={() => setIdeas(prev => prev.filter(i => i.id !== idea.id))}
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>
                    <div className="mt-2">
                      <RichText text={idea.body} />
                    </div>
                    {(idea.refinement || idea.refining) && (
                      <div className="mt-3 rounded-lg bg-base-200/60 border border-base-300 p-3">
                        <div className="text-xs font-semibold text-base-content/60 mb-1.5 flex items-center gap-1.5">
                          <TrendingUp size={11} /> Analysis plan
                          {idea.refining && <TypingDots />}
                        </div>
                        {idea.refinement && <RichText text={idea.refinement} />}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

export default withAiAccess(HypothesisLab, {requireAdmin: true});
