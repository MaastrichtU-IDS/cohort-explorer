'use client';

// Conversation Starters manager — admin only.
// Manage the model-generated conversation starters shown on the iCARE-AI chat
// landing page and the keyword themes used in Guided Exploration:
//   - generate new starters, optionally steered by a direction/theme prompt
//   - re-run the keyword grouping over the pool
//   - inspect and prune the pool
// Generation is admin-driven only (no automatic generation at app startup).
import React, {useEffect, useMemo, useState} from 'react';
import {Cpu, MessageSquare, Plus, RefreshCw, Search, Tag, Trash2, Zap} from 'react-feather';
import {withAiAccess} from '@/components/ai/guards';
import {
  ContextDiagnostics,
  StarterManageData,
  adminAddStarter,
  adminContextDiagnostics,
  adminDeleteStarters,
  adminFetchStarterPool,
  adminGenerateStarters,
  adminRegroupStarters
} from '@/components/ai/chatClient';
import {DisabledNotice, ExperimentBadge} from '@/components/ai/ui';

function ConversationStarterManager() {
  const [data, setData] = useState<StarterManageData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [direction, setDirection] = useState('');
  const [generating, setGenerating] = useState(false);
  const [regrouping, setRegrouping] = useState(false);
  const [filter, setFilter] = useState('');
  const [openKeyword, setOpenKeyword] = useState<string | null>(null);
  const [diagnostics, setDiagnostics] = useState<ContextDiagnostics | null>(null);
  const [diagnosing, setDiagnosing] = useState(false);
  const [probing, setProbing] = useState(false);
  const [newText, setNewText] = useState('');
  const [newKind, setNewKind] = useState<'interesting' | 'basic'>('interesting');
  const [adding, setAdding] = useState(false);

  const refresh = async () => {
    try {
      setData(await adminFetchStarterPool());
    } catch (e: any) {
      setError(e?.message || 'Failed to load the starter pool.');
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const run = async (
    action: () => Promise<any>,
    setBusy: (b: boolean) => void,
    describe: (result: any) => string
  ) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const result = await action();
      if (result?.error) {
        setError(result.error);
      } else {
        setNotice(describe(result));
      }
      await refresh();
    } catch (e: any) {
      setError(e?.message || 'The request failed.');
    } finally {
      setBusy(false);
    }
  };

  const generate = () =>
    run(
      () => adminGenerateStarters(direction),
      setGenerating,
      r =>
        `Generated ${r.parsed} starter(s), ${r.added} new after deduplication` +
        `${r.direction ? ` (direction: “${r.direction}”)` : ''}. Pool size: ${r.pool_size}.`
    );

  const regroup = () =>
    run(() => adminRegroupStarters(), setRegrouping, r => `Regrouped the pool into ${r.groups} keyword theme(s).`);

  const remove = (text: string) =>
    run(() => adminDeleteStarters([text]), () => {}, r => `Deleted ${r.deleted} starter(s); ${r.remaining} remain.`);

  const addStarter = () => {
    if (!newText.trim()) return;
    run(
      () => adminAddStarter(newText.trim(), newKind),
      setAdding,
      r => (r.added ? `Added starter. Pool size: ${r.pool_size}.` : 'That starter is already in the pool.')
    ).then(() => setNewText(''));
  };

  const runDiagnostics = async (probeWindow: boolean) => {
    const setBusy = probeWindow ? setProbing : setDiagnosing;
    setBusy(true);
    setError(null);
    try {
      setDiagnostics(await adminContextDiagnostics(probeWindow));
    } catch (e: any) {
      setError(e?.message || 'Diagnostics failed.');
    } finally {
      setBusy(false);
    }
  };

  const filteredStarters = useMemo(() => {
    if (!data) return [];
    const q = filter.trim().toLowerCase();
    return q
      ? data.starters.filter(
          s => s.text.toLowerCase().includes(q) || (s.direction || '').toLowerCase().includes(q)
        )
      : data.starters;
  }, [data, filter]);

  const busy = generating || regrouping || adding;

  return (
    <main className="min-h-[calc(100vh-8rem)] bg-base-200">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-2xl font-bold">Conversation Starters</h1>
          <ExperimentBadge />
          <span className="badge badge-sm badge-neutral">admin</span>
          {data?.model && (
            <span className="text-xs text-base-content/50 ml-auto">model: {data.model}</span>
          )}
        </div>
        <p className="text-sm text-base-content/60 mb-6">
          The pool of model-generated questions shown on the iCARE-AI chat page, and the keyword
          themes offered in Guided Exploration. Generation runs only from this page.
        </p>

        {data && !data.chat_enabled && (
          <div className="mb-4">
            <DisabledNotice />
          </div>
        )}
        {error && (
          <div className="alert alert-error text-sm mb-4">
            <span>{error}</span>
          </div>
        )}
        {notice && (
          <div className="alert alert-success text-sm mb-4">
            <span>{notice}</span>
          </div>
        )}

        {/* Generate */}
        <section className="rounded-xl border border-base-300 bg-base-100 p-4 mb-5">
          <div className="flex items-center gap-2 font-semibold mb-2">
            <Zap size={16} /> Generate new starters
          </div>
          <p className="text-xs text-base-content/60 mb-3">
            Asks the model for 8 interesting + 6 basic questions grounded in the catalog, appends the
            new ones to the pool, then refreshes the keyword themes. Optionally steer it with a
            direction or theme.
          </p>
          <textarea
            className="textarea textarea-bordered w-full text-sm"
            rows={2}
            placeholder="Optional direction/theme — e.g. medication adherence in elderly heart-failure patients"
            value={direction}
            onChange={e => setDirection(e.target.value)}
          />
          <button
            className="btn btn-primary btn-sm mt-3 gap-2"
            disabled={busy || !data?.chat_enabled}
            onClick={generate}
          >
            {generating ? (
              <>
                <span className="loading loading-spinner loading-xs" /> Generating… this can take a minute
              </>
            ) : (
              <>
                <Zap size={14} /> Generate
              </>
            )}
          </button>
        </section>

        {/* Keyword themes */}
        <section className="rounded-xl border border-base-300 bg-base-100 p-4 mb-5">
          <div className="flex items-center gap-2 font-semibold mb-2">
            <Tag size={16} /> Keyword themes
            <span className="text-xs font-normal text-base-content/50">
              {data?.keywords_meta?.generated_at
                ? `last grouped ${data.keywords_meta.generated_at} over ${data.keywords_meta.pool_size} starter(s)`
                : 'not grouped yet'}
            </span>
            <button
              className="btn btn-outline btn-xs ml-auto gap-1"
              disabled={busy || !data?.chat_enabled}
              onClick={regroup}
            >
              {regrouping ? (
                <span className="loading loading-spinner loading-xs" />
              ) : (
                <RefreshCw size={12} />
              )}
              Re-group now
            </button>
          </div>
          {data && data.keywords.length === 0 ? (
            <p className="text-sm text-base-content/50">
              No keyword themes yet — generate starters or re-group the pool.
            </p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {data?.keywords.map(k => (
                <button
                  key={k.keyword}
                  onClick={() => setOpenKeyword(openKeyword === k.keyword ? null : k.keyword)}
                  className={`badge badge-lg gap-1 cursor-pointer ${
                    openKeyword === k.keyword ? 'badge-primary' : 'badge-outline'
                  }`}
                >
                  {k.keyword} <span className="opacity-60">({k.count})</span>
                </button>
              ))}
            </div>
          )}
          {openKeyword && (
            <ul className="mt-3 text-sm text-base-content/70 list-disc ml-5 space-y-0.5">
              {data?.keywords
                .find(k => k.keyword === openKeyword)
                ?.questions.map(q => <li key={q}>{q}</li>)}
            </ul>
          )}
        </section>

        {/* Pool */}
        <section className="rounded-xl border border-base-300 bg-base-100 p-4">
          <div className="flex items-center gap-2 font-semibold mb-3">
            <MessageSquare size={16} /> Starter pool
            <span className="text-xs font-normal text-base-content/50">
              {data ? `${data.starters.length} starter(s)` : 'loading…'}
            </span>
            <label className="input input-xs input-bordered flex items-center gap-1 ml-auto w-56">
              <Search size={12} className="opacity-50" />
              <input
                className="grow"
                placeholder="Filter…"
                value={filter}
                onChange={e => setFilter(e.target.value)}
              />
            </label>
          </div>

          {/* Manually add a starter */}
          <div className="flex flex-col sm:flex-row gap-2 mb-4 items-stretch">
            <input
              className="input input-sm input-bordered flex-1"
              placeholder="Write a conversation starter…"
              value={newText}
              onChange={e => setNewText(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  addStarter();
                }
              }}
            />
            <select
              className="select select-sm select-bordered"
              value={newKind}
              onChange={e => setNewKind(e.target.value as 'interesting' | 'basic')}
            >
              <option value="interesting">interesting</option>
              <option value="basic">basic</option>
            </select>
            <button className="btn btn-primary btn-sm gap-1" disabled={busy || !newText.trim()} onClick={addStarter}>
              {adding ? <span className="loading loading-spinner loading-xs" /> : <Plus size={14} />}
              Add
            </button>
          </div>

          {data && data.starters.length === 0 ? (
            <p className="text-sm text-base-content/50">
              The pool is empty — the chat page is showing its static fallback questions. Generate
              some starters above.
            </p>
          ) : (
            <div className="divide-y divide-base-200">
              {filteredStarters.map(s => (
                <div key={s.text} className="py-2 flex items-start gap-3 group">
                  <span
                    className={`badge badge-xs mt-1 shrink-0 ${
                      s.kind === 'basic' ? 'badge-ghost' : 'badge-primary badge-outline'
                    }`}
                  >
                    {s.kind}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm leading-snug">{s.text}</div>
                    <div className="text-[11px] text-base-content/40">
                      {s.generated_at}
                      {s.direction ? ` · direction: ${s.direction}` : ''}
                    </div>
                  </div>
                  <button
                    className="btn btn-ghost btn-xs text-error opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                    title="Delete this starter"
                    disabled={busy}
                    onClick={() => remove(s.text)}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Context diagnostics — kept at the bottom (least frequently used). */}
        <section className="rounded-xl border border-base-300 bg-base-100 p-4 mt-5">
          <div className="flex items-center gap-2 font-semibold mb-2">
            <Cpu size={16} /> Context diagnostics
            <div className="ml-auto flex gap-2">
              <button className="btn btn-outline btn-xs gap-1" disabled={diagnosing || probing} onClick={() => runDiagnostics(false)}>
                {diagnosing ? <span className="loading loading-spinner loading-xs" /> : <RefreshCw size={12} />}
                Measure sizes
              </button>
              <button
                className="btn btn-outline btn-xs gap-1"
                disabled={diagnosing || probing || !data?.chat_enabled}
                onClick={() => runDiagnostics(true)}
                title="Sends increasingly large prompts to the model until one is rejected — can take minutes"
              >
                {probing ? <span className="loading loading-spinner loading-xs" /> : <Cpu size={12} />}
                Probe context window
              </button>
            </div>
          </div>
          <p className="text-xs text-base-content/60 mb-3">
            Measures how large the catalog would be in each candidate encoding, what LiteLLM claims the
            model&apos;s limits are, and (via the probe) what prompt size the deployed server actually accepts.
          </p>
          {diagnostics && (
            <div className="text-sm space-y-3">
              <div className="grid sm:grid-cols-3 gap-2">
                <div className="rounded-lg bg-base-200 p-2.5">
                  <div className="text-xs text-base-content/50">Catalog</div>
                  <div className="font-semibold">
                    {diagnostics.sizes.n_cohorts} cohorts · {diagnostics.sizes.n_variables.toLocaleString()} variables
                  </div>
                  <div className="text-xs text-base-content/50">
                    {diagnostics.sizes.n_distinct_concepts.toLocaleString()} distinct concepts
                  </div>
                </div>
                <div className="rounded-lg bg-base-200 p-2.5">
                  <div className="text-xs text-base-content/50">Context sizes (≈ tokens)</div>
                  <div className="text-xs mt-1">
                    current thin catalog: <b>{diagnostics.sizes.current_catalog_context_tokens.toLocaleString()}</b>
                    <br />
                    concept index: <b>{diagnostics.sizes.concept_index_tokens.toLocaleString()}</b>
                    <br />
                    full detail: <b>{diagnostics.sizes.full_detail_tokens.toLocaleString()}</b>
                  </div>
                </div>
                <div className="rounded-lg bg-base-200 p-2.5">
                  <div className="text-xs text-base-content/50">LiteLLM model info</div>
                  <div className="text-xs mt-1">
                    {diagnostics.model_info
                      ? `max input: ${diagnostics.model_info.max_input_tokens?.toLocaleString() ?? '—'} · max output: ${
                          diagnostics.model_info.max_output_tokens?.toLocaleString() ?? '—'
                        }`
                      : diagnostics.model_info_error || 'not reported'}
                  </div>
                </div>
              </div>
              {diagnostics.window_probe && (
                <div>
                  <div className="text-xs font-semibold text-base-content/60 mb-1">Empirical window probe</div>
                  <div className="flex flex-wrap gap-1.5">
                    {diagnostics.window_probe.map(p => (
                      <span
                        key={p.approx_tokens}
                        title={p.error}
                        className={`badge badge-sm ${p.ok ? 'badge-success badge-outline' : 'badge-error'}`}
                      >
                        ~{(p.approx_tokens / 1000).toFixed(0)}k {p.ok ? '✓' : '✗'}
                      </span>
                    ))}
                  </div>
                  {diagnostics.window_probe.some(p => !p.ok) && (
                    <p className="text-[11px] text-base-content/50 mt-1">
                      First rejected size’s error:{' '}
                      {diagnostics.window_probe.find(p => !p.ok)?.error}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

export default withAiAccess(ConversationStarterManager, {requireAdmin: true});
