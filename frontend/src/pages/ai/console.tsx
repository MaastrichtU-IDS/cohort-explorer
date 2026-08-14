'use client';

// Layout B — "Cohort Console": a split workspace.
// Left pane: browse the cohort catalog and drill into variables.
// Right pane: chat with the model. Selecting a cohort pins it as context.
import React, {useMemo, useState} from 'react';
import Link from 'next/link';
import {ArrowLeft, Search, ChevronRight, X, Database, Eye, Layers} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {useCohortChat} from '@/components/ai/useCohortChat';
import {buildSuggestions} from '@/components/ai/chatClient';
import {Cohort, Variable} from '@/types';
import {Composer, DisabledNotice, ExperimentBadge, LoginNotice, MessageList} from '@/components/ai/ui';
import {withAiAccess} from '@/components/ai/guards';

// ---- Cohort + variable browser ---------------------------------------------

function VariableRow({v}: {v: Variable}) {
  return (
    <div className="px-3 py-2 border-b border-base-200 last:border-0">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-medium text-sm truncate">{v.var_name}</span>
        {v.var_type && <span className="badge badge-ghost badge-xs shrink-0">{v.var_type}</span>}
      </div>
      {v.var_label && <div className="text-xs text-base-content/60 truncate">{v.var_label}</div>}
      {(v.omop_domain || v.concept_name) && (
        <div className="text-[11px] text-base-content/40 truncate mt-0.5">
          {v.omop_domain ? `${v.omop_domain}` : ''}
          {v.omop_domain && v.concept_name ? ' · ' : ''}
          {v.concept_name ? v.concept_name : ''}
        </div>
      )}
    </div>
  );
}

function CohortDetail({cohort, onPin, pinned}: {cohort: Cohort; onPin: () => void; pinned: boolean}) {
  const [varQuery, setVarQuery] = useState('');
  const variables = useMemo(() => {
    const all = Object.values(cohort.variables || {});
    if (!varQuery.trim()) return all.slice(0, 80);
    const q = varQuery.toLowerCase();
    return all
      .filter(
        v =>
          v.var_name?.toLowerCase().includes(q) ||
          v.var_label?.toLowerCase().includes(q) ||
          v.omop_domain?.toLowerCase().includes(q) ||
          v.concept_name?.toLowerCase().includes(q)
      )
      .slice(0, 80);
  }, [cohort.variables, varQuery]);

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-base-300">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="font-bold text-sm truncate">{cohort.cohort_id}</div>
            {cohort.institution && <div className="text-xs text-base-content/50 truncate">{cohort.institution}</div>}
          </div>
          <button
            onClick={onPin}
            className={`btn btn-xs gap-1 shrink-0 ${pinned ? 'btn-primary' : 'btn-outline btn-primary'}`}
          >
            {pinned ? <><Eye size={12} /> Pinned</> : <><Eye size={12} /> Pin</>}
          </button>
        </div>
        {cohort.study_type && (
          <div className="mt-2 flex flex-wrap gap-1">
            <span className="badge badge-ghost badge-xs">{cohort.study_type}</span>
            {cohort.study_participants && (
              <span className="badge badge-ghost badge-xs">{cohort.study_participants}</span>
            )}
            {cohort.eda_version && (
              <span className="badge badge-ghost badge-xs">EDA {cohort.eda_version}</span>
            )}
          </div>
        )}
        {cohort.study_objective && (
          <p className="text-xs text-base-content/60 mt-2 line-clamp-3">{cohort.study_objective}</p>
        )}
      </div>
      <div className="p-2 border-b border-base-200">
        <label className="input input-sm input-bordered flex items-center gap-2">
          <Search size={14} className="opacity-50" />
          <input
            className="grow"
            placeholder="Search variables…"
            value={varQuery}
            onChange={e => setVarQuery(e.target.value)}
          />
        </label>
      </div>
      <div className="flex-1 overflow-y-auto">
        {variables.length === 0 ? (
          <div className="text-center text-sm text-base-content/40 py-8">No variables found</div>
        ) : (
          variables.map(v => <VariableRow key={v.var_name + v.index} v={v} />)
        )}
        {Object.keys(cohort.variables || {}).length > 80 && !varQuery.trim() && (
          <div className="text-center text-xs text-base-content/40 py-3">
            Showing first 80 of {Object.keys(cohort.variables).length} variables — search to filter
          </div>
        )}
      </div>
    </div>
  );
}

function CohortList({
  cohorts,
  onSelect,
  selectedId,
  pinnedIds,
  onPin,
}: {
  cohorts: Cohort[];
  onSelect: (id: string) => void;
  selectedId: string | null;
  pinnedIds: string[];
  onPin: (id: string) => void;
}) {
  const [query, setQuery] = useState('');
  const filtered = useMemo(() => {
    const sorted = [...cohorts].sort(
      (a, b) => Object.keys(b.variables || {}).length - Object.keys(a.variables || {}).length
    );
    if (!query.trim()) return sorted;
    const q = query.toLowerCase();
    return sorted.filter(
      c =>
        c.cohort_id?.toLowerCase().includes(q) ||
        c.institution?.toLowerCase().includes(q) ||
        c.study_type?.toLowerCase().includes(q)
    );
  }, [cohorts, query]);

  return (
    <div className="flex flex-col h-full">
      <div className="p-2 border-b border-base-200">
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
      <div className="flex-1 overflow-y-auto">
        {filtered.map(c => {
          const isActive = c.cohort_id === selectedId;
          const isPinned = pinnedIds.includes(c.cohort_id);
          const varCount = Object.keys(c.variables || {}).length;
          return (
            <button
              key={c.cohort_id}
              onClick={() => onSelect(c.cohort_id)}
              className={`w-full text-left px-3 py-2 border-b border-base-200 last:border-0 transition-colors ${
                isActive ? 'bg-primary/10' : 'hover:bg-base-200'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium text-sm truncate">{c.cohort_id}</span>
                <div className="flex items-center gap-1 shrink-0">
                  {isPinned && <span className="badge badge-primary badge-xs">pinned</span>}
                  <span className="badge badge-ghost badge-xs">{varCount}</span>
                </div>
              </div>
              {c.study_type && <div className="text-xs text-base-content/50 truncate">{c.study_type}</div>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ---- Main layout -----------------------------------------------------------

function CohortConsole() {
  const {cohortsData, userEmail} = useCohorts();
  const chat = useCohortChat();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<'list' | 'detail'>('list');

  const cohorts = useMemo(() => Object.values(cohortsData || {}) as Cohort[], [cohortsData]);
  const selectedCohort = selectedId ? cohortsData?.[selectedId] : null;
  const suggestions = useMemo(
    () => buildSuggestions(cohortsData || {}, chat.selected),
    [cohortsData, chat.selected]
  );
  const blocked = !chat.enabled || !userEmail;

  const handleSelect = (id: string) => {
    setSelectedId(id);
    setView('detail');
  };

  const handlePin = (id: string) => {
    chat.toggleCohort(id);
  };

  return (
    <main className="h-[calc(100vh-8rem)] bg-base-200 flex flex-col">
      {/* Top bar */}
      <div className="border-b border-base-300 bg-base-100 px-6 py-3 flex items-center gap-3">
        <Link href="/ai/alternatives" className="btn btn-ghost btn-sm gap-1">
          <ArrowLeft size={16} /> Alternatives
        </Link>
        <h1 className="font-bold text-lg">Cohort Console</h1>
        <ExperimentBadge />
        {chat.model && <span className="text-xs text-base-content/50 ml-auto">model: {chat.model}</span>}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: browser pane */}
        <aside className="w-96 border-r border-base-300 bg-base-100 hidden md:flex flex-col">
          {/* Breadcrumb / back */}
          <div className="px-3 py-2 border-b border-base-300 flex items-center gap-2">
            {view === 'detail' && selectedCohort ? (
              <>
                <button
                  className="btn btn-ghost btn-xs gap-1"
                  onClick={() => setView('list')}
                >
                  <ChevronRight size={14} className="rotate-180" /> Back
                </button>
                <span className="text-xs text-base-content/50 truncate">{selectedCohort.cohort_id}</span>
              </>
            ) : (
              <div className="flex items-center gap-2 text-sm font-semibold">
                <Database size={15} className="opacity-60" />
                Cohort Catalog
              </div>
            )}
          </div>
          {view === 'detail' && selectedCohort ? (
            <CohortDetail
              cohort={selectedCohort}
              onPin={() => handlePin(selectedCohort.cohort_id)}
              pinned={chat.selected.includes(selectedCohort.cohort_id)}
            />
          ) : (
            <CohortList
              cohorts={cohorts}
              onSelect={handleSelect}
              selectedId={selectedId}
              pinnedIds={chat.selected}
              onPin={handlePin}
            />
          )}
        </aside>

        {/* Right: chat pane */}
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
                  <div className="text-2xl font-bold mb-2">Interrogate the catalog</div>
                  <p className="text-base-content/60 mb-6">
                    Browse cohorts on the left, pin them as context, then ask the model anything.
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
                <div className="flex flex-wrap gap-1 mb-2 items-center">
                  <span className="text-xs text-base-content/50 mr-1 flex items-center gap-1">
                    <Layers size={12} /> Context:
                  </span>
                  {chat.selected.map(id => (
                    <span key={id} className="badge badge-primary badge-sm gap-1">
                      {id}
                      <button onClick={() => chat.toggleCohort(id)}>
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                  <button className="btn btn-ghost btn-xs" onClick={chat.clearSelection}>
                    Clear
                  </button>
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
      </div>
    </main>
  );
}

export default withAiAccess(CohortConsole, {requireAdmin: true});
