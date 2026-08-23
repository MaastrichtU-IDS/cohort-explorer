'use client';

// EDA overlay for the AI chat: chart icons (📊) next to variables — in the
// assistant's answers and in the catalog-search panel — open the variable's
// EDA detail (stats + the original distribution graph) in an overlay window.
// Opening is done through a window event so the icon can live inside rendered
// HTML (the chat bubbles) as well as in React components.
import React, {useEffect, useRef, useState} from 'react';
import EdaVariableDetailModal from '@/components/eda/EdaVariableDetailModal';
import {EdaData, EdaVariable, parseEdaJson} from '@/utils/edaParsing';

const OPEN_EVENT = 'icare-open-eda';

export function openEda(cohortId: string, varName: string) {
  if (!cohortId || !varName) return;
  window.dispatchEvent(new CustomEvent(OPEN_EVENT, {detail: {cohortId, varName}}));
}

type OverlayState = {
  cohortId: string;
  varName: string;
  status: 'loading' | 'error' | 'ready';
  variable?: EdaVariable;
  error?: string;
};

export default function EdaOverlayHost() {
  const [state, setState] = useState<OverlayState | null>(null);
  // One fetch per cohort per page: null = fetched but unavailable.
  const cache = useRef<Record<string, EdaData | null>>({});

  useEffect(() => {
    const onOpen = async (e: Event) => {
      const {cohortId, varName} = (e as CustomEvent).detail || {};
      if (!cohortId || !varName) return;
      setState({cohortId, varName, status: 'loading'});
      try {
        let data = cache.current[cohortId];
        if (data === undefined) {
          const res = await fetch(`/api/cohort-eda-output/${encodeURIComponent(cohortId)}`);
          data = res.ok ? parseEdaJson(await res.json()) : null;
          cache.current[cohortId] = data;
        }
        const v = data?.variables.find(x => x.name.toLowerCase().trim() === String(varName).toLowerCase().trim());
        if (v) {
          setState({cohortId, varName, status: 'ready', variable: v});
        } else {
          setState({
            cohortId,
            varName,
            status: 'error',
            error: data ? `No EDA entry found for "${varName}" in ${cohortId}.` : `No EDA output is available for ${cohortId}.`
          });
        }
      } catch (err: any) {
        setState({cohortId, varName, status: 'error', error: err?.message || 'Could not load the EDA output.'});
      }
    };
    window.addEventListener(OPEN_EVENT, onOpen as EventListener);
    return () => window.removeEventListener(OPEN_EVENT, onOpen as EventListener);
  }, []);

  if (!state) return null;
  if (state.status === 'ready' && state.variable) {
    return <EdaVariableDetailModal variable={state.variable} cohortId={state.cohortId} onClose={() => setState(null)} />;
  }
  // Loading / error shell with the same close semantics as the full overlay:
  // large X top right, Close button, click anywhere outside to close.
  return (
    <div className="modal modal-open" onClick={() => setState(null)}>
      <div className="modal-box max-w-md" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-start mb-3">
          <h3 className="font-bold text-lg">
            EDA: {state.varName} <span className="text-base-content/50 font-normal">({state.cohortId})</span>
          </h3>
          <button onClick={() => setState(null)} className="btn btn-circle text-xl" aria-label="Close">
            ✕
          </button>
        </div>
        {state.status === 'loading' ? (
          <div className="flex items-center gap-3 py-6">
            <span className="loading loading-spinner loading-md"></span>
            <span className="text-sm text-base-content/70">Loading the EDA output…</span>
          </div>
        ) : (
          <p className="text-sm text-base-content/80 py-2">{state.error}</p>
        )}
        <div className="modal-action">
          <button className="btn" onClick={() => setState(null)}>
            Close
          </button>
        </div>
      </div>
      <div className="modal-backdrop" onClick={() => setState(null)}></div>
    </div>
  );
}
