'use client';

// Small shared UI atoms for the experimental AI chat layouts.
import React, {useEffect, useRef, useState} from 'react';
import {ChatMessage, IntersectionRow, SearchCohort, SearchConcept, SearchRun, SearchVariable} from '@/components/ai/chatClient';
import EdaOverlayHost, {openEda} from '@/components/ai/EdaOverlay';
import {apiUrl} from '@/utils';

// Very small, safe markdown-ish renderer: escapes HTML then applies headings,
// bold, italics, inline code, pipe tables, and turns "- " / "* " lines into
// bullets. No external deps.

// A line that looks like a markdown table row: |cell|cell| (leading pipe).
function isTableRow(line: string): boolean {
  return /^\s*\|.*\|\s*$/.test(line);
}

// A markdown table separator row: | --- | :---: | ---- |
function isTableSeparator(line: string): boolean {
  return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(line);
}

function splitRow(line: string): string[] {
  // Strip the outer pipes then split on the inner ones.
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map(c => c.trim());
}

// Render a block of consecutive table lines to an HTML table. Cells arrive
// already escaped + inline-formatted. Wrapped in overflow-x-auto so wide
// tables scroll inside the bubble instead of breaking the layout.
function renderTable(rows: string[]): string {
  if (rows.length === 0) return '';
  const headerCells = splitRow(rows[0]);
  const hasSeparator = rows.length > 1 && isTableSeparator(rows[1]);
  const bodyRows = rows.slice(hasSeparator ? 2 : 1).filter(r => !isTableSeparator(r));

  let html = '<div class="overflow-x-auto my-2"><table class="table table-xs table-zebra w-auto min-w-[50%] border border-base-300">';
  html += '<thead><tr>';
  for (const c of headerCells) {
    html += `<th class="bg-base-200 text-base-content font-semibold whitespace-nowrap">${c}</th>`;
  }
  html += '</tr></thead><tbody>';
  for (const row of bodyRows) {
    const cells = splitRow(row);
    html += '<tr>';
    // Pad/truncate to the header width so ragged rows don't skew columns.
    for (let i = 0; i < headerCells.length; i++) {
      html += `<td class="align-top">${cells[i] ?? ''}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  return html;
}

function renderRich(text: string, validEda?: Set<string>): string {
  const esc = text
    // Models sometimes emit non-breaking/narrow spaces (U+00A0, U+202F, U+2007);
    // a long list joined by those never wraps and runs out of its bubble.
    .replace(/[\u00A0\u202F\u2007]/g, ' ')
    // literal <br> tags the model sometimes writes would otherwise render as text
    .replace(/<br\s*\/?>/gi, ' ')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  // Fenced code blocks (```sql ... ```) are lifted out BEFORE inline
  // processing, so their backticks/asterisks stay literal, and rendered as one
  // scrollable <pre> block instead of a chip per line. An unterminated fence
  // (mid-stream) swallows to the end so the block looks right while streaming.
  const codeBlocks: string[] = [];
  const liftBlock = (body: string) => {
    const idx = codeBlocks.length;
    codeBlocks.push(
      `<div class="my-2 overflow-x-auto"><pre class="bg-base-200 border border-base-300 rounded-lg px-3 py-2 text-[0.85em] font-mono leading-relaxed whitespace-pre text-base-content">${body.replace(/\n+$/, '')}</pre></div>`
    );
    return `\u0000CODE${idx}\u0000`;
  };
  const withBlocks = esc
    .replace(/```[^\S\n]*\w*[^\S\n]*\n([\s\S]*?)```/g, (_m, body) => liftBlock(body))
    .replace(/```[^\S\n]*\w*[^\S\n]*\n([\s\S]*)$/, (_m, body) => liftBlock(body));
  const withInline = withBlocks
    // Inline code needs explicit foreground + background so it stays readable
    // regardless of the surrounding prose/theme colors. Single line only:
    // pairing backticks across lines used to mangle everything between them.
    .replace(
      /`([^`\n]+)`/g,
      '<code class="px-1 py-0.5 rounded bg-base-200 border border-base-300 text-base-content text-[0.9em] font-mono">$1</code>'
    )
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    // Single-asterisk italics (bold already consumed its double asterisks).
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![\w*])/g, '$1<em>$2</em>')
    // Chart markers the model copies from the search results: 📊[cohort::var]
    // becomes a clickable icon that opens the variable's EDA overlay.
    .replace(/📊\[([^\[\]\n]+?)::([^\[\]\n]+?)\]/g, (_m, c, v) => {
      // When a validation set is given (built from the turn's search results),
      // markers the model invented for variables without an EDA are dropped.
      if (validEda && !validEda.has(`${c}::${v}`.trim().toLowerCase())) return '';
      const attr = (t: string) => t.replace(/"/g, '&quot;').trim();
      return `<a href="#" class="eda-open no-underline" data-eda-cohort="${attr(c)}" data-eda-var="${attr(v)}" title="Open the EDA graph of ${attr(v)} (${attr(c)})">📊</a>`;
    });
  const lines = withInline.split('\n');
  let html = '';
  let inList = false;
  const closeList = () => {
    if (inList) {
      html += '</ul>';
      inList = false;
    }
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Table block: two or more consecutive |...| rows (the model may or may
    // not emit a ---|--- separator row; both shapes render).
    if (isTableRow(line) && i + 1 < lines.length && (isTableRow(lines[i + 1]) || isTableSeparator(lines[i + 1]))) {
      closeList();
      const block: string[] = [];
      while (i < lines.length && (isTableRow(lines[i]) || isTableSeparator(lines[i]))) {
        block.push(lines[i]);
        i++;
      }
      i--; // the for-loop increment moves past the block's last line
      html += renderTable(block);
      continue;
    }
    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    if (heading) {
      closeList();
      html += `<div class="font-semibold mt-3 mb-1">${heading[1]}</div>`;
      continue;
    }
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (bullet) {
      if (!inList) {
        html += '<ul class="list-disc ml-5 my-1 space-y-0.5">';
        inList = true;
      }
      html += `<li>${bullet[1]}</li>`;
    } else {
      closeList();
      html += line.trim() === '' ? '<div class="h-2"></div>' : `<div>${line}</div>`;
    }
  }
  closeList();
  // Put the lifted code blocks back in place of their placeholders.
  html = html.replace(/\u0000CODE(\d+)\u0000/g, (_m, i) => codeBlocks[Number(i)] || '');
  return html;
}

// Rendered markdown-ish text, for use outside chat bubbles (reports, panels).
export function RichText({text, className}: {text: string; className?: string}) {
  return (
    <div
      className={className || 'prose prose-sm max-w-none leading-relaxed [&_*]:my-0'}
      dangerouslySetInnerHTML={{__html: renderRich(text)}}
    />
  );
}

export function TypingDots() {
  return (
    <span className="inline-flex gap-1 items-center">
      <span className="w-2 h-2 rounded-full bg-current opacity-60 animate-bounce" style={{animationDelay: '0ms'}} />
      <span className="w-2 h-2 rounded-full bg-current opacity-60 animate-bounce" style={{animationDelay: '150ms'}} />
      <span className="w-2 h-2 rounded-full bg-current opacity-60 animate-bounce" style={{animationDelay: '300ms'}} />
    </span>
  );
}

function VariantToggle({
  variant,
  onChange
}: {
  variant: 'summary' | 'detailed';
  onChange: (v: 'summary' | 'detailed') => void;
}) {
  const btn = (v: 'summary' | 'detailed', label: string) => (
    <button
      onClick={() => onChange(v)}
      className={`px-4 py-1 rounded-full text-base font-semibold border transition-all ${
        variant === v
          ? 'bg-blue-100 text-blue-900 border-blue-300'
          : 'bg-base-100 text-base-content/50 border-base-300 hover:border-blue-300'
      }`}
    >
      {label}
    </button>
  );
  return (
    <span className="inline-flex gap-1.5">
      {btn('detailed', 'Detailed')}
      {btn('summary', 'Summary')}
    </span>
  );
}

// ---- Catalog search panel ----------------------------------------------------
// The chat's search tool ran for this turn: a dedicated display, distinct from
// the chat bubbles, showing exactly what the model was given — every matching
// cohort with its counts, a capped variable list per cohort, and the
// equivalent-by-code links across cohorts.

// One variable in a results list: name, label, [units, OMOP domain] and the
// EDA chart button. Deliberately no data type / 'categorical' flag, no
// standard-code badges and no equivalent-variable links — those stay in the
// model's context but only add noise here.
function VariableLine({cohortId, v}: {cohortId: string; v: SearchVariable}) {
  return (
    <li className="text-xs leading-snug">
      <span className="font-mono font-semibold">{v.var_name}</span>
      {v.var_label && v.var_label.toLowerCase() !== v.var_name.toLowerCase() && <span className="text-base-content/70"> — {v.var_label}</span>}
      {(v.units || v.omop_domain) && (
        <span className="text-base-content/50"> [{[v.units, v.omop_domain].filter(Boolean).join(', ')}]</span>
      )}
      {v.has_eda && (
        <button
          type="button"
          className="ml-1 align-middle hover:scale-110 transition-transform"
          title={`Open the EDA graph of ${v.var_name} (${cohortId})`}
          onClick={() => openEda(cohortId, v.var_name)}
        >
          📊
        </button>
      )}
    </li>
  );
}

// The variable list of one cohort, shown when its chip in the results row is
// clicked (up to the per-cohort cap; the chip's count tells the full number).
function CohortVariablesCard({cohort}: {cohort: SearchCohort}) {
  const shown = cohort.variables || [];
  return (
    <div className="rounded-lg border border-base-300 bg-base-100">
      <div className="px-3 py-1.5 text-sm flex items-center gap-2">
        <span className="font-semibold">{cohort.cohort_id}</span>
        <span className="text-xs text-base-content/60">
          {cohort.matches} matching variable{cohort.matches === 1 ? '' : 's'}
          {cohort.matches > shown.length && shown.length > 0 && <> · showing {shown.length}</>}
        </span>
      </div>
      {shown.length === 0 && (
        <div className="px-3 pb-2 text-xs text-base-content/50 italic">
          No variable details were stored for this cohort in this search (full lists on the explore page).
        </div>
      )}
      <ul className="px-3 pb-2 space-y-1">
        {shown.map(v => (
          <VariableLine key={v.var_name} cohortId={cohort.cohort_id} v={v} />
        ))}
        {cohort.matches > shown.length && shown.length > 0 && (
          <li className="text-xs text-base-content/50 italic">+{cohort.matches - shown.length} more matching variables in this cohort (see the explore page for the full list)</li>
        )}
      </ul>
    </div>
  );
}

// A run is an "expansion" when its term is not the FIRST term of its concept
// (the planner groups equivalent terms under one concept). The panel then
// reports only what the term ADDED beyond the concept's earlier terms,
// mirroring how the results are presented to the model.
type ExpansionInfo = {concept: string; known: number; newCohorts: SearchCohort[]};

function classifyExpansions(runs: SearchRun[], concepts?: SearchConcept[]): (ExpansionInfo | null)[] {
  const norm = (s: string) => (s || '').trim().toLowerCase();
  const runByTerm = new Map<string, SearchRun>();
  runs.forEach(r => {
    if (!runByTerm.has(norm(r.term))) runByTerm.set(norm(r.term), r);
  });
  return runs.map(r => {
    for (const c of concepts || []) {
      const terms = (c.terms || []).map(norm);
      const idx = terms.indexOf(norm(r.term));
      if (idx === 0) return null; // the concept's main term: full presentation
      if (idx > 0) {
        const seen = new Set<string>();
        for (const t of terms.slice(0, idx)) {
          runByTerm.get(t)?.cohorts.forEach(x => seen.add(x.cohort_id));
        }
        const newCohorts = r.cohorts.filter(x => !seen.has(x.cohort_id));
        return {concept: c.name || c.terms[0] || '', known: r.cohorts.length - newCohorts.length, newCohorts};
      }
    }
    return null;
  });
}

// `live` = the answer is still being written: the panel then sits ABOVE the
// answer, expanded, so the user can watch what was searched. Once the answer is
// done the panel is rendered BELOW it, collapsed to a summary rectangle with a
// button to review the full results.
// For one cohort in the intersection: its matching variables grouped by
// concept, gathered from every run of each concept's terms (deduped by name).
function gatherConceptVariables(
  runs: SearchRun[],
  concept: SearchConcept,
  cohortId: string
): SearchVariable[] {
  const terms = new Set((concept.terms || []).map(t => (t || '').trim().toLowerCase()));
  const seen = new Set<string>();
  const vars: SearchVariable[] = [];
  (runs || []).forEach(r => {
    if (!terms.has((r.term || '').trim().toLowerCase())) return;
    const c = r.cohorts.find(x => x.cohort_id === cohortId);
    (c?.variables || []).forEach(v => {
      const k = (v.var_name || '').toLowerCase();
      if (k && !seen.has(k)) {
        seen.add(k);
        vars.push(v);
      }
    });
  });
  return vars;
}

const INTERSECTION_VARS_SHOWN = 10;

function IntersectionBlock({
  concepts,
  intersection,
  runs
}: {
  concepts?: SearchConcept[];
  intersection?: IntersectionRow[] | null;
  runs: SearchRun[];
}) {
  const [openCohorts, setOpenCohorts] = useState<Record<string, boolean>>({});
  const named = (concepts || []).filter(c => c && Object.keys(c.cohorts || {}).length > 0);
  if (named.length < 2) return null;
  const labels = named.map(c => c.name || c.terms.slice(0, 2).join(' / '));
  const toggle = (id: string) => setOpenCohorts(prev => ({...prev, [id]: !prev[id]}));
  const openRows = (intersection || []).filter(row => openCohorts[row.cohort_id]);
  return (
    <div className="rounded-lg border border-emerald-300 bg-emerald-50/70 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide font-semibold text-emerald-900/70">
        Cohorts matching all criteria: {labels.join(' + ')}
      </div>
      {intersection && intersection.length > 0 ? (
        <>
          <div className="flex flex-wrap gap-1.5 mt-1.5">
            {intersection.map(row => (
              <button
                key={row.cohort_id}
                type="button"
                onClick={() => toggle(row.cohort_id)}
                className={`px-2 py-0.5 rounded border text-sm text-left transition-colors cursor-pointer ${
                  openCohorts[row.cohort_id]
                    ? 'bg-emerald-100 border-emerald-500'
                    : 'bg-base-100 border-emerald-300 hover:border-emerald-500 hover:bg-emerald-50'
                }`}
                title="Click to show this cohort's matching variables per criterion"
              >
                <b>{row.cohort_id}</b>{' '}
                <span className="text-xs text-base-content/60">
                  {Object.entries(row.per_concept).map(([k, v]) => `${k} ${v}`).join(' · ')}
                </span>
                <span className="ml-0.5 text-xs opacity-50">{openCohorts[row.cohort_id] ? '▾' : '▸'}</span>
              </button>
            ))}
          </div>
          {openRows.map(row => (
            <div key={row.cohort_id} className="mt-2 rounded-lg border border-emerald-300 bg-base-100 px-3 py-2">
              <div className="text-sm font-semibold mb-1">{row.cohort_id}</div>
              <div className="space-y-1.5">
                {named.map((c, i) => {
                  const vars = gatherConceptVariables(runs, c, row.cohort_id);
                  const total = row.per_concept[labels[i]] ?? row.per_concept[c.name] ?? vars.length;
                  return (
                    <div key={labels[i]}>
                      <div className="text-xs font-semibold text-emerald-900/80">
                        {labels[i]}{' '}
                        <span className="font-normal text-base-content/60">
                          — {total} matching variable{total === 1 ? '' : 's'}
                          {total > vars.length && vars.length > 0 && <> · showing {Math.min(vars.length, INTERSECTION_VARS_SHOWN)}</>}
                        </span>
                      </div>
                      {vars.length > 0 ? (
                        <ul className="mt-0.5 space-y-0.5">
                          {vars.slice(0, INTERSECTION_VARS_SHOWN).map(v => (
                            <VariableLine key={v.var_name} cohortId={row.cohort_id} v={v} />
                          ))}
                          {vars.length > INTERSECTION_VARS_SHOWN && (
                            <li className="text-xs text-base-content/50 italic">+{vars.length - INTERSECTION_VARS_SHOWN} more (see the explore page)</li>
                          )}
                        </ul>
                      ) : (
                        <div className="text-xs text-base-content/50 italic">
                          No variable details were stored for this cohort in this search (full lists on the explore page).
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </>
      ) : (
        <div className="text-sm text-base-content/70 mt-1">
          None - no cohort matches every criterion at once; the per-criterion matches are below.
        </div>
      )}
    </div>
  );
}

// One search term's results: the header line, the clickable cohort chips
// (click = show that cohort's top matching variables below), and — for an
// expansion term — only the cohorts it newly discovered.
function SearchRunBlock({run, expansion}: {run: SearchRun; expansion: ExpansionInfo | null}) {
  const [openCohorts, setOpenCohorts] = useState<Record<string, boolean>>({});
  const toggle = (id: string) => setOpenCohorts(prev => ({...prev, [id]: !prev[id]}));
  const chipCohorts = expansion ? expansion.newCohorts : run.cohorts;
  const openList = chipCohorts.filter(c => openCohorts[c.cohort_id]);
  return (
    <div className="space-y-1.5">
      <div className="text-sm">
        <span className="px-2 py-0.5 rounded-full bg-sky-100 border border-sky-300 text-sky-900 font-mono text-xs">{run.term}</span>
        {expansion && (
          <span className="ml-1 px-1.5 py-0.5 rounded bg-violet-100 border border-violet-300 text-violet-800 text-[10px] uppercase tracking-wide align-middle" title={`An equivalent term of the concept "${expansion.concept}": only the cohorts it newly discovered are listed`}>
            equivalent term
          </span>
        )}{' '}
        {run.total_matches === 0 ? (
          <span className="text-base-content/60">no matching variables in any cohort</span>
        ) : expansion ? (
          expansion.newCohorts.length > 0 ? (
            <span className="text-base-content/80">
              matches <b>{run.cohorts.length}</b> cohort{run.cohorts.length === 1 ? '' : 's'} — {expansion.known} already matched
              earlier terms of &ldquo;{expansion.concept}&rdquo;, <b>{expansion.newCohorts.length}</b> newly discovered:
            </span>
          ) : (
            <span className="text-base-content/80">
              matches <b>{run.cohorts.length}</b> cohort{run.cohorts.length === 1 ? '' : 's'}, all already matched by earlier terms of
              &ldquo;{expansion.concept}&rdquo; — no new cohorts.
            </span>
          )
        ) : (
          <span className="text-base-content/80">
            <b>{run.total_matches}</b> matching variable{run.total_matches === 1 ? '' : 's'} across <b>{run.cohorts.length}</b> cohort
            {run.cohorts.length === 1 ? '' : 's'}:
          </span>
        )}
      </div>
      {(() => {
        // Only the standard concept NAMES the search matched through — no raw
        // codes, no matched-via wording. (Older saved runs lack the name field:
        // it is parsed out of the "code (name)" display string.)
        const names: string[] = [];
        (run.codes || []).forEach(c => {
          const n = (c.name || c.display.match(/\(([^()]+)\)\s*$/)?.[1] || '').trim();
          if (n && !names.some(x => x.toLowerCase() === n.toLowerCase())) names.push(n);
        });
        return names.length > 0 ? (
          <div className="text-xs text-violet-800">standard concepts: {names.join('; ')}</div>
        ) : null;
      })()}
      {chipCohorts.length > 0 && (
        <div className="flex flex-wrap gap-1.5 text-xs">
          {chipCohorts.map(c => (
            <button
              key={c.cohort_id}
              type="button"
              onClick={() => toggle(c.cohort_id)}
              className={`px-1.5 py-0.5 rounded border transition-colors cursor-pointer ${
                openCohorts[c.cohort_id]
                  ? 'bg-sky-100 border-sky-400 text-sky-900'
                  : 'bg-base-100 border-base-300 hover:border-sky-400 hover:bg-sky-50'
              }`}
              title="Click to show the top matching variables"
            >
              {c.cohort_id} <b>{c.matches}</b>
              <span className="ml-0.5 opacity-50">{openCohorts[c.cohort_id] ? '▾' : '▸'}</span>
            </button>
          ))}
        </div>
      )}
      {openList.length > 0 && (
        <div className="space-y-1">
          {openList.map(c => (
            <CohortVariablesCard key={c.cohort_id} cohort={c} />
          ))}
        </div>
      )}
    </div>
  );
}

export function SearchResultsPanel({
  runs,
  concepts,
  intersection,
  live = false
}: {
  runs: SearchRun[];
  concepts?: SearchConcept[];
  intersection?: IntersectionRow[] | null;
  live?: boolean;
}) {
  const [open, setOpen] = useState(live);
  if (!runs || runs.length === 0) return null;
  const expansions = classifyExpansions(runs, concepts);
  const hasIntersection = (concepts || []).filter(c => c && Object.keys(c.cohorts || {}).length > 0).length >= 2;
  const totalMatches = runs.reduce((sum, r) => sum + (r.total_matches || 0), 0);
  const cohortIds = new Set<string>();
  runs.forEach(r => r.cohorts.forEach(c => cohortIds.add(c.cohort_id)));
  const headline = [
    `${runs.length} search${runs.length === 1 ? '' : 'es'}`,
    `${totalMatches.toLocaleString()} matching variable${totalMatches === 1 ? '' : 's'}`,
    `${cohortIds.size} cohort${cohortIds.size === 1 ? '' : 's'}`
  ].join(' · ');

  if (!open) {
    return (
      <div className="rounded-2xl border border-sky-200 bg-sky-50/60 px-4 py-3 space-y-2">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] uppercase tracking-wide font-semibold text-sky-900/70">
          <span aria-hidden>🔎</span> Catalog search
          <span className="normal-case tracking-normal text-sm font-semibold text-base-content/80">{headline}</span>
        </div>
        {hasIntersection && <IntersectionBlock concepts={concepts} intersection={intersection} runs={runs} />}
        <div className="flex flex-wrap gap-1.5 mt-2">
          {runs.map(r => (
            <span key={r.term} className="px-2 py-0.5 rounded-full bg-sky-100 border border-sky-300 text-sky-900 font-mono text-xs">
              {r.term}
            </span>
          ))}
        </div>
        <button
          className="btn btn-outline border-sky-300 text-sky-900 hover:bg-sky-100 hover:border-sky-400 gap-2 w-full mt-3"
          onClick={() => setOpen(true)}
        >
          🔎 Review the search results
        </button>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-sky-200 bg-sky-50/60 px-4 py-3 space-y-3">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide font-semibold text-sky-900/70">
        <span aria-hidden>🔎</span> Catalog search — run by the assistant with the platform&rsquo;s search tool
        <span className="normal-case tracking-normal text-xs font-normal text-base-content/60">{headline}</span>
        {!live && (
          <button className="btn btn-xs btn-ghost ml-auto" onClick={() => setOpen(false)}>
            Hide
          </button>
        )}
      </div>
      {hasIntersection && <IntersectionBlock concepts={concepts} intersection={intersection} runs={runs} />}
      {runs.map((run, i) => (
        <SearchRunBlock key={`${run.term}-${i}`} run={run} expansion={expansions[i]} />
      ))}
      <div className="flex items-center gap-3">
        <div className="text-[11px] text-base-content/50 flex-1">
          Every matching cohort is listed — click a cohort to see its top matching variables (lists are capped per cohort; the counts show the full number). Equivalent terms list only the cohorts they newly discovered.
        </div>
        {!live && (
          <button className="btn btn-sm btn-ghost" onClick={() => setOpen(false)}>
            Hide the search results
          </button>
        )}
      </div>
    </div>
  );
}

export function MessageBubble({
  message,
  streaming,
  validEda,
  onSummaryViewed
}: {
  message: ChatMessage;
  streaming?: boolean;
  validEda?: Set<string>;
  onSummaryViewed?: () => void;
}) {
  const isUser = message.role === 'user';
  // Assistant turns carry two answer variants; the in-depth one is the
  // default view and the user can switch to the short summary.
  const [variant, setVariant] = useState<'summary' | 'detailed'>('detailed');
  const pickVariant = (v: 'summary' | 'detailed') => {
    setVariant(v);
    if (v === 'summary') onSummaryViewed?.();
  };
  if (message.role === 'system') return null;

  const hasVariants = !isUser && (message.summary !== undefined || message.detailed !== undefined);
  const shown = hasVariants
    ? (variant === 'summary' ? message.summary : message.detailed) || ''
    : message.content;

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`rounded-2xl px-4 py-3 shadow-sm ${
          isUser
            ? 'max-w-[85%] bg-blue-100 text-blue-900 border border-blue-200 rounded-br-sm'
            : 'max-w-[97%] grow min-w-0 overflow-hidden bg-base-100 border border-base-300 rounded-bl-sm'
        }`}
      >
        {!isUser && (
          <div className="flex items-center gap-3 mb-1.5">
            <span className="text-[11px] uppercase tracking-wide opacity-50 font-semibold">Assistant</span>
            {message.clarify && (
              <span className="px-4 py-1 rounded-full text-base font-semibold bg-pink-100 text-pink-900 border border-pink-300">
                Request for disambiguation
              </span>
            )}
            {hasVariants && <VariantToggle variant={variant} onChange={pickVariant} />}
          </div>
        )}
        {shown ? (
          <div
            className="prose prose-sm max-w-none leading-relaxed break-words [&_*]:my-0"
            onClick={e => {
              const t = (e.target as HTMLElement).closest?.('a.eda-open') as HTMLElement | null;
              if (t) {
                e.preventDefault();
                openEda(t.getAttribute('data-eda-cohort') || '', t.getAttribute('data-eda-var') || '');
              }
            }}
            dangerouslySetInnerHTML={{__html: renderRich(shown, validEda)}}
          />
        ) : streaming ? (
          <TypingDots />
        ) : hasVariants ? (
          <span className="text-sm text-base-content/40 italic">
            No {variant} answer{streaming ? ' yet' : ' was produced'}. Try the other tab.
          </span>
        ) : null}
        {/* Long answers get a second toggle at the bottom so switching back
            doesn't require scrolling up. */}
        {hasVariants && shown.length > 700 && !streaming && (
          <div className="mt-3 pt-2 border-t border-base-200">
            <VariantToggle variant={variant} onChange={pickVariant} />
          </div>
        )}
      </div>
    </div>
  );
}

export function MessageList({
  messages,
  streaming,
  onSummaryViewed
}: {
  messages: ChatMessage[];
  streaming: boolean;
  onSummaryViewed?: (index: number) => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  // Follow the stream only while the reader is already at the bottom. Scrolling
  // up (say, to study the search panel mid-answer) sets nearBottom false and
  // the auto-scroll stops fighting; scrolling back down resumes following. A
  // NEW message (the user just sent one) always jumps to the end.
  const nearBottom = useRef(true);
  const prevCount = useRef(0);
  useEffect(() => {
    const check = () => {
      const el = endRef.current;
      if (!el) return;
      nearBottom.current = el.getBoundingClientRect().top <= (window.innerHeight || document.documentElement.clientHeight) + 160;
    };
    // capture: also catches scrolling inside nested containers
    window.addEventListener('scroll', check, {passive: true, capture: true});
    return () => window.removeEventListener('scroll', check, {capture: true});
  }, []);
  useEffect(() => {
    const isNewMessage = messages.length !== prevCount.current;
    prevCount.current = messages.length;
    if (isNewMessage || nearBottom.current) {
      // instant while following chunk-by-chunk; smooth only on a new turn
      endRef.current?.scrollIntoView({behavior: isNewMessage ? 'smooth' : 'auto', block: 'end'});
      nearBottom.current = true;
    }
  }, [messages]);
  // Chart markers are only trusted for variables the searches actually flagged
  // with an EDA; anything else the model wrote is dropped at render time.
  const validEda = new Set<string>();
  messages.forEach(m =>
    (m.searches || []).forEach(run =>
      run.cohorts.forEach(c =>
        c.variables.forEach(v => {
          if (v.has_eda) validEda.add(`${c.cohort_id}::${v.var_name}`.toLowerCase());
        })
      )
    )
  );
  return (
    <div className="space-y-4">
      <EdaOverlayHost />
      {messages.map((m, i) => (
        <React.Fragment key={i}>
          {m.role === 'assistant' && m.searches && m.searches.length > 0 && streaming && i === messages.length - 1 && (
            <SearchResultsPanel runs={m.searches} concepts={m.searchConcepts} intersection={m.searchIntersection} live />
          )}
          {m.role === 'assistant' && m.searchError && (
            <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-1.5">
              The assistant&rsquo;s catalog search could not run ({m.searchError}); the answer relies on the basic context only.
            </div>
          )}
          <MessageBubble
            message={m}
            streaming={streaming && i === messages.length - 1}
            validEda={validEda}
            onSummaryViewed={onSummaryViewed ? () => onSummaryViewed(i) : undefined}
          />
          {m.role === 'assistant' && m.searches && m.searches.length > 0 && !(streaming && i === messages.length - 1) && (
            <SearchResultsPanel runs={m.searches} concepts={m.searchConcepts} intersection={m.searchIntersection} />
          )}
        </React.Fragment>
      ))}
      <div ref={endRef} />
    </div>
  );
}

export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  isStreaming,
  disabled,
  placeholder,
  large
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
  placeholder?: string;
  // Roomier variant used on the chat landing page.
  large?: boolean;
}) {
  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };
  return (
    <div className={`flex items-end gap-2 bg-base-100 border border-base-300 rounded-2xl shadow-sm ${large ? 'p-3' : 'p-2'}`}>
      <textarea
        className={`textarea textarea-ghost flex-1 resize-none focus:outline-none ${
          large ? 'max-h-60 min-h-[4.5rem] text-lg' : 'max-h-40 min-h-[2.75rem] text-base'
        }`}
        rows={large ? 2 : 1}
        value={value}
        placeholder={placeholder || 'Ask about the studies…'}
        disabled={disabled}
        onChange={e => onChange(e.target.value)}
        onKeyDown={onKeyDown}
      />
      {isStreaming ? (
        <button className={`btn btn-error gap-1 ${large ? '' : 'btn-sm'}`} onClick={onStop}>
          Stop
        </button>
      ) : (
        <button
          className={`btn gap-1 bg-blue-100 text-blue-900 hover:bg-blue-200 border-blue-300 ${large ? '' : 'btn-sm'}`}
          onClick={onSend}
          disabled={disabled || !value.trim()}
        >
          Send
        </button>
      )}
    </div>
  );
}

export function DisabledNotice() {
  return (
    <div className="alert bg-warning/20 border border-warning/40 text-base-content">
      <div>
        <div className="font-semibold">AI chat is not configured</div>
        <div className="text-sm opacity-80">
          Set <code className="px-1 bg-base-300 rounded">LITELLM_BASE_URL</code> (and{' '}
          <code className="px-1 bg-base-300 rounded">LITELLM_API_KEY</code>) in the backend{' '}
          <code className="px-1 bg-base-300 rounded">.env</code> to enable the local model.
        </div>
      </div>
    </div>
  );
}

export function LoginNotice() {
  return (
    <div className="alert bg-info/20 border border-info/40 text-base-content">
      <div>
        <div className="font-semibold">Please log in</div>
        <div className="text-sm opacity-80">
          You need to be signed in to talk to the model.{' '}
          <a className="link" href={`${apiUrl}/login`}>
            Log in
          </a>
          .
        </div>
      </div>
    </div>
  );
}

export function ExperimentBadge() {
  return (
    <span className="badge badge-sm bg-purple-100 text-purple-800 border border-purple-200">experimental</span>
  );
}

// Privacy note shown wherever the model is referenced.
export function LocalModelNote({className}: {className?: string}) {
  return (
    <span className={className || 'text-[11px] text-base-content/40'}>
      Local LLM running exclusively on Maastricht University servers
    </span>
  );
}
