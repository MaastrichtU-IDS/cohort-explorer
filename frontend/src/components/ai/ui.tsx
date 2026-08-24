'use client';

// Small shared UI atoms for the experimental AI chat layouts.
import React, {useEffect, useRef, useState} from 'react';
import {ChatMessage, SearchCohort, SearchRun} from '@/components/ai/chatClient';
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

function renderRich(text: string): string {
  const esc = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const withInline = esc
    // Inline code needs explicit foreground + background so it stays readable
    // regardless of the surrounding prose/theme colors.
    .replace(
      /`([^`]+)`/g,
      '<code class="px-1 py-0.5 rounded bg-base-200 border border-base-300 text-base-content text-[0.9em] font-mono">$1</code>'
    )
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    // Single-asterisk italics (bold already consumed its double asterisks).
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![\w*])/g, '$1<em>$2</em>')
    // Chart markers the model copies from the search results: 📊[cohort::var]
    // becomes a clickable icon that opens the variable's EDA overlay.
    .replace(/📊\[([^\[\]\n]+?)::([^\[\]\n]+?)\]/g, (_m, c, v) => {
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
      {btn('summary', 'Summary')}
      {btn('detailed', 'Detailed')}
    </span>
  );
}

// ---- Catalog search panel ----------------------------------------------------
// The chat's search tool ran for this turn: a dedicated display, distinct from
// the chat bubbles, showing exactly what the model was given — every matching
// cohort with its counts, a capped variable list per cohort, and the
// equivalent-by-code links across cohorts.

function SearchCohortBlock({cohort, defaultOpen}: {cohort: SearchCohort; defaultOpen: boolean}) {
  const shown = cohort.variables || [];
  return (
    <details className="rounded-lg border border-base-300 bg-base-100" open={defaultOpen}>
      <summary className="cursor-pointer select-none px-3 py-1.5 text-sm flex items-center gap-2">
        <span className="font-semibold">{cohort.cohort_id}</span>
        <span className="text-xs text-base-content/60">
          {cohort.matches} matching variable{cohort.matches === 1 ? '' : 's'}
          {(cohort.code_matches || 0) > 0 && <> ({cohort.code_matches} via standard code)</>}
          {cohort.matches > shown.length && <> · showing {shown.length}</>}
        </span>
      </summary>
      <ul className="px-3 pb-2 space-y-1">
        {shown.map(v => (
          <li key={v.var_name} className="text-xs leading-snug">
            <span className="font-mono font-semibold">{v.var_name}</span>
            {v.var_label && v.var_label.toLowerCase() !== v.var_name.toLowerCase() && <span className="text-base-content/70"> — {v.var_label}</span>}
            {(v.var_type || v.units || v.omop_domain || v.categorical) && (
              <span className="text-base-content/50"> [{[v.var_type, v.units, v.omop_domain, v.categorical ? 'categorical' : ''].filter(Boolean).join(', ')}]</span>
            )}
            {v.via_code && (
              <span className="ml-1 px-1 py-0.5 rounded bg-violet-100 border border-violet-300 text-violet-800 text-[10px]" title={v.matched_code ? `Shares the standard code ${v.matched_code}` : 'Matched via a shared standard code'}>
                same code
              </span>
            )}
            {v.has_eda && (
              <button
                type="button"
                className="ml-1 align-middle hover:scale-110 transition-transform"
                title={`Open the EDA graph of ${v.var_name} (${cohort.cohort_id})`}
                onClick={() => openEda(cohort.cohort_id, v.var_name)}
              >
                📊
              </button>
            )}
            {v.equivalents && v.equivalents.length > 0 && (
              <span className="text-violet-700"> ⇄ {v.equivalents.map(e => `${e.cohort_id}::${e.var_name}`).join(', ')}</span>
            )}
          </li>
        ))}
        {cohort.matches > shown.length && (
          <li className="text-xs text-base-content/50 italic">+{cohort.matches - shown.length} more matching variables in this cohort (see the explore page for the full list)</li>
        )}
      </ul>
    </details>
  );
}

export function SearchResultsPanel({runs}: {runs: SearchRun[]}) {
  if (!runs || runs.length === 0) return null;
  return (
    <div className="rounded-2xl border border-sky-200 bg-sky-50/60 px-4 py-3 space-y-3">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide font-semibold text-sky-900/70">
        <span aria-hidden>🔎</span> Catalog search — run by the assistant with the platform&rsquo;s search tool
      </div>
      {runs.map(run => (
        <div key={run.term} className="space-y-1.5">
          <div className="text-sm">
            <span className="px-2 py-0.5 rounded-full bg-sky-100 border border-sky-300 text-sky-900 font-mono text-xs">{run.term}</span>{' '}
            {run.total_matches === 0 ? (
              <span className="text-base-content/60">no matching variables in any cohort</span>
            ) : (
              <span className="text-base-content/80">
                <b>{run.total_matches}</b> matching variable{run.total_matches === 1 ? '' : 's'} across <b>{run.cohorts.length}</b> cohort
                {run.cohorts.length === 1 ? '' : 's'}:
              </span>
            )}
          </div>
          {run.codes && run.codes.length > 0 && (
            <div className="text-xs text-violet-800">
              includes variables matched via shared standard code{run.codes.length === 1 ? '' : 's'}: {run.codes.map(c => c.display).join('; ')}
            </div>
          )}
          {run.cohorts.length > 0 && (
            <div className="flex flex-wrap gap-1.5 text-xs">
              {run.cohorts.map(c => (
                <span key={c.cohort_id} className="px-1.5 py-0.5 rounded bg-base-100 border border-base-300" title={c.code_matches ? `${c.text_matches || 0} by text + ${c.code_matches} via standard code` : undefined}>
                  {c.cohort_id} <b>{c.matches}</b>
                  {(c.code_matches || 0) > 0 && (c.text_matches || 0) === 0 && <span className="text-violet-700"> code</span>}
                </span>
              ))}
            </div>
          )}
          <div className="space-y-1">
            {run.cohorts
              .filter(c => c.variables.length > 0)
              .map((c, i) => (
                <SearchCohortBlock key={c.cohort_id} cohort={c} defaultOpen={i === 0 && runs.length === 1} />
              ))}
            {run.cohorts.some(c => c.matches > 0 && c.variables.length === 0) && (
              <div className="text-xs text-base-content/50 italic">
                Variable details are expanded for the top cohorts only; the counts above cover every matching cohort (full lists on the explore page).
              </div>
            )}
          </div>
        </div>
      ))}
      <div className="text-[11px] text-base-content/50">
        Every matching cohort is listed; variable lists are capped per cohort — the counts show how many more there are.
      </div>
    </div>
  );
}

export function MessageBubble({message, streaming}: {message: ChatMessage; streaming?: boolean}) {
  const isUser = message.role === 'user';
  // Assistant turns carry two answer variants (short summary / in-depth);
  // default to the summary and let the user toggle.
  const [variant, setVariant] = useState<'summary' | 'detailed'>('summary');
  if (message.role === 'system') return null;

  const hasVariants = !isUser && (message.summary !== undefined || message.detailed !== undefined);
  const shown = hasVariants
    ? (variant === 'summary' ? message.summary : message.detailed) || ''
    : message.content;

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm ${
          isUser
            ? 'bg-blue-100 text-blue-900 border border-blue-200 rounded-br-sm'
            : 'bg-base-100 border border-base-300 rounded-bl-sm'
        }`}
      >
        {!isUser && (
          <div className="flex items-center gap-3 mb-1.5">
            <span className="text-[11px] uppercase tracking-wide opacity-50 font-semibold">Assistant</span>
            {hasVariants && <VariantToggle variant={variant} onChange={setVariant} />}
          </div>
        )}
        {shown ? (
          <div
            className="prose prose-sm max-w-none leading-relaxed [&_*]:my-0"
            onClick={e => {
              const t = (e.target as HTMLElement).closest?.('a.eda-open') as HTMLElement | null;
              if (t) {
                e.preventDefault();
                openEda(t.getAttribute('data-eda-cohort') || '', t.getAttribute('data-eda-var') || '');
              }
            }}
            dangerouslySetInnerHTML={{__html: renderRich(shown)}}
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
            <VariantToggle variant={variant} onChange={setVariant} />
          </div>
        )}
      </div>
    </div>
  );
}

export function MessageList({messages, streaming}: {messages: ChatMessage[]; streaming: boolean}) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({behavior: 'smooth', block: 'end'});
  }, [messages]);
  return (
    <div className="space-y-4">
      <EdaOverlayHost />
      {messages.map((m, i) => (
        <React.Fragment key={i}>
          {m.role === 'assistant' && m.searches && m.searches.length > 0 && <SearchResultsPanel runs={m.searches} />}
          {m.role === 'assistant' && m.searchError && (
            <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-1.5">
              The assistant&rsquo;s catalog search could not run ({m.searchError}); the answer relies on the basic context only.
            </div>
          )}
          <MessageBubble message={m} streaming={streaming && i === messages.length - 1} />
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
