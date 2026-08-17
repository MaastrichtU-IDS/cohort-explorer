'use client';

// Small shared UI atoms for the experimental AI chat layouts.
import React, {useEffect, useRef, useState} from 'react';
import {ChatMessage} from '@/components/ai/chatClient';
import {apiUrl} from '@/utils';

// Very small, safe markdown-ish renderer: escapes HTML then applies headings,
// bold, italics, inline code, and turns "- " / "* " lines into bullets.
// No external deps.
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
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![\w*])/g, '$1<em>$2</em>');
  const lines = withInline.split('\n');
  let html = '';
  let inList = false;
  const closeList = () => {
    if (inList) {
      html += '</ul>';
      inList = false;
    }
  };
  for (const line of lines) {
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
            dangerouslySetInnerHTML={{__html: renderRich(shown)}}
          />
        ) : streaming ? (
          <TypingDots />
        ) : hasVariants ? (
          <span className="text-sm text-base-content/40 italic">
            No {variant} answer{streaming ? ' yet' : ' was produced'} — try the other tab.
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
      {messages.map((m, i) => (
        <MessageBubble key={i} message={m} streaming={streaming && i === messages.length - 1} />
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
