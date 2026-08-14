'use client';

// Small shared UI atoms for the experimental AI chat layouts.
import React, {useEffect, useRef} from 'react';
import {ChatMessage} from '@/components/ai/chatClient';
import {apiUrl} from '@/utils';

// Very small, safe markdown-ish renderer: escapes HTML then applies bold,
// inline code, and turns "- " / "* " lines into bullets. No external deps.
function renderRich(text: string): string {
  const esc = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const withInline = esc
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-base-300 text-sm">$1</code>');
  const lines = withInline.split('\n');
  let html = '';
  let inList = false;
  for (const line of lines) {
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (bullet) {
      if (!inList) {
        html += '<ul class="list-disc ml-5 my-1 space-y-0.5">';
        inList = true;
      }
      html += `<li>${bullet[1]}</li>`;
    } else {
      if (inList) {
        html += '</ul>';
        inList = false;
      }
      html += line.trim() === '' ? '<div class="h-2"></div>' : `<div>${line}</div>`;
    }
  }
  if (inList) html += '</ul>';
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

export function MessageBubble({message, streaming}: {message: ChatMessage; streaming?: boolean}) {
  const isUser = message.role === 'user';
  if (message.role === 'system') return null;
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm ${
          isUser ? 'bg-primary text-primary-content rounded-br-sm' : 'bg-base-100 border border-base-300 rounded-bl-sm'
        }`}
      >
        {!isUser && (
          <div className="text-[11px] uppercase tracking-wide opacity-50 mb-1 font-semibold">Assistant</div>
        )}
        {message.content ? (
          <div
            className="prose prose-sm max-w-none leading-relaxed [&_*]:my-0"
            dangerouslySetInnerHTML={{__html: renderRich(message.content)}}
          />
        ) : streaming ? (
          <TypingDots />
        ) : null}
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
  placeholder
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
  placeholder?: string;
}) {
  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };
  return (
    <div className="flex items-end gap-2 bg-base-100 border border-base-300 rounded-2xl p-2 shadow-sm">
      <textarea
        className="textarea textarea-ghost flex-1 resize-none focus:outline-none max-h-40 min-h-[2.75rem] text-base"
        rows={1}
        value={value}
        placeholder={placeholder || 'Ask about the cohorts…'}
        disabled={disabled}
        onChange={e => onChange(e.target.value)}
        onKeyDown={onKeyDown}
      />
      {isStreaming ? (
        <button className="btn btn-error btn-sm gap-1" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button className="btn btn-primary btn-sm gap-1" onClick={onSend} disabled={disabled || !value.trim()}>
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
