'use client';

// Temporary service notice shown as a toast at the top of every iCARE-AI page
// (rendered by AiAccessGuard). Dismissing it hides it for the rest of the
// browser session; change NOTICE_ID when the text changes so it shows again.
// Remove the component (or set NOTICE_TEXT to '') once the notice is obsolete.
import React, {useEffect, useState} from 'react';
import {Clock, X} from 'react-feather';

const NOTICE_ID = 'llm-switch-2026-10';
const NOTICE_TEXT =
  'We are switching iCARE-AI to a different language model. While this is under way, answers can take up to 3 minutes. We apologise for the inconvenience.';

export function ServiceNotice() {
  // Hidden until mounted, so a notice dismissed earlier never flashes.
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    let dismissed = false;
    try {
      dismissed = sessionStorage.getItem(`ai-notice-dismissed:${NOTICE_ID}`) === '1';
    } catch {
      /* storage unavailable: show the notice */
    }
    setVisible(!dismissed && !!NOTICE_TEXT);
  }, []);

  if (!visible) return null;

  const dismiss = () => {
    setVisible(false);
    try {
      sessionStorage.setItem(`ai-notice-dismissed:${NOTICE_ID}`, '1');
    } catch {
      /* storage unavailable: hidden for this page view only */
    }
  };

  return (
    <div className="fixed top-3 inset-x-0 z-50 flex justify-center px-4 pointer-events-none">
      <div
        role="status"
        className="pointer-events-auto flex items-start gap-3 max-w-2xl w-full rounded-xl border border-amber-300 bg-amber-50 text-amber-900 shadow-lg px-4 py-3"
      >
        <Clock size={18} className="shrink-0 mt-0.5" />
        <span className="text-sm flex-1">{NOTICE_TEXT}</span>
        <button
          type="button"
          className="btn btn-ghost btn-xs btn-square text-amber-900"
          onClick={dismiss}
          aria-label="Dismiss notice"
        >
          <X size={16} />
        </button>
      </div>
    </div>
  );
}
