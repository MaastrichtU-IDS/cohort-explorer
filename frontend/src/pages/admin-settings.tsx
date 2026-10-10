'use client';

import React, {useEffect, useState} from 'react';
import Link from 'next/link';
import {Settings, Shield, AlertTriangle, ArrowRight} from 'react-feather';
import {SparklesIcon} from '@/components/Icons';
import {useCohorts} from '@/components/CohortsContext';
import {apiUrl} from '@/utils';

export default function AdminSettingsPage() {
  const {userEmail} = useCohorts();
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);
  const [timechfTesting, setTimechfTesting] = useState(false);
  const [aiNavEnabled, setAiNavEnabled] = useState(false);
  const [togglingAiNav, setTogglingAiNav] = useState(false);
  const [codeExplain, setCodeExplain] = useState(true);
  const [codeExplainStopped, setCodeExplainStopped] = useState(true);
  const [togglingCodeExplain, setTogglingCodeExplain] = useState(false);
  const [toggling, setToggling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Check if the user is an admin
  useEffect(() => {
    if (!userEmail) return;

    fetch(`${apiUrl}/admin/check`, {credentials: 'include'})
      .then(res => {
        if (!res.ok) throw new Error('Not authenticated');
        return res.json();
      })
      .then(data => {
        setIsAdmin(data.is_admin);
        if (data.is_admin) {
          // Fetch current settings
          return fetch(`${apiUrl}/admin/settings`, {credentials: 'include'});
        }
      })
      .then(res => {
        if (!res) return;
        if (!res.ok) throw new Error('Failed to load admin settings');
        return res.json();
      })
      .then(data => {
        if (data) {
          setTimechfTesting(data.timechf_testing_enabled);
          setAiNavEnabled(!!data.ai_nav_enabled);
          setCodeExplain(!!data.code_explain_enabled);
          setCodeExplainStopped(!!data.code_explain_stopped_enabled);
        }
      })
      .catch(err => {
        console.error('Admin check failed:', err);
        setError(err.message);
      })
      .finally(() => setLoading(false));
  }, [userEmail]);

  const handleToggle = async () => {
    setToggling(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/admin/toggle-timechf-testing`, {
        method: 'POST',
        credentials: 'include',
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || 'Toggle failed');
      }
      const data = await res.json();
      setTimechfTesting(data.timechf_testing_enabled);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setToggling(false);
    }
  };

  const handleToggleAiNav = async () => {
    setTogglingAiNav(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/admin/toggle-ai-nav`, {
        method: 'POST',
        credentials: 'include',
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || 'Toggle failed');
      }
      const data = await res.json();
      setAiNavEnabled(!!data.ai_nav_enabled);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setTogglingAiNav(false);
    }
  };

  const handleToggleCodeExplain = async (which: 'code-explain' | 'code-explain-stopped') => {
    setTogglingCodeExplain(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/admin/toggle-${which}`, {method: 'POST', credentials: 'include'});
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || 'Toggle failed');
      }
      const data = await res.json();
      setCodeExplain(!!data.code_explain_enabled);
      setCodeExplainStopped(!!data.code_explain_stopped_enabled);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setTogglingCodeExplain(false);
    }
  };

  // Loading state
  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <span className="loading loading-spinner loading-lg"></span>
      </div>
    );
  }

  // Not authenticated
  // userEmail is '' while the session is still being verified (see
  // CohortsContext): show a spinner rather than flashing the login notice.
  if (userEmail === '') {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <span className="loading loading-spinner loading-lg"></span>
      </div>
    );
  }
  if (!userEmail) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <div className="alert alert-warning max-w-md">
          <AlertTriangle size={20} />
          <span>Please log in to access this page.</span>
        </div>
      </div>
    );
  }

  // Not an admin
  if (isAdmin === false) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <div className="alert alert-error max-w-md">
          <Shield size={20} />
          <span>Access denied. This page is restricted to administrators.</span>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-2xl">
      <div className="flex items-center gap-3 mb-8">
        <Settings size={28} />
        <h1 className="text-2xl font-bold">Admin Settings</h1>
      </div>

      {error && (
        <div className="alert alert-error mb-6">
          <AlertTriangle size={16} />
          <span>{error}</span>
          <button className="btn btn-sm btn-ghost" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <div className="card bg-base-200 shadow-md">
        <div className="card-body">
          <h2 className="card-title text-lg">TIME-CHF Testing</h2>
          <p className="text-sm text-base-content/70 mb-4">
            Allow using the TIME-CHF cohort for testing purposes.
          </p>

          <div className="form-control">
            <label className="label cursor-pointer justify-start gap-4">
              <input
                type="checkbox"
                className={`toggle toggle-primary toggle-lg ${toggling ? 'opacity-50' : ''}`}
                checked={timechfTesting}
                onChange={handleToggle}
                disabled={toggling}
              />
              <div>
                <span className="label-text text-base font-medium">
                  Use TIME-CHF in testing capacity
                </span>
                <p className="text-xs text-base-content/50 mt-1">
                  {timechfTesting
                    ? 'Enabled — TIME-CHF is available for testing'
                    : 'Disabled — TIME-CHF testing is off'}
                </p>
              </div>
              {toggling && <span className="loading loading-spinner loading-sm ml-2"></span>}
            </label>
          </div>
        </div>
      </div>

      {/* AI code explanation (My DCRs) */}
      <div className="card bg-base-200 shadow-md mt-6">
        <div className="card-body">
          <h2 className="card-title text-lg flex items-center gap-2">
            <SparklesIcon size={18} /> AI - Explain the Code
          </h2>
          <p className="text-sm text-base-content/70 mb-4">
            The button on the My DCRs page that has the local model explain a DCR&apos;s compute nodes. Turning it off also
            blocks the underlying requests.
          </p>

          <div className="form-control">
            <label className="label cursor-pointer justify-start gap-4">
              <input
                type="checkbox"
                className={`toggle toggle-primary toggle-lg ${togglingCodeExplain ? 'opacity-50' : ''}`}
                checked={codeExplain}
                onChange={() => handleToggleCodeExplain('code-explain')}
                disabled={togglingCodeExplain}
              />
              <div>
                <span className="label-text text-base font-medium">Offer the AI code explanation on My DCRs</span>
                <p className="text-xs text-base-content/50 mt-1">
                  {codeExplain ? 'Enabled.' : 'Disabled. The button is hidden for everyone.'}
                </p>
              </div>
            </label>
          </div>

          <div className={`form-control ml-10 ${codeExplain ? '' : 'opacity-50'}`}>
            <label className="label cursor-pointer justify-start gap-4">
              <input
                type="checkbox"
                className="toggle toggle-primary"
                checked={codeExplainStopped}
                onChange={() => handleToggleCodeExplain('code-explain-stopped')}
                disabled={togglingCodeExplain || !codeExplain}
              />
              <div>
                <span className="label-text font-medium">Also offer it for stopped DCRs</span>
                <p className="text-xs text-base-content/50 mt-1">
                  {codeExplainStopped
                    ? 'Enabled. The button shows on stopped (deactivated) DCRs too.'
                    : 'Disabled. The button is hidden on stopped DCRs.'}
                  {!codeExplain && ' Has no effect while the feature above is off.'}
                </p>
              </div>
              {togglingCodeExplain && <span className="loading loading-spinner loading-sm ml-2"></span>}
            </label>
          </div>
        </div>
      </div>

      {/* iCARE-AI nav toggle */}
      <div className="card bg-base-200 shadow-md mt-6">
        <div className="card-body">
          <h2 className="card-title text-lg flex items-center gap-2">
            <SparklesIcon size={18} /> iCARE-AI
          </h2>
          <p className="text-sm text-base-content/70 mb-4">
            Show or hide the iCARE-AI button in the navigation bar for all users.
          </p>

          <div className="form-control">
            <label className="label cursor-pointer justify-start gap-4">
              <input
                type="checkbox"
                className={`toggle toggle-primary toggle-lg ${togglingAiNav ? 'opacity-50' : ''}`}
                checked={aiNavEnabled}
                onChange={handleToggleAiNav}
                disabled={togglingAiNav}
              />
              <div>
                <span className="label-text text-base font-medium">Show iCARE-AI in the navigation bar</span>
                <p className="text-xs text-base-content/50 mt-1">
                  {aiNavEnabled
                    ? 'Enabled. The iCARE-AI button is visible to users.'
                    : 'Disabled. The iCARE-AI button is hidden.'}
                </p>
              </div>
              {togglingAiNav && <span className="loading loading-spinner loading-sm ml-2"></span>}
            </label>
          </div>

          <div className="divider my-2"></div>

          <Link
            href="/ai/starters"
            className="flex items-center justify-between gap-3 rounded-lg border border-base-300 bg-base-100 px-4 py-3 hover:border-primary transition-colors"
          >
            <div>
              <div className="font-medium">Manage conversation starters</div>
              <p className="text-xs text-base-content/60 mt-0.5">
                Add, delete, and generate the example questions shown on the iCARE-AI chat page, and
                regroup them into keyword themes.
              </p>
            </div>
            <ArrowRight size={18} className="shrink-0 text-primary" />
          </Link>
        </div>
      </div>
    </div>
  );
}
