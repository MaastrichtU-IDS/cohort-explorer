'use client';

// /dcr-settings — one admin toggle: whether "Create analysis DCR" opens the
// Flexible / No-code chooser first (on) or the traditional wizard directly (off).
import React, {useEffect, useState} from 'react';
import {AlertTriangle, Shield, Sliders} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {apiUrl} from '@/utils';

export default function DcrSettingsPage() {
  const {userEmail} = useCohorts();
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [enabled, setEnabled] = useState<boolean>(true);
  const [toggling, setToggling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!userEmail) return;
    fetch(`${apiUrl}/admin/check`, {credentials: 'include'})
      .then(res => (res.ok ? res.json() : {is_admin: false}))
      .then(data => {
        setIsAdmin(!!data.is_admin);
        if (data.is_admin) return fetch(`${apiUrl}/admin/settings`, {credentials: 'include'}).then(r => (r.ok ? r.json() : null));
        return null;
      })
      .then(settings => {
        if (settings) setEnabled(settings.dcr_chooser_enabled !== false);
      })
      .catch(() => setIsAdmin(false));
  }, [userEmail]);

  const toggle = async () => {
    setToggling(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/admin/toggle-dcr-chooser`, {method: 'POST', credentials: 'include'});
      if (!res.ok) throw new Error(`Could not save the setting (${res.status})`);
      const data = await res.json();
      setEnabled(!!data.dcr_chooser_enabled);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setToggling(false);
    }
  };

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
  if (isAdmin === null) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <span className="loading loading-spinner loading-lg"></span>
      </div>
    );
  }
  if (!isAdmin) {
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
        <Sliders size={28} />
        <h1 className="text-2xl font-bold">DCR Settings</h1>
      </div>
      {error && (
        <div className="alert alert-error mb-6">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}
      <div className="card bg-base-200 shadow-md">
        <div className="card-body">
          <h2 className="card-title text-lg">Analysis DCR entry screen</h2>
          <p className="text-sm text-base-content/70 mb-4">
            When on, &ldquo;Create analysis DCR&rdquo; first asks whether to build a <b>Flexible DCR</b> (write your own Python/R) or a{' '}
            <b>No-code DCR</b> (point-and-click, code pre-built). When off, the traditional wizard opens directly and the no-code path is not offered.
          </p>
          <div className="form-control">
            <label className="label cursor-pointer justify-start gap-4">
              <input type="checkbox" className={`toggle toggle-primary toggle-lg ${toggling ? 'opacity-50' : ''}`} checked={enabled} onChange={toggle} disabled={toggling} />
              <div>
                <span className="label-text text-base font-medium">Show the Flexible / No-code chooser</span>
                <p className="text-xs text-base-content/50 mt-1">{enabled ? 'Enabled — users choose between the two kinds of DCR' : 'Disabled — the traditional wizard is the default'}</p>
              </div>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}
