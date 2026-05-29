'use client';

import React, {useEffect, useState} from 'react';
import {Settings, Shield, AlertTriangle} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {apiUrl} from '@/utils';

export default function AdminSettingsPage() {
  const {userEmail} = useCohorts();
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);
  const [timechfTesting, setTimechfTesting] = useState(false);
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

  // Loading state
  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <span className="loading loading-spinner loading-lg"></span>
      </div>
    );
  }

  // Not authenticated
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
    </div>
  );
}
