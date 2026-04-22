'use client';

import React, { useEffect, useState } from 'react';
import { apiUrl } from '@/utils';
import { AlertTriangle, CheckCircle, Clock, Users } from 'react-feather';
import {
  DcrEvent,
  DcrSession,
  formatTimestamp,
  groupEventsBySession,
} from '@/components/dcrSessions';
import { DcrLogPanel } from '@/components/DcrLogPanel';

export default function DcrsPage() {
  const [sessions, setSessions] = useState<DcrSession[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchEvents = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${apiUrl}/dcr-events/successful`, {
          credentials: 'include',
        });
        if (response.status === 401 || response.status === 403) {
          throw new Error('You must be signed in to view this page.');
        }
        if (!response.ok) {
          throw new Error(`Failed to fetch events: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        const events: DcrEvent[] = Array.isArray(data?.events) ? data.events : [];
        // Backend already filters to successful sessions; still run through the
        // grouper to consolidate events by session_id.
        setSessions(groupEventsBySession(events).filter((s) => s.outcome === 'completed'));
      } catch (err: any) {
        setError(err?.message || 'Failed to load DCRs');
        setSessions([]);
      } finally {
        setIsLoading(false);
      }
    };
    fetchEvents();
  }, []);

  return (
    <main className="flex flex-col items-center justify-start p-6 min-h-screen bg-base-200">
      <div className="w-full max-w-5xl space-y-6">
        <header className="text-center">
          <h1 className="text-3xl font-bold">Data Clean Rooms</h1>
          <p className="text-sm text-base-content/70 mt-1">
            All successfully created Data Clean Rooms.
          </p>
        </header>

        {isLoading && (
          <div className="flex justify-center py-16">
            <span className="loading loading-spinner loading-lg"></span>
          </div>
        )}

        {error && !isLoading && (
          <div className="alert alert-error">
            <AlertTriangle size={20} />
            <span>{error}</span>
          </div>
        )}

        {!isLoading && !error && sessions.length === 0 && (
          <div className="text-center text-base-content/60 py-16">
            No Data Clean Rooms have been successfully created yet.
          </div>
        )}

        {!isLoading && !error && sessions.length > 0 && (
          <div className="space-y-3">
            {sessions.map((session) => (
              <SimpleSessionCard
                key={session.session_id || `${session.started_at}-${session.user_email}`}
                session={session}
              />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

// ---------- Subcomponents ----------------------------------------------------

function SimpleSessionCard({ session }: { session: DcrSession }) {
  const hasShuffled = session.include_shuffled_samples !== null && session.include_shuffled_samples !== undefined;
  const hasAirlock = session.airlock_settings && Object.keys(session.airlock_settings).length > 0;

  return (
    <div className="card bg-base-100 shadow-sm border border-base-300">
      <div className="card-body p-4">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <span className="badge badge-success font-semibold gap-1">
            <CheckCircle size={12} /> Successfully created
          </span>
          <h2 className="font-semibold text-lg">
            {session.dcr_title || session.dcr_name || (
              <span className="text-base-content/50">Untitled DCR</span>
            )}
          </h2>
        </div>

        <div className="flex flex-wrap gap-4 text-sm text-base-content/70">
          <span className="flex items-center gap-1">
            <Users size={14} /> {session.user_email || 'unknown'}
          </span>
          <span className="flex items-center gap-1">
            <Clock size={14} /> {formatTimestamp(session.started_at)}
          </span>
        </div>

        <div className="mt-3 text-sm space-y-2">
          {session.cohorts.length > 0 && (
            <div>
              <span className="font-semibold">Cohorts:</span>{' '}
              <span className="text-base-content/80">{session.cohorts.join(', ')}</span>
            </div>
          )}
          {session.research_question && (
            <div>
              <span className="font-semibold">Research question:</span>{' '}
              <span className="text-base-content/80">{session.research_question}</span>
            </div>
          )}
          {(hasShuffled || hasAirlock) && (
            <div>
              <span className="font-semibold">Data sample choices:</span>
              <ul className="list-disc ml-5 mt-1 text-base-content/80">
                {hasShuffled && (
                  <li>
                    Shuffled samples:{' '}
                    {typeof session.include_shuffled_samples === 'boolean'
                      ? session.include_shuffled_samples
                        ? 'enabled for all cohorts'
                        : 'disabled for all cohorts'
                      : Object.entries(
                          session.include_shuffled_samples as Record<string, boolean>
                        )
                          .map(([cohort, enabled]) => `${cohort}: ${enabled ? 'yes' : 'no'}`)
                          .join(' · ')}
                  </li>
                )}
                {hasAirlock && (
                  <li>
                    Airlock:{' '}
                    {Object.entries(session.airlock_settings!)
                      .map(([cohort, value]) => {
                        const label =
                          typeof value === 'boolean' ? (value ? 'on' : 'off') : `quota ${value}`;
                        return `${cohort}: ${label}`;
                      })
                      .join(' · ')}
                  </li>
                )}
              </ul>
            </div>
          )}
          {session.dcr_url && (
            <div>
              <span className="font-semibold">DCR URL:</span>{' '}
              <a
                href={session.dcr_url}
                target="_blank"
                rel="noopener noreferrer"
                className="link link-primary break-all"
              >
                {session.dcr_url}
              </a>
            </div>
          )}
        </div>

        {session.dcr_id && <DcrLogPanel dcrId={session.dcr_id} />}
      </div>
    </div>
  );
}
