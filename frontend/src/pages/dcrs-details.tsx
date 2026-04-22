'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { apiUrl } from '@/utils';
import { AlertTriangle, CheckCircle, XCircle, Clock, Users, FileText } from 'react-feather';
import {
  DcrEvent,
  DcrSession,
  OUTCOME_BADGE,
  OUTCOME_LABEL,
  formatTimestamp,
  groupEventsBySession,
} from '@/components/dcrSessions';
import { DcrLogPanel } from '@/components/DcrLogPanel';

type FilterMode = 'all' | 'completed' | 'abandoned';

export default function DcrsDetailsPage() {
  const [sessions, setSessions] = useState<DcrSession[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterMode>('all');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    const fetchEvents = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${apiUrl}/dcr-events`, { credentials: 'include' });
        if (response.status === 401 || response.status === 403) {
          throw new Error('You must be signed in as an administrator to view this page.');
        }
        if (!response.ok) {
          throw new Error(`Failed to fetch events: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        const events: DcrEvent[] = Array.isArray(data?.events) ? data.events : [];
        setSessions(groupEventsBySession(events));
      } catch (err: any) {
        setError(err?.message || 'Failed to load DCR events');
        setSessions([]);
      } finally {
        setIsLoading(false);
      }
    };
    fetchEvents();
  }, []);

  const counts = useMemo(() => {
    const c = { all: sessions.length, completed: 0, abandoned: 0, failed: 0, in_progress: 0 };
    for (const s of sessions) {
      c[s.outcome] += 1;
    }
    return c;
  }, [sessions]);

  const visibleSessions = useMemo(() => {
    if (filter === 'completed') {
      return sessions.filter((s) => s.outcome === 'completed');
    }
    if (filter === 'abandoned') {
      return sessions.filter((s) => s.outcome === 'abandoned');
    }
    return sessions;
  }, [sessions, filter]);

  return (
    <main className="flex flex-col items-center justify-start p-6 min-h-screen bg-base-200">
      <div className="w-full max-w-6xl space-y-6">
        <header className="text-center">
          <h1 className="text-3xl font-bold">DCRs Full Log</h1>
          <p className="text-sm text-base-content/70 mt-1">
            Every Data Clean Room wizard session — successfully created, abandoned, and in progress.
          </p>
        </header>

        {/* Filter toggles */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <FilterButton
            active={filter === 'all'}
            label="All"
            count={counts.all}
            icon={<FileText size={20} />}
            onClick={() => setFilter('all')}
          />
          <FilterButton
            active={filter === 'completed'}
            label="Successfully created"
            count={counts.completed}
            icon={<CheckCircle size={20} />}
            onClick={() => setFilter('completed')}
            accent="success"
          />
          <FilterButton
            active={filter === 'abandoned'}
            label="Abandoned"
            count={counts.abandoned}
            icon={<XCircle size={20} />}
            onClick={() => setFilter('abandoned')}
            accent="warning"
          />
        </div>

        {/* Body */}
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

        {!isLoading && !error && visibleSessions.length === 0 && (
          <div className="text-center text-base-content/60 py-16">
            No DCR sessions to display for this filter.
          </div>
        )}

        {!isLoading && !error && visibleSessions.length > 0 && (
          <div className="space-y-3">
            {visibleSessions.map((session) => (
              <SessionCard
                key={session.session_id || `${session.started_at}-${session.user_email}`}
                session={session}
                expanded={expandedId === (session.session_id || session.started_at)}
                onToggle={() =>
                  setExpandedId((prev) => {
                    const id = session.session_id || session.started_at;
                    return prev === id ? null : id;
                  })
                }
              />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

// ---------- Subcomponents ----------------------------------------------------

function FilterButton(props: {
  active: boolean;
  label: string;
  count: number;
  icon: React.ReactNode;
  accent?: 'success' | 'warning';
  onClick: () => void;
}) {
  const base =
    'btn btn-lg h-24 flex-col gap-1 normal-case w-full text-base shadow-md transition-all';
  const accentClass =
    props.accent === 'success'
      ? props.active
        ? 'btn-success'
        : 'btn-outline btn-success'
      : props.accent === 'warning'
        ? props.active
          ? 'btn-warning'
          : 'btn-outline btn-warning'
        : props.active
          ? 'btn-primary'
          : 'btn-outline';
  return (
    <button className={`${base} ${accentClass}`} onClick={props.onClick}>
      <span className="flex items-center gap-2">
        {props.icon}
        <span className="font-bold">{props.label}</span>
      </span>
      <span className="text-2xl font-extrabold">{props.count}</span>
    </button>
  );
}

function SessionCard(props: { session: DcrSession; expanded: boolean; onToggle: () => void }) {
  const { session, expanded, onToggle } = props;
  const stepEntries = Object.entries(session.time_per_step);
  const showFetchEvents = session.outcome === 'completed' && !!session.dcr_id;

  return (
    <div className="card bg-base-100 shadow-sm border border-base-300">
      <div className="card-body p-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className={`badge ${OUTCOME_BADGE[session.outcome]} font-semibold`}>
                {OUTCOME_LABEL[session.outcome]}
              </span>
              <h2 className="font-semibold text-lg truncate">
                {session.dcr_title || session.dcr_name || <span className="text-base-content/50">Untitled DCR</span>}
              </h2>
            </div>
            <div className="flex flex-wrap gap-4 text-sm text-base-content/70">
              <span className="flex items-center gap-1">
                <Users size={14} /> {session.user_email || 'unknown'}
              </span>
              <span className="flex items-center gap-1">
                <Clock size={14} /> {formatTimestamp(session.started_at)}
              </span>
              {session.total_seconds > 0 && (
                <span className="flex items-center gap-1">
                  Total wizard time: <strong>{session.total_seconds}s</strong>
                </span>
              )}
            </div>
          </div>
          <button className="btn btn-sm btn-ghost" onClick={onToggle}>
            {expanded ? 'Hide details' : 'Details'}
          </button>
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
          {session.outcome === 'failed' && session.error_message && (
            <div className="text-error">
              <span className="font-semibold">Error ({session.error_type}):</span>{' '}
              <span>{session.error_message}</span>
            </div>
          )}
        </div>

        {expanded && (
          <div className="mt-4 pt-4 border-t border-base-300 space-y-3 text-sm">
            <DataSampleChoices session={session} />
            <ParticipantsBlock session={session} />

            {stepEntries.length > 0 && (
              <div>
                <div className="font-semibold mb-1">Time on each step</div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {stepEntries.map(([name, seconds]) => (
                    <div
                      key={name}
                      className="flex justify-between bg-base-200 rounded px-2 py-1"
                    >
                      <span className="font-medium">{name}</span>
                      <span>{seconds}s</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div>
              <div className="font-semibold mb-1">Event timeline ({session.events.length})</div>
              <div className="max-h-80 overflow-auto rounded border border-base-300">
                <table className="table table-xs">
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Event</th>
                      <th>Step</th>
                      <th>Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {session.events.map((evt, idx) => (
                      <tr key={`${evt.ts}-${idx}`}>
                        <td className="whitespace-nowrap">{formatTimestamp(evt.ts)}</td>
                        <td className="font-mono text-xs">{evt.event}</td>
                        <td>{evt.step_name || (evt.step !== undefined && evt.step !== null ? `#${evt.step}` : '')}</td>
                        <td className="text-xs">
                          {typeof evt.time_on_step_seconds === 'number' && (
                            <span className="mr-2">{evt.time_on_step_seconds}s</span>
                          )}
                          {evt.duration_ms !== undefined && evt.duration_ms !== null && (
                            <span className="mr-2">{Math.round(evt.duration_ms / 1000)}s (server)</span>
                          )}
                          {evt.error_message && (
                            <span className="text-error">{evt.error_message}</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {session.session_id && (
              <div className="text-xs text-base-content/50 font-mono">
                session_id: {session.session_id}
              </div>
            )}
          </div>
        )}

        {showFetchEvents && <DcrLogPanel dcrId={session.dcr_id!} />}
      </div>
    </div>
  );
}

function DataSampleChoices({ session }: { session: DcrSession }) {
  const { airlock_settings, include_shuffled_samples } = session;
  const hasAny =
    (airlock_settings && Object.keys(airlock_settings).length > 0) ||
    include_shuffled_samples !== null;
  if (!hasAny) return null;

  const renderShuffled = () => {
    if (include_shuffled_samples === null || include_shuffled_samples === undefined) return null;
    if (typeof include_shuffled_samples === 'boolean') {
      return <span>{include_shuffled_samples ? 'enabled for all cohorts' : 'disabled for all cohorts'}</span>;
    }
    const entries = Object.entries(include_shuffled_samples);
    if (entries.length === 0) return <span>none</span>;
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 gap-1 mt-1">
        {entries.map(([cohort, enabled]) => (
          <div key={cohort} className="flex justify-between bg-base-200 rounded px-2 py-0.5">
            <span className="font-medium truncate">{cohort}</span>
            <span>{enabled ? 'yes' : 'no'}</span>
          </div>
        ))}
      </div>
    );
  };

  const renderAirlock = () => {
    if (!airlock_settings) return null;
    const entries = Object.entries(airlock_settings);
    if (entries.length === 0) return null;
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 gap-1 mt-1">
        {entries.map(([cohort, value]) => {
          const label = typeof value === 'boolean' ? (value ? 'on' : 'off') : `quota ${value}`;
          return (
            <div key={cohort} className="flex justify-between bg-base-200 rounded px-2 py-0.5">
              <span className="font-medium truncate">{cohort}</span>
              <span>{label}</span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div>
      <div className="font-semibold mb-1">Data sample choices</div>
      {include_shuffled_samples !== null && (
        <div>
          <span className="font-medium">Shuffled samples:</span>
          {renderShuffled()}
        </div>
      )}
      {airlock_settings && Object.keys(airlock_settings).length > 0 && (
        <div className="mt-2">
          <span className="font-medium">Airlock settings:</span>
          {renderAirlock()}
        </div>
      )}
    </div>
  );
}

function ParticipantsBlock({ session }: { session: DcrSession }) {
  const { participants, additional_analysts, excluded_data_owners } = session;
  if (!participants && additional_analysts.length === 0 && excluded_data_owners.length === 0) {
    return null;
  }

  const dataOwners: string[] = [];
  const analysts: string[] = [];
  if (participants) {
    for (const [email, roles] of Object.entries(participants)) {
      if (roles?.data_owner_of && roles.data_owner_of.length > 0) dataOwners.push(email);
      if (roles?.analyst_of && roles.analyst_of.length > 0) analysts.push(email);
    }
  }

  return (
    <div>
      <div className="font-semibold mb-1">Participants</div>
      {dataOwners.length > 0 && (
        <div className="text-sm">
          <span className="font-medium">Data owners ({dataOwners.length}):</span>{' '}
          <span className="text-base-content/80">{dataOwners.join(', ')}</span>
        </div>
      )}
      {analysts.length > 0 && (
        <div className="text-sm mt-1">
          <span className="font-medium">Analysts ({analysts.length}):</span>{' '}
          <span className="text-base-content/80">{analysts.join(', ')}</span>
        </div>
      )}
      {additional_analysts.length > 0 && (
        <div className="text-sm mt-1">
          <span className="font-medium">Added analysts:</span>{' '}
          <span className="text-base-content/80">{additional_analysts.join(', ')}</span>
        </div>
      )}
      {excluded_data_owners.length > 0 && (
        <div className="text-sm mt-1">
          <span className="font-medium">Excluded data owners:</span>{' '}
          <span className="text-base-content/80">{excluded_data_owners.join(', ')}</span>
        </div>
      )}
    </div>
  );
}
