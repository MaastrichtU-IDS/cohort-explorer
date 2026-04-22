'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { apiUrl } from '@/utils';
import { AlertTriangle, CheckCircle, XCircle, Clock, Users, Database, FileText } from 'react-feather';

// ---------- Types ------------------------------------------------------------

interface DcrEvent {
  ts: string;
  event: string;
  user_email: string | null;
  session_id: string | null;
  step?: number | null;
  step_name?: string | null;
  time_on_step_seconds?: number | null;
  dcr_name?: string | null;
  dcr_id?: string | null;
  dcr_url?: string | null;
  dcr_title?: string | null;
  cohorts?: string[];
  research_question?: string | null;
  duration_ms?: number | null;
  error_type?: string | null;
  error_message?: string | null;
  [key: string]: any;
}

type Outcome = 'completed' | 'abandoned' | 'failed' | 'in_progress';

interface DcrSession {
  session_id: string;
  user_email: string | null;
  started_at: string;
  last_event_at: string;
  dcr_name: string | null;
  dcr_title: string | null;
  dcr_url: string | null;
  cohorts: string[];
  research_question: string | null;
  outcome: Outcome;
  error_type: string | null;
  error_message: string | null;
  // Aggregated seconds spent on each wizard step (keyed by step_name).
  time_per_step: Record<string, number>;
  total_seconds: number;
  // Full list of events for optional detail view.
  events: DcrEvent[];
}

// ---------- Helpers ----------------------------------------------------------

const OUTCOME_LABEL: Record<Outcome, string> = {
  completed: 'Completed',
  abandoned: 'Abandoned',
  failed: 'Failed',
  in_progress: 'In progress',
};

const OUTCOME_BADGE: Record<Outcome, string> = {
  completed: 'badge-success',
  abandoned: 'badge-warning',
  failed: 'badge-error',
  in_progress: 'badge-info',
};

function pickFirst<T>(values: (T | null | undefined)[]): T | null {
  for (const v of values) {
    if (v !== null && v !== undefined && (typeof v !== 'string' || v.length > 0)) {
      return v as T;
    }
  }
  return null;
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function groupEventsBySession(events: DcrEvent[]): DcrSession[] {
  // Events arrive newest-first from the backend; normalize to oldest-first
  // for chronological aggregation, then re-sort sessions by start time desc.
  const byId = new Map<string, DcrEvent[]>();
  const NO_SESSION = '__no_session__';

  for (const evt of events) {
    const key = evt.session_id || NO_SESSION;
    const bucket = byId.get(key);
    if (bucket) {
      bucket.push(evt);
    } else {
      byId.set(key, [evt]);
    }
  }

  const sessions: DcrSession[] = [];
  byId.forEach((rawList, sessionKey) => {
    const list = [...rawList].sort((a, b) => a.ts.localeCompare(b.ts));
    const eventNames = new Set(list.map((e) => e.event));

    let outcome: Outcome;
    if (eventNames.has('dcr_publish_succeeded') || eventNames.has('dcr_download_config_clicked')) {
      outcome = 'completed';
    } else if (eventNames.has('dcr_publish_failed')) {
      outcome = 'failed';
    } else if (eventNames.has('wizard_abandoned')) {
      outcome = 'abandoned';
    } else if (eventNames.has('wizard_closed')) {
      // Closed without completing (e.g. closed before publish). Treat as abandoned.
      outcome = 'abandoned';
    } else {
      outcome = 'in_progress';
    }

    const cohortCandidates = list
      .map((e) => (Array.isArray(e.cohorts) ? e.cohorts : null))
      .filter((x): x is string[] => Array.isArray(x) && x.length > 0);
    const cohorts = cohortCandidates.length > 0 ? cohortCandidates[cohortCandidates.length - 1] : [];

    const timePerStep: Record<string, number> = {};
    let totalSeconds = 0;
    for (const evt of list) {
      if (evt.event === 'wizard_step_left' && typeof evt.time_on_step_seconds === 'number') {
        const key = evt.step_name || `step_${evt.step ?? '?'}`;
        timePerStep[key] = (timePerStep[key] || 0) + evt.time_on_step_seconds;
        totalSeconds += evt.time_on_step_seconds;
      }
      if (
        (evt.event === 'wizard_closed' || evt.event === 'wizard_abandoned') &&
        typeof evt.time_on_step_seconds === 'number'
      ) {
        const key = evt.step_name || `step_${evt.step ?? '?'}`;
        timePerStep[key] = (timePerStep[key] || 0) + evt.time_on_step_seconds;
        totalSeconds += evt.time_on_step_seconds;
      }
    }

    const failedEvt = list.find((e) => e.event === 'dcr_publish_failed');

    sessions.push({
      session_id: sessionKey === NO_SESSION ? '' : sessionKey,
      user_email: pickFirst(list.map((e) => e.user_email)),
      started_at: list[0].ts,
      last_event_at: list[list.length - 1].ts,
      dcr_name: pickFirst(list.map((e) => e.dcr_name ?? null)),
      dcr_title: pickFirst(list.map((e) => e.dcr_title ?? null)),
      dcr_url: pickFirst(list.map((e) => e.dcr_url ?? null)),
      cohorts,
      research_question: pickFirst(list.map((e) => e.research_question ?? null)),
      outcome,
      error_type: failedEvt?.error_type ?? null,
      error_message: failedEvt?.error_message ?? null,
      time_per_step: timePerStep,
      total_seconds: totalSeconds,
      events: list,
    });
  });

  sessions.sort((a, b) => b.started_at.localeCompare(a.started_at));
  return sessions;
}

// ---------- Component --------------------------------------------------------

type FilterMode = 'all' | 'completed' | 'abandoned';

export default function DcrsPage() {
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
          <h1 className="text-3xl font-bold flex items-center justify-center gap-2">
            <Database size={28} /> DCR Activity Log
          </h1>
          <p className="text-sm text-base-content/70 mt-1">
            Every Data Clean Room wizard session — completed, abandoned, and in progress.
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
            label="Completed"
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
      </div>
    </div>
  );
}
