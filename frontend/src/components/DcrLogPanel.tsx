'use client';

import React, { useState } from 'react';
import { apiUrl } from '@/utils';
import { RefreshCw, AlertTriangle } from 'react-feather';

interface DcrAuditEvent {
  timestamp: string;
  user: string;
  desc: string;
}

interface Props {
  dcrId: string;
}

/** Inline "Fetch latest events" panel that pulls the Decentriq-side audit log
 *  for a specific DCR and renders it as a small table. Used on both /dcrs and
 *  /dcrs-details to expose platform events (data provisioned, computations
 *  run, etc.) alongside our wizard activity log. */
export function DcrLogPanel({ dcrId }: Props) {
  const [events, setEvents] = useState<DcrAuditEvent[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFetchedAt, setLastFetchedAt] = useState<Date | null>(null);

  const fetchEvents = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiUrl}/dcr-log-main/${encodeURIComponent(dcrId)}`, {
        credentials: 'include',
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch DCR log: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      const parsed: DcrAuditEvent[] = Array.isArray(data)
        ? data.map((e: any) => ({
            timestamp: String(e.timestamp ?? ''),
            user: String(e.user ?? ''),
            desc: String(e.desc ?? ''),
          }))
        : [];
      // Newest first for display.
      parsed.reverse();
      setEvents(parsed);
      setLastFetchedAt(new Date());
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch DCR log');
      setEvents(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mt-3 pt-3 border-t border-base-300">
      <div className="flex flex-wrap items-center gap-2">
        <button
          className="btn btn-sm btn-outline gap-2"
          onClick={fetchEvents}
          disabled={isLoading || !dcrId}
        >
          <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
          {events === null ? 'Fetch latest events' : 'Refresh'}
        </button>
        {lastFetchedAt && (
          <span className="text-xs text-base-content/60">
            last fetched {lastFetchedAt.toLocaleTimeString()}
          </span>
        )}
      </div>

      {error && (
        <div className="alert alert-error mt-2 py-2 text-sm">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {events !== null && events.length === 0 && !error && (
        <div className="text-sm text-base-content/60 mt-2">
          No events returned for this DCR yet.
        </div>
      )}

      {events !== null && events.length > 0 && (
        <div className="mt-2 max-h-80 overflow-auto rounded border border-base-300 bg-base-200">
          <table className="table table-xs">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>User</th>
                <th>Event</th>
              </tr>
            </thead>
            <tbody>
              {events.map((evt, idx) => (
                <tr key={`${evt.timestamp}-${idx}`}>
                  <td className="whitespace-nowrap">{evt.timestamp}</td>
                  <td className="whitespace-nowrap">{evt.user}</td>
                  <td>{evt.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default DcrLogPanel;
