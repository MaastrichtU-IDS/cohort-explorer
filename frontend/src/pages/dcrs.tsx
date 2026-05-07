'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { apiUrl } from '@/utils';
import { AlertTriangle, Clock, RefreshCw, ExternalLink } from 'react-feather';
import { DcrLogPanel } from '@/components/DcrLogPanel';

/** Shape of a single DCR record returned by the /my-dcrs endpoint. */
interface DcrRecord {
  id?: string;
  title?: string;
  description?: string;
  createdAt?: string;
  owner?: { email?: string; [key: string]: any };
  participants?: { email?: string; roles?: string[]; data_owner_of?: string[]; analyst_of?: string[] }[];
  nodes?: { name?: string; type?: string; script?: string }[];
  error?: string;
  [key: string]: any;
}

export default function DcrsPage() {
  const [dcrs, setDcrs] = useState<DcrRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [lastRefreshedAt, setLastRefreshedAt] = useState<Date | null>(null);

  const fetchMyDcrs = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiUrl}/my-dcrs`, { credentials: 'include' });
      if (response.status === 401 || response.status === 403) {
        throw new Error('You must be signed in to view this page.');
      }
      if (!response.ok) {
        throw new Error(`Failed to fetch DCRs: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      const dcrs = Array.isArray(data?.dcrs) ? data.dcrs : [];
      // Sort reverse chronologically by createdAt (newest first)
      dcrs.sort((a: DcrRecord, b: DcrRecord) => {
        if (!a.createdAt) return 1;
        if (!b.createdAt) return -1;
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      });
      setDcrs(dcrs);
      setUserEmail(data?.email ?? null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load DCRs');
      setDcrs([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);
    try {
      const response = await fetch(`${apiUrl}/my-dcrs/refresh`, {
        method: 'POST',
        credentials: 'include',
      });
      if (response.status === 401 || response.status === 403) {
        throw new Error('You must be signed in to refresh.');
      }
      if (!response.ok) {
        throw new Error(`Refresh failed: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      setDcrs(Array.isArray(data?.dcrs) ? data.dcrs : []);
      setUserEmail(data?.email ?? null);
      setLastRefreshedAt(new Date());
    } catch (err: any) {
      setError(err?.message || 'Failed to refresh DCRs');
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchMyDcrs();
  }, [fetchMyDcrs]);

  return (
    <main className="flex flex-col items-center justify-start p-6 min-h-screen bg-base-200">
      <div className="w-full max-w-5xl space-y-6">
        <header className="text-center">
          <h1 className="text-3xl font-bold">My Data Clean Rooms</h1>
          <p className="text-sm text-base-content/70 mt-1">
            Data Clean Rooms you participate in{userEmail ? ` (${userEmail})` : ''}.
          </p>
        </header>

        {/* Refresh button */}
        <div className="flex justify-start items-center gap-3">
          <button
            className="btn btn-sm btn-outline gap-2"
            onClick={handleRefresh}
            disabled={isRefreshing}
            title="Re-fetch DCR data from Decentriq"
          >
            <RefreshCw size={14} className={isRefreshing ? 'animate-spin' : ''} />
            {isRefreshing ? 'Refreshing...' : 'Refresh from Decentriq'}
          </button>
          {lastRefreshedAt && (
            <span className="text-xs text-base-content/60">
              last refreshed {lastRefreshedAt.toLocaleTimeString()}
            </span>
          )}
        </div>

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

        {!isLoading && !error && dcrs.length === 0 && (
          <div className="text-center text-base-content/60 py-16">
            No Data Clean Rooms found for your account.
          </div>
        )}

        {!isLoading && !error && dcrs.length > 0 && (
          <div className="space-y-3">
            {dcrs.map((dcr, idx) => (
              <DcrCard key={dcr.id || idx} dcr={dcr} />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

// ---------- Subcomponents ----------------------------------------------------

function formatTimestamp(iso?: string): string {
  if (!iso) return '';
  try {
    const date = new Date(iso);
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${day}.${month}.${year} ${hours}:${minutes}`;
  } catch {
    return iso;
  }
}

function DcrCard({ dcr }: { dcr: DcrRecord }) {
  const participantCount = dcr.participants?.length ?? 0;
  const dcrUrl = dcr.id
    ? `https://platform.decentriq.com/datarooms/p/${dcr.id}`
    : null;

  // Determine DCR type based on whether any compute node has a script starting with c3_eda
  const hasC3EdaScript = dcr.nodes?.some(
    (node) =>
      (node.type === 'PreviewComputeNodeDefinition' || node.type === 'PythonComputeNodeDefinition') &&
      node.script?.startsWith('c3_eda')
  );
  const dcrType = hasC3EdaScript ? 'Provision/EDA' : 'Analysis';
  const badgeColor = hasC3EdaScript ? 'badge-success' : 'badge-secondary';

  return (
    <div className="card bg-base-100 shadow-sm border border-base-300">
      <div className="card-body p-4">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <span className={`badge ${badgeColor} badge-sm font-semibold`}>
            {dcrType}
          </span>
          <h2 className="font-semibold text-lg">
            {dcr.title || <span className="text-base-content/50">Untitled DCR</span>}
          </h2>
        </div>

        <div className="flex flex-wrap gap-4 text-sm text-base-content/70">
          {dcr.createdAt && (
            <span className="flex items-center gap-1">
              <Clock size={14} /> {formatTimestamp(dcr.createdAt)}
            </span>
          )}
          {participantCount > 0 && (
            <span className="badge badge-ghost badge-sm">
              {participantCount} participant{participantCount !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {dcr.description && (
          <div className="mt-2 text-sm text-base-content/80">{dcr.description}</div>
        )}

        {/* Participants detail */}
        {dcr.participants && dcr.participants.length > 0 && (
          <div className="mt-3 text-sm">
            <span className="font-semibold">Participants:</span>
            <ul className="list-disc ml-5 mt-1 text-base-content/80">
              {dcr.participants.map((p, i) => {
                // Determine if participant is data owner (owns non-metadata data nodes)
                // Metadata nodes have names ending with "-metadata" or "_metadata_dictionary"
                const dataOwnerOf = p.data_owner_of || [];
                const isDataOwner = dataOwnerOf.some(
                  nodeId => !nodeId.endsWith('-metadata') && !nodeId.endsWith('_metadata_dictionary')
                );
                const role = isDataOwner ? 'data owner' : 'analyst';

                return (
                  <li key={p.email || i}>
                    {p.email || 'unknown'}
                    <span className="text-xs text-base-content/60 ml-1">
                      ({role})
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {/* Nodes summary */}
        {dcr.nodes && dcr.nodes.length > 0 && (
          <div className="mt-2 text-sm">
            <span className="font-semibold">Nodes:</span>
            <div className="ml-4 space-y-1">
              <div className="text-base-content/80">
                <span className="font-medium">Data nodes:</span>
                <ul className="list-disc ml-4 mt-1">
                  {dcr.nodes
                    .filter(n => n.type === 'TableDataNodeDefinition' || n.type === 'RawDataNodeDefinition')
                    .map(n => n.name)
                    .filter(Boolean)
                    .map((name, idx) => <li key={idx}>{name}</li>)}
                  {dcr.nodes.filter(n => n.type === 'TableDataNodeDefinition' || n.type === 'RawDataNodeDefinition').length === 0 && <li>none</li>}
                </ul>
              </div>
              <div className="text-base-content/80">
                <span className="font-medium">Compute nodes:</span>
                <ul className="list-disc ml-4 mt-1">
                  {dcr.nodes
                    .filter(n => n.type === 'PreviewComputeNodeDefinition' || n.type === 'PythonComputeNodeDefinition')
                    .map(n => n.name)
                    .filter(Boolean)
                    .map((name, idx) => <li key={idx}>{name}</li>)}
                  {dcr.nodes.filter(n => n.type === 'PreviewComputeNodeDefinition' || n.type === 'PythonComputeNodeDefinition').length === 0 && <li>none</li>}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Link to platform */}
        {dcrUrl && (
          <div className="mt-2">
            <a
              href={dcrUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="link link-primary text-sm flex items-center gap-1 w-fit"
            >
              <ExternalLink size={14} /> Open on Decentriq
            </a>
          </div>
        )}

        {dcr.id && <DcrLogPanel dcrId={dcr.id} />}
      </div>
    </div>
  );
}
