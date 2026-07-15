'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { apiUrl } from '@/utils';
import { AlertTriangle, Clock, RefreshCw, ExternalLink } from 'react-feather';
import { DcrLogPanel } from '@/components/DcrLogPanel';
import {DcrResultPanel} from '@/components/DcrResultPanel';
import {DcrCapabilities, projectDcrProvider} from '@/utils/dcrProvider';

/** Shape of a single DCR record returned by the /my-dcrs endpoint. */
interface DcrRecord {
  id?: string;
  title?: string;
  description?: string;
  createdAt?: string;
  owner?: { email?: string; [key: string]: any };
  participants?: { email?: string; roles?: string[]; data_owner_of?: string[]; analyst_of?: string[] }[];
  nodes?: { name?: string; type?: string; script?: string }[];
  cohorts?: string[];
  dcr_url?: string;
  provider?: string;
  capabilities?: DcrCapabilities;
  provisioned_datasets?: {
    dataset_name?: string;
    node_name?: string;
    dataset_node_name?: string;
    status?: string;
    [key: string]: any;
  }[];
  error?: string;
  [key: string]: any;
}

function normalizeRooms(data: any): DcrRecord[] {
  const rooms = Array.isArray(data?.dcrs) ? data.dcrs : [];
  return rooms.map((room: DcrRecord) => ({
    ...room,
    provider: room.provider ?? data?.provider,
    capabilities: room.capabilities ?? data?.capabilities
  }));
}

export default function DcrsPage() {
  const [dcrs, setDcrs] = useState<DcrRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [lastRefreshedAt, setLastRefreshedAt] = useState<Date | null>(null);
  const [providerUi, setProviderUi] = useState(() => projectDcrProvider());

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
      const dcrs = normalizeRooms(data);
      // Sort reverse chronologically by createdAt (newest first)
      dcrs.sort((a: DcrRecord, b: DcrRecord) => {
        if (!a.createdAt) return 1;
        if (!b.createdAt) return -1;
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      });
      setDcrs(dcrs);
      setUserEmail(data?.email ?? null);
      setProviderUi(projectDcrProvider(data?.provider, data?.capabilities));

      // Fetch the last modified timestamp of the DCR history file
      try {
        const lastModResponse = await fetch(`${apiUrl}/my-dcrs/last-modified`, { credentials: 'include' });
        if (lastModResponse.ok) {
          const lastModData = await lastModResponse.json();
          if (lastModData?.last_modified) {
            setLastRefreshedAt(new Date(lastModData.last_modified));
          }
        }
      } catch {
        // Silently fail if we can't get the last modified timestamp
      }
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
      const dcrs = normalizeRooms(data);
      // Sort reverse chronologically by createdAt (newest first)
      dcrs.sort((a: DcrRecord, b: DcrRecord) => {
        if (!a.createdAt) return 1;
        if (!b.createdAt) return -1;
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      });
      setDcrs(dcrs);
      setUserEmail(data?.email ?? null);
      setProviderUi(projectDcrProvider(data?.provider, data?.capabilities));
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
    <main
      className="flex flex-col items-center justify-start p-6 min-h-screen bg-base-200"
      data-testid="my-dcrs-page"
    >
      <div className="w-full max-w-5xl space-y-6">
        <header className="text-center">
          <h1 className="text-3xl font-bold">My Data Clean Rooms</h1>
          <p className="text-lg text-base-content/70 mt-1">
            Data Clean Rooms you participate in{userEmail && (
              <> (<span className="italic">{userEmail}</span>)</>
            )}.
          </p>
        </header>

        {/* Refresh button */}
        <div className="flex justify-start items-center gap-3">
          <button
            className="btn btn-sm btn-outline gap-2"
            onClick={handleRefresh}
            disabled={isRefreshing}
            title={providerUi.refreshLabel}
            data-testid="my-dcrs-refresh"
          >
            <RefreshCw size={14} className={isRefreshing ? 'animate-spin' : ''} />
            {isRefreshing ? 'Refreshing...' : providerUi.refreshLabel}
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
  const dcrUrl = dcr.dcr_url || null;
  const providerUi = projectDcrProvider(dcr.provider, dcr.capabilities);

  // Determine DCR type based on whether any compute node name starts with c3_eda
  const hasC3EdaScript = dcr.nodes?.some(
    (node) =>
      (node.type === 'PreviewComputeNodeDefinition' || node.type === 'PythonComputeNodeDefinition') &&
      node.name?.startsWith('c3_eda')
  );
  const dcrType = hasC3EdaScript ? 'Provision/EDA' : 'Analysis';
  const badgeColor = hasC3EdaScript ? 'badge-success' : 'badge-secondary';

  return (
    <div
      className={`card bg-base-100 shadow-sm border ${hasC3EdaScript ? 'border-success/30' : 'border-secondary/60'}`}
      data-testid="dcr-room-card"
    >
      <div className="card-body p-4">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <span className={`badge ${badgeColor} badge-lg font-semibold text-base px-4 py-2`}>
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
          {dcr.cohorts && dcr.cohorts.length > 0 && (
            <div className="flex gap-1 flex-wrap">
              {dcr.cohorts.map((cohort, idx) => (
                <span key={idx} className="badge badge-primary badge-lg">
                  {cohort}
                </span>
              ))}
            </div>
          )}
        </div>

        {dcr.description && (
          <div className="mt-2 text-sm text-base-content/80">{dcr.description}</div>
        )}

        {providerUi.localSimulation && (
          <div className="alert alert-info py-2 text-sm mt-2">
            Local synthetic-data simulation; this is not a confidential-computing boundary.
          </div>
        )}

        {/* Participants detail */}
        {dcr.participants && dcr.participants.length > 0 && (
          <div className="mt-3 text-sm">
            <span className="font-semibold">Participants:</span>
            <ul className="list-disc ml-5 mt-1 text-base-content/80">
              {dcr.participants.map((p, i) => {
                // Determine if participant is data owner (owns data nodes without "metadata", "sample", or "mapping" in name)
                const dataOwnerOf = p.data_owner_of || [];
                const isDataOwner = dataOwnerOf.some(
                  nodeId => {
                    // Look up the node name from the nodes array
                    const node = dcr.nodes?.find(n => n.name === nodeId);
                    if (!node) return false;
                    const lowerName = node.name?.toLowerCase() || '';
                    return !lowerName.includes('metadata') &&
                           !lowerName.includes('sample') &&
                           !lowerName.includes('mapping');
                  }
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
            <div className="flex gap-8">
              <div className="flex-1 text-base-content/80">
                <span className="font-semibold">Data nodes:</span>
                <ul className="list-disc ml-4 mt-1">
                  {dcr.nodes
                    .filter(n => n.type === 'TableDataNodeDefinition' || n.type === 'RawDataNodeDefinition')
                    .map(n => n.name)
                    .filter(Boolean)
                    .map((name, idx) => <li key={idx}>{name}</li>)}
                  {dcr.nodes.filter(n => n.type === 'TableDataNodeDefinition' || n.type === 'RawDataNodeDefinition').length === 0 && <li>none</li>}
                </ul>
              </div>
              <div className="flex-1 text-base-content/80">
                <span className="font-semibold">Compute nodes:</span>
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

        {dcr.provisioned_datasets && dcr.provisioned_datasets.length > 0 && (
          <div className="mt-3 text-sm">
            <span className="font-semibold">Provisioned datasets:</span>
            <div className="mt-1 space-y-1">
              {dcr.provisioned_datasets.map((dataset, index) => (
                <div
                  key={`${dataset.dataset_name || 'dataset'}-${index}`}
                  className="flex flex-wrap gap-2 rounded bg-base-200 px-2 py-1"
                  data-testid="dcr-provisioning-row"
                >
                  <span>{dataset.dataset_name || 'dataset'}</span>
                  <span className="text-base-content/60">→ {dataset.node_name || dataset.dataset_node_name || 'PROD node'}</span>
                  <span className="badge badge-sm">{dataset.status || 'provisioned'}</span>
                </div>
              ))}
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
              <ExternalLink size={14} /> {providerUi.openLabel}
            </a>
          </div>
        )}

        {dcr.id && dcr.capabilities?.supports_audit_log !== false && <DcrLogPanel dcrId={dcr.id} />}
        {dcr.id && (
          <DcrResultPanel
            dcrId={dcr.id}
            provider={dcr.provider}
            capabilities={dcr.capabilities}
          />
        )}
      </div>
    </div>
  );
}
