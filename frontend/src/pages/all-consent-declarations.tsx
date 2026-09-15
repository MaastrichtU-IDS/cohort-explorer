import React, {useEffect, useState} from 'react';
import {Shield, Users, FileText, CheckCircle, Clock, XCircle, RefreshCw, ChevronDown, ChevronRight, Trash2} from 'react-feather';
import {apiUrl} from '@/utils';
import {useCohorts} from '@/components/CohortsContext';

const PERMISSION_LABELS: Record<string, {label: string; color: string}> = {
  NRES: {label: 'No Restrictions', color: 'badge-success'},
  GRU:  {label: 'General Research Use', color: 'badge-info'},
  HMB:  {label: 'Health/Medical/Biomedical', color: 'badge-info'},
  DS:   {label: 'Disease Specific', color: 'badge-warning'},
  POA:  {label: 'Population Origins/Ancestry', color: 'badge-warning'},
};

const STATUS_META: Record<string, {icon: React.ReactNode; color: string}> = {
  approved: {icon: <CheckCircle size={13} />, color: 'text-success'},
  pending:  {icon: <Clock size={13} />,       color: 'text-warning'},
  rejected: {icon: <XCircle size={13} />,     color: 'text-error'},
  denied:   {icon: <XCircle size={13} />,     color: 'text-error'},
};

const codesOf = (x: {disease_code?: string; disease_codes?: string[]}): string[] =>
  x.disease_codes && x.disease_codes.length > 0 ? x.disease_codes : (x.disease_code ? [x.disease_code] : []);

interface AccessGrant {
  requester: string;
  requester_hash: string;
  status: string;
  intended_use?: string;
  disease_code?: string;
  disease_codes?: string[];
  reason?: string | null;
  reason_detail?: string | null;
  decided_at?: string;
  project_id?: string;
  abstract?: string;
  requested_at?: string;
  granted_at?: string;
  request_id?: string;
  tx_hash?: string;
  profile?: {institution_id?: string; requester_type?: string; country_code?: string} | null;
}

interface ConsentRecord {
  cohort_id: string;
  cohort_hash: string;
  permission: string;
  modifiers: string[];
  disease_code?: string;
  disease_codes?: string[];
  data_use_description?: string;
  additional_restrictions?: string;
  research_scope?: string;
  allowed_countries: string[];
  allowed_institutions: string[];
  moratorium_months?: number;
  active: boolean;
  valid_until?: string;
  recorded_at?: string;
  owners: string[];
  access_grants: AccessGrant[];
}

interface RequesterProfile {
  email_hash: string;
  address: string;
  institution_id: string;
  requester_type: string;
  country_code?: string;
  public_profile: boolean;
  updated_at?: string;
}

interface OverviewData {
  consents: ConsentRecord[];
  requester_profiles: RequesterProfile[];
  stats: {
    total_consents: number;
    active_consents: number;
    total_requester_profiles: number;
    total_access_requests: number;
    approved_access_requests: number;
    pending_access_requests: number;
    rejected_access_requests?: number;
  };
}

function fmtDate(iso?: string) {
  if (!iso) return '—';
  try {return new Date(iso).toLocaleString('en-GB', {dateStyle: 'medium', timeStyle: 'short'});}
  catch {return iso;}
}

function shortHash(h: string) {
  if (!h) return '—';
  return h.length > 12 ? `${h.slice(0, 6)}…${h.slice(-4)}` : h;
}

function ConsentCard({c, defaultOpen}: {c: ConsentRecord; defaultOpen?: boolean}) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  const perm = PERMISSION_LABELS[c.permission] ?? {label: c.permission, color: 'badge-ghost'};
  const approvedCount = c.access_grants.filter(g => g.status === 'approved').length;
  const pendingCount  = c.access_grants.filter(g => g.status === 'pending').length;
  const rejectedCount = c.access_grants.filter(g => g.status === 'rejected' || g.status === 'denied').length;

  return (
    <div className="border border-base-300 rounded-xl overflow-hidden bg-base-100 shadow-sm">
      <button
        type="button"
        className="w-full flex items-center gap-3 px-5 py-4 hover:bg-base-200 transition-colors text-left"
        onClick={() => setOpen(o => !o)}
      >
        <span className="text-base-content/40">{open ? <ChevronDown size={18} /> : <ChevronRight size={18} />}</span>
        <span className="font-mono font-bold text-base-content flex-shrink-0 w-40 truncate">{c.cohort_id}</span>
        <span className={`badge badge-sm ${perm.color} shrink-0`}>{perm.label}</span>
        {c.modifiers.length > 0 && (
          <span className="flex gap-1 flex-wrap">
            {c.modifiers.map(m => <span key={m} className="badge badge-xs badge-ghost">{m}</span>)}
          </span>
        )}
        {codesOf(c).length > 0 && (
          <span className="flex gap-1 flex-wrap">
            {codesOf(c).map(d => <span key={d} className="badge badge-xs badge-outline font-mono">{d}</span>)}
          </span>
        )}
        <span className="ml-auto flex gap-3 shrink-0 items-center">
          {!c.active && <span className="badge badge-xs badge-error">Revoked</span>}
          {approvedCount > 0 && <span className="text-xs text-success font-medium flex items-center gap-1"><CheckCircle size={12}/>{approvedCount} approved</span>}
          {pendingCount  > 0 && <span className="text-xs text-warning font-medium flex items-center gap-1"><Clock size={12}/>{pendingCount} pending</span>}
          {rejectedCount > 0 && <span className="text-xs text-error font-medium flex items-center gap-1"><XCircle size={12}/>{rejectedCount} rejected</span>}
          {c.access_grants.length === 0 && <span className="text-xs text-base-content/40">No requests</span>}
        </span>
      </button>

      {open && (
        <div className="border-t border-base-300 bg-base-50">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 px-5 py-4 text-xs text-base-content/70 bg-base-200/40 border-b border-base-300">
            <div><span className="font-semibold block mb-0.5">Recorded</span>{fmtDate(c.recorded_at)}</div>
            <div><span className="font-semibold block mb-0.5">Expires</span>{c.valid_until ? fmtDate(c.valid_until) : 'Never'}</div>
            <div><span className="font-semibold block mb-0.5">Status</span>{c.active ? <span className="text-success font-semibold">Active</span> : <span className="text-error font-semibold">Revoked</span>}</div>
            <div><span className="font-semibold block mb-0.5">Hash</span><span className="font-mono">{shortHash(c.cohort_hash)}</span></div>
            {c.data_use_description && <div className="col-span-2 sm:col-span-4"><span className="font-semibold block mb-0.5">Data Use Description</span>{c.data_use_description}</div>}
            {c.research_scope && <div className="col-span-2 sm:col-span-4"><span className="font-semibold block mb-0.5">Research Scope</span>{c.research_scope}</div>}
            {c.allowed_countries?.length > 0 && <div><span className="font-semibold block mb-0.5">Allowed Countries</span>{c.allowed_countries.join(', ')}</div>}
            {c.moratorium_months && <div><span className="font-semibold block mb-0.5">Moratorium</span>{c.moratorium_months} months</div>}
          </div>

          {c.access_grants.length === 0 ? (
            <p className="px-5 py-6 text-sm text-base-content/50 italic text-center">No access requests for this cohort yet.</p>
          ) : (
            <div className="divide-y divide-base-300">
              {c.access_grants.map((g, i) => {
                const sm = STATUS_META[g.status] ?? STATUS_META['pending'];
                return (
                  <div key={g.request_id || i} className="px-5 py-3 flex flex-col sm:flex-row sm:items-start gap-3">
                    <div className="flex items-center gap-2 sm:w-36 shrink-0">
                      <span className={`flex items-center gap-1 text-xs font-semibold ${sm.color}`}>{sm.icon}{g.status}</span>
                    </div>
                    <div className="flex-1 grid grid-cols-1 sm:grid-cols-3 gap-x-4 gap-y-1 text-xs">
                      <div>
                        <span className="text-base-content/50 font-semibold block">Requester</span>
                        <span className="font-mono">{shortHash(g.requester)}</span>
                        {g.profile?.institution_id && <span className="ml-1 text-base-content/60">({g.profile.institution_id}{g.profile.country_code ? `, ${g.profile.country_code}` : ''})</span>}
                      </div>
                      <div>
                        <span className="text-base-content/50 font-semibold block">Intended Use</span>
                        {g.intended_use || '—'}
                        {codesOf(g).length > 0 && <span className="ml-1 font-mono text-primary">· {codesOf(g).join(', ')}</span>}
                      </div>
                      <div>
                        <span className="text-base-content/50 font-semibold block">Requested</span>
                        {fmtDate(g.requested_at)}
                        {g.granted_at && g.status === 'approved' && <span className="block text-success">✓ Granted {fmtDate(g.granted_at)}</span>}
                        {(g.status === 'rejected' || g.status === 'denied') && (g.reason_detail || g.reason) && (
                          <span className="block text-error">✗ {g.reason_detail || g.reason}{g.reason && g.reason_detail ? ` (${g.reason})` : ''}</span>
                        )}
                      </div>
                      {g.abstract && (
                        <div className="col-span-1 sm:col-span-3 mt-1">
                          <span className="text-base-content/50 font-semibold block">Abstract</span>
                          <span className="text-base-content/70 italic">{g.abstract}</span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function AllConsentDeclarationsPage() {
  const {userEmail} = useCohorts();
  const [data, setData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<'consents' | 'requesters'>('consents');
  const [filter, setFilter] = useState('');
  const [resetting, setResetting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const resetChain = async () => {
    if (!window.confirm('Reset the LOCAL blockchain?\n\nAll consent declarations, access requests and requester profiles on the local Hardhat chain will be wiped and the contracts restored to their freshly deployed state. This cannot be undone.')) return;
    setResetting(true);
    setError(null);
    setNotice(null);
    try {
      const r = await fetch(`${apiUrl}/blockchain/admin/reset`, {method: 'POST', credentials: 'include'});
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
      setNotice(`Local chain reset to block ${body.blockNumber}. Cohorts must re-declare consent to reappear here.`);
      load();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setResetting(false);
    }
  };

  const load = () => {
    if (!userEmail) return;
    setLoading(true);
    setError(null);
    fetch(`${apiUrl}/blockchain/admin/overview`, {credentials: 'include'})
      .then(r => {
        if (!r.ok) return r.json().then(e => { throw new Error(e.detail || `HTTP ${r.status}`); });
        return r.json();
      })
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  };

  useEffect(() => { load(); }, [userEmail]);

  if (!userEmail) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-3">
          <Shield size={48} className="mx-auto text-base-content/30" />
          <p className="text-base-content/60">Please log in to view consent activity.</p>
        </div>
      </div>
    );
  }

  const q = filter.trim().toLowerCase();
  const filteredConsents = data?.consents.filter(c =>
    !q ||
    c.cohort_id.toLowerCase().includes(q) ||
    c.permission.toLowerCase().includes(q) ||
    codesOf(c).some(d => d.toLowerCase().includes(q)) ||
    c.access_grants.some(g => codesOf(g).some(d => d.toLowerCase().includes(q))) ||
    c.modifiers.some(m => m.toLowerCase().includes(q))
  ) ?? [];

  const filteredProfiles = data?.requester_profiles.filter(p =>
    !q ||
    p.institution_id.toLowerCase().includes(q) ||
    p.requester_type.toLowerCase().includes(q) ||
    (p.country_code || '').toLowerCase().includes(q) ||
    p.email_hash.toLowerCase().includes(q)
  ) ?? [];

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-3">
            <Shield size={26} className="text-primary" />
            Consent &amp; Access Activity
          </h1>
          <p className="text-sm text-base-content/60 mt-1">Admin-level view of all blockchain consent declarations, access requests, and registered requester profiles.</p>
        </div>
        <div className="flex gap-2">
          <button type="button" className="btn btn-outline btn-sm gap-2" onClick={load} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
          <button type="button" className="btn btn-outline btn-error btn-sm gap-2" onClick={resetChain} disabled={resetting || loading} title="Dev only: revert the local Hardhat chain to its post-deploy state and flush the cache">
            <Trash2 size={14} className={resetting ? 'animate-pulse' : ''} />
            {resetting ? 'Resetting…' : 'Reset local chain'}
          </button>
        </div>
      </div>

      {notice && (
        <div className="alert alert-success shadow-sm">
          <CheckCircle size={18} />
          <span>{notice}</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="alert alert-error shadow-sm">
          <XCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Stats */}
      {data && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {[
            {label: 'Total Consents',      value: data.stats.total_consents,          icon: <FileText size={18}/>,    color: 'text-primary'},
            {label: 'Requester Profiles',  value: data.stats.total_requester_profiles, icon: <Users size={18}/>,       color: 'text-info'},
            {label: 'Access Requests',     value: data.stats.total_access_requests,    icon: <FileText size={18}/>,    color: 'text-base-content'},
            {label: 'Approved',            value: data.stats.approved_access_requests, icon: <CheckCircle size={18}/>, color: 'text-success'},
            {label: 'Pending',             value: data.stats.pending_access_requests,  icon: <Clock size={18}/>,       color: 'text-warning'},
            {label: 'Rejected',            value: data.stats.rejected_access_requests ?? 0, icon: <XCircle size={18}/>, color: 'text-error'},
          ].map(s => (
            <div key={s.label} className="stat bg-base-100 border border-base-300 rounded-xl p-3">
              <div className={`stat-figure ${s.color} opacity-60`}>{s.icon}</div>
              <div className="stat-title text-xs">{s.label}</div>
              <div className={`stat-value text-2xl ${s.color}`}>{s.value}</div>
            </div>
          ))}
        </div>
      )}

      {loading && !data && (
        <div className="flex items-center justify-center py-20">
          <span className="loading loading-spinner loading-lg text-primary" />
        </div>
      )}

      {data && (
        <>
          {/* Tabs + Search */}
          <div className="flex flex-col sm:flex-row sm:items-center gap-3">
            <div className="tabs tabs-boxed bg-base-200 self-start">
              <button className={`tab gap-2 ${tab === 'consents' ? 'tab-active' : ''}`} onClick={() => setTab('consents')}>
                <FileText size={14} /> Consent Declarations <span className="badge badge-sm ml-1">{data.consents.length}</span>
              </button>
              <button className={`tab gap-2 ${tab === 'requesters' ? 'tab-active' : ''}`} onClick={() => setTab('requesters')}>
                <Users size={14} /> Requester Profiles <span className="badge badge-sm ml-1">{data.requester_profiles.length}</span>
              </button>
            </div>
            <input
              type="text"
              className="input input-bordered input-sm flex-1 max-w-xs ml-auto"
              placeholder="Filter…"
              value={filter}
              onChange={e => setFilter(e.target.value)}
            />
          </div>

          {/* Consent Declarations Tab */}
          {tab === 'consents' && (
            <div className="space-y-3">
              {filteredConsents.length === 0 ? (
                <div className="text-center py-16 text-base-content/40">
                  <FileText size={40} className="mx-auto mb-3 opacity-30" />
                  {q ? `No consent declarations match "${filter}"` : 'No consent declarations recorded yet.'}
                </div>
              ) : (
                filteredConsents.map(c => <ConsentCard key={c.cohort_id} c={c} />)
              )}
            </div>
          )}

          {/* Requester Profiles Tab */}
          {tab === 'requesters' && (
            <div>
              {filteredProfiles.length === 0 ? (
                <div className="text-center py-16 text-base-content/40">
                  <Users size={40} className="mx-auto mb-3 opacity-30" />
                  {q ? `No profiles match "${filter}"` : 'No requester profiles found. Profiles appear here once a requester submits an access request.'}
                </div>
              ) : (
                <div className="overflow-x-auto rounded-xl border border-base-300">
                  <table className="table table-zebra w-full text-sm">
                    <thead>
                      <tr className="bg-base-200 text-xs">
                        <th>Email Hash</th>
                        <th>Institution</th>
                        <th>Type</th>
                        <th>Country</th>
                        <th>Address</th>
                        <th>Public</th>
                        <th>Last Updated</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredProfiles.map(p => (
                        <tr key={p.email_hash}>
                          <td className="font-mono text-xs text-base-content/60">{shortHash(p.email_hash)}</td>
                          <td className="font-medium">{p.institution_id || '—'}</td>
                          <td><span className="badge badge-sm badge-ghost">{p.requester_type || '—'}</span></td>
                          <td>{p.country_code || '—'}</td>
                          <td className="font-mono text-xs text-base-content/60">{shortHash(p.address)}</td>
                          <td>{p.public_profile ? <span className="text-success text-xs">Yes</span> : <span className="text-base-content/40 text-xs">No</span>}</td>
                          <td className="text-xs text-base-content/60">{fmtDate(p.updated_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
