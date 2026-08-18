// iCARE-AI conversation history — ADMIN ONLY.
//
// Shows every stored AI conversation (backend: src/ai_history.py) with usage
// metrics and a transcript viewer. Gated behind the same /admin/check the other
// admin pages use. Filtering is by arrival path and message count for now
// (deliberately NOT by user yet). Not linked from the /ai hub on purpose.
import React, {useEffect, useState} from 'react';
import {AlertTriangle, Shield, Clock, MessageSquare, Users, Activity, X} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {apiUrl} from '@/utils';
import {
  ConversationSummary,
  ConversationDetail,
  UsageSummary,
  fetchHistory,
  fetchConversation,
  fetchUsageSummary
} from '@/components/ai/chatClient';

const PATH_LABELS: Record<string, string> = {
  chat: 'Chat',
  intention_cards: 'Guided'
};

function pathLabel(p: string): string {
  return PATH_LABELS[p] || p || 'unknown';
}

function pathBadgeClass(p: string): string {
  if (p === 'chat') return 'badge-info';
  if (p === 'intention_cards') return 'badge-secondary';
  return 'badge-ghost';
}

function fmtDuration(s: number | null): string {
  if (s == null) return '—';
  if (s < 1) return '0s';
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  if (m < 60) return `${m}m ${sec}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function fmtTime(iso: string): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function fmtNum(n: number | null | undefined, digits = 0): string {
  if (n == null) return '—';
  return n.toLocaleString(undefined, {maximumFractionDigits: digits});
}

function StatCard({icon, label, value}: {icon: React.ReactNode; label: string; value: string}) {
  return (
    <div className="card bg-base-200 shadow-sm">
      <div className="card-body p-4 flex-row items-center gap-3">
        <div className="text-primary">{icon}</div>
        <div>
          <div className="text-xs text-base-content/60">{label}</div>
          <div className="text-xl font-bold leading-tight">{value}</div>
        </div>
      </div>
    </div>
  );
}

export default function AiHistoryPage() {
  const {userEmail} = useCohorts();
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);

  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters (not by user — by design, for now).
  const [path, setPath] = useState<'' | 'chat' | 'intention_cards'>('');
  const [minMessages, setMinMessages] = useState<string>('');
  const [search, setSearch] = useState('');

  // Transcript viewer.
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Admin gate.
  useEffect(() => {
    if (!userEmail) return;
    let cancelled = false;
    fetch(`${apiUrl}/admin/check`, {credentials: 'include'})
      .then(res => (res.ok ? res.json() : {is_admin: false}))
      .then(data => {
        if (!cancelled) setIsAdmin(!!data.is_admin);
      })
      .catch(() => {
        if (!cancelled) setIsAdmin(false);
      });
    return () => {
      cancelled = true;
    };
  }, [userEmail]);

  // Load list + summary whenever filters change (admins see all users).
  useEffect(() => {
    if (!isAdmin) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    const min = minMessages.trim() === '' ? undefined : Math.max(0, parseInt(minMessages, 10) || 0);
    Promise.all([
      fetchHistory({
        scope: 'all',
        path: path || undefined,
        search: search.trim() || undefined,
        minMessages: min,
        limit: 200
      }),
      fetchUsageSummary('all')
    ])
      .then(([page, sum]) => {
        if (cancelled) return;
        setItems(page.items);
        setTotal(page.total);
        setSummary(sum);
      })
      .catch(err => {
        if (!cancelled) setError(err.message || 'Failed to load history.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [isAdmin, path, minMessages, search]);

  const openDetail = (id: string) => {
    setDetailLoading(true);
    setDetail(null);
    fetchConversation(id)
      .then(setDetail)
      .catch(err => setError(err.message || 'Failed to load conversation.'))
      .finally(() => setDetailLoading(false));
  };

  // ---- gate states ----
  if (isAdmin === null && userEmail) {
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
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="flex items-center gap-3 mb-2">
        <Clock size={26} />
        <h1 className="text-2xl font-bold">AI Conversation History</h1>
      </div>
      <p className="text-sm text-base-content/60 mb-6">
        All iCARE-AI conversations across users. Admin-only.
      </p>

      {error && (
        <div className="alert alert-error mb-6">
          <AlertTriangle size={16} />
          <span>{error}</span>
          <button className="btn btn-sm btn-ghost" onClick={() => setError(null)}>
            ✕
          </button>
        </div>
      )}

      {/* Usage summary */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <StatCard
            icon={<MessageSquare size={20} />}
            label="Conversations"
            value={fmtNum(summary.conversations)}
          />
          <StatCard
            icon={<Activity size={20} />}
            label="Total messages"
            value={fmtNum(summary.messages)}
          />
          <StatCard
            icon={<Users size={20} />}
            label="Users"
            value={fmtNum(summary.users)}
          />
          <StatCard
            icon={<Clock size={20} />}
            label="Avg. duration"
            value={fmtDuration(summary.avg_duration_seconds)}
          />
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-end gap-4 mb-4 p-4 bg-base-200 rounded-xl">
        <div className="form-control">
          <label className="label py-1">
            <span className="label-text text-xs">Arrival path</span>
          </label>
          <select
            className="select select-bordered select-sm"
            value={path}
            onChange={e => setPath(e.target.value as '' | 'chat' | 'intention_cards')}
          >
            <option value="">All paths</option>
            <option value="chat">Chat</option>
            <option value="intention_cards">Guided (intention cards)</option>
          </select>
        </div>

        <div className="form-control">
          <label className="label py-1">
            <span className="label-text text-xs">Min. messages</span>
          </label>
          <input
            type="number"
            min={0}
            placeholder="any"
            className="input input-bordered input-sm w-28"
            value={minMessages}
            onChange={e => setMinMessages(e.target.value)}
          />
        </div>

        <div className="form-control flex-1 min-w-[200px]">
          <label className="label py-1">
            <span className="label-text text-xs">Search transcript</span>
          </label>
          <input
            type="text"
            placeholder="keyword…"
            className="input input-bordered input-sm w-full"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>

        <div className="text-sm text-base-content/60 ml-auto self-center">
          {loading ? 'Loading…' : `${items.length} of ${total} shown`}
        </div>
      </div>

      {/* List */}
      <div className="overflow-x-auto bg-base-100 rounded-xl border border-base-300">
        <table className="table table-sm">
          <thead>
            <tr>
              <th>Last activity</th>
              <th>User</th>
              <th>Path</th>
              <th className="text-right">Msgs</th>
              <th className="text-right">Duration</th>
              <th>First message</th>
            </tr>
          </thead>
          <tbody>
            {items.map(c => (
              <tr
                key={c.id}
                className="hover cursor-pointer"
                onClick={() => openDetail(c.id)}
              >
                <td className="whitespace-nowrap text-xs">{fmtTime(c.updated_at)}</td>
                <td className="text-xs">{c.user_id}</td>
                <td>
                  <span className={`badge badge-sm ${pathBadgeClass(c.arrival_path)}`}>
                    {pathLabel(c.arrival_path)}
                  </span>
                </td>
                <td className="text-right tabular-nums">{c.message_count}</td>
                <td className="text-right tabular-nums text-xs">
                  {fmtDuration(c.duration_seconds)}
                </td>
                <td className="max-w-md truncate text-xs text-base-content/70">{c.preview}</td>
              </tr>
            ))}
            {!loading && items.length === 0 && (
              <tr>
                <td colSpan={6} className="text-center text-base-content/50 py-8">
                  No conversations match these filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Transcript modal */}
      {(detail || detailLoading) && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          onClick={() => {
            setDetail(null);
            setDetailLoading(false);
          }}
        >
          <div
            className="bg-base-100 rounded-2xl shadow-xl max-w-3xl w-full max-h-[85vh] flex flex-col"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-base-300">
              <h3 className="font-bold">Conversation</h3>
              <button
                className="btn btn-sm btn-ghost btn-circle"
                onClick={() => {
                  setDetail(null);
                  setDetailLoading(false);
                }}
              >
                <X size={18} />
              </button>
            </div>

            {detailLoading && (
              <div className="flex justify-center py-12">
                <span className="loading loading-spinner loading-lg"></span>
              </div>
            )}

            {detail && (
              <>
                <div className="px-6 py-3 border-b border-base-300 text-xs text-base-content/60 flex flex-wrap gap-x-6 gap-y-1">
                  <span>
                    <b>User:</b> {detail.user_id}
                  </span>
                  <span>
                    <b>Path:</b> {pathLabel(detail.arrival_path)}
                  </span>
                  <span>
                    <b>Messages:</b> {detail.message_count}
                  </span>
                  <span>
                    <b>Duration:</b> {fmtDuration(detail.duration_seconds)}
                  </span>
                  <span>
                    <b>Model:</b> {detail.model || '—'}
                  </span>
                  <span>
                    <b>Started:</b> {fmtTime(detail.started_at)}
                  </span>
                </div>
                <div className="overflow-y-auto px-6 py-4 space-y-4">
                  {detail.messages.map((m, i) => (
                    <div
                      key={i}
                      className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}
                    >
                      <div
                        className={`rounded-2xl px-4 py-2 max-w-[85%] text-sm whitespace-pre-wrap ${
                          m.role === 'user'
                            ? 'bg-primary text-primary-content'
                            : 'bg-base-200'
                        }`}
                      >
                        <div className="text-[10px] uppercase tracking-wide opacity-60 mb-1">
                          {m.role}
                        </div>
                        {m.role === 'assistant'
                          ? m.detailed || m.summary || m.content
                          : m.content}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
