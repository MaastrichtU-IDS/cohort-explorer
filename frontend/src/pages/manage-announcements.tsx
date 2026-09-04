import React, {useCallback, useEffect, useState} from 'react';
import {apiUrl} from '@/utils';
import {useCohorts} from '@/components/CohortsContext';
import LoginPrompt from '@/components/LoginPrompt';

// Admin page: add and delete the announcements shown on the front page.
// Adding needs a text (one or two sentences), an obligatory date (backdating
// allowed) and one of the fixed tags. No edit on purpose: delete and re-add.

interface AdminAnnouncement {
  id: string;
  text: string;
  date: string;
  tag: string;
  added_by?: string;
  created_at?: string;
}

const TAGS = ['new cohort', 'new feature', 'analysis', 'event'];

const TAG_STYLES: Record<string, string> = {
  'new cohort': 'bg-emerald-100 text-emerald-900 border-emerald-300',
  'new feature': 'bg-purple-100 text-purple-900 border-purple-300',
  analysis: 'bg-amber-100 text-amber-900 border-amber-300',
  event: 'bg-sky-100 text-sky-900 border-sky-300'
};

export default function ManageAnnouncements() {
  const {userEmail} = useCohorts();
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [announcements, setAnnouncements] = useState<AdminAnnouncement[]>([]);
  const [boxEnabled, setBoxEnabled] = useState(true);
  const [text, setText] = useState('');
  const [date, setDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [tag, setTag] = useState(TAGS[0]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    fetch(`${apiUrl}/announcements/all`, {credentials: 'include'})
      .then(res => (res.ok ? res.json() : {enabled: true, items: []}))
      .then((data: {enabled?: boolean; items?: AdminAnnouncement[]}) => {
        setBoxEnabled(data.enabled !== false);
        setAnnouncements(Array.isArray(data.items) ? data.items : []);
      })
      .catch(() => {});
  }, []);

  const setVisibility = async (enabled: boolean) => {
    setError(null);
    setBoxEnabled(enabled); // optimistic; corrected on refresh if the call fails
    try {
      const res = await fetch(`${apiUrl}/announcements/visibility`, {
        method: 'POST',
        credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled})
      });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
    } catch (e: any) {
      setError(e?.message || 'Could not change the box visibility.');
    } finally {
      refresh();
    }
  };

  useEffect(() => {
    fetch(`${apiUrl}/admin/check`, {credentials: 'include'})
      .then(res => (res.ok ? res.json() : {is_admin: false}))
      .then(d => setIsAdmin(!!d.is_admin))
      .catch(() => setIsAdmin(false));
    refresh();
  }, [refresh]);

  const add = async () => {
    setError(null);
    setBusy(true);
    try {
      const res = await fetch(`${apiUrl}/announcements`, {
        method: 'POST',
        credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text.trim(), date, tag})
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail || `Request failed (${res.status})`);
      }
      setText('');
      refresh();
    } catch (e: any) {
      setError(e?.message || 'Could not add the announcement.');
    } finally {
      setBusy(false);
    }
  };

  const remove = async (a: AdminAnnouncement) => {
    if (!window.confirm(`Delete this announcement?\n\n"${a.text}"`)) return;
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/announcements/${encodeURIComponent(a.id)}`, {
        method: 'DELETE',
        credentials: 'include'
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail || `Request failed (${res.status})`);
      }
      refresh();
    } catch (e: any) {
      setError(e?.message || 'Could not delete the announcement.');
    }
  };

  if (userEmail === null) {
    return <LoginPrompt message="Authenticate to manage announcements" />;
  }
  if (isAdmin === null) {
    return <div className="p-8 text-sm text-base-content/60">Checking access…</div>;
  }
  if (!isAdmin) {
    return (
      <div className="p-8">
        <div className="alert alert-warning max-w-xl">Admin access required to manage announcements.</div>
      </div>
    );
  }

  return (
    <main className="p-6 md:p-10 max-w-4xl mx-auto space-y-8">
      <h1 className="text-2xl font-bold">Manage Announcements</h1>

      <section className="card card-bordered bg-base-100 shadow">
        <div className="card-body py-4 flex-row items-center gap-4 flex-wrap">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              className="toggle toggle-primary"
              checked={boxEnabled}
              onChange={e => setVisibility(e.target.checked)}
            />
            <span className="font-semibold">Show the announcements box on the front page</span>
          </label>
          {!boxEnabled && <span className="badge badge-warning">currently hidden for everyone</span>}
          <span className="text-xs text-base-content/60 basis-full">
            The box also hides itself automatically when there are no announcements.
          </span>
        </div>
      </section>

      <section className="card card-bordered bg-base-100 shadow">
        <div className="card-body space-y-3">
          <h2 className="card-title text-lg">Add an announcement</h2>
          <textarea
            className="textarea textarea-bordered w-full"
            rows={2}
            maxLength={500}
            placeholder="One or two sentences…"
            value={text}
            onChange={e => setText(e.target.value)}
          />
          <div className="flex flex-wrap items-end gap-4">
            <label className="form-control">
              <span className="label-text text-xs mb-1">Date (obligatory, backdating allowed)</span>
              <input
                type="date"
                className="input input-bordered input-sm"
                value={date}
                onChange={e => setDate(e.target.value)}
              />
            </label>
            <label className="form-control">
              <span className="label-text text-xs mb-1">Tag</span>
              <select className="select select-bordered select-sm" value={tag} onChange={e => setTag(e.target.value)}>
                {TAGS.map(t => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <button className="btn btn-primary btn-sm" onClick={add} disabled={busy || !text.trim() || !date}>
              Add announcement
            </button>
          </div>
          {error && <div className="text-sm text-error">{error}</div>}
        </div>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold">
          Current announcements <span className="text-sm font-normal text-base-content/60">({announcements.length})</span>
        </h2>
        {announcements.length === 0 ? (
          <p className="text-sm text-base-content/60">No announcements yet.</p>
        ) : (
          <ul className="space-y-2">
            {announcements.map(a => (
              <li key={a.id} className="card card-compact card-bordered bg-base-100">
                <div className="card-body flex-row items-start gap-3">
                  <span className="text-xs text-base-content/50 whitespace-nowrap w-24 shrink-0 pt-1">{a.date}</span>
                  <span
                    className={`px-2 py-0.5 rounded-full border text-[11px] font-semibold uppercase tracking-wide whitespace-nowrap ${
                      TAG_STYLES[a.tag] || 'bg-base-200 border-base-300'
                    }`}
                  >
                    {a.tag}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm">{a.text}</p>
                    <p className="text-xs text-base-content/50 mt-1">
                      added by {a.added_by || 'unknown'}
                      {a.created_at ? ` on ${a.created_at.slice(0, 10)}` : ''}
                    </p>
                  </div>
                  <button className="btn btn-xs btn-error btn-outline" onClick={() => remove(a)}>
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
