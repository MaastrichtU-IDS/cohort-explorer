import React, {useEffect, useMemo, useRef, useState} from 'react';
import {ChevronLeft, ChevronRight} from 'react-feather';
import {apiUrl} from '@/utils';
import {useCohorts} from '@/components/CohortsContext';
import CohortLinkedText from '@/components/CohortLinkedText';

// Rotating announcements box for the front page. Announcements are short (a
// couple of sentences), so the box stays compact: one announcement at a time,
// auto-rotating and wrapping around continuously, with prev/next arrows on
// either side, dots for position and a "Show all" link opening the full list.
// Hidden entirely when there are no announcements (or the fetch fails).

export interface Announcement {
  id: string;
  text: string;
  date: string; // YYYY-MM-DD
  tag: string;
  // Catalog cohorts mentioned in the text, matched by the backend. Lets the
  // names link even for visitors who are not logged in (they have no cohort
  // list of their own); the link then leads to the explore page's login prompt.
  cohorts?: string[];
}

const ROTATE_MS = 7000;

const TAG_STYLES: Record<string, string> = {
  'new cohort': 'bg-emerald-100 text-emerald-900 border-emerald-300',
  'new feature': 'bg-purple-100 text-purple-900 border-purple-300',
  analysis: 'bg-amber-100 text-amber-900 border-amber-300',
  event: 'bg-sky-100 text-sky-900 border-sky-300'
};

function TagChip({tag}: {tag: string}) {
  const style = TAG_STYLES[tag] || 'bg-base-200 text-base-content border-base-300';
  return (
    <span className={`px-2 py-0.5 rounded-full border text-[11px] font-semibold uppercase tracking-wide whitespace-nowrap ${style}`}>
      {tag}
    </span>
  );
}

function formatDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleDateString('en-GB', {day: 'numeric', month: 'short', year: 'numeric'});
}

export default function AnnouncementsBox() {
  const {cohortsData} = useCohorts();
  // Catalog cohort names: mentions in announcement texts become links to the
  // explore page with that cohort's section opened. The loaded cohorts (when
  // logged in) are joined with the names the backend found in each text.
  const cohortNames = useMemo(() => Object.keys(cohortsData || {}), [cohortsData]);
  const namesFor = (a: Announcement) => [...cohortNames, ...(a.cohorts || [])];
  const [announcements, setAnnouncements] = useState<Announcement[]>([]);
  const [index, setIndex] = useState(0);
  const [showAll, setShowAll] = useState(false);
  const paused = useRef(false);

  useEffect(() => {
    let alive = true;
    fetch(`${apiUrl}/announcements`, {credentials: 'include'})
      .then(res => (res.ok ? res.json() : []))
      .then((data: Announcement[]) => {
        if (alive && Array.isArray(data)) setAnnouncements(data.filter(a => a && a.text));
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  // Continuous rotation: wraps from the last announcement back to the first.
  useEffect(() => {
    if (announcements.length < 2) return;
    const t = setInterval(() => {
      if (!paused.current) setIndex(i => (i + 1) % announcements.length);
    }, ROTATE_MS);
    return () => clearInterval(t);
  }, [announcements.length]);

  if (announcements.length === 0) return null;
  const n = announcements.length;
  const current = announcements[Math.min(index, n - 1)];
  const prev = () => setIndex(i => (i - 1 + n) % n);
  const next = () => setIndex(i => (i + 1) % n);

  return (
    // Same width as the stats grid below; the card itself spans ~73% of it
    // (centered - about 2.2x the original middle-third width), with the
    // arrows just outside it.
    <div
      className="mt-10 w-full max-w-5xl flex items-center justify-center gap-2"
      onMouseEnter={() => (paused.current = true)}
      onMouseLeave={() => (paused.current = false)}
    >
      <style jsx>{`
        @keyframes announceFade {
          from { opacity: 0; transform: translateY(2px); }
          to { opacity: 1; transform: none; }
        }
        .announce-fade { animation: announceFade 350ms ease-out; }
      `}</style>
      <button
        type="button"
        onClick={prev}
        disabled={n < 2}
        className="btn btn-circle btn-sm btn-ghost disabled:opacity-20"
        aria-label="Previous announcement"
        title="Previous announcement"
      >
        <ChevronLeft size={18} />
      </button>

      <div className="bg-base-100 shadow rounded-lg px-4 py-3 border-2 border-base-300 w-full md:w-[73%]">
        <div className="flex items-start gap-3">
          <span className="flex flex-col items-start gap-1">
            <TagChip tag={current.tag} />
            <span className="text-xs text-base-content/50 whitespace-nowrap">{formatDate(current.date)}</span>
          </span>
          <span className="flex-1" />
          {n > 1 && (
            <span className="hidden sm:inline-flex gap-1 items-center" aria-hidden>
              {announcements.map((a, i) => (
                <button
                  key={a.id}
                  onClick={() => setIndex(i)}
                  className={`w-1.5 h-1.5 rounded-full ${i === index ? 'bg-primary' : 'bg-base-300 hover:bg-base-content/30'}`}
                  aria-label={`Announcement ${i + 1}`}
                />
              ))}
            </span>
          )}
          <button className="text-xs text-primary underline whitespace-nowrap" onClick={() => setShowAll(true)}>
            Show all
          </button>
        </div>
        {/* Fixed three-line text area: the box keeps the same height while
            announcements rotate, so nothing below it shifts. Keyed on the
            announcement so each change fades in. */}
        <p key={current.id} className="announce-fade text-base leading-6 mt-2 min-h-[4.5rem] line-clamp-3">
          <CohortLinkedText text={current.text} names={namesFor(current)} />
        </p>
      </div>

      <button
        type="button"
        onClick={next}
        disabled={n < 2}
        className="btn btn-circle btn-sm btn-ghost disabled:opacity-20"
        aria-label="Next announcement"
        title="Next announcement"
      >
        <ChevronRight size={18} />
      </button>

      {showAll && (
        <div className="modal modal-open" onClick={() => setShowAll(false)}>
          <div className="modal-box max-w-2xl" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-start mb-3">
              <h3 className="font-bold text-lg">Announcements</h3>
              <button onClick={() => setShowAll(false)} className="btn btn-sm btn-circle btn-ghost" aria-label="Close">
                ✕
              </button>
            </div>
            <ul className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
              {announcements.map(a => (
                <li key={a.id} className="flex items-start gap-3 border-b border-base-200 pb-2 last:border-b-0">
                  <span className="text-xs text-base-content/50 whitespace-nowrap w-24 shrink-0 pt-0.5">{formatDate(a.date)}</span>
                  <TagChip tag={a.tag} />
                  <span className="text-sm">
                    <CohortLinkedText text={a.text} names={namesFor(a)} />
                  </span>
                </li>
              ))}
            </ul>
            <div className="modal-action">
              <button className="btn btn-sm" onClick={() => setShowAll(false)}>
                Close
              </button>
            </div>
          </div>
          <div className="modal-backdrop" onClick={() => setShowAll(false)}></div>
        </div>
      )}
    </div>
  );
}
