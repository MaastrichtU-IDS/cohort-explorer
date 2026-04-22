// Shared types and helpers for the /dcrs and /dcrs-details pages.
// Groups the raw JSONL activity log produced by the backend into
// per-wizard-session DcrSession records.

export interface DcrEvent {
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
  additional_analysts?: string[] | null;
  excluded_data_owners?: string[] | null;
  airlock_settings?: Record<string, number | boolean> | null;
  include_shuffled_samples?: Record<string, boolean> | boolean | null;
  participants?: Record<string, { data_owner_of: string[]; analyst_of: string[] }> | null;
  [key: string]: any;
}

export type Outcome = 'completed' | 'abandoned' | 'failed' | 'in_progress';

export interface DcrSession {
  session_id: string;
  user_email: string | null;
  started_at: string;
  last_event_at: string;
  dcr_name: string | null;
  dcr_title: string | null;
  dcr_url: string | null;
  dcr_id: string | null;
  cohorts: string[];
  research_question: string | null;
  outcome: Outcome;
  error_type: string | null;
  error_message: string | null;
  // Aggregated seconds spent on each wizard step (keyed by step_name).
  time_per_step: Record<string, number>;
  total_seconds: number;
  // Data sample choices captured during wizard/publish.
  airlock_settings: Record<string, number | boolean> | null;
  include_shuffled_samples: Record<string, boolean> | boolean | null;
  // Participants captured at publish time.
  additional_analysts: string[];
  excluded_data_owners: string[];
  participants: Record<string, { data_owner_of: string[]; analyst_of: string[] }> | null;
  // Full list of events for optional detail view.
  events: DcrEvent[];
}

export const OUTCOME_LABEL: Record<Outcome, string> = {
  completed: 'Successfully created',
  abandoned: 'Abandoned',
  failed: 'Failed',
  in_progress: 'In progress',
};

export const OUTCOME_BADGE: Record<Outcome, string> = {
  completed: 'badge-success',
  abandoned: 'badge-warning',
  failed: 'badge-error',
  in_progress: 'badge-info',
};

export function pickFirst<T>(values: (T | null | undefined)[]): T | null {
  for (const v of values) {
    if (v !== null && v !== undefined && (typeof v !== 'string' || v.length > 0)) {
      return v as T;
    }
  }
  return null;
}

export function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export function groupEventsBySession(events: DcrEvent[]): DcrSession[] {
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
    if (eventNames.has('dcr_publish_succeeded')) {
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
      if (
        (evt.event === 'wizard_step_left' ||
          evt.event === 'wizard_closed' ||
          evt.event === 'wizard_abandoned') &&
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
      dcr_id: pickFirst(list.map((e) => e.dcr_id ?? null)),
      cohorts,
      research_question: pickFirst(list.map((e) => e.research_question ?? null)),
      outcome,
      error_type: failedEvt?.error_type ?? null,
      error_message: failedEvt?.error_message ?? null,
      time_per_step: timePerStep,
      total_seconds: totalSeconds,
      airlock_settings: pickFirst(list.map((e) => e.airlock_settings ?? null)),
      include_shuffled_samples: pickFirst(list.map((e) => e.include_shuffled_samples ?? null)),
      additional_analysts:
        (pickFirst(list.map((e) => (Array.isArray(e.additional_analysts) ? e.additional_analysts : null))) as
          | string[]
          | null) || [],
      excluded_data_owners:
        (pickFirst(list.map((e) => (Array.isArray(e.excluded_data_owners) ? e.excluded_data_owners : null))) as
          | string[]
          | null) || [],
      participants: pickFirst(list.map((e) => e.participants ?? null)),
      events: list,
    });
  });

  sessions.sort((a, b) => b.started_at.localeCompare(a.started_at));
  return sessions;
}
