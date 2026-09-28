export const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';

// Leading participant count of the spreadsheet's "Number of participants"
// field, always an integer: "6450", "6.450" / "6,450" / "6 450" (thousands
// separators; the dot is the European convention), or "73883 total
// participants; ..." (free text: its leading number). null when there is none.
export function parseParticipantCount(raw: unknown): number | null {
  if (raw === null || raw === undefined) return null;
  if (typeof raw === 'number') return Number.isFinite(raw) && raw > 0 ? Math.round(raw) : null;
  const m = /^\s*(\d{1,3}(?:[.,\s]\d{3})+(?!\d)|\d+)/.exec(String(raw));
  if (!m) return null;
  const n = Number(m[1].replace(/[.,\s]/g, ''));
  return Number.isFinite(n) && n > 0 ? n : null;
}
