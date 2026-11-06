'use client';

import React from 'react';
import Link from 'next/link';

/** One metadata element of a study (its study objective, a criterion, ...)
 *  with the matching elements found in other studies. Shape of
 *  /api/metadata-clustering/results. */
export interface MetadataMatch {
  element_id: string;
  cohort: string;
  field: string;
  label: string;
  text: string;
  source: 'lexical' | 'ai' | 'both';
  score: number;
  fuzzy: number;
  tfidf: number;
  same_field: boolean;
  ai_strength: 'strong' | 'partial' | null;
  ai_reason: string | null;
  /** The structured explanation (runs since Nov 2026): what both say, what differs, whether the match is in the content. */
  ai_reason_parts?: {shared?: string; differs?: string; basis?: 'content' | 'field'} | null;
  ai_votes: number;
  /** Raw tags from the pair pass, and the canonical theme ids chosen from the vocabulary. */
  themes?: string[];
  theme_ids?: string[];
  /** Content terms found in both texts by the deterministic pass, and the ones the model judged informative. */
  shared_terms?: string[] | null;
  useful_terms?: string[] | null;
  /** For outcome specifications: "primary outcome" | "secondary outcome" when both are, "outcome" when mixed. */
  outcome_kind?: string | null;
}

export interface MetadataElement {
  id: string;
  cohort: string;
  field: string;
  label: string;
  group: string;
  text: string;
  matches?: MetadataMatch[];
}

export const SOURCE_LABEL: Record<MetadataMatch['source'], string> = {
  both: 'Lexical + AI',
  lexical: 'Lexical',
  ai: 'AI only',
};

/** Badge for how a match was found. AI-only matches (not found by the
 *  text-based pass) are the ones to look at twice, so they stand out. */
export function SourceBadge({source, strength}: {source: MetadataMatch['source']; strength?: string | null}) {
  const cls =
    source === 'ai' ? 'badge-secondary' : source === 'both' ? 'badge-primary' : 'badge-ghost';
  return (
    <span className={`badge badge-sm ${cls} whitespace-nowrap`} title={
      source === 'ai'
        ? 'Found by the AI pass only: the text-similarity pass did not flag it (same-field pairs need 55% similarity, cross-field pairs 75%). The AI read the content and matched it anyway.'
        : source === 'both'
          ? 'Found by both the lexical pass and the AI pass'
          : 'Found by the lexical (text similarity) pass; the AI did not confirm it'
    }>
      {SOURCE_LABEL[source]}{strength && source !== 'lexical' ? ` · ${strength}` : ''}
    </span>
  );
}

/** The AI's explanation of a match: the structured form (shared / differs) when the
 *  run produced it, else the single sentence; then the themes, canonical labels when
 *  the vocabulary is at hand, raw tags otherwise. */
export function AiExplanation({m, themeLabels, className = ''}: {
  m: MetadataMatch;
  themeLabels?: Record<string, string>;
  className?: string;
}) {
  if (!m.ai_strength) return null;
  const parts = m.ai_reason_parts;
  const ids = m.theme_ids || [];
  const chips = themeLabels && ids.length
    ? ids.map(id => ({key: id, label: themeLabels[id] || id, canonical: true}))
    : (m.themes || []).map(t => ({key: t, label: t, canonical: false}));
  return (
    <div className={`text-xs ${className}`}>
      {parts?.shared ? (
        <>
          <p className="text-base-content/80"><span className="font-medium">Shared:</span> {parts.shared}</p>
          {parts.differs && !/^(nothing material|nothing|none|n\/a)\.?$/i.test(parts.differs.trim()) && (
            <p className="text-base-content/60 mt-0.5"><span className="font-medium">Differs:</span> {parts.differs}</p>
          )}
          {parts.basis === 'field' && (
            <p className="text-warning mt-0.5">Matched on the kind of field rather than on the content.</p>
          )}
        </>
      ) : m.ai_reason ? (
        <p className="text-base-content/70"><span className="font-medium">AI:</span> {m.ai_reason}</p>
      ) : null}
      {(chips.length > 0 || m.outcome_kind) && (
        <div className="flex flex-wrap gap-1 mt-1">
          {m.outcome_kind && (
            <span className="badge badge-xs badge-info badge-outline" title="Both elements are outcome specifications: primary, secondary, or one of each">
              {m.outcome_kind}
            </span>
          )}
          {chips.map(c => (
            <span key={c.key} className={`badge badge-xs ${c.canonical ? 'badge-outline' : 'badge-ghost'}`}
                  title={c.canonical ? 'Canonical theme from the vocabulary' : 'Raw tag from the pair pass (not yet assigned to the vocabulary)'}>
              {c.label}
            </span>
          ))}
        </div>
      )}
      {((m.useful_terms && m.useful_terms.length > 0) || (m.shared_terms && m.shared_terms.length > 0)) && (
        <div className="mt-0.5" title="Words and phrases found in both texts before the AI ran; struck through: judged non-informative by the AI">
          <span className="text-base-content/50">in both texts: </span>
          {(m.useful_terms || []).map(t => <span key={'u' + t} className="mr-1.5 text-base-content/70">{t}</span>)}
          {(m.shared_terms || []).filter(t => !(m.useful_terms || []).includes(t)).map(t => (
            <span key={'d' + t} className="mr-1.5 text-base-content/35 line-through">{t}</span>
          ))}
          {m.useful_terms === undefined && <span className="text-warning text-[10px]"> (not yet judged)</span>}
        </div>
      )}
    </div>
  );
}

/** The matches of one element, as a compact list. */
export function MatchList({matches, showField = true, aiOnly = false, emptyText = 'No matches found.', themeLabels}: {
  matches: MetadataMatch[];
  showField?: boolean;
  aiOnly?: boolean;
  emptyText?: string;
  themeLabels?: Record<string, string>;
}) {
  const rows = aiOnly ? matches.filter(m => m.source === 'ai') : matches;
  if (rows.length === 0) return <p className="text-sm text-base-content/60 italic">{emptyText}</p>;
  return (
    <ul className="space-y-2">
      {rows.map(m => (
        <li key={m.element_id} className="rounded-md border border-base-300 bg-base-100 p-2 text-sm">
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <Link href={`/cohorts?search=${encodeURIComponent(m.cohort)}`} className="font-semibold link link-hover">
              {m.cohort}
            </Link>
            {showField && (
              <span className={`text-xs ${m.same_field ? 'text-base-content/60' : 'text-warning font-medium'}`}
                    title={m.same_field ? 'Same metadata field' : 'Different metadata field in the other study'}>
                {m.label}{!m.same_field && ' (different field)'}
              </span>
            )}
            <SourceBadge source={m.source} strength={m.ai_strength} />
            <span className="text-xs text-base-content/50 ml-auto" title={`token-set ${m.fuzzy}, TF-IDF ${m.tfidf}`}>
              lexical {Math.round(m.score * 100)}%
            </span>
          </div>
          <p className="whitespace-pre-wrap">{m.text}</p>
          <AiExplanation m={m} themeLabels={themeLabels} className="mt-1" />
        </li>
      ))}
    </ul>
  );
}
