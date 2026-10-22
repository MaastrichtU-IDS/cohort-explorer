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
  ai_votes: number;
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
        ? 'Found by the AI pass only; the lexical (text similarity) pass did not flag it'
        : source === 'both'
          ? 'Found by both the lexical pass and the AI pass'
          : 'Found by the lexical (text similarity) pass; the AI did not confirm it'
    }>
      {SOURCE_LABEL[source]}{strength && source !== 'lexical' ? ` · ${strength}` : ''}
    </span>
  );
}

/** The matches of one element, as a compact list. */
export function MatchList({matches, showField = true, aiOnly = false, emptyText = 'No matches found.'}: {
  matches: MetadataMatch[];
  showField?: boolean;
  aiOnly?: boolean;
  emptyText?: string;
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
          {m.ai_reason && (
            <p className="mt-1 text-xs text-base-content/70">
              <span className="font-medium">AI:</span> {m.ai_reason}
            </p>
          )}
        </li>
      ))}
    </ul>
  );
}
