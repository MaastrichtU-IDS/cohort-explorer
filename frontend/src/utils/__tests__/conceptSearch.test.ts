import {describe, expect, it} from 'vitest';

import {
  ACCEPTED_CONCEPT_DOMAINS,
  resolveConceptSuggestionState,
  resolveInitialConceptDomains
} from '@/utils/conceptSearch';

describe('concept-search domain contract', () => {
  it('normalizes the metadata Person domain to the provider-compatible value', () => {
    expect(ACCEPTED_CONCEPT_DOMAINS).toContain('Person');
    expect(resolveInitialConceptDomains('person')).toEqual(['Person']);
    expect(resolveInitialConceptDomains(' Person ')).toEqual(['Person']);
  });

  it('retains the all-domain fallback for unknown or absent metadata domains', () => {
    expect(resolveInitialConceptDomains('')).toEqual(ACCEPTED_CONCEPT_DOMAINS);
    expect(resolveInitialConceptDomains('custom domain')).toEqual(ACCEPTED_CONCEPT_DOMAINS);
  });
});

describe('concept-search result state', () => {
  it('distinguishes loading, empty, error, and populated results', () => {
    expect(resolveConceptSuggestionState({isLoading: false, hasSearched: false, errorMsg: '', resultCount: 0})).toBe(
      'idle'
    );
    expect(resolveConceptSuggestionState({isLoading: true, hasSearched: false, errorMsg: '', resultCount: 0})).toBe(
      'loading'
    );
    expect(resolveConceptSuggestionState({isLoading: false, hasSearched: true, errorMsg: '', resultCount: 0})).toBe(
      'empty'
    );
    expect(
      resolveConceptSuggestionState({isLoading: false, hasSearched: true, errorMsg: 'offline', resultCount: 0})
    ).toBe('error');
    expect(resolveConceptSuggestionState({isLoading: false, hasSearched: true, errorMsg: '', resultCount: 1})).toBe(
      'results'
    );
  });
});
