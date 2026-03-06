/**
 * Enhanced search utilities with word boundary matching, OR search, and highlighting
 */

export interface SearchResult {
  matches: boolean;
  highlightedText?: string;
}

/**
 * Normalize text for fuzzy matching: split camelCase and replace common separators with spaces
 */
const normalizeText = (text: string): string =>
  text
    .replace(/([a-z])([A-Z])/g, '$1 $2')   // camelCase: aB → a B
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2') // acronym boundary: ABc → A Bc
    .replace(/[_\-.,—]/g, ' ');

/**
 * Create a regex pattern for word boundary matching
 * Treats common punctuation as word boundaries so e.g. "body_mass" matches "body mass"
 * @param term - Search term
 * @returns RegExp for word boundary matching
 */
const createWordBoundaryRegex = (term: string): RegExp => {
  // Normalize punctuation in the term to spaces, then split into sub-terms
  const normalized = normalizeText(term);
  // Escape special regex characters in each part
  const parts = normalized.split(/\s+/).filter(p => p.length > 0);
  // Join with a pattern that matches any word boundary or common punctuation
  const pattern = parts.map(p => p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('[_\\-.,—\\s]*');
  return new RegExp(`\\b${pattern}`, 'gi');
};

/**
 * Check if text matches search terms with word boundaries
 * @param text - Text to search in
 * @param searchTerms - Array of search terms
 * @param searchMode - 'or' | 'and' | 'exact' - determines matching logic
 * @returns true if terms match according to the specified mode
 */
export const matchesSearchTerms = (text: string, searchTerms: string[], searchMode: 'or' | 'and' | 'exact' = 'or'): boolean => {
  if (!text || searchTerms.length === 0) return true;
  
  // Normalize punctuation in text for matching
  const textNorm = normalizeText(text);
  
  if (searchMode === 'exact') {
    // For exact phrase, join terms back and search as single phrase
    const fullPhrase = searchTerms.join(' ');
    if (!fullPhrase.trim()) return true;
    const regex = createWordBoundaryRegex(fullPhrase.trim());
    return regex.test(text) || regex.test(textNorm);
  }
  
  if (searchMode === 'and') {
    // AND logic - all terms must match
    return searchTerms.every(term => {
      if (!term.trim()) return true;
      const regex = createWordBoundaryRegex(term.trim());
      return regex.test(text) || regex.test(textNorm);
    });
  }
  
  // OR logic - any term matches (default)
  return searchTerms.some(term => {
    if (!term.trim()) return true;
    const regex = createWordBoundaryRegex(term.trim());
    return regex.test(text) || regex.test(textNorm);
  });
};

/**
 * Highlight search terms in text
 * @param text - Text to highlight
 * @param searchTerms - Array of search terms to highlight
 * @param searchMode - 'or' | 'and' | 'exact' - determines highlighting logic
 * @returns Text with highlighted terms wrapped in <mark> tags
 */
export const highlightSearchTerms = (text: string, searchTerms: string[], searchMode: 'or' | 'and' | 'exact' = 'or'): string => {
  if (!text || searchTerms.length === 0) return text;
  
  let highlightedText = text;
  
  if (searchMode === 'exact') {
    // Highlight the full phrase
    const fullPhrase = searchTerms.join(' ');
    if (fullPhrase.trim()) {
      const regex = createWordBoundaryRegex(fullPhrase.trim());
      highlightedText = highlightedText.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-600 px-1 py-0.5 rounded font-medium">$&</mark>');
    }
  } else {
    // Highlight individual terms (both OR and AND logic highlight individual terms)
    searchTerms.forEach(term => {
      if (!term.trim()) return;
      const regex = createWordBoundaryRegex(term.trim());
      highlightedText = highlightedText.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-600 px-1 py-0.5 rounded font-medium">$&</mark>');
    });
  }
  
  return highlightedText;
};

/**
 * Search in object fields with word boundaries and highlighting
 * @param obj - Object to search in
 * @param searchTerms - Array of search terms
 * @param searchableFields - Specific fields to search in (optional)
 * @param searchMode - 'or' | 'and' | 'exact' - determines matching logic
 * @returns SearchResult with match status and highlighted content
 */
export const searchInObject = (
  obj: any, 
  searchTerms: string[], 
  searchableFields?: string[],
  searchMode: 'or' | 'and' | 'exact' = 'or'
): SearchResult => {
  if (searchTerms.length === 0 || searchTerms.every(term => !term.trim())) {
    return { matches: true };
  }
  
  // If specific fields are provided, search only in those
  if (searchableFields) {
    const matches = searchableFields.some(field => {
      const value = obj[field];
      if (value != null) {
        return matchesSearchTerms(String(value), searchTerms, searchMode);
      }
      return false;
    });
    return { matches };
  }
  
  // Otherwise, search in all string fields
  const searchableText = Object.values(obj)
    .filter(value => value != null)
    .map(value => String(value))
    .join(' ');
    
  const matches = matchesSearchTerms(searchableText, searchTerms, searchMode);
  return { matches };
};

/**
 * Parse search query into terms (split by space)
 * @param query - Search query string
 * @returns Array of search terms
 */
export const parseSearchQuery = (query: string): string[] => {
  return query
    .trim()
    .split(/\s+/)
    .filter(term => term.length > 0);
};

/**
 * Create a highlighted version of a field value
 * @param value - Field value to highlight
 * @param searchTerms - Search terms to highlight
 * @returns Object with original value and highlighted HTML
 */
export const createHighlightedField = (value: any, searchTerms: string[]) => {
  if (value == null) return { original: value, highlighted: value };
  
  const stringValue = String(value);
  const highlighted = highlightSearchTerms(stringValue, searchTerms);
  
  return {
    original: value,
    highlighted: highlighted !== stringValue ? highlighted : null
  };
};
