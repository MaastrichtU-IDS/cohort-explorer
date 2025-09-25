/**
 * Enhanced search utilities with word boundary matching, OR search, and highlighting
 */

export interface SearchResult {
  matches: boolean;
  highlightedText?: string;
}

/**
 * Create a regex pattern for word boundary matching
 * @param term - Search term
 * @returns RegExp for word boundary matching
 */
const createWordBoundaryRegex = (term: string): RegExp => {
  // Escape special regex characters
  const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  // Create word boundary regex (case insensitive)
  return new RegExp(`\\b${escapedTerm}`, 'gi');
};

/**
 * Check if text matches search terms with word boundaries
 * @param text - Text to search in
 * @param searchTerms - Array of search terms (OR logic)
 * @param exactPhrase - If true, search for exact phrase instead of OR logic
 * @returns true if any term matches with word boundaries (OR) or exact phrase matches
 */
export const matchesSearchTerms = (text: string, searchTerms: string[], exactPhrase: boolean = false): boolean => {
  if (!text || searchTerms.length === 0) return true;
  
  if (exactPhrase) {
    // For exact phrase, join terms back and search as single phrase
    const fullPhrase = searchTerms.join(' ');
    if (!fullPhrase.trim()) return true;
    const regex = createWordBoundaryRegex(fullPhrase.trim());
    return regex.test(text);
  }
  
  // OR logic - any term matches
  return searchTerms.some(term => {
    if (!term.trim()) return true;
    const regex = createWordBoundaryRegex(term.trim());
    return regex.test(text);
  });
};

/**
 * Highlight search terms in text
 * @param text - Text to highlight
 * @param searchTerms - Array of search terms to highlight
 * @param exactPhrase - If true, highlight the exact phrase instead of individual terms
 * @returns Text with highlighted terms wrapped in <mark> tags
 */
export const highlightSearchTerms = (text: string, searchTerms: string[], exactPhrase: boolean = false): string => {
  if (!text || searchTerms.length === 0) return text;
  
  let highlightedText = text;
  
  if (exactPhrase) {
    // Highlight the full phrase
    const fullPhrase = searchTerms.join(' ');
    if (fullPhrase.trim()) {
      const regex = createWordBoundaryRegex(fullPhrase.trim());
      highlightedText = highlightedText.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-600 px-1 py-0.5 rounded font-medium">$&</mark>');
    }
  } else {
    // Highlight individual terms (OR logic)
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
 * @param exactPhrase - If true, search for exact phrase instead of OR logic
 * @returns SearchResult with match status and highlighted content
 */
export const searchInObject = (
  obj: any, 
  searchTerms: string[], 
  searchableFields?: string[],
  exactPhrase: boolean = false
): SearchResult => {
  if (searchTerms.length === 0 || searchTerms.every(term => !term.trim())) {
    return { matches: true };
  }
  
  // If specific fields are provided, search only in those
  if (searchableFields) {
    const matches = searchableFields.some(field => {
      const value = obj[field];
      if (value != null) {
        return matchesSearchTerms(String(value), searchTerms, exactPhrase);
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
    
  const matches = matchesSearchTerms(searchableText, searchTerms, exactPhrase);
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
