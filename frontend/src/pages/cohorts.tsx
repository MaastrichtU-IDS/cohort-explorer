'use client';

import React, {useState, useMemo, useEffect, useCallback} from 'react';
import {useRouter} from 'next/router';
import {useCohorts} from '@/components/CohortsContext';
import FilterByMetadata from '@/components/FilterByMetadata';
import {Cohort} from '@/types';
import VariablesList from '@/components/VariablesList';
import CohortSummaryGraphs from '@/components/CohortSummaryGraphs';
import {parseSearchQuery, searchInObject, highlightSearchTerms} from '@/utils/search';

// Helper component to render highlighted text
const HighlightedText = ({text, searchTerms, searchMode}: {text: string, searchTerms: string[], searchMode?: 'or' | 'and' | 'exact'}) => {
  const highlightedHtml = highlightSearchTerms(text, searchTerms, searchMode);
  
  if (highlightedHtml === text) {
    return <span>{text}</span>;
  }
  
  return <span dangerouslySetInnerHTML={{__html: highlightedHtml}} />;
};

// Helper function to check if text matches search terms (reusable)
const matchesSearchTerms = (text: string | null | undefined, searchTerms: string[], searchMode: 'or' | 'and' | 'exact'): boolean => {
  if (!text || searchTerms.length === 0) return false;
  const textLower = String(text).toLowerCase();
  
  if (searchMode === 'exact') {
    return textLower.includes(searchTerms.join(' ').toLowerCase());
  } else if (searchMode === 'and') {
    return searchTerms.every(term => textLower.includes(term.toLowerCase()));
  } else { // 'or' mode
    return searchTerms.some(term => textLower.includes(term.toLowerCase()));
  }
};

// Component to count and display variable search results
// Detailed search results component that shows matched names
const SearchResultsDisplay = React.memo(({cohortsData, searchTerms, searchMode, searchScope}: {
  cohortsData: Record<string, Cohort>,
  searchTerms: string[], 
  searchMode: 'or' | 'and' | 'exact',
  searchScope: 'cohorts' | 'variables' | 'all'
}) => {
  const results = useMemo(() => {
    if (searchTerms.length === 0) return { 
      matchedCohorts: [] as string[], 
      variablesByCohort: {} as Record<string, string[]>,
      totalVariables: 0 
    };
    
    const searchableCohortFields = [
      'cohort_id', 'institution', 'study_type', 'study_objective', 'morbidity',
      'study_participants', 'study_population', 'administrator', 'population_location',
      'primary_outcome_spec', 'secondary_outcome_spec'
    ];
    const searchableVarFields = ['var_name', 'var_label', 'concept_name', 'mapped_label', 'omop_domain', 'concept_code', 'omop_id'];
    const searchableCatFields = ['value', 'label', 'mapped_label'];
    
    const matchedCohorts: string[] = [];
    const variablesByCohort: Record<string, string[]> = {};
    let totalVariables = 0;
    
    Object.entries(cohortsData).forEach(([cohortId, cohortData]) => {
      // Check cohort metadata match (for 'cohorts' or 'all' scope)
      if (searchScope === 'cohorts' || searchScope === 'all') {
        const cohortWithId = { ...cohortData, cohort_id: cohortId };
        const cohortMatches = searchableCohortFields.some(field => 
          matchesSearchTerms((cohortWithId as any)[field], searchTerms, searchMode)
        );
        if (cohortMatches && !matchedCohorts.includes(cohortId)) {
          matchedCohorts.push(cohortId);
        }
      }
      
      // Check variable matches (for 'variables' or 'all' scope)
      if (searchScope === 'variables' || searchScope === 'all') {
        const matchingVars: string[] = [];
        
        Object.entries(cohortData.variables || {}).forEach(([varName, varData]: any) => {
          const variableWithName = { ...varData, var_name: varName };
          const varMatches = searchableVarFields.some(field => 
            matchesSearchTerms(variableWithName[field], searchTerms, searchMode)
          );
          
          const catMatches = !varMatches && varData.categories?.some((cat: any) =>
            searchableCatFields.some(field => matchesSearchTerms(cat[field], searchTerms, searchMode))
          );
          
          if (varMatches || catMatches) {
            matchingVars.push(varName);
            totalVariables++;
          }
        });
        
        if (matchingVars.length > 0) {
          variablesByCohort[cohortId] = matchingVars;
        }
      }
    });
    
    return { matchedCohorts, variablesByCohort, totalVariables };
  }, [cohortsData, searchTerms, searchMode, searchScope]);
  
  const cohortsWithVarMatches = Object.keys(results.variablesByCohort).length;
  
  // Format variable results: "var1, var2 (CohortA); var3 (CohortB)"
  const formatVariableResults = () => {
    return Object.entries(results.variablesByCohort)
      .map(([cohortId, vars]) => `${vars.join(', ')} (${cohortId})`)
      .join('; ');
  };
  
  if (searchScope === 'cohorts') {
    return (
      <div>
        <span>
          Search matched <strong className="text-primary">{results.matchedCohorts.length}</strong> cohort{results.matchedCohorts.length !== 1 ? 's' : ''} metadata
        </span>
        {results.matchedCohorts.length > 0 && (
          <div className="mt-1 text-xs text-gray-600 dark:text-gray-400">
            <strong>Cohorts:</strong> {results.matchedCohorts.join(', ')}
          </div>
        )}
      </div>
    );
  }
  
  if (searchScope === 'variables') {
    return (
      <div>
        <span>
          Search matched <strong className="text-primary">{results.totalVariables}</strong> variable description{results.totalVariables !== 1 ? 's' : ''} in <strong className="text-primary">{cohortsWithVarMatches}</strong> cohort{cohortsWithVarMatches !== 1 ? 's' : ''}
        </span>
        {results.totalVariables > 0 && (
          <div className="mt-1 text-xs text-gray-600 dark:text-gray-400 max-h-20 overflow-y-auto">
            <strong>Variables:</strong> {formatVariableResults()}
          </div>
        )}
      </div>
    );
  }
  
  // 'all' scope - show both
  return (
    <div>
      <span>
        Search matched <strong className="text-primary">{results.matchedCohorts.length}</strong> cohort{results.matchedCohorts.length !== 1 ? 's' : ''} metadata and <strong className="text-primary">{results.totalVariables}</strong> variable description{results.totalVariables !== 1 ? 's' : ''} in <strong className="text-primary">{cohortsWithVarMatches}</strong> cohort{cohortsWithVarMatches !== 1 ? 's' : ''}
      </span>
      {(results.matchedCohorts.length > 0 || results.totalVariables > 0) && (
        <div className="mt-1 text-xs text-gray-600 dark:text-gray-400 max-h-24 overflow-y-auto">
          {results.matchedCohorts.length > 0 && (
            <div><strong>Cohorts:</strong> {results.matchedCohorts.join(', ')}</div>
          )}
          {results.totalVariables > 0 && (
            <div><strong>Variables:</strong> {formatVariableResults()}</div>
          )}
        </div>
      )}
    </div>
  );
});

SearchResultsDisplay.displayName = 'SearchResultsDisplay';

// Helper function to format participants value for display in tags
const formatParticipantsForTag = (value: string | number | null | undefined): string => {
  if (!value) return '';
  
  // Convert to string if it's a number
  const strValue = String(value);
  
  // If it contains spaces, return only the first token
  if (strValue.includes(' ')) {
    return strValue.split(' ')[0];
  }
  
  return strValue;
};

export default function CohortsList() {
  const router = useRouter();
  const {cohortsData, userEmail, loadingMetrics, isLoading, fetchCohortsData, calculateStatistics} = useCohorts();
  const [searchQuery, setSearchQuery] = useState('');
  
  // Check if we should use SPARQL mode based on query parameter
  const useSparqlMode = router.query.mode === 'sparql';
  const [selectedStudyTypes, setSelectedStudyTypes] = useState(new Set());
  const [selectedInstitutes, setSelectedInstitutes] = useState(new Set());
  // State to track which cohorts have aggregate data analysis available
  const [analysisAvailability, setAnalysisAvailability] = useState<{[key: string]: boolean}>({});
  // State to track which cohorts are expanded
  const [expandedCohorts, setExpandedCohorts] = useState<{[key: string]: boolean}>({});
  // State to track which cohorts have collapsed metadata
  const [collapsedMetadata, setCollapsedMetadata] = useState<{[key: string]: boolean}>({});
  // Search configuration states
  const [searchScope, setSearchScope] = useState<'cohorts' | 'variables' | 'all'>('all');
  const [searchMode, setSearchMode] = useState<'or' | 'and' | 'exact'>('or');
  
  // Shared filter state for each cohort (synced between charts and sidebar)
  const [cohortFilters, setCohortFilters] = useState<{[cohortId: string]: {
    selectedOMOPDomains: Set<string>;
    selectedDataTypes: Set<string>;
    selectedCategoryTypes: Set<string>;
    selectedVisitTypes: Set<string>;
  }}>({});
  // State to track showOnlyOutcomes per cohort
  const [showOnlyOutcomes, setShowOnlyOutcomes] = useState<{[cohortId: string]: boolean}>({});
  // State to track variable counts per cohort
  const [variableCounts, setVariableCounts] = useState<{[cohortId: string]: {filtered: number, total: number}}>({});
  // selectedMorbidities state removed

  // Debounced search for better performance
  const [searchInput, setSearchInput] = useState('');
  
  // Debounce the search query update
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setSearchQuery(searchInput);
    }, 300); // 300ms debounce
    
    return () => clearTimeout(timeoutId);
  }, [searchInput]);
  
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchInput(event.target.value);
  };

  // Helper function to get filters for a cohort (initialize if doesn't exist)
  const getFiltersForCohort = (cohortId: string) => {
    if (!cohortFilters[cohortId]) {
      return {
        selectedOMOPDomains: new Set<string>(),
        selectedDataTypes: new Set<string>(),
        selectedCategoryTypes: new Set<string>(),
        selectedVisitTypes: new Set<string>(),
      };
    }
    return cohortFilters[cohortId];
  };

  // Helper function to toggle showOnlyOutcomes for a cohort
  const toggleShowOnlyOutcomes = useCallback((cohortId: string) => {
    setShowOnlyOutcomes(prev => ({
      ...prev,
      [cohortId]: !prev[cohortId]
    }));
  }, []);

  // Helper function to update variable counts for a cohort
  const updateVariableCounts = useCallback((cohortId: string, filtered: number, total: number) => {
    setVariableCounts(prev => {
      // Only update if values actually changed
      if (prev[cohortId]?.filtered === filtered && prev[cohortId]?.total === total) {
        return prev;
      }
      return {
        ...prev,
        [cohortId]: { filtered, total }
      };
    });
  }, []);

  // Helper function to update filters for a specific cohort
  const updateCohortFilters = (cohortId: string, filterType: 'selectedOMOPDomains' | 'selectedDataTypes' | 'selectedCategoryTypes' | 'selectedVisitTypes', value: Set<string>) => {
    setCohortFilters(prev => ({
      ...prev,
      [cohortId]: {
        ...getFiltersForCohort(cohortId),
        [filterType]: value
      }
    }));
  };

  // Helper function to reset all filters for a cohort
  const resetCohortFilters = (cohortId: string) => {
    setCohortFilters(prev => ({
      ...prev,
      [cohortId]: {
        selectedOMOPDomains: new Set<string>(),
        selectedDataTypes: new Set<string>(),
        selectedCategoryTypes: new Set<string>(),
        selectedVisitTypes: new Set<string>(),
      }
    }));
    // Also reset showOnlyOutcomes for this cohort
    setShowOnlyOutcomes(prev => ({
      ...prev,
      [cohortId]: false
    }));
  };

  // Helper functions to handle chart clicks (toggle values in Sets)
  const handleDomainClick = (cohortId: string, domain: string | null) => {
    const currentFilters = getFiltersForCohort(cohortId);
    const newDomains = new Set(currentFilters.selectedOMOPDomains);
    
    if (domain) {
      if (newDomains.has(domain)) {
        newDomains.delete(domain);
      } else {
        newDomains.add(domain);
      }
    }
    
    updateCohortFilters(cohortId, 'selectedOMOPDomains', newDomains);
  };

  const handleTypeClick = (cohortId: string, type: string | null) => {
    const currentFilters = getFiltersForCohort(cohortId);
    const newTypes = new Set(currentFilters.selectedDataTypes);
    
    if (type) {
      if (newTypes.has(type)) {
        newTypes.delete(type);
      } else {
        newTypes.add(type);
      }
    }
    
    updateCohortFilters(cohortId, 'selectedDataTypes', newTypes);
  };

  const handleCategoryClick = (cohortId: string, category: string | null) => {
    const currentFilters = getFiltersForCohort(cohortId);
    const newCategories = new Set(currentFilters.selectedCategoryTypes);
    
    if (category) {
      if (newCategories.has(category)) {
        newCategories.delete(category);
      } else {
        newCategories.add(category);
      }
    }
    
    updateCohortFilters(cohortId, 'selectedCategoryTypes', newCategories);
  };

  const handleVisitTypeClick = (cohortId: string, visitType: string | null) => {
    const currentFilters = getFiltersForCohort(cohortId);
    const newVisitTypes = new Set(currentFilters.selectedVisitTypes);
    
    if (visitType) {
      if (newVisitTypes.has(visitType)) {
        newVisitTypes.delete(visitType);
      } else {
        newVisitTypes.add(visitType);
      }
    }
    
    updateCohortFilters(cohortId, 'selectedVisitTypes', newVisitTypes);
  };

  // Function to toggle the expanded state for a cohort
  const toggleCohortExpanded = (cohortId: string) => {
    setExpandedCohorts((prev: Record<string, boolean>) => ({
      ...prev,
      [cohortId]: !prev[cohortId]
    }));
  };

  // Function to toggle metadata collapse for a cohort
  const toggleMetadataCollapsed = (cohortId: string) => {
    setCollapsedMetadata((prev: Record<string, boolean>) => ({
      ...prev,
      [cohortId]: !prev[cohortId]
    }));
  };

  // Check for analysis folder availability when cohorts data changes
  useEffect(() => {
    let isMounted = true;
    const abortController = new AbortController();
    
    const checkAnalysisAvailability = async () => {
      const availability: Record<string, boolean> = {};
      
      // Only proceed if we have cohorts data
      if (!cohortsData || Object.keys(cohortsData).length === 0) return;
      
      // Check each cohort for analysis folder availability
      for (const cohortId of Object.keys(cohortsData)) {
        if (!isMounted) break; // Stop if component unmounted
        
        try {
          const response = await fetch(`/api/check-analysis-folder/${cohortId}`, {
            signal: abortController.signal
          });
          const data = await response.json();
          availability[cohortId] = data.exists;
        } catch (error: any) {
          if (error.name === 'AbortError') {
            console.log('Analysis check aborted');
            break;
          }
          console.error(`Error checking analysis for cohort ${cohortId}:`, error);
          availability[cohortId] = false;
        }
      }
      
      if (isMounted) {
        setAnalysisAvailability(availability);
      }
    };
    
    checkAnalysisAvailability();
    
    return () => {
      isMounted = false;
      abortController.abort();
    };
  }, [cohortsData]);

  // Parse search query into terms - simple space-separated approach
  const searchTerms = useMemo(() => {
    if (!searchQuery) return [];
    // Simple split on spaces, trim, and filter out empty strings
    return searchQuery
      .split(' ')
      .map(term => term.trim())
      .filter(term => term.length > 0);
  }, [searchQuery]);

  // Filter cohorts based on metadata filters and search
  // - When searchScope='cohorts': filter by cohort metadata only
  // - When searchScope='variables': filter by variable matches only
  // - When searchScope='all': filter by either cohort metadata OR variable matches
  const filteredCohorts = useMemo(() => {
    const searchableCohortFields = [
      'cohort_id', 'institution', 'study_type', 'study_objective', 'morbidity',
      'study_participants', 'study_population', 'administrator', 'population_location',
      'primary_outcome_spec', 'secondary_outcome_spec'
    ];
    const searchableVarFields = ['var_name', 'var_label', 'concept_name', 'mapped_label', 'omop_domain', 'concept_code', 'omop_id'];
    const searchableCatFields = ['value', 'label', 'mapped_label'];
    
    // Helper to check if cohort metadata matches
    const cohortMetadataMatches = (key: string, value: Cohort) => {
      const cohortWithId = { ...value, cohort_id: key };
      return searchableCohortFields.some(field => 
        matchesSearchTerms((cohortWithId as any)[field], searchTerms, searchMode)
      );
    };
    
    // Helper to check if any variable matches
    const variableMatches = (value: Cohort) => {
      return Object.entries(value.variables || {}).some(([varName, varData]: any) => {
        const variableWithName = { ...varData, var_name: varName };
        
        const varMatches = searchableVarFields.some(field => 
          matchesSearchTerms(variableWithName[field], searchTerms, searchMode)
        );
        
        if (varMatches) return true;
        return varData.categories?.some((cat: any) =>
          searchableCatFields.some(field => matchesSearchTerms(cat[field], searchTerms, searchMode))
        );
      });
    };
    
    return Object.entries(cohortsData as Record<string, Cohort>)
      .filter(([key, value]) => {
        // Apply metadata filters (study type, institution)
        if (selectedStudyTypes.size > 0 && !selectedStudyTypes.has(value.study_type)) return false;
        if (selectedInstitutes.size > 0 && !selectedInstitutes.has(value.institution)) return false;
        
        // No search terms - show all cohorts
        if (searchTerms.length === 0) return true;
        
        if (searchScope === 'cohorts') {
          return cohortMetadataMatches(key, value);
        } else if (searchScope === 'variables') {
          return variableMatches(value);
        } else {
          // 'all' scope - match if either cohort metadata OR variables match
          return cohortMetadataMatches(key, value) || variableMatches(value);
        }
      })
      .map(([, cohortData]) => cohortData);
  }, [selectedStudyTypes, selectedInstitutes, cohortsData, searchScope, searchTerms, searchMode]);
  // NOTE: Filters work on ALL variables in the cohort, not just the search-matched ones.

  // Function to toggle between cache and SPARQL modes
  const toggleDataSource = () => {
    const newMode = useSparqlMode ? undefined : 'sparql';
    // Trigger loading state by refreshing the page with new mode
    // This will cause CohortsContext to refetch data
    window.location.href = newMode ? `${router.pathname}?mode=${newMode}` : router.pathname;
  };

  return (
    <>

      <main className="flex w-full p-4 bg-base-200 h-full min-h-screen space-x-4">
      <aside className="flex-shrink-0 w-64 flex flex-col">
        <div className="text-center mb-2">
          {filteredCohorts.length == Object.keys(cohortsData).length ? (
            <span className="badge badge-outline">{Object.keys(cohortsData).length} cohorts</span>
          ) : (
            <span className="badge badge-outline">
              {filteredCohorts.length}/{Object.keys(cohortsData).length} cohorts
            </span>
          )}
        </div>
        
        {/* Filter by cohorts type removed */}
        <FilterByMetadata
          label="Filter by study design"
          metadata_id="study_type"
          options={Array.from(new Set(Object.values(cohortsData).map((cohort: any) => cohort.study_type)))}
          searchResults={filteredCohorts}
          selectedValues={selectedStudyTypes}
          onFiltersChange={(optionsSelected: any) => setSelectedStudyTypes(optionsSelected)}
        />
        {/* Filter by morbidity removed */}
        <FilterByMetadata
          label="Filter by providers"
          metadata_id="institution"
          options={Array.from(new Set(Object.values(cohortsData).map((cohort: any) => cohort.institution)))}
          searchResults={filteredCohorts}
          selectedValues={selectedInstitutes}
          onFiltersChange={(optionsSelected: any) => setSelectedInstitutes(optionsSelected)}
        />
        {/* TODO: add by ongoing? */}
        
        {/* Data source indicator and toggle - hidden for now */}
        <div className="mt-6 pt-4 border-t border-base-300 hidden">
          <div className="text-sm font-medium mb-3 text-center">Data Source</div>
          <div className="flex flex-col items-center gap-3">
            <span className={`badge badge-sm ${useSparqlMode ? 'badge-warning' : 'badge-success'}`}>
              {useSparqlMode ? 'SPARQL (Up-to-date)' : 'Cache'}
            </span>
            
            {/* Loading metrics display */}
            {isLoading ? (
              <div className="text-sm text-gray-600 flex items-center gap-2">
                <span className="loading loading-spinner loading-sm"></span>
                Loading...
              </div>
            ) : loadingMetrics.loadTime !== null ? (
              <div className="text-xs text-gray-600 text-center space-y-1">
                <div className="font-medium">
                  Loaded in {Math.ceil(loadingMetrics.loadTime / 1000)}s
                </div>
                <div>
                  {loadingMetrics.cohortCount} cohorts • {loadingMetrics.variableCount} variables
                </div>
                {useSparqlMode ? (
                  loadingMetrics.sparqlRows && (
                    <div>
                      {loadingMetrics.sparqlRows.toLocaleString()} SPARQL rows
                    </div>
                  )
                ) : (
                  <div>
                    {(loadingMetrics.cohortCount + loadingMetrics.variableCount + loadingMetrics.categoryCount).toLocaleString()} cache objects
                  </div>
                )}
              </div>
            ) : null}
            
            {/* Buttons row */}
            <div className="flex gap-2 justify-center">
              <button 
                onClick={toggleDataSource}
                className="btn btn-xs btn-outline btn-neutral"
                disabled={isLoading}
              >
                Switch to {useSparqlMode ? 'Cache' : 'SPARQL'}
              </button>
            </div>
          </div>
        </div>
      </aside>

      <div className="w-full">
        <div className="mb-4">
          <input
            type="text"
            placeholder={searchScope === 'all' ? "Search cohorts and variables..." : searchScope === 'cohorts' ? "Search cohort metadata..." : "Search variable information..."}
            className="input input-bordered w-full"
            value={searchInput}
            onChange={handleSearchChange}
          />
          
          {/* Search Configuration Toggles */}
          <div className="flex flex-wrap gap-4 mt-3">
            {/* Search Scope Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">Search in:</span>
              <div className="join">
                <button
                  className={`btn btn-sm join-item ${searchScope === 'cohorts' ? '' : 'btn-ghost'}`}
                  style={searchScope === 'cohorts' ? { backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' } : {}}
                  onClick={() => setSearchScope('cohorts')}
                  title="Search only in cohort metadata (name, institution, study objective, etc.)"
                >
                  {searchScope === 'cohorts' && '●'} Cohorts Metadata
                </button>
                <button
                  className={`btn btn-sm join-item ${searchScope === 'variables' ? '' : 'btn-ghost'}`}
                  style={searchScope === 'variables' ? { backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' } : {}}
                  onClick={() => setSearchScope('variables')}
                  title="Search only in variable information (name, label, OMOP domain, etc.)"
                >
                  {searchScope === 'variables' && '●'} Variables Information
                </button>
                <button
                  className={`btn btn-sm join-item ${searchScope === 'all' ? '' : 'btn-ghost'}`}
                  style={searchScope === 'all' ? { backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' } : {}}
                  onClick={() => setSearchScope('all')}
                  title="Search in both cohort metadata and variable information"
                >
                  {searchScope === 'all' && '●'} All
                </button>
              </div>
            </div>
            
            {/* Search Mode Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">Mode:</span>
              <div className="join">
                <button
                  className={`btn btn-sm join-item ${searchMode === 'or' ? '' : 'btn-ghost'}`}
                  style={searchMode === 'or' ? { backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' } : {}}
                  onClick={() => setSearchMode('or')}
                  title="Search for any of the words (relaxed search)"
                >
                  {searchMode === 'or' && '●'} OR Search
                </button>
                <button
                  className={`btn btn-sm join-item ${searchMode === 'and' ? '' : 'btn-ghost'}`}
                  style={searchMode === 'and' ? { backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' } : {}}
                  onClick={() => setSearchMode('and')}
                  title="Search for all of the words (must contain all terms)"
                >
                  {searchMode === 'and' && '●'} AND Search
                </button>
                <button
                  className={`btn btn-sm join-item ${searchMode === 'exact' ? '' : 'btn-ghost'}`}
                  style={searchMode === 'exact' ? { backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' } : {}}
                  onClick={() => setSearchMode('exact')}
                  title="Search for the exact phrase"
                >
                  {searchMode === 'exact' && '●'} Exact Phrase
                </button>
              </div>
            </div>
          </div>
          
          {/* Search Results Display */}
          {searchInput.trim() && (
            <div className="mt-2 p-2 bg-base-200 rounded-lg text-sm">
              <div className="flex items-start gap-3">
                <span className="text-gray-600 dark:text-gray-400 mt-0.5">🔍</span>
                <div className="flex-1">
                  <SearchResultsDisplay 
                    cohortsData={cohortsData as Record<string, Cohort>}
                    searchTerms={searchTerms}
                    searchMode={searchMode}
                    searchScope={searchScope}
                  />
                </div>
                <button
                  onClick={() => {
                    setSearchInput('');
                    setSearchQuery('');
                  }}
                  className="btn btn-sm btn-outline btn-error hover:btn-error"
                  title="Clear search and show all results"
                >
                  ✕ Clear
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-2">
          {userEmail !== null && Object.keys(cohortsData).length === 0 && (
            <div className="flex flex-col items-center opacity-70 text-slate-500 mt-[20%]">
              <span className="loading loading-spinner loading-lg mb-4"></span>
              <p>Loading cohorts...</p>
            </div>
          )}
          {userEmail === null && (
            <p className="text-red-500 text-center mt-[20%]">Authenticate to access the explorer</p>
          )}
          {filteredCohorts.map((cohortData: Cohort) => (
            <div
              key={cohortData.cohort_id}
              className={`collapse card card-compact bg-base-100 shadow-xl ${!(Object.keys(cohortData.variables).length > 0) ? 'opacity-50' : ''}`}
            >
              <input 
                type="checkbox" 
                checked={expandedCohorts[cohortData.cohort_id] || false}
                onChange={() => toggleCohortExpanded(cohortData.cohort_id)}
                className="collapse-checkbox"
              />
              <div 
                className="collapse-title"
                onClick={() => toggleCohortExpanded(cohortData.cohort_id)}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <HighlightedText text={cohortData.cohort_id} searchTerms={searchTerms} searchMode={searchMode} />
                  <span className="badge badge-outline mx-2">{cohortData.institution}</span>
                  {cohortData.study_type && <span className="badge badge-ghost mx-1">{cohortData.study_type}</span>}
                  {cohortData.cohort_type && <span className="badge badge-ghost mx-1">{cohortData.cohort_type}</span>}
                  {/* Sex distribution moved to More Details section */}
                  {cohortData.study_duration && (
                    <span className="badge badge-default mx-1">⏱️ {cohortData.study_duration}</span>
                  )}
                  {cohortData.study_ongoing && cohortData.study_ongoing === 'yes' && (
                    <span className="badge badge-default mx-1">Ongoing study</span>
                  )}
                  {cohortData.study_ongoing && cohortData.study_ongoing === 'no' && (
                    <span className="badge badge-default mx-1">Completed study</span>
                  )}
                  {(cohortData.study_participants || cohortData.study_population) && (
                    <span className="badge badge-ghost mx-1">
                      👥 {formatParticipantsForTag(cohortData.study_participants)} {cohortData.study_population}
                    </span>
                  )}
                  {/* Display aggregate data analysis tag if available */}
                  {analysisAvailability[cohortData.cohort_id] && (
                    <span className="badge mx-1" style={{ backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' }}>
                      aggregate analysis added
                    </span>
                  )}
                  {/* Removed start date - end date tag as it's shown in the More Details section */}
                  {/* Removed contact email tags as they're shown in the More Details section */}
                </div>
                
              </div>
              <div className="collapse-content">
                {/* Control buttons - shown when cohort is expanded */}
                {expandedCohorts[cohortData.cohort_id] && (
                  <div className="flex justify-center gap-2 mb-4">
                    <button 
                      onClick={(e: React.MouseEvent) => {
                        e.stopPropagation();
                        toggleMetadataCollapsed(cohortData.cohort_id);
                      }} 
                      className="btn btn-sm btn-outline btn-neutral rounded-full px-4 opacity-80"
                    >
                      {collapsedMetadata[cohortData.cohort_id] ? 'Show Metadata' : 'Hide Metadata'}
                    </button>
                    <button 
                      onClick={(e: React.MouseEvent) => {
                        e.stopPropagation();
                        toggleCohortExpanded(cohortData.cohort_id);
                      }} 
                      className="btn btn-sm btn-outline btn-neutral rounded-full px-4"
                    >
                      Close
                    </button>
                  </div>
                )}
                
                {/* Metadata section - collapsible */}
                {!collapsedMetadata[cohortData.cohort_id] && (
                  <>
                {/* Display study objective section */}
                {cohortData.study_objective && (
                  <div className="mb-4 p-3 bg-base-200 rounded-lg">
                    <h3 className="font-bold mb-2">Study Objective:</h3>
                    <p>
                      <HighlightedText text={cohortData.study_objective} searchTerms={searchTerms} searchMode={searchMode} />
                    </p>
                  </div>
                )}
                
                {/* Display morbidity section */}
                {cohortData.morbidity && (
                  <div className="mb-4 p-3 bg-base-200 rounded-lg">
                    <h3 className="font-bold mb-2">Morbidity:</h3>
                    <p>
                      <HighlightedText text={cohortData.morbidity} searchTerms={searchTerms} searchMode={searchMode} />
                    </p>
                  </div>
                )}
                {/* Display outcome specifications section */}
                {(cohortData.primary_outcome_spec || cohortData.secondary_outcome_spec) && (
                  <div className="mb-4 p-3 bg-base-200 rounded-lg">
                    <h3 className="font-bold mb-2">Outcome Specifications:</h3>
                    {cohortData.primary_outcome_spec && (
                      <div className="mb-2">
                        <h4 className="font-semibold">Primary:</h4>
                        <p>
                          <HighlightedText text={cohortData.primary_outcome_spec} searchTerms={searchTerms} searchMode={searchMode} />
                        </p>
                      </div>
                    )}
                    {cohortData.secondary_outcome_spec && (
                      <div className="mb-2">
                        <h4 className="font-semibold">Secondary:</h4>
                        <p>
                          <HighlightedText text={cohortData.secondary_outcome_spec} searchTerms={searchTerms} searchMode={searchMode} />
                        </p>
                      </div>
                    )}
                  </div>
                )}
                
                {/* Display inclusion and exclusion criteria section */}
                <div className="p-4 bg-base-200 rounded-lg mb-4">
                  <div className="flex flex-row">
                    {/* Inclusion criteria - Left side */}
                    <div className="flex-1 pr-4 border-r border-gray-300">
                      <h4 className="font-semibold mb-2">Inclusion Criteria:</h4>
                      <div className="grid grid-cols-1 gap-2">
                        {cohortData.sex_inclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Sex: </span>
                            <span>{cohortData.sex_inclusion}</span>
                          </div>
                        )}
                        {cohortData.health_status_inclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Health Status: </span>
                            <span>{cohortData.health_status_inclusion}</span>
                          </div>
                        )}
                        {cohortData.clinically_relevant_exposure_inclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Clinically Relevant Exposure: </span>
                            <span>{cohortData.clinically_relevant_exposure_inclusion}</span>
                          </div>
                        )}
                        {cohortData.age_group_inclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Age Group: </span>
                            <span>{cohortData.age_group_inclusion}</span>
                          </div>
                        )}
                        {cohortData.bmi_range_inclusion && (
                          <div className="mb-1">
                            <span className="font-medium">BMI Range: </span>
                            <span>{cohortData.bmi_range_inclusion}</span>
                          </div>
                        )}
                        {cohortData.ethnicity_inclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Ethnicity: </span>
                            <span>{cohortData.ethnicity_inclusion}</span>
                          </div>
                        )}
                        {cohortData.family_status_inclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Family Status: </span>
                            <span>{cohortData.family_status_inclusion}</span>
                          </div>
                        )}
                        {cohortData.hospital_patient_inclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Hospital Patient: </span>
                            <span>{cohortData.hospital_patient_inclusion}</span>
                          </div>
                        )}
                        {cohortData.use_of_medication_inclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Use of Medication: </span>
                            <span>{cohortData.use_of_medication_inclusion}</span>
                          </div>
                        )}
                        {!cohortData.sex_inclusion && 
                         !cohortData.health_status_inclusion && 
                         !cohortData.clinically_relevant_exposure_inclusion && 
                         !cohortData.age_group_inclusion && 
                         !cohortData.bmi_range_inclusion && 
                         !cohortData.ethnicity_inclusion && 
                         !cohortData.family_status_inclusion && 
                         !cohortData.hospital_patient_inclusion && 
                         !cohortData.use_of_medication_inclusion && (
                          <p className="text-gray-500"><em>No inclusion criteria specified</em></p>
                        )}
                      </div>
                    </div>
                    
                    {/* Exclusion criteria - Right side */}
                    <div className="flex-1 pl-4">
                      <h4 className="font-semibold mb-2">Exclusion Criteria:</h4>
                      <div className="grid grid-cols-1 gap-2">
                        {cohortData.health_status_exclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Health Status: </span>
                            <span>{cohortData.health_status_exclusion}</span>
                          </div>
                        )}
                        {cohortData.bmi_range_exclusion && (
                          <div className="mb-1">
                            <span className="font-medium">BMI Range: </span>
                            <span>{cohortData.bmi_range_exclusion}</span>
                          </div>
                        )}
                        {cohortData.limited_life_expectancy_exclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Limited Life Expectancy: </span>
                            <span>{cohortData.limited_life_expectancy_exclusion}</span>
                          </div>
                        )}
                        {cohortData.need_for_surgery_exclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Need for Surgery: </span>
                            <span>{cohortData.need_for_surgery_exclusion}</span>
                          </div>
                        )}
                        {cohortData.surgical_procedure_history_exclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Surgical Procedure History: </span>
                            <span>{cohortData.surgical_procedure_history_exclusion}</span>
                          </div>
                        )}
                        {cohortData.clinically_relevant_exposure_exclusion && (
                          <div className="mb-1">
                            <span className="font-medium">Clinically Relevant Exposure: </span>
                            <span>{cohortData.clinically_relevant_exposure_exclusion}</span>
                          </div>
                        )}
                        {!cohortData.health_status_exclusion && 
                         !cohortData.bmi_range_exclusion && 
                         !cohortData.limited_life_expectancy_exclusion && 
                         !cohortData.need_for_surgery_exclusion && 
                         !cohortData.surgical_procedure_history_exclusion && 
                         !cohortData.clinically_relevant_exposure_exclusion && (
                          <p className="text-gray-500"><em>No exclusion criteria specified</em></p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Cohort Metadata Box */}
                <div className="bg-white shadow-md rounded-lg p-4 mb-4">
                  <h3 className="text-lg font-semibold mb-3 border-b pb-2">More Details</h3>
                  {/* Create an array of detail fields to render dynamically */}
                  {(() => {
                    // Define all possible detail fields with their labels and values
                    const detailFields = [
                      { label: 'Institute', value: cohortData.institution },
                      { label: 'Administrator', value: cohortData.administrator },
                      { label: 'Administrator Email', value: cohortData.administrator_email },
                      { label: 'Study Contact Person', value: cohortData.study_contact_person },
                      { label: 'Study Contact Person Email', value: cohortData.study_contact_person_email },
                      { label: 'Start Date', value: cohortData.study_start },
                      { label: 'End Date', value: cohortData.study_end },
                      { label: 'Population Location', value: cohortData.population_location },
                      { label: 'Language', value: cohortData.language },
                      { label: 'Number of Participants', value: cohortData.study_participants },
                      { label: 'Frequency of Data Collection', value: cohortData.data_collection_frequency },
                      { 
                        label: 'Sex Distribution', 
                        value: (cohortData.male_percentage !== null && cohortData.female_percentage !== null) ? 
                          `Male: ${cohortData.male_percentage}%, Female: ${cohortData.female_percentage}%` : null 
                      },
                      { label: 'Interventions', value: cohortData.interventions },
                      { 
                        label: 'References', 
                        value: cohortData.references && cohortData.references.length > 0 ? cohortData.references : null,
                        isReference: true 
                      },
                    ];
                    
                    // Helper function to check if a value is empty or empty-like
                    const isEmptyValue = (value: any): boolean => {
                      if (value === null || value === undefined) return true;
                      if (typeof value === 'string' && value.trim() === '') return true;
                      if (Array.isArray(value) && value.length === 0) return true;
                      return false;
                    };
                    
                    // Filter out fields with no values or empty values
                    const availableFields = detailFields.filter(field => !isEmptyValue(field.value));
                    
                    // Calculate the midpoint to split the array into two roughly equal parts
                    const midpoint = Math.ceil(availableFields.length / 2);
                    
                    // Split the array into two columns
                    const leftColumnFields = availableFields.slice(0, midpoint);
                    const rightColumnFields = availableFields.slice(midpoint);
                    
                    // Render the two columns
                    return (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Left Column */}
                        <div>
                          {leftColumnFields.map((field, index) => (
                            <div key={index} className="mb-2">
                              <span className="font-medium">{field.label}: </span>
                              {field.isReference ? (
                                <div>
                                  {(field.value as string[]).map((reference, refIndex) => (
                                    <div key={refIndex} className="mb-1">
                                      <a 
                                        href={reference.startsWith('http') ? reference : `https://${reference}`} 
                                        target="_blank" 
                                        rel="noopener noreferrer"
                                        className="text-blue-600 hover:underline"
                                      >
                                        {reference}
                                      </a>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <span>{field.value as string}</span>
                              )}
                            </div>
                          ))}
                        </div>
                        
                        {/* Right Column */}
                        <div>
                          {rightColumnFields.map((field, index) => (
                            <div key={index} className="mb-2">
                              <span className="font-medium">{field.label}: </span>
                              {field.isReference ? (
                                <div>
                                  {(field.value as string[]).map((reference, refIndex) => (
                                    <div key={refIndex} className="mb-1">
                                      <a 
                                        href={reference.startsWith('http') ? reference : `https://${reference}`} 
                                        target="_blank" 
                                        rel="noopener noreferrer"
                                        className="text-blue-600 hover:underline"
                                      >
                                        {reference}
                                      </a>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <span>{field.value as string}</span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })()}
                  {!cohortData.institution && 
                   !cohortData.administrator && 
                   !cohortData.administrator_email && 
                   !cohortData.study_contact_person && 
                   !cohortData.study_contact_person_email && 
                   !cohortData.study_start && 
                   !cohortData.study_end && 
                   (!cohortData.references || cohortData.references.length === 0) && 
                   !cohortData.population_location && 
                   !cohortData.language && 
                   !cohortData.data_collection_frequency && 
                   !(cohortData.male_percentage !== null && cohortData.female_percentage !== null) && 
                   !cohortData.study_participants && 
                   !cohortData.interventions && (
                    <p className="text-gray-500"><em>No metadata available</em></p>
                  )}
                  
                  {/* Visit Distribution Section */}
                  {(() => {
                    // Calculate visit distribution from variables
                    const visitCounts: { [key: string]: number } = {};
                    let totalWithVisits = 0;
                    
                    Object.values(cohortData.variables).forEach((variable) => {
                      if (variable.visit_concept_name && variable.visit_concept_name.trim()) {
                        // Capitalize first letter of each word for display
                        const visitValue = variable.visit_concept_name.trim()
                          .split(' ')
                          .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                          .join(' ');
                        visitCounts[visitValue] = (visitCounts[visitValue] || 0) + 1;
                        totalWithVisits++;
                      }
                    });
                    
                    // Sort by count descending
                    const sortedVisits = Object.entries(visitCounts)
                      .sort((a, b) => b[1] - a[1]);
                    
                    if (sortedVisits.length > 0) {
                      return (
                        <div className="mt-4 pt-4 border-t">
                          <h4 className="font-semibold mb-2">Visit Distribution:</h4>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                            {sortedVisits.map(([visit, count]) => (
                              <div key={visit} className="flex justify-between items-center bg-gray-50 px-3 py-2 rounded">
                                <span className="text-sm">{visit}</span>
                                <span className="badge badge-primary badge-sm">
                                  {count} {count === 1 ? 'variable' : 'variables'}
                                </span>
                              </div>
                            ))}
                          </div>
                          <p className="text-xs text-gray-500 mt-2">
                            {totalWithVisits} of {Object.keys(cohortData.variables).length} variables have visit information
                          </p>
                        </div>
                      );
                    }
                    return null;
                  })()}
                </div>
                  </>
                )}
                
                {/* Summary Graphs Section */}
                <CohortSummaryGraphs 
                  variables={cohortData.variables}
                  isExpanded={expandedCohorts[cohortData.cohort_id] || false}
                  cohortId={cohortData.cohort_id}
                  selectedOMOPDomains={getFiltersForCohort(cohortData.cohort_id).selectedOMOPDomains}
                  selectedDataTypes={getFiltersForCohort(cohortData.cohort_id).selectedDataTypes}
                  selectedCategoryTypes={getFiltersForCohort(cohortData.cohort_id).selectedCategoryTypes}
                  selectedVisitTypes={getFiltersForCohort(cohortData.cohort_id).selectedVisitTypes}
                  showOnlyOutcomes={showOnlyOutcomes[cohortData.cohort_id] || false}
                  filteredVariableCount={variableCounts[cohortData.cohort_id]?.filtered}
                  totalVariableCount={variableCounts[cohortData.cohort_id]?.total}
                  onDomainClick={(domain) => handleDomainClick(cohortData.cohort_id, domain)}
                  onTypeClick={(type) => handleTypeClick(cohortData.cohort_id, type)}
                  onCategoryClick={(category) => handleCategoryClick(cohortData.cohort_id, category)}
                  onVisitTypeClick={(visitType) => handleVisitTypeClick(cohortData.cohort_id, visitType)}
                  onResetFilters={() => resetCohortFilters(cohortData.cohort_id)}
                />
                
                <VariablesList 
                  cohortId={cohortData.cohort_id} 
                  searchFilters={{
                    searchQuery: searchScope === 'variables' ? searchQuery : '', 
                    searchMode,
                    searchTerms,
                    searchScope
                  }}
                  selectedOMOPDomains={getFiltersForCohort(cohortData.cohort_id).selectedOMOPDomains}
                  selectedDataTypes={getFiltersForCohort(cohortData.cohort_id).selectedDataTypes}
                  selectedCategoryTypes={getFiltersForCohort(cohortData.cohort_id).selectedCategoryTypes}
                  selectedVisitTypes={getFiltersForCohort(cohortData.cohort_id).selectedVisitTypes}
                  showOnlyOutcomes={showOnlyOutcomes[cohortData.cohort_id] || false}
                  onOMOPDomainsChange={(domains: Set<string>) => updateCohortFilters(cohortData.cohort_id, 'selectedOMOPDomains', domains)}
                  onDataTypesChange={(types: Set<string>) => updateCohortFilters(cohortData.cohort_id, 'selectedDataTypes', types)}
                  onCategoryTypesChange={(categories: Set<string>) => updateCohortFilters(cohortData.cohort_id, 'selectedCategoryTypes', categories)}
                  onVisitTypesChange={(visitTypes: Set<string>) => updateCohortFilters(cohortData.cohort_id, 'selectedVisitTypes', visitTypes)}
                  onShowOnlyOutcomesChange={(value: boolean) => toggleShowOnlyOutcomes(cohortData.cohort_id)}
                  onVariableCountsChange={(filtered: number, total: number) => updateVariableCounts(cohortData.cohort_id, filtered, total)}
                  onResetFilters={() => resetCohortFilters(cohortData.cohort_id)}
                  onCloseCohort={() => setExpandedCohorts(prev => ({...prev, [cohortData.cohort_id]: false}))}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
    </>
  );
}
