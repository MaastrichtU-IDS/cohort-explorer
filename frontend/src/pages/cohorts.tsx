'use client';

import React, {useState, useMemo, useEffect} from 'react';
import {useRouter} from 'next/router';
import {useCohorts} from '@/components/CohortsContext';
import FilterByMetadata from '@/components/FilterByMetadata';
import {Cohort} from '@/types';
import VariablesList from '@/components/VariablesList';
import {parseSearchQuery, searchInObject, highlightSearchTerms} from '@/utils/search';

// Helper component to render highlighted text
const HighlightedText = ({text, searchTerms, searchMode}: {text: string, searchTerms: string[], searchMode?: 'or' | 'and' | 'exact'}) => {
  const highlightedHtml = highlightSearchTerms(text, searchTerms, searchMode);
  
  if (highlightedHtml === text) {
    return <span>{text}</span>;
  }
  
  return <span dangerouslySetInnerHTML={{__html: highlightedHtml}} />;
};

// Component to count and display variable search results
const SearchResultsCounter = ({cohortsData, searchTerms, searchMode}: {
  cohortsData: Record<string, Cohort>, 
  searchTerms: string[], 
  searchMode: 'or' | 'and' | 'exact'
}) => {
  const results = useMemo(() => {
    let totalVariables = 0;
    let cohortsWithMatches = 0;
    
    Object.entries(cohortsData).forEach(([cohortId, cohortData]) => {
      let cohortMatchingVariables = 0;
      
      Object.entries(cohortData.variables || {}).forEach(([varName, varData]: any) => {
        const searchableFields = [
          'var_name', 'var_label', 'var_type', 'omop_domain', 'concept_code', 
          'concept_name', 'mapped_label', 'unit', 'stats_type'
        ];
        
        const variableWithName = { ...varData, var_name: varName };
        
        const matchesSearch = searchInObject(variableWithName, searchTerms, searchableFields, searchMode).matches ||
          // Also search in categories
          varData.categories?.some((category: any) => 
            searchInObject(category, searchTerms, ['value', 'label', 'mapped_label'], searchMode).matches
          );
        
        if (matchesSearch) {
          totalVariables++;
          cohortMatchingVariables++;
        }
      });
      
      // Only count cohort if it has at least one matching variable
      if (cohortMatchingVariables > 0) {
        cohortsWithMatches++;
      }
    });
    
    return { totalVariables, cohortsWithMatches };
  }, [cohortsData, searchTerms, searchMode]);
  
  return (
    <span>
      The search found <strong className="text-primary">{results.totalVariables}</strong> variable{results.totalVariables !== 1 ? 's' : ''} in <strong className="text-primary">{results.cohortsWithMatches}</strong> cohort{results.cohortsWithMatches !== 1 ? 's' : ''}
    </span>
  );
};

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
  const {cohortsData, userEmail, loadingMetrics, isLoading} = useCohorts();
  const [searchQuery, setSearchQuery] = useState('');
  
  // Check if we should use SPARQL mode based on query parameter
  const useSparqlMode = router.query.mode === 'sparql';
  const [selectedStudyTypes, setSelectedStudyTypes] = useState(new Set());
  const [selectedInstitutes, setSelectedInstitutes] = useState(new Set());
  // State to track which cohorts have aggregate data analysis available
  const [analysisAvailability, setAnalysisAvailability] = useState<{[key: string]: boolean}>({});
  // State to track which cohorts are expanded
  const [expandedCohorts, setExpandedCohorts] = useState<{[key: string]: boolean}>({});
  // Search configuration states
  const [searchScope, setSearchScope] = useState<'cohorts' | 'variables'>('cohorts');
  const [searchMode, setSearchMode] = useState<'or' | 'and' | 'exact'>('or');
  // selectedMorbidities state removed

  // TODO: debounce search to improve performance
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  // Function to toggle the expanded state for a cohort
  const toggleCohortExpanded = (cohortId: string) => {
    setExpandedCohorts((prev: Record<string, boolean>) => ({
      ...prev,
      [cohortId]: !prev[cohortId]
    }));
  };

  // Check for analysis folder availability when cohorts data changes
  useEffect(() => {
    const checkAnalysisAvailability = async () => {
      const availability: Record<string, boolean> = {};
      
      // Only proceed if we have cohorts data
      if (!cohortsData || Object.keys(cohortsData).length === 0) return;
      
      // Check each cohort for analysis folder availability
      for (const cohortId of Object.keys(cohortsData)) {
        try {
          const response = await fetch(`/api/check-analysis-folder/${cohortId}`);
          const data = await response.json();
          availability[cohortId] = data.exists;
        } catch (error) {
          console.error(`Error checking analysis for cohort ${cohortId}:`, error);
          availability[cohortId] = false;
        }
      }
      
      setAnalysisAvailability(availability);
    };
    
    checkAnalysisAvailability();
  }, [cohortsData]);

  // Parse search query into terms for search with word boundaries
  const searchTerms = useMemo(() => parseSearchQuery(searchQuery), [searchQuery]);

  // Filter cohorts based on search query and selected filters
  // TODO: we might want to perform the search and filtering directly with SPARQL queries to the oxigraph endpoint
  // if the data gets too big to be handled in the client.
  const filteredCohorts = useMemo(() => {
    return Object.entries(cohortsData as Record<string, Cohort>)
      .filter(([key, value]) => {
        let matchesSearchQuery = true;
        
        if (searchScope === 'cohorts') {
          // Search in cohort metadata only
          const searchableFields = [
            'cohort_id', 'institution', 'study_type', 'study_objective', 'morbidity',
            'study_participants', 'study_population', 'administrator', 'population_location',
            'primary_outcome_spec', 'secondary_outcome_spec'
          ];
          
          const cohortWithId = { ...value, cohort_id: key };
          matchesSearchQuery = searchInObject(cohortWithId, searchTerms, searchableFields, searchMode).matches;
        } else {
          // Search in variables only
          if (searchQuery.trim()) {
            matchesSearchQuery = Object.entries(value.variables || {}).some(([varName, varData]: any) => {
              const searchableFields = [
                'var_name', 'var_label', 'var_type', 'omop_domain', 'concept_code', 
                'concept_name', 'mapped_label', 'unit', 'stats_type'
              ];
              
              const variableWithName = { ...varData, var_name: varName };
              
              return searchInObject(variableWithName, searchTerms, searchableFields, searchMode).matches ||
                // Also search in categories
                varData.categories?.some((category: any) => 
                  searchInObject(category, searchTerms, ['value', 'label', 'mapped_label'], searchMode).matches
                );
            });
          }
        }

        // Apply other filters
        const matchesStudyType = selectedStudyTypes.size === 0 || selectedStudyTypes.has(value.study_type);
        const matchesInstitute = selectedInstitutes.size === 0 || selectedInstitutes.has(value.institution);
        
        return matchesSearchQuery && matchesStudyType && matchesInstitute;
      })
      .map(([, cohortData]) => cohortData);
  }, [searchTerms, searchMode, searchScope, selectedStudyTypes, selectedInstitutes, cohortsData, searchQuery]);
  // NOTE: filtering variables is done in VariablesList component

  // Function to toggle between cache and SPARQL modes
  const toggleDataSource = () => {
    const newMode = useSparqlMode ? undefined : 'sparql';
    router.push({
      pathname: router.pathname,
      query: newMode ? { mode: newMode } : {}
    });
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
        
        {/* Data source indicator and toggle - HIDDEN */}
        <div className="text-center mb-4 hidden">
          <div className="flex flex-col items-center gap-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Data Source:</span>
              <span className={`badge ${useSparqlMode ? 'badge-warning' : 'badge-success'}`}>
                {useSparqlMode ? 'SPARQL (Real-time)' : 'Cache (Fast)'}
              </span>
            </div>
            
            {/* Loading metrics display */}
            {isLoading ? (
              <div className="text-sm text-gray-600 flex items-center gap-2">
                <span className="loading loading-spinner loading-xs"></span>
                Loading data...
              </div>
            ) : loadingMetrics.loadTime !== null ? (
              <div className="text-xs text-gray-600 text-center">
                <div>Loaded in {loadingMetrics.loadTime}ms</div>
                <div>
                  {loadingMetrics.cohortCount} cohorts • {loadingMetrics.variableCount} variables • {loadingMetrics.categoryCount} categories
                </div>
                {loadingMetrics.sparqlRows && (
                  <div className="text-orange-600">
                    {loadingMetrics.sparqlRows} SPARQL result rows processed
                  </div>
                )}
              </div>
            ) : null}
            
            <button 
              onClick={toggleDataSource}
              className="btn btn-sm btn-outline btn-neutral"
              disabled={isLoading}
            >
              Switch to {useSparqlMode ? 'Cache' : 'SPARQL'}
            </button>
          </div>
        </div>
        {/* Filter by cohorts type removed */}
        <FilterByMetadata
          label="Filter by study design"
          metadata_id="study_type"
          options={Array.from(new Set(Object.values(cohortsData).map((cohort: any) => cohort.study_type)))}
          searchResults={filteredCohorts}
          onFiltersChange={(optionsSelected: any) => setSelectedStudyTypes(optionsSelected)}
        />
        {/* Filter by morbidity removed */}
        <FilterByMetadata
          label="Filter by providers"
          metadata_id="institution"
          options={Array.from(new Set(Object.values(cohortsData).map((cohort: any) => cohort.institution)))}
          searchResults={filteredCohorts}
          onFiltersChange={(optionsSelected: any) => setSelectedInstitutes(optionsSelected)}
        />
        {/* TODO: add by ongoing? */}
      </aside>

      <div className="w-full">
        <div className="mb-4">
          <input
            type="text"
            placeholder={searchScope === 'cohorts' ? "Search cohorts..." : "Search variables..."}
            className="input input-bordered w-full"
            value={searchQuery}
            onChange={handleSearchChange}
          />
          
          {/* Search Configuration Toggles */}
          <div className="flex flex-wrap gap-4 mt-3">
            {/* Search Scope Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Search in:</span>
              <div className="join">
                <button
                  className={`btn btn-sm join-item ${searchScope === 'cohorts' ? '' : 'btn-ghost'}`}
                  style={searchScope === 'cohorts' ? { backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' } : {}}
                  onClick={() => setSearchScope('cohorts')}
                >
                  {searchScope === 'cohorts' && '●'} Cohorts
                </button>
                <button
                  className={`btn btn-sm join-item ${searchScope === 'variables' ? '' : 'btn-ghost'}`}
                  style={searchScope === 'variables' ? { backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' } : {}}
                  onClick={() => setSearchScope('variables')}
                >
                  {searchScope === 'variables' && '●'} Variables
                </button>
              </div>
            </div>
            
            {/* Search Mode Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Mode:</span>
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
          
          {/* Search Results Counter */}
          {searchQuery.trim() && (
            <div className="mt-2 p-2 bg-base-200 rounded-lg text-sm">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-gray-600 dark:text-gray-400">🔍</span>
                  {searchScope === 'cohorts' ? (
                    <span>
                      Found <strong className="text-primary">{filteredCohorts.length}</strong> cohort{filteredCohorts.length !== 1 ? 's' : ''}
                      {filteredCohorts.length !== Object.keys(cohortsData).length && (
                        <span className="text-gray-500"> out of {Object.keys(cohortsData).length} total</span>
                      )}
                    </span>
                  ) : (
                    <SearchResultsCounter 
                      cohortsData={cohortsData}
                      searchTerms={searchTerms}
                      searchMode={searchMode}
                    />
                  )}
                </div>
                <button
                  onClick={() => setSearchQuery('')}
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
                
                {/* Only show Close button when cohort is expanded */}
                {expandedCohorts[cohortData.cohort_id] && (
                  <div className="flex justify-center mt-2">
                    <button 
                      onClick={(e: React.MouseEvent) => {
                        e.stopPropagation(); // Prevent event bubbling
                        toggleCohortExpanded(cohortData.cohort_id);
                      }} 
                      className="btn btn-sm btn-outline btn-neutral rounded-full px-4"
                    >
                      Close
                    </button>
                  </div>
                )}
              </div>
              <div className="collapse-content">
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
                </div>
                
                <VariablesList 
                  cohortId={cohortData.cohort_id} 
                  searchFilters={{
                    searchQuery: searchScope === 'variables' ? searchQuery : '', 
                    searchMode,
                    searchTerms
                  }} 
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
