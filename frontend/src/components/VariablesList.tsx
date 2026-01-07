import React, {useState, useMemo, useEffect} from 'react';
import {useCohorts} from '@/components/CohortsContext';
import AutocompleteConcept from '@/components/AutocompleteConcept';
import FilterByMetadata from '@/components/FilterByMetadata';
import VariableGraphModal from '@/components/VariableGraphModal';
import {InfoIcon} from '@/components/Icons';
import {Concept, Variable} from '@/types';
import {apiUrl} from '@/utils';
import {parseSearchQuery, searchInObject, highlightSearchTerms} from '@/utils/search';

// Helper component to render highlighted text
const HighlightedText = ({text, searchTerms, searchMode}: {text: string, searchTerms: string[], searchMode?: 'or' | 'and' | 'exact'}) => {
  const highlightedHtml = highlightSearchTerms(text, searchTerms, searchMode);
  
  if (highlightedHtml === text) {
    return <span>{text}</span>;
  }
  
  return <span dangerouslySetInnerHTML={{__html: highlightedHtml}} />;
};

interface VariablesListProps {
  cohortId: string;
  searchFilters?: {
    searchQuery?: string;
    searchMode?: 'or' | 'and' | 'exact';
    searchTerms?: string[];
  };
  selectedOMOPDomains: Set<string>;
  selectedDataTypes: Set<string>;
  selectedCategoryTypes: Set<string>;
  selectedVisitTypes: Set<string>;
  showOnlyOutcomes?: boolean;
  onOMOPDomainsChange: (domains: Set<string>) => void;
  onDataTypesChange: (types: Set<string>) => void;
  onCategoryTypesChange: (categories: Set<string>) => void;
  onVisitTypesChange: (visitTypes: Set<string>) => void;
  onShowOnlyOutcomesChange?: (value: boolean) => void;
  onVariableCountsChange?: (filtered: number, total: number) => void;
  onResetFilters: () => void;
}

const VariablesList = ({
  cohortId, 
  searchFilters = {searchQuery: ''}, 
  selectedOMOPDomains,
  selectedDataTypes,
  selectedCategoryTypes,
  selectedVisitTypes,
  showOnlyOutcomes = false,
  onOMOPDomainsChange,
  onDataTypesChange,
  onCategoryTypesChange,
  onVisitTypesChange,
  onShowOnlyOutcomesChange,
  onVariableCountsChange,
  onResetFilters
}: VariablesListProps) => {
  const {cohortsData, updateCohortData, dataCleanRoom, setDataCleanRoom} = useCohorts();
  const [openedModal, setOpenedModal] = useState('');
  const [openedGraphModal, setOpenedGraphModal] = useState<string | null>(null);

  // When concept is selected, insert the triples into the database
  const handleConceptSelect = (varId: any, concept: Concept, categoryId: any = null) => {
    // console.log(`Selected concept for ${varId}:`, concept);
    const updatedCohortData = {...cohortsData[cohortId]};
    // TODO: remove, should be handled in API
    // const vocab = concept.vocabulary.toLowerCase() === 'snomed' ? 'snomedct' : concept.vocabulary.toLowerCase();
    const formData = new FormData();
    formData.append('cohort_id', cohortId);
    formData.append('var_id', varId);
    formData.append('predicate', 'icare:mappedId');
    formData.append('value', concept.id);
    formData.append('label', concept.label);
    if (categoryId !== null) {
      formData.append('category_id', categoryId);
      updatedCohortData.variables[varId]['categories'][categoryId]['mapped_id'] = concept.id;
      updatedCohortData.variables[varId]['categories'][categoryId]['mapped_label'] = concept.label;
    } else {
      updatedCohortData.variables[varId]['mapped_id'] = concept.id;
      updatedCohortData.variables[varId]['mapped_label'] = concept.label;
    }
    updateCohortData(cohortId, updatedCohortData);
    fetch(`${apiUrl}/insert-triples`, {
      method: 'POST',
      body: formData,
      credentials: 'include'
    })
      .then(response => response.json())
      .then(data => {
        console.log('Triples for concept mapping inserted', updatedCohortData);
      });
  };

  // Button to add cohort to data clean room
  const addToDataCleanRoom = (var_name: string | null = null) => {
    const updatedDcr = {...dataCleanRoom};
    console.log('updatedDcr!', updatedDcr);
    // updatedDcr.cohorts[cohortId] = cohortsData[cohortId]['variables'].map((variable: Variable) => variable.var_name);
    // Add all variables to the DCR for this cohort
    if (var_name) {
      if (!updatedDcr.cohorts[cohortId]) {
        updatedDcr.cohorts[cohortId] = [];
      }
      updatedDcr.cohorts[cohortId].push(var_name);
    } else {
      updatedDcr.cohorts[cohortId] = Object.entries(cohortsData[cohortId]['variables']).map(
        ([variableName, variableData]: any) => variableName
      );
    }
    setDataCleanRoom(updatedDcr);
    sessionStorage.setItem('dataCleanRoom', JSON.stringify(updatedDcr));
  };

  // Button to remove cohort or variable from data clean room
  const removeFromDataCleanRoom = (var_name: string | null = null) => {
    const updatedDcr = {...dataCleanRoom};
    if (var_name) {
      // Remove specific variable
      if (updatedDcr.cohorts[cohortId]) {
        updatedDcr.cohorts[cohortId] = updatedDcr.cohorts[cohortId].filter(
          (v: string) => v !== var_name
        );
        // If no variables left, remove the cohort entry
        if (updatedDcr.cohorts[cohortId].length === 0) {
          delete updatedDcr.cohorts[cohortId];
        }
      }
    } else {
      // Remove entire cohort
      delete updatedDcr.cohorts[cohortId];
    }
    setDataCleanRoom(updatedDcr);
    sessionStorage.setItem('dataCleanRoom', JSON.stringify(updatedDcr));
  };

  // Collect unique OMOP domains, data types, and visit types from variables for filtering options
  const omopDomains = new Set();
  const dataTypes: any = new Set();
  const visitTypes: any = new Set();
  Object.values(cohortsData[cohortId]?.variables || {}).forEach((variable: any) => {
    omopDomains.add(variable.omop_domain);
    dataTypes.add(variable.var_type);
    if (variable.visits) {
      visitTypes.add(variable.visits);
    }
  });

  // Get search configuration from props - simple space-separated terms
  const searchTerms = useMemo(() => {
    if (searchFilters.searchTerms) {
      return searchFilters.searchTerms;
    }
    if (searchFilters.searchQuery) {
      // Simple split on spaces, trim, and filter out empty strings
      return searchFilters.searchQuery
        .split(' ')
        .map((term: string) => term.trim())
        .filter((term: string) => term.length > 0);
    }
    return [];
  }, [searchFilters.searchTerms, searchFilters.searchQuery]);
  
  const searchMode = searchFilters.searchMode || 'or';

  // Filter variables based on search query and selected filters
  const filteredVars = useMemo(() => {
    if (cohortsData && cohortsData[cohortId]) {
      return Object.entries(cohortsData[cohortId]['variables'])
        .filter(([variableName, variableData]: any) => {
          // Enhanced search with word boundaries and configurable logic
          // Only search in fields that contain actual variable content, not metadata
          const searchableFields = [
            'var_name', 'var_label', 'concept_name', 'mapped_label', 'omop_domain', 'concept_code', 'omop_id'
          ];
          
          // Add variable name to the data for searching
          const variableWithName = { ...variableData, var_name: variableName };
          
          // Only filter by search if there are actual search terms
          if (searchTerms.length === 0 || searchTerms.every((term: string) => !term.trim())) {
            return true; // No search terms, show all variables
          }
          
          // Simple direct search implementation for better control
          let variableMatches = false;
          
          // Check each searchable field directly
          for (const field of searchableFields) {
            const fieldValue = variableWithName[field];
            if (fieldValue != null) {
              const fieldText = String(fieldValue).toLowerCase();
              
              if (searchMode === 'exact') {
                const fullPhrase = searchTerms.join(' ').toLowerCase();
                if (fieldText.includes(fullPhrase)) {
                  variableMatches = true;
                  break;
                }
              } else if (searchMode === 'and') {
                if (searchTerms.every((term: string) => fieldText.includes(term.toLowerCase()))) {
                  variableMatches = true;
                  break;
                }
              } else { // 'or' mode
                if (searchTerms.some((term: string) => fieldText.includes(term.toLowerCase()))) {
                  variableMatches = true;
                  break;
                }
              }
            }
          }
          
          // Check categories
          let categoryMatches = false;
          if (variableData.categories) {
            for (const category of variableData.categories) {
              const categoryFields = ['value', 'label', 'mapped_label'];
              for (const field of categoryFields) {
                const fieldValue = category[field];
                if (fieldValue != null) {
                  const fieldText = String(fieldValue).toLowerCase();
                  
                  if (searchMode === 'exact') {
                    const fullPhrase = searchTerms.join(' ').toLowerCase();
                    if (fieldText.includes(fullPhrase)) {
                      categoryMatches = true;
                      break;
                    }
                  } else if (searchMode === 'and') {
                    if (searchTerms.every((term: string) => fieldText.includes(term.toLowerCase()))) {
                      categoryMatches = true;
                      break;
                    }
                  } else { // 'or' mode
                    if (searchTerms.some((term: string) => fieldText.includes(term.toLowerCase()))) {
                      categoryMatches = true;
                      break;
                    }
                  }
                }
              }
              if (categoryMatches) break;
            }
          }
          
          const matchesSearch = variableMatches || categoryMatches;
          
          return matchesSearch;
        })
        .filter(
          ([variableName, variableData]: any) => {
            // Filter by outcome keywords if enabled
            if (showOnlyOutcomes) {
              const outcomeKeywords = ['outcome', 'endpoint', 'end point'];
              const searchableFields = ['var_name', 'var_label', 'concept_name', 'mapped_label', 'omop_domain', 'concept_code', 'omop_id'];
              
              // Add variable name to the data for searching
              const variableWithName = { ...variableData, var_name: variableName };
              
              // Check if any searchable field contains any outcome keyword
              let hasOutcomeKeyword = false;
              for (const field of searchableFields) {
                const fieldValue = variableWithName[field];
                if (fieldValue != null) {
                  const fieldText = String(fieldValue).toLowerCase();
                  if (outcomeKeywords.some(keyword => fieldText.includes(keyword))) {
                    hasOutcomeKeyword = true;
                    break;
                  }
                }
              }
              
              if (!hasOutcomeKeyword) return false;
            }
            
            // Apply other filters
            const passesOMOPFilter = selectedOMOPDomains.size === 0 || selectedOMOPDomains.has(variableData.omop_domain);
            const passesDataTypeFilter = selectedDataTypes.size === 0 || selectedDataTypes.has(variableData.var_type);
            const passesVisitTypeFilter = selectedVisitTypes.size === 0 || selectedVisitTypes.has(variableData.visits);
            
            // Category filter logic
            let passesCategoryFilter = true;
            if (selectedCategoryTypes.size > 0) {
              const catCount = variableData.categories.length;
              passesCategoryFilter = false;
              if (selectedCategoryTypes.has('Non-categorical') && catCount === 0) passesCategoryFilter = true;
              if (selectedCategoryTypes.has('All categorical') && catCount > 0) passesCategoryFilter = true;
              if (selectedCategoryTypes.has('2 categories') && catCount === 2) passesCategoryFilter = true;
              if (selectedCategoryTypes.has('3 categories') && catCount === 3) passesCategoryFilter = true;
              if (selectedCategoryTypes.has('4+ categories') && catCount >= 4) passesCategoryFilter = true;
            }
            
            return passesOMOPFilter && passesDataTypeFilter && passesCategoryFilter && passesVisitTypeFilter;
          }
        )
        .map(([variableName, variableData]: any) => ({...variableData, var_name: variableName}));
    } else {
      return [];
    }
  }, [
    cohortsData,
    cohortId,
    searchTerms,
    searchMode,
    selectedOMOPDomains,
    selectedDataTypes,
    selectedCategoryTypes,
    selectedVisitTypes,
    showOnlyOutcomes
  ]);

  // Report variable counts to parent - only when they actually change
  useEffect(() => {
    if (onVariableCountsChange && cohortsData[cohortId]) {
      const totalCount = Object.keys(cohortsData[cohortId].variables).length;
      const filteredCount = filteredVars.length;
      onVariableCountsChange(filteredCount, totalCount);
    }
  }, [filteredVars.length, cohortId, onVariableCountsChange]);

  // Function to handle downloading the cohort CSV
  const downloadMetadataCSV = async () => {
    const downloadUrl = `${apiUrl}/cohort-spreadsheet/${encodeURIComponent(cohortId)}`;
    // Create a temporary anchor element and trigger a download
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `${cohortId}-datadictionary.csv`; // Downloaded file name
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
};


  const handleCloseGraphModal = () => {
    setOpenedGraphModal(null);
  };

  // Function to count filtered vars based on filter type
  // const countMatches = (filterType: string, item: string | null) => {
  //   return filteredVars.filter(variable => {
  //     if (filterType === 'categorical') {
  //       return variable.categories.length > 0;
  //     } else if (filterType === 'non_categorical') {
  //       return variable.categories.length === 0;
  //     } else {
  //       return variable[filterType] === item;
  //     }
  //   }).length;
  // };

  return (
    <main className="flex w-full space-x-4">
      <aside className="flex-shrink-0 text-center flex flex-col items-center w-52">
        {Object.keys(cohortsData[cohortId]['variables']).length > 0 && (
          dataCleanRoom.cohorts[cohortId] &&
          dataCleanRoom.cohorts[cohortId].length === Object.keys(cohortsData[cohortId]['variables']).length ? (
            <button
              onClick={() => removeFromDataCleanRoom()}
              className="btn btn-sm mb-2 bg-gray-200 hover:bg-gray-300 text-gray-700 tooltip tooltip-right"
              data-tip={`Remove all variables of the cohort ${cohortId} from your Data Clean Room`}
            >
              Remove from DCR
            </button>
          ) : (
            <button
              onClick={() => addToDataCleanRoom()}
              className="btn btn-neutral btn-sm mb-2 hover:bg-slate-600 tooltip tooltip-right"
              data-tip={`Add all variables of the cohort ${cohortId} to your Data Clean Room`}
            >
              Add to DCR
            </button>
          )
        )}
        {filteredVars.length > 0 && (
          <button
            onClick={downloadMetadataCSV}
            className="btn btn-neutral btn-sm mb-2 hover:bg-slate-600 tooltip tooltip-right"
            data-tip="Download the cohort metadata as CSV file. You can edit and re-upload it, if you are the person who published it originally."
          >
            Download Metadata
          </button>
        )}
        {/* NOTE: You will need to create an API endpoint just to ddl the imgs for the cohort stats
        If the <img> I set up dont work, then you'll need to use a fetch call
        e.g. const response = await fetch(`${apiUrl}/cohort-stats/${cohortId}`, {credentials: 'include'});
        You could also add "stats_available=True/False" field when sending the list of variables from the API
        This way you only show the "Cohort stats" button for cohorts where you have stats
        cohortsData[cohortId].stats_available to check if the stats are available
        */}
        {/*<button
        onClick={() => {
          const modal = document.getElementById('stats_modal') as HTMLDialogElement;
          modal?.showModal();
        }}
        className="btn btn-neutral btn-sm mb-2 hover:bg-slate-600 tooltip tooltip-right"
        data-tip="View cohort statistics"
        >
        Cohort stats
       </button> */}
        <dialog id="stats_modal" className="modal">
          <div className="modal-box max-w-5xl">
            <h3 className="font-bold text-lg mb-4">{cohortId} Cohort Statistics</h3>
            <img
              // src={`${apiUrl}/cohort-stats/${cohortId}`}
              src="/icare4cvd_logo.png"
              alt="Cohort statistics"
              className="w-full"
              crossOrigin="use-credentials"
            />
          </div>
          <form method="dialog" className="modal-backdrop">
            <button>close</button>
          </form>
        </dialog>
        {filteredVars.length == Object.keys(cohortsData[cohortId]['variables']).length ? (
          <span className="badge badge-lg mb-2 font-semibold" style={{ backgroundColor: '#fef08a', color: '#854d0e' }}>
            {Object.keys(cohortsData[cohortId]['variables']).length} variables
          </span>
        ) : (
          <div className="mb-2">
            <span className="badge badge-lg font-semibold" style={{ backgroundColor: '#fef08a', color: '#854d0e' }}>
              {filteredVars.length}/{Object.keys(cohortsData[cohortId]['variables']).length} variables
            </span>
            {searchFilters.searchQuery && searchTerms.length > 0 && (
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                {filteredVars.length} variable{filteredVars.length !== 1 ? 's' : ''} match{filteredVars.length === 1 ? 'es' : ''} your search
              </div>
            )}
          </div>
        )}
        
        {/* Outcome Variables Filter Button */}
        <div className="my-4">
          <button
            onClick={() => {
              const newValue = !showOnlyOutcomes;
              if (onShowOnlyOutcomesChange) {
                onShowOnlyOutcomesChange(newValue);
              }
            }}
            className="btn btn-sm w-full border"
            style={{ backgroundColor: '#dbeafe', color: '#1e3a8a', borderColor: '#bfdbfe' }}
          >
            {showOnlyOutcomes ? 'Show All Variables' : 'Show Outcome Variables'}
          </button>
        </div>
        
        <FilterByMetadata
          label="OMOP domains"
          metadata_id="omop_domain"
          options={[...omopDomains]}
          searchResults={filteredVars}
          selectedValues={selectedOMOPDomains}
          onFiltersChange={(optionsSelected: any) => onOMOPDomainsChange(optionsSelected)}
        />
        <FilterByMetadata
          label="Data types"
          metadata_id="var_type"
          options={[...dataTypes]}
          searchResults={filteredVars}
          selectedValues={selectedDataTypes}
          onFiltersChange={(optionsSelected: any) => onDataTypesChange(optionsSelected)}
        />
        <FilterByMetadata
          label="Categorical"
          metadata_id="categorical"
          options={['Non-categorical', 'All categorical', '2 categories', '3 categories', '4+ categories']}
          searchResults={filteredVars}
          selectedValues={selectedCategoryTypes}
          onFiltersChange={(optionsSelected: any) => onCategoryTypesChange(optionsSelected)}
        />
        {visitTypes.size > 1 && (
          <FilterByMetadata
            label="Visit types"
            metadata_id="visits"
            options={[...visitTypes]}
            searchResults={filteredVars}
            selectedValues={selectedVisitTypes}
            onFiltersChange={(optionsSelected: any) => onVisitTypesChange(optionsSelected)}
          />
        )}
      </aside>

      {/* List of variables */}
      <div className="flex flex-col">
        <div className="space-y-2">
          {filteredVars?.map((variable: any) => (
            <div key={variable.var_name} className="card card-compact card-bordered bg-base-100 shadow-xl">
              <div className="card-body">
                <div className="flex justify-between">
                  <div className="flex flex-wrap items-center space-x-3">
                    <h2 className="font-bold text-lg">
                      <HighlightedText text={variable.var_name} searchTerms={searchTerms} searchMode={searchMode} />
                    </h2>
                    {/* Badges for units and categorical variable */}
                    <span className="badge badge-ghost">{variable.var_type}</span>
                    {variable.units && <span className="badge badge-ghost">{variable.units}</span>}
                    {variable.categories.length > 0 && (
                      <span className="badge badge-ghost">🏷️ {variable.categories.length} categories</span>
                    )}
                    {variable.omop_domain && <span className="badge badge-default">{variable.omop_domain}</span>}
                    {variable.formula && <span className="badge badge-outline">🧪 {variable.formula}</span>}
                    {variable.concept_code && <span className="badge" style={{ backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' }}>{variable.concept_code}</span>}
                    {variable.omop_id && <span className="badge" style={{ backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' }}>OMOP ID: {variable.omop_id}</span>}
                    {/* {(variable.concept_id || variable.mapped_id) && ( */}
                    <AutocompleteConcept
                      query={variable.var_label}
                      value={variable.mapped_id || variable.concept_id}
                      domain={variable.omop_domain}
                      index={`${cohortId}_${variable.index}`}
                      tooltip={variable.mapped_label || variable.mapped_id || variable.concept_id}
                      onSelect={(concept: any) => handleConceptSelect(variable.var_name, concept)}
                      canEdit={cohortsData[cohortId].can_edit}
                    />
                    {/* )} */}
                    <button
                      className="btn-sm hover:bg-base-300 rounded-lg"
                      onClick={() => {
                        setOpenedModal(variable.var_name);
                        setTimeout(() => {
                          // @ts-ignore
                          document.getElementById(`source_modal_${cohortId}_${variable.var_name}`)?.showModal();
                        }, 0);
                      }}
                    >
                      <InfoIcon />
                    </button>
                    {/* <div className="grow"></div> */}
                    <button
                      className="btn-sm hover:bg-base-300 rounded-lg"
                      onClick={() => {
                        setOpenedGraphModal(variable.var_name);
                        setTimeout(() => {
                          const modal = document.getElementById(`graph_modal_${cohortId}_${variable.var_name}`) as HTMLDialogElement;
                          if (modal) modal.showModal();
                        }, 0);
                      }}
                    >
                      📊
                    </button>
                  </div>
                  {dataCleanRoom.cohorts[cohortId]?.includes(variable.var_name) ? (
                    <button
                      onClick={() => removeFromDataCleanRoom(variable.var_name)}
                      className="btn btn-sm bg-gray-200 hover:bg-gray-300 text-gray-700 tooltip tooltip-left"
                      data-tip={`Remove the \`${variable.var_name}\` variable of the ${cohortId} cohort from your Data Clean Room`}
                    >
                      Remove from DCR
                    </button>
                  ) : (
                    <button
                      onClick={() => addToDataCleanRoom(variable.var_name)}
                      className="btn btn-neutral btn-sm hover:bg-slate-600 tooltip tooltip-left"
                      data-tip={`Add the \`${variable.var_name}\` variable of the ${cohortId} cohort to your Data Clean Room`}
                    >
                      Add to DCR
                    </button>
                  )}
                </div>
                <p>
                  <HighlightedText text={variable.var_label || ''} searchTerms={searchTerms} searchMode={searchMode} />
                </p>
                
                {/* Display concept_name and mapped_label if they exist */}
                {(variable.concept_name || variable.mapped_label) && (
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {variable.concept_name && (
                      <span className="flex-shrink-0">
                        <span className="font-semibold">Concept:</span>{' '}
                        <HighlightedText text={variable.concept_name} searchTerms={searchTerms} searchMode={searchMode} />
                      </span>
                    )}
                    {variable.mapped_label && (
                      <span className="flex-shrink-0">
                        <span className="font-semibold">Mapped:</span>{' '}
                        <HighlightedText text={variable.mapped_label} searchTerms={searchTerms} searchMode={searchMode} />
                      </span>
                    )}
                  </div>
                )}

                {/* Popup with additional infos about the variable */}
                {openedModal === variable.var_name && (
                  <dialog id={`source_modal_${cohortId}_${variable.var_name}`} className="modal">
                    <div className="modal-box space-y-2 max-w-none w-fit">
                      <div className="flex justify-between items-start items-center">
                        <div>
                          <h5 className="font-bold text-lg">{variable.var_name}</h5>
                        </div>
                        <div className="ml-8">
                          <AutocompleteConcept
                            query={variable.var_label}
                            value={variable.mapped_id || variable.concept_id}
                            domain={variable.omop_domain}
                            index={`${cohortId}_${variable.index}_inside`}
                            tooltip={variable.mapped_label || variable.mapped_id || variable.concept_id}
                            onSelect={(concept: any) => handleConceptSelect(variable.var_name, concept)}
                            canEdit={cohortsData[cohortId].can_edit}
                          />
                        </div>
                      </div>
                      <p className="py-2 lg:mr-32">{variable.var_label}</p>
                      <p>
                        Type: {variable.categorical ? 'Categorical ' : ''}
                        {variable.var_type}
                      </p>
                      {variable.units && (
                        <p>
                          Unit: <span className="badge badge-ghost mx-2">{variable.units}</span>
                        </p>
                      )}
                      {(variable.min || variable.max) && (
                        <p>
                          Min: {variable.min} {variable.units} | Max: {variable.max} {variable.units}
                        </p>
                      )}
                      <p>
                        {variable.count} values | {variable.na} na
                      </p>
                      {variable.categories.length > 0 && (
                        <table className="table w-full">
                          <thead>
                            <tr>
                              <th>Category value</th>
                              <th>Meaning</th>
                              <th>Map to concept</th>
                            </tr>
                          </thead>
                          <tbody>
                            {variable.categories.map((option: any, index: number) => (
                              <tr key={index}>
                                <td>
                                  <HighlightedText text={option.value || ''} searchTerms={searchTerms} searchMode={searchMode} />
                                </td>
                                <td>
                                  <HighlightedText text={option.label || ''} searchTerms={searchTerms} searchMode={searchMode} />
                                </td>
                                <td>
                                  <AutocompleteConcept
                                    query={option.label}
                                    index={`${cohortId}_${variable.index}_category_${index}`}
                                    value={option.mapped_id || option.concept_id}
                                    tooltip={option.mapped_label || option.mapped_id || option.concept_id}
                                    onSelect={concept => handleConceptSelect(variable.var_name, concept, index)}
                                    canEdit={cohortsData[cohortId].can_edit}
                                  />
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                      {variable.formula && (
                        <p>
                          Formula: <code className="p-1 bg-base-300 rounded-md">{variable.formula}</code>
                        </p>
                      )}
                      {variable.definition && <p>Definition: {variable.definition}</p>}
                      {variable.visit_concept_name && <p>Visit concept name: {variable.visit_concept_name}</p>}
                      {variable.visits && <p>Visit: {variable.visits}</p>}
                      {variable.frequency && <p>Frequency: {variable.frequency}</p>}
                      {variable.duration && <p>Duration: {variable.duration}</p>}
                      {variable.omop_domain && (
                        <p>
                          OMOP Domain: <span className="badge badge-ghost">{variable.omop_domain}</span>
                        </p>
                      )}
                    </div>

                    <form method="dialog" className="modal-backdrop">
                      <button>close</button>
                    </form>
                  </dialog>
                )}


               
                {/* Graph modal - now using the separate component */}
                <VariableGraphModal
                  isOpen={openedGraphModal === variable.var_name}
                  cohortId={cohortId}
                  variableName={variable.var_name}
                  variableLabel={variable.var_label}
                  onClose={handleCloseGraphModal}
                />
                {/* <div className='flex-grow'/>
                  <AutocompleteConcept
                    query={variable.var_label}
                    index={variable.index}
                    domain={variable.omop_domain}
                    cohortId={cohortId}
                    onSelect={(concept: any) => handleConceptSelect(variable["var_name"], concept)}
                  /> */}
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
};

export default VariablesList;
