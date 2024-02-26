'use client';

import React, {useState} from 'react';
import {useCohorts} from '@/components/CohortsContext';
import AutocompleteConcept from '@/components/AutocompleteConcept';
import FilterByMetadata from '@/components/FilterByMetadata';
import {Variable} from '@/types';
import {InfoIcon} from '@/components/icons';

const VariablesList = ({cohortId, searchFilters = {searchQuery: ''}}: any) => {
  const {cohortsData, updateCohortData, dataCleanRoom, setDataCleanRoom} = useCohorts();
  const [selectedOMOPDomains, setSelectedOMOPDomains] = useState(new Set());
  const [selectedDataTypes, setSelectedDataTypes] = useState(new Set());
  const [includeCategorical, setIncludeCategorical] = useState(true);
  const [includeNonCategorical, setIncludeNonCategorical] = useState(true);

  const handleConceptSelect = (variableName: any, concept: any) => {
    console.log(`Selected concept for ${variableName}:`, concept);
    // const mapDict = cohortDict[selectedFile];
    // mapDict["@mapping"] = {}
    // mapDict["@mapping"][variableName] = concept
    const updatedCohortData = {...cohortsData[cohortId]};
    updatedCohortData.variables[variableName]['concept_id'] = `${concept.vocabulary}:${concept.id}`;
    updateCohortData(cohortId, updatedCohortData);
    console.log('updatedCohortData:', updatedCohortData);
    // TODO: Store mappings in the triplestore
  };

  // Button to add cohort to data clean room
  const addToDataCleanRoom = () => {
    if (!dataCleanRoom.cohorts.includes(cohortId)) {
      const updatedCleanRoom = {...dataCleanRoom};
      updatedCleanRoom.cohorts.push(cohortId);
      setDataCleanRoom(updatedCleanRoom);
      // cleanRoomData.cohorts.push(selectedFile);
      sessionStorage.setItem('dataCleanRoom', JSON.stringify(updatedCleanRoom));
      // window.location.reload();
    }
  };

  // Collect unique OMOP domains and data types from variables for filtering options
  const omopDomains = new Set();
  const dataTypes: any = new Set();
  Object.values(cohortsData[cohortId]?.variables || {}).forEach((variable: any) => {
    omopDomains.add(variable.omop_domain);
    dataTypes.add(variable.var_type);
  });

  // Filter variables based on search query and selected filters
  let filteredVars: Variable[] = [];
  if (cohortsData && cohortsData[cohortId]) {
    filteredVars = Object.entries(cohortsData[cohortId]['variables'])
      .filter(
        ([variableName, variableData]: any) =>
          variableName.toLowerCase().includes(searchFilters.searchQuery.toLowerCase()) ||
          variableData.var_label.toLowerCase().includes(searchFilters.searchQuery.toLowerCase())
      )
      .filter(
        ([variableName, variableData]: any) =>
          (selectedOMOPDomains.size === 0 || selectedOMOPDomains.has(variableData.omop_domain)) &&
          (selectedDataTypes.size === 0 || selectedDataTypes.has(variableData.var_type)) &&
          ((includeCategorical && variableData.categories.length === 0) ||
            (includeNonCategorical && variableData.categories.length !== 0) ||
            (!includeNonCategorical && !includeCategorical))
      )
      .map(([variableName, variableData]: any) => ({...variableData, var_name: variableName}));
  }

  // Function to count filtered vars based on filter type
  const countMatches = (filterType: string, item: string | null) => {
    return filteredVars.filter(variable => {
      if (filterType === 'categorical') {
        return variable.categories.length > 0;
      } else if (filterType === 'non_categorical') {
        return variable.categories.length === 0;
      } else {
        return variable[filterType] === item;
      }
    }).length;
  };

  return (
    <main className="flex">
      <aside className="pr-4 text-center">
        {filteredVars.length > 0 && !dataCleanRoom.cohorts.includes(cohortId) ? (
          <button
            onClick={addToDataCleanRoom}
            className="btn btn-neutral btn-sm mb-2 hover:bg-slate-600 tooltip tooltip-right"
            data-tip="Add to Data Clean Room"
          >
            Add to DCR
          </button>
        ) : (
          <div />
        )}
        <span className="badge badge-ghost mb-2">
          {Object.entries(cohortsData[cohortId]['variables']).length} variables
        </span>
        <FilterByMetadata
          label="OMOP domains"
          metadata_id="omop_domain"
          options={[...omopDomains]}
          searchResults={filteredVars}
          onFiltersChange={(optionsSelected: any) => setSelectedOMOPDomains(optionsSelected)}
        />
        <FilterByMetadata
          label="Data types"
          metadata_id="var_type"
          options={[...dataTypes]}
          searchResults={filteredVars}
          onFiltersChange={(optionsSelected: any) => setSelectedDataTypes(optionsSelected)}
        />
        <FilterByMetadata
          label="Categorical"
          metadata_id="categorical"
          options={['Categorical', 'Non-categorical']}
          searchResults={filteredVars}
          onFiltersChange={(optionsSelected: any) => {
            // TODO: this bit could be improved
            if (optionsSelected.has('Categorical')) {
              setIncludeCategorical(false);
            } else {
              setIncludeCategorical(true);
            }
            if (optionsSelected.has('Non-categorical')) {
              setIncludeNonCategorical(false);
            } else {
              setIncludeNonCategorical(true);
            }
          }}
        />
      </aside>

      {/* List of variables */}
      <div className="w-full">
        <div className="variable-list space-y-2">
          {filteredVars?.map((variable: any) => (
            <div key={variable.var_name} className="card card-compact card-bordered bg-base-100 shadow-xl">
              <div className="card-body flex flex-row">
                <div className="mr-4">
                  <div className="flex items-center space-x-3">
                    <h2 className="font-bold text-lg">{variable.var_name}</h2>
                    {/* Badges for units and categorical variable */}
                    {variable.units && <span className="badge badge-ghost">{variable.units}</span>}
                    {variable.categories.length > 0 && (
                      <span className="badge badge-ghost">🏷️ {variable.categories.length} categories</span>
                    )}
                    {variable.omop_domain && <span className="badge badge-default">{variable.omop_domain}</span>}
                    {variable.formula && <span className="badge badge-outline">🧪 {variable.formula}</span>}
                    {variable.concept_id && <span className="badge badge-outline">🪪 {variable.concept_id}</span>}
                    <button
                      className="btn-sm hover:bg-base-300 rounded-lg"
                      // @ts-ignore
                      onClick={() => document.getElementById(`source_modal_${variable.var_name}`)?.showModal()}
                    >
                      <InfoIcon />
                    </button>
                  </div>
                  <p>{variable.var_label}</p>
                </div>

                {/* Popup with additional infos about the variable */}
                <dialog id={`source_modal_${variable.var_name}`} className="modal">
                  <div className="modal-box space-y-2 max-w-none w-fit min-w-80">
                    {/* <h5 className="font-bold text-lg">{variable["var_name"]}</h5> */}
                    <div className="flex justify-between items-start">
                      <div>
                        <h5 className="font-bold text-lg">{variable.var_name}</h5>
                      </div>
                      <div className="map-autocomplete-top-right">
                        <AutocompleteConcept
                          query={variable.var_label}
                          value={variable.concept_id}
                          domain={variable.omop_domain}
                          index={variable.index}
                          onSelect={(concept: any) => handleConceptSelect(variable.var_name, concept)}
                        />
                      </div>
                    </div>
                    <p className="py-2">{variable.var_label}</p>
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
                              <td>{option.value}</td>
                              <td>{option.label}</td>
                              <td>
                                <AutocompleteConcept
                                  query={option.label}
                                  index={variable.index}
                                  // value={variable.concept_id}
                                  // domain={variable.omop_domain}
                                  onSelect={(concept: any) => handleConceptSelect(variable.var_name, concept)}
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

                {/* <div className='flex-grow'/>
                  <AutocompleteConcept
                    query={variable.var_label}
                    index={variable.index}
                    domain={variable.omop_domain}
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
