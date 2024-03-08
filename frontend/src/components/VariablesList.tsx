'use client';

import React, {useState} from 'react';
import {useCohorts} from '@/components/CohortsContext';
import AutocompleteConcept from '@/components/AutocompleteConcept';
import FilterByMetadata from '@/components/FilterByMetadata';
import {Variable} from '@/types';
import {InfoIcon} from '@/components/Icons';

const VariablesList = ({cohortId, searchFilters = {searchQuery: ''}}: any) => {
  const {cohortsData, updateCohortData, dataCleanRoom, setDataCleanRoom} = useCohorts();
  const [selectedOMOPDomains, setSelectedOMOPDomains] = useState(new Set());
  const [selectedDataTypes, setSelectedDataTypes] = useState(new Set());
  const [includeCategorical, setIncludeCategorical] = useState(true);
  const [includeNonCategorical, setIncludeNonCategorical] = useState(true);

  // TODO: use only URIs in the graph, use curie python package to convert URIs/curies
  const handleConceptSelect = (varId: any, concept: any, categoryId: any = null) => {
    // console.log(`Selected concept for ${varId}:`, concept);
    const updatedCohortData = {...cohortsData[cohortId]};
    const vocab = concept.vocabulary.toLowerCase() === 'snomed' ? 'snomedct' : concept.vocabulary.toLowerCase();
    const curie = `${vocab}:${concept.id}`;
    // TODO: Store mappings in the triplestore
    // fetch to API: uploadTriples(cohortId, variableName, "icare:mapped_id", curie)
    const formData = new FormData();
    formData.append('cohort_id', cohortId);
    formData.append('var_id', varId);
    formData.append('predicate', 'icare:mapped_id');
    formData.append('value', curie);
    formData.append('label', concept.name);
    if (categoryId !== null) {
      formData.append('category_id', categoryId);
      updatedCohortData.variables[varId]['categories'][categoryId]['mapped_id'] = curie;
      updatedCohortData.variables[varId]['categories'][categoryId]['mapped_label'] = concept.name;
    } else {
      updatedCohortData.variables[varId]['mapped_id'] = curie;
      updatedCohortData.variables[varId]['mapped_label'] = concept.name;
    }
    updateCohortData(cohortId, updatedCohortData);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
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

  // Function to handle downloading the cohort CSV
  const downloadCohortCSV = () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    const downloadUrl = `${apiUrl}/cohort-spreadsheet/${encodeURIComponent(cohortId)}`;
    // Create a temporary anchor element and trigger a download
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `${cohortId}-datadictionary.csv`; // Downloaded file name
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
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
    <main className="flex">
      <aside className="pr-4 text-center flex flex-col items-center min-w-fit">
        {filteredVars.length > 0 && !dataCleanRoom.cohorts.includes(cohortId) ? (
          <button
            onClick={addToDataCleanRoom}
            className="btn btn-neutral btn-sm mb-2 hover:bg-slate-600 tooltip tooltip-right"
            data-tip="Add cohort to Data Clean Room"
          >
            Add to DCR
          </button>
        ) : (
          <div />
        )}
        {filteredVars.length > 0 && (
          <button
            onClick={downloadCohortCSV}
            className="btn btn-neutral btn-sm mb-2 hover:bg-slate-600 tooltip tooltip-right"
            data-tip="Download the cohort metadata as CSV file. You can edit and re-upload it, if you are the person who published it originally."
          >
            Download CSV
          </button>
        )}
        {filteredVars.length == Object.keys(cohortsData[cohortId]['variables']).length ? (
          <span className="badge badge-ghost mb-2">
            {Object.keys(cohortsData[cohortId]['variables']).length} variables
          </span>
        ) : (
          <span className="badge badge-ghost mb-2">
            {filteredVars.length}/{Object.keys(cohortsData[cohortId]['variables']).length} variables
          </span>
        )}
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
        {cohortsData[cohortId].study_objective && (
          <div className="card p-3 mb-3 bg-base-300">🎯 Study objective: {cohortsData[cohortId].study_objective}</div>
        )}
        <div className="variable-list space-y-2">
          {filteredVars?.map((variable: any) => (
            <div key={variable.var_name} className="card card-compact card-bordered bg-base-100 shadow-xl">
              <div className="card-body flex flex-row">
                <div className="mr-4">
                  <div className="flex items-center space-x-3">
                    <h2 className="font-bold text-lg">{variable.var_name}</h2>
                    {/* Badges for units and categorical variable */}
                    <span className="badge badge-ghost">{variable.var_type}</span>
                    {variable.units && <span className="badge badge-ghost">{variable.units}</span>}
                    {variable.categories.length > 0 && (
                      <span className="badge badge-ghost">🏷️ {variable.categories.length} categories</span>
                    )}
                    {variable.omop_domain && <span className="badge badge-default">{variable.omop_domain}</span>}
                    {variable.formula && <span className="badge badge-outline">🧪 {variable.formula}</span>}
                    {(variable.concept_id || variable.mapped_id) && (
                      <AutocompleteConcept
                        query={variable.var_label}
                        value={variable.mapped_id || variable.concept_id}
                        domain={variable.omop_domain}
                        index={variable.index}
                        cohortId={cohortId}
                        tooltip={variable.mapped_label || variable.mapped_id || variable.concept_id}
                        onSelect={(concept: any) => handleConceptSelect(variable.var_name, concept)}
                      />
                    )}
                    <button
                      className="btn-sm hover:bg-base-300 rounded-lg"
                      onClick={() =>
                        // @ts-ignore
                        document.getElementById(`source_modal_${cohortId}_${variable.var_name}`)?.showModal()
                      }
                    >
                      <InfoIcon />
                    </button>
                  </div>
                  <p>{variable.var_label}</p>
                </div>

                {/* Popup with additional infos about the variable */}
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
                          index={`${variable.index}inside`}
                          cohortId={cohortId}
                          tooltip={variable.mapped_label || variable.mapped_id || variable.concept_id}
                          onSelect={(concept: any) => handleConceptSelect(variable.var_name, concept)}
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
                              <td>{option.value}</td>
                              <td>{option.label}</td>
                              <td>
                                <AutocompleteConcept
                                  query={option.label}
                                  index={`${variable.index}category${index}`}
                                  value={option.mapped_id}
                                  tooltip={option.mapped_label || option.mapped_id}
                                  // domain={variable.omop_domain}
                                  // TODO: properly handle the category concept mapping
                                  cohortId={cohortId}
                                  onSelect={(concept: any) => handleConceptSelect(variable.var_name, concept, index)}
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
