import React, {useState, useMemo} from 'react';
import {useCohorts} from '@/components/CohortsContext';
import AutocompleteConcept from '@/components/AutocompleteConcept';
import FilterByMetadata from '@/components/FilterByMetadata';
import VariableGraphModal from '@/components/VariableGraphModal';
import {InfoIcon} from '@/components/Icons';
import {Concept, Variable} from '@/types';
import {apiUrl} from '@/utils';

const VariablesList = ({cohortId, searchFilters = {searchQuery: ''}}: any) => {
  const {cohortsData, updateCohortData, dataCleanRoom, setDataCleanRoom} = useCohorts();
  const [selectedOMOPDomains, setSelectedOMOPDomains] = useState(new Set());
  const [selectedDataTypes, setSelectedDataTypes] = useState(new Set());
  const [includeCategorical, setIncludeCategorical] = useState(true);
  const [includeNonCategorical, setIncludeNonCategorical] = useState(true);
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

  // Collect unique OMOP domains and data types from variables for filtering options
  const omopDomains = new Set();
  const dataTypes: any = new Set();
  Object.values(cohortsData[cohortId]?.variables || {}).forEach((variable: any) => {
    omopDomains.add(variable.omop_domain);
    dataTypes.add(variable.var_type);
  });

  // Filter variables based on search query and selected filters
  const filteredVars = useMemo(() => {
    if (cohortsData && cohortsData[cohortId]) {
      return Object.entries(cohortsData[cohortId]['variables'])
        .filter(
          ([variableName, variableData]: any) =>
            variableName.toLowerCase().includes(searchFilters.searchQuery.toLowerCase()) ||
            JSON.stringify(variableData).toLowerCase().includes(searchFilters.searchQuery.toLowerCase())
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
    } else {
      return [];
    }
  }, [
    cohortsData,
    cohortId,
    searchFilters,
    selectedOMOPDomains,
    selectedDataTypes,
    includeCategorical,
    includeNonCategorical
  ]);

  // Function to handle downloading the cohort CSV
  const downloadMetadataCSV = async () => {
    // Fetch the list of CSV files from the backend
    const res = await fetch(`${apiUrl}/list-csvs/${encodeURIComponent(cohortId)}`, { credentials: 'include' });
    const files = await res.json(); 
    // Filter out "noHeader" and non-CSV files, then sort by name (or timestamp if available)
    const filtered = files
      .filter((name: string) => !name.includes('noHeader') && name.endsWith('.csv'))
      .sort((a: string, b: string) => b.localeCompare(a)); // or sort by date if available

  if (filtered.length > 0) {
    const mostRecent = filtered[0];
    const downloadUrl = `${apiUrl}/csvs/${encodeURIComponent(mostRecent)}`;
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = mostRecent;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
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
      <aside className="flex-shrink-0 text-center flex flex-col items-center min-w-fit">
        {Object.keys(cohortsData[cohortId]['variables']).length > 0 &&
        (!dataCleanRoom.cohorts[cohortId] ||
          dataCleanRoom.cohorts[cohortId].length !== Object.keys(cohortsData[cohortId]['variables']).length) ? (
          <button
            onClick={() => addToDataCleanRoom()}
            className="btn btn-neutral btn-sm mb-2 hover:bg-slate-600 tooltip tooltip-right"
            data-tip={`Add all variables of the cohort ${cohortId} to your Data Clean Room`}
          >
            Add to DCR
          </button>
        ) : (
          <div />
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
      <div className="flex flex-col">
        {cohortsData[cohortId].study_objective && (
          <div className="card p-3 mb-3 bg-base-300">🎯 Study objective: {cohortsData[cohortId].study_objective}</div>
        )}
        <div className="space-y-2">
          {filteredVars?.map((variable: any) => (
            <div key={variable.var_name} className="card card-compact card-bordered bg-base-100 shadow-xl">
              <div className="card-body">
                <div className="flex justify-between">
                  <div className="flex flex-wrap items-center space-x-3">
                    <h2 className="font-bold text-lg">{variable.var_name}</h2>
                    {/* Badges for units and categorical variable */}
                    <span className="badge badge-ghost">{variable.var_type}</span>
                    {variable.units && <span className="badge badge-ghost">{variable.units}</span>}
                    {variable.categories.length > 0 && (
                      <span className="badge badge-ghost">🏷️ {variable.categories.length} categories</span>
                    )}
                    {variable.omop_domain && <span className="badge badge-default">{variable.omop_domain}</span>}
                    {variable.formula && <span className="badge badge-outline">🧪 {variable.formula}</span>}
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
                  {!dataCleanRoom.cohorts[cohortId]?.includes(variable.var_name) ? (
                    <button
                      onClick={() => addToDataCleanRoom(variable.var_name)}
                      className="btn btn-neutral btn-sm hover:bg-slate-600 tooltip tooltip-left"
                      data-tip={`Add the \`${variable.var_name}\` variable of the ${cohortId} cohort to your Data Clean Room`}
                    >
                      Add to DCR
                    </button>
                  ) : (
                    <div />
                  )}
                </div>
                <p>{variable.var_label}</p>

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
                                <td>{option.value}</td>
                                <td>{option.label}</td>
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
