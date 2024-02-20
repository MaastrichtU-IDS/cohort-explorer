'use client';

import React, {useState} from 'react';
import {useRouter} from 'next/router';
import {useCohorts} from '../../components/CohortsContext';
import AutocompleteConcept from '../../components/AutocompleteConcept';

export default function Mapping() {
  const router = useRouter();
  const selectedFile = router.query.cohortId?.toString() || '';

  const {cohortsData, updateCohortData, dataCleanRoom, setDataCleanRoom} = useCohorts();
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearchChange = (event: any) => {
    setSearchQuery(event.target.value);
  };

  let filteredVars: any = [];
  if (cohortsData && cohortsData[selectedFile]) {
    filteredVars = Object.entries(cohortsData[selectedFile]['variables'])
      .filter(
        ([variableName, variableData]: [string, any]) =>
          variableName.toLowerCase().includes(searchQuery.toLowerCase()) ||
          variableData['VARIABLE LABEL'].toLowerCase().includes(searchQuery.toLowerCase())
      )
      // @ts-ignore
      .map(([variableName, variableData]) => ({...variableData, 'VARIABLE NAME': variableName}));
  }

  const handleConceptSelect = (variableName: any, concept: any) => {
    console.log(`Selected concept for ${variableName}:`, concept);
    // const mapDict = cohortDict[selectedFile];
    // mapDict["@mapping"] = {}
    // mapDict["@mapping"][variableName] = concept
    const updatedCohortData = {...cohortsData[selectedFile]};
    updatedCohortData.variables[variableName]['concept_id'] = `${concept.vocabulary}:${concept.id}`;
    updateCohortData(selectedFile, updatedCohortData);
    console.log('updatedCohortData:', updatedCohortData);
    // TODO: Store mappings in the triplestore
  };

  const addToDataCleanRoom = () => {
    if (!dataCleanRoom.cohorts.includes(selectedFile)) {
      const updatedCleanRoom = {...dataCleanRoom};
      updatedCleanRoom.cohorts.push(selectedFile);
      setDataCleanRoom(updatedCleanRoom);
      // cleanRoomData.cohorts.push(selectedFile);
      sessionStorage.setItem('dataCleanRoom', JSON.stringify(updatedCleanRoom));
      // window.location.reload();
    }
  };

  return (
    <main className="w-full p-4 bg-base-200">
      <div className="mb-4 text-center flex justify-between items-center">
        <div />
        <h2 className="font-bold">{selectedFile}</h2>
        {filteredVars.length > 0 && !dataCleanRoom.cohorts.includes(selectedFile) ? (
          <button onClick={addToDataCleanRoom} className="btn btn-neutral btn-sm hover:bg-slate-600">
            Add to Data Clean Room
          </button>
        ) : (
          <div />
        )}
      </div>
      <div className="mb-4">
        <input
          type="text"
          placeholder="Search for variables..."
          className="input input-bordered w-full"
          value={searchQuery}
          onChange={handleSearchChange}
        />
      </div>

      <div className="variable-list space-y-2">
        {filteredVars?.map((variable: any) => (
          <div key={variable['VARIABLE NAME']} className="card card-compact bg-base-100 shadow-xl">
            <div className="card-body flex flex-row">
              <div className="mr-4">
                <div className="flex items-center space-x-3">
                  <h2 className="font-bold text-lg">{variable['VARIABLE NAME']}</h2>
                  {/* Badges for units and categorical variable */}
                  {variable['UNITS'] && <span className="badge badge-ghost">{variable['UNITS']}</span>}
                  {variable['Formula'] && <span className="badge badge-outline">🧪 {variable['Formula']}</span>}
                  {variable['categories'].length > 0 && (
                    <span className="badge badge-ghost">🏷️ {variable['categories'].length} categories</span>
                  )}
                  {variable['concept_id'] && <span className="badge badge-outline">{variable['concept_id']}</span>}
                  {variable['OMOP'] && <span className="badge badge-ghost">{variable['OMOP']}</span>}
                  <button
                    className="btn-sm hover:bg-base-300 rounded-lg"
                    // @ts-ignore
                    onClick={() => document.getElementById(`source_modal_${variable['VARIABLE NAME']}`)?.showModal()}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                  </button>
                </div>
                <p>{variable['VARIABLE LABEL']}</p>
              </div>

              {/* Popup with additional infos about the variable */}
              <dialog id={`source_modal_${variable['VARIABLE NAME']}`} className="modal">
                <div className="modal-box space-y-2 max-w-none w-fit min-w-80">
                  {/* <h5 className="font-bold text-lg">{variable["VARIABLE NAME"]}</h5> */}
                  <div className="flex justify-between items-start">
                    <div>
                      <h5 className="font-bold text-lg">{variable['VARIABLE NAME']}</h5>
                    </div>
                    <div className="map-autocomplete-top-right">
                      <AutocompleteConcept
                        query={variable['VARIABLE LABEL']}
                        value={variable['concept_id']}
                        domain={variable['OMOP']}
                        onSelect={(concept: any) => handleConceptSelect(variable['VARIABLE NAME'], concept)}
                      />
                    </div>
                  </div>
                  <p className="py-2">{variable['VARIABLE LABEL']}</p>
                  <p>
                    Type: {variable['CATEGORICAL'] ? 'Categorical ' : ''}
                    {variable['VAR TYPE']}
                  </p>
                  {variable['UNITS'] && (
                    <p>
                      Unit: <span className="badge badge-ghost mx-2">{variable['UNITS']}</span>
                    </p>
                  )}
                  {(variable['MIN'] || variable['MAX']) && (
                    <p>
                      Min: {variable['MIN']} {variable['UNITS']} | Max: {variable['MAX']} {variable['UNITS']}
                    </p>
                  )}
                  <p>
                    {variable['COUNT']} values | {variable['NA']} na
                  </p>
                  {variable['categories'].length > 0 && (
                    <table className="table w-full">
                      <thead>
                        <tr>
                          <th>Category value</th>
                          <th>Meaning</th>
                          <th>Map to concept</th>
                        </tr>
                      </thead>
                      <tbody>
                        {variable['categories'].map((option: any, index: number) => (
                          <tr key={index}>
                            <td>{option.value}</td>
                            <td>{option.label}</td>
                            <td>
                              <AutocompleteConcept
                                query={option.label}
                                // value={variable["concept_id"]}
                                // domain={variable["OMOP"]}
                                onSelect={(concept: any) => handleConceptSelect(variable['VARIABLE NAME'], concept)}
                              />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  {variable['Formula'] && (
                    <p>
                      Formula: <code className="p-1 bg-base-300 rounded-md">{variable['Formula']}</code>
                    </p>
                  )}
                  {variable['Definition'] && <p>Definition: {variable['Definition']}</p>}
                  {variable['Visits'] && <p>Visit: {variable['Visits']}</p>}
                  {variable['Frequency'] && <p>Frequency: {variable['Frequency']}</p>}
                  {variable['Duration'] && <p>Duration: {variable['Duration']}</p>}
                  {variable['OMOP'] && (
                    <p>
                      OMOP Domain: <span className="badge badge-ghost">{variable['OMOP']}</span>
                    </p>
                  )}
                </div>

                <form method="dialog" className="modal-backdrop">
                  <button>close</button>
                </form>
              </dialog>

              {/* <div className='flex-grow'/>
                <AutocompleteConcept
                  query={variable["VARIABLE LABEL"]}
                  domain={variable["OMOP"]}
                  onSelect={(concept: any) => handleConceptSelect(variable["VARIABLE NAME"], concept)}
                /> */}
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
