'use client';

import React, {useState, useMemo} from 'react';
import {useCohorts} from '@/components/CohortsContext';
import FilterByMetadata from '@/components/FilterByMetadata';
import {Cohort} from '@/types';
import VariablesList from '@/components/VariablesList';

export default function CohortsList() {
  const {cohortsData, userEmail} = useCohorts();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDataTypes, setSelectedDataTypes] = useState(new Set());
  const [selectedStudyTypes, setSelectedStudyTypes] = useState(new Set());
  const [selectedInstitutes, setSelectedInstitutes] = useState(new Set());

  // TODO: debounce search to improve performance
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  // Filter cohorts based on search query and selected filters
  // TODO: we might want to perform the search and filtering directly with SPARQL queries to the oxigraph endpoint
  // if the data gets too big to be handled in the client.
  const filteredCohorts = useMemo(() => {
    return Object.entries(cohortsData as Record<string, Cohort>)
      .filter(([key, value]) => {
        const matchesSearchQuery =
          key.toLowerCase().includes(searchQuery.toLowerCase()) ||
          JSON.stringify(value).toLowerCase().includes(searchQuery.toLowerCase());

        const matchesDataType = selectedDataTypes.size === 0 || selectedDataTypes.has(value.cohort_type);
        const matchesStudyType = selectedStudyTypes.size === 0 || selectedStudyTypes.has(value.study_type);
        const matchesInstitute = selectedInstitutes.size === 0 || selectedInstitutes.has(value.institution);
        return matchesSearchQuery && matchesDataType && matchesStudyType && matchesInstitute;
      })
      .map(([, cohortData]) => cohortData);
  }, [searchQuery, selectedDataTypes, selectedStudyTypes, selectedInstitutes, cohortsData]);
  // NOTE: filtering variables is done in VariablesList component

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
        <FilterByMetadata
          label="Filter by cohorts type"
          metadata_id="cohort_type"
          // Collect unique cohort types from variables for filtering options
          options={Array.from(new Set(Object.values(cohortsData).map((cohort: any) => cohort.cohort_type)))}
          searchResults={filteredCohorts}
          onFiltersChange={(optionsSelected: any) => setSelectedDataTypes(optionsSelected)}
        />
        <FilterByMetadata
          label="Filter by study design"
          metadata_id="study_type"
          options={Array.from(new Set(Object.values(cohortsData).map((cohort: any) => cohort.study_type)))}
          searchResults={filteredCohorts}
          onFiltersChange={(optionsSelected: any) => setSelectedStudyTypes(optionsSelected)}
        />
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
            placeholder="Search for cohorts and variables"
            className="input input-bordered w-full"
            value={searchQuery}
            onChange={handleSearchChange}
          />
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
              <input type="checkbox" />
              <div className="collapse-title">
                <div className="flex flex-wrap items-center gap-2">
                  {cohortData.cohort_id}
                  <span className="badge badge-outline mx-2">{cohortData.institution}</span>
                  {cohortData.study_type && <span className="badge badge-ghost mx-1">{cohortData.study_type}</span>}
                  {cohortData.cohort_type && <span className="badge badge-ghost mx-1">{cohortData.cohort_type}</span>}
                  {(cohortData.study_participants || cohortData.study_population) && (
                    <span className="badge badge-ghost mx-1">
                      👥 {cohortData.study_participants} {cohortData.study_population}
                    </span>
                  )}
                  {(cohortData.male_percentage !== null && cohortData.female_percentage !== null) && (
                    <span className="badge badge-ghost mx-1">male: {cohortData.male_percentage}%, female: {cohortData.female_percentage}%</span>
                  )}
                  {cohortData.study_duration && (
                    <span className="badge badge-default mx-1">⏱️ {cohortData.study_duration}</span>
                  )}
                  {cohortData.study_ongoing && cohortData.study_ongoing === 'yes' && (
                    <span className="badge badge-default mx-1">Ongoing study</span>
                  )}
                  {cohortData.study_ongoing && cohortData.study_ongoing === 'no' && (
                    <span className="badge badge-default mx-1">Completed study</span>
                  )}
                  {cohortData.study_start && cohortData.study_end && (
                    <span className="badge badge-default mx-1">📅 {cohortData.study_start} - {cohortData.study_end}</span>
                  )}
                  {cohortData.cohort_email.map(email => (
                    <span className="badge mx-2" key={cohortData.cohort_id + email}>
                      ✉️ {email}
                    </span>
                  ))}
                </div>
              </div>
              <div className="collapse-content">
                {/* Display study objective section */}
                {cohortData.study_objective && (
                  <div className="mb-4 p-3 bg-base-200 rounded-lg">
                    <h3 className="font-bold mb-2">Study Objective:</h3>
                    <p>{cohortData.study_objective}</p>
                  </div>
                )}
                
                {/* Display morbidity section */}
                {cohortData.morbidity && (
                  <div className="mb-4 p-3 bg-base-200 rounded-lg">
                    <h3 className="font-bold mb-2">Morbidity:</h3>
                    <p>{cohortData.morbidity}</p>
                  </div>
                )}
                {/* Display outcome specifications section */}
                <div className="mb-4 p-3 bg-base-200 rounded-lg">
                  <h3 className="font-bold mb-2">Outcome Specifications:</h3>
                  {cohortData.primary_outcome_spec ? (
                    <div className="mb-2">
                      <h4 className="font-semibold">Primary:</h4>
                      <p>{cohortData.primary_outcome_spec}</p>
                    </div>
                  ) : (
                    <div className="mb-2 text-gray-500">
                      <h4 className="font-semibold">Primary:</h4>
                      <p><em>No primary outcome specification available</em></p>
                    </div>
                  )}
                  {cohortData.secondary_outcome_spec ? (
                    <div className="mb-2">
                      <h4 className="font-semibold">Secondary:</h4>
                      <p>{cohortData.secondary_outcome_spec}</p>
                    </div>
                  ) : (
                    <div className="mb-2 text-gray-500">
                      <h4 className="font-semibold">Secondary:</h4>
                      <p><em>No secondary outcome specification available</em></p>
                    </div>
                  )}
                </div>
                
                {/* Display inclusion and exclusion criteria section */}
                <div className="mb-4 p-3 bg-base-200 rounded-lg">
                  <h3 className="font-bold mb-2">Inclusion & Exclusion Criteria:</h3>
                  
                  {/* Inclusion criteria */}
                  <div className="mb-4">
                    <h4 className="font-semibold mb-2">Inclusion Criteria:</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
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
                    </div>
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
                  
                  {/* Exclusion criteria */}
                  <div>
                    <h4 className="font-semibold mb-2">Exclusion Criteria:</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
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
                    </div>
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
                <VariablesList cohortId={cohortData.cohort_id} searchFilters={{searchQuery: searchQuery}} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
    </>
  );
}
