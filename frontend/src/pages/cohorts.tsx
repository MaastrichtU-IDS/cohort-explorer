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
          {filteredCohorts.map(cohortData => (
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
                  {cohortData.study_duration && (
                    <span className="badge badge-default mx-1">⏱️ {cohortData.study_duration}</span>
                  )}
                  {cohortData.study_ongoing && cohortData.study_ongoing === 'yes' && (
                    <span className="badge badge-default mx-1">Ongoing study</span>
                  )}
                  {cohortData.study_ongoing && cohortData.study_ongoing === 'no' && (
                    <span className="badge badge-default mx-1">Completed study</span>
                  )}
                  {cohortData.cohort_email.map(email => (
                    <span className="badge mx-2" key={cohortData.cohort_id + email}>
                      ✉️ {email}
                    </span>
                  ))}
                  {cohortData.airlock && <span className="badge badge-outline mx-1">🔎 Data preview</span>}
                </div>
              </div>
              <div className="collapse-content">
                <VariablesList cohortId={cohortData.cohort_id} searchFilters={{searchQuery: searchQuery}} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
