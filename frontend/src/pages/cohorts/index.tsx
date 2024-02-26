'use client';

import React, {useState, useEffect} from 'react';
import Link from 'next/link';
import {useCohorts} from '@/components/CohortsContext';
import FilterByMetadata from '@/components/FilterByMetadata';
import {Cohort} from '@/types';
import VariablesList from '@/components/VariablesList';

export default function CohortsList() {
  const {cohortsData, updateCohortData, dataCleanRoom, setDataCleanRoom} = useCohorts();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTypes, setSelectedTypes] = useState(new Set());
  const [selectedInstitutes, setSelectedInstitutes] = useState(new Set());
  const [selectedOMOPDomains, setSelectedOMOPDomains] = useState(new Set());
  const [selectedDataTypes, setSelectedDataTypes] = useState(new Set());
  const [includeCategorical, setIncludeCategorical] = useState(true);
  const [includeNonCategorical, setIncludeNonCategorical] = useState(true);

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  // Filter cohorts based on search query and selected filters
  // TODO: we might want to perform the search and filtering directly with SPARQL queries to the oxigraph endpoint
  // if the data gets too big to be handled in the client.
  const filteredCohorts = Object.entries(cohortsData as Record<string, Cohort>)
    .filter(([key, value]) => {
      const matchesSearchQuery =
        key.toLowerCase().includes(searchQuery.toLowerCase()) ||
        JSON.stringify(value).toLowerCase().includes(searchQuery.toLowerCase());

      const matchesType = selectedTypes.size === 0 || selectedTypes.has(value.cohort_type);
      const matchesInstitute = selectedInstitutes.size === 0 || selectedInstitutes.has(value.institution);
      return matchesSearchQuery && matchesType && matchesInstitute;
    })
    .map(([, cohortData]) => cohortData);

  // Collect unique OMOP domains and data types from variables for filtering options
  const omopDomains = new Set();
  const dataTypes: any = new Set();
  Object.values(cohortsData[Object.keys(cohortsData)[0]]?.variables || {}).forEach((variable: any) => {
    omopDomains.add(variable.omop_domain);
    dataTypes.add(variable.var_type);
  });
  const cohortTypes = Array.from(new Set(Object.values(cohortsData).map((cohort: any) => cohort.cohort_type)));
  const institutions = Array.from(new Set(Object.values(cohortsData).map((cohort: any) => cohort.institution)));

  return (
    <main className="w-full p-4 bg-base-200 flex h-full min-h-screen">
      <aside className="p-4">
        <FilterByMetadata
          label="Filter by cohorts type"
          metadata_id="cohort_type"
          options={cohortTypes}
          searchResults={filteredCohorts}
          onFiltersChange={(optionsSelected: any) => setSelectedTypes(optionsSelected)}
        />
        <FilterByMetadata
          label="Filter by providers"
          metadata_id="institution"
          options={institutions}
          searchResults={filteredCohorts}
          onFiltersChange={(optionsSelected: any) => setSelectedInstitutes(optionsSelected)}
        />
        {/* TODO: add by study_type, ongoing */}
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
          {filteredCohorts.map(cohortData => (
            <div
              key={cohortData.cohort_id}
              className={`collapse card card-compact bg-base-100 shadow-xl ${!(Object.keys(cohortData.variables).length > 0) ? 'opacity-50' : ''}`}
            >
              <input type="checkbox" />
              <div className="collapse-title">
                <div className="gap-2">
                  {cohortData.cohort_id}
                  <span className="badge badge-outline mx-2">{cohortData.institution}</span>
                  {cohortData.study_type && <span className="badge badge-ghost mx-1">{cohortData.study_type}</span>}
                  {cohortData.cohort_type && <span className="badge badge-ghost mx-1">{cohortData.cohort_type}</span>}
                  {cohortData.cohort_email && <span className="badge mx-2">✉️ {cohortData.cohort_email}</span>}
                  {cohortData.study_participants && <span className="badge badge-ghost mx-1">👥 {cohortData.study_participants}</span>}
                  {cohortData.study_population && <span className="badge badge-ghost mx-1">{cohortData.study_population}</span>}
                  {cohortData.study_duration && <span className="badge badge-default mx-1">⏱️ {cohortData.study_duration}</span>}
                  {cohortData.study_ongoing && <span className="badge badge-ghost mx-1">Ongoing: {cohortData.study_ongoing}</span>}
                  {/* {cohortData.study_objective && <span className="badge badge-ghost">🎯 {cohortData.study_objective}</span>} */}
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
