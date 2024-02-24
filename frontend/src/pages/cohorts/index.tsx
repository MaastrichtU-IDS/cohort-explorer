'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useCohorts } from '@/components/CohortsContext';
import FilterByMetadata from '@/components/FilterByMetadata';
import { Cohort } from '@/types';

export default function CohortsList() {
  const { cohortsData } = useCohorts();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTypes, setSelectedTypes] = useState(new Set());
  const [selectedInstitutes, setSelectedInstitutes] = useState(new Set());

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  const filteredCohorts = Object.entries(cohortsData as Record<string, Cohort>).filter(
    ([key, value]) => {
      const matchesSearchQuery = key.toLowerCase().includes(searchQuery.toLowerCase()) ||
        JSON.stringify(value).toLowerCase().includes(searchQuery.toLowerCase());

      const matchesType = selectedTypes.size === 0 || selectedTypes.has(value.cohort_type);
      const matchesInstitute = selectedInstitutes.size === 0 || selectedInstitutes.has(value.institution);
      return matchesSearchQuery && matchesType && matchesInstitute;
    }
  ).map(([, cohortData]) => cohortData);

  const cohortTypes = Array.from(new Set(Object.values(cohortsData).map((cohort: any) => cohort.cohort_type)));
  const institutions = Array.from(new Set(Object.values(cohortsData).map((cohort: any) => cohort.institution)));

  return (
    <main className="w-full p-4 bg-base-200 flex">
      <aside className="p-4">
        <FilterByMetadata
            label="Filter by Type"
            metadata_id='cohort_type'
            options={cohortTypes}
            searchResults={filteredCohorts}
            onFiltersChange={(newSelected: any) => setSelectedTypes(newSelected)}
        />
        <FilterByMetadata
            label="Filter by Institute"
            metadata_id='institution'
            options={institutions}
            searchResults={filteredCohorts}
            onFiltersChange={(newSelected: any) => setSelectedInstitutes(newSelected)}
        />
      </aside>

      <div className="w-full">
        <div className="mb-4">
          <input
            type="text"
            placeholder="Search for cohorts..."
            className="input input-bordered w-full"
            value={searchQuery}
            onChange={handleSearchChange}
          />
        </div>

        <div className="space-y-2">
          {filteredCohorts.map((cohortData) => (
            <div
              key={cohortData.cohort_id}
              className={`card card-compact bg-base-100 shadow-xl ${!(Object.keys(cohortData.variables).length > 0) ? 'opacity-50' : ''}`}
            >
              <Link className="card-body flex flex-row" href={'/cohorts/' + cohortData.cohort_id}>
                <p>
                  {cohortData.cohort_id}
                  <span className="badge badge-outline mx-3">{cohortData.institution}</span>
                  {cohortData.cohort_type && (
                    <span className="badge badge-ghost">{cohortData.cohort_type}</span>
                  )}
                  {cohortData.cohort_email && (
                    <span className="badge mx-3">{cohortData.cohort_email}</span>
                  )}
                </p>
              </Link>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
