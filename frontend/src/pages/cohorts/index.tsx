'use client';

import React, {useState} from 'react';
import Link from 'next/link';
import {useCohorts} from '../../components/CohortsContext';

export default function Mapping() {
  const {cohortsData} = useCohorts();
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearchChange = (event: any) => {
    setSearchQuery(event.target.value);
  };

  const filteredCohortDict = searchQuery
    ? cohortsData &&
      Object.entries(cohortsData)
        .filter(
          ([key, value]) =>
            key.toLowerCase().includes(searchQuery.toLowerCase()) ||
            JSON.stringify(value).toLowerCase().includes(searchQuery.toLowerCase())
        )
        .reduce((obj: any, [key, value]) => {
          obj[key] = value;
          return obj;
        }, {})
    : cohortsData;

  return (
    <main className="w-full p-4 bg-base-200">
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
        {filteredCohortDict &&
          Object.keys(filteredCohortDict).map((cohortId: any) => (
            <div
              key={cohortId}
              className={`card card-compact bg-base-100 shadow-xl ${!(Object.keys(filteredCohortDict[cohortId]['variables']).length > 0) ? 'opacity-50' : ''}`}
            >
              {/* <div className="card-body flex flex-row"> */}
              <Link className="card-body flex flex-row" href={'/cohorts/' + cohortId}>
                <p>
                  {cohortId}
                  <span className="badge badge-outline mx-3">{filteredCohortDict[cohortId].institution}</span>
                  {filteredCohortDict[cohortId].cohort_type && (
                    <span className="badge badge-ghost">{filteredCohortDict[cohortId].cohort_type}</span>
                  )}
                  {filteredCohortDict[cohortId].cohort_email && (
                    <span className="badge mx-3">{filteredCohortDict[cohortId].cohort_email}</span>
                  )}
                </p>
              </Link>
              {/* {(filteredCohortDict[cohortId].cohort_email) && (
                <span className="badge mx-3">{filteredCohortDict[cohortId].cohort_email}</span>
              )} */}
            </div>
          ))}
      </div>
    </main>
  );
}
