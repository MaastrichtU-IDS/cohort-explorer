'use client';

import React, {createContext, useState, useEffect, useContext, useRef, MutableRefObject} from 'react';
import {Cohort} from '@/types';
import {apiUrl} from '@/utils';

// Define statistics interface
interface CohortStatistics {
  totalCohorts: number;
  cohortsWithMetadata: number;
  cohortsWithAggregateAnalysis: number;
  totalPatients: number;
  patientsInCohortsWithMetadata: number;
  totalVariables: number;
}

const CohortsContext = createContext(null);

export const useCohorts = (): any => useContext(CohortsContext) || {};

export const CohortsProvider = ({children, useSparql = false}: {children: any, useSparql?: boolean}) => {
  const [cohortsData, setCohortsData]: [{[cohortId: string]: Cohort}, any] = useState({});
  const [dataCleanRoom, setDataCleanRoom] = useState({cohorts: {}});
  // Dict with cohort ID and list of variables ID?
  const [userEmail, setUserEmail]: [string | null, any] = useState('');
  const worker: MutableRefObject<Worker | null> = useRef(null);
  
  // Add state for statistics
  const [cohortStatistics, setCohortStatistics] = useState<CohortStatistics>({
    totalCohorts: 0,
    cohortsWithMetadata: 0,
    cohortsWithAggregateAnalysis: 0,
    totalPatients: 0,
    patientsInCohortsWithMetadata: 0,
    totalVariables: 0
  });

  // Helper function to parse participant count
  const parseParticipants = (participants: string | number | undefined | null): number => {
    if (participants === undefined || participants === null) return 0;
    
    // If it's already a number, return it directly
    if (typeof participants === 'number') return participants;
    
    // Otherwise, parse the string
    // Split on spaces and take the first part
    const parts = participants.toString().split(' ');
    // Get the first part which should be the number
    const numericPart = parts[0];
    // Remove any non-numeric characters except for commas
    const cleanNumeric = numericPart.replace(/[^0-9,]/g, '');
    // Remove commas and parse as integer
    const parsedValue = parseInt(cleanNumeric.replace(/,/g, ''), 10);
    return isNaN(parsedValue) ? 0 : parsedValue;
  };

  // Calculate statistics whenever cohort data changes
  useEffect(() => {
    const calculateStatistics = async () => {
      if (Object.keys(cohortsData).length === 0) return;
      
      // Convert cohortsData to a typed array for safer operations
      const cohortsList: Cohort[] = Object.values(cohortsData);
      
      // Calculate basic statistics
      const totalCohorts = cohortsList.length;
      
      // Cohorts with metadata (has variables)
      const cohortsWithMetadata = cohortsList.filter(
        (cohort: Cohort) => Object.keys(cohort.variables || {}).length > 0
      );
      const cohortsWithMetadataCount = cohortsWithMetadata.length;
      
      // Total patients across all cohorts
      const totalPatients = cohortsList.reduce(
        (sum: number, cohort: Cohort) => sum + parseParticipants(cohort.study_participants), 
        0
      );
      
      // Patients in cohorts with metadata
      const patientsInCohortsWithMetadata = cohortsWithMetadata.reduce(
        (sum: number, cohort: Cohort) => sum + parseParticipants(cohort.study_participants),
        0
      );
      
      // Total unique variables across all cohorts
      let totalVariables = 0;
      cohortsList.forEach((cohort: Cohort) => {
        if (cohort.variables) {
          totalVariables += Object.keys(cohort.variables).length;
        }
      });
      
      // Check for cohorts with aggregate analysis
      let aggregateAnalysisCount = 0;
      for (const cohort of cohortsList) {
        try {
          const response = await fetch(`/api/check-analysis-folder/${cohort.cohort_id}`);
          const data = await response.json();
          if (data.exists) {
            aggregateAnalysisCount++;
          }
        } catch (error) {
          console.error(`Error checking analysis for cohort ${cohort.cohort_id}:`, error);
        }
      }
      
      // Create the statistics object
      const statistics = {
        totalCohorts,
        cohortsWithMetadata: cohortsWithMetadataCount,
        cohortsWithAggregateAnalysis: aggregateAnalysisCount,
        totalPatients,
        patientsInCohortsWithMetadata,
        totalVariables
      };
      
      // Update statistics state
      setCohortStatistics(statistics);
      
      // Save statistics to JSON file via API
      fetch('/api/save-statistics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(statistics)
      }).catch(error => {
        console.error('Error saving statistics:', error);
      });
    };
    
    calculateStatistics();
  }, [cohortsData]);

  useEffect(() => {
    setDataCleanRoom(JSON.parse(sessionStorage.getItem('dataCleanRoom') || '{"cohorts": {}}'));

    // Update cohorts data with a web worker in the background for smoothness
    // Use different worker based on useSparql flag
    const workerFile = useSparql ? '/cohortsSparqlWorker.js' : '/cohortsWorker.js';
    worker.current = new Worker(workerFile);
    worker.current.onmessage = event => {
      const data = event.data;
      if (!data.detail) {
        setCohortsData(data);
        // TODO: store actual user email?
        setUserEmail('loggedIn');
        console.log(`Updated context with data from ${useSparql ? 'SPARQL' : 'cache'}`, Object.keys(data).length, 'cohorts');
      } else {
        setUserEmail(null);
        console.error(`Error fetching data in ${useSparql ? 'SPARQL' : 'cache'} worker:`, data.detail);
      }
    };

    // Initial fetch only - auto-refresh disabled
    fetchCohortsData();
    return () => {
      worker.current?.terminate();
    };
  }, [useSparql]);

  // Fetch cohorts data from the API using the web worker
  const fetchCohortsData = () => {
    worker.current?.postMessage({apiUrl});
  };

  // Update the metadata of a specific cohort in the context
  const updateCohortData = (cohortId: string, updatedData: any) => {
    setCohortsData((prevData: any) => {
      return {
        ...prevData,
        [cohortId]: updatedData
      };
    });
  };

  return (
    <CohortsContext.Provider
      // @ts-ignore
      value={{
        cohortsData,
        setCohortsData,
        fetchCohortsData,
        updateCohortData,
        dataCleanRoom,
        setDataCleanRoom,
        userEmail,
        setUserEmail,
        // Expose the statistics
        cohortStatistics
      }}
    >
      {children}
    </CohortsContext.Provider>
  );
};
