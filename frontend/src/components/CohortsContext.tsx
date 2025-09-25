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

// Define loading metrics interface
interface LoadingMetrics {
  loadTime: number | null; // in milliseconds
  dataSource: 'cache' | 'sparql';
  cohortCount: number;
  variableCount: number;
  categoryCount: number;
  sparqlRows?: number; // only for SPARQL mode
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

  // Add state for loading metrics
  const [loadingMetrics, setLoadingMetrics] = useState<LoadingMetrics>({
    loadTime: null,
    dataSource: useSparql ? 'sparql' : 'cache',
    cohortCount: 0,
    variableCount: 0,
    categoryCount: 0
  });

  // Add loading state
  const [isLoading, setIsLoading] = useState(false);

  // Function to calculate data metrics
  const calculateDataMetrics = (data: {[cohortId: string]: Cohort}): {cohortCount: number, variableCount: number, categoryCount: number} => {
    const cohortCount = Object.keys(data).length;
    let variableCount = 0;
    let categoryCount = 0;

    Object.values(data).forEach((cohort: Cohort) => {
      if (cohort.variables) {
        variableCount += Object.keys(cohort.variables).length;
        Object.values(cohort.variables).forEach(variable => {
          if (variable.categories) {
            categoryCount += variable.categories.length;
          }
        });
      }
    });

    return { cohortCount, variableCount, categoryCount };
  };

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

    // Reset loading metrics when switching data sources
    setLoadingMetrics({
      loadTime: null,
      dataSource: useSparql ? 'sparql' : 'cache',
      cohortCount: 0,
      variableCount: 0,
      categoryCount: 0
    });

    // Update cohorts data with a web worker in the background for smoothness
    // Use different worker based on useSparql flag
    const workerFile = useSparql ? '/cohortsSparqlWorker.js' : '/cohortsWorker.js';
    worker.current = new Worker(workerFile);
    
    // Track start time
    const startTime = performance.now();
    setIsLoading(true);
    
    worker.current.onmessage = event => {
      const endTime = performance.now();
      const loadTime = endTime - startTime;
      
      const data = event.data;
      if (!data.detail && !data.error) {
        setCohortsData(data);
        setUserEmail('loggedIn');
        setIsLoading(false);
        
        // Calculate metrics
        const metrics = calculateDataMetrics(data);
        
        // Update loading metrics
        setLoadingMetrics({
          loadTime: Math.round(loadTime),
          dataSource: useSparql ? 'sparql' : 'cache',
          cohortCount: metrics.cohortCount,
          variableCount: metrics.variableCount,
          categoryCount: metrics.categoryCount,
          sparqlRows: data.sparqlRows // This will be undefined for cache mode
        });
        
        console.log(`Updated context with data from ${useSparql ? 'SPARQL' : 'cache'}:`, 
          `${metrics.cohortCount} cohorts, ${metrics.variableCount} variables, ${metrics.categoryCount} categories in ${Math.round(loadTime)}ms`);
      } else {
        setUserEmail(null);
        setIsLoading(false);
        console.error(`Error fetching data in ${useSparql ? 'SPARQL' : 'cache'} worker:`, data.detail || data.error);
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
        cohortStatistics,
        // Expose loading metrics and state
        loadingMetrics,
        isLoading
      }}
    >
      {children}
    </CohortsContext.Provider>
  );
};
