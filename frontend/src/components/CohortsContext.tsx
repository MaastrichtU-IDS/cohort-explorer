'use client';

import React, {createContext, useState, useEffect, useContext, useRef, useCallback, MutableRefObject} from 'react';
import {Cohort} from '@/types';
import {apiUrl} from '@/utils';
import {calculateCohortStatistics, type CohortStatistics} from '@/utils/cohortStatistics';
import {isLatestMetadataResponse} from '@/utils/metadataRequest';

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
  const metadataRequestGeneration = useRef(0);
  const statisticsGeneration = useRef(0);
  
  // Add state for statistics
  const [cohortStatistics, setCohortStatistics] = useState<CohortStatistics>({
    totalCohorts: 0,
    cohortsWithMetadata: 0,
    cohortsWithAggregateAnalysis: 0,
    totalPatients: 0,
    patientsInCohortsWithMetadata: 0,
    totalVariables: 0
  });
  const [statisticsStatus, setStatisticsStatus] = useState<'loading' | 'loaded' | 'error'>('loading');

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

  const calculateStatisticsFor = useCallback(async (snapshot: {[cohortId: string]: Cohort}) => {
    const generation = ++statisticsGeneration.current;
    setStatisticsStatus('loading');

    try {
      const statistics = await calculateCohortStatistics(snapshot, async cohortId => {
        const response = await fetch(`/api/check-analysis-folder/${cohortId}`);
        if (!response.ok) throw new Error(`Aggregate-analysis check failed for ${cohortId}: ${response.status}`);
        const data = await response.json();
        return Boolean(data.exists);
      });
      if (generation !== statisticsGeneration.current) return;

      setCohortStatistics(statistics);
      setStatisticsStatus('loaded');
    } catch (error) {
      if (generation === statisticsGeneration.current) setStatisticsStatus('error');
      throw error;
    }
  }, []);

  useEffect(() => {
    metadataRequestGeneration.current += 1;
    statisticsGeneration.current += 1;
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
    setStatisticsStatus('loading');
    
    worker.current.onmessage = event => {
      const endTime = performance.now();
      const loadTime = endTime - startTime;
      
      const {requestId, payload: data, error: workerError} = event.data;
      if (!isLatestMetadataResponse(requestId, metadataRequestGeneration.current)) return;

      if (data && !data.detail && !data.error && !workerError) {
        // Extract cohorts data (filter out metadata properties)
        const { sparqlRows, sparqlMetadata, userEmail: fetchedUserEmail, ...cohortsData } = data;
        
        setCohortsData(cohortsData);
        setUserEmail(fetchedUserEmail || 'loggedIn');
        setIsLoading(false);
        
        // Calculate metrics using only cohorts data
        const metrics = calculateDataMetrics(cohortsData);
        
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
        
        // Calculate against the worker payload, not the callback's pre-fetch render.
        void calculateStatisticsFor(cohortsData).catch(error => {
          console.error('Error calculating cohort statistics:', error);
        });
      } else {
        statisticsGeneration.current += 1;
        setStatisticsStatus('error');
        setUserEmail(null);
        setIsLoading(false);
        console.error(
          `Error fetching data in ${useSparql ? 'SPARQL' : 'cache'} worker:`,
          data?.detail || data?.error || workerError
        );
      }
    };

    // Initial fetch only - auto-refresh disabled
    fetchCohortsData();
    return () => {
      worker.current?.terminate();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useSparql]);

  // Fetch cohorts data from the API using the web worker
  const fetchCohortsData = () => {
    if (!worker.current) return;
    worker.current.postMessage({apiUrl, requestId: ++metadataRequestGeneration.current});
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
        statisticsStatus,
        // Expose loading metrics and state
        loadingMetrics,
        isLoading
      }}
    >
      {children}
    </CohortsContext.Provider>
  );
};
