'use client';

import React, {createContext, useState, useEffect, useContext, useRef, MutableRefObject} from 'react';
import {Cohort} from '@/types';
import {apiUrl} from '@/utils';

const CohortsContext = createContext(null);

export const useCohorts = (): any => useContext(CohortsContext) || {};

export const CohortsProvider = ({children}: any) => {
  const [cohortsData, setCohortsData]: [{[cohortId: string]: Cohort}, any] = useState({});
  const [dataCleanRoom, setDataCleanRoom] = useState({cohorts: {}});
  // Dict with cohort ID and list of variables ID?
  const [userEmail, setUserEmail]: [string | null, any] = useState('');
  const worker: MutableRefObject<Worker | null> = useRef(null);

  // Update cohorts data with a web worker in the background for smoothness
  // const worker = new Worker('/cohortsWorker.js');
  // worker.onmessage = (event) => {
  //   // Update your state with the new data
  //   const data = event.data;
  //   if (!data.error) {
  //     setCohortsData(data);
  //     console.log('Updated context with data', data);
  //   } else {
  //     console.error('Error fetching data in worker:', data.error);
  //   }
  // };

  useEffect(() => {
    setDataCleanRoom(JSON.parse(sessionStorage.getItem('dataCleanRoom') || '{"cohorts": {}}'));

    // Update cohorts data with a web worker in the background for smoothness
    worker.current = new Worker('/cohortsWorker.js');
    worker.current.onmessage = event => {
      const data = event.data;
      if (!data.detail) {
        setCohortsData(data);
        // TODO: store actual user email?
        setUserEmail('loggedIn');
        // console.log('Updated context with data', data);
      } else {
        setUserEmail(null);
        console.error('Error fetching data in worker:', data.detail);
      }
    };

    // Fetch cohort data every minute with the worker
    const intervalId = setInterval(() => {
      fetchCohortsData();
    }, 60000);
    // Initial fetch
    fetchCohortsData();
    return () => {
      clearInterval(intervalId);
      worker.current?.terminate();
    };
  }, []);

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
        setUserEmail
      }}
    >
      {children}
    </CohortsContext.Provider>
  );
};
