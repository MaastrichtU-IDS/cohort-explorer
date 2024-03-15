'use client';

import React, {createContext, useState, useEffect, useContext} from 'react';
import {Cohort} from '@/types';
import {apiUrl} from '@/utils';

const CohortsContext = createContext(null);

export const useCohorts = (): any => useContext(CohortsContext) || {};

export const CohortsProvider = ({children}: any) => {
  const [cohortsData, setCohortsData]: [{[cohortId: string]: Cohort}, any] = useState({});
  const [dataCleanRoom, setDataCleanRoom] = useState({cohorts: []});
  const [userEmail, setUserEmail] = useState('');

  useEffect(() => {
    setDataCleanRoom(JSON.parse(sessionStorage.getItem('dataCleanRoom') || '{"cohorts": []}'));
    if (Object.keys(cohortsData).length === 0) {
      fetch(`${apiUrl}/cohorts-metadata`, {
        credentials: 'include'
      })
        .then(response => response.json())
        .then(data => {
          console.log('Context data dict', data);
          if (!data['detail']) {
            setUserEmail('anything');
            setCohortsData(data);
          }
        });
    }
  }, [cohortsData]);

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
      value={{cohortsData, setCohortsData, updateCohortData, dataCleanRoom, setDataCleanRoom, userEmail, setUserEmail}}
    >
      {children}
    </CohortsContext.Provider>
  );
};
