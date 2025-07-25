'use client';

import React, {useState, useEffect} from 'react';
import Link from 'next/link';
import {LogIn, LogOut, Compass, Upload, HardDrive, Map} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {DarkThemeIcon, LightThemeIcon} from '@/components/Icons';
import {apiUrl} from '@/utils';

// Not used: Next Auth.js: https://authjs.dev/getting-started/providers/oauth-tutorial
// Auth0: https://github.com/nextauthjs/next-auth/blob/main/packages/core/src/providers/auth0.ts
// OAuth: https://github.com/nextauthjs/next-auth/blob/main/packages/core/src/providers/oauth.ts
// https://github.com/nextauthjs/next-auth-example/blob/cc1c91a65c70e1a51bfbbb550dbc85e605f0e402/auth.ts

export function Nav() {
  const {dataCleanRoom, setDataCleanRoom, cohortsData, setCohortsData, userEmail, setUserEmail} = useCohorts();
  const [theme, setTheme] = useState('light');
  const [showModal, setShowModal] = useState(false);
  const [publishedDCR, setPublishedDCR]: any = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  // const [cleanRoomData, setCleanRoomData]: any = useState(null);
  // const cleanRoomData = JSON.parse(sessionStorage.getItem('dataCleanRoom') || '{"cohorts": []}');
  // const cohortsCount = cleanRoomData.cohorts.length;

  useEffect(() => {
    const storedTheme = sessionStorage.getItem('theme') || 'light';
    setTheme(storedTheme);
    document.querySelector('html')?.setAttribute('data-theme', storedTheme);
    const root = document.documentElement;
    if (storedTheme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme]);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    sessionStorage.setItem('theme', newTheme);
    setTheme(newTheme);
  };

  const handleLogout = () => {
    fetch(`${apiUrl}/logout`, {
      method: 'POST',
      credentials: 'include'
    })
      .then(response => response.json())
      .then(data => {
        if (!data['detail']) {
          setUserEmail(null);
          setCohortsData(null);
          setDataCleanRoom({cohorts: {}});
          // Redirect to home page or login page after logout
          window.location.href = '/';
        }
      });
  };

  const getDCRDefinitionFile = async () => {
    setIsLoading(true);
    // Replace with actual API endpoint and required request format
    // console.log('Sending request to Decentriq', dataCleanRoom);
    try {
      const response = await fetch(`${apiUrl}/get-compute-dcr-definition`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(dataCleanRoom)
      });
      const res = await response.json();
      const blob = new Blob([JSON.stringify(res, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'dcr_definition.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setPublishedDCR((
        <p>✅ Data Clean Room configuration file has been downloaded. <br />
        Please go to <a href="https://platform.decentriq.com/" target="_blank">platform.decentriq.com</a> to create a new DCR from the configuration file. </p>
      ))
      setIsLoading(false);
      // Handle response
    } catch (error) {
      console.error('Error getting DCR definition file:', error);
      setIsLoading(false);
      // Handle error
    }
  };

  const clearCohortsList = () => {
    sessionStorage.setItem('dataCleanRoom', JSON.stringify({cohorts: {}}));
    setDataCleanRoom({cohorts: {}});
    setPublishedDCR(null);
  };

  return (
    <div className="navbar bg-base-300 min-h-0 p-0">
      <div className="navbar-start">
        <ul className="menu menu-horizontal gap-2 my-0 py-0 pl-6 hidden lg:flex">
          <li>
            <Link href="/upload">
              <Upload />
              Upload
            </Link>
          </li>
          <li>
            <Link href="/cohorts">
              <Compass />
              Explore
            </Link>
          </li>   
          <li>
            <Link href="/mapping">
              <Map />
              Mapping
            </Link>
          </li>

      {/*<li> 
        <Link href="/docs_store">
          <HardDrive /> 
          Documents
        </Link>
      </li>*/}
        </ul>
        {/* <div className="dropdown lg:hidden">
        <div tabindex="0" role="button" className="btn btn-ghost lg:hidden">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h8m-8 6h16" /></svg>
        </div>
        <ul tabindex="0" className="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow bg-base-100 rounded-box w-52">
          <li><a href="/">Home</a></li>
          <li><a href="/admin">Admin</a></li>
        </ul>
      </div> */}
      </div>

      <div className="navbar-center">
        <Link className="text-xl font-thin" href="/">
          iCARE4CVD Cohort Explorer
        </Link>
      </div>

      <div className="navbar-end">
        {/* Desktop */}
        <div className="menu menu-horizontal my-0 py-0 space-x-6 pr-6 items-center">
          <button onClick={() => setShowModal(true)} className="btn">
            Data Clean Room <div className="badge badge-neutral">{Object.keys(dataCleanRoom?.cohorts).length || 0}</div>
          </button>

          {userEmail ? (
            <button onClick={handleLogout} className="flex space-x-2 p-2 rounded-lg hover:bg-neutral-300">
              <LogOut />
              <span>Logout</span>
            </button>
          ) : (
            <a href={`${apiUrl}/login`} className="flex space-x-2 p-2 rounded-lg hover:bg-neutral-300">
              <LogIn />
              <span>Login</span>
            </a>
          )}

          {/* Add light/dark theme switch */}
          <label className="cursor-pointer grid place-items-center">
            <input
              type="checkbox"
              checked={theme === 'dark'}
              onClick={toggleTheme}
              onChange={toggleTheme}
              value={theme}
              className="toggle theme-controller bg-base-content row-start-1 col-start-1 col-span-2"
            />
            <LightThemeIcon />
            <DarkThemeIcon />
          </label>
          {/* <a href="/docs" target="_blank" data-tooltip="OpenAPI documentation">
            <button className="p-1 rounded-lg hover:bg-gray-500">
                <img className="h-5" src="/openapi_logo.svg" />
            </button>
        </a> */}
        </div>
      </div>
      {/* Popup to publish a Data Clean Room with selected cohorts */}
      {showModal && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg mb-3">Cohorts to load in Decentriq Data Clean Room</h3>
            <ul>
              {Object.entries(dataCleanRoom?.cohorts).map(([cohortId, variables]: any) => (
                <li key={cohortId}>
                  {cohortId} ({variables.length} variables)
                </li>
              ))}
            </ul>
            {/* TODO: add a section to merge added cohorts? (merge automatically on variables using mapped_id)
            - An id for the new generated dataframe
            - A list of autocomplete using the dataCleanRoom.cohorts
            Once the first is selected we only show the cohorts with same number of variables?
            */}
            <div className="modal-action flex flex-wrap justify-end gap-2 mt-4">
                {/* <div className="flex flex-wrap space-y-2"> */}
                <button className="btn btn-neutral" onClick={getDCRDefinitionFile}>
                  Download DCR definition file
                </button>
                <button className="btn btn-error" onClick={clearCohortsList}>
                  Clear cohorts
                </button>
                <button className="btn" onClick={() => setShowModal(false)}>
                  Close
                </button>
                {/* </div> */}
            </div>
            {/* TODO: {isLoading && <div className="loader"></div>} */}
            {isLoading && (
              <div className="flex flex-col items-center opacity-70 text-slate-500 mt-5 mb-5">
                <span className="loading loading-spinner loading-lg mb-4"></span>
                <p>Creating the file specification for a DCR draft...</p>
              </div>
            )}
            {publishedDCR && (
              <div className="card card-compact">
                <div className="card-body bg-success mt-5 rounded-lg text-slate-900">
                    {publishedDCR}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
