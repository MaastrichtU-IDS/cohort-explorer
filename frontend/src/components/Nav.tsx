'use client';

import React, {useState, useEffect} from 'react';
import Link from 'next/link';
import {useCohorts} from '../components/CohortsContext';

// Not used: Next Auth.js: https://authjs.dev/getting-started/providers/oauth-tutorial
// Auth0: https://github.com/nextauthjs/next-auth/blob/main/packages/core/src/providers/auth0.ts
// OAuth: https://github.com/nextauthjs/next-auth/blob/main/packages/core/src/providers/oauth.ts
// https://github.com/nextauthjs/next-auth-example/blob/cc1c91a65c70e1a51bfbbb550dbc85e605f0e402/auth.ts

export function Nav() {
  const {dataCleanRoom, setDataCleanRoom, cohortsData, setCohortsData, userEmail, setUserEmail} = useCohorts();
  const [theme, setTheme] = useState('light');
  const [showModal, setShowModal] = useState(false);
  const [publishedDCR, setPublishedDCR]: any = useState(null);
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
    setDataCleanRoom(JSON.parse(sessionStorage.getItem('dataCleanRoom') || '{"cohorts": []}'));
  }, [theme]);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    // console.log("NEW THEME", newTheme)
    sessionStorage.setItem('theme', newTheme);
    setTheme(newTheme);
    // document.querySelector('html')?.setAttribute('data-theme', newTheme);
    // document.documentElement.setAttribute("data-theme", newTheme);
  };

  const handleLogout = () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    fetch(`${apiUrl}/logout`, {
      method: 'POST',
      credentials: 'include'
    })
      .then(response => response.json())
      .then(data => {
        if (!data['detail']) {
          setUserEmail('');
          setCohortsData(null);
          setDataCleanRoom({cohorts: []});
          // Redirect to home page or login page after logout
          window.location.href = '/';
        }
      });
  };

  const sendCohortsToDecentriq = async () => {
    // Replace with actual API endpoint and required request format
    console.log('Sending request to Decentriq', dataCleanRoom);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    const requestBody = { cohorts: {} };
    for (const cohortId of dataCleanRoom.cohorts as string[]) {
      // @ts-ignore
      requestBody.cohorts[cohortId] = cohortsData[cohortId];
    }
    console.log('requestBody', requestBody);
    try {
      const response = await fetch(`${apiUrl}/create-dcr`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      const res = await response.json();
      console.log(res);
      setPublishedDCR(res);
      // Handle response
    } catch (error) {
      console.error('Error sending cohorts:', error);
      // Handle error
    }
  };

  const clearCohortsList = () => {
    sessionStorage.setItem('dataCleanRoom', JSON.stringify({cohorts: []}));
    setDataCleanRoom({cohorts: []});
    setPublishedDCR(null);
  };

  return (
    <div className="navbar bg-base-300 min-h-0 p-0">
      <div className="navbar-start">
        <ul className="menu menu-horizontal my-0 py-0 pl-6 hidden lg:flex">
          <li>
            <Link href="/upload">Upload</Link>
          </li>
          {/* <li><a href="/admin">Admin</a></li> */}
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
        <Link className="text-xl font-thin" href="/cohorts">
          iCare4CVD Cohort Explorer
        </Link>
      </div>

      <div className="navbar-end">
        {/* Desktop */}
        <div className="menu menu-horizontal my-0 py-0 space-x-6 pr-6 items-center">
          <button onClick={() => setShowModal(true)} className="btn">
            Data Clean Room <div className="badge badge-neutral">{dataCleanRoom?.cohorts.length || 0}</div>
          </button>

          {userEmail ? <button onClick={handleLogout}>Logout</button> : <a href="http://localhost:3000/login">Login</a>}

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
            <svg
              className="col-start-1 row-start-1 stroke-base-100 fill-base-100"
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="5" />
              <path d="M12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4" />
            </svg>
            <svg
              className="col-start-2 row-start-1 stroke-base-100 fill-base-100"
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            </svg>
          </label>
          {/* <a href="/docs" target="_blank" data-tooltip="OpenAPI documentation">
            <button className="p-1 rounded-lg hover:bg-gray-500">
                <img className="h-5" src="/openapi_logo.svg" />
            </button>
        </a> */}
        </div>
      </div>
      {showModal && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg mb-3">Cohorts to load in Decentriq Data Clean Room</h3>
            <ul>{dataCleanRoom?.cohorts.map((cohortId: any) => <li key={cohortId}>{cohortId}</li>)}</ul>
            <div className="modal-action">
              <button className="btn btn-neutral" onClick={sendCohortsToDecentriq}>
                Access in Decentriq
              </button>
              <button className="btn btn-error" onClick={clearCohortsList}>
                Clear cohorts
              </button>
              <button className="btn" onClick={() => setShowModal(false)}>
                Close
              </button>
            </div>
            {publishedDCR && (
              <div className="card card-compact">
                <div className="card-body bg-success mt-5 rounded-lg text-slate-900">
                  <p>
                    Data Clean Room <a href={publishedDCR["dcr_url"]}><b>{publishedDCR["dcr_title"]}</b></a> published in Decentriq
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
