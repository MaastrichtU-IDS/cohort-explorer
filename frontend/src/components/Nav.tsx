'use client';

import React, {useState, useEffect, useMemo, useCallback} from 'react';
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
  const [loadingAction, setLoadingAction] = useState<'live' | 'config' | null>(null);
  const [includeShuffledSamples, setIncludeShuffledSamples] = useState(true);
  const [dcrMode, setDcrMode] = useState<'current' | 'future'>('current');
  const [showParticipantsModal, setShowParticipantsModal] = useState(false);
  const [additionalAnalysts, setAdditionalAnalysts] = useState<string[]>([]);
  const [newAnalystEmail, setNewAnalystEmail] = useState('');
  const [airlockSettings, setAirlockSettings] = useState<Record<string, number>>({});
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
    setLoadingAction('config');
    // Replace with actual API endpoint and required request format
    // console.log('Sending request to Decentriq', dataCleanRoom);
    try {
      const response = await fetch(`${apiUrl}/get-compute-dcr-definition`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...dataCleanRoom,
          include_shuffled_samples: includeShuffledSamples
        })
      });
      
      // Check content type to determine if it's a ZIP file or JSON
      const contentType = response.headers.get('content-type');
      
      if (contentType && contentType.includes('application/zip')) {
        // Handle ZIP file response (with shuffled samples)
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'dcr_config_with_samples.zip';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setPublishedDCR((
          <p>✅ Data Clean Room configuration package (with shuffled samples) has been downloaded. <br />
          Please go to <a href="https://platform.decentriq.com/" target="_blank" className="underline text-blue-600 hover:text-blue-800">https://platform.decentriq.com</a> to create a new DCR from the configuration file. </p>
        ))
      } else {
        // Handle JSON response (no shuffled samples)
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
          Please go to <a href="https://platform.decentriq.com/" target="_blank" className="underline text-blue-600 hover:text-blue-800">https://platform.decentriq.com</a> to create a new DCR from the configuration file. </p>
        ))
      }
      
      setIsLoading(false);
      setLoadingAction(null);
      // Handle response
    } catch (error) {
      console.error('Error getting DCR definition file:', error);
      setIsLoading(false);
      setLoadingAction(null);
      // Handle error
    }
  };

  const createLiveDCR = async () => {
    setIsLoading(true);
    setLoadingAction('live');
    try {
      const response = await fetch(`${apiUrl}/create-live-compute-dcr`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...dataCleanRoom,
          include_shuffled_samples: includeShuffledSamples,
          additional_analysts: additionalAnalysts,
          airlock_settings: airlockSettings
        })
      });
      
      const result = await response.json();
      
      if (response.ok) {
        // Categorize cohorts by shuffled sample availability
        const cohortsWithSamples: string[] = [];
        const cohortsWithoutSamples: string[] = [];
        
        result.cohort_ids.forEach((cohortId: string) => {
          const status = result.shuffled_upload_results[cohortId];
          if (status === 'success') {
            cohortsWithSamples.push(cohortId);
          } else if (status === 'no_file' || status === 'file_not_exists') {
            cohortsWithoutSamples.push(cohortId);
          }
        });
        
        setPublishedDCR((
          <div className="bg-success text-slate-900 p-4 rounded-lg">
            <p className="font-bold mb-4 text-lg">✅ {result.message}</p>
            <div className="flex justify-center mb-4">
              <a 
                href={result.dcr_url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="btn btn-lg gap-2 bg-blue-100 text-blue-900 hover:bg-blue-200 border-blue-300 font-semibold"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
                Go to the created DCR (On The Decentriq Platform)
              </a>
            </div>
            <p className="mb-2">Title: {result.dcr_title}</p>
            <p className="mb-2">Cohorts: {result.num_cohorts}</p>
            <p className="mb-2">Metadata uploads: {result.metadata_uploads_successful}/{result.num_cohorts}</p>
            <p className="mb-2">Shuffled samples: {result.shuffled_uploads_successful}/{result.num_cohorts}</p>
            
            {cohortsWithSamples.length > 0 && (
              <p className="mb-2 text-sm">
                <span className="font-semibold">Cohorts with shuffled samples available:</span> {cohortsWithSamples.join(', ')}
              </p>
            )}
            {cohortsWithoutSamples.length > 0 && (
              <p className="mb-2 text-sm">
                <span className="font-semibold">Cohorts with no shuffled samples:</span> {cohortsWithoutSamples.join(', ')}
              </p>
            )}
          </div>
        ));
      } else {
        // Non-OK response - show detailed error information
        console.error('DCR creation failed with response:', result);
        setPublishedDCR((
          <div className="bg-error text-error-content p-4 rounded-lg space-y-2">
            <p className="font-bold text-lg">❌ Error: Failed to create live DCR</p>
            <div className="bg-black bg-opacity-20 p-3 rounded text-xs font-mono overflow-auto max-h-96">
              <p className="font-bold mb-2">Error Details:</p>
              <p className="mb-2">Status: {response.status} {response.statusText}</p>
              <p className="mb-2">Message: {result.detail || 'No error message provided'}</p>
              <p className="font-bold mt-3 mb-2">Full Response:</p>
              <pre className="whitespace-pre-wrap break-words">{JSON.stringify(result, null, 2)}</pre>
            </div>
          </div>
        ));
      }
      
      setIsLoading(false);
      setLoadingAction(null);
    } catch (error: any) {
      console.error('Error creating live DCR:', error);
      console.error('Error stack:', error?.stack);
      
      setPublishedDCR((
        <div className="bg-error text-error-content p-4 rounded-lg space-y-2">
          <p className="font-bold text-lg">❌ Error: Failed to create live DCR</p>
          <div className="bg-black bg-opacity-20 p-3 rounded text-xs font-mono overflow-auto max-h-96">
            <p className="font-bold mb-2">Exception Details:</p>
            <p className="mb-2">Type: {error?.name || 'Unknown'}</p>
            <p className="mb-2">Message: {error?.message || 'No error message'}</p>
            {error?.stack && (
              <>
                <p className="font-bold mt-3 mb-2">Stack Trace:</p>
                <pre className="whitespace-pre-wrap break-words text-xs">{error.stack}</pre>
              </>
            )}
            <p className="font-bold mt-3 mb-2">Full Error Object:</p>
            <pre className="whitespace-pre-wrap break-words">{JSON.stringify(error, Object.getOwnPropertyNames(error), 2)}</pre>
          </div>
        </div>
      ));
      setIsLoading(false);
      setLoadingAction(null);
    }
  };

  const clearCohortsList = () => {
    sessionStorage.setItem('dataCleanRoom', JSON.stringify({cohorts: {}}));
    setDataCleanRoom({cohorts: {}});
    setPublishedDCR(null);
  };

  const addAnalyst = useCallback(() => {
    const email = newAnalystEmail.trim();
    if (email && !additionalAnalysts.includes(email) && email !== userEmail) {
      setAdditionalAnalysts([...additionalAnalysts, email]);
      setNewAnalystEmail('');
    }
  }, [newAnalystEmail, additionalAnalysts, userEmail]);

  const removeAnalyst = useCallback((email: string) => {
    setAdditionalAnalysts(prev => prev.filter(e => e !== email));
  }, []);

  // Memoize cohort IDs to prevent unnecessary recalculations
  const dcrCohortIds = useMemo(() => 
    dataCleanRoom?.cohorts ? Object.keys(dataCleanRoom.cohorts) : [],
    [dataCleanRoom?.cohorts]
  );

  const dataOwners = useMemo(() => {
    const ownersMap: Record<string, string[]> = {};
    if (dcrCohortIds.length > 0 && cohortsData) {
      dcrCohortIds.forEach((cohortId) => {
        const cohort = cohortsData[cohortId];
        if (cohort?.cohort_email) {
          // Handle both string and array formats
          const emails = Array.isArray(cohort.cohort_email) 
            ? cohort.cohort_email 
            : [cohort.cohort_email];
          
          emails.forEach((email: string) => {
            // Include all data owners, even if they're the current user
            if (email && email.trim()) {
              const trimmedEmail = email.trim();
              if (!ownersMap[trimmedEmail]) {
                ownersMap[trimmedEmail] = [];
              }
              ownersMap[trimmedEmail].push(cohortId);
            }
          });
        }
      });
    }
    return Object.entries(ownersMap).map(([email, cohorts]) => ({ email, cohorts }));
  }, [dcrCohortIds, cohortsData]);

  return (
    <div className="navbar bg-base-300 min-h-0 p-0">
      <div className="navbar-start">
        <ul className="menu menu-horizontal gap-2 my-0 py-0 pl-6 lg:flex">
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
          <li> 
            <Link href="/docs_store">
              <HardDrive /> 
              Documents
            </Link>
          </li>
        </ul>
        <div className="dropdown lg:hidden">
          <div tabIndex={0} role="button" className="btn btn-ghost lg:hidden">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h8m-8 6h16" /></svg>
          </div>
          <ul tabIndex={0} className="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow bg-base-100 rounded-box w-52">
            <li><Link href="/upload">Upload</Link></li>
            <li><Link href="/cohorts">Explore</Link></li>
            <li><Link href="/mapping">Mapping</Link></li>
            <li><Link href="/docs_store">Documents</Link></li>
          </ul>
        </div>
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
          <div className="modal-box max-w-4xl">
            {/* Toggle switch for Current/Future mode */}
            <div className="flex items-center gap-3 mb-4">
              <span className="text-sm font-medium">Mode:</span>
              <div className="join">
                <button 
                  className={`btn btn-sm join-item ${dcrMode === 'current' ? 'btn-active' : ''}`}
                  onClick={() => setDcrMode('current')}
                >
                  Current
                </button>
                <button 
                  className={`btn btn-sm join-item ${dcrMode === 'future' ? 'btn-active' : ''}`}
                  onClick={() => setDcrMode('future')}
                >
                  Future (Beta)
                </button>
              </div>
            </div>
            
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
            
            {/* Checkbox for including shuffled samples - only visible in future mode */}
            {dcrMode === 'future' && (
              <>
                <div className="form-control mt-4">
                  <label className="label cursor-pointer justify-start gap-3">
                    <input 
                      type="checkbox" 
                      checked={includeShuffledSamples}
                      onChange={(e) => setIncludeShuffledSamples(e.target.checked)}
                      className="checkbox checkbox-primary" 
                    />
                    <span className="label-text">Incorporate shuffled samples</span>
                  </label>
                </div>
                
                <div className="form-control mt-2">
                  <label className="label cursor-pointer justify-start gap-3">
                    <input 
                      type="checkbox" 
                      checked={showParticipantsModal || additionalAnalysts.length > 0}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setShowParticipantsModal(true);
                        } else {
                          setShowParticipantsModal(false);
                          setAdditionalAnalysts([]);
                        }
                      }}
                      className="checkbox checkbox-primary" 
                    />
                    <span className="label-text">Add additional participants</span>
                  </label>
                </div>
                
                {/* Airlock Settings */}
                <div className="mt-4">
                  <div className="divider"></div>
                  <h3 className="font-bold text-lg mb-3">Airlock Settings</h3>
                  <p className="text-sm text-base-content/70 mb-3">Set the percentage of data (0-100) to export as a fragment for each cohort:</p>
                  <div className="space-y-3">
                    {dataCleanRoom?.cohorts && Object.keys(dataCleanRoom.cohorts).map((cohortId) => (
                      <div key={cohortId} className="flex items-center gap-2">
                        <label className="flex-1 font-medium">{cohortId}</label>
                        <input 
                          type="number"
                          min="0"
                          max="100"
                          placeholder="0"
                          className="input input-bordered w-24 text-center"
                          value={airlockSettings[cohortId] ?? 0}
                          onChange={(e) => {
                            const value = Math.min(100, Math.max(0, parseInt(e.target.value) || 0));
                            setAirlockSettings({...airlockSettings, [cohortId]: value});
                          }}
                        />
                        <span className="text-sm text-base-content/70 w-8">%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
            
            <div className="modal-action flex flex-wrap justify-end gap-2 mt-4">
                {/* <div className="flex flex-wrap space-y-2"> */}
                {dcrMode === 'future' && (
                  <button 
                    className="btn btn-primary" 
                    onClick={createLiveDCR} 
                    disabled={isLoading || publishedDCR}
                  >
                    Create Live DCR
                  </button>
                )}
                <button 
                  className="btn btn-neutral" 
                  onClick={getDCRDefinitionFile} 
                  disabled={isLoading || publishedDCR}
                >
                  Download DCR Config
                </button>
                <button 
                  className="btn btn-error" 
                  onClick={clearCohortsList}
                  disabled={publishedDCR}
                >
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
                <p>
                  {loadingAction === 'live' 
                    ? 'Creating the Data Clean Room on Decentriq Platform. Will take a few seconds...'
                    : 'Creating the file specification for a DCR draft...'}
                </p>
              </div>
            )}
            {publishedDCR && (
              <div className="card card-compact">
                <div className="card-body mt-5">
                    {publishedDCR}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Participants Management Modal */}
      {showParticipantsModal && (
        <ParticipantsModal
          dataOwners={dataOwners}
          userEmail={userEmail}
          additionalAnalysts={additionalAnalysts}
          newAnalystEmail={newAnalystEmail}
          setNewAnalystEmail={setNewAnalystEmail}
          addAnalyst={addAnalyst}
          removeAnalyst={removeAnalyst}
          onClose={() => setShowParticipantsModal(false)}
        />
      )}
    </div>
  );
}

// Memoized Participants Modal to prevent re-renders on every keystroke
const ParticipantsModal = React.memo(({
  dataOwners,
  userEmail,
  additionalAnalysts,
  newAnalystEmail,
  setNewAnalystEmail,
  addAnalyst,
  removeAnalyst,
  onClose
}: {
  dataOwners: { email: string; cohorts: string[] }[];
  userEmail: string | null;
  additionalAnalysts: string[];
  newAnalystEmail: string;
  setNewAnalystEmail: (email: string) => void;
  addAnalyst: () => void;
  removeAnalyst: (email: string) => void;
  onClose: () => void;
}) => {
  return (
    <div className="modal modal-open">
      <div className="modal-box">
        <h3 className="font-bold text-lg mb-4">DCR Participants</h3>
        
        <div className="space-y-4">
          {/* Data owners */}
          {dataOwners.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Data Owners</h4>
              {dataOwners.map((owner) => (
                <div key={owner.email} className="bg-base-200 p-3 rounded-lg mb-2">
                  <div>
                    <p className="font-semibold">
                      {owner.email}
                      {owner.email === userEmail && <span className="ml-2 text-xs badge badge-primary">You</span>}
                    </p>
                    <p className="text-sm text-gray-500">
                      Data Owner for: {owner.cohorts.join(', ')}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {/* Analysts */}
          <div>
            <h4 className="font-semibold mb-2">Analysts</h4>
            {/* Current user */}
            <div className="bg-base-200 p-3 rounded-lg mb-2">
              <div>
                <p className="font-semibold">{userEmail}</p>
                <p className="text-sm text-gray-500">
                  Analyst (You)
                  {dataOwners.some(owner => owner.email === userEmail) && ' • Also Data Owner'}
                </p>
              </div>
            </div>
            
            {/* Additional analysts */}
            {additionalAnalysts.map((email) => (
              <div key={email} className="bg-base-200 p-3 rounded-lg mb-2 flex justify-between items-center">
                <div>
                  <p className="font-semibold">{email}</p>
                  <p className="text-sm text-gray-500">Analyst</p>
                </div>
                <button 
                  className="btn btn-sm btn-error btn-outline"
                  onClick={() => removeAnalyst(email)}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
          
          {/* Add new analyst */}
          <div className="divider">Add Analyst</div>
          <div className="flex gap-2">
            <input 
              type="text"
              placeholder="Enter email address"
              className="input input-bordered flex-1"
              value={newAnalystEmail}
              onChange={(e) => setNewAnalystEmail(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && addAnalyst()}
            />
            <button 
              className="btn btn-primary"
              onClick={addAnalyst}
              disabled={!newAnalystEmail.trim()}
            >
              Add Analyst
            </button>
          </div>
        </div>
        
        <div className="modal-action">
          <button className="btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
});

ParticipantsModal.displayName = 'ParticipantsModal';
