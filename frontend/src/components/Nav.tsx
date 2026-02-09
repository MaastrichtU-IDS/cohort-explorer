'use client';

import React, {useState, useEffect, useMemo, useCallback} from 'react';
import Link from 'next/link';
import {useRouter} from 'next/router';
import {LogIn, LogOut, Compass, Upload, HardDrive, Map} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {DarkThemeIcon, LightThemeIcon} from '@/components/Icons';
import {apiUrl} from '@/utils';

// Not used: Next Auth.js: https://authjs.dev/getting-started/providers/oauth-tutorial
// Auth0: https://github.com/nextauthjs/next-auth/blob/main/packages/core/src/providers/auth0.ts
// OAuth: https://github.com/nextauthjs/next-auth/blob/main/packages/core/src/providers/oauth.ts
// https://github.com/nextauthjs/next-auth-example/blob/cc1c91a65c70e1a51bfbbb550dbc85e605f0e402/auth.ts

export function Nav() {
  const router = useRouter();
  const { pathname } = router;
  const {dataCleanRoom, setDataCleanRoom, cohortsData, setCohortsData, userEmail, setUserEmail} = useCohorts();
  const [theme, setTheme] = useState('light');
  const [showModal, setShowModal] = useState(false);
  const [publishedDCR, setPublishedDCR]: any = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingAction, setLoadingAction] = useState<'live' | 'config' | null>(null);
  const [dcrCreated, setDcrCreated] = useState(false);
  const [configDownloaded, setConfigDownloaded] = useState(false);
  const [shuffledSampleSettings, setShuffledSampleSettings] = useState<Record<string, boolean>>({});
  const [cohortsWithShuffledSamples, setCohortsWithShuffledSamples] = useState<string[]>([]);
  const [cohortsWithoutShuffledSamples, setCohortsWithoutShuffledSamples] = useState<string[]>([]);
  const [loadingShuffledSamples, setLoadingShuffledSamples] = useState(false);
  const [showParticipantsModal, setShowParticipantsModal] = useState(false);
  const [additionalAnalysts, setAdditionalAnalysts] = useState<string[]>([]);
  const [newAnalystEmail, setNewAnalystEmail] = useState('');
  const [airlockSettings, setAirlockSettings] = useState<Record<string, boolean>>({});
  const [participantsPreview, setParticipantsPreview] = useState<any>(null);
  const [loadingParticipants, setLoadingParticipants] = useState(false);
  const [excludedDataOwners, setExcludedDataOwners] = useState<string[]>([]);
  const [dcrName, setDcrName] = useState('');
  const [availableMappingFiles, setAvailableMappingFiles] = useState<any[]>([]);
  const [selectedMappingFiles, setSelectedMappingFiles] = useState<Record<string, boolean>>({});
  const [loadingMappingFiles, setLoadingMappingFiles] = useState(false);
  const [includeMappingUploadSlot, setIncludeMappingUploadSlot] = useState(false);
  const [wizardMode, setWizardMode] = useState(true);
  const [wizardStep, setWizardStep] = useState(0);
  const notificationRef = React.useRef<HTMLDivElement>(null);
  
  // Wizard step definitions
  const wizardSteps = [
    { id: 'name', title: 'DCR Name & Cohorts' },
    { id: 'participants', title: 'Participants' },
    { id: 'airlock', title: 'Airlock Settings' },
    { id: 'shuffled', title: 'Shuffled Samples' },
    { id: 'mapping', title: 'Mapping Files' },
    { id: 'review', title: 'Review & Create' },
  ];
  
  // Helper function to scroll to notification box
  const scrollToNotification = () => {
    setTimeout(() => {
      notificationRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }, 100);
  };
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
    scrollToNotification();
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
          include_shuffled_samples: shuffledSampleSettings,
          dcr_name: dcrName
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
        setConfigDownloaded(true);
        setPublishedDCR((
          <p>✅ Data Clean Room configuration package (with shuffled samples) has been downloaded. <br />
          Please go to <a href="https://platform.decentriq.com/" target="_blank" className="underline text-blue-600 hover:text-blue-800">https://platform.decentriq.com</a> to create a new DCR from the configuration file. </p>
        ));
        scrollToNotification();
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
        setConfigDownloaded(true);
        setPublishedDCR((
          <p>✅ Data Clean Room configuration file has been downloaded. <br />
          Please go to <a href="https://platform.decentriq.com/" target="_blank" className="underline text-blue-600 hover:text-blue-800">https://platform.decentriq.com</a> to create a new DCR from the configuration file. </p>
        ));
        scrollToNotification();
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
    scrollToNotification();
    try {
      const response = await fetch(`${apiUrl}/create-live-compute-dcr`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...dataCleanRoom,
          include_shuffled_samples: shuffledSampleSettings,
          additional_analysts: additionalAnalysts,
          excluded_data_owners: excludedDataOwners,
          airlock_settings: Object.fromEntries(
            Object.entries(airlockSettings).map(([cohortId, isEnabled]) => [
              cohortId,
              isEnabled ? 20 : 0
            ])
          ),
          dcr_name: dcrName,
          selected_mapping_files: availableMappingFiles
            .filter(m => selectedMappingFiles[m.filename] !== false)
            .map(m => ({ filename: m.filename, filepath: m.filepath, display_name: m.display_name })),
          include_mapping_upload_slot: includeMappingUploadSlot
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
        
        setDcrCreated(true);
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
        scrollToNotification();
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
        scrollToNotification();
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
      scrollToNotification();
      setIsLoading(false);
      setLoadingAction(null);
    }
  };

  const clearCohortsList = () => {
    sessionStorage.setItem('dataCleanRoom', JSON.stringify({cohorts: {}}));
    setDataCleanRoom({cohorts: {}});
    setPublishedDCR(null);
    setDcrCreated(false);
    setConfigDownloaded(false);
    setShuffledSampleSettings({});
    setCohortsWithShuffledSamples([]);
    setCohortsWithoutShuffledSamples([]);
    setAdditionalAnalysts([]);
    setAirlockSettings({});
    setDcrName('');
    setExcludedDataOwners([]);
    setAvailableMappingFiles([]);
    setSelectedMappingFiles({});
    setIncludeMappingUploadSlot(false);
    setWizardMode(false);
    setWizardStep(0);
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

  // Track cohort IDs as a string for dependency comparison
  const cohortIdsKey = useMemo(() => 
    Object.keys(dataCleanRoom?.cohorts || {}).sort().join(','), 
    [dataCleanRoom?.cohorts]
  );

  // Fetch shuffled sample availability when modal opens or cohorts change
  useEffect(() => {
    if (showModal && dataCleanRoom?.cohorts && Object.keys(dataCleanRoom.cohorts).length > 0) {
      const fetchShuffledSamples = async () => {
        setLoadingShuffledSamples(true);
        try {
          const response = await fetch(`${apiUrl}/check-shuffled-samples`, {
            method: 'POST',
            credentials: 'include',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              cohorts: dataCleanRoom.cohorts
            })
          });
          
          if (response.ok) {
            const result = await response.json();
            setCohortsWithShuffledSamples(result.cohorts_with_samples);
            setCohortsWithoutShuffledSamples(result.cohorts_without_samples);
            // Initialize shuffled sample settings - default to true for cohorts that have samples
            const initialSettings: Record<string, boolean> = {};
            result.cohorts_with_samples.forEach((cohortId: string) => {
              initialSettings[cohortId] = shuffledSampleSettings[cohortId] ?? true;
            });
            setShuffledSampleSettings(prev => ({...prev, ...initialSettings}));
          }
        } catch (error) {
          console.error('Failed to fetch shuffled samples availability:', error);
        } finally {
          setLoadingShuffledSamples(false);
        }
      };
      
      fetchShuffledSamples();
    }
  }, [showModal, cohortIdsKey]);

  // Fetch available mapping files when modal opens
  useEffect(() => {
    if (showModal && dataCleanRoom?.cohorts && Object.keys(dataCleanRoom.cohorts).length > 1) {
      const fetchMappingFiles = async () => {
        setLoadingMappingFiles(true);
        try {
          const cohortIds = Object.keys(dataCleanRoom.cohorts);
          const response = await fetch(`${apiUrl}/api/get-available-mapping-files`, {
            method: 'POST',
            credentials: 'include',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(cohortIds)
          });
          
          if (response.ok) {
            const result = await response.json();
            setAvailableMappingFiles(result.available_mappings);
            // Initialize selection - default to false (unselected) for all available mappings
            const initialSettings: Record<string, boolean> = {};
            result.available_mappings.forEach((mapping: any) => {
              initialSettings[mapping.filename] = selectedMappingFiles[mapping.filename] ?? false;
            });
            setSelectedMappingFiles(prev => ({...prev, ...initialSettings}));
          }
        } catch (error) {
          console.error('Failed to fetch available mapping files:', error);
        } finally {
          setLoadingMappingFiles(false);
        }
      };
      
      fetchMappingFiles();
    }
  }, [showModal, cohortIdsKey]);

  // Fetch participants preview when modal opens
  useEffect(() => {
    if (showParticipantsModal && dataCleanRoom?.cohorts) {
      const fetchParticipants = async () => {
        setLoadingParticipants(true);
        try {
          const response = await fetch(`${apiUrl}/preview-dcr-participants`, {
            method: 'POST',
            credentials: 'include',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              cohorts: dataCleanRoom.cohorts,
              additional_analysts: additionalAnalysts
            })
          });
          
          if (response.ok) {
            const result = await response.json();
            setParticipantsPreview(result.participants);
          }
        } catch (error) {
          console.error('Failed to fetch participants preview:', error);
        } finally {
          setLoadingParticipants(false);
        }
      };
      
      fetchParticipants();
    }
  }, [showParticipantsModal, dataCleanRoom?.cohorts, additionalAnalysts]);

  // Memoize cohort IDs to prevent unnecessary recalculations
  const dcrCohortIds = useMemo(() => 
    dataCleanRoom?.cohorts ? Object.keys(dataCleanRoom.cohorts) : [],
    [dataCleanRoom?.cohorts]
  );

  // Use participants preview from backend when available
  const dataOwners = useMemo((): { email: string; cohorts: string[] }[] => {
    if (participantsPreview) {
      // Create an object to deduplicate owners by email
      const ownersMap: Record<string, Set<string>> = {};
      
      Object.entries(participantsPreview).forEach(([email, roles]: [string, any]) => {
        if (roles.data_owner_of && roles.data_owner_of.length > 0) {
          // Extract cohort names from node IDs (remove suffixes like _metadata_dictionary, _shuffled_sample)
          if (!ownersMap[email]) {
            ownersMap[email] = new Set();
          }
          
          roles.data_owner_of.forEach((nodeId: string) => {
            // Remove common suffixes to get the base cohort name
            const cohortName = nodeId
              .replace(/_metadata_dictionary$/, '')
              .replace(/_shuffled_sample$/, '')
              .replace(/-/g, ' '); // Convert hyphens back to spaces for display
            ownersMap[email].add(cohortName);
          });
        }
      });
      
      // Convert object to array format
      return Object.entries(ownersMap).map(([email, cohortSet]) => ({
        email,
        cohorts: Array.from(cohortSet).sort()
      }));
    }
    return [];
  }, [participantsPreview]);

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
          {(pathname === '/' || pathname === '/cohorts' || pathname === '/mapping') && (
            <button id="dcr-button" onClick={() => { setShowModal(true); setWizardMode(true); setWizardStep(0); }} className="btn btn-outline btn-lg shadow-md hover:shadow-lg hover:bg-gray-600 hover:text-white transition-all duration-300">
              Create a Data Clean Room <div className="badge badge-neutral">{Object.keys(dataCleanRoom?.cohorts).length || 0}</div>
            </button>
          )}

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
          <div className="modal-box max-w-4xl max-h-[85vh]">
            {/* Toggle link between standard and wizard view */}
            <div className="flex justify-end mb-2">
              <button 
                className="text-xs text-primary hover:underline"
                onClick={() => { setWizardMode(!wizardMode); setWizardStep(0); }}
              >
                {wizardMode ? 'Switch to 1-page view →' : '← Back to step-by-step wizard'}
              </button>
            </div>
            
            {wizardMode ? (
              /* ========== WIZARD VIEW ========== */
              <>
                {/* Step indicator */}
                <ul className="steps steps-horizontal w-full mb-6 text-xs">
                  {wizardSteps.map((step, idx) => (
                    <li 
                      key={step.id} 
                      className={`step ${idx <= wizardStep ? 'step-primary' : ''} cursor-pointer`}
                      onClick={() => setWizardStep(idx)}
                    >
                      {step.title}
                    </li>
                  ))}
                </ul>
                
                {/* Step content */}
                <div className="min-h-[300px]">
                  {/* Step 0: DCR Name & Cohorts */}
                  {wizardStep === 0 && (
                    <>
                      <h3 className="font-bold text-lg mb-4">Step 1: DCR Name & Cohorts</h3>
                      <div className="form-control mb-4">
                        <label className="label">
                          <span className="label-text font-semibold">DCR Name</span>
                        </label>
                        <input 
                          type="text"
                          placeholder="iCARE4CVD DCR compute XXX"
                          className="input input-bordered w-full"
                          value={dcrName}
                          onChange={(e) => setDcrName(e.target.value)}
                        />
                        <span className="text-xs text-base-content/60 mt-1">
                          Leave empty to use the default naming. Your email ({userEmail || 'not logged in'}) will be appended to the title for clarity.
                        </span>
                      </div>
                      <div className="mt-4">
                        <label className="label">
                          <span className="label-text font-semibold">Selected Cohorts</span>
                        </label>
                        <div className="bg-base-200 p-3 rounded-lg">
                          {Object.entries(dataCleanRoom?.cohorts || {}).map(([cohortId, variables]: any) => (
                            <div key={cohortId} className="badge badge-outline mr-2 mb-1">
                              {cohortId} ({variables.length} vars)
                            </div>
                          ))}
                        </div>
                      </div>
                    </>
                  )}
                  
                  {/* Step 1: Participants */}
                  {wizardStep === 1 && (
                    <>
                      <h3 className="font-bold text-lg mb-4">Step 2: Manage Participants</h3>
                      <p className="text-sm text-base-content/70 mb-4">
                        Invite additional analysts to the DCR.
                      </p>
                      <button 
                        className="btn btn-outline"
                        onClick={() => setShowParticipantsModal(true)}
                      >
                        Edit Participants List
                      </button>
                      {(additionalAnalysts.length > 0 || excludedDataOwners.length > 0) && (
                        <div className="mt-4 p-3 bg-base-200 rounded-lg">
                          {additionalAnalysts.length > 0 && (
                            <p className="text-sm">
                              <strong>Additional analysts:</strong> {additionalAnalysts.join(', ')}
                            </p>
                          )}
                          {excludedDataOwners.length > 0 && (
                            <p className="text-sm mt-1">
                              <strong>Excluded data owners:</strong> {excludedDataOwners.join(', ')}
                            </p>
                          )}
                        </div>
                      )}
                    </>
                  )}
                  
                  {/* Step 2: Airlock Settings */}
                  {wizardStep === 2 && (
                    <>
                      <h3 className="font-bold text-lg mb-4">Step 3: Airlock Settings</h3>
                      <p className="text-sm text-base-content/70 mb-4">
                        Select which cohorts to include in the airlock (the airlock allows 20% of the data to be visible to analysts inside the DCR).
                      </p>
                      <div className="space-y-2">
                        {dataCleanRoom?.cohorts && Object.keys(dataCleanRoom.cohorts).map((cohortId) => (
                          <div key={cohortId} className="form-control">
                            <label className="label cursor-pointer justify-start gap-3">
                              <input 
                                type="checkbox"
                                checked={airlockSettings[cohortId] ?? true}
                                onChange={(e) => {
                                  setAirlockSettings({...airlockSettings, [cohortId]: e.target.checked});
                                }}
                                className="checkbox checkbox-primary"
                              />
                              <span className="label-text text-base">{cohortId}</span>
                            </label>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                  
                  {/* Step 3: Shuffled Samples */}
                  {wizardStep === 3 && (
                    <>
                      <h3 className="font-bold text-lg mb-4">Step 4: Shuffled Samples</h3>
                      {loadingShuffledSamples ? (
                        <p className="text-sm text-base-content/70">Checking shuffled sample availability...</p>
                      ) : (
                        <>
                          {cohortsWithShuffledSamples.length > 0 ? (
                            <>
                              <p className="text-sm text-base-content/70 mb-4">Select which shuffled samples to include:</p>
                              <div className="space-y-2">
                                {cohortsWithShuffledSamples.map((cohortId) => (
                                  <div key={cohortId} className="form-control">
                                    <label className="label cursor-pointer justify-start gap-3">
                                      <input 
                                        type="checkbox"
                                        checked={shuffledSampleSettings[cohortId] ?? true}
                                        onChange={(e) => {
                                          setShuffledSampleSettings({...shuffledSampleSettings, [cohortId]: e.target.checked});
                                        }}
                                        className="checkbox checkbox-primary"
                                      />
                                      <span className="label-text text-base">{cohortId}</span>
                                    </label>
                                  </div>
                                ))}
                              </div>
                            </>
                          ) : (
                            <p className="text-sm text-base-content/50 italic">
                              None of the selected cohorts have shuffled samples available.
                            </p>
                          )}
                          {cohortsWithoutShuffledSamples.length > 0 && cohortsWithShuffledSamples.length > 0 && (
                            <p className="text-sm text-base-content/50 mt-4 italic">
                              Cohorts without shuffled samples: {cohortsWithoutShuffledSamples.join(', ')}
                            </p>
                          )}
                        </>
                      )}
                    </>
                  )}
                  
                  {/* Step 4: Mapping Files */}
                  {wizardStep === 4 && (
                    <>
                      <h3 className="font-bold text-lg mb-4">Step 5: Mapping Files (Optional)</h3>
                      {loadingMappingFiles ? (
                        <p className="text-sm text-base-content/70">Checking for available mapping files...</p>
                      ) : availableMappingFiles.length > 0 ? (
                        <>
                          <p className="text-sm text-base-content/70 mb-4">Select which mapping files to include in the DCR (optional):</p>
                          <div className="space-y-2">
                            {availableMappingFiles.map((mapping) => (
                              <div key={mapping.filename} className="form-control">
                                <label className="label cursor-pointer justify-start gap-3">
                                  <input 
                                    type="checkbox"
                                    checked={selectedMappingFiles[mapping.filename] ?? true}
                                    onChange={(e) => {
                                      setSelectedMappingFiles({...selectedMappingFiles, [mapping.filename]: e.target.checked});
                                    }}
                                    className="checkbox checkbox-primary"
                                  />
                                  <span className="label-text text-base">{mapping.display_name}</span>
                                </label>
                              </div>
                            ))}
                          </div>
                        </>
                      ) : Object.keys(dataCleanRoom?.cohorts || {}).length < 2 ? (
                        <p className="text-sm text-base-content/50 italic">Select at least 2 cohorts to see available mapping files.</p>
                      ) : (
                        <p className="text-sm text-base-content/50 italic">No mapping files available for the selected cohorts.</p>
                      )}
                      
                      <div className="form-control mt-6">
                        <label className="label cursor-pointer justify-start gap-3">
                          <input 
                            type="checkbox"
                            checked={includeMappingUploadSlot}
                            onChange={(e) => setIncludeMappingUploadSlot(e.target.checked)}
                            className="checkbox checkbox-primary"
                          />
                          <span className="label-text text-base">Include a file upload slot for cross-study mapping</span>
                        </label>
                      </div>
                      
                      <p className="text-xs text-base-content/50 mt-4 italic">
                        Missing a mapping file? Generate it from the <Link href="/mapping" className="underline hover:text-primary">Mapping page</Link>.
                      </p>
                    </>
                  )}
                  
                  {/* Step 5: Review & Create */}
                  {wizardStep === 5 && (
                    <>
                      <h3 className="font-bold text-lg mb-4">Step 6: Review & Create</h3>
                      <div className="space-y-3 text-sm">
                        <div className="p-3 bg-base-200 rounded-lg">
                          <strong>DCR Name:</strong> {dcrName || '(default naming)'}
                        </div>
                        <div className="p-3 bg-base-200 rounded-lg">
                          <strong>Cohorts:</strong> {Object.keys(dataCleanRoom?.cohorts || {}).join(', ')}
                        </div>
                        <div className="p-3 bg-base-200 rounded-lg">
                          <strong>Additional Analysts:</strong> {additionalAnalysts.length > 0 ? additionalAnalysts.join(', ') : 'None'}
                        </div>
                        <div className="p-3 bg-base-200 rounded-lg">
                          <strong>Airlock Cohorts:</strong> {Object.entries(airlockSettings).filter(([_, v]) => v !== false).map(([k]) => k).join(', ') || 'All selected'}
                        </div>
                        <div className="p-3 bg-base-200 rounded-lg">
                          <strong>Shuffled Samples:</strong> {Object.entries(shuffledSampleSettings).filter(([_, v]) => v !== false).map(([k]) => k).join(', ') || 'None selected'}
                        </div>
                        <div className="p-3 bg-base-200 rounded-lg">
                          <strong>Mapping Files:</strong> {availableMappingFiles.filter(m => selectedMappingFiles[m.filename] !== false).map(m => m.display_name).join(', ') || 'None'}
                          {includeMappingUploadSlot && ' + Upload slot'}
                        </div>
                      </div>
                      
                      <div className="flex flex-wrap gap-2 mt-6">
                        <button 
                          className="btn btn-primary" 
                          onClick={createLiveDCR} 
                          disabled={isLoading || dcrCreated}
                        >
                          Create Live DCR
                        </button>
                        <button 
                          className="btn btn-neutral" 
                          onClick={getDCRDefinitionFile} 
                          disabled={isLoading || configDownloaded}
                        >
                          Download DCR Config
                        </button>
                      </div>
                    </>
                  )}
                </div>
                
                {/* Wizard navigation buttons */}
                <div className="flex justify-between mt-6 pt-4 border-t">
                  <div>
                    {wizardStep > 0 && (
                      <button 
                        className="btn btn-outline"
                        onClick={() => setWizardStep(wizardStep - 1)}
                      >
                        ← Previous
                      </button>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <button className="btn" onClick={() => setShowModal(false)}>
                      Close
                    </button>
                    {wizardStep < wizardSteps.length - 1 && (
                      <button 
                        className="btn btn-primary"
                        onClick={() => setWizardStep(wizardStep + 1)}
                      >
                        Next →
                      </button>
                    )}
                  </div>
                </div>
                
                {/* Loading and notification for wizard */}
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
                <div ref={wizardStep === 5 ? notificationRef : undefined}>
                  {publishedDCR && wizardStep === 5 && (
                    <div className="card card-compact">
                      <div className="card-body mt-5">
                        {publishedDCR}
                      </div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              /* ========== STANDARD VIEW ========== */
              <>
                {/* DCR Name field */}
                <div className="form-control mb-3">
                    <div className="flex items-center gap-3">
                      <span className="label-text font-semibold whitespace-nowrap">DCR Name</span>
                      <input 
                        type="text"
                        placeholder="iCARE4CVD DCR compute XXX"
                        className="input input-bordered w-full input-sm"
                        value={dcrName}
                        onChange={(e) => setDcrName(e.target.value)}
                      />
                    </div>
                    <span className="text-xs text-base-content/60 mt-1">
                      Leave empty to use the default naming. Note: your email address, &quot; - created by {userEmail}&quot;, will be appended to the name.
                    </span>
                </div>
                
                <h3 className="font-bold text-lg mb-3">Cohorts to load in Decentriq Data Clean Room</h3>
                <p className="flex flex-wrap gap-x-1">
                  {Object.entries(dataCleanRoom?.cohorts).map(([cohortId, variables]: any, index, arr) => (
                    <span key={cohortId}>
                      {cohortId} ({variables.length} variables){index < arr.length - 1 ? ',' : ''}
                    </span>
                  ))}
                </p>
                
                {/* Other settings */}
                    
                    <div className="mt-2">
                      <button 
                        className="btn btn-sm btn-outline"
                        onClick={() => setShowParticipantsModal(true)}
                      >
                        Manage Participants
                      </button>
                      {(additionalAnalysts.length > 0 || excludedDataOwners.length > 0) && (
                        <span className="ml-2 text-sm text-base-content/70">
                          {additionalAnalysts.length > 0 && `+${additionalAnalysts.length} analyst${additionalAnalysts.length > 1 ? 's' : ''}`}
                          {additionalAnalysts.length > 0 && excludedDataOwners.length > 0 && ', '}
                          {excludedDataOwners.length > 0 && `${excludedDataOwners.length} data owner${excludedDataOwners.length > 1 ? 's' : ''} excluded`}
                        </span>
                      )}
                    </div>
                    
                    {/* Airlock Settings */}
                    <div className="mt-2">
                      <div className="divider my-2"></div>
                      <h3 className="font-bold text-lg mb-1">Airlock Settings</h3>
                      <p className="text-sm text-base-content/70 mb-2">Select the cohorts to include in the airlock (20% of the data will be visible to analysts inside the DCR):</p>
                      <div className="flex flex-wrap gap-x-4 gap-y-1">
                        {dataCleanRoom?.cohorts && Object.keys(dataCleanRoom.cohorts).map((cohortId) => (
                          <div key={cohortId} className="form-control">
                            <label className="label cursor-pointer justify-start gap-2 py-1">
                              <input 
                                type="checkbox"
                                checked={airlockSettings[cohortId] ?? true}
                                onChange={(e) => {
                                  setAirlockSettings({...airlockSettings, [cohortId]: e.target.checked});
                                }}
                                className="checkbox checkbox-primary"
                              />
                              <span className="label-text">{cohortId}</span>
                            </label>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    {/* Shuffled Samples Settings */}
                    <div className="mt-2">
                      <div className="divider my-2"></div>
                      <h3 className="font-bold text-lg mb-1">Shuffled Samples Settings</h3>
                      {loadingShuffledSamples ? (
                        <p className="text-sm text-base-content/70">Checking shuffled sample availability...</p>
                      ) : (
                        <>
                          {cohortsWithShuffledSamples.length > 0 && (
                            <>
                              <p className="text-sm text-base-content/70 mb-2">Select the shuffled samples to include:</p>
                              <div className="flex flex-wrap gap-x-4 gap-y-1">
                                {cohortsWithShuffledSamples.map((cohortId) => (
                                  <div key={cohortId} className="form-control">
                                    <label className="label cursor-pointer justify-start gap-2 py-1">
                                      <input 
                                        type="checkbox"
                                        checked={shuffledSampleSettings[cohortId] ?? true}
                                        onChange={(e) => {
                                          setShuffledSampleSettings({...shuffledSampleSettings, [cohortId]: e.target.checked});
                                        }}
                                        className="checkbox checkbox-primary"
                                      />
                                      <span className="label-text">{cohortId}</span>
                                    </label>
                                  </div>
                                ))}
                              </div>
                            </>
                          )}
                          {cohortsWithoutShuffledSamples.length > 0 && (
                            <p className="text-sm text-base-content/50 mt-3 italic">
                              Cohorts without shuffled samples: {cohortsWithoutShuffledSamples.join(', ')}
                            </p>
                          )}
                          {cohortsWithShuffledSamples.length === 0 && cohortsWithoutShuffledSamples.length === 0 && (
                            <p className="text-sm text-base-content/50 italic">None of the selected cohorts have a shuffled sample</p>
                          )}
                        </>
                      )}
                    </div>
                    
                    {/* Mapping File Settings */}
                    <div className="mt-2">
                      <div className="divider my-2"></div>
                      <h3 className="font-bold text-lg mb-1">Mapping File Settings</h3>
                      {loadingMappingFiles ? (
                        <p className="text-sm text-base-content/70">Checking for available mapping files...</p>
                      ) : availableMappingFiles.length > 0 ? (
                        <>
                          <p className="text-sm text-base-content/70 mb-2">Select mapping files to include in the DCR:</p>
                          <div className="flex flex-wrap gap-x-4 gap-y-1">
                            {availableMappingFiles.map((mapping) => (
                              <div key={mapping.filename} className="form-control">
                                <label className="label cursor-pointer justify-start gap-2 py-1">
                                  <input 
                                    type="checkbox"
                                    checked={selectedMappingFiles[mapping.filename] ?? true}
                                    onChange={(e) => {
                                      setSelectedMappingFiles({...selectedMappingFiles, [mapping.filename]: e.target.checked});
                                    }}
                                    className="checkbox checkbox-primary"
                                  />
                                  <span className="label-text">{mapping.display_name}</span>
                                </label>
                              </div>
                            ))}
                          </div>
                        </>
                      ) : Object.keys(dataCleanRoom?.cohorts || {}).length < 2 ? (
                        <p className="text-sm text-base-content/50 italic">Select at least 2 cohorts to see available mapping files</p>
                      ) : (
                        <p className="text-sm text-base-content/50 italic">No mapping files available for the selected cohorts</p>
                      )}
                      
                      {/* Option to include upload slot for mapping file */}
                      <div className="form-control mt-4">
                        <label className="label cursor-pointer justify-start gap-2">
                          <input 
                            type="checkbox"
                            checked={includeMappingUploadSlot}
                            onChange={(e) => setIncludeMappingUploadSlot(e.target.checked)}
                            className="checkbox checkbox-primary"
                          />
                          <span className="label-text">Include file upload slot for cross-study mapping</span>
                        </label>
                      </div>
                      
                      <p className="text-xs text-base-content/50 mt-3 italic">
                        Missing a mapping file? Generate it from the <Link href="/mapping" className="underline hover:text-primary">Mapping page</Link> to add it to the cache.
                      </p>
                    </div>
                
                <div className="modal-action flex flex-wrap justify-end gap-2 mt-4">
                    <button 
                      className="btn btn-primary" 
                      onClick={createLiveDCR} 
                      disabled={isLoading || dcrCreated}
                    >
                      Create Live DCR
                    </button>
                    <button 
                      className="btn btn-neutral" 
                      onClick={getDCRDefinitionFile} 
                      disabled={isLoading || configDownloaded}
                    >
                      Download DCR Config
                    </button>
                    <button 
                      className="btn" 
                      onClick={clearCohortsList}
                    >
                      Clear cohorts
                    </button>
                    <button className="btn" onClick={() => setShowModal(false)}>
                      Close
                    </button>
                </div>
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
                <div ref={notificationRef}>
                  {publishedDCR && (
                    <div className="card card-compact">
                      <div className="card-body mt-5">
                          {publishedDCR}
                      </div>
                    </div>
                  )}
                </div>
              </>
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
          excludedDataOwners={excludedDataOwners}
          setExcludedDataOwners={setExcludedDataOwners}
          onClose={() => setShowParticipantsModal(false)}
          isLoading={loadingParticipants}
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
  excludedDataOwners,
  setExcludedDataOwners,
  onClose,
  isLoading
}: {
  dataOwners: { email: string; cohorts: string[] }[];
  userEmail: string | null;
  additionalAnalysts: string[];
  newAnalystEmail: string;
  setNewAnalystEmail: (email: string) => void;
  addAnalyst: () => void;
  removeAnalyst: (email: string) => void;
  excludedDataOwners: string[];
  setExcludedDataOwners: (emails: string[]) => void;
  onClose: () => void;
  isLoading: boolean;
}) => {
  const toggleDataOwner = (email: string) => {
    if (excludedDataOwners.includes(email)) {
      setExcludedDataOwners(excludedDataOwners.filter(e => e !== email));
    } else {
      setExcludedDataOwners([...excludedDataOwners, email]);
    }
  };
  return (
    <div className="modal modal-open">
      <div className="modal-box">
        <h3 className="font-bold text-lg mb-4">DCR Participants</h3>
        
        <div className="space-y-4">
          {/* Data owners */}
          <div>
            <h4 className="font-semibold mb-2">Data Owners</h4>
            {isLoading ? (
              <div className="bg-base-200 p-3 rounded-lg mb-2">
                <p className="text-sm text-gray-500">
                  Retrieving list of data owners for the selected cohorts...
                </p>
              </div>
            ) : dataOwners.length > 0 ? (
              dataOwners.map((owner) => (
                <div key={owner.email} className={`p-3 rounded-lg mb-2 flex items-start gap-3 ${excludedDataOwners.includes(owner.email) ? 'bg-base-200 opacity-50' : 'bg-base-200'}`}>
                  <input
                    type="checkbox"
                    checked={!excludedDataOwners.includes(owner.email)}
                    onChange={() => toggleDataOwner(owner.email)}
                    className="checkbox checkbox-primary mt-1"
                  />
                  <div className="flex-1">
                    <p className={`font-semibold ${excludedDataOwners.includes(owner.email) ? 'line-through' : ''}`}>
                      {owner.email}
                      {owner.email === userEmail && <span className="ml-2 text-xs badge badge-primary">You</span>}
                    </p>
                    <p className="text-sm text-gray-500">
                      Data Owner for: {owner.cohorts.join(', ')}
                    </p>
                  </div>
                </div>
              ))
            ) : (
              <div className="bg-warning bg-opacity-20 p-3 rounded-lg mb-2">
                <p className="text-sm text-gray-700">
                  ⚠️ No data owner emails found for the selected cohorts. Please ensure the cohort metadata includes data owner contact information.
                </p>
              </div>
            )}
          </div>
          
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
