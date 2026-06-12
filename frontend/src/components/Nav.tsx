'use client';

import React, {useState, useEffect, useMemo, useCallback, useRef} from 'react';
import Link from 'next/link';
import {useRouter} from 'next/router';
import {LogIn, LogOut, Compass, Upload, HardDrive, Map as MapIcon, Box, FileText, Settings, Check, Activity} from 'react-feather';
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
  // Track owners the user has explicitly opted back in. By default ALL data
  // owners are excluded (boxes unchecked). To include one, the user checks
  // the box; that adds the email to manuallyIncludedOwners.
  const [manuallyIncludedOwners, setManuallyIncludedOwners] = useState<string[]>([]);
  const [dcrName, setDcrName] = useState('');
  // True once the user has explicitly edited the DCR name: prevents our
  // "update default on cohort change" effect from overwriting their choice.
  const [dcrNameCustomized, setDcrNameCustomized] = useState(false);
  // When true the Step 1 name field is shown as an editable input; otherwise
  // it is displayed read-only with a pencil button to switch to edit mode.
  const [isEditingDcrName, setIsEditingDcrName] = useState(false);
  const [availableMappingFiles, setAvailableMappingFiles] = useState<any[]>([]);
  const [selectedMappingFiles, setSelectedMappingFiles] = useState<Record<string, boolean>>({});
  const [loadingMappingFiles, setLoadingMappingFiles] = useState(false);
  const [includeMappingUploadSlot, setIncludeMappingUploadSlot] = useState(false);
  const [wizardStep, setWizardStep] = useState(0);
  const [researchQuestion, setResearchQuestion] = useState('');
  // ICD-10 state for DCR wizard (research goals step)
  const [dcrDiseaseCodes, setDcrDiseaseCodes] = useState<{code: string; label: string; kind: string}[]>([]);
  const [dcrIcd10Entries, setDcrIcd10Entries] = useState<{code: string; label: string; kind: string; parent?: string | null}[]>([]);
  const [dcrIcd10Loading, setDcrIcd10Loading] = useState(false);
  const [dcrIcd10LoadFailed, setDcrIcd10LoadFailed] = useState(false);
  const [dcrIcd10SuggestionsOpen, setDcrIcd10SuggestionsOpen] = useState(false);
  const [dcrIcd10SearchQuery, setDcrIcd10SearchQuery] = useState('');
  const [dcrIcd10BrowseOpen, setDcrIcd10BrowseOpen] = useState(false);
  const [dcrBrowseExpanded, setDcrBrowseExpanded] = useState<Set<string>>(new Set());
  const [dcrBrowseSearch, setDcrBrowseSearch] = useState('');
  const [dcrBrowseSelected, setDcrBrowseSelected] = useState<{code: string; label: string; kind: string} | null>(null);
  const [dcrOrgType, setDcrOrgType] = useState<'' | 'for-profit' | 'non-profit'>('');
  const [dcrOrgCountry, setDcrOrgCountry] = useState('');
  const [dcrOrgName, setDcrOrgName] = useState('');
  const [dcrOrgDropdownOpen, setDcrOrgDropdownOpen] = useState(false);
  const [dcrIntendedUses, setDcrIntendedUses] = useState<string[]>([]);
  const [dcrPurpose, setDcrPurpose] = useState<number>(0);
  // Requester auth state for blockchain integration
  const [dcrRequesterAuthResult, setDcrRequesterAuthResult] = useState<any>(null);
  const [dcrBlockchainToken, setDcrBlockchainToken] = useState<string | null>(null);
  const [dcrProfileResult, setDcrProfileResult] = useState<any>(null);
  const [dcrAccessRequestResult, setDcrAccessRequestResult] = useState<Record<string, any>>({});
  const [dcrCohortLoading, setDcrCohortLoading] = useState<Record<string, boolean>>({});
  const [dcrRequesterLoading, setDcrRequesterLoading] = useState(false);
  const [dcrIrbApprovalId, setDcrIrbApprovalId] = useState('');
  const [dcrRequesterType, setDcrRequesterType] = useState<'' | 'ACADEMIC' | 'NONPROFIT' | 'PROFIT' | 'GOVERNMENT' | 'INDIVIDUAL'>('');
  const [dcrPublicProfile, setDcrPublicProfile] = useState(false);
  const [dcrGaslessOptIn, setDcrGaslessOptIn] = useState(false);
  const [showAddCohortModal, setShowAddCohortModal] = useState(false);
  const [cohortSearchQuery, setCohortSearchQuery] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const notificationRef = React.useRef<HTMLDivElement>(null);
  const dcrIcd10WrapperRef = useRef<HTMLDivElement>(null);

  // Check admin status
  useEffect(() => {
    if (!userEmail) return;
    fetch(`${apiUrl}/admin/check`, {credentials: 'include'})
      .then(res => res.ok ? res.json() : null)
      .then(data => { if (data) setIsAdmin(data.is_admin); })
      .catch(() => {});
  }, [userEmail]);

  // Wizard step definitions (declared before the logging hooks below so their
  // closures can read step ids safely).
  const wizardSteps = [
    { id: 'name', title: 'DCR Name & Cohorts' },
    { id: 'participants', title: 'Participants' },
    { id: 'research-goals', title: 'Research Goals' },
    { id: 'data-samples', title: 'Data Samples' },
    { id: 'mapping', title: 'Mapping Files' },
    { id: 'review', title: 'Review & Create' },
  ];

  // --- DCR activity logging (wizard) -------------------------------------
  // Session id correlates every event a user emits during one wizard session.
  // Kept in a ref so it is stable across renders without triggering re-renders.
  const sessionIdRef = useRef<string | null>(null);
  // Timestamp when the current wizard step was entered. Used to compute
  // time_on_step_seconds when the user leaves a step, closes the wizard,
  // or the tab unloads.
  const stepEnteredAtRef = useRef<number>(Date.now());
  // Tracks which step the user is currently on so we can compute the time
  // on the *previous* step when wizardStep changes.
  const currentStepRef = useRef<number>(0);
  // Becomes true once the user successfully publishes or downloads; used to
  // decide whether a close/unload counts as "closed" vs "abandoned".
  const wizardCompletedRef = useRef<boolean>(false);

  const logWizardEvent = useCallback(
    (event: string, extra: Record<string, any> = {}, opts?: { beacon?: boolean }) => {
      if (!userEmail || !sessionIdRef.current) return;
      const payload = {
        event,
        session_id: sessionIdRef.current,
        ...extra,
      };
      const url = `${apiUrl}/dcr-wizard-event`;
      try {
        if (opts?.beacon && typeof navigator !== 'undefined' && navigator.sendBeacon) {
          const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
          navigator.sendBeacon(url, blob);
          return;
        }
        // Fire-and-forget; do not await so the UI stays responsive.
        fetch(url, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          keepalive: true,
        }).catch(() => {});
      } catch {
        // Logging must never throw into the UI.
      }
    },
    [userEmail]
  );

  // Open / close lifecycle: generate a session_id on open, log close/abandon
  // when the modal is dismissed.
  useEffect(() => {
    if (showModal && userEmail) {
      const sid =
        typeof crypto !== 'undefined' && 'randomUUID' in crypto
          ? crypto.randomUUID()
          : `sid-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      sessionIdRef.current = sid;
      stepEnteredAtRef.current = Date.now();
      currentStepRef.current = wizardStep;
      wizardCompletedRef.current = false;
      logWizardEvent('wizard_opened', {
        step: wizardStep,
        step_name: wizardSteps[wizardStep]?.id,
        details: {
          cohorts: Object.keys(dataCleanRoom?.cohorts || {}),
        },
      });
    }
    // Intentionally excluding wizardStep/wizardSteps/dataCleanRoom from deps;
    // we only want to fire "opened" when the modal transitions open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showModal, userEmail]);

  // When wizardStep changes while the modal is open, emit a step_left event
  // for the previous step (with its duration) and a step_advanced for the
  // new step.
  useEffect(() => {
    if (!showModal || !userEmail || !sessionIdRef.current) return;
    const prevStep = currentStepRef.current;
    if (prevStep !== wizardStep) {
      const seconds = Math.round((Date.now() - stepEnteredAtRef.current) / 1000);
      logWizardEvent('wizard_step_left', {
        step: prevStep,
        step_name: wizardSteps[prevStep]?.id,
        time_on_step_seconds: seconds,
      });
      logWizardEvent('wizard_step_advanced', {
        step: wizardStep,
        step_name: wizardSteps[wizardStep]?.id,
        details: { from_step: prevStep },
      });
      currentStepRef.current = wizardStep;
      stepEnteredAtRef.current = Date.now();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wizardStep]);

  // Fire abandonment via sendBeacon if the tab is closed while the wizard
  // is open and the user has not completed a publish/download.
  useEffect(() => {
    if (!showModal || !userEmail) return;
    const handler = () => {
      if (!sessionIdRef.current) return;
      const seconds = Math.round((Date.now() - stepEnteredAtRef.current) / 1000);
      logWizardEvent(
        wizardCompletedRef.current ? 'wizard_closed' : 'wizard_abandoned',
        {
          step: currentStepRef.current,
          step_name: wizardSteps[currentStepRef.current]?.id,
          time_on_step_seconds: seconds,
          details: { via: 'unload' },
        },
        { beacon: true }
      );
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showModal, userEmail]);

  // Explicit close handler so both the ✕ and "Close" buttons log consistently.
  const closeWizard = useCallback(() => {
    if (sessionIdRef.current && userEmail) {
      const seconds = Math.round((Date.now() - stepEnteredAtRef.current) / 1000);
      logWizardEvent(
        wizardCompletedRef.current ? 'wizard_closed' : 'wizard_abandoned',
        {
          step: currentStepRef.current,
          step_name: wizardSteps[currentStepRef.current]?.id,
          time_on_step_seconds: seconds,
          details: { via: 'close_button' },
        }
      );
    }
    sessionIdRef.current = null;
    setShowModal(false);
    setDcrRequesterAuthResult(null);
    setDcrBlockchainToken(null);
    setDcrProfileResult(null);
    setDcrAccessRequestResult({});
    setDcrCohortLoading({});
    setDcrRequesterLoading(false);
    setDcrPurpose(0);
    setDcrIntendedUses([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userEmail, logWizardEvent]);

  // --- Default DCR name --------------------------------------------------
  // Format: "<cohort1>-<cohort2>-...-<Month><Day>" e.g. "CohortA-CohortB-April22".
  // Re-computed whenever the set of selected cohorts changes. When the user
  // has not manually edited the name (dcrNameCustomized=false) we push this
  // default into the dcrName field so the wizard always shows a meaningful
  // title; once they click the pencil and type their own, we stop touching it.
  const defaultDcrName = useMemo(() => {
    const cohortIds = Object.keys(dataCleanRoom?.cohorts || {});
    const now = new Date();
    const month = now.toLocaleString('en-US', { month: 'long' });
    const day = now.getDate();
    const dateSuffix = `${month}${day}`;
    return cohortIds.length > 0 ? `${cohortIds.join('-')}-${dateSuffix}` : dateSuffix;
  }, [dataCleanRoom?.cohorts]);

  useEffect(() => {
    if (!dcrNameCustomized) {
      setDcrName(defaultDcrName);
    }
  }, [defaultDcrName, dcrNameCustomized]);

  useEffect(() => {
    if (wizardStep === 2 && dcrIcd10Entries.length === 0 && !dcrIcd10Loading && !dcrIcd10LoadFailed) {
      setDcrIcd10Loading(true);
      fetch('/icd10.json')
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then((data: {code: string; label: string; kind: string; parent?: string | null}[]) => {
          setDcrIcd10Entries(data);
          setDcrIcd10Loading(false);
        })
        .catch(() => {
          setDcrIcd10Loading(false);
          setDcrIcd10LoadFailed(true);
        });
    }
  }, [wizardStep, dcrIcd10Entries.length, dcrIcd10Loading, dcrIcd10LoadFailed]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dcrIcd10WrapperRef.current && !dcrIcd10WrapperRef.current.contains(e.target as Node)) {
        setDcrIcd10SuggestionsOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  
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
          dcr_name: dcrName,
          session_id: sessionIdRef.current
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
        wizardCompletedRef.current = true;
        logWizardEvent('dcr_download_config_clicked', {
          step: currentStepRef.current,
          step_name: wizardSteps[currentStepRef.current]?.id,
          details: { variant: 'zip_with_samples' },
        });
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
        wizardCompletedRef.current = true;
        logWizardEvent('dcr_download_config_clicked', {
          step: currentStepRef.current,
          step_name: wizardSteps[currentStepRef.current]?.id,
          details: { variant: 'json_only' },
        });
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
          research_question: researchQuestion,
          disease_codes: dcrDiseaseCodes,
          org_name: dcrOrgName || null,
          org_type: dcrOrgType || null,
          org_country: dcrOrgCountry || null,
          intended_uses: dcrIntendedUses,
          session_id: sessionIdRef.current,
          selected_mapping_files: availableMappingFiles
            .filter(m => selectedMappingFiles[m.filename] !== false)
            .map(m => ({ filename: m.filename, filepath: m.filepath, display_name: m.display_name, cohorts: m.cohorts })),
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
        wizardCompletedRef.current = true;
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
    setDcrNameCustomized(false);
    setIsEditingDcrName(false);
    setManuallyIncludedOwners([]);
    setAvailableMappingFiles([]);
    setSelectedMappingFiles({});
    setIncludeMappingUploadSlot(false);
    setWizardStep(0);
    // Reset requester auth state
    setDcrRequesterAuthResult(null);
    setDcrBlockchainToken(null);
    setDcrProfileResult(null);
    setDcrAccessRequestResult({});
    setDcrCohortLoading({});
    setDcrRequesterLoading(false);
    setDcrIrbApprovalId('');
    setDcrRequesterType('');
    setDcrPublicProfile(false);
    setDcrGaslessOptIn(false);
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
  }, [showParticipantsModal, cohortIdsKey, additionalAnalysts]);

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

  // Derive the list of excluded data owners. By default ALL data owners are
  // excluded; the user can opt them back in via the participants modal which
  // adds them to manuallyIncludedOwners.
  const excludedDataOwners = useMemo(
    () => dataOwners.map(o => o.email).filter(e => !manuallyIncludedOwners.includes(e)),
    [dataOwners, manuallyIncludedOwners]
  );

  return (
    <div className="navbar bg-base-300 min-h-16 py-2 px-0">
      <div className="navbar-start">
        <ul className="menu menu-horizontal gap-3 my-0 py-0 pl-6 lg:flex flex-wrap">
          <li>
            <Link href="/upload" className={pathname === '/upload' ? 'active' : ''}>
              <Upload size={24} />
              <span className="text-base">Upload</span>
            </Link>
          </li>
          <li>
            <Link href="/cohorts" className={pathname === '/cohorts' ? 'active' : ''}>
              <Compass size={24} />
              <span className="text-base">Explore</span>
            </Link>
          </li>
          <li>
            <Link href="/mapping" className={pathname === '/mapping' ? 'active' : ''}>
              <MapIcon size={24} />
              <span className="text-base">Mapping</span>
            </Link>
          </li>
          <li>
            <Link href="/dcrs" className={pathname === '/dcrs' ? 'active' : ''}>
              <Box size={24} />
              <span className="text-base">My DCRs</span>
            </Link>
          </li>
          <li>
            <Link href="/docs_store" className={pathname === '/docs_store' ? 'active' : ''}>
              <FileText size={24} />
              <span className="text-base">Documents</span>
            </Link>
          </li>
          {isAdmin && (
            <li>
              <Link href="/admin-settings" className={pathname === '/admin-settings' ? 'active' : ''}>
                <Settings size={24} />
                <span className="text-base">Admin</span>
              </Link>
            </li>
          )}
        </ul>
        <div className="dropdown lg:hidden">
          <div tabIndex={0} role="button" className="btn btn-ghost lg:hidden">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h8m-8 6h16" /></svg>
          </div>
          <ul tabIndex={0} className="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow bg-base-100 rounded-box w-52">
            <li><Link href="/upload">Upload</Link></li>
            <li><Link href="/cohorts">Explore</Link></li>
            <li><Link href="/mapping">Mapping</Link></li>
            <li><Link href="/dcrs">My DCRs</Link></li>
            <li><Link href="/docs_store">Documents</Link></li>
            {isAdmin && <li><Link href="/admin-settings">Admin</Link></li>}
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
          {(pathname === '/' || pathname === '/cohorts' || pathname === '/mapping' || pathname === '/dcrs') && (
            <button id="dcr-button" onClick={() => { setShowModal(true); setWizardStep(0); }} className="btn bg-white border-2 border-gray-300 shadow-md hover:shadow-lg hover:bg-gray-50 transition-all duration-200 py-3 px-6" style={{ minWidth: '280px' }}>
              Create a Data Clean Room <div className="badge badge-neutral badge-sm">{Object.keys(dataCleanRoom?.cohorts).length || 0}</div>
            </button>
          )}

          {userEmail ? (
            <button onClick={handleLogout} className="flex items-center space-x-2 p-2 rounded-lg hover:bg-neutral-300">
              <LogOut size={24} />
              <span className="text-base">Logout</span>
            </button>
          ) : (
            <a href={`${apiUrl}/login`} className="flex items-center space-x-2 p-2 rounded-lg hover:bg-neutral-300">
              <LogIn size={24} />
              <span className="text-base">Login</span>
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
            {!userEmail ? (
              /* ========== NOT LOGGED IN ========== */
              <>
                <div className="flex justify-end mb-2">
                  <button className="btn btn-sm btn-ghost" onClick={closeWizard}>✕</button>
                </div>
                <div className="min-h-[200px] flex items-center justify-center">
                  <p className="text-red-500 text-center">Authenticate to access the explorer</p>
                </div>
              </>
            ) : (
              <>
              {/* ========== WIZARD VIEW ========== */}
              <>
                {/* Step indicator */}
                <ul className="steps steps-horizontal w-full mb-6 text-xs">
                  {wizardSteps.map((step, idx) => (
                    <li
                      key={step.id}
                      className={`step ${idx <= wizardStep ? 'step-primary' : ''} ${idx < wizardStep ? 'cursor-pointer' : 'cursor-default'}`}
                      onClick={() => { if (idx < wizardStep) setWizardStep(idx); }}
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
                        {!isEditingDcrName ? (
                          <div className="flex items-center gap-2">
                            <div className="flex-1 input input-bordered flex items-center bg-base-200 cursor-default">
                              <span className="truncate">{dcrName || defaultDcrName}</span>
                            </div>
                            <button
                              type="button"
                              className="btn btn-outline btn-sm"
                              title="Edit DCR name"
                              onClick={() => setIsEditingDcrName(true)}
                            >
                              ✏️ Edit
                            </button>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2">
                            <input
                              type="text"
                              placeholder={defaultDcrName}
                              className="input input-bordered w-full"
                              value={dcrName}
                              autoFocus
                              onChange={(e) => {
                                setDcrName(e.target.value);
                                setDcrNameCustomized(true);
                              }}
                              onBlur={() => setIsEditingDcrName(false)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' || e.key === 'Escape') {
                                  setIsEditingDcrName(false);
                                }
                              }}
                            />
                            {dcrNameCustomized && (
                              <button
                                type="button"
                                className="btn btn-ghost btn-sm"
                                title="Reset to default"
                                onClick={() => {
                                  setDcrNameCustomized(false);
                                  setDcrName(defaultDcrName);
                                  setIsEditingDcrName(false);
                                }}
                              >
                                Reset
                              </button>
                            )}
                          </div>
                        )}
                        <span className="text-xs text-base-content/60 mt-1">
                          A default name is generated from your selected cohorts and today&apos;s date. Click Edit to customize. Your email ({userEmail || 'not logged in'}) will be appended to the title for clarity.
                        </span>
                      </div>
                      <div className="mt-4">
                        <label className="label">
                          <span className="label-text font-semibold">Selected Cohorts</span>
                        </label>
                        <div className="bg-base-200 p-3 rounded-lg cursor-pointer hover:bg-base-300 transition-colors" onClick={() => setShowAddCohortModal(true)}>
                          {Object.keys(dataCleanRoom?.cohorts || {}).length === 0 ? (
                            <span className="text-base-content/60">No cohorts selected — click to add</span>
                          ) : (
                            Object.entries(dataCleanRoom?.cohorts || {}).map(([cohortId, variables]: any) => (
                              <div key={cohortId} className="badge badge-outline mr-2 mb-1">
                                {cohortId} ({variables.length} vars)
                              </div>
                            ))
                          )}
                        </div>
                        <div className="flex gap-2 mt-3">
                          <button onClick={() => setShowAddCohortModal(true)} className="btn btn-outline btn-sm">
                            Add/Remove Cohorts
                          </button>
                          {Object.keys(dataCleanRoom?.cohorts || {}).length > 0 && (
                            <button className="btn btn-outline btn-sm btn-error" onClick={clearCohortsList}>
                              Clear Cohorts
                            </button>
                          )}
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
                  
                  {/* Step 2: Research Goals */}
                  {wizardStep === 2 && (
                    <>
                      <h3 className="font-bold text-lg mb-4">Step 3: Information about Data Use &amp; Research Goals</h3>

                      {/* ICD-10 disease code selector */}
                      <div className="form-control mb-8" ref={dcrIcd10WrapperRef}>
                        <label className="label">
                          <span className="label-text font-semibold flex items-center gap-2">
                            <span>Target Disease(s) — ICD-10</span>
                            {dcrIcd10Loading && <span className="loading loading-spinner loading-xs opacity-60"></span>}
                            {!dcrIcd10Loading && dcrIcd10Entries.length > 0 && (
                              <button type="button" className="btn btn-ghost btn-xs text-primary/70 hover:text-primary normal-case font-normal" onClick={() => { setDcrIcd10BrowseOpen(true); setDcrBrowseSearch(''); setDcrBrowseSelected(null); }}>
                                Browse hierarchy
                              </button>
                            )}
                          </span>
                        </label>

                        {dcrDiseaseCodes.length > 0 && (
                          <div className="flex flex-wrap gap-2 mb-2">
                            {dcrDiseaseCodes.map(entry => (
                              <span key={entry.code} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-base border border-primary/40 bg-primary/10 text-primary font-medium">
                                <span className="font-mono font-bold">{entry.code}</span>
                                <span className="text-primary/70">—</span>
                                <span className="font-normal">{entry.label}</span>
                                {entry.kind !== 'category' && (
                                  <span className="text-xs opacity-60 italic">({entry.kind})</span>
                                )}
                                <button type="button" className="ml-1 text-primary/60 hover:text-error transition-colors leading-none" onClick={() => setDcrDiseaseCodes(prev => prev.filter(e => e.code !== entry.code))} aria-label={`Remove ${entry.code}`}>×</button>
                              </span>
                            ))}
                          </div>
                        )}

                        {(dcrDiseaseCodes.length === 0 || dcrIcd10SuggestionsOpen || dcrIcd10SearchQuery) && (
                          <div className="relative">
                            <input type="text" className="input input-bordered w-full"
                              placeholder={dcrDiseaseCodes.length === 0 ? 'e.g. I50 or "heart failure" — type to search' : 'Search to add another…'}
                              value={dcrIcd10SearchQuery} autoComplete="off"
                              onChange={e => { setDcrIcd10SearchQuery(e.target.value); setDcrIcd10SuggestionsOpen(e.target.value.trim().length > 0); }}
                              onFocus={() => { if (dcrIcd10SearchQuery.length > 0) setDcrIcd10SuggestionsOpen(true); }}
                            />
                            {dcrIcd10SuggestionsOpen && dcrIcd10Entries.length > 0 && (() => {
                              const q = dcrIcd10SearchQuery.trim().toLowerCase();
                              const kindOrder: Record<string, number> = {chapter: 0, block: 1, category: 2};
                              const selectedCodes = new Set(dcrDiseaseCodes.map(e => e.code));
                              const entryMap = new Map(dcrIcd10Entries.map(e => [e.code, e]));
                              const directMatches = dcrIcd10Entries
                                .filter(e => !selectedCodes.has(e.code) && (e.code.toLowerCase().startsWith(q) || e.label.toLowerCase().includes(q)))
                                .sort((a, b) => { const kd = (kindOrder[a.kind]??2)-(kindOrder[b.kind]??2); return kd !== 0 ? kd : a.code.length - b.code.length || a.code.localeCompare(b.code); })
                                .slice(0, 20);
                              const directCodes = new Set(directMatches.map(e => e.code));
                              const ancestorCodes = new Set<string>();
                              directMatches.forEach(entry => { let p = entry.parent; while (p) { if (!directCodes.has(p) && !selectedCodes.has(p)) ancestorCodes.add(p); p = entryMap.get(p)?.parent ?? undefined; } });
                              type IcdItem = typeof dcrIcd10Entries[0] & {isContext: boolean};
                              const allItems: IcdItem[] = [
                                ...directMatches.map(e => ({...e, isContext: false})),
                                ...Array.from(ancestorCodes).map(code => entryMap.get(code)).filter((e): e is typeof dcrIcd10Entries[0] => !!e).map(e => ({...e, isContext: true})),
                              ].sort((a, b) => { const kd = (kindOrder[a.kind]??2)-(kindOrder[b.kind]??2); return kd !== 0 ? kd : a.code.length - b.code.length || a.code.localeCompare(b.code); });
                              const getIndent = (e: IcdItem) => e.kind === 'chapter' ? 'pl-3' : e.kind === 'block' ? 'pl-6' : e.code.includes('.') ? 'pl-12' : 'pl-9';
                              return allItems.length > 0 ? (
                                <ul className="absolute z-50 w-full bg-base-100 border border-base-300 rounded-lg shadow-lg mt-1 max-h-60 overflow-y-auto">
                                  {allItems.map(e => (
                                    <li key={e.code}>
                                      <button type="button" className={`w-full text-left pr-3 py-1.5 hover:bg-base-200 text-sm flex items-center gap-2 ${getIndent(e)} ${e.isContext ? 'opacity-50' : ''}`}
                                        onMouseDown={ev => { ev.preventDefault(); setDcrDiseaseCodes(prev => [...prev, {code: e.code, label: e.label, kind: e.kind}]); setDcrIcd10SearchQuery(''); setDcrIcd10SuggestionsOpen(false); }}>
                                        <span className="font-mono font-semibold text-primary w-16 shrink-0">{e.code}</span>
                                        <span className="text-base-content/80 flex-1">{e.label}</span>
                                        {e.kind !== 'category' && <span className={`badge badge-xs ml-1 shrink-0 ${e.kind === 'chapter' ? 'badge-accent' : 'badge-ghost'}`}>{e.kind}</span>}
                                      </button>
                                    </li>
                                  ))}
                                </ul>
                              ) : null;
                            })()}
                          </div>
                        )}

                        {dcrDiseaseCodes.length > 0 && !dcrIcd10SuggestionsOpen && !dcrIcd10SearchQuery && (
                          <button type="button" className="btn btn-ghost btn-xs text-primary mt-1 self-start"
                            onClick={() => { setDcrIcd10SearchQuery(' '); setDcrIcd10SuggestionsOpen(true); setTimeout(() => setDcrIcd10SearchQuery(''), 0); }}>
                            + Specify another disease
                          </button>
                        )}

                        {dcrIcd10LoadFailed && (
                          <label className="label"><span className="label-text-alt text-warning">Could not load ICD-10 data. Restart the dev server or rebuild the Docker container.</span></label>
                        )}
                        {dcrDiseaseCodes.length === 0 && !dcrIcd10SearchQuery && (
                          <label className="label">
                            <span className="label-text-alt text-base-content/60">ICD-10 (2019). Specify target disease(s) for this DCR.</span>
                          </label>
                        )}

                        {dcrIcd10BrowseOpen && dcrIcd10Entries.length > 0 && (() => {
                          const selCodes = new Set(dcrDiseaseCodes.map(e => e.code));
                          const childrenMap = new Map<string, typeof dcrIcd10Entries>();
                          dcrIcd10Entries.forEach(e => { if (e.parent) { if (!childrenMap.has(e.parent)) childrenMap.set(e.parent, []); childrenMap.get(e.parent)!.push(e); } });
                          const chapters = dcrIcd10Entries.filter(e => e.kind === 'chapter');
                          const bq = dcrBrowseSearch.trim().toLowerCase();
                          const kindOrder: Record<string, number> = {chapter: 0, block: 1, category: 2};
                          const selectEntry = (entry: typeof dcrIcd10Entries[0]) => setDcrBrowseSelected({code: entry.code, label: entry.label, kind: entry.kind});
                          const renderEntry = (entry: typeof dcrIcd10Entries[0], depth: number): React.ReactNode => {
                            const children = childrenMap.get(entry.code) || [];
                            const isExpanded = dcrBrowseExpanded.has(entry.code);
                            const isPicked = dcrBrowseSelected?.code === entry.code;
                            const alreadyAdded = selCodes.has(entry.code);
                            return (
                              <React.Fragment key={entry.code}>
                                <div className={`flex items-center gap-2 py-1.5 cursor-pointer hover:bg-base-200 select-none ${isPicked ? 'bg-primary/20 ring-1 ring-inset ring-primary/40' : alreadyAdded ? 'opacity-50' : ''}`}
                                  style={{paddingLeft: `${12 + depth * 18}px`, paddingRight: '12px'}}
                                  onClick={() => { selectEntry(entry); if (children.length) setDcrBrowseExpanded(prev => { const n = new Set(prev); n.has(entry.code) ? n.delete(entry.code) : n.add(entry.code); return n; }); }}>
                                  <span className="w-4 shrink-0 text-center text-xs text-base-content/40 hover:text-base-content"
                                    onClick={children.length ? (ev) => { ev.stopPropagation(); setDcrBrowseExpanded(prev => { const n = new Set(prev); n.has(entry.code) ? n.delete(entry.code) : n.add(entry.code); return n; }); } : undefined}>
                                    {children.length ? (isExpanded ? '▾' : '▸') : ''}
                                  </span>
                                  <span className="font-mono font-semibold text-primary text-sm w-16 shrink-0">{entry.code}</span>
                                  <span className="text-sm flex-1 text-base-content/80">{entry.label}</span>
                                  {entry.kind !== 'category' && <span className={`badge badge-xs shrink-0 ${entry.kind === 'chapter' ? 'badge-accent' : 'badge-ghost'}`}>{entry.kind}</span>}
                                  {alreadyAdded && <Check className="w-4 h-4 text-primary/50 shrink-0" />}
                                </div>
                                {isExpanded && children.map(child => renderEntry(child, depth + 1))}
                              </React.Fragment>
                            );
                          };
                          const filteredItems = bq ? (() => {
                            const entryMap = new Map(dcrIcd10Entries.map(e => [e.code, e]));
                            const direct = dcrIcd10Entries.filter(e => e.code.toLowerCase().startsWith(bq) || e.label.toLowerCase().includes(bq))
                              .sort((a, b) => { const kd = (kindOrder[a.kind]??2)-(kindOrder[b.kind]??2); return kd !== 0 ? kd : a.code.length - b.code.length || a.code.localeCompare(b.code); }).slice(0, 60);
                            const directCodes = new Set(direct.map(e => e.code));
                            const anc = new Set<string>();
                            direct.forEach(e => { let p = e.parent; while (p) { if (!directCodes.has(p)) anc.add(p); p = entryMap.get(p)?.parent ?? undefined; } });
                            type FItem = typeof dcrIcd10Entries[0] & {isCtx: boolean};
                            return [...direct.map(e => ({...e, isCtx: false})), ...Array.from(anc).map(c => entryMap.get(c)).filter((e): e is typeof dcrIcd10Entries[0] => !!e).map(e => ({...e, isCtx: true}))]
                              .sort((a, b) => { const kd = (kindOrder[a.kind]??2)-(kindOrder[b.kind]??2); return kd !== 0 ? kd : a.code.length - b.code.length || a.code.localeCompare(b.code); }) as FItem[];
                          })() : null;
                          const filteredIndent = (e: {kind: string; code: string}) => e.kind === 'chapter' ? 12 : e.kind === 'block' ? 30 : e.code.includes('.') ? 66 : 48;
                          return (
                            <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-sm" onMouseDown={() => setDcrIcd10BrowseOpen(false)}>
                              <div className="bg-base-100 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[82vh] flex flex-col mx-4" onMouseDown={e => e.stopPropagation()}>
                                <div className="flex items-center justify-between px-5 py-4 border-b border-base-300 shrink-0">
                                  <h3 className="font-bold text-base">Browse ICD-10 Hierarchy</h3>
                                  <button type="button" className="btn btn-sm btn-ghost btn-circle" onClick={() => setDcrIcd10BrowseOpen(false)}>✕</button>
                                </div>
                                <div className="px-4 py-3 border-b border-base-300 shrink-0">
                                  <input type="text" className="input input-bordered input-sm w-full" placeholder="Filter by code or name…" value={dcrBrowseSearch} onChange={e => setDcrBrowseSearch(e.target.value)} autoFocus />
                                </div>
                                <div className="overflow-y-auto flex-1">
                                  {filteredItems ? filteredItems.map(e => (
                                    <div key={e.code}
                                      className={`flex items-center gap-2 py-1.5 cursor-pointer hover:bg-base-200 select-none ${e.isCtx ? 'opacity-50' : ''} ${dcrBrowseSelected?.code === e.code ? 'bg-primary/20 ring-1 ring-inset ring-primary/40' : selCodes.has(e.code) ? 'opacity-50' : ''}`}
                                      style={{paddingLeft: `${filteredIndent(e)}px`, paddingRight: '12px'}}
                                      onClick={() => !e.isCtx && selectEntry(e)}>
                                      <span className="font-mono font-semibold text-primary text-sm w-16 shrink-0">{e.code}</span>
                                      <span className="text-sm flex-1 text-base-content/80">{e.label}</span>
                                      {e.kind !== 'category' && <span className={`badge badge-xs shrink-0 ${e.kind === 'chapter' ? 'badge-accent' : 'badge-ghost'}`}>{e.kind}</span>}
                                      {selCodes.has(e.code) && <Check className="w-4 h-4 text-primary/50 shrink-0" />}
                                    </div>
                                  )) : chapters.map(chapter => renderEntry(chapter, 0))}
                                </div>
                                <div className="px-5 py-3 border-t border-base-300 flex justify-between items-center shrink-0">
                                  <span className="text-sm text-base-content/60">{dcrDiseaseCodes.length === 0 ? 'No diseases selected yet' : `${dcrDiseaseCodes.length} selected`}</span>
                                  <div className="flex gap-2">
                                    <button type="button" className="btn btn-sm btn-ghost" onClick={() => setDcrIcd10BrowseOpen(false)}>Cancel</button>
                                    <button type="button" className="btn btn-sm btn-primary" disabled={!dcrBrowseSelected}
                                      onClick={() => { if (dcrBrowseSelected && !selCodes.has(dcrBrowseSelected.code)) setDcrDiseaseCodes(prev => [...prev, dcrBrowseSelected]); setDcrIcd10BrowseOpen(false); }}>
                                      {dcrBrowseSelected ? `Select ${dcrBrowseSelected.code}` : 'Select'}
                                    </button>
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })()}
                      </div>

                      <div className="form-control mb-6">
                        <label className="label">
                          <span className="label-text font-semibold">Your organization</span>
                        </label>
                        <div className="relative">
                          <button type="button" className="btn btn-outline w-full justify-between font-normal text-left"
                            onClick={() => setDcrOrgDropdownOpen(o => !o)}>
                            <span className={dcrOrgName ? 'text-base-content' : 'text-base-content/60'}>{dcrOrgName || 'Select your institution…'}</span>
                            <span className="text-xs opacity-60">▾</span>
                          </button>
                          {dcrOrgDropdownOpen && (() => {
                            const institutionNames = [...new Set(Object.values(cohortsData as Record<string, {institution?: string}>).map(c => c.institution).filter((x): x is string => Boolean(x)))].sort();
                            return (
                              <div className="absolute z-50 left-0 w-full bg-base-100 border border-base-300 rounded-lg shadow-xl mt-1">
                                <div className="max-h-[40vh] overflow-y-auto p-2" style={{columnCount: 2, columnGap: '0.5rem'}}>
                                  {institutionNames.map(name => (
                                    <label key={name} className="flex items-center gap-1.5 py-1 cursor-pointer hover:bg-base-200 rounded px-1" style={{breakInside: 'avoid' as const}}>
                                      <input type="radio" name="dcrOrgName" className="radio radio-xs radio-primary shrink-0"
                                        checked={dcrOrgName === name}
                                        onChange={() => { setDcrOrgName(name); setDcrOrgDropdownOpen(false); }} />
                                      <span className="text-xs text-base-content/80">{name}</span>
                                    </label>
                                  ))}
                                </div>
                                {dcrOrgName && (
                                  <div className="px-3 py-2 border-t border-base-300 flex justify-between items-center">
                                    <span className="text-xs text-primary font-medium truncate">{dcrOrgName}</span>
                                    <button type="button" className="btn btn-xs btn-ghost text-error ml-2 shrink-0" onClick={() => { setDcrOrgName(''); setDcrOrgDropdownOpen(false); }}>Clear</button>
                                  </div>
                                )}
                              </div>
                            );
                          })()}
                        </div>
                      </div>

                      <div className="form-control mb-6">
                        <label className="label">
                          <span className="label-text font-semibold">Organization type</span>
                        </label>
                        <div className="flex gap-6">
                          <label className="flex items-center gap-2 cursor-pointer">
                            <input type="radio" className="radio radio-primary radio-sm" name="dcrOrgType" value="non-profit" checked={dcrOrgType === 'non-profit'} onChange={() => { setDcrOrgType('non-profit'); setDcrRequesterType('NONPROFIT'); }} />
                            <span className="text-sm">Non-profit</span>
                          </label>
                          <label className="flex items-center gap-2 cursor-pointer">
                            <input type="radio" className="radio radio-primary radio-sm" name="dcrOrgType" value="for-profit" checked={dcrOrgType === 'for-profit'} onChange={() => { setDcrOrgType('for-profit'); setDcrRequesterType('PROFIT'); }} />
                            <span className="text-sm">For-profit</span>
                          </label>
                        </div>
                      </div>

                      <div className="form-control mb-6">
                        <label className="label">
                          <span className="label-text font-semibold">Organization country</span>
                        </label>
                        <input
                          type="text"
                          className="input input-bordered w-28"
                          maxLength={2}
                          placeholder="e.g. NL"
                          value={dcrOrgCountry}
                          onChange={e => setDcrOrgCountry(e.target.value.toUpperCase().replace(/[^A-Z]/g, ''))}
                        />
                        <label className="label"><span className="label-text-alt">Two-letter ISO 3166-1 alpha-2 country code — <a href="https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2#Officially_assigned_code_elements" target="_blank" rel="noopener noreferrer" className="link link-primary">see full list on Wikipedia</a></span></label>
                      </div>

                      <div className="form-control mb-6">
                        <label className="label">
                          <span className="label-text font-semibold">Intended data use - check all that apply</span>
                        </label>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 p-3 border border-base-300 rounded-lg">
                          {([
                            { id: 'genetic',    purpose: 6, duo: 'GRU', label: 'My research involves genetic or genomic studies' },
                            { id: 'ancestry',   purpose: 4, duo: 'POA', label: 'My research involves population origins or ancestry analysis' },
                            { id: 'commercial', purpose: 1, duo: 'GRU', label: 'This work is not part of any commercial product or service' },
                            { id: 'irb',        purpose: 1, duo: 'GRU', label: 'Ethics approval has been obtained for this project' },
                            { id: 'publish',    purpose: 1, duo: 'GRU', label: 'I intend to publish findings related to this analysis' },
                          ] as const).map(opt => (
                            <label key={opt.id} className="flex items-start gap-2 cursor-pointer p-1 hover:bg-base-200 rounded">
                              <input
                                type="checkbox"
                                className="checkbox checkbox-sm checkbox-primary mt-0.5"
                                checked={dcrIntendedUses.includes(opt.id)}
                                onChange={e => {
                                  if (e.target.checked) {
                                    setDcrIntendedUses(prev => [...prev, opt.id]);
                                    if (dcrPurpose === 0) { setDcrPurpose(opt.purpose); }
                                  } else {
                                    setDcrIntendedUses(prev => prev.filter(u => u !== opt.id));
                                  }
                                }}
                              />
                              <span className="text-xs leading-relaxed">{opt.label}</span>
                            </label>
                          ))}
                        </div>
                      </div>

                      <div className="form-control mb-6">
                        <label className="label">
                          <span className="label-text font-semibold">Please describe your research question</span>
                        </label>
                        <textarea
                          placeholder="Describe the research question you aim to answer with this DCR..."
                          className="textarea textarea-bordered w-full h-20"
                          value={researchQuestion}
                          onChange={(e) => setResearchQuestion(e.target.value.slice(0, 1000))}
                          maxLength={1000}
                        />
                        <span className="text-xs text-base-content/60 mt-1 self-end">
                          {researchQuestion.length}/1000
                        </span>
                      </div>

                      {/* Requester Blockchain Auth Section */}
                      <div className="divider mt-8 mb-4">Blockchain Authentication</div>
                      <div className="bg-base-200/50 rounded-lg p-4 space-y-4">
                        {!dcrRequesterAuthResult && (
                          <div>
                            <button
                              type="button"
                              className="btn btn-accent w-full"
                              disabled={dcrRequesterLoading}
                              onClick={async () => {
                                setDcrRequesterLoading(true);
                                try {
                                  const resp = await fetch(`${apiUrl}/blockchain/verify`, {
                                    method: 'POST',
                                    credentials: 'include',
                                  });
                                  const data = await resp.json();
                                  if (!resp.ok) {
                                    throw new Error(data.detail || JSON.stringify(data));
                                  }
                                  setDcrRequesterAuthResult(data);
                                  setDcrBlockchainToken(data.verify?.token || null);
                                } catch (err: any) {
                                  console.error('Requester auth failed:', err);
                                  alert(`Authentication failed: ${err.message}`);
                                } finally {
                                  setDcrRequesterLoading(false);
                                }
                              }}
                            >
                              {dcrRequesterLoading ? <span className="loading loading-spinner loading-xs"></span> : 'Authenticate with Blockchain API'}
                            </button>
                            <p className="text-xs text-base-content/60 mt-2">Authenticate as a requester to register your profile and submit access requests on the blockchain.</p>
                          </div>
                        )}

                        {dcrRequesterAuthResult && (
                          <div className="alert alert-success">
                            <Check className="w-5 h-5" />
                            <div className="text-sm">
                              <p className="font-semibold">Requester authentication successful</p>
                              <p>Address: <code className="text-xs">{dcrRequesterAuthResult.verify?.address}</code></p>
                              <p>Email hash: <code className="text-xs">{dcrRequesterAuthResult.verify?.emailHash}</code></p>
                            </div>
                          </div>
                        )}

                        {/* Requester Profile Form */}
                        {dcrRequesterAuthResult && !dcrProfileResult && (
                          <div className="space-y-3">
                            <div className="form-control">
                              <label className="label"><span className="label-text font-semibold">IRB Approval ID (optional)</span></label>
                              <input
                                type="text"
                                className="input input-bordered input-sm w-full"
                                placeholder="e.g. IRB-2024-12345"
                                value={dcrIrbApprovalId}
                                onChange={e => setDcrIrbApprovalId(e.target.value)}
                              />
                            </div>

                            <div className="flex gap-4">
                              <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                  type="checkbox"
                                  className="checkbox checkbox-sm checkbox-primary"
                                  checked={dcrPublicProfile}
                                  onChange={e => setDcrPublicProfile(e.target.checked)}
                                />
                                <span className="text-sm">Public profile</span>
                              </label>
                              <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                  type="checkbox"
                                  className="checkbox checkbox-sm checkbox-primary"
                                  checked={dcrGaslessOptIn}
                                  onChange={e => setDcrGaslessOptIn(e.target.checked)}
                                />
                                <span className="text-sm">Gasless transactions</span>
                              </label>
                            </div>

                            <button
                              type="button"
                              className="btn btn-primary btn-sm w-full"
                              disabled={dcrRequesterLoading || !dcrRequesterType}
                              onClick={async () => {
                                setDcrRequesterLoading(true);
                                try {
                                  const profileBody = {
                                    email: userEmail,
                                    institutionId: dcrOrgName,
                                    requesterType: dcrRequesterType,
                                    irbApprovalId: dcrIrbApprovalId || undefined,
                                    countryCode: dcrOrgCountry || undefined,
                                    publicProfile: dcrPublicProfile,
                                    gaslessOptIn: dcrGaslessOptIn,
                                  };
                                  const resp = await fetch('http://localhost:8020/api/requesters/profile', {
                                    method: 'PUT',
                                    headers: {
                                      'Content-Type': 'application/json',
                                      'Authorization': `Bearer ${dcrBlockchainToken}`,
                                    },
                                    body: JSON.stringify(profileBody),
                                    credentials: 'include',
                                  });
                                  const data = await resp.json();
                                  if (!resp.ok) {
                                    throw new Error(data.detail || JSON.stringify(data));
                                  }
                                  setDcrProfileResult(data);
                                } catch (err: any) {
                                  console.error('Profile creation failed:', err);
                                  alert(`Profile creation failed: ${err.message}`);
                                } finally {
                                  setDcrRequesterLoading(false);
                                }
                              }}
                            >
                              {dcrRequesterLoading ? <span className="loading loading-spinner loading-xs"></span> : 'Create Requester Profile'}
                            </button>
                          </div>
                        )}

                        {/* Profile Result */}
                        {dcrProfileResult && (
                          <div className="space-y-3">
                            <div className="alert alert-success">
                              <Check className="w-5 h-5" />
                              <div className="text-sm">
                                <p className="font-semibold">Requester profile created</p>
                                <p>Address: <code className="text-xs">{dcrProfileResult.address}</code></p>
                                <p>Email hash: <code className="text-xs">{dcrProfileResult.emailHash}</code></p>
                                <p>Institution: <code className="text-xs">{dcrProfileResult.profile.institutionId}</code></p>
                                <p>Type: <code className="text-xs">{dcrProfileResult.profile.requesterType}</code></p>
                              </div>
                            </div>

                            {/* Access Request Buttons — one per cohort */}
                            <div className="space-y-3">
                              <p className="text-sm font-semibold">Submit access request for each cohort:</p>
                              {Object.keys(dataCleanRoom?.cohorts || {}).map(cohortId => {
                                const result = dcrAccessRequestResult[cohortId];
                                const loading = dcrCohortLoading[cohortId];
                                return (
                                  <div key={cohortId} className="border border-base-300 rounded-lg p-3 space-y-2">
                                    <p className="text-sm font-semibold">{cohortId}</p>
                                    {!result && (
                                      <button
                                        type="button"
                                        className="btn btn-accent btn-sm w-full"
                                        disabled={!!loading}
                                        onClick={async () => {
                                          setDcrCohortLoading(prev => ({ ...prev, [cohortId]: true }));
                                          try {
                                            const accessRequestBody = {
                                              email: userEmail,
                                              cohortId,
                                              intendedUse: dcrIntendedUses.includes('ancestry') ? 'POA' : 'GRU',
                                              purpose: dcrPurpose || 1,
                                              diseaseCode: dcrDiseaseCodes.length > 0 ? dcrDiseaseCodes[0].code : undefined,
                                              projectId: `DCR-${Date.now()}`,
                                              abstract: researchQuestion || undefined,
                                            };
                                            const resp = await fetch('http://localhost:8020/api/requesters/access-requests', {
                                              method: 'POST',
                                              headers: {
                                                'Content-Type': 'application/json',
                                                'Authorization': `Bearer ${dcrBlockchainToken}`,
                                              },
                                              body: JSON.stringify(accessRequestBody),
                                              credentials: 'include',
                                            });
                                            const data = await resp.json();
                                            if (!resp.ok) throw new Error(data.detail || JSON.stringify(data));
                                            setDcrAccessRequestResult(prev => ({ ...prev, [cohortId]: data }));
                                          } catch (err: any) {
                                            console.error('Access request failed:', err);
                                            setDcrAccessRequestResult(prev => ({ ...prev, [cohortId]: { auto_approved: false, error: err.message } }));
                                          } finally {
                                            setDcrCohortLoading(prev => ({ ...prev, [cohortId]: false }));
                                          }
                                        }}
                                      >
                                        {loading ? <span className="loading loading-spinner loading-xs"></span> : 'Submit Access Request'}
                                      </button>
                                    )}
                                    {result && (
                                      <div className="text-sm space-y-1">
                                        {result.error ? (
                                          <p className="text-error text-xs">{result.error}</p>
                                        ) : (
                                          <>
                                            <p>Status: <span className={`badge badge-xs ${result.auto_approved ? 'badge-success' : 'badge-warning'}`}>{result.auto_approved ? 'Approved' : (result.status || 'Pending')}</span></p>
                                            {result.request_id && <p>Request ID: <code className="text-xs">{result.request_id}</code></p>}
                                            {result.tx_hash && <p>Tx: <code className="text-xs break-all">{result.tx_hash}</code></p>}
                                          </>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                  
                  {/* Step 3: Data Samples Settings (merged Airlock + Shuffled Samples) */}
                  {wizardStep === 3 && (
                    <>
                      <h3 className="font-bold text-lg mb-4">Step 4: Data Samples</h3>
                      <div className="text-sm text-base-content/70 mb-4 space-y-2">
                        <p><strong>Shuffled Sample:</strong> A shuffled sample contains a maximum of 500 rows from dataset without the patient id column. Each column in the sample is shuffled independently of the other columns. The sample can be downloaded for code testing and validation outside this platform.</p>
                        <p><strong>Airlock:</strong> The airlock allows 20% of the real data to be available to analysts inside the DCR for code testing and validation purposes. Note that the patient id column and the rows with outlier values (|z-score - mean| &gt; 2) are excluded from the airlock sample.</p>
                      </div>
                      
                      {loadingShuffledSamples ? (
                        <p className="text-sm text-base-content/70">Checking data availability...</p>
                      ) : (
                        <div className="space-y-4">
                          {dataCleanRoom?.cohorts && Object.keys(dataCleanRoom.cohorts).sort().map((cohortId) => {
                            const hasShuffledSample = cohortsWithShuffledSamples.includes(cohortId);
                            // Default: shuffled sample if available, otherwise airlock
                            const defaultAirlock = !hasShuffledSample;
                            const defaultShuffled = hasShuffledSample;
                            const currentAirlock = airlockSettings[cohortId] ?? defaultAirlock;
                            const currentShuffled = shuffledSampleSettings[cohortId] ?? defaultShuffled;
                            
                            // Determine current selection
                            let currentSelection = 'none';
                            if (currentAirlock && currentShuffled && hasShuffledSample) currentSelection = 'both';
                            else if (currentAirlock && !currentShuffled) currentSelection = 'airlock';
                            else if (!currentAirlock && currentShuffled && hasShuffledSample) currentSelection = 'shuffled';
                            else if (currentAirlock) currentSelection = 'airlock';
                            
                            return (
                              <div key={cohortId} className="p-3 bg-base-200 rounded-lg">
                                <p className="font-medium mb-2">{cohortId}</p>
                                <div className="flex flex-wrap gap-2">
                                  <button
                                    className={`btn btn-sm ${currentSelection === 'none' ? 'btn-primary' : 'btn-outline'}`}
                                    onClick={() => {
                                      setAirlockSettings({...airlockSettings, [cohortId]: false});
                                      setShuffledSampleSettings({...shuffledSampleSettings, [cohortId]: false});
                                    }}
                                  >
                                    None
                                  </button>
                                  {hasShuffledSample && (
                                    <button
                                      className={`btn btn-sm ${currentSelection === 'shuffled' ? 'btn-primary' : 'btn-outline'}`}
                                      onClick={() => {
                                        setAirlockSettings({...airlockSettings, [cohortId]: false});
                                        setShuffledSampleSettings({...shuffledSampleSettings, [cohortId]: true});
                                      }}
                                    >
                                      Shuffled Sample
                                    </button>
                                  )}
                                  <button
                                    className={`btn btn-sm ${currentSelection === 'airlock' ? 'btn-primary' : 'btn-outline'}`}
                                    onClick={() => {
                                      setAirlockSettings({...airlockSettings, [cohortId]: true});
                                      setShuffledSampleSettings({...shuffledSampleSettings, [cohortId]: false});
                                    }}
                                  >
                                    Airlocked Sample
                                  </button>
                                  {hasShuffledSample && (
                                    <button
                                      className={`btn btn-sm ${currentSelection === 'both' ? 'btn-primary' : 'btn-outline'}`}
                                      onClick={() => {
                                        setAirlockSettings({...airlockSettings, [cohortId]: true});
                                        setShuffledSampleSettings({...shuffledSampleSettings, [cohortId]: true});
                                      }}
                                    >
                                      Both
                                    </button>
                                  )}
                                </div>
                                {!hasShuffledSample && (
                                  <p className="text-xs text-base-content/50 mt-2 italic">No shuffled sample available for this cohort</p>
                                )}
                              </div>
                            );
                          })}
                        </div>
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
                          <strong>DCR Name:</strong> {(dcrName || defaultDcrName)} - created by {userEmail}
                        </div>
                        <div className="p-3 bg-base-200 rounded-lg">
                          <strong>Cohorts:</strong> {Object.keys(dataCleanRoom?.cohorts || {}).join(', ')}
                        </div>
                        <div className="p-3 bg-base-200 rounded-lg">
                          <strong>Additional Analysts:</strong> {additionalAnalysts.length > 0 ? additionalAnalysts.join(', ') : 'None'}
                        </div>
                        {researchQuestion && (
                          <div className="p-3 bg-base-200 rounded-lg">
                            <strong>Research Question:</strong> {researchQuestion}
                          </div>
                        )}
                        {dcrDiseaseCodes.length > 0 && (
                          <div className="p-3 bg-base-200 rounded-lg">
                            <strong>Target Disease(s):</strong> {dcrDiseaseCodes.map(e => `${e.code} — ${e.label}`).join('; ')}
                          </div>
                        )}
                        {(dcrOrgName || dcrOrgType || dcrOrgCountry) && (
                          <div className="p-3 bg-base-200 rounded-lg">
                            <strong>Organization:</strong> {[dcrOrgName, dcrOrgType, dcrOrgCountry].filter(Boolean).join(' · ')}
                          </div>
                        )}
                        {dcrIntendedUses.length > 0 && (
                          <div className="p-3 bg-base-200 rounded-lg">
                            <strong>Intended uses:</strong> {dcrIntendedUses.join(', ')}
                          </div>
                        )}
                        <div className="p-3 bg-base-200 rounded-lg">
                          <strong>Airlock Cohorts:</strong> {Object.entries(airlockSettings).filter(([_, v]) => v !== false).map(([k]) => k).join(', ') || 'None'}
                        </div>
                        <div className="p-3 bg-base-200 rounded-lg">
                          <strong>Shuffled Samples:</strong> {Object.entries(shuffledSampleSettings).filter(([_, v]) => v !== false).map(([k]) => k).join(', ') || 'None'}
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
                          disabled={isLoading || dcrCreated || Object.keys(dataCleanRoom?.cohorts || {}).length === 0}
                        >
                          Create Live DCR
                        </button>
                        <button 
                          className="btn btn-neutral h-auto min-h-0 py-2 px-4" 
                          onClick={getDCRDefinitionFile} 
                          disabled={isLoading || configDownloaded || Object.keys(dataCleanRoom?.cohorts || {}).length === 0}
                        >
                          <span className="flex flex-col items-center">
                            <span>Download DCR Config</span>
                            <span className="text-xs opacity-70 font-normal">+ ancillary files</span>
                          </span>
                        </button>
                        {Object.keys(dataCleanRoom?.cohorts || {}).length > 0 && (
                          <button className="btn btn-outline btn-error" onClick={clearCohortsList}>
                            Clear Cohorts
                          </button>
                        )}
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
                  {wizardStep === 2 && Object.values(dcrAccessRequestResult).some((r: any) => r && !r.auto_approved) && (
                    <div className="alert alert-error text-sm mb-3">
                      <span>The DCR creation process cannot proceed due to incompatibility in data use between provider and requester.</span>
                    </div>
                  )}
                  <div className="flex gap-2">
                    <button className="btn" onClick={closeWizard}>
                      Close
                    </button>
                    {wizardStep < wizardSteps.length - 1 && (
                      <button
                        className="btn btn-primary"
                        disabled={wizardStep === 2 && !(Object.keys(dataCleanRoom?.cohorts || {}).length > 0 && Object.keys(dataCleanRoom?.cohorts || {}).every(id => dcrAccessRequestResult[id]?.auto_approved === true))}
                        title={wizardStep === 2 ? 'All cohorts must have an approved blockchain access grant before proceeding' : undefined}
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
                
                {/* Feedback link at bottom of final step */}
                {wizardStep === 5 && (
                  <div className="mt-6 text-center text-sm text-base-content/60">
                    <a
                      href="https://docs.google.com/forms/d/e/1FAIpQLSd7EmQJgfNJJej8cuKN_eOv5ROYcjVVE-aM_sruNW6P0wySOQ/viewform?hl=en%2Fedit&hl=en"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline hover:text-primary"
                    >
                    Feedback? Here is the form →
                    </a>
                  </div>
                )}
              </>
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
          manuallyIncludedOwners={manuallyIncludedOwners}
          setManuallyIncludedOwners={setManuallyIncludedOwners}
          onClose={() => setShowParticipantsModal(false)}
          isLoading={loadingParticipants}
        />
      )}

      {/* Add Cohort Modal */}
      {showAddCohortModal && (
        <div className="modal modal-open">
          <div className="modal-box max-w-2xl">
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-bold text-lg">Add/Remove Cohorts</h3>
              <button className="btn btn-sm btn-ghost" onClick={() => { setShowAddCohortModal(false); setCohortSearchQuery(''); }}>✕</button>
            </div>
            
            <input 
              type="text"
              placeholder="Search cohorts by name..."
              className="input input-bordered w-full mb-4"
              value={cohortSearchQuery}
              onChange={(e) => setCohortSearchQuery(e.target.value)}
              autoFocus
            />
            
            <p className="text-xs text-base-content/60 mb-3 italic">Only cohorts with uploaded metadata can be added to the DCR.</p>
            
            <div className="max-h-[400px] overflow-y-auto space-y-2">
              {Object.entries(cohortsData || {})
                .filter(([cohortId, cohortInfo]: [string, any]) => {
                  // Only show cohorts with uploaded metadata (i.e., those with variables)
                  const hasMetadata = cohortInfo?.variables && Object.keys(cohortInfo.variables).length > 0;
                  const matchesSearch = cohortSearchQuery === '' || 
                    cohortId.toLowerCase().includes(cohortSearchQuery.toLowerCase());
                  return hasMetadata && matchesSearch;
                })
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([cohortId, cohortInfo]: [string, any]) => {
                  const isSelected = !!dataCleanRoom?.cohorts?.[cohortId];
                  const variableCount = Object.keys(cohortInfo.variables).length;
                  return (
                    <div 
                      key={cohortId}
                      className={`p-3 rounded-lg cursor-pointer flex justify-between items-center ${isSelected ? 'bg-primary/20 border border-primary' : 'bg-base-200 hover:bg-base-300'}`}
                      onClick={() => {
                        if (isSelected) {
                          // Remove cohort - create new cohorts object to trigger re-render
                          const newCohorts = {...dataCleanRoom.cohorts};
                          delete newCohorts[cohortId];
                          const updatedDcr = {...dataCleanRoom, cohorts: newCohorts};
                          setDataCleanRoom(updatedDcr);
                          sessionStorage.setItem('dataCleanRoom', JSON.stringify(updatedDcr));
                        } else {
                          // Add cohort with all variables - create new cohorts object to trigger re-render
                          const newCohorts = {...(dataCleanRoom.cohorts || {})};
                          newCohorts[cohortId] = Object.keys(cohortInfo?.variables || {});
                          const updatedDcr = {...dataCleanRoom, cohorts: newCohorts};
                          setDataCleanRoom(updatedDcr);
                          sessionStorage.setItem('dataCleanRoom', JSON.stringify(updatedDcr));
                        }
                      }}
                    >
                      <div>
                        <span className="font-medium">{cohortId}</span>
                        <span className="text-sm text-base-content/60 ml-2">({variableCount} variables)</span>
                      </div>
                      <div>
                        {isSelected ? (
                          <span className="badge badge-primary">Selected</span>
                        ) : (
                          <span className="badge badge-ghost">Click to add</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              {Object.entries(cohortsData || {}).filter(([_, cohortInfo]: [string, any]) => 
                cohortInfo?.variables && Object.keys(cohortInfo.variables).length > 0
              ).length === 0 && (
                <p className="text-center text-base-content/60 py-4">No cohorts with uploaded metadata available</p>
              )}
            </div>
            
            <div className="modal-action">
              <button className="btn" onClick={() => { setShowAddCohortModal(false); setCohortSearchQuery(''); }}>Done</button>
            </div>
          </div>
        </div>
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
  manuallyIncludedOwners,
  setManuallyIncludedOwners,
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
  manuallyIncludedOwners: string[];
  setManuallyIncludedOwners: (emails: string[]) => void;
  onClose: () => void;
  isLoading: boolean;
}) => {
  // By default a data owner is EXCLUDED. They are only included when the
  // user explicitly opts them in (checks the box).
  const isExcluded = (email: string) => !manuallyIncludedOwners.includes(email);
  const toggleDataOwner = (email: string) => {
    if (manuallyIncludedOwners.includes(email)) {
      setManuallyIncludedOwners(manuallyIncludedOwners.filter(e => e !== email));
    } else {
      setManuallyIncludedOwners([...manuallyIncludedOwners, email]);
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
            ) : dataOwners.length === 0 ? (
              <div className="bg-base-200 p-3 rounded-lg mb-2">
                <p className="text-sm text-base-content/60">
                  No cohorts have been selected.
                </p>
              </div>
            ) : (
              dataOwners.map((owner) => (
                <div key={owner.email} className={`p-3 rounded-lg mb-2 flex items-start gap-3 ${isExcluded(owner.email) ? 'bg-base-200 opacity-50' : 'bg-base-200'}`}>
                  <input
                    type="checkbox"
                    checked={!isExcluded(owner.email)}
                    onChange={() => toggleDataOwner(owner.email)}
                    className="checkbox checkbox-primary mt-1"
                  />
                  <div className="flex-1">
                    <p className={`font-semibold ${isExcluded(owner.email) ? 'line-through' : ''}`}>
                      {owner.email}
                      {owner.email === userEmail && <span className="ml-2 text-xs badge badge-primary">You</span>}
                    </p>
                    <p className="text-sm text-gray-500">
                      Data Owner for: {owner.cohorts.join(', ')}
                    </p>
                  </div>
                </div>
              ))
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
