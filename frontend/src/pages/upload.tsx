'use client';

import React, {useEffect, useRef, useState} from 'react';
import {ArrowLeft, Check, SkipForward, Upload, AlertTriangle, Info as InfoIcon, XCircle, Shield, Link as LinkIcon, Lock, Unlock} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {TrashIcon} from '@/components/Icons';
import {apiUrl} from '@/utils';

// DUO Permission and Modifier definitions (from GA4GH DUO ontology)
const DUO_PERMISSIONS: {[key: string]: string} = {
  NRES: 'No Restrictions',
  GRU: 'General Research Use',
  HMB: 'Health/Medical/Biomedical Research',
  DS: 'Disease Specific Research',
  POA: 'Population Origins/Ancestry Research',
};

const DUO_MODIFIERS: {[key: string]: string} = {
  NPU: 'Not-for-profit organisations only (DUO:0000045)',
  NCU: 'Non-commercial use only (DUO:0000046)',
  PUB: 'Publication required (DUO:0000019)',
  COL: 'Collaboration required (DUO:0000020)',
  IRB: 'Ethics approval required (DUO:0000021)',
  GS: 'Geographic restriction (DUO:0000022)',
  TS: 'Time limit on use (DUO:0000025)',
  US: 'User-specific restriction (DUO:0000026)',
  PS: 'Project-specific restriction (DUO:0000027)',
  IS: 'Institution-specific restriction (DUO:0000028)',
  NPOA: 'Population origins/ancestry research prohibited (DUO:0000044)',
  GSO: 'Genetic studies only (DUO:0000016)',
  RS: 'Research-specific restrictions (DUO:0000012)',
};

const COUNTRY_PRESETS: { label: string; title: string; codes: string[] }[] = [
  {
    label: 'European Union',
    title: 'European Union (27 members)',
    codes: ['AT','BE','BG','CY','CZ','DE','DK','EE','ES','FI','FR','GR','HR','HU','IE','IT','LT','LU','LV','MT','NL','PL','PT','RO','SE','SI','SK'],
  },
  {
    label: 'EFTA',
    title: 'European Free Trade Association + all EU countries',
    codes: ['AT','BE','BG','CH','CY','CZ','DE','DK','EE','ES','FI','FR','GR','HR','HU','IE','IS','IT','LI','LT','LU','LV','MT','NL','NO','PL','PT','RO','SE','SI','SK'],
  },
  {
    label: 'Commonwealth of Nations',
    title: 'Commonwealth of Nations (54 members)',
    codes: ['AG','AU','BS','BD','BB','BZ','BW','BN','CM','CA','CY','DM','SZ','FJ','GM','GH','GD','GY','IN','JM','KE','KI','LS','MW','MY','MV','MT','MU','MZ','NA','NR','NZ','NG','PK','PG','RW','KN','LC','VC','WS','SC','SL','SG','SB','ZA','LK','TZ','TO','TT','TV','UG','GB','VU','ZM'],
  },
];

// Helper component for wizard steps
const WizardSteps = ({currentStep}: {currentStep: number}) => {
  return (
    <ul className="steps w-full mb-8">
      <li className={`step ${currentStep >= 1 ? 'step-info' : ''}`}>
        <span className="text-sm sm:text-base">Upload Metadata</span>
      </li>
      <li className={`step ${currentStep >= 2 ? 'step-info' : ''}`}>
        <span className="text-sm sm:text-base">Record Data Use Permission</span>
      </li>
      <li className={`step ${currentStep >= 3 ? 'step-info' : ''}`}>
        <span className="text-sm sm:text-base">Create Data Clean Room</span>
      </li>
    </ul>
  );
};

export default function UploadPage() {
  const {cohortsData, fetchCohortsData, calculateStatistics, userEmail, isLoading: isLoadingCohorts} = useCohorts();
  const [cohortId, setCohortId] = useState('');
  const [uploadedCohort, setUploadedCohort]: any = useState(null);
  const [metadataFile, setMetadataFile]: any = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isValdating, setIsValidating] = useState(false);

  const [dcrIsLoading, setDcrIsLoading] = useState(false);
  const [publishedDCR, setPublishedDCR]: any = useState(null);
  
  const [operationMessage, setOperationMessage] = useState<{text: string, type: 'error' | 'success' | 'info' | 'warning'} | null>(null);

  const [validationErrors, setValidationErrors] = useState<string[] | null>(null);
  const [validationStatusMessage, setValidationStatusMessage] = useState<string | null>(null);
  const [isValidated, setIsValidated] = useState(false);
  const [isValidationPaneOpen, setIsValidationPaneOpen] = useState(false);

  const [step, setStep] = useState(1);
  const [metadataExists, setMetadataExists] = useState(false);

  // Blockchain consent state
  const [showAllParams, setShowAllParams] = useState(false);
  const [blockchainLoading, setBlockchainLoading] = useState(false);
  const [blockchainAuthResult, setBlockchainAuthResult] = useState<any>(null);
  const [blockchainConsentResult, setBlockchainConsentResult] = useState<any>(null);
  const [blockchainAuthEnabled, setBlockchainAuthEnabled] = useState(false);
  const [consentPermission, setConsentPermission] = useState('NRES');
  const [consentModifiers, setConsentModifiers] = useState<string[]>([]);
  const [consentDiseaseCodes, setConsentDiseaseCodes] = useState<{code: string; label: string; kind: string}[]>([]);
  const [consentAllowedCountries, setConsentAllowedCountries] = useState<string[]>([]);
  const [consentAllowedCountriesInput, setConsentAllowedCountriesInput] = useState('');
  const [consentAllowedCountriesInputOpen, setConsentAllowedCountriesInputOpen] = useState(false);
  const [consentAllowedInstitutions, setConsentAllowedInstitutions] = useState<string[]>([]);
  const [consentAllowedInstitutionsInput, setConsentAllowedInstitutionsInput] = useState('');
  const [consentAllowedInstitutionsInputOpen, setConsentAllowedInstitutionsInputOpen] = useState(false);
  const [consentAllowedInstitutionsDropdownOpen, setConsentAllowedInstitutionsDropdownOpen] = useState(false);
  const [consentAllowedProjects, setConsentAllowedProjects] = useState('');
  const [consentAllowedUsers, setConsentAllowedUsers] = useState<string[]>([]);
  const [consentAllowedUsersInput, setConsentAllowedUsersInput] = useState('');
  const [consentMoratoriumMonths, setConsentMoratoriumMonths] = useState('');
  const [consentResearchScope, setConsentResearchScope] = useState('');
  const [consentReturnTargetUri, setConsentReturnTargetUri] = useState('');
  const [consentPublicationDeadlineDays, setConsentPublicationDeadlineDays] = useState('');
  const [consentExpirationDays, setConsentExpirationDays] = useState('365');
  const [consentDataUseDescription, setConsentDataUseDescription] = useState('');
  const [consentAdditionalRestrictions, setConsentAdditionalRestrictions] = useState('');
  const [consentFormUri, setConsentFormUri] = useState('');
  const [consentMetadataUri, setConsentMetadataUri] = useState('');

  // ICD-10 targeted dropdown state
  const [icd10TargetedEntries, setIcd10TargetedEntries] = useState<{code: string; label: string; kind: string; parent?: string | null; is_target: boolean}[]>([]);
  const [icd10Loading, setIcd10Loading] = useState(false);
  const [icd10LoadFailed, setIcd10LoadFailed] = useState(false);
  const [icd10DropdownOpen, setIcd10DropdownOpen] = useState(false);
  const [icd10ExpandedChapters, setIcd10ExpandedChapters] = useState<Set<string>>(new Set());
  const [showAdditionalConstraints, setShowAdditionalConstraints] = useState(false);
  const icd10WrapperRef = useRef<HTMLDivElement>(null);
  const additionalConstraintsRef = useRef<HTMLDivElement>(null);
  const gsSectionRef = useRef<HTMLDivElement>(null);
  const isSectionRef = useRef<HTMLDivElement>(null);
  const prevModifiersRef = useRef<string[]>([]);

  const cohortsUserCanEdit = cohortsData ? Object.keys(cohortsData).filter(cohortId => cohortsData[cohortId]['can_edit']) : [];

  useEffect(() => {
    if (cohortId && cohortsData?.[cohortId]?.variables && Object.keys(cohortsData[cohortId].variables).length > 0) {
      setMetadataExists(true);
    } else {
      setMetadataExists(false);
    }
  }, [cohortId, cohortsData]);

  useEffect(() => {
    setOperationMessage(null);
    setUploadedCohort(null);
    setPublishedDCR(null);
    setStep(1);
    clearMetadataFile();
    setIsValidationPaneOpen(false);
  }, [cohortId]);

  useEffect(() => {
    setIsValidated(false);
    setValidationErrors(null);
    setValidationStatusMessage(null);
  }, [metadataFile]);

  useEffect(() => {
    if ((consentPermission === 'DS' || consentPermission === 'HMB') && icd10TargetedEntries.length === 0 && !icd10Loading && !icd10LoadFailed) {
      setIcd10Loading(true);
      fetch('/icd10-targeted.json')
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then((data: {code: string; label: string; kind: string; parent?: string | null; is_target: boolean}[]) => {
          setIcd10TargetedEntries(data);
          setIcd10Loading(false);
        })
        .catch(() => {
          setIcd10Loading(false);
          setIcd10LoadFailed(true);
        });
    }
  }, [consentPermission, icd10TargetedEntries.length, icd10Loading, icd10LoadFailed]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (icd10WrapperRef.current && !icd10WrapperRef.current.contains(e.target as Node)) {
        setIcd10DropdownOpen(false);
      }

    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  useEffect(() => {
    if ((consentPermission === 'DS' || consentPermission === 'HMB') && icd10WrapperRef.current) {
      setTimeout(() => icd10WrapperRef.current?.scrollIntoView({behavior: 'smooth', block: 'nearest'}), 80);
    }
  }, [consentPermission]);

  useEffect(() => {
    if (showAdditionalConstraints && additionalConstraintsRef.current) {
      setTimeout(() => additionalConstraintsRef.current?.scrollIntoView({behavior: 'smooth', block: 'start'}), 80);
    }
  }, [showAdditionalConstraints]);

  useEffect(() => {
    const prev = prevModifiersRef.current;
    if (consentModifiers.includes('GS') && !prev.includes('GS') && gsSectionRef.current) {
      setTimeout(() => gsSectionRef.current?.scrollIntoView({behavior: 'smooth', block: 'nearest'}), 80);
    }
    if (consentModifiers.includes('IS') && !prev.includes('IS') && isSectionRef.current) {
      setTimeout(() => isSectionRef.current?.scrollIntoView({behavior: 'smooth', block: 'nearest'}), 80);
    }
    prevModifiersRef.current = consentModifiers;
  }, [consentModifiers]);

  const clearMetadataFile = () => {
    setMetadataFile(null);
    setIsValidated(false);
    setValidationErrors(null);
    setValidationStatusMessage(null);
    setIsValidationPaneOpen(false);
    const fileInput = document.getElementById('metadataFile') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handleValidateDictionary = async () => {
    if (!cohortId || !metadataFile) {
      setOperationMessage({ text: "Please select a cohort and a metadata dictionary file first.", type: 'info' });
      return;
    }

    setIsValidating(true);
    setIsValidated(false);
    setValidationErrors(null);
    setValidationStatusMessage(null);
    setOperationMessage(null);

    const formData = new FormData();
    formData.append('cohort_id', cohortId);
    formData.append('cohort_dictionary', metadataFile);

    try {
      const response = await fetch(`${apiUrl}/upload-cohort`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });
      const result = await response.json();

      if (!response.ok) {
        if (response.status === 422 && result.detail) {
          const errors = result.detail.split('\n\n').map((e: string) => e.trim()).filter((e: string) => e);
          const criticalErrorMsg = "Critical columns are missing. Further detailed validation of rows cannot proceed.";
          
          if (result.detail.includes(criticalErrorMsg)) {
            setValidationStatusMessage(criticalErrorMsg);
            setValidationErrors(errors.filter((err: string) => err !== criticalErrorMsg)); 
          } else {
            setValidationStatusMessage("Please review the validation issues found in your file:");
            setValidationErrors(errors);
          }
        } else {
          setValidationStatusMessage("An unexpected error occurred during validation.");
          setValidationErrors([result.detail || result.message || 'Unknown error from server']);
        }
        setIsValidated(false);
      } else {
        setValidationStatusMessage("Validation successful! Your dictionary is ready for upload.");
        setValidationErrors(null);
        setIsValidated(true);
        fetchCohortsData(); 
      }
      setIsValidationPaneOpen(true);
    } catch (error: any) {
      console.error('Error during dictionary validation:', error);
      setValidationStatusMessage("A client-side error occurred during validation attempt.");
      setValidationErrors([error.message || 'Failed to connect to the server or parse response.']);
      setIsValidated(false);
      setIsValidationPaneOpen(true);
    } finally {
      setIsValidating(false);
    }
  };

  const handleMetadataSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!isValidated || !metadataFile) {
        setOperationMessage({text: "Please validate the metadata dictionary file first, or re-select a file if changed.", type: 'error'});
        return;
    }
    setUploadedCohort(null);
    setPublishedDCR(null);
    setIsLoading(true);
    setOperationMessage(null);
    const formData = new FormData();
    formData.append('cohort_id', cohortId);
    if (metadataFile) {
        formData.append('cohort_dictionary', metadataFile);
    }

    try {
      const response = await fetch(`${apiUrl}/upload-cohort`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });
      const result = await response.json();
      if (!response.ok) {
        console.error(result);
        let errorMsg = JSON.stringify(result);
        if (result?.detail) {
          errorMsg = typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail);
        }
        throw new Error(errorMsg);
      }
      
      setUploadedCohort(result);
      fetchCohortsData();
      // Recalculate statistics after successful upload
      if (calculateStatistics) {
        setTimeout(() => calculateStatistics(), 1000); // Wait for cache to update
      }
      setOperationMessage({text: result.message || 'Metadata uploaded successfully!', type: 'success'});
      clearMetadataFile();
      setStep(2);
      setBlockchainAuthResult(null);
      setBlockchainConsentResult(null);
    } catch (error: any) {
      console.error('Error uploading file:', error);
      setOperationMessage({text: error.message || 'Failed to upload file', type: 'error'});
      setMetadataFile(null);
      setIsValidated(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDcrSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!cohortId) {
        setOperationMessage({text: 'No cohort selected from Step 1.', type: 'error'});
        return;
    }
    setPublishedDCR(null);
    setDcrIsLoading(true);
    setOperationMessage(null);
    const formData = new FormData();
    formData.append('cohort_id', cohortId);
    try {
      const response = await fetch(`${apiUrl}/create-provision-dcr`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });
      const result = await response.json();
      if (!response.ok) {
        console.error(result);
        let errorMsg = JSON.stringify(result);
        if (result?.detail) {
          errorMsg = typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail);
        }
        throw new Error(errorMsg);
      }
      const dcrMessage = result.message.replace(result.dcr_url, '').replace('provisioned at', 'provisioned.').trim();
      setPublishedDCR({ 
        ...result, 
        message: dcrMessage
      }); 
      setOperationMessage({text: `${dcrMessage} You can view it on Decentriq.`, type: 'success'});
    } catch (error: any) {
      console.error('Error creating DCR:', error);
      setOperationMessage({text: error.message || 'Failed to create DCR', type: 'error'});
    } finally {
      setDcrIsLoading(false);
    }
  };

  if (userEmail === null && apiUrl !== 'mock') {
    return (
      <main className="flex flex-col items-center justify-center p-4">
         <p className="text-red-500 text-center mt-[20%]" role="alert">Authenticate to access the explorer</p>
      </main>
    );
  }

  return (
    <main className="flex flex-col items-center p-4 sm:p-8">
      <div className="w-full max-w-3xl">
        <WizardSteps currentStep={step} />

        {operationMessage && (
          <div role="alert" className={`alert alert-${operationMessage.type} mb-4 shadow-md`}>
            {operationMessage.type === 'error' && <XCircle className="stroke-current shrink-0 h-6 w-6" />}
            {operationMessage.type === 'success' && <Check className="stroke-current shrink-0 h-6 w-6" />}
            {operationMessage.type === 'warning' && <AlertTriangle className="stroke-current shrink-0 h-6 w-6" />}
            {operationMessage.type === 'info' && <InfoIcon className="stroke-current shrink-0 h-6 w-6" />}
            <span>{operationMessage.text}</span>
          </div>
        )}

        {isValidationPaneOpen && (
          <div className="fixed top-0 right-0 w-full md:w-1/3 lg:w-1/4 h-full bg-base-200 shadow-xl p-6 overflow-y-auto z-50 flex flex-col">
            <div className="flex-grow">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-semibold">Validation Results</h3>
                <button className="btn btn-sm btn-circle btn-ghost" onClick={() => setIsValidationPaneOpen(false)}>
                  <XCircle size={20} />
                </button>
              </div>

              {validationStatusMessage && (
                <div 
                  role="alert" 
                  className={`p-4 mb-4 rounded-lg shadow {
                    validationStatusMessage.includes("Critical") ? 'alert alert-error' : 
                    (isValidated ? 'alert alert-success' : 'alert alert-info')
                  }`}>
                  <div className="flex items-center">
                    {validationStatusMessage.includes("Critical") && <AlertTriangle size={20} className="mr-2 shrink-0" />}
                    {isValidated && <Check size={20} className="mr-2 shrink-0" />}
                    {!isValidated && !validationStatusMessage.includes("Critical") && <InfoIcon size={20} className="mr-2 shrink-0" />}
                    <p className="font-semibold">{validationStatusMessage}</p>
                  </div>
                </div>
              )}

              {validationErrors && validationErrors.length > 0 && (
                <div className="mt-2">
                  <p className="text-sm text-base-content/80 mb-2">
                    {validationStatusMessage && validationStatusMessage.includes("Critical") 
                      ? "The following critical issues were found:"
                      : "Details:"}
                  </p>
                  <ul className="list-disc list-inside pl-1 space-y-2 bg-base-100 p-3 rounded-md shadow">
                    {validationErrors.map((err, idx) => (
                      <li key={idx} className="text-sm">
                        {err}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {!validationStatusMessage && (!validationErrors || validationErrors.length === 0) && (
                <div className="alert alert-info mt-4">
                  <InfoIcon size={20} className="mr-2 shrink-0" />
                  <span>Validation has not been performed for the current file, or no issues were found previously.</span>
                </div>
              )}
            </div>
            <div className="mt-6 flex-shrink-0">
              <button className="btn btn-primary w-full" onClick={() => setIsValidationPaneOpen(false)}>
                Got it
              </button>
            </div>
          </div>
        )}

        {step === 1 && (
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h2 className="card-title text-xl mb-4">Step 1: Add or Replace Metadata Dictionary</h2>
              <div className="prose prose-sm max-w-none mb-6 text-base-content/80">
                <p>
                  In this step, you upload or replace the <strong>metadata dictionary</strong> (a .csv file) for your cohort.
                  This file describes the variables (columns) in your dataset but contains <strong>no actual patient data</strong>.
                </p>
                <p>
                  Providing accurate metadata is crucial for enabling data scientists to understand and effectively utilize the data within the secure Decentriq platform later.
                </p>
                
              </div>

              <form onSubmit={handleMetadataSubmit} className="space-y-5">
                <div className="form-control">
                  <label htmlFor="cohortId" className="label">
                    <span className="label-text font-semibold">Select Cohort</span>
                  </label>
                  <select
                    id="cohortId"
                    className="select select-bordered w-full"
                    value={cohortId}
                    onChange={(event) => setCohortId(event.target.value)}
                    required
                    disabled={isLoading || isValdating}
                  >
                    <option value="" disabled>Select the cohort to upload metadata for</option>
                    {cohortsUserCanEdit.map((id: string) => (
                      <option key={id} value={id}>
                        {cohortsData?.[id]?.label || id} ({id})
                      </option>
                    ))}
                    {cohortsUserCanEdit.length === 0 && (
                      <option value="" disabled>
                        {isLoadingCohorts ? "Retrieving list of cohorts..." : "No editable cohorts available"}
                      </option>
                    )}
                  </select>
                  {cohortId && (
                    <div className="mt-2">
                      {metadataExists ? (
                        <div role="alert" className="alert alert-info">
                          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-current shrink-0 w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                          <span>Metadata already exists for cohort <strong>{cohortsData?.[cohortId]?.label || cohortId}</strong>. You can proceed directly to Step 2 or upload a new file to replace the existing metadata. You may view the current metadata dictionary{' '}
                            <a
                              href={`${apiUrl}/cohort-spreadsheet/${cohortId}`}
                              className="link font-medium underline"
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              here
                            </a>.
                          </span>
                        </div>
                      ) : (
                        <p className="text-sm text-base-content/70">
                          This cohort does not yet have a metadata dictionary in the explorer. Please upload the dictionary as a CSV file using the button below.
                        </p>
                      )}
                    </div>
                  )}
                </div>

                <div className="form-control">
                   <label htmlFor="metadataFile" className="label">
                     <span className="label-text font-semibold">Metadata Dictionary File (.csv)</span>
                      {metadataExists && <span className="label-text-alt">(Optional: only needed to replace existing)</span>}
                   </label>
                   <div className="flex items-center gap-2">
                      <input
                        type="file"
                        id="metadataFile"
                        className={`file-input file-input-bordered file-input-info w-full max-w-xs ${!metadataExists ? 'file-input-required' : ''}`}
                        accept=".csv"
                        onChange={(event) => {if (event.target.files) setMetadataFile(event.target.files[0])}}
                        required={!metadataExists}
                        disabled={isLoading || isValdating}
                      />
                      {metadataFile && (
                         <button type="button" onClick={clearMetadataFile} className="btn btn-ghost btn-sm" title="Clear file" disabled={isLoading || isValdating}>
                           <TrashIcon />
                         </button>
                      )}
                   </div>
                   <div className="label">
                     <span className="label-text-alt">Must be a CSV file containing variable descriptions.</span>
                   </div>
                </div>

                <div className="card-actions justify-end items-center gap-2">
                   {metadataExists && (
                     <button type="button" className="btn btn-ghost" onClick={() => { setStep(2); setBlockchainAuthResult(null); setBlockchainConsentResult(null); }} disabled={isLoading || isValdating}>
                       <SkipForward className="w-4 h-4" />
                       Skip to Step 2
                     </button>
                   )}
                  <button 
                    type="button" 
                    className="btn btn-outline btn-accent" 
                    onClick={handleValidateDictionary}
                    disabled={isLoading || isValdating || !cohortId || !metadataFile}
                  >
                    {isValdating ? <span className="loading loading-spinner loading-xs"></span> : <Check className="w-4 h-4" />}
                    {metadataExists ? "Re-validate Selected File" : "Validate Dictionary"}
                  </button>

                   <button 
                     type="submit" 
                     className="btn btn-info" 
                     disabled={isLoading || isValdating || !cohortId || !isValidated || !metadataFile}
                   >
                     {isLoading ? <span className="loading loading-spinner loading-xs"></span> : <Upload className="w-4 h-4" />}
                     {metadataExists && isValidated ? 'Replace & Proceed' : (isValidated ? 'Upload & Proceed' : (metadataExists ? 'Replace Metadata' : 'Upload Metadata'))}
                   </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {step === 2 && (
           <div className="card bg-base-100 shadow-xl">
             <div className="card-body">
               <h2 className="card-title text-xl mb-4">Step 2: Record Permission on Blockchain</h2>
                <div className="prose prose-sm max-w-none mb-6 text-base-content/80">
                  <p>
                     Record the <strong>DUO (Data Use Ontology) permission constraints</strong> for cohort <strong>{cohortsData?.[cohortId]?.label || cohortId}</strong> on the blockchain.
                     This registers what researchers are allowed to do with the data.
                  </p>
                  <p>
                     You will first authenticate with the blockchain API, then specify the permission parameters below.
                  </p>
               </div>

               {/* Data ownership warning */}
               {cohortId && cohortsData?.[cohortId] && userEmail && !(cohortsData[cohortId].cohort_email || []).map((e: string) => e.toLowerCase()).includes(userEmail.toLowerCase()) && (
                 <div className="alert alert-warning mb-4 items-start">
                   <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
                   <div className="text-sm flex-1">
                     <p className="font-semibold">You are not listed as a data owner for this cohort</p>
                     <p className="mt-1 opacity-80">Only registered data owners are authorised to record permissions for this cohort. If you are not the data owner, you can skip to DCR creation instead.</p>
                     <div className="flex flex-wrap gap-2 mt-3">
                       <button type="button" className="btn btn-sm btn-neutral" onClick={() => setStep(3)}>
                         Skip to Creating DCR
                       </button>
                       {!blockchainAuthEnabled ? (
                         <button type="button" className="btn btn-sm btn-warning" onClick={() => setBlockchainAuthEnabled(true)}>
                           Continue with Authentication and Permission Recording (TESTING)
                         </button>
                       ) : (
                         <span className="text-xs self-center opacity-70 italic">Authentication allowed for testing purposes</span>
                       )}
                     </div>
                   </div>
                 </div>
               )}

               {/* Blockchain Auth Section */}
               {!blockchainAuthResult && (
                 <div className="mb-6">
                   <button
                     type="button"
                     className="btn btn-accent w-full"
                     disabled={blockchainLoading || (!blockchainAuthEnabled && !!(cohortId && cohortsData?.[cohortId] && userEmail && !(cohortsData[cohortId].cohort_email || []).map((e: string) => e.toLowerCase()).includes(userEmail.toLowerCase())))}
                     onClick={async () => {
                       setBlockchainLoading(true);
                       setOperationMessage(null);
                       try {
                         const resp = await fetch(`${apiUrl}/blockchain/verify`, {
                           method: 'POST',
                           credentials: 'include',
                         });
                         const data = await resp.json();
                         if (!resp.ok) {
                           throw new Error(data.detail || JSON.stringify(data));
                         }
                         setBlockchainAuthResult(data);
                         setOperationMessage({text: data.message || 'Blockchain authentication successful', type: 'success'});
                       } catch (err: any) {
                         setOperationMessage({text: `Blockchain auth failed: ${err.message}`, type: 'error'});
                       } finally {
                         setBlockchainLoading(false);
                       }
                     }}
                   >
                     {blockchainLoading
                       ? <Unlock className="w-4 h-4 animate-pulse" />
                       : <Lock className="w-4 h-4" />
                     }
                     Authenticate with Blockchain API
                   </button>
                 </div>
               )}

               {blockchainAuthResult && (
                 <div className="alert alert-success mb-4">
                   <Check className="w-5 h-5" />
                   <div className="text-sm">
                     <p className="font-semibold">Blockchain authentication successful</p>
                     <p>Address: <code className="text-xs">{blockchainAuthResult.verify?.address}</code></p>
                     <p>Email hash: <code className="text-xs">{blockchainAuthResult.verify?.emailHash}</code></p>
                   </div>
                 </div>
               )}

               {/* Consent Form - shown after auth */}
               {blockchainAuthResult && !blockchainConsentResult && (
                 <form
                   onSubmit={async (e) => {
                     e.preventDefault();
                     setBlockchainLoading(true);
                     setOperationMessage(null);
                     try {
                       const consentBody: any = {
                         cohortId: cohortId,
                         consent: {
                           permission: consentPermission,
                           modifiers: consentModifiers,
                           expirationDays: parseInt(consentExpirationDays) || 0,
                           metadataUri: consentMetadataUri || '',
                         },
                       };
                       // Include all non-empty fields — the API will validate
                       if (consentDiseaseCodes.length > 0) consentBody.consent.diseaseCode = consentDiseaseCodes.map(e => e.code).join(',');
                       if (consentAllowedCountries.length > 0) consentBody.consent.allowedCountries = consentAllowedCountries;
                       if (consentAllowedInstitutions.length > 0) consentBody.consent.allowedInstitutions = consentAllowedInstitutions;
                       if (consentAllowedProjects) consentBody.consent.allowedProjects = consentAllowedProjects.split(',').map((s: string) => s.trim()).filter(Boolean);
                       if (consentAllowedUsers.length > 0) consentBody.consent.allowedUsers = consentAllowedUsers;
                       if (consentMoratoriumMonths) consentBody.consent.moratoriumMonths = parseInt(consentMoratoriumMonths);
                       if (consentResearchScope) consentBody.consent.researchScope = consentResearchScope;
                       if (consentReturnTargetUri) consentBody.consent.returnTargetUri = consentReturnTargetUri;
                       if (consentPublicationDeadlineDays) consentBody.consent.publicationDeadlineDays = parseInt(consentPublicationDeadlineDays);
                       if (consentDataUseDescription) consentBody.consent.dataUseDescription = consentDataUseDescription;
                       if (consentAdditionalRestrictions) consentBody.consent.additionalRestrictions = consentAdditionalRestrictions;
                       consentBody.consent.consentDate = new Date().toISOString().split('T')[0];
                       if (consentFormUri) consentBody.consent.consentFormUri = consentFormUri;

                       const resp = await fetch(`${apiUrl}/blockchain/record-consent`, {
                         method: 'POST',
                         headers: {'Content-Type': 'application/json'},
                         body: JSON.stringify(consentBody),
                         credentials: 'include',
                       });
                       const data = await resp.json();
                       if (!resp.ok) {
                         throw new Error(data.detail || JSON.stringify(data));
                       }
                       setBlockchainConsentResult(data);
                       setOperationMessage({text: data.message || 'Permission recorded on blockchain', type: 'success'});
                     } catch (err: any) {
                       setOperationMessage({text: `Permission recording failed: ${err.message}`, type: 'error'});
                     } finally {
                       setBlockchainLoading(false);
                     }
                   }}
                   className="space-y-4"
                 >
                   <div className="divider">DUO Permission Declaration</div>

                   {/* Permission */}
                   <div className="form-control">
                     <label className="label"><span className="label-text font-bold text-base">Permission Type *</span></label>
                     <div className="flex flex-col gap-2 pt-1">
                       <label className="flex items-center gap-3 cursor-pointer">
                         <input type="radio" name="consentPermission" className="radio radio-primary" value="NRES" checked={consentPermission === 'NRES'} onChange={e => setConsentPermission(e.target.value)} />
                         <span className="font-medium">No Restrictions <span className="text-base-content/50 text-sm font-normal">(NRES)</span></span>
                       </label>
                       <label className="flex items-center gap-3 cursor-pointer">
                         <input type="radio" name="consentPermission" className="radio radio-primary" value="GRU" checked={consentPermission === 'GRU'} onChange={e => setConsentPermission(e.target.value)} />
                         <span className="font-medium">General Research Use <span className="text-base-content/50 text-sm font-normal">(GRU)</span></span>
                       </label>
                       <label className="flex items-center gap-3 cursor-pointer pl-7">
                         <input type="radio" name="consentPermission" className="radio radio-primary radio-sm" value="HMB" checked={consentPermission === 'HMB'} onChange={e => setConsentPermission(e.target.value)} />
                         <span>Health/Medical/Biomedical Research <span className="text-base-content/50 text-sm">(HMB)</span></span>
                       </label>
                       <label className="flex items-center gap-3 cursor-pointer pl-7">
                         <input type="radio" name="consentPermission" className="radio radio-primary radio-sm" value="DS" checked={consentPermission === 'DS'} onChange={e => setConsentPermission(e.target.value)} />
                         <span>Disease Specific Research <span className="text-base-content/50 text-sm">(DS)</span></span>
                       </label>
                       <label className="flex items-center gap-3 cursor-pointer pl-7">
                         <input type="radio" name="consentPermission" className="radio radio-primary radio-sm" value="POA" checked={consentPermission === 'POA'} onChange={e => setConsentPermission(e.target.value)} />
                         <span>Population Origins/Ancestry Research <span className="text-base-content/50 text-sm">(POA)</span></span>
                       </label>
                     </div>
                   </div>

                   {/* Target Disease(s) — ICD-10: required for DS, optional for HMB */}
                   {(consentPermission === 'DS' || consentPermission === 'HMB') && (
                     <div className="form-control" ref={icd10WrapperRef}>
                       <label className="label">
                         <span className="label-text font-semibold flex items-center gap-2">
                           <span className="text-base">Target Disease(s) — ICD-10</span>
                           {consentPermission === 'DS' && <span className="text-error ml-0.5">*</span>}
                           {icd10Loading && <span className="loading loading-spinner loading-xs opacity-60"></span>}
                         </span>
                       </label>

                       {(() => {
                         const pMap = new Map(icd10TargetedEntries.map(e => [e.code, e.parent ?? null]));
                         const selSet = new Set(consentDiseaseCodes.map(e => e.code));
                         const topLevel = consentDiseaseCodes.filter(entry => {
                           let p = pMap.get(entry.code) ?? null;
                           while (p) { if (selSet.has(p)) return false; p = pMap.get(p) ?? null; }
                           return true;
                         });
                         const removeCascade = (code: string) => setConsentDiseaseCodes(prev =>
                           prev.filter(e => {
                             if (e.code === code) return false;
                             let p = pMap.get(e.code) ?? null;
                             while (p) { if (p === code) return false; p = pMap.get(p) ?? null; }
                             return true;
                           })
                         );
                         return (
                           <>
                             {consentDiseaseCodes.length > 0 && (
                               <div className="flex flex-wrap gap-1.5 mb-2">
                                 {topLevel.map(entry => (
                                   <span key={entry.code} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border border-primary/40 bg-primary/10 text-primary font-medium">
                                     <span className="font-mono font-bold">{entry.code}</span>
                                     <span className="opacity-40 mx-0.5">—</span>
                                     <span className="font-normal max-w-[140px] truncate">{entry.label}</span>
                                     <button type="button" className="ml-0.5 opacity-50 hover:text-error hover:opacity-100 leading-none" onClick={() => removeCascade(entry.code)} aria-label={`Remove ${entry.code}`}>×</button>
                                   </span>
                                 ))}
                                 {consentDiseaseCodes.length > topLevel.length && (
                                   <span className="text-xs text-base-content/40 self-center italic">+{consentDiseaseCodes.length - topLevel.length} sub-codes</span>
                                 )}
                               </div>
                             )}
                             <div className="relative">
                               <button type="button"
                                 className="btn btn-outline w-full justify-between font-normal text-left"
                                 onClick={() => setIcd10DropdownOpen(o => !o)}
                                 disabled={icd10TargetedEntries.length === 0 && !icd10LoadFailed}>
                                 <span className="text-base-content/60">{consentDiseaseCodes.length === 0 ? 'Select target disease(s)…' : `${topLevel.length} disease group${topLevel.length !== 1 ? 's' : ''} selected — click to edit`}</span>
                                 <span className="text-xs opacity-60">▾</span>
                               </button>

                               {icd10DropdownOpen && icd10TargetedEntries.length > 0 && (() => {
                                 const selCodes = new Set(consentDiseaseCodes.map(e => e.code));
                                 const childrenMap = new Map<string, typeof icd10TargetedEntries>();
                                 icd10TargetedEntries.forEach(e => { if (e.parent) { if (!childrenMap.has(e.parent)) childrenMap.set(e.parent, []); childrenMap.get(e.parent)!.push(e); } });
                                 const parentMap = new Map(icd10TargetedEntries.map(e => [e.code, e.parent ?? null]));
                                 const ancestorSelected = (code: string): boolean => {
                                   let p = parentMap.get(code) ?? null;
                                   while (p) { if (selCodes.has(p)) return true; p = parentMap.get(p) ?? null; }
                                   return false;
                                 };
                                 const getAllDescendants = (code: string): typeof icd10TargetedEntries => {
                                   const result: typeof icd10TargetedEntries = [];
                                   const queue = [...(childrenMap.get(code) ?? [])];
                                   while (queue.length > 0) {
                                     const child = queue.shift()!;
                                     result.push(child);
                                     queue.push(...(childrenMap.get(child.code) ?? []));
                                   }
                                   return result;
                                 };
                                 const handleCheck = (entry: typeof icd10TargetedEntries[0], checked: boolean) => {
                                   if (checked) {
                                     setConsentDiseaseCodes(prev => {
                                       if (prev.some(e => e.code === entry.code)) return prev;
                                       return [...prev, {code: entry.code, label: entry.label, kind: entry.kind}];
                                     });
                                   } else {
                                     const allDesc = getAllDescendants(entry.code);
                                     const toRemove = new Set([entry.code, ...allDesc.map(e => e.code)]);
                                     setConsentDiseaseCodes(prev => prev.filter(e => !toRemove.has(e.code)));
                                   }
                                 };
                                 const chapters = icd10TargetedEntries.filter(e => e.kind === 'chapter');
                                 const blocksByParent = new Map<string, typeof icd10TargetedEntries>();
                                 const categoriesByParent = new Map<string, typeof icd10TargetedEntries>();
                                 icd10TargetedEntries.forEach(e => {
                                   if (e.kind === 'block' && e.parent) {
                                     if (!blocksByParent.has(e.parent)) blocksByParent.set(e.parent, []);
                                     blocksByParent.get(e.parent)!.push(e);
                                   }
                                   if (e.kind === 'category' && e.parent) {
                                     if (!categoriesByParent.has(e.parent)) categoriesByParent.set(e.parent, []);
                                     categoriesByParent.get(e.parent)!.push(e);
                                   }
                                 });
                                 return (
                                   <>
                                     <div className="fixed inset-0 z-[9998] bg-black/50 backdrop-blur-sm" onClick={() => setIcd10DropdownOpen(false)} />
                                     <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 pointer-events-none">
                                       <div className="w-full max-w-5xl bg-base-100 rounded-2xl shadow-2xl flex flex-col pointer-events-auto" style={{maxHeight: '85vh'}}>
                                         <div className="flex items-center justify-between px-5 py-3 border-b border-base-300 shrink-0">
                                           <div className="flex items-baseline gap-3">
                                             <h3 className="font-bold text-base">Select Target Disease(s) — ICD-10</h3>
                                             <button type="button" className="text-xs text-primary/70 hover:text-primary underline-offset-2 hover:underline"
                                               onClick={() => {
                                                 const allExpanded = chapters.every(c => icd10ExpandedChapters.has(c.code));
                                                 setIcd10ExpandedChapters(allExpanded ? new Set() : new Set(chapters.map(c => c.code)));
                                               }}>
                                               {chapters.every(c => icd10ExpandedChapters.has(c.code)) ? 'Collapse all' : 'Expand all'}
                                             </button>
                                           </div>
                                           <button type="button" className="btn btn-ghost btn-sm btn-circle" onClick={() => setIcd10DropdownOpen(false)}>✕</button>
                                         </div>
                                         <div className="flex-1 p-4 flex flex-col gap-5" style={{overflowY: 'scroll', scrollbarWidth: 'auto', scrollbarColor: '#64748b #e2e8f0'}}>
                                           {chapters.map((chapter, idx) => {
                                             const cExplicit = selCodes.has(chapter.code);
                                             const cInherited = !cExplicit && ancestorSelected(chapter.code);
                                             const cChecked = cExplicit || cInherited;
                                             const blocks = blocksByParent.get(chapter.code) ?? [];
                                             const isExpanded = icd10ExpandedChapters.has(chapter.code);
                                             const toggleChapter = () => setIcd10ExpandedChapters(prev => {
                                               const next = new Set(prev);
                                               if (next.has(chapter.code)) next.delete(chapter.code); else next.add(chapter.code);
                                               return next;
                                             });
                                             return (
                                               <div key={chapter.code}>
                                                 <div className="flex items-center gap-2 pb-1.5 mb-2 border-b-2 border-base-300 cursor-pointer" onClick={toggleChapter}>
                                                   <label className="flex items-center gap-2 cursor-pointer flex-1 min-w-0" onClick={e => e.stopPropagation()}>
                                                     <input type="checkbox" className="checkbox checkbox-sm checkbox-primary shrink-0" checked={cChecked}
                                                       onChange={ev => { if (!cInherited) { handleCheck(chapter, ev.target.checked); if (!isExpanded) toggleChapter(); } }} />
                                                     <span className="font-mono font-bold text-sm text-primary shrink-0">{chapter.code}</span>
                                                     <span className="text-sm font-semibold text-base-content/80 min-w-0 overflow-hidden">{chapter.label}</span>
                                                   </label>
                                                   <button type="button"
                                                     className="shrink-0 w-5 h-5 flex items-center justify-center rounded hover:bg-base-300 text-base-content/50 hover:text-base-content transition-colors"
                                                     onClick={e => e.stopPropagation()}
                                                     aria-label={isExpanded ? 'Collapse chapter' : 'Expand chapter'}>
                                                     <span className={`text-[10px] inline-block transition-transform duration-150 ${isExpanded ? 'rotate-90' : ''}`}>▶</span>
                                                   </button>
                                                 </div>
                                                 {isExpanded && <div className="grid grid-cols-3 gap-x-3 gap-y-0">
                                                   {blocks.map(block => {
                                                     const bExplicit = selCodes.has(block.code);
                                                     const bInherited = !bExplicit && ancestorSelected(block.code);
                                                     const bChecked = bExplicit || bInherited;
                                                     const categories = categoriesByParent.get(block.code) ?? [];
                                                     return (
                                                       <div key={block.code} className="flex flex-col">
                                                         <label className="flex items-center gap-1.5 py-1 px-1 cursor-pointer hover:bg-base-200 rounded min-w-0">
                                                           <input type="checkbox" className="checkbox checkbox-xs checkbox-primary shrink-0" checked={bChecked}
                                                             onChange={ev => { if (!bInherited) handleCheck(block, ev.target.checked); }} />
                                                           <span className="font-mono font-semibold text-sm text-primary shrink-0 w-[4.5rem]">{block.code}</span>
                                                           <span className="text-xs text-base-content/80 min-w-0 overflow-hidden leading-tight">{block.label}</span>
                                                         </label>
                                                         {categories.map(cat => {
                                                           const dExplicit = selCodes.has(cat.code);
                                                           const dInherited = !dExplicit && ancestorSelected(cat.code);
                                                           const dChecked = dExplicit || dInherited;
                                                           return (
                                                             <label key={cat.code} className="flex items-center gap-1 py-0.5 pl-4 pr-1 cursor-pointer hover:bg-base-200 rounded min-w-0">
                                                               <input type="checkbox" className="checkbox checkbox-primary shrink-0" checked={dChecked}
                                                                 style={{width: '0.6rem', height: '0.6rem', minWidth: '0.6rem'}}
                                                                 onChange={ev => { if (!dInherited) handleCheck(cat, ev.target.checked); }} />
                                                               <span className="font-mono text-primary font-semibold shrink-0 w-[3.5rem]" style={{fontSize: '0.65rem'}}>{cat.code}</span>
                                                               <span className="text-base-content/65 min-w-0 overflow-hidden leading-tight" style={{fontSize: '0.65rem'}}>{cat.label}</span>
                                                             </label>
                                                           );
                                                         })}
                                                       </div>
                                                     );
                                                   })}
                                                 </div>}
                                               </div>
                                             );
                                           })}
                                         </div>
                                         <div className="px-5 py-3 border-t border-base-300 flex justify-between items-center shrink-0">
                                           <span className="text-sm text-base-content/60">{topLevel.length} disease group{topLevel.length !== 1 ? 's' : ''} selected</span>
                                           <button type="button" className="btn btn-sm btn-primary" onClick={() => setIcd10DropdownOpen(false)}>Done</button>
                                         </div>
                                       </div>
                                     </div>
                                   </>
                                 );
                               })()}
                             </div>
                           </>
                         );
                       })()}

                       {icd10LoadFailed && (
                         <label className="label"><span className="label-text-alt text-warning">Could not load ICD-10 data. Restart the dev server or rebuild the Docker container.</span></label>
                       )}
                       {consentDiseaseCodes.length === 0 && (
                         <label className="label">
                           <span className="label-text-alt text-base-content/60">
                             ICD-10 (2019). Select from curated disease list.
                             {consentPermission === 'DS' ? ' Required for Disease Specific Research.' : ' Optional for HMB — specify to narrow scope.'}
                           </span>
                         </label>
                       )}
                     </div>
                   )}

                   {consentPermission !== 'NRES' && (
                     <div className="mt-6">
                       <div className="divider my-4"></div>
                       <div className="flex justify-center">
                         <button type="button" className="btn btn-primary normal-case gap-2 px-8 shadow-md" onClick={() => setShowAdditionalConstraints(v => !v)}>
                           {showAdditionalConstraints ? '▾ Hide additional constraints' : '▸ Show additional constraints'}
                         </button>
                       </div>
                       {showAdditionalConstraints && (
                         <div className="space-y-4 mt-3 pt-3 border-t border-base-200" ref={additionalConstraintsRef}>
                           <div className="form-control">
                             <label className="label"><span className="label-text font-semibold">DUO Modifiers</span></label>
                             <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 p-2 border rounded-lg">
                               {Object.entries(DUO_MODIFIERS).map(([code, label]) => (
                                 <label key={code} className="flex items-start gap-2 cursor-pointer p-1 hover:bg-base-200 rounded">
                                   <input
                                     type="checkbox"
                                     className="checkbox checkbox-sm checkbox-accent mt-0.5"
                                     checked={consentModifiers.includes(code)}
                                     onChange={e => {
                                       if (e.target.checked) setConsentModifiers([...consentModifiers, code]);
                                       else {
                                         setConsentModifiers(consentModifiers.filter(m => m !== code));
                                         if (code === 'GS') { setConsentAllowedCountries([]); setConsentAllowedCountriesInput(''); setConsentAllowedCountriesInputOpen(false); }
                                         if (code === 'IS') { setConsentAllowedInstitutions([]); setConsentAllowedInstitutionsInput(''); setConsentAllowedInstitutionsInputOpen(false); }
                                         if (code === 'US') { setConsentAllowedUsers([]); setConsentAllowedUsersInput(''); }
                                       }
                                     }}
                                   />
                                   <span className="text-xs"><strong>{code}</strong> — {label}</span>
                                 </label>
                               ))}
                             </div>
                           </div>

                           {consentModifiers.includes('GS') && (
                             <div className="form-control" ref={gsSectionRef}>
                               <label className="label"><span className="label-text font-semibold">Allowed Countries {consentModifiers.includes('GS') && <span className="text-error">*</span>}</span></label>
                               {(consentAllowedCountries.length === 0 || consentAllowedCountriesInputOpen || consentAllowedCountriesInput) && (
                               <input type="text" className="input input-bordered" placeholder="e.g. NL, DE, KP — press Enter to add" value={consentAllowedCountriesInput}
                                 onChange={e => setConsentAllowedCountriesInput(e.target.value)}
                                 onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); const tokens = consentAllowedCountriesInput.split(',').map(s => s.trim().toUpperCase()).filter(Boolean); if (tokens.length) { setConsentAllowedCountries(prev => [...prev, ...tokens.filter(t => !prev.includes(t))]); setConsentAllowedCountriesInput(''); setConsentAllowedCountriesInputOpen(false); } } }}
                                 onBlur={() => { const tokens = consentAllowedCountriesInput.split(',').map(s => s.trim().toUpperCase()).filter(Boolean); if (tokens.length) { setConsentAllowedCountries(prev => [...prev, ...tokens.filter(t => !prev.includes(t))]); setConsentAllowedCountriesInput(''); setConsentAllowedCountriesInputOpen(false); } }}
                               />
                               )}
                               {consentAllowedCountries.length > 0 && (
                                 <div className="flex flex-wrap gap-1.5 mt-2">
                                   {consentAllowedCountries.map(c => (
                                     <span key={c} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border border-primary/40 bg-primary/10 text-primary font-medium">
                                       <span>{c}</span><button type="button" className="ml-0.5 opacity-50 hover:text-error hover:opacity-100 leading-none" onClick={() => setConsentAllowedCountries(prev => prev.filter(x => x !== c))}>×</button>
                                     </span>
                                   ))}
                                 </div>
                               )}
                               <div className="flex flex-wrap items-center gap-1 mt-2">
                                 <span className="text-xs text-base-content/50 mr-1">Quick add:</span>
                                 {COUNTRY_PRESETS.map(preset => (
                                   <button key={preset.label} type="button" title={preset.title} className="btn btn-xs btn-outline" onClick={() => setConsentAllowedCountries(prev => [...prev, ...preset.codes.filter(c => !prev.includes(c))])}>{preset.label}</button>
                                 ))}
                                 {consentAllowedCountries.length > 0 && !consentAllowedCountriesInputOpen && (
                                   <button type="button" className="btn btn-xs btn-ghost text-primary" onClick={() => setConsentAllowedCountriesInputOpen(true)}>+ Add another country</button>
                                 )}
                                 {consentAllowedCountries.length > 0 && (
                                   <button type="button" className="btn btn-xs btn-error btn-outline ml-2" onClick={() => { setConsentAllowedCountries([]); setConsentAllowedCountriesInput(''); setConsentAllowedCountriesInputOpen(false); }}>Clear all</button>
                                 )}
                               </div>
                               <label className="label"><span className="label-text-alt">ISO-3166 country codes. Required when GS modifier is set.</span></label>
                             </div>
                           )}

                           {consentModifiers.includes('IS') && (
                             <div className="form-control" ref={isSectionRef}>
                               <label className="label"><span className="label-text font-semibold">Allowed Institutions {consentModifiers.includes('IS') && <span className="text-error">*</span>}</span></label>
                               {consentAllowedInstitutions.length > 0 && (
                                 <div className="flex flex-wrap gap-1.5 mb-2">
                                   {consentAllowedInstitutions.map(inst => (
                                     <span key={inst} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border border-primary/40 bg-primary/10 text-primary font-medium">
                                       <span>{inst}</span>
                                       <button type="button" className="ml-0.5 opacity-50 hover:text-error hover:opacity-100 leading-none" onClick={() => setConsentAllowedInstitutions(prev => prev.filter(x => x !== inst))}>×</button>
                                     </span>
                                   ))}
                                 </div>
                               )}
                               <div className="relative">
                                 <button type="button" className="btn btn-outline w-full justify-between font-normal text-left"
                                   onClick={() => setConsentAllowedInstitutionsDropdownOpen(o => !o)}>
                                   <span className="text-base-content/60">{consentAllowedInstitutions.length === 0 ? 'Select allowed institutions…' : `${consentAllowedInstitutions.length} institution${consentAllowedInstitutions.length !== 1 ? 's' : ''} selected — click to edit`}</span>
                                   <span className="text-xs opacity-60">▾</span>
                                 </button>
                                 {consentAllowedInstitutionsDropdownOpen && (() => {
                                   const institutionNames = [...new Set(Object.values(cohortsData as Record<string, {institution?: string}>).map(c => c.institution).filter((x): x is string => Boolean(x)))].sort();
                                   return (
                                     <div className="absolute z-50 left-0 w-full bg-base-100 border border-base-300 rounded-lg shadow-xl mt-1">
                                       <div className="max-h-[40vh] overflow-y-auto p-2" style={{columnCount: 2, columnGap: '0.5rem'}}>
                                         {institutionNames.map(name => (
                                           <label key={name} className="flex items-center gap-1.5 py-1 cursor-pointer hover:bg-base-200 rounded px-1" style={{breakInside: 'avoid' as const}}>
                                             <input type="checkbox" className="checkbox checkbox-xs checkbox-primary shrink-0"
                                               checked={consentAllowedInstitutions.includes(name)}
                                               onChange={ev => {
                                                 if (ev.target.checked) setConsentAllowedInstitutions(prev => [...prev, name]);
                                                 else setConsentAllowedInstitutions(prev => prev.filter(x => x !== name));
                                               }} />
                                             <span className="text-xs text-base-content/80">{name}</span>
                                           </label>
                                         ))}
                                       </div>
                                       <div className="px-3 py-2 border-t border-base-300 flex justify-between items-center">
                                         <span className="text-xs text-base-content/60">{consentAllowedInstitutions.length} selected</span>
                                         <button type="button" className="btn btn-xs btn-primary" onClick={() => setConsentAllowedInstitutionsDropdownOpen(false)}>Done</button>
                                       </div>
                                     </div>
                                   );
                                 })()}
                               </div>
                               <label className="label"><span className="label-text-alt">Select from cohort member institutions. Required when IS modifier is set.</span></label>
                             </div>
                           )}

                           {consentModifiers.includes('PS') && (
                             <div className="form-control">
                               <label className="label"><span className="label-text font-semibold">Allowed Projects {consentModifiers.includes('PS') && <span className="text-error">*</span>}</span></label>
                               <input type="text" className="input input-bordered" placeholder="e.g. project-001, project-002" value={consentAllowedProjects} onChange={e => setConsentAllowedProjects(e.target.value)} required={consentModifiers.includes('PS')} />
                               <label className="label"><span className="label-text-alt">Comma-separated project IDs. Required when PS modifier is set.</span></label>
                             </div>
                           )}

                           {consentModifiers.includes('US') && (
                             <div className="form-control">
                               <label className="label"><span className="label-text font-semibold">Allowed Users {consentModifiers.includes('US') && <span className="text-error">*</span>}</span></label>
                               <input type="text" className="input input-bordered" placeholder="User address or email hash — press Enter to add" value={consentAllowedUsersInput}
                                 onChange={e => setConsentAllowedUsersInput(e.target.value)}
                                 onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); const tokens = consentAllowedUsersInput.split(',').map(s => s.trim()).filter(Boolean); if (tokens.length) { setConsentAllowedUsers(prev => [...prev, ...tokens.filter(t => !prev.includes(t))]); setConsentAllowedUsersInput(''); } } }}
                                 onBlur={() => { const tokens = consentAllowedUsersInput.split(',').map(s => s.trim()).filter(Boolean); if (tokens.length) { setConsentAllowedUsers(prev => [...prev, ...tokens.filter(t => !prev.includes(t))]); setConsentAllowedUsersInput(''); } }}
                               />
                               {consentAllowedUsers.length > 0 && (
                                 <div className="flex flex-wrap gap-2 mt-2">
                                   {consentAllowedUsers.map(u => (
                                     <span key={u} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-base border border-primary/40 bg-primary/10 text-primary font-medium">
                                       {u}<button type="button" className="ml-1 text-primary/60 hover:text-error transition-colors leading-none" onClick={() => setConsentAllowedUsers(prev => prev.filter(x => x !== u))}>×</button>
                                     </span>
                                   ))}
                                 </div>
                               )}
                               <label className="label"><span className="label-text-alt">User addresses or email hashes. Required when US modifier is set.</span></label>
                             </div>
                           )}

                           {consentModifiers.includes('MOR') && (
                             <div className="form-control">
                               <label className="label"><span className="label-text font-semibold">Moratorium Months {consentModifiers.includes('MOR') && <span className="text-error">*</span>}</span></label>
                               <input type="number" className="input input-bordered" min="1" placeholder="e.g. 12" value={consentMoratoriumMonths} onChange={e => setConsentMoratoriumMonths(e.target.value)} required={consentModifiers.includes('MOR')} />
                               <label className="label"><span className="label-text-alt">Number of months for publication moratorium. Required when MOR modifier is set.</span></label>
                             </div>
                           )}

                           {consentModifiers.includes('RS') && (
                             <div className="form-control">
                               <label className="label"><span className="label-text font-semibold">Research Scope {consentModifiers.includes('RS') && <span className="text-error">*</span>}</span></label>
                               <textarea className="textarea textarea-bordered" placeholder="Describe the allowed research scope" value={consentResearchScope} onChange={e => setConsentResearchScope(e.target.value)} required={consentModifiers.includes('RS')} />
                               <label className="label"><span className="label-text-alt">Free-text research scope. Required when RS modifier is set.</span></label>
                             </div>
                           )}

                           {consentModifiers.includes('RTN') && (
                             <div className="form-control">
                               <label className="label"><span className="label-text font-semibold">Return Target URI {consentModifiers.includes('RTN') && <span className="text-error">*</span>}</span></label>
                               <input type="url" className="input input-bordered" placeholder="https://..." value={consentReturnTargetUri} onChange={e => setConsentReturnTargetUri(e.target.value)} required={consentModifiers.includes('RTN')} />
                               <label className="label"><span className="label-text-alt">URI where derived data should be returned. Required when RTN modifier is set.</span></label>
                             </div>
                           )}

                           {consentModifiers.includes('PUB') && (
                             <div className="form-control">
                               <label className="label"><span className="label-text font-semibold">Publication Deadline (days)</span></label>
                               <input type="number" className="input input-bordered" min="1" placeholder="e.g. 365" value={consentPublicationDeadlineDays} onChange={e => setConsentPublicationDeadlineDays(e.target.value)} />
                               <label className="label"><span className="label-text-alt">Days allowed for publication obligation. Relevant when PUB modifier is set.</span></label>
                             </div>
                           )}

                           {consentModifiers.includes('TS') && (
                           <div className="form-control">
                             <label className="label"><span className="label-text font-semibold">Time Limit (days) <span className="text-error">*</span></span></label>
                             <input type="number" className="input input-bordered" min="1" placeholder="e.g. 365" value={consentExpirationDays} onChange={e => setConsentExpirationDays(e.target.value)} required />
                             <label className="label"><span className="label-text-alt">Number of days the permission is valid for. Required when TS modifier is set.</span></label>
                           </div>
                           )}
                         </div>
                       )}
                     </div>
                   )}


                   <div className="card-actions justify-between items-center pt-4">
                     <button type="button" className="btn btn-ghost" onClick={() => setStep(1)} disabled={blockchainLoading}>
                       <ArrowLeft className="w-4 h-4" />
                       Back to Metadata
                     </button>
                     <button type="submit" className="btn btn-accent" disabled={blockchainLoading}>
                       {blockchainLoading ? <span className="loading loading-spinner loading-xs"></span> : <Shield className="w-4 h-4" />}
                       Record Permission on Blockchain
                     </button>
                   </div>
                 </form>
               )}

               {/* Consent Result */}
               {blockchainConsentResult && (
                 <div className="space-y-4">
                   <div className="alert alert-success">
                     <Check className="w-5 h-5" />
                     <div>
                       <p className="font-semibold">{blockchainConsentResult.message}</p>
                     </div>
                   </div>
                   <div className="bg-base-200 rounded-lg p-4 text-sm space-y-2">
                     <p><strong>Cohort:</strong> {blockchainConsentResult.blockchain_response?.cohortId}</p>
                     <p><strong>Cohort Hash:</strong> <code className="text-xs break-all">{blockchainConsentResult.blockchain_response?.cohortHash}</code></p>
                     <p><strong>Transaction Hash:</strong> <code className="text-xs break-all">{blockchainConsentResult.blockchain_response?.txHash}</code></p>
                     <p><strong>Owner Address:</strong> <code className="text-xs break-all">{blockchainConsentResult.blockchain_response?.ownerAddress}</code></p>
                     {blockchainConsentResult.blockchain_response?.consent && (
                       <>
                         <p><strong>Permission:</strong> {blockchainConsentResult.blockchain_response.consent.permission} — {blockchainConsentResult.blockchain_response.consent.permission_label}</p>
                         {blockchainConsentResult.blockchain_response.consent.modifiers?.length > 0 && (
                           <p><strong>Modifiers:</strong> {blockchainConsentResult.blockchain_response.consent.modifiers.join(', ')}</p>
                         )}
                         {blockchainConsentResult.blockchain_response.consent.modifier_details?.length > 0 && (
                           <div><strong>Modifier Details:</strong>
                             <ul className="list-disc list-inside ml-2">
                               {blockchainConsentResult.blockchain_response.consent.modifier_details.map((d: any, i: number) => (
                                 <li key={i}>{d.code} — {d.label}</li>
                               ))}
                             </ul>
                           </div>
                         )}
                         {blockchainConsentResult.blockchain_response.consent.expires_at && (
                           <p><strong>Expires:</strong> {blockchainConsentResult.blockchain_response.consent.expires_at}</p>
                         )}
                       </>
                     )}
                   </div>
                   <div className="card-actions justify-between items-center pt-2">
                     <button type="button" className="btn btn-ghost" onClick={() => setStep(1)}>
                       <ArrowLeft className="w-4 h-4" />
                       Back to Metadata
                     </button>
                     <button type="button" className="btn btn-warning" onClick={() => setStep(3)}>
                       Proceed to DCR Creation
                       <SkipForward className="w-4 h-4" />
                     </button>
                   </div>
                 </div>
               )}
             </div>
           </div>
        )}

        {step === 3 && (
           <div className="card bg-base-100 shadow-xl">
             <div className="card-body">
               <h2 className="card-title text-xl mb-4">Step 3: Initiate Data Clean Room (DCR) Creation</h2>
                <div className="prose prose-sm max-w-none mb-6 text-base-content/80">
                  <p>
                     Your metadata dictionary structure for cohort <strong>{cohortsData?.[cohortId]?.label || cohortId}</strong> has been processed
                     and consent constraints have been recorded on the blockchain.
                     The next step is to initiate the creation of its secure <strong>Data Clean Room (DCR)</strong> on the external Decentriq platform.
                  </p>
                  <p>
                     This DCR will be configured based on the variables defined in your metadata.
                     Once the DCR is provisioned on Decentriq:
                  </p>
                  <ul className="list-disc pl-5">
                    <li>You (or the designated data custodian) will need to <strong>separately upload the actual patient-level data</strong> directly and securely into the Decentriq DCR.</li>
                    <li>Patient data <strong>never</strong> passes through or is stored by this Cohort Explorer application.</li>
                    <li>Data scientists can then request access to perform analysis within the secure confines of the DCR.</li>
                  </ul>
               </div>

               <form onSubmit={handleDcrSubmit} className="space-y-5">
                 <div className="form-control">
                    <label className="label">
                     <span className="label-text font-semibold">Cohort for DCR Creation</span>
                   </label>
                   <div className="input input-bordered flex items-center h-12">
                     {cohortsData?.[cohortId]?.label || cohortId || '(No cohort selected in Step 1)'}
                   </div>
                 </div>

                 <div className="card-actions justify-between items-center">
                    <button type="button" className="btn btn-ghost" onClick={() => setStep(2)} disabled={dcrIsLoading}>
                      <ArrowLeft className="w-4 h-4" />
                      Back to Consent
                    </button>
                   <button type="submit" className="btn btn-warning" disabled={dcrIsLoading || !cohortId}>
                     {dcrIsLoading ? <span className="loading loading-spinner loading-xs"></span> : <Upload className="w-4 h-4" /> }
                     Create Data Clean Room for {cohortsData?.[cohortId]?.label || cohortId}
                   </button>
                 </div>
               </form>
             </div>
           </div>
        )}

        {/* Display publishedDCR information - this block is specifically for DCR success */}
        {publishedDCR && !operationMessage?.text.includes("Failed to create DCR") && (
          <div role="alert" className="alert alert-success mb-4 shadow-md">
            <div className="flex items-start">
              <Check className="w-6 h-6 mr-2 shrink-0" />
              <div>
                <span className="font-semibold">{publishedDCR.message}</span>
                {publishedDCR.dcr_url && (
                  <p className="text-sm mt-1">
                    <a href={publishedDCR.dcr_url} className="link link-neutral hover:link-primary" target="_blank" rel="noopener noreferrer">
                      View on Decentriq: <span className="break-all">{publishedDCR.dcr_url}</span>
                    </a>
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
