'use client';

import React, {useEffect, useState} from 'react';
import {ArrowLeft, Check, SkipForward, Upload, AlertTriangle, Info as InfoIcon, XCircle, Shield, Link as LinkIcon} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {TrashIcon} from '@/components/Icons';
import {apiUrl} from '@/utils';

// DUO Permission and Modifier definitions (from GA4GH DUO ontology)
const DUO_PERMISSIONS: {[key: string]: string} = {
  NRES: 'No Restriction',
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
  MOR: 'Publication moratorium (DUO:0000024)',
  TS: 'Time limit on use (DUO:0000025)',
  US: 'User-specific restriction (DUO:0000026)',
  PS: 'Project-specific restriction (DUO:0000027)',
  IS: 'Institution-specific restriction (DUO:0000028)',
  RTN: 'Return to database or resource (DUO:0000029)',
  CC: 'Clinical care use (DUO:0000043)',
  NPOA: 'Population origins/ancestry research prohibited (DUO:0000044)',
  GSO: 'Genetic studies only (DUO:0000016)',
  RS: 'Research-specific restrictions (DUO:0000012)',
  NMDS: 'No general methods research (DUO:0000015)',
};

// Helper component for wizard steps
const WizardSteps = ({currentStep}: {currentStep: number}) => {
  return (
    <ul className="steps w-full mb-8">
      <li className={`step ${currentStep >= 1 ? 'step-info' : ''}`}>
        <span className="text-sm sm:text-base">Upload Metadata</span>
      </li>
      <li className={`step ${currentStep >= 2 ? 'step-info' : ''}`}>
        <span className="text-sm sm:text-base">Record Data Use Consent</span>
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
  const [consentPermission, setConsentPermission] = useState('GRU');
  const [consentModifiers, setConsentModifiers] = useState<string[]>([]);
  const [consentDiseaseCode, setConsentDiseaseCode] = useState('');
  const [consentAllowedCountries, setConsentAllowedCountries] = useState('');
  const [consentAllowedInstitutions, setConsentAllowedInstitutions] = useState('');
  const [consentAllowedProjects, setConsentAllowedProjects] = useState('');
  const [consentAllowedUsers, setConsentAllowedUsers] = useState('');
  const [consentMoratoriumMonths, setConsentMoratoriumMonths] = useState('');
  const [consentResearchScope, setConsentResearchScope] = useState('');
  const [consentReturnTargetUri, setConsentReturnTargetUri] = useState('');
  const [consentPublicationDeadlineDays, setConsentPublicationDeadlineDays] = useState('');
  const [consentExpirationDays, setConsentExpirationDays] = useState('0');
  const [consentDataUseDescription, setConsentDataUseDescription] = useState('');
  const [consentAdditionalRestrictions, setConsentAdditionalRestrictions] = useState('');
  const [consentDate, setConsentDate] = useState('');
  const [consentFormUri, setConsentFormUri] = useState('');
  const [consentMetadataUri, setConsentMetadataUri] = useState('');

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
               <h2 className="card-title text-xl mb-4">Step 2: Record Consent on Blockchain</h2>
                <div className="prose prose-sm max-w-none mb-6 text-base-content/80">
                  <p>
                     Record the <strong>DUO (Data Use Ontology) consent constraints</strong> for cohort <strong>{cohortsData?.[cohortId]?.label || cohortId}</strong> on the blockchain.
                     This registers what researchers are allowed to do with the data.
                  </p>
                  <p>
                     You will first authenticate with the blockchain API, then specify the consent parameters below.
                  </p>
               </div>

               {/* Data ownership warning — check against cohort_email (actual data owners), not can_edit (which includes admins) */}
               {cohortId && cohortsData?.[cohortId] && userEmail && !(cohortsData[cohortId].cohort_email || []).map((e: string) => e.toLowerCase()).includes(userEmail.toLowerCase()) && (
                 <div className="alert alert-warning mb-4">
                   <AlertTriangle className="w-5 h-5 shrink-0" />
                   <div className="text-sm">
                     <p className="font-semibold">You are not listed as a data owner for this cohort</p>
                     <p>
                       The logged-in user ({userEmail}) is not among the registered data owners of <strong>{cohortsData[cohortId].label || cohortId}</strong>.
                       {cohortsData[cohortId].cohort_email?.length > 0 && (
                         <> Known data owner(s): <strong>{cohortsData[cohortId].cohort_email.join(', ')}</strong>.</>
                       )}
                     </p>
                     <p className="mt-1 opacity-80">You may proceed for now, but in the future only data owners will be allowed to record consent.</p>
                   </div>
                 </div>
               )}

               {/* Blockchain Auth Section */}
               {!blockchainAuthResult && (
                 <div className="mb-6">
                   <button
                     type="button"
                     className="btn btn-accent w-full"
                     disabled={blockchainLoading}
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
                     {blockchainLoading ? <span className="loading loading-spinner loading-xs"></span> : <Shield className="w-4 h-4" />}
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
                       if (consentDiseaseCode) consentBody.consent.diseaseCode = consentDiseaseCode;
                       if (consentAllowedCountries) consentBody.consent.allowedCountries = consentAllowedCountries.split(',').map((s: string) => s.trim()).filter(Boolean);
                       if (consentAllowedInstitutions) consentBody.consent.allowedInstitutions = consentAllowedInstitutions.split(',').map((s: string) => s.trim()).filter(Boolean);
                       if (consentAllowedProjects) consentBody.consent.allowedProjects = consentAllowedProjects.split(',').map((s: string) => s.trim()).filter(Boolean);
                       if (consentAllowedUsers) consentBody.consent.allowedUsers = consentAllowedUsers.split(',').map((s: string) => s.trim()).filter(Boolean);
                       if (consentMoratoriumMonths) consentBody.consent.moratoriumMonths = parseInt(consentMoratoriumMonths);
                       if (consentResearchScope) consentBody.consent.researchScope = consentResearchScope;
                       if (consentReturnTargetUri) consentBody.consent.returnTargetUri = consentReturnTargetUri;
                       if (consentPublicationDeadlineDays) consentBody.consent.publicationDeadlineDays = parseInt(consentPublicationDeadlineDays);
                       if (consentDataUseDescription) consentBody.consent.dataUseDescription = consentDataUseDescription;
                       if (consentAdditionalRestrictions) consentBody.consent.additionalRestrictions = consentAdditionalRestrictions;
                       if (consentDate) consentBody.consent.consentDate = consentDate;
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
                       setOperationMessage({text: data.message || 'Consent recorded on blockchain', type: 'success'});
                     } catch (err: any) {
                       setOperationMessage({text: `Consent recording failed: ${err.message}`, type: 'error'});
                     } finally {
                       setBlockchainLoading(false);
                     }
                   }}
                   className="space-y-4"
                 >
                   <div className="divider">DUO Consent Declaration</div>

                   {/* Permission */}
                   <div className="form-control">
                     <label className="label"><span className="label-text font-semibold">Primary DUO Permission *</span></label>
                     <select className="select select-bordered w-full" value={consentPermission} onChange={e => setConsentPermission(e.target.value)} required>
                       {Object.entries(DUO_PERMISSIONS).map(([code, label]) => (
                         <option key={code} value={code}>{code} — {label}</option>
                       ))}
                     </select>
                   </div>

                   {/* Modifiers */}
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
                               else setConsentModifiers(consentModifiers.filter(m => m !== code));
                             }}
                           />
                           <span className="text-xs"><strong>{code}</strong> — {label}</span>
                         </label>
                       ))}
                     </div>
                   </div>

                   <div className="divider">Consent Parameters</div>
                   <div className="flex justify-center mb-2">
                     <div className="join">
                       <button type="button" className={`btn btn-sm join-item ${!showAllParams ? 'btn-accent' : 'btn-outline'}`} onClick={() => setShowAllParams(false)}>
                         Show Relevant Only
                       </button>
                       <button type="button" className={`btn btn-sm join-item ${showAllParams ? 'btn-accent' : 'btn-outline'}`} onClick={() => setShowAllParams(true)}>
                         Show All Parameters
                       </button>
                     </div>
                   </div>

                   {/* Disease Code */}
                   {(showAllParams || consentPermission === 'DS') && (
                     <div className="form-control">
                       <label className="label"><span className="label-text font-semibold">Disease Code {consentPermission === 'DS' && <span className="text-error">*</span>}</span></label>
                       <input type="text" className="input input-bordered" placeholder="e.g. MONDO:0005148" value={consentDiseaseCode} onChange={e => setConsentDiseaseCode(e.target.value)} required={consentPermission === 'DS'} />
                       <label className="label"><span className="label-text-alt">Ontology CURIE (MONDO/DOID/HP). Required when permission is DS.</span></label>
                     </div>
                   )}

                   {/* Allowed Countries */}
                   {(showAllParams || consentModifiers.includes('GS')) && (
                     <div className="form-control">
                       <label className="label"><span className="label-text font-semibold">Allowed Countries {consentModifiers.includes('GS') && <span className="text-error">*</span>}</span></label>
                       <input type="text" className="input input-bordered" placeholder="e.g. NL, DE, FR" value={consentAllowedCountries} onChange={e => setConsentAllowedCountries(e.target.value)} required={consentModifiers.includes('GS')} />
                       <label className="label"><span className="label-text-alt">Comma-separated ISO-3166 country codes. Required when GS modifier is set.</span></label>
                     </div>
                   )}

                   {/* Allowed Institutions */}
                   {(showAllParams || consentModifiers.includes('IS')) && (
                     <div className="form-control">
                       <label className="label"><span className="label-text font-semibold">Allowed Institutions {consentModifiers.includes('IS') && <span className="text-error">*</span>}</span></label>
                       <input type="text" className="input input-bordered" placeholder="e.g. https://ror.org/02jz4aj89" value={consentAllowedInstitutions} onChange={e => setConsentAllowedInstitutions(e.target.value)} required={consentModifiers.includes('IS')} />
                       <label className="label"><span className="label-text-alt">Comma-separated ROR IDs. Required when IS modifier is set.</span></label>
                     </div>
                   )}

                   {/* Allowed Projects */}
                   {(showAllParams || consentModifiers.includes('PS')) && (
                     <div className="form-control">
                       <label className="label"><span className="label-text font-semibold">Allowed Projects {consentModifiers.includes('PS') && <span className="text-error">*</span>}</span></label>
                       <input type="text" className="input input-bordered" placeholder="e.g. project-001, project-002" value={consentAllowedProjects} onChange={e => setConsentAllowedProjects(e.target.value)} required={consentModifiers.includes('PS')} />
                       <label className="label"><span className="label-text-alt">Comma-separated project IDs. Required when PS modifier is set.</span></label>
                     </div>
                   )}

                   {/* Allowed Users */}
                   {(showAllParams || consentModifiers.includes('US')) && (
                     <div className="form-control">
                       <label className="label"><span className="label-text font-semibold">Allowed Users {consentModifiers.includes('US') && <span className="text-error">*</span>}</span></label>
                       <input type="text" className="input input-bordered" placeholder="User addresses or email hashes" value={consentAllowedUsers} onChange={e => setConsentAllowedUsers(e.target.value)} required={consentModifiers.includes('US')} />
                       <label className="label"><span className="label-text-alt">Comma-separated user addresses or email hashes. Required when US modifier is set.</span></label>
                     </div>
                   )}

                   {/* Moratorium Months */}
                   {(showAllParams || consentModifiers.includes('MOR')) && (
                     <div className="form-control">
                       <label className="label"><span className="label-text font-semibold">Moratorium Months {consentModifiers.includes('MOR') && <span className="text-error">*</span>}</span></label>
                       <input type="number" className="input input-bordered" min="1" placeholder="e.g. 12" value={consentMoratoriumMonths} onChange={e => setConsentMoratoriumMonths(e.target.value)} required={consentModifiers.includes('MOR')} />
                       <label className="label"><span className="label-text-alt">Number of months for publication moratorium. Required when MOR modifier is set.</span></label>
                     </div>
                   )}

                   {/* Research Scope */}
                   {(showAllParams || consentModifiers.includes('RS')) && (
                     <div className="form-control">
                       <label className="label"><span className="label-text font-semibold">Research Scope {consentModifiers.includes('RS') && <span className="text-error">*</span>}</span></label>
                       <textarea className="textarea textarea-bordered" placeholder="Describe the allowed research scope" value={consentResearchScope} onChange={e => setConsentResearchScope(e.target.value)} required={consentModifiers.includes('RS')} />
                       <label className="label"><span className="label-text-alt">Free-text research scope. Required when RS modifier is set.</span></label>
                     </div>
                   )}

                   {/* Return Target URI */}
                   {(showAllParams || consentModifiers.includes('RTN')) && (
                     <div className="form-control">
                       <label className="label"><span className="label-text font-semibold">Return Target URI {consentModifiers.includes('RTN') && <span className="text-error">*</span>}</span></label>
                       <input type="url" className="input input-bordered" placeholder="https://..." value={consentReturnTargetUri} onChange={e => setConsentReturnTargetUri(e.target.value)} required={consentModifiers.includes('RTN')} />
                       <label className="label"><span className="label-text-alt">URI where derived data should be returned. Required when RTN modifier is set.</span></label>
                     </div>
                   )}

                   {/* Publication Deadline Days */}
                   {(showAllParams || consentModifiers.includes('PUB')) && (
                     <div className="form-control">
                       <label className="label"><span className="label-text font-semibold">Publication Deadline (days)</span></label>
                       <input type="number" className="input input-bordered" min="1" placeholder="e.g. 365" value={consentPublicationDeadlineDays} onChange={e => setConsentPublicationDeadlineDays(e.target.value)} />
                       <label className="label"><span className="label-text-alt">Days allowed for publication obligation. Relevant when PUB modifier is set.</span></label>
                     </div>
                   )}

                   {/* Expiration Days */}
                   <div className="form-control">
                     <label className="label"><span className="label-text font-semibold">Expiration Days {consentModifiers.includes('TS') && <span className="text-error">*</span>}</span></label>
                     <input type="number" className="input input-bordered" min="0" placeholder="0 = no expiry" value={consentExpirationDays} onChange={e => setConsentExpirationDays(e.target.value)} required={consentModifiers.includes('TS')} />
                     <label className="label"><span className="label-text-alt">Days until consent expires. 0 = no expiry. Must be &gt; 0 when TS modifier is set.</span></label>
                   </div>

                   {/* Data Use Description */}
                   <div className="form-control">
                     <label className="label"><span className="label-text font-semibold">Data Use Description</span></label>
                     <textarea className="textarea textarea-bordered" placeholder="Describe what the data may be used for" value={consentDataUseDescription} onChange={e => setConsentDataUseDescription(e.target.value)} />
                   </div>

                   {/* Additional Restrictions */}
                   <div className="form-control">
                     <label className="label"><span className="label-text font-semibold">Additional Restrictions</span></label>
                     <textarea className="textarea textarea-bordered" placeholder="Free-text restrictions beyond the modifier list" value={consentAdditionalRestrictions} onChange={e => setConsentAdditionalRestrictions(e.target.value)} />
                   </div>

                   {/* Consent Date */}
                   <div className="form-control">
                     <label className="label"><span className="label-text font-semibold">Consent Date</span></label>
                     <input type="date" className="input input-bordered" value={consentDate} onChange={e => setConsentDate(e.target.value)} />
                     <label className="label"><span className="label-text-alt">Date the original consent was obtained from data subjects</span></label>
                   </div>

                   {/* Consent Form URI */}
                   <div className="form-control">
                     <label className="label"><span className="label-text font-semibold">Consent Form URI</span></label>
                     <input type="url" className="input input-bordered" placeholder="https://... or ipfs://..." value={consentFormUri} onChange={e => setConsentFormUri(e.target.value)} />
                     <label className="label"><span className="label-text-alt">URI of the signed consent form</span></label>
                   </div>

                   {/* Metadata URI */}
                   <div className="form-control">
                     <label className="label"><span className="label-text font-semibold">Metadata URI</span></label>
                     <input type="text" className="input input-bordered" placeholder="Generic metadata URI" value={consentMetadataUri} onChange={e => setConsentMetadataUri(e.target.value)} />
                   </div>

                   <div className="card-actions justify-between items-center pt-4">
                     <button type="button" className="btn btn-ghost" onClick={() => setStep(1)} disabled={blockchainLoading}>
                       <ArrowLeft className="w-4 h-4" />
                       Back to Metadata
                     </button>
                     <button type="submit" className="btn btn-accent" disabled={blockchainLoading}>
                       {blockchainLoading ? <span className="loading loading-spinner loading-xs"></span> : <Shield className="w-4 h-4" />}
                       Record Consent on Blockchain
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
