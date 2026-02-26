'use client';

import React, {useEffect, useState} from 'react';
import {ArrowLeft, Check, SkipForward, Upload, AlertTriangle, Info as InfoIcon, XCircle} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {TrashIcon} from '@/components/Icons';
import {apiUrl} from '@/utils';

// Helper component for wizard steps
const WizardSteps = ({currentStep}: {currentStep: number}) => {
  return (
    <ul className="steps w-full mb-8">
      <li className={`step ${currentStep >= 1 ? 'step-info' : ''}`}>
        <span className="text-sm sm:text-base">Upload Metadata</span>
      </li>
      <li className={`step ${currentStep >= 2 ? 'step-info' : ''}`}>
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
                
                 {metadataExists && (
                   <div role="alert" className="alert alert-info mt-4">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-current shrink-0 w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                     <span>Metadata already exists for cohort <strong>{cohortsData?.[cohortId]?.label || cohortId}</strong>. You can proceed directly to Step 2 or upload a new file to replace the existing metadata.</span>
                   </div>
                 )}
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
                    <div className="mt-2 text-sm">
                      {metadataExists ? (
                        <p className="text-base-content/80">
                          This cohort already has a metadata dictionary in the explorer. You can{' '}
                          <a
                            href={`${apiUrl}/cohort-spreadsheet/${cohortId}`}
                            className="link link-primary font-medium"
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            download the dictionary here
                          </a>. You may replace or update the existing dictionary using the button below.
                        </p>
                      ) : (
                        <p className="text-base-content/70">
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
                     <button type="button" className="btn btn-ghost" onClick={() => setStep(2)} disabled={isLoading || isValdating}>
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
               <h2 className="card-title text-xl mb-4">Step 2: Initiate Data Clean Room (DCR) Creation</h2>
                <div className="prose prose-sm max-w-none mb-6 text-base-content/80">
                  <p>
                     Your metadata dictionary structure for cohort <strong>{cohortsData?.[cohortId]?.label || cohortId}</strong> has been processed (or simulated).
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
                    <button type="button" className="btn btn-ghost" onClick={() => setStep(1)} disabled={dcrIsLoading}>
                      <ArrowLeft className="w-4 h-4" />
                      Back to Metadata
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
