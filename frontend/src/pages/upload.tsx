'use client';

import React, {useEffect, useState} from 'react';
import {ArrowLeft, Check, SkipForward, Upload} from 'react-feather';
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
  const {cohortsData, fetchCohortsData, userEmail} = useCohorts();
  const [cohortId, setCohortId] = useState('');
  const [uploadedCohort, setUploadedCohort]: any = useState(null);
  const [metadataFile, setMetadataFile]: any = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const [dcrIsLoading, setDcrIsLoading] = useState(false);
  const [publishedDCR, setPublishedDCR]: any = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  const [step, setStep] = useState(1);
  const [metadataExists, setMetadataExists] = useState(false);

  const cohortsUserCanEdit = cohortsData ? Object.keys(cohortsData).filter(cohortId => cohortsData[cohortId]['can_edit']) : [];

  // Check if metadata exists for selected cohort when data changes
  useEffect(() => {
    if (cohortId && cohortsData?.[cohortId]?.variables && Object.keys(cohortsData[cohortId].variables).length > 0) {
      setMetadataExists(true);
    } else {
      setMetadataExists(false);
    }
    clearMetadataFile(); // Also clear file selection when cohort changes
    setErrorMessage('');
  }, [cohortId, cohortsData]);

  // Function to clear metadata file input
  const clearMetadataFile = () => {
    setMetadataFile(null);
    const fileInput = document.getElementById('metadataFile') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  };

  // Handle Step 1: Metadata submission (or replacement)
  const handleMetadataSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!metadataExists && !metadataFile) {
        setErrorMessage("Please select a metadata dictionary file (.csv) to upload.");
        return;
    }
    setUploadedCohort(null);
    setPublishedDCR(null);
    setIsLoading(true);
    setErrorMessage('');
    const formData = new FormData();
    formData.append('cohort_id', cohortId);
    if (metadataFile) {
        formData.append('cohort_dictionary', metadataFile);
    }

    try {
      // Restore actual API call
      const response = await fetch(`${apiUrl}/upload-cohort`, {
        method: 'POST',
        body: formData,
        credentials: 'include' // Important for sending cookies
      });
      const result = await response.json();
      if (!response.ok) {
        console.error(result);
        // Improve error message extraction
        let errorMsg = JSON.stringify(result);
        if (result?.detail) {
          errorMsg = typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail);
        }
        throw new Error(errorMsg);
      }
      
      setUploadedCohort(result);
      fetchCohortsData(); 
      clearMetadataFile();
      setStep(2); // Move to next step
    } catch (error: any) {
      console.error('Error uploading file:', error);
      setErrorMessage(error.message || 'Failed to upload file');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Step 2: DCR creation submission
  const handleDcrSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!cohortId) {
        setErrorMessage('No cohort selected from Step 1.');
        return;
    }
    setPublishedDCR(null);
    setUploadedCohort(null);
    setDcrIsLoading(true);
    setErrorMessage('');
    const formData = new FormData();
    formData.append('cohort_id', cohortId);
    try {
      // Restore actual API call
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
      setPublishedDCR(result);
    } catch (error: any) {
      console.error('Error creating DCR:', error);
      setErrorMessage(error.message || 'Failed to create DCR');
    } finally {
      setDcrIsLoading(false);
    }
  };

  // Render logic
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

        {/* Common Messages Area - display errors or success messages */}
        {errorMessage && (
          <div role="alert" className="alert alert-error mb-4 shadow-md">
            <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            <span>Error: {errorMessage}</span>
          </div>
        )}
         {uploadedCohort && !errorMessage && (
          <div role="alert" className="alert alert-success mb-4 shadow-md">
            <Check className="w-6 h-6" />
            <span>{uploadedCohort.message}</span>
          </div>
        )}
        {publishedDCR && !errorMessage && (
          <div role="alert" className="alert alert-success mb-4 shadow-md">
             <Check className="w-6 h-6" />
            <span>{publishedDCR.message} {publishedDCR.dcr_url && <a href={publishedDCR.dcr_url} className="link link-neutral" target="_blank" rel="noopener noreferrer">View DCR (simulated)</a>}</span>
          </div>
        )}


        {/* Step 1: Metadata Upload Card */}
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
                 {/* Conditional message if metadata exists */} 
                 {metadataExists && (
                   <div role="alert" className="alert alert-info mt-4">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-current shrink-0 w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                     <span>Metadata already exists for cohort <strong>{cohortsData?.[cohortId]?.label || cohortId}</strong>. You can proceed directly to Step 2 or upload a new file to replace the existing metadata.</span>
                   </div>
                 )}
              </div>

              <form onSubmit={handleMetadataSubmit} className="space-y-5">
                {/* Cohort Selection */}
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
                    disabled={isLoading}
                  >
                    <option value="" disabled>Select the cohort to upload metadata for</option>
                    {cohortsUserCanEdit.map((id: string) => (
                      <option key={id} value={id}>
                        {cohortsData?.[id]?.label || id} ({id}) {/* Show label and ID */}
                      </option>
                    ))}
                    {cohortsUserCanEdit.length === 0 && <option value="" disabled>No editable cohorts available</option>}
                  </select>
                </div>

                {/* Metadata File Input - Conditionally required */} 
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
                        required={!metadataExists} // Only required if metadata doesn't exist
                        disabled={isLoading}
                      />
                      {metadataFile && (
                         <button type="button" onClick={clearMetadataFile} className="btn btn-ghost btn-sm" title="Clear file" disabled={isLoading}>
                           <TrashIcon />
                         </button>
                      )}
                   </div>
                   <div className="label">
                     <span className="label-text-alt">Must be a CSV file containing variable descriptions.</span>
                   </div>
                </div>

                {/* Action Buttons for Step 1 */}
                <div className="card-actions justify-end items-center gap-2"> {/* Added gap */}
                   {/* Show Skip button only if metadata exists */}
                   {metadataExists && (
                     <button type="button" className="btn btn-ghost" onClick={() => setStep(2)} disabled={isLoading}>
                       <SkipForward className="w-4 h-4" />
                       Skip to Step 2
                     </button>
                   )}
                   <button type="submit" className="btn btn-info" disabled={isLoading || !cohortId || (!metadataExists && !metadataFile)}>
                     {isLoading ? <span className="loading loading-spinner loading-xs"></span> : <Upload className="w-4 h-4" />}
                     {metadataExists ? 'Replace Metadata & Proceed' : 'Upload Metadata & Proceed'}
                   </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Step 2: Create DCR */}
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
                 {/* <p className="font-semibold">Select the cohort you just uploaded metadata for to create its corresponding DCR.</p> */}
               </div>

               <form onSubmit={handleDcrSubmit} className="space-y-5">
                 {/* Display the selected cohort for confirmation */} 
                 <div className="form-control">
                    <label className="label">
                     <span className="label-text font-semibold">Cohort for DCR Creation</span>
                   </label>
                   <div className="input input-bordered flex items-center h-12"> {/* Use input style for display */}
                     {cohortsData?.[cohortId]?.label || cohortId || '(No cohort selected in Step 1)'}
                   </div>
                 </div>

                 {/* Action Buttons for Step 2 */}
                 <div className="card-actions justify-between items-center">
                    <button type="button" className="btn btn-ghost" onClick={() => setStep(1)} disabled={dcrIsLoading}>
                      <ArrowLeft className="w-4 h-4" />
                      Back to Metadata
                    </button>
                    {/* Update disabled condition */}
                   <button type="submit" className="btn btn-warning" disabled={dcrIsLoading || !cohortId}>
                     {dcrIsLoading ? <span className="loading loading-spinner loading-xs"></span> : <Upload className="w-4 h-4" /> }
                     Create Data Clean Room for {cohortsData?.[cohortId]?.label || cohortId}
                   </button>
                 </div>
               </form>
             </div>
           </div>
        )}
      </div>
    </main>
  );
}
