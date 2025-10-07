'use client';

import React, {useEffect, useState} from 'react';
import {ArrowLeft, Check, SkipForward, Upload, AlertTriangle, Info as InfoIcon, XCircle, FileText, Shield, Database} from 'react-feather';
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
        <span className="text-sm sm:text-base">Choose Next Action</span>
      </li>
      <li className={`step ${currentStep >= 3 ? 'step-info' : ''}`}>
        <span className="text-sm sm:text-base">Provide Data for Exploratory Analysis</span>
      </li>
    </ul>
  );
};

// Consent Form Component
const ConsentForm = ({ 
  isOpen, 
  onClose, 
  onSubmit 
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  onSubmit: (formData: any) => void; 
}) => {
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);
  const [additionalRestrictions, setAdditionalRestrictions] = useState('');
  const [showAdditionalInput, setShowAdditionalInput] = useState(false);
  const [diseaseName, setDiseaseName] = useState('');

  const dataAccessOptions = [
    { id: 'no_restriction', label: 'no restriction', info: 'Data can be used for any research purpose' },
    { id: 'general_research_use', label: 'general research use', info: 'Data can be used for general research purposes' },
    { id: 'health_medical_biomedical_use', label: 'health/medical/biomedical use', info: 'Data restricted to health, medical, or biomedical research' },
    { id: 'disease_specific', label: 'disease specific', info: 'Data restricted to research on specific diseases' },
    { id: 'population_origins_ancestry_only', label: 'population origins/ancestry only', info: 'Data restricted to population genetics and ancestry research' }
  ];

  const handleOptionChange = (optionId: string) => {
    setSelectedOptions(prev => 
      prev.includes(optionId) 
        ? prev.filter(id => id !== optionId)
        : [...prev, optionId]
    );
  };

  const handleSubmit = () => {
    const formData = {
      selectedOptions,
      additionalRestrictions: showAdditionalInput ? additionalRestrictions : '',
      diseaseName: selectedOptions.includes('disease_specific') ? diseaseName : '',
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent
    };
    onSubmit(formData);
  };

  const resetForm = () => {
    setSelectedOptions([]);
    setAdditionalRestrictions('');
    setShowAdditionalInput(false);
    setDiseaseName('');
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-base-100 rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold">Step 1(a): Select Data Access Rules</h2>
            <button 
              onClick={handleClose}
              className="btn btn-sm btn-circle btn-ghost"
            >
              <XCircle size={20} />
            </button>
          </div>

          <div className="mb-6">
            <p className="text-base-content/80 mb-4">
              Which of these data sharing options apply to your data?
            </p>

            <div className="space-y-3">
              {dataAccessOptions.map((option) => (
                <div key={option.id}>
                  <label className="flex items-center space-x-3 cursor-pointer">
                    <input
                      type="checkbox"
                      className="checkbox checkbox-primary"
                      checked={selectedOptions.includes(option.id)}
                      onChange={() => handleOptionChange(option.id)}
                    />
                    <div className="flex-1">
                      <span className="font-medium">{option.label}</span>
                      <div className="flex items-center space-x-2">
                        <InfoIcon size={16} className="text-base-content/60" />
                        <span className="text-sm text-base-content/60">{option.info}</span>
                      </div>
                    </div>
                  </label>
                  
                  {/* Disease name input - appears directly under disease specific checkbox */}
                  {option.id === 'disease_specific' && selectedOptions.includes('disease_specific') && (
                    <div className="mt-3 ml-8 p-4 bg-base-200 rounded-lg">
                      <label className="form-control w-full">
                        <div className="label">
                          <span className="label-text font-medium">Specify the disease(s):</span>
                        </div>
                        <input
                          type="text"
                          placeholder="Enter the specific disease name(s)"
                          className="input input-bordered w-full"
                          value={diseaseName}
                          onChange={(e) => setDiseaseName(e.target.value)}
                        />
                        <div className="label">
                          <span className="label-text-alt">Please specify which disease(s) this data can be used to research</span>
                        </div>
                      </label>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="mt-6">
              <button
                type="button"
                onClick={() => setShowAdditionalInput(!showAdditionalInput)}
                className="btn btn-outline btn-sm"
              >
                Add additional restrictions
              </button>
            </div>

            {showAdditionalInput && (
              <div className="mt-4">
                <label className="label">
                  <span className="label-text">Additional restrictions:</span>
                </label>
                <textarea
                  className="textarea textarea-bordered w-full"
                  placeholder="Describe any additional restrictions..."
                  value={additionalRestrictions}
                  onChange={(e) => setAdditionalRestrictions(e.target.value)}
                  rows={3}
                />
              </div>
            )}
          </div>

          <div className="flex justify-between">
            <button 
              onClick={handleClose}
              className="btn btn-ghost"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </button>
            <button 
              onClick={handleSubmit}
              className="btn btn-primary"
              disabled={selectedOptions.length === 0}
            >
              Save Access Rules
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function UploadPage() {
  const {cohortsData, fetchCohortsData, userEmail} = useCohorts();
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

  // New state for consent form
  const [showConsentForm, setShowConsentForm] = useState(false);
  const [uploadMode, setUploadMode] = useState<'dictionary' | 'consent' | null>(null);
  const [cohortStatus, setCohortStatus] = useState<{
    metadataExists: boolean;
    metadataTimestamp?: string;
    metadataFileCount?: number;
    latestMetadataFile?: string;
    consentExists: boolean;
    consentTimestamp?: string;
    consentFileCount?: number;
    historicalConsentCount?: number;
  } | null>(null);

  const cohortsUserCanEdit = cohortsData ? Object.keys(cohortsData).filter(cohortId => cohortsData[cohortId]['can_edit']) : [];

  const checkCohortStatus = async (selectedCohortId: string) => {
    if (!selectedCohortId) {
      setCohortStatus(null);
      return;
    }

    try {
      const response = await fetch(`${apiUrl}/check-cohort-status/${selectedCohortId}`, {
        credentials: 'include'
      });
      
      if (response.ok) {
        const status = await response.json();
        setCohortStatus(status);
      } else {
        console.error('Failed to check cohort status:', response.status);
        setCohortStatus(null);
      }
    } catch (error) {
      console.error('Error checking cohort status:', error);
      setCohortStatus(null);
    }
  };

  useEffect(() => {
    if (cohortId && cohortsData?.[cohortId]?.physical_dictionary_exists) {
      setMetadataExists(true);
    } else {
      setMetadataExists(false);
    }
    
    // Check cohort status when cohort changes
    if (cohortId) {
      checkCohortStatus(cohortId);
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
      setOperationMessage({text: result.message || 'Metadata uploaded successfully!', type: 'success'});
      clearMetadataFile();
      
      // Refresh cohort status to show updated metadata information
      if (cohortId) {
        await checkCohortStatus(cohortId);
      }
      
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

  const handleConsentSubmit = async (consentData: any) => {
    try {
      // Log consent data to backend
      const response = await fetch(`${apiUrl}/log-consent-declaration`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userEmail,
          cohortId,
          consentData,
          timestamp: new Date().toISOString()
        }),
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error('Failed to log consent declaration');
      }

      setShowConsentForm(false);
      setUploadMode(null);
      setOperationMessage({
        text: 'Consent declaration submitted successfully!',
        type: 'success'
      });
      
      // Refresh cohort status to show updated consent information
      if (cohortId) {
        await checkCohortStatus(cohortId);
      }
      
      // Stay on step 2 if we're in the upload flow
      if (step === 2) {
        // Keep on step 2 to show the updated status
      }
    } catch (error: any) {
      console.error('Error submitting consent:', error);
      setOperationMessage({
        text: error.message || 'Failed to submit consent declaration',
        type: 'error'
      });
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

        {step === 1 && !uploadMode && (
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h2 className="card-title text-xl mb-4">Step 1: Select Cohort and Choose Action</h2>
              <div className="prose prose-sm max-w-none mb-6 text-base-content/80">
                <p>
                  Select the cohort you want to work with and choose your action:
                </p>
              </div>

              <div className="form-control mb-6">
                <label htmlFor="cohortId" className="label">
                  <span className="label-text font-semibold">Select Cohort</span>
                </label>
                <select
                  id="cohortId"
                  className="select select-bordered w-full"
                  value={cohortId}
                  onChange={(event) => setCohortId(event.target.value)}
                  required
                >
                  <option value="" disabled>Choose a cohort to work with</option>
                  {cohortsUserCanEdit.map((id: string) => (
                    <option key={id} value={id}>
                      {cohortsData?.[id]?.label || id} ({id})
                    </option>
                  ))}
                  {cohortsUserCanEdit.length === 0 && <option value="" disabled>No editable cohorts available</option>}
                </select>
              </div>

              {cohortId && (
                <div className="mb-6">
                  {!cohortStatus && (
                    <div className="alert alert-info mb-4">
                      <span>Loading status information...</span>
                    </div>
                  )}
                  
                  {cohortStatus && (
                    <div className="card bg-base-200 mb-4">
                      <div className="card-body p-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div>
                            <div className="font-medium mb-2">Metadata Dictionary</div>
                            {cohortStatus.metadataExists ? (
                              <div className="space-y-1">
                                <div className="text-sm text-success">
                                  ✓ Latest upload: {cohortStatus.metadataTimestamp ? new Date(cohortStatus.metadataTimestamp).toLocaleString() : 'Unknown date'}
                                </div>
                                {cohortStatus.latestMetadataFile && (
                                  <div className="text-xs text-base-content/70">
                                    File: {cohortStatus.latestMetadataFile}
                                  </div>
                                )}
                                {cohortStatus.metadataFileCount && cohortStatus.metadataFileCount > 1 && (
                                  <div className="text-xs text-info">
                                    ({cohortStatus.metadataFileCount} files total, showing latest)
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div className="text-sm text-warning">
                                No metadata dictionary uploaded yet
                              </div>
                            )}
                          </div>

                          <div>
                            <div className="font-medium mb-2">Consent Declaration</div>
                            {cohortStatus.consentExists ? (
                              <div className="space-y-1">
                                <div className="text-sm text-success">
                                  ✓ Latest submission: {cohortStatus.consentTimestamp ? new Date(cohortStatus.consentTimestamp).toLocaleString() : 'Unknown date'}
                                </div>
                                {cohortStatus.historicalConsentCount && cohortStatus.historicalConsentCount > 1 && (
                                  <div className="text-xs text-info">
                                    ({cohortStatus.historicalConsentCount} submissions total, showing latest)
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div className="text-sm text-warning">
                                No consent declaration submitted yet
                              </div>
                            )}
                          </div>
                        </div>
                        
                        {/* Skip to Exploratory Analysis link - appears when both metadata and consent exist */}
                        {cohortStatus.metadataExists && cohortStatus.consentExists && (
                          <div className="mt-4 pt-4 border-t border-base-300">
                            <button 
                              onClick={() => setStep(3)}
                              className="btn btn-outline btn-sm w-full"
                            >
                              <SkipForward className="w-4 h-4" />
                              Proceed to provide data for exploratory analysis
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="prose prose-sm max-w-none mb-6 text-base-content/80">
                <p>
                  Choose what you would like to do:
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div 
                  className={`card bg-base-200 hover:bg-base-300 transition-colors ${!cohortId ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`} 
                  onClick={() => cohortId && setUploadMode('dictionary')}
                >
                  <div className="card-body text-center">
                    <FileText className="w-12 h-12 mx-auto mb-4 text-primary" />
                    <h3 className="card-title justify-center">Upload Dictionary</h3>
                    <p className="text-sm text-base-content/70">
                      Upload or replace the metadata dictionary for your cohort
                    </p>
                    {!cohortId && (
                      <p className="text-xs text-warning mt-2">
                        Select a cohort first
                      </p>
                    )}
                  </div>
                </div>

                <div 
                  className={`card bg-base-200 hover:bg-base-300 transition-colors ${!cohortId ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`} 
                  onClick={() => {
                    if (cohortId) {
                      setUploadMode('consent');
                      setShowConsentForm(true);
                    }
                  }}
                >
                  <div className="card-body text-center">
                    <Shield className="w-12 h-12 mx-auto mb-4 text-secondary" />
                    <h3 className="card-title justify-center">Submit Consent Form</h3>
                    <p className="text-sm text-base-content/70">
                      Declare data access rules and restrictions
                    </p>
                    {!cohortId && (
                      <p className="text-xs text-warning mt-2">
                        Select a cohort first
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 1 && uploadMode === 'dictionary' && (
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <div className="flex items-center justify-between mb-4">
                <h2 className="card-title text-xl">Step 1: Add or Replace Metadata Dictionary</h2>
                <button 
                  onClick={() => setUploadMode(null)}
                  className="btn btn-sm btn-ghost"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back
                </button>
              </div>
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


        <ConsentForm 
          isOpen={showConsentForm}
          onClose={() => {
            setShowConsentForm(false);
            setUploadMode(null);
          }}
          onSubmit={handleConsentSubmit}
        />

        {step === 2 && (
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h2 className="card-title text-xl mb-4">Step 2: Choose Next Action</h2>
              <div className="prose prose-sm max-w-none mb-6 text-base-content/80">
                <p>
                  Your metadata dictionary has been successfully uploaded and validated. 
                  What would you like to do next?
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div 
                  className="card bg-base-200 hover:bg-base-300 cursor-pointer transition-colors" 
                  onClick={() => {
                    setShowConsentForm(true);
                  }}
                >
                  <div className="card-body text-center">
                    <Shield className="w-12 h-12 mx-auto mb-4 text-secondary" />
                    <h3 className="card-title justify-center">Submit Consent Form</h3>
                    <p className="text-sm text-base-content/70">
                      Declare data access rules and restrictions for your cohort
                    </p>
                    {cohortStatus?.consentExists && (
                      <div className="text-xs text-success mt-2">
                        ✓ Already submitted (you can update it)
                      </div>
                    )}
                  </div>
                </div>

                <div 
                  className="card bg-base-200 hover:bg-base-300 cursor-pointer transition-colors" 
                  onClick={() => setStep(3)}
                >
                  <div className="card-body text-center">
                    <Database className="w-12 h-12 mx-auto mb-4 text-primary" />
                    <h3 className="card-title justify-center">Provide Data for Exploratory Analysis</h3>
                    <p className="text-sm text-base-content/70">
                      Proceed to provide your data for secure exploratory analysis
                    </p>
                  </div>
                </div>
              </div>

              <div className="card-actions justify-between">
                <button 
                  className="btn btn-ghost" 
                  onClick={() => setStep(1)}
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back to Upload
                </button>
              </div>
            </div>
          </div>
        )}

        {step === 3 && (
           <div className="card bg-base-100 shadow-xl">
             <div className="card-body">
               <h2 className="card-title text-xl mb-4">Step 3: Provide Data for Exploratory Analysis</h2>
                <div className="prose prose-sm max-w-none mb-6 text-base-content/80">
                  <p>
                     Your metadata dictionary structure for cohort <strong>{cohortsData?.[cohortId]?.label || cohortId}</strong> has been processed (or simulated).
                     The next step is to provide your data for secure <strong>exploratory analysis</strong> on the external Decentriq platform.
                  </p>
                  <p>
                     The analysis environment will be configured based on the variables defined in your metadata.
                     Once the environment is provisioned on Decentriq:
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
                      Back to Actions
                    </button>
                   <button type="submit" className="btn btn-warning" disabled={dcrIsLoading || !cohortId}>
                     {dcrIsLoading ? <span className="loading loading-spinner loading-xs"></span> : <Upload className="w-4 h-4" /> }
                     Provide Data for Exploratory Analysis - {cohortsData?.[cohortId]?.label || cohortId}
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
