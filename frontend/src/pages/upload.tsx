'use client';

import React, {useState} from 'react';
import {Upload, UploadCloud} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {TrashIcon} from '@/components/Icons';
import {apiUrl} from '@/utils';

export default function UploadPage() {
  const {cohortsData, fetchCohortsData, userEmail} = useCohorts();
  const [cohortId, setCohortId] = useState('');
  const [metadataFile, setMetadataFile]: any = useState(null);
  const [dataFile, setDataFile]: any = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [publishedDCR, setPublishedDCR]: any = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // const cohortsNoVariables = cohortsData
  //   ? Object.keys(cohortsData).filter(cohortId => Object.keys(cohortsData[cohortId]['variables']).length === 0)
  //   : [];
  // Show all cohorts for now, in case people need to reupload
  const cohortsNoVariables = cohortsData ? Object.keys(cohortsData).filter(cohortId => true) : [];

  const handleCohortIdChange = (event: any) => {
    setCohortId(event.target.value);
  };
  const handleMetadataFileChange = (event: any) => {
    setMetadataFile(event.target.files[0]);
  };
  const handleDataFilesChange = (event: any) => {
    setDataFile(event.target.files[0]);
  };

  // Function to clear metadata file
  const clearMetadataFile = () => {
    setMetadataFile(null);
    // @ts-ignore
    document.getElementById('metadataFile').value = '';
  };

  // Function to clear data file
  const clearDataFile = () => {
    setDataFile(null);
    // @ts-ignore
    document.getElementById('dataFile').value = '';
  };

  const handleSubmit = async (event: any) => {
    event.preventDefault();
    setPublishedDCR(null);
    setIsLoading(true);
    setErrorMessage('');
    const formData = new FormData();
    formData.append('cohort_id', cohortId);
    formData.append('cohort_dictionary', metadataFile);
    if (dataFile) formData.append('cohort_data', dataFile);
    try {
      const response = await fetch(`${apiUrl}/upload-cohort`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });
      const result = await response.json();
      if (!response.ok) {
        // TODO: Improve extraction of error returned by FastAPI here
        console.error(result);
        if (result['detail']) {
          if (result['detail'][0].msg) {
            // Handle pydantic errors
            throw new Error(result['detail'][0].loc + ': ' + result['detail'][0].msg);
          }
          // Handle our errors
          throw new Error(result['detail']);
        }
        throw new Error(JSON.stringify(result, null, 2));
      }
      setPublishedDCR(result);
      clearMetadataFile();
      clearDataFile();
      setCohortId('');
      setErrorMessage('');
      setIsLoading(false);
      console.log(result);
      fetchCohortsData();
      // Handle success
    } catch (error: any) {
      setIsLoading(false);
      console.error('Error uploading files:', error);
      setErrorMessage(error.message || 'Failed to upload files');
      setPublishedDCR(null);
      // Handle error
    }
  };

  return (
    <main className="flex justify-center p-4">
      {userEmail === null ? (
        <p className="text-red-500 text-center mt-[20%]">Authenticate to access the explorer</p>
      ) : (
        <form onSubmit={handleSubmit} className="max-w-xl w-full space-y-4">
          <div>
            <label htmlFor="cohortId" className="block text-sm">
              Cohort ID
            </label>
            <select
              id="cohortId"
              className="select mt-1 block w-full input-bordered"
              value={cohortId}
              onChange={handleCohortIdChange}
              required
            >
              <option value="">Select the cohort to upload</option>
              {cohortsNoVariables.map((cohortId: string) =>
                cohortsData[cohortId]['can_edit'] ? (
                  <React.Fragment key={cohortId}>
                    <option value={cohortId}>{cohortId}</option>
                  </React.Fragment>
                ) : null
              )}
            </select>
          </div>

          {/* Upload cohort metadata file */}
          <div className="flex items-center">
            <label htmlFor="metadataFile" className="block text-sm">
              <div role="alert" className="alert">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="stroke-current shrink-0 h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span>
                  Upload the cohort variables dictionary{' '}
                  <b>
                    <code>.csv</code>
                  </b>{' '}
                  file (just metadata)
                </span>
              </div>
              {/* SAFE: Cohort variables data dictionary{' '}
              <b>
                <code>.csv</code>
              </b>{' '}
              file (metadata) */}
            </label>
            {metadataFile && (
              <button type="button" onClick={clearMetadataFile} className="ml-2 btn btn-xs btn-neutral">
                <TrashIcon />
              </button>
            )}
          </div>
          <input type="file" id="metadataFile" className="mt-2" onChange={handleMetadataFileChange} required />

          {/* Upload data file */}
          {/* {metadataFile && <>
            <div className="flex items-center">
              <label htmlFor="dataFile" className="block text-sm">
                <div role="alert" className="alert">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="stroke-current shrink-0 h-6 w-6"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                  <span>
                    <b>Optional</b> sensible data: if you want Maastricht University to store the cohort data
                    on their server, you can upload it here. It will be used by researchers to better understand what type of data they are working with (no obligations)
                  </span>
                </div>
              </label>
              {dataFile && (
                <button type="button" onClick={clearDataFile} className="ml-2 btn btn-xs btn-neutral">
                  <TrashIcon />
                </button>
              )}
            </div>
            <input type="file" id="dataFile" className="mt-2" onChange={handleDataFilesChange} />
          </>} */}
          <div>
            <button type="submit" className="btn btn-sm btn-info mt-6 text-slate-900 font-normal">
              <Upload className="w-4 h-4" />
              Upload
            </button>
            {/* {successMessage && <p className="bg-success text-slate-900 mt-8 rounded-lg p-3">{successMessage}</p>} */}
            {isLoading && (
              <div className="flex flex-col items-center opacity-70 text-slate-500 mt-5 mb-5">
                <span className="loading loading-spinner loading-lg mb-4"></span>
                <p>Creating Data Clean Room to provision cohort {cohortId} in Decentriq...</p>
              </div>
            )}
            {publishedDCR && (
              <div className="card card-compact">
                <div className="card-body bg-success mt-8 rounded-lg text-slate-900">
                  <p>
                    ✅ Data Clean Room{' '}
                    <a href={publishedDCR['dcr_url']} className="link">
                      <b>{publishedDCR['dcr_title']}</b>
                    </a>{' '}
                    published.
                  </p>
                  <p>You can now access this Data Clean Room in Decentriq to safely upload the cohort dataset.</p>
                  {/* {!dataFile ? (
                    <p>You can now access this Data Clean Room in Decentriq to safely upload the cohort dataset.</p>
                  ) : (
                    <p>The provided cohort data file has been automatically uploaded to Decentriq.</p>
                  )} */}
                </div>
              </div>
            )}
            {errorMessage && <p className="bg-red-500 mt-8 rounded-lg p-3 whitespace-pre-line">{errorMessage}</p>}
          </div>
        </form>
      )}
    </main>
  );
}
