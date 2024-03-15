'use client';

import React, {useState} from 'react';
import {useCohorts} from '@/components/CohortsContext';
import {TrashIcon} from '@/components/Icons';
import {apiUrl} from '@/utils';

export default function UploadPage() {
  const {cohortsData} = useCohorts();
  const [cohortId, setCohortId] = useState('');
  const [metadataFile, setMetadataFile]: any = useState(null);
  const [dataFile, setDataFile]: any = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [publishedDCR, setPublishedDCR]: any = useState(null);

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
      // setSuccessMessage(result.message + ' Reload the page to see the new uploaded cohort.');
      clearMetadataFile();
      clearDataFile();
      setCohortId('');
      setErrorMessage('');
      console.log(result);
      // Handle success
    } catch (error: any) {
      console.error('Error uploading files:', error);
      setErrorMessage(error.message || 'Failed to upload files');
      setPublishedDCR(null);
      // Handle error
    }
  };

  return (
    <main className="flex justify-center p-4">
      <form onSubmit={handleSubmit} className="max-w-lg w-full space-y-4">
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
            {cohortsNoVariables.map(cohortId => (
              <option key={cohortId} value={cohortId}>
                {cohortId}
              </option>
            ))}
          </select>
        </div>

        {/* Upload cohort metadata file */}
        <div className="flex items-center">
          <label htmlFor="metadataFile" className="block text-sm">
            Cohort data dictionary{' '}
            <b>
              <code>.csv</code>
            </b>{' '}
            file (variables metadata)
          </label>
          {metadataFile && (
            <button type="button" onClick={clearMetadataFile} className="ml-2 btn btn-xs btn-neutral">
              <TrashIcon />
            </button>
          )}
        </div>
        <input type="file" id="metadataFile" className="mt-2" onChange={handleMetadataFileChange} required />

        {/* Upload data file */}
        <div className="flex items-center">
          <label htmlFor="dataFile" className="block text-sm">
            Cohort data file (optional)
          </label>
          {dataFile && (
            <button type="button" onClick={clearDataFile} className="ml-2 btn btn-xs btn-neutral">
              <TrashIcon />
            </button>
          )}
        </div>
        <input type="file" id="dataFile" className="mt-2" onChange={handleDataFilesChange} />

        <div>
          <button type="submit" className="btn btn-sm btn-success mt-6 text-slate-900 font-normal">
            Upload
          </button>
          {/* {successMessage && <p className="bg-success text-slate-900 mt-8 rounded-lg p-3">{successMessage}</p>} */}
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
                <p>You can now access this Data Clean Room in Decentriq to upload the cohort dataset.</p>
                <p>ℹ️ You will need to reload the page to see the new uploaded cohort in the explorer (ctrl+R).</p>
              </div>
            </div>
          )}
          {errorMessage && <p className="bg-red-500 mt-8 rounded-lg p-3">{errorMessage}</p>}
        </div>
      </form>
    </main>
  );
}
