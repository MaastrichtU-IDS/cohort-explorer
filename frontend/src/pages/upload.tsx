'use client';

import React, {useState} from 'react';
import {useCohorts} from '../components/CohortsContext';

export default function UploadPage() {
  const {cohortsData} = useCohorts();
  const [cohortId, setCohortId] = useState('');
  const [metadataFile, setMetadataFile]: any = useState(null);
  const [dataFile, setDataFile]: any = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const cohortsNoVariables = cohortsData
    ? Object.keys(cohortsData).filter(cohortId => Object.keys(cohortsData[cohortId]['variables']).length === 0)
    : [];

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
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      console.log(formData);
      const response = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      const result = await response.json();
      if (!response.ok) {
        // TODO: Improve extraction of error returned by FastAPI here
        console.log(result);
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
      setSuccessMessage(result.message + ' Reload the page to see the new uploaded cohort.');
      clearMetadataFile();
      clearDataFile();
      setCohortId('');
      setErrorMessage('');
      console.log(result);
      // Handle success
    } catch (error: any) {
      console.error('Error uploading files:', error);
      setErrorMessage(error.message || 'Failed to upload files');
      setSuccessMessage('');
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
            Cohort data dictionary (variables metadata)
          </label>
          {metadataFile && (
            <button type="button" onClick={clearMetadataFile} className="ml-2 btn btn-xs btn-neutral">
              <TrashIcon />
              {/* <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="bi bi-trash" viewBox="0 0 16 16"> <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z"/> <path fill-rule="evenodd" d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"/></svg> */}
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
          {successMessage && <p className="bg-success text-slate-900 mt-8 rounded-lg p-3">{successMessage}</p>}
          {errorMessage && <p className="bg-red-500 mt-8 rounded-lg p-3">{errorMessage}</p>}
        </div>
      </form>
    </main>
  );
}

const TrashIcon = ({width = 16, height = 16, fill = 'currentColor', className = ''}) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={width}
      height={height}
      fill={fill}
      className={className}
      viewBox="0 0 16 16"
    >
      <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z" />
      <path
        fillRule="evenodd"
        d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"
      />
    </svg>
  );
};
