'use client';

import React, { useState } from 'react';

// Define the shape of our row data
interface RowData {
  [key: string]: string | number | boolean | null | undefined;
}

// Helper function to extract relevant fields from the mapping JSON
function transformMappingDataForPreview(jsonData: any): RowData[] {
  let allMappings: RowData[] = [];
  if (typeof jsonData !== 'object' || jsonData === null) {
    return [];
  }

  Object.values(jsonData).forEach((value: any) => {
    if (value && Array.isArray(value.mappings)) {
      const transformed = value.mappings.map((mapping: any) => {
        const newRow: RowData = {
          s_source: mapping.s_source,
          s_label: mapping.s_slabel,
          target_study: mapping.target_study,
        };

        // Find wildcard keys
        Object.keys(mapping).forEach(key => {
          if (key.endsWith('_target')) {
            newRow['target'] = mapping[key];
          } else if (key.endsWith('_tlabel')) {
            newRow['target_label'] = mapping[key];
          } else if (key.endsWith('_mapping_type')) {
            newRow['mapping_type'] = mapping[key];
          }
        });
        return newRow;
      });
      allMappings = allMappings.concat(transformed);
    }
  });

  return allMappings;
}

// Helper component for the mapping preview table
interface MappingPreviewJsonTableProps {
  data: RowData[];
}

function MappingPreviewJsonTable({ data }: MappingPreviewJsonTableProps) {
  if (!data || !Array.isArray(data) || data.length === 0) return <div className="italic text-slate-400">No mapping data to preview.</div>;
  
  // Define columns in a specific order for consistency
  const columns = ['s_source', 's_label', 'target_study', 'target', 'target_label', 'mapping_type'];

  return (
    <table className="table table-zebra w-full text-xs">
      <thead>
        <tr>
          {columns.map(col => (
            <th key={col} className="font-bold bg-base-300">{col}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row, i) => (
          <tr key={i}>
            {columns.map(col => (
              <td key={col}>{(row[col] as string | number | boolean | null | undefined)?.toString() || ''}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}


import { useCohorts } from '@/components/CohortsContext';
import {apiUrl} from '@/utils';

export default function MappingPage() {
  const { cohortsData, userEmail } = useCohorts();
  const [sourceCohort, setSourceCohort] = useState('');
  // Allow multiple target cohorts, each listed only once
  // Store selected target cohorts as strings
  const [selectedTargets, setSelectedTargets] = useState<string[]>([]);
  const [mappingOutput, setMappingOutput] = useState<RowData[] | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Filtered cohorts for both source and target menus based on search
  const filteredCohorts = Object.entries(cohortsData).filter(([cohortId, cohort]) =>
    cohortId.toLowerCase().includes(searchQuery.toLowerCase()) ||
    JSON.stringify(cohort).toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  // Loading and error state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ... inside MappingPage component, replace the existing handleMapConcepts function with this one ...

  // Backend integration
  const handleMapConcepts = async () => {
    if (!sourceCohort || selectedTargets.length === 0) {
      alert('Please select a source cohort and at least one target cohort');
      return;
    }
    setLoading(true);
    setError(null);
    setMappingOutput(null);
    try {
      // Send as [cohortId, false] for each selected target
      const target_studies = selectedTargets.map((cohortId: string) => [cohortId, false]);
      const response = await fetch(`${apiUrl}/api/generate-mapping`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          source_study: sourceCohort,
          target_studies,
        }),
      });
      if (!response.ok) {
        const result = await response.json();
        let errorMsg = result.detail || result.error || 'Failed to generate mapping';
        if (
          response.status === 404 &&
          typeof errorMsg === 'string' &&
          errorMsg.endsWith("metadata has not been added yet!")
        ) {
          setError(`The metadata of ${sourceCohort} has not been added yet!`);
          return;
        }
        throw new Error(errorMsg);
      }
      // Read the response body as text once to avoid consuming the stream multiple times
      const responseText = await response.text();

      // The backend may incorrectly return NaN, which is not valid JSON.
      // Replace all instances of NaN with null before parsing.
      const cleanedResponseText = responseText.replace(/NaN/g, 'null');

      // Handle download by creating a blob from the cleaned response text
      const blob = new Blob([cleanedResponseText], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `mapping_${sourceCohort}_to_${selectedTargets.join('_')}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      // Handle preview by parsing the cleaned response text
      try {
        const jsonData = JSON.parse(cleanedResponseText);
        const previewData = transformMappingDataForPreview(jsonData);
        setMappingOutput(previewData);
      } catch (error) {
        console.error('Error parsing JSON response for preview:', error);
        setMappingOutput([]); // Clear the preview on error
      }
    } catch (err: any) {
      setError(
        typeof err.message === 'string' && err.message.endsWith("metadata has not been added yet!")
          ? `The metadata of ${sourceCohort} has not been added yet!`
          : (err.message || 'Unknown error')
      );
    } finally {
      setLoading(false);
    }
  };

  // Show loading state if data is not yet loaded
  if (userEmail !== null && (!cohortsData || Object.keys(cohortsData).length === 0)) {
    return (
      <div className="flex flex-col items-center opacity-70 text-slate-500 mt-[20%]">
        <span className="loading loading-spinner loading-lg mb-4"></span>
        <p>Loading cohorts...</p>
      </div>
    );
  }

  // Show authentication message if not logged in
  if (userEmail === null) {
    return (
      <p className="text-red-500 text-center mt-[20%]">Authenticate to access the explorer</p>
    );
  }

  return (
    <main className="flex flex-col items-center justify-start p-8 min-h-screen bg-base-200">
      {error && (
        <span style={{ color: 'red', fontWeight: 500, marginBottom: 16, display: 'block' }}>{error}</span>
      )}
      <div className="w-full max-w-6xl space-y-8">
        <h1 className="text-3xl font-bold text-center mb-8">Cohort Mapping</h1>

        {/* Search box for filtering target cohorts */}
        <div className="mb-6 flex justify-center">
          <input
            type="text"
            placeholder="[optional] Enter a keyword to filter cohorts"
            className="input input-bordered w-full max-w-md"
            value={searchQuery}
            onChange={handleSearchChange}
          />
        </div>

        <div className="flex gap-4 justify-center">
          <div className="form-control w-full max-w-xs">
            <label className="label">
              <span className="label-text">Source Cohort</span>
            </label>
            <select 
              className="select select-bordered w-full"
              value={sourceCohort}
              onChange={(e) => setSourceCohort(e.target.value)}
            >
              <option value="">Select source cohort</option>
              {filteredCohorts.map(([cohortId, cohort]) => (
                <option key={cohortId} value={cohortId}>
                  {cohortId}
                </option>
              ))}
            </select>
          </div>

          <div className="form-control w-full max-w-xs">
            <label className="label">
              <span className="label-text">Target Cohorts</span>
            </label>
            <div className="flex flex-col max-h-64 overflow-y-scroll border rounded p-2 bg-base-100 scrollbar scrollbar-thumb-base-300 scrollbar-track-base-200">
              {filteredCohorts.map(([cohortId]) => {
                const checked = selectedTargets.includes(cohortId);
                return (
                  <label key={cohortId} className="cursor-pointer flex items-center gap-2 py-1">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={e => {
                        if (e.target.checked) {
                          setSelectedTargets(prev => [...prev, cohortId]);
                        } else {
                          setSelectedTargets(prev => prev.filter(t => t !== cohortId));
                        }
                      }}
                    />
                    <span>{cohortId}</span>
                  </label>
                );
              })}
            </div>
          </div>
        </div>

        <div className="flex justify-center mt-6">
          <button 
            className="btn btn-primary"
            onClick={handleMapConcepts}
            disabled={!sourceCohort || selectedTargets.length === 0 || loading}
          >
            {loading ? 'Mapping... (may take several minutes)' : 'Map Concepts & Download File'}
          </button>
        </div>

        {error && (
          <div className="mt-4 text-red-500 text-center">{error}</div>
        )}

        {mappingOutput && (
          <div className="mt-4 p-4 border rounded-lg bg-base-100 w-full max-w-4xl">
          <h2 className="text-lg font-bold mb-2">Mapping Preview</h2>
          <div className="overflow-x-auto">
            <MappingPreviewJsonTable data={mappingOutput} />
          </div>
        </div>
        )}
      </div>
    </main>
  );
}