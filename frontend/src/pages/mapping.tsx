'use client';

import React, { useState } from 'react';

// Define the shape of our row data
interface RowData {
  [key: string]: string | number | boolean | null | undefined;
}

// Helper component for CSV preview
interface MappingPreviewJsonTableProps {
  data: RowData[];
  maxRows: number;
}

function MappingPreviewJsonTable({ data, maxRows }: MappingPreviewJsonTableProps) {
  if (!data || !Array.isArray(data) || data.length === 0) return <div className="italic text-slate-400">No mapping data to preview.</div>;
  // Get all unique keys for columns
  const columns = Array.from(
    data.reduce<Set<string>>((cols, row) => {
      Object.keys(row).forEach(k => cols.add(k));
      return cols;
    }, new Set<string>())
  );
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
        {data.slice(0, maxRows).map((row, i) => (
          <tr key={i}>
            {columns.map(col => (
              <td key={col}>{(row[col] as string | number | boolean | null | undefined)?.toString() || ''}</td>
            ))}
          </tr>
        ))}
        {data.length > maxRows && (
          <tr><td colSpan={columns.length} className="italic text-slate-400">... (truncated)</td></tr>
        )}
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
      // First, create a copy of the response to use for both download and preview
      const responseClone = response.clone();
      
      // Handle download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `mapping_${sourceCohort}_to_${selectedTargets.join('_')}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      
      // Handle preview by parsing the response as JSON directly
      try {
        const jsonData = await responseClone.json();
        
        // Type guard to check if the data matches RowData[]
        const isValidData = Array.isArray(jsonData) && 
          jsonData.every(item => 
            typeof item === 'object' && 
            item !== null &&
            Object.values(item).every(val => 
              val === null || 
              val === undefined ||
              typeof val === 'string' || 
              typeof val === 'number' || 
              typeof val === 'boolean' ||
              (Array.isArray(val) && val.every(v => 
                v === null || 
                v === undefined ||
                typeof v === 'string' || 
                typeof v === 'number' || 
                typeof v === 'boolean'
              ))
            )
          );
        
        if (isValidData) {
          // Flatten any nested arrays in the data for display
          const flattenedData = jsonData.map(item => {
            const flatItem: any = {};
            Object.entries(item).forEach(([key, value]) => {
              if (Array.isArray(value)) {
                flatItem[key] = value.join(', ');
              } else if (value !== null && typeof value === 'object') {
                flatItem[key] = JSON.stringify(value);
              } else {
                flatItem[key] = value;
              }
            });
            return flatItem;
          });
          
          setMappingOutput(flattenedData);
        } else {
          console.warn('Received data does not match expected format');
          setMappingOutput([]);
        }
      } catch (error) {
        console.error('Error parsing JSON response:', error);
        setMappingOutput([]);
      }
      
      // Clean up the object URL
      window.URL.revokeObjectURL(url);
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
          <div className="mt-8">
            <h2 className="text-xl font-semibold mb-4">Mapping Preview</h2>
            <div className="bg-base-100 p-4 rounded-lg shadow overflow-x-auto">
              <MappingPreviewJsonTable data={mappingOutput} maxRows={40} />
            </div>
          </div>
        )}
      </div>
    </main>
  );
}