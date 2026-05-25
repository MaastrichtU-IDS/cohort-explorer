'use client';

import React, { useState, useRef, useEffect } from 'react';

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
        // Construct source categories codes/labels
        const sourceLabels = mapping.s_source_categories_labels || '';
        const sourceCodes = mapping.s_source_original_categories || '';
        const sourceCategoriesCodesLabels = sourceLabels && sourceCodes 
          ? `${sourceCodes} (${sourceLabels})` 
          : sourceLabels || sourceCodes || '';

        const newRow: RowData = {
          s_source: mapping.s_source,
          s_label: mapping.s_slabel,
          target_study: mapping.target_study,
          harmonization_status: mapping.harmonization_status || 'pending',
          source_categories_codes_labels: sourceCategoriesCodesLabels,
          mapping_relation: mapping.mapping_relation || '',
        };

        // Find wildcard keys for target fields
        let targetLabels = '';
        let targetCodes = '';
        
        Object.keys(mapping).forEach(key => {
          if (key.endsWith('_target')) {
            newRow['target'] = mapping[key];
          } else if (key.endsWith('_tlabel')) {
            newRow['target_label'] = mapping[key];
          } else if (key.endsWith('_target_categories_labels')) {
            targetLabels = mapping[key] || '';
          } else if (key.endsWith('_target_original_categories')) {
            targetCodes = mapping[key] || '';
          }
        });
        
        // Construct target categories codes/labels
        const targetCategoriesCodesLabels = targetLabels && targetCodes 
          ? `${targetCodes} (${targetLabels})` 
          : targetLabels || targetCodes || '';
        newRow['target_categories_codes_labels'] = targetCategoriesCodesLabels;
        
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
  sourceCohort: string;
}

function MappingPreviewJsonTable({ data, sourceCohort }: MappingPreviewJsonTableProps) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [comparisonDetails, setComparisonDetails] = useState<{
    sourceVar: string;
    sourceCohort: string;
    targetVar: string;
    targetCohort: string;
  } | null>(null);
  
  if (!data || !Array.isArray(data) || data.length === 0) return <div className="italic text-slate-400">No mapping data to preview.</div>;
  
  // Define columns in a specific order for consistency
  const columns = ['s_source', 's_label', 'target_study', 'target', 'target_label', 'compare_eda', 'mapping_relation', 'source_categories_codes_labels', 'target_categories_codes_labels', 'harmonization_status'];
  
  // Define display names for columns
  const columnDisplayNames: Record<string, string> = {
    's_source': 'source variable',
    'target': 'target variable',
    's_label': 's_label',
    'target_study': 'target_study',
    'target_label': 'target_label',
    'compare_eda': 'Compare EDAs',
    'mapping_relation': 'mapping_relation',
    'source_categories_codes_labels': 'source categories codes/labels',
    'target_categories_codes_labels': 'target categories codes/labels',
    'harmonization_status': 'harmonization_status'
  };

  return (
    <>
      <table className="table table-zebra w-full text-xs">
        <thead>
          <tr>
            {columns.map(col => (
              <th key={col} className="font-bold bg-base-300" style={col === 'mapping_relation' ? { minWidth: '180px' } : undefined}>{columnDisplayNames[col] || col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              {columns.map(col => {
                // Special handling for compare_eda column
                if (col === 'compare_eda') {
                  const sourceVar = row['s_source'] as string;
                  const targetVar = row['target'] as string;
                  const targetStudy = row['target_study'] as string;
                  
                  const handleCompare = async () => {
                    if (sourceCohort && sourceVar && targetStudy && targetVar) {
                      const imageUrl = `/api/compare-eda/${encodeURIComponent(sourceCohort)}/${encodeURIComponent(sourceVar)}/${encodeURIComponent(targetStudy)}/${encodeURIComponent(targetVar)}`;
                      console.log('Compare EDA clicked:', { sourceCohort, sourceVar, targetStudy, targetVar, imageUrl });
                      
                      setImageError(null);
                      setComparisonDetails({
                        sourceVar,
                        sourceCohort,
                        targetVar,
                        targetCohort: targetStudy
                      });
                      
                      // Fetch the image to check for errors before displaying
                      try {
                        const response = await fetch(imageUrl);
                        if (!response.ok) {
                          // Try to parse error message from JSON response
                          const contentType = response.headers.get('content-type');
                          if (contentType && contentType.includes('application/json')) {
                            const errorData = await response.json();
                            setImageError(errorData.details || errorData.error || 'Failed to load image');
                          } else {
                            const errorText = await response.text();
                            setImageError(errorText || 'Failed to load image');
                          }
                          setSelectedImage(null);
                        } else {
                          // Image loaded successfully
                          setSelectedImage(imageUrl);
                        }
                      } catch (error) {
                        console.error('Error fetching image:', error);
                        setImageError('Failed to fetch image: ' + (error as Error).message);
                        setSelectedImage(null);
                      }
                    } else {
                      console.error('Missing required fields:', { sourceCohort, sourceVar, targetStudy, targetVar });
                    }
                  };
                  
                  return (
                    <td key={col} className="text-center">
                      <button 
                        className="btn btn-xs btn-primary"
                        onClick={handleCompare}
                        disabled={!sourceVar || !targetVar || !targetStudy}
                      >
                        Compare
                      </button>
                    </td>
                  );
                }
                
                const value = row[col] as string | number | boolean | null | undefined;
                const displayValue = value === null || value === undefined || value === 'null' ? '--' : value.toString();
                const isLongText = displayValue.length > 30;
                
                return (
                  <td 
                    key={col} 
                    className={`${
                      col === 's_source' || col === 'target' ? 'bg-blue-100' : ''
                    } ${
                      (col === 'source_categories_codes_labels' || col === 'target_categories_codes_labels') && isLongText 
                        ? 'max-w-xs break-words' : ''
                    } ${
                      col === 's_label' || col === 'target_label' ? 'max-w-32 break-words' : ''
                    } ${
                      col === 'target_study' ? 'max-w-24 break-words' : ''
                    } ${
                      col === 'mapping_relation' || col === 'harmonization_status' ? 'max-w-20 break-words text-xs' : ''
                    }`}
                  >
                    {displayValue}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* Modal to display merged EDA image or error */}
      {(selectedImage || imageError) && comparisonDetails && (
        <div className="modal modal-open">
          <div className="modal-box max-w-5xl">
            <h3 className="font-bold text-lg mb-4">
              {comparisonDetails 
                ? `Statistical Comparison: ${comparisonDetails.sourceVar} (${comparisonDetails.sourceCohort}) vs ${comparisonDetails.targetVar} (${comparisonDetails.targetCohort})`
                : 'EDA Comparison'
              }
            </h3>
            {imageError ? (
              <div className="alert alert-error">
                <div className="whitespace-pre-wrap">{imageError}</div>
              </div>
            ) : selectedImage ? (
              <img 
                src={selectedImage} 
                alt="Merged EDA comparison" 
                style={{ width: '75%', height: 'auto' }}
                className="mx-auto"
                onError={(e) => {
                  console.error('Image failed to load:', selectedImage);
                  setImageError('Image not found or failed to load');
                }}
                onLoad={() => console.log('Image loaded successfully')}
              />
            ) : null}
            <div className="modal-action">
              {!imageError && selectedImage && (
                <a 
                  href={selectedImage} 
                  download={comparisonDetails 
                    ? `comparison_${comparisonDetails.sourceVar}_${comparisonDetails.sourceCohort}_vs_${comparisonDetails.targetVar}_${comparisonDetails.targetCohort}.png`
                    : "eda-comparison.png"
                  }
                  className="btn btn-primary"
                >
                  Save Image
                </a>
              )}
              <button className="btn" onClick={() => { setSelectedImage(null); setImageError(null); setComparisonDetails(null); }}>Close</button>
            </div>
          </div>
          <div className="modal-backdrop" onClick={() => { setSelectedImage(null); setImageError(null); setComparisonDetails(null); }}></div>
        </div>
      )}
    </>
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
  const [sourceFilter, setSourceFilter] = useState('');
  const [targetFilter, setTargetFilter] = useState('');
  const [sourceDropdownOpen, setSourceDropdownOpen] = useState(false);
  const [selectedMappingTypes, setSelectedMappingTypes] = useState<string[]>([]);
  const [selectedHarmonizationStatuses, setSelectedHarmonizationStatuses] = useState<string[]>([]);
  const [cacheInfo, setCacheInfo] = useState<{
    cached_pairs: Array<{source: string, target: string, timestamp: number}>,
    uncached_pairs: Array<{source: string, target: string}>,
    outdated_pairs: Array<{source: string, target: string, timestamp: number, outdated_cohort: string}>,
    dictionary_timestamps: Record<string, number>
  } | null>(null);
  const [tableScrollWidth, setTableScrollWidth] = useState(2000);
  const [mappingStartTime, setMappingStartTime] = useState<number | null>(null);
  const [computeDuration, setComputeDuration] = useState<{minutes: number, seconds: number} | null>(null);
  const [hadUncachedPairs, setHadUncachedPairs] = useState(false);
  const [showCachePanel, setShowCachePanel] = useState(false);
  const [cachedFiles, setCachedFiles] = useState<any[]>([]);
  const [loadingCacheFiles, setLoadingCacheFiles] = useState(false);
  const [loadingCacheAction, setLoadingCacheAction] = useState<string | null>(null);
  
  // Reference to the mapping output section
  const mappingOutputRef = useRef<HTMLDivElement>(null);
  // Reference to the map button section
  const mapButtonRef = useRef<HTMLDivElement>(null);
  // Reference for source dropdown click-outside
  const sourceDropdownRef = useRef<HTMLDivElement>(null);

  // Close source dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (sourceDropdownRef.current && !sourceDropdownRef.current.contains(e.target as Node)) {
        setSourceDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Fetch cached mapping files for the modal
  const fetchCachedFiles = async () => {
    setLoadingCacheFiles(true);
    try {
      const cohortIds = Object.keys(cohortsData);
      const response = await fetch(`${apiUrl}/api/get-available-mapping-files`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cohortIds),
      });
      if (response.ok) {
        const result = await response.json();
        setCachedFiles(result.available_mappings || []);
      }
    } catch (error) {
      console.error('Failed to fetch cached mapping files:', error);
    } finally {
      setLoadingCacheFiles(false);
    }
  };

  // Handle preview of a cached mapping file
  const handleCachePreview = async (file: any) => {
    setLoadingCacheAction(file.filename);
    try {
      const response = await fetch(`${apiUrl}/api/get-cached-mapping-file/${encodeURIComponent(file.filename)}`, {
        credentials: 'include',
      });
      if (!response.ok) throw new Error('Failed to fetch file');
      const jsonData = await response.json();
      const previewData = transformMappingDataForPreview(jsonData);
      setMappingOutput(previewData);
      // Set the source cohort from the file's first cohort for the EDA comparison feature
      if (file.cohorts && file.cohorts.length > 0) {
        setSourceCohort(file.cohorts[0]);
      }
      setShowCachePanel(false);
    } catch (error) {
      console.error('Failed to preview cached file:', error);
    } finally {
      setLoadingCacheAction(null);
    }
  };

  // Handle download of a cached mapping file
  const handleCacheDownload = async (file: any) => {
    setLoadingCacheAction(file.filename);
    try {
      const response = await fetch(`${apiUrl}/api/get-cached-mapping-file/${encodeURIComponent(file.filename)}`, {
        credentials: 'include',
      });
      if (!response.ok) throw new Error('Failed to fetch file');
      const blob = await response.blob();
      const downloadFilename = response.headers.get('X-Filename') || file.filename;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = downloadFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      setShowCachePanel(false);
    } catch (error) {
      console.error('Failed to download cached file:', error);
    } finally {
      setLoadingCacheAction(null);
    }
  };

  // All cohorts sorted alphabetically (per-box filtering happens inline in JSX)
  const filteredCohorts = Object.entries(cohortsData)
    .sort(([idA], [idB]) => idA.localeCompare(idB));
  
  // Scroll to the map button section when data becomes available
  useEffect(() => {
    if (mappingOutput && mapButtonRef.current) {
      // Scroll to a few pixels above the map button
      const buttonTop = mapButtonRef.current.offsetTop - 20;
      window.scrollTo({ top: buttonTop, behavior: 'smooth' });
    }
  }, [mappingOutput]);

  // Update table scroll width when mapping output changes
  useEffect(() => {
    if (mappingOutput) {
      const bottomScroll = document.getElementById('bottom-scroll');
      if (bottomScroll) {
        const table = bottomScroll.querySelector('table');
        if (table) {
          setTableScrollWidth(table.scrollWidth);
        }
      }
    }
  }, [mappingOutput]);


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
    
    // Check if source cohort is among target cohorts
    if (selectedTargets.includes(sourceCohort)) {
      setError('Source cohort should not be among mapping target cohorts');
      return;
    }
    
    setLoading(true);
    setError(null);
    setMappingOutput(null);
    setCacheInfo(null);
    setComputeDuration(null);
    setMappingStartTime(Date.now());
    setCacheInfo(null);
    setHadUncachedPairs(false);
    
    // Declare timeout and controller outside try block for cleanup in catch
    const controller = new AbortController();
    let timeoutId: NodeJS.Timeout | null = null;
    
    try {
      // Send as [cohortId, false] for each selected target
      const target_studies = selectedTargets.map((cohortId: string) => [cohortId, false]);
      
      // First, check cache status immediately
      const cacheResponse = await fetch(`${apiUrl}/api/check-mapping-cache`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          source_study: sourceCohort,
          target_studies,
        }),
      });
      
      if (cacheResponse.ok) {
        const cacheData = await cacheResponse.json();
        setCacheInfo(cacheData);
        // Track if there are uncached pairs that will need computation
        setHadUncachedPairs(cacheData.uncached_pairs.length > 0 || (cacheData.outdated_pairs && cacheData.outdated_pairs.length > 0));
      }
      
      // Then proceed with mapping generation
      // Use AbortController with 35-minute timeout to prevent premature timeout
      // (new mappings can take up to 30 minutes to compute)
      timeoutId = setTimeout(() => controller.abort(), 35 * 60 * 1000); // 35 minutes
      
      const response = await fetch(`${apiUrl}/api/generate-mapping`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          source_study: sourceCohort,
          target_studies,
        }),
        signal: controller.signal,
      });
      if (timeoutId) clearTimeout(timeoutId);
      
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
      // Parse the JSON response
      const responseData = await response.json();
      
      // Cache info is already set from the initial cache check
      // No need to update it again from the response

      // Get file content and filename
      const fileContent = responseData.file_content;
      const filename = responseData.filename || `mapping_${sourceCohort}_to_${selectedTargets.join('_')}.json`;

      // The backend may incorrectly return NaN, which is not valid JSON.
      // Replace all instances of NaN with null before parsing.
      const cleanedFileContent = fileContent.replace(/NaN/g, 'null');

      // Handle download by creating a blob from the cleaned file content
      const blob = new Blob([cleanedFileContent], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      // Handle preview by parsing the cleaned file content
      try {
        const jsonData = JSON.parse(cleanedFileContent);
        const previewData = transformMappingDataForPreview(jsonData);
        setMappingOutput(previewData);
      } catch (error) {
        console.error('Error parsing JSON response for preview:', error);
        setMappingOutput([]); // Clear the preview on error
      }
      
      // Calculate compute duration (do this regardless of parse success)
      const startTime = mappingStartTime;
      if (startTime) {
        const durationMs = Date.now() - startTime;
        const totalSeconds = Math.floor(durationMs / 1000);
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        setComputeDuration({ minutes, seconds });
      }
      
      setLoading(false);
      // Refresh cached files list so the new mapping appears
      fetchCachedFiles();
    } catch (err: any) {
      if (timeoutId) clearTimeout(timeoutId);
      // Handle abort errors specifically
      if (err.name === 'AbortError') {
        setError('Request timed out after 35 minutes. The mapping computation may still be running on the server.');
      } else {
        setError(
          typeof err.message === 'string' && err.message.endsWith("metadata has not been added yet!")
            ? `The metadata of ${sourceCohort} has not been added yet!`
            : (err.message || 'Unknown error')
        );
      }
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
      <div className="w-full space-y-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-center mb-8">Cohort Mapping</h1>
        </div>

        {/* Cohort selection - constrained width */}
        <div className="max-w-6xl mx-auto">

        <div className="relative">
        <div className="flex gap-4 justify-center">
          {/* Source Cohort - searchable single-select */}
          <div className="form-control w-full max-w-sm">
            <label className="label">
              <span className="label-text">Source Cohort</span>
              {sourceCohort && (
                <span className="label-text-alt">
                  <button className="link link-error text-xs" onClick={() => { setSourceCohort(''); setSourceFilter(''); }}>clear</button>
                </span>
              )}
            </label>
            <input
              type="text"
              placeholder="Type to search..."
              className="input input-bordered input-sm w-full mb-1"
              value={sourceCohort || sourceFilter}
              onChange={(e) => { setSourceFilter(e.target.value); setSourceCohort(''); setSourceDropdownOpen(true); }}
              onFocus={() => { if (!sourceCohort) setSourceDropdownOpen(true); }}
              readOnly={!!sourceCohort}
            />
            <div className="flex flex-col max-h-48 overflow-y-auto border rounded p-2 bg-base-100" ref={sourceDropdownRef}>
              {filteredCohorts
                .filter(([id]) => id.toLowerCase().includes(sourceFilter.toLowerCase()))
                .map(([cohortId]) => (
                <label
                  key={cohortId}
                  className={`cursor-pointer flex items-center gap-2 py-0.5 text-sm ${cohortId === sourceCohort ? 'bg-primary/10 font-semibold' : ''}`}
                  onClick={() => { setSourceCohort(cohortId); setSourceFilter(''); setSourceDropdownOpen(false); }}
                >
                  <span>{cohortId}</span>
                </label>
              ))}
              {filteredCohorts.filter(([id]) => id.toLowerCase().includes(sourceFilter.toLowerCase())).length === 0 && (
                <span className="text-sm text-gray-400 py-1">No matches</span>
              )}
            </div>
          </div>

          {/* Target Cohorts - searchable multi-select */}
          <div className="form-control w-full max-w-sm">
            <label className="label">
              <span className="label-text">Target Cohorts</span>
              {selectedTargets.length > 0 && (
                <span className="label-text-alt">
                  <button className="link link-error text-xs" onClick={() => setSelectedTargets([])}>clear all</button>
                </span>
              )}
            </label>
            <input
              type="text"
              placeholder="Type to filter targets..."
              className="input input-bordered input-sm w-full mb-1"
              value={targetFilter}
              onChange={(e) => setTargetFilter(e.target.value)}
            />
            <div className="flex flex-col max-h-48 overflow-y-auto border rounded p-2 bg-base-100">
              {filteredCohorts
                .filter(([id]) => id.toLowerCase().includes(targetFilter.toLowerCase()))
                .map(([cohortId]) => {
                  const checked = selectedTargets.includes(cohortId);
                  return (
                    <label key={cohortId} className="cursor-pointer flex items-center gap-2 py-0.5 text-sm">
                      <input
                        type="checkbox"
                        className="checkbox checkbox-xs"
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
              {filteredCohorts.filter(([id]) => id.toLowerCase().includes(targetFilter.toLowerCase())).length === 0 && (
                <span className="text-sm text-gray-400 py-1">No matches</span>
              )}
            </div>
            {/* Selected targets chips */}
            {selectedTargets.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {selectedTargets.map(t => (
                  <span key={t} className="badge badge-sm badge-primary gap-1">
                    {t}
                    <button className="text-xs" onClick={() => setSelectedTargets(prev => prev.filter(x => x !== t))}>✕</button>
                  </span>
                ))}
              </div>
            )}
            {/* Show cached pairs link - aligned to right edge of target box */}
            <div className="flex justify-end mt-1">
              <button
                className="link link-primary text-xs italic"
                onClick={() => {
                  if (!showCachePanel) fetchCachedFiles();
                  setShowCachePanel(!showCachePanel);
                }}
              >
                {showCachePanel ? 'hide cached pairs' : 'show cached pairs'}
              </button>
            </div>
          </div>
        </div>

        {/* Cached pairs panel - overlays source/target boxes */}
        {showCachePanel && (
          <div className="absolute inset-x-0 top-10 z-10 p-4 border rounded-lg bg-base-100 shadow-lg flex flex-col overflow-auto" style={{minHeight: '400px', maxHeight: 'calc(100% - 2.5rem)' }}>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-semibold">Cached Pairs</span>
              <button
                className="btn btn-md btn-circle btn-ghost text-lg"
                onClick={() => setShowCachePanel(false)}
              >✕</button>
            </div>
            {loadingCacheFiles ? (
              <div className="flex items-center gap-2 py-4 justify-center">
                <span className="loading loading-spinner loading-sm"></span>
                <span className="text-sm">Loading cached pairs...</span>
              </div>
            ) : cachedFiles.filter(f => f.cohorts.length === 2).length === 0 ? (
              <p className="text-sm text-gray-500 text-center py-2">No cached mapping pairs found.</p>
            ) : (
              <div className="overflow-x-auto flex-1">
                <table className="table table-zebra table-sm w-full">
                  <thead>
                    <tr>
                      <th className="bg-base-300 whitespace-nowrap">Cohorts</th>
                      <th className="bg-base-300">Generated</th>
                      <th className="bg-base-300 text-center whitespace-nowrap">Source Vars</th>
                      <th className="bg-base-300 text-center whitespace-nowrap">Target Vars</th>
                      <th className="bg-base-300 text-center">Mappings</th>
                      <th className="bg-base-300">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cachedFiles
                      .filter(f => f.cohorts.length === 2)
                      .sort((a, b) => b.timestamp - a.timestamp)
                      .map((file, idx) => {
                        const srcId = file.cohorts[0];
                        const tgtId = file.cohorts[1];
                        // Case-insensitive lookup: cached file cohort names are lowercase
                        const srcKey = Object.keys(cohortsData).find(k => k.toLowerCase() === srcId.toLowerCase());
                        const tgtKey = Object.keys(cohortsData).find(k => k.toLowerCase() === tgtId.toLowerCase());
                        const srcVarCount = srcKey && cohortsData[srcKey] ? Object.keys(cohortsData[srcKey].variables || {}).length : null;
                        const tgtVarCount = tgtKey && cohortsData[tgtKey] ? Object.keys(cohortsData[tgtKey].variables || {}).length : null;
                        return (
                      <tr key={idx}>
                        <td className="font-medium text-xs whitespace-nowrap">
                          {file.cohorts.join(' → ')}
                        </td>
                        <td className="text-xs whitespace-nowrap">
                          {new Date(file.timestamp * 1000).toLocaleDateString('de-DE')}{' '}
                          <span className="text-gray-400">
                            {new Date(file.timestamp * 1000).toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' })}
                          </span>
                        </td>
                        <td className="text-xs text-center">
                          {srcVarCount ?? '—'}
                        </td>
                        <td className="text-xs text-center">
                          {tgtVarCount ?? '—'}
                        </td>
                        <td className="text-xs text-center">
                          {file.stats?.total_mappings ?? '—'}
                        </td>
                        <td className="text-xs">
                          <div className="flex gap-1">
                            <button
                              className="btn btn-xs btn-outline"
                              onClick={() => handleCachePreview(file)}
                              disabled={loadingCacheAction === file.filename}
                            >
                              {loadingCacheAction === file.filename ? <span className="loading loading-spinner loading-xs"></span> : 'Preview'}
                            </button>
                            <button
                              className="btn btn-xs btn-outline"
                              onClick={() => handleCacheDownload(file)}
                              disabled={loadingCacheAction === file.filename}
                            >
                              Download
                            </button>
                          </div>
                        </td>
                      </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
        </div>

        <div className="text-center mt-8" ref={mapButtonRef}>
          <button
            className="btn btn-primary"
            onClick={handleMapConcepts}
            disabled={!sourceCohort || selectedTargets.length === 0 || loading}
          >
{loading 
              ? (cacheInfo && cacheInfo.uncached_pairs.length > 0 
                  ? 'Finding concept mappings... (may take up to 30 minutes)' 
                  : 'Mapping...')
              : 'Map Concepts & Download File'
            }
          </button>
        </div>
        </div>

        {/* Success Message - only shown after mapping completes for uncached pairs */}
        <div className="max-w-6xl mx-auto">
        {mappingOutput && mappingOutput.length > 0 && computeDuration && hadUncachedPairs && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-start gap-2">
              <span className="text-green-600 text-xl">✓</span>
              <div>
                <h4 className="font-semibold text-green-800 mb-1">Mapping Completed</h4>
                <p className="text-sm text-green-700">
                  Variable mapping for <strong>{sourceCohort}</strong> → <strong>{selectedTargets.join(', ')}</strong> has been generated.
                </p>
                <p className="text-sm text-green-600 mt-1">
                  Compute time: {computeDuration.minutes > 0 ? `${computeDuration.minutes} minute${computeDuration.minutes !== 1 ? 's' : ''} ` : ''}{computeDuration.seconds} second{computeDuration.seconds !== 1 ? 's' : ''}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Cache Information Display - shown after mapping completes for cached pairs */}
        {cacheInfo && mappingOutput && !hadUncachedPairs && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-semibold mb-2">Cache Status:</h4>
            
            {/* Up-to-date cached pairs */}
            {cacheInfo.cached_pairs.length > 0 && (
              <div className="mb-2">
                <span className="text-green-600 font-medium">Cached pairs (up to date):</span>
                <ul className="ml-4 mt-1">
                  {cacheInfo.cached_pairs.map((pair, index) => (
                    <li key={index} className="text-sm">
                      {pair.source} → {pair.target} 
                      <span className="text-gray-500 ml-2">
                        (cached {new Date(pair.timestamp * 1000).toLocaleDateString('de-DE')})
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {cacheInfo.cached_pairs.length > 0 && (
              <div className="mt-3 p-2 bg-green-100 rounded text-sm text-green-800">
                ✅ All cached mappings are up to date with the latest dictionaries
              </div>
            )}
          </div>
        )}

        {/* Cache Information Display - only shown before mapping completes */}
        {cacheInfo && !mappingOutput && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-semibold mb-2">Cache Status:</h4>
            
            <ul className="space-y-1">
              {/* Cached pairs (up to date) */}
              {cacheInfo.cached_pairs.map((pair, index) => (
                <li key={`cached-${index}`} className="text-sm">
                  {pair.source} → {pair.target} 
                  <span className="text-green-600 ml-2 font-medium">
                    — Cached and up-to-date, compute date: {new Date(pair.timestamp * 1000).toLocaleDateString('de-DE')}
                  </span>
                </li>
              ))}
              
              {/* Outdated cached pairs */}
              {cacheInfo.outdated_pairs && cacheInfo.outdated_pairs.map((pair, index) => (
                <li key={`outdated-${index}`} className="text-sm">
                  {pair.source} → {pair.target} 
                  <span className="text-orange-600 ml-2 font-medium">
                    — Cached but outdated, compute date: {new Date(pair.timestamp * 1000).toLocaleDateString('de-DE')}
                  </span>
                </li>
              ))}
              
              {/* Uncached pairs */}
              {cacheInfo.uncached_pairs.map((pair, index) => (
                <li key={`uncached-${index}`} className="text-sm">
                  {pair.source} → {pair.target} 
                  <span className="text-blue-600 ml-2 font-medium">
                    — New (uncached)
                  </span>
                </li>
              ))}
            </ul>
            
            {/* Summary message */}
            {(cacheInfo.uncached_pairs.length > 0 || (cacheInfo.outdated_pairs && cacheInfo.outdated_pairs.length > 0)) && (
              <div className="mt-3 p-2 bg-blue-100 rounded text-sm text-blue-800">
                ⏳ Uncached and outdated mappings will be computed. This may take up to 30 minutes. If this page times out, please revisit later. The computed mappings will be cached.
              </div>
            )}
            
            {/* Dictionary timestamps at the bottom */}
            {Object.keys(cacheInfo.dictionary_timestamps).length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-200">
                <span className="text-gray-700 font-medium">Dictionary dates:</span>
                <ul className="ml-4 mt-1">
                  {Object.entries(cacheInfo.dictionary_timestamps).map(([cohort, timestamp]) => (
                    <li key={cohort} className="text-sm">
                      {cohort}: {new Date(timestamp * 1000).toLocaleDateString('de-DE')}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="mt-4 text-red-500 text-center">{error}</div>
        )}
        </div>

        {/* Mapping Preview - wider container, breaks out of max-w-6xl constraint */}
        {mappingOutput && (
          <div 
            ref={mappingOutputRef}
            className="mt-4 p-4 border rounded-lg bg-base-100 w-[85vw] mx-auto"
          >
            <h2 className="text-lg font-bold mb-3">Mapping Preview</h2>
            
            {/* Filter Controls - Moved to top */}
            <div className="flex justify-end mb-4">
              <div className="bg-gray-50 border rounded-lg p-4 w-full max-w-2xl">
                <h4 className="font-semibold text-sm mb-3">Filters</h4>
                <div className="grid grid-cols-2 gap-4">
                  {/* Mapping Relation Filter */}
                  <div>
                    <h5 className="font-medium text-xs mb-2">mapping_relation</h5>
                    <div className="space-y-1 max-h-32 overflow-y-auto text-xs">
                      {(() => {
                        const mappingRelationCounts = mappingOutput.reduce((acc, row) => {
                          const value = (row.mapping_relation?.toString() || '--');
                          const currentCount = acc[value] as number || 0;
                          acc[value] = currentCount + 1;
                          return acc;
                        }, {} as Record<string, number>);
                        
                        return Object.entries(mappingRelationCounts).map(([value, count]) => (
                          <div key={value} className="flex items-center gap-2">
                            <input 
                              type="checkbox" 
                              className="checkbox checkbox-xs" 
                              id={`mapping-${value}`}
                              checked={selectedMappingTypes.includes(value)}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setSelectedMappingTypes(prev => [...prev, value]);
                                } else {
                                  setSelectedMappingTypes(prev => prev.filter(v => v !== value));
                                }
                              }}
                            />
                            <label htmlFor={`mapping-${value}`} className="cursor-pointer truncate">
                              {value} ({count})
                            </label>
                          </div>
                        ));
                      })()}
                    </div>
                  </div>

                  {/* Harmonization Status Filter */}
                  <div>
                    <h5 className="font-medium text-xs mb-2">harmonization_status</h5>
                    <div className="space-y-1 max-h-32 overflow-y-auto text-xs">
                      {(() => {
                        const statusCounts = mappingOutput.reduce((acc, row) => {
                          const value = (row.harmonization_status?.toString() || '--');
                          const currentCount = acc[value] as number || 0;
                          acc[value] = currentCount + 1;
                          return acc;
                        }, {} as Record<string, number>);
                        
                        return Object.entries(statusCounts).map(([value, count]) => (
                          <div key={value} className="flex items-center gap-2">
                            <input 
                              type="checkbox" 
                              className="checkbox checkbox-xs" 
                              id={`status-${value}`}
                              checked={selectedHarmonizationStatuses.includes(value)}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setSelectedHarmonizationStatuses(prev => [...prev, value]);
                                } else {
                                  setSelectedHarmonizationStatuses(prev => prev.filter(v => v !== value));
                                }
                              }}
                            />
                            <label htmlFor={`status-${value}`} className="cursor-pointer truncate">
                              {value} ({count})
                            </label>
                          </div>
                        ));
                      })()}
                    </div>
                  </div>
                </div>
                
                {/* Clear Filters Button */}
                {(selectedMappingTypes.length > 0 || selectedHarmonizationStatuses.length > 0) && (
                  <button
                    className="btn btn-xs btn-outline mt-3 w-full"
                    onClick={() => {
                      setSelectedMappingTypes([]);
                      setSelectedHarmonizationStatuses([]);
                    }}
                  >
                    Clear Filters
                  </button>
                )}
              </div>
            </div>

            {/* Row count and target info */}
            {(() => {
              // Filter the data based on selected filters
              const filteredData = mappingOutput.filter(row => {
                const mappingRelation = (row.mapping_relation?.toString() || '--');
                const harmonizationStatus = (row.harmonization_status?.toString() || '--');
                
                const mappingRelationMatch = selectedMappingTypes.length === 0 || selectedMappingTypes.includes(mappingRelation);
                const harmonizationStatusMatch = selectedHarmonizationStatuses.length === 0 || selectedHarmonizationStatuses.includes(harmonizationStatus);
                
                return mappingRelationMatch && harmonizationStatusMatch;
              });

              return (
                <div className="text-xs text-gray-500 mb-3">
                  <p>{filteredData.length} rows {filteredData.length !== mappingOutput.length && `(filtered from ${mappingOutput.length})`}</p>
                  {(() => {
                    // Calculate mappings per target cohort for filtered data
                    const targetCounts: Record<string, number> = {};
                    filteredData.forEach(row => {
                      const targetStudy = row.target_study as string;
                      if (targetStudy) {
                        targetCounts[targetStudy] = (targetCounts[targetStudy] || 0) + 1;
                      }
                    });
                    return (
                      <p>
                        Mappings per target: {Object.entries(targetCounts)
                          .map(([target, count]) => `${target} (${count})`)
                          .join(', ')}
                      </p>
                    );
                  })()}
                </div>
              );
            })()}

            {/* Top horizontal scrollbar with arrow buttons */}
            <div className="flex items-center gap-1 mb-2">
              <button
                className="btn btn-xs btn-square"
                onClick={() => {
                  const topScroll = document.getElementById('top-scroll');
                  const bottomScroll = document.getElementById('bottom-scroll');
                  if (topScroll && bottomScroll) {
                    const scrollAmount = 200;
                    topScroll.scrollLeft -= scrollAmount;
                    bottomScroll.scrollLeft -= scrollAmount;
                  }
                }}
                title="Scroll left"
              >
                ←
              </button>
              <div 
                id="top-scroll"
                className="overflow-x-scroll flex-1 border border-gray-200 rounded"
                style={{
                  height: '20px',
                  overflowY: 'hidden',
                  scrollbarWidth: 'auto',
                  scrollbarColor: '#94a3b8 #e2e8f0'
                }}
                onScroll={(e) => {
                  const bottomScroll = document.getElementById('bottom-scroll');
                  if (bottomScroll) {
                    bottomScroll.scrollLeft = e.currentTarget.scrollLeft;
                  }
                }}
              >
                <div style={{ width: `${tableScrollWidth}px`, height: '1px' }}></div>
              </div>
              <button
                className="btn btn-xs btn-square"
                onClick={() => {
                  const topScroll = document.getElementById('top-scroll');
                  const bottomScroll = document.getElementById('bottom-scroll');
                  if (topScroll && bottomScroll) {
                    const scrollAmount = 200;
                    topScroll.scrollLeft += scrollAmount;
                    bottomScroll.scrollLeft += scrollAmount;
                  }
                }}
                title="Scroll right"
              >
                →
              </button>
            </div>
            
            {/* Main table with scrollbars */}
            <div 
              id="bottom-scroll"
              className="overflow-x-scroll overflow-y-scroll max-h-[600px] border border-gray-200 rounded-b" 
              style={{
                scrollbarWidth: 'auto',
                scrollbarColor: '#94a3b8 #e2e8f0',
                overflowX: 'scroll',
                overflowY: 'scroll'
              }}
              onScroll={(e) => {
                const topScroll = e.currentTarget.previousElementSibling as HTMLElement;
                if (topScroll) {
                  topScroll.scrollLeft = e.currentTarget.scrollLeft;
                }
              }}
            >
              {(() => {
                // Calculate filtered data for the table
                const filteredData = mappingOutput.filter(row => {
                  const mappingRelation = (row.mapping_relation?.toString() || '--');
                  const harmonizationStatus = (row.harmonization_status?.toString() || '--');
                  
                  const mappingRelationMatch = selectedMappingTypes.length === 0 || selectedMappingTypes.includes(mappingRelation);
                  const harmonizationStatusMatch = selectedHarmonizationStatuses.length === 0 || selectedHarmonizationStatuses.includes(harmonizationStatus);
                  
                  return mappingRelationMatch && harmonizationStatusMatch;
                });
                
                return <MappingPreviewJsonTable data={filteredData} sourceCohort={sourceCohort} />;
              })()}
            </div>
          </div>
        )}
      </div>

    </main>
  );
}