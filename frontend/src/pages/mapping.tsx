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
              <th key={col} className="font-bold bg-base-300">{columnDisplayNames[col] || col}</th>
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
                  
                  const handleCompare = () => {
                    if (sourceCohort && sourceVar && targetStudy && targetVar) {
                      // Use Next.js API route instead of direct backend call
                      const imageUrl = `/api/compare-eda/${encodeURIComponent(sourceCohort)}/${encodeURIComponent(sourceVar)}/${encodeURIComponent(targetStudy)}/${encodeURIComponent(targetVar)}`;
                      console.log('Compare EDA clicked:', { sourceCohort, sourceVar, targetStudy, targetVar, imageUrl });
                      setImageError(null);
                      setSelectedImage(imageUrl);
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
      
      {/* Modal to display merged EDA image */}
      {selectedImage && (
        <div className="modal modal-open">
          <div className="modal-box max-w-5xl">
            <h3 className="font-bold text-lg mb-4">EDA Comparison</h3>
            {imageError ? (
              <div className="alert alert-error">
                <span>Failed to load image: {imageError}</span>
              </div>
            ) : (
              <img 
                src={selectedImage} 
                alt="Merged EDA comparison" 
                className="w-full"
                onError={(e) => {
                  console.error('Image failed to load:', selectedImage);
                  setImageError('Image not found or failed to load');
                }}
                onLoad={() => console.log('Image loaded successfully')}
              />
            )}
            <div className="modal-action">
              {!imageError && (
                <a 
                  href={selectedImage} 
                  download="eda-comparison.png"
                  className="btn btn-primary"
                >
                  Save Image
                </a>
              )}
              <button className="btn" onClick={() => { setSelectedImage(null); setImageError(null); }}>Close</button>
            </div>
          </div>
          <div className="modal-backdrop" onClick={() => { setSelectedImage(null); setImageError(null); }}></div>
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
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedMappingTypes, setSelectedMappingTypes] = useState<string[]>([]);
  const [selectedHarmonizationStatuses, setSelectedHarmonizationStatuses] = useState<string[]>([]);
  const [cacheInfo, setCacheInfo] = useState<{
    cached_pairs: Array<{source: string, target: string, timestamp: number}>,
    uncached_pairs: Array<{source: string, target: string}>,
    outdated_pairs: Array<{source: string, target: string, timestamp: number, outdated_cohort: string}>,
    dictionary_timestamps: Record<string, number>
  } | null>(null);
  
  // Reference to the mapping output section
  const mappingOutputRef = useRef<HTMLDivElement>(null);
  // Reference to the map button section
  const mapButtonRef = useRef<HTMLDivElement>(null);

  // Filtered cohorts for both source and target menus based on search
  const filteredCohorts = Object.entries(cohortsData).filter(([cohortId, cohort]) =>
    cohortId.toLowerCase().includes(searchQuery.toLowerCase()) ||
    JSON.stringify(cohort).toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  // Scroll to the map button section when data becomes available
  useEffect(() => {
    if (mappingOutput && mapButtonRef.current) {
      // Scroll to a few pixels above the map button
      const buttonTop = mapButtonRef.current.offsetTop - 20;
      window.scrollTo({ top: buttonTop, behavior: 'smooth' });
    }
  }, [mappingOutput]);

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
    setCacheInfo(null);
    
    try {
      // Send as [cohortId, false] for each selected target
      const target_studies = selectedTargets.map((cohortId: string) => [cohortId, false]);
      
      // First, check cache status immediately
      const cacheResponse = await fetch(`${apiUrl}/api/check-mapping-cache`, {
        method: 'POST',
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
      }
      
      // Then proceed with mapping generation
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

        <div className="text-center" ref={mapButtonRef}>
          <button
            className="btn btn-primary"
            onClick={handleMapConcepts}
            disabled={!sourceCohort || selectedTargets.length === 0 || loading}
          >
{loading 
              ? (cacheInfo && cacheInfo.uncached_pairs.length > 0 
                  ? 'Mapping... (will take several minutes)' 
                  : 'Mapping...')
              : 'Map Concepts & Download File'
            }
          </button>
        </div>

        {/* Cache Information Display */}
        {cacheInfo && (
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
            
            {/* Outdated cached pairs */}
            {cacheInfo.outdated_pairs && cacheInfo.outdated_pairs.length > 0 && (
              <div className="mb-2">
                <span className="text-orange-600 font-medium">Outdated cached pairs:</span>
                <ul className="ml-4 mt-1">
                  {cacheInfo.outdated_pairs.map((pair, index) => (
                    <li key={index} className="text-sm">
                      {pair.source} → {pair.target} 
                      <span className="text-gray-500 ml-2">
                        (cached {new Date(pair.timestamp * 1000).toLocaleDateString('de-DE')})
                      </span>
                      <br />
                      <span className="text-orange-600 text-xs ml-2">
                        ⚠️ Cached mapping is out of date. There is an updated dictionary for cohort {pair.outdated_cohort}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* New mappings */}
            {cacheInfo.uncached_pairs.length > 0 && (
              <div className="mb-2">
                <span className="text-blue-600 font-medium">New mappings:</span>
                <ul className="ml-4 mt-1">
                  {cacheInfo.uncached_pairs.map((pair, index) => (
                    <li key={index} className="text-sm">
                      {pair.source} → {pair.target}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* Summary message */}
            {(cacheInfo.uncached_pairs.length > 0 || (cacheInfo.outdated_pairs && cacheInfo.outdated_pairs.length > 0)) ? (
              <div className="mt-3 p-2 bg-blue-100 rounded text-sm text-blue-800">
                ⏳ Uncached and outdated mappings will be computed. This may take up to 15 minutes. If this page times out, please revisit in 15-20 minutes when computed mappings are likely to be ready
              </div>
            ) : (
              cacheInfo.cached_pairs.length > 0 && (
                <div className="mt-3 p-2 bg-green-100 rounded text-sm text-green-800">
                  ✅ All cached mappings are up to date with the latest dictionaries
                </div>
              )
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

        {mappingOutput && (
          <div 
            ref={mappingOutputRef}
            className="mt-4 p-4 border rounded-lg bg-base-100 w-full max-w-[110rem] mx-auto"
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

            <div className="overflow-x-auto">
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