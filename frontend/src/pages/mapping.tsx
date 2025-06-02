'use client';

import React, { useState } from 'react';
import { useCohorts } from '@/components/CohortsContext';

export default function MappingPage() {
  const { cohortsData, userEmail } = useCohorts();
  const [sourceCohort, setSourceCohort] = useState('');
  const [targetCohort, setTargetCohort] = useState('');
  const [mappingOutput, setMappingOutput] = useState('');

  // Function to generate sample CSV content
  const generateSampleCSV = () => {
    const sampleData = [
      'Source Variable,Target Variable,Mapping Type,Confidence',
      'age,patient_age,EXACT,0.95',
      'gender,sex,SYNONYM,0.85',
      'weight_kg,weight,EXACT,0.92',
      'height_cm,height,EXACT,0.90',
      'systolic_bp,blood_pressure_systolic,RELATED,0.75',
      'diastolic_bp,blood_pressure_diastolic,RELATED,0.75',
      'smoking_status,smoking_history,SYNONYM,0.82',
      'diabetes,diabetes_mellitus,EXACT,0.95',
      'heart_disease,cardiovascular_disease,BROADER,0.70',
      'medication_name,drug_name,SYNONYM,0.88'
    ].join('\n');
    return sampleData;
  };

  const handleMapConcepts = () => {
    if (!sourceCohort || !targetCohort) {
      alert('Please select both cohorts');
      return;
    }

    // Generate sample mapping output
    const csvContent = generateSampleCSV();
    setMappingOutput(csvContent);

    // Create and trigger download
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mapping_${sourceCohort}_to_${targetCohort}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
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
      <div className="w-full max-w-4xl space-y-8">
        <h1 className="text-3xl font-bold text-center mb-8">Cohort Mapping</h1>
        
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
              {Object.entries(cohortsData).map(([cohortId, cohort]) => (
                <option key={cohortId} value={cohortId}>
                  {cohortId}
                </option>
              ))}
            </select>
          </div>

          <div className="form-control w-full max-w-xs">
            <label className="label">
              <span className="label-text">Target Cohort</span>
            </label>
            <select 
              className="select select-bordered w-full"
              value={targetCohort}
              onChange={(e) => setTargetCohort(e.target.value)}
            >
              <option value="">Select target cohort</option>
              {Object.entries(cohortsData).map(([cohortId, cohort]) => (
                <option key={cohortId} value={cohortId}>
                  {cohortId}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex justify-center mt-6">
          <button 
            className="btn btn-primary"
            onClick={handleMapConcepts}
            disabled={!sourceCohort || !targetCohort}
          >
            Map Concepts
          </button>
        </div>

        {mappingOutput && (
          <div className="mt-8">
            <h2 className="text-xl font-semibold mb-4">Mapping Preview</h2>
            <div className="bg-base-100 p-4 rounded-lg shadow overflow-x-auto">
              <pre className="whitespace-pre-wrap font-mono text-sm">
                {mappingOutput.split('\n').slice(0, 5).join('\n')}
                {mappingOutput.split('\n').length > 5 && '\n...'}
              </pre>
            </div>
          </div>
        )}
      </div>
    </main>
  );
} 