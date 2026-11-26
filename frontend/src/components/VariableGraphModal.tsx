import React, { useState, useEffect } from 'react';

interface VariableGraphModalProps {
  isOpen: boolean;
  cohortId: string;
  variableName: string;
  variableLabel?: string;
  omopId?: string;
  conceptCode?: string;
  onClose: () => void; // Add this new prop
}

const VariableGraphModal: React.FC<VariableGraphModalProps> = ({
  isOpen,
  cohortId,
  variableName,
  variableLabel,
  omopId,
  conceptCode,
  onClose // Use this prop to handle closing
}) => {
  const [imageState, setImageState] = useState<'loading' | 'error' | 'loaded'>('loading');

  useEffect(() => {
    if (isOpen) {
      setImageState('loading');
    }
  }, [isOpen, cohortId, variableName]);

  if (!isOpen) return null;

  const isPatientIdVariable = omopId === '4086934' || conceptCode === 'snomed:184107009';

  const imageUrl = `/api/variable-graph/${encodeURIComponent(cohortId)}/${encodeURIComponent(variableName.toLowerCase())}`;

  return (
    <div className="modal modal-open">
      {/* The width it always had. The height may reach the window's less 1rem,
          about four lines more than a modal box's default (100vh - 5em), which
          fell a few lines short of the profiling charts; a chart taller still
          is scaled down to the room under the header rather than scrolled. */}
      <div className="modal-box space-y-2 max-w-5xl max-h-[calc(100vh-1rem)] overflow-hidden">
        {/* Add a header with the close button */}
        <div className="flex justify-between items-center">
          <h3 className="font-bold text-lg">{variableLabel || variableName} Distribution</h3>
          <button 
            onClick={onClose}
            className="btn btn-sm btn-circle"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        
        <div className="flex justify-center items-center min-h-[200px]">
          {isPatientIdVariable ? (
            <div className="text-center p-4">
              <p>This is the patient ID variable, and is not included in the visualization.</p>
            </div>
          ) : (
            <>
              {imageState === 'loading' && (
                <div className="absolute">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
                </div>
              )}
              
              {imageState !== 'error' && (
                <img
                  key={`${cohortId}-${variableName}`}
                  src={imageUrl}
                  alt={`${variableName} distribution graph`}
                  style={{maxHeight: 'calc(100vh - 7rem)'}}
                  className={`max-w-full w-auto h-auto object-contain mx-auto transition-opacity duration-300 ${
                    imageState === 'loaded' ? 'opacity-100' : 'opacity-0'
                  }`}
                  onError={() => setImageState('error')}
                  onLoad={() => setImageState('loaded')}
                  crossOrigin="use-credentials"
                />
              )}

              {imageState === 'error' && (
                <div className="text-center p-4">
                  <p>No graph available for this variable, or the computation for the cohort has not been run yet.</p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
      
      {/* Backdrop that also closes the modal when clicked */}
      <div className="modal-backdrop" onClick={onClose}></div>
    </div>
  );
};

export default VariableGraphModal;