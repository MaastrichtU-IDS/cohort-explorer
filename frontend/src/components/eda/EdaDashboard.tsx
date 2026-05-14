import React, { useState, useEffect, useMemo } from 'react';
import { parseEdaJson, EdaData } from '@/utils/edaParsing';
import EdaOverviewPanel from './EdaOverviewPanel';
import EdaNumericBoxPlots from './EdaNumericBoxPlots';
import EdaCategoricalCharts from './EdaCategoricalCharts';
import EdaDataQualityMatrix from './EdaDataQualityMatrix';
import EdaTimePointComparison from './EdaTimePointComparison';
import EdaStatisticsTable from './EdaStatisticsTable';
import EdaOutlierAnalysis from './EdaOutlierAnalysis';
import EdaLongitudinalRanking from './EdaLongitudinalRanking';
import EdaVariableDetailModal from './EdaVariableDetailModal';
import { EdaVariable } from '@/utils/edaParsing';

interface EdaDashboardProps {
  cohortId: string;
}

type EdaSection = 'overview' | 'numeric' | 'categorical' | 'quality' | 'timepoints' | 'longitudinal' | 'table' | 'outliers';

const SECTIONS: { key: EdaSection; label: string; icon: string }[] = [
  { key: 'overview', label: 'Overview', icon: '📊' },
  { key: 'numeric', label: 'Numeric Distributions', icon: '📈' },
  { key: 'categorical', label: 'Categorical Distributions', icon: '📉' },
  { key: 'quality', label: 'Data Quality Matrix', icon: '🔍' },
  { key: 'timepoints', label: 'Time-Point Comparison', icon: '⏱️' },
  { key: 'longitudinal', label: 'Longitudinal Ranking', icon: '📐' },
  { key: 'outliers', label: 'Outlier Analysis', icon: '🎯' },
  { key: 'table', label: 'Statistics Table', icon: '📋' },
];

const EdaDashboard: React.FC<EdaDashboardProps> = ({ cohortId }) => {
  const [rawData, setRawData] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<EdaSection>('overview');
  const [selectedVariable, setSelectedVariable] = useState<EdaVariable | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(`/api/cohort-eda-output/${encodeURIComponent(cohortId)}`, { credentials: 'include' })
      .then(res => {
        if (!res.ok) throw new Error(res.status === 404 ? 'No EDA data available for this cohort' : `Error ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (!cancelled) {
          setRawData(data);
          setLoading(false);
        }
      })
      .catch(err => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [cohortId]);

  const edaData: EdaData | null = useMemo(() => {
    if (!rawData) return null;
    return parseEdaJson(rawData);
  }, [rawData]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-16">
        <span className="loading loading-spinner loading-lg mb-4"></span>
        <p className="text-gray-500">Loading EDA data for <strong>{cohortId}</strong>...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alert alert-warning shadow-lg max-w-xl mx-auto my-8">
        <div>
          <span className="text-lg">⚠️</span>
          <div>
            <h3 className="font-bold">EDA Data Not Available</h3>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!edaData) return null;

  const { variables, numericVars, categoricalVars, dateVars, timePointGroups } = edaData;

  return (
    <div className="space-y-4">
      {/* Header stats bar */}
      <div className="stats shadow w-full bg-base-100">
        <div className="stat">
          <div className="stat-title">Total Variables</div>
          <div className="stat-value text-primary">{variables.length}</div>
        </div>
        <div className="stat">
          <div className="stat-title">Numeric</div>
          <div className="stat-value text-blue-500">{numericVars.length}</div>
        </div>
        <div className="stat">
          <div className="stat-title">Categorical</div>
          <div className="stat-value text-green-500">{categoricalVars.length}</div>
        </div>
        {dateVars.length > 0 && (
          <div className="stat">
            <div className="stat-title">Date</div>
            <div className="stat-value text-amber-500">{dateVars.length}</div>
          </div>
        )}
        <div className="stat">
          <div className="stat-title">Longitudinal Variables</div>
          <div className="stat-value text-purple-500">{timePointGroups.length}</div>
        </div>
      </div>

      {/* Section navigation */}
      <div className="tabs tabs-boxed bg-base-200 p-1 flex flex-wrap gap-1">
        {SECTIONS.map(({ key, label, icon }) => {
          // Hide sections with no data
          if (key === 'numeric' && numericVars.length === 0) return null;
          if (key === 'categorical' && categoricalVars.length === 0) return null;
          if (key === 'timepoints' && timePointGroups.length === 0) return null;
          if (key === 'longitudinal' && timePointGroups.length === 0) return null;
          if (key === 'outliers' && numericVars.length === 0) return null;
          return (
            <button
              key={key}
              className={`tab tab-lg ${activeSection === key ? 'tab-active font-bold' : ''}`}
              onClick={() => setActiveSection(key)}
            >
              {icon} {label}
            </button>
          );
        })}
      </div>

      {/* Section content */}
      <div className="min-h-[400px]">
        {activeSection === 'overview' && (
          <EdaOverviewPanel edaData={edaData} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'numeric' && (
          <EdaNumericBoxPlots variables={numericVars} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'categorical' && (
          <EdaCategoricalCharts variables={categoricalVars} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'quality' && (
          <EdaDataQualityMatrix variables={variables} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'timepoints' && (
          <EdaTimePointComparison groups={timePointGroups} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'longitudinal' && (
          <EdaLongitudinalRanking groups={timePointGroups} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'outliers' && (
          <EdaOutlierAnalysis variables={numericVars} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'table' && (
          <EdaStatisticsTable variables={variables} onVariableClick={setSelectedVariable} />
        )}
      </div>

      {/* Detail modal */}
      {selectedVariable && (
        <EdaVariableDetailModal
          variable={selectedVariable}
          onClose={() => setSelectedVariable(null)}
          cohortId={cohortId}
        />
      )}
    </div>
  );
};

export default EdaDashboard;
