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
import EdaLongitudinalClassBalance from './EdaLongitudinalClassBalance';
import EdaSkewnessIqrScatter from './EdaSkewnessIqrScatter';
import EdaCoverageRanking from './EdaCoverageRanking';
import EdaOutlierRanking from './EdaOutlierRanking';
import EdaCVRanking from './EdaCVRanking';
import EdaImbalanceRanking from './EdaImbalanceRanking';
import EdaVariableDetailModal from './EdaVariableDetailModal';
import { EdaVariable } from '@/utils/edaParsing';

interface EdaDashboardProps {
  cohortId: string;
}

type EdaSection = 'longitudinal' | 'longClassBalance' | 'longOutliersZ' | 'longOutliersIqr' | 'longIqr' | 'skewnessIqr' | 'coverage' | 'outlierRank' | 'cvRank' | 'imbalanceRank' | 'table' | 'outliersIqr' | 'outliersZ';

const LONGITUDINAL_SECTIONS = [
  { key: 'longitudinal' as const, label: 'Change in Mean Ranking', icon: '' },
  { key: 'longClassBalance' as const, label: 'Change in Class Balance', icon: '' },
  { key: 'longOutliersZ' as const, label: 'Change in Outliers (Z)', icon: '' },
  { key: 'longOutliersIqr' as const, label: 'Change in Outliers (IQR)', icon: '' },
  { key: 'longIqr' as const, label: 'Change in IQR Ranking', icon: '' },
];

const ALL_VARIABLES_SECTIONS = [
  { key: 'cvRank' as const, label: 'Variance Ranking (numeric vars)', icon: '' },
  { key: 'imbalanceRank' as const, label: 'Imbalance Ranking (categorical vars)', icon: '' },
  { key: 'coverage' as const, label: 'Coverage Ranking', icon: '' },
  { key: 'outlierRank' as const, label: 'Outlier Ranking', icon: '' },
  { key: 'skewnessIqr' as const, label: 'Skewness vs IQR', icon: '' },
  { key: 'outliersIqr' as const, label: 'Count of Outliers (IQR)', icon: '' },
  { key: 'outliersZ' as const, label: 'Count of Outliers (Z-scores)', icon: '' },
  { key: 'table' as const, label: 'Statistics Table', icon: '' },
];

const EdaDashboard: React.FC<EdaDashboardProps> = ({ cohortId }) => {
  const [rawData, setRawData] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<EdaSection>('cvRank');
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
      <div className="space-y-2">
        <div className="text-sm font-semibold text-gray-600">All Variables</div>
        <div className="tabs tabs-boxed bg-base-200 p-1 flex flex-wrap gap-1">
          {ALL_VARIABLES_SECTIONS.map(({ key, label, icon }) => {
            if (key === 'skewnessIqr' && numericVars.length === 0) return null;
            if (key === 'outlierRank' && numericVars.length === 0) return null;
            if (key === 'cvRank' && numericVars.length === 0) return null;
            if (key === 'imbalanceRank' && categoricalVars.length === 0) return null;
            if (key === 'outliersIqr' && numericVars.length === 0) return null;
            if (key === 'outliersZ' && numericVars.length === 0) return null;
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
        {timePointGroups.length > 2 && (
          <>
            <div className="text-sm font-semibold text-gray-600">Longitudinal Variables</div>
            <div className="tabs tabs-boxed bg-base-200 p-1 flex flex-wrap gap-1">
              {LONGITUDINAL_SECTIONS.map(({ key, label, icon }) => {
                const categoricalLongitudinalVars = timePointGroups.flatMap(g => g.variables.filter(m => m.variable.type === 'categorical' && m.variable.classBalance));
                if (key === 'longClassBalance' && categoricalLongitudinalVars.length === 0) return null;
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
          </>
        )}
      </div>

      {/* Section content */}
      <div className="min-h-[400px]">
        {activeSection === 'longitudinal' && (
          <EdaLongitudinalRanking groups={timePointGroups} onVariableClick={setSelectedVariable} metric="mean" />
        )}
        {activeSection === 'longClassBalance' && (
          <EdaLongitudinalClassBalance groups={timePointGroups} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'longOutliersZ' && (
          <EdaLongitudinalRanking groups={timePointGroups} onVariableClick={setSelectedVariable} metric="outliersZ" />
        )}
        {activeSection === 'longOutliersIqr' && (
          <EdaLongitudinalRanking groups={timePointGroups} onVariableClick={setSelectedVariable} metric="outliersIqr" />
        )}
        {activeSection === 'longIqr' && (
          <EdaLongitudinalRanking groups={timePointGroups} onVariableClick={setSelectedVariable} metric="iqr" />
        )}
        {activeSection === 'skewnessIqr' && (
          <EdaSkewnessIqrScatter variables={numericVars} timePointGroups={timePointGroups} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'coverage' && (
          <EdaCoverageRanking variables={variables} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'outlierRank' && (
          <EdaOutlierRanking variables={numericVars} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'cvRank' && (
          <EdaCVRanking variables={numericVars} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'imbalanceRank' && (
          <EdaImbalanceRanking variables={categoricalVars} onVariableClick={setSelectedVariable} />
        )}
        {activeSection === 'outliersIqr' && (
          <EdaOutlierAnalysis variables={numericVars} onVariableClick={setSelectedVariable} metric="iqr" />
        )}
        {activeSection === 'outliersZ' && (
          <EdaOutlierAnalysis variables={numericVars} onVariableClick={setSelectedVariable} metric="z" />
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
