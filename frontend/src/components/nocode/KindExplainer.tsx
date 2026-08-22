'use client';

// "What is this?" overlay for an analysis type: a larger, labelled example
// figure, the plain-language explanation and a few concrete examples. Closes
// on the big X, on a click outside the panel, or with Escape.
import React, {useEffect} from 'react';
import {X} from 'react-feather';
import {Kind, KindMeta} from './client';
import MiniChart from './MiniChart';

// Worked examples per analysis type: what a researcher would actually ask.
const EXAMPLES: Record<Kind, {question: string; setup: string}[]> = {
  stratified: [
    {question: 'Is body weight different between men and women?', setup: 'Variable of interest: weight. Break down by: sex.'},
    {question: 'How is NYHA class distributed in patients with and without diabetes?', setup: 'Variable of interest: NYHA class. Break down by: diabetes.'},
    {question: 'Does systolic blood pressure differ by smoking status?', setup: 'Variable of interest: systolic BP. Break down by: smoking status.'}
  ],
  correlation: [
    {question: 'Do heavier patients tend to be taller?', setup: 'x: height. y: weight.'},
    {question: 'Does ejection fraction decrease with age?', setup: 'x: age. y: LVEF.'},
    {question: 'Are creatinine and potassium related?', setup: 'x: creatinine. y: potassium.'}
  ],
  crosstab: [
    {question: 'Is diabetes more frequent in men or in women?', setup: 'Rows: sex. Columns: diabetes.'},
    {question: 'Are patients in higher NYHA classes hospitalised more often?', setup: 'Rows: NYHA class. Columns: hospitalisation.'},
    {question: 'Is atrial fibrillation associated with smoking?', setup: 'Rows: smoking status. Columns: atrial fibrillation.'}
  ],
  compare: [
    {question: 'How does the age distribution of TIME-CHF compare with Aachen-HF?', setup: 'Variable: age, harmonized across the two cohorts.'},
    {question: 'What is the sex distribution in each cohort and in all of them pooled?', setup: 'Variable: sex (e.g. Geschlecht M/W mapped to gender 1/2).'},
    {question: 'What does LVEF look like across cohorts, separately for diabetics?', setup: 'Variable: LVEF. Break down by: diabetes, harmonized across cohorts.'}
  ]
};

// Axis and legend labels for the enlarged example figure.
const FIGURE_LABELS: Record<Kind, {x: string; y: string; legend?: string[]; caption: string}> = {
  stratified: {x: 'variable of interest (e.g. weight, kg)', y: 'density / count', legend: ['group A (e.g. women)', 'group B (e.g. men)'], caption: 'Left: one distribution curve per group. Right: box plots per group with outliers.'},
  correlation: {x: 'first variable (x)', y: 'second variable (y)', caption: 'Each dot is one patient; the red line is the least-squares fit.'},
  crosstab: {x: 'categories of the row variable', y: 'patients', legend: ['column category 1', 'column category 2', 'column category 3'], caption: 'Stacked bars: how the column variable splits within each row category.'},
  compare: {x: 'harmonized variable', y: 'density (left) / patients (right)', legend: ['cohort A', 'cohort B', 'cohort C'], caption: 'Left: one curve per cohort, side by side. Right: all cohorts stacked into one pooled distribution.'}
};

export default function KindExplainer({kind, meta, onClose}: {kind: Kind; meta: KindMeta; onClose: () => void}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const labels = FIGURE_LABELS[kind];
  const examples = EXAMPLES[kind] || [];

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4" onMouseDown={onClose} role="dialog" aria-modal="true" aria-label={meta.label}>
      <div className="bg-base-100 rounded-2xl shadow-2xl w-full max-w-3xl max-h-[90vh] overflow-y-auto" onMouseDown={e => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-4 px-6 pt-5 pb-3 border-b border-base-200">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-base-content/50">Analysis type</div>
            <h2 className="text-xl font-bold">{meta.label}</h2>
            <div className="text-xs text-base-content/50 mt-0.5">{meta.min_cohorts === meta.max_cohorts ? `${meta.min_cohorts} cohort` : `${meta.min_cohorts} to ${meta.max_cohorts} cohorts`}</div>
          </div>
          <button type="button" className="btn btn-circle btn-ghost btn-lg -mr-2 -mt-1" onClick={onClose} aria-label="Close">
            <X size={28} />
          </button>
        </div>

        <div className="px-6 py-4 space-y-5">
          {/* Enlarged, labelled example figure */}
          <div className="rounded-xl border border-base-300 bg-base-200/50 p-4">
            <div className="text-[11px] uppercase tracking-wide text-base-content/50 mb-2">Example of the figure this produces</div>
            <div className="flex gap-3">
              <div className="flex items-center">
                <span className="text-xs text-base-content/60" style={{writingMode: 'vertical-rl', transform: 'rotate(180deg)'}}>{labels.y}</span>
              </div>
              <div className="flex-1 min-w-0">
                <div className="[&_svg]:h-56">
                  <MiniChart kind={kind} />
                </div>
                <div className="text-center text-xs text-base-content/60 mt-1">{labels.x}</div>
              </div>
            </div>
            {labels.legend && (
              <div className="flex flex-wrap gap-3 mt-3 justify-center">
                {labels.legend.map((l, i) => (
                  <span key={l} className="inline-flex items-center gap-1.5 text-xs text-base-content/70">
                    <span className="w-3 h-3 rounded-sm" style={{background: ['#3b6ea5', '#e08a2e', '#3a9a6a'][i % 3]}} />
                    {l}
                  </span>
                ))}
              </div>
            )}
            <p className="text-xs text-base-content/60 mt-3 text-center">{labels.caption}</p>
          </div>

          <p className="text-sm leading-relaxed text-base-content/90">{meta.explain}</p>

          {examples.length > 0 && (
            <div>
              <div className="text-[11px] uppercase tracking-wide text-base-content/50 mb-2">Examples</div>
              <ul className="space-y-2">
                {examples.map(ex => (
                  <li key={ex.question} className="rounded-lg bg-base-200/60 px-3 py-2">
                    <div className="text-sm font-medium">{ex.question}</div>
                    <div className="text-xs text-base-content/60 mt-0.5">{ex.setup}</div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
