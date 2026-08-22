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

// Statistics each analysis computes. The monospace identifiers are the exact
// column names of the result tables, so users can recognise them later.
const STATISTICS: Record<Kind, {group: string; items: {id: string; gloss: string}[]}[]> = {
  stratified: [
    {
      group: 'Numeric variable, per group',
      items: [
        {id: 'n', gloss: 'patients with a value'},
        {id: 'missing', gloss: 'patients without a value'},
        {id: 'mean', gloss: 'average'},
        {id: 'sd', gloss: 'standard deviation'},
        {id: 'median', gloss: 'middle value'},
        {id: 'q1, q3', gloss: 'quartiles (25th and 75th percentile)'},
        {id: 'min, max', gloss: 'smallest and largest value'},
        {id: 'p5, p95', gloss: '5th and 95th percentile'}
      ]
    },
    {
      group: 'Categorical variable',
      items: [{id: 'count', gloss: 'patients per combination of category and group'}]
    }
  ],
  correlation: [
    {
      group: 'Coefficients',
      items: [
        {id: 'n', gloss: 'patients with both values'},
        {id: 'pearson_r', gloss: 'Pearson correlation (linear relationship, -1 to 1)'},
        {id: 'pearson_ci95_low, pearson_ci95_high', gloss: '95% confidence interval of Pearson r'},
        {id: 'pearson_p', gloss: 'p-value of Pearson r'},
        {id: 'spearman_rho', gloss: 'Spearman rank correlation (monotonic relationship)'},
        {id: 'spearman_p', gloss: 'p-value of Spearman rho'}
      ]
    },
    {
      group: 'Binned means',
      items: [
        {id: 'x_bin', gloss: 'decile of x'},
        {id: 'mean_y', gloss: 'average of y within that decile'},
        {id: 'n', gloss: 'patients in the decile'}
      ]
    }
  ],
  crosstab: [
    {
      group: 'Table',
      items: [{id: 'count (row %)', gloss: 'patients per combination, with the percentage within the row'}]
    },
    {
      group: 'Test of independence',
      items: [
        {id: 'chi_square', gloss: 'chi-square statistic'},
        {id: 'dof', gloss: 'degrees of freedom'},
        {id: 'p_value', gloss: 'probability of this association by chance'},
        {id: 'cramers_v', gloss: 'strength of association (0 = none, 1 = perfect)'},
        {id: 'note', gloss: 'warning when expected counts are below 5'}
      ]
    }
  ],
  compare: [
    {
      group: 'Numeric variable, per cohort and pooled',
      items: [
        {id: 'n, missing', gloss: 'patients with / without a value'},
        {id: 'mean, sd', gloss: 'average and standard deviation'},
        {id: 'median, q1, q3', gloss: 'median and quartiles'},
        {id: 'min, max, p5, p95', gloss: 'extremes and 5th / 95th percentile'},
        {id: 'SMD', gloss: 'standardized mean difference for every pair of cohorts'}
      ]
    },
    {
      group: 'Categorical variable',
      items: [{id: 'count, percent', gloss: 'per category, per cohort and pooled'}]
    },
    {
      group: 'With a break-down variable',
      items: [{id: 'per group', gloss: 'the statistics above within each group of the pooled data'}]
    }
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
        <div className="sticky top-0 z-10 bg-base-100 flex items-start justify-between gap-4 px-6 pt-5 pb-3 border-b border-base-200">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-base-content/50">Analysis type</div>
            <h2 className="text-xl font-bold">{meta.label}</h2>
            <div className="text-xs text-base-content/50 mt-0.5">{meta.min_cohorts === meta.max_cohorts ? `${meta.min_cohorts} cohort` : `${meta.min_cohorts}–${meta.max_cohorts} cohorts`}</div>
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

          {STATISTICS[kind] && (
            <div>
              <div className="text-[11px] uppercase tracking-wide text-base-content/50 mb-2">Statistics computed</div>
              <div className="rounded-xl border border-base-300 overflow-hidden">
                {STATISTICS[kind].map(block => (
                  <div key={block.group} className="border-b border-base-200 last:border-b-0">
                    <div className="px-3 py-1.5 bg-base-200/60 text-xs font-semibold">{block.group}</div>
                    <dl className="px-3 py-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                      {block.items.map(it => (
                        <React.Fragment key={it.id}>
                          <dt className="font-mono text-xs text-base-content/90 pt-0.5 whitespace-nowrap">{it.id}</dt>
                          <dd className="text-base-content/70">{it.gloss}</dd>
                        </React.Fragment>
                      ))}
                    </dl>
                  </div>
                ))}
              </div>
            </div>
          )}

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

        <div className="sticky bottom-0 bg-base-100 border-t border-base-200 px-6 py-4">
          <button type="button" className="btn btn-primary btn-block btn-lg" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
