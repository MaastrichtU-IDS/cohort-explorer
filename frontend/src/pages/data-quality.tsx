'use client';

// Data quality (/data-quality): one entry point for the checks that say how
// sound the catalogue's data is, rather than what it contains. Most of the
// linked pages already existed and were reachable only by URL. The two
// dictionary reports run here and render their findings; the text file each
// has always produced is still a download next to them.
// Pages marked "admins only" render their own refusal for everyone else, so
// this hub stays visible to any signed-in user.

import React, {useState} from 'react';
import Link from 'next/link';
import {Activity, AlertTriangle, Calendar, CheckSquare, Download, GitMerge, Grid, Layers, Play, Shield} from 'react-feather';
import {apiUrl} from '@/utils';

interface Entry {
  title: string;
  href: string;
  description: string;
  admin?: boolean;
  icon: React.ReactNode;
}

interface Section {
  heading: string;
  blurb: string;
  entries: Entry[];
}

const SECTIONS: Section[] = [
  {
    heading: 'Values and distributions',
    blurb: 'Whether the numbers in a column are what they claim to be.',
    entries: [
      {
        title: 'Suspect values',
        href: '/suspect-values',
        admin: true,
        icon: <AlertTriangle size={18} />,
        description:
          'Numeric variables whose profiling looks distorted by values that are probably not real measurements, most often a stand-in for missing that the data owner never declared. Scored from the variable profiling already on disk.',
      },
    ],
  },
  {
    heading: 'Visits',
    blurb: 'Whether the same visit means the same thing across cohorts.',
    entries: [
      {
        title: 'Visit mapping check',
        href: '/visit-mapping-check',
        icon: <Calendar size={18} />,
        description:
          'Visit values that map to more than one visit concept, split into real disagreements and trivial ones, with the variables behind each. Also lists visit values that carry no concept at all.',
      },
    ],
  },
  {
    heading: 'Concept mappings',
    blurb: 'Whether variables are mapped to OMOP concepts consistently and traceably.',
    entries: [
      {
        title: 'Concept clusters',
        href: '/concept-clusters',
        icon: <Layers size={18} />,
        description:
          'Variables grouped by concept name, concept code or OMOP id, so a concept that several cohorts map differently stands out from one they agree on.',
      },
      {
        title: 'Mapping comparison',
        href: '/mapping-comparison',
        icon: <GitMerge size={18} />,
        description:
          'Two mapping result files side by side, to see what a change of model, mode or prompt did to the pairs that were found.',
      },
      {
        title: 'Mapping activity log',
        href: '/mapping-log',
        icon: <Activity size={18} />,
        description:
          'Every cross-study mapping run with who asked for it, the cohorts involved and how it ended.',
      },
      {
        title: 'LLM mapping decisions',
        href: '/llm-decisions',
        admin: true,
        icon: <Shield size={18} />,
        description:
          'The stored decisions behind the mappings, with the consistency view that shows where the same pair was judged differently on different runs.',
      },
    ],
  },
  {
    heading: 'Coverage across cohorts',
    blurb: 'Which concepts the catalogue actually holds, and where the gaps are.',
    entries: [
      {
        title: 'Concept coverage',
        href: '/concept-coverage',
        icon: <Grid size={18} />,
        description:
          'Every concept threaded through every cohort, with the observation counts of the cohorts that went through variable profiling.',
      },
    ],
  },
];

// ---------------------------------------------------------------------------
// Dictionary reports
// ---------------------------------------------------------------------------

interface CohortIssues {
  cohort_id: string;
  status: string;
  file?: string;
  folder?: string;
  dictionary_date?: string;
  errors?: string[];
  error?: string;
}

interface AthenaCohort {
  cohort_id: string;
  status: string;
  file?: string;
  n_pass?: number;
  n_fail?: number;
  n_na?: number;
  fail_variables?: string[];
  na_variables?: string[];
  error?: string;
}

interface ReportData<T> {
  generated_at: string;
  summary: Record<string, number>;
  cohorts: T[];
}

const SUMMARY_LABELS: Record<string, string> = {
  total_cohorts: 'cohorts',
  cohorts_with_errors: 'with issues',
  cohorts_without_dict: 'without a dictionary',
  total_errors: 'issues in total',
  cohorts_passed: 'passed',
  cohorts_failed: 'with failures',
  cohorts_errored: 'could not be checked',
};

const STATUS_BADGE: Record<string, string> = {
  ok: 'badge-success',
  passed: 'badge-success',
  errors: 'badge-warning',
  failed: 'badge-warning',
  structure_failed: 'badge-error',
  error: 'badge-error',
  no_dictionary: 'badge-ghost',
  no_folder: 'badge-ghost',
};

const STATUS_LABEL: Record<string, string> = {
  ok: 'OK',
  passed: 'passed',
  errors: 'issues',
  failed: 'failures',
  structure_failed: 'structure check failed',
  error: 'could not be checked',
  no_dictionary: 'no dictionary',
  no_folder: 'no folder',
};

const Summary = ({summary}: {summary: Record<string, number>}) => (
  <div className="flex flex-wrap gap-2 mb-3">
    {Object.entries(summary).map(([key, value]) => (
      <span key={key} className="badge badge-outline badge-sm">
        {value} {SUMMARY_LABELS[key] || key.replace(/_/g, ' ')}
      </span>
    ))}
  </div>
);

const VariableList = ({label, names}: {label: string; names: string[]}) => {
  const [open, setOpen] = useState(false);
  if (!names.length) return null;
  const shown = open ? names : names.slice(0, 12);
  return (
    <div className="mt-1">
      <span className="text-base-content/60">{label}: </span>
      <span className="font-mono">{shown.join(', ')}</span>
      {names.length > shown.length && (
        <button className="link link-primary ml-1" onClick={() => setOpen(true)}>
          and {names.length - shown.length} more
        </button>
      )}
    </div>
  );
};

export default function DataQualityPage() {
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState<Record<string, string>>({});
  const [issues, setIssues] = useState<ReportData<CohortIssues> | null>(null);
  const [athena, setAthena] = useState<ReportData<AthenaCohort> | null>(null);

  const post = async (path: string) => {
    const response = await fetch(`${apiUrl}${path}`, {method: 'POST', credentials: 'include'});
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }
    return response;
  };

  const run = async (key: string, path: string, apply: (data: any) => void) => {
    setRunning(key);
    setError(e => ({...e, [key]: ''}));
    try {
      apply(await (await post(path)).json());
    } catch (e: any) {
      setError(err => ({...err, [key]: e?.message || 'The check could not be run'}));
    } finally {
      setRunning(null);
    }
  };

  const download = async (key: string, path: string, filename: string) => {
    setRunning(key);
    setError(e => ({...e, [key]: ''}));
    try {
      const response = await post(path);
      const blob = await response.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${filename}_${new Date().toISOString().slice(0, 10)}.txt`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e: any) {
      setError(err => ({...err, [key]: e?.message || 'The report could not be generated'}));
    } finally {
      setRunning(null);
    }
  };

  const Buttons = ({k, jsonPath, filePath, filename}: {k: string; jsonPath: string; filePath: string; filename: string}) => (
    <div className="flex flex-wrap gap-2 mt-3">
      <button className="btn btn-sm btn-primary" disabled={running !== null}
              onClick={() => run(k, jsonPath, k === 'issues' ? setIssues : setAthena)}>
        {running === k
          ? <><span className="loading loading-spinner loading-xs" /> Running…</>
          : <><Play size={14} /> Run the check</>}
      </button>
      <button className="btn btn-sm btn-outline" disabled={running !== null}
              onClick={() => download(`${k}-file`, filePath, filename)}>
        {running === `${k}-file`
          ? <><span className="loading loading-spinner loading-xs" /> Generating…</>
          : <><Download size={14} /> Download the text report</>}
      </button>
    </div>
  );

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="flex items-center gap-3 mb-2">
        <CheckSquare size={28} />
        <h1 className="text-3xl font-bold">Data quality</h1>
      </div>
      <p className="text-sm text-base-content/70 mb-8 max-w-3xl">
        The checks and views that say something about how sound the catalogue&apos;s data is, rather
        than what it contains. Entries marked as administrator pages will refuse anyone else.
      </p>

      {SECTIONS.map(section => (
        <section key={section.heading} className="mb-8">
          <h2 className="text-xl font-semibold">{section.heading}</h2>
          <p className="text-sm text-base-content/60 mb-3">{section.blurb}</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {section.entries.map(entry => (
              <Link
                key={entry.href}
                href={entry.href}
                className="flex gap-3 rounded-lg border border-base-300 bg-base-100 p-4 hover:border-primary transition-colors"
              >
                <span className="mt-0.5 shrink-0 text-primary">{entry.icon}</span>
                <span>
                  <span className="flex items-center gap-2">
                    <span className="font-medium">{entry.title}</span>
                    {entry.admin && <span className="badge badge-ghost badge-sm">admins only</span>}
                  </span>
                  <span className="block text-xs text-base-content/60 mt-1">{entry.description}</span>
                </span>
              </Link>
            ))}
          </div>
        </section>
      ))}

      <section className="mb-8">
        <h2 className="text-xl font-semibold">Dictionary checks</h2>
        <p className="text-sm text-base-content/60 mb-3">
          Run over the current data dictionary of every cohort, on demand. Administrators only, and
          they take a while on a full catalogue. The text report each one has always produced is
          still available next to it.
        </p>

        <div className="rounded-lg border border-base-300 bg-base-100 p-4 mb-3">
          <div className="font-medium">Metadata syntax issues</div>
          <p className="text-xs text-base-content/60 mt-1">
            Every validation and syntax problem in the dictionaries: missing required columns,
            malformed categorical definitions, missing-value codes that cannot be read, and the rest
            of the upload validation, applied to what is on the server now.
          </p>
          <Buttons k="issues" jsonPath="/metadata-syntax-issues-report-json"
                   filePath="/metadata-syntax-issues-report" filename="metadata_syntax_issues" />
          {error['issues'] && <div className="alert alert-error mt-3 text-sm">{error['issues']}</div>}
          {error['issues-file'] && <div className="alert alert-error mt-3 text-sm">{error['issues-file']}</div>}

          {issues && (
            <div className="mt-4">
              <Summary summary={issues.summary} />
              <p className="text-xs text-base-content/50 mb-2">Run at {issues.generated_at}.</p>
              <div className="space-y-2">
                {issues.cohorts.filter(c => c.status !== 'ok' && c.status !== 'no_folder').map(c => (
                  <details key={c.cohort_id} className="rounded border border-base-300 p-2">
                    <summary className="cursor-pointer text-sm flex items-center gap-2">
                      <span className={`badge badge-sm ${STATUS_BADGE[c.status] || 'badge-ghost'}`}>
                        {STATUS_LABEL[c.status] || c.status}
                      </span>
                      <span className="font-medium">{c.cohort_id}</span>
                      {!!c.errors?.length && <span className="text-base-content/60">{c.errors.length} issues</span>}
                    </summary>
                    <div className="text-xs mt-2 space-y-1">
                      {c.file && <div className="text-base-content/60">{c.file} of {c.dictionary_date}</div>}
                      {c.folder && <div className="text-base-content/60 font-mono">{c.folder}</div>}
                      {c.error && <div className="text-error">{c.error}</div>}
                      {!!c.errors?.length && (
                        <ol className="list-decimal list-inside space-y-1">
                          {c.errors.map((e, i) => <li key={i}>{e}</li>)}
                        </ol>
                      )}
                    </div>
                  </details>
                ))}
                {issues.cohorts.every(c => c.status === 'ok' || c.status === 'no_folder') && (
                  <div className="alert alert-success text-sm">Every dictionary on the server passes validation.</div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="rounded-lg border border-base-300 bg-base-100 p-4">
          <div className="font-medium">Athena concept code validation</div>
          <p className="text-xs text-base-content/60 mt-1">
            Checks the concept name, concept code and OMOP id of every variable, additional context,
            categorical value, unit and visit against the OMOP graph and the Athena API, and reports
            the triples that disagree.
          </p>
          <Buttons k="athena" jsonPath="/validate-athena-codes-all-summary-json"
                   filePath="/validate-athena-codes-all-summary" filename="athena_code_validation" />
          {error['athena'] && <div className="alert alert-error mt-3 text-sm">{error['athena']}</div>}
          {error['athena-file'] && <div className="alert alert-error mt-3 text-sm">{error['athena-file']}</div>}

          {athena && (
            <div className="mt-4">
              <Summary summary={athena.summary} />
              <p className="text-xs text-base-content/50 mb-2">Run at {athena.generated_at}.</p>
              <div className="overflow-x-auto">
                <table className="table table-sm">
                  <thead>
                    <tr>
                      <th>Cohort</th><th>Status</th>
                      <th className="text-right">Pass</th>
                      <th className="text-right">Fail</th>
                      <th className="text-right">N/A</th>
                      <th>Variables that did not validate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {athena.cohorts.filter(c => c.status !== 'no_folder' && c.status !== 'no_dictionary').map(c => (
                      <tr key={c.cohort_id}>
                        <td className="whitespace-nowrap font-medium">{c.cohort_id}</td>
                        <td>
                          <span className={`badge badge-sm ${STATUS_BADGE[c.status] || 'badge-ghost'}`}>
                            {STATUS_LABEL[c.status] || c.status}
                          </span>
                        </td>
                        <td className="text-right">{c.n_pass ?? ''}</td>
                        <td className="text-right">{c.n_fail ?? ''}</td>
                        <td className="text-right">{c.n_na ?? ''}</td>
                        <td className="text-xs max-w-xl">
                          {c.error && <div className="text-error">{c.error}</div>}
                          <VariableList label="failed" names={c.fail_variables || []} />
                          <VariableList label="not applicable" names={c.na_variables || []} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
