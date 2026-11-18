'use client';

// Data quality (/data-quality): one entry point for the checks and views that
// say something about how sound the catalogue's data is, rather than what it
// contains. Most of them already existed and were reachable only by URL.
// Pages marked "admins only" render their own refusal for everyone else, so
// this hub stays visible to any signed-in user.

import React, {useState} from 'react';
import Link from 'next/link';
import {Activity, AlertTriangle, BarChart2, Calendar, CheckSquare, Download, FileText, GitMerge, Grid, Layers, Share2, Shield} from 'react-feather';
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
      {
        title: 'Concept clusters heatmap',
        href: '/concept-clusters-heatmap',
        icon: <BarChart2 size={18} />,
        description:
          'The same coverage as a shaded grid of clusters against cohorts, which is the view linked from the navigation bar.',
      },
    ],
  },
  {
    heading: 'Metadata themes',
    blurb: 'How the free-text study metadata clusters, and which studies resemble each other.',
    entries: [
      {
        title: 'Metadata clustering',
        href: '/metadata-clustering',
        icon: <Layers size={18} />,
        description:
          'Studies grouped by the themes found in their metadata fields, with the admin review of the theme vocabulary.',
      },
      {
        title: 'Metadata heatmap',
        href: '/metadata-heatmap',
        admin: true,
        icon: <Grid size={18} />,
        description: 'Studies against themes as a shaded grid, from the metadata clustering theme view.',
      },
      {
        title: 'Metadata constellation',
        href: '/metadata-constellation',
        admin: true,
        icon: <Share2 size={18} />,
        description: 'Studies as nodes, linked and drawn closer when they share more themes.',
      },
    ],
  },
];

interface Report {
  title: string;
  path: string;
  filename: string;
  description: string;
  summary: (h: Headers) => string;
}

const REPORTS: Report[] = [
  {
    title: 'Metadata syntax issues',
    path: '/metadata-syntax-issues-report',
    filename: 'metadata_syntax_issues',
    description:
      'Every validation and syntax problem found in the data dictionaries of all cohorts, in one text report. Also regenerated on each backend start.',
    summary: h => {
      const total = h.get('X-Total-Cohorts');
      const errors = h.get('X-Cohorts-With-Errors');
      const without = h.get('X-Cohorts-Without-Dict');
      return `${errors ?? '?'} of ${total ?? '?'} cohorts have issues, ${without ?? '?'} have no dictionary`;
    },
  },
  {
    title: 'Athena concept code validation',
    path: '/validate-athena-codes-all-summary',
    filename: 'athena_code_validation',
    description:
      'Checks the concept code, concept name and OMOP id of every dictionary against the OMOP graph and the Athena API, and reports the triples that disagree.',
    summary: h => `${h.get('X-Total-Errors') ?? '?'} problems found`,
  },
];

export default function DataQualityPage() {
  const [running, setRunning] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, string>>({});

  const runReport = async (report: Report) => {
    setRunning(report.path);
    setResult(r => ({...r, [report.path]: ''}));
    try {
      const response = await fetch(`${apiUrl}${report.path}`, {method: 'POST', credentials: 'include'});
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${response.status}`);
      }
      const summary = report.summary(response.headers);
      const blob = await response.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${report.filename}_${new Date().toISOString().slice(0, 10)}.txt`;
      a.click();
      URL.revokeObjectURL(a.href);
      setResult(r => ({...r, [report.path]: `Downloaded. ${summary}.`}));
    } catch (e: any) {
      setResult(r => ({...r, [report.path]: e?.message || 'The report could not be generated'}));
    } finally {
      setRunning(null);
    }
  };

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
        <h2 className="text-xl font-semibold">Dictionary reports</h2>
        <p className="text-sm text-base-content/60 mb-3">
          Generated on demand and downloaded as a text file. Administrators only, and they take a
          while on a full catalogue.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {REPORTS.map(report => (
            <div key={report.path} className="rounded-lg border border-base-300 bg-base-100 p-4">
              <div className="flex items-start gap-3">
                <span className="mt-0.5 shrink-0 text-primary"><FileText size={18} /></span>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{report.title}</span>
                    <span className="badge badge-ghost badge-sm">admins only</span>
                  </div>
                  <p className="text-xs text-base-content/60 mt-1">{report.description}</p>
                  <button
                    className="btn btn-sm btn-outline mt-3"
                    onClick={() => runReport(report)}
                    disabled={running !== null}
                  >
                    {running === report.path
                      ? <><span className="loading loading-spinner loading-xs" /> Generating…</>
                      : <><Download size={14} /> Generate and download</>}
                  </button>
                  {result[report.path] && (
                    <p className="text-xs mt-2 text-base-content/70">{result[report.path]}</p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
