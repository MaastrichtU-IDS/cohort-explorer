'use client';

// Alternative iCARE-AI interfaces. The main interface (the Copilot layout)
// lives at /ai — the layouts below are experimental takes kept around for
// evaluation only. Reachable by direct URL, restricted to admins, and
// deliberately not linked from anywhere.
import React from 'react';
import Link from 'next/link';
import {Sliders, Grid, Eye, Map, BookOpen, Zap, ArrowRight, ArrowLeft} from 'react-feather';
import type {ComponentType} from 'react';
import {useCohorts} from '@/components/CohortsContext';
import {ExperimentBadge} from '@/components/ai/ui';
import {withAiAccess} from '@/components/ai/guards';

interface Layout {
  href: string;
  title: string;
  icon: ComponentType<any>;
  tagline: string;
  blurb: string;
  accent: string;
}

const waveOne: Layout[] = [
  {
    href: '/ai/console',
    title: 'Cohort Console',
    icon: Sliders,
    tagline: 'Browse cohorts and variables, then interrogate them.',
    blurb:
      'A split workspace: explore the cohort catalog and variables on the left, chat on the right. Selecting cohorts injects them as grounding context automatically.',
    accent: 'from-emerald-500 to-teal-500'
  },
  {
    href: '/ai/studio',
    title: 'Prompt Studio',
    icon: Grid,
    tagline: 'Compose questions from building blocks.',
    blurb:
      'A guided, card-driven canvas. Pick an intent, a cohort, and a topic to assemble a well-formed question, or free-type. Great for discovering what to ask.',
    accent: 'from-rose-500 to-orange-500'
  }
];

const waveTwo: Layout[] = [
  {
    href: '/ai/glassbox',
    title: 'Glass Box',
    icon: Eye,
    tagline: 'See, and control, exactly what the model sees.',
    blurb:
      'Edit the model’s instructions (or pick a persona preset), toggle which metadata groups are shared, dial variable detail up or down, and preview the payload byte-for-byte before sending.',
    accent: 'from-cyan-500 to-sky-600'
  },
  {
    href: '/ai/atlas',
    title: 'Cohort Atlas',
    icon: Map,
    tagline: 'A visual comparison canvas, narrated by the model.',
    blurb:
      'Put up to four cohorts on a canvas of real charts (domain mix, demographics, concept overlap) computed from the catalog itself. The narrator sees exactly the numbers you see, and nothing else.',
    accent: 'from-fuchsia-500 to-purple-600'
  },
  {
    href: '/ai/notebook',
    title: 'Field Notebook',
    icon: BookOpen,
    tagline: 'Ask in sequence, leave with a document.',
    blurb:
      'Every question becomes a cell: question, answer, and the cohort scope it was asked under. Re-run cells as your scope evolves, then export the whole brief as Markdown.',
    accent: 'from-amber-500 to-orange-600'
  },
  {
    href: '/ai/ideas',
    title: 'Hypothesis Lab',
    icon: Zap,
    tagline: 'From data you have to theses you can test.',
    blurb:
      'Research brainstorming over the catalog: generate testable hypotheses, map analysis opportunities, hunt for gaps, or feasibility-check your own thesis. Ideas land as cards to keep, refine into analysis plans, and export.',
    accent: 'from-lime-500 to-emerald-600'
  }
];

function LayoutCard({layout}: {layout: Layout}) {
  const Icon = layout.icon;
  return (
    <Link
      href={layout.href}
      className="group relative rounded-2xl border border-base-300 bg-base-100 p-5 shadow-sm hover:shadow-lg hover:-translate-y-0.5 transition-all overflow-hidden"
    >
      <div className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${layout.accent}`} />
      <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${layout.accent} text-white mb-4 shadow`}>
        <Icon size={22} />
      </div>
      <div className="font-bold text-lg">{layout.title}</div>
      <div className="text-sm text-base-content/60 mb-3">{layout.tagline}</div>
      <p className="text-sm text-base-content/70 leading-relaxed">{layout.blurb}</p>
      <div className="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-primary group-hover:gap-2 transition-all">
        Open <ArrowRight size={16} />
      </div>
    </Link>
  );
}

function AiAlternatives() {
  const {cohortsData} = useCohorts();
  const cohortCount = Object.keys(cohortsData || {}).length;

  return (
    <main className="min-h-screen bg-gradient-to-b from-base-200 to-base-100">
      <div className="max-w-5xl mx-auto px-6 py-12">
        <Link href="/ai" className="btn btn-ghost btn-sm gap-1 mb-4 -ml-3">
          <ArrowLeft size={16} /> iCARE-AI
        </Link>
        <div className="flex items-center gap-3 mb-2">
          <h1 className="text-3xl font-bold">iCARE-AI · Alternative interfaces</h1>
          <ExperimentBadge />
        </div>
        <p className="text-base-content/70 max-w-2xl">
          The main iCARE-AI interface lives at <Link href="/ai" className="link">/ai</Link>. The layouts
          below are alternative takes being prototyped. Each explores a different way to blend chat
          with structured cohort context{cohortCount ? ` across ${cohortCount} cohorts` : ''}, powered by
          the same local model running exclusively on Maastricht University servers.
        </p>

        <div className="mt-10">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-base-content/50 mb-4">
            Chat-first layouts
          </h2>
          <div className="grid md:grid-cols-2 gap-5">
            {waveOne.map(l => (
              <LayoutCard key={l.href} layout={l} />
            ))}
          </div>
        </div>

        <div className="mt-10">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-base-content/50 mb-4">
            Second wave: control, visuals, documents, ideation
          </h2>
          <div className="grid md:grid-cols-2 gap-5">
            {waveTwo.map(l => (
              <LayoutCard key={l.href} layout={l} />
            ))}
          </div>
        </div>

        <p className="text-xs text-base-content/50 mt-10">
          These pages are experimental and intentionally not linked from the navigation yet.
        </p>
      </div>
    </main>
  );
}

export default withAiAccess(AiAlternatives, {requireAdmin: true});
