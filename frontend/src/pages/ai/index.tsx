'use client';

// Experimental AI hub. Not linked from the nav on purpose — reachable via URL
// only while we iterate on the design.
import React from 'react';
import Link from 'next/link';
import {MessageCircle, Sliders, Grid, Eye, Map, BookOpen, Zap, ArrowRight} from 'react-feather';
import type {ComponentType} from 'react';
import {useCohorts} from '@/components/CohortsContext';
import {ExperimentBadge} from '@/components/ai/ui';

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
    href: '/ai/chat',
    title: 'Copilot',
    icon: MessageCircle,
    tagline: 'A focused conversation with a smart context rail.',
    blurb:
      'Classic chat, front and center. The right-hand rail pins cohorts as context and now embeds a Guide — intent and topic building blocks that assemble questions straight into the composer.',
    accent: 'from-indigo-500 to-violet-500'
  },
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
      'A guided, card-driven canvas. Pick an intent, a cohort, and a topic to assemble a well-formed question — or free-type. Great for discovering what to ask.',
    accent: 'from-rose-500 to-orange-500'
  }
];

const waveTwo: Layout[] = [
  {
    href: '/ai/glassbox',
    title: 'Glass Box',
    icon: Eye,
    tagline: 'See — and control — exactly what the model sees.',
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
      'Put up to four cohorts on a canvas of real charts — domain mix, demographics, concept overlap — computed from the catalog itself. The narrator sees exactly the numbers you see, and nothing else.',
    accent: 'from-fuchsia-500 to-purple-600'
  },
  {
    href: '/ai/notebook',
    title: 'Field Notebook',
    icon: BookOpen,
    tagline: 'Ask in sequence, leave with a document.',
    blurb:
      'Every question becomes a cell — question, answer, and the cohort scope it was asked under. Re-run cells as your scope evolves, then export the whole brief as Markdown.',
    accent: 'from-amber-500 to-orange-600'
  },
  {
    href: '/ai/ideas',
    title: 'Hypothesis Lab',
    icon: Zap,
    tagline: 'From data you have to theses you can test.',
    blurb:
      'Research brainstorming over the catalog: generate testable hypotheses, map analysis opportunities, hunt for gaps — or feasibility-check your own thesis. Ideas land as cards to keep, refine into analysis plans, and export.',
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

export default function AiHub() {
  const {cohortsData} = useCohorts();
  const cohortCount = Object.keys(cohortsData || {}).length;

  return (
    <main className="min-h-screen bg-gradient-to-b from-base-200 to-base-100">
      <div className="max-w-5xl mx-auto px-6 py-12">
        <div className="flex items-center gap-3 mb-2">
          <h1 className="text-3xl font-bold">Cohort AI</h1>
          <ExperimentBadge />
        </div>
        <p className="text-base-content/70 max-w-2xl">
          Talk to a locally-hosted model about the cohorts, their metadata and variables. Seven interfaces
          are being prototyped — each explores a different way to blend chat with structured cohort
          context{cohortCount ? ` across ${cohortCount} cohorts` : ''}.
        </p>

        <div className="mt-10">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-base-content/50 mb-4">
            Chat-first layouts
          </h2>
          <div className="grid md:grid-cols-3 gap-5">
            {waveOne.map(l => (
              <LayoutCard key={l.href} layout={l} />
            ))}
          </div>
        </div>

        <div className="mt-10">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-base-content/50 mb-4">
            Second wave — control, visuals, documents, ideation
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
