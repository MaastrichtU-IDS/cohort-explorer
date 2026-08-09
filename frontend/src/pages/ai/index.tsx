'use client';

// Experimental AI hub. Not linked from the nav on purpose — reachable via URL
// only while we iterate on the design.
import React from 'react';
import Link from 'next/link';
import {MessageCircle, Sliders, Grid, ArrowRight} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {ExperimentBadge} from '@/components/ai/ui';

const layouts = [
  {
    href: '/ai/chat',
    title: 'Copilot',
    icon: MessageCircle,
    tagline: 'A focused conversation with a smart context rail.',
    blurb:
      'Classic chat, front and center. A right-hand rail lets you pin cohorts as context and offers guided prompt suggestions that adapt to your selection.',
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
          Talk to a locally-hosted model about the cohorts, their metadata and variables. Three interfaces are being
          prototyped — each explores a different way to blend chat with structured cohort context
          {cohortCount ? ` across ${cohortCount} cohorts` : ''}.
        </p>

        <div className="grid md:grid-cols-3 gap-5 mt-10">
          {layouts.map(l => {
            const Icon = l.icon;
            return (
              <Link
                key={l.href}
                href={l.href}
                className="group relative rounded-2xl border border-base-300 bg-base-100 p-5 shadow-sm hover:shadow-lg hover:-translate-y-0.5 transition-all overflow-hidden"
              >
                <div className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${l.accent}`} />
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${l.accent} text-white mb-4 shadow`}>
                  <Icon size={22} />
                </div>
                <div className="font-bold text-lg">{l.title}</div>
                <div className="text-sm text-base-content/60 mb-3">{l.tagline}</div>
                <p className="text-sm text-base-content/70 leading-relaxed">{l.blurb}</p>
                <div className="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-primary group-hover:gap-2 transition-all">
                  Open <ArrowRight size={16} />
                </div>
              </Link>
            );
          })}
        </div>

        <p className="text-xs text-base-content/50 mt-10">
          These pages are experimental and intentionally not linked from the navigation yet.
        </p>
      </div>
    </main>
  );
}
