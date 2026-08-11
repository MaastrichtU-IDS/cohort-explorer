'use client';

// Layout E — "Cohort Atlas": a visual comparison canvas with an AI narrator.
// Every chart is computed client-side from the actual catalog — the model
// cannot hallucinate a bar. The narrator receives, as its context, exactly the
// numbers rendered on the canvas, and comments on what the user is looking at.
import React, {useMemo, useState} from 'react';
import Link from 'next/link';
import {useRouter} from 'next/router';
import {ArrowLeft, Map, Plus, Search, X} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {useCohortChat} from '@/components/ai/useCohortChat';
import {Composer, DisabledNotice, ExperimentBadge, LoginNotice, MessageList} from '@/components/ai/ui';
import {Cohort, Variable} from '@/types';

const MAX_PICKED = 4;

// Validated categorical palette (light / dark steps) — fixed slot order.
// Slots follow the *domain entity* by global rank; never re-colored on filter.
const SLOTS = [
  {bar: 'bg-[#2a78d6] dark:bg-[#3987e5]', swatch: 'bg-[#2a78d6] dark:bg-[#3987e5]', ink: 'text-white'},
  {bar: 'bg-[#eb6834] dark:bg-[#d95926]', swatch: 'bg-[#eb6834] dark:bg-[#d95926]', ink: 'text-white'},
  {bar: 'bg-[#1baf7a] dark:bg-[#199e70]', swatch: 'bg-[#1baf7a] dark:bg-[#199e70]', ink: 'text-white'},
  {bar: 'bg-[#eda100] dark:bg-[#c98500]', swatch: 'bg-[#eda100] dark:bg-[#c98500]', ink: 'text-black/70'}
];
const OTHER_SLOT = {bar: 'bg-[#c3c2b7] dark:bg-[#52514e]', swatch: 'bg-[#c3c2b7] dark:bg-[#52514e]', ink: 'text-black/60 dark:text-white/80'};
// Sex split: fixed semantic hues (male = slot-1 blue, female = magenta).
const MALE_CLS = 'bg-[#2a78d6] dark:bg-[#3987e5]';
const FEMALE_CLS = 'bg-[#e87ba4] dark:bg-[#d55181]';
// Single-hue magnitude bars (catalog overview).
const MAGNITUDE_CLS = 'bg-[#2a78d6] dark:bg-[#3987e5]';

const clean = (v: any): string => {
  if (v === undefined || v === null) return '';
  const t = String(v).trim();
  return ['', 'na', 'n/a', 'nan', 'none', 'null', '-', '--'].includes(t.toLowerCase()) ? '' : t;
};

// ---- Aggregates (all deterministic, all client-side) -----------------------

interface CohortStats {
  id: string;
  cohort: Cohort;
  varCount: number;
  domains: {[domain: string]: number};
  concepts: Set<string>;
  mappedCount: number;
}

function conceptKey(v: Variable): string | null {
  return clean(v.mapped_id) || clean(v.concept_id) || clean(v.omop_id) || null;
}

function computeStats(cohort: Cohort): CohortStats {
  const variables = Object.values(cohort.variables || {});
  const domains: {[d: string]: number} = {};
  const concepts = new Set<string>();
  for (const v of variables) {
    const d = clean(v.omop_domain) || 'Unmapped';
    domains[d] = (domains[d] || 0) + 1;
    const key = conceptKey(v);
    if (key) concepts.add(key);
  }
  return {id: cohort.cohort_id, cohort, varCount: variables.length, domains, concepts, mappedCount: concepts.size};
}

// Rank domains across the whole selection; top 4 keep their slot, rest fold to Other.
function rankDomains(stats: CohortStats[]): string[] {
  const totals: {[d: string]: number} = {};
  for (const s of stats) {
    for (const [d, n] of Object.entries(s.domains)) totals[d] = (totals[d] || 0) + n;
  }
  return Object.entries(totals)
    .sort((a, b) => b[1] - a[1])
    .map(([d]) => d)
    .slice(0, SLOTS.length);
}

function intersectCount(a: Set<string>, b: Set<string>): number {
  let n = 0;
  a.forEach(k => {
    if (b.has(k)) n++;
  });
  return n;
}

// The narrator's context: a faithful textual transcript of the canvas.
function buildAtlasContext(stats: CohortStats[], rankedDomains: string[]): string {
  const parts: string[] = [
    `The user is looking at a visual comparison canvas ("Atlas") showing ${stats.length} cohort(s) side by side. ` +
      'Everything below is computed directly from the cohort catalog and is rendered as charts on screen.'
  ];
  for (const s of stats) {
    const c = s.cohort;
    const lines = [`### ${s.id}`];
    const facts: [string, string][] = [
      ['Institution', clean(c.institution)],
      ['Study type', clean(c.study_type)],
      ['Study design', clean(c.study_design)],
      ['Participants', clean(c.study_participants)],
      ['Population', clean(c.study_population)],
      ['Objective', clean(c.study_objective)]
    ];
    for (const [label, value] of facts) if (value) lines.push(`- ${label}: ${value}`);
    lines.push(`- Total variables: ${s.varCount}`);
    lines.push(`- Variables mapped to OMOP concepts: ${s.mappedCount}`);
    if (c.male_percentage != null || c.female_percentage != null) {
      lines.push(`- Sex split: ${c.male_percentage ?? '?'}% male / ${c.female_percentage ?? '?'}% female`);
    }
    if (c.age_distribution && Object.keys(c.age_distribution).length) {
      lines.push(
        `- Age distribution: ${Object.entries(c.age_distribution)
          .map(([band, pct]) => `${band}: ${pct}%`)
          .join(', ')}`
      );
    }
    const domainLine = Object.entries(s.domains)
      .sort((a, b) => b[1] - a[1])
      .map(([d, n]) => `${d}: ${n} (${s.varCount ? Math.round((100 * n) / s.varCount) : 0}%)`)
      .join(', ');
    lines.push(`- OMOP domain mix (shown as a stacked bar): ${domainLine}`);
    parts.push(lines.join('\n'));
  }
  if (stats.length >= 2) {
    const lines = ['### Concept overlap (shown as an overlap chart/matrix)'];
    for (let i = 0; i < stats.length; i++) {
      for (let j = i + 1; j < stats.length; j++) {
        lines.push(
          `- ${stats[i].id} ∩ ${stats[j].id}: ${intersectCount(stats[i].concepts, stats[j].concepts)} shared mapped OMOP concepts`
        );
      }
    }
    parts.push(lines.join('\n'));
  }
  parts.push(
    `The charted domains, most common first across the selection: ${rankedDomains.join(', ')}.` +
      ' Use ONLY the numbers above — they are what the user sees. Do not invent or extrapolate any statistic.'
  );
  return parts.join('\n\n');
}

const NARRATOR_PROMPT =
  'You are the narrator of a visual dashboard comparing cardiovascular research cohorts. The user ' +
  'is looking at charts of exactly the numbers given in the context: per-cohort profile cards, a ' +
  'stacked bar of OMOP domain mix per cohort, and a concept-overlap panel. Comment like a sharp ' +
  'colleague standing at the same screen: point at panels ("the domain mix bars show…"), explain ' +
  'what differences mean for research practice, and stay strictly within the provided numbers. ' +
  'Be concise; short paragraphs and bullets.';

// ---- Chart pieces ----------------------------------------------------------

function Swatch({cls}: {cls: string}) {
  return <span className={`inline-block w-2.5 h-2.5 rounded-sm ${cls}`} />;
}

function Segment({
  cls,
  ink,
  pct,
  first,
  last,
  label,
  tooltip
}: {
  cls: string;
  ink: string;
  pct: number;
  first: boolean;
  last: boolean;
  label?: string;
  tooltip: string;
}) {
  if (pct <= 0) return null;
  return (
    <div className="relative group h-full" style={{width: `${pct}%`}}>
      <div
        className={`h-full ${cls} ${first ? 'rounded-l' : ''} ${last ? 'rounded-r' : ''} flex items-center justify-center overflow-hidden`}
      >
        {label && pct >= 14 && <span className={`text-[10px] font-medium ${ink}`}>{label}</span>}
      </div>
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:block z-20 whitespace-nowrap rounded-md bg-neutral text-neutral-content text-[11px] px-2 py-1 shadow-lg">
        {tooltip}
      </div>
    </div>
  );
}

function StackedBar({segments}: {segments: {cls: string; ink: string; pct: number; label?: string; tooltip: string}[]}) {
  const visible = segments.filter(s => s.pct > 0);
  return (
    <div className="flex h-6 w-full gap-[2px]">
      {visible.map((s, i) => (
        <Segment key={i} {...s} first={i === 0} last={i === visible.length - 1} />
      ))}
    </div>
  );
}

function ProfileCard({stats}: {stats: CohortStats}) {
  const c = stats.cohort;
  const male = c.male_percentage;
  const female = c.female_percentage;
  return (
    <div className="rounded-xl border border-base-300 bg-base-100 p-4 shadow-sm min-w-0">
      <div className="font-bold truncate">{stats.id}</div>
      {clean(c.institution) && <div className="text-xs text-base-content/50 truncate">{c.institution}</div>}
      <div className="flex items-baseline gap-2 mt-3">
        <span className="text-3xl font-bold tabular-nums">{stats.varCount}</span>
        <span className="text-xs text-base-content/50">variables</span>
      </div>
      <div className="text-xs text-base-content/60 mt-1">
        {stats.mappedCount} mapped to OMOP
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {clean(c.study_type) && <span className="badge badge-ghost badge-xs">{c.study_type}</span>}
        {clean(c.study_participants) && (
          <span className="badge badge-ghost badge-xs">{c.study_participants} participants</span>
        )}
      </div>
      {male != null && female != null && (
        <div className="mt-3">
          <div className="flex justify-between text-[11px] text-base-content/60 mb-1">
            <span>{male}% male</span>
            <span>{female}% female</span>
          </div>
          <StackedBar
            segments={[
              {cls: MALE_CLS, ink: 'text-white', pct: male, tooltip: `Male: ${male}%`},
              {cls: FEMALE_CLS, ink: 'text-white', pct: female, tooltip: `Female: ${female}%`}
            ]}
          />
        </div>
      )}
      {c.age_distribution && Object.keys(c.age_distribution).length > 0 && (
        <div className="mt-3 space-y-1">
          {Object.entries(c.age_distribution).map(([band, pct]) => (
            <div key={band} className="flex items-center gap-2">
              <span className="text-[11px] text-base-content/60 w-12 shrink-0 text-right">{band}</span>
              <div className="flex-1 h-2 bg-base-200 rounded">
                <div className={`h-2 rounded ${MAGNITUDE_CLS}`} style={{width: `${Math.min(100, pct)}%`}} />
              </div>
              <span className="text-[11px] tabular-nums text-base-content/60 w-8">{pct}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function DomainMixPanel({stats, ranked}: {stats: CohortStats[]; ranked: string[]}) {
  return (
    <div className="rounded-xl border border-base-300 bg-base-100 p-4 shadow-sm">
      <div className="font-semibold text-sm mb-1">What they measure — OMOP domain mix</div>
      <div className="text-xs text-base-content/50 mb-3">Share of each cohort&apos;s variables by domain</div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mb-4">
        {ranked.map((d, i) => (
          <span key={d} className="inline-flex items-center gap-1.5 text-xs text-base-content/70">
            <Swatch cls={SLOTS[i].swatch} /> {d}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5 text-xs text-base-content/70">
          <Swatch cls={OTHER_SLOT.swatch} /> Other
        </span>
      </div>
      <div className="space-y-3">
        {stats.map(s => {
          const rankedCounts = ranked.map(d => s.domains[d] || 0);
          const otherCount = s.varCount - rankedCounts.reduce((a, b) => a + b, 0);
          const toPct = (n: number) => (s.varCount ? (100 * n) / s.varCount : 0);
          return (
            <div key={s.id}>
              <div className="text-xs font-medium mb-1">{s.id}</div>
              <StackedBar
                segments={[
                  ...ranked.map((d, i) => ({
                    cls: SLOTS[i].bar,
                    ink: SLOTS[i].ink,
                    pct: toPct(s.domains[d] || 0),
                    label: `${Math.round(toPct(s.domains[d] || 0))}%`,
                    tooltip: `${d}: ${s.domains[d] || 0} variables (${Math.round(toPct(s.domains[d] || 0))}%)`
                  })),
                  {
                    cls: OTHER_SLOT.bar,
                    ink: OTHER_SLOT.ink,
                    pct: toPct(otherCount),
                    label: `${Math.round(toPct(otherCount))}%`,
                    tooltip: `Other domains: ${otherCount} variables (${Math.round(toPct(otherCount))}%)`
                  }
                ]}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function OverlapPanel({stats}: {stats: CohortStats[]}) {
  if (stats.length === 2) {
    const [a, b] = stats;
    const shared = intersectCount(a.concepts, b.concepts);
    const aOnly = a.mappedCount - shared;
    const bOnly = b.mappedCount - shared;
    const union = aOnly + shared + bOnly;
    const toPct = (n: number) => (union ? (100 * n) / union : 0);
    return (
      <div className="rounded-xl border border-base-300 bg-base-100 p-4 shadow-sm">
        <div className="font-semibold text-sm mb-1">Shared ground — concept overlap</div>
        <div className="text-xs text-base-content/50 mb-3">
          Distinct OMOP concepts mapped in each cohort, and where they coincide
        </div>
        <StackedBar
          segments={[
            {
              cls: SLOTS[0].bar,
              ink: SLOTS[0].ink,
              pct: toPct(aOnly),
              label: `${aOnly}`,
              tooltip: `Only in ${a.id}: ${aOnly} concepts`
            },
            {
              cls: SLOTS[2].bar,
              ink: SLOTS[2].ink,
              pct: toPct(shared),
              label: `${shared}`,
              tooltip: `Shared: ${shared} concepts`
            },
            {
              cls: SLOTS[1].bar,
              ink: SLOTS[1].ink,
              pct: toPct(bOnly),
              label: `${bOnly}`,
              tooltip: `Only in ${b.id}: ${bOnly} concepts`
            }
          ]}
        />
        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3">
          <span className="inline-flex items-center gap-1.5 text-xs text-base-content/70">
            <Swatch cls={SLOTS[0].swatch} /> Only {a.id} ({aOnly})
          </span>
          <span className="inline-flex items-center gap-1.5 text-xs text-base-content/70">
            <Swatch cls={SLOTS[2].swatch} /> Shared ({shared})
          </span>
          <span className="inline-flex items-center gap-1.5 text-xs text-base-content/70">
            <Swatch cls={SLOTS[1].swatch} /> Only {b.id} ({bOnly})
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-base-300 bg-base-100 p-4 shadow-sm overflow-x-auto">
      <div className="font-semibold text-sm mb-1">Shared ground — concept overlap</div>
      <div className="text-xs text-base-content/50 mb-3">
        Shared mapped OMOP concepts per pair; diagonal = concepts mapped in that cohort
      </div>
      <table className="table table-xs w-auto">
        <thead>
          <tr>
            <th></th>
            {stats.map(s => (
              <th key={s.id} className="text-xs">
                {s.id}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {stats.map((row, i) => (
            <tr key={row.id}>
              <th className="text-xs">{row.id}</th>
              {stats.map((col, j) => (
                <td key={col.id} className={`tabular-nums text-sm ${i === j ? 'font-bold' : ''}`}>
                  {i === j ? row.mappedCount : intersectCount(row.concepts, col.concepts)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CatalogOverview({cohorts, onPick}: {cohorts: Cohort[]; onPick: (id: string) => void}) {
  const top = useMemo(
    () =>
      [...cohorts]
        .map(c => ({id: c.cohort_id, n: Object.keys(c.variables || {}).length}))
        .filter(c => c.n > 0)
        .sort((a, b) => b.n - a.n)
        .slice(0, 12),
    [cohorts]
  );
  const max = top[0]?.n || 1;
  return (
    <div className="rounded-xl border border-base-300 bg-base-100 p-4 shadow-sm max-w-2xl mx-auto">
      <div className="font-semibold text-sm mb-1">The catalog at a glance</div>
      <div className="text-xs text-base-content/50 mb-3">
        Cohorts with the most uploaded variables — click one to start mapping
      </div>
      <div className="space-y-1.5">
        {top.map(c => (
          <button key={c.id} onClick={() => onPick(c.id)} className="flex items-center gap-2 w-full group">
            <span className="text-xs w-32 shrink-0 text-right truncate group-hover:text-primary transition-colors">
              {c.id}
            </span>
            <div className="flex-1 h-4">
              <div
                className={`h-4 rounded ${MAGNITUDE_CLS} opacity-90 group-hover:opacity-100 transition-opacity`}
                style={{width: `${(100 * c.n) / max}%`}}
              />
            </div>
            <span className="text-xs tabular-nums text-base-content/60 w-10 text-left">{c.n}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ---- Demo data (open /ai/atlas?demo to design-review without a backend) ----

function makeDemoCohorts(): {[id: string]: Cohort} {
  const mk = (
    id: string,
    meta: Partial<Cohort>,
    domains: {[domain: string]: [number, number]} // domain -> [count, conceptIdOffset]
  ): Cohort => {
    const variables: {[k: string]: Variable} = {};
    let i = 0;
    for (const [domain, [count, offset]] of Object.entries(domains)) {
      for (let n = 0; n < count; n++, i++) {
        variables[`v${i}`] = {
          var_name: `${domain.toLowerCase()}_${n}`,
          var_label: `${domain} variable ${n}`,
          var_type: n % 3 ? 'FLOAT' : 'STR',
          omop_domain: domain,
          // Overlapping concept-id ranges between cohorts simulate shared concepts.
          concept_id: `${offset + n}`,
          mapped_id: null,
          mapped_label: null,
          count: 0, na: 0, max: '', min: '', units: '', visits: '', visit_concept_name: '',
          formula: '', definition: '', index: i, categories: []
        } as Variable;
      }
    }
    return {cohort_id: id, cohort_email: [], institution: '', study_type: '', study_participants: '',
      study_population: '', study_duration: '', study_ongoing: '', study_objective: '',
      primary_outcome_spec: '', secondary_outcome_spec: '', morbidity: '', study_start: '',
      study_end: '', male_percentage: null, female_percentage: null, variables, ...meta} as Cohort;
  };
  return {
    'DEMO-HEART': mk(
      'DEMO-HEART',
      {institution: 'Demo University Hospital', study_type: 'Observational', study_participants: '2100',
        male_percentage: 61, female_percentage: 39, age_distribution: {'18-39': 8, '40-64': 47, '65+': 45},
        study_objective: 'Demo cohort: long-term outcomes in chronic heart failure.'},
      {Measurement: [120, 1000], Condition: [40, 5000], Drug: [30, 8000], Observation: [20, 12000], Procedure: [10, 15000]}
    ),
    'DEMO-METAB': mk(
      'DEMO-METAB',
      {institution: 'Demo Institute of Metabolism', study_type: 'Prospective cohort', study_participants: '5400',
        male_percentage: 48, female_percentage: 52, age_distribution: {'18-39': 22, '40-64': 58, '65+': 20},
        study_objective: 'Demo cohort: metabolic risk factors and cardiovascular disease.'},
      {Measurement: [90, 1060], Condition: [25, 5020], Drug: [55, 8010], Observation: [45, 13000], Procedure: [5, 15500]}
    ),
    'DEMO-AGING': mk(
      'DEMO-AGING',
      {institution: 'Demo Centre for Healthy Aging', study_type: 'Longitudinal', study_participants: '900',
        male_percentage: 44, female_percentage: 56, age_distribution: {'40-64': 18, '65+': 82},
        study_objective: 'Demo cohort: cardiovascular aging in the very old.'},
      {Measurement: [60, 1100], Observation: [70, 12550], Condition: [30, 5010], Device: [15, 20000]}
    )
  };
}

// ---- Main layout -----------------------------------------------------------

export default function CohortAtlas() {
  const {cohortsData, userEmail} = useCohorts();
  const chat = useCohortChat();
  const router = useRouter();
  const [picked, setPicked] = useState<string[]>([]);
  const [query, setQuery] = useState('');

  const isDemo = router.query.demo !== undefined;
  const demoData = useMemo(() => (isDemo ? makeDemoCohorts() : null), [isDemo]);
  const data = demoData || cohortsData;

  const allCohorts = useMemo(() => Object.values(data || {}) as Cohort[], [data]);

  const stats = useMemo(
    () => picked.map(id => data?.[id]).filter(Boolean).map(c => computeStats(c as Cohort)),
    [picked, data]
  );
  const rankedDomains = useMemo(() => rankDomains(stats), [stats]);
  const atlasContext = useMemo(
    () => (stats.length ? buildAtlasContext(stats, rankedDomains) : ''),
    [stats, rankedDomains]
  );

  const addCohort = (id: string) => {
    setPicked(prev => (prev.includes(id) || prev.length >= MAX_PICKED ? prev : [...prev, id]));
    setQuery('');
  };
  const removeCohort = (id: string) => setPicked(prev => prev.filter(x => x !== id));

  const candidates = useMemo(() => {
    const q = query.trim().toLowerCase();
    const pool = allCohorts
      .filter(c => !picked.includes(c.cohort_id))
      .sort((a, b) => Object.keys(b.variables || {}).length - Object.keys(a.variables || {}).length);
    const matches = q ? pool.filter(c => c.cohort_id.toLowerCase().includes(q)) : pool;
    return matches.slice(0, 8);
  }, [allCohorts, picked, query]);

  const blocked = !chat.enabled || !userEmail;

  const narratorChips =
    stats.length >= 2
      ? [
          'Narrate the key differences on this canvas.',
          'What harmonization challenges do these cohorts pose?',
          'Suggest a research question these cohorts could answer together.'
        ]
      : [
          'Describe this cohort’s profile in plain terms.',
          'What stands out in its domain mix?',
          'What kinds of studies would this cohort support?'
        ];

  const ask = (text?: string) =>
    chat.send(text, {systemPrompt: NARRATOR_PROMPT, contextOverride: atlasContext});

  return (
    <main className="h-[calc(100vh-8rem)] bg-base-200 flex flex-col">
      {/* Top bar */}
      <div className="border-b border-base-300 bg-base-100 px-6 py-3 flex items-center gap-3">
        <Link href="/ai" className="btn btn-ghost btn-sm gap-1">
          <ArrowLeft size={16} /> Hub
        </Link>
        <h1 className="font-bold text-lg">Cohort Atlas</h1>
        <ExperimentBadge />
        {isDemo && <span className="badge badge-warning badge-sm">demo data</span>}
        <span className="hidden md:inline text-xs text-base-content/50">
          charts from real data · narrated by the model
        </span>
        {chat.model && <span className="text-xs text-base-content/50 ml-auto">model: {chat.model}</span>}
      </div>

      {/* Picker strip */}
      <div className="border-b border-base-300 bg-base-100 px-6 py-2 flex items-center gap-2 flex-wrap">
        {picked.map(id => (
          <span key={id} className="badge badge-primary gap-1">
            {id}
            <button onClick={() => removeCohort(id)}>
              <X size={12} />
            </button>
          </span>
        ))}
        {picked.length < MAX_PICKED && (
          <>
            <label className="input input-xs input-bordered flex items-center gap-1 w-48">
              <Search size={12} className="opacity-50" />
              <input
                className="grow"
                placeholder={picked.length ? 'Add another…' : 'Pick cohorts to map…'}
                value={query}
                onChange={e => setQuery(e.target.value)}
              />
            </label>
            {candidates.map(c => (
              <button
                key={c.cohort_id}
                onClick={() => addCohort(c.cohort_id)}
                className="badge badge-outline gap-1 hover:badge-primary transition-all"
              >
                <Plus size={11} /> {c.cohort_id}
              </button>
            ))}
          </>
        )}
        {picked.length >= MAX_PICKED && (
          <span className="text-xs text-base-content/40">maximum {MAX_PICKED} cohorts on the canvas</span>
        )}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Canvas */}
        <section className="flex-1 overflow-y-auto p-6">
          {chat.configLoaded && !chat.enabled && (
            <div className="max-w-2xl mx-auto mb-4">
              <DisabledNotice />
            </div>
          )}
          {!userEmail && (
            <div className="max-w-2xl mx-auto mb-4">
              <LoginNotice />
            </div>
          )}
          {stats.length === 0 ? (
            <div className="mt-6">
              <div className="text-center mb-6">
                <div className="inline-flex p-4 rounded-2xl bg-gradient-to-br from-fuchsia-500 to-purple-600 text-white shadow-lg mb-4">
                  <Map size={28} />
                </div>
                <div className="text-2xl font-bold mb-2">Put cohorts on the map</div>
                <p className="text-base-content/60 max-w-md mx-auto">
                  Pick up to {MAX_PICKED} cohorts above. The Atlas draws real charts from the catalog —
                  and the narrator on the right sees exactly what you see.
                </p>
              </div>
              <CatalogOverview cohorts={allCohorts} onPick={addCohort} />
            </div>
          ) : (
            <div className="space-y-5 max-w-4xl mx-auto">
              <div
                className={`grid gap-4 md:grid-cols-2 ${
                  stats.length >= 4 ? 'lg:grid-cols-4' : stats.length === 3 ? 'lg:grid-cols-3' : 'lg:grid-cols-2'
                }`}
              >
                {stats.map(s => (
                  <ProfileCard key={s.id} stats={s} />
                ))}
              </div>
              <DomainMixPanel stats={stats} ranked={rankedDomains} />
              {stats.length >= 2 && <OverlapPanel stats={stats} />}
            </div>
          )}
        </section>

        {/* Narrator rail */}
        <aside className="w-[24rem] border-l border-base-300 bg-base-100 hidden lg:flex flex-col">
          <div className="px-4 py-3 border-b border-base-300">
            <div className="font-semibold text-sm">Narrator</div>
            <div className="text-[11px] text-base-content/50">
              Sees exactly the numbers on this canvas — nothing more.
            </div>
          </div>
          <div className="flex-1 overflow-y-auto px-4 py-4">
            {chat.messages.length === 0 ? (
              <div className="space-y-2 mt-2">
                {stats.length === 0 ? (
                  <p className="text-xs text-base-content/50">
                    Add cohorts to the canvas, then ask the narrator about what appears.
                  </p>
                ) : (
                  narratorChips.map(s => (
                    <button
                      key={s}
                      disabled={blocked}
                      onClick={() => ask(s)}
                      className="w-full text-left rounded-xl border border-base-300 bg-base-100 p-2.5 hover:border-primary hover:shadow-sm transition-all text-xs disabled:opacity-50"
                    >
                      {s}
                    </button>
                  ))
                )}
              </div>
            ) : (
              <MessageList messages={chat.messages} streaming={chat.isStreaming} />
            )}
            {chat.error && (
              <div className="alert alert-error mt-3 text-xs">
                <span>{chat.error}</span>
              </div>
            )}
          </div>
          <div className="border-t border-base-300 px-3 py-3">
            <Composer
              value={chat.input}
              onChange={chat.setInput}
              onSend={() => ask()}
              onStop={chat.stop}
              isStreaming={chat.isStreaming}
              disabled={blocked || stats.length === 0}
              placeholder={stats.length ? 'Ask about this canvas…' : 'Add cohorts first…'}
            />
          </div>
        </aside>
      </div>
    </main>
  );
}
