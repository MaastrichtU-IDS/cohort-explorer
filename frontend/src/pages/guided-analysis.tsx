'use client';

// Guided Analysis — a no-code path to a Data Clean Room for domain experts.
//
// The user says what they want to learn (an analysis class), picks cohorts and
// variables, harmonizes variables across cohorts in the Mapping Workbench, and
// creates a DCR whose generated node computes aggregates only. Results come
// back into the explorer (see /guided-results).
import React, {useEffect, useMemo, useState} from 'react';
import Link from 'next/link';
import {ArrowLeft, ArrowRight, BarChart2, Check, Compass, GitMerge, Grid, Layers, TrendingUp, AlertTriangle} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import MappingWorkbench, {RoleDef, cohortColor} from '@/components/guided/MappingWorkbench';
import {
  AnalysisSpec,
  Kind,
  KindMeta,
  MappingSpec,
  createGuidedDcr,
  describeSpec,
  fetchKinds,
  provenanceLine
} from '@/components/guided/client';

const KIND_ICONS: Record<Kind, any> = {
  distribution: BarChart2,
  stratified: Layers,
  correlation: TrendingUp,
  crosstab: Grid,
  compare: Compass,
  pooled: GitMerge
};

const ROLE_LABELS: Record<Kind, RoleDef[]> = {
  distribution: [{key: 'variable', label: 'Variable of interest', hint: 'The variable whose distribution you want to see'}],
  stratified: [
    {key: 'variable', label: 'Variable of interest', hint: 'What is being measured'},
    {key: 'group', label: 'Break down by', hint: 'The grouping variable (e.g. sex, diabetes)', kind: 'categorical'}
  ],
  correlation: [
    {key: 'x', label: 'First variable (x)', kind: 'numeric'},
    {key: 'y', label: 'Second variable (y)', kind: 'numeric'}
  ],
  crosstab: [
    {key: 'x', label: 'Rows', kind: 'categorical'},
    {key: 'y', label: 'Columns', kind: 'categorical'}
  ],
  compare: [{key: 'variable', label: 'Variable to compare', hint: 'Harmonize it across the cohorts'}],
  pooled: [
    {key: 'variable', label: 'Variable to pool', hint: 'Harmonize it across the cohorts'},
    {key: 'group', label: 'Break down by', optional: true, kind: 'categorical', hint: 'Optional harmonized grouping'}
  ]
};

const STEPS = ['What to learn', 'Cohorts', 'Variables & harmonization', 'Settings', 'Review & create'];

export default function GuidedAnalysisPage() {
  const {cohortsData, userEmail} = useCohorts();
  const [kinds, setKinds] = useState<Record<Kind, KindMeta> | null>(null);
  const [step, setStep] = useState(0);
  const [kind, setKind] = useState<Kind | null>(null);
  const [cohorts, setCohorts] = useState<string[]>([]);
  const [mapping, setMapping] = useState<MappingSpec>({name: '', cohorts: [], variables: []});
  const [roleAssignments, setRoleAssignments] = useState<Record<string, string>>({});
  const [title, setTitle] = useState('');
  const [k, setK] = useState(5);
  const [bins, setBins] = useState(20);
  const [dcrName, setDcrName] = useState('');
  const [description, setDescription] = useState('');
  const [script, setScript] = useState('');
  const [showScript, setShowScript] = useState(false);
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchKinds().then(r => setKinds(r.kinds)).catch(() => setKinds(null));
  }, []);

  const meta = kind && kinds ? kinds[kind] : null;
  const roles = kind ? ROLE_LABELS[kind] : [];
  const cohortList = useMemo(
    () =>
      Object.values(cohortsData || {})
        .map((c: any) => ({id: c.cohort_id, n: Object.keys(c.variables || {}).length, type: c.study_type}))
        .sort((a: any, b: any) => b.n - a.n),
    [cohortsData]
  );

  const toggleCohort = (id: string) => {
    setCohorts(prev => {
      const next = prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id];
      setMapping(m => ({...m, cohorts: next}));
      return next;
    });
  };

  const rolesComplete = roles.every(r => {
    if (r.optional) return true;
    const name = roleAssignments[r.key];
    const v = mapping.variables.find(x => x.harmonized_name === name);
    return !!v && cohorts.every(c => v.members[c]?.var_name);
  });

  const spec: AnalysisSpec | null = kind
    ? {
        analysis: {kind, title: title || (meta?.label ?? kind), suppression_k: k, bins, roles: Object.fromEntries(roles.map(r => [r.key, roleAssignments[r.key] || '']).filter(([, v]) => v))},
        cohorts,
        mapping: {...mapping, cohorts, created_by: mapping.created_by || userEmail || undefined}
      }
    : null;

  useEffect(() => {
    if (step === 4 && spec) {
      describeSpec(spec)
        .then(r => {
          setDescription(r.description);
          setScript(r.script);
        })
        .catch(e => setError(e.message));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  const canNext = () => {
    if (step === 0) return !!kind;
    if (step === 1) return !!meta && cohorts.length >= meta.min_cohorts && cohorts.length <= meta.max_cohorts;
    if (step === 2) return rolesComplete;
    return true;
  };

  const create = async () => {
    if (!spec) return;
    setCreating(true);
    setError(null);
    try {
      const body = {
        cohorts: Object.fromEntries(cohorts.map(c => [c, []])),
        include_shuffled_samples: Object.fromEntries(cohorts.map(c => [c, false])),
        airlock_settings: Object.fromEntries(cohorts.map(c => [c, 0])),
        additional_analysts: [],
        excluded_data_owners: [],
        selected_mapping_files: [],
        include_mapping_upload_slot: false,
        merge_use_shuffled: false,
        dcr_name: dcrName || `Guided: ${spec.analysis.title}`,
        research_question: description,
        guided_analyses: [spec]
      };
      const r = await createGuidedDcr(body);
      setCreated(r);
    } catch (e: any) {
      setError(e.message || 'Creating the DCR failed');
    } finally {
      setCreating(false);
    }
  };

  if (!userEmail) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <div className="alert alert-warning max-w-md">
          <AlertTriangle size={20} />
          <span>Please log in to use the guided analysis.</span>
        </div>
      </div>
    );
  }

  return (
    <main className="max-w-7xl mx-auto px-4 py-6">
      <div className="flex items-end justify-between flex-wrap gap-3 mb-4">
        <div>
          <h1 className="text-2xl font-bold">Guided Analysis</h1>
          <p className="text-sm text-base-content/60 max-w-2xl">
            Describe the analysis you want in plain choices. The explorer builds a Data Clean Room that computes it on the
            real data and returns only aggregate figures and tables — every figure states which variable mapping produced it.
          </p>
        </div>
        <Link href="/dcrs" className="btn btn-sm btn-ghost">
          My DCRs
        </Link>
      </div>

      {/* Step rail */}
      <ol className="flex flex-wrap gap-1 mb-6">
        {STEPS.map((s, i) => (
          <li key={s} className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${i === step ? 'bg-base-content text-base-100' : i < step ? 'bg-emerald-100 text-emerald-900' : 'bg-base-200 text-base-content/60'}`}>
            <span className="w-5 h-5 rounded-full bg-base-100/30 flex items-center justify-center text-xs">{i < step ? <Check size={12} /> : i + 1}</span>
            {s}
          </li>
        ))}
      </ol>

      {error && (
        <div className="alert alert-error mb-4 text-sm">
          <span>{error}</span>
          <button className="btn btn-ghost btn-xs" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* Step 0: kind */}
      {step === 0 && (
        <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">
          {kinds &&
            (Object.entries(kinds) as [Kind, KindMeta][]).map(([key, m]) => {
              const Icon = KIND_ICONS[key] || BarChart2;
              return (
                <button
                  key={key}
                  onClick={() => {
                    setKind(key);
                    setRoleAssignments({});
                    setMapping(mm => ({...mm, variables: []}));
                  }}
                  className={`text-left rounded-2xl border-2 p-4 transition-all hover:shadow-md ${kind === key ? 'border-base-content bg-base-100' : 'border-base-300 bg-base-100/70'}`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Icon size={20} />
                    <span className="font-semibold">{m.label}</span>
                  </div>
                  <p className="text-sm text-base-content/70">{m.blurb}</p>
                  <p className="text-xs text-base-content/50 mt-2">
                    {m.min_cohorts === m.max_cohorts ? `${m.min_cohorts} cohort` : `${m.min_cohorts}–${m.max_cohorts} cohorts`}
                  </p>
                </button>
              );
            })}
          {!kinds && <div className="text-sm text-base-content/50">Loading analysis types…</div>}
        </div>
      )}

      {/* Step 1: cohorts */}
      {step === 1 && meta && (
        <div>
          <p className="text-sm text-base-content/60 mb-3">
            Choose {meta.min_cohorts === meta.max_cohorts ? meta.min_cohorts : `${meta.min_cohorts} to ${meta.max_cohorts}`} cohort{meta.max_cohorts > 1 ? 's' : ''}. Only cohorts with uploaded variables can be analysed.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {cohortList.map((c: any) => {
              const on = cohorts.includes(c.id);
              const disabled = c.n === 0 || (!on && cohorts.length >= meta.max_cohorts);
              return (
                <button
                  key={c.id}
                  disabled={disabled}
                  onClick={() => toggleCohort(c.id)}
                  className={`text-left rounded-xl border-2 px-3 py-2 transition-colors disabled:opacity-40 ${on ? `${cohortColor(cohorts, c.id)} border-current` : 'border-base-300 bg-base-100 hover:border-base-content/40'}`}
                >
                  <div className="font-semibold">{c.id}</div>
                  <div className="text-xs opacity-70">
                    {c.n} variables{c.type ? ` · ${c.type}` : ''}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Step 2: variables & harmonization */}
      {step === 2 && kind && (
        <MappingWorkbench
          cohorts={cohorts}
          roles={roles}
          mapping={mapping}
          roleAssignments={roleAssignments}
          onMappingChange={setMapping}
          onRolesChange={setRoleAssignments}
          userEmail={userEmail}
        />
      )}

      {/* Step 3: settings */}
      {step === 3 && (
        <div className="max-w-xl space-y-4">
          <label className="block text-sm">
            Title of the analysis (shown on the figures)
            <input className="input input-bordered w-full" value={title} placeholder={meta?.label} onChange={e => setTitle(e.target.value)} />
          </label>
          <label className="block text-sm">
            Small-cell suppression threshold
            <input type="number" min={1} className="input input-bordered w-32 ml-2" value={k} onChange={e => setK(parseInt(e.target.value || '5', 10))} />
            <span className="block text-xs text-base-content/60 mt-1">Any count below this number is hidden; histogram bins and table cells below it are blanked. 5 is the usual minimum.</span>
          </label>
          <label className="block text-sm">
            Histogram bins (numeric variables)
            <input type="number" min={5} max={100} className="input input-bordered w-32 ml-2" value={bins} onChange={e => setBins(parseInt(e.target.value || '20', 10))} />
          </label>
        </div>
      )}

      {/* Step 4: review */}
      {step === 4 && spec && !created && (
        <div className="grid lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="rounded-xl border border-base-300 bg-base-100 p-4">
              <div className="text-[11px] uppercase tracking-wide text-base-content/50">What will be computed</div>
              <p className="mt-1">{description || '…'}</p>
              <div className="mt-3 text-sm">
                <div className="font-semibold mb-1">Variables and mapping</div>
                <ul className="font-mono text-xs space-y-1">
                  {spec.mapping.variables.map(v => (
                    <li key={v.harmonized_name} className="break-words">{provenanceLine(v)}</li>
                  ))}
                </ul>
              </div>
              <div className="mt-3 text-xs text-base-content/60">
                Cohorts: {cohorts.join(', ')} · suppression k = {k} · bins = {bins}
                {spec.mapping.sources && spec.mapping.sources.length > 0 && <> · cached files consulted: {spec.mapping.sources.join(', ')}</>}
              </div>
            </div>
            <label className="block text-sm">
              Name of the Data Clean Room
              <input className="input input-bordered w-full" value={dcrName} placeholder={`Guided: ${spec.analysis.title}`} onChange={e => setDcrName(e.target.value)} />
            </label>
            <div className="rounded-xl bg-amber-50 border border-amber-200 text-amber-900 p-3 text-sm">
              Creating the room invites the cohorts' data owners, exactly like a regular analysis DCR. The analysis can run
              once they have provisioned their data; you can then view the aggregate results here in the explorer.
            </div>
            <button className="btn btn-primary" onClick={create} disabled={creating}>
              {creating ? 'Creating the Data Clean Room…' : 'Create the Data Clean Room'}
            </button>
          </div>
          <div>
            <button className="btn btn-sm btn-ghost" onClick={() => setShowScript(s => !s)}>
              {showScript ? 'Hide' : 'Show'} the generated script (for the curious)
            </button>
            {showScript && <pre className="mt-2 text-[11px] leading-snug bg-base-200 rounded-lg p-3 overflow-auto max-h-[600px]">{script}</pre>}
          </div>
        </div>
      )}

      {created && (
        <div className="rounded-2xl bg-emerald-50 border border-emerald-300 text-emerald-900 p-5 max-w-2xl">
          <div className="font-bold text-lg mb-2">✅ Data Clean Room created</div>
          <p className="text-sm mb-3">{created.message}</p>
          <div className="flex flex-wrap gap-2">
            <a href={created.dcr_url} target="_blank" rel="noopener noreferrer" className="btn btn-sm btn-outline">
              Open on Decentriq
            </a>
            {(created.guided_nodes || []).map((n: string) => (
              <Link key={n} href={`/guided-results?dcr=${encodeURIComponent(created.dcr_id)}&node=${encodeURIComponent(n)}&title=${encodeURIComponent(spec?.analysis.title || '')}`} className="btn btn-sm btn-primary">
                Run & view results
              </Link>
            ))}
          </div>
          {created.merge_warnings && created.merge_warnings.length > 0 && (
            <ul className="mt-3 text-xs list-disc ml-5">{created.merge_warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>
          )}
        </div>
      )}

      {/* Nav */}
      {!created && (
        <div className="flex justify-between mt-6">
          <button className="btn btn-ghost gap-1" disabled={step === 0} onClick={() => setStep(s => s - 1)}>
            <ArrowLeft size={16} /> Back
          </button>
          {step < STEPS.length - 1 && (
            <button className="btn btn-primary gap-1" disabled={!canNext()} onClick={() => setStep(s => s + 1)}>
              Next <ArrowRight size={16} />
            </button>
          )}
        </div>
      )}
    </main>
  );
}
