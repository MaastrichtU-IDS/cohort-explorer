'use client';

// No-code DCR wizard — the no-code path to a Data Clean Room for domain
// experts (the other path being the Flexible DCR, i.e. the traditional wizard). Rendered inside the "Create analysis DCR" modal (Nav.tsx) behind
// the Flexible / No-code chooser, and on its own page (/nocode-dcr).
//
// The user says what they want to learn (an analysis class), picks cohorts,
// chooses the data source (full data or shuffled samples) and participants,
// picks/harmonizes variables in the Mapping Workbench, and creates a DCR whose
// generated node computes figures and tables. Results come back into the
// explorer (see /nocode-results).
import React, {useCallback, useEffect, useMemo, useState} from 'react';
import Link from 'next/link';
import {ArrowLeft, ArrowRight, Check, HelpCircle, AlertTriangle, Users} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {ParticipantsModal, useOwnersIncludedByDefault} from '@/components/ParticipantsModal';
import MappingWorkbench, {RoleDef} from '@/components/nocode/MappingWorkbench';
import KindExplainer from '@/components/nocode/KindExplainer';
import {KindGlyph} from '@/components/nocode/MiniChart';
import {AnalysisSpec, Kind, KindMeta, MappingSpec, createNocodeDcr, describeSpec, fetchKinds, provenanceLine} from '@/components/nocode/client';
import {apiUrl} from '@/utils';


const ROLE_LABELS: Record<Kind, RoleDef[]> = {
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
  compare: [
    {key: 'variable', label: 'Variable to compare and pool', hint: 'Harmonize it across the cohorts'},
    {key: 'group', label: 'Break down by', optional: true, kind: 'categorical', hint: 'Optional harmonized grouping'}
  ]
};

const STEPS = ['Choose analysis type', 'Cohorts', 'Data & participants', 'Variables & harmonization', 'Settings', 'Review & create'];

// "weight by sex in TIME-CHF", "sex across TIME-CHF and Aachen-HF", ...
function defaultTitle(kind: Kind | null, mapping: MappingSpec, roles: Record<string, string>, cohorts: string[]): string {
  if (!kind) return '';
  const name = (role: string) => {
    const hv = mapping.variables.find(v => v.harmonized_name === roles[role]);
    if (!hv) return '';
    // prefer the raw variable name(s) the user picked; fall back to the harmonized label
    const raw = Array.from(new Set(Object.values(hv.members).map(m => m.var_name).filter(Boolean)));
    return raw.length === 1 ? raw[0] : hv.label || hv.harmonized_name;
  };
  const where = cohorts.length === 1 ? `in ${cohorts[0]}` : `across ${cohorts.slice(0, -1).join(', ')} and ${cohorts[cohorts.length - 1]}`;
  switch (kind) {
    case 'stratified':
      return `${name('variable')} by ${name('group')} ${where}`;
    case 'correlation':
      return `${name('x')} vs ${name('y')} ${where}`;
    case 'crosstab':
      return `${name('x')} × ${name('y')} ${where}`;
    case 'compare':
      return `${name('variable')} ${where}${roles.group ? `, by ${name('group')}` : ''}`;
    default:
      return '';
  }
}

export default function NocodeWizard({embedded = false, onClose}: {embedded?: boolean; onClose?: () => void}) {
  const {cohortsData, userEmail} = useCohorts();
  const [kinds, setKinds] = useState<Record<Kind, KindMeta> | null>(null);
  const [step, setStep] = useState(0);
  const [kind, setKind] = useState<Kind | null>(null);
  const [explain, setExplain] = useState<Kind | null>(null);
  const [cohorts, setCohorts] = useState<string[]>([]);
  const [mapping, setMapping] = useState<MappingSpec>({name: '', cohorts: [], variables: []});
  const [roleAssignments, setRoleAssignments] = useState<Record<string, string>>({});
  const [title, setTitle] = useState('');
  const [titleTouched, setTitleTouched] = useState(false);
  const [k, setK] = useState(0);
  const [bins, setBins] = useState(20);
  const [dcrName, setDcrName] = useState('');
  const [description, setDescription] = useState('');
  const [script, setScript] = useState('');
  const [showScript, setShowScript] = useState(false);
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Data source: the full cohort data (default) or the shuffled samples
  // (offered only when every selected cohort has one; code testing only).
  const [dataSource, setDataSource] = useState<'full' | 'shuffled'>('full');
  const [cohortsWithSamples, setCohortsWithSamples] = useState<string[]>([]);
  const allHaveSamples = cohorts.length > 0 && cohorts.every(c => cohortsWithSamples.includes(c));

  useEffect(() => {
    if (cohorts.length === 0) return;
    fetch(`${apiUrl}/check-shuffled-samples`, {
      method: 'POST',
      credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cohorts: Object.fromEntries(cohorts.map(c => [c, []]))})
    })
      .then(r => (r.ok ? r.json() : null))
      .then(r => setCohortsWithSamples(r?.cohorts_with_samples || []))
      .catch(() => setCohortsWithSamples([]));
  }, [cohorts]);
  useEffect(() => {
    if (!allHaveSamples && dataSource === 'shuffled') setDataSource('full');
  }, [allHaveSamples, dataSource]);

  // Participants (same model as the analysis DCR wizard: data owners are
  // opted in through the modal; additional analysts are added by email).
  const [participantsPreview, setParticipantsPreview] = useState<any>(null);
  const [loadingParticipants, setLoadingParticipants] = useState(false);
  const [showParticipantsModal, setShowParticipantsModal] = useState(false);
  const [additionalAnalysts, setAdditionalAnalysts] = useState<string[]>([]);
  const [newAnalystEmail, setNewAnalystEmail] = useState('');
  const [manuallyIncludedOwners, setManuallyIncludedOwners] = useState<string[]>([]);

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

  // Participants preview for the selected cohorts.
  useEffect(() => {
    if (cohorts.length === 0) {
      setParticipantsPreview(null);
      return;
    }
    let cancelled = false;
    setLoadingParticipants(true);
    fetch(`${apiUrl}/preview-dcr-participants`, {
      method: 'POST',
      credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cohorts: Object.fromEntries(cohorts.map(c => [c, []])), additional_analysts: additionalAnalysts})
    })
      .then(r => (r.ok ? r.json() : null))
      .then(r => {
        if (!cancelled) setParticipantsPreview(r?.participants || null);
      })
      .catch(() => null)
      .finally(() => {
        if (!cancelled) setLoadingParticipants(false);
      });
    return () => {
      cancelled = true;
    };
  }, [cohorts, additionalAnalysts]);

  const dataOwners = useMemo((): {email: string; cohorts: string[]}[] => {
    if (!participantsPreview) return [];
    const owners: Record<string, Set<string>> = {};
    Object.entries(participantsPreview).forEach(([email, roles]: [string, any]) => {
      (roles.data_owner_of || []).forEach((nodeId: string) => {
        const cohortName = nodeId.replace(/_metadata_dictionary$/, '').replace(/_shuffled_sample$/, '').replace(/-/g, ' ');
        (owners[email] = owners[email] || new Set()).add(cohortName);
      });
    });
    return Object.entries(owners).map(([email, set]) => ({email, cohorts: Array.from(set).sort()}));
  }, [participantsPreview]);
  // Data owners are included by default; unticked ones are sent as excluded.
  useOwnersIncludedByDefault(dataOwners, manuallyIncludedOwners, setManuallyIncludedOwners);
  const excludedDataOwners = useMemo(() => dataOwners.map(o => o.email).filter(e => !manuallyIncludedOwners.includes(e)), [dataOwners, manuallyIncludedOwners]);

  const addAnalyst = useCallback(() => {
    const email = newAnalystEmail.trim().toLowerCase();
    if (email && !additionalAnalysts.includes(email) && email !== userEmail) setAdditionalAnalysts([...additionalAnalysts, email]);
    setNewAnalystEmail('');
  }, [newAnalystEmail, additionalAnalysts, userEmail]);
  const removeAnalyst = useCallback((email: string) => setAdditionalAnalysts(prev => prev.filter(e => e !== email)), []);

  const rolesComplete = roles.every(r => {
    if (r.optional) return true;
    const name = roleAssignments[r.key];
    const v = mapping.variables.find(x => x.harmonized_name === name);
    return !!v && cohorts.every(c => v.members[c]?.var_name);
  });

  const autoTitle = defaultTitle(kind, mapping, roleAssignments, cohorts);
  const effectiveTitle = (titleTouched && title.trim()) || autoTitle || meta?.label || kind || '';

  const spec: AnalysisSpec | null = kind
    ? {
        analysis: {
          kind,
          title: effectiveTitle,
          suppression_k: k,
          bins,
          roles: Object.fromEntries(roles.map(r => [r.key, roleAssignments[r.key] || '']).filter(([, v]) => v))
        },
        cohorts,
        data_source: dataSource,
        mapping: {...mapping, cohorts, created_by: mapping.created_by || userEmail || undefined}
      }
    : null;

  useEffect(() => {
    if (step === 5 && spec) {
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
    if (step === 3) return rolesComplete;
    return true;
  };

  const create = async () => {
    if (!spec) return;
    setCreating(true);
    setError(null);
    try {
      const body = {
        cohorts: Object.fromEntries(cohorts.map(c => [c, []])),
        include_shuffled_samples: Object.fromEntries(cohorts.map(c => [c, dataSource === 'shuffled'])),
        airlock_settings: Object.fromEntries(cohorts.map(c => [c, 0])),
        additional_analysts: additionalAnalysts,
        excluded_data_owners: excludedDataOwners,
        selected_mapping_files: [],
        include_mapping_upload_slot: false,
        merge_use_shuffled: false,
        dcr_name: dcrName || `No-code: ${spec.analysis.title}`,
        research_question: description,
        nocode_analyses: [spec]
      };
      const r = await createNocodeDcr(body);
      setCreated(r);
    } catch (e: any) {
      setError(e.message || 'Creating the DCR failed');
    } finally {
      setCreating(false);
    }
  };

  // userEmail is '' while the session is still being verified (see
  // CohortsContext): show a spinner rather than flashing the login notice.
  if (userEmail === '') {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <span className="loading loading-spinner loading-lg"></span>
      </div>
    );
  }
  if (!userEmail) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <div className="alert alert-warning max-w-md">
          <AlertTriangle size={20} />
          <span>Please log in to use the no-code DCR analysis.</span>
        </div>
      </div>
    );
  }

  return (
    <main className={embedded ? '' : 'max-w-7xl mx-auto px-4 py-6'}>
      <div className="flex items-end justify-between flex-wrap gap-3 mb-4">
        <div>
          <h1 className={embedded ? 'text-xl font-bold' : 'text-2xl font-bold'}>No-code analysis DCR</h1>
          <p className="text-sm text-base-content/60 max-w-2xl">
            Describe the analysis you want in plain choices. The explorer builds a Data Clean Room that computes it on the real data and
            returns figures and tables. Every figure states which variable mapping produced it.
          </p>
        </div>
        {onClose ? (
          <button className="btn btn-sm btn-ghost" onClick={onClose}>
            ✕
          </button>
        ) : (
          <Link href="/dcrs" className="btn btn-sm btn-ghost">
            My DCRs
          </Link>
        )}
      </div>

      {/* Step rail */}
      <ol className="flex flex-wrap gap-1 mb-6">
        {STEPS.map((s, i) => (
          <li
            key={s}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${i === step ? 'bg-base-content text-base-100' : i < step ? 'bg-emerald-100 text-emerald-900' : 'bg-base-200 text-base-content/60'}`}
          >
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
        <div className="grid md:grid-cols-2 gap-3">
          {kinds &&
            (Object.entries(kinds) as [Kind, KindMeta][]).map(([key, m]) => {
              return (
                <div
                  key={key}
                  role="button"
                  tabIndex={0}
                  onClick={() => {
                    setKind(key);
                    setRoleAssignments({});
                    setMapping(mm => ({...mm, variables: []}));
                  }}
                  onKeyDown={e => e.key === 'Enter' && setKind(key)}
                  className={`text-left rounded-2xl border-2 p-4 transition-all hover:shadow-md cursor-pointer ${kind === key ? 'border-base-content bg-base-100' : 'border-base-300 bg-base-100/70'}`}
                >
                  <div className="flex items-center gap-3 mb-2">
                    <span className="rounded-lg bg-base-200/70 px-1.5 py-1 text-base-content/70">
                      <KindGlyph kind={key} />
                    </span>
                    <span className="font-semibold">{m.label}</span>
                  </div>
                  <p className="text-sm text-base-content/70">{m.blurb}</p>
                  <div className="flex items-center justify-between mt-2">
                    <p className="text-xs text-base-content/50">{m.min_cohorts === m.max_cohorts ? `${m.min_cohorts} cohort` : `${m.min_cohorts}–${m.max_cohorts} cohorts`}</p>
                    <button
                      type="button"
                      className="btn btn-ghost btn-xs gap-1"
                      onClick={e => {
                        e.stopPropagation();
                        setExplain(key);
                      }}
                    >
                      <HelpCircle size={13} /> What is this?
                    </button>
                  </div>
                </div>
              );
            })}
          {!kinds && <div className="text-sm text-base-content/50">Loading analysis types…</div>}
          {explain && kinds && <KindExplainer kind={explain} meta={kinds[explain]} onClose={() => setExplain(null)} />}
          {kinds && (
            <p className="md:col-span-2 text-sm text-base-content/50 mt-1">More analysis types will be added over time. If you need one that is not listed, let us know.</p>
          )}
        </div>
      )}

      {/* Step 1: cohorts */}
      {step === 1 && meta && (
        <div>
          <div className="flex justify-between mb-3">
            <button className="btn btn-ghost gap-1" onClick={() => setStep(s => s - 1)}>
              <ArrowLeft size={16} /> Back
            </button>
            <button className="btn btn-primary gap-1" disabled={!canNext()} onClick={() => setStep(s => s + 1)}>
              Next <ArrowRight size={16} />
            </button>
          </div>
          <p className="text-sm text-base-content/60 mb-3">
            Choose {meta.min_cohorts === meta.max_cohorts ? meta.min_cohorts : `${meta.min_cohorts} to ${meta.max_cohorts}`} cohort{meta.max_cohorts > 1 ? 's' : ''}. Only cohorts with uploaded
            variables can be analysed.
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
                  className={`text-left rounded-xl border-2 px-3 py-2 transition-colors disabled:opacity-40 ${on ? 'bg-amber-100 border-amber-400 text-amber-900' : 'border-base-300 bg-base-100 hover:border-base-content/40'}`}
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

      {/* Step 2: data source + participants */}
      {step === 2 && (
        <div className="max-w-2xl">
          {allHaveSamples && (
            <div className="mb-6">
              <p className="text-sm font-semibold mb-1">Data to analyse</p>
              <p className="text-xs text-base-content/60 mb-3">All selected cohorts have a shuffled sample, so you can choose.</p>
              <div className="flex gap-2">
                {(['full', 'shuffled'] as const).map(opt => (
                  <button
                    key={opt}
                    type="button"
                    onClick={() => setDataSource(opt)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${dataSource === opt ? 'bg-primary text-primary-content' : 'bg-base-200 text-base-content/70 hover:bg-base-300'}`}
                  >
                    {opt === 'full' ? 'Full datasets' : 'Shuffled samples'}
                  </button>
                ))}
              </div>
              {dataSource === 'shuffled' && (
                <div className="mt-3 space-y-2">
                  <div className="rounded-xl bg-violet-50 border border-violet-300 text-violet-900 p-3 text-sm">
                    <span className="font-semibold">Shuffled samples are for testing that the analysis runs, not for results.</span> They are a small
                    fragment of each dataset in which every column was shuffled independently, so the relationships between variables are
                    destroyed on purpose: no actual findings can be drawn from figures computed on them. Every figure will carry this notice.
                  </div>
                  <div className="rounded-xl bg-base-200 text-base-content/80 p-3 text-sm">
                    Shuffled samples are uploaded by the platform itself, so using them requires neither the permission nor the
                    participation of the data owners. The analysis can be run right after the room is created.
                  </div>
                </div>
              )}
            </div>
          )}
          <p className="text-sm font-semibold mb-1">Participants</p>
          <p className="text-sm text-base-content/70 mb-4">
            {dataSource === 'shuffled'
              ? 'With shuffled samples the data owners need not take part; you may still invite them or additional analysts.'
              : 'The data owners of the selected cohorts are invited to the room and asked to provision their data; you can also invite additional analysts.'}
          </p>
          <button className="btn btn-outline gap-2" onClick={() => setShowParticipantsModal(true)}>
            <Users size={16} /> Edit participants list
          </button>
          <div className="mt-4 p-3 bg-base-200 rounded-lg text-sm space-y-1">
            <p>
              <strong>Data owners invited:</strong> {loadingParticipants ? 'loading…' : manuallyIncludedOwners.length > 0 ? manuallyIncludedOwners.join(', ') : 'none (open the participants list to include them)'}
            </p>
            {excludedDataOwners.length > 0 && (
              <p>
                <strong>Data owners not invited:</strong> {excludedDataOwners.join(', ')}
              </p>
            )}
            <p>
              <strong>Additional analysts:</strong> {additionalAnalysts.length > 0 ? additionalAnalysts.join(', ') : 'none'}
            </p>
          </div>
        </div>
      )}

      {/* Step 3: variables & harmonization */}
      {step === 3 && kind && (
        <MappingWorkbench cohorts={cohorts} roles={roles} mapping={mapping} roleAssignments={roleAssignments} onMappingChange={setMapping} onRolesChange={setRoleAssignments} userEmail={userEmail} />
      )}

      {/* Step 4: settings */}
      {step === 4 && (
        <div className="max-w-xl space-y-4">
          <label className="block text-sm">
            Title of the analysis (shown on the figures)
            <input
              className="input input-bordered w-full"
              value={titleTouched ? title : autoTitle}
              onChange={e => {
                setTitleTouched(true);
                setTitle(e.target.value);
              }}
            />
            <span className="block text-xs text-base-content/60 mt-1">Built from the chosen variables and cohorts; edit freely.</span>
          </label>
          <label className="block text-sm">
            Small-cell suppression threshold (optional)
            <input type="number" min={0} className="input input-bordered w-32 ml-2" value={k} onChange={e => setK(parseInt(e.target.value || '0', 10))} />
            <span className="block text-xs text-base-content/60 mt-1">0 shows everything (the default). Set e.g. 5 to hide counts, bins and table cells below that number.</span>
          </label>
          <label className="block text-sm">
            Histogram bins (numeric variables)
            <input type="number" min={5} max={100} className="input input-bordered w-32 ml-2" value={bins} onChange={e => setBins(parseInt(e.target.value || '20', 10))} />
          </label>
        </div>
      )}

      {/* Step 5: review */}
      {step === 5 && spec && !created && (
        <div className="grid lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="rounded-xl border border-base-300 bg-base-100 p-4">
              <div className="text-[11px] uppercase tracking-wide text-base-content/50">What will be computed</div>
              <p className="mt-1 font-semibold">{spec.analysis.title}</p>
              <p className="text-sm text-base-content/80">{description || '…'}</p>
              <div className="mt-3 text-sm">
                <div className="font-semibold mb-1">Variables and mapping</div>
                <ul className="font-mono text-xs space-y-1">
                  {spec.mapping.variables.map(v => (
                    <li key={v.harmonized_name} className="break-words">
                      {provenanceLine(v)}
                    </li>
                  ))}
                </ul>
              </div>
              <div className="mt-3 text-xs text-base-content/60">
                Cohorts: {cohorts.join(', ')} · data: {dataSource === 'shuffled' ? 'SHUFFLED SAMPLES (code test only)' : 'full datasets'} · {k > 0 ? `suppression k = ${k}` : 'no suppression'} · bins = {bins}
                {spec.mapping.sources && spec.mapping.sources.length > 0 && <> · cached files consulted: {spec.mapping.sources.join(', ')}</>}
              </div>
              <div className="mt-2 text-xs text-base-content/60">
                Participants: {manuallyIncludedOwners.length} data owner{manuallyIncludedOwners.length === 1 ? '' : 's'} invited
                {additionalAnalysts.length > 0 && <>, analysts: {additionalAnalysts.join(', ')}</>}
              </div>
            </div>
            <label className="block text-sm">
              Name of the Data Clean Room
              <input className="input input-bordered w-full" value={dcrName} placeholder={`No-code: ${spec.analysis.title}`} onChange={e => setDcrName(e.target.value)} />
            </label>
            <div className="rounded-xl bg-amber-50 border border-amber-200 text-amber-900 p-3 text-sm">
              {dataSource === 'shuffled'
                ? 'The platform uploads the shuffled samples when the room is created, so the analysis can be run immediately afterwards. Figures will be marked as computed on shuffled samples.'
                : 'Creating the room invites the participants above, exactly like a regular analysis DCR. The analysis can run once the data owners have provisioned their data; you can then view the results here in the explorer.'}
            </div>
            <button className="btn btn-primary" onClick={create} disabled={creating}>
              {creating ? 'Creating the Data Clean Room…' : 'Create the Data Clean Room'}
            </button>
          </div>
          <div>
            <button className="btn btn-sm btn-ghost" onClick={() => setShowScript(s => !s)}>
              {showScript ? 'Hide' : 'Show'} the generated script
            </button>
            {showScript && <pre className="mt-2 text-[11px] leading-snug bg-base-200 rounded-lg p-3 overflow-auto max-h-[600px]">{script}</pre>}
          </div>
        </div>
      )}

      {created && (
        <div className="rounded-2xl bg-emerald-50 border border-emerald-300 text-emerald-900 p-5 max-w-2xl">
          <div className="font-bold text-lg mb-2">✅ Data Clean Room created</div>
          <p className="text-sm mb-3">{created.message}</p>
          {dataSource === 'shuffled' && <p className="text-sm mb-3 font-semibold">Shuffled samples are in place. You can run the analysis right away.</p>}
          <div className="flex flex-wrap gap-2">
            <a href={created.dcr_url} target="_blank" rel="noopener noreferrer" className="btn btn-sm btn-outline">
              Open on Decentriq
            </a>
            {(created.nocode_nodes || []).map((n: string) => (
              <Link key={n} href={`/nocode-results?dcr=${encodeURIComponent(created.dcr_id)}&node=${encodeURIComponent(n)}&title=${encodeURIComponent(spec?.analysis.title || '')}`} className="btn btn-sm btn-primary">
                Run & view results
              </Link>
            ))}
          </div>
          {created.merge_warnings && created.merge_warnings.length > 0 && <ul className="mt-3 text-xs list-disc ml-5">{created.merge_warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>}
          {onClose && (
            <button className="btn btn-sm btn-ghost mt-3" onClick={onClose}>
              Close
            </button>
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

      {showParticipantsModal && (
        <ParticipantsModal
          dataOwners={dataOwners}
          userEmail={userEmail}
          additionalAnalysts={additionalAnalysts}
          newAnalystEmail={newAnalystEmail}
          setNewAnalystEmail={setNewAnalystEmail}
          addAnalyst={addAnalyst}
          removeAnalyst={removeAnalyst}
          manuallyIncludedOwners={manuallyIncludedOwners}
          setManuallyIncludedOwners={setManuallyIncludedOwners}
          onClose={() => setShowParticipantsModal(false)}
          isLoading={loadingParticipants}
        />
      )}
    </main>
  );
}
