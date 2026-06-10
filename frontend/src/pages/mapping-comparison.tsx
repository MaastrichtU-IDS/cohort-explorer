'use client';

import React, {useState, useCallback, useMemo} from 'react';
import {GitMerge, Upload, X, ChevronDown, ChevronUp, Eye, EyeOff} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';

interface MappingEntry {
  target_study: string;
  target: string;
  source_label?: string;
  target_label?: string;
  mapping_relation?: string;
  harmonization_status?: string;
  sim_score?: number;
  [key: string]: any;
}

interface SourceVarData {
  from: string;
  mappings: MappingEntry[];
}

interface MappingFile {
  [sourceVar: string]: SourceVarData;
}

interface ParsedFile {
  name: string;
  data: MappingFile;
  pairs: Set<string>;
}

function pairKey(from: string, target: string) {
  return `${from} → ${target}`;
}

function parseFile(name: string, data: MappingFile): ParsedFile {
  const pairs = new Set<string>();
  for (const entry of Object.values(data)) {
    if (entry?.from && Array.isArray(entry.mappings)) {
      for (const m of entry.mappings) {
        if (m.target_study) pairs.add(pairKey(entry.from, m.target_study));
      }
    }
  }
  return {name, data, pairs};
}

function getSourceVarsForPair(file: ParsedFile, pair: string): Map<string, MappingEntry[]> {
  const [from, targetStudy] = pair.split(' → ');
  const result = new Map<string, MappingEntry[]>();
  for (const [varKey, entry] of Object.entries(file.data)) {
    if (entry.from !== from) continue;
    const relevant = entry.mappings.filter(m => m.target_study === targetStudy);
    if (relevant.length > 0) result.set(varKey, relevant);
  }
  return result;
}

interface AtomicPair {
  srcKey: string;
  srcLabel: string;
  tgtKey: string;
  tgtLabel: string;
  mapping_relation?: string;
  harmonization_status?: string;
  sim_score?: number;
}

function getAtomicPairs(file: ParsedFile, pair: string): AtomicPair[] {
  const [from, targetStudy] = pair.split(' → ');
  const result: AtomicPair[] = [];
  for (const [varKey, entry] of Object.entries(file.data)) {
    if (entry.from !== from) continue;
    for (const m of entry.mappings) {
      if (m.target_study !== targetStudy || !m.target) continue;
      result.push({
        srcKey: varKey,
        srcLabel: m.source_label || '',
        tgtKey: m.target,
        tgtLabel: m.target_label || '',
        mapping_relation: m.mapping_relation,
        harmonization_status: m.harmonization_status,
        sim_score: m.sim_score,
      });
    }
  }
  return result.sort((a, b) => a.srcKey.localeCompare(b.srcKey) || a.tgtKey.localeCompare(b.tgtKey));
}

type DiffStatus = 'both-same' | 'both-diff' | 'only-left' | 'only-right';

function getHarmonizationColor(status?: string) {
  if (!status) return '';
  const s = status.toLowerCase();
  if (s.includes('identical')) return 'text-success';
  if (s.includes('compatible')) return 'text-info';
  if (s.includes('partial')) return 'text-warning';
  if (s.includes('not applicable')) return 'text-error opacity-70';
  return '';
}

function mappingSignature(m: MappingEntry) {
  return `${m.target}|${m.mapping_relation}|${m.harmonization_status}`;
}

function FileDropZone({
  label,
  file,
  onLoad,
  onClear
}: {
  label: string;
  file: ParsedFile | null;
  onLoad: (f: ParsedFile) => void;
  onClear: () => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const processFile = useCallback(
    (f: File) => {
      setError(null);
      const reader = new FileReader();
      reader.onload = e => {
        try {
          const json = JSON.parse(e.target?.result as string);
          onLoad(parseFile(f.name, json));
        } catch {
          setError('Invalid JSON file');
        }
      };
      reader.readAsText(f);
    },
    [onLoad]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) processFile(f);
    },
    [processFile]
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) processFile(f);
    },
    [processFile]
  );

  return (
    <div className="flex-1 min-w-0">
      <div className="text-sm font-semibold mb-2 opacity-70 uppercase tracking-wide">{label}</div>
      {file ? (
        <div className="border border-success/40 bg-success/5 rounded-lg p-4 flex items-center gap-3">
          <div className="flex-1 min-w-0">
            <div className="font-medium truncate">{file.name}</div>
            <div className="text-xs opacity-60 mt-0.5">
              {Object.keys(file.data).length} source variables · {file.pairs.size} mapping pairs
            </div>
          </div>
          <button className="btn btn-ghost btn-sm btn-square" onClick={onClear}>
            <X size={16} />
          </button>
        </div>
      ) : (
        <label
          className={`flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-8 cursor-pointer transition-colors ${
            dragging ? 'border-primary bg-primary/5' : 'border-base-300 hover:border-primary/50'
          }`}
          onDragOver={e => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <Upload size={28} className="opacity-40 mb-2" />
          <span className="text-sm opacity-60">Drop a JSON file or click to browse</span>
          {error && <span className="text-error text-xs mt-2">{error}</span>}
          <input type="file" accept=".json" className="hidden" onChange={onInputChange} />
        </label>
      )}
    </div>
  );
}

function MappingRow({
  varKey,
  mappings,
  status,
  isStale,
  staleTargets
}: {
  varKey: string;
  mappings: MappingEntry[];
  status: DiffStatus;
  isStale?: boolean;
  staleTargets?: Set<string>;
}) {
  const [expanded, setExpanded] = useState(false);
  const bgClass = isStale
    ? 'bg-orange-50 dark:bg-orange-900/20 border-orange-300 dark:border-orange-700'
    : status === 'only-left'
      ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
      : status === 'only-right'
        ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800'
        : status === 'both-diff'
          ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
          : 'bg-base-200/40 border-base-300';

  const label = mappings[0]?.source_label || varKey;

  return (
    <div className={`border rounded-lg mb-1.5 overflow-hidden text-sm ${bgClass}`}>
      <div
        className="flex items-center gap-2 px-3 py-2 cursor-pointer select-none"
        onClick={() => setExpanded(v => !v)}
      >
        <span className="font-mono text-xs opacity-60 shrink-0">{varKey}</span>
        <span className="truncate flex-1">{label}</span>
        {isStale && <span className="badge badge-xs badge-warning shrink-0">stale</span>}
        <span className="badge badge-sm shrink-0">{mappings.length}</span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>
      {expanded && (
        <div className="border-t border-current/10 divide-y divide-current/10">
          {mappings.map((m, i) => {
            const targetStale = !!(staleTargets && m.target && staleTargets.has(m.target.toLowerCase()));
            return (
              <div key={i} className={`px-3 py-2 flex flex-col gap-0.5 ${targetStale ? 'bg-orange-50/60 dark:bg-orange-900/10' : ''}`}>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-mono text-xs font-semibold">{m.target}</span>
                  {targetStale && <span className="badge badge-xs badge-warning">stale</span>}
                  <span className="text-xs opacity-70 truncate">{m.target_label}</span>
                </div>
                <div className="flex items-center gap-2 flex-wrap mt-0.5">
                  <span className="badge badge-outline badge-xs">{m.mapping_relation}</span>
                  <span className={`text-xs ${getHarmonizationColor(m.harmonization_status)}`}>
                    {m.harmonization_status}
                  </span>
                  {m.sim_score !== undefined && m.sim_score !== null && (
                    <span className="text-xs opacity-50">score: {m.sim_score}</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}


function CoverageBar({value, total, label}: {value: number; total: number; label: string}) {
  const p = Math.round((value / total) * 100);
  return (
    <div className="text-xs">
      <span className="opacity-70">{label}: </span>
      <span className="font-semibold">{value}/{total}</span>
      <span className="opacity-60"> ({p}%)</span>
    </div>
  );
}

function UnmappedPanel({
  title,
  vars,
  accent
}: {
  title: string;
  vars: {key: string; label?: string}[];
  accent: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex-1 min-w-0">
      <button
        className={`btn btn-xs btn-outline gap-1 w-full justify-start ${accent}`}
        onClick={() => setOpen(v => !v)}
      >
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        {title} ({vars.length})
      </button>
      {open && (
        <div className="mt-1 border border-base-300 rounded-lg p-2 max-h-48 overflow-y-auto space-y-0.5">
          {vars.length === 0 ? (
            <span className="text-xs opacity-40 italic">none</span>
          ) : (
            vars.map(v => (
              <div key={v.key} className="text-xs flex gap-1.5">
                <span className="font-mono opacity-60 shrink-0">{v.key}</span>
                {v.label && <span className="opacity-70 truncate">{v.label}</span>}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function CoverageDiffPanel({
  title,
  items,
  accent
}: {
  title: string;
  items: {key: string; label: string}[];
  accent: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        className={`btn btn-xs btn-outline gap-1 w-full justify-start ${accent}`}
        onClick={() => setOpen(v => !v)}
      >
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        {title} ({items.length})
      </button>
      {open && (
        <div className="mt-1 border border-base-300 rounded-lg p-2 max-h-48 overflow-y-auto space-y-0.5">
          {items.length === 0 ? (
            <span className="text-xs opacity-40 italic">none</span>
          ) : (
            items.map(item => (
              <div key={item.key} className="text-xs flex gap-2">
                <span className="font-mono opacity-70 shrink-0">{item.key}</span>
                {item.label && <span className="opacity-60 truncate">{item.label}</span>}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function ProminentCoverage({label, value, total}: {label: string; value: number; total: number | null}) {
  const pct = total ? Math.round((value / total) * 100) : null;
  return (
    <div className="mb-2">
      <div className="flex justify-between items-baseline">
        <span className="text-xs opacity-60">{label}</span>
        <span className="text-base font-bold leading-none">
          {value}
          {total != null && <span className="text-xs font-normal opacity-50">/{total}</span>}
          {pct != null && <span className="text-xs font-semibold opacity-70 ml-1">{pct}%</span>}
        </span>
      </div>
      {pct != null && (
        <div className="h-1.5 bg-black/10 dark:bg-white/10 rounded-full mt-1">
          <div className="h-full rounded-full bg-current opacity-50" style={{width: `${pct}%`}} />
        </div>
      )}
    </div>
  );
}

function AtomRow({atom}: {atom: AtomicPair & {presence: 'f1' | 'f2' | 'both'}}) {
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 text-xs border rounded ${
      atom.presence === 'both' ? 'border-base-300 bg-base-200/40'
      : atom.presence === 'f1' ? 'border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20'
      : 'border-purple-200 dark:border-purple-800 bg-purple-50 dark:bg-purple-900/20'
    }`}>
      <div className="flex-1 min-w-0 flex items-center gap-1 flex-wrap">
        <span className="font-mono">{atom.srcKey}</span>
        {atom.srcLabel && <span className="opacity-50">({atom.srcLabel})</span>}
        <span className="opacity-40 mx-0.5">→</span>
        <span className="font-mono font-semibold">{atom.tgtKey}</span>
        {atom.tgtLabel && <span className="opacity-50">({atom.tgtLabel})</span>}
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        {atom.presence === 'f1' && <span className="badge badge-xs badge-outline" style={{borderColor:'#93c5fd',color:'#2563eb'}}>F1</span>}
        {atom.presence === 'f2' && <span className="badge badge-xs badge-outline" style={{borderColor:'#c4b5fd',color:'#7c3aed'}}>F2</span>}
        {atom.presence === 'both' && <span className="badge badge-xs badge-info">both</span>}
        {atom.mapping_relation && <span className="badge badge-outline badge-xs">{atom.mapping_relation}</span>}
        {atom.harmonization_status && <span className={`shrink-0 ${getHarmonizationColor(atom.harmonization_status)}`}>{atom.harmonization_status}</span>}
      </div>
    </div>
  );
}

function GroupedSection({groupKey, groupLabel, atoms, kind}: {
  groupKey: string; groupLabel: string;
  atoms: (AtomicPair & {presence: 'f1' | 'f2' | 'both'})[];
  kind: 'src' | 'tgt';
}) {
  const [open, setOpen] = useState(true);
  const f1Count = atoms.filter(a => a.presence !== 'f2').length;
  const f2Count = atoms.filter(a => a.presence !== 'f1').length;
  return (
    <div className="border border-base-300 rounded-lg overflow-hidden mb-1">
      <div className="flex items-center gap-2 px-3 py-2 bg-base-200/50 cursor-pointer select-none"
        onClick={() => setOpen(v => !v)}>
        {open ? <ChevronUp size={13}/> : <ChevronDown size={13}/>}
        <span className="font-mono text-xs font-semibold">{groupKey}</span>
        {groupLabel && <span className="text-xs opacity-60 truncate flex-1">{groupLabel}</span>}
        <span className="badge badge-sm">{atoms.length}</span>
        <span className="badge badge-xs badge-outline" style={{borderColor:'#93c5fd',color:'#2563eb'}}>{f1Count} F1</span>
        <span className="badge badge-xs badge-outline" style={{borderColor:'#c4b5fd',color:'#7c3aed'}}>{f2Count} F2</span>
      </div>
      {open && (
        <div className="divide-y divide-base-200">
          {atoms.map((atom, i) => (
            <div key={i} className={`flex items-center gap-2 px-3 py-1.5 text-xs ${
              atom.presence === 'f1' ? 'bg-blue-50 dark:bg-blue-900/20'
              : atom.presence === 'f2' ? 'bg-purple-50 dark:bg-purple-900/20' : ''
            }`}>
              <span className="font-mono">{kind === 'src' ? atom.tgtKey : atom.srcKey}</span>
              <span className="opacity-50 truncate flex-1">{kind === 'src' ? atom.tgtLabel : atom.srcLabel}</span>
              <div className="flex items-center gap-1 shrink-0">
                {atom.presence === 'f1' && <span className="badge badge-xs badge-outline" style={{borderColor:'#93c5fd',color:'#2563eb'}}>F1</span>}
                {atom.presence === 'f2' && <span className="badge badge-xs badge-outline" style={{borderColor:'#c4b5fd',color:'#7c3aed'}}>F2</span>}
                {atom.presence === 'both' && <span className="badge badge-xs badge-info">both</span>}
                {atom.mapping_relation && <span className="badge badge-outline badge-xs">{atom.mapping_relation}</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function FullSidePanel({
  fileLabel,
  fileTag,
  sourceVars,
  allSourceVars,
  side,
  otherVars,
  showSame,
  fromStudy,
  targetStudy,
  cohortsData
}: {
  fileLabel: string;
  fileTag: string;
  sourceVars: Map<string, MappingEntry[]>;
  allSourceVars: string[];
  side: 'left' | 'right';
  otherVars: Map<string, MappingEntry[]>;
  showSame: boolean;
  fromStudy: string;
  targetStudy: string;
  cohortsData: Record<string, any>;
}) {
  const targetVarSet = useMemo(() => {
    const s = new Set<string>();
    for (const ms of sourceVars.values()) for (const m of ms) if (m.target) s.add(m.target);
    return s;
  }, [sourceVars]);

  const sourceCohort = useMemo(
    () => Object.values(cohortsData).find((c: any) => c.cohort_id?.toLowerCase() === fromStudy.toLowerCase()) as any,
    [cohortsData, fromStudy]
  );
  const targetCohort = useMemo(
    () => Object.values(cohortsData).find((c: any) => c.cohort_id?.toLowerCase() === targetStudy.toLowerCase()) as any,
    [cohortsData, targetStudy]
  );

  const totalSourceVars = sourceCohort ? Object.keys(sourceCohort.variables || {}).length : null;
  const totalTargetVars = targetCohort ? Object.keys(targetCohort.variables || {}).length : null;

  const unmappedSourceVars = useMemo(() => {
    if (!sourceCohort) return [];
    const mappedLower = new Set([...sourceVars.keys()].filter(Boolean).map(k => k.toLowerCase()));
    return Object.keys(sourceCohort.variables || {})
      .filter(k => !mappedLower.has(k.toLowerCase()))
      .map(k => ({key: k, label: sourceCohort.variables[k]?.var_label}));
  }, [sourceCohort, sourceVars]);

  const unmappedTargetVars = useMemo(() => {
    if (!targetCohort) return [];
    const coveredLower = new Set([...targetVarSet].filter(Boolean).map(k => k.toLowerCase()));
    return Object.keys(targetCohort.variables || {})
      .filter(k => !coveredLower.has(k.toLowerCase()))
      .map(k => ({key: k, label: targetCohort.variables[k]?.var_label}));
  }, [targetCohort, targetVarSet]);

  const sourceCohortVarsLower = useMemo(
    () => new Set(Object.keys(sourceCohort?.variables || {}).map((k: string) => k.toLowerCase())),
    [sourceCohort]
  );
  const targetCohortVarsLower = useMemo(
    () => new Set(Object.keys(targetCohort?.variables || {}).map((k: string) => k.toLowerCase())),
    [targetCohort]
  );

  const headerBg = side === 'left' ? 'bg-blue-100 dark:bg-blue-900/30' : 'bg-purple-100 dark:bg-purple-900/30';
  const accentBtn = side === 'left' ? 'border-blue-400 text-blue-600' : 'border-purple-400 text-purple-600';

  return (
    <div className="flex-1 min-w-0 flex flex-col">
      <div className={`rounded-lg p-3 mb-3 ${headerBg}`}>
        <div className="flex items-center gap-2">
          <span className="badge badge-sm font-bold">{fileTag}</span>
          <span className="font-semibold truncate text-sm">{fileLabel}</span>
        </div>
        <div className="mt-1.5 flex gap-4">
          <div className="flex-1 min-w-0 space-y-0.5">
            <CoverageBar value={sourceVars.size} total={totalSourceVars!} label="source vars mapped" />
            <CoverageBar value={targetVarSet.size} total={totalTargetVars!} label="target vars covered" />
          </div>
        </div>
        <div className="flex gap-2 mt-2">
          <UnmappedPanel
            title="Unmapped source vars"
            vars={unmappedSourceVars}
            accent={accentBtn}
          />
          <UnmappedPanel
            title="Uncovered target vars"
            vars={unmappedTargetVars}
            accent={accentBtn}
          />
        </div>
      </div>
      <div className="space-y-1">
        {allSourceVars.map(varKey => {
          const mappings = sourceVars.get(varKey);
          const inOther = otherVars.has(varKey);

          if (!mappings) {
            return (
              <div
                key={varKey}
                className="border border-dashed border-base-300 rounded-lg px-3 py-1.5 text-xs opacity-30 italic"
              >
                {varKey} — not present
              </div>
            );
          }

          let status: DiffStatus;
          if (!inOther) {
            status = side === 'left' ? 'only-left' : 'only-right';
          } else {
            const thisSigs = new Set(mappings.map(mappingSignature));
            const otherSigs = new Set((otherVars.get(varKey) || []).map(mappingSignature));
            status =
              JSON.stringify([...thisSigs].sort()) === JSON.stringify([...otherSigs].sort())
                ? 'both-same'
                : 'both-diff';
          }

          if (status === 'both-same' && !showSame) return null;

          const isStale = sourceCohortVarsLower.size > 0 && !sourceCohortVarsLower.has(varKey.toLowerCase());
          const staleTargets = new Set(
            mappings.filter(m => m.target && targetCohortVarsLower.size > 0 && !targetCohortVarsLower.has(m.target.toLowerCase())).map(m => m.target.toLowerCase())
          );

          return <MappingRow key={varKey} varKey={varKey} mappings={mappings} status={status} isStale={isStale} staleTargets={staleTargets} />;
        })}
      </div>
    </div>
  );
}

function ComparisonView({file1, file2, pair, cohortsData}: {file1: ParsedFile; file2: ParsedFile; pair: string; cohortsData: Record<string, any>}) {
  const [showSame, setShowSame] = useState(false);
  const [mode, setMode] = useState<'individual' | 'by-src' | 'by-tgt'>('individual');
  const [activeFilter, setActiveFilter] = useState<null | 'src-f1' | 'src-f2' | 'tgt-f1' | 'tgt-f2'>(null);

  const [fromStudy, targetStudy] = pair.split(' → ');

  const sourceCohort = useMemo(() => Object.values(cohortsData).find((c: any) => c.cohort_id?.toLowerCase() === fromStudy.toLowerCase()) as any, [cohortsData, fromStudy]);
  const targetCohort = useMemo(() => Object.values(cohortsData).find((c: any) => c.cohort_id?.toLowerCase() === targetStudy.toLowerCase()) as any, [cohortsData, targetStudy]);
  const totalSourceVars = sourceCohort ? Object.keys(sourceCohort.variables || {}).length : null;
  const totalTargetVars = targetCohort ? Object.keys(targetCohort.variables || {}).length : null;

  const f1Atoms = useMemo(() => getAtomicPairs(file1, pair), [file1, pair]);
  const f2Atoms = useMemo(() => getAtomicPairs(file2, pair), [file2, pair]);

  const f1SrcMap = useMemo(() => {
    const m = new Map<string, {key: string; label: string}>();
    for (const p of f1Atoms) { const k = p.srcKey.toLowerCase(); if (!m.has(k)) m.set(k, {key: p.srcKey, label: p.srcLabel}); }
    return m;
  }, [f1Atoms]);
  const f2SrcMap = useMemo(() => {
    const m = new Map<string, {key: string; label: string}>();
    for (const p of f2Atoms) { const k = p.srcKey.toLowerCase(); if (!m.has(k)) m.set(k, {key: p.srcKey, label: p.srcLabel}); }
    return m;
  }, [f2Atoms]);
  const f1TgtMap = useMemo(() => {
    const m = new Map<string, {key: string; label: string}>();
    for (const p of f1Atoms) { const k = p.tgtKey.toLowerCase(); if (!m.has(k)) m.set(k, {key: p.tgtKey, label: p.tgtLabel}); }
    return m;
  }, [f1Atoms]);
  const f2TgtMap = useMemo(() => {
    const m = new Map<string, {key: string; label: string}>();
    for (const p of f2Atoms) { const k = p.tgtKey.toLowerCase(); if (!m.has(k)) m.set(k, {key: p.tgtKey, label: p.tgtLabel}); }
    return m;
  }, [f2Atoms]);

  const srcOnlyF1 = useMemo(() =>
    [...f1SrcMap.values()].filter(v => !f2SrcMap.has(v.key.toLowerCase())).sort((a, b) => a.key.localeCompare(b.key)),
    [f1SrcMap, f2SrcMap]);
  const srcOnlyF2 = useMemo(() =>
    [...f2SrcMap.values()].filter(v => !f1SrcMap.has(v.key.toLowerCase())).sort((a, b) => a.key.localeCompare(b.key)),
    [f1SrcMap, f2SrcMap]);
  const tgtOnlyF1 = useMemo(() =>
    [...f1TgtMap.values()].filter(v => !f2TgtMap.has(v.key.toLowerCase())).sort((a, b) => a.key.localeCompare(b.key)),
    [f1TgtMap, f2TgtMap]);
  const tgtOnlyF2 = useMemo(() =>
    [...f2TgtMap.values()].filter(v => !f1TgtMap.has(v.key.toLowerCase())).sort((a, b) => a.key.localeCompare(b.key)),
    [f1TgtMap, f2TgtMap]);

  const unifiedAtoms = useMemo(() => {
    const f1Sig = new Map<string, AtomicPair>();
    for (const p of f1Atoms) f1Sig.set(`${p.srcKey.toLowerCase()}|${p.tgtKey.toLowerCase()}`, p);
    const f2Sig = new Map<string, AtomicPair>();
    for (const p of f2Atoms) f2Sig.set(`${p.srcKey.toLowerCase()}|${p.tgtKey.toLowerCase()}`, p);
    const allSigs = new Set([...f1Sig.keys(), ...f2Sig.keys()]);
    const result: (AtomicPair & {presence: 'f1' | 'f2' | 'both'})[] = [];
    for (const sig of allSigs) {
      const inF1 = f1Sig.has(sig);
      const inF2 = f2Sig.has(sig);
      const base = (inF1 ? f1Sig.get(sig) : f2Sig.get(sig))!;
      result.push({...base, presence: inF1 && inF2 ? 'both' : inF1 ? 'f1' : 'f2'});
    }
    return result.sort((a, b) => a.srcKey.localeCompare(b.srcKey) || a.tgtKey.localeCompare(b.tgtKey));
  }, [f1Atoms, f2Atoms]);

  const filteredAtoms = useMemo(() => {
    let atoms = showSame ? unifiedAtoms : unifiedAtoms.filter(a => a.presence !== 'both');
    if (activeFilter === 'src-f1') { const ks = new Set(srcOnlyF1.map(v => v.key.toLowerCase())); atoms = atoms.filter(a => ks.has(a.srcKey.toLowerCase())); }
    else if (activeFilter === 'src-f2') { const ks = new Set(srcOnlyF2.map(v => v.key.toLowerCase())); atoms = atoms.filter(a => ks.has(a.srcKey.toLowerCase())); }
    else if (activeFilter === 'tgt-f1') { const ks = new Set(tgtOnlyF1.map(v => v.key.toLowerCase())); atoms = atoms.filter(a => ks.has(a.tgtKey.toLowerCase())); }
    else if (activeFilter === 'tgt-f2') { const ks = new Set(tgtOnlyF2.map(v => v.key.toLowerCase())); atoms = atoms.filter(a => ks.has(a.tgtKey.toLowerCase())); }
    return atoms;
  }, [unifiedAtoms, showSame, activeFilter, srcOnlyF1, srcOnlyF2, tgtOnlyF1, tgtOnlyF2]);

  const groupedBySrc = useMemo(() => {
    const m = new Map<string, (AtomicPair & {presence: 'f1'|'f2'|'both'})[]>();
    for (const a of filteredAtoms) { if (!m.has(a.srcKey)) m.set(a.srcKey, []); m.get(a.srcKey)!.push(a); }
    return [...m.entries()].sort((a, b) => b[1].length - a[1].length);
  }, [filteredAtoms]);

  const groupedByTgt = useMemo(() => {
    const m = new Map<string, (AtomicPair & {presence: 'f1'|'f2'|'both'})[]>();
    for (const a of filteredAtoms) { if (!m.has(a.tgtKey)) m.set(a.tgtKey, []); m.get(a.tgtKey)!.push(a); }
    return [...m.entries()].sort((a, b) => b[1].length - a[1].length);
  }, [filteredAtoms]);

  type FilterKey = 'src-f1' | 'src-f2' | 'tgt-f1' | 'tgt-f2';
  const filterBtn = (key: FilterKey, label: string, count: number, style: string) => (
    <button
      key={key}
      className={`btn btn-xs ${activeFilter === key ? 'btn-neutral' : `btn-outline ${style}`}`}
      onClick={() => setActiveFilter(v => v === key ? null : key)}
    >
      {label} <span className="badge badge-xs ml-1">{count}</span>
    </button>
  );

  return (
    <div className="mt-4">
      {/* Prominent coverage stats */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="rounded-lg p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
          <div className="flex items-center gap-2 mb-2">
            <span className="badge badge-sm font-bold">F1</span>
            <span className="text-xs font-semibold truncate">{file1.name}</span>
          </div>
          <ProminentCoverage label="Source vars mapped" value={f1SrcMap.size} total={totalSourceVars} />
          <ProminentCoverage label="Target vars covered" value={f1TgtMap.size} total={totalTargetVars} />
        </div>
        <div className="rounded-lg p-3 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800">
          <div className="flex items-center gap-2 mb-2">
            <span className="badge badge-sm font-bold">F2</span>
            <span className="text-xs font-semibold truncate">{file2.name}</span>
          </div>
          <ProminentCoverage label="Source vars mapped" value={f2SrcMap.size} total={totalSourceVars} />
          <ProminentCoverage label="Target vars covered" value={f2TgtMap.size} total={totalTargetVars} />
        </div>
      </div>

      {/* Filter buttons */}
      <div className="flex items-center gap-2 flex-wrap mb-3">
        <span className="text-xs opacity-50 font-semibold uppercase tracking-wide mr-1">Filter:</span>
        {filterBtn('src-f1', 'Source vars in F1 not in F2', srcOnlyF1.length, 'border-blue-400 text-blue-600')}
        {filterBtn('src-f2', 'Source vars in F2 not in F1', srcOnlyF2.length, 'border-purple-400 text-purple-600')}
        {filterBtn('tgt-f1', 'Target vars in F1 not in F2', tgtOnlyF1.length, 'border-blue-400 text-blue-600')}
        {filterBtn('tgt-f2', 'Target vars in F2 not in F1', tgtOnlyF2.length, 'border-purple-400 text-purple-600')}
        {activeFilter && (
          <button className="btn btn-xs btn-ghost gap-1" onClick={() => setActiveFilter(null)}>
            <X size={10} /> Clear filter
          </button>
        )}
      </div>

      {/* Mode + show-same toolbar */}
      <div className="flex items-center gap-3 flex-wrap mb-3">
        <div className="join">
          <button className={`join-item btn btn-xs ${mode === 'individual' ? 'btn-primary' : 'btn-outline'}`} onClick={() => setMode('individual')}>Individual</button>
          <button className={`join-item btn btn-xs ${mode === 'by-src' ? 'btn-primary' : 'btn-outline'}`} onClick={() => setMode('by-src')}>By source var</button>
          <button className={`join-item btn btn-xs ${mode === 'by-tgt' ? 'btn-primary' : 'btn-outline'}`} onClick={() => setMode('by-tgt')}>By target var</button>
        </div>
        <button
          className={`btn btn-xs gap-1 ml-auto ${showSame ? 'btn-neutral' : 'btn-outline'}`}
          onClick={() => setShowSame(v => !v)}
        >
          {showSame ? <EyeOff size={12} /> : <Eye size={12} />}
          Show mappings identical in F1 and F2
        </button>
        <span className="text-xs opacity-40">{filteredAtoms.length} pairs</span>
      </div>

      {/* Legend */}
      <div className="text-xs opacity-50 mb-2 flex gap-3 flex-wrap">
        <span><span className="badge badge-xs mr-0.5" style={{borderColor:'#93c5fd',color:'#2563eb'}}>F1</span> only in F1</span>
        <span><span className="badge badge-xs mr-0.5" style={{borderColor:'#c4b5fd',color:'#7c3aed'}}>F2</span> only in F2</span>
        <span><span className="badge badge-xs badge-info mr-0.5">both</span> in both files</span>
      </div>

      {/* Individual mode */}
      {mode === 'individual' && (
        <div className="space-y-0.5">
          {filteredAtoms.map((atom, i) => <AtomRow key={i} atom={atom} />)}
        </div>
      )}

      {/* By source var mode */}
      {mode === 'by-src' && (
        <div>
          {groupedBySrc.map(([key, atoms]) => (
            <GroupedSection key={key} groupKey={key} groupLabel={atoms[0]?.srcLabel || ''} atoms={atoms} kind="src" />
          ))}
        </div>
      )}

      {/* By target var mode */}
      {mode === 'by-tgt' && (
        <div>
          {groupedByTgt.map(([key, atoms]) => (
            <GroupedSection key={key} groupKey={key} groupLabel={atoms[0]?.tgtLabel || ''} atoms={atoms} kind="tgt" />
          ))}
        </div>
      )}
    </div>
  );
}

export default function MappingComparison() {
  const {cohortsData} = useCohorts();
  const [file1, setFile1] = useState<ParsedFile | null>(null);
  const [file2, setFile2] = useState<ParsedFile | null>(null);
  const [selectedPair, setSelectedPair] = useState<string | null>(null);

  const commonPairs = useMemo(() => {
    if (!file1 || !file2) return [];
    return [...file1.pairs].filter(p => file2.pairs.has(p)).sort();
  }, [file1, file2]);

  const onlyInFile1 = useMemo(() => {
    if (!file1 || !file2) return [];
    return [...file1.pairs].filter(p => !file2.pairs.has(p)).sort();
  }, [file1, file2]);

  const onlyInFile2 = useMemo(() => {
    if (!file1 || !file2) return [];
    return [...file2.pairs].filter(p => !file1.pairs.has(p)).sort();
  }, [file1, file2]);

  const handleFile1Load = useCallback((f: ParsedFile) => { setFile1(f); setSelectedPair(null); }, []);
  const handleFile2Load = useCallback((f: ParsedFile) => { setFile2(f); setSelectedPair(null); }, []);

  return (
    <div className="container mx-auto px-4 py-6 max-w-7xl">
      <div className="flex items-center gap-3 mb-6">
        <GitMerge size={28} className="text-primary" />
        <div>
          <h1 className="text-2xl font-bold">Mapping Comparison</h1>
          <p className="text-sm opacity-60">Compare two mapping JSON files side-by-side</p>
        </div>
      </div>

      {/* File Upload Row */}
      <div className="flex gap-4 mb-6">
        <FileDropZone label="F1 — File 1" file={file1} onLoad={handleFile1Load} onClear={() => { setFile1(null); setSelectedPair(null); }} />
        <FileDropZone label="F2 — File 2" file={file2} onLoad={handleFile2Load} onClear={() => { setFile2(null); setSelectedPair(null); }} />
      </div>

      {/* Pair Selector */}
      {file1 && file2 && (
        <div className="card bg-base-100 border border-base-300 p-4 mb-6">
          <div className="font-semibold mb-3 text-sm">Detected Mapping Pairs</div>

          {commonPairs.length > 0 && (
            <div className="mb-3">
              <div className="text-xs opacity-60 mb-1 uppercase tracking-wide">Common pairs ({commonPairs.length})</div>
              <div className="text-xs opacity-50 italic mb-2">Only these pairs exist in both F1 and F2 and can be directly compared. Click a pair to open the comparison.</div>
              <div className="flex flex-wrap gap-2">
                {commonPairs.map(pair => (
                  <button
                    key={pair}
                    className={`btn btn-sm gap-1 ${selectedPair === pair ? 'btn-primary' : 'btn-outline'}`}
                    onClick={() => setSelectedPair(selectedPair === pair ? null : pair)}
                  >
                    {pair}
                  </button>
                ))}
              </div>
            </div>
          )}

          {(onlyInFile1.length > 0 || onlyInFile2.length > 0) && (
            <div className="flex gap-6 flex-wrap mt-1">
              {onlyInFile1.length > 0 && (
                <div>
                  <div className="text-xs opacity-50 mb-1 uppercase tracking-wide">Only in F1 ({onlyInFile1.length})</div>
                  <div className="flex flex-wrap gap-1.5">
                    {onlyInFile1.map(p => (
                      <span key={p} className="badge badge-sm badge-ghost">{p}</span>
                    ))}
                  </div>
                </div>
              )}
              {onlyInFile2.length > 0 && (
                <div>
                  <div className="text-xs opacity-50 mb-1 uppercase tracking-wide">Only in F2 ({onlyInFile2.length})</div>
                  <div className="flex flex-wrap gap-1.5">
                    {onlyInFile2.map(p => (
                      <span key={p} className="badge badge-sm badge-ghost">{p}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {commonPairs.length === 0 && (
            <div className="text-sm opacity-50 italic">No common mapping pairs found between the two files.</div>
          )}
        </div>
      )}

      {/* Comparison View */}
      {file1 && file2 && selectedPair && (
        <div className="card bg-base-100 border border-base-300 p-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">Comparing:</span>
            <span className="badge badge-primary">{selectedPair}</span>
          </div>
          <ComparisonView file1={file1} file2={file2} pair={selectedPair} cohortsData={cohortsData || {}} />
        </div>
      )}

      {!file1 && !file2 && (
        <div className="text-center py-16 opacity-40">
          <GitMerge size={48} className="mx-auto mb-3" />
          <p>Upload two mapping JSON files above to start comparing</p>
        </div>
      )}
    </div>
  );
}
