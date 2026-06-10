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

function MappingRow({varKey, mappings, status}: {varKey: string; mappings: MappingEntry[]; status: DiffStatus}) {
  const [expanded, setExpanded] = useState(false);
  const bgClass =
    status === 'only-left'
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
        <span className="badge badge-sm shrink-0">{mappings.length}</span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>
      {expanded && (
        <div className="border-t border-current/10 divide-y divide-current/10">
          {mappings.map((m, i) => (
            <div key={i} className="px-3 py-2 flex flex-col gap-0.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono text-xs font-semibold">{m.target}</span>
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
          ))}
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
    for (const ms of sourceVars.values()) for (const m of ms) s.add(m.target);
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
    return Object.keys(sourceCohort.variables || {})
      .filter(k => !sourceVars.has(k))
      .map(k => ({key: k, label: sourceCohort.variables[k]?.var_label}));
  }, [sourceCohort, sourceVars]);

  const unmappedTargetVars = useMemo(() => {
    if (!targetCohort) return [];
    return Object.keys(targetCohort.variables || {})
      .filter(k => !targetVarSet.has(k))
      .map(k => ({key: k, label: targetCohort.variables[k]?.var_label}));
  }, [targetCohort, targetVarSet]);

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

          return <MappingRow key={varKey} varKey={varKey} mappings={mappings} status={status} />;
        })}
      </div>
    </div>
  );
}

function ComparisonView({file1, file2, pair, cohortsData}: {file1: ParsedFile; file2: ParsedFile; pair: string; cohortsData: Record<string, any>}) {
  const [showSame, setShowSame] = useState(false);

  const [fromStudy, targetStudy] = pair.split(' → ');
  const left = useMemo(() => getSourceVarsForPair(file1, pair), [file1, pair]);
  const right = useMemo(() => getSourceVarsForPair(file2, pair), [file2, pair]);

  const allSourceVars = useMemo(() => {
    const s = new Set([...left.keys(), ...right.keys()]);
    return Array.from(s).sort();
  }, [left, right]);

  const onlyLeft = allSourceVars.filter(v => left.has(v) && !right.has(v));
  const onlyRight = allSourceVars.filter(v => !left.has(v) && right.has(v));
  const inBoth = allSourceVars.filter(v => left.has(v) && right.has(v));

  const sameCount = inBoth.filter(v => {
    const lSigs = new Set((left.get(v) || []).map(mappingSignature));
    const rSigs = new Set((right.get(v) || []).map(mappingSignature));
    return JSON.stringify([...lSigs].sort()) === JSON.stringify([...rSigs].sort());
  }).length;
  const diffCount = inBoth.length - sameCount;

  return (
    <div className="mt-4">
      <div className="flex items-center gap-3 flex-wrap mb-4">
        <div className="flex gap-2 flex-wrap text-sm">
          <span className="badge badge-info">{inBoth.length} in both</span>
          <span className="badge badge-outline border-blue-400 text-blue-600">{onlyLeft.length} only in F1</span>
          <span className="badge badge-outline border-purple-400 text-purple-600">{onlyRight.length} only in F2</span>
          {diffCount > 0 && <span className="badge badge-warning">{diffCount} with differences</span>}
          {sameCount > 0 && <span className="badge badge-ghost">{sameCount} identical</span>}
        </div>
        <button
          className={`btn btn-xs gap-1 ml-auto ${showSame ? 'btn-neutral' : 'btn-outline'}`}
          onClick={() => setShowSame(v => !v)}
          title={showSame ? 'Hide identical mappings' : 'Show identical mappings'}
        >
          {showSame ? <EyeOff size={12} /> : <Eye size={12} />}
          {showSame ? 'Hide identical' : 'Show identical'}
        </button>
      </div>

      <div className="text-xs opacity-50 mb-3 flex flex-wrap gap-3">
        <span><span className="inline-block w-3 h-3 rounded bg-blue-200 dark:bg-blue-800 mr-1" />only in F1</span>
        <span><span className="inline-block w-3 h-3 rounded bg-purple-200 dark:bg-purple-800 mr-1" />only in F2</span>
        <span><span className="inline-block w-3 h-3 rounded bg-yellow-200 dark:bg-yellow-800 mr-1" />different mappings</span>
        {showSame && <span><span className="inline-block w-3 h-3 rounded bg-base-200 mr-1" />identical</span>}
      </div>

      <div className="flex gap-4">
        <FullSidePanel
          fileLabel={file1.name}
          fileTag="F1"
          sourceVars={left}
          allSourceVars={allSourceVars}
          side="left"
          otherVars={right}
          showSame={showSame}
          fromStudy={fromStudy}
          targetStudy={targetStudy}
          cohortsData={cohortsData}
        />
        <div className="w-px bg-base-300 shrink-0" />
        <FullSidePanel
          fileLabel={file2.name}
          fileTag="F2"
          sourceVars={right}
          allSourceVars={allSourceVars}
          side="right"
          otherVars={left}
          showSame={showSame}
          fromStudy={fromStudy}
          targetStudy={targetStudy}
          cohortsData={cohortsData}
        />
      </div>
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
              <div className="text-xs opacity-60 mb-1.5 uppercase tracking-wide">
                Common pairs ({commonPairs.length}) — click to compare
              </div>
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
                      <span key={p} className="badge badge-sm border-blue-300 text-blue-600 bg-blue-50 dark:bg-blue-900/20">{p}</span>
                    ))}
                  </div>
                </div>
              )}
              {onlyInFile2.length > 0 && (
                <div>
                  <div className="text-xs opacity-50 mb-1 uppercase tracking-wide">Only in F2 ({onlyInFile2.length})</div>
                  <div className="flex flex-wrap gap-1.5">
                    {onlyInFile2.map(p => (
                      <span key={p} className="badge badge-sm border-purple-300 text-purple-600 bg-purple-50 dark:bg-purple-900/20">{p}</span>
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
