import React, { useMemo, useState } from 'react';
import { EdaVariable, completenessScore } from '@/utils/edaParsing';
import DomainFilterBar from './DomainFilterBar';

interface Props {
  variables: EdaVariable[];
  onVariableClick: (v: EdaVariable) => void;
}

type SortKey = 'name' | 'type' | 'observations' | 'completeness' | 'mean' | 'median' | 'stdDev' | 'skewness' | 'kurtosis' | 'uniqueValues' | 'normality';

const EdaStatisticsTable: React.FC<Props> = ({ variables, onVariableClick }) => {
  const [sortKey, setSortKey] = useState<SortKey>('name');
  const [sortAsc, setSortAsc] = useState(true);
  const [filterType, setFilterType] = useState<'all' | 'numeric' | 'categorical'>('all');
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 50;

  const handleSort = (key: SortKey) => {
    if (key === sortKey) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(true); }
  };

  const filtered = useMemo(() => {
    let list = [...variables];
    if (filterType !== 'all') list = list.filter(v => v.type === filterType);
    if (selectedDomain) list = list.filter(v => v.domain === selectedDomain);
    if (searchFilter) {
      const q = searchFilter.toLowerCase();
      list = list.filter(v => v.name.toLowerCase().includes(q) || v.label.toLowerCase().includes(q) || (v.conceptName && v.conceptName.toLowerCase().includes(q)));
    }

    const dir = sortAsc ? 1 : -1;
    list.sort((a, b) => {
      const getVal = (v: EdaVariable): number | string => {
        switch (sortKey) {
          case 'name': return v.name.toLowerCase();
          case 'type': return v.type;
          case 'observations': return v.totalObservations;
          case 'completeness': return completenessScore(v);
          case 'mean': return v.mean ?? -Infinity;
          case 'median': return v.median ?? -Infinity;
          case 'stdDev': return v.stdDev ?? -Infinity;
          case 'skewness': return v.skewness ?? -Infinity;
          case 'kurtosis': return v.kurtosis ?? -Infinity;
          case 'uniqueValues': return v.uniqueValues;
          case 'normality': return v.isNormal ? 1 : 0;
          default: return 0;
        }
      };
      const va = getVal(a);
      const vb = getVal(b);
      if (typeof va === 'string' && typeof vb === 'string') return va.localeCompare(vb) * dir;
      return ((va as number) - (vb as number)) * dir;
    });
    return list;
  }, [variables, filterType, searchFilter, sortKey, sortAsc, selectedDomain]);

  const pageCount = Math.ceil(filtered.length / PAGE_SIZE);
  const pageVars = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const SortHeader = ({ label, field }: { label: string; field: SortKey }) => (
    <th
      className="cursor-pointer select-none hover:bg-base-200 whitespace-nowrap"
      onClick={() => handleSort(field)}
    >
      {label} {sortKey === field ? (sortAsc ? '▲' : '▼') : ''}
    </th>
  );

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Filter variables..."
          className="input input-sm input-bordered w-60"
          value={searchFilter}
          onChange={e => { setSearchFilter(e.target.value); setPage(0); }}
        />
        <DomainFilterBar variables={variables} selectedDomain={selectedDomain} onChange={d => { setSelectedDomain(d); setPage(0); }} />
        <div className="join">
          {(['all', 'numeric', 'categorical'] as const).map(t => (
            <button
              key={t}
              className={`btn btn-sm join-item ${filterType === t ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => { setFilterType(t); setPage(0); }}
            >
              {t === 'all' ? 'All' : t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
        <span className="text-sm text-gray-500">{filtered.length} variables</span>
      </div>

      {/* Table */}
      <div className="card bg-base-100 shadow-md">
        <div className="overflow-x-auto">
          <table className="table table-sm table-zebra table-pin-rows">
            <thead>
              <tr>
                <SortHeader label="Variable" field="name" />
                <th>Label</th>
                <SortHeader label="Type" field="type" />
                <SortHeader label="Obs" field="observations" />
                <SortHeader label="Complete %" field="completeness" />
                <SortHeader label="Unique" field="uniqueValues" />
                <SortHeader label="Mean" field="mean" />
                <SortHeader label="Median" field="median" />
                <SortHeader label="Std Dev" field="stdDev" />
                <SortHeader label="Skewness" field="skewness" />
                <SortHeader label="Kurtosis" field="kurtosis" />
                <SortHeader label="Normal" field="normality" />
                <th>Units</th>
                <th>Domain</th>
              </tr>
            </thead>
            <tbody>
              {pageVars.map(v => {
                const score = completenessScore(v);
                return (
                  <tr
                    key={v.name}
                    className="cursor-pointer hover:bg-primary/5"
                    onClick={() => onVariableClick(v)}
                  >
                    <td className="font-medium text-xs max-w-[150px] truncate" title={v.name}>{v.name}</td>
                    <td className="text-xs max-w-[200px] truncate" title={v.label}>{v.label}</td>
                    <td>
                      <span className={`badge badge-xs ${v.type === 'numeric' ? 'badge-primary' : v.type === 'categorical' ? 'badge-success' : 'badge-warning'}`}>
                        {v.type}
                      </span>
                    </td>
                    <td>{v.totalObservations}</td>
                    <td>
                      <div className="flex items-center gap-1">
                        <div className="w-12 bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${score >= 80 ? 'bg-green-500' : score >= 50 ? 'bg-amber-500' : 'bg-red-500'}`}
                            style={{ width: `${score}%` }}
                          />
                        </div>
                        <span className="text-xs">{score.toFixed(0)}%</span>
                      </div>
                    </td>
                    <td>{v.uniqueValues}</td>
                    <td className="text-xs">{v.mean !== undefined ? v.mean.toFixed(2) : '—'}</td>
                    <td className="text-xs">{v.median !== undefined ? v.median : '—'}</td>
                    <td className="text-xs">{v.stdDev !== undefined ? v.stdDev.toFixed(2) : '—'}</td>
                    <td className="text-xs">{v.skewness !== undefined ? v.skewness.toFixed(2) : '—'}</td>
                    <td className="text-xs">{v.kurtosis !== undefined ? v.kurtosis.toFixed(2) : '—'}</td>
                    <td>
                      {v.type === 'numeric' ? (
                        <span className={`badge badge-xs ${v.isNormal ? 'badge-success' : 'badge-warning'}`}>
                          {v.isNormal ? 'Yes' : 'No'}
                        </span>
                      ) : '—'}
                    </td>
                    <td className="text-xs">{v.units || '—'}</td>
                    <td className="text-xs">{v.domain || '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {pageCount > 1 && (
        <div className="flex justify-center gap-2">
          <button className="btn btn-sm" disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Prev</button>
          <span className="flex items-center text-sm">Page {page + 1} of {pageCount}</span>
          <button className="btn btn-sm" disabled={page >= pageCount - 1} onClick={() => setPage(p => p + 1)}>Next →</button>
        </div>
      )}
    </div>
  );
};

export default EdaStatisticsTable;
