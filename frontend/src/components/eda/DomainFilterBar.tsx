import React, { useMemo } from 'react';
import { EdaVariable } from '@/utils/edaParsing';

interface Props {
  variables: EdaVariable[];
  selectedDomain: string | null;
  onChange: (domain: string | null) => void;
}

/**
 * A cluster of badge-style domain filter buttons.
 * Extracts unique OMOP domains from variables.
 * Clicking an active domain unselects it (shows all).
 */
const DomainFilterBar: React.FC<Props> = ({ variables, selectedDomain, onChange }) => {
  const domains = useMemo(() => {
    const set = new Set<string>();
    for (const v of variables) {
      if (v.domain) set.add(v.domain);
    }
    return Array.from(set).sort();
  }, [variables]);

  if (domains.length === 0) return null;

  return (
    <div className="ml-auto max-w-[33%]">
      <span className="text-sm font-semibold text-gray-600 mb-1 block">Domain:</span>
      <div className="grid grid-cols-2 gap-1">
        <button
          className={`badge badge-lg cursor-pointer px-4 py-2 ${selectedDomain === null ? 'badge-primary' : 'badge-neutral'}`}
          onClick={() => onChange(null)}
        >
          All
        </button>
        {domains.map(d => (
          <button
            key={d}
            className={`badge badge-lg cursor-pointer px-4 py-2 ${selectedDomain === d ? 'badge-primary' : 'badge-neutral'}`}
            onClick={() => onChange(selectedDomain === d ? null : d)}
          >
            {d}
          </button>
        ))}
      </div>
    </div>
  );
};

export default DomainFilterBar;
