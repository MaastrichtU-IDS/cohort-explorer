import React, {useState, useEffect} from 'react';
import {AutocompleteConceptProps, Concept} from '@/types';
import {apiUrl} from '@/utils';
import {useCohorts} from '@/components/CohortsContext';

const acceptedDomains = [
  'Condition',
  'Device',
  'Drug',
  'Geography',
  'Meas Value',
  'Measurement',
  'Metadata',
  'Observation',
  'Procedure',
  'Spec Anatomic Site',
  'Specimen',
  'Condition Status',
  'Condition/Device',
  'Condition/Meas',
  'Condition/Obs',
  'Condition/Procedure',
  'Cost',
  'Currency',
  'Device/Drug',
  'Device/Procedure',
  'Drug/Procedure',
  'Episode',
  'Ethnicity',
  'Gender',
  'Language',
  'Meas Value Operator',
  'Meas/Procedure',
  'Note',
  'Obs/Procedure',
  'Payer',
  'Place of Service',
  'Plan',
  'Plan Stop Reason',
  'Provider',
  'Race',
  'Regimen',
  'Relationship',
  'Revenue Code',
  'Route',
  'Spec Disease Status',
  'Sponsor',
  'Type Concept',
  'Unit',
  'Visit'
];

const AutocompleteConcept: React.FC<AutocompleteConceptProps> = ({
  onSelect,
  query = '',
  value = '',
  domain = '',
  index = '',
  tooltip = '',
  canEdit = false
}: any) => {
  // const {cohortsData, fetchCohortsData} = useCohorts();
  const [filteredSuggestions, setFilteredSuggestions] = useState<Concept[]>([]);
  const [inputValue, setInputValue] = useState(query);
  const [debouncedInput, setDebouncedInput] = useState('');
  const [isUserInteracted, setIsUserInteracted] = useState(false);
  const [selectedDomains, setSelectedDomains] = useState<string[]>((domain && acceptedDomains.includes(domain.trim())) ? [domain.trim()] : acceptedDomains );
  const [errorMsg, setErrorMsg] = useState('');
  // const [selectedConcept, setSelectedConcept] = useState<Concept | null>(null);

  if (!tooltip) {
    tooltip = 'Map this variable to a standard concept';
  }

  // Debounce input value
  useEffect(() => {
    if (!isUserInteracted) return;
    const handler = setTimeout(() => {
      // console.log('Debounced input!!', inputValue);
      setDebouncedInput(inputValue);
    }, 300);

    return () => {
      clearTimeout(handler);
    };
  }, [inputValue, isUserInteracted]);

  // Fetch suggestions from the API
  useEffect(() => {
    if (!isUserInteracted) return;
    const domainBit = selectedDomains.map((domain) => `&domain=${domain}`).join("")
    if (debouncedInput.length > 0 && isUserInteracted) {
      fetch(`${apiUrl}/search-concepts?query=${debouncedInput}${domainBit}`, {
        credentials: 'include'
      })
        .then(async response => {
          if (!response.ok) {
            const res = await response.json()
            if (res['detail']) {
              throw new Error(`${res['detail']} (status ${response.status})`);
            }
            throw new Error(`Error getting suggestions (status ${response.status})`);
          }
          return response.json();
        })
        .then(data => {
          // console.log('DEBUG: Autocomplete response', data);
          setFilteredSuggestions(data);
        })
        .catch(error => setErrorMsg(error.message));
    } else {
      setFilteredSuggestions([]);
    }
  }, [debouncedInput, selectedDomains, isUserInteracted]);

  const handleInputChange = (event: any) => {
    setInputValue(event.target.value);
  };

  const handleDomainChange = (domain: string) => {
    setSelectedDomains(prev =>
      prev.includes(domain)
        ? prev.filter(d => d !== domain)
        : [...prev, domain]
    );
  };

  const handleSuggestionClick = (suggestion: Concept) => {
    // console.log('Selected suggestion', suggestion);
    onSelect(suggestion);
    // Close the modal after selecting a suggestion
    const modal = document.getElementById(autocompleteModalId);
    if (modal && modal.tagName === 'DIALOG') {
      (modal as HTMLDialogElement).close();
    }
    setIsUserInteracted(false);
  };

  const autocompleteModalId = `autocomplete_concept_modal_${index}`;

  return (
    <div>
      {canEdit &&
        <button
          className={`badge badge-outline tooltip tooltip-bottom hover:bg-base-300 before:max-w-[10rem] before:content-[attr(data-tip)] before:whitespace-pre-wrap`}
          data-tip={tooltip}
          onClick={() => {
            if (query && !inputValue) setInputValue(query);
            setIsUserInteracted(true);
            setTimeout(() => {
              // @ts-ignore
              document.getElementById(autocompleteModalId)?.showModal();
            }, 0)
          }}
        >
          {value ? `🪪 ${value}` : 'Map to concept'}
        </button>
      }
      {(!canEdit && value) &&
        <span className="badge badge-outline">{`🪪 ${value}`}</span>
      }

      {isUserInteracted &&
        <dialog id={autocompleteModalId} className="modal">
          <div className="modal-box space-y-2 max-w-none w-fit">
            <div className="justify-between items-start">
            <div className="flex">
              <input
                type="text"
                className="input input-bordered w-full mb-4"
                value={inputValue}
                onChange={handleInputChange}
                placeholder="Search..."
              />
              {/* Domain filter dropdown */}
              <div className="dropdown dropdown-end ml-2">
                <label tabIndex={0} className="btn btn-md">Filter by domains</label>
                <ul tabIndex={0} className="dropdown-content menu menu-horizontal shadow bg-base-100 rounded-box w-52 z-50">
                  {acceptedDomains.map(domain => (
                    <li key={domain} className='opacity-100'>
                      <label>
                        <input
                          type="checkbox"
                          checked={selectedDomains.includes(domain)}
                          className="checkbox"
                          onChange={() => handleDomainChange(domain)}
                        /> {domain}
                      </label>
                    </li>
                  ))}
                </ul>
              </div>
              </div>
              {filteredSuggestions.length > 0 ? (
                <table className="table-auto w-full">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Domain</th>
                      <th>ID</th>
                      <th>Used by</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredSuggestions.map((suggestion: any, i: number) => (
                      <tr
                        key={i}
                        className="hover:bg-base-200 cursor-pointer"
                        onClick={() => handleSuggestionClick(suggestion)}
                      >
                        <td>{suggestion.label}</td>
                        <td className="px-2">{suggestion.domain}</td>
                        <td>
                          {suggestion.id}
                        </td>
                        <td className={`tooltip tooltip-left before:max-w-[30rem] before:whitespace-pre-wrap text-center w-full`}
                          data-tip={suggestion.used_by.map((variab: any) => `${variab.cohort_id} - ${variab.var_name} (${variab.var_label})`).join('\n')}
                        >
                          {suggestion.used_by.length}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <>
                  {errorMsg ? (
                    <div className="text-red-500 text-center">{errorMsg}</div>
                  ) : (
                    <div className='flex flex-col items-center opacity-70 text-slate-500 mt-5 mb-5'>
                      <span className="loading loading-spinner loading-lg mb-4"></span>
                      <p>Getting concepts suggestions...</p>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          <form method="dialog" className="modal-backdrop">
            <button>close</button>
          </form>
        </dialog>
      }
    </div>
  );
};

export default AutocompleteConcept;
