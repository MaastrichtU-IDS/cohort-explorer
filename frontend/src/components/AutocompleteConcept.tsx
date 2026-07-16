import React, {useState, useEffect} from 'react';
import {AutocompleteConceptProps, Concept} from '@/types';
import {apiUrl} from '@/utils';
import {useCohorts} from '@/components/CohortsContext';
import {conceptMapElementId} from '@/utils/variableFiltering';
import {resolveMappingApiUrl} from '@/utils/apiUrls';
import {
  ACCEPTED_CONCEPT_DOMAINS,
  resolveConceptSuggestionState,
  resolveInitialConceptDomains
} from '@/utils/conceptSearch';

const AutocompleteConcept: React.FC<AutocompleteConceptProps> = ({
  onSelect,
  query = '',
  value = '',
  domain = '',
  index = '',
  cohortId = '',
  tooltip = '',
  canEdit = false
}: any) => {
  // const {cohortsData, fetchCohortsData} = useCohorts();
  const [filteredSuggestions, setFilteredSuggestions] = useState<Concept[]>([]);
  const [inputValue, setInputValue] = useState(query);
  const [debouncedInput, setDebouncedInput] = useState('');
  const [isUserInteracted, setIsUserInteracted] = useState(false);
  const [selectedDomains, setSelectedDomains] = useState<string[]>(() => resolveInitialConceptDomains(domain));
  const [errorMsg, setErrorMsg] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
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
    const domainBit = selectedDomains.map(domain => `&domain=${domain}`).join('');
    if (debouncedInput.length > 0 && isUserInteracted) {
      const controller = new AbortController();
      setIsLoading(true);
      setHasSearched(false);
      setErrorMsg('');
      setFilteredSuggestions([]);
      fetch(resolveMappingApiUrl(apiUrl, `search-concepts?query=${debouncedInput}${domainBit}`), {
        credentials: 'include',
        signal: controller.signal
      })
        .then(async response => {
          if (!response.ok) {
            const res = await response.json();
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
          setHasSearched(true);
        })
        .catch(error => {
          if (error.name !== 'AbortError') {
            setErrorMsg(error.message);
            setHasSearched(true);
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) setIsLoading(false);
        });
      return () => controller.abort();
    } else {
      setFilteredSuggestions([]);
      setIsLoading(false);
      setHasSearched(false);
      setErrorMsg('');
    }
  }, [debouncedInput, selectedDomains, isUserInteracted]);

  const handleInputChange = (event: any) => {
    setInputValue(event.target.value);
  };

  const handleDomainChange = (domain: string) => {
    setSelectedDomains(prev => (prev.includes(domain) ? prev.filter(d => d !== domain) : [...prev, domain]));
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

  const autocompleteModalId = `autocomplete_concept_modal_${cohortId ? `${cohortId}_` : ''}${index}`;
  const suggestionState = resolveConceptSuggestionState({
    isLoading,
    hasSearched,
    errorMsg,
    resultCount: filteredSuggestions.length
  });

  return (
    <div>
      {canEdit && (
        <button
          id={cohortId && index ? conceptMapElementId(cohortId, String(index)) : undefined}
          data-testid={cohortId && index ? conceptMapElementId(cohortId, String(index)) : undefined}
          className={`badge badge-outline tooltip tooltip-bottom hover:bg-base-300 before:max-w-[10rem] before:content-[attr(data-tip)] before:whitespace-pre-wrap`}
          data-tip={tooltip}
          onClick={() => {
            if (query && !inputValue) setInputValue(query);
            setIsUserInteracted(true);
            setTimeout(() => {
              // @ts-ignore
              document.getElementById(autocompleteModalId)?.showModal();
            }, 0);
          }}
        >
          {value ? `🪪 ${value}` : 'Map to concept'}
        </button>
      )}
      {!canEdit && value && <span className="badge badge-outline">{`🪪 ${value}`}</span>}

      {isUserInteracted && (
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
                  <label tabIndex={0} className="btn btn-md">
                    Filter by domains
                  </label>
                  <ul
                    tabIndex={0}
                    className="dropdown-content menu menu-horizontal shadow bg-base-100 rounded-box w-52 z-50"
                  >
                    {ACCEPTED_CONCEPT_DOMAINS.map(domain => (
                      <li key={domain} className="opacity-100">
                        <label>
                          <input
                            type="checkbox"
                            checked={selectedDomains.includes(domain)}
                            className="checkbox"
                            onChange={() => handleDomainChange(domain)}
                          />{' '}
                          {domain}
                        </label>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              {suggestionState === 'results' ? (
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
                        <td>{suggestion.id}</td>
                        <td
                          className={`tooltip tooltip-left before:max-w-[30rem] before:whitespace-pre-wrap text-center w-full`}
                          data-tip={suggestion.used_by
                            .map((variab: any) => `${variab.cohort_id} - ${variab.var_name} (${variab.var_label})`)
                            .join('\n')}
                        >
                          {suggestion.used_by.length}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <>
                  {suggestionState === 'error' ? (
                    <div className="text-red-500 text-center">{errorMsg}</div>
                  ) : suggestionState === 'loading' ? (
                    <div className="flex flex-col items-center opacity-70 text-slate-500 mt-5 mb-5">
                      <span className="loading loading-spinner loading-lg mb-4"></span>
                      <p>Getting concepts suggestions...</p>
                    </div>
                  ) : suggestionState === 'empty' ? (
                    <div className="text-center opacity-70 text-slate-500 mt-5 mb-5">No matching concepts found.</div>
                  ) : (
                    <div className="text-center opacity-70 text-slate-500 mt-5 mb-5">Enter a concept name to search.</div>
                  )}
                </>
              )}
            </div>
          </div>

          <form method="dialog" className="modal-backdrop">
            <button>close</button>
          </form>
        </dialog>
      )}
    </div>
  );
};

export default AutocompleteConcept;
