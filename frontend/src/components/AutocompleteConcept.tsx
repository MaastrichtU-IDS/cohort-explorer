import React, {useState, useEffect} from 'react';

const AutocompleteConcept = ({
  onSelect,
  query = '',
  value = '',
  domain = '',
  index = '',
  cohortId = '',
  tooltip = ''
}: any) => {
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);
  const [inputValue, setInputValue] = useState(query);
  const [debouncedInput, setDebouncedInput] = useState('');
  const [selectedConcept, setSelectedConcept]: any = useState(null);
  const [isUserInteracted, setIsUserInteracted] = useState(false);

  if (!tooltip) {
    tooltip = 'Map this variable to a standard concept';
  }

  // Debounce input value
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedInput(inputValue);
    }, 300);

    return () => {
      clearTimeout(handler);
    };
  }, [inputValue]);

  // Fetch suggestions from the API
  useEffect(() => {
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
    const domainBit = domain && acceptedDomains.includes(domain) ? `&domain=${domain}` : '';
    const vocabs = '&vocabulary=LOINC&vocabulary=SNOMED&vocabulary=RxNorm';
    // &conceptClass=Clinical Finding
    if (debouncedInput.length > 0 && isUserInteracted) {
      fetch(
        `https://athena.ohdsi.org/api/v1/concepts?pageSize=15${domainBit}${vocabs}&standardConcept=Standard&page=1&query=${debouncedInput}`
      )
        .then(response => response.json())
        .then(data => {
          // console.log('Autocomplete response', data);
          setFilteredSuggestions(data.content);
        })
        .catch(error => console.error('Error fetching data: ', error));
    } else {
      setFilteredSuggestions([]);
    }
  }, [debouncedInput, domain, isUserInteracted]);

  const handleInputChange = (event: any) => {
    setInputValue(event.target.value);
  };

  const handleSuggestionClick = (suggestion: any) => {
    setSelectedConcept(suggestion);
    onSelect(suggestion);
    // Close the modal after selecting a suggestion
    const modal = document.getElementById(autocompleteModalId);
    if (modal && modal.tagName === 'DIALOG') {
      (modal as HTMLDialogElement).close();
    }
  };

  const autocompleteModalId = `autocomplete_modal_${cohortId}_${index}`;

  return (
    <div>
      <button
        className="badge badge-outline tooltip tooltip-bottom hover:bg-base-300 before:max-w-[10rem] before:content-[attr(data-tip)]"
        data-tip={tooltip}
        onClick={() => {
          // @ts-ignore
          document.getElementById(autocompleteModalId)?.showModal();
          setIsUserInteracted(true);
          if (query && !inputValue) setInputValue(query);
        }}
      >
        {value ? `🪪 ${value}` : 'Map to concept'}
      </button>
      <dialog id={autocompleteModalId} className="modal">
        <div className="modal-box space-y-2 max-w-none w-fit">
          <div className="justify-between items-start">
            <input
              type="text"
              className="input input-bordered w-full mb-4"
              value={inputValue}
              onChange={handleInputChange}
              placeholder="Search..."
            />
            {filteredSuggestions.length > 0 && (
              <table className="table-auto w-full">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Domain</th>
                    <th>ID</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredSuggestions.map((suggestion: any, i: number) => (
                    <tr
                      key={i}
                      className="hover:bg-base-200 cursor-pointer"
                      onClick={() => handleSuggestionClick(suggestion)}
                    >
                      <td>{suggestion.name}</td>
                      <td className="px-2">{suggestion.domain}</td>
                      <td>
                        {suggestion.vocabulary}:{suggestion.id}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        <form method="dialog" className="modal-backdrop">
          <button>close</button>
        </form>
      </dialog>

      {/* {filteredSuggestions.length > 0 && (
        <div className="absolute z-10 w-full mt-1 rounded-lg shadow-lg bg-base-100">
          <table className="table-auto w-full">
            <thead>
              <tr>
                <th>Name</th>
                <th>Domain</th>
                <th>ID</th>
              </tr>
            </thead>
            <tbody>
              {filteredSuggestions.map((suggestion: any, index: number) => (
                <tr
                  key={index}
                  className="hover:bg-base-200 cursor-pointer"
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  <td>{suggestion.name}</td>
                  <td>{suggestion.domain}</td>
                  <td>{suggestion.vocabulary}:{suggestion.id}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )} */}
    </div>
  );
};

export default AutocompleteConcept;
