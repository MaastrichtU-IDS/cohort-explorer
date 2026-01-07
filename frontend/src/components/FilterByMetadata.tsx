import React, {useState, useEffect} from 'react';

const FilterByMetadata = ({label, options, onFiltersChange, metadata_id, searchResults = [], selectedValues}: any) => {
  const [selectedOptions, setSelectedOptions] = useState(new Set());

  // Sync internal state with parent's selected values
  useEffect(() => {
    if (selectedValues) {
      setSelectedOptions(selectedValues);
    }
  }, [selectedValues]);

  const handleOptionChange = (option: string) => {
    const newSet = new Set(selectedOptions);
    if (newSet.has(option)) {
      newSet.delete(option);
    } else {
      newSet.add(option);
    }
    setSelectedOptions(newSet);
    onFiltersChange(newSet); // Callback to inform parent component about the change
  };

  // Function to count filtered vars based on filter type
  const countMatches = (filterType: string, item: string | null) => {
    return searchResults.filter((variable: any) => {
      if (filterType === 'categorical') {
        const catCount = variable.categories.length;
        if (item === 'Non-categorical') return catCount === 0;
        if (item === 'All categorical') return catCount > 0;
        if (item === '2 categories') return catCount === 2;
        if (item === '3 categories') return catCount === 3;
        if (item === '4+ categories') return catCount >= 4;
        return false;
      } else {
        return variable[filterType] === item;
      }
    }).length;
  };

  return (
    <div className="mb-3 w-full space-y-1">
      <h3 className="text-sm font-bold">{label}</h3>
      {options.map((option: any, index: number) => (
        <div key={index} className="form-control text-left">
          <label className="label cursor-pointer p-0">
            <span className="label-text text-xs">
              {option} <span className="opacity-50">({countMatches(metadata_id, option)})</span>
            </span>
            <input
              type="checkbox"
              checked={selectedOptions.has(option)}
              onChange={() => handleOptionChange(option)}
              className="checkbox checkbox-xs"
            />
          </label>
        </div>
      ))}
    </div>
  );
};

export default FilterByMetadata;
