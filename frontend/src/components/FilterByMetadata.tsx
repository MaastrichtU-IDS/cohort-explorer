import React, {useState} from 'react';

const FilterByMetadata = ({label, options, onFiltersChange, metadata_id, searchResults = []}: any) => {
  const [selectedOptions, setSelectedOptions] = useState(new Set());

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
      if (filterType === 'categorical' && item === 'Categorical') {
        return variable.categories.length > 0;
      } else if (filterType === 'categorical' && item === 'Non-categorical') {
        return variable.categories.length === 0;
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
