import { Variable } from '@/types';
import React, {useState, useEffect} from 'react';

const FilterByMetadata = ({ label, options, onFiltersChange, metadata_id, searchResults = [] }: any) => {

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
      if (filterType === 'categorical') {
        return variable.categories.length > 0;
      } else if (filterType === 'non_categorical') {
        return variable.categories.length === 0;
      } else {
        return variable[filterType] === item;
      }
    }).length;
  };

    // countMatches('OMOP', domain)

    return (
        <div className='mb-3 space-y-1'>
            <h3 className="font-bold">{label}</h3>
            {options.map((option: any, index: number) => (
                <div key={index} className="form-control">
                    <label className="label cursor-pointer p-0">
                        {/* <span className="label-text text-xs">{option} ({countMatches(metadata_id, null)})</span> */}
                        <span className="label-text text-xs">{option} ({countMatches(metadata_id, option)})</span>
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
