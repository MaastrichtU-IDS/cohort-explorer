// Backend integration - FIXED VERSION
const handleMapConcepts = async () => {
  if (!sourceCohort || selectedTargets.length === 0) {
    alert('Please select a source cohort and at least one target cohort');
    return;
  }
  setLoading(true);
  setError(null);
  setMappingOutput(null);
  
  try {
    const target_studies = selectedTargets.map((cohortId: string) => [cohortId, false]);
    const response = await fetch(`${apiUrl}/api/generate-mapping`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        source_study: sourceCohort,
        target_studies,
      }),
    });
    
    if (!response.ok) {
      const result = await response.json();
      let errorMsg = result.detail || result.error || 'Failed to generate mapping';
      if (
        response.status === 404 &&
        typeof errorMsg === 'string' &&
        errorMsg.endsWith("metadata has not been added yet!")
      ) {
        setError(`The metadata of ${sourceCohort} has not been added yet!`);
        return;
      }
      throw new Error(errorMsg);
    }

    // Get response as text first, then work with it
    const responseText = await response.text();
    
    // Handle download
    const blob = new Blob([responseText], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mapping_${sourceCohort}_to_${selectedTargets.join('_')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    // Handle preview by parsing the text as JSON
    try {
      const jsonData = JSON.parse(responseText);
      console.log('Parsed JSON data:', jsonData); // Debug log
      
      // Simplified type validation - just check if it's an array
      if (Array.isArray(jsonData)) {
        // Flatten any nested data for display
        const flattenedData = jsonData.map(item => {
          if (typeof item !== 'object' || item === null) {
            return { value: String(item) };
          }
          
          const flatItem: any = {};
          Object.entries(item).forEach(([key, value]) => {
            if (Array.isArray(value)) {
              flatItem[key] = value.join(', ');
            } else if (value !== null && typeof value === 'object') {
              flatItem[key] = JSON.stringify(value);
            } else {
              flatItem[key] = value;
            }
          });
          return flatItem;
        });
        
        console.log('Flattened data for preview:', flattenedData); // Debug log
        setMappingOutput(flattenedData);
      } else if (typeof jsonData === 'object' && jsonData !== null) {
        // Handle case where response is a single object instead of array
        const flatItem: any = {};
        Object.entries(jsonData).forEach(([key, value]) => {
          if (Array.isArray(value)) {
            flatItem[key] = value.join(', ');
          } else if (value !== null && typeof value === 'object') {
            flatItem[key] = JSON.stringify(value);
          } else {
            flatItem[key] = value;
          }
        });
        setMappingOutput([flatItem]);
      } else {
        console.warn('JSON data is not an array or object:', typeof jsonData, jsonData);
        setMappingOutput([{ data: JSON.stringify(jsonData) }]);
      }
    } catch (parseError) {
      console.error('Error parsing JSON response:', parseError);
      console.log('Raw response text:', responseText.substring(0, 500)); // Debug log
      // Still show something in preview - the raw text
      setMappingOutput([{ 
        error: 'Failed to parse JSON response', 
        raw_data: responseText.substring(0, 200) + (responseText.length > 200 ? '...' : '')
      }]);
    }
    
  } catch (err: any) {
    console.error('Request error:', err); // Debug log
    setError(
      typeof err.message === 'string' && err.message.endsWith("metadata has not been added yet!")
        ? `The metadata of ${sourceCohort} has not been added yet!`
        : (err.message || 'Unknown error')
    );
  } finally {
    setLoading(false);
  }
};