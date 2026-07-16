self.onmessage = async e => {
  const {apiUrl, requestId} = e.data;
  try {
    const response = await fetch(`${apiUrl}/cohorts-metadata-sparql`, {
      credentials: 'include'
    });
    const data = await response.json();
    
    // Handle the new response format with metadata
    if (data.cohorts && data.sparql_metadata) {
      // Extract cohorts and send with metadata as separate properties
      const response = {
        ...data.cohorts,
        sparqlRows: data.sparql_metadata.row_count,
        sparqlMetadata: data.sparql_metadata
      };
      self.postMessage({requestId, payload: response});
    } else {
      // Fallback for old format
      self.postMessage({requestId, payload: data});
    }
  } catch (error) {
    self.postMessage({requestId, error: error.message});
  }
};
