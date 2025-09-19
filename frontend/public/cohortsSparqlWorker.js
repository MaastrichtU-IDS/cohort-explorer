self.onmessage = async e => {
  const apiUrl = e.data.apiUrl;
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
      self.postMessage(response);
    } else {
      // Fallback for old format
      self.postMessage(data);
    }
  } catch (error) {
    self.postMessage({error: error.message});
  }
};
