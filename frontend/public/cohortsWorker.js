self.onmessage = async e => {
  const {apiUrl, requestId} = e.data;
  try {
    const response = await fetch(`${apiUrl}/cohorts-metadata`, {
      credentials: 'include'
    });
    const data = await response.json();
    self.postMessage({requestId, payload: data});
  } catch (error) {
    self.postMessage({requestId, error: error.message});
  }
};
