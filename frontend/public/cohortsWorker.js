self.onmessage = async e => {
  const apiUrl = e.data.apiUrl;
  try {
    const response = await fetch(`${apiUrl}/cohorts-metadata`, {
      credentials: 'include'
    });
    const data = await response.json();
    self.postMessage(data);
  } catch (error) {
    self.postMessage({error: error.message});
  }
};
