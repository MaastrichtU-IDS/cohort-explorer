export default async function handler(req, res) {
  const { sourceCohort, sourceVar, targetCohort, targetVar } = req.query;
  
  console.log('Compare EDA API handler called with:', { sourceCohort, sourceVar, targetCohort, targetVar });
  
  try {
    // Get the API URL from environment or use default
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const backendUrl = `${apiUrl}/api/compare-eda/${sourceCohort}/${sourceVar}/${targetCohort}/${targetVar}`;
    
    console.log('Fetching from backend:', backendUrl);
    
    // Fetch the merged image from the FastAPI backend
    const response = await fetch(backendUrl);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Backend error:', response.status, errorText);
      return res.status(response.status).json({ 
        error: 'Failed to fetch merged EDA image',
        details: errorText,
        status: response.status
      });
    }
    
    // Get the image buffer
    const imageBuffer = await response.arrayBuffer();
    
    // Set proper headers and send the image
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Cache-Control', 'public, max-age=3600');
    return res.send(Buffer.from(imageBuffer));
    
  } catch (error) {
    console.error('Error in compare-eda API route:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
}
