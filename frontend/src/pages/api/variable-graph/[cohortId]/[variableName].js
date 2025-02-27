import fs from 'fs';
import path from 'path';

export default function handler(req, res) {
  // Get parameters from the dynamic route
  const { cohortId, variableName } = req.query;
  
  console.log('API handler called with:', { cohortId, variableName });
  
  // Construct the path to the image file
  const cwd = process.cwd();
  const imagePath = path.join('/data', `dcr_output_${cohortId}`, `${variableName.toLowerCase()}.png`);
  
  console.log('Looking for image at:', imagePath);
  console.log('File exists?', fs.existsSync(imagePath));
  
  try {
    if (fs.existsSync(imagePath)) {
      // If the file exists, serve it
      const imageBuffer = fs.readFileSync(imagePath);
      res.setHeader('Content-Type', 'image/png');
      return res.send(imageBuffer);
    } else {
      // Return 404 with debugging information
      return res.status(404).json({
        error: 'Image not found',
        requested: {
          cohortId,
          variableName,
          fullPath: imagePath
        }
      });
    }
  } catch (error) {
    console.error('Error handling request:', error);
    return res.status(500).json({ error: error.message });
  }
}