import fs from 'fs';
import path from 'path';

export default function handler(req, res) {
  const { cohortId, variableName } = req.query;
  
  console.log('API handler called with:', { cohortId, variableName });

  const cwd = process.cwd();
  const imagePath = path.join('/data', `dcr_output_${cohortId}`, `${variableName.toLowerCase()}.png`);
  
  console.log('Looking for image at:', imagePath);
  console.log('File exists?', fs.existsSync(imagePath));
  
  try {
    if (fs.existsSync(imagePath)) {
      const imageBuffer = fs.readFileSync(imagePath);
      res.setHeader('Content-Type', 'image/png');
      return res.send(imageBuffer);
    } else {
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