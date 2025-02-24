import fs from 'fs';
import path from 'path';

export default function handler(req, res) {
  // Get parameters from the dynamic route
  const { cohortId, variableName } = req.query;
  
  console.log('API handler called with:', { cohortId, variableName });
  
  // Construct the path to the image file
  const cwd = process.cwd();
  const imagePath = path.join(cwd, 'public', 'graphs', cohortId, `${variableName}.png`);
  
  console.log('Looking for image at:', imagePath);
  console.log('File exists?', fs.existsSync(imagePath));
  
  try {
    if (fs.existsSync(imagePath)) {
      // If the file exists, serve it
      const imageBuffer = fs.readFileSync(imagePath);
      res.setHeader('Content-Type', 'image/png');
      return res.send(imageBuffer);
    } else {
      // If the file doesn't exist, check the directory contents to help debugging
      const graphsDir = path.join(cwd, 'public', 'graphs');
      const cohortDir = path.join(graphsDir, cohortId);
      
      console.log('Current working directory:', cwd);
      
      if (fs.existsSync(graphsDir)) {
        console.log('Graphs directory exists, contents:', fs.readdirSync(graphsDir));
      } else {
        console.log('Graphs directory does not exist at:', graphsDir);
      }
      
      if (fs.existsSync(cohortDir)) {
        console.log('Cohort directory exists, contents:', fs.readdirSync(cohortDir));
      } else {
        console.log('Cohort directory does not exist at:', cohortDir);
      }
      
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