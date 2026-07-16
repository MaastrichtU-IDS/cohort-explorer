import fs from 'fs';
import {resolveVariableGraphPath} from '@/utils/safeDataPath';

export default function handler(req, res) {
  const { cohortId, variableName } = req.query;

  let imagePath;
  try {
    imagePath = resolveVariableGraphPath(cohortId, variableName);
  } catch {
    return res.status(400).json({error: 'Invalid cohortId or variableName'});
  }
  
  try {
    if (fs.existsSync(imagePath)) {
      const imageBuffer = fs.readFileSync(imagePath);
      res.setHeader('Content-Type', 'image/png');
      return res.send(imageBuffer);
    } else {
      return res.status(404).json({
        error: 'Image not found'
      });
    }
  } catch (error) {
    console.error('Error handling request:', error);
    return res.status(500).json({ error: error.message });
  }
}
