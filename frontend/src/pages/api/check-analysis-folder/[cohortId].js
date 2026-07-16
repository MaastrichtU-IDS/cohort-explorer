import fs from 'fs';
import {resolveCohortOutputDirectory} from '@/utils/safeDataPath';

export default function handler(req, res) {
  const { cohortId } = req.query;

  let folderPath;
  try {
    folderPath = resolveCohortOutputDirectory(cohortId);
  } catch {
    return res.status(400).json({exists: false, error: 'Invalid cohortId'});
  }
  
  try {
    const exists = fs.existsSync(folderPath);
    
    return res.status(200).json({ 
      exists,
      cohortId
    });
  } catch (error) {
    console.error('Error checking analysis folder:', error);
    return res.status(500).json({ 
      exists: false,
      cohortId,
      error: error.message 
    });
  }
}
