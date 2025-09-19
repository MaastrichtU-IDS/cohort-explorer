import fs from 'fs';
import path from 'path';

export default function handler(req, res) {
  const { cohortId } = req.query;
  
  try {
    const folderPath = path.join('/data', `dcr_output_${cohortId}`);
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
