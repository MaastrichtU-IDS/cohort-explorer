import type { NextApiRequest, NextApiResponse } from 'next';
import fs from 'fs';
import path from 'path';

type CohortStatistics = {
  totalCohorts: number;
  cohortsWithMetadata: number;
  cohortsWithAggregateAnalysis: number;
  totalPatients: number;
  patientsInCohortsWithMetadata: number;
  totalVariables: number;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const statistics: CohortStatistics = req.body;
    
    // Validate the statistics object
    if (!statistics || typeof statistics !== 'object') {
      return res.status(400).json({ message: 'Invalid statistics data' });
    }
    
    // Create the public directory if it doesn't exist
    const publicDir = path.join(process.cwd(), 'public');
    if (!fs.existsSync(publicDir)) {
      fs.mkdirSync(publicDir);
    }
    
    // Write the statistics to a JSON file
    const filePath = path.join(publicDir, 'cohort-statistics.json');
    fs.writeFileSync(filePath, JSON.stringify(statistics, null, 2));
    
    return res.status(200).json({ message: 'Statistics saved successfully' });
  } catch (error) {
    console.error('Error saving statistics:', error);
    return res.status(500).json({ message: 'Error saving statistics', error: String(error) });
  }
}
