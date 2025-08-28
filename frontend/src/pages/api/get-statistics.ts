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

// Default statistics to return if file doesn't exist
const defaultStatistics: CohortStatistics = {
  totalCohorts: 0,
  cohortsWithMetadata: 0,
  cohortsWithAggregateAnalysis: 0,
  totalPatients: 0,
  patientsInCohortsWithMetadata: 0,
  totalVariables: 0
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Only allow GET requests
  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const filePath = path.join(process.cwd(), 'public', 'cohort-statistics.json');
    
    // Check if the file exists
    if (!fs.existsSync(filePath)) {
      return res.status(200).json(defaultStatistics);
    }
    
    // Read the statistics from the JSON file
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const statistics = JSON.parse(fileContent);
    
    return res.status(200).json(statistics);
  } catch (error) {
    console.error('Error reading statistics:', error);
    // Return default statistics in case of error
    return res.status(200).json(defaultStatistics);
  }
}
