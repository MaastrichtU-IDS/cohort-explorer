import fs from 'fs';
import path from 'path';

export default function handler(req, res) {
  const { cohortName } = req.query;

  const dcrOutputDir = path.join('/data', `dcr_output_${cohortName}`);
  // Two EDA generations with DIFFERENT file names: the legacy v1 file and the
  // v2 file (eda_output_v2_<id>.json). Serve v1 when it exists (the dashboards
  // grew up on it), else fall back to v2 so v2-only cohorts are not reported
  // as having no EDA.
  const v1Path = path.join(dcrOutputDir, `eda_output_${cohortName}.json`);
  const v2Path = path.join(dcrOutputDir, `eda_output_v2_${cohortName}.json`);
  const edaFilePath = fs.existsSync(v1Path) ? v1Path : v2Path;

  // HEAD request — just check existence
  if (req.method === 'HEAD') {
    if (fs.existsSync(edaFilePath)) {
      return res.status(200).end();
    }
    return res.status(404).end();
  }

  // GET request — return the JSON
  if (!fs.existsSync(edaFilePath)) {
    return res.status(404).json({ detail: `EDA output file not found for cohort '${cohortName}'` });
  }

  try {
    const raw = fs.readFileSync(edaFilePath, 'utf-8');
    const data = JSON.parse(raw);
    return res.status(200).json(data);
  } catch (error) {
    console.error('Error reading EDA output:', error);
    return res.status(500).json({ detail: `Error reading EDA output: ${error.message}` });
  }
}
