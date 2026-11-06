import fs from 'fs';
import path from 'path';

// Python's json.dump writes bare NaN / Infinity / -Infinity for non-finite
// numbers. That is valid Python but NOT valid JSON, and JSON.parse rejects it
// ("Unexpected token 'N'"), which used to make a whole cohort's EDA unreadable.
// Replace those literals with null, skipping anything inside a string so a
// value like "NaN reported by the site" is left untouched.
function nullifyNonFinite(text) {
  let out = '';
  let inString = false;
  let escaped = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      out += ch;
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') {
      inString = true;
      out += ch;
      continue;
    }
    if (ch === 'N' && text.startsWith('NaN', i)) {
      out += 'null';
      i += 2;
      continue;
    }
    if (ch === 'I' && text.startsWith('Infinity', i)) {
      out += 'null';
      i += 7;
      continue;
    }
    if (ch === '-' && text.startsWith('-Infinity', i)) {
      out += 'null';
      i += 8;
      continue;
    }
    out += ch;
  }
  return out;
}

function parseEdaText(raw) {
  try {
    return JSON.parse(raw);           // fast path: already valid JSON
  } catch (error) {
    return JSON.parse(nullifyNonFinite(raw));
  }
}

export default function handler(req, res) {
  const { cohortName } = req.query;

  const dcrOutputDir = path.join('/data', `dcr_output_${cohortName}`);
  // ?version=v1 : the file the profiling runs wrote until July 2026 (not rewritten since).
  // ?version=v2 : the same format as produced by every run since July 2026, under a
  //               new name so the old file is never overwritten; the default.
  // Each fails with 404 naming the missing file; the other version is hinted when present.
  const version = String(req.query.version || 'v2').toLowerCase();
  const files = {
    v1: path.join(dcrOutputDir, `eda_output_${cohortName}.json`),
    v2: path.join(dcrOutputDir, `eda_output_flat_v2_${cohortName}.json`),
  };
  if (!files[version]) {
    return res.status(422).json({ detail: `version must be 'v1' or 'v2', not '${version}'` });
  }
  const edaFilePath = files[version];
  if (!fs.existsSync(edaFilePath)) {
    const other = version === 'v2' ? 'v1' : 'v2';
    const hint = fs.existsSync(files[other]) ? ` (a ${other} file exists: ask for version=${other})` : '';
    const detail = fs.existsSync(dcrOutputDir)
      ? `No ${version} EDA output for cohort '${cohortName}' (${path.basename(edaFilePath)} is not there)${hint}`
      : `No EDA output for cohort '${cohortName}': no DCR output directory`;
    if (req.method === 'HEAD') return res.status(404).end();
    return res.status(404).json({ detail });
  }

  // HEAD request — the file exists
  if (req.method === 'HEAD') {
    res.setHeader('X-EDA-Version', version);
    return res.status(200).end();
  }

  // GET request — return the JSON
  try {
    const raw = fs.readFileSync(edaFilePath, 'utf-8');
    const data = parseEdaText(raw);
    res.setHeader('X-EDA-Version', version);
    return res.status(200).json(data);
  } catch (error) {
    console.error('Error reading EDA output:', error);
    return res.status(500).json({ detail: `Error reading EDA output: ${error.message}` });
  }
}
