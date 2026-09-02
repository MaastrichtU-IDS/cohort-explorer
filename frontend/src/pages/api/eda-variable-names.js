import fs from 'fs';
import path from 'path';

// Map of cohort id -> lowercased names of the variables that have summary
// statistics (EDA) on record. Served in one small payload so cross-cohort
// views (the explore page's search results table) can mark variables without
// downloading every cohort's full EDA JSON. Results are cached per file mtime.

// Python's json.dump writes bare NaN / Infinity for non-finite numbers, which
// JSON.parse rejects; replace them with null outside of strings (same logic as
// /api/cohort-eda-output).
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
    return JSON.parse(raw);
  } catch (error) {
    return JSON.parse(nullifyNonFinite(raw));
  }
}

// A v1 entry that carries ONLY '(metadata dictionary)' fields has no actual
// statistics (and no graph) - skip it.
function hasStats(entry) {
  if (!entry || typeof entry !== 'object') return false;
  return Object.keys(entry).some(k => !k.toLowerCase().includes('(metadata dictionary)'));
}

const DATA_DIR = '/data';
// cohortId -> {mtimeMs, names}
const cache = new Map();

function namesForCohort(cohortId) {
  const dir = path.join(DATA_DIR, `dcr_output_${cohortId}`);
  const v1Path = path.join(dir, `eda_output_${cohortId}.json`);
  const v2Path = path.join(dir, `eda_output_v2_${cohortId}.json`);
  const filePath = fs.existsSync(v1Path) ? v1Path : v2Path;
  if (!fs.existsSync(filePath)) return null;
  const mtimeMs = fs.statSync(filePath).mtimeMs;
  const cached = cache.get(cohortId);
  if (cached && cached.mtimeMs === mtimeMs) return cached.names;
  try {
    const data = parseEdaText(fs.readFileSync(filePath, 'utf-8'));
    const entries = data && typeof data === 'object' && data.variables && typeof data.variables === 'object'
      ? data.variables
      : data;
    const names = Object.entries(entries || {})
      .filter(([, entry]) => hasStats(entry))
      .map(([name]) => String(name).toLowerCase().trim());
    cache.set(cohortId, { mtimeMs, names });
    return names;
  } catch (error) {
    console.error(`eda-variable-names: unreadable EDA for ${cohortId}:`, error.message);
    return null;
  }
}

export default function handler(req, res) {
  const out = {};
  let dirs = [];
  try {
    dirs = fs.readdirSync(DATA_DIR).filter(d => d.startsWith('dcr_output_'));
  } catch (error) {
    return res.status(200).json(out);
  }
  for (const d of dirs) {
    const cohortId = d.slice('dcr_output_'.length);
    const names = namesForCohort(cohortId);
    if (names && names.length > 0) out[cohortId] = names;
  }
  res.setHeader('Cache-Control', 'private, max-age=300');
  return res.status(200).json(out);
}
