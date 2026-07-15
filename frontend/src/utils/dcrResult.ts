export interface JsonDcrResult {
  kind: 'json';
  status: 'ready';
  columns: string[];
  rows: Record<string, string>[];
}

export interface ArchiveDcrResult {
  kind: 'archive';
  status: 'ready';
  filename: string;
  byteSize: number;
}

export type DcrResultProjection = JsonDcrResult | ArchiveDcrResult;

function displayValue(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function projectRows(rows: Record<string, unknown>[]): JsonDcrResult {
  const columns = Array.from(new Set(rows.flatMap(row => Object.keys(row))));
  return {
    kind: 'json',
    status: 'ready',
    columns,
    rows: rows.map(row =>
      Object.fromEntries(columns.map(column => [column, displayValue(row[column])]))
    )
  };
}

export function projectJsonResult(payload: unknown): JsonDcrResult {
  if (Array.isArray(payload) && payload.every(row => row && typeof row === 'object' && !Array.isArray(row))) {
    return projectRows(payload as Record<string, unknown>[]);
  }

  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    const objectPayload = payload as Record<string, unknown>;
    if (
      Array.isArray(objectPayload.rows) &&
      objectPayload.rows.every(row => row && typeof row === 'object' && !Array.isArray(row))
    ) {
      return projectRows(objectPayload.rows as Record<string, unknown>[]);
    }
    return {
      kind: 'json',
      status: 'ready',
      columns: ['metric', 'value'],
      rows: Object.entries(objectPayload).map(([metric, value]) => ({
        metric,
        value: displayValue(value)
      }))
    };
  }

  return {
    kind: 'json',
    status: 'ready',
    columns: ['value'],
    rows: [{value: displayValue(payload)}]
  };
}

export function projectArchiveResult(filename: string, byteSize: number): ArchiveDcrResult {
  return {
    kind: 'archive',
    status: 'ready',
    filename,
    byteSize
  };
}
