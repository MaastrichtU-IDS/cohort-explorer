'use client';

import React, {useState} from 'react';
import {AlertTriangle, Download, Play} from 'react-feather';
import {apiUrl} from '@/utils';
import {DcrCapabilities, projectDcrProvider} from '@/utils/dcrProvider';
import {
  DcrResultProjection,
  projectArchiveResult,
  projectJsonResult
} from '@/utils/dcrResult';

interface Props {
  dcrId: string;
  provider?: string;
  capabilities?: DcrCapabilities;
}

function responseFilename(response: Response): string {
  const disposition = response.headers.get('content-disposition') || '';
  const encodedMatch = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (encodedMatch) return decodeURIComponent(encodedMatch[1]);
  const match = disposition.match(/filename="?([^";]+)"?/i);
  return match?.[1] || 'aggregate-result.zip';
}

export function DcrResultPanel({dcrId, provider, capabilities}: Props) {
  const providerUi = projectDcrProvider(provider, capabilities);
  const [result, setResult] = useState<DcrResultProjection | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!providerUi.canRunResult) return null;

  const fetchResult = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiUrl}/compute-get-output/${encodeURIComponent(dcrId)}`, {
        credentials: 'include'
      });
      if (!response.ok) {
        throw new Error(`Result request failed: ${response.status} ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/zip') || contentType.includes('application/octet-stream')) {
        const blob = await response.blob();
        const filename = responseFilename(response);
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        setResult(projectArchiveResult(filename, blob.size));
      } else {
        setResult(projectJsonResult(await response.json()));
      }
    } catch (caught: any) {
      setError(caught?.message || 'Failed to retrieve computation result');
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className="mt-3 pt-3 border-t border-base-300"
      data-testid="dcr-result-panel"
    >
      <button
        className="btn btn-sm btn-outline gap-2"
        onClick={fetchResult}
        disabled={isLoading}
        data-testid="dcr-result-run"
      >
        <Play size={14} />
        {isLoading ? 'Running / fetching result...' : result ? 'Refresh aggregate result' : 'Run / fetch aggregate result'}
      </button>

      {error && (
        <div className="alert alert-error mt-2 py-2 text-sm">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {result?.kind === 'archive' && (
        <div
          className="alert alert-success mt-2 py-2 text-sm"
          data-testid="dcr-result-ready"
        >
          <Download size={16} />
          <span>
            Result ready: {result.filename} ({result.byteSize.toLocaleString()} bytes)
          </span>
        </div>
      )}

      {result?.kind === 'json' && (
        <div
          className="mt-2 max-h-80 overflow-auto rounded border border-base-300 bg-base-200"
          data-testid="dcr-result-ready"
        >
          <table className="table table-xs">
            <thead>
              <tr>{result.columns.map(column => <th key={column}>{column}</th>)}</tr>
            </thead>
            <tbody>
              {result.rows.map((row, index) => (
                <tr key={index}>
                  {result.columns.map(column => <td key={column}>{row[column]}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default DcrResultPanel;
