'use client';

// Inline viewer and download helper for DCR result files, used on My DCRs and
// on Shared Results. The API always serves files as attachments; the
// bytes are fetched here and rendered by type (HTML only inside a sandboxed
// iframe, so a result never runs with the explorer's origin).
import React, {useEffect, useState} from 'react';
import {createPortal} from 'react-dom';
import {AlertTriangle, Download, X} from 'react-feather';

const TEXT_EXT = ['csv', 'tsv', 'txt', 'json', 'md', 'log', 'py', 'r', 'sql', 'yaml', 'yml', 'xml'];
const IMAGE_EXT = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp'];
const MAX_TEXT_BYTES = 5 * 1024 * 1024;

export const fileExt = (name: string) => (name.split('.').pop() || '').toLowerCase();

export function isViewable(name: string): boolean {
  const ext = fileExt(name);
  return TEXT_EXT.includes(ext) || IMAGE_EXT.includes(ext) || ['pdf', 'html', 'htm'].includes(ext);
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDateTime(iso?: string | null): string {
  if (!iso) return '';
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString();
}

async function fetchBlob(url: string): Promise<Blob> {
  const res = await fetch(url, {credentials: 'include'});
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      detail = (await res.json())?.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.blob();
}

export async function downloadFile(url: string, fileName: string): Promise<void> {
  const blob = await fetchBlob(url);
  const href = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = href;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}

function parseDelimited(text: string, sep: string): string[][] {
  return text
    .replace(/\r?\n$/, '')
    .split(/\r?\n/)
    .map(line => {
      const out: string[] = [];
      let cur = '';
      let q = false;
      for (const ch of line) {
        if (ch === '"') q = !q;
        else if (ch === sep && !q) {
          out.push(cur);
          cur = '';
        } else cur += ch;
      }
      out.push(cur);
      return out;
    });
}

function TableView({text, sep}: {text: string; sep: string}) {
  const rows = parseDelimited(text, sep);
  const shown = rows.slice(0, 1001);
  if (rows.length === 0) return <div className="text-sm opacity-60">Empty file</div>;
  return (
    <div className="overflow-auto max-h-[65vh]">
      <table className="table table-xs table-zebra table-pin-rows">
        <thead>
          <tr>{shown[0].map((h, i) => <th key={i}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {shown.slice(1).map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j} className="whitespace-nowrap">{c}</td>)}</tr>
          ))}
        </tbody>
      </table>
      {rows.length > shown.length && (
        <div className="text-xs opacity-60 mt-2">Showing the first 1,000 of {rows.length - 1} rows. Download the file for the rest.</div>
      )}
    </div>
  );
}

/** Renders a result file's content inline. */
export function ResultFileContent({url, fileName}: {url: string; fileName: string}) {
  const [text, setText] = useState<string | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const ext = fileExt(fileName);

  useEffect(() => {
    let cancelled = false;
    let created: string | null = null;
    setLoading(true);
    setError(null);
    setText(null);
    setObjectUrl(null);
    fetchBlob(url)
      .then(async blob => {
        if (cancelled) return;
        if (TEXT_EXT.includes(ext) || ext === 'html' || ext === 'htm') {
          if (blob.size > MAX_TEXT_BYTES) throw new Error(`File too large to preview (${formatBytes(blob.size)}). Download it instead.`);
          const t = await blob.text();
          if (!cancelled) setText(t);
        } else {
          // Re-type PDFs so the browser's viewer opens them; images keep their type.
          const typed = ext === 'pdf' ? new Blob([blob], {type: 'application/pdf'}) : blob;
          created = URL.createObjectURL(typed);
          setObjectUrl(created);
        }
      })
      .catch(err => !cancelled && setError(err?.message || 'Failed to load file'))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
      if (created) URL.revokeObjectURL(created);
    };
  }, [url, ext]);

  if (loading) return <div className="flex justify-center py-10"><span className="loading loading-spinner" /></div>;
  if (error) return <div className="alert alert-error text-sm"><AlertTriangle size={16} /> {error}</div>;
  // eslint-disable-next-line @next/next/no-img-element -- a blob: URL, nothing for next/image to optimize
  if (objectUrl && IMAGE_EXT.includes(ext)) return <img src={objectUrl} alt={fileName} className="max-w-full max-h-[65vh] mx-auto" />;
  if (objectUrl && ext === 'pdf') return <iframe src={objectUrl} title={fileName} className="w-full h-[65vh] border rounded" />;
  if (text !== null && (ext === 'html' || ext === 'htm'))
    return <iframe srcDoc={text} sandbox="allow-scripts" title={fileName} className="w-full h-[65vh] border rounded bg-white" />;
  if (text !== null && (ext === 'csv' || ext === 'tsv')) return <TableView text={text} sep={ext === 'tsv' ? '\t' : ','} />;
  if (text !== null && ext === 'json') {
    let pretty = text;
    try {
      pretty = JSON.stringify(JSON.parse(text), null, 2);
    } catch {}
    return <pre className="text-xs bg-base-200 rounded p-3 overflow-auto max-h-[65vh] whitespace-pre-wrap">{pretty}</pre>;
  }
  if (text !== null) return <pre className="text-xs bg-base-200 rounded p-3 overflow-auto max-h-[65vh] whitespace-pre-wrap">{text}</pre>;
  return <div className="text-sm opacity-60">This file type cannot be previewed. Download it instead.</div>;
}

/** Modal wrapping ResultFileContent with a download button. */
export function ResultFileModal({url, fileName, subtitle, onClose}: {url: string; fileName: string; subtitle?: string; onClose: () => void}) {
  const [dlError, setDlError] = useState<string | null>(null);
  const modal = (
    <div className="modal modal-open z-[10000]" onMouseDown={onClose}>
      <div className="modal-box max-w-6xl w-11/12 flex flex-col max-h-[90vh]" onMouseDown={e => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-3 mb-3 shrink-0">
          <div className="min-w-0">
            <h3 className="font-bold text-lg break-all">{fileName}</h3>
            {subtitle && <div className="text-xs opacity-60 break-all">{subtitle}</div>}
          </div>
          <div className="flex gap-2 shrink-0">
            <button
              className="btn btn-sm btn-outline gap-1"
              onClick={() => downloadFile(url, fileName).catch(e => setDlError(e?.message || 'Download failed'))}
            >
              <Download size={14} /> Download
            </button>
            <button className="btn btn-sm btn-ghost" onClick={onClose} aria-label="Close">
              <X size={16} />
            </button>
          </div>
        </div>
        {dlError && <div className="alert alert-error text-sm mb-2">{dlError}</div>}
        <div className="overflow-auto min-h-0">
          <ResultFileContent url={url} fileName={fileName} />
        </div>
      </div>
    </div>
  );
  return typeof document === 'undefined' ? modal : createPortal(modal, document.body);
}
