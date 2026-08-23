'use client';

// Results of a no-code DCR analysis: runs the generated node through the platform
// (once the data owners have provisioned their data) and shows the aggregate
// figures and tables in the explorer, each with its provenance subtext.
import React, {useEffect, useState} from 'react';
import Link from 'next/link';
import {useRouter} from 'next/router';
import {AlertTriangle, Download, Play, RefreshCw} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {fetchNocodeResults, fetchResultBlob, fetchResultText, resultFileUrl, runNocode} from '@/components/nocode/client';

function StatsBlock({text}: {text: string}) {
  const rows = text
    .trim()
    .split(/\r?\n/)
    .map(l => {
      const i = l.indexOf(':');
      return i > 0 ? [l.slice(0, i).trim(), l.slice(i + 1).trim()] : [l, ''];
    });
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-sm max-w-xl">
      {rows.map(([k, v], i) => (
        <React.Fragment key={i}>
          <dt className="font-mono text-xs pt-0.5 text-base-content/80">{k}</dt>
          <dd className="tabular-nums">{v}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

function CsvTable({text}: {text: string}) {
  const rows = text
    .trim()
    .split(/\r?\n/)
    .map(line => {
      // minimal CSV parse (handles quoted commas)
      const out: string[] = [];
      let cur = '';
      let q = false;
      for (const ch of line) {
        if (ch === '"') q = !q;
        else if (ch === ',' && !q) {
          out.push(cur);
          cur = '';
        } else cur += ch;
      }
      out.push(cur);
      return out;
    });
  if (rows.length === 0) return null;
  return (
    <div className="overflow-x-auto">
      <table className="table table-xs table-zebra">
        <thead>
          <tr>{rows[0].map((h, i) => <th key={i}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.slice(1).map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j} className="tabular-nums">{c}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function NocodeResultsPage() {
  const router = useRouter();
  const {userEmail} = useCohorts();
  const dcr = typeof router.query.dcr === 'string' ? router.query.dcr : '';
  const node = typeof router.query.node === 'string' ? router.query.node : '';
  const title = typeof router.query.title === 'string' ? router.query.title : '';
  const [summary, setSummary] = useState<any>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'running' | 'none' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [images, setImages] = useState<Record<string, string>>({});
  const [tables, setTables] = useState<Record<string, string>>({});

  const loadAssets = async (s: any) => {
    const imgs: Record<string, string> = {};
    const tabs: Record<string, string> = {};
    for (const item of s.items || []) {
      try {
        if (item.figure) imgs[item.figure] = await fetchResultBlob(dcr, node, item.figure);
        if (item.table) tabs[item.table] = await fetchResultText(dcr, node, item.table);
        if (item.text) tabs[item.text] = await fetchResultText(dcr, node, item.text);
        if (item.doc) tabs[item.doc] = await fetchResultText(dcr, node, item.doc);
      } catch {
        /* skip */
      }
    }
    setImages(imgs);
    setTables(tabs);
  };

  const load = () => {
    if (!dcr || !node) return;
    setStatus('loading');
    fetchNocodeResults(dcr, node)
      .then(async s => {
        setSummary(s);
        await loadAssets(s);
        setStatus('idle');
      })
      .catch(e => {
        if (e.status === 404) setStatus('none');
        else {
          setError(e.message);
          setStatus('error');
        }
      });
  };

  useEffect(load, [dcr, node]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = () => {
    setStatus('running');
    setError(null);
    runNocode(dcr, node)
      .then(async r => {
        setSummary(r.summary);
        await loadAssets(r.summary);
        setStatus('idle');
      })
      .catch(e => {
        setError(e.message);
        setStatus('error');
      });
  };

  // userEmail is '' while the session is still being verified (see
  // CohortsContext): show a spinner rather than flashing the login notice.
  if (userEmail === '') {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <span className="loading loading-spinner loading-lg"></span>
      </div>
    );
  }
  if (!userEmail) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <div className="alert alert-warning max-w-md">
          <AlertTriangle size={20} />
          <span>Please log in to view results.</span>
        </div>
      </div>
    );
  }

  return (
    <main className="max-w-6xl mx-auto px-4 py-6">
      <div className="flex flex-wrap items-end justify-between gap-3 mb-4">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-base-content/50">No-code DCR results</div>
          <h1 className="text-2xl font-bold">{summary?.title || title || node}</h1>
          <div className="text-xs text-base-content/50 font-mono mt-1">
            DCR {dcr.slice(0, 12)}… · node {node}
            {summary?.fetched_at && <> · fetched {summary.fetched_at.replace('T', ' ')}</>}
          </div>
        </div>
        <div className="flex gap-2">
          <Link href="/nocode-dcr" className="btn btn-sm btn-ghost">
            New no-code DCR
          </Link>
          <button className="btn btn-sm btn-primary gap-1" onClick={run} disabled={status === 'running' || status === 'loading'}>
            {status === 'running' ? (
              <>
                <span className="loading loading-spinner loading-xs" /> Running in the clean room…
              </>
            ) : summary ? (
              <>
                <RefreshCw size={14} /> Run again
              </>
            ) : (
              <>
                <Play size={14} /> Run the analysis
              </>
            )}
          </button>
        </div>
      </div>

      {status === 'none' && !summary && (
        <div className="rounded-xl border border-base-300 bg-base-100 p-6 max-w-2xl">
          <p className="font-semibold mb-1">No results yet.</p>
          <p className="text-sm text-base-content/70">
            Once the data owners of every cohort have provisioned their data in the clean room, press <b>Run the analysis</b>.
            The computation happens inside the enclave; only aggregate figures and tables come back here. Running usually
            takes one to a few minutes.
          </p>
        </div>
      )}

      {error && (
        <div className="alert alert-error text-sm mb-4">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {summary && (
        <div className="space-y-6">
          {summary.notes && summary.notes.length > 0 && (
            <div className="rounded-lg bg-amber-50 border border-amber-200 text-amber-900 text-sm p-3">
              <div className="font-semibold mb-1">Notes from the computation</div>
              <ul className="list-disc ml-5">{summary.notes.map((n: string, i: number) => <li key={i}>{n}</li>)}</ul>
            </div>
          )}
          {(summary.items || []).map((item: any, i: number) => (
            <section key={i} className="rounded-xl border border-base-300 bg-base-100 p-4">
              {!item.doc && <h2 className="font-semibold mb-2">{item.caption}</h2>}
              {item.doc && (
                <details open>
                  <summary className="font-semibold cursor-pointer">{item.caption}</summary>
                  {tables[item.doc] ? (
                    <pre className="mt-2 text-xs leading-relaxed whitespace-pre-wrap text-base-content/80">{tables[item.doc]}</pre>
                  ) : (
                    <div className="text-sm text-base-content/50 mt-2">Loading…</div>
                  )}
                  <a className="btn btn-xs btn-ghost gap-1 mt-1" href={resultFileUrl(dcr, node, item.doc)} target="_blank" rel="noreferrer">
                    <Download size={12} /> {item.doc}
                  </a>
                </details>
              )}
              {item.figure && (
                <div>
                  {images[item.figure] ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={images[item.figure]} alt={item.caption} className="max-w-full rounded-lg border border-base-200" />
                  ) : (
                    <div className="text-sm text-base-content/50">Loading figure…</div>
                  )}
                  {item.provenance && (
                    <pre className="mt-2 text-[11px] leading-snug whitespace-pre-wrap text-base-content/60 bg-base-200 rounded p-2">{item.provenance}</pre>
                  )}
                </div>
              )}
              {item.text && (
                <div>
                  {tables[item.text] ? <StatsBlock text={tables[item.text]} /> : <div className="text-sm text-base-content/50">Loading…</div>}
                  <a className="btn btn-xs btn-ghost gap-1 mt-1" href={resultFileUrl(dcr, node, item.text)} target="_blank" rel="noreferrer">
                    <Download size={12} /> {item.text}
                  </a>
                </div>
              )}
              {item.table && (
                <div>
                  {tables[item.table] ? <CsvTable text={tables[item.table]} /> : <div className="text-sm text-base-content/50">Loading table…</div>}
                  <a className="btn btn-xs btn-ghost gap-1 mt-1" href={resultFileUrl(dcr, node, item.table)} target="_blank" rel="noreferrer">
                    <Download size={12} /> {item.table}
                  </a>
                </div>
              )}
            </section>
          ))}
          {summary.provenance_md && (
            <section className="rounded-xl border border-base-300 bg-base-100 p-4">
              <h2 className="font-semibold mb-2">Mapping record</h2>
              <pre className="text-xs whitespace-pre-wrap text-base-content/70">{summary.provenance_md}</pre>
            </section>
          )}
        </div>
      )}
    </main>
  );
}
