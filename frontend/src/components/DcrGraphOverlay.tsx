'use client';

// Node graph of a DCR in its own overlay (My DCRs page): the dependency graph
// large on the left; on the right a side box with the node under the pointer
// (or the clicked, pinned node): what it is, what it reads and writes, who
// provisions it or may run it, and where its output goes. No AI involved.
import React, {useEffect, useMemo, useState} from 'react';
import {AlertTriangle, X} from 'react-feather';
import {RoomNode, RoomOverview, fetchRoomOverview} from '@/components/ai/chatClient';
import {DcrGraph, KIND_STYLE} from '@/components/DcrGraph';

export interface DcrParticipantLite {
  email?: string;
  roles?: string[];
  data_owner_of?: string[];
  analyst_of?: string[];
}

export function DcrGraphOverlay({
  dcrId,
  dcrTitle,
  participants,
  onClose
}: {
  dcrId: string;
  dcrTitle: string;
  // The DCR's participants with their per-node permissions (from the history record).
  participants: DcrParticipantLite[];
  onClose: () => void;
}) {
  const [room, setRoom] = useState<RoomOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [pinned, setPinned] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchRoomOverview(dcrId)
      .then(r => {
        if (!cancelled) setRoom(r);
      })
      .catch(e => {
        if (!cancelled) setError(e?.message || 'Could not load the room structure.');
      });
    return () => {
      cancelled = true;
    };
  }, [dcrId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const byName = useMemo(() => new Map((room?.nodes || []).map(n => [n.name, n])), [room]);
  const shown: RoomNode | undefined = byName.get(hover || pinned || '');
  const readers = (name: string) => (room?.nodes || []).filter(n => n.dependencies.includes(name));
  const owners = (name: string) => participants.filter(p => (p.data_owner_of || []).includes(name)).map(p => p.email || '?');
  const runners = (name: string) => participants.filter(p => (p.analyst_of || []).includes(name)).map(p => p.email || '?');
  const kindLabel = (n: RoomNode) => KIND_STYLE[n.kind].label;
  // Everyone in the room: say so instead of listing every address.
  const allParticipants = participants.length;
  const People = ({emails}: {emails: string[]}) =>
    allParticipants > 0 && emails.length === allParticipants ? (
      <p className="text-xs">All DCR participants ({allParticipants}).</p>
    ) : (
      <ul className="space-y-0.5">{emails.map(e => <li key={e} className="font-mono text-xs">{e}</li>)}</ul>
    );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-3 md:p-6" onClick={onClose}>
      <div className="bg-base-100 rounded-2xl shadow-2xl w-full h-full max-w-[1600px] flex flex-col overflow-hidden" onClick={e => e.stopPropagation()} role="dialog" aria-label="Node graph of the data clean room">
        <div className="px-5 py-3 border-b border-base-300 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h3 className="font-bold">Node graph</h3>
            <div className="text-xs text-base-content/60 truncate">{dcrTitle || 'Untitled DCR'}</div>
          </div>
          <button className="btn btn-sm btn-ghost btn-circle" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_24rem] divide-y lg:divide-y-0 lg:divide-x divide-base-300">
          <section className="min-h-0 flex flex-col">
            {!room && !error && (
              <div className="flex justify-center py-12"><span className="loading loading-spinner loading-md" /></div>
            )}
            {error && (
              <div className="alert alert-error text-sm m-4 w-auto"><AlertTriangle size={16} /> <span>{error}</span></div>
            )}
            {room && (
              <DcrGraph
                room={room}
                selected={pinned}
                onHover={setHover}
                onSelect={name => setPinned(p => (p === name ? null : name))}
                selectableKinds="all"
                pinHighlight="ancestors"
              />
            )}
          </section>

          <aside className="min-h-0 overflow-y-auto p-4 text-sm space-y-3">
            {!shown && room && (
              <>
                <h4 className="font-semibold">This room</h4>
                <ul className="text-base-content/80 space-y-1">
                  <li>{room.nodes.filter(n => n.kind === 'cohort_data').length} cohort data node{room.nodes.filter(n => n.kind === 'cohort_data').length === 1 ? '' : 's'}, {room.nodes.filter(n => n.kind === 'python').length} compute nodes, {room.nodes.filter(n => n.kind === 'airlock').length} airlock{room.nodes.filter(n => n.kind === 'airlock').length === 1 ? '' : 's'}.</li>
                  {room.provisioning?.known && (
                    <li>{room.provisioning.provisioned} of {room.provisioning.data_nodes} data nodes hold data{room.provisioning.provisioned < room.provisioning.data_nodes ? '; nodes reading an empty one cannot run yet.' : '.'}</li>
                  )}
                  <li>{participants.length} participant{participants.length === 1 ? '' : 's'}.</li>
                </ul>
                <div>
                  <div className="text-xs font-semibold uppercase text-base-content/60 mb-1">Where analysts see output</div>
                  {room.exits.length === 0 && <p className="text-base-content/50 italic">No airlock and no node a participant may run directly.</p>}
                  <ul className="space-y-1">
                    {room.exits.map(e => (
                      <li key={e.node} className="flex items-start gap-2">
                        <span className="inline-block mt-1.5 w-2.5 h-2.5 rounded-sm shrink-0 border" style={{background: KIND_STYLE[e.kind === 'airlock' ? 'airlock' : 'python'].fill, borderColor: KIND_STYLE[e.kind === 'airlock' ? 'airlock' : 'python'].stroke}} />
                        <span>
                          <button className="font-mono text-xs link link-hover" onClick={() => setPinned(e.node)}>{e.node}</button>
                          <span className="text-xs text-base-content/60"> · {e.kind === 'airlock' ? `airlock, ${e.viewers} viewer${e.viewers === 1 ? '' : 's'}${e.quota_bytes ? `, ${Math.round(e.quota_bytes / 1e6)} MB` : ''}` : `${e.runners} can run it`}</span>
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
                <p className="text-xs text-base-content/50">Hover a node for its details; click to pin it here and light up everything upstream of it. Click it again to unpin.</p>
              </>
            )}

            {shown && (
              <>
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <h4 className="font-mono text-sm font-semibold break-all">{shown.name}</h4>
                    <div className="flex items-center gap-1.5 text-xs text-base-content/60">
                      <span className="inline-block w-2.5 h-2.5 rounded-sm border" style={{background: KIND_STYLE[shown.kind].fill, borderColor: KIND_STYLE[shown.kind].stroke}} />
                      {kindLabel(shown)}
                      {pinned === shown.name && <span className="badge badge-ghost badge-xs ml-1">pinned</span>}
                    </div>
                  </div>
                  {pinned === shown.name && (
                    <button className="btn btn-ghost btn-xs btn-circle" onClick={() => setPinned(null)} aria-label="Unpin"><X size={14} /></button>
                  )}
                </div>

                {shown.provisioned !== undefined && shown.provisioned !== null && (
                  <div className={`rounded-md px-3 py-2 text-xs ${shown.provisioned ? 'bg-success/10' : 'bg-base-200 text-base-content/70'}`}>
                    {shown.provisioned ? <>Data provisioned{shown.provisioned_by ? ` by ${shown.provisioned_by}` : ''}.</> : 'No data provisioned yet. Nodes that read it can run once the owner uploads.'}
                  </div>
                )}

                {shown.kind !== 'python' && shown.kind !== 'airlock' && shown.kind !== 'environment' && (
                  <div>
                    <div className="text-xs font-semibold uppercase text-base-content/60 mb-1">Data owners ({owners(shown.name).length})</div>
                    {owners(shown.name).length === 0 ? <p className="text-base-content/50 italic">Nobody owns this node.</p> : <People emails={owners(shown.name)} />}
                  </div>
                )}

                {shown.dependencies.length > 0 && (
                  <div>
                    <div className="text-xs font-semibold uppercase text-base-content/60 mb-1">Reads ({shown.dependencies.length})</div>
                    <ul className="space-y-0.5">
                      {shown.dependencies.map(d => {
                        const n = byName.get(d);
                        return (
                          <li key={d} className="flex items-center gap-1.5 text-xs">
                            {n && <span className="inline-block w-2 h-2 rounded-sm border shrink-0" style={{background: KIND_STYLE[n.kind].fill, borderColor: KIND_STYLE[n.kind].stroke}} />}
                            <button className="font-mono link link-hover text-left" onClick={() => setPinned(d)}>{d}</button>
                            {n?.provisioned === false && <span className="badge badge-ghost badge-xs">no data yet</span>}
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                )}

                {shown.kind === 'python' && (
                  <div>
                    <div className="text-xs font-semibold uppercase text-base-content/60 mb-1">Writes</div>
                    {shown.outputs && shown.outputs.length > 0 && (
                      <ul className="flex flex-wrap gap-1 mb-1">{shown.outputs.map(o => <li key={o} className="badge badge-outline font-mono text-[11px]">{o}</li>)}</ul>
                    )}
                    {shown.output_kinds && shown.output_kinds.length > 0 ? (
                      <p className="text-xs text-base-content/60">Writes {shown.output_kinds.join(', ')}{shown.outputs?.length ? '' : ' (file names computed at run time)'}.</p>
                    ) : !(shown.outputs && shown.outputs.length > 0) && <p className="text-xs text-base-content/50 italic">No file output found in the script.</p>}
                    {shown.external_writes && <p className="text-xs text-base-content/60 mt-1">Also uses {(shown.external_packages || []).join(', ')}, which writes its own files into the output folder.</p>}
                    {shown.prints && <p className="text-xs text-base-content/60 mt-1">Prints or logs text{shown.logs.length ? ` (logs shown ${shown.logs.join(' and ').replace(/_/g, ' ')})` : ''}.</p>}
                  </div>
                )}

                {shown.kind === 'airlock' && (
                  <p className="text-xs text-base-content/70">Shows the output of {shown.dependencies[0] ? <code>{shown.dependencies[0]}</code> : 'its source'} to its viewers{shown.quota_bytes ? `, up to ${Math.round(shown.quota_bytes / 1e6)} MB` : ''}. Airlocks have no Run button: running a node downstream computes the chain.</p>
                )}

                {(shown.kind === 'python' || shown.kind === 'airlock') && (
                  <div>
                    <div className="text-xs font-semibold uppercase text-base-content/60 mb-1">{shown.kind === 'airlock' ? 'Viewers' : 'Can run it'} ({runners(shown.name).length})</div>
                    {runners(shown.name).length === 0 ? <p className="text-base-content/50 italic">Nobody directly: it only runs as a dependency of other nodes.</p> : <People emails={runners(shown.name)} />}
                  </div>
                )}

                {readers(shown.name).length > 0 && (
                  <div>
                    <div className="text-xs font-semibold uppercase text-base-content/60 mb-1">Read by ({readers(shown.name).length})</div>
                    <ul className="space-y-0.5">
                      {readers(shown.name).map(r => (
                        <li key={r.name} className="flex items-center gap-1.5 text-xs">
                          <span className="inline-block w-2 h-2 rounded-sm border shrink-0" style={{background: KIND_STYLE[r.kind].fill, borderColor: KIND_STYLE[r.kind].stroke}} />
                          <button className="font-mono link link-hover text-left" onClick={() => setPinned(r.name)}>{r.name}</button>
                          {r.kind === 'airlock' && <span className="text-base-content/50">airlock</span>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
