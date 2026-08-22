// Tiny illustrative charts for the analysis-type cards: a sketch of the kind
// of figure each analysis produces. Pure inline SVG, themed with currentColor.
import React from 'react';
import {Kind} from './client';

const W = 220;
const H = 110;

function Frame({children}: {children: React.ReactNode}) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} className="text-base-content/70" role="img">
      <line x1="24" y1="92" x2={W - 8} y2="92" stroke="currentColor" strokeOpacity="0.35" />
      <line x1="24" y1="8" x2="24" y2="92" stroke="currentColor" strokeOpacity="0.35" />
      {children}
    </svg>
  );
}

const BLUE = '#3b6ea5';
const ORANGE = '#e08a2e';
const GREEN = '#3a9a6a';

function Stratified() {
  // two overlaid density curves + small box plots
  const curve = (shift: number, scale: number) => {
    const pts: string[] = [];
    for (let x = 0; x <= 100; x += 4) {
      const y = 80 * Math.exp(-Math.pow((x - 50 - shift) / (16 * scale), 2));
      pts.push(`${26 + x * 1.1},${92 - y}`);
    }
    return pts.join(' ');
  };
  return (
    <Frame>
      <polyline points={curve(-10, 1)} fill="none" stroke={BLUE} strokeWidth="2" />
      <polyline points={curve(14, 1.2)} fill="none" stroke={ORANGE} strokeWidth="2" />
      <g transform="translate(150,14)">
        <rect x="0" y="10" width="14" height="26" fill="none" stroke={BLUE} strokeWidth="1.5" />
        <line x1="0" y1="22" x2="14" y2="22" stroke={BLUE} strokeWidth="1.5" />
        <line x1="7" y1="0" x2="7" y2="10" stroke={BLUE} />
        <line x1="7" y1="36" x2="7" y2="46" stroke={BLUE} />
        <rect x="24" y="6" width="14" height="30" fill="none" stroke={ORANGE} strokeWidth="1.5" />
        <line x1="24" y1="20" x2="38" y2="20" stroke={ORANGE} strokeWidth="1.5" />
        <line x1="31" y1="0" x2="31" y2="6" stroke={ORANGE} />
        <line x1="31" y1="36" x2="31" y2="50" stroke={ORANGE} />
        <circle cx="31" cy="56" r="1.6" fill={ORANGE} />
      </g>
    </Frame>
  );
}

function Scatter() {
  // deterministic pseudo-random points around a line
  const pts: [number, number][] = [];
  let seed = 7;
  const rnd = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };
  for (let i = 0; i < 55; i++) {
    const x = rnd();
    const y = 0.15 + 0.6 * x + (rnd() - 0.5) * 0.3;
    pts.push([x, y]);
  }
  return (
    <Frame>
      {pts.map(([x, y], i) => (
        <circle key={i} cx={28 + x * (W - 44)} cy={90 - y * 80} r="2.2" fill={BLUE} fillOpacity="0.6" />
      ))}
      <line x1="28" y1={90 - 0.15 * 80} x2={W - 16} y2={90 - 0.75 * 80} stroke="#c0392b" strokeWidth="1.5" />
    </Frame>
  );
}

function Crosstab() {
  const cols = [
    [40, 22, 10],
    [18, 30, 26],
    [12, 20, 38]
  ];
  const bw = 40;
  return (
    <Frame>
      {cols.map((stack, i) => {
        let y = 92;
        return stack.map((h, j) => {
          y -= h;
          return <rect key={`${i}-${j}`} x={40 + i * (bw + 18)} y={y} width={bw} height={h} fill={[BLUE, ORANGE, GREEN][j]} />;
        });
      })}
    </Frame>
  );
}

function Compare() {
  const curve = (shift: number, scale: number) => {
    const pts: string[] = [];
    for (let x = 0; x <= 100; x += 4) {
      const y = 80 * Math.exp(-Math.pow((x - 50 - shift) / (18 * scale), 2));
      pts.push(`${26 + x * 1.1},${92 - y}`);
    }
    return pts.join(' ');
  };
  return (
    <Frame>
      <polyline points={curve(-8, 1)} fill="none" stroke={BLUE} strokeWidth="2" />
      <polyline points={curve(6, 0.8)} fill="none" stroke={ORANGE} strokeWidth="2" />
      <polyline points={curve(18, 1.3)} fill="none" stroke={GREEN} strokeWidth="2" />
    </Frame>
  );
}

function Pooled() {
  const a = [4, 10, 20, 30, 40, 44, 38, 30, 20, 12, 6, 3];
  const b = [2, 6, 12, 20, 26, 30, 28, 22, 14, 8, 4, 2];
  const bw = (W - 40) / a.length;
  return (
    <Frame>
      {a.map((h, i) => (
        <g key={i}>
          <rect x={26 + i * bw} y={92 - h} width={bw - 2} height={h} fill={BLUE} />
          <rect x={26 + i * bw} y={92 - h - b[i]} width={bw - 2} height={b[i]} fill={ORANGE} />
        </g>
      ))}
    </Frame>
  );
}

// ---- Glyph-sized variants for the analysis-type cards (no axes, bold strokes)

const GW = 64;
const GH = 32;

function Glyph({children}: {children: React.ReactNode}) {
  return (
    <svg viewBox={`0 0 ${GW} ${GH}`} width="56" height="28" className="shrink-0" aria-hidden="true">
      <line x1="2" y1={GH - 2} x2={GW - 2} y2={GH - 2} stroke="currentColor" strokeOpacity="0.3" />
      {children}
    </svg>
  );
}

function curveGlyph(shift: number, width: number, height: number) {
  const pts: string[] = [];
  for (let x = 0; x <= 60; x += 3) {
    const y = height * Math.exp(-Math.pow((x - 30 - shift) / width, 2));
    pts.push(`${2 + x},${GH - 3 - y}`);
  }
  return pts.join(' ');
}

function StratifiedGlyph() {
  return (
    <Glyph>
      <polyline points={curveGlyph(-7, 9, 24)} fill="none" stroke={BLUE} strokeWidth="2.5" strokeLinecap="round" />
      <polyline points={curveGlyph(8, 11, 20)} fill="none" stroke={ORANGE} strokeWidth="2.5" strokeLinecap="round" />
    </Glyph>
  );
}

function ScatterGlyph() {
  const pts: [number, number][] = [
    [6, 24], [11, 21], [14, 25], [19, 18], [23, 20], [27, 14], [31, 17], [35, 12], [40, 14], [44, 9], [49, 11], [54, 6], [58, 8]
  ];
  return (
    <Glyph>
      {pts.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r="2" fill={BLUE} fillOpacity="0.85" />
      ))}
      <line x1="5" y1="25" x2="59" y2="6" stroke="#c0392b" strokeWidth="2" strokeLinecap="round" />
    </Glyph>
  );
}

function CrosstabGlyph() {
  const cols = [
    [14, 7, 4],
    [6, 11, 9],
    [4, 7, 14]
  ];
  const bw = 14;
  return (
    <Glyph>
      {cols.map((stack, i) => {
        let y = GH - 2;
        return stack.map((h, j) => {
          y -= h;
          return <rect key={`${i}-${j}`} x={8 + i * (bw + 6)} y={y} width={bw} height={h} fill={[BLUE, ORANGE, GREEN][j]} />;
        });
      })}
    </Glyph>
  );
}

function CompareGlyph() {
  return (
    <Glyph>
      <polyline points={curveGlyph(-9, 8, 22)} fill="none" stroke={BLUE} strokeWidth="2.5" strokeLinecap="round" />
      <polyline points={curveGlyph(1, 7, 25)} fill="none" stroke={ORANGE} strokeWidth="2.5" strokeLinecap="round" />
      <polyline points={curveGlyph(11, 9, 18)} fill="none" stroke={GREEN} strokeWidth="2.5" strokeLinecap="round" />
    </Glyph>
  );
}

export function KindGlyph({kind}: {kind: Kind}) {
  switch (kind) {
    case 'stratified':
      return <StratifiedGlyph />;
    case 'correlation':
      return <ScatterGlyph />;
    case 'crosstab':
      return <CrosstabGlyph />;
    case 'compare':
      return <CompareGlyph />;
    default:
      return null;
  }
}

export default function MiniChart({kind}: {kind: Kind}) {
  switch (kind) {
    case 'stratified':
      return <Stratified />;
    case 'correlation':
      return <Scatter />;
    case 'crosstab':
      return <Crosstab />;
    case 'compare':
      return (
        <div className="grid grid-cols-2 gap-1">
          <Compare />
          <Pooled />
        </div>
      );
    default:
      return null;
  }
}
