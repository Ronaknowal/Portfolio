import { cloneElement, isValidElement, useId, useState } from 'react';
import { trailNames, trailPoints, cornerPoints, fifthCorner, dbscan, sortedRoster, irisReport, compareReports, intervalFixture, transformRows, sameNeighborGraph } from '../../data/dbscan-models';
import { irisFeatures, irisSpecies, irisRows } from '../../data/dbscan-iris-data';
import './dbscan-labs.css';

const number = (value, digits = 4) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (value === Infinity) return '∞';
  if (Number.isInteger(value)) return String(value);
  const fixed = value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
  return fixed === '-0' ? '0' : fixed;
};
const percent = value => `${number(100 * value, 1)}%`;
const componentColors = ['#e7b94a', '#8eb9a5', '#91aecf', '#da9c86', '#b8a0cd', '#d0c7ae', '#c7d08a', '#9fb7d8', '#d6a3b8', '#a8c8b8'];
const nameOf = (index, names) => names ? names[index] : `row ${index}`;

export function Investigation({ title, question, children, onReset }) {
  const id = useId();
  return <section className="db-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="db-question">{question}</p>}
    {children}
  </section>;
}
/** One or more prediction fields, a commit button and feedback. `fields` is a
 * list of {key, label, options, answer}; `values` holds the learner's choices. */
export function Prediction({ prompt, fields, values, onChange, committed, stale, onCommit, commitLabel = 'Commit prediction and apply', explanation }) {
  const ready = fields.every(field => values[field.key]);
  const matches = fields.map(field => ({ ...field, chosen: values[field.key], neutral: (field.neutral ?? []).includes(values[field.key]), match: values[field.key] === String(field.answer), answerLabel: field.options.find(([key]) => key === String(field.answer))?.[1] ?? String(field.answer) }));
  const allMatch = matches.every(field => field.match || field.neutral);
  const anyNeutral = matches.some(field => field.neutral);
  return <div className="db-prediction">
    <p><strong>Predict first:</strong> {prompt}</p>
    <div className="db-prediction-row">
      {fields.map(field => <label key={field.key}>{field.label}<select value={values[field.key] ?? ''} onChange={event => onChange(field.key, event.target.value)}>
        <option value="">Choose</option>{field.options.map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>)}
      <button type="button" disabled={!ready || (committed && !stale)} onClick={onCommit}>{commitLabel}</button>
    </div>
    {stale && <p className="db-feedback is-stale" role="status">The inputs changed after your last comparison. Record a new prediction for the current settings.</p>}
    {committed && !stale && <p className={`db-feedback ${allMatch ? '' : 'is-miss'}`} role="status">
      {allMatch ? (anyNeutral ? 'Here is the answer you chose to inspect.' : 'Your prediction matches.') : 'Not this time.'} {matches.map(field => `${field.label}: ${field.answerLabel}${field.match || field.neutral ? '' : ` (you chose ${field.options.find(([key]) => key === field.chosen)?.[1] ?? field.chosen})`}`).join('; ')}. {explanation}
    </p>}
  </div>;
}
export function Field({ label, children, value }) {
  const id = useId();
  const control = isValidElement(children) && typeof children.type === 'string' && ['input', 'select'].includes(children.type) ? cloneElement(children, { id }) : children;
  return <label className="db-field" htmlFor={id}><span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>{control}</label>;
}
export function Table({ caption, headings, rows, highlight = () => false }) {
  return <div className="db-table-scroll" role="region" aria-label={caption} tabIndex={0}>
    <table><caption>{caption}</caption><thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index} className={highlight(index) ? 'is-selected' : undefined}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}
/** Marker for a row: filled circle core, hollow circle border, cross noise, plain circle when types are hidden. */
export function Glyph({ x, y, type, color }) {
  if (type === 'core') return <circle className="db-core" cx={x} cy={y} r="5" style={color ? { fill: color } : undefined} />;
  if (type === 'border') return <circle className="db-border" cx={x} cy={y} r="4.5" style={color ? { stroke: color } : undefined} />;
  if (type === 'noise') return <path className="db-noise" d={`M${x - 4.5},${y - 4.5}l9,9 M${x - 4.5},${y + 4.5}l9,-9`} fill="none" />;
  return <circle className="db-plain" cx={x} cy={y} r="4" />;
}
export function GlyphLegend({ components = false }) {
  return <div className="db-legend-row">
    <span><svg viewBox="-8 -8 16 16"><Glyph x={0} y={0} type="core" /></svg> core</span>
    <span><svg viewBox="-8 -8 16 16"><Glyph x={0} y={0} type="border" /></svg> border</span>
    <span><svg viewBox="-8 -8 16 16"><Glyph x={0} y={0} type="noise" /></svg> noise</span>
    {components && <span>color = core component; solid line = core edge; dotted line = border attachment</span>}
  </div>;
}
/** Equal-unit square plot around the supplied points. */
export function SquarePlot({ points, title, describe, xLabel = 'x (meters)', yLabel = 'y (meters)', pad = 0.16, children }) {
  const minima = [0, 1].map(axis => Math.min(...points.map(p => p[axis])));
  const maxima = [0, 1].map(axis => Math.max(...points.map(p => p[axis])));
  const span = Math.max(1, ...[0, 1].map(axis => maxima[axis] - minima[axis]));
  const centers = [0, 1].map(axis => (minima[axis] + maxima[axis]) / 2);
  const extent = span * (1 + 2 * pad);
  const domain = centers.map(center => center - extent / 2);
  const project = point => [32 + 256 * (point[0] - domain[0]) / extent, 288 - 256 * (point[1] - domain[1]) / extent];
  const scale = 256 / extent;
  const tick = value => String(Number(value.toPrecision(3)));
  return <figure className="db-plot">
    {title && <figcaption>{title}</figcaption>}
    <div className="db-square">
      <svg viewBox="0 0 320 320" role="img" aria-label={describe ?? title}>
        {[32, 160, 288].map(position => <path key={position} className="db-grid" d={`M32,${position}H288 M${position},32V288`} />)}
        {children(project, scale, domain, extent)}
      </svg>
      {[0, 0.5, 1].map(fraction => <span key={`x${fraction}`} className="db-tick is-x" style={{ left: `${10 + 80 * fraction}%` }}>{tick(domain[0] + fraction * extent)}</span>)}
      {[0, 0.5, 1].map(fraction => <span key={`y${fraction}`} className="db-tick is-y" style={{ top: `${90 - 80 * fraction}%` }}>{tick(domain[1] + fraction * extent)}</span>)}
    </div>
    <div className="db-axis-caption"><span>horizontal: {xLabel}</span><span>vertical: {yLabel}</span></div>
  </figure>;
}
/** Draw the neighbourhood graph: core edges, border attachments and glyphs. */
export function GraphLayer({ points, fit, project, scale, selected = null, names = null, colored = true, showEdges = true, showLabels = true }) {
  const color = index => colored && fit.labels[index] >= 0 ? componentColors[fit.labels[index] % componentColors.length] : undefined;
  const drawn = new Set();
  return <>
    {selected !== null && (() => { const [cx, cy] = project(points[selected]); return <circle className="db-disk" cx={cx} cy={cy} r={fit.eps * scale} />; })()}
    {showEdges && points.map((_, i) => fit.neighbors[i].map(j => {
      if (j <= i) return null;
      const key = `${i}-${j}`;
      if (drawn.has(key)) return null; drawn.add(key);
      const [x1, y1] = project(points[i]), [x2, y2] = project(points[j]);
      if (fit.core[i] && fit.core[j]) return <line key={key} className="db-edge" x1={x1} y1={y1} x2={x2} y2={y2} style={fit.labels[i] >= 0 ? { stroke: componentColors[fit.labels[i] % componentColors.length] } : undefined} />;
      if (fit.core[i] || fit.core[j]) return <line key={key} className="db-attach" x1={x1} y1={y1} x2={x2} y2={y2} />;
      return null;
    }))}
    {points.map((point, i) => { const [x, y] = project(point); return <g key={i}><Glyph x={x} y={y} type={fit.types[i]} color={color(i)} />{showLabels && <text x={x} y={y + [-9, 18, -21, 30][i % 4]} textAnchor="middle">{nameOf(i, names)}</text>}</g>; })}
  </>;
}
const lattice = (value, step, low, high) => Math.min(high, Math.max(low, Math.round(value / step) * step));

/* ---------------- L1: trail workbench ---------------- */
const presets = {
  trail: { label: 'Ten-row trail A–J', points: trailPoints.map(p => [...p]), names: [...trailNames], eps: 1, m: 4 },
  duplicates: { label: 'Four identical rows at x = 2', points: [[2, 0], [2, 0], [2, 0], [2, 0]], names: ['P', 'Q', 'R', 'S'], eps: 0.125, m: 4 },
};
export function DbscanTrailLab() {
  const [pending, setPending] = useState({ preset: 'trail', points: presets.trail.points, eps: 1, m: 4, order: 'original', start: '' });
  const [applied, setApplied] = useState(null);
  const [selected, setSelected] = useState(8);
  const [prediction, setPrediction] = useState({});
  const names = presets[pending.preset].names;
  const orderFor = state => {
    const n = state.points.length;
    let order = Array.from({ length: n }, (_, i) => i);
    if (state.order === 'reversed') order.reverse();
    if (state.start !== '') { const first = Number(state.start); order = [first, ...order.filter(i => i !== first)]; }
    return order;
  };
  const currentFit = dbscan(pending.points, pending.eps, pending.m, { order: orderFor(pending) });
  const key = JSON.stringify(pending);
  const stale = applied !== null && applied.key !== key;
  const shown = applied && !stale ? applied.fit : null;
  const update = patch => setPending({ ...pending, ...patch });
  const setPoint = (axis, raw) => {
    if (String(raw).trim() === '' || String(raw).trim() === '-') return;
    const value = Number(raw);
    if (!Number.isFinite(value)) return;
    const bounded = axis === 0 ? lattice(value, 0.125, -3, 5) : lattice(value, 0.125, -2, 2);
    update({ points: pending.points.map((point, i) => i === selected ? point.map((old, which) => which === axis ? bounded : old) : point) });
  };
  const loadPreset = presetKey => { setPending({ preset: presetKey, points: presets[presetKey].points.map(p => [...p]), eps: presets[presetKey].eps, m: presets[presetKey].m, order: 'original', start: '' }); setSelected(presetKey === 'trail' ? 8 : 0); setApplied(null); setPrediction({}); };
  const componentOptions = Array.from({ length: pending.points.length + 1 }, (_, k) => [String(k), `${k} component${k === 1 ? '' : 's'}`]);
  return <Investigation title="Can one row join two groups together?" question="Edit any coordinate, the radius or the count, choose a visiting order, then commit what you expect for the number of core components and the selected row's type before the result is revealed." onReset={() => loadPreset('trail')}>
    <div className="db-controls">
      <Field label="Point configuration"><select value={pending.preset} onChange={event => loadPreset(event.target.value)}>{Object.entries(presets).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}</select></Field>
      <Field label="Selected row"><select value={selected} onChange={event => setSelected(Number(event.target.value))}>{pending.points.map((p, i) => <option key={i} value={i}>{nameOf(i, names)} ({number(p[0])}, {number(p[1])})</option>)}</select></Field>
      <Field label={`${nameOf(selected, names)} x, meters (−3..5, step 0.125)`}><input type="number" step="0.125" min="-3" max="5" value={pending.points[selected][0]} onChange={event => setPoint(0, event.target.value)} /></Field>
      <Field label={`${nameOf(selected, names)} y, meters (−2..2)`}><input type="number" step="0.125" min="-2" max="2" value={pending.points[selected][1]} onChange={event => setPoint(1, event.target.value)} /></Field>
      <Field label="Radius ε, meters" value={number(pending.eps)}><input type="range" min="0.125" max="3" step="0.125" value={pending.eps} onChange={event => update({ eps: Number(event.target.value) })} /></Field>
      <Field label="Count m, rows including self" value={String(pending.m)}><input type="range" min="1" max="10" step="1" value={pending.m} onChange={event => update({ m: Number(event.target.value) })} /></Field>
      <Field label="Visiting order"><select value={pending.order} onChange={event => update({ order: event.target.value })}><option value="original">Original row order</option><option value="reversed">Reversed row order</option></select></Field>
      <Field label="Row visited first (optional)"><select value={pending.start} onChange={event => update({ start: event.target.value })}><option value="">Follow the order above</option>{pending.points.map((_, i) => <option key={i} value={i}>{nameOf(i, names)}</option>)}</select></Field>
    </div>
    <Prediction prompt={`At ε = ${number(pending.eps)} and m = ${pending.m}, after applying the edits above:`} fields={[
      { key: 'components', label: 'Number of core components', options: componentOptions, answer: currentFit.clusters },
      { key: 'type', label: `Type of ${nameOf(selected, names)}`, options: [['core', 'Core'], ['border', 'Border'], ['noise', 'Noise']], answer: currentFit.types[selected] },
    ]} values={prediction} onChange={(field, value) => setPrediction({ ...prediction, [field]: value })} committed={applied !== null} stale={stale} onCommit={() => setApplied({ key, fit: currentFit })} explanation={`${nameOf(selected, names)} has ${currentFit.counts[selected]} neighbour${currentFit.counts[selected] === 1 ? '' : 's'} including itself${currentFit.types[selected] === 'border' ? `, and its core neighbours belong to component${currentFit.eligible[selected].length > 1 ? 's' : ''} ${currentFit.eligible[selected].join(' and ')}` : ''}.`} />
    {shown ? <>
      <p className="db-readout" aria-live="polite"><strong>{shown.clusters} core component{shown.clusters === 1 ? '' : 's'}</strong>: {shown.components.map(group => `{${group.map(i => nameOf(i, names)).join(', ')}}`).join(' and ') || 'none'}. Core {shown.coreIds.length}, border {shown.borderIds.length}, noise {shown.noiseIds.length}. Visit order: {shown.visit.map(i => nameOf(i, names)).join(' ')}.</p>
      <SquarePlot points={pending.points} title={`Neighbourhood graph at ε = ${number(pending.eps)}, m = ${pending.m}; the dashed disk is ${nameOf(selected, names)}'s closed neighbourhood`} describe={`${pending.points.length} rows with core, border and noise glyphs, core edges within components and dotted border attachments. Exact values are in the tables.`}>
        {(project, scale) => <GraphLayer points={pending.points} fit={shown} project={project} scale={scale} selected={selected} names={names} />}
      </SquarePlot>
      <GlyphLegend components />
      <Table caption={`${nameOf(selected, names)}'s sorted distances including itself; rows within ε are neighbours`} headings={['row', 'distance', 'within ε', 'type']} rows={sortedRoster(pending.points, selected).map(entry => [nameOf(entry.index, names), number(entry.distance), entry.distance <= pending.eps ? 'yes' : 'no', shown.types[entry.index]])} highlight={index => sortedRoster(pending.points, selected)[index].index === selected} />
      <Table caption="Every row: neighbour count, type, component label and eligible components for borders" headings={['row', 'coordinates', 'neighbours incl. self', 'type', 'label', 'eligible components']} rows={pending.points.map((p, i) => [nameOf(i, names), `(${number(p[0])}, ${number(p[1])})`, shown.counts[i], shown.types[i], shown.labels[i], shown.types[i] === 'border' ? shown.eligible[i].join(', ') : shown.types[i] === 'core' ? String(shown.labels[i]) : '—'])} highlight={index => index === selected} />
    </> : <div className="db-hidden">Commit a prediction to reveal the neighbourhood graph, rosters and types for the current settings.</div>}
    <p className="db-caption">Suggested investigations: raise ε from 0.75 to 1 to 1.25 and watch I go noise → border → core while the components go two → two → one. Reverse the order at ε = 1: every type and both core components are unchanged, only I's attachment moves. Move I to (0, 0.125): its distance to D and E becomes √(1 + 0.125²) {'>'} 1, so order stops mattering. Load the four identical rows: at ε = 0.125 all four are core because a neighbourhood is a set of row identities. Set m = 1 on the trail: every row is core and there is no noise.</p>
  </Investigation>;
}

/* ---------------- L2: units versus metric ---------------- */
const factorOptions = [0.125, 0.25, 0.5, 1, 2, 4, 10, 100];
export function DbscanMetricLab() {
  const initial = { rows: cornerPoints.map(p => [...p]), fifth: false, xFactor: 1, yFactor: 1, baseEps: 1, eps: 1, m: 2 };
  const [pending, setPending] = useState(initial);
  const [applied, setApplied] = useState(null);
  const [prediction, setPrediction] = useState({});
  const [pair, setPair] = useState([1, 4]);
  const rows = pending.fifth ? [...pending.rows, [...fifthCorner]] : pending.rows;
  const names = ['P', 'Q', 'R', 'S', 'T'].slice(0, rows.length);
  const before = dbscan(rows, pending.baseEps, pending.m);
  const after = (() => { const t = transformRows(rows, [pending.xFactor, pending.yFactor], pending.eps, 1); return dbscan(t.points, pending.eps, pending.m); })();
  const same = sameNeighborGraph(before, after);
  const key = JSON.stringify(pending);
  const stale = applied !== null && applied.key !== key;
  const shown = applied && !stale;
  const update = patch => setPending({ ...pending, ...patch });
  const pairIndices = pair.map(i => Math.min(i, rows.length - 1));
  const pairBefore = Math.hypot(rows[pairIndices[0]][0] - rows[pairIndices[1]][0], rows[pairIndices[0]][1] - rows[pairIndices[1]][1]);
  const transformed = rows.map(p => [p[0] * pending.xFactor, p[1] * pending.yFactor]);
  const pairAfter = Math.hypot(transformed[pairIndices[0]][0] - transformed[pairIndices[1]][0], transformed[pairIndices[0]][1] - transformed[pairIndices[1]][1]);
  return <Investigation title="Change a unit, or change the question?" question="Multiply one or both coordinates by a factor and keep or change the radius. Will every neighbourhood decision stay the same? Commit yes or no and name a pair you expect to decide it." onReset={() => { setPending(initial); setApplied(null); setPrediction({}); setPair([1, 4]); }}>
    <div className="db-controls">
      <Field label="Which corners"><select value={pending.fifth ? 'five' : 'four'} onChange={event => update({ fifth: event.target.value === 'five' })}><option value="four">Four corners (0,0) (1,0) (0,2) (1,2)</option><option value="five">Four corners plus T = (3, 0)</option></select></Field>
      <Field label="Multiply x by"><select value={pending.xFactor} onChange={event => update({ xFactor: Number(event.target.value) })}>{factorOptions.map(f => <option key={f} value={f}>× {f}</option>)}</select></Field>
      <Field label="Multiply y by"><select value={pending.yFactor} onChange={event => update({ yFactor: Number(event.target.value) })}>{factorOptions.map(f => <option key={f} value={f}>× {f}</option>)}</select></Field>
      <Field label="Radius ε on the original rows (0.125–20)"><input type="number" min="0.125" max="20" step="any" value={pending.baseEps} onChange={event => { const v = Number(event.target.value); if (event.target.value.trim() !== '' && v >= 0.125 && v <= 20) update({ baseEps: v }); }} /></Field>
      <Field label="Radius ε after the change (0.125–2000)"><input type="number" min="0.125" max="2000" step="any" value={pending.eps} onChange={event => { const v = Number(event.target.value); if (event.target.value.trim() !== '' && v >= 0.125 && v <= 2000) update({ eps: v }); }} /></Field>
      <Field label="Count m" value={String(pending.m)}><input type="range" min="1" max="5" step="1" value={pending.m} onChange={event => update({ m: Number(event.target.value) })} /></Field>
      <Field label="Cited pair, first row"><select value={pairIndices[0]} onChange={event => setPair([Number(event.target.value), pairIndices[1]])}>{names.map((n, i) => <option key={i} value={i}>{n}</option>)}</select></Field>
      <Field label="Cited pair, second row"><select value={pairIndices[1]} onChange={event => setPair([pairIndices[0], Number(event.target.value)])}>{names.map((n, i) => <option key={i} value={i}>{n}</option>)}</select></Field>
    </div>
    <div className="db-buttons">
      <button type="button" onClick={() => update({ xFactor: 100, yFactor: 100, eps: pending.baseEps * 100 })}>Convert both coordinates and ε by 100 (a unit change)</button>
      <button type="button" onClick={() => update({ xFactor: 1, yFactor: 100, eps: pending.baseEps * 100 })}>Convert only y and ε by 100 (the faulty fix)</button>
      <button type="button" onClick={() => update({ xFactor: 1, yFactor: 0.5, eps: pending.baseEps })}>Halve y, keep ε (a metric change)</button>
      <span>The buttons only set the controls; nothing is revealed until you commit.</span>
    </div>
    <Prediction prompt={`With x × ${pending.xFactor}, y × ${pending.yFactor} and ε = ${number(pending.eps)} afterwards (ε = ${number(pending.baseEps)} on the original rows), will the radius-neighbour graph be identical?`} fields={[{ key: 'same', label: 'Same neighbourhoods?', options: [['yes', 'Yes, every decision is preserved'], ['no', 'No, at least one pair changes']], answer: same.same ? 'yes' : 'no' }]} values={prediction} onChange={(field, value) => setPrediction({ ...prediction, [field]: value })} committed={applied !== null} stale={stale} onCommit={() => setApplied({ key })} explanation={same.same ? `Every row keeps exactly the same neighbour set. Your cited pair ${names[pairIndices[0]]}–${names[pairIndices[1]]} is ${number(pairBefore)} apart before and ${number(pairAfter)} after, against radii ${number(pending.baseEps)} and ${number(pending.eps)}.` : `Rows whose neighbour sets changed: ${same.differing.map(i => names[i]).join(', ')}. Your cited pair ${names[pairIndices[0]]}–${names[pairIndices[1]]} is ${number(pairBefore)} apart before (radius ${number(pending.baseEps)}) and ${number(pairAfter)} after (radius ${number(pending.eps)}).`} />
    {shown ? <>
      <p className="db-readout" aria-live="polite">Before: {before.clusters} component{before.clusters === 1 ? '' : 's'}, types {before.types.map((t, i) => `${names[i]} ${t}`).join(', ')}. After: {after.clusters} component{after.clusters === 1 ? '' : 's'}, types {after.types.map((t, i) => `${names[i]} ${t}`).join(', ')}.</p>
      <div className="db-figure-pair">
        <SquarePlot points={rows} title={`Original rows, ε = ${number(pending.baseEps)}`} xLabel="x (original units)" yLabel="y (original units)">{(project, scale) => <GraphLayer points={rows} fit={before} project={project} scale={scale} names={names} />}</SquarePlot>
        <SquarePlot points={transformed} title={`Transformed rows, ε = ${number(pending.eps)}`} xLabel={`x × ${pending.xFactor}`} yLabel={`y × ${pending.yFactor}`}>{(project, scale) => <GraphLayer points={transformed} fit={after} project={project} scale={scale} names={names} />}</SquarePlot>
      </div>
      <GlyphLegend components />
      <Table caption="Pairwise distances before → after, with whether each pair is within the radius" headings={['pair', 'before', 'within ε before', 'after', 'within ε after']} rows={rows.flatMap((_, i) => rows.slice(i + 1).map((__, k) => { const j = i + 1 + k; return [`${names[i]}–${names[j]}`, number(before.matrix[i][j]), before.matrix[i][j] <= pending.baseEps ? 'yes' : 'no', number(after.matrix[i][j]), after.matrix[i][j] <= pending.eps ? 'yes' : 'no']; }))} highlight={index => { const pairs = rows.flatMap((_, i) => rows.slice(i + 1).map((__, k) => [i, i + 1 + k])); return pairs[index][0] === Math.min(...pairIndices) && pairs[index][1] === Math.max(...pairIndices); }} />
    </> : <div className="db-hidden">Commit a prediction to reveal both neighbourhood graphs and the distance table.</div>}
    <p className="db-caption">Multiplying both coordinates and ε by the same positive factor preserves every comparison: a unit change. Multiplying only y by 100 while multiplying ε by 100 is a change of metric dressed as a unit change. With the four corners alone it happens to leave both horizontal pairs unchanged, which proves nothing; with T = (3, 0) added, T's horizontal distance 2 to Q is suddenly far below the new radius 100 and T becomes core. Halving y at fixed ε = 1 merges the two pairs into one component. Standardizing each axis independently is one particular metric choice, not a neutral default.</p>
  </Investigation>;
}

/* ---------------- L3: Iris coverage ---------------- */
const coverageBands = [['lt25', 'Below 25% of the 150 flowers'], ['25to50', '25% to 50%'], ['50to75', '50% to 75%'], ['gt75', 'Above 75%']];
const bandOf = coverage => coverage < 0.25 ? 'lt25' : coverage < 0.5 ? '25to50' : coverage < 0.75 ? '50to75' : 'gt75';
const speciesNames = ['setosa', 'versicolor', 'virginica'];
export function DbscanIrisLab() {
  const initial = { eps: 0.5, m: 5, representation: 'standardized' };
  const [pending, setPending] = useState(initial);
  const [applied, setApplied] = useState(null);
  const [prediction, setPrediction] = useState({});
  const [axes, setAxes] = useState([2, 3]);
  const [showSpecies, setShowSpecies] = useState(false);
  const [row, setRow] = useState(41);
  const [saved, setSaved] = useState(null);
  const report = irisReport(pending.eps, pending.m, pending.representation);
  const key = JSON.stringify(pending);
  const stale = applied !== null && applied.key !== key;
  const shown = applied && !stale ? applied.report : null;
  const update = patch => setPending({ ...pending, ...patch });
  const groupOptions = [['0', 'No group survives'], ['1', 'One group'], ['2', 'Two groups'], ['3', 'Three groups'], ['4', 'Four or more groups']];
  const groupAnswer = Math.min(report.clusters, 4);
  const projected = shown ? shown.space.map(p => [p[axes[0]], p[axes[1]]]) : null;
  const rowReport = shown ? { space: shown.space[row], type: shown.fit.types[row], label: shown.fit.labels[row], neighbours: sortedRoster(shown.space, row).slice(0, pending.m + 2) } : null;
  return <Investigation title="A better score for fewer flowers?" question="Choose a radius and count for all four Iris measurements, then commit how many groups and how much of the collection you expect to survive before the result is revealed. Species stay hidden until you have recorded a parameter decision." onReset={() => { setPending(initial); setApplied(null); setPrediction({}); setAxes([2, 3]); setShowSpecies(false); setRow(41); setSaved(null); }}>
    <div className="db-controls">
      <Field label={`Radius ε in ${pending.representation === 'raw' ? 'centimetres' : 'standardized units'}`} value={number(pending.eps)}><input type="range" min="0.1" max="2" step="0.05" value={pending.eps} onChange={event => update({ eps: Number(event.target.value) })} /></Field>
      <Field label="Same radius, typed (0.1 to 2)"><input type="number" min="0.1" max="2" step="0.01" value={pending.eps} onChange={event => { const v = Number(event.target.value); if (event.target.value.trim() !== '' && v >= 0.1 && v <= 2) update({ eps: v }); }} /></Field>
      <Field label="Count m, rows including self" value={String(pending.m)}><input type="range" min="1" max="20" step="1" value={pending.m} onChange={event => update({ m: Number(event.target.value) })} /></Field>
      <Field label="Feature scaling"><select value={pending.representation} onChange={event => update({ representation: event.target.value })}><option value="standardized">Four standardized features</option><option value="raw">Four original centimetre features</option></select></Field>
    </div>
    <Prediction prompt={`With ε = ${number(pending.eps)}, m = ${pending.m} on ${pending.representation === 'raw' ? 'raw centimetre' : 'standardized'} features:`} fields={[
      { key: 'groups', label: 'Returned groups', options: groupOptions, answer: groupAnswer },
      { key: 'coverage', label: 'Coverage (assigned rows / 150)', options: coverageBands, answer: bandOf(report.coverage) },
    ]} values={prediction} onChange={(field, value) => setPrediction({ ...prediction, [field]: value })} committed={applied !== null} stale={stale} onCommit={() => { setApplied({ key, report }); setShowSpecies(false); }} commitLabel="Commit and reveal the report" explanation={`${report.clusters} group${report.clusters === 1 ? '' : 's'} of sizes ${report.sizes.join(', ') || '—'}; ${report.assignedIds.length} of 150 assigned (${percent(report.coverage)}), ${report.noiseCount} noise.`} />
    {shown ? <>
      <Table caption="Whole-collection report for the committed setting" headings={['quantity', 'value']} rows={[
        ['groups (sizes)', `${shown.clusters} (${shown.sizes.join(', ') || '—'})`], ['core / border / noise', `${shown.coreCount} / ${shown.borderCount} / ${shown.noiseCount}`], ['coverage', `${shown.assignedIds.length} / 150 = ${percent(shown.coverage)}`],
        ['silhouette, assigned rows only', shown.silhouetteAssigned === null ? `undefined: needs 2 ≤ groups < assigned rows (groups ${shown.clusters}, assigned ${shown.assignedIds.length})` : `${number(shown.silhouetteAssigned, 3)} on ${shown.assignedIds.length} rows`],
        ['ARI against species, all 150 rows, −1 as one label', showSpecies ? number(shown.ariAllRows, 3) : 'hidden until species are revealed'], ['ARI against species, assigned rows only', showSpecies ? (shown.ariAssigned === null ? 'undefined' : `${number(shown.ariAssigned, 3)} on ${shown.assignedIds.length} rows`) : 'hidden until species are revealed'],
      ]} />
      <div className="db-buttons">
        <button type="button" disabled={showSpecies} onClick={() => setShowSpecies(true)}>Reveal species (a retrospective reference, not an input)</button>
        <button type="button" onClick={() => setSaved(shown)}>Save this report as snapshot A</button>
        <span>{saved ? `Snapshot A: ε ${number(saved.eps)}, m ${saved.minimum}, ${saved.representation}` : 'No snapshot saved yet.'}</span>
      </div>
      <div className="db-controls">
        <Field label="Projection: horizontal feature"><select value={axes[0]} onChange={event => setAxes([Number(event.target.value), axes[1]])}>{irisFeatures.map((f, i) => <option key={f} value={i}>{f}</option>)}</select></Field>
        <Field label="Projection: vertical feature"><select value={axes[1]} onChange={event => setAxes([axes[0], Number(event.target.value)])}>{irisFeatures.map((f, i) => <option key={f} value={i}>{f}</option>)}</select></Field>
        <Field label="Row report"><select value={row} onChange={event => setRow(Number(event.target.value))}>{irisRows.map((_, i) => <option key={i} value={i}>row {i}{shown.fit.types[i] === 'noise' ? ' (noise)' : ''}</option>)}</select></Field>
      </div>
      <SquarePlot points={projected} title={`Projection of the four-feature fit onto ${irisFeatures[axes[0]]} and ${irisFeatures[axes[1]]} (${pending.representation === 'raw' ? 'cm' : 'standardized units'})`} xLabel={irisFeatures[axes[0]]} yLabel={irisFeatures[axes[1]]} pad={0.08} describe={`150 flowers projected onto two of the four features; glyphs give core, border and noise from the four-feature fit${showSpecies ? ', with species drawn as shapes' : ''}.`}>
        {(project) => projected.map((p, i) => { const [x, y] = project(p); const color = shown.fit.labels[i] >= 0 ? componentColors[shown.fit.labels[i] % componentColors.length] : undefined;
          if (showSpecies) { const s = irisSpecies[i]; const fill = shown.fit.labels[i] >= 0 ? color : '#da9c86'; if (s === 0) return <circle key={i} cx={x} cy={y} r="3.2" fill={fill} fillOpacity=".85" />; if (s === 1) return <rect key={i} x={x - 3} y={y - 3} width="6" height="6" fill={fill} fillOpacity=".85" />; return <path key={i} d={`M${x},${y - 4}l3.6,6.5h-7.2z`} fill={fill} fillOpacity=".85" />; }
          return <g key={i}><Glyph x={x} y={y} type={shown.fit.types[i]} color={color} /></g>; })}
      </SquarePlot>
      {showSpecies ? <p className="db-legend">Species shapes: circle setosa, square versicolor, triangle virginica. Fill color = density group; red = noise (−1). The species were never given to the fit.</p> : <GlyphLegend components />}
      <p className="db-readout">Row {row}: measurements {irisRows[row].map((v, j) => `${irisFeatures[j]} ${v}`).join(', ')}; in the fitted space {rowReport.space.map(v => number(v, 3)).join(', ')}; type <strong>{rowReport.type}</strong>, label {rowReport.label}. Nearest full-space distances including itself: {rowReport.neighbours.map(e => `row ${e.index} ${number(e.distance, 3)}`).join('; ')}.{showSpecies ? ` Species: ${speciesNames[irisSpecies[row]]}.` : ''}</p>
      {saved && saved !== shown && (() => { const c = compareReports(saved, shown); return <Table caption="Snapshot A versus the current report on their common retained rows" headings={['quantity', 'snapshot A', 'current', 'common rows']} rows={[
        ['setting', `ε ${number(saved.eps)}, m ${saved.minimum}, ${saved.representation}`, `ε ${number(shown.eps)}, m ${shown.minimum}, ${shown.representation}`, `${c.common.length} rows assigned by both`],
        ['assigned rows', saved.assignedIds.length, shown.assignedIds.length, `${c.onlyFirst} only A, ${c.onlySecond} only current`],
        ['groups', saved.clusters, shown.clusters, c.ariCommon === null ? 'agreement undefined' : `partition agreement (ARI) on common rows ${number(c.ariCommon, 3)}`],
        ['ARI vs species, all rows', showSpecies ? number(saved.ariAllRows, 3) : 'hidden', showSpecies ? number(shown.ariAllRows, 3) : 'hidden', '—'],
        ['ARI vs species on the common rows only', showSpecies ? (c.ariSpeciesFirstCommon === null ? 'undefined' : number(c.ariSpeciesFirstCommon, 3)) : 'hidden', showSpecies ? (c.ariSpeciesSecondCommon === null ? 'undefined' : number(c.ariSpeciesSecondCommon, 3)) : 'hidden', `${c.common.length} rows, same population for both`],
      ]} />; })()}
    </> : <div className="db-hidden">Commit a prediction to reveal the whole-collection report, the projection and the row report.</div>}
    <p className="db-caption">Two settings worth committing to: standardized ε = 0.5 with m = 5 keeps 116 of 150 flowers in two groups, and with m = 10 keeps 61 in three groups whose assigned-row agreement with species is perfect while the all-row agreement falls. The score improved by describing fewer flowers. The plot is a projection of a four-feature fit: two points close on screen can be far apart in the two omitted features, and changing the projection never refits the model. Setting m = 1 removes noise by definition, and ε = 0.3 keeps only 30 flowers.</p>
  </Investigation>;
}

/** A magnified one-dimensional strip so rows 0.125 m apart are individually countable. */
export function DenseStrip({ points, fit, names, from, to, caption }) {
  const shown = points.map((p, i) => [p[0], i]).filter(([x]) => x >= from && x <= to);
  const px = x => 24 + (x - from) * (512 / (to - from));
  return <div className="db-scroll"><svg className="db-svg-wide" viewBox="0 0 560 80" role="img" aria-label={`${caption}: ${shown.map(([x, i]) => `${names[i]} at ${number(x)} is ${fit.types[i]}`).join('; ')}.`}>
    <text x="8" y="14">{caption}</text>
    <line x1={px(from)} x2={px(to)} y1="46" y2="46" className="db-grid" />
    {Array.from({ length: Math.floor((to - from) / 0.25) + 1 }, (_, k) => from + 0.25 * k).map(v => <g key={v}><line x1={px(v)} x2={px(v)} y1="42" y2="50" className="db-grid" /><text x={px(v)} y="72" textAnchor="middle">{number(v)}</text></g>)}
    {shown.map(([x, i]) => <g key={i}><Glyph x={px(x)} y={46} type={fit.types[i]} color={fit.labels[i] >= 0 ? componentColors[fit.labels[i] % componentColors.length] : undefined} /><text x={px(x)} y={i % 2 ? 62 : 32} textAnchor="middle">{names[i]}</text></g>)}
  </svg></div>;
}

/* ---------------- L4: incompatible intervals ---------------- */
export function DbscanIntervalLab() {
  const initial = { offset: 0.75, spacing: 0.75, eps: 0.25 };
  const [pending, setPending] = useState(initial);
  const [applied, setApplied] = useState(null);
  const [prediction, setPrediction] = useState({});
  const fixture = intervalFixture(pending.offset, pending.spacing);
  const fit = dbscan(fixture.points, pending.eps, 3);
  const key = JSON.stringify(pending);
  const stale = applied !== null && applied.key !== key;
  const shown = applied && !stale;
  const update = patch => setPending({ ...pending, ...patch });
  const names = fixture.points.map((_, i) => `${['L', 'M', 'R'][Math.floor(i / 4)]}${i % 4 + 1}`);
  const complete = fit.clusters === 3 && fit.noiseIds.length === 0;
  const axisX = value => 20 + 280 * value / 2;
  return <Investigation title="Can any radius meet both requirements?" question="Three intended groups on one line with m = 3: two dense groups and one you can make sparser or denser. Commit whether some single radius keeps the two dense groups apart and still makes the third group viable, then test a radius of your own." onReset={() => { setPending(initial); setApplied(null); setPrediction({}); }}>
    <div className="db-controls">
      <Field label="Middle group offset from the left group" value={number(pending.offset)}><input type="range" min="0.625" max="2" step="0.125" value={pending.offset} onChange={event => update({ offset: Number(event.target.value) })} /></Field>
      <Field label="Spacing inside the right group" value={number(pending.spacing)}><input type="range" min="0.125" max="1" step="0.125" value={pending.spacing} onChange={event => update({ spacing: Number(event.target.value) })} /></Field>
      <Field label="Radius ε to test" value={number(pending.eps)}><input type="range" min="0.125" max="2" step="0.125" value={pending.eps} onChange={event => update({ eps: Number(event.target.value) })} /></Field>
    </div>
    <Prediction prompt={`Left group ${fixture.groups.left.join(', ')}; middle ${fixture.groups.middle.map(v => number(v)).join(', ')}; right ${fixture.groups.right.map(v => number(v)).join(', ')}; m = 3.`} fields={[
      { key: 'exists', label: 'Does a radius recover all three groups?', options: [['yes', 'Yes, some radius works'], ['no', 'No radius can'], ['inspect', 'I need to inspect the intervals (not graded)']], answer: fixture.exists ? 'yes' : 'no', neutral: ['inspect'] },
      { key: 'tested', label: `Does ε = ${number(pending.eps)} recover all three?`, options: [['yes', 'Yes'], ['no', 'No']], answer: complete ? 'yes' : 'no' },
    ]} values={prediction} onChange={(field, value) => setPrediction({ ...prediction, [field]: value })} committed={applied !== null} stale={stale} onCommit={() => setApplied({ key })} explanation={`The right group is viable from ε = ${number(fixture.lower)} (closed); the two dense groups stay separate below ε = ${number(fixture.upper)} (open, because the closed boundary joins them at exactly that radius). ${fixture.exists ? `Every ε in [${number(fixture.lower)}, ${number(fixture.upper)}) works.` : 'The first requirement begins where the second has already failed: the intervals do not overlap.'} At ε = ${number(pending.eps)}: ${fit.clusters} component${fit.clusters === 1 ? '' : 's'}, ${fit.noiseIds.length} noise.`} />
    {shown ? <>
      <div className="db-scroll"><svg className="db-interval-svg" viewBox="0 0 320 96" role="img" aria-label={`Two requirement intervals on a radius axis from 0 to 2: right group viable from ${number(fixture.lower)} upward; dense groups separate below ${number(fixture.upper)}. ${fixture.exists ? 'They overlap.' : 'They do not overlap.'}`} style={{ width: '100%', maxWidth: 560, display: 'block', margin: '1rem auto' }}>
        {[0, 0.5, 1, 1.5, 2].map(v => <g key={v}><line x1={axisX(v)} x2={axisX(v)} y1="14" y2="74" className="db-grid" /><text x={axisX(v)} y="88" textAnchor="middle">{v}</text></g>)}
        <line className="db-interval" x1={axisX(fixture.lower)} x2={axisX(2)} y1="28" y2="28" style={{ stroke: '#8eb9a5' }} /><circle cx={axisX(fixture.lower)} cy="28" r="4" fill="#8eb9a5" /><text x={axisX(Math.min(fixture.lower + 0.05, 1.5))} y="20">right group viable: ε ≥ {number(fixture.lower)}</text>
        <line className="db-interval" x1={axisX(0)} x2={axisX(fixture.upper)} y1="56" y2="56" style={{ stroke: '#e7b94a' }} /><circle cx={axisX(fixture.upper)} cy="56" r="4" fill="#0b0f10" stroke="#e7b94a" strokeWidth="2" /><text x={axisX(0.05)} y="48">dense groups separate: ε {'<'} {number(fixture.upper)}</text>
        {fixture.exists && <rect x={axisX(fixture.lower)} y="22" width={axisX(fixture.upper) - axisX(fixture.lower)} height="40" fill="#8eb9a5" fillOpacity=".18" />}
        <line x1={axisX(pending.eps)} x2={axisX(pending.eps)} y1="10" y2="76" stroke="#f2e7ca" strokeDasharray="4 3" /><text x={axisX(pending.eps) + 4} y="72" fill="#f2e7ca">tested ε</text>
      </svg></div>
      <p className="db-readout" aria-live="polite">{fixture.exists ? `Usable interval [${number(fixture.lower)}, ${number(fixture.upper)}).` : 'No radius satisfies both requirements.'} At ε = {number(pending.eps)}: components {fit.components.map(g => `{${g.map(i => names[i]).join(', ')}}`).join(' ') || 'none'}; border {fit.borderIds.map(i => names[i]).join(', ') || 'none'}; noise {fit.noiseIds.map(i => names[i]).join(', ') || 'none'}.</p>
      <SquarePlot points={fixture.points} title={`All twelve rows on the true distance axis at ε = ${number(pending.eps)}`} pad={0.1} describe="Twelve rows on a line in three intended groups, with their types at the tested radius.">
        {(project, scale) => <GraphLayer points={fixture.points} fit={fit} project={project} scale={scale} names={names} showLabels={false} />}
      </SquarePlot>
      <DenseStrip points={fixture.points} fit={fit} names={names} from={-0.125} to={Math.max(1.5, pending.offset + 0.5)} caption={`Magnified: the two dense groups from −0.125 to ${number(Math.max(1.5, pending.offset + 0.5))} meters, every row countable`} />
      <GlyphLegend components />
      <Table caption="Each intended group and the radius it needs" headings={['group', 'rows', 'requirement']} rows={[['left', fixture.groups.left.join(', '), 'dense; separate from middle while ε < ' + number(fixture.upper)], ['middle', fixture.groups.middle.map(v => number(v)).join(', '), 'dense; joins left at ε = ' + number(fixture.upper)], ['right', fixture.groups.right.map(v => number(v)).join(', '), `needs ε ≥ ${number(fixture.lower)} for an interior row to reach three neighbours`]]} />
    </> : <div className="db-hidden">Commit a prediction to reveal the interval calculation and the graph at your tested radius.</div>}
    <p className="db-caption">The right group's interior rows gain their third neighbour when ε reaches the spacing; the left and middle groups join through their exactly representable gap of offset − 0.375 because the neighbourhood is closed. A radius exists only when max(0.125, spacing) is below that gap. At the baseline (offset 0.75, spacing 0.75) it does not; with spacing 0.125 every ε in [0.125, 0.375) recovers three groups; with spacing 0.25, practice F's repair, the interval is [0.25, 0.375). This is a proof about intervals, not a coarse sweep that failed to find a value.</p>
  </Investigation>;
}
