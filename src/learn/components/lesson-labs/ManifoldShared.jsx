import { useId, useState } from 'react';
import './manifold-labs.css';

export const format = (value, places = 5) => {
  if (value === null) return 'No path';
  if (!Number.isFinite(value)) return '—';
  if (value !== 0 && Math.abs(value) < 0.00001) return value.toExponential(2);
  return Number(value.toFixed(places)).toString();
};

// Scientific notation for tiny values also needs enough significant digits.
export function comparisonValues(before, after) {
  const compact = [format(before), format(after)];
  if (compact[0] !== compact[1] || Math.abs(after - before) <= 1e-9) return compact;
  return [before, after].map(value => Number(value.toPrecision(12)).toString());
}

const copy = value => JSON.parse(JSON.stringify(value));

export function useManifoldInvestigation(initial, validate = () => null) {
  const [view, setView] = useState(() => ({draft:copy(initial), active:copy(initial), previous:null, resetCount:0}));
  const validation = validate(view.draft, view.active);
  const error = typeof validation === 'string' ? validation : null;
  const edit = changes => setView(current => {
    const draft = {...current.draft,...changes};
    const problem = validate(draft, current.active);
    return {...current,draft,...(typeof problem === 'string' && problem ? {} : {active:copy(draft),previous:{active:current.active}})};
  });
  const reset = () => setView(current => ({draft:copy(initial),active:copy(initial),previous:null,resetCount:current.resetCount+1}));
  const undo = () => setView(current => current.previous ? {...current,draft:copy(current.previous.active),active:copy(current.previous.active),previous:null} : current);
  return {...view,key:JSON.stringify(view.draft),error,edit,reset,undo};
}

export function Investigation({ kind, title, children, state, onReset }) {
  const titleId = useId();
  return <section className="mf-investigation" data-manifold-lab={kind} aria-labelledby={titleId} data-live-exploration>
    <header><h3 id={titleId}>{title}</h3><div className="mf-buttons">
      <button type="button" onClick={onReset ?? state.reset}>Reset</button>
      <button type="button" onClick={state.undo} disabled={!state.previous}>Undo edit</button>
    </div></header>
    {state.error && <p className="mf-error" role="status">{state.error} The last valid view is retained.</p>}
    {children}
  </section>;
}

export function NumberField({ label, value, onChange, min, max, step = 'any', disabled = false }) {
  return <label className="mf-field"><span>{label}</span><input type="number" value={value}
    min={min} max={max} step={step} disabled={disabled}
    onChange={event => onChange(event.target.value === '' ? '' : Number(event.target.value))} /></label>;
}

export function SelectField({ label, value, onChange, options }) {
  const labelId = useId();
  return <label className="mf-field"><span id={labelId}>{label}</span><select aria-labelledby={labelId} value={value} onChange={event => onChange(event.target.value)}>
    {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
  </select></label>;
}

export function DataTable({ caption, headings, rows }) {
  return <div className="mf-table-wrap"><table><caption>{caption}</caption>
    <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
    <tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
  </table></div>;
}

/** A square plotting area guarantees that one x unit has the same length as one y unit. */
export function EqualPlot({ points, title, children, describe, bounds }) {
  const titleId = useId();
  const xValues = points.map(point => point.x);
  const yValues = points.map(point => point.y);
  const xMin = bounds?.[0] ?? Math.min(...xValues);
  const xMax = bounds?.[1] ?? Math.max(...xValues);
  const yMin = bounds?.[2] ?? Math.min(...yValues);
  const yMax = bounds?.[3] ?? Math.max(...yValues);
  const span = Math.max(xMax - xMin, yMax - yMin, 1) * 1.15;
  const centerX = (xMin + xMax) / 2;
  const centerY = (yMin + yMax) / 2;
  const xLow = centerX - span / 2;
  const yLow = centerY - span / 2;
  const sx = value => 58 + (value - xLow) * 230 / span;
  const sy = value => 258 - (value - yLow) * 230 / span;
  const ticksX = [xLow, centerX, xLow + span];
  const ticksY = [yLow, centerY, yLow + span];
  return <figure className="mf-equal-plot">
    <figcaption id={titleId}>{title}</figcaption>
    <svg viewBox="0 0 330 298" role="img" aria-labelledby={titleId}>
      <desc>{describe}</desc>
      {ticksX.map((tick, index) => <g key={`x${index}`}>
        <line className="mf-grid" x1={sx(tick)} x2={sx(tick)} y1={28} y2={258} />
        <text x={sx(tick)} y={278} textAnchor="middle">{format(tick, 1)}</text>
      </g>)}
      {ticksY.map((tick, index) => <g key={`y${index}`}>
        <line className="mf-grid" x1={58} x2={288} y1={sy(tick)} y2={sy(tick)} />
        <text x={50} y={sy(tick) + 4} textAnchor="end">{format(tick, 1)}</text>
      </g>)}
      {children(sx, sy)}
    </svg>
  </figure>;
}

const digitColors = ['#d9b35b', '#90bda4', '#b0b8d7', '#dba697', '#9fc0c9', '#d0b8d2', '#b9bf82', '#a6b3a4', '#c1a17d', '#e2cfaa'];

export function DigitLegend() {
  return <div className="mf-digit-legend" aria-label="Digit color key">{digitColors.map((color, digit) =>
    <span key={digit}><i style={{ background: color }} aria-hidden="true" />Digit {digit}</span>)}</div>;
}

export function DigitScatter({ rows, coordinates, queryId = null, onSelect, neighbors = null, colorLabels = false, title }) {
  const points = coordinates.map(([x, y], index) => ({ x, y, ...rows[index] }));
  const query = points.find(point => point.sourceRow === queryId);
  const inputSet = new Set(neighbors?.input ?? []);
  const mapSet = new Set(neighbors?.map ?? []);
  const relation = id => inputSet.has(id) && mapSet.has(id) ? 'retained' : inputSet.has(id) ? 'input-only' : 'map-only';
  return <div className="mf-digit-scatter">
    <EqualPlot points={points} title={title}
      describe={`300 actual saved observations, with equal scales on both map-coordinate axes.${query ? ` Selected source row ${queryId}.` : ''} Select any source ID using the adjacent control.`}>
      {(sx, sy) => <>
        {query && neighbors && points.filter(point => inputSet.has(point.sourceRow) || mapSet.has(point.sourceRow)).map(point =>
          <line key={`edge${point.sourceRow}`} className={`mf-neighbor-edge is-${relation(point.sourceRow)}`}
            x1={sx(query.x)} y1={sy(query.y)} x2={sx(point.x)} y2={sy(point.y)} />)}
        {points.map(point => <circle key={point.sourceRow} cx={sx(point.x)} cy={sy(point.y)}
          r={point.sourceRow === queryId ? 4.5 : 2.1} className={`mf-digit-point${point.sourceRow === queryId ? ' is-query' : ''}`}
          fill={colorLabels ? digitColors[point.digit] : '#a8b7ae'}
          onClick={onSelect ? () => onSelect(point.sourceRow) : undefined}>
          <title>{`Source ${point.sourceRow}${colorLabels ? ` · digit ${point.digit}` : ''}`}</title>
        </circle>)}
        {query && <circle cx={sx(query.x)} cy={sy(query.y)} r={7} className="mf-query-ring" />}
      </>}
    </EqualPlot>
    <p className="mf-caption">Map coordinates · equal axis scale within this map{query ? ` · selected source ${queryId}` : ''}.</p>
  </div>;
}

export function DigitTile({ row, caption, showLabel = true }) {
  return <figure className="mf-digit-tile">
    <svg viewBox="0 0 80 80" role="img" aria-label={`Observed 8 by 8 pixels from source ${row.sourceRow}${showLabel ? `, digit ${row.digit}` : ''}`}>
      {row.pixels.map((value, index) => <rect key={index} x={(index % 8) * 10} y={Math.floor(index / 8) * 10}
        width={10} height={10} fill={`rgb(${Math.round(value * 255 / 16)},${Math.round(value * 255 / 16)},${Math.round(value * 255 / 16)})`} />)}
    </svg>
    <figcaption>{caption ?? `Source ${row.sourceRow}${showLabel ? ` · digit ${row.digit}` : ''}`}</figcaption>
  </figure>;
}

export function Reflection({ prompt }) {
  return <label className="mf-field mf-reflection"><span>{prompt}</span><textarea rows={3} placeholder="Write your explanation here; it is not automatically scored." /></label>;
}
