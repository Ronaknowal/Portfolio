import { useId, useState } from 'react';
import './ica-labs.css';

/** Exact values are the primary data; this only chooses how many places show. */
export function format(value, places = 6) {
  if (value === null || value === undefined) return '—';
  if (!Number.isFinite(value)) return '—';
  const rounded = Number(value.toFixed(places));
  const text = Object.is(rounded, -0) ? '0' : String(rounded);
  return text.replace('-', '−');
}

export const signedFormat = (value, places = 6) => {
  const text = format(Math.abs(value), places);
  return text === '0' ? '0' : `${value < 0 ? '−' : '+'}${text}`;
};

const copy = value => JSON.parse(JSON.stringify(value));

export const feedbackNumber = value => Number(value.toPrecision(12)).toString().replace('-', '−');
export const feedbackDifference = value => value === 0 ? '0' : `${value < 0 ? '−' : '+'}${feedbackNumber(Math.abs(value))}`;

export function useIcaInvestigation(initial, validate = () => null) {
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

export function Investigation({ kind, title, children, state }) {
  const titleId = useId();
  return <section className="ic-investigation" data-ica-lab={kind} aria-labelledby={titleId} data-live-exploration>
    <header>
      <h3 id={titleId}>{title}</h3>
      <div className="ic-buttons">
        <button type="button" onClick={state.reset}>Reset</button>
        <button type="button" onClick={state.undo} disabled={!state.previous}>Undo edit</button>
      </div>
    </header>
    {state.error && <p className="ic-error" role="status">{state.error} The last valid view is retained.</p>}
    {children}
  </section>;
}

export function NumberField({ label, value, onChange, min, max, step = 'any', disabled = false }) {
  return <label className="ic-field">
    <span>{label}</span>
    <span className="ic-input-row">
      <input type="number" value={value} min={min} max={max} step={step} disabled={disabled}
        onChange={event => onChange(event.target.value === '' ? '' : Number(event.target.value))} />
    </span>
  </label>;
}

export function RangeField({ label, value, onChange, min, max, step }) {
  const labelId = useId();
  return <label className="ic-field ic-range">
    <span id={labelId}>{label}</span>
    <input type="range" aria-labelledby={labelId} value={Number.isFinite(value) ? value : min}
      min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}

export function SelectField({ label, value, onChange, options }) {
  const labelId = useId();
  return <label className="ic-field">
    <span id={labelId}>{label}</span>
    <select aria-labelledby={labelId} value={value} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={text === undefined ? key : key}>{text ?? key}</option>)}
    </select>
  </label>;
}

/** The caption sits outside the scroll region so a wide table never clips it.
 *  `stack` turns the table into one labelled block per row on a narrow screen,
 *  because a silently clipped numeral reads as a complete, wrong number.
 */
export function DataTable({ caption, headings, rows, dense = false, stack = false }) {
  const captionId = useId();
  return <div className="ic-table-block">
    <p className="ic-table-caption" id={captionId}>{caption}</p>
    <div className={`ic-table-wrap${dense ? ' is-dense' : ''}${stack ? ' is-stack' : ''}`}
      tabIndex={0} role="region" aria-labelledby={captionId}>
      <table>
        <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
        <tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, column) => column === 0
          ? <th key={column} scope="row">{cell}</th>
          : <td key={column} data-label={headings[column]}>{cell}</td>)}</tr>)}</tbody>
      </table>
    </div>
  </div>;
}

/** A square drawing area so one x unit is exactly as long as one y unit.
 *  The viewBox matches the rendered CSS width, so 11 SVG units really are 11px.
 */
export function EqualPlot({ title, describe, bounds, children, ticks, size = 300, className = '' }) {
  const titleId = useId();
  const [xLow, xHigh, yLow, yHigh] = bounds;
  const span = Math.max(xHigh - xLow, yHigh - yLow);
  const padLeft = 34;
  const padBottom = 32;
  const padTop = 10;
  // Room for half of the right-most tick label, which is centred on the axis.
  const plot = size - padLeft - 22;
  const centreX = (xLow + xHigh) / 2;
  const centreY = (yLow + yHigh) / 2;
  const sx = value => padLeft + ((value - (centreX - span / 2)) * plot) / span;
  const sy = value => padTop + plot - ((value - (centreY - span / 2)) * plot) / span;
  const tickValues = ticks ?? [centreX - span / 2, centreX, centreX + span / 2];
  return <figure className={`ic-plot ${className}`}>
    <figcaption id={titleId}>{title}</figcaption>
    <svg viewBox={`0 0 ${size} ${plot + padTop + padBottom}`} style={{ maxWidth: `${size}px` }} role="img" aria-labelledby={titleId}>
      <desc>{describe}</desc>
      {tickValues.map(tick => <g key={`x${tick}`}>
        <line className="ic-grid" x1={sx(tick)} x2={sx(tick)} y1={padTop} y2={padTop + plot} />
        <text x={sx(tick)} y={padTop + plot + 22} textAnchor="middle">{format(tick, 2)}</text>
      </g>)}
      {tickValues.map(tick => <g key={`y${tick}`}>
        <line className="ic-grid" x1={padLeft} x2={padLeft + plot} y1={sy(tick)} y2={sy(tick)} />
        <text x={padLeft - 5} y={sy(tick) + 4} textAnchor="end">{format(tick, 2)}</text>
      </g>)}
      {children(sx, sy, { plot, padLeft, padTop })}
    </svg>
  </figure>;
}

export function Readout({ label, value, note }) {
  return <p className="ic-readout"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</p>;
}

/** `resetKey` remounts the field, so Reset really does clear everything. */
export function Reflection({ prompt, resetKey = 0 }) {
  return <label className="ic-field ic-reflection">
    <span>{prompt}</span>
    <textarea key={resetKey} rows={3} placeholder="Write your explanation here; it is not automatically scored." />
  </label>;
}
