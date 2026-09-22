import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { cloneElement, isValidElement, useId, useState } from 'react';
import './timeseries-labs.css';



const sign = text => String(text).replace('-', '−');

export const round = (value, digits = 6) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '−∞';
  if (Number.isInteger(value)) return sign(String(value));
  const text = value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
  return text === '-0' ? '0' : sign(text);
};

/** The canonical form for a COMPUTED value: every decimal place, always. This
 *  is the form the browser verifier looks for, so it must not be used for an
 *  input the learner typed or for a recorded count. */
export const fixed = (value, digits = 6) =>
  (Number.isFinite(value) ? sign(value.toFixed(digits)) : round(value));

/** A short form for an INPUT the learner set, or a recorded count. Deliberately
 *  never six decimals. */
export const asInput = value => (Number.isInteger(value) ? String(value) : sign(String(value)));

/** Rentals are whole bicycles. Grouped, because a five-figure count read as one
 *  run of digits at phone size is a count nobody can check. */
export const rentals = value => (Number.isFinite(value)
  ? sign(Math.round(value).toLocaleString('en-US'))
  : '—');

/* ========================================================== the SVG wrapper */

const DIAGRAM_KINDS = { line: 'ts-line', plot: 'ts-plot', panel: 'ts-panel' };

/**
 * The only place in this lesson that writes an `<svg>` element.
 *
 * `ts-diagram` is what the stylesheet's layout rule is scoped to; the kind
 * class carries the sizing appropriate to a timeline, a chart or a small panel.
 * Both are applied here so that a figure cannot forget them, and the hygiene
 * verifier refuses any other file that opens an svg tag.
 */
export function Diagram({ kind = 'plot', width, height, title, describe, children, extraClass }) {
  const titleId = useId();
  const descriptionId = useId();
  const layout = DIAGRAM_KINDS[kind];
  if (!layout) throw new RangeError(`unknown diagram kind ${kind}`);
  return <svg
    className={`ts-diagram ${layout}${extraClass ? ` ${extraClass}` : ''}`}
    viewBox={`0 0 ${width} ${height}`} role="img"
    aria-labelledby={titleId} aria-describedby={descriptionId} style={{ maxWidth: `${width}px` }}>
    <title id={titleId}>{title}</title>
    <desc id={descriptionId}>{describe}</desc>
    {children}
  </svg>;
}

/** A diagram with its caption below it, outside any scrolling box. */
export function Plate({ caption, children }) {
  return <figure className="ts-figure-inline">
    {children}
    {caption && <p className="ts-caption">{caption}</p>}
  </figure>;
}

/* ================================================================= controls */

export function Investigation({ title, question, note, role, children, onReset }) {
  const id = useId();
  return <section className="ts-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="ts-question">{question}</p>}
    {role && <p className={`ts-role${role.kind ? ` is-${role.kind}` : ''}`} role="status">{role.text}</p>}
    {note && <p className="ts-note">{note}</p>}
    {children}
  </section>;
}

export function Field({ label, value, error, hint, children }) {
  const id = useId();
  const described = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ');
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': described || undefined })
    : children;
  return <label className={`ts-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {hint && <span className="ts-caption" id={`${id}-hint`}>{hint}</span>}
    {error && <span className="ts-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A whole-number field that publishes only a valid value. An unparsable or
 *  out-of-range draft stays on screen with an explanation naming the field, and
 *  the model keeps its last good state rather than being handed a substitute.
 *
 *  Typed, not a slider: a rental count runs to five figures, and a slider that
 *  could not reach 1754 would put this lesson's own worked contrast out of
 *  reach inside its own lab. */
export function NumberField({ label, value, onChange, min, max, suffix, hint, disabled = false, step = 1 }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const problem = (() => {
    if (draft === null) return null;
    const trimmed = draft.trim();
    if (trimmed === '' || trimmed === '-' || trimmed === '−') return 'Type a whole number.';
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (!Number.isInteger(parsed)) return 'Counts are whole bicycles, so use a whole number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    return null;
  })();
  return <Field label={label} error={problem} value={suffix} hint={hint}>
    <input type="number" inputMode="numeric" min={min} max={max} step={step} value={shown} disabled={disabled}
      onChange={event => {
        setDraft(event.target.value);
        const parsed = Number(event.target.value);
        if (event.target.value.trim() !== '' && Number.isInteger(parsed) && parsed >= min && parsed <= max) {
          onChange(parsed);
        }
      }}
      onBlur={() => setDraft(null)} aria-invalid={Boolean(problem)} />
  </Field>;
}

export function Select({ label, value, onChange, options, hint, disabled = false }) {
  return <Field label={label} hint={hint}>
    <select value={value} disabled={disabled} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
    </select>
  </Field>;
}

/** A set of origins the learner ticks. Keyboard-reachable by construction, and
 *  each box names its own day in words as well as its index. */
export function OriginChecklist({ legend, origins, selected, onToggle, disabled = false, describe }) {
  const name = useId();
  return <fieldset className="ts-checklist">
    <legend>{legend}</legend>
    <div className="ts-checklist-options">
      {origins.map(origin => (
        <label key={origin} className={`ts-checkbox${selected.includes(origin) ? ' is-on' : ''}`}>
          <input type="checkbox" name={name} value={origin} checked={selected.includes(origin)}
            disabled={disabled} onChange={() => onToggle(origin)} />
          <span>{describe ? describe(origin) : `origin ${origin}`}</span>
        </label>
      ))}
    </div>
  </fieldset>;
}

/** A table whose caption sits OUTSIDE the scroll box.
 *
 *  Below the narrow breakpoint every cell becomes its own labelled row, which
 *  needs `display: block`; that drops the implicit table roles, so they are
 *  restored explicitly. A `<caption>` inside a blockified table collapses to
 *  the width of its longest word, so the caption is a sibling paragraph. */
export function Table({ caption, headings, rows, rowClass = () => undefined, footnote, wrap = [] }) {
  const id = useId();
  /* `wrap` names the column indices whose contents are PROSE rather than
     values. Numeric cells keep `white-space: nowrap`, because a wrapped number
     is a number a reader has to reassemble; prose columns must be allowed to
     wrap or the table scrolls sideways at desktop and clips a column, which is
     a table that has lost data rather than a table that scrolls. The browser
     verifier fails on any table wider than its own box at 1366 px, and this
     prop is how a wide prose column is admitted without that. */
  const cellClass = column => (wrap.includes(column) ? 'is-wrap' : undefined);
  return <div className="ts-table">
    <p className="ts-caption" id={id}>{caption}</p>
    <div className="ts-table-scroll" role="region" aria-labelledby={id} tabIndex={0}>
      <table role="table">
        <thead role="rowgroup"><tr role="row">
          {headings.map((heading, column) => <th key={heading} role="columnheader" scope="col"
            className={cellClass(column)}>{heading}</th>)}
        </tr></thead>
        <tbody role="rowgroup">{rows.map((row, index) => <tr key={index} role="row" className={rowClass(index)}>
          {row.map((cell, column) => (column === 0
            ? <th key={column} role="rowheader" scope="row" data-label={headings[column]}
              className={cellClass(column)}>{cell}</th>
            : <td key={column} role="cell" data-label={headings[column]}
              className={cellClass(column)}>{cell}</td>))}
        </tr>)}</tbody>
      </table>
    </div>
    {footnote && <p className="ts-caption">{footnote}</p>}
  </div>;
}


export const useInvestigation = useLiveInvestigation;





export function LiveResult({ state, calculateInputs, blocked, describe }) {
  const problem = useLiveResult(state, calculateInputs, blocked);
  return <div data-live-exploration="result">
    {problem ? <p role="status">{problem} The plots retain the last valid calculation; correct the inputs to update them.</p>
      : <p role="status">Live calculation for the current controls. {describe}</p>}
    <button type="button" disabled={Boolean(problem) || !state.result} onClick={state.snapshot}>Use current values as comparison baseline</button>
  </div>;
}

/* ====================================================== drawing conveniences */

/** A plot frame: axes, ticks and their labels, with the drawing area left to
 *  the caller. The ticks come from the geometry so the verifier asserts the
 *  same positions the browser paints. */
export function PlotFrame({
  geometry, xLabel, yLabel, xTickText = String, yTickText = String, children, title, describe, caption,
}) {
  const { width, height, padding } = geometry;
  return <Plate caption={caption}>
    <Diagram kind="plot" width={width} height={height} title={title} describe={describe}>
      {geometry.yTicks.map(tick => <g key={`y${tick.value}`}>
        <line className="ts-grid" x1={padding.left} y1={tick.y} x2={width - padding.right} y2={tick.y} />
        <text className="ts-small" x={padding.left - 5} y={tick.y + 3} textAnchor="end">{yTickText(tick.value)}</text>
      </g>)}
      {/* The first and last tick labels are anchored inward. Centring them put
          the left one half outside the viewBox and the right one past its
          edge. */}
      {geometry.xTicks.map((tick, index) => <g key={`x${tick.value}`}>
        <line className="ts-tick" x1={tick.x} y1={height - padding.bottom} x2={tick.x} y2={height - padding.bottom + 4} />
        <text className="ts-small" x={tick.x} y={height - padding.bottom + 15}
          textAnchor={index === 0 ? 'start' : index === geometry.xTicks.length - 1 ? 'end' : 'middle'}>
          {xTickText(tick.value)}
        </text>
      </g>)}
      <line className="ts-axis" x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} />
      <line className="ts-axis" x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} />
      {children}
      <text className="ts-small ts-axis-title" x={width - padding.right} y={height - 3} textAnchor="end">{xLabel}</text>
      {/* The y-axis title sits in its own band ABOVE the plot, anchored at the
          left edge. End-anchored beside the axis it ran off the left of the
          viewBox and collided with the topmost tick label. `padding.top` is
          sized to keep that band clear. */}
      <text className="ts-small ts-axis-title" x={2} y={10} textAnchor="start">{yLabel}</text>
    </Diagram>
  </Plate>;
}

/** A polyline through computed points. Marks are drawn as well as the line so
 *  the individual values stay visible at phone width, and so a point shared by
 *  two methods can be given its own ring rather than being hidden underneath. */
export function Series({ points, className, marker = 'dot', encoding }) {
  /* `data-encoding` on the marks, not the line: the browser verifier asserts
     that every encoding a legend NAMES is actually painted and not covered, and
     a mark is the thing a reader looks for. */
  return <g className={`ts-series ${className}`}>
    <polyline className="ts-series-line"
      points={points.map(point => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ')} />
    {points.map((point, index) => (marker === 'square'
      ? <rect key={index} className="ts-series-mark" data-encoding={encoding}
        x={point.x - 2.6} y={point.y - 2.6} width={5.2} height={5.2} />
      : marker === 'triangle'
        ? <polygon key={index} className="ts-series-mark" data-encoding={encoding}
          points={`${point.x},${point.y - 3.2} ${point.x + 3.0},${point.y + 2.4} ${point.x - 3.0},${point.y + 2.4}`} />
        : <circle key={index} className="ts-series-mark" data-encoding={encoding}
          cx={point.x} cy={point.y} r={2.6} />))}
  </g>;
}

/** A reflowing key. The rows are named here rather than inside the drawing:
 *  labelling five rows in SVG would eat a third of a 300-unit line, and these
 *  names reflow. Each carries a shape or a pattern as well as a colour. */
export function Legend({ entries }) {
  return <div className="ts-legend">
    {entries.map(entry => <span key={entry.key}>
      <i className={`ts-swatch is-${entry.key}`} aria-hidden="true" />{entry.text}
    </span>)}
  </div>;
}

/** A row of small labelled cells: a history strip, a forecast strip, an
 *  eligible-origin strip. Text, not geometry, so it reflows at phone width. */
export function CellStrip({ cells, label }) {
  return <div className="ts-strip" role="group" aria-label={label}>
    {cells.map((cell, index) => <span key={index}
      className={`ts-strip-cell${cell.highlight ? ' is-used' : ''}${cell.muted ? ' is-muted' : ''}`}>
      <b>{cell.value}</b>{cell.label}
    </span>)}
  </div>;
}
