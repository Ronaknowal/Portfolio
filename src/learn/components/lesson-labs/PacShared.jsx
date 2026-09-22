import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { cloneElement, isValidElement, useId, useState } from 'react';
import './pac-labs.css';
import { linearScale } from '../../data/pac-models.js';



const sign = text => String(text).replace('-', '−');

export const round = (value, digits = 6) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '−∞';
  if (Number.isInteger(value)) return sign(String(value));
  if (value !== 0 && Math.abs(value) < 1e-4) {
    const [mantissa, exponent] = value.toExponential(3).split('e');
    const marks = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    const superscript = [...exponent.replace('+', '')].map(character => marks[character] ?? character).join('');
    return `${sign(mantissa)} × 10${superscript}`;
  }
  const text = value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
  return text === '-0' ? '0' : sign(text);
};

/** The canonical form for a COMPUTED value: every decimal place, always. This
 *  is the form the browser verifier looks for, so it must not be used for an
 *  input the learner typed. */
export const fixed = (value, digits = 6) =>
  (Number.isFinite(value) ? sign(value.toFixed(digits)) : round(value));

/** A short form for an INPUT the learner set. Deliberately never six decimals. */
export const asInput = value => (Number.isInteger(value) ? String(value) : sign(String(value)));

export function Investigation({ title, question, note, role, children, onReset }) {
  const id = useId();
  return <section className="pac-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="pac-question">{question}</p>}
    {role && <p className={`pac-role${role.kind ? ` is-${role.kind}` : ''}`} role="status">{role.text}</p>}
    {note && <p className="pac-note">{note}</p>}
    {children}
  </section>;
}

export function Field({ label, value, error, hint, children }) {
  const id = useId();
  const described = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ');
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': described || undefined })
    : children;
  return <label className={`pac-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {hint && <span className="pac-caption" id={`${id}-hint`}>{hint}</span>}
    {error && <span className="pac-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that publishes only a valid value. An unparsable or
 *  out-of-range draft stays on screen with an explanation naming the field, and
 *  the model keeps its last good state rather than being handed a substitute.
 *
 *  Typed, not a slider: this lesson's coordinates go to three decimals and a
 *  slider that could not reach .31 would make its own worked example
 *  unreachable inside its own lab. */
export function NumberField({
  label, value, onChange, min, max, step = 'any', decimals = 3, suffix, hint, disabled = false, placeholder,
}) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const scale = 10 ** decimals;
  const problem = (() => {
    if (draft === null) return null;
    const trimmed = draft.trim();
    if (trimmed === '' || trimmed === '-' || trimmed === '−') return 'Type a number.';
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    if (Math.abs(parsed * scale - Math.round(parsed * scale)) > 1e-9) {
      return `Use at most ${decimals} decimal place${decimals === 1 ? '' : 's'}.`;
    }
    return null;
  })();
  return <Field label={label} error={problem} value={suffix} hint={hint}>
    <input type="number" inputMode="decimal" min={min} max={max} step={step} value={shown} disabled={disabled}
      placeholder={placeholder}
      onChange={event => {
        setDraft(event.target.value);
        const parsed = Number(event.target.value);
        if (event.target.value.trim() !== '' && Number.isFinite(parsed) && parsed >= min && parsed <= max
          && Math.abs(parsed * scale - Math.round(parsed * scale)) <= 1e-9) {
          onChange(Number(parsed.toFixed(decimals)));
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

/** A 0/1 label switch for one named point. Two states, both named in words as
 *  well as digits, because "1" alone does not say positive. */
export function LabelChoice({ label, value, onChange, disabled = false }) {
  const name = useId();
  return <div className="pac-label-choice">
    <span className="pac-label-name">{label}</span>
    <span className="pac-label-options">
      {[[0, '0 · negative'], [1, '1 · positive']].map(([key, text]) => (
        <label key={key}>
          <input type="radio" name={name} value={key} checked={value === key} disabled={disabled}
            onChange={() => onChange(key)} />
          <span>{text}</span>
        </label>
      ))}
    </span>
  </div>;
}

/** A table whose caption sits OUTSIDE the scroll box.
 *
 *  Below the narrow breakpoint every cell becomes its own labelled row, which
 *  needs `display: block`; that drops the implicit table roles, so they are
 *  restored explicitly. A `<caption>` inside a blockified table collapses to
 *  the width of its longest word, so the caption is a sibling paragraph. */
export function Table({ caption, headings, rows, rowClass = () => undefined, footnote }) {
  const id = useId();
  return <div className="pac-table">
    <p className="pac-caption" id={id}>{caption}</p>
    <div className="pac-table-scroll" role="region" aria-labelledby={id} tabIndex={0}>
      <table role="table">
        <thead role="rowgroup"><tr role="row">
          {headings.map(heading => <th key={heading} role="columnheader" scope="col">{heading}</th>)}
        </tr></thead>
        <tbody role="rowgroup">{rows.map((row, index) => <tr key={index} role="row" className={rowClass(index)}>
          {row.map((cell, column) => (column === 0
            ? <th key={column} role="rowheader" scope="row" data-label={headings[column]}>{cell}</th>
            : <td key={column} role="cell" data-label={headings[column]}>{cell}</td>))}
        </tr>)}</tbody>
      </table>
    </div>
    {footnote && <p className="pac-caption">{footnote}</p>}
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

/* ================================================== shared drawing primitives */

/**
 * One labelled axis from 0 to 1 (or any stated domain) carrying bands, strips
 * and points.
 *
 * Every x position comes from the passed scale, which comes from
 * `pac-models.js`. Bands state their own fill in CSS rather than relying on a
 * presentation attribute, because a CSS `fill` rule beats an attribute and that
 * is how a figure's bands once became invisible while every assertion passed.
 */
/** Vertical stipple inside a band. Drawn as real line elements rather than a
 *  `url(#pattern)` fill: a pattern reference that does not resolve fails
 *  silently and leaves the band unpainted. */
function stipple(band, y, height, kind, step = 4) {
  const marks = [];
  for (let offset = step / 2; offset < band.width; offset += step) marks.push(band.x + offset);
  return marks.map((x, index) => <line key={`${kind}-${index}`} className={`pac-hatch-line is-${kind}`}
    x1={x} y1={y} x2={x} y2={y + height} />);
}

export function NumberLine({ geometry, caption, describe, showStrips = false, pointLabels = true }) {
  const titleId = useId();
  const descriptionId = useId();
  const { width, height, rows, inset } = geometry;
  /* Every row's x comes from the shared scale, so the rows line up vertically by
     construction; only the y offsets live here, and they come from the model so
     the verifier asserts the same ones the browser paints. */
  return <figure className="pac-figure-inline">
    <svg className="pac-line" viewBox={`0 0 ${width} ${height}`} role="img"
      aria-labelledby={titleId} aria-describedby={descriptionId} style={{ maxWidth: `${width}px` }}>
      <title id={titleId}>{caption}</title>
      <desc id={descriptionId}>{describe}</desc>

      {/* 1 · what was observed */}
      {geometry.points.map((point, index) => <g key={`p${index}`}
        className={`pac-point${point.label === 1 ? ' is-positive' : ''}`}>
        {point.label === 1
          ? <circle cx={point.position} cy={rows.markY} r={3.4} />
          : <rect x={point.position - 3} y={rows.markY - 3} width={6} height={6} />}
        {pointLabels && point.labelRow !== null && <text className="pac-small" x={point.position}
          y={rows.pointLabelY - point.labelRow * rows.pointLabelRowGap} textAnchor="middle">
          {point.x}
        </text>}
      </g>)}

      {/* 2 · the target */}
      <rect className="pac-band is-target" x={geometry.target.x} y={rows.targetY}
        width={Math.max(geometry.target.width, 0.6)} height={rows.targetHeight} />

      {/* 3 · where a sample would have to land to force risk at most epsilon */}
      {showStrips && geometry.strips.map((strip, index) => <g key={`strip${index}`}>
        <rect className="pac-band is-coverage"
          x={strip.x} y={rows.stripY} width={Math.max(strip.width, 0.6)} height={rows.stripHeight} />
        {stipple(strip, rows.stripY, rows.stripHeight, 'coverage', 5)}
      </g>)}

      {/* 4 · where the fit is wrong */}
      {geometry.segments.map((segment, index) => <g key={`seg${index}`}>
        <rect className={`pac-band is-${segment.kind}`}
          x={segment.x} y={rows.segmentY} width={Math.max(segment.width, 0.6)} height={rows.segmentHeight} />
        {stipple(segment, rows.segmentY, rows.segmentHeight, segment.kind, 3)}
      </g>)}

      {/* 5 · what was fitted */}
      {geometry.fitted && <rect className="pac-band is-fitted" x={geometry.fitted.x} y={rows.fittedY}
        width={Math.max(geometry.fitted.width, 0.6)} height={rows.fittedHeight} />}

      {/* 6 · one shared axis, below everything it indexes */}
      <line className="pac-axis" x1={inset} y1={rows.axisY} x2={width - inset} y2={rows.axisY} />
      {geometry.ticks.map((tick, index) => <g key={tick.value}>
        <line className="pac-tick" x1={tick.x} y1={rows.axisY} x2={tick.x} y2={rows.tickY} />
        <text className="pac-small" x={tick.x} y={rows.tickLabelY}
          textAnchor={index === 0 ? 'start' : index === geometry.ticks.length - 1 ? 'end' : 'middle'}>
          {tick.value}
        </text>
      </g>)}
    </svg>
    {/* The rows are named here rather than inside the drawing: labelling five
        rows in SVG would eat a third of a 300-unit line, and these names reflow.
        Each carries a shape or pattern as well as a colour. */}
    <div className="pac-legend">
      <span><i className="pac-swatch is-observed-positive" aria-hidden="true" />positive observation</span>
      <span><i className="pac-swatch is-observed-negative" aria-hidden="true" />negative observation</span>
      <span><i className="pac-swatch is-target-row" aria-hidden="true" />target region</span>
      {showStrips && <span><i className="pac-swatch is-coverage-row" aria-hidden="true" />ε/2 coverage strips</span>}
      <span><i className="pac-swatch is-missed-row" aria-hidden="true" />where the fit is wrong</span>
      <span><i className="pac-swatch is-fitted-row" aria-hidden="true" />fitted region</span>
    </div>
    <p className="pac-caption">{caption}</p>
  </figure>;
}

/** A plot frame: axes, ticks and their labels, with the drawing area left to
 *  the caller. The ticks come from the geometry so the verifier asserts the
 *  same positions the browser paints. */
export function PlotFrame({ geometry, xLabel, yLabel, xTickText = value => String(value), yTickText = value => String(value), children, caption, describe }) {
  const titleId = useId();
  const descriptionId = useId();
  const { width, height, padding } = geometry;
  return <figure className="pac-figure-inline">
    <svg className="pac-plot" viewBox={`0 0 ${width} ${height}`} role="img"
      aria-labelledby={titleId} aria-describedby={descriptionId} style={{ maxWidth: `${width}px` }}>
      <title id={titleId}>{caption}</title>
      <desc id={descriptionId}>{describe}</desc>
      {geometry.yTicks.map(tick => <g key={`y${tick.value}`}>
        <line className="pac-grid" x1={padding.left} y1={tick.y} x2={width - padding.right} y2={tick.y} />
        <text className="pac-small" x={padding.left - 5} y={tick.y + 3} textAnchor="end">{yTickText(tick.value)}</text>
      </g>)}
      {/* The first and last tick labels are anchored inward. Centring them put
          "0" half outside the left edge and "100,000" a clear four pixels past
          the right one, which the layout inspector found before a reader did. */}
      {geometry.xTicks.map((tick, index) => <g key={`x${tick.value}`}>
        <line className="pac-tick" x1={tick.x} y1={height - padding.bottom} x2={tick.x} y2={height - padding.bottom + 4} />
        <text className="pac-small" x={tick.x} y={height - padding.bottom + 15}
          textAnchor={index === 0 ? 'start' : index === geometry.xTicks.length - 1 ? 'end' : 'middle'}>
          {xTickText(tick.value)}
        </text>
      </g>)}
      <line className="pac-axis" x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} />
      <line className="pac-axis" x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} />
      {children}
      <text className="pac-small pac-axis-title" x={width - padding.right} y={height - 3} textAnchor="end">{xLabel}</text>
      {/* The y-axis title sits in its own band ABOVE the plot, anchored at the
          left edge. End-anchored beside the axis it ran off the left of the
          viewBox and collided with the topmost tick label. `padding.top` is
          sized to keep that band clear. */}
      <text className="pac-small pac-axis-title" x={2} y={10} textAnchor="start">{yLabel}</text>
    </svg>
    <p className="pac-caption">{caption}</p>
  </figure>;
}

/** A polyline through computed points. Marks are drawn as well as the line so
 *  the individual values stay visible at phone width. */
export function Series({ points, className, marker = 'dot' }) {
  return <g className={`pac-series ${className}`}>
    <polyline className="pac-series-line"
      points={points.map(point => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ')} />
    {points.map((point, index) => (marker === 'square'
      ? <rect key={index} className="pac-series-mark" x={point.x - 2.6} y={point.y - 2.6} width={5.2} height={5.2} />
      : <circle key={index} className="pac-series-mark" cx={point.x} cy={point.y} r={2.6} />))}
  </g>;
}

/** A row of bits with an accessible text equivalent beside it. */
export function BitRow({ bits, highlight = [], title }) {
  return <span className="pac-bits" title={title}>
    {bits.map((bit, index) => <span key={index}
      className={`pac-bit${highlight.includes(index) ? ' is-wrong' : ''}`}>{bit}</span>)}
  </span>;
}

export { linearScale };
