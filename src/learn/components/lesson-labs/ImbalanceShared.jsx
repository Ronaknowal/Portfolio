import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { cloneElement, isValidElement, useId, useState } from 'react';
import './imbalance-labs.css';



/** Every printed number uses a typographic minus sign, matching the prose. */
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
/** Keep every decimal place, for a column whose rows must line up. */
export const fixed = (value, digits = 6) => (Number.isFinite(value) ? sign(value.toFixed(digits)) : round(value));
export const signed = (value, digits = 6) => {
  if (value === 0) return '0';
  return value > 0 ? `+${round(value, digits)}` : round(value, digits);
};
/** An exact zero is said, not shaded. A tiny nonzero value is never printed 0. */
export const exactly = (value, digits = 6) => (value === 0 ? 'exactly 0' : round(value, digits));

/** A quantity whose denominator is empty is undefined, and says which one.
 *
 * This is the single most load-bearing formatting rule in the lesson: precision
 * with nothing selected is not zero, and printing it as zero would teach the
 * mistake the section exists to prevent.
 */
export function Undefined({ because }) {
  return <span className="imb-undefined" title={because}>undefined <span className="imb-caption">({because})</span></span>;
}
export const ratio = (value, because, digits = 6) =>
  (value === null || value === undefined ? <Undefined because={because} /> : round(value, digits));

/** A small exact fraction reads better than six decimals when it is exact. */
export function asFraction(value, maximumDenominator = 24) {
  if (value === null || !Number.isFinite(value)) return null;
  for (let denominator = 1; denominator <= maximumDenominator; denominator += 1) {
    const numerator = value * denominator;
    if (Math.abs(numerator - Math.round(numerator)) < 1e-12) {
      return denominator === 1 ? String(Math.round(numerator)) : `${Math.round(numerator)}/${denominator}`;
    }
  }
  return null;
}

export function Investigation({ title, question, note, role, children, onReset }) {
  const id = useId();
  return <section className="imb-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="imb-question">{question}</p>}
    {role && <p className={`imb-role${role.kind ? ` is-${role.kind}` : ''}`} role="status">{role.text}</p>}
    {note && <p className="imb-note">{note}</p>}
    {children}
  </section>;
}

/** A labelled control. The id is explicit so the label describes the control
 * rather than the output that echoes its value. */
export function Field({ label, value, error, hint, children }) {
  const id = useId();
  const described = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ');
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': described || undefined })
    : children;
  return <label className={`imb-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {hint && <span className="imb-caption" id={`${id}-hint`}>{hint}</span>}
    {error && <span className="imb-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation naming the field, and
 * the model keeps its last good state instead of being handed a substitute. */
export function NumberField({ label, value, onChange, min, max, step = 'any', decimals = 2, suffix, hint, disabled = false }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const problem = (() => {
    if (draft === null) return null;
    const trimmed = draft.trim();
    if (trimmed === '' || trimmed === '-' || trimmed === '−') return 'Type a number.';
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    if (Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) > 1e-9) {
      return `Use at most ${decimals} decimal place${decimals === 1 ? '' : 's'}.`;
    }
    return null;
  })();
  return <Field label={label} error={problem} value={suffix} hint={hint}>
    <input type="number" inputMode="decimal" min={min} max={max} step={step} value={shown} disabled={disabled}
      onChange={event => {
        setDraft(event.target.value);
        const parsed = Number(event.target.value);
        if (event.target.value.trim() !== '' && Number.isFinite(parsed) && parsed >= min && parsed <= max
          && Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) <= 1e-9) {
          onChange(Number(parsed.toFixed(decimals)));
        }
      }}
      onBlur={() => setDraft(null)} aria-invalid={Boolean(problem)} />
  </Field>;
}

/** A slider paired with the same value as a typed number: the keyboard route
 * and the precise route are both first class, neither is a fallback. */
export function SliderField({ label, value, onChange, min, max, step, decimals = 2, suffix, hint, disabled = false }) {
  // The two controls are one control in two forms, so they occupy one grid cell.
  // Left as siblings they pack independently and a slider ends up on a different
  // row from the number that types the same quantity.
  return <div className="imb-paired">
    <Field label={label} value={suffix ?? round(value, decimals)} hint={hint}>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={event => onChange(Number(Number(event.target.value).toFixed(decimals)))} />
    </Field>
    <NumberField label={`${label} — exact value`} value={value} onChange={onChange}
      min={min} max={max} step={step} decimals={decimals} disabled={disabled} />
  </div>;
}

export function Select({ label, value, onChange, options, hint, disabled = false }) {
  return <Field label={label} hint={hint}>
    <select value={value} disabled={disabled} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
    </select>
  </Field>;
}

/** A table whose caption sits OUTSIDE the scroll box.
 *
 * Below the narrow breakpoint every cell becomes its own labelled row, which
 * needs `display: block`; that drops the implicit table roles, so they are
 * restored explicitly here. A `<caption>` inside a blockified table collapses to
 * the width of its longest word, so the caption is a sibling paragraph that
 * labels the region instead.
 */
export function Table({ caption, headings, rows, rowClass = () => undefined, scroll = false, footnote }) {
  const id = useId();
  return <div className="imb-table">
    <p className="imb-caption" id={id}>{caption}</p>
    <div className={`imb-table-scroll${scroll ? ' imb-rows' : ''}`} role="region" aria-labelledby={id} tabIndex={0}>
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
    {footnote && <p className="imb-caption">{footnote}</p>}
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

/** A framed plot with one shared scale for everything drawn on it. */
export function Plot({
  caption, describe, width = 340, height = 200,
  padding = { left: 48, right: 14, top: 16, bottom: 34 },
  domain, range, ticks, valueTicks,
  formatTick = value => round(value, 2), formatValue = value => round(value, 3), children,
}) {
  const scaleX = value => padding.left + (width - padding.left - padding.right)
    * (value - domain[0]) / (domain[1] - domain[0]);
  const scaleY = value => height - padding.bottom - (height - padding.top - padding.bottom)
    * (value - range[0]) / (range[1] - range[0]);
  const marks = ticks ?? Array.from({ length: 5 }, (_, index) => domain[0] + (domain[1] - domain[0]) * index / 4);
  const sideMarks = valueTicks ?? [range[0], (range[0] + range[1]) / 2, range[1]];
  return <figure className="imb-plot">
    {caption && <figcaption>{caption}</figcaption>}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {marks.map(value => <g key={`x${value}`}>
        <line className="imb-grid" x1={scaleX(value)} x2={scaleX(value)} y1={padding.top} y2={height - padding.bottom} />
        <text x={scaleX(value)} y={height - padding.bottom + 18} textAnchor="middle">{formatTick(value)}</text>
      </g>)}
      {/* Clamped inside the drawn box, so the lowest value label never lands on
          the row of horizontal tick labels beneath the axis. */}
      {sideMarks.map(value => <text key={`y${value}`} x={padding.left - 6} textAnchor="end"
        y={Math.min(Math.max(scaleY(value) + 4, padding.top + 9), height - padding.bottom - 2)}>{formatValue(value)}</text>)}
      <line className="imb-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
      <line className="imb-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
      {children(scaleX, scaleY)}
    </svg>
  </figure>;
}

/** A polyline through a function on the plot's own scale. */
export function curve(scaleX, scaleY, domain, evaluate, samples = 121) {
  return Array.from({ length: samples }, (_, index) => {
    const x = domain[0] + (domain[1] - domain[0]) * index / (samples - 1);
    return `${scaleX(x).toFixed(2)},${scaleY(evaluate(x)).toFixed(2)}`;
  }).join(' ');
}

/** A polyline through recorded points. */
export function polyline(points, scaleX, scaleY) {
  return points.map(([x, y]) => `${scaleX(x).toFixed(2)},${scaleY(y).toFixed(2)}`).join(' ');
}

/** A step function through operating points that exist, with no segment drawn
 * between two thresholds the data never reaches. */
export function steps(points, scaleX, scaleY) {
  const parts = [];
  points.forEach(([x, y], index) => {
    if (index === 0) parts.push(`${scaleX(x).toFixed(2)},${scaleY(y).toFixed(2)}`);
    else {
      parts.push(`${scaleX(x).toFixed(2)},${scaleY(points[index - 1][1]).toFixed(2)}`);
      parts.push(`${scaleX(x).toFixed(2)},${scaleY(y).toFixed(2)}`);
    }
  });
  return parts.join(' ');
}

/** Nonnegative quantities adding to one total, drawn as one track.
 *
 * Every part carries its own fill pattern and its own printed value, so the
 * parts are distinguishable without colour and a part that is exactly zero says
 * so instead of becoming an invisible sliver. */
export function Stack({ caption, describe, parts, totalLabel = 'total', digits = 6, unit = '' }) {
  const total = parts.reduce((accumulated, part) => accumulated + part.value, 0);
  const extent = Math.max(total, 1e-12);
  return <div className="imb-waterfall" role="img" aria-label={describe}>
    <p className="imb-caption">{caption}</p>
    <div className="imb-waterfall-track">
      {parts.map(part => (part.value > 0
        ? <span key={part.name} className={`imb-waterfall-part ${part.className}`}
          style={{ width: `${100 * part.value / extent}%` }} />
        : null))}
    </div>
    <dl className="imb-waterfall-key">
      {parts.map(part => <div key={part.name} className="imb-waterfall-row">
        <dt><span className={`imb-swatch ${part.className}`} aria-hidden="true" />{part.name}</dt>
        <dd>{part.value === 0 ? 'exactly 0' : `${round(part.value, digits)}${unit}`}</dd>
      </div>)}
      <div className="imb-waterfall-row is-total">
        <dt>{totalLabel}</dt>
        <dd>{round(total, digits)}{unit}</dd>
      </div>
    </dl>
  </div>;
}

/** The four confusion counts as one labelled strip, with precision and recall
 * shown as defined values or as named undefined states. */
export function CountStrip({ counts, showRates = true }) {
  return <p className="imb-counts">
    <span>TP <b>{counts.tp}</b></span>
    <span>FP <b>{counts.fp}</b></span>
    <span>FN <b>{counts.fn}</b></span>
    <span>TN <b>{counts.tn}</b></span>
    <span>total <b>{counts.total}</b></span>
    {showRates && (counts.precision === null
      ? <span className="is-undefined">precision <b>undefined</b> — nothing selected, so TP + FP is 0</span>
      : <span>precision <b>{round(counts.precision, 6)}</b></span>)}
    {showRates && (counts.recall === null
      ? <span className="is-undefined">recall <b>undefined</b> — no actual positives, so TP + FN is 0</span>
      : <span>recall <b>{round(counts.recall, 6)}</b></span>)}
  </p>;
}
