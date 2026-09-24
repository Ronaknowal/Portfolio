import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { cloneElement, isValidElement, useId, useState } from 'react';
import './automl-labs.css';



/** Every printed number uses a typographic minus sign, matching the prose. */
const sign = text => text.replace('-', '−');
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
/** A sign is information about a direction, so an exact zero carries none. */
export const signed = (value, digits = 6) => {
  if (value === 0) return '0';
  return value > 0 ? `+${round(value, digits)}` : round(value, digits);
};
/** An exact zero is said, not shaded. A tiny nonzero value is never printed as 0. */
export const exactly = (value, digits = 6) => (value === 0 ? 'exactly 0' : round(value, digits));
/** A fraction the prose can quote beside its decimal. */
export const ratio = (numerator, denominator) => `${numerator}/${denominator}`;

export function Investigation({ title, question, note, evidence, children, onReset }) {
  const id = useId();
  return <section className="am-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="am-question">{question}</p>}
    {evidence && <p className="am-evidence">{evidence}</p>}
    {note && <p className="am-note">{note}</p>}
    {children}
  </section>;
}

/** A labelled control. The id is explicit so the label describes the control
 * rather than the output that echoes its value. */
export function Field({ label, value, error, children }) {
  const id = useId();
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': error ? `${id}-error` : undefined })
    : children;
  return <label className={`am-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {error && <span className="am-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation, and the model keeps
 * its last good state instead of being handed a silent substitute. */
export function NumberField({
  label, value, onChange, min, max, step = 'any', decimals = 2, suffix, disabled = false,
  /** Inside a table the column header already says what the cell is, so the
   * visible label is redundant; it stays as the accessible name. */
  labelHidden = false,
}) {
  const [draft, setDraft] = useState(null);
  // Display at the control's own resolution. A field that accepts six decimals
  // must not show a value with seventeen: it overflows the box, and a learner
  // who retypes what they can see is told it has too many decimals. The model
  // keeps the exact value until an edit actually replaces it.
  const shown = draft === null ? String(Number(value.toFixed(decimals))) : draft;
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
  const control = <input type="number" inputMode="decimal" min={min} max={max} step={step} value={shown}
    disabled={disabled} aria-label={labelHidden ? label : undefined}
    onChange={event => {
      setDraft(event.target.value);
      const parsed = Number(event.target.value);
      if (event.target.value.trim() !== '' && Number.isFinite(parsed) && parsed >= min && parsed <= max
        && Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) <= 1e-9) {
        onChange(Number(parsed.toFixed(decimals)));
      }
    }}
    onBlur={() => setDraft(null)} aria-invalid={Boolean(problem)} />;
  if (labelHidden) {
    return <span className={`am-field is-bare${problem ? ' is-invalid' : ''}`}>
      {control}
      {problem && <span className="am-field-error">{problem}</span>}
    </span>;
  }
  return <Field label={label} error={problem} value={suffix}>
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

/** A labelled select over named options. */
export function SelectField({ label, value, options, onChange, disabled = false }) {
  return <Field label={label}>
    <select value={value} disabled={disabled} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
    </select>
  </Field>;
}

/** A text field that only publishes a value its own validator accepts, and
 * shows the validator's own message otherwise. */
export function TextField({ label, value, onChange, validate, placeholder, disabled = false }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? value : draft;
  const problem = draft === null ? null : validate(draft);
  return <Field label={label} error={problem}>
    <input type="text" value={shown} placeholder={placeholder} disabled={disabled}
      onChange={event => {
        setDraft(event.target.value);
        if (!validate(event.target.value)) onChange(event.target.value);
      }}
      onBlur={() => setDraft(null)} aria-invalid={Boolean(problem)} />
  </Field>;
}

export function Table({ caption, headings, rows, rowClass = () => undefined, scroll = false }) {
  // The caption lives OUTSIDE the scroll box. A `<caption>` sizes itself to the
  // table, not to the scroller, so a wide table pushes its own caption out of
  // view at desktop width and the reader has to scroll sideways to finish the
  // sentence — which happened to the one caption that disambiguates F3's two
  // numeric columns.
  //
  // Below the narrow breakpoint every cell becomes its own labelled row, which
  // needs `display: block`; that drops the implicit table roles, so they are
  // restored explicitly here rather than left to the layout.
  return <figure className="am-table">
    <figcaption className="am-table-caption">{caption}</figcaption>
    <div className={`am-table-scroll${scroll ? ' am-rows' : ''}`} role="region" aria-label={caption} tabIndex={0}>
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
  </figure>;
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

/** Earlier attempts, kept as history and never as the answer to edited inputs. */
export function Attempts({ entries, label = 'Saved comparison snapshots' }) {
  if (!entries.length) return null;
  return <details><summary>{label} ({entries.length})</summary><ol>{entries.map((entry, index) => <li key={index}><code>{entry.key}</code></li>)}</ol></details>;
}

/** A framed plot with one shared scale for everything drawn on it. */
export function Plot({
  caption, describe, width = 340, height = 200,
  padding = { left: 48, right: 16, top: 16, bottom: 36 },
  domain, range, ticks, valueTicks,
  formatTick = value => round(value, 2), formatValue = value => round(value, 3), children,
}) {
  const scaleX = value => padding.left + (width - padding.left - padding.right)
    * (value - domain[0]) / (domain[1] - domain[0]);
  const scaleY = value => height - padding.bottom - (height - padding.top - padding.bottom)
    * (value - range[0]) / (range[1] - range[0]);
  const marks = ticks ?? Array.from({ length: 5 }, (_, index) => domain[0] + (domain[1] - domain[0]) * index / 4);
  const sideMarks = valueTicks ?? [range[0], (range[0] + range[1]) / 2, range[1]];
  return <figure className="am-plot">
    {caption && <figcaption>{caption}</figcaption>}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {marks.map(value => <g key={`x${value}`}>
        <line className="am-grid" x1={scaleX(value)} x2={scaleX(value)} y1={padding.top} y2={height - padding.bottom} />
        <text x={scaleX(value)} y={height - padding.bottom + 18} textAnchor="middle">{formatTick(value)}</text>
      </g>)}
      {/* Clamped inside the drawn box, so the lowest value label never lands on
          the row of horizontal tick labels beneath the axis. */}
      {sideMarks.map(value => <text key={`y${value}`} x={padding.left - 6} textAnchor="end"
        y={Math.min(Math.max(scaleY(value) + 4, padding.top + 9), height - padding.bottom - 2)}>{formatValue(value)}</text>)}
      <line className="am-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
      <line className="am-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
      {children(scaleX, scaleY)}
    </svg>
  </figure>;
}

/** A polyline through recorded points on the plot's own scale. */
export function polyline(points, scaleX, scaleY) {
  return points.map(([x, y]) => `${scaleX(x).toFixed(2)},${scaleY(y).toFixed(2)}`).join(' ');
}

/** A legend that names a line pattern or shape as well as a colour. */
export function Legend({ items }) {
  return <ul className="am-legend">
    {items.map(item => <li key={item.label}>
      <svg viewBox="0 0 34 10" aria-hidden="true">
        {item.shape === 'dot' ? <circle className={item.className} cx="17" cy="5" r="4" />
          : item.shape === 'square' ? <rect className={item.className} x="11" y="1" width="12" height="8" />
            : <line className={item.className} x1="1" x2="33" y1="5" y2="5" />}
      </svg>
      <span>{item.label}</span>
    </li>)}
  </ul>;
}

/** A named stage inside an investigation, so a multi-part activity reads as a
 * sequence in ordinary HTML rather than as one crowded drawing. */
export function Stage({ title, children }) {
  return <div className="am-stage">{title && <h4>{title}</h4>}{children}</div>;
}
