import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { cloneElement, isValidElement, useId, useState } from 'react';
import './formulation-labs.css';
import { gradingAllowance, linearScale } from '../../data/formulation-models.js';



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


export const fixed = (value, digits = 6) =>
  (Number.isFinite(value) ? sign(value.toFixed(digits)) : round(value));

/** A short form for an INPUT the learner set. Deliberately never six decimals.
 *
 *  This was written as `Number.isInteger(value) ? sign(String(value)) :
 *  sign(String(value))` -- two byte-identical branches, the exact pattern the
 *  source-hygiene scan exists to catch, sitting unflagged in a file that scan
 *  reads. The scan's regex matched only quoted string literals, so an
 *  expression-valued pair was invisible to it, and its falsification case
 *  injected a string-literal instance and passed. Both the conditional and the
 *  scan's domain are fixed. */
export const asInput = value => sign(String(value));


export const probabilityText = value => sign(value.toFixed(4));

export function Investigation({ title, question, note, role, children, onReset }) {
  const id = useId();
  return <section className="formulation-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="formulation-question">{question}</p>}
    {role && <p className={`formulation-role${role.kind ? ` is-${role.kind}` : ''}`} role="status">{role.text}</p>}
    {note && <p className="formulation-note">{note}</p>}
    {children}
  </section>;
}

export function Field({ label, value, error, hint, children }) {
  const id = useId();
  const described = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ');
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': described || undefined })
    : children;
  return <label className={`formulation-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {hint && <span className="formulation-caption" id={`${id}-hint`}>{hint}</span>}
    {error && <span className="formulation-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/**
 * A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation naming the field, and
 * the model keeps its last good state rather than being handed a substitute.
 *
 * Typed rather than dragged: every quantity this lesson edits is an integer
 * time, an integer value or a capacity, all of which a keyboard reaches
 * directly and a slider only approximates.
 */
export function NumberField({
  label, value, onChange, min, max, step = 1, decimals = 0, suffix, hint, disabled = false, placeholder,
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
      return decimals === 0
        ? 'Use a whole number.'
        : `Use at most ${decimals} decimal place${decimals === 1 ? '' : 's'}.`;
    }
    return null;
  })();
  return <Field label={label} error={problem} value={suffix} hint={hint}>
    <input type="number" inputMode="numeric" min={min} max={max} step={step} value={shown} disabled={disabled}
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

/**
 * A table whose caption sits OUTSIDE the scroll box.
 *
 * Below the narrow breakpoint every cell becomes its own labelled row, which
 * needs `display: block`; that drops the implicit table roles, so they are
 * restored explicitly. A `<caption>` inside a blockified table collapses to the
 * width of its longest word, so the caption is a sibling paragraph.
 */
export function Table({ caption, headings, rows, rowClass = () => undefined, footnote }) {
  const id = useId();
  return <div className="formulation-table">
    <p className="formulation-caption" id={id}>{caption}</p>
    <div className="formulation-table-scroll" role="region" aria-labelledby={id} tabIndex={0}>
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
    {footnote && <p className="formulation-caption">{footnote}</p>}
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

/* ================================================== the only SVG on this page
 *
 * Every drawing goes through here, so the layout class the stylesheet is scoped
 * to cannot be forgotten by a new figure. `kind` selects the aspect-ratio and
 * maximum-width behaviour; it does not change the class the layout rule
 * matches, which is always `form-svg`.
 */
export function Drawing({ kind = 'plot', width, height, caption, describe, children, maxWidth }) {
  const titleId = useId();
  const descriptionId = useId();
  return <svg className={`form-svg is-${kind}`} viewBox={`0 0 ${width} ${height}`} role="img"
    aria-labelledby={titleId} aria-describedby={descriptionId}
    style={{ maxWidth: `${maxWidth ?? width}px` }}>
    <title id={titleId}>{caption}</title>
    <desc id={descriptionId}>{describe}</desc>
    {children}
  </svg>;
}

/** A drawing with its caption underneath, in reflowing markup. Prose never goes
 *  inside the SVG: a fifty-character sentence in a 290-unit viewBox runs off
 *  the edge and scales with the figure instead of with the reader's text. */
export function Figure({ children, caption, className = '' }) {
  return <figure className={`formulation-figure-inline ${className}`.trim()}>
    {children}
    {caption && <p className="formulation-caption">{caption}</p>}
  </figure>;
}

/** A legend row. Every entry carries a shape or a pattern as well as a colour,
 *  because these rows are told apart by more than hue. */
export function Legend({ entries }) {
  return <div className="formulation-legend">
    {entries.map(([kind, text]) => <span key={kind}>
      <i className={`formulation-swatch is-${kind}`} aria-hidden="true" />{text}
    </span>)}
  </div>;
}

/** A compact strip of labelled readouts. */
export function Readouts({ cells }) {
  return <div className="formulation-strip">
    {cells.map(([label, value, used]) => (
      <span key={label} className={`formulation-strip-cell${used ? ' is-used' : ''}`}>{label}<b>{value}</b></span>
    ))}
  </div>;
}

export { linearScale };
