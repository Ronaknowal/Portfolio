import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { cloneElement, isValidElement, useId, useState } from 'react';
import './rademacher-labs.css';
import { compareOutcome, displayDigits, displayValue } from '../../data/rademacher-models.js';



const minus = text => String(text).replace(/-/g, '−');

/** A number for reading. Small magnitudes keep their exponent rather than
 *  collapsing to a row of zeros. */
export function round(value, digits = displayDigits) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '−∞';
  if (Number.isInteger(value)) return minus(String(value));
  if (value !== 0 && Math.abs(value) < 1e-4) {
    const [mantissa, exponent] = value.toExponential(3).split('e');
    const marks = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    const superscript = [...exponent.replace('+', '')].map(character => marks[character] ?? character).join('');
    return `${minus(mantissa)} × 10${superscript}`;
  }
  const text = value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
  return text === '-0' ? '0' : minus(text);
}

/** Every decimal place kept, for a column whose rows must line up. */
export const fixed = (value, digits = displayDigits) =>
  (Number.isFinite(value) ? minus(value.toFixed(digits)) : round(value));

/** An exact fraction reads better than six decimals when the value is exact,
 *  and most of this lesson's small quantities are. */
export function asFraction(value, maximumDenominator = 64) {
  if (value === null || !Number.isFinite(value)) return null;
  for (let denominator = 1; denominator <= maximumDenominator; denominator += 1) {
    const numerator = value * denominator;
    if (Math.abs(numerator - Math.round(numerator)) < 1e-12) {
      return denominator === 1 ? minus(String(Math.round(numerator))) : `${minus(String(Math.round(numerator)))}/${denominator}`;
    }
  }
  return null;
}

/** A value with its exact fraction beside it when one exists. */
export function exactly(value, digits = displayDigits) {
  const fraction = asFraction(value);
  return fraction && fraction !== round(value, digits) ? `${round(value, digits)} (exactly ${fraction})` : round(value, digits);
}

export const signGlyph = sign => (sign > 0 ? '+1' : '−1');

export function Investigation({ title, question, note, role, children, onReset }) {
  const id = useId();
  return <section className="rad-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="rad-question">{question}</p>}
    {role && <p className={`rad-role${role.kind ? ` is-${role.kind}` : ''}`} role="status">{role.text}</p>}
    {note && <p className="rad-note">{note}</p>}
    {children}
  </section>;
}

export function Field({ label, value, error, hint, children, hideLabel = false }) {
  const id = useId();
  const described = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ');
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': described || undefined })
    : children;
  /* `hideLabel` removes the label from the LAYOUT, never from the accessibility
     tree. It is used inside the editable matrix, where the row header and the
     column header already name each cell on screen. */
  return <label className={`rad-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span className={hideLabel ? 'rad-visually-hidden' : undefined}>
      {label}{value !== undefined && <output htmlFor={id}>{value}</output>}
    </span>
    {control}
    {hint && <span className="rad-caption" id={`${id}-hint`}>{hint}</span>}
    {error && <span className="rad-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/**
 * A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation naming the field, and
 * the model keeps its last good state rather than being handed a substitute.
 *
 * Typed numbers, not sliders: rho reaches .1, delta reaches .01, and a
 * coefficient has to be able to sit exactly on the boundary of its ball.
 */
export function NumberField({
  label, value, onChange, min, max, step = 'any', decimals = 3, suffix, hint, disabled = false, placeholder,
  hideLabel = false,
}) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const scale = 10 ** decimals;
  const problem = (() => {
    if (draft === null) return null;
    const trimmed = draft.trim();
    if (trimmed === '' || trimmed === '-' || trimmed === '−') return 'Type a number.';
    const parsed = Number(trimmed.replace('−', '-'));
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    if (Math.abs(parsed * scale - Math.round(parsed * scale)) > 1e-9) {
      return `Use at most ${decimals} decimal place${decimals === 1 ? '' : 's'}.`;
    }
    return null;
  })();
  return <Field label={label} error={problem} value={suffix} hint={hint} hideLabel={hideLabel}>
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

/** A row of +1 / −1 toggles. Each carries its own printed sign as well as
 *  its state, so the pattern is readable without colour and reachable from the
 *  keyboard without a drag. */
export function SignRow({ label, signs, onChange, disabled = false, describe }) {
  const name = useId();
  return <div className="rad-sign-row">
    <span className="rad-sign-label">{label}</span>
    <span className="rad-sign-options" role="group" aria-label={describe ?? label}>
      {signs.map((sign, index) => (
        <button key={`${name}-${index}`} type="button" disabled={disabled}
          className={`rad-sign-button${sign > 0 ? ' is-positive' : ' is-negative'}`}
          aria-pressed={sign > 0}
          onClick={() => onChange(signs.map((value, position) => (position === index ? -value : value)))}>
          <span className="rad-sign-index">{index + 1}</span>
          <span className="rad-sign-value">{signGlyph(sign)}</span>
        </button>
      ))}
    </span>
  </div>;
}

/**
 * A table whose caption sits OUTSIDE the scroll box.
 *
 * Below the narrow breakpoint every cell becomes its own labelled row, which
 * needs `display: block`; that drops the implicit table roles, so they are
 * restored explicitly. A `<caption>` inside a blockified table collapses to the
 * width of its longest word, so the caption is a sibling paragraph instead.
 */
export function Table({ caption, headings, rows, rowClass = () => undefined, cellClass = () => undefined, footnote }) {
  const id = useId();
  return <div className="rad-table">
    <p className="rad-caption" id={id}>{caption}</p>
    <div className="rad-table-scroll" role="region" aria-labelledby={id} tabIndex={0}>
      <table role="table">
        <thead role="rowgroup"><tr role="row">
          {headings.map(heading => <th key={heading} role="columnheader" scope="col">{heading}</th>)}
        </tr></thead>
        <tbody role="rowgroup">{rows.map((row, index) => <tr key={index} role="row" className={rowClass(index)}>
          {row.map((cell, column) => (column === 0
            ? <th key={column} role="rowheader" scope="row" data-label={headings[column]}>{cell}</th>
            : <td key={column} role="cell" data-label={headings[column]} className={cellClass(index, column)}>{cell}</td>))}
        </tr>)}</tbody>
      </table>
    </div>
    {footnote && <p className="rad-caption">{footnote}</p>}
  </div>;
}


export const useInvestigation = useLiveInvestigation;


export const useStagedInvestigation = useLiveStages;





export function LiveResult({ state, calculateInputs, blocked, describe }) {
  const problem = useLiveResult(state, calculateInputs, blocked);
  return <div data-live-exploration="result">
    {problem ? <p role="status">{problem} The plots retain the last valid calculation; correct the inputs to update them.</p>
      : <p role="status">Live calculation for the current controls. {describe}</p>}
    <button type="button" disabled={Boolean(problem) || !state.result} onClick={state.snapshot}>Use current values as comparison baseline</button>
  </div>;
}

/**
 * The one place a lab turns two numbers into the words "unchanged",
 * "increased" or "decreased".
 *
 * It routes through `compareOutcome`, which compares the DISPLAYED values, so
 * a verdict saying unchanged and two identical printed numbers are the same
 * statement. When the raw doubles differ below the displayed precision the
 * sentence says that too, rather than calling the difference zero.
 */
export function movementAnswer(before, after, { quantity, digits = displayDigits } = {}) {
  const comparison = compareOutcome(before, after, digits);
  return {
    outcome: comparison.outcome,
    value: comparison.displayedAfter,
    comparison,
    explain: `${quantity ?? 'The quantity'} was ${fixed(comparison.displayedBefore, digits)} and is now `
      + `${fixed(comparison.displayedAfter, digits)}.`
      + (comparison.belowDisplayPrecision
        ? ` The two differ by ${round(comparison.delta, 12)}, which is below the ${digits} decimals shown here;`
          + ' this is a difference too small to display, not an exact null.'
        : ''),
  };
}

export { compareOutcome, displayValue };

/* ============================================================ drawing parts */

/** A figure wrapper. The caption is a real <figcaption> above the drawing, and
 *  the text equivalent is a sibling paragraph, so neither depends on being
 *  inside the SVG. */
export function Figure({ caption, children, describe, footnote, className }) {
  return <figure className={`rad-figure${className ? ` ${className}` : ''}`}>
    {caption && <figcaption>{caption}</figcaption>}
    {children}
    {describe && <p className="rad-caption">{describe}</p>}
    {footnote && <p className="rad-caption rad-footnote">{footnote}</p>}
  </figure>;
}

/** An SVG whose viewBox is in the same units as its rendered width, so text
 *  inside it is sized in real pixels rather than scaled by an accidental
 *  ratio. */
/**
 * Every SVG this lesson draws carries `rad-drawing`, and the stylesheet targets
 * that class rather than a bare descendant `svg`.
 *
 * The rule used to be `.rad-lesson svg { width: 100%; height: auto; ... }`.
 * Scoped at the lesson root it also matched KaTeX's own inline SVGs — the ones
 * that draw radical signs and stretchy accents — and `height: auto` collapsed
 * them to under a pixel. All 24 radicals on the page rendered at 0.05–0.72px,
 * so √(ln(2K/δ)/(2n)) read as ln(2K/δ)/(2n): the radius squared rather than the
 * radius. On this topic a collapsed accent is its own correctness defect, since
 * R̂ losing its hat turns the empirical complexity into the population quantity
 * the lesson exists to distinguish it from.
 *
 * The class is applied HERE rather than left to each caller, so a new figure
 * cannot forget it.
 */
export function Drawing({ width, height, title, describe, children, className }) {
  const id = useId();
  return <svg className={`rad-drawing${className ? ` ${className}` : ''}`} viewBox={`0 0 ${width} ${height}`} role="img"
    aria-labelledby={`${id}-title`} aria-describedby={`${id}-desc`} style={{ maxWidth: `${width}px` }}>
    <title id={`${id}-title`}>{title}</title>
    <desc id={`${id}-desc`}>{describe}</desc>
    {children}
  </svg>;
}

/** A labelled bar with its exact value printed beside it. */
export function BarRow({ label, value, share, tone = 'neutral', suffix }) {
  return <div className={`rad-bar-row is-${tone}`}>
    <span className="rad-bar-label">{label}</span>
    <span className="rad-bar-track">
      <span className="rad-bar-fill" style={{ width: `${Math.max(0, Math.min(100, 100 * share))}%` }} />
    </span>
    <span className="rad-bar-value">{value}{suffix}</span>
  </div>;
}

/** A short definition list of computed quantities. Used where a table would be
 *  heavier than the three numbers it holds. */
export function Readout({ items, className }) {
  return <dl className={`rad-readout${className ? ` ${className}` : ''}`}>
    {items.map(([term, value, note]) => <div key={term}>
      <dt>{term}</dt>
      <dd>{value}{note && <span className="rad-caption"> {note}</span>}</dd>
    </div>)}
  </dl>;
}
