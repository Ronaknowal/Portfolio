import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { cloneElement, isValidElement, useId, useState } from 'react';
import './calibration-labs.css';
import { quantityKinds } from '../../data/calibration-models.js';



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

/** Every decimal place kept, for a column whose rows must line up. */
export const fixed = (value, digits = 6) => (Number.isFinite(value) ? sign(value.toFixed(digits)) : round(value));

/** A small exact fraction reads better than six decimals when it is exact. */
export function asFraction(value, maximumDenominator = 40) {
  if (value === null || !Number.isFinite(value)) return null;
  for (let denominator = 1; denominator <= maximumDenominator; denominator += 1) {
    const numerator = value * denominator;
    if (Math.abs(numerator - Math.round(numerator)) < 1e-12) {
      return denominator === 1 ? String(Math.round(numerator)) : `${Math.round(numerator)}/${denominator}`;
    }
  }
  return null;
}

/** A quantity with no value says so, and says which denominator is empty. */
export function Undefined({ because }) {
  return <span className="cal-undefined">no value <span className="cal-caption">({because})</span></span>;
}

export const ratio = (value, because, digits = 6) =>
  (value === null || value === undefined ? <Undefined because={because} /> : round(value, digits));

/**
 * Which of the three kinds of number this is.
 *
 * Printed, not implied. A coverage count is never shown on this page without
 * its denominator and this tag, because "90%" lifted out of a table and carried
 * away as a guarantee is the failure the whole lesson exists to prevent.
 */
export function KindTag({ kind, extra }) {
  const entry = quantityKinds[kind];
  if (!entry) return null;
  return <span className={`cal-kind is-${entry.key}`}>
    <span className="cal-kind-label">{entry.label}</span>
    <span className="cal-caption">{extra ?? entry.note}</span>
  </span>;
}

export function Investigation({ title, question, note, role, children, onReset }) {
  const id = useId();
  return <section className="cal-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="cal-question">{question}</p>}
    {role && <p className={`cal-role${role.kind ? ` is-${role.kind}` : ''}`} role="status">{role.text}</p>}
    {note && <p className="cal-note">{note}</p>}
    {children}
  </section>;
}

export function Field({ label, value, error, hint, children }) {
  const id = useId();
  const described = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ');
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': described || undefined })
    : children;
  return <label className={`cal-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {hint && <span className="cal-caption" id={`${id}-hint`}>{hint}</span>}
    {error && <span className="cal-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/**
 * A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation naming the field, and
 * the model keeps its last good state rather than being handed a substitute.
 *
 * Typed, not a slider. Alpha has to reach .01 and a forecast has to reach an
 * exact .5 boundary; a coarse slider would make the lesson's own fixtures
 * unreachable in its own labs.
 */
export function NumberField({
  label, value, onChange, min, max, step = 'any', decimals = 3, suffix, hint,
  disabled = false, placeholder,
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

/** A binary outcome as two radio buttons, so 0 is a chosen value and not a default. */
export function OutcomeChoice({ label, value, onChange, disabled = false, labels = ['0 — did not happen', '1 — happened'] }) {
  const name = useId();
  return <div className="cal-outcome-choice">
    <span className="cal-outcome-label">{label}</span>
    <span className="cal-outcome-options">
      {[0, 1].map(state => <label key={state}>
        <input type="radio" name={name} value={state} checked={value === state} disabled={disabled}
          onChange={() => onChange(state)} />
        <span>{labels[state]}</span>
      </label>)}
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
export function Table({ caption, headings, rows, rowClass = () => undefined, footnote, kind, className }) {
  const id = useId();
  return <div className={`cal-table${className ? ` ${className}` : ''}`}>
    <p className="cal-caption" id={id}>{caption}</p>
    {kind && <KindTag kind={kind} />}
    <div className="cal-table-scroll" role="region" aria-labelledby={id} tabIndex={0}>
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
    {footnote && <p className="cal-caption">{footnote}</p>}
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

/* ============================================================ drawn figures */

/**
 * The square plotting frame the reliability diagrams share.
 *
 * Every coordinate comes from `calibration-models.js`, so the verifier asserts
 * the numbers the browser draws. The diagonal, the ticks and the axis labels
 * are geometry; the legend and interpretation are reflowing HTML beside the
 * drawing, not text crammed into the viewBox.
 */
export function PlotFrame({
  geometry, xLabel, yLabel, describe, caption, children, titleText, axes = 'both',
}) {
  const titleId = useId();
  const descriptionId = useId();
  const { box } = geometry;
  return <figure className="cal-figure">
    {caption && <figcaption>{caption}</figcaption>}
    <svg className="cal-plot" viewBox={`0 0 ${box.width} ${box.height}`} role="img"
      aria-labelledby={titleId} aria-describedby={descriptionId} style={{ maxWidth: `${box.width}px` }}>
      <title id={titleId}>{titleText ?? caption}</title>
      <desc id={descriptionId}>{describe}</desc>
      {/* A one-dimensional rail has no quantity on its vertical axis, so it gets
          no vertical line: an axis drawn where nothing is measured invites a
          reading that is not there. */}
      {axes === 'both' && <line className="cal-axis" x1={box.left} y1={box.top}
        x2={box.left} y2={box.height - box.bottom} />}
      <line className="cal-axis" x1={box.left} y1={box.height - box.bottom}
        x2={box.width - box.right} y2={box.height - box.bottom} />
      {geometry.ticks?.map(tick => <g key={tick.value}>
        <line className="cal-tick" x1={tick.x} y1={box.height - box.bottom}
          x2={tick.x} y2={box.height - box.bottom + 4} />
        <text className="cal-small" x={tick.x} y={box.height - box.bottom + 14} textAnchor="middle">
          {tick.value}
        </text>
        <line className="cal-tick" x1={box.left - 4} y1={tick.y} x2={box.left} y2={tick.y} />
        <text className="cal-small" x={box.left - 7} y={tick.y + 3} textAnchor="end">{tick.value}</text>
      </g>)}
      {children}
      <text className="cal-small cal-axis-name" x={(box.left + box.width - box.right) / 2}
        y={box.height - 6} textAnchor="middle">{xLabel}</text>
      {/* Only when there is something to say. A one-dimensional rail passes an
          empty vertical label, and an empty <text> element still occupies a
          degenerate box that a layout check has to reason about. */}
      {yLabel ? <text className="cal-small cal-axis-name" x={12} y={(box.top + box.height - box.bottom) / 2}
        textAnchor="middle" transform={`rotate(-90 12 ${(box.top + box.height - box.bottom) / 2})`}>{yLabel}</text>
        : null}
    </svg>
  </figure>;
}

/** A row of proportional bars with their exact values printed beside them. */
export function BarRows({ caption, rows, highlight = null, digits = 6, describe, className, unit }) {
  return <div className={`cal-bars${className ? ` ${className}` : ''}`} role="img"
    aria-label={describe ?? caption}>
    {caption && <p className="cal-caption">{caption}</p>}
    <dl>
      {rows.map((row, index) => <div key={row.name}
        className={`cal-bar-row${highlight === index ? ' is-leading' : ''}`}>
        <dt>{row.name}</dt>
        <dd>
          <span className="cal-bar-track">
            <span className="cal-bar-fill" style={{ width: `${Math.max(0, Math.min(100, 100 * row.share))}%` }} />
          </span>
          <span className="cal-bar-value">
            {row.value === 0 ? 'exactly 0' : round(row.value, digits)}{unit ? ` ${unit}` : ''}
          </span>
        </dd>
      </div>)}
    </dl>
  </div>;
}

/** A labelled pill, for a set membership or a verdict word. */
export function Pill({ children, tone = 'neutral', mark }) {
  return <span className={`cal-pill is-${tone}`}>
    {mark && <span className="cal-pill-mark" aria-hidden="true">{mark}</span>}{children}
  </span>;
}
