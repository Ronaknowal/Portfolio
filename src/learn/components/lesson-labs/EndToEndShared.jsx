import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { createContext, cloneElement, isValidElement, useContext, useId, useMemo, useState } from 'react';
import './endtoend-labs.css';
import { ROLE_LABELS, SCORE_ROLES, fixed } from '../../data/endtoend-models.js';



const minus = text => String(text).replace('-', '−');

export { fixed };

/** A short form for an INPUT the learner set. Deliberately never six decimals,
 *  so the browser verifier's six-decimal pin can only match a computed value. */
export const asInput = value => (Number.isInteger(value) ? String(value) : minus(String(value)));

/* ================================================================ score roles */

/**
 * One number, with the role that says what it is evidence about.
 *
 * `record` is a score record from the model layer: `{ metric, value, role }`.
 * The role is rendered as text, not as a colour, and it is carried on the
 * element as a data attribute so the browser verifier can assert that every
 * printed score has one.
 */
export function Score({ record, digits = 6, format, quantityKey }) {
  if (!record || !SCORE_ROLES.includes(record.role)) {
    throw new RangeError('Score needs a record carrying one of the four score roles');
  }
  const text = format ? format(record.value) : fixed(record.value, digits);
  return <span className={`ete-score is-${record.role.replace('-', '')}`} data-role={record.role}>
    <span className="ete-score-value" data-graded-quantity={quantityKey}>{text}</span>
    <span className="ete-score-role">{ROLE_LABELS[record.role]}</span>
  </span>;
}

/**
 * A number that deliberately carries no role, with the reason attached.
 *
 * The page's invariant is that every score of THIS STUDY'S MODELS is badged with
 * one of the four roles. A very small number of printed quantities are not such
 * a score -- a difference BETWEEN two roles, which is in neither. Rendering them
 * bare would make the invariant unenforceable, because a sweep could not tell a
 * deliberate exception from an omission. So the exception is declared on the
 * element, carries its reason, and the sweep requires the reason to be there.
 */
export function NonScore({ children, because }) {
  if (typeof because !== 'string' || because.trim().length < 12) {
    throw new RangeError('a number printed without a role must say why, in a sentence');
  }
  return <span className="ete-nonscore" data-role-exempt={because}>{children}</span>;
}

/** A count with its denominator kept attached. "Three errors" means different
 *  things out of five and out of five hundred, so the denominator travels. */
export function Count({ part, whole, role, noun = '', quantityKey }) {
  if (!SCORE_ROLES.includes(role)) throw new RangeError(`Count needs a score role, not "${role}"`);
  return <span className="ete-score" data-role={role}>
    <span className="ete-score-value">
      <span data-graded-quantity={quantityKey}>{part}</span> of {whole}{noun ? ` ${noun}` : ''}
    </span>
    <span className="ete-score-role">{ROLE_LABELS[role]}</span>
  </span>;
}

/* ============================================================== the held-out gate */

const HeldOutContext = createContext(null);


export function HeldOutProvider({ children }) {
  const [earned, setEarned] = useState(false);
  const [record, setRecord] = useState(null);
  const value = useMemo(() => ({
    earned,
    record,
    earn: committed => { setRecord(committed ?? null); setEarned(true); },
    release: () => { setEarned(false); setRecord(null); },
  }), [earned, record]);
  return <HeldOutContext.Provider value={value}>{children}</HeldOutContext.Provider>;
}

export function useHeldOut() {
  const value = useContext(HeldOutContext);
  if (!value) throw new Error('a held-out quantity was reached outside the gate that guards it');
  return value;
}

/**
 * Renders its children only once the held-out report has been earned.
 *
 * The children are not rendered and hidden; they are not rendered. That
 * distinction is the whole guard: a CSS rule that hides an element leaves its
 * text in the document, in the accessibility tree and in a copy-paste, and the
 * claim this lesson makes is that the number is not readable.
 */
export function HeldOutOnly({ children, placeholder }) {
  const { earned } = useHeldOut();
  if (earned) return children;
  return <p className="ete-sealed" role="note">
    <span className="ete-sealed-mark" aria-hidden="true">◼</span>
    {placeholder ?? 'This part of the page reports the held-out result. It stays closed until the decision it '
      + 'reports on has been frozen in the investigation above.'}
  </p>;
}

/* ================================================================== controls */

/**
 * `constructedFixture` declares that the numbers inside this investigation are
 * not scores of this study's models at all.
 *
 * The four roles say what a number is evidence about *for this study*. The
 * acceptance queue's ten cases are invented to expose coverage and cost
 * arithmetic; badging one of them "validation" would claim it came from the 36
 * wine development rows, which is exactly the mislabelling this lesson teaches
 * a learner to catch. So the subtree is declared out of the role system, with
 * its reason on the element, and the role sweep reads the declaration rather
 * than being quietly silent about a region it cannot classify.
 */
export function Investigation({
  title, question, note, role, children, onReset, constructedFixture, investigationKey,
}) {
  const id = useId();
  if (constructedFixture !== undefined
    && (typeof constructedFixture !== 'string' || constructedFixture.trim().length < 12)) {
    throw new RangeError('a constructed-fixture declaration must say what the numbers are, in a sentence');
  }
  return <section className="ete-investigation" aria-labelledby={id}
    data-investigation={investigationKey} data-constructed-fixture={constructedFixture}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="ete-question">{question}</p>}
    {role && <p className={`ete-role${role.kind ? ` is-${role.kind}` : ''}`} role="status">{role.text}</p>}
    {note && <p className="ete-note">{note}</p>}
    {children}
  </section>;
}

export function Field({ label, value, error, hint, children }) {
  const id = useId();
  const described = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ');
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': described || undefined })
    : children;
  return <label className={`ete-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {hint && <span className="ete-caption" id={`${id}-hint`}>{hint}</span>}
    {error && <span className="ete-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/**
 * A number field that publishes only a valid value, in hundredths.
 *
 * The model keeps the cutoff and the confidence scores as integer hundredths so
 * the browser and the offline verifier compare bit-identical operands. An
 * unparsable or out-of-range draft stays on screen with an explanation naming
 * the field, and the model keeps its last good state rather than being handed a
 * substitute.
 */
export function HundredthsField({
  label, hundredths, onChange, min, max, step = 0.01, suffix, hint, disabled = false,
}) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(hundredths / 100) : draft;
  const problem = (() => {
    if (draft === null) return null;
    const trimmed = draft.trim();
    if (trimmed === '' || trimmed === '-' || trimmed === '−') return 'Type a number.';
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min / 100 || parsed > max / 100) return `Keep it between ${min / 100} and ${max / 100}.`;
    if (Math.abs(parsed * 100 - Math.round(parsed * 100)) > 1e-9) return 'Use at most two decimal places.';
    return null;
  })();
  const publish = text => {
    const parsed = Number(text);
    if (text.trim() === '' || !Number.isFinite(parsed)) return;
    const asHundredths = Math.round(parsed * 100);
    if (asHundredths < min || asHundredths > max) return;
    if (Math.abs(parsed * 100 - asHundredths) > 1e-9) return;
    onChange(asHundredths);
  };
  return <Field label={label} error={problem} value={suffix} hint={hint}>
    <input type="number" inputMode="decimal" min={min / 100} max={max / 100} step={step} value={shown}
      disabled={disabled} aria-invalid={Boolean(problem)}
      onChange={event => { setDraft(event.target.value); publish(event.target.value); }}
      onBlur={() => setDraft(null)} />
  </Field>;
}

/** A plain number field for a cost, which is a whole quantity rather than a
 *  two-decimal one. */
export function CostField({ label, value, onChange, min, max, hint, disabled = false }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const problem = (() => {
    if (draft === null) return null;
    const trimmed = draft.trim();
    if (trimmed === '') return 'Type a number.';
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    return null;
  })();
  return <Field label={label} error={problem} hint={hint}>
    <input type="number" inputMode="decimal" min={min} max={max} step="0.5" value={shown} disabled={disabled}
      aria-invalid={Boolean(problem)}
      onChange={event => {
        setDraft(event.target.value);
        const parsed = Number(event.target.value);
        if (event.target.value.trim() !== '' && Number.isFinite(parsed) && parsed >= min && parsed <= max) {
          onChange(parsed);
        }
      }}
      onBlur={() => setDraft(null)} />
  </Field>;
}

export function Select({ label, value, onChange, options, hint, disabled = false }) {
  return <Field label={label} hint={hint}>
    <select value={value} disabled={disabled} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={key}>{text ?? key}</option>)}
    </select>
  </Field>;
}

/** A named set of checkboxes. Used for the eligible-candidate set, where the
 *  learner's own choice of what is being compared is the input. */
export function CheckboxSet({ legend, options, selected, onChange, disabled = false, hint }) {
  return <fieldset className="ete-checkset">
    <legend>{legend}</legend>
    {hint && <p className="ete-caption">{hint}</p>}
    {options.map(([key, text, extra]) => (
      <label key={key} className="ete-checkbox">
        <input type="checkbox" checked={selected.includes(key)} disabled={disabled}
          onChange={() => onChange(selected.includes(key)
            ? selected.filter(value => value !== key)
            : [...selected, key])} />
        <span><strong>{text}</strong>{extra ? <em> — {extra}</em> : null}</span>
      </label>
    ))}
  </fieldset>;
}

/** A table whose caption sits OUTSIDE the scroll box.
 *
 *  Below the narrow breakpoint every cell becomes its own labelled row, which
 *  needs `display: block`; that drops the implicit table roles, so they are
 *  restored explicitly. A `<caption>` inside a blockified table collapses to the
 *  width of its longest word, so the caption is a sibling paragraph. */
export function Table({ caption, headings, rows, rowClass = () => undefined, footnote }) {
  const id = useId();
  return <div className="ete-table">
    <p className="ete-caption" id={id}>{caption}</p>
    <div className="ete-table-scroll" role="region" aria-labelledby={id} tabIndex={0}>
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
    {footnote && <p className="ete-caption">{footnote}</p>}
  </div>;
}

/* ============================================ the investigation state machine */

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
 * A plot frame: axes, ticks and their labels, with the drawing area left to the
 * caller. Ticks come from the geometry so the verifier asserts the same
 * positions the browser paints.
 *
 * Every SVG this lesson renders carries one of its own layout classes, because
 * the layout rule in the stylesheet is scoped to those classes. An unscoped
 * `.endtoend-lesson svg { height: auto }` would also match KaTeX's radical
 * SVGs, whose height comes from `height: inherit`, and collapse every square
 * root on the page to nothing.
 */
export function PlotFrame({
  geometry, xLabel, yLabel, children, caption, describe, className = 'ete-plot',
  xTickText = value => String(value), yTickText = value => String(value),
}) {
  const titleId = useId();
  const descriptionId = useId();
  const { width, height, padding } = geometry;
  return <figure className="ete-figure">
    <svg className={className} viewBox={`0 0 ${width} ${height}`} role="img"
      aria-labelledby={titleId} aria-describedby={descriptionId}
      style={{ maxWidth: `${geometry.maxWidth ?? width}px` }}>
      <title id={titleId}>{caption}</title>
      <desc id={descriptionId}>{describe}</desc>
      {geometry.yTicks.map(tick => <g key={`y${tick.value}`}>
        <line className="ete-grid" x1={padding.left} y1={tick.y} x2={width - padding.right} y2={tick.y} />
        <text className="ete-small" x={padding.left - 5} y={tick.y + 3} textAnchor="end">
          {yTickText(tick.value)}
        </text>
      </g>)}
      {/* The first and last tick labels anchor inward. Centred, the leftmost
          runs half outside the viewBox and the rightmost past its edge. */}
      {geometry.xTicks.map((tick, index) => <g key={`x${tick.value}`}>
        <line className="ete-tick" x1={tick.x} y1={height - padding.bottom}
          x2={tick.x} y2={height - padding.bottom + 4} />
        <text className="ete-small" x={tick.x} y={height - padding.bottom + 15}
          textAnchor={index === 0 ? 'start' : index === geometry.xTicks.length - 1 ? 'end' : 'middle'}>
          {xTickText(tick.value)}
        </text>
      </g>)}
      <line className="ete-axis" x1={padding.left} y1={padding.top}
        x2={padding.left} y2={height - padding.bottom} />
      <line className="ete-axis" x1={padding.left} y1={height - padding.bottom}
        x2={width - padding.right} y2={height - padding.bottom} />
      {children}
      {/* A CATEGORICAL axis puts its title at the left. End-anchored, it landed
          directly beneath the rightmost category label and read as a second line
          of it -- "linear_three / candidate", "cultivar 2 / actual cultivar".
          A quantitative axis has tick numbers rather than words under it, so the
          title stays at the right where it does not compete with the data. */}
      <text className="ete-small ete-axis-title"
        x={geometry.xTicks.length ? width - padding.right : padding.left} y={height - 3}
        textAnchor={geometry.xTicks.length ? 'end' : 'start'}>{xLabel}</text>
      {/* The y-axis title sits in its own band above the plot, anchored left.
          Beside the axis it collided with the topmost tick label. */}
      <text className="ete-small ete-axis-title" x={2} y={10} textAnchor="start">{yLabel}</text>
    </svg>
    <p className="ete-caption">{caption}</p>
  </figure>;
}

/** A bare SVG canvas for the diagrams that are not plots. It tags the element
 *  with a layout class for the same reason `PlotFrame` does. */
export function Canvas({ geometry, className, caption, describe, children }) {
  const titleId = useId();
  const descriptionId = useId();
  /* A geometry may ask to be rendered WIDER than its viewBox. The viewBox fixes
     the drawing's proportions; the rendered width fixes how large its text is in
     CSS pixels. A diagram carrying several short labels reads better given more
     room on a desktop, and constraining it to its own coordinate count would
     shrink those labels for no reason. */
  return <figure className="ete-figure">
    <svg className={className} viewBox={`0 0 ${geometry.width} ${geometry.height}`} role="img"
      aria-labelledby={titleId} aria-describedby={descriptionId}
      style={{ maxWidth: `${geometry.maxWidth ?? geometry.width}px` }}>
      <title id={titleId}>{caption}</title>
      <desc id={descriptionId}>{describe}</desc>
      {children}
    </svg>
    <p className="ete-caption">{caption}</p>
  </figure>;
}

/** A legend in reflowing HTML rather than SVG text. Names, not colours: each
 *  entry carries a word as well as a swatch. */
export function Legend({ entries }) {
  return <div className="ete-legend">
    {entries.map(([className, text]) => (
      <span key={text}><i className={`ete-swatch ${className}`} aria-hidden="true" />{text}</span>
    ))}
  </div>;
}
