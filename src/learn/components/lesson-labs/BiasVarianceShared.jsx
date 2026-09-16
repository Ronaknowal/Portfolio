import { cloneElement, isValidElement, useId, useState } from 'react';
import './bias-variance-labs.css';

/** Shared controls for the bias–variance investigations.
 *
 * The contract every investigation keeps: the learner edits a draft, records a
 * prediction, and commits both together. A result is always computed from the
 * draft at the moment of committing, so a prediction is graded against the
 * inputs it was recorded with and never against whatever is on screen later.
 * Any relevant edit retires the recorded prediction and hides its feedback.
 *
 * One addition over the earlier lessons: the committed state keeps the state it
 * replaced. Investigation 1 grades a *direction* — will expected error fall,
 * stay put or rise — which only means anything against the previous applied
 * state, so `check` hands both to the answer function.
 */

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
/** Keep every decimal place, for a column whose rows must line up and for a
 * value the prose quotes exactly. */
export const fixed = (value, digits = 6) => (Number.isFinite(value) ? sign(value.toFixed(digits)) : round(value));
/** A sign is information about a direction, so an exact zero carries none: it is
 * printed as 0 rather than as a signed zero. */
export const signed = (value, digits = 6) => {
  if (value === 0) return '0';
  return value > 0 ? `+${round(value, digits)}` : round(value, digits);
};
/** An exact zero is said, not shaded. A tiny nonzero value is never printed as 0. */
export const exactly = (value, digits = 6) => (value === 0 ? 'exactly 0' : round(value, digits));

export function Investigation({ title, question, note, children, onReset }) {
  const id = useId();
  return <section className="bv-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="bv-question">{question}</p>}
    {note && <p className="bv-note">{note}</p>}
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
  return <label className={`bv-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {error && <span className="bv-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation, and the model keeps
 * its last good state instead of being handed a silent substitute. */
export function NumberField({ label, value, onChange, min, max, step = 'any', decimals = 2, suffix, disabled = false }) {
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

export function Table({ caption, headings, rows, rowClass = () => undefined, scroll = false }) {
  // Below the narrow breakpoint every cell becomes its own labelled row, which
  // needs `display: block`; that drops the implicit table roles, so they are
  // restored explicitly here rather than left to the layout.
  return <div className={`bv-table-scroll${scroll ? ' bv-rows' : ''}`} role="region" aria-label={caption} tabIndex={0}>
    <table role="table">
      <caption>{caption}</caption>
      <thead role="rowgroup"><tr role="row">
        {headings.map(heading => <th key={heading} role="columnheader" scope="col">{heading}</th>)}
      </tr></thead>
      <tbody role="rowgroup">{rows.map((row, index) => <tr key={index} role="row" className={rowClass(index)}>
        {row.map((cell, column) => (column === 0
          ? <th key={column} role="rowheader" scope="row" data-label={headings[column]}>{cell}</th>
          : <td key={column} role="cell" data-label={headings[column]}>{cell}</td>))}
      </tr>)}</tbody>
    </table>
  </div>;
}

/** Draft inputs, unset commitments, and one action that commits them together.
 *
 * `describeKey` turns the active inputs into a string, so a recorded result can
 * never be shown beside inputs it was not computed from. `previous` is the state
 * the commit replaced, which is what a direction-of-change prediction is graded
 * against.
 */
export function useInvestigation(initial, describeKey = JSON.stringify) {
  const [draft, setDraft] = useState(initial);
  const [active, setActive] = useState(initial);
  const [previous, setPrevious] = useState(initial);
  const [choice, setChoice] = useState('');
  const [guess, setGuess] = useState('');
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const pending = describeKey(draft) !== describeKey(active);
  const retire = () => { setResult(null); setChoice(''); setGuess(''); };
  return {
    draft, active, previous, choice, setChoice, guess, setGuess, reason, setReason, result, pending,
    /** Any relevant edit retires the commitment and conceals stale feedback. */
    edit: update => { setDraft(previousDraft => ({ ...previousDraft, ...update })); retire(); },
    /** Commit the draft and record the prediction against it. */
    check: answerFor => {
      setPrevious(active);
      setActive(draft);
      setResult({
        key: describeKey(draft), previousKey: describeKey(active),
        choice, guess, reason, answer: answerFor(draft, active),
      });
    },
    reset: () => {
      setDraft(initial); setActive(initial); setPrevious(initial);
      setChoice(''); setGuess(''); setReason(''); setResult(null);
    },
    /** Replace the whole declared setup, which also retires the prediction. */
    load: inputs => {
      setDraft(inputs); setActive(inputs); setPrevious(inputs);
      setChoice(''); setGuess(''); setReason(''); setResult(null);
    },
    /** Fill the draft from a suggested setup without applying or grading it. */
    suggest: inputs => { setDraft(inputs); retire(); },
  };
}

/** An optional sentence saying why. It is never graded. */
export function Reason({ value, onChange, disabled = false, label = 'Why? Optional, never graded' }) {
  const id = useId();
  return <label className="bv-field bv-reason" htmlFor={id}>
    <span>{label}</span>
    <textarea id={id} rows={2} value={value} disabled={disabled} onChange={event => onChange(event.target.value)}
      placeholder="One sentence on the mechanism you expect to decide it" />
  </label>;
}

/** A radio group that starts with nothing selected, optionally a second numeric
 * commitment, and one action that commits both.
 *
 * There is deliberately no way to reach the answer without recording a
 * prediction: the contract requires one before Apply. Suggested setups fill
 * inputs, never the outcome. */
export function Prediction({
  prompt, options, state, answerFor, describe, numeric, committed,
  applyLabel = 'Apply and check',
}) {
  const name = useId();
  const numericId = useId();
  const label = key => options.find(([value]) => value === key)?.[1] ?? key;
  const shown = state.result;
  const correct = shown && shown.choice === shown.answer.outcome;
  const guessed = shown && numeric && shown.guess !== ''
    ? Math.abs(Number(shown.guess) - shown.answer.value) <= numeric.tolerance + 4 * Number.EPSILON * Math.max(1, Math.abs(Number(shown.guess)), Math.abs(shown.answer.value))
    : null;
  const ready = state.choice !== ''
    && (!numeric?.required || (state.guess.trim() !== '' && Number.isFinite(Number(state.guess))));
  return <div className="bv-prediction">
    <fieldset>
      <legend>Record a prediction first.</legend>
      <p>{prompt}</p>
      <div className="bv-choices">
        {options.map(([value, text]) => (
          <label className="bv-choice" key={value}>
            <input type="radio" name={name} value={value} checked={state.choice === value}
              onChange={() => state.setChoice(value)} disabled={Boolean(shown)} />
            <span>{text}</span>
          </label>
        ))}
      </div>
      {numeric && <label className="bv-field bv-numeric-guess" htmlFor={numericId}>
        <span>{numeric.label}</span>
        <span>Answers within {numeric.tolerance} are accepted.</span>
        <input id={numericId} type="number" inputMode="decimal" step="any" value={state.guess} disabled={Boolean(shown)}
          placeholder={numeric.placeholder ?? 'your number'} onChange={event => state.setGuess(event.target.value)} />
      </label>}
    </fieldset>
    <Reason value={state.reason} onChange={state.setReason} disabled={Boolean(shown)} />
    {state.pending && <p className="bv-pending" role="status">
      Draft inputs differ from the applied ones. The calculation below will use the values now in the fields, and the
      comparison will be against the state currently applied.
    </p>}
    <div className="bv-buttons">
      <button type="button" className="is-primary" disabled={!ready || Boolean(shown)} onClick={() => state.check(answerFor)}>{applyLabel}</button>
      {!shown && <span>The answer appears once a prediction is recorded. Reset, or edit an input, to try another setup.</span>}
    </div>
    {committed && shown && <p className="bv-caption">Graded against the committed state: {committed(shown)}</p>}
    {shown?.reason && <p className="bv-caption">Your reason, kept as you wrote it: “{shown.reason}”</p>}
    {shown && <p className={`bv-verdict ${correct ? '' : 'is-miss'}`} role="status">
      <span className="bv-verdict-mark" aria-hidden="true">{correct ? '=' : '≠'}</span>
      {correct
        ? `Your prediction matches: ${label(shown.answer.outcome)}.`
        : `You recorded ${label(shown.choice)}; the calculation gives ${label(shown.answer.outcome)}.`}
      {numeric && shown.guess !== '' && ` You wrote ${shown.guess} for ${numeric.name}; the calculation gives ${round(shown.answer.value, numeric.digits ?? 6)}, ${guessed ? `within ${numeric.tolerance}` : `outside ${numeric.tolerance}`}.`}
      {describe ? ` ${describe}` : ''}
    </p>}
  </div>;
}

/** A framed plot with one shared scale for everything drawn on it.
 *
 * `valueScale: 'log'` is available because one figure in this lesson spans two
 * orders of magnitude; the transform is always named in the caption. */
export function Plot({
  caption, describe, width = 340, height = 200,
  padding = { left: 46, right: 14, top: 16, bottom: 34 },
  domain, range, ticks, valueTicks, scale = 'linear', valueScale = 'linear',
  formatTick = value => round(value, 2), formatValue = value => round(value, 3), children,
}) {
  const forward = (value, kind) => (kind === 'log' ? Math.log10(value) : value);
  const scaleX = value => padding.left + (width - padding.left - padding.right)
    * (forward(value, scale) - forward(domain[0], scale)) / (forward(domain[1], scale) - forward(domain[0], scale));
  const scaleY = value => height - padding.bottom - (height - padding.top - padding.bottom)
    * (forward(value, valueScale) - forward(range[0], valueScale)) / (forward(range[1], valueScale) - forward(range[0], valueScale));
  const marks = ticks ?? Array.from({ length: 5 }, (_, index) => domain[0] + (domain[1] - domain[0]) * index / 4);
  const sideMarks = valueTicks ?? [range[0], (range[0] + range[1]) / 2, range[1]];
  return <figure className="bv-plot">
    {caption && <figcaption>{caption}</figcaption>}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {marks.map(value => <g key={`x${value}`}>
        <line className="bv-grid" x1={scaleX(value)} x2={scaleX(value)} y1={padding.top} y2={height - padding.bottom} />
        <text x={scaleX(value)} y={height - padding.bottom + 18} textAnchor="middle">{formatTick(value)}</text>
      </g>)}
      {/* Clamped inside the drawn box, so the lowest value label never lands on
          the row of horizontal tick labels beneath the axis. */}
      {sideMarks.map(value => <text key={`y${value}`} x={padding.left - 6} textAnchor="end"
        y={Math.min(Math.max(scaleY(value) + 4, padding.top + 9), height - padding.bottom - 2)}>{formatValue(value)}</text>)}
      <line className="bv-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
      <line className="bv-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
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

/** Three nonnegative quantities adding to one total, drawn as one track.
 *
 * Every part carries its own fill pattern and its own printed value, so the
 * three are distinguishable without colour and a part that is exactly zero says
 * so instead of becoming an invisible sliver. */
export function Waterfall({ caption, describe, parts, totalLabel = 'total', digits = 6, unit = '' }) {
  const total = parts.reduce((sum, part) => sum + part.value, 0);
  const extent = Math.max(total, 1e-12);
  return <div className="bv-waterfall" role="img" aria-label={describe}>
    <p className="bv-caption">{caption}</p>
    <div className="bv-waterfall-track">
      {parts.map(part => (part.value > 0
        ? <span key={part.name} className={`bv-waterfall-part ${part.className}`}
          style={{ width: `${100 * part.value / extent}%` }} />
        : null))}
    </div>
    <dl className="bv-waterfall-key">
      {parts.map(part => <div key={part.name} className="bv-waterfall-row">
        <dt><span className={`bv-swatch ${part.className}`} aria-hidden="true" />{part.name}</dt>
        <dd>{part.value === 0 ? 'exactly 0' : `${round(part.value, digits)}${unit}`}</dd>
      </div>)}
      <div className="bv-waterfall-row is-total">
        <dt>{totalLabel}</dt>
        <dd>{round(total, digits)}{unit}</dd>
      </div>
    </dl>
  </div>;
}
