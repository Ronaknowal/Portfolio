import { cloneElement, isValidElement, useId, useState } from 'react';
import './regularization-labs.css';

/** Shared controls for the regularization investigations.
 *
 * The contract every investigation keeps: the learner edits a draft, records a
 * prediction, and commits both together. A result is always computed from the
 * draft at the moment of committing, so a prediction is graded against the
 * inputs it was recorded with and never against whatever is on screen later.
 * Any relevant edit retires the recorded prediction and hides its feedback.
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
export const signed = (value, digits = 6) => (value >= 0 ? `+${round(value, digits)}` : round(value, digits));
/** An exact zero is said, not shaded. A tiny nonzero value is never printed as 0. */
export const coefficientText = (value, digits = 6) => (value === 0 ? 'exactly 0' : round(value, digits));

export function Investigation({ title, question, note, children, onReset }) {
  const id = useId();
  return <section className="rg-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="rg-question">{question}</p>}
    {note && <p className="rg-note">{note}</p>}
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
  return <label className={`rg-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {error && <span className="rg-field-error" id={`${id}-error`}>{error}</span>}
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
  return <div className={`rg-table-scroll${scroll ? ' rg-rows' : ''}`} role="region" aria-label={caption} tabIndex={0}>
    <table>
      <caption>{caption}</caption>
      <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index} className={rowClass(index)}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

/** Draft inputs, unset commitments, and one action that commits them together.
 *
 * `key` turns the active inputs into a string, so a recorded result can never
 * be shown beside inputs it was not computed from. */
export function useInvestigation(initial, describeKey = JSON.stringify) {
  const [draft, setDraft] = useState(initial);
  const [active, setActive] = useState(initial);
  const [choice, setChoice] = useState('');
  const [guess, setGuess] = useState('');
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const pending = describeKey(draft) !== describeKey(active);
  const retire = () => { setResult(null); setChoice(''); setGuess(''); };
  return {
    draft, active, choice, setChoice, guess, setGuess, reason, setReason, result, pending,
    /** Any relevant edit retires the commitment and conceals stale feedback. */
    edit: update => { setDraft(previous => ({ ...previous, ...update })); retire(); },
    /** Commit the draft and record the prediction against it. */
    check: answerFor => {
      setActive(draft);
      setResult({ key: describeKey(draft), choice, guess, reason, answer: answerFor(draft) });
    },
    reset: () => { setDraft(initial); setActive(initial); setChoice(''); setGuess(''); setReason(''); setResult(null); },
    /** Replace the whole declared setup, which also retires the prediction. */
    load: inputs => { setDraft(inputs); setActive(inputs); setChoice(''); setGuess(''); setReason(''); setResult(null); },
  };
}

/** An optional sentence saying why. It is never graded. */
export function Reason({ value, onChange, disabled = false, label = 'Why? Optional, never graded' }) {
  const id = useId();
  return <label className="rg-field rg-reason" htmlFor={id}>
    <span>{label}</span>
    <textarea id={id} rows={2} value={value} disabled={disabled} onChange={event => onChange(event.target.value)}
      placeholder="One sentence on the mechanism you expect to decide it" />
  </label>;
}

/** A radio group that starts with nothing selected, optionally a second numeric
 * commitment, and one action that commits both.
 *
 * There is deliberately no way to reach the answer without recording a
 * prediction: the binding contract requires one before Apply, and grants relief
 * only to explanatory figures. Presets fill inputs, never the outcome. */
export function Prediction({
  prompt, options, state, answerFor, describe, numeric,
  applyLabel = 'Check prediction',
}) {
  const name = useId();
  const numericId = useId();
  const label = key => options.find(([value]) => value === key)?.[1] ?? key;
  const shown = state.result;
  const correct = shown && shown.choice === shown.answer.outcome;
  const guessed = shown && numeric && shown.guess !== ''
    ? Math.abs(Number(shown.guess) - shown.answer.value) <= numeric.tolerance
      + 4 * Number.EPSILON * Math.max(1, Math.abs(Number(shown.guess)), Math.abs(shown.answer.value))
    : null;
  const ready = state.choice !== ''
    && (!numeric?.required || (state.guess.trim() !== '' && Number.isFinite(Number(state.guess))));
  return <div className="rg-prediction">
    <fieldset>
      <legend>Record a prediction first.</legend>
      <p>{prompt}</p>
      <div className="rg-choices">
        {options.map(([value, text]) => (
          <label className="rg-choice" key={value}>
            <input type="radio" name={name} value={value} checked={state.choice === value}
              onChange={() => state.setChoice(value)} disabled={Boolean(shown)} />
            <span>{text}</span>
          </label>
        ))}
      </div>
      {numeric && <label className="rg-field rg-numeric-guess" htmlFor={numericId}>
        <span>{numeric.label}</span>
        <span className="rg-caption">Answers within {numeric.tolerance} are accepted.</span>
        <input id={numericId} type="number" inputMode="decimal" step="any" value={state.guess} disabled={Boolean(shown)}
          placeholder={numeric.placeholder ?? 'your number'} onChange={event => state.setGuess(event.target.value)} />
      </label>}
    </fieldset>
    <Reason value={state.reason} onChange={state.setReason} disabled={Boolean(shown)} />
    {state.pending && <p className="rg-pending" role="status">
      Inputs changed; record a new prediction. The calculation below will use the values now in the fields.
    </p>}
    <div className="rg-buttons">
      <button type="button" className="is-primary" disabled={!ready || Boolean(shown)} onClick={() => state.check(answerFor)}>{applyLabel}</button>
      {!shown && <span>The answer appears once a prediction is recorded. Reset, or edit an input, to try another setup.</span>}
    </div>
    {shown?.reason && <p className="rg-caption">Your reason, kept as you wrote it: “{shown.reason}”</p>}
    {shown && <p className={`rg-verdict ${correct ? '' : 'is-miss'}`} role="status">
      <span className="rg-verdict-mark" aria-hidden="true">{correct ? '=' : '≠'}</span>
      {correct
        ? `Your prediction matches: ${label(shown.answer.outcome)}.`
        : `You recorded ${label(shown.choice)}; the calculation gives ${label(shown.answer.outcome)}.`}
      {numeric && shown.guess !== '' && ` You wrote ${shown.guess} for ${numeric.name}; the calculation gives ${round(shown.answer.value, numeric.digits ?? 6)}, ${guessed ? `within ${numeric.tolerance}` : `outside ${numeric.tolerance}`}.`}
      {describe ? ` ${describe}` : ''}
    </p>}
  </div>;
}

/** A framed plot with one shared scale for everything drawn on it. */
export function Plot({
  caption, describe, width = 340, height = 200,
  padding = { left: 44, right: 14, top: 16, bottom: 32 },
  domain, range, ticks, valueTicks, formatTick = value => round(value, 2), children,
}) {
  const scaleX = value => padding.left + (width - padding.left - padding.right) * (value - domain[0]) / (domain[1] - domain[0]);
  const scaleY = value => height - padding.bottom - (height - padding.top - padding.bottom) * (value - range[0]) / (range[1] - range[0]);
  const marks = ticks ?? Array.from({ length: 5 }, (_, index) => domain[0] + (domain[1] - domain[0]) * index / 4);
  const sideMarks = valueTicks ?? [range[0], (range[0] + range[1]) / 2, range[1]];
  return <figure className="rg-plot">
    {caption && <figcaption>{caption}</figcaption>}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {marks.map(value => <g key={`x${value}`}>
        <line className="rg-grid" x1={scaleX(value)} x2={scaleX(value)} y1={padding.top} y2={height - padding.bottom} />
        <text x={scaleX(value)} y={height - padding.bottom + 18} textAnchor="middle">{formatTick(value)}</text>
      </g>)}
      {/* Clamped inside the drawn box, so the lowest value label never lands on
          the row of horizontal tick labels beneath the axis. */}
      {sideMarks.map(value => <text key={`y${value}`} x={padding.left - 6} textAnchor="end"
        y={Math.min(Math.max(scaleY(value) + 4, padding.top + 9), height - padding.bottom - 2)}>{round(value, 3)}</text>)}
      <line className="rg-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
      <line className="rg-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
      {children(scaleX, scaleY)}
    </svg>
  </figure>;
}

/** A polyline through a function on the plot's own scale. */
export function curve(scaleX, scaleY, domain, evaluate, samples = 241) {
  return Array.from({ length: samples }, (_, index) => {
    const x = domain[0] + (domain[1] - domain[0]) * index / (samples - 1);
    return `${scaleX(x).toFixed(2)},${scaleY(evaluate(x)).toFixed(2)}`;
  }).join(' ');
}

/** Signed contribution bars around a common zero line. A value that is exactly
 * zero gets its own mark and its own word, never a faint colour. */
export function SignedBars({ caption, describe, items, unit = '', digits = 6, highlight = () => false }) {
  const extent = Math.max(...items.map(item => Math.abs(item.value)), 1e-12);
  return <div className="rg-bars" role="img" aria-label={describe}>
    <p className="rg-caption">{caption}</p>
    {items.map((item, index) => {
      const share = 50 * Math.abs(item.value) / extent;
      const zero = item.value === 0;
      return <div className={`rg-bar-row${highlight(index) ? ' is-highlight' : ''}`} key={item.name}>
        <span className="rg-bar-name">{item.name}</span>
        <span className="rg-bar-track">
          <span className="rg-bar-zero" />
          {zero
            ? <span className="rg-bar-exact-zero" aria-hidden="true">0</span>
            : <span className={`rg-bar-fill ${item.value > 0 ? 'is-positive' : 'is-negative'}`}
              style={item.value > 0
                ? { left: '50%', width: `${share}%` }
                : { right: '50%', width: `${share}%` }} />}
        </span>
        <span className="rg-bar-value">{zero ? 'exactly 0' : `${signed(item.value, digits)}${unit}`}</span>
      </div>;
    })}
  </div>;
}
