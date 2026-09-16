import { cloneElement, isValidElement, useId, useState } from 'react';
import './gmm-labs.css';

/** Shared controls for the Gaussian mixture investigations.
 *
 * The contract every investigation keeps: the learner edits a draft, records a
 * prediction, and commits both together. A result is always tied to the inputs
 * it was computed from, so editing anything retires the answer instead of
 * quietly re-grading an old choice against a new model.
 */

export const round = (value, digits = 6) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '−∞';
  if (Number.isInteger(value)) return String(value);
  if (value !== 0 && Math.abs(value) < 1e-4) {
    const [mantissa, exponent] = value.toExponential(3).split('e');
    const digits = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    const superscript = [...exponent.replace('+', '')].map(character => digits[character] ?? character).join('');
    return `${mantissa} × 10${superscript}`;
  }
  const fixed = value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
  return fixed === '-0' ? '0' : fixed;
};
/** Keep every decimal place, for a column whose rows must line up and for a
 * value the prose quotes exactly. */
export const fixed = (value, digits = 6) => (Number.isFinite(value) ? value.toFixed(digits) : round(value));
export const percent = (value, digits = 2) => `${round(100 * value, digits)}%`;
export const integer = value => value.toLocaleString('en-US');
export const signed = (value, digits = 6) => (value >= 0 ? `+${round(value, digits)}` : round(value, digits));

export function Investigation({ title, question, note, children, onReset }) {
  const id = useId();
  return <section className="gm-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="gm-question">{question}</p>}
    {note && <p className="gm-note">{note}</p>}
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
  return <label className={`gm-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {error && <span className="gm-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation, and the model keeps
 * its last good state instead of being handed a silent substitute. */
export function NumberField({ label, value, onChange, min, max, step = 'any', decimals = 2, suffix }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const problem = (() => {
    if (draft === null) return null;
    if (draft.trim() === '' || draft.trim() === '-' || draft.trim() === '−') return 'Type a number.';
    const parsed = Number(draft);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    if (Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) > 1e-9) {
      return `Use at most ${decimals} decimal place${decimals === 1 ? '' : 's'}.`;
    }
    return null;
  })();
  return <Field label={label} error={problem} value={suffix}>
    <input type="number" inputMode="decimal" min={min} max={max} step={step} value={shown}
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
  return <div className={`gm-table-scroll${scroll ? ' gm-rows' : ''}`} role="region" aria-label={caption} tabIndex={0}>
    <table>
      <caption>{caption}</caption>
      <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index} className={rowClass(index)}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

/** Draft inputs, an unset prediction, and one action that commits both.
 *
 * `key` turns the active inputs into a string, so a recorded result can never
 * be shown beside inputs it was not computed from. */
export function useInvestigation(initial, describeKey = JSON.stringify) {
  const [draft, setDraft] = useState(initial);
  const [active, setActive] = useState(initial);
  const [choice, setChoice] = useState('');
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const pending = describeKey(draft) !== describeKey(active);
  const edit = update => {
    setDraft(previous => ({ ...previous, ...update }));
    setResult(null);
    setChoice('');
  };
  return {
    draft, active, choice, setChoice, reason, setReason, result, pending,
    edit,
    // The answer is computed from the draft at the moment of committing, so a
    // prediction is never graded against the inputs that were on screen before
    // the learner edited them.
    /** Commit the draft and record the prediction against it. */
    check: answerFor => { setActive(draft); setResult({ key: describeKey(draft), choice, reason, answer: answerFor(draft), graded: true }); },
    /** Compute the same thing without grading, for a learner who wants to look first. */
    explore: answerFor => { setActive(draft); setResult({ key: describeKey(draft), choice: '', reason, answer: answerFor(draft), graded: false }); },
    reset: () => { setDraft(initial); setActive(initial); setChoice(''); setReason(''); setResult(null); },
    /** Replace the whole declared setup, which also retires the prediction. */
    load: inputs => { setDraft(inputs); setActive(inputs); setChoice(''); setReason(''); setResult(null); },
  };
}

/** A radio group that starts with nothing selected, plus the two commit
 * actions. The check is disabled until a choice exists. */
export function Prediction({ prompt, options, state, answerFor, describe, exploreLabel = 'Calculate without recording a prediction' }) {
  const name = useId();
  const label = key => options.find(([value]) => value === key)?.[1] ?? key;
  const shown = state.result;
  const correct = shown?.graded && shown.choice === shown.answer;
  return <div className="gm-prediction">
    <fieldset>
      <legend>Record a prediction first.</legend>
      <p>{prompt}</p>
      <div className="gm-choices">
        {options.map(([value, text]) => (
          <label className="gm-choice" key={value}>
            <input type="radio" name={name} value={value} checked={state.choice === value}
              onChange={() => state.setChoice(value)} disabled={Boolean(shown)} />
            <span>{text}</span>
          </label>
        ))}
      </div>
    </fieldset>
    <Reason value={state.reason} onChange={state.setReason} disabled={Boolean(shown)} />
    {state.pending && <p className="gm-pending" role="status">
      Inputs changed; record a new prediction. The calculation below will use the values now in the fields.
    </p>}
    <div className="gm-buttons">
      <button type="button" className="is-primary" disabled={state.choice === '' || Boolean(shown)} onClick={() => state.check(answerFor)}>Check prediction</button>
      <button type="button" onClick={() => state.explore(answerFor)}>{exploreLabel}</button>
    </div>
    {shown?.reason && <p className="gm-caption">Your reason, kept as you wrote it: “{shown.reason}”</p>}
    {shown && (shown.graded
      ? <p className={`gm-verdict ${correct ? '' : 'is-miss'}`} role="status">
        <span className="gm-verdict-mark" aria-hidden="true">{correct ? '=' : '≠'}</span>
        {correct
          ? `Your prediction matches: ${label(shown.answer)}.`
          : `You recorded ${label(shown.choice)}; the calculation gives ${label(shown.answer)}.`}
        {describe ? ` ${describe}` : ''}
      </p>
      : <p className="gm-verdict is-plain" role="status">
        <span className="gm-verdict-mark" aria-hidden="true">·</span>
        Calculated without a recorded prediction: {label(shown.answer)}. {describe}
      </p>)}
  </div>;
}

/** An optional sentence saying why. It is never graded: a right answer with a
 * wrong reason is still worth inspecting, and that is the learner's to judge. */
export function Reason({ value, onChange, disabled = false, label = 'Why? Optional, never graded' }) {
  const id = useId();
  return <label className="gm-field gm-reason" htmlFor={id}>
    <span>{label}</span>
    <textarea id={id} rows={2} value={value} disabled={disabled} onChange={event => onChange(event.target.value)}
      placeholder="One sentence on the mechanism you expect to decide it" />
  </label>;
}

/** A framed plot with one shared scale for everything drawn on it. */
export function Plot({ caption, describe, width = 340, height = 200, padding = { left: 40, right: 14, top: 14, bottom: 30 }, domain, range, ticks, children }) {
  const scaleX = value => padding.left + (width - padding.left - padding.right) * (value - domain[0]) / (domain[1] - domain[0]);
  const scaleY = value => height - padding.bottom - (height - padding.top - padding.bottom) * (value - range[0]) / (range[1] - range[0]);
  const marks = ticks ?? Array.from({ length: 5 }, (_, index) => domain[0] + (domain[1] - domain[0]) * index / 4);
  return <figure className="gm-plot">
    {caption && <figcaption>{caption}</figcaption>}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {marks.map(value => <g key={value}>
        <line className="gm-grid" x1={scaleX(value)} x2={scaleX(value)} y1={padding.top} y2={height - padding.bottom} />
        <text x={scaleX(value)} y={height - padding.bottom + 14} textAnchor="middle">{round(value, 2)}</text>
      </g>)}
      <line className="gm-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
      <line className="gm-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
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
