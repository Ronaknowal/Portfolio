import { cloneElement, isValidElement, useId, useState } from 'react';
import './anomaly-detection-labs.css';

/** Shared controls for the anomaly-detection investigations.
 *
 * Every investigation follows the same contract: nothing is answered until the
 * learner commits a prediction, the commitment remembers the inputs it was made
 * against, and editing any input retires that answer instead of quietly
 * re-grading it against a different question.
 */

export const round = (value, digits = 4) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (Number.isInteger(value)) return String(value);
  const fixed = value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
  return fixed === '-0' ? '0' : fixed;
};
export const percent = (value, digits = 2) => (value === null ? 'undefined' : `${round(100 * value, digits)}%`);
export const integer = value => value.toLocaleString('en-US');

export function Investigation({ title, question, children, onReset }) {
  const id = useId();
  return <section className="ad-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="ad-question">{question}</p>}
    {children}
  </section>;
}

/** A labelled control. The explicit id ties the label to the control rather
 * than to the output that shows the current value. */
export function Field({ label, value, error, children }) {
  const id = useId();
  const control = isValidElement(children) && typeof children.type === 'string' ? cloneElement(children, { id, 'aria-describedby': error ? `${id}-error` : undefined }) : children;
  return <label className={`ad-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {error && <span className="ad-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that only publishes a valid value. Drafts that are empty,
 * unparsable or out of range stay on screen with an explanation and leave the
 * model on its last good state. */
export function NumberField({ label, value, onChange, min, max, step = 'any', decimals = 2, suffix }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const problem = (() => {
    if (draft === null) return null;
    if (draft.trim() === '' || draft.trim() === '-') return 'Type a number.';
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
      onBlur={() => setDraft(null)} />
  </Field>;
}

export function Table({ caption, headings, rows, rowClass = () => undefined, scroll = false }) {
  return <div className={`ad-table-scroll${scroll ? ' ad-rows' : ''}`} role="region" aria-label={caption} tabIndex={0}>
    <table>
      <caption>{caption}</caption>
      <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index} className={rowClass(index)}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

export const UNSURE = 'unsure';

/** Predict, commit, reveal.
 *
 * `snapshot` is a string describing the inputs the question refers to. When it
 * changes, the committed answer is retired: the learner sees what they last
 * answered and against which inputs, and must commit again before revealing.
 */
export function usePrediction(snapshot) {
  const [choice, setChoice] = useState('');
  const [committed, setCommitted] = useState(null);
  const [revealed, setRevealed] = useState(false);
  const stale = committed !== null && committed.snapshot !== snapshot;
  return {
    choice,
    setChoice,
    committed,
    stale,
    revealed: revealed && !stale,
    canCommit: choice !== '' && (committed === null || stale),
    canReveal: committed !== null && !stale && !revealed,
    commit: answer => { setCommitted({ snapshot, choice, answer }); setRevealed(false); },
    reveal: () => setRevealed(true),
    reset: () => { setChoice(''); setCommitted(null); setRevealed(false); },
  };
}

export function Prediction({ prompt, options, state, answer, describe, revealLabel = 'Reveal the calculation' }) {
  const id = useId();
  const label = key => options.find(([value]) => value === key)?.[1] ?? key;
  const outcome = state.committed && !state.stale
    ? (state.committed.choice === UNSURE ? 'unsure' : state.committed.choice === state.committed.answer ? 'match' : 'miss')
    : null;
  return <div className="ad-prediction">
    <p><label htmlFor={id}><strong>Predict first.</strong> {prompt}</label></p>
    <div className="ad-prediction-row">
      <select id={id} value={state.choice} onChange={event => state.setChoice(event.target.value)}>
        <option value="">Choose a prediction</option>
        {options.map(([value, text]) => <option key={value} value={value}>{text}</option>)}
        <option value={UNSURE}>I am not sure</option>
      </select>
      <button type="button" disabled={!state.canCommit} onClick={() => state.commit(answer)}>Commit prediction</button>
      <button type="button" className="is-primary" disabled={!state.canReveal} onClick={state.reveal}>{revealLabel}</button>
    </div>
    {state.stale && <p className="ad-stale" role="status">
      The inputs changed after your last prediction, so that answer is retired. You last chose
      “{label(state.committed.choice)}”. Commit a prediction for the settings now on screen.
    </p>}
    {state.revealed && outcome && <p className={`ad-verdict ${outcome === 'match' ? 'is-match' : outcome === 'miss' ? 'is-miss' : 'is-unsure'}`} role="status">
      <span className="ad-verdict-mark" aria-hidden="true">{outcome === 'match' ? '=' : outcome === 'miss' ? '≠' : '?'}</span>
      {outcome === 'match' ? `Your prediction matches: ${label(state.committed.answer)}.`
        : outcome === 'miss' ? `You chose ${label(state.committed.choice)}; the calculation gives ${label(state.committed.answer)}.`
          : `You committed “not sure”. The calculation gives ${label(state.committed.answer)}.`}
      {describe ? ` ${describe}` : ''}
    </p>}
  </div>;
}

/** A number line that keeps one shared scale for every mark drawn on it.
 *
 * `tickSide` moves the scale's own numbers above the plot, which frees the
 * space under the axis for labels belonging to the marks. */
export function NumberLine({ from, to, height = 96, children, describe, caption, tickSide = 'below', bottomPadding = 26 }) {
  const project = value => 34 + 272 * (value - from) / (to - from);
  const ticks = Array.from({ length: 5 }, (_, index) => from + (to - from) * index / 4);
  const baseline = height - bottomPadding;
  const above = tickSide === 'above';
  return <figure className="ad-plot">
    {caption && <figcaption>{caption}</figcaption>}
    <svg viewBox={`0 0 340 ${height}`} role="img" aria-label={describe}>
      <line className="ad-axis" x1="34" x2="306" y1={baseline} y2={baseline} />
      {ticks.map(value => <g key={value}>
        <line className="ad-grid" x1={project(value)} x2={project(value)} y1={above ? 18 : 8} y2={baseline} />
        <text x={project(value)} y={above ? 12 : height - 10} textAnchor="middle">{round(value, 2)}</text>
      </g>)}
      {children(project, baseline)}
    </svg>
  </figure>;
}
