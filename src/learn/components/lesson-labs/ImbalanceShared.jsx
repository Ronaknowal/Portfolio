import { cloneElement, isValidElement, useId, useState } from 'react';
import './imbalance-labs.css';

/** Shared controls for the imbalanced-learning investigations.
 *
 * The contract every investigation keeps: the learner edits a draft, records a
 * prediction, and commits both together. A result is always computed from the
 * draft at the moment of committing, so a prediction is graded against the
 * inputs it was recorded with and never against whatever is on screen later.
 * Any relevant edit retires the recorded prediction and hides its feedback.
 *
 * Two additions over the earlier lessons, both required by this topic:
 *
 *   * A retired verdict is kept as clearly labelled history rather than being
 *     erased, because several investigations here ask a learner to change one
 *     input and see that the answer did *not* move. Losing the previous attempt
 *     would destroy the comparison the null is about.
 *   * A `role` badge travels with the state, so a panel can say whether the
 *     records on screen are tuning records, a locked inspection result, or an
 *     explicitly exploratory what-if copy. Reading an inspection outcome after
 *     experimenting with it is a different claim, and the interface has to say
 *     so rather than quietly keep the old label.
 */

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

/** Draft inputs, unset commitments, and one action that commits them together.
 *
 * `describeKey` turns the active inputs into a string, so a recorded result can
 * never be shown beside inputs it was not computed from. `previous` is the state
 * the commit replaced, which is what a direction-of-change prediction is graded
 * against. `history` keeps retired verdicts as clearly previous attempts.
 */
export function useInvestigation(initial, describeKey = JSON.stringify) {
  const [draft, setDraft] = useState(initial);
  const [active, setActive] = useState(initial);
  const [previous, setPrevious] = useState(initial);
  const [choice, setChoice] = useState('');
  const [guess, setGuess] = useState('');
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const pending = describeKey(draft) !== describeKey(active);
  /** An edit retires the verdict. The retired one is kept, clearly marked as a
   * previous attempt, because several tasks here are about an answer NOT
   * moving, and that comparison needs the earlier result to still be readable. */
  const retire = () => {
    setResult(current => {
      if (current) setHistory(entries => [...entries.slice(-2), current]);
      return null;
    });
    setChoice(''); setGuess('');
  };
  return {
    draft, active, previous, choice, setChoice, guess, setGuess, reason, setReason, result, pending, history,
    edit: update => { setDraft(previousDraft => ({ ...previousDraft, ...update })); retire(); },
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
      setChoice(''); setGuess(''); setReason(''); setResult(null); setHistory([]);
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
  return <label className="imb-field imb-reason" htmlFor={id}>
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
 * inputs, never the outcome.
 */
export function Prediction({
  prompt, options, state, answerFor, describe, numeric, committed,
  applyLabel = 'Apply and check', historyLabel, requireChange, pendingHint, sameQuestion,
}) {
  const name = useId();
  const numericId = useId();
  const label = key => options.find(([value]) => value === key)?.[1] ?? key;
  const shown = state.result;
  const correct = shown && shown.choice === shown.answer.outcome;
  /** A graded quantity can legitimately have no value: precision with nothing
   * selected is undefined, not zero and not a dash. A numeric guess against
   * such an answer is neither inside nor outside a tolerance — there is no
   * number to be outside of — and saying so is the whole point of section 2. */
  const answerIsNumeric = shown ? Number.isFinite(shown.answer.value) : false;
  const guessed = shown && numeric && shown.guess !== '' && answerIsNumeric
    ? Math.abs(Number(shown.guess) - shown.answer.value) <= numeric.tolerance + 4 * Number.EPSILON * Math.max(1, Math.abs(Number(shown.guess)), Math.abs(shown.answer.value))
    : null;
  /** A question that grades a CHANGE needs the thing it grades to have changed.
   * `state.pending` is too weak: selecting the question is itself an edit, so a
   * learner could choose "does moving a majority point change it?" and apply
   * without moving anything, getting "unchanged" — correct by construction and
   * evidence of nothing. The predicate names the inputs that must differ. */
  const changeMissing = requireChange ? !requireChange(state.draft, state.active) : false;
  const ready = state.choice !== ''
    && !changeMissing
    && (!numeric?.required || (state.guess.trim() !== '' && Number.isFinite(Number(state.guess))));
  const previous = state.history.at(-1);
  /** Two verdicts are only comparable if they answer the same question. An
   * investigation whose question is itself an input can retire a verdict and
   * then grade a different quantity; reporting "the answer moved" across that
   * would compare a collision with a displacement. */
  const comparable = previous && shown && (!sameQuestion || sameQuestion(previous.key, shown.key));
  const heldOutcome = comparable && previous.answer.outcome === shown.answer.outcome;
  return <div className="imb-prediction">
    <fieldset>
      <legend>Record a prediction first.</legend>
      <p>{prompt}</p>
      <div className="imb-choices">
        {options.map(([value, text]) => (
          <label className="imb-choice" key={value}>
            <input type="radio" name={name} value={value} checked={state.choice === value}
              onChange={() => state.setChoice(value)} disabled={Boolean(shown)} />
            <span>{text}</span>
          </label>
        ))}
      </div>
      {numeric && <label className="imb-field imb-numeric-guess" htmlFor={numericId}>
        <span>{numeric.label}</span>
        <span>Answers within {numeric.tolerance} are accepted when the quantity is defined.</span>
        <input id={numericId} type="number" inputMode="decimal" step="any" value={state.guess} disabled={Boolean(shown)}
          placeholder={numeric.placeholder ?? 'your number'} onChange={event => state.setGuess(event.target.value)} />
      </label>}
    </fieldset>
    <Reason value={state.reason} onChange={state.setReason} disabled={Boolean(shown)} />
    {state.pending && <p className="imb-pending" role="status">
      Draft inputs differ from the applied ones. The calculation below will use the values now in the fields, and the
      comparison will be against the state currently applied.
    </p>}
    <div className="imb-buttons">
      <button type="button" className="is-primary" disabled={!ready || Boolean(shown)}
        onClick={() => state.check(answerFor)}>{applyLabel}</button>
      {!shown && <span>The answer appears once a prediction is recorded. Reset, or edit an input, to try another setup.</span>}
    </div>
    {changeMissing && !shown && <p className="imb-note" role="status">
      {pendingHint ?? 'This question compares two states, so it needs a change to compare. Edit an input, or load one of the setups above, before applying.'}
    </p>}
    {committed && shown && <p className="imb-caption">Graded against the committed state: {committed(shown)}</p>}
    {shown?.reason && <p className="imb-caption">Your reason, kept as you wrote it: “{shown.reason}”</p>}
    {shown && <p className={`imb-verdict ${correct ? '' : 'is-miss'}`} role="status">
      <span className="imb-verdict-mark" aria-hidden="true">{correct ? '=' : '≠'}</span>
      {correct
        ? `Your prediction matches: ${label(shown.answer.outcome)}.`
        : `You recorded ${label(shown.choice)}; the calculation gives ${label(shown.answer.outcome)}.`}
      {numeric && shown.guess !== '' && (answerIsNumeric
        ? ` You wrote ${shown.guess} for ${numeric.name}; the calculation gives ${round(shown.answer.value, numeric.digits ?? 6)}, ${guessed ? `within ${numeric.tolerance}` : `outside ${numeric.tolerance}`}.`
        : ` You wrote ${shown.guess} for ${numeric.name}, but there is no number to compare it against here: ${numeric.undefinedNote ?? 'the quantity is undefined for these inputs'}. That is not a near miss, and it is not zero.`)}
      {describe ? ` ${describe}` : ''}
    </p>}
    {/* The retired verdict appears ONLY beside the new one.
     *
     * Three of this lesson's investigations grade an absolute property of the
     * committed draft, so printing the previous outcome while the next
     * prediction is being recorded hands over the answer — and for a null, the
     * previous outcome IS the answer. It is also backwards: a null is a claim
     * about two results, which can only be compared once the second exists.
     * So nothing but a neutral notice is shown while the gate is open, and the
     * comparison is drawn after the commitment it describes. */}
    {!shown && previous && <p className="imb-history is-pending" role="status">
      An earlier attempt is held. It stays hidden until you apply, so that it cannot answer the question now on
      screen; the two results are then shown side by side.
    </p>}
    {shown && previous && (comparable
      ? <p className="imb-history">
        {historyLabel ?? 'Compared with your previous attempt'}: the calculation gave {label(previous.answer.outcome)}
        {' '}then and {label(shown.answer.outcome)} now.{' '}
        {heldOutcome
          ? 'The answer did not move between the two applied states — which is what a null looks like when it holds.'
          : 'The answer moved between the two applied states.'}
      </p>
      : <p className="imb-history">
        Your previous attempt answered a different question, so the two results are not put side by side: comparing
        them would compare two different quantities.
      </p>)}
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
