import { cloneElement, isValidElement, useId, useState } from 'react';
import './validation-labs.css';

/** Shared primitives for the cross-validation investigations.
 *
 * The contract every investigation in this lesson keeps:
 *
 *   edit the entities → record a prediction → commit it → apply
 *
 * Committing freezes the inputs alongside the prediction. Applying computes the
 * answer *from those committed inputs*, so a recorded prediction is graded
 * against the fold plan it was made about and never against whatever happens to
 * be in the fields afterwards. Any edit between committing and applying retires
 * the commitment rather than re-grading it.
 *
 * There is deliberately no "just show me the answer" button: this lesson is
 * about what a score was allowed to see before it was read.
 */

export const round = (value, digits = 6) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '−∞';
  if (Number.isInteger(value)) return String(value);
  const text = value.toFixed(digits);
  const trimmed = text.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  return trimmed === '-0' ? '0' : trimmed;
};
/** Keep every decimal place, for a value the prose quotes exactly and for a
 * column whose rows must line up. */
export const fixed = (value, digits = 6) => (Number.isFinite(value) ? value.toFixed(digits) : round(value));

/** The role vocabulary, as a word plus a glyph. Nothing in this lesson relies
 * on hue to say what a row was allowed to influence. */
export const ROLES = {
  train: { label: 'training', glyph: '●', description: 'fits this copy of the whole procedure' },
  held: { label: 'held out', glyph: '□', description: 'assessed by this fit, and kept out of it' },
  protected: { label: 'protected', glyph: '▣', description: 'outside every choice being made here' },
  absent: { label: 'no prediction', glyph: '·', description: 'this plan never holds the row out' },
};

export function RoleMark({ kind, children }) {
  const role = ROLES[kind];
  return <span className={`cv-role is-${kind}`}>{role.glyph} {children ?? role.label}</span>;
}

export function Legend({ kinds = ['train', 'held'], extra }) {
  return <p className="cv-legend">
    <span>Legend:</span>
    {kinds.map(kind => <span key={kind}><RoleMark kind={kind} /> {ROLES[kind].description}</span>)}
    {extra}
  </p>;
}

export function Investigation({ id, title, question, note, children, onReset }) {
  const headingId = useId();
  return <section className="cv-lab" data-cv-lab={id} aria-labelledby={headingId}>
    <header><h3 id={headingId}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="cv-question">{question}</p>}
    {note && <p className="cv-note">{note}</p>}
    {children}
  </section>;
}

export function Figure({ id, caption, children, className = '' }) {
  return <figure className={`cv-figure ${className}`} data-cv-figure={id}>
    {caption && <figcaption><strong>{caption}</strong></figcaption>}
    {children}
  </figure>;
}

/** A labelled control whose range is visible before it is broken.
 *
 * `shortLabel` lets a dense row of controls show "x" while the control still
 * carries the full "feature x of row 3" as its accessible name. */
export function Field({ label, shortLabel, range, error, children }) {
  const id = useId();
  // The visible range hint and a select's own option text would otherwise join
  // the accessible name, so the control carries the label explicitly.
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-label': label, 'aria-describedby': error ? `${id}-error` : undefined })
    : children;
  return <label className={`cv-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{shortLabel ?? label}{range !== undefined && <span className="cv-range" aria-hidden="true">{range}</span>}</span>
    {control}
    {error && <span className="cv-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation while the model keeps
 * its last good state, so nothing is silently substituted. */
export function NumberField({ label, shortLabel, value, onChange, min, max, step = 'any', decimals = 2 }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const problem = (() => {
    if (draft === null) return null;
    const text = draft.trim();
    if (text === '' || text === '-' || text === '−') return 'Type a number.';
    const parsed = Number(text);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    if (Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) > 1e-9) {
      return `Use at most ${decimals} decimal place${decimals === 1 ? '' : 's'}.`;
    }
    return null;
  })();
  return <Field label={label} shortLabel={shortLabel} range={`${min} to ${max}`} error={problem}>
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

export function SelectField({ label, shortLabel, value, onChange, options, range }) {
  return <Field label={label} shortLabel={shortLabel} range={range}>
    <select value={String(value)} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={String(key)}>{text}</option>)}
    </select>
  </Field>;
}

export function Table({ caption, headings, rows, rowClass = () => undefined, numeric = [], id }) {
  return <div className="cv-table-scroll" role="region" aria-label={caption} tabIndex={0} data-cv-table={id}>
    <table>
      <caption>{caption}</caption>
      <thead><tr>{headings.map((heading, index) =>
        <th key={heading} scope="col" className={numeric.includes(index) ? 'is-numeric' : undefined}>{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index} className={rowClass(index)}>
        {row.map((cell, column) => column === 0
          ? <th key={column} scope="row">{cell}</th>
          : <td key={column} className={numeric.includes(column) ? 'is-numeric' : undefined}>{cell}</td>)}
      </tr>)}</tbody>
    </table>
  </div>;
}

/** Draft entities, a committed prediction bound to them, and an applied result
 * computed from the committed copy. */
export function useInvestigation(initial, describeKey = JSON.stringify) {
  const [draft, setDraft] = useState(initial);
  const [answers, setAnswers] = useState({});
  const [reason, setReason] = useState('');
  const [committed, setCommitted] = useState(null);
  const [result, setResult] = useState(null);
  const [previous, setPrevious] = useState(null);

  // An edit retires the answer on screen. The retired answer is kept as a
  // labelled comparison, because the moment a learner changes one label and
  // re-runs is exactly the moment the earlier answer is worth looking at.
  const retire = () => {
    if (result) setPrevious({ inputs: result.inputs, answer: result.answer, answers: result.answers, label: result.label, questions: result.questions });
    setCommitted(null);
    setResult(null);
    setAnswers({});
  };
  const edit = update => { setDraft(current => ({ ...current, ...update })); retire(); };

  return {
    draft, answers, setAnswers, reason, setReason, committed, result, previous,
    edit,
    /** Freeze the prediction together with the inputs it is about. */
    commit: questions => setCommitted({ key: describeKey(draft), inputs: draft, answers, reason,
      questions: questions.map(({ key, short, options }) => ({ key, short, options: options.map(pair => [...pair]) })),
    }),
    /** Compute from the committed inputs, never from the live fields. */
    apply: answerFor => {
      if (!committed) return;
      setResult({
        key: committed.key, inputs: committed.inputs, answers: committed.answers,
        reason: committed.reason, answer: answerFor(committed.inputs), label: describeKey(committed.inputs), questions: committed.questions,
      });
    },
    reset: () => { setDraft(initial); setAnswers({}); setReason(''); setCommitted(null); setResult(null); setPrevious(null); },
    /** Load a declared setup. This is an input change, so it retires the
     * prediction and the previous comparison with it. */
    load: inputs => { setDraft(inputs); setAnswers({}); setReason(''); setCommitted(null); setResult(null); setPrevious(null); },
  };
}

/** An optional sentence saying why. Never graded: a right answer for a wrong
 * reason is worth noticing, and that judgement belongs to the learner. */
export function Reason({ value, onChange, disabled = false }) {
  const id = useId();
  return <label className="cv-field cv-reason" htmlFor={id}>
    <span>Why? Optional, never graded</span>
    <textarea id={id} rows={2} value={value} disabled={disabled} onChange={event => onChange(event.target.value)}
      placeholder="One sentence on the mechanism you expect to decide it" />
  </label>;
}

/** One or more questions, all of which must be answered before the prediction
 * can be committed, and an Apply action that only unlocks afterwards. */
export function Prediction({ questions, state, answerFor, applyLabel, describe, disabled = false, disabledReason }) {
  const name = useId();
  const committed = state.committed;
  const shown = state.result;
  const answered = questions.every(question => state.answers[question.key] !== undefined && state.answers[question.key] !== '');
  const labelOf = (question, value) => question.options.find(([key]) => String(key) === String(value))?.[1] ?? String(value);
  const verdicts = shown ? shown.questions.map(question => ({
    question,
    predicted: shown.answers[question.key],
    actual: shown.answer[question.key],
    correct: String(shown.answers[question.key]) === String(shown.answer[question.key]),
  })) : [];
  const allCorrect = verdicts.length > 0 && verdicts.every(item => item.correct);
  return <div className="cv-prediction" data-cv-prediction>
    {questions.map(question => (
      <fieldset key={question.key} disabled={Boolean(committed) || Boolean(shown)}>
        <legend>{question.legend ?? 'Record a prediction first.'}</legend>
        <p>{question.prompt}</p>
        <div className="cv-choices">
          {question.options.map(([value, text]) => (
            <label className="cv-choice" key={String(value)}>
              <input type="radio" name={`${name}-${question.key}`} value={String(value)}
                checked={String(state.answers[question.key] ?? '') === String(value)}
                onChange={() => state.setAnswers({ ...state.answers, [question.key]: String(value) })} />
              <span>{text}</span>
            </label>
          ))}
        </div>
      </fieldset>
    ))}
    <Reason value={state.reason} onChange={state.setReason} disabled={Boolean(committed) || Boolean(shown)} />
    {!committed && !shown && <p className="cv-caption" data-cv-status>
      Nothing is computed yet. Commit a prediction to bind it to the inputs now in the fields.
    </p>}
    {committed && !shown && <p className="cv-pending" role="status" data-cv-committed>
      Prediction recorded against these inputs. Apply to compute the answer from that same committed copy.
    </p>}
    <div className="cv-buttons">
      <button type="button" disabled={!answered || Boolean(committed) || Boolean(shown) || disabled}
        onClick={() => state.commit(questions)}>Commit prediction</button>
      <button type="button" className="is-primary" disabled={!committed || Boolean(shown) || disabled}
        onClick={() => state.apply(answerFor)}>{applyLabel}</button>
    </div>
    {disabled && disabledReason && <p className="cv-field-error" role="alert">{disabledReason}</p>}
    {shown?.reason && <p className="cv-caption">Your reason, kept as you wrote it: “{shown.reason}”</p>}
    {shown && <div data-cv-result>
      {verdicts.map(item => (
        <p key={item.question.key} className={`cv-verdict ${item.correct ? '' : 'is-miss'}`} role="status">
          <span className="cv-verdict-mark" aria-hidden="true">{item.correct ? '=' : '≠'}</span>
          {item.correct
            ? `${item.question.short}: your prediction matches — ${labelOf(item.question, item.actual)}.`
            : `${item.question.short}: you recorded ${labelOf(item.question, item.predicted)}; the calculation gives ${labelOf(item.question, item.actual)}.`}
        </p>
      ))}
      {describe && <p className="cv-caption" data-cv-mechanism>{describe(shown.answer, shown.inputs, allCorrect)}</p>}
    </div>}
    {state.previous && <p className="cv-previous" data-cv-previous>
      <strong>Previous trial, kept as a labelled comparison.</strong>{' '}
      {state.previous.questions.map(question => `${question.short}: ${labelOf(question, state.previous.answer[question.key])}`).join(' · ')}.
      That answer was computed from the inputs you have since edited, not from the ones in the fields now. Reset clears it.
    </p>}
  </div>;
}

/** A notice that an edit retired a recorded prediction. */
export function RetiredNotice({ state }) {
  if (state.committed || state.result) return null;
  return <p className="cv-caption" data-cv-retired>Inputs are editable; any change retires a recorded prediction and its answer.</p>;
}

/** A framed plot with one shared scale. The viewBox is 360 units wide, which is
 * the narrowest column this lesson renders a figure into. */
export function Plot({
  caption, describe, height = 200, width = 360,
  padding = { left: 44, right: 14, top: 16, bottom: 34 },
  domain, range, xTicks, yTicks, xLabel, yLabel, children,
}) {
  const scaleX = value => padding.left + (width - padding.left - padding.right) * (value - domain[0]) / (domain[1] - domain[0]);
  const scaleY = value => height - padding.bottom - (height - padding.top - padding.bottom) * (value - range[0]) / (range[1] - range[0]);
  const across = xTicks ?? Array.from({ length: 5 }, (_, index) => domain[0] + (domain[1] - domain[0]) * index / 4);
  const up = yTicks ?? Array.from({ length: 3 }, (_, index) => range[0] + (range[1] - range[0]) * index / 2);
  return <figure className="cv-plot">
    {caption && <figcaption className="cv-caption">{caption}</figcaption>}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {across.map(value => <g key={`x${value}`}>
        <line className="cv-grid" x1={scaleX(value)} x2={scaleX(value)} y1={padding.top} y2={height - padding.bottom} />
        <text x={scaleX(value)} y={height - padding.bottom + 15} textAnchor="middle">{round(value, 2)}</text>
      </g>)}
      {up.map(value => <g key={`y${value}`}>
        <line className="cv-grid" x1={padding.left} x2={width - padding.right} y1={scaleY(value)} y2={scaleY(value)} />
        <text x={padding.left - 5} y={scaleY(value) + 4} textAnchor="end">{round(value, 2)}</text>
      </g>)}
      <line className="cv-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
      <line className="cv-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
      {children(scaleX, scaleY)}
    </svg>
    {(xLabel || yLabel) && <p className="cv-caption">
      {xLabel && <>Horizontal axis: {xLabel}. </>}{yLabel && <>Vertical axis: {yLabel}.</>}
    </p>}
  </figure>;
}

/** A polyline through a function on a plot's own scale. */
export function curve(scaleX, scaleY, domain, evaluate, samples = 181) {
  return Array.from({ length: samples }, (_, index) => {
    const x = domain[0] + (domain[1] - domain[0]) * index / (samples - 1);
    return `${scaleX(x).toFixed(2)},${scaleY(evaluate(x)).toFixed(2)}`;
  }).join(' ');
}
