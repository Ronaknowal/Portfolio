import { useId, useState } from 'react';
import './ica-labs.css';

/** Exact values are the primary data; this only chooses how many places show. */
export function format(value, places = 6) {
  if (value === null || value === undefined) return '—';
  if (!Number.isFinite(value)) return '—';
  const rounded = Number(value.toFixed(places));
  const text = Object.is(rounded, -0) ? '0' : String(rounded);
  return text.replace('-', '−');
}

export const signedFormat = (value, places = 6) => {
  const text = format(Math.abs(value), places);
  return text === '0' ? '0' : `${value < 0 ? '−' : '+'}${text}`;
};

const copy = value => JSON.parse(JSON.stringify(value));

/** Feedback must preserve differences that the numerical grader can detect. */
export const feedbackNumber = value => Number(value.toPrecision(12)).toString().replace('-', '−');
export const feedbackDifference = value => value === 0 ? '0' : `${value < 0 ? '−' : '+'}${feedbackNumber(Math.abs(value))}`;

/** A prediction belongs to the serialized draft it was committed with.
 *  Editing any prediction-dependent input retires it, and a retired result is
 *  kept only as visibly labelled history.
 */
export function useIcaInvestigation(initial, validate = () => null) {
  const [draft, setDraft] = useState(() => copy(initial));
  const [active, setActive] = useState(() => copy(initial));
  const [prediction, setPrediction] = useState('');
  const [committed, setCommitted] = useState(null);
  const [result, setResult] = useState(null);
  const [previous, setPrevious] = useState(null);
  const [status, setStatus] = useState('Prediction not recorded.');
  const [resetCount, setResetCount] = useState(0);
  const key = JSON.stringify(draft);
  // A validator returns a string for an invalid draft, or {notice} for a draft
  // that is merely not ready yet. Both block Apply; only the first reads as an
  // error, so a required first action is not presented as a failure.
  const validation = validate(draft, active);
  const error = typeof validation === 'string' ? validation : null;
  const notice = validation && typeof validation === 'object' ? validation.notice : null;
  const blocked = Boolean(error || notice);
  const invalidate = (message = 'Draft changed. The recorded prediction was cleared; the last applied state is unchanged.') => {
    setPrediction('');
    setCommitted(null);
    setResult(current => (current ? { ...current, historical: true } : null));
    setStatus(message);
  };
  const edit = changes => {
    setDraft(current => ({ ...current, ...changes }));
    invalidate();
  };
  const choose = value => {
    setPrediction(value);
    setCommitted(null);
    setStatus('Prediction chosen. Record it before applying the draft.');
  };
  const commit = () => {
    if (blocked || prediction === '') return;
    setCommitted({ key, inputs: copy(draft), prediction });
    setStatus('Prediction recorded against these inputs. Apply them to reveal the result.');
  };
  const apply = evaluate => {
    if (blocked || !committed || committed.key !== key) return;
    const inputs = copy(committed.inputs);
    const evaluation = evaluate(inputs, active, committed.prediction);
    setPrevious({ active: copy(active), result });
    setActive(evaluation.nextActive ? copy(evaluation.nextActive) : inputs);
    if (evaluation.nextActive) setDraft(copy(evaluation.nextActive));
    setResult({ inputs, key, prediction: committed.prediction, evaluation });
    setPrediction('');
    setCommitted(null);
    setStatus(evaluation.status ?? 'Draft applied. Compare your recorded prediction with the result below.');
  };
  const reset = () => {
    setDraft(copy(initial));
    setActive(copy(initial));
    setPrediction('');
    setCommitted(null);
    setResult(null);
    setPrevious(null);
    setResetCount(count => count + 1);
    setStatus('Documented starting inputs restored. Prediction and result cleared.');
  };
  const undo = () => {
    if (!previous) return;
    setActive(copy(previous.active));
    setDraft(copy(previous.active));
    setResult(previous.result ? { ...previous.result, historical: true } : null);
    setPrevious(null);
    setPrediction('');
    setCommitted(null);
    setStatus('Previous applied inputs restored. Any earlier answer is history; record a new prediction.');
  };
  return {
    draft, active, prediction, committed, result, previous, status, key, error, notice, blocked, resetCount,
    currentResult: result && result.key === key && !result.historical ? result : null,
    edit, choose, commit, apply, reset, undo, invalidate, setStatus,
  };
}

export function Investigation({ kind, title, children, state }) {
  const titleId = useId();
  return <section className="ic-investigation" data-ica-lab={kind} aria-labelledby={titleId}>
    <header>
      <h3 id={titleId}>{title}</h3>
      <div className="ic-buttons">
        <button type="button" onClick={state.reset}>Reset</button>
        <button type="button" onClick={state.undo} disabled={!state.previous}>Undo apply</button>
      </div>
    </header>
    {children}
  </section>;
}

export function NumberField({ label, value, onChange, min, max, step = 'any', disabled = false }) {
  return <label className="ic-field">
    <span>{label}</span>
    <span className="ic-input-row">
      <input type="number" value={value} min={min} max={max} step={step} disabled={disabled}
        onChange={event => onChange(event.target.value === '' ? '' : Number(event.target.value))} />
    </span>
  </label>;
}

export function RangeField({ label, value, onChange, min, max, step }) {
  const labelId = useId();
  return <label className="ic-field ic-range">
    <span id={labelId}>{label}</span>
    <input type="range" aria-labelledby={labelId} value={Number.isFinite(value) ? value : min}
      min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}

export function SelectField({ label, value, onChange, options }) {
  const labelId = useId();
  return <label className="ic-field">
    <span id={labelId}>{label}</span>
    <select aria-labelledby={labelId} value={value} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={text === undefined ? key : key}>{text ?? key}</option>)}
    </select>
  </label>;
}

/** Radio or numeric prediction, unset at first render and never preselected. */
export function Prediction({ state, prompt, choices, numeric, action, evaluate, extra }) {
  const name = useId();
  const inRange = value => Number.isFinite(value)
    && (numeric.min === undefined || value >= numeric.min) && (numeric.max === undefined || value <= numeric.max);
  const fields = numeric?.fields;
  const numericInvalid = Boolean(numeric) && (fields
    ? state.prediction === '' || fields.some(field => (field.required
      ? !inRange(state.prediction[field.key])
      : state.prediction[field.key] !== '' && state.prediction[field.key] !== undefined && !inRange(state.prediction[field.key])))
    : state.prediction === '' || !inRange(state.prediction));
  const invalid = choices ? state.prediction === '' : numericInvalid;
  const announced = Boolean(state.currentResult) && !state.committed && state.prediction === '';
  const resultParagraph = state.result && <p className={`ic-result${state.currentResult ? '' : ' is-history'}`} data-ica-result>
    <strong>{state.currentResult ? 'Applied result. ' : 'Previous result (history, not feedback on the edited draft). '}</strong>
    {state.result.evaluation.message}
  </p>;
  return <div className="ic-prediction">
    <fieldset>
      <legend>{prompt}</legend>
      {choices && <div className="ic-choices">{choices.map(([value, label]) => <label key={value}>
        <input type="radio" name={name} value={value} checked={state.prediction === value} onChange={() => state.choose(value)} />
        {label}
      </label>)}</div>}
      {numeric && fields && <div className="ic-controls">{fields.map(field => <NumberField key={field.key} label={field.label}
        value={state.prediction === '' ? '' : state.prediction[field.key] ?? ''} min={numeric.min} max={numeric.max}
        step={numeric.step ?? 'any'}
        onChange={value => state.choose({
          ...(state.prediction === '' ? Object.fromEntries(fields.map(item => [item.key, ''])) : state.prediction),
          [field.key]: value,
        })} />)}</div>}
      {numeric && !fields && <NumberField label={numeric.label} value={state.prediction} onChange={state.choose}
        min={numeric.min} max={numeric.max} step={numeric.step ?? 'any'} />}
      {extra}
    </fieldset>
    {state.error && <p className="ic-error" role={state.result ? 'alert' : undefined}>
      {state.error}{state.result ? ' The last valid applied result is retained.' : ''}
    </p>}
    {state.notice && <p className="ic-note" data-ica-notice>{state.notice}</p>}
    {numeric && state.prediction !== '' && numericInvalid
      && <p className="ic-error">Enter a finite prediction between {format(numeric.min, 2)} and {format(numeric.max, 2)}.</p>}
    <div className="ic-buttons">
      <button type="button" disabled={state.blocked || invalid} onClick={state.commit}>Commit prediction</button>
      <button className="is-primary" type="button" disabled={state.blocked || !state.committed}
        onClick={() => state.apply(evaluate)}>{action}</button>
    </div>
    <p className="ic-status" role={announced ? undefined : 'status'} aria-live={announced ? 'off' : 'polite'} data-ica-status>{state.status}</p>
    <div aria-live="polite" aria-atomic="true" data-ica-announcement>{state.currentResult ? resultParagraph : null}</div>
    {!state.currentResult && resultParagraph}
  </div>;
}

/** The caption sits outside the scroll region so a wide table never clips it.
 *  `stack` turns the table into one labelled block per row on a narrow screen,
 *  because a silently clipped numeral reads as a complete, wrong number.
 */
export function DataTable({ caption, headings, rows, dense = false, stack = false }) {
  const captionId = useId();
  return <div className="ic-table-block">
    <p className="ic-table-caption" id={captionId}>{caption}</p>
    <div className={`ic-table-wrap${dense ? ' is-dense' : ''}${stack ? ' is-stack' : ''}`}
      tabIndex={0} role="region" aria-labelledby={captionId}>
      <table>
        <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
        <tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, column) => column === 0
          ? <th key={column} scope="row">{cell}</th>
          : <td key={column} data-label={headings[column]}>{cell}</td>)}</tr>)}</tbody>
      </table>
    </div>
  </div>;
}

/** A square drawing area so one x unit is exactly as long as one y unit.
 *  The viewBox matches the rendered CSS width, so 11 SVG units really are 11px.
 */
export function EqualPlot({ title, describe, bounds, children, ticks, size = 300, className = '' }) {
  const titleId = useId();
  const [xLow, xHigh, yLow, yHigh] = bounds;
  const span = Math.max(xHigh - xLow, yHigh - yLow);
  const padLeft = 34;
  const padBottom = 32;
  const padTop = 10;
  // Room for half of the right-most tick label, which is centred on the axis.
  const plot = size - padLeft - 22;
  const centreX = (xLow + xHigh) / 2;
  const centreY = (yLow + yHigh) / 2;
  const sx = value => padLeft + ((value - (centreX - span / 2)) * plot) / span;
  const sy = value => padTop + plot - ((value - (centreY - span / 2)) * plot) / span;
  const tickValues = ticks ?? [centreX - span / 2, centreX, centreX + span / 2];
  return <figure className={`ic-plot ${className}`}>
    <figcaption id={titleId}>{title}</figcaption>
    <svg viewBox={`0 0 ${size} ${plot + padTop + padBottom}`} style={{ maxWidth: `${size}px` }} role="img" aria-labelledby={titleId}>
      <desc>{describe}</desc>
      {tickValues.map(tick => <g key={`x${tick}`}>
        <line className="ic-grid" x1={sx(tick)} x2={sx(tick)} y1={padTop} y2={padTop + plot} />
        <text x={sx(tick)} y={padTop + plot + 22} textAnchor="middle">{format(tick, 2)}</text>
      </g>)}
      {tickValues.map(tick => <g key={`y${tick}`}>
        <line className="ic-grid" x1={padLeft} x2={padLeft + plot} y1={sy(tick)} y2={sy(tick)} />
        <text x={padLeft - 5} y={sy(tick) + 4} textAnchor="end">{format(tick, 2)}</text>
      </g>)}
      {children(sx, sy, { plot, padLeft, padTop })}
    </svg>
  </figure>;
}

export function Readout({ label, value, note }) {
  return <p className="ic-readout"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</p>;
}

/** `resetKey` remounts the field, so Reset really does clear everything. */
export function Reflection({ prompt, resetKey = 0 }) {
  return <label className="ic-field ic-reflection">
    <span>{prompt}</span>
    <textarea key={resetKey} rows={3} placeholder="Write your explanation here; it is not automatically scored." />
  </label>;
}
