import { useId, useState } from 'react';
import './manifold-labs.css';

export const format = (value, places = 5) => {
  if (value === null) return 'No path';
  if (!Number.isFinite(value)) return '—';
  if (value !== 0 && Math.abs(value) < 0.00001) return value.toExponential(2);
  return Number(value.toFixed(places)).toString();
};

// Keep a graded difference visible even when compact readouts round alike.
// Scientific notation for tiny values also needs enough significant digits.
export function comparisonValues(before, after) {
  const compact = [format(before), format(after)];
  if (compact[0] !== compact[1] || Math.abs(after - before) <= 1e-9) return compact;
  return [before, after].map(value => Number(value.toPrecision(12)).toString());
}

const copy = value => JSON.parse(JSON.stringify(value));

/** A prediction is a snapshot of every candidate input, never a remembered radio choice. */
export function useManifoldInvestigation(initial, validate) {
  const [draft, setDraft] = useState(() => copy(initial));
  const [active, setActive] = useState(() => copy(initial));
  const [prediction, setPrediction] = useState('');
  const [committed, setCommitted] = useState(null);
  const [result, setResult] = useState(null);
  const [previous, setPrevious] = useState(null);
  const [resetCount, setResetCount] = useState(0);
  const [status, setStatus] = useState('Prediction not recorded.');
  const key = JSON.stringify(draft);
  const error = validate(draft);
  const invalidate = (message = 'Draft changed. Prediction cleared; the last applied state is unchanged.') => {
    setPrediction('');
    setCommitted(null);
    setResult(current => current ? { ...current, historical: true } : null);
    setStatus(message);
  };
  const edit = changes => {
    setDraft(current => ({ ...current, ...changes }));
    invalidate();
  };
  const choose = value => {
    setPrediction(value);
    setCommitted(null);
    setStatus('Prediction chosen. Commit it before applying the draft.');
  };
  const commit = () => {
    if (error || prediction === '') return;
    setCommitted({ key, inputs: copy(draft), prediction });
    setStatus('Prediction committed to this draft. Apply it to reveal the result.');
  };
  const apply = evaluate => {
    if (error || !committed || committed.key !== key) return;
    const inputs = copy(committed.inputs);
    const evaluation = evaluate(inputs, active, committed.prediction);
    setPrevious({ active: copy(active), result });
    setActive(inputs);
    setResult({ inputs, key, prediction: committed.prediction, evaluation });
    setPrediction('');
    setCommitted(null);
    setStatus('Draft applied. Compare your prediction with the result below.');
  };
  const reset = () => {
    setDraft(copy(initial));
    setActive(copy(initial));
    setPrediction('');
    setCommitted(null);
    setResult(null);
    setPrevious(null);
    setResetCount(count => count + 1);
    setStatus('Starting inputs restored. Prediction and result cleared.');
  };
  const undo = () => {
    if (!previous) return;
    setActive(copy(previous.active));
    setDraft(copy(previous.active));
    setResult(previous.result ? { ...previous.result, historical: true } : null);
    setPrevious(null);
    setPrediction('');
    setCommitted(null);
    setStatus('Previous applied inputs restored. Any previous answer is history; make a new prediction.');
  };
  return {
    draft, active, prediction, committed, result, previous, status, key, error, resetCount,
    currentResult: result && result.key === key && !result.historical ? result : null,
    edit, choose, commit, apply, reset, undo, invalidate,
  };
}

export function Investigation({ kind, title, children, state, onReset }) {
  const titleId = useId();
  return <section className="mf-investigation" data-manifold-lab={kind} aria-labelledby={titleId}>
    <header><h3 id={titleId}>{title}</h3><div className="mf-buttons">
      <button type="button" onClick={onReset ?? state.reset}>Reset</button>
      <button type="button" onClick={state.undo} disabled={!state.previous}>Undo apply</button>
    </div></header>
    {children}
  </section>;
}

export function NumberField({ label, value, onChange, min, max, step = 'any', disabled = false }) {
  return <label className="mf-field"><span>{label}</span><input type="number" value={value}
    min={min} max={max} step={step} disabled={disabled}
    onChange={event => onChange(event.target.value === '' ? '' : Number(event.target.value))} /></label>;
}

export function SelectField({ label, value, onChange, options }) {
  const labelId = useId();
  return <label className="mf-field"><span id={labelId}>{label}</span><select aria-labelledby={labelId} value={value} onChange={event => onChange(event.target.value)}>
    {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
  </select></label>;
}

export function Prediction({ state, prompt, choices, maximum, numericStep = 1, action, evaluate }) {
  const name = useId();
  const resultAnnounced = Boolean(state.currentResult) && !state.committed && state.prediction === '';
  const resultParagraph = state.result && <p className="mf-result" data-manifold-result>
    <strong>{state.currentResult ? 'Applied result. ' : 'Previous result (history). '}</strong>
    {state.result.evaluation.message}
  </p>;
  const predictionInvalid = state.prediction === '' || (maximum !== undefined
    && (!Number.isFinite(state.prediction) || state.prediction < 0 || state.prediction > maximum
      || Math.abs(state.prediction / numericStep - Math.round(state.prediction / numericStep)) > 1e-7));
  return <div className="mf-prediction">
    <fieldset><legend>{prompt}</legend>
      {choices ? <div className="mf-choices">{choices.map(([value, label]) => <label key={value}>
        <input type="radio" name={name} value={value} checked={state.prediction === value}
          onChange={() => state.choose(value)} />{label}
      </label>)}</div> : <NumberField label="Your predicted value" min={0} max={maximum} step={numericStep}
        value={state.prediction} onChange={state.choose} />}
    </fieldset>
    {state.error && <p className="mf-error" role="alert">{state.error} The last valid applied result is retained.</p>}
    {!choices && state.prediction !== '' && predictionInvalid && <p className="mf-error">Enter a prediction from 0 to {maximum}{numericStep === 1 ? ' as a whole number' : ` in steps of ${numericStep}`}.</p>}
    <div className="mf-buttons">
      <button type="button" disabled={Boolean(state.error) || predictionInvalid} onClick={state.commit}>Commit prediction</button>
      <button className="is-primary" type="button" disabled={Boolean(state.error) || !state.committed}
        onClick={() => state.apply(evaluate)}>{action}</button>
    </div>
    <p className="mf-status" role={resultAnnounced ? undefined : 'status'} aria-live={resultAnnounced ? 'off' : 'polite'} data-manifold-status>{state.status}</p>
    <div aria-live="polite" aria-atomic="true" data-manifold-result-announcement>{state.currentResult ? resultParagraph : null}</div>
    {!state.currentResult && resultParagraph}
  </div>;
}

export function DataTable({ caption, headings, rows }) {
  return <div className="mf-table-wrap"><table><caption>{caption}</caption>
    <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
    <tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
  </table></div>;
}

/** A square plotting area guarantees that one x unit has the same length as one y unit. */
export function EqualPlot({ points, title, children, describe, bounds }) {
  const titleId = useId();
  const xValues = points.map(point => point.x);
  const yValues = points.map(point => point.y);
  const xMin = bounds?.[0] ?? Math.min(...xValues);
  const xMax = bounds?.[1] ?? Math.max(...xValues);
  const yMin = bounds?.[2] ?? Math.min(...yValues);
  const yMax = bounds?.[3] ?? Math.max(...yValues);
  const span = Math.max(xMax - xMin, yMax - yMin, 1) * 1.15;
  const centerX = (xMin + xMax) / 2;
  const centerY = (yMin + yMax) / 2;
  const xLow = centerX - span / 2;
  const yLow = centerY - span / 2;
  const sx = value => 58 + (value - xLow) * 230 / span;
  const sy = value => 258 - (value - yLow) * 230 / span;
  const ticksX = [xLow, centerX, xLow + span];
  const ticksY = [yLow, centerY, yLow + span];
  return <figure className="mf-equal-plot">
    <figcaption id={titleId}>{title}</figcaption>
    <svg viewBox="0 0 330 298" role="img" aria-labelledby={titleId}>
      <desc>{describe}</desc>
      {ticksX.map((tick, index) => <g key={`x${index}`}>
        <line className="mf-grid" x1={sx(tick)} x2={sx(tick)} y1={28} y2={258} />
        <text x={sx(tick)} y={278} textAnchor="middle">{format(tick, 1)}</text>
      </g>)}
      {ticksY.map((tick, index) => <g key={`y${index}`}>
        <line className="mf-grid" x1={58} x2={288} y1={sy(tick)} y2={sy(tick)} />
        <text x={50} y={sy(tick) + 4} textAnchor="end">{format(tick, 1)}</text>
      </g>)}
      {children(sx, sy)}
    </svg>
  </figure>;
}

const digitColors = ['#d9b35b', '#90bda4', '#b0b8d7', '#dba697', '#9fc0c9', '#d0b8d2', '#b9bf82', '#a6b3a4', '#c1a17d', '#e2cfaa'];

export function DigitLegend() {
  return <div className="mf-digit-legend" aria-label="Digit color key">{digitColors.map((color, digit) =>
    <span key={digit}><i style={{ background: color }} aria-hidden="true" />Digit {digit}</span>)}</div>;
}

export function DigitScatter({ rows, coordinates, queryId = null, onSelect, neighbors = null, colorLabels = false, title }) {
  const points = coordinates.map(([x, y], index) => ({ x, y, ...rows[index] }));
  const query = points.find(point => point.sourceRow === queryId);
  const inputSet = new Set(neighbors?.input ?? []);
  const mapSet = new Set(neighbors?.map ?? []);
  const relation = id => inputSet.has(id) && mapSet.has(id) ? 'retained' : inputSet.has(id) ? 'input-only' : 'map-only';
  return <div className="mf-digit-scatter">
    <EqualPlot points={points} title={title}
      describe={`300 actual saved observations, with equal scales on both map-coordinate axes.${query ? ` Selected source row ${queryId}.` : ''} Select any source ID using the adjacent control.`}>
      {(sx, sy) => <>
        {query && neighbors && points.filter(point => inputSet.has(point.sourceRow) || mapSet.has(point.sourceRow)).map(point =>
          <line key={`edge${point.sourceRow}`} className={`mf-neighbor-edge is-${relation(point.sourceRow)}`}
            x1={sx(query.x)} y1={sy(query.y)} x2={sx(point.x)} y2={sy(point.y)} />)}
        {points.map(point => <circle key={point.sourceRow} cx={sx(point.x)} cy={sy(point.y)}
          r={point.sourceRow === queryId ? 4.5 : 2.1} className={`mf-digit-point${point.sourceRow === queryId ? ' is-query' : ''}`}
          fill={colorLabels ? digitColors[point.digit] : '#a8b7ae'}
          onClick={onSelect ? () => onSelect(point.sourceRow) : undefined}>
          <title>{`Source ${point.sourceRow}${colorLabels ? ` · digit ${point.digit}` : ''}`}</title>
        </circle>)}
        {query && <circle cx={sx(query.x)} cy={sy(query.y)} r={7} className="mf-query-ring" />}
      </>}
    </EqualPlot>
    <p className="mf-caption">Map coordinates · equal axis scale within this map{query ? ` · selected source ${queryId}` : ''}.</p>
  </div>;
}

export function DigitTile({ row, caption, showLabel = true }) {
  return <figure className="mf-digit-tile">
    <svg viewBox="0 0 80 80" role="img" aria-label={`Observed 8 by 8 pixels from source ${row.sourceRow}${showLabel ? `, digit ${row.digit}` : ''}`}>
      {row.pixels.map((value, index) => <rect key={index} x={(index % 8) * 10} y={Math.floor(index / 8) * 10}
        width={10} height={10} fill={`rgb(${Math.round(value * 255 / 16)},${Math.round(value * 255 / 16)},${Math.round(value * 255 / 16)})`} />)}
    </svg>
    <figcaption>{caption ?? `Source ${row.sourceRow}${showLabel ? ` · digit ${row.digit}` : ''}`}</figcaption>
  </figure>;
}

export function Reflection({ prompt }) {
  return <label className="mf-field mf-reflection"><span>{prompt}</span><textarea rows={3} placeholder="Write your explanation here; it is not automatically scored." /></label>;
}
