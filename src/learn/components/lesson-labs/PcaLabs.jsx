import { cloneElement, isValidElement, useId, useState } from 'react';
import { fourPoints, editorBound, pointLimit, projectAtAngle, principalDirections, rectangleMetric, labelCollisions, wineValidationCurve, smallestComponentCount, wineReconstruction } from '../../data/pca-models';
import { wineFeatures, wineSplit, wineCultivar } from '../../data/pca-wine-data';
import './pca-labs.css';

const number = (value, digits = 4) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (Number.isInteger(value)) return String(value);
  const fixed = value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
  return fixed === '-0' ? '0' : fixed;
};
const percent = value => `${number(100 * value, 2)}%`;
const pointName = index => String.fromCharCode(65 + index);
const coordinate = point => `(${number(point[0])}, ${number(point[1])})`;

export function Investigation({ title, question, children, onReset }) {
  const id = useId();
  return <section className="pca-investigation" aria-labelledby={id} data-live-exploration>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="pca-question">{question}</p>}
    {children}
  </section>;
}

/** A labeled control. The explicit id binds the label to the control rather
 * than to the labelable <output> that displays the current value. */
export function Field({ label, children, value }) {
  const id = useId();
  const control = isValidElement(children) && typeof children.type === 'string' && ['input', 'select'].includes(children.type) ? cloneElement(children, { id }) : children;
  return <label className="pca-field" htmlFor={id}><span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>{control}</label>;
}
export function Table({ caption, headings, rows, highlight = () => false }) {
  return <div className="pca-table-scroll" role="region" aria-label={caption} tabIndex={0}>
    <table><caption>{caption}</caption><thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index} className={highlight(index) ? 'is-selected' : undefined}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}
/** Square plot with equal units on both axes. `children(project)` draws in
 * a 0..320 viewBox; the domain is padded around the supplied points. */
export function SquarePlot({ points, extra = [], title, xLabel = 'first reading', yLabel = 'second reading', children, describe }) {
  const all = [...points, ...extra];
  const minima = [0, 1].map(axis => Math.min(...all.map(point => point[axis])));
  const maxima = [0, 1].map(axis => Math.max(...all.map(point => point[axis])));
  const span = Math.max(1, ...[0, 1].map(axis => maxima[axis] - minima[axis]));
  const centers = [0, 1].map(axis => (minima[axis] + maxima[axis]) / 2);
  const extent = span * 1.4;
  const domain = centers.map(center => center - extent / 2);
  const project = point => [32 + 256 * (point[0] - domain[0]) / extent, 288 - 256 * (point[1] - domain[1]) / extent];
  const tick = value => String(Number(value.toPrecision(3)));
  return <figure className="pca-plot">
    {title && <figcaption>{title}</figcaption>}
    <div className="pca-square">
      <svg viewBox="0 0 320 320" role="img" aria-label={describe ?? title}>
        {[32, 160, 288].map(position => <path key={position} className="pca-grid" d={`M32,${position}H288 M${position},32V288`} />)}
        {children(project, domain, extent)}
      </svg>
      {[0, 0.5, 1].map(fraction => <span key={`x${fraction}`} className="pca-tick is-x" style={{ left: `${10 + 80 * fraction}%` }}>{tick(domain[0] + fraction * extent)}</span>)}
      {[0, 0.5, 1].map(fraction => <span key={`y${fraction}`} className="pca-tick is-y" style={{ top: `${90 - 80 * fraction}%` }}>{tick(domain[1] + fraction * extent)}</span>)}
    </div>
    <div className="pca-axis-caption"><span>horizontal: {xLabel}</span><span>vertical: {yLabel}</span></div>
  </figure>;
}
/** Draw a line through `mean` with the given unit direction, clipped to the padded domain. */
export function lineThrough(mean, direction, domain, extent, project) {
  // Clip the infinite line to the visible square so it never spills outside the plot.
  const low = domain, high = [domain[0] + extent, domain[1] + extent];
  const parameters = [];
  [0, 1].forEach(axis => {
    if (Math.abs(direction[axis]) < 1e-12) return;
    for (const bound of [low[axis], high[axis]]) {
      const t = (bound - mean[axis]) / direction[axis];
      const other = mean[1 - axis] + t * direction[1 - axis];
      if (other >= low[1 - axis] - 1e-9 && other <= high[1 - axis] + 1e-9) parameters.push(t);
    }
  });
  const tStart = Math.min(...parameters), tEnd = Math.max(...parameters);
  const [x1, y1] = project([mean[0] + tStart * direction[0], mean[1] + tStart * direction[1]]);
  const [x2, y2] = project([mean[0] + tEnd * direction[0], mean[1] + tEnd * direction[1]]);
  return { x1, y1, x2, y2 };
}
/** One-dimensional strip of scores with coincident labels stacked, not jittered. */
export function ScoreStrip({ scores, names, caption }) {
  const magnitude = Math.max(1, ...scores.map(Math.abs)) * 1.25;
  const position = score => 50 + 50 * score / magnitude;
  const groups = new Map();
  scores.forEach((score, index) => {
    const key = score.toFixed(9);
    if (!groups.has(key)) groups.set(key, { score, names: [] });
    groups.get(key).names.push(names[index]);
  });
  return <div>
    <p className="pca-legend">{caption}</p>
    <div className="pca-strip" role="img" aria-label={`Scores on the chosen direction: ${[...groups.values()].map(group => `${group.names.join(' and ')} at ${number(group.score)}`).join('; ')}.`}>
      <span className="pca-strip-axis" />
      <span className="pca-strip-zero" style={{ left: '50%' }} />
      {[...groups.values()].map(group => <span key={group.score}><span className="pca-strip-mark" style={{ left: `${position(group.score)}%` }} /><span className="pca-strip-label" style={{ left: `${position(group.score)}%` }}>{group.names.join('/')}<br />{number(group.score)}</span></span>)}
    </div>
  </div>;
}

const clampAngle = value => ((value % 180) + 180) % 180;

export function PcaProjectionLab() {
  const [points, setPoints] = useState(fourPoints.map(point => [...point]));
  const [history, setHistory] = useState([]);
  const [reference, setReference] = useState(0);
  const [proposed, setProposed] = useState(45);
  const [selected, setSelected] = useState(0);

  const [shift, setShift] = useState([2, -1]);
  
  const current = projectAtAngle(points, proposed);
  const baseline = projectAtAngle(points, reference);
  const fit = principalDirections(points);
  
  const remember = () => setHistory([...history.slice(-11), { points: points.map(point => [...point]), reference, proposed, selected }]);
  const editPoint = (axis, raw) => {
    if (String(raw).trim() === '' || String(raw).trim() === '-') return; // still typing
    const value = Number(raw);
    if (!Number.isFinite(value) || Math.abs(value) > editorBound) return;
    remember();
    setPoints(points.map((point, index) => index === selected ? point.map((old, which) => which === axis ? value : old) : point));
  };
  const reset = () => {
    setPoints(fourPoints.map(point => [...point]));
    setHistory([]); setReference(0); setProposed(45); setSelected(0);   setShift([2, -1]);
  };
  return <Investigation title="Find the most useful ruler" question="Turning the ruler changes which differences survive as one number. Turn the direction slider and compare its squared error with the reference. Then change the data yourself and see where the best direction goes." onReset={reset}>
    <div className="pca-controls">
      <Field label="Reference angle (degrees from the first axis)" value={`${number(reference, 2)}°`}><input type="range" min="0" max="179" step="1" value={reference} onChange={event => { remember(); setReference(Number(event.target.value)); }} /></Field>
      <Field label="Proposed angle" value={`${number(proposed, 2)}°`}><input type="range" min="0" max="180" step="any" value={proposed} onChange={event => { remember(); setProposed(Number(event.target.value)); }} /></Field>
    </div>
    
    <p className="pca-readout" aria-live="polite">Mean {coordinate(current.mean)}. Total centered squared length {number(current.total)} = retained {number(current.retained)} + residual {number(current.sse)}. {`Retained fraction ${current.retainedFraction === null ? 'undefined (no variation)' : percent(current.retainedFraction)}.`}</p>
    <p className="pca-readout">Reference at {number(reference, 2)}°: squared error {number(baseline.sse)}. Current direction at {number(proposed, 2)}°: {number(current.sse)}. The two lines use the same observations and mean, so their residual sums compare only the chosen direction.</p>
    <SquarePlot points={points} extra={[current.mean]} title="Original coordinates: the ruler passes through the mean" describe={`${points.length} points, a line through the mean at ${proposed}° with perpendicular feet, and the selected point ${pointName(selected)} with its residual segment. Exact values are in the table below.`}>
      {(project, domain, extent) => <>
        <line className="pca-line is-reference" {...lineThrough(current.mean, baseline.direction, domain, extent, project)} />
        <line className="pca-line" {...lineThrough(current.mean, current.direction, domain, extent, project)} />
        {points.map((point, index) => {
          const [px, py] = project(point), [fx, fy] = project(current.projections[index]);
          return <g key={index}>
            <line className="pca-residual" x1={px} y1={py} x2={fx} y2={fy} strokeOpacity={index === selected ? 1 : 0.45} />
            <circle className="pca-foot" cx={fx} cy={fy} r="3" />
            <circle className={`pca-point ${index === selected ? 'is-selected' : ''}`} cx={px} cy={py} r="4.5" />
            <text x={px + 7} y={py - 6}>{pointName(index)}</text>
          </g>;
        })}
        {(() => { const [mx, my] = project(current.mean); return <g><circle className="pca-mean" cx={mx} cy={my} r="6" /><text x={mx + 9} y={my + 14}>mean</text></g>; })()}
      </>}
    </SquarePlot>
    <ScoreStrip scores={current.scores} names={points.map((_, index) => pointName(index))} caption="Scores on the proposed direction: one number per observation, zero at the mean. Coincident scores are stacked, not jittered." />
    <div className="pca-controls">
      <Field label="Selected observation"><select value={selected} onChange={event => setSelected(Number(event.target.value))}>{points.map((point, index) => <option key={index} value={index}>{pointName(index)} {coordinate(point)}</option>)}</select></Field>
      <Field label={`${pointName(selected)} first reading (−${editorBound}..${editorBound})`}><input type="number" step="0.5" min={-editorBound} max={editorBound} value={points[selected][0]} onChange={event => editPoint(0, event.target.value)} /></Field>
      <Field label={`${pointName(selected)} second reading`}><input type="number" step="0.5" min={-editorBound} max={editorBound} value={points[selected][1]} onChange={event => editPoint(1, event.target.value)} /></Field>
    </div>
    <div className="pca-buttons">
      <button type="button" disabled={points.length >= pointLimit} onClick={() => { remember(); setPoints([...points, current.mean.map(value => Math.round(value * 2) / 2)]); setSelected(points.length); }}>Add a point at the mean</button>
      <button type="button" disabled={points.length <= 4} onClick={() => { remember(); setPoints(points.filter((_, index) => index !== selected)); setSelected(0); }}>Remove selected</button>
      <button type="button" disabled={fit.degenerate} onClick={() => { remember(); setProposed(clampAngle(fit.angles[0])); }}>Fit best direction</button>
      <button type="button" disabled={history.length === 0} onClick={() => { const last = history.at(-1); setHistory(history.slice(0, -1)); setPoints(last.points); setReference(last.reference); setProposed(last.proposed); setSelected(last.selected); }}>Back</button>
    </div>
    <div className="pca-controls">
      <Field label="Translate every point by (Δ first, Δ second)"><div className="pca-prediction-row"><input type="number" step="1" min={-5} max={5} aria-label="Shift of the first reading" value={shift[0]} onChange={event => setShift([Number(event.target.value), shift[1]])} /><input type="number" step="1" min={-5} max={5} aria-label="Shift of the second reading" value={shift[1]} onChange={event => setShift([shift[0], Number(event.target.value)])} /><button type="button" onClick={() => { const moved = points.map(point => [point[0] + shift[0], point[1] + shift[1]]); if (moved.every(point => point.every(value => Math.abs(value) <= editorBound))) { remember(); setPoints(moved); } }}>Apply translation</button></div></Field>
    </div>
    <Table caption={`Selected direction at ${proposed}°: centered vectors, scores, projections and residuals`} headings={['point', 'original', 'centered', 'score', 'projection', 'residual²']} rows={points.map((point, index) => [pointName(index), coordinate(point), coordinate(current.centered[index]), number(current.scores[index]), coordinate(current.projections[index]), number(current.residuals[index][0] ** 2 + current.residuals[index][1] ** 2)])} highlight={index => index === selected} />
    <p className="pca-caption">{fit.degenerate ? 'Every point is identical: the mean reconstructs all of them and no direction is preferred; a retained fraction would divide zero by zero.' : fit.tie ? 'The two sample variances are equal in every direction: this cloud has no preferred axis, so every angle gives the same loss.' : `The exact best direction for these points is ${number(clampAngle(fit.angles[0]), 1)}°, with score variance ${number(fit.eigenvalues[0])} and second variance ${number(fit.eigenvalues[1])}. Press “Fit best direction” after comparing to move the ruler there.`} Translating all points moves the mean and leaves every centered vector, variance and loss unchanged.</p>
  </Investigation>;
}

export function PcaMetricLab() {
  const initial = { a: 2, b: 1, multiplier: 1, standardized: false };
  const [pending, setPending] = useState(initial);
  const applied = pending;

  const state = rectangleMetric(applied.a, applied.b, applied.multiplier, applied.standardized);

  const update = patch => { setPending({ ...pending, ...patch });  };
  
  const axisLabel = { first: 'the first axis', second: 'the second axis', tie: 'no preferred axis (a tie)' };
  return <Investigation title="Units define the metric" question="The four corners of a rectangle have no diagonal structure. Change only the unit of the second reading, or standardize, and see which axis PCA prefers and when it cannot prefer either." onReset={() => { setPending(initial);    }}>
    <div className="pca-controls">
      <Field label="Rectangle half-width a" value={number(pending.a)}><input type="range" min="0.25" max="4" step="0.25" value={pending.a} onChange={event => update({ a: Number(event.target.value) })} /></Field>
      <Field label="Rectangle half-height b" value={number(pending.b)}><input type="range" min="0.25" max="4" step="0.25" value={pending.b} onChange={event => update({ b: Number(event.target.value) })} /></Field>
      <Field label="Unit multiplier on the second reading" value={`× ${number(pending.multiplier)}`}><input type="range" min="0.25" max="10" step="0.25" value={pending.multiplier} onChange={event => update({ multiplier: Number(event.target.value) })} /></Field>
      <Field label="Same multiplier, typed (0.25 to 10)"><input type="number" min="0.25" max="10" step="0.01" value={pending.multiplier} onChange={event => { const value = Number(event.target.value); if (event.target.value.trim() !== '' && value >= 0.25 && value <= 10) update({ multiplier: value }); }} /></Field>
      <Field label="Coordinate geometry"><select value={pending.standardized ? 'standardized' : 'raw'} onChange={event => update({ standardized: event.target.value === 'standardized' })}><option value="raw">Raw readings</option><option value="standardized">Standardized (each column divided by its fitted standard deviation)</option></select></Field>
    </div>
    
    <p className="pca-readout" aria-live="polite">Applied geometry: corners (±{number(state.a)}, ±{number(state.b)} × {number(state.multiplier)}){state.standardized ? ', then each column divided by its scale (' + state.scales.map(number).join(', ') + ')' : ''}. {`Axis variances ${number(state.variances[0])} and ${number(state.variances[1])}: PC1 follows ${axisLabel[state.leadingAxis]}${state.leadingAxis === 'tie' ? '' : `, retaining ${percent(Math.max(...state.fractions))}`}.`}</p>
    <SquarePlot points={state.points} title={state.standardized ? 'Standardized coordinates: equal spread by construction' : 'Raw coordinates with equal-length units on both axes'} xLabel={state.standardized ? 'first reading in fitted standard deviations' : 'first reading'} yLabel={state.standardized ? 'second reading in fitted standard deviations' : `second reading × ${number(state.multiplier)}`} describe={`Four corners at ${state.points.map(coordinate).join(', ')}. ${`PC1 follows ${axisLabel[state.leadingAxis]}.`}`}>
      {(project, domain, extent) => <>
        {state.leadingAxis !== 'tie' && <line className="pca-line" {...lineThrough([0, 0], state.fit.directions[0], domain, extent, project)} />}
        {state.leadingAxis === 'tie' && [0, 45, 90, 135].map(angle => <line key={angle} className="pca-line is-reference" {...lineThrough([0, 0], [Math.cos(angle * Math.PI / 180), Math.sin(angle * Math.PI / 180)], domain, extent, project)} />)}
        {state.points.map((point, index) => { const [x, y] = project(point); return <g key={index}><circle className="pca-point" cx={x} cy={y} r="4.5" /><text x={x + 7} y={y - 6}>{pointName(index)}</text></g>; })}
      </>}
    </SquarePlot>
    {<div className="pca-bars" role="img" aria-label={`Variance along the first axis ${number(state.variances[0])}, along the second axis ${number(state.variances[1])}.`}>
      {['first axis', 'second axis'].map((label, axis) => <div key={label} className="pca-bar-row"><span>variance, {label}</span><span className="pca-bar-track"><span className={`pca-bar-fill ${axis === 1 ? 'is-second' : ''}`} style={{
        width: `${100 * state.variances[axis] / Math.max(...state.variances)}%`
      }} /></span><span className="pca-bar-value">{number(state.variances[axis])} · {percent(state.fractions[axis])}</span></div>)}
    </div>}
    <p className="pca-caption">A tie happens when the two axis variances are equal: in raw geometry exactly when the multiplier equals a/b, and in standardized geometry always, because standardization gives every column unit variance. Variances that agree to within 0.01% are reported as a tie, so a multiplier typed to a few decimals, such as 2.6667 for a/b = 8/3, reads as the tie it is meant to be. At a tie the dashed lines show four of the infinitely many equally good directions; a library still returns one of them, but that choice is arbitrary. Multiplying both readings by the same factor changes both variances by its square and changes no direction or ratio.</p>
  </Investigation>;
}

const curve = wineValidationCurve();
export function PcaBudgetLab() {
  const [budget, setBudget] = useState(0.10);

  const [exploreK, setExploreK] = useState(null);
  const [position, setPosition] = useState(0);
  const [feature, setFeature] = useState(6);
  const selection = smallestComponentCount(curve.ratios, budget);

  const k = exploreK ?? (selection.k);
  const record = wineReconstruction(position, k);
  const xPosition = index => 40 + index * 20;
  const yPosition = ratio => 185 - 160 * ratio;
  return <Investigation title="Choose a component budget, then inspect the loss" question="Fitted on 133 training wines, scored on 45 validation wines. Set the fraction of the mean-only error you are willing to keep, watch the smallest sufficient count move along the curve and look at one wine's recovered measurements." onReset={() => { setBudget(0.10);   setExploreK(null); setPosition(0); setFeature(6); }}>
    <div className="pca-controls">
      <Field label="Allowed validation error as a fraction of the mean-only baseline" value={number(budget, 2)}><input type="range" min="0.01" max="0.6" step="0.01" value={budget} onChange={event => { setBudget(Number(event.target.value)); setExploreK(null); }} /></Field>
      <Field label="Same budget, typed"><input type="number" min="0.01" max="0.6" step="0.01" value={budget} onChange={event => { const value = Number(event.target.value); if (value >= 0.01 && value <= 0.6) { setBudget(value); setExploreK(null); } }} /></Field>
    </div>
    
    {<figure className="pca-curve">
      <figcaption>Validation error as a fraction of the training-mean baseline, k = 0 to 13 (fixed 133/45 split, training-only standardization and directions)</figcaption>
      <svg viewBox="0 0 320 228" role="img" aria-label={`Validation loss ratio by component count: ${curve.ratios.map((ratio, index) => `k ${index}: ${number(ratio, 3)}`).join('; ')}. Budget ${number(budget, 2)} first met at k = ${selection.k}.`}>
        {[0, 0.5, 1].map(value => <g key={value}><line x1="40" x2="300" y1={yPosition(value)} y2={yPosition(value)} className="pca-grid" /><text x="34" y={yPosition(value) + 4} textAnchor="end">{value}</text></g>)}
        <line x1="40" x2="300" y1={yPosition(budget)} y2={yPosition(budget)} stroke="#e7b94a" strokeDasharray="5 4" />
        <text x="298" y={yPosition(budget) - 4} fill="#e7b94a" textAnchor="end">budget {number(budget, 2)}</text>
        <polyline points={curve.ratios.map((ratio, index) => `${xPosition(index)},${yPosition(ratio)}`).join(' ')} fill="none" stroke="#91aecf" strokeWidth="2" />
        {curve.ratios.map((ratio, index) => <g key={index}>
          <circle cx={xPosition(index)} cy={yPosition(ratio)} r={index === selection.k ? 5 : 3.5} fill={index === selection.k ? '#e7b94a' : index === k ? '#f2e7ca' : '#91aecf'} />
          <text x={xPosition(index)} y="202" textAnchor="middle">{index}</text>
        </g>)}
        <text x="170" y="222" textAnchor="middle">k · number of components kept</text>
      </svg>
    </figure>}
    {<div className="pca-controls">
      <Field label="Explore a different k for the record below" value={String(k)}><input type="range" min="0" max="13" step="1" value={k} onChange={event => setExploreK(Number(event.target.value))} /></Field>
      <Field label="Validation wine"><select value={position} onChange={event => setPosition(Number(event.target.value))}>{wineSplit.validation.map((rowIndex, index) => <option key={rowIndex} value={index}>row {rowIndex} (cultivar {wineCultivar[rowIndex]})</option>)}</select></Field>
      <Field label="Feature to read in original units"><select value={feature} onChange={event => setFeature(Number(event.target.value))}>{wineFeatures.map((name, index) => <option key={name} value={index}>{name}</option>)}</select></Field>
    </div>}
    {<p className="pca-readout" aria-live="polite">Row {record.rowIndex} with {k} component{k === 1 ? '' : 's'}: {wineFeatures[feature]} measured {number(record.original[feature])}, recovered {number(record.rebuiltOriginal[feature])}; this wine's squared error over all 13 standardized features is {number(record.squaredError)} (validation average at this k: {number(curve.ratios[k] * curve.baselineMse * 13)}).</p>}
    {<Table caption={`Row ${record.rowIndex}: original, recovered and standardized residual for each feature at k = ${k}`} headings={['feature', 'original', 'recovered (original units)', 'standardized residual']} rows={wineFeatures.map((name, index) => [name, number(record.original[index]), number(record.rebuiltOriginal[index]), number(record.residuals[index])])} highlight={index => index === feature} />}
    <details><summary>Training cumulative variance for comparison (a different quantity)</summary>
      <Table caption="Fraction of training variance retained by the first k components; this is not the validation error above" headings={['k', 'cumulative training variance']} rows={wineSplit.trainingRatios.map((_, index) => [index + 1, percent(wineSplit.trainingRatios.slice(0, index + 1).reduce((sum, value) => sum + value, 0))])} />
    </details>
    <p className="pca-caption">The curve steps down as k grows and can never rise, because each added direction removes a nonnegative residual from every wine. That is why a budget is needed: minimizing validation error alone selects all 13. Budgets of 0.10 and 0.11 select the same 8 components; 0.06 selects 10. At k = 13 every validation wine is recovered to machine precision, which says only that thirteen directions span thirteen features. An acceptable average can still hide a badly recovered measurement, so read one row before trusting the curve.</p>
  </Investigation>;
}

export function PcaTaskInformationLab() {
  const [a, setA] = useState(10);
  const [b, setB] = useState(1);
  const [labelAxis, setLabelAxis] = useState('y');
  const [kept, setKept] = useState('pc1');

  const state = labelCollisions(a, b, labelAxis, kept);
  const change = setter => value => { setter(value);  };
  const keptLabel = { pc1: 'PC1 only', pc2: 'PC2 only', both: 'both components' };
  return <Investigation title="Variance is not the label" question="Four observations, two classes. Keep one principal coordinate and ask whether observations with different labels still land at different places. Then change which coordinate defines the labels: the fit does not move, but its usefulness does." onReset={() => { setA(10); setB(1); setLabelAxis('y'); setKept('pc1');   }}>
    <div className="pca-controls">
      <Field label="Horizontal spread a" value={number(a)}><input type="range" min="2" max="12" step="0.5" value={a} onChange={event => change(setA)(Number(event.target.value))} /></Field>
      <Field label="Vertical spread b" value={number(b)}><input type="range" min="0.25" max="1.5" step="0.25" value={b} onChange={event => change(setB)(Number(event.target.value))} /></Field>
      <Field label="Label rule"><select value={labelAxis} onChange={event => change(setLabelAxis)(event.target.value)}><option value="y">Class by the sign of the second coordinate</option><option value="x">Class by the sign of the first coordinate</option></select></Field>
      <Field label="Retained components"><select value={kept} onChange={event => change(setKept)(event.target.value)}><option value="pc1">PC1 only</option><option value="pc2">PC2 only</option><option value="both">Both</option></select></Field>
    </div>
    
    <p className="pca-readout" aria-live="polite">PC1 is the first axis with variance {number(state.fit.eigenvalues[0])}; PC2 has {number(state.fit.eigenvalues[1])}. Retaining {keptLabel[kept]} keeps {percent(state.retainedFraction)} of the variance. {state.distinguishable ? 'Labels remain distinguishable.' : 'Two pairs with different labels now share one coordinate.'}</p>
    <SquarePlot points={state.points} title="All four observations in both coordinates (equal units; the inset magnifies the vertical separation)" describe={`Corners at ±${number(a)} horizontally and ±${number(b)} vertically; class A shown as circles, class B as squares.`}>
      {(project) => <>
        {state.points.map((point, index) => { const [x, y] = project(point); return <g key={index}>{state.labels[index] === 'A' ? <circle className="pca-point" cx={x} cy={y} r="5" /> : <rect className="pca-point" x={x - 5} y={y - 5} width="10" height="10" />}<text x={x + 8} y={y - 7}>{pointName(index)}·{state.labels[index]}</text></g>; })}
        <g transform="translate(196 186)">
          <rect x="0" y="0" width="100" height="100" fill="#0b0f10" stroke="#746138" />
          {state.points.filter(point => point[0] > 0).map((point, index) => { const y = 50 - 30 * point[1] / b; const realIndex = state.points.indexOf(point); return <g key={index}>{state.labels[realIndex] === 'A' ? <circle className="pca-point" cx="50" cy={y} r="5" /> : <rect className="pca-point" x="45" y={y - 5} width="10" height="10" />}<text x="60" y={y + 4}>{pointName(realIndex)}·{state.labels[realIndex]}</text></g>; })}
          <text x="4" y="96" fill="#e7b94a">C, D · ×{number(30 / b / (256 / (Math.max(2 * a, 2 * b) * 1.4)), 0)} vertical</text>
        </g>
      </>}
    </SquarePlot>
    <ScoreStrip scores={state.coordinates.map(coordinate => coordinate[0])} names={state.points.map((_, index) => `${pointName(index)}·${state.labels[index]}`)} caption={kept === 'both' ? 'First retained coordinate (PC1). With both components kept, the second coordinate separates any pair that ties here.' : `Retained coordinate on ${keptLabel[kept]}. Stacked labels share exactly one score.`} />
    {kept === 'both' && <ScoreStrip scores={state.coordinates.map(coordinate => coordinate[1])} names={state.points.map((_, index) => `${pointName(index)}·${state.labels[index]}`)} caption="Second retained coordinate (PC2)." />}
    <Table caption="Observations, labels and retained coordinates" headings={['point', 'coordinates', 'class', 'retained coordinate(s)']} rows={state.points.map((point, index) => [pointName(index), coordinate(point), state.labels[index], state.coordinates[index].map(value => number(value)).join(', ')])} />
    <p className="pca-caption">With labels from the second coordinate, PC1 alone collides at every horizontal spread you can set: a larger a raises PC1's variance share and changes nothing about the collision, while PC2 alone, with about {percent(state.fit.fractions[1])} of the variance, separates the classes perfectly. Switching the label rule to the first coordinate reverses which single component is useful without changing a single eigenvalue. This is distinguishability of four constructed points, not a classifier's accuracy on data.</p>
  </Investigation>;
}
