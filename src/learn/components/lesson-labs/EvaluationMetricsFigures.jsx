import { useRef, useState } from 'react';
import { METRIC_ROWS, RESIDUAL_ROWS, DOCUMENT_ROWS, binaryMetrics, rankingMetrics, probabilityMetrics, regressionMetrics, retrievalMetrics, displayMetric as fmt } from '../../data/evaluation-metrics-models';
import measured from '../../data/evaluation-metrics-data.json';
import './evaluation-metrics.css';

export function MetricFigure({ title, children, caption, id }) {
  return <figure className="metric-figure" data-metric-figure={id}><figcaption><strong>{title}</strong>{caption && <span>{caption}</span>}</figcaption>{children}</figure>;
}
export function MetricTable({ label, headings, rows }) {
  return <div className="metric-table-scroll" tabIndex={0} role="region" aria-label={label}><table><thead><tr>{headings.map((h, i) => <th key={i} scope="col">{h}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((value, j) => j === 0 ? <th key={j} scope="row">{value}</th> : <td key={j}>{value}</td>)}</tr>)}</tbody></table></div>;
}
export function PredictionContracts() {
  return <MetricFigure id="contracts" title="One prediction can answer different questions" caption="Observed outcomes are required on every route. A real-valued score is not automatically a probability."><div className="metric-contracts">{[
    ['Score → order', 'Keep the unthresholded score', 'Compare positive and negative outcomes', 'ROC-AUC / AP'],
    ['Score → gate → action', 'Choose a threshold and a positive class', 'Compare observed and predicted classes', 'Precision / recall / cost'],
    ['Probability → forecast loss', 'Require a probability interpretation in [0, 1]', 'Use the probability of the outcome that occurred', 'Log loss / Brier'],
    ['Numerical value → residual', 'Keep the target’s units', 'Subtract prediction from observed value', 'MAE / RMSE / R²'],
  ].map(([name, input, outcome, metric]) => <div key={name}><h4>{name}</h4><p>{input}</p><span aria-hidden="true">↓</span><p>{outcome}</p><strong>{metric}</strong></div>)}</div></MetricFigure>;
}
export function ScoreBlocks({ rows, threshold = 0.8, activeIds = [] }) {
  const groups = rankingMetrics(rows).groups;
  const min = Math.min(...rows.map(r => r.score), Number.isFinite(threshold) ? threshold : Infinity);
  const max = Math.max(...rows.map(r => r.score), Number.isFinite(threshold) ? threshold : -Infinity);
  const x = score => max === min ? 200 : 12 + 376 * (score - min) / (max - min);
  return <><div className="metric-score-ruler"><svg viewBox="0 0 400 76" role="img" aria-label={`Shared score ruler from ${min} to ${max}. Positive observations in the upper lane, negative observations in the lower lane; decision gate ${Number.isFinite(threshold) ? threshold : 'above all scores'}.`}><line x1="12" y1="38" x2="388" y2="38" className="metric-axis"/>{Number.isFinite(threshold) && <line x1={x(threshold)} y1="5" x2={x(threshold)} y2="71" className="metric-gate-line"/>}{groups.flatMap(group => group.rows.map((r, i) => <g key={r.id}><title>{r.id}: score {r.score}, actual {r.y}</title>{r.y ? <circle cx={x(r.score)} cy={19 - (i % 2) * 5} r="5" fill="#a6c8b6"/> : <rect x={x(r.score) - 5} y={51 + (i % 2) * 5} width="10" height="10" fill="#d5b785"/>}</g>))}</svg><p>Score scale {fmt(min)} → {fmt(max)} · dashed line: decision gate {Number.isFinite(threshold) ? fmt(threshold) : 'above all scores'}<br/>● upper lane: actual positive · ■ lower lane: actual negative. At or right of the gate is flagged. Exact IDs and ties are grouped below.</p></div><div className="metric-score-blocks" aria-label="Score groups, descending">{groups.map(group => <div key={group.score} className={`metric-score-block ${group.score >= threshold ? 'admitted' : ''} ${group.rows.some(r => activeIds.includes(r.id)) ? 'current' : ''}`}><span>score {fmt(group.score)}</span><div>{group.rows.map(row => <span className={`metric-item ${row.y ? 'positive' : 'negative'}`} key={row.id}>{row.id} · y={row.y}</span>)}</div><small>{group.score >= threshold ? '≥ gate: positive decision' : '< gate: negative decision'}</small></div>)}</div></>;
}
export function ConfusionBins({ result, highlight = 'none' }) {
  const highlighted = { precision: ['tp', 'fp'], recall: ['tp', 'fn'], fpr: ['fp', 'tn'], none: [] }[highlight];
  const cell = key => <div className={`metric-bin ${highlighted.includes(key) ? 'denominator' : ''}`}><strong>{key.toUpperCase()} = {result[key]}</strong><span>{result.cells[key].join(', ') || 'No items'}</span></div>;
  return <div className="metric-confusion"><div aria-hidden="true"/><strong>Predicted −</strong><strong>Predicted +</strong><strong>Actual −</strong>{cell('tn')}{cell('fp')}<strong>Actual +</strong>{cell('fn')}{cell('tp')}</div>;
}
export function ThresholdCards() {
  const [highlight, setHighlight] = useState('precision');
  return <MetricFigure id="threshold" title="Eight named items, one gate at 0.8" caption="Equal scores move together. Text and observed labels identify the cards independently of color."><ScoreBlocks rows={METRIC_ROWS}/><ConfusionBins result={binaryMetrics(METRIC_ROWS, 0.8)} highlight={highlight}/><label className="metric-field">Highlight the denominator<select aria-label="Highlight the denominator" value={highlight} onChange={e => setHighlight(e.target.value)}><option value="precision">Precision: positive decisions</option><option value="recall">Recall: actual positives</option><option value="fpr">FPR: actual negatives</option><option value="none">No highlight</option></select></label></MetricFigure>;
}
export function PrevalenceTrays() {
  return <MetricFigure id="prevalence" title="The same rates create very different alert trays" caption="Constructed populations of 10,000. Hold TPR = 0.8 and FPR = 0.1 fixed; only prevalence changes."><div className="metric-paired">{[0.5, 0.01].map(p => { const tp = 10000 * p * 0.8, fp = 10000 * (1 - p) * 0.1; return <section key={p}><h4>{p * 100}% positive prevalence</h4><p>{10000 * p} actual positives → {tp} true alerts<br/>{10000 * (1 - p)} actual negatives → {fp} false alerts</p><div className="metric-alert-tray" role="img" aria-label={`${tp} true and ${fp} false alerts`}><span style={{ width: `${100 * tp / (tp + fp)}%` }}/><span style={{ width: `${100 * fp / (tp + fp)}%` }}/></div><p><span className="metric-key positive">True alerts: {tp}</span><br/><span className="metric-key negative">False alerts: {fp}</span></p><p>Missed positives: {10000 * p - tp}; correctly left-alone negatives: {10000 * (1 - p) - fp}.</p><strong>Precision = {tp}/{tp + fp} = {fmt(tp / (tp + fp))}</strong></section>; })}</div><p>Tray width represents 100% of each alert set, not an equal absolute number of alerts. Counts retain the population sizes.</p></MetricFigure>;
}
function UnitCurve({ points, kind, label }) {
  const valid = kind === 'roc' ? points.every(p => p.fpr !== null && p.recall !== null) : points.every(p => p.recall !== null);
  if (!valid) return <p>{label} undefined: {kind === 'roc' ? 'both observed classes are required.' : 'there are no actual positives.'}</p>;
  const coords = points.map(p => kind === 'roc' ? [p.fpr, p.recall] : [p.recall, p.precision ?? 1]);
  // PR connectors are a plotting aid, not the AP integration convention.
  return <div className="metric-curve"><h4>{label}</h4><svg viewBox="0 0 320 245" role="img" aria-label={`${label}, exact grouped operating points; axes zero to one`}><path d="M42 18V210H300" className="metric-axis"/>{[0, 0.5, 1].map(t => <g key={t}><text x={42 + t * 250} y="229" textAnchor="middle">{t}</text><text x="34" y={214 - t * 190} textAnchor="end">{t}</text></g>)}{kind === 'roc' && <path d="M42 210L292 20" className="metric-reference"/>}<polyline points={coords.map(([x, y]) => `${42 + x * 250},${210 - y * 190}`).join(' ')} className="metric-curve-line"/>{coords.map(([x, y], i) => <circle key={i} cx={42 + x * 250} cy={210 - y * 190} r={i === coords.length - 1 ? 5 : 3} className="metric-curve-point"/>)}</svg><p>{kind === 'roc' ? 'x: false positive rate · y: recall / TPR' : 'x: recall · y: precision'}</p></div>;
}
export function CurveViews({ rows = METRIC_ROWS, pointCount }) {
  const model = rankingMetrics(rows), points = pointCount ? model.points.slice(0, pointCount) : model.points;
  return <><div className="metric-paired"><UnitCurve points={points} kind="roc" label="ROC operating points"/><UnitCurve points={points} kind="pr" label="Precision–recall operating points"/></div><p>At no alerts, the PR anchor uses precision 1 only for plotting; measured precision is undefined. Straight PR connectors are not the AP integration rule.</p><MetricTable label="Exact threshold coordinates" headings={['Threshold', 'Admitted IDs', 'TP', 'FP', 'Recall', 'FPR', 'Precision', 'AP contribution']} rows={points.map((p, i) => [p.threshold === Infinity ? 'Above maximum' : fmt(p.threshold), p.ids.join(', ') || 'None', p.tp, p.fp, fmt(p.recall), fmt(p.fpr), fmt(p.precision), i === 0 ? '—' : p.recall === null ? 'undefined' : fmt((p.recall - points[i - 1].recall) * p.precision)])}/></>;
}
export function GroupedCurves() {
  return <MetricFigure id="curves" title="The tied block B + C advances both coordinates" caption="The points and table come from the same exact eight-item data. B and C cannot be separated by a score threshold."><ScoreBlocks rows={METRIC_ROWS} activeIds={['B', 'C']}/><CurveViews/></MetricFigure>;
}
export function PairGrid({ rows = METRIC_ROWS }) {
  const model = rankingMetrics(rows);
  if (model.auc === null) return <p>Pairwise AUC is undefined: at least one positive and one negative are required.</p>;
  return <><MetricTable label="Positive-negative pair credit grid" headings={['Positive ↓ / Negative →', ...model.negatives.map(r => `${r.id} (${fmt(r.score)})`)]} rows={model.pairs.map((row, i) => [`${model.positives[i].id} (${fmt(model.positives[i].score)})`, ...row.map(p => <span className={`metric-pair-credit credit-${p.credit === 0.5 ? 'tie' : p.credit}`}>{p.credit} {p.credit === 0.5 ? '(tie)' : p.credit ? '(higher)' : '(lower)'}</span>)])}/><p>Credits total {fmt(model.auc * model.positives.length * model.negatives.length)} / {model.positives.length * model.negatives.length} pairs → <strong>AUC {fmt(model.auc)}</strong>. AP = {fmt(model.ap)}; trapezoidal PR area = {fmt(model.prTrapezoid)}.</p></>;
}
export function PairCredits() { return <MetricFigure id="pairs" title="AUC: sixteen comparisons, including a half-credit tie"><PairGrid/></MetricFigure>; }
export function ProbabilityPenalties() {
  const [selected, setSelected] = useState('B'), [threshold, setThreshold] = useState(0.5);
  const squared = METRIC_ROWS.map(r => ({ ...r, score: r.score ** 2 }));
  const old = probabilityMetrics(METRIC_ROWS), changed = probabilityMetrics(squared);
  const first = old.contributions.find(r => r.id === selected), second = changed.contributions.find(r => r.id === selected);
  const logScale = Math.max(...old.contributions.map(r => r.log), ...changed.contributions.map(r => r.log));
  return <MetricFigure id="probabilities" title="Same order, different probabilities and penalties" caption="All ruler positions are probabilities on the same 0–1 scale. Squaring preserves ordering on this interval."><label className="metric-field">Inspect an item<select aria-label="Inspect an item" value={selected} onChange={e => setSelected(e.target.value)}>{METRIC_ROWS.map(r => <option key={r.id} value={r.id}>{r.id}, actual class {r.y}</option>)}</select></label><MetricTable label="Every probability and its squared forecast" headings={['Item (actual)', 'p → p²', 'Ordering']} rows={METRIC_ROWS.map(r => [`${r.id} (y=${r.y})`, `${fmt(r.score)} → ${fmt(r.score ** 2)}`, 'Preserved, including ties'])}/><div className="metric-probability-rulers">{[{ label: 'Original p', value: first.score }, { label: 'Squared p²', value: second.score }].map(r => <div key={r.label}><span>{r.label} = {fmt(r.value)}</span><div className="metric-ruler"><i style={{ left: `${r.value * 100}%` }}/></div></div>)}<p>Position markers are readouts, not handles: use “Inspect an item” above to compare another item. Both rulers run from 0 at left to 1 at right. {selected} has actual class {first.y}.</p></div><div className="metric-paired">{[['log', 'Log contribution (nats)', logScale], ['brier', 'Brier contribution', 1]].map(([key, label, max]) => <section key={key}><h4>{label}</h4>{[first, second].map((r, i) => <div className="metric-loss-row" key={i}><span>{i ? 'p²' : 'p'}: {fmt(r[key])}</span><div><i style={{ width: `${100 * r[key] / max}%` }}/></div></div>)}<small>Bar scale: 0 to {fmt(max)} {key === 'log' ? 'nats' : 'squared probability error'}.</small></section>)}</div><MetricTable label="Probability vector comparison" headings={['Forecasts', 'AUC', 'AP', 'Mean log loss', 'Mean Brier']} rows={[METRIC_ROWS, squared, METRIC_ROWS.map(r => ({ ...r, score: 0.5 }))].map((rows, i) => [i === 2 ? 'Constant 0.5' : i ? 'p²' : 'p', fmt(rankingMetrics(rows).auc), fmt(rankingMetrics(rows).ap), fmt(probabilityMetrics(rows).log), fmt(probabilityMetrics(rows).brier)])}/><label className="metric-field">Decision threshold (forecasts stay fixed)<input type="range" min="0" max="1" step="0.05" value={threshold} onChange={e => setThreshold(Number(e.target.value))}/></label><p aria-live="polite">Threshold {fmt(threshold)} → {binaryMetrics(METRIC_ROWS, threshold).tp + binaryMetrics(METRIC_ROWS, threshold).fp} original-score alerts. Original log loss stays {fmt(old.log)} and Brier stays {fmt(old.brier)}: neither uses this threshold.</p></MetricFigure>;
}
function ResidualRuler({ item, predictor, min, max, unit, onForecastChange, onDragStart, onDragEnd }) {
  const drag = useRef(null);
  const position = value => 24 + 252 * (value - min) / (max - min);
  const change = value => onForecastChange(item.id, predictor, Math.max(min / unit, Math.min(max / unit, value)));
  const move = event => {
    if (drag.current?.pointerId !== event.pointerId) return;
    // Invert the rendered SVG transform, including responsive scaling/letterboxing.
    const svg = event.currentTarget.ownerSVGElement;
    const matrix = svg.getScreenCTM();
    if (!matrix) return;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const local = point.matrixTransform(matrix.inverse());
    const value = (min + (local.x - drag.current.offset - 24) / 252 * (max - min)) / unit;
    change(Math.round(value * 4) / 4);
  };
  const finish = event => {
    if (drag.current?.pointerId !== event.pointerId) return;
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    onDragEnd?.();
  };
  const keyDown = event => {
    const current = item.prediction / unit;
    const targets = { ArrowRight: current + .25, ArrowUp: current + .25, ArrowLeft: current - .25, ArrowDown: current - .25, PageUp: current + 1, PageDown: current - 1, Home: min / unit, End: max / unit };
    if (!(event.key in targets)) return;
    event.preventDefault();
    change(targets[event.key]);
  };
  const x = position(item.prediction);
  return <svg className={onForecastChange ? 'metric-editable-ruler' : undefined} viewBox="0 0 300 52" role={onForecastChange ? 'group' : 'img'} aria-label={`${item.id} observed ${fmt(item.y)}, forecast ${fmt(item.prediction)} ${unit === 60 ? 'seconds' : 'minutes'}`}>
    <line x1="24" y1="26" x2="276" y2="26" className="metric-axis"/>
    <line x1={position(item.y)} y1="26" x2={x} y2="26" className="metric-residual-line"/>
    <circle cx={position(item.y)} cy="26" r="4" fill="currentColor"/>
    {onForecastChange ? <g className="metric-forecast-handle" role="slider" tabIndex={0} aria-label={`${predictor.toUpperCase()} ${item.id} forecast`} aria-valuemin={min / unit} aria-valuemax={max / unit} aria-valuenow={item.prediction / unit} aria-valuetext={`${fmt(item.prediction / unit)} minutes; residual ${fmt(item.residual)} ${unit === 60 ? 'seconds' : 'minutes'}`} aria-orientation="horizontal" onKeyDown={keyDown} onPointerDown={event => {
      if (event.button !== 0 || drag.current !== null) return;
      event.preventDefault();
      event.currentTarget.focus();
      const svg = event.currentTarget.ownerSVGElement;
      const matrix = svg.getScreenCTM();
      if (!matrix) return;
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      drag.current = { pointerId: event.pointerId, offset: point.matrixTransform(matrix.inverse()).x - x };
      event.currentTarget.setPointerCapture(event.pointerId);
      onDragStart?.();
    }} onPointerMove={move} onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={finish}>
      <rect x={x - 22} y="4" width="44" height="44" fill="transparent"/>
      <path d={`M${x} 18l8 8-8 8-8-8Z`} className="metric-prediction"/>
    </g> : <path d={`M${x} 18l8 8-8 8-8-8Z`} className="metric-prediction"/>}
  </svg>;
}

export function ResidualGeometry({ rows = RESIDUAL_ROWS, unit = 1, trainingMean = 3, onForecastChange, fixedDomain }) {
  const [dragDomain, setDragDomain] = useState(null);
  const a = regressionMetrics(rows, 'a', unit), b = regressionMetrics(rows, 'b', unit);
  const evaluationReference = regressionMetrics(rows.map(r => ({ ...r, reference: a.mean / unit })), 'reference', unit);
  const baseline = regressionMetrics(rows.map(r => ({ ...r, baseline: trainingMean })), 'baseline', unit);
  const values = rows.flatMap(r => [r.y, r.a, r.b]);
  const domain = fixedDomain ?? dragDomain ?? [Math.max(-100, Math.floor(Math.min(-2, ...values) - (Math.min(...values) < -2 ? 1 : 0))), Math.min(100, Math.ceil(Math.max(12, ...values) + (Math.max(...values) > 12 ? 1 : 0)))];
  const [min, max] = domain.map(value => value * unit);
  // Keep the area encoding fixed while a handle moves: shrinking the largest
  // residual must shrink its square, rather than silently rescaling it to 64 px.
  const squareError = (domain[1] - domain[0]) * unit;
  return <>
    <p>Shared ruler: {fmt(min)} to {fmt(max)} {unit === 60 ? 'seconds' : 'minutes'}. ● observed value; ◇ forecast. The gold segment is the error length. Positive residual means the forecast is too low.</p>
    {onForecastChange && <p className="metric-drag-help"><strong>Drag a gold diamond.</strong> Or Tab to it and use arrow keys (¼ minute), Page Up/Down (1 minute), Home/End (ruler limits). Observed circles stay fixed while dragging.</p>}
    <div className="metric-paired">{[['A', 'a', a], ['B', 'b', b]].map(([name, key, result]) => <section key={name}>
      <h4>Predictor {name}</h4>
      <p className="metric-residual-totals">MAE <strong>{fmt(result.mae)}</strong> · RMSE <strong>{fmt(result.rmse)}</strong><br/>SSE <strong>{fmt(result.sse)}</strong> · R² <strong>{fmt(result.r2)}</strong></p>
      {result.contributions.map(item => <div className="metric-residual" key={item.id}>
        <p><strong>{item.id}</strong> · observed {fmt(item.y)} · forecast {fmt(item.prediction)}<br/>Residual {fmt(item.residual)} · squared error {fmt(item.square)}</p>
        <div className="metric-residual-geometry"><ResidualRuler item={item} predictor={key} min={min} max={max} unit={unit} onForecastChange={onForecastChange} onDragStart={() => setDragDomain(domain)} onDragEnd={() => setDragDomain(null)}/>
        <div className="metric-square-space">{item.residual === 0 ? <span>0 area</span> : <div className="metric-error-square" style={{ width: `${64 * Math.abs(item.residual) / squareError}px`, height: `${64 * Math.abs(item.residual) / squareError}px` }} aria-label={`Squared residual area ${fmt(item.square)}`}/>}</div></div>
      </div>)}
    </section>)}</div>
    <p>Both columns share one area scale: a 64 px side represents |error| = {fmt(squareError)} {unit === 60 ? 'seconds' : 'minutes'}; square area is proportional to squared error. Zero errors have no square. {fixedDomain ? 'The ruler and square scales stay fixed through every edit.' : 'Scales stay fixed during a drag; numerical edits can expand the shared ruler.'}</p>
    <MetricTable label="Residual summary and actual fitted baseline" headings={['Predictor', 'MAE', 'MSE', 'RMSE', 'Median |error|', 'SSE', 'R²']} rows={[[ 'A', a], ['B', b], ['Evaluation-mean formula reference', evaluationReference], [`Training constant ${fmt(trainingMean * unit)}`, baseline]].map(([name, result]) => [name, ...['mae', 'mse', 'rmse', 'medae', 'sse', 'r2'].map(key => fmt(result[key]))])}/>
    <p>Evaluation mean = {fmt(a.mean)}; SST = {fmt(a.sst)} {unit === 60 ? 'seconds²' : 'minutes²'}. This formula reference is distinct from the training constant. {a.r2 === null && <strong>R² undefined: the evaluation target has zero variance.</strong>}</p>
    {values.some(value => value < 0) && <p className="metric-notice">A negative duration is physically invalid for this example even though the numerical errors are calculable.</p>}
  </>;
}

export function ResidualSquares() {
  const [rows, setRows] = useState(() => RESIDUAL_ROWS.map(row => ({ ...row })));
  const changeForecast = (id, key, value) => setRows(current => current.map(row => row.id === id ? { ...row, [key]: value } : row));
  return <MetricFigure id="residuals" title="Explore how one large error can outweigh five moderate ones" caption="Playable constructed durations, not a fitted model experiment. The five observations stay fixed; the two sets of forecasts are independent.">
    <p>Start by moving A’s T5 diamond from 4 toward its observed value, 10. Watch the error square shrink and compare MAE with RMSE. Then move any B diamond away from its circle.</p>
    <div className="metric-actions"><button onClick={() => setRows(RESIDUAL_ROWS.map(row => ({ ...row })))}>Reset original forecasts</button><button onClick={() => setRows(current => current.map(row => ({ ...row, a: row.y, b: row.y })))}>Match all observations</button></div>
    <div data-live-exploration="residual-squares"><ResidualGeometry rows={rows} fixedDomain={[-2, 12]} onForecastChange={changeForecast}/></div>
  </MetricFigure>;
}
export function RetrievalShelves({ rows = DOCUMENT_ROWS, cutoff = 3, gain = 'exponential' }) {
  const result = retrievalMetrics(rows, cutoff, gain);
  return <><div className="metric-paired">{[['Actual shelf', result.actual], ['Ideal: same judged candidates', result.ideal]].map(([title, shelf]) => <section key={title}><h4>{title}</h4><ol className="metric-shelf">{shelf.map(r => <li key={r.id} className={r.rank <= cutoff ? 'within-cutoff' : ''}><strong>{r.id} · grade {r.grade}</strong><span>gain {r.gain} ÷ log₂({r.rank + 1})</span><span>{r.rank <= cutoff ? `contribution ${fmt(r.contribution)}` : `outside K=${cutoff}: 0 credit`}</span><div className="metric-gain-track"><i style={{ width: `${100 * r.contribution / (Math.max(...result.ideal.map(item => item.gain)) || 1)}%` }}/></div></li>)}</ol></section>)}</div><p>Contribution bar scale is shared by both shelves: 0 to {Math.max(...result.ideal.map(item => item.gain)) || 1} discounted gain units.</p><p>DCG {fmt(result.dcg)} ÷ ideal DCG {fmt(result.idcg)} = <strong>NDCG {fmt(result.ndcg)}</strong>. {result.ndcg === null && 'No candidate has positive gain; the raw ratio is undefined.'}</p><MetricTable label="Retrieval summaries" headings={['P@K', 'Recall@K', 'Reciprocal rank', 'Full-list AP', 'NDCG@K']} rows={[[result.precision, result.recall, result.rr, result.ap, result.ndcg].map(fmt)]}/></>;
}
export function RankedShelves() { return <MetricFigure id="retrieval" title="Same relevant set, different positions" caption="Constructed judgments. Exponential gains 2^grade − 1; cutoff K=3."><RetrievalShelves/></MetricFigure>; }
export function MeasuredEvaluation() {
  const points = measured.development_thresholds.filter(p => Number.isFinite(p.threshold));
  const maxCost = Math.max(...points.map(p => p.cost_fp1_fn5)), selected = measured.selected_development_record;
  const test = measured.reports;
  return <MetricFigure id="measured" title="Choose on development data. Preserve the test result." caption="Measured two-feature Banknote Authentication experiment; illustrative costs FP + 5FN."><div className="metric-data-roles"><span>320 training rows<br/><strong>fit scaler + model</strong></span><span>80 development rows<br/><strong>choose threshold</strong></span><span>80 test rows<br/><strong>report frozen policy</strong></span></div><h4>Development threshold versus total cost</h4><svg className="metric-cost-curve" viewBox="0 0 320 240" role="img" aria-label={`Actual development cost curve, chosen threshold ${selected.threshold}, cost ${selected.cost_fp1_fn5}`}><path d="M42 20V195H300" className="metric-axis"/>{[0, 0.5, 1].map(x => <text key={x} x={42 + x * 250} y="217" textAnchor="middle">{x}</text>)}{[0, maxCost].map(y => <text key={y} x="34" y={199 - 175 * y / maxCost} textAnchor="end">{y}</text>)}<polyline className="metric-curve-line" points={[...points].sort((a, b) => a.threshold - b.threshold).map(p => `${42 + p.threshold * 250},${195 - 175 * p.cost_fp1_fn5 / maxCost}`).join(' ')}/><circle cx={42 + selected.threshold * 250} cy={195 - 175 * selected.cost_fp1_fn5 / maxCost} r="6" className="metric-selected-point"/></svg><p>x: threshold · y: development cost. Selected threshold {fmt(selected.threshold)}, development cost {selected.cost_fp1_fn5}. Line connects actual evaluated finite-score candidates. The separate no-alert candidate costs {measured.development_thresholds.find(p => p.threshold === 'above_maximum').cost_fp1_fn5}; it is listed in the table because its above-maximum threshold is not a finite point on this score axis.</p><details><summary>Inspect all development candidates and exact counts</summary><MetricTable label="Measured development candidate thresholds" headings={['Threshold', 'TP', 'FP', 'FN', 'TN', 'Cost']} rows={measured.development_thresholds.map(p => [p.threshold === 'above_maximum' ? 'Above maximum (no alerts)' : fmt(p.threshold), p.tp, p.fp, p.fn, p.tn, p.cost_fp1_fn5])}/></details><MetricTable label="Frozen held-out test policies" headings={['Policy', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'Cost']} rows={Object.entries(test).map(([name, r]) => [name.replaceAll('_', ' '), r.tp, r.fp, r.fn, r.tn, fmt(r.precision), fmt(r.recall), r.cost_fp1_fn5])}/><div className="metric-count-changes"><span style={{ flex: 17 }}>+17 false alerts</span><span style={{ flex: 3 }}>+3 recovered positives</span></div><p>Segment widths share one count scale (17:3); they are count changes, not costs.</p><p className="metric-notice"><strong>17 added false alerts − 5 × 3 recovered positives = 2 extra cost units.</strong> The development-selected policy costs 26 on test, versus 24 at the predeclared 0.5 threshold.</p><p>The two logistic-regression policies share one unchanged probability vector: AUC 0.956728 · AP 0.945489 · log loss 0.252139 · Brier 0.073915. Thresholds change decisions, not these four scores.</p></MetricFigure>;
}
