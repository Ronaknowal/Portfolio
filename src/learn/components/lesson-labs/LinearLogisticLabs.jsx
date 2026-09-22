import { useId, useState } from 'react';
import { Investigation, Stepper } from './LessonInvestigation.jsx';
import { LessonTable } from './LessonElements.jsx';
import { residualReport, gradientTrace, logisticReport, sigmoid, thresholdReport, separationReport, separationOptimum, featureMapReport, uncertaintyReport } from '../../data/linear-logistic-models.js';
import './linear-logistic-labs.css';

const WIDTH = 560;
const HEIGHT = 260;
const LEFT = 58;
const RIGHT = 532;
const TOP = 22;
const BOTTOM = 212;
const fixed = (value, places = 3) => value === null ? 'undefined' : value.toFixed(places);

function Chart({ title, description, xDomain, yDomain, xLabel, yLabel, children }) {
  const xScale = value => LEFT + (value - xDomain[0]) / (xDomain[1] - xDomain[0]) * (RIGHT - LEFT);
  const yScale = value => BOTTOM - (value - yDomain[0]) / (yDomain[1] - yDomain[0]) * (BOTTOM - TOP);
  const clipId = useId().replaceAll(':', '');
  return <figure className="regression-figure">
    <div className="regression-chart-scroll" tabIndex={0} role="region" aria-label={`${title}; scroll horizontally if needed`}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={title}>
        <title>{title}</title><desc>{description}</desc>
        <defs><clipPath id={clipId}><rect x={LEFT - 3} y={TOP - 3} width={RIGHT - LEFT + 6} height={BOTTOM - TOP + 6} /></clipPath></defs>
        {[0, 0.25, 0.5, 0.75, 1].map(fraction => {
          const x = LEFT + fraction * (RIGHT - LEFT);
          const y = BOTTOM - fraction * (BOTTOM - TOP);
          return <g key={fraction}>
            <line x1={LEFT} x2={RIGHT} y1={y} y2={y} className="regression-grid" />
            <text x={LEFT - 10} y={y + 4} textAnchor="end">{fixed(yDomain[0] + fraction * (yDomain[1] - yDomain[0]), 1)}</text>
            <text x={x} y={BOTTOM + 19} textAnchor="middle">{fixed(xDomain[0] + fraction * (xDomain[1] - xDomain[0]), 1)}</text>
          </g>;
        })}
        <line x1={LEFT} x2={RIGHT} y1={BOTTOM} y2={BOTTOM} className="regression-axis" />
        <line x1={LEFT} x2={LEFT} y1={TOP} y2={BOTTOM} className="regression-axis" />
        <text x={(LEFT + RIGHT) / 2} y={HEIGHT - 7} textAnchor="middle">{xLabel}</text>
        <text x={14} y={(TOP + BOTTOM) / 2} transform={`rotate(-90 14 ${(TOP + BOTTOM) / 2})`} textAnchor="middle">{yLabel}</text>
        <g clipPath={`url(#${clipId})`}>{children({ xScale, yScale })}</g>
      </svg>
    </div>
    <p className="regression-scroll-hint">If the plot extends beyond the screen, swipe sideways or focus it and use the arrow keys to inspect the remaining axis.</p>
    <figcaption>{description}</figcaption>
  </figure>;
}

function Polyline({ points, xScale, yScale, className = 'regression-line' }) {
  return <polyline points={points.map(([x, y]) => `${xScale(x)},${yScale(y)}`).join(' ')} fill="none" className={className} />;
}

function Slider({ label, value, onChange, min, max, step = 0.1 }) {
  const id = useId();
  return <label className="regression-control" htmlFor={id}><span>{label}: <output>{value}</output></span><input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}

export function PredictionProtocolFigure() {
  return <figure className="regression-protocol">
    <div><strong>Before dispatch</strong><span>distance, service, departure time</span><span className="regression-arrow">↓ frozen feature row x</span><strong>Fitted rule → prediction</strong></div>
    <div><strong>After arrival</strong><span>elapsed hours y, deadline outcome</span><span className="regression-arrow">↓ later evaluation</span><strong>Compare with stored prediction</strong></div>
    <figcaption>Training uses past completed shipments. A new prediction cannot use its own eventual arrival time or post-delivery complaint. The arrows encode when the information becomes available.</figcaption>
  </figure>;
}

export function ResidualGeometryLab() {
  const [intercept, setIntercept] = useState(0);
  const [slope, setSlope] = useState(1);
  const [lastHours, setLastHours] = useState(4);
  const report = residualReport(intercept, slope, lastHours);
  const reset = () => { setIntercept(0); setSlope(1); setLastHours(4); };
  return <Investigation id="regression-residuals" kicker="FIT A NUMBER" title="Move the line and account for every error">
    <p className="lesson-live-note">At slope 1 and intercept 0, which shipment contributes most squared error? Explore what happens if D takes 8 hours instead of 4.</p>
    <div className="regression-controls">
      <Slider label="Intercept (hours)" value={intercept} onChange={setIntercept} min={-1} max={4} step={0.05} />
      <Slider label="Slope (hours per 100 km)" value={slope} onChange={setSlope} min={-1} max={3} />
      <label>Shipment D hours<select value={lastHours} onChange={event => setLastHours(Number(event.target.value))}><option value={4}>4 — original observation</option><option value={8}>8 — changed observation</option></select></label>
    </div>
    <div className="regression-actions"><button onClick={() => { setSlope(0); setIntercept(report.meanHours); }}>Use mean baseline</button><button onClick={() => { setSlope(report.bestSlope); setIntercept(report.bestIntercept); }}>Fit least squares</button><button onClick={reset}>Reset</button></div>
    <Chart title="Shipment predictions and vertical residuals" description="Circles are observed hours; the solid line is the current prediction. Dashed vertical segments connect each prediction to its observation. All values use the same four rows shown below." xDomain={[0, 3]} yDomain={[-4, 14]} xLabel="Distance (100 km)" yLabel="Elapsed hours">
      {({ xScale, yScale }) => <><Polyline points={[[0, intercept], [3, intercept + 3 * slope]]} xScale={xScale} yScale={yScale} />{report.rows.map(row => <g key={row.id}><line x1={xScale(row.distance)} x2={xScale(row.distance)} y1={yScale(row.hours)} y2={yScale(row.predicted)} className="regression-residual" /><circle cx={xScale(row.distance)} cy={yScale(row.hours)} r={5} className="regression-point" /></g>)}</>}
    </Chart>
    <p className="regression-readout" aria-live="polite">Mean squared error: <strong>{fixed(report.mse)} hours²</strong>. Current line: {fixed(intercept)} + {fixed(slope)}x.</p>
    <LessonTable caption="The exact contributions behind the picture (display rounded)" headers={['Shipment', 'Observed', 'Predicted', 'Residual y − ŷ', 'Squared residual']} rows={report.rows.map(row => [row.id, fixed(row.hours), fixed(row.predicted), fixed(row.residual), fixed(row.squared)])} />
    <p>Fit, then change D. The previous coefficients stay fixed until you choose “Fit least squares” again: changed evidence and refitting are separate operations. Explain why a larger error has a stronger effect on this loss.</p>
  </Investigation>;
}

export function GradientGeometryLab() {
  const [rate, setRate] = useState(0.05);
  const [step, setStep] = useState(0);
  const states = gradientTrace(rate, 16);
  const current = states[step];
  const outside = states.slice(0, step + 1).some(state => state.intercept < -0.5 || state.intercept > 2.5 || state.slope < -0.5 || state.slope > 2.5);
  const contours = [0.05, 0.2, 0.8, 3].map(level => {
    const radius = Math.sqrt(14 * level / 5);
    const branch = sign => Array.from({ length: 81 }, (_, index) => {
      const deltaIntercept = -radius + 2 * radius * index / 80;
      const deltaSlope = (-3 * deltaIntercept + sign * Math.sqrt(Math.max(0, 14 * level - 5 * deltaIntercept ** 2))) / 7;
      return [0.9 + deltaIntercept, 0.9 + deltaSlope];
    });
    return [...branch(1), ...branch(-1).reverse()];
  });
  return <Investigation id="regression-gradient" kicker="TRAINING CHANGES PARAMETERS" title="Follow an update through coefficient space">
    <p className="lesson-live-note">The first gradient is (−4.5, −9). With rate 0.05, where should the next point appear?</p>
    <label>Learning rate<select value={rate} onChange={event => { setRate(Number(event.target.value)); setStep(0); }}><option value={0.05}>0.05 — conservative</option><option value={0.2}>0.20 — stable, oscillating</option><option value={0.3}>0.30 — unstable here</option></select></label>
    <Chart title="Actual full-batch gradient path" description="Horizontal position is intercept b; vertical position is slope w. Gray ellipses are exact equal-MSE contours, not shipment coordinates. The × marks the least-squares optimum (0.9, 0.9)." xDomain={[-0.5, 2.5]} yDomain={[-0.5, 2.5]} xLabel="Intercept b (hours)" yLabel="Slope w (hours per 100 km)">
      {({ xScale, yScale }) => <>{contours.map((points, index) => <Polyline key={index} points={points} xScale={xScale} yScale={yScale} className="regression-contour" />)}<Polyline points={states.slice(0, step + 1).map(state => [state.intercept, state.slope])} xScale={xScale} yScale={yScale} /><text x={xScale(0.9)} y={yScale(0.9) + 5} textAnchor="middle" className="regression-optimum">×</text><circle cx={xScale(current.intercept)} cy={yScale(current.slope)} r={6} className="regression-point" /></>}
    </Chart>
    <Stepper step={step} count={states.length} setStep={setStep} onReset={() => { setStep(0); setRate(0.05); }} />
    <p aria-live="polite">Iteration {step}: b={fixed(current.intercept, 6)}, w={fixed(current.slope, 6)}, MSE={fixed(current.mse, 6)}. Gradient=({fixed(current.gradient[0], 6)}, {fixed(current.gradient[1], 6)}). {outside && <strong>Part of the path is outside the fixed plot window; the numeric values remain shown.</strong>}</p>
    <p>This is deterministic full-batch gradient descent on four rows, not stochastic gradient descent. The safe rate range belongs to this particular objective and feature scaling.</p>
  </Investigation>;
}

export function LogisticScoreLab() {
  const [intercept, setIntercept] = useState(-1);
  const [weight, setWeight] = useState(1);
  const [feature, setFeature] = useState(2);
  const [label, setLabel] = useState(1);
  const report = logisticReport({ intercept, weight, feature, label });
  const reset = () => { setIntercept(-1); setWeight(1); setFeature(2); setLabel(1); };
  return <Investigation id="regression-logistic-score" kicker="SCORE → PROBABILITY → LOSS" title="Follow one observation without hiding its contribution">
    <p className="lesson-live-note">With b=−1, w=1 and x=2 the score is 1. Is the probability change from adding one score unit the same everywhere?</p>
    <div className="regression-controls"><Slider label="Intercept b" value={intercept} onChange={setIntercept} min={-3} max={3} step={0.5} /><Slider label="Weight w" value={weight} onChange={setWeight} min={-2} max={2} step={0.5} /><Slider label="Feature x" value={feature} onChange={setFeature} min={-2} max={2} step={0.5} /><label>Observed label<select value={label} onChange={event => setLabel(Number(event.target.value))}><option value={1}>1 — missed deadline</option><option value={0}>0 — met deadline</option></select></label><button onClick={reset}>Reset</button></div>
    <p className="regression-calculation">b + wx = {intercept} + ({weight} × {feature}) = <strong>{fixed(report.score)}</strong> → σ(z) = <strong>{fixed(report.probability)}</strong></p>
    <Chart title="The sigmoid and this observation's score" description="The curve is σ(z)=1/(1+exp(−z)). The selected point moves with the score. Changing the observed label changes the loss and gradient, not this fitted probability." xDomain={[-7, 7]} yDomain={[0, 1]} xLabel="Score z (log odds)" yLabel="Model probability of label 1">
      {({ xScale, yScale }) => <><Polyline points={Array.from({ length: 113 }, (_, index) => { const score = -7 + index / 8; return [score, sigmoid(score)]; })} xScale={xScale} yScale={yScale} /><line x1={xScale(report.score)} x2={xScale(report.score)} y1={yScale(0)} y2={yScale(report.probability)} className="regression-residual" /><circle cx={xScale(report.score)} cy={yScale(report.probability)} r={6} className="regression-point" /></>}
    </Chart>
    <LessonTable caption="One observation's probability, loss and parameter derivatives" headers={['Quantity', 'Value', 'Meaning']} rows={[
      ['Odds p/(1−p)', fixed(report.odds), 'Multiplicative comparison, not probability'],
      ['Log loss', fixed(report.loss), `Penalty for observed label ${label}; natural logarithm`],
      ['Intercept derivative p−y', fixed(report.biasGradient), 'Direction of the bias update'],
      ['Weight derivative (p−y)x', fixed(report.weightGradient), 'The feature multiplies the error signal'],
    ]} />
    <p>Change only the label to make a confident prediction wrong. Then change x: a bounded probability error does not bound a weight gradient when feature magnitudes are unbounded. No fitting occurs in this inspection.</p>
  </Investigation>;
}

export function ThresholdDecisionsLab() {
  const [threshold, setThreshold] = useState(0.5);
  const [missedCost, setMissedCost] = useState(4);
  const report = thresholdReport(threshold, missedCost);
  return <Investigation id="regression-threshold" kicker="CHOOSE AN ACTION" title="Move the decision gate, keep the scores fixed">
    <p className="lesson-live-note">At threshold 0.5, how many missed deadlines are missed by the warning rule? Will lowering the threshold remove every error?</p>
    <div className="regression-controls"><Slider label="Warning threshold" value={threshold} onChange={setThreshold} min={0} max={1} step={0.05} /><label>Cost of a missed warning<select value={missedCost} onChange={event => setMissedCost(Number(event.target.value))}><option value={1}>1</option><option value={4}>4</option><option value={10}>10</option></select></label><button onClick={() => { setThreshold(0.5); setMissedCost(4); }}>Reset</button></div>
    <Chart title="Six validation observations against a threshold" description="Each row is one fixed illustrative validation observation; its horizontal position is its supplied probability. A filled circle means true label 1, an empty square means true label 0. The dashed gate warns for p≥threshold. These supplied probabilities are not outputs of the earlier parcel fit." xDomain={[0, 1]} yDomain={[0.5, 6.5]} xLabel="Supplied model probability" yLabel="Validation row number">
      {({ xScale, yScale }) => <><line x1={xScale(threshold)} x2={xScale(threshold)} y1={TOP} y2={BOTTOM} className="regression-residual" />{report.rows.map((row, index) => row.label ? <circle key={row.id} cx={xScale(row.probability)} cy={yScale(index + 1)} r={6} className="regression-point" /> : <rect key={row.id} x={xScale(row.probability) - 5} y={yScale(index + 1) - 5} width={10} height={10} className="regression-negative" />)}</>}
    </Chart>
    <LessonTable caption="Observed counts; rows are truth, columns are decisions" headers={['Truth', 'No warning', 'Warning']} rows={[[0, `${report.tn} true negatives`, `${report.fp} false positives`], [1, `${report.fn} false negatives`, `${report.tp} true positives`]]} />
    <LessonTable caption="Read the exact values behind each point" headers={['Row', 'Fixed probability', 'Truth', 'Current decision']} rows={report.rows.map(row => [row.id, row.probability, row.label, row.prediction ? 'Warning' : 'No warning'])} />
    <p aria-live="polite">Total cost = FP + {missedCost}×FN = <strong>{report.cost}</strong>. Accuracy {fixed(report.accuracy)}, precision {fixed(report.precision)}, recall {fixed(report.recall)}.</p>
    <p>The threshold uses “≥”; a score exactly on the gate gets a warning. A precision of “undefined” means no warning was issued, so its denominator is zero. Choosing a rule on these validation rows does not establish its future cost.</p>
  </Investigation>;
}

export function SeparationPenaltyLab() {
  const [weight, setWeight] = useState(2);
  const [penalty, setPenalty] = useState(0);
  const report = separationReport(weight, penalty);
  const optimum = separationOptimum(penalty);
  const points = Array.from({ length: 97 }, (_, index) => { const currentWeight = index / 8; return [currentWeight, separationReport(currentWeight, penalty).objective]; });
  const maxLoss = Math.max(1, ...points.map(point => point[1]));
  return <Investigation id="regression-separation" kicker="EXISTENCE BEFORE CONVERGENCE" title="A perfect separator can keep asking for larger weights">
    <p className="lesson-live-note">For observations (−1,0) and (+1,1), every positive w separates the labels. What happens to log loss as w grows?</p>
    <div className="regression-controls"><Slider label="Weight w" value={weight} onChange={setWeight} min={0} max={12} step={0.25} /><label>Penalty λ<select value={penalty} onChange={event => setPenalty(Number(event.target.value))}><option value={0}>0 — no penalty</option><option value={0.02}>0.02</option><option value={0.2}>0.20</option></select></label><button onClick={() => { setWeight(2); setPenalty(0); }}>Reset</button></div>
    <Chart title="Separated data: loss plus a stated weight penalty" description="Exact objective: log(1+exp(−w)) + λw²/2, with intercept fixed at zero. The vertical scale adapts to λ; compare numeric values across settings. The plotted finite window is not a search over all real weights." xDomain={[0, 12]} yDomain={[0, maxLoss]} xLabel="Weight w" yLabel="Penalized mean objective">
      {({ xScale, yScale }) => <><Polyline points={points} xScale={xScale} yScale={yScale} /><circle cx={xScale(weight)} cy={yScale(report.objective)} r={6} className="regression-point" />{optimum !== null && <line x1={xScale(optimum)} x2={xScale(optimum)} y1={TOP} y2={BOTTOM} className="regression-residual" />}</>}
    </Chart>
    <p aria-live="polite">Data loss {fixed(report.dataLoss, 6)} + penalty {fixed(report.penaltyLoss, 6)} = {fixed(report.objective, 6)}. {optimum === null ? 'No finite unpenalized minimizer: the infimum 0 is approached as w→∞.' : `The finite minimizer is w≈${fixed(optimum, 6)} (dashed line).`}</p>
    <p>For λ&gt;0 the penalty eventually dominates. This symmetric fixed-intercept example does not prove that penalizing only slopes cures an all-one-label dataset with a free, unpenalized intercept.</p>
  </Investigation>;
}

export function FeatureMapFigure() {
  const rows = featureMapReport();
  return <figure className="regression-feature-map"><LessonTable caption="One feature map: x → (x, x²) → score x²−1 → probability" headers={['Original x', 'New feature x²', 'Score x²−1', 'Probability']} rows={rows.map(row => [row.input, row.square, row.score, fixed(row.probability)])} /><figcaption>The score is linear in the supplied features (1, x, x²), with coefficients (−1, 0, 1). At threshold 0.5, the score-zero boundary consists of the two input values −1 and +1. The prediction regions are intervals on a line.</figcaption></figure>;
}

export function UncertaintyFigure() {
  const rows = Array.from({ length: 81 }, (_, index) => uncertaintyReport(index / 10));
  const interval = (key, xScale, yScale) => [...rows.map(row => [row.input, row.predicted + row[key]]), ...[...rows].reverse().map(row => [row.input, row.predicted - row[key]])].map(([x, y]) => `${xScale(x)},${yScale(y)}`).join(' ');
  return <>
    <Chart title="Uncertainty about a mean and about one future outcome" description="Calculated pointwise 95% t intervals for the six observations in the program below: inner gold shading estimates the mean, outer blue shading predicts one independent outcome. These are not simultaneous 95% bands over every x. Vertical dashed line x=5 marks the last observed input; the assumed model continues beyond it, not the evidence." xDomain={[0, 8]} yDomain={[-1, 11]} xLabel="Input x (observed 0–5)" yLabel="Output y">
      {({ xScale, yScale }) => <><polygon points={interval('individualHalfWidth', xScale, yScale)} fill="#89c9e5" fillOpacity=".16" stroke="#89c9e5" /><polygon points={interval('meanHalfWidth', xScale, yScale)} fill="#eec26c" fillOpacity=".25" stroke="#eec26c" /><Polyline points={rows.map(row => [row.input, row.predicted])} xScale={xScale} yScale={yScale} /><line x1={xScale(5)} x2={xScale(5)} y1={TOP} y2={BOTTOM} className="regression-residual" />{rows[0].observations.map((value, index) => <circle key={index} cx={xScale(index)} cy={yScale(value)} r={5} className="regression-point" />)}</>}
    </Chart>
    <LessonTable caption="Same model, two different questions (half-widths rounded)" headers={['Input', 'Fitted mean', 'Mean half-width', 'Individual half-width']} rows={[2.5, 8].map(input => { const row = uncertaintyReport(input); return [input, fixed(row.predicted, 4), fixed(row.meanHalfWidth, 4), fixed(row.individualHalfWidth, 4)]; })} />
  </>;
}
