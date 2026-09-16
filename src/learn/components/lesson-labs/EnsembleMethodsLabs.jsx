import { useId, useState } from 'react';
import { Investigation, Predict, Stepper } from './LessonInvestigation.jsx';
import { LessonTable } from './LessonElements.jsx';
import { ensembleNumber as number, errorBlendPresets, errorBlendState, votingState, bootstrapRows, bootstrapPresets, bootstrapState, boostingPresets, signedBoostingTrace, oofOwnershipState, calibratedAverageLaw } from '../../data/ensemble-methods-models.js';
import './ensemble-methods-labs.css';
import predictionMap from '../../data/ensemble-prediction-map.json';
const colors = {
  a: '#7fc9ba',
  b: '#b2b7e1',
  blend: '#e8bb60',
  muted: '#969ca3',
  bad: '#eb9b91'
};
const rowName = index => String.fromCharCode(65 + index);
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.01
}) {
  const id = useId();
  return <label className="ensemble-range" htmlFor={id}>
    <span>{label}<output>{number(value, 3)}</output></span>
    <input id={id} type="range" value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}
function ScrollFigure({
  children,
  label,
  caption,
  compact = false
}) {
  return <figure className="ensemble-figure">
    <div className={compact ? 'ensemble-compact-plot' : 'ensemble-plot-scroll'} tabIndex={compact ? undefined : 0} role={compact ? undefined : 'region'} aria-label={compact ? undefined : label}>{children}</div>
    {!compact && <p className="ensemble-note">On a narrow screen, swipe the detailed plot sideways or focus it and use the arrow keys.</p>}
    <figcaption>{caption}</figcaption>
  </figure>;
}
export function PredictionCombinationFigure() {
  return <figure className="ensemble-flow" aria-label="A new observation produces several fitted predictions and one combined prediction">
    <div><strong>One new shipment</strong><span>Same features, same unknown arrival outcome</span></div>
    <p className="ensemble-flow-arrow" aria-hidden="true">↓</p>
    <div className="ensemble-three"><span>Saved rule A<br /><strong>6 hours</strong></span><span>Saved rule B<br /><strong>10 hours</strong></span><span>Saved rule C<br /><strong>14 hours</strong></span></div>
    <p className="ensemble-flow-arrow" aria-hidden="true">↓</p>
    <div><strong>Equal average: 10 hours</strong><span>After arrival, compare with the actual elapsed time.</span></div>
    <figcaption>This arithmetic combines predictions about one outcome. It neither creates three independent shipments nor proves that 10 hours is correct.</figcaption>
  </figure>;
}
export function VotingComparisonLab() {
  const [probabilities, setProbabilities] = useState([0.51, 0.51, 0.01]);
  const [lastWeight, setLastWeight] = useState(1);
  const state = votingState(probabilities, [1, 1, lastWeight]);
  return <Investigation id="ensemble-votes" kicker="WHAT DOES A MODEL CONTRIBUTE?" title="The same three forecasts can produce different decisions">
    <Predict>Two models assign a 51% chance of delay. A third assigns 1%. Predict the majority ballot and the mean probability before changing anything.</Predict>
    <p>Every probability refers to class 1: <strong>delayed</strong>. Class 0 means on time. Each member votes for its larger probability; all exact ties choose class 0.</p>
    <div className="ensemble-controls">
      {probabilities.map((value, index) => <Range key={index} label={`Model ${rowName(index)}: probability of delay`} value={value} min={0} max={1} onChange={next => setProbabilities(previous => previous.map((old, position) => position === index ? next : old))} />)}
      <Range label="Relative weight of model C" value={lastWeight} min={0} max={4} step={0.25} onChange={setLastWeight} />
    </div>
    <div className="ensemble-ballots" aria-label="Weighted votes">
      {state.labels.map((label, index) => <div key={index}><span>Model {rowName(index)}</span><strong>{label === 1 ? 'DELAY' : 'ON TIME'}</strong><span>weight share {number(state.shares[index])}</span></div>)}
    </div>
    <div className="ensemble-comparison" aria-live="polite">
      <p><strong>Hard vote → {state.hardClass === 1 ? 'delay' : 'on time'}</strong><br />Share voting delay: {number(state.ballotMass)}.</p>
      <p><strong>Probability mean → {state.softClass === 1 ? 'delay' : 'on time'}</strong><br />Mean delay probability: {number(state.probability)}.</p>
    </div>
    <p>The weight multiplies the model's contribution to either rule. A ballot discards the difference between 0.51 and 0.99; a probability mean retains it. Keeping that detail helps only when the probabilities carry useful information.</p>
    <button type="button" onClick={() => {
      setProbabilities([0.51, 0.51, 0.01]);
      setLastWeight(1);
    }}>Reset forecasts</button>
  </Investigation>;
}
export function ErrorCancellationLab() {
  const [preset, setPreset] = useState('complementary');
  const [weight, setWeight] = useState(0.5);
  const state = errorBlendState(preset, weight);
  return <Investigation id="ensemble-errors" kicker="AVERAGE SIGNED ERRORS" title="Move the prediction, then square its error">
    <Predict>Can a weighted average have less squared error than both members? Predict what changes when both members are wrong in the same direction.</Predict>
    <label className="ensemble-select">Held-out error pattern<select value={preset} onChange={event => {
        setPreset(event.target.value);
        setWeight(0.5);
      }}>{Object.entries(errorBlendPresets).map(([key, value]) => <option key={key} value={key}>{value.label}</option>)}</select></label>
    <Range label="Weight on forecast A" value={weight} onChange={setWeight} min={0} max={1} />
    <p>Error means prediction minus actual hours. Left of zero is too early; right is too late. The target and both members remain fixed when the weight changes.</p>
    <div className="ensemble-error-rows">
      {state.rows.map(row => <figure key={row.id}>
        <figcaption>Shipment {row.id}</figcaption>
        <svg viewBox="0 0 300 130" role="img" aria-label={`Shipment ${row.id}: A error ${row.a}, B error ${row.b}, combined error ${number(row.blended)} hours`}>
          <line x1="150" x2="150" y1="22" y2="116" className="ensemble-zero" />
          <text x="150" y="15" textAnchor="middle">zero error</text>
          {[['A', row.a, colors.a], ['B', row.b, colors.b], ['mix', row.blended, colors.blend]].map(([label, value, color], index) => <g key={label}>
            <text x="8" y={40 + index * 31} fill={color}>{label}</text>
            <line x1="150" x2={150 + value * 30} y1={35 + index * 31} y2={35 + index * 31} stroke={color} strokeWidth="3" />
            <circle cx={150 + value * 30} cy={35 + index * 31} r="4" fill={color} />
            <text x={150 + value * 30 + (Math.abs(value) < 1e-8 ? 9 : 0)} y={54 + index * 31} textAnchor={Math.abs(value) < 1e-8 ? 'start' : 'middle'} fill={color}>{number(value, 2)}</text>
          </g>)}
        </svg>
      </figure>)}
    </div>
    <LessonTable caption="Squared error on these four fixed outcome rows" headers={['Quantity', 'Hours²']} rows={[['A mean squared error', number(state.mseA)], ['B mean squared error', number(state.mseB)], ['Weighted mean of member losses', number(state.averageMemberLoss)], ['Disagreement term', number(state.disagreement)], ['Loss of combined prediction', number(state.mse)]]} />
    <p className="ensemble-readout" aria-live="polite">{number(state.averageMemberLoss)} − {number(state.disagreement)} = {number(state.mse)} hours². {state.optimum === null ? 'Identical errors make every weight equivalent.' : `The best convex weight for these four labeled rows is ${number(state.optimum)} on A.`}</p>
    <p>Choosing this weight uses the outcomes. That makes these rows combination-training data; use other reserved rows to evaluate the chosen rule.</p>
  </Investigation>;
}
function BootstrapFitPlot({
  fit,
  inspectedRow
}) {
  const px = value => 35 + value * 52;
  const py = value => 207 - value * 20;
  const thresholdX = fit.threshold === null ? 321 : px(fit.threshold);
  return <svg viewBox="0 0 330 276" role="img" aria-label="Actual regression stump fitted to the displayed bootstrap sample">
    {[0, 2, 4, 6, 8].map(value => <g key={value}><line x1="35" x2="309" y1={py(value)} y2={py(value)} className="ensemble-grid" /><text x="27" y={py(value) + 5} textAnchor="end">{value}</text></g>)}
    <path d="M35,24V207H309" className="ensemble-axis" />
    <path d={`M35,${py(fit.leftMean)}H${Math.min(thresholdX, 309)}${fit.threshold === null ? '' : `V${py(fit.rightMean)}H309`}`} stroke={colors.blend} fill="none" strokeWidth="3" />
    {fit.threshold !== null && <line x1={thresholdX} x2={thresholdX} y1="24" y2="207" className="ensemble-split" />}
    {bootstrapRows.map((row, index) => <g key={row.id}>
      <circle cx={px(row.x)} cy={py(row.y)} r={fit.counts[index] ? 5 * Math.sqrt(fit.counts[index]) : 5} fill={fit.counts[index] ? colors.a : '#12181b'} stroke={inspectedRow === index ? colors.blend : colors.muted} strokeWidth={inspectedRow === index ? 3 : 1} />
      <text x={px(row.x)} y="229" textAnchor="middle">{row.id}</text>
    </g>)}
    <text x="13" y="16">hours</text><text x="170" y="269" textAnchor="middle">row / input x = 0…5</text>
  </svg>;
}
export function BootstrapOwnershipLab() {
  const [preset, setPreset] = useState('repeated');
  const [bag, setBag] = useState(0);
  const [row, setRow] = useState(0);
  const state = bootstrapState(preset, bag, row);
  const fit = state.fits[bag];
  return <Investigation id="ensemble-bootstrap" kicker="DRAW ROWS, FIT, THEN CHECK ELIGIBILITY" title="A repeated row counts repeatedly in this fitted rule">
    <Predict>In bag 1, A appears twice. May that fitted model provide an out-of-bag prediction for A? Predict which other models are allowed.</Predict>
    <div className="ensemble-controls">
      <label className="ensemble-select">Draw preset<select value={preset} onChange={event => {
          setPreset(event.target.value);
          setBag(0);
        }}><option value="repeated">Six draws with repetition</option><option value="allRows">Every row appears once</option><option value="smallBags">Only three draws per bag</option></select></label>
      <label className="ensemble-select">Inspect outcome row<select value={row} onChange={event => setRow(Number(event.target.value))}>{bootstrapRows.map((value, index) => <option value={index} key={value.id}>{value.id}: x={value.x}, y={value.y} hours</option>)}</select></label>
    </div>
    <Stepper step={bag} count={3} setStep={setBag} onReset={() => {
      setPreset('repeated');
      setBag(0);
      setRow(0);
    }} />
    <p><strong>Bag {bag + 1} draws, in order</strong></p>
    <ol className="ensemble-draws" aria-label={`Drawn row identities for bag ${bag + 1}`}>{state.draws[bag].map((index, position) => <li key={position}>{rowName(index)}</li>)}</ol>
    <ScrollFigure compact label="Bootstrap fitted rule" caption="Green point area encodes its multiplicity in this bag. Hollow points were not drawn. The amber step is the least-squares two-leaf fit; a highlighted outline marks the inspected row."><BootstrapFitPlot fit={fit} inspectedRow={row} /></ScrollFigure>
    <p>Threshold: {fit.threshold === null ? 'none; one constant leaf' : `x ≤ ${number(fit.threshold)}`}. Left prediction {number(fit.leftMean)} hours; right prediction {number(fit.rightMean)} hours. Training SSE counts every repeated draw: {number(fit.sse)} hours².</p>
    <LessonTable caption={`Which fitted rules may predict ${rowName(row)} out of bag?`} headers={['Bag', 'Copies of row', 'Prediction', 'OOB use']} rows={state.fits.map((current, index) => [String(index + 1), current.counts[row], number(current.predictions[row]), current.counts[row] === 0 ? 'eligible' : 'excluded'])} />
    <p className="ensemble-readout" aria-live="polite">All-model mean: {number(state.ensemblePrediction)} hours. OOB mean for {rowName(row)}: {state.oobPrediction === null ? 'unavailable — no eligible model' : `${number(state.oobPrediction)} hours from ${state.eligible.length} eligible model(s)`}.</p>
    <p>These named samples are reproducible examples, not a random performance study. Under independent uniform draws, a specified row would be absent with probability {number(state.missingProbability)}; the expected number of represented rows is {number(state.expectedRepresented)} out of 6.</p>
  </Investigation>;
}
function WeightedStumpPlot({
  x,
  y,
  frame
}) {
  const minimum = Math.min(...x) - 0.6;
  const maximum = Math.max(...x) + 0.6;
  const px = value => 47 + (value - minimum) / (maximum - minimum) * 422;
  const split = px(frame.stump.threshold);
  return <svg viewBox="0 0 510 235" role="img" aria-label={`Round ${frame.round} stump, fitted with the weights shown by point areas`}>
    <rect x="47" y="24" width={Math.max(0, Math.min(469, split) - 47)} height="156" fill={frame.stump.polarity === 1 ? '#1b2928' : '#27202a'} />
    <rect x={Math.max(47, split)} y="24" width={Math.max(0, 469 - Math.max(47, split))} height="156" fill={frame.stump.polarity === 1 ? '#27202a' : '#1b2928'} />
    {[1, -1].map(label => <g key={label}><line x1="47" x2="469" y1={label === 1 ? 65 : 147} y2={label === 1 ? 65 : 147} className="ensemble-grid" /><text x="34" y={label === 1 ? 70 : 152} textAnchor="end">{label === 1 ? '+1' : '−1'}</text></g>)}
    <line x1={split} x2={split} y1="24" y2="180" className="ensemble-split" />
    {x.map((value, index) => <g key={index}>
      <circle cx={px(value)} cy={y[index] === 1 ? 65 : 147} r={40 * Math.sqrt(frame.before[index])} fill={y[index] === 1 ? colors.a : colors.b} stroke={frame.stump.predictions[index] === y[index] ? '#151b1d' : colors.bad} strokeWidth={frame.stump.predictions[index] === y[index] ? 1 : 4} />
      <text x={px(value)} y={(y[index] === 1 ? 65 : 147) - 40 * Math.sqrt(frame.before[index]) - 6} textAnchor="middle" className="ensemble-point-id">{rowName(index)}</text>
    </g>)}
    {[...new Set(x)].map(value => <text key={value} x={px(value)} y="205" textAnchor="middle">{value}</text>)}
    <text x="260" y="230" textAnchor="middle">input x; vertical lane = actual label</text>
  </svg>;
}
export function AdaBoostWeightsLab() {
  const [preset, setPreset] = useState('mixed');
  const [step, setStep] = useState(0);
  const trace = signedBoostingTrace(preset);
  const frame = trace.frames[Math.min(step, trace.frames.length - 1)];
  return <Investigation id="ensemble-boosting" kicker="FIT USING WEIGHTS, THEN UPDATE THEM" title="One threshold changes which observations matter next">
    <Predict>With six equally weighted cases, the best first stump misses only E. Predict E's share after the normalized update. Does a large point mean a confident model probability?</Predict>
    <label className="ensemble-select">Training cases<select value={preset} onChange={event => {
        setPreset(event.target.value);
        setStep(0);
      }}>{Object.entries(boostingPresets).map(([key, value]) => <option key={key} value={key}>{value.label}</option>)}</select></label>
    <Stepper step={step} count={trace.frames.length} setStep={setStep} onReset={() => {
      setPreset('mixed');
      setStep(0);
    }} />
    <ScrollFigure label="Weighted classification cases and fitted stump" caption={`Point area is the row weight used to fit round ${frame.round}. True label is its vertical lane; a red outline marks this stump's mistake. The threshold predicts ${frame.stump.polarity > 0 ? '+1' : '−1'} on the left and ${frame.stump.polarity > 0 ? '−1' : '+1'} on the right.`}><WeightedStumpPlot x={trace.x} y={trace.y} frame={frame} /></ScrollFigure>
    <p>Selected threshold x ≤ {number(frame.stump.threshold)}. Weighted mistake mass ε = {number(frame.stump.error, 6)}.</p>
    {frame.status === 'accepted' ? <>
      <p className="ensemble-readout" aria-live="polite">α = ½ log[(1−ε)/ε] = {number(frame.alpha, 6)}. Normalizer Z = {number(frame.normalizer, 6)}.</p>
      <LessonTable caption={`Round ${frame.round}: fitting weight versus next-round weight and cumulative vote`} headers={['Row', 'Before', 'After', 'Vote F']} rows={trace.x.map((_, index) => [rowName(index), number(frame.before[index]), number(frame.after[index]), number(frame.scores[index])])} />
      <p>Training mistake fraction: {number(frame.trainingError)}. Actual mean exponential loss: {number(frame.loss, 6)}. Product of normalizers: {number(frame.bound, 6)}.</p>
      <p>“After” supplies the next fit. F is a signed accumulated vote, not a calibrated probability. At exactly zero vote this teaching rule chooses +1.</p>
    </> : <div className="ensemble-readout" aria-live="polite">{frame.status === 'perfect' ? <p><strong>Perfect weak learner: stop.</strong> ε=0 has no finite minimizing α. This teaching implementation returns the perfect stump, the limiting dominant vote; it does not print an invented finite alpha. Library boundary conventions may differ.</p> : <p><strong>No positive edge: stop.</strong> The best stump has ε=½. No update is accepted. Identical inputs with opposite labels cannot all be classified correctly by a deterministic function of these inputs. An empty vote uses the declared +1 tie rule, giving training error {number(frame.trainingError)}.</p>}</div>}
  </Investigation>;
}
export function OOFOwnershipLab() {
  const [mode, setMode] = useState('honest');
  const [completed, setCompleted] = useState(1);
  const [query, setQuery] = useState(2.5);
  const state = oofOwnershipState(mode, completed, query);
  const active = state.stages[Math.max(0, completed - 1)];
  return <Investigation id="ensemble-oof" kicker="WHO WAS ALLOWED TO TRAIN ON THIS ROW?" title="Build the combiner's input matrix without leaking its targets">
    <Predict>A one-nearest-neighbor regressor exactly recalls each training row. Would its training-set predictions tell the combiner how it behaves on a new row? Inspect what changes when that row is held out.</Predict>
    <label className="ensemble-select">Base-fit ownership<select value={mode} onChange={event => {
        setMode(event.target.value);
        setCompleted(1);
      }}><option value="honest">Hold out each prediction row</option><option value="leaky">Incorrect: fit all rows first</option></select></label>
    <Stepper step={completed} count={4} setStep={setCompleted} onReset={() => {
      setMode('honest');
      setCompleted(1);
      setQuery(2.5);
    }} />
    <div className={'ensemble-ownership ' + (mode === 'leaky' ? 'ensemble-leaking' : '')}>
      <p><strong>{completed ? `Fold ${completed} just filled` : 'No folds filled yet'}</strong></p>
      <p>Held-out destinations: <strong>{active.held.map(rowName).join(', ')}</strong>.</p>
      <p>Rows used by both base fits: <strong>{active.train.map(rowName).join(', ')}</strong>.</p>
      <p>{mode === 'honest' ? 'None of these destinations occurs in this fit set.' : 'The destinations occur in the fit set: their targets can influence their own input predictions.'}</p>
    </div>
    <LessonTable caption="Combiner-training matrix: one row per observed shipment" headers={['ID', 'Target', 'Nearest', 'Line']} rows={bootstrapRows.map((row, index) => [row.id, row.y, number(state.matrix[index][0]), number(state.matrix[index][1])])} />
    <p>Inputs x are 0,1,2,3,4,5 for A…F. Empty cells mean unavailable predictions; they are never replaced by zero. The target belongs beside the matrix for fitting the combiner, not inside that row's held-out base fit.</p>
    {state.weightNearest === null ? <p className="ensemble-readout">Fill all three folds before fitting the combination weight.</p> : <>
      <p className="ensemble-readout" aria-live="polite">Learned convex weight on nearest neighbor: {number(state.weightNearest, 6)}. On the line: {number(1 - state.weightNearest, 6)}. Combiner-training MSE: {number(state.trainMse, 6)} hours².</p>
      <Range label="New input x" value={query} onChange={setQuery} min={0} max={5} step={0.1} />
      <LessonTable caption="Inference uses full-data refits, then the saved combination weight" headers={['Saved rule', 'Hours']} rows={[['Nearest neighbor', number(state.nearest)], ['Line', number(state.linear)], ['Combination', number(state.ensemble)]]} />
      <p>Both bases were refitted on A…F; the combiner was not refitted on their in-sample outputs. Nearest-neighbor distance ties choose the earliest row. These few toy rows demonstrate ownership and computation, not held-out accuracy.</p>
    </>}
  </Investigation>;
}
export function TimeOwnershipFigure() {
  return <figure className="ensemble-time" aria-label="A forward-only meta-training construction has an uncovered initial prefix">
    <div><strong>Fit 1</strong><span className="ensemble-time-train">past A B</span><span className="ensemble-time-predict">predict C D</span><span>future E F</span></div>
    <div><strong>Fit 2</strong><span className="ensemble-time-train">past A B C D</span><span className="ensemble-time-predict">predict E F</span></div>
    <div><strong>Meta rows</strong><span className="ensemble-time-gap">A B: unavailable</span><span className="ensemble-time-predict">C D E F: covered</span></div>
    <figcaption>Time runs left to right within each row. The initial training prefix has no earlier fitted model here. A manual stack uses only covered rows to fit its combiner. Actual label-availability delays may require a gap too.</figcaption>
  </figure>;
}
export function CalibratedAverageFigure() {
  const rows = calibratedAverageLaw();
  return <figure className="ensemble-calibration" aria-label="Two calibrated probabilities can have a mean that is not calibrated">
    <LessonTable caption="Four equally likely signal pairs; every entry is an exact model probability" headers={['Signals A,B', 'True chance', 'A forecast', 'B forecast', 'Mean']} rows={rows.map(row => [`${row.a}, ${row.b}`, number(row.trueChance), number(row.forecastA), number(row.forecastB), number(row.average)])} />
    <figcaption>Each forecaster knows only its own signal. Grouping by one forecast gives its stated event rate. Grouping by their mean gives event rates 0, ½ and 1 at mean forecasts ¼, ½ and ¾. This finite model is not an empirical reliability plot.</figcaption>
  </figure>;
}
export function ContextInteractionFigure() {
  return <figure className="ensemble-context" aria-label="Context-prediction products change a model's effective coefficient">
    <div><strong>Original meta row</strong><code>[pA, pB, context]</code><span>A linear fit gives fixed slopes on pA and pB.</span></div>
    <p className="ensemble-flow-arrow" aria-hidden="true">↓ form products</p>
    <div><strong>Interaction meta row</strong><code>[(1−context)pA, context·pB]</code><span>At context 0, use pA. At context 1, use pB. Intermediate context produces a blend.</span></div>
    <figcaption>This chosen formula illustrates the representation. A learned feature-weighted stack estimates its own coefficients from honest meta-training rows and needs separate evaluation.</figcaption>
  </figure>;
}
const mapLabels = {
  line: 'Linear logistic rule',
  tree: 'Depth-four tree',
  neighbor: 'Nine nearest neighbors',
  soft: 'Equal probability average',
  stack: 'OOF logistic stack'
};
export function EnsemblePredictionLab() {
  const [model, setModel] = useState('stack');
  const [horizontal, setHorizontal] = useState(16);
  const [vertical, setVertical] = useState(12);
  const coordinates = predictionMap.coordinates;
  const index = vertical * coordinates.length + horizontal;
  const selected = predictionMap.models[model];
  const px = value => 49 + (value + 1.5) * 92;
  const py = value => 300 - (value + 1.5) * 92;
  const cell = 276 / 24;
  const color = value => `rgb(${Math.round(64 + 74 * value)}, ${Math.round(97 - 25 * value)}, ${Math.round(112 + 26 * value)})`;
  return <Investigation id="ensemble-boundaries" kicker="ACTUAL FITTED RULES, ONE FIXED EXPERIMENT" title="Which differences survive the combination?">
    <Predict>The line cannot surround an inner circle with one linear boundary. Will adding its probabilities necessarily improve a neighbor model that can follow the ring? Compare the saved models and their validation losses.</Predict>
    <p>These are native scikit-learn predictions from the complete experiment below: 216 training, 72 validation and 72 test points. Both features are dimensionless synthetic coordinates. Controls inspect fixed fitted models; they do not retrain them.</p>
    <label className="ensemble-select">Saved fitted rule<select value={model} onChange={event => setModel(event.target.value)}>{Object.entries(mapLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
    <div className="ensemble-controls">
      <Range label="Horizontal grid index" value={horizontal} onChange={setHorizontal} min={0} max={24} step={1} />
      <Range label="Vertical grid index" value={vertical} onChange={setVertical} min={0} max={24} step={1} />
    </div>
    <ScrollFigure compact label="Native-fitted probability field" caption="Blue cells mean lower P(class 1); purple cells mean higher P(class 1). Each cell uses its grid-center probability, so the field is a sampled approximation. Hollow circles are test class 0; filled amber circles are test class 1. The white cross selects a grid point.">
      <svg className="ensemble-map" viewBox="0 0 360 353" role="img" aria-label={`${mapLabels[model]} probability field; query (${coordinates[horizontal]}, ${coordinates[vertical]}) reports ${number(selected.probabilities[index], 6)}`}>
        {selected.probabilities.map((probability, position) => {
          const column = position % 25;
          const row = Math.floor(position / 25);
          const x = px(coordinates[column]);
          const y = py(coordinates[row]);
          return <rect key={position} x={Math.max(49, x - cell / 2)} y={Math.max(24, y - cell / 2)} width={column === 0 || column === 24 ? cell / 2 : cell} height={row === 0 || row === 24 ? cell / 2 : cell} fill={color(probability)} data-probability={probability} />;
        })}
        {predictionMap.testPoints.map((point, position) => <circle key={position} cx={px(point.x)} cy={py(point.y)} r="3.8" fill={point.label ? colors.blend : '#101719'} stroke={point.label ? '#181a15' : '#e8eff0'} strokeWidth="1.2" />)}
        <path d={`M${px(coordinates[horizontal]) - 7},${py(coordinates[vertical])}h14 M${px(coordinates[horizontal])},${py(coordinates[vertical]) - 7}v14`} stroke="#fff" strokeWidth="2.5" />
        <path d="M49,24V300H325" className="ensemble-axis" />
        {[-1.5, 0, 1.5].map(value => <g key={value}><text x={px(value)} y="324" textAnchor={value === -1.5 ? 'start' : value === 1.5 ? 'end' : 'middle'}>{value}</text><text x="41" y={py(value) + 6} textAnchor="end">{value}</text></g>)}
        <text x="187" y="347" textAnchor="middle">horizontal feature x₁</text>
        <text x="49" y="17">vertical feature x₂</text>
      </svg>
    </ScrollFigure>
    <p className="ensemble-readout" aria-live="polite">Query x₁={coordinates[horizontal]}, x₂={coordinates[vertical]}. {mapLabels[model]}: P(class 1)={number(selected.probabilities[index], 6)}. Equal-cost argmax class: {selected.probabilities[index] > 0.5 ? '1' : '0'}.</p>
    <LessonTable caption="Same query and validation rows, different saved models" headers={['Rule', 'Query P(1)', 'Validation log loss']} rows={Object.entries(predictionMap.models).map(([key, value]) => [mapLabels[key], number(value.probabilities[index], 6), number(value.report.validationLogLoss, 6)])} />
    <p>Validation selected <strong>{mapLabels[predictionMap.selected]}</strong>. The displayed class rule is the ordinary 0.5 comparison, not the separate cost threshold selected in the report. Visual smoothness is not evidence of better accuracy or calibrated probabilities.</p>
    <button type="button" onClick={() => {
      setModel('stack');
      setHorizontal(16);
      setVertical(12);
    }}>Reset fitted-map view</button>
  </Investigation>;
}
