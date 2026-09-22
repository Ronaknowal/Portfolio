import { useId, useState } from 'react';
import { labelJointDecisions, multioutputMetrics, multioutputPredictions, multioutputTruth, pooledLabelAssociation, sharedFeatureShrinkage, sharedOutputStump, thresholdInspection } from '../../data/multioutput-models.js';
import './multioutput-labs.css';
const gold = '#e8b44a',
  blue = '#75b7ed',
  mint = '#78cfab',
  rose = '#e899a9';
const number = value => value === null ? 'undefined' : Number(value.toFixed(4)).toString();
const vector = values => values.join('');
function Range({
  label,
  value,
  change,
  min = 0,
  max = 1,
  step = .05
}) {
  const id = useId();
  return <label className="mo-control" htmlFor={id}><span>{label} <output>{number(value)}</output></span><input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => change(Number(event.target.value))} /></label>;
}
function Investigation({
  title,
  prompt,
  reset,
  children
}) {
  return <section className="mo-investigation" aria-label={title} data-live-exploration><header><h3>{title}</h3><button onClick={reset}>Reset</button></header><p>{prompt}</p>{children}</section>;
}
export function LabelErrorLab() {
  const [prediction, setPrediction] = useState(multioutputPredictions.map(row => [...row]));
  const [mask, setMask] = useState(false);
  const truth = multioutputTruth.map((row, index) => row.map((value, column) => mask && index === 3 && column === 1 ? null : value));
  const result = multioutputMetrics(truth, prediction);
  const labels = ['code', 'hardware', 'energy'];
  return <Investigation title="Read errors by message and by label" prompt="Correct message 4's hardware prediction. Which totals change? Then hide that annotation: is an unknown value a correct negative?" reset={() => {
    setPrediction(multioutputPredictions.map(row => [...row]));
    setMask(false);
  }}>
    <label className="mo-checkbox"><input type="checkbox" checked={mask} onChange={event => setMask(event.target.checked)} /> Message 4 hardware annotation is unknown</label>
    <div className="mo-error-grid" role="group" aria-label="Toggle predicted labels"><span />{labels.map(label => <strong key={label}>{label}</strong>)}<strong>exact set</strong>{truth.map((row, rowIndex) => <div className="mo-contents" key={rowIndex}><strong>M{rowIndex + 1}</strong>{row.map((target, column) => <button key={column} className={`mo-cell mo-${result.rows[rowIndex].cells[column]}`} aria-pressed={Boolean(prediction[rowIndex][column])} aria-label={`Message ${rowIndex + 1}, ${labels[column]}, observed ${target === null ? 'unknown' : target}, predicted ${prediction[rowIndex][column]}`} onClick={() => setPrediction(old => old.map((values, index) => index === rowIndex ? values.map((value, label) => label === column ? 1 - value : value) : values))}><small>truth {target ?? '?'}</small><strong>pred {prediction[rowIndex][column]}</strong><span>{result.rows[rowIndex].cells[column].toUpperCase()}</span></button>)}<span className="mo-row-result">{result.rows[rowIndex].exact === null ? 'unknown' : result.rows[rowIndex].exact ? 'yes' : 'no'}</span></div>)}<strong>label F1</strong>{result.labels.map((label, index) => <span key={index}>{number(label.f1)}</span>)}<span /></div>
    <p className="mo-caption">Each button changes a prediction, never its observed truth. Rose FN/FP cells are mistakes; blue TN and green TP are distinct correct decisions. Unknown gray cells leave the observed-cell denominator.</p>
    <dl className="mo-readouts"><div><dt>Observed decisions</dt><dd>{result.observed}</dd></div><div><dt>TP / FP / FN</dt><dd>{result.tp} / {result.fp} / {result.fn}</dd></div><div><dt>Hamming loss</dt><dd>{number(result.hamming)}</dd></div><div><dt>Micro-F1</dt><dd>{number(result.f1)}</dd></div><div><dt>Macro-F1</dt><dd>{number(result.macroF1)}</dd></div><div><dt>Exact-set accuracy</dt><dd>{number(result.subsetAccuracy)}</dd></div></dl>
    <p aria-live="polite">Exact-set accuracy uses {result.completeRowCount} fully annotated messages. {result.missing} annotation is unknown. F1 uses zero for an observed label with a zero F1 denominator; an entirely unobserved label would be excluded and reported undefined.</p>
  </Investigation>;
}
function ProbabilityTree({
  result
}) {
  // Fixed viewbox fits narrow reading columns; branch widths do not encode mass.
  const x = [65, 215],
    rootY = 32,
    branchY = 108,
    leafY = 210;
  const firstName = result.order[0],
    secondName = result.order[1];
  return <svg className="mo-tree" viewBox="0 0 280 288" role="img" aria-label={`Exact ${firstName} then ${secondName} probability tree; multiply each path to get joint leaf mass`}>
    <text x="140" y="18" textAnchor="middle" fill={gold}>one fixed input x</text>
    {result.branches.map((branch, index) => <g key={index}><path d={`M140 ${rootY} L${x[index]} ${branchY - 25}`} stroke={blue} fill="none" /><text x={x[index]} y="65" textAnchor="middle" fill={blue}>{number(branch.support)}</text><text x={x[index]} y={branchY} textAnchor="middle" fill={gold}>{firstName} = {index}</text>{branch.leaves.map((leaf, leafIndex) => {
        const leafX = x[index] + (leafIndex ? 32 : -32);
        return <g key={leaf.index}><path d={`M${x[index]} ${branchY + 10} L${leafX} ${leafY - 20}`} stroke={mint} fill="none" /><text x={leafX} y="156" textAnchor="middle" fill={mint}>{number(leaf.conditional)}</text><text x={leafX} y={leafY} textAnchor="middle" fill="currentColor">{vector(leaf.state)}</text><rect x={leafX - 23} y="222" width="46" height="29" fill={gold} fillOpacity=".12" stroke={gold} /><text x={leafX} y="242" textAnchor="middle" fill={gold}>{number(leaf.mass)}</text></g>;
      })}</g>)}
    <text x="140" y="279" textAnchor="middle" fill="currentColor">leaves remain in AB label order</text>
  </svg>;
}
export function JointDecisionLab() {
  const [counts, setCounts] = useState([6, 5, 1, 8]);
  const [order, setOrder] = useState('AB');
  const total = counts.reduce((sum, value) => sum + value, 0);
  const result = total ? labelJointDecisions(counts, order) : null;
  return <Investigation title="One joint table, three different decisions" prompt="Keep the four masses unchanged and swap the chain order. Should the best whole-set decision move? Should a greedy path move?" reset={() => {
    setCounts([6, 5, 1, 8]);
    setOrder('AB');
  }}>
    <div className="mo-controls">{['00', '01', '10', '11'].map((state, index) => <Range key={state} label={`Count ${state}`} value={counts[index]} min={0} max={20} step={1} change={value => setCounts(old => old.map((count, column) => column === index ? value : count))} />)}<label>Chain order <select aria-label="Chain order" value={order} onChange={event => setOrder(event.target.value)}><option value="AB">A then B</option><option value="BA">B then A</option></select></label></div>
    {!result ? <p role="alert">A probability distribution needs at least one positive count. Increase a count or reset.</p> : <><div className="mo-joint-layout"><ProbabilityTree result={result} /><div className="mo-leaf-table"><table><caption>Exact expected losses under the entered table</caption><thead><tr><th>Action</th><th>Joint mass</th><th>Hamming risk</th><th>Subset risk</th></tr></thead><tbody>{result.candidates.map(candidate => <tr key={vector(candidate.prediction)}><th>{vector(candidate.prediction)}</th><td>{number(candidate.mass)}</td><td>{number(candidate.hammingRisk)}</td><td>{number(candidate.subsetLoss)}</td></tr>)}</tbody></table><p>Marginals: P(A=1) = {number(result.marginals[0])}; P(B=1) = {number(result.marginals[1])}.</p></div></div>
    <dl className="mo-readouts" aria-live="polite"><div><dt>Marginal threshold action</dt><dd>{vector(result.marginalDecision)}</dd></div><div><dt>Greedy {order}</dt><dd>{vector(result.greedy)}</dd></div><div><dt>All joint modes</dt><dd>{result.modes.map(vector).join(', ')}</dd></div></dl><p className="mo-caption">Exact finite probabilities, rounded to four decimal places. Branch labels are conditional probabilities; bottom boxes are joint masses. A zero-support branch has undefined conditionals and zero leaves. Binary decisions use 1 on a .5 tie; all joint modes are listed. This is a supplied distribution, not a trained model or causal diagram.</p></>}
  </Investigation>;
}
function ProbabilityMosaic({
  mass,
  title
}) {
  const left = mass[0] + mass[1],
    right = mass[2] + mass[3];
  const rectangles = [{
    x: 0,
    y: 0,
    width: left,
    height: left ? mass[0] / left : 0,
    state: '00',
    value: mass[0]
  }, {
    x: 0,
    y: left ? mass[0] / left : 0,
    width: left,
    height: left ? mass[1] / left : 0,
    state: '01',
    value: mass[1]
  }, {
    x: left,
    y: 0,
    width: right,
    height: right ? mass[2] / right : 0,
    state: '10',
    value: mass[2]
  }, {
    x: left,
    y: right ? mass[2] / right : 0,
    width: right,
    height: right ? mass[3] / right : 0,
    state: '11',
    value: mass[3]
  }];
  return <figure><figcaption>{title}</figcaption><svg viewBox="0 0 200 200" role="img" aria-label={`${title}; area of each labeled rectangle equals its joint probability`}>
    {rectangles.map((rectangle, index) => <g key={rectangle.state}><rect x={rectangle.x * 190 + 5} y={rectangle.y * 190 + 5} width={rectangle.width * 190} height={rectangle.height * 190} fill={[blue, mint, rose, gold][index]} fillOpacity=".65" stroke="#151719" />{rectangle.width > .15 && rectangle.height > .1 && <text x={(rectangle.x + rectangle.width / 2) * 190 + 5} y={(rectangle.y + rectangle.height / 2) * 190 + 10} textAnchor="middle" fill="#111">{rectangle.state}</text>}</g>)}
  </svg><p className="mo-caption">00 {number(mass[0])} · 01 {number(mass[1])}<br />10 {number(mass[2])} · 11 {number(mass[3])}</p></figure>;
}
export function ConditionalAssociationLab() {
  const [share, setShare] = useState(.5);
  const result = pooledLabelAssociation({
    highShare: share
  });
  return <Investigation title="Unmix a co-occurrence table" prompt="At either pure group, labels are independent. Can mixing those groups create apparent association?" reset={() => setShare(.5)}>
    <Range label="Fraction in high-probability group" value={share} change={setShare} />
    <div className="mo-mosaics"><ProbabilityMosaic mass={result.strata[0].mass} title="X = low: each marginal .1" /><ProbabilityMosaic mass={result.pooled} title="Pooled: X is hidden" /><ProbabilityMosaic mass={result.strata[1].mass} title="X = high: each marginal .9" /></div>
    <p aria-live="polite">Pooled P(B=1) = {number(result.marginal)}, but P(B=1 | A=1) = {number(result.conditional)}. The conditioning group has probability {number(result.marginal)}. Within each X group, the conditional stays .1 or .9, equal to its own marginal.</p><p className="mo-caption">Exact mixture model: rectangle area represents mass. Small regions are identified in the numeric legend. X may explain pooled association; whether a fitted model captures X well is a separate empirical question.</p>
  </Investigation>;
}
export function ValidationThresholdLab() {
  const [threshold, setThreshold] = useState(.5);
  const result = thresholdInspection({
    threshold
  });
  const x = value => 35 + value / 1.01 * 220,
    y = value => 155 - value * 120;
  return <Investigation title="Move a decision gate across tied scores" prompt="The two .8 scores are tied. Can this single threshold select the positive one without also selecting the negative one?" reset={() => setThreshold(.5)}>
    <Range label="Validation threshold" value={threshold} change={setThreshold} max={1.01} step={.01} />
    <div className="mo-threshold-layout"><div className="mo-score-rail" aria-label="Validation candidates ranked by score">{result.ranked.map(row => <div key={row.index} className={row.selected ? 'mo-selected-score' : ''}><span>row {row.index + 1}</span><strong>{row.score}</strong><span>truth {row.target}</span><span>{row.selected ? 'selected' : 'off'}</span></div>)}</div><figure><svg viewBox="0 0 280 210" role="img" aria-label="Exact validation F1 staircase; filled right endpoint and hollow left endpoint implement score greater than or equal to threshold"><path d="M35 30 V155 H255" stroke="currentColor" fill="none" />{[0, .5, 1].map(value => <g key={value}><text x="28" y={y(value) + 4} textAnchor="end">{value}</text><text x={x(value)} y="175" textAnchor="middle">{value}</text></g>)}{result.candidates.slice(1).map((candidate, index) => <g key={candidate.threshold}><line x1={x(result.candidates[index].threshold)} x2={x(candidate.threshold)} y1={y(candidate.f1)} y2={y(candidate.f1)} stroke={gold} strokeWidth="2" /><circle cx={x(result.candidates[index].threshold)} cy={y(candidate.f1)} r="3" fill="#121516" stroke={gold} /><circle cx={x(candidate.threshold)} cy={y(candidate.f1)} r="3" fill={gold} /></g>)}<circle cx={x(0)} cy={y(result.candidates[0].f1)} r="3" fill={gold} /><line x1={x(threshold)} x2={x(threshold)} y1="25" y2="155" stroke={rose} strokeDasharray="4 3" /><circle cx={x(threshold)} cy={y(result.f1)} r="5" fill={rose} /><text x="38" y="19">F1</text><text x="145" y="199" textAnchor="middle">threshold t; select score ≥ t</text></svg><figcaption>Exact finite steps; scores do not move.</figcaption></figure></div>
    <dl className="mo-readouts" aria-live="polite"><div><dt>TP / FP / FN</dt><dd>{result.tp} / {result.fp} / {result.fn}</dd></div><div><dt>Precision</dt><dd>{number(result.precision)}</dd></div><div><dt>Recall</dt><dd>{number(result.recall)}</dd></div><div><dt>F1</dt><dd>{number(result.f1)}</dd></div></dl><p>Best enumerated breakpoint(s): {result.best.map(candidate => number(candidate.threshold)).join(', ')}. Each describes a decision pattern; a whole adjacent interval may give that same pattern. Threshold 1.01 selects none, including a hypothetical score of 1.</p><p className="mo-caption">Horizontal segments are constant on (left, right]; filled right endpoints retain scores equal to the threshold. Zero-denominator precision/F1 is defined as 0. These are six validation observations, not a calibrated probability experiment or a population performance curve.</p>
  </Investigation>;
}
function OutputStumpPlot({
  output,
  threshold,
  title,
  unit
}) {
  const low = Math.min(0, ...output.values),
    high = Math.max(1, ...output.values);
  const x = value => 45 + value * 65,
    y = value => 157 - (value - low) / (high - low) * 115;
  return <figure><figcaption>{title}</figcaption><svg viewBox="0 0 280 205" role="img" aria-label={`${title}; actual observations, fitted leaf means and chosen split ${threshold}`}><path d="M45 27 V157 H250" stroke="currentColor" fill="none" /><text x="8" y="20">{unit}</text>{[low, high].map(value => <text key={value} x="37" y={y(value) + 4} textAnchor="end">{number(value)}</text>)}{[0, 1, 2, 3].map(value => <text key={value} x={x(value)} y="177" textAnchor="middle">{value}</text>)}<line x1={x(threshold)} x2={x(threshold)} y1="27" y2="157" stroke={rose} strokeDasharray="4 3" /><path d={`M${x(0)} ${y(output.leftMean)} H${x(threshold)} M${x(threshold)} ${y(output.rightMean)} H${x(3)}`} fill="none" stroke={gold} strokeWidth="3" />{output.values.map((value, index) => <g key={index}><line x1={x(index)} x2={x(index)} y1={y(value)} y2={y(output.predictions[index])} stroke={mint} strokeWidth="2" /><circle cx={x(index)} cy={y(value)} r="5" fill={blue} /></g>)}<text x="150" y="199" textAnchor="middle">input x</text></svg><p className="mo-caption">Raw-unit SSE {number(output.sse)}; leaf means {number(output.leftMean)}, {number(output.rightMean)}.</p></figure>;
}
export function SharedRegressionLab() {
  const [scale, setScale] = useState(1),
    [separate, setSeparate] = useState(false);
  const result = sharedOutputStump({
    energyScale: scale
  });
  return <Investigation title="A shared split must serve two measurements" prompt="Rescale energy from Wh to units of 100 Wh. Will a shared stump still choose x=.5? What happens when each output can choose its own split?" reset={() => {
    setScale(1);
    setSeparate(false);
  }}>
    <div className="mo-controls"><label>Energy divisor <select aria-label="Energy divisor" value={scale} onChange={event => setScale(Number(event.target.value))}><option value="1">1 Wh</option><option value="10">10 Wh</option><option value="100">100 Wh</option></select></label><label className="mo-checkbox"><input type="checkbox" checked={separate} onChange={event => setSeparate(event.target.checked)} /> Separate output stumps</label></div>
    <div className="mo-output-plots">{['Temperature change', 'Energy'].map((title, index) => {
        const chosen = separate ? result.separate[index] : result.shared;
        return <OutputStumpPlot key={title} title={title} unit={index ? 'Wh' : '°C'} output={chosen.outputs[index]} threshold={chosen.threshold} />;
      })}</div>
    <div className="mo-table-scroll"><table><caption>Common-split candidates: SSE(temperature) + SSE(energy) / divisor²</caption><thead><tr><th>Split</th><th>Temperature SSE</th><th>Energy SSE / divisor²</th><th>Total</th></tr></thead><tbody>{result.candidates.map(candidate => <tr key={candidate.cut}><th>x &lt; {candidate.threshold}</th><td>{number(candidate.outputs[0].sse)}</td><td>{number(candidate.outputs[1].scaledSse)}</td><td>{number(candidate.scaledSse)}</td></tr>)}</tbody></table></div>
    <p aria-live="polite">{separate ? 'Separate best splits: temperature 1.5, energy .5. Each has zero training error in this particular four-row example.' : `Best common split: ${result.shared.threshold}. The loss is measured after the declared scaling.`}</p><p className="mo-caption">Blue points are the four supplied observations; gold horizontal segments are their exact within-leaf means; green lines are residuals. Both plots retain physical units while the selection objective changes. This is training fit, not evidence of better future predictions.</p>
  </Investigation>;
}
export function SharedFeatureLab() {
  const [penalty, setPenalty] = useState(2),
    [first, setFirst] = useState(3),
    [second, setSecond] = useState(4);
  const result = sharedFeatureShrinkage({
    first,
    second,
    penalty
  });
  const x = value => 140 + value * 17,
    y = value => 130 - value * 17;
  return <Investigation title="Shrink one feature across two outputs" prompt="Increase the penalty past the smaller coefficient. Does grouping necessarily set that coefficient to zero first?" reset={() => {
    setPenalty(2);
    setFirst(3);
    setSecond(4);
  }}>
    <div className="mo-controls"><Range label="Unpenalized output 1 coefficient" value={first} change={setFirst} min={-5} max={5} step={.5} /><Range label="Unpenalized output 2 coefficient" value={second} change={setSecond} min={-5} max={5} step={.5} /><Range label="Penalty λ" value={penalty} change={setPenalty} min={0} max={8} step={.25} /></div>
    <svg className="mo-shrink-plot" viewBox="0 0 280 280" role="img" aria-label="Coefficient plane: initial point, radial grouped shrinkage and coordinate-wise separate shrinkage"><path d="M30 130 H250 M140 25 V235" stroke="currentColor" fill="none" />{[-5, 0, 5].map(value => <g key={value}><text x={x(value)} y="148" textAnchor="middle">{value}</text>{value !== 0 && <text x="131" y={y(value) + 4} textAnchor="end">{value}</text>}</g>)}<path d={`M${x(first)} ${y(second)} L${x(result.grouped[0])} ${y(result.grouped[1])}`} stroke={mint} strokeWidth="3" /><path d={`M${x(first)} ${y(second)} L${x(result.separate[0])} ${y(second)} L${x(result.separate[0])} ${y(result.separate[1])}`} stroke={rose} fill="none" strokeDasharray="4 3" /><circle cx={x(first)} cy={y(second)} r="7" fill={blue} /><circle cx={x(result.grouped[0])} cy={y(result.grouped[1])} r="5" fill={mint} /><rect x={x(result.separate[0]) - 4} y={y(result.separate[1]) - 4} width="8" height="8" fill={rose} /><text x="140" y="259" textAnchor="middle">coefficient for output 1 →</text><text x="140" y="18" textAnchor="middle">output 2 ↑</text></svg>
    <p aria-live="polite">Blue initial ({number(first)}, {number(second)}), norm {number(result.norm)}. Green grouped ({result.grouped.map(number).join(', ')}). Rose separate ({result.separate.map(number).join(', ')}).</p><p className="mo-caption">Exact solutions for an orthonormal-design row objective. Grouped shrinkage penalizes the row's Euclidean norm; separate shrinkage penalizes the sum of absolute coefficients. They solve different objectives. Coincident marks are expected at λ=0 or when both solutions vanish.</p>
  </Investigation>;
}
