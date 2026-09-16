import { useId, useState } from 'react';
import { bprPairStep, evaluateRecommendationList, exposurePolicy, factorUpdateTrace, feedbackCell, implicitFactorBlock, neighborhoodPrediction, recommenderCounts, recommenderRatings, rotatedFactors } from '../../data/recommender-models.js';
import './recommender-labs.css';
const gold = '#e8b44a';
const blue = '#75b7ed';
const mint = '#78cfab';
const rose = '#e899a9';
const number = (value, digits = 3) => value === null ? 'undefined' : value === 0 ? '0' : Math.abs(value) < .001 || Math.abs(value) >= 1e5 ? value.toExponential(2) : Number(value.toFixed(digits)).toString();
function Range({
  label,
  value,
  change,
  min = 0,
  max = 1,
  step = .05
}) {
  const id = useId();
  return <label className="rec-control" htmlFor={id}><span>{label} <output>{number(value)}</output></span><input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => change(Number(event.target.value))} /></label>;
}
function Investigation({
  title,
  prompt,
  reset,
  children
}) {
  return <section className="rec-investigation" aria-label={title}><div className="rec-heading"><h3>{title}</h3><button onClick={reset}>Reset</button></div><p className="rec-predict"><strong>Predict first.</strong> {prompt}</p>{children}</section>;
}
function Matrix({
  rows,
  labels,
  caption
}) {
  return <div className="rec-table-scroll"><table className="rec-table"><caption>{caption}</caption><thead><tr><th>Row</th>{labels.map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}><th>{index + 1}</th>{row.map((value, column) => <td key={column}>{typeof value === 'number' ? number(value) : value}</td>)}</tr>)}</tbody></table></div>;
}
export function FeedbackEvidenceLab() {
  const [cell, setCell] = useState([0, 2]);
  const [mode, setMode] = useState('explicit');
  const [prediction, setPrediction] = useState(.5);
  const [alpha, setAlpha] = useState(2);
  const [user, item] = cell;
  const rating = recommenderRatings[user][item];
  const count = recommenderCounts[user][item];
  const result = feedbackCell({
    rating,
    count,
    prediction,
    alpha,
    mode
  });
  return <Investigation title="Read a missing cell correctly" prompt="Select user 0, item 2. Will switching objectives add a term even though no rating was observed?" reset={() => {
    setCell([0, 2]);
    setMode('explicit');
    setPrediction(.5);
    setAlpha(2);
  }}>
    <div className="rec-controls"><label>Objective <select aria-label="Objective" value={mode} onChange={event => setMode(event.target.value)}><option value="explicit">Observed ratings</option><option value="implicit">All-pair implicit feedback</option></select></label><Range label="Proposed score" value={prediction} change={setPrediction} min={-1} max={5} step={.1} />{mode === 'implicit' && <Range label="Confidence multiplier α" value={alpha} change={setAlpha} max={10} step={1} />}</div>
    <div className="rec-evidence-grid" role="group" aria-label="Select a user-item cell"><span />{recommenderRatings[0].map((_, index) => <strong key={index}>I{index}</strong>)}{recommenderRatings.map((row, rowIndex) => <div className="rec-grid-row" key={rowIndex}><strong>U{rowIndex}</strong>{row.map((value, column) => <button key={column} className={value === null ? 'rec-missing' : 'rec-observed'} aria-pressed={rowIndex === user && column === item} aria-label={`User ${rowIndex}, item ${column}: ${value === null ? 'missing rating' : `rating ${value}`}; count ${recommenderCounts[rowIndex][column]}`} onClick={() => setCell([rowIndex, column])}>{mode === 'explicit' ? value ?? '?' : recommenderCounts[rowIndex][column]}</button>)}</div>)}</div>
    <p className="rec-caption">Same 5 × 7 invented catalogue. Gold cells contain observed ratings; ? means missing. Implicit mode displays event counts, including observed absence of an event in this window.</p>
    <div className="rec-equation-flow"><span>U{user} → I{item}</span><span>{result.included ? `target ${number(result.target)}` : 'no target'}</span><span>weight {result.weight}</span><strong>loss {number(result.loss)}</strong></div>
    <p aria-live="polite">{result.included ? `${result.weight} × (${number(result.target)} − ${number(prediction)})² = ${number(result.loss)}. ` : 'The explicit objective omits this cell; it does not train toward zero. '}{mode === 'implicit' ? `Count ${count} becomes target ${Number(count > 0)} with confidence ${result.weight}. A zero target here is a weak modeling target, not a verified dislike.` : `The stored rating is ${rating === null ? 'missing' : rating}. Other unseen scores are not evaluated by this term.`}</p>
  </Investigation>;
}
export function NeighborhoodEvidenceLab() {
  const [item, setItem] = useState(2);
  const [centered, setCentered] = useState(false);
  const [minimumOverlap, setMinimumOverlap] = useState(1);
  const [shrinkage, setShrinkage] = useState(0);
  const [signed, setSigned] = useState(false);
  const result = neighborhoodPrediction({
    item,
    centered,
    minimumOverlap,
    shrinkage,
    signed
  });
  const [inspected, setInspected] = useState(0);
  const evidence = result.evidence.find(entry => entry.item === inspected) ?? result.evidence[0];
  return <Investigation title="Audit a neighbor before trusting its score" prompt="A similarity of 1 may come from only one person. Raise minimum overlap, then center each person's ratings. Which evidence survives?" reset={() => {
    setItem(2);
    setCentered(false);
    setMinimumOverlap(1);
    setShrinkage(0);
    setSigned(false);
    setInspected(0);
  }}>
    <div className="rec-controls"><label>Unseen target for user 0 <select aria-label="Unseen target for user 0" value={item} onChange={event => setItem(Number(event.target.value))}>{[2, 4, 6].map(value => <option key={value} value={value}>Item {value}</option>)}</select></label><Range label="Minimum co-raters" value={minimumOverlap} change={setMinimumOverlap} min={1} max={4} step={1} /><Range label="Shrinkage s" value={shrinkage} change={setShrinkage} max={6} step={1} /><label><input type="checkbox" checked={centered} onChange={event => setCentered(event.target.checked)} /> Center by each co-rater's mean</label><label><input type="checkbox" checked={signed} onChange={event => setSigned(event.target.checked)} /> Allow negative weights</label></div>
    <div className="rec-neighbors">{result.evidence.map(entry => <button key={entry.item} className="rec-neighbor" aria-pressed={evidence.item === entry.item} onClick={() => setInspected(entry.item)}><span>I{entry.item} · {entry.pairs.length} co-rater{entry.pairs.length === 1 ? '' : 's'}</span><span className="rec-weight-track"><i style={{
            left: `${50 + Math.min(0, entry.similarity) * 50}%`,
            width: `${Math.abs(entry.similarity) * 50}%`,
            background: entry.similarity >= 0 ? gold : rose
          }} /><b /></span><span>{number(entry.similarity)} {result.selected.some(selected => selected.item === entry.item) ? '· used' : '· omitted'}</span></button>)}</div>
    <Matrix caption={`Co-raters for target I${item} and neighbor I${evidence.item}; displayed values after the selected centering`} labels={['User', 'Target', 'Neighbor']} rows={evidence.pairs.map(pair => [`U${pair.user}`, pair.left, pair.right])} />
    <p>Dot numerator {number(evidence.numerator)}; norms {number(evidence.leftNorm)} and {number(evidence.rightNorm)}. Raw cosine {number(evidence.rawSimilarity)}; supported weight {number(evidence.similarity)} after overlap and shrinkage.</p>
    <p className="rec-result" aria-live="polite">Prediction for U0, I{item}: <strong>{number(result.prediction)}</strong>. {result.usedFallback ? `No usable neighbors: fall back to the user's observed mean ${number(result.fallback)}.` : `Weighted numerator ${number(result.numerator)} ÷ absolute-weight sum ${number(result.denominator)}${centered ? ` + baseline ${number(result.fallback)}` : ''}.`}</p>
    <p className="rec-caption">At most three neighbors, largest absolute weight first; ties use item ID. Scores are unbounded estimates: a signed residual prediction can exceed the rating scale. Clipping, if chosen for an application, changes the evaluated predictor.</p>
  </Investigation>;
}
export function FactorUpdateLab() {
  const [rate, setRate] = useState(.05);
  const [penalty, setPenalty] = useState(.1);
  const [steps, setSteps] = useState(1);
  const result = factorUpdateTrace({
    rate,
    penalty,
    steps
  });
  const update = result.trace.at(-1) ?? result.final;
  const names = ['p₁', 'p₂', 'q₁', 'q₂', 'user bias', 'item bias'];
  const flatten = state => [...state.userFactors, ...state.itemFactors, state.userBias, state.itemBias];
  const before = flatten(update.before);
  const gradient = flatten(update.gradients);
  const after = flatten(update.after);
  const objectives = [result.trace[0]?.loss ?? result.final.loss, ...result.trace.map(entry => entry.nextLoss)];
  const maximum = Math.max(...objectives, .1);
  return <Investigation title="Move the factors using one shared old state" prompt="Increase the step size. Does moving opposite the gradient guarantee that a finite step lowers the objective?" reset={() => {
    setRate(.05);
    setPenalty(.1);
    setSteps(1);
  }}>
    <div className="rec-controls"><Range label="Learning rate η" value={rate} change={setRate} min={.01} max={1} step={.01} /><Range label="Local penalty λ" value={penalty} change={setPenalty} max={1} step={.1} /><Range label="Requested updates" value={steps} change={setSteps} min={1} max={8} step={1} /></div>
    <div className="rec-equation-flow"><span>rating 5</span><span>old score {number(update.prediction)}</span><strong>error {number(update.error)}</strong></div>
    <div className="rec-table-scroll"><table className="rec-table"><caption>Last completed update: new = old − η × gradient</caption><thead><tr><th>Parameter</th><th>Old</th><th>Gradient</th><th>New</th></tr></thead><tbody>{names.map((name, index) => <tr key={name}><th>{name}</th><td>{number(before[index])}</td><td>{number(gradient[index])}</td><td>{number(after[index])}</td></tr>)}</tbody></table></div>
    <svg viewBox="0 0 280 180" className="rec-plot" role="img" aria-label="Computed local objective after each completed update"><line x1="65" y1="142" x2="258" y2="142" className="rec-axis" /><line x1="65" y1="25" x2="65" y2="142" className="rec-axis" /><text x="65" y="16">Local objective (linear scale)</text><text x="58" y="32" textAnchor="end">{number(maximum, 2)}</text><text x="58" y="146" textAnchor="end">0</text><polyline fill="none" stroke={gold} strokeWidth="2" points={objectives.map((value, index) => `${65 + 190 * index / Math.max(1, objectives.length - 1)},${142 - 112 * value / maximum}`).join(' ')} />{objectives.map((value, index) => <circle key={index} cx={65 + 190 * index / Math.max(1, objectives.length - 1)} cy={142 - 112 * value / maximum} r="3" fill={gold} />)}<text x="65" y="163">0</text><text x="255" y="163" textAnchor="end">{result.completedSteps} updates</text></svg>
    <p aria-live="polite">Completed {result.completedSteps} updates. Local objective {number(objectives[0])} → {number(objectives.at(-1))}. {result.stopped && <strong>Stopped before a parameter would exceed this demonstration's ±10,000 arithmetic range; this is a divergent-step diagnosis, not convergence.</strong>}</p>
    <p className="rec-caption">This is one repeated rating, with mean 3 and two factors. It reveals a gradient mechanism; the full training program cycles through all observed records. All six gradients use the same old values.</p>
  </Investigation>;
}
export function FactorRotationLab() {
  const [angle, setAngle] = useState(0);
  const result = rotatedFactors(angle);
  const vectors = [result.user, ...result.items];
  const colors = [gold, blue, mint, rose];
  const labels = ['user', 'I0', 'I1', 'I2'];
  return <Investigation title="Rotate the coordinates without changing recommendations" prompt="Rotate every user and item by the same angle. Which changes: coordinates, dot scores, or their order?" reset={() => setAngle(0)}>
    <Range label="Shared rotation in degrees" value={angle} change={setAngle} min={-180} max={180} step={15} />
    <svg className="rec-plot rec-rotation" viewBox="0 0 280 280" role="img" aria-label="Toy user and item vectors under one shared orthogonal rotation"><line x1="25" y1="140" x2="255" y2="140" className="rec-axis" /><line x1="140" y1="25" x2="140" y2="255" className="rec-axis" /><text x="253" y="274" textAnchor="end">factor 1</text><text x="148" y="27">factor 2</text>{vectors.map((vector, index) => <g key={index}><line x1="140" y1="140" x2={140 + 75 * vector[0]} y2={140 - 75 * vector[1]} stroke={colors[index]} strokeWidth="2" /><circle cx={140 + 75 * vector[0]} cy={140 - 75 * vector[1]} r={index === 0 ? 5 : 3} fill={colors[index]} /><text x={140 + 75 * vector[0] + (vector[0] < -0.1 ? -10 : 10)} y={140 - 75 * vector[1] + (vector[1] < -0.1 ? 18 : -10)} fill={colors[index]} textAnchor={vector[0] < -0.1 ? 'end' : 'start'}>{labels[index]}</text></g>)}</svg>
    <div className="rec-equation-flow">{result.scores.map((score, item) => <span key={item}>I{item} score <strong>{number(score)}</strong></span>)}</div>
    <p className="rec-caption">Declared toy coordinates, not fitted genres. Rotation preserves dot products and L2 penalties. An arbitrary rescaling can preserve products if the other side is inversely rescaled, but generally changes the L2 penalty.</p>
  </Investigation>;
}
export function ImplicitConfidenceLab() {
  const [alpha, setAlpha] = useState(2);
  const [penalty, setPenalty] = useState(1);
  const [includeMissing, setIncludeMissing] = useState(true);
  const result = implicitFactorBlock({
    alpha,
    penalty,
    includeMissing
  });
  const all = result.contours.flatMap(contour => contour.points).concat([[0, 0], result.user]);
  const limits = [0, 1].map(axis => [Math.min(...all.map(point => point[axis])) - .08, Math.max(...all.map(point => point[axis])) + .08]);
  const x = value => 47 + 210 * (value - limits[0][0]) / (limits[0][1] - limits[0][0]);
  const y = value => 230 - 190 * (value - limits[1][0]) / (limits[1][1] - limits[1][0]);
  return <Investigation title="See the all-pair confidence bowl" prompt="Remove the missing-item term. Does the target vector change, or does the penalty for one score disappear?" reset={() => {
    setAlpha(2);
    setPenalty(1);
    setIncludeMissing(true);
  }}>
    <div className="rec-controls"><Range label="Confidence multiplier α" value={alpha} change={setAlpha} max={12} step={1} /><Range label="Once-per-user penalty λ" value={penalty} change={setPenalty} min={.1} max={3} step={.1} /><label><input type="checkbox" checked={includeMissing} onChange={event => setIncludeMissing(event.target.checked)} /> Include missing-item target 0 with weight 1</label></div>
    <div className="rec-two"><div><Matrix caption="Fixed item factors; counts [2, 0, 1]" labels={['q₁', 'q₂', 'z', 'weight']} rows={result.items.map((item, index) => [...item, result.targets[index], result.confidence[index]])} /><Matrix caption="Normal system A p = b" labels={['A column 1', 'A column 2', 'b']} rows={result.normal.map((row, index) => [...row, result.right[index]])} /></div><svg className="rec-plot" viewBox="0 0 280 296" role="img" aria-label="Exact quadratic contours at minimum plus .1, .5 and 1; axes are user factors"><text x="47" y="17">Objective above its minimum</text><line x1="47" y1="230" x2="257" y2="230" className="rec-axis" /><line x1="47" y1="40" x2="47" y2="230" className="rec-axis" />{[0, .5, 1].map(fraction => <g key={fraction}><text x={47 + 210 * fraction} y="252" textAnchor={fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle'}>{number(limits[0][0] + fraction * (limits[0][1] - limits[0][0]), 2)}</text><text x="40" y={234 - 190 * fraction} textAnchor="end">{number(limits[1][0] + fraction * (limits[1][1] - limits[1][0]), 2)}</text></g>)}{result.contours.map((contour, index) => <polyline key={contour.excess} points={contour.points.map(point => `${x(point[0])},${y(point[1])}`).join(' ')} stroke={[gold, blue, mint][index]} strokeWidth="2" fill="none" />)}<circle cx={x(result.user[0])} cy={y(result.user[1])} r="4" fill="white" /><text x="150" y="280" textAnchor="middle">user factor p₁</text><text x="47" y="33">p₂</text></svg></div>
    <p aria-live="polite">Minimum at p = ({number(result.user[0])}, {number(result.user[1])}); objective {number(result.objective)}. Item scores: {result.scores.map(value => number(value)).join(', ')}.</p>
    <p className="rec-caption">Gold, blue and mint curves have exact quadratic excess .1, .5 and 1; each is sampled at 65 calculated points for drawing. Axes refit the shown range when parameters change; these are objective contours, not data or uncertainty regions. White dot solves the normal system.</p>
  </Investigation>;
}
export function PairwiseRankingLab() {
  const [negativeFirst, setNegativeFirst] = useState(1);
  const [rate, setRate] = useState(.1);
  const result = bprPairStep({
    negative: [negativeFirst, 0],
    rate
  });
  const rows = [['Before', result.positiveScore, result.negativeScore], ['After', result.after.user.reduce((sum, value, index) => sum + value * result.after.positive[index], 0), result.after.user.reduce((sum, value, index) => sum + value * result.after.negative[index], 0)]];
  const lower = Math.min(0, ...rows.flatMap(row => row.slice(1))) - .1;
  const upper = Math.max(2, ...rows.flatMap(row => row.slice(1))) + .1;
  const position = value => 100 * (value - lower) / (upper - lower);
  return <Investigation title="Train a preference gap rather than a star rating" prompt="Give the sampled unobserved item a higher score. How does the pair's learning signal respond?" reset={() => {
    setNegativeFirst(1);
    setRate(.1);
  }}>
    <div className="rec-controls"><Range label="Sampled item's first factor" value={negativeFirst} change={setNegativeFirst} max={2} step={.1} /><Range label="Pair step size" value={rate} change={setRate} max={.8} step={.05} /></div>
    <div className="rec-pair-key"><span style={{
        color: gold
      }}>● interacted item i</span><span style={{
        color: blue
      }}>◆ sampled unobserved item j</span></div>{rows.map(([label, positive, negative]) => <div className="rec-pair-row" key={label}><strong>{label}</strong><div className="rec-score-line"><i className="rec-positive-point" style={{
          left: `${position(positive)}%`
        }} /><i className="rec-negative-point" style={{
          left: `${position(negative)}%`
        }} /></div><span>i: {number(positive)} · j: {number(negative)}</span></div>)}
    <p aria-live="polite">Gap i − j: <strong>{number(result.gap)} → {number(result.nextGap)}</strong>. Logistic signal σ(−gap) = {number(result.gradientMagnitude)}. Regularized local loss {number(result.loss)} → {number(result.nextLoss)}.</p>
    <p className="rec-caption">Shared score axis; all three factor vectors update from their old values, with λ=.1. The sampled comparison is a training assumption. Repeated controls do not pretend this one pair represents a whole ranking dataset.</p>
  </Investigation>;
}
export function RecommendationSlateLab() {
  const [order, setOrder] = useState([2, 0, 4, 1, 3]);
  const [cutoff, setCutoff] = useState(3);
  const [gradeMode, setGradeMode] = useState('binary');
  const [missingCandidate, setMissingCandidate] = useState(false);
  const grades = gradeMode === 'binary' ? [1, 1, 0, 0, 0] : gradeMode === 'graded' ? [2, 1, 0, 3, 0] : [0, 0, 0, 0, 0];
  const candidates = order.filter(item => !missingCandidate || item !== 1);
  const result = evaluateRecommendationList({
    order: candidates,
    grades,
    cutoff
  });
  const move = (item, direction) => {
    const next = [...order];
    const position = next.indexOf(item);
    const otherItem = candidates[candidates.indexOf(item) + direction];
    const destination = next.indexOf(otherItem);
    [next[position], next[destination]] = [next[destination], next[position]];
    setOrder(next);
  };
  return <Investigation title="Move the slate and watch the ranking evidence" prompt="Move the first relevant item up one slot, then hide item 1 from candidate generation. Which improvements become impossible?" reset={() => {
    setOrder([2, 0, 4, 1, 3]);
    setCutoff(3);
    setGradeMode('binary');
    setMissingCandidate(false);
  }}>
    <div className="rec-controls"><Range label="Visible slots K" value={cutoff} change={setCutoff} min={1} max={5} step={1} /><label>Evaluation labels <select aria-label="Evaluation labels" value={gradeMode} onChange={event => setGradeMode(event.target.value)}><option value="binary">Items 0 and 1 relevant</option><option value="graded">Grades [2, 1, 0, 3, 0]</option><option value="none">No labeled relevant items</option></select></label><label><input type="checkbox" checked={missingCandidate} onChange={event => setMissingCandidate(event.target.checked)} /> Candidate generator misses item 1</label></div>
    <ol className="rec-slate">{candidates.map((item, index) => <li key={item} className={index < cutoff ? 'rec-visible-slot' : 'rec-below-cutoff'}><span className="rec-rank">{index + 1}</span><span className="rec-item-title">Item {item}<small>grade {grades[item]} · {index < cutoff ? 'visible' : 'below cutoff'}</small></span><div><button aria-label={`Move item ${item} up`} disabled={index === 0} onClick={() => move(item, -1)}>↑</button><button aria-label={`Move item ${item} down`} disabled={index === candidates.length - 1} onClick={() => move(item, 1)}>↓</button></div></li>)}</ol>
    <dl className="rec-metrics">{[['Precision@K', result.precision], ['Recall@K', result.recall], ['NDCG@K', result.ndcg], ['RR@K', result.reciprocalRank], ['AP@K', result.averagePrecision], ['Fill rate', result.fillRate], ['Best possible candidate Recall@K', result.bestCandidateRecall]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{number(value)}</dd></div>)}</dl>
    <Matrix caption="Visible DCG terms: (2^grade − 1) / log₂(rank + 1)" labels={['Item', 'Gain', 'Discount', 'Term']} rows={result.contributions.map(entry => [`I${entry.item}`, entry.gain, entry.discount, entry.dcg])} />
    <p aria-live="polite">DCG {number(result.dcg)} ÷ ideal DCG {number(result.ideal)}. Eligible relevant items: {result.relevantCount}; filled slots: {result.contributions.length}/{cutoff}. {result.relevantCount === 0 ? 'Recall, AP, RR and NDCG are undefined for this cohort under our declared convention; do not silently report a perfect score.' : 'The ideal and relevant denominator still include item 1 even when retrieval misses it.'}</p>
  </Investigation>;
}
export function LoggedPolicyLab() {
  const [loggingA, setLoggingA] = useState(.8);
  const [targetA, setTargetA] = useState(.5);
  const [requests, setRequests] = useState(100);
  const result = exposurePolicy({
    loggingA,
    targetA,
    requests
  });
  return <Investigation title="Separate exposure from a policy's expected reward" prompt="Show A almost every time. Does the raw click average approach A's quality or the target policy's value? What happens when B is never shown?" reset={() => {
    setLoggingA(.8);
    setTargetA(.5);
    setRequests(100);
  }}>
    <div className="rec-controls"><Range label="Logging probability of A" value={loggingA} change={setLoggingA} /><Range label="Target probability of A" value={targetA} change={setTargetA} /><Range label="Independent requests" value={requests} change={setRequests} min={20} max={500} step={20} /></div>
    <div className="rec-policy-root">One randomized impression</div><div className="rec-policy-branches">{['A', 'B'].map((name, item) => <div key={name}><h4>{name} shown<br /><small>probability {number(result.logging[item])}</small></h4><div className="rec-policy-leaves">{[0, 1].map(reward => {
            const leaf = result.leaves.find(entry => entry.item === item && entry.reward === reward);
            return <div key={reward}><strong>{reward ? 'Click' : 'No click'}</strong><span>joint mass {number(leaf.probability)}</span><span>weighted reward {number(leaf.weightedReward)}</span></div>;
          })}</div></div>)}</div>
    <dl className="rec-metrics"><div><dt>Expected logged reward</dt><dd>{number(result.expectedLogged)}</dd></div><div><dt>Declared toy target value</dt><dd>{number(result.knownToyTargetValue)}</dd></div><div><dt>IPS expectation</dt><dd>{number(result.expectation)}</dd></div><div><dt>Standard error of mean IPS</dt><dd>{number(result.meanStandardError)}</dd></div></dl>
    <p aria-live="polite">{result.supported ? `Support holds. With the declared independent request model, one-request IPS variance is ${number(result.variance)}. A finite sample fluctuates; this expected-value calculation is not a simulated dataset.` : 'Support fails: the target sometimes selects an action the logger never selects. Its reward cannot be recovered from those logs. The toy target value is known here only because we declared both reward probabilities.'}</p>
    <p className="rec-caption">A clicks with probability .4, B with .8 in this fixed synthetic environment. Logging is randomized independently of potential rewards, propensities are known, and the displayed target is fixed. This is an optional finite-policy example, not a causal claim from arbitrary click logs.</p>
  </Investigation>;
}
