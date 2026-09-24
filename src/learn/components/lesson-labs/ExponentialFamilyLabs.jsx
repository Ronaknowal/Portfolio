import { useId, useMemo, useState } from 'react';
import { betaCoordinateModel, finiteExponentialFamily, finiteMomentFit, formatFamilyNumber, SUFFICIENCY_REFERENCE, summaryComparison } from '../../data/exponential-family-models.js';
import './exponential-family-labs.css';
const GOLD = '#e9c077';
const BLUE = '#9dc7dc';
const GREEN = '#adc68e';
const number = formatFamilyNumber;
function Facts({
  entries
}) {
  return <dl className="family-facts">{entries.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function BinaryString({
  values
}) {
  return <span className="family-binary" aria-label={values.join(' ')}>{values.map((value, index) => <span key={index} className={value ? 'is-success' : ''} aria-hidden="true">{value}</span>)}</span>;
}
export function SufficiencyFiberFigure() {
  const strings = ['1100', '1010', '1001', '0110', '0101', '0011'];
  return <figure className="family-figure family-fiber" aria-label="Six datasets in the two-success group">
    <div className="family-fiber-strings">{strings.map(value => <div key={value}><BinaryString values={value.split('').map(Number)} /><span>conditional mass 1/6</span></div>)}</div>
    <div className="family-fiber-result"><span aria-hidden="true">↓</span><strong>S = 2, n = 4</strong><span>one summary group</span></div>
    <figcaption>Exactly six ordered datasets have two successes. Under four independent trials with the same 0 &lt; p &lt; 1, their conditional probabilities given S=2 are equal. The picture groups possibilities; it does not say each dataset has unconditional probability 1/6.</figcaption>
  </figure>;
}
export function SufficiencyModelLab() {
  const id = useId();
  const [observations, setObservations] = useState([...SUFFICIENCY_REFERENCE]);
  const [grouped, setGrouped] = useState(false);
  const [first, setFirst] = useState(80);
  const [second, setSecond] = useState(30);
  const state = summaryComparison(observations, first / 100, second / 100, grouped);
  const reset = () => {
    setObservations([...SUFFICIENCY_REFERENCE]);
    setGrouped(false);
    setFirst(80);
    setSecond(30);
  };
  return <section className="family-lab" aria-labelledby={`${id}-title`} data-family-lab="sufficiency">
    <p className="lesson-eyebrow">INVESTIGATE · WHAT THE SUMMARY FORGETS</p>
    <h3 id={`${id}-title`}>Keep six successes. Change where they occurred.</h3>
    <p>Inspect: if one success moves from group A to group B, will the relative likelihood change? Start with the common-p model, then select two group probabilities.</p>
    <div className="family-controls">
      <label>Probability model<select aria-label="Probability model" value={grouped ? 'groups' : 'common'} onChange={event => setGrouped(event.target.value === 'groups')}><option value="common">One common probability</option><option value="groups">Two fixed groups</option></select></label>
      <label>{grouped ? 'Group A probability' : 'Common probability'}: {first / 100}<input aria-label="First probability" type="range" min="5" max="95" step="5" value={first} onChange={event => setFirst(Number(event.target.value))} /></label>
      <label>Group B probability: {second / 100}<input aria-label="Second probability" type="range" min="5" max="95" step="5" value={second} disabled={!grouped} onChange={event => setSecond(Number(event.target.value))} /></label>
    </div>
    <div className="family-data-rows">
      <div><strong>Reference data</strong><BinaryString values={SUFFICIENCY_REFERENCE} /><span>A: 4 successes · B: 2</span></div>
      <div><strong>Your data · select a bit to toggle</strong><div className="family-edit-bits">{[0, 1].map(group => <fieldset key={group}><legend>Group {group === 0 ? 'A' : 'B'}</legend>{observations.slice(group * 4, group * 4 + 4).map((value, offset) => <button type="button" key={offset} aria-label={`Observation ${group * 4 + offset + 1}`} aria-pressed={Boolean(value)} onClick={() => setObservations(values => values.map((old, index) => index === group * 4 + offset ? 1 - old : old))}>{value}</button>)}</fieldset>)}</div></div>
    </div>
    <div className="family-actions"><button type="button" onClick={() => setObservations([1, 1, 1, 0, 1, 1, 1, 0])}>Move one success A → B</button><button type="button" onClick={reset}>Reset comparison</button></div>
    <Facts entries={[["Your summary (A, B)", `(${state.groupCounts.join(', ')})`], ['Your total / reference total', `${state.total} / 6`], ['Likelihood ratio: yours / reference', number(state.ratio)], ['Log likelihood ratio', number(state.logRatio)]]} />
    <p className="family-feedback" aria-live="polite">{state.total !== 6 ? 'The totals now differ, so this comparison does not test whether one total is sufficient.' : grouped ? 'With two group probabilities, a shared total can conceal different allocations. Change either probability and inspect the ratio; equality at one parameter value cannot prove sufficiency.' : 'The equal totals give ratio 1 for every common p here. Changing order alone adds no information about that one parameter.'}</p>
    <p className="lesson-note">Calculated ordered-data likelihoods under eight independent Bernoulli variables. Groups and their four observation slots are fixed. The ratio is not a posterior probability; equality of two probabilities at one parameter value is not a sufficiency proof.</p>
    <p><strong>Transfer:</strong> swap two observations within group A. Explain why its group count survives even in the richer model.</p>
  </section>;
}
export function FamilyNormalizerLab() {
  const id = useId();
  const [eta, setEta] = useState(0);
  const state = finiteExponentialFamily(eta, 0, [1, 2, 1]);
  return <section className="family-lab" aria-labelledby={`${id}-title`} data-family-lab="normalizer">
    <p className="lesson-eyebrow">INVESTIGATE · FROM WEIGHTS TO PROBABILITIES</p>
    <h3 id={`${id}-title`}>One denominator changes every outcome</h3>
    <p>At η=0, predict why the middle outcome is twice as likely. Increase η: which weights grow, and what happens to the expected outcome?</p>
    <label className="family-slider">Natural parameter η: {eta}<input aria-label="Normalizer natural parameter" type="range" min="-5" max="5" step="0.1" value={eta} onChange={event => setEta(Number(event.target.value))} /></label>
    <div className="family-weight-ledger" role="table" aria-label="Normalization calculation">
      <div role="row" className="family-weight-head"><span role="columnheader">x / base h</span><span role="columnheader">log weight</span><span role="columnheader">shifted weight</span><span role="columnheader">probability</span></div>
      {state.outcomes.map((outcome, index) => <div role="row" key={outcome}><span role="cell"><strong>{outcome}</strong> / {state.base[index]}</span><span role="cell">{number(state.logWeights[index])}</span><span role="cell">{number(state.relativeWeights[index])}</span><span role="cell"><span className="family-mass-bar" aria-hidden="true"><i style={{
              width: `${state.probabilities[index] * 100}%`
            }} /></span>{number(state.probabilities[index])}</span></div>)}
    </div>
    <Facts entries={[["Subtracted shift m", number(state.shift)], ['Sum of shifted weights', number(state.relativeTotal)], ['A = m + log(sum)', number(state.logPartition)], ['A′ = expected X', number(state.mean[0])], ['A″ = variance of X', number(state.covariance[0][0])]]} />
    <p className="family-feedback">All bars use the same 0–1 probability scale. Divide each shifted weight by their sum. Subtracting the same m from all log weights changes no probability; add m back when computing A.</p>
    <button type="button" onClick={() => setEta(0)}>Reset weights</button>
    <p className="lesson-note">Exact finite model evaluated numerically: x∈{'{−1,0,1}'}, h=(1,2,1), T(x)=x. These bars are calculated masses, not sample frequencies. Variance shrinks as mass concentrates, but finite η keeps all three outcomes possible.</p>
    <p><strong>Transfer:</strong> compare η=2 and η=−2. Explain the opposite means and equal variance from the symmetric base weights.</p>
  </section>;
}
function MomentTriangle({
  model,
  target
}) {
  const x = value => 160 + 115 * value;
  const y = value => 210 - 160 * value;
  return <svg className="family-svg" viewBox="0 0 320 285" role="img" aria-label="Possible moment triangle and observed versus modeled means">
    <title>Moment geometry for X in −1, 0, 1</title><desc>The vertices are (−1,1), (0,0), (1,1). All probability mixtures lie in their triangle. Blue marks the modeled mean; gold outlines the observed mean, if data exist. A finite model assigns positive mass to all vertices, so its mean is interior.</desc>
    <polygon points="45,50 160,210 275,50" fill="#17211c" stroke="#586b5e" />
    <line x1="30" y1="210" x2="290" y2="210" className="family-axis" /><line x1="160" y1="225" x2="160" y2="25" className="family-axis" />
    <text x="160" y="15" textAnchor="middle">E[X²]</text><text x="160" y="252" textAnchor="middle">E[X]</text>
    <text x="32" y="41">(−1,1)</text><text x="244" y="41">(1,1)</text><text x="164" y="229">(0,0)</text>
    {[-1, 1].map(value => <text key={value} x={x(value)} y="229" textAnchor="middle">{value}</text>)}
    {target && <line x1={x(target[0])} y1={y(target[1])} x2={x(model[0])} y2={y(model[1])} stroke={GOLD} strokeDasharray="3 3" />}
    <circle cx={x(model[0])} cy={y(model[1])} r="5" fill={BLUE} />
    {target && <circle cx={x(target[0])} cy={y(target[1])} r="8" fill="none" stroke={GOLD} strokeWidth="2" />}
    <text x="12" y="278" fill={BLUE}>● modeled</text><text x="155" y="278" fill={GOLD}>○ observed</text>
  </svg>;
}
export function FamilyMomentLab() {
  const id = useId();
  const [counts, setCounts] = useState([2, 5, 3]);
  const [parameters, setParameters] = useState([0, 0]);
  const model = finiteExponentialFamily(...parameters);
  const target = finiteMomentFit(counts);
  const outsideSliders = target.parameters?.some(value => Math.abs(value) > 5);
  const reset = () => {
    setCounts([2, 5, 3]);
    setParameters([0, 0]);
  };
  return <section className="family-lab" aria-labelledby={`${id}-title`} data-family-lab="moments">
    <p className="lesson-eyebrow">INVESTIGATE · WHERE A FIT CAN LIVE</p>
    <h3 id={`${id}-title`}>Fit two moments inside a triangle</h3>
    <p>Every distribution is a weighted mixture of three vertices T(x)=(x,x²). Inspect where the observed mean moves if the zero outcome disappears. The two-parameter family here uses h=1 for all three outcomes.</p>
    <div className="family-controls">{[-1, 0, 1].map((outcome, index) => <label key={outcome}>Count at x={outcome}<input type="number" aria-label={`Count at ${outcome}`} min="0" max="20" step="1" value={counts[index]} onChange={event => {
          const value = Number(event.target.value);
          if (Number.isInteger(value) && value >= 0 && value <= 20) setCounts(previous => previous.map((count, position) => position === index ? value : count));
        }} /></label>)}</div>
    <div className="family-controls">{['Linear η₁', 'Square η₂'].map((label, index) => <label key={label}>{label}: {number(parameters[index])}<input type="range" aria-label={label} min="-5" max="5" step="any" value={parameters[index]} onChange={event => setParameters(previous => previous.map((value, position) => position === index ? Number(event.target.value) : value))} /></label>)}</div>
    <div className="family-moment-layout"><MomentTriangle model={model.mean} target={target.mean} /><div>
      <Facts entries={[["Modeled (E[X], E[X²])", model.mean.map(number).join(', ')], ['Observed (mean X, mean X²)', target.mean ? target.mean.map(number).join(', ') : 'No observations'], ['Model masses at −1, 0, 1', model.probabilities.map(number).join(', ')], ['Observed proportions', target.probabilities ? target.probabilities.map(number).join(', ') : 'Undefined']]} />
    </div></div>
    <div className="family-actions"><button type="button" disabled={target.status !== 'finite' || outsideSliders} onClick={() => setParameters([...target.parameters])}>Fit observed moments</button><button type="button" onClick={() => setCounts([2, 0, 3])}>Remove zero category</button><button type="button" onClick={reset}>Reset moments</button></div>
    <p className="family-feedback" aria-live="polite">{target.status === 'empty' ? 'No data: the log likelihood is constant. There is no observed mean to match.' : target.status === 'boundary' ? 'Boundary target: at least one observed proportion is zero. This finite-support family can approach that target, but no finite natural parameter assigns zero mass to a category. The fit button stays disabled.' : outsideSliders ? 'The finite fitted parameters exceed this displayed slider range. They are not clipped into a false fit.' : 'All counts are positive: the unique finite fit exists in this explicitly solved family. Fit the moments, then compare all three masses, not only one mean.'}</p>
    <p className="lesson-note">Triangle coordinates and circles are calculated expectations, not projected data points or confidence regions. Sliders display η∈[−5,5]; fitted values retain full internal precision. Gold and blue coincide only when both moments match.</p>
    <p><strong>Transfer:</strong> use counts (1,8,1), then (4,2,4). Both have mean X=0; explain why their second moments and η₂ differ.</p>
  </section>;
}
export function GaussianSummaryFigure() {
  return <figure className="family-figure">
    <svg className="family-svg family-wide-svg" viewBox="0 0 320 245" role="img" aria-label="Same sample mean, different spread">
      <title>Two Gaussian datasets need different spread summaries</title><desc>Dataset A is 1,2,3 and dataset B is 0,2,4. Both have sample mean2. Their sums of squared deviations are2 and8.</desc>
      {[[1, 2, 3], [0, 2, 4]].map((values, row) => <g key={row}>
        <text x="15" y={22 + row * 103}>{row === 0 ? 'A: 1, 2, 3' : 'B: 0, 2, 4'}</text>
        <line x1="35" y1={54 + row * 103} x2="285" y2={54 + row * 103} className="family-axis" />
        <line x1="160" y1={35 + row * 103} x2="160" y2={60 + row * 103} stroke={GREEN} strokeDasharray="3 4" />
        {values.map(value => <circle key={value} cx={35 + 62.5 * value} cy={54 + row * 103} r="5" fill={row === 0 ? BLUE : GOLD} />)}
        {[0, 1, 2, 3, 4].map(value => <text key={value} x={35 + 62.5 * value} y={75 + row * 103} textAnchor="middle">{value}</text>)}
        <text x="15" y={99 + row * 103}>mean 2 · squared deviations {row === 0 ? '2' : '8'}</text>
      </g>)}
      <text x="160" y="227" textAnchor="middle">Shared measurement units</text>
    </svg>
    <figcaption>A shared sum of 6 preserves the mean information when variance is known. When variance is also unknown, the different spreads distinguish parameter pairs; retain n, sum and squared sum, or an equivalent centered summary.</figcaption>
  </figure>;
}
function DensityPlot({
  points,
  interval,
  domain,
  xLabel,
  title
}) {
  const maximum = Math.max(...points.map(([, density]) => density));
  const x = value => 40 + 264 * (value - domain[0]) / (domain[1] - domain[0]);
  const y = value => 175 - 140 * value / maximum;
  const selected = points.filter(([value]) => value >= interval[0] && value <= interval[1]);
  return <svg className="family-svg" viewBox="0 0 320 225" role="img" aria-label={title}>
    <title>{title}</title><desc>Analytic Beta density in the labelled coordinate, calculated at displayed points. Gold shows the interval corresponding to p from one quarter to three quarters. Vertical scales differ because density units differ.</desc>
    <text x="40" y="17">density per {xLabel} unit</text>
    <line x1="40" y1="175" x2="305" y2="175" className="family-axis" /><line x1="40" y1="30" x2="40" y2="175" className="family-axis" />
    <text x="35" y="179" textAnchor="end">0</text><text x="35" y="39" textAnchor="end">{maximum.toPrecision(2)}</text>
    <path d={`M${x(selected[0][0])},175 ${selected.map(([value, density]) => `L${x(value)},${y(density)}`).join(' ')} L${x(selected.at(-1)[0])},175 Z`} fill="#e9c07728" />
    <path d={points.map(([value, density], index) => `${index ? 'L' : 'M'}${x(value)},${y(density)}`).join(' ')} fill="none" stroke={BLUE} strokeWidth="2" />
    {interval.map(value => <line key={value} x1={x(value)} y1="30" x2={x(value)} y2="175" stroke={GOLD} strokeDasharray="3 4" />)}
    {[domain[0], (domain[0] + domain[1]) / 2, domain[1]].map((value, index) => <text key={value} x={x(value)} y="195" textAnchor={index === 0 ? 'start' : index === 2 ? 'end' : 'middle'}>{value}</text>)}
    <text x="172" y="218" textAnchor="middle">{xLabel}</text>
  </svg>;
}
export function PriorCoordinateLab() {
  const id = useId();
  const [alpha, setAlpha] = useState(2);
  const [beta, setBeta] = useState(2);
  const model = useMemo(() => betaCoordinateModel(alpha, beta), [alpha, beta]);
  return <section className="family-lab" aria-labelledby={`${id}-title`} data-family-lab="coordinates">
    <p className="lesson-eyebrow">INVESTIGATE · SAME PRIOR, DIFFERENT DENSITY</p>
    <h3 id={`${id}-title`}>Probability mass survives the change of coordinates</h3>
    <p>Inspect: should a density have the same height after replacing p with log-odds η? Compare the corresponding gold intervals. They contain the same probability, despite different widths and heights.</p>
    <div className="family-controls"><label>Beta α: {alpha}<input type="range" aria-label="Prior alpha" min="1" max="8" step="1" value={alpha} onChange={event => setAlpha(Number(event.target.value))} /></label><label>Beta β: {beta}<input type="range" aria-label="Prior beta" min="1" max="8" step="1" value={beta} onChange={event => setBeta(Number(event.target.value))} /></label></div>
    <div className="family-density-pair"><DensityPlot points={model.probabilityCurve} interval={[0.25, 0.75]} domain={[0, 1]} xLabel="p" title="Beta density with respect to probability p" /><DensityPlot points={model.etaCurve} interval={model.interval} domain={[-6, 6]} xLabel="η" title="The same Beta prior density with respect to log odds eta" /></div>
    <Facts entries={[["p interval", '[0.25, 0.75]'], ['Corresponding η interval', '[−log 3, log 3]'], ['Probability in either interval', number(model.intervalMass)], ['Mass inside displayed η window', number(model.displayedEtaMass)], ['Density at p=0.5', number(model.probabilityDensity(0.5))], ['Density at η=0', number(model.etaDensity(0))]]} />
    <p className="family-feedback">At p=0.5, dp/dη=0.25. The η-density height is one quarter of the p-density height there. Both describe the same prior; treating them as the same function without the Jacobian would change that prior.</p>
    <button type="button" onClick={() => {
      setAlpha(2);
      setBeta(2);
    }}>Reset prior</button>
    <p className="lesson-note">Calculated Beta densities for integer α,β. Each vertical axis has its own labelled scale. The η plot stops at ±6 and omits the remaining tails; its curve is not renormalized. The interval probability uses a finite exact-form polynomial evaluated numerically, not the shaded pixel area.</p>
    <p><strong>Transfer:</strong> set α=β=1. The prior is flat in p; explain why it cannot also be a flat proper density on the whole real η line.</p>
  </section>;
}
export function CanonicalFeatureFigure() {
  return <figure className="family-figure family-feature-flow" aria-label="A fixed feature vector maps through natural parameter to mean probability">
    <div><strong>Known features</strong><span>z = (1, 2)</span><small>intercept + measured input</small></div><span aria-hidden="true">→</span><div><strong>Linear score</strong><span>η = zᵀβ = −1 + 2×0.5 = 0</span><small>β = (−1, 0.5)</small></div><span aria-hidden="true">→</span><div><strong>Bernoulli mean</strong><span>p = sigmoid(0) = 0.5</span><small>valid probability</small></div>
    <figcaption>Calculated one-row canonical Bernoulli GLM. The feature vector is known context; the unknown coefficients are shared across rows. A different feature value can give a different success probability.</figcaption>
  </figure>;
}
