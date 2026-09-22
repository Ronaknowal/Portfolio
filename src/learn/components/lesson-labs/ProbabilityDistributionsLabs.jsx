import { useId, useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { arrivalWindowState, bayesPopulationState, eventConditionState, mixedDelayState, pairedEvidenceState, probabilityNumber, urnCountState } from '../../data/probability-distributions-models.js';
import './probability-distributions-labs.css';
const number = probabilityNumber;
const percent = value => value === null ? 'undefined: zero evidence' : `${number(100 * value)}%`;
const colors = ['#e5b95f', '#ad8bca', '#70b9c7', '#719c84'];
function Field({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01
}) {
  const id = useId();
  return <label className="probability-field" htmlFor={id}><span>{label}: <strong>{number(value)}</strong></span>
    <input id={id} aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}
function Investigation({
  id,
  title,
  guidance,
  children,
  onReset
}) {
  const heading = useId();
  return <section data-live-exploration className="probability-lab lesson-lab" data-investigation={id} aria-labelledby={heading}>
    <p className="lesson-eyebrow">CALCULATED MODEL · EXPLORE THE ASSUMPTIONS</p><h3 id={heading}>{title}</h3>
    <p> {guidance}</p>{children}
    <button type="button" className="probability-reset" onClick={onReset}>Reset investigation</button>
  </section>;
}
function Readout({
  values
}) {
  return <dl className="probability-readout">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function MassStrip({
  masses,
  labels,
  caption
}) {
  return <figure className="probability-mass-strip"><figcaption>{caption}</figcaption>
    <div className="probability-strip-track" role="img" aria-label={masses.map((mass, index) => `${labels[index]}: ${percent(mass)}`).join('; ')}>
      {masses.map((mass, index) => <span key={index} style={{
        width: `${mass * 100}%`,
        background: colors[index % colors.length]
      }} />)}
    </div>
    <ul className="probability-legend">{labels.map((label, index) => <li key={label}><i aria-hidden="true" style={{
          background: colors[index % colors.length]
        }} />{label}: {percent(masses[index])}</li>)}</ul>
  </figure>;
}
export function EventConditionLab() {
  const [event, setEvent] = useState([2, 4, 6]);
  const [condition, setCondition] = useState([4, 5, 6]);
  const state = eventConditionState(event, condition);
  const toggle = (values, value, setValues) => setValues(values.includes(value) ? values.filter(item => item !== value) : [...values, value]);
  return <Investigation id="probability-events" title="Conditioning changes which outcomes remain" guidance="Keep A = even faces. If B changes from {4,5,6} to {5,6}, does P(A|B) increase or decrease?" onReset={() => {
    setEvent([2, 4, 6]);
    setCondition([4, 5, 6]);
  }}>
    <p>One fair die: each original face has probability 1/6. Toggle membership in A and B separately. An outcome can belong to both.</p>
    {[['Event A', event, setEvent], ['Given event B', condition, setCondition]].map(([label, values, setValues]) => <fieldset key={label} className="probability-ticket-picker"><legend>{label}</legend>
      {state.outcomes.map(outcome => <button type="button" key={outcome.value} aria-label={`${label}, face ${outcome.value}`} aria-pressed={values.includes(outcome.value)} onClick={() => toggle(values, outcome.value, setValues)}>{outcome.value}</button>)}
    </fieldset>)}
    <figure className="probability-restricted"><figcaption>Inside B: retain these tickets, then give them equal shares of 1</figcaption>
      <div className="probability-ticket-result">{state.outcomes.map(outcome => <span key={outcome.value} className={`${outcome.inCondition ? 'retained' : 'excluded'} ${outcome.inEvent && outcome.inCondition ? 'intersection' : ''}`}>
        <strong>{outcome.value}</strong><small>{!outcome.inCondition ? 'outside B' : outcome.inEvent ? 'also A' : 'not A'}</small>
      </span>)}</div>
    </figure>
    <Readout values={[[`P(A)`, number(state.eventProbability)], [`P(B)`, number(state.conditionProbability)], [`P(A ∩ B)`, number(state.intersectionProbability)], [`P(A ∪ B)`, number(state.unionProbability)], [`P(A|B)`, number(state.conditionalProbability)], ['Independent / disjoint', `${state.independent ? 'yes' : 'no'} / ${state.disjoint ? 'yes' : 'no'}`]]} />
    <p aria-live="polite">{condition.length ? `${event.filter(face => condition.includes(face)).length} of the ${condition.length} retained faces also belong to A. Dividing by the original six would calculate the joint probability instead.` : 'B has no outcomes. P(B)=0, so elementary conditional probability is undefined; there is no retained group to normalize.'}</p>
    <p><strong>Try a different relationship.</strong> Set A={(1, 2, 3)} and B={(2, 4)}. Their intersection has one of six faces: 1/6=(3/6)(2/6), so they are independent even though they overlap.</p>
  </Investigation>;
}
export function BayesPopulationLab() {
  const [prior, setPrior] = useState(0.01);
  const [sensitivity, setSensitivity] = useState(0.95);
  const [falsePositive, setFalsePositive] = useState(0.1);
  const [positive, setPositive] = useState(true);
  const state = bayesPopulationState(prior, sensitivity, falsePositive);
  const posterior = positive ? state.positivePosterior : state.negativePosterior;
  const labels = ['H and +', 'H and −', 'not H and +', 'not H and −'];
  return <Investigation id="probability-bayes" title="A strong detector can flag mostly ordinary cases" guidance="The detector flags 95% of H cases. Does that mean 95% of flagged cases are H when only 1% of the population is H?" onReset={() => {
    setPrior(0.01);
    setSensitivity(0.95);
    setFalsePositive(0.1);
    setPositive(true);
  }}>
    <div className="probability-controls"><Field label="Prior P(H)" value={prior} onChange={setPrior} /><Field label="Sensitivity P(+|H)" value={sensitivity} onChange={setSensitivity} /><Field label="False-positive rate P(+|not H)" value={falsePositive} onChange={setFalsePositive} /></div>
    <MassStrip masses={state.cells.map(cell => cell.mass)} labels={labels} caption="Whole population, split into four joint events — widths are actual probability" />
    <LessonTable caption="Expected counts in 100,000 synthetic cases; not measured observations" headers={['Actual class', 'Flagged +', 'Not flagged −', 'Total']} rows={[['H', number(state.expectedCounts[0]), number(state.expectedCounts[1]), number(100000 * prior)], ['not H', number(state.expectedCounts[2]), number(state.expectedCounts[3]), number(100000 * (1 - prior))], ['Both classes', number(state.positiveMass * 100000), number(state.negativeMass * 100000), '100,000']]} />
    <fieldset className="probability-choice"><legend>Restrict attention to the observed result</legend><button type="button" aria-pressed={positive} onClick={() => setPositive(true)}>Flagged +</button><button type="button" aria-pressed={!positive} onClick={() => setPositive(false)}>Not flagged −</button></fieldset>
    {posterior === null ? <p role="status">This result has zero total probability under the current prior and class-conditional model. The posterior is undefined.</p> : <MassStrip masses={[posterior, 1 - posterior]} labels={['H within selected result', 'not H within selected result']} caption="Zoom into that result and renormalize its mass to 1" />}
    <p aria-live="polite">P(H|{positive ? '+' : '−'}) = <strong>{percent(posterior)}</strong>. The population strip is not enlarged to make rare groups look common; use the counts and this conditional view to inspect them.</p>
    <p><strong>Transfer.</strong> Change only the prior to .2, then inspect both results. The detector's two class-conditional rates stay fixed; the composition of its flagged group changes.</p>
  </Investigation>;
}
export function ReusedEvidenceLab() {
  const [copyShare, setCopyShare] = useState(0);
  const [pattern, setPattern] = useState(0);
  const state = pairedEvidenceState(0.01, 0.95, 0.1, copyShare);
  const selected = state.patterns[pattern];
  return <Investigation id="probability-evidence" title="Two messages are not always two pieces of evidence" guidance="If the second flag is an exact copy of the first, should two positive messages change your belief more than one?" onReset={() => {
    setCopyShare(0);
    setPattern(0);
  }}>
    <p>Same prior .01 and detector rates .95/.10. With probability c, copy one detector result. Otherwise generate two fresh results independently <em>within each actual class</em>. Each individual message keeps its original accuracy.</p>
    <Field label="Copy share c" value={copyShare} onChange={setCopyShare} step={0.05} />
    <div className="probability-evidence-branches"><div><strong>c: one draw → copy</strong><span>first + → ++</span><span>first − → −−</span></div><div><strong>1−c: two fresh draws</strong><span>+, then + or −</span><span>−, then + or −</span></div></div>
    <MassStrip masses={state.patterns.map(item => item.givenHypothesis)} labels={state.patterns.map(item => item.label)} caption="Pair distribution within H" />
    <MassStrip masses={state.patterns.map(item => item.givenAlternative)} labels={state.patterns.map(item => item.label)} caption="Pair distribution within not H" />
    <fieldset className="probability-choice"><legend>Observed pair</legend>{state.patterns.map((item, index) => <button type="button" key={item.label} aria-pressed={pattern === index} onClick={() => setPattern(index)}>{item.label}</button>)}</fieldset>
    <Readout values={[[`P(${selected.label}|H)`, number(selected.givenHypothesis)], [`P(${selected.label}|not H)`, number(selected.givenAlternative)], ['Actual posterior P(H|pair)', percent(selected.posterior)], ['If you assumed two fresh results', percent(selected.independentPosterior)]]} />
    <p aria-live="polite">{selected.posterior === null ? 'This mixed-sign pair is impossible when every second result is a copy. A confident numerical answer would conceal a model contradiction.' : copyShare === 0 ? 'The fresh-result calculation is valid here, conditional on the class. Mixing H and not-H cases still makes the two messages dependent overall.' : copyShare === 1 ? 'The second message repeats the first. Equal-sign pairs provide exactly the same evidence as one result.' : 'Some apparent repetition is copied. Multiplying the two single-result likelihoods would use the wrong joint model.'}</p>
    <p><strong>Try the middle.</strong> At c=.5 and ++, compare the actual 14.538% posterior with the 47.6882% obtained by assuming fresh results.</p>
  </Investigation>;
}
function CountChart({
  masses,
  threshold,
  cumulative = false
}) {
  const id = useId();
  const x = value => 35 + 35 * value;
  const y = value => 165 - 130 * value;
  return <figure className="probability-chart"><svg viewBox="0 0 285 216" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
    <title id={`${id}-title`}>{cumulative ? 'Cumulative probability of the marked count' : 'Probability mass of each marked count'}</title>
    <desc id={`${id}-desc`}>Horizontal axis: marked count k, 0 to 6. Vertical probability axis: 0 to 1. {cumulative ? 'The staircase accumulates mass at each integer; a filled point gives the value at the jump.' : 'Gold bars are counts at or below the selected threshold. Bar height is P(X=k), not density.'}</desc>
    <line x1="35" x2="266" y1="165" y2="165" /><line x1="35" x2="35" y1="30" y2="165" />
    <text x="10" y="35">1</text><text x="10" y="169">0</text><text x="35" y="19">{cumulative ? 'F(k) = P(X ≤ k)' : 'P(X = k)'}</text>
    {Array.from({
        length: 7
      }, (_, count) => <text key={count} x={x(count)} y="187" textAnchor="middle">{count}</text>)}<text x="145" y="209" textAnchor="middle">Marked count k</text>
    {masses.map((item, index) => cumulative ? <g key={item.successes}>
      <line x1={x(item.successes)} x2={index === masses.length - 1 ? 266 : x(item.successes + 1) - 1} y1={y(item.cumulative)} y2={y(item.cumulative)} style={{
          stroke: colors[2],
          strokeWidth: 3
        }} />
      <line x1={x(item.successes)} x2={x(item.successes)} y1={y(index ? masses[index - 1].cumulative : 0)} y2={y(item.cumulative)} style={{
          stroke: colors[2],
          strokeDasharray: '2 3'
        }} />
      <circle cx={x(item.successes)} cy={y(item.cumulative)} r="3" fill={colors[2]} />
    </g> : <rect key={item.successes} x={x(item.successes) - 8} y={y(item.mass)} width="16" height={130 * item.mass} fill={item.successes <= threshold ? colors[0] : '#6e7c88'} />)}
    {cumulative && <line x1={x(threshold)} x2={x(threshold)} y1="30" y2="165" style={{
        stroke: colors[0],
        strokeDasharray: '4 4'
      }} />}
  </svg></figure>;
}
export function UrnCountLab() {
  const [marked, setMarked] = useState(2);
  const [draws, setDraws] = useState(3);
  const [replacement, setReplacement] = useState(false);
  const [threshold, setThreshold] = useState(1);
  const state = urnCountState(marked, draws, replacement, threshold);
  return <Investigation id="probability-counts" title="The sampling rule changes the distribution" guidance="Three draws from six objects, only two marked: can the marked count be 3? Does your answer change if each object is replaced?" onReset={() => {
    setMarked(2);
    setDraws(3);
    setReplacement(false);
    setThreshold(1);
  }}>
    <div className="probability-urn" aria-label="Six individually labeled objects">{Array.from({
        length: 6
      }, (_, index) => <span key={index} className={index < marked ? 'marked' : ''}><strong>{String.fromCharCode(65 + index)}</strong><small>{index < marked ? 'marked' : 'plain'}</small></span>)}</div>
    <div className="probability-controls"><Field label="Marked objects" value={marked} min={0} max={6} step={1} onChange={setMarked} /><Field label="Number of draws" value={draws} min={0} max={6} step={1} onChange={setDraws} /><Field label="Threshold k in X ≤ k" value={threshold} min={0} max={6} step={1} onChange={setThreshold} /></div>
    <fieldset className="probability-choice"><legend>After each draw</legend><button type="button" aria-pressed={!replacement} onClick={() => setReplacement(false)}>Keep it out</button><button type="button" aria-pressed={replacement} onClick={() => setReplacement(true)}>Replace and remix</button></fieldset>
    <div className="probability-paired-plots"><CountChart masses={state.masses} threshold={threshold} /><CountChart masses={state.masses} threshold={threshold} cumulative /></div>
    <Readout values={[[`P(X ≤ ${threshold})`, number(state.cumulativeMass)], ['Mean marked count', number(state.mean)], ['Variance of count', number(state.variance)], ['Model', replacement ? 'Binomial' : 'Hypergeometric']]} />
    <LessonTable caption="Exact model values rounded for display" headers={['Marked count', 'P(X=k)', 'P(X≤k)']} rows={state.masses.map(item => [item.successes, number(item.mass), number(item.cumulative)])} />
    <p aria-live="polite">{replacement ? 'Replacement keeps each draw at the same marked probability; independent remixing is an explicit model assumption.' : 'Without replacement, draws affect what remains. Drawing all six makes the marked count certain, even though the order remains random.'} The displayed bars sum to 1; zero-height counts are impossible, not merely rare.</p>
  </Investigation>;
}
function DelayPlots({
  state
}) {
  const id = useId();
  const x = value => 40 + 210 * value / state.width;
  const densityMaximum = Math.max(1 / (state.unit === 'seconds' ? 1 : 1000), state.density * 1.25);
  const densityY = value => 155 - 115 * value / densityMaximum;
  const cdfY = value => 155 - 115 * value;
  return <div className="probability-paired-plots">
    <figure className="probability-chart"><svg viewBox="0 0 280 225" role="img" aria-labelledby={`${id}-density-title ${id}-density-desc`}>
      <title id={`${id}-density-title`}>Continuous delay density and selected interval area</title><desc id={`${id}-density-desc`}>Density is constant at {state.density} per {state.unit === 'seconds' ? 'second' : 'millisecond'} on 0 to {state.width}. Shaded area is {state.continuousMass}. A point mass at zero is reported separately and has no finite density height. Axes rescale with units and parameters.</desc>
      <line x1="40" x2="255" y1="155" y2="155" /><line x1="40" x2="40" y1="30" y2="155" />
      <text x="40" y="19">Density / {state.unit === 'seconds' ? 'second' : 'millisecond'}</text>
      <rect x={x(state.left)} y={densityY(state.density)} width={x(state.right) - x(state.left)} height={155 - densityY(state.density)} fill="#e5b95f55" />
      <line x1="40" x2="250" y1={densityY(state.density)} y2={densityY(state.density)} style={{
          stroke: colors[0],
          strokeWidth: 3
        }} />
      <text x="45" y={Math.max(37, densityY(state.density) - 9)}>{number(state.density)}</text><text x="28" y="159" textAnchor="end">0</text>
      <text x="40" y="179">0</text><text x="250" y="179" textAnchor="end">{number(state.width)}</text><text x="145" y="201" textAnchor="middle">{state.unit}</text>
    </svg><figcaption>Area = density × interval width. The atom is not included in this rectangle.</figcaption></figure>
    <figure className="probability-chart"><svg viewBox="0 0 280 225" role="img" aria-labelledby={`${id}-cdf-title ${id}-cdf-desc`}>
      <title id={`${id}-cdf-title`}>CDF with a possible jump at zero</title><desc id={`${id}-cdf-desc`}>F(x) is zero for x below zero, jumps to {state.atom} at zero, then increases linearly to 1 at {state.width}. Gold horizontal levels mark F(a minus)={state.cdfLeftLimit} and F(b)={state.cdfRight}; their difference includes any selected atom.</desc>
      <line x1="25" x2="260" y1="155" y2="155" /><line x1="40" x2="40" y1="30" y2="155" />
      <text x="40" y="19">CDF: F(x) = P(X ≤ x)</text><text x="25" y="44">1</text><text x="18" y="159" textAnchor="end">0</text>
      <line x1="25" x2="40" y1="155" y2="155" style={{
          stroke: colors[2],
          strokeWidth: 3
        }} />
      <line x1="40" x2="40" y1="155" y2={cdfY(state.atom)} style={{
          stroke: colors[2],
          strokeDasharray: '3 3'
        }} />
      <path d={state.cdfSamples.map((point, index) => `${index ? 'L' : 'M'}${x(point.x)},${cdfY(point.value)}`).join(' ')} fill="none" style={{
          stroke: colors[2],
          strokeWidth: 3
        }} />
      <line x1="250" x2="265" y1="40" y2="40" style={{
          stroke: colors[2],
          strokeWidth: 3
        }} />
      {state.atom > 0 && <circle cx="40" cy="155" r="3" fill="#10161b" stroke={colors[2]} />}
      <circle cx="40" cy={cdfY(state.atom)} r="3" fill={colors[2]} />
      {[state.cdfLeftLimit, state.cdfRight].map((value, index) => <line key={index} x1="45" x2="250" y1={cdfY(value)} y2={cdfY(value)} style={{
          stroke: colors[0],
          strokeDasharray: '3 4'
        }} />)}
      <text x="40" y="179">0</text><text x="250" y="179" textAnchor="end">{number(state.width)}</text><text x="145" y="201" textAnchor="middle">{state.unit}</text>
    </svg><figcaption>A filled point gives F(0). A dashed jump connects limits; it is not an intermediate set of CDF values.</figcaption></figure>
  </div>;
}
export function DensityAreaLab() {
  const [width, setWidth] = useState(0.2);
  const [atom, setAtom] = useState(0);
  const [left, setLeft] = useState(0.25);
  const [right, setRight] = useState(0.75);
  const [unit, setUnit] = useState('seconds');
  const state = mixedDelayState(width, atom, left, right, unit);
  return <Investigation id="probability-density" title="Probability is area; a point mass is a jump" guidance="A uniform delay lies between 0 and .2 seconds. Its density is 5 per second. Why is that allowed? What changes when you express the same interval in milliseconds?" onReset={() => {
    setWidth(0.2);
    setAtom(0);
    setLeft(0.25);
    setRight(0.75);
    setUnit('seconds');
  }}>
    <div className="probability-controls"><Field label="Continuous delay width, seconds" value={width} onChange={setWidth} min={0.1} max={2} step={0.1} /><Field label="Immediate completion probability q" value={atom} onChange={setAtom} step={0.1} /><Field label="Left endpoint as fraction of width" value={left} onChange={value => setLeft(Math.min(value, right))} step={0.05} /><Field label="Right endpoint as fraction of width" value={right} onChange={value => setRight(Math.max(value, left))} step={0.05} /></div>
    <fieldset className="probability-choice"><legend>Display the same physical delay in</legend>{['seconds', 'milliseconds'].map(value => <button type="button" key={value} aria-pressed={unit === value} onClick={() => setUnit(value)}>{value}</button>)}</fieldset>
    <p>Selected <strong>closed interval [{number(state.left)}, {number(state.right)}] {unit}</strong>. Endpoint controls cannot cross; an endpoint stops at the other one.</p>
    <DelayPlots state={state} />
    <div className="probability-atom"><strong>Separate point mass: P(X=0) = {number(state.atom)}</strong><span>{state.left === 0 ? 'Zero belongs to the selected interval: add this mass.' : 'Zero is outside the selected interval: do not add it.'}</span></div>
    <Readout values={[[`Continuous interval area`, number(state.continuousMass)], ['Included point mass', number(state.includedAtom)], ['Total interval probability', number(state.intervalMass)], ['F(b) − F(a−)', `${number(state.cdfRight)} − ${number(state.cdfLeftLimit)}`]]} />
    <p aria-live="polite">Continuous density is {number(state.density)} per {unit === 'seconds' ? 'second' : 'millisecond'}. Changing units rescales both axes' numbers, while the selected interval probability stays {number(state.intervalMass)}.</p>
    <p><strong>Try a mixed distribution.</strong> Set q=.3, then both endpoints to zero. The interval has no width and no continuous area, yet P(X=0)=.3. An ordinary PDF alone cannot represent this whole distribution.</p>
  </Investigation>;
}
export function ArrivalCountWaitLab() {
  const [rate, setRate] = useState(2);
  const [window, setWindow] = useState(1.5);
  const [quantile, setQuantile] = useState(0.5);
  const state = arrivalWindowState(rate, window, quantile);
  const id = useId();
  const x = value => 32 + 256 * value / 2;
  return <Investigation id="probability-arrivals" title="No arrivals and a long first wait are the same event" guidance="At two events per minute, is a 1.5-minute window guaranteed to contain three events? What does changing the rate do to the first-wait distribution?" onReset={() => {
    setRate(2);
    setWindow(1.5);
    setQuantile(0.5);
  }}>
    <p>Assume a homogeneous Poisson process: constant rate and independent counts in disjoint intervals. The timeline uses twelve disclosed fixed uniform values transformed into waits; it is one synthetic realization, not observed traffic or an estimate of probability.</p>
    <div className="probability-controls"><Field label="Rate, events per minute" value={rate} min={0.25} max={6} step={0.25} onChange={setRate} /><Field label="Observation window, minutes" value={window} min={0} max={2} step={0.1} onChange={setWindow} /><Field label="First-wait quantile probability" value={quantile} min={0.01} max={0.99} step={0.01} onChange={setQuantile} /></div>
    <figure className="probability-chart probability-timeline"><svg viewBox="0 0 320 150" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
      <title id={`${id}-title`}>Fixed realization of arrivals over two minutes</title><desc id={`${id}-desc`}>Gold background marks the selected observation window. Event ticks use actual cumulative transformed waits. {state.displayedArrivalsInWindow} supplied arrivals are within the window. {state.realizationTruncated ? 'All twelve supplied events have occurred; additional arrivals are not generated.' : 'A supplied event beyond the window establishes that the displayed count inside this window is complete.'}</desc>
      <rect x="32" y="35" width={x(window) - 32} height="62" fill="#e5b95f22" />
      <line x1="32" x2="288" y1="68" y2="68" />
      {[0, 0.5, 1, 1.5, 2].map(time => <g key={time}><line x1={x(time)} x2={x(time)} y1="99" y2="104" /><text x={x(time)} y="124" textAnchor="middle">{time}</text></g>)}
      {state.arrivals.filter(event => event.time <= 2).map(event => <line key={event.event} x1={x(event.time)} x2={x(event.time)} y1="48" y2="89" style={{
          stroke: event.inWindow ? colors[0] : colors[2],
          strokeWidth: 2
        }} />)}
      <text x="32" y="22">Arrival ticks; shaded observation window</text><text x="160" y="145" textAnchor="middle">Time in minutes</text>
    </svg><figcaption>{state.realizationTruncated ? 'At least ' : ''}{state.displayedArrivalsInWindow} arrivals in this fixed realization's selected window.{state.realizationTruncated && ' The twelve-value list is exhausted; later events are not shown.'}</figcaption></figure>
    <Readout values={[[`Mean count λt`, number(state.meanCount)], ['P(no arrivals in window)', number(state.noArrivals)], ['P(first wait > window)', number(state.noArrivals)], ['P(at least one arrival)', number(state.atLeastOne)], [`Wait quantile at ${percent(quantile)}`, `${number(state.quantileWait)} minutes`], ['Mean first wait', `${number(state.meanWait)} minutes`]]} />
    <figure className="probability-chart"><svg viewBox="0 0 320 210" role="img" aria-label="Poisson count probability masses for counts 0 through 24, vertical axis from zero to one; the probability beyond 24 is reported below">
      <line x1="32" x2="294" y1="163" y2="163" /><line x1="32" x2="32" y1="33" y2="163" /><text x="32" y="20">P(N(t) = k)</text><text x="14" y="37">1</text><text x="14" y="167">0</text>
      {state.masses.map(item => <rect key={item.count} x={34 + 10 * item.count} y={163 - 125 * item.mass} width="7" height={125 * item.mass} fill={item.count === 0 ? colors[0] : colors[2]} />)}
      {[0, 6, 12, 18, 24].map(count => <text key={count} x={37.5 + 10 * count} y="183" textAnchor="middle">{count}</text>)}<text x="160" y="205" textAnchor="middle">Arrival count k</text>
    </svg><figcaption>Count probabilities, not a histogram of the single timeline. Unplotted P(N&gt;24) = {number(state.tailBeyond24)}; bars are not renormalized.</figcaption></figure>
    <details><summary>Inspect the fixed arrival calculations and count probabilities</summary>
      <LessonTable caption="Fixed uniform inputs transformed to cumulative arrival times" headers={['Event', 'Uniform u', 'Wait (minutes)', 'Arrival time', 'Inside window?']} rows={state.arrivals.map(event => [event.event, event.uniform, number(event.wait), number(event.time), event.inWindow ? 'yes' : 'no'])} />
      <LessonTable caption="Poisson masses calculated from the model" headers={['Count k', 'P(N=k)']} rows={state.masses.map(item => [item.count, number(item.mass)])} />
    </details>
    <p><strong>Transfer.</strong> Double the rate and halve the window. The mean count and its distribution stay the same, but the waiting-time distribution in minutes changes. A mean count is an average, not a schedule.</p>
  </Investigation>;
}
export function MomentTailFigure() {
  return <figure className="probability-tail-figure"><figcaption>Same mean 0 and variance 1, different tail probability</figcaption>
    {[['A', [0, 0.5, 0, 0.5, 0]], ['B', [0.125, 0, 0.75, 0, 0.125]]].map(([label, masses]) => <div className="probability-tail-row" key={label}><strong>{label}</strong><div>{masses.map((mass, index) => <span key={index}><i style={{
            height: `${mass * 100}px`
          }} /><small>x={index - 2}</small><small>{number(mass)}</small></span>)}</div></div>)}
    <p>Every column is a point mass at its labelled x. Outside [−1.5,1.5], A has mass 0 and B has mass 1/4. The same two summary numbers do not specify a distribution.</p>
  </figure>;
}
export function NormalUnitsFigure() {
  return <figure className="probability-normal-units"><figcaption>The same error interval in three coordinate systems</figcaption>
    <div><strong>Volts</strong><span>−.1</span><span>0</span><span>+.1</span></div>
    <div><strong>Millivolts<br /><small>multiply by 1000</small></strong><span>−100</span><span>0</span><span>+100</span></div>
    <div><strong>Standard deviations<br /><small>divide volts by .2</small></strong><span>−.5</span><span>0</span><span>+.5</span></div>
    <p>Aligned endpoints refer to the same event. Its probability is about .382925 in all three rows; the density's numerical height changes with units.</p>
  </figure>;
}
