import { useId, useMemo, useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { batchPredictionState, bayesianNumber as number, betaPriors, betaUpdateState, exposureUpdateState, normalPrecisionState, predictivePatternState, sequencePatterns } from '../../data/bayesian-inference-models.js';
import './bayesian-inference-labs.css';
const gold = '#e5b95f',
  blue = '#79bbc9',
  green = '#a0c9ac';
function Slider({
  label,
  value,
  setValue,
  min,
  max,
  step = 1
}) {
  const id = useId();
  return <label className="bayesian-field" htmlFor={id}><span>{label}: <strong>{number(value)}</strong></span>
    <input id={id} aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => setValue(Number(event.target.value))} />
  </label>;
}
function Investigation({
  id,
  title,
  guidance,
  children,
  reset
}) {
  const heading = useId();
  return <section data-live-exploration className="bayesian-lab lesson-lab" data-investigation={id} aria-labelledby={heading}>
    <h3 id={heading}>{title}</h3><p> {guidance}</p>
    {children}<button type="button" className="bayesian-reset" onClick={reset}>Reset investigation</button>
  </section>;
}
function Readout({
  values
}) {
  return <dl className="bayesian-readout">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function DensityPlot({
  points,
  maxX = 1,
  shade,
  caption,
  axis
}) {
  const ymax = Math.max(...points.flatMap(point => [point.prior, point.posterior])) * 1.08;
  const x = value => 48 + 280 * value / maxX,
    y = value => 190 - 155 * value / ymax;
  const curve = field => points.map((point, index) => `${index ? 'L' : 'M'}${x(point.x)},${y(point[field])}`).join(' ');
  const fill = shade ? `M${x(shade[0].x)},190 ${shade.map(point => `L${x(point.x)},${y(point.density)}`).join(' ')} L${x(shade.at(-1).x)},190 Z` : '';
  return <figure className="bayesian-plot"><figcaption>{caption}</figcaption>
    <svg viewBox="0 0 360 245" role="img" aria-label={caption}>
      {[0, ymax / 2, ymax].map(value => <g key={value}><line x1="48" x2="328" y1={y(value)} y2={y(value)} stroke="#394c57" /><text x="42" y={y(value) + 5} textAnchor="end">{number(value, 1)}</text></g>)}
      {fill && <path d={fill} fill={gold} opacity=".22" />}
      <path d={curve('prior')} stroke="#d7dee1" strokeDasharray="6 5" fill="none" strokeWidth="2" />
      <path d={curve('posterior')} stroke={gold} fill="none" strokeWidth="3" />
      {[0, maxX / 2, maxX].map(value => <text key={value} x={x(value)} y="214" textAnchor="middle">{number(value, 2)}</text>)}
      <text x="188" y="241" textAnchor="middle">{axis}</text>
    </svg>
    <p className="bayesian-legend">Dashed: prior · Gold: posterior · Vertical axis: density</p>
  </figure>;
}
export function BetaEvidenceFigure() {
  return <figure className="bayesian-inline" aria-label="Separate observed successes and failures update separate Beta shapes">
    <figcaption><strong>Two kinds of evidence, two separate additions</strong></figcaption>
    <div className="bayesian-outcomes">{Array.from({
        length: 10
      }, (_, index) => <span className={index < 8 ? 'success' : 'failure'} key={index}>{index < 8 ? 'S' : 'F'}</span>)}</div>
    <p>Eight successes and two failures. Grouping is for counting; these are actual observed outcomes, while prior shapes are model parameters.</p>
    <div className="bayesian-evidence-lanes"><div><span>α = 2</span><span>+ 8 successes</span><strong>a = 10</strong></div><div><span>β = 2</span><span>+ 2 failures</span><strong>b = 4</strong></div></div>
    <p>Beta(2, 2) → Beta(10, 4). Mean = 10/14 ≈ .714; the prior does not add four real visitors.</p>
  </figure>;
}
export function BayesianUpdateLab() {
  const [prior, setPrior] = useState('moderate'),
    [successes, setSuccesses] = useState(8),
    [failures, setFailures] = useState(2);
  const [threshold, setThreshold] = useState(.7),
    [level, setLevel] = useState(.95);
  const state = useMemo(() => betaUpdateState(prior, successes, failures, threshold, level), [prior, successes, failures, threshold, level]);
  return <Investigation id="beta-update" title="Keep the whole posterior visible" guidance="With the same 8/10 observations, will a stronger middle prior move the mean toward .8 or toward .5?" reset={() => {
    setPrior('moderate');
    setSuccesses(8);
    setFailures(2);
    setThreshold(.7);
    setLevel(.95);
  }}>
    <div className="bayesian-controls">
      <label className="bayesian-field">Prior<select aria-label="Prior" value={prior} onChange={event => setPrior(event.target.value)}>{betaPriors.map(item => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label>
      <Slider label="Successes" value={successes} setValue={setSuccesses} min={0} max={80} />
      <Slider label="Failures" value={failures} setValue={setFailures} min={0} max={20} />
      <Slider label="Rate threshold" value={threshold} setValue={setThreshold} min={0} max={1} step={.01} />
      <label className="bayesian-field">Credible mass<select aria-label="Credible mass" value={level} onChange={event => setLevel(Number(event.target.value))}>{[.8, .9, .95].map(value => <option key={value} value={value}>{100 * value}% equal-tailed</option>)}</select></label>
    </div>
    <DensityPlot points={state.points} shade={state.shade} caption={`Prior Beta(${state.prior.a}, ${state.prior.b}); posterior Beta(${state.a}, ${state.b}). Shaded posterior area contains ${100 * state.level}% probability.`} axis="Unknown probability θ" />
    <Readout values={[['Posterior mean / next-success chance', number(state.mean, 6)], ['Mode in θ coordinates', number(state.mode, 6)], ['Equal-tailed credible interval', `[${number(state.low, 6)}, ${number(state.high, 6)}]`], [`P(θ > ${number(threshold)} | data)`, number(state.above, 6)], ['Observed fraction', state.sampleRate === null ? 'No observations' : number(state.sampleRate, 6)]]} />
    <p className="bayesian-result" aria-live="polite">Posterior Beta({state.a}, {state.b}); mean {number(state.mean, 6)}. {successes + failures === 0 ? 'With no data, posterior equals prior.' : 'The prior and posterior share one density axis; heights are not point probabilities.'}</p>
    <p>Try 8 successes / 2 failures, then 80 / 20. Next remove all observations. Changing the displayed credible mass changes the interval, not the posterior itself. The drawn curve samples a bounded analytic density; the interval endpoints use a separate tail inversion.</p>
  </Investigation>;
}
export function SharedParameterFigure() {
  return <figure className="bayesian-inline bayesian-shared" aria-label="One uncertain rate is shared by all future outcomes in a batch">
    <figcaption><strong>The whole batch shares one unknown rate</strong></figcaption>
    <div className="bayesian-shared-source">θ from the posterior</div><div className="bayesian-shared-branches"><span>Visitor 1</span><span>Visitor 2</span><span>Visitor 3</span></div>
    <p>At a fixed θ, outcomes are conditionally independent. Averaging over that same θ connects them: learning that one succeeded also informs the rate shared with the others. Drawing a new independent rate for every visitor defines a different model.</p>
  </figure>;
}
export function BatchPredictionLab() {
  const [posterior, setPosterior] = useState('10,4'),
    [size, setSize] = useState(10),
    [threshold, setThreshold] = useState(9);
  const [a, b] = posterior.split(',').map(Number);
  const state = useMemo(() => batchPredictionState(a, b, size, threshold), [a, b, size, threshold]);
  const ymax = Math.max(...state.points.flatMap(point => [point.integrated, point.plugin])) * 1.1;
  const bar = 280 / (size + 1);
  return <Investigation id="batch-prediction" title="Same expected count, different tail risk" guidance="For ten future visitors, does fixing θ at its posterior mean preserve the chance of at least nine successes?" reset={() => {
    setPosterior('10,4');
    setSize(10);
    setThreshold(9);
  }}>
    <div className="bayesian-controls"><label className="bayesian-field">Rate uncertainty<select aria-label="Rate uncertainty" value={posterior} onChange={event => setPosterior(event.target.value)}><option value="10,4">Beta(10, 4): original posterior</option><option value="100,40">Beta(100, 40): same mean, stronger concentration</option><option value="1,1">Beta(1, 1): no data, uniform prior</option></select></label>
      <Slider label="Future batch size" value={size} setValue={value => {
        setSize(value);
        setThreshold(Math.min(value, threshold));
      }} min={1} max={20} />
      <Slider label="At least this many successes" value={threshold} setValue={setThreshold} min={0} max={size} />
    </div>
    <figure className="bayesian-plot"><figcaption>Gold: integrated Beta-Binomial. Blue: Binomial at the mean. Faint columns are outside the selected tail.</figcaption>
      <svg viewBox="0 0 360 245" role="img" aria-label={`Future count probabilities; integrated tail ${number(state.integratedTail, 6)}, plug-in tail ${number(state.pluginTail, 6)}.`}>
        {[0, ymax / 2, ymax].map(value => <g key={value}><line x1="48" x2="328" y1={190 - 155 * value / ymax} y2={190 - 155 * value / ymax} stroke="#394c57" /><text x="42" y={195 - 155 * value / ymax} textAnchor="end">{number(value, 2)}</text></g>)}
        {state.points.map(point => <g key={point.k} opacity={point.k >= threshold ? 1 : .3}>
          <rect x={48 + point.k * bar + bar * .08} y={190 - 155 * point.integrated / ymax} width={bar * .38} height={155 * point.integrated / ymax} fill={gold} />
          <rect x={48 + point.k * bar + bar * .53} y={190 - 155 * point.plugin / ymax} width={bar * .38} height={155 * point.plugin / ymax} fill={blue} />
        </g>)}
        {[...new Set([0, Math.floor(size / 2), size])].map(k => <text key={k} x={48 + (k + .5) * bar} y="213" textAnchor="middle">{k}</text>)}
        <text x="188" y="240" textAnchor="middle">Future successes K</text>
      </svg>
    </figure>
    <Readout values={[['Expected count, both models', number(state.mean, 6)], ['Integrated / plug-in variance', `${number(state.variance, 6)} / ${number(state.pluginVariance, 6)}`], ['Integrated selected tail', number(state.integratedTail, 6)], ['Plug-in selected tail', number(state.pluginTail, 6)]]} />
    <p className="bayesian-result" aria-live="polite">Correlation between two distinct future outcomes: {number(state.correlation, 6)}. {size === 1 ? 'For one future visitor, the two count distributions agree.' : 'A shared uncertain rate changes the joint distribution, even though each visitor has the same marginal success chance.'}</p>
    <details><summary>Inspect every count probability</summary><LessonTable caption="Calculated count masses" headers={['K', 'Integrated', 'Plug-in']} rows={state.points.map(point => [point.k, number(point.integrated, 6), number(point.plugin, 6)])} /></details>
  </Investigation>;
}
export function GammaExposureLab() {
  const [count1, setCount1] = useState(3),
    [count2, setCount2] = useState(6),
    [hours1, setHours1] = useState(.5),
    [hours2, setHours2] = useState(2);
  const state = useMemo(() => exposureUpdateState(count1, hours1, count2, hours2), [count1, count2, hours1, hours2]);
  return <Investigation id="gamma-exposure" title="How much observation time produced those events?" guidance="Keep both event counts fixed and double an observation window. Which direction should the inferred events-per-hour rate move?" reset={() => {
    setCount1(3);
    setCount2(6);
    setHours1(.5);
    setHours2(2);
  }}>
    <div className="bayesian-controls">
      <Slider label="First interval events" value={count1} setValue={setCount1} min={0} max={20} />
      <Slider label="First interval hours" value={hours1} setValue={setHours1} min={.25} max={4} step={.25} />
      <Slider label="Second interval events" value={count2} setValue={setCount2} min={0} max={20} />
      <Slider label="Second interval hours" value={hours2} setValue={setHours2} min={.25} max={4} step={.25} />
    </div>
    <figure className="bayesian-exposures"><figcaption>Each track represents four hours. Width is actual exposure; individual event times were not observed.</figcaption>{[[hours1, count1], [hours2, count2]].map(([hours, count], index) => <div key={index}><span>Interval {index + 1}: {number(hours)} h, {count} events</span><div className="bayesian-exposure-track"><i style={{
            width: `${100 * hours / 4}%`
          }} /></div></div>)}</figure>
    <DensityPlot points={state.points} maxX={state.max} caption={`Gamma prior shape 2, rate 1 hour. Posterior shape ${state.shape}, rate ${number(state.rate)} hours. The plotted window contains at least 99.5% of each curve's probability; positive tails continue right.`} axis="Event rate λ, per hour" />
    <Readout values={[['Count / exposure MLE', number(state.mle, 6)], ['Posterior mean, events/hour', number(state.mean, 6)], ['95% parameter interval, events/hour', `[${number(state.low, 6)}, ${number(state.high, 6)}]`]]} />
    <p className="bayesian-result" aria-live="polite">Posterior Gamma({state.shape}, {number(state.rate)} hours). Event counts add to shape; observation durations add to the Gamma rate parameter. The two uses of “rate” have different units.</p>
  </Investigation>;
}
export function DirichletCompositionFigure() {
  const labels = ['Category A', 'Category B', 'Category C'],
    colors = [gold, blue, green];
  return <figure className="bayesian-inline" aria-label="Three category probabilities stay on a simplex and sum to one">
    <figcaption><strong>A composition, not three independent probabilities</strong></figcaption>
    <div className="bayesian-composition" role="img" aria-label="Posterior mean probabilities: seven thirteenths, four thirteenths, two thirteenths">{[7, 4, 2].map((value, index) => <span key={value} style={{
        width: `${value / 13 * 100}%`,
        background: colors[index]
      }} />)}</div>
    <div className="bayesian-category-key">{labels.map((label, index) => <span key={label}><i style={{
          background: colors[index]
        }} />{label}: {[7, 4, 2][index]}/13</span>)}</div>
    <p>Prior (1, 1, 1) + observed counts (6, 3, 1) = posterior shapes (7, 4, 2). The strip shows the mean composition. It is not a credible region and does not show all the posterior's uncertainty.</p>
  </figure>;
}
export function NormalPrecisionLab() {
  const [priorSd, setPriorSd] = useState(1),
    [noiseSd, setNoiseSd] = useState(2),
    [mean, setMean] = useState(4),
    [count, setCount] = useState(4);
  const state = useMemo(() => normalPrecisionState(0, priorSd, mean, count, noiseSd), [priorSd, noiseSd, mean, count]);
  const low = Math.min(...state.intervals.map(item => item.low)) - .5,
    high = Math.max(...state.intervals.map(item => item.high)) + .5;
  const x = value => 34 + 290 * (value - low) / (high - low);
  return <Investigation id="normal-precision" title="A better-known mean still produces noisy readings" guidance="If you collect more genuinely independent measurements with the same average, must the next individual reading become nearly certain?" reset={() => {
    setPriorSd(1);
    setNoiseSd(2);
    setMean(4);
    setCount(4);
  }}>
    <div className="bayesian-controls">
      <Slider label="Prior standard deviation" value={priorSd} setValue={setPriorSd} min={.25} max={4} step={.25} />
      <Slider label="Known observation standard deviation" value={noiseSd} setValue={setNoiseSd} min={.25} max={4} step={.25} />
      <Slider label="Observed sample mean" value={mean} setValue={setMean} min={-6} max={6} step={.25} />
      <Slider label="Independent measurement count" value={count} setValue={setCount} min={1} max={40} />
    </div>
    <div className="bayesian-precision-balance"><span>Blue · prior precision: {number(state.priorPrecision)}</span><span>Gold · data precision: {number(state.dataPrecision)}</span><div><i style={{
          width: `${100 * state.priorPrecision / (state.priorPrecision + state.dataPrecision)}%`
        }} /></div></div>
    <figure className="bayesian-plot"><figcaption>All three intervals contain 95% of their respective normal distributions. Dots are means; endpoints share one measurement axis.</figcaption>
      <svg viewBox="0 0 360 294" role="img" aria-label={state.intervals.map(item => `${item.name}: ${number(item.low)} to ${number(item.high)}`).join('; ')}>
        {state.intervals.map((item, index) => <g key={item.kind}>
          <text x="34" y={25 + 76 * index}>{item.name}</text>
          <line x1={x(item.low)} x2={x(item.high)} y1={50 + 76 * index} y2={50 + 76 * index} stroke={[blue, gold, green][index]} strokeWidth="4" />
          {[item.low, item.high].map((value, edge) => <line key={edge} x1={x(value)} x2={x(value)} y1={42 + 76 * index} y2={58 + 76 * index} stroke={[blue, gold, green][index]} strokeWidth="2" />)}
          <circle cx={x(item.mean)} cy={50 + 76 * index} r="4" fill={[blue, gold, green][index]} />
        </g>)}
        <line x1="34" x2="324" y1="231" y2="231" stroke="#697d88" />
        {[low, (low + high) / 2, high].map(value => <text key={value} x={x(value)} y="254" textAnchor="middle">{number(value, 1)}</text>)}
        <text x="180" y="278" textAnchor="middle">Measurement units</text>
      </svg>
    </figure>
    <Readout values={[['Posterior mean', number(state.mean, 6)], ['Posterior variance of mean', number(state.variance, 6)], ['Variance of next reading', number(state.variance + noiseSd ** 2, 6)]]} />
    <p className="bayesian-result" aria-live="polite">More independent readings can reduce uncertainty in μ. Prediction still includes the known observation variance {number(noiseSd ** 2)}. Changing n here compares hypothetical datasets with the displayed mean; it does not duplicate existing measurements.</p>
    <details><summary>Inspect interval endpoints</summary><LessonTable caption="Equal-tailed normal intervals" headers={['Quantity', 'Lower', 'Upper']} rows={state.intervals.map(item => [item.name, number(item.low, 6), number(item.high, 6)])} /></details>
  </Investigation>;
}
export function PredictivePatternLab() {
  const [pattern, setPattern] = useState('clustered'),
    [conditioning, setConditioning] = useState('posterior');
  const state = useMemo(() => predictivePatternState(pattern, conditioning), [pattern, conditioning]);
  const ymax = Math.max(...state.masses.map(item => item.mass)) * 1.1;
  return <Investigation id="predictive-patterns" title="The posterior can miss an ordering problem" guidance="Reorder the same eight successes and two failures. Does the common-rate Beta posterior change? Does the visible pattern change?" reset={() => {
    setPattern('clustered');
    setConditioning('posterior');
  }}>
    <fieldset className="bayesian-choices"><legend>Observed sequence</legend>{sequencePatterns.map(item => <button type="button" key={item.id} aria-pressed={pattern === item.id} onClick={() => setPattern(item.id)}>{item.label}</button>)}</fieldset>
    <div className="bayesian-outcomes">{state.pattern.values.map((value, index) => <span key={index} className={value ? 'success' : 'failure'}>{value ? 'S' : 'F'}</span>)}</div>
    <p>A <strong>run</strong> is a consecutive block of one outcome. This sequence has {state.observedRuns} runs. Both sequences give posterior Beta(10, 4) because that likelihood uses totals.</p>
    <label className="bayesian-field">Which reference distribution?<select aria-label="Predictive reference" value={conditioning} onChange={event => setConditioning(event.target.value)}><option value="posterior">Posterior predictive: repeat ten new trials</option><option value="prior">Prior predictive: ten trials before these data</option><option value="same-count">Condition on exactly eight successes</option></select></label>
    <figure className="bayesian-plot"><figcaption>{conditioning === 'same-count' ? 'All 45 orderings of eight successes are equally likely under the common-rate model. This conditions on the observed total.' : 'Exact integrated probabilities over all 1,024 ten-trial sequences. Counts may differ from the observed total.'} Gold marks at most the observed number of runs.</figcaption>
      <svg viewBox="0 0 360 245" role="img" aria-label={`Run count reference; P(runs at most ${state.observedRuns}) equals ${number(state.lowerTail, 6)}.`}>
        {[0, ymax / 2, ymax].map(value => <g key={value}><line x1="48" x2="328" y1={190 - 155 * value / ymax} y2={190 - 155 * value / ymax} stroke="#394c57" /><text x="42" y={195 - 155 * value / ymax} textAnchor="end">{number(value, 2)}</text></g>)}
        {state.masses.map(item => <g key={item.runs}><rect x={48 + (item.runs - 1) * 28 + 4} y={190 - 155 * item.mass / ymax} height={155 * item.mass / ymax} width="20" fill={item.runs <= state.observedRuns ? gold : blue} /></g>)}
        {[1, 5, 10].map(value => <text key={value} x={48 + (value - .5) * 28} y="213" textAnchor="middle">{value}</text>)}
        <text x="188" y="240" textAnchor="middle">Number of runs</text>
      </svg>
    </figure>
    <p className="bayesian-result" aria-live="polite">Selected lower-tail probability: <strong>{number(state.lowerTail, 6)}</strong>. {conditioning === 'same-count' ? 'Conditioning on the total removes θ and compares orderings. This is a different reference question.' : 'A predictive diagnostic is conditional on the chosen model and prior. It is not a classical uniformly calibrated p-value.'}</p>
    <p>Choosing this statistic after noticing a pattern changes the assessment context. A small tail is a reason to investigate time drift or dependence; neither this single check nor a comfortable value proves the model correct.</p>
    <details><summary>Inspect run probabilities</summary><LessonTable caption="Exact finite reference distribution" headers={['Runs', 'Probability']} rows={state.masses.map(item => [item.runs, number(item.mass, 6)])} /></details>
  </Investigation>;
}
