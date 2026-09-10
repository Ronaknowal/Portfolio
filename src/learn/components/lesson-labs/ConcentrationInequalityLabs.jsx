import { useId, useMemo, useState } from 'react';
import { countTail, exponentialWitness, familyBudget, formatLogProbability, precisionBudget, samplingComparison } from '../../data/concentration-inequalities-models.js';
import './concentration-inequality-labs.css';
const COLORS = {
  gold: '#e9c077',
  blue: '#9dc7dc',
  green: '#adc68e',
  red: '#e6a599',
  muted: '#6f8176'
};
const numeric = value => value === 0 ? '0' : Math.abs(value) < .0001 ? value.toExponential(3) : Number(value.toPrecision(6)).toString();
function Facts({
  values
}) {
  return <dl className="concentration-facts">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function MassPlot({
  law,
  eventCounts,
  title,
  count
}) {
  const events = new Set(eventCounts);
  const probabilities = law.map(Math.exp);
  const maximum = Math.max(...probabilities);
  const left = 38,
    right = 306,
    top = 25,
    bottom = 171;
  const barWidth = (right - left) / (count + 1);
  const y = probability => bottom - (bottom - top) * probability / maximum;
  const ticks = [...new Set([0, Math.floor(count / 2), count])];
  return <svg className="concentration-plot" viewBox="0 0 320 215" role="img" aria-label={title}>
    <title>{title}</title><desc>Each bar is the numerically evaluated mass of an exact finite distribution. Warm bars belong to the declared event; pale bars do not. Tiny bars can be below a pixel; the event-probability readout retains log-scale values.</desc>
    <text x="38" y="13">probability mass</text>
    <line x1={left} y1={bottom} x2={right} y2={bottom} className="concentration-axis" />
    <line x1={left} y1={top} x2={left} y2={bottom} className="concentration-axis" />
    <text x="33" y={top + 4} textAnchor="end">{maximum.toPrecision(2)}</text><text x="33" y={bottom + 4} textAnchor="end">0</text>
    {probabilities.map((probability, index) => <rect key={index} x={left + barWidth * index + .15} y={y(probability)} width={Math.max(.25, barWidth - .3)} height={bottom - y(probability)} fill={events.has(index) ? COLORS.gold : COLORS.muted}><title>{index} successes: {formatLogProbability(Math.min(0, law[index]))}{events.has(index) ? '; in event' : ''}</title></rect>)}
    {ticks.map(tick => <text key={tick} x={left + barWidth * (tick + .5)} y="187" textAnchor="middle">{tick}</text>)}
    <text x="172" y="207" textAnchor="middle">success count S</text>
  </svg>;
}
function CurvePlot({
  series,
  xDomain,
  yDomain,
  xTicks,
  yTicks,
  xLabel,
  yLabel,
  title,
  marker
}) {
  const left = 42,
    right = 307,
    top = 27,
    bottom = 177;
  const x = value => left + (right - left) * (value - xDomain[0]) / (xDomain[1] - xDomain[0]);
  const y = value => bottom - (bottom - top) * (value - yDomain[0]) / (yDomain[1] - yDomain[0]);
  return <svg className="concentration-plot" viewBox="0 0 320 220" role="img" aria-label={title}>
    <title>{title}</title><desc>Curves use the stated analytic finite model. Axes and exact selected values appear beside the figure; they are not timing or simulation measurements.</desc>
    <text x="42" y="14">{yLabel}</text>
    {yTicks.map(([value, label]) => <g key={value}><line x1={left} y1={y(value)} x2={right} y2={y(value)} className="concentration-grid" /><text x="37" y={y(value) + 4} textAnchor="end">{label}</text></g>)}
    <line x1={left} y1={top} x2={left} y2={bottom} className="concentration-axis" /><line x1={left} y1={bottom} x2={right} y2={bottom} className="concentration-axis" />
    {series.map(line => <path key={line.label} d={line.points.map(([a, b], index) => `${index ? 'L' : 'M'}${x(a)},${y(b)}`).join(' ')} fill="none" stroke={line.color} strokeWidth="2" strokeDasharray={line.dashed ? '5 4' : undefined}><title>{line.label}</title></path>)}
    {marker && <g><line x1={x(marker.x)} y1={top} x2={x(marker.x)} y2={bottom} stroke={COLORS.gold} strokeDasharray="2 4" /><circle cx={x(marker.x)} cy={y(marker.y)} r="4" fill={COLORS.gold} /></g>}
    {xTicks.map(([value, label], index) => <text key={value} x={x(value)} y="193" textAnchor={index === 0 ? 'start' : index === xTicks.length - 1 ? 'end' : 'middle'}>{label}</text>)}
    <text x="174" y="213" textAnchor="middle">{xLabel}</text>
  </svg>;
}
function Legend({
  entries
}) {
  return <ul className="concentration-legend">{entries.map(([label, color]) => <li key={label}><span style={{
        background: color
      }} aria-hidden="true" />{label}</li>)}</ul>;
}
export function ConcentrationTailLab() {
  const [count, setCount] = useState(20),
    [percent, setPercent] = useState(25),
    [threshold, setThreshold] = useState(8);
  const state = useMemo(() => countTail(count, percent, threshold), [count, percent, threshold]);
  const id = useId();
  const choose = (nextCount, nextPercent) => {
    setCount(nextCount);
    setPercent(nextPercent);
    setThreshold(Math.min(nextCount + 1, Math.floor(nextCount * nextPercent / 100) + Math.max(1, Math.ceil(Math.sqrt(nextCount * nextPercent / 100 * (1 - nextPercent / 100))))));
  };
  const names = {
    markov: 'Markov',
    chebyshev: 'Chebyshev (two-sided)',
    hoeffding: 'Hoeffding (upper tail)',
    bernstein: 'Bernstein, centered cap 1',
    multiplicative: 'Mean-only Chernoff',
    kl: 'Binomial KL Chernoff'
  };
  return <section className="concentration-lab" aria-label="Exact count tails and bounds">
    <h3>Move the threshold; distinguish mass from a guarantee</h3>
    <p>The model is S∼Binomial(n,p): n independent binary observations with known p for this comparison. Warm bars are exactly the event S≥k. The theorem need not know p to apply in other settings.</p>
    <div className="concentration-controls">
      <label htmlFor={`${id}-n`}>Independent observations n<select id={`${id}-n`} value={count} onChange={event => choose(Number(event.target.value), percent)}>{[20, 50, 100, 200].map(value => <option key={value}>{value}</option>)}</select></label>
      <label htmlFor={`${id}-p`}>True success probability p<select id={`${id}-p`} value={percent} onChange={event => choose(count, Number(event.target.value))}>{[0, 2, 10, 25, 50, 90, 100].map(value => <option value={value} key={value}>{value}%</option>)}</select></label>
      <label htmlFor={`${id}-k`}>Inclusive threshold k: {threshold}<input id={`${id}-k`} aria-label="Inclusive count threshold" type="range" min="0" max={count + 1} value={threshold} onChange={event => setThreshold(Number(event.target.value))} /></label>
    </div>
    <figure><MassPlot law={state.law} count={count} eventCounts={Array.from({
        length: Math.max(0, count - threshold + 1)
      }, (_, index) => threshold + index)} title="Binomial mass with the inclusive upper tail highlighted" /><figcaption>Distribution bars are evaluated finite probabilities, not observed frequencies. A bound above 1 is reported as the trivial probability cap 1. The count threshold is integral, so moving it changes entire probability atoms.</figcaption></figure>
    <Facts values={[["Expected count", numeric(state.mean)], ['Requested event', `S ≥ ${threshold}`], ['Actual finite-law probability', formatLogProbability(state.exactLog, state.exactComplementLog)]]} />
    <div className="concentration-table" tabIndex={0} role="region" aria-label="Tail certificates"><table><thead><tr><th>Information used</th><th>Upper bound</th></tr></thead><tbody>{Object.entries(state.bounds).map(([name, value]) => <tr key={name}><th scope="row">{names[name]}</th><td>{formatLogProbability(value)}</td></tr>)}</tbody></table></div>
    {state.supportDetermines ? <p role="status">The support alone determines this event. The displayed certificates use that exact support fact, including p=0/1 and an impossible threshold.</p> : state.deviation <= 0 ? <p role="status">The threshold is at or below the mean. The positive upper-deviation formulas do not apply; their displayed certificate is 1.</p> : <p role="status">Deviation above the mean: {numeric(state.deviation)} counts, or {numeric(state.deviation / count)} in the sample mean. All displayed bounds apply to this same upper-tail event.</p>}
    <button onClick={() => {
      setCount(20);
      setPercent(25);
      setThreshold(8);
    }}>Reset tail comparison</button>
    <p>Start by comparing the exact mass with Hoeffding. Return after the later sections to explain what additional information each tighter bound uses. Chebyshev here bounds a larger two-sided event; Bernstein uses known variance np(1−p) and the valid conservative centered cap 1.</p>
  </section>;
}
export function ExponentialWitnessLab() {
  const [lambda, setLambda] = useState(1);
  const state = exponentialWitness(lambda);
  const id = useId();
  const curves = [{
    label: 'Exact log-MGF bound',
    color: COLORS.blue,
    points: state.curve.map(point => [point.lambda, point.exact])
  }, {
    label: 'Hoeffding envelope',
    color: COLORS.gold,
    points: state.curve.map(point => [point.lambda, point.envelope])
  }, {
    label: 'Actual log tail probability',
    color: COLORS.green,
    dashed: true,
    points: [[0, state.exactLog], [2, state.exactLog]]
  }];
  return <section className="concentration-lab" aria-label="Exponential Markov witness">
    <h3>Choose the witness, not a different experiment</h3>
    <p>Keep S∼Binomial(20,¼) and the event S≥10 fixed. λ controls which positive exponential dominates the event indicator. A larger λ emphasizes large counts but also increases their contribution to the expectation.</p>
    <label htmlFor={id}>Exponential weight λ: {lambda.toFixed(6)}<input id={id} aria-label="Exponential weight lambda" type="range" min="0" max="2" step="any" value={lambda} onChange={event => setLambda(Number(event.target.value))} /></label>
    <figure><CurvePlot series={curves} xDomain={[0, 2]} yDomain={[-5, .3]} xTicks={[[0, '0'], [1, '1'], [2, '2']]} yTicks={[[-4, '−4'], [-2, '−2'], [0, '0']]} xLabel="exponential weight λ" yLabel="natural log of bound / probability" title="Exponential bound objectives and the fixed exact tail" marker={{
        x: lambda,
        y: state.logBound
      }} /><Legend entries={curves.map(line => [line.label, line.color])} /><figcaption>The blue curve is log E[exp(λ(S−10))]; the gold curve bounds it using only the binary interval. The green horizontal line is the fixed actual log tail. Lower valid curves are sharper; a log bound 0 means probability bound 1.</figcaption></figure>
    <Facts values={[["Selected raw upper bound", numeric(Math.exp(state.logBound))], ['Actual tail probability', formatLogProbability(state.exactLog, state.exactComplementLog)], ['Optimal λ for this finite model', numeric(state.optimalLambda)]]} />
    <div className="concentration-actions"><button onClick={() => setLambda(state.optimalLambda)}>Use exact optimum</button><button onClick={() => setLambda(1)}>Reset exponential witness</button></div>
    <details><summary>Inspect contributions around the threshold</summary><p>Each row compares the event's contribution P(S=s)·1[s≥10] with P(S=s)·exp(λ(s−10)). The latter is at least the former pointwise; summing all rows gives the upper bound.</p><div className="concentration-table" tabIndex={0} role="region" aria-label="Exponential contributions"><table><thead><tr><th>s</th><th>Event contribution</th><th>Exponential contribution</th></tr></thead><tbody>{state.contributions.filter(row => [0, 5, 9, 10, 11, 15, 20].includes(row.successes)).map(row => <tr key={row.successes}><th scope="row">{row.successes}</th><td>{numeric(row.tail)}</td><td>{numeric(row.witness)}</td></tr>)}</tbody></table></div></details>
    <p>The grid curve samples λ for display; “Use exact optimum” uses ln 3 directly. This is a bound-optimization calculation, not a sampled or reweighted dataset whose tail replaces the original probability.</p>
  </section>;
}
export function ConcentrationBudgetLab() {
  const [count, setCount] = useState(100),
    [delta, setDelta] = useState(.05),
    [width, setWidth] = useState(1),
    [variance, setVariance] = useState(.09),
    [epsilon, setEpsilon] = useState(.1);
  const state = precisionBudget(count, delta, width, variance, epsilon);
  const id = useId();
  const curves = [{
    label: 'Hoeffding radius',
    color: COLORS.gold,
    points: state.curve.map(point => [Math.log10(point.n), point.hoeffding])
  }, {
    label: 'Bernstein quadratic radius',
    color: COLORS.blue,
    points: state.curve.map(point => [Math.log10(point.n), point.bernsteinExact])
  }, {
    label: 'Bernstein relaxed radius',
    color: COLORS.green,
    dashed: true,
    points: state.curve.map(point => [Math.log10(point.n), point.bernsteinRelaxed])
  }];
  const maximum = Math.max(...curves.flatMap(line => line.points.map(point => point[1]))) * 1.08;
  return <section className="concentration-lab" aria-label="Sample and variance budgets">
    <h3>Choose precision and justified information separately</h3>
    <p>All radii target two-sided failure probability at most δ. The variance control represents an externally justified upper bound on each observation's variance, not the variance measured in the current sample.</p>
    <div className="concentration-controls">
      <label htmlFor={`${id}-n`}>Observation count n<select id={`${id}-n`} value={count} onChange={event => setCount(Number(event.target.value))}>{[10, 50, 100, 185, 400, 1000].map(value => <option key={value}>{value}</option>)}</select></label>
      <label htmlFor={`${id}-delta`}>Failure budget δ<select id={`${id}-delta`} value={delta} onChange={event => setDelta(Number(event.target.value))}>{[.005, .01, .05, .1].map(value => <option key={value}>{value}</option>)}</select></label>
      <label htmlFor={`${id}-width`}>Known range width<select id={`${id}-width`} value={width} onChange={event => setWidth(Number(event.target.value))}>{[1, 10].map(value => <option key={value}>{value}</option>)}</select></label>
      <label htmlFor={`${id}-variance`}>Variance upper bound / width²<select id={`${id}-variance`} value={variance} onChange={event => setVariance(Number(event.target.value))}>{[0, .01, .09, .25].map(value => <option key={value}>{value}</option>)}</select></label>
      <label htmlFor={`${id}-epsilon`}>Absolute tolerance ε<select id={`${id}-epsilon`} value={epsilon} onChange={event => setEpsilon(Number(event.target.value))}>{[.01, .05, .1, .2, 1].map(value => <option key={value}>{value}</option>)}</select></label>
    </div>
    <figure><CurvePlot series={curves} xDomain={[1, 3]} yDomain={[0, maximum]} xTicks={[[1, '10'], [2, '100'], [3, '1000']]} yTicks={[[0, '0'], [maximum / 2, (maximum / 2).toPrecision(2)], [maximum, maximum.toPrecision(2)]]} xLabel="observations n · logarithmic axis" yLabel="error radius · metric units" title="Analytic Hoeffding and Bernstein radius curves" /><Legend entries={curves.map(line => [line.label, line.color])} /><figcaption>Analytic curves use fixed δ, width and variance bound. n is logarithmic and the radius axis is linear. Radii are not observed errors; values larger than the metric's possible range give a vacuous guarantee.</figcaption></figure>
    <Facts values={[["Hoeffding radius at selected n", numeric(state.hoeffding)], ['Bernstein quadratic radius', numeric(state.bernsteinExact)], ['Bernstein relaxed radius', numeric(state.bernsteinRelaxed)], ['Variance upper bound in units²', numeric(state.variance)], ['Sufficient Hoeffding sample count', state.needed.toLocaleString('en-US')]]} />
    <p role="status">At n={count}, the selected known variance bound is {numeric(state.variance)}. For absolute tolerance {epsilon} in the same metric units and δ={delta}, Hoeffding needs at least {state.needed.toLocaleString('en-US')} observations by this sufficient calculation.</p>
    {variance === 0 && <p>A truly known variance 0 makes every centered observation zero almost surely, so direct reasoning gives radius 0. The displayed generic Bernstein formula retains a linear term and is deliberately not optimized for that extra degeneracy fact.</p>}
    <button onClick={() => {
      setCount(100);
      setDelta(.05);
      setWidth(1);
      setVariance(.09);
      setEpsilon(.1);
    }}>Reset precision budget</button>
    <p>Try variance 0.25 to see why Bernstein need not always beat Hoeffding. Then multiply the range width by 10 while keeping ε fixed; the required Hoeffding sample count grows by about 100. This is a change of the allowed range, not a mere unit conversion unless ε changes too.</p>
  </section>;
}
export function ZeroVarianceFigure() {
  return <figure className="concentration-inline concentration-zero-variance">
    <div><h4>Population model · p=1/40</h4><div className="concentration-tickets">{Array.from({
          length: 40
        }, (_, index) => <span className={index === 0 ? 'is-marked' : ''} key={index}>{index === 0 ? '1' : '0'}</span>)}</div><p>Independent uniform draws with replacement from these forty labelled records have success probability 1/40.</p></div>
    <div><h4>One possible sample · 100 zeros</h4><div className="concentration-zero-sample" aria-label="One hundred observed zeros">{Array.from({
          length: 100
        }, (_, index) => <span key={index} aria-hidden="true">0</span>)}</div><p>The sample mean and sample variance are both 0. The true mean is 0.025, and the true variance is positive.</p></div>
    <figcaption>This is an explicitly chosen possible outcome, not simulation evidence. Its probability is exactly (39/40)¹⁰⁰≈0.07951729. The native calculation shows that blindly plugging variance 0 into the known-variance Bernstein formula gives radius 0.02459253 and misses the true mean.</figcaption>
  </figure>;
}
export function SamplingDependenceLab() {
  const [count, setCount] = useState(20),
    [epsilon, setEpsilon] = useState(20);
  const state = samplingComparison(count, epsilon);
  const id = useId();
  return <section className="concentration-lab" aria-label="Sampling dependence comparison">
    <h3>Same marginal mean; three different information patterns</h3>
    <p>The finite population has 20 labelled records, 5 marked. Compare independent draws with replacement, one random draw copied n times, and a uniform n-record subset without replacement. Every individual observation has mean ¼.</p>
    <div className="concentration-controls"><label htmlFor={`${id}-n`}>Number of observations n<select id={`${id}-n`} value={count} onChange={event => setCount(Number(event.target.value))}>{[1, 5, 10, 15, 20].map(value => <option key={value}>{value}</option>)}</select></label><label htmlFor={`${id}-epsilon`}>Absolute tolerance: {epsilon} percentage points<input id={`${id}-epsilon`} aria-label="Absolute tolerance in percentage points" type="range" min="1" max="80" value={epsilon} onChange={event => setEpsilon(Number(event.target.value))} /></label></div>
    <p className="concentration-event">Event: |S/n −¼|≥{epsilon / 100}. Warm bars are the event, with equality included.</p>
    <div className="concentration-law-comparison">{state.laws.map(law => <figure key={law.id}><h4>{law.label}</h4><MassPlot law={law.logMass} count={count} eventCounts={state.eventCounts} title={`${law.label}: exact count distribution`} /><Facts values={[["Exact finite-law event probability", formatLogProbability(law.logFailure)], ['Variance of the sample mean', numeric(law.meanVariance)]]} /><figcaption>{law.guarantee}. Each plot's vertical scale is labelled; use the probability readout rather than comparing bar heights across different scales.</figcaption></figure>)}</div>
    <p role="status">The common Hoeffding expression is {formatLogProbability(state.nominalHoeffdingLog)}. It is valid for the first law and, by the separately stated without-replacement theorem, the third; applying it to the copied observations is unjustified.</p>
    <button onClick={() => {
      setCount(20);
      setEpsilon(20);
    }}>Reset sampling comparison</button>
    <p>At n=20 the uniform subset is the complete population, so its mean equals ¼ exactly. The copied mean remains either 0 or 1. More stored rows need not mean more independent information.</p>
  </section>;
}
export function SimultaneousBudgetLab() {
  const [count, setCount] = useState(100),
    [checks, setChecks] = useState(100),
    [delta, setDelta] = useState(.05),
    [relation, setRelation] = useState('independent');
  const state = familyBudget(count, checks, delta, relation);
  const id = useId();
  const curves = [{
    label: 'Unadjusted actual family failure',
    color: COLORS.red,
    points: state.curve.map(point => [point.checks, point.naive])
  }, {
    label: 'Allocated actual family failure',
    color: COLORS.blue,
    points: state.curve.map(point => [point.checks, point.allocated])
  }, {
    label: 'Requested global δ',
    color: COLORS.gold,
    dashed: true,
    points: [[1, delta], [100, delta]]
  }];
  return <section className="concentration-lab" aria-label="Simultaneous error budget">
    <h3>A family of checks spends one global budget</h3>
    <p>Each toy check estimates a Bernoulli(½) mean with n observations. Compare a Hoeffding radius using δ for every check with one using δ/K for each of K fixed checks. Failure means at least one absolute error reaches or exceeds its radius. This also bounds missing a closed interval, whose boundary equality would still count as coverage.</p>
    <div className="concentration-controls"><label htmlFor={`${id}-n`}>Observations per check<select id={`${id}-n`} value={count} onChange={event => setCount(Number(event.target.value))}>{[20, 50, 100, 400].map(value => <option key={value}>{value}</option>)}</select></label><label htmlFor={`${id}-k`}>Fixed checks K: {checks}<input id={`${id}-k`} aria-label="Number of fixed checks" type="range" min="1" max="100" value={checks} onChange={event => setChecks(Number(event.target.value))} /></label><label htmlFor={`${id}-delta`}>Global failure budget<select id={`${id}-delta`} value={delta} onChange={event => setDelta(Number(event.target.value))}>{[.01, .05, .1].map(value => <option key={value}>{value}</option>)}</select></label><label htmlFor={`${id}-relation`}>Relation between check failures<select id={`${id}-relation`} value={relation} onChange={event => setRelation(event.target.value)}><option value="independent">Independent toy checks</option><option value="identical">The same check repeated</option></select></label></div>
    <figure><CurvePlot series={curves} xDomain={[1, 100]} yDomain={[0, 1]} xTicks={[[1, '1'], [50, '50'], [100, '100']]} yTicks={[[0, '0'], [.5, '0.5'], [1, '1']]} xLabel="fixed checks K" yLabel="probability of any failure" title="Finite family failure with and without error allocation" /><Legend entries={curves.map(line => [line.label, line.color])} /><figcaption>Exact finite binomial miss probabilities are combined using the selected independence/identity model. Between-integer lines are guides. Allocated curves can jump when a radius passes a discrete count. The union guarantee itself requires neither relation.</figcaption></figure>
    <Facts values={[["Unadjusted radius", numeric(state.naiveRadius)], ['Allocated radius', numeric(state.allocatedRadius)], ['Actual unadjusted family failure', formatLogProbability(state.naiveFamilyLog)], ['Actual allocated family failure', formatLogProbability(state.allocatedFamilyLog)], ['Unadjusted union certificate', numeric(state.naiveCertificate)], ['Allocated union certificate', numeric(state.allocatedCertificate)]]} />
    <p role="status">{checks} fixed checks, each with {count} observations. The allocated per-check failure budget is {numeric(delta / checks)}; adding those budgets gives {delta} even when the failure events are dependent.</p>
    <button onClick={() => {
      setCount(100);
      setChecks(100);
      setDelta(.05);
      setRelation('independent');
    }}>Reset family budget</button>
    <p>The independent family is a declared toy model, not a claim that models evaluated on one shared dataset are independent. A finite family chosen before evaluation may be handled uniformly; inventing unrestricted new checks from the same outcomes needs additional reasoning.</p>
  </section>;
}
