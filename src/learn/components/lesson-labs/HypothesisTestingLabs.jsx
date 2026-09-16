import { useState } from 'react';
import { LessonTable } from './LessonElements';
import { PAIRED_OLD, PAIRED_NEW, PAIRED_DIFFERENCES, EFFECT_PRESETS, inferenceNumber as number, inferencePercent as percent, intervalCoverageState, pairedTailState, practicalEffectState, t4Density, normalDensity, plannedPowerState, predictionWidths, familyErrorState, optionalLooksState, clusterPrecisionState, signFlipState, proportionCoverageState } from '../../data/hypothesis-testing-models.js';
import './hypothesis-testing-labs.css';
const GOLD = '#e2b55a';
const PINK = '#ff9db2';
const BLUE = '#8ecbd1';
const GREY = '#777';
const WHITE = '#eee';
function Range({
  label,
  value,
  setValue,
  min,
  max,
  step = 1,
  unit = ''
}) {
  return <label>{label}: <strong>{number(value)}{unit}</strong><input type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={event => setValue(Number(event.target.value))} /></label>;
}
function Axis({
  scale,
  values,
  y = 190
}) {
  return <g className="hypothesis-axis"><line x1="28" x2="338" y1={y} y2={y} />{values.map(value => <g key={value}><line x1={scale(value)} x2={scale(value)} y1={y} y2={y + 5} /><text x={scale(value)} y={y + 25} textAnchor="middle">{number(value, 1)}</text></g>)}</g>;
}
function curvePath(fn, low, high, scaleX, scaleY, samples = 220) {
  return Array.from({
    length: samples + 1
  }, (_, index) => {
    const x = low + (high - low) * index / samples;
    return `${index ? 'L' : 'M'}${scaleX(x)},${scaleY(fn(x))}`;
  }).join(' ');
}
function areaPath(fn, low, high, scaleX, scaleY, base) {
  if (!(high > low)) return '';
  return `M${scaleX(low)},${base} ${curvePath(fn, low, high, scaleX, scaleY).replace(/^M/, 'L')} L${scaleX(high)},${base} Z`;
}
function Marker({
  x,
  lowY,
  highY,
  color = GOLD,
  dashed = false
}) {
  return <line x1={x} x2={x} y1={lowY} y2={highY} stroke={color} strokeWidth="1.7" strokeDasharray={dashed ? '5 4' : undefined} />;
}
function Metric({
  label,
  children
}) {
  return <div><dt>{label}</dt><dd>{children}</dd></div>;
}
export function PairedDifferenceFigure() {
  return <figure className="hypothesis-inline" aria-label="Five matched workloads become five saving measurements">
    <figcaption>Keep each workload together before calculating a saving</figcaption>
    <LessonTable caption="Milliseconds per workload; positive means the new version is faster" headers={['Matched workload', 'Old → new', 'Old − new']} rows={PAIRED_OLD.map((old, index) => [`${index + 1}`, `${old} → ${PAIRED_NEW[index]} ms`, `${PAIRED_DIFFERENCES[index]} ms`])} />
    <p>Five pairs → five differences → one sample mean of <strong>2 ms saved</strong>. Workload 3 became 1 ms slower; the mean does not say every workload improved.</p>
  </figure>;
}
export function ConfidenceCoverageLab() {
  const [n, setN] = useState(25);
  const [level, setLevel] = useState(95);
  const [batch, setBatch] = useState(0);
  const [mode, setMode] = useState('known');
  const state = intervalCoverageState(n, level, batch, mode);
  const minimum = Math.min(78, Math.floor(Math.min(...state.intervals.map(row => row.low)) / 10) * 10);
  const maximum = Math.max(122, Math.ceil(Math.max(...state.intervals.map(row => row.high)) / 10) * 10);
  const scale = value => 28 + 310 * (value - minimum) / (maximum - minimum);
  const ticks = [minimum, 100, maximum];
  const reset = () => {
    setN(25);
    setLevel(95);
    setBatch(0);
    setMode('known');
  };
  return <section className="hypothesis-lab" data-lab="confidence-coverage" aria-label="Repeated confidence interval coverage">
    <h3>Watch 40 intervals move while the truth stays fixed</h3>
    <p>Imagine independent Normal measurements with population mean <strong>100 ms</strong> and population SD <strong>10 ms</strong>. Each row represents a separate experiment. Start with known σ; then estimate σ to see why t intervals have different widths.</p>
    <div className="hypothesis-controls">
      <Range label="Observations per experiment" value={n} setValue={setN} min={5} max={100} step={5} />
      <label>Confidence level<select aria-label="Coverage confidence level" value={level} onChange={event => setLevel(Number(event.target.value))}><option value="90">90%</option><option value="95">95%</option><option value="99">99%</option></select></label>
      <label>How spread is obtained<select aria-label="Coverage spread model" value={mode} onChange={event => setMode(event.target.value)}><option value="known">Known σ: z interval</option><option value="estimated">Estimate σ: t interval</option></select></label>
      <button type="button" onClick={() => setBatch(value => value + 1)} disabled={batch === 1000000}>Draw 40 new experiments</button>
      <button type="button" onClick={reset}>Reset coverage</button>
    </div>
    <div className="hypothesis-legend"><span className="hypothesis-gold">Solid: covers 100</span><span className="hypothesis-pink">Dashed: misses 100</span><span>White line: fixed truth</span></div>
    <svg className="hypothesis-plot hypothesis-coverage" viewBox="0 0 366 465" role="img" aria-label={`${state.covered} of 40 intervals cover the fixed mean 100 milliseconds`}>
      <Marker x={scale(100)} lowY={8} highY={424} color={WHITE} />
      {state.intervals.map((row, index) => {
        const y = 14 + index * 10;
        return <g key={index}><line x1={scale(row.low)} x2={scale(row.high)} y1={y} y2={y} stroke={row.covers ? GOLD : PINK} strokeWidth="2" strokeDasharray={row.covers ? undefined : '5 3'} /><circle cx={scale(row.mean)} cy={y} r="2.4" fill={row.covers ? GOLD : PINK} /></g>;
      })}
      <Axis scale={scale} values={ticks} y={428} />
    </svg>
    <p className="hypothesis-axis-label">Population-mean interval endpoints, in milliseconds</p>
    <dl className="hypothesis-metrics" aria-live="polite"><Metric label="This finite batch">{state.covered}/40 cover ({(state.covered * 2.5).toFixed(1)}%)</Metric><Metric label="Procedure target">{level}% under the stated model</Metric><Metric label="Critical multiplier">{number(state.critical, 5)}{mode === 'estimated' ? `, df ${n - 1}` : ''}</Metric></dl>
    <p><strong>Predict, then change:</strong> 25→100 observations shrinks the known-σ half-width by half. The standardized mean draws are shared until you request a new batch, so those means also move halfway toward 100 and the same rows still cover. That is a controlled comparison, not evidence that sample size cannot improve precision.</p>
    <p>Switching to estimated spread preserves the means and generates a separate Normal-model sample variance for each row. Varying confidence keeps that batch fixed. The finite pseudorandom display is reproducible; a fresh batch need not cover exactly {level}%.</p>
    <details><summary>Inspect all 40 intervals and the sampling construction</summary><p>Known mode samples each mean as 100+(10/√n)Z. Estimated mode independently samples (n−1)S²/100 as a sum of n−1 squared standard Normal draws, the exact mean/variance factorization for independent Normal data. A seeded linear congruential generator and Box–Muller transformation supply the reproducible demonstration draws.</p><LessonTable caption={`Batch ${batch}; displayed values rounded to three decimals`} headers={['Experiment', 'Mean', 'SE used', 'Interval', 'Covers 100?']} rows={state.intervals.map(row => [row.index, number(row.mean), number(row.se), `[${number(row.low)}, ${number(row.high)}]`, row.covers ? 'Yes' : 'No'])} /></details>
  </section>;
}
export function MeanPredictionLab() {
  const [n, setN] = useState(25);
  const state = predictionWidths(n);
  const scale = value => 28 + (value - 75) / 50 * 310;
  const rows = [{
    label: 'Mean CI',
    width: state.meanHalfWidth,
    y: 55,
    color: GOLD
  }, {
    label: 'Next observation PI',
    width: state.predictionHalfWidth,
    y: 125,
    color: BLUE
  }];
  return <section className="hypothesis-lab" data-lab="mean-prediction" aria-label="Mean uncertainty versus prediction uncertainty">
    <h3>The mean can become precise while new requests remain variable</h3>
    <p>Hold the observed mean at 100 ms and known population SD at 10 ms. Compare two 95% procedures using the same horizontal axis.</p>
    <div className="hypothesis-controls"><Range label="Sample size for interval comparison" value={n} setValue={setN} min={5} max={100} step={5} /></div>
    <svg className="hypothesis-plot" viewBox="0 0 366 215" role="img" aria-label={`Mean interval half-width ${number(state.meanHalfWidth)} ms; prediction half-width ${number(state.predictionHalfWidth)} ms`}>
      <Marker x={scale(100)} lowY={35} highY={150} color={GREY} dashed />
      {rows.map(row => <g key={row.label}><text x="28" y={row.y - 20} fill={row.color}>{row.label}</text><line x1={scale(100 - row.width)} x2={scale(100 + row.width)} y1={row.y} y2={row.y} stroke={row.color} strokeWidth="5" /><circle cx={scale(100)} cy={row.y} r="4" fill={WHITE} /></g>)}
      <Axis scale={scale} values={[80, 100, 120]} y={175} />
    </svg>
    <p className="hypothesis-axis-label">Milliseconds; line widths are interval lengths, not probabilities along a line</p>
    <dl className="hypothesis-metrics" aria-live="polite"><Metric label="Mean interval">[{number(100 - state.meanHalfWidth)}, {number(100 + state.meanHalfWidth)}]</Metric><Metric label="Independent future observation">[{number(100 - state.predictionHalfWidth)}, {number(100 + state.predictionHalfWidth)}]</Metric></dl>
    <p>The future observation is independently drawn from the same Normal population. Its interval includes new-observation variance <em>and</em> mean-estimation variance. The prediction half-width approaches 19.600 ms, while the mean half-width approaches zero.</p>
  </section>;
}
export function NullTailLab() {
  const [shift, setShift] = useState(0);
  const [spread, setSpread] = useState(1);
  const [reference, setReference] = useState(0);
  const [alpha, setAlpha] = useState(0.05);
  const [alternative, setAlternative] = useState('two-sided');
  const state = pairedTailState({
    shift,
    spread,
    reference,
    alpha,
    alternative
  });
  const limit = Math.max(4, Math.ceil(Math.abs(state.t) + 0.5), Math.ceil(state.critical + 0.5));
  const scaleX = t => 28 + (t + limit) / (2 * limit) * 310;
  const scaleY = density => 190 - density / 0.375 * 140;
  const regions = alternative === 'two-sided' ? [[-limit, -Math.abs(state.t)], [Math.abs(state.t), limit]] : alternative === 'greater' ? [[state.t, limit]] : [[-limit, state.t]];
  const reset = () => {
    setShift(0);
    setSpread(1);
    setReference(0);
    setAlpha(0.05);
    setAlternative('two-sided');
  };
  return <section className="hypothesis-lab" data-lab="null-tail" aria-label="Null-tail p-value and compatible confidence interval">
    <h3>Count extreme outcomes under the null model</h3>
    <p>The curve is the t distribution with four degrees of freedom. Its horizontal coordinate is a standardized statistic, not a saving in milliseconds. The shaded area is the <strong>p-value for the selected question</strong>; it is not the probability that the null is true.</p>
    <div className="hypothesis-controls">
      <Range label="Shift every saving" value={shift} setValue={setShift} min={-2} max={3} step={0.25} unit=" ms" />
      <Range label="Multiply residual spread" value={spread} setValue={setSpread} min={0.25} max={2} step={0.25} />
      <Range label="Null mean saving" value={reference} setValue={setReference} min={-1} max={2} step={0.25} unit=" ms" />
      <label>Alternative<select aria-label="Test alternative" value={alternative} onChange={event => setAlternative(event.target.value)}><option value="two-sided">Mean ≠ reference</option><option value="greater">Mean &gt; reference</option><option value="less">Mean &lt; reference</option></select></label>
      <label>Predeclared α<select aria-label="Test significance level" value={alpha} onChange={event => setAlpha(Number(event.target.value))}><option value="0.01">0.01</option><option value="0.05">0.05</option><option value="0.1">0.10</option></select></label>
      <button type="button" onClick={reset}>Reset paired test</button>
    </div>
    <svg className="hypothesis-plot" viewBox="0 0 366 235" role="img" aria-label={`${alternative} t4 test: observed t ${number(state.t)}, p ${number(state.p, 6)}`}>
      {regions.map(([low, high], index) => <path key={index} d={areaPath(t4Density, low, high, scaleX, scaleY, 190)} fill={GOLD} fillOpacity="0.45" />)}
      <path d={curvePath(t4Density, -limit, limit, scaleX, scaleY)} fill="none" stroke={WHITE} strokeWidth="2" />
      <Marker x={scaleX(state.t)} lowY={25} highY={190} color={PINK} />
      {alternative !== 'less' && <Marker x={scaleX(state.critical)} lowY={45} highY={190} color={BLUE} dashed />}
      {alternative !== 'greater' && <Marker x={scaleX(-state.critical)} lowY={45} highY={190} color={BLUE} dashed />}
      <Axis scale={scaleX} values={[-limit, -limit / 2, 0, limit / 2, limit]} />
    </svg>
    <div className="hypothesis-legend"><span className="hypothesis-gold">Area: p-value</span><span className="hypothesis-pink">Solid line: observed t</span><span className="hypothesis-blue">Dashed: α critical boundary</span></div>
    <dl className="hypothesis-metrics" aria-live="polite"><Metric label="Changed paired savings">{state.values.map(value => number(value)).join(', ')} ms</Metric><Metric label="Standardized observation">({number(state.mean)} − {number(reference)}) / {number(state.se)} = {number(state.t)}</Metric><Metric label="p-value">{number(state.p, 6)}</Metric><Metric label={`${((1 - alpha) * 100).toFixed(0)}% ${alternative === 'two-sided' ? 'two-sided interval' : 'one-sided confidence set'}`}>[{number(state.low)}, {number(state.high)}] ms</Metric><Metric label="Compatible decision">{state.reject ? 'Reject the stated null at this α' : 'Do not reject the stated null at this α'}; reference is {state.nullInside ? 'inside' : 'outside'} the confidence set</Metric></dl>
    <p>Tail areas include the part beyond the plotted window. A change of alternative is a different analysis: choose it before seeing data. The slider creates a synthetic dataset by changing its mean or residual spread; it does not collect new evidence. The ordinary t model still requires independent Normal differences.</p>
  </section>;
}
export function PracticalEffectLab() {
  const [preset, setPreset] = useState('original');
  const [tolerance, setTolerance] = useState(0.5);
  const state = practicalEffectState(preset, tolerance);
  const scale = value => 28 + (value + 3) / 9 * 310;
  return <section className="hypothesis-lab" data-lab="practical-effect" aria-label="Statistical difference, useful saving and equivalence">
    <h3>Ask which effect claim the interval actually supports</h3>
    <div className="hypothesis-controls"><label>Paired dataset<select aria-label="Practical-effect dataset" value={preset} onChange={event => setPreset(event.target.value)}>{Object.entries(EFFECT_PRESETS).map(([key, value]) => <option key={key} value={key}>{value.label}</option>)}</select></label><Range label="Predeclared equivalence tolerance" value={tolerance} setValue={setTolerance} min={0.25} max={2} step={0.25} unit=" ms" /></div>
    <p>The zero line asks about <em>any</em> mean difference. The 1 ms line is a hypothetical minimum useful saving. The shaded band (−Δ,+Δ) is a separately chosen negligible-effect range. These thresholds come from the application.</p>
    <svg className="hypothesis-plot" viewBox="0 0 366 220" role="img" aria-label={`95% interval ${number(state.low95)} to ${number(state.high95)}; 90% interval ${number(state.low90)} to ${number(state.high90)} milliseconds`}>
      <rect x={scale(-tolerance)} y="12" width={scale(tolerance) - scale(-tolerance)} height="143" fill={GOLD} fillOpacity="0.12" />
      {[[42, 77], [112, 155]].map(([lowY, highY]) => <g key={lowY}>
        <Marker x={scale(0)} lowY={lowY} highY={highY} color={WHITE} />
        <Marker x={scale(1)} lowY={lowY} highY={highY} color={BLUE} dashed />
      </g>)}
      <text x="28" y="30" fill={GOLD}>95% two-sided</text><line x1={scale(state.low95)} x2={scale(state.high95)} y1="55" y2="55" stroke={GOLD} strokeWidth="5" /><circle cx={scale(state.mean)} cy="55" r="4" fill={WHITE} />
      <text x="28" y="100" fill={PINK}>90% two-sided</text><line x1={scale(state.low90)} x2={scale(state.high90)} y1="125" y2="125" stroke={PINK} strokeWidth="5" /><circle cx={scale(state.mean)} cy="125" r="4" fill={WHITE} />
      <Axis scale={scale} values={[-3, 0, 3, 6]} y={175} />
    </svg>
    <p className="hypothesis-axis-label">Mean saving in milliseconds; white = zero, dashed blue = 1 ms</p>
    <dl className="hypothesis-metrics" aria-live="polite"><Metric label="95% two-sided interval">[{number(state.low95)}, {number(state.high95)}] ms</Metric><Metric label="90% two-sided interval">[{number(state.low90)}, {number(state.high90)}] ms</Metric><Metric label="Different from zero? Two-sided α=.05">p={number(state.pZero, 6)}; {state.pZero < 0.05 ? 'reject zero' : 'do not reject zero'}</Metric><Metric label="Greater than 1 ms? One-sided α=.05">p={number(state.pUseful, 6)}; {state.pUseful < 0.05 ? 'supports the predeclared superiority claim' : 'insufficient for that superiority claim'}</Metric><Metric label={`Equivalent within ±${number(tolerance)} ms? TOST α=.05`}>p={number(state.equivalenceP, 6)}; {state.equivalent ? 'both boundary nulls rejected' : 'at least one boundary null not rejected'}</Metric></dl>
    <p><strong>Compare:</strong> the two centered datasets both have p=1 for the point null. Only the quarter-spread dataset establishes equivalence within ±0.5 ms. Its 90% interval fits the band, even though its 95% interval does not. TOST uses two one-sided tests at α=.05, as derived below.</p>
    <details><summary>See both equivalence tests and the one-sided bound</summary><LessonTable caption="One-sided p-values for the same five paired differences" headers={['Question', 'Computed value']} rows={[[`Mean > −${number(tolerance)} ms`, `p=${number(state.lowerP, 6)}`], [`Mean < +${number(tolerance)} ms`, `p=${number(state.upperP, 6)}`], ['95% one-sided lower bound', `${number(state.low90)} ms`], ['Complete changed dataset', state.values.map(value => number(value)).join(', ')]]} /><p>For the +1 ms shift, the 95% two-sided interval contains values below 1 ms, but the 95% one-sided lower bound exceeds 1 ms. Those answer different predeclared questions; comparing incompatible confidence levels would obscure that distinction.</p></details>
  </section>;
}
export function PlannedPowerLab() {
  const [n, setN] = useState(25);
  const [sigma, setSigma] = useState(4);
  const [effect, setEffect] = useState(2);
  const [alpha, setAlpha] = useState(0.05);
  const [alternative, setAlternative] = useState('greater');
  const state = plannedPowerState(n, sigma, effect, alpha, alternative);
  const low = -4 * state.se,
    high = effect + 4 * state.se;
  const scale = value => 28 + (value - low) / (high - low) * 310;
  const nullDensity = value => normalDensity(value / state.se);
  const altDensity = value => normalDensity((value - effect) / state.se);
  const rows = [{
    label: 'Null: true mean 0',
    fn: nullDensity,
    base: 105,
    color: PINK
  }, {
    label: `Alternative: true mean ${number(effect)}`,
    fn: altDensity,
    base: 240,
    color: GOLD
  }];
  const regions = alternative === 'two-sided' ? [[low, -state.boundary], [state.boundary, high]] : [[state.boundary, high]];
  return <section className="hypothesis-lab" data-lab="planned-power" aria-label="Prospective power under a known-variance normal model">
    <h3>Move the true effect and see how often a planned test detects it</h3>
    <p>This is a separate, deliberately simple planning model: independent Normal savings with a <strong>known</strong> population SD. Both curves show the sampling distribution of the mean. The rejection region is the same for both.</p>
    <div className="hypothesis-controls"><Range label="Planned independent observations" value={n} setValue={setN} min={5} max={200} step={5} /><Range label="Known population SD" value={sigma} setValue={setSigma} min={2} max={10} step={0.5} unit=" ms" /><Range label="Hypothetical true saving" value={effect} setValue={setEffect} min={0} max={4} step={0.25} unit=" ms" /><label>Planned direction<select aria-label="Power test direction" value={alternative} onChange={event => setAlternative(event.target.value)}><option value="greater">Greater than zero</option><option value="two-sided">Different from zero</option></select></label><label>Predeclared α<select aria-label="Power significance level" value={alpha} onChange={event => setAlpha(Number(event.target.value))}><option value="0.01">0.01</option><option value="0.05">0.05</option><option value="0.1">0.10</option></select></label></div>
    <svg className="hypothesis-plot" viewBox="0 0 366 305" role="img" aria-label={`Null rejection chance ${percent(alpha)}; detection chance ${percent(state.power)} at true saving ${effect} ms`}>
      {rows.map(row => {
        const scaleY = density => row.base - density / normalDensity(0) * 65;
        return <g key={row.label}><text x="28" y={row.base - 80} fill={row.color}>{row.label}</text>{regions.map(([start, end], index) => <path key={index} d={areaPath(row.fn, start, end, scale, scaleY, row.base)} fill={row.color} fillOpacity="0.5" />)}<path d={curvePath(row.fn, low, high, scale, scaleY)} fill="none" stroke={row.color} strokeWidth="2" /><line x1="28" x2="338" y1={row.base} y2={row.base} stroke={GREY} /></g>;
      })}
      {rows.map(row => <g key={row.label}>
        <Marker x={scale(state.boundary)} lowY={row.base - 68} highY={row.base} color={WHITE} dashed />
        {alternative === 'two-sided' && <Marker x={scale(-state.boundary)} lowY={row.base - 68} highY={row.base} color={WHITE} dashed />}
      </g>)}
      <Axis scale={scale} values={[low, (low + high) / 2, high]} y={263} />
    </svg>
    <p className="hypothesis-axis-label">Sample mean saving, milliseconds; dashed boundary is the planned rejection cutoff</p>
    <dl className="hypothesis-metrics" aria-live="polite"><Metric label="Type I error under mean 0">α={percent(alpha)}</Metric><Metric label={`Power under mean ${number(effect)} ms`}>{percent(state.power)}</Metric><Metric label="Type II error for this alternative">β={percent(state.beta)}</Metric><Metric label="Upper rejection cutoff">Sample mean &gt; {number(state.boundary)} ms{alternative === 'two-sided' ? ` or < ${number(-state.boundary)} ms` : ''}</Metric></dl>
    <p>Shaded probability under the null is a false-alarm chance; shaded probability under the alternative is power. The unshaded alternative area is β. At a true saving of zero, the two curves coincide and rejection probability is α; there is then no nonzero effect to detect. Areas include tails outside the displayed window.</p>
  </section>;
}
export function MultipleChancesLab() {
  const [count, setCount] = useState(20);
  const [looks, setLooks] = useState(12);
  const family = familyErrorState(count);
  const sequential = optionalLooksState(looks);
  return <section className="hypothesis-lab" data-lab="multiple-chances" aria-label="Multiple tests and repeated looks use different probability models">
    <h3>More opportunities to reject form a different procedure</h3>
    <div className="hypothesis-controls"><Range label="Independent true-null tests" value={count} setValue={setCount} min={1} max={100} /><Range label="Maximum fair-coin looks" value={looks} setValue={setLooks} min={1} max={12} /></div>
    <div className="hypothesis-comparisons"><div><h4>Separate independent tests</h4><p>Each test has continuous-null false-alarm probability .05. The chance of at least one false alarm in {count} tests is <strong>{percent(family.independentFamilyError)}</strong>.</p><div className="hypothesis-probability-bar"><span style={{
            width: `${100 * family.independentFamilyError}%`
          }} /></div><p>Bonferroni gives each test α/{count}={number(family.bonferroniPerTest, 6)} to bound the whole family's error by .05, even without independence.</p></div><div><h4>Repeated looks at one growing sample</h4><p>Flip an independent fair coin, inspect after every flip up to {looks}, and stop at the first two-sided binomial p&lt;.05. The exact false-alarm chance is <strong>{percent(sequential.cumulativeError)}</strong>.</p><p>Testing only at the fixed final sample size gives <strong>{percent(sequential.fixedError)}</strong>. These looks share observations, so the independent-tests formula does not apply.</p></div></div>
    <LessonTable caption="Exact fair-coin probabilities; every length-n sequence has probability 2⁻ⁿ" headers={['Look n', 'Only this fixed look', 'Any first rejection by here']} rows={sequential.rows.map(row => [row.n, percent(row.fixedError), percent(row.cumulativeError)])} />
    <p>Discrete tests can be conservative: a fixed-look rejection chance need not equal .05 and can move irregularly with n. The cumulative first-rejection chance cannot decrease. This is exact finite counting, not a simulation; the complete program below enumerates all 4,096 length-12 paths independently.</p>
  </section>;
}
export function IndependentUnitsFigure() {
  const state = clusterPrecisionState();
  return <figure className="hypothesis-inline" aria-label="Ten thousand requests grouped within five independent users">
    <figcaption>Count the independent units, not just the rows</figcaption>
    <div className="hypothesis-unit-grid">{Array.from({
        length: 5
      }, (_, index) => <div key={index}><strong>User {index + 1}</strong><span>2,000 requests</span><small>Shared user effect</small></div>)}</div>
    <p>In the equal-size random-intercept model below, within-user correlation 0.2 gives a mean-variance inflation of <strong>{number(state.designEffect)}</strong> and an SE multiplier of <strong>{number(state.seRatio)}</strong>. The 10,000 rows provide the same mean precision as about <strong>{number(state.effectiveN, 2)}</strong> independent rows of the same marginal variance—not 10,000.</p>
    <p>This effective-size comparison belongs to this specified variance model. It does not turn five users into 25 actual independent users, or supply a universal t degrees-of-freedom rule.</p>
  </figure>;
}
export function SignFlipLab() {
  const [preset, setPreset] = useState('original');
  const [mask, setMask] = useState(0);
  const state = signFlipState(preset, mask);
  const maximumCount = Math.max(...state.histogram.map(row => row.count));
  const scale = value => 28 + (value + 3) / 6 * 310;
  const barWidth = Math.min(18, 260 / state.histogram.length);
  return <section className="hypothesis-lab" data-lab="sign-flip" aria-label="Exact paired sign-flip null distribution">
    <h3>Build a null distribution from justified relabelings</h3>
    <p>Under independent sign symmetry about zero, each of the 32 sign assignments is equally likely conditional on the magnitudes. A randomized paired experiment can instead justify swaps under an appropriate sharp no-effect null. Merely assuming a zero population mean is insufficient.</p>
    <div className="hypothesis-controls"><label>Five paired differences<select aria-label="Sign-flip dataset" value={preset} onChange={event => {
          setPreset(event.target.value);
          setMask(0);
        }}><option value="original">Original: 2, 4, −1, 3, 2</option><option value="allPositive">Five positive ones</option><option value="centered">Centered: 0, 2, −3, 1, 0</option></select></label><Range label="Sign assignment index" value={mask} setValue={setMask} min={0} max={31} /></div>
    <svg className="hypothesis-plot" viewBox="0 0 366 230" role="img" aria-label={`${state.extremeCount} of 32 sign assignments are at least as extreme as observed mean ${number(state.observedMean)} ms`}>
      {state.histogram.map(row => <rect key={row.sum} x={scale(row.mean) - barWidth / 2} y={180 - row.count / maximumCount * 115} width={barWidth} height={row.count / maximumCount * 115} fill={row.extreme ? GOLD : GREY} />)}
      <Marker x={scale(state.current.mean)} lowY={30} highY={180} color={PINK} />
      <text x="28" y="22" fill={WHITE}>Height = assignment count</text>
      <Axis scale={scale} values={[-3, 0, 3]} y={185} />
    </svg>
    <p className="hypothesis-axis-label">Mean after sign assignment, milliseconds</p>
    <div className="hypothesis-legend"><span className="hypothesis-gold">At least as far from zero as observed</span><span className="hypothesis-pink">Current assignment</span></div>
    <dl className="hypothesis-metrics" aria-live="polite"><Metric label="Current signs">{state.current.signs.map(value => value === 1 ? '+' : '−').join(' ')}</Metric><Metric label="Transformed differences">{state.current.transformed.map(value => number(value)).join(', ')} ms</Metric><Metric label="Current mean">{number(state.current.mean)} ms</Metric><Metric label="Observed mean">{number(state.observedMean)} ms</Metric><Metric label="Exact two-sided p-value">{state.extremeCount}/32 = {number(state.p, 6)}</Metric></dl>
    <p>Select five positive ones: the best possible two-sided sign-flip p for five nonzero pairs is still 2/32=.0625. All-plus and all-minus tie for greatest absolute mean. There is no continuous tail to interpolate between those assignments.</p>
    <details><summary>Inspect assignment counts and ties</summary><LessonTable caption="Different assignments can have the same statistic; all still count" headers={['Mean in ms', 'Assignments', 'Extreme?']} rows={state.histogram.map(row => [number(row.mean), row.count, row.extreme ? 'Yes' : 'No'])} /><p>A zero difference has two sign labels with the same value. Counting both preserves their conditional probabilities; deleting duplicate statistics and then treating the unique values as equally likely would change the test.</p></details>
  </section>;
}
export function ProportionCoverageLab() {
  const [n, setN] = useState(10);
  const [observed, setObserved] = useState(0);
  const [truth, setTruth] = useState(0.2);
  const state = proportionCoverageState(n, observed, truth);
  const scale = value => 28 + (value + 0.3) / 1.6 * 310;
  const methods = [{
    key: 'wald',
    label: 'Plug-in Wald',
    color: PINK
  }, {
    key: 'wilson',
    label: 'Score / Wilson',
    color: GOLD
  }, {
    key: 'exact',
    label: 'Equal-tailed exact',
    color: BLUE
  }];
  return <section className="hypothesis-lab" data-lab="proportion-coverage" aria-label="Binomial boundary intervals and exact finite coverage">
    <h3>Zero observed successes does not imply zero uncertainty</h3>
    <p>Start with 0 successes out of 10 independent Bernoulli trials. Compare three 95% interval rules. Then specify a hypothetical true probability and count the probability of <em>every</em> possible success count to measure each rule's coverage.</p>
    <div className="hypothesis-controls"><Range label="Independent Bernoulli trials" value={n} setValue={value => {
        setN(value);
        setObserved(current => Math.min(current, value));
      }} min={5} max={20} step={5} /><Range label="Observed successes" value={observed} setValue={setObserved} min={0} max={n} /><Range label="Hypothetical true success probability" value={truth} setValue={setTruth} min={0} max={1} step={0.01} /></div>
    <svg className="hypothesis-plot" viewBox="0 0 366 295" role="img" aria-label={`Intervals for ${observed} successes in ${n} trials; true probability ${truth} is specified only to assess coverage`}>
      {methods.map((method, index) => <Marker key={method.key} x={scale(truth)} lowY={46 + 76 * index} highY={76 + 76 * index} color={WHITE} dashed />)}
      {methods.map((method, index) => {
        const y = 58 + 76 * index;
        const [low, high] = state.current.intervals[method.key];
        return <g key={method.key}><text x="28" y={y - 25} fill={method.color}>{method.label}</text><line x1={scale(low)} x2={scale(high)} y1={y} y2={y} stroke={method.color} strokeWidth="5" /><circle cx={scale(observed / n)} cy={y} r="3.5" fill={WHITE} /></g>;
      })}
      <Axis scale={scale} values={[0, 0.5, 1]} y={251} />
    </svg>
    <p className="hypothesis-axis-label">Success probability; dashed white is hypothetical truth, dots are the observed estimate</p>
    <LessonTable caption={`Observed ${observed}/${n}, but coverage averages over all counts at true p=${number(truth)}`} headers={['95% method', 'Observed interval', 'Actual finite coverage']} rows={methods.map(method => [method.label, `[${number(state.current.intervals[method.key][0], 5)}, ${number(state.current.intervals[method.key][1], 5)}]`, percent(state.coverage[method.key])])} />
    <p><strong>Keep the questions separate:</strong> changing the observed success count changes the displayed interval, but it does not change a procedure's coverage at fixed n and true p. Changing hypothetical p changes coverage and the reference line, but does not change intervals computed from the fixed observed data.</p>
    <p>Wilson uses a normal score approximation and can under-cover. Equal-tailed Clopper–Pearson inverts exact binomial tails and has coverage at least 95% under this sampling model; “exact” does not mean coverage equals exactly 95%. Neither interval is a posterior probability statement.</p>
    <details><summary>Inspect the complete finite coverage calculation</summary><p>For each possible k, add P(K=k) if that method's interval contains the stated true p; otherwise add zero. Here K follows Binomial(n,p), so no simulation is necessary.</p><LessonTable caption="All possible data counts; interval endpoints are computed, not simulated" headers={['k', 'Probability', 'Wald covers?', 'Wilson covers?', 'Exact covers?']} rows={state.rows.map(row => [row.k, percent(row.mass), ...methods.map(method => row.covers[method.key] ? 'Yes' : 'No')])} /></details>
  </section>;
}
