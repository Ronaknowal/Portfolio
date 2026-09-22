import { useId, useMemo, useState } from 'react';
import { brownianIncrements, groupIncrements, integralTrace, growthLaw, growthPath, ouLaw, gbmErrorMoments, sampledErrors } from '../../data/ito-sde-models.js';
import './ito-sde-labs.css';
const display = value => value === null ? 'not defined' : value === 0 ? '0' : Math.abs(value) < 0.0001 || Math.abs(value) >= 10000 ? value.toExponential(3) : Number(value.toFixed(5)).toString();
const colors = {
  amber: '#f5c366',
  green: '#77dfb2',
  blue: '#90c5ff',
  rose: '#eea5bd'
};
function Choice({
  label,
  value,
  options,
  change,
  numeric = true,
  disabled = false
}) {
  return <label className="ito-control">{label}<select aria-label={label} value={value} disabled={disabled} onChange={event => change(numeric ? Number(event.target.value) : event.target.value)}>
    {options.map(([key, name]) => <option key={key} value={key}>{name}</option>)}
  </select></label>;
}
function Slider({
  label,
  value,
  min = 0,
  max,
  step = 1,
  change
}) {
  return <label className="ito-control">{label}: <strong>{value}</strong>
    <input type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={event => change(Number(event.target.value))} />
  </label>;
}
function SeedControl({
  active,
  change,
  label
}) {
  const [draft, setDraft] = useState(String(active));
  const [error, setError] = useState('');
  return <form className="ito-seed" onSubmit={event => {
    event.preventDefault();
    const candidate = Number(draft);
    if (!/^\d+$/.test(draft.trim()) || !Number.isInteger(candidate) || candidate < 1 || candidate > 2147483647) {
      setError('Use a whole seed from 1 through 2147483647. The active path is unchanged.');
      return;
    }
    setError('');
    change(candidate);
  }}><label>{label}<input aria-label={label} inputMode="numeric" value={draft} onChange={event => setDraft(event.target.value)} /></label>
    <button type="submit">Apply seed</button><span>Active seed: {active}</span>
    {error && <p role="alert">{error}</p>}
  </form>;
}
function Metrics({
  rows
}) {
  return <dl className="ito-metrics">{rows.map(([name, value]) => <div key={name}>
    <dt>{name}</dt><dd>{typeof value === 'number' ? display(value) : value}</dd>
  </div>)}</dl>;
}
function Plot({
  label,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  series = [],
  children,
  xTicks,
  xTick = value => Number(value.toPrecision(3)).toString(),
  square = false,
  compact = false
}) {
  const id = useId();
  const height = compact ? 290 : square ? 480 : 310;
  const width = compact ? 360 : 505;
  const left = compact ? 70 : 66;
  const right = compact ? 330 : 480;
  const top = 35;
  const bottom = height - 56;
  const x = value => left + (value - xDomain[0]) / (xDomain[1] - xDomain[0]) * (right - left);
  const y = value => bottom - (value - yDomain[0]) / (yDomain[1] - yDomain[0]) * (bottom - top);
  const ticks = xTicks ?? Array.from({
    length: compact ? 3 : 5
  }, (_, i) => xDomain[0] + i / (compact ? 2 : 4) * (xDomain[1] - xDomain[0]));
  const path = points => {
    let connected = false;
    return points.map(point => {
      if (!point || !point.every(Number.isFinite)) {
        connected = false;
        return '';
      }
      const command = connected ? 'L' : 'M';
      connected = true;
      return command + x(point[0]).toFixed(3) + ',' + y(point[1]).toFixed(3);
    }).join(' ');
  };
  return <><p className="ito-axis-label">Vertical axis: {yLabel}</p>{!compact && <p className="ito-scroll-hint">Scroll this plot sideways for the full axes.</p>}
    <div className={'ito-plot' + (compact ? ' compact' : '')} role="region" aria-label={label + (compact ? '' : '; horizontally scrollable')} tabIndex={0}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
        <defs><clipPath id={id}><rect x={left} y={top} width={right - left} height={bottom - top} /></clipPath></defs>
        {ticks.map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1={top} y2={bottom} />
          <text x={x(value)} y={bottom + 24} textAnchor={value === xDomain[0] ? 'start' : value === xDomain[1] ? 'end' : 'middle'}>{xTick(value)}</text></g>)}
        {Array.from({
          length: 5
        }, (_, i) => {
          const value = yDomain[0] + i / 4 * (yDomain[1] - yDomain[0]);
          return <g key={i}><line x1={left} x2={right} y1={y(value)} y2={y(value)} />
            <text x={left - 8} y={y(value) + 5} textAnchor="end">{Number(value.toPrecision(3))}</text></g>;
        })}
        <g clipPath={`url(#${id})`}>{series.map((item, index) => <path key={index} d={path(item.points)} fill="none" stroke={colors[item.color ?? 'amber']} strokeWidth={2.3} strokeDasharray={item.dashed ? '6 4' : undefined} />)}{children?.({
            x,
            y,
            top,
            bottom
          })}</g>
      </svg>
    </div><p className="ito-axis-label">Horizontal axis: {xLabel}</p></>;
}
export function NoiseScalingFigure() {
  return <figure className="ito-figure"><div className="ito-scaling">
    {[[1, 'One interval'], [0.25, 'One quarter interval']].map(([step, title]) => <div key={step}>
      <strong>{title}</strong><span>h={step}</span>
      <div className="ito-scale-bar"><i style={{
            width: `${step * 100}%`
          }} /></div>
      <span>Drift magnitude: |a| h → {step} |a|</span>
      <div className="ito-scale-bar noise"><i style={{
            width: `${Math.sqrt(step) * 100}%`
          }} /></div>
      <span>Noise SD: |b| √h → {Math.sqrt(step)} |b|</span>
    </div>)}
  </div><figcaption>Amber compares deterministic drift; blue compares noise standard deviation. Quartering time quarters the drift but only halves the noise SD. The bars compare scales, not realized random increments. Four independent quarter-interval increments recover the full-interval variance.</figcaption></figure>;
}
export function CurvatureCorrectionFigure() {
  const values = Array.from({
    length: 81
  }, (_, index) => 0.25 + index / 80 * 1.5);
  return <figure className="ito-figure"><Plot compact label="A quadratic curve differs from its tangent under symmetric perturbations" xLabel="state w" yLabel="curve and tangent" xDomain={[0.25, 1.75]} yDomain={[-0.5, 3.1]} series={[{
      points: values.map(w => [w, w * w]),
      color: 'green'
    }, {
      points: values.map(w => [w, 2 * w - 1]),
      color: 'blue',
      dashed: true
    }]}>
    {({
        x,
        y
      }) => <>{[0.5, 1.5].map(w => <g key={w}>
      <line x1={x(w)} x2={x(w)} y1={y(2 * w - 1)} y2={y(w * w)} className="ito-correction" />
      <circle cx={x(w)} cy={y(w * w)} r={5} fill={colors.amber} />
    </g>)}<circle cx={x(1)} cy={y(1)} r={5} fill={colors.blue} /></>}
  </Plot><div className="ito-value-pair"><span>At w=0.5: tangent=0, curve=0.25</span>
    <span>At w=1.5: tangent=2, curve=2.25</span></div>
    <figcaption>Green: f(w)=w². Blue dashed: its tangent at w=1. Both ±0.5 changes from w=1 add a positive curvature correction 0.25. Their tangent changes cancel in an equal-weight average; their curvature changes do not. These are exact finite values, motivating the subsequent stochastic limit.</figcaption></figure>;
}
export function ConventionConversionFigure() {
  return <figure className="ito-figure"><div className="ito-conventions">
    <div><strong>Written as Itô</strong><span>dX=aX dt+σX dW</span><span>log drift: a−σ²/2</span><span>mean: X₀ exp(a t)</span></div>
    <div><strong>Written as Stratonovich</strong><span>dX=aX dt+σX ∘dW</span><span>log drift: a</span><span>mean: X₀ exp((a+σ²/2)t)</span></div>
    <div><strong>Equivalent Itô model</strong><span>dX=(a+σ²/2)X dt+σX dW</span><span>log drift: a</span><span>same law as the middle lane</span></div>
  </div><figcaption>Positive initial state and constant a,σ. The circle marks the Stratonovich convention. The last two equations describe the same model after converting the drift; the first generally does not. This is a comparison of exact model laws, not three empirical fits.</figcaption></figure>;
}
export function ProbabilityDualityFigure() {
  return <figure className="ito-figure"><div className="ito-duality">
    <div><strong>Move a population forward</strong><span>initial law p₀ → law pₜ</span><span>∂ₜp=−∂ₓ(a p)+½∂ₓₓ(b²p)</span><span>Derivatives act on coefficient × density.</span></div>
    <div><strong>Move a question backward</strong><span>terminal reward g → current value u</span><span>∂ₜu+a∂ₓu+½b²∂ₓₓu=0</span><span>Coefficients multiply derivatives of u.</span></div>
  </div><figcaption>The same SDE yields two equations with different jobs. Here u(t,x)=E[g(X_T) | X_t=x]. The displayed classical equations require the smoothness and boundary conditions explained nearby; no numerical PDE solution is implied.</figcaption></figure>;
}
export function AdaptedIntegralLab() {
  const [source, setSource] = useState('hand');
  const [resetId, setResetId] = useState(0);
  const [seed, setSeed] = useState(5);
  const [group, setGroup] = useState(16);
  const [step, setStep] = useState(0);
  const [choice, setChoice] = useState('left');
  const fine = useMemo(() => brownianIncrements({
    seed,
    steps: 256
  }), [seed]);
  const state = useMemo(() => integralTrace(source === 'hand' ? [0.5, -0.25, 0.75, -0.5] : groupIncrements(fine, group)), [source, fine, group]);
  const selected = state.rows[step];
  const coefficient = choice === 'left' ? selected.before : choice === 'right' ? selected.after : (selected.before + selected.after) / 2;
  const reset = () => {
    setResetId(value => value + 1);
    setSource('hand');
    setSeed(5);
    setGroup(16);
    setStep(0);
    setChoice('left');
  };
  return <section className="ito-lab" aria-label="Adapted integral investigation">
    <h3>Choose what you know before the increment arrives</h3>
    <p>Inspect the selected contribution before inspecting the totals. Left, right and symmetric sums use the same observations; only the evaluation rule changes.</p>
    <div className="ito-controls"><Choice label="Increment source" numeric={false} value={source} options={[['hand', 'Four fixed increments'], ['brownian', 'Seeded Brownian grid']]} change={value => {
        setSource(value);
        setStep(0);
      }} />
      <Choice label="Coefficient choice" numeric={false} value={choice} change={setChoice} options={[['left', 'Before the increment'], ['right', 'After the increment'], ['symmetric', 'Average of endpoints']]} />
      {source === 'brownian' && <Choice label="Observed intervals" value={group} options={[[64, '4'], [16, '16'], [4, '64'], [1, '256']]} change={value => {
        setGroup(value);
        setStep(0);
      }} />}
    </div>
    {source === 'brownian' && <SeedControl key={'integral-' + seed + '-' + resetId} label="Integral seed" active={seed} change={value => {
      setSeed(value);
      setStep(0);
    }} />}
    <div className="ito-actions"><button disabled={step === 0} onClick={() => setStep(value => value - 1)}>Previous interval</button>
      <button disabled={step === state.rows.length - 1} onClick={() => setStep(value => value + 1)}>Next interval</button></div>
    <div className="ito-information"><div><span>Known at the left endpoint</span><strong>W={display(selected.before)}</strong></div>
      <div className="ito-increment"><span>Then a new increment arrives</span><strong>ΔW={display(selected.increment)}</strong></div>
      <div><span>Known only afterward</span><strong>W={display(selected.after)}</strong></div></div>
    <p className="ito-readout">Selected interval {step + 1}/{state.rows.length}: {display(coefficient)} × {display(selected.increment)} = <strong>{display(coefficient * selected.increment)}</strong>.</p>
    <p>{choice === 'left' ? 'This coefficient is available before the increment. Brownian independence then gives conditional mean zero.' : choice === 'right' ? 'This coefficient already contains the increment. Treating it as information known beforehand would introduce look-ahead.' : 'This symmetric endpoint average is a different convention. It is not an adapted coefficient chosen before the increment.'}</p>
    <div aria-live="polite"><Metrics rows={[['Left sum', state.left], ['Right sum', state.right], ['Symmetric sum', state.symmetric], ['Q = sum of squared increments', state.quadratic], ['Terminal W', state.terminal], ['½(W²−T), T=1', state.itoLimitAtTerminal]]} /></div>
    <div className="ito-identity">2 × left sum + Q = W² → {display(2 * state.left)} + {display(state.quadratic)} = {display(state.terminal ** 2)}</div>
    <details><summary>Inspect every interval and cumulative contribution</summary><div className="ito-table" tabIndex={0}>
      <table><thead><tr><th>Interval</th><th>Before</th><th>ΔW</th><th>Left contribution</th><th>Left total</th><th>Q</th></tr></thead>
        <tbody>{state.rows.map(row => <tr key={row.index} className={step === row.index ? 'selected' : ''}>
          <td>{row.index + 1}</td><td>{display(row.before)}</td><td>{display(row.increment)}</td><td>{display(row.leftContribution)}</td><td>{display(row.left)}</td><td>{display(row.quadratic)}</td>
        </tr>)}</tbody></table></div></details>
    <p>{source === 'hand' ? 'The four increments are an exact hand fixture, not a statistical sample.' : 'The 256 finest increments are seeded synthetic normal draws. Coarser observations sum the same increments, so the terminal value is retained; Q need not improve monotonically on one path.'}
      {' '}The identities are exact algebra, evaluated in binary64. The limit target uses Brownian Q→T; this finite sample does not prove that theorem.</p>
    <button onClick={reset}>Reset integral</button>
  </section>;
}
export function GrowthLawLab() {
  const [mu, setMu] = useState(0.4);
  const [sigma, setSigma] = useState(0.3);
  const [index, setIndex] = useState(64);
  const [resetId, setResetId] = useState(0);
  const [seed, setSeed] = useState(5);
  const fine = useMemo(() => brownianIncrements({
    seed,
    steps: 256,
    horizon: 4
  }), [seed]);
  const trace = useMemo(() => growthPath({
    mu,
    sigma,
    horizon: 4,
    increments: fine
  }), [mu, sigma, fine]);
  const time = index / 64;
  const law = growthLaw({
    mu,
    sigma,
    horizon: time
  });
  const selected = trace.rows[index];
  const spread = Math.sqrt(law.logVariance);
  const low = law.logMean - Math.max(0.5, 3.5 * spread);
  const high = law.logMean + Math.max(0.5, 3.5 * spread);
  const density = law.atom ? [] : Array.from({
    length: 181
  }, (_, i) => {
    const logX = low + (high - low) * i / 180;
    return [logX, Math.exp(-((logX - law.logMean) ** 2) / (2 * law.logVariance)) / Math.sqrt(2 * Math.PI * law.logVariance)];
  });
  const pathValues = trace.rows.slice(0, index + 1);
  const lower = Math.min(-1, ...pathValues.map(row => row.logExact));
  const upper = Math.max(1, ...pathValues.map(row => row.logExact));
  return <section className="ito-lab" aria-label="Growth law investigation"><h3>One path, one median and one mean</h3>
    <p>Keep the Brownian driver fixed and change the growth or noise coefficient. Inspect whether the median and mean move together. The elapsed-time cursor reveals the same already-generated path.</p>
    <div className="ito-controls"><Choice label="Growth drift mu" value={mu} change={setMu} options={[[-0.5, 'μ=−0.5'], [0.2, 'μ=0.2'], [0.4, 'μ=0.4'], [1, 'μ=1']]} />
      <Choice label="Growth noise sigma" value={sigma} change={setSigma} options={[[0, 'σ=0'], [0.3, 'σ=0.3'], [0.6, 'σ=0.6'], [1, 'σ=1']]} />
      <Slider label="Growth time index (64 per unit time)" value={index} max={256} step={4} change={setIndex} /></div>
    <SeedControl key={'growth-' + seed + '-' + resetId} label="Growth seed" active={seed} change={setSeed} />
    <Plot label="The selected exact sampled GBM path in log coordinates" xLabel="time t" yLabel="log X (natural log)" xDomain={[0, 4]} yDomain={[lower - 0.2, upper + 0.2]} series={[{
      points: pathValues.map(row => [row.time, row.logExact])
    }]} />
    <Plot compact label="Analytic terminal distribution in log coordinates" xLabel="log X at selected time" yLabel={law.atom ? 'point mass (probability 1)' : 'density of log X'} xDomain={[low, high]} yDomain={[0, law.atom ? 1.1 : 1.1 / Math.sqrt(2 * Math.PI * law.logVariance)]} series={[{
      points: density,
      color: 'green'
    }]}>
      {({
        x,
        y,
        top,
        bottom
      }) => law.atom ? <line x1={x(law.logMean)} x2={x(law.logMean)} y1={y(0)} y2={y(1)} className="ito-correction" /> : <>
        <line x1={x(law.logMean)} x2={x(law.logMean)} y1={top} y2={bottom} className="ito-median-marker" />
        <line x1={x(Math.log(law.mean))} x2={x(Math.log(law.mean))} y1={top} y2={bottom} className="ito-mean-marker" />
      </>}
    </Plot>
    <div className="ito-mean-median"><div><span>Median</span><strong>{display(law.median)}</strong><small>half the terminal law lies on either side when σ²t&gt;0</small></div>
      <div><span>Mean</span><strong>{display(law.mean)}</strong><small>population average; larger upper-tail outcomes receive their actual weight</small></div></div>
    <div aria-live="polite"><Metrics rows={[['Selected time', time], ['Selected path X', selected.exact], ['5th / 95th percentiles', `${display(law.lower)} / ${display(law.upper)}`], ['Variance', law.variance], ['Long-run log rate', law.almostSureLogRate], ['Second-moment rate', law.secondMomentRate]]} /></div>
    <p>The green curve is an analytic normal density for log X, not a histogram of simulations. The amber vertical marker is log(median X); the blue dashed marker is log(E[X]), which differs from E[log X]. It shows at least ±3.5 log standard deviations; axes adapt to the selected law. The amber path line joins exact GBM samples at 1/64 time spacing, not the full continuous path. The percentiles describe one chosen time, not simultaneous path coverage.</p>
    <p>Try μ=0.2 and σ=1: the long-run log rate is negative while the mean growth rate is positive. A small set of large outcomes can raise the mean even as the median falls.</p>
    <button onClick={() => {
      setResetId(value => value + 1);
      setMu(0.4);
      setSigma(0.3);
      setIndex(64);
      setSeed(5);
    }}>Reset growth law</button>
  </section>;
}
export function OUResponseLab() {
  const [theta, setTheta] = useState(1);
  const [eta, setEta] = useState(0.8);
  const [time, setTime] = useState(1);
  const [initialMode, setInitialMode] = useState('fixed');
  const stationary = initialMode === 'stationary' && theta > 0;
  const initialVariance = stationary ? eta * eta / (2 * theta) : 0;
  const initial = stationary ? 0 : 1.5;
  const law = ouLaw({
    theta,
    eta,
    time,
    initial,
    initialVariance
  });
  const extent = Math.max(2, Math.abs(law.mean) + 3.5 * law.standardDeviation);
  const density = law.atom ? [] : Array.from({
    length: 181
  }, (_, i) => {
    const value = -extent + 2 * extent * i / 180;
    return [value, Math.exp(-((value - law.mean) ** 2) / (2 * law.variance)) / Math.sqrt(2 * Math.PI * law.variance)];
  });
  const budgetScale = Math.max(law.varianceInjection, law.varianceRemoval, 0.1);
  return <section className="ito-lab" aria-label="OU restoring flow investigation"><h3>Restoring drift narrows; noise replenishes</h3>
    <p>Inspect whether the distribution is already stationary or still changing. This model restores toward zero: dX=−θX dt+η dW. The variance budget explains what its probability distribution does.</p>
    <div className="ito-controls"><Choice label="OU reversion theta" value={theta} options={[[0, 'θ=0'], [0.3, 'θ=0.3'], [1, 'θ=1'], [2, 'θ=2']]} change={value => {
        setTheta(value);
        if (value === 0) setInitialMode('fixed');
      }} />
      <Choice label="OU noise eta" value={eta} change={setEta} options={[[0, 'η=0'], [0.4, 'η=0.4'], [0.8, 'η=0.8']]} />
      <Choice label="OU initial law" numeric={false} value={initialMode} change={setInitialMode} options={theta > 0 ? [['fixed', 'Fixed X₀=1.5'], ['stationary', 'Stationary law']] : [['fixed', 'Fixed X₀=1.5']]} />
      <Slider label="OU elapsed time" value={time} max={4} step={0.125} change={setTime} /></div>
    <div className="ito-restoring"><span>x&lt;0</span><strong>{theta > 0 ? '→' : '·'}</strong><span>target 0</span><strong>{theta > 0 ? '←' : '·'}</strong><span>x&gt;0</span></div>
    <Plot compact label="Analytic OU marginal law" xLabel="state x" yLabel={law.atom ? 'point mass (probability 1)' : 'density p(t,x)'} xDomain={[-extent, extent]} yDomain={[0, law.atom ? 1.1 : 1.1 / Math.sqrt(2 * Math.PI * law.variance)]} series={[{
      points: density,
      color: 'green'
    }]}>
      {({
        x,
        y
      }) => law.atom ? <line x1={x(law.mean)} x2={x(law.mean)} y1={y(0)} y2={y(1)} className="ito-correction" /> : null}
    </Plot>
    <div className="ito-budget"><div><span>+η²: variance injected per time</span><i style={{
          width: `${100 * law.varianceInjection / budgetScale}%`
        }} /><strong>{display(law.varianceInjection)}</strong></div>
      <div className="removal"><span>−2θv: variance removed per time</span><i style={{
          width: `${100 * law.varianceRemoval / budgetScale}%`
        }} /><strong>−{display(law.varianceRemoval)}</strong></div></div>
    <div aria-live="polite"><Metrics rows={[['Mean', law.mean], ['Variance v', law.variance], ['Variance change rate', law.varianceRate], ['Stationary variance for θ>0', law.stationaryVariance], ['Initial law', stationary ? 'Invariant from t=0' : 'A deterministic start']]} /></div>
    <p>The curve is calculated from the exact Gaussian marginal, not fitted to a handful of paths. Arrows encode direction only; the two bars share a scale within the current variance budget. The state axis adapts to the law. Zero variance is drawn as a probability atom rather than a density spike.</p>
    <p>{theta === 0 ? 'With θ=0 there is no restoring drift. Positive η gives variance η²t and no finite stationary Gaussian law. With η=0 the state remains its initial constant.' : 'Starting from the stationary Gaussian balances variance injection and removal immediately. A fixed initial state approaches that law; it is not stationary merely because the model has a stationary distribution. A displayed budget residual near 10⁻¹⁶ is floating-point rounding, not physical evolution.'}</p>
    <button onClick={() => {
      setTheta(1);
      setEta(0.8);
      setTime(1);
      setInitialMode('fixed');
    }}>Reset OU</button>
  </section>;
}
export function CoupledSdeSolverLab() {
  const [resetId, setResetId] = useState(0);
  const [seed, setSeed] = useState(17);
  const [group, setGroup] = useState(8);
  const [sigma, setSigma] = useState(0.6);
  const [step, setStep] = useState(1);
  const [stress, setStress] = useState(false);
  const fine = useMemo(() => brownianIncrements({
    seed,
    steps: 128,
    horizon: 1
  }), [seed]);
  const active = stress ? [-2] : groupIncrements(fine, group);
  const state = growthPath({
    mu: 0.4,
    sigma: stress ? 1 : sigma,
    increments: active
  });
  const selected = state.rows[Math.min(step, state.steps)];
  const previous = state.rows[Math.max(0, Math.min(step, state.steps) - 1)];
  const lower = Math.min(0, ...state.rows.flatMap(row => [row.exact, row.euler, row.milstein]));
  const upper = Math.max(1, ...state.rows.flatMap(row => [row.exact, row.euler, row.milstein]));
  return <section className="ito-lab" aria-label="Coupled SDE solver investigation"><h3>Give each solver the same noise</h3>
    <p>Inspect the coarse increment by adding the fine increments. Then compare exact GBM, Euler-Maruyama and scalar Milstein at matching times. Drift μ=0.4, X₀=1 and terminal time T=1 are held fixed.</p>
    <div className="ito-controls"><Choice label="Solver fixture" numeric={false} value={stress ? 'stress' : 'seeded'} options={[['seeded', 'Seeded Brownian grid'], ['stress', 'Positivity stress case']]} change={value => {
        setStress(value === 'stress');
        setStep(1);
      }} />
      <Choice label="Solver coarse steps" value={stress ? 1 : group} disabled={stress} options={stress ? [[1, '1']] : [[32, '4'], [8, '16'], [2, '64'], [1, '128']]} change={value => {
        setGroup(value);
        setStep(1);
      }} />
      <Choice label="Solver noise scale" value={stress ? 1 : sigma} disabled={stress} change={setSigma} options={[[0, 'σ=0'], [0.3, 'σ=0.3'], [0.6, 'σ=0.6'], [1, 'σ=1']]} /></div>
    {!stress && <SeedControl key={'solver-' + seed + '-' + resetId} label="Solver seed" active={seed} change={value => {
      setSeed(value);
      setStep(1);
    }} />}
    <div className="ito-actions"><button disabled={step === 0} onClick={() => setStep(value => value - 1)}>Previous solver step</button>
      <button disabled={step >= state.steps} onClick={() => setStep(value => value + 1)}>Next solver step</button></div>
    <div className="ito-noise-group"><strong>Selected ΔW: {display(selected.increment)}</strong>
      <span>{step === 0 ? 'Initial state: no increment used yet.' : stress ? 'Controlled one-step increment −2; h=1, σ=1.' : `${group} fine increments, each over time 1/128, form this interval.`}</span>
      {!stress && step > 0 && <div>{fine.slice((step - 1) * group, step * group).map((value, i) => <span key={i} className={value < 0 ? 'negative' : 'positive'}>{display(value)}</span>)}</div>}
    </div>
    <Plot compact={stress} label="Three solver paths driven by identical Brownian increments" xLabel="time t" yLabel="state X" xDomain={[0, 1]} yDomain={[lower - 0.15 * (upper - lower), upper + 0.15 * (upper - lower)]} series={[{
      points: state.rows.map(row => [row.time, row.exact]),
      color: 'green'
    }, {
      points: state.rows.map(row => [row.time, row.euler]),
      color: 'amber',
      dashed: true
    }, {
      points: state.rows.map(row => [row.time, row.milstein]),
      color: 'blue',
      dashed: true
    }]}>
      {({
        x,
        top,
        bottom
      }) => <line x1={x(selected.time)} x2={x(selected.time)} y1={top} y2={bottom} className="ito-cursor" />}
    </Plot>
    <div className="ito-step-columns"><div><strong>Euler update</strong><span>previous: {display(previous.euler)}</span>
      <span>drift: {display(step ? 0.4 * previous.euler * state.step : 0)}</span>
      <span>noise: {display(step ? state.sigma * previous.euler * selected.increment : 0)}</span>
      <span>new: {display(selected.euler)}</span></div>
      <div><strong>Milstein update</strong><span>previous: {display(previous.milstein)}</span>
        <span>drift: {display(selected.drift)}</span><span>noise: {display(selected.noise)}</span>
        <span>quadratic correction: {display(selected.correction)}</span><span>new: {display(selected.milstein)}</span></div></div>
    <div aria-live="polite"><Metrics rows={[['Selected time', selected.time], ['Exact state', selected.exact], ['Exact terminal value', state.rows.at(-1).exact], ['Euler nonpositive?', state.eulerNegative ? 'Yes' : 'No']]} /></div>
    <p>Green: exact sampled GBM. Amber dashed: EM. Blue dashed: Milstein. Lines join grid values; no continuous interpolation guarantee is implied. Every coarse grid sums the same finest Brownian increments, so the exact terminal value stays fixed. A seed change deliberately changes the underlying example.</p>
    <p>{stress ? 'This controlled stress increment is not a random sample or an estimate of failure frequency. Exact GBM stays positive, while the Euler factor 1+0.4−2 is negative. No method silently clips the state.' : 'The 128 finest increments use a repeatable synthetic normal generator. One path need not improve monotonically under refinement; strong error is an expectation over coupled paths.'}</p>
    <button onClick={() => {
      setResetId(value => value + 1);
      setSeed(17);
      setGroup(8);
      setSigma(0.6);
      setStep(1);
      setStress(false);
    }}>Reset coupled solvers</button>
  </section>;
}
export function SdeErrorLab() {
  const [mu, setMu] = useState(0.4);
  const [sigma, setSigma] = useState(0.6);
  const [steps, setSteps] = useState(16);
  const [samples, setSamples] = useState(512);
  const [method, setMethod] = useState('euler');
  const [result, setResult] = useState(null);
  const rows = useMemo(() => Array.from({
    length: 10
  }, (_, power) => ({
    steps: 2 ** power,
    euler: gbmErrorMoments({
      mu,
      sigma,
      steps: 2 ** power,
      method: 'euler'
    }),
    milstein: gbmErrorMoments({
      mu,
      sigma,
      steps: 2 ** power,
      method: 'milstein'
    })
  })), [mu, sigma]);
  const selected = gbmErrorMoments({
    mu,
    sigma,
    steps,
    method
  });
  const logErrors = rows.flatMap(row => [row.euler.rms, row.milstein.rms]).filter(value => value > 0).map(Math.log10);
  const low = Math.min(-3, ...(logErrors.length ? logErrors : [-3]));
  const high = Math.max(0, ...(logErrors.length ? logErrors : [0]));
  const change = setter => value => {
    setter(value);
    setResult(null);
  };
  return <section className="ito-lab" aria-label="SDE error budget investigation"><h3>Which error did you actually measure?</h3>
    <p>The curves below are finite-grid analytic errors for GBM under shared Brownian noise, with X₀=1 and T=1. Run a separate sample experiment to see estimation uncertainty. Sample count and step count do different jobs.</p>
    <div className="ito-controls"><Choice label="Error drift mu" value={mu} change={change(setMu)} options={[[-0.5, 'μ=−0.5'], [0, 'μ=0'], [0.4, 'μ=0.4'], [1, 'μ=1']]} />
      <Choice label="Error noise sigma" value={sigma} change={change(setSigma)} options={[[0, 'σ=0'], [0.05, 'σ=0.05'], [0.6, 'σ=0.6'], [1, 'σ=1']]} />
      <Choice label="Error method" numeric={false} value={method} change={change(setMethod)} options={[['euler', 'Euler-Maruyama'], ['milstein', 'Scalar Milstein']]} />
      <Choice label="Error step count" value={steps} change={change(setSteps)} options={[[4, '4'], [16, '16'], [64, '64'], [256, '256']]} />
      <Choice label="Monte Carlo path count" value={samples} change={change(setSamples)} options={[[64, '64'], [512, '512'], [4096, '4096']]} /></div>
    <Plot compact label="Analytic terminal RMS errors under shared Brownian noise" xLabel="step count n (log₂ spacing)" yLabel="log₁₀ terminal RMS error" xDomain={[0, 9]} xTicks={[0, 3, 6, 9]} xTick={value => 2 ** value} yDomain={[low - 0.3, high + 0.3]} series={['euler', 'milstein'].map((name, i) => ({
      color: i ? 'blue' : 'amber',
      points: rows.map(row => row[name].rms === 0 ? null : [Math.log2(row.steps), Math.log10(row[name].rms)])
    }))} />
    <p>Amber: EM. Blue: Milstein. Zero errors have no finite logarithm and are omitted; their exact value remains in the table. Axes adapt to the computed range. These are exact finite-problem moment formulas evaluated in binary64 with cancellation-safe expressions, not empirical convergence measurements.</p>
    <div aria-live="polite"><Metrics rows={[['Analytic terminal RMS', selected.rms], ['Analytic first-moment bias', selected.meanBias], ['Analytic second-moment bias', selected.secondBias], ['Time step h', selected.h]]} /></div>
    <button onClick={() => setResult(sampledErrors({
      mu,
      sigma,
      steps,
      samples,
      method,
      seed: 29
    }))}>Run paired Monte Carlo</button>
    {result && <div className="ito-sample-result" aria-live="polite"><h4>Seed 29 · {result.samples} paired paths</h4>
      <Metrics rows={[['Estimated bias of mean', result.bias], ['Bias standard error', result.biasStandardError], ['Sample mean squared error', result.mse], ['SE of mean squared error', result.mseStandardError], ['Square root of sample MSE', result.rms]]} />
      <p>The standard error belongs to the sample mean squared error, not directly to its square root. It estimates finite-sample uncertainty; it neither removes discretization bias nor guarantees coverage in every run. Repeating with the same controls and seed reproduces this finite experiment.</p></div>}
    <details><summary>Inspect all analytic resolutions</summary><div className="ito-table" tabIndex={0}><table>
      <thead><tr><th>Steps</th><th>EM RMS</th><th>Milstein RMS</th><th>Mean bias (both)</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.steps}><td>{row.steps}</td><td>{display(row.euler.rms)}</td><td>{display(row.milstein.rms)}</td><td>{display(row.euler.meanBias)}</td></tr>)}</tbody>
    </table></div></details>
    <p>Try μ=0 with positive noise: the first-moment bias is zero, but path error is not. Matching one moment does not make the full distribution or path correct. General convergence orders need the smoothness and integrability conditions stated in the lesson.</p>
    <button onClick={() => {
      setMu(0.4);
      setSigma(0.6);
      setSteps(16);
      setSamples(512);
      setMethod('euler');
      setResult(null);
    }}>Reset error experiment</button>
  </section>;
}
