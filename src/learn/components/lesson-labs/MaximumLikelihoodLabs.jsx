import { useState } from 'react';
import { bernoulli, betaDensity, betaPosterior, locationFit, logistic, logit, parseBinary, parseMeasurements, samplingMass, transformedBetaDensity } from '../../data/maximum-likelihood-models.js';
import './maximum-likelihood-labs.css';
const fmt = value => value === null ? 'not unique' : value === -Infinity ? '−∞' : value !== 0 && Math.abs(value) < .0001 ? value.toExponential(2) : Number(value.toFixed(5)).toString();
const axisNumber = value => value < .01 ? value.toExponential(0) : Number(value.toPrecision(2)).toString();
const colors = ['#f0bd58', '#70d5b0', '#c2a4f3'];
function Curve({
  title,
  series,
  domain = [0, 1],
  marker,
  xlabel = 'candidate probability p',
  ymax: suppliedMax
}) {
  const samples = Array.from({
    length: 161
  }, (_, index) => domain[0] + index * (domain[1] - domain[0]) / 160);
  const evaluated = series.map(row => samples.map(x => ({
    x,
    y: row.fn(x)
  })));
  const ymax = suppliedMax || Math.max(Number.MIN_VALUE, ...evaluated.flat().map(row => row.y)) * 1.08;
  const X = x => 49 + 289 * (x - domain[0]) / (domain[1] - domain[0]);
  const Y = y => 172 - 127 * y / ymax;
  return <figure className="mle-curve">
    <figcaption>{title}</figcaption>
    <svg viewBox="0 0 360 237" role="img" aria-label={`${title}. ${series.map(row => row.label).join(' and ')} plotted from the stated formulas.`}>
      <path d="M49 40V172H338" className="mle-axis" />
      <path d="M49 45H338" className="mle-axis" strokeDasharray="3 6" opacity=".35" />
      <text x="49" y="29">{axisNumber(ymax)}</text><text x="44" y="178" textAnchor="end">0</text>
      {[domain[0], (domain[0] + domain[1]) / 2, domain[1]].map(x => <text key={x} x={X(x)} y="201" textAnchor={x === domain[0] ? 'start' : x === domain[1] ? 'end' : 'middle'}>{fmt(x)}</text>)}
      <text x="194" y="226" textAnchor="middle">{xlabel}</text>
      {evaluated.map((points, index) => <path key={series[index].label} fill="none" stroke={colors[index]} strokeWidth="2.8" strokeDasharray={index === 1 ? '7 3' : undefined} d={points.map((point, i) => `${i ? 'L' : 'M'}${X(point.x)},${Y(point.y)}`).join(' ')} />)}
      {marker !== undefined && <><line x1={X(marker)} x2={X(marker)} y1="42" y2="172" stroke="#eee" strokeDasharray="3 4" /><circle cx={X(marker)} cy={Y(series[0].fn(marker))} r="5" fill="#fff" /></>}
    </svg>
    <div className="mle-legend">{series.map((row, index) => <span key={row.label} style={{
        color: colors[index]
      }}>{index === 1 ? '┄' : '━'} {row.label}</span>)}</div>
  </figure>;
}
function Readouts({
  rows
}) {
  return <dl className="mle-readouts">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Range({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = .01
}) {
  return <label className="mle-range">{label}: <strong>{fmt(value)}</strong><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
export function BernoulliLikelihoodLab() {
  const [draft, setDraft] = useState('1 1 1 0');
  const [data, setData] = useState([1, 1, 1, 0]);
  const [p, setP] = useState(.5);
  const [countEvent, setCountEvent] = useState(false);
  const [error, setError] = useState('');
  const successes = data.reduce((sum, value) => sum + value, 0);
  const failures = data.length - successes;
  const score = bernoulli(successes, failures, p, countEvent);
  const apply = () => {
    try {
      setData(parseBinary(draft));
      setError('');
    } catch (caught) {
      setError(caught.message);
    }
  };
  const reset = () => {
    setDraft('1 1 1 0');
    setData([1, 1, 1, 0]);
    setP(.5);
    setCountEvent(false);
    setError('');
  };
  return <section className="mle-lab" aria-label="Bernoulli likelihood investigation">
    <h3>Hold the observations still. Move the candidate.</h3>
    <p>Predict whether p = 1 can explain three successes <em>and one failure</em>. Then move the white candidate line. Applied observations stay fixed until you apply a new draft.</p>
    <label>Binary observations<textarea aria-label="Binary observations" value={draft} onChange={event => setDraft(event.target.value)} /></label>
    <div className="mle-actions"><button onClick={apply}>Apply observations</button><button onClick={reset}>Reset binary experiment</button></div>
    {error && <p role="alert">{error}</p>}
    <div className="mle-data-strip" aria-label="Applied observations">{data.length ? data.map((value, index) => <span key={index} className={value ? 'success' : 'failure'}>{value === 1 ? '✓ 1' : '× 0'}</span>) : <span>No observations · empty product = 1</span>}</div>
    <Range label="Candidate p" value={p} onChange={setP} />
    <label>Observed event<select aria-label="Observed event" value={countEvent ? 'count' : 'sequence'} onChange={event => setCountEvent(event.target.value === 'count')}><option value="sequence">This ordered sequence</option><option value="count">This many successes, any order</option></select></label>
    <Curve title={countEvent ? 'Probability of this count · likelihood as p varies' : 'Probability of this sequence · likelihood as p varies'} series={[{
      label: 'Likelihood',
      fn: x => bernoulli(successes, failures, x, countEvent).value
    }]} marker={p} />
    <div aria-live="polite"><Readouts rows={[[`${successes} successes + ${failures} failures`, `${data.length} observations`], ['Candidate likelihood', fmt(score.value)], ['Candidate log-likelihood', fmt(score.log)], ['MLE', score.mle === null ? 'Every p in [0,1]' : fmt(score.mle)]]} /></div>
    <p className="mle-note">The vertical scale is a probability of the chosen <em>data event</em>. Area across candidate p is not a parameter probability. Changing sequence to count multiplies all candidate scores by the same binomial coefficient.</p>
    <details><summary>Try a changed situation</summary><p>Apply <code>1 1 1 1</code>: the maximum moves to p = 1. Clear the observations and apply: the curve is flat and no p is preferred. Return to the original data and compare .5 with .75: the ratio is 1.6875 in both event views.</p></details>
  </section>;
}
export function LocationFitLab() {
  const [draft, setDraft] = useState('2 3 4 7');
  const [values, setValues] = useState([2, 3, 4, 7]);
  const [center, setCenter] = useState(4);
  const [error, setError] = useState('');
  const fit = locationFit(values, center);
  const low = Math.min(...values) - 2;
  const high = Math.max(...values) + 2;
  const X = value => 53 + (value - low) / (high - low) * 280;
  const apply = () => {
    try {
      const next = parseMeasurements(draft);
      setValues(next);
      setCenter(locationFit(next, 0).mean);
      setError('');
    } catch (caught) {
      setError(caught.message);
    }
  };
  return <section className="mle-lab" aria-label="Measurement residual investigation">
    <h3>Which center pays the smallest residual bill?</h3>
    <p>Each row is one observation. Its line reaches from the white candidate center to that value. Longer errors contribute proportionally to absolute cost, but quadratically to squared cost.</p>
    <label>Measurements<textarea aria-label="Measurements" value={draft} onChange={event => setDraft(event.target.value)} /></label>
    <div className="mle-actions"><button onClick={apply}>Apply measurements</button><button onClick={() => {
        setDraft('2 3 4 7 27');
        setValues([2, 3, 4, 7, 27]);
        setCenter(4);
        setError('');
      }}>Add the outlier 27</button><button onClick={() => {
        setDraft('2 3 4 7');
        setValues([2, 3, 4, 7]);
        setCenter(4);
        setError('');
      }}>Reset measurements</button></div>
    {error && <p role="alert">{error}</p>}
    <Range label="Candidate center" value={center} onChange={setCenter} min={low} max={high} step={.1} />
    <div className="mle-actions"><button onClick={() => setCenter(fit.mean)}>Use sample mean</button><button onClick={() => setCenter((fit.medianLow + fit.medianHigh) / 2)}>Use median midpoint</button></div>
    <figure><figcaption>Signed residuals · observation − center</figcaption><svg viewBox={`0 0 360 ${values.length * 33 + 85}`} role="img" aria-label="Each observation is joined horizontally to the candidate center; the table below gives exact residuals.">
      <line x1={X(center)} x2={X(center)} y1="18" y2={values.length * 33 + 12} stroke="#eee" strokeDasharray="4 3" />
      {values.map((value, index) => <g key={index}><text x="5" y={index * 33 + 34}>x{index + 1}</text><line x1={X(center)} x2={X(value)} y1={index * 33 + 28} y2={index * 33 + 28} stroke={colors[value >= center ? 0 : 1]} strokeWidth="5" /><circle cx={X(value)} cy={index * 33 + 28} r="6" fill={colors[value >= center ? 0 : 1]} /></g>)}
      <path d={`M53 ${values.length * 33 + 26}H333`} className="mle-axis" /><text x="53" y={values.length * 33 + 49}>{fmt(low)}</text><text x="333" y={values.length * 33 + 49} textAnchor="end">{fmt(high)}</text><text x="193" y={values.length * 33 + 77} textAnchor="middle">measurement units</text>
    </svg></figure>
    <div aria-live="polite"><Readouts rows={[["Squared cost Σ(x−c)²", fmt(fit.squared)], ['Absolute cost Σ|x−c|', fmt(fit.absolute)], ['Squared-cost minimizer', fmt(fit.mean)], ['Absolute-cost minimizers', fit.medianLow === fit.medianHigh ? fmt(fit.medianLow) : `[${fit.medianLow}, ${fit.medianHigh}]`]]} /></div>
    <details><summary>Observation and residual table</summary><table><thead><tr><th>Observation</th><th>Residual</th><th>Squared</th></tr></thead><tbody>{values.map((value, index) => <tr key={index}><td>{value}</td><td>{fmt(fit.residuals[index])}</td><td>{fmt(fit.residuals[index] ** 2)}</td></tr>)}</tbody></table></details>
    <p className="mle-note">Lines encode signed errors, not uncertainty intervals. The normal model with fixed variance minimizes squared cost. A Laplace model with fixed scale minimizes absolute cost. Cost units differ; their raw totals are not directly comparable likelihoods.</p>
  </section>;
}
export function BetaMapLab() {
  const [s, setS] = useState(3);
  const [f, setF] = useState(1);
  const [a, setA] = useState(2);
  const [b, setB] = useState(2);
  const posterior = betaPosterior(s, f, a, b);
  const mle = s + f ? s / (s + f) : null;
  return <section className="mle-lab" aria-label="Prior and posterior investigation">
    <h3>Evidence changes a distribution before you choose a point</h3>
    <p>Predict what happens if four successes have no failures. Compare the highest posterior density with the chance of the next success.</p>
    <div className="mle-controls"><Range label="Successes" value={s} onChange={setS} max={20} step={1} /><Range label="Failures" value={f} onChange={setF} max={20} step={1} /><Range label="Prior alpha" value={a} onChange={setA} min={1} max={12} step={1} /><Range label="Prior beta" value={b} onChange={setB} min={1} max={12} step={1} /></div>
    <div className="mle-actions"><button onClick={() => {
        setS(4);
        setF(0);
        setA(1);
        setB(1);
      }}>Four successes, uniform prior</button><button onClick={() => {
        setS(0);
        setF(0);
        setA(1);
        setB(1);
      }}>No data, uniform prior</button><button onClick={() => {
        setS(3);
        setF(1);
        setA(2);
        setB(2);
      }}>Reset prior experiment</button></div>
    <div className="mle-two-plots"><Curve title="Parameter density · area = 1 for each curve" series={[{
        label: `Posterior Beta(${posterior.a},${posterior.b})`,
        fn: p => betaDensity(p, posterior.a, posterior.b)
      }, {
        label: `Prior Beta(${a},${b})`,
        fn: p => betaDensity(p, a, b)
      }]} /><Curve title="Relative likelihood · peak scaled to 1" ymax={1.08} series={[{
        label: 'Likelihood / maximum',
        fn: p => Math.exp(bernoulli(s, f, p).log - bernoulli(s, f, mle ?? .5).log)
      }]} /></div>
    <div aria-live="polite"><Readouts rows={[["MLE", mle === null ? 'Not unique' : fmt(mle)], ['Posterior mode / MAP', posterior.mode === null ? 'Every p in [0,1]' : fmt(posterior.mode)], ['Posterior mean', fmt(posterior.mean)], ['Next-success probability', fmt(posterior.mean)], ['Posterior standard deviation', fmt(Math.sqrt(posterior.variance))]]} /></div>
    <p className="mle-note">Positive integer prior shapes keep every plotted density finite. The readouts use exact analytic formulas evaluated in floating point; the lines sample those formulas. Posterior standard deviation describes spread in p under this model, not a confidence interval.</p>
    <details><summary>What to notice</summary><p>With four successes and a uniform prior, MLE and MAP equal 1, but the predictive probability is 5/6. A density mode at the boundary does not mean all posterior mass sits there. With no data and a uniform prior, every p is a mode while the mean is 1/2.</p></details>
  </section>;
}
export function SamplingEstimateLab() {
  const [n, setN] = useState(4);
  const [p, setP] = useState(.5);
  const rows = samplingMass(n, p);
  const top = Math.max(...rows.map(row => row.mass));
  const X = x => 48 + 285 * x;
  return <section className="mle-lab" aria-label="Repeated sample estimate investigation">
    <h3>Imagine repeating the entire study</h3>
    <p>Here the true p is stipulated. The data and estimate vary. No studies are simulated: each bar is the exact binomial probability of one possible estimate k/n.</p>
    <Range label="Assumed true p" value={p} onChange={setP} /><Range label="Observations per study" value={n} onChange={setN} min={1} max={30} step={1} />
    <button onClick={() => {
      setN(4);
      setP(.5);
    }}>Reset repeated studies</button>
    <figure><figcaption>Probability mass for each possible estimate</figcaption><svg viewBox="0 0 360 244" role="img" aria-label={`Binomial sampling distribution of the sample proportion for n=${n}, p=${p}; exact rows available below.`}>
      <path d="M48 39V179H333" className="mle-axis" /><text x="43" y="49" textAnchor="end">{fmt(top).slice(0, 4)}</text><text x="43" y="184" textAnchor="end">0</text>
      {rows.map(row => <line key={row.k} x1={X(row.estimate)} x2={X(row.estimate)} y1="179" y2={179 - 130 * row.mass / top} stroke={colors[0]} strokeWidth={Math.min(13, 200 / (n + 1))} />)}
      <line x1={X(p)} x2={X(p)} y1="42" y2="179" stroke={colors[1]} strokeDasharray="4 4" />
      {[0, .5, 1].map(x => <text key={x} x={X(x)} y="204" textAnchor="middle">{x}</text>)}<text x="192" y="234" textAnchor="middle">possible estimate k/n</text>
    </svg><div className="mle-legend"><span style={{
          color: colors[0]
        }}>┃ Outcome mass</span><span style={{
          color: colors[1]
        }}>┄ True p</span></div></figure>
    <div aria-live="polite"><Readouts rows={[["Expected estimate", fmt(p)], ['Variance across studies', fmt(p * (1 - p) / n)], ['Standard deviation', fmt(Math.sqrt(p * (1 - p) / n))]]} /></div>
    <details><summary>Exact mass table (rounded for display)</summary><table><thead><tr><th>Successes k</th><th>Estimate k/n</th><th>Probability</th></tr></thead><tbody>{rows.map(row => <tr key={row.k}><td>{row.k}</td><td>{fmt(row.estimate)}</td><td>{fmt(row.mass)}</td></tr>)}</tbody></table></details>
    <p className="mle-note">The bar height is probability mass, not density. Increase n: possible estimates become more closely spaced and concentrate near p. This is a sampling distribution, not the fixed-data likelihood from the first investigation.</p>
  </section>;
}
export function CoordinateModeFigure() {
  return <figure className="mle-coordinate" aria-label="A mode depends on the parameter coordinate">
    <figcaption>One posterior · two density coordinates</figcaption>
    <div className="mle-two-plots"><Curve title="Density per unit p" series={[{
        label: 'Beta(5,3)',
        fn: p => betaDensity(p, 5, 3)
      }]} marker={2 / 3} /><Curve title="Density per unit log-odds η" domain={[-4, 4]} xlabel="log-odds η" series={[{
        label: 'Beta density × p(1−p)',
        fn: eta => transformedBetaDensity(eta, 5, 3)
      }]} marker={logit(5 / 8)} /></div>
    <Readouts rows={[["Mode in p", '2/3 ≈ .66667'], ['Mode in η', 'log(5/3) ≈ .51083'], ['That η maps back to p', fmt(logistic(logit(5 / 8)))]]} />
    <p>The first white line marks p = 2/3; the second marks η = log(5/3). A uniform-width step in η covers a variable-width interval in p. The Jacobian p(1−p) adjusts density height to preserve probability mass. The log-odds plot shows η from −4 to 4; its tails continue beyond the frame.</p>
  </figure>;
}
