import { useId, useState } from 'react';
import { finiteApproximation, restrictedFiniteOptimum, gaussianProjection, covarianceEllipse, coordinateAscent, mixtureProjection, variationalGradients } from '../../data/variational-inference-models.js';
import './variational-inference-labs.css';
const amber = '#f0bd58',
  green = '#70d5b0';
const fmt = value => !Number.isFinite(value) ? String(value) : Math.abs(value) < 1e-12 ? '0' : Math.abs(value) < 0.00001 ? value.toExponential(2) : Number(value.toFixed(5)).toString();
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 1
}) {
  return <label className="vi-range">{label}: <strong>{fmt(value)}</strong><input type="range" aria-label={label} value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Readouts({
  rows
}) {
  return <dl className="vi-readouts">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Table({
  caption,
  headings,
  rows
}) {
  return <details className="vi-table"><summary>{caption}</summary><div role="region" aria-label={caption} tabIndex={0}><table><thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => j === 0 ? <th key={j} scope="row">{cell}</th> : <td key={j}>{cell}</td>)}</tr>)}</tbody></table></div></details>;
}
function Legend({
  rows
}) {
  return <div className="vi-legend">{rows.map(([label, color, dashed]) => <span key={label} style={{
      color
    }}>{dashed ? '┄' : '━'} {label}</span>)}</div>;
}
function Geometry({
  title,
  target,
  candidate,
  path,
  current,
  domain = [-1.5, 1.5],
  xLabel = 'latent θ₁',
  yLabel = 'latent θ₂'
}) {
  const clip = useId().replaceAll(':', '');
  const X = x => 58 + 260 * (x - domain[0]) / (domain[1] - domain[0]);
  const Y = y => 308 - 260 * (y - domain[0]) / (domain[1] - domain[0]);
  const draw = points => points.map(([x, y], index) => `${index ? 'L' : 'M'}${X(x)} ${Y(y)}`).join(' ');
  const ticks = [domain[0], (domain[0] + domain[1]) / 2, domain[1]];
  return <figure className="vi-geometry"><figcaption>{title}</figcaption><svg viewBox="0 0 360 376" role="img" aria-label={`${title}. Equal coordinate scales; exact coordinates are available in the adjoining readouts or table.`}>
    <defs><clipPath id={clip}><rect x="58" y="48" width="260" height="260" /></clipPath></defs>
    <text x="58" y="28">{yLabel}</text><path d="M58 48V308H318" className="vi-axis" />
    {ticks.map(value => <g key={value}><path d={`M${X(value)} 48V308M58 ${Y(value)}H318`} className="vi-grid" /><text x={X(value)} y="338" textAnchor="middle">{fmt(value)}</text><text x="51" y={Y(value) + 6} textAnchor="end">{fmt(value)}</text></g>)}
    <g clipPath={`url(#${clip})`}><path d={draw(target)} fill="none" stroke={green} strokeWidth="3" strokeDasharray="7 5" />{candidate && <path d={draw(candidate)} fill="none" stroke={amber} strokeWidth="3" />}{path && <><path d={draw(path)} fill="none" stroke={amber} strokeWidth="3" />{path.map(([x, y], i) => <circle key={i} cx={X(x)} cy={Y(y)} r="3" fill={amber} />)}</>}{current && <circle cx={X(current[0])} cy={Y(current[1])} r="6" fill="#fff" stroke="#111" strokeWidth="2" />}</g>
    <text x="190" y="369" textAnchor="middle">{xLabel}</text>
  </svg></figure>;
}
export function ElboBalanceLab() {
  const [first, setFirst] = useState(0.2),
    [share, setShare] = useState(0.5),
    [restricted, setRestricted] = useState(true);
  const result = finiteApproximation(first, restricted ? 0.5 : share);
  const optimum = restrictedFiniteOptimum();
  const applyBest = () => {
    setFirst(restricted ? optimum.q[0] : 0.2);
    setShare(restricted ? 0.5 : 0.625);
  };
  return <section className="vi-lab" role="region" aria-label="ELBO probability balance investigation">
    <header><span className="vi-eyebrow">Probability mass → objective</span><h3>Which approximation fits these three possibilities?</h3><p>Predict whether making B and C equally probable can reproduce the target. Amber is q; dashed green is the exact posterior.</p></header>
    <div className="vi-controls"><label>Allowed family<select aria-label="Allowed family" value={restricted ? 'restricted' : 'free'} onChange={event => setRestricted(event.target.value === 'restricted')}><option value="restricted">Restricted: B and C equal</option><option value="free">Free: any three probabilities</option></select></label><Range label="Probability of A" value={first} min={0} max={1} step="any" onChange={setFirst} />{!restricted && <Range label="B share of remaining mass" value={share} min={0} max={1} step={0.025} onChange={setShare} />}</div>
    <div className="vi-actions"><button onClick={applyBest}>Fit best allowed distribution</button><button onClick={() => {
        setFirst(0.2);
        setShare(0.5);
        setRestricted(true);
      }}>Reset probability balance</button></div>
    <div className="vi-mass-bars" role="img" aria-label={`Probabilities for A, B, C. q ${result.q.map(fmt).join(', ')}; target .2, .5, .3.`}>{result.rows.map((row, i) => <div key={i}><strong>{['A', 'B', 'C'][i]}</strong><div className="vi-mass-track"><i style={{
            width: `${row.probability * 100}%`
          }} /><b style={{
            left: `${row.posterior * 100}%`
          }} /></div><span>q {fmt(row.probability)}<br />p {fmt(row.posterior)}</span></div>)}</div>
    <p className="vi-scale-note">Every track runs from probability 0 to 1. A green marker denotes the exact target mass.</p>
    <div className="vi-balance"><span>ELBO<br /><strong>{fmt(result.elbo)}</strong></span><b>+</b><span>KL gap<br /><strong>{fmt(result.kl)}</strong></span><b>=</b><span>log evidence<br /><strong>{fmt(result.logEvidence)}</strong></span></div>
    <p role="status">{restricted ? `Even the best restricted distribution leaves a KL gap of ${fmt(optimum.kl)}. The current excess above that family gap is ${fmt(Math.max(0, result.kl - optimum.kl))}.` : 'The free family contains the target. Fit it to close the gap; changing q does not change the evidence.'}</p>
    <Table caption="Inspect each probability and ELBO term" headings={['State', 'Joint', 'q', 'q log joint', '−q log q', 'q log(q/p)']} rows={result.rows.map((row, i) => [['A', 'B', 'C'][i], ...[row.joint, row.probability, row.expectedLogJoint, row.entropy, row.kl].map(fmt)])} />
  </section>;
}
export function GaussianFamilyLab() {
  const [rho, setRho] = useState(0.8),
    [family, setFamily] = useState('mean-field');
  const model = gaussianProjection(rho, family);
  return <section className="vi-lab" role="region" aria-label="Gaussian dependence investigation">
    <header><span className="vi-eyebrow">Dependence → decision uncertainty</span><h3>A round approximation loses a tilted relationship</h3><p>Predict the uncertainty of θ₁+θ₂ and θ₁−θ₂ before changing the correlation. The two decisions need different directions through the same distribution.</p></header>
    <div className="vi-controls"><Range label="Target correlation" value={rho} min={-0.9} max={0.9} step={0.1} onChange={setRho} /><label>Approximation<select aria-label="Gaussian approximation" value={family} onChange={event => setFamily(event.target.value)}><option value="mean-field">Best reverse-KL mean-field</option><option value="marginals">Product of true marginals</option><option value="full">Full covariance: exact target</option></select></label></div>
    <Geometry title="Equal-distance contours in latent space" target={covarianceEllipse(model.target)} candidate={covarianceEllipse(model.covariance)} /><Legend rows={[[family === 'full' ? 'q matches p' : 'approximation q', amber], ['target p', green, true]]} />
    <p>Each contour has Mahalanobis radius 1 and encloses 39.35% under its own two-dimensional Gaussian. It is not a 68% or 95% interval. Both axes use the same scale.</p>
    <div className="vi-decision-table"><table><caption>Variance of the quantity being reported</caption><thead><tr><th>Quantity</th><th>Target p</th><th>Fit q</th></tr></thead><tbody><tr><th>θ₁</th><td>1</td><td>{fmt(model.variance)}</td></tr><tr><th>θ₁+θ₂</th><td>{fmt(model.targetSumVariance)}</td><td>{fmt(model.sumVariance)}</td></tr><tr><th>θ₁−θ₂</th><td>{fmt(model.targetDifferenceVariance)}</td><td>{fmt(model.differenceVariance)}</td></tr></tbody></table></div>
    <Readouts rows={[["KL(q || p)", fmt(model.kl)], ['q covariance', fmt(model.covariance[0][1])]]} />
    <p role="status">{family === 'full' ? 'Keeping the covariance recovers both directional uncertainties exactly in this Gaussian target.' : rho === 0 ? 'At zero correlation the target factorizes, so this approximation is exact.' : 'Compare each row separately: losing covariance does not mean every derived variance changes in the same direction.'}</p>
    <button onClick={() => {
      setRho(0.8);
      setFamily('mean-field');
    }}>Reset Gaussian comparison</button>
  </section>;
}
export function CoordinateAscentLab() {
  const [rho, setRho] = useState(0.8),
    [step, setStep] = useState(0);
  const trace = coordinateAscent(rho, 24),
    row = trace.rows[step];
  return <section className="vi-lab" role="region" aria-label="Coordinate ascent investigation">
    <header><span className="vi-eyebrow">One factor at a time</span><h3>Follow the means as each factor is optimized</h3><p>The axes now show adjustable q means m₁,m₂. They do not show random latent draws. The diagonal variances are already fixed to their optimal value 1−ρ².</p></header>
    <Range label="Coordinate target correlation" value={rho} min={-0.9} max={0.9} step={0.1} onChange={value => {
      setRho(value);
      setStep(0);
    }} />
    <div className="vi-actions"><button disabled={step === 24} onClick={() => setStep(step + 1)}>Update factor {step % 2 + 1}</button><button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous update</button><button onClick={() => setStep(24)}>Inspect 24 updates</button><button onClick={() => {
        setRho(0.8);
        setStep(0);
      }}>Reset coordinate ascent</button></div>
    <Geometry title="Coordinate-update path in parameter space" target={covarianceEllipse(trace.target, trace.targetMean, 1)} path={trace.rows.slice(0, step + 1).map(item => item.mean)} current={row.mean} domain={[-3, 4]} xLabel="variational mean m₁" yLabel="variational mean m₂" /><Legend rows={[["mean-update path", amber], ['constant excess-KL contour', green, true], ['white dot: current means', '#ffffff']]} />
    <p>The dashed contour marks optimization gap ½ around the best means (1,−1). It describes the objective over means, not a posterior probability region. This demonstration optimizes a normalized target directly, so its log normalizer is 0 and the displayed ELBO equals −KL.</p>
    <Readouts rows={[[`After ${step} updates`, `(${row.mean.map(fmt).join(', ')})`], ['Remaining optimization gap', fmt(row.optimizationGap)], ['Unavoidable family gap', fmt(row.familyGap)], ['Total KL', fmt(row.kl)]]} />
    <p role="status">{step === 0 ? 'Start at(−2,2). Predict the first new m₁ using1+ρ(m₂+1).' : `Factor ${row.updated + 1} changed. The other mean stayed fixed; the exact ELBO did not decrease. Reducing the optimization gap leaves the separate family gap.`}</p>
    <Table caption="Coordinate means and objective history" headings={['Update', 'm₁', 'm₂', 'ELBO', 'KL']} rows={trace.rows.slice(0, step + 1).map(item => [item.step, ...item.mean.map(fmt), fmt(item.elbo), fmt(item.kl)])} />
  </section>;
}
export function MixtureProjectionLab() {
  const [mean, setMean] = useState(0),
    [sd, setSd] = useState(3),
    [separation, setSeparation] = useState(3);
  const model = mixtureProjection(mean, sd, separation);
  const ymax = Math.ceil(Math.max(...model.curves.flatMap(row => [row.target, row.approximation])) * 20) / 20;
  const X = value => 55 + 270 * (value + 8) / 16,
    Y = value => 200 - 150 * value / ymax;
  const path = name => model.curves.map((row, i) => `${i ? 'L' : 'M'}${X(row.value)} ${Y(row[name])}`).join(' ');
  return <section className="vi-lab" role="region" aria-label="Two-mode approximation investigation">
    <header><span className="vi-eyebrow">Density shape → missed explanations</span><h3>Which side would your approximation report?</h3><p>The target has equally weighted unit-variance peaks at ±d. Try the broad candidate and each narrow candidate; then ask whether their sign probabilities preserve the two explanations.</p></header>
    <div className="vi-controls"><Range label="Approximation mean" value={mean} min={-4} max={4} step={0.1} onChange={setMean} /><Range label="Approximation standard deviation" value={sd} min={0.3} max={3} step={0.1} onChange={setSd} /><Range label="Target peak distance d" value={separation} min={0} max={4} step={0.5} onChange={setSeparation} /></div>
    <div className="vi-actions"><button onClick={() => {
        setMean(-3);
        setSd(1);
      }}>Candidate N(−3,1)</button><button onClick={() => {
        setMean(3);
        setSd(1);
      }}>Candidate N(3,1)</button><button onClick={() => {
        setMean(0);
        setSd(3);
        setSeparation(3);
      }}>Reset broad candidate</button></div>
    <figure className="vi-density"><figcaption>Calculated densities; area represents probability</figcaption><svg viewBox="0 0 360 270" role="img" aria-label={`Target and approximation density curves. Approximation mean${mean}, standard deviation${sd}; target peaks plus/minus${separation}.`}><text x="55" y="27">density</text><path d="M55 48V200H325" className="vi-axis" /><text x="50" y="65" textAnchor="end">{fmt(ymax)}</text><text x="50" y="207" textAnchor="end">0</text><line x1={X(0)} x2={X(0)} y1="50" y2="200" className="vi-grid" /><path d={path('target')} fill="none" stroke={green} strokeWidth="3" strokeDasharray="6 5" /><path d={path('approximation')} fill="none" stroke={amber} strokeWidth="3" />{[-8, 0, 8].map(x => <text key={x} x={X(x)} y="231" textAnchor="middle">{x}</text>)}<text x="190" y="263" textAnchor="middle">latent θ</text></svg><Legend rows={[["candidate q", amber], ['target p', green, true]]} /></figure>
    <Readouts rows={[["Approximate KL(q || p)", fmt(model.kl)], ['q(θ>0)', fmt(model.rightProbability)], ['Target p(θ>0)', '0.5']]} />
    <p role="status">These are candidate comparisons, not a proof of the global optimum. A lower reverse KL can still discard a consequential explanation.</p><p>KL uses deterministic Simpson integration over standard-normal noise[−9,9],1440 subintervals. The density picture displays θ∈[−8,8]; it does not truncate the target. Values are numerical approximations, not measurements.</p>
    <Table caption="Density values at selected coordinates" headings={['θ', 'p density', 'q density']} rows={model.curves.filter((_, i) => i % 30 === 0).map(row => [fmt(row.value), fmt(row.target), fmt(row.approximation)])} />
  </section>;
}
export function VariationalGradientLab() {
  const [mean, setMean] = useState(0),
    [sd, setSd] = useState(1.2),
    [count, setCount] = useState(100),
    [seed, setSeed] = useState(7),
    [draft, setDraft] = useState('7'),
    [error, setError] = useState(''),
    [index, setIndex] = useState(1);
  const model = variationalGradients({
      mean,
      sd,
      count,
      seed
    }),
    row = model.rows[index - 1];
  const reset = () => {
    setMean(0);
    setSd(1.2);
    setCount(100);
    setSeed(7);
    setDraft('7');
    setError('');
    setIndex(1);
  };
  const apply = event => {
    event.preventDefault();
    const value = Number(draft);
    if (!Number.isInteger(value) || value < 1 || value > 4294967295) {
      setError('Use an integer seed1–4294967295. The applied draws have not changed.');
      return;
    }
    setSeed(value);
    setError('');
    setIndex(1);
  };
  return <section className="vi-lab" role="region" aria-label="Variational gradient investigation">
    <header><span className="vi-eyebrow">Noise → draw → gradient contribution</span><h3>Follow one random draw through the derivative</h3><p>Target N(1.5,0.7²). q uses mean m and log standard deviation a=log s. Moving m or s reuses the same underlying noise, exposing what the transformation changes.</p></header>
    <div className="vi-controls"><Range label="Gradient approximation mean" value={mean} min={-2} max={3} step={0.1} onChange={setMean} /><Range label="Gradient approximation standard deviation" value={sd} min={0.2} max={2} step={0.1} onChange={setSd} /><label>Independent noise draws<select aria-label="Independent noise draws" value={count} onChange={event => {
          setCount(Number(event.target.value));
          setIndex(1);
        }}><option value="10">10</option><option value="100">100</option><option value="1000">1000</option></select></label></div>
    <form className="vi-seed" onSubmit={apply}><label>Gradient seed<input aria-label="Gradient seed" inputMode="numeric" value={draft} onChange={event => setDraft(event.target.value)} /></label><button type="submit">Apply gradient seed</button>{error && <p role="alert">{error}</p>}</form>
    <Range label="Inspect noise draw" value={index} min={1} max={count} onChange={setIndex} />
    <div className="vi-gradient-flow"><div><span>Fixed noise ε</span><strong>{fmt(row.noise)}</strong></div><b aria-hidden="true">↓</b><div><span>θ = m+sε</span><strong>{fmt(row.value)}</strong></div><b aria-hidden="true">↓</b><div><span>Target slope (1.5−θ)/0.49</span><strong>{fmt(row.targetSlope)}</strong></div><b aria-hidden="true">↓</b><div className="vi-gradient-pair"><span>Contribution to ∂L/∂m<strong>{fmt(row.pathMean)}</strong></span><span>Contribution to ∂L/∂a<strong>{fmt(row.pathLogSd)}</strong><small>1+sε × target slope</small></span></div></div>
    <div className="vi-decision-table"><table><caption>Gradient averages at this fixed q</caption><thead><tr><th>Estimator</th><th>Mean m</th><th>Log sd a</th></tr></thead><tbody><tr><th>Exact</th><td>{fmt(model.exactMean)}</td><td>{fmt(model.exactLogSd)}</td></tr><tr><th>Pathwise</th><td>{fmt(model.estimates.pathMean.average)}</td><td>{fmt(model.estimates.pathLogSd.average)}</td></tr><tr><th>Score</th><td>{fmt(model.estimates.scoreMean.average)}</td><td>{fmt(model.estimates.scoreLogSd.average)}</td></tr></tbody></table></div>
    <div className="vi-actions"><button onClick={() => {
        setMean(1.5);
        setSd(0.7);
      }}>Match q to target</button><button onClick={reset}>Reset gradient experiment</button></div>
    <p role="status">{mean === 1.5 && sd === 0.7 ? 'At p=q the exact gradient is zero. This normalized score estimator is zero for every draw, while the shown pathwise estimator still fluctuates. Variance rankings depend on the estimator and target.' : 'More draws reduce typical Monte Carlo error, not necessarily the realized error for every seed. Neither estimator changes q until an optimizer applies its gradient.'}</p>
    <p>Pathwise gradients use the exact Gaussian entropy derivative. Score gradients use log p−log q with baseline 0. MCSE below is the sample standard error of each independent contribution average at fixed q, not posterior uncertainty or an optimization-convergence guarantee.</p>
    <Table caption="Inspect gradient MCSE and per-draw ingredients" headings={['Quantity', 'Value']} rows={[["Pathwise mean MCSE", fmt(model.estimates.pathMean.mcse)], ['Pathwise log-sd MCSE', fmt(model.estimates.pathLogSd.mcse)], ['Score mean MCSE', fmt(model.estimates.scoreMean.mcse)], ['Score log-sd MCSE', fmt(model.estimates.scoreLogSd.mcse)], ['Current log(p/q)', fmt(row.logRatio)], ['Current score mean contribution', fmt(row.scoreMean)], ['Current score log-sd contribution', fmt(row.scoreLogSd)]]} />
  </section>;
}
export function AmortizationFigure() {
  return <figure className="vi-amortization"><figcaption>One shared rule, three different posterior distributions</figcaption><div className="vi-encoder-rule"><strong>Shared encoder parameters</strong><span>m(x)=0.5x; variance=0.5</span><small>Prior variance 1 + observation-noise variance 1</small></div><div className="vi-encoder-cases">{[-2, 0, 2].map(value => <div key={value}><span>Observation x={value}</span><b aria-hidden="true">↓</b><strong>q(θ|x)=N({value / 2},0.5)</strong><small>New measurement variance<br />1+0.5=1.5</small></div>)}</div><p>Each column is a separate inference case with its own latent θ. N(mean, variance) notation is used here. Shared computation does not mean identical posterior parameters.</p></figure>;
}
