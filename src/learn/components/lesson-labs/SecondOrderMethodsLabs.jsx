import { useId, useMemo, useState } from 'react';
import { bernoulliGeometry, curvatureTrace, kfacFactorState, lbfgsHistoryTrace, safeguardedNewtonState, shampooMatrixState } from '../../data/second-order-methods-models.js';
import './second-order-methods-labs.css';
const format = value => value === null ? 'undefined' : value === 0 ? '0' : Math.abs(value) < 0.0001 || Math.abs(value) > 10000 ? value.toExponential(3) : Number(value.toFixed(5)).toString();
const formatProbability = value => value < 1 && value > 0.9999 ? `1 − ${format(1 - value)}` : format(value);
function Range({
  label,
  value,
  minimum,
  maximum,
  step = 1,
  onChange
}) {
  const id = useId();
  return <div className="second-order-field"><label htmlFor={id}>{label}: {format(value)}</label><input id={id} aria-label={label} type="range" min={minimum} max={maximum} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></div>;
}
function Readout({
  values
}) {
  return <dl className="second-order-readout">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Matrix({
  label,
  matrix,
  scaleMaximum,
  columnLabels,
  rowLabels
}) {
  const maximum = scaleMaximum ?? Math.max(1e-12, ...matrix.flat().map(Math.abs));
  return <figure className="second-order-matrix"><figcaption>{label}</figcaption><div className="second-order-matrix-grid" style={{
      '--columns': matrix[0].length
    }} role="table" aria-label={label}>
    {columnLabels && <div role="row" className="second-order-matrix-row">{columnLabels.map(value => <span role="columnheader" className="second-order-matrix-heading" key={value}>{value}</span>)}</div>}
    {matrix.map((row, i) => <div role="row" className="second-order-matrix-row" key={i}>{row.map((value, j) => <span role="cell" key={`${i}-${j}`} aria-label={`${rowLabels?.[i] ?? `row ${i + 1}`}, ${columnLabels?.[j] ?? `column ${j + 1}`}: ${format(value)}`} style={{
        backgroundColor: value >= 0 ? `rgba(226,181,90,${0.06 + 0.26 * Math.abs(value) / maximum})` : `rgba(115,171,204,${0.06 + 0.26 * Math.abs(value) / maximum})`
      }}>{format(value)}</span>)}</div>)}
  </div></figure>;
}
function LineChart({
  title,
  description,
  samples,
  series,
  xKey,
  xLabel,
  yLabel,
  selectedX = null
}) {
  const id = useId();
  if (!samples.length) return <p>No finite direction is available to plot.</p>;
  const values = samples.flatMap(sample => series.map(item => sample[item.key]));
  const minimum = Math.min(0, ...values);
  const maximum = Math.max(1e-12, ...values);
  const xMaximum = Math.max(1e-12, ...samples.map(sample => sample[xKey]));
  const x = value => 45 + 235 * value / xMaximum;
  const y = value => 185 - 150 * (value - minimum) / (maximum - minimum);
  return <figure className="second-order-chart"><svg viewBox="0 0 310 238" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
    <title id={`${id}-title`}>{title}</title><desc id={`${id}-desc`}>{description} Horizontal axis: {xLabel}. Vertical axis: {yLabel}. Values are calculated from the stated model; axes rescale when controls change.</desc>
    <line x1="45" x2="280" y1="185" y2="185" /><line x1="45" x2="45" y1="35" y2="185" />
    <text x="45" y="20">{yLabel}</text><text x="47" y="49">{format(maximum)}</text><text x="47" y="180">{format(minimum)}</text>
    <text x="45" y="205">0</text><text x="280" y="205" textAnchor="end">{format(xMaximum)}</text><text x="160" y="230" textAnchor="middle">{xLabel}</text>
    {series.map(item => <path key={item.key} d={samples.map((sample, index) => `${index ? 'L' : 'M'}${x(sample[xKey])},${y(sample[item.key])}`).join(' ')} stroke={item.color} strokeWidth="2" fill="none" strokeDasharray={item.dashed ? '5 4' : undefined} />)}
    {selectedX !== null && <line x1={x(selectedX)} x2={x(selectedX)} y1="35" y2="185" stroke="#d4e1e9" strokeDasharray="3 4" />}
  </svg><figcaption>{series.map(item => <span key={item.key} style={{
        color: item.color
      }}>{item.label}{item.dashed ? ' (dashed)' : ' (solid)'} </span>)} · Calculated; linear axes rescale.</figcaption></figure>;
}
export function CurvatureGeometryLab() {
  const id = useId();
  const [angle, setAngle] = useState(0);
  const [curvature, setCurvature] = useState(200);
  const [rateFraction, setRateFraction] = useState(1.6);
  const [step, setStep] = useState(0);
  const state = useMemo(() => curvatureTrace(curvature, angle, rateFraction, 24), [curvature, angle, rateFraction]);
  const coordinate = value => 160 + value * 82;
  const path = points => points.map((point, index) => `${index ? 'L' : 'M'}${coordinate(point[0])},${coordinate(-point[1])}`).join(' ');
  const contours = [0.5, 2, 8, 32, 100].map(level => Array.from({
    length: 121
  }, (_, index) => {
    const phase = 2 * Math.PI * index / 120;
    const local = [Math.sqrt(2 * level / curvature) * Math.cos(phase), Math.sqrt(level) * Math.sin(phase)];
    return state.rotation.map(row => row[0] * local[0] + row[1] * local[1]);
  }));
  const current = state.frames[step];
  return <section className="second-order-lab" aria-label="Curvature and update geometry"><h3>Cross the valley, then travel along it</h3><p>Inspect which coordinate will settle first. Rotate the valley: does dividing each raw coordinate by its diagonal entry still describe Newton?</p>
    <div className="second-order-controls"><Range label="Steep curvature" value={curvature} minimum={2} maximum={200} step={2} onChange={value => {
        setCurvature(value);
        setStep(0);
      }} /><Range label="Valley rotation in degrees" value={angle} minimum={0} maximum={90} step={5} onChange={value => {
        setAngle(value);
        setStep(0);
      }} /><Range label="Rate times steep curvature" value={rateFraction} minimum={0.1} maximum={1.9} step={0.1} onChange={value => {
        setRateFraction(value);
        setStep(0);
      }} /></div>
    <figure className="second-order-plane"><svg viewBox="0 0 320 335" role="img" aria-labelledby={`${id}-title ${id}-desc`}><title id={`${id}-title`}>Equal-scale quadratic contours and actual optimizer paths</title><desc id={`${id}-desc`}>Both coordinate axes use the same scale from minus 1.5 to 1.5. The gold connected points show gradient descent from (1,1). The dashed blue segment is the single Newton direction to (0,0). Current gradient-descent point: {current.point.map(format).join(', ')}.</desc><defs><clipPath id={`${id}-clip`}><rect x="37" y="37" width="246" height="246" /></clipPath></defs><g clipPath={`url(#${id}-clip)`}>{contours.map((points, index) => <path key={index} d={path(points)} fill="none" stroke="#5b626b" />)}<path d={path([[1, 1], [0, 0]])} stroke="#84bcd7" strokeDasharray="6 4" fill="none" strokeWidth="2" /><path d={path(state.frames.slice(0, step + 1).map(frame => frame.point))} stroke="#e2b55a" fill="none" strokeWidth="2" /></g><line x1="37" x2="283" y1="160" y2="160" /><line x1="160" x2="160" y1="37" y2="283" /><circle cx={coordinate(current.point[0])} cy={coordinate(-current.point[1])} r="5" fill="#e2b55a" /><circle cx="160" cy="160" r="4" fill="#84bcd7" /><text x="242" y="64" textAnchor="end">start (1,1)</text><text x="166" y="178">minimum</text><text x="160" y="311" textAnchor="middle">x · equal coordinate units</text><text x="10" y="30">y</text>{[-1, 1].map(value => <g key={value}><text x={coordinate(value)} y="181" textAnchor="middle">{value}</text><text x="142" y={coordinate(-value) + 5} textAnchor="end">{value}</text></g>)}</svg><figcaption>Gold: gradient descent. Dashed blue: Newton. Grey curves have f=.5,2,8,32,100; parts outside the viewport are not drawn. These are exact quadratic level sets.</figcaption></figure>
    <div className="second-order-actions"><button type="button" disabled={step === 0} onClick={() => setStep(value => value - 1)}>Previous update</button><button type="button" disabled={step === 24} onClick={() => setStep(value => value + 1)}>Next update</button><button type="button" onClick={() => {
        setAngle(0);
        setCurvature(200);
        setRateFraction(1.6);
        setStep(0);
      }}>Reset geometry</button></div>
    <Readout values={[[`GD after ${step} updates`, `(${current.point.map(format).join(', ')})`], ['Actual f', format(current.objective)], ['Learning rate', format(state.rate)], ['Hessian condition number', format(state.conditionNumber)]]} /><p>The rate is held fixed during the trace. The Newton solve reaches the minimum in one full step because this objective is an exact positive definite quadratic, including after rotation.</p>
  </section>;
}
export function NewtonSafeguardLab() {
  const [x, setX] = useState(0.2);
  const [damping, setDamping] = useState(1.2);
  const state = useMemo(() => safeguardedNewtonState(x, damping), [x, damping]);
  return <section className="second-order-lab" aria-label="Newton direction and step acceptance"><h3>A useful direction still needs a step check</h3><p>Start at (x,.5). Damping changes the direction; backtracking changes how far to follow it. Investigate why the positive-curvature near-flat example needs a shorter step.</p>
    <div className="second-order-actions"><button type="button" onClick={() => {
        setX(0.2);
        setDamping(0);
      }}>Inspect an indefinite model</button><button type="button" onClick={() => {
        setX(0.6);
        setDamping(0.01);
      }}>Try a near-flat model</button><button type="button" onClick={() => {
        setX(0.2);
        setDamping(1.2);
      }}>Reset safeguard</button></div>
    <div className="second-order-controls"><Range label="Initial x" value={x} minimum={-1.4} maximum={1.4} step={0.05} onChange={setX} /><Range label="Newton damping" value={damping} minimum={0} maximum={3} step={0.01} onChange={setDamping} /></div>
    <LineChart title="Actual objective and its local quadratic along a fixed direction" description="The solid gold line evaluates the original double-well objective. The dashed blue line evaluates its undamped Taylor quadratic along the same computed direction. A vertical line marks the accepted multiplier when backtracking finds one." samples={state.samples} series={[{
      key: 'actual',
      label: 'Actual f',
      color: '#e2b55a'
    }, {
      key: 'model',
      label: 'Taylor model',
      color: '#84bcd7',
      dashed: true
    }]} xKey="multiplier" xLabel="step multiplier α" yLabel="objective value" selectedX={state.accepted?.multiplier ?? null} />
    <Readout values={[['Shifted eigenvalues', state.shifted.map(format).join(', ')], ['Direction d', state.direction ? state.direction.map(format).join(', ') : 'Numerically singular'], ['Directional slope gᵀd', format(state.slope)], ['Accepted multiplier', state.accepted ? format(state.accepted.multiplier) : 'None'], ['Actual starting loss', format(state.initialValue)], ['Actual accepted loss', state.accepted ? format(state.accepted.value) : 'No accepted step']]} />
    <p>{state.singular ? 'The shifted system is numerically singular; no direction is plotted.' : !state.positiveDefinite ? 'The quadratic model is indefinite. A direction can happen to decrease total loss while moving its x-coordinate toward the saddle. This is not a positive definite Newton model and has no SPD descent guarantee.' : 'The shifted system is positive definite. Its direction descends locally; the actual loss, not the quadratic prediction, determines accepted distance.'}</p>
    {state.attempts.length > 0 && <div className="second-order-attempts" aria-label="Backtracking attempts">{state.attempts.map(attempt => <div key={attempt.multiplier}><strong>α={format(attempt.multiplier)}</strong><span>f={format(attempt.value)}</span><span>{attempt.passes ? 'Accept: sufficient decrease' : 'Reject: shrink again'}</span></div>)}</div>}
    <p>Armijo uses c=.0001, halving and at most 20 trials. For inspection, a descending direction from an indefinite system can also be tested; an implementation requiring an SPD model would reject that system before line search.</p>
  </section>;
}
export function SecantCorrespondenceFigure() {
  const id = useId();
  return <figure className="second-order-inline"><svg className="second-order-secant" viewBox="0 0 320 252" role="img" aria-labelledby={`${id}-title ${id}-desc`}><title id={`${id}-title`}>One displacement and its measured gradient change</title><desc id={`${id}-desc`}>A common coordinate scale shows displacement s=(1,0) in gold and gradient change y=(4,1) in blue for H=[[4,1],[1,2]]. The inverse approximation must map the blue vector to the gold vector. Position and gradient have distinct meanings even though their numerical coordinates share a display scale.</desc><defs><marker id={`${id}-gold`} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6Z" fill="#e2b55a" /></marker><marker id={`${id}-blue`} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6Z" fill="#84bcd7" /></marker></defs><line x1="40" x2="292" y1="150" y2="150" stroke="#667782" /><line x1="40" x2="40" y1="165" y2="55" stroke="#667782" /><path d="M40 150 L99 150" fill="none" stroke="#e2b55a" strokeWidth="3" markerEnd={`url(#${id}-gold)`} /><path d="M40 150 L276 91" fill="none" stroke="#84bcd7" strokeWidth="3" markerEnd={`url(#${id}-blue)`} /><text x="108" y="174">s = (1,0)</text><text x="178" y="69">y = (4,1)</text><text x="20" y="171">0</text><text x="40" y="25">Observed: Hs = y</text><text x="40" y="209">Required inverse action:</text><text x="40" y="235">M new × (4,1) = (1,0)</text></svg><figcaption>For one concrete pair, moving by s=(1,0) changes the gradient by y=(4,1). The inverse model must send y back to s. A single arrow correspondence does not determine its action on every other vector.</figcaption></figure>;
}
export function LimitedMemoryLab() {
  const [memory, setMemory] = useState(2);
  const [scaled, setScaled] = useState(true);
  const [badPair, setBadPair] = useState(false);
  const [step, setStep] = useState(0);
  const id = useId();
  const state = useMemo(() => lbfgsHistoryTrace(memory, scaled, badPair), [memory, scaled, badPair]);
  const currentStep = Math.min(step, state.frames.length - 1);
  const current = state.frames[currentStep];
  return <section className="second-order-lab" aria-label="L-BFGS history and two-loop recursion"><h3>Spend memory on observed changes</h3><p>These pairs came from H=[[4,1],[1,2]]. The current gradient is (3,1). Inspect whether retaining only the latest direction identifies the inverse action on this different gradient.</p>
    <Range label="Retained history budget" value={memory} minimum={0} maximum={3} onChange={value => {
      setMemory(value);
      setStep(0);
    }} /><div className="second-order-controls"><label><input type="checkbox" checked={scaled} onChange={event => {
          setScaled(event.target.checked);
          setStep(0);
        }} /> Scale initial inverse by γ=sᵀy/yᵀy</label><label><input type="checkbox" checked={badPair} onChange={event => {
          setBadPair(event.target.checked);
          setStep(0);
        }} /> Reverse the last gradient change</label></div>
    <ol className="second-order-history">{state.checkedPairs.map(pair => <li key={pair.name} className={state.pairs.some(retained => retained.name === pair.name) ? 'is-retained' : ''}><strong>{pair.name}</strong><span>s=({pair.step.join(',')})</span><span>y=({pair.change.join(',')})</span><span>sᵀy={format(pair.curvature)}</span><em>{!pair.accepted ? 'Rejected: nonpositive curvature' : state.pairs.some(retained => retained.name === pair.name) ? 'Retained' : 'Outside retained window'}</em></li>)}</ol>
    <div className="second-order-vector-stage" aria-live="polite"><p>Transformation {currentStep + 1} of {state.frames.length}: <strong>{current.operation}</strong></p><div className="second-order-vector-cells">{current.vector.map((value, index) => <div key={index}><span>{index === 0 ? 'first coordinate' : 'second coordinate'}</span><strong>{format(value)}</strong><div className="second-order-signed-track"><span className={value < 0 ? 'is-negative' : ''} style={{
              width: `${Math.min(48, Math.abs(value) * 12)}%`,
              left: value < 0 ? `${50 - Math.min(48, Math.abs(value) * 12)}%` : '50%'
            }} /></div></div>)}</div><p>{current.pair ?? 'No pair consumed at this stage'}{current.scalar !== null ? ` · coefficient ${format(current.scalar)}` : ''}</p></div>
    <div className="second-order-actions"><button type="button" disabled={currentStep === 0} onClick={() => setStep(value => value - 1)}>Previous transformation</button><button type="button" disabled={currentStep === state.frames.length - 1} onClick={() => setStep(value => value + 1)}>Next transformation</button><button type="button" onClick={() => {
        setMemory(2);
        setScaled(true);
        setBadPair(false);
        setStep(0);
      }}>Reset history</button></div>
    <Readout values={[['Final L-BFGS direction −r', state.direction.map(format).join(', ')], ['Exact Newton direction', state.exactNewtonDirection.map(format).join(', ')], ['Starting scale γ', format(state.gamma)]]} /><details><summary id={`${id}-reference`}>Inspect the tiny dense verification matrix</summary><Matrix label="M: inverse approximation built from retained pairs" matrix={state.inverseApproximation} /><p>M times the current gradient is ({state.denseProduct.map(format).join(', ')}), agreeing with the two-loop result. Real L-BFGS obtains this product without constructing M.</p></details><p>Bars show signed coordinate magnitude on a common −4 to 4 scale; their numbers are authoritative. These are algebraic operations on one gradient, not successive optimizer iterations. A positive-curvature pair is rejected here only when it fails the stated threshold; this is a bounded teaching policy.</p>
  </section>;
}
export function NaturalGradientLab() {
  const [probability, setProbability] = useState(0.2);
  const [fraction, setFraction] = useState(0.25);
  const state = useMemo(() => bernoulliGeometry(probability, 0.8, fraction), [probability, fraction]);
  const distributions = [['Before', probability], ['Direct probability update', state.directNext], ['Logit update mapped back', state.logitNext]];
  return <section className="second-order-lab" aria-label="Natural gradient probability geometry"><h3>Same tangent, different finite endpoints</h3><p>The observations have target success fraction .8. Explore how closely the two coordinate updates agree when you shrink the step fraction.</p><div className="second-order-controls"><Range label="Starting success probability" value={probability} minimum={0.05} maximum={0.95} step={0.05} onChange={setProbability} /><Range label="Natural step fraction" value={fraction} minimum={0.01} maximum={1} step={0.01} onChange={setFraction} /></div>
    <div className="second-order-probabilities">{distributions.map(([label, value]) => <div key={label}><strong>{label}: p={formatProbability(value)}</strong><div className="second-order-probability-bar" role="img" aria-label={`${label}: success probability ${formatProbability(value)}, failure probability ${formatProbability(1 - value)}`}><span style={{
            width: `${100 * value}%`
          }} /><span style={{
            width: `${100 * (1 - value)}%`
          }} /></div><span>success {formatProbability(value)} · failure {formatProbability(1 - value)}</span></div>)}</div>
    <Readout values={[['Probability direction', format(state.probabilityDirection)], ['Mapped logit tangent', format(state.mappedLogitDirection)], ['True probability Fisher', format(state.probabilityFisher)], ['Observed-label outer product', format(state.empiricalFisher)], ['Exact KL: direct update', format(state.directKl)], ['Exact KL: logit update', format(state.logitKl)]]} />
    <LineChart title="Exact Bernoulli KL and the local Fisher approximation" description="Both curves follow the direct probability update from the fixed initial distribution toward target .8. The dashed curve is a local quadratic approximation, not a measured divergence." samples={state.samples} series={[{
      key: 'exact',
      label: 'Exact KL',
      color: '#e2b55a'
    }, {
      key: 'local',
      label: 'Local ½FΔp²',
      color: '#84bcd7',
      dashed: true
    }]} xKey="step" xLabel="fraction along direct update" yLabel="KL in natural-log units" /><button type="button" onClick={() => {
      setProbability(0.2);
      setFraction(0.25);
    }}>Reset probability geometry</button><p>Gold denotes success and blue failure; text gives both masses. Numerical readouts are rounded; an interior probability near one is written as one minus its small remaining failure mass. A unit fraction puts the direct p-update at the target, but a unit logit update need not. The invariant object is the infinitesimal distribution change, not an arbitrary finite Euler step.</p>
  </section>;
}
export function FisherFactorLab() {
  const [strength, setStrength] = useState(1);
  const [damping, setDamping] = useState(0.1);
  const [view, setView] = useState('exact');
  const id = useId();
  const state = useMemo(() => kfacFactorState(strength, damping), [strength, damping]);
  const displayed = view === 'exact' ? state.exactFisher : view === 'factored' ? state.factoredFisher : state.difference;
  const maximum = Math.max(...state.exactFisher.flat().map(Math.abs), ...state.factoredFisher.flat().map(Math.abs));
  return <section className="second-order-lab" aria-label="K-FAC expectation factorization"><h3>What changes when an expectation is split?</h3><p>Two inputs, (1,−1) and (1,2), are equally likely. The first coordinate is a bias input. At zero layer strength both outputs have probability .5 for both inputs. Explore when the factorization error vanishes.</p><div className="second-order-controls"><Range label="Layer strength" value={strength} minimum={0} maximum={2} step={0.1} onChange={setStrength} /><Range label="Full diagonal damping" value={damping} minimum={0.01} maximum={1} step={0.01} onChange={setDamping} /></div>
    <div className="second-order-factor-pair"><Matrix label="A = mean input outer product" matrix={state.inputFactor} /><Matrix label="S = mean model score covariance" matrix={state.outputFactor} /></div>
    <div className="second-order-field"><label htmlFor={id}>Inspect the 4×4 block</label><select id={id} value={view} onChange={event => setView(event.target.value)}><option value="exact">Exact Fisher: mean of Kronecker products</option><option value="factored">K-FAC: Kronecker product of means</option><option value="difference">K-FAC minus exact Fisher</option></select></div>
    <Matrix label={view === 'exact' ? 'Exact F' : view === 'factored' ? 'Approximation A ⊗ S' : 'Approximation error in the same cell scale'} matrix={displayed} scaleMaximum={maximum} columnLabels={['w₁₁', 'w₂₁', 'w₁₂', 'w₂₂']} rowLabels={['w₁₁', 'w₂₁', 'w₁₂', 'w₂₂']} />
    <p>Rows use the same weight order as columns. Gold cells are nonnegative, blue cells negative; numbers carry sign and magnitude. The three block views share a cell scale. Factor panels use their own labelled numerical values.</p><Readout values={[['Frobenius approximation error', format(state.approximationError)], ['Exact (F+λI) direction', state.exactDirection.map(format).join(', ')], ['Approximate (A⊗S+λI) direction', state.factoredDirection.map(format).join(', ')], ['Separately damped factor direction', state.factorDampedDirection.map(format).join(', ')]]} /><p>The third direction instead uses (A+√λI)⊗(S+√λI), which adds cross terms. It is shown to expose a different damping convention, not to claim that one is universally better. The plotted matrices are curvature blocks; damping is applied only to the direction calculations.</p><button type="button" onClick={() => {
      setStrength(1);
      setDamping(0.1);
      setView('exact');
    }}>Reset Fisher factors</button>
  </section>;
}
export function ShampooContractionFigure() {
  return <figure className="second-order-inline"><div className="second-order-contraction-example"><Matrix label="One gradient G" matrix={[[1, 2], [0, 1]]} /><div><p><strong>Row pairing → (GGᵀ)₁₂</strong></p><div className="second-order-dot-pair"><span>(1, 2)</span><span>·</span><span>(0, 1)</span></div><p>1×0 + 2×1 = <strong>2</strong></p><p><strong>Column pairing → (GᵀG)₁₂</strong></p><div className="second-order-dot-pair"><span>(1, 0)</span><span>·</span><span>(2, 1)</span></div><p>1×2 + 0×1 = <strong>2</strong></p></div></div><figcaption>The same gradient supports two different contractions: sum matching column positions to compare rows, or matching row positions to compare columns. Both off-diagonal values happen to be 2 here; their diagonal values differ.</figcaption></figure>;
}
export function ShampooMatrixLab() {
  const [updates, setUpdates] = useState(1);
  const [epsilon, setEpsilon] = useState(0.1);
  const [rotation, setRotation] = useState(0);
  const state = useMemo(() => shampooMatrixState(updates, epsilon, rotation), [updates, epsilon, rotation]);
  const current = state.current;
  return <section className="second-order-lab" aria-label="Shampoo matrix accumulation and inverse roots"><h3>Accumulate relationships across rows and columns</h3><p>Step through three fixed gradient matrices. Inspect which off-diagonal relationships survive in each accumulator. Rotating row coordinates changes the representation; it should rotate the resulting direction in the same way.</p><div className="second-order-actions"><button type="button" disabled={updates === 1} onClick={() => setUpdates(value => value - 1)}>Previous gradient</button><button type="button" disabled={updates === 3} onClick={() => setUpdates(value => value + 1)}>Accumulate next gradient</button><button type="button" onClick={() => {
        setUpdates(1);
        setEpsilon(0.1);
        setRotation(0);
      }}>Reset Shampoo</button></div><div className="second-order-controls"><Range label="Accumulator epsilon" value={epsilon} minimum={0.01} maximum={1} step={0.01} onChange={setEpsilon} /><Range label="Row rotation in degrees" value={rotation} minimum={0} maximum={90} step={5} onChange={setRotation} /></div>
    <Matrix label={`Current gradient G${updates}`} matrix={current.gradient} /><div className="second-order-factor-pair"><Matrix label="L = εI + sum GGᵀ" matrix={current.left} /><Matrix label="R = εI + sum GᵀG" matrix={current.right} /></div><div className="second-order-factor-pair"><Matrix label="Spectral L to power −¼" matrix={current.leftRoot.matrix} /><Matrix label="Spectral R to power −¼" matrix={current.rightRoot.matrix} /></div><Matrix label="Transformed gradient L⁻¼ G R⁻¼" matrix={current.direction} /><Readout values={[['Eigenvalues of L', current.leftRoot.eigenvalues.map(format).join(', ')], ['Eigenvalues of R', current.rightRoot.eigenvalues.map(format).join(', ')]]} /><p>Subtract a learning rate times this transformed gradient to update a parameter matrix. The original gradient replay is fixed, so this view isolates accumulation and preconditioning; it does not compare the training loss of different optimizers. Individual matrix colors use their own scale and are not cross-matrix magnitude comparisons.</p>
  </section>;
}
