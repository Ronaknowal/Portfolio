import { useState } from 'react';
import { factorValue, interpolationValue, landscapeNumber as number, modePaths, saddleNoiseState, stationaryPresets, stationaryState, stationaryValue, symmetryState, wellState, wellValue } from '../../data/nonconvex-landscape-models.js';
import './nonconvex-landscape-labs.css';
function Range({
  label,
  value,
  set,
  min,
  max,
  step
}) {
  return <label className="landscape-field"><span>{label}: <strong>{number(value)}</strong></span><input type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={event => set(Number(event.target.value))} /></label>;
}
function Readout({
  rows
}) {
  return <dl className="landscape-readout">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Plot({
  title,
  children
}) {
  return <svg className="landscape-plot" viewBox="0 0 340 276" role="img" aria-label={title}><title>{title}</title>{children}</svg>;
}
function path(points, x, y) {
  return points.map(([a, b], index) => `${index ? 'L' : 'M'}${x(a).toFixed(3)},${y(b).toFixed(3)}`).join(' ');
}
function sample(fn, low, high, count = 121) {
  return Array.from({
    length: count
  }, (_, index) => {
    const input = low + (high - low) * index / (count - 1);
    return [input, fn(input)];
  });
}
function CurveAxes({
  xLabel,
  yLabel,
  xTicks,
  yTicks,
  x,
  y
}) {
  return <><line x1="48" x2="313" y1="224" y2="224" className="landscape-axis" /><line x1="48" x2="48" y1="36" y2="224" className="landscape-axis" />{xTicks.map(value => <text key={value} x={x(value)} y="245" textAnchor="middle" className="landscape-tick">{number(value)}</text>)}{yTicks.map(value => <g key={value}><line x1="48" x2="313" y1={y(value)} y2={y(value)} className="landscape-grid" /><text x="41" y={y(value) + 4} textAnchor="end" className="landscape-tick">{number(value)}</text></g>)}<text x="48" y="22" className="landscape-label">{yLabel}</text><text x="310" y="269" textAnchor="end" className="landscape-label">{xLabel}</text></>;
}
function StepControls({
  step,
  setStep,
  last
}) {
  return <div className="landscape-buttons"><button onClick={() => setStep(0)} disabled={step === 0}>Restart trace</button><button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>Previous step</button><button onClick={() => setStep(Math.min(last, step + 1))} disabled={step === last}>Next step</button><button onClick={() => setStep(last)} disabled={step === last}>Show final step</button><output>Step {step} / {last}</output></div>;
}
export function WellBasinLab() {
  const [tilt, setTilt] = useState(0.15);
  const [initial, setInitial] = useState(0.8);
  const [rate, setRate] = useState(0.12);
  const [step, setStep] = useState(0);
  const state = wellState(tilt, initial, rate);
  const active = state.frames[step];
  const x = value => 48 + (value + 1.5) * 265 / 3;
  const y = value => 224 - (value + 0.35) * 188 / 1.2;
  const change = setter => value => {
    setter(value);
    setStep(0);
  };
  return <section className="landscape-lab" aria-label="Competing wells investigation" data-lab="wells" data-state={JSON.stringify({
    tilt,
    initial,
    rate,
    ...active
  })}>
    <p className="landscape-eyebrow">INVESTIGATE · LOCATION VERSUS VALUE</p><h3>Does this run reach the better well?</h3>
    <p>Inspect where a start at x=0.8 will go. The two minima belong to the same objective; the green rings mark both. Positive tilt lowers the left well. Inputs reset the trace immediately.</p>
    <div className="landscape-controls"><Range label="Tilt δ" value={tilt} set={change(setTilt)} min={-0.25} max={0.25} step={0.05} /><Range label="Initial x" value={initial} set={change(setInitial)} min={-1.5} max={1.5} step={0.05} /><Range label="Learning rate η" value={rate} set={change(setRate)} min={0} max={0.3} step={0.01} /></div>
    <div className="landscape-pair"><figure><Plot title="Tilted quartic objective with stationary points and actual gradient descent iterates"><CurveAxes xLabel="parameter x" yLabel="cost Wδ(x)" xTicks={[-1, 0, 1]} yTicks={[-0.25, 0.25, 0.75]} x={x} y={y} /><path d={path(sample(value => wellValue(value, tilt), -1.5, 1.5), x, y)} className="landscape-line blue" />{state.critical.map((point, index) => <circle key={index} cx={x(point.x)} cy={y(point.value)} r="6" className={`landscape-ring ${index === 1 ? 'pink' : 'green'}`} />)}{state.frames.slice(0, step + 1).map(frame => <circle key={frame.step} cx={x(frame.x)} cy={y(frame.value)} r="2.5" className="landscape-dot amber" />)}<circle cx={x(active.x)} cy={y(active.value)} r="6" className="landscape-dot amber" /></Plot><figcaption>Calculated Wδ curve; green rings: minima, pink ring: maximum, amber: this run. All panels use dimensionless parameters and cost. Lines between curve samples are visual interpolation.</figcaption></figure><Readout rows={[['Current x; gradient', `${number(active.x)}; ${number(active.gradient)}`], ['Current cost', number(active.value)], ['Left minimum: x; cost', `${number(state.critical[0].x)}; ${number(state.critical[0].value)}`], ['Right minimum: x; cost', `${number(state.critical[2].x)}; ${number(state.critical[2].value)}`], ['Lowest value among all minima', number(state.best)], ['Next update', `x − ${number(rate)} × gradient`]]} /></div>
    <StepControls step={step} setStep={setStep} last={state.frames.length - 1} />
    <p role="status">{rate === 0 ? 'The update is exactly zero because η=0; that says nothing about stationary geometry.' : `This is step ${step} of a fixed gradient-descent run. A small gradient near the higher well does not certify the global minimum.`}</p>
    <p className="landscape-transfer">Change the tilt to −0.15, then start at −0.8. Which well is now better? Can the same stopping rule confuse local success with global success?</p>
  </section>;
}
export function StationaryGeometryLab() {
  const [kind, setKind] = useState('flatSaddle');
  const [degrees, setDegrees] = useState(90);
  const [radius, setRadius] = useState(0.5);
  const state = stationaryState(kind, degrees, radius);
  const px = value => 180 + value * 88;
  const py = value => 130 - value * 88;
  const sx = value => 48 + (value + 1) * 265 / 2;
  const sy = value => 130 - value * 87;
  const cells = Array.from({
    length: 21 * 21
  }, (_, index) => {
    const x = -1 + 2 * (index % 21) / 20;
    const y = -1 + 2 * Math.floor(index / 21) / 20;
    const value = stationaryValue(kind, x, y);
    return {
      x,
      y,
      value
    };
  });
  const slice = t => stationaryValue(kind, t * state.direction[0], t * state.direction[1]);
  return <section className="landscape-lab" aria-label="Stationary geometry investigation" data-lab="stationary" data-state={JSON.stringify(state)}>
    <p className="landscape-eyebrow">INVESTIGATE · WHAT ZERO CURVATURE HIDES</p><h3>Same gradient. Sometimes the same Hessian. Different answer.</h3>
    <p>Every selected function has gradient zero at the origin. Inspect whether the chosen direction rises or falls before reading its exact change.</p>
    <div className="landscape-controls"><label className="landscape-field"><span>Surface at the origin</span><select aria-label="Surface at the origin" value={kind} onChange={event => setKind(event.target.value)}>{Object.entries(stationaryPresets).map(([key, value]) => <option key={key} value={key}>{value.label}</option>)}</select></label><Range label="Direction angle in degrees" value={degrees} set={setDegrees} min={0} max={360} step={15} /><Range label="Distance r" value={radius} set={setRadius} min={0} max={1} step={0.05} /></div>
    <div className="landscape-pair"><figure><Plot title="Signed function values sampled on a square grid and selected line through the origin">{cells.map((cell, index) => <rect key={index} x={px(cell.x) - 4.4} y={py(cell.y) - 4.4} width="8.8" height="8.8" fill={cell.value < 0 ? '#72b4f1' : '#f3c36c'} opacity={cell.value === 0 ? 0 : 0.14 + 0.65 * Math.min(1, Math.abs(cell.value) / 2)} />)}<line x1="80" x2="280" y1={py(0)} y2={py(0)} className="landscape-axis" /><line x1={px(0)} x2={px(0)} y1="30" y2="230" className="landscape-axis" /><line x1={px(-state.direction[0])} y1={py(-state.direction[1])} x2={px(state.direction[0])} y2={py(state.direction[1])} className="landscape-line green" /><circle cx={px(state.point[0])} cy={py(state.point[1])} r="6" className="landscape-dot green" /><circle cx={px(0)} cy={py(0)} r="3" fill="#fff" /><text x="77" y="248" className="landscape-tick">x=−1</text><text x="243" y="248" className="landscape-tick">x=1</text><text x="287" y="46" className="landscape-tick">y=1</text><text x="287" y="228" className="landscape-tick">y=−1</text><text x="48" y="20" className="landscape-label">Above / below the origin’s value</text></Plot><figcaption>21×21 samples over [−1,1]²: amber positive, blue negative, blank zero. Opacity uses |f| on the same 0–2 scale in every preset. Green: selected direction and point. These are samples, not exact contour boundaries.</figcaption></figure><figure><Plot title="Exact directional slice compared with its second-order Taylor approximation"><CurveAxes xLabel="signed distance t" yLabel="change f(tv) − f(0)" xTicks={[-1, 0, 1]} yTicks={[-1, 0, 1]} x={sx} y={sy} /><path d={path(sample(slice, -1, 1), sx, sy)} className="landscape-line amber" /><path d={path(sample(t => 0.5 * t * t * state.direction.reduce((sum, value, index) => sum + value * value * state.eigenvalues[index], 0), -1, 1), sx, sy)} className="landscape-line blue dashed" /><circle cx={sx(radius)} cy={sy(state.actual)} r="5" className="landscape-dot green" /></Plot><figcaption>Amber: exact function on the green line. Blue dashed: quadratic Taylor term. A flat dashed line does not prove that the exact curve is flat.</figcaption></figure></div>
    <Readout rows={[["Hessian diagonal / eigenvalues at the origin", state.eigenvalues.join(', ')], ['Selected point (x,y)', state.point.map(number).join(', ')], ['Exact change; quadratic prediction', `${number(state.actual)}; ${number(state.quadratic)}`], ['Classification using the full function', state.classification]]} />
    <p role="status">{state.reason}</p><p className="landscape-transfer">Switch between x⁴+y⁴ and x⁴−y⁴. Then inspect 45° and 90°. Why can one harmless-looking direction fail to classify the entire point?</p>
  </section>;
}
export function SaddleNoiseLab() {
  const [direction, setDirection] = useState('stable');
  const [rate, setRate] = useState(0.12);
  const [amplitude, setAmplitude] = useState(0.15);
  const [initialY, setInitialY] = useState(0);
  const [step, setStep] = useState(0);
  const state = saddleNoiseState(direction, rate, amplitude, initialY);
  const active = state.frames[Math.min(step, state.frames.length - 1)];
  const change = setter => value => {
    setter(value);
    setStep(0);
  };
  const extent = Math.max(0.75, ...state.frames.flatMap(frame => frame.point.map(Math.abs))) * 1.08;
  const px = value => 180 + 90 * value / extent;
  const py = value => 130 - 90 * value / extent;
  const tx = value => 48 + value * 265 / 24;
  const ty = value => 130 - 85 * value / extent;
  return <section className="landscape-lab" aria-label="Saddle noise investigation" data-lab="noise" data-state={JSON.stringify({
    ...active,
    direction,
    rate,
    amplitude,
    initialY,
    stopped: state.stopped
  })}>
    <p className="landscape-eyebrow">INVESTIGATE · DIRECTION MATTERS</p><h3>Noise in x does not create a y displacement</h3>
    <p>The surface is F=x²−y². Start at (0.6,0); mini-batch gradient noise has a selectable direction. Inspect whether the trajectory can leave the x-axis.</p>
    <div className="landscape-controls"><label className="landscape-field"><span>Sample-noise direction</span><select aria-label="Sample-noise direction" value={direction} onChange={event => change(setDirection)(event.target.value)}><option value="none">No sample noise</option><option value="stable">x-axis only (stable)</option><option value="unstable">y-axis only (unstable)</option><option value="both">Unit diagonal (both axes)</option></select></label><Range label="Saddle learning rate η" value={rate} set={change(setRate)} min={0.02} max={0.25} step={0.01} /><Range label="Noise amplitude a" value={amplitude} set={change(setAmplitude)} min={0} max={0.5} step={0.05} /><Range label="Initial y perturbation" value={initialY} set={change(setInitialY)} min={-0.05} max={0.05} step={0.001} /></div>
    <div className="landscape-pair"><figure><Plot title="Actual trajectory in saddle parameter coordinates"><line x1="80" x2="280" y1="130" y2="130" className="landscape-axis" /><line x1="180" x2="180" y1="30" y2="230" className="landscape-axis" /><path d={path(state.frames.slice(0, active.step + 1).map(frame => frame.point), px, py)} className="landscape-line amber" /><circle cx={px(0.6)} cy={py(initialY)} r="5" className="landscape-ring green" /><circle cx={px(active.point[0])} cy={py(active.point[1])} r="6" className="landscape-dot amber" /><text x="48" y="22" className="landscape-label">Parameter trajectory (equal scales)</text><text x="182" y="246" className="landscape-tick">0</text><text x="48" y="269" className="landscape-label">x →; y ↑; extent ±{number(extent)}</text></Plot><figcaption>Green ring: start; amber: actual visited points. Equal axis scales; extent adapts to the selected full trace and is printed, so compare the readouts across runs.</figcaption></figure><figure><Plot title="Stable x and unstable y coordinates over actual updates"><CurveAxes xLabel="update k" yLabel="coordinate value" xTicks={[0, 12, 24]} yTicks={[0]} x={tx} y={ty} /><text x="61" y="46" className="landscape-tick">range ±{number(extent)}</text><path d={path(state.frames.slice(0, active.step + 1).map(frame => [frame.step, frame.point[0]]), tx, ty)} className="landscape-line blue" /><path d={path(state.frames.slice(0, active.step + 1).map(frame => [frame.step, frame.point[1]]), tx, ty)} className="landscape-line pink" /></Plot><figcaption>Blue: x; pink: y. Multipliers without noise are {number(state.stableFactor)} and {number(state.unstableFactor)}. The vertical scale matches the printed extent, not a measured network benchmark.</figcaption></figure></div>
    <StepControls step={active.step} setStep={setStep} last={state.frames.length - 1} />
    <Readout rows={[["Current (x,y)", active.point.map(number).join(', ')], ['Full objective; full gradient', `${number(active.value)}; (${active.trueGradient.map(number).join(', ')})`], ['Next sign; added gradient noise', `${active.sign === null ? 'sequence complete' : active.sign}; (${active.noise.map(number).join(', ')})`], ['Next sample gradient', active.sampleGradient.map(number).join(', ')]]} />
    <p role="status">{state.stopped && active.step === state.frames.length - 1 ? `Stopped before update ${state.stopped.attemptedStep}: its proposed point (${state.stopped.point.map(number).join(', ')}) leaves |coordinate|≤4. No point was clipped.` : direction === 'stable' && initialY === 0 ? 'Every y remains exactly zero in this recurrence: x-only noise has no component in the unstable direction.' : 'This finite trace shows one outcome, not a probability of success or a global convergence certificate.'}</p>
    <details><summary>See the fixed sample realization and boundary</summary><p>The signs are +,−,−,+,+,+,−,+,−,−,+,−,+,+,−,−,−,+,+,−,+,−,−,+. This is one disclosed possible realization of uniform independent signs, held fixed for comparisons; the browser does not draw random batches. F is an unbounded local saddle model. The trace stops before leaving the declared window.</p></details>
    <p className="landscape-transfer">Turn noise off and set initial y=0.001. Inspect its sign and growth. Then return y to zero and add only x noise. Is “the gradients are noisy” sufficient evidence for escape?</p>
  </section>;
}
export function ReparameterizationLab() {
  const [logScale, setLogScale] = useState(0);
  const [displacement, setDisplacement] = useState(0.15);
  const [direction, setDirection] = useState('normal');
  const state = symmetryState(logScale, displacement, direction);
  const curve = sample(t => factorValue(state.a + t * state.unit[0], state.b + t * state.unit[1]), -0.3, 0.3);
  const top = Math.max(0.0001, ...curve.map(([, value]) => value)) * 1.12;
  const x = value => 48 + (value + 0.3) * 265 / 0.6;
  const y = value => 224 - value * 188 / top;
  return <section className="landscape-lab" aria-label="Equivalent predictor curvature investigation" data-lab="symmetry" data-state={JSON.stringify(state)}>
    <p className="landscape-eyebrow">INVESTIGATE · SAME FUNCTION, DIFFERENT COORDINATES</p><h3>Rescale the factors; keep every prediction</h3>
    <p>For ŷ=abx and F=½(ab−1)², every starting pair (a,b)=(s,1/s) predicts ŷ=x. Change s and compare the loss curvature while the model output stays fixed.</p>
    <div className="landscape-controls"><Range label="Scale exponent log₂(s)" value={logScale} set={setLogScale} min={-3} max={3} step={0.25} /><label className="landscape-field"><span>Unit perturbation direction</span><select aria-label="Unit perturbation direction" value={direction} onChange={event => setDirection(event.target.value)}><option value="normal">Normal (b,a) / √(a²+b²)</option><option value="tangent">Tangent (a,−b) / √(a²+b²)</option></select></label><Range label="Signed displacement ε" value={displacement} set={setDisplacement} min={-0.3} max={0.3} step={0.01} /></div>
    <div className="landscape-pair"><figure><Plot title="Loss under a unit Euclidean parameter perturbation of equivalent factors"><CurveAxes xLabel="parameter displacement ε" yLabel="loss F along this line" xTicks={[-0.3, 0, 0.3]} yTicks={[0]} x={x} y={y} /><path d={path(curve, x, y)} className="landscape-line amber" /><circle cx={x(displacement)} cy={y(state.loss)} r="6" className="landscape-dot green" /><text x="61" y="46" className="landscape-tick">top {number(top)}</text></Plot><figcaption>Calculated loss along the selected straight unit direction. Vertical scale adapts and is printed; compare exact curvature and loss numbers. This curve is not a worst-direction finite-radius sharpness search.</figcaption></figure><Readout rows={[["Equivalent factors (a,b)", `${number(state.a)}, ${number(state.b)}`], ['Unperturbed coefficient ab; training loss', `${number(state.coefficient)}; 0`], ['Unperturbed predictions at x=−2,0,3', '−2, 0, 3'], ['Hessian eigenvalues at the unperturbed pair', state.eigenvalues.map(number).join(', ')], ['Perturbed coefficient; exact loss', `${number(state.perturbedCoefficient)}; ${number(state.loss)}`], ['Quadratic loss prediction', number(state.quadratic)], ['Changing coefficient ab from 1 to 1+ε instead', `loss = ${number(state.functionPerturbationLoss)}, independent of s`]]} /></div>
    <p role="status">{direction === 'normal' ? 'The nonzero curvature is a²+b². It changes with the scale even though the unperturbed predictor remains exactly the same function.' : 'The tangent has zero quadratic curvature. A finite straight step leaves the curved ab=1 path, so a small fourth-order loss can remain.'}</p>
    <p className="landscape-transfer">Compare s=1 and s=4 at ε=0.1. Which statement concerns parameter sensitivity? Which concerns a changed predictor? Would their identical unperturbed predictions permit different test errors on the same test data?</p>
  </section>;
}
export function HiddenUnitPermutationFigure() {
  return <figure className="landscape-inline"><div className="landscape-network"><div><strong>Input x=1.5</strong><p>h₁=ReLU(x−1)=0.5<br />h₂=ReLU(−x+2)=0.5</p><p className="landscape-equation">2h₁ − h₂ = 0.5</p></div><div><strong>Swap whole units</strong><p>new h₁ = old h₂<br />new h₂ = old h₁<br />new output weights = (−1, 2)</p><p className="landscape-equation">−new h₁ + 2new h₂ = 0.5</p></div></div><figcaption>Each hidden unit carries its incoming weight, bias and outgoing connection. Swapping only incoming weights would change the network. The calculation uses the same input; the algebra establishes equivalence for every input.</figcaption></figure>;
}
export function ModePathFigure() {
  const values = sample(t => modePaths(t).straightLoss, 0, 1);
  const x = value => 48 + value * 265;
  const y = value => 224 - value * 188 / 0.01;
  const px = value => 48 + (value - 0.8) * 265 / 1.4;
  const py = value => 224 - (value - 0.3) * 188 / 1;
  const states = Array.from({
    length: 61
  }, (_, index) => modePaths(index / 60));
  return <figure className="landscape-inline"><div className="landscape-pair"><Plot title="Straight and curved parameter paths joining the same two exact minima"><CurveAxes xLabel="factor a" yLabel="factor b" xTicks={[1, 1.5, 2]} yTicks={[0.5, 1]} x={px} y={py} /><path d={path(states.map(state => state.curved), px, py)} className="landscape-line green" /><path d={path(states.map(state => state.straight), px, py)} className="landscape-line amber" />{[[1, 1], [2, 0.5]].map(([a, b]) => <circle key={a} cx={px(a)} cy={py(b)} r="5" className="landscape-dot green" />)}</Plot><Plot title="The straight path has a barrier while the curved path keeps zero loss"><CurveAxes xLabel="path fraction t" yLabel="loss along the chosen path" xTicks={[0, 0.5, 1]} yTicks={[0, 0.01]} x={x} y={y} /><path d={path(values, x, y)} className="landscape-line amber" /><path d={path([[0, 0], [1, 0]], x, y)} className="landscape-line green" /></Plot></div><figcaption>Calculated paths from (1,1) to (2,½). Amber line segment leaves ab=1 and reaches loss 1/128=0.0078125. Green curved path stays on ab=1 and has zero loss throughout. The plotted parameter axes have their own labelled scales.</figcaption></figure>;
}
export function InterpolationFigure() {
  const x = value => 48 + (value + 1.25) * 265 / 2.5;
  const y = value => 224 - (value + 2) * 188 / 4;
  return <figure className="landscape-inline"><Plot title="Three predictors match both training examples but disagree at the unseen input zero"><CurveAxes xLabel="input x (not a weight)" yLabel="prediction hₐ(x)" xTicks={[-1, 0, 1]} yTicks={[-2, 0, 2]} x={x} y={y} />{[[-1, 'blue'], [0, 'green'], [1, 'amber']].map(([a, color]) => <path key={a} d={path(sample(input => interpolationValue(a, input), -1.25, 1.25), x, y)} className={`landscape-line ${color}`} />)}{[-1, 1].map(input => <circle key={input} cx={x(input)} cy={y(input)} r="6" fill="#f5f6f7" />)}<circle cx={x(0)} cy={y(0)} r="8" className="landscape-ring pink" /></Plot><figcaption>Calculated hₐ(x)=x+a(x²−1): blue a=−1, green a=0, amber a=1. White training points are (−1,−1) and (1,1). The pink ring is the declared held-out target (0,0). All three training losses and training curvatures in a are zero; their held-out predictions differ.</figcaption></figure>;
}
