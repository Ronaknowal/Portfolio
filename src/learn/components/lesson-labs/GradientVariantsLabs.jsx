import { useId, useState } from 'react';
import { adaptiveHistoryState, batchGradientState, decayComparisonState, formatOptimizerNumber as number, layerScaleState, momentumTrajectoryState, optimizerDecayPresets, optimizerGradientProfiles } from '../../data/gradient-variants-models.js';
import './gradient-variants-labs.css';
function Field({
  label,
  children
}) {
  return <label className="optimizer-field"><span>{label}</span>{children}</label>;
}
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step
}) {
  return <Field label={`${label} · ${number(value)}`}><input aria-label={label} type="range" value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} /></Field>;
}
function Steps({
  step,
  onChange,
  maximum,
  label = 'Update'
}) {
  return <div className="optimizer-steps"><button type="button" disabled={step === 0} onClick={() => onChange(step - 1)}>Previous</button><output>{label} {step} of {maximum}</output><button type="button" disabled={step >= maximum} onClick={() => onChange(step + 1)}>Next</button><button type="button" onClick={() => onChange(0)}>Restart</button></div>;
}
function Plot({
  title,
  children,
  className = ''
}) {
  const id = useId();
  return <svg className={`optimizer-plot ${className}`} viewBox="0 0 330 255" role="img" aria-labelledby={id}><title id={id}>{title}</title>{children}</svg>;
}
function path(points, x, y) {
  return points.map((point, index) => `${index ? 'L' : 'M'}${x(point[0]).toFixed(3)},${y(point[1]).toFixed(3)}`).join(' ');
}
function ShortAxis({
  xLabel,
  yLabel,
  lower,
  upper
}) {
  return <><line x1="44" x2="309" y1="214" y2="214" className="optimizer-axis" /><line x1="44" x2="44" y1="32" y2="214" className="optimizer-axis" /><text x="48" y="22" className="optimizer-label">{yLabel}</text><text x="306" y="247" textAnchor="end" className="optimizer-label">{xLabel}</text>{lower !== undefined && <text x="44" y="235" className="optimizer-tick">{number(lower)}</text>}{upper !== undefined && <text x="309" y="235" textAnchor="end" className="optimizer-tick">{number(upper)}</text>}</>;
}
function TimePlot({
  series,
  current,
  title,
  coordinate = null
}) {
  const values = series.flatMap(item => item.values.slice(0, current + 1));
  const low = Math.min(0, ...values);
  const high = Math.max(1e-5, ...values);
  const margin = Math.max((high - low) * 0.12, 1e-5);
  const x = value => 44 + value * 264 / Math.max(1, series[0].values.length - 1);
  const y = value => 214 - (value - low + margin) / (high - low + 2 * margin) * 180;
  return <Plot title={title}>
    <ShortAxis xLabel="update index" yLabel={coordinate === null ? 'loss' : `coordinate ${coordinate + 1}: gradient units`} lower="0" upper={series[0].values.length - 1} />
    <line x1="44" x2="308" y1={y(0)} y2={y(0)} className="optimizer-grid" />
    <text x="51" y="47" className="optimizer-tick">top {Number((high + margin).toPrecision(3))}</text>
    <text x="51" y="206" className="optimizer-tick">bottom {Number((low - margin).toPrecision(3))}</text>
    {series.map(item => <g key={item.label}><path d={path(item.values.slice(0, current + 1).map((value, index) => [index, value]), x, y)} className={`optimizer-line ${item.style}`} />{item.values.slice(0, current + 1).map((value, index) => <circle key={index} cx={x(index)} cy={y(value)} r={index === current ? 4 : 2} className={`optimizer-dot ${item.style}`} />)}</g>)}
  </Plot>;
}
export function BatchGradientLab() {
  const [theta, setTheta] = useState(1);
  const [selected, setSelected] = useState([3]);
  const [rate, setRate] = useState(0.2);
  const [reduction, setReduction] = useState('mean');
  const [error, setError] = useState('');
  const state = batchGradientState(theta, selected, rate, reduction);
  const bound = Math.max(4, Math.abs(state.next) * 1.15);
  const x = value => 44 + (value + bound) / (2 * bound) * 264;
  const y = value => 214 - value / ((bound ** 2 + 5) / 2 * 1.05) * 180;
  const curve = Array.from({
    length: 81
  }, (_, index) => {
    const value = -bound + 2 * bound * index / 80;
    return [value, (value ** 2 + 5) / 2];
  });
  function toggle(index) {
    const next = selected.includes(index) ? selected.filter(item => item !== index) : [...selected, index].sort();
    try {
      batchGradientState(theta, next, rate, reduction);
      setSelected(next);
      setError('');
    } catch (problem) {
      setError(problem.message);
    }
  }
  function reset() {
    setTheta(1);
    setSelected([3]);
    setRate(0.2);
    setReduction('mean');
    setError('');
  }
  return <section className="optimizer-lab" aria-label="Sampled gradient investigation">
    <p className="optimizer-eyebrow">Investigate · which observations supply the slope?</p><h3>A valid batch can point away from the full minimum</h3>
    <p>At θ=1 the full mean gradient is positive. Predict the direction when only observation 3 is selected. Then select all four observations. The curve is always the same full empirical objective.</p>
    <div className="optimizer-controls"><Range label="Parameter theta" value={theta} onChange={setTheta} min={-4} max={4} step={0.25} /><Range label="Batch learning rate" value={rate} onChange={setRate} min={0} max={0.8} step={0.05} /><Field label="Batch reduction"><select aria-label="Batch reduction" value={reduction} onChange={event => setReduction(event.target.value)}><option value="mean">Mean</option><option value="sum">Sum</option></select></Field><button type="button" onClick={reset}>Reset batch</button></div>
    <fieldset className="optimizer-observations"><legend>Select observations · at least one</legend>{state.measurements.map((value, index) => <label key={index}><input type="checkbox" checked={selected.includes(index)} onChange={() => toggle(index)} />a={value}<small>g={number(state.gradients[index])}</small></label>)}</fieldset>
    {error && <p role="alert" className="optimizer-error">{error}</p>}
    <div className="optimizer-pair"><figure><Plot title="Full empirical quadratic objective, current parameter and proposed batch update"><ShortAxis xLabel="parameter θ" yLabel="full mean loss" lower={-bound} upper={bound} /><path d={path(curve, x, y)} className="optimizer-line green" /><line x1={x(theta)} x2={x(state.next)} y1={y(state.fullLoss)} y2={y(state.nextFullLoss)} className="optimizer-line amber" /><circle cx={x(theta)} cy={y(state.fullLoss)} r="5" className="optimizer-dot green" /><circle cx={x(state.next)} cy={y(state.nextFullLoss)} r="6" className="optimizer-dot amber" /></Plot><figcaption>Green point: before. Amber point: the proposed batch step. Straight connector shows endpoints, not an optimization path or tangent.</figcaption></figure>
      <dl className="optimizer-readout"><div><dt>Selected mean gradient</dt><dd>{number(state.meanGradient)}</dd></div><div><dt>Full mean gradient</dt><dd>{number(state.fullGradient)}</dd></div><div><dt>Gradient used after reduction</dt><dd>{number(state.usedGradient)}</dd></div><div><dt>Proposed θ</dt><dd>{number(theta)} → {number(state.next)}</dd></div><div><dt>Full loss before → after</dt><dd>{number(state.fullLoss)} → {number(state.nextFullLoss)}</dd></div><div><dt>Mean over all size-{selected.length} subset means</dt><dd>{number(state.expectedMean)}</dd></div><div><dt>Variance across those subset means</dt><dd>{number(state.variance)}</dd></div></dl>
    </div><p role="status">This batch {state.nextFullLoss > state.fullLoss + 1e-12 ? 'raises' : state.nextFullLoss < state.fullLoss - 1e-12 ? 'lowers' : 'does not change'} the full loss. Averaging all equally likely size-{selected.length} subsets recovers the full gradient {number(state.fullGradient)}; this particular chosen subset need not equal it.</p>
    <p className="optimizer-transfer">Try θ=0. Can a one-observation step move away from the exact full-data optimum? Then compare mean and sum for two selected observations: which learning rate would preserve the same step?</p>
  </section>;
}
export function MomentumTrajectoryLab() {
  const [method, setMethod] = useState('momentum');
  const [rate, setRate] = useState(0.08);
  const [beta, setBeta] = useState(0.8);
  const [curvature, setCurvature] = useState(20);
  const [step, setStep] = useState(0);
  const state = momentumTrajectoryState(method, rate, beta, 24, curvature);
  const index = Math.min(step, state.frames.length - 1);
  const frame = state.frames[index];
  const shown = state.frames.slice(0, index + 1);
  const currentGradient = frame.currentGradient || frame.gradient;
  const plainEnd = frame.before.map((value, coordinate) => value - rate * currentGradient[coordinate]);
  const extent = Math.max(2.5, ...shown.flatMap(item => item.theta.map(Math.abs)), ...plainEnd.map(Math.abs), ...frame.evaluationPoint.map(Math.abs)) * 1.1;
  const x = value => 176 + value / extent * 90;
  const y = value => 214 - (value + extent) / (2 * extent) * 180;
  const arrowId = useId().replaceAll(':', '');
  function change(setter, value) {
    setter(value);
    setStep(0);
  }
  return <section className="optimizer-lab" aria-label="Momentum and lookahead investigation"><p className="optimizer-eyebrow">Investigate · current slope and stored direction</p><h3>The update has a memory</h3><p>Minimize F(x,y)=(x²+c·y²)/2 from (2,1). Advance one update at a time. The amber displacement is the actual move; the dashed green displacement is what plain descent at the current point would propose with the same rate.</p>
    <div className="optimizer-controls"><Field label="Trajectory method"><select aria-label="Trajectory method" value={method} onChange={event => change(setMethod, event.target.value)}><option value="sgd">Plain descent</option><option value="momentum">Momentum buffer</option><option value="nesterov">Nesterov lookahead</option></select></Field><Range label="Trajectory learning rate" value={rate} onChange={value => change(setRate, value)} min={0.005} max={0.25} step={0.005} /><Range label="Momentum beta" value={beta} onChange={value => change(setBeta, value)} min={0} max={0.95} step={0.05} /><Range label="Vertical curvature c" value={curvature} onChange={value => change(setCurvature, value)} min={1} max={30} step={1} /></div>
    <div className="optimizer-pair"><figure><Plot title="Quadratic parameter-plane trajectory and two proposed displacements"><defs><marker id={arrowId} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="context-stroke" /></marker><clipPath id={`${arrowId}-clip`}><rect x="44" y="32" width="264" height="182" /></clipPath></defs><ShortAxis xLabel="parameter x" yLabel="parameter y" lower={-Number((extent * 264 / 180).toPrecision(2))} upper={Number((extent * 264 / 180).toPrecision(2))} /><line x1={x(0)} x2={x(0)} y1="32" y2="214" className="optimizer-grid" /><line x1="44" x2="308" y1={y(0)} y2={y(0)} className="optimizer-grid" /><g clipPath={`url(#${arrowId}-clip)`}>{[0.5, 2, 8, 20].map(level => <ellipse key={level} cx={x(0)} cy={y(0)} rx={Math.sqrt(2 * level) / (2 * extent) * 180} ry={Math.sqrt(2 * level / curvature) / (2 * extent) * 180} className="optimizer-contour" />)}</g><path d={path(shown.map(item => item.theta), x, y)} className="optimizer-line amber" />{shown.map(item => <circle key={item.step} cx={x(item.theta[0])} cy={y(item.theta[1])} r="3" className="optimizer-dot amber" />)}{index > 0 && <><line x1={x(frame.before[0])} y1={y(frame.before[1])} x2={x(plainEnd[0])} y2={y(plainEnd[1])} markerEnd={`url(#${arrowId})`} className="optimizer-line green dashed" /><line x1={x(frame.before[0])} y1={y(frame.before[1])} x2={x(frame.theta[0])} y2={y(frame.theta[1])} markerEnd={`url(#${arrowId})`} className="optimizer-line amber" /></>}{method === 'nesterov' && <rect x={x(frame.evaluationPoint[0]) - 4} y={y(frame.evaluationPoint[1]) - 4} width="8" height="8" className="optimizer-lookahead" />}</Plot><figcaption>Contours are exact level sets. Both arrows use parameter-displacement units and one coordinate scale. A pink square marks Nesterov's evaluation point. Plot bounds expand for large iterates.</figcaption></figure><figure><TimePlot title="Actual quadratic loss after each displayed update" series={[{
          label: 'loss',
          values: state.frames.map(item => item.loss),
          style: 'amber'
        }]} current={index} /><figcaption>Linear vertical scale, computed F at each iterate. This deterministic bowl has no sampling noise; a rise exposes the update rule or chosen rate.</figcaption></figure></div>
    <div className="optimizer-table" tabIndex="0" role="region" aria-label="Momentum exact state"><table><thead><tr><th>Current update</th><th>x coordinate</th><th>y coordinate</th></tr></thead><tbody>{[['Before', frame.before], ['Gradient evaluation point', frame.evaluationPoint], ['Gradient used', frame.gradient], ['New buffer / plain gradient', frame.velocity], ['Actual displacement', frame.displacement], ['After', frame.theta]].map(([label, values]) => <tr key={label}><th>{label}</th>{values.map((value, coordinate) => <td key={coordinate}>{number(value)}</td>)}</tr>)}</tbody></table></div>
    <Steps step={index} onChange={setStep} maximum={state.frames.length - 1} /><p role="status">Loss {number(frame.loss)} at update {index}. {state.truncated ? `The finite illustration stops before a coordinate would exceed ${state.bound}; this is divergence protection, not convergence.` : 'Changing a control restarts the same initial point and zero memory.'}</p><p className="optimizer-transfer">Compare β=0 with plain descent. Then find a momentum update whose current gradient and actual displacement have the same sign in one coordinate. Why does “opposite the gradient” no longer describe that move?</p></section>;
}
export function AdaptiveHistoryLab() {
  const [method, setMethod] = useState('adam');
  const [profile, setProfile] = useState('sparse');
  const [beta2, setBeta2] = useState(0.9);
  const [correction, setCorrection] = useState(true);
  const [step, setStep] = useState(0);
  const state = adaptiveHistoryState(method, profile, 0.1, 0.9, beta2, 1e-6, correction);
  const frame = state.frames[step];
  function change(setter, value) {
    setter(value);
    setStep(0);
  }
  return <section className="optimizer-lab" aria-label="Adaptive gradient history investigation"><p className="optimizer-eyebrow">Investigate · numerator, history and denominator</p><h3>The same present gradient can produce a different step</h3><p>These are supplied gradient vectors, so every method sees the same history. Predict the second-coordinate update immediately after its one large nonzero gradient. This replay isolates arithmetic; it is not a training-loss comparison.</p><div className="optimizer-controls"><Field label="Adaptive method"><select aria-label="Adaptive method" value={method} onChange={event => change(setMethod, event.target.value)}><option value="adagrad">AdaGrad</option><option value="rmsprop">RMSProp</option><option value="adam">Adam</option></select></Field><Field label="Gradient history"><select aria-label="Gradient history" value={profile} onChange={event => change(setProfile, event.target.value)}>{Object.entries(optimizerGradientProfiles).map(([key, item]) => <option value={key} key={key}>{item.title}</option>)}</select></Field><Range label="Recent-square decay beta2" value={beta2} onChange={value => change(setBeta2, value)} min={0} max={0.99} step={0.01} /><label className="optimizer-check"><input type="checkbox" checked={correction} disabled={method !== 'adam'} onChange={event => change(setCorrection, event.target.checked)} />Adam bias correction</label></div>
    <p>η=0.1, Adam β₁=0.9, ε=10⁻⁶ outside the square root. AdaGrad accumulates every square and ignores the recent-square control. RMSProp uses the shown β₂ as its averaging coefficient. Updates below are numbered from 1; plot index 0 is the first supplied gradient.</p>
    <div className="optimizer-pair">{[0, 1].map(coordinate => <figure key={coordinate}><TimePlot title={`Gradient, numerator and denominator history for coordinate ${coordinate + 1}`} current={step} coordinate={coordinate} series={[{
          label: 'gradient',
          values: state.frames.map(item => item.gradient[coordinate]),
          style: 'green'
        }, {
          label: 'numerator',
          values: state.frames.map(item => item.numerator[coordinate]),
          style: 'amber dashed'
        }, {
          label: 'denominator',
          values: state.frames.map(item => item.denominator[coordinate]),
          style: 'blue dotted'
        }]} /><figcaption>Green: present gradient. Dashed amber: numerator. Dotted blue: denominator. All three have gradient units; the parameter step is in the table.</figcaption></figure>)}</div>
    <div className="optimizer-table" tabIndex="0" role="region" aria-label="Adaptive exact arithmetic"><table><thead><tr><th>Coordinate</th><th>g</th><th>Raw square history</th><th>Numerator</th><th>Denominator</th><th>Δθ = −0.1·num/den</th></tr></thead><tbody>{[0, 1].map(coordinate => <tr key={coordinate}><th>{coordinate + 1}</th>{[frame.gradient[coordinate], frame.second[coordinate], frame.numerator[coordinate], frame.denominator[coordinate], frame.displacement[coordinate]].map((value, index) => <td key={index}>{number(value)}</td>)}</tr>)}</tbody></table></div>
    <Steps step={step} onChange={setStep} maximum={7} label="History index" /><p role="status">Applied gradient number {step + 1}. {method === 'adam' ? `Raw first moment [${frame.first.map(number).join(', ')}]. ${correction ? 'Bias correction is active.' : 'Correction is off only to expose the initialization effect; this differs from canonical Adam.'}` : 'The current gradient is the numerator; there is no first-moment buffer in this chosen variant.'}</p><p className="optimizer-transfer">Use the constant history. Why does AdaGrad's step shrink even though g does not? Now use the sparse history and reach its final zero gradient: why can Adam still move?</p></section>;
}
export function DecayComparisonLab() {
  const [preset, setPreset] = useState('zero');
  const [rate, setRate] = useState(0.1);
  const [decay, setDecay] = useState(0.1);
  const [steps, setSteps] = useState(1);
  const state = decayComparisonState(preset, rate, decay, steps);
  return <section className="optimizer-lab" aria-label="Coupled L2 and AdamW investigation"><p className="optimizer-eyebrow">Investigate · where the penalty enters</p><h3>Passing a penalty through moments changes its effect</h3><p>Start both methods from the same parameters and zero moments. Replay a fixed supplied data gradient for the chosen number of steps. In the zero-gradient case, predict whether parameters 2 and 10 lose the same absolute amount or the same fraction on the first step.</p><div className="optimizer-controls"><Field label="Decay input case"><select aria-label="Decay input case" value={preset} onChange={event => setPreset(event.target.value)}>{Object.entries(optimizerDecayPresets).map(([key, item]) => <option value={key} key={key}>{item.title}</option>)}</select></Field><Range label="Decay learning rate" value={rate} onChange={setRate} min={0.01} max={0.3} step={0.01} /><Range label="Decay coefficient lambda" value={decay} onChange={setDecay} min={0} max={1} step={0.05} /><Range label="Decay update count" value={steps} onChange={setSteps} min={1} max={8} step={1} /></div>
    <p>Initial θ=[{state.initial.theta.join(', ')}], supplied data gradient=[{state.initial.gradient.join(', ')}]. Both use Adam β₁=.9, β₂=.99 and ε=10⁻⁸. Each control change recomputes from zero state.</p>
    <div className="optimizer-pair">{state.methods.map(item => <div className="optimizer-route" key={item.method}><h4>{item.method === 'coupled' ? 'Coupled L2 → Adam' : 'AdamW → direct shrinkage'}</h4><ol><li>Moment input: [{item.final.gradient.map(number).join(', ')}]</li><li>First raw moment: [{item.final.first.map(number).join(', ')}]</li><li>Second raw moment: [{item.final.second.map(number).join(', ')}]</li><li>Separate shrinkage: [{item.final.shrinkage.map(number).join(', ')}]</li><li>Final θ: <strong>[{item.theta.map(number).join(', ')}]</strong></li></ol></div>)}</div>
    <div className="optimizer-table" tabIndex="0" role="region" aria-label="Decay parameter trajectories"><table><thead><tr><th>After update</th><th>Coupled L2 θ</th><th>AdamW θ</th></tr></thead><tbody>{state.methods[0].frames.map((frame, index) => <tr key={index}><th>{index + 1}</th><td>[{frame.theta.map(number).join(', ')}]</td><td>[{state.methods[1].frames[index].theta.map(number).join(', ')}]</td></tr>)}</tbody></table></div><p role="status">At update {steps}, coupled moments use the data gradient plus λθ. AdamW moments use only the supplied data gradient; shrinkage is −ηλθ before adding the adaptive displacement.</p><p className="optimizer-transfer">Set λ=0: the paths should agree. Restore λ=.1, use the zero-gradient case and one step, then change η. Does decoupled mean independent of learning rate?</p></section>;
}
export function LayerRatioLab() {
  const [method, setMethod] = useState('lars');
  const [smallScale, setSmallScale] = useState(0.1);
  const [rate, setRate] = useState(0.1);
  const [coefficient, setCoefficient] = useState(0.1);
  const [decay, setDecay] = useState(0);
  const [preset, setPreset] = useState('ordinary');
  const state = layerScaleState(method, smallScale, rate, decay, coefficient, preset);
  const normMaximum = Math.max(0.5, ...state.blocks.flatMap(block => [block.weightNorm, block.updateNorm]));
  return <section className="optimizer-lab" aria-label="Layer relative update investigation"><p className="optimizer-eyebrow">Investigate · absolute movement versus relative movement</p><h3>Two blocks, one global rate</h3><p>Each block has two parameters. This is the first update with zero momentum/moments. Predict which block moves farther relative to its own norm under plain SGD, then inspect LARS or LAMB. No neural network or distributed throughput is simulated.</p><div className="optimizer-controls"><Field label="Layer method"><select aria-label="Layer method" value={method} onChange={event => setMethod(event.target.value)}><option value="sgd">Plain SGD</option><option value="lars">LARS</option><option value="lamb">LAMB</option></select></Field><Field label="Layer boundary case"><select aria-label="Layer boundary case" value={preset} onChange={event => setPreset(event.target.value)}><option value="ordinary">Both blocks nonzero</option><option value="zeroWeight">Zero second-block weights</option><option value="zeroGradient">Zero second-block data gradient</option></select></Field><Range label="Second-block weight scale" value={smallScale} onChange={setSmallScale} min={0.02} max={2} step={0.02} /><Range label="Layer learning rate" value={rate} onChange={setRate} min={0.01} max={0.5} step={0.01} /><Range label="LARS trust coefficient" value={coefficient} onChange={setCoefficient} min={0.01} max={1} step={0.01} /><Range label="Layer decay coefficient" value={decay} onChange={setDecay} min={0} max={1} step={0.05} /></div>
    <div className="optimizer-pair">{state.blocks.map((block, index) => {
        const maximum = normMaximum;
        return <figure key={index}><Plot title={`Block ${index + 1} parameter norm and update norm on a shared linear scale`}><text x="22" y="29" className="optimizer-label">Block {index + 1}: absolute norms</text><text x="22" y="65" className="optimizer-label">||θ|| = {number(block.weightNorm)}</text><rect x="22" y="79" width={block.weightNorm / maximum * 280} height="25" className="optimizer-bar green" /><text x="22" y="140" className="optimizer-label">||Δθ|| = {number(block.updateNorm)}</text><rect x="22" y="154" width={block.updateNorm / maximum * 280} height="25" className="optimizer-bar amber" /><text x="22" y="214" className="optimizer-label">Relative: {block.relativeUpdate === null ? 'undefined at θ=0' : number(block.relativeUpdate)}</text><text x="22" y="241" className="optimizer-tick">Shared linear norm scale across both blocks.</text></Plot><figcaption>θ=[{block.theta.map(number).join(', ')}], data g=[{block.gradient.map(number).join(', ')}]. {block.fallback ? 'Zero norm triggers the declared ratio-1 fallback.' : `Ratio ${number(block.ratio)}.`}</figcaption></figure>;
      })}</div><div className="optimizer-table" tabIndex="0" role="region" aria-label="Layer ratio arithmetic"><table><thead><tr><th>Block</th><th>Direction before ratio</th><th>Ratio denominator</th><th>Ratio</th><th>After θ</th></tr></thead><tbody>{state.blocks.map((block, index) => <tr key={index}><th>{index + 1}</th><td>[{block.direction.map(number).join(', ')}]</td><td>{number(block.denominator)}</td><td>{number(block.ratio)}</td><td>[{block.next.map(number).join(', ')}]</td></tr>)}</tbody></table></div>
    <p role="status">{method === 'lars' ? 'LARS divides the trust coefficient times parameter norm by gradient norm plus λ times parameter norm.' : method === 'lamb' ? 'LAMB uses the first corrected Adam direction plus λθ, then divides parameter norm by that combined direction norm.' : 'Plain SGD uses ratio 1 for both blocks.'} LAMB uses identity norm scaling without clipping; a zero parameter norm or denominator uses ratio 1.</p><p className="optimizer-transfer">Set the second block to zero weights. Why is its relative update undefined even though it can move? In the complete program, compare LARS's second update: stored momentum means the first-step relative bound is no longer an equality.</p></section>;
}
export function OptimizerPipelineFigure() {
  return <figure className="optimizer-inline"><ol className="optimizer-pipeline"><li><strong>Observe</strong><span>Choose data and compute g at the declared parameter point.</span></li><li><strong>Remember</strong><span>Update a buffer, squared history, or both.</span></li><li><strong>Scale</strong><span>Choose the learning rate, coordinate denominator and any block ratio.</span></li><li><strong>Move</strong><span>Apply the defined displacement and decay; keep state for the next step.</span></li></ol><figcaption>These are separate choices. A named optimizer specifies their formulas and order; changing the order can change the algorithm.</figcaption></figure>;
}
export function AdamWeightFigure() {
  const beta = 0.8;
  const weights = [0, 1, 2, 3].map(index => (1 - beta) * beta ** (3 - index));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return <figure className="optimizer-inline"><div className="optimizer-weight-strip">{weights.map((weight, index) => <div key={index}><span>g{index + 1}</span><div className="optimizer-weight-bar" style={{
          height: `${weight * 260}px`
        }} /><strong>{number(weight)}</strong><small>corrected {number(weight / total)}</small></div>)}</div><figcaption>At t=4 and β=.8, the raw EMA weights sum to {number(total)}=1−.8⁴. Dividing each weight by {number(total)} makes them sum to one. Recent gradients receive larger weight; this does not replace the history by the current gradient.</figcaption></figure>;
}
