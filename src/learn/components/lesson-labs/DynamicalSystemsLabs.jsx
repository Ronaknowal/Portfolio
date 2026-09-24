import { useId, useMemo, useState } from 'react';
import { CUBIC_FOLD, scalarFlowState, planarField, planarTrace, logisticTrace, logisticSensitivity, logisticLyapunovEstimate, bifurcationAtlas, tentCylinder, tentStep, logisticCoordinate, logisticInvariantCdf, quantizedTentTrace, coolingTrace, oscillatorTrace, lorenzTrace } from '../../data/dynamical-systems-models.js';
import './dynamical-systems-labs.css';

const format = value => value === null ? '−∞' : Math.abs(value) < 1e-4 && value !== 0 ? value.toExponential(2) : Number(value.toFixed(4)).toString();
const tickFormat = value => Number(value.toPrecision(3)).toString();
const pointsPath = (points, x, y) => {
  let connected = false;
  return points.map(point => {
    if (!point || point.some(value => !Number.isFinite(value))) { connected = false; return ''; }
    const instruction = (connected ? 'L' : 'M') + x(point[0]).toFixed(2) + ',' + y(point[1]).toFixed(2);
    connected = true;
    return instruction;
  }).join(' ');
};

function Choice({ label, value, choices, onChange, numeric = false }) {
  return <label className="dynamics-control">{label}<select aria-label={label} value={value} onChange={event => onChange(numeric ? Number(event.target.value) : event.target.value)}>{choices.map(([key, name]) => <option key={key} value={key}>{name}</option>)}</select></label>;
}

function Range({ label, value, min, max, step = 1, onChange }) {
  return <label className="dynamics-control">{label}: <strong>{format(value)}</strong><input aria-label={label} type="range" value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} /></label>;
}

function Metrics({ rows }) {
  return <dl className="dynamics-metrics">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}

function Plot({ label, horizontal, vertical, xDomain, yDomain, series = [], square = false, children }) {
  const clipId = useId();
  const height = square ? 540 : 330;
  const bottom = height - 60;
  const x = value => 65 + (value - xDomain[0]) / (xDomain[1] - xDomain[0]) * 450;
  const y = value => bottom - (value - yDomain[0]) / (yDomain[1] - yDomain[0]) * (bottom - 30);
  return <><p className="dynamics-axis-label">Vertical axis: {vertical}</p><p className="dynamics-scroll-hint">Scroll this plot sideways to inspect the full axes.</p><div className="dynamics-plot" tabIndex={0} role="region" aria-label={label + '; scroll horizontally on a narrow screen'}>
    <svg viewBox={'0 0 540 ' + height} role="img" aria-label={label}>
      <defs><clipPath id={clipId}><rect x={65} y={30} width={450} height={bottom - 30} /></clipPath></defs>
      {Array.from({ length: 5 }, (_, index) => {
        const horizontalValue = xDomain[0] + index / 4 * (xDomain[1] - xDomain[0]);
        const verticalValue = yDomain[0] + index / 4 * (yDomain[1] - yDomain[0]);
        return <g key={index}><line className="dynamics-grid" x1={x(horizontalValue)} x2={x(horizontalValue)} y1={30} y2={bottom} /><line className="dynamics-grid" x1={65} x2={515} y1={y(verticalValue)} y2={y(verticalValue)} /><text x={x(horizontalValue)} y={bottom + 24} textAnchor={index === 0 ? 'start' : index === 4 ? 'end' : 'middle'}>{tickFormat(horizontalValue)}</text><text x={57} y={y(verticalValue) + 4} textAnchor="end">{tickFormat(verticalValue)}</text></g>;
      })}
      <g clipPath={'url(#' + clipId + ')'}>{series.map(({ points, tone = 'amber', dashed = false }, index) => <path key={index} className={'dynamics-line dynamics-' + tone + (dashed ? ' dynamics-dashed' : '')} d={pointsPath(points, x, y)} />)}{children?.({ x, y })}</g>
    </svg>
  </div><p className="dynamics-axis-label">Horizontal axis: {horizontal}</p></>;
}

function CartState({ direction }) {
  const leftward = direction < 0;
  const tip = leftward ? 55 : 245;
  return <svg className="dynamics-cart-drawing" viewBox="0 0 300 115" role="img" aria-label={'Cart at the center, moving ' + (leftward ? 'left' : 'right')}><line x1={20} x2={280} y1={98} y2={98} stroke="#8392a7" strokeWidth={2} /><line x1={150} x2={150} y1={83} y2={108} stroke="#d5deeb" strokeWidth={2} /><rect x={120} y={63} width={60} height={23} rx={3} fill="#213950" stroke="#b9d4eb" strokeWidth={2} /><circle cx={132} cy={91} r={6} fill="#d5deeb" /><circle cx={168} cy={91} r={6} fill="#d5deeb" /><path d={'M150,38H' + tip + 'M' + (tip - direction * 12) + ',28L' + tip + ',38L' + (tip - direction * 12) + ',48'} fill="none" stroke="#f4bf58" strokeWidth={3} /><line x1={150} x2={150} y1={52} y2={62} stroke="#b9d4eb" strokeDasharray="3 3" /></svg>;
}

export function StateMeaningFigure() {
  return <figure className="dynamics-figure"><div className="dynamics-state-pair"><div><CartState direction={-1} /><strong>Same position q=0</strong><span>Velocity p=−1: moving left</span></div><div><CartState direction={1} /><strong>Same position q=0</strong><span>Velocity p=+1: moving right</span></div></div><div className="dynamics-state-flow"><span>Complete state (q,p)</span><span aria-hidden="true">→</span><span>Rule + elapsed time</span><span aria-hidden="true">→</span><span>Next complete state</span></div><figcaption>Position alone cannot choose the next position. The state must retain the information the rule needs. Both carts are at the track's center tick; their velocity arrows differ. This is a state comparison, not a scaled physical simulation.</figcaption></figure>;
}

export function StabilityMeaningFigure() {
  const modes = [['−x', time => Math.exp(-time), 'Returns exponentially'], ['−x³', time => 1 / Math.sqrt(1 + 2 * time), 'Returns algebraically'], ['0', () => 1, 'Stays near; does not return']];
  return <figure className="dynamics-figure"><Plot label="Three perturbations starting at one under three scalar rules" horizontal="time t (model units)" vertical="perturbation x(t)" xDomain={[0, 6]} yDomain={[0, 1.1]} series={modes.map(([, value], index) => ({ points: Array.from({ length: 101 }, (_, step) => [step * 0.06, value(step * 0.06)]), tone: ['amber', 'green', 'blue'][index], dashed: index === 2 }))} /><figcaption>Amber: x′=−x. Green: x′=−x³. Blue dashed: x′=0. All start at 1 and use their exact solutions. Staying near, returning, and returning at an exponential rate are distinct properties.</figcaption></figure>;
}

export function EquilibriumBranchFigure() {
  const positiveParameters = Array.from({ length: 101 }, (_, index) => index / 100);
  const turningState = 1 / Math.sqrt(3);
  const tiltedBranch = (minimum, maximum) => Array.from({ length: 121 }, (_, index) => {
    const state = minimum + (maximum - minimum) * index / 120;
    return [state ** 3 - state, state];
  });
  return <figure className="dynamics-figure">
    <Plot label="Pitchfork equilibrium branches as parameter a changes" horizontal="parameter a" vertical="equilibrium state x*" xDomain={[-1, 1]} yDomain={[-1.5, 1.5]} series={[{ points: [[-1, 0], [0, 0]], tone: 'green' }, { points: [[0, 0], [1, 0]], dashed: true }, ...[-1, 1].map(sign => ({ points: positiveParameters.map(parameter => [parameter, sign * Math.sqrt(parameter)]), tone: 'green' }))]} />
    <Plot label="Tilted cubic equilibrium branches and the two fold thresholds" horizontal="tilt parameter b" vertical="equilibrium state x*" xDomain={[-0.6, 0.6]} yDomain={[-1.5, 1.5]} series={[{ points: tiltedBranch(-1.4, -turningState), tone: 'green' }, { points: tiltedBranch(-turningState, turningState), dashed: true }, { points: tiltedBranch(turningState, 1.4), tone: 'green' }]}>{({ x, y }) => [-1, 1].map(sign => <circle key={sign} className="dynamics-open" cx={x(-sign * CUBIC_FOLD)} cy={y(sign * turningState)} r={6} />)}</Plot>
    <figcaption>These curves plot equilibria against a changing parameter, not trajectories against time. Green solid branches attract; amber dashed branches repel away from the critical junctions. The top plot branches at a=0. The lower plot turns at b=±2/(3√3), where the open fold points attract from only one side. At a=0 the pitchfork's nonlinear cubic still attracts. The exact curves come from x*=0 or ±√a, and b=(x*)³−x*. A slow branch-following experiment can switch at different folds on its outward and return sweeps; finite sweep speed is not modeled here.</figcaption>
  </figure>;
}

export function ScalarFlowLab() {
  const [kind, setKind] = useState('pitchfork');
  const [parameter, setParameter] = useState(1);
  const [initial, setInitial] = useState(0.2);
  const state = useMemo(() => scalarFlowState({ kind, parameter, initial }), [kind, parameter, initial]);
  const final = state.rows.at(-1);
  const options = kind === 'pitchfork' ? [[-1, 'a=−1'], [0, 'a=0: marginal linearization'], [0.25, 'a=0.25'], [1, 'a=1: two basins']] : [[-0.5, 'b=−0.5'], [-CUBIC_FOLD, 'b=−2/(3√3): fold'], [0, 'b=0'], [CUBIC_FOLD, 'b=+2/(3√3): fold'], [0.5, 'b=+0.5']];
  return <section className="dynamics-lab" aria-label="Scalar phase line investigation"><h3>Follow the arrows into a potential well</h3><p>Inspect the destination from the sign of the rate before reading the trajectory. Change the initial state across zero at a=1; then inspect the exact fold settings in the tilted rule.</p>
    <div className="dynamics-controls"><Choice label="Scalar rule" value={kind} onChange={value => { setKind(value); setParameter(value === 'pitchfork' ? 1 : 0); }} choices={[['pitchfork', 'x′=a x−x³'], ['tilted', 'x′=b+x−x³']]} /><Choice label="Scalar parameter" numeric value={parameter} onChange={setParameter} choices={options} /><Range label="Scalar initial state" value={initial} min={-1.8} max={1.8} step={0.1} onChange={setInitial} /></div>
    <Plot label="Rate curve with equilibrium dots and phase-line arrows" horizontal="state x" vertical="rate x′" xDomain={[-2, 2]} yDomain={[-6.6, 6.6]} series={[{ points: state.curve.map(row => [row.value, row.rate]) }]}>{({ x, y }) => <><line className="dynamics-zero" x1={x(-2)} x2={x(2)} y1={y(0)} y2={y(0)} />{state.equilibria.map(equilibrium => <circle key={equilibrium.value} cx={x(equilibrium.value)} cy={y(0)} r={6} className={equilibrium.stability === 'attracting' ? 'dynamics-filled' : 'dynamics-open'} />)}{[-1.6, -0.8, -0.25, 0.25, 0.8, 1.6].map(value => { const rate = kind === 'pitchfork' ? parameter * value - value ** 3 : parameter + value - value ** 3; return <text key={value} x={x(value)} y={y(0) - 14} textAnchor="middle">{Math.abs(rate) < 1e-10 ? '·' : rate > 0 ? '→' : '←'}</text>; })}</>}</Plot>
    <Plot label="Potential curve with the initial and evolved states" horizontal="state x" vertical="potential V(x), arbitrary zero" xDomain={[-2, 2]} yDomain={[-1, 6]} series={[{ points: state.curve.map(row => [row.value, row.potential]), tone: 'green' }]}>{({ x, y }) => <><circle className="dynamics-open" cx={x(initial)} cy={y(state.rows[0].potential)} r={7} /><circle className="dynamics-filled" cx={x(final.value)} cy={y(final.potential)} r={5} /></>}</Plot>
    <Plot label="Scalar evolution over eight time units" horizontal="time t" vertical="state x(t)" xDomain={[0, 8]} yDomain={[-2, 2]} series={[{ points: state.rows.map(row => [row.time, row.value]) }]} />
    <div aria-live="polite"><Metrics rows={[[ 'At t=8', 'x=' + format(final.value)], ['Equilibria', state.equilibria.map(point => format(point.value) + ': ' + point.stability).join('; ')]]} /></div>
    <p>Filled rate dots attract from both sides; open dots repel or attract from one side only (read the listed classification). On the potential curve, the open dot is the initial state and the filled dot is its state at t=8. Arrows show direction, not speed. The trajectory uses RK4 with h=0.02; roots and curves are calculated from the stated polynomial. A finite endpoint is not an exact equilibrium.</p><button onClick={() => { setKind('pitchfork'); setParameter(1); setInitial(0.2); }}>Reset phase line</button>
  </section>;
}

const planarNames = [['center', 'Center: perpetual circles'], ['spiral', 'Spiral: return to zero'], ['saddle', 'Saddle: one escaping direction'], ['transient', 'Stable with transient growth'], ['hopf', 'Hopf: birth of an attracting cycle']];

export function PlanarFlowLab() {
  const [mode, setMode] = useState('center');
  const [parameter, setParameter] = useState(0.25);
  const [initialChoice, setInitialChoice] = useState('right');
  const [timeIndex, setTimeIndex] = useState(160);
  const initials = { right: [1, 0], upper: [0, 1], near: [0.2, 0], origin: [0, 0] };
  const duration = mode === 'saddle' ? 4 : 4 * Math.PI;
  const state = useMemo(() => planarTrace({ mode, parameter, initial: initials[initialChoice], duration, steps: 320 }), [mode, parameter, initialChoice, duration]);
  const current = state.rows[timeIndex];
  const extent = mode === 'saddle' ? 5.5 : 2;
  const radiusExtent = mode === 'saddle' ? 5.5 : 2;
  const rules = { center: 'q′=−p, p′=q', spiral: 'q′=−0.4q−p, p′=q−0.4p', saddle: 'q′=0.4q, p′=−0.5p', transient: 'q′=−q+6p, p′=−2p', hopf: 'q′=(a−r²)q−p, p′=q+(a−r²)p; r²=q²+p²' };
  return <section className="dynamics-lab" aria-label="Linked phase and time investigation"><h3>Read one state in phase space and in time</h3><p>Inspect whether a nearby orbit approaches the origin, keeps circling, or escapes. The phase plot uses equal coordinate scales; its arrows give direction only. The time cursor identifies the same state in both views.</p><div className="dynamics-controls"><Choice label="Planar system" value={mode} onChange={setMode} choices={planarNames} /><Choice label="Planar initial state" value={initialChoice} onChange={setInitialChoice} choices={[['right', '(q,p)=(1,0)'], ['upper', '(q,p)=(0,1)'], ['near', '(q,p)=(0.2,0)'], ['origin', '(q,p)=(0,0)']]} />{mode === 'hopf' && <Choice label="Hopf parameter" numeric value={parameter} onChange={setParameter} choices={[[-0.25, 'a=−0.25'], [0, 'a=0'], [0.25, 'a=0.25'], [1, 'a=1']]} />}<Range label="Phase trace step" value={timeIndex} min={0} max={320} onChange={setTimeIndex} /></div><p className="dynamics-equation">{rules[mode]}</p>
    <Plot square label="Planar trajectory, direction field and selected state" horizontal="q (dimensionless)" vertical="p (dimensionless)" xDomain={[-extent, extent]} yDomain={[-extent, extent]} series={[{ points: state.rows.slice(0, timeIndex + 1).map(row => row.position) }]}>{({ x, y }) => <>{[-0.75, -0.375, 0, 0.375, 0.75].flatMap(horizontal => [-0.75, -0.375, 0, 0.375, 0.75].map(vertical => {
      const position = [horizontal * extent, vertical * extent];
      const field = planarField(mode, position, parameter);
      const norm = Math.hypot(...field);
      if (norm === 0) return null;
      const dx = field[0] / norm * 10;
      const dy = -field[1] / norm * 10;
      const endX = x(position[0]) + dx;
      const endY = y(position[1]) + dy;
      return <g key={horizontal + ',' + vertical} className="dynamics-arrow"><line x1={x(position[0]) - dx} x2={endX} y1={y(position[1]) - dy} y2={endY} /><path d={'M' + (endX - dx * 0.6 - dy * 0.35) + ',' + (endY - dy * 0.6 + dx * 0.35) + 'L' + endX + ',' + endY + 'L' + (endX - dx * 0.6 + dy * 0.35) + ',' + (endY - dy * 0.6 - dx * 0.35)} /></g>;
    }))}{state.cycleRadius !== null && <circle className="dynamics-cycle" cx={x(0)} cy={y(0)} r={state.cycleRadius * 450 / (2 * extent)} />}<circle className="dynamics-open" cx={x(state.rows[0].position[0])} cy={y(state.rows[0].position[1])} r={7} /><circle className="dynamics-filled" cx={x(current.position[0])} cy={y(current.position[1])} r={5} /></>}</Plot>
    <Plot label="Distance to the origin over time" horizontal="time t (model units)" vertical="radius √(q²+p²)" xDomain={[0, duration]} yDomain={[0, radiusExtent]} series={[{ points: state.rows.map(row => [row.time, row.radius]), tone: 'green' }]}>{({ x, y }) => <circle className="dynamics-filled" cx={x(current.time)} cy={y(current.radius)} r={5} />}</Plot>
    <div aria-live="polite"><Metrics rows={[[ 'Selected time', format(current.time)], ['State (q,p)', current.position.map(format).join(', ')], ['Radius', format(current.radius)], ...(state.cycleRadius !== null ? [['Cycle radius', format(state.cycleRadius)], ['Radial multiplier per turn', format(state.radialReturnMultiplier)]] : [])]} /></div>
    {mode === 'hopf' && <p>The dashed green circle exists for a&gt;0 and has radius √a. Compare the initial radius with the states at t=2π and 4π (steps 160 and 320): they return to the same angle. At the exact origin there is no orbit to leave, even when that equilibrium is unstable. Cycle attraction does not force two different phases to coincide.</p>}
    {mode === 'transient' && <p>Choose (0,1) and inspect the early radius peak. The eigenvalues −1 and −2 govern eventual decay; transfer from p into q can still amplify the Euclidean norm before decay wins.</p>}
    <p>Every displayed path here uses its exact closed-form solution evaluated in binary64. Open dot: start. Filled dot: selected time. The saddle uses a shorter, explicitly labelled horizon and wider axes so escape remains visible. A projection crossing in a higher-dimensional example would not imply an actual state collision.</p><button onClick={() => { setMode('center'); setParameter(0.25); setInitialChoice('right'); setTimeIndex(160); }}>Reset planar view</button>
  </section>;
}

export function LogisticIterationLab() {
  const [growth, setGrowth] = useState(3.2);
  const [initial, setInitial] = useState(0.2);
  const [steps, setSteps] = useState(1);
  const [atlasVisible, setAtlasVisible] = useState(false);
  const trace = useMemo(() => logisticTrace({ growth, initial, steps: 40 }), [growth, initial]);
  const atlas = useMemo(() => atlasVisible ? bifurcationAtlas() : null, [atlasVisible]);
  const curve = Array.from({ length: 151 }, (_, index) => { const value = index / 150; return [value, growth * value * (1 - value)]; });
  const cobweb = [[initial, 0]];
  for (let index = 0; index < steps; index += 1) cobweb.push([trace.values[index], trace.values[index + 1]], [trace.values[index + 1], trace.values[index + 1]]);
  return <section className="dynamics-lab" aria-label="Logistic cobweb investigation"><h3>Turn the output into the next input</h3><p>First move vertically to the curved rule. Then move horizontally to the diagonal, which copies the output onto the input axis. Inspect the next value before advancing.</p><div className="dynamics-controls"><Choice label="Logistic growth" numeric value={growth} onChange={value => { setGrowth(value); setSteps(1); }} choices={[[1, 'r=1'], [2.5, 'r=2.5'], [3, 'r=3'], [3.2, 'r=3.2'], [3.5, 'r=3.5'], [3.83, 'r=3.83'], [3.9, 'r=3.9'], [4, 'r=4']]} /><Choice label="Logistic initial state" numeric value={initial} onChange={value => { setInitial(value); setSteps(1); }} choices={[[0, '0'], [0.2, '0.2'], [0.5, '0.5'], [0.75, '0.75']]} /></div>
    <Plot square label="Logistic parabola, copying diagonal and stepped cobweb" horizontal="input xₙ" vertical="output xₙ₊₁" xDomain={[0, 1]} yDomain={[0, 1]} series={[{ points: curve, tone: 'green' }, { points: [[0, 0], [1, 1]], tone: 'blue', dashed: true }, { points: cobweb }]} />
    <div className="dynamics-actions"><button disabled={steps === 0} onClick={() => setSteps(value => value - 1)}>Previous update</button><button disabled={steps === 40} onClick={() => setSteps(value => value + 1)}>Advance one update</button><button onClick={() => { setGrowth(3.2); setInitial(0.2); setSteps(1); setAtlasVisible(false); }}>Reset logistic</button></div>
    <div aria-live="polite"><Metrics rows={[[ 'Iteration n', steps], ['Current xₙ', format(trace.values[steps])], ['Fixed points and multipliers', trace.fixedPoints.map(point => format(point.value) + ' (m=' + format(point.multiplier) + ')').join('; ')]]} /></div>
    <Plot label="The same logistic iterates on a time axis" horizontal="iteration n" vertical="state xₙ" xDomain={[0, 40]} yDomain={[0, 1]} series={[{ points: trace.values.slice(0, steps + 1).map((value, index) => [index, value]) }]}>{({ x, y }) => trace.values.slice(0, steps + 1).map((value, index) => <circle key={index} cx={x(index)} cy={y(value)} r={3} className="dynamics-filled" />)}</Plot>
    <p>Green: update curve. Blue dashed: copy diagonal. Amber: the selected finite trace. At r=1 or r=3, a multiplier on the unit-circle boundary makes the strict linear test inconclusive; the displayed short trace does not settle that boundary theorem.</p>
    <button aria-expanded={atlasVisible} onClick={() => setAtlasVisible(value => !value)}>{atlasVisible ? 'Hide finite bifurcation atlas' : 'Compute finite bifurcation atlas'}</button>
    {atlas && <><Plot label="Finite sampled bifurcation atlas" horizontal="growth parameter r" vertical="retained state x" xDomain={[2.5, 4]} yDomain={[0, 1]}>{({ x, y }) => <path className="dynamics-atlas" d={atlas.rows.flatMap(row => row.values.map(value => 'M' + x(row.growth).toFixed(2) + ',' + y(value).toFixed(2) + 'h0.7')).join(' ')} />}</Plot><p>151 equally spaced r values from 2.5 to 4, all starting at 0.217; discard 1,000 updates and draw the next 48. Each mark is a computed iterate. Thin windows may fall between sampled columns; repeated values may overplot. This is a finite binary64 sampling, not a complete bifurcation diagram or a chaos certificate. It computes only when opened and uses one SVG path for the marks.</p></>}
  </section>;
}

export function SensitivityLab() {
  const [preset, setPreset] = useState('irregular');
  const [iteration, setIteration] = useState(20);
  const settings = { irregular: [3.9, 0.2, 1e-6], stable: [2.5, 0.2, 1e-6], critical: [3.9, 0.5, 1e-6], fixed: [4, 0.75, 1e-6], equal: [3.9, 0.2, 0] };
  const [growth, initial, difference] = settings[preset];
  const state = useMemo(() => logisticSensitivity({ growth, initial, difference, steps: 70 }), [growth, initial, difference]);
  const finite = useMemo(() => logisticLyapunovEstimate({ growth, initial, burn: preset === 'fixed' || preset === 'critical' ? 0 : 1000, samples: 2000 }), [growth, initial, preset]);
  const current = state.rows[iteration];
  return <section className="dynamics-lab" aria-label="Finite sensitivity investigation"><h3>Compare an actual pair with its tangent approximation</h3><p>Inspect where a straight local-growth picture stops describing the actual pair. Choose the exactly fixed reference to test whether a positive exponent alone means that reference orbit is aperiodic.</p><Choice label="Sensitivity case" value={preset} onChange={setPreset} choices={[['irregular', 'r=3.9, x₀=0.2, δ=10⁻⁶'], ['stable', 'r=2.5, x₀=0.2, δ=10⁻⁶'], ['critical', 'r=3.9, x₀=0.5 (zero derivative)'], ['fixed', 'r=4, x₀=0.75 (exact fixed point)'], ['equal', 'r=3.9, identical initial states']]} /><Range label="Sensitivity iteration" value={iteration} min={0} max={70} onChange={setIteration} />
    <Plot label="Nearby logistic states" horizontal="iteration n" vertical="state" xDomain={[0, 70]} yDomain={[0, 1]} series={[{ points: state.rows.map(row => [row.iteration, row.reference]) }, { points: state.rows.map(row => [row.iteration, row.other]), tone: 'blue', dashed: true }]}>{({ x, y }) => <><line className="dynamics-cursor" x1={x(iteration)} x2={x(iteration)} y1={y(0)} y2={y(1)} /><circle className="dynamics-filled" cx={x(iteration)} cy={y(current.reference)} r={5} /><circle className="dynamics-open" cx={x(iteration)} cy={y(current.other)} r={5} /></>}</Plot>
    <Plot label="Log separation: actual pair and linear tangent prediction" horizontal="iteration n" vertical="log₁₀ absolute separation" xDomain={[0, 70]} yDomain={[-18, 1]} series={[{ points: state.rows.map(row => row.logSeparation === null ? null : [row.iteration, row.logSeparation / Math.LN10]) }, { points: state.rows.map(row => row.logTangentSeparation === null ? null : [row.iteration, row.logTangentSeparation / Math.LN10]), tone: 'green', dashed: true }]}>{({ x, y }) => <><line className="dynamics-cursor" x1={x(iteration)} x2={x(iteration)} y1={y(-18)} y2={y(1)} />{current.logSeparation !== null && <circle className="dynamics-filled" cx={x(iteration)} cy={y(current.logSeparation / Math.LN10)} r={5} />}</>}</Plot>
    <div aria-live="polite"><Metrics rows={[[ 'Selected reference / nearby', format(current.reference) + ' / ' + format(current.other)], ['Actual separation', format(current.separation)], ['Tangent log gain (natural log)', format(current.logGain)], ['Finite mean log derivative', format(finite.estimate)], ['Averaging window', 'discard ' + finite.burn + ', average ' + finite.samples]]} /></div>
    <p>Amber: actual pair; blue dashed: nearby state; green dashed in the lower view: the linear tangent prediction. The log view is explicitly cropped to [−18,1]. Zero separations and vanished derivatives have no finite log point; they are omitted, never replaced by a small invented value. Binary64 rounding can make contracting trajectories coincide exactly. A nonzero difference can survive a zero first derivative through higher-order terms.</p>
    <p>The finite exponent uses the reference alone and the stated averaging window. It is not the slope of the saturated two-trajectory separation. At x=0.75,r=4 the reference stays fixed, its derivative magnitude is 2 and its exponent is ln(2): this is an unstable periodic orbit, not an aperiodic one.</p><button onClick={() => { setPreset('irregular'); setIteration(20); }}>Reset sensitivity</button>
  </section>;
}

export function TentFoldingLab() {
  const [word, setWord] = useState('LR');
  const [coordinate, setCoordinate] = useState(0.2);
  const cylinder = tentCylinder(word);
  const tentOutput = tentStep(coordinate);
  const mappedInput = logisticCoordinate(coordinate);
  const mappedOutput = logisticCoordinate(tentOutput);
  const quantized = useMemo(() => quantizedTentTrace(), []);
  return <section className="dynamics-lab" aria-label="Stretch and fold coordinate investigation"><h3>Follow a branch word and change coordinates</h3><p>L means the left half of the tent; R means the right half. Appending a branch halves the interval of possible starting points. Inspect the interval before adding the next letter.</p><div className="dynamics-actions"><button disabled={word.length === 10} onClick={() => setWord(value => value + 'L')}>Append L</button><button disabled={word.length === 10} onClick={() => setWord(value => value + 'R')}>Append R</button><button disabled={word.length === 1} onClick={() => setWord(value => value.slice(0, -1))}>Remove last branch</button></div>
    <div className="dynamics-cylinder"><span>0</span><div><i style={{ left: (cylinder.interval[0] * 100) + '%', width: ((cylinder.interval[1] - cylinder.interval[0]) * 100) + '%' }} /></div><span>1</span></div>
    <Metrics rows={[[ 'Branch word', word], ['Starting interval', '[' + cylinder.interval.map(format).join(', ') + ']'], ['Width', '2⁻' + word.length + ' = ' + format(2 ** -word.length)], ['A periodic point in the interval', format(cylinder.periodicPoint)]]} />
    <p>The narrow amber interval is drawn to scale, so long words can be thinner than one pixel. Read its bounds above. Branch boundaries can have two labels; the image interval includes endpoints. Repeating this inverse branch has one fixed point because its slope has magnitude 2⁻ⁿ&lt;1.</p>
    <Range label="Tent coordinate y" min={0} max={1} step={0.01} value={coordinate} onChange={setCoordinate} />
    <Plot square label="The tent rule and its selected input/output" horizontal="input y" vertical="T(y)" xDomain={[0, 1]} yDomain={[0, 1]} series={[{ points: [[0, 0], [0.5, 1], [1, 0]] }]}>{({ x, y }) => <circle className="dynamics-filled" cx={x(coordinate)} cy={y(tentOutput)} r={6} />}</Plot>
    <div className="dynamics-commute"><div>y={format(coordinate)} <span>→ T →</span> {format(tentOutput)}</div><div>↓ h(y)=sin²(πy/2) <span /> ↓ h</div><div>x={format(mappedInput)} <span>→ 4x(1−x) →</span> {format(mappedOutput)}</div></div>
    <p>Both routes around this diagram agree by the double-angle identity. Equal widths in y do not become equal widths in x. The invariant mass below x=0.25 is {format(logisticInvariantCdf(0.25))}, not 0.25.</p>
    <details><summary>Inspect the finite binary-grid exception</summary><p>Start at exactly 51/256 on a grid with eight fraction bits. Every update below is exact integer arithmetic; repeated doubling and folding eventually reaches zero.</p><ol className="dynamics-orbit">{quantized.rows.map(row => <li key={row.iteration}><span>n={row.iteration}</span><strong>{row.numerator}/{row.denominator}</strong></li>)}</ol><p>This finite dyadic orbit does not refute a statement about almost every real initial point under the continuous invariant probability law.</p></details><button onClick={() => { setWord('LR'); setCoordinate(0.2); }}>Reset folding</button>
  </section>;
}

export function LorenzProjectionLab() {
  const [rho, setRho] = useState(28);
  const [projection, setProjection] = useState('xz');
  const [timeIndex, setTimeIndex] = useState(2400);
  const state = useMemo(() => lorenzTrace({ rho }), [rho]);
  const current = state.rows[timeIndex];
  const coordinates = projection === 'xz' ? [0, 2] : [0, 1];
  const verticalDomain = projection === 'xz' ? [0, 55] : [-35, 35];
  const visible = state.rows.slice(0, timeIndex + 1).filter((_, index) => index % 3 === 0 || index === timeIndex);
  return <section className="dynamics-lab" aria-label="Lorenz state projection investigation"><h3>A crossing in a projection is not a repeated full state</h3><p>Follow the same three-component state in two projections. Switch views at one time cursor and compare the missing coordinate. These are dimensionless equations with σ=10 and β=8/3.</p><div className="dynamics-controls"><Choice label="Lorenz rho" numeric value={rho} onChange={setRho} choices={[[0.5, 'ρ=0.5'], [10, 'ρ=10'], [28, 'ρ=28']]} /><Choice label="Lorenz projection" value={projection} onChange={setProjection} choices={[['xz', 'x versus z'], ['xy', 'x versus y']]} /><Range label="Lorenz trace step" value={timeIndex} min={0} max={4800} step={20} onChange={setTimeIndex} /></div>
    <Plot label="Finite Lorenz trajectory projection" horizontal="x" vertical={projection === 'xz' ? 'z' : 'y'} xDomain={[-25, 25]} yDomain={verticalDomain} series={[{ points: visible.map(row => coordinates.map(index => row.state[index])) }]}>{({ x, y }) => <>{state.equilibria.map(point => <circle key={point[0]} className="dynamics-open" cx={x(point[coordinates[0]])} cy={y(point[coordinates[1]])} r={6} />)}<circle className="dynamics-filled" cx={x(current.state[coordinates[0]])} cy={y(current.state[coordinates[1]])} r={5} /></>}</Plot>
    <div aria-live="polite"><Metrics rows={[[ 'Time', format(current.time)], ['Full state (x,y,z)', current.state.map(format).join(', ')], ['Equilibria', state.equilibria.map(point => '(' + point.map(format).join(', ') + ')').join('; ')], ['Volume divergence', '−41/3 ≈ ' + format(state.divergence)]]} /></div>
    <p>Open dots are equilibria; the filled dot is the selected full state. RK4 uses h=0.005 for 24 model-time units from (1,1,1). Every third point is drawn, with the selected endpoint retained; the time cursor uses the full trace. Axes are fixed across presets. This finite numerical path illustrates geometry; it is neither a proof of chaos nor a measured weather forecast.</p><button onClick={() => { setRho(28); setProjection('xz'); setTimeIndex(2400); }}>Reset Lorenz</button>
  </section>;
}

export function NumericalDynamicsLab() {
  const [step, setStep] = useState(0.2);
  const [coolingStep, setCoolingStep] = useState(0.5);
  const energy = useMemo(() => oscillatorTrace({ step, steps: Math.round(20 / step) }), [step]);
  const cooling = useMemo(() => coolingTrace({ decay: 2, initial: 8, step: coolingStep, steps: Math.round(6 / coolingStep) }), [coolingStep]);
  const final = energy.rows.at(-1);
  const coolingLimit = Math.max(8, ...cooling.rows.map(row => Math.abs(row.euler)));
  return <section className="dynamics-lab" aria-label="Numerical integration energy investigation"><h3>Did the system change, or did the numerical rule change?</h3><p>First test cooling: x′=−2x, x(0)=8. All updates run to t=6. Inspect which step sizes make the Euler multiplier negative or give it magnitude at least one.</p><Choice label="Cooling integration step" numeric value={coolingStep} onChange={setCoolingStep} choices={[[0.25, 'h=0.25'], [0.5, 'h=0.5'], [1, 'h=1'], [1.5, 'h=1.5']]} />
    <Plot label="Exact cooling and Euler updates at the same times" horizontal="time t" vertical="excess temperature x" xDomain={[0, 6]} yDomain={[-coolingLimit * 1.05, coolingLimit * 1.05]} series={[{ points: cooling.rows.map(row => [row.time, row.exact]), tone: 'green' }, { points: cooling.rows.map(row => [row.time, row.euler]), dashed: true }]} />
    <Metrics rows={[[ 'Euler multiplier', format(cooling.multiplier)], ['Exact multiplier per interval', format(cooling.exactMultiplier)], ['Euler strictly attracting?', cooling.attractingEuler ? 'Yes' : 'No']]} />
    <p>Green connects exact samples; amber dashed connects discrete Euler states, not the path followed between updates. The temperature axis rescales to include all iterates. Even h=0.5, which jumps to zero immediately, is not the exact cooling law.</p>
    <p>Now keep the oscillator q′=p, p′=−q and initial state (1,0). Its exact energy E=(q²+p²)/2 stays at 0.5. Compare methods over the same t=20 horizon.</p><Choice label="Oscillator integration step" numeric value={step} onChange={setStep} choices={[[0.1, 'h=0.1'], [0.2, 'h=0.2'], [0.25, 'h=0.25'], [0.5, 'h=0.5']]} />
    <Plot label="Exact and numerical oscillator energies on a logarithmic scale" horizontal="time t" vertical="log₁₀ energy E" xDomain={[0, 20]} yDomain={[-0.5, 4]} series={[{ points: energy.rows.map(row => [row.time, Math.log10(row.eulerEnergy)]), dashed: true }, { points: energy.rows.map(row => [row.time, Math.log10(row.symplecticEnergy)]), tone: 'green' }, { points: [[0, Math.log10(0.5)], [20, Math.log10(0.5)]], tone: 'blue', dashed: true }]} />
    <Plot label="Ordinary and modified symplectic energy on a linear scale" horizontal="time t" vertical="energy (linear, dimensionless)" xDomain={[0, 20]} yDomain={[0.35, 0.7]} series={[{ points: energy.rows.map(row => [row.time, row.symplecticEnergy]), tone: 'green' }, { points: energy.rows.map(row => [row.time, row.modifiedEnergy]), tone: 'blue', dashed: true }]} />
    <div aria-live="polite"><Metrics rows={[[ 'Steps to t=20', energy.steps], ['Final Euler E', format(final.eulerEnergy)], ['Final symplectic E', format(final.symplecticEnergy)], ['Final modified E', format(final.modifiedEnergy)]]} /></div>
    <p>Amber dashed: forward Euler. Green: kick-then-drift symplectic Euler. Blue dashed in the upper plot: exact energy; in the lower plot: modified energy (q²+p²−hqp)/2. They happen to share the initial value 0.5 for (1,0), but are different functions. Symplectic Euler preserves the modified form up to arithmetic error; ordinary energy oscillates rather than staying exactly constant.</p><button onClick={() => { setStep(0.2); setCoolingStep(0.5); }}>Reset integration</button>
  </section>;
}
