import { useId, useMemo, useState } from 'react';
import { bvpFamily, coolingAmplification, coolingState, forcingTimeline, formatOdeNumber, linearSystemState, logisticState, numericalCooling, odeModePresets, oscillatorState, shearSchedule, waitingState } from '../../data/ordinary-differential-equations-models.js';
import './ordinary-differential-equations-labs.css';
const colors = ['#e7bd53', '#80c9cf', '#c3a1df', '#92c789'];
const samples = (end, compute, count = 80) => Array.from({
  length: count + 1
}, (_, index) => compute(end * index / count));
const number = formatOdeNumber;
function Choice({
  label,
  value,
  options,
  onChange
}) {
  return <label className="ode-control"><span>{label}</span><select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{options.map(([key, title]) => <option key={key} value={key}>{title}</option>)}</select></label>;
}
function Slider({
  label,
  value,
  min = 0,
  max,
  step = 0.1,
  onChange
}) {
  return <label className="ode-control"><span>{label}: <output>{number(value)}</output></span><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Plot({
  label,
  series,
  xLabel = 'time',
  yLabel = 'state',
  xDomain,
  yDomain,
  point
}) {
  const clipId = `ode-${useId().replace(/:/g, '')}`;
  const allPoints = series.flatMap(line => line.points).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  const xMinimum = xDomain?.[0] ?? Math.min(0, ...allPoints.map(([x]) => x));
  const xMaximum = xDomain?.[1] ?? Math.max(1, ...allPoints.map(([x]) => x));
  const low = yDomain?.[0] ?? Math.min(0, ...allPoints.map(([, y]) => y));
  const high = yDomain?.[1] ?? Math.max(1, ...allPoints.map(([, y]) => y));
  const padding = yDomain ? 0 : 0.06 * (high - low);
  const yMinimum = low - padding;
  const yMaximum = high + padding;
  const xPosition = value => 48 + (value - xMinimum) / (xMaximum - xMinimum) * 218;
  const yPosition = value => 182 - (value - yMinimum) / (yMaximum - yMinimum) * 144;
  const path = points => points.map(([x, y], index) => `${index ? 'L' : 'M'}${xPosition(x)},${yPosition(y)}`).join(' ');
  return <figure className="ode-plot"><svg viewBox="0 0 282 220" role="img" aria-label={label}>
    <title>{label}</title><defs><clipPath id={clipId}><rect x="47" y="37" width="220" height="146" /></clipPath></defs>
    <text x="48" y="18">{yLabel}</text>
    {[0, 0.5, 1].map(fraction => {
        const y = yMinimum + fraction * (yMaximum - yMinimum);
        return <g key={fraction}><line x1="48" x2="266" y1={yPosition(y)} y2={yPosition(y)} className="ode-grid" /><text x="42" y={yPosition(y) + 4} textAnchor="end">{number(y, 2)}</text></g>;
      })}
    {[0, 0.5, 1].map(fraction => {
        const x = xMinimum + fraction * (xMaximum - xMinimum);
        return <text key={fraction} x={xPosition(x)} y="199" textAnchor={fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle'}>{number(x, 2)}</text>;
      })}
    <text x="157" y="216" textAnchor="middle">{xLabel}</text>
    <g clipPath={`url(#${clipId})`}>{series.map((line, index) => <path key={line.label} d={path(line.points)} fill="none" stroke={line.color || colors[index % colors.length]} strokeWidth="2" strokeDasharray={line.dashed ? '5 3' : undefined} />)}</g>
    {series.map((line, index) => {
      const first = line.points[0];
      if (!first || !line.points.every(([x, y]) => x === first[0] && y === first[1])) return null;
      return <circle key={line.label} data-stationary-trajectory={line.label} cx={xPosition(first[0])} cy={yPosition(first[1])} r="3.5" fill={line.color || colors[index % colors.length]}><title>{line.label}: a stationary state</title></circle>;
    })}
      {point && <circle cx={xPosition(point[0])} cy={yPosition(point[1])} r="4" fill="#fff" stroke="#191b1c" />}
  </svg><figcaption><span>{label}</span><span className="ode-legend">{series.map((line, index) => <span key={line.label}><i style={{
            background: line.color || colors[index % colors.length]
          }} />{line.label}</span>)}</span></figcaption></figure>;
}
function Vector({
  values,
  label
}) {
  return <div className="ode-vector"><span>{label}</span><strong>[{values.map(value => number(value)).join(', ')}]</strong></div>;
}
export function ThermalBalanceFigure() {
  return <figure className="ode-balance"><div className="ode-balance-flow"><span><strong>Heater</strong><br />P watts in</span><span aria-hidden="true">→</span><span className="ode-object"><strong>Stored excess energy</strong><br />C × x joules</span><span aria-hidden="true">→</span><span><strong>Room</strong><br />G × x watts out</span></div><figcaption>Rate of stored energy = input − loss. With seconds, Cx′ = P − Gx. With minutes, multiply the right-hand side by 60. The arrows describe energy transfer, not temperature flowing between boxes.</figcaption></figure>;
}
export function RateFieldLab() {
  const [kind, setKind] = useState('cooling');
  const [initial, setInitial] = useState(10);
  const [time, setTime] = useState(1);
  const compute = value => kind === 'cooling' ? coolingState(value, initial, 0.5, 0) : logisticState(value, initial, 1, 10);
  const current = compute(time);
  const yMaximum = 20;
  const field = [];
  for (let column = 0; column <= 6; column += 1) {
    for (let row = 0; row <= 5; row += 1) {
      const state = row * 4;
      const slope = kind === 'cooling' ? -0.5 * state : state * (1 - state / 10);
      const pixelSlope = -slope * (142 / yMaximum) / (210 / 6);
      const length = Math.hypot(1, pixelSlope);
      const dx = 7 / length;
      const dy = 7 * pixelSlope / length;
      field.push(<line key={`${column}-${row}`} x1={49 + column * 35 - dx} x2={49 + column * 35 + dx} y1={180 - row * 28.4 - dy} y2={180 - row * 28.4 + dy} stroke="#777f85" strokeWidth="1.2" />);
    }
  }
  const curve = samples(6, value => [49 + value * 35, 180 - compute(value).state * 142 / yMaximum]);
  return <section className="ode-lab" aria-label="Rate and direction field investigation"><h3>Follow a slope, not a one-minute jump</h3><p>Inspect whether the selected state will rise, fall or stay fixed. Change the initial state while keeping the rate law fixed.</p><div className="ode-controls"><Choice label="Rate law" value={kind} options={[["cooling", "Cooling: x′ = −0.5x"], ["logistic", "Logistic: x′ = x(1 − x/10)"]]} onChange={setKind} /><Slider label="Initial state" value={initial} max={20} step={1} onChange={setInitial} /><Slider label="Inspection time" value={time} max={6} onChange={setTime} /><button onClick={() => {
        setKind('cooling');
        setInitial(10);
        setTime(1);
      }}>Reset rate field</button></div>
    <figure className="ode-plot"><svg viewBox="0 0 282 220" role="img" aria-label="A time-state slope field with an exact solution curve"><title>Same rate law at every point; one initial value chooses a curve</title><text x="49" y="18">state x</text>{field}<path d={curve.map(([x, y], index) => `${index ? 'L' : 'M'}${x},${y}`).join(' ')} fill="none" stroke={colors[0]} strokeWidth="2.5" /><circle cx={49 + time * 35} cy={180 - current.state * 142 / 20} r="4" fill="white" />{[0, 10, 20].map(value => <text key={value} x="40" y={184 - value * 142 / 20} textAnchor="end">{value}</text>)}{[0, 3, 6].map(value => <text key={value} x={49 + value * 35} y="199" textAnchor="middle">{value}</text>)}<text x="155" y="217" textAnchor="middle">time (model units)</text></svg><figcaption>Segments have equal drawn length; their angle encodes dx/dt in these axis units. The gold curve is evaluated analytically, not produced by stepping between segments.</figcaption></figure>
    <p className="ode-readout" aria-live="polite">At t = {number(time)}, x = {number(current.state)} and x′ = {number(current.derivative)}. {current.derivative > 0 ? 'The state is rising.' : current.derivative < 0 ? 'The state is falling.' : 'This state is an equilibrium.'}</p><p>For the logistic law, 0 and 10 are equilibrium states. Values between them rise; values above 10 fall. These exact solution curves cannot cross an equilibrium under this uniquely solvable rate law.</p></section>;
}
export function WaitingSolutionsLab() {
  const [departure, setDeparture] = useState(1);
  const series = [0, 1, 2].map(value => ({
    label: `depart at ${value}`,
    points: samples(3, time => [time, waitingState(time, value).state])
  }));
  series.push({
    label: 'your departure',
    color: '#ffffff',
    dashed: true,
    points: samples(3, time => [time, waitingState(time, departure).state])
  });
  return <section className="ode-lab" aria-label="Nonunique waiting solutions investigation"><h3>Same initial data, different departures</h3><p>Every curve obeys x′ = 2√|x| and x(0) = 0. Inspect what changes when the departure time changes, then inspect the family.</p><div className="ode-controls"><Slider label="Departure time" value={departure} max={3} onChange={setDeparture} /><button onClick={() => setDeparture(1)}>Reset waiting family</button></div><Plot label="Analytical waiting solutions" series={series} xDomain={[0, 3]} yDomain={[0, 9]} /><p aria-live="polite">Your solution stays at zero until t = {number(departure)}. At t = 3 it equals {number(waitingState(3, departure).state)}. Its derivative is zero at departure on both sides.</p><p>The square-root rate is continuous but is not locally Lipschitz in x at zero. Seeing one numerical trajectory would not rule out these others.</p></section>;
}
export function OscillatorLab() {
  const [damping, setDamping] = useState(2);
  const [velocity, setVelocity] = useState(0);
  const [time, setTime] = useState(1);
  const trajectory = useMemo(() => samples(6, value => oscillatorState(value, damping, 1, velocity)), [damping, velocity]);
  const current = oscillatorState(time, damping, 1, velocity);
  return <section className="ode-lab" aria-label="Position velocity and energy investigation"><h3>One state, three views of the same motion</h3><p>Keep m = 1 kg, k = 4 N/m and q(0) = 1 m fixed. Inspect what an initial velocity changes that an initial position cannot specify alone.</p><div className="ode-controls"><Choice label="Damping c" value={damping} options={[[0, '0: undamped'], [2, '2: underdamped'], [4, '4: critical'], [6, '6: overdamped']]} onChange={value => setDamping(Number(value))} /><Choice label="Initial velocity" value={velocity} options={[[-2, '−2 m/s'], [0, '0 m/s'], [2, '+2 m/s']]} onChange={value => setVelocity(Number(value))} /><Slider label="Motion inspection time" value={time} max={6} onChange={setTime} /><button onClick={() => {
        setDamping(2);
        setVelocity(0);
        setTime(1);
      }}>Reset oscillator</button></div><div className="ode-two-plots"><Plot label="Position over time" series={[{
        label: 'q(t)',
        points: trajectory.map(value => [value.time, value.position])
      }]} point={[time, current.position]} xLabel="seconds" yLabel="position (m)" /><Plot label="Phase curve: velocity versus position" series={[{
        label: '(q(t), v(t))',
        points: trajectory.map(value => [value.position, value.velocity])
      }]} point={[current.position, current.velocity]} xLabel="position (m)" yLabel="velocity (m/s)" /></div><div className="ode-state-strip" aria-live="polite"><span>q = {number(current.position)} m</span><span>v = {number(current.velocity)} m/s</span><span>E = {number(current.energy)} J</span><span>E′ = {number(current.energyRate)} W</span></div><p>Both white markers represent the same instant. A repeated position can have a different velocity. With no forcing, E′ = −cv²: damping cannot increase this mechanical energy. A moment with v = 0 gives E′ = 0 even if damping is present.</p></section>;
}
export function FundamentalMatrixLab() {
  const [preset, setPreset] = useState('jordan');
  const [initial, setInitial] = useState('second');
  const [time, setTime] = useState(1);
  const matrix = odeModePresets[preset].matrix;
  const initialVector = {
    first: [1, 0],
    second: [0, 1],
    sum: [1, 1]
  }[initial];
  const current = linearSystemState(matrix, initialVector, time);
  const series = [[1, 0], [0, 1], initialVector].map((vector, index) => ({
    label: index === 0 ? 'initial e₁' : index === 1 ? 'initial e₂' : 'selected initial state',
    dashed: index === 2,
    points: samples(3, value => linearSystemState(matrix, vector, value).state)
  }));
  return <section className="ode-lab" aria-label="Fundamental matrix columns investigation"><h3>Build a solution from evolved initial columns</h3><p>Each column of eᵗᴬ is the result of starting from one coordinate vector. Inspect the result for e₁ + e₂ when selecting it.</p><div className="ode-controls"><Choice label="Linear system" value={preset} options={Object.entries(odeModePresets).map(([key, value]) => [key, value.label])} onChange={setPreset} /><Choice label="Initial vector" value={initial} options={[["first", "e₁ = [1, 0]"], ["second", "e₂ = [0, 1]"], ["sum", "e₁ + e₂ = [1, 1]"]]} onChange={setInitial} /><Slider label="Matrix inspection time" value={time} max={3} onChange={setTime} /><button onClick={() => {
        setPreset('jordan');
        setInitial('second');
        setTime(1);
      }}>Reset matrix columns</button></div><div className="ode-matrix-readout"><Vector label="A row 1" values={matrix[0]} /><Vector label="A row 2" values={matrix[1]} /><Vector label="eᵗᴬ column 1" values={current.transition.map(row => row[0])} /><Vector label="eᵗᴬ column 2" values={current.transition.map(row => row[1])} /></div><Plot label="Full state-plane trajectories from t = 0 to 3" series={series} point={current.state} xLabel="first coordinate" yLabel="second coordinate" /><p aria-live="polite">At t = {number(time)}, the selected state is [{current.state.map(value => number(value)).join(', ')}]. Axes rescale to the complete displayed trajectories; compare their numeric ticks as well as their shapes.</p><p>{preset === 'saddle' ? 'Starting on e₂ hides the growing e₁ mode. One decaying trajectory cannot establish stability of every initial state.' : preset === 'nilpotent' ? 'Both eigenvalues are zero, but starting on e₂ gives [t, 1]. The nontrivial repeated block creates unbounded drift.' : preset === 'jordan' ? 'From e₂, the state is [3t exp(−t), exp(−t)]. The first coordinate is forced by the second even though both eigenvalues are −1.' : 'The columns use the same transition matrix. Their sum solves the system from the sum of their initial states.'}</p></section>;
}
export function ForcingTimelineLab() {
  const [firstPower, setFirstPower] = useState(20);
  const [secondPower, setSecondPower] = useState(0);
  const [switchTime, setSwitchTime] = useState(3);
  const [time, setTime] = useState(6);
  const compute = value => forcingTimeline(value, 40, firstPower, secondPower, switchTime);
  const current = compute(time);
  const trajectory = samples(6, compute);
  const series = [{
    label: 'total state',
    points: trajectory.map(value => [value.time, value.state])
  }, {
    label: 'initial response',
    points: trajectory.map(value => [value.time, value.initialContribution])
  }, {
    label: 'first input interval',
    points: trajectory.map(value => [value.time, value.firstContribution])
  }, {
    label: 'second input interval',
    points: trajectory.map(value => [value.time, value.secondContribution])
  }];
  return <section className="ode-lab" aria-label="Initial and forced response investigation"><h3>See when the input enters—and how long it decays</h3><p>Initial excess temperature remains 40 K. Inspect whether moving the same heating interval later leaves a warmer final object.</p><div className="ode-controls"><Slider label="First interval power" value={firstPower} max={40} step={5} onChange={setFirstPower} /><Slider label="Second interval power" value={secondPower} max={40} step={5} onChange={setSecondPower} /><Slider label="Input switch time" value={switchTime} max={6} onChange={setSwitchTime} /><Slider label="Response inspection time" value={time} max={6} onChange={setTime} /><button onClick={() => {
        setFirstPower(20);
        setSecondPower(0);
        setSwitchTime(3);
        setTime(6);
      }}>Reset input response</button></div><div className="ode-input-timeline"><span style={{
        flexGrow: switchTime || 0.001
      }}>{firstPower} W<br />0 → {number(switchTime)} min</span><span style={{
        flexGrow: 6 - switchTime || 0.001
      }}>{secondPower} W<br />{number(switchTime)} → 6 min</span></div><Plot label="Analytical initial and input contributions" series={series} point={[time, current.state]} xLabel="minutes" yLabel="excess temperature (K)" /><p className="ode-readout" aria-live="polite">At {number(time)} min: {number(current.initialContribution)} + {number(current.firstContribution)} + {number(current.secondContribution)} = {number(current.state)} K.</p><p>At the switch the state is continuous. Its derivative can jump. The second interval contributes nothing before it begins; the first interval's contribution keeps cooling afterward. These are exact formulas for the declared piecewise-constant input.</p></section>;
}
export function TimeOrderFigure() {
  const [first, setFirst] = useState('upper');
  const [initial, setInitial] = useState('first');
  const result = shearSchedule(initial === 'first' ? [1, 0] : [0, 1], first);
  return <section className="ode-lab ode-shear" aria-label="Order of two continuous stages investigation"><h3>Swap stages while keeping their total durations fixed</h3><p>U changes the first coordinate using the second. V changes the second using the first. Each rate law acts for one unit of time.</p><div className="ode-controls"><Choice label="Stage order" value={first} options={[["upper", "U then V"], ["lower", "V then U"]]} onChange={setFirst} /><Choice label="Stage initial state" value={initial} options={[["first", "[1, 0]"], ["second", "[0, 1]"]]} onChange={setInitial} /><button onClick={() => {
        setFirst('upper');
        setInitial('first');
      }}>Reset stage order</button></div><div className="ode-stage-chain"><Vector label="Start" values={result.initial} /><span>↓ after {first === 'upper' ? 'U' : 'V'}</span><Vector label="Intermediate" values={result.intermediate} /><span>↓ after {first === 'upper' ? 'V' : 'U'}</span><Vector label="Finish" values={result.final} /></div><p aria-live="polite">Final state: [{result.final.join(', ')}]. Both schedules spend one unit under each matrix. The later matrix exponential multiplies on the left; changing order changes which intermediate state it acts on.</p></section>;
}
export function OdeNumericalLab() {
  const [method, setMethod] = useState('euler');
  const [step, setStep] = useState(0.7);
  const [rate, setRate] = useState(0.2);
  const result = numericalCooling(method, step, rate);
  const firstStep = result.history[1];
  const multiplier = coolingAmplification(rate, step);
  const exact = samples(5, time => [time, 40 * Math.exp(-rate * time)]);
  return <section className="ode-lab" aria-label="Numerical stages and error investigation"><h3>Probe slopes, accept one state, measure its error</h3><p>All runs start at 40 and request t = 5. Inspect the effect of halving h. Then switch to the faster decay and inspect stability as a separate question.</p><div className="ode-controls"><Choice label="Step method" value={method} options={[["euler", "Euler: one slope"], ["midpoint", "Explicit midpoint: two slopes"], ["rk4", "Classical RK4: four slopes"]]} onChange={setMethod} /><Choice label="Requested step h" value={step} options={[[0.175, '0.175'], [0.35, '0.35'], [0.7, '0.7'], [0.5, '0.5'], [0.6, '0.6']]} onChange={value => setStep(Number(value))} /><Choice label="Decay rate k" value={rate} options={[[0.2, '0.2: thermal rate'], [4, '4: faster decay']]} onChange={value => setRate(Number(value))} /><button onClick={() => {
        setMethod('euler');
        setStep(0.7);
        setRate(0.2);
      }}>Reset numerical steps</button></div><Plot label="Computed states against the analytical solution" series={[{
      label: method,
      points: result.history.map(value => [value.time, value.state])
    }, {
      label: 'exact',
      points: exact
    }]} yLabel="state" /><div className="ode-table-wrap"><table><caption>First accepted step: actual derivative probes</caption><thead><tr><th>Stage</th><th>Time</th><th>Trial state</th><th>Slope</th></tr></thead><tbody>{firstStep.stages.map((stage, index) => <tr key={index}><th>{index + 1}</th><td>{number(stage.time)}</td><td>{number(stage.state)}</td><td>{number(stage.slope)}</td></tr>)}</tbody></table></div><p>Weighted slope = {number(firstStep.average)}. Accept 40 + {number(firstStep.step)} × ({number(firstStep.average)}) = {number(firstStep.state)}. Intermediate trial states are derivative probes; they are not all accepted points.</p><p className="ode-readout" aria-live="polite">{result.status}; {result.steps} steps; final time {number(result.endTime)}. Exact final state {number(result.exact)}; signed final error {number(result.signedError)}.</p><p>For Euler at the requested full step, kh = {number(multiplier.product)} and the factor is {number(multiplier.explicit)}. {multiplier.explicitStrictDecay ? 'Its magnitude is below one.' : 'This is outside strict decay for positive k.'} The exact factor is {number(multiplier.exact)} and is positive. The last numerical step is shortened when needed; the graph joins computed samples for reading and does not certify the solution between them.</p></section>;
}
export function BoundaryValueLab() {
  const [endpoint, setEndpoint] = useState('pi');
  const [target, setTarget] = useState(0);
  const [slope, setSlope] = useState(1);
  const result = bvpFamily(endpoint, target, slope);
  const curves = [-1, 0, 1, 2].map(value => ({
    label: `initial slope ${value}`,
    points: samples(result.endTime, time => [time, value * Math.sin(time)])
  }));
  return <section className="ode-lab" aria-label="Boundary conditions and uniqueness investigation"><h3>Two endpoint conditions do not always choose one solution</h3><p>Every curve solves y″ + y = 0 and y(0) = 0. Inspect which endpoint data can distinguish the initial slope.</p><div className="ode-controls"><Choice label="Right endpoint" value={endpoint} options={[["pi", "Exactly π"], ["half-pi", "Exactly π/2"]]} onChange={setEndpoint} /><Choice label="Requested endpoint value" value={target} options={[[0, '0'], [1, '1']]} onChange={value => setTarget(Number(value))} /><Slider label="Candidate initial slope" value={slope} min={-2} max={2} step={0.5} onChange={setSlope} /><button onClick={() => {
        setEndpoint('pi');
        setTarget(0);
        setSlope(1);
      }}>Reset boundary conditions</button></div><Plot label="The family b sin(t) and the requested endpoint" series={[...curves, {
      label: 'candidate',
      color: '#fff',
      dashed: true,
      points: samples(result.endTime, time => [time, slope * Math.sin(time)])
    }]} point={[result.endTime, target]} xLabel="t (radians)" yLabel="y" /><p aria-live="polite"><strong>{result.classification}.</strong> {endpoint === 'pi' ? 'At the exact endpoint π, every b sin(t) has value zero.' : `At π/2, the endpoint value equals b, so the required slope is ${target}.`} The white endpoint marker is the requested value.</p><p>Classification uses the exact symbolic endpoints, not a tolerance test on the computer's approximation of sin(π). A tiny residual from floating-point trigonometry does not turn infinitely many exact solutions into one.</p></section>;
}
