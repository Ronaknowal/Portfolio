import { useId, useMemo, useState } from 'react';
import { controlVolume, transportState, heatBoundaryState, heatValue, boundaryMode, heatKernel, waveState, standingWave, poissonState, POISSON_SOURCES, harmonicState, harmonicValue, pointSourceState, burgersState, inverseHeatState, periodicDepth, forcedRod } from '../../data/pde-models.js';
import './pde-labs.css';
const colors = ['#e8b34b', '#79c9a1', '#94b8d8', '#e39584'];
const format = (value, digits = 4) => value === null ? 'not applicable' : Math.abs(value) > 1e5 || value !== 0 && Math.abs(value) < 1e-5 ? value.toExponential(3) : Number(value.toFixed(digits)).toString();
const series = (rows, key) => rows.map(point => [point.x, point[key]]);
const linspace = (a, b, n) => Array.from({
  length: n + 1
}, (_, i) => a + (b - a) * i / n);
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = .01,
  display
}) {
  return <label className="pde-range"><span>{label}<output>{display ?? format(value)}</output></span><input type="range" aria-label={label} value={value} onChange={event => onChange(Number(event.target.value))} min={min} max={max} step={step} /></label>;
}
function Choice({
  label,
  value,
  onChange,
  options
}) {
  return <label className="pde-choice"><span>{label}</span><select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}</select></label>;
}
function Lab({
  id,
  title,
  children
}) {
  return <section className="pde-lab" data-pde-lab={id} aria-label={title}><h3>{title}</h3>{children}</section>;
}
function Values({
  items
}) {
  return <dl className="pde-values">{items.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Legend({
  labels
}) {
  return <ul className="pde-legend">{labels.map((label, index) => <li key={label}><span style={{
        borderColor: colors[index],
        borderTopStyle: index > 1 ? 'dashed' : 'solid'
      }} />{label}</li>)}</ul>;
}
function Data({
  headers,
  rows,
  title = 'Inspect calculated values'
}) {
  return <details className="pde-data"><summary>{title}</summary><div className="pde-table-scroll"><table><thead><tr>{headers.map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{row.map((value, column) => <td key={column}>{value}</td>)}</tr>)}</tbody></table></div></details>;
}
function Plot({
  title,
  lines,
  xDomain,
  yDomain,
  xLabel = 'position x',
  yLabel = 'field u',
  cursor,
  children
}) {
  const all = lines.flat();
  const low = yDomain?.[0] ?? Math.min(0, ...all.map(point => point[1]));
  const high = yDomain?.[1] ?? Math.max(0, ...all.map(point => point[1]));
  const padding = yDomain ? 0 : Math.max(.1, high - low) * .12;
  const ymin = low - padding,
    ymax = high + padding;
  const x = value => 47 + 251 * (value - xDomain[0]) / (xDomain[1] - xDomain[0]);
  const y = value => 166 - 138 * (value - ymin) / (ymax - ymin);
  return <figure className="pde-figure"><figcaption>{title}</figcaption><svg viewBox="0 0 320 220" role="img" aria-label={`${title}. ${xLabel}; ${yLabel}.`}>
    {[ymin, (ymin + ymax) / 2, ymax].map(value => <g key={value}><line x1="47" x2="298" y1={y(value)} y2={y(value)} className="pde-grid" /><text x="41" y={y(value) + 4} textAnchor="end" className="pde-tick">{format(value, 2)}</text></g>)}
    {[xDomain[0], (xDomain[0] + xDomain[1]) / 2, xDomain[1]].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="28" y2="166" className="pde-grid" /><text x={x(value)} y="187" textAnchor={value === xDomain[0] ? 'start' : value === xDomain[1] ? 'end' : 'middle'} className="pde-tick">{format(value, 2)}</text></g>)}
    {ymin < 0 && ymax > 0 && <line x1="47" x2="298" y1={y(0)} y2={y(0)} className="pde-axis" />}
    {lines.map((line, index) => <path key={index} data-pde-curve={index} d={line.map((point, k) => `${k ? 'L' : 'M'}${x(point[0]).toFixed(3)},${y(point[1]).toFixed(3)}`).join(' ')} fill="none" stroke={colors[index]} strokeWidth="2.3" strokeDasharray={index > 1 ? '5 3' : undefined} />)}
    {cursor !== undefined && <line x1={x(cursor)} x2={x(cursor)} y1="28" y2="166" className="pde-cursor" />}
    <text x="47" y="17">{yLabel}</text><text x="173" y="213" textAnchor="middle">{xLabel}</text>{children?.({
        x,
        y
      })}
  </svg></figure>;
}
export function FieldControlVolumeFigure() {
  const state = controlVolume();
  const marker = useId().replace(/:/g, '');
  return <section className="pde-inline" aria-label="A field and its control-volume balance"><div className="pde-two">
    <Plot title="One snapshot; the shaded slice is the control volume" lines={[series(state.profile, 'value')]} xDomain={[0, 1]} xLabel="normalized position" yLabel="temperature departure">{({
          x
        }) => <rect x={x(.25)} y="28" width={x(.75) - x(.25)} height="138" fill="#e8b34b" opacity=".08" />}</Plot>
    <figure className="pde-figure"><figcaption>Signed axial flux points left here</figcaption><svg viewBox="0 0 320 220" role="img" aria-label="Right-side incoming flux 6.5625 exceeds left-side outgoing flux 2.1875; source removes 3.125. Accumulation is 1.25."><defs><marker id={marker} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#79c9a1" /></marker></defs>
      <rect x="100" y="62" width="120" height="68" fill="#20251d" stroke="#949b89" /><text x="160" y="88" textAnchor="middle">stored heat</text><text x="160" y="113" textAnchor="middle">in this slice</text>
      {["M100 96 H32", "M288 96 H222"].map(path => <path key={path} d={path} stroke="#79c9a1" strokeWidth="3" markerEnd={`url(#${marker})`} />)}<text x="37" y="49">leaves</text><text x="237" y="49">enters</text>
      <text x="30" y="154">j(a)=−2.1875</text><text x="183" y="178">j(b)=−6.5625</text><text x="102" y="204">source: −3.125</text>
    </svg></figure>
  </div><p className="pde-balance">Accumulation = incoming − outgoing + source = 6.5625 − 2.1875 − 3.125 = <strong>1.25</strong>.</p><p>The arrows encode the actual negative axial flux in the polynomial example. Their lengths are diagram layout, not a flux scale. A sensor trace would instead fix x and follow u(x,t) through time.</p></section>;
}
export function RectangularFluxFigure() {
  const marker = useId().replace(/:/g, '') + '-rectangle-flux';
  return <section className="pde-inline"><figure className="pde-figure pde-rectangle"><figcaption>Four faces become two local outflow differences</figcaption><svg viewBox="0 0 320 280" role="img" aria-label="Positive horizontal flux enters on the left and leaves on the right. Positive vertical flux enters below and leaves above. Outward total is right minus left plus top minus bottom."><defs><marker id={marker} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#79c9a1" /></marker></defs><rect x="105" y="85" width="110" height="92" fill="#24291e" stroke="#a5a999" />{["M45 130 H102", "M218 130 H282", "M160 84 V42", "M160 219 V180"].map(path => <path key={path} d={path} fill="none" stroke="#79c9a1" strokeWidth="3" markerEnd={`url(#${marker})`} />)}<text x="160" y="24" textAnchor="middle">top out: j₂(x,d)</text><text x="12" y="106">left in</text><text x="246" y="106">right out</text><text x="12" y="160">j₁(a,y)</text><text x="242" y="160">j₁(b,y)</text><text x="160" y="135" textAnchor="middle">small region</text><text x="160" y="242" textAnchor="middle">bottom in: j₂(x,c)</text><text x="160" y="271" textAnchor="middle">x rightward · y upward</text></svg></figure><p>For the rectangle [a,b] × [c,d], integrate each component along its face. The arrows depict positive axial components; a negative component reverses its actual flow. Subtracting opposite-face values and applying the FTC gives the area integral of ∂ₓj₁ + ∂ᵧj₂.</p></section>;
}
function SpaceTime({
  title,
  xDomain = [0, 1],
  timeMax = 1,
  trajectories = [],
  point,
  foot,
  band,
  children
}) {
  const x = value => 47 + 251 * (value - xDomain[0]) / (xDomain[1] - xDomain[0]);
  const y = value => 168 - 138 * value / timeMax;
  return <figure className="pde-figure"><figcaption>{title}</figcaption><svg viewBox="0 0 320 220" role="img" aria-label={`${title}. Horizontal position, vertical time.`}>
    {band && <path d={`M${x(band[0])} ${y(0)} L${x(point[0])} ${y(point[1])} L${x(band[1])} ${y(0)} Z`} fill="#79c9a1" opacity=".15" />}
    <path d="M47 30 V168 H298" fill="none" className="pde-axis" />
    {trajectories.map((line, i) => <path key={i} d={line.map((p, k) => `${k ? 'L' : 'M'}${x(p[0])},${y(p[1])}`).join(' ')} fill="none" stroke="#799584" strokeWidth="1.1" />)}
    {foot && <line x1={x(foot[0])} x2={x(point[0])} y1={y(foot[1])} y2={y(point[1])} stroke="#e8b34b" strokeWidth="3" />}
    {point && <circle cx={x(point[0])} cy={y(point[1])} r="5" fill="#e8b34b" stroke="#181a15" />}
    {foot && <circle cx={x(foot[0])} cy={y(foot[1])} r="5" fill="#79c9a1" />}
    <text x="47" y="17">time</text><text x="39" y="171" textAnchor="end">0</text><text x="39" y="35" textAnchor="end">{timeMax}</text>
    <text x="47" y="188">{xDomain[0]}</text><text x="298" y="188" textAnchor="end">{xDomain[1]}</text><text x="172" y="213" textAnchor="middle">position x</text>{children?.({
        x,
        y
      })}
  </svg></figure>;
}
export function BoundaryDataFigure() {
  return <section className="pde-inline"><div className="pde-two">
    <SpaceTime title="Heat: initial line and two spatial sides">{({
          x,
          y
        }) => <><path d={`M${x(0)} ${y(1)} V${y(0)} H${x(1)} V${y(1)}`} fill="none" stroke="#79c9a1" strokeWidth="3" /><text x="113" y="57">unknown field</text><text x="101" y="151">initial profile f(x)</text><text x="54" y="91">left</text><text x="256" y="91">right</text></>}</SpaceTime>
    <SpaceTime title="Positive transport: initial line and inflow">{({
          x,
          y
        }) => <><path d={`M${x(0)} ${y(1)} V${y(0)} H${x(1)}`} fill="none" stroke="#79c9a1" strokeWidth="3" /><line x1={x(1)} x2={x(1)} y1={y(0)} y2={y(1)} stroke="#8d998c" strokeDasharray="4 4" /><text x="64" y="66">inflow</text><text x="243" y="103">outflow</text><text x="105" y="151">initial profile f(x)</text></>}</SpaceTime>
  </div><p>The colored edges mark supplied data. The upper edge is a later prediction, not extra prescribed data. A wave problem additionally supplies an initial velocity along the bottom line. The dashed transport edge is where the solution leaves the domain.</p></section>;
}
export function TransportCharacteristicLab() {
  const [time, setTime] = useState(.35),
    [position, setPosition] = useState(.6),
    [curvature, setCurvature] = useState(2);
  const state = useMemo(() => transportState(time, position, curvature), [time, position, curvature]);
  const trajectories = linspace(-1, 1, 9).map(start => [[Math.max(0, start), Math.max(0, -start)], [Math.min(1, start + 1), Math.min(1, 1 - start)]]);
  return <Lab id="transport" title="Trace the observation to the data that determine it"><p>Inspect which datum the highlighted point will reach when traced backward. The speed is fixed at one normalized length per time unit.</p>
    <div className="pde-controls"><Range label="Transport time" value={time} onChange={setTime} min={0} max={1} /><Range label="Transport observation x" value={position} onChange={setPosition} min={0} max={1} /><Choice label="Inflow history" value={curvature} onChange={value => setCurvature(Number(value))} options={[[0, 'g(t)=1−t'], [2, 'g(t)=1−t+2t²']]} /></div>
    <div className="pde-two"><SpaceTime title="The backward characteristic" trajectories={trajectories} point={[position, time]} foot={[state.dataPosition, state.dataTime]} /><Plot title="Current transported profile" lines={[series(state.profile, 'value')]} xDomain={[0, 1]} cursor={position} /></div>
    <Values items={[["Determining data", state.fromInitial ? 'Initial profile at t=0' : 'Left inflow history'], ['Data coordinate', state.fromInitial ? `x=${format(state.dataPosition)}` : `t=${format(state.dataTime)}`], ['Observed value', format(state.value)], ['Total amount', format(state.mass)], ['Incoming minus outgoing', format(state.incoming - state.outgoing)], ['Rate of total change', format(state.massDerivative)]]} />
    <p>Changing the inflow affects only points whose backward path reaches that boundary. At x=t both data agree; the model labels that shared corner as initial data. Curves and amount are calculated from the piecewise analytic solution.</p>
    <button onClick={() => {
      setTime(.35);
      setPosition(.6);
      setCurvature(2);
    }}>Reset transport</button>
  </Lab>;
}
export function BoundaryModesFigure() {
  const d = [1, 2].map(n => boundaryMode('dirichlet', n));
  const n = [0, 1].map(k => boundaryMode('neumann', k));
  return <section className="pde-inline"><div className="pde-two"><div><Plot title="Fixed values: no nonzero constant mode" lines={d.map(row => series(row.profile, 'value'))} xDomain={[0, 1]} yDomain={[-1.2, 1.2]} /><Legend labels={['n=1: sin(πx)', 'n=2: sin(2πx)']} /><p>Both endpoint values are zero.</p></div><div><Plot title="Insulated ends: the constant mode survives" lines={n.map(row => series(row.profile, 'value'))} xDomain={[0, 1]} yDomain={[-1.2, 1.2]} /><Legend labels={['n=0: constant 1', 'n=1: cos(πx)']} /><p>Both endpoint slopes are zero.</p></div></div><p>These are shapes, not heat profiles required to be nonnegative. Their coefficients can combine them into the actual initial field. The eigenvalue multiplies each shape's time evolution.</p></section>;
}
export function HeatBoundaryLab() {
  const [theta, setTheta] = useState(.05),
    [position, setPosition] = useState(.5);
  const state = useMemo(() => heatBoundaryState(theta), [theta]);
  const selected = [heatValue(position, theta), heatValue(position, theta, 'neumann')];
  return <Lab id="heat" title="The same initial heat, two different exits"><p>Both rods start at sin²(πx). Inspect which mean can change. θ is normalized time αt/L²; the plotted x is x/L.</p>
    <div className="pde-controls">{theta > 0 && <Range label="Positive heat time theta" value={theta} onChange={setTheta} min={.002} max={.5} step={.002} />}<Range label="Heat inspection position" value={position} onChange={setPosition} min={0} max={1} /></div>
    <div className="pde-buttons"><button onClick={() => setTheta(0)} aria-pressed={theta === 0}>Show exact initial profile</button><button onClick={() => setTheta(.05)}>Compare at θ=0.05</button></div>
    <Plot title={theta === 0 ? 'Initial data agree in both rods' : 'Two analytic boundary solutions'} lines={[series(state.profile, 'dirichlet'), series(state.profile, 'neumann'), series(state.profile, 'initial')]} xDomain={[0, 1]} yDomain={[0, 1.08]} cursor={position} /><Legend labels={['Zero endpoint values', 'Zero endpoint flux', 'Initial profile']} />
    <Values items={[["Selected: fixed values", format(selected[0])], ['Selected: insulated', format(selected[1])], ['Mean: fixed values', format(state.dirichlet.mean)], ['Mean: insulated', format(state.neumann.mean)], ['Squared norm: fixed values', format(state.dirichlet.squaredNorm)], ['Squared norm: insulated', format(state.neumann.squaredNorm)], ['Total outward flux: fixed values', format(2 * state.dirichlet.leftOutflow)], ['Total outward flux: insulated', '0']]} />
    <p>{theta === 0 ? 'This view evaluates the exact initial function, not a truncated series. Its value and slope agree with both endpoint data; stronger corner-derivative conditions still differ.' : `The Dirichlet calculation keeps odd modes through n=63. The natural log of its analytic uniform truncation bound is ${format(state.logTailBound, 2)}. That bound excludes floating-point roundoff. The Neumann formula has only two terms.`}</p>
    <Data headers={['x/L', 'Fixed values', 'Insulated', 'Initial']} rows={state.profile.filter((_, i) => i % 16 === 0).map(p => [format(p.x), format(p.dirichlet, 7), format(p.neumann, 7), format(p.initial, 7)])} />
    <button onClick={() => {
      setTheta(.05);
      setPosition(.5);
    }}>Reset heat comparison</button>
  </Lab>;
}
export function HeatKernelFigure() {
  const times = [.1, .4, 1];
  return <section className="pde-inline"><Plot title="The same unit amount spreads over a wider region" lines={times.map(t => linspace(-4, 4, 160).map(x => [x, heatKernel(x, t, .5)]))} xDomain={[-4, 4]} xLabel="position x" yLabel="kernel density" /><Legend labels={times.map(t => `t=${t}, variance=${t}`)} /><p>Diffusivity is α=0.5. Each full curve has area 1 and variance 2αt, while its peak falls. The finite viewing window does not cut off the mathematical Gaussian support.</p></section>;
}
export function WaveConeLab() {
  const [time, setTime] = useState(.6),
    [position, setPosition] = useState(.4),
    [velocity, setVelocity] = useState(.5);
  const state = useMemo(() => waveState(time, position, velocity), [time, position, velocity]);
  return <Lab id="wave" title="A wave observation depends on an initial interval"><p>The displacement starts as a compact bump on [−1,1]; outside that interval it is zero. Set the initial velocity independently and inspect the two translated shape terms and the velocity integral.</p>
    <div className="pde-controls"><Range label="Wave time" value={time} onChange={setTime} min={0} max={1} step={.02} /><Range label="Wave observation x" value={position} onChange={setPosition} min={-2} max={2} step={.05} /><Choice label="Initial velocity" value={velocity} onChange={value => setVelocity(Number(value))} options={[[0, 'Zero everywhere'], [.5, 'Half the initial bump']]} /></div>
    <div className="pde-two"><SpaceTime title="Backward dependence cone, c=1" xDomain={[-3, 3]} point={[position, time]} band={[state.leftFoot, state.rightFoot]} trajectories={[[[state.leftFoot, 0], [position, time]], [[state.rightFoot, 0], [position, time]]]} /><Plot title="Displacement now" lines={[series(state.profile, 'total'), series(state.profile, 'rightMoving'), series(state.profile, 'leftMoving'), series(state.profile, 'velocityContribution')]} xDomain={[-3, 3]} cursor={position} yLabel="displacement" /></div>
    <Legend labels={['Total displacement', 'Right-moving half', 'Left-moving half', 'Velocity integral']} /><Values items={[["Initial interval", `[${format(state.leftFoot)}, ${format(state.rightFoot)}]`], ['Right-moving contribution', format(state.rightMoving)], ['Left-moving contribution', format(state.leftMoving)], ['Velocity contribution', format(state.velocityContribution)], ['Total at observation', format(state.total)]]} />
    <p>The cone bounds influence, not a physical barrier. The displayed [−3,3] is a window into a whole-line problem. At t=0 the interval collapses and the velocity integral is zero; its time derivative supplies the prescribed initial velocity.</p>
    <Data headers={['x', 'Total', 'Velocity contribution']} rows={state.profile.filter((_, i) => i % 24 === 0).map(p => [format(p.x), format(p.total), format(p.velocityContribution)])} />
    <button onClick={() => {
      setTime(.6);
      setPosition(.4);
      setVelocity(.5);
    }}>Reset wave</button>
  </Lab>;
}
export function StandingWaveEnergyFigure() {
  const states = [0, .25, .5].map(standingWave);
  return <section className="pde-inline"><Plot title="Two fixed-end modes at three times" lines={states.map(row => series(row.profile, 'value'))} xDomain={[0, 1]} yDomain={[-1.2, 1.2]} yLabel="displacement" /><Legend labels={states.map(row => `t=${row.time}`)} /><Data title="Inspect kinetic and strain energy" headers={['t', 'Kinetic', 'Strain', 'Total']} rows={states.map(row => [row.time, format(row.kinetic), format(row.strain), format(row.total)])} /><p>Every endpoint stays fixed. The separately integrated kinetic and strain terms change, but their sum is 5π²/16≈{format(states[0].total)} for every time, by the derived energy identity.</p></section>;
}
export function PoissonCompatibilityLab() {
  const [source, setSource] = useState('uniform'),
    [boundary, setBoundary] = useState('neumann'),
    [left, setLeft] = useState(1),
    [right, setRight] = useState(1),
    [mean, setMean] = useState(0);
  const state = useMemo(() => poissonState(source, boundary, left, right, mean), [source, boundary, left, right, mean]);
  function changeBoundary(value) {
    setBoundary(value);
    setLeft(value === 'dirichlet' ? 0 : 1);
    setRight(value === 'dirichlet' ? 0 : 1);
  }
  const datum = boundary === 'neumann' ? 'outward flux' : 'value';
  return <Lab id="poisson" title="Can these sources and boundary data balance?"><p>For −u″=f on [0,1], k=1. Outward flux is u′(0) on the left and −u′(1) on the right. Test compatibility before choosing the mean of a Neumann solution.</p>
    <div className="pde-controls"><Choice label="Poisson source" value={source} onChange={setSource} options={Object.entries(POISSON_SOURCES).map(([key, value]) => [key, value.label])} /><Choice label="Poisson boundary type" value={boundary} onChange={changeBoundary} options={[['neumann', 'Prescribe outward fluxes'], ['dirichlet', 'Prescribe endpoint values']]} /><Range label={`Left ${datum}`} value={left} onChange={setLeft} min={-2} max={2} step={1} /><Range label={`Right ${datum}`} value={right} onChange={setRight} min={-2} max={2} step={1} />{boundary === 'neumann' && <Range label="Selected solution mean" value={mean} onChange={setMean} min={-1} max={1} step={1} />}</div>
    {state.compatible ? <><Plot title="A solution of the stated boundary problem" lines={[series(state.profile, 'value')]} xDomain={[0, 1]} /><Values items={[["Integrated source", format(state.sourceIntegral)], ['Left outward flux', format(state.leftOutflow)], ['Right outward flux', format(state.rightOutflow)], ['Solution mean', format(state.selectedMean)]]} /><p>{boundary === 'neumann' ? 'Changing the selected mean shifts every potential value by the same amount and leaves both fluxes unchanged. The mean chooses one member of the constant-offset family.' : 'Endpoint values select one solution. Its two outward fluxes are consequences, not extra independently prescribed constraints.'}</p><Data headers={['x', 'u', 'Axial flux j', 'Source f']} rows={state.profile.filter((_, i) => i % 16 === 0).map(p => [format(p.x), format(p.value), format(p.flux), format(p.source)])} /></> : <div className="pde-message" role="status"><strong>No steady solution for these data.</strong><p>Outward flux sum={left + right}, but integrated source={state.sourceIntegral}. The mismatch is {state.mismatch}. Choosing a different mean cannot repair this imbalance. A time-dependent field could accumulate the unmatched source; it would not be this steady problem.</p></div>}
    <button onClick={() => {
      setSource('uniform');
      setBoundary('neumann');
      setLeft(1);
      setRight(1);
      setMean(0);
    }}>Reset Poisson problem</button>
  </Lab>;
}
export function HarmonicInteriorLab() {
  const [frequency, setFrequency] = useState(1),
    [position, setPosition] = useState(.5),
    [depth, setDepth] = useState(.5);
  const state = useMemo(() => harmonicState(frequency), [frequency]);
  const slice = useMemo(() => linspace(0, 1, 96).map(y => [y, harmonicValue(position, y, frequency)]), [position, frequency]);
  const x = p => 47 + 240 * p,
    y = p => 269 - 240 * p;
  const cellColor = value => value >= 0 ? `rgb(${Math.round(35 + 180 * value)},${Math.round(43 + 116 * value)},${Math.round(39 + 15 * value)})` : `rgb(${Math.round(35 - 45 * value)},${Math.round(43 - 90 * value)},${Math.round(39 - 157 * value)})`;
  return <Lab id="harmonic" title="A boundary pattern reaches into a two-dimensional field"><p>The top boundary is sin(nπx); the other three sides are zero. Inspect what a higher n does to the amplitude halfway down from the top. Every view uses the same [−1,1] color scale.</p>
    <div className="pde-controls"><Choice label="Boundary harmonic n" value={frequency} onChange={v => setFrequency(Number(v))} options={[[1, 'n=1'], [3, 'n=3'], [5, 'n=5']]} /><Range label="Harmonic horizontal position" value={position} onChange={setPosition} min={0} max={1} /><Range label="Harmonic vertical position y" value={depth} onChange={setDepth} min={0} max={1} /></div>
    <div className="pde-two"><figure className="pde-figure"><figcaption>Analytic harmonic field on the square</figcaption><svg viewBox="0 0 320 325" role="img" aria-label={`Square harmonic field n=${frequency}, selected x=${position}, y=${depth}. Color is amplitude with fixed scale.`}>
      {state.cells.map(cell => <rect key={`${cell.row}-${cell.column}`} x={x(cell.column / 24)} y={y((cell.row + 1) / 24)} width="10.2" height="10.2" fill={cellColor(cell.value)} />)}
      <rect x="47" y="29" width="240" height="240" fill="none" stroke="#9da493" /><line x1={x(position)} x2={x(position)} y1="29" y2="269" className="pde-cursor" /><line x1="47" x2="287" y1={y(depth)} y2={y(depth)} className="pde-cursor" /><circle cx={x(position)} cy={y(depth)} r="5" fill="#f4e6cd" stroke="#161b16" />
      <text x="47" y="18">top: sin(nπx)</text><text x="40" y="34" textAnchor="end">1</text><text x="40" y="273" textAnchor="end">0</text><text x="13" y="156">y</text><text x="47" y="289">0</text><text x="287" y="289" textAnchor="end">1</text><text x="171" y="316" textAnchor="middle">x; blue −1 · dark 0 · gold +1</text>
    </svg></figure><Plot title={`Vertical slice at x=${format(position)}`} lines={[slice]} xDomain={[0, 1]} yDomain={[-1.08, 1.08]} xLabel="vertical position y" cursor={depth} /></div>
    <Values items={[["Selected field value", format(harmonicValue(position, depth, frequency), 7)], ['Half-depth envelope ratio', format(state.midpointRatio, 8)]]} /><p>The tiles sample the exact formula at their centers; they are not finite elements. The slice and selected point evaluate the formula directly. Finer boundary oscillations fade more rapidly into the interior, even though every top pattern has the same amplitude.</p>
    <Data headers={['y', 'Selected-x value']} rows={slice.filter((_, i) => i % 12 === 0).map(p => [format(p[0]), format(p[1], 8)])} />
    <button onClick={() => {
      setFrequency(1);
      setPosition(.5);
      setDepth(.5);
    }}>Reset harmonic field</button>
  </Lab>;
}
export function WeakSourceFigure() {
  const state = pointSourceState();
  return <section className="pde-inline"><div className="pde-two"><Plot title="Finite value, but a kink at the source" lines={[series(state.profile, 'value')]} xDomain={[0, 1]} yLabel="G(x, 1/3)" cursor={state.location} /><Plot title="The derivative jumps by −1" lines={[[[0, state.leftSlope], [state.location, state.leftSlope], [state.location, state.rightSlope], [1, state.rightSlope]]]} xDomain={[0, 1]} yDomain={[-.6, .9]} yLabel="slope G′" cursor={state.location} /></div><p>Left slope 2/3, right slope −1/3. The source is a unit point evaluation, not a finite-height sampled spike. Integrating G′v′ separately on the two sides leaves exactly v(1/3).</p></section>;
}
export function BurgersRiemannLab() {
  const [reversed, setReversed] = useState(false),
    [time, setTime] = useState(.5),
    [showExpansion, setShowExpansion] = useState(false);
  const left = reversed ? 0 : 2,
    right = reversed ? 2 : 0;
  const state = useMemo(() => burgersState(left, right, time, showExpansion), [left, right, time, showExpansion]);
  const trajectories = useMemo(() => {
    if (reversed && !showExpansion) return linspace(0, 2, 7).map(speed => [[0, 0], [speed, 1]]);
    if (reversed) return [...linspace(-2, -.1, 6).map(start => [[start, 0], [start, 1]]), ...linspace(.1, 2, 6).map(start => [[start, 0], [start + 2, 1]])];
    return [...linspace(-2, -.1, 6).map(start => {
      const hit = -start;
      return [[start, 0], [start + 2 * Math.min(1, hit), Math.min(1, hit)]];
    }), ...linspace(.1, 2, 6).map(start => [[start, 0], [start, Math.min(1, start)]])];
  }, [reversed, showExpansion]);
  return <Lab id="burgers" title="A balanced jump can still be the wrong solution"><p>For Burgers flux f(u)=u²/2, the characteristic speed is u. Reverse the initial states, then compare the entropy-admissible solution with a proposed expansion jump.</p>
    <div className="pde-controls"><Choice label="Burgers initial states" value={reversed ? 'ascending' : 'descending'} onChange={v => {
        setReversed(v === 'ascending');
        setShowExpansion(false);
      }} options={[['descending', 'Left 2, right 0'], ['ascending', 'Left 0, right 2']]} /><Range label="Burgers time" value={time} onChange={setTime} min={0} max={1} step={.02} /></div>
    {reversed && <label className="pde-toggle"><input type="checkbox" checked={showExpansion} onChange={e => setShowExpansion(e.target.checked)} />Inspect the inadmissible expansion jump instead</label>}
    <div className="pde-two"><SpaceTime title={reversed ? showExpansion ? 'Characteristics leave a gap' : 'A fan fills the gap' : 'Characteristics meet the shock'} xDomain={[-3, 4]} trajectories={trajectories}>{({
          x,
          y
        }) => <line x1={x(0)} x2={x(state.speed)} y1={y(0)} y2={y(1)} stroke={showExpansion ? '#e39584' : '#e8b34b'} strokeDasharray={reversed && !showExpansion ? '4 4' : undefined} strokeWidth="2" />}</SpaceTime><Plot title={time === 0 ? 'Initial Riemann data' : reversed && !showExpansion ? 'Entropy-admissible rarefaction' : showExpansion ? 'Inadmissible expansion jump' : 'Entropy-admissible shock'} lines={[series(state.profile, 'value')]} xDomain={[-3, 4]} yDomain={[-.2, 2.3]} /></div>
    <Values items={[["Candidate jump speed", format(state.speed)], ['Jump balance residual', '0'], ['Candidate jump entropy production', format(state.entropyProduction)], ['Displayed state', time === 0 ? 'Initial data' : showExpansion ? 'Inadmissible jump' : reversed ? 'Rarefaction fan' : 'Compressive shock']]} />
    <p>{reversed ? 'The +2/3 entropy production belongs to the candidate jump, even when the displayed solution is the admissible fan. A single jump balance does not select the correct nonlinear solution.' : 'The −2/3 entropy production agrees with the compressive orientation. Characteristics enter the shock; the jump is a weak balance, not a classical derivative.'} At t=0 only the initial data are shown. These conclusions concern this convex Burgers Riemann example.</p>
    <button onClick={() => {
      setReversed(false);
      setTime(.5);
      setShowExpansion(false);
    }}>Reset conservation law</button>
  </Lab>;
}
export function InverseHeatLab() {
  const [index, setIndex] = useState(6),
    [time, setTime] = useState(.03);
  const state = useMemo(() => inverseHeatState(index, time), [index, time]);
  return <Lab id="inverse" title="Forward smoothing hides an initial pattern"><p>Every initial mode has maximum magnitude 1. Increase its frequency while keeping the observation time positive. The same vertical scale exposes how little survives.</p>
    <div className="pde-controls"><Range label="Inverse heat mode n" value={index} onChange={setIndex} min={1} max={12} step={1} /><Range label="Inverse observation time" value={time} onChange={setTime} min={.002} max={.05} step={.002} /></div>
    <Plot title="Initial mode and later observation" lines={[series(state.profile, 'initial'), series(state.profile, 'final')]} xDomain={[0, 1]} yDomain={[-1.1, 1.1]} /><Legend labels={['Unit initial mode', 'Forward observation']} /><Values items={[["Forward maximum amplitude", format(state.attenuation)], ['Natural log amplitude', format(state.logAttenuation)], ['Required inverse gain', format(state.gain)], ['Natural log inverse gain', format(state.logGain)]]} />
    <p>The finite controls illustrate a separate sequence proof: as n→∞, later data approach zero while the initial maximum stays 1. An exact noiseless inverse is not continuously stable in this norm. This is a property of the inverse problem, not a failed timestep scheme.</p><button onClick={() => {
      setIndex(6);
      setTime(.03);
    }}>Reset inverse heat</button>
  </Lab>;
}
export function PeriodicDepthFigure() {
  const depths = linspace(0, 4, 128),
    phases = [0, Math.PI / 2, Math.PI];
  return <section className="pde-inline"><Plot title="A periodic boundary signal fades and lags with depth" lines={phases.map(phase => depths.map(z => [z, periodicDepth(z, phase).value]))} xDomain={[0, 4]} yDomain={[-1.1, 1.1]} xLabel="depth x / δ" yLabel="u / surface amplitude" /><Legend labels={['Surface phase 0', 'Surface phase π/2', 'Surface phase π']} /><p>At x=δ the amplitude is e⁻¹ of the surface amplitude and the phase lag is 1 radian. The profile is an analytic periodic steady regime on a homogeneous half-line; an arbitrary startup is a different initial-value problem.</p></section>;
}
export function ForcedRodLab() {
  const [time, setTime] = useState(0),
    [length, setLength] = useState(1),
    [amplitude, setAmplitude] = useState(.2),
    [tolerance, setTolerance] = useState(.05);
  const state = useMemo(() => forcedRod(time, amplitude, tolerance, length), [time, amplitude, tolerance, length]);
  return <Lab id="rod" title="Check a whole-rod settling requirement"><p>Capacity C=2, conductivity k=0.5 and source s=1 stay fixed. Endpoint temperature departure is zero. Changing length alters the steady profile as well as the diffusion timescale.</p>
    <div className="pde-controls"><Range label="Rod length L" value={length} onChange={setLength} min={.25} max={2} step={.25} display={`${length} m`} /><Range label="Initial excess amplitude" value={amplitude} onChange={setAmplitude} min={0} max={1} step={.05} display={`${amplitude} K`} /><Range label="Uniform temperature tolerance" value={tolerance} onChange={setTolerance} min={.01} max={.2} step={.01} display={`${tolerance} K`} /><Range label="Physical rod time" value={time} onChange={setTime} min={0} max={20} step={.02} display={`${format(time)} s`} /></div>
    <Plot title="Forced transient and steady profile" lines={[series(state.profile, 'value'), series(state.profile, 'steady')]} xDomain={[0, length]} xLabel="position x (m)" yLabel="temperature departure (K)" /><Legend labels={['Current profile', 'Steady profile']} />
    <Values items={[["Exact uniform excess", `${format(state.excess)} K`], ['Earliest required time', `${format(state.settlingTime)} s`], ['Requirement now', state.excess <= tolerance ? 'Within tolerance' : 'Not yet within tolerance'], ['Source per cross-sectional area', format(state.totalSource)], ['Total outward heat flux', format(2 * state.eachOutflow)], ['Stored-heat change rate', format(state.accumulation)]]} />
    <p>The uniform excess is known analytically from max|sin(πx/L)|=1, not estimated from plotted nodes. Source minus outward flux equals the stored-heat change rate. The chosen constants describe a teaching scenario, not a calibrated named material.</p>
    <button onClick={() => {
      setTime(0);
      setLength(1);
      setAmplitude(.2);
      setTolerance(.05);
    }}>Reset forced rod</button>
  </Lab>;
}
