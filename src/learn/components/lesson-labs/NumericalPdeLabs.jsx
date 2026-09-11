import { useId, useMemo, useState } from 'react';
import { advectionEvolution, burgersGodunovFlux, diffusionEvolution, finiteElement1D, formatPdeNumber, materialInterface, neumannBalance, poissonProblem, poissonRefinement, rectangleStencil, triangleElement, twoGridCorrection } from '../../data/numerical-pde-models.js';
import './numerical-pde-labs.css';
const gold = '#e8b44a';
const blue = '#78b9ee';
const green = '#83d3aa';
const pink = '#ed9eba';
const number = value => formatPdeNumber(value, 4);
const axisTick = value => value !== 0 && (Math.abs(value) < .01 || Math.abs(value) >= 10000) ? value.toExponential(1).replace("e+", "e") : formatPdeNumber(value, 2);
function Range({
  label,
  value,
  change,
  min,
  max,
  step = 1
}) {
  const id = useId();
  return <label className="npde-control" htmlFor={id}><span>{label}: <output>{number(value)}</output></span><input id={id} type="range" value={value} min={min} max={max} step={step} onChange={event => change(Number(event.target.value))} /></label>;
}
function Choice({
  label,
  value,
  change,
  options
}) {
  const id = useId();
  return <label className="npde-control"><span id={id}>{label}</span><select aria-labelledby={id} value={value} onChange={event => change(event.target.value)}>{options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}</select></label>;
}
function Investigation({
  title,
  prompt,
  reset,
  children
}) {
  return <section className="npde-investigation" aria-label={title}><header><h3>{title}</h3><button onClick={reset}>Reset</button></header><p><strong>Predict first.</strong> {prompt}</p>{children}</section>;
}
function Legend({
  series
}) {
  return <ul className="npde-legend">{series.map(item => <li key={item.label}><i style={{
        background: item.color
      }} />{item.label}</li>)}</ul>;
}
function Plot({
  series,
  label,
  xLabel = 'Position x',
  yLabel = 'Field value',
  markers = [],
  zero = true
}) {
  const points = series.flatMap(item => item.points);
  const xMin = Math.min(...points.map(point => point[0]));
  const xMax = Math.max(...points.map(point => point[0]));
  const low = Math.min(...points.map(point => point[1]), ...(zero ? [0] : []));
  const high = Math.max(...points.map(point => point[1]), ...(zero ? [0] : []));
  const padding = Math.max((high - low) * 0.08, high === low ? 0.1 : 0);
  const yMin = low - padding;
  const yMax = high + padding;
  const x = value => 60 + 238 * (value - xMin) / (xMax - xMin || 1);
  const y = value => 219 - 174 * (value - yMin) / (yMax - yMin);
  return <figure className="npde-figure"><figcaption>{label}</figcaption><svg viewBox="0 0 320 260" role="img" aria-label={label}>
    {[0, 0.5, 1].map(fraction => <g key={fraction}><line x1="60" x2="298" y1={y(yMin + fraction * (yMax - yMin))} y2={y(yMin + fraction * (yMax - yMin))} className="npde-grid" /><text x="54" y={y(yMin + fraction * (yMax - yMin)) + 4} textAnchor="end">{axisTick(yMin + fraction * (yMax - yMin))}</text><text x={60 + 238 * fraction} y="239" textAnchor="middle">{axisTick(xMin + fraction * (xMax - xMin))}</text></g>)}
    <text x="60" y="23">{yLabel}</text>
    {series.map(item => <polyline key={item.label} fill="none" stroke={item.color} strokeWidth="2.2" strokeDasharray={item.dashed ? '5 4' : undefined} points={item.points.map(([horizontal, vertical]) => `${x(horizontal)},${y(vertical)}`).join(' ')} />)}
    {markers.map((point, index) => <circle key={index} cx={x(point[0])} cy={y(point[1])} r="3" fill={gold} />)}
  </svg><p className="npde-axis">{xLabel}</p><Legend series={series} /></figure>;
}
export function RestrictionReconstructionFigure() {
  const result = useMemo(() => finiteElement1D({
    nodes: [0, 0.25, 0.5, 0.75, 1]
  }), []);
  return <div className="npde-inline" aria-label="A field, its samples and its reconstructed line"><p className="npde-flow"><span>Continuous field u(x)</span><b>→ sample Rₕ →</b><span>Five nodal values</span><b>→ connect Iₕ →</b><span>A new piecewise-linear field</span></p><Plot label="Same values at the dots; different values between them" series={[{
      label: 'Exact x(1−x)',
      color: blue,
      points: result.curve.map(row => [row.x, row.exact])
    }, {
      label: 'Linear reconstruction',
      color: gold,
      points: result.nodes.map((x, index) => [x, result.values[index]])
    }]} markers={result.nodes.map((x, index) => [x, result.values[index]])} /><p>Here the nodal error is zero in exact arithmetic. The reconstructed field still misses the parabola by as much as 1/64. The smooth curve is sampled for drawing; this error value comes from the polynomial calculation.</p></div>;
}
export function PoissonBudgetLab() {
  const [intervals, setIntervals] = useState(8);
  const [method, setMethod] = useState('direct');
  const [iterations, setIterations] = useState(0);
  const [profile, setProfile] = useState('quartic');
  const [boundary, setBoundary] = useState('zero');
  const [selected, setSelected] = useState(1);
  const result = useMemo(() => poissonProblem({
    intervals,
    method,
    iterations,
    profile,
    left: boundary === 'tilted' ? 2 : 0,
    right: boundary === 'tilted' ? -1 : 0
  }), [intervals, method, iterations, profile, boundary]);
  const row = Math.min(selected, intervals - 1);
  const certificate = result.certificate;
  return <Investigation title="From a stencil row to a field certificate" prompt="Can a small scaled residual conceal a visible field error? Try Jacobi before refining the grid." reset={() => {
    setIntervals(8);
    setMethod('direct');
    setIterations(0);
    setProfile('quartic');
    setBoundary('zero');
    setSelected(1);
  }}>
    <div className="npde-controls"><Choice label="Intervals N" value={intervals} change={value => {
        setIntervals(Number(value));
        setSelected(1);
      }} options={[4, 8, 16, 32, 64].map(value => [value, value])} /><Choice label="Manufactured target" value={profile} change={setProfile} options={[['quartic', 'Quartic: truncation visible'], ['quadratic', 'Quadratic: exact at nodes'], ['linear', 'Linear: no mesh error']]} /><Choice label="Solve" value={method} change={setMethod} options={[['direct', 'Positive-pivot direct solve'], ['jacobi', 'Bounded Jacobi iteration']]} /><Choice label="Endpoint values" value={boundary} change={setBoundary} options={[['zero', '0 and 0'], ['tilted', '2 and −1']]} /></div>
    {method === 'jacobi' && <Range label="Jacobi updates" value={iterations} change={setIterations} min={0} max={4000} step={25} />}
    <Range label="Interior row j" value={row} change={setSelected} min={1} max={intervals - 1} />
    <div className="npde-stencil"><span>−U<sub>{row - 1}</sub><b>{number(-result.values[row - 1])}</b></span><span>+2U<sub>{row}</sub><b>{number(2 * result.values[row])}</b></span><span>−U<sub>{row + 1}</sub><b>{number(-result.values[row + 1])}</b></span></div>
    <p>Divide that row sum by h²={number(result.h ** 2)} to compare with f(xⱼ)={number(result.source[row - 1])}. Known endpoint terms move to the right-hand side when assembling interior unknowns.</p>
    <Plot label="Calculated nodal solution and its straight-line reconstruction" series={[{
      label: 'Manufactured continuous field',
      color: blue,
      points: result.curve.map(point => [point.x, point.exact])
    }, {
      label: 'Computed reconstruction',
      color: gold,
      points: result.nodes.map((x, index) => [x, result.values[index]])
    }]} markers={[[result.nodes[row], result.values[row]]]} />
    <div className="npde-budget"><p><strong>Algebraic contribution</strong><output>{number(certificate.algebraicBound)}</output></p><p><strong>Nodal discretization bound</strong><output>{number(certificate.nodalDiscretizationBound)}</output></p><p><strong>Between-node bound</strong><output>{number(certificate.interpolationBound)}</output></p><p className="npde-total"><strong>Total field bound</strong><output>{number(certificate.fieldBound)}</output></p></div>
    <p className="npde-result" aria-live="polite">{certificate.certified ? 'Certified within' : 'Not yet certified within'} the requested .001 field tolerance. A residual-based upper bound can be conservative; failure to certify does not prove the actual error exceeds the tolerance.</p>
    <p>Physical residual upper bound: {number(certificate.residualUpper)}. Scaled residual diagnostic: {number(Math.max(...result.residuals.map(Math.abs)) * result.h ** 2)}. Weighted nodal L2 error: {number(result.weightedNodalL2)}; sampled field discrepancy: {number(result.sampledFieldError)}.</p>
    <p className="lesson-note">The certificate uses exact rational arithmetic on the stored floating-point vector and these declared polynomial data, followed by upward rounding of bounds. Displayed decimals are rounded. The curve and sampled discrepancy use 257 positions and are diagnostics, not the certificate. This certifies the declared mathematical field, not the accuracy of a physical model.</p>
  </Investigation>;
}
export function RefinementLab() {
  const [profile, setProfile] = useState('quartic');
  const rows = useMemo(() => poissonRefinement(profile), [profile]);
  return <Investigation title="A refinement test can tell the wrong story" prompt="Will an exact-on-grid quadratic show order two in its nodal error?" reset={() => setProfile('quartic')}><Choice label="Refinement target" value={profile} change={setProfile} options={[['quartic', 'Quartic'], ['quadratic', 'Quadratic'], ['linear', 'Linear']]} />
    {profile === 'quartic' && <Plot label="Actual computed nodal errors on logarithmic coordinates" xLabel="log₂ N" yLabel="log₂ nodal error" zero={false} series={[{
      label: 'Direct-solve error',
      color: gold,
      points: rows.map(row => [Math.log2(row.intervals), Math.log2(row.error)])
    }]} />}
    <div className="npde-table" tabIndex={0} role="region" aria-label="Refinement values"><table><thead><tr><th>N</th><th>Computed nodal error</th><th>Observed order</th><th>Exact mesh error</th></tr></thead><tbody>{rows.map(row => <tr key={row.intervals}><th>{row.intervals}</th><td>{number(row.error)}</td><td>{row.observedOrder === null ? 'undefined' : number(row.observedOrder)}</td><td>{number(row.exactNodalDiscretizationError)}</td></tr>)}</tbody></table></div><p>For the quadratic and linear targets, exact nodal discretization error is zero. Small computed discrepancies are arithmetic effects, so this investigation does not assign them a convergence order or put zero on a logarithmic axis.</p></Investigation>;
}
export function DiffusionGridLab() {
  const [timeSteps, setTimeSteps] = useState(8);
  const [mode, setMode] = useState(1);
  const [frameIndex, setFrameIndex] = useState(0);
  const [intervals, setIntervals] = useState(8);
  const result = useMemo(() => diffusionEvolution({
    intervals,
    timeSteps,
    mode,
    finalTime: 0.1
  }), [intervals, timeSteps, mode]);
  const index = Math.min(frameIndex, result.frames.length - 1);
  const frame = result.frames[index];
  const series = [['continuous', 'Continuous heat field', blue], ['semidiscrete', 'Exact spatial ODE', green], ['explicit', 'Explicit Euler', pink], ['backward', 'Backward Euler', gold], ['crank', 'Crank–Nicolson', '#bbb3f0']].map(([key, label, color]) => ({
    label,
    color,
    dashed: key === 'continuous',
    points: result.nodes.map((x, j) => [x, frame[key][j]])
  }));
  return <Investigation title="Watch a mode, not just a stability badge" prompt="Choose the near-grid-scale mode and two time steps. Which method alternates sign, and which strongly damps it?" reset={() => {
    setTimeSteps(8);
    setMode(1);
    setFrameIndex(0);
    setIntervals(8);
  }}>
    <div className="npde-controls"><Choice label="Diffusion intervals" value={intervals} change={value => {
        setIntervals(Number(value));
        setMode(1);
        setFrameIndex(0);
      }} options={[4, 8, 16].map(value => [value, value])} /><Choice label="Steps to time .1" value={timeSteps} change={value => {
        setTimeSteps(Number(value));
        setFrameIndex(0);
      }} options={[2, 4, 8, 16, 32, 64].map(value => [value, value])} /><Choice label="Initial sine mode" value={mode} change={value => {
        setMode(Number(value));
        setFrameIndex(0);
      }} options={[[1, 'k=1: smooth'], [intervals - 1, `k=${intervals - 1}: near grid scale`]]} /></div>
    <Range label="Displayed time step" value={index} change={setFrameIndex} min={0} max={result.frames.length - 1} />
    <Plot label={`All five fields at the same physical time t=${number(frame.time)}`} series={series} />
    <div className="npde-table" tabIndex={0} role="region" aria-label="Mode multipliers"><table><thead><tr><th>Update</th><th>One-step factor</th></tr></thead><tbody>{Object.entries(result.factors).map(([name, value]) => <tr key={name}><th>{name}</th><td>{number(value)}</td></tr>)}</tbody></table></div>
    <p className="npde-result" aria-live="polite">r={number(result.ratio)}. Explicit monotonicity condition r≤.5: {result.explicitMonotone ? 'met' : 'not met'}. This finite grid's spectral threshold is {number(result.finiteGridThreshold)}. State: {result.status}.</p>
    <p className="lesson-note">Actual stencil updates and factored tridiagonal solves generate the three numerical fields. Continuous and spatial-ODE sine solutions are calculated references; lines join grid samples. Vertical axes rescale to include overshoots. Changing N while selecting k=N−1 changes the initial field, so it is not a fixed-data convergence experiment.</p>
  </Investigation>;
}
export function InterfaceFluxLab() {
  const [conductivity, setConductivity] = useState(10);
  const [position, setPosition] = useState(0.5);
  const result = materialInterface({
    rightConductivity: conductivity,
    interfacePosition: position
  });
  return <Investigation title="Two materials share one steady flux" prompt="Increase the right conductivity. Does a more conductive half develop a steeper or flatter temperature slope?" reset={() => {
    setConductivity(10);
    setPosition(0.5);
  }}><Range label="Right conductivity k₂" value={conductivity} change={setConductivity} min={0.5} max={20} step={0.5} /><Range label="Interface position" value={position} change={setPosition} min={0.1} max={0.9} step={0.05} />
    <div className="npde-resistance"><div style={{
        flex: position
      }}>k₁=1<br />R₁={number(result.resistances[0])}</div><b>→ q →</b><div style={{
        flex: 1 - position
      }}>k₂={number(conductivity)}<br />R₂={number(result.resistances[1])}</div></div>
    <Plot label="Exact piecewise-affine temperature for the declared interface" yLabel="Temperature" series={[{
      label: 'Continuous temperature, common flux',
      color: gold,
      points: result.curve.map(point => [point.x, point.y])
    }]} markers={[[position, result.interfaceTemperature]]} />
    <p className="npde-result" aria-live="polite">q=1/(R₁+R₂)={number(result.flux)}; interface temperature {number(result.interfaceTemperature)}. A length-weighted arithmetic coefficient would predict {number(result.arithmeticFlux)}, generally a different flux.</p><p className="lesson-note">Normalized rod length 1, endpoint temperatures 1 and 0, no internal source. If length is metres and k is W/(m·K), R has units m²K/W and q is W/m². The diagram uses declared inputs, not measured materials.</p></Investigation>;
}
export function FluxCompatibilityLab() {
  const [right, setRight] = useState(1);
  const result = neumannBalance({
    rightOutward: right
  });
  return <Investigation title="A boundary gauge cannot repair missing heat" prompt="The unit rod produces 2 units of heat and loses 1 at its left end. What must leave at its right end?" reset={() => setRight(1)}><Range label="Right outward flux" value={right} change={setRight} min={0} max={2} step={0.25} /><div className="npde-flow"><span>← left outflow 1</span><span>Four cells: +.5 each</span><span>right outflow {number(right)} →</span></div>
    <p className="npde-result" aria-live="polite">{result.compatible ? `Compatible. The chosen mean-zero representative is [${result.values.map(number).join(', ')}].` : `Incompatible: source minus outward flux = ${number(result.mismatch)}. No steady solution is returned.`}</p>
    {result.compatible && <p>Right-oriented face fluxes: [{result.fluxes.map(number).join(', ')}]. Neighboring cells use the same face with opposite signs. Adding any constant to every temperature changes none of these fluxes.</p>}<p>We check the full balance before fixing a zero mean. Pinning a temperature first and dropping an incompatible row would silently solve a different boundary problem.</p></Investigation>;
}
export function TransportCellsLab() {
  const [courant, setCourant] = useState(0.75);
  const [velocity, setVelocity] = useState(1);
  const [scheme, setScheme] = useState('upwind');
  const [profile, setProfile] = useState('pulse');
  const [index, setIndex] = useState(0);
  const result = useMemo(() => advectionEvolution({
    courant,
    velocity,
    scheme,
    profile,
    steps: 16
  }), [courant, velocity, scheme, profile]);
  const frame = result.frames[index];
  const x = frame.values.map((_, j) => (j + 0.5) / 16);
  return <Investigation title="Move cell averages through shared faces" prompt="Reverse the velocity. Which neighbor is now upstream? Try c=1, then centered differences." reset={() => {
    setCourant(0.75);
    setVelocity(1);
    setScheme('upwind');
    setProfile('pulse');
    setIndex(0);
  }}>
    <div className="npde-controls"><Choice label="Transport scheme" value={scheme} change={value => {
        setScheme(value);
        setIndex(0);
      }} options={[['upwind', 'Upwind flux'], ['centered', 'Centered flux + Euler']]} /><Choice label="Velocity" value={velocity} change={value => {
        setVelocity(Number(value));
        setIndex(0);
      }} options={[[1, '+1: right'], [-1, '−1: left']]} /><Choice label="Initial cell averages" value={profile} change={value => {
        setProfile(value);
        setIndex(0);
      }} options={[['pulse', 'Discontinuous pulse'], ['sine', 'Smooth sine']]} /></div>
    <Range label="Courant number c" value={courant} change={value => {
      setCourant(value);
      setIndex(0);
    }} min={0} max={1.5} step={0.25} /><Range label="Transport step" value={index} change={setIndex} min={0} max={16} />
    <div className="npde-cell-strip" aria-label="Sixteen periodic cell averages">{frame.values.map((value, j) => <div key={j} title={`Cell ${j}: ${number(value)}`} style={{
        background: value < 0 ? '#74334a' : `rgba(232,180,74,${Math.max(0.08, Math.min(0.9, value * 0.75))})`
      }}><span>{j}</span><b>{number(value)}</b></div>)}</div>
    <Plot label={`Cell averages at t=${number(frame.time)}; periodic domain [0,1)`} series={[{
      label: 'Computed averages',
      color: gold,
      points: x.map((position, j) => [position, frame.values[j]])
    }, {
      label: 'Exactly translated averages',
      color: blue,
      points: x.map((position, j) => [position, frame.exact[j]])
    }]} />
    <p className="npde-result" aria-live="polite">Mass hΣU={number(frame.mass)}; minimum {number(frame.minimum)}, maximum {number(frame.maximum)}. Upwind convex-weight condition: {result.monotone ? 'met' : 'not met'}.</p><p className="lesson-note">Colors are clipped for legibility; numerical labels and the plot retain actual overshoots. Lines join cell-center values and do not represent a continuous pulse profile. Time changes with c at fixed step; the reference always uses that same time. Mass preservation alone does not establish stability.</p></Investigation>;
}
export function GodunovFaceLab() {
  const [left, setLeft] = useState(-1);
  const [right, setRight] = useState(2);
  const result = burgersGodunovFlux(left, right);
  return <Investigation title="Choose a Burgers face flux from the wave" prompt="A rarefaction spreads from −1 to 2. Which value reaches the fixed face x/t=0?" reset={() => {
    setLeft(-1);
    setRight(2);
  }}><Range label="Left state" value={left} change={setLeft} min={-3} max={3} step={0.5} /><Range label="Right state" value={right} change={setRight} min={-3} max={3} step={0.5} /><div className="npde-riemann"><span>uL={number(left)}</span><b>{result.kind === 'rarefaction' ? '↖ fan ↗' : result.kind === 'shock' ? '→ shock ←' : 'constant'}</b><span>uR={number(right)}</span></div><p className="npde-result" aria-live="polite">{result.kind}; face flux u²/2 = {number(result.flux)}. {result.faceState === null ? 'Stationary shock: either trace gives the same flux; no unique face value is claimed.' : `Face state ${number(result.faceState)}.`} {result.speed !== undefined && `Shock speed ${number(result.speed)}.`}</p><p>This is an exact scalar Riemann-face calculation, not an animation or a complete nonlinear PDE time solver.</p></Investigation>;
}
const meshes = {
  uniform: [0, 0.25, 0.5, 0.75, 1],
  aligned: [0, 0.25, 1 / 3, 0.5, 0.75, 1],
  nonuniform: [0, 0.2, 0.6, 1]
};
export function HatAssemblyLab() {
  const [mesh, setMesh] = useState('uniform');
  const [sourceKind, setSourceKind] = useState('constant');
  const [selected, setSelected] = useState(1);
  const result = useMemo(() => finiteElement1D({
    nodes: meshes[mesh],
    sourceKind,
    source: sourceKind === 'point' ? 1 : 2
  }), [mesh, sourceKind]);
  const elementIndex = Math.min(selected, result.elements.length - 1);
  const element = result.elements[elementIndex];
  const [a, b] = [result.nodes[elementIndex], result.nodes[elementIndex + 1]];
  return <Investigation title="Assemble overlapping hats; resolve a point source" prompt="Switch to the point load at 1/3. Will inserting a node there improve nodal values, the field between nodes, or both?" reset={() => {
    setMesh('uniform');
    setSourceKind('constant');
    setSelected(1);
  }}>
    <div className="npde-controls"><Choice label="Finite-element mesh" value={mesh} change={value => {
        setMesh(value);
        setSelected(1);
      }} options={[['uniform', 'Uniform quarters'], ['aligned', 'Add a node at 1/3'], ['nonuniform', 'Nonuniform: .2, .6']]} /><Choice label="Load" value={sourceKind} change={setSourceKind} options={[['constant', 'Constant f=2'], ['point', 'Unit point load at 1/3']]} /></div>
    <Range label="Selected element" value={elementIndex} change={setSelected} min={0} max={result.elements.length - 1} />
    <Plot label={`Local shape functions on [${number(a)}, ${number(b)}]`} yLabel="Shape-function value" series={[{
      label: `Left hat → global node ${elementIndex}`,
      color: blue,
      points: [[a, 1], [b, 0]]
    }, {
      label: `Right hat → global node ${elementIndex + 1}`,
      color: gold,
      points: [[a, 0], [b, 1]]
    }]} />
    <p>Element width {number(element.width)}. Its stiffness contribution is {number(1 / element.width)} × [[1,−1],[−1,1]], added to global rows and columns {elementIndex}, {elementIndex + 1}.</p>
    <div className="npde-table" tabIndex={0} role="region" aria-label="Assembled stiffness and load"><table><caption>Full matrix before Dirichlet elimination</caption><thead><tr><th>Node</th>{result.nodes.map((_, j) => <th key={j}>{j}</th>)}<th>Load</th></tr></thead><tbody>{result.stiffness.map((row, j) => <tr key={j}><th>{j}</th>{row.map((value, k) => <td key={k} className={[elementIndex, elementIndex + 1].includes(j) && [elementIndex, elementIndex + 1].includes(k) ? 'npde-highlight' : ''}>{number(value)}</td>)}<td>{number(result.load[j])}</td></tr>)}</tbody></table></div>
    <Plot label="Calculated finite-element function and its exact reference" series={[{
      label: 'Exact quadratic or Green field',
      color: blue,
      points: result.curve.map(point => [point.x, point.exact])
    }, {
      label: 'Finite-element field',
      color: gold,
      points: result.nodes.map((x, j) => [x, result.values[j]])
    }]} markers={result.nodes.map((x, j) => [x, result.values[j]])} />
    <p className="npde-result" aria-live="polite">Exact discretization errors: field max {number(result.fieldError)}; continuous L2 {number(result.l2Error)}; energy {number(result.energyError)}. Computed nodal discrepancy {number(result.nodalError)}.</p><p className="lesson-note">The three error formulas integrate these specific constant-load or point-load solutions; they exclude floating-point algebraic error. Plot samples are not used to claim these exact errors. The source is fixed at 1/3; “aligned” specifically means aligned to that source.</p></Investigation>;
}
export function RectangleIndexLab() {
  const [i, setI] = useState(2);
  const [j, setJ] = useState(2);
  const result = rectangleStencil({
    selectedX: i,
    selectedY: j,
    xLength: 2
  });
  return <Investigation title="A two-dimensional neighbor is not always a vector neighbor" prompt="Move to the rightmost interior node. Should its right neighbor wrap into the next row of unknowns?" reset={() => {
    setI(2);
    setJ(2);
  }}><div className="npde-controls"><Range label="Horizontal index i" value={i} change={setI} min={1} max={4} /><Range label="Vertical index j" value={j} change={setJ} min={1} max={3} /></div>
    <div className="npde-coordinate-grid" role="img" aria-label="Six by five grid; interior numbers are flattened unknown indices">{Array.from({
        length: 5
      }, (_, row) => 4 - row).flatMap(y => Array.from({
        length: 6
      }, (_, x) => {
        const selected = x === i && y === j;
        const neighbor = result.neighbors.find(point => point.i === x && point.j === y);
        return <span key={`${x},${y}`} className={selected ? 'npde-selected' : neighbor ? 'npde-neighbor' : ''}><small>({x},{y})</small><b>{x === 0 || y === 0 || x === 5 || y === 4 ? 'B' : (y - 1) * 4 + x - 1}</b></span>;
      }))}</div>
    <p>Physical rectangle 2×1. hₓ=.4, hᵧ=.25. Selected unknown index {result.selectedIndex}. B denotes a prescribed boundary value, not an unknown.</p><ul>{result.neighbors.map(point => <li key={`${point.i},${point.j}`}>({point.i},{point.j}): coefficient {number(point.coefficient)}; {point.boundary ? 'known boundary → right-hand side' : `vector index ${point.index}`}.</li>)}</ul><p className="npde-result">The 12×12 interior matrix has {result.nonzeros} nonzeros. The map and coefficients are calculated, not a sampled illustration.</p></Investigation>;
}
export function TriangleElementLab() {
  const [preset, setPreset] = useState('reference');
  const vertices = preset === 'reference' ? [[0, 0], [1, 0], [0, 1]] : [[1, 2], [3, 2], [1, 5]];
  const result = triangleElement(vertices);
  const minX = Math.min(...vertices.map(point => point[0]));
  const minY = Math.min(...vertices.map(point => point[1]));
  const extent = Math.max(...vertices.map(point => point[0] - minX), ...vertices.map(point => point[1] - minY));
  const positions = vertices.map(([x, y]) => [65 + 165 * (x - minX) / extent, 210 - 165 * (y - minY) / extent]);
  return <Investigation title="Map a triangle, then map its gradients" prompt="Double one side and triple the other. Do gradients grow with those lengths or shrink?" reset={() => setPreset('reference')}><Choice label="Triangle geometry" value={preset} change={setPreset} options={[['reference', 'Width 1, height 1'], ['scaled', 'Width 2, height 3']]} /><svg className="npde-triangle" viewBox="0 0 300 265" role="img" aria-label="Triangle with actual vertex coordinates"><polygon points={positions.map(point => point.join(",")).join(" ")} fill="rgba(232,180,74,.08)" stroke={gold} strokeWidth="2" />{positions.map(([x, y], index) => <g key={index}><circle cx={x} cy={y} r="4" fill={blue} /><text x={index === 0 ? 20 : index === 1 ? 280 : x + 7} y={index === 2 ? y - 12 : y + 23} textAnchor={index === 1 ? 'end' : 'start'}>{`v${index}=(${vertices[index].join(',')})`}</text></g>)}</svg><p>Area {number(result.area)}. Coordinates determine the actual shape. Horizontal and vertical scales are equal within each view; the view rescales between presets to keep the triangle readable.</p><div className="npde-table" tabIndex={0} role="region" aria-label="Triangle gradients and stiffness"><table><thead><tr><th>Basis</th><th>Gradient</th><th>Stiffness row</th></tr></thead><tbody>{result.gradients.map((gradient, index) => <tr key={index}><th>φ{index}</th><td>[{gradient.map(number).join(', ')}]</td><td>[{result.stiffness[index].map(number).join(', ')}]</td></tr>)}</tbody></table></div><p>The gradients sum to zero, so a constant coefficient vector has zero element energy. Global Dirichlet constraints remove the constant mode only when they constrain the connected problem appropriately.</p></Investigation>;
}
export function CoarseCorrectionLab() {
  const [mode, setMode] = useState(1);
  const result = twoGridCorrection(mode);
  return <Investigation title="Remove error on two resolutions" prompt="Weighted Jacobi weakly changes the smooth mode. Can a coarse piecewise-linear correction remove what remains?" reset={() => setMode(1)}><Choice label="Fine-grid error mode" value={mode} change={value => setMode(Number(value))} options={[[1, 'Smooth sine: k=1'], [7, 'Near-grid-scale sine: k=7']]} /><Plot label="One smoothing step and one exact coarse correction, N=8" yLabel="Error" series={[{
      label: 'Initial error',
      color: blue,
      points: result.initial.map((value, j) => [j / 8, value])
    }, {
      label: 'After weighted Jacobi',
      color: pink,
      points: result.smooth.map((value, j) => [j / 8, value])
    }, {
      label: 'After coarse correction',
      color: gold,
      points: result.final.map((value, j) => [j / 8, value])
    }]} /><div className="npde-table" tabIndex={0} role="region" aria-label="Two-grid error norms"><table><thead><tr><th>Stage</th><th>Weighted L2</th><th>Energy norm</th></tr></thead><tbody>{['initial', 'smooth', 'final'].map(stage => <tr key={stage}><th>{stage}</th><td>{number(result.norms[stage])}</td><td>{number(result.energyNorms[stage])}</td></tr>)}</tbody></table></div><p>Projection decreases the energy norm. It need not decrease every other norm at that same step: the high-mode example can gain a small smooth component and increase L2 after coarse correction. This is a declared two-grid calculation, not a claim about a full multigrid cycle's complexity.</p></Investigation>;
}
