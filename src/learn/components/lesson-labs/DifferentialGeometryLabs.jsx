import { useState } from 'react';
import { circleCharts, spherePatch, metricDifferential, sphereArc, sphereStep, sphereBands, polarConnection, transportTriangle, curvatureComparison, covariancePaths, scale, add, formatGeometry as fmt } from '../../data/differential-geometry-models.js';
import './differential-geometry-labs.css';
const AMBER = '#f3c16a',
  BLUE = '#99caff',
  GREEN = '#a0dfbb',
  PINK = '#ecb3d4';
const vectorText = vector => `(${vector.map(value => fmt(value, 4)).join(', ')})`;
const pointsText = points => points.map(point => point.join(',')).join(' ');
function Control({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  unit = ''
}) {
  return <label className="dg-control"><span>{label}: <strong>{fmt(value)}{unit}</strong></span>
    <input type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}
function Readouts({
  rows
}) {
  return <dl className="dg-readouts">{rows.map(([label, value]) => <div key={label}>
    <dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Arrow({
  from,
  to,
  color = AMBER,
  dashed = false
}) {
  const dx = to[0] - from[0],
    dy = to[1] - from[1],
    length = Math.hypot(dx, dy);
  if (length < 1e-7) return <circle cx={from[0]} cy={from[1]} r="3" fill={color} />;
  const ux = dx / length,
    uy = dy / length,
    size = Math.min(8, length * .45);
  const head = [to, [to[0] - size * ux + size * .45 * uy, to[1] - size * uy - size * .45 * ux], [to[0] - size * ux - size * .45 * uy, to[1] - size * uy + size * .45 * ux]];
  return <g><line x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]} stroke={color} strokeWidth="2.5" strokeDasharray={dashed ? '5 4' : undefined} />
    <polygon points={pointsText(head)} fill={color} /></g>;
}
function Dot({
  point,
  color = AMBER,
  label,
  dx = 8,
  dy = -9,
  outsideLabel = false
}) {
  return <g><circle cx={point[0]} cy={point[1]} r="4.5" fill={color} />
    {label && <text x={point[0] + dx} y={point[1] + dy} fill={color} textAnchor={outsideLabel ? 'middle' : undefined} style={outsideLabel ? { paintOrder: 'stroke', stroke: '#0c141d', strokeWidth: 4, strokeLinejoin: 'round' } : undefined}>{label}</text>}</g>;
}
function Plot({
  title,
  children,
  height = 290
}) {
  return <svg className="dg-plot" viewBox={`0 0 340 ${height}`} role="img" aria-label={title}>
    <title>{title}</title>{children}</svg>;
}
function planePoint(point, extent = 1.45) {
  return [170 + 105 * point[0] / extent, 145 - 105 * point[1] / extent];
}
function Axes({
  extent = 1.45,
  xLabel = 'x',
  yLabel = 'y'
}) {
  return <g className="dg-axes"><line x1="35" y1="145" x2="305" y2="145" />
    <line x1="170" y1="25" x2="170" y2="265" />
    <text x="311" y="151">{xLabel}</text><text x="178" y="27">{yLabel}</text>
    <line x1="275" y1="141" x2="275" y2="149" />
    <text x="260" y="170">{fmt(extent)}</text><text x="151" y="165">0</text></g>;
}
const circlePoints = (radius = 1, first = 0, last = Math.PI * 2) => Array.from({
  length: 97
}, (_, i) => {
  const angle = first + (last - first) * i / 96;
  return [radius * Math.cos(angle), radius * Math.sin(angle)];
});
function Circle({
  extent = 1.45
}) {
  return <polyline points={pointsText(circlePoints().map(point => planePoint(point, extent)))} stroke="#6b8296" strokeWidth="1.5" fill="none" />;
}
const project = point => [170 + 96 * (.8 * point[0] - .6 * point[1]), 150 - 96 * (.36 * point[0] + .48 * point[1] + .8 * point[2])];
function SphereWire() {
  const circles = [0, 1, 2].map(axis => Array.from({
    length: 97
  }, (_, i) => {
    const angle = Math.PI * 2 * i / 96;
    const point = [0, 0, 0];
    point[(axis + 1) % 3] = Math.cos(angle);
    point[(axis + 2) % 3] = Math.sin(angle);
    return project(point);
  }));
  return <g fill="none" stroke="#50677b" strokeWidth="1.2" strokeDasharray="3 4">
    <circle cx="170" cy="150" r="96" strokeDasharray="none" />
    {circles.map((points, i) => <polyline key={i} points={pointsText(points)} />)}</g>;
}
export function CircleAtlasLab() {
  const [angle, setAngle] = useState(225);
  const state = circleCharts(angle);
  return <section className="dg-lab" aria-label="Circle atlas investigation">
    <h3>A point survives the coordinate seam</h3>
    <p>Move through 180°, then through 0°/360°. Predict which coordinate label disappears. The circle point itself stays present.</p>
    <div className="dg-atlas-layout"><Plot title="One circle point and the two excluded chart points">
      <Axes /><Circle /><Dot point={planePoint(state.point)} label="P" />
        <Dot point={planePoint([-1, 0])} color={PINK} label="α seam" dx={-44} dy={45} />
        <Dot point={planePoint([1, 0])} color={BLUE} label="β seam" dx={-28} dy={45} />
      <Arrow from={planePoint(state.point)} to={planePoint(add(state.point, scale(state.tangent, .4)))} color={GREEN} />
    </Plot><div className="dg-chart-strips">{[['α chart', state.alpha, -Math.PI, Math.PI, PINK], ['β chart', state.beta, 0, 2 * Math.PI, BLUE]].map(([label, value, min, max, color]) => <div key={label}><h4>{label}</h4><div className="dg-coordinate-strip">
        <span className="dg-open-end left" /><span className="dg-open-end right" />
        {value !== null && <span className="dg-coordinate-marker" style={{
              left: `${100 * (value - min) / (max - min)}%`,
              background: color
            }} />}
      </div><div className="dg-strip-labels"><span>{min === 0 ? '0' : '−π'}</span><span>{max === Math.PI ? 'π' : '2π'}</span></div>
      <p>{value === null ? 'Not in this chart’s domain.' : `${fmt(value)} radians (${fmt(value * 180 / Math.PI)}°)`}</p></div>)}</div></div>
    <Control label="Circle position" min={0} max={360} value={angle} onChange={setAngle} unit="°" />
    <div className="dg-actions"><button onClick={() => setAngle(180)}>Inspect α seam</button><button onClick={() => setAngle(0)}>Inspect β seam</button><button onClick={() => setAngle(225)}>Reset atlas</button></div>
    <Readouts rows={[["Ambient point P", vectorText(state.point)], ['β − α', state.transition === null ? 'One chart unavailable' : `${fmt(state.transition)} radians`]]} />
    <p className="dg-caption">Open circles mark excluded endpoints of each coordinate interval. The green arrow is one tangent direction. At 225°, β=α+2π; on the other overlap component the transition is β=α.</p>
  </section>;
}
export function TangentPatchFigure() {
  const state = spherePatch();
  const point = state.point;
  return <figure className="dg-figure"><div className="dg-two-column">
    <Plot title="A sphere point with two tangent coordinate directions and a radial direction">
      <SphereWire /><Arrow from={project([0, 0, 0])} to={project(point)} color={PINK} dashed />
      <Arrow from={project(point)} to={project(add(point, scale(state.thetaBasis, .6)))} color={AMBER} />
      <Arrow from={project(point)} to={project(add(point, scale(state.phiBasis, .6)))} color={BLUE} />
      <Dot point={project(point)} label="P" color={GREEN} dy={-15} />
    </Plot><div><h4>Directions at one point</h4><p><span className="dg-key amber">θ direction</span> {vectorText(state.thetaBasis)}</p>
      <p><span className="dg-key blue">φ direction</span> {vectorText(state.phiBasis)}</p>
      <p><span className="dg-key pink">Radial position</span> {vectorText(point)}</p>
      <p>Both tangent vectors satisfy P·v=0. Their lengths differ: 1 versus √3/2.</p></div></div>
    <figcaption>Computed at θ=60°, φ=35° on the unit sphere. The picture is an orthographic projection; dashes distinguish reference circles and the radial direction. The affine arrows start at P for drawing, while tangent vectors are velocities, not new sphere points.</figcaption>
  </figure>;
}
export function MetricDifferentialLab() {
  const [shear, setShear] = useState(1),
    [cost, setCost] = useState(1),
    [direction, setDirection] = useState(45);
  const state = metricDifferential(shear, cost, direction);
  const views = [['Coordinate components u,v', state.coordinateEllipse, state.coordinateDirection, scale(state.gradient, 1 / state.gradientNorm), 3.6, 'u', 'v'], ['Physical components x,y', state.worldEllipse, state.worldDirection, scale(state.worldGradient, 1 / state.gradientNorm), 2.3, 'x', 'y']];
  return <section className="dg-lab" aria-label="Metric and differential investigation">
    <h3>Relabel the space, or change what movement costs</h3>
    <p>Keep f(x,y)=2x−y. First change only the shear. Then change the cost of y-motion. Which operation changes the physical steepest-ascent direction?</p>
    <div className="dg-two-column">{views.map(([title, ellipse, chosen, gradient, extent, xLabel, yLabel]) => <div key={title}><h4>{title}</h4>
      <Plot title={`${title}: metric unit ellipse and two unit directions`}><Axes extent={extent} xLabel={xLabel} yLabel={yLabel} />
        <polyline points={pointsText(ellipse.map(point => planePoint(point, extent)))} fill="none" stroke={GREEN} strokeWidth="2" />
        <Arrow from={planePoint([0, 0], extent)} to={planePoint(gradient, extent)} color={AMBER} />
        <Arrow from={planePoint([0, 0], extent)} to={planePoint(chosen, extent)} color={BLUE} />
      </Plot></div>)}</div>
    <p className="dg-legend"><span className="dg-key green">Metric length 1</span><span className="dg-key amber">Unit steepest direction</span><span className="dg-key blue">Your unit direction</span></p>
    <div className="dg-controls"><Control label="Coordinate shear" min={-1.5} max={1.5} step={.1} value={shear} onChange={setShear} />
      <Control label="Physical y-motion cost" min={.5} max={3} step={.1} value={cost} onChange={setCost} />
      <Control label="Chosen direction angle" min={0} max={360} value={direction} onChange={setDirection} unit="°" /></div>
    <Readouts rows={[["Metric G", state.metric.map(vectorText).join(' ; ')], ['Differential coefficients', vectorText(state.covector)], ['Gradient in u,v', vectorText(state.gradient)], ['Gradient in x,y', vectorText(state.worldGradient)], ['df on your unit direction', fmt(state.directionalDerivative)], ['Maximum df on a unit direction', fmt(state.gradientNorm)]]} />
    <button onClick={() => {
      setShear(1);
      setCost(1);
      setDirection(45);
    }}>Reset metric</button>
    <p className="dg-caption">Both arrows are normalized using the metric, not their drawn Euclidean lengths. Axes have different stated scales. The level sets of f are 2x−y=constant in the physical view; the differential remains the same functional while its components change under x=u+sv, y=v.</p>
  </section>;
}
export function SphereArcLab() {
  const [angle, setAngle] = useState(90),
    [fraction, setFraction] = useState(.5),
    [radius, setRadius] = useState(1);
  const state = sphereArc(angle, fraction, radius);
  const pos = point => planePoint(scale(point, 1 / radius));
  return <section className="dg-lab" aria-label="Sphere paths investigation"><h3>Three routes between the same endpoints</h3>
    <p>A great circle lies in a plane through the sphere’s center. This cross-section preserves its actual angles and relative lengths. Move along each route at the same fractional progress.</p>
    <Plot title="Short arc, long arc and off-sphere chord between shared endpoints"><Axes /><Circle />
      <polyline points={pointsText(circlePoints(1, 0, state.angle).map(point => planePoint(point)))} fill="none" stroke={AMBER} strokeWidth="4" />
      <polyline points={pointsText(circlePoints(1, 0, -(2 * Math.PI - state.angle)).map(point => planePoint(point)))} fill="none" stroke={BLUE} strokeWidth="2" strokeDasharray="6 4" />
      <line x1={pos(state.first)[0]} y1={pos(state.first)[1]} x2={pos(state.second)[0]} y2={pos(state.second)[1]} stroke={PINK} strokeWidth="2" />
      <Dot point={pos(state.first)} label="A" color={GREEN} /><Dot point={pos(state.second)} label="B" color={GREEN} dx={26 * state.second[0] / radius} dy={2 - 26 * state.second[1] / radius} outsideLabel />
      <Dot point={pos(state.shortPoint)} color={AMBER} /><Dot point={pos(state.longPoint)} color={BLUE} /><Dot point={pos(state.chordPoint)} color={PINK} />
    </Plot><p className="dg-legend"><span className="dg-key amber">Chosen short arc</span><span className="dg-key blue">Long arc</span><span className="dg-key pink">Chord</span></p>
    <div className="dg-controls"><Control label="Endpoint separation" min={0} max={180} value={angle} onChange={setAngle} unit="°" />
      <Control label="Route fraction" min={0} max={1} step={.01} value={fraction} onChange={setFraction} />
      <Control label="Sphere radius" min={.5} max={3} step={.5} value={radius} onChange={setRadius} /></div>
    <Readouts rows={[["Short / long lengths", `${fmt(state.shortLength)} / ${fmt(state.longLength)}`], ['Chord length', fmt(state.chordLength)], ['Shortest Log', state.logarithmStatus]]} />
    <p className="dg-identity">{angle === 180 ? 'At antipodes, this plane shows two equally short semicircles. On the sphere, infinitely many great-circle directions join them: no unique shortest Log.' : angle === 0 ? 'The shortest route has length zero. The blue full-circle geodesic returns to A but does not minimize distance.' : 'The long great-circle arc is locally straight, but the shorter arc determines the endpoint distance.'}</p>
    <button onClick={() => {
      setAngle(90);
      setFraction(.5);
      setRadius(1);
    }}>Reset paths</button>
    <p className="dg-caption">Lengths use the chosen radius units; the drawing rescales the whole sphere to fit. The path length is independent of the progress marker. Geometry, not a numerical integration or observed trajectory, supplies these values.</p>
  </section>;
}
export function SphereBandFigure() {
  const state = sphereBands();
  return <figure className="dg-figure"><h4>Equal angle does not mean equal area</h4>
    <div className="dg-bands">{state.bands.map((band, i) => <div key={band.lower}><strong>θ: {band.lower}°–{band.upper}°</strong>
      <div className="dg-area-strip"><i style={{
            width: `${band.surfaceFraction * 400}%`,
            background: [PINK, BLUE, AMBER][i]
          }} /></div>
      <span>{fmt(100 * band.surfaceFraction, 2)}% of the whole sphere</span></div>)}</div>
    <figcaption>The bars encode the exact areas of three 30° northern bands, with the largest bar representing 25% of the full sphere. Their sizes come from integrating sin θ. A uniformly chosen θ overweights the polar region relative to uniform surface area.</figcaption>
  </figure>;
}
export function SphereStepLab() {
  const [start, setStart] = useState(0),
    [rate, setRate] = useState(.1);
  const state = sphereStep(start, rate);
  const extent = Math.max(1.45, Math.hypot(...state.candidate) * 1.15, Math.hypot(...state.rawAmbientStep) * 1.15);
  const pos = point => planePoint(point, extent);
  return <section className="dg-lab" aria-label="Sphere update investigation"><h3>A tangent direction still needs a return map</h3>
    <p>Minimize f(x,y)=x+2y on the unit circle. Compare the two valid endpoints and their objective values; validity alone does not establish descent. The drawing rescales to include long candidates.</p>
    <Plot title="Tangent candidate, normalization and exact exponential endpoint" height={290}><Axes extent={extent} /><Circle extent={extent} />
      <Arrow from={pos(state.point)} to={pos(state.candidate)} color={BLUE} />
      <Arrow from={pos(state.point)} to={pos(state.rawAmbientStep)} color={PINK} dashed />
      <line x1={170} y1={145} x2={pos(state.candidate)[0]} y2={pos(state.candidate)[1]} stroke={GREEN} strokeDasharray="4 4" />
      <Dot point={pos(state.point)} color="#fff" label="start" dx={8} dy={-14} />
      <Dot point={pos(state.candidate)} color={BLUE} label="x+v" dx={8} dy={34} />
      <Dot point={pos(state.retracted)} color={GREEN} /><Dot point={pos(state.exponential)} color={AMBER} />
    </Plot>
    <div className="dg-step-endpoints"><div><h4 className="dg-key green">Normalize x+v</h4><strong>{vectorText(state.retracted.slice(0, 2))}</strong><span>Angle: {fmt(state.retractionAngle)} rad</span><span>f: {fmt(state.costs.retracted)}</span></div>
      <div><h4 className="dg-key amber">Follow Expₓ(v)</h4><strong>{vectorText(state.exponential.slice(0, 2))}</strong><span>Traveled angle: {fmt(state.tangentLength)} rad</span><span>f: {fmt(state.costs.exponential)}</span></div></div>
    <div className="dg-controls"><Control label="Starting direction" min={-180} max={180} value={start} onChange={setStart} unit="°" />
      <Control label="Update step size" min={0} max={1.5} step={.01} value={rate} onChange={setRate} /></div>
    <Readouts rows={[["Tangent gradient", vectorText(state.tangentGradient.slice(0, 2))], ['Candidate norm', fmt(state.candidateNorm)], ['Starting f', fmt(state.costs.original)]]} />
    <p className="dg-caption">Blue: x plus the negative tangent-gradient step. Dashed pink: the raw ambient-gradient step, which has not removed the radial part. Green: normalized candidate. Amber: exact geodesic step. These endpoints can overlap for a zero or very small step; the numbers distinguish them.</p>
    <button onClick={() => {
      setStart(0);
      setRate(.1);
    }}>Reset original update</button>
  </section>;
}
export function PolarConnectionLab() {
  const [time, setTime] = useState(1),
    [height, setHeight] = useState(1);
  const state = polarConnection(time, height),
    pos = point => planePoint(point, 3);
  const rows = [['Radial', state.radialAcceleration, state.radialCorrection], ['Angular', state.angularAcceleration, state.angularCorrection]];
  return <section className="dg-lab" aria-label="Moving polar basis investigation"><h3>A straight path seen through a moving basis</h3>
    <p>The point follows (t,b) at constant Cartesian speed. Move t and watch the radial and angular coordinate directions change. Their changes cancel the apparent coordinate acceleration.</p>
    <div className="dg-two-column"><Plot title="Straight Cartesian line with moving radial and angular basis"><Axes extent={3} />
      <line x1={pos([-2.5, height])[0]} y1={pos([-2.5, height])[1]} x2={pos([2.5, height])[0]} y2={pos([2.5, height])[1]} stroke={GREEN} strokeWidth="3" />
      <line x1="170" y1="145" x2={pos(state.point)[0]} y2={pos(state.point)[1]} stroke={PINK} strokeDasharray="4 4" />
      <Arrow from={pos(state.point)} to={pos(add(state.point, scale(state.radialBasis, .8)))} color={AMBER} />
      <Arrow from={pos(state.point)} to={pos(add(state.point, scale(state.angularBasis, .65)))} color={BLUE} />
      <Dot point={pos(state.point)} label="P" color={GREEN} dy={20} />
    </Plot><div className="dg-cancellations">{rows.map(([label, derivative, correction]) => <div key={label}><h4>{label} acceleration</h4>
      <span>Coordinate second derivative: <strong>{fmt(derivative)}</strong></span>
      <span>Moving-basis correction: <strong>{fmt(correction)}</strong></span>
      <span className="dg-identity">Sum: {fmt(derivative + correction)}</span></div>)}</div></div>
    <div className="dg-controls"><Control label="Position along straight path" min={-2} max={2} step={.1} value={time} onChange={setTime} />
      <Control label="Line height b" min={.25} max={2} step={.25} value={height} onChange={setHeight} /></div>
    <Readouts rows={[["Polar position (r,θ)", vectorText([state.r, state.theta])], ['Polar velocity (r′,θ′)', vectorText([state.radialVelocity, state.angularVelocity])], ['Reconstructed Cartesian velocity', vectorText(state.cartesianVelocity)]]} />
    <p className="dg-caption">Amber is ∂r; blue is ∂θ, whose physical length is r. Display arrows use fixed multipliers 0.8 and 0.65, respectively. The second-derivative and correction numbers are exact analytic expressions evaluated in binary64; tiny residuals are rounding.</p>
    <button onClick={() => {
      setTime(1);
      setHeight(1);
    }}>Reset moving basis</button>
  </section>;
}
export function SphereTransportLab() {
  const [wedge, setWedge] = useState(90),
    [initial, setInitial] = useState(0),
    [progress, setProgress] = useState(3);
  const [reverse, setReverse] = useState(false),
    [radius, setRadius] = useState(1);
  const state = transportTriangle(wedge, initial, progress, reverse, radius);
  const pos = point => project(scale(point, 1 / radius));
  return <section className="dg-lab" aria-label="Parallel transport investigation"><h3>Return to the same point with a different arrow</h3>
    <p>Trace N→A→B→N, then reverse the loop. Parallel means no intrinsic turning along each leg. The returned arrow compares two vectors in the same tangent plane at N.</p>
    <div className="dg-two-column"><div><h4>Specified spherical route</h4><Plot title="Spherical triangle and a parallel-transported tangent arrow"><SphereWire />
      {state.paths.map((path, i) => <polyline key={i} points={pointsText(path.map(pos))} fill="none" stroke={AMBER} strokeWidth="3" />)}
      {state.vertices.slice(0, 3).map((point, i) => <Dot key={i} point={pos(point)} label={state.labels[i]} color={GREEN} dy={i === 0 ? -14 : 22} />)}
      <Dot point={pos(state.point)} color={BLUE} />
      <Arrow from={pos(state.point)} to={pos(add(state.point, scale(state.currentVector, radius * .55)))} color={BLUE} />
    </Plot></div><div><h4>Final comparison at N</h4><Plot title="Initial and returned vectors in the north-pole tangent plane"><Axes /><Circle />
      <Arrow from={planePoint([0, 0])} to={planePoint(state.initial)} color={PINK} />
      <Arrow from={planePoint([0, 0])} to={planePoint(state.finalVector)} color={BLUE} />
    </Plot></div></div>
    <p className="dg-legend"><span className="dg-key amber">Route</span><span className="dg-key pink">Initial arrow</span><span className="dg-key blue">Transported arrow</span></p>
    <div className="dg-controls"><Control label="Longitude wedge" min={15} max={150} step={5} value={wedge} onChange={setWedge} unit="°" />
      <Control label="Initial tangent angle" min={-180} max={180} step={15} value={initial} onChange={setInitial} unit="°" />
      <Control label="Transport progress" min={0} max={3} step={.05} value={progress} onChange={setProgress} />
      <Control label="Transport sphere radius" min={.5} max={3} step={.5} value={radius} onChange={setRadius} /></div>
    <div className="dg-actions"><button aria-pressed={reverse} onClick={() => setReverse(!reverse)}>{reverse ? 'Route: N → B → A → N' : 'Route: N → A → B → N'}</button>
      <button onClick={() => {
        setWedge(90);
        setInitial(0);
        setProgress(3);
        setReverse(false);
        setRadius(1);
      }}>Reset transport</button></div>
    <Readouts rows={[["Current ambient arrow", vectorText(state.currentVector)], ['Arrow norm / radial dot', `${fmt(state.currentNorm)} / ${fmt(state.tangencyResidual)}`], ['Final oriented turn', `${fmt(state.finalTurn * 180 / Math.PI)}°`], ['Enclosed area', fmt(state.enclosedArea)], ['Gaussian curvature', fmt(state.curvature)]]} />
    <p className="dg-caption">The left sphere is an orthographic projection: its screen angles are not geometric measurements. The right compass shows the complete loop’s result even while the left progress marker is mid-route. Radius changes area and curvature reciprocally; this loop’s turn stays the same. All coordinates come from exact great-circle formulas, independently checked against the transport ODE.</p>
  </section>;
}
export function CurvatureComparisonLab() {
  const [radius, setRadius] = useState(1),
    [distance, setDistance] = useState(.75);
  const state = curvatureComparison(radius, distance),
    maximum = Math.max(...state.rows.map(row => row.circumference));
  const colors = [GREEN, GREEN, AMBER, BLUE];
  return <section className="dg-lab" aria-label="Intrinsic curvature investigation"><h3>Measure a circle from inside the surface</h3>
    <p>Give every geodesic disk the same radius s. Compare its circumference with 2πs. A rolled cylinder is locally flat; it agrees with the plane until a disk wraps around it.</p>
    <div className="dg-curvature-bars">{state.rows.map((row, i) => <div key={row.kind}><div><strong>{row.kind}</strong><span>K={fmt(row.curvature)}</span></div>
      <div className="dg-area-strip"><i style={{
            width: `${100 * row.circumference / maximum}%`,
            background: colors[i]
          }} /></div>
      <span>C={fmt(row.circumference)}; disk area={fmt(row.area)}</span></div>)}</div>
    <div className="dg-controls"><Control label="Curvature length scale R" min={.5} max={3} step={.5} value={radius} onChange={setRadius} />
      <Control label="Geodesic radius s over R" min={.05} max={1.5} step={.05} value={distance} onChange={setDistance} /></div>
    <p className="dg-identity">Common geodesic radius s={fmt(state.distance)}. Euclidean circumference 2πs={fmt(2 * Math.PI * state.distance)}.</p>
    <p className="dg-caption">Calculated constant-curvature geometry, not samples: C is 2πs for the plane/cylinder, 2πR sin(s/R) for the sphere and 2πR sinh(s/R) for the hyperbolic plane. The cylindrical radius is R and s&lt;πR; sphere disks also remain below the antipodal cut radius. Bars share a scale within the current state and rescale when controls change.</p>
    <button onClick={() => {
      setRadius(1);
      setDistance(.75);
    }}>Reset curvature</button>
  </section>;
}
export function CovariancePathsFigure() {
  const state = covariancePaths();
  return <figure className="dg-figure"><h4>Same covariances, different chosen midpoint</h4>
    <div className="dg-two-column">{[['Euclidean: 2.5I', state.arithmetic, AMBER], ['Affine-invariant: 2I', state.affine, BLUE]].map(([label, variances, color]) => <div key={label}><h4>{label}</h4><Plot title={`${label}: unit Mahalanobis covariance contour`}><Axes extent={2.5} />
        {[state.first, state.second].map((diagonal, i) => <polyline key={i} points={pointsText(circlePoints().map(([x, y]) => planePoint([x * Math.sqrt(diagonal[0]), y * Math.sqrt(diagonal[1])], 2.5)))} fill="none" stroke="#687e92" strokeDasharray="4 4" strokeWidth="1.5" />)}
        <polyline points={pointsText(circlePoints().map(([x, y]) => planePoint([x * Math.sqrt(variances[0]), y * Math.sqrt(variances[1])], 2.5)))} fill="none" stroke={color} strokeWidth="3" />
      </Plot></div>)}</div>
    <figcaption>Dashed input contours are diag(1,4) and diag(4,1). Axis lengths represent square roots of variances; these are unit Mahalanobis contours, not stated confidence regions. Arithmetic interpolation and an affine-invariant geodesic choose different midpoint matrices.</figcaption>
  </figure>;
}
