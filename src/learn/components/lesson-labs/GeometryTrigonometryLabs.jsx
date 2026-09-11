import { useState } from 'react';
import { arcState, similarityState, circleComponents, bearingState, frameState, dissectionState, ambiguousTriangleState, formatGeometry as f } from '../../data/geometry-trigonometry-models.js';
import './geometry-trigonometry-labs.css';
function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  suffix = ''
}) {
  return <label>{label}: {value}{suffix}<input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Plane({
  extent = 7,
  children,
  label
}) {
  const scale = 130 / extent;
  const x = value => 165 + value * scale;
  const y = value => 160 - value * scale;
  return <svg viewBox="0 0 330 325" role="img" aria-label={label}>
    {Array.from({
      length: extent * 2 + 1
    }, (_, index) => index - extent).map(tick => <g key={tick}><line className="grid" x1={x(tick)} y1={30} x2={x(tick)} y2={290} /><line className="grid" x1={35} y1={y(tick)} x2={295} y2={y(tick)} /></g>)}
    <path className="axis" d="M25 160H306m-7-4 7 4-7 4M165 300V18m-4 7 4-7 4 7" />
    <text x="306" y="185">x</text><text x="175" y="25">y</text>
    {children({
      x,
      y,
      scale
    })}
  </svg>;
}
export function DistanceFigure() {
  return <figure className="geometry-figure"><Plane extent={7} label="P at one two and Q at four six; three horizontal and four vertical units form a right triangle">{({
        x,
        y
      }) => <>
    <path className="blue dashed" d={`M${x(1)} ${y(2)}H${x(4)}V${y(6)}`} />
    <line className="gold" x1={x(1)} y1={y(2)} x2={x(4)} y2={y(6)} />
    <path className="axis" d={`M${x(4) - 9} ${y(2)}v-9h9`} />
    <circle className="point" cx={x(1)} cy={y(2)} r="5" /><circle className="point" cx={x(4)} cy={y(6)} r="5" />
    <text x={x(1) - 60} y={y(2) + 25}>P(1, 2)</text><text x={x(4) - 18} y={y(6) - 13}>Q(4, 6)</text>
    <text x={x(2.5)} y={y(2) + 23}>3</text><text x={x(4) + 10} y={y(4)}>4</text>
  </>}</Plane><figcaption>Both axes use metres with the same physical scale. The dashed path travels 3 + 4 = 7 m. The direct segment has length 5 m. A coordinate pair locates a point; subtracting two pairs describes a displacement.</figcaption></figure>;
}
export function PythagorasDissectionFigure() {
  const state = dissectionState();
  const position = ([x, y]) => `${40 + 35 * x},${40 + 35 * y}`;
  const corners = [[0, 0], [7, 0], [7, 7], [0, 7]];
  return <figure className="geometry-figure"><svg viewBox="0 0 330 335" role="img" aria-label="A seven by seven square tiled by four right triangles of area six and a central square of area twenty-five">
    {corners.map((corner, index) => <polygon key={index} className="triangle-fill" points={[corner, state.central[index], state.central[(index + 3) % 4]].map(position).join(' ')} />)}
    <polygon className="centre-fill" points={state.central.map(position).join(' ')} />
    <text x="162" y="163" textAnchor="middle">c² = 25</text>
    <text x="91" y="28">a = 3</text><text x="214" y="28">b = 4</text>
    <text x="57" y="103">6</text><text x="235" y="93">6</text><text x="220" y="246">6</text><text x="70" y="255">6</text>
    <text x="165" y="318" textAnchor="middle">49 = 4 × 6 + 25</text>
  </svg><figcaption>The numbers inside the pieces are areas. Each blue-square edge is the hypotenuse of a 3–4–5 triangle. The four corners of that central shape are right angles: the two acute angles of a right triangle add to 90°. Its area is therefore c².</figcaption></figure>;
}
export function AngleArcLab() {
  const [radius, setRadius] = useState(2);
  const [degrees, setDegrees] = useState(60);
  const state = arcState(radius, degrees);
  const r = radius * 39;
  const ex = 160 + r * state.cosine;
  const ey = 160 - r * state.sine;
  const arc = `M${160 + r} 160A${r} ${r} 0 ${degrees > 180 ? 1 : 0} 0 ${ex} ${ey}`;
  return <section className="geometry-lab" aria-label="Angle and arc investigation"><h3>Keep the angle; change the circle</h3><p>Predict: if the radius doubles, does the angle double? Does the sector area double? The drawing keeps a fixed scale while you change the radius.</p>
    <div className="geometry-controls"><Slider label="Radius" value={radius} min={1} max={3} step={0.5} onChange={setRadius} /><Slider label="Sweep" value={degrees} min={15} max={330} step={15} onChange={setDegrees} suffix="°" /></div>
    <svg viewBox="0 0 330 330" role="img" aria-label={`${degrees} degree counterclockwise sector on radius ${radius}`}>
      <circle cx="160" cy="160" r={r} className="axis" />
      <path d={`M160 160L${160 + r} 160A${r} ${r} 0 ${degrees > 180 ? 1 : 0} 0 ${ex} ${ey}Z`} fill="#bd8b3730" />
      <path className="gold" d={arc} /><path className="blue" d={`M${160 + r} 160H160L${ex} ${ey}`} />
      <circle className="point" cx={ex} cy={ey} r="5" /><text x="165" y="181">O</text>
      <text x="165" y="315" textAnchor="middle">Positive sweep: counterclockwise</text>
    </svg>
    <div className="readout" aria-live="polite">Angle = {degrees}° = {f(state.radians)} rad<br />Arc s = rθ = {f(state.arc)} length units<br />Sector area = r²θ/2 = {f(state.area)} square units</div>
    <p>The arc grows with r, but s/r stays {f(state.radians)}. The sector takes the same fraction of a larger disk. These controls show a single positive sweep below one full turn; they do not count repeated windings.</p>
    <button onClick={() => {
      setRadius(2);
      setDegrees(60);
    }}>Reset arc</button>
  </section>;
}
export function TriangleSimilarityLab() {
  const [shape, setShape] = useState('3-4-5');
  const [scale, setScale] = useState(2);
  const state = similarityState(shape, scale);
  const drawScale = 230 / (Math.max(state.adjacent, state.opposite) * Math.max(1, scale));
  const triangle = factor => `45,275 ${45 + state.adjacent * drawScale * factor},275 ${45 + state.adjacent * drawScale * factor},${275 - state.opposite * drawScale * factor}`;
  return <section className="geometry-lab" aria-label="Similar triangles investigation"><h3>Change the size, preserve the shape</h3><p>Predict which changes: side lengths, opposite/hypotenuse, or area. The marked angle sits at the shared left corner. Both triangles use one scale inside this drawing; the view fits their combined size.</p>
    <div className="geometry-controls"><label>Triangle shape<select aria-label="Triangle shape" value={shape} onChange={event => setShape(event.target.value)}><option value="3-4-5">3–4–5</option><option value="5-12-13">5–12–13</option><option value="equal-legs">Equal legs</option></select></label><Slider label="Positive scale" value={scale} min={0.5} max={3} step={0.25} onChange={setScale} /></div>
    <svg viewBox="0 0 330 325" role="img" aria-label={`Original and ${scale} times scaled ${shape} similar triangles`}>
      <polygon points={triangle(scale)} className="gold" /><polygon points={triangle(1)} className="blue dashed" />
      <path className="axis" d={`M76 275A31 31 0 0 0 ${45 + 31 * Math.cos(state.angle)} ${275 - 31 * Math.sin(state.angle)}`} />
      <text x="43" y="304">θ</text><text x="158" y="304">adjacent</text><text x="210" y="37">opposite ↑</text>
    </svg>
    <div className="geometry-legend"><span className="blue-key">Dashed blue: original</span><span className="gold-key">Gold: scaled</span></div>
    <div className="readout" aria-live="polite">Original (adjacent, opposite, hypotenuse): ({f(state.adjacent)}, {f(state.opposite)}, {f(state.hypotenuse)})<br />Scaled: ({f(state.adjacent * scale)}, {f(state.opposite * scale)}, {f(state.hypotenuse * scale)})<br />sin θ = {f(state.sine)} · cos θ = {f(state.cosine)} · tan θ = {f(state.tangent)}<br />Area: {f(state.originalArea)} → {f(state.scaledArea)} square units</div>
    <p>Every side acquires the same factor, so it cancels from a ratio. Area has two length factors and acquires scale². Switching shape changes the angle and the ratios; changing positive scale does not.</p><button onClick={() => {
      setShape('3-4-5');
      setScale(2);
    }}>Reset similarity</button>
  </section>;
}
function ComponentTrace({
  degrees,
  kind
}) {
  const fn = kind === 'cosine' ? Math.cos : Math.sin;
  const path = Array.from({
    length: 145
  }, (_, index) => {
    const degree = -360 + 5 * index;
    return `${index ? 'L' : 'M'}${30 + (degree + 360) * 270 / 720} ${73 - 43 * fn(degree * Math.PI / 180)}`;
  }).join(' ');
  return <svg viewBox="0 0 330 150" role="img" aria-label={`${kind} over minus one to plus one turn with chosen angle marked`}>
    <path className="axis" d="M30 73H300M165 22V123" /><path className={kind === 'cosine' ? 'gold' : 'blue'} d={path} />
    <line className="axis dashed" x1={30 + (degrees + 360) * 270 / 720} x2={30 + (degrees + 360) * 270 / 720} y1="23" y2="121" />
    <circle className="point" cx={30 + (degrees + 360) * 270 / 720} cy={73 - 43 * fn(degrees * Math.PI / 180)} r="4" />
    <text x="8" y="33" className="small">1</text><text x="1" y="122" className="small">−1</text><text x="30" y="144" className="small">−360°</text><text x="166" y="144" className="small">0°</text><text x="258" y="144" className="small">360°</text><text x="188" y="22">{kind}</text>
  </svg>;
}
export function CircleComponentsLab() {
  const [degrees, setDegrees] = useState(30);
  const state = circleComponents(degrees);
  const px = 165 + 108 * state.cosine;
  const py = 155 - 108 * state.sine;
  return <section className="geometry-lab" aria-label="Circle components investigation"><h3>Read two shadows of one rotating radius</h3><p>Before crossing 90°, predict which component changes sign. Then compare −90° with 270°: the endpoint agrees while the chosen signed turn differs.</p>
    <div className="geometry-controls"><Slider label="Signed angle" value={degrees} min={-360} max={360} step={15} onChange={setDegrees} suffix="°" /></div>
    <svg viewBox="0 0 330 320" role="img" aria-label={`Unit-circle endpoint at cosine ${f(state.cosine)}, sine ${f(state.sine)}`}>
      <circle cx="165" cy="155" r="108" className="axis" /><path className="axis" d="M30 155H305M165 290V20" />
      <path className="gold" d={`M165 155H${px}`} /><path className="blue" d={`M${px} 155V${py}`} />
      <line className="axis" x1="165" y1="155" x2={px} y2={py} /><circle className="point" cx={px} cy={py} r="6" />
      <text x="285" y="181">1</text><text x="27" y="181">−1</text><text x="175" y="40">1</text><text x="175" y="285">−1</text>
      <text x="165" y="313" textAnchor="middle">Radius = 1</text>
    </svg>
    <div className="geometry-legend"><span className="gold-key">Gold: horizontal cosine</span><span className="blue-key">Blue: vertical sine</span></div>
    <ComponentTrace degrees={degrees} kind="cosine" /><ComponentTrace degrees={degrees} kind="sine" />
    <div className="readout" aria-live="polite">{degrees}° = {f(state.radians)} rad<br />(cos θ, sin θ) = ({f(state.cosine)}, {f(state.sine)})<br />tan θ = {state.tangent === null ? 'undefined: cosine is zero' : f(state.tangent)}</div>
    <p>Coordinates are signed; the radius length is positive. Full turns return to the same endpoint. Cardinal-angle zeros here are exact geometric identities; other numerical readouts are rounded.</p><button onClick={() => setDegrees(30)}>Reset circle</button>
  </section>;
}
export function BearingLab() {
  const [draft, setDraft] = useState(['-3', '4']);
  const [state, setState] = useState(() => bearingState(-3, 4));
  const [error, setError] = useState('');
  function apply(values = draft) {
    try {
      if (values.some(value => value.trim() === '')) throw new RangeError('Enter both integer coordinates.');
      const next = bearingState(...values.map(Number));
      setState(next);
      setDraft(values);
      setError('');
    } catch (problem) {
      setError(problem.message);
    }
  }
  return <section className="geometry-lab" aria-label="Bearing investigation"><h3>Recover a direction without losing its quadrant</h3><p>Predict the failure of atan(y/x) for (−3, 4). Both signs are needed. The direction convention here is greater than −180° and at most 180°.</p>
    <div className="geometry-controls">{['x', 'y'].map((name, index) => <label key={name}>{name}<input aria-label={`Bearing ${name}`} value={draft[index]} onChange={event => setDraft(draft.map((value, i) => i === index ? event.target.value : value))} inputMode="numeric" /></label>)}<button onClick={() => apply()}>Apply coordinates</button></div>
    <div className="geometry-buttons">{[[3, 4], [-3, 4], [-3, -4], [3, -4], [-4, 0], [0, 4], [0, 0]].map(pair => <button key={pair.join()} onClick={() => apply(pair.map(String))}>({pair.join(', ')})</button>)}</div>
    {error && <p role="alert">{error} The last valid point is retained.</p>}
    <Plane label={`Point ${state.x}, ${state.y}; ${state.quadrant}`}>{({
        x,
        y
      }) => <>
      <path className="blue dashed" d={`M165 160H${x(state.x)}V${y(state.y)}`} /><line className="gold" x1="165" y1="160" x2={x(state.x)} y2={y(state.y)} />
      <circle className="point" cx={x(state.x)} cy={y(state.y)} r="6" />
      <text x={x(state.x) + (state.x > 3 ? -8 : 10)} y={y(state.y) - 13} textAnchor={state.x > 3 ? 'end' : 'start'}>{state.radius === 0 ? 'P = O' : 'P'}</text>
    </>}</Plane>
    <div className="readout" aria-live="polite">Active point: ({state.x}, {state.y}) · {state.quadrant}<br />Radius = {f(state.radius)}<br />Bearing = {state.angle === null ? 'undefined: the origin has no direction' : `${f(state.degrees)}° = ${f(state.angle)} rad`}<br />atan(y/x) alone = {state.naive === null ? 'undefined: division by zero' : `${f(state.naive * 180 / Math.PI)}°`}</div>
    <p>Input contract: integer coordinates from −6 to 6. Try an axis, then the origin. A software convention at (0, 0) cannot recover a direction that the geometry does not contain.</p>
  </section>;
}
export function AmbiguousTriangleFigure() {
  const state = ambiguousTriangleState();
  const scale = 18;
  const point = ([x, y]) => `${35 + x * scale},${175 - y * scale}`;
  return <figure className="geometry-figure"><svg viewBox="0 0 330 245" role="img" aria-label="Two triangles share A thirty degrees, side a seven and side b ten; two base endpoints lie on the same ray">
    <path className="axis" d="M25 175H309" />
    <path className="axis dashed" d={`M${point([state.candidates[0].c, 0])}A126 126 0 0 0 ${point([state.candidates[1].c, 0])}`} />
    {state.candidates.map((item, index) => <polygon key={index} className={index ? 'gold' : 'blue dashed'} points={[[0, 0], [item.c, 0], state.C].map(point).join(' ')} />)}
    <text x="20" y="198">A</text><text x={35 + state.C[0] * scale + 5} y="68">C</text>
    {state.candidates.map((item, index) => <text key={index} x={35 + item.c * scale} y="234" textAnchor="middle">B{index + 1}</text>)}
    <text x="73" y="105">b = 10</text><text x="238" y="117">a = 7</text><text x="69" y="167" className="small">30°</text>
  </svg><figcaption>A stays at (0, 0), C at (5√3, 5). A radius-7 circle about C meets the positive horizontal ray twice: the base lengths are {f(state.candidates[0].c)} and {f(state.candidates[1].c)}. Both triangles satisfy the supplied data. Their B angles are {f(state.candidates[0].B * 180 / Math.PI)}° and {f(state.candidates[1].B * 180 / Math.PI)}°.</figcaption></figure>;
}
export function CoordinateFrameLab() {
  const defaults = {
    px: 4,
    py: 2,
    ox: 1,
    oy: -1,
    degrees: 30,
    mode: 'passive'
  };
  const [inputs, setInputs] = useState(defaults);
  const state = frameState(inputs.px, inputs.py, inputs.ox, inputs.oy, inputs.degrees, inputs.mode);
  function update(name, value) {
    setInputs(current => ({
      ...current,
      [name]: value
    }));
  }
  return <section className="geometry-lab" aria-label="Coordinate frame investigation"><h3>Move the description—or move the point</h3><p>In passive mode, turn the frame while watching world point P. In active mode, the same positive angle turns the point about O. Predict what a 90° turn does before moving the angle slider.</p>
    <div className="geometry-controls"><label>Interpretation<select aria-label="Interpretation" value={inputs.mode} onChange={event => update('mode', event.target.value)}><option value="passive">Passive: change coordinates</option><option value="active">Active: rotate point</option></select></label><Slider label="Rotation angle" value={inputs.degrees} min={-180} max={180} step={15} suffix="°" onChange={value => update('degrees', value)} /></div>
    <div className="geometry-controls">{[['px', 'Point x', -4, 4], ['py', 'Point y', -4, 4], ['ox', 'Origin x', -2, 2], ['oy', 'Origin y', -2, 2]].map(([name, label, low, high]) => <Slider key={name} label={label} value={inputs[name]} min={low} max={high} onChange={value => update(name, value)} />)}</div>
    <Plane extent={10} label={`${state.mode} rotation; world point ${state.point.join(',')}, origin ${state.origin.join(',')}`}>
      {({
        x,
        y
      }) => {
        const [ox, oy] = state.origin;
        const endX = [ox + 5 * state.cosine, oy + 5 * state.sine];
        const endY = [ox - 5 * state.sine, oy + 5 * state.cosine];
        const localFoot = [ox + state.local[0] * state.cosine, oy + state.local[0] * state.sine];
        return <>
          {state.mode === 'passive' && <><path className="blue" d={`M${x(endX[0])} ${y(endX[1])}L${x(ox)} ${y(oy)}L${x(endY[0])} ${y(endY[1])}`} /><text x={x(endX[0]) + 5} y={y(endX[1]) - 6}>x′</text><text x={x(endY[0]) + 5} y={y(endY[1]) - 6}>y′</text><path className="blue dashed" d={`M${x(ox)} ${y(oy)}L${x(localFoot[0])} ${y(localFoot[1])}L${x(state.point[0])} ${y(state.point[1])}`} /></>}
          <line className="gold" x1={x(ox)} y1={y(oy)} x2={x(state.point[0])} y2={y(state.point[1])} />
          <circle className="point" cx={x(state.point[0])} cy={y(state.point[1])} r="5" />
          {state.radius !== 0 && <text x={x(state.point[0]) + 9} y={y(state.point[1]) - 10}>{state.mode === 'active' && state.degrees === 0 ? 'P = P′' : 'P'}</text>}
          {state.mode === 'active' && <><line className="blue" x1={x(ox)} y1={y(oy)} x2={x(state.rotated[0])} y2={y(state.rotated[1])} /><circle cx={x(state.rotated[0])} cy={y(state.rotated[1])} r="5" fill="#8db9df" />{state.radius !== 0 && state.degrees !== 0 && <text x={x(state.rotated[0]) + (state.rotated[0] > 7 ? -10 : 10)} y={y(state.rotated[1]) + (state.rotated[1] > 7 ? 23 : -12)} textAnchor={state.rotated[0] > 7 ? 'end' : 'start'}>P′</text>}</>}
          <circle cx={x(ox)} cy={y(oy)} r="4" fill="#e9a8a5" /><text x={x(ox) - 19} y={y(oy) + 23}>{state.radius === 0 ? state.mode === 'active' ? 'P = P′ = O' : 'P = O' : 'O'}</text>
        </>;
      }}
    </Plane>
    <div className="readout" aria-live="polite">World P = ({state.point.map(value => f(value)).join(', ')}) · O = ({state.origin.join(', ')})<br />{state.mode === 'passive' ? 'Local coordinates q' : 'Rotated world point P′'} = ({state.result.map(value => f(value)).join(', ')})<br />Local q reconstructed in world = ({state.reconstructed.map(value => f(value)).join(', ')})<br />Distance from O = {f(state.radius)} before and after rotation</div>
    <p>{state.mode === 'passive' ? 'Gold P stays physically fixed when only the frame angle changes. Dashed blue legs are its local components. Blue axes show five-unit positive directions, not unit-length basis arrows.' : 'Gold P is the starting point; blue P′ is the actively rotated point. The original world axes stay fixed. The local-coordinate reconstruction readout remains a separate check of the same input P.'} One grid spacing is one length unit. Numerical components are rounded.</p><button onClick={() => setInputs(defaults)}>Reset frame</button>
  </section>;
}
export function ScreenCoordinateFigure() {
  return <figure className="geometry-figure"><svg viewBox="0 0 330 245" role="img" aria-label="World y increases up while screen v increases down; three metres horizontal map to sixty pixels and four metres vertical map to minus forty pixels">
    <path className="axis" d="M35 175H300M65 217V30" /><path className="gold" d="M65 175H245V75" />
    <circle className="point" cx="245" cy="75" r="6" /><text x="265" y="69">Q</text><text x="42" y="194">P</text>
    <text x="96" y="199">3 m → 60 px</text><text x="88" y="40">World y ↑ · screen v ↓</text><text x="76" y="105">4 m → −40 px</text>
  </svg><figcaption>Annotated mapping, not a pixel ruler: horizontal and vertical screen scales can differ. Convert each pixel displacement back with its own scale before measuring physical distance.</figcaption></figure>;
}
export function TwoLinkFigure() {
  const elbow = [2 * Math.cos(Math.PI / 6), 1];
  const end = [elbow[0], 2];
  const x = value => 60 + 95 * value;
  const y = value => 260 - 95 * value;
  return <figure className="geometry-figure"><svg viewBox="0 0 330 310" role="img" aria-label="A length-two first link at thirty degrees and length-one second link at absolute ninety degrees; the relative elbow angle is sixty degrees">
    <path className="axis" d="M35 260H295M60 280V30" />
    <path className="gold" d={`M60 260L${x(elbow[0])} ${y(elbow[1])}`} /><path className="blue" d={`M${x(elbow[0])} ${y(elbow[1])}L${x(end[0])} ${y(end[1])}`} />
    <path className="axis dashed" d={`M${x(elbow[0])} ${y(elbow[1])}l52-30`} />
    {[elbow, end, [0, 0]].map((point, index) => <circle key={index} className="point" cx={x(point[0])} cy={y(point[1])} r="5" />)}
    <text x="123" y="185">L₁ = 2</text><text x="239" y="111">L₂ = 1</text><text x="97" y="252">30°</text><text x="232" y="155">60°</text><text x="115" y="46">End: (√3, 2)</text>
  </svg><figcaption>The dashed continuation shows the first link's direction at the elbow. The second link turns another 60° from that direction; its world angle is 30° + 60° = 90°. Link lengths use the same units.</figcaption></figure>;
}
