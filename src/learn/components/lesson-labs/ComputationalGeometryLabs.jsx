import { useId, useMemo, useState } from 'react';
import { hullPresets, hullTrace, orientation, parseGridPoints, polygonPresets, polygonState, precisionState, segmentPresets, segmentState } from '../../data/computational-geometry-models.js';
import './computational-geometry-labs.css';
const coordinates = value => `(${value.join(', ')})`;
const screenX = value => 30 + 32 * value;
const screenY = value => 286 - 32 * value;
const path = (points, closed = false) => points.map((value, index) => `${index ? 'L' : 'M'}${screenX(value[0])},${screenY(value[1])}`).join(' ') + (closed && points.length >= 3 ? ' Z' : '');
function GeometryPlane({
  title,
  description,
  points = [],
  children
}) {
  const id = useId();
  const grouped = new Map();
  for (const [label, value] of points) {
    const key = value.join(',');
    if (grouped.has(key)) grouped.get(key).labels.push(label);else grouped.set(key, {
      value,
      labels: [label]
    });
  }
  return <svg className="geometry-plane" viewBox="0 0 316 320" role="img" aria-labelledby={`${id}-title ${id}-description`}>
    <title id={`${id}-title`}>{title}</title>
    <desc id={`${id}-description`}>{description} Cartesian coordinates, y upward, equal units on both axes from zero through eight. {points.map(([label, value]) => `${label}=${coordinates(value)}`).join('; ')}.</desc>
    {Array.from({
      length: 9
    }, (_, value) => <g key={value}>
      <line className="geometry-grid" x1={screenX(value)} x2={screenX(value)} y1="30" y2="286" />
      <line className="geometry-grid" x1="30" x2="286" y1={screenY(value)} y2={screenY(value)} />
      {value % 2 === 0 && <><text x={screenX(value)} y="304" textAnchor="middle">{value}</text><text x="19" y={screenY(value) + 4} textAnchor="end">{value}</text></>}
    </g>)}
    {children}
    {[...grouped.values()].map(({
      value,
      labels
    }) => <g key={value.join(',')}>
      <circle cx={screenX(value[0])} cy={screenY(value[1])} r="4" className="geometry-point" />
      <text x={screenX(value[0])} y={screenY(value[1]) - 9} textAnchor="middle" className="geometry-point-label">{labels.join('/')}</text>
    </g>)}
    <text x="297" y="306">x</text><text x="13" y="17">y</text>
  </svg>;
}
function GridCoordinate({
  label,
  value,
  onChange
}) {
  const id = useId();
  return <div className="geometry-controls">
    {[0, 1].map(axis => <label key={axis} htmlFor={`${id}-${axis}`}>{label} {axis === 0 ? 'x' : 'y'}: {value[axis]}
      <input id={`${id}-${axis}`} aria-label={`${label} ${axis === 0 ? 'x' : 'y'}`} type="range" min="0" max="8" step="1" value={value[axis]} onChange={event => onChange(value.map((coordinate, index) => index === axis ? Number(event.target.value) : coordinate))} />
    </label>)}
  </div>;
}
function Facts({
  values
}) {
  return <dl className="geometry-facts">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
export function OrientationLab() {
  const [query, setQuery] = useState([4, 5]);
  const [reversed, setReversed] = useState(false);
  const a = reversed ? [7, 3] : [1, 1];
  const b = reversed ? [1, 1] : [7, 3];
  const determinant = orientation(a, b, query);
  const first = (b[0] - a[0]) * (query[1] - a[1]);
  const second = (b[1] - a[1]) * (query[0] - a[0]);
  return <section className="geometry-lab" aria-label="Orientation and signed area">
    <h3>Which side of the directed baseline?</h3>
    <p>Predict the sign before moving C. Find a zero without making A and B coincide, then reverse the baseline.</p>
    <GridCoordinate label="Point C" value={query} onChange={setQuery} />
    <figure><GeometryPlane title="Directed triangle ABC" description={`A to B is the solid gold baseline; the filled triangle has area ${Math.abs(determinant) / 2}. C is ${determinant > 0 ? 'left' : determinant < 0 ? 'right' : 'on the line'} of A to B.`} points={[['A', a], ['B', b], ['C', query]]}>
      <path d={path([a, b, query], true)} className="geometry-fill" />
      <path d={path([a, b])} className="geometry-gold" />
      <path d={path([a, query])} className="geometry-blue" />
    </GeometryPlane><figcaption>Gold connects A to B; blue connects A to C. Read the labels for direction. Shaded area is geometric area, so it stays nonnegative even when the determinant changes sign.</figcaption></figure>
    <Facts values={[[`Product 1: (${b[0] - a[0]}) × (${query[1] - a[1]})`, first], [`Product 2: (${b[1] - a[1]}) × (${query[0] - a[0]})`, second], ['D = product 1 − product 2', determinant], ['Turn / unsigned area', `${determinant > 0 ? 'left' : determinant < 0 ? 'right' : 'collinear'} / ${Math.abs(determinant) / 2}`]]} />
    <div className="geometry-actions"><button onClick={() => setReversed(value => !value)}>Reverse A and B</button><button onClick={() => {
        setQuery([4, 5]);
        setReversed(false);
      }}>Reset orientation</button></div>
    <p>Coordinates are exact bounded integers. Set C=(4,2) to place it between the original A and B. A zero determinant also occurs if C equals either endpoint; it does not imply three distinct points.</p>
  </section>;
}
export function SegmentIntersectionLab() {
  const [segments, setSegments] = useState(segmentPresets.crossing);
  const [selected, setSelected] = useState(3);
  const state = segmentState(...segments);
  const id = useId();
  return <section className="geometry-lab" aria-label="Closed segment intersection">
    <h3>Distinguish crossing, contact and overlap</h3>
    <p>Choose a case, predict its classification, then move one endpoint. Every endpoint belongs to its segment.</p>
    <div className="geometry-actions">{Object.entries(segmentPresets).map(([name, points]) => <button key={name} onClick={() => {
        setSegments(points);
        if (name === 'crossing') setSelected(3);
      }}>{name === 'crossing' ? 'Reset crossing' : `Try ${name}`}</button>)}</div>
    <label htmlFor={id}>Endpoint to move<select id={id} value={selected} onChange={event => setSelected(Number(event.target.value))}>{['A', 'B', 'C', 'D'].map((name, index) => <option key={name} value={index}>{name}</option>)}</select></label>
    <GridCoordinate label="Selected endpoint" value={segments[selected]} onChange={value => setSegments(points => points.map((old, index) => index === selected ? value : old))} />
    <figure><GeometryPlane title="Two closed line segments" description={`AB is solid gold, CD dashed blue. Result: ${state.kind}. Shared endpoints: ${state.sharedEndpoints.map(coordinates).join('; ') || 'none'}.`} points={segments.map((value, index) => ['ABCD'[index], value])}>
      <path d={path(segments.slice(0, 2))} className="geometry-gold" />
      <path d={path(segments.slice(2))} className="geometry-blue geometry-dashed" />
      {state.sharedEndpoints.length >= 2 && <path d={path([state.sharedEndpoints[0], state.sharedEndpoints.at(-1)])} className="geometry-overlap" />}
    </GeometryPlane><figcaption>One location may carry several labels, such as A/B for a point segment. The thick pale line marks an overlapping segment, not a unique intersection point.</figcaption></figure>
    <Facts values={[["Classification", state.kind], ['Closed bounding boxes overlap?', state.boxesOverlap ? 'yes; still needs the geometric test' : 'no; disjoint'], ['D(A,B,C), D(A,B,D)', state.determinants.slice(0, 2).join(', ')], ['D(C,D,A), D(C,D,B)', state.determinants.slice(2).join(', ')]]} />
    <p>A strict sign change on both sides gives a proper crossing. Zeros need the between-endpoints test. Try extending two collinear segments so their intervals just separate: collinearity alone cannot detect contact.</p>
  </section>;
}
export function PredicatePrecisionLab() {
  const [exponent, setExponent] = useState(27);
  const [scenario, setScenario] = useState('products');
  const state = precisionState(exponent, scenario);
  const id = useId();
  return <section className="geometry-lab geometry-precision" aria-label="Exact and floating predicate comparison">
    <h3>A one-bit decision can change the shape</h3>
    <p>Predict whether exact input coordinates are enough to guarantee an exact determinant. Compare where the two arithmetic paths diverge.</p>
    <label htmlFor={id}>Failure stage<select id={id} value={scenario} onChange={event => setScenario(event.target.value)}><option value="products">Product cancellation</option><option value="input">A coordinate lost on input</option></select></label>
    {scenario === 'products' && <label>N = 2 to the power {exponent}<input aria-label="Precision exponent" type="range" min="20" max="30" step="1" value={exponent} onChange={event => setExponent(Number(event.target.value))} /></label>}
    <div className="geometry-arithmetic">{state.points.map((value, index) => <div key={index}><strong>{'ABC'[index]} intended integer</strong><span>{coordinates(value)}</span><small>Stored Number: {coordinates(state.represented[index])}</small></div>)}</div>
    <div className="geometry-arithmetic-paths"><div><h4>Exact integer products</h4><p>{state.exactProducts[0]}<br />− {state.exactProducts[1]}</p><strong>D = {state.exactDeterminant}</strong></div><div><h4>Binary64 products</h4><p>{state.floatProducts[0]}<br />− {state.floatProducts[1]}</p><strong>D = {state.floatDeterminant}</strong></div></div>
    <p role="status">{state.allInputsPreserved ? 'Every input integer survived conversion. The products can still round together.' : 'A and B became the same represented point. Exact arithmetic applied afterward cannot recover the lost coordinate.'}</p>
    <p>The exact path uses BigInt constructed before Number conversion. The other path uses actual JavaScript Number arithmetic. These extremely close directions are not drawn with a fabricated visible gap. Change the exponent to see the first failing range in this fixture; it is not a universal robustness threshold.</p>
    <button onClick={() => {
      setExponent(27);
      setScenario('products');
    }}>Reset precision comparison</button>
  </section>;
}
export function HullEnvelopeFigure() {
  const polygon = polygonPresets.courtyard;
  const hull = hullTrace(polygon).hull;
  return <figure className="geometry-inline"><GeometryPlane title="A concave footprint and its convex envelope" description="The solid gold courtyard boundary has a rectangular notch open at the top. Its dashed blue hull bridges the opening. Q=(4,5) lies inside the hull but outside the courtyard." points={[['Q', [4, 5]]]}>
    <path d={path(polygon, true)} className="geometry-fill geometry-gold" />
    <path d={path(hull, true)} className="geometry-blue geometry-dashed" />
  </GeometryPlane><figcaption>The hull is the smallest <em>convex</em> containing region. It fills this notch; the original footprint does not. Use an envelope for outer extent, not as proof that Q belongs to the original region.</figcaption></figure>;
}
export function ConvexHullLab() {
  const [points, setPoints] = useState(hullPresets.fence);
  const [draft, setDraft] = useState(hullPresets.fence.map(value => value.join(',')).join('\n'));
  const [boundary, setBoundary] = useState(false);
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const state = useMemo(() => hullTrace(points, boundary), [points, boundary]);
  const frame = state.frames[Math.min(step, state.frames.length - 1)];
  const id = useId();
  const load = values => {
    setPoints(values);
    setDraft(values.map(value => value.join(',')).join('\n'));
    setStep(0);
    setError('');
  };
  const completed = frame.phase === 'complete' || frame.phase === 'degenerate';
  return <section className="geometry-lab" aria-label="Monotone convex hull construction">
    <h3>Remove the turn that cannot stay on this chain</h3>
    <p>Predict the next pop before stepping. The lower scan visits sorted points left to right; the upper scan reverses that order.</p>
    <div className="geometry-actions">{Object.entries(hullPresets).map(([name, values]) => <button key={name} onClick={() => load(values)}>{name === 'fence' ? 'Reset fence points' : `Load ${name}`}</button>)}</div>
    <label htmlFor={id}>Point records, one x,y per line<textarea id={id} value={draft} onChange={event => setDraft(event.target.value)} rows="4" spellCheck="false" /></label>
    <button onClick={() => {
      try {
        load(parseGridPoints(draft));
      } catch (failure) {
        setError(failure.message);
      }
    }}>Apply point records</button>
    {error && <p role="alert">{error} The current geometry is unchanged.</p>}
    <p>At most 20 records; integer coordinates 0 through 8. Duplicate locations are merged. Editing text does not change the active trace until Apply.</p>
    <label className="geometry-toggle"><input type="checkbox" checked={boundary} onChange={event => {
        setBoundary(event.target.checked);
        setStep(0);
      }} /> Include every boundary point</label>
    <figure><GeometryPlane title="The active monotone chain" description={`${frame.phase}: ${frame.action}. Active stack ${frame.stack.map(coordinates).join('; ') || 'empty'}. ${frame.removed ? `Removed ${coordinates(frame.removed)} with determinant ${frame.determinant}.` : ''}`} points={state.sorted.map((value, index) => [`P${index + 1}`, value])}>
      {completed && <path d={path(frame.stack, !state.collinear)} className="geometry-fill geometry-gold" />}
      {!completed && <><path d={path(frame.lower)} className="geometry-blue" /><path d={path(frame.stack)} className="geometry-gold" />
        {frame.candidate && frame.stack.length > 0 && <path d={path([frame.stack.at(-1), frame.candidate])} className="geometry-gold geometry-dashed" />}</>}
      {frame.removed && <circle cx={screenX(frame.removed[0])} cy={screenY(frame.removed[1])} r="10" className="geometry-removed" />}
    </GeometryPlane><figcaption>Solid gold: active chain. Blue: finished lower chain. Dashed gold: candidate connection. Ring: just removed from this chain, not deleted from the input or necessarily from the final hull.</figcaption></figure>
    <p className="geometry-current" aria-live="polite">Step {Math.min(step, state.frames.length - 1) + 1}/{state.frames.length} · {frame.phase}: {frame.action}{frame.determinant !== null ? `; D=${frame.determinant}` : ''}.</p>
    <div className="geometry-actions"><button disabled={step === 0} onClick={() => setStep(value => value - 1)}>Previous hull step</button><button disabled={step >= state.frames.length - 1} onClick={() => setStep(value => value + 1)}>Next hull step</button><button onClick={() => setStep(state.frames.length - 1)}>Show completed hull</button><button onClick={() => setStep(0)}>Restart hull trace</button></div>
    <ol className="geometry-point-list">{state.sorted.map((value, index) => <li key={value.join(',')}>P{index + 1} = {coordinates(value)}</li>)}</ol>
    <Facts values={[["Active stack", frame.stack.map(coordinates).join(' → ') || 'empty'], ['Output contract', boundary ? 'all boundary locations' : 'extreme corners only']]} />
    <p>For a nondegenerate hull, output is counterclockwise and does not repeat its first point. An all-collinear result is a sorted segment/list, not a polygon. Try switching boundary policy on the fence and on the line.</p>
  </section>;
}
export function PolygonQueryLab() {
  const [shape, setShape] = useState('courtyard');
  const [query, setQuery] = useState([4, 5]);
  const [edgeIndex, setEdgeIndex] = useState(0);
  const vertices = polygonPresets[shape];
  const state = polygonState(vertices, query);
  const edge = state.edges[edgeIndex];
  const id = useId();
  return <section className="geometry-lab" aria-label="Polygon boundary and crossing parity">
    <h3>Count crossings without double-counting a vertex</h3>
    <p>Q starts in the courtyard opening. Predict its classification, then inspect each edge touched by the horizontal ray.</p>
    <label htmlFor={id}>Simple polygon<select id={id} value={shape} onChange={event => {
        setShape(event.target.value);
        setEdgeIndex(0);
      }}>{Object.keys(polygonPresets).map(name => <option key={name}>{name}</option>)}</select></label>
    <GridCoordinate label="Query Q" value={query} onChange={value => {
      setQuery(value);
      setEdgeIndex(0);
    }} />
    <figure><GeometryPlane title="A horizontal ray from the query" description={`Q is ${state.classification}. Selected edge ${edgeIndex + 1} from ${coordinates(edge.a)} to ${coordinates(edge.b)} ${edge.rightCrossing ? 'counts as a right crossing' : 'does not count'}.`} points={vertices.map((value, index) => [`V${index + 1}`, value]).concat([['Q', query]])}>
      <path d={path(vertices, true)} className="geometry-fill geometry-muted" />
      <path d={path([query, [8, query[1]]])} className="geometry-blue geometry-dashed" />
      <path d={path([edge.a, edge.b])} className="geometry-gold" />
    </GeometryPlane><figcaption>Dashed blue ray continues right beyond the display. Gold is the inspected edge. If Q is on any edge, boundary wins before parity; crossing counts alone are not the boundary classifier.</figcaption></figure>
    <div className="geometry-actions"><button disabled={edgeIndex === 0} onClick={() => setEdgeIndex(value => value - 1)}>Previous polygon edge</button><button disabled={edgeIndex === vertices.length - 1} onClick={() => setEdgeIndex(value => value + 1)}>Next polygon edge</button><button onClick={() => {
        setShape('courtyard');
        setQuery([4, 5]);
        setEdgeIndex(0);
      }}>Reset polygon query</button></div>
    <Facts values={[["Query classification", state.classification], ['Total right crossings', state.crossings], ['Selected edge / its determinant', `${edgeIndex + 1} / ${edge.determinant}`], ['Endpoint above flags', `${edge.a[1] > query[1]} / ${edge.b[1] > query[1]}`], ['Counts on this edge?', edge.rightCrossing ? 'yes' : 'no'], ['Signed doubled area / area', `${state.doubledArea} / ${state.area}`]]} />
    <p>Try Q=(4,3) on the notch floor and Q=(3,4) on its wall. Move Q to (2,5) inside the left wing. The model assumes a simple, hole-free polygon; its presets satisfy that contract.</p>
  </section>;
}
