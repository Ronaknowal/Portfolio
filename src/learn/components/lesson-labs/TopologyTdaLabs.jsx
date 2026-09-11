import { useId, useMemo, useState } from 'react';
import { TOPOLOGY_COMPLEX_FIXTURES, TOPOLOGY_POINT_FIXTURES, complexFromFacets, chainBoundary, bettiAt, ripsFiltration, persistentHomology, evaluateDiagramMatching, optimalDiagramMatching, pixelComplex, landscapeAt, persistenceImage, mapperGraph } from '../../data/topology-tda-models.js';
import './topology-tda-labs.css';
const format = value => value === Infinity ? '∞' : Number(value.toPrecision(6)).toString();
const cellName = vertices => '{' + vertices.join(',') + '}';
const keyName = key => cellName(key.split('-'));
const tones = ['#eac06e', '#8fd9bb', '#9dc4ee', '#d4aff0', '#e4a998'];
function Choice({
  label,
  value,
  options,
  onChange,
  numeric = false
}) {
  return <label className="tda-control">{label}<select aria-label={label} value={value} onChange={event => onChange(numeric ? Number(event.target.value) : event.target.value)}>{options.map(([key, name]) => <option key={key} value={key} disabled={key === ''}>{name}</option>)}</select></label>;
}
function Range({
  label,
  value,
  min,
  max,
  step = 1,
  onChange
}) {
  return <label className="tda-control">{label}: <strong>{format(value)}</strong><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Metrics({
  rows
}) {
  return <dl className="tda-metrics">{rows.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl>;
}
function Geometry({
  points,
  simplices = [],
  selected = [],
  boundary = [],
  activePoints = [],
  visibleVertices = null,
  label,
  solid = false
}) {
  const extent = Math.max(1.2, ...points.flatMap(point => point.map(Math.abs))) * 1.15;
  const x = value => 150 + value / extent * 104;
  const y = value => 150 - value / extent * 104;
  const selectedKeys = new Set(selected);
  const boundaryKeys = new Set(boundary);
  return <svg className="tda-geometry" viewBox="0 0 300 300" role="img" aria-label={label}>
    {solid && <text x={150} y={24} textAnchor="middle">3-cell included</text>}
    {simplices.filter(simplex => simplex.dimension === 2).map(simplex => <polygon key={simplex.key} points={simplex.vertices.map(vertex => x(points[vertex][0]) + ',' + y(points[vertex][1])).join(' ')} className={selectedKeys.has(simplex.key) ? 'tda-face tda-selected-face' : 'tda-face'} />)}
    {simplices.filter(simplex => simplex.dimension === 1).map(simplex => {
      const [left, right] = simplex.vertices.map(vertex => points[vertex]);
      return <line key={simplex.key} x1={x(left[0])} y1={y(left[1])} x2={x(right[0])} y2={y(right[1])} className={boundaryKeys.has(simplex.key) ? 'tda-edge tda-boundary-edge' : selectedKeys.has(simplex.key) ? 'tda-edge tda-selected-edge' : 'tda-edge'} />;
    })}
    {points.map((point, index) => visibleVertices && !visibleVertices.includes(index) ? null : <g key={index}><circle cx={x(point[0])} cy={y(point[1])} r={boundaryKeys.has(String(index)) ? 8 : activePoints.includes(index) || selectedKeys.has(String(index)) ? 7 : 5} className={boundaryKeys.has(String(index)) ? 'tda-boundary-vertex' : activePoints.includes(index) || selectedKeys.has(String(index)) ? 'tda-selected-vertex' : 'tda-vertex'} /><text x={x(point[0]) + (point[0] < 0 ? -13 : 13)} y={y(point[1]) - 10} textAnchor={point[0] < 0 ? 'end' : 'start'}>{index}</text></g>)}
  </svg>;
}
export function ShapeEquivalenceFigure() {
  return <figure className="tda-figure"><div className="tda-triptych">
    <div><svg viewBox="0 0 260 170" role="img" aria-label="Circle before a coordinate transformation"><circle cx={130} cy={85} r={58} className="tda-outline" /></svg><strong>Circle</strong><span>The object is the circumference.</span></div>
    <div><svg viewBox="0 0 260 170" role="img" aria-label="Ellipse after an invertible vertical scaling"><ellipse cx={130} cy={85} rx={58} ry={29} className="tda-outline" /></svg><strong>Ellipse: invertible scaling</strong><span>Vertical scaling changes distances; an inverse recovers each point.</span></div>
    <div><svg viewBox="0 0 260 170" role="img" aria-label="Line segment after collapsing the second coordinate"><line x1={72} y1={85} x2={188} y2={85} className="tda-outline" /><circle cx={72} cy={85} r={4} className="tda-vertex" /><circle cx={188} cy={85} r={4} className="tda-vertex" /></svg><strong>Segment: collapse</strong><span>Different circle points become the same point. There is no inverse.</span></div>
  </div><figcaption>“Continuous” alone does not mean “same shape.” The map (x,y) ↦ (x,ay) is invertible for a&gt;0, but at a=0 it identifies the upper and lower arcs. These drawings compare the objects, not a distance-preserving motion.</figcaption>
  <div className="tda-relative"><svg viewBox="0 0 300 120" role="img" aria-label="A relative neighborhood of the left endpoint of the closed interval"><line x1={45} y1={70} x2={260} y2={70} stroke="#96a8bb" strokeWidth={3} /><line x1={45} y1={70} x2={115} y2={70} stroke="#eac06e" strokeWidth={7} /><circle cx={45} cy={70} r={6} fill="#eac06e" /><circle cx={115} cy={70} r={6} fill="#0c1118" stroke="#eac06e" strokeWidth={2} /><text x={45} y={105} textAnchor="middle">0</text><text x={115} y={105} textAnchor="middle">a</text><text x={260} y={105} textAnchor="middle">1</text><text x={150} y={30} textAnchor="middle">[0,a) inside [0,1]</text></svg><p>The highlighted set is open <em>relative to</em> [0,1], even though it contains the endpoint 0. It is [0,1] intersected with an open real interval (−a,a), for 0&lt;a&lt;1. “Open” needs an ambient space.</p></div></figure>;
}
export function ComplexBoundaryLab() {
  const [kind, setKind] = useState('triangle');
  const fixture = TOPOLOGY_COMPLEX_FIXTURES[kind];
  const simplices = useMemo(() => complexFromFacets(fixture.facets), [fixture]);
  const selectable = simplices.filter(simplex => simplex.dimension === fixture.chainDimension);
  const [selected, setSelected] = useState(['0-1', '0-2', '1-2']);
  const result = chainBoundary(simplices, selected);
  const counts = bettiAt(simplices);
  function changeKind(next) {
    setKind(next);
    const nextFixture = TOPOLOGY_COMPLEX_FIXTURES[next];
    setSelected(complexFromFacets(nextFixture.facets).filter(simplex => simplex.dimension === nextFixture.chainDimension).map(simplex => simplex.key));
  }
  return <section className="tda-lab" aria-label="Chains and fillings investigation"><h3>What makes this cycle count as a hole?</h3><p>Start with all three edges selected. Predict whether adding the face changes their boundary, their status as a cycle, or their status as a <em>boundary of something else</em>.</p>
    <Choice label="Choose the complex" value={kind} options={Object.entries(TOPOLOGY_COMPLEX_FIXTURES).map(([key, item]) => [key, item.label])} onChange={changeKind} />
    <div className="tda-two"><div><Geometry points={fixture.points} simplices={simplices} selected={selected} boundary={result.boundary} solid={kind === 'solid'} label="Selected chain in amber; its boundary in rose; other cells muted" /><p className="tda-note">Amber = selected {fixture.chainDimension === 1 ? 'edges' : 'faces'}. Rose = their boundary. {kind === 'shell' || kind === 'solid' ? 'This is a projection of an abstract tetrahedron; overlapping faces are not additional cells.' : 'A filled triangle is an actual 2-cell, not just three drawn edges.'}</p></div>
      <div><p>Select {fixture.chainDimension}-simplices. Adding a second copy would cancel it over F₂.</p><div className="tda-cell-buttons">{selectable.map(simplex => <button key={simplex.key} type="button" aria-pressed={selected.includes(simplex.key)} onClick={() => setSelected(previous => previous.includes(simplex.key) ? previous.filter(key => key !== simplex.key) : [...previous, simplex.key])}>{cellName(simplex.vertices)}</button>)}</div>
        <Metrics rows={[["Selected chain", selected.length ? selected.map(keyName).join(' + ') : '0 (empty chain)'], ['Its boundary', result.boundary.length ? result.boundary.map(keyName).join(' + ') : '0'], ['Cycle?', result.isCycle ? 'Yes: boundary is zero' : 'No'], ['Boundary of a higher chain?', result.isBoundary ? 'Yes: class is zero' : result.isCycle ? 'No: nonzero homology class' : 'No: it is not a cycle']]} />
        <p className="tda-result" aria-live="polite">Whole complex: β₀={counts.betti[0]}, β₁={counts.betti[1]}, β₂={counts.betti[2]}. These count all classes, not just your selected chain.</p>
      </div></div>
    <div className="tda-actions"><button type="button" onClick={() => setSelected(selectable.map(simplex => simplex.key))}>Select all</button><button type="button" onClick={() => setSelected([])}>Clear chain</button><button type="button" onClick={() => changeKind('triangle')}>Reset</button></div>
    <p>For the square with a diagonal, selecting all five edges leaves a boundary. Select the four perimeter edges instead. Then compare the tetrahedral shell with the solid: the same four-face 2-cycle gains a 3-cell filling.</p>
  </section>;
}
function Barcode({
  bars,
  threshold,
  maximum,
  selected,
  onSelect
}) {
  return <div className="tda-barcode"><p>Positive-duration bars. Left closed, right open.</p>{bars.map((bar, index) => <button key={index} type="button" className={'tda-bar-row' + (selected === index ? ' tda-active-bar' : '')} aria-pressed={selected === index} onClick={() => onSelect(index)} aria-label={'Select H' + bar.dimension + ' bar [' + format(bar.birth) + ', ' + format(bar.death) + ')'}><span>H{bar.dimension}</span><span className="tda-bar-track"><i style={{
          left: bar.birth / maximum * 100 + '%',
          width: ((bar.death === Infinity ? maximum : bar.death) - bar.birth) / maximum * 100 + '%',
          background: bar.dimension ? tones[0] : tones[1]
        }} /><b style={{
          left: threshold / maximum * 100 + '%'
        }} /><em className="tda-bar-start" style={{
          left: bar.birth / maximum * 100 + '%'
        }} />{bar.death !== Infinity && <em className="tda-bar-end" style={{
          left: bar.death / maximum * 100 + '%'
        }} />}</span><span>{bar.death === Infinity ? '∞' : format(bar.death)}</span></button>)}<div className="tda-bar-axis"><span>ε=0</span><span>ε={format(maximum)}</span></div></div>;
}
function BirthDeathPlot({
  bars,
  maximum,
  selected = null,
  label = 'Finite persistence points'
}) {
  const x = value => 40 + value / maximum * 225;
  const y = value => 265 - value / maximum * 225;
  const grouped = new Map();
  bars.forEach((bar, index) => {
    if (bar.death === Infinity) return;
    const key = bar.birth + ',' + bar.death;
    if (!grouped.has(key)) grouped.set(key, {
      bar,
      indices: []
    });
    grouped.get(key).indices.push(index);
  });
  return <svg className="tda-geometry" viewBox="0 0 300 310" role="img" aria-label={label}><line x1={40} y1={265} x2={265} y2={265} className="tda-axis" /><line x1={40} y1={265} x2={40} y2={40} className="tda-axis" /><line x1={40} y1={265} x2={265} y2={40} className="tda-diagonal" /><text x={38} y={286}>0</text><text x={265} y={286} textAnchor="end">{format(maximum)}</text><text x={145} y={306}>birth ε</text><text x={40} y={24}>death ε</text><text x={46} y={54}>{format(maximum)}</text>{[...grouped.values()].map(({
      bar,
      indices
    }) => <g key={bar.birth + ',' + bar.death}><circle cx={x(bar.birth)} cy={y(bar.death)} r={indices.includes(selected) ? 8 : 5} fill={bar.dimension ? tones[0] : tones[1]} stroke="#0c1118" strokeWidth={2} /><text x={x(bar.birth) + 11} y={y(bar.death) + 6}>{'H' + bar.dimension + (indices.length > 1 ? ' ×' + indices.length : '')}</text></g>)}</svg>;
}
export function RipsFiltrationLab() {
  const [kind, setKind] = useState('square');
  const [scale, setScale] = useState(1);
  const [threshold, setThreshold] = useState(2);
  const [selected, setSelected] = useState(4);
  const computation = useMemo(() => {
    const points = TOPOLOGY_POINT_FIXTURES[kind].points.map(point => point.map(value => value * scale));
    const filtration = ripsFiltration(points);
    const result = persistentHomology(filtration);
    const bars = result.intervals.filter(bar => bar.dimension <= 1 && bar.death > bar.birth);
    const maximum = Math.max(...filtration.map(simplex => simplex.birth));
    return {
      points,
      filtration,
      bars,
      maximum
    };
  }, [kind, scale]);
  const active = computation.filtration.filter(simplex => simplex.birth <= threshold);
  const result = useMemo(() => bettiAt(computation.filtration, threshold), [computation, threshold]);
  const bar = computation.bars[Math.min(selected, computation.bars.length - 1)];
  const representative = bar && bar.birth <= threshold && threshold < bar.death ? bar.representative.map(index => computation.filtration[index].key) : [];
  function changeFixture(nextKind, nextScale) {
    setKind(nextKind);
    setScale(nextScale);
    setThreshold(0);
    setSelected(0);
  }
  return <section className="tda-lab" aria-label="Rips filtration and persistence investigation"><h3>Connect, create a loop, then fill it</h3><p>At ε=2 the square has its perimeter but no diagonal. Predict the next event: an extra edge by itself or a whole tied group of edges and faces?</p><div className="tda-controls"><Choice label="Point sample" value={kind} options={Object.entries(TOPOLOGY_POINT_FIXTURES).map(([key, item]) => [key, item.label])} onChange={next => changeFixture(next, scale)} /><Choice label="Uniform coordinate scale" value={scale} numeric options={[[0.5, 'Half size'], [1, 'Original size'], [2, 'Double size']]} onChange={next => changeFixture(kind, next)} /></div>
    <Range label="Rips edge threshold ε" value={threshold} min={0} max={computation.maximum} step="any" onChange={setThreshold} />
    <div className="tda-actions"><button type="button" onClick={() => setThreshold(0)}>Vertices</button>{computation.bars.filter(candidate => candidate.dimension === 1).slice(0, 1).map(candidate => <span className="tda-actions" key="events"><button type="button" onClick={() => setThreshold(candidate.birth)}>Exact H₁ birth</button><button type="button" onClick={() => setThreshold(candidate.death)}>Exact H₁ death</button></span>)}<button type="button" onClick={() => setThreshold(computation.maximum)}>Full 2-skeleton</button><button type="button" onClick={() => {
        changeFixture('square', 1);
        setThreshold(2);
        setSelected(4);
      }}>Reset</button></div>
    <div className="tda-two"><div><Geometry points={computation.points} simplices={active} selected={representative} label="Rips complex at the selected scale with selected live representative in amber" /><Metrics rows={[["Active vertices / edges / faces", result.counts.slice(0, 3).join(' / ')], ['Components β₀', result.betti[0]], ['Unfilled loop classes β₁', result.betti[1]]]} /><p className="tda-note">Vertices are the observations. Lines and translucent faces are constructed cells. Crossings in this projection are not new vertices.</p></div><div><Barcode bars={computation.bars} threshold={threshold} maximum={computation.maximum} selected={selected} onSelect={setSelected} /><BirthDeathPlot bars={computation.bars} maximum={computation.maximum * 1.2} selected={selected} /><p className="tda-note">The essential H₀ bar at ∞ is listed above and omitted from this finite plot. Coincident finite points display their multiplicity.</p></div></div>
    <p className="tda-result" aria-live="polite">Selected H{bar.dimension} interval [{format(bar.birth)}, {format(bar.death)}): {bar.birth <= threshold && threshold < bar.death ? 'alive at this scale' : threshold < bar.birth ? 'not born yet' : 'already died'}. Its representative is one valid choice, not necessarily shortest or unique.</p>
    <p>All computations use F₂ and the diameter ≤ ε convention. Cells through dimension 2 give complete H₀/H₁ for this finite full filtration; this panel makes no H₂ claim. Geometry automatically fits the selected points; compare coordinate scales through ε values, not screen pixel widths. Coordinates and thresholds use binary64 arithmetic; displayed decimals are rounded. Exact-event buttons use the computed threshold rather than its rounded label.</p>
  </section>;
}
export function CechRipsFigure() {
  return <figure className="tda-figure"><div className="tda-two">{[1, 2 / Math.sqrt(3)].map((radius, index) => <div key={index}><svg className="tda-geometry" viewBox="0 0 300 280" role="img" aria-label={index ? 'Three closed balls first have a common center point' : 'Pairwise ball intersections leave the center uncovered'}>{[[0, 2 / Math.sqrt(3)], [-1, -1 / Math.sqrt(3)], [1, -1 / Math.sqrt(3)]].map(([x, y], vertex) => <g key={vertex}><circle cx={150 + 55 * x} cy={145 - 55 * y} r={55 * radius} fill={tones[vertex]} fillOpacity={0.12} stroke={tones[vertex]} strokeWidth={2} /><circle cx={150 + 55 * x} cy={145 - 55 * y} r={4} fill={tones[vertex]} /><line x1={150} y1={145} x2={150 + 55 * x} y2={145 - 55 * y} className="tda-diagonal" /></g>)}<circle cx={150} cy={145} r={4} fill={index ? '#f5ead4' : '#0c1118'} stroke="#f5ead4" strokeWidth={2} /><text x={150} y={258} textAnchor="middle">ball radius = {index ? '2/√3' : '1'}</text></svg><strong>ε = {index ? '4/√3 ≈ 2.3094' : '2'}</strong><p>{index ? 'Čech now includes the triangle: the center belongs to all three closed balls.' : 'All pairs touch. Rips includes the triangle, but Čech does not: there is no common three-ball point.'}</p></div>)}</div><figcaption>Equilateral side length 2, with balls of radius ε/2. The center-to-vertex distance is 2/√3. Pairwise intersection is weaker than a common intersection; changing the convention from diameter ε to radius ε would change the displayed numbers.</figcaption></figure>;
}
export function BoundaryReductionLab() {
  const [kind, setKind] = useState('triangle');
  const [step, setStep] = useState(0);
  const computation = useMemo(() => {
    const points = kind === 'triangle' ? [[0, 1], [-1, -1], [1, -1]] : TOPOLOGY_POINT_FIXTURES.square.points;
    const filtration = kind === 'triangle' ? complexFromFacets([[0, 1, 2]]) : ripsFiltration(points);
    const result = persistentHomology(filtration, {
      captureTrace: true
    });
    const frames = result.trace.flatMap(trace => trace.stages.map((stage, stageIndex) => ({
      ...stage,
      column: trace.column,
      final: stageIndex === trace.stages.length - 1,
      pivot: trace.pivot,
      event: trace.event
    })));
    return {
      points,
      result,
      frames
    };
  }, [kind]);
  const frame = computation.frames[step];
  const {
    ordered,
    boundaries,
    trace
  } = computation.result;
  const combinationKeys = frame.combination.map(index => ordered[index].key);
  const boundaryKeys = frame.boundary.map(index => ordered[index].key);
  function nextKind(value) {
    setKind(value);
    setStep(0);
  }
  return <section className="tda-lab" aria-label="Persistent boundary reduction investigation"><h3>Watch a boundary cancel, one column addition at a time</h3><p>Each 1 says that the row simplex occurs in the boundary of the column chain. Predict which entries flip when an earlier column is added over F₂. An empty column is a computed event, not missing data.</p><Choice label="Reduction fixture" value={kind} options={[["triangle", "Filled triangle: seven simplices"], ["square", "Square diameter filtration: fourteen simplices"]]} onChange={nextKind} />
    <div className="tda-actions"><button type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>Back</button><span>Step {step + 1} of {computation.frames.length}</span><button type="button" disabled={step === computation.frames.length - 1} onClick={() => setStep(step + 1)}>Next operation</button><button type="button" onClick={() => setStep(computation.frames.findIndex(candidate => candidate.addedColumn !== null))}>Jump to first XOR</button><button type="button" onClick={() => setStep(0)}>Reset</button></div>
    <p className="tda-result" aria-live="polite">Column {frame.column}: {cellName(ordered[frame.column].vertices)}. {frame.addedColumn === null ? 'Start from its original boundary.' : 'XOR the reduced column ' + frame.addedColumn + '.'} {frame.final ? frame.event === 'birth' ? 'Reduction is zero: this column creates a class.' : 'Final pivot row ' + frame.pivot + ': pair its creator with this column.' : 'The current pivot still needs cancellation.'}</p>
    <p className="tda-note">Earlier columns show their completed reductions; the highlighted column shows the current intermediate state; future columns retain their original boundaries. Scroll the matrix sideways on a narrow screen.</p>
    <div className="tda-matrix-scroll" role="region" tabIndex={0} aria-label="Boundary matrix; horizontal scrolling available"><table className="tda-matrix"><caption>Rows and columns in the same filtration order</caption><thead><tr><th>row / column</th>{ordered.map((simplex, index) => <th key={simplex.key} className={index === frame.column ? 'tda-current-cell' : ''} title={cellName(simplex.vertices)}>{index}</th>)}</tr></thead><tbody>{ordered.map((simplex, row) => <tr key={simplex.key}><th scope="row">{row}: {cellName(simplex.vertices)}</th>{ordered.map((column, index) => {
              const values = index === frame.column ? frame.boundary : index < frame.column ? trace[index].stages.at(-1).boundary : boundaries[index];
              return <td key={column.key} className={(index === frame.column ? 'tda-current-cell ' : index > frame.column ? 'tda-future-cell ' : '') + (values.includes(row) ? 'tda-one' : '')}>{values.includes(row) ? 1 : 0}</td>;
            })}</tr>)}</tbody></table></div>
    <div className="tda-two"><Geometry points={computation.points} simplices={ordered.slice(0, frame.column + 1)} selected={combinationKeys} boundary={boundaryKeys} visibleVertices={ordered.slice(0, frame.column + 1).filter(simplex => simplex.dimension === 0).map(simplex => simplex.vertices[0])} label="The actual combined chain and its remaining boundary after the selected operation" /><div><Metrics rows={[["Current chain combination", combinationKeys.map(keyName).join(' + ')], ['Its boundary', boundaryKeys.length ? boundaryKeys.map(keyName).join(' + ') : '0'], ['Greatest nonzero row', frame.boundary.length ? Math.max(...frame.boundary) : 'none']]} /><p>The amber chain is reconstructed from all contributing original columns. The rose boundary agrees with the highlighted matrix column. Geometry shows the processed simplex prefix; within tied values this is a computational state, not a new physical scale.</p></div></div>
    {kind === 'triangle' && <p>Every triangle cell has filtration value 0 in this fixture. The loop is created and filled at that same value, so its pair has zero duration. Switch to the square to see a positive-duration H₁ interval.</p>}
  </section>;
}
const matchingFixtures = {
  close: {
    label: 'One slightly changed interval',
    first: [[1, 4]],
    second: [[1.2, 3.8]]
  },
  greedy: {
    label: 'Two points: greedy assignment trap',
    first: [[2, 10], [4, 10]],
    second: [[3, 10], [0, 10]]
  },
  copies: {
    label: 'Multiplicity: two copies versus one',
    first: [[1, 4], [1, 4]],
    second: [[1, 4]]
  },
  empty: {
    label: 'One diagram is empty',
    first: [[1, 4]],
    second: []
  }
};
export function DiagramMatchingLab() {
  const clipId = useId();
  const [kind, setKind] = useState('close');
  const [assignments, setAssignments] = useState([0]);
  const [power, setPower] = useState(Infinity);
  const fixture = matchingFixtures[kind];
  const current = evaluateDiagramMatching(fixture.first, fixture.second, assignments, power);
  const optimum = useMemo(() => optimalDiagramMatching(fixture.first, fixture.second, power), [fixture, power]);
  const maximum = Math.max(...fixture.first.concat(fixture.second).map(point => point[1])) + 1;
  const x = value => 40 + value / maximum * 225;
  const y = value => 265 - value / maximum * 225;
  function changeKind(value) {
    setKind(value);
    setAssignments(matchingFixtures[value].first.map((_, index) => index < matchingFixtures[value].second.length ? index : null));
  }
  function assign(index, target) {
    setAssignments(previous => previous.map((value, position) => position === index ? target : target !== null && value === target ? null : value));
  }
  const groupedMarkers = [];
  for (const [name, diagram, tone] of [['A', fixture.first, tones[0]], ['B', fixture.second, tones[1]]]) {
    const groups = new Map();
    diagram.forEach((point, index) => {
      const key = point.join(',');
      if (!groups.has(key)) groups.set(key, {
        point,
        names: []
      });
      groups.get(key).names.push(name + index);
    });
    groups.forEach(group => groupedMarkers.push({
      ...group,
      name,
      tone
    }));
  }
  return <section className="tda-lab" aria-label="Diagram matching investigation"><h3>A feature may match another feature—or the diagonal</h3><p>For the two-point trap, predict the cost of swapping partners. Matching is one-to-one; selecting an occupied partner sends its previous match to the diagonal.</p><div className="tda-controls"><Choice label="Diagram pair" value={kind} options={Object.entries(matchingFixtures).map(([key, item]) => [key, item.label])} onChange={changeKind} /><Choice label="Cost aggregation" value={power} numeric options={[[Infinity, 'Bottleneck: largest cost'], [1, '1-Wasserstein: sum'], [2, '2-Wasserstein: root sum of squares']]} onChange={setPower} /></div>
    <div className="tda-two"><svg className="tda-geometry" viewBox="0 0 300 310" role="img" aria-label="Birth-death diagrams A and B, chosen match segments and L-infinity neighborhoods"><defs><clipPath id={clipId}><rect x={40} y={40} width={225} height={225} /></clipPath></defs><line x1={40} y1={265} x2={265} y2={265} className="tda-axis" /><line x1={40} y1={265} x2={40} y2={40} className="tda-axis" /><line x1={40} y1={265} x2={265} y2={40} className="tda-diagonal" /><text x={40} y={26}>death</text><text x={145} y={306}>birth</text><text x={36} y={285}>0</text><text x={253} y={285}>{maximum}</text><g clipPath={'url(#' + clipId + ')'}>{fixture.first.map((point, index) => {
            const target = assignments[index] === null ? [(point[0] + point[1]) / 2, (point[0] + point[1]) / 2] : fixture.second[assignments[index]];
            const cost = Math.max(Math.abs(point[0] - target[0]), Math.abs(point[1] - target[1]));
            return <g key={index}><rect x={x(point[0] - cost)} y={y(point[1] + cost)} width={cost / maximum * 450} height={cost / maximum * 450} fill="none" stroke={tones[0]} strokeOpacity={0.28} strokeDasharray="4 3" /><line x1={x(point[0])} y1={y(point[1])} x2={x(target[0])} y2={y(target[1])} stroke={tones[0]} strokeWidth={2} /><circle cx={x(target[0])} cy={y(target[1])} r={3} fill="#f0ede5" /></g>;
          })}{current.unmatchedSecond.map(index => {
            const point = fixture.second[index];
            const middle = (point[0] + point[1]) / 2;
            return <line key={index} x1={x(point[0])} y1={y(point[1])} x2={x(middle)} y2={y(middle)} stroke={tones[1]} strokeWidth={2} strokeDasharray="3 3" />;
          })}</g>{groupedMarkers.map(({
          point,
          names,
          name,
          tone
        }) => <g key={names.join()}>{name === 'A' ? <circle cx={x(point[0])} cy={y(point[1])} r={5} fill={tone} /> : <rect x={x(point[0]) - 4} y={y(point[1]) - 4} width={8} height={8} fill={tone} transform={'rotate(45 ' + x(point[0]) + ' ' + y(point[1]) + ')'} />}<text x={x(point[0]) + 8} y={y(point[1]) + (name === 'A' ? -12 : 20)}>{names.join('/')}</text></g>)}</svg>
      <div>{fixture.first.map((point, index) => <Choice key={index} label={'Match A' + index + ' (' + point.join(', ') + ')'} value={assignments[index] === null ? 'diagonal' : assignments[index]} options={[["diagonal", "Diagonal: half the lifetime"], ...fixture.second.map((target, targetIndex) => [targetIndex, 'B' + targetIndex + ' (' + target.join(', ') + ')'])]} onChange={value => assign(index, value === 'diagonal' ? null : Number(value))} />)}<Metrics rows={[["Individual costs (including unmatched B)", current.costs.map(format).join(', ')], ['Your matching cost', format(current.value)], ['Optimal cost', format(optimum.value)]]} /><div className="tda-actions"><button type="button" onClick={() => setAssignments(optimum.assignments)}>Use an optimal matching</button><button type="button" onClick={() => {
            changeKind('close');
            setPower(Infinity);
          }}>Reset</button></div></div></div>
    <p>Amber circles are A; green diamonds are B. Dashed squares show the L∞ radius around each A point, clipped to the plot. Unmatched green points also pay to reach the diagonal. Coincident labels retain distinct copies. These are finite diagrams; essential bars require compatible treatment outside this panel.</p>
  </section>;
}
const pixelPresets = {
  ring: {
    label: 'Low ring, high center',
    values: [1, 1, 1, 1, 3, 1, 1, 1, 1]
  },
  gap: {
    label: 'Ring with a high top gap',
    values: [1, 3, 1, 1, 3, 1, 1, 1, 1]
  },
  corner: {
    label: 'Two diagonal low pixels',
    values: [1, 3, 3, 3, 1, 3, 3, 3, 3]
  }
};
export function PixelFiltrationLab() {
  const [values, setValues] = useState(pixelPresets.ring.values);
  const [threshold, setThreshold] = useState(1);
  const [preset, setPreset] = useState('ring');
  const result = pixelComplex(values, threshold);
  function changePreset(value) {
    setPreset(value);
    setValues([...pixelPresets[value].values]);
  }
  return <section className="tda-lab" aria-label="Cubical image filtration investigation"><h3>The same intensity threshold builds a different kind of space</h3><p>Predict whether filling the center or raising the top-middle pixel destroys the ring's hole. Click a pixel to switch its value between 1 and 3.</p><Choice label="Image starting pattern" value={preset} options={Object.entries(pixelPresets).map(([key, item]) => [key, item.label])} onChange={changePreset} /><Range label="Include pixels with value at most" value={threshold} min={0} max={3} onChange={setThreshold} /><div className="tda-two"><div className="tda-pixel-grid">{values.map((value, index) => <button key={index} type="button" className={value <= threshold ? 'tda-pixel-in' : 'tda-pixel-out'} aria-label={'Pixel row ' + (Math.floor(index / 3) + 1) + ', column ' + (index % 3 + 1) + ', value ' + value + ', ' + (value <= threshold ? 'included' : 'not included') + '; click to change value'} onClick={() => setValues(previous => previous.map((number, pixel) => pixel === index ? 4 - number : number))}><strong>{value}</strong><span>{value <= threshold ? 'included' : 'waiting'}</span></button>)}</div><div><Metrics rows={[["Vertices V", result.counts[0]], ['Edges E', result.counts[1]], ['Filled squares F', result.counts[2]], ['Euler: V − E + F', result.euler], ['Components β₀', result.betti[0]], ['Holes β₁ = β₀ − Euler', result.betti[1]]]} /><p className="tda-result" aria-live="polite">{result.activePixels.length} closed pixels are included. Shared edges and corner vertices are counted once.</p></div></div><button type="button" onClick={() => {
      changePreset('ring');
      setThreshold(1);
    }}>Reset</button><p>The display has no periodic boundary identification. In this finite planar union H₂=0; that is the reason the Euler calculation determines β₁ here. A four-neighbor pixel-labeling algorithm would use a different connectivity convention.</p></section>;
}
export function DelayEmbeddingFigure() {
  return <figure className="tda-figure"><div className="tda-triptych">{[0, Math.PI / 6, Math.PI / 2].map((delay, index) => {
        const coordinates = Array.from({
          length: 121
        }, (_, step) => {
          const time = step / 120 * 2 * Math.PI;
          return [Math.cos(time), Math.cos(time - delay)];
        });
        return <div key={index}><svg viewBox="0 0 260 260" role="img" aria-label={['Zero delay collapses to a diagonal segment', 'Pi over six delay makes a thin ellipse', 'Pi over two delay makes a circle'][index]}><line x1={30} y1={130} x2={230} y2={130} className="tda-axis" /><line x1={130} y1={30} x2={130} y2={230} className="tda-axis" /><path d={coordinates.map(([first, second], point) => (point ? 'L' : 'M') + (130 + 85 * first) + ',' + (130 - 85 * second)).join(' ')} className="tda-outline" /><circle cx={215} cy={130 - 85 * Math.cos(delay)} r={5} className="tda-vertex" /><text x={130} y={20} textAnchor="middle">s(t−τ)</text><text x={230} y={150} textAnchor="end">s(t)</text><text x={215} y={250} textAnchor="middle">+1</text><text x={45} y={250} textAnchor="middle">−1</text></svg><strong>τ = {['0', 'π/6', 'π/2'][index]}</strong><span>determinant = {['0', '1/2', '1'][index]}</span></div>;
      })}</div><figcaption>For s(t)=cos t, the delayed coordinate is cos t cos τ + sin t sin τ. An invertible two-coordinate map preserves the continuous loop; zero determinant collapses it. The curves are derived from this synthetic signal. Finite samples, noise and a real sensor's unknown dynamics require separate analysis.</figcaption></figure>;
}
const featureDiagrams = {
  overlap: {
    label: 'Two overlapping bars',
    bars: [[0, 3], [1, 4]]
  },
  single: {
    label: 'A single bar',
    bars: [[0, 3]]
  },
  short: {
    label: 'Short bar near the diagonal',
    bars: [[1, 1.2]]
  },
  empty: {
    label: 'No finite bars',
    bars: []
  }
};
export function PersistenceRepresentationLab() {
  const [kind, setKind] = useState('overlap');
  const [time, setTime] = useState(2);
  const [bandwidth, setBandwidth] = useState(0.5);
  const [pixelIndex, setPixelIndex] = useState(8);
  const diagram = featureDiagrams[kind].bars;
  const image = useMemo(() => persistenceImage(diagram, {
    bandwidth
  }), [diagram, bandwidth]);
  const pixel = image.pixels[pixelIndex];
  const maximum = Math.max(1e-12, ...image.pixels.map(item => item.value));
  const x = value => 35 + value * 57;
  const y = value => 230 - value * 100;
  const curve = level => Array.from({
    length: 161
  }, (_, index) => [index / 40, landscapeAt(diagram, index / 40, level).value]).map(([first, second], index) => (index ? 'L' : 'M') + x(first) + ',' + y(second)).join(' ');
  return <section className="tda-lab" aria-label="Persistence landscapes and images investigation"><h3>Keep the bars fixed; change how their information is represented</h3><p>The landscape ranks tent heights. The image integrates weighted Gaussian contributions over rectangles. Neither picture is a probability that a loop exists.</p><div className="tda-controls"><Choice label="Finite bars" value={kind} options={Object.entries(featureDiagrams).map(([key, item]) => [key, item.label])} onChange={setKind} /><Range label="Gaussian bandwidth σ" value={bandwidth} min={0.1} max={1.5} step={0.1} onChange={setBandwidth} /></div><p>Bars: {diagram.length ? diagram.map(([birth, death]) => '[' + birth + ', ' + death + ')').join(' and ') : 'empty diagram'}. Image weight is min(persistence,1). The fixed window is birth [0,4], persistence [0,4].</p>
    <div className="tda-two"><div><svg className="tda-geometry" viewBox="0 0 300 290" role="img" aria-label="First and second persistence landscape levels"><line x1={35} y1={230} x2={263} y2={230} className="tda-axis" /><line x1={35} y1={230} x2={35} y2={30} className="tda-axis" />{[0, 1, 2, 3, 4].map(tick => <text key={tick} x={x(tick)} y={253} textAnchor="middle">{tick}</text>)}{[1, 2].map(tick => <text key={tick} x={27} y={y(tick) + 5} textAnchor="end">{tick}</text>)}<path d={curve(1)} fill="none" stroke={tones[0]} strokeWidth={3} /><path d={curve(2)} fill="none" stroke={tones[1]} strokeWidth={3} strokeDasharray="6 3" /><line x1={x(time)} y1={30} x2={x(time)} y2={230} className="tda-diagonal" /><text x={40} y={20}>tent height</text><text x={150} y={279} textAnchor="middle">landscape coordinate t</text></svg><p className="tda-note">Amber solid: λ₁, largest tent. Green dashed: λ₂, second largest. Higher levels are zero for these fixtures.</p><Range label="Inspect landscape coordinate t" value={time} min={0} max={4} step={0.05} onChange={setTime} /><p>Individual tents: {landscapeAt(diagram, time).tents.map(format).join(', ') || 'none'}. λ₁={format(landscapeAt(diagram, time, 1).value)}, λ₂={format(landscapeAt(diagram, time, 2).value)}.</p></div>
      <div><p><strong>Integrated persistence image</strong><br />Persistence increases upward; birth increases rightward.</p><div className="tda-image-grid">{[3, 2, 1, 0].flatMap(row => image.pixels.filter(item => item.row === row).map(item => <button key={item.row * 4 + item.column} type="button" aria-pressed={pixelIndex === item.row * 4 + item.column} aria-label={'Select pixel birth ' + item.column + ' to ' + (item.column + 1) + ', persistence ' + row + ' to ' + (row + 1) + ', weight ' + format(item.value)} onClick={() => setPixelIndex(item.row * 4 + item.column)} style={{
            backgroundColor: 'rgba(201, 143, 49, ' + (0.08 + 0.65 * item.value / maximum) + ')'
          }}><span>{item.value < 0.001 ? item.value.toExponential(1) : item.value.toFixed(3)}</span></button>))}</div><p className="tda-note">Shade is scaled to the largest pixel in this image; the printed numbers carry the actual weights.</p><Metrics rows={[["Selected birth × persistence rectangle", '[' + pixel.column + ',' + (pixel.column + 1) + '] × [' + pixel.row + ',' + (pixel.row + 1) + ']'], ['Contributions from each bar', pixel.contributions.map(format).join(' + ') || 'none'], ['Selected pixel sum', format(pixel.value)], ['Weight inside / outside window', format(image.capturedWeight) + ' / ' + format(image.omittedWeight)]]} /></div></div>
    <button type="button" onClick={() => {
      setKind('overlap');
      setTime(2);
      setBandwidth(0.5);
      setPixelIndex(8);
    }}>Reset</button><p>Changing σ spreads the same weighted surface over a wider area; it also changes how much falls outside this fixed window. Pixel integrals use normal-CDF differences. The plotted landscape is sampled for drawing, while the displayed height is evaluated directly.</p>
  </section>;
}
export function MapperLab() {
  const [intervalCount, setIntervalCount] = useState(4);
  const [overlap, setOverlap] = useState(0.4);
  const [clusterDistance, setClusterDistance] = useState(0.6);
  const [focus, setFocus] = useState({
    kind: 'point',
    id: 3
  });
  const result = useMemo(() => mapperGraph({
    intervalCount,
    overlap,
    clusterDistance
  }), [intervalCount, overlap, clusterDistance]);
  const focusedNode = focus.kind === 'node' ? result.nodes[Math.min(focus.id, result.nodes.length - 1)] : null;
  const focusedPoints = focusedNode ? focusedNode.members : [focus.id];
  const focusedNodes = focusedNode ? [focusedNode.id] : result.membership[focus.id];
  const x = value => 150 + value * 105;
  const y = value => 150 - value * 105;
  const nodePositions = result.nodes.map(node => {
    const group = result.nodes.filter(candidate => candidate.interval === node.interval);
    const index = group.findIndex(candidate => candidate.id === node.id);
    return [40 + node.interval / (intervalCount - 1) * 220, group.length === 1 ? 150 : 50 + index / (group.length - 1) * 200];
  });
  function reset() {
    setIntervalCount(4);
    setOverlap(0.4);
    setClusterDistance(0.6);
    setFocus({
      kind: 'point',
      id: 3
    });
  }
  return <section className="tda-lab" aria-label="Mapper lens and overlap investigation"><h3>Build an overlap graph from the same observation IDs</h3><p>Trace point 3 through the colored x-coordinate bands. It belongs to two clusters, so those two Mapper nodes share an edge. Change the clustering threshold to 1.8 and predict which within-band components merge.</p><div className="tda-controls"><Choice label="Number of cover intervals" value={intervalCount} numeric options={[2, 3, 4, 5].map(value => [value, String(value)])} onChange={value => {
        setIntervalCount(value);
        setFocus({
          kind: 'point',
          id: 3
        });
      }} /><Choice label="Overlap fraction of interval width" value={overlap} numeric options={[[0.1, '0.1'], [0.4, '0.4'], [0.65, '0.65']]} onChange={value => {
        setOverlap(value);
        setFocus({
          kind: 'point',
          id: 3
        });
      }} /><Choice label="Within-band cluster edge threshold" value={clusterDistance} numeric options={[[0.1, '0.1: isolated observations'], [0.6, '0.6: short neighbor links'], [1.8, '1.8: merge across the band']]} onChange={value => {
        setClusterDistance(value);
        setFocus({
          kind: 'point',
          id: 3
        });
      }} /></div><Choice label="Trace an observation ID" value={focus.kind === 'point' ? focus.id : ''} numeric options={[["", "Choose an observation"], ...result.points.map((_, index) => [index, 'Point ' + index])]} onChange={id => setFocus({
      kind: 'point',
      id
    })} />
    <div className="tda-two"><div><svg className="tda-geometry" viewBox="0 0 300 315" role="img" aria-label="Twelve circle observations with overlapping vertical lens cover bands">{result.cover.map((interval, index) => <rect key={index} x={x(interval.lower)} y={35} width={x(interval.upper) - x(interval.lower)} height={230} fill={tones[index]} fillOpacity={0.11} stroke={tones[index]} strokeOpacity={0.45} />)}{result.points.map((point, index) => <g key={index}><circle cx={x(point[0])} cy={y(point[1])} r={focusedPoints.includes(index) ? 8 : 4} fill={focusedPoints.includes(index) ? '#f0d08b' : '#bdcad9'} /><text x={x(point[0]) + (point[0] < 0 ? -11 : 10)} y={y(point[1]) + (point[1] < 0 ? 20 : -10)} textAnchor={point[0] < 0 ? 'end' : 'start'}>{index}</text></g>)}<text x={150} y={303} textAnchor="middle">lens = x-coordinate</text></svg><p className="tda-note">The points lie on the same radius-one circle in every state. Closed intervals overlap in lens space; shared observations, not overlap alone, create nerve edges.</p></div>
      <div><svg className="tda-geometry" viewBox="0 0 300 315" role="img" aria-label="Mapper nerve graph; nodes grouped by cover interval">{result.edges.map(edge => <line key={edge.first + '-' + edge.second} x1={nodePositions[edge.first][0]} y1={nodePositions[edge.first][1]} x2={nodePositions[edge.second][0]} y2={nodePositions[edge.second][1]} stroke={edge.members.some(point => focusedPoints.includes(point)) ? tones[0] : '#65768a'} strokeWidth={edge.members.some(point => focusedPoints.includes(point)) ? 3 : 1.5} />)}{result.nodes.map(node => <g key={node.id}><circle cx={nodePositions[node.id][0]} cy={nodePositions[node.id][1]} r={12} fill={focusedNodes.includes(node.id) ? tones[0] : '#172335'} stroke={tones[node.interval]} strokeWidth={2} /><text x={nodePositions[node.id][0]} y={nodePositions[node.id][1] + 5} textAnchor="middle" className={focusedNodes.includes(node.id) ? 'tda-dark-label' : ''}>{node.id}</text></g>)}<text x={150} y={303} textAnchor="middle">cover index increases →</text></svg><Metrics rows={[["Mapper nodes / overlap edges", result.nodes.length + ' / ' + result.edges.length], ['Focused observations', focusedPoints.join(', ')], ['Focused node IDs', focusedNodes.join(', ')]]} /><p className="tda-note">Layout positions separate cover groups and cluster rows. Their lengths, angles and spacing are not measurements of the source data.</p></div></div>
    <div className="tda-member-list">{result.nodes.map(node => <button type="button" key={node.id} aria-pressed={focusedNode?.id === node.id} onClick={() => setFocus({
        kind: 'node',
        id: node.id
      })}><strong>Node {node.id} · cover {node.interval}</strong><span>points {node.members.join(', ')}</span></button>)}</div>
    <details><summary>Inspect exact cover intervals and overlap members</summary><ul>{result.cover.map((interval, index) => <li key={index}>Cover {index}: [{format(interval.lower)}, {format(interval.upper)}], x-coordinate units.</li>)}</ul><ul>{result.edges.map(edge => <li key={edge.first + '-' + edge.second}>Nodes {edge.first}–{edge.second} share points {edge.members.join(', ')}.</li>)}</ul>{result.edges.length === 0 && <p>No two clusters share an observation in this state.</p>}</details>
    {result.maximumMembership >= 3 && <p className="tda-result">Some observations occur in {result.maximumMembership} nodes. Their common intersection supplies higher-dimensional nerve simplices. This displayed 1-skeleton omits those fillings; do not count its graph loops as the full nerve's H₁.</p>}
    <button type="button" onClick={reset}>Reset</button><p>Try three intervals with overlap 0.4: the cover intervals overlap, but this particular finite sample leaves their intersections empty. Then return to four intervals. Parameter dependence is part of interpreting Mapper, not a defect to hide.</p>
  </section>;
}
