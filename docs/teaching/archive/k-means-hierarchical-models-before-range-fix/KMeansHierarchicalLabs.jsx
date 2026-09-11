import { useId, useState } from 'react';
import { clusteringPoints, rectanglePoints, lloydTrace, featureGeometry, seedingDistribution, hierarchyTrace, cutHierarchy, palettePixels, quantizePalette } from '../../data/k-means-hierarchical-models';
import './k-means-hierarchical-labs.css';
const colors = ['#e7b94a', '#8eb9a5', '#91aecf', '#da9c86', '#b8a0cd', '#d0c7ae'];
const number = value => Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
const tick = value => String(Number(value.toPrecision(3)));
const coordinate = point => `(${point.map(number).join(', ')})`;
function Investigation({
  title,
  prediction,
  children,
  onReset
}) {
  const id = useId();
  return <section className="kh-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    <p className="kh-predict"><strong>Predict:</strong> {prediction}</p>
    {children}
  </section>;
}
function Choice({
  label,
  value,
  onChange,
  options
}) {
  return <label className="kh-choice"><span>{label}</span>
    <select value={value} onChange={event => onChange(event.target.value)}>
      {options.map(([key, title]) => <option value={key} key={key}>{title}</option>)}
    </select>
  </label>;
}
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 'any',
  display = number(value)
}) {
  return <label className="kh-range"><span>{label}<output>{display}</output></span>
    <input type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}
function Table({
  caption,
  headings,
  rows
}) {
  return <div className="kh-table-scroll" role="region" aria-label={caption} tabIndex={0}>
    <table><caption>{caption}</caption><thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index}>{row.map((value, cell) => <td key={cell}>{value}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}
function PointMark({
  x,
  y,
  cluster,
  center = false
}) {
  const color = cluster === null ? '#c8cbce' : colors[cluster % colors.length];
  if (center) return <g stroke={color} strokeWidth="3"><path d={`M${x - 7},${y}h14 M${x},${y - 7}v14`} /><circle cx={x} cy={y} r="10" fill="none" strokeWidth="1" /></g>;
  if (cluster !== null && cluster % 3 === 1) return <rect x={x - 4} y={y - 4} width="8" height="8" fill={color} />;
  if (cluster !== null && cluster % 3 === 2) return <path d={`M${x},${y - 5}l5,9h-10z`} fill={color} />;
  return <circle cx={x} cy={y} r="4.5" fill={color} />;
}
function Scatter({
  points,
  centers = [],
  labels = null,
  connections = false,
  title,
  xLabel = 'x coordinate',
  yLabel = 'y coordinate'
}) {
  const all = [...points, ...centers];
  const minima = [0, 1].map(axis => Math.min(...all.map(point => point[axis])));
  const span = Math.max(1, ...[0, 1].map(axis => Math.max(...all.map(point => point[axis])) - minima[axis]));
  const domain = minima.map(minimum => minimum - 0.18 * span);
  const extent = span * 1.36;
  const project = point => [32 + 256 * (point[0] - domain[0]) / extent, 288 - 256 * (point[1] - domain[1]) / extent];
  const pointGroups = points.reduce((groups, point, index) => {
    const key = point.join(',');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(index);
    return groups;
  }, new Map());
  const centerGroups = centers.reduce((groups, center, index) => {
    const key = center.join(',');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(index);
    return groups;
  }, new Map());
  return <figure className="kh-scatter">
    <figcaption>{title}</figcaption>
    <div className="kh-square-plot">
      <svg viewBox="0 0 320 320" role="img" aria-label={`${title}. Equal coordinate units have equal lengths on both axes; exact values are in the accompanying table.`}>
        {[32, 160, 288].map(position => <g key={position} stroke="#2b3337" strokeWidth="1"><path d={`M32,${position}H288 M${position},32V288`} /></g>)}
        {connections && labels && points.map((point, index) => {
          const [x, y] = project(point),
            [cx, cy] = project(centers[labels[index]]);
          return <line key={index} x1={x} y1={y} x2={cx} y2={cy} stroke={colors[labels[index] % colors.length]} strokeOpacity=".45" strokeDasharray="3 3" />;
        })}
        {points.map((point, index) => <PointMark key={index} x={project(point)[0]} y={project(point)[1]} cluster={labels ? labels[index] : null} />)}
        {centers.map((center, index) => <PointMark key={index} x={project(center)[0]} y={project(center)[1]} cluster={index} center />)}
      </svg>
      {[...pointGroups.values()].map(indices => {
        const index = indices[0],
          point = points[index];
        return <span key={index} className={`kh-point-label ${index % 2 ? 'below' : ''} ${indices.length > 1 ? 'coincident' : ''}`} style={{
          left: `${project(point)[0] / 3.2}%`,
          top: `${project(point)[1] / 3.2}%`
        }}>{indices.length > 1 ? `${indices.length} rows here` : `P${index}${labels ? `·${labels[index]}` : ''}`}</span>;
      })}
      {[...centerGroups.values()].map(indices => {
        const index = indices[0],
          center = centers[index];
        return <span key={index} className="kh-center-label" style={{
          left: `${project(center)[0] / 3.2}%`,
          top: `${project(center)[1] / 3.2}%`,
          color: colors[index]
        }}>{indices.map(cluster => `C${cluster}`).join('/')}</span>;
      })}
      {[0, 0.5, 1].map(fraction => <span key={`x${fraction}`} className="kh-tick kh-x-tick" style={{
        left: `${10 + 80 * fraction}%`
      }}>{tick(domain[0] + fraction * extent)}</span>)}
      {[0, 0.5, 1].map(fraction => <span key={`y${fraction}`} className="kh-tick kh-y-tick" style={{
        top: `${90 - 80 * fraction}%`
      }}>{tick(domain[1] + fraction * extent)}</span>)}
    </div>
    <div className="kh-axis-caption"><span>horizontal: {xLabel}</span><span>vertical: {yLabel}</span></div>
    {labels && <p className="kh-legend">Point labels read P<em>row</em>·<em>cluster</em>. Circle = cluster 0; square = 1; triangle = 2; higher cluster numbers remain explicit. Crosses mark centers.</p>}
  </figure>;
}
export function LloydLab() {
  const [fixture, setFixture] = useState('six');
  const [initialization, setInitialization] = useState('separated');
  const [step, setStep] = useState(0);
  const points = fixture === 'six' ? clusteringPoints : rectanglePoints;
  const indices = initialization === 'duplicate' ? [0, 0] : initialization === 'nearby' ? [0, 1] : fixture === 'six' ? [0, 4] : [0, 2];
  const trace = lloydTrace(points, indices.map(index => points[index]));
  const state = trace[Math.min(step, trace.length - 1)];
  const reset = () => {
    setFixture('six');
    setInitialization('separated');
    setStep(0);
  };
  const phase = state.phase === 'initial' ? 'Choose centers; no assignment yet' : state.phase === 'assign' ? 'Assign to the nearest current center' : 'Move each center to its assigned mean';
  return <Investigation title="Watch assignment and movement change the same objective" prediction="Will moving a center change the labels immediately, or only on the next assignment?" onReset={reset}>
    <div className="kh-controls">
      <Choice label="Point configuration" value={fixture} onChange={value => {
        setFixture(value);
        setStep(0);
      }} options={[["six", 'Six-point running example'], ['rectangle', 'Four-point local-minimum counterexample']]} />
      <Choice label="Initial center rows" value={initialization} onChange={value => {
        setInitialization(value);
        setStep(0);
      }} options={[["separated", 'Separated centers'], ['nearby', 'Nearby centers'], ['duplicate', 'Two copies of P0']]} />
    </div>
    <div className="kh-step-controls"><button type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>Back</button><button type="button" disabled={step >= trace.length - 1} onClick={() => setStep(step + 1)}>Next phase</button><span>Phase {step + 1} of {trace.length}</span></div>
    <p className="kh-readout" aria-live="polite"><strong>{phase}.</strong> {state.sse === null ? 'The objective is shown after the first assignment.' : `SSE = ${number(state.sse)} using exactly the displayed labels and centers.`} {state.status === 'fixed assignment' ? 'Assignments are unchanged after the previous mean update: this run has reached a fixed point.' : ''}</p>
    <Scatter points={points} centers={state.centers} labels={state.labels} connections title="The dashed segments are the current residuals" />
    {state.empty.length > 0 && <p className="kh-note">Empty center(s): {state.empty.join(', ')}. This demonstration retains each empty center at its previous position. Exact assignment ties go to the lower center index; requesting two centers therefore need not yield two occupied clusters.</p>}
    <Table caption="Current labels, centers and squared-error contributions" headings={['point', 'coordinates', 'cluster', 'weighted squared error']} rows={points.map((point, index) => [`P${index}`, coordinate(point), state.labels?.[index] ?? 'not assigned', state.contributions ? number(state.contributions[index]) : '—'])} />
    <p>Centers: {state.centers.map((center, index) => `C${index}=${coordinate(center)}`).join('; ')}.</p>
    <details><summary>Read the objective history</summary><Table caption="SSE is attached to each complete state" headings={['phase', 'operation', 'SSE']} rows={trace.slice(0, step + 1).map((entry, index) => [index + 1, entry.phase, entry.sse === null ? 'not assigned' : number(entry.sse)])} /></details>
    <p className="kh-caption">Try the rectangle with separated centers, then nearby centers. The two fixed points have SSE 1 and 9 respectively; only this four-point example has its global minimum established by the lesson’s finite enumeration. A decreasing objective and unchanged assignments do not certify a globally optimal partition.</p>
  </Investigation>;
}
export function FeatureGeometryLab() {
  const [unit, setUnit] = useState(1);
  const [weight, setWeight] = useState(1);
  const state = featureGeometry(unit, weight);
  return <Investigation title="A change of units can change the clustering question" prediction="If only the vertical coordinate is reported in decimeters, should an unadjusted Euclidean distance preserve the original grouping?" onReset={() => {
    setUnit(1);
    setWeight(1);
  }}>
    <div className="kh-controls"><Choice label="Vertical measurement unit" value={unit} onChange={value => setUnit(Number(value))} options={[[1, 'Meters: multiply by 1'], [10, 'Decimeters: multiply by 10']]} /><Choice label="Weight on squared vertical differences" value={weight} onChange={value => setWeight(Number(value))} options={[[0.01, '0.01'], [0.25, '0.25'], [1, '1'], [4, '4']]} /></div>
    <p className="kh-readout" aria-live="polite">The plotted vertical coordinate is y × {unit} × √{weight} = y × {number(state.effective)}. Best SSE among the two declared initializations: <strong>{number(state.sse)}</strong>.</p>
    <Scatter points={state.points} centers={state.centers} labels={state.labels} connections title="Actual transformed geometry, with equal-length coordinate units" xLabel="original horizontal meters" yLabel="weighted numeric vertical coordinate" />
    <Table caption="The same records after the declared transformation" headings={['record', 'original meters', 'plotted coordinates', 'cluster']} rows={rectanglePoints.map((point, index) => [`P${index}`, coordinate(point), coordinate(state.points[index]), state.labels[index]])} />
    <p className="kh-caption">The two candidate fixed points have SSE {state.trials.map(trial => number(trial.sse)).join(' and ')} in this metric. Choosing decimeters and weight 0.01 exactly restores the original distances. Choosing a different weight deliberately changes what “similar” means. Standardizing everything is not automatically right: justify units and feature importance from the task, and estimate any learned scaling using the permitted reference data.</p>
  </Investigation>;
}
export function SeedingLab() {
  const [fixture, setFixture] = useState('six');
  const [first, setFirst] = useState(0);
  const [quantile, setQuantile] = useState(0.5);
  const points = fixture === 'six' ? clusteringPoints : [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]];
  const state = seedingDistribution(points, [first], quantile);
  return <Investigation title="Read a D² draw as probability, not a farthest-point rule" prediction="Can the next center be a closer row even when a farther row exists?" onReset={() => {
    setFixture('six');
    setFirst(0);
    setQuantile(0.5);
  }}>
    <div className="kh-controls"><Choice label="Seeding data" value={fixture} onChange={setFixture} options={[["six", 'Six distinct points'], ['duplicates', 'Six identical rows']]} /><Choice label="Fixed first center" value={first} onChange={value => setFirst(Number(value))} options={points.map((_, index) => [index, `P${index}`])} /></div>
    <Range label="Draw position in the cumulative probability line" value={quantile} onChange={setQuantile} min={0} max={0.999} step={0.001} />
    <div className="kh-probability-line" aria-label="Cumulative D squared probabilities">
      {state.rows.filter(row => row.probability > 0).map(row => <span key={row.index} title={`P${row.index}: ${number(row.probability)}`} style={{
        width: `${100 * row.probability}%`,
        background: colors[row.index]
      }} />)}
      {!state.stopped && <i style={{
        left: `${quantile * 100}%`
      }} aria-hidden="true" />}
    </div>
    <p className="kh-readout" aria-live="polite">{state.stopped ? 'Every D² is zero. All rows are already represented exactly; this illustration stops because extra centers cannot reduce the current error.' : `Draw ${number(quantile)} selects P${state.selected}. The interval lengths are D² / ${number(state.total)}; boundary draws use the interval to the right.`}</p>
    <Scatter points={points} centers={state.selected === null ? [points[first]] : [points[first], points[state.selected]]} title="C0 is fixed; C1 marks the center selected by this draw" />
    <Table caption="Conditional second-draw probabilities (rounded)" headings={['row', 'D²', 'probability', 'interval [start, end)']} rows={state.rows.map(row => [`P${row.index}`, number(row.distance), number(row.probability), `${number(row.start)}–${number(row.end)}`])} />
    <p className="kh-caption">The numbers shown are rounded; selection uses unrounded values. This is the conditional second step with a fixed first center. The original k-means++ method chooses the first row uniformly and repeats the weighted draw. If an implementation must return exactly k row indices when all distances vanish, it may choose an unused row deterministically, including duplicate coordinates. No positive-probability D² distribution exists in that zero-total case.</p>
  </Investigation>;
}
function Dendrogram({
  tree,
  cut,
  mode,
  height
}) {
  const maxHeight = Math.max(tree.root.height * 1.08, 1);
  const projectHeight = value => 224 - 196 * value / maxHeight;
  const positions = new Map(tree.leafOrder.map((id, index) => [id, 44 + 248 * index / Math.max(1, tree.points.length - 1)]));
  tree.merges.forEach(merge => positions.set(merge.id, (positions.get(merge.left) + positions.get(merge.right)) / 2));
  return <figure className="kh-dendrogram"><figcaption>{tree.linkage === 'ward' ? 'Ward height = √(2 × SSE increase)' : `${tree.linkage} Euclidean linkage height`}</figcaption>
    <div className="kh-tree-plot"><svg viewBox="0 0 320 256" role="img" aria-label="Dendrogram: leaf row IDs align below their branches; exact merge heights appear in the table.">
      {[0, 0.5, 1].map(fraction => <path key={fraction} d={`M32,${projectHeight(fraction * maxHeight)}H304`} stroke="#2b3337" />)}
      {tree.merges.map(merge => {
          const left = tree.nodes[merge.left],
            right = tree.nodes[merge.right];
          const labels = merge.members.map(index => cut.labels[index]);
          const same = labels.every(label => label === labels[0]);
          return <path key={merge.id} d={`M${positions.get(left.id)},${projectHeight(left.height)}V${projectHeight(merge.height)}H${positions.get(right.id)}V${projectHeight(right.height)}`} fill="none" stroke={same ? colors[labels[0]] : '#bdc2c4'} strokeWidth="2" />;
        })}
      {mode === 'height' && <path d={`M32,${projectHeight(height)}H304`} stroke="#e7b94a" strokeWidth="2" strokeDasharray="5 4" />}
    </svg>
      {tree.leafOrder.map(id => <span key={id} className="kh-leaf" style={{
        left: `${positions.get(id) / 3.2}%`,
        color: colors[cut.labels[id]]
      }}>P{id}·{cut.labels[id]}</span>)}
      {[0, 0.5, 1].map(fraction => <span key={fraction} className="kh-tree-tick" style={{
        top: `${projectHeight(fraction * maxHeight) / 2.56}%`
      }}>{tick(fraction * maxHeight)}</span>)}
    </div>
  </figure>;
}
export function HierarchyLab() {
  const [linkage, setLinkage] = useState('ward');
  const [mode, setMode] = useState('height');
  const [height, setHeight] = useState(Math.SQRT1_2);
  const [count, setCount] = useState(4);
  const tree = hierarchyTrace(clusteringPoints, linkage);
  const cut = cutHierarchy(tree, mode, mode === 'height' ? height : count);
  return <Investigation title="A merge count and a horizontal cut need not mean the same partition" prediction="Two disjoint pairs merge at the same height. Can a horizontal cut leave exactly one of those two merges completed?" onReset={() => {
    setLinkage('ward');
    setMode('height');
    setHeight(Math.SQRT1_2);
    setCount(4);
  }}>
    <div className="kh-controls"><Choice label="Linkage definition" value={linkage} onChange={value => {
        setLinkage(value);
        setHeight(Math.SQRT1_2);
      }} options={['single', 'complete', 'average', 'ward'].map(value => [value, value])} /><Choice label="Partition rule" value={mode} onChange={setMode} options={[["height", 'All merges at or below a height'], ['count', 'Stop after exactly n − k merges']]} /></div>
    {mode === 'height' ? <Range label="Cut height" value={height} onChange={setHeight} min={0} max={tree.root.height * 1.05} /> : <Range label="Requested cluster count k" value={count} onChange={setCount} min={1} max={6} step={1} />}
    <button type="button" className="kh-tie-preset" onClick={() => {
      setMode('height');
      setHeight(Math.SQRT1_2);
    }}>Set the tied height √0.5 exactly</button>
    <p className="kh-readout" aria-live="polite"><strong>{cut.count} clusters</strong> after {cut.mergesUsed} merges. Partition SSE = {number(cut.sse)} in squared coordinate units. {mode === 'count' && count === 4 ? 'This four-cluster result uses the deterministic merge prefix; no single horizontal height cut produces it on this fixture.' : ''}</p>
    <Dendrogram tree={tree} cut={cut} mode={mode} height={height} />
    <Scatter points={clusteringPoints} labels={cut.labels} title="The same cut, back in the original coordinates" />
    <Table caption="Every actual merge and its two member sets" headings={['merge', 'left rows', 'right rows', 'height', 'SSE increase Δ']} rows={tree.merges.map((merge, index) => [index + 1, tree.nodes[merge.left].members.map(id => `P${id}`).join(', '), tree.nodes[merge.right].members.map(id => `P${id}`).join(', '), number(merge.height), number(merge.delta)])} />
    <p className="kh-caption">Single, complete and average use the minimum, maximum or average of all cross-cluster Euclidean point distances. Ward chooses the smallest increase Δ in within-cluster SSE; its plotted SciPy convention is √(2Δ), not plain center distance. All four heights have coordinate units; Δ has squared coordinate units. Exact pair-cost ties use increasing cluster IDs. At a height cut, all equal-height merges are included together; moving between 5 and 3 clusters therefore skips 4 here. The full hierarchy and tie convention, not horizontal spacing of leaves, determine the grouping.</p>
  </Investigation>;
}
function PixelImage({
  pixels,
  title
}) {
  return <figure className="kh-pixel-figure"><figcaption>{title}</figcaption><div className="kh-pixels" role="img" aria-label={`${title}: a constructed twelve by eight pixel image; its palette and counts are tabulated below.`}>{pixels.map((pixel, index) => <span key={index} style={{
        background: `rgb(${pixel.join(',')})`
      }} />)}</div></figure>;
}
export function PaletteLab() {
  const [count, setCount] = useState(3);
  const state = quantizePalette(count);
  return <Investigation title="Let repeated pixels vote for a small palette" prediction="Should a color appearing 30 times count the same as a color appearing only once?" onReset={() => setCount(3)}>
    <Range label="Requested palette size" value={count} onChange={setCount} min={1} max={6} step={1} />
    <div className="kh-image-pair"><PixelImage pixels={palettePixels} title="Original: six exact RGB colors" /><PixelImage pixels={state.reconstructed} title="Reconstructed with rounded palette entries" /></div>
    <p className="kh-readout" aria-live="polite">Weighted floating-center SSE: <strong>{number(state.sse)}</strong>. Actual displayed integer-RGB SSE: <strong>{number(state.displayedSse)}</strong>; mean squared channel error = {number(state.meanSquaredChannelError)} over 96 × 3 channels.</p>
    <div className="kh-palette">{state.roundedCenters.map((color, index) => <div key={index}><span style={{
          background: `rgb(${color.join(',')})`
        }} /><strong>C{index}</strong><code>{color.join(', ')}</code></div>)}</div>
    <Table caption="Unique colors keep their pixel-frequency weights" headings={['RGB', 'pixel count', 'assigned palette entry']} rows={state.colors.map((color, index) => [color.join(', '), state.counts[index], `C${state.labels[index]}`])} />
    <p className="kh-caption">These are constructed pixels, not a photograph or benchmark. The first center is the most frequent color; later centers are deterministically farthest from the chosen set, followed by weighted Lloyd updates. This is one declared initialization, not a global optimum guarantee or a D² sample. Squared distance in numeric sRGB channels is the model being minimized, not perceptual color difference. Rounding can change the displayed error. A palette with k entries alone does not establish a file-size saving: indices, palette storage and an actual codec also matter.</p>
  </Investigation>;
}
