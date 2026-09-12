import { useId, useLayoutEffect, useRef, useState } from 'react';
import { clusteringPoints, rectanglePoints, hierarchyFixtures, lloydTrace, featureGeometry, seedingDistribution, seedingFrequencies, hierarchyTrace, cutHierarchy, paletteImages, paletteLimit, quantizePalette } from '../../data/k-means-hierarchical-models';
import './k-means-hierarchical-labs.css';
const colors = ['#e7b94a', '#8eb9a5', '#91aecf', '#da9c86', '#b8a0cd', '#d0c7ae', '#c7d08a', '#9fb7d8', '#d6a3b8'];
const number = value => Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
const tick = value => String(Number(value.toPrecision(3)));
const coordinate = point => `(${point.map(number).join(', ')})`;
const rowList = indices => indices.map(index => `P${index}`).join(', ');
function Investigation({
  title,
  question,
  children,
  onReset
}) {
  const id = useId();
  return <section className="kh-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="kh-predict">{question}</p>}
    {children}
  </section>;
}
/** Predict → manipulate → observe. The learner records a prediction; feedback
 * compares it with the value the model actually produced. */
function Prediction({
  prompt,
  options,
  value,
  onChange,
  answer,
  revealed,
  onCheck,
  disabled = false,
  explanation
}) {
  const id = useId();
  const answerLabel = options.find(([key]) => key === answer)?.[1] ?? answer;
  return <div className="kh-prediction">
    <label htmlFor={id}><strong>Predict first:</strong> {prompt}</label>
    <div className="kh-prediction-row">
      <select id={id} value={value} disabled={disabled} onChange={event => onChange(event.target.value)}>
        <option value="">Choose a prediction</option>
        {options.map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </select>
      {onCheck && <button type="button" disabled={!value || revealed} onClick={onCheck}>Check</button>}
    </div>
    {revealed && value && <p className={`kh-feedback ${value === answer ? 'is-match' : 'is-miss'}`} role="status">
      {value === answer ? `Your prediction matches: ${answerLabel}.` : `Not this time. The model gives: ${answerLabel}.`} {explanation}
    </p>}
  </div>;
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
  const controlId = useId();
  return <label className="kh-range" htmlFor={controlId}><span>{label}<output htmlFor={controlId}>{display}</output></span>
    <input id={controlId} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}
function Table({
  caption,
  headings,
  rows,
  highlight = () => false
}) {
  return <div className="kh-table-scroll" role="region" aria-label={caption} tabIndex={0}>
    <table><caption>{caption}</caption><thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index} className={highlight(index) ? 'is-best' : undefined}>{row.map((value, cell) => <td key={cell}>{value}</td>)}</tr>)}</tbody>
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

// This moves text only. The shared geometric projection below still determines
// every data marker, center and residual endpoint without any displacement.
function positionScatterLabels(entries, markers, obstacles, width) {
  const placed = [];
  const overlaps = (left, right) => left.x < right.x + right.width && left.x + left.width > right.x
    && left.y < right.y + right.height && left.y + left.height > right.y;
  for (const entry of entries) {
    const textWidth = entry.text.length * 7.6 + 4;
    const height = 17;
    const candidates = [];
    for (const gap of [8, 18, 30, 44, 60, 78]) {
      for (const [horizontal, vertical] of [[entry.horizontal, entry.vertical], [entry.horizontal, -entry.vertical], [-entry.horizontal, entry.vertical], [-entry.horizontal, -entry.vertical]]) {
        candidates.push({ x: horizontal < 0 ? entry.x - gap - textWidth : entry.x + gap, y: vertical < 0 ? entry.y - gap - height : entry.y + gap, width: textWidth, height, gap });
      }
    }
    const scored = candidates.map((candidate, order) => ({
      ...candidate,
      score: (candidate.x < 2 || candidate.y < 2 || candidate.x + candidate.width > width - 2 || candidate.y + height > width - 2 ? 100000 : 0)
        + [...markers, ...obstacles, ...placed].filter(other => overlaps(candidate, other)).length * 1000 + order,
    }));
    scored.sort((left, right) => left.score - right.score);
    const chosen = scored[0];
    placed.push({ ...chosen, ...entry, x: chosen.x, y: chosen.y, width: textWidth, height,
      anchorX: entry.x, anchorY: entry.y,
      lineX: Math.max(chosen.x, Math.min(entry.x, chosen.x + textWidth)),
      lineY: Math.max(chosen.y, Math.min(entry.y, chosen.y + height)),
    });
  }
  return placed;
}

/** The perpendicular bisector of two distinct centers, clipped to the visible
 * square in data coordinates. Every point on it is equally far from both
 * centers, so it is the nearest-center boundary for k = 2. */
function bisectorSegment(first, second, low, high) {
  if (first[0] === second[0] && first[1] === second[1]) return null;
  const middle = [(first[0] + second[0]) / 2, (first[1] + second[1]) / 2];
  const direction = [-(second[1] - first[1]), second[0] - first[0]];
  const parameters = [];
  [0, 1].forEach(axis => {
    if (direction[axis] === 0) return;
    for (const bound of [low[axis], high[axis]]) {
      const t = (bound - middle[axis]) / direction[axis];
      const other = middle[1 - axis] + t * direction[1 - axis];
      if (other >= low[1 - axis] - 1e-9 && other <= high[1 - axis] + 1e-9) parameters.push(t);
    }
  });
  if (parameters.length < 2) return null;
  const [start, end] = [Math.min(...parameters), Math.max(...parameters)];
  return [[middle[0] + start * direction[0], middle[1] + start * direction[1]], [middle[0] + end * direction[0], middle[1] + end * direction[1]]];
}

function Scatter({
  points,
  centers = [],
  labels = null,
  connections = false,
  boundary = false,
  title,
  xLabel = 'x coordinate',
  yLabel = 'y coordinate'
}) {
  const plotRef = useRef(null);
  const [plotWidth, setPlotWidth] = useState(320);
  useLayoutEffect(() => {
    const update = () => {
      const width = plotRef.current?.getBoundingClientRect().width;
      if (width > 0) setPlotWidth(width);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(plotRef.current);
    return () => observer.disconnect();
  }, []);
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
  const scale = plotWidth / 320;
  const horizontalMiddle = (Math.min(...points.map(point => point[0])) + Math.max(...points.map(point => point[0]))) / 2;
  const labelEntries = [...pointGroups.values()].map(indices => {
    const index = indices[0], point = points[index], position = project(point);
    return { kind: 'point', id: index, text: indices.length > 1 ? `${indices.length} rows here` : `P${index}${labels ? `·${labels[index]}` : ''}`,
      x: position[0] * scale, y: position[1] * scale,
      horizontal: point[0] <= horizontalMiddle ? -1 : 1, vertical: index % 2 === 0 ? 1 : -1 };
  });
  labelEntries.push(...[...centerGroups.values()].map(indices => {
    const index = indices[0], center = centers[index], position = project(center);
    return { kind: 'center', id: index, text: indices.map(cluster => `C${cluster}`).join('/'),
      x: position[0] * scale, y: position[1] * scale,
      horizontal: center[0] <= horizontalMiddle ? 1 : -1, vertical: index % 2 === 0 ? 1 : -1 };
  }));
  const markers = all.map((point, index) => {
    const [x, y] = project(point).map(value => value * scale);
    const radius = (index < points.length ? 5 : 11) * scale + 3;
    return { x: x - radius, y: y - radius, width: 2 * radius, height: 2 * radius };
  });
  const tickObstacles = [0, 0.5, 1].flatMap(fraction => [
    { x: 0, y: (0.9 - 0.8 * fraction) * plotWidth - 8, width: tick(domain[1] + fraction * extent).length * 7.6, height: 17 },
    { x: (0.1 + 0.8 * fraction) * plotWidth - 24, y: 0.92 * plotWidth, width: 48, height: 17 },
  ]);
  const positionedLabels = positionScatterLabels(labelEntries, markers, tickObstacles, plotWidth);
  const bisector = boundary && centers.length === 2 ? bisectorSegment(centers[0], centers[1], domain, domain.map(value => value + extent)) : null;
  return <figure className="kh-scatter">
    <figcaption>{title}</figcaption>
    <div className="kh-square-plot" ref={plotRef}>
      <svg viewBox="0 0 320 320" role="img" aria-label={`${title}. Equal coordinate units have equal lengths on both axes; exact values are in the accompanying table.`}>
        {[32, 160, 288].map(position => <g key={position} stroke="#2b3337" strokeWidth="1"><path d={`M32,${position}H288 M${position},32V288`} /></g>)}
        {bisector && <line className="kh-boundary" x1={project(bisector[0])[0]} y1={project(bisector[0])[1]} x2={project(bisector[1])[0]} y2={project(bisector[1])[1]} />}
        {positionedLabels.map(label => <line key={`${label.kind}-${label.id}`} className="kh-label-leader" x1={label.anchorX / scale} y1={label.anchorY / scale} x2={label.lineX / scale} y2={label.lineY / scale} vectorEffect="non-scaling-stroke" />)}
        {connections && labels && points.map((point, index) => {
          const [x, y] = project(point),
            [cx, cy] = project(centers[labels[index]]);
          return <line key={index} x1={x} y1={y} x2={cx} y2={cy} stroke={colors[labels[index] % colors.length]} strokeOpacity=".45" strokeDasharray="3 3" />;
        })}
        {points.map((point, index) => <PointMark key={index} x={project(point)[0]} y={project(point)[1]} cluster={labels ? labels[index] : null} />)}
        {centers.map((center, index) => <PointMark key={index} x={project(center)[0]} y={project(center)[1]} cluster={index} center />)}
      </svg>
      {positionedLabels.map(label => <span key={`${label.kind}-${label.id}`} className={label.kind === 'center' ? 'kh-center-label' : 'kh-point-label'} style={{ left: label.x, top: label.y, color: label.kind === 'center' ? colors[label.id] : undefined }}>{label.text}</span>)}
      {[0, 0.5, 1].map(fraction => <span key={`x${fraction}`} className="kh-tick kh-x-tick" style={{
        left: `${10 + 80 * fraction}%`
      }}>{tick(domain[0] + fraction * extent)}</span>)}
      {[0, 0.5, 1].map(fraction => <span key={`y${fraction}`} className="kh-tick kh-y-tick" style={{
        top: `${90 - 80 * fraction}%`
      }}>{tick(domain[1] + fraction * extent)}</span>)}
    </div>
    <div className="kh-axis-caption"><span>horizontal: {xLabel}</span><span>vertical: {yLabel}</span></div>
    <p className="kh-label-explanation">Fine gray lines connect displaced labels to unchanged markers.{bisector ? ' The solid pale line is the equal-distance boundary between the two centers: every point on one side is nearer to that side’s center.' : ''}</p>
    {labels && <p className="kh-legend">Point labels read P<em>row</em>·<em>cluster</em>. Circle = cluster 0; square = 1; triangle = 2; higher cluster numbers remain explicit. Crosses mark centers; dashed segments are residuals.</p>}
  </figure>;
}
const updateBucket = trace => {
  const updates = trace.filter(state => state.phase === 'move').length;
  return updates >= 3 ? '3+' : String(updates);
};
export function LloydLab() {
  const [fixture, setFixture] = useState('six');
  const [rows, setRows] = useState([0, 4]);
  const [step, setStep] = useState(0);
  const [prediction, setPrediction] = useState('');
  const points = fixture === 'six' ? clusteringPoints : rectanglePoints;
  const trace = lloydTrace(points, rows.map(index => points[index]));
  const state = trace[Math.min(step, trace.length - 1)];
  const finished = step >= trace.length - 1;
  const restart = (nextFixture, nextRows) => {
    setFixture(nextFixture);
    setRows(nextRows);
    setStep(0);
    setPrediction('');
  };
  const chooseRow = (position, value) => {
    const next = [...rows];
    next[position] = Number(value);
    restart(fixture, next);
  };
  const phase = state.phase === 'initial' ? 'Choose centers; no assignment yet' : state.phase === 'assign' ? 'Assign to the nearest current center' : 'Move each center to its assigned mean';
  return <Investigation title="Watch assignment and movement change the same objective" question="Pick any two rows as the starting centers. Moving a center never changes a label by itself; only the next assignment does. Can you find a start on the rectangle that settles at SSE 9 instead of 1?" onReset={() => restart('six', [0, 4])}>
    <div className="kh-controls">
      <Choice label="Point configuration" value={fixture} onChange={value => restart(value, value === 'six' ? [0, 4] : [0, 2])} options={[["six", 'Six-point running example'], ['rectangle', 'Four-point local-minimum counterexample']]} />
      <Choice label="Center 0 starts at row" value={rows[0]} onChange={value => chooseRow(0, value)} options={points.map((_, index) => [index, `P${index} ${coordinate(points[index])}`])} />
      <Choice label="Center 1 starts at row" value={rows[1]} onChange={value => chooseRow(1, value)} options={points.map((_, index) => [index, `P${index} ${coordinate(points[index])}`])} />
    </div>
    <Prediction prompt="From these starting rows, how many mean updates happen before the labels stop changing?" options={[["1", 'One update'], ['2', 'Two updates'], ['3+', 'Three or more']]} value={prediction} onChange={setPrediction} answer={updateBucket(trace)} revealed={finished} disabled={step > 0} explanation="Count the “move” phases in the objective history below." />
    <div className="kh-step-controls"><button type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>Back</button><button type="button" disabled={finished} onClick={() => setStep(step + 1)}>Next phase</button><button type="button" disabled={finished} onClick={() => setStep(trace.length - 1)}>Run to fixed point</button><span>Phase {step + 1} of {trace.length}</span></div>
    <p className="kh-readout" aria-live="polite"><strong>{phase}.</strong> {state.sse === null ? 'The objective is shown after the first assignment.' : `SSE = ${number(state.sse)} using exactly the displayed labels and centers.`} {state.status === 'fixed assignment' ? 'Assignments are unchanged after the previous mean update: this run has reached a fixed point.' : ''}</p>
    <Scatter points={points} centers={state.centers} labels={state.labels} connections boundary title={state.labels === null ? 'Choose the first assignment to connect points to their centers' : 'The dashed segments are the current residuals'} />
    {state.empty.length > 0 && <p className="kh-note">Empty center(s): {state.empty.join(', ')}. This demonstration retains each empty center at its previous position. Exact assignment ties go to the lower center index; requesting two centers therefore need not yield two occupied clusters.</p>}
    <Table caption="Current labels, centers and squared-error contributions" headings={['point', 'coordinates', 'cluster', 'weighted squared error']} rows={points.map((point, index) => [`P${index}`, coordinate(point), state.labels?.[index] ?? 'not assigned', state.contributions ? number(state.contributions[index]) : '—'])} />
    <p>Centers: {state.centers.map((center, index) => `C${index}=${coordinate(center)}`).join('; ')}.</p>
    <details><summary>Read the objective history</summary><Table caption="SSE is attached to each complete state" headings={['phase', 'operation', 'SSE']} rows={trace.slice(0, step + 1).map((entry, index) => [index + 1, entry.phase, entry.sse === null ? 'not assigned' : number(entry.sse)])} /></details>
    <p className="kh-caption">Suggested investigations: start the rectangle at P0 and P2, then at P0 and P1. Both runs stop with unchanged labels, at SSE 1 and SSE 9. Start the six points at two copies of the same row to see an empty center. Watch the boundary line: it moves only when a center moves, and labels change only when a point crosses it at the next assignment.</p>
  </Investigation>;
}
const partitionName = labels => labels.join('') === '0011' ? 'left-right' : labels.join('') === '0101' ? 'bottom-top' : 'other';
export function FeatureGeometryLab() {
  const [unit, setUnit] = useState(1);
  const [weight, setWeight] = useState(1);
  const [prediction, setPrediction] = useState('');
  const [checked, setChecked] = useState(false);
  const state = featureGeometry(unit, weight);
  const change = setter => value => {
    setter(Number(value));
    setChecked(false);
  };
  return <Investigation title="A change of units can change the clustering question" question="Only the vertical coordinate is reported in a different unit. Does an unadjusted Euclidean distance still prefer the same two groups?" onReset={() => {
    setUnit(1);
    setWeight(1);
    setPrediction('');
    setChecked(false);
  }}>
    <div className="kh-controls"><Choice label="Vertical measurement unit" value={unit} onChange={change(setUnit)} options={[[1, 'Meters: multiply by 1'], [10, 'Decimeters: multiply by 10']]} /><Choice label="Weight on squared vertical differences" value={weight} onChange={change(setWeight)} options={[[0.01, '0.01'], [0.25, '0.25'], [1, '1'], [4, '4']]} /></div>
    <Prediction prompt="Under the selected unit and weight, which two-group split has the lowest squared error?" options={[["left-right", 'Left and right pairs: {P0, P1} and {P2, P3}'], ['bottom-top', 'Bottom and top pairs: {P0, P2} and {P1, P3}'], ['other', 'Some other split']]} value={prediction} onChange={setPrediction} answer={partitionName(state.labels)} revealed={checked} onCheck={() => setChecked(true)} explanation="The table lists every possible two-group split with its exact error." />
    <p className="kh-readout" aria-live="polite">The plotted vertical coordinate is y × {unit} × √{weight} = y × {number(state.effective)}. Exact optimum over all seven two-group splits: <strong>SSE {number(state.sse)}</strong>.</p>
    <Scatter points={state.points} centers={state.centers} labels={state.labels} connections boundary title="Actual transformed geometry, with equal-length coordinate units" xLabel="original horizontal meters" yLabel="weighted numeric vertical coordinate" />
    <Table caption="Every two-group split of the four records, in this metric" headings={['group A', 'group B', 'SSE']} rows={state.partitions.map(partition => [rowList(partition.groups[0]), rowList(partition.groups[1]), number(partition.sse)])} highlight={index => index === 0} />
    <Table caption="The same records after the declared transformation" headings={['record', 'original meters', 'plotted coordinates', 'cluster']} rows={rectanglePoints.map((point, index) => [`P${index}`, coordinate(point), coordinate(state.points[index]), state.labels[index]])} />
    <p className="kh-caption">Four points allow only seven two-group splits, so the winner here is exact rather than the result of a lucky start. Choosing decimeters and weight 0.01 exactly restores the original distances. Choosing a different weight deliberately changes what “similar” means. Standardizing everything is not automatically right: justify units and feature importance from the task, and estimate any learned scaling on the permitted reference data.</p>
  </Investigation>;
}
export function SeedingLab() {
  const [fixture, setFixture] = useState('six');
  const [selected, setSelected] = useState([0]);
  const [quantile, setQuantile] = useState(0.5);
  const [prediction, setPrediction] = useState('');
  const [checked, setChecked] = useState(false);
  const points = fixture === 'six' ? clusteringPoints : [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]];
  const state = seedingDistribution(points, selected, quantile);
  const frequencies = seedingFrequencies(points, selected, 200, 1);
  const farthestAnswer = frequencies.farthestProbability >= 0.999 ? 'always' : frequencies.farthestProbability >= 0.5 ? 'more' : 'less';
  const restart = (nextFixture, first) => {
    setFixture(nextFixture);
    setSelected([first]);
    setQuantile(0.5);
    setPrediction('');
    setChecked(false);
  };
  return <Investigation title="Read a D² draw as probability, not a farthest-point rule" question="The farthest row from the chosen centers is the most likely next center. Is it certain? Draw several centers in sequence and watch the remaining probability mass move." onReset={() => restart('six', 0)}>
    <div className="kh-controls"><Choice label="Seeding data" value={fixture} onChange={value => restart(value, 0)} options={[["six", 'Six distinct points'], ['duplicates', 'Six identical rows']]} /><Choice label="Fixed first center" value={selected[0]} onChange={value => restart(fixture, Number(value))} options={points.map((_, index) => [index, `P${index}`])} /></div>
    <Prediction prompt={`With ${rowList(selected)} already chosen, how often will 200 repeated draws pick the farthest remaining row${frequencies.farthest === null ? '' : ` (P${frequencies.farthest})`}?`} options={[["always", 'Every time'], ['more', 'More than half the time'], ['less', 'Less than half the time']]} value={prediction} onChange={setPrediction} answer={farthestAnswer} revealed={checked} onCheck={() => setChecked(true)} explanation={frequencies.stopped ? 'No draw is possible once every row is already a center.' : `Its exact probability is ${number(100 * frequencies.farthestProbability)}%; the bars below show the seeded 200-draw frequencies.`} />
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
    {!state.stopped && <ul className="kh-probability-key" aria-label="Point colors and next-center probabilities">
      {state.rows.filter(row => row.probability > 0).map(row => {
        const percent = row.probability * 100;
        return <li key={row.index} className={row.index === state.selected ? 'is-selected' : ''}>
          <span className="kh-probability-swatch" style={{ background: colors[row.index] }} aria-hidden="true" />
          <span>P{row.index}: {percent < 0.0001 ? percent.toExponential(2) : number(percent)}%</span>
          {row.index === state.selected && <strong>← selected</strong>}
        </li>;
      })}
    </ul>}
    <p className="kh-readout" aria-live="polite">{state.stopped ? 'Every D² is zero. All rows are already represented exactly; this illustration stops because extra centers cannot reduce the current error.' : `Draw ${number(quantile)} selects P${state.selected} as center C${selected.length}. The interval lengths are D² / ${number(state.total)}; boundary draws use the interval to the right.`}</p>
    <div className="kh-step-controls">
      <button type="button" disabled={state.stopped || selected.length >= points.length} onClick={() => {
        setSelected([...selected, state.selected]);
        setChecked(false);
        setPrediction('');
      }}>Accept this draw and pick the next center</button>
      <button type="button" disabled={selected.length <= 1} onClick={() => {
        setSelected(selected.slice(0, -1));
        setChecked(false);
        setPrediction('');
      }}>Undo last draw</button>
      <span>Centers so far: {rowList(selected)}</span>
    </div>
    <Scatter points={points} centers={state.selected === null ? selected.map(index => points[index]) : [...selected.map(index => points[index]), points[state.selected]]} title={state.stopped ? `C0 already represents every row; no further center is drawn` : `Chosen centers are fixed; C${selected.length} marks the row selected by this draw`} />
    {!frequencies.stopped && <figure className="kh-frequency">
      <figcaption>Exact probability versus 200 seeded repeated draws (seed 1). Uniform random seeding would give every remaining row {number(100 * frequencies.uniformProbability)}%.</figcaption>
      <ul aria-label="Per-row probability and observed frequency">
        {state.rows.map(row => <li key={row.index}>
          <span className="kh-frequency-label">P{row.index}</span>
          <span className="kh-frequency-bars">
            <span className="kh-frequency-bar is-exact" style={{ width: `${100 * row.probability}%`, background: colors[row.index] }} title={`exact ${number(100 * row.probability)}%`} />
            <span className="kh-frequency-bar is-observed" style={{ width: `${100 * frequencies.frequencies[row.index]}%` }} title={`observed ${frequencies.counts[row.index]} of 200`} />
            <span className="kh-frequency-uniform" style={{ left: `${100 * frequencies.uniformProbability}%` }} aria-hidden="true" />
          </span>
          <span className="kh-frequency-value">{number(100 * row.probability)}% · {frequencies.counts[row.index]}/200</span>
        </li>)}
      </ul>
      <p className="kh-label-explanation">Colored bar: exact D² probability. Gray bar: how often this row was picked in 200 seeded draws. The thin marker is the uniform-seeding probability.</p>
    </figure>}
    <Table caption="Conditional next-draw probabilities (rounded)" headings={['row', 'D²', 'probability', 'interval [start, end)']} rows={state.rows.map(row => [`P${row.index}`, number(row.distance), number(row.probability), `${number(row.start)}–${number(row.end)}`])} />
    <p className="kh-caption">The numbers shown are rounded; selection uses unrounded values. The original k-means++ method chooses the first row uniformly and repeats the weighted draw until k centers exist. After each accepted draw the D² values are recomputed against the nearest chosen center, so rows near any center lose almost all of their probability. If an implementation must return exactly k row indices when all distances vanish, it may choose an unused row deterministically, including duplicate coordinates; no positive-probability D² distribution exists in that zero-total case.</p>
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
      {tree.merges.map((merge, index) => {
          const left = tree.nodes[merge.left],
            right = tree.nodes[merge.right];
          const labels = merge.members.map(member => cut.labels[member]);
          const same = labels.every(label => label === labels[0]);
          const pending = index >= cut.mergesUsed && mode === 'count';
          return <path key={merge.id} d={`M${positions.get(left.id)},${projectHeight(left.height)}V${projectHeight(merge.height)}H${positions.get(right.id)}V${projectHeight(right.height)}`} fill="none" stroke={same ? colors[labels[0] % colors.length] : '#bdc2c4'} strokeWidth="2" strokeDasharray={pending ? '3 3' : undefined} strokeOpacity={pending ? 0.55 : 1} />;
        })}
      {mode === 'height' && <path d={`M32,${projectHeight(height)}H304`} stroke="#e7b94a" strokeWidth="2" strokeDasharray="5 4" />}
    </svg>
      {tree.leafOrder.map(id => <span key={id} className="kh-leaf" style={{
        left: `${positions.get(id) / 3.2}%`,
        color: colors[cut.labels[id] % colors.length]
      }}>P{id}·{cut.labels[id]}</span>)}
      {[0, 0.5, 1].map(fraction => <span key={fraction} className="kh-tree-tick" style={{
        top: `${projectHeight(fraction * maxHeight) / 2.56}%`
      }}>{tick(fraction * maxHeight)}</span>)}
    </div>
  </figure>;
}
const tiedHeight = tree => {
  const heights = tree.merges.map(merge => merge.height);
  return heights.find((value, index) => heights.some((other, otherIndex) => otherIndex !== index && Math.abs(other - value) < 1e-9)) ?? null;
};
const chainIntact = fixture => fixture !== 'chain' ? null : ['single', 'complete', 'average', 'ward'].filter(linkage => cutHierarchy(hierarchyTrace(hierarchyFixtures.chain, linkage), 'count', 2).groups.some(group => [...group].sort((a, b) => a - b).join(',') === '0,1,2,3,4,5'));
const chainAnswer = intact => intact.length === 4 ? 'all' : intact.length === 3 && !intact.includes('complete') ? 'not-complete' : intact.length === 1 && intact[0] === 'single' ? 'single' : 'other';
export function HierarchyLab() {
  const [fixture, setFixture] = useState('six');
  const [linkage, setLinkage] = useState('ward');
  const [mode, setMode] = useState('height');
  const [height, setHeight] = useState(Math.SQRT1_2);
  const [count, setCount] = useState(4);
  const [prediction, setPrediction] = useState('');
  const [checked, setChecked] = useState(false);
  const points = hierarchyFixtures[fixture];
  const tree = hierarchyTrace(points, linkage);
  const cut = cutHierarchy(tree, mode, mode === 'height' ? height : count);
  const tied = tiedHeight(tree);
  const intact = chainIntact(fixture);
  const restart = nextFixture => {
    setFixture(nextFixture);
    setLinkage(nextFixture === 'six' ? 'ward' : 'single');
    setMode(nextFixture === 'six' ? 'height' : 'count');
    setHeight(Math.SQRT1_2);
    setCount(nextFixture === 'six' ? 4 : 2);
    setPrediction('');
    setChecked(false);
  };
  const stepMerges = delta => {
    const merged = mode === 'count' ? points.length - count : cut.mergesUsed;
    const next = Math.max(0, Math.min(points.length - 1, merged + delta));
    setMode('count');
    setCount(points.length - next);
  };
  return <Investigation title="Build the tree one merge at a time, then cut it two different ways" question={fixture === 'six' ? 'Two disjoint pairs merge at the same height. Can a horizontal cut leave exactly one of those two merges completed?' : 'Six points form an evenly spaced chain; three more form a compact triple. Which linkage rules keep the chain together at k = 2, and which break it?'} onReset={() => restart('six')}>
    <div className="kh-controls">
      <Choice label="Point configuration" value={fixture} onChange={restart} options={[["six", 'Six-point running example'], ['chain', 'Chain of six plus a compact triple']]} />
      <Choice label="Linkage definition" value={linkage} onChange={value => {
        setLinkage(value);
        setHeight(Math.SQRT1_2);
        setChecked(false);
      }} options={['single', 'complete', 'average', 'ward'].map(value => [value, value])} />
      <Choice label="Partition rule" value={mode} onChange={value => {
        setMode(value);
        setChecked(false);
      }} options={[["height", 'All merges at or below a height'], ['count', 'Stop after exactly n − k merges']]} />
    </div>
    {fixture === 'six' ? <Prediction prompt="At the tied height, how many clusters does a horizontal cut leave?" options={[["5", 'Five clusters'], ['4', 'Four clusters'], ['3', 'Three clusters']]} value={prediction} onChange={setPrediction} answer={String(cutHierarchy(hierarchyTrace(points, 'ward'), 'height', Math.SQRT1_2).count)} revealed={checked} onCheck={() => setChecked(true)} explanation="Press the tied-height button under Ward linkage to see it." />
      : <Prediction prompt="Cutting each linkage’s tree to k = 2, which rules keep all six chain points in one cluster?" options={[["all", 'All four linkages'], ['not-complete', 'All except complete'], ['single', 'Only single'], ['other', 'Another combination']]} value={prediction} onChange={setPrediction} answer={chainAnswer(intact)} revealed={checked} onCheck={() => setChecked(true)} explanation={`Chain intact under: ${intact.join(', ')}. Switch the linkage with k = 2 to compare.`} />}
    {mode === 'height' ? <Range label="Cut height" value={height} onChange={setHeight} min={0} max={tree.root.height * 1.05} /> : <Range label="Requested cluster count k" value={count} onChange={setCount} min={1} max={points.length} step={1} />}
    <div className="kh-step-controls">
      <button type="button" disabled={cut.mergesUsed === 0} onClick={() => stepMerges(-1)}>Undo merge</button>
      <button type="button" disabled={cut.mergesUsed >= points.length - 1} onClick={() => stepMerges(1)}>Next merge</button>
      {tied !== null && <button type="button" className="kh-tie-preset" onClick={() => {
        setMode('height');
        setHeight(tied);
      }}>Set the tied height {number(tied)} exactly</button>}
    </div>
    <p className="kh-readout" aria-live="polite"><strong>{cut.count} clusters</strong> after {cut.mergesUsed} merges. Partition SSE = {number(cut.sse)} in squared coordinate units. {mode === 'count' && cut.mergesUsed < points.length - 1 ? `Next merge joins ${rowList(tree.nodes[tree.merges[cut.mergesUsed].left].members)} with ${rowList(tree.nodes[tree.merges[cut.mergesUsed].right].members)} at height ${number(tree.merges[cut.mergesUsed].height)}.` : ''} {fixture === 'six' && mode === 'count' && count === 4 ? 'This four-cluster result uses the deterministic merge prefix; no single horizontal height cut produces it on this fixture.' : ''}</p>
    <Dendrogram tree={tree} cut={cut} mode={mode} height={height} />
    <Scatter points={points} labels={cut.labels} title="The same cut, back in the original coordinates" />
    <Table caption="Every actual merge and its two member sets" headings={['merge', 'left rows', 'right rows', 'height', 'SSE increase Δ']} rows={tree.merges.map((merge, index) => [index + 1, rowList(tree.nodes[merge.left].members), rowList(tree.nodes[merge.right].members), number(merge.height), number(merge.delta)])} highlight={index => mode === 'count' && index === cut.mergesUsed - 1} />
    <p className="kh-caption">{fixture === 'six' ? 'On these six points all four linkages produce the same tree shape; only the heights differ. Ward chooses the smallest increase Δ in within-cluster SSE and plots the SciPy convention √(2Δ), not plain center distance. At a height cut, all equal-height merges are included together; moving between 5 and 3 clusters therefore skips 4 here.' : 'Under single linkage every chain neighbour is exactly 1 apart, so all five chain merges share height 1 and the triple attaches last. Complete linkage scores a candidate merge by its farthest cross pair, so extending the chain gets more expensive at every step; it attaches the triple to the right pair {P4, P5} before the left four rejoin, and its k = 2 cut splits the chain. Same points, same distances, different question.'} Dashed branches in count mode are merges not yet applied. Exact pair-cost ties use increasing cluster IDs.</p>
  </Investigation>;
}
function PixelImage({
  image,
  pixels,
  title
}) {
  return <figure className="kh-pixel-figure"><figcaption>{title}</figcaption><div className="kh-pixels" role="img" aria-label={`${title}: a constructed ${image.columns} by ${image.rows} pixel image; its palette and counts are tabulated below.`} style={{ gridTemplateColumns: `repeat(${image.columns}, 1fr)`, aspectRatio: `${image.columns}/${image.rows}` }}>{pixels.map((pixel, index) => <span key={index} style={{
        background: `rgb(${pixel.join(',')})`
      }} />)}</div></figure>;
}
const imageTitles = { mosaic: 'Mosaic: six exact colors', sky: 'Sky: gradient, sun and clouds', gradient: 'Gradient: every pixel a different color' };
export function PaletteLab() {
  const [imageId, setImageId] = useState('mosaic');
  const [count, setCount] = useState(3);
  const [prediction, setPrediction] = useState('');
  const [checked, setChecked] = useState(false);
  const limit = paletteLimit(imageId);
  const state = quantizePalette(count, imageId);
  const nextState = count < limit ? quantizePalette(count + 1, imageId) : null;
  const halved = nextState === null ? 'none' : nextState.displayedSse < state.displayedSse / 2 ? 'yes' : 'no';
  const shownRows = 12;
  return <Investigation title="Let repeated pixels vote for a small palette" question="A color that fills a third of the image and a color that appears once are both single rows in the unique-color table. Should they pull a palette entry equally hard?" onReset={() => {
    setImageId('mosaic');
    setCount(3);
    setPrediction('');
    setChecked(false);
  }}>
    <div className="kh-controls"><Choice label="Image to quantize" value={imageId} onChange={value => {
      setImageId(value);
      setCount(Math.min(3, paletteLimit(value)));
      setPrediction('');
      setChecked(false);
    }} options={Object.keys(paletteImages).map(key => [key, imageTitles[key]])} /></div>
    <Range label="Requested palette size" value={count} onChange={value => {
      setCount(value);
      setChecked(false);
    }} min={1} max={limit} step={1} />
    <Prediction prompt={`If the palette grows from ${count} to ${count + 1} entries on this image, will the displayed error fall by more than half?`} options={[["yes", 'Yes, more than half'], ['no', 'No, less than half'], ['none', 'No larger palette is available']]} value={prediction} onChange={setPrediction} answer={halved} revealed={checked} onCheck={() => setChecked(true)} explanation={nextState ? `Error ${number(state.displayedSse)} becomes ${number(nextState.displayedSse)}.` : 'Every unique color already has its own entry.'} />
    <div className="kh-image-pair"><PixelImage image={state.image} pixels={state.image.pixels} title={`Original: ${imageTitles[imageId]}`} /><PixelImage image={state.image} pixels={state.reconstructed} title={`Reconstructed with ${count} rounded palette entries`} /></div>
    <p className="kh-readout" aria-live="polite">Weighted floating-center SSE: <strong>{number(state.sse)}</strong>. Actual displayed integer-RGB SSE: <strong>{number(state.displayedSse)}</strong>; mean squared channel error = {number(state.meanSquaredChannelError)} over {state.image.pixels.length} × 3 channels. {state.uniqueCount} unique colors in {state.image.pixels.length} pixels.</p>
    <div className="kh-palette">{state.roundedCenters.map((color, index) => <div key={index}><span style={{
          background: `rgb(${color.join(',')})`
        }} /><strong>C{index}</strong><code>{color.join(', ')}</code></div>)}</div>
    <Table caption={state.uniqueCount > shownRows ? `The ${shownRows} most frequent unique colors of ${state.uniqueCount}; every color keeps its pixel-count weight in the fit` : 'Unique colors keep their pixel-frequency weights'} headings={['RGB', 'pixel count', 'assigned palette entry']} rows={state.colors.slice(0, shownRows).map((color, index) => [color.join(', '), state.counts[index], `C${state.labels[index]}`])} />
    <p className="kh-caption">These are constructed pixels, not a photograph or benchmark. The first center is the most frequent color; later centers are deterministically farthest from the chosen set, followed by weighted Lloyd updates until labels stop changing. This is one declared initialization, not a global optimum or a D² sample. On the gradient every count is one, so weighting changes nothing and the palette can only trade detail for size; on the sky the two ground rows and the sky bands dominate the fit while the sun and clouds are cheap to misrepresent. Squared distance in numeric sRGB channels is the model being minimized, not perceptual color difference, and rounding to bytes changes the displayed error.</p>
  </Investigation>;
}
