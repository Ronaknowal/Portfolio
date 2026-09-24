import { useState } from 'react';
import {
  U_POINTS, radiusGraph, gaussianRow, umapConnection, idealPair, queryNeighbors,
} from '../../data/manifold-models.js';
import { MANIFOLD_DIGITS } from '../../data/manifold-data.js';
import { DataTable, DigitLegend, DigitScatter, DigitTile, EqualPlot, Investigation, NumberField, Reflection, SelectField, format, useManifoldInvestigation } from './ManifoldShared.jsx';
import './manifold-labs.css';

const within = (value, min, max, step = null) => typeof value === 'number' && Number.isFinite(value)
  && value >= min && value <= max && (step === null || Math.abs(value / step - Math.round(value / step)) < 1e-8);

const ids = U_POINTS.map(point => [point.id, point.id]);

function validateGraph(draft) {
  if (!within(draft.radius, 0.25, 3, 0.25)) return 'Radius must be 0.25–3 in steps of 0.25.';
  if (draft.start === draft.end) return 'Choose two different endpoint IDs.';
  if (draft.points.some(point => !within(point.x, -1, 4, 0.25) || !within(point.y, -1, 4, 0.25))) {
    return 'Each coordinate must be −1–4 in steps of 0.25.';
  }
  if (new Set(draft.points.map(point => `${point.x},${point.y}`)).size !== draft.points.length) {
    return 'Give the seven points distinct coordinates; this model excludes duplicate locations.';
  }
  return '';
}

function GraphMarks({ points, edges, start, end, sx, sy }) {
  const marks = points.map(point => ({ ...point, px: sx(point.x), py: sy(point.y) }));
  const close = marks.filter(point => marks.some(other => other.id !== point.id
    && Math.hypot(point.px - other.px, point.py - other.py) < 28));
  const closeIds = new Set(close.map(point => point.id));
  const overlaps = (a, b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
  const reserved = marks.map(point => {
    const radius = closeIds.has(point.id) ? 6 : 12;
    return { left: point.px - radius, right: point.px + radius, top: point.py - radius, bottom: point.py + radius };
  });
  // Reserve the numeric y-axis labels as well as the actual node positions.
  [28, 143, 258].forEach(y => reserved.push({ left: 4, right: 53, top: y - 11, bottom: y + 11 }));
  const lineHits = (edge, box) => {
    const a = marks.find(point => point.id === edge.source);
    const b = marks.find(point => point.id === edge.target);
    let low = 0;
    let high = 1;
    for (const [origin, direction, minimum, maximum] of [[a.px, b.px - a.px, box.left, box.right], [a.py, b.py - a.py, box.top, box.bottom]]) {
      if (Math.abs(direction) < 1e-12) {
        if (origin < minimum || origin > maximum) return false;
      } else {
        const first = (minimum - origin) / direction;
        const second = (maximum - origin) / direction;
        low = Math.max(low, Math.min(first, second));
        high = Math.min(high, Math.max(first, second));
        if (low > high) return false;
      }
    }
    return low <= high;
  };
  const labels = close.map(point => {
    const offsets = [22, 34, 48, 64].flatMap(distance => [[-distance, 0], [distance, 0], [0, -distance], [0, distance], [-distance, -distance], [distance, -distance], [-distance, distance], [distance, distance]]);
    const candidates = offsets.map(([dx, dy]) => {
      const x = point.px + dx;
      const y = point.py + dy;
      return { x, y, left: x - 10, right: x + 10, top: y - 11, bottom: y + 11 };
    }).filter(box => box.left >= 5 && box.right <= 323 && box.top >= 8 && box.bottom <= 267);
    const label = candidates.find(box => !reserved.some(other => overlaps(box, other)) && !edges.some(edge => lineHits(edge, box)))
      ?? candidates.find(box => !reserved.some(other => overlaps(box, other)));
    if (label) reserved.push(label);
    return { ...point, label };
  });
  return <>
    {marks.map(point => <g key={point.id}>
      <circle className={`mf-graph-node${point.id === start || point.id === end ? ' is-endpoint' : ''}`}
        data-graph-node={point.id} cx={point.px} cy={point.py} r={closeIds.has(point.id) ? 4 : 10} />
      {!closeIds.has(point.id) && <text className="mf-node-label" x={point.px} y={point.py + 4} textAnchor="middle">{point.id}</text>}
    </g>)}
    {labels.map(point => point.label && <g key={point.id} data-graph-callout={point.id}>
      <line className="mf-label-leader" x1={point.px} y1={point.py} x2={point.label.x} y2={point.label.y} />
      <rect className="mf-label-backplate" x={point.label.left} y={point.label.top} width={20} height={22} />
      <text className="mf-node-label" x={point.label.x} y={point.label.y + 4} textAnchor="middle">{point.id}</text>
    </g>)}
  </>;
}

export function ManifoldGraphLab() {
  const state = useManifoldInvestigation({ points: U_POINTS, radius: 1, start: 'A', end: 'G' }, validateGraph);
  const [pointId, setPointId] = useState('D');
  const [edgeIndex, setEdgeIndex] = useState(0);
  const { draft, active } = state;
  const graph = radiusGraph(active.points, active.radius, active.start, active.end);
  const reference = state.previous?.active ?? active;
  const baseline = draft.start !== draft.end
    ? radiusGraph(reference.points, reference.radius, draft.start, draft.end) : graph;
  const point = draft.points.find(value => value.id === pointId);
  const changePoint = (axis, value) => state.edit({ points: draft.points.map(item => item.id === pointId ? { ...item, [axis]: value } : item) });
  const routeSet = new Set(graph.routeEdges.map(edge => [edge.source, edge.target].sort().join('')));
  const selectedEdge = graph.routeEdges[Math.min(edgeIndex, graph.routeEdges.length - 1)];
  const coordinate = id => active.points.find(item => item.id === id);
  const previews = !state.error ? draft.points.filter(item => {
    const old = coordinate(item.id);
    return item.x !== old.x || item.y !== old.y;
  }) : [];
  
  return <Investigation kind="graph" title="A · Build a route before flattening it" state={state}
    onReset={() => { state.reset(); setPointId('D'); setEdgeIndex(0); }}>
    <p>Move D sideways, or choose another point and edit its coordinates. The graph and shortest route update with every valid edit.</p>
    <div className="mf-controls">
      <SelectField label="Point to edit" value={pointId} onChange={setPointId} options={ids} />
      <NumberField label={`${pointId} x coordinate`} value={point.x} onChange={value => changePoint('x', value)} min={-1} max={4} step={0.25} />
      <NumberField label={`${pointId} y coordinate`} value={point.y} onChange={value => changePoint('y', value)} min={-1} max={4} step={0.25} />
      <NumberField label="Radius ε" value={draft.radius} onChange={radius => state.edit({ radius })} min={0.25} max={3} step={0.25} />
      <SelectField label="Start ID" value={draft.start} onChange={start => state.edit({ start })} options={ids} />
      <SelectField label="End ID" value={draft.end} onChange={end => state.edit({ end })} options={ids} />
    </div>
    <div className="mf-buttons" aria-label="Graph draft fixtures">
      <button type="button" onClick={() => state.edit({ points: U_POINTS, radius: 0.75 })}>Set U · ε 0.75</button>
      <button type="button" onClick={() => state.edit({ points: U_POINTS, radius: 1.5 })}>Set U · ε 1.5</button>
      <button type="button" onClick={() => state.edit({ points: U_POINTS, radius: 2 })}>Set U · ε 2</button>
      <button type="button" onClick={() => state.edit({ radius: 2.5 })}>Set ε 2.5</button>
      <button type="button" onClick={() => state.edit({ points: U_POINTS.map(item => item.id === 'D' ? { ...item, y: 3 } : item), radius: 1 })}>Set D at (1, 3)</button>
    </div>
    <p className="mf-caption">Compare the same proposed endpoints, {draft.start} and {draft.end}, on the previous valid graph (ε = {format(reference.radius)}) and the current graph. Their previous distance is {format(baseline.distance)}.</p>
    
    <div className="mf-geometry-pair">
      <div>
        <EqualPlot title={`Applied graph · ε ${format(active.radius)}`} points={[...active.points, ...previews]}
          describe={`Input coordinate units. ${graph.distance === null ? 'No path.' : `Shortest path ${graph.path.join(', ')}, length ${format(graph.distance)}.`}`}>
          {(sx, sy) => <>
            {graph.edges.map(edge => {
              const from = coordinate(edge.source);
              const to = coordinate(edge.target);
              const selected = selectedEdge && [edge.source, edge.target].sort().join('') === [selectedEdge.source, selectedEdge.target].sort().join('');
              return <line key={`${edge.source}${edge.target}`} x1={sx(from.x)} y1={sy(from.y)} x2={sx(to.x)} y2={sy(to.y)}
                className={`mf-graph-edge${routeSet.has([edge.source, edge.target].sort().join('')) ? ' is-route' : ''}${selected ? ' is-selected' : ''}`} />;
            })}
            <GraphMarks points={active.points} edges={graph.edges} start={active.start} end={active.end} sx={sx} sy={sy} />
            {previews.map(item => <circle key={item.id} className="mf-preview-point" cx={sx(item.x)} cy={sy(item.y)} r={13} />)}
          </>}
        </EqualPlot>
        <p className="mf-caption">Input coordinate units · endpoints {active.start}/{active.end} · solid gold: shortest route · hollow ring: draft, not applied.</p>
        <p className="mf-caption">Applied IDs: {active.points.map(item => `${item.id} (${format(item.x)}, ${format(item.y)})`).join('; ')}. Close points use separate labels with leaders; their centers stay at the entered coordinates.</p>
        {previews.length > 0 && <p className="mf-caption">Set locations: {previews.map(item => `${item.id} (${item.x}, ${item.y})`).join('; ')}.</p>}
      </div>
      <div>
        <p className="mf-readout" data-manifold-distance>Graph shortest path: <strong>{format(graph.distance)}</strong><br />Direct Euclidean distance: {format(graph.directDistance)}</p>
        {graph.routeEdges.length > 0 ? <>
          <SelectField label="Inspect a route edge" value={Math.min(edgeIndex, graph.routeEdges.length - 1)}
            onChange={value => setEdgeIndex(Number(value))} options={graph.routeEdges.map((edge, index) => [index, `${edge.source} → ${edge.target}`])} />
          <DataTable caption="Route lengths in input coordinate units" headings={['Edge', 'Length', 'Sum']}
            rows={graph.routeEdges.map(edge => [`${edge.source}–${edge.target}`, format(edge.length), format(edge.cumulative)])} />
          {selectedEdge && <p className="mf-caption">{selectedEdge.source} = ({coordinate(selectedEdge.source).x}, {coordinate(selectedEdge.source).y}); {selectedEdge.target} = ({coordinate(selectedEdge.target).x}, {coordinate(selectedEdge.target).y}). Length = √(({format(coordinate(selectedEdge.source).x - coordinate(selectedEdge.target).x)})² + ({format(coordinate(selectedEdge.source).y - coordinate(selectedEdge.target).y)})²) = {format(selectedEdge.length)}.</p>}
        </> : <p>No route edges to trace. Adjust the graph and watch where a route reconnects.</p>}
      </div>
    </div>
    <details><summary>All applied coordinates and edge choices</summary>
      <DataTable caption="Vertices" headings={['ID', 'x', 'y']} rows={active.points.map(item => [item.id, item.x, item.y])} />
      <p>{graph.edges.length} undirected edges: {graph.edges.map(edge => `${edge.source}–${edge.target} (${format(edge.length)})`).join(', ') || 'none'}.</p>
      <p>Edges use distance ≤ ε. Equal shortest paths choose lexicographic ID order; comparisons use tolerance 10⁻⁹.</p>
    </details>
    <Reflection key={state.resetCount} prompt="Which added or removed edge explains your result? Place a point that reconnects the route, then explain whether that edge plausibly follows a surface." />
  </Investigation>;
}

function validateProbability(draft) {
  if (!draft.distances.every(value => within(value, 0.25, 6, 0.25))) return 'Each distance must be 0.25–6 in steps of 0.25.';
  if (!within(draft.sigma, 0.25, 3, 0.25)) return 'Bandwidth must be 0.25–3 in steps of 0.25.';
  return '';
}

export function ManifoldProbabilityLab() {
  const state = useManifoldInvestigation({ distances: [1, 2, 3], sigma: 1, candidate: 'B', question: 'probability' }, validateProbability);
  const { draft, active } = state;
  const row = gaussianRow(active.distances, active.sigma);
  const candidateIds = ['B', 'C', 'D'];
  
  return <Investigation kind="probability" title="B · Change who receives the probability" state={state}>
    <p>These are three local distances from one query. Edit the row, choose a candidate or perplexity, and compare two bandwidths.</p>
    <div className="mf-controls">
      {candidateIds.map((id, index) => <NumberField key={id} label={`Distance to ${id}`} value={draft.distances[index]} min={0.25} max={6} step={0.25}
        onChange={value => state.edit({ distances: draft.distances.map((distance, position) => position === index ? value : distance) })} />)}
      <NumberField label="Proposed bandwidth σ" value={draft.sigma} onChange={sigma => state.edit({ sigma })} min={0.25} max={3} step={0.25} />
      <SelectField label="Quantity to inspect" value={draft.question} onChange={question => state.edit({ question })}
        options={[["probability", 'Candidate probability'], ['perplexity', 'Row perplexity']]} />
      {draft.question === 'probability' && <SelectField label="Candidate ID" value={draft.candidate} onChange={candidate => state.edit({ candidate })} options={candidateIds.map(id => [id, id])} />}
    </div>
    <div className="mf-buttons">
      <button type="button" onClick={() => state.edit({ distances: [1, 2, 3] })}>Set distances 1, 2, 3</button>
      <button type="button" onClick={() => state.edit({ distances: [2, 2, 2] })}>Set distances 2, 2, 2</button>
      <button type="button" onClick={() => state.edit({ sigma: 0.5 })}>Set σ 0.5</button>
      <button type="button" onClick={() => state.edit({ sigma: 2 })}>Set σ 2</button>
    </div>
    <p className="mf-caption">Compare the proposed distance row at the old bandwidth σ = {format(active.sigma)} with that same row at proposed σ = {format(draft.sigma)}. Editing distances changes both comparison rows.</p>
    <p className="mf-caption">Readouts are rounded. Feedback adds precision when a change would otherwise be hidden; differences no greater than 10⁻⁹ count as staying the same.</p>
    
    <div className="mf-affinity-flow" aria-label="Applied distance to Gaussian weight to normalized probability">
      <div className="mf-affinity-head"><span>Candidate · distance</span><span>Gaussian weight</span><span>Probability · 0 to 1</span></div>
      {candidateIds.map((id, index) => <div className="mf-affinity-row" key={id}>
        <div><strong>{id}</strong> · d = {format(active.distances[index])}<span className="mf-distance-track"><i style={{ left: `${active.distances[index] / 6 * 100}%` }} /></span><small>distance scale 0–6</small></div>
        <div><span className="mf-bar-track"><i className="is-weight" style={{ width: `${Math.exp(-(active.distances[index] ** 2) / (2 * active.sigma ** 2)) * 100}%` }} /></span><span>exp(−{format(active.distances[index] ** 2 / (2 * active.sigma ** 2))}) = {format(Math.exp(-(active.distances[index] ** 2) / (2 * active.sigma ** 2)))}</span></div>
        <div><span className="mf-bar-track"><i style={{ width: `${row.probabilities[index] * 100}%` }} /></span><strong data-manifold-probability={id}>{format(row.probabilities[index], 6)}</strong></div>
      </div>)}
    </div>
    <p className="mf-readout">Applied σ = {format(active.sigma)}. Probability sum = {format(row.probabilities.reduce((sum, value) => sum + value, 0))}; entropy = {format(row.entropyBits)} bits; perplexity = {format(row.perplexity)}.</p>
    <details><summary>Inspect the entropy calculation</summary>
      <DataTable caption="Each term contributes −p log₂ p bits" headings={['Candidate', 'p', 'Entropy term']}
        rows={candidateIds.map((id, index) => [id, format(row.probabilities[index], 8), format(row.probabilities[index] > 0 ? -row.probabilities[index] * Math.log2(row.probabilities[index]) : 0)])} />
      <p>Perplexity = 2<sup>{format(row.entropyBits)}</sup> = {format(row.perplexity)}. The model subtracts the minimum squared distance inside the exponent before normalizing to protect numerical precision. The displayed unnormalized weights use the original Gaussian formula.</p>
    </details>
    <Reflection key={state.resetCount} prompt="Try scaling all three distances and σ by the same factor. Why do the probabilities match? Can a middle candidate gain probability and then lose it as σ grows?" />
  </Investigation>;
}

const fuzzyInitial = { distance: 2, rhoI: 1, rhoJ: 1, sigmaI: 1 / Math.log(2), sigmaJ: 1 / Math.log(4), retainedI: true, retainedJ: true };
function validateFuzzy(draft) {
  if (!within(draft.distance, 0.25, 4)) return 'Shared distance must be 0.25–4.';
  if (![draft.rhoI, draft.rhoJ].every(value => within(value, 0, draft.distance))) return 'Each local offset ρ must be between 0 and the shared distance.';
  if (![draft.sigmaI, draft.sigmaJ].every(value => within(value, 0.1, 3))) return 'Each local scale σ must be 0.1–3.';
  return '';
}

function ConnectionDrawing({ inputs, connection }) {
  return <div className="mf-connection-flow">
    {[['i → j', connection.forward, inputs.rhoI, inputs.sigmaI, inputs.retainedI], ['j → i', connection.reverse, inputs.rhoJ, inputs.sigmaJ, inputs.retainedJ]].map(([label, value, rho, sigma, retained]) =>
      <div className="mf-directed-row" key={label}>
        <strong>{label}</strong>
        <svg viewBox="0 0 160 38" role="img" aria-label={`${label}, membership ${format(value)}`}>
          <line x1={16} x2={142} y1={19} y2={19} className="mf-directed-edge" strokeWidth={retained ? 1 + 6 * value : 1} strokeDasharray={retained ? undefined : '3 5'} />
          <path className="mf-arrow-head" d="M132 13 L142 19 L132 25" />
          <circle className="mf-graph-node" cx={16} cy={19} r={7} />
        </svg>
        <span>{retained ? `exp(−max(0, ${format(inputs.distance)} − ${format(rho)}) / ${format(sigma)})` : 'Excluded from directed support'}<br /><strong>v = {format(value)}</strong></span>
      </div>)}
    <div className="mf-union-row"><strong>i ↔ j</strong><span className="mf-union-edge" style={{ borderTopWidth: `${1 + 7 * connection.weight}px` }} /><p>w = {format(connection.forward)} + {format(connection.reverse)} − {format(connection.forward * connection.reverse)} = <strong data-manifold-union>{format(connection.weight)}</strong></p></div>
  </div>;
}

function IdealPairLab({ weight, graphDraftKey }) {
  const state = useManifoldInvestigation({ weight, separation: 1, reflected: false }, draft => within(draft.separation, 0.1, 3) ? '' : 'Separation must be 0.1–3.');

  const pair = idealPair(weight, state.active.separation);
  const samples = Array.from({ length: 121 }, (_, index) => {
    const separation = 0.1 + 2.9 * index / 120;
    return { separation, cost: idealPair(weight, separation).cost };
  });
  const maxCost = Math.max(...samples.map(sample => sample.cost)) * 1.1;
  const sx = value => 38 + (value - 0.1) / 2.9 * 228;
  const sy = value => 156 - value / maxCost * 130;
  const optimum = weight > 0 && weight < 1 ? Math.sqrt((1 - weight) / weight) : null;
  return <div className="mf-ideal-pair" data-manifold-pair>
    <h4>Ideal single-pair cost</h4>
    <p>Keep the applied graph weight w = {format(weight)} fixed, with a = b = 1. Change only the pair's separation. This is a scalar analytic cost, not the sampled UMAP optimizer.</p>
    <div className="mf-controls"><NumberField label="Separation r" value={state.draft.separation} onChange={separation => state.edit({ separation })} min={0.1} max={3} step={0.1} />
      <label className="mf-check"><input type="checkbox" checked={state.draft.reflected} onChange={event => state.edit({ reflected: event.target.checked })} />Reflect the pair's orientation</label>
    </div>
    
    <figure className="mf-cost-plot"><figcaption>Ideal cost over supported separations</figcaption>
      <svg viewBox="0 0 300 194" role="img" aria-label={`Ideal pair cost. Applied r ${state.active.separation}, cost ${format(pair.cost)}.`}>
        {[0, maxCost / 2, maxCost].map(value => <g key={value}><line className="mf-grid" x1={38} x2={266} y1={sy(value)} y2={sy(value)} /><text x={31} y={sy(value) + 4} textAnchor="end">{format(value, 1)}</text></g>)}
        {[0.1, 1, 2, 3].map(value => <text key={value} x={sx(value)} y={176} textAnchor="middle">{value}</text>)}
        <polyline className="mf-cost-curve" points={samples.map(sample => `${sx(sample.separation)},${sy(sample.cost)}`).join(' ')} />
        <circle className="mf-cost-point" cx={sx(state.active.separation)} cy={sy(pair.cost)} r={5} />
        {optimum !== null && optimum >= 0.1 && optimum <= 3 && <circle className="mf-optimum-point" cx={sx(optimum)} cy={sy(idealPair(weight, optimum).cost)} r={5} />}
      </svg>
      <p className="mf-caption">Horizontal: separation r · vertical: ideal cost in nats. Filled point: applied r. Hollow point: finite minimum, when in range.</p>
    </figure>
    <div className="mf-pair-positions"><span>{state.active.reflected ? 'j' : 'i'} at 0</span><span aria-hidden="true">← r = {format(state.active.separation)} →</span><span>{state.active.reflected ? 'i' : 'j'} at {format(state.active.separation)}</span></div>
    <p className="mf-readout">ν = 1/(1 + {format(state.active.separation)}²) = {format(pair.similarity)}. L = −w ln ν − (1−w) ln(1−ν) = {format(pair.cost)}. {optimum !== null ? `Finite minimum at r = ${format(optimum)} where ν = w${optimum < 0.1 || optimum > 3 ? '; this minimum is outside the supported range' : ''}.` : weight === 1 ? 'The minimum is approached as r → 0.' : 'The minimum is approached as r → ∞.'}</p>
    <div className="mf-buttons"><button type="button" onClick={state.reset}>Reset pair</button><button type="button" onClick={state.undo} disabled={!state.previous}>Undo pair apply</button></div>
  </div>;
}

export function ManifoldFuzzyLab() {
  const state = useManifoldInvestigation(fuzzyInitial, validateFuzzy);
  const { draft, active } = state;
  const connection = umapConnection(active);
  return <Investigation kind="fuzzy" title="C · Build a fuzzy connection, then inspect a pair" state={state}>
    <p>One shared distance, two local scales. Change a local quantity or remove one direction, then estimate the combined edge weight. <strong>Local scales are held as entered</strong>; this sandbox does not recalibrate a full neighbor graph.</p>
    <div className="mf-controls">
      <NumberField label="Shared distance d" value={draft.distance} onChange={distance => state.edit({ distance })} min={0.25} max={4} step={0.25} />
      <NumberField label="Local offset ρᵢ" value={draft.rhoI} onChange={rhoI => state.edit({ rhoI })} min={0} max={draft.distance || 4} />
      <NumberField label="Local scale σᵢ" value={draft.sigmaI} onChange={sigmaI => state.edit({ sigmaI })} min={0.1} max={3} />
      <NumberField label="Local offset ρⱼ" value={draft.rhoJ} onChange={rhoJ => state.edit({ rhoJ })} min={0} max={draft.distance || 4} />
      <NumberField label="Local scale σⱼ" value={draft.sigmaJ} onChange={sigmaJ => state.edit({ sigmaJ })} min={0.1} max={3} />
    </div>
    <div className="mf-choices">
      <label><input type="checkbox" checked={draft.retainedI} onChange={event => state.edit({ retainedI: event.target.checked })} />Retain i → j</label>
      <label><input type="checkbox" checked={draft.retainedJ} onChange={event => state.edit({ retainedJ: event.target.checked })} />Retain j → i</label>
    </div>
    <div className="mf-buttons">
      <button type="button" onClick={() => state.edit({ ...fuzzyInitial })}>Set d 2 · original scales</button>
      <button type="button" onClick={() => state.edit({ rhoI: draft.distance, rhoJ: draft.distance })}>Set both offsets at d</button>
      <button type="button" onClick={() => state.edit({ retainedI: false, retainedJ: false })}>Set neither direction retained</button>
    </div>
    
    <ConnectionDrawing inputs={active} connection={connection} />
    <details><summary>Explore the ideal pair cost with this applied weight</summary>
      <IdealPairLab key={`${state.resetCount}:${JSON.stringify(active)}`} weight={connection.weight} graphDraftKey={state.key} />
    </details>
    <Reflection key={state.resetCount} prompt="Find two different directed configurations with the same union. Which changes affect graph support, and which only change an existing edge's strength?" />
  </Investigation>;
}

function NeighborList({ kind, entries, audit, rows }) {
  const retained = new Set(audit.retained);
  const rowById = new Map(rows.map(row => [row.sourceRow, row]));
  return <div className="mf-neighbor-list" data-manifold-neighbors={kind}>
    <h4>{kind === 'input' ? 'Input-space neighbors' : 'Map-space neighbors'}</h4>
    <p className="mf-caption">{kind === 'input' ? 'Euclidean distance on pixels / 16' : 'Euclidean distance on saved map coordinates'} · nearest first.</p>
    <ol>{entries.map((entry, index) => {
      const status = retained.has(entry.sourceRow) ? 'retained' : kind === 'input' ? 'input-only' : 'map-only';
      return <li key={entry.sourceRow} data-source-row={entry.sourceRow}>
        <DigitTile row={rowById.get(entry.sourceRow)} caption={`#${index + 1} · source ${entry.sourceRow}`} />
        <span className={`mf-neighbor-badge is-${status}`}>{status}</span>
        <span className="mf-neighbor-detail">d = {format(entry.distance, 4)}<br />digit {rowById.get(entry.sourceRow).digit}</span>
      </li>;
    })}</ol>
  </div>;
}

export function ManifoldDigitLab() {
  const { rows, layouts, hash } = MANIFOLD_DIGITS;
  const initial = { dataHash: hash, query: rows[0].sourceRow, k: 10, map: 'tsne-p30-s7', metric: 'euclidean-pixels-divided-by-16' };
  const state = useManifoldInvestigation(initial, draft => {
    if (!rows.some(row => row.sourceRow === draft.query)) return 'Choose a source ID in the supplied collection.';
    if (![5, 10, 20].includes(draft.k)) return 'Choose k = 5, 10 or 20.';
    if (!layouts.some(layout => layout.key === draft.map)) return 'Choose an available saved map.';
    return '';
  });
  const [colorLabels, setColorLabels] = useState(false);
  const [display, setDisplay] = useState('candidate');
  const draft = state.draft;
  const selected = rows.find(row => row.sourceRow === draft.query);
  const candidate = layouts.find(layout => layout.key === draft.map);
  const pca = layouts.find(layout => layout.key === 'pca');
  const audit = queryNeighbors(rows, candidate.coordinates, draft.query, draft.k);
  const neighbors = audit ? { input: audit.input.map(item => item.sourceRow), map: audit.map.map(item => item.sourceRow) } : null;
  const metric = candidate.metrics[String(draft.k)];
  const selectQuery = query => state.edit({ query });
  return <Investigation kind="digits" title="D · Audit an image's neighbors" state={state}
    onReset={() => { state.reset(); setColorLabels(false); setDisplay('candidate'); }}>
    <p>Choose an observed image by its original source-row ID, then choose a saved map and k. Inspect which input neighbors survive and which map neighbors replace them as each choice changes.</p>
    <p className="mf-caption">Fixed collection: 300 observed digits, 30 per class. Class labels were used for balanced selection; the 64 pixels divided by 16 are the fit features. No per-pixel standardization. Every map contains these same source IDs.</p>
    <div className="mf-query-controls">
      <DigitTile row={selected} showLabel={colorLabels || Boolean(audit)} />
      <div className="mf-controls">
        <SelectField label="Query source-row ID" value={draft.query} onChange={value => selectQuery(Number(value))}
          options={rows.map(row => [row.sourceRow, `Source ${row.sourceRow}${colorLabels ? ` · digit ${row.digit}` : ''}`])} />
        <SelectField label="Neighbor count k" value={draft.k} onChange={value => state.edit({ k: Number(value) })}
          options={[[5, '5 neighbors'], [10, '10 neighbors'], [20, '20 neighbors']]} />
        <SelectField label="Candidate saved map" value={draft.map} onChange={map => state.edit({ map })}
          options={layouts.map(layout => [layout.key, layout.title])} />
      </div>
    </div>
    <div className="mf-buttons">
      {[30, 133, 92].filter(id => rows.some(row => row.sourceRow === id)).map(id => <button key={id} type="button" onClick={() => selectQuery(id)}>Choose source {id}</button>)}
      <label className="mf-check"><input type="checkbox" checked={colorLabels} onChange={event => setColorLabels(event.target.checked)} />Show digit labels and colors</label>
    </div>
    <p className="mf-caption">Collection average at k = {draft.k}: retained {format(metric.retention * 100, 2)}% · native trustworthiness {format(metric.trustworthiness)} · native continuity {format(metric.continuity)}. These are global scores; your query can differ.</p>
    
    <div className="mf-map-switch" role="group" aria-label="Visible map on a narrow screen">
      <button type="button" aria-pressed={display === 'pca'} onClick={() => setDisplay('pca')}>PCA reference</button>
      <button type="button" aria-pressed={display === 'candidate'} onClick={() => setDisplay('candidate')}>Candidate map</button>
    </div>
    <div className="mf-digit-maps" data-visible-map={display}>
      <div className="mf-pca-map"><DigitScatter rows={rows} coordinates={pca.coordinates} queryId={draft.query}
        onSelect={selectQuery} colorLabels={colorLabels} title="PCA reference map" /></div>
      <div className="mf-candidate-map"><DigitScatter rows={rows} coordinates={candidate.coordinates} queryId={draft.query}
        onSelect={selectQuery} colorLabels={colorLabels} neighbors={neighbors} title={candidate.title} /></div>
    </div>
    {colorLabels && <><DigitLegend /><p className="mf-caption">The image caption and source-ID selector give digit text; colored points provide an optional class overlay. Pixel intensities run from 0 (black) to 16 (white).</p></>}
    {<>
      <p className="mf-readout" data-manifold-retained>{audit.count} / {draft.k} retained. Solid gold edges: retained; dashed green: input-only; dotted blue: map-only.</p>
      <div className="mf-neighbor-pair">
        <NeighborList kind="input" entries={audit.input} audit={audit} rows={rows} />
        <NeighborList kind="map" entries={audit.map} audit={audit} rows={rows} />
      </div>
      <p className="mf-caption">Retained IDs: {audit.retained.join(', ') || 'none'}. Input-only IDs: {audit.missing.join(', ') || 'none'}. Map-only IDs: {audit.falseNeighbors.join(', ') || 'none'}.</p>
      <Reflection key={state.key} prompt="Pick one changed neighbor. Which pixel strokes make the replacement plausible or surprising? Compare a second map, then explain whether its global ranking describes this query." />
    </>}
    <details><summary>Inspect the selected image's 64 pixel values and distance rules</summary>
      <DataTable caption={`Source ${selected.sourceRow} · raw 0–16 pixel values, row-major 8 × 8`}
        headings={Array.from({ length: 8 }, (_, index) => `c${index}`)}
        rows={Array.from({ length: 8 }, (_, index) => selected.pixels.slice(index * 8, index * 8 + 8))} />
      <p>Local lists exclude the query and break exact distance ties by ascending source-row ID. Distances use full saved coordinate precision before display rounding. Native trustworthiness and continuity use scikit-learn's own sorting convention. Source IDs identify original rows, not positions in this 300-row subset.</p>
      <p>Input collection SHA-256: <code className="mf-hash">{hash}</code>. Every selectable setting is an actual stored fit. Changing the display tab preserves the query; changing query, k or map immediately recomputes its neighbor comparison.</p>
    </details>
  </Investigation>;
}
