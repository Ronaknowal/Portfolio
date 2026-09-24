import { graphDefault, propagateGraph } from '../../data/semi-supervised-models.js';
import './semi-supervised.css';

export const formatScore = value => value === null || value === undefined ? 'unavailable' : Number(value.toFixed(6)).toString();
export function DataTable({ caption, headers, rows }) {
  return <div className="ssl-table" tabIndex={0} role="region" aria-label={caption}>
    <table><caption>{caption}</caption><thead><tr>{headers.map((header, index) => <th key={index} scope="col">{header}</th>)}</tr></thead>
      <tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => j === 0 ? <th key={j} scope="row">{cell}</th> : <td key={j}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

export function GraphPicture({ graph, scores, title = 'Authored weighted graph' }) {
  const count = graph.nodes.length;
  const positions = graph.nodes.map((_, i) => count <= 4
    ? [35 + i * 270 / (count - 1), 75]
    : [170 + 120 * Math.cos(-Math.PI / 2 + i * 2 * Math.PI / count), 150 + 115 * Math.sin(-Math.PI / 2 + i * 2 * Math.PI / count)]);
  return <div className="ssl-graph-picture">
    <svg viewBox={`0 0 340 ${count <= 4 ? 150 : 300}`} role="img" aria-label={`${title}. Weights and scores are listed below.`}>
      {graph.edges.map(([a, b, weight]) => count <= 4 && Math.abs(a - b) > 1
        ? <path key={`${a}-${b}`} d={`M ${positions[a][0]} ${positions[a][1]} Q ${(positions[a][0] + positions[b][0]) / 2} 0 ${positions[b][0]} ${positions[b][1]}`} fill="none" stroke="#83929c" strokeWidth={1 + weight} />
        : <line key={`${a}-${b}`} x1={positions[a][0]} y1={positions[a][1]} x2={positions[b][0]} y2={positions[b][1]} stroke="#83929c" strokeWidth={1 + weight} />)}
      {graph.nodes.map((node, i) => <g key={node.name}>
        <circle cx={positions[i][0]} cy={positions[i][1]} r={18} fill={node.label === null ? '#10171c' : node.label === 0 ? '#287763' : '#b98d36'} stroke="#dce4e9" strokeWidth={node.label === null ? 1 : 3} strokeDasharray={scores?.[i] === null ? '3 3' : undefined} />
        <text x={positions[i][0]} y={positions[i][1] + 6} textAnchor="middle" fill="#fff" fontSize={18}>{node.name.length <= 2 ? node.name : i + 1}</text>
      </g>)}
    </svg>
    <p className="ssl-caption">Thick outline = observed label; thin outline = inferred node; dashed outline = no evidence. Node position is layout only. Edge thickness follows weight.</p>
    {graph.nodes.some(node => node.name.length > 2) && <p className="ssl-caption">Node key: {graph.nodes.map((node, i) => `${node.name.length <= 2 ? node.name : i + 1} = ${node.name}`).join('; ')}.</p>}
    <div className="ssl-edge-key">{graph.edges.length ? graph.edges.map(([a, b, weight]) => <span key={`${a}-${b}`}>{graph.nodes[a].name} ↔ {graph.nodes[b].name}: {weight}</span>) : <span>No edges</span>}</div>
  </div>;
}

export function LabelLedgerFigure() {
  return <figure className="ssl-figure" data-ssl-figure="ledger">
    <div className="ssl-split-ledger">
      <section><h4>Fit · 320 training inputs</h4><p><strong className="ssl-observed">6 observed labels</strong></p><p>314 targets sealed during fitting</p><p className="ssl-pseudo">Model guesses stay guesses<br />Origin: donor + round</p></section>
      <section><h4>Select · 80 development rows</h4><p>80 observed labels compare the predefined candidates.</p><p>Candidate choice ↓</p></section>
      <section><h4>Report · 80 locked test rows</h4><p>80 observed labels assess only the selected model.</p><p>No return arrow into fitting.</p></section>
    </div>
    <figcaption>Six fitting labels + 160 evaluation labels. Predictions never acquire the status of an observed label.</figcaption>
  </figure>;
}

export function SameInputsFigure() {
  const coordinates = [-2.2, -2, -1.8, 1.8, 2, 2.2];
  const heights = [0, 0.2, -0.2, -0.1, 0.1, 0];
  const targets = [[0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1]];
  return <figure className="ssl-figure" data-ssl-figure="same-inputs">
    <div className="ssl-two-panels">{targets.map((labels, panel) => <section key={panel}><h4>{panel === 0 ? 'A target aligned with groups' : 'A different realized target'}</h4>
      <svg viewBox="0 0 300 150" role="img" aria-label={`Same six input coordinates, target rule ${panel + 1}; table below lists all values.`}>
        <line x1="15" y1="118" x2="285" y2="118" stroke="#60727d" />
        {coordinates.map((x, i) => labels[i] === 0
          ? <circle key={i} cx={150 + x * 48} cy={65 - heights[i] * 130} r="7" fill="#72cbb0" />
          : <rect key={i} x={143 + x * 48} y={58 - heights[i] * 130} width="14" height="14" fill="#e2b55a" />)}
      </svg></section>)}</div>
    <figcaption>Authored counterexample. Circles = class 0; squares = class 1. The inputs and display coordinates are identical in both panels. Density alone cannot choose the target rule.</figcaption>
    <details><summary>Exact authored coordinates and labels</summary><DataTable caption="Same inputs, changed targets" headers={['x', 'Display y', 'Target 1', 'Target 2']} rows={coordinates.map((x, i) => [x, heights[i], targets[0][i], targets[1][i]])} /></details>
  </figure>;
}

export function HardFlowFigure() {
  const result = propagateGraph(graphDefault);
  return <figure className="ssl-figure" data-ssl-figure="harmonic">
    <GraphPicture graph={graphDefault} scores={result.scores} title="Hard-clamped chain with A = 0 and D = 1" />
    <DataTable caption="Synchronous averaging; endpoints stay fixed" headers={['State', 'A', 'B', 'C', 'D']} rows={[
      ['Initial', 0, 0.5, 0.5, 1], ['First update', 0, 0.25, 0.75, 1], ['Equilibrium', 0, '1/3', '2/3', 1],
    ]} />
    <figcaption>Read the same drawing as a circuit: A/D are fixed voltages, edges have conductance 1, and each interior node has zero net current. The values are exact solutions of this authored graph.</figcaption>
  </figure>;
}

export function EvidenceFigure() {
  const result = propagateGraph(graphDefault, 'soft', 0.8);
  return <figure className="ssl-figure" data-ssl-figure="evidence">
    <h4>Raw evidence → normalize each supported row</h4>
    <div className="ssl-evidence-rows">{result.equilibrium.map((row, i) => <div key={i} className="ssl-evidence-row">
      <strong>{graphDefault.nodes[i].name}</strong>
      <div><span>Raw · scale 0–0.5</span><div className="ssl-bar-frame">{row.map((value, c) => <div key={c} className={`ssl-segment ssl-class-${c}`} style={{ width: `${value / 0.5 * 100}%` }} />)}</div><small>{row.map(formatScore).join(' + ')}</small></div>
      <div><span>Readout · scale 0–1</span><div className="ssl-bar-frame"><div className="ssl-segment ssl-class-0" style={{ width: `${(1 - result.scores[i]) * 100}%` }} /><div className="ssl-segment ssl-class-1" style={{ width: `${result.scores[i] * 100}%` }} /></div><small>class 1: {formatScore(result.scores[i])}</small></div>
    </div>)}</div>
    <p>Class 0 = green left segment; class 1 = gold right segment. B has total evidence {formatScore(result.equilibrium[1].reduce((a, b) => a + b, 0))}.</p>
    <p>S row sums: {result.symmetric.map(row => formatScore(row.reduce((a, b) => a + b, 0))).join(', ')}. They are not all one.</p>
    <figcaption>Exact calculation, α = 0.8. S is symmetric adjacency, P is the random-walk matrix, and F holds propagated evidence. At α = 0, unknown rows have zero mass and no normalized readout.</figcaption>
  </figure>;
}

export function PrototypeFlowFigure() {
  return <figure className="ssl-figure" data-ssl-figure="provenance">
    <div className="ssl-flow"><section><h4>Observe</h4><p className="ssl-observed">−2 → 0<br />2 → 1</p></section><span aria-hidden="true">→</span><section><h4>Propose · round 1</h4><p className="ssl-pseudo">−1 → 0<br />1, 3 → 1</p></section><span aria-hidden="true">→</span><section><h4>Refit</h4><p>Means: −1.5 and 2<br />Boundary: 0.25</p></section></div>
    <p>Still unlabeled: x = 0. Its new class-0 score is about 0.852. Round 2 accepts it; refitting then moves the boundary to 0.5.</p>
    <figcaption>Authored prototype model, threshold 0.8. Solid badges are observed; dashed badges are model guesses. Each accepted batch changes the next model.</figcaption>
  </figure>;
}

export function PairedViewsFigure() {
  return <figure className="ssl-figure" data-ssl-figure="paired-views">
    <div className="ssl-flow"><section><h4>Donor: view 1</h4><p>Observed rule<br /><strong>red → 0</strong></p></section><span aria-hidden="true">→</span><section><h4>Same row 2</h4><p>red | triangle<br /><span className="ssl-pseudo">Offer class 0</span></p></section><span aria-hidden="true">→</span><section><h4>Recipient: view 2</h4><p>Use its own feature<br /><strong>triangle → 0</strong></p></section></div>
    <p>Next round, triangle on row 3 can teach view 1 <strong>green → 0</strong>. On row 6, red → 0 disagrees with square → 1; both offers are deferred.</p>
    <figcaption>Authored category rules. A label crosses views through the shared row. A feature value is never passed to a model trained on the other representation.</figcaption>
  </figure>;
}

export function PromotionAuditFigure() {
  const accepted = [48, 106, 60, 18, 11, 6, 3, 0];
  const wrong = [0, 12, 28, 14, 10, 6, 3, 0];
  let cumulative = 0;
  const rows = accepted.map((count, i) => { cumulative += count; return [i + 1, count - wrong[i], wrong[i], cumulative]; });
  return <figure className="ssl-figure" data-ssl-figure="promotion-audit">
    <h4>The first batch succeeds; later guesses deteriorate</h4>
    <div className="ssl-audit-panels"><section><h5>Newly accepted · axis 0–110 specimens</h5>{rows.map(([round, correct, incorrect]) => <div className="ssl-count-row" key={round}><span>R{round}</span><div className="ssl-count-track"><span className="ssl-class-0" style={{ width: `${correct / 110 * 100}%` }} /><span className="ssl-error" style={{ width: `${incorrect / 110 * 100}%` }} /></div><span>{correct + incorrect}</span></div>)}<p>Green: correct. Rose: wrong.</p></section>
      <section><h5>Cumulative accepted · axis 0–320 specimens</h5>{rows.map(([round, , , total]) => <div className="ssl-count-row" key={round}><span>R{round}</span><div className="ssl-count-track"><span className="ssl-class-1" style={{ width: `${total / 320 * 100}%` }} /></div><span>{total}</span></div>)}<p>252 accepted; 73 wrong; 62 still unlabeled.</p></section></div>
    <figcaption>Executed Banknote Authentication experiment, threshold 0.8. Hidden benchmark targets were opened after fitting for explanation. No unmeasured performance curve is implied between these rounds.</figcaption>
    <details><summary>Exact counts behind the bars</summary><DataTable caption="Retrospective promotion audit" headers={['Round', 'Correct', 'Wrong', 'Cumulative accepted']} rows={rows} /></details>
  </figure>;
}
