import { useId, useMemo, useState } from 'react';
import { FLOW_EDGES, FLOW_LABELS, MATCHING_DEFAULT, MATCHING_DEFICIENT, PIXEL_BACKGROUND_COST, PIXEL_FOREGROUND_COST, PIXEL_NEIGHBORS, inspectFlow, solveFlow, cutCapacity, solveMatching, solvePixelCut, pixelEnergy } from '../../data/network-flow-models.js';
import './network-flow-labs.css';
const positions = [[180, 30], [80, 105], [280, 105], [80, 230], [280, 230], [180, 305]];
const edgeLabels = [[110, 49], [252, 49], [48, 171], [131, 145], [228, 145], [112, 280], [249, 280]];
const pairName = edge => `${FLOW_LABELS[edge[0]]} → ${FLOW_LABELS[edge[1]]}`;
const copyMatrix = matrix => matrix.map(row => row.slice());
const leftNames = ['A', 'B', 'C'];
const rightNames = ['1', '2', '3'];
const names = (indices, labels) => indices.length ? indices.map(i => labels[i]).join(', ') : '∅';
function FlowDrawing({
  edges,
  flows,
  path = [],
  badEdges = [],
  badVertices = [],
  sourceSide = null,
  cutEdges = [],
  title
}) {
  const marker = `nf-${useId().replaceAll(':', '')}`;
  const line = (from, to) => {
    const [x1, y1] = positions[from];
    const [x2, y2] = positions[to];
    const distance = Math.hypot(x2 - x1, y2 - y1);
    const dx = (x2 - x1) / distance;
    const dy = (y2 - y1) / distance;
    return {
      x1: x1 + dx * 23,
      y1: y1 + dy * 23,
      x2: x2 - dx * 26,
      y2: y2 - dy * 26
    };
  };
  return <svg className="nf-network" viewBox="0 0 360 340" role="img" aria-label={title}>
    <defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke" /></marker></defs>
    {edges.map(([from, to], i) => <line key={i} {...line(from, to)} className={`nf-edge ${badEdges.includes(i) ? 'nf-invalid' : ''} ${cutEdges.includes(i) ? 'nf-cut-edge' : ''}`} markerEnd={`url(#${marker})`} />)}
    {path.map(arc => <line key={`${arc.edgeId}-${arc.direction}`} {...line(arc.from, arc.to)} className={`nf-edge nf-path ${arc.direction === -1 ? 'nf-cancel' : ''}`} markerEnd={`url(#${marker})`} />)}
    {edges.map((edge, i) => <g key={i} transform={`translate(${edgeLabels[i].join(',')})`}>
      <rect x="-26" y="-15" width="52" height="26" rx="4" />
      <text textAnchor="middle" y="5" className={badEdges.includes(i) ? 'nf-invalid-text' : ''}>{flows[i]}/{edge[2]}</text>
    </g>)}
    {positions.map(([x, y], vertex) => <g key={vertex} transform={`translate(${x},${y})`}>
      <circle r="21" className={`${badVertices.includes(vertex) ? 'nf-node-invalid' : ''} ${sourceSide?.includes(vertex) ? 'nf-source-side' : ''}`} />
      <text textAnchor="middle" y="7">{FLOW_LABELS[vertex]}</text>
    </g>)}
  </svg>;
}
function EdgeTable({
  edges,
  flows
}) {
  return <div className="nf-table-scroll" tabIndex={0} role="region" aria-label="Detailed flow data"><table>
      <caption>Original edge identities and current flow</caption>
      <thead><tr>
          <th>Edge</th>
          <th>Direction</th>
          <th>Flow</th>
          <th>Capacity</th>
        </tr></thead>
      <tbody>{edges.map((edge, i) => <tr key={i}>
          <th>e{i}</th>
          <td>{pairName(edge)}</td>
          <td>{flows[i]}</td>
          <td>{edge[2]}</td>
        </tr>)}</tbody>
    </table></div>;
}
export function FlowConservationLab() {
  const [flows, setFlows] = useState(FLOW_EDGES.map(() => 0));
  const audit = inspectFlow(6, FLOW_EDGES, flows);
  return <section className="nf-lab" aria-label="Flow conservation investigation">
    <p className="nf-eyebrow">INVESTIGATE · CAN THIS ALLOCATION EXIST?</p>
    <h3>Every internal vertex must balance</h3>
    <p>Predict what breaks if S sends one unit to A and A sends nothing onward. Each number is a proposed flow; capacities stay fixed at 1. Edits take effect immediately.</p>
    <div className="nf-buttons">
      <button onClick={() => setFlows([1, 0, 1, 0, 0, 1, 0])}>One balanced route</button>
      <button onClick={() => setFlows([1, 0, 0, 0, 0, 0, 0])}>Stranded at A</button>
      <button onClick={() => setFlows(FLOW_EDGES.map(() => 0))}>Reset flows</button>
    </div>
    <div className="nf-two-view">
      <figure>
        <FlowDrawing edges={FLOW_EDGES} flows={flows} badEdges={audit.badEdges} badVertices={audit.badVertices} title="Directed six-node network. Edge labels show proposed flow over capacity; violations also appear in the tables." />
        <figcaption>S is the source; T is the sink. Labels are <strong>flow/capacity</strong>. Crossed lines are separate edges, with no vertex at the crossing.</figcaption>
      </figure>
      <div className="nf-table-scroll" tabIndex={0} role="region" aria-label="Detailed flow data"><table>
          <caption>Edit a proposed allocation</caption>
          <thead><tr>
              <th>Edge</th>
              <th>Proposed flow</th>
            </tr></thead>
          <tbody>{FLOW_EDGES.map((edge, i) => <tr key={i}>
              <th>e{i} · {pairName(edge)}</th>
              <td>
                <select aria-label={`Proposed flow ${pairName(edge)}`} value={flows[i]} onChange={event => setFlows(previous => previous.map((value, j) => j === i ? Number(event.target.value) : value))}>{[0, 1, 2].map(value => <option key={value}>{value}</option>)}</select>
                {audit.badEdges.includes(i) && <span className="nf-error">Above capacity 1</span>}
              </td>
            </tr>)}</tbody>
        </table></div>
    </div>
    <p className={audit.feasible ? 'nf-result' : 'nf-result nf-error'} role="status">{audit.feasible ? `Feasible flow. Net source outflow = sink inflow = ${audit.sourceValue}.` : `Infeasible proposal. ${audit.badEdges.length ? `Capacity violated on ${audit.badEdges.map(i => `e${i}`).join(', ')}. ` : ''}${audit.badVertices.length ? `Conservation violated at ${names(audit.badVertices, FLOW_LABELS)}.` : ''}`}</p>
    <div className="nf-table-scroll" tabIndex={0} role="region" aria-label="Detailed flow data"><table>
        <caption>Incoming and outgoing totals; internal vertices require in − out = 0</caption>
        <thead><tr>
            <th>Vertex</th>
            <th>In</th>
            <th>Out</th>
            <th>In − out</th>
          </tr></thead>
        <tbody>{FLOW_LABELS.map((label, i) => <tr key={i}>
            <th>{label}</th>
            <td>{audit.incoming[i]}</td>
            <td>{audit.outgoing[i]}</td>
            <td>{audit.balance[i]}</td>
          </tr>)}</tbody>
      </table></div>
    <p className="nf-note">A balanced proposal can still exceed a capacity. Try 2 on every edge of S → A → C → T. These are steady-state amounts in one abstract unit, not a time-based fluid simulation.</p>
  </section>;
}
export function ResidualPairFigure() {
  return <figure className="nf-figure" aria-label="Residual edge identity figure">
    <p className="nf-eyebrow">ONE ORIGINAL EDGE · TWO WAYS TO CHANGE IT</p>
    <div className="nf-residual-pairs">
      <div>
        <h3>Original e0: A → C</h3>
        <p>capacity 5 · current flow 3</p>
        <p className="nf-residual-lane">A <span>→ add up to 2 →</span> C</p>
        <p className="nf-residual-lane nf-cancellation">A <span>← undo up to 3 ←</span> C</p>
      </div>
      <div>
        <h3>Separate original e1: C → A</h3>
        <p>capacity 4 · current flow 1</p>
        <p className="nf-residual-lane">C <span>→ add up to 3 →</span> A</p>
        <p className="nf-residual-lane nf-cancellation">C <span>← undo up to 1 ←</span> A</p>
      </div>
    </div>
    <figcaption>Four distinct residual arcs belong to two original edges. C → A can mean “undo e0” or “add on e1”; preserve that identity when updating flow. This is a local accounting example, not a complete feasible network.</figcaption>
  </figure>;
}
export function AugmentingFlowLab() {
  const [draft, setDraft] = useState(FLOW_EDGES.map(edge => String(edge[2])));
  const [edges, setEdges] = useState(FLOW_EDGES.map(edge => edge.slice()));
  const [step, setStep] = useState(0);
  const [preview, setPreview] = useState(false);
  const [error, setError] = useState('');
  const [side, setSide] = useState([0]);
  const result = useMemo(() => solveFlow(6, edges), [edges]);
  const flows = result.states[step];
  const next = result.steps[step];
  const selectedPath = preview && next ? next.path : [];
  const finished = !next;
  const audit = inspectFlow(6, edges, flows);
  const cut = cutCapacity(6, edges, side);
  function apply() {
    if (draft.some(value => !/^[0-9]$/.test(value.trim()))) {
      setError('Use one integer from 0 to 9 in every capacity field. The active network is unchanged.');
      return;
    }
    setEdges(FLOW_EDGES.map(([u, v], i) => [u, v, Number(draft[i])]));
    setStep(0);
    setPreview(false);
    setSide([0]);
    setError('');
  }
  function reset() {
    setDraft(FLOW_EDGES.map(() => '1'));
    setEdges(FLOW_EDGES.map(edge => edge.slice()));
    setStep(0);
    setPreview(false);
    setSide([0]);
    setError('');
  }
  return <section className="nf-lab" aria-label="Residual augmentation investigation">
    <p className="nf-eyebrow">INVESTIGATE · REROUTE AN EARLIER CHOICE</p>
    <h3>A reverse arc releases an earlier allocation</h3>
    <p>First preview the residual route, predict its bottleneck, then send that amount. The default second augmentation cancels flow on A → C. BFS visits residual arcs in original-edge order; it minimizes hop count.</p>
    <details>
      <summary>Edit original capacities</summary>
      <div className="nf-capacity-fields">{edges.map((edge, i) => <label key={i}>e{i} · {pairName(edge)}<input aria-label={`Capacity ${pairName(edge)}`} type="text" inputMode="numeric" maxLength={8} value={draft[i]} onChange={event => setDraft(previous => previous.map((value, j) => j === i ? event.target.value : value))} /></label>)}</div>
      <button onClick={apply}>Apply capacities</button>
      <p className="nf-note">Draft edits take effect only when applied. The vertex and original-edge set stays fixed; capacities are integers 0…9.</p>
    </details>
    {error && <p role="alert" className="nf-error">{error}</p>}
    <div className="nf-buttons">
      <button disabled={!next || preview} onClick={() => setPreview(true)}>Preview next path</button>
      <button disabled={!next || !preview} onClick={() => {
        setStep(step + 1);
        setPreview(false);
      }}>Send bottleneck</button>
      <button disabled={step === 0 && !preview} onClick={() => {
        if (preview) setPreview(false);else setStep(step - 1);
      }}>Back</button>
      <button onClick={reset}>Reset network</button>
    </div>
    <p className="nf-result" role="status">{finished ? `No residual S–T path. Maximum flow = ${result.value}.` : preview ? `Bottleneck ${next.delta}. ${next.path.some(arc => arc.direction === -1) ? 'This path includes cancellation.' : 'Every chosen arc adds forward flow.'}` : `${step} augmentation${step === 1 ? '' : 's'} applied. Current feasible flow = ${audit.sourceValue}.`}</p>
    <div className="nf-two-view">
      <figure>
        <FlowDrawing edges={edges} flows={flows} path={selectedPath} sourceSide={finished ? side : null} cutEdges={finished ? cut.outgoingIds : []} title="Current original flow and previewed residual path. Amber adds flow, dashed pink cancels flow. After completion, green vertices show the inspected source side and green edges leave that side." />
        <figcaption>Edge labels remain current <strong>flow/capacity</strong> until you send. Amber arrows add; dashed pink arrows cancel. {finished && `Inspected source side: ${names(side, FLOW_LABELS)}. Green arrows are its outgoing original cut edges. Toggle the cut vertices below to change this view.`}</figcaption>
      </figure>
      <div>{preview && next ? <div className="nf-table-scroll" tabIndex={0} role="region" aria-label="Detailed flow data"><table>
            <caption>Residual route: {next.path.map(arc => FLOW_LABELS[arc.from]).concat('T').join(' → ')}</caption>
            <thead><tr>
                <th>Traverse</th>
                <th>Owned change</th>
                <th>Available</th>
              </tr></thead>
            <tbody>{next.path.map(arc => <tr key={`${arc.edgeId}-${arc.direction}`}>
                <th>{FLOW_LABELS[arc.from]} → {FLOW_LABELS[arc.to]}</th>
                <td>{arc.direction === 1 ? 'Add' : 'Cancel'} e{arc.edgeId}</td>
                <td>{arc.residual}</td>
              </tr>)}</tbody>
          </table></div> : <EdgeTable edges={edges} flows={flows} />}</div>
    </div>
    {finished && <div className="nf-cut-inspector">
      <h4>Inspect a cut as an upper bound</h4>
      <p>S stays inside; T stays outside. Toggle internal vertices to change the source side. This does not alter the completed flow.</p>
      <div className="nf-buttons">
        {[1, 2, 3, 4].map(vertex => <button key={vertex} aria-pressed={side.includes(vertex)} onClick={() => setSide(previous => previous.includes(vertex) ? previous.filter(v => v !== vertex) : [...previous, vertex])}>{FLOW_LABELS[vertex]} {side.includes(vertex) ? 'inside' : 'outside'}</button>)}
        <button onClick={() => setSide(result.reachable.slice())}>Use residual source side</button>
      </div>
      <p className="nf-result">Source side: {names(side.slice().sort((a, b) => a - b), FLOW_LABELS)}. Cut capacity {cut.capacity} {cut.capacity === result.value ? '= flow: both are optimal.' : `> flow ${result.value}: this cut is a looser upper bound.`}</p>
      <p>Outgoing original edges counted: {cut.outgoingIds.length ? cut.outgoingIds.map(i => `e${i} (${pairName(edges[i])}, capacity ${edges[i][2]})`).join('; ') : 'none'}. Incoming original edges excluded: {cut.incomingIds.length ? cut.incomingIds.map(i => `e${i}`).join(', ') : 'none'}.</p>
    </div>}
    <p className="nf-note">Original parallel/opposite edges retain separate IDs in the native solver. This six-node editor limits topology so the diagram stays readable. Geometry is connectivity; arrow length and line crossing have no cost meaning.</p>
  </section>;
}
export function MatchingCoverLab() {
  const marker = `nf-matching-${useId().replaceAll(':', '')}`;
  const [matrix, setMatrix] = useState(copyMatrix(MATCHING_DEFAULT));
  const [step, setStep] = useState(0);
  const [certificate, setCertificate] = useState(false);
  const result = useMemo(() => solveMatching(matrix), [matrix]);
  const current = result.flow.states[step];
  const matching = result.pairEdges.filter(pair => current[pair.edgeId] === 1);
  const complete = step === result.flow.steps.length;
  function load(next) {
    setMatrix(copyMatrix(next));
    setStep(0);
    setCertificate(false);
  }
  const last = step ? result.flow.steps[step - 1] : null;
  return <section className="nf-lab" aria-label="Bipartite matching and cover investigation">
    <p className="nf-eyebrow">INVESTIGATE · ASSIGN, REROUTE, CERTIFY</p>
    <h3>Compatibility is not yet an assignment</h3>
    <p>Rows are workers A–C; columns are tasks 1–3. Toggle compatibility to start a new problem immediately. Each worker and task can be used once.</p>
    <div className="nf-buttons">
      <button onClick={() => load(MATCHING_DEFAULT)}>Reset compatibility</button>
      <button onClick={() => load(MATCHING_DEFICIENT)}>Three workers, two neighbors</button>
      <button onClick={() => load(Array.from({
        length: 3
      }, () => [false, false, false]))}>Clear compatibility</button>
    </div>
    <div className="nf-matching-views">
      <div className="nf-table-scroll" tabIndex={0} role="region" aria-label="Detailed flow data"><table className="nf-matrix">
          <caption>Allowed pair? Press to toggle</caption>
          <thead><tr>
              <th>Worker</th>
              {rightNames.map(name => <th key={name}>Task {name}</th>)}
            </tr></thead>
          <tbody>{matrix.map((row, left) => <tr key={left}>
              <th>{leftNames[left]}</th>
              {row.map((allowed, right) => <td key={right}><button aria-label={`Compatibility ${leftNames[left]} to ${rightNames[right]}`} aria-pressed={allowed} onClick={() => {
                  const edited = copyMatrix(matrix);
                  edited[left][right] = !allowed;
                  load(edited);
                }}>{allowed ? '✓' : '—'}</button></td>)}
            </tr>)}</tbody>
        </table></div>
      <figure>
        <svg className="nf-matching" viewBox="0 0 340 285" role="img" aria-label="Bipartite compatibility graph. Thick amber edges are currently assigned. Cover and reachability labels appear when the certificate is revealed.">
          <defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke" /></marker></defs>
          {result.pairEdges.map(({
            left,
            right,
            edgeId
          }) => {
            const matched = current[edgeId] === 1;
            return <line key={edgeId} x1="73" y1={55 + left * 85} x2="267" y2={55 + right * 85} className={matched ? 'nf-edge nf-path' : 'nf-edge'} markerEnd={certificate && !matched ? `url(#${marker})` : undefined} markerStart={certificate && matched ? `url(#${marker})` : undefined} />;
          })}
          {certificate && result.reachableLeft.map(i => <circle key={`left-${i}`} cx="45" cy={55 + i * 85} r="28" className="nf-reached-ring" />)}
          {certificate && result.reachableRight.map(i => <circle key={`right-${i}`} cx="295" cy={55 + i * 85} r="28" className="nf-reached-ring" />)}
          {[0, 1, 2].map(i => <g key={i}>
            <circle cx="45" cy={55 + i * 85} r="22" className={certificate && result.coverLeft.includes(i) ? 'nf-cover-node' : ''} />
            <text x="45" y={62 + i * 85} textAnchor="middle">{leftNames[i]}</text>
            <circle cx="295" cy={55 + i * 85} r="22" className={certificate && result.coverRight.includes(i) ? 'nf-cover-node' : ''} />
            <text x="295" y={62 + i * 85} textAnchor="middle">{rightNames[i]}</text>
          </g>)}
          <text x="45" y="19" textAnchor="middle" className="nf-partition-label">LEFT</text>
          <text x="295" y="19" textAnchor="middle" className="nf-partition-label">RIGHT</text>
        </svg>
        <figcaption>Thick amber lines are the current matching. {certificate ? 'Arrows show alternating-search directions; dashed outer rings mark reached vertices. Green vertices form the minimum cover.' : 'One incident matching edge at most per vertex.'} Crossing lines are not extra vertices.</figcaption>
      </figure>
    </div>
    <div className="nf-buttons">
      <button disabled={complete} onClick={() => {
        setStep(step + 1);
        setCertificate(false);
      }}>Augment matching</button>
      <button disabled={step === 0} onClick={() => {
        setStep(step - 1);
        setCertificate(false);
      }}>Back one augmentation</button>
      <button disabled={!complete} onClick={() => setCertificate(value => !value)}>{certificate ? 'Hide certificate' : 'Reveal cover and shortage'}</button>
    </div>
    <p className="nf-result" role="status">{complete ? 'Maximum' : 'Current'} matching size {matching.length}: {matching.length ? matching.map(({
        left,
        right
      }) => `${leftNames[left]}–${rightNames[right]}`).join(', ') : '∅'}.</p>
    {last && <p>Last alternating change: {last.path.filter(arc => result.pairEdges.some(pair => pair.edgeId === arc.edgeId)).map(arc => {
        const pair = result.pairEdges.find(value => value.edgeId === arc.edgeId);
        return `${arc.direction === 1 ? 'add' : 'remove'} ${leftNames[pair.left]}–${rightNames[pair.right]}`;
      }).join('; ')}.</p>}
    {certificate && <div className="nf-cover-proof">
      <h4>Start at unmatched left vertices</h4>
      <p>Follow unmatched edges L → R and matched edges R → L. Reached left ZL: <strong>{names(result.reachableLeft, leftNames)}</strong>. Reached right ZR: <strong>{names(result.reachableRight, rightNames)}</strong>.</p>
      <p className="nf-result">Minimum cover = unvisited left {names(result.coverLeft, leftNames)} plus visited right {names(result.coverRight, rightNames)}. Size {result.coverLeft.length + result.coverRight.length} = matching size {result.matching.length}.</p>
      {result.deficiency > 0 ? <p>Shortage certificate: {result.reachableLeft.length} workers ({names(result.reachableLeft, leftNames)}) have only {result.reachableRight.length} distinct allowed tasks ({names(result.reachableRight, rightNames)}). At least {result.deficiency} of these workers must remain unmatched.</p> : <p>Every left vertex is matched. Both partitions have size 3 here, so this matching is also perfect.</p>}
    </div>}
    <p className="nf-note">This models allowed pairings, not preferences or assignment costs. An unmatched worker starts a new alternating search; never permanently freeze a greedy pair. The native program also supports unequal and empty partitions.</p>
  </section>;
}
export function NodeSplitFigure() {
  return <figure className="nf-figure" aria-label="Vertex capacity transformation">
    <p className="nf-eyebrow">TRANSFORM · A VERTEX BECOMES A GATE</p>
    <div className="nf-split-flow">
      <div>All incoming edges<span>arrive at v-in</span></div>
      <b aria-hidden="true">→</b>
      <div className="nf-split-gate">v-in → v-out<span>capacity 2</span></div>
      <b aria-hidden="true">→</b>
      <div>All outgoing edges<span>leave v-out</span></div>
    </div>
    <figcaption>Every unit passing through the original internal vertex must cross its one middle edge. That edge enforces a shared total of 2; placing capacity 2 on each incoming edge would permit more than 2 in total.</figcaption>
  </figure>;
}
export function BinaryCutLab() {
  const [penalty, setPenalty] = useState(2);
  const [labels, setLabels] = useState([false, false, true, false, true, true]);
  const current = pixelEnergy(labels, penalty);
  const optimum = useMemo(() => solvePixelCut(penalty), [penalty]);
  return <section className="nf-lab" aria-label="Binary labeling minimum cut investigation">
    <p className="nf-eyebrow">INVESTIGATE · A CUT CHOOSES TWO LABELS</p>
    <h3>Balance local preference and shared boundaries</h3>
    <p>Press a cell to switch background B / foreground F. Its two numbers are costs, not probabilities. Neighbor penalty applies once to every horizontal or vertical pair with different labels.</p>
    <label className="nf-penalty-control">Penalty per disagreeing neighbor pair<select aria-label="Neighbor disagreement penalty" value={penalty} onChange={event => setPenalty(Number(event.target.value))}>{[0, 1, 2, 3, 4, 5].map(value => <option key={value}>{value}</option>)}</select></label>
    <div className="nf-pixel-grid">{labels.map((foreground, vertex) => <button key={vertex} className={foreground ? 'nf-pixel nf-foreground' : 'nf-pixel'} aria-label={`Cell ${vertex}, ${foreground ? 'foreground' : 'background'}; switch label`} aria-pressed={foreground} onClick={() => setLabels(previous => previous.map((value, i) => i === vertex ? !value : value))}>
        <span>Cell {vertex}</span>
        <strong>{foreground ? 'F' : 'B'}</strong>
        <span>B:{PIXEL_BACKGROUND_COST[vertex]} · F:{PIXEL_FOREGROUND_COST[vertex]}</span>
      </button>)}</div>
    <p className="nf-result" role="status">Current energy = unary {current.unary} + boundaries {current.boundary.length} × penalty {penalty} = <strong>{current.total}</strong>.</p>
    <p>Disagreeing pairs: {current.boundary.length ? current.boundary.map(([u, v]) => `${u}–${v}`).join(', ') : 'none'}. The same seven fixed neighbor pairs are used for every calculation.</p>
    <div className="nf-buttons">
      <button onClick={() => setLabels(optimum.labels.slice())}>Apply a minimum-cut labeling</button>
      <button onClick={() => {
        setPenalty(2);
        setLabels([false, false, true, false, true, true]);
      }}>Reset labeling</button>
    </div>
    <p className="nf-result">Proven minimum {optimum.flow.value}. Current excess cost {current.total - optimum.flow.value}. {current.total === optimum.flow.value ? 'This labeling is optimal; another tie may also be optimal.' : 'A lower-cost labeling exists under this stated objective.'}</p>
    <details>
      <summary>Inspect the cut-to-energy correspondence</summary>
      <div className="nf-table-scroll" tabIndex={0} role="region" aria-label="Detailed flow data"><table>
          <caption>Cell costs become terminal-edge capacities</caption>
          <thead><tr>
              <th>Cell</th>
              <th>S → cell</th>
              <th>cell → T</th>
              <th>Current label pays</th>
            </tr></thead>
          <tbody>{labels.map((foreground, vertex) => <tr key={vertex}>
              <th>{vertex}</th>
              <td>{PIXEL_BACKGROUND_COST[vertex]}</td>
              <td>{PIXEL_FOREGROUND_COST[vertex]}</td>
              <td>{foreground ? `F: ${PIXEL_FOREGROUND_COST[vertex]}` : `B: ${PIXEL_BACKGROUND_COST[vertex]}`}</td>
            </tr>)}</tbody>
        </table></div>
      <p>Each neighbor pair {PIXEL_NEIGHBORS.map(([u, v]) => `${u}–${v}`).join(', ')} becomes two opposite original edges of capacity {penalty}. Exactly one crosses out of the source side when labels differ; zero cross when labels agree.</p>
    </details>
    <p className="nf-note">Calculated exactly from the displayed six-cell integer objective, independently checked against all 64 labelings. This is a small optimization model, not measured image-segmentation accuracy. Larger penalties favor fewer boundaries but need not improve an actual image label.</p>
  </section>;
}
