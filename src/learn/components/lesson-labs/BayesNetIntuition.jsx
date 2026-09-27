import './bayesnet-intuition.css';

function CausalNode({ x, y, name, fixed = false }) {
  return <g><circle cx={x} cy={y} r="18" fill="#171717" stroke={fixed ? '#e8b84b' : '#ddd'} strokeWidth="2" /><text x={x} y={y + 5} textAnchor="middle">{name}</text></g>;
}

function CausalArrow({ from, to, removed = false }) {
  const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
  const unit = [Math.cos(angle), Math.sin(angle)];
  const start = from.map((value, axis) => value + 21 * unit[axis]);
  const end = to.map((value, axis) => value - 21 * unit[axis]);
  const base = end.map((value, axis) => value - 8 * unit[axis]);
  return <g><line x1={start[0]} y1={start[1]} x2={end[0]} y2={end[1]} stroke={removed ? '#777' : '#ddd'} strokeWidth="2" strokeDasharray={removed ? '4 5' : undefined} />
    {removed ? <text x={(from[0] + to[0]) / 2} y={(from[1] + to[1]) / 2 - 9} textAnchor="middle" className="bn-intuition-cut">× cut</text> : <polygon points={`${end} ${base[0] - 4 * unit[1]},${base[1] + 4 * unit[0]} ${base[0] + 4 * unit[1]},${base[1] - 4 * unit[0]}`} fill="#ddd" />}</g>;
}

export function DoCalculusSurgeryFigure() {
  return <figure className="bn-intuition" data-concept="do-calculus-graph-surgery">
    <figcaption><strong>Each rule asks a different question in a specifically edited graph.</strong></figcaption>
    <p>Three separate causal models, each with independent external noises and the required probability support. Amber marks an already fixed X. Dashed, crossed edges are deleted for the displayed test.</p>
    <div className="bn-intuition-panels">
      <section><strong>1. Can an observation be ignored?</strong>
        <svg viewBox="0 0 260 185" role="img" aria-label="X points to Y and Z; the fixed X blocks the path Y back through X to Z">
          <CausalArrow from={[130, 40]} to={[65, 135]} /><CausalArrow from={[130, 40]} to={[195, 135]} />
          <CausalNode x={130} y={40} name="X" fixed /><CausalNode x={65} y={135} name="Y" /><CausalNode x={195} y={135} name="Z" />
        </svg>
        <p>Original model: X→Y and X→Z. There are no incoming X arrows to cut. Given X, the fork Y←X→Z is blocked.</p><p className="bn-intuition-result">P(y | do(x), z) = P(y | do(x))</p>
        <p>Observing a second consequence of the already fixed X adds no information about Y in this model.</p>
      </section>
      <section><strong>2. Can action become observation?</strong>
        <svg viewBox="0 0 260 185" role="img" aria-label="The sole arrow from X to Y is removed for the rule two graphical test">
          <CausalArrow from={[60, 100]} to={[200, 100]} removed /><CausalNode x={60} y={100} name="X" /><CausalNode x={200} y={100} name="Y" />
        </svg>
        <p>Original model: X→Y only. For this test cut X's outgoing arrow. No route between X and Y remains.</p><p className="bn-intuition-result">P(y | do(x)) = P(y | x)</p>
        <p>The cut is a test for exchanging the expressions; it does not claim that the real effect X→Y vanishes. A hidden common cause would leave a route and invalidate this argument.</p>
      </section>
      <section><strong>3. Can an extra action be ignored?</strong>
        <svg viewBox="0 0 260 185" role="img" aria-label="In Z to X to Y, fixing X cuts Z to X; Z is then disconnected from Y">
          <CausalArrow from={[35, 100]} to={[130, 100]} removed /><CausalArrow from={[130, 100]} to={[225, 100]} />
          <CausalNode x={35} y={100} name="Z" /><CausalNode x={130} y={100} name="X" fixed /><CausalNode x={225} y={100} name="Y" />
        </svg>
        <p>Original model: Z→X→Y. Fixing X cuts Z→X. There are no incoming Z arrows or extra W observations; Z is separated from Y.</p><p className="bn-intuition-result">P(y | do(x), do(z)) = P(y | do(x))</p>
        <p>Once X is fixed, changing the variable that used to determine X cannot travel through that replaced mechanism.</p>
      </section>
    </div>
    <p>These small cases explain the operations. In a larger graph, use the full rule—including its conditioning set and, for rule 3, the ancestor restriction—rather than matching only a familiar-looking fragment.</p>
  </figure>;
}

export function JunctionMessageFigure() {
  const left = [[2, 1], [1, 3]];
  const right = [[4, 1], [1, 2]];
  const message = [0, 1].map(b => left[0][b] + left[1][b]);
  const marginal = [0, 1].map(c => message.reduce((sum, weight, b) => sum + weight * right[b][c], 0));
  return <figure className="bn-intuition" data-concept="junction-tree-separator-message">
    <figcaption><strong>A message preserves every way the other cluster can be affected.</strong></figcaption>
    <p>New constructed model: P(A,B,C) is proportional to φ(A,B)ψ(B,C). Both factors are unnormalized. The clusters {'{A,B}'} and {'{B,C}'} share only B, so their separator is {'{B}'}.</p>
    <div className="bn-message-stages">
      <section><strong>Left cluster: {'{A,B}'}</strong><p>φ rows are A; columns are B.</p>
        <table><caption>Factor φ</caption><thead><tr><th>A \ B</th><th>0</th><th>1</th></tr></thead><tbody>{left.map((row, a) => <tr key={a}><th>{a}</th>{row.map((value, b) => <td key={b}>{value}</td>)}</tr>)}</tbody></table>
      </section>
      <section className="bn-message-separator"><strong>Send a function of B →</strong><p>Sum out A separately for each B.</p><p>B=0: 2+1 = <b>{message[0]}</b><br />B=1: 1+3 = <b>{message[1]}</b></p><p>Keep both entries. A single total 7 would erase which B was supported.</p></section>
      <section><strong>Right cluster: {'{B,C}'}</strong><p>ψ rows are B; columns are C.</p>
        <table><caption>Factor ψ</caption><thead><tr><th>B \ C</th><th>0</th><th>1</th></tr></thead><tbody>{right.map((row, b) => <tr key={b}><th>{b}</th>{row.map((value, c) => <td key={c}>{value}</td>)}</tr>)}</tbody></table>
      </section>
    </div>
    <p>At the receiving cluster, multiply each B row by its incoming message, then sum B:<br />C=0: 3·4+4·1 = <strong>{marginal[0]}</strong><br />C=1: 3·1+4·2 = <strong>{marginal[1]}</strong>.</p>
    <p className="bn-intuition-result">Total mass = {marginal[0] + marginal[1]}; P(C=1) = 11/27≈{(marginal[1] / (marginal[0] + marginal[1])).toFixed(6)}.</p>
    <p>The message already accounts for A, so the receiving cluster never needs to enumerate A again. A reverse message would similarly sum out C. In a larger junction tree, the running-intersection property ensures that a variable absent from the separator cannot reappear elsewhere beyond that separator.</p>
  </figure>;
}
