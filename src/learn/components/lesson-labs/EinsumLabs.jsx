import { useState } from 'react';
import { attentionFixtures, attentionRow, basisChange, basisPresets, basisVectors, contractionOrders, contractionPresets, einsumNumber as number, inspectContraction } from '../../data/einsum-models.js';
import './einsum-labs.css';
const pair = values => '(' + values.map(number).join(', ') + ')';
function Matrix({
  values,
  title,
  selected = []
}) {
  const rows = Array.isArray(values[0]) ? values : [values];
  return <div className="einsum-matrix-scroll" tabIndex={0} role="region" aria-label={title}>
    <table className="einsum-matrix"><caption>{title}</caption><tbody>{rows.map((row, i) => <tr key={i}>{row.map((value, j) => {
            const active = selected.some(coordinates => coordinates.length === 1 ? coordinates[0] === j : coordinates[0] === i && coordinates[1] === j);
            return <td key={j} className={active ? 'is-selected' : ''}>{number(value)}</td>;
          })}</tr>)}</tbody></table>
  </div>;
}
function Lab({
  id,
  title,
  children
}) {
  return <section className="einsum-lab" data-lab={id} aria-label={title}><h3>{title}</h3>{children}</section>;
}
export function IndexContractionLab() {
  const [preset, setPreset] = useState('product');
  const [expression, setExpression] = useState(contractionPresets.product.expression);
  const [applied, setApplied] = useState(contractionPresets.product.expression);
  const [selectedCell, setSelectedCell] = useState(0);
  const [error, setError] = useState('');
  const operands = contractionPresets[preset].operands;
  const state = inspectContraction(applied, operands);
  const cell = state.cells[selectedCell];
  function choosePreset(key) {
    setPreset(key);
    setExpression(contractionPresets[key].expression);
    setApplied(contractionPresets[key].expression);
    setSelectedCell(0);
    setError('');
  }
  function apply(event) {
    event.preventDefault();
    try {
      inspectContraction(expression, operands);
      setApplied(expression);
      setSelectedCell(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  return <Lab id="index-contraction" title="Follow every contribution to one output">
    <p>Choose an operation, predict an entry, then select that output. Highlighted source cells supply its products. The inspector supports the displayed one/two operands, lower-case labels and explicit output; ellipses are taught separately below.</p>
    <label>Starting operation<select value={preset} onChange={event => choosePreset(event.target.value)}>{Object.entries(contractionPresets).map(([key, value]) => <option key={key} value={key}>{value.name}</option>)}</select></label>
    <form onSubmit={apply} className="einsum-expression"><label>Explicit expression<input value={expression} maxLength={30} onChange={event => setExpression(event.target.value)} spellCheck={false} /></label><button type="submit">Inspect expression</button><button type="button" onClick={() => choosePreset('product')}>Reset inspector</button></form>
    {error && <p role="alert">{error} The last valid result remains below.</p>}
    <div className="einsum-operands">{operands.map((operand, index) => <Matrix key={index} values={operand} title={'Operand ' + (index + 1) + ' · axes ' + state.inputs[index]} selected={cell.terms.flatMap(term => term.sources.filter(source => source.operand === index).map(source => source.coordinates))} />)}</div>
    <p className="einsum-equation" data-result="contract">{state.expression} · output shape {pair(state.outputShape)} · summed labels: {state.reduced.join(', ') || 'none'}</p>
    <div className="einsum-output" role="group" aria-label="Choose output entry">{state.cells.map((entry, index) => <button type="button" key={index} aria-pressed={selectedCell === index} onClick={() => setSelectedCell(index)}>{entry.coordinates.length ? pair(entry.coordinates) : 'scalar'} = {number(entry.value)}</button>)}</div>
    <p><strong>Selected output {cell.coordinates.length ? pair(cell.coordinates) : 'scalar'}:</strong> {cell.terms.length} contribution{cell.terms.length === 1 ? '' : 's'}. One operand contributes its selected value; multiple operands multiply their selected values.</p>
    <ol className="einsum-terms">{cell.terms.map((term, index) => <li key={index}><span>{Object.entries(term.indices).map(([label, value]) => label + '=' + value).join(', ')}</span><strong>{term.sources.map(source => 'A' + (source.operand + 1) + '[' + source.coordinates.join(',') + ']=' + number(source.value)).join(' × ')}</strong><span>Contribution: {number(term.product)}</span></li>)}</ol>
    <p data-result="sum">Sum: {cell.terms.map(term => number(term.product)).join(' + ')} = <strong>{number(cell.value)}</strong></p>
    <details><summary>Transfer: keep k in the matrix-product output</summary><p>Choose Matrix product, then inspect <code>ik,kj-&gt;ikj</code>. The shape is (2,2,2). Each cell is a single product; k is retained. Summing that result over its middle axis recovers the matrix product. A repeated input label can survive.</p></details>
  </Lab>;
}
export function DiagonalReductionFigure() {
  return <figure className="einsum-inline" aria-label="Select the diagonal before deciding whether to sum">
    <Matrix values={[[1, 2], [3, 4]]} title="A · only equal row/column indices" selected={[[0, 0], [1, 1]]} />
    <div className="einsum-diagonal-routes"><p><code>ii-&gt;i</code><span> keep i → </span><strong>[1, 4]</strong></p><p><code>ii-&gt;</code><span> sum i → </span><strong>1 + 4 = 5</strong></p></div>
    <figcaption>The repeated i selects A[0,0] and A[1,1]. The explicit output decides whether these two values remain separate.</figcaption>
  </figure>;
}
export function BatchPairingFigure() {
  return <figure className="einsum-inline" aria-label="Same-batch comparisons versus every pair of batches">
    <div className="einsum-batch-pairs"><div><strong>bi,bi→b · matching batches</strong><p>X batch 0 ↔ Y batch 0 → 50</p><p>X batch 1 ↔ Y batch 1 → 250</p></div><div><strong>bi,ci→bc · all batch pairs</strong><Matrix values={[[50, 110], [110, 250]]} title="Rows: X batch b · columns: Y batch c" selected={[[0, 0], [1, 1]]} /></div></div>
    <figcaption>X=[[1,2],[3,4]], Y=[[10,20],[30,40]]. The matching results are the diagonal of the all-pairs table. Both are valid computations; only one may answer your question.</figcaption>
  </figure>;
}
export function AttentionContractionLab() {
  const [batch, setBatch] = useState(0);
  const [query, setQuery] = useState(0);
  const [allowed, setAllowed] = useState([true, true, true]);
  const [scaled, setScaled] = useState(true);
  let state;
  let error;
  try {
    state = attentionRow(batch, query, allowed, scaled);
  } catch (failure) {
    error = failure.message;
  }
  function reset() {
    setBatch(0);
    setQuery(0);
    setAllowed([true, true, true]);
    setScaled(true);
  }
  return <Lab id="attention-contraction" title="From one query to a weighted value">
    <p>These are small invented, untrained vectors. Select a query and decide which keys it may use. Inspect which weights change when a key is blocked. Every row below belongs to the selected batch; no other batch contributes.</p>
    <div className="einsum-controls"><label>Batch<select value={batch} onChange={event => setBatch(Number(event.target.value))}><option value="0">0</option><option value="1">1</option></select></label><label>Query position<select value={query} onChange={event => setQuery(Number(event.target.value))}><option value="0">0</option><option value="1">1</option></select></label></div>
    <label className="einsum-checkbox"><input type="checkbox" checked={scaled} onChange={event => setScaled(event.target.checked)} />Divide scores by √2</label>
    <fieldset><legend>Allowed key positions</legend>{allowed.map((value, key) => <label className="einsum-checkbox" key={key}><input type="checkbox" checked={value} onChange={() => setAllowed(previous => previous.map((item, index) => index === key ? !item : item))} />Key {key}</label>)}</fieldset>
    <p>Query: {pair(attentionFixtures[batch].queries[query])}. Feature products are summed over d; the final weighted values are summed over key position s.</p>
    {error ? <p role="alert">{error}</p> : <>
      <div className="einsum-attention-rows">{state.rows.map(row => <div className="einsum-attention-row" key={row.key} data-key={row.key}>
        <strong>Key {row.key}: {pair(attentionFixtures[batch].keys[row.key])}</strong>
        <p>Dot product: {row.products.map(number).join(' + ')} = {number(row.products.reduce((sum, value) => sum + value, 0))}</p>
        <p>Score: {number(row.score)}{allowed[row.key] ? '' : ' · blocked before softmax'}</p>
        <div className="einsum-weight" aria-label={'Key ' + row.key + ' weight ' + number(row.weight)}><span style={{
              width: row.weight * 100 + '%'
            }} /></div>
        <p>Weight {number(row.weight)} × value {pair(row.value)}</p><strong>Contribution {pair(row.contribution)}</strong>
      </div>)}</div>
      <p className="einsum-equation" data-result="context">Context = {pair(state.context)} · weight sum = {number(state.rows.reduce((sum, row) => sum + row.weight, 0))}</p>
      <p>The bars share the range 0–1. For an allowed key, weight = exp(score − largest allowed score) / {number(state.denominator)}. A blocked key has weight 0. Subtracting that common maximum preserves the allowed-key ratios and reduces overflow risk for these finite scores.</p>
    </>}
    <button type="button" onClick={reset}>Reset attention</button>
    <details><summary>Transfer: only key 2 is allowed</summary><p>Its weight becomes 1 and the context equals that batch's value at key 2, regardless of the query's score. Blocking all keys removes the normalization denominator; this lab reports an error rather than inventing a probability distribution.</p></details>
  </Lab>;
}
export function ContractionOrderLab() {
  const [dimensions, setDimensions] = useState([5, 40, 2, 30]);
  const [draft, setDraft] = useState(['5', '40', '2', '30']);
  const [error, setError] = useState('');
  const state = contractionOrders(dimensions);
  const maximum = Math.max(state.left.total, state.right.total);
  function apply(event) {
    event.preventDefault();
    try {
      if (draft.some(value => value.trim() === '')) throw new Error('Fill all four axis lengths.');
      const parsed = draft.map(Number);
      contractionOrders(parsed);
      setDimensions(parsed);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDimensions([5, 40, 2, 30]);
    setDraft(['5', '40', '2', '30']);
    setError('');
  }
  return <Lab id="contraction-order" title="Same equation, different intermediate work">
    <p>Compute A(a,b) B(b,c) C(c,d) → result(a,d). Inspect which pair to multiply first. These counts use the conventional dense algorithm: one scalar multiplication for each output entry and contracted-index value. They are not measured timings.</p>
    <form noValidate onSubmit={apply}><div className="einsum-controls">{['a', 'b', 'c', 'd'].map((label, index) => <label key={label}>Axis {label}<input type="number" min="1" max="100" step="1" value={draft[index]} onChange={event => setDraft(previous => previous.map((value, position) => position === index ? event.target.value : value))} /></label>)}</div><button type="submit">Compare orders</button><button type="button" onClick={reset}>Reset dimensions</button></form>
    {error && <p role="alert">{error} The last valid comparison remains below.</p>}
    <div className="einsum-order-comparison">{[['left', '(A B) C', 'A × B', 'temporary × C'], ['right', 'A (B C)', 'B × C', 'A × temporary']].map(([key, title, first, last]) => {
        const route = state[key];
        return <div key={key} className="einsum-order-route"><h4>{title}</h4><p>{first}</p><div className="einsum-intermediate"><strong>Temporary {pair(route.firstShape)}</strong><p>{route.firstSize.toLocaleString()} entries<br />{route.firstCost.toLocaleString()} multiplications</p></div><p>↓ {last}</p><p>Output {pair(state.outputShape)}<br />{route.lastCost.toLocaleString()} more multiplications</p><div className="einsum-weight" aria-label={title + ' scalar multiplication count ' + route.total}><span style={{
              width: 100 * route.total / maximum + '%'
            }} /></div><strong data-result={key + '-cost'}>{route.total.toLocaleString()} total multiplications</strong></div>;
      })}</div>
    <p>Bar lengths share a zero baseline and are proportional to the calculated counts. First-temporary size excludes input arrays, final output and library workspace; it is not a whole-program peak-memory measurement.</p>
    <details><summary>Transfer: can the preferred order reverse?</summary><p>Try a=40,b=5,c=30,d=2. Right-first uses 700 multiplications; left-first uses 8400. Axis lengths, not the spelling of the equation, determine this comparison.</p></details>
  </Lab>;
}
export function TensorBasisLab() {
  const [basis, setBasis] = useState('shear');
  const [vectorIndex, setVectorIndex] = useState(0);
  const vector = basisVectors[vectorIndex];
  const state = basisChange(basis, vector);
  const first = state.S.map(row => row[0] * state.coordinates[0]);
  const points = [[0, 0], vector, first, ...[0, 1].map(j => state.S.map(row => row[j]))];
  const extent = Math.max(3, ...points.flat().map(Math.abs)) + 1;
  const x = value => 165 + 118 * value / extent;
  const y = value => 154 - 118 * value / extent;
  const line = (start, end, className) => <line x1={x(start[0])} y1={y(start[1])} x2={x(end[0])} y2={y(end[1])} className={className} />;
  return <Lab id="tensor-basis" title="Change the coordinates while keeping the object">
    <p>The old axes are orthonormal. Columns of S specify the new basis in that old plane. Gold is the actual vector v. Dashed segments add its new-basis components; the final endpoint must remain v.</p>
    <div className="einsum-controls"><label>New basis<select value={basis} onChange={event => setBasis(event.target.value)}>{Object.keys(basisPresets).map(key => <option key={key} value={key}>{key}</option>)}</select></label><label>Old vector coordinates<select value={vectorIndex} onChange={event => setVectorIndex(Number(event.target.value))}>{basisVectors.map((value, index) => <option key={index} value={index}>{pair(value)}</option>)}</select></label></div>
    <div className="einsum-basis-layout"><svg viewBox="0 0 330 310" role="img" aria-label={'Same vector ' + pair(vector) + ', new components ' + pair(state.coordinates)} className="einsum-basis-plot">
      {[-extent, 0, extent].map(value => <g key={value}><line x1={x(value)} y1="36" x2={x(value)} y2="272" className="grid" /><line x1="47" y1={y(value)} x2="283" y2={y(value)} className="grid" /></g>)}
      <text x="10" y="20">Old coordinates · same scale on x/y</text>
      {line([0, 0], state.S.map(row => row[0]), 'basis-one')}
      {line([0, 0], state.S.map(row => row[1]), 'basis-two')}
      <text x={x(state.S[0][0]) + 8} y={y(state.S[1][0]) + 20}>b1</text>
      <text x={x(state.S[0][1]) + 8} y={y(state.S[1][1]) - 10}>b2</text>
      {line([0, 0], first, 'component')}
      {line(first, vector, 'component')}
      {line([0, 0], vector, 'world-vector')}
      <circle cx={x(vector[0])} cy={y(vector[1])} r="5" className="endpoint" />
      <text x="48" y="294">x: −{extent} to {extent} · y: −{extent} to {extent}</text><text x={x(0) + 5} y={y(0) + 17}>0</text>
    </svg><div><Matrix values={state.S} title="S · new basis columns" /><p>New coordinates: <strong>{pair(state.coordinates)}</strong></p><p>{number(state.coordinates[0])} × basis 1 {pair(state.S.map(row => row[0]))}<br />+ {number(state.coordinates[1])} × basis 2 {pair(state.S.map(row => row[1]))}<br />= {pair(state.reconstructed)}</p></div></div>
    <div className="einsum-invariants" data-result="basis"><p>Functional f=(2,−1) → f′={pair(state.newFunctional)}<br />Measurement: {number(state.measurement)} = {number(state.newMeasurement)}</p><p>Ordinary new-coordinate squares: {number(state.coordinateSquares)}<br />True squared length: {number(state.normSquared)} = metric expression {number(state.metricNormSquared)}</p></div>
    <Matrix values={state.metric} title="G′ = SᵀS · new metric" />
    <p>The metric supplies the cross-terms missing from an ordinary coordinate-square sum in an oblique basis. A zero vector has zero length in every basis; unlike an eigenvector, it is allowed here.</p>
    <button type="button" onClick={() => {
      setBasis('shear');
      setVectorIndex(0);
    }}>Reset basis</button>
  </Lab>;
}
