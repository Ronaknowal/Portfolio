import { useMemo, useState } from 'react';
import { matrixChainPlan, balloonPlan, treeBoundaryPlan, treeBoundaryWitness, treePresetEdges, createDigitCounter, defaultChainDimensions, defaultTreeWeights } from '../../data/dp-state-families-models.js';
import './dp-state-families-labs.css';
function parseIntegers(text) {
  if (!text.trim()) return [];
  return text.split(',').map(part => {
    if (!/^\s*-?\d+\s*$/.test(part)) throw new Error('Use comma-separated integers without empty entries.');
    const value = Number(part);
    if (!Number.isSafeInteger(value)) throw new Error('Every entry must be an exactly representable integer.');
    return value;
  });
}
function Shape({
  rows,
  columns,
  label
}) {
  return <span className="dpf-shape"><strong>{label}</strong><span>{rows} × {columns}</span></span>;
}
export function MatrixShapeFigure() {
  return <figure className="dpf-inline" aria-label="The same three matrices have grouping costs 480 and 120 scalar multiplications">
    <div className="dpf-shape-plan"><b>Make the wide intermediate first</b><div><Shape rows={8} columns={2} label="A0" /><span>×</span><Shape rows={2} columns={12} label="A1" /><span>→</span><Shape rows={8} columns={12} label="product" /></div><p>8·2·12 = <b>192</b>; then multiply by A2 (12×3): 8·12·3 = <b>288</b>. Total <strong>480</strong>.</p></div>
    <div className="dpf-shape-plan dpf-emphasis"><b>Keep the intermediate narrow</b><div><Shape rows={2} columns={12} label="A1" /><span>×</span><Shape rows={12} columns={3} label="A2" /><span>→</span><Shape rows={2} columns={3} label="product" /></div><p>2·12·3 = <b>72</b>; then multiply A0 (8×2) by it: 8·2·3 = <b>48</b>. Total <strong>120</strong>.</p></div>
    <figcaption>Original factor order stays A0,A1,A2. Labeled shapes are schematic; their drawn area is not a timing or memory measurement.</figcaption>
  </figure>;
}
function ExpressionTree({
  model,
  left,
  right,
  selectedSplit
}) {
  const nodes = [];
  const edges = [];
  const width = Math.max(256, (right - left) * 64);
  let maximumDepth = 0;
  function visit(first, last, depth, forced = null) {
    maximumDepth = Math.max(maximumDepth, depth);
    const node = {
      first,
      last,
      x: ((first + last) / 2 - left) * width / (right - left),
      y: 30 + depth * 76
    };
    nodes.push(node);
    if (last > first + 1) {
      const split = forced ?? model.splits[first][last];
      for (const [childLeft, childRight] of [[first, split], [split, last]]) {
        const child = visit(childLeft, childRight, depth + 1);
        edges.push({
          parent: node,
          child
        });
      }
    }
    return node;
  }
  visit(left, right, 0, selectedSplit);
  return <div className="dpf-scroll" role="region" aria-label="Candidate expression tree" tabIndex={0}>
    <svg width={width} height={maximumDepth * 76 + 63} role="img" aria-label={`Ordered expression tree for matrices ${left} through ${right - 1}; leaves keep their original order and children are multiplied before their parent.`}>
      {edges.map(({
        parent,
        child
      }, index) => <path key={index} d={`M${parent.x},${parent.y + 23}L${child.x},${child.y - 23}`} className="dpf-edge" />)}
      {nodes.map(node => <g key={`${node.first},${node.last}`} transform={`translate(${node.x},${node.y})`}><rect x="-29" y="-23" width="58" height="46" rx="4" className={node.last === node.first + 1 ? 'dpf-leaf-node' : 'dpf-parent-node'} /><text textAnchor="middle" y="-4">{node.last === node.first + 1 ? `A${node.first}` : `[${node.first},${node.last})`}</text><text textAnchor="middle" y="13">{model.dimensions[node.first]}×{model.dimensions[node.last]}</text></g>)}
    </svg>
  </div>;
}
export function IntervalSplitLab() {
  const [dimensions, setDimensions] = useState(defaultChainDimensions);
  const [draft, setDraft] = useState(defaultChainDimensions.join(', '));
  const [interval, setInterval] = useState([0, 4]);
  const [split, setSplit] = useState(1);
  const [operation, setOperation] = useState(0);
  const [error, setError] = useState('');
  const model = useMemo(() => matrixChainPlan(dimensions), [dimensions]);
  const [left, right] = interval;
  const cell = model.cells.find(candidate => candidate.left === left && candidate.right === right);
  const candidate = cell.candidates.find(candidate => candidate.split === split);
  const activeOperation = model.operations[operation];
  function chooseInterval(first, last) {
    setInterval([first, last]);
    setSplit(model.splits[first][last]);
  }
  function apply(event) {
    event.preventDefault();
    try {
      const values = parseIntegers(draft);
      const next = matrixChainPlan(values);
      setDimensions(values);
      setInterval([0, next.count]);
      setSplit(next.splits[0][next.count]);
      setOperation(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDimensions(defaultChainDimensions);
    setDraft(defaultChainDimensions.join(', '));
    setInterval([0, 4]);
    setSplit(1);
    setOperation(0);
    setError('');
  }
  return <section className="dpf-lab" aria-label="Interval split investigation">
    <h3>Where should the final multiplication split?</h3>
    <p>Predict which grouping avoids an expensive intermediate. Every interval answer includes the best grouping inside that interval; its boundary dimensions remain fixed.</p>
    <form className="dpf-controls" onSubmit={apply}><label>Matrix-chain dimensions<input value={draft} onChange={event => setDraft(event.target.value)} /></label><button>Apply dimensions</button><button type="button" onClick={reset}>Reset interval lab</button></form>
    {error && <p role="alert">{error}</p>}
    <div className="dpf-factor-row">{dimensions.slice(0, -1).map((rows, index) => <Shape key={index} rows={rows} columns={dimensions[index + 1]} label={`A${index}`} />)}</div>
    <p>Active dimensions: [{dimensions.join(', ')}]. All costs below count ordinary scalar multiplications.</p>
    <div className="dpf-scroll" role="region" aria-label="Interval cost table" tabIndex={0}><table><caption>Choose a cell · row i, column j describes matrices [i,j). Fill shorter spans first.</caption><thead><tr><th>i \ j</th>{Array.from({
              length: model.count
            }, (_, index) => <th key={index}>{index + 1}</th>)}</tr></thead><tbody>{Array.from({
            length: model.count
          }, (_, first) => <tr key={first}><th>{first}</th>{Array.from({
              length: model.count
            }, (_, index) => <td key={index}>{index + 1 > first ? <button aria-label={`Inspect interval ${first} to ${index + 1}`} aria-pressed={left === first && right === index + 1} onClick={() => chooseInterval(first, index + 1)}>{model.costs[first][index + 1]}</button> : '—'}</td>)}</tr>)}</tbody></table></div>
    <h4>Inspect [<span>{left},{right}</span>) · result shape {dimensions[left]}×{dimensions[right]}</h4>
    {cell.candidates.length ? <><div className="dpf-controls"><label>Candidate final split<select aria-label="Candidate final split" value={split} onChange={event => setSplit(Number(event.target.value))}>{cell.candidates.map(item => <option key={item.split} value={item.split}>k={item.split} · {item.total}{item.split === cell.split ? ' (chosen)' : ''}</option>)}</select></label></div><p className="dpf-result" data-dpf-status>Left {candidate.first} + right {candidate.second} + final {candidate.merge} = <strong>{candidate.total}</strong>. Best interval cost: <strong>{cell.cost}</strong>.</p><div className="dpf-split-band"><span>left [{left},{split})<b>{dimensions[left]}×{dimensions[split]}</b></span><span>×</span><span>right [{split},{right})<b>{dimensions[split]}×{dimensions[right]}</b></span></div><ExpressionTree model={model} left={left} right={right} selectedSplit={split} /><p>Children use their optimal plans. This tree inspects one candidate for the selected interval; changing the inspected split does not change the computed whole-chain optimum.</p></> : <p className="dpf-result" data-dpf-status>One existing matrix: cost 0, no multiplication and no split.</p>}
    <h4>Recover the whole-chain optimum</h4><p className="dpf-expression">{model.expression}</p>
    {activeOperation ? <><div className="dpf-controls"><button disabled={operation === 0} onClick={() => setOperation(operation - 1)}>Previous multiplication</button><button disabled={operation === model.operations.length - 1} onClick={() => setOperation(operation + 1)}>Next multiplication</button><span>{operation + 1}/{model.operations.length}</span></div><ol className="dpf-operation-list">{model.operations.map((item, index) => <li key={index} aria-current={index === operation ? 'step' : undefined}>[{item.left},{item.split}) × [{item.split},{item.right}) → [{item.left},{item.right})<b>{item.cost} multiplications</b></li>)}</ol><p>At operation {operation + 1}, both child products already exist. All operation costs sum to <strong>{model.result}</strong>. The tree's final multiplication is executed last.</p></> : <p>The only matrix already exists; there are no operations to recover.</p>}
    <p className="dpf-note">Try dimensions 3,7,2,5 and explain why the winning side changes. Equal dimensions 2,2,2,2 produce a tie; the first split is retained. No wall-clock speed or floating-point accuracy comparison is claimed.</p>
  </section>;
}
export function BalloonLastFigure() {
  const model = balloonPlan();
  return <figure className="dpf-inline" aria-label="Balloon removal snapshots for values two, four, three; original IDs one, zero, two earn 24, six, three">
    <p><strong>The split-tree root is the last removal.</strong> The surviving boundary sentinels always have value 1.</p>
    <ol className="dpf-balloon-replay">{model.replay.map((step, index) => <li key={index}><div className="dpf-balloon-row"><span className="dpf-sentinel">1<small>left</small></span>{step.live.map(balloon => <span key={balloon.id} className="dpf-balloon">{balloon.value}<small>ID {balloon.id}</small></span>)}<span className="dpf-sentinel">1<small>right</small></span></div><span>{index === 0 ? 'Initial row' : `Remove ID ${step.removed}: ${step.first}·${step.value}·${step.second} = ${step.earned}; total ${step.total}`}</span></li>)}</ol>
    <figcaption>The whole interval's chosen last object is ID 2. Its left child interval removes ID 1 and then ID 0. Reading a split tree root-first would give the wrong chronological order.</figcaption>
  </figure>;
}
function treeGeometry(plan) {
  let leaves = 0;
  const positions = {};
  function place(node) {
    const children = plan.children[node];
    const childPositions = children.map(place);
    const x = children.length ? (childPositions[0] + childPositions.at(-1)) / 2 : 32 + 64 * leaves++;
    positions[node] = {
      x,
      y: 34 + 76 * plan.depth[node]
    };
    return x;
  }
  if (plan.count) place(plan.root);
  return {
    positions,
    width: Math.max(256, leaves * 64),
    height: Math.max(90, (Math.max(0, ...plan.depth) + 1) * 76)
  };
}
export function TreeBoundaryLab() {
  const [weights, setWeights] = useState(defaultTreeWeights);
  const [draft, setDraft] = useState(defaultTreeWeights.join(', '));
  const [shape, setShape] = useState('binary');
  const [node, setNode] = useState(1);
  const [parentSelected, setParentSelected] = useState(false);
  const [error, setError] = useState('');
  const plan = useMemo(() => treeBoundaryPlan(weights, treePresetEdges(weights.length, shape)), [weights, shape]);
  const witness = useMemo(() => treeBoundaryWitness(plan, node, parentSelected), [plan, node, parentSelected]);
  const geometry = useMemo(() => treeGeometry(plan), [plan]);
  const activeIndex = plan.postorder.indexOf(node);
  const requested = new Map(witness.requests.map(request => [request.node, request]));
  function apply(event) {
    event.preventDefault();
    try {
      const values = parseIntegers(draft);
      treeBoundaryPlan(values, treePresetEdges(values.length, shape));
      setWeights(values);
      setNode(values.length > 1 ? 1 : 0);
      setParentSelected(false);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setWeights(defaultTreeWeights);
    setDraft(defaultTreeWeights.join(', '));
    setShape('binary');
    setNode(1);
    setParentSelected(false);
    setError('');
  }
  return <section className="dpf-lab" aria-label="Tree parent boundary investigation">
    <h3>Which answer can cross the parent edge?</h3><p>For the selected subtree, compare a free root with a root whose external parent is selected. The parent's weight is outside this question. Green ✓ nodes form the conditional witness; dim nodes are outside this subtree.</p>
    <form className="dpf-controls" onSubmit={apply}><label>Node weights in ID order<input value={draft} onChange={event => setDraft(event.target.value)} /></label><button>Apply tree weights</button><button type="button" onClick={reset}>Reset tree lab</button></form>
    <div className="dpf-controls"><label>Tree shape<select aria-label="Tree shape" value={shape} onChange={event => {
          setShape(event.target.value);
          setNode(weights.length > 1 ? 1 : 0);
        }}><option value="binary">Binary</option><option value="chain">Chain</option><option value="star">Star</option></select></label>{plan.count > 0 && <label>Subtree root<select aria-label="Subtree root" value={node} onChange={event => setNode(Number(event.target.value))}>{weights.map((_, index) => <option key={index} value={index}>Node {index} · weight {weights[index]}</option>)}</select></label>}<label className="dpf-check"><input type="checkbox" checked={parentSelected} onChange={event => setParentSelected(event.target.checked)} />External parent selected</label></div>
    {error && <p role="alert">{error}</p>}
    {!plan.count ? <p data-dpf-status className="dpf-result">Empty tree: value 0, no selected nodes.</p> : <><div className="dpf-scroll" role="region" tabIndex={0} aria-label="Weighted tree and conditional witness"><svg width={geometry.width} height={geometry.height} role="img" aria-label={`Input ${shape} tree, selected subtree ${node}, parent ${parentSelected ? 'selected' : 'not selected'}, chosen nodes ${witness.selected.join(', ') || 'none'}`}>
      {plan.edges.map(([first, second], index) => <path key={index} className={requested.has(first) && requested.has(second) ? 'dpf-edge dpf-active-edge' : 'dpf-edge'} d={`M${geometry.positions[first].x},${geometry.positions[first].y + 23}L${geometry.positions[second].x},${geometry.positions[second].y - 23}`} />)}
      {weights.map((weight, index) => <g key={index} transform={`translate(${geometry.positions[index].x},${geometry.positions[index].y})`} className={requested.has(index) ? '' : 'dpf-outside'}><rect x="-29" y="-23" width="58" height="48" rx="5" className={witness.selected.includes(index) ? 'dpf-selected-node' : index === node ? 'dpf-parent-node' : 'dpf-leaf-node'} /><text textAnchor="middle" y="-5">{`N${index}${witness.selected.includes(index) ? ' ✓' : ''}`}</text><text textAnchor="middle" y="13">w {weight}</text></g>)}
    </svg></div><div className="dpf-boundary-values"><div><span>Parent not selected</span><strong>F({node},0) = {plan.free[node]}</strong></div><div><span>Parent selected</span><strong>F({node},1) = {plan.blocked[node]}</strong></div></div><p data-dpf-status className="dpf-result">Requested F({node},{Number(parentSelected)}) = <strong>{witness.result}</strong>; selected IDs: {witness.selected.join(', ') || 'none'}.</p><p>Skip node {node}: sum child-free answers = {plan.blocked[node]}. Take it: weight {weights[node]} + child-forced-off answers = {plan.take[node]}.{parentSelected ? ' The parent condition forbids taking this root.' : ' Choose the larger value; skip ties.'}</p><div className="dpf-child-requests">{plan.children[node].map(child => <div key={child}><span>Child N{child}</span><span>If N{node} skipped: F({child},0)={plan.free[child]}</span><span>If N{node} taken: F({child},1)={plan.blocked[child]}</span></div>)}{!plan.children[node].length && <p>A leaf has no child contributions. Its skipped value is 0.</p>}</div>
    <h4>Inspect the completed table in child-before-parent order</h4><p>Postorder: {plan.postorder.join(' → ')}. These are already computed answers; the controls inspect their dependency order.</p><div className="dpf-controls"><button disabled={activeIndex === 0} onClick={() => setNode(plan.postorder[activeIndex - 1])}>Previous postorder node</button><button disabled={activeIndex === plan.postorder.length - 1} onClick={() => setNode(plan.postorder[activeIndex + 1])}>Next postorder node</button></div><div className="dpf-scroll" role="region" tabIndex={0} aria-label="Exact tree boundary values"><table><thead><tr><th>Node</th><th>Children</th><th>Free</th><th>Parent selected</th></tr></thead><tbody>{plan.postorder.map(current => <tr key={current} className={node === current ? 'dpf-current-row' : ''}><th>{current}</th><td>{plan.children[current].join(', ') || 'none'}</td><td>{plan.free[current]}</td><td>{plan.blocked[current]}</td></tr>)}</tbody></table></div><p>For the whole input tree with no external parent, F(0,0)={plan.result}. A conditional answer for a smaller subtree is not a different whole-tree optimum.</p></>}
    <p className="dpf-note">Use at most nine weights from −9 through 20; an empty list is allowed. Try a negative leaf and a star. Geometry shows only tree structure; an extra edge between child subtrees would invalidate this recurrence.</p>
  </section>;
}
export function DigitPaddingFigure() {
  return <figure className="dpf-inline" aria-label="Padding zero is not an actual digit, whereas zero after the number starts consumes digit zero">
    <div className="dpf-padding-row"><span><i>0</i><i>0</i><b>7</b></span><div><strong>007 → number 7</strong><p>Two padding positions; used digits {'{'}7{'}'}. Count this number once.</p></div></div><div className="dpf-padding-row"><span><b>1</b><b>0</b><b>2</b></span><div><strong>102 → valid</strong><p>Zero is real after starting; used digits {'{'}1,0,2{'}'}.</p></div></div><div className="dpf-padding-row"><span><b>1</b><b>0</b><b className="dpf-invalid-digit">0</b></span><div><strong>100 → rejected</strong><p>The second real zero repeats a used digit.</p></div></div><figcaption>Each fixed-width spelling corresponds to one integer. Padding does not create extra numbers or consume the actual digit zero.</figcaption>
  </figure>;
}
export function DigitPrefixLab() {
  const [bound, setBound] = useState(213);
  const [draft, setDraft] = useState('213');
  const [prefix, setPrefix] = useState('');
  const [error, setError] = useState('');
  const counter = useMemo(() => createDigitCounter(bound), [bound]);
  const model = useMemo(() => counter.inspect(prefix), [counter, prefix]);
  function apply(event) {
    event.preventDefault();
    try {
      if (!/^(0|[1-9]\d*)$/.test(draft.trim())) throw new Error('Use ordinary decimal notation, without signs, spaces inside the number or extra leading zeros.');
      const next = Number(draft.trim());
      createDigitCounter(next);
      setBound(next);
      setPrefix('');
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setBound(213);
    setDraft('213');
    setPrefix('');
    setError('');
  }
  const usedDigits = Array.from({
    length: 10
  }, (_, digit) => digit).filter(digit => model.state.used & 1 << digit);
  return <section className="dpf-lab" aria-label="Digit prefix investigation">
    <h3>Count all continuations of one prefix</h3><p>Predict the allowed final digits for prefixes 12 and 21 under bound 213. They used the same digits, but only 21 still equals the bound prefix.</p><form className="dpf-controls" onSubmit={apply}><label>Inclusive upper bound<input inputMode="numeric" value={draft} onChange={event => setDraft(event.target.value)} /></label><button>Apply digit bound</button><button type="button" onClick={reset}>Reset digit lab</button></form>{error && <p role="alert">{error}</p>}
    <p>Active bound {bound}; positive integers with all digits distinct: <strong>{model.total}</strong>.</p><div className="dpf-prefix-strip"><span>Bound</span><div>{model.digits.map((digit, index) => <b key={index}>{digit}</b>)}</div><span>Prefix</span><div>{model.digits.map((_, index) => <b key={index} className={index < prefix.length && !model.history[index + 1].started ? 'dpf-padding' : index === prefix.length ? 'dpf-cursor' : ''}>{prefix[index] ?? '·'}</b>)}</div></div>
    <div className="dpf-prefix-facts"><span>Next position <b>{model.state.position}/{model.digits.length}</b></span><span>Bound prefix <b>{model.state.tight ? 'equal · tight' : 'smaller · loose'}</b></span><span>Number <b>{model.state.started ? 'started' : 'not started'}</b></span></div><div className="dpf-digit-tray" aria-label={`Used digit set: ${usedDigits.join(', ') || 'empty'}`}>{Array.from({
        length: 10
      }, (_, digit) => <span key={digit} className={usedDigits.includes(digit) ? 'dpf-used' : ''}>{digit}<small>{usedDigits.includes(digit) ? 'used' : 'free'}</small></span>)}</div>
    <div className="dpf-controls"><button disabled={!prefix.length} onClick={() => setPrefix(prefix.slice(0, -1))}>Previous prefix</button><button disabled={!prefix.length} onClick={() => setPrefix('')}>Return to empty prefix</button>{bound === 213 && <><button onClick={() => setPrefix('12')}>Inspect prefix 12</button><button onClick={() => setPrefix('21')}>Inspect prefix 21</button></>}</div>
    <p className="dpf-result" data-dpf-status>Prefix {prefix || '(empty)'} has <strong>{model.remaining}</strong> valid positive completions.{model.completion !== null ? ` One is ${model.completion}.` : ' No positive integer completes this prefix.'}</p>
    {model.complete ? <p>{model.state.started ? `End of spelling: ${Number(prefix)} is one valid number, so this leaf contributes 1.` : 'All positions were padding. This represents 0, which is excluded, so this leaf contributes 0.'}</p> : <><h4>Choose the next digit · counts partition the remaining answers</h4><div className="dpf-digit-choices">{model.branches.map(branch => <button key={branch.digit} disabled={!branch.allowed} aria-label={`Append digit ${branch.digit}: ${branch.allowed ? `${branch.count} completions${branch.padding ? ', padding zero' : ''}` : branch.reason}`} onClick={() => setPrefix(prefix + branch.digit)}><strong>{branch.digit}</strong><span>{branch.allowed ? `${branch.count} ${branch.count === 1 ? 'way' : 'ways'}` : branch.reason === 'above bound' ? 'bound' : 'used'}</span><small>{branch.allowed ? branch.padding ? 'padding' : 'real digit' : 'rejected'}</small></button>)}</div><p>Allowed branch counts sum to {model.remaining}. “Bound” means the digit is too large while tight; “used” means an actual digit would repeat. A legal branch with zero completions may still be inspected.</p></>}
    <p className="dpf-note">Bound 0–999999 keeps the browser finite. The fixed-bound cache is reused when only the prefix cursor changes. Gray italic zeros are padding; real zeros mark digit 0 as used. Try 100 or 102, and explain which endpoint count changes.</p>
  </section>;
}
