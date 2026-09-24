import { useId, useMemo, useState } from 'react';
import { Investigation } from './LessonInvestigation.jsx';
import { LessonTable } from './LessonElements.jsx';
import { inspectionRows, xorRows, growTree, treePrediction, leafRegions, splitCandidates, pruningReport, forestReport, ensembleVariance, permutationRows } from '../../data/decision-tree-models.js';
import './decision-tree-labs.css';
const number = (value, places = 3) => value.toFixed(places);
function Slider({
  label,
  value,
  setValue,
  min,
  max,
  step = .25
}) {
  const id = useId();
  return <label htmlFor={id}><span>{label}: <output>{value}</output></span><input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => setValue(Number(event.target.value))} /></label>;
}
function Plot({
  title,
  children,
  description
}) {
  return <figure className="tree-figure"><div className="tree-plot-scroll" tabIndex={0} role="region" aria-label={`${title}; horizontally scrollable`}><svg viewBox="0 0 400 300" role="img" aria-label={title}><title>{title}</title><desc>{description}</desc>{children}</svg></div><p className="tree-scroll-hint">Scroll the figure sideways if its right edge is outside your screen; the labels retain their reading size.</p><figcaption>{description}</figcaption></figure>;
}
function CoordinateMap({
  tree,
  rows = inspectionRows,
  query = null,
  bounds = [0, 6, 0, 4],
  title = 'The tree and its feature-space regions'
}) {
  const scaleX = value => 50 + 320 * (value - bounds[0]) / (bounds[1] - bounds[0]);
  const scaleY = value => 253 - 213 * (value - bounds[2]) / (bounds[3] - bounds[2]);
  const regions = leafRegions(tree, bounds);
  return <Plot title={title} description="Each rectangle is one actual leaf, labelled with its fitted positive-class fraction. Gold circles are observed label 1; blue squares are label 0. A white cross marks the query. Shading is a fitted region, not certainty or a smooth interpolation.">
    {regions.map((leaf, index) => <g key={leaf.path}><rect x={scaleX(leaf.bounds[0])} y={scaleY(leaf.bounds[3])} width={scaleX(leaf.bounds[1]) - scaleX(leaf.bounds[0])} height={scaleY(leaf.bounds[2]) - scaleY(leaf.bounds[3])} fill={leaf.probability > .5 ? '#72541e' : '#214858'} fillOpacity=".5" stroke="#9aa5ad" /><text x={scaleX(leaf.bounds[0]) + 6} y={scaleY(leaf.bounds[3]) + 16}>L{index + 1}: {number(leaf.probability, 2)}</text></g>)}
    {rows.map(row => <g key={row.id}>{row.label ? <circle cx={scaleX(row.features[0])} cy={scaleY(row.features[1])} r="5" className="tree-positive" /> : <rect x={scaleX(row.features[0]) - 5} y={scaleY(row.features[1]) - 5} width="10" height="10" className="tree-negative" />}<text x={scaleX(row.features[0]) + 8} y={scaleY(row.features[1]) + 5}>{row.id}</text></g>)}
    {query && <path d={`M${scaleX(query[0]) - 7},${scaleY(query[1]) - 7}l14,14m-14,0l14,-14`} stroke="white" strokeWidth="3" />}
    {[0, .5, 1].map(fraction => <g key={fraction}><text x={50 + 320 * fraction} y="273" textAnchor="middle">{number(bounds[0] + fraction * (bounds[1] - bounds[0]), 1)}</text><text x="42" y={257 - 213 * fraction} textAnchor="end">{number(bounds[2] + fraction * (bounds[3] - bounds[2]), 1)}</text></g>)}
    <text x="210" y="295" textAnchor="middle">Measurement x₁</text><text x="14" y="145" transform="rotate(-90 14 145)" textAnchor="middle">Measurement x₂</text>
  </Plot>;
}
function TreeRules({
  node,
  selectedPath = ''
}) {
  const selected = selectedPath.startsWith(node.path);
  return <ul className="tree-rules"><li><div className={selected ? 'tree-rule tree-rule-selected' : 'tree-rule'}>{node.left ? <><strong>x{node.feature + 1} ≤ {number(node.threshold, 2)}?</strong><span>{node.count} training rows; considered {node.consideredFeatures.map(feature => `x${feature + 1}`).join(', ')}</span></> : <><strong>Leaf: p(1)={number(node.probability)}</strong><span>{node.count} rows; label {node.label}; {node.rows.join(', ')}</span></>}</div>{node.left && <><div className="tree-branch-label">Yes → left</div><TreeRules node={node.left} selectedPath={selectedPath} /><div className="tree-branch-label">No → right</div><TreeRules node={node.right} selectedPath={selectedPath} /></>}</li></ul>;
}
export function TreeTrainingFigure() {
  return <figure className="tree-lifecycle"><div><strong>Learn from completed inspections</strong><span>known measurements + later labels</span><span>↓ compare candidate splits</span><strong>Store questions and leaf counts</strong></div><div><strong>Predict a new inspection</strong><span>available measurements only</span><span>↓ follow the stored questions</span><strong>Return the reached leaf's distribution</strong></div><figcaption>Learning chooses the questions. Prediction follows them; the new label does not participate. Measurements here are invented dimensionless teaching coordinates.</figcaption></figure>;
}
export function TreePartitionLab() {
  const [depth, setDepth] = useState(2);
  const [minimum, setMinimum] = useState(1);
  const [x, setX] = useState(4);
  const [y, setY] = useState(3);
  const tree = useMemo(() => growTree(inspectionRows, {
    maxDepth: depth,
    minimumLeaf: minimum
  }), [depth, minimum]);
  const result = treePrediction(tree, [x, y]);
  return <Investigation id="tree-partitions" kicker="QUESTION ↔ REGION ↔ LEAF" title="Follow one inspection through the same tree in two views">
    <p className="lesson-live-note">At x₁=4,x₂=3, where does the depth-two tree stop? Would another question change the leaf's mixed labels?</p>
    <div className="tree-controls"><label>Maximum depth<select value={depth} onChange={event => setDepth(Number(event.target.value))}>{[0, 1, 2, 3].map(value => <option key={value} value={value}>{value}</option>)}</select></label><label>Minimum leaf rows<select value={minimum} onChange={event => setMinimum(Number(event.target.value))}>{[1, 2, 3].map(value => <option key={value} value={value}>{value}</option>)}</select></label><Slider label="Query x1" value={x} setValue={setX} min={0} max={6} /><Slider label="Query x2" value={y} setValue={setY} min={0} max={4} /><button onClick={() => {
        setDepth(2);
        setMinimum(1);
        setX(4);
        setY(3);
      }}>Reset</button></div>
    <CoordinateMap tree={tree} query={[x, y]} />
    <p className="tree-readout" aria-live="polite">Query probability {number(result.probability)}; predicted label {result.label}; reached {result.count} training rows: {result.rows.join(', ')}.</p>
    <ol className="tree-path">{result.path.map((step, index) => <li key={index}>Compare x{step.feature + 1}={number(step.value, 2)} with {number(step.threshold, 2)}: go {step.direction}.</li>)}<li>Use this leaf's counts; no new training occurs when the query moves.</li></ol>
    <details><summary>Inspect the complete branching rule</summary><TreeRules node={tree} selectedPath={result.leafPath} /></details>
    <LessonTable caption="The eight rows used to learn every displayed split" headers={['Row', 'x₁', 'x₂', 'Label']} rows={inspectionRows.map(row => [row.id, ...row.features, row.label])} />
    <p>Set x₁ exactly to 4.5: equality goes left. Increase depth to 3, then minimum leaf rows to 2. Explain why a one-row region can be representable but disallowed by the training constraint.</p>
  </Investigation>;
}
export function SplitLedgerLab() {
  const [criterion, setCriterion] = useState('gini');
  const [minimum, setMinimum] = useState(1);
  const [selected, setSelected] = useState(2);
  const candidates = splitCandidates(inspectionRows, {
    criterion,
    minimumLeaf: minimum
  });
  const current = candidates[selected];
  const best = candidates.filter(row => row.allowed).reduce((winner, row) => !winner || row.gain > winner.gain + 1e-12 ? row : winner, null);
  return <Investigation id="tree-split-ledger" kicker="MAKE THE TRAINING CHOICE" title="Put the rows on each side before scoring a split">
    <p className="lesson-live-note">Is splitting off two pure positive rows better than a balanced split with mixed labels on both sides? Compute the weighted contribution.</p>
    <div className="tree-controls"><label>Criterion<select value={criterion} onChange={event => setCriterion(event.target.value)}><option value="gini">Gini</option><option value="entropy">Entropy (bits)</option></select></label><label>Minimum child rows<select value={minimum} onChange={event => setMinimum(Number(event.target.value))}><option value={1}>1</option><option value={3}>3</option></select></label><label>Inspect candidate<select value={selected} onChange={event => setSelected(Number(event.target.value))}>{candidates.map((row, index) => <option key={index} value={index}>x{row.feature + 1} ≤ {row.threshold}</option>)}</select></label><button onClick={() => {
        setCriterion('gini');
        setMinimum(1);
        setSelected(2);
      }}>Reset</button></div>
    <div className="tree-buckets">{[['Yes → left', current.left, current.leftImpurity], ['No → right', current.right, current.rightImpurity]].map(([label, rows, score]) => <div key={label}><strong>{label}</strong><div className="tree-tokens">{rows.map(row => <span key={row.id} className={row.label ? 'tree-token-positive' : 'tree-token-negative'}>{row.id}: {row.label}</span>)}</div><p>{rows.length}/8 of the rows × impurity {number(score)} = {number(rows.length / 8 * score)}</p></div>)}</div>
    <p className="tree-readout" aria-live="polite">Gain = {number(current.parentImpurity)} − {number(current.weighted)} = <strong>{number(current.gain)}</strong>. {current.allowed ? 'Both children satisfy the size constraint.' : 'This split is ineligible: a child is too small.'}</p>
    <LessonTable caption="All candidate partitions, with deterministic feature/threshold tie order" headers={['Question', 'Left/right', 'Weighted impurity', 'Gain', 'Legal?']} rows={candidates.map(row => [`x${row.feature + 1} ≤ ${row.threshold}`, `${row.left.length}/${row.right.length}`, number(row.weighted), number(row.gain), row.allowed ? 'Yes' : 'No'])} />
    <p>Best legal candidate: x{best.feature + 1} ≤ {best.threshold}. Changing the constraint can change the selected tree even though every impurity value for a given partition stays the same.</p>
  </Investigation>;
}
export function XorTreeLab() {
  const [allowZero, setAllowZero] = useState(false);
  const tree = growTree(xorRows, {
    maxDepth: 2,
    allowZero
  });
  const correct = xorRows.filter(row => treePrediction(tree, row.features).label === row.label).length;
  return <Investigation id="tree-xor" kicker="CAPACITY IS NOT A SEARCH POLICY" title="A useful first question can have zero immediate gain">
    <p className="lesson-live-note">Both coordinates split these four corners into one label of each kind. Can two successive questions still identify every label?</p>
    <div className="tree-controls"><label>Zero-gain policy<select value={String(allowZero)} onChange={event => setAllowZero(event.target.value === 'true')}><option value="false">Stop at zero gain</option><option value="true">Allow zero gain and continue to depth 2</option></select></label><button onClick={() => setAllowZero(false)}>Reset</button></div>
    <CoordinateMap tree={tree} rows={xorRows} bounds={[-.3, 1.3, -.3, 1.3]} title="XOR: interaction and the chosen greedy stopping rule" />
    <p aria-live="polite">Correct training labels: {correct}/4. {allowZero ? 'A second question in each branch produces pure leaves.' : 'The unsplit leaf ties at p(1)=0.5 and chooses label 0.'}</p>
    <p>This is an exact representability and search example, not evidence of generalization from four points. The feature interaction matters. A decision tree can represent XOR; a policy that refuses its zero-gain first step cannot discover this representation.</p>
  </Investigation>;
}
export function TreePruningLab() {
  const [alpha, setAlpha] = useState(.04);
  const report = pruningReport(alpha);
  const lines = report.candidates.filter((row, index, all) => all.findIndex(other => other.leaves === row.leaves && Math.abs(other.risk - row.risk) < 1e-12) === index);
  const xScale = value => 50 + 320 * value / .3;
  const yScale = value => 250 - 210 * value / 1.6;
  return <Investigation id="tree-pruning" kicker="TRADE FIT AGAINST COMPLEXITY" title="Choose a subtree by its actual cost line">
    <p className="lesson-live-note">Must pruning remove one leaf at a time? Compare the five-leaf tree with the root-only tree as α grows.</p>
    <div className="tree-controls"><Slider label="Complexity cost alpha" value={alpha} setValue={setAlpha} min={0} max={.3} step={.01} /><button onClick={() => setAlpha(.04)}>Reset</button></div>
    <Plot title="Every distinct subtree cost: weighted impurity plus alpha times leaves" description="Lines are calculated from all six pruned subtrees of the displayed full tree; two share the same risk and leaf count. Gold marks the currently chosen cost line. The dashed vertical marker is α; the lowest line wins, with fewer leaves preferred at exact ties.">
      {[0, .1, .2, .3].map(value => <g key={value}><line x1={xScale(value)} x2={xScale(value)} y1="40" y2="250" className="tree-grid" /><text x={xScale(value)} y="271" textAnchor="middle">{value}</text></g>)}
      {[0, .8, 1.6].map(value => <g key={value}><text x="43" y={yScale(value) + 4} textAnchor="end">{value}</text><line x1="50" x2="370" y1={yScale(value)} y2={yScale(value)} className="tree-grid" /></g>)}
      {lines.map(row => <line key={row.index} x1={xScale(0)} x2={xScale(.3)} y1={yScale(row.risk)} y2={yScale(row.risk + .3 * row.leaves)} stroke={row.leaves === report.best.leaves ? '#eec26c' : '#84939d'} strokeWidth={row.leaves === report.best.leaves ? 3 : 1.5} />)}
      <line x1={xScale(alpha)} x2={xScale(alpha)} y1="40" y2="250" stroke="#89c9e5" strokeDasharray="5 4" /><circle cx={xScale(alpha)} cy={yScale(report.best.objective)} r="5" className="tree-positive" />
      <text x="210" y="295" textAnchor="middle">Complexity cost α</text><text x="14" y="145" transform="rotate(-90 14 145)" textAnchor="middle">Weighted impurity + αL</text>
    </Plot>
    <p aria-live="polite">Selected {report.best.leaves} leaves; risk {number(report.best.risk, 6)} + {alpha}×{report.best.leaves} = {number(report.best.objective, 6)}.</p>
    <LessonTable caption="Distinct cost lines, not validation performance" headers={['Leaves', 'Training risk', 'Objective at α']} rows={[...lines].sort((a, b) => a.leaves - b.leaves).map(row => [row.leaves, number(row.risk, 6), number(row.objective, 6)])} />
    <details><summary>Inspect the selected pruned rule</summary><TreeRules node={report.best.tree} /></details>
    <p>In this fixture the optimum jumps from five leaves to one at α=0.1171875. Intermediate subtree sizes exist but never minimize this objective. Choosing α for future prediction still needs validation; this graph uses training impurity.</p>
  </Investigation>;
}
export function BootstrapForestLab() {
  const [trees, setTrees] = useState(6);
  const [selectedRow, setSelectedRow] = useState(0);
  const [featureSampling, setFeatureSampling] = useState(true);
  const [member, setMember] = useState(0);
  const report = useMemo(() => forestReport({
    trees,
    selectedRow,
    featureSampling
  }), [trees, selectedRow, featureSampling]);
  const selectedMember = Math.min(member, trees - 1);
  return <Investigation id="tree-bootstrap" kicker="SAMPLED ROWS → TREES → ELIGIBLE PREDICTIONS" title="See exactly which trees may judge an omitted row">
    <p className="lesson-live-note">If A appears twice in a bootstrap sample, can that tree supply A's out-of-bag prediction? What if none of the current trees omit A?</p>
    <div className="tree-controls"><Slider label="Number of trees" value={trees} setValue={setTrees} min={1} max={12} step={1} /><label>Inspect original row<select value={selectedRow} onChange={event => setSelectedRow(Number(event.target.value))}>{inspectionRows.map((row, index) => <option key={row.id} value={index}>{row.id} — true label {row.label}</option>)}</select></label><label>Candidate features per node<select value={String(featureSampling)} onChange={event => setFeatureSampling(event.target.value === 'true')}><option value="true">One newly sampled feature</option><option value="false">Both features (bagging)</option></select></label><button onClick={() => {
        setTrees(6);
        setSelectedRow(0);
        setFeatureSampling(true);
        setMember(0);
      }}>Reset</button></div>
    <div className="tree-bootstrap-ledger">{report.members.map(row => <div className="tree-bootstrap-member" key={row.index}><strong>Tree {row.index + 1}</strong><div className="tree-tokens" aria-label={`Bootstrap draws for tree ${row.index + 1}`}>{row.sampleIndices.map((index, draw) => <span key={draw} className={index === selectedRow ? 'tree-token-selected' : ''}>{inspectionRows[index].id}</span>)}</div><span>p(1)={number(row.prediction.probability)} · {row.omitted ? 'Eligible for this row’s OOB' : 'In sample — exclude from this row’s OOB'}</span></div>)}</div>
    <p className="tree-readout" aria-live="polite">All-tree mean p(1)={number(report.probability)}. OOB uses {report.oobCount} trees: <strong>{report.oobProbability === null ? 'unavailable — no eligible tree' : number(report.oobProbability)}</strong>.</p>
    <label>Inspect an individual fitted tree<select value={selectedMember} onChange={event => setMember(Number(event.target.value))}>{report.members.map(row => <option key={row.index} value={row.index}>Tree {row.index + 1}</option>)}</select></label>
    <details><summary>Show its questions and sampled candidate features</summary><TreeRules node={report.members[selectedMember].tree} selectedPath={report.members[selectedMember].prediction.leafPath} /></details>
    <p>Each displayed sample contains eight draws with replacement, including repetitions; leaf counts count draws. A fixed small seeded generator makes this reproducible. These twelve possible shallow teaching trees are not sklearn's RNG or a performance benchmark. No-eligible-feature split stops this model; library search policies can differ.</p>
  </Investigation>;
}
export function ForestAveragingFigure() {
  const probabilities = [.49, .49, .99];
  return <figure className="tree-vote-figure"><figcaption><strong>Three illustrative leaf distributions: averaging probabilities differs from counting hard votes.</strong></figcaption>{probabilities.map((probability, index) => <div key={index} className="tree-probability-row"><span>Tree {index + 1}: p(1)={probability}</span><span className="tree-probability-track"><span style={{
          width: `${probability * 100}%`
        }} /></span><span>hard label {Number(probability > .5)}</span></div>)}<p>Mean probability = (0.49+0.49+0.99)/3 = 0.6567 → label 1. Hard votes are 0,0,1 → majority label 0. These are supplied distributions to isolate aggregation; sklearn RandomForestClassifier averages tree class probabilities.</p></figure>;
}
export function ForestVarianceLab() {
  const [trees, setTrees] = useState(20);
  const [correlation, setCorrelation] = useState(.3);
  const x = value => 50 + 320 * (value - 1) / 199;
  const y = value => 250 - 210 * value;
  return <Investigation id="tree-variance" kicker="A MATHEMATICAL MODEL, NOT A BENCHMARK" title="Separate averaging away noise from a shared component">
    <p className="lesson-live-note">With each predictor variance fixed at 1 and average pairwise correlation 0.3, can 200 trees reduce variance below 0.3?</p>
    <div className="tree-controls"><Slider label="Predictor count B" value={trees} setValue={setTrees} min={1} max={200} step={1} /><Slider label="Average correlation rho" value={correlation} setValue={setCorrelation} min={0} max={1} step={.1} /><button onClick={() => {
        setTrees(20);
        setCorrelation(.3);
      }}>Reset</button></div>
    <Plot title="Variance of the mean under an equal-variance covariance model" description="Calculated Var(mean)=ρ+(1−ρ)/B for individual variance 1 and a valid nonnegative equicorrelation model. The dashed horizontal line is the limiting shared component. This is not observed classification error, calibration or a measured forest speed/accuracy curve.">
      {[0, .5, 1].map(value => <g key={value}><line x1="50" x2="370" y1={y(value)} y2={y(value)} className="tree-grid" /><text x="43" y={y(value) + 4} textAnchor="end">{value}</text></g>)}
      {[1, 50, 100, 150, 200].map(value => <text key={value} x={x(value)} y="271" textAnchor="middle">{value}</text>)}
      <polyline points={Array.from({
        length: 200
      }, (_, index) => `${x(index + 1)},${y(ensembleVariance(index + 1, correlation))}`).join(' ')} fill="none" stroke="#eec26c" strokeWidth="2.5" /><line x1="50" x2="370" y1={y(correlation)} y2={y(correlation)} stroke="#89c9e5" strokeDasharray="5 4" /><circle cx={x(trees)} cy={y(ensembleVariance(trees, correlation))} r="5" className="tree-positive" />
      <text x="210" y="295" textAnchor="middle">Number of predictors B</text><text x="14" y="145" transform="rotate(-90 14 145)" textAnchor="middle">Variance of their mean</text>
    </Plot>
    <p aria-live="polite">Variance = {number(correlation)} + (1−{number(correlation)})/{trees} = <strong>{number(ensembleVariance(trees, correlation), 6)}</strong>.</p>
    <p>In a real model comparison, changing feature sampling can change individual variance, bias and correlation together. Moving this slider while holding variance 1 fixed does not promise the effect of a library hyperparameter.</p>
  </Investigation>;
}
export function PermutationRelianceLab() {
  const [mode, setMode] = useState('copy');
  const rows = permutationRows(mode);
  const accuracy = rows.filter(row => row.label === row.predicted).length / rows.length;
  return <Investigation id="tree-permutation" kicker="ASK WHAT THIS MODEL USES" title="Shuffle a copy, a used column, or their relationship">
    <p className="lesson-live-note">The fixed rule predicts x₁, while x₂ is an exact copy in the observed data. Does a zero score drop for x₂ imply it contains no information?</p>
    <div className="tree-controls"><label>Columns to permute together<select value={mode} onChange={event => setMode(event.target.value)}><option value="copy">Copy x₂ only</option><option value="used">Used x₁ only</option><option value="group">Both as one group</option></select></label><button onClick={() => setMode('copy')}>Reset</button></div>
    <div className="tree-permutation-rows">{rows.map(row => <div key={row.id} className={row.impossiblePair ? 'tree-permutation-row tree-off-support' : 'tree-permutation-row'}><span>Row {row.id}: [{row.original.join(', ')}]</span><span aria-label="becomes">→</span><strong>[{row.transformed.join(', ')}]</strong><span>truth {row.label}; predict {row.predicted}</span><span>{row.impossiblePair ? 'Not present in the assumed duplicate relationship' : 'Duplicate relationship preserved'}</span></div>)}</div>
    <p aria-live="polite">Original accuracy 1.000; permuted accuracy {number(accuracy)}; score drop {number(1 - accuracy)}.</p>
    <p>The rule is deliberately supplied rather than fitted, so the mechanism is exact. A copied column can be predictive yet unused by this particular rule. Separate shuffling breaks the observed relationship; grouped shuffling preserves it but asks about the whole group. Neither is an intervention or universal measure of intrinsic importance.</p>
  </Investigation>;
}
