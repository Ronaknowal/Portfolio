import { useId, useState } from 'react';
import { CAUSAL_PATH_PRESETS, CAUSAL_RULE_PRESETS, counterfactualResponseTypes, formatCausalNumber, frontdoorPopulation, inspectDoRule, latentCausalAmbiguity, offerPopulation, traceCausalPaths } from '../../data/causal-inference-models.js';
import './causal-inference-labs.css';
const format = formatCausalNumber;
const edgeKey = edge => edge.join('→');
function Range({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = .05
}) {
  return <label className="causal-range">{label}: <strong>{format(value)}</strong>
    <input type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}
function Facts({
  rows
}) {
  return <dl className="causal-facts">{rows.map(([label, value]) => <div key={label}>
    <dt>{label}</dt><dd>{value}</dd>
  </div>)}</dl>;
}
function DataTable({
  caption,
  headings,
  rows
}) {
  return <details className="causal-table"><summary>{caption}</summary>
    <div role="region" aria-label={caption} tabIndex={0}><table><thead><tr>
      {headings.map(heading => <th key={heading} scope="col">{heading}</th>)}
    </tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>
      {row.map((value, index) => index === 0 ? <th scope="row" key={index}>{value}</th> : <td key={index}>{value}</td>)}
    </tr>)}</tbody></table></div>
  </details>;
}
function CausalGraph({
  graph,
  title,
  conditioned = [],
  removed = [],
  highlight = [],
  intervention = []
}) {
  const arrow = `causal-arrow-${useId().replaceAll(':', '')}`;
  const hasEdge = (edges, edge) => edges.some(item => edgeKey(item) === edgeKey(edge));
  function edgePath([source, target]) {
    const [sx, sy] = graph.positions[source],
      [tx, ty] = graph.positions[target];
    const curved = source === 'X' && target === 'Y' && graph.nodes.includes('M');
    if (curved) {
      const control = [(sx + tx) / 2, Math.max(sy, ty) + 100];
      const startLength = Math.hypot(control[0] - sx, control[1] - sy);
      const endLength = Math.hypot(control[0] - tx, control[1] - ty);
      const start = [sx + 26 * (control[0] - sx) / startLength, sy + 26 * (control[1] - sy) / startLength];
      const end = [tx + 28 * (control[0] - tx) / endLength, ty + 28 * (control[1] - ty) / endLength];
      return `M${start} Q${control} ${end}`;
    }
    const length = Math.hypot(tx - sx, ty - sy);
    return `M${sx + 26 * (tx - sx) / length} ${sy + 26 * (ty - sy) / length} L${tx - 28 * (tx - sx) / length} ${ty - 28 * (ty - sy) / length}`;
  }
  return <figure className="causal-graph"><figcaption>{title}</figcaption>
    <svg viewBox="0 0 360 290" role="img" aria-label={`${title}. Nodes and directed edges are listed immediately below.`}>
      <defs><marker id={arrow} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0L10 5L0 10Z" fill="context-stroke" />
      </marker></defs>
      {graph.edges.map(edge => <path key={edgeKey(edge)} d={edgePath(edge)} markerEnd={`url(#${arrow})`} className={`causal-edge${hasEdge(removed, edge) ? ' is-cut' : ''}${hasEdge(highlight, edge) ? ' is-active' : ''}`} />)}
      {graph.nodes.map(node => <g key={node} transform={`translate(${graph.positions[node].join(' ')})`}>
        {conditioned.includes(node) && <circle r="31" className="causal-conditioned-ring" />}
        <circle r="24" className={`causal-node${(graph.hidden || []).includes(node) ? ' is-hidden' : ''}${intervention.includes(node) ? ' is-forced' : ''}`} />
        <text textAnchor="middle" dy="7">{node}</text>
      </g>)}
    </svg>
    <div className="causal-edge-list">{graph.edges.map(edge => <span key={edgeKey(edge)} className={hasEdge(removed, edge) ? 'causal-cut-label' : ''}>
      {edgeKey(edge)}{hasEdge(removed, edge) ? ' cut' : ''}
    </span>)}</div>
    {conditioned.length > 0 && <p className="causal-caption">Double ring: conditioned {conditioned.join(', ')}.</p>}
    {(graph.hidden || []).length > 0 && <p className="causal-caption">Dashed node: {graph.hidden.join(', ')} is unobserved.</p>}
  </figure>;
}
const offerGraph = {
  nodes: ['Z', 'X', 'Y'],
  edges: [['Z', 'X'], ['Z', 'Y'], ['X', 'Y']],
  positions: {
    Z: [100, 55],
    X: [55, 210],
    Y: [305, 210]
  }
};
export function CausalSurgeryFigure() {
  return <figure className="causal-figure" data-causal-figure="surgery">
    <figcaption>Same population and outcome mechanism. A different assignment mechanism.</figcaption>
    <div className="causal-paired-graphs">
      <div><CausalGraph graph={offerGraph} title="Observe the system" />
        <p>Z generates X. Selecting X=1 changes which Z values are common among selected cases.</p>
        <code>P(Z) × P(X|Z) × P(Y|X,Z)</code></div>
      <div><CausalGraph graph={offerGraph} title="Force X to x" removed={[["Z", "X"]]} intervention={['X']} />
        <p>Replace the X equation. Keep the original distribution of Z and the mechanism for Y.</p>
        <code>P(Z) × 1[X=x] × P(Y|x,Z)</code></div>
    </div>
    <p className="causal-caption">Z = prior activity, X = offer, Y = conversion. A solid amber node denotes the forced variable; dashed arrow denotes its removed input. These factors describe the stated causal model.</p>
  </figure>;
}
export function CausalPathLab() {
  const [presetName, setPresetName] = useState('fork');
  const [conditioned, setConditioned] = useState([]);
  const graph = CAUSAL_PATH_PRESETS[presetName];
  const result = traceCausalPaths(graph, 'X', 'Y', conditioned);
  const activeEdges = result.paths.filter(path => path.active).flatMap(path => path.nodes.slice(1).map((node, index) => graph.edges.find(edge => edge.includes(node) && edge.includes(path.nodes[index]))));
  function selectPreset(name) {
    setPresetName(name);
    setConditioned([]);
  }
  return <section className="causal-lab" aria-label="Causal path investigation" data-causal-lab="paths">
    <header><span className="causal-eyebrow">Follow the path</span><h3>What does conditioning open or close?</h3>
      <p>Predict the effect of selecting the middle node—or its descendant—before checking a box. Conditioning adds a ring; it does not delete a causal arrow.</p></header>
    <div className="causal-controls"><label>Path structure<select aria-label="Path structure" value={presetName} onChange={event => selectPreset(event.target.value)}>
      {Object.entries(CAUSAL_PATH_PRESETS).map(([key, preset]) => <option key={key} value={key}>{preset.title}</option>)}
    </select></label><fieldset><legend>Condition on</legend>{graph.allowed.map(node => <label className="causal-check" key={node}>
      <input type="checkbox" checked={conditioned.includes(node)} onChange={() => setConditioned(current => current.includes(node) ? current.filter(item => item !== node) : [...current, node])} />{node}
    </label>)}</fieldset></div>
    <div className="causal-investigation-layout"><CausalGraph graph={graph} title={graph.title} conditioned={conditioned} highlight={activeEdges} />
      <div><p>{graph.description}</p><p className="causal-feedback" aria-live="polite">
        {result.separated ? 'All X–Y paths are blocked: the graph implies conditional independence.' : 'An active X–Y path remains: the graph permits dependence.'}
      </p>{result.paths.map((path, index) => <div className="causal-path-reason" key={index}>
        <strong>{path.nodes.join(' — ')}</strong>{path.interior.map(item => <p key={item.node}>{item.node}: {item.reason}.</p>)}
      </div>)}<p className="causal-caption">An active path is not proof that a particular numerical model has nonzero dependence. Special parameter cancellations can still occur.</p></div>
    </div><button onClick={() => selectPreset('fork')}>Reset path</button>
  </section>;
}
function PopulationMix({
  label,
  high
}) {
  return <div className="causal-mixture"><div><strong>{label}</strong><span>{high === null ? 'No such cases' : `High activity: ${format(high)}`}</span></div>
    {high !== null && <div className="causal-composition" role="img" aria-label={`${label}: low activity ${format(1 - high)}, high activity ${format(high)}`}>
      <span className="is-low" style={{
        width: `${100 * (1 - high)}%`
      }} /><span className="is-high" style={{
        width: `${100 * high}%`
      }} />
    </div>}
  </div>;
}
export function CausalAdjustmentLab() {
  const [highShare, setHighShare] = useState(.5);
  const [lowAssignment, setLowAssignment] = useState(.2);
  const [highAssignment, setHighAssignment] = useState(.8);
  const result = offerPopulation({
    highShare,
    lowAssignment,
    highAssignment
  });
  function reset() {
    setHighShare(.5);
    setLowAssignment(.2);
    setHighAssignment(.8);
  }
  return <section className="causal-lab" aria-label="Population adjustment investigation" data-causal-lab="adjustment">
    <header><span className="causal-eyebrow">Compare the same population</span><h3>Change who receives the offer</h3>
      <p>The outcome mechanisms stay fixed: low-activity conversion risks are .1/.2 without/with the offer; high-activity risks are .3/.4. Predict which comparisons change when assignment changes.</p></header>
    <div className="causal-controls">
      <Range label="High-activity population share" value={highShare} onChange={setHighShare} />
      <Range label="Offer probability in low group" value={lowAssignment} onChange={setLowAssignment} />
      <Range label="Offer probability in high group" value={highAssignment} onChange={setHighAssignment} />
    </div>
    <div className="causal-weight-comparison">
      <PopulationMix label="Naturally treated" high={result.treatedHighShare} />
      <PopulationMix label="Naturally untreated" high={result.untreatedHighShare} />
      <PopulationMix label="Target population, both interventions" high={highShare} />
      <p className="causal-caption">Muted segment = low activity; amber segment = high activity. Bar length represents population composition, not conversion risk.</p>
    </div>
    <Facts rows={[['Raw observed risk difference', format(result.naiveDifference)], ['Adjusted observed risk difference', format(result.adjustedDifference)], ['Model-known causal risk difference', format(result.causalDifference)], ['Intervention risks without / with', result.interventionRisks.map(format).join(' / ')]]} />
    <p className="causal-feedback" aria-live="polite">{result.overlap ? 'Each population stratum has both treatment states. Standardization uses the same population weights for both risks.' : 'A positive-size stratum lacks one treatment state. Its missing observed risk prevents full adjustment. The model-known answer remains visible because this simulator supplies the hidden mechanism; a dataset alone would not.'}</p>
    <DataTable caption="Inspect the stratum probabilities" headings={['Activity', 'Population', 'P(X=1|Z)', 'Observed Y risk X=0', 'Observed Y risk X=1']} rows={result.strata.map(stratum => [stratum.z ? 'High' : 'Low', format(stratum.share), format(stratum.assignment), ...stratum.risks.map(format)])} />
    <div className="causal-actions"><button onClick={() => {
        setLowAssignment(.5);
        setHighAssignment(.5);
      }}>Randomize assignment</button>
      <button onClick={() => {
        setLowAssignment(.8);
        setHighAssignment(.2);
      }}>Reverse targeting</button>
      <button onClick={reset}>Reset population</button></div>
  </section>;
}
export function CausalAmbiguityFigure() {
  const models = latentCausalAmbiguity();
  return <figure className="causal-figure" data-causal-figure="ambiguity">
    <figcaption>The observed diagonal agrees. Interventions also need the off-diagonal cells.</figcaption>
    <p>U is hidden, equally likely 0/1, and naturally X=U. Each cell gives Y's success probability from its structural mechanism for the supplied U and X. Off-diagonal cells are specified by the model; they are not conditional risks learned from observations.</p>
    <div className="causal-ambiguity-models">{models.map(model => <div key={model.name}>
      <h3>{model.name}</h3><table><caption>Outcome risk by hidden U and treatment X</caption>
        <thead><tr><th scope="col">U \ X</th><th scope="col">0</th><th scope="col">1</th></tr></thead>
        <tbody>{model.risks.map((row, u) => <tr key={u}><th scope="row">{u}</th>{row.map((risk, x) => <td className={u === x ? 'is-observed' : 'is-unseen'} key={x}>
          <strong>{format(risk)}</strong><span>{u === x ? 'Used in observation' : 'Needed under intervention'}</span>
        </td>)}</tr>)}</tbody></table>
      <p>Do risks: {model.interventionRisks.map(format).join(' → ')}.<br /><strong>ATE: {format(model.effect)}</strong>.</p>
    </div>)}</div>
    <p className="causal-caption">Both models give the same measured probabilities: (X,Y)=(0,0): .4; (0,1): .1; (1,0): .1; (1,1): .4. All four measured cells are positive. The hidden mechanisms disagree about forcing X away from U.</p>
  </figure>;
}
export function DoCalculusLab() {
  const [presetName, setPresetName] = useState('observation');
  const preset = CAUSAL_RULE_PRESETS[presetName];
  const result = inspectDoRule(preset);
  const conditioned = [...preset.X, ...preset.W];
  const optionLabels = {
    observation: '1 · Observe: blocked fork',
    observationFails: '1 · Observe: direct path',
    exchange: '2 · Exchange: adjust W',
    exchangeFails: '2 · Exchange: hidden U',
    deletion: '3 · Delete: no selection',
    deletionFails: '3 · Delete: selected W'
  };
  function query(actions, observations) {
    const terms = [...actions.map(node => `do(${node.toLowerCase()})`), ...observations.map(node => node.toLowerCase())];
    return `p(y${terms.length ? ` | ${terms.join(', ')}` : ''})`;
  }
  const left = preset.rule === 1 ? query(preset.X, [...preset.Z, ...preset.W]) : query([...preset.X, ...preset.Z], preset.W);
  const right = preset.rule === 2 ? query(preset.X, [...preset.Z, ...preset.W]) : query(preset.X, preset.W);
  return <section className="causal-lab" aria-label="Do-calculus graph investigation" data-causal-lab="rules">
    <header><span className="causal-eyebrow">Check the modified graph</span><h3>Which equality does this graph justify?</h3>
      <p>Choose a rule and its contrasting case. Predict which arrows the rule asks you to cut, then inspect the Y–Z separation test. These are conditions for a rule, not a complete automatic identification algorithm.</p></header>
    <label className="causal-wide-control">Rule and causal story<select aria-label="Rule and causal story" value={presetName} onChange={event => setPresetName(event.target.value)}>
      {Object.keys(CAUSAL_RULE_PRESETS).map(key => <option key={key} value={key}>{optionLabels[key]}</option>)}
    </select></label>
    <p className="causal-caption">{preset.title}</p>
    <div className="causal-rule-expression" aria-label="Candidate equality">
      <p>Candidate equality — the graph test below decides whether this rule justifies it.</p>
      <code>{left}</code><span>=</span><code>{right}</code>
    </div>
    <div className="causal-paired-graphs"><CausalGraph graph={preset} title="Original causal assumptions" />
      <CausalGraph graph={preset} title="Graph used for the rule's test" removed={result.removed} conditioned={conditioned} />
    </div>
    <Facts rows={[['Existing action set X', preset.X.join(', ') || 'Empty'], ['Additional observation set W', preset.W.join(', ') || 'Empty'], ['Cut arrows', result.removed.map(edgeKey).join(', ') || 'None'], ...(preset.rule === 3 ? [['Eligible action nodes Z(W)', result.eligibleActions.join(', ') || 'Empty: Z is an ancestor of conditioned W']] : [])]} />
    <p className="causal-feedback" aria-live="polite"><strong>{result.valid ? 'Separation test passes.' : 'Separation test fails.'}</strong> {preset.explanation}</p>
    <DataTable caption="Inspect every path in the transformed graph" headings={['Path', 'State', 'Interior-node reasoning']} rows={result.tests.flatMap(test => test.paths.length ? test.paths.map(path => [path.nodes.join(' — '), path.active ? 'Active' : 'Blocked', path.interior.map(item => `${item.node}: ${item.reason}`).join('; ') || 'Direct path, no interior node']) : [['No Y–Z path', 'Separated', 'No connected path remains']])} />
    <p className="causal-caption">When the separation test passes, this rule justifies the candidate equality for compatible causal models where its conditional expressions are defined. A failed test does not prove that every numerical parameter choice violates the equality.</p>
    <button onClick={() => setPresetName('observation')}>Reset rule</button>
  </section>;
}
export function FrontdoorLab() {
  const [mediatorLow, setMediatorLow] = useState(.1);
  const [mediatorHigh, setMediatorHigh] = useState(.8);
  const [hasDirectPath, setHasDirectPath] = useState(false);
  const result = frontdoorPopulation({
    mediatorLow,
    mediatorHigh,
    directEffect: hasDirectPath ? .1 : 0
  });
  const graph = {
    nodes: ['U', 'X', 'M', 'Y'],
    hidden: ['U'],
    edges: [['U', 'X'], ['U', 'Y'], ['X', 'M'], ['M', 'Y'], ...(hasDirectPath ? [['X', 'Y']] : [])],
    positions: {
      U: [180, 45],
      X: [55, 185],
      M: [180, 185],
      Y: [305, 185]
    }
  };
  function reset() {
    setMediatorLow(.1);
    setMediatorHigh(.8);
    setHasDirectPath(false);
  }
  return <section className="causal-lab" aria-label="Frontdoor identification investigation" data-causal-lab="frontdoor">
    <header><span className="causal-eyebrow">An observed intermediate mechanism</span><h3>Can M carry the identifying information?</h3>
      <p>U is hidden. The model has P(U=1)=.5 and P(X=1|U)=.2/.8. Without a direct path, Y's risk is .1+.5M+.2U. Predict the failure before adding X directly to the outcome mechanism.</p></header>
    <div className="causal-controls"><Range label="P(M=1 | X=0)" value={mediatorLow} onChange={setMediatorLow} />
      <Range label="P(M=1 | X=1)" value={mediatorHigh} onChange={setMediatorHigh} />
      <label className="causal-check"><input type="checkbox" checked={hasDirectPath} onChange={event => setHasDirectPath(event.target.checked)} />Add direct X→Y effect of .1</label>
    </div>
    <div className="causal-investigation-layout"><CausalGraph graph={graph} title="Candidate frontdoor graph" />
      <div className="causal-frontdoor-chain"><div><span>1. Observed mediator response</span><strong>{result.mediator.map(format).join(' → ')}</strong><p>P(M=1|X=0) → P(M=1|X=1)</p></div>
        <div><span>2. Average outcome over original X</span><strong>{result.mediatorRisks.map(format).join(' / ')}</strong><p>For M=0 / M=1; each X state receives weight .5.</p></div>
        <div><span>3. Mix using the mediator response</span><strong>{result.frontdoorRisks.map(format).join(' / ')}</strong><p>Proposed outcome risks under X=0 / X=1.</p></div>
      </div></div>
    <Facts rows={[['Observed outcome risks X=0 / X=1', result.observedRisks.map(format).join(' / ')], ['Model-known do risks X=0 / X=1', result.interventionRisks.map(format).join(' / ')], ['All directed X→Y paths pass through M', hasDirectPath ? 'No' : 'Yes']]} />
    <p className="causal-feedback" aria-live="polite">{!result.hasRequiredSupport ? 'A required observed mediator/treatment combination has zero mass. The proposed formula cannot supply every conditional risk.' : hasDirectPath ? 'The direct X→Y path violates full mediation. The same arithmetic is still shown so you can compare its answer with the actual intervention; it is not a valid identification formula for this graph.' : 'The graph meets all three frontdoor criteria and the required factors have support. The observed-only formula matches the model-known intervention.'}</p>
    <DataTable caption="Inspect the observed outcome conditionals" headings={['Mediator', 'P(Y=1|M,X=0)', 'P(Y=1|M,X=1)']} rows={result.outcomeTable.map((row, m) => [m, ...row.map(format)])} />
    <button onClick={reset}>Reset frontdoor</button>
  </section>;
}
export function CounterfactualWorldsLab() {
  const [overlap, setOverlap] = useState(.125);
  const [observedTreatment, setObservedTreatment] = useState(1);
  const [observedOutcome, setObservedOutcome] = useState(1);
  const [intervention, setIntervention] = useState(0);
  const result = counterfactualResponseTypes({
    overlap,
    observedTreatment,
    observedOutcome,
    intervention
  });
  function reset() {
    setOverlap(.125);
    setObservedTreatment(1);
    setObservedOutcome(1);
    setIntervention(0);
  }
  return <section className="causal-lab" aria-label="Counterfactual paired worlds investigation" data-causal-lab="counterfactual">
    <header><span className="causal-eyebrow">Same unit, two possible treatments</span><h3>Which hidden response types survive the evidence?</h3>
      <p>Every chosen model has P(Y(0)=1)=.25 and P(Y(1)=1)=.75. Moving their overlap changes how the two outcomes are paired within people. It does not change those experimental marginals.</p></header>
    <div className="causal-controls"><Range label="Both-outcome success overlap" value={overlap} onChange={setOverlap} max={.25} step={.025} />
      <label>Observed treatment<select aria-label="Observed treatment" value={observedTreatment} onChange={event => setObservedTreatment(Number(event.target.value))}><option value={0}>X=0: no offer</option><option value={1}>X=1: offer</option></select></label>
      <label>Observed outcome<select aria-label="Observed outcome" value={observedOutcome} onChange={event => setObservedOutcome(Number(event.target.value))}><option value={0}>Y=0: failure</option><option value={1}>Y=1: success</option></select></label>
      <label>Counterfactual treatment<select aria-label="Counterfactual treatment" value={intervention} onChange={event => setIntervention(Number(event.target.value))}><option value={0}>Set X=0</option><option value={1}>Set X=1</option></select></label>
    </div>
    <p className="causal-caption">Treatment was randomized independently of response type. Abduction conditions on the observed outcome, action changes X, and prediction reads the new column from the same surviving type.</p>
    <div className="causal-response-types">{result.types.map(type => <div key={type.label} className={`causal-response-type${type.compatible ? '' : ' is-excluded'}`}>
      <div className="causal-outcome-pair"><strong>Type {type.label}</strong><div>{type.outcomes.map((outcome, x) => <span key={x} className={x === intervention ? 'is-read' : ''}>
        <small>Y({x})</small><b>{outcome}</b></span>)}</div></div>
      <div className="causal-type-weights"><div><span>Population: {format(type.prior)}</span><i style={{
              width: `${100 * type.prior}%`
            }} /></div>
        <div><span>After evidence: {format(type.posterior)}</span><i className="is-posterior" style={{
              width: `${100 * type.posterior}%`
            }} /></div>
        <p>{type.compatible ? `Read Y(${intervention})=${type.counterfactualOutcome} from this same type.` : 'This type contradicts the observed outcome.'}</p></div>
    </div>)}</div>
    <Facts rows={[['Population risks Y(0) / Y(1)', '.25 / .75'], ['Population average treatment effect', '.5'], ['Counterfactual success given the evidence', format(result.counterfactualRisk)], ['Probability of the outcome used as evidence', format(result.compatibleMass)]]} />
    <p className="causal-feedback" aria-live="polite">{intervention === observedTreatment ? 'The treatment is unchanged, so consistency reproduces the observed outcome with probability one.' : 'This conditional counterfactual depends on the chosen pairing of potential outcomes. The fixed experimental marginal risks alone do not select that pairing.'}</p>
    <button onClick={reset}>Reset worlds</button>
  </section>;
}
