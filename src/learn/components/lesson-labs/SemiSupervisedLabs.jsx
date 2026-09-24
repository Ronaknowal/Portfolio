import { useId, useMemo, useState } from 'react';
import {
  parseGraph, propagateGraph, scoreSide, parsePrototype, prototypeTraining, prototypeScores,
  coTrainingRows, parseViews, categoricalCoTraining, coTrainingAnswer,
} from '../../data/semi-supervised-models.js';
import { DataTable, GraphPicture, formatScore } from './SemiSupervisedFigures.jsx';
import './semi-supervised.css';

function attempt(callback) {
  try { return { value: callback(), error: null }; } catch (error) { return { value: null, error: error.message }; }
}

function ExplanationPrompt() {
  return <details><summary>Explain it, then construct a different case</summary><p>Name the input that supplied or removed evidence. Change an edge, coordinate or view value of your own; inspect the live result. The presets are comparisons, not a replacement for your own experiment.</p><label>Your explanation (kept in this lab while you read)<textarea placeholder="Which input caused the change? What observation would test whether it helps?" /></label></details>;
}

export function LabelPropagationLab() {
  const identifier = useId();
  const initialNodes = 'A,0\nB,?\nC,?\nD,1';
  const initialEdges = 'A,B,1\nB,C,1\nC,D,1';
  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);
  const [mode, setMode] = useState('hard');
  const [alpha, setAlpha] = useState('0.8');
  const [target, setTarget] = useState('B');
  const [step, setStep] = useState(0);
  const [showEquilibrium, setShowEquilibrium] = useState(true);
  const signature = JSON.stringify([nodes, edges, mode, alpha, target]);
  const parsed = useMemo(() => attempt(() => parseGraph(nodes, edges)), [nodes, edges]);

  const update = callback => { callback(); setStep(0); setShowEquilibrium(true); };
  const preset = (kind) => update(() => {
    setNodes(kind === 'island' ? `${initialNodes}\nE,?\nF,?` : initialNodes);
    setEdges(kind === 'shortcut' ? `${initialEdges}\nB,D,2` : kind === 'separated' ? 'A,B,1\nC,D,1' : kind === 'island' ? `${initialEdges}\nE,F,1` : initialEdges);
    setTarget(kind === 'island' ? 'E' : 'B');
  });
  const current = useMemo(() => attempt(() => {
    if (!parsed.value) throw new Error(parsed.error);
    const index = parsed.value.nodes.findIndex(node => node.name === target && node.label === null);
    if (index < 0) throw new Error('Choose an unknown target node.');
    const result = propagateGraph(parsed.value, mode, alpha);
    return { result, targetIndex: index, answer: scoreSide(result.scores[index]) };
  }), [parsed, target, mode, alpha]);
  const model = current?.value?.result;
  const state = model ? showEquilibrium ? model.equilibrium : model.trace[step] : null;
  const scores = model ? model.readout(state) : null;
  return <section className="ssl-lab" aria-labelledby={`${identifier}-title`} data-live-exploration="semi-supervised" data-ssl-lab="graph">
    <h3 id={`${identifier}-title`}>Investigation · repair the neighborhood</h3>
    <p>Edit the actual edges and observed labels. Observe the selected unknown node’s <strong>equilibrium class-1 score</strong>. A half-score is tied; “unavailable” means no evidence. Compare your result with the chain calculation above.</p>
    <div className="ssl-controls">{[['line', 'Original chain'], ['shortcut', 'B–D shortcut'], ['separated', 'Cut B–C'], ['island', 'Unanchored E–F']].map(([key, label]) => <button key={key} onClick={() => preset(key)}>{label}</button>)}</div>
    <div className="ssl-two-panels"><label>Nodes · name, observed label (0, 1 or ?)<textarea aria-label="Nodes · name, observed label (0, 1 or ?)" value={nodes} onChange={event => update(() => setNodes(event.target.value))} /></label><label>Edges · from, to, weight (0 removes; maximum 5)<textarea aria-label="Edges · from, to, weight (0 removes; maximum 5)" value={edges} onChange={event => update(() => setEdges(event.target.value))} /></label></div>
    <p className="ssl-caption">Add or delete lines: 2–10 nodes, symmetric edges, no self-edge. Positions are layout only. A graph with no observed labels is a valid null experiment.</p>
    <div className="ssl-controls">
      <label>Algorithm<select aria-label="Algorithm" value={mode} onChange={event => update(() => setMode(event.target.value))}><option value="hard">Hard clamping</option><option value="soft">Soft spreading</option></select></label>
      {mode === 'soft' && <label>Alpha (0–0.99)<input aria-label="Alpha (0–0.99)" type="number" min="0" max="0.99" step="0.01" value={alpha} onChange={event => update(() => setAlpha(event.target.value))} /></label>}
      <label>Target unknown node<select aria-label="Target unknown node" value={target} onChange={event => update(() => setTarget(event.target.value))}>{!parsed.value?.nodes.some(node => node.name === target && node.label === null) && <option value={target}>Choose an unknown node</option>}{parsed.value?.nodes.filter(node => node.label === null).map(node => <option key={node.name} value={node.name}>{node.name}</option>)}</select></label>
      
    </div>
    {parsed.error && <p className="ssl-input-error" role="alert">{parsed.error}</p>}
    <div className="ssl-controls"><button onClick={() => update(() => { setNodes(initialNodes); setEdges(initialEdges); setMode('hard'); setAlpha('0.8'); setTarget('B'); })}>Reset graph</button></div>
    {current?.error && <p role="alert" className="ssl-input-error">{current.error}</p>}
    {model && <div className="ssl-result">
      <div className="ssl-controls"><button disabled={showEquilibrium || step >= model.trace.length - 1} onClick={() => setStep(value => value + 1)}>One synchronous update</button><button disabled={showEquilibrium} onClick={() => setShowEquilibrium(true)}>Show equilibrium</button><button onClick={() => { setStep(0); setShowEquilibrium(false); }}>Replay the algorithm</button></div>
      <p role="status">{showEquilibrium ? 'Direct equilibrium solution' : `Update ${step}; this is an intermediate state`}. {mode === 'soft' ? 'Unknown input evidence starts at zero.' : 'Observed values stay fixed.'}</p>
      <GraphPicture graph={parsed.value} scores={scores} />
      <DataTable caption={mode === 'soft' ? 'Raw evidence and separate normalized readout' : 'Node scores and incoming weighted contributions'} headers={mode === 'soft' ? ['Node / origin', 'Raw 0', 'Raw 1', 'Class-1 readout'] : ['Node / origin', 'Class-1 score', 'Weighted neighbor terms']} rows={parsed.value.nodes.map((node, i) => mode === 'soft'
        ? [`${node.name} · ${node.label === null ? 'inferred' : `observed ${node.label}`}`, formatScore(state[i][0]), formatScore(state[i][1]), formatScore(scores[i])]
        : [`${node.name} · ${node.label === null ? 'inferred' : `observed ${node.label}`}`, formatScore(scores[i]), model.weights[i].map((weight, j) => weight > 0 ? `${weight} × ${parsed.value.nodes[j].name} (${formatScore(scores[j])})` : null).filter(Boolean).join(' + ') || 'No neighbors'])} />
      {showEquilibrium && <div className="ssl-feedback" role="status"> {target} is {formatScore(model.scores[current.value.targetIndex])} ({current.value.answer}).<p>{model.scores[current.value.targetIndex] === null ? 'No label evidence reaches the selected readout. In hard mode, an unanchored component has no unique boundary solution; in soft mode it receives zero evidence. Alpha zero also prevents evidence from spreading.' : 'The score follows the edited edges and boundary evidence. Use the neighboring weighted terms above to explain why this side won; the inferred score remains distinct from an observed label.'}</p><p>Direct-system maximum residual: {model.residual.toExponential(2)}. Iterative trace: {model.converged ? `within 1e−10 of the direct solution after ${model.trace.length - 1} updates` : '5,000-update cap reached; iterative convergence incomplete'}.</p></div>}
    </div>}
    <ExplanationPrompt />
  </section>;
}

function PrototypePicture({ points, remaining, means, query }) {
  const positions = [-10, -5, 0, 5, 10];
  const xScale = value => 25 + (value + 10) / 20 * 550;
  const boundary = Math.abs(means[0] - means[1]) < 1e-12 ? null : (means[0] + means[1]) / 2;
  return <figure>
    <svg viewBox="0 0 600 220" role="img" aria-label="Prototype coordinate lanes; all coordinates, means and origins are also listed in text.">
      {[65, 120, 175].map(y => <line key={y} x1="25" x2="575" y1={y} y2={y} stroke="#526773" />)}
      {boundary !== null && <line x1={xScale(boundary)} x2={xScale(boundary)} y1="35" y2="193" stroke="#ecdcab" strokeDasharray="4 5" />}
      {points.map(point => <circle key={point.id} cx={xScale(point.x)} cy={point.origin === 'observed' ? 65 : 120} r="7" fill={point.label === 0 ? '#72cbb0' : '#e2b55a'} stroke="#fff" strokeWidth={point.origin === 'observed' ? 2 : 1} strokeDasharray={point.origin === 'observed' ? undefined : '2 2'} />)}
      {remaining.map(point => <circle key={point.id} cx={xScale(point.x)} cy="175" r="7" fill="#0b1013" stroke="#dce4e9" />)}
      {means.map((mean, i) => <path key={i} d={`M ${xScale(mean)} 20 l -7 -12 h 14 Z`} fill={i === 0 ? '#72cbb0' : '#e2b55a'} />)}
      <path d={`M ${xScale(query)} 198 l -6 8 l 6 8 l 6 -8 Z`} fill="#9dcfff" />
    </svg>
    <div className="ssl-coordinate-ticks" aria-label="Coordinate axis ticks">{positions.map(value => <span key={value} data-ssl-tick={value} style={{ left: `${xScale(value) / 600 * 100}%` }}>{value}</span>)}</div>
    <figcaption>Authored coordinate x. From top: prototype triangles; observed points; pseudo-labels; remaining pool. Dashed line: midpoint boundary. Blue diamond: query, excluded from fitting. Green = class 0; gold = class 1. Coincident marks overlap intentionally; the tables keep each point’s identity.</figcaption>
  </figure>;
}

export function SelfTrainingLab() {
  const identifier = useId();
  const [observed, setObserved] = useState('-2,0\n2,1');
  const [pool, setPool] = useState('-1, 0, 1, 3');
  const [threshold, setThreshold] = useState('0.8');
  const [query, setQuery] = useState('1.25');
  const [step, setStep] = useState(0);
  const signature = JSON.stringify([observed, pool, threshold, query]);
  const parsed = useMemo(() => attempt(() => parsePrototype(observed, pool, threshold, query)), [observed, pool, threshold, query]);
  const current = useMemo(() => parsed.value ? { result: prototypeTraining(parsed.value) } : null, [parsed]);
  const update = callback => { callback(); setStep(0); };
  const baseline = parsed.value ? [0, 1].map(label => {
    const group = parsed.value.observed.filter(point => point.label === label);
    return group.reduce((sum, point) => sum + point.x, 0) / group.length;
  }) : null;
  const model = current?.result;
  const round = step > 0 ? model?.history[step - 1] : null;
  const means = round?.after ?? baseline;
  const points = round?.points ?? parsed.value?.observed.map((point, i) => ({ ...point, origin: 'observed', id: `L${i}` }));
  const remaining = round?.remaining ?? parsed.value?.pool.map((x, i) => ({ x, id: `U${i}` }));
  const finished = model && step === model.history.length;
  return <section className="ssl-lab" aria-labelledby={`${identifier}-title`} data-live-exploration="semi-supervised" data-ssl-lab="self-training">
    <h3 id={`${identifier}-title`}>Investigation · a guess changes the next guess</h3>
    <p>Compare the final model with the observed-only baseline for <em>your current input</em>. See how the boundary moves and whether the query’s class changes. A tie in either query comparison has its own answer.</p>
    <div className="ssl-controls"><button onClick={() => update(() => setPool('-1, 0, 1, 3'))}>Original pool</button><button onClick={() => update(() => setPool('-1, 0, 1, 9'))}>Move 3 to 9</button><button onClick={() => update(() => setPool('0, 0'))}>No-acceptance pool</button></div>
    <div className="ssl-two-panels"><label>Observed points · one x,class row per line<textarea aria-label="Observed points · one x,class row per line" value={observed} onChange={event => update(() => setObserved(event.target.value))} /></label><label>Unlabeled x coordinates · comma or space separated<textarea aria-label="Unlabeled x coordinates · comma or space separated" value={pool} onChange={event => update(() => setPool(event.target.value))} /></label></div>
    <p className="ssl-caption">2–10 observed points, both classes; 0–20 unlabeled points; coordinates −10 to 10. Add or delete rows freely. Pseudo-labels are retained in this simplified algorithm; exact score ties choose class 0 if the threshold accepts them.</p>
    {parsed.value && <details><summary>Drag individual unlabeled coordinates (keyboard arrows also work)</summary><div className="ssl-point-editor">{parsed.value.pool.map((x, i) => <label key={i}>U{i}: {x}<input aria-label={`Unlabeled point U${i}`} type="range" min="-10" max="10" step="0.01" value={x} onChange={event => update(() => { const changed = [...parsed.value.pool]; changed[i] = Number(event.target.value); setPool(changed.join(', ')); })} /></label>)}</div></details>}
    <div className="ssl-controls"><label>Acceptance threshold<input aria-label="Acceptance threshold" type="number" min="0.5" max="0.99" step="0.01" value={threshold} onChange={event => update(() => setThreshold(event.target.value))} /></label><label>Query x (not a training row)<input aria-label="Query x (not a training row)" type="number" min="-10" max="10" step="0.01" value={query} onChange={event => update(() => setQuery(event.target.value))} /></label></div>
    {baseline && <p>Observed-only prototypes: {baseline.map(formatScore).join(' and ')}. Boundary: {Math.abs(baseline[0] - baseline[1]) < 1e-12 ? 'unavailable; every coordinate is tied' : formatScore((baseline[0] + baseline[1]) / 2)}.</p>}
    <div className="ssl-controls"></div>
    {parsed.error && <p className="ssl-input-error" role="alert">{parsed.error}</p>}
    <div className="ssl-controls"><button onClick={() => update(() => { setObserved('-2,0\n2,1'); setPool('-1, 0, 1, 3'); setThreshold('0.8'); setQuery('1.25'); })}>Reset self-training</button></div>
    {model && <div className="ssl-result">
      <div className="ssl-controls"><button disabled={finished} onClick={() => setStep(value => value + 1)}>Promote one batch and refit</button><button disabled={finished} onClick={() => setStep(model.history.length)}>Show final model</button><button onClick={() => setStep(0)}>Replay the algorithm</button></div>
      <p role="status">{step === 0 ? 'Observed-only model; no guesses accepted yet.' : `Round ${step}: accepted ${round.accepted.length}; refitted means ${means.map(formatScore).join(' and ')}.`}</p>
      <PrototypePicture points={points} remaining={remaining} means={means} query={parsed.value.query} />
      <p>Current prototypes μ₀ = {formatScore(means[0])}, μ₁ = {formatScore(means[1])}. Midpoint: {Math.abs(means[0] - means[1]) < 1e-12 ? 'no separating boundary' : formatScore((means[0] + means[1]) / 2)}. {means[0] > means[1] ? 'The class ordering is reversed: class 1 is on the left.' : 'Class 0 is on the left when the means differ.'}</p>
      {round && <DataTable caption={`Round ${step} proposals use OLD means ${round.before.map(formatScore).join(', ')}; then all accepted rows refit together`} headers={['ID / x', 'Squared distances 0 / 1', 'Scores 0 / 1', 'Decision']} rows={round.proposals.map(point => [ `${point.id} / ${point.x}`, round.before.map(mean => formatScore((point.x - mean) ** 2)).join(' / '), point.probabilities.map(formatScore).join(' / '), point.accepted ? `Accept class ${point.label} · round ${step}` : 'Keep unlabeled' ])} />}
      <DataTable caption="Current training-label provenance" headers={['ID', 'x', 'Label', 'Origin']} rows={points.map(point => [point.id, point.x, point.label, point.origin])} />
      <p>Remaining pool: {remaining.map(point => `${point.id}: ${point.x}`).join('; ') || 'empty'}.</p>
      {model && <div className="ssl-feedback" role="status"><p>Final boundary: {formatScore(model.finalBoundary)} ({model.movement}); query: {model.beforeQuery} → {model.afterQuery} (change: {model.queryChange}).</p><p>{model.history[0]?.accepted.length === 0 ? 'No candidate passed the threshold; the observed-only model stays unchanged.' : `The accepted coordinates changed the means to ${model.final.map(formatScore).join(' and ')}. Their origin remains a model guess. Use the batch and origin tables to identify which class mean moved.`} {model.capped ? 'Round cap reached; the pool is not exhausted.' : 'The loop stopped on no offers or an empty pool.'}</p></div>}
    </div>}
    <ExplanationPrompt />
  </section>;
}

const serializeViews = rows => rows.map(row => `${row.views.join(' | ')} | ${row.label ?? '?'}`).join('\n');
function RuleBoards({ rules }) {
  return <div className="ssl-two-panels">{rules.map((board, view) => <section key={view} className="ssl-rule-board"><h4>View {view + 1} rules</h4>{Object.keys(board).length ? Object.entries(board).map(([category, label]) => <p key={category}>{category} → {label}</p>) : <p>No supported category rules</p>}</section>)}</div>;
}
function Provenance({ entries }) {
  // Provenance is a DAG: list each event once rather than expanding every path
  // into a potentially exponential tree when many categories share evidence.
  const seen = new Set();
  const events = [];
  const pending = [...entries];
  while (pending.length) {
    const entry = pending.shift();
    if (seen.has(entry.provenance)) continue;
    seen.add(entry.provenance);
    events.push(entry);
    pending.push(...(entry.provenance?.evidenceRows ?? []));
  }
  return <ul>{events.map((entry, index) => <li key={index}>Row {entry.row}: {entry.provenance?.kind === 'observed' ? 'observed label' : `pseudo-label from view ${(entry.provenance?.donor ?? 0) + 1}, round ${entry.provenance?.round}`}{entry.provenance?.evidenceRows && `; supported by rows ${entry.provenance.evidenceRows.map(source => source.row).join(', ')}`}</li>)}</ul>;
}

export function CoTrainingLab() {
  const identifier = useId();
  const [text, setText] = useState(serializeViews(coTrainingRows));
  const [target, setTarget] = useState('3');
  const [view, setView] = useState('0');
  const [step, setStep] = useState(0);
  const [stage, setStage] = useState(0);
  const parsed = useMemo(() => attempt(() => parseViews(text)), [text]);
  const signature = JSON.stringify([text, target, view]);
  const current = useMemo(() => parsed.value ? { result: categoricalCoTraining(parsed.value) } : null, [parsed]);
  const update = callback => { callback(); setStep(0); setStage(0); };
  const model = current?.result;
  const round = model?.history[step];
  const reveal = current && stage === 4;
  const final = reveal && step === model.history.length - 1;
  const stages = ['Fit category rules', 'Propose synchronously', 'Resolve conflicting offers', 'Receive labels using recipient features', 'Show refitted rules'];
  return <section className="ssl-lab" aria-labelledby={`${identifier}-title`} data-live-exploration="semi-supervised" data-ssl-lab="co-training">
    <h3 id={`${identifier}-title`}>Investigation · pass a label through the other view</h3>
    <p>Inspect the final category rule for one row in one view. The result is “conflict” when the final views give different answers; unresolved evidence is “unknown.” Each recipient keeps its own label array.</p>
    <div className="ssl-controls"><button onClick={() => update(() => setText(serializeViews(coTrainingRows)))}>Original views</button><button onClick={() => update(() => setText(serializeViews(coTrainingRows.map((row, i) => i === 2 ? { ...row, views: ['blue', 'triangle'] } : row))))}>Change bridge to blue</button><button disabled={!parsed.value} onClick={() => update(() => setText(serializeViews(parsed.value.map(row => ({ ...row, views: [row.views[0], row.views[0]] })))))}>Duplicate view 1</button><button disabled={!parsed.value} onClick={() => update(() => setText(serializeViews(parsed.value.map(row => ({ ...row, label: null })))))}>Remove all anchors</button></div>
    <label>Paired rows · view 1 | view 2 | observed label<textarea aria-label="Paired rows · view 1 | view 2 | observed label" rows="8" value={text} onChange={event => update(() => setText(event.target.value))} /></label>
    <p className="ssl-caption">2–20 rows. Category strings are case-sensitive and 1–20 characters. “?” is unknown. Add/delete rows or create new categories to build a different information bridge.</p>
    <div className="ssl-controls"><label>Target row<select aria-label="Target row" value={target} onChange={event => update(() => setTarget(event.target.value))}>{Number(target) >= (parsed.value?.length ?? 0) && <option value={target}>Choose a row</option>}{parsed.value?.map((row, i) => <option key={i} value={i}>{i}: {row.views.join(' / ')}</option>)}</select></label><label>Recipient view<select aria-label="Recipient view" value={view} onChange={event => update(() => setView(event.target.value))}><option value="0">View 1</option><option value="1">View 2</option></select></label></div>
    {parsed.error && <p role="alert" className="ssl-input-error">{parsed.error}</p>}
    <div className="ssl-controls"><button onClick={() => update(() => { setText(serializeViews(coTrainingRows)); setTarget('3'); setView('0'); })}>Reset co-training</button></div>
    {model && <div className="ssl-result">
      <div className="ssl-controls"><button disabled={final} onClick={() => { if (stage < 4) setStage(value => value + 1); else { setStep(value => value + 1); setStage(0); } }}>Next transfer stage</button><button disabled={final} onClick={() => { setStep(model.history.length - 1); setStage(4); }}>Show final rules</button><button onClick={() => { setStep(0); setStage(0); }}>Replay the algorithm</button></div>
      <p role="status">Round {round.round} · {stages[stage]}</p>
      <RuleBoards rules={reveal ? round.rulesAfter : round.rulesBefore} />
      {stage >= 1 && <DataTable caption="Both learners propose from their pre-transfer rules" headers={['Row', 'View 1 / proposed', 'View 2 / proposed']} rows={parsed.value.map((row, i) => [i, `${row.views[0]} / ${round.predictions[0][i] ?? 'abstain'}`, `${row.views[1]} / ${round.predictions[1][i] ?? 'abstain'}`])} />}
      {stage >= 2 && <p>Conflicting rows deferred: {round.conflicts.join(', ') || 'none'}. Accepted offers: {round.offers.length}. Distinct newly reachable category rules this round: {round.newRules}.</p>}
      {stage >= 3 && <div>{round.offers.length ? round.offers.map((offer, i) => <div className="ssl-offer" key={i}><strong>Row {offer.row}: view {offer.donor + 1} → view {offer.recipient + 1}</strong><p>{offer.donorCategory} → {offer.label} supplies a label for recipient feature <strong>{offer.recipientCategory}</strong>. {offer.newCategory ? 'First access to this recipient category.' : 'A row confirmation; not new independent evidence.'}</p><details><summary>Follow the donor’s evidence chain</summary><Provenance entries={offer.evidenceRows} /></details></div>) : <p>No offers: the loop stops.</p>}</div>}
      {reveal && <DataTable caption="Separate recipient label arrays after receiving this round" headers={['Row', 'Observed', 'View 1 label / origin', 'View 2 label / origin']} rows={parsed.value.map((row, i) => [i, row.label ?? 'unknown', ...[0, 1].map(index => `${round.labelsAfter[index][i] ?? 'unknown'} / ${round.provenance[index][i]?.kind === 'observed' ? 'observed' : round.provenance[index][i] ? `view ${round.provenance[index][i].donor + 1}, round ${round.provenance[index][i].round}` : 'no offer'}`)])} />}
      {model && Number(target) < parsed.value.length && <div className="ssl-feedback" role="status"><p>Row {target}, view {Number(view) + 1}: {coTrainingAnswer(model, parsed.value, Number(target), Number(view))}. Inspect the accepted donor chain, or the absent category rule, to explain this result.</p><p>{model.capped ? 'Eight-round cap reached; further transfers may be possible.' : 'Stopped because no new offers remain.'} Reciprocal confirmations reuse earlier evidence; they do not create independent observations.</p></div>}
    </div>}
    <ExplanationPrompt />
  </section>;
}
