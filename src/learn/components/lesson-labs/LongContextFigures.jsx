import { NeuralTable, NeuralPlot, formatNeural as f } from './NeuralLessonElements.jsx';
import { latentRead, recurrenceTrace, composeAffine } from '../../data/long-context-models.js';
import './long-context-labs.css';

const Strip = ({ values }) => <div className="long-mini-strip">{values.map((v, i) => <span key={i}>{v}</span>)}</div>;
function Figure({ title, children }) { return <figure className="long-figure"><figcaption>{title}</figcaption>{children}</figure>; }
function Stage({ title, children }) { return <div className="long-stage"><strong>{title}</strong><span>{children}</span></div>; }

export function MemoryWorkspacesFigure() {
  return <Figure title="The same inputs leave three different kinds of workspace"><div className="long-workspaces">
    <section><h4>Retained records</h4><Strip values={['x₀', 'x₁', 'x₂', 'x₃', 'x₄']} /><p>Store ↓</p><Strip values={['h₀', 'h₁', 'h₂', 'h₃', 'h₄']} /><p>The next query can address each stored position separately.</p></section>
    <section><h4>Recurrent state</h4><Strip values={['x₀', 'x₁', 'x₂', 'x₃', 'x₄']} /><p>Update the same object at each step ↓</p><div className="long-state-slot">h₄ = update(h₃, x₄)</div><p>Only the current state remains. Earlier vectors are not separately addressable.</p></section>
    <section><h4>Latent workspace</h4><Strip values={['x₀', 'x₁', 'x₂', 'x₃', 'x₄']} /><p>Two different weighted reads ↓</p><Strip values={['z₁', 'z₂']} /><p>Both latents can consult the input again while it remains available.</p></section>
  </div><p>Arrows represent available information paths. Successful recall is a property to test.</p></Figure>;
}
export function MaskedReadFigure() {
  return <Figure title="Remove a future item from both numerator and denominator"><div className="long-weight-read">{[[0, 2, 2, '2/3', '4/3'], [1, 1, 4, '1/3', '4/3'], [2, 3, 8, '0', '0 (future)']].map(([position, scoreWeight, value, weight, term]) => <div key={position}><span>position {position}<br />value {value}</span><span className={position === 2 ? 'long-blocked' : ''}>exp(score) {scoreWeight}<br />normalized {weight}</span><strong>term {term}</strong></div>)}</div><p>Legal denominator: 2 + 1 = <strong>3</strong>; output 4/3 + 4/3 = <strong>8/3</strong>.</p><p>All-legal comparison: denominator 6; output 16/3. Leaving the future score in the denominator gives 4/3 and is the wrong operation.</p></Figure>;
}
export function AttentionShapesFigure() {
  return <Figure title="Queries determine the number of output rows"><div className="long-flow"><Stage title="Q: L × dₖ">L questions, dₖ features each</Stage><b>×</b><Stage title="Kᵀ: dₖ × N">N available keys</Stage><b>→</b><Stage title="Scores: L × N">One normalized row per question</Stage><b>×</b><Stage title="V: N × dᵥ">Values for the same N keys</Stage><b>→</b><Stage title="Output: L × dᵥ">L answers, dᵥ features each</Stage></div><p>This is matrix multiplication. One selected query produces one complete output row.</p></Figure>;
}
export function SegmentLayersFigure() {
  return <Figure title="Segment memory crosses a boundary through the lower layer"><div className="long-segment-grid">
    <section><h4>Previous segment: positions 0–3</h4><Stage title="Lower layer ℓ−1">Retain its tail as memory M</Stage><p>Cached values → next segment's upper-layer read</p><small>Backward credit stops at the cached values.</small></section>
    <section><h4>Current segment: positions 4–7</h4><Stage title="Current lower layer ℓ−1">Current representations supply Q</Stage><p>Concatenate [stopgrad(M); current] for K/V ↓</p><Stage title="Upper layer ℓ">Only current queries get new outputs</Stage></section>
    <section><h4>Following segment: positions 8–11</h4><Stage title="Lower layer ℓ−1">Retain the most recent lower-layer tail again</Stage><p>Same forward path; separate truncated gradient boundary</p></section>
  </div><p>At the first current query, position 4 may read retained 0–3 plus itself. Positions 5–7 are stored with the segment but masked for that query. This is not a same-layer state loop.</p></Figure>;
}
export function PositionAddressFigure() {
  return <Figure title="Repeated local labels; different distances"><NeuralTable caption="One query, two position-1 labels" headers={['Record', 'Segment-local index', 'Global index', 'Distance from query']} rows={[[ 'Older key', 1, 1, 4], ['Current key and query', 1, 5, 0], ['Shifted older key', 1, 101, 4], ['Shifted current query', 1, 105, 0]]} /><div className="long-rulers"><span>key 1 ─── distance 4 ─── query 5</span><span>key 101 ─ distance 4 ─ query 105</span></div><p>Adding the same offset leaves relative distances unchanged. The lab's −β·distance is an illustrative recency bias, not the full Transformer-XL score.</p></Figure>;
}
export function RetentionFigure() {
  const ordinary = recurrenceTrace(Array.from({ length: 6 }, (_, i) => ({ x: i ? 0 : 1, input: 1, recurrence: .125 })));
  const hold = recurrenceTrace(Array.from({ length: 6 }, (_, i) => ({ x: i ? 0 : 1, input: 1, recurrence: i ? .001 : .125 })));
  return <Figure title="One impulse, two retention choices"><Strip values={['x₁ = 1', 'x₂ = 0', 'x₃ = 0', 'x₄ = 0', 'x₅ = 0', 'x₆ = 0']} /><NeuralPlot title="The recurrence gate controls decay through silence" xLabel="event" yLabel="state (feature units)" xDomain={[1, 6]} yDomain={[0, .65]} series={[{ label: 'r = 1/8 throughout', color: '#bcbcbc', values: ordinary.map((r, i) => [i + 1, r.state]) }, { label: 'r = .001 after first event', color: '#e2b55a', values: hold.map((r, i) => [i + 1, r.state]) }]} /><p>First injection: 0.6 in both cases. Final state: 0.196608 versus 0.594668384. Every later injection is zero; the difference is retention.</p></Figure>;
}
export function GriffinPathsFigure() {
  return <Figure title="Griffin combines two distinct memory paths"><div className="long-flow"><Stage title="Recurrent block R">Fixed-width state + short convolution history</Stage><b>→</b><Stage title="Recurrent block R">Another learned recurrent transformation</Stage><b>→</b><Stage title="Local attention A">Separately addressable recent K/V entries</Stage></div><div className="long-segment-grid"><section><h4>Inside R</h4><p>Projection → causal depthwise convolution → RG-LRU</p><p>Parallel projection → nonlinear gate</p><p>Multiply branches → output projection</p></section><section><h4>A's direct view</h4><Strip values={['evicted…', 'recent K/V', 'recent K/V', 'current']} /><p>Older information may arrive through recurrent state, not a distant direct attention edge.</p></section></div><p>Normalization, residual paths and a gated MLP surround the temporal mixer. Paper-specific settings: c = 8, convolution kernel 4, usual local window 1,024.</p></Figure>;
}
export function PerceiverReadFigure() {
  return <Figure title="Read a large array; do deep work in a smaller one"><div className="long-input-bank"><strong>T input rows remain available</strong><Strip values={['x₀', 'x₁', 'x₂', '…', 'xT−1']} /></div><div className="long-flow"><Stage title="N latent queries">Read input K/V: N × T scores</Stage><b>→</b><Stage title="N updated latents">Self-process: N × N scores</Stage><b>→</b><Stage title="Read inputs again">Updated queries; same retained input bank</Stage></div><p>Read requests go from latents to input keys; weighted values return to the latents. N output rows follow the N queries.</p></Figure>;
}
export function PositionAttachmentFigure() {
  const original = [-1, 0, 1].map((position, i) => ({ position, value: 2 + 4 * i })), query = [-Math.log(2), Math.log(2)];
  const paired = [original[2], original[0], original[1]], reassigned = original.map((r, i) => ({ ...r, value: paired[i].value }));
  return <Figure title="Moving records is different from changing their timestamps"><div className="long-workspaces">{[['Original', original], ['Whole-record rotation', paired], ['Values reassigned to fixed tags', reassigned]].map(([name, records]) => <section key={name}><h4>{name}</h4><Strip values={records.map(r => `tag ${r.position} ↔ value ${r.value}`)} /><p>Latent outputs: {latentRead(records, query).map(r => f(r.output, 6)).join(', ')}</p></section>)}</div><p>The first two representations agree. The third input assigns different values to the same temporal tags.</p></Figure>;
}
export function OutputQueriesFigure() {
  return <Figure title="Perceiver IO: output queries decide what and where to read"><div className="long-flow"><Stage title="Input array">Many observed records</Stage><b>→</b><Stage title="Two processed latents">Supply K/V to the decoder</Stage><b>←</b><Stage title="Four output requests">Coordinate queries q₁, q₂, q₃, q₄</Stage></div><Strip values={['q₁ → output₁', 'q₂ → output₂', 'q₃ → output₃', 'q₄ → output₄']} /><p>Four query rows yield four answer vectors. This is an array-shape illustration, with no invented optical-flow values.</p></Figure>;
}
export function CausalLatentsFigure() {
  return <Figure title="Every path to an autoregressive answer must stay causal"><NeuralTable caption="Cross-attention: ✓ is an allowed input" headers={['Latent / target', 'input 0', 'input 1', 'input 2', 'input 3', 'input 4']} rows={[[ 'z₃ → token 4', '✓', '✓', '✓', '✓', 'blocked'], ['z₄ → token 5', '✓', '✓', '✓', '✓', '✓']]} /><NeuralTable caption="Latent attention also needs its own mask" headers={['Reader', 'read z₃', 'read z₄']} rows={[[ 'z₃', '✓', 'blocked'], ['z₄', '✓', '✓']]} /><p className="long-forbidden">Forbidden detour: input 4 → z₄ → z₃ → prediction of token 4.</p><p>The first cross-mask is correct even in this failure. The second mask closes the indirect route.</p></Figure>;
}
export function TrajectoryArchitectureFigure() {
  return <Figure title="The actual fitted classifier's tensor route"><div className="long-flow"><Stage title="45 × 3 input">Measured x/y plus ordinal position tag</Stage><b>→</b><Stage title="45 × 24 projection">Valid-point mask applies before read softmax</Stage><b>→</b><Stage title="N × 24 latents">Input read → latent attention → MLP, repeated twice with shared weights</Stage><b>→</b><Stage title="24 pooled features">Mean across N latents</Stage><b>→</b><Stage title="15 logits">One score for each movement class</Stage></div><p>N is 1 or 4. This complete-path classification task uses every valid observed point; it does not require a causal mask. The following workbench draws the actual 45-point geometry and exposes the tagged rows.</p></Figure>;
}
const results = [['Mean coordinates', 45, 54], ['Ordered coordinates', 17, 22], ['N=1, seed 11', 36, 32], ['N=1, seed 29', 24, 25], ['N=4, seed 11', 29, 28], ['N=4, seed 29', 30, 34]];
export function TrajectoryEvidenceFigure() {
  return <Figure title="Measured errors: show every run, including the stronger simple baseline"><div className="long-measured-panels">{[['Validation', 50, 1], ['Test', 60, 2]].map(([title, count, column]) => <section key={title}><h4>{title} · error percent, 0–100</h4>{results.map(row => <div className="long-measured-row" key={row[0]}><span>{row[0]}</span><div className="long-measured-track"><i style={{ left: `${100 * row[column] / count}%` }} /></div><span>{row[column]}/{count} = {f(100 * row[column] / count, 1)}%</span></div>)}</section>)}</div><p>Measured selected checkpoints, not simulated curves. The mean baseline receives two features; the ordered baseline receives 90. Two seeds do not define a confidence interval.</p></Figure>;
}
export function DetachedMemoryFigure() {
  return <Figure title="Same forward value; two different derivative paths"><div className="long-segment-grid">{[['Detached', 'current multiplier m = 6', '6'], ['Connected', 'current multiplier 6 + earlier-state contribution 6', '12']].map(([title, gradient, result]) => <section key={title}><h4>{title}</h4><p>w = 3 → m = 2w = 6 → y = w·m = 18</p><p>Backward: {gradient}</p><strong>dy/dw = {result}</strong></section>)}</div><p>Detaching removes an earlier differentiation path. It keeps the cached value 6 in the current calculation.</p></Figure>;
}
export function MemoryBudgetsFigure() {
  const rows = [['Dense, one layer', '4096²', 16777216], ['Segments, one layer', '16 × 256 × 384', 1572864], ['Local, one layer', '4096 × 128', 524288], ['Latents: one read + 8 latent layers', '4096 × 32 + 8 × 32²', 139264]];
  return <Figure title="Count score interactions and retained bytes separately"><div className="long-log-axis"><span>2¹⁷ score cells</span><span>2²⁴ score cells</span></div>{rows.map(([name, formula, value]) => <div key={name} className="long-cost-row"><strong>{name}</strong><div className="long-cost-track"><i style={{ left: `${(Math.log2(value) - 17) / 7 * 100}%` }} /></div><span>{formula} = {value.toLocaleString('en-US')}</span></div>)}<p>Log₂ position axis; these are arithmetic counts, with rectangular upper counts where stated. No timing ranking is implied.</p><NeuralTable caption="One-layer full-MHA K/V storage, d=64 and float16" headers={['Retained positions', '2 × positions × 64 × 2 bytes']} rows={[[4096, '1,048,576 bytes = 1 MiB'], [128, '32,768 bytes = 32 KiB']]} /><p>Weights, activation history and allocator overhead are excluded. A score count is not the same allocation as a persistent K/V cache.</p></Figure>;
}
export function AffineScanFigure() {
  const a = [.5, 1], b = [.4, 2], c = [.4, 1.56], ab = composeAffine(b, a), bc = composeAffine(c, b), left = composeAffine(c, ab), right = composeAffine(bc, a);
  return <Figure title="Associative composition exposes a parallel-prefix route"><Strip values={['A: h ↦ .5h + 1', 'B: h ↦ .4h + 2', 'C: h ↦ .4h + 1.56']} /><div className="long-segment-grid"><section><h4>Group first two</h4><p>B∘A: {f(ab[0])}h + {f(ab[1])}</p><p>C∘(B∘A): {f(left[0])}h + {f(left[1])}</p></section><section><h4>Group last two</h4><p>C∘B: {f(bc[0])}h + {f(bc[1])}</p><p>(C∘B)∘A: {f(right[0])}h + {f(right[1])}</p></section></div><p>Combine maps in a tree, then read each prefix. Vector coefficients use coordinatewise products. This is an algorithm identity, not a measured device schedule.</p></Figure>;
}
