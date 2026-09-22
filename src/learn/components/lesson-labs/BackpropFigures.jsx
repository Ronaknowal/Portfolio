import { useState } from 'react';
import { biasGradient, broadcastSensitivity, checkpointSchedule, directionalProducts } from '../../data/backprop-models.js';
import { backpropTraining } from '../../data/backprop-training.js';
import { BackpropFigure, BackpropSteps, BackpropTable, numberText as f } from './BackpropShared.jsx';

export function BackpropScalarFigure() {
  const [stage, setStage] = useState(0);
  const trace = [
    ['Multiply and add bias', 'z = 3 × 2 − 1 = 5.'],
    ['Apply ReLU', 'a = max(0, 5) = 5; local slope = 1.'],
    ['Evaluate half-square loss', 'error = 5 − 1 = 4; L = ½ × 4² = 8.'],
    ['Seed the loss', '∂L/∂L = 1.'],
    ['Pull back through the loss', '∂L/∂a = 4 × 1 = 4.'],
    ['Pull back through ReLU', '∂L/∂z = 4 × 1 = 4.'],
    ['Collect input and parameter sensitivities', '∂L/∂w = 4 × 2 = 8; ∂L/∂b = 4; ∂L/∂x = 4 × 3 = 12.'],
  ];
  return <BackpropFigure id="scalar" title="Forward values → local rules → backward sensitivities" caption="Exact scalar example. The full values and sensitivities stay visible; stepping highlights a computation stage. Loss 8 and weight gradient 8 have different meanings.">
    <div className="backprop-scalar-flow">
      {[
        ['Inputs', 'x = 2, w = 3, b = −1', 'x sensitivity 12; w sensitivity 8; b sensitivity 4', 6],
        ['Affine score', 'z = wx + b = 5', 'score sensitivity 4', 0],
        ['Activation', 'a = ReLU(z) = 5', 'activation sensitivity 4', 1],
        ['Loss', 'L = ½(a − 1)² = 8', 'loss seed 1', 2],
      ].map(([name, forward, backward, active], index) => <div key={name} className={`backprop-flow-node ${(stage === active || (index === 3 && stage === 3) || (index === 2 && stage === 4) || (index === 1 && stage === 5)) ? 'is-active' : ''}`}><strong>{name}</strong><span>{forward}</span><span className="bp-reverse">← {backward}</span></div>)}
    </div>
    <BackpropSteps index={stage} setIndex={setStage} count={trace.length} name="Scalar" /><p className="backprop-stage"><strong>{trace[stage][0]}.</strong> {trace[stage][1]}</p>
  </BackpropFigure>;
}

export function BackpropBroadcastFigure() {
  return <BackpropFigure id="broadcast" title="A shared bias collects a column, not an average" caption="The incoming array G already contains any downstream mean factor. Forward bias shape (2,) becomes three row uses; the reverse sum restores shape (2,). Column identity is shown by labels as well as color.">
    <div className="backprop-pair">{[0, 1].map(column => <div className="backprop-broadcast-column" key={column}>
      <h4>Feature column {column + 1}</h4><div className="backprop-bias">One bias b{column + 1}</div>
      <div className="backprop-fan" aria-label={`Bias ${column + 1} used in each of three rows`}><span>↙</span><span>↓</span><span>↘</span></div>
      <div className="backprop-cells">{broadcastSensitivity.map((row, index) => <div key={index}><span>Row {index + 1}</span><strong>G = {row[column]}</strong></div>)}</div>
      <div className="backprop-fan" aria-hidden="true"><span>↘</span><span>↓</span><span>↙</span></div><div className="backprop-bias">b{column + 1} sensitivity: {broadcastSensitivity.map(row => row[column]).join(' + ')} = <strong>{biasGradient[column]}</strong></div>
    </div>)}</div>
    <BackpropTable caption="Exact broadcast reversal" headers={['Bias feature', 'Three incoming row sensitivities', 'Gradient']} rows={[[1, '1 + 3 + 5', 9], [2, '2 + 4 + 6', 12]]} />
  </BackpropFigure>;
}

export function BackpropShapeFigure() {
  const stages = [
    ['Input → hidden score', 'X[N,D] × W₁[D,H] + b₁[H] → Z₁[N,H]', 'G₁[N,H] → Xᵀ[D,N]G₁[N,H] = W₁ sensitivity[D,H]; sum rows → b₁ sensitivity[H].', 'The sample index N is summed when collecting a shared weight or bias.'],
    ['Hidden score → activation', 'tanh(Z₁[N,H]) → A₁[N,H]', 'A₁ sensitivity ⊙ (1 − A₁²) → G₁[N,H].', 'Elementwise multiplication preserves N,H.'],
    ['Hidden activation → logits', 'A₁[N,H] × W₂[H,C] + b₂[C] → Z₂[N,C]', 'G₂[N,C] → A₁ᵀ[H,N]G₂[N,C] = W₂ sensitivity[H,C]; sum rows → b₂ sensitivity[C]; G₂[N,C]W₂ᵀ[C,H] → A₁ sensitivity[N,H].', 'The class index C is summed when passing sensitivity back to each hidden feature.'],
    ['Logits → scalar mean loss', 'Z₂[N,C] → softmax probabilities P[N,C] → mean CE scalar', 'Scalar seed 1 → G₂ = (P − onehot(y)) / N, shape [N,C].', 'The mean contributes 1/N exactly once. Labels y have shape [N].'],
  ];
  return <BackpropFigure id="shapes" title="Every backward operation returns the receiving object’s shape" caption="Weights use input × output storage. Each block pairs one forward step with its local pullback; follow the blocks downward for forward execution and upward for reverse execution. PyTorch nn.Linear stores transposed weights.">
    <div className="backprop-shape-lanes">{stages.map(([name, forward, reverse, contraction]) => <div className="backprop-shape-stage" key={name}><h4>{name}</h4><p><strong>Forward ↓</strong> {forward}</p><p><strong>Reverse ↑</strong> {reverse}</p><p>{contraction}</p></div>)}</div>
    <BackpropTable caption="Objects and their gradient shapes" headers={['Object', 'Shape', 'Gradient shape']} rows={[
      ['X', '[N,D]', '[N,D] when requested'], ['W₁', '[D,H]', '[D,H]'], ['b₁', '[H]', '[H]'], ['Z₁, A₁', '[N,H]', '[N,H]'], ['W₂', '[H,C]', '[H,C]'], ['b₂', '[C]', '[C]'], ['Z₂, P', '[N,C]', '[N,C]'], ['y (integer labels)', '[N]', 'not differentiated'], ['L (mean CE)', 'scalar', 'scalar seed 1'],
    ]} />
  </BackpropFigure>;
}

function TrainingChart({ rows, valueKey, maxStep, yDomain, yTicks, logarithmic, label }) {
  const transform = value => logarithmic ? Math.log10(value) : value;
  const px = value => 80 + value / maxStep * 230;
  const py = value => 215 - (transform(value) - yDomain[0]) / (yDomain[1] - yDomain[0]) * 180;
  return <svg className="backprop-svg" viewBox="0 0 338 275" role="img" aria-label={label}>
    {yTicks.map(value => <g key={value}><line x1="80" x2="310" y1={py(value)} y2={py(value)} className="bp-grid" /><text x="74" y={py(value) + 4} textAnchor="end">{logarithmic ? `10^${Math.log10(value)}` : f(value)}</text></g>)}
    <line x1="80" x2="310" y1="215" y2="215" className="bp-axis" /><line x1="80" x2="80" y1="35" y2="215" className="bp-axis" />
    <polyline points={rows.map(row => `${px(row.step)},${py(row[valueKey])}`).join(' ')} className="bp-current" />
    {rows.map(row => <circle key={row.step} cx={px(row.step)} cy={py(row[valueKey])} r="4" className="bp-target" />)}
    {[0, maxStep / 2, maxStep].map(step => <text key={step} x={px(step)} y="240" textAnchor="middle">{step}</text>)}
    <text x="195" y="266" textAnchor="middle">Parameter updates</text>
  </svg>;
}

export function BackpropTrainingFigure() {
  const { xorTraining, digitTraining } = backpropTraining;
  return <BackpropFigure id="training" title="Recorded CPU runs from the downloadable teaching engine" caption="These are saved float64 NumPy observations, not browser training. Dots are recorded samples; connecting segments are interpolation. The two tasks have different objectives. The digit count axis spans all 0–120 validation examples.">
    <div className="backprop-training-panel"><h4>XOR · Mean squared error, all 4 truth-table rows</h4><p>2 → 4 → 1 tanh; seed 3; learning rate 0.1; 2,000 full-batch updates. Vertical axis: log₁₀ MSE. The final positive loss is approximately 7.82 × 10⁻³⁰, a finite-arithmetic fit to these four rows.</p>
      <TrainingChart rows={xorTraining} valueKey="mse" maxStep={2000} yDomain={[-30, 1]} yTicks={[1e-30, 1e-20, 1e-10, 1]} logarithmic label="XOR recorded mean squared errors, logarithmic loss axis" />
      <BackpropTable caption="XOR observed losses and outputs, target order (0,1,1,0)" headers={['Updates', 'MSE', 'Outputs for (00,01,10,11)']} rows={xorTraining.map(row => [row.step, f(row.mse), row.outputs.flat().map(f).join(', ')])} />
    </div>
    <div className="backprop-training-panel"><h4>Digits · Mean training cross-entropy over 280 images</h4><p>64 → 16 → 10 tanh; seed 4; learning rate 0.2; 500 full-batch updates; stratified split seed 22; pixels / 16. Initial training CE is 3.597477, not forced to log(10).</p>
      <TrainingChart rows={digitTraining} valueKey="trainLoss" maxStep={500} yDomain={[0, 4]} yTicks={[0, 1, 2, 3, 4]} label="Digit training cross-entropy, linear loss axis" />
      <h4>Digits · Validation correct out of 120 images</h4><TrainingChart rows={digitTraining} valueKey="validationCorrect" maxStep={500} yDomain={[0, 120]} yTicks={[0, 30, 60, 90, 120]} label="Digit validation correct count out of 120" />
      <BackpropTable caption="Digit observations" headers={['Updates', 'Training CE / 280', 'Validation correct / 120']} rows={digitTraining.map(row => [row.step, f(row.trainLoss), `${row.validationCorrect} / 120`])} />
    </div>
  </BackpropFigure>;
}

function SignedProductBars({ labels, values }) {
  return <div className="backprop-bars">{values.map((value, index) => <div className="backprop-bar-row" key={labels[index]}><span>{labels[index]} = {f(value)}</span><div className="backprop-bar-track"><span className="backprop-zero" /><span className="backprop-bar" style={{ left: `${value >= 0 ? 50 : 50 + value / 4 * 50}%`, width: `${Math.abs(value) / 4 * 50}%` }} /></div></div>)}<p className="backprop-bar-scale"><span>−4</span><span>0</span><span>+4</span></p></div>;
}

export function BackpropProductsFigure() {
  const model = directionalProducts;
  return <BackpropFigure id="products" title="Push a direction forward; pull an output weighting backward" caption="Exact local correspondence for f = (x₁x₂, sin x₁, x₂²) at x = (0.3, 0.7). Signed bars use the same −4 to +4 coordinate scale. They are rates/sensitivities, not probabilities.">
    <BackpropTable caption="Jacobian: output rows × input columns" headers={['Output', '∂ / ∂x₁', '∂ / ∂x₂']} rows={[['x₁x₂', '0.7', '0.3'], ['sin x₁', f(Math.cos(0.3)), '0'], ['x₂²', '0', '1.4']]} />
    <div className="backprop-pair"><div><h4>JVP · 2 input coordinates → 3 output rates</h4><p>v = (1, 2). Each Jacobian row is dotted with v.</p><SignedProductBars labels={['f₁ rate', 'f₂ rate', 'f₃ rate']} values={model.jvp} /></div><div><h4>VJP · 3 output weights → 2 input sensitivities</h4><p>u = (1, −1, 2). Each Jacobian column is dotted with u.</p><SignedProductBars labels={['x₁ sensitivity', 'x₂ sensitivity']} values={model.vjp} /></div></div>
    <BackpropTable caption="Matrix-vector arithmetic" headers={['Product coordinate', 'Calculation', 'Result']} rows={[
      ['(Jv)₁', '.7 × 1 + .3 × 2', f(model.jvp[0])], ['(Jv)₂', 'cos(.3) × 1 + 0 × 2', f(model.jvp[1])], ['(Jv)₃', '0 × 1 + 1.4 × 2', f(model.jvp[2])], ['(Jᵀu)₁', '.7 × 1 − cos(.3) + 0 × 2', f(model.vjp[0])], ['(Jᵀu)₂', '.3 × 1 + 0 × (−1) + 1.4 × 2', f(model.vjp[1])],
    ]} />
    <p>Matching scalar products: uᵀ(Jv) = {model.left.toPrecision(16)}; (Jᵀu)ᵀv = {model.right.toPrecision(16)}.</p>
  </BackpropFigure>;
}

export function BackpropCheckpointFigure() {
  const [stage, setStage] = useState(0);
  const state = checkpointSchedule[stage];
  return <BackpropFigure id="checkpoint" title="Eight operations: keep boundaries, regenerate local inputs" caption="Idealized chain with equal-size outputs and pullbacks that need the operation’s input/output. Slots identify activation states, not measured bytes. Parameters, gradients and workspaces are not counted; the K + L/K model is approximate storage reasoning, not a memory benchmark.">
    <p>Saved boundaries: states 0, 4, 8. Solid amber = retained; hatch = regenerated; empty = absent. State 0 is the input; state n is the result of operation n. Every operation’s pullback needs states n−1 and n.</p>
    <ol className="backprop-checkpoint-chain">{Array.from({ length: 9 }, (_, i) => <li key={i} className={state.boundaries.includes(i) ? 'is-saved' : state.regenerated.includes(i) ? 'is-regenerated' : 'is-absent'}><strong>State {i}</strong><span>{state.boundaries.includes(i) ? 'Saved' : state.regenerated.includes(i) ? 'Regenerated' : 'Absent'}</span></li>)}</ol>
    <BackpropSteps index={stage} setIndex={setStage} count={checkpointSchedule.length} name="Storage" />
    <p className="backprop-stage"><strong>{state.title}.</strong> {state.note}</p>
    <p>Current distinct retained/regenerated activation slots: <strong>{new Set([...state.boundaries, ...state.regenerated]).size}</strong>. {state.reverse.length > 0 && `This stage just processed pullbacks ${state.reverse.join(' → ')} before releasing their inputs.`}</p>
    <BackpropTable caption="Full recomputation and reverse schedule" headers={['Stage', 'Dependency action']} rows={checkpointSchedule.map(row => [row.title, row.note])} />
    <button type="button" onClick={() => setStage(0)}>Reset checkpoint timeline</button>
  </BackpropFigure>;
}
