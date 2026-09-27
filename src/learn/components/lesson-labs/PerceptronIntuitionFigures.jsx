import { batchNeuronExample, evaluateBatchNeuronExample, neuronOutputShares, perceptronUpdateExample } from '../../data/perceptron-intuition.js';
import './perceptron-intuition.css';

const number = value => Number(value.toFixed(4)).toString().replace('-', '−');
function Figure({ id, title, children }) {
  return <figure className="neuron-intuition" id={id}>
    <figcaption>{title}</figcaption>{children}
  </figure>;
}

export function MistakeCorrectionFigure() {
  const model = perceptronUpdateExample();
  return <Figure id="neuron-mistake-correction" title="A mistaken row supplies the direction of its own correction">
    <p>Constructed input (1, 2), desired label +1, step 0.5. The score starts at −1.</p>
    <div className="neuron-update-rows">
      {[
        ['Current parameters', [...model.weights, model.bias]],
        ['Add half the augmented input', [.5, 1, .5]],
        ['Updated parameters', [...model.nextWeights, model.nextBias]],
      ].map(([label, values]) => <div className="neuron-update-row" key={label}>
        <strong>{label}</strong><div>{values.map((value, i) => <span key={i}><small>{['w₁', 'w₂', 'bias'][i]}</small>{number(value)}</span>)}</div>
      </div>)}
    </div>
    <p className="neuron-calculation">New score: (−0.5) × 1 + 1 × 2 + 0.5 = <strong>{number(model.after)}</strong></p>
    <p>The first input changes its own contribution by 0.5; the second by 2; the bias by 0.5. Together they raise this row’s score by 3. For a mistaken negative label, the same correction is subtracted. Another row has different input products and need not improve.</p>
  </Figure>;
}

export function BatchNeuronCorrespondence() {
  const scores = evaluateBatchNeuronExample();
  return <Figure id="neuron-batch-correspondence" title="A batch adds examples, not extra neurons">
    <div className="neuron-matrix-layout">
      <div><h4>Two example rows X</h4><div className="neuron-matrix">{batchNeuronExample.inputs.flatMap((row, i) => row.map((value, j) => <span key={i + ':' + j}>{number(value)}</span>))}</div><p>Columns are input features.</p></div>
      <div><h4>Two neuron rules W</h4><div className="neuron-matrix">{batchNeuronExample.weights.flatMap((row, i) => row.map((value, j) => <span key={i + ':' + j}>{number(value)}</span>))}</div><p>Rows are neurons. Biases: −1 and 0.5.</p></div>
    </div>
    <div className="neuron-matrix-results">
      {scores.map((row, i) => <div key={i}><h4>Example {i + 1}: ({batchNeuronExample.inputs[i].join(', ')})</h4>
        <div className="neuron-matrix">{row.map((value, j) => <span key={j}><small>Neuron {j + 1}</small>{number(value)}<small>ReLU → {number(Math.max(0, value))}</small></span>)}</div>
      </div>)}
    </div>
    <p>Top-left result: 2 × 1.5 + (−1) × (−2) − 1 = 4. Top-right: 2 × (−1) + (−1) × 1 + 0.5 = −2.5. The second example uses those same two rules and biases. Transposing W puts each neuron’s rule in a column for <code>X @ W.T</code>.</p>
  </Figure>;
}

export function SoftmaxCompetitionFigure() {
  return <Figure id="neuron-softmax-competition" title="One score rises; every probability can change">
    {[[1, 2, 3], [1, 2, 4]].map(scores => {
      const result = neuronOutputShares(scores);
      return <div className="neuron-share-example" key={scores.join(',')}>
        <h4>Scores ({scores.join(', ')})</h4>
        <div className="neuron-share-strip" role="img" aria-label={'Probability shares: ' + result.shares.map(number).join(', ')}>
          {result.shares.map((share, i) => <span key={i} style={{ flexGrow: share }} />)}
        </div>
        <div className="neuron-share-labels">{result.shares.map((share, i) => <span key={i}>Class {i + 1}: <strong>{number(share)}</strong></span>)}</div>
      </div>;
    })}
    <p>Each strip is one unit of probability; segments retain class order. Only score 3 changed. Its larger exponential takes more of the shared total, reducing the other two shares. Their ratio stays e¹:e². Independent sigmoids would leave the first two outputs unchanged.</p>
  </Figure>;
}

export function SmoothGateDecomposition() {
  const rows = [
    { input: -1, relu: 0, gelu: .15865525393145707, silu: 1 / (1 + Math.E) },
    { input: 1, relu: 1, gelu: .8413447460685429, silu: 1 / (1 + Math.exp(-1)) },
  ];
  return <Figure id="neuron-smooth-gate" title="A hard cutoff becomes an input-dependent fraction">
    <p>Each bar shows the multiplier between 0 and 1. Multiply it by the signed input to obtain the activation. GELU values here use the standard-normal CDF.</p>
    {rows.map(row => <section key={row.input}><h4>Input z = {row.input}</h4>{[['ReLU', row.relu], ['GELU', row.gelu], ['SiLU', row.silu]].map(([name, multiplier]) =>
      <div className="neuron-gate-row" key={name}><span>{name}</span><div className="neuron-gate-track"><span style={{width: multiplier * 100 + '%'}} /></div><span>{number(multiplier)} × {row.input} = <strong>{number(multiplier * row.input)}</strong></span></div>
    )}</section>)}
    <p>At a negative input, a positive multiplier preserves a small negative output; it does not flip the sign. SwiGLU’s later multiplication can flip a separate value because its SiLU gate itself can be negative.</p>
  </Figure>;
}
