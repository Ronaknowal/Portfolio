import { BackpropFigure, BackpropTable } from './BackpropShared.jsx';
import './backprop-intuition.css';

export function MatrixSensitivityFigure() {
  return <BackpropFigure id="matrix-sensitivity" title="One input reaches two outputs; one weight is reused by two rows" caption="Constructed linear example. Arrows carry local derivatives, not forward values. The incoming output sensitivities (2, −1) already include the rest of the loss.">
    <div className="backprop-intuition-fork"><div><strong>One input a₁ = 2</strong><p>Other input a₂ = −1</p></div><div className="backprop-intuition-paths"><p>↗ coefficient 3 → y₁ = 3a₁ + a₂ = 5<br /><strong>← contribution 2 × 3 = 6</strong></p><p>↘ coefficient −2 → y₂ = −2a₁ + 4a₂ = −8<br /><strong>← contribution (−1) × (−2) = 2</strong></p></div></div>
    <p>Collect at a₁: <strong>6 + 2 = 8</strong>. Increasing a₁ by ε changes the outputs by (3ε, −2ε); their first-order loss effect is 2(3ε) − 1(−2ε) = 8ε.</p>
    <BackpropTable caption="The same collection rule for a shared weight" headers={['Use of weight w', 'Input multiplying w', 'Output sensitivity', 'Contribution to w']} rows={[[1, 2, 2, '2 × 2 = 4'], [2, 5, -1, '5 × (−1) = −5'], ['Collect both uses', 'Two training rows', 'Sum contributions', '4 − 5 = −1']]} />
    <p>The first calculation sums across <strong>output columns</strong> for an input. The second sums across <strong>example rows</strong> for a shared weight. Those are the two contractions in GBᵀ and AᵀG.</p>
  </BackpropFigure>;
}

export function CrossEntropySignalFigure() {
  const probabilities = [.2, .3, .5];
  return <BackpropFigure id="ce-signal" title="Cross-entropy compares allocated probability with the target allocation" caption="One unweighted example, true class B. These are logit derivatives; a parameter gradient also multiplies by the derivatives of its logits. No batch averaging factor is needed for this one row.">
    <div className="backprop-intuition-shares" aria-label="Predicted probability: A 20%, B 30%, C 50%">{probabilities.map((probability, index) => <span key={index} style={{ flex: probability }}>{['A', 'B', 'C'][index]} · {probability}</span>)}</div>
    <BackpropTable caption="Probability minus one-hot target" headers={['Class', 'p − target', 'Logit derivative', 'Gradient descent direction']} rows={[[ 'A', '.2 − 0', '+.2', 'Lower this logit'], ['B (true)', '.3 − 1', '−.7', 'Raise this logit'], ['C', '.5 − 0', '+.5', 'Lower more strongly']]} />
    <p>The derivatives add to zero: .2 − .7 + .5 = 0. Shifting all logits equally changes no probability, so there is no loss sensitivity along that common-shift direction.</p>
  </BackpropFigure>;
}

export function MicrobatchWeightFigure() {
  const schemes = [
    { name: 'Desired full-batch mean', weights: [.2, .2, .2, .2, .2], equation: '(ℓ₁ + ℓ₂ + ℓ₃ + ℓ₄ + ℓ₅) / 5' },
    { name: 'Unweighted sum of two means', weights: [.5, .5, 1 / 3, 1 / 3, 1 / 3], equation: '(ℓ₁ + ℓ₂) / 2 + (ℓ₃ + ℓ₄ + ℓ₅) / 3' },
    { name: 'Means weighted by group size', weights: [.2, .2, .2, .2, .2], equation: '(2/5) × mean(first 2) + (3/5) × mean(last 3)' },
  ];
  return <BackpropFigure id="microbatch-weights" title="Each example should keep the same share of the objective" caption="Five examples split into groups of two and three. Column heights encode each example's coefficient on a common 0–0.5 scale. These are objective weights, not loss magnitudes.">
    {schemes.map(scheme => <div className="backprop-intuition-weight-row" key={scheme.name}><h4>{scheme.name}</h4><p>{scheme.equation}</p><div className="backprop-intuition-weight-bars">{scheme.weights.map((weight, index) => <div key={index}><span className="backprop-intuition-weight-number">{weight === 1 / 3 ? '⅓' : weight}</span><span className="backprop-intuition-weight-track"><span style={{ height: `${weight / .5 * 100}%` }} /></span><span>ℓ{index + 1}</span></div>)}</div></div>)}
    <p>In the unweighted sum, an example in the small group has 1.5 times the coefficient of an example in the large group. Dividing that sum by two rescales it but preserves the unequal weighting. Multiplying each group mean by its fraction of the five examples repairs both.</p>
  </BackpropFigure>;
}
