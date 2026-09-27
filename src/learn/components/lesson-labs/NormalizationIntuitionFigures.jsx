import './normalization-intuition.css';

export function NormalizationAxesFigure() {
  return <figure className="normalization-intuition" aria-label="LayerNorm statistics and parameter gradients reduce different axes"><h3>Statistics travel across features; parameter contributions collect across tokens</h3><p>One sequence, two tokens, three features per token. LayerNorm(3) gives each row its own mean and scale; the same three γ values serve both rows.</p><div className="normalization-axis-grid">
    <span /><strong>Feature 1<br />γ₁</strong><strong>Feature 2<br />γ₂</strong><strong>Feature 3<br />γ₃</strong>
    <strong>Token 1 →</strong><span className="normalization-axis-first">1</span><span className="normalization-axis-first">2</span><span className="normalization-axis-first">3</span>
    <strong>Token 2 →</strong><span className="normalization-axis-second">2</span><span className="normalization-axis-second">4</span><span className="normalization-axis-second">6</span>
    <strong>Collect γ gradients ↓</strong><span>d₁₁h₁₁<br />+ d₂₁h₂₁</span><span>d₁₂h₁₂<br />+ d₂₂h₂₂</span><span>d₁₃h₁₃<br />+ d₂₃h₂₃</span>
    </div><figcaption>h is the normalized input; d is the arriving output derivative. Row membership determines statistics. Column reuse determines γ-gradient sums. Across a batch, each column also collects the other examples' contributions. β collects d over the same parameter-sharing uses.</figcaption></figure>;
}

export function CausalNormalizationFigure() {
  const cases = [
    { name: 'Original two-token sequence', future: '[5, 7]', mean: 4, variance: 5, result: -3 / Math.sqrt(5 + 1e-5) },
    { name: 'Only the future token changes', future: '[15, 17]', mean: 9, variance: 50, result: -8 / Math.sqrt(50 + 1e-5) },
  ];
  return <figure className="normalization-intuition" aria-label="Normalizing over time lets a future token change an earlier output"><h3>A hidden route from the future into the present</h3>{cases.map(example => <div className="normalization-causal-case" key={example.name}><h4>{example.name}</h4><div className="normalization-causal-flow"><span>Present token<br /><strong>[1, 3]</strong></span><span>Future token<br /><strong>{example.future}</strong></span></div><p>↓ Shared statistics across all four values: μ = {example.mean}, v = {example.variance}</p><p>↖ First output = (1 − {example.mean}) / √({example.variance} + ε) ≈ <strong>{example.result.toFixed(6)}</strong></p></div>)}<figcaption>ε = 10⁻⁵, identity affine parameters. Time-and-feature normalization changes the present output even though its input is fixed. Per-token LayerNorm uses [1, 3] alone and keeps the first output at approximately −.999995 in both cases. A causal attention mask cannot block this separate statistics route.</figcaption></figure>;
}

export function RunningStatisticWeightsFigure() {
  const contributions = [
    { label: 'Initial buffer', weight: .729, statistic: 0 },
    { label: 'Batch 1', weight: .081, statistic: 4 },
    { label: 'Batch 2', weight: .09, statistic: 8 },
    { label: 'Batch 3', weight: .1, statistic: 2 },
  ];
  return <figure className="normalization-intuition" aria-label="Exponential running mean retains most initial weight after only three updates"><h3>Running statistics retain a fading history</h3><p>Momentum .1; initial mean 0; successive batch means 4, 8, 2. Expanding the update after three batches gives:</p><div className="normalization-memory-strip" aria-label="Weight shares: initial .729, first batch .081, second .09, third .1">{contributions.map((entry, index) => <span key={entry.label} style={{ flex: entry.weight }} className={`normalization-memory-${index}`} />)}</div><ol className="normalization-memory-key">{contributions.map(entry => <li key={entry.label}>{entry.label}: weight <strong>{entry.weight}</strong> × mean {entry.statistic} = {(entry.weight * entry.statistic).toFixed(3)}</li>)}</ol><p>Running mean = .324 + .720 + .200 = <strong>1.244</strong>. The initial zero still owns 72.9% of the weighting. It has not become an equal average of the three batches.</p><figcaption>The strip encodes weights, not statistic values. A batch's weight is multiplied by .9 after every subsequent update. This is fixed momentum, not the optional cumulative-average setting.</figcaption></figure>;
}
