import './landmark-family-intuition.css';

export function DenseFeatureReuseFigure() {
  return <figure className="landmark-family-intuition"><h4>New channels are appended; earlier channels remain available</h4><p>Constructed dense block: 8 initial channels, growth rate 3, four layers. Every layer reads all groups accumulated before it.</p>
    {[0, 1, 2, 3, 4].map(step => <section className="landmark-dense-row" key={step}><h5>{step === 0 ? 'Initial state' : `After layer ${step}`}: {8 + 3 * step} channels</h5><div><span>Original 8</span>{Array.from({ length: step }, (_, i) => <span key={i} className={i === step - 1 ? 'landmark-new-features' : ''}>L{i + 1}: 3</span>)}</div></section>)}
    <figcaption>For a simplified 3×3 dense layer that directly emits 3 channels, layer 1 has 9 × 8 × 3 = 216 weights; layer 4 has 9 × 17 × 3 = 459. Constant output growth does not imply constant per-layer cost. Actual DenseNet bottlenecks and transition compression add their own contracts.</figcaption>
  </figure>;
}

export function RegNetStageFigure() {
  const widths = Array.from({ length: 6 }, (_, j) => {
    const raw = 16 + 8 * j, exponent = Math.round(Math.log2(raw / 16));
    return { raw, exponent, width: 16 * 2 ** exponent };
  });
  return <figure className="landmark-family-intuition"><h4>A smooth width rule becomes repeated stages</h4><p>Constructed rule: uⱼ = 16 + 8j, then wⱼ = 16 × 2<sup>round(log₂(uⱼ/16))</sup>. Six blocks, indexed 0–5.</p>
    <ol className="landmark-width-steps">{widths.map(({ raw, exponent, width }, j) => <li key={j}><span>Block {j}</span><p>Proposal {raw}<br />Rounded exponent {exponent}</p><strong>{width} channels</strong><div className="landmark-width-bar" aria-hidden="true"><i style={{ width: `${width / 64 * 100}%` }} /></div></li>)}</ol>
    <figcaption>Consecutive equal widths form stages: 16 channels × 1 block, 32 × 3, then 64 × 2. This is a width-rule illustration, not a trained named RegNet. All widths here already divide by 8; a complete builder must also satisfy its channel rounding, bottleneck and group divisibility rules.</figcaption>
  </figure>;
}

export function FrozenFeatureGradientFigure() {
  return <figure className="landmark-family-intuition"><h4>Frozen weights still transmit a gradient to their input</h4><p>Two-pixel toy feature: φ(u) = u₁ + u₂, with fixed coefficients [1, 1]. Target y = [1, 1], so φ(y) = 2. Feature loss L = (φ(ŷ) − 2)².</p>
    <div className="landmark-feature-flow"><span>Trainable generator → ŷ</span><span>Fixed feature rule → φ(ŷ)</span><span>Compare with 2 → loss</span></div>
    <div className="landmark-family-panels"><section><h5>Gradient can pass through</h5><p>ŷ = [2, 1] → φ(ŷ) = 3 → L = 1</p><p>∂L/∂φ = 2(3 − 2) = 2<br />∂φ/∂ŷ = [1, 1]<br /><strong>∂L/∂ŷ = [2, 2]</strong></p><p>Continue through the generator to its parameters. The feature coefficients stay fixed.</p></section><section><h5>The feature can miss a change</h5><p>ŷ = [2, 0] → φ(ŷ) = 2 → <strong>L = 0</strong></p><p>The target is [1, 1]. Different pixels give the same sum. Pixelwise mean squared error is 1 despite zero feature loss.</p></section></div>
    <figcaption>Freezing φ's parameters differs from detaching φ(ŷ) or computing that branch under no_grad: either would break the generator's needed gradient. The target-feature branch can be computed without a gradient when y is fixed. Real feature networks retain much richer structure, but still define their own notions of similarity.</figcaption>
  </figure>;
}
