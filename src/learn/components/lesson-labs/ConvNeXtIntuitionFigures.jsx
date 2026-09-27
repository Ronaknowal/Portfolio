import './convnext-intuition.css';

export function ConvNeXtHeadOrderFigure() {
  return <figure className="convnext-intuition" data-figure="convnext-head-order" data-constructed-fixture="true">
    <div className="convnext-intuition-paths">
      <section><h4>Pool, then normalize</h4><p>Two locations: [1, 3] and [9, 5]</p><p className="convnext-intuition-step">↓ average each channel</p><p>[5, 4]</p><p className="convnext-intuition-step">↓ compare channels</p><strong>approximately [1, −1]</strong></section>
      <section><h4>Normalize, then pool</h4><p>Two locations: [1, 3] and [9, 5]</p><p className="convnext-intuition-step">↓ compare channels at each location</p><p>approximately [−1, 1] and [1, −1]</p><p className="convnext-intuition-step">↓ average each channel</p><strong>approximately [0, 0]</strong></section>
    </div>
    <figcaption>Constructed two-location, two-channel map. LayerNorm uses unit scale, zero shift and ε=10⁻⁶, so displayed ±1 values are approximate. Local normalization removes each location’s scale before pooling; the stronger second location can no longer dominate the pooled channel contrast. The reference head uses the pool-then-normalize order.</figcaption>
  </figure>;
}
