import { useId, useState } from 'react';
import './convolution-operator-intuition.css';

export function AdaptivePoolingBinsLab() {
  const [count, setCount] = useState(3), id = useId();
  const values = [1, 2, 3, 4, 5];
  const bins = Array.from({ length: count }, (_, i) => {
    const start = Math.floor(i * values.length / count), end = Math.ceil((i + 1) * values.length / count);
    return { start, end, mean: values.slice(start, end).reduce((sum, value) => sum + value, 0) / (end - start) };
  });
  return <figure className="conv-op-intuition">
    <h4>Adaptive bins are defined by boundaries, not equal-sized partitions</h4>
    <p>Fixed input [1, 2, 3, 4, 5]. Change the requested output count to see exactly which input indices each bin averages.</p>
    <label htmlFor={id}>Output count: <strong>{count}</strong></label>
    <input id={id} type="range" min="1" max="7" step="1" value={count} onChange={event => setCount(Number(event.target.value))} />
    <div className="conv-bin-head" aria-hidden="true">{values.map((value, i) => <span key={i}>i={i}<br />{value}</span>)}</div>
    {bins.map(({ start, end, mean }, i) => <section className="conv-bin-row" key={i}><p>Bin {i}: [{start}, {end}) → mean <strong>{Number(mean.toFixed(3))}</strong></p><div className="conv-bin-head" role="img" aria-label={`Bin ${i}, input indices ${Array.from({ length: end - start }, (_, j) => start + j).join(', ')}, each weighted by 1/${end - start}`}>{values.map((_, j) => <span key={j} className={j >= start && j < end ? 'conv-bin-active' : ''} aria-hidden="true">{j >= start && j < end ? `1/${end - start}` : '—'}</span>)}</div></section>)}
    <button type="button" onClick={() => setCount(3)}>Reset to three bins</button>
    <figcaption>Start = floor(i × 5 / output count); end = ceil((i + 1) × 5 / output count), excluding the end index. A column highlighted in several rows contributes to several outputs. Counts above five can repeat bins; this is averaging existing values, not interpolation of a new signal.</figcaption>
  </figure>;
}

export function BatchNormFoldingFigure() {
  return <figure className="conv-op-intuition">
    <h4>Absorb a fixed channel scale and offset into the filter</h4>
    <p>One constructed output channel: kernel [2, −1], bias 1, patch [3, 4]. Frozen BN values: μ = 1, v = 3.99, ε = 0.01, γ = 4, β = −3.</p>
    <div className="conv-op-panels"><section><h5>Convolution, then evaluation BN</h5><p>z = 2(3) − 4 + 1 = 3<br />√(v + ε) = 2<br />BN(z) = 4(3 − 1) / 2 − 3</p><strong>Output = 1</strong></section><section><h5>Fold constants, then convolve</h5><p>α = γ / √(v + ε) = 2<br />New kernel = [4, −2]<br />New bias = −3 + 2(1 − 1) = −3</p><strong>Output = 4(3) − 2(4) − 3 = 1</strong></section></div>
    <figcaption>Substitute z = Wx + b into α(z − μ) + β: distribute α over both Wx and b − μ. It is this algebra, with fixed statistics, that removes the separate BN operation. The scale multiplies every tap for this output channel. A negative γ would also reverse their signs.</figcaption>
  </figure>;
}
