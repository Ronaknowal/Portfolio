import './gaussian-process-intuition.css';

export function LocallyPeriodicKernelFigure() {
  const periodic = distance => Math.exp(-2 * Math.sin(Math.PI * distance) ** 2);
  const points = decay => Array.from({ length: 241 }, (_, index) => {
    const distance = index / 40;
    const covariance = periodic(distance) * (decay ? Math.exp(-(distance ** 2) / 8) : 1);
    return `${38 + distance * 40},${165 - covariance * 135}`;
  }).join(' ');
  return <figure className="gp-intuition" data-concept="locally-periodic-kernel">
    <figcaption><strong>Recurrence and persistence are two different assumptions.</strong></figcaption>
    <p>Unit variance, period one day. The dashed periodic covariance returns to 1 every day; multiplying it by an RBF with decay length two days produces the amber curve.</p>
    <svg viewBox="0 0 308 220" role="img" aria-label="Periodic covariance peaks at one every day; locally periodic covariance peaks decay with elapsed days">
      {[0, 0.5, 1].map(value => <g key={value}><line x1="38" x2="278" y1={165 - value * 135} y2={165 - value * 135} stroke="#444" /><text x="30" y={170 - value * 135} textAnchor="end">{value}</text></g>)}
      <polyline points={points(false)} fill="none" stroke="#ccc" strokeDasharray="5 4" strokeWidth="2" /><polyline points={points(true)} fill="none" stroke="#e8b84b" strokeWidth="2.5" />
      {[0, 2, 4, 6].map(value => <text key={value} x={38 + value * 40} y="188" textAnchor="middle">{value}</text>)}<text x="38" y="18">covariance</text><text x="157" y="211" textAnchor="middle">separation in days</text>
    </svg>
    <p>Same-hour covariance after 2, 4 and 6 days is respectively e⁻⁰·⁵≈.6065, e⁻²≈.1353 and e⁻⁴·⁵≈.0111 in the product model. Periodicity says which phases resemble one another; the decay says how long that resemblance persists. These are computed covariance curves, not sampled functions or measurements.</p>
  </figure>;
}

export function GaussianEvidenceDirectionsFigure() {
  return <figure className="gp-intuition" data-concept="marginal-likelihood-directions">
    <figcaption><strong>A long length scale expects observations to move together.</strong></figcaption>
    <p>Same two observations y=[1,−1], inputs [0,2] and noise variance .25. Each ellipse is the one-standard-deviation contour of the joint observation Gaussian—not a 95% region. Amber is the actual observation vector.</p>
    <div className="gp-intuition-panels">{[0.3, 3].map(length => {
      const covariance = Math.exp(-2 / length ** 2);
      const together = 1.25 + covariance;
      const opposite = 1.25 - covariance;
      return <section key={length}><strong>Length ℓ={length}</strong>
        <svg viewBox="0 0 300 285" role="img" aria-label={`Joint Gaussian at length ${length}; together-direction variance ${together.toFixed(4)}, opposite-direction variance ${opposite.toFixed(4)}`}>
          <line x1="25" x2="275" y1="140" y2="140" stroke="#777" /><line x1="150" x2="150" y1="20" y2="260" stroke="#777" />
          <ellipse cx="150" cy="140" rx={60 * Math.sqrt(together)} ry={60 * Math.sqrt(opposite)} transform="rotate(-45 150 140)" fill="none" stroke="#ddd" strokeWidth="2" />
          <line x1="150" x2="210" y1="140" y2="200" stroke="#e8b84b" strokeDasharray="4 4" /><circle cx="210" cy="200" r="5" fill="#e8b84b" />
          <text x="278" y="134" textAnchor="end">y₁</text><text x="159" y="25">y₂</text>
          {[-1, 0, 1].map(value => <g key={value}><text x={150 + 60 * value} y="158" textAnchor="middle">{value}</text>{value !== 0 && <text x="140" y={145 - 60 * value} textAnchor="end">{value}</text>}</g>)}
        </svg>
        <p>Variance along together direction (1,1)/√2: <b>{together.toFixed(6)}</b><br />Variance along opposite direction (1,−1)/√2: <b>{opposite.toFixed(6)}</b></p>
      </section>;
    })}</div>
    <p>The residual vector points entirely along the opposite direction. Its squared length is 2, so the quadratic fit cost rᵀC⁻¹r is 2 divided by that direction's variance. The long-scale model squeezes precisely the direction these observations use. The determinant term also matters; this geometric fact explains the fit-cost change, not the entire model-selection decision by itself.</p>
  </figure>;
}

export function InducingResidualFigure() {
  const rows = [-1, 0, 1].map(x => ({ x, explained: Math.exp(-x * x), residual: 1 - Math.exp(-x * x) }));
  const trace = rows.reduce((sum, row) => sum + row.residual, 0);
  return <figure className="gp-intuition" data-concept="inducing-residual-variance">
    <figcaption><strong>One inducing value leaves uncertainty about how the function bends around it.</strong></figcaption>
    <p>Constructed RBF prior with length one and variance one. Training inputs are −1,0,1; retain one inducing variable u=f(0). Each column splits prior variance 1 into Q's diagonal and the residual diagonal of K−Q.</p>
    <div className="gp-inducing-bars">{rows.map(row => <section key={row.x}><div className="gp-inducing-column" aria-label={`At x=${row.x}, represented variance ${row.explained.toFixed(6)}, residual ${row.residual.toFixed(6)}`}><span className="gp-inducing-residual" style={{ height: `${100 * row.residual}%` }} /><span className="gp-inducing-explained" style={{ height: `${100 * row.explained}%` }} /></div><strong>x={row.x}</strong><p>Represented {row.explained.toFixed(3)}<br />Residual {row.residual.toFixed(3)}</p></section>)}</div>
    <p>White: Qᵢᵢ = k(xᵢ,0)²/k(0,0)=exp(−xᵢ²). Amber: remaining prior variance 1−Qᵢᵢ. Given u exactly, f(0) is known, but f(−1) and f(1) are not.</p>
    <p className="gp-intuition-result">Residual trace = {trace.toFixed(6)}. With noise variance .25, the bound subtracts trace/(2·.25) = {(trace / 0.5).toFixed(6)}.</p>
    <p>This residual is about what u represents before observing y; it is not the final noisy-data posterior variance. More useful inducing locations can reduce the unexplained directions. Simply discarding the amber portion would make the approximation appear more informative than this representation warrants.</p>
  </figure>;
}
