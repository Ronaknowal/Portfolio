import './boosting-intuition.css';

export function BoostingDirectionFigure() {
  const residuals = [-3, -3, -2, 2, 3, 3];
  return <figure className="boosting-intuition" data-concept="restricted-correction">
    <figcaption><strong>A small tree approximates six requests with two shared answers.</strong></figcaption>
    <p>Same six training rows as section 1; current prediction is 5 everywhere. Each row requests its own residual. The stump at x=3.5 can offer only one correction per side.</p>
    <div className="boosting-direction-key"><span>Neutral = requested change</span><span>Amber = available leaf correction</span></div>
    {residuals.map((residual, index) => {
      const correction = index < 3 ? -8 / 3 : 8 / 3;
      return <div className="boosting-direction-row" key={index}>
        <span>x={index + 1}</span>
        <svg viewBox="0 0 210 34" role="img" aria-label={`Row ${index + 1}: desired ${residual}, leaf correction ${correction.toFixed(3)}`}>
          <line x1="105" x2="105" y1="0" y2="34" stroke="#777" />
          <rect x={105 + Math.min(residual, 0) * 30} y="3" width={Math.abs(residual) * 30} height="10" fill="#aaa" />
          <rect x={105 + Math.min(correction, 0) * 30} y="20" width={Math.abs(correction) * 30} height="10" fill="#e8b84b" />
        </svg>
        <span>{residual} → {correction.toFixed(2)}</span>
      </div>;
    })}
    <p>Left means decrease; right means increase. Both bar types use the same scale. A rate of 0.5 applies half of the amber correction. Rows in one leaf still have different remaining errors, so another round can ask a different question.</p>
  </figure>;
}

export function BoostingCurvatureFigure() {
  const horizontal = value => 32 + value * 105;
  const vertical = value => 180 - value * 24;
  return <figure className="boosting-intuition" data-concept="curvature-step">
    <figcaption><strong>The same initial push can justify different step sizes.</strong></figcaption>
    <p>Two constructed leaf surrogates share gradient G=−2 and penalty λ=1. Their Hessian sums differ. Each curve is −2w + (H+1)w²/2, with the old loss subtracted.</p>
    <div className="boosting-curvature-panels">{[1, 4].map(curvature => {
      const best = 2 / (curvature + 1);
      const objective = weight => -2 * weight + (curvature + 1) * weight * weight / 2;
      return <section key={curvature}><p><strong>H={curvature}: best w={best}</strong></p>
        <svg viewBox="0 0 275 260" role="img" aria-label={`Quadratic with H ${curvature} is minimized at correction ${best}`}>
          <line x1="32" x2="242" y1="180" y2="180" stroke="#777" />
          <line x1="32" x2="32" y1="24" y2="209" stroke="#777" />
          <polyline points={Array.from({ length: 81 }, (_, index) => { const weight = index / 40; return `${horizontal(weight)},${vertical(objective(weight))}`; }).join(' ')} fill="none" stroke="#e8b84b" strokeWidth="3" />
          <circle cx={horizontal(best)} cy={vertical(objective(best))} r="5" fill="#eee" />
          <text x="26" y="185" textAnchor="end">0</text>
          <text x="26" y="209" textAnchor="end">−1</text>
          <text x="26" y="41" textAnchor="end">6</text>
          <text x="32" y="234" textAnchor="middle">0</text><text x="137" y="234" textAnchor="middle">1</text><text x="242" y="234" textAnchor="middle">2</text>
          <text x="137" y="256" textAnchor="middle">proposed correction w →</text>
        </svg>
      </section>;
    })}</div>
    <p>Both curves initially fall at rate −2. The more curved one turns upward sooner, so its minimizing correction is smaller. The shared axes show actual surrogate values; a logistic loss may depart from this quadratic away from the expansion point.</p>
  </figure>;
}

export function OrderedBoostingDependencyFigure() {
  return <figure className="boosting-intuition" data-concept="ordered-gradient-dependency">
    <figcaption><strong>The model used to judge row 4 determines its training residual.</strong></figcaption>
    <div className="boosting-prefix-comparison">
      <section><strong>Exclude row 4 from its judging model</strong><p>Rows 1, 2, 3 → prefix model M₃</p><p>Prediction for row 4 = 1</p><p>Then compare with its target 6</p><p className="boosting-prefix-result">Residual = 6 − 1 = 5</p></section>
      <section><strong>Use the full-prefix model instead</strong><p>Rows 1, 2, 3, 4 → model M₄</p><p>Prediction for row 4 = 1.5</p><p>Row 4 already helped fit this value</p><p className="boosting-prefix-result">Residual = 6 − 1.5 = 4.5</p></section>
    </div>
    <p>These are the constant-learner miniature's second-round inputs, not a CatBoost benchmark. The prefix-only route reveals why an eligible prefix matters: the target is used after producing that row's prediction, instead of already influencing the model used to judge it. Ordinary boosting intentionally fits training residuals; Ordered boosting changes this particular dependency.</p>
  </figure>;
}
