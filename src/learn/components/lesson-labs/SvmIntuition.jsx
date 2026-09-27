import './svm-intuition.css';

export function HingeShortfallFigure() {
  const samples = [{ margin: -0.2, description: 'Wrong side' }, { margin: 0.6, description: 'Correct side, too close' }, { margin: 1.7, description: 'Beyond the requested margin' }];
  const horizontal = value => 42 + (value + 0.5) * 80;
  return <figure className="svm-intuition" data-concept="hinge-shortfall">
    <figcaption><strong>Hinge loss pays for the missing score margin, not just wrong labels.</strong></figcaption>
    {samples.map(sample => <section className="svm-hinge-row" key={sample.margin}>
      <p>{sample.description}: y f = {sample.margin}, loss = {Math.max(0, 1 - sample.margin).toFixed(1)}</p>
      <svg viewBox="0 0 285 65" role="img" aria-label={`Label-signed score ${sample.margin}; shortfall to one ${Math.max(0, 1 - sample.margin).toFixed(1)}`}>
        <line x1="42" x2="250" y1="24" y2="24" stroke="#666" />
        {[0, 1, 2].map(value => <g key={value}><line x1={horizontal(value)} x2={horizontal(value)} y1="17" y2="31" stroke="#ccc" /><text x={horizontal(value)} y="53" textAnchor="middle">{value}</text></g>)}
        {sample.margin < 1 && <line x1={horizontal(sample.margin)} x2={horizontal(1)} y1="24" y2="24" stroke="#e8b84b" strokeWidth="5" />}
        <circle cx={horizontal(sample.margin)} cy="24" r="5" fill="#eee" />
      </svg>
    </section>)}
    <p>Zero separates correct from incorrect sides; one marks the requested label-signed score. Amber length is max(0,1−y f). This axis is a normalized score, not an input-space distance: divide by ‖w‖ for the corresponding geometric distance when w is nonzero.</p>
  </figure>;
}

export function SvmObjectiveBoundsFigure() {
  const stages = [0, 0.1, 0.25].map(alpha => {
    const weight = 2 * alpha;
    const primal = weight * weight / 2 + 0.25 * 2 * Math.max(0, 1 - weight);
    const dual = 2 * alpha - weight * weight / 2;
    return { alpha, weight, primal, dual };
  });
  return <figure className="svm-intuition" data-concept="primal-dual-bounds">
    <figcaption><strong>A feasible solution and a lower bound squeeze the optimum from opposite sides.</strong></figcaption>
    <p>Same two training points: x=−1,y=−1 and x=1,y=1; C=0.25. Set both dual coefficients to α, w=2α and b=0. All three shown states are feasible.</p>
    {stages.map(stage => <section className="svm-bounds-state" key={stage.alpha}>
      <p>α₁=α₂={stage.alpha}, w={stage.weight}<br />D={stage.dual.toFixed(3)} ≤ optimum ≤ P={stage.primal.toFixed(3)}; gap {(stage.primal - stage.dual).toFixed(3)}</p>
      <svg viewBox="0 0 280 60" role="img" aria-label={`Objective bracket from ${stage.dual.toFixed(3)} to ${stage.primal.toFixed(3)}`}>
        <line x1="30" x2="250" y1="20" y2="20" stroke="#777" />
        <line x1={30 + stage.dual * 440} x2={30 + stage.primal * 440} y1="20" y2="20" stroke="#e8b84b" strokeWidth="5" />
        <circle cx={30 + stage.dual * 440} cy="20" r="5" fill="#e8b84b" />
        <circle cx={30 + stage.primal * 440} cy="20" r="7" fill="none" stroke="#eee" strokeWidth="2" />
        <text x="30" y="48" textAnchor="middle">0</text><text x="140" y="48" textAnchor="middle">0.25</text><text x="250" y="48" textAnchor="middle">0.5</text>
      </svg>
    </section>)}
    <p>Amber dot = feasible dual lower bound. White ring = attained primal cost. At the last state both equal 0.375, proving this training optimum. The dots coincide intentionally. This proves optimization, not future accuracy.</p>
  </figure>;
}

export function PrecomputedKernelFigure() {
  return <figure className="svm-intuition" data-concept="kernel-column-identity">
    <figcaption><strong>Every kernel column has a stored observation and coefficient attached to it.</strong></figcaption>
    <p>Linear-kernel classifier trained on A: x=0,label −1 and B: x=2,label +1. Signed coefficients are [−0.5,+0.5], with b=−1.</p>
    <div className="svm-kernel-rows">
      {[1, 3].map(query => <section key={query}>
        <strong>Query x={query}</strong>
        <span>Similarity to A: {query} × 0 = 0</span>
        <span>Similarity to B: {query} × 2 = {query * 2}</span>
        <span className="svm-kernel-result">Score: 0 × (−0.5) + {query * 2} × 0.5 − 1 = {query - 1}</span>
      </section>)}
    </div>
    <p>The query-by-training matrix is [[0,2],[0,6]]. Its two columns mean A then B. Reversing them without reversing the coefficients changes both scores. The first query is exactly on the boundary; the second is on the positive side. Query-to-query similarities cannot replace these required training references.</p>
  </figure>;
}
