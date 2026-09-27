import './active-learning-intuition.css';

export function TargetVarianceFigure() {
  const candidates = [{ name: 'A', own: 4, covariance: 0.2 }, { name: 'B', own: 1, covariance: 0.8 }];
  return <figure className="active-intuition">
    <h4>An uncertain measurement can help little at the location you care about</h4>
    <p>Constructed valid GP covariance: target variance 1; noise variance 0.25; covariance between A and B is zero. All variances and covariances use squared response units.</p>
    <div className="active-intuition-panels">{candidates.map(candidate => {
      const reduction = candidate.covariance ** 2 / (candidate.own + 0.25);
      return <section key={candidate.name}><h5>Measure candidate {candidate.name}</h5>
        <p>Own variance: {candidate.own}<br />Covariance with target: {candidate.covariance}</p>
        <svg viewBox="0 0 300 160" role="img" aria-label={`Target variance before 1; after measuring ${candidate.name}, ${(1 - reduction).toFixed(6)}.`}>
          <text x="15" y="25">Before</text><rect x="15" y="35" width="270" height="22" fill="#d8d4cb" />
          <text x="15" y="85">After</text><rect x="15" y="95" width={270 * (1 - reduction)} height="22" fill="#d8d4cb" /><rect x={15 + 270 * (1 - reduction)} y="95" width={270 * reduction} height="22" fill="#e9b949" />
          <text x="15" y="145">0</text><text x="285" y="145" textAnchor="end">1 variance unit</text>
        </svg>
        <p>Reduction: {reduction.toFixed(6)}<br />Remaining: {(1 - reduction).toFixed(6)}</p>
      </section>;
    })}</div>
    <figcaption>White = remaining target variance; amber = reduction. A has the larger uncertainty about itself, but B is more informative about this target. These bars compute the stated GP formula; they do not show measured prediction error or refitted kernel parameters.</figcaption>
  </figure>;
}

export function GradientDiversityFigure() {
  const points = [{ name: 'A', z: [2, 0] }, { name: 'B', z: [1.9, 0.1] }, { name: 'C', z: [0, 2] }];
  const distances = points.slice(1).map(point => 0.32 * ((point.z[0] - 2) ** 2 + point.z[1] ** 2));
  const total = distances.reduce((sum, value) => sum + value, 0);
  return <figure className="active-intuition">
    <h4>Similar uncertainty can point toward different parameter changes</h4>
    <p>All three candidates have probabilities [0.6, 0.4], with pseudo-label 0. Their two class-gradient blocks are [−0.4z, +0.4z].</p>
    <div className="active-intuition-panels"><section>
      <svg viewBox="0 0 300 280" role="img" aria-label="Feature vectors A at (2,0), B at (1.9,0.1), C at (0,2). A and B point almost in the same direction; C differs.">
        <line x1="40" y1="225" x2="270" y2="225" stroke="currentColor" /><line x1="40" y1="225" x2="40" y2="20" stroke="currentColor" />
        {points.map(({ name, z }, index) => <g key={name}><line x1="40" y1="225" x2={40 + 90 * z[0]} y2={225 - 90 * z[1]} stroke={name === 'A' ? '#e9b949' : '#d8d4cb'} strokeWidth="2" strokeDasharray={name === 'B' ? '4 3' : undefined} /><circle cx={40 + 90 * z[0]} cy={225 - 90 * z[1]} r="4" fill={name === 'A' ? '#e9b949' : '#d8d4cb'} /><text x={index === 1 ? 183 : 50 + 90 * z[0]} y={index === 1 ? 193 : 221 - 90 * z[1]}>{name}</text></g>)}
        <text x="240" y="254">z₁</text><text x="13" y="25">z₂</text><text x="27" y="243">0</text>
      </svg>
      <p>This is the two-dimensional feature plane. Here the full gradient squared distance equals 0.32 times squared feature distance, so it preserves this geometry exactly up to scale.</p>
    </section><section><h5>After A is selected</h5>
      <table><caption>Distance-squared sampling for the next center</caption><thead><tr><th scope="col">Candidate</th><th scope="col">d² to A</th><th scope="col">Probability</th></tr></thead><tbody>{points.slice(1).map((point, index) => <tr key={point.name}><th scope="row">{point.name}</th><td>{distances[index].toFixed(4)}</td><td>{(distances[index] / total).toFixed(4)}</td></tr>)}</tbody></table>
      <p>B repeats almost the same gradient direction. C receives {(100 * distances[1] / total).toFixed(2)}% of this sampling probability. It is favored, not guaranteed to be selected.</p>
    </section></div>
    <figcaption>Constructed conditional sampling step with A already selected. For varying probabilities the gradient geometry need not preserve feature distances this simply. The true label remains unobserved; acquisition still asks the oracle before training on the selected item.</figcaption>
  </figure>;
}
