import './recommender-intuition.css';

export function FixedFactorBowlFigure() {
  const cost = p => (4 - p) ** 2 + (2 - 2 * p) ** 2 + p ** 2;
  const x = p => 48 + p * 82;
  const y = value => 214 - value * 6;
  const points = Array.from({ length: 61 }, (_, index) => {
    const p = index / 20;
    return `${x(p)},${y(cost(p))}`;
  }).join(' ');
  return <figure className="rec-intuition" data-figure="fixed-factor-bowl">
    <p><strong>Freeze item factors; move just this user's one coordinate</strong></p>
    <svg viewBox="0 0 330 270" role="img" aria-label="The loss falls from 20 at user factor zero to nine and one third at four thirds, then rises. Item factors one and two remain fixed.">
      <path d="M48 35V214H300" fill="none" stroke="#8b8b8b" />
      {[0, 10, 20].map(value => <g key={value}><path d={`M44 ${y(value)}H300`} stroke="#303030" /><text x="37" y={y(value) + 5} textAnchor="end">{value}</text></g>)}
      {[0, 1, 2, 3].map(value => <text key={value} x={x(value)} y="236" textAnchor="middle">{value}</text>)}
      <polyline points={points} fill="none" stroke="#e8b44a" strokeWidth="3" />
      <circle cx={x(4 / 3)} cy={y(cost(4 / 3))} r="5" fill="#e8b44a" />
      <text x="57" y="23">squared error + penalty</text>
      <text x="166" y="126">minimum at p = 4/3</text>
      <text x="168" y="261" textAnchor="middle">user factor p</text>
    </svg>
    <figcaption>Constructed scalar block: fixed item factors 1 and 2, observed ratings 4 and 2, λ=1. Predictions are p and 2p. Loss is (4−p)² + (2−2p)² + p². Its slope 12p−16 becomes zero at p=4/3, where loss is 28/3. ALS finds this block's bottom before switching which side is allowed to move.</figcaption>
  </figure>;
}

function Matrix({ values }) {
  return <span className="rec-intuition-matrix" aria-label={values.map(row => row.join(', ')).join('; ')}>{values.flat().map((value, index) => <span key={index}>{value}</span>)}</span>;
}

export function ConfidenceGramFigure() {
  const terms = [
    ['Every item, weight 1', [[2, 1], [1, 2]]],
    ['Item 1: four extra copies', [[4, 0], [0, 0]]],
    ['Item 3: two extra copies', [[2, 2], [2, 2]]],
    ['Penalty λI', [[1, 0], [0, 1]]],
  ];
  return <figure className="rec-intuition" data-figure="confidence-gram-decomposition">
    <p><strong>The unobserved second item is still inside the first matrix</strong></p>
    <div className="rec-intuition-sum">{terms.map(([label, values], index) => <div key={label}><span className="rec-intuition-operator" aria-hidden="true">{index ? '+' : ''}</span><div><Matrix values={values} /><p>{label}</p></div></div>)}<div><span className="rec-intuition-operator">=</span><div><Matrix values={[[9, 3], [3, 5]]} /><p>User normal matrix</p></div></div></div>
    <figcaption>Same factors and counts as the confidence lab: q₁=(1,0), q₂=(0,1), q₃=(1,1), with weights 5, 1, 3. The all-item Gram gives one copy of every outer product. Sparse corrections add only 5−1 and 3−1 copies. The unobserved item contributes [[0,0],[0,1]] to the baseline; skipping its stored row does not erase its loss.</figcaption>
  </figure>;
}

export function PolicyMixtureFigure() {
  return <figure className="rec-intuition" data-figure="policy-mixture-reweighting">
    <p><strong>Reweight the opportunities to click, not the click probabilities</strong></p>
    {[
      { name: 'Logging mix', shares: [.8, .2], products: 'Expected clicks: .8 × .4 + .2 × .8 = .48' },
      { name: 'Target mix', shares: [.5, .5], products: 'Expected clicks: .5 × .4 + .5 × .8 = .60' },
    ].map(({ name, shares, products }) => <div className="rec-intuition-mixture" key={name}><strong>{name}</strong><div className="rec-intuition-bar" role="img" aria-label={`${name}: A ${shares[0] * 100} percent, B ${shares[1] * 100} percent`}>{shares.map((share, index) => <span key={index} style={{ width: `${100 * share}%` }}>{index ? 'B' : 'A'} {share * 100}%</span>)}</div><p>{products}</p></div>)}
    <figcaption>Each bar is one full probability mass. A's reward probability stays .4; B's stays .8. Multipliers .5/.8=.625 and .5/.2=2.5 turn the logging contributions .32 and .16 into target contributions .20 and .40. A rare logged B click counts more because B is more common in the target policy.</figcaption>
  </figure>;
}
