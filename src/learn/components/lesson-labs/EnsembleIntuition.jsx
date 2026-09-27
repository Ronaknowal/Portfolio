import './ensemble-intuition.css';

export function AdaBoostVoteBalanceFigure() {
  const error = 1 / 6;
  const optimum = Math.log((1 - error) / error) / 2;
  const horizontal = alpha => 44 + alpha * 140;
  const vertical = loss => 206 - loss * 150;
  const correctMass = alpha => (1 - error) * Math.exp(-alpha);
  const wrongMass = alpha => error * Math.exp(alpha);
  const curve = value => Array.from({ length: 81 }, (_, index) => {
    const alpha = index / 50;
    return `${horizontal(alpha)},${vertical(value(alpha))}`;
  }).join(' ');
  return <figure className="ensemble-intuition" data-concept="adaboost-vote-balance">
    <figcaption><strong>A stronger vote helps the rows this stump gets right and hurts the rows it gets wrong.</strong></figcaption>
    <p>Keep the first stump fixed, with weighted error ε=1/6. These curves are computed from its exact loss multiplier, not from a new training experiment.</p>
    <div className="ensemble-balance-layout">
      <svg viewBox="0 0 302 260" role="img" aria-label="Correct-row loss mass decreases with vote strength, wrong-row mass increases, and their sum is minimized at alpha 0.804719">
        {[0, 0.5, 1].map(value => <g key={value}><line x1="44" x2="268" y1={vertical(value)} y2={vertical(value)} stroke="#383838" /><text x="34" y={vertical(value) + 5} textAnchor="end">{value}</text></g>)}
        <line x1="44" x2="44" y1="32" y2="206" stroke="#777" />
        <line x1="44" x2="268" y1="206" y2="206" stroke="#777" />
        {[0, 0.8, 1.6].map(alpha => <text key={alpha} x={horizontal(alpha)} y="229" textAnchor="middle">{alpha}</text>)}
        <polyline points={curve(correctMass)} fill="none" stroke="#eee" strokeWidth="2" />
        <polyline points={curve(wrongMass)} fill="none" stroke="#aaa" strokeWidth="2" strokeDasharray="5 4" />
        <polyline points={curve(alpha => correctMass(alpha) + wrongMass(alpha))} fill="none" stroke="#e8b84b" strokeWidth="3" />
        <line x1={horizontal(optimum)} x2={horizontal(optimum)} y1={vertical(correctMass(optimum) + wrongMass(optimum))} y2="206" stroke="#e8b84b" strokeDasharray="3 4" />
        <circle cx={horizontal(optimum)} cy={vertical(correctMass(optimum) + wrongMass(optimum))} r="5" fill="#e8b84b" />
        <text x="44" y="21">loss multiplier</text><text x="158" y="252" textAnchor="middle">vote strength α</text>
      </svg>
      <div className="ensemble-balance-reading">
        <p><span className="ensemble-line-key ensemble-line-correct" />Correct rows: (5/6)e⁻ᵅ</p>
        <p><span className="ensemble-line-key ensemble-line-wrong" />Wrong rows: (1/6)eᵅ</p>
        <p><span className="ensemble-line-key ensemble-line-total" />Total Z: the sum of both</p>
        <p>At α=½ log 5≈0.804719, both groups contribute √5/6≈0.372678. Their total is √5/3≈0.745356.</p>
      </div>
    </div>
    <p>Initially the benefit of strengthening this mostly correct stump outweighs the damage. Eventually the growing wrong-row term catches up. At the minimum, one small extra increase saves exactly as much loss as it adds. This is why the derivative balances the two terms—and why the wrong rows hold half the newly normalized mass.</p>
  </figure>;
}

export function StackTrainingServingFigure() {
  return <figure className="ensemble-intuition" data-concept="stack-training-serving">
    <figcaption><strong>Train the combiner on honest predictions; serve it with refitted bases.</strong></figcaption>
    <p>The two columns below always mean [neighbor forecast, line forecast]. They retain that meaning even though the fitted base objects change.</p>
    <ol className="ensemble-lifecycle">
      <li><strong>Collect out-of-fold features</strong><span>For held-out A,D, fit bases only on B,C,E,F.</span><span>A contributes [2, −0.35] with target 1. Repeat for every fold to fill all six rows.</span></li>
      <li><strong>Learn and keep the combiner</strong><span>Fit on the six-row OOF matrix and its six targets.</span><span>Keep neighbor weight w≈0.228273; the line receives 1−w.</span></li>
      <li><strong>Refit the bases for serving</strong><span>Fit a fresh line and neighbor model using all A–F.</span><span>The saved bases now use all available training information. The combiner stays unchanged.</span></li>
      <li><strong>Answer a new query</strong><span>x=2.5 → full-data bases → [2, 4.333333]</span><span className="ensemble-intuition-result">Apply the stored rule: (1−w)·4.333333+w·2≈3.800695 hours.</span></li>
    </ol>
    <p>The fold-specific models construct a training dataset for the combiner. They are not the models ordinarily retained for serving this stack. A full-data base prediction on A would be an in-sample result; substituting it into A's OOF row would change the learning problem.</p>
  </figure>;
}

export function MixedStackScoresFigure() {
  return <figure className="ensemble-intuition" data-concept="mixed-stack-scores">
    <figcaption><strong>A meta-model gives each score its own meaning before combining it.</strong></figcaption>
    <p>Constructed binary example, independent of the measured API run. Suppose an already fitted logistic combiner has intercept −1 and coefficients [2, 0.5].</p>
    <div className="ensemble-score-branches">
      <section><strong>Base A: probability</strong><span>Class-1 probability = 0.8</span><span>Learned coefficient 2</span><b>Contribution: 2 × 0.8 = 1.6</b></section>
      <section><strong>Base B: margin</strong><span>Signed decision score = 2</span><span>Learned coefficient 0.5</span><b>Contribution: 0.5 × 2 = 1</b></section>
    </div>
    <p className="ensemble-intuition-result">Add contributions and intercept: −1 + 1.6 + 1 = 1.6<br />Apply the logistic link: 1/(1+e⁻¹·⁶)≈0.832018.</p>
    <p>The margin 2 is allowed because it is a feature, not a probability. A raw mean (0.8+2)/2=1.4 would not be a valid probability. Learned coefficients and the link define a new prediction rule; they do not guarantee its calibration. Their values must come from correctly owned meta-training data.</p>
  </figure>;
}
