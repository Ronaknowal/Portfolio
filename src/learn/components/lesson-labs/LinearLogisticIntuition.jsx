import './linear-logistic-intuition.css';

const residuals = [0.1, 0.2, -0.7, 0.4];

export function NormalEquationBalanceFigure() {
  return <figure className="linear-intuition" data-concept="normal-equation-balance">
    <figcaption><strong>At the best line, neither allowed adjustment has a remaining first-order pull.</strong></figcaption>
    <div className="linear-balance-panels">
      {[
        { title: 'Move every prediction equally: change the intercept', terms: residuals, total: '0.1 + 0.2 − 0.7 + 0.4 = 0' },
        { title: 'Move each prediction in proportion to x: change the slope', terms: residuals.map((value, index) => value * index), total: '0 + 0.2 − 1.4 + 1.2 = 0' },
      ].map(panel => <section key={panel.title}>
        <p><strong>{panel.title}</strong></p>
        {panel.terms.map((term, index) => <div className="linear-balance-row" key={index}>
          <span>{String.fromCharCode(65 + index)}</span>
          <svg viewBox="0 0 180 22" role="img" aria-label={`Shipment ${String.fromCharCode(65 + index)} contribution ${term.toFixed(1)}`}>
            <line x1="90" x2="90" y1="0" y2="22" stroke="#8c8c8c" />
            <rect x={term < 0 ? 90 + term * 55 : 90} y="5" width={Math.abs(term) * 55} height="12" fill={term < 0 ? '#b6b6b6' : '#e8b84b'} />
          </svg>
          <span>{term.toFixed(1)}</span>
        </div>)}
        <p className="linear-balance-sum">{panel.total}</p>
      </section>)}
    </div>
    <p>Bars share one scale. Left means negative; right means positive. Residuals are observed minus fitted hours. The first sum is the intercept column dotted with the residuals; the second is the distance column dotted with them. Individual errors remain, but these two sums cancel.</p>
  </figure>;
}

export function LogisticLearningFlowFigure() {
  const probability = 1 / (1 + Math.exp(-1));
  const scoreGradient = probability - 1;
  return <figure className="linear-intuition" data-concept="logistic-learning-flow">
    <figcaption><strong>One observed late shipment asks the model to increase its score.</strong></figcaption>
    <ol className="linear-gradient-flow">
      <li><strong>Read the current belief</strong><span>x = 2, y = 1; b = −1, w = 1</span><span>z = −1 + 1 × 2 = 1</span><span>p = {probability.toFixed(3)}</span></li>
      <li><strong>Find the score's pull</strong><span>p − y = {scoreGradient.toFixed(3)}</span><span>A higher score reduces this row's loss.</span></li>
      <li><strong>Send that pull to the weight</strong><span>∂ℓ/∂w = (p − y) × 2</span><span>= {(scoreGradient * 2).toFixed(3)}</span><span>At rate 0.1: w → {(1 - 0.1 * scoreGradient * 2).toFixed(4)}</span></li>
    </ol>
    <p>This is one row's unpenalized contribution. The intercept receives p − y because its input is 1. A batch averages the contributions from all rows before one update; a penalty adds its own derivative.</p>
  </figure>;
}

export function PenaltyResponseFigure() {
  const x = value => 28 + (value + 2) * 60;
  const y = value => 145 - value * 60;
  const values = Array.from({ length: 81 }, (_, index) => -2 + index * 0.05);
  const rules = [
    { title: 'No penalty', formula: 'w = a', solve: value => value },
    { title: 'Ridge, λ = 0.5', formula: 'w = a / 1.5', solve: value => value / 1.5 },
    { title: 'Lasso, λ = 0.5', formula: 'w = sign(a) max(|a| − 0.5, 0)', solve: value => Math.sign(value) * Math.max(Math.abs(value) - 0.5, 0) },
  ];
  return <figure className="linear-intuition" data-concept="penalty-response">
    <figcaption><strong>Different penalties answer “how much evidence is enough?” differently.</strong></figcaption>
    <div className="linear-penalty-panels">{rules.map(rule => <section key={rule.title}>
      <p><strong>{rule.title}</strong><br />{rule.formula}</p>
      <svg viewBox="0 0 296 290" role="img" aria-label={`${rule.title}: coefficient w as unpenalized optimum a varies from minus two to two`}>
        <line x1="28" x2="268" y1="145" y2="145" stroke="#777" />
        <line x1="148" x2="148" y1="25" y2="265" stroke="#777" />
        <polyline points={values.map(value => `${x(value)},${y(rule.solve(value))}`).join(' ')} fill="none" stroke="#e8b84b" strokeWidth="3" />
        <text x="148" y="284" textAnchor="middle">a: −2 to 2 →</text>
        <text x="151" y="17">w: −2 to 2 ↑</text>
        <circle cx={x(0.3)} cy={y(rule.solve(0.3))} r="5" fill="#ededed" stroke="#080808" strokeWidth="2" />
      </svg>
      <p>For a = 0.3: <strong>w = {rule.solve(0.3).toFixed(1)}</strong></p>
    </section>)}</div>
    <p>Calculated for the scalar objective (w − a)²/2 plus the named penalty. Axes have equal units in all three panels. Ridge continuously shrinks the evidence; lasso creates a whole interval of evidence values that map to exactly zero.</p>
  </figure>;
}
