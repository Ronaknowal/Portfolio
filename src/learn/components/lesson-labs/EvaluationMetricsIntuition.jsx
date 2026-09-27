import './evaluation-metrics-intuition.css';

const durations = [1, 2, 3, 4, 10];
const objectives = [
  { name: 'Absolute error', unit: 'minutes', minimum: 3, maximum: 9, loss: (actual, forecast) => Math.abs(actual - forecast) },
  { name: 'Squared error', unit: 'minutes²', minimum: 4, maximum: 80, loss: (actual, forecast) => (actual - forecast) ** 2 },
  { name: '90% pinball loss', unit: 'minutes', minimum: 10, maximum: 4, loss: (actual, forecast) => actual >= forecast ? 0.9 * (actual - forecast) : 0.1 * (forecast - actual) },
];

export function ForecastTargetFigure() {
  return <figure className="metric-intuition">
    <h4>The best single number depends on what errors cost</h4>
    <p>Treat the five durations [1, 2, 3, 4, 10] as equally likely possible outcomes for one forecast context. Each curve averages the stated loss over all five outcomes.</p>
    <div className="metric-intuition-panels">{objectives.map(objective => {
      const meanLoss = forecast => durations.reduce((sum, value) => sum + objective.loss(value, forecast), 0) / durations.length;
      const path = Array.from({ length: 121 }, (_, index) => { const forecast = index / 10; return `${35 + forecast * 19},${180 - meanLoss(forecast) / objective.maximum * 145}`; }).join(' ');
      return <section key={objective.name}><h5>{objective.name}</h5>
        <svg viewBox="0 0 300 245" role="img" aria-label={`${objective.name}, mean loss minimized at forecast ${objective.minimum} minutes with value ${meanLoss(objective.minimum).toFixed(2)} ${objective.unit}.`}>
          <line x1="35" y1="20" x2="35" y2="180" stroke="currentColor" /><line x1="35" y1="180" x2="270" y2="180" stroke="currentColor" />
          <text x="28" y="38" textAnchor="end">{objective.maximum}</text><text x="28" y="185" textAnchor="end">0</text>
          <polyline points={path} fill="none" stroke="#e9b949" strokeWidth="2.5" />
          <circle cx={35 + objective.minimum * 19} cy={180 - meanLoss(objective.minimum) / objective.maximum * 145} r="5" fill="#e9b949" />
          {[0, 3, 4, 10, 12].map(value => <g key={value}><line x1={35 + value * 19} y1="180" x2={35 + value * 19} y2="187" stroke="currentColor" /><text x={35 + value * 19} y={value === 4 ? 216 : 202} textAnchor="middle">{value}</text></g>)}
          <text x="150" y="238" textAnchor="middle">One forecast, minutes</text>
        </svg>
        <p>Vertical: mean loss, {objective.unit}.<br /><strong>Best forecast: {objective.minimum}</strong><br />Loss there: {meanLoss(objective.minimum).toFixed(2)}</p>
      </section>;
    })}</div>
    <figcaption>The different vertical scales are labeled; compare each curve's minimizer, not their heights across panels. These exact empirical objectives explain median, mean and upper-quantile targets. They are not fitted-model performance curves.</figcaption>
  </figure>;
}

export function AggregationWeightFigure() {
  const rows = [
    { name: 'Class 0', count: 10, f1: 16 / 21 },
    { name: 'Class 1', count: 4, f1: 0.5 },
    { name: 'Class 2', count: 2, f1: 0 },
  ];
  return <figure className="metric-intuition">
    <h4>Choose what receives one vote before averaging</h4>
    <div className="metric-intuition-panels">{['macro', 'support'].map(mode => <section key={mode}>
      <h5>{mode === 'macro' ? 'Each class gets the same weight' : 'Each class gets its support weight'}</h5>
      {rows.map(row => {
        const weight = mode === 'macro' ? 1 / 3 : row.count / 16;
        return <div className="metric-intuition-weight" key={row.name}><p>{row.name}: weight {weight.toFixed(4)} × F1 {row.f1.toFixed(4)}</p><div className="metric-intuition-track"><span style={{ width: `${weight * 100}%` }} /></div><p>Contribution: {(weight * row.f1).toFixed(6)}</p></div>;
      })}
      <strong>Sum: {rows.reduce((sum, row) => sum + (mode === 'macro' ? 1 / 3 : row.count / 16) * row.f1, 0).toFixed(6)}</strong>
    </section>)}</div>
    <figcaption>Every weight bar has a fixed 0–1 scale. Class 2 contributes zero in both averages because its F1 is zero; its missing performance receives more weight under macro averaging. Micro averaging follows a different route: pool the individual decision counts first, then form the ratio.</figcaption>
  </figure>;
}
