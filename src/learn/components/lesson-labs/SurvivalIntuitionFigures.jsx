import './survival-intuition.css';

export function TiedRiskWeightsFigure() {
  return <figure className="sv-intuition" data-figure="tied-risk-denominators">
    <p><strong>Two tied failures: what remains in the second denominator?</strong></p>
    <div className="sv-intuition-risk-rows">{[
      ['Breslow · first factor', 6.5, '6.5'], ['Breslow · second factor', 6.5, '6.5'],
      ['Efron · first factor', 6.5, '6.5'], ['Efron · second factor', 5, '6.5 − ½ × 3 = 5'],
    ].map(([name, weight, calculation]) => <div key={name}><span>{name}</span><div className="sv-intuition-track"><span style={{ width: `${weight / 6.5 * 100}%` }} /></div><strong>{calculation}</strong></div>)}</div>
    <figcaption>All bars use the same 0–6.5 risk-weight scale. The tied failures carry total weight 3. Efron subtracts half that mass before the second factor; it does not claim that a particular subject with weight 1.5 actually failed first. Both numerators remain the product 1×2. The figure isolates the denominator approximation before the full likelihood multiplies event-time contributions.</figcaption>
  </figure>;
}

export function TimeDilationFigure() {
  const quantiles = [.25, .5, .75].map(p => -Math.log(1 - p) / .1);
  const x = days => 30 + days * 9;
  return <figure className="sv-intuition" data-figure="aft-time-dilation">
    <p><strong>Time ratio 2 moves each quantile to twice its original day</strong></p>
    <svg viewBox="0 0 330 230" role="img" aria-label="Exponential reference has quartile event times 2.88, 6.93, 13.86 days. A time-doubled population has 5.75, 13.86, 27.73 days on the same clock.">
      <text x="30" y="24">Reference</text><text x="30" y="131">Time multiplied by 2</text>
      <path d="M30 48H300M30 157H300M30 192H300" stroke="#777" fill="none" />
      {quantiles.map((q, index) => <g key={q}><path d={`M${x(q)} 55L${x(2 * q)} 148`} stroke="#696969" strokeDasharray="4 4" /><circle cx={x(q)} cy="48" r="5" fill="#ddd" /><circle cx={x(2 * q)} cy="157" r="5" fill="#e8b44a" /><text x={x(2 * q)} y="180" textAnchor="middle">{['25%', '50%', '75%'][index]}</text></g>)}
      {[0, 10, 20, 30].map(day => <text x={x(day)} y="211" key={day} textAnchor="middle">{day}</text>)}
      <text x="165" y="229" textAnchor="middle">event time (days)</text>
    </svg>
    <figcaption>Calculated illustration: reference rate .1/day; event-time percentiles 25%, 50%, 75%. Matching dots denote the same percentile in two distributions, not the same subject's observed counterfactual. Medians move from 6.93 to 13.86 days. At day 10, the stretched survival reads the original curve at day 5: S₁(10)=S₀(5). AFT changes where you read the time axis.</figcaption>
  </figure>;
}

export function CompetingDenominatorFigure() {
  return <figure className="sv-intuition" data-figure="competing-risk-denominators">
    <p><strong>The two hazards divide by different probability masses</strong></p>
    <div className="sv-intuition-population" role="img" aria-label="Illustrative population: twenty percent event-free, thirty percent cause one, fifty percent cause two."><span style={{ width: '20%' }}>20%</span><span style={{ width: '30%' }}>30%</span><span style={{ width: '50%' }}>50%</span></div>
    <p className="sv-intuition-key">Amber: event-free · white: cause 1 · charcoal: cause 2</p>
    <div className="sv-intuition-denominator"><strong>Cause-specific denominator</strong><span>Event-free only: .20</span></div>
    <div className="sv-intuition-denominator"><strong>Subdistribution denominator</strong><span>No cause 1 yet: .20 + .50 = .70</span></div>
    <figcaption>Constructed snapshot, separate from the six-record lab. Both descriptions concern the same next infinitesimal cause-1 probability increment dF₁. Cause-specific hazard divides it by .20; subdistribution hazard divides it by .70. The latter bookkeeping includes cause-2 outcomes, although those subjects cannot experience a new first cause-1 event. This is why its denominator is not an ordinary physical risk set.</figcaption>
  </figure>;
}
