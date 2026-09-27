import './calibration-intuition.css';

export function PinballTiltFigure() {
  const x = value => 30 + value * 65;
  const y = value => 180 - value * 70;
  const loss = (prediction, tau) => prediction < 2 ? tau * (2-prediction) : (1-tau) * (prediction-2);
  return <figure className="cal-intuition" data-figure="pinball-asymmetric-cost"><svg viewBox="0 0 325 240" role="img" aria-label="Pinball loss against proposed prediction, with outcome fixed at two. Tau point eight penalizes underprediction more steeply; tau point two penalizes overprediction more steeply.">
    <path d="M30 30V180H290" fill="none" stroke="#888" />
    {[.2,.8].map(tau=><path key={tau} d={`M${x(0)} ${y(loss(0,tau))}L${x(2)} ${y(0)}L${x(4)} ${y(loss(4,tau))}`} fill="none" stroke={tau===.8?'#e8b44a':'#ccc'} strokeWidth="2.5" strokeDasharray={tau===.8?undefined:'5 4'} />)}
    {[0,1,2,3,4].map(v=><text key={v} x={x(v)} y="204" textAnchor="middle">{v}</text>)}
    <text x="37" y="25">loss</text><text x="159" y="229" textAnchor="middle">proposed prediction q; y=2</text>
  </svg><figcaption>Solid amber: τ=.8. Dashed neutral: τ=.2. At τ=.8, predicting 1 when y=2 costs .8; predicting 3 costs .2. The asymmetric cost pushes a fitted upper quantile upward. At τ=.5 the two sides have equal slope, giving absolute-error median fitting. These are single-outcome loss curves, not fitted quantiles.</figcaption></figure>;
}

export function JackknifePairedCandidatesFigure() {
  const rows = [{ name:'omit 1', prediction:10, residual:1 },{ name:'omit 2', prediction:14, residual:2 },{ name:'omit 3', prediction:8, residual:1 }];
  const x=value=>85+(value-6)*21;
  return <figure className="cal-intuition" data-figure="jackknife-paired-candidates"><svg viewBox="0 0 325 220" role="img" aria-label="Three constructed leave-one-out model predictions at one new input: ten with residual one, fourteen with residual two, and eight with residual one. Paired endpoints are nine and eleven, twelve and sixteen, seven and nine.">
    {rows.map((row,i)=>{const y=42+i*49; return <g key={row.name}><text x="8" y={y+5}>{row.name}</text><path d={`M${x(row.prediction-row.residual)} ${y}H${x(row.prediction+row.residual)}`} stroke="#e8b44a" strokeWidth="3"/><circle cx={x(row.prediction)} cy={y} r="5" fill="#ddd"/><text x={x(row.prediction-row.residual)} y={y+22} textAnchor="middle">{row.prediction-row.residual}</text><text x={x(row.prediction+row.residual)} y={y+22} textAnchor="middle">{row.prediction+row.residual}</text></g>;})}
    <path d="M85 186H295" stroke="#888"/>{[6,8,10,12,14,16].map(v=><text key={v} x={x(v)} y="207" textAnchor="middle">{v}</text>)}
  </svg><figcaption>Constructed intermediate values at one new input. Each center and residual belong to the same leave-one-out fit. Keep lower candidates [9,12,7] and upper candidates [11,16,9], then apply the method's finite-sample lower/upper order-statistic rules. Replacing all three centers by one final prediction discards the between-fit variation. These segments are intermediate candidates, not three separately guaranteed intervals.</figcaption></figure>;
}
