import { wilsonInterval } from '../../data/endtoend-models.js';
import './endtoend-intuition.css';

export function PerfectSampleIntervalFigure() {
  const interval = wilsonInterval(10,10,1.96);
  const x = p => 30 + 240*p;
  return <figure className="ete-intuition" data-figure="wilson-perfect-sample" data-constructed-fixture="Ten hypothetical independent successes, not this study's Wine scores">
    <p className="ete-intuition-title">10 successes / 10 hypothetical trials</p>
    <svg viewBox="0 0 300 88" role="img" aria-label={`Constructed ten successes from ten independent trials. Wilson interval about ${interval.lower.toFixed(3)} to 1, although observed fraction is 1.`}>
      <path d="M30 42H270" stroke="#888"/>
      <path d={`M${x(interval.lower)} 42H270`} stroke="#e8b44a" strokeWidth="7"/>
      <circle cx="270" cy="42" r="6" fill="#ddd"/>
      <text x={x(interval.lower)} y="22" textAnchor="middle">{interval.lower.toFixed(3)}</text>
      {[0,.5,1].map(p=><text key={p} x={x(p)} y="73" textAnchor="middle">{p}</text>)}
    </svg>
    <p className="ete-intuition-axis">Candidate population success rate</p>
    <figcaption>The observed fraction is one, but a small perfect sample does not establish a perfect population. Wilson's approximate 95% interval remains about [{interval.lower.toFixed(3)}, 1]. A naive plug-in normal interval uses zero estimated variance at this boundary and collapses to [1,1]. Neither interval repairs dependence or unrepresentative sampling.</figcaption>
  </figure>;
}
