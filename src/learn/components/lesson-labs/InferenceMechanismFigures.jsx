import {useId, useState} from 'react';
import {pairedSaving, laplacianRowExample} from '../../data/visual-review-models';
import './mechanism-figures.css';

export function IntervalDecisionFigure() {
  const [shift, setShift] = useState(0);
  const result = pairedSaving(shift);
  const x = value => 38 + (value + 1) * 40;
  return <figure className="mechanism-figure interval-decision" aria-label="Interval and practical threshold investigation">
    <figcaption><strong>Positive and useful are two different questions.</strong> Predict whether adding 1 ms to every saving will put the whole interval beyond the planned 1 ms threshold.</figcaption>
    <div className="mechanism-controls">
      <button type="button" aria-pressed={shift === 0} onClick={() => setShift(0)}>Original five pairs</button>
      <button type="button" aria-pressed={shift === 1} onClick={() => setShift(1)}>Add 1 ms to each saving</button>
    </div>
    <svg viewBox="0 0 360 175" role="img" aria-label={`Mean ${result.mean} ms; 95% t interval ${result.low.toFixed(3)} to ${result.high.toFixed(3)} ms. Reference zero and useful saving threshold one ms.`}>
      <rect x={x(1)} y="25" width={x(6) - x(1)} height="92" fill="#213126" />
      <line x1={x(0)} x2={x(0)} y1="25" y2="120" stroke="#ddd1bd" strokeDasharray="4 4" />
      <line x1={x(1)} x2={x(1)} y1="25" y2="120" stroke="#a3c7ae" strokeDasharray="2 3" />
      <text x={x(0)} y="16" textAnchor="middle">zero</text>
      <text x={x(1) + 9} y="16">1 ms useful threshold</text>
      <line x1={x(result.low)} x2={x(result.high)} y1="63" y2="63" stroke="#e6bc69" strokeWidth="5" />
      {[result.low, result.high].map(value => <line key={value} x1={x(value)} x2={x(value)} y1="52" y2="74" stroke="#e6bc69" strokeWidth="2" />)}
      <circle cx={x(result.mean)} cy="63" r="7" fill="#f2d38f" />
      <text x={x(result.mean)} y="97" textAnchor="middle">mean {result.mean.toFixed(1)} ms</text>
      <line x1={x(-1)} x2={x(6)} y1="120" y2="120" stroke="#a7977c" />
      {[-1, 0, 1, 2, 3, 4, 5, 6].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="120" y2="126" stroke="#a7977c" /><text x={x(value)} y="142" textAnchor="middle">{value}</text></g>)}
      <text x="180" y="166" textAnchor="middle">Population mean saving, old − new (ms)</text>
    </svg>
    <p className="mechanism-result" aria-live="polite">Differences: [{result.differences.join(', ')}]. 95% interval: [{result.low.toFixed(3)}, {result.high.toFixed(3)}] ms. {shift ? 'Zero is excluded, but savings below 1 ms remain inside the interval.' : 'Both zero and the 1 ms threshold lie inside the interval.'}</p>
    <p className="lesson-note">The dot is the estimate; the line is its interval. Green marks savings at least 1 ms, not a probability region. Adding a constant shifts the interval without changing its width. This uses the five-pair t calculation above, not the known-variance simulation.</p>
  </figure>;
}

export function BetaUpdateFigure() {
  return <figure className="mechanism-figure beta-update" aria-label="Two evidence counts update two Beta shape parameters">
    <figcaption><strong>Keep the two kinds of evidence separate.</strong> These ten symbols represent the ten observed visitors, not the prior.</figcaption>
    <div className="beta-observations" aria-label="Eight successes followed by two failures; display order is illustrative">
      {Array.from({length: 10}, (_, i) => <span key={i} className={i < 8 ? 'beta-success' : 'beta-failure'}>{i < 8 ? 'S' : 'F'}</span>)}
    </div>
    <p className="lesson-note">S = success (8); F = failure (2). The grouping makes counting easy; the shared-rate model uses the totals.</p>
    <div className="beta-parameter-lanes">
      <div><span>Prior shape <strong>α = 2</strong></span><b aria-hidden="true">+</b><span>Successes <strong>s = 8</strong></span><b aria-hidden="true">→</b><span>Posterior shape <strong>a = 10</strong></span></div>
      <div><span>Prior shape <strong>β = 2</strong></span><b aria-hidden="true">+</b><span>Failures <strong>f = 2</strong></span><b aria-hidden="true">→</b><span>Posterior shape <strong>b = 4</strong></span></div>
    </div>
    <p><strong>Beta(2, 2) → Beta(10, 4).</strong> Posterior mean = 10 / (10 + 4) ≈ 0.714. The prior shapes are model parameters, not four extra observed visitors. The derivation below explains why counts add.</p>
  </figure>;
}

export function LaplacianRowFigure() {
  const [split, setSplit] = useState(false);
  const uid = useId();
  const {center, neighbors, total} = laplacianRowExample(split);
  const positions = [[48, 65], [48, 193], [290, 130]];
  return <figure className="mechanism-figure laplacian-row" aria-label="Laplacian row disagreement investigation">
    <figcaption><strong>Read C’s row as three weighted disagreements.</strong> Predict which term changes if the right group has value −1 instead of +1.</figcaption>
    <div className="mechanism-controls"><button type="button" aria-pressed={!split} onClick={() => setSplit(false)}>Constant signal</button><button type="button" aria-pressed={split} onClick={() => setSplit(true)}>Two group values</button></div>
    <svg viewBox="0 0 360 250" role="img" aria-labelledby={uid}>
      <title id={uid}>C has value 1. {neighbors.map(n => `${n.name} has value ${n.value}, weight ${n.weight}, contribution ${n.contribution.toFixed(1)}.`).join(' ')} Row result {total.toFixed(1)}.</title>
      {neighbors.map((node, i) => <g key={node.name}>
        <line x1="160" y1="130" x2={positions[i][0]} y2={positions[i][1]} stroke={node.name === 'D' ? '#e6bc69' : '#a3b6a4'} strokeWidth={node.name === 'D' ? 2 : 4} />
        <circle cx={positions[i][0]} cy={positions[i][1]} r="29" fill="#1c2b21" stroke="#a3b6a4" />
        <text x={positions[i][0]} y={positions[i][1] - 4} textAnchor="middle">{node.name}</text><text x={positions[i][0]} y={positions[i][1] + 16} textAnchor="middle">{node.value > 0 ? '+' : ''}{node.value}</text>
      </g>)}
      <circle cx="160" cy="130" r="31" fill="#362918" stroke="#e6bc69" strokeWidth="2" /><text x="160" y="125" textAnchor="middle">C</text><text x="160" y="146" textAnchor="middle">+{center}</text>
      <text x="110" y="70" textAnchor="middle">w = 1</text><text x="110" y="205" textAnchor="middle">w = 1</text><text x="225" y="105" textAnchor="middle">w = 0.2</text>
      <text x="180" y="241" textAnchor="middle">Only C’s incident edges are drawn</text>
    </svg>
    <div className="laplacian-terms" aria-live="polite">{neighbors.map(n => <div key={n.name}><span>C − {n.name}</span><strong>{n.weight} × (1 − {n.value < 0 ? `(${n.value})` : n.value}) = {n.contribution.toFixed(1)}</strong></div>)}<p><strong>(Lx)<sub>C</sub> = {total.toFixed(1)}</strong></p></div>
    <p className="lesson-note">Weights multiply signal differences; line lengths only arrange the picture. E and F contribute zero to this row because neither is adjacent to C. These chosen signals illustrate matrix multiplication and need not be eigenvectors. The row result is signed; whole-graph energy uses squared differences.</p>
  </figure>;
}
