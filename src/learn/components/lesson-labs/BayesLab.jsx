import { useState } from "react";
import { betaDensity, betaCdf, betaQuantile } from "./math";
import "./lessons.css";

export default function BayesLab() {
  const [prior, setPrior] = useState(2);
  const [successes, setSuccesses] = useState(8);
  const [failures, setFailures] = useState(2);
  const a = prior + successes, b = prior + failures;
  const xs = Array.from({ length: 301 }, (_, i) => i / 300);
  const before = xs.map(x => betaDensity(x, prior, prior));
  const after = xs.map(x => betaDensity(x, a, b));
  const ymax = Math.max(...before, ...after) * 1.1;
  const low = betaQuantile(.025, a, b), high = betaQuantile(.975, a, b);
  const path = values => values.map((v, i) => `${i ? "L" : "M"}${45 + xs[i] * 460},${220 - v / ymax * 180}`).join(" ");
  return <section className="lesson-lab" aria-label="Bayesian updating lab">
    <h3>Change the evidence. Watch the uncertainty change.</h3>
    <p>Start with a symmetric Beta prior, then count successes and failures. The two curves share the same density axis; a taller density is not a probability above 100%.</p>
    <div className="lesson-controls">
      <label>Prior<select aria-label="Prior" value={prior} onChange={e => setPrior(+e.target.value)}><option value="1">Beta(1, 1): uniform</option><option value="2">Beta(2, 2): moderate centre preference</option><option value="20">Beta(20, 20): strong centre preference</option></select></label>
      <label>Successes: {successes}<input aria-label="Successes" type="range" min="0" max="80" value={successes} onChange={e => setSuccesses(+e.target.value)} /></label>
      <label>Failures: {failures}<input aria-label="Failures" type="range" min="0" max="20" value={failures} onChange={e => setFailures(+e.target.value)} /></label>
      <button type="button" onClick={() => { setPrior(2); setSuccesses(8); setFailures(2); }}>Reset example</button>
    </div>
    <div className="lesson-legend"><span>Dashed white: prior</span><span>Solid gold: posterior</span><span>Shaded: equal-tailed 95% credible interval</span></div>
    <svg viewBox="0 0 550 268" role="img" aria-label={`Prior Beta(${prior}, ${prior}), posterior Beta(${a}, ${b}), mean ${(a / (a + b)).toFixed(3)}`}>
      <rect x={45 + low * 460} y="35" width={(high - low) * 460} height="185" fill="#e2b55a" opacity=".1" />
      {[0, .5, 1].map(f => <g key={f}><line x1="45" x2="505" y1={220 - f * 180} y2={220 - f * 180} stroke="#333" /><text x="38" y={225 - f * 180} textAnchor="end">{(ymax * f).toFixed(1)}</text></g>)}
      <text x="45" y="20">Density</text>
      <path d={path(before)} fill="none" stroke="#ddd" strokeWidth="2" strokeDasharray="6 4" />
      <path d={path(after)} fill="none" stroke="#e2b55a" strokeWidth="3" />
      {[0, .25, .5, .75, 1].map(x => <text key={x} x={45 + x * 460} y="243" textAnchor="middle">{x}</text>)}
      <text x="275" y="265" textAnchor="middle">Unknown success probability θ</text>
    </svg>
    <div className="lesson-results" aria-live="polite">
      Posterior <strong>Beta({a}, {b})</strong> · Mean <strong>{(a / (a + b)).toFixed(3)}</strong><br />
      95% credible interval <strong>[{low.toFixed(3)}, {high.toFixed(3)}]</strong> · P(θ &gt; 0.70) <strong>{(1 - betaCdf(.7, a, b)).toFixed(3)}</strong>
    </div>
    <p className="lesson-note">Try 8 successes / 2 failures, then 80 / 20. The observed fraction is identical; the larger sample usually concentrates this posterior more tightly. Set both counts to zero: posterior equals prior. Raising the prior strength pulls the posterior toward 0.5.</p>
  </section>;
}
