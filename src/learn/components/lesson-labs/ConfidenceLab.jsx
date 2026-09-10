import { useState } from "react";
import { coverageIntervals } from "./math";
import { LessonTable } from "./LessonElements";

export default function ConfidenceLab() {
  const [n, setN] = useState(25);
  const [level, setLevel] = useState("95");
  const [batch, setBatch] = useState(0);
  const critical = { 90: 1.644853626951, 95: 1.959963984540, 99: 2.575829303549 }[level];
  const intervals = coverageIntervals(n, critical, batch);
  const covered = intervals.filter(d => d.low <= 100 && d.high >= 100).length;
  // Keep the usual axis stable for comparisons, extending it for extreme draws.
  const axisMin = Math.min(78, Math.floor(Math.min(...intervals.map(d => d.low)) / 10) * 10);
  const axisMax = Math.max(122, Math.ceil(Math.max(...intervals.map(d => d.high)) / 10) * 10);
  const ticks = Array.from({ length: Math.floor(axisMax / 10) - Math.ceil(axisMin / 10) + 1 }, (_, i) => (Math.ceil(axisMin / 10) + i) * 10);
  const scale = value => 45 + (value - axisMin) / (axisMax - axisMin) * 460;
  return <section className="lesson-lab" aria-label="Confidence interval coverage lab">
    <h3>Watch the intervals move, while the truth stays fixed</h3>
    <p>Each row is a fresh experiment. Here we know the population: independent Normal measurements with mean 100 ms and standard deviation 10 ms. This demonstration uses the known-standard-deviation z interval.</p>
    <div className="lesson-controls">
      <label>Observations per experiment: {n}<input aria-label="Observations per experiment" type="range" min="5" max="100" step="5" value={n} onChange={e => setN(+e.target.value)} /></label>
      <label>Confidence level<select aria-label="Confidence level" value={level} onChange={e => setLevel(e.target.value)}><option>90</option><option>95</option><option>99</option></select></label>
      <button type="button" onClick={() => setBatch(b => b + 1)}>Draw 40 new experiments</button>
    </div>
    <div className="lesson-legend"><span>Solid gold: covers 100 ms</span><span>Dashed pink: misses 100 ms</span><span>Vertical white: true mean</span></div>
    <svg viewBox="0 0 550 450" role="img" aria-label={`${covered} of 40 intervals contain 100 milliseconds at ${level}% confidence`}>
      <line x1={scale(100)} x2={scale(100)} y1="12" y2="380" stroke="#eee" />
      {intervals.map((d, i) => { const ok = d.low <= 100 && d.high >= 100; return <g key={i}>
        <line x1={scale(d.low)} x2={scale(d.high)} y1={18 + i * 9} y2={18 + i * 9} stroke={ok ? "#e2b55a" : "#ff9db2"} strokeWidth="2" strokeDasharray={ok ? undefined : "4 3"} />
        <circle cx={scale(d.mean)} cy={18 + i * 9} r="2" fill={ok ? "#e2b55a" : "#ff9db2"} />
      </g>; })}
      {ticks.map(t => <text key={t} x={scale(t)} y="400" textAnchor="middle">{t}</text>)}
      <text x="275" y="438" textAnchor="middle">Mean latency (ms)</text>
    </svg>
    <div className="lesson-results" aria-live="polite">This batch: <strong>{covered}/40 contain the truth ({(covered / 40 * 100).toFixed(1)}%)</strong>. Interval half-width: <strong>{(critical * 10 / Math.sqrt(n)).toFixed(2)} ms</strong>.</div>
    <p className="lesson-note">Predict, then try: increase n from 25 to 100. The width halves. Raising confidence widens intervals. A batch need not cover at exactly the requested percentage; coverage is a long-run property. Controls reuse the same random draws until you request a new batch.</p>
    <details><summary>Read the exact intervals as a table</summary><LessonTable caption="Simulated experiments (milliseconds)" headers={["Experiment", "Mean", "Lower", "Upper", "Covers 100?"]} rows={intervals.map((d, i) => [i + 1, d.mean.toFixed(2), d.low.toFixed(2), d.high.toFixed(2), d.low <= 100 && d.high >= 100 ? "Yes" : "No"])} /></details>
  </section>;
}
