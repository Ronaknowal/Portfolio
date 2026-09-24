import { coolingState, formatOdeNumber } from '../../data/ordinary-differential-equations-models.js';
import './ordinary-differential-equations-labs.css';
export function CoolingAccumulationFigure() {
  const curve = Array.from({
    length: 61
  }, (_, index) => {
    const time = index / 60;
    const lossRate = -coolingState(time, 40, 0.2).derivative;
    return [48 + 216 * time, 176 - 16 * lossRate];
  });
  const path = curve.map(([x, y], index) => `${index ? 'L' : 'M'}${x},${y}`).join(' ');
  return <figure className="ode-plot ode-inline-figure"><svg viewBox="0 0 282 217" role="img" aria-label="Area under the decreasing loss-rate curve during the first minute"><title>A decreasing loss rate accumulates less loss than the initial-rate rectangle</title><text x="48" y="19">loss rate (K/min)</text><path d={`${path} L264,176 L48,176 Z`} fill="#335351" opacity="0.7" /><path d="M48,176 V48 H264 V176" fill="none" stroke="#ae9494" strokeDasharray="4 3" /><path d={path} stroke="#e7bd53" strokeWidth="2.5" fill="none" /><line x1="48" x2="264" y1="176" y2="176" stroke="#78858c" /><text x="39" y="180" textAnchor="end">0</text><text x="39" y="52" textAnchor="end">8</text><text x="48" y="195">0</text><text x="264" y="195" textAnchor="end">1</text><text x="158" y="213" textAnchor="middle">time (minutes)</text><text x="78" y="132">actual loss: 7.2508 K</text></svg><figcaption>The area under the positive loss rate 0.2x(t) is the lost temperature, 40−40e⁻⁰·² = {formatOdeNumber(40 - coolingState(1, 40, 0.2).state)} K. The dashed rectangle holds the initial 8 K/min rate for the whole minute and overestimates that loss. Axes and area therefore have different units.</figcaption></figure>;
}
export function BlowupDomainFigure() {
  const points = Array.from({
    length: 91
  }, (_, index) => {
    const time = index / 100;
    return [48 + 180 * time, 178 - 14 * (1 / (1 - time))];
  });
  return <figure className="ode-plot ode-inline-figure"><svg viewBox="0 0 282 221" role="img" aria-label="The solution 1 divided by 1 minus t rises toward its excluded time t equals 1"><title>The IVP solution stops before the pole; no branch is connected across it</title><text x="48" y="17">state x</text><line x1="48" x2="267" y1="178" y2="178" stroke="#87939a" /><line x1="228" x2="228" y1="26" y2="178" stroke="#d59d98" strokeDasharray="4 4" /><path d={points.map(([x, y], index) => `${index ? 'L' : 'M'}${x},${y}`).join(' ')} stroke="#e7bd53" strokeWidth="2.5" fill="none" /><circle cx="48" cy="164" r="3" fill="white" /><text x="40" y="168" textAnchor="end">1</text><text x="40" y="42" textAnchor="end">10</text><text x="48" y="197">0</text><text x="228" y="197" textAnchor="middle">1</text><text x="155" y="216" textAnchor="middle">time t</text></svg><figcaption>The plotted analytical samples run from t = 0 to 0.9; the state continues upward without bound as t approaches 1 from below. The dashed line is an excluded time, not a state reached by the solution. Its maximal interval through zero extends left to −∞ and right only to 1.</figcaption></figure>;
}
export function SpringForcesFigure() {
  return <figure className="ode-plot ode-inline-figure"><svg viewBox="0 0 282 192" role="img" aria-label="A displaced mass moving right with spring and damping forces to the left"><title>For positive displacement and velocity, both restoring and damping forces oppose rightward motion</title><line x1="26" x2="26" y1="32" y2="112" stroke="#9fa8ac" strokeWidth="3" /><path d="M26,78 H43 L49,62 L61,94 L73,62 L85,94 L97,62 L109,94 L121,78 H155" stroke="#e7bd53" strokeWidth="2" fill="none" /><rect x="155" y="49" width="53" height="58" fill="#252e32" stroke="#c9cdd0" /><text x="181" y="83" textAnchor="middle">m</text><line x1="34" x2="249" y1="114" y2="114" stroke="#67747b" /><path d="M179,33 H243 L237,28 M243,33 L237,38" stroke="#80c9cf" fill="none" strokeWidth="2" /><text x="183" y="18">v &gt; 0</text><path d="M154,139 H83 L89,134 M83,139 L89,144" stroke="#c3a1df" fill="none" strokeWidth="2" /><text x="114" y="162" textAnchor="middle">−kq and −cv</text><text x="178" y="184" textAnchor="middle">positive q →</text></svg><figcaption>This sign diagram fixes q &gt; 0 and v &gt; 0. The spring pulls toward equilibrium and damping opposes velocity, so both forces point left. If v changes sign, the damping force reverses even when q has not changed sign. An applied force F(t) is a separate signed input.</figcaption></figure>;
}
