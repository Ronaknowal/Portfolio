import { intervalPatterns } from '../../data/pac-models.js';
import { LessonTable } from './LessonElements.jsx';
import './pac-intuition.css';

export function SauerExtensionFigure() {
  const patterns = intervalPatterns(3).map(row => row.join(''));
  const prefixes = [...new Set(patterns.map(row => row.slice(0, 2)))].sort();
  const rows = prefixes.map(prefix => {
    const extensions = patterns.filter(row => row.startsWith(prefix)).sort();
    return [prefix, extensions.join(', '), extensions.length === 2 ? 'one extra copy' : 'no extra copy'];
  });
  const extras = rows.filter(row => row[2] === 'one extra copy').length;
  return <figure className="pac-intuition" data-figure="sauer-pattern-extension"><LessonTable caption="Remove the last of three ordered points, then restore its label" headers={['First two labels', 'Allowed full patterns', 'Beyond counting the prefix once']} rows={rows} /><figcaption>There are {prefixes.length} distinct prefixes and {extras} prefixes with both extensions, giving {prefixes.length}+{extras}={patterns.length} full patterns. Prefix 10 cannot extend to 101. The extra-copy collection contains 00, 01 and 11 but not 10: its flexibility has dropped from the interval class's two-point shattering to at most one-point shattering. This small enumeration exposes the two collections in Sauer's recurrence.</figcaption></figure>;
}

export function FatMarginFigure() {
  const threshold = .5, margin = .1;
  const x = value => 25 + (value - .3) * 650;
  return <figure className="pac-intuition" data-figure="fat-shattering-margin"><svg viewBox="0 0 315 175" role="img" aria-label="Function output axis from .3 to .7, threshold .5, margin .1. Outputs must reach .6 to count as a positive margin witness or .4 to count as a negative margin witness. .55 is above the threshold but inside the margin gap.">
    <text x="15" y="20">At one fixed input</text>
    <rect x={x(threshold-margin)} y="42" width={x(threshold+margin)-x(threshold-margin)} height="64" fill="#282828" />
    <path d="M25 75H285" stroke="#aaa" /><path d={`M${x(threshold)} 36V114`} stroke="#e8b44a" strokeDasharray="4 4" />
    {[.3,.4,.5,.6,.7].map(v=><g key={v}><path d={`M${x(v)} 70V80`} stroke="#aaa"/><text x={x(v)} y="132" textAnchor="middle">{v.toFixed(1)}</text></g>)}
    <circle cx={x(.55)} cy="75" r="6" fill="#e8b44a" /><text x={x(.55)} y="57" textAnchor="middle">.55</text><text x="155" y="160" textAnchor="middle">function output f(x)</text>
  </svg><figcaption>The comparison threshold is .5 and the required separation γ is .1. Output .55 is above the threshold, but it cannot witness the positive margin condition f(x)≥.6. Output .65 can; output .35 satisfies the negative condition f(x)≤.4. Shattering asks for every requested pattern across all selected inputs with their thresholds fixed in advance. A single point crossing a threshold does not establish the dimension.</figcaption></figure>;
}
