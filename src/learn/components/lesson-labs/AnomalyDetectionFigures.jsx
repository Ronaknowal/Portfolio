import { firstCutIntervals, lofState } from '../../data/anomaly-detection-models';
import { Table, round } from './AnomalyDetectionShared.jsx';
import './anomaly-detection-labs.css';

/** §2. Which period supplies which decision, and which direction is closed. */
const periods = [
  { key: 'reference', name: 'Reference', x: 12, supplies: ['scaler and', 'detector'] },
  { key: 'calibration', name: 'Calibration', x: 122, supplies: ['threshold'] },
  { key: 'later', name: 'Later', x: 232, supplies: ['scores, then', 'alerts'] },
];
export function ProvenanceFigure() {
  return <figure className="ad-figure">
    <figcaption><strong>The reference defines the comparison, calibration chooses the action, and later data test the declared protocol</strong></figcaption>
    <svg viewBox="0 0 340 252" role="img" aria-label="Three periods in time order. Reference rows fit the scaler and detector; calibration rows set the threshold; later rows receive scores and alerts. Those alerts and the separately supplied event annotations meet only at evaluation. Annotations do not enter fitting, scoring or threshold selection. A dashed route from later rows back to the fit is crossed out.">
      <defs>
        <marker id="ad-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#8eb9a5" />
        </marker>
      </defs>
      {periods.map(period => <g key={period.key}>
        <rect className="ad-lane" x={period.x} y="74" width="96" height="34" rx="3" />
        {period.supplies.map((line, index) => (
          <text key={line} x={period.x + 48} y={period.supplies.length === 1 ? 95 : 89 + index * 12} textAnchor="middle">{line}</text>
        ))}
        <line className="ad-flow" x1={period.x + 48} y1="54" x2={period.x + 48} y2="72" />
        <rect className="ad-lane" x={period.x} y="20" width="96" height="34" rx="3" />
        <text className="ad-lane-label" x={period.x + 48} y="35" textAnchor="middle">{period.name}</text>
        <text x={period.x + 48} y="47" textAnchor="middle">period</text>
      </g>)}
      <text x="12" y="12">earlier</text>
      <text x="328" y="12" textAnchor="end">later</text>
      <path className="ad-flow is-blocked" d="M280,108 L280,136 L60,136 L60,110" />
      <g stroke="#da9c86" strokeWidth="2">
        <line x1="163" y1="130" x2="177" y2="142" />
        <line x1="177" y1="130" x2="163" y2="142" />
      </g>
      <text x="170" y="158" textAnchor="middle" fill="#da9c86">later rows never reach back into the fit</text>
      <rect className="ad-lane" x="12" y="196" width="112" height="34" rx="3" />
      <text x="68" y="211" textAnchor="middle">event</text>
      <text x="68" y="223" textAnchor="middle">annotations</text>
      <rect className="ad-lane" x="232" y="196" width="96" height="34" rx="3" />
      <text x="280" y="217" textAnchor="middle">evaluation</text>
      <path className="ad-flow" d="M328,91 L336,91 L336,176 L280,176 L280,194" />
      <line className="ad-flow" x1="124" y1="213" x2="230" y2="213" />
      <text x="176" y="205" textAnchor="middle">compare</text>
      <text x="170" y="247" textAnchor="middle">annotations enter evaluation only</text>
    </svg>
    <p>
      Fitting and thresholding are separate decisions taken from separate data. Fit a scaler on the whole series and the later rows have already influenced the early distances. The annotations arrive last of all: they judge the result, and they never help to choose it. Where rows are grouped by device, person or incident, the split has to respect those groups, because neighbouring rows can belong to one event even when their timestamps differ.
    </p>
  </figure>;
}

/** §3. The empty gap, before any formula. */
export function FirstCutFigure() {
  const values = [0, 1, 2, 3, 12];
  const { intervals } = firstCutIntervals(values);
  const project = value => 30 + 280 * (value - values[0]) / (values.at(-1) - values[0]);
  return <figure className="ad-figure">
    <figcaption><strong>An empty gap is an early opportunity</strong> · five reference positions, one uniform cut</figcaption>
    <svg viewBox="0 0 340 96" role="img" aria-label={`Positions 0, 1, 2, 3 and 12 on a line. A cut between 3 and 12 separates 12 alone and covers ${intervals.at(-1).probability.toString()} of the range; a cut between 0 and 1 separates 0 alone and covers ${intervals[0].probability.toString()}.`}>
      {intervals.map((interval, index) => {
        const left = project(interval.from.toNumber());
        const width = project(interval.to.toNumber()) - left;
        const isolating = interval.left.length === 1 || interval.right.length === 1;
        return <rect key={index} className={`ad-span${isolating ? ' is-selected' : ''}`} x={left} y="20" width={width} height="34" />;
      })}
      <line className="ad-axis" x1="30" x2="310" y1="54" y2="54" />
      {values.map(value => <g key={value}>
        <circle className={`ad-point${value === 12 ? ' is-query' : ''}`} cx={project(value)} cy="54" r="5" />
        <text x={project(value)} y="70" textAnchor="middle">{value}</text>
      </g>)}
      <text x={(project(3) + project(12)) / 2} y="16" textAnchor="middle">{intervals.at(-1).probability.toString()} of the range separates 12</text>
      <text x={project(0.5)} y="88" textAnchor="middle">{intervals[0].probability.toString()}</text>
    </svg>
    <p>
      The cut is drawn uniformly between the smallest and largest position, so a gap's share of the range is its probability. The gap between 3 and 12 is nine of the twelve units, so three cuts in four separate 12 immediately. Isolating 0 needs the cut to land in a single unit, one chance in twelve. Nothing here has learned a label: the gap alone gives 12 its head start.
    </p>
  </figure>;
}

/** §4. Whose radius is it? The floor belongs to the neighbour. */
const reachReferences = [0, 1, 2, 20, 24, 28];
export function ReachFloorFigure() {
  const state = lofState(reachReferences, 2);
  const near = state.rows[1];
  const far = state.rows[4];
  const project = value => 28 + 288 * (value + 2) / 32;
  return <figure className="ad-figure">
    <figcaption><strong>The floor under a distance belongs to the neighbour, not to the query</strong> · k = 2 on two groups with different spacing</figcaption>
    <svg viewBox="0 0 340 146" role="img" aria-label={`Six reference rows at 0, 1, 2, 20, 24 and 28. With k = 2 the tight group's radii are ${state.rows.slice(0, 3).map(row => row.radius.toString()).join(', ')} and the loose group's are ${state.rows.slice(3).map(row => row.radius.toString()).join(', ')}. The row at 1 has radius ${near.radius.toString()}; the row at 24 has radius ${far.radius.toString()}.`}>
      <text x="16" y="16">the same k = 2, two very different radii</text>
      <line className="ad-radius" x1={project(24)} x2={project(28)} y1="44" y2="44" />
      <line className="ad-axis" x1={project(24)} x2={project(24)} y1="38" y2="50" />
      <line className="ad-axis" x1={project(28)} x2={project(28)} y1="38" y2="50" />
      <text x={project(24) - 8} y="48" textAnchor="end">r(24) = {far.radius.toString()}</text>
      <line className="ad-radius" x1={project(1)} x2={project(2)} y1="72" y2="72" />
      <line className="ad-axis" x1={project(1)} x2={project(1)} y1="66" y2="78" />
      <line className="ad-axis" x1={project(2)} x2={project(2)} y1="66" y2="78" />
      <text x={project(2) + 8} y="76">r(1) = {near.radius.toString()}</text>
      <line className="ad-grid" x1={project(24)} x2={project(24)} y1="50" y2="96" />
      <line className="ad-grid" x1={project(1)} x2={project(1)} y1="78" y2="96" />
      <line className="ad-axis" x1="16" x2="324" y1="100" y2="100" />
      {reachReferences.map(value => <circle key={value} className="ad-point" cx={project(value)} cy="100" r="5" />)}
      <text x={project(1)} y="118" textAnchor="middle">0, 1, 2</text>
      <text x={project(20)} y="118" textAnchor="middle">20</text>
      <text x={project(24)} y="118" textAnchor="middle">24</text>
      <text x={project(28)} y="118" textAnchor="middle">28</text>
      <text x={project(1)} y="136" textAnchor="middle">tight group</text>
      <text x={project(24)} y="136" textAnchor="middle">loose group</text>
    </svg>
    <Table caption="Each reference row's second-nearest other row sets its radius, and that radius is the floor every distance to it must clear"
      headings={['row', 'its two neighbours', 'radius r(o)', 'lrd(o)', 'its own factor']}
      rows={state.rows.map(row => [
        `at ${round(row.value.toNumber())}`,
        row.neighbours.map(entry => round(entry.value.toNumber())).join(' and '),
        row.radius.toString(), row.density.toString(), row.factor.toString(),
      ])} />
    <p>
      Reachability is deliberately asymmetric: swapping the two points swaps whose radius applies. The floor stops a query that almost coincides with one reference from claiming an enormous local density on the strength of one tiny distance. Notice that even these six regular rows do not all score 1, so treating 1 as a pass mark would misread a finite-neighbourhood effect as a verdict.
    </p>
  </figure>;
}

/** §8. A percentile cannot always select the fraction it names. */
export function ThresholdRulerFigure() {
  const scores = [1, 1, 2, 4, 4];
  const project = value => 40 + 240 * (value - 0.5) / 4;
  const alerts = threshold => scores.filter(score => score > threshold).length;
  return <figure className="ad-figure">
    <figcaption><strong>Ties decide what a threshold can select</strong> · five calibration scores, alert when the score is strictly above</figcaption>
    <svg viewBox="0 0 340 104" role="img" aria-label={`Five calibration scores: 1, 1, 2, 4 and 4. A threshold at 4 flags none of them. A threshold at 2 flags the two rows scoring 4. No threshold flags exactly one row.`}>
      <line className="ad-axis" x1="30" x2="310" y1="62" y2="62" />
      {[0.5, 1, 2, 3, 4, 4.5].map(value => <text key={value} x={project(value)} y="78" textAnchor="middle">{value}</text>)}
      {scores.map((score, index) => (
        <circle key={index} className="ad-point" cx={project(score)} cy={62 - (scores.slice(0, index).filter(other => other === score).length) * 12} r="5" />
      ))}
      <line className="ad-threshold" x1={project(4)} x2={project(4)} y1="14" y2="62" />
      <text x={project(4) - 6} y="12" textAnchor="end">threshold 4 flags {alerts(4)}</text>
      <line className="ad-threshold" x1={project(2)} x2={project(2)} y1="30" y2="62" stroke="#8eb9a5" />
      <text x={project(2) - 6} y="28" textAnchor="end">threshold 2 flags {alerts(2)}</text>
      <text x="30" y="98">stacked dots share a score</text>
    </svg>
    <p>
      Every dot is one calibration row, and dots stacked above one another share a score. Asking for the top 20% of these five rows has no answer a score threshold can give. Setting it at 4 flags nothing, because the rule is strictly greater; lowering it to 2 flags both rows that scored 4. A rule that takes the top B rows can return exactly one, but it needs its own tie-breaking rule and it answers a different question from “alert above this score”.
    </p>
  </figure>;
}
