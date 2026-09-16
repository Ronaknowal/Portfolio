import { trailNames, trailPoints, variedGroups, dbscan, sortedRoster, coreRadius, ringPoints } from '../../data/dbscan-models';
import { trailOptics, ringKmeans } from '../../data/dbscan-iris-data';
import { Glyph, GlyphLegend, Table, DenseStrip } from './DbscanLabs.jsx';
import './dbscan-labs.css';

const number = (value, digits = 4) => value === null || value === undefined ? 'undefined' : value === Infinity ? '∞' : Number.isInteger(value) ? String(value) : value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
const names = [...trailNames];
const trailX = trailPoints.map(p => p[0]);
const fitAt1 = dbscan(trailPoints, 1, 4);

/** Horizontal trail strip: x from −2.5 to 4.5 on a 640-wide viewBox. */
function trailPixel(x) { return 40 + (x + 2.5) * (560 / 7); }
function TrailStrip({ y, fit, selected, eps }) {
  return <g>
    <line x1={trailPixel(-2.5)} x2={trailPixel(4.5)} y1={y} y2={y} className="db-grid" />
    {[-2, -1, 0, 1, 2, 3, 4].map(v => <g key={v}><line x1={trailPixel(v)} x2={trailPixel(v)} y1={y - 4} y2={y + 4} className="db-grid" /><text x={trailPixel(v)} y={y + 43} textAnchor="middle">{v}</text></g>)}
    <rect x={trailPixel(trailX[selected] - eps)} y={y - 12} width={2 * eps * 80} height="24" fill="#e7b94a" fillOpacity=".12" />
    <line x1={trailPixel(trailX[selected] - eps)} x2={trailPixel(trailX[selected] + eps)} y1={y - 12} y2={y - 12} stroke="#e7b94a" />
    <circle cx={trailPixel(trailX[selected] - eps)} cy={y - 12} r="3" fill="#e7b94a" /><circle cx={trailPixel(trailX[selected] + eps)} cy={y - 12} r="3" fill="#e7b94a" />
    <text x={trailPixel(trailX[selected] - eps)} y={y - 30} textAnchor="middle">[{number(trailX[selected] - eps)}</text><text x={trailPixel(trailX[selected] + eps)} y={y - 30} textAnchor="middle">{number(trailX[selected] + eps)}]</text>
    {trailX.map((x, i) => <g key={i}><Glyph x={trailPixel(x)} y={y} type={fit.types[i]} />{i === selected && <circle cx={trailPixel(x)} cy={y} r="9" fill="none" stroke="#f2e7ca" strokeWidth="1" />}<text x={trailPixel(x)} y={y + (i % 2 === 0 ? -14 : 18)} textAnchor="middle" fill={fit.matrix[selected][i] <= eps ? '#f2e7ca' : '#8a9590'}>{names[i]}</text></g>)}
  </g>;
}
/** F1: one radius, two counts. */
export function TrailRosterFigure() {
  const rosterD = fitAt1.neighbors[3].map(i => names[i]).join(', '), rosterI = fitAt1.neighbors[8].map(i => names[i]).join(', ');
  return <figure className="db-figure">
    <figcaption><strong>One radius, two counts</strong> · ε = 1 meter, m = 4 rows; the closed interval of D and then of I, with each row counting itself</figcaption>
    <p className="db-legend"><strong>D selected:</strong> interval [−2, 0] holds {rosterD} → 5 rows ≥ 4, so D is core.</p>
    <div className="db-scroll"><svg className="db-svg-wide" viewBox="0 0 640 110" role="img" aria-label={`Trail strip with D selected. D's interval [−2, 0] contains ${rosterD}: five rows, at least four, so D is core.`}>
      <TrailStrip y={50} fit={fitAt1} selected={3} eps={1} />
    </svg></div>
    <p className="db-legend"><strong>I selected:</strong> interval [−1, 1] holds {rosterI} → 3 rows {'<'} 4, so I is not core.</p>
    <div className="db-scroll"><svg className="db-svg-wide" viewBox="0 0 640 110" role="img" aria-label={`Trail strip with I selected. I's interval [−1, 1] contains ${rosterI}: three rows, fewer than four, so I is not core, though its neighbours D and E are.`}>
      <TrailStrip y={50} fit={fitAt1} selected={8} eps={1} />
    </svg></div>
    <GlyphLegend />
    <p>Filled interval endpoints mean the boundary is included: a row exactly 1 meter away counts. The highlighted names are the rows inside each interval, and the selected row is always one of them, at distance 0 from itself. I falls short of four, yet it is not noise, because D and E inside its interval are core. Which row is selected is marked by the thin ring; the glyph shows its type.</p>
  </figure>;
}
/** F2: the bridge that cannot transmit. */
export function BorderTransmissionFigure() {
  const y = 60;
  const edge = (i, j, className) => <line key={`${i}-${j}`} className={className} x1={trailPixel(trailX[i])} y1={y} x2={trailPixel(trailX[j])} y2={y} />;
  const arcs = (i, j, className, up = true) => { const x1 = trailPixel(trailX[i]), x2 = trailPixel(trailX[j]); const mid = (x1 + x2) / 2; return <path key={`${i}-${j}-arc`} className={className} d={`M${x1},${y} Q${mid},${y + (up ? -34 : 34)} ${x2},${y}`} fill="none" />; };
  return <figure className="db-figure">
    <figcaption><strong>The bridge that cannot transmit</strong> · ε = 1, m = 4: two core components, a shared border I, and the tempting wrong graph</figcaption>
    <div className="db-scroll"><svg className="db-svg-wide" viewBox="0 0 640 260" role="img" aria-label="Top: core components A–D and E–H joined by solid edges, I hollow with dotted attachments to D and to E, J a cross at 4. Bottom: the same rows with a struck-through arrow from I to E, labelled I has only 3 neighbours and needs 4.">
      <text x="8" y="16">Correct: I may attach to either component; it links nothing</text>
      {[[0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [0, 3]].map(([i, j]) => i + 1 === j ? edge(i, j, 'db-edge') : arcs(i, j, 'db-edge', j - i === 2))}
      {[[4, 5], [5, 6], [6, 7], [4, 6], [5, 7], [4, 7]].map(([i, j]) => i + 1 === j ? edge(i, j, 'db-edge') : arcs(i, j, 'db-edge', j - i === 2))}
      {edge(3, 8, 'db-attach')}{edge(8, 4, 'db-attach')}
      <text x={trailPixel(-0.5)} y={y + 40} textAnchor="middle">may attach</text><text x={trailPixel(0.5)} y={y + 40} textAnchor="middle">may attach</text>
      {trailX.map((x, i) => <g key={i}><Glyph x={trailPixel(x)} y={y} type={fitAt1.types[i]} /><text x={trailPixel(x)} y={y + (i % 2 === 0 ? -14 : 18)} textAnchor="middle">{names[i]}</text></g>)}
      <text x="8" y="150">Wrong: using I as a transmitting bridge</text>
      {(() => { const yy = 195; return <g>
        {[[3, 8]].map(([i, j]) => <line key="di" className="db-edge" x1={trailPixel(trailX[i])} y1={yy} x2={trailPixel(trailX[j])} y2={yy} />)}
        <line className="db-wrong" x1={trailPixel(0)} y1={yy} x2={trailPixel(1) - 8} y2={yy} markerEnd="url(#dbArrow)" />
        <line className="db-wrong" x1={trailPixel(0.5) - 6} y1={yy - 10} x2={trailPixel(0.5) + 6} y2={yy + 10} /><line className="db-wrong" x1={trailPixel(0.5) - 6} y1={yy + 10} x2={trailPixel(0.5) + 6} y2={yy - 10} />
        <text x={trailPixel(0.5)} y={yy + 34} textAnchor="middle">I → E forbidden: I has only 3 neighbours; needs 4</text>
        {trailX.map((x, i) => <g key={i}><Glyph x={trailPixel(x)} y={yy} type={fitAt1.types[i]} /><text x={trailPixel(x)} y={yy + (i % 2 === 0 ? -14 : 18)} textAnchor="middle">{names[i]}</text></g>)}
      </g>; })()}
      <defs><marker id="dbArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0L10,5L0,10z" fill="#da9c86" /></marker></defs>
    </svg></div>
    <GlyphLegend components />
    <p>Within {'{A, B, C, D}'} and within {'{E, F, G, H}'} every pair is at most 0.75 apart, so each is one connected core component (the arcs are the non-adjacent pairs). D and E are 2 apart: no core edge. I is exactly 1 from both D and E, so it can attach to either component, but a border row has no outgoing arrow. Core, border and noise types and the two core components are the same under any visiting order; only I's final label can change.</p>
  </figure>;
}
/** F3: counts and radii are inverse views. */
export function CoreRadiusFigure() {
  const roster = sortedRoster(trailPoints, 3);
  const radii = coreRadius(trailPoints, 4);
  const sorted = [...radii].sort((a, b) => a - b);
  const x = rank => 40 + (rank - 1) * 30, y = value => 200 - 60 * value;
  return <figure className="db-figure">
    <figcaption><strong>Counts and radii are inverse views</strong> · D's sorted distances, then every row's fourth-nearest distance c₄ in ascending order</figcaption>
    <Table caption="D's distances to all ten rows including itself, sorted; the fourth entry is c₄(D)" headings={['rank', 'row', 'distance (m)']} rows={roster.map((entry, k) => [k + 1, names[entry.index], number(entry.distance)])} highlight={index => index === 3} />
    <div className="db-scroll"><svg className="db-svg-mid" viewBox="0 0 340 236" role="img" aria-label={`Sorted fourth-neighbour distances for the ten trail rows: ${sorted.map(value => number(value)).join(', ')}. A line at ε = 1 lies above eight of them, so eight rows are core at ε = 1.`}>
      {[0, 1, 2, 3].map(v => <g key={v}><line x1="34" x2="320" y1={y(v)} y2={y(v)} className="db-grid" /><text x="28" y={y(v) + 4} textAnchor="end">{v}</text></g>)}
      <line x1="34" x2="320" y1={y(1)} y2={y(1)} stroke="#e7b94a" strokeDasharray="5 4" />
      {sorted.map((value, k) => <g key={k}><circle cx={x(k + 1)} cy={y(value)} r="4.5" fill={value <= 1 ? '#e7b94a' : '#da9c86'} /><text x={x(k + 1)} y={y(value) + (value > 1 ? -9 : 18)} textAnchor="middle">{number(value)}</text><text x={x(k + 1)} y="216" textAnchor="middle">{k + 1}</text></g>)}
      <text x="180" y="232" textAnchor="middle">rank of c₄ · distance in meters (vertical)</text>
    </svg></div>
    <p className="db-legend">Dashed gold line: ε = 1 meter. Eight rows have c₄ ≤ 1 and are core.</p>
    <p>A row is core at radius ε exactly when its fourth-smallest distance, counting itself, is at most ε. Reading the sorted curve against the ε = 1 line gives the same eight core rows as counting neighbourhoods; the two views are one test seen from opposite directions. The steps at 0.5, 0.75 and 1.25 are individual rows with tied values, not a smooth elbow, and J's 2.75 stays on the same truthful axis.</p>
  </figure>;
}
/** F4: incompatible radius intervals, baseline and null. */
export function IncompatibleIntervalFigure() {
  const baseline = [...variedGroups.left, ...variedGroups.middle, ...variedGroups.right];
  const nullCase = [...variedGroups.left, ...variedGroups.middle, ...variedGroups.rightNull];
  const px = v => 24 + v * (300 / 7.5);
  const ax = v => 24 + v * (300 / 1);
  const rowStrip = (values, yy, fit) => values.map((v, i) => <g key={i}><Glyph x={px(v)} y={yy} type={fit.types[i]} /></g>);
  const fitBase = dbscan(baseline.map(v => [v, 0]), 0.25, 3), fitNull = dbscan(nullCase.map(v => [v, 0]), 0.25, 3);
  return <figure className="db-figure">
    <figcaption><strong>Incompatible radius intervals</strong> · m = 3; the two dense groups must stay apart while the right group must become viable</figcaption>
    <div className="db-scroll"><svg className="db-svg-mid" viewBox="0 0 340 250" role="img" aria-label="Top strip: twelve rows on a true distance axis from 0 to 7.25 at radius 0.25, right group all noise. Interval bars: dense groups separate for radius below 0.375 (open end); right group viable from radius 0.75 (closed start); no overlap. Bottom strip: the equal-spacing variant, all three groups core at radius 0.25.">
      <text x="8" y="14">1 · baseline at ε = 0.25</text>
      <line x1={px(0)} x2={px(7.25)} y1="40" y2="40" className="db-grid" />{[0, 1, 2, 3, 4, 5, 6, 7].map(v => <text key={v} x={px(v)} y="60" textAnchor="middle">{v}</text>)}
      {rowStrip(baseline, 40, fitBase)}
      <text x="8" y="92">2 · requirements on ε (0 to 1 shown)</text>
      <line x1={ax(0)} x2={ax(1)} y1="150" y2="150" className="db-grid" />{[0, 0.25, 0.5, 0.75, 1].map(v => <g key={v}><line x1={ax(v)} x2={ax(v)} y1="146" y2="154" className="db-grid" /><text x={ax(v)} y="168" textAnchor="middle">{v}</text></g>)}
      <line x1={ax(0)} x2={ax(0.375)} y1="112" y2="112" stroke="#e7b94a" strokeWidth="6" /><circle cx={ax(0.375)} cy="112" r="4" fill="#0b0f10" stroke="#e7b94a" strokeWidth="2" /><text x={ax(0.02)} y="104">dense groups separate: ε {'<'} 0.375</text>
      <line x1={ax(0.75)} x2={ax(1)} y1="134" y2="134" stroke="#8eb9a5" strokeWidth="6" /><circle cx={ax(0.75)} cy="134" r="4" fill="#8eb9a5" /><text x={ax(0.98)} y="128" textAnchor="end">right group viable: ε ≥ 0.75</text>
      <text x="8" y="196">3 · null variant at ε = 0.25</text>
      <line x1={px(0)} x2={px(7.25)} y1="222" y2="222" className="db-grid" />
      {rowStrip(nullCase, 222, fitNull)}
    </svg></div>
    <DenseStrip points={baseline.map(v => [v, 0])} fit={fitBase} names={baseline.map((_, i) => `${['L', 'M', 'R'][Math.floor(i / 4)]}${i % 4 + 1}`)} from={-0.125} to={1.5} caption="Magnified: the two dense groups of the baseline at ε = 0.25, from −0.125 to 1.5 meters" />
    <p className="db-legend">1 · baseline: left and middle are core groups, the right group is noise. 2 · dense groups separate while ε {'<'} 0.375 (open end); the right group is viable from ε ≥ 0.75 (closed start). 3 · null variant: the right group respaced to 5, 5.125, 5.25, 5.375 makes all three groups core at ε = 0.25.</p>
    <GlyphLegend />
    <p>The left and middle groups join through the exactly 0.375 gap between 0.375 and 0.75 as soon as ε reaches 0.375, because the neighbourhood boundary is closed. The right group's inner rows need three neighbours, which first happens at ε = 0.75. One requirement ends before the other begins, so no single radius recovers the three intended groups with m = 3. Respacing the right group to 0.125 steps makes ε = 0.25 recover all three: the conflict was density, not a badly chosen number.</p>
  </figure>;
}
/** F5: a high reachability can start a cluster. */
export function OpticsOrderingFigure() {
  const { ordering, coreDistances, orderedReachability } = trailOptics;
  const x = k => 40 + k * 30, y = v => 170 - 80 * v;
  return <figure className="db-figure">
    <figcaption><strong>High bars can start clusters</strong> · OPTICS ordering of the trail, m = 4, max_eps = 2, read at a cut of ε = 1</figcaption>
    <div className="db-scroll"><svg className="db-svg-mid" viewBox="0 0 340 200" role="img" aria-label={`Ordered rows ${ordering.map(i => names[i]).join(', ')} with reachability ${orderedReachability.map(value => number(value)).join(', ')} and core distances ${ordering.map(i => number(coreDistances[i])).join(', ')}. A and E start clusters at ε = 1; J cannot start because its core distance is undefined within max_eps 2.`}>
      {[0, 0.5, 1, 1.5].map(v => <g key={v}><line x1="34" x2="330" y1={y(v)} y2={y(v)} className="db-grid" /><text x="28" y={y(v) + 4} textAnchor="end">{v}</text></g>)}
      <line x1="34" x2="330" y1={y(1)} y2={y(1)} stroke="#e7b94a" strokeDasharray="5 4" /><text x="328" y={y(1) - 5} textAnchor="end" fill="#e7b94a">cut ε = 1</text>
      {ordering.map((rowIndex, k) => {
        const reach = orderedReachability[k], core = coreDistances[rowIndex];
        const starts = reach === null || reach > 1 ? core !== null && core <= 1 : false;
        return <g key={k}>
          {reach === null ? <text x={x(k)} y={y(1.5) - 6} textAnchor="middle" fill="#8a9590">restart</text> : <rect x={x(k) - 9} y={y(reach)} width="18" height={y(0) - y(reach)} fill={reach <= 1 ? '#91aecf' : '#da9c86'} />}
          {core !== null && <line x1={x(k) - 12} x2={x(k) + 12} y1={y(core)} y2={y(core)} stroke="#f2e7ca" strokeWidth="2" />}
          {starts && <text x={x(k)} y="20" textAnchor={k === 0 ? 'start' : 'middle'} fill="#8eb9a5">core start</text>}
          {rowIndex === 9 && <text x={x(k)} y="20" textAnchor="end" fill="#da9c86">no start</text>}
          <text x={x(k)} y="188" textAnchor="middle">{names[rowIndex]}</text>
        </g>;
      })}
    </svg></div>
    <p className="db-legend">Bars: reachability distance. Short white marks: core distance c₄. Rows are placed in OPTICS order; "restart" marks an undefined reachability.</p>
    <Table caption="The same ordering as a table" headings={['ordered row', 'reachability', 'core distance', 'role at ε = 1']} rows={ordering.map((rowIndex, k) => { const reach = orderedReachability[k], core = coreDistances[rowIndex]; const role = reach === null || reach > 1 ? core !== null && core <= 1 ? 'starts a cluster (core at 1)' : 'noise at this step' : 'joins the current cluster'; return [names[rowIndex], number(reach), number(core), role]; })} />
    <p>A cut at ε = 1 does not simply paint every bar above the line as noise. E's reachability is 1.25 because it is far from the rows processed before it, yet E's own core distance is 0.75, so E starts a new cluster. J's reachability is undefined and its core distance within max_eps = 2 is undefined too, so J starts nothing. The two low stretches B–D and F–H are the valleys of the two trail groups, and every bar keeps its row identity.</p>
  </figure>;
}
/** F6: stability is lifetime area with a selection constraint. */
export function StabilityTreeFigure() {
  const y = lambda => 180 - 25 * lambda;
  const scenario = exit => {
    const children = 2 * 3 * (exit - 3), parent = 6 * (3 - 1);
    return <div key={exit}>
      <p><strong>Children persist to λ = {exit}</strong></p>
      <svg data-stability-exit={exit} viewBox="0 0 220 194" role="img"
        aria-label={`Children persist to λ = ${exit}. The parent has six rows from λ = 1 to 3, with stability 12. Each child has three rows from λ = 3 to ${exit}, with combined stability ${children}. Select ${children > parent ? 'both children' : 'the parent'}. Both scenarios use the same row-count width and density-level height scales.`}>
        {[1, 3, 4, 6].map(lambda => <g key={lambda}>
          <line x1="45" x2="210" y1={y(lambda)} y2={y(lambda)} className="db-grid" />
          <text x="4" y={y(lambda) - 2}>λ = {lambda}</text>
        </g>)}
        <rect data-branch="parent" x="65" y={y(3)} width="120" height={y(1) - y(3)} fill="#e7b94a" fillOpacity={parent >= children ? 0.45 : 0.15} stroke="#e7b94a" />
        <text x="125" y={(y(1) + y(3)) / 2 + 4} textAnchor="middle">parent area 12</text>
        <rect data-branch="child" x="60" y={y(exit)} width="60" height={y(3) - y(exit)} fill="#8eb9a5" fillOpacity={children > parent ? 0.5 : 0.15} stroke="#8eb9a5" />
        <rect data-branch="child" x="130" y={y(exit)} width="60" height={y(3) - y(exit)} fill="#8eb9a5" fillOpacity={children > parent ? 0.5 : 0.15} stroke="#8eb9a5" />
        <text x="125" y={y(exit) - 6} textAnchor="middle">children area {children}</text>
        <text x="125" y={y(1) + 22} textAnchor="middle" fill={children > parent ? '#8eb9a5' : '#e7b94a'}>select {children > parent ? 'both children' : 'the parent'}</text>
      </svg>
      <p className="db-legend">Parent: 6 × (3 − 1) = 12. Children: 2 × 3 × ({exit} − 3) = {children}. Select {children > parent ? 'both children' : 'the parent'}.</p>
    </div>;
  };
  return <figure className="db-figure">
    <figcaption><strong>Stability is lifetime area with a selection constraint</strong> · an abstract condensed tree: six rows born at λ = 1 split into two children of three at λ = 3</figcaption>
    <div className="db-figure-pair">{scenario(6)}{scenario(4)}</div>
    <p>Width is retained row count, the vertical axis is λ = 1/ε, and a branch's stability is its area: rows × how long they persist. Selecting the parent and its children together is inadmissible because they contain the same rows; the excess-of-mass rule takes whichever disjoint choice has the larger total. At an exit of λ = 5 the two totals tie at 12 and a library must declare a tie rule. These lifetimes are declared for the illustration, not fitted to the trail or to Iris.</p>
  </figure>;
}
/** F7: rings versus a bisector. */
export function RingsFigure() {
  const { points, ring } = ringPoints();
  const fit = dbscan(points, 0.6, 3);
  const project = p => [140 + p[0] * 38, 130 - p[1] * 38];
  const [c0, c1] = ringKmeans.centers;
  const mid = [(c0[0] + c1[0]) / 2, (c0[1] + c1[1]) / 2], dir = [-(c1[1] - c0[1]), c1[0] - c0[0]];
  const norm = Math.hypot(...dir); const unit = [dir[0] / norm, dir[1] / norm];
  const bound = 3.4, ts = [];
  [0, 1].forEach(axis => { if (Math.abs(unit[axis]) < 1e-12) return; for (const edge of [-bound, bound]) { const t = (edge - mid[axis]) / unit[axis]; const other = mid[1 - axis] + t * unit[1 - axis]; if (Math.abs(other) <= bound + 1e-9) ts.push(t); } });
  const seg = [[mid[0] + Math.min(...ts) * unit[0], mid[1] + Math.min(...ts) * unit[1]], [mid[0] + Math.max(...ts) * unit[0], mid[1] + Math.max(...ts) * unit[1]]];
  const ringEdges = [];
  [0, 1].forEach(which => { const idx = ring.map((r, i) => r === which ? i : -1).filter(i => i >= 0); idx.forEach((i, k) => ringEdges.push([i, idx[(k + 1) % idx.length]])); });
  return <figure className="db-figure">
    <figcaption><strong>Two grouping questions on the same rings</strong> · 12 constructed rows at radius 1 and 36 at radius 3</figcaption>
    <div className="db-figure-pair">
      <div><p>DBSCAN, ε = 0.6, m = 3: every row core, two connected rings</p>
        <svg viewBox="0 0 280 245" role="img" aria-label="Two concentric rings of points. Each ring is one connected component through its neighbouring points; the rings are at least 2 apart and never connect.">
          {ringEdges.map(([i, j]) => { const [x1, y1] = project(points[i]), [x2, y2] = project(points[j]); return <line key={`${i}-${j}`} className="db-edge" x1={x1} y1={y1} x2={x2} y2={y2} style={{ stroke: fit.labels[i] === 0 ? '#e7b94a' : '#8eb9a5' }} />; })}
          {points.map((p, i) => { const [x, y] = project(p); return <circle key={i} cx={x} cy={y} r="3.5" fill={fit.labels[i] === 0 ? '#e7b94a' : '#8eb9a5'} />; })}
        </svg>
        <p className="db-legend">ring ARI 1.0 · drawn edges: each row's two ring neighbours</p></div>
      <div><p>K-Means, two centers, fixed seed: memberships split by the bisector</p>
        <svg viewBox="0 0 280 245" role="img" aria-label="The same 48 points colored by a two-center K-Means fit. The two centers sit on opposite sides of the origin and a straight bisector through the middle divides both rings.">
          {(() => { const [x1, y1] = project(seg[0]), [x2, y2] = project(seg[1]); return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#d9d3bf" strokeDasharray="4 3" />; })()}
          {points.map((p, i) => { const [x, y] = project(p); return <circle key={i} cx={x} cy={y} r="3.5" fill={ringKmeans.labels[i] === 0 ? '#91aecf' : '#da9c86'} />; })}
          {ringKmeans.centers.map((c, k) => { const [x, y] = project(c); return <g key={k} stroke={k === 0 ? '#91aecf' : '#da9c86'} strokeWidth="3"><path d={`M${x - 7},${y}h14 M${x},${y - 7}v14`} /><circle cx={x} cy={y} r="9" fill="none" strokeWidth="1" /></g>; })}
        </svg>
        <p className="db-legend">ring ARI {number(ringKmeans.ari, 3)} · dashed line: actual bisector</p></div>
    </div>
    <p>Adjacent rows on the inner ring are about 0.518 apart and on the outer ring about 0.523, so at ε = 0.6 each row counts itself and its two ring neighbours and the whole ring becomes one component; the rings are at least 2 apart and never join. Two Euclidean centers divide the plane by a straight line, which cannot put the whole inner ring on one side and every outer row on the other.</p>
  </figure>;
}
