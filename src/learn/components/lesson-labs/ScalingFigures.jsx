import {
  categoryDistances, circularCoordinates, crossFitEncoding, fitOneHot, fitPreparation, poolEstimates, rankCoordinates,
  scalerComparison, transformRecord,
} from '../../data/scaling-models';
import { comparison, featureNames, rows, split } from '../../data/scaling-data';
import { NumberLine, Table, fixed, fraction, round } from './ScalingShared.jsx';
import './scaling-labs.css';

const fixtureColumn = [1, 2, 3, 4, 100];
const fixtureIds = ['a', 'b', 'c', 'd', 'e'];
const rulerNames = { standard: 'Standard scaling', minmax: 'Min–max scaling', robust: 'Robust (median/IQR) scaling' };

/** Shared arrowheads. Direction is the whole subject of the boundary figure, so
 * a flow has to be able to say which way it runs. Defined once and referenced
 * document-wide by every figure that draws a flow. */
export function FlowDefs() {
  return <defs>
    {[['sc-arrow', '#8eb9a5'], ['sc-arrow-target', '#e7b94a'], ['sc-arrow-absent', '#b08a7c'], ['sc-arrow-forbidden', '#da9c86']]
      .map(([id, fill]) => (
        <marker key={id} id={id} viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill={fill} />
        </marker>
      ))}
  </defs>;
}

/** §1 · F1. One row, three preparation questions, and a target that never
 * enters the ordinary feature matrix. */
export function RecordFigure() {
  const cell = (x, width, name, value, kind = '') => <g key={name}>
    <text x={x + width / 2} y="20" textAnchor="middle" style={{ fontSize: 10.5 }}>{name}</text>
    <rect className={`sc-lane ${kind}`.trim()} x={x} y="28" width={width} height="30" rx="4" />
    <text x={x + width / 2} y="48" textAnchor="middle">{value}</text>
  </g>;
  const lane = (x, width, title, subtitle) => <g key={title}>
    <rect className="sc-lane" x={x} y="92" width={width} height="38" rx="4" />
    <text x={x + width / 2} y="108" textAnchor="middle" style={{ fontSize: 10.5 }}>{title}</text>
    <text x={x + width / 2} y="123" textAnchor="middle" style={{ fontSize: 10.5 }}>{subtitle}</text>
  </g>;
  return <figure className="sc-figure">
    <figcaption><strong>Figure 1 — One record, three meanings.</strong> Units, categories, and absence require different representation choices.</figcaption>
    <div className="sc-stage">
      <h4>The three feature cells of a complete row</h4>
      <svg viewBox="0 0 340 206" role="img" aria-label="One observed row's three feature cells: 40 millimetres, 4,000 grams and female. The two measurements enter a numerical preparation branch and become the named coordinates scaled_bill_length and scaled_body_mass. The recorded sex enters a categorical preparation branch and becomes the three indicator coordinates sex_female, sex_male and sex_not_recorded.">
        <FlowDefs />
        {cell(6, 104, 'bill length', '40 mm')}
        {cell(118, 104, 'body mass', '4,000 g')}
        {cell(230, 104, 'recorded sex', 'female')}
        <path className="sc-flow" d="M58,58 L104,88" markerEnd="url(#sc-arrow)" />
        <path className="sc-flow" d="M170,58 L128,88" markerEnd="url(#sc-arrow)" />
        <path className="sc-flow" d="M282,58 L282,88" markerEnd="url(#sc-arrow)" />
        {lane(6, 216, 'numerical preparation', 'scale the difference')}
        {lane(230, 104, 'categorical', 'one per category')}
        <path className="sc-flow" d="M114,130 L114,146" markerEnd="url(#sc-arrow)" />
        <path className="sc-flow" d="M282,130 L282,146" markerEnd="url(#sc-arrow)" />
        <rect className="sc-lane" x="6" y="150" width="216" height="50" rx="4" />
        <text x="114" y="170" textAnchor="middle" style={{ fontSize: 10.5 }}>scaled_bill_length</text>
        <text x="114" y="188" textAnchor="middle" style={{ fontSize: 10.5 }}>scaled_body_mass</text>
        <rect className="sc-lane" x="230" y="150" width="104" height="50" rx="4" />
        <text x="282" y="167" textAnchor="middle" style={{ fontSize: 10.5 }}>sex_female</text>
        <text x="282" y="181" textAnchor="middle" style={{ fontSize: 10.5 }}>sex_male</text>
        <text x="282" y="195" textAnchor="middle" style={{ fontSize: 10.5 }}>sex_not_recorded</text>
      </svg>
      <p className="sc-caption">
        The output names are the coordinates a model receives, not fitted numbers; the actual fitted statistics arrive in Figure 4b and
        Investigation 3.
      </p>
    </div>
    <div className="sc-stage">
      <h4>The target leaves by its own rail</h4>
      <svg viewBox="0 0 340 104" role="img" aria-label="The species cell, Adelie, travels a separate training-answer rail into model fitting. It never becomes one of the feature coordinates above.">
        <text x="62" y="20" textAnchor="middle" style={{ fontSize: 10.5 }}>species</text>
        <rect className="sc-lane is-target" x="6" y="28" width="112" height="30" rx="4" />
        <text x="62" y="48" textAnchor="middle">Adelie</text>
        <path className="sc-flow is-target" d="M118,43 L166,43" markerEnd="url(#sc-arrow-target)" />
        <rect className="sc-lane is-target" x="172" y="24" width="162" height="40" rx="4" />
        <text x="253" y="41" textAnchor="middle" style={{ fontSize: 10.5 }}>training answer,</text>
        <text x="253" y="56" textAnchor="middle" style={{ fontSize: 10.5 }}>into model fitting</text>
        <text x="6" y="92" style={{ fontSize: 10.5 }}>it never joins the feature coordinates</text>
      </svg>
      <p className="sc-caption">
        The target is the answer available in the training examples. Keeping its rail separate is what stops it from being read back as an input.
      </p>
    </div>
    <div className="sc-stage">
      <h4>The same row with one measurement absent</h4>
      <svg viewBox="0 0 340 174" role="img" aria-label="The same row with the body mass cell explicitly absent. Its branch inserts a labelled estimate, the training median, and passes an estimated value forward. The absent symbol is not turned into a measured number.">
        {cell(6, 104, 'bill length', '40 mm')}
        <text x="170" y="20" textAnchor="middle" style={{ fontSize: 10.5 }}>body mass</text>
        <rect className="sc-lane is-absent" x="118" y="28" width="104" height="30" rx="4" />
        <text x="170" y="48" textAnchor="middle">absent</text>
        {cell(230, 104, 'recorded sex', 'female')}
        <path className="sc-flow" d="M58,58 L58,86" markerEnd="url(#sc-arrow)" />
        <path className="sc-flow is-absent" d="M170,58 L170,86" markerEnd="url(#sc-arrow-absent)" />
        <path className="sc-flow" d="M282,58 L282,86" markerEnd="url(#sc-arrow)" />
        <rect className="sc-lane" x="6" y="90" width="104" height="38" rx="4" />
        <text x="58" y="106" textAnchor="middle" style={{ fontSize: 10.5 }}>measured</text>
        <text x="58" y="121" textAnchor="middle" style={{ fontSize: 10.5 }}>kept as is</text>
        <rect className="sc-lane is-absent" x="118" y="90" width="104" height="38" rx="4" />
        <text x="170" y="106" textAnchor="middle" style={{ fontSize: 10.5 }}>training median</text>
        <text x="170" y="121" textAnchor="middle" style={{ fontSize: 10.5 }}>inserted estimate</text>
        <rect className="sc-lane" x="230" y="90" width="104" height="38" rx="4" />
        <text x="282" y="106" textAnchor="middle" style={{ fontSize: 10.5 }}>female</text>
        <text x="282" y="121" textAnchor="middle" style={{ fontSize: 10.5 }}>own indicator</text>
        <path className="sc-flow" d="M58,128 L58,146" markerEnd="url(#sc-arrow)" />
        <path className="sc-flow" d="M170,128 L170,146" markerEnd="url(#sc-arrow)" />
        <path className="sc-flow" d="M282,128 L282,146" markerEnd="url(#sc-arrow)" />
        <rect className="sc-lane" x="6" y="150" width="328" height="22" rx="4" />
        <text x="170" y="165" textAnchor="middle" style={{ fontSize: 10.5 }}>the same named output coordinates</text>
      </svg>
      <p className="sc-caption">
        An imputer supplies a usable model input. It does not certify that an estimated value was measured, which is why the inserted cell keeps a
        different outline from the measured one.
      </p>
    </div>
  </figure>;
}

/** §2 · F2. The same five identified observations on three fitted rulers,
 * full range and magnified, with the later 150 marked separately. */
const rulerViews = {
  source: { domain: [-4, 158], ticks: [0, 25, 50, 75, 100, 125, 150], zoom: [0.4, 4.6], zoomTicks: [1, 2, 3, 4], zoomFrom: '1 to 4', decimals: 0 },
  standard: { domain: [-1, 3.6], ticks: [-1, 0, 1, 2, 3], zoom: [-0.56, -0.44], zoomTicks: [-0.55, -0.5, -0.45], zoomFrom: '−0.5383 to −0.4614', decimals: 2 },
  minmax: { domain: [-0.12, 1.62], ticks: [0, 0.5, 1, 1.5], zoom: [-0.006, 0.037], zoomTicks: [0, 0.01, 0.02, 0.03], zoomFrom: '0 to 0.0303', decimals: 2 },
  robust: { domain: [-6, 80], ticks: [0, 20, 40, 60, 80], zoom: [-1.3, 0.8], zoomTicks: [-1, -0.5, 0, 0.5], zoomFrom: '−1 to 0.5', decimals: 0 },
};
export function RulerFigure() {
  const comparisonRows = scalerComparison(fixtureColumn, [150]);
  const coordinate = (kind, row) => (kind === 'source' ? row.value : row[kind]);
  const line = kind => {
    const view = rulerViews[kind];
    const points = comparisonRows.rows.map((row, index) => ({
      id: index === 5 ? 'new' : fixtureIds[index], at: coordinate(kind, row), later: !row.training,
    }));
    const inside = (points_, [low, high]) => points_.filter(point => point.at >= low && point.at <= high);
    return <div className="sc-panel" key={kind}>
      <h4>{kind === 'source' ? 'Source values' : rulerNames[kind]}</h4>
      <p>{kind === 'source'
        ? 'The training column 1, 2, 3, 4, 100, with the later observation 150 as a hollow marker.'
        : `Fitted ${comparisonRows.fits[kind].centerName} ${round(comparisonRows.fits[kind].center, 4)}, fitted ${comparisonRows.fits[kind].scaleName} ${round(comparisonRows.fits[kind].scale, 4)}${kind === 'minmax' ? `, fitted maximum ${round(comparisonRows.fits[kind].high, 4)}` : ''}.`}</p>
      <NumberLine title="Full range" domain={view.domain} ticks={view.ticks} decimals={view.decimals}
        points={inside(points, view.domain)} highlight={view.zoom}
        describe={`${kind === 'source' ? 'Source' : rulerNames[kind]} full range from ${view.domain[0]} to ${view.domain[1]}. ${points.map(point => `${point.id} at ${round(point.at, 4)}`).join(', ')}. A bracket marks the interval ${view.zoomFrom}, which the magnified line below expands.`} />
      <NumberLine title={`Magnified: the four small values, ${view.zoomFrom}`} domain={view.zoom} ticks={view.zoomTicks} decimals={4}
        points={inside(points, view.zoom)}
        describe={`The same ruler magnified over ${view.zoomFrom}, containing ${inside(points, view.zoom).map(point => `${point.id} at ${round(point.at, 4)}`).join(', ')}.`} />
    </div>;
  };
  return <figure className="sc-figure">
    <figcaption><strong>Figure 2 — Five observations on three rulers.</strong> Each row keeps its own actual coordinates and its own labelled ticks; the four small values are magnified beneath, not stretched to a common width.</figcaption>
    <div className="sc-panels">{['source', 'standard', 'minmax', 'robust'].map(line)}</div>
    <Table caption="The same observations, transformed by each fitted ruler. Observations a to e are the training column; 150 is a later value."
      headings={['observation', 'value', 'standard', 'min–max', 'robust']}
      rows={comparisonRows.rows.map((row, index) => [
        index === 5 ? 'new value' : fixtureIds[index], String(row.value),
        fixed(row.standard, 4), fixed(row.minmax, 4), fixed(row.robust, 4),
      ])} />
    <p>
      Standard scaling divides by the fitted standard deviation {round(comparisonRows.fits.standard.scale, 4)}, which the single observation 100
      inflates. Min–max maps the fitted minimum and maximum to 0 and 1, so 1, 2, 3 and 4 crowd into the first 3% of the interval and the later 150
      lands at {fixed(comparisonRows.rows[5].minmax, 4)}, outside it. Robust scaling keeps 1, 2, 3 and 4 half a unit apart and still places 100 at
      48.5: it does not remove the observation, it changes what one unit means.
    </p>
  </figure>;
}

/** §3 · F3. The chosen geometry of a one-hot block, and what dropping a
 * reference column does to it. */
export function CategoryFigure() {
  const colours = fitOneHot(['red', 'green', 'blue']);
  const order = ['red', 'green', 'blue'];
  const full = categoryDistances(colours, { includeUnknown: false });
  const droppedRed = categoryDistances(colours, { drop: 'red' });
  const pairText = (set, from, to) => {
    const pair = set.pairs.find(item => (item.from === from && item.to === to) || (item.from === to && item.to === from));
    return round(pair.distance, 4);
  };
  const place = value => 40 + value * 120;
  const lift = value => 200 - value * 120;
  return <figure className="sc-figure">
    <figcaption><strong>Figure 3 — A category simplex and a reference corner.</strong> Equal separation between different categories is a chosen geometry, and dropping a column changes it.</figcaption>
    <div className="sc-panels">
      <div className="sc-panel">
        <h4>All three columns kept</h4>
        <svg viewBox="0 0 260 190" role="img" aria-label="A schematic triangle whose three vertices are labelled red, green and blue. Every edge is labelled square root of 2, because any two different rows of the full one-hot table are that far apart. This is a schematic of three points in a three-coordinate representation, not a plot of their coordinates.">
          <polygon className="sc-ring" points="130,32 34,158 226,158" />
          <circle className="sc-mark-neutral" cx="130" cy="32" r="6" />
          <circle className="sc-mark-neutral" cx="34" cy="158" r="6" />
          <circle className="sc-mark-neutral" cx="226" cy="158" r="6" />
          <text x="130" y="22" textAnchor="middle">red</text>
          <text x="24" y="176" textAnchor="start">green</text>
          <text x="236" y="176" textAnchor="end">blue</text>
          <text x="62" y="96" textAnchor="middle">√2</text>
          <text x="200" y="96" textAnchor="middle">√2</text>
          <text x="130" y="151" textAnchor="middle">√2</text>
        </svg>
        <p>A schematic of three points in a three-coordinate representation, not a plot of those coordinates. Every pair is exactly {pairText(full, 'red', 'green')} apart.</p>
      </div>
      <div className="sc-panel">
        <h4>Red dropped: a true coordinate plane</h4>
        <svg viewBox="0 0 260 258" role="img" aria-label="An equal-aspect coordinate plane with a green axis across and a blue axis up. Red sits at the origin 0,0; green at 1,0; blue at 0,1. Red to green is 1, red to blue is 1, and green to blue is the square root of 2. An unknown input also encodes as 0,0 and therefore coincides with the reference corner in this representation.">
          <line className="sc-axis" x1="40" x2="236" y1="200" y2="200" />
          <line className="sc-axis" x1="40" x2="40" y1="44" y2="200" />
          <line className="sc-ring" x1={place(0)} x2={place(1)} y1={lift(0)} y2={lift(0)} />
          <line className="sc-ring" x1={place(0)} x2={place(0)} y1={lift(0)} y2={lift(1)} />
          <line className="sc-ring" x1={place(1)} x2={place(0)} y1={lift(0)} y2={lift(1)} />
          <rect className="sc-mark-reference" x={place(0) - 6} y={lift(0) - 6} width="12" height="12" />
          <circle className="sc-mark-neutral" cx={place(1)} cy={lift(0)} r="6" />
          <circle className="sc-mark-neutral" cx={place(0)} cy={lift(1)} r="6" />
          <text x={place(0) + 10} y={lift(0) + 20} style={{ fontSize: 10.5 }}>red (0,0)</text>
          <text x={place(1) + 9} y={lift(0) - 6} style={{ fontSize: 10.5 }}>green (1,0)</text>
          <text x={place(0) + 10} y={lift(1) - 8} style={{ fontSize: 10.5 }}>blue (0,1)</text>
          <text x={place(0.5)} y={lift(0) - 8} textAnchor="middle" style={{ fontSize: 10.5 }}>1</text>
          <text x={place(0) - 8} y={lift(0.5)} textAnchor="end" style={{ fontSize: 10.5 }}>1</text>
          <text x={place(0.62) + 6} y={lift(0.52)} style={{ fontSize: 10.5 }}>√2</text>
          <text x="138" y="238" textAnchor="middle" style={{ fontSize: 10.5 }}>horizontal axis: green column</text>
          <text x="138" y="250" textAnchor="middle" style={{ fontSize: 10.5 }}>vertical axis: blue column</text>
        </svg>
        <p>
          The hollow square is the dropped reference corner; the two discs are the kept categories. Markers are deliberately neutral, because in this
          example the categories are themselves colours. An unknown input under <code>handle_unknown=&quot;ignore&quot;</code> is also (0, 0), so in
          this representation it coincides with the reference category red.
        </p>
      </div>
    </div>
    <Table caption="The full one-hot table. Any two different rows are √2 apart."
      headings={['category', 'red', 'green', 'blue']}
      rows={order.map(name => {
        const point = full.points.find(item => item.name === name);
        const byName = Object.fromEntries(colours.categories.map((category, index) => [category, point.vector[index]]));
        return [name, String(byName.red), String(byName.green), String(byName.blue)];
      })} />
    <Table caption="Every pairwise distance, with and without the dropped reference column"
      headings={['pair', 'all three columns', 'red dropped']}
      rows={[
        ['red to green', pairText(full, 'red', 'green'), pairText(droppedRed, 'red', 'green')],
        ['red to blue', pairText(full, 'red', 'blue'), pairText(droppedRed, 'red', 'blue')],
        ['green to blue', pairText(full, 'green', 'blue'), pairText(droppedRed, 'green', 'blue')],
        ['unknown to green', '1', pairText(droppedRed, 'unknown input', 'green')],
      ]} />
  </figure>;
}

/** §§5–6 · F4. The fitting boundary, then the actual fitted pipeline. */
const numericOf = row => [row[2], row[3], row[4], row[5]];
const sexOf = row => (row[6] === null ? null : ['female', 'male'][row[6]]);
export const trainingRows = rows.filter(row => row[7] === 0);
export const preparation = fitPreparation({
  numericRows: trainingRows.map(numericOf),
  categoryValues: trainingRows.map(sexOf),
  scaler: 'standard',
});
const sourceRow = rows[309];
const sourceResult = transformRecord(preparation, { numeric: numericOf(sourceRow), category: sexOf(sourceRow) });

export function BoundaryFigure() {
  return <figure className="sc-figure">
    <figcaption><strong>Figure 4 — A fitted object crosses the boundary; observations do not cross backward.</strong> One bundle is learned from the training rows and then applied, unchanged, to both sides.</figcaption>
    <svg viewBox="0 0 340 266" role="img" aria-label={`The ${split.training} training rows, which carry both features and answers, are the only rows used to fit the saved bundle of medians, centres, scales and the category vocabulary. That one frozen bundle is then applied to the training rows and to the ${split.heldOut} held-out rows alike, producing two transformed blocks. The held-out answers leave the held-out box by their own rail down the right-hand side and join only the final comparison. A separate crossed arrow points backward from the held-out rows into the training rows, the direction this diagram forbids.`}>
      <FlowDefs />
      <rect className="sc-lane" x="6" y="10" width="142" height="38" rx="4" />
      <text x="77" y="26" textAnchor="middle" style={{ fontSize: 10.5 }}>{split.training} training rows</text>
      <text x="77" y="40" textAnchor="middle" style={{ fontSize: 10.5 }}>features and answers</text>
      <rect className="sc-lane is-absent" x="192" y="10" width="142" height="38" rx="4" />
      <text x="263" y="26" textAnchor="middle" style={{ fontSize: 10.5 }}>{split.heldOut} held-out rows</text>
      <text x="263" y="40" textAnchor="middle" style={{ fontSize: 10.5 }}>features only, for now</text>
      {/* The forbidden direction: an arrowhead, so "backward" is drawn and not
          only written. Its label sits directly under the crossed marker and
          clear of every permitted line. */}
      <path className="sc-flow is-forbidden" d="M192,29 L154,29" strokeDasharray="5 3" markerEnd="url(#sc-arrow-forbidden)" />
      <line className="sc-flow is-forbidden" x1="166" y1="21" x2="180" y2="37" />
      <line className="sc-flow is-forbidden" x1="180" y1="21" x2="166" y2="37" />
      <text x="170" y="62" textAnchor="middle" style={{ fontSize: 10.5 }}>fitting on all rows</text>
      <path className="sc-flow" d="M77,48 L77,76" markerEnd="url(#sc-arrow)" />
      <text x="84" y="76" style={{ fontSize: 10.5 }}>fit</text>
      <rect className="sc-lane is-target" x="6" y="80" width="142" height="48" rx="4" />
      <text x="77" y="96" textAnchor="middle" style={{ fontSize: 10.5 }}>saved bundle</text>
      <text x="77" y="109" textAnchor="middle" style={{ fontSize: 10.5 }}>medians · centres · scales</text>
      <text x="77" y="122" textAnchor="middle" style={{ fontSize: 10.5 }}>category vocabulary</text>
      <path className="sc-flow" d="M77,128 L77,150" />
      <path className="sc-flow" d="M263,48 L263,150" />
      <line className="sc-flow" x1="77" y1="150" x2="263" y2="150" />
      <text x="110" y="144" textAnchor="middle" style={{ fontSize: 10.5 }}>apply, frozen</text>
      <path className="sc-flow" d="M77,150 L77,170" markerEnd="url(#sc-arrow)" />
      <path className="sc-flow" d="M263,150 L263,170" markerEnd="url(#sc-arrow)" />
      <rect className="sc-lane" x="6" y="170" width="142" height="30" rx="4" />
      <text x="77" y="189" textAnchor="middle" style={{ fontSize: 10.5 }}>transformed training</text>
      <rect className="sc-lane" x="192" y="170" width="126" height="30" rx="4" />
      <text x="255" y="189" textAnchor="middle" style={{ fontSize: 10.5 }}>transformed held out</text>
      <path className="sc-flow" d="M255,200 L255,220" markerEnd="url(#sc-arrow)" />
      {/* The answers rail leaves the held-out box itself, down the right margin,
          past the transformed block and into the comparison. */}
      <path className="sc-flow is-target" d="M326,48 L326,220" markerEnd="url(#sc-arrow-target)" />
      <text x="334" y="120" textAnchor="end" style={{ fontSize: 10.5 }}>answers</text>
      <rect className="sc-lane is-target" x="150" y="220" width="184" height="38" rx="4" />
      <text x="242" y="237" textAnchor="middle" style={{ fontSize: 10.5 }}>final comparison, here only:</text>
      <text x="242" y="251" textAnchor="middle" style={{ fontSize: 10.5 }}>against the withheld answers</text>
    </svg>
    <p>
      A pipeline packages this sequence so software repeats it consistently. It does not repair a feature that already leaks the answer, a split that
      puts repeated subjects on both sides, or a transformer fitted by hand before the split.
    </p>
  </figure>;
}

export function PipelineFigure() {
  const units = ['mm', 'mm', 'mm', 'g'];
  return <figure className="sc-figure">
    <figcaption><strong>Figure 4b — The actual fitted bundle, and one held-out record through it.</strong> These are the statistics the standard-scaling model fitted on the {split.training} training rows.</figcaption>
    <svg viewBox="0 0 340 140" role="img" aria-label="The fitted preparation forks into a numeric branch, which imputes a training median and then subtracts a centre and divides by a scale, and a categorical branch, which replaces an absent value by not_recorded and then activates one indicator coordinate. The two branches rejoin into the seven named output coordinates.">
      <rect className="sc-lane" x="90" y="8" width="160" height="24" rx="4" />
      <text x="170" y="24" textAnchor="middle" style={{ fontSize: 10.5 }}>one record, five input cells</text>
      <path className="sc-flow" d="M130,32 L74,54" />
      <path className="sc-flow" d="M210,32 L250,54" />
      <rect className="sc-lane" x="6" y="54" width="136" height="48" rx="4" />
      <text x="74" y="68" textAnchor="middle" style={{ fontSize: 10.5 }}>numeric branch</text>
      <text x="74" y="80" textAnchor="middle" style={{ fontSize: 10.5 }}>absent → training median</text>
      <text x="74" y="92" textAnchor="middle" style={{ fontSize: 10.5 }}>(value − centre) ÷ scale</text>
      <rect className="sc-lane" x="188" y="54" width="146" height="48" rx="4" />
      <text x="261" y="68" textAnchor="middle" style={{ fontSize: 10.5 }}>categorical branch</text>
      <text x="261" y="80" textAnchor="middle" style={{ fontSize: 10.5 }}>absent → not_recorded</text>
      <text x="261" y="92" textAnchor="middle" style={{ fontSize: 10.5 }}>activate one indicator</text>
      <path className="sc-flow" d="M74,102 L120,122" />
      <path className="sc-flow" d="M261,102 L220,122" />
      <rect className="sc-lane is-target" x="90" y="122" width="160" height="16" rx="4" />
      <text x="170" y="134" textAnchor="middle" style={{ fontSize: 10.5 }}>seven named coordinates</text>
    </svg>
    <Table caption={`The numeric branch, fitted on the ${split.training} training rows and then frozen`}
      headings={['column', 'training median', 'fitted centre (mean)', 'fitted scale (sd)']}
      rows={preparation.imputer.medians.map((value, index) => [
        `${['bill length', 'bill depth', 'flipper length', 'body mass'][index]} (${units[index]})`,
        round(value, 4), round(preparation.scalers[index].center, 10), round(preparation.scalers[index].scale, 10),
      ])} />
    <p className="sc-caption">
      The categorical branch fills an absent value with <code>not_recorded</code> and then encodes the fitted vocabulary{' '}
      {preparation.vocabulary.categories.join(', ')} in that order, giving the last three coordinates.
    </p>
    <Table caption={`Source row ${sourceRow[0]}, the first held-out record, through that frozen bundle`}
      headings={['output coordinate', 'input', 'calculation', 'value']}
      rows={[
        ...preparation.scalers.map((ruler, index) => [
          featureNames[index],
          `${round(numericOf(sourceRow)[index], 4)} ${units[index]}`,
          `(${round(numericOf(sourceRow)[index], 4)} − ${round(ruler.center, 6)}) ÷ ${round(ruler.scale, 6)}`,
          fixed(sourceResult.coordinates[index], 10),
        ]),
        ...preparation.vocabulary.categories.map((name, index) => [
          featureNames[4 + index], sexOf(sourceRow) ?? 'absent',
          `is the record ${name}?`, String(sourceResult.coordinates[4 + index]),
        ]),
      ]} />
    <p>
      The negative mass coordinate says this mass sits below the fitted training mean. It does not mean a negative mass. The row number is a position
      in the file, not one of the five inputs; species, island and year were not included as features.
    </p>
  </figure>;
}

/** §6 · the controlled comparison, on a zero-origin count axis. */
export function ComparisonFigure() {
  const width = 340;
  const left = 150;
  const scale = value => left + ((width - left - 30) * value) / 86;
  return <figure className="sc-figure">
    <figcaption><strong>One split, one neighbour count, one categorical preparation; only the numerical ruler changes.</strong></figcaption>
    <svg viewBox={`0 0 ${width} 158`} role="img" aria-label={`Correct held-out predictions out of 86, on a zero-origin axis: ${comparison.map(row => `${row.label} ${row.correct}`).join('; ')}.`}>
      {[0, 43, 86].map(tick => <g key={tick}>
        <line className="sc-grid" x1={scale(tick)} x2={scale(tick)} y1="14" y2="122" />
        <text x={scale(tick)} y="136" textAnchor="middle" style={{ fontSize: 10.5 }}>{tick}</text>
      </g>)}
      <text x={(scale(0) + scale(86)) / 2} y="150" textAnchor="middle" style={{ fontSize: 10.5 }}>correct out of 86</text>
      {comparison.map((row, index) => <g key={row.method}>
        <text x={left - 6} y={26 + index * 22} textAnchor="end" style={{ fontSize: 10.5 }}>{row.method}</text>
        <rect className={`sc-bar${row.method === 'majority' ? ' is-baseline' : ''}`}
          x={scale(0)} y={18 + index * 22} width={scale(row.correct) - scale(0)} height="12" />
        <text x={scale(row.correct) + 4} y={28 + index * 22} style={{ fontSize: 10.5 }}>{row.correct}</text>
      </g>)}
      <line className="sc-axis" x1={scale(0)} x2={scale(86)} y1="122" y2="122" />
    </svg>
    <Table caption="The recorded results. Accuracy here means correct species predictions divided by 86."
      headings={['preparation for numeric features', 'correct / held-out rows', 'accuracy']}
      rows={comparison.map(row => [row.label, `${row.correct} / 86`, row.accuracy.toFixed(4)])} />
    <p>
      The majority baseline is drawn in a different colour because it is not a preparation at all: it ignores every measurement and always answers
      Adelie. The axis starts at zero, so the one-row gap between standard scaling and the other two looks the size it is. One extra correct row does not
      establish that min–max or robust scaling is generally superior to standard scaling; we have now inspected this held-out set across several
      alternatives, and selecting a procedure from these results would need a separate evaluation plan.
    </p>
  </figure>;
}

/** §7 · F5. What a rank map keeps and what a log map keeps. */
export function RankFigure() {
  const ranks = rankCoordinates(fixtureColumn);
  const changed = rankCoordinates([1, 2, 3, 4, 1000]);
  // Data lives between x = 60 and x = 300, leaving a clear gutter at each end
  // for the axis end values, so no connector can cross a label.
  const span = (value, high) => 60 + (240 * value) / high;
  const source = value => span(value, 105);
  const rank = value => span(value, 1);
  const logPlace = value => span(value, Math.log(105));
  const rows = [
    { y: 34, label: 'source value', low: '0', high: '105', at: source, key: 'value' },
    { y: 104, label: 'rank coordinate (r − 1)/(n − 1)', low: '0', high: '1', at: rank, key: 'coordinate' },
    { y: 174, label: 'natural logarithm', low: '0', high: '4.65', at: logPlace, key: 'log' },
  ];
  const angles = [1, 359].map(circularCoordinates);
  return <figure className="sc-figure">
    <figcaption><strong>Figure 5 — Distance versus rank.</strong> The same five values, linked to their exact rank coordinates and to their natural logarithms.</figcaption>
    <ol className="sc-axis-key">
      {rows.map(row => <li key={row.key}>{row.label}, from {row.low} to {row.high}</li>)}
    </ol>
    <div className="sc-wide">
      <svg viewBox="0 0 340 196" role="img" aria-label={`Three stacked numeric axes. The top axis carries the source values ${ranks.map(row => row.value).join(', ')}. Each value links down to its rank coordinate ${ranks.map(row => round(row.coordinate, 2)).join(', ')} on the middle axis, and on to its natural logarithm ${ranks.map(row => round(row.log, 4)).join(', ')} on the bottom axis. The value 100 stays identified on all three.`}>
        {rows.map(row => <g key={row.key}>
          <line className="sc-axis" x1="56" x2="304" y1={row.y} y2={row.y} />
          <text x="52" y={row.y + 4} textAnchor="end" style={{ fontSize: 10.5 }}>{row.low}</text>
          <text x="308" y={row.y + 4} style={{ fontSize: 10.5 }}>{row.high}</text>
          {ranks.map(point => <circle key={point.value} className={row.key === 'value' ? 'sc-point' : (row.key === 'coordinate' ? 'sc-mark-a' : 'sc-mark-b')}
            cx={row.at(point[row.key])} cy={row.y} r="4" />)}
        </g>)}
        {ranks.map(point => <g key={point.value}>
          <line className="sc-ring" x1={source(point.value)} x2={rank(point.coordinate)} y1="40" y2="98" />
          <line className="sc-ring" x1={rank(point.coordinate)} x2={logPlace(point.log)} y1="110" y2="168" />
        </g>)}
        <text className="sc-point-id" x={source(100)} y="22" textAnchor="middle" style={{ fontSize: 10.5 }}>100</text>
        <text className="sc-point-id" x={source(2.5)} y="22" textAnchor="middle" style={{ fontSize: 10.5 }}>1–4</text>
        <text x="56" y="192" style={{ fontSize: 10.5 }}>top: source · middle: rank · bottom: log</text>
      </svg>
    </div>
    <Table caption="Exact coordinates under the two maps, and what changes when 100 becomes 1,000"
      headings={['value', 'rank coordinate', 'natural log', 'value if 100 → 1,000', 'its rank', 'its log']}
      rows={ranks.map((row, index) => [
        String(row.value), fixed(row.coordinate, 2), fixed(row.log, 6),
        String(changed[index].value), fixed(changed[index].coordinate, 2), fixed(changed[index].log, 6),
      ])} />
    <p>
      Replacing 100 by 1,000 is a <strong>null for the rank</strong>: it is still the largest, so its coordinate is still 1. It is a{' '}
      <strong>contrast for the log</strong>: {fixed(ranks[4].log, 6)} becomes {fixed(changed[4].log, 6)}. The two representations preserve different
      information, and a rank map that produces a tidy histogram has discarded the magnitude that told them apart. This is the explicit rank
      convention, not a claim to reproduce every interpolation, tie and endpoint rule of a fitted quantile transformer.
    </p>
    <div className="sc-panels">
      <div className="sc-panel">
        <h4>A circular feature keeps 359° near 1°</h4>
        <svg viewBox="0 0 260 200" role="img" aria-label={`A labelled unit circle. One vector at 1 degree has cosine ${round(angles[0].cos, 6)} and sine ${round(angles[0].sin, 6)}; another at 359 degrees has cosine ${round(angles[1].cos, 6)} and sine ${round(angles[1].sin, 6)}. The two points nearly coincide on the circle although 1 and 359 are far apart as plain numbers.`}>
          <circle className="sc-ring" cx="94" cy="100" r="66" />
          <line className="sc-axis" x1="18" x2="170" y1="100" y2="100" />
          <line className="sc-axis" x1="94" x2="94" y1="24" y2="176" />
          {angles.map((angle, index) => <g key={angle.degrees}>
            <line className={`sc-vector ${index === 0 ? 'is-a' : 'is-b'}`} x1="94" y1="100" x2={94 + 66 * angle.cos} y2={100 - 66 * angle.sin} />
            <circle className={index === 0 ? 'sc-mark-a' : 'sc-mark-b'} cx={94 + 66 * angle.cos} cy={100 - 66 * angle.sin} r="4" />
          </g>)}
          <text x="176" y="88" style={{ fontSize: 10.5 }}>1°</text>
          <text x="176" y="120" style={{ fontSize: 10.5 }}>359°</text>
          <text x="18" y="192" style={{ fontSize: 10.5 }}>(cos θ, sin θ) with θ in radians</text>
        </svg>
        <p>
          1° is {round(angles[0].radians, 6)} radians and 359° is {round(angles[1].radians, 6)} radians; NumPy&apos;s <code>sin</code> and{' '}
          <code>cos</code> expect radians, so the conversion is part of the feature.
        </p>
      </div>
      <div className="sc-panel">
        <h4>The two coordinates, exactly</h4>
        <Table caption="Sine and cosine at the two angles"
          headings={['angle', 'radians', 'cos', 'sin']}
          rows={angles.map(angle => [`${angle.degrees}°`, fixed(angle.radians, 6), fixed(angle.cos, 6), fixed(angle.sin, 6)])} />
        <p>
          The two points are {round(Math.hypot(angles[0].cos - angles[1].cos, angles[0].sin - angles[1].sin), 6)} apart on the circle, while the raw
          numbers 1 and 359 differ by 358. This is right for direction or time of day and wrong for elapsed time, where completing a cycle does not
          erase duration.
        </p>
      </div>
    </div>
  </figure>;
}

/** §8 · the donor graph that the target-encoding investigation makes editable. */
export function EncodingFigure() {
  const encoding = crossFitEncoding({
    categories: ['A', 'A', 'B', 'B', 'C', 'C'], target: [1, 1, 0, 0, 1, 0], folds: [0, 1, 0, 1, 0, 1], smoothing: 2,
  });
  return <figure className="sc-figure">
    <figcaption><strong>Which target can reach which encoded row.</strong> Each internal fold takes its sums, its counts and its prior from the other fold only.</figcaption>
    <Table caption="The exact six-row example with smoothing 2"
      headings={['row', 'category', 'y', 'fold', 'donor rows', 'sum / count', 'prior', 'encoded']}
      rows={encoding.rows.map(row => [
        String(row.row), row.category, String(row.target), String(row.fold),
        row.donors.join(', '), `${row.sum} / ${row.count}`, fraction(row.prior), fraction(row.value),
      ])} />
    <p>
      For fold 0, donor rows 1, 3 and 5 have targets 1, 0 and 0, so the prior is {fraction(encoding.priors[0])}. Row 0&apos;s A encoding is
      (1 + 2 × {fraction(encoding.priors[0])}) ÷ (1 + 2) = {fraction(encoding.rows[0].value)}. For fold 1 the donor targets are 1, 0 and 1, giving
      prior {fraction(encoding.priors[1])}, so row 1 receives {fraction(encoding.rows[1].value)}. No row appears in its own donor list, which is what
      keeps its own target out of its own encoded value even through the smoothing term.
    </p>
  </figure>;
}

/** §9 · F6. Several plausible completions, several analyses, one pooled result. */
export function PoolingFigure() {
  const pooled = poolEstimates([9, 10, 11], [4, 4, 4]);
  return <figure className="sc-figure">
    <figcaption><strong>Figure 6 — Several plausible tables, several estimates, one pooled analysis.</strong> The completions are symbolic: no cell below was measured.</figcaption>
    <svg viewBox="0 0 340 336" role="img" aria-label="An incomplete table branches into three symbolic plausible completions. Each completion passes through its own analysis box, producing estimates 9, 10 and 11, each with within-analysis variance 4. Those three estimates combine into a pooled estimate of 10 with total variance 16 over 3. A separate crossed arrow at the bottom shows averaging the completed tables first and then analysing once, which is the wrong order.">
      <FlowDefs />
      <rect className="sc-lane is-absent" x="6" y="84" width="80" height="44" rx="4" />
      <text x="46" y="102" textAnchor="middle" style={{ fontSize: 10.5 }}>incomplete</text>
      <text x="46" y="118" textAnchor="middle" style={{ fontSize: 10.5 }}>table</text>
      {[0, 1, 2].map(index => {
        const top = 10 + index * 76;
        return <g key={index}>
          <path className="sc-flow" d={`M86,106 L96,${top + 20}`} markerEnd="url(#sc-arrow)" />
          <rect className="sc-lane" x="96" y={top} width="96" height="40" rx="4" />
          <text x="144" y={top + 17} textAnchor="middle" style={{ fontSize: 10.5 }}>completion {index + 1}</text>
          <text x="144" y={top + 33} textAnchor="middle" style={{ fontSize: 10.5 }}>symbolic</text>
          <path className="sc-flow" d={`M192,${top + 20} L202,${top + 20}`} markerEnd="url(#sc-arrow)" />
          <rect className="sc-lane is-target" x="202" y={top} width="132" height="40" rx="4" />
          <text x="268" y={top + 17} textAnchor="middle" style={{ fontSize: 10.5 }}>analysis</text>
          <text x="268" y={top + 33} textAnchor="middle" style={{ fontSize: 10.5 }}>estimate {9 + index}, U = 4</text>
          <path className="sc-flow" d={`M268,${top + 40} L200,226`} markerEnd="url(#sc-arrow)" />
        </g>;
      })}
      <rect className="sc-lane is-target" x="100" y="226" width="140" height="44" rx="4" />
      <text x="170" y="244" textAnchor="middle" style={{ fontSize: 10.5 }}>pooled: mean 10</text>
      <text x="170" y="260" textAnchor="middle" style={{ fontSize: 10.5 }}>T = 16/3</text>
      <path className="sc-flow is-forbidden" d="M96,292 L300,292" strokeDasharray="5 3" markerEnd="url(#sc-arrow-forbidden)" />
      <line className="sc-flow is-forbidden" x1="190" y1="284" x2="204" y2="300" />
      <line className="sc-flow is-forbidden" x1="204" y1="284" x2="190" y2="300" />
      <text x="6" y="318" style={{ fontSize: 10.5 }}>the wrong order: average the completed</text>
      <text x="6" y="332" style={{ fontSize: 10.5 }}>tables, then analyse once</text>
    </svg>
    <div className="sc-bars" role="img" aria-label={`Variance contributions in estimate units squared, from a zero origin: within-analysis variance ${pooled.within}, the between-completion contribution ${fraction(pooled.correction)}, total ${fraction(pooled.total)}.`}>
      {[['within-analysis Ū', pooled.within], ['between, with the finite-m adjustment', pooled.correction], ['total T', pooled.total]].map(([label, value]) => (
        <div className="sc-bar-row" key={label}>
          <span>{label}</span>
          <span className="sc-bar-track"><span className="sc-bar-fill" style={{ width: `${(value / pooled.total) * 100}%` }} /></span>
          <span className="sc-bar-value">{fraction(value)}</span>
        </div>
      ))}
    </div>
    <p>
      The bars start at zero and their units are <strong>estimate units squared</strong>. The standard error is a separate value:{' '}
      √({fraction(pooled.total)}) ≈ {round(pooled.standardError, 4)}, against 2 from treating the completion as certain. Validity still depends on
      the imputation model, the analysis and the appropriate degrees-of-freedom calculation; this arithmetic is not a certificate for any of them.
    </p>
  </figure>;
}
