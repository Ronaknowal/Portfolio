import {
  boundDecomposition, collapseLogLikelihood, emCycle, expectation, mixtureAt, normal, presets, symmetricEigenpairs,
} from '../../data/gmm-models';
import { candidates, narrowComponent, observations, scaler, selection } from '../../data/gmm-iris-data';
import { Table, curve, fixed, round } from './GmmShared.jsx';
import './gmm-labs.css';

const families = ['full', 'tied', 'diag', 'spherical'];
const familyStroke = { full: '#e7b94a', tied: '#8eb9a5', diag: '#91aecf', spherical: '#c8a2c8' };
const familyDash = { full: '', tied: '5 3', diag: '2 3', spherical: '7 3' };
const names = ['A', 'B', 'C', 'D'];

/** §2 · F1. Select, draw, then hide the selector. */
export function SelectorFigure() {
  const components = presets.twoBell;
  const span = [-6, 6];
  const place = value => 30 + 280 * (value - span[0]) / (span[1] - span[0]);
  const lift = value => 116 - 94 * value / 0.23;
  const at = mixtureAt(components, 2);
  return <figure className="gm-figure">
    <figcaption><strong>The model selects one component for each observation. Adding the weighted curves describes what we see after the selector is hidden.</strong></figcaption>
    <svg viewBox="0 0 340 266" role="img" aria-label="A hidden selector chooses A or B with probability one half each. Given A the measurement is drawn from a normal distribution with mean minus 2 and variance 1; given B, mean 2 and variance 1. Below, the two weighted curves and their sum share one axis. At x equals 2 the mixture density is 0.199538055.">
      <rect className="gm-hidden-node" x="98" y="6" width="144" height="28" rx="14" />
      <text x="170" y="24" textAnchor="middle">z: A or B, hidden</text>
      <path className="gm-flow" d="M140,34 L96,58" />
      <path className="gm-flow" d="M200,34 L244,58" />
      <text x="86" y="48" textAnchor="end">P(A) = 0.5</text>
      <text x="254" y="48">P(B) = 0.5</text>
      <rect className="gm-lane" x="34" y="60" width="120" height="26" rx="3" />
      <text x="94" y="77" textAnchor="middle">N(x; −2, 1)</text>
      <rect className="gm-lane" x="186" y="60" width="120" height="26" rx="3" />
      <text x="246" y="77" textAnchor="middle">N(x; 2, 1)</text>
      <path className="gm-flow" d="M94,86 L156,104" />
      <path className="gm-flow" d="M246,86 L184,104" />
      <text x="170" y="118" textAnchor="middle">one observed measurement</text>
      <g transform="translate(0 126)">
        <text x="30" y="10">dashed: each component, weighted</text>
        <polyline className="gm-curve is-left" points={curve(place, lift, span, value => components[0].weight * normal(value, components[0].mean, components[0].variance))} />
        <polyline className="gm-curve is-right" points={curve(place, lift, span, value => components[1].weight * normal(value, components[1].mean, components[1].variance))} />
        <polyline className="gm-curve is-mixture" points={curve(place, lift, span, value => mixtureAt(components, value).density)} />
        <line className="gm-axis" x1="30" x2="310" y1={lift(0)} y2={lift(0)} />
        {[-6, -4, -2, 0, 2, 4, 6].map(value => <text key={value} x={place(value)} y={lift(0) + 16} textAnchor="middle">{value}</text>)}
        <circle className="gm-mark" cx={place(2)} cy={lift(0)} r="5" />
        <line className="gm-stem is-right" x1={place(2)} x2={place(2)} y1={lift(0)} y2={lift(at.density)} />
        <text x={place(2) + 8} y={lift(at.density) - 5}>p(2) = {round(at.density)}</text>
        <text x="30" y="24">solid: their sum</text>
      </g>
    </svg>
    <p>
      Each weighted curve has area 0.5 and the sum has area 1. At a shared location you add the two heights; you do not average the two
      means and draw one bell, which would usually be a different density. The drawn point at 2 is an example location, not a new random draw.
    </p>
  </figure>;
}

/** §3 · F2. Fractional observations become weighted statistics. */
export function AllocationFigure() {
  const cycle = emCycle(presets.observations, presets.start, 0.05);
  const span = [-3.4, 3.4];
  const place = value => 34 + 272 * (value - span[0]) / (span[1] - span[0]);
  const counts = [0, 1].map(column => cycle.responsibilities.reduce((sum, row) => sum + row[column], 0));
  return <figure className="gm-figure gm-allocation-figure">
    <figcaption><strong>The observations stay fixed. Their fractional allocations move the fitted centres and spreads.</strong></figcaption>
    <div className="gm-allocation-stage">
      <h4>1. Keep the four measurements fixed</h4>
      <p>Each column below its measurement adds up to one observation: green is its Left share; blue is its Right share.</p>
    <svg className="gm-allocation-measurements" viewBox="0 0 340 142" role="img" aria-label={`Four fixed measurements and their allocation shares: ${cycle.responsibilities.map((row, index) => `${names[index]} at ${presets.observations[index]} gives ${round(row[0], 6)} to Left and ${round(row[1], 6)} to Right`).join('; ')}. Each column has total mass one.`}>
      <line className="gm-axis" x1="30" x2="314" y1="34" y2="34" />
      {presets.observations.map((value, index) => <g key={index}>
        <circle className="gm-observation" cx={place(value)} cy="34" r="4" />
        <text x={place(value)} y="18" textAnchor="middle">{names[index]}</text>
        <text x={place(value)} y="57" textAnchor="middle">{value}</text>
        <rect x={place(value) - 12} y="76" width="24" height={54 * cycle.responsibilities[index][0]} fill="#8eb9a5" />
        <rect x={place(value) - 12} y={76 + 54 * cycle.responsibilities[index][0]} width="24" height={54 * cycle.responsibilities[index][1]} fill="#91aecf" />
      </g>)}
    </svg>
      <p className="gm-caption">A and B mostly support Left; C and D mostly support Right. The exact shares are in the table below.</p>
    </div>
    <div className="gm-allocation-stage">
      <h4>2. Update each component from its weighted measurements</h4>
      <p>A dot marks the mean; a bar extends one standard deviation on either side. Every strip uses the same horizontal scale.</p>
      <div className="gm-allocation-components">
        {[0, 1].map(column => {
          const name = column === 0 ? 'Left' : 'Right';
          const component = cycle.after[column];
          return <section className="gm-panel" key={name} aria-label={`${name} component update`}>
            <h4>{name} component</h4>
            <p>Effective count <strong>{round(counts[column], 6)}</strong> · weight <strong>{round(component.weight, 4)}</strong></p>
            {[['Before', presets.start[column]], ['After', component]].map(([stage, parameters]) => {
              const spread = Math.sqrt(parameters.variance);
              return <div className="gm-allocation-step" key={stage}>
                <p><strong>{stage}</strong><br />Mean {round(parameters.mean, 4)} · standard deviation {round(spread, 4)}</p>
                <svg viewBox="0 0 340 62" role="img" aria-label={`${name}, ${stage.toLowerCase()}: mean ${round(parameters.mean, 9)}; standard deviation ${round(spread, 9)}. Bar from ${round(parameters.mean - spread, 9)} to ${round(parameters.mean + spread, 9)}, on the shared minus 3.4 to 3.4 scale.`}>
                  <line className="gm-axis" x1="30" x2="314" y1="22" y2="22" />
                  {[-3, -2, -1, 0, 1, 2, 3].map(tick => <g key={tick}>
                    <line className="gm-axis" x1={place(tick)} x2={place(tick)} y1="29" y2="33" />
                    <text x={place(tick)} y="53" textAnchor="middle">{tick}</text>
                  </g>)}
                  <line className={`gm-stem ${column === 0 ? 'is-left' : 'is-right'}`} strokeDasharray={stage === 'Before' ? '4 3' : undefined}
                    x1={place(parameters.mean - spread)} x2={place(parameters.mean + spread)} y1="22" y2="22" />
                  <circle className={stage === 'Before' ? 'gm-observation' : 'gm-mark'} cx={place(parameters.mean)} cy="22" r="4" />
                </svg>
              </div>;
            })}
          </section>;
        })}
      </div>
    </div>
    <Table caption="The responsibility matrix that produced this update, and the weighted statistics it implies"
      headings={['observation', 'x', 'Left share', 'Right share']}
      rows={presets.observations.map((value, index) => [
        names[index], round(value), round(cycle.responsibilities[index][0], 9), round(cycle.responsibilities[index][1], 9)])} />
    <p>
      The horizontal bars are one component standard deviation either side of a fitted centre, not a confidence interval for the centre.
      Left's weighted sum is {round(presets.observations.reduce((sum, value, index) => sum + value * cycle.responsibilities[index][0], 0), 9)} over
      an effective count of {round(counts[0], 6)}, giving the mean {round(cycle.after[0].mean, 9)} and variance {round(cycle.after[0].variance, 9)}.
      The total log-likelihood rises from {round(cycle.logLikelihoodBefore, 9)} to {round(cycle.logLikelihoodAfter, 9)}.
    </p>
  </figure>;
}

/** §4 · F3. A spike can game the likelihood. */
const collapseSigmas = [1, 0.1, 0.01, 0.001];
export function CollapseFigure() {
  const values = collapseSigmas.map(sigma => ({ sigma, logLikelihood: collapseLogLikelihood(sigma) }));
  const broad = { weight: 0.5, mean: 0, variance: 4 };
  // One window for all three, so the width contrast is the thing you see.
  const shared = [-3, -1];
  const sharedPlace = value => 40 + 256 * (value - shared[0]) / (shared[1] - shared[0]);
  const sharedTop = 0.5 * normal(-2, -2, collapseSigmas[2] ** 2);
  // A square-root height axis keeps the widest bell visible while the narrow one
  // is still obviously taller; the exact peaks are printed beside it.
  const sharedLift = value => 144 - 84 * Math.sqrt(value / sharedTop);
  const strip = value => 40 + 250 * (Math.log10(value) + 3) / 3;
  const lowest = Math.min(...values.map(row => row.logLikelihood));
  const highest = Math.max(...values.map(row => row.logLikelihood));
  const stripLift = value => 88 - 64 * (value - lowest) / (highest - lowest);
  return <figure className="gm-figure">
    <figcaption><strong>A component can increase the likelihood by concentrating on one observed location. A fixed variance floor stops this particular route to infinity.</strong></figcaption>
    <div className="gm-panels">
      <div className="gm-panel">
        <h4>Three standard deviations on one window</h4>
        <svg viewBox="0 0 300 168" role="img" aria-label={`The narrow component at standard deviations 1, 0.1 and 0.01, all drawn on the same window from minus 3 to minus 1, with a square-root height axis. Their weighted peaks at minus 2 are ${collapseSigmas.slice(0, 3).map(sigma => round(0.5 * normal(-2, -2, sigma * sigma), 4)).join(', ')}. The broad component stays at ${round(0.5 * normal(-2, 0, 4), 6)} throughout.`}>
          {collapseSigmas.slice(0, 3).map((sigma, index) => (
            <polyline key={sigma} className="gm-curve" stroke={['#8eb9a5', '#e7b94a', '#da9c86'][index]}
              strokeDasharray={['5 3', '', '1 2'][index]}
              points={curve(sharedPlace, sharedLift, shared, value => 0.5 * normal(value, -2, sigma * sigma))} />
          ))}
          <polyline className="gm-curve is-right" points={curve(sharedPlace, sharedLift, shared, value => broad.weight * normal(value, broad.mean, broad.variance))} />
          <line className="gm-axis" x1="40" x2="296" y1={sharedLift(0)} y2={sharedLift(0)} />
          <circle className="gm-observation" cx={sharedPlace(-2)} cy={sharedLift(0)} r="4" />
          <text x={sharedPlace(-2)} y={sharedLift(0) + 16} textAnchor="middle">x = −2</text>
          <text x="10" y="12">σ = 1 dashed green, 0.1 gold, 0.01 dotted</text>
          <text x="10" y="26">blue: the unchanged broad component</text>
          <text x="10" y="40">height axis is a square root of density</text>
        </svg>
        <p>The same window and the same broad component throughout: only the narrow component changes, and it narrows as its peak grows.</p>
      </div>
      <div className="gm-panel">
        <h4>The objective, against a logarithmic σ</h4>
        <svg viewBox="0 0 300 128" role="img" aria-label={`Total four-row log-likelihood against standard deviation on a logarithmic axis: ${values.map(row => `at ${row.sigma} it is ${round(row.logLikelihood, 9)}`).join('; ')}. It rises without bound as the standard deviation falls.`}>
          <line className="gm-axis" x1="40" x2="296" y1="88" y2="88" />
          <polyline className="gm-curve is-mixture" points={values.map(row => `${strip(row.sigma)},${stripLift(row.logLikelihood)}`).join(' ')} />
          {values.map(row => <g key={row.sigma}>
            <circle className="gm-mark" cx={strip(row.sigma)} cy={stripLift(row.logLikelihood)} r="3.5" />
            <text x={strip(row.sigma)} y="104" textAnchor="middle">{row.sigma}</text>
          </g>)}
          <text x="10" y="12">log-likelihood, rising as σ falls</text>
          <text x={strip(0.001) + 8} y={stripLift(highest) + 4}>{round(highest, 6)}</text>
          <text x={strip(1) - 8} y={stripLift(lowest) - 6} textAnchor="end">{round(lowest, 6)}</text>
          <text x="40" y="120">standard deviation, logarithmic</text>
        </svg>
        <p>Four points, no fitted curve between them: the values are exact and the trend is the argument.</p>
      </div>
    </div>
    <Table caption="Exact four-row log-likelihood as the narrow component shrinks, with no variance floor in force"
      headings={['standard deviation', 'narrow variance', 'weighted peak at −2', 'total log-likelihood']}
      rows={values.map(row => [row.sigma, round(row.sigma ** 2, 9), round(0.5 * normal(-2, -2, row.sigma ** 2), 4), fixed(row.logLikelihood, 9)])} />
    <p>
      One component stays broad and keeps the other three rows finite. The other narrows onto the observed −2, where its density grows like
      1/σ, so the total rises without bound. The plotted values do not prove the limit; the algebra in the section does. This is covariance
      collapse, not a successful recovery of four measurements.
    </p>
  </figure>;
}

/** §5 · F4. What each covariance restriction removes. */
const gallery = [
  { family: 'full', left: [[2, 0.75], [0.75, 1]], right: [[0.5, -0.25], [-0.25, 1.5]], note: 'Separate general matrices: different spreads and different orientations.' },
  { family: 'tied', left: [[1, 0.5], [0.5, 1]], right: [[1, 0.5], [0.5, 1]], note: 'One shared matrix: the same ellipse translated to each mean.' },
  { family: 'diag', left: [[2, 0], [0, 1]], right: [[0.5, 0], [0, 1.5]], note: 'Zero off-diagonals: axes aligned with the chosen feature coordinates.' },
  { family: 'spherical', left: [[1.5, 0], [0, 1.5]], right: [[1, 0], [0, 1]], note: 'One scalar per component: circles of different radii, not one shared radius.' },
];
function ellipsePoints(covariance, centre, place, lift, level = 1) {
  const { eigenvalues, directions } = symmetricEigenpairs(covariance);
  return Array.from({ length: 97 }, (_, index) => {
    const angle = 2 * Math.PI * index / 96;
    const x = centre[0] + Math.sqrt(level * eigenvalues[0]) * Math.cos(angle) * directions[0][0]
      + Math.sqrt(level * eigenvalues[1]) * Math.sin(angle) * directions[1][0];
    const y = centre[1] + Math.sqrt(level * eigenvalues[0]) * Math.cos(angle) * directions[0][1]
      + Math.sqrt(level * eigenvalues[1]) * Math.sin(angle) * directions[1][1];
    return `${place(x).toFixed(2)},${lift(y).toFixed(2)}`;
  }).join(' ');
}
export function CovarianceGalleryFigure() {
  // One scale for both axes, so a circle is drawn as a circle.
  const unit = 34;
  const place = value => 150 + unit * value;
  const lift = value => 80 - unit * value;
  return <figure className="gm-figure">
    <figcaption><strong>Four covariance families, the same two means, and the shapes each restriction still allows</strong></figcaption>
    <div className="gm-panels">
      {gallery.map(entry => (
        <div className="gm-panel" key={entry.family}>
          <h4>{entry.family}</h4>
          <svg viewBox="0 0 300 168" role="img"
            aria-label={`${entry.family}: the left component at minus 2, 0 has covariance rows ${entry.left.map(row => row.join(' and ')).join('; ')}, and the right component at 2, 0 has ${entry.right.map(row => row.join(' and ')).join('; ')}. Both axes use the same scale.`}>
            <line className="gm-grid" x1={place(-4.1)} x2={place(4.1)} y1={lift(0)} y2={lift(0)} />
            <line className="gm-grid" x1={place(0)} x2={place(0)} y1={lift(-2.1)} y2={lift(2.1)} />
            <polyline className="gm-ellipse" points={ellipsePoints(entry.left, [-2, 0], place, lift)} />
            <polyline className="gm-ellipse is-outer" points={ellipsePoints(entry.right, [2, 0], place, lift)} />
            <circle className="gm-observation" cx={place(-2)} cy={lift(0)} r="3" />
            <rect className="gm-mark is-hollow" x={place(2) - 3} y={lift(0) - 3} width="6" height="6" />
            <text x={place(-2)} y="164" textAnchor="middle">solid: left</text>
            <text x={place(2)} y="164" textAnchor="middle">dashed: right</text>
          </svg>
          <p className="gm-matrices">
            {entry.family === 'spherical'
              ? <><span>left: {entry.left[0][0]} I</span><span>right: {entry.right[0][0]} I</span></>
              : entry.family === 'tied'
                ? <span>shared by both: [{entry.left[0].join(', ')}; {entry.left[1].join(', ')}]</span>
                : <><span>left: [{entry.left[0].join(', ')}; {entry.left[1].join(', ')}]</span><span>right: [{entry.right[0].join(', ')}; {entry.right[1].join(', ')}]</span></>}
          </p>
          <p>{entry.note}</p>
        </div>
      ))}
    </div>
    <Table caption="Parameter totals for two features and three components, from the section's table"
      headings={['family', 'covariance parameters', 'means', 'free weights', 'total']}
      rows={families.map(family => {
        const covariance = { full: 9, tied: 3, diag: 6, spherical: 3 }[family];
        return [family, covariance, 6, 2, covariance + 8];
      })} />
    <p>
      Every contour is where the squared Mahalanobis distance equals 1, drawn from each matrix's own eigenvectors and eigenvalues rather than
      a chosen rotation, with both axes on one scale. Tied shares one matrix, so its two shapes are congruent. Spherical gives each component
      its own radius scale; it is not a shared-variance k-means switch. Rotating correlated data can turn the diagonal restriction into a
      different model family, so the coordinate system is part of the choice.
    </p>
  </figure>;
}

/** §7 · F5. Held-out selection, training BIC, and the narrow component. */
export function CandidateFigure() {
  const byFamily = families.map(family => ({
    family,
    rows: candidates.filter(row => row[0] === family).sort((left, right) => left[1] - right[1]),
  }));
  const place = value => 52 + 218 * (value - 1) / 3;
  const liftScore = value => 118 - 96 * (value + 3.35) / 0.75;
  const liftBic = value => 118 - 96 * (value - 450) / 100;
  const selected = candidates.find(row => row[0] === 'full' && row[1] === 2);
  const bicWinner = candidates.find(row => row[0] === 'full' && row[1] === 4);
  const trainRows = observations.filter(row => row[3] === 0);
  const rawWidth = narrowComponent.rawMean[1];
  const half = 0.06;
  const lengthPlace = value => 52 + 218 * (value - 4.2) / 3.6;
  const widthLift = value => 96 - 72 * (value - 1.8) / 2.6;
  const zoomLift = value => 96 - 72 * (value - (rawWidth - half)) / (2 * half);
  const inStrip = trainRows.filter(row => Math.abs(row[2] - rawWidth) <= half);
  // The component's own one-deviation contour, transformed back to centimetres.
  // Its length axis is about 0.95 cm wide and its width axis 0.0042 cm, so it
  // draws as the extremely eccentric ellipse it is.
  const narrowEllipse = (place, liftValue) => {
    const lengthSpread = Math.sqrt(narrowComponent.covariance[0][0]) * scaler.scale[0];
    const widthSpread = narrowComponent.rawWidthStandardDeviation;
    return Array.from({ length: 97 }, (_, index) => {
      const angle = 2 * Math.PI * index / 96;
      return `${place(narrowComponent.rawMean[0] + lengthSpread * Math.cos(angle)).toFixed(2)},${liftValue(rawWidth + widthSpread * Math.sin(angle)).toFixed(2)}`;
    }).join(' ');
  };
  return <figure className="gm-figure">
    <figcaption><strong>The chosen validation rule selects full K = 2. Training BIC favours a sharper K = 4 fit; inspect the component geometry before interpreting that preference.</strong></figcaption>
    <div className="gm-panels">
      <div className="gm-panel">
        <h4>Validation mean log-density, higher is preferred</h4>
        <svg viewBox="0 0 300 140" role="img" aria-label={`Validation mean log-density by number of components for each covariance family. ${byFamily.map(entry => `${entry.family}: ${entry.rows.map(row => `K equals ${row[1]} gives ${row[2]}`).join(', ')}`).join('. ')}. The maximum is full with two components at ${selected[2]}, with full K equals 3 very close at ${candidates.find(row => row[0] === 'full' && row[1] === 3)[2]}.`}>
          {[1, 2, 3, 4].map(k => <text key={k} x={place(k)} y="134" textAnchor="middle">K = {k}</text>)}
          <line className="gm-axis" x1="52" x2="280" y1="118" y2="118" />
          <line className="gm-axis" x1="52" x2="52" y1="16" y2="118" />
          <text x="8" y={liftScore(-2.65) + 4}>−2.65</text>
          <text x="8" y={liftScore(-3.3) + 4}>−3.30</text>
          {byFamily.map(entry => <g key={entry.family}>
            <polyline className="gm-curve" stroke={familyStroke[entry.family]} strokeDasharray={familyDash[entry.family]}
              points={entry.rows.map(row => `${place(row[1])},${liftScore(row[2])}`).join(' ')} />
            {entry.rows.map(row => <circle key={row[1]} cx={place(row[1])} cy={liftScore(row[2])} r="3" fill={familyStroke[entry.family]} />)}
          </g>)}
          <circle className="gm-mark" cx={place(2)} cy={liftScore(selected[2])} r="6" />
          <text x={place(2) + 10} y={liftScore(selected[2]) - 8}>selected</text>
        </svg>
        <p>Filled marker: the declared validation winner. Solid gold is full, dashed green tied, dotted blue diagonal, long-dashed violet spherical.</p>
      </div>
      <div className="gm-panel">
        <h4>Training BIC, lower is preferred</h4>
        <svg viewBox="0 0 300 140" role="img" aria-label={`Training BIC by number of components. ${byFamily.map(entry => `${entry.family}: ${entry.rows.map(row => `K equals ${row[1]} gives ${row[3]}`).join(', ')}`).join('. ')}. The minimum is full with four components at ${bicWinner[3]}.`}>
          {[1, 2, 3, 4].map(k => <text key={k} x={place(k)} y="134" textAnchor="middle">K = {k}</text>)}
          <line className="gm-axis" x1="52" x2="280" y1="118" y2="118" />
          <line className="gm-axis" x1="52" x2="52" y1="16" y2="118" />
          <text x="14" y={liftBic(540) + 4}>540</text>
          <text x="14" y={liftBic(460) + 4}>460</text>
          {byFamily.map(entry => <g key={entry.family}>
            <polyline className="gm-curve" stroke={familyStroke[entry.family]} strokeDasharray={familyDash[entry.family]}
              points={entry.rows.map(row => `${place(row[1])},${liftBic(row[3])}`).join(' ')} />
            {entry.rows.map(row => <circle key={row[1]} cx={place(row[1])} cy={liftBic(row[3])} r="3" fill={familyStroke[entry.family]} />)}
          </g>)}
          <circle className="gm-mark is-hollow" cx={place(4)} cy={liftBic(bicWinner[3])} r="6" />
          <text x={place(4) - 12} y={liftBic(bicWinner[3]) - 10} textAnchor="end">lowest BIC</text>
        </svg>
        <p>Open marker: the training-BIC winner, shown as a diagnostic and not as the selection rule.</p>
      </div>
    </div>
    <div className="gm-panels">
      <div className="gm-panel">
        <h4>The 90 training flowers, in centimetres</h4>
        <svg viewBox="0 0 300 116" role="img" aria-label={`Ninety training observations, sepal length against sepal width in centimetres. Length runs from 4.2 to 7.8 and width from 1.8 to 4.4. The narrow component of the full K equals 4 fit is centred at ${round(narrowComponent.rawMean[0], 3)} centimetres of length and ${round(rawWidth, 4)} of width, and its one-deviation contour spans about ${round(2 * Math.sqrt(narrowComponent.covariance[0][0]) * scaler.scale[0], 3)} centimetres along the length axis and ${round(2 * narrowComponent.rawWidthStandardDeviation, 6)} across the width axis.`}>
          <line className="gm-axis" x1="52" x2="280" y1="96" y2="96" />
          <line className="gm-axis" x1="52" x2="52" y1="12" y2="96" />
          {trainRows.map(row => (
            <circle key={row[0]} className="gm-observation" cx={lengthPlace(row[1])} cy={widthLift(row[2])} r="2.2" fillOpacity=".7" />
          ))}
          <polyline className="gm-ellipse" points={narrowEllipse(lengthPlace, widthLift)} />
          <line className="gm-stem is-right" x1="52" x2="280" y1={widthLift(rawWidth)} y2={widthLift(rawWidth)} />
          <text x="8" y={widthLift(4.4) + 4}>4.4</text>
          <text x="8" y={widthLift(1.8)}>1.8</text>
          <text x="52" y="112">4.2 cm</text>
          <text x="280" y="112" textAnchor="end">7.8 cm</text>
        </svg>
        <p>Sepal width against sepal length, at the scale the measurements were recorded, with the component's one-deviation contour drawn over them.</p>
      </div>
      <div className="gm-panel">
        <h4>The same component's width, magnified</h4>
        <svg viewBox="0 0 300 116" role="img" aria-label={`A magnified strip around sepal width ${round(rawWidth, 4)} centimetres, spanning ${round(2 * half, 3)} centimetres in total. The component's fitted width standard deviation is ${round(narrowComponent.rawWidthStandardDeviation, 6)} centimetres, far narrower than the recorded resolution of ${narrowComponent.recordedResolution} centimetres. The ${inStrip.length} training rows inside the strip are observation IDs ${inStrip.map(row => row[0]).join(', ')}, and every one sits exactly on one recorded value.`}>
          <line className="gm-axis" x1="52" x2="280" y1="96" y2="96" />
          <line className="gm-axis" x1="52" x2="52" y1="12" y2="96" />
          {/* Drawn at its true height on this scale: the band is thin because the
              fitted width is thin, and padding it would misreport the diagnosis. */}
          <rect x="52" y={zoomLift(rawWidth + narrowComponent.rawWidthStandardDeviation)}
            width="228"
            height={zoomLift(rawWidth - narrowComponent.rawWidthStandardDeviation) - zoomLift(rawWidth + narrowComponent.rawWidthStandardDeviation)}
            fill="#91aecf" fillOpacity=".7" />
          {inStrip.map(row => (
            <circle key={row[0]} className="gm-observation" cx={lengthPlace(row[1])} cy={zoomLift(row[2])} r="3" />
          ))}
          <text x="8" y={zoomLift(rawWidth + half) + 10}>{round(rawWidth + half, 2)}</text>
          <text x="8" y={zoomLift(rawWidth - half)}>{round(rawWidth - half, 2)}</text>
          <text x="52" y="112">band: one fitted SD, {round(narrowComponent.rawWidthStandardDeviation, 6)} cm</text>
        </svg>
        <p>This vertical scale spans {round(2 * half, 2)} cm, not the 2.6 cm beside it. Every row in the strip sits on one recorded value.</p>
      </div>
    </div>
    <Table caption="Every candidate, in training-standardized coordinates"
      headings={['covariance', 'K', 'validation mean log-density', 'training BIC']}
      rows={candidates.map(row => [row[0], row[1], fixed(row[2], 6), fixed(row[3], 3)])}
      rowClass={index => (candidates[index][0] === 'full' && candidates[index][1] === 2 ? 'is-selected' : undefined)}
      scroll />
    <details>
      <summary>Inspect all 150 observation IDs, measurements and split memberships</summary>
      <Table caption="The observed sepal measurements and their fixed evaluation roles"
        headings={['observation ID', 'sepal length (cm)', 'sepal width (cm)', 'split']}
        rows={observations.map(row => [row[0], row[1], row[2], ['training', 'validation', 'test'][row[3]]])} scroll />
    </details>
    <p>
      The component drawn above holds weight {round(narrowComponent.weight, 6)} and a fitted sepal-width variance of exactly
      {' '}{narrowComponent.covariance[1][1]} in standardized coordinates, which is the additive regularization level supplied to the fit.
      Back in centimetres that is a standard deviation of {round(narrowComponent.rawWidthStandardDeviation, 6)} cm, narrower than the
      {' '}{narrowComponent.recordedResolution} cm at which the measurements were recorded. It is a diagnosis to investigate, not an
      established account of every candidate's behaviour. Component numbering belongs to this fitted run; it names no species, and species
      colours are deliberately absent from every panel here.
    </p>
  </figure>;
}

/** §9 · F6. Touch, lift, touch again. */
export function BoundChainFigure() {
  const oldComponents = presets.start;
  const start = expectation(presets.observations, oldComponents);
  const cycle = emCycle(presets.observations, oldComponents, 0.05);
  const oldBound = boundDecomposition(presets.observations, oldComponents, start.responsibilities);
  const newBound = boundDecomposition(presets.observations, cycle.after, start.responsibilities);
  const newObjective = cycle.logLikelihoodAfter;
  const lift = value => 180 - 140 * (value + 7.4) / 1.1;
  return <figure className="gm-figure">
    <figcaption><strong>The old E-step makes the bound equal to the old objective. The M-step raises that bound. The new objective lies at least as high.</strong></figcaption>
    <svg viewBox="0 0 340 216" role="img" aria-label={`At the old parameters the objective and the bound are both ${round(oldBound.elbo, 9)}. Holding the same responsibilities, the bound at the new parameters is ${round(newBound.elbo, 9)}, and the new objective is ${round(newObjective, 9)}, which is ${round(newObjective - newBound.elbo, 9)} higher.`}>
      <text x="80" y="14" textAnchor="middle">old parameters</text>
      <text x="260" y="14" textAnchor="middle">new parameters</text>
      <line className="gm-grid" x1="24" x2="316" y1={lift(oldBound.elbo)} y2={lift(oldBound.elbo)} />
      <line className="gm-stem is-left" x1="30" x2="130" y1={lift(oldBound.elbo)} y2={lift(oldBound.elbo)} />
      <text x="30" y={lift(oldBound.elbo) - 8}>objective = bound</text>
      <text x="30" y={lift(oldBound.elbo) + 18}>{round(oldBound.elbo, 9)}</text>
      <line className="gm-stem" stroke="#e7b94a" x1="210" x2="310" y1={lift(newObjective)} y2={lift(newObjective)} />
      <text x="310" y={lift(newObjective) - 8} textAnchor="end">objective {round(newObjective, 9)}</text>
      <line className="gm-stem is-right" x1="210" x2="310" y1={lift(newBound.elbo)} y2={lift(newBound.elbo)} />
      <text x="310" y={lift(newBound.elbo) + 18} textAnchor="end">bound {round(newBound.elbo, 9)}</text>
      <text x="260" y={(lift(newObjective) + lift(newBound.elbo)) / 2 + 4} textAnchor="middle">gap {round(newObjective - newBound.elbo, 6)}</text>
      <path className="gm-flow" d={`M132,${lift(oldBound.elbo)} C168,${lift(oldBound.elbo)} 172,${lift(newBound.elbo)} 208,${lift(newBound.elbo)}`} />
      <text x="170" y="196" textAnchor="middle">the M-step raises the same bound</text>
      <text x="170" y="210" textAnchor="middle">and the next E-step closes the gap</text>
    </svg>
    <Table caption="The bound is the expected complete-data log-likelihood plus the entropy of the same allocations"
      headings={['evaluated at', 'Q', 'entropy H', 'bound Q + H', 'observed log-likelihood']}
      rows={[
        ['old parameters, old responsibilities', fixed(oldBound.qFunction, 9), fixed(oldBound.entropy, 9), fixed(oldBound.elbo, 9), fixed(start.logLikelihood, 9)],
        ['new parameters, old responsibilities', fixed(newBound.qFunction, 9), fixed(newBound.entropy, 9), fixed(newBound.elbo, 9), fixed(newObjective, 9)],
      ]} />
    <p>
      Read the chain right to left: {round(newObjective, 9)} ≥ {round(newBound.elbo, 9)} ≥ {round(oldBound.elbo, 9)}, and the last value
      equals the old objective exactly because the E-step used that model's own posterior. The entropy {round(oldBound.entropy, 9)} is the
      same in both rows, which is why maximising Q with the allocations held fixed also maximises the bound. Raising an arbitrary lower bound
      would prove nothing: the equality between the old bound and old objective is what makes the chain work.
    </p>
  </figure>;
}
