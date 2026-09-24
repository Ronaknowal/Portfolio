import { useState } from 'react';
import {
  cellTerms, coneCoordinates, contributionRows, fixtures, gradientH, halfSquaredLoss, integerRank, lossesAt,
  maskedReport, nonsingularMinor, normalizePatterns, reconstruct, residual, supportPairs, sweep, updateH,
} from '../../data/nmf-models.js';
import { NMF_DIGITS } from '../../data/nmf-data.js';
import {
  Figure, ImagePanel, ImageValues, Plot, ScaleKey, Strip, Table, fixed, round, signed,
} from './NmfShared.jsx';
import './nmf-labs.css';

const featureNames = ['feature 1', 'feature 2', 'feature 3'];

/** A small read-only matrix with one cell highlighted and its row or column
 * partners marked, so a dot product can be pointed at rather than described. */
function Matrix({ title, rows, shape, marked = [], related = [], note, format = value => round(value, 6) }) {
  return <div className="nm-matrix-group">
    <h4>{title} <span className="nm-caption">{shape}</span></h4>
    <div className="nm-matrix" style={{ gridTemplateColumns: `repeat(${rows[0].length}, auto)` }}
      role="img" aria-label={`${title}, ${shape}. Rows: ${rows.map(row => row.map(value => round(value, 6)).join(', ')).join('; ')}.`}>
      {rows.flatMap((row, r) => row.map((value, c) => {
        const isMarked = marked.some(([mr, mc]) => mr === r && mc === c);
        const isRelated = related.some(([mr, mc]) => mr === r && mc === c);
        return <span key={`${r}-${c}`} className={`nm-matrix-cell${isMarked ? ' is-marked' : ''}${isRelated ? ' is-related' : ''}`}>{format(value)}</span>;
      }))}
    </div>
    {note && <p>{note}</p>}
  </div>;
}

/** §1 · F1. One observation is the sum of its weighted component patterns. */
export function BuildRowFigure() {
  const [feature, setFeature] = useState(2);
  const weighted = contributionRows(fixtures.W1, fixtures.H1, 0);
  const total = reconstruct(fixtures.W1, fixtures.H1)[0];
  const terms = cellTerms(fixtures.W1, fixtures.H1, 0, feature);
  return <Figure title="Build one observation: scale each pattern by its amount, then add the strips."
    caption="Cells are the three features in a fixed order, not time. Intensity and the printed number both encode the amount, on one scale from 0 to 3. The two weighted strips are the k = 2 contributions to a single observation, not two categories of observation.">
    <p className="nm-caption">Select a feature to expand its sum:</p>
    <ul className="nm-toggles">
      {featureNames.map((name, index) => <li key={name}>
        <button type="button" aria-pressed={feature === index} onClick={() => setFeature(index)}>{name}</button>
      </li>)}
    </ul>
    <Strip label="Component 1 pattern, H row 1 = [1, 0, 1]" values={fixtures.H1[0]} maximum={3} highlight={[feature]} />
    <Strip label="Component 2 pattern, H row 2 = [0, 1, 1]" values={fixtures.H1[1]} maximum={3} highlight={[feature]} />
    <p className="nm-caption">Observation 1 uses amount <strong>2</strong> of component 1 and amount <strong>1</strong> of component 2. Multiply every cell of a pattern by its amount:</p>
    <Strip label="2 × component 1 = [2, 0, 2]" values={weighted[0]} maximum={3} highlight={[feature]} />
    <Strip label="1 × component 2 = [0, 1, 1]" values={weighted[1]} maximum={3} highlight={[feature]} />
    <p className="nm-caption">Those two contributions are the <em>k</em> = 2 addends. Adding them cell by cell gives the reconstruction, which here equals the observation exactly:</p>
    <Strip label="Sum of the two contributions = reconstructed row" values={total} maximum={3} highlight={[feature]} />
    <Strip label="Observed row, X row 1 = [2, 1, 3]" values={fixtures.X[0]} maximum={3} highlight={[feature]} />
    <p className="nm-readout" aria-live="polite">
      {featureNames[feature]}: {terms.terms.map(term => `${round(term.amount)} × ${round(term.patternCell)} = ${round(term.product)}`).join(' and ')},
      and {terms.terms.map(term => round(term.product)).join(' + ')} = {round(terms.total)}. Every contribution is zero or positive, so nothing cancels.
    </p>
    <Table caption="The same multiplication as a table, one row per addend"
      headings={['addend', 'feature 1', 'feature 2', 'feature 3']}
      rows={[
        ['amount 2 × pattern [1, 0, 1]', ...weighted[0].map(value => round(value))],
        ['amount 1 × pattern [0, 1, 1]', ...weighted[1].map(value => round(value))],
        ['sum = reconstructed row', ...total.map(value => round(value))],
        ['observed row', ...fixtures.X[0].map(value => round(value))],
      ]}
      rowClass={index => (index === 2 ? 'is-selected' : undefined)} />
  </Figure>;
}

/** §2 · F2. The residual is signed before it is squared. */
const lossRows = [[2, 4], [20, 22]];
export function ResidualFigure() {
  const signedResidual = residual([fixtures.observed], [fixtures.approximated])[0];
  const squared = signedResidual.map(value => value * value);
  const half = squared.reduce((sum, value) => sum + value, 0) / 2;
  const comparison = lossRows.map(([x, y]) => ({ x, y, ...lossesAt(x, y) }));
  const barWidth = (value, top) => `${Math.max(2, 100 * value / top)}%`;
  const klTop = Math.max(...comparison.map(row => row.kl));
  const isTop = Math.max(...comparison.map(row => row.itakuraSaito));
  return <Figure title="Error is a signed difference before it is squared."
    caption="The first two strips share a nonnegative 0 to 3 intensity scale. The residual strip has its own symmetric scale: gold is a positive residual, meaning missing reconstructed mass, and blue would be excess. Zero stays neutral grey with a printed 0.">
    <Strip label="Observed row = [2, 1, 3]" values={fixtures.observed} maximum={3} />
    <Strip label="Reconstructed row = [1.5, 1, 2.5]" values={fixtures.approximated} maximum={3} />
    <Strip label="Signed residual = observed − reconstructed = [.5, 0, .5]" values={signedResidual} tone="signed" />
    <ScaleKey tone="signed" maximum={0.5} note="Positive means the reconstruction is missing mass at that feature." />
    <Table caption="Each residual cell and the squared contribution it makes to the objective"
      headings={['feature', 'observed', 'reconstructed', 'signed residual', 'squared residual']}
      rows={fixtures.observed.map((value, index) => [
        featureNames[index], round(value), round(fixtures.approximated[index]),
        signed(signedResidual[index], 3), round(squared[index], 4),
      ]).concat([['total', '—', '—', signed(signedResidual.reduce((sum, value) => sum + value, 0), 3), round(squared.reduce((sum, value) => sum + value, 0), 4)]])} />
    <p className="nm-readout">Half the sum of the squared residuals is {round(half)}, which is this row's contribution to <em>F</em>.</p>
    <h4>The same absolute overestimate in two contexts</h4>
    <Table caption="One absolute overestimate of 2, valued by three entrywise losses"
      headings={['observed x, reconstructed y', 'half squared error', 'generalized KL', 'Itakura–Saito']}
      rows={comparison.map(row => [`${row.x}, ${row.y}`, round(row.frobeniusHalf), fixed(row.kl, 6), fixed(row.itakuraSaito, 6)])} />
    <div className="nm-panels">
      <div className="nm-panel">
        <h4>Generalized KL, its own axis</h4>
        {comparison.map(row => <div key={row.x} className="nm-strip-row">
          <span className="nm-strip-label">{row.x} against {row.y}</span>
          <span className="nm-bar-track" style={{ display: 'block', background: '#1d2622', borderRadius: '2px', height: '16px' }}>
            <span style={{ display: 'block', height: '100%', width: barWidth(row.kl, klTop), background: '#e7b94a', borderRadius: '2px' }} />
          </span>
        </div>)}
        <p>Bars share this panel's axis only, topping out at {fixed(klTop, 6)}.</p>
      </div>
      <div className="nm-panel">
        <h4>Itakura–Saito, its own axis</h4>
        {comparison.map(row => <div key={row.x} className="nm-strip-row">
          <span className="nm-strip-label">{row.x} against {row.y}</span>
          <span className="nm-bar-track" style={{ display: 'block', background: '#1d2622', borderRadius: '2px', height: '16px' }}>
            <span style={{ display: 'block', height: '100%', width: barWidth(row.itakuraSaito, isTop), background: '#8eb9a5', borderRadius: '2px' }} />
          </span>
        </div>)}
        <p>A different axis, topping out at {fixed(isTop, 6)}. The two panels are not comparable lengths, which is why they are not drawn on one axis.</p>
      </div>
    </div>
    <p className="nm-caption">
      These are exact values of three formulas on two pairs of numbers. Nothing here is an empirical study of which loss fits a dataset better.
    </p>
  </Figure>;
}

/** §3 · F3. The two phases of one alternating sweep. */
export function UpdatePhaseFigure() {
  const step = sweep(fixtures.X, fixtures.startW, fixtures.startH);
  const phase = updateH(fixtures.X, fixtures.startW, fixtures.startH);
  const gram = fixtures.startW[0].map((_, r) => fixtures.startW[0].map((__, c) =>
    fixtures.startW.reduce((sum, row) => sum + row[r] * row[c], 0)));
  return <Figure title="One sweep has two phases: H changes using the old W, then W changes using the new H."
    caption="Dimensions are labelled because the shapes decide which products are legal. The gold cell is H₁₁ throughout; the green cells are the entries that enter its numerator and denominator.">
    <div className="nm-matrices">
      <Matrix title="X, the data" shape="3 × 3" rows={fixtures.X} related={[[0, 0], [1, 0], [2, 0]]}
        note="Column 1 supplies the numerator." />
      <Matrix title="W⁽⁰⁾, the activations" shape="3 × 2" rows={fixtures.startW} related={[[0, 0], [1, 0], [2, 0]]}
        note="Column 1 is component 1 across the three observations." />
      {/* H₂₁ is the denominator's second summand, so it carries the same
          "enters the calculation" marking as X's and W's first columns. */}
      <Matrix title="H⁽⁰⁾, the patterns" shape="2 × 3" rows={fixtures.startH} marked={[[0, 0]]} related={[[1, 0]]}
        note="H₁₁ is the cell being updated; H₂₁ is the denominator's other summand." />
    </div>
    <p className="nm-readout">
      Numerator (WᵀX)₁₁ = 1·2 + .5·1 + 1·3 = {round(phase.numerator[0][0])}.
      The first row of WᵀW is [{gram[0].map(value => round(value)).join(', ')}], so the denominator ((WᵀW)H)₁₁ = {round(gram[0][0])}·1 + {round(gram[0][1])}·0.2 = {round(phase.denominator[0][0])}.
      The ratio is {round(phase.ratio[0][0], 9)}, so H₁₁ becomes 1 × {round(phase.numerator[0][0])}/{round(phase.denominator[0][0])} = {fixed(phase.next[0][0], 9)}.
    </p>
    <div className="nm-matrices">
      <Matrix title="Loss before any update" shape="F⁽⁰⁾" rows={[[step.lossBefore]]} format={value => fixed(value, 6)}
        note="Half the squared Frobenius norm of the residual." />
      <Matrix title="H⁽¹⁾, after the H phase" shape="2 × 3" rows={step.H} marked={[[0, 0]]} format={value => fixed(value, 6)}
        note={`Loss after this phase alone: ${fixed(step.lossAfterH, 9)}.`} />
      <Matrix title="W⁽¹⁾, after the W phase" shape="3 × 2" rows={step.W} format={value => fixed(value, 6)}
        note={`The W phase uses the new H. Loss after the full sweep: ${fixed(step.loss, 9)}.`} />
    </div>
    <p className="nm-caption">
      The post-H state and the post-W state are different states. Only after both is the sweep loss {fixed(step.loss, 9)}. Some activations fell while
      the matching pattern entries rose; the product is what the objective sees.
    </p>
  </Figure>;
}

/** §3. The zero-lock contrast, stated once as a static case. */
export function ZeroLockFigure() {
  const lock = fixtures.zeroLock;
  const gradient = gradientH(lock.X, lock.W, lock.H);
  return <Figure title="A zero multiplied by any ratio is still zero."
    caption="This is a defined convention, not an executed 0 / 0. The rule here is “leave a zero entry at zero”, which is exactly what a guarded implementation does.">
    <div className="nm-matrices">
      <Matrix title="Fixed W" shape="1 × 1" rows={lock.W} />
      <Matrix title="X" shape="1 × 2" rows={lock.X} />
      <Matrix title="H, with a zero first entry" shape="1 × 2" rows={lock.H} marked={[[0, 0]]} />
      <Matrix title="The gradient of F with respect to H" shape="1 × 2" rows={gradient} marked={[[0, 0]]}
        note="The first coordinate has gradient −2, so increasing it would reduce the loss." />
    </div>
    <Table caption="What two different procedures do with the same fixed W"
      headings={['procedure', 'resulting H', 'half squared loss']}
      rows={[
        ['Multiplicative update that leaves zeros at zero', `[${lock.H.map(row => row.join(', '))}]`, fixed(halfSquaredLoss(lock.X, lock.W, lock.H), 6)],
        ['Nonnegative least squares with this same fixed W', `[${lock.nnls.join(', ')}]`, fixed(halfSquaredLoss(lock.X, lock.W, [lock.nnls]), 6)],
      ]} />
    <p className="nm-caption">
      A flat objective is therefore not proof of a good fit. The move that would help is forbidden by the rule, not by the mathematics.
    </p>
  </Figure>;
}

/** §4 · F4. Two dictionaries, one measured region. */
const coneViews = [
  { key: 'both', label: 'Both dictionaries' },
  { key: 'first', label: 'First dictionary only' },
  { key: 'second', label: 'Second dictionary only' },
];
export function AmbiguityFigure() {
  const [view, setView] = useState('both');
  const size = 340;
  const high = 3.5;
  const pad = { left: 44, right: 16, top: 18, bottom: 40 };
  const place = value => pad.left + (size - pad.left - pad.right) * value / high;
  const lift = value => size - pad.bottom - (size - pad.top - pad.bottom) * value / high;
  const points = fixtures.X.map(row => [row[0], row[1]]);
  const dictionaries = [
    { key: 'first', name: 'First dictionary', rays: [[1, 0], [0, 1]], colour: '#e7b94a', polygon: [[0, 0], [3.5, 0], [3.5, 3.5], [0, 3.5]], fill: 'url(#nm-cone-a)' },
    { key: 'second', name: 'Second dictionary', rays: [[1.25, 0.25], [0.25, 1.25]], colour: '#8eb9a5', polygon: [[0, 0], [3.5, 0.7], [3.5, 3.5], [0.7, 3.5]], fill: 'url(#nm-cone-b)' },
  ];
  const shown = dictionaries.filter(entry => view === 'both' || view === entry.key);
  const coefficients = dictionaries.map(entry => points.map(point => coneCoordinates(entry.rays, point).coefficients));
  const doubled = { W: fixtures.W1.map(row => [row[0] / 2, row[1]]), H: fixtures.H1.map((row, index) => (index === 0 ? row.map(value => value * 2) : row)) };
  return <Figure title="Two different nonnegative dictionaries, and the same three observations inside both."
    caption="Axes are features 1 and 2 on one equal scale. Feature 3 equals feature 1 plus feature 2 throughout this fixture, so this plane view keeps the containment relationship intact. Both products equal X exactly; neither is a worse local minimum than the other.">
    <ul className="nm-toggles">
      {coneViews.map(entry => <li key={entry.key}>
        <button type="button" aria-pressed={view === entry.key} onClick={() => setView(entry.key)}>{entry.label}</button>
      </li>)}
    </ul>
    <svg viewBox={`0 0 ${size} ${size}`} role="img"
      aria-label={`Feature 1 horizontal, feature 2 vertical, both from 0 to 3.5 on the same scale. The first dictionary's rays run along the axes, through (1, 0) and (0, 1). The second dictionary's rays run through (1.25, 0.25) and (0.25, 1.25). The three observations (2, 1), (1, 2) and (3, 3) lie inside both cones. In the first cone their coefficients are ${coefficients[0].map(pair => `(${pair.map(value => round(value, 3)).join(', ')})`).join(', ')}; in the second they are ${coefficients[1].map(pair => `(${pair.map(value => round(value, 3)).join(', ')})`).join(', ')}.`}>
      <defs>
        <pattern id="nm-cone-a" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="8" height="8" fill="#e7b94a" fillOpacity=".07" />
          <line x1="0" y1="0" x2="0" y2="8" stroke="#e7b94a" strokeOpacity=".45" strokeWidth="1.4" />
        </pattern>
        <pattern id="nm-cone-b" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
          <rect width="8" height="8" fill="#8eb9a5" fillOpacity=".07" />
          <line x1="0" y1="0" x2="0" y2="8" stroke="#8eb9a5" strokeOpacity=".45" strokeWidth="1.4" />
        </pattern>
      </defs>
      {[0, 1, 2, 3].map(tick => <g key={tick}>
        <line className="nm-grid" x1={place(tick)} x2={place(tick)} y1={lift(0)} y2={lift(high)} />
        <line className="nm-grid" x1={place(0)} x2={place(high)} y1={lift(tick)} y2={lift(tick)} />
        <text x={place(tick)} y={lift(0) + 22} textAnchor="middle">{tick}</text>
        <text x={place(0) - 8} y={lift(tick) + 5} textAnchor="end">{tick}</text>
      </g>)}
      {shown.map(entry => <polygon key={entry.key} points={entry.polygon.map(([x, y]) => `${place(x)},${lift(y)}`).join(' ')} fill={entry.fill} />)}
      <line className="nm-axis" x1={place(0)} x2={place(high)} y1={lift(0)} y2={lift(0)} />
      <line className="nm-axis" x1={place(0)} x2={place(0)} y1={lift(0)} y2={lift(high)} />
      {shown.flatMap(entry => entry.rays.map((ray, index) => {
        const t = Math.min(high / ray[0], high / ray[1]);
        return <line key={`${entry.key}-${index}`} className="nm-ray" stroke={entry.colour} strokeWidth="3"
          strokeDasharray={entry.key === 'second' ? '7 4' : undefined}
          x1={place(0)} y1={lift(0)} x2={place(ray[0] * t)} y2={lift(ray[1] * t)} />;
      }))}
      {points.map((point, index) => {
        // Keep the label inside the drawing: past the middle it goes to the left
        // of its marker instead of running off the right edge.
        const toLeft = point[0] > 2.2;
        return <g key={index}>
          <circle className="nm-observation" cx={place(point[0])} cy={lift(point[1])} r="6" />
          <text x={place(point[0]) + (toLeft ? -12 : 12)} y={lift(point[1]) - 10} textAnchor={toLeft ? 'end' : 'start'}>({point.join(', ')})</text>
        </g>;
      })}
    </svg>
    <p className="nm-caption">
      Horizontal axis: feature 1, from 0 to 3.5. Vertical axis: feature 2, on the same scale. The axis names sit here rather than inside the
      drawing, where the first dictionary’s rays run along both axes.
    </p>
    <ul className="nm-legend">
      <li><span className="nm-swatch" style={{ background: '#e7b94a' }} /> First dictionary: rays along the two axes, solid gold. Its cone is the whole nonnegative quadrant here, so the gold rays run along the axes themselves.</li>
      <li><span className="nm-swatch" style={{ background: '#8eb9a5' }} /> Second dictionary: rays through (1.25, .25) and (.25, 1.25), dashed green, counter-hatched region strictly inside the first</li>
    </ul>
    <p className="nm-caption">
      Each set of rays contains all three observations. The second dictionary's components both contribute to every feature, so they are not the first
      components reordered or rescaled.
    </p>
    <div className="nm-matrices">
      <Matrix title="W₁" shape="3 × 2" rows={fixtures.W1} />
      <Matrix title="H₁" shape="2 × 3" rows={fixtures.H1} />
      <Matrix title="W₁H₁" shape="3 × 3" rows={reconstruct(fixtures.W1, fixtures.H1)} note="Exactly X." />
    </div>
    <div className="nm-matrices">
      <Matrix title="W₂" shape="3 × 2" rows={fixtures.W2} />
      <Matrix title="H₂" shape="2 × 3" rows={fixtures.H2} />
      <Matrix title="W₂H₂" shape="3 × 3" rows={reconstruct(fixtures.W2, fixtures.H2)} note="Also exactly X." />
    </div>
    <h4>Scale is a separate, simpler ambiguity</h4>
    <div className="nm-matrices">
      <Matrix title="H₁ row 1 doubled" shape="2 × 3" rows={doubled.H} marked={[[0, 0], [0, 1], [0, 2]]} />
      <Matrix title="W₁ column 1 halved" shape="3 × 2" rows={doubled.W} marked={[[0, 0], [1, 0], [2, 0]]} />
      <Matrix title="Their product" shape="3 × 3" rows={reconstruct(doubled.W, doubled.H)} note="Unchanged: every contribution is the same." />
    </div>
    <p className="nm-caption">
      Permutation and scale leave the product alone, so a larger raw activation in one fit says nothing about a physical component until identities
      are aligned and a scale convention is declared. The two dictionaries above differ by more than that.
    </p>
  </Figure>;
}

/** §5 · F5. Fitting learns a dictionary; transform learns new amounts. */
export function FitTransformFigure() {
  const { splitSizes, fit } = NMF_DIGITS;
  const lane = (name, rows, updatesH) => <div className="nm-panel" key={name}>
    <h4>{name} · {rows} images</h4>
    <ol className="nm-flow-steps">
      <li>{rows} × 64 raw block counts, values 0…16</li>
      <li>÷ 16 — a fixed known maximum, not a learned scaler</li>
      <li>{updatesH
        ? <><strong>fit</strong>: update both W<sub>train</sub> ({rows} × {fit.k}) <em>and</em> H ({fit.k} × 64)</>
        : <><strong>transform</strong>: H is locked; solve for a new W ({rows} × {fit.k}) only</>}</li>
      <li>reconstruction = W H, {rows} × 64</li>
    </ol>
    <p>{updatesH
      ? 'The only lane with an update arrow into H.'
      : 'A small nonnegative optimization per row, not multiplication by Hᵀ.'}</p>
  </div>;
  const rows = [
    { key: 'train', label: 'Training', step: 'fit', y: 78 },
    { key: 'validation', label: 'Validation', step: 'transform', y: 140 },
    { key: 'test', label: 'Reserved test', step: 'transform', y: 202 },
  ];
  const boxHeight = 32;
  const mid = y => y + boxHeight / 2;
  return <Figure title="One dictionary, three lanes: only training changes H."
    caption="H is a single shared object, not a copy refitted per lane. The ID and digit columns branch into labels for splitting and diagnostics; they never become factorization features.">
    <svg viewBox="0 0 340 248" role="img"
      aria-label={`Three data lanes and one shared dictionary. Each lane runs: raw block counts, then a divide-by-16 operator, then a step, then the activations W. The training lane's step is fit, and a single arrow runs from it into the shared H node, so training is the only lane that writes the dictionary. The validation and reserved-test lanes' step is transform, and an arrow runs from H into each of them, so they read the same dictionary and write only their own W. H is one node, drawn once, with ${splitSizes.train} training images, ${splitSizes.validation} validation images and ${splitSizes.test} reserved test images.`}>
      <defs>
        <marker id="nm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,1 L9,5 L0,9 z" fill="#8eb9a5" />
        </marker>
        <marker id="nm-arrow-gold" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,1 L9,5 L0,9 z" fill="#e7b94a" />
        </marker>
      </defs>
      <rect className="nm-shared" x="110" y="6" width="104" height="34" rx="4" />
      <text x="162" y="28" textAnchor="middle">H · {fit.k} × 64</text>
      {rows.map(row => <g key={row.key}>
        <text x="6" y={row.y - 7}>{row.label}</text>
        <rect className="nm-lane" x="10" y={row.y} width="44" height={boxHeight} rx="3" />
        <text x="32" y={mid(row.y) + 5} textAnchor="middle">X</text>
        <rect className="nm-lane" x="64" y={row.y} width="40" height={boxHeight} rx="3" />
        <text x="84" y={mid(row.y) + 5} textAnchor="middle">÷16</text>
        <rect className="nm-lane" x="114" y={row.y} width="88" height={boxHeight} rx="3" />
        <text x="158" y={mid(row.y) + 5} textAnchor="middle">{row.step}</text>
        <rect className="nm-lane" x="212" y={row.y} width="48" height={boxHeight} rx="3" />
        <text x="236" y={mid(row.y) + 5} textAnchor="middle">W</text>
        <line className="nm-flow" x1="54" x2="62" y1={mid(row.y)} y2={mid(row.y)} markerEnd="url(#nm-arrow)" />
        <line className="nm-flow" x1="104" x2="112" y1={mid(row.y)} y2={mid(row.y)} markerEnd="url(#nm-arrow)" />
        <line className="nm-flow" x1="202" x2="210" y1={mid(row.y)} y2={mid(row.y)} markerEnd="url(#nm-arrow)" />
      </g>)}
      {/* The one update edge: training writes H. */}
      <line className="nm-flow is-update" x1="158" y1="78" x2="158" y2="42" markerEnd="url(#nm-arrow-gold)" />
      {/* Two read edges: the other lanes take the same H and never write it. */}
      <path className="nm-flow" d="M214,20 L284,20 L284,128 L158,128 L158,138" markerEnd="url(#nm-arrow)" />
      <path className="nm-flow" d="M214,32 L300,32 L300,190 L158,190 L158,200" markerEnd="url(#nm-arrow)" />
    </svg>
    <ul className="nm-legend">
      <li><span className="nm-swatch" style={{ background: '#e7b94a' }} /><span>One gold arrow <strong>into</strong> H: the training lane is the only one that updates the dictionary.</span></li>
      <li><span className="nm-swatch" style={{ background: '#8eb9a5' }} /><span>Two green arrows <strong>out of</strong> H: validation and test read that same locked dictionary and write only their own W.</span></li>
    </ul>
    <div className="nm-panels">
      {lane('Training', splitSizes.train, true)}
      {lane('Validation', splitSizes.validation, false)}
      {lane('Reserved test', splitSizes.test, false)}
    </div>
    <div className="nm-panel" style={{ marginTop: '1rem' }}>
      <h4>The shared object</h4>
      <p>
        H is the {fit.k} × 64 dictionary produced by the training lane. Validation and test both read that same H and never write to it. Reconstruction
        error in those lanes therefore measures a fixed dictionary on images it has not seen.
      </p>
    </div>
    <Table caption="What each column of the CSV is used for"
      headings={['column', 'used as', 'enters the factorization?']}
      rows={[
        ['source_row', 'a stable identity for citing an image', 'no'],
        ['digit', 'stratification of the split and a diagnostic label', 'no'],
        ['pixel_0 … pixel_63', 'the 64 nonnegative features, divided by 16', 'yes'],
      ]} />
    <p className="nm-caption">
      For a purely descriptive analysis of one fixed collection, fitting the whole collection answers a different question and can be stated as such.
      This diagram describes evaluating a dictionary on withheld images.
    </p>
  </Figure>;
}

/** §5 · F6. Every recorded candidate, with no manufactured elbow. */
const seriesStyles = {
  'train-7': { colour: '#e7b94a', dash: '', label: 'training, seed 7' },
  'validation-7': { colour: '#e7b94a', dash: '5 3', label: 'validation, seed 7' },
  'train-19': { colour: '#8eb9a5', dash: '', label: 'training, seed 19' },
  'validation-19': { colour: '#8eb9a5', dash: '5 3', label: 'validation, seed 19' },
};
export function CandidateFigure() {
  const { runs, baselines } = NMF_DIGITS;
  const ks = [1, 4, 8, 16];
  const series = Object.keys(seriesStyles).map(key => {
    const [kind, seed] = key.split('-');
    return { key, ...seriesStyles[key], points: ks.map(k => {
      const run = runs.find(item => item.k === k && item.seed === Number(seed));
      return { k, value: kind === 'train' ? run.trainMse : run.validationMse };
    }) };
  });
  const best = runs.reduce((low, run) => (run.validationMse < low.validationMse ? run : low));
  const declared = runs.find(run => run.k === 8 && run.seed === 19);
  // A point outside the drawn range is left out of the line entirely. Clamping it
  // to the axis would draw a flat segment at a value no run produced.
  const segments = (item, range) => {
    const runs_ = [];
    let current = [];
    item.points.forEach((point, index) => {
      if (point.value >= range[0] && point.value <= range[1]) current.push([index, point.value]);
      else if (current.length) { runs_.push(current); current = []; }
    });
    if (current.length) runs_.push(current);
    return runs_.filter(run => run.length > 1);
  };
  const panel = (title, range, yTicks, describeSuffix) => (
    <div className="nm-panel" key={title}>
      <h4>{title}</h4>
      <Plot width={300} height={230} padding={{ left: 54, right: 14, top: 16, bottom: 40 }}
        domain={[-0.35, 3.35]} range={range} yTicks={yTicks} xTicks={[0, 1, 2, 3]}
        formatX={index => `k=${ks[index]}`} formatY={value => value.toFixed(3)}
        describe={`Mean squared reconstruction error against component count. ${series.map(item => `${item.label}: ${item.points.map(point => `k equals ${point.k} gives ${round(point.value, 6)}`).join(', ')}`).join('. ')}. ${describeSuffix} The ringed point is the best validation result seen, k equals ${best.k} with seed ${best.seed} at ${round(best.validationMse, 6)}.`}>
        {(scaleX, scaleY) => <>
          {series.map(item => <g key={item.key}>
            {segments(item, range).map((run, index) => (
              <polyline key={index} className="nm-curve" stroke={item.colour} strokeDasharray={item.dash || undefined}
                points={run.map(([position, value]) => `${scaleX(position)},${scaleY(value)}`).join(' ')} />
            ))}
            {item.points.map((point, index) => (point.value >= range[0] && point.value <= range[1]
              ? <circle key={point.k} cx={scaleX(index)} cy={scaleY(point.value)} r="4" fill={item.colour}
                fillOpacity={item.dash ? 1 : 0.35} stroke={item.colour} strokeWidth="1.4" />
              : null))}
          </g>)}
          {best.validationMse >= range[0] && best.validationMse <= range[1]
            && <circle className="nm-mark is-hollow" cx={scaleX(ks.indexOf(best.k))} cy={scaleY(best.validationMse)} r="9" />}
        </>}
      </Plot>
      <p>
        {describeSuffix} The ringed point is k = {best.k} with seed {best.seed}, the best validation result among the eight.
      </p>
    </div>
  );
  return <Figure title="Every recorded candidate, on one axis that keeps the one-component baseline and one that magnifies the rest."
    caption="Solid lines with faded markers are training error; dashed lines with filled markers are validation error. Gold is seed 7 and green is seed 19. Both validation fits improve over this inspected component range; the two initializations disagree more at larger k. No elbow is drawn and no true component count is claimed.">
    <div className="nm-panels">
      {panel('Full range, 0 to 0.08', [0, 0.08], [0, 0.02, 0.04, 0.06, 0.08], 'This axis keeps the one-component baseline visible.')}
      {panel('Magnified, 0.010 to 0.030', [0.01, 0.03], [0.01, 0.015, 0.02, 0.025, 0.03], 'The one- and four-component runs fall outside this window, so they are neither drawn nor connected to what is; the exact table below keeps them.')}
    </div>
    <ul className="nm-legend">
      {series.map(item => <li key={item.key}>
        <span className="nm-swatch" style={{ background: item.colour, opacity: item.dash ? 1 : 0.45 }} /> {item.label} ({item.dash ? 'dashed' : 'solid'})
      </li>)}
    </ul>
    <p className="nm-readout">
      The best validation result among these eight candidates is k = {best.k} with seed {best.seed}, at {fixed(best.validationMse, 6)}. The
      eight-component, seed-{declared.seed} run at {fixed(declared.validationMse, 6)} was declared in advance for readable image panels; it is not
      the winner of this grid and is not described as one.
    </p>
    <Table caption="Every candidate fit, in mean squared error on features scaled to [0, 1]"
      headings={['components', 'seed', 'iterations', 'training MSE', 'validation MSE']}
      rows={runs.map(run => [run.k, run.seed, run.iterations, fixed(run.trainMse, 6), fixed(run.validationMse, 6)])}
      rowClass={index => (runs[index].k === 16 && runs[index].seed === 7 ? 'is-selected' : undefined)} />
    <h4>The separately declared eight-component comparison, on the reserved images</h4>
    <p className="nm-caption">Bar length is mean squared error relative to the training-mean baseline, so a <strong>shorter bar is the better reconstruction</strong>.</p>
    <div className="nm-strip-row">
      {[['training-mean image', baselines.meanTestMse], ['PCA, 8 components', baselines.pcaTestMse], ['NMF, 8 components', baselines.nmfTestMse]].map(([name, value]) => (
        <div key={name} style={{ display: 'contents' }}>
          <span className="nm-strip-label">{name}</span>
          <span style={{ display: 'block', background: '#1d2622', borderRadius: '2px', height: '18px' }}>
            <span style={{ display: 'block', height: '100%', borderRadius: '2px', width: `${100 * value / baselines.meanTestMse}%`, background: name.startsWith('PCA') ? '#91aecf' : name.startsWith('NMF') ? '#e7b94a' : '#5c6b64' }} />
          </span>
        </div>
      ))}
    </div>
    <Table caption="Reserved-image mean squared error, on scaled pixel units, for the predeclared eight-component panel"
      headings={['reconstruction', 'test MSE']}
      rows={[
        ['training-mean image', fixed(baselines.meanTestMse, 6)],
        ['PCA with 8 components and its learned mean', fixed(baselines.pcaTestMse, 6)],
        ['NMF with 8 components', fixed(baselines.nmfTestMse, 6)],
      ]} rowClass={index => (index === 1 ? 'is-selected' : undefined)} />
    <p className="nm-caption">
      PCA wins this reconstruction comparison. It is allowed signed, centred patterns and uses an additional mean image; NMF supplies the additive
      constraint we wanted to inspect. This compares representation choices, not equal storage bits or digit classification, and the example has not
      been tuned until NMF wins.
    </p>
  </Figure>;
}

/** §5 · F7. The actual factors and one actual reconstructed image. */
const galleryModes = [
  { key: 'actual', label: 'Actual pattern values' },
  { key: 'contribution', label: 'Contribution to this image' },
  { key: 'normalized', label: 'Patterns normalized to sum to 1' },
];
export function DictionaryFigure() {
  const [mode, setMode] = useState('actual');
  const { dictionary, activations, test, scale, side } = NMF_DIGITS;
  const observed = test[0].pixels.map(value => value / scale);
  const report = maskedReport(observed, activations[0], dictionary, activations[0].map(() => true));
  const patternSums = dictionary.map(row => row.reduce((sum, value) => sum + value, 0));
  const normalizedPatterns = normalizePatterns([activations[0]], dictionary).H;
  const contributions = contributionRows(activations, dictionary, 0);
  const intensityMax = Math.max(1, Math.max(...report.full), Math.max(...observed));
  const residualExtent = Math.max(...report.fullResidual.map(Math.abs));
  const ordered = [...dictionary.keys()].sort((a, b) => report.contributionTotals[b] - report.contributionTotals[a]);
  // Each display mode has its own honest maximum. A raw pattern entry is not an
  // image intensity, so forcing it onto the image scale would saturate it.
  const gallery = {
    actual: { values: dictionary, max: Math.max(...dictionary.flat()) },
    contribution: { values: contributions, max: intensityMax },
    normalized: { values: normalizedPatterns, max: Math.max(...normalizedPatterns.flat()) },
  }[mode];
  return <Figure title={`The eight fitted patterns, and the first reserved image they rebuild (source row ${test[0].sourceRow}).`}
    caption={`The first two panels above, observed and reconstructed, share one intensity scale: 0 to ${round(intensityMax, 4)}, the larger of 1 and the largest value either of them draws. Nothing is clipped. The third panel, the signed residual, has its own symmetric scale from −${round(residualExtent, 4)} to +${round(residualExtent, 4)}, neutral grey at exactly 0. The eight component panels carry a third scale, stated with them. Each patch is 8 × 8 block counts at a true square aspect ratio.`}>
    <div className="nm-image-grid">
      <ImagePanel title={`Observed, source row ${test[0].sourceRow}`} values={observed} side={side} maximum={intensityMax}
        note={`Recorded digit ${test[0].digit}, a diagnostic label that never entered the fit. Maximum ${round(Math.max(...observed), 4)}.`} />
      <ImagePanel title="Reconstructed, W H" values={report.full} side={side} maximum={intensityMax}
        note={`Maximum ${round(Math.max(...report.full), 4)}, which is above 1: the reconstruction is not constrained to the data's range.`} />
      <ImagePanel title="Signed residual" values={report.fullResidual} side={side} tone="signed"
        note={`Observed minus reconstructed. Mean squared residual ${fixed(report.fullMse, 9)}.`} />
    </div>
    <ScaleKey maximum={intensityMax} note="Used by the observed and reconstructed panels." />
    <ScaleKey tone="signed" maximum={residualExtent} note="Used by the residual panel only." />
    <h4>The eight components, three ways to look at them</h4>
    <ul className="nm-toggles">
      {galleryModes.map(entry => <li key={entry.key}>
        <button type="button" aria-pressed={mode === entry.key} onClick={() => setMode(entry.key)}>{entry.label}</button>
      </li>)}
    </ul>
    <ScaleKey maximum={gallery.max} note={{
      actual: 'The dictionary rows as fitted. These are pattern weights, not image intensities, so they carry their own maximum rather than the image scale.',
      contribution: `The weighted contribution W[i, r] × H[r, ·] for this image, on the same intensity scale as the panels above. These eight add up to the reconstruction exactly.`,
      normalized: 'Each pattern divided by its own total. A display convenience only: every number elsewhere on this page still uses the actual values.',
    }[mode]} />
    <div className="nm-image-grid">
      {dictionary.map((_, component) => <ImagePanel key={component}
        title={`Component ${component + 1}`}
        values={gallery.values[component]} side={side} maximum={gallery.max}
        note={{
          actual: `Total mass ${fixed(patternSums[component], 6)} · activation on this image ${fixed(activations[0][component], 6)}`,
          contribution: `Contribution total ${fixed(report.contributionTotals[component], 6)} = activation ${fixed(activations[0][component], 6)} × mass ${fixed(patternSums[component], 6)}`,
          normalized: `Divided by its own total ${fixed(patternSums[component], 6)}`,
        }[mode]} />)}
    </div>
    <Table caption="Each component's contribution to this image, ordered by contribution total rather than by raw coefficient"
      headings={['component', 'activation W[i, r]', 'pattern mass Σ H[r, ·]', 'contribution total', 'row MSE if removed']}
      rows={ordered.map(component => {
        const mask = activations[0].map((_, index) => index !== component);
        return [
          `Component ${component + 1}`, fixed(activations[0][component], 6), fixed(patternSums[component], 6),
          fixed(report.contributionTotals[component], 6),
          fixed(maskedReport(observed, activations[0], dictionary, mask).maskedMse, 9),
        ];
      })} />
    <ImageValues summary={`All 64 values for the observed, reconstructed and residual panels of source row ${test[0].sourceRow}`}
      side={side} panels={[
        { title: 'observed', values: observed },
        { title: 'reconstructed', values: report.full },
        { title: 'signed residual', values: report.fullResidual },
      ]} />
    <p className="nm-caption">
      Removing a component changes the reconstruction by exactly its contribution image. You can now say whether a pattern is concentrated around a
      stroke, spreads over several regions, or overlaps another pattern. That is stronger evidence than naming every component a digit part in advance.
      Component 8 has an activation of exactly {fixed(activations[0][7], 6)} here, so it contributes nothing to this particular image.
    </p>
  </Figure>;
}

/** §7 · F8. A positive rectangle cannot cover crossed zeros. */
export function SupportFigure() {
  const [pair, setPair] = useState(0);
  const pairs = supportPairs(fixtures.support, fixtures.supportMarks);
  const names = fixtures.supportNames;
  const rank = integerRank(fixtures.support);
  const minor = nonsingularMinor(fixtures.support, 3);
  const active = pairs[pair];
  const rectRows = [active.cells[0][0], active.cells[1][0]];
  const rectColumns = [active.cells[0][1], active.cells[1][1]];
  const inRectangle = (r, c) => rectRows.includes(r) && rectColumns.includes(c);
  return <Figure title="Four marked positive cells, and why no two of them share one positive rectangle."
    caption="A nonnegative rank-one contribution has rectangular positive support: if it is positive at two cells, it is positive at both crossed corners too. Another contribution cannot cancel that, because nothing here is negative.">
    <ul className="nm-toggles">
      {pairs.map((item, index) => <li key={index}>
        <button type="button" aria-pressed={pair === index} onClick={() => setPair(index)}>
          {names[item.a]} and {names[item.b]}
        </button>
      </li>)}
    </ul>
    <div className="nm-matrix" style={{ gridTemplateColumns: 'repeat(4, auto)' }} role="img"
      aria-label={`The 4 by 4 binary matrix S, rows ${fixtures.support.map(row => row.join(' ')).join('; ')}. Marked positive cells are ${fixtures.supportMarks.map((cell, index) => `${names[index]} at row ${cell[0] + 1}, column ${cell[1] + 1}`).join(', ')}. The selected pair is ${names[active.a]} and ${names[active.b]}; the rectangle spanning them crosses a zero at ${active.crossed.map(cell => `row ${cell[0] + 1}, column ${cell[1] + 1}`).join(' and ')}.`}>
      {fixtures.support.flatMap((row, r) => row.map((value, c) => {
        const mark = fixtures.supportMarks.findIndex(([mr, mc]) => mr === r && mc === c);
        const crossed = active.crossed.some(([cr, cc]) => cr === r && cc === c);
        const selected = active.cells.some(([sr, sc]) => sr === r && sc === c);
        return <span key={`${r}-${c}`}
          className={`nm-matrix-cell${selected ? ' is-marked' : crossed ? ' is-related' : ''}`}
          style={crossed ? { outline: '2px dashed #da9c86', color: '#f0b9a5' } : inRectangle(r, c) ? { outline: '1px dotted #8a9590' } : undefined}>
          {value}{mark >= 0 ? ` ${names[mark]}` : ''}
        </span>;
      }))}
    </div>
    <p className="nm-readout" aria-live="polite">
      {names[active.a]} sits at row {active.cells[0][0] + 1}, column {active.cells[0][1] + 1}; {names[active.b]} at row {active.cells[1][0] + 1},
      column {active.cells[1][1] + 1}. The rectangle spanning them also contains
      {' '}{active.crossed.map(cell => `row ${cell[0] + 1}, column ${cell[1] + 1}`).join(' and ')}, where S is 0. One nonnegative rank-one contribution
      cannot be positive at both marks and zero there, so it cannot cover both.
    </p>
    <Table caption="All six pairs of marked cells, each with a crossed zero"
      headings={['pair', 'first cell', 'second cell', 'crossed zero cells']}
      rows={pairs.map(item => [
        `${names[item.a]} and ${names[item.b]}`,
        `(${item.cells[0][0] + 1}, ${item.cells[0][1] + 1})`,
        `(${item.cells[1][0] + 1}, ${item.cells[1][1] + 1})`,
        item.crossed.map(cell => `(${cell[0] + 1}, ${cell[1] + 1})`).join(', '),
      ])}
      rowClass={index => (index === pair ? 'is-selected' : undefined)} />
    <h4>The ordinary rank, shown rather than asserted</h4>
    <div className="nm-matrices">
      <Matrix title="A nonsingular 3 × 3 minor"
        shape={`rows ${minor.rows.map(row => row + 1).join(', ')} × columns ${minor.columns.map(column => column + 1).join(', ')}`}
        rows={minor.minor}
        note={`Determinant ${minor.determinant}, computed exactly in integers, so these three rows are independent and the rank is at least 3.`} />
      <Matrix title="Row 1 + row 3" shape="1 × 4" rows={[fixtures.support[0].map((value, index) => value + fixtures.support[2][index])]}
        note="Equal to row 2 + row 4, so the four rows are dependent and the rank is at most 3." />
      <Matrix title="Row 2 + row 4" shape="1 × 4" rows={[fixtures.support[1].map((value, index) => value + fixtures.support[3][index])]}
        note={`Ordinary rank is therefore exactly ${rank}.`} />
    </div>
    <p className="nm-caption">
      So at least four rank-one contributions are necessary, and W = I₄ with H = S shows four suffice: the nonnegative rank is 4 while the ordinary
      rank is {rank}. This is exact combinatorial reasoning about one matrix, not an optimization run or a complexity slogan.
    </p>
  </Figure>;
}
