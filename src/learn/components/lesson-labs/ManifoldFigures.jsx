import { useId, useState } from 'react';
import { U_POINTS, gaussianRow, tsneObjective, umapConnection } from '../../data/manifold-models';
import { MANIFOLD_DIGITS, MANIFOLD_FIXTURES } from '../../data/manifold-data';
import { DigitLegend, DigitScatter, DigitTile } from './ManifoldShared';
import './manifold-figures.css';

const names = ['A', 'B', 'C', 'D'];
const format = (value, places = 6) => Number(value.toFixed(places)).toString().replace('-', '−');
const signed = (value, places = 9) => `${value >= 0 ? '+' : '−'}${format(Math.abs(value), places)}`;

function TeachingFigure({ number, title, category = 'Exact constructed calculation', children }) {
  return <figure className={`mnfig mnfig-${number.toLowerCase()}`} data-manifold-figure={number}>
    <figcaption><span className="mnfig-evidence">{number} · {category}</span><strong>{title}</strong></figcaption>
    {children}
  </figure>;
}

function DataTable({ caption, headings, rows, className = '' }) {
  return <div className={`mnfig-table-wrap ${className}`}>
    <table>
      <caption>{caption}</caption>
      <thead><tr>{headings.map(heading => <th scope="col" key={heading}>{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, column) => column === 0
        ? <th key={column} scope="row">{cell}</th>
        : <td key={column} data-label={headings[column]}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

function ArrowDefinitions({ id, color = '#e7b94a' }) {
  return <defs><marker id={id} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L8,4 L0,8 Z" fill={color} />
  </marker></defs>;
}

/** F1 · Same seven identities, ambient distance and cumulative path distance. */
export function ManifoldRouteFigure() {
  const horizontal = value => 70 + 80 * value;
  const vertical = value => 218 - 80 * value;
  return <TeachingFigure number="F1" title="Two units across the opening; six along the U.">
    <div className="mnfig-panels">
      <section className="mnfig-panel">
        <h4>Input coordinates</h4>
        <svg data-manifold-geometry="true" className="mnfig-square" viewBox="0 0 300 278" role="img" aria-label="A at 0,0; B at 0,1; C at 0,2; D at 1,2; E at 2,2; F at 2,1; G at 2,0. Six solid edges each have length 1. Dashed A to G has length 2. Both axes have the same scale.">
          {[0, 1, 2].map(tick => <g key={tick}>
            <line className="mnfig-grid" x1="46" x2="258" y1={vertical(tick)} y2={vertical(tick)} />
            <line className="mnfig-grid" x1={horizontal(tick)} x2={horizontal(tick)} y1="36" y2="244" />
            <text className="mnfig-tick" x="28" y={vertical(tick) + 4} textAnchor="middle">{tick}</text>
            <text className="mnfig-tick" x={horizontal(tick)} y="262" textAnchor="middle">{tick}</text>
          </g>)}
          <polyline className="mnfig-path" points={U_POINTS.map(point => `${horizontal(point.x)},${vertical(point.y)}`).join(' ')} />
          <line className="mnfig-shortcut" x1={horizontal(0)} x2={horizontal(2)} y1={vertical(0)} y2={vertical(0)} />
          {[[85, 181], [85, 101], [110, 80], [190, 80], [215, 101], [215, 181]].map(([x, y], index) => <text key={index} x={x} y={y} textAnchor="middle">1</text>)}
          <text x="150" y="205" textAnchor="middle">2</text>
          {U_POINTS.map((point, index) => {
            const offsets = [[-14, 18], [-15, 4], [-14, -12], [0, -12], [14, -12], [15, 4], [14, 18]];
            return <g key={point.id}>
              <circle className="mnfig-point" cx={horizontal(point.x)} cy={vertical(point.y)} r="4.5" />
              <text x={horizontal(point.x) + offsets[index][0]} y={vertical(point.y) + offsets[index][1]} textAnchor="middle">{point.id}</text>
            </g>;
          })}
        </svg>
        <p className="mnfig-note">Input coordinate units; equal scale on x and y. Solid: six unit steps. Dashed: the direct crossing.</p>
      </section>
      <section className="mnfig-panel mnfig-route-strip">
        <h4>Straightened route distance</h4>
        <svg data-manifold-geometry="true" viewBox="0 0 300 106" role="img" aria-label="Cumulative path distances are A 0, B 1, C 2, D 3, E 4, F 5, G 6. A to G takes 6 units.">
          <line className="mnfig-path" x1="30" x2="270" y1="50" y2="50" />
          {U_POINTS.map((point, index) => <g key={point.id}>
            <circle className="mnfig-point" cx={30 + index * 40} cy="50" r="4.5" />
            <text x={30 + index * 40} y="28" textAnchor="middle">{point.id}</text>
            <text className="mnfig-tick" x={30 + index * 40} y="76" textAnchor="middle">{index}</text>
          </g>)}
        </svg>
        <p><strong>A → B → C → D → E → F → G</strong><br />1 + 1 + 1 + 1 + 1 + 1 = <strong>6</strong>.</p>
        <p>A → G directly: √((2 − 0)² + (0 − 0)²) = <strong>2</strong>.</p>
        <p className="mnfig-note">The strip represents the supplied path lengths. It is not a fitted t-SNE or PCA map.</p>
      </section>
    </div>
    <p>Six unit steps along the U connect endpoints only two units apart across it.</p>
  </TeachingFigure>;
}

function NeighborLine({ positions, edges, label, map = false }) {
  const arrowId = useId().replace(/:/g, '');
  const scale = value => 30 + 240 * value / Math.max(...positions);
  return <section className="mnfig-panel">
    <h4>{label}</h4>
    <svg data-manifold-geometry="true" viewBox="0 0 300 186" role="img" aria-label={`${label}. Positions ${positions.map((value, index) => `${names[index]} ${value}`).join(', ')}. ${edges.map(edge => `${names[edge.from]} ${edge.both ? 'and' : 'to'} ${names[edge.to]}${edge.both ? ' select each other' : ''}`).join('; ')}.`}>
      <ArrowDefinitions id={arrowId} color={map ? '#b6a0cf' : '#8eb9a5'} />
      <line className="mnfig-axis" x1="20" x2="280" y1="120" y2="120" />
      {edges.map((edge, index) => {
        const from = scale(positions[edge.from]);
        const to = scale(positions[edge.to]);
        return <path key={index} className={`mnfig-neighbor-edge${map ? ' is-map' : ''}`}
          d={`M${from},114 Q${(from + to) / 2},${[58, 18, -8][index]} ${to},114`}
          markerStart={edge.both ? `url(#${arrowId})` : undefined} markerEnd={`url(#${arrowId})`} />;
      })}
      {positions.map((position, index) => <g key={index}>
        <circle className="mnfig-point" cx={scale(position)} cy="120" r="4" />
        <text x={scale(position)} y="145" textAnchor="middle">{names[index]}</text>
        <text className="mnfig-tick" x={scale(position)} y="169" textAnchor="middle">{position}</text>
      </g>)}
    </svg>
    <p className="mnfig-note">{map ? 'Dashed arrows: map-only choices. Arbitrary map coordinates.' : 'Solid arrows: input-only choices. Input coordinate units.'} Arrowheads identify who chooses whom.</p>
  </section>;
}

/** F2 · Two different directed edge sets, including grouped reciprocal edges. */
export function ManifoldNeighborFigure() {
  const input = MANIFOLD_FIXTURES.false_neighbors.X.map(row => row[0]);
  const proposed = MANIFOLD_FIXTURES.false_neighbors.Y.map(row => row[0]);
  return <TeachingFigure number="F2" title="A false neighbor and a missing neighbor are two sides of a changed choice.">
    <p><strong>k = 1.</strong> Each arrow points from an observation to its nearest other observation.</p>
    <div className="mnfig-panels">
      <NeighborLine positions={input} label="Original input line" edges={[{ from: 0, to: 1, both: true }, { from: 2, to: 1 }, { from: 3, to: 2 }]} />
      <NeighborLine positions={proposed} label="Constructed proposed map" map edges={[{ from: 0, to: 2, both: true }, { from: 1, to: 2 }, { from: 3, to: 1 }]} />
    </div>
    <DataTable caption="Identity audit: no original nearest-neighbor choice survives"
      headings={['Query', 'Input only', 'Map only', 'False input rank']}
      rows={names.map((name, index) => [name, ['B', 'A', 'B', 'C'][index], ['C', 'C', 'A', 'B'][index], '2'])} />
    <div className="mnfig-summary"><p><strong>R₁ = 0 / 4 = 0</strong><br />All four choices changed.</p><p><strong>T₁ = 1 − (1/8) × 4 = 0.5</strong><br />Each rank penalty is 2 − 1 = 1.</p></div>
    <p>All four nearest-neighbor choices changed; each replacement was second in the input ranking.</p>
  </TeachingFigure>;
}

function ProbabilityBars({ title, values, description }) {
  return <section className="mnfig-panel">
    <h4>{title}</h4>
    <p className="mnfig-note">{description}</p>
    <div className="mnfig-bars">
      {values.map((value, index) => <div className="mnfig-bar-row" key={index}>
        <div className="mnfig-bar-label"><strong>{['B', 'C', 'D'][index]}</strong><span>{format(value)}</span></div>
        <div className="mnfig-bar-track" aria-hidden="true"><span style={{ width: `${100 * value}%` }} /></div>
      </div>)}
      <div className="mnfig-scale" aria-hidden="true"><span>0</span><span>0.5</span><span>1</span></div>
    </div>
  </section>;
}

/** F3 · Actual unnormalized weights and probabilities, both on honest 0–1 scales. */
export function ManifoldProbabilityFigure() {
  const distribution = gaussianRow([1, 2, 3], 1);
  const weights = distribution.weights;
  const weightSum = weights.reduce((sum, weight) => sum + weight, 0);
  return <TeachingFigure number="F3" title="Normalize the three Gaussian weights into one probability row.">
    <h4>Distance from i, with bandwidth σ = 1</h4>
    <svg data-manifold-geometry="true" className="mnfig-compact" viewBox="0 0 300 94" role="img" aria-label="Source i at distance 0; candidates B, C and D are at distances 1, 2 and 3 from i. This is one distance row, not a complete point configuration.">
      <line className="mnfig-axis" x1="24" x2="276" y1="42" y2="42" />
      {['i', 'B', 'C', 'D'].map((name, index) => <g key={name}>
        <circle className={index ? 'mnfig-point' : 'mnfig-source'} cx={30 + index * 80} cy="42" r="4.5" />
        <text x={30 + index * 80} y="20" textAnchor="middle">{name}</text>
        <text className="mnfig-tick" x={30 + index * 80} y="69" textAnchor="middle">{index}</text>
      </g>)}
    </svg>
    <div className="mnfig-panels">
      <ProbabilityBars title="1. Unnormalized weights" values={weights} description="w = exp(−d²/2), with the same 0–1 scale below." />
      <ProbabilityBars title="2. Normalized probabilities" values={distribution.probabilities} description={`p = w / ${format(weightSum)}. Divide every weight by their shared sum.`} />
    </div>
    <p className="mnfig-note">D receives {format(distribution.probabilities[2])}; its short bar is drawn at that actual value. The unrounded probabilities sum to 1.</p>
    <div className="mnfig-summary"><p><strong>Entropy: {format(distribution.entropyBits)} bits</strong><br />−Σ p log₂ p</p><p><strong>Perplexity: {format(distribution.perplexity)}</strong><br />2 raised to the entropy</p></div>
  </TeachingFigure>;
}

function SignedContribution({ value, title, children, total = false }) {
  const arrowId = useId().replace(/:/g, '');
  const scale = displacement => 26 + 246 * (displacement + 0.025) / 0.11;
  return <section className={`mnfig-force-row${total ? ' is-total' : ''}`}>
    <h4>{title} <span>{signed(value)}</span></h4>
    {children}
    <svg data-manifold-geometry="true" viewBox="0 0 300 76" role="img" aria-label={`${title}: signed negative-gradient contribution ${value}. Shared axis from minus 0.025 to plus 0.085 in map-coordinate units per unit step size.`}>
      <ArrowDefinitions id={arrowId} color={value < 0 ? '#caa0ad' : '#e7b94a'} />
      <line className="mnfig-axis" x1="26" x2="272" y1="26" y2="26" />
      <line className="mnfig-zero" x1={scale(0)} x2={scale(0)} y1="12" y2="40" />
      <line className={value < 0 ? 'mnfig-force is-negative' : 'mnfig-force'} x1={scale(0)} x2={scale(value)} y1="26" y2="26" markerEnd={`url(#${arrowId})`} />
      {[-0.02, 0, 0.04, 0.08].map(tick => <g key={tick}>
        <line className="mnfig-axis" x1={scale(tick)} x2={scale(tick)} y1="42" y2="46" />
        <text className="mnfig-tick" x={scale(tick)} y="65" textAnchor="middle">{tick === 0 ? '0' : signed(tick, 2)}</text>
      </g>)}
    </svg>
  </section>;
}

/** F4 · Contributions are negative gradients; the actual first update uses η=.5. */
export function ManifoldForceFigure() {
  const fixture = MANIFOLD_FIXTURES.t_sne_tiny;
  const initial = fixture.initial_Y;
  const state = tsneObjective(fixture.P, initial);
  const total = -state.gradient[0][0];
  return <TeachingFigure number="F4" title="One attractive pair is only part of A’s total update.">
    <p>Input A,B,C,D = [0, 1, 3, 7], row perplexity 2. Starting map Y = [−1.5, −0.5, 0.5, 1.5]. Its Student-t weights normalize over ordered pairs with <strong>Z = {format(state.normalization)}</strong>.</p>
    <p className="mnfig-equation">Contribution from j = <strong>−4</strong> × (p<sub>Aj</sub> − q<sub>Aj</sub>) × t<sub>Aj</sub> × (y<sub>A</sub> − y<sub>j</sub>).</p>
    <p className="mnfig-note">Every arrow uses the same signed axis: map displacement per unit step size. Right is positive; left is negative.</p>
    <div className="mnfig-forces">
      {[1, 2, 3].map(index => <SignedContribution key={index} value={state.pairContributions[0][index][0]} title={`A due to ${names[index]} · ${fixture.P[0][index] > state.Q[0][index] ? 'attraction' : 'repulsion'}`}>
        <dl className="mnfig-values">
          <div><dt>p</dt><dd>{format(fixture.P[0][index], 10)}</dd></div>
          <div><dt>q</dt><dd>{format(state.Q[0][index])}</dd></div>
          <div><dt>t</dt><dd>{format(state.kernel[0][index])}</dd></div>
          <div><dt>Offset</dt><dd>{format(initial[0][0] - initial[index][0])}</dd></div>
        </dl>
      </SignedContribution>)}
      <SignedContribution value={total} title="Sum: A’s negative gradient" total>
        <p>At step size 0.5, Δy<sub>A</sub> = 0.5 × {format(total, 10)} = <strong>{signed(0.5 * total, 10)}</strong>.</p>
      </SignedContribution>
    </div>
    <DataTable className="mnfig-stack-narrow" caption="The complete initial gradient and first descent step, from the same P and Y"
      headings={['Point', 'Gradient', '0.5 × −gradient', 'New position']}
      rows={initial.map((row, index) => [names[index], signed(state.gradient[index][0], 6), signed(-0.5 * state.gradient[index][0], 6), format(row[0] - 0.5 * state.gradient[index][0], 6)])} />
    <p>A–B attracts A rightward, A–C also attracts it, and A–D repels it leftward. Their sum moves A right. The four gradients sum to zero, so this step already preserves the centered origin.</p>
    <p className="mnfig-note">The saved independent finite-difference check differed by {fixture.gradient_max_error.toExponential(3)} at most. Section 10 runs the full 200-step example; this figure exposes its first update.</p>
  </TeachingFigure>;
}

function CandidatePair({ separation }) {
  return <div className="mnfig-candidate-pair">
    <p><strong>Candidate separation r = {separation}</strong> · graph w = 0.625</p>
    <svg data-manifold-geometry="true" viewBox="0 0 300 92" role="img" aria-label={`Illustrative candidate map coordinates i 0 and j ${separation}. The input graph weight remains 0.625.`}>
      <line className="mnfig-axis" x1="24" x2="276" y1="40" y2="40" />
      {[0, 1, 2].map(tick => <text key={tick} className="mnfig-tick" x={30 + 120 * tick} y="72" textAnchor="middle">{tick}</text>)}
      {[[0, 'i'], [separation, 'j']].map(([value, label]) => <g key={label}>
        <circle className="mnfig-point" cx={30 + 120 * value} cy="40" r="5" />
        <text x={30 + 120 * value} y="19" textAnchor="middle">{label}</text>
      </g>)}
    </svg>
  </div>;
}

/** F5 · Calibration excludes self; reciprocal weights share one physical distance. */
export function ManifoldFuzzyGraphFigure() {
  const arrowId = useId().replace(/:/g, '');
  const edge = umapConnection({ distance: 2, rhoI: 1, rhoJ: 1, sigmaI: 1 / Math.log(2), sigmaJ: 1 / Math.log(4) });
  return <TeachingFigure number="F5" title="Local scales set directed strengths; their union fixes the graph edge." category="Exact calculation + algorithm diagram">
    <div className="mnfig-panels">
      <section className="mnfig-panel">
        <h4>1. Calibrate one local row</h4>
        <p>Neighbor array includes self. With ρ = σ = 1, the nonself mass reaches log₂ 4 = <strong>2</strong>.</p>
        <svg data-manifold-geometry="true" viewBox="0 0 300 190" role="img" aria-label="Algorithm diagram: i connects to B with strength 1, C with strength one half, and D with strength one half. The self edge is crossed out; it is excluded from the sum. Positions only organize this graph diagram.">
          <line className="mnfig-membership" x1="70" y1="94" x2="236" y2="30" strokeWidth="5" />
          <line className="mnfig-membership" x1="70" y1="94" x2="236" y2="94" strokeWidth="2.5" />
          <line className="mnfig-membership" x1="70" y1="94" x2="236" y2="158" strokeWidth="2.5" />
          <path className="mnfig-self-edge" d="M62,89 C14,48 62,8 74,85" />
          <path className="mnfig-exclusion" d="M31,37 L61,67 M31,67 L61,37" />
          <text x="30" y="19">self</text>
          {[[70, 94, 'i'], [236, 30, 'B'], [236, 94, 'C'], [236, 158, 'D']].map(([x, y, label]) => <g key={label}>
            <circle className="mnfig-point" cx={x} cy={y} r="5" />
            <text x={label === 'i' ? x - 16 : x + 16} y={y + 4} textAnchor={label === 'i' ? 'end' : 'start'}>{label}</text>
          </g>)}
        </svg>
        <p className="mnfig-note">Graph diagram: positions arrange the labels; edge width represents strength.</p>
      </section>
      <section className="mnfig-panel">
        <DataTable caption="One training neighbor row"
          headings={['Entry', 'Distance', 'Membership']}
          rows={[[<s>Self</s>, '0', 'Excluded'], ['B', '1', '1'], ['C', '1 + ln 2', '1/2'], ['D', '1 + ln 2', '1/2']]} />
        <p><strong>1 + 1/2 + 1/2 = 2.</strong> The omitted self connection is not a fourth probability competing for mass.</p>
      </section>
    </div>
    <section className="mnfig-stage">
      <h4>2. Combine two directions at the same distance d = 2</h4>
      <div className="mnfig-panels">
        <div>
          <svg data-manifold-geometry="true" className="mnfig-compact" viewBox="0 0 300 143" role="img" aria-label="i to j strength 0.5; j to i strength 0.25. Both use distance 2 and rho 1. Different local sigma values produce different strengths.">
            <ArrowDefinitions id={arrowId} />
            <path className="mnfig-direction" d="M48,63 Q150,3 252,63" markerEnd={`url(#${arrowId})`} />
            <path className="mnfig-direction is-reverse" d="M252,77 Q150,137 48,77" markerEnd={`url(#${arrowId})`} />
            <circle className="mnfig-point" cx="42" cy="70" r="5" /><circle className="mnfig-point" cx="258" cy="70" r="5" />
            <text x="23" y="74" textAnchor="middle">i</text><text x="277" y="74" textAnchor="middle">j</text>
          </svg>
          <p className="mnfig-note">Solid upper arrow: i → j. Dashed lower arrow: j → i.</p>
        </div>
        <div>
          <p><strong>i → j:</strong> ρᵢ = 1, σᵢ = 1/ln 2.<br />vᵢⱼ = exp(−ln 2) = {format(edge.forward)}.</p>
          <p><strong>j → i:</strong> ρⱼ = 1, σⱼ = 1/ln 4.<br />vⱼᵢ = exp(−ln 4) = {format(edge.reverse)}.</p>
        </div>
      </div>
      <p className="mnfig-equation"><strong>Union w = {format(edge.forward)} + {format(edge.reverse)} − ({format(edge.forward)} × {format(edge.reverse)}) = {format(edge.weight)}.</strong></p>
      <svg data-manifold-geometry="true" className="mnfig-compact" viewBox="0 0 300 66" role="img" aria-label="One undirected fuzzy union edge between i and j with weight 0.625.">
        <line className="mnfig-membership" x1="42" x2="258" y1="32" y2="32" strokeWidth={8 * edge.weight} />
        <circle className="mnfig-point" cx="42" cy="32" r="5" /><circle className="mnfig-point" cx="258" cy="32" r="5" />
        <text x="23" y="36" textAnchor="middle">i</text><text x="277" y="36" textAnchor="middle">j</text>
      </svg>
    </section>
    <section className="mnfig-stage">
      <h4>3. Hold the graph fixed while considering coordinates</h4>
      <div className="mnfig-panels"><CandidatePair separation={1} /><CandidatePair separation={2} /></div>
      <p className="mnfig-note">These two candidate coordinate pairs are algorithm illustrations, not fitted UMAP outputs. Their separations differ; the input edge stays w = 0.625. Investigation C checks the corresponding ideal pair costs.</p>
    </section>
  </TeachingFigure>;
}

function RetentionOverview({ layouts }) {
  return <section className="mnfig-stage">
    <h4>Fraction of original ten-neighbor choices retained</h4>
    <p className="mnfig-note">Every row uses the same 0–1 axis. A filled seed-7 dot and seed-19 ring coincide exactly for each saved PCA-initialized t-SNE pair.</p>
    <div className="mnfig-retention">
      {layouts.map(layout => <div className="mnfig-retention-row" key={layout.key}>
        <div className="mnfig-bar-label"><strong>{layout.key === 'pca' ? 'PCA · 2 components' : `t-SNE · perplexity ${layout.key.split('-')[1].slice(1)}`}</strong><span>{format(layout.metrics['10'].retention)}</span></div>
        <svg data-manifold-geometry="true" viewBox="0 0 300 42" role="img" aria-label={`${layout.title}, mean retention at k 10: ${layout.metrics['10'].retention}. ${layout.key === 'pca' ? '' : 'Seed 7 and seed 19 have identical values.'}`}>
          <line className="mnfig-axis" x1="18" x2="282" y1="21" y2="21" />
          {[0, 0.25, 0.5, 0.75, 1].map(tick => <line className="mnfig-grid" key={tick} x1={18 + 264 * tick} x2={18 + 264 * tick} y1="12" y2="30" />)}
          {layout.key !== 'pca' && <circle className="mnfig-seed-ring" cx={18 + 264 * layout.metrics['10'].retention} cy="21" r="6.5" />}
          <circle className="mnfig-retention-point" cx={18 + 264 * layout.metrics['10'].retention} cy="21" r="3" />
        </svg>
      </div>)}
      <svg data-manifold-geometry="true" className="mnfig-retention-axis" viewBox="0 0 300 32" aria-hidden="true">
        {[0, 0.25, 0.5, 0.75, 1].map(tick => <text className="mnfig-tick" key={tick} x={18 + 264 * tick} y="18" textAnchor="middle">{tick}</text>)}
      </svg>
    </div>
  </section>;
}

/** F6 · Actual coordinate overview; selected identity never reveals D's local answer. */
export function ManifoldDigitsFigure() {
  const selectorId = useId();
  const [queryId, setQueryId] = useState(MANIFOLD_DIGITS.rows[0].sourceRow);
  const [colorLabels, setColorLabels] = useState(false);
  const [mobileMap, setMobileMap] = useState('pca');
  const row = MANIFOLD_DIGITS.rows.find(candidate => candidate.sourceRow === queryId);
  const maps = ['pca', 'tsne-p30-s7'].map(key => MANIFOLD_DIGITS.layouts.find(layout => layout.key === key));
  const comparisons = ['pca', 'tsne-p5-s7', 'tsne-p30-s7', 'tsne-p80-s7'].map(key => MANIFOLD_DIGITS.layouts.find(layout => layout.key === key));
  return <TeachingFigure number="F6" title="The same 300 measured digits, shown through two fitted maps." category="Native fitted coordinates + measured neighbor scores">
    <p className="mnfig-note">UCI optical digits: first 30 observations per digit label, retained in original source-row order. Labels influenced this balanced selection; fits use only the 64 pixels divided by 16. No per-pixel standardization.</p>
    <div className="mnfig-digit-selection">
      <div>
        <label className="mnfig-field" htmlFor={selectorId}>Inspect an observation by its source row
          <select id={selectorId} aria-label="Inspect an observation by its source row" value={queryId} onChange={event => setQueryId(Number(event.target.value))}>
            {MANIFOLD_DIGITS.rows.map(candidate => <option value={candidate.sourceRow} key={candidate.sourceRow}>Source {candidate.sourceRow} · digit {candidate.digit}</option>)}
          </select>
        </label>
        <label className="mnfig-checkbox"><input type="checkbox" checked={colorLabels} onChange={event => setColorLabels(event.target.checked)} /> Color by recorded digit label</label>
        <p className="mnfig-note">Clicking a map point selects the same source row in both views. Investigation D will ask you to predict its local neighbor retention.</p>
      </div>
      <DigitTile row={row} caption={`Source ${row.sourceRow} · digit ${row.digit}`} />
    </div>
    <div className="mnfig-map-switch" role="group" aria-label="Visible overview map on a narrow screen">
      {maps.map(layout => <button type="button" key={layout.key} aria-pressed={mobileMap === layout.key} onClick={() => setMobileMap(layout.key)}>{layout.key === 'pca' ? 'PCA' : 't-SNE · p30'}</button>)}
    </div>
    <div className="mnfig-digit-maps">
      {maps.map(layout => <div key={layout.key} className={`mnfig-digit-map${mobileMap === layout.key ? ' is-visible' : ''}`}>
        <DigitScatter rows={MANIFOLD_DIGITS.rows} coordinates={layout.coordinates} queryId={queryId} onSelect={setQueryId} colorLabels={colorLabels} title={layout.key === 'pca' ? 'PCA · two components' : 't-SNE · p30, PCA init, seed 7'} />
      </div>)}
    </div>
    {colorLabels && <DigitLegend />}
    <p className="mnfig-note">Equal x/y scale within each map. Map coordinates have arbitrary units; raw radii cannot be compared across these methods.</p>
    <details className="mnfig-measurements">
      <summary>Exact measurements for source {queryId}</summary>
      <p className="mnfig-note">The 8 × 8 integer pixel grid below is in row-major order, on the original 0–16 scale.</p>
      {Array.from({ length: 8 }, (_, index) => <p key={index} className="mnfig-pixel-row"><strong>Row {index + 1}:</strong> {row.pixels.slice(index * 8, index * 8 + 8).join(', ')}</p>)}
      <DataTable className="mnfig-stack-narrow" caption="The selected observation’s fitted coordinates, rounded to nine decimal places"
        headings={['Map', 'Coordinate 1', 'Coordinate 2']}
        rows={maps.map(layout => {
          const index = MANIFOLD_DIGITS.rows.findIndex(candidate => candidate.sourceRow === queryId);
          return [layout.key === 'pca' ? 'PCA' : 't-SNE p30', ...layout.coordinates[index].map(value => format(value, 9))];
        })} />
    </details>
    <RetentionOverview layouts={comparisons} />
    <DataTable caption="Trustworthiness T₁₀ and continuity C₁₀; native scikit-learn sorting convention"
      headings={['Map', 'T₁₀', 'C₁₀']}
      rows={comparisons.map(layout => [layout.key === 'pca' ? 'PCA' : `t-SNE p${layout.key.split('-')[1].slice(1)}`, format(layout.metrics['10'].trustworthiness), format(layout.metrics['10'].continuity)])} />
    <p>A collection average and one image’s neighborhood answer different questions. The next investigation supplies the original-space image lists for a local audit.</p>
  </TeachingFigure>;
}

function LabeledMatrix({ title, entries, numerator = false }) {
  return <section className="mnfig-matrix">
    <h4>{title}</h4>
    {numerator && <p>B = 1/9 × the matrix below</p>}
    <DataTable caption={numerator ? 'Exact integer numerators of B' : 'Exact input distances D'} headings={['', 'A', 'B', 'C']}
      rows={entries.map((row, index) => [names[index], ...row.map(value => format(value, 0))])} />
  </section>;
}

/** F7 · Exact matrix cells, identity-preserving coordinate reconstruction. */
export function ManifoldMdsFigure() {
  const fixture = MANIFOLD_FIXTURES.classical_mds;
  const coordinates = fixture.coordinates;
  const scale = value => 30 + 40 * (value + 3);
  return <TeachingFigure number="F7" title="Double centering turns distances into dot products, then coordinates.">
    <div className="mnfig-panels">
      <LabeledMatrix title="1. Distances D" entries={fixture.distances} />
      <LabeledMatrix title="2. Centered Gram matrix B" entries={fixture.B.map(row => row.map(value => Math.round(9 * value)))} numerator />
    </div>
    <p className="mnfig-equation">B = −½ J D<sup>∘2</sup> J = yyᵀ, with D squared entry by entry.<br />One positive eigenvalue: <strong>38/3</strong>. Two zero eigenvalues.</p>
    <h4>3. Recovered centered line</h4>
    <svg data-manifold-geometry="true" className="mnfig-compact" viewBox="0 0 300 174" role="img" aria-label="Recovered line A minus seven thirds, B minus one third, C eight thirds. The adjacent gaps are 2 and 3; A to C is 5.">
      <line className="mnfig-axis" x1="24" x2="276" y1="63" y2="63" />
      {[[0, 1, '2'], [1, 2, '3']].map(([left, right, gap]) => <g key={gap}>
        <path className="mnfig-bracket" d={`M${scale(coordinates[left])},38 V29 H${scale(coordinates[right])} V38`} />
        <text x={scale((coordinates[left] + coordinates[right]) / 2)} y="19" textAnchor="middle">gap {gap}</text>
      </g>)}
      {coordinates.map((value, index) => <g key={index}>
        <circle className="mnfig-point" cx={scale(value)} cy="63" r="4.5" />
        <text x={scale(value)} y="51" textAnchor="middle">{names[index]}</text>
        <text x={scale(value)} y="85" textAnchor="middle">{['−7/3', '−1/3', '8/3'][index]}</text>
      </g>)}
      {[-3, -2, -1, 0, 1, 2, 3].map(tick => <text className="mnfig-tick" x={scale(tick)} y="112" key={tick} textAnchor="middle">{tick}</text>)}
      <path className="mnfig-bracket" d={`M${scale(coordinates[0])},132 V141 H${scale(coordinates[2])} V132`} />
      <text x={scale((coordinates[0] + coordinates[2]) / 2)} y="162" textAnchor="middle">total distance 5</text>
    </svg>
    <p><strong>B<sub>AB</sub> = y<sub>A</sub> × y<sub>B</sub> = (−7/3)(−1/3) = 7/9.</strong> The recovered line gives back all three input distances.</p>
    <p className="mnfig-note">The displayed fractions are exact. Numerical eigensolver residues with absolute value below 10⁻¹² are treated as zero here.</p>
  </TeachingFigure>;
}

function ReconstructionLine({ positions, title }) {
  const arrowId = useId().replace(/:/g, '');
  const scale = value => 30 + 40 * value;
  return <section className="mnfig-panel">
    <h4>{title}</h4>
    <svg data-manifold-geometry="true" viewBox="0 0 300 168" role="img" aria-label={`${title}: left endpoint ${positions[0]}, interior ${positions[1]}, right endpoint ${positions[2]}. The recipe assigns weight two thirds to the left endpoint and one third to the right endpoint. Both panels use the same 0 to 6 coordinate scale.`}>
      <ArrowDefinitions id={arrowId} />
      <line className="mnfig-axis" x1="24" x2="276" y1="82" y2="82" />
      <path className="mnfig-recipe-edge" d={`M${scale(positions[0]) + 2},74 Q${scale((positions[0] + positions[1]) / 2)},30 ${scale(positions[1]) - 5},74`} markerEnd={`url(#${arrowId})`} />
      <path className="mnfig-recipe-edge is-right" d={`M${scale(positions[2]) - 2},74 Q${scale((positions[1] + positions[2]) / 2)},6 ${scale(positions[1]) + 5},74`} markerEnd={`url(#${arrowId})`} />
      {positions.map((value, index) => <g key={index}>
        <circle className={index === 1 ? 'mnfig-source' : 'mnfig-point'} cx={scale(value)} cy="82" r="4.5" />
        <text x={scale(value)} y="105" textAnchor="middle">{['L', 'i', 'R'][index]}</text>
      </g>)}
      {[[0, 1], [1, 2]].map(([left, right]) => <g key={left}>
        <path className="mnfig-bracket" d={`M${scale(positions[left])},116 V122 H${scale(positions[right])} V116`} />
        <text x={scale((positions[left] + positions[right]) / 2)} y="139" textAnchor="middle">{positions[right] - positions[left]}</text>
      </g>)}
      {[0, 1, 2, 3, 4, 5, 6].map(tick => <text className="mnfig-tick" x={scale(tick)} y="162" key={tick} textAnchor="middle">{tick}</text>)}
    </svg>
    <p><strong>{positions[1]} = (2/3 × {positions[0]}) + (1/3 × {positions[2]}).</strong></p>
  </section>;
}

/** F8 · The recipe survives a changed coordinate scale; distances do not. */
export function ManifoldLleFigure() {
  return <TeachingFigure number="F8" title="Keep the reconstruction weights when the coordinate spacing changes.">
    <p className="mnfig-note">L and R are the neighbors; i is reconstructed. Solid arrow from L: weight <strong>2/3</strong>. Dashed arrow from R: weight <strong>1/3</strong>. Brackets show coordinate gaps.</p>
    <div className="mnfig-panels">
      <ReconstructionLine title="Input positions: 0, 1, 3" positions={MANIFOLD_FIXTURES.lle.input} />
      <ReconstructionLine title="Output positions: 0, 2, 6" positions={MANIFOLD_FIXTURES.lle.output} />
    </div>
    <p>The point is one third of the way from L to R in each view. The gaps double from 1 and 2 to 2 and 4; the weights stay 2/3 and 1/3.</p>
    <p className="mnfig-note">This particular recipe has nonnegative weights. LLE’s sum-to-one constraint also permits negative weights in other neighborhoods.</p>
  </TeachingFigure>;
}

function AbstractComplex({ filled, title, description }) {
  const points = [[55, 158], [205, 158], [205, 28], [55, 28]];
  const edges = filled ? [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]] : [[0, 1], [1, 2], [2, 3], [3, 0]];
  return <section className="mnfig-complex">
    <h4>{title}</h4>
    <svg data-manifold-geometry="true" viewBox="0 0 260 191" role="img" aria-label={`${title}. ${description} Abstract simplicial-complex drawing; positions only separate the four identities.`}>
      {filled && <polygon className="mnfig-simplex-fill" points={points.map(point => point.join(',')).join(' ')} />}
      {edges.map(([left, right]) => <line key={`${left}-${right}`} className="mnfig-complex-edge" x1={points[left][0]} y1={points[left][1]} x2={points[right][0]} y2={points[right][1]} />)}
      {points.map(([x, y], index) => <g key={index}>
        <circle className="mnfig-point" cx={x} cy={y} r="4.5" />
        <text x={x + (x < 100 ? -17 : 17)} y={y + 4} textAnchor="middle">{names[index]}</text>
      </g>)}
    </svg>
    <p className={`mnfig-complex-state${filled ? ' is-filled' : ''}`}><strong>{filled ? 'Filled complex · no 1D loop' : 'One unfilled cycle'}</strong></p>
    <p className="mnfig-note">{description}</p>
  </section>;
}

/** F9 · Metric drawing and abstract complex stay separate for coincident points. */
export function ManifoldTopologyFigure() {
  return <TeachingFigure number="F9" title="Projection changes the distances before the Rips construction begins.">
    <div className="mnfig-panels">
      <section className="mnfig-panel">
        <h4>Input: unit square</h4>
        <svg data-manifold-geometry="true" className="mnfig-compact" viewBox="0 0 300 253" role="img" aria-label="Metric geometry: square A at 0,0; B at 1,0; C at 1,1; D at 0,1. Every side is 1, both diagonals are square root 2. Equal x and y scales.">
          {[0, 1].map(tick => <g key={tick}>
            <line className="mnfig-grid" x1="50" x2="248" y1={195 - 150 * tick} y2={195 - 150 * tick} />
            <line className="mnfig-grid" x1={75 + 150 * tick} x2={75 + 150 * tick} y1="24" y2="219" />
            <text className="mnfig-tick" x="35" y={199 - 150 * tick} textAnchor="middle">{tick}</text>
            <text className="mnfig-tick" x={75 + 150 * tick} y="243" textAnchor="middle">{tick}</text>
          </g>)}
          {[[75, 195], [225, 195], [225, 45], [75, 45]].map(([x, y], index) => <g key={index}>
            <circle className="mnfig-point" cx={x} cy={y} r="5" />
            <text x={x + (x < 100 ? -14 : 14)} y={y + (y < 100 ? -12 : 18)} textAnchor="middle">{names[index]}</text>
          </g>)}
        </svg>
        <p className="mnfig-note">Metric view: input coordinate units, equal x/y scale.</p>
      </section>
      <section className="mnfig-panel mnfig-route-strip">
        <h4>Projection: keep only x</h4>
        <svg data-manifold-geometry="true" viewBox="0 0 300 111" role="img" aria-label="Metric projection: A and D both have coordinate 0; B and C both have coordinate 1. Four observations remain, at two coincident positions. Duplicate-pair distances are zero.">
          <line className="mnfig-axis" x1="43" x2="257" y1="54" y2="54" />
          {[75, 225].map((x, index) => <g key={index}>
            <circle className="mnfig-duplicate" cx={x} cy="54" r="8" />
            <circle className="mnfig-point" cx={x} cy="54" r="3.5" />
            <text x={x} y="28" textAnchor="middle">{index ? 'B, C' : 'A, D'}</text>
            <text className="mnfig-tick" x={x} y="89" textAnchor="middle">{index}</text>
          </g>)}
        </svg>
        <p><strong>Four IDs; two positions.</strong> A–D and B–C connect at ε = 0. Every other projected pair is one unit apart.</p>
      </section>
    </div>
    <p className="mnfig-note">The panels below are abstract complexes: the four IDs are drawn separately for legibility. Their drawing positions do not assert metric distances. An edge enters when d ≤ ε; every clique includes its filled simplex.</p>
    <h4 className="mnfig-complex-row-heading">Rips complexes from the input square</h4>
    <div className="mnfig-panels">
      <AbstractComplex title="Input · ε = 1" description="Four side edges, no filled triangles. The loop is present." />
      <AbstractComplex title="Input · ε = √2" filled description="All six edges, four filled triangles and the tetrahedral simplex. The loop is filled." />
    </div>
    <h4 className="mnfig-complex-row-heading">Rips complexes from the projection</h4>
    <div className="mnfig-panels">
      <AbstractComplex title="Projection · ε = 1" filled description="All six edges, four filled triangles and the tetrahedral simplex already enter by this threshold." />
      <AbstractComplex title="Projection · ε = √2" filled description="The same complete filled complex remains. There is no corresponding 1D loop interval." />
    </div>
    <DataTable caption="Exact pair thresholds under the closed rule d ≤ ε"
      headings={['Pair', 'Input threshold', 'Projected threshold']}
      rows={[[ 'A–B', '1', '1'], ['B–C', '1', '0'], ['C–D', '1', '1'], ['D–A', '1', '0'], ['A–C', '√2', '1'], ['B–D', '√2', '1']]} />
    <p><strong>Input loop interval: [1, √2).</strong> The projected construction has no corresponding interval. This is an exact projection counterexample, not a t-SNE or UMAP result.</p>
  </TeachingFigure>;
}
