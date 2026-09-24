import {
  MIXING, applyMatrix, invert2, whitenedFixture, fastIcaStep, dependenceSummary, DEPENDENCE_POINTS,
} from '../../data/ica-models';
import { ICA_RECORDING } from '../../data/ica-data';
import { DataTable, EqualPlot, format, signedFormat } from './IcaShared';
import './ica-labs.css';

const fixture = whitenedFixture();
const STATE_CLASS = { A: 'ic-state-a', B: 'ic-state-b', C: 'ic-state-c', D: 'ic-state-d' };

function TeachingFigure({ id, title, category, children }) {
  return <figure className={`ic-figure ic-figure-${id.toLowerCase()}`} data-ica-figure={id}>
    <figcaption>
      <span className="ic-evidence">{id} · {category}</span>
      <strong>{title}</strong>
    </figcaption>
    {children}
  </figure>;
}

function StatePoints({ points, sx, sy, radius = 5.5 }) {
  return <>{points.map(point => <g key={point.id}>
    <circle className={`ic-state-point ${STATE_CLASS[point.id]}`} cx={sx(point.value[0])} cy={sy(point.value[1])} r={radius} />
    <text className="ic-state-label" x={sx(point.value[0]) + point.offset[0]} y={sy(point.value[1]) + point.offset[1]} textAnchor="middle">{point.id}</text>
  </g>)}</>;
}

const labelOffsets = { A: [0, 18], B: [-13, -9], C: [13, 13], D: [0, -11] };

const asPoints = select => fixture.states.map(state => ({ id: state.id, value: select(state), offset: labelOffsets[state.id] }));

/** M1 · One sample through two sensors, grouped by the sum each arrow enters. */
export function IcaMixingFigure() {
  const source = [1, -1];
  const observed = applyMatrix(MIXING, source);
  const inverse = invert2(MIXING);
  const recovered = applyMatrix(inverse, observed);
  const panels = [0, 1].map(sensor => ({
    sensor,
    coefficients: [MIXING[sensor][0], MIXING[sensor][1]],
    contributions: [MIXING[sensor][0] * source[0], MIXING[sensor][1] * source[1]],
    total: observed[sensor],
  }));
  return <TeachingFigure id="M1" title="Two sources, four weighted contributions, two sensor sums." category="Exact constructed calculation">
    <DataTable stack caption="The mixing matrix A. Row i is sensor i's recipe; column j is where source j appears."
      headings={['', 'from source 1', 'from source 2']}
      rows={[['Sensor 1', format(MIXING[0][0]), format(MIXING[0][1])], ['Sensor 2', format(MIXING[1][0]), format(MIXING[1][1])]]} />
    <div className="ic-panels">
      {panels.map(panel => <section className="ic-panel" key={panel.sensor}>
        <h4>{`Sensor ${panel.sensor + 1} = ${format(panel.total)}`}</h4>
        <svg className="ic-diagram" viewBox="0 0 300 148" style={{ maxWidth: '300px' }} role="img"
          aria-label={`Source 1 at plus 1 contributes ${format(panel.contributions[0])} and source 2 at minus 1 contributes ${format(panel.contributions[1])}; sensor ${panel.sensor + 1} totals ${format(panel.total)}.`}>
          <circle className="ic-node is-source1" cx="56" cy="34" r="17" />
          <text className="ic-label" x="56" y="38" textAnchor="middle">+1</text>
          <text className="ic-dim" x="56" y="12" textAnchor="middle">source 1</text>
          <circle className="ic-node is-source2" cx="56" cy="106" r="17" />
          <text className="ic-label" x="56" y="110" textAnchor="middle">{'−1'}</text>
          <text className="ic-dim" x="56" y="138" textAnchor="middle">source 2</text>
          <line className="ic-arrow is-source1" x1="74" y1="38" x2="214" y2="64" />
          <line className="ic-arrow is-source2" x1="74" y1="102" x2="214" y2="76" />
          <text x="132" y="36" textAnchor="middle">{`× ${format(panel.coefficients[0])} = ${format(panel.contributions[0])}`}</text>
          <text x="132" y="110" textAnchor="middle">{`× ${format(panel.coefficients[1])} = ${format(panel.contributions[1])}`}</text>
          <circle className="ic-node is-sum" cx="240" cy="70" r="21" />
          <text className="ic-label" x="240" y="74" textAnchor="middle">{format(panel.total)}</text>
          <text className="ic-dim" x="240" y="116" textAnchor="middle">{`sensor ${panel.sensor + 1}`}</text>
        </svg>
        <p className="ic-equation">{`${format(panel.contributions[0])} + (${format(panel.contributions[1])}) = `}<strong>{format(panel.total)}</strong></p>
      </section>)}
    </div>
    <p>Source 1 contributes {format(MIXING[0][0] * source[0])} and {format(MIXING[1][0] * source[0])} to sensors 1 and 2; source 2 contributes {format(MIXING[0][1] * source[1])} and {format(MIXING[1][1] * source[1])}. The sensor totals are {format(observed[0])} and {format(observed[1])}. A source column is a pattern across sensors, not an unmixing direction.</p>
    <h4>The first inverse row acts on the observed values</h4>
    <p className="ic-equation">
      {`b₁ᵀ = (${format(MIXING[1][1])}, ${format(-MIXING[0][1])}) / 3`}, so {`[${format(MIXING[1][1])}(${format(observed[0])}) − (${format(observed[1])})] / 3 = ${format(2 * observed[0] - observed[1])} / 3 = `}
      <strong>{format(recovered[0])}</strong>
    </p>
    <p className="ic-note">Mixing column 1 is ({format(MIXING[0][0])}, {format(MIXING[1][0])}); inverse row 1 is ({format(inverse[0][0] * 3)}, {format(inverse[0][1] * 3)})/3. They carry different signs because they answer different questions. Arrow lengths are layout only; every number above is exact.</p>
  </TeachingFigure>;
}

/** D1 · Zero covariance beside the conditional restriction it cannot see. */
export function IcaDependenceFigure() {
  const summary = dependenceSummary();
  const scale = 92;
  const originX = 34 + 1.3 * scale;
  const originY = 10 + 1.3 * scale;
  const sx = u => originX + u * scale;
  const sy = v => originY - v * scale;
  const height = originY + 0.15 * scale + 26;
  const panel = (title, visible, describe) => <section className="ic-panel" key={title}>
    <h4>{title}</h4>
    <svg className="ic-diagram" viewBox={`0 0 ${originX + 1.3 * scale + 10} ${height}`} style={{ maxWidth: '300px' }} role="img" aria-label={describe}>
      <line className="ic-axis" x1={sx(-1.3)} x2={sx(1.3)} y1={sy(0)} y2={sy(0)} />
      <line className="ic-axis" x1={sx(0)} x2={sx(0)} y1={sy(1.3)} y2={sy(-0.15)} />
      {[-1, 0, 1].map(tick => <text key={`u${tick}`} x={sx(tick)} y={sy(0) + 18} textAnchor="middle">{format(tick)}</text>)}
      {[1].map(tick => <text key={`v${tick}`} x={sx(0) - 8} y={sy(tick) + 4} textAnchor="end">{format(tick)}</text>)}
      <text className="ic-dim" x={sx(1.3)} y={sy(0) - 8} textAnchor="end">u</text>
      <text className="ic-dim" x={sx(0) + 10} y={sy(1.3) + 10} textAnchor="start">v</text>
      <path className="ic-absent" d={Array.from({ length: 41 }, (unused, index) => {
        const u = -1 + index / 20;
        return `${index === 0 ? 'M' : 'L'}${sx(u)},${sy(u * u)}`;
      }).join(' ')} />
      <text className="ic-dim" x={sx(-1.3) + 6} y={sy(1.15)} textAnchor="start">relation v = u²</text>
      {DEPENDENCE_POINTS.map((point, index) => (visible.includes(index)
        ? <circle key={index} className="ic-state-point ic-state-a" cx={sx(point.u)} cy={sy(point.v)} r="6" />
        : <circle key={index} className="ic-absent" cx={sx(point.u)} cy={sy(point.v)} r="6" />))}
    </svg>
  </section>;
  return <TeachingFigure id="D1" title="Covariance cancels to zero while one value still fixes the other." category="Exact constructed probabilities">
    <div className="ic-panels">
      {panel('All three equiprobable outcomes', [0, 1, 2], 'Three points at minus one comma one, zero comma zero, and one comma one, on the curve v equals u squared.')}
      {panel('After observing v = 0', [1], 'Only the point zero comma zero remains; the other two outcomes are drawn as empty circles.')}
    </div>
    <DataTable caption="Each outcome has probability one third. The products cancel; the joint event does not factor."
      headings={['Outcome', 'u', 'v = u²', 'u · v']}
      rows={summary.points.map((point, index) => [`#${index + 1}`, format(point.u), format(point.v), signedFormat(point.product)])} />
    <p className="ic-equation">
      E[u] = {format(summary.meanU)} · E[v] = {format(summary.meanV, 4)} · E[uv] = {format(summary.covariance)} → <strong>Cov(u, v) = {format(summary.covariance)}</strong>
    </p>
    <p className="ic-equation">
      P(u = 0, v = 0) = <strong>{format(summary.jointZero, 4)}</strong>, while P(u = 0) · P(v = 0) = {format(summary.marginalU, 4)} × {format(summary.marginalV, 4)} = <strong>{format(summary.marginalProduct, 4)}</strong>
    </p>
    <p className="ic-note">These are the section&apos;s own u and v, not the binary source mixture. The dashed curve marks the relation; it is not sampled continuous data.</p>
  </TeachingFigure>;
}

/** W1 · Four corresponding clouds, each drawn from the shared fixture. */
export function IcaWhiteningFigure() {
  const panels = [
    { key: 'source', title: 'Source s', bounds: [-1.5, 1.5, -1.5, 1.5], ticks: [-1.5, 0, 1.5], select: state => state.source, transform: null },
    { key: 'observed', title: 'Observation x', bounds: [-3.5, 3.5, -3.5, 3.5], ticks: [-3.5, 0, 3.5], select: state => state.observed, transform: 'A = [[2, 1], [1, 2]]' },
    { key: 'whitened', title: 'Whitened z', bounds: [-1.8, 1.8, -1.8, 1.8], ticks: [-1.8, 0, 1.8], select: state => state.whitened, transform: 'K = [[1, 1]/(3√2), [1, −1]/√2]' },
    { key: 'recovered', title: 'Recovered s', bounds: [-1.5, 1.5, -1.5, 1.5], ticks: [-1.5, 0, 1.5], select: state => state.recovered, transform: 'Q = [[1, 1], [1, −1]]/√2' },
  ];
  return <TeachingFigure id="W1" title="Whitening removes the stretch; the four states stay dependent." category="Exact constructed calculation">
    <div className="ic-panels">
      {panels.map(panel => <section className="ic-panel" key={panel.key}>
        <p className="ic-note">{panel.transform ? `→ ${panel.transform} →` : 'start'}</p>
        <EqualPlot title={panel.title} bounds={panel.bounds} ticks={panel.ticks} size={280}
          describe={`States A to D at ${fixture.states.map(state => `${state.id} (${format(panel.select(state)[0], 4)}, ${format(panel.select(state)[1], 4)})`).join(', ')}. Both axes use the same scale over ${format(panel.bounds[0], 2)} to ${format(panel.bounds[1], 2)}.`}>
          {(sx, sy) => <>
            <line className="ic-axis" x1={sx(panel.bounds[0])} x2={sx(panel.bounds[1])} y1={sy(0)} y2={sy(0)} />
            <line className="ic-axis" x1={sx(0)} x2={sx(0)} y1={sy(panel.bounds[2])} y2={sy(panel.bounds[3])} />
            {panel.key === 'whitened' && <>
              <circle className="ic-absent" cx={sx(0)} cy={sy(0)} r="7" />
              <text className="ic-dim" x={sx(0) - 12} y={sy(0) - 12} textAnchor="end">no mass here</text>
            </>}
            <StatePoints points={asPoints(panel.select)} sx={sx} sy={sy} />
          </>}
        </EqualPlot>
        {panel.key === 'observed' && <p className="ic-equation">{`Σₓ = [[${format(fixture.covariance[0][0])}, ${format(fixture.covariance[0][1])}], [${format(fixture.covariance[1][0])}, ${format(fixture.covariance[1][1])}]]`}</p>}
        {panel.key === 'whitened' && <>
          <p className="ic-equation">{`E[zzᵀ] = [[${format(fixture.whitenedCovariance[0][0])}, ${format(fixture.whitenedCovariance[0][1])}], [${format(fixture.whitenedCovariance[1][0])}, ${format(fixture.whitenedCovariance[1][1])}]]`}</p>
          <p className="ic-equation">{'P(z₁ = 0) = P(z₂ = 0) = 0.5, but P(z₁ = 0, z₂ = 0) = 0'}</p>
        </>}
        {panel.key === 'recovered' && <p className="ic-note">Orthogonal change, including a reflection.</p>}
      </section>)}
    </div>
    <DataTable stack caption="The same four states, carried through every panel. PCA scores have variances 9 and 1; whitening divides by 3 and 1."
      headings={['State', 'source s', 'observed x', 'whitened z', 'recovered s']}
      rows={fixture.states.map(state => [state.id,
        `(${format(state.source[0])}, ${format(state.source[1])})`,
        `(${format(state.observed[0])}, ${format(state.observed[1])})`,
        `(${format(state.whitened[0], 5)}, ${format(state.whitened[1], 5)})`,
        `(${format(state.recovered[0], 5)}, ${format(state.recovered[1], 5)})`])} />
    <p className="ic-note">Each panel has equal x and y scales and its own labelled extent: ±1.5, ±3.5, ±1.8, ±1.5. Correspondence is the A–D identity, not a shared scale. The hollow circle marks the joint zero event that carries no probability.</p>
  </TeachingFigure>;
}

/** F1 · One fixed-point operation, exposed as its actual averages. */
export function IcaFixedPointFigure() {
  const start = [0.8, 0.6];
  const step = fastIcaStep(fixture.states.map(state => state.whitened), start);
  const radius = 82;
  const centre = 130;
  const point = (vector, length = radius) => [centre + vector[0] * length, centre - vector[1] * length];
  const [oldX, oldY] = point(start);
  const [newX, newY] = point(step.next);
  const [flipX, flipY] = point(step.signAligned);
  return <TeachingFigure id="F1" title="Four projections, two averages, one normalized direction." category="Exact constructed calculation">
    <DataTable dense stack caption="The four whitened states under w = (0.8, 0.6), with g(u) = u³."
      headings={['State', 'z', 'y = wᵀz', 'y³', 'z y³', '3y²']}
      rows={step.rows.map((row, index) => [fixture.states[index].id,
        `(${format(row.point[0], 5)}, ${format(row.point[1], 5)})`,
        format(row.projection, 5),
        format(row.cube, 5),
        `(${format(row.weighted[0], 5)}, ${format(row.weighted[1], 5)})`,
        format(row.derivative, 5)])} />
    <div className="ic-panels">
      <section className="ic-panel">
        <h4>The operation, in order</h4>
        <p className="ic-equation">1 · average of z y³ = <strong>({format(step.weightedMean[0])}, {format(step.weightedMean[1])})</strong></p>
        <p className="ic-equation">2 · average of 3y² = <strong>{format(step.derivativeMean)}</strong>, so the correction is {format(step.derivativeMean)}({format(start[0])}, {format(start[1])}) = ({format(step.correction[0])}, {format(step.correction[1])})</p>
        <p className="ic-equation">3 · r = ({format(step.weightedMean[0])}, {format(step.weightedMean[1])}) {'−'} ({format(step.correction[0])}, {format(step.correction[1])}) = <strong>({format(step.raw[0])}, {format(step.raw[1])})</strong></p>
        <p className="ic-equation">4 · divide by {'‖'}r{'‖'} = {format(step.rawNorm)} {'→'} <strong>({format(step.next[0], 9)}, {format(step.next[1], 9)})</strong></p>
        <p className="ic-equation">5 · the sign-equivalent direction is ({format(step.signAligned[0], 9)}, {format(step.signAligned[1], 9)})</p>
        <p className="ic-note">Parallel and antiparallel vectors describe the same source direction, so 1 − |w_newᵀw| = {format(step.convergence, 6)} is the quantity a stopping test would watch. One update is not convergence.</p>
      </section>
      <section className="ic-panel">
        <h4>Old and new directions on the unit circle</h4>
        <svg className="ic-diagram" viewBox="0 0 260 260" style={{ maxWidth: '260px' }} role="img"
          aria-label={`The old direction 0.8, 0.6 points up and right. The actual new direction ${format(step.next[0], 4)}, ${format(step.next[1], 4)} points down and left. Its dashed sign-equivalent points up and right toward the source axis.`}>
          <circle cx={centre} cy={centre} r={radius} fill="none" className="ic-grid" />
          <line className="ic-axis" x1={centre - radius - 14} x2={centre + radius + 14} y1={centre} y2={centre} />
          <line className="ic-axis" x1={centre} x2={centre} y1={centre - radius - 14} y2={centre + radius + 14} />
          <line className="ic-direction" x1={centre} y1={centre} x2={oldX} y2={oldY} />
          <line className="ic-direction is-companion" x1={centre} y1={centre} x2={flipX} y2={flipY} />
          <line className="ic-direction is-proposed" x1={centre} y1={centre} x2={newX} y2={newY} />
          <text x={oldX + 8} y={oldY + 14} textAnchor="start">old w</text>
          <text x={flipX - 10} y={flipY - 10} textAnchor="end">flipped new w</text>
          <text x={newX} y={newY + 18} textAnchor="middle">new w</text>
        </svg>
        <p className="ic-note">The update lands in the negative quadrant; flipping its sign gives the equivalent direction near (1, 1)/√2, which extracts s₁. The flip is a display convention, not a second update.</p>
      </section>
    </div>
  </TeachingFigure>;
}

/** A small padlock: the frozen coordinate index is a lock, not only a word. */
function LockMark() {
  return <svg className="ic-lock" viewBox="0 0 14 16" aria-hidden="true" focusable="false">
    <path d="M3.5 7V4.5a3.5 3.5 0 1 1 7 0V7" fill="none" strokeWidth="1.6" />
    <rect x="1.6" y="7" width="10.8" height="7.6" rx="1.2" />
  </svg>;
}

/** P1 · Which information is allowed to influence which decision.
 *  Laid out in HTML rather than one wide SVG so every label keeps its real
 *  reading size at 320 px instead of shrinking with a viewBox.
 */
export function IcaSplitFigure() {
  const { split, samplingHz } = ICA_RECORDING;
  const lanes = ['Abdomen 1', 'Abdomen 2', 'Abdomen 3', 'Abdomen 4'];
  const stages = [
    ['Fit', `0–12 s (${split.train[1]} instants)`, 'four abdominal channels', 'PCA and ICA fits, means and whitening', 'the direct reference, and every later interval'],
    ['Select', '12–16 s', 'four abdominal channels and the reference', 'one frozen coordinate index per representation', 'test-interval values'],
    ['Evaluate', '16–20 s', 'the already frozen indices and the reference', 'one held-out absolute correlation per representation', 'changing the selection after seeing this'],
  ];
  const outputs = [
    ['fit (blind)', '0–12 s', 'PCA and ICA fitted here', false],
    ['select', '12–16 s', 'one coordinate index per method, then locked', true],
    ['evaluate', '16–20 s', 'one held-out |r| per method', false],
  ];
  return <TeachingFigure id="P1" title="A chronological evidence boundary: fit, then select, then evaluate." category="Declared protocol">
    <div className="ic-lane-grid" role="img"
      aria-label="Four abdominal lanes run through all three stages. The direct reference lane is empty under the fit stage and present only under selection and evaluation, so no reference value reaches either decomposition.">
      <div className="ic-lane-row is-heads" aria-hidden="true">
        <span className="ic-lane-name" />
        <div className="ic-lane-stages">{outputs.map(([stage]) => <span key={stage}>{stage}</span>)}</div>
      </div>
      {lanes.map(lane => <div className="ic-lane-row" key={lane}>
        <span className="ic-lane-name">{lane}</span>
        <div className="ic-lane-stages">
          <i className="ic-lane-cell is-fit" /><i className="ic-lane-cell is-select" /><i className="ic-lane-cell is-evaluate" />
        </div>
      </div>)}
      <div className="ic-lane-row is-reference">
        <span className="ic-lane-name">Direct reference</span>
        <div className="ic-lane-stages">
          <i className="ic-lane-cell is-absent"><em>not used</em></i>
          <i className="ic-lane-cell is-select is-reference" /><i className="ic-lane-cell is-evaluate is-reference" />
        </div>
      </div>
      <div className="ic-lane-row is-ruler">
        <span className="ic-lane-name">seconds</span>
        <div className="ic-lane-stages">
          {outputs.map(([stage, interval]) => <span key={stage}>{interval}</span>)}
        </div>
      </div>
      <div className="ic-lane-row is-output">
        <span className="ic-lane-name" aria-hidden="true" />
        <div className="ic-lane-stages">
          {outputs.map(([stage, interval, output, locked]) => <span key={stage}>
            <b aria-hidden="true">↓</b>
            <em className="ic-stage-name">{stage} · {interval}</em>
            {locked && <LockMark />}
            {output}
          </span>)}
        </div>
      </div>
    </div>
    <div className="ic-lane-legend">
      <span><i style={{ background: '#8b9aa7' }} aria-hidden="true" />abdominal input, four channels</span>
      <span><i style={{ background: '#cf958f' }} aria-hidden="true" />direct reference, never an input to a fit</span>
    </div>
    <DataTable stack caption={`Each stage, its interval at ${samplingHz} Hz, what it may read, and what it must not.`}
      headings={['Stage', 'Interval', 'Arrays available', 'Output', 'Forbidden dependency']}
      rows={stages} />
    <p className="ic-note">The decomposition itself is blind: no reference value enters the PCA or ICA fit. Labelling one coordinate as “the” candidate does use the reference, on development data only, so the overall procedure is not unsupervised end to end. The provider filtered this recording offline; the diagram describes a retrospective comparison, not an online pipeline.</p>
  </TeachingFigure>;
}

function TraceRow({ name, title, series, limit, unitNote }) {
  const columns = series.length;
  const height = 56;
  const sy = value => height / 2 - (value / limit) * (height / 2 - 3);
  const upper = series.map((column, index) => `${index === 0 ? 'M' : 'L'}${index},${sy(column[1])}`).join(' ');
  const lower = series.slice().reverse().map((column, index) => `L${columns - 1 - index},${sy(column[0])}`).join(' ');
  return <div className="ic-trace">
    <p className="ic-trace-title">{title}</p>
    <svg viewBox={`0 0 ${columns} ${height}`} preserveAspectRatio="none" role="img"
      aria-label={`${title}. A peak-preserving envelope of 2,000 samples between 16 and 18 seconds, shown as a display z-score. ${unitNote}`}>
      <line className="ic-trace-zero" x1="0" x2={columns} y1={sy(0)} y2={sy(0)} />
      <path className={`ic-trace-path${name === 'reference' ? ' is-reference' : ''}`} d={`${upper} ${lower} Z`} />
    </svg>
  </div>;
}

/** E1 · The actual outcome, and what the correlation corresponds to. */
export function IcaOutcomeFigure() {
  const { results, trace, window: exact, limits } = ICA_RECORDING;
  // Derived from the published envelope, so the figure can never silently clip
  // a peak if the recording module is regenerated.
  const extreme = Math.max(...Object.values(trace.series).flat().flat().map(Math.abs));
  const limit = Math.ceil((extreme + 0.1) * 10) / 10;
  const rowTitle = {
    channel: `Abdominal channel ${results[0].chosen} (recorded microvolts, display z-score)`,
    PCA: `PCA coordinate ${results[1].chosen} (arbitrary units, display z-score)`,
    ICA: `ICA coordinate ${results[2].chosen} (arbitrary units, display z-score)`,
    reference: 'Direct reference, recorded polarity (microvolts, display z-score)',
  };
  return <TeachingFigure id="E1" title="Three held-out correlations on one axis, and the waveforms behind them." category="Actual measured result">
    <div className="ic-bars">
      {results.map(item => <div className="ic-bar-row" key={item.method}>
        <span>{item.method === 'channel' ? `Channel ${item.chosen}` : `${item.method} ${item.chosen}`}</span>
        <div className="ic-bar-line">
          <i style={{ left: `${item.developmentAbs * 100}%` }} title={`development |r| ${format(item.developmentAbs)}`} />
          <b style={{ left: `${item.testAbs * 100}%` }} title={`held-out |r| ${format(item.testAbs)}`} />
        </div>
      </div>)}
      <div className="ic-bar-row">
        <span className="ic-note">absolute correlation</span>
        <div className="ic-bar-scale"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1</span></div>
      </div>
    </div>
    <div className="ic-lane-legend">
      <span><i style={{ border: '1.5px solid #8fc0a8', width: 10, height: 10, borderRadius: '50%' }} aria-hidden="true" />hollow: development |r|, which made the choice</span>
      <span><i style={{ background: '#e0b95f', width: 10, height: 10, borderRadius: '50%' }} aria-hidden="true" />solid: held-out |r|, reported for that frozen choice</span>
    </div>
    <DataTable stack caption="Every candidate's signed development correlation, the frozen choice, and the held-out result. Author calculations on this fixed extract, not a benchmark claim."
      headings={['Representation', 'chosen', 'development |r|', 'held-out |r|', 'signed development r, coordinates 1–4']}
      rows={results.map(item => [item.description, format(item.chosen), format(item.developmentAbs), format(item.testAbs),
        item.development.map(value => signedFormat(value, 6)).join('  ')])} />
    <h4>Aligned traces, 16 to 18 seconds</h4>
    <div className="ic-trace-rows">
      {['reference', 'channel', 'PCA', 'ICA'].map(name => <TraceRow key={name} name={name} title={rowTitle[name]}
        series={trace.series[name]} limit={limit}
        unitNote={name === 'reference' ? 'Recorded polarity is kept.' : 'Sign-aligned by its development correlation, frozen before the test interval.'} />)}
    </div>
    <div className="ic-bar-scale"><span>16.0 s</span><span>16.5</span><span>17.0</span><span>17.5</span><span>18.0 s</span></div>
    <p className="ic-note">Each row is centred and divided by its own standard deviation inside the displayed interval: a display z-score, not the fitted model&apos;s normalization, and no fitted parameter is changed by it. The plot shows {trace.columns} peak-preserving columns, each the smallest and largest of {trace.samplesPerColumn} consecutive samples; correlations above use all full-resolution samples in their respective development or test intervals (4,000 each), not the plotted envelopes. A shared vertical range of ±{format(limit, 1)} display units keeps the four rows comparable.</p>
    <DataTable dense caption="The first 20 instants of the held-out interval, exactly as computed. Channel values are microvolts; PCA and ICA scores are in arbitrary units."
      headings={['second', 'reference µV', `channel ${results[0].chosen} µV`, `PCA ${results[1].chosen}`, `ICA ${results[2].chosen}`]}
      rows={exact.map(row => [format(row.second, 3), format(row.reference, 3), format(row.channel, 3), format(row.PCA, 3), format(row.ICA, 3)])} />
    <div className="ic-provenance">
      <p><strong>Data.</strong> {ICA_RECORDING.dataset}, record {ICA_RECORDING.record}, first 20 seconds at {ICA_RECORDING.samplingHz} Hz. {ICA_RECORDING.attribution} <a href={ICA_RECORDING.datasetUrl}>Dataset page</a> · <a href={ICA_RECORDING.licenseUrl}>{ICA_RECORDING.license}</a> · <a href={ICA_RECORDING.csv}>download the extract</a> · <a href={ICA_RECORDING.provenance}>provenance, calibration and hashes</a>.</p>
      <p className="ic-note">SHA-256 {ICA_RECORDING.sha256}. Calibration: {ICA_RECORDING.calibration}. Settings: {ICA_RECORDING.settings} ICA converged in {ICA_RECORDING.iterations} iterations. Computed with Python {ICA_RECORDING.versions.python}, NumPy {ICA_RECORDING.versions.numpy}, SciPy {ICA_RECORDING.versions.scipy} and scikit-learn {ICA_RECORDING.versions['scikit-learn']}; another library version can change the rounding or the component identity.</p>
      <ul>{limits.map(limitText => <li key={limitText} className="ic-note">{limitText}</li>)}</ul>
    </div>
  </TeachingFigure>;
}
