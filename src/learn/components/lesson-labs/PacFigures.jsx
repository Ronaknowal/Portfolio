import { useId, useState } from 'react';
import {
  BitRow, NumberLine, PlotFrame, Series, Table, asInput, fixed, round,
} from './PacShared.jsx';
import {
  boundGrid, boundPlotGeometry, candidateBandGeometry, diagonalGeometry, eliminationTrace,
  finiteClassFailureBound,
  finiteFamilyRadii, finiteRadius, fixtures, ghostCollapse, growthPlotGeometry, growthTable,
  interiorCombination, intervalExperiment, intervalPatterns, learningCurveGeometry, meterGeometry,
  planePanelGeometry, realizableSufficientTable, simulationPlotGeometry, sinePlotGeometry, stripGeometry,
  twoStripBound, vcRadius,
} from '../../data/pac-models.js';
import { pacData } from '../../data/pac-data.js';

/** Inline figures for the PAC/VC lesson.
 *
 * Every figure draws from `pac-models.js`, never from a literal copied out of
 * the prose, so the verifier checks the same numbers and the same geometry the
 * reader sees. A drawn bound curve and a drawn separator are mathematical
 * claims: their coordinates are computed in the model layer and asserted there.
 *
 * Figures carry no graded prediction. The three investigations own that
 * contract, and a figure that quietly answered one of their questions would
 * break it -- which is why figure 2 deliberately uses a different target from
 * the investigation beneath it.
 */

const percent = (value, digits = 1) => `${Number((100 * value).toFixed(digits))}%`;

/* ------------------------------------------- F1 · two levels of randomness */

export function TwoLevelsFigure() {
  const [epsilon, setEpsilon] = useState(0.1);
  const geometry = meterGeometry({ epsilon });
  const titleId = useId();
  const descriptionId = useId();
  const rowHeight = 26;
  const height = geometry.rows.length * rowHeight + 42;
  const width = 290;
  const left = 96;
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 1.</strong> Two different levels of randomness. Each row is one
      independently drawn training sample fed to the same algorithm; the meter beside it is that fitted rule's
      error rate on <em>fresh</em> examples. ε is where the meter has to stop. δ counts <em>rows</em>, not
      wrong predictions.</p>
    <p className="pac-role is-constructed" role="status">Constructed schematic. The four error values are
      illustrative numbers chosen to show one failing draw; nothing here was measured or simulated.</p>
    <div className="pac-buttons">
      <span>ε</span>
      {[0.05, 0.1, 0.15].map(value => <button key={value} type="button"
        className={epsilon === value ? 'is-selected' : undefined} onClick={() => setEpsilon(value)}>
        ε = {asInput(value)}
      </button>)}
    </div>
    <svg className="pac-plot" viewBox={`0 0 ${width} ${height}`} role="img"
      aria-labelledby={titleId} aria-describedby={descriptionId} style={{ maxWidth: `${width}px` }}>
      <title id={titleId}>Four training draws, each with its own fresh-example error meter</title>
      <desc id={descriptionId}>{`Four rows. ${geometry.rows.map(row =>
        `draw ${row.id} gives population error ${row.populationError}`).join('; ')}. `
        + `The cutoff is at ${epsilon}, and ${geometry.exceedingDraws} of ${geometry.totalDraws} rows `
        + 'sit beyond it.'}</desc>
      {geometry.rows.map((row, index) => {
        const y = 10 + index * rowHeight;
        return <g key={row.id}>
          <text className="pac-small" x={0} y={y + 11}>{`sample ${row.id} → rule`}</text>
          <rect className="pac-meter-track" x={left} y={y} width={geometry.width} height={geometry.height} />
          <rect className={`pac-meter-fill${row.exceeds ? ' is-over' : ''}`}
            x={left} y={y} width={Math.max(row.barWidth, 0.6)} height={geometry.height} />
          <text className={`pac-small${row.exceeds ? ' pac-muted' : ''}`}
            x={left + geometry.width + 4} y={y + 10}>{row.exceeds ? 'over ε' : 'within ε'}</text>
        </g>;
      })}
      <line className="pac-cutoff" x1={left + geometry.cutoffX} y1={4}
        x2={left + geometry.cutoffX} y2={geometry.rows.length * rowHeight + 8} />
      <text className="pac-small pac-axis-title" x={left + geometry.cutoffX} y={height - 26}
        textAnchor="middle">{`ε = ${asInput(epsilon)}`}</text>
      {/* The axis name is its own end-anchored label on the right, and the two
          scale values sit under their own ends. Running "0" straight into
          "0.25 error on fresh examples" read as one string. */}
      <text className="pac-small" x={left} y={height - 15}>0</text>
      <text className="pac-small" x={left + geometry.width} y={height - 15}
        textAnchor="middle">{asInput(geometry.axisMaximum)}</text>
      <text className="pac-small pac-axis-title" x={width} y={height - 3} textAnchor="end">
        error on fresh examples
      </text>
    </svg>
    <p className="pac-caption">
      At ε = {asInput(epsilon)}, {geometry.exceedingDraws} of the {geometry.totalDraws} drawn samples produce a
      rule whose population error is above the cutoff. δ is a bound on how often that row can appear over
      training draws. It is not the fraction of predictions a successful rule gets wrong, and no amount of
      inspecting one fitted model's training score identifies which row you are on.
    </p>
  </div>;
}

/* --------------------------------- F2 · eliminating rules by their mistakes */

const ELIMINATION = { target: [0, 1, 1, 0], sample: [1, 3, 0] };

export function EliminationFigure() {
  const trace = eliminationTrace(ELIMINATION);
  const [step, setStep] = useState(0);
  const current = trace.steps[step];
  const bound = finiteClassFailureBound(16, current.taken, 0.25);
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 2.</strong> Sixteen rules are fixed <em>before</em> any
      observation. Each sampled input reveals one true label and crosses out every rule that disagrees with
      it. What survives is protected as a set; the rule the learner returns is protected because it is one of
      the survivors, not because anything was proved about it alone.</p>
    <p className="pac-role is-constructed" role="status">A constructed four-input world with target
      {' '}<BitRow bits={ELIMINATION.target} />. This is deliberately a <em>different</em> target from the
      investigation below, so that stepping through this figure does not answer its opening question.</p>
    <div className="pac-buttons">
      <button type="button" onClick={() => setStep(0)} disabled={step === 0}>Start</button>
      <button type="button" onClick={() => setStep(value => Math.max(0, value - 1))} disabled={step === 0}>Back</button>
      <button type="button" className="is-primary" disabled={step === trace.steps.length - 1}
        onClick={() => setStep(value => Math.min(trace.steps.length - 1, value + 1))}>Sample one more input</button>
      <span>{current.taken} of {trace.sample.length} observations used{current.taken
        ? `, at inputs ${current.observed.join(', ')}` : ''}.</span>
    </div>
    <Table
      caption={`All sixteen predeclared rules. ${current.survivors} still agree with every label seen so far.`}
      headings={['Rule', 'Disagrees with the target at', 'Population risk', 'After this many observations']}
      rowClass={index => (trace.steps[step].rules[index].consistent
        ? (trace.steps[step].rules[index] === current.selected ? 'is-leading' : undefined)
        : 'is-eliminated')}
      rows={current.rules.map(entry => [
        <BitRow key={entry.ruleText} bits={entry.rule} highlight={entry.wrong} />,
        entry.wrong.length ? `inputs ${entry.wrong.join(', ')}` : 'nowhere',
        fixed(entry.risk, 2),
        entry.consistent ? (entry === current.selected ? 'survives; returned' : 'survives') : 'crossed out',
      ])}
      footnote={'Highlighted bits are where a rule differs from the target. Population risk is the number of '
        + 'those bits divided by four, because the four inputs are equally likely.'} />
    <p className="pac-caption">
      An individual bad rule survives n independent draws only by missing its error region every time, which
      has probability at most (1 − ε)ⁿ. The union bound multiplies that by the whole predeclared family:
      at {current.taken} observation{current.taken === 1 ? '' : 's'} and ε = .25 it gives
      16 e<sup>−{asInput(0.25 * current.taken)}</sup> = {fixed(bound.raw, 6)}
      {bound.vacuous
        ? ', which is above one and so restricts nothing at all. A valid bound that says nothing is not a wrong bound.'
        : `, below one and therefore a real restriction. ${current.badSurvivors} of the survivors still carry risk above .25.`}
    </p>
  </div>;
}

/* ------------------------- F3 · one simultaneous event over a fixed family */

export function CandidateBandFigure() {
  const [revealed, setRevealed] = useState(false);
  const [shrunk, setShrunk] = useState(false);
  const geometry = candidateBandGeometry();
  const titleId = useId();
  const descriptionId = useId();
  const width = 300;
  const left = 34;
  const height = geometry.height + 30;
  const shrunkRadius = geometry.shrunkRadius;
  const scale = geometry.scale;
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 3.</strong> Twenty-five predictors fixed before the evaluation
      sample, each scored on the same 500 examples. Every row gets an interval of half-width
      {' '}{fixed(geometry.radius, 12)}, and one event covers all twenty-five at once — which is what makes it
      legitimate to look at the scores and then pick.</p>
    <p className="pac-role is-constructed" role="status">Constructed error counts, out of 500. No model was
      fitted and no data collected for this figure; the radius is the theorem's expression evaluated at
      K = {geometry.k}, n = {geometry.n}, δ = {asInput(geometry.delta)}.</p>
    <div className="pac-buttons">
      <button type="button" className={revealed ? 'is-selected' : 'is-primary'}
        onClick={() => setRevealed(value => !value)}>
        {revealed ? 'Hide which row was selected' : 'Now pick the lowest observed error'}
      </button>
      <button type="button" className={shrunk ? 'is-selected' : undefined}
        onClick={() => setShrunk(value => !value)}>
        {shrunk ? 'Remove the K = 3 band' : 'Try charging for only the 3 finalists'}
      </button>
    </div>
    <svg className="pac-plot" viewBox={`0 0 ${width} ${height}`} role="img"
      aria-labelledby={titleId} aria-describedby={descriptionId} style={{ maxWidth: `${width}px` }}>
      <title id={titleId}>Twenty-five simultaneous confidence intervals on one evaluation sample</title>
      <desc id={descriptionId}>{`Twenty-five horizontal intervals, each of half-width `
        + `${geometry.radius.toFixed(6)} around an observed error between ${geometry.rows[0].empirical} and `
        + `${geometry.rows[geometry.rows.length - 1].empirical}.`
        + (revealed ? ` The lowest observed error is ${geometry.best.empirical}.` : '')}</desc>
      {geometry.rows.map(row => <g key={row.name}
        className={revealed && row.selected ? 'is-selected' : undefined}>
        <line className={`pac-interval-band${revealed && row.selected ? ' is-selected' : ''}`}
          x1={left + row.lowX} y1={row.y} x2={left + row.highX} y2={row.y} />
        <line className={`pac-cap${revealed && row.selected ? ' is-selected' : ''}`}
          x1={left + row.lowX} y1={row.y - 3} x2={left + row.lowX} y2={row.y + 3} />
        <line className={`pac-cap${revealed && row.selected ? ' is-selected' : ''}`}
          x1={left + row.highX} y1={row.y - 3} x2={left + row.highX} y2={row.y + 3} />
        <circle className={`pac-interval-dot${revealed && row.selected ? ' is-selected' : ''}`}
          cx={left + row.centreX} cy={row.y} r={2.2} />
        {revealed && row.selected && <text className="pac-small pac-strong"
          x={left + row.highX + 4} y={row.y + 3}>selected</text>}
      </g>)}
      {shrunk && <line className="pac-interval-band is-shrunk"
        x1={left + scale(geometry.best.empirical - shrunkRadius)} y1={geometry.height + 8}
        x2={left + scale(geometry.best.empirical + shrunkRadius)} y2={geometry.height + 8} />}
      {/* The sentence that used to sit here is now HTML below the drawing. A
          fifty-five-character label inside a 300-unit viewBox ran off the right
          edge and across two axis labels; prose belongs in reflowing markup and
          the SVG keeps geometry and short labels. */}
      {/* Beside its own band, not at the left margin: at the margin it sat four
          units above the axis-range label and overlapped it. */}
      {shrunk && <text className="pac-small" x={left + scale(geometry.best.empirical + shrunkRadius) + 5}
        y={geometry.height + 11} textAnchor="start">K = 3</text>}
      <text className="pac-small" x={left} y={height - 4}>{round(scale.domain[0], 3)}</text>
      <text className="pac-small" x={width - 2} y={height - 4} textAnchor="end">
        {round(scale.domain[1], 3)} observed error
      </text>
    </svg>
    {shrunk && <p className="pac-note" role="status">
      The narrow dashed band is what K = 3 would give: ±{fixed(shrunkRadius, 6)} instead
      of ±{fixed(geometry.radius, 6)}. It is drawn dashed and set apart from the rows because it is not a band
      anyone here is entitled to: the search really did range over
      twenty-five candidates; discarding twenty-two of them after reading their scores does not turn that into
      a search over three.
    </p>}
    <p className="pac-caption">
      The widest information in this picture is that the intervals overlap. The lowest and second-lowest
      observed errors differ by {fixed(geometry.separation, 3)}, which is less than twice the radius
      ({fixed(2 * geometry.radius, 6)}), so the simultaneous event covers the selected row without settling
      which predictor is actually better. A single fixed rule on the same 500 examples would get
      ±{fixed(geometry.singleRadius, 12)}; selecting among {geometry.k} costs the difference. Discarding
      twenty-two candidates after seeing their scores does not turn the search into a search over three.
    </p>
  </div>;
}

/* ------------------------------------------ F4 · half-planes and shattering */

function PlanePanel({ points, labels, witness, caption, extra, size = 96 }) {
  const panel = planePanelGeometry({ points, labels, witness, size });
  const titleId = useId();
  const descriptionId = useId();
  return <figure className="pac-figure-inline">
    <svg className="pac-panel" viewBox={`0 0 ${size} ${size}`} role="img" aria-labelledby={titleId} aria-describedby={descriptionId}
      style={{ maxWidth: `${size}px` }}>
      <title id={titleId}>{caption}</title>
      <desc id={descriptionId}>{`Points labelled ${labels.join('')}. `
        + (panel.separator?.kind === 'line'
          ? 'A straight separator is drawn between the two groups.'
          : panel.separator?.kind === 'constant'
            ? 'No line is drawn: this rule gives every point the same label.'
            : 'No separator exists for this labeling.')}</desc>
      {panel.separator?.kind === 'line' && <line className="pac-separator"
        x1={panel.separator.from[0]} y1={panel.separator.from[1]}
        x2={panel.separator.to[0]} y2={panel.separator.to[1]} />}
      {extra && extra(panel)}
      {panel.marks.map((mark, index) => <g key={index}
        className={`pac-plane-point is-${mark.label === 1 ? 'positive' : 'negative'}`}>
        {mark.label === 1
          ? <circle cx={mark.cx} cy={mark.cy} r={3.6} />
          : <rect x={mark.cx - 3.2} y={mark.cy - 3.2} width={6.4} height={6.4} />}
      </g>)}
    </svg>
    <p className="pac-caption">{caption}</p>
  </figure>;
}

export function HalfPlaneFigure() {
  const triangle = pacData.halfPlaneWitnesses.triangle;
  const square = pacData.halfPlaneWitnesses.square;
  const interior = pacData.halfPlaneWitnesses.interior;
  const collinear = pacData.halfPlaneWitnesses.collinear;
  const alternating = [0, 1, 0, 1];
  const squarePanel = planePanelGeometry({ points: square.points, labels: alternating, witness: null, size: 96 });
  const diagonals = diagonalGeometry({ points: square.points, labels: alternating, panel: squarePanel });
  const combination = interiorCombination({
    triangle: [interior.points[0], interior.points[1], interior.points[2]],
    interior: interior.points[3],
  });
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 4.</strong> A triangle is shattered; a quadrilateral and an
      interior point are not. Filled circles are labelled 1, open squares 0. The gold line is an actual
      separator, computed and re-checked: every circle lies strictly on one side of it and every square
      strictly on the other.</p>
    {/* A grid, not a flex row: `flex: 1 1 260px` gave each 88-unit panel a
        260-pixel track, so eight small drawings sprawled over four rows with
        their captions stranded at the left of empty space. */}
    <div className="pac-panel-grid is-eight">
      {triangle.realized.map(entry => <PlanePanel key={entry.labels.join('')}
        points={triangle.points} labels={entry.labels} witness={entry}
        caption={`${entry.labels.join('')}`} size={88} />)}
    </div>
    <p className="pac-caption">
      All {triangle.count} labelings of the noncollinear triple have a witness, so this set is shattered and the
      VC dimension of affine half-planes is at least 3. Two of the eight need no line at all: when every point
      takes the same label the rule is a constant sign, and the panel draws no separator rather than inventing
      one.
    </p>
    <div className="pac-panel-grid is-wide">
      <PlanePanel points={square.points} labels={alternating} witness={null}
        caption="Convex quadrilateral, labels 0101: impossible"
        extra={() => <>
          <polyline className="pac-diagonal"
            points={diagonals.positive.map(point => `${point[0].toFixed(2)},${point[1].toFixed(2)}`).join(' ')} />
          <polyline className="pac-diagonal"
            points={diagonals.negative.map(point => `${point[0].toFixed(2)},${point[1].toFixed(2)}`).join(' ')} />
        </>} />
      <PlanePanel points={interior.points} labels={[1, 1, 1, 0]} witness={null}
        caption="A point inside the triangle, labelled 0: impossible"
        extra={panel => <polygon className="pac-hull"
          points={panel.marks.slice(0, 3).map(mark => `${mark.cx.toFixed(2)},${mark.cy.toFixed(2)}`).join(' ')} />} />
      <PlanePanel points={collinear.points} labels={[1, 0, 1]} witness={null}
        caption="A collinear triple, labels 101: impossible" />
    </div>
    <p className="pac-caption">
      The two diagonals of the quadrilateral {diagonals.crosses ? 'cross' : 'do not cross'}, so the positive
      pair and the negative pair cannot be pulled apart by any straight line. The interior point sits at the
      convex combination {combination.weights.map(weight => round(weight, 3)).join(' + ')} of the three
      vertices around it, so making those three positive forces it positive too. The collinear triple realizes
      only {collinear.count} of its 8 labelings. The quadrilateral loses
      exactly {square.infeasible.length}: {square.infeasible.map(labels => labels.join('')).join(' and ')}.
    </p>
    <p className="pac-note">
      These are checks on four specific configurations, drawn from computed witnesses with verified margins.
      They establish the lower bound, that <em>some</em> triple is shattered. The upper bound — that
      <em> every</em> four-point set has an impossible labeling — is the geometric argument in the prose, and no
      number of failed searches would substitute for it.
    </p>
  </div>;
}

/* ------------------------------------------------- F5 · the growth function */

const GROWTH_LABELS = {
  thresholds: 'increasing thresholds, n + 1',
  intervals: 'one interval, 1 + n(n+1)/2',
  sauerD2: "Sauer's bound at d = 2",
  allBinary: 'every binary pattern, 2ⁿ',
};

export function GrowthFigure() {
  const geometry = growthPlotGeometry();
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 5.</strong> How many distinct label patterns each class can make
      on n points. The vertical axis is logarithmic, because 2ⁿ reaches 1,024 while the interval count reaches
      56 — on a linear axis the two structured classes would both be a flat line along the bottom.</p>
    <PlotFrame geometry={geometry} xLabel="n points" yLabel="patterns"
      caption="Pattern counts against the number of points, on a logarithmic vertical axis"
      describe={'Four series. Every binary pattern doubles each step to 1024 at n=10. Sauer\'s d=2 bound and '
        + 'the interval count coincide at 56. Thresholds reach only 11.'}
      yTickText={value => value.toLocaleString('en-US')}>
      {geometry.series.map(series => <Series key={series.key} className={`is-${series.key}`}
        points={series.points} marker={series.key === 'allBinary' ? 'square' : 'dot'} />)}
    </PlotFrame>
    <div className="pac-legend">
      {Object.entries(GROWTH_LABELS).map(([key, text]) => <span key={key}>
        <i className={`pac-swatch is-${key}`} aria-hidden="true" />{text}
      </span>)}
    </div>
    <Table caption="The exact counts. Sauer's bound at d = 2 and the interval class agree at every n here."
      headings={['n', 'every pattern 2ⁿ', 'thresholds', 'intervals', "Sauer's bound, d = 2"]}
      rows={growthTable.map(row => [
        String(row.n), row.allBinary.toLocaleString('en-US'), String(row.thresholds),
        String(row.intervals), String(row.sauerD2),
      ])}
      footnote={`At n = 5 the interval class realizes ${growthTable[4].intervals} of the `
        + `${growthTable[4].allBinary} possible patterns, which is ${percent(growthTable[4].intervals / growthTable[4].allBinary)}; `
        + `at n = 10 it realizes ${growthTable[9].intervals} of ${growthTable[9].allBinary.toLocaleString('en-US')}, `
        // Five decimals, because 56/1024 is exactly 5.46875% and printing 5.4688%
        // would round an exact fraction for no reason.
        + `or ${percent(growthTable[9].intervals / growthTable[9].allBinary, 5)}. The numerator grows; the `
        + 'denominator grows faster. Read both.'} />
  </div>;
}

/* ------------------------------------- F6 · the ghost sample and collapse */

export function GhostSampleFigure() {
  const collapse = ghostCollapse();
  const width = 290;
  const titleId = useId();
  const descriptionId = useId();
  const height = 74;
  const scale = value => 18 + value * (width - 36);
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 6.</strong> The proof device. A second, independent sample of the
      same size is imagined alongside the real one; the infinitely many intervals are then restricted to the
      combined inputs, where only finitely many distinct prediction rows remain.</p>
    <svg className="pac-plot" viewBox={`0 0 ${width} ${height}`} role="img"
      aria-labelledby={titleId} aria-describedby={descriptionId} style={{ maxWidth: `${width}px` }}>
      <title id={titleId}>A training sample, a ghost sample, and their combined inputs</title>
      <desc id={descriptionId}>{`Training inputs at ${collapse.training.join(', ')}; ghost inputs at `
        + `${collapse.ghost.join(', ')}; the combined ${collapse.combinedSize} inputs below them.`}</desc>
      <text className="pac-small" x={0} y={14}>training</text>
      {collapse.training.map(value => <circle key={`t${value}`} className="pac-interval-dot"
        cx={scale(value)} cy={22} r={3} />)}
      <text className="pac-small" x={0} y={40}>ghost</text>
      {collapse.ghost.map(value => <rect key={`g${value}`} className="pac-ghost-mark"
        x={scale(value) - 2.8} y={45} width={5.6} height={5.6} />)}
      <text className="pac-small" x={0} y={66}>combined</text>
      {collapse.combined.map(entry => (entry.from === 'training'
        ? <circle key={`c${entry.x}`} className="pac-interval-dot" cx={scale(entry.x)} cy={70} r={3} />
        : <rect key={`c${entry.x}`} className="pac-ghost-mark" x={scale(entry.x) - 2.8} y={67} width={5.6} height={5.6} />))}
      <line className="pac-axis" x1={18} y1={height - 1} x2={width - 18} y2={height - 1} />
    </svg>
    <p className="pac-caption">
      On the {collapse.combinedSize} combined inputs, the interval class realizes
      exactly {collapse.patternCount} distinct rows out of {collapse.allBinary.toLocaleString('en-US')} possible
      ones — the closed form 1 + n(n+1)/2 gives the same {collapse.patternCountFormula}. Every interval whose
      endpoints fall between the same two inputs collapses to one row. On the training inputs alone there
      are {collapse.trainingOnlyPatternCount}, and that smaller, sample-dependent number is precisely what may
      not be substituted for K in the finite-family theorem: it was chosen by the data.
    </p>
    <Table caption="Six of the rows the class realizes on the combined inputs, in order"
      headings={['Row', 'Pattern on the combined inputs']}
      rows={collapse.sample.map((pattern, index) => [
        `row ${index + 1}`, <BitRow key={index} bits={pattern} />,
      ])}
      footnote={'The random exchange between the two samples that makes the counting argument work is part of '
        + 'the proof, not part of any algorithm. A learner never obtains the ghost sample.'} />
  </div>;
}

/* --------------------------------------------- F7 · the explicit VC bound */

export function BoundCurveFigure() {
  const geometry = boundPlotGeometry();
  const sufficient = realizableSufficientTable();
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 7.</strong> The conservative explicit uniform-convergence radius
      √(32[d ln(en/d) + ln(8/δ)]/n) at δ = {asInput(boundGrid.delta)}. The shaded region is everything above 1:
      a 0–1 risk gap is already bounded by 1, so a radius up there is <em>vacuous</em> — valid, and adding
      nothing.</p>
    <PlotFrame geometry={geometry} xLabel="n examples (log)" yLabel="radius"
      caption="The raw radius against sample size for three VC dimensions, with the vacuous region marked"
      describe={'Three falling curves. At n=100 all three are above 1. The d=1 curve falls below 1 first; the '
        + 'd=10 curve is still above 1 at n=1000.'}
      xTickText={value => value.toLocaleString('en-US')}>
      <rect className="pac-vacuous" x={geometry.padding.left} y={geometry.vacuousTo}
        width={geometry.width - geometry.padding.left - geometry.padding.right}
        height={Math.max(geometry.vacuousFrom - geometry.vacuousTo, 0.6)} />
      {/* The sentence that explained this shading used to sit inside it. The
          d = 10 curve ran through the text for 2.5% of its length -- caught by
          sampling the curve's own geometry, which the straight-line inspector
          cannot see. It is a legend entry now; prose reflows, curves do not. */}
      {geometry.series.map(series => <Series key={series.d} className={`is-d${series.d}`} points={series.points} />)}
    </PlotFrame>
    <div className="pac-legend">
      {geometry.series.map(series => <span key={series.d}>
        <i className={`pac-swatch is-d${series.d}`} aria-hidden="true" />d = {series.d}
      </span>)}
      <span><i className="pac-swatch is-vacuous" aria-hidden="true" />shaded: radius above 1, no stronger than
        the trivial 0–1 range</span>
    </div>
    <Table caption="The same expression evaluated exactly, at the four sizes the lesson quotes"
      headings={['d', 'n', 'raw radius', 'stronger than the trivial bound?']}
      rows={geometry.table.map(row => [
        String(row.d), row.n.toLocaleString('en-US'), fixed(row.radius, 6),
        row.radius < 1 ? 'yes' : 'no — vacuous',
      ])}
      footnote={'The raw value is shown, never clipped to 1. Clipping would hide exactly the fact this table '
        + 'exists to show.'} />
    <Table caption="A different question, on its own axis: the classical realizable sufficient sample size"
      headings={['d', 'ε', 'δ', 'sufficient n']}
      rows={sufficient.map(row => [
        String(row.d), asInput(row.epsilon), asInput(row.delta), row.sufficientN.toLocaleString('en-US'),
      ])}
      footnote={'max{(4/ε)log₂(2/δ), (8d/ε)log₂(13/ε)}, with base-2 logarithms as in the original paper. This '
        + 'is a sufficient size, not a minimum, and it answers a different question from the radius above. The '
        + 'two are never drawn on one axis.'} />
  </div>;
}

/* ------------------------------------------------- F8 · the boundary strips */

export function BoundaryStripFigure() {
  const experiment = intervalExperiment(fixtures.interval);
  const geometry = stripGeometry({ experiment });
  const strip = twoStripBound(experiment.epsilon, experiment.points.length);
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 8.</strong> The target [{asInput(experiment.target[0])},
      {' '}{asInput(experiment.target[1])}], the fitted interval, and the two strips the fit missed. Filled
      circles are positive observations, open squares negative.</p>
    <NumberLine geometry={geometry} showStrips
      caption={`Fitted [${experiment.interval[0]}, ${experiment.interval[1]}] inside the target; the two hatched `
        + 'strips are where it is wrong'}
      describe={'Five rows over one axis from 0 to 1. Top: the six observations, positive ones as filled '
        + 'circles at .35, .55 and .65. Then the target region from .3 to .7. Then the two coverage strips of '
        + 'width epsilon over two just inside each target edge. Then the two hatched pieces the fit misses, '
        + '.3 to .35 and .65 to .7. Then the fitted region from .35 to .65.'} />
    <div className="pac-strip">
      <span className="pac-strip-cell">training error<b>{fixed(experiment.empiricalRisk, 6)}</b></span>
      <span className="pac-strip-cell is-used">population risk<b>{fixed(experiment.risk, 6)}</b></span>
      <span className="pac-strip-cell">missed lengths<b>{experiment.segments.map(segment =>
        round(segment.to - segment.from, 6)).join(' + ')}</b></span>
    </div>
    <p className="pac-caption">
      A fit that gets every training label right still has population risk {fixed(experiment.risk, 6)}: the two
      missed strips have total length {fixed(experiment.segmentTotal, 6)}, and under a <em>uniform</em> input
      distribution a length is a probability. Change the input distribution and that identity goes with it;
      the two strips would still be the disagreement, but their probability would have to be integrated.
    </p>
    <p className="pac-caption">
      The stippled strips in the third row are the two coverage strips of width ε/2 = {asInput(experiment.epsilon / 2)}
      {' '}inside each edge of the target. Landing at least one observation in <em>each</em> of them is enough to
      force risk at most ε; the failure bound min{'{'}1, 2(1 − ε/2)ⁿ{'}'} at n = {experiment.points.length}
      {' '}is {fixed(strip.clipped, 6)}{strip.vacuous ? ', which is the trivial value 1' : ''}. It is a
      sufficient event, not a necessary one: a sample can do well with one strip empty.
    </p>
  </div>;
}

/* ---------------------------------------- F9 · risks across training draws */

export function SimulationFigure() {
  const rows = pacData.simulation.rows;
  const geometry = simulationPlotGeometry({ rows });
  const [preview, setPreview] = useState(false);
  const previewRow = rows[1];
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 9.</strong> One thousand independent training draws at each
      size. The vertical bar spans the 5th to 95th percentile of the fitted rules' <em>exact</em> risks, the
      crossbar is the median and the diamond the mean. The dashed line is ε = {asInput(pacData.simulation.epsilon)}.</p>
    <p className="pac-role is-measured" role="status">A retained simulation, NumPy default_rng
      seed {pacData.simulation.seed}. Each run's risk is exact geometry; the failure counts across runs are
      Monte Carlo estimates of a probability, not the probability.</p>
    <PlotFrame geometry={geometry} xLabel="training size n (log)" yLabel="risk (log)"
      caption="Risk quantiles of the fitted interval across 1,000 draws at each of five training sizes"
      describe={'Five vertical bars falling from left to right on a logarithmic risk axis. At n=10 the bar '
        + 'spans .043 to .398 and its median sits above the epsilon line; by n=200 it spans .002 to .024, an '
        + 'order of magnitude below it.'}
      xTickText={value => String(value)}
      yTickText={value => (value >= 0.1 ? String(value) : value.toFixed(3))}>
      <line className="pac-cutoff" x1={geometry.padding.left} y1={geometry.epsilonY}
        x2={geometry.width - geometry.padding.right} y2={geometry.epsilonY} />
      {geometry.bars.map(bar => <g key={bar.n}>
        <line className="pac-interval-band" x1={bar.x} y1={bar.lowY} x2={bar.x} y2={bar.highY} />
        <line className="pac-cap is-selected" x1={bar.x - 5} y1={bar.medianY} x2={bar.x + 5} y2={bar.medianY} />
        <rect className="pac-interval-dot is-selected" x={bar.x - 2.6} y={bar.meanY - 2.6}
          width={5.2} height={5.2} transform={`rotate(45 ${bar.x} ${bar.meanY})`} />
      </g>)}
      {/* End-anchored at the right: at the left it sat across the first bar. */}
      <text className="pac-small pac-muted" x={geometry.width - geometry.padding.right}
        y={geometry.epsilonY - 4} textAnchor="end">
        ε = {asInput(pacData.simulation.epsilon)}
      </text>
    </PlotFrame>
    <Table caption="The retained summary. The bound and the observed frequency are different objects."
      headings={['n', 'draws above ε', 'of', 'mean risk', '5th / 50th / 95th percentile', 'two-strip bound']}
      rows={rows.map(row => [
        String(row.n), String(row.failureCount), row.repetitions.toLocaleString('en-US'),
        fixed(row.meanTrueError, 6), row.riskQuantiles.map(value => fixed(value, 4)).join(' / '),
        row.twoStripBound >= 1 ? '1, trivial' : fixed(row.twoStripBound, 7),
      ])}
      footnote={'The bound is a theorem evaluated at these sizes. The counts are one experiment. Zero failures '
        + 'in 1,000 draws at n = 200 does not make the failure probability zero, and the bound at that size is '
        + 'strictly positive.'} />
    <div className="pac-buttons">
      <button type="button" onClick={() => setPreview(value => !value)}>
        {preview ? 'Hide the retained individual risks' : `Show the ${pacData.simulation.previewSize} retained individual risks at n = ${previewRow.n}`}
      </button>
    </div>
    {preview && <p className="pac-caption">
      The first {pacData.simulation.previewSize} of the {previewRow.repetitions.toLocaleString('en-US')} risks at
      n = {previewRow.n}, exactly as recorded: {previewRow.previewTrueErrors.map(value => round(value, 4)).join(', ')}.
      Only these {pacData.simulation.previewSize} were retained, so this is a labelled sample of the run and not
      a histogram of it. Drawing a thousand-run distribution from twenty stored values would be an invention.
    </p>}
  </div>;
}

/* ----------------------------------------- F10 · measured learning curves */

export function LearningCurveFigure() {
  const { rows, sizes, labels, developmentN } = pacData.learningCurves;
  const geometry = learningCurveGeometry({ rows, sizes });
  const curveAtEighty = Object.fromEntries(rows.filter(row => row.n === 80).map(row => [row.model, row]));
  /* The default view is all three DEVELOPMENT curves together, because the
     claim this section makes -- that the procedure fitting every training label
     loses on held-out data -- is a comparison between procedures and was
     invisible while the figure showed one at a time. Selecting a procedure then
     adds its training curve beside its development curve. */
  const [shown, setShown] = useState('all');
  const visible = shown === 'all'
    ? geometry.series.filter(series => series.kind === 'development')
    : geometry.series.filter(series => series.model === shown);
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 10.</strong> Measured development error for all three procedures
      on nested training prefixes of the banknote pool; select one to add its training curve beside its
      development curve. Every view shares one axis, scaled to the largest error any of them makes, and the
      development denominator is a fixed {developmentN} rows at every size.</p>
    <p className="pac-role is-measured" role="status">Measured results from one nested sequence and one shared
      development set. These are not capacity estimates, not independent replicates, and no test row was
      evaluated.</p>
    <div className="pac-buttons">
      <button type="button" className={shown === 'all' ? 'is-selected' : undefined}
        onClick={() => setShown('all')}>All three, development only</button>
      {Object.entries(labels).map(([key, text]) => <button key={key} type="button"
        className={shown === key ? 'is-selected' : undefined} onClick={() => setShown(key)}>{text}</button>)}
    </div>
    <PlotFrame geometry={geometry} xLabel="training examples (log)" yLabel="error"
      caption={shown === 'all'
        ? 'Development error against training size, all three procedures on one axis'
        : `${labels[shown]}: training and development error against training size`}
      describe={visible.map(series => `${shown === 'all' ? labels[series.model] : series.kind} error goes `
        + series.points.map(point => point.value.toFixed(4)).join(', ')).join('; ')}
      xTickText={value => String(value)} yTickText={value => value.toFixed(2)}>
      {visible.map(series => <Series key={`${series.model}-${series.kind}`}
        className={shown === 'all' ? `is-${series.model}` : `is-${series.kind}`}
        points={series.points} marker={series.kind === 'train' ? 'square' : 'dot'} />)}
    </PlotFrame>
    <div className="pac-legend">
      {shown === 'all'
        ? Object.entries(labels).map(([key, text]) => <span key={key}>
          <i className={`pac-swatch is-${key}`} aria-hidden="true" />{text}
        </span>)
        : <>
          <span><i className="pac-swatch is-train" aria-hidden="true" />training error, squares</span>
          <span><i className="pac-swatch is-development" aria-hidden="true" />development error, circles</span>
        </>}
    </div>
    {shown === 'all' && <p className="pac-caption">
      At 80 training examples the depth-5 tree fits every one of its training labels and still gets
      {' '}{curveAtEighty.depth5_tree.developmentCorrect} of {curveAtEighty.depth5_tree.developmentN} development
      labels right, while the RBF SVC fits {curveAtEighty.rbf_svc.trainCorrect} of 80 and
      gets {curveAtEighty.rbf_svc.developmentCorrect}. Select a procedure to see its training curve beside its
      development curve.
    </p>}
    <Table caption="Every measured count. Training denominators vary with n; the development denominator does not."
      headings={['procedure', 'n', 'training correct', 'development correct', 'development error']}
      rows={rows.map(row => [
        labels[row.model], String(row.n), `${row.trainCorrect}/${row.n}`,
        `${row.developmentCorrect}/${row.developmentN}`,
        fixed(1 - row.developmentCorrect / row.developmentN, 4),
      ])}
      footnote={'One trajectory. There is no error band here because there is nothing to compute one from: a '
        + 'shared development set and nested prefixes do not produce independent replicates. The theorem '
        + 'curves in figure 7 answer a different question and are deliberately on a different axis.'} />
  </div>;
}

/* -------------------------------------- F11 · one parameter, all the labels */

export function SineFigure() {
  const geometry = sinePlotGeometry({ labels: fixtures.sineLabels });
  const witness = geometry.witness;
  const radius = geometry.radius;
  const panel = 2 * radius + 22;
  return <div className="pac-figure">
    <p className="pac-caption"><strong>Figure 11.</strong> One real parameter, every labeling. The requested
      labels are complemented, written as the first bits of a binary fraction, and read back as positions on a
      circle: the upper half is where the sine is positive.</p>
    <div className="pac-strip">
      <span className="pac-strip-cell">requested labels<b>{witness.labels.join('')}</b></span>
      <span className="pac-strip-cell">complemented bits<b>{witness.labels.map(bit => 1 - bit).join('')}</b></span>
      <span className="pac-strip-cell">plus the guard bits 01<b>{witness.rBits}</b></span>
      <span className="pac-strip-cell is-used">r = {witness.rText}, θ = 2πr<b>{fixed(witness.theta, 6)}</b></span>
    </div>
    {/* BOTH halves are painted on every dial, not just the one the mark lands
        in. Drawing only the occupied half made every dial look the same, so the
        figure could not show the thing it exists to show -- that the upper half
        is where the sine is positive. The two fills are also far enough apart
        to tell apart on this ground; the first pair was within a few points of
        the dial's own colour and read as unpainted. */}
    <div className="pac-panel-grid is-wide">
      {geometry.dials.map(dial => {
        const titleId = `pac-dial-${dial.index}`;
        const centre = panel / 2;
        return <figure className="pac-figure-inline" key={dial.index}>
          <svg className="pac-panel" viewBox={`0 0 ${panel} ${panel}`} role="img" aria-labelledby={titleId}
            style={{ maxWidth: `${panel}px` }}>
            <title id={titleId}>{`At x = ${dial.x} the cycle is ${dial.cycleText}, which lands in the `
              + `${dial.positive ? 'upper, positive' : 'lower, negative'} half`}</title>
            <path className="pac-dial-half is-positive"
              d={`M ${centre - radius} ${centre} A ${radius} ${radius} 0 0 1 ${centre + radius} ${centre} Z`} />
            <path className="pac-dial-half is-negative"
              d={`M ${centre - radius} ${centre} A ${radius} ${radius} 0 0 0 ${centre + radius} ${centre} Z`} />
            <circle className="pac-dial" cx={centre} cy={centre} r={radius} />
            <line className="pac-dial-axis" x1={centre - radius} y1={centre} x2={centre + radius} y2={centre} />
            <line className="pac-dial-mark" x1={centre} y1={centre}
              x2={centre + dial.markX} y2={centre + dial.markY} />
            <circle className="pac-dial-dot" cx={centre + dial.markX} cy={centre + dial.markY} r={2.8} />
            <text className="pac-small" x={centre} y={panel - 2} textAnchor="middle">{`x = ${dial.x}`}</text>
          </svg>
          <p className="pac-caption">{`r·x mod 1 = ${dial.cycleText}, `
            + `${dial.positive ? 'upper half' : 'lower half'} → ${dial.positive ? '1' : '0'}`}</p>
        </figure>;
      })}
    </div>
    <div className="pac-legend">
      <span><i className="pac-swatch is-sine-positive" aria-hidden="true" />upper half: sin(θx) ≥ 0, predicts 1</span>
      <span><i className="pac-swatch is-sine-negative" aria-hidden="true" />lower half: sin(θx) &lt; 0, predicts 0</span>
    </div>
    <p className="pac-caption">
      Multiplying r by {witness.points.join(', ')} shifts the binary point so that the i-th chosen bit becomes
      the first fractional bit. A leading 0 puts the cycle strictly inside (0, ½) where the sine is positive; a
      leading 1 puts it strictly inside (½, 1) where the sine is negative. The appended 01 is what keeps every
      cycle away from the boundaries, where the sign would be undecided. The result
      is {witness.predicted.join('')}, which is the request.
    </p>
    <p className="pac-note">
      Extending this to n labels needs inputs up to 2ⁿ⁻¹ and about n + 2 bits of precision in θ. That premise is
      the whole trick, and it is why a fixed finite-precision encoding of θ does not inherit the conclusion: a
      parameter stored in 64 bits names only finitely many rules.
    </p>
  </div>;
}

/* Every figure's numbers come from the model layer, and these are the values
 * the lesson body quotes beside them. Exported so the prose and the figure read
 * one object rather than two literals. */
export const figureValues = {
  selectionRadius: finiteFamilyRadii.selection,
  singleRadius: finiteFamilyRadii.single,
  shrunkRadius: finiteRadius(3, fixtures.finiteFamily.n, fixtures.finiteFamily.delta),
  vacuousAt: vcRadius(2, 100, 0.05),
  usefulAt: vcRadius(2, 100000, 0.05),
  intervalPatternsAtThree: intervalPatterns(3),
};
