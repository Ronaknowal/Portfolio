import {
  MISSING, barHeight, beliefTrack, countFlow, durationBars, enumeratePaths, fixtures, infer,
  insideWindow, objectiveWindows, parameterCount, pathShare, repeatedProduct, trellis,
  withRainyEmission,
} from '../../data/hmm-models.js';
import { emTrack, provenance } from '../../data/hmm-data.js';
import {
  BeliefBars, Diagram, Plot, Stack, StateGraph, StateKey, Table, Trellis, TrellisStep, TrellisTable,
  Undefined, exactly, fixed, polyline, round, statePattern,
} from './HmmShared.jsx';
import './hmm-labs.css';

/* Every proportion, length, coordinate, width and total drawn below comes out of
   hmm-models.js or hmm-data.js. Nothing here chooses a shape. */

const weather = fixtures.weather;
const reports = fixtures.reports;
const main = infer(weather, reports);

/* ================================================================ figure 1 */

/** The graph unrolled: one hidden node per time, observation cards below.
 *
 * Values are deliberately absent. Section 1 has not introduced a forward mass
 * yet, and putting numbers here would answer the question the next section asks.
 * What this drawing carries is the STRUCTURE: three transitions inside four
 * reports, and an emission hanging off every hidden node.
 */
function Unroll({ model, observations, path, label, describe }) {
  const width = 420;
  const left = 30;
  const right = 14;
  const nodeY = 56;
  const cardY = 116;
  const height = 168;
  const spacing = (width - left - right) / observations.length;
  const centre = time => left + spacing * (time + 0.5);
  const radius = 21;
  return <Diagram describe={describe}><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
    <title>{label}</title>
    {/* The two row captions moved into the surrounding HTML. In the drawing
        they were crossed by the emission connectors, and the layout contract
        puts an explanation that encodes no geometry in reflowing markup. */}
    {observations.map((value, time) => (time === 0 ? null : (
      <g key={`edge-${time}`}>
        <line className="hmm-edge is-chosen" strokeWidth="2.4" x1={centre(time - 1) + radius} y1={nodeY}
          x2={centre(time) - radius} y2={nodeY} />
        {/* Above the circles, not between them: the gap between two node edges
            is narrower than the label, so a centred label overlapped both. */}
        <text className="hmm-small hmm-halo hmm-muted" x={(centre(time - 1) + centre(time)) / 2} y={nodeY - 30}
          textAnchor="middle">transition {time}</text>
      </g>
    )))}
    {observations.map((value, time) => <g key={`node-${time}`}>
      <circle className="hmm-node is-on-path" cx={centre(time)} cy={nodeY} r={radius} />
      <line className={`hmm-key-line is-${statePattern(path[time])}`}
        x1={centre(time) - 12} y1={nodeY - 11} x2={centre(time) + 12} y2={nodeY - 11} />
      <text className="hmm-small hmm-halo hmm-strong" x={centre(time)} y={nodeY + 6} textAnchor="middle">
        {model.stateNames[path[time]].slice(0, 5)}
      </text>
      {/* An emission is a different kind of edge from a transition, so it is
          drawn in a different vocabulary rather than a thinner version. */}
      <line className="hmm-emission" x1={centre(time)} y1={nodeY + radius} x2={centre(time)} y2={cardY} />
      <rect className="hmm-card" x={centre(time) - 30} y={cardY} width="60" height="26" rx="3" />
      <text className="hmm-small hmm-halo" x={centre(time)} y={cardY + 17} textAnchor="middle">
        {value === MISSING ? 'missing' : model.symbolNames[value]}
      </text>
      <text className="hmm-small hmm-muted" x={centre(time)} y={cardY + 42} textAnchor="middle">t = {time}</text>
    </g>)}
  </svg></Diagram>;
}

export function GraphUnrollFigure() {
  const path = main.path;
  const joint = enumeratePaths(weather, reports).paths
    .find(entry => entry.path.every((state, time) => state === path[time]));
  return <figure className="hmm-figure">
    <figcaption>
      <strong>Figure 1 — Two rows, two kinds of arrow.</strong> The graph on the left is a conditional
      factorisation, not a claim that weather causes errands. A transition connects one hidden state to the next; an
      emission hangs an observation off a hidden state, and the two are drawn in different vocabularies because they
      are different conditional probabilities. Four reports contain <strong>three</strong> transitions.
    </figcaption>
    <StateKey names={weather.stateNames} prefix="hidden state" />
    <div className="hmm-panels is-pair">
      <div className="hmm-panel">
        <h4>The model as a graph</h4>
        <StateGraph model={weather} label="The two-state transition graph"
          describe={`Two hidden states. Rainy starts with probability ${weather.start[0]} and stays Rainy with probability ${weather.transition[0][0]}, moving to Sunny with probability ${weather.transition[0][1]}. Sunny starts with probability ${weather.start[1]}, moves to Rainy with probability ${weather.transition[1][0]} and stays Sunny with probability ${weather.transition[1][1]}.`} />
        <p>Each row of the transition table is its own distribution. Rainy→Sunny is not Sunny→Rainy, and neither is
          an emission.</p>
      </div>
      <div className="hmm-panel">
        <h4>The emission rows, separately</h4>
        <Table caption="Each state's distribution over the three reportable activities"
          headings={['state', ...weather.symbolNames]}
          rows={weather.stateNames.map((name, state) => [name, ...weather.emission[state].map(value => round(value, 4))])}
          footnote={`P(Clean | Rainy) = ${weather.emission[0][2]} says nothing about P(Rainy | Clean); the arrow does not give its own reverse.`} />
      </div>
    </div>
    <div className="hmm-panel">
      <h4>The same model unrolled over the four reports</h4>
      <p>Hidden states run along the top, joined by three transitions. The report you actually see hangs beneath
        each one on a connector of a different kind.</p>
      <Unroll model={weather} observations={reports} path={path}
        label="The unrolled sequence with one highlighted path"
        describe={`Four hidden nodes in a row, connected by three transitions, each with an observation card hanging beneath it. The highlighted path is ${path.map(state => weather.stateNames[state]).join(', then ')}, and the reports beneath are ${reports.map(value => weather.symbolNames[value]).join(', ')}.`} />
      <p>
        One highlighted story: {path.map(state => weather.stateNames[state]).join(' → ')}. Multiplying start, emit,
        transition, emit, transition, emit, transition, emit gives this one path a joint probability of{' '}
        {round(joint.joint, 8)}. The next sections add up all sixteen such stories without listing them.
      </p>
    </div>
  </figure>;
}

/* ================================================================ figure 2 */

export function ForwardTrellisFigure() {
  const built = trellis(weather, reports, 'sum');
  return <figure className="hmm-figure">
    <figcaption>
      <strong>Figure 2 — Every destination adds its incoming contributions.</strong> Edge width encodes the share
      of its destination's incoming total that the edge carries, and every exact amount is printed below and in the
      table. The value inside each node is a <em>joint</em> mass, P(reports so far and this state) — not a
      probability of the state on its own. Row identity is carried by the labels at the left and by their line
      patterns, so a node can spend its space on its number; on a narrow screen it keeps only its state initial
      and the values are read from the table.
    </figcaption>
    <StateKey names={weather.stateNames} prefix="hidden state" />
    <div className="hmm-panel">
      <h4>Forward trellis for Walk, Shop, Walk, Clean</h4>
      <Trellis model={weather} observations={reports} mode="sum" focusTime={1}
        label="The forward trellis, with the second step emphasised"
        describe={`A two-row trellis over four time steps. ${built.columns.map(column => `At time ${column.time}, report ${weather.symbolNames[column.observation]}, the forward masses are ${column.cells.map(cell => `${weather.stateNames[cell.state]} ${round(cell.value, 8)}`).join(' and ')}.`).join(' ')}`} />
      <TrellisStep model={weather} observations={reports} mode="sum" time={1} />
    </div>
    <TrellisTable model={weather} observations={reports} mode="sum"
      caption="Every forward cell, its column total, and the filtered belief that total normalises to" />
    <Table caption="Dividing a column by its own total turns joint mass into the filtered belief"
      headings={['time and report', ...weather.stateNames.map(name => `filtered ${name}`)]}
      rows={main.filtered.map((row, time) => [
        `${time} ${weather.symbolNames[reports[time]]}`,
        ...row.map(value => fixed(value, 6)),
      ])}
      footnote={`The whole sequence has probability ${round(built.final, 8)}. Notice that the Rainy mass rises at the last step while the column total falls: contributions moved between states, and no individual cell has to shrink monotonically.`} />
  </figure>;
}

/* ================================================================ figure 3 */

export function BeliefFigure() {
  const track = beliefTrack(weather, reports, 0);
  const changed = beliefTrack(weather, fixtures.correctedFinal, 0);
  return <figure className="hmm-figure">
    <figcaption>
      <strong>Figure 3 — The same state, two questions.</strong> Filtering conditions on the reports up to that
      time; smoothing conditions on all four. Both bars share one 0 to 1 scale, so their heights are comparable, and
      each carries its exact value. The last pair agrees exactly, because at the last time there is no later report
      to add.
    </figcaption>
    <div className="hmm-panel">
      <h4>P(Rainy) under the original reports</h4>
      <BeliefBars rows={track.rows} stateName="Rainy" label="Filtered and smoothed probability of Rainy"
        describe={track.rows.map(row => `At time ${row.time}, report ${row.label}, filtering gives ${round(row.filtered, 6)} and smoothing gives ${round(row.smoothed, 6)}.`).join(' ')} />
    </div>
    <Table caption="The two beliefs side by side, and the difference the later reports make"
      headings={['time and report', 'filtered Rainy', 'smoothed Rainy', 'smoothed − filtered']}
      rows={track.rows.map(row => [
        `${row.time} ${row.label}`, fixed(row.filtered, 6), fixed(row.smoothed, 6),
        row.difference === 0 ? 'exactly 0' : fixed(row.difference, 6),
      ])}
      footnote="A dashboard running on Tuesday can only have the filtered column. A retrospective analyst can have both. Neither is a verified hidden truth: both are probabilities under this model." />
    <Table caption="What changes when only the final report changes, from Clean to Walk"
      headings={['time and report', 'filtered Rainy, original', 'filtered Rainy, corrected', 'smoothed Rainy, original', 'smoothed Rainy, corrected']}
      rows={track.rows.map((row, time) => [
        `${time}`,
        fixed(row.filtered, 6), fixed(changed.rows[time].filtered, 6),
        fixed(row.smoothed, 6), fixed(changed.rows[time].smoothed, 6),
      ])}
      footnote="The first three filtered values are bit-for-bit identical: their prefix did not change. Every smoothed value moved, because every one of them conditions on the report that did." />
  </figure>;
}

/* ================================================================ figure 4 */

export function ViterbiTrellisFigure() {
  const built = trellis(weather, reports, 'max');
  const share = pathShare(weather, reports);
  const altered = pathShare(withRainyEmission(fixtures.changedRainyEmission), reports);
  return <figure className="hmm-figure">
    <figcaption>
      <strong>Figure 4 — Keep the best prefix, not the sum.</strong> The same trellis with one operator changed.
      A solid edge is the predecessor its destination actually stored; a dotted edge is a candidate that lost. At
      the third step the best Rainy predecessor is <em>Rainy</em>, even though the Sunny cell before it is larger:
      the comparison happens after multiplying by each transition, never before.
    </figcaption>
    <StateKey names={weather.stateNames} prefix="hidden state" />
    <div className="hmm-panel">
      <h4>Maximum trellis, with the decoded path emphasised</h4>
      <Trellis model={weather} observations={reports} mode="max" focusTime={2} path={share.path}
        label="The Viterbi trellis with its backtracked path"
        describe={`A two-row trellis over four time steps under the maximum operator. ${built.columns.map(column => `At time ${column.time} the best prefix scores are ${column.cells.map(cell => `${weather.stateNames[cell.state]} ${round(cell.value, 8)}`).join(' and ')}.`).join(' ')} Backtracking gives ${share.path.map(state => weather.stateNames[state]).join(', then ')}.`} />
      <TrellisStep model={weather} observations={reports} mode="max" time={2} />
    </div>
    <TrellisTable model={weather} observations={reports} mode="max"
      caption="Every best-prefix score and the predecessor each destination stored" />
    {/* Stacked, not paired. At 336px this comparison table kept its third
        column outside its own panel with no visible affordance, so the
        contrast the panel exists for was invisible at desktop width. */}
    <div className="hmm-panels is-stacked">
      <div className="hmm-panel">
        <h4>How much of the posterior the winner holds</h4>
        <Stack caption={`Posterior mass of ${share.path.map(state => weather.stateNames[state]).join(' → ')} against everything else`}
          describe={`A single track split into the best path's posterior share, ${round(share.posterior, 6)}, and the remaining ${round(share.remaining, 6)} belonging to the other fifteen paths together.`}
          parts={[
            { name: 'the best path', value: share.posterior, className: 'is-best-path' },
            { name: 'the other fifteen paths together', value: share.remaining, className: 'is-other-paths' },
          ]} totalLabel="whole posterior" />
        <p>
          Its joint mass is {round(share.joint, 8)} and the evidence is {round(share.evidence, 8)}, so its
          posterior is {round(share.posterior, 6)}. “Most likely” is not “nearly certain”: about two thirds of the
          posterior belongs elsewhere.
        </p>
      </div>
      <div className="hmm-panel">
        <h4>A changed emission row moves one of these and not the other</h4>
        <Table caption="Replacing the Rainy emission row with .2, .3, .5"
          headings={['quantity', 'original model', 'changed Rainy row']}
          rows={[
            ['best path', share.path.map(state => weather.stateNames[state]).join(' → '),
              altered.path.map(state => weather.stateNames[state]).join(' → ')],
            ['its joint mass', round(share.joint, 8), round(altered.joint, 8)],
            ['total evidence', round(share.evidence, 8), round(altered.evidence, 8)],
            ['its posterior share', round(share.posterior, 6), round(altered.posterior, 6)],
          ]}
          footnote="The joint mass is unchanged to the last bit, because none of the factors on that particular path was edited. The posterior moved, because the evidence it is divided by did. A joint probability and a conditional probability are different quantities, and this is what that difference looks like." />
      </div>
    </div>
  </figure>;
}

/* ================================================================ figure 5 */

const constrained = fixtures.constrained;
const constrainedReports = [0, 0];

export function PointwiseFigure() {
  const result = infer(constrained, constrainedReports);
  const jointTable = [[0, 0.2, 0.2], [0.35, 0, 0], [0.25, 0, 0]];
  return <figure className="hmm-figure">
    <figcaption>
      <strong>Figure 5 — The most probable state at each time can form an impossible path.</strong> Three states,
      two times, and one observation symbol that every state emits with certainty, so the reports carry no
      information and all of the model's content is in the path masses. A dotted hairline marks an edge the model
      forbids; it is deliberately not drawn as a thin permitted edge.
    </figcaption>
    <StateKey names={constrained.stateNames} prefix="hidden state" />
    <div className="hmm-panel">
      <h4>Only four transitions exist</h4>
      <Trellis model={constrained} observations={constrainedReports} mode="max" showForbidden
        path={result.path} label="The constrained two-step graph"
        describe={`Three hidden states, A, B and C, over two time steps. A can only be followed by B or C; B and C can only be followed by A. The path A to A is forbidden. The decoded path is ${result.path.map(state => constrained.stateNames[state]).join(' then ')}.`} />
    </div>
    {/* Stacked, not paired. At 336px a comparison column of this table sat
        outside its own panel with no visible affordance, so the contrast the
        panel exists for was invisible at desktop width. */}
    <div className="hmm-panels is-stacked">
      <div className="hmm-panel">
        <h4>The joint mass of every legal two-step path</h4>
        <Table caption="Row sums are the initial probabilities; dividing each nonzero row by its sum gives the transition row"
          headings={['first state', ...constrained.stateNames.map(name => `then ${name}`), 'row sum']}
          rows={jointTable.map((row, index) => [
            constrained.stateNames[index], ...row.map(value => (value === 0 ? 'forbidden' : round(value, 4))),
            round(row.reduce((total, value) => total + value, 0), 4),
          ])}
          footnote="A forbidden cell is exactly zero. Adding a pseudocount here to make every answer legal would create a different model, not stabilise this one." />
      </div>
      <div className="hmm-panel">
        <h4>Two decoders, two objectives</h4>
        <Table caption="The pointwise modes against the highest-probability legal path"
          headings={['decision rule', 'output', 'joint probability', 'expected correct positions']}
          rows={[
            ['pointwise marginal modes', result.marginalModes.map(state => constrained.stateNames[state]).join(' → '),
              exactly(result.modesJoint, 6), round(result.modesExpectedCorrect, 6)],
            ['Viterbi', result.path.map(state => constrained.stateNames[state]).join(' → '),
              round(result.pathJoint, 6), round(result.pathExpectedCorrect, 6)],
          ]}
          footnote="No contradiction: they optimise different things over different decision sets. The pointwise rule scores better on expected position count and returns a path the model assigns probability exactly zero." />
        <p>
          The marginals are {result.smoothed[0].map(value => round(value, 4)).join(', ')} at the first time and{' '}
          {result.smoothed[1].map(value => round(value, 4)).join(', ')} at the second. A has the largest share at
          both, so the pointwise rule returns A→A — whose joint probability is <strong>exactly 0</strong>, because
          the transition from A to A is a structural zero. That is a different thing from an undefined quantity: the
          model assigns this route a probability, and the probability it assigns is nothing.
        </p>
      </div>
    </div>
  </figure>;
}

/* ================================================================ figure 6 */

export function CountFlowFigure() {
  const split = countFlow(weather, fixtures.splitRecordings);
  const joined = countFlow(weather, fixtures.joinedRecording);
  const axis = Math.max(
    ...split.start.map(entry => entry.mass),
    ...split.edges.flat().map(entry => entry.mass),
    ...split.symbols.flat().map(entry => entry.mass),
  );
  const bars = (entries, className, labelOf) => {
    const width = 420;
    const left = 128;
    const right = 52;
    const rowHeight = 22;
    const height = entries.length * rowHeight + 24;
    const span = width - left - right;
    return <Diagram><svg viewBox={`0 0 ${width} ${height}`} role="img"
      aria-label={entries.map(entry => `${labelOf(entry)} ${round(entry.mass, 6)}`).join('; ')}>
      {entries.map((entry, index) => {
        const y = 14 + index * rowHeight;
        const length = span * barHeight(entry.mass, axis);
        return <g key={labelOf(entry)}>
          <text className="hmm-small" x="2" y={y + 11}>{labelOf(entry)}</text>
          <rect className={`hmm-bar ${className}`} x={left} y={y} width={Math.max(length, 0)} height="14" />
          <text className="hmm-small hmm-halo" x={left + length + 5} y={y + 11}>{round(entry.mass, 6)}</text>
        </g>;
      })}
    </svg></Diagram>;
  };
  return <figure className="hmm-figure">
    <figcaption>
      <strong>Figure 6 — Fractional events whose totals are exact whole numbers.</strong> Two recordings,
      Walk→Shop and Walk→Clean. Every bar is an expected count under the current parameters, so no single bar is a
      whole number; their totals are, because each recording contributes exactly one start, one within-recording
      transition and two emissions. All three bar groups share one scale, printed beside each bar.
    </figcaption>
    {/* Stacked, so each bar group keeps a viewBox-sized drawing. Laid out in
        three columns the same SVGs shrank to about 350px against a 420-unit
        viewBox and their labels rendered at 8.35px. */}
    <div className="hmm-panels is-stacked">
      <div className="hmm-panel">
        <h4>Expected starts — they total {round(split.totals.startMass, 6)}</h4>
        {bars(split.start, 'is-start', entry => `start in ${entry.name}`)}
      </div>
      <div className="hmm-panel">
        <h4>Expected transitions — they total {round(split.totals.edgeMass, 6)}</h4>
        {bars(split.edges.flat(), 'is-edge',
          entry => `${weather.stateNames[entry.origin]} → ${weather.stateNames[entry.destination]}`)}
      </div>
      <div className="hmm-panel">
        <h4>Expected emissions — they total {round(split.totals.symbolMass, 6)}</h4>
        {bars(split.symbols.flat(), 'is-symbol',
          entry => `${weather.stateNames[entry.state]} emits ${weather.symbolNames[entry.symbol]}`)}
      </div>
    </div>
    <Table caption="Moving the boundary is a different data model, not a storage choice"
      headings={['event', 'two independent recordings', 'the same four reports concatenated']}
      rows={[
        ['starts', String(split.totals.starts), String(joined.totals.starts)],
        ['within-recording transitions', String(split.totals.transitions), String(joined.totals.transitions)],
        ['emissions', String(split.totals.emissions), String(joined.totals.emissions)],
        ['training log likelihood', round(split.logLikelihood, 6), round(joined.logLikelihood, 6)],
        ['expected Rainy→Rainy count', round(split.edges[0][0].mass, 6), round(joined.edges[0][0].mass, 6)],
      ]}
      footnote="Concatenation invents a transition across a boundary that was never observed, and it changes the posterior fractions of the transitions that were." />
  </figure>;
}

/* ================================================================ figure 7 */

export function EmHistoryFigure() {
  const window = objectiveWindows.detail;
  const overview = objectiveWindows.overview;
  const tracks = [
    ...emTrack.fits.map(fit => ({
      key: `seed-${fit.seed}`, label: `seed ${fit.seed}`, values: fit.logLikelihood,
      className: `is-seed-${fit.seed}`,
    })),
    { key: 'uniform', label: 'uniform start', values: emTrack.uniformStart.logLikelihood, className: 'is-uniform' },
  ];
  const generating = emTrack.generatingLogLikelihood;
  return <figure className="hmm-figure">
    <figcaption>
      <strong>Figure 7 — The measured objective, on two windows.</strong> Twelve length-30 recordings sampled from
      the constructed model with seed {emTrack.dataSeed}, then forty exact EM updates from each of three declared
      starts. The vertical axis is the training log likelihood and increases upward, so a rising curve is an
      improving fit. The overview has to hold an initial score near {round(Math.min(...emTrack.fits.map(fit => fit.logLikelihood[0])), 1)};
      at that range the <em>closest pair</em> of final scores lands under half a pixel apart and all four span barely
      three, which is why the second panel exists rather than as a convenience. On the window beside it that same
      closest pair separates by several pixels. Both distances are measured in the rendered figure, not assumed
      from the numbers.
    </figcaption>
    <p className="hmm-legend">
      {tracks.map(track => <span key={track.key}>
        <svg viewBox="0 0 34 12" aria-hidden="true">
          <line className={`hmm-track ${track.className}`} x1="1" y1="6" x2="33" y2="6" />
        </svg>
        {track.label}
      </span>)}
      <span>
        <svg viewBox="0 0 34 12" aria-hidden="true"><line className="hmm-reference" x1="1" y1="6" x2="33" y2="6" /></svg>
        generating parameters, {round(generating, 6)}
      </span>
    </p>
    {/* Stacked, not paired. At 336px a comparison column of this table sat
        outside its own panel with no visible affordance, so the contrast the
        panel exists for was invisible at desktop width. */}
    <div className="hmm-panels is-stacked">
      <div className="hmm-panel">
        <h4>Overview, including the initial models</h4>
        <Plot caption="Every update from iteration 0" height={150} padding={{ left: 56, right: 12, top: 14, bottom: 32 }}
          domain={[0, 40]} range={[overview.low, overview.high]} ticks={[0, 10, 20, 30, 40]}
          valueTicks={[-570, -480, -385]} formatValue={value => round(value, 0)}
          describe={`Four rising tracks from iteration 0 to 40. ${tracks.map(track => `${track.label} starts at ${round(track.values[0], 6)} and ends at ${round(track.values[track.values.length - 1], 6)}.`).join(' ')}`}>
          {(scaleX, scaleY) => <>
            <line className="hmm-reference" x1={scaleX(0)} x2={scaleX(40)} y1={scaleY(generating)} y2={scaleY(generating)} />
            {tracks.map(track => <polyline key={track.key} className={`hmm-track ${track.className}`}
              points={polyline(track.values.map((value, index) => [index, value]), scaleX, scaleY)} />)}
          </>}
        </Plot>
      </div>
      <div className="hmm-panel">
        <h4>Updates 1 to 40, on a window that separates them</h4>
        <Plot caption="The same tracks, without iteration 0" height={150} padding={{ left: 56, right: 12, top: 14, bottom: 32 }}
          domain={[1, 40]} range={[window.low, window.high]} ticks={[1, 10, 20, 30, 40]}
          valueTicks={[window.low, (window.low + window.high) / 2, window.high]}
          formatValue={value => round(value, 1)}
          describe={`The same four tracks over updates 1 to 40 on a narrow vertical window. Their final values are ${tracks.map(track => `${track.label} ${round(track.values[track.values.length - 1], 6)}`).join(', ')}, against generating parameters at ${round(generating, 6)}.`}>
          {(scaleX, scaleY) => <>
            <line className="hmm-reference" x1={scaleX(1)} x2={scaleX(40)} y1={scaleY(generating)} y2={scaleY(generating)} />
            {tracks.map(track => <polyline key={track.key} className={`hmm-track ${track.className}`}
              points={polyline(track.values.map((value, index) => [index, value])
                .filter(([index, value]) => index >= 1 && insideWindow(value, window)), scaleX, scaleY)} />)}
          </>}
        </Plot>
      </div>
    </div>
    <Table caption="Initial and final objective for each declared start"
      headings={['start', 'initial', 'after 40 updates', 'against the generating parameters']}
      rows={[
        ...emTrack.fits.map(fit => [
          `seed ${fit.seed}`, fixed(fit.logLikelihood[0], 6), fixed(fit.logLikelihood[40], 6),
          fit.logLikelihood[40] > generating ? 'higher' : 'lower',
        ]),
        ['uniform', fixed(emTrack.uniformStart.logLikelihood[0], 6),
          fixed(emTrack.uniformStart.logLikelihood[1], 6), 'lower'],
      ]}
      footnote={`The uniform row reaches its value after a single update and then holds it: the remaining iterations move it by less than 10⁻⁸. The generating parameters score ${round(generating, 6)} on this particular finite sample, and all three seeded fits score above it — which is what adapting to sample variation looks like, not evidence that they recovered the generating parameters.`} />
    <Table caption="The uniform start cannot break its own symmetry by iterating longer"
      headings={['row', ...weather.symbolNames]}
      rows={[
        ...emTrack.uniformStart.emission.map((row, state) => [
          `emission row ${state}`, ...row.map(value => fixed(value, 6)),
        ]),
        ['empirical symbol frequency', ...emTrack.uniformStart.empiricalSymbolFrequencies.map(value => fixed(value, 6))],
      ]}
      footnote="Both rows are identical to each other and equal to the empirical frequencies, because a symmetric start gives every state identical responsibilities at every step. This is an identifiability fact about the start, not a slow convergence." />
  </figure>;
}

/* ================================================================ figure 8 */

export function NumericScaleFigure() {
  const rare = repeatedProduct(fixtures.rareFactor, fixtures.rareCount);
  const representable = repeatedProduct(fixtures.representableFactor, fixtures.representableCount);
  const scaled = infer(weather, reports);
  const impossible = infer(fixtures.impossibleModel, fixtures.impossibleObservations);
  return <figure className="hmm-figure">
    <figcaption>
      <strong>Figure 8 — Three different things that all look like a very small number.</strong> A representable
      tiny probability, a positive probability that float64 rounds to zero, and an event the model genuinely
      forbids. Only the middle one is a numerical problem, and only the last one has no posterior at all.
    </figcaption>
    <Table caption="The same repeated product, three ways of holding it"
      headings={['quantity', 'ordinary float64 product', 'in log space', 'by retained scaling']}
      rows={[
        [`${representable.factor} multiplied ${representable.count} times`,
          round(representable.ordinary, 6), fixed(representable.logValue, 6), fixed(representable.scaledLogValue, 6)],
        [`${rare.factor} multiplied ${rare.count} times`,
          `${exactly(rare.ordinary, 6)} — underflow`,
          fixed(rare.logValue, 6), fixed(rare.scaledLogValue, 6)],
      ]}
      footnote="The second row's true value is 10⁻⁸⁰⁰, which float64 cannot hold, so the ordinary column rounds it to exactly zero while both other columns keep it. The first row is the counterexample worth remembering: .3 to the hundredth power is about 5.15 × 10⁻⁵³ and is perfectly representable. Underflow is not a property of small exponents in general." />
    <div className="hmm-panels is-pair">
      <div className="hmm-panel">
        <h4>The four scale factors of our own sequence</h4>
        <Table caption="Each factor is the predictive probability of the next report given those before it"
          headings={['time and report', 'scale factor', 'running product']}
          rows={scaled.factors.map((factor, time) => [
            `${time} ${weather.symbolNames[reports[time]]}`, fixed(factor, 6),
            fixed(scaled.factors.slice(0, time + 1).reduce((product, value) => product * value, 1), 8),
          ])}
          footnote={`Their product is the evidence, ${round(scaled.evidence, 8)}, and the sum of their logarithms is its logarithm, ${round(scaled.logEvidence, 6)}. Some texts define the factor as the reciprocal; their final formula then carries a minus sign, so compare definitions before comparing code.`} />
      </div>
      <div className="hmm-panel">
        <h4>An event the model forbids</h4>
        <p>
          If every state assigns probability zero to an observed symbol, the sequence really is impossible under the
          model. Its evidence is {exactly(impossible.evidence, 6)}, and its posterior is{' '}
          <Undefined because="dividing zero mass by zero evidence has no value; the model assigned this observation nothing to condition on" />.
        </p>
        <p>
          Neither scaling nor log space should turn that into a distribution, and adding an epsilon to every entry
          would change which observations the model permits. That is a modelling decision — a deliberately learned
          unknown-symbol category, or an admission that the constraint was wrong — and not a numerical repair.
        </p>
      </div>
    </div>
  </figure>;
}

/* ================================================================ figure 9 */

export function TopologyFigure() {
  const width = 420;
  const columnX = index => 72 + index * 104;
  const matchY = 112;
  const insertY = 62;
  const deleteY = 26;
  const chainY = [30, 74];
  return <figure className="hmm-figure">
    <figcaption>
      <strong>Figure 9 — Two ways to change the state structure rather than the number of iterations.</strong>{' '}
      Both drawings are schematic illustrations of a topology, not fitted models. Nothing here is estimated from
      data, and no residue, protein or appliance is being claimed.
    </figcaption>
    {/* Stacked, so each 420-unit drawing keeps its capped width. Side by side
        they rendered at about 350px and their 10px labels arrived at 8.35px. */}
    <div className="hmm-panels is-stacked">
      <div className="hmm-panel">
        <h4>A profile alignment topology</h4>
        <Diagram><svg viewBox={`0 0 ${width} 168`} role="img"
          aria-label="Three profile positions. Each has a match state that emits an aligned residue, an insert state above it that emits without advancing the match column, and a delete state above that which advances the position while emitting nothing.">
          <title>A profile HMM's match, insert and delete states</title>
          {[0, 1, 2].map(index => <g key={index}>
            {index > 0 && <>
              <line className="hmm-edge is-carrying" strokeWidth="2" x1={columnX(index - 1) + 26} y1={matchY}
                x2={columnX(index) - 26} y2={matchY} />
              <line className="hmm-edge is-carrying" strokeWidth="1.4" x1={columnX(index - 1) + 22} y1={deleteY}
                x2={columnX(index) - 22} y2={deleteY} />
            </>}
            <rect className="hmm-node is-on-path" x={columnX(index) - 26} y={matchY - 14} width="52" height="28" rx="4" />
            <text className="hmm-small hmm-halo hmm-strong" x={columnX(index)} y={matchY + 4} textAnchor="middle">match {index + 1}</text>
            <circle className="hmm-node" cx={columnX(index)} cy={insertY} r="17" />
            <text className="hmm-small hmm-halo" x={columnX(index)} y={insertY + 4} textAnchor="middle">insert</text>
            <rect className="hmm-node" x={columnX(index) - 22} y={deleteY - 12} width="44" height="24" rx="10" />
            <text className="hmm-small hmm-halo" x={columnX(index)} y={deleteY + 4} textAnchor="middle">delete</text>
            <line className="hmm-edge is-carrying" strokeWidth="1.6" x1={columnX(index)} y1={matchY - 14}
              x2={columnX(index)} y2={insertY + 17} />
            <line className="hmm-edge is-rejected" strokeWidth="1.4" x1={columnX(index)} y1={insertY - 17}
              x2={columnX(index)} y2={deleteY + 12} />
            <line className="hmm-emission" x1={columnX(index)} y1={matchY + 14} x2={columnX(index)} y2={matchY + 34} />
            <text className="hmm-small hmm-muted" x={columnX(index)} y={matchY + 46} textAnchor="middle">residue</text>
          </g>)}
        </svg></Diagram>
        <p>A delete state is silent: it advances the position and emits nothing.</p>
        <p>
          A silent delete is therefore not the same object as a missing measurement at a real time step: one skips
          a position in the profile, the other keeps a time step and marginalises its observation.
        </p>
      </div>
      <div className="hmm-panel">
        <h4>Two hidden chains, one shared observation</h4>
        <Diagram><svg viewBox={`0 0 ${width} 168`} role="img"
          aria-label="Two independent hidden chains running left to right across three time steps, each time step's two states both feeding one shared observation node beneath them.">
          <title>A factorial hidden Markov model</title>
          {[0, 1].map(chain => [0, 1, 2].map(time => <g key={`${chain}-${time}`}>
            {time > 0 && <line className="hmm-edge is-carrying" strokeWidth="1.8"
              x1={columnX(time - 1) + 22} y1={chainY[chain]} x2={columnX(time) - 22} y2={chainY[chain]} />}
            <rect className="hmm-node" x={columnX(time) - 22} y={chainY[chain] - 12} width="44" height="24" rx="4" />
            <line className={`hmm-key-line is-${statePattern(chain)}`}
              x1={columnX(time) - 12} y1={chainY[chain] - 8} x2={columnX(time) + 12} y2={chainY[chain] - 8} />
            <text className="hmm-small hmm-halo" x={columnX(time)} y={chainY[chain] + 6} textAnchor="middle">
              chain {chain + 1}
            </text>
            {/* The upper chain's connector bows around the lower chain's node.
                Drawn straight down it passed through that node and its label,
                which read as the two chains being connected to each other
                rather than each to the shared observation. */}
            {chain === 0
              ? <path className="hmm-emission" fill="none"
                d={`M ${columnX(time) - 22} ${chainY[0]} Q ${columnX(time) - 48} ${(chainY[1] + 116) / 2} ${columnX(time) - 10} 116`} />
              : <line className="hmm-emission" x1={columnX(time)} y1={chainY[chain] + 12}
                x2={columnX(time)} y2="116" />}
          </g>))}
          {[0, 1, 2].map(time => <g key={`obs-${time}`}>
            <rect className="hmm-card" x={columnX(time) - 26} y="116" width="52" height="26" rx="3" />
            <text className="hmm-small hmm-halo" x={columnX(time)} y="133" textAnchor="middle">reading</text>
            <text className="hmm-small hmm-muted" x={columnX(time)} y="158" textAnchor="middle">t = {time}</text>
          </g>)}
        </svg></Diagram>
        <p>
          The two chains transition independently <em>a priori</em>. Once they explain the same measurement they are
          coupled in the posterior, which is why exact inference over the joint state grows quickly and why
          structured approximations exist.
        </p>
      </div>
    </div>
    <Table caption="Free parameters of a fully unconstrained categorical model"
      headings={['states N', 'symbols M', 'initial row', 'transitions', 'emissions', 'total p']}
      rows={[[2, 3], [3, 3], [2, 5], [4, 4]].map(([states, symbols]) => [
        String(states), String(symbols), String(states - 1), String(states * (states - 1)),
        String(states * (symbols - 1)), String(parameterCount(states, symbols)),
      ])}
      footnote="Our two-state, three-symbol model has seven free parameters. Structural zeros, tied rows, fixed parameters and other emission families all change that count, so it is a starting point for a selection criterion rather than an answer." />
  </figure>;
}

/* ============================================================ provenance note */

export function DataProvenanceNote() {
  return <p className="hmm-caption">
    The sentences come from the <a href={provenance.page}>Universal Dependencies English Web Treebank</a>,
    release {provenance.release}, published {provenance.published}, annotations licensed{' '}
    <a href={provenance.licenseUrl}>{provenance.license}</a>. This page serves{' '}
    <a href={provenance.file} download>its own unchanged copy</a> of the extract,{' '}
    {provenance.bytes.toLocaleString('en-US')} bytes, SHA-256 <code>{provenance.sha256}</code>, beside its{' '}
    <a href={provenance.attribution}>attribution</a>, which records the extraction, the original sentence
    identifiers and the coarse-label transformation. {provenance.sentences.train} training,{' '}
    {provenance.sentences.development} development and {provenance.sentences.reserved} reserved sentences hold{' '}
    {provenance.tokens.train.toLocaleString('en-US')}, {provenance.tokens.development} and{' '}
    {provenance.tokens.reserved} tokens.
  </p>;
}

/* ================================================== the duration bars, shared */

export function DurationBars({ stay, label, describe }) {
  const model = durationBars(stay);
  const axis = Math.max(...model.probabilities, model.tailMass, 1e-9);
  const width = 420;
  const left = 34;
  const right = 14;
  const top = 18;
  const plotHeight = 84;
  const base = top + plotHeight;
  const slots = model.bars.length + 1;
  const spacing = (width - left - right) / slots;
  const barWidth = Math.min(26, spacing - 8);
  return <Diagram describe={describe}><svg viewBox={`0 0 ${width} ${base + 48}`} role="img" aria-label={describe}>
    <title>{label}</title>
    {[0, axis / 2, axis].map(value => <g key={value}>
      <line className="hmm-grid" x1={left} x2={width - right} y1={base - plotHeight * (value / axis)} y2={base - plotHeight * (value / axis)} />
      <text className="hmm-small" x={left - 4} y={base - plotHeight * (value / axis) + 4} textAnchor="end">{round(value, 2)}</text>
    </g>)}
    {model.bars.map((bar, index) => {
      const centre = left + spacing * (index + 0.5);
      const length = plotHeight * barHeight(bar.probability, axis);
      return <g key={bar.duration}>
        <rect className="hmm-bar is-duration" x={centre - barWidth / 2} y={base - length} width={barWidth} height={length} />
        <text className="hmm-small hmm-halo" x={centre} y={base + 14} textAnchor="middle">{bar.duration}</text>
        <text className="hmm-small hmm-halo hmm-muted" x={centre} y={base + 28} textAnchor="middle">
          {round(bar.probability, 4)}
        </text>
      </g>;
    })}
    {(() => {
      const centre = left + spacing * (model.bars.length + 0.5);
      const length = plotHeight * barHeight(model.tailMass, axis);
      return <g>
        <rect className="hmm-bar is-tail" x={centre - barWidth / 2} y={base - length} width={barWidth} height={length} />
        <text className="hmm-small hmm-halo" x={centre} y={base + 14} textAnchor="middle">9+</text>
        <text className="hmm-small hmm-halo hmm-muted" x={centre} y={base + 28} textAnchor="middle">
          {round(model.tailMass, 4)}
        </text>
      </g>;
    })()}
    <line className="hmm-axis" x1={left} x2={width - right} y1={base} y2={base} />
    <text className="hmm-small hmm-muted" x="2" y={base + 44}>duration in steps; the last bar is all durations of nine or more</text>
  </svg></Diagram>;
}
