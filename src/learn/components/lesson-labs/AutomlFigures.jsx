import { useId, useState } from 'react';
import {
  activationKernel, bilevelDerivatives, candidateSeries, evidenceBoundary, evidenceFlow,
  networkBlocks, provenanceBoxWidth, provenanceLanes, sharedErrors,
} from '../../data/automl-models.js';
import {
  candidates as recordedCandidates, inspection as recordedInspection, foldRows,
  majorityBaseline, roles, sourceRows,
} from '../../data/automl-data.js';
import { Legend, Table, fixed, round } from './AutomlShared.jsx';
import './automl-labs.css';

/** Inline figures for the AutoML & NAS lesson.
 *
 * Every number, coordinate and drawn edge comes from automl-models.js or from
 * the recorded study in automl-data.js. Nothing here is a shape chosen to look
 * right: a rung, a block, a point or an arrow is a claim the model verifier
 * asserts.
 */

/** One arrowhead definition per mounted figure, so two figures on one page
 * cannot share a marker id. */
function Arrowheads({ id }) {
  return <defs>
    <marker id={`${id}-head`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#8eb9a5" />
    </marker>
    <marker id={`${id}-gold`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#e7b94a" />
    </marker>
    <marker id={`${id}-quiet`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#6f7d77" />
    </marker>
  </defs>;
}

/* ================================================ F1 · the evidence boundary */

/** Each stage's rows are named *inside* its own box.
 *
 * They were floating labels until a screenshot showed "inspection rows, once"
 * sitting directly above the Reserved rows box, which invites precisely the
 * misreading this figure exists to prevent. A row source now belongs to exactly
 * one node, visibly.
 */
const F1_NODES = {
  proposer: { x: 16, y: 40, w: 132, h: 44, lines: ['Propose a', 'configuration'] },
  fit: { x: 192, y: 40, w: 132, h: 56, lines: ['Fit the whole', 'pipeline'], sub: 'fitting rows only' },
  score: { x: 192, y: 140, w: 132, h: 56, lines: ['Score validation', 'predictions'], sub: 'validation rows only' },
  record: { x: 16, y: 140, w: 132, h: 44, lines: ['Search', 'record'] },
  refit: { x: 16, y: 236, w: 132, h: 44, lines: ['Refit on all', 'development rows'] },
  inspect: { x: 192, y: 236, w: 132, h: 56, lines: ['Inspection', 'comparison'], sub: 'inspection rows, once' },
  reserved: { x: 192, y: 306, w: 132, h: 44, lines: ['Reserved rows'], sub: 'no arrow in or out' },
};

function Node({ id, node, kind }) {
  const centre = node.x + node.w / 2;
  const top = node.y;
  return <g>
    <rect className={`am-node is-${kind}`} x={node.x} y={top} width={node.w} height={node.h} rx="4" />
    {node.lines.map((line, index) => (
      <text key={line} x={centre} textAnchor="middle"
        y={top + (node.lines.length === 1 ? 18 : 17 + index * 15)}>{line}</text>
    ))}
    {node.sub && (
      <text className="is-small" x={centre} textAnchor="middle"
        y={top + (node.lines.length === 1 ? 35 : 48)}>{node.sub}</text>
    )}
    <title>{id}</title>
  </g>;
}

export function EvidenceLoopFigure() {
  const markers = useId().replace(/:/g, '');
  const boundary = evidenceBoundary();
  const development = roles.find(role => role.role === 'development');
  const inspectionRole = roles.find(role => role.role === 'inspection');
  const reserved = roles.find(role => role.role === 'reserved');
  return <figure className="am-figure">
    <figcaption>
      <strong>Which evidence returns to the search?</strong> Four stages form one loop, and the arrow that closes it is
      the only path a score takes back to the proposer. Everything inside the framed region marked <em>selection</em>{' '}
      can change the next configuration proposed. The two partitions outside the frame are reached once, or never.
    </figcaption>
    <svg viewBox="0 0 340 360" role="img"
      aria-label="A four-stage loop. Propose a configuration feeds Fit the whole pipeline, which sees each fold's fitting rows only; that feeds Score validation predictions, which sees that fold's validation rows only; that feeds the Search record; and a dashed gold arrow returns from the search record to the proposer, closing the selection loop. A separate solid arrow leaves the search record for Refit on all development rows and then for the Inspection comparison, which sees the inspection rows once and has no outgoing arrow. Reserved rows sit alone, labelled no arrow in or out.">
      <Arrowheads id={markers} />
      <rect className="am-node is-nested" x="8" y="26" width="324" height="184" rx="6" fillOpacity="0" />
      {/* The frame carries a short scale-bound label; the sentence explaining
          what the frame means is prose and reflows in the caption below. */}
      <text className="is-small" x="12" y="20">selection</text>
      {['proposer', 'fit', 'score', 'record'].map(id => (
        <Node key={id} id={id} node={F1_NODES[id]} kind="selection" />
      ))}
      <Node id="refit" node={F1_NODES.refit} kind="commitment" />
      <Node id="inspect" node={F1_NODES.inspect} kind="assessment" />
      <Node id="reserved" node={F1_NODES.reserved} kind="untouched" />

      {/* proposer → fit */}
      <line className="am-flow" x1="148" y1="62" x2="186" y2="62" markerEnd={`url(#${markers}-head)`} />
      {/* fit → score */}
      <line className="am-flow" x1="258" y1="96" x2="258" y2="134" markerEnd={`url(#${markers}-head)`} />
      {/* score → record */}
      <line className="am-flow" x1="186" y1="162" x2="154" y2="162" markerEnd={`url(#${markers}-head)`} />
      {/* record → proposer, the arrow that closes the loop */}
      <line className="am-flow is-feedback" x1="82" y1="134" x2="82" y2="90" markerEnd={`url(#${markers}-gold)`} />
      <text className="is-small am-halo" x="88" y="116">validation score</text>
      {/* record → refit → inspect */}
      <line className="am-flow is-commit" x1="82" y1="186" x2="82" y2="230" markerEnd={`url(#${markers}-head)`} />
      <text className="is-small am-halo" x="88" y="220">selection fixed</text>
      <line className="am-flow is-commit" x1="148" y1="258" x2="186" y2="258" markerEnd={`url(#${markers}-head)`} />
    </svg>
    <p>
      The nested step inside <strong>Fit the whole pipeline</strong> is “{boundary.scalerRefitPerFold}”. That is the part
      people leave out: a scaler fitted once over all development rows has already seen each fold&rsquo;s validation rows.
      Here it is refitted {foldRows.length} times, once per fold.
    </p>
    <Table caption="What each stage receives, and what it is allowed to send back"
      headings={['stage', 'rows it sees', 'what leaves it']}
      rows={[
        ['Propose a configuration', 'none', 'one candidate configuration'],
        ['Fit the whole pipeline', `each fold's fitting rows`, 'a fitted scaler and model'],
        ['Score validation predictions', `each fold's validation rows`, 'one accuracy per fold'],
        ['Search record', 'none', 'evidence back to the proposer, then one fixed selection'],
        ['Refit on all development rows', `all ${development.rows} development rows`, 'two frozen procedures'],
        ['Inspection comparison', `${inspectionRole.rows} inspection rows, once`, 'nothing — no arrow leaves it'],
        ['Reserved rows', 'none', 'nothing — it receives no prediction'],
      ]} />
    <p className="am-caption">
      The counts are this lesson&rsquo;s declared study, introduced in section 5: {development.rows} development,
      {' '}{inspectionRole.rows} inspection and {reserved.rows} reserved rows. They are not a universal split ratio.
      The loop closes at the search record; the inspection stage has {boundary.inspectionFeedsNothing ? 'no' : 'an'} outgoing
      arrow, and the reserved partition is {boundary.reservedIsIsolated ? 'connected to nothing at all' : 'connected'}.
    </p>
  </figure>;
}

/* =============================================== F2 · the three architectures */

const ARCHITECTURES = [
  { id: 'w8', label: '4 → 8 → 1', widths: [8] },
  { id: 'w16', label: '4 → 16 → 1', widths: [16] },
  { id: 'w88', label: '4 → 8 → 8 → 1', widths: [8, 8] },
];

export function ArchitectureFigure() {
  const markers = useId().replace(/:/g, '');
  const [selected, setSelected] = useState('w8');
  const [block, setBlock] = useState(0);
  const chosen = ARCHITECTURES.find(entry => entry.id === selected);
  const model = networkBlocks(4, chosen.widths, 1);
  const detail = model.blocks[Math.min(block, model.blocks.length - 1)];
  const columnWidth = 300 / model.sizes.length;
  return <figure className="am-figure">
    <figcaption>
      <strong>Count the parameters you actually changed.</strong> Choosing a width changes which weight matrices exist and
      how large they are. Select a block to see its shape equation; a bias is counted, never quietly dropped.
    </figcaption>
    <div className="am-buttons" role="group" aria-label="Architecture">
      {ARCHITECTURES.map(entry => (
        <button key={entry.id} type="button" className={entry.id === selected ? 'is-primary' : 'is-quiet'}
          aria-pressed={entry.id === selected}
          onClick={() => { setSelected(entry.id); setBlock(0); }}>{entry.label}</button>
      ))}
    </div>
    <div className="am-panel">
      <svg viewBox="0 0 340 176" role="img"
        aria-label={`The shape path ${model.shapePath}. ${model.blocks.map(entry => `${entry.role}: a ${entry.shape} weight matrix and ${entry.biases} bias${entry.biases === 1 ? '' : 'es'}, ${entry.equation}, followed by ${entry.activation}`).join('. ')}. The parameter total is ${model.total}.`}>
        <Arrowheads id={markers} />
        {model.sizes.map((size, index) => {
          const x = 14 + index * columnWidth;
          const height = Math.min(96, 16 + size * 4.6);
          const y = 88 - height / 2;
          return <g key={index}>
            <rect className={index === 0 ? 'am-block is-output' : 'am-block'} x={x} y={y} width="42" height={height} rx="3" />
            <text x={x + 21} y={y - 7} textAnchor="middle">{size}</text>
            {/* One baseline for every column label, below the tallest box, so a
                short box's label cannot rise into the activation label beside
                it — which "output" and "sigmoid" did on the deeper network. */}
            <text className="is-small" x={x + 21} y="152" textAnchor="middle">
              {index === 0 ? 'inputs' : index === model.sizes.length - 1 ? 'output' : `hidden ${index}`}
            </text>
          </g>;
        })}
        {model.blocks.map((entry, index) => {
          const from = 14 + index * columnWidth + 42;
          const to = 14 + (index + 1) * columnWidth;
          const middle = (from + to) / 2;
          const active = entry.index === detail.index;
          return <g key={entry.index} onClick={() => setBlock(entry.index)} style={{ cursor: 'pointer' }}>
            <line className={active ? 'am-flow is-feedback' : 'am-flow'} x1={from + 2} y1="88" x2={to - 4} y2="88"
              markerEnd={`url(#${markers}-${active ? 'gold' : 'head'})`} />
            <text className="is-small am-halo" x={middle} y="80" textAnchor="middle">{entry.shape}</text>
            <text className="is-small am-halo" x={middle} y="106" textAnchor="middle">{entry.activation}</text>
          </g>;
        })}
      </svg>
    </div>
    <div className="am-buttons" role="group" aria-label="Affine block">
      {model.blocks.map(entry => (
        <button key={entry.index} type="button" className={entry.index === detail.index ? 'is-primary' : 'is-quiet'}
          aria-pressed={entry.index === detail.index}
          onClick={() => setBlock(entry.index)}>{entry.role}</button>
      ))}
    </div>
    <div className="am-panel">
      <h4>{detail.role}: {detail.equation}</h4>
      <svg viewBox="0 0 340 120" role="img"
        aria-label={`A ${detail.from} by ${detail.to} weight rectangle holding ${detail.weights} weights, beside a bias strip of ${detail.biases}. Their sum is ${detail.total}. The rectangle's sides are dimensions, not measured feature units.`}>
        <rect className="am-block" x="18" y="16" width="150" height="76" rx="3" />
        <text x="93" y="10" textAnchor="middle">{detail.to} columns out</text>
        <text className="is-small" x="12" y="58" textAnchor="end" transform="rotate(-90 12 58)">{detail.from} rows in</text>
        <text x="93" y="60" textAnchor="middle">{detail.weights} weights</text>
        <rect className="am-block is-bias" x="192" y="16" width="44" height="76" rx="3" />
        <text className="is-small" x="214" y="10" textAnchor="middle">bias</text>
        <text x="214" y="60" textAnchor="middle">{detail.biases}</text>
        <text x="258" y="60">= {detail.total}</text>
        <text className="is-small" x="18" y="110">then {detail.activation}</text>
      </svg>
      <p>
        A unit in this block computes <code>h = {detail.activation}(w · x + b)</code> over its {detail.from} incoming
        values. The rectangle&rsquo;s sides are dimensions, not measured feature units, and no fitted weights are stored
        anywhere in this lesson. In the diagram above, box heights are schematic, with minimum padding and a cap for readability; the exact layer widths are the printed numbers, not a proportional height scale;
        the number printed over each box says the same thing exactly.
      </p>
    </div>
    <Table caption="All three architectures, block by block. Adding parameters is not the same as adding accuracy; section 5 supplies the evidence."
      headings={['architecture', 'blocks', 'parameters', 'in the study']}
      rows={ARCHITECTURES.map(entry => {
        const built = networkBlocks(4, entry.widths, 1);
        const record = recordedCandidates.find(candidate => candidate.parameterCount === built.total);
        return [
          built.shapePath,
          built.blocks.map(item => item.equation.split(' = ')[0]).join(' + '),
          String(built.total),
          record ? record.label : '—',
        ];
      })}
      rowClass={index => (ARCHITECTURES[index].id === selected ? 'is-selected' : undefined)} />
  </figure>;
}

/* ================================================ F3 · the observed outcomes */

const SHORT_NAMES = [
  'log raw .1', 'log raw 1', 'log std .1', 'log std 1', 'tree 2', 'tree 5',
  'knn 3', 'knn 9', 'net 8', 'net 16', 'net 8,8',
];

export function ObservedResultsFigure() {
  const [openRow, setOpenRow] = useState(null);
  const series = candidateSeries();
  const shared = sharedErrors().filter(entry => entry.ids.length > 1);
  const low = 0.85;
  const width = 340;
  // A wide left gutter and full-size row labels: eleven rows at the small size
  // were unreadable at 320 px, and the figure carries its own max-width so the
  // same labels are not magnified on a desktop column either.
  const left = 100;
  const right = 326;
  const place = value => left + (right - left) * (value - low) / (1 - low);
  const rowY = index => 28 + index * 27;
  const height = rowY(series.rows.length) + 22;
  const selectedRow = series.rows[9];
  const detailRows = openRow === null ? [] : series.rows[openRow].errorRows;
  return <figure className="am-figure">
    <figcaption>
      <strong>Eleven candidates, three folds each.</strong> The filled marks are the three fold accuracies; the hollow
      mark is their arithmetic mean, which is what selection uses. The axis starts at {low}, not at zero, so that the
      interesting region is visible — the label says so on the axis itself.
    </figcaption>
    <svg className="am-results-plot" viewBox={`0 0 ${width} ${height}`} role="img"
      aria-label={`A dot plot of eleven candidates. ${series.rows.map((row, index) => `${SHORT_NAMES[index]}: folds ${row.folds.map(fold => fold.accuracy.toFixed(6)).join(', ')}, mean ${row.mean.toFixed(6)}`).join('. ')}. The horizontal axis runs from ${low} to 1.`}>
      {[0.85, 0.9, 0.95, 1].map(value => <g key={value}>
        <line className="am-grid" x1={place(value)} x2={place(value)} y1="18" y2={rowY(series.rows.length) - 10} />
        <text className="is-small" x={place(value)} y={rowY(series.rows.length) + 8} textAnchor="middle">{value === 0.85 ? '0.85 →' : value}</text>
      </g>)}
      <text className="is-small" x={left} y="12">accuracy, axis starts at {low}</text>
      {series.rows.map((row, index) => <g key={row.id}>
        <text x="4" y={rowY(index) + 4}>{SHORT_NAMES[index]}</text>
        <line className="am-grid" x1={left} x2={right} y1={rowY(index)} y2={rowY(index)} />
        {row.folds.map(fold => (
          <circle key={fold.fold} className="am-mark is-fold" cx={place(fold.accuracy)} cy={rowY(index)} r="3.4" />
        ))}
        <circle className={`am-mark ${index === 9 ? 'is-frontier' : 'is-mean'}`} cx={place(row.mean)} cy={rowY(index)} r="5.2" />
      </g>)}
    </svg>
    <Legend items={[
      { label: 'one fold accuracy', shape: 'dot', className: 'am-mark is-fold' },
      { label: 'mean of the three folds — selection uses this mark', shape: 'dot', className: 'am-mark is-mean' },
      { label: 'the selected candidate', shape: 'dot', className: 'am-mark is-frontier' },
    ]} />
    <Table caption={`Exact outcomes. Fold sizes are ${foldRows.join(', ')} rows, so the mean of fold accuracies and the pooled out-of-fold accuracy are two different rules and need not agree.`}
      headings={['configuration', 'fold 1', 'fold 2', 'fold 3', 'mean of folds', 'pooled out of fold', 'parameters']}
      rows={series.rows.map((row, index) => [
        row.label,
        `${row.folds[0].correct}/${row.folds[0].rows}`,
        `${row.folds[1].correct}/${row.folds[1].rows}`,
        `${row.folds[2].correct}/${row.folds[2].rows}`,
        fixed(row.mean, 6),
        `${row.pooledCorrect}/${row.pooledRows} = ${fixed(row.pooled, 6)}`,
        row.parameterCount === null ? '—' : String(row.parameterCount),
      ])}
      rowClass={index => (index === 9 ? 'is-selected' : undefined)} />
    <p>
      The width-16 network wins the declared criterion with {fixed(selectedRow.mean, 6)}, and it is the only candidate
      with no out-of-fold mistake at all. {series.tiedRunnersUp.length} other
      candidates — {series.tiedRunnersUp.map(row => row.label).join('; ')} — tie each other exactly
      at {fixed(series.runnerUpMean, 6)}, and they tie because each makes <strong>one</strong> out-of-fold mistake
      and it is <strong>the same one</strong>: development source row {series.sharedByTied[0]}, file
      line {sourceRows[String(series.sharedByTied[0])].line}. Three near-perfect models sharing a single error is not
      three independent pieces of evidence, which is worth remembering before averaging them into an ensemble.
    </p>
    <p className="am-caption">
      That row is the hardest one in the development pool: {shared[0].ids.length} of
      the {series.rows.length} candidates miss it. The weaker ones miss a great many others besides, so sharing this
      mistake is what the tied three have in common, not what the seven have in common.
    </p>
    <div className="am-buttons" role="group" aria-label="Inspect a candidate's out-of-fold mistakes">
      {series.rows.map((row, index) => (
        <button key={row.id} type="button" className={openRow === index ? 'is-primary' : 'is-quiet'}
          aria-pressed={openRow === index}
          onClick={() => setOpenRow(openRow === index ? null : index)}>
          {SHORT_NAMES[index]} ({row.errorRows.length})
        </button>
      ))}
    </div>
    {openRow !== null && (detailRows.length === 0
      ? <p className="am-readout" role="status">
        {series.rows[openRow].label} makes no out-of-fold mistake at all: {series.rows[openRow].pooledCorrect} of
        {' '}{series.rows[openRow].pooledRows} development rows are correct.
      </p>
      : <Table scroll
        caption={`${series.rows[openRow].label}: every out-of-fold mistake, with the four supplied descriptors. No source photograph, physical identity, fitted probability or decision boundary is retained.`}
        headings={['file line', 'variance', 'skewness', 'kurtosis', 'entropy', 'true class', 'predicted']}
        rows={detailRows.map(row => {
          const record = sourceRows[String(row)];
          return [
            String(record.line), ...record.features.map(value => round(value, 5)),
            String(record.label), String(1 - record.label),
          ];
        })} />)}
    <h4>The declared final comparison, on a separate partition</h4>
    <div className="am-panels is-pair">
      {recordedInspection.map(entry => <div className="am-panel" key={entry.role}>
        <h4>{entry.role === 'selected' ? 'Selected by the search' : 'Predeclared baseline'}: {entry.label}</h4>
        <Table caption={`${entry.correct} of ${entry.rows} correct. Rows are the true class, columns the predicted class.`}
          headings={['true class', 'predicted 0', 'predicted 1']}
          rows={entry.confusion.map((row, index) => [String(index), String(row[0]), String(row[1])])} />
        <p>
          {entry.errorRows.length === 0
            ? 'No mistake on this partition. A finite perfect score is still a finite observation.'
            : `Mistakes at file lines ${entry.errorRows.map(row => sourceRows[String(row)].line).join(', ')}, each a true class 0 predicted as 1.`}
        </p>
      </div>)}
    </div>
    <p className="am-caption">
      Always predicting the development majority class {majorityBaseline.predictedClass} would get
      {' '}{majorityBaseline.correct} of {majorityBaseline.rows} right. It is context, not a fitted search candidate.
      The {roles.find(role => role.role === 'reserved').rows} reserved rows received no prediction, so no panel here can
      show one. These two models are the only ones with a recorded inspection outcome.
    </p>
  </figure>;
}

/* ============================================ F4 · three derivatives, one problem */

export function BilevelFigure() {
  const markers = useId().replace(/:/g, '');
  const [stationary, setStationary] = useState(false);
  const setup = stationary ? { w: 0.2, alpha: 0.2, xi: 0.1 } : { w: 0, alpha: 0.2, xi: 0.1 };
  const derived = bilevelDerivatives(setup);
  return <figure className="am-figure">
    <figcaption>
      <strong>Differentiate the function that was actually defined.</strong> Three lanes, one small problem. They give
      different numbers because the weight each one differentiates through depends on <em>α</em> differently — not
      because one of them is an approximation error of another.
    </figcaption>
    <div className="am-buttons" role="group" aria-label="Current weight">
      <button type="button" className={stationary ? 'is-quiet' : 'is-primary'} aria-pressed={!stationary}
        onClick={() => setStationary(false)}>w = 0, away from the training optimum</button>
      <button type="button" className={stationary ? 'is-primary' : 'is-quiet'} aria-pressed={stationary}
        onClick={() => setStationary(true)}>w = 0.2, where the training gradient is zero</button>
    </div>
    <div className="am-panels is-stacked">
      {derived.lanes.map(lane => (
        <div className="am-panel" key={lane.id}>
          <h4>
            {lane.id === 'direct' ? 'Hold w fixed' : lane.id === 'oneStep' ? 'One training step, then differentiate' : 'Solve the inner problem exactly'}
            {' '}— outer derivative {fixed(lane.outer, 6)}
          </h4>
          {/* Only geometry lives in the drawing. The held-fixed sentence is
              explanatory prose whose position encodes nothing, so it reflows in
              HTML beneath rather than overflowing a 340-unit viewBox. */}
          <svg viewBox="0 0 340 68" role="img"
            aria-label={`Lane ${lane.id}. ${lane.held}. Alpha connects to ${lane.weightSymbol} with dependency derivative ${lane.dependency}, and ${lane.weightSymbol} equals ${lane.weightValue}, which enters the validation loss. The outer derivative is ${lane.outer}.`}>
            <Arrowheads id={`${markers}-${lane.id}`} />
            {/* The dependency label sits above the whole row rather than between
                the two nodes: at 1366 px it was flush against both their edges,
                and the longest of the three would have overlapped them. */}
            <text className="is-small am-halo" x="98" y="12" textAnchor="middle">{lane.dependencyLabel}</text>
            <rect className="am-node is-selection" x="8" y="18" width="60" height="34" rx="4" />
            <text x="38" y="39" textAnchor="middle">α = {derived.alpha}</text>
            <line className={lane.dependency === 0 ? 'am-flow is-rows' : 'am-flow'} x1="70" y1="35" x2="126" y2="35"
              markerEnd={`url(#${markers}-${lane.id}-${lane.dependency === 0 ? 'quiet' : 'head'})`} />
            <rect className="am-node is-commitment" x="130" y="18" width="96" height="34" rx="4" />
            <text x="178" y="39" textAnchor="middle">{lane.weightSymbol} = {round(lane.weightValue, 4)}</text>
            <line className="am-flow" x1="228" y1="35" x2="256" y2="35" markerEnd={`url(#${markers}-${lane.id}-head)`} />
            <rect className="am-node is-assessment" x="260" y="18" width="72" height="34" rx="4" />
            <text className="is-small" x="296" y="39" textAnchor="middle">L_val</text>
            <text className="is-small" x="332" y="62" textAnchor="end">∂L_val/∂α = {fixed(lane.outer, 6)}</text>
          </svg>
          <p>{lane.note}. Here {lane.held}.</p>
        </div>
      ))}
    </div>
    <Table caption={`L_train = ½(w − α)², L_val = ½(w − ${derived.valTarget})², with α = ${derived.alpha} and ξ = ${derived.xi}. The current training gradient is ${fixed(derived.trainingGradient, 6)}.`}
      headings={['lane', 'weight it differentiates through', 'its value', 'dependency on α', 'outer derivative']}
      rows={derived.lanes.map(lane => [
        lane.id === 'direct' ? 'direct, ξ treated as 0' : lane.id === 'oneStep' ? 'one-step unroll' : 'exact inner solution',
        lane.weightSymbol, round(lane.weightValue, 6), round(lane.dependency, 6), fixed(lane.outer, 6),
      ])} />
    <p>
      {stationary
        ? <>At w = 0.2 the training gradient is exactly zero, so the one-step weight <em>w&prime;</em> has the same value
          as <em>w</em>. Its <strong>derivative</strong> with respect to α is still ξ = {derived.xi}, so the one-step outer
          derivative is {fixed(derived.lanes[1].outer, 6)}, not zero. Two functions can agree at a point and have
          different derivatives there; that is the whole distinction this figure exists to make.</>
        : <>The one-step derivative keeps the chain-rule term through the training step. Setting ξ = 0 removes it and
          gives the first lane — which is what DARTS calls the first-order approximation. A one-step unroll is therefore
          not automatically “first order” merely because it contains one training step.</>}
    </p>
  </figure>;
}

/* ============================================= F5 · where the weights came from */

export function WeightProvenanceFigure() {
  const markers = useId().replace(/:/g, '');
  const [codes, setCodes] = useState(['110', '101']);
  const kernel = activationKernel(codes);
  return <figure className="am-figure">
    <figcaption>
      <strong>What evidence did this architecture receive?</strong> Three lanes produce three numbers that are often
      quoted the same way. Only the first is a validation metric for weights fitted for that architecture alone.
    </figcaption>
    <div className="am-panels is-stacked">
      {provenanceLanes.map(lane => (
        <div className="am-panel" key={lane.id}>
          <h4>{lane.label}</h4>
          {/* The drawing carries the chain; what each lane reuses and measures
              is prose, and reflows in HTML beneath instead of running past the
              right edge of a 340-unit viewBox. */}
          <svg viewBox="0 0 340 50" role="img"
            aria-label={`${lane.label}: ${lane.steps.join(', then ')}. State reused: ${lane.stateReused}. It measures ${lane.measures}.`}>
            <Arrowheads id={`${markers}-${lane.id}`} />
            {lane.stepLabels.map((step, index) => {
              const boxWidth = provenanceBoxWidth(lane.stepLabels.length);
              const x = 2 + index * (boxWidth + 14);
              return <g key={step}>
                <rect className={`am-lane is-${lane.id}`} x={x} y="9" width={boxWidth} height="32" rx="3" />
                <text className="is-small" x={x + boxWidth / 2} y="28" textAnchor="middle">{step}</text>
                {index < lane.stepLabels.length - 1 && (
                  <line className="am-flow" x1={x + boxWidth + 1} y1="25" x2={x + boxWidth + 11} y2="25"
                    markerEnd={`url(#${markers}-${lane.id}-head)`} />
                )}
              </g>;
            })}
          </svg>
          <p>
            {lane.steps.join(' → ')}. It reuses {lane.stateReused}, and it measures {lane.measures}.
          </p>
        </div>
      ))}
    </div>
    <div className="am-panel">
      <h4>The activation-code inset</h4>
      <p>
        Call a ReLU unit active when its incoming weighted sum is positive. For a small batch, record one bit per unit.
        With <code>N_A = {kernel.width}</code> recorded units, <code>K_ij = N_A − d_H(c_i, c_j)</code>.
      </p>
      <div className="am-buttons" role="group" aria-label="Second activation code">
        <button type="button" className={codes[1] === '101' ? 'is-primary' : 'is-quiet'} aria-pressed={codes[1] === '101'}
          onClick={() => setCodes(['110', '101'])}>second code 101 — differentiated</button>
        <button type="button" className={codes[1] === '110' ? 'is-primary' : 'is-quiet'} aria-pressed={codes[1] === '110'}
          onClick={() => setCodes(['110', '110'])}>second code 110 — identical</button>
      </div>
      <Table caption={`Codes ${codes.join(' and ')}: Hamming distance ${kernel.distance}, so the off-diagonal entry is ${kernel.width} − ${kernel.distance} = ${kernel.matrix[0][1]}.`}
        headings={['K', 'c₁', 'c₂']}
        rows={kernel.matrix.map((row, index) => [index === 0 ? 'c₁' : 'c₂', String(row[0]), String(row[1])])} />
      <p className="am-readout" role="status">
        Determinant {kernel.determinant}.{' '}
        {kernel.singular
          ? <>Identical codes give a singular kernel, and its log determinant has {kernel.logDeterminantLabel}. A real
            implementation must say what it does here; substituting a stabilized value silently would change this exact
            example into a different one.</>
          : <>Log determinant ln {kernel.determinant} ≈ {fixed(kernel.logDeterminant, 6)}. The score prefers differentiated
            activation patterns. That preference is not a proof that the architecture will generalize after training.</>}
      </p>
    </div>
    <p className="am-caption">
      Code positions record ReLU states, not classes or image bits. No arrow runs from the proxy to a test accuracy, and
      no speed-up multiplier is drawn: the target claim still needs an independently trained, independently evaluated
      final architecture.
    </p>
  </figure>;
}
