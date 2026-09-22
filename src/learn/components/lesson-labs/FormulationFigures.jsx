import { useState } from 'react';
import { Code } from '../content';
import { Drawing, Figure, Legend, Table, asInput, fixed, round } from './FormulationShared.jsx';
import {
  availabilityAxisGeometry, costThreshold, criticalQuantile, expectedLosses, flowGeometry, lineageGeometry,
  majorityBaseline, metricAxes, metricBarGeometry, nestedPartitionGeometry, optimalStock, selectedStripGeometry,
  splitLaneGeometry, stockingCost, thresholdPlotGeometry, upliftGeometry,
} from '../../data/formulation-models.js';
import { formulationData } from '../../data/formulation-data.js';

/** Inline figures for ML Problem Formulation, Baselines & Data Leakage.
 *
 * Every coordinate comes from formulation-models.js, never from a literal
 * typed into the drawing, so the verifier checks the same geometry a reader
 * sees. A drawn bar length and a drawn crossing point are quantitative claims.
 *
 * No figure carries a graded prediction: the two investigations own that
 * contract, and a figure that quietly answered one of their questions would
 * break it. Figure 5 is the one that shows measured scores, and it is placed
 * after the section that states them in prose, not before the investigation
 * that grades a change to them.
 */

const procedureShort = { prior: 'baseline', candidate: 'candidate', duration: '+ duration' };

/* ------------------------------------ figure 1 · outcome, prediction, action */

export function ContractFlowFigure() {
  const geometry = flowGeometry({});
  return <div className="formulation-figure">
    <p className="formulation-caption"><strong>Figure 1.</strong> A prediction is not a decision and a decision
      is not the outcome. The model produces the second box only; the capacity rule in the third box is what
      turns an estimate into an action, and the fifth box is the thing anyone actually cares about. The arrow
      back up the side returns <em>mature</em> outcomes to later training. There is no arrow from an outcome
      into a prediction about the same case, and that absence is the whole contract.</p>
    <p className="formulation-role is-constructed" role="status">A constructed schematic of the proposed
      pre-call system. The probability shown is an illustrative value, not a figure extracted from the bank
      records.</p>
    <Figure caption="Read downwards for one opportunity; the side channel is the only legitimate route backwards.">
      <Drawing kind="flow" width={geometry.width} height={geometry.height}
        caption="Five stages from available records to observed outcome, with a feedback channel"
        describe={`Five stacked boxes: ${geometry.boxes.map(box => box.title).join(', then ')}. `
          + 'A channel on the right returns the observed outcome to the available-record store; that return '
          + 'is legitimate only after the outcome exists. No arrow runs from the outcome forward into the '
          + 'prediction.'}>
        {geometry.arrows.map(arrow => (
          <line key={`${arrow.fromId}-${arrow.toId}`} className="form-arrow"
            x1={arrow.from.x} y1={arrow.from.y} x2={arrow.to.x} y2={arrow.to.y} markerEnd="url(#form-arrowhead)" />
        ))}
        {/* Its OWN marker. A shared arrowhead is filled once, so a purple dashed
            channel terminated in a green triangle -- the legend colour saying
            one thing and the mark saying another. */}
        <polyline className="form-arrow is-feedback" markerEnd="url(#form-feedback-head)"
          points={geometry.feedback.points.map(point => `${point.x},${point.y}`).join(' ')} />
        {geometry.boxes.map(box => <g key={box.id} className={`form-flow-box is-${box.id}`}>
          <rect x={box.x} y={box.y} width={box.width} height={box.height} rx={3} />
          <text className="form-label" x={box.centreX} y={box.y + 14} textAnchor="middle">{box.title}</text>
          <text className="form-small form-muted" x={box.centreX} y={box.y + 26} textAnchor="middle">{box.badge}</text>
        </g>)}
        <defs>
          <marker id="form-arrowhead" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path className="form-arrowhead" d="M0,0 L7,3.5 L0,7 z" />
          </marker>
          <marker id="form-feedback-head" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path className="form-arrowhead is-feedback" d="M0,0 L7,3.5 L0,7 z" />
          </marker>
        </defs>
      </Drawing>
    </Figure>
    <Table caption="What each stage is, and what it is not"
      headings={['Stage', 'What it holds']}
      rows={geometry.boxes.map(box => [box.title, box.detail])}
      footnote={'The side channel carries mature outcomes only: it is legitimate ' + geometry.feedback.note
        + '. Training on outcomes that have already happened is ordinary supervised learning; the failure is '
        + 'an outcome reaching a prediction that was supposed to be made before it existed. The channel is '
        + 'drawn as geometry and described here, because a sentence inside a 290-unit drawing scales with the '
        + 'figure instead of with your text.'} />
  </div>;
}

/* ------------------------------------------ figure 2 · one parcel, many rows */

export function UnitSplitFigure() {
  const geometry = splitLaneGeometry({});
  const [viewIndex, setViewIndex] = useState(0);
  const view = geometry.views[viewIndex];
  const names = ['Split the rows at random', 'Hold out whole parcel identities'];
  return <div className="formulation-figure">
    <p className="formulation-caption"><strong>Figure 2.</strong> Three parcels, each observed at hours 0, 1
      and 2: nine rows, three identities. The two allocations use the same nine rows and answer different
      questions. Watch the identity count, not the row count.</p>
    <p className="formulation-role is-constructed" role="status">A constructed fixture with nine rows. Three
      parcels are enough to show the mechanism and far too few to evaluate anything; no accuracy is claimed or
      drawn here.</p>
    <div className="formulation-buttons">
      {names.map((name, index) => <button key={name} type="button"
        className={viewIndex === index ? 'is-selected' : undefined} onClick={() => setViewIndex(index)}>
        {name}
      </button>)}
    </div>
    <Figure caption={view.identityOverlap > 0
      ? `Every one of the ${view.distinctTrainParcels} training identities also appears in validation, so this `
        + 'allocation asks how well the model interpolates among parcels it has already seen.'
      : `The ${view.distinctTrainParcels} training identities and the ${view.distinctValidationParcels} `
        + 'validation identity are disjoint, so this allocation asks about a parcel the model has never met.'}>
      <Drawing kind="lanes" width={geometry.width} height={geometry.height}
        caption={`${names[viewIndex]}: nine observations in three identity lanes`}
        describe={`Three rows labelled ${geometry.parcels.join(', ')}, each with three cells for hours `
          + `${geometry.hours.join(', ')}. ${view.trainRows} cells are marked training and `
          + `${view.validationRows} validation. ${view.identityOverlap} parcel identities appear on both sides.`}>
        {geometry.parcels.map((parcel, index) => (
          <text key={parcel} className="form-small" x={2}
            y={geometry.padding.top + index * geometry.laneHeight + 13}>{parcel}</text>
        ))}
        {view.cells.map(cell => <g key={cell.id} className={`form-cell is-${cell.side}`}>
          <rect x={cell.x} y={cell.y} width={cell.width} height={cell.height} rx={2} />
          <text className="form-small" x={cell.x + cell.width / 2} y={cell.y + cell.height / 2 + 3}
            textAnchor="middle">{`h${cell.hour} · ${cell.side === 'train' ? 'fit' : 'val'}`}</text>
        </g>)}
      </Drawing>
    </Figure>
    <Legend entries={[['train', 'fitted on'], ['validation', 'scored on']]} />
    <Table caption="The same nine rows, counted two ways"
      headings={['Allocation', 'Fit rows', 'Validation rows', 'Distinct fit parcels', 'Distinct validation parcels', 'Identities on both sides']}
      rows={geometry.views.map((entry, index) => [
        names[index], entry.trainRows, entry.validationRows,
        entry.distinctTrainParcels, entry.distinctValidationParcels,
        entry.identityOverlap ? entry.sharedParcels.join(', ') : 'none',
      ])}
      footnote={'Neither allocation is wrong in general. The row split is the right one if the deployment '
        + 'question really is “another hour from a parcel already in flight”. It is the wrong one if the '
        + 'question is “a parcel we have never seen”, and the row count alone cannot tell you which you have.'} />
  </div>;
}

/* ---------------------------------- figure 3 · where the cheaper action flips */

const COST_SETTINGS = [
  { id: 'even', label: 'equal costs: 5 and 5', falsePositiveCost: 5, falseNegativeCost: 5 },
  { id: 'section4', label: 'the section’s example: 3 and 9', falsePositiveCost: 3, falseNegativeCost: 9 },
  { id: 'reversed', label: 'reversed: 9 and 3', falsePositiveCost: 9, falseNegativeCost: 3 },
];

export function ThresholdCostFigure() {
  const [settingId, setSettingId] = useState('section4');
  const setting = COST_SETTINGS.find(entry => entry.id === settingId);
  const geometry = thresholdPlotGeometry({
    falsePositiveCost: setting.falsePositiveCost, falseNegativeCost: setting.falseNegativeCost,
  });
  const at = expectedLosses(0.2, setting.falsePositiveCost, setting.falseNegativeCost);
  return <div className="formulation-figure">
    <p className="formulation-caption"><strong>Figure 3.</strong> Two expected losses as the probability moves.
      Acting costs <em>C</em><sub>FP</sub>(1 − p); waiting costs <em>C</em><sub>FN</sub>p. They cross at
      exactly <em>C</em><sub>FP</sub>/(<em>C</em><sub>FP</sub> + <em>C</em><sub>FN</sub>), and .5 is that
      crossing only when the two costs are equal.</p>
    <p className="formulation-role is-constructed" role="status">Exact arithmetic on declared costs. Both lines
      are the formulas above evaluated at every p, and the marked crossing is the closed form, not a point read
      off the drawing.</p>
    <div className="formulation-buttons">
      {COST_SETTINGS.map(entry => <button key={entry.id} type="button"
        className={settingId === entry.id ? 'is-selected' : undefined} onClick={() => setSettingId(entry.id)}>
        {entry.label}
      </button>)}
    </div>
    <Figure caption={`With these costs the threshold is ${round(geometry.threshold, 6)}. At p = .2, acting costs `
      + `${round(at.acting, 6)} and waiting costs ${round(at.waiting, 6)}, so the cheaper choice is `
      + `to ${at.preferred === 'act' ? 'act' : at.preferred === 'wait' ? 'wait' : 'either: they are equal'}.`}>
      <Drawing kind="plot" width={geometry.width} height={geometry.height}
        caption="Expected loss of acting and of waiting, against the probability"
        describe={`Two straight lines on a probability axis from 0 to 1. Acting falls from `
          + `${geometry.falsePositiveCost} at p = 0 to 0 at p = 1; waiting rises from 0 to `
          + `${geometry.falseNegativeCost}. They cross at p = ${geometry.threshold.toFixed(4)}.`}>
        {geometry.yTicks.map(tick => <g key={`y${tick.value}`}>
          <line className="form-grid" x1={geometry.padding.left} y1={tick.y}
            x2={geometry.width - geometry.padding.right} y2={tick.y} />
          <text className="form-small" x={geometry.padding.left - 4} y={tick.y + 3}
            textAnchor="end">{round(tick.value, 2)}</text>
        </g>)}
        {geometry.xTicks.map((tick, index) => <g key={`x${tick.value}`}>
          <line className="form-tick" x1={tick.x} y1={geometry.height - geometry.padding.bottom}
            x2={tick.x} y2={geometry.height - geometry.padding.bottom + 4} />
          <text className="form-small" x={tick.x} y={geometry.height - geometry.padding.bottom + 14}
            textAnchor={index === 0 ? 'start' : index === geometry.xTicks.length - 1 ? 'end' : 'middle'}>
            {asInput(tick.value)}
          </text>
        </g>)}
        <line className="form-axis" x1={geometry.padding.left} y1={geometry.padding.top}
          x2={geometry.padding.left} y2={geometry.height - geometry.padding.bottom} />
        <line className="form-axis" x1={geometry.padding.left} y1={geometry.height - geometry.padding.bottom}
          x2={geometry.width - geometry.padding.right} y2={geometry.height - geometry.padding.bottom} />
        <line className="form-line is-acting" x1={geometry.actingLine.from.x} y1={geometry.actingLine.from.y}
          x2={geometry.actingLine.to.x} y2={geometry.actingLine.to.y} />
        <line className="form-line is-waiting" x1={geometry.waitingLine.from.x} y1={geometry.waitingLine.from.y}
          x2={geometry.waitingLine.to.x} y2={geometry.waitingLine.to.y} />
        <line className="form-cutoff" x1={geometry.crossing.x} y1={geometry.padding.top}
          x2={geometry.crossing.x} y2={geometry.height - geometry.padding.bottom} />
        <circle className="form-crossing" cx={geometry.crossing.x} cy={geometry.crossing.y} r={3} />
        <text className="form-small form-axis-title" x={geometry.width - geometry.padding.right}
          y={geometry.height - 3} textAnchor="end">probability p</text>
        <text className="form-small form-axis-title" x={2} y={10} textAnchor="start">expected loss</text>
      </Drawing>
    </Figure>
    <Legend entries={[['acting', 'cost of acting'], ['waiting', 'cost of waiting'], ['threshold', 'the crossing']]} />
    <p className="formulation-caption">
      The crossing moves with the costs, and nothing about the model changed between these three settings. This
      is also why an accuracy figure cannot stand in for a decision: accuracy fixes the comparison at .5 and
      then reports how often that particular rule was right.
    </p>
  </div>;
}

/* ------------------------- figure 4 · two boundaries that are not the same one */

export function FittingBoundaryFigure() {
  const partition = formulationData.partition;
  const geometry = nestedPartitionGeometry({ partition });
  const availability = availabilityAxisGeometry({});
  const prevalence = majorityBaseline({ negatives: 90, positives: 10 });
  return <div className="formulation-figure">
    <p className="formulation-caption"><strong>Figure 4.</strong> Two different boundaries, drawn separately
      because they are not the same boundary. The top pair is <em>which rows</em> a fitted transformation may
      look at. The bottom is <em>when</em> a value exists. A pipeline that fits its imputer and scaler only on
      the fit rows still crosses the second boundary if one of its columns is measured after the prediction had
      to be made.</p>
    <p className="formulation-role is-measured" role="status">The row counts are this lesson's actual fixed
      partition of the served file. The feature placement is the manuscript's reading of the provider's own
      variable description, not a measurement.</p>
    <Figure caption={`One scale across both bars: ${partition.sourceRows.toLocaleString('en-US')} rows fill the `
      + `width, so the second bar sits inside the development segment of the first. The `
      + `${partition.reservedRows} reserved rows receive no score anywhere on this page.`}>
      <Drawing kind="bars" width={geometry.width} height={2 * geometry.barHeight + 42}
        caption="The source rows split into development and reserved, then development into fit and validation"
        describe={`A bar of ${partition.sourceRows} rows divided into ${partition.developmentRows} development `
          + `and ${partition.reservedRows} reserved. Beneath it, on the same scale, the development portion is `
          + `divided into ${partition.trainRows} fit and ${partition.validationRows} validation rows.`}>
        {[geometry.first, geometry.second].map((bar, barIndex) => (
          <g key={barIndex} transform={`translate(0 ${barIndex * (geometry.barHeight + 21) + 4})`}>
            {bar.segments.map(segment => <g key={segment.id} className={`form-partition is-${segment.id}`}>
              <rect x={segment.x} y={0} width={segment.width} height={geometry.barHeight} rx={2} />
              <text className="form-small" x={segment.x + segment.width / 2} y={geometry.barHeight - 5}
                textAnchor="middle">{segment.rows.toLocaleString('en-US')}</text>
              <text className="form-small form-muted" x={segment.x + segment.width / 2}
                y={geometry.barHeight + 11} textAnchor="middle">{segment.label}</text>
            </g>)}
          </g>
        ))}
      </Drawing>
    </Figure>
    <Figure caption={'Four feature families placed against the moment the call happens. Everything to the left '
      + 'of the line is proposed as available before it; the family to the right is not available until '
      + 'afterwards. Fitting the pipeline only on the fit rows does not move it.'}>
      <Drawing kind="lanes" width={availability.width} height={availability.height}
        caption="Feature families placed before and after the call"
        describe={`${availability.beforeCount} families end at the call cutoff and `
          + `${availability.afterCount} begins there: ${availability.rows.filter(row => row.side === 'after')
            .map(row => row.label).join(', ')}.`}>
        {availability.rows.map(row => <g key={row.id} className={`form-availability is-${row.side}`}>
          <rect x={row.x} y={row.y + 3} width={row.width} height={availability.rowHeight - 7} rx={2} />
          <text className="form-small" x={row.side === 'before' ? row.x + 4 : row.x + row.width - 4}
            y={row.markY + 3} textAnchor={row.side === 'before' ? 'start' : 'end'}>{row.label}</text>
        </g>)}
        <line className="form-cutoff" x1={availability.cutoffX} y1={6}
          x2={availability.cutoffX} y2={availability.height - availability.padding.bottom + 2} />
        <text className="form-small form-axis-title" x={availability.cutoffX}
          y={availability.height - availability.padding.bottom + 15} textAnchor="middle">the call</text>
      </Drawing>
    </Figure>
    <Legend entries={[['before', 'proposed as available before the call'], ['after', 'does not exist until afterwards']]} />
    <p className="formulation-caption">
      The sentinel is its own case. The provider writes {formulationData.features.sentinel} in{' '}
      <Code>pdays</Code> to mean “no previous contact”, in
      {' '}{formulationData.features.sentinelRows.toLocaleString('en-US')} of
      the {partition.sourceRows.toLocaleString('en-US')} rows. That translation — an indicator plus a missing
      elapsed time — is a deterministic rule from the provider's documentation, applied to every row
      identically, and it reads no outcome. The median that later fills the missing values is{' '}
      <em>not</em> deterministic: it is fitted, on the fit rows only, inside the pipeline.
    </p>
    <p className="formulation-caption">
      One honest qualification about the top bar: the two splits are stratified on the outcome, so the reserved
      rows' labels were read to construct the partition. What does not happen is any reserved prediction, score,
      metric or selection. And a random split of these rows does not establish independence between clients; the
      file supplies no customer identity with which to check it. A class prevalence of {prevalence.prevalence * 100}%
      would let “always predict the majority” reach {prevalence.accuracy * 100}% accuracy
      with {prevalence.positiveRecall} recall of the class anyone cares about — which is why the baseline row is
      in figure 5 rather than assumed away.
    </p>
  </div>;
}

/* ----------------------------- figure 5 · the scores beside their contracts */

const METRIC_ORDER = ['averagePrecision', 'precisionAt50', 'logLoss'];

export function ResultsFigure() {
  const [metric, setMetric] = useState('averagePrecision');
  const rows = formulationData.procedures.map(procedure => ({
    id: procedure.id,
    short: procedureShort[procedure.id],
    averagePrecision: procedure.averagePrecision,
    logLoss: procedure.logLoss,
    precisionAt50: procedure.precisionAt50,
  }));
  const geometry = metricBarGeometry({ rows, metric });
  const axis = metricAxes[metric];
  const partition = formulationData.partition;
  return <div className="formulation-figure">
    <p className="formulation-caption"><strong>Figure 5.</strong> One metric at a time, each on its own axis
      with its own direction. The three procedures scored the same {partition.validationRows} validation rows,
      of which {partition.validationPositives} ended in a subscription. The row that scores best on every
      measure here is the row whose inputs the requested prediction could not have had, and it carries that
      mark rather than a winner's badge.</p>
    <p className="formulation-role is-measured" role="status">Measured. These are the recorded outputs of the
      fixed experiment printed above, on this development split of the served file. They are not an estimate of
      deployed performance and no reserved row contributed to any of them.</p>
    <div className="formulation-buttons">
      {METRIC_ORDER.map(name => <button key={name} type="button"
        className={metric === name ? 'is-selected' : undefined} onClick={() => setMetric(name)}>
        {metricAxes[name].label}
      </button>)}
    </div>
    <Figure caption={`${axis.label}, ${axis.better} is better, drawn on ${axis.domain[0]} to ${axis.domain[1]}. `
      + 'Switching metric switches the axis; the three bars are never placed on a shared numeric scale with a '
      + 'count of correct decisions.'}>
      <Drawing kind="bars" width={geometry.width} height={geometry.height} maxWidth={470}
        caption={`${axis.label} for the three procedures`}
        describe={geometry.bars.map(bar => `${bar.short}: ${bar.value.toFixed(6)}`).join('; ')}>
        {geometry.xTicks.map((tick, index) => <g key={tick.value}>
          <line className="form-grid" x1={tick.x} y1={geometry.padding.top - 6}
            x2={tick.x} y2={geometry.height - geometry.padding.bottom} />
          <text className="form-small" x={tick.x} y={geometry.height - geometry.padding.bottom + 13}
            textAnchor={index === 0 ? 'start' : index === geometry.xTicks.length - 1 ? 'end' : 'middle'}>
            {round(tick.value, 3)}
          </text>
        </g>)}
        {geometry.bars.map(bar => <g key={bar.id} className={`form-metric-bar is-${bar.id}`}>
          <rect x={bar.x} y={bar.y} width={bar.width} height={geometry.barHeight} rx={2} />
          <text className="form-small" x={bar.labelX} y={bar.labelY} textAnchor="end">{bar.short}</text>
          <text className="form-small" x={bar.valueX} y={bar.labelY} textAnchor="start">
            {round(bar.value, geometry.axis.digits)}{bar.id === 'duration' ? ' ⊘' : ''}
          </text>
        </g>)}
      </Drawing>
    </Figure>
    <Legend entries={[['prior', 'constant baseline'], ['candidate', 'recorded features'],
      ['duration', 'includes an input the prediction could not have had']]} />
    <Table caption="Every measured number, with the information contract each procedure needs"
      headings={['Procedure', 'Available before the call?', 'Average precision', 'Log loss ↓', 'Correct at .5',
        'Positives in the top 50']}
      rowClass={index => (formulationData.procedures[index].id === 'duration' ? 'is-unavailable' : undefined)}
      rows={formulationData.procedures.map(procedure => [
        procedure.label,
        procedure.id === 'duration' ? 'no — final call duration' : 'proposed; not established by this file',
        fixed(procedure.averagePrecision, 6),
        fixed(procedure.logLoss, 6),
        `${procedure.correct}/${partition.validationRows}`,
        procedure.top50Positives,
      ])}
      footnote={'Read the fourth and sixth columns together. The candidate makes one FEWER correct decision at '
        + 'threshold .5 than the constant baseline, and concentrates more than three times as many positives '
        + 'in the selected 50. Those two facts are about the same fitted model on the same rows.'} />
    <dl className="formulation-contracts">
      {formulationData.procedures.map(procedure => <div key={procedure.id}>
        <dt>{procedure.label}</dt>
        <dd><strong>Inputs:</strong> {procedure.features}<br /><strong>Availability:</strong> {procedure.availability}</dd>
      </div>)}
    </dl>
    <div className="formulation-strip-list">
      {formulationData.procedures.map(procedure => {
        const strip = selectedStripGeometry({
          capacity: partition.capacity, positives: procedure.top50Positives,
        });
        return <div key={procedure.id} className="formulation-strip-row">
          <span className="formulation-caption">{procedureShort[procedure.id]}: {strip.positives} of {strip.capacity}</span>
          <Drawing kind="strip" width={strip.width} height={strip.height} maxWidth={430}
            caption={`${procedureShort[procedure.id]}: ${strip.positives} of the selected ${strip.capacity} ended in a subscription`}
            describe={`A strip of ${strip.capacity} cells with ${strip.positives} marked.`}>
            {strip.cells.map(cell => (
              <rect key={cell.index} className={`form-strip-cell${cell.positive ? ' is-positive' : ''}`}
                x={cell.x} y={0} width={cell.width} height={strip.cellHeight} />
            ))}
          </Drawing>
        </div>;
      })}
    </div>
    <p className="formulation-caption">
      The strips are aggregate counts, not the ranked order: they say how many of a selected fifty ended in a
      subscription, not which. The baseline's {formulationData.procedures[0].top50Positives} are the arithmetic
      of a tie — every one of its scores is the same number, so its “top 50” is whichever fifty the source-row
      tie rule reaches first. Investigation 2 is where the identities become visible.
    </p>
  </div>;
}

/* ------------------------------------ figure 6 · three routes for a shortcut */

export function LineageFigure() {
  const geometry = lineageGeometry({});
  return <div className="formulation-figure">
    <p className="formulation-caption"><strong>Figure 6.</strong> Three different things called leakage, with
      three different repairs. Only the middle one is fixed by putting the transformation inside a pipeline.
      Naming which arrow you have is what selects the repair.</p>
    <p className="formulation-role is-constructed" role="status">A schematic of information routes, not a
      measurement. Each row is a path some real project has taken; none is drawn from the bank records.</p>
    <Figure caption="Follow each row left to right: that is the direction information travels, and each row travels it once.">
      <Drawing kind="flow" width={geometry.width} height={geometry.height}
        caption="Three information paths that each deliver disallowed information"
        describe={geometry.rows.map(row =>
          `${row.role}: ${row.nodes.map(node => node.lines.join(' ')).join(' → ')}`).join('. ')}>
        {geometry.rows.map(row => <g key={row.id} className={`form-lineage is-${row.kind}`}>
          {row.arrows.map((arrow, index) => (
            <line key={index} className="form-arrow" x1={arrow.from.x} y1={arrow.from.y}
              x2={arrow.to.x} y2={arrow.to.y} markerEnd="url(#form-lineage-head)" />
          ))}
          {row.nodes.map(node => <g key={node.position} className="form-lineage-node">
            <rect x={node.x} y={node.y} width={node.width} height={node.height} rx={3} />
            {node.lines.map((line, index) => (
              <text key={index} className="form-small" x={node.centreX} y={node.y + 12 + index * 10}
                textAnchor="middle">{line}</text>
            ))}
          </g>)}
        </g>)}
        <defs>
          <marker id="form-lineage-head" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path className="form-arrowhead" d="M0,0 L6,3 L0,6 z" />
          </marker>
        </defs>
      </Drawing>
    </Figure>
    <Table caption="The same three paths, with the repair each one needs"
      headings={['Path', 'What travels', 'Repair']}
      rows={geometry.rows.map(row => [row.role, row.title, row.repair])}
      footnote={'A single held-out split repairs none of the first and third rows. The first is a definition '
        + 'problem and the third is a procedure problem; only the second is about where a transformation was '
        + 'fitted.'} />
  </div>;
}

/* ----------------------- figure 7 · ranking by propensity against by impact */

const DEMANDS = [10, 20];
const DEMAND_PROBABILITIES = [0.5, 0.5];
const UNDERAGE = 3;
const OVERAGE = 1;

export function UpliftFigure() {
  const geometry = upliftGeometry({});
  const costs = DEMANDS.map(stock => ({
    stock,
    cost: stockingCost({
      stock, demands: DEMANDS, probabilities: DEMAND_PROBABILITIES,
      underageCost: UNDERAGE, overageCost: OVERAGE,
    }),
  }));
  const best = optimalStock({
    demands: DEMANDS, probabilities: DEMAND_PROBABILITIES, underageCost: UNDERAGE, overageCost: OVERAGE,
  });
  const mean = DEMANDS.reduce((sum, value, index) => sum + value * DEMAND_PROBABILITIES[index], 0);
  return <div className="formulation-figure">
    <p className="formulation-caption"><strong>Figure 7.</strong> Two groups on one probability axis. Ranking by
      the chance of subscribing under a call puts group {geometry.leaderByCalledProbability} first; ranking by
      how much the call changed the chance puts group {geometry.leaderByIncrement} first. The bars share a scale
      precisely so the reversal is visible rather than asserted.</p>
    <p className="formulation-role is-constructed" role="status">Hypothetical potential-outcome probabilities,
      invented to expose the distinction. The bank file records one outcome per customer under one historical
      policy and identifies no counterfactual, so no number here is an estimate from it.</p>
    <Figure caption={`Difference brackets: ${geometry.rows.map(row =>
      `${row.id} ${round(row.increment, 6)}`).join(', ')}. Both orderings are correct answers to different questions.`}>
      <Drawing kind="bars" width={geometry.width} height={geometry.height}
        caption="Subscription probability with and without a call, for two constructed groups"
        describe={geometry.rows.map(row => `group ${row.id}: ${row.called.value} if called, `
          + `${row.notCalled.value} if not, a difference of ${row.increment.toFixed(2)}`).join('; ')}>
        {geometry.xTicks.map((tick, index) => <g key={tick.value}>
          <line className="form-grid" x1={tick.x} y1={geometry.padding.top - 6}
            x2={tick.x} y2={geometry.height - geometry.padding.bottom} />
          <text className="form-small" x={tick.x} y={geometry.height - geometry.padding.bottom + 13}
            textAnchor={index === 0 ? 'start' : index === geometry.xTicks.length - 1 ? 'end' : 'middle'}>
            {asInput(tick.value)}
          </text>
        </g>)}
        {geometry.rows.map(row => <g key={row.id}>
          <text className="form-small" x={2} y={row.called.y + geometry.barHeight} textAnchor="start">{row.id}</text>
          <rect className="form-uplift is-called" x={geometry.padding.left} y={row.called.y}
            width={row.called.width} height={geometry.barHeight} rx={2} />
          <rect className="form-uplift is-notcalled" x={geometry.padding.left} y={row.notCalled.y}
            width={row.notCalled.width} height={geometry.barHeight} rx={2} />
          <line className="form-bracket" x1={row.bracket.fromX} y1={row.bracket.y}
            x2={row.bracket.toX} y2={row.bracket.y} />
          <text className="form-small" x={Math.max(row.called.endX, row.notCalled.endX) + 4}
            y={row.bracket.y + 3} textAnchor="start">{`+${round(row.increment, 2)}`}</text>
        </g>)}
      </Drawing>
    </Figure>
    <Legend entries={[['called', 'probability if called'], ['notcalled', 'probability if not called']]} />
    <Table caption="The same four numbers as exact values, with both orderings"
      headings={['Group', 'If called', 'If not called', 'Increment due to calling']}
      rows={geometry.rows.map(row => [row.id, round(row.called.value, 6), round(row.notCalled.value, 6),
        round(row.increment, 6)])}
      footnote={`Highest probability under the action: group ${geometry.leaderByCalledProbability}. `
        + `Largest increment: group ${geometry.leaderByIncrement}. `
        + (geometry.reverses ? 'The two orderings disagree.' : 'The two orderings agree here.')} />
    <Table caption={`A one-period stocking decision: demand is ${DEMANDS.join(' or ')}, each with probability `
      + `${DEMAND_PROBABILITIES[0]}, a missing part costs ${UNDERAGE} and an unused part costs ${OVERAGE}`}
      headings={['Stock', 'Expected cost']}
      rowClass={index => (costs[index].stock === best.stock ? 'is-leading' : undefined)}
      rows={costs.map(entry => [entry.stock, round(entry.cost, 6)])}
      footnote={`Mean demand is ${mean}, and stocking the mean is not available here; of the two levels that `
        + `are, the cheaper is ${best.stock}. The critical fraction is `
        + `${UNDERAGE}/(${UNDERAGE} + ${OVERAGE}) = ${round(criticalQuantile(UNDERAGE, OVERAGE), 6)}, and the `
        + `smallest demand level whose cumulative probability reaches it is ${best.stock}. A point prediction `
        + `of the mean, plus a rule that stocks it, is not the decision this cost structure asks for.`} />
    <p className="formulation-caption">
      Both tables are about the same thing from opposite ends. The first shows an output that answers the wrong
      question about <em>whom</em>; the second an output that answers the wrong question about <em>how much</em>.
      In neither case is the model's accuracy the problem. Section 4's acting threshold and this critical
      fraction are both one ratio of two costs, and they are not the same ratio written twice: the threshold
      puts the cost of acting wrongly on top, the stocking fraction puts the cost of running short on top. With
      these two numbers they come out at {round(costThreshold(OVERAGE, UNDERAGE), 6)}{' '}
      and {round(criticalQuantile(UNDERAGE, OVERAGE), 6)} — complements, not equals. Copying one formula into
      the other's problem is its own way of answering the wrong question.
    </p>
  </div>;
}
