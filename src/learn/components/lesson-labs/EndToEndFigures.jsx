import { CodeBlock, H3, Prose } from '../content';
import { Canvas, Count, HeldOutOnly, Legend, NonScore, PlotFrame, Score, Table } from './EndToEndShared.jsx';
import { endToEndExamples } from '../../data/endtoend-examples.js';
import {
  candidateBarGeometry, candidateByKey, classRecallTable, informationLaneGeometry,
  pairedStripGeometry, recallComparisonGeometry, selectionMetricByKey, fixed,
} from '../../data/endtoend-models.js';

/** The lesson's inline figures. Every coordinate comes from the model layer, so
 *  the verifier asserts the same numbers the browser paints; nothing here places
 *  a mark by hand.
 *
 *  Each `<svg>` carries one of this lesson's own layout classes -- ete-lanes,
 *  ete-strip, ete-bars or ete-plot -- because the stylesheet's layout rule is
 *  scoped to those classes rather than to every svg under the lesson root. The
 *  hygiene checker refuses an untagged one.
 */

/* ==================================================== the study, and its output */

/**
 * The complete program, and the half of its output the page is allowed to show
 * before the decision is frozen.
 *
 * The program is displayed whole, exactly as the manuscript displays it,
 * because the final block is part of the research sequence a learner needs to
 * read. What is split is its OUTPUT: the twelve development lines appear here,
 * and the four lines that report the test set appear in section 6, once the
 * decision they report on has been frozen. The two halves are checked by
 * scripts/verify-endtoend-examples.py to concatenate back to the exact bytes
 * the program printed, so the split cannot quietly lose a line.
 *
 * A learner who runs the file locally sees everything at once, which is the
 * point: the discipline is the researcher's, not the interpreter's.
 */
export function StudyProgram() {
  const example = endToEndExamples.study;
  return <section className="python-example">
    <H3>{example.title}</H3>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <Prose>
      Save the program as <code>{example.file}</code> beside{' '}
      <a href={example.dataset} download>wine.csv</a>, which this page serves, or
      download <a href={example.download} download>the program itself</a>. The CSV already contains the input,
      so running the example downloads nothing.
    </Prose>
    <CodeBlock language="sh">{example.setup}</CodeBlock>
    <CodeBlock language={example.language}>{example.code}</CodeBlock>
    <Prose dim>What it prints, up to the moment it opens the test set:</Prose>
    {/* Declared program output. The shared CodeBlock renders a styled <div>
        rather than a <pre>, so the unbadged-score sweep cannot recognise it by
        element name; it reads this marker instead. Program stdout is the one
        place a role badge cannot go — the bytes are the program's, not the
        page's — and the surrounding prose names what the three numbers are. */}
    <div data-program-output="the study program's own stdout, which the page reproduces verbatim">
      <CodeBlock language="output">{example.developmentOutput}</CodeBlock>
    </div>
    <Prose>
      The three numbers after each candidate name are balanced accuracy, accuracy and log loss, all measured on
      the 36 development specimens. The program does open the test set in its final block — the last four
      printed lines are in section 6, after the decision those lines report on has been frozen.
    </Prose>
  </section>;
}

/** The held-out half of the same execution. Rendered only inside the gate. */
export function StudyHeldOutOutput() {
  const example = endToEndExamples.study;
  return <HeldOutOnly placeholder={'The last four lines the program printed open the test set. They appear '
    + 'here once the decision they report on has been frozen in the investigation above.'}>
    <div data-program-output="the study program's own stdout, which the page reproduces verbatim">
      <CodeBlock language="output">{example.heldOutOutput}</CodeBlock>
    </div>
  </HeldOutOnly>;
}

/* ============================================ figure A · where information goes */

function Arrow({ arrow, markerId }) {
  return <g className={`ete-arrow is-${arrow.kind}`}>
    <line x1={arrow.x1} y1={arrow.y1} x2={arrow.x2} y2={arrow.y2} markerEnd={`url(#${markerId}-${arrow.kind})`} />
  </g>;
}

export function InformationLaneFigure() {
  const geometry = informationLaneGeometry();
  const markerId = 'ete-lane-head';
  return <>
    <Canvas geometry={geometry} className="ete-lanes"
      caption="Figure A · which specimens may reach which fitted object, and where each result may go"
      describe={'Three horizontal lanes, Train 106, Validation 36 and Test 36, sit at the left. All three '
        + 'reach one fitted-pipeline box containing the scaler\'s stored mean and scale, the classifier\'s '
        + 'fitted coefficients, and the transform-then-predict step. Only the training lane fits it; the other '
        + 'two only apply it. From the pipeline, validation scores reach a candidate-selection box and the '
        + 'test result reaches a final-report box. A closed gate sits on the test lane. A dashed '
        + 'counterexample arrow runs from the final report back to candidate selection, marked as the '
        + 'mistake. No arrow runs from a target label into a transform, and no arrow runs from the test lane '
        + 'into candidate selection.'}>
      <defs>
        {['fits', 'applies', 'informs', 'reports', 'counterexample'].map(kind => (
          <marker key={kind} id={`${markerId}-${kind}`} viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path className={`ete-arrow-head is-${kind}`} d="M 0 1 L 7 4 L 0 7 z" />
          </marker>
        ))}
      </defs>

      <rect className="ete-pipeline-box" x={geometry.pipeline.x} y={geometry.pipeline.y}
        width={geometry.pipeline.width} height={geometry.pipeline.height} rx="4" />
      <text className="ete-small ete-strong" x={geometry.pipeline.centreX}
        y={geometry.pipeline.titleY} textAnchor="middle">one fitted pipeline</text>
      {geometry.pipeline.parts.map(part => (
        <text key={part.label} className="ete-small" x={geometry.pipeline.centreX}
          y={part.y} textAnchor="middle">{part.label}</text>
      ))}

      {/* One short label per lane. What each lane may change is a sentence, and
          a sentence belongs in the table below, where it can reflow. */}
      {geometry.lanes.map(lane => <g key={lane.key} className={`ete-lane is-${lane.key}`}>
        <rect x={lane.x} y={lane.y} width={lane.width} height={lane.height} rx="3" />
        <text className="ete-small ete-strong" x={lane.x + 6} y={lane.labelY}>{lane.label}</text>
      </g>)}

      <g className="ete-gate">
        <rect x={geometry.gateOnTest.x} y={geometry.gateOnTest.y}
          width={geometry.gateOnTest.size} height={geometry.gateOnTest.size} rx="2" />
        <line x1={geometry.gateOnTest.x + 4} y1={geometry.gateOnTest.y + geometry.gateOnTest.size / 2}
          x2={geometry.gateOnTest.x + geometry.gateOnTest.size - 4}
          y2={geometry.gateOnTest.y + geometry.gateOnTest.size / 2} />
      </g>

      {geometry.arrows.map(arrow => <Arrow key={arrow.key} arrow={arrow} markerId={markerId} />)}

      {geometry.sinks.map(sink => <g key={sink.key} className={`ete-sink is-${sink.key}`}>
        <rect x={sink.x} y={sink.y} width={sink.width} height={sink.height} rx="3" />
        <text className="ete-small" x={sink.x + sink.width / 2} y={sink.y + 15}
          textAnchor="middle">{sink.label}</text>
      </g>)}
    </Canvas>

    <Legend entries={[
      ['is-fits', 'fits the pipeline — training only'],
      ['is-applies', 'applies the already-fitted pipeline'],
      ['is-informs', 'scores inform the choice'],
      ['is-reports', 'reports once, after the choice'],
      ['is-counterexample', 'the mistake: reading the report and revising the model'],
    ]} />

    <Table caption="What each lane is allowed to change, and what it is not"
      headings={['Lane', 'Where its measurements come from', 'What it may change', 'What it must not do']}
      rows={[
        ['Train, 106 specimens', 'The fitted scaler\'s mean and scale and the classifier\'s coefficients',
          'The model\'s fitted state', 'Nothing here is an estimate of performance on a new specimen'],
        ['Validation, 36 specimens', 'The already-fitted pipeline\'s predictions',
          'The researcher\'s choice of candidate', 'It cannot also be the independent estimate of that choice'],
        ['Test, 36 specimens', 'The same already-fitted pipeline\'s predictions',
          'Nothing, until the candidate is frozen', 'It must not be read and then used to revise the model'],
      ]}
      footnote={'The dashed arrow is the counterexample, not part of the protocol: a researcher who reads the '
        + 'test errors, adds a feature to repair them, and reports the improved score on those same rows has '
        + 'turned the test set into development data. The information travelled through a person, which is '
        + 'exactly as effective as a function call.'} />
  </>;
}

/* ================================ figure B · what the declared comparison found */

export function CandidateComparisonFigure({ metricKey = 'validationBalancedAccuracy' }) {
  const geometry = candidateBarGeometry({ metricKey });
  const metric = selectionMetricByKey[metricKey];
  return <>
    <PlotFrame geometry={geometry} className="ete-bars"
      xLabel="candidate" yLabel={metric.label.toLowerCase()}
      caption={`Figure B · ${metric.label.toLowerCase()} for all four candidates, with the baseline kept in `
        + 'the frame'}
      describe={'A bar for each of the four candidates. The majority baseline is included so the comparison '
        + 'has an anchor: without it every bar looks similar and the question "does measuring anything help" '
        + 'has no visible answer.'}>
      {geometry.bars.map(bar => <g key={bar.key} className={`ete-bar${bar.isBaseline ? ' is-baseline' : ''}`}>
        <rect x={bar.x} y={bar.y} width={bar.barWidth} height={bar.barHeight} />
        <text className="ete-small" x={bar.labelX} y={bar.y - 4} textAnchor="middle">
          {bar.value.toFixed(3)}
        </text>
      </g>)}
      {geometry.bars.map(bar => (
        <text key={`label-${bar.key}`} className="ete-small ete-muted" x={bar.labelX}
          y={geometry.height - geometry.padding.bottom + 28} textAnchor="middle">{bar.label}</text>
      ))}
    </PlotFrame>
    <p className="ete-caption">
      Every bar is a validation measurement. Reading them to pick one is what turns the chosen bar into a
      selection criterion, and that is a different claim about the same number.
    </p>
  </>;
}

/* ================================ figure C · a net gain contains two directions */

export function PairedChangeFigure({ reference = 'linear_two', candidate = 'linear_three' }) {
  const geometry = pairedStripGeometry({ reference, candidate });
  const change = geometry.change;
  const changed = change.rows.filter(row => row.state === 'repaired' || row.state === 'new error');
  return <>
    <Canvas geometry={geometry} className="ete-strip"
      caption={`Figure C · the same 36 development specimens before and after, in identifier order`}
      describe={`Two rows of 36 marks, one per validation specimen in identifier order. The upper row is `
        + `whether ${candidateByKey[reference].label} was right; the lower row is whether `
        + `${candidateByKey[candidate].label} was right. A specimen whose state changes is linked between `
        + `the rows and named. ${change.repaired.length} move from wrong to correct and `
        + `${change.broken.length} move from correct to wrong, taking ${change.referenceCorrect} correct to `
        + `${change.candidateCorrect}.`}>
      <text className="ete-small ete-muted" x={geometry.inset} y={geometry.rowLabelY.before}>
        before · {candidateByKey[reference].short}
      </text>
      <text className="ete-small ete-muted" x={geometry.inset} y={geometry.rowLabelY.after}>
        after · {candidateByKey[candidate].short}
      </text>
      {geometry.columns.map(column => <g key={column.id}
        className={`ete-cell is-${column.state.replace(/\s+/g, '-')}`}>
        <rect className={`ete-mark ${column.before ? 'is-right' : 'is-wrong'}`}
          x={column.left} y={column.beforeY} width={column.barWidth} height={column.markHeight} />
        <rect className={`ete-mark ${column.after ? 'is-right' : 'is-wrong'}`}
          x={column.left} y={column.afterY} width={column.barWidth} height={column.markHeight} />
        {column.changed && <line className={`ete-link is-${column.state.replace(/\s+/g, '-')}`}
          x1={column.x} y1={column.beforeY + column.markHeight} x2={column.x} y2={column.afterY} />}
      </g>)}
    </Canvas>

    <Legend entries={[
      ['is-right', 'correct'],
      ['is-wrong', 'wrong'],
      ['is-repaired', 'repaired: wrong before, correct after'],
      ['is-new-error', 'new error: correct before, wrong after'],
    ]} />

    <p className="ete-caption">
      <Count part={change.referenceCorrect} whole={change.total} role="validation" noun="correct" /> before
      and <Count part={change.candidateCorrect} whole={change.total} role="validation" noun="correct" /> after.
      The net move of {change.net > 0 ? '+' : ''}{change.net} is {change.repaired.length} repairs
      minus {change.broken.length} new {change.broken.length === 1 ? 'error' : 'errors'}, which is the whole
      point of keeping the identifiers aligned: the average alone cannot tell you that anything moved
      backwards.
    </p>

    <Table caption="The specimens whose outcome changed, by identifier"
      headings={['Specimen', 'Actual cultivar', 'Before', 'After', 'Change']}
      rows={changed.map(row => [
        String(row.id), String(row.actual),
        row.before ? 'correct' : 'wrong', row.after ? 'correct' : 'wrong', row.state,
      ])}
      rowClass={index => `is-${changed[index].state.replace(/\s+/g, '-')}`}
      footnote={`The other ${change.total - changed.length} specimens keep the same outcome under both `
        + 'candidates. They are not evidence that the two models agree about them for the same reason — only '
        + 'that the outcome did not move.'} />
  </>;
}

/* ============================= figure D · where the improvement went backwards */

export function RecallComparisonFigure({ reference = 'linear_two', candidate = 'linear_three' }) {
  const geometry = recallComparisonGeometry({ reference, candidate });
  const referenceTable = classRecallTable(candidateByKey[reference].validationConfusion);
  const candidateTable = classRecallTable(candidateByKey[candidate].validationConfusion);
  return <>
    <PlotFrame geometry={geometry} className="ete-bars"
      xLabel="actual cultivar" yLabel="validation recall"
      caption="Figure D · per-class recall before and after the added measurement"
      describe={'Three pairs of bars, one pair per cultivar. Each pair shows the recall of the two-feature '
        + 'linear model and of the three-feature linear model on the validation specimens of that cultivar. '
        + 'Cultivars 1 and 2 rise to every specimen correct; cultivar 0 falls. Balanced accuracy is the mean '
        + 'of these three numbers, so it rises while one of them falls.'}>
      {geometry.groups.map(group => <g key={group.actual}
        className={`ete-recall-group${group.worsened ? ' is-worsened' : ''}`}>
        {group.bars.map(bar => <g key={bar.key} className={`ete-bar is-${bar.key}`}>
          <rect x={bar.x} y={bar.y} width={bar.barWidth} height={bar.barHeight} />
          <text className="ete-small" x={bar.x + bar.barWidth / 2} y={bar.y - 4} textAnchor="middle">
            {bar.correct}/{bar.support}
          </text>
        </g>)}
        <text className="ete-small ete-muted" x={group.labelX}
          y={geometry.height - geometry.padding.bottom + 28} textAnchor="middle">
          cultivar {group.actual}
        </text>
      </g>)}
    </PlotFrame>

    <Legend entries={[
      [`is-${reference}`, `${candidateByKey[reference].short} — two measurements`],
      [`is-${candidate}`, `${candidateByKey[candidate].short} — three measurements`],
      ['is-worsened', 'the cultivar whose recall fell'],
    ]} />

    <Table caption="The same three numbers, with their denominators"
      headings={['Actual cultivar', 'Validation specimens', `${candidateByKey[reference].short} correct`,
        `${candidateByKey[candidate].short} correct`, 'Recall moved']}
      rows={referenceTable.map((row, index) => [
        `cultivar ${row.actual}`,
        String(row.support),
        <Score key={`r${index}`} digits={4}
          record={{ metric: `recall for cultivar ${row.actual}`, value: row.recall, role: 'validation' }} />,
        <Score key={`c${index}`} digits={4} record={{ metric: `recall for cultivar ${row.actual}`,
          value: candidateTable[index].recall, role: 'validation' }} />,
        candidateTable[index].recall > row.recall ? 'up'
          : candidateTable[index].recall < row.recall ? 'down' : 'unchanged',
      ])}
      rowClass={index => (candidateTable[index].recall < referenceTable[index].recall ? 'is-worsened' : undefined)}
      footnote={'Balanced accuracy is the mean of the three recall values in the two middle columns, so a '
        + 'cultivar with fewer specimens still contributes a third of it. Two of the three rise to every '
        + `specimen correct while cultivar ${geometry.worsenedClasses.join(' and ')} falls, and the mean `
        + 'rises anyway.'} />
  </>;
}

/* =================================== figure E · training against validation */

export function FitAndGeneralisationFigure() {
  const rows = ['linear_two', 'forest_two', 'linear_three'].map(key => candidateByKey[key]);
  return <Table caption="What each candidate achieved on the rows it was fitted on, and on the rows it was not"
    headings={['Candidate', 'Training balanced accuracy', 'Validation balanced accuracy', 'Gap']}
    rows={rows.map(candidate => [
      candidate.short,
      <Score key={`${candidate.key}-t`} record={candidate.trainingBalancedAccuracy} />,
      <Score key={`${candidate.key}-v`} record={candidate.validationBalancedAccuracy} />,
      /* The one printed number on this page that is in none of the four roles:
         it subtracts a validation quantity from a training one. Rather than
         print it bare -- which would make the "every score carries its role"
         sweep unenforceable, since it could not tell a deliberate exception
         from an omission -- the exception is declared on the element with its
         reason, and the sweep requires that reason to be there. */
      <NonScore key={`${candidate.key}-gap`}
        because={'a training quantity minus a validation one, which is a diagnostic about the fit and is not '
          + 'a score in either role'}>
        {fixed(candidate.trainingBalancedAccuracy.value - candidate.validationBalancedAccuracy.value, 6)}
      </NonScore>,
    ])}
    footnote={'The gap column subtracts a validation quantity from a training quantity, which is the only '
      + 'thing on this page that mixes two roles in one number — and it is a diagnostic about the fit, not an '
      + 'estimate of anything. The forest has the widest gap and the middle validation score; a bigger gap on '
      + 'its own neither condemns nor recommends a candidate.'} />;
}
