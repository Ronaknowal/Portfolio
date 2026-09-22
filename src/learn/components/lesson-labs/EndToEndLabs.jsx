import { useState } from 'react';
import {
  Canvas, CheckboxSet, Count, CostField, HundredthsField, Investigation, Legend, PlotFrame, LiveResult,
  Score, Select, Table, asInput, fixed, useHeldOut, useInvestigation,
} from './EndToEndShared.jsx';
import {
  CUTOFF_MAX_HUNDREDTHS, CUTOFF_MIN_HUNDREDTHS, DEFERRAL_CASES, SCORE_MAX_HUNDREDTHS, SCORE_MIN_HUNDREDTHS,
  THRESHOLD_MAX_HUNDREDTHS, THRESHOLD_MIN_HUNDREDTHS, SELECTION_METRICS, acceptanceComparison,
  acceptanceLedger, acceptanceRailGeometry, asSelectionCriterion, candidateByKey, candidateOrder,
  cutoffValue, eligibleByDeclaration, flavanoidStripGeometry, scatterGeometry, selectionOutcome,
  sliceComparison,
} from '../../data/endtoend-models.js';
import { endToEndData } from '../../data/endtoend-data.js';



const stateKey = state => JSON.stringify(state);
const MODEL_OPTIONS = candidateOrder.map(key => [key, candidateByKey[key].short]);

/* ====================================== investigation B · development errors */

const SLICE_PRESETS = [
  ['The cutoff the lesson used, lower side', { cutoffHundredths: 400, side: 'lower' }],
  ['The same cutoff, upper side', { cutoffHundredths: 400, side: 'upper' }],
  ['A cutoff no specimen falls below', { cutoffHundredths: 0, side: 'lower' }],
  ['Compare a candidate with itself', { reference: 'linear_two', candidate: 'linear_two' }],
];

export function DevelopmentErrorLab() {
  const state = useInvestigation({
    reference: 'linear_two', candidate: 'linear_three', cutoffHundredths: 400, side: 'lower',
  }, stateKey);
  const [selectedId, setSelectedId] = useState(null);
  const draft = state.draft;
  const shown = state.result;

  const calculateInputs = inputs => {
    const comparison = sliceComparison(inputs);
    if (comparison.outcome === null) {
      return { outcome: null, value: null, comparison, explain: comparison.explain };
    }
    return {
      outcome: comparison.outcome,
      
      value: comparison.candidateErrors,
      comparison,
      explain: `${comparison.explain} The difference is `
        + `${comparison.difference > 0 ? '+' : ''}${comparison.difference}, made of `
        + `${comparison.repairedIds.length} repaired `
        + `(${comparison.repairedIds.join(', ') || 'none'}) and ${comparison.brokenIds.length} newly wrong `
        + `(${comparison.brokenIds.join(', ') || 'none'}).`,
    };
  };

  
  const geometry = scatterGeometry({
    cutoffHundredths: draft.cutoffHundredths, side: draft.side,
    reference: draft.reference, candidate: draft.candidate,
  });
  const strip = flavanoidStripGeometry({ selectedId });
  const inSliceIds = geometry.points.filter(point => point.inSlice).map(point => point.id);
  const selected = geometry.points.find(point => point.id === selectedId) ?? null;
  const comparison = shown?.calculation.comparison ?? null;

  return <Investigation investigationKey="slice" title="Investigation B · follow specimens, not just scores"
    question={'The 36 development specimens sit at their measured alcohol and colour intensity. Choose a '
      + 'colour-intensity cutoff and a side, pick the two candidates to compare inside that slice, and say '
      + 'whether the candidate will make fewer, the same number of, or more errors there than the reference '
      + 'does.'}
    role={{ kind: 'development', text: 'Every number in this investigation is a validation measurement on the '
      + '36 development specimens. The 36 test specimens are not in this explorer at all, and no quantity '
      + 'here is an estimate of performance on a new specimen.' }}
    note="Reset returns the comparison to linear_two against linear_three, the cutoff to 4, the lower side, and clears the selected specimen."
    onReset={() => { state.reset(); setSelectedId(null); }}>

    <div className="ete-controls">
      <Select label="Reference candidate" value={draft.reference} options={MODEL_OPTIONS}
        onChange={value => state.edit({ reference: value })}
        hint="Choosing the same candidate on both sides is supported; it is the null case." />
      <Select label="Candidate being compared" value={draft.candidate} options={MODEL_OPTIONS}
        onChange={value => state.edit({ candidate: value })} />
      <HundredthsField label="Colour-intensity cutoff" hundredths={draft.cutoffHundredths}
        min={CUTOFF_MIN_HUNDREDTHS} max={CUTOFF_MAX_HUNDREDTHS}
        onChange={value => state.edit({ cutoffHundredths: value })}
        hint="Any value from 0 to 14, to two decimals. The boundary belongs to the upper side: a specimen at exactly the cutoff is in ≥, not in <." />
      <Select label="Side of the cutoff" value={draft.side}
        options={[['lower', 'below the cutoff (<)'], ['upper', 'at or above the cutoff (≥)']]}
        onChange={value => state.edit({ side: value })} />
    </div>

    <div className="ete-presets">
      {SLICE_PRESETS.map(([label, update]) => (
        <button key={label} type="button"
          onClick={() => state.suggest({ ...draft, ...update })}>{label}</button>
      ))}
    </div>

    <PlotFrame geometry={geometry} className="ete-plot"
      xLabel="alcohol · source measurement scale" yLabel="colour intensity · source measurement scale"
      caption={`The 36 development specimens, with the slice ${draft.side === 'lower' ? 'below' : 'at or above'} `
        + `colour intensity ${asInput(cutoffValue(draft.cutoffHundredths))} picked out`}
      describe={'A scatter of the 36 validation specimens. Alcohol runs left to right from 11 to 15 and '
        + 'colour intensity runs bottom to top from 0 to 14, on the recorded source scale. Actual cultivar is '
        + 'shown by shape and named in the table below. A horizontal line marks the cutoff; specimens inside '
        + 'the chosen slice are drawn solid and the rest are dimmed. Positions never change when the '
        + 'candidate changes, because the plotting coordinates are the measurements and not the model. '
        + (shown ? 'Inside the slice, dashed inner rings mark reference errors and solid outer rings mark '
          + 'candidate errors; the identifiers are listed below.' : 'Error rings appear after comparison.')}>
      <line className="ete-cutoff" x1={geometry.padding.left} y1={geometry.cutoffY}
        x2={geometry.width - geometry.padding.right} y2={geometry.cutoffY} />
      {geometry.points.map(point => <g key={point.id}
        className={`ete-point is-class${point.actual}${point.inSlice ? ' is-in-slice' : ' is-out'}`
          + `${point.id === selectedId ? ' is-selected' : ''}`}>
        {point.actual === 0 && <circle cx={point.cx} cy={point.cy} r={4} />}
        {point.actual === 1 && <rect x={point.cx - 3.6} y={point.cy - 3.6} width={7.2} height={7.2} />}
        {point.actual === 2 && <polygon points={`${point.cx},${point.cy - 4.4} ${point.cx + 4.4},${point.cy + 3.2} `
          + `${point.cx - 4.4},${point.cy + 3.2}`} />}
      </g>)}
      {shown && geometry.points.filter(point => point.inSlice).map(point => <g key={`errors-${point.id}`}>
        {point.referenceWrong && <circle className="ete-error-ring is-reference"
          data-reference-error={point.id} cx={point.cx} cy={point.cy} r={6.5} />}
        {point.candidateWrong && <circle className="ete-error-ring is-candidate"
          data-candidate-error={point.id} cx={point.cx} cy={point.cy} r={9} />}
      </g>)}
    </PlotFrame>

    <Legend entries={[
      ['is-class0', 'cultivar 0 · circle'],
      ['is-class1', 'cultivar 1 · square'],
      ['is-class2', 'cultivar 2 · triangle'],
      ['is-in-slice', 'inside the chosen slice'],
      ['is-cutoff', 'the cutoff you set'],
      ...(shown ? [
        ['is-reference-error', `${candidateByKey[draft.reference].short} error · dashed inner ring`],
        ['is-candidate-error', `${candidateByKey[draft.candidate].short} error · solid outer ring`],
      ] : []),
    ]} />

    <p className="ete-caption">
      The slice holds <strong>{inSliceIds.length}</strong> of the 36 development specimens. That is a count of
      the slice, not a score: which candidate gets how many of them wrong is the question below.
      {shown && comparison.slice.n > 0 && <> After comparison, the reference&apos;s error IDs are{' '}
        {geometry.points.filter(point => point.inSlice && point.referenceWrong).map(point => point.id).join(', ') || 'none'};
        {' '}the candidate&apos;s are{' '}
        {geometry.points.filter(point => point.inSlice && point.candidateWrong).map(point => point.id).join(', ') || 'none'}.
      </>}
    </p>
    {/* Said once, here, because it is a limit of the picture rather than of the
        study. Two development specimens sit .05 apart in colour intensity on
        opposite sides of the default cutoff; this compact scale cannot separate them reliably,
        so the count is the authority and the figure is not. */}
    <p className="ete-caption">
      At this plot&apos;s compact scale, specimens within a few hundredths of the cutoff are hard to distinguish —
      two sit .05 apart on opposite sides of the default cutoff of 4. The count above and the specimen inspector
      below are the authority on which side a particular specimen falls; the scatter shows the shape of the
      slice, not its membership case by case.
    </p>

    <Canvas geometry={strip} className="ete-strip"
      caption="Flavanoids, the measurement the three-feature candidate adds, on its own scale"
      describe={'A one-dimensional strip of the 36 specimens\' flavanoid values. The added measurement gets '
        + 'its own axis rather than moving a point in the scatter: the model\'s input dimension changes when '
        + 'the candidate changes, and the plotting coordinates deliberately do not.'}>
      <line className="ete-axis" x1={strip.inset} y1={strip.railY} x2={strip.width - strip.inset}
        y2={strip.railY} />
      {strip.ticks.map(tick => <g key={tick.value}>
        <line className="ete-tick" x1={tick.x} y1={strip.railY} x2={tick.x} y2={strip.railY + 4} />
        <text className="ete-small" x={tick.x} y={strip.railY + 15} textAnchor="middle">{tick.value}</text>
      </g>)}
      {strip.marks.map(mark => <line key={mark.id}
        className={`ete-strip-mark${mark.selected ? ' is-selected' : ''}`}
        x1={mark.x} y1={strip.railY - 12} x2={mark.x} y2={strip.railY - 2} />)}
    </Canvas>

    <div className="ete-controls">
      <Select label="Inspect one specimen" value={selectedId === null ? '' : String(selectedId)}
        options={[['', 'none selected'],
          ...geometry.points.map(point => [String(point.id), `specimen ${point.id}`])]}
        onChange={value => setSelectedId(value === '' ? null : Number(value))}
        hint="Selecting a specimen highlights its unchanged position in both views and shows its row below." />
    </div>

    {selected && <Table caption={`Specimen ${selected.id}, whichever candidate is selected`}
      headings={['Field', 'Value']}
      rows={[
        ['Actual cultivar', String(selected.actual)],
        ['Alcohol', asInput(selected.alcohol)],
        ['Colour intensity', asInput(selected.colorIntensity)],
        ['Flavanoids', asInput(selected.flavanoids)],
        ['In the chosen slice', selected.inSlice ? 'yes' : 'no'],
        [`Predicted by ${candidateByKey[draft.reference].short}`,
          String(endToEndData.validationRows.find(row => row.id === selected.id).prediction[draft.reference])],
        [`Probability of the actual cultivar, ${candidateByKey[draft.reference].short}`,
          <Score key="pr" record={{ metric: 'probability of the actual cultivar',
            value: endToEndData.validationRows.find(row => row.id === selected.id)
              .probabilityOfActual[draft.reference], role: 'validation' }} />],
        [`Predicted by ${candidateByKey[draft.candidate].short}`,
          String(endToEndData.validationRows.find(row => row.id === selected.id).prediction[draft.candidate])],
        [`Probability of the actual cultivar, ${candidateByKey[draft.candidate].short}`,
          <Score key="pc" record={{ metric: 'probability of the actual cultivar',
            value: endToEndData.validationRows.find(row => row.id === selected.id)
              .probabilityOfActual[draft.candidate], role: 'validation' }} />],
      ]}
      footnote="These probabilities are the saved values at full precision, rounded only for display." />}

    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      
       />

    {comparison && comparison.slice.n > 0 && <div className="ete-reveal">
      <Table caption={`Errors inside ${comparison.slice.description}`}
        headings={['Candidate', 'Specimens in the slice', 'Errors', 'Error rate']}
        rows={[
          [candidateByKey[comparison.reference].short, String(comparison.slice.n),
            <Count key="r" part={comparison.referenceErrors} whole={comparison.slice.n} role="validation" />,
            <Score key="rr" digits={4} record={{ metric: 'error rate', role: 'validation',
              value: comparison.referenceErrors / comparison.slice.n }} />],
          [candidateByKey[comparison.candidate].short, String(comparison.slice.n),
            <Count key="c" part={comparison.candidateErrors} whole={comparison.slice.n} role="validation"
              quantityKey="slice-candidate-errors" />,
            <Score key="cr" digits={4} record={{ metric: 'error rate', role: 'validation',
              value: comparison.candidateErrors / comparison.slice.n }} />],
        ]}
        footnote={`Difference in error count: ${comparison.difference > 0 ? '+' : ''}${comparison.difference}. `
          + `Repaired inside the slice: ${comparison.repairedIds.join(', ') || 'none'}. Newly wrong inside the `
          + `slice: ${comparison.brokenIds.join(', ') || 'none'}. Equal counts do not mean the same specimens, `
          + 'which is why the identifiers are listed rather than summarised.'} />
      <p className="ete-caption">
        The cutoff is an explicit exploratory diagnostic, not a known biological boundary. Trying many cutoffs
        until one looks dramatic would turn chance variation into a story; the lesson keeps one cutoff fixed
        when it compares candidates, and so should a report.
      </p>
    </div>}
  </Investigation>;
}

/* ================================ investigation D · freeze, then open the test */

export function FreezeAndReportLab() {
  const gate = useHeldOut();
  const state = useInvestigation({
    eligible: [...eligibleByDeclaration], metricKey: 'validationBalancedAccuracy',
  }, stateKey);
  const draft = state.draft;
  const shown = state.result;
  const outcome = shown?.calculation.selection?.winner ? shown.calculation.selection : null;

  const calculateInputs = inputs => {
    const selection = selectionOutcome(inputs);
    if (!selection.winner) {
      return { outcome: null, value: null, selection, explain: selection.reason };
    }
    return {
      outcome: selection.winner,
      value: candidateByKey[selection.winner][inputs.metricKey].value,
      selection,
      /* The ranking is named in ORDER here and its numbers are left to the
         badged table above, which is where they already are. Repeating them in
         this sentence printed six-decimal scores with no role beside them --
         the same omission the review found in the compact report, in a place
         the page generates rather than an author types. */
      explain: `${selection.reason} Under ${selection.metric.label.toLowerCase()} the order is `
        + `${selection.ranking.map(key => candidateByKey[key].short).join(', then ')}`
        + ', on the scores in the table above'
        + `${selection.tiedWith.length
          ? `. ${selection.tiedWith.length} other candidate${selection.tiedWith.length === 1 ? '' : 's'} `
            + `scored exactly the same, and the declared candidate order broke the tie.`
          : '. No two candidates tied, so the tie rule did not come into it.'}`,
    };
  };

  const blocked = draft.eligible.length === 0
    ? 'No candidate is eligible, so there is nothing for a selection rule to choose between.'
    : null;

  return <Investigation investigationKey="freeze" title="Investigation D · freeze a decision, then open the test once"
    question={'Choose which candidates are in the comparison and which measure decides it, then say which '
      + 'candidate the rule selects. Only after that does this page show what the frozen model scored on the '
      + '36 specimens it has never touched.'}
    role={{ kind: 'selection', text: 'The scores this rule reads are validation measurements. Reading them to '
      + 'choose is what makes them selection evidence — the same numbers, a different claim.' }}
    note="Reset returns the comparison to the three declared candidates under validation balanced accuracy, and closes the held-out report again."
    onReset={() => { state.reset(); gate.release(); }}>

    <CheckboxSet legend="Candidates eligible for selection" selected={draft.eligible}
     
      hint="The study declared three. Including the baseline, or dropping a candidate, is an exploration of the development evidence rather than the comparison this study declared."
      options={candidateOrder.map(key => [key, candidateByKey[key].short, candidateByKey[key].purpose])}
      onChange={value => state.edit({ eligible: value })} />

    <div className="ete-controls">
      <Select label="Measure that decides" value={draft.metricKey}
        options={SELECTION_METRICS.map(metric => [metric.key,
          `${metric.label}${metric.declared ? ' — the declared one' : ''}`])}
        onChange={value => state.edit({ metricKey: value })}
        hint="Log loss is better when lower; the other two are better when higher. The rule knows which." />
    </div>
    {blocked && <p className="ete-note" role="status">{blocked}</p>}

    <Table caption="What each candidate scored on the development rows, before any of it is used to choose"
      headings={['Candidate', 'Inputs', 'Validation balanced accuracy', 'Validation accuracy',
        'Validation log loss']}
      rows={candidateOrder.map(key => [
        candidateByKey[key].short,
        candidateByKey[key].features.join(' + '),
        <Score key={`${key}-b`} record={candidateByKey[key].validationBalancedAccuracy} />,
        <Score key={`${key}-a`} record={candidateByKey[key].validationAccuracy} />,
        <Score key={`${key}-l`} record={candidateByKey[key].validationLogLoss} />,
      ])}
      footnote={'Every number in this table is a validation measurement. None of them is an estimate of what '
        + 'the chosen model will do on a new specimen, and the table says nothing about which one your rule '
        + 'picks — that is the question.'} />

    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs} blocked={blocked}
      
      
       />

    {outcome && <div className="ete-reveal">
      <p>
        {}
        The rule selects <strong>{candidateByKey[outcome.winner].short}</strong> with a selection criterion
        of <Score record={asSelectionCriterion(candidateByKey[outcome.winner][shown.inputs.metricKey])} />.
        {outcome.departsFromProtocol
          ? ' This is not the comparison the study declared before it looked at the scores.'
          : ' This is the comparison the study declared before it looked at the scores.'}
      </p>
      {outcome.heldOutAvailable
        ? <HeldOutStep selection={outcome} onEarned={gate.earn} earned={gate.earned} />
        : <p className="ete-refusal" role="status">
          <span className="ete-verdict-mark" aria-hidden="true">◼</span>
          {outcome.heldOutRefusal} Reset this investigation to the declared comparison if you want to see the
          report this study actually produced.
        </p>}
    </div>}
  </Investigation>;
}


function HeldOutStep({ selection, onEarned, earned }) {
  const state = useInvestigation({ selected: selection.winner }, stateKey);
  const validation = candidateByKey[selection.winner].validationBalancedAccuracy;
  const heldOut = endToEndData.heldOut.balancedAccuracy;
  const calculateInputs = () => {
    const difference = heldOut.value - validation.value;
    return {
      outcome: difference < -1e-12 ? 'lower' : difference > 1e-12 ? 'higher' : 'same',
      value: difference,
      explain: 'A held-out estimate is not required to be worse than the development one. Its specimens are '
        + 'different, and the estimate varies with which 36 specimens happened to land in it.',
    };
  };
  return <div className="ete-heldout-step">
    <p>
      <strong>The model is now frozen.</strong> Its fitted scaler and coefficients are exactly those learned on
      the {endToEndData.contract.trainRows} training rows, and nothing below will change them. Before the
      report opens, record what you expect.
    </p>
    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      />
    <button type="button" onClick={() => onEarned({ selected: selection.winner })} disabled={earned}>Open the frozen model’s held-out report</button>
  </div>;
}

/* ===================================== investigation C · acceptance and cost */

const ACCEPTANCE_PRESETS = [
  ['The stricter rule the text proposes', { thresholdHundredths: 80 }],
  ['Make a wrong answer as cheap as a deferral', { wrongCost: 2, deferCost: 2 }],
  ['A threshold no case reaches', { thresholdHundredths: 101 }],
  ['Move case 6 above the default threshold', { scoreHundredths: { 6: 95 } }],
];

export function AcceptanceCostLab() {
  const defaults = endToEndData.deferralDefaults;
  const state = useInvestigation({
    thresholdHundredths: Math.round(defaults.proposedThreshold * 100),
    wrongCost: defaults.wrongCost, deferCost: defaults.deferCost, scoreHundredths: {},
  }, stateKey);
  const draft = state.draft;
  const shown = state.result;

  /* The baseline is the rule the prose states, recomputed under whatever costs
     and scores are currently set: a cost edit must move both sides, or the
     comparison would be between two different scenarios. */
  const baselineFor = inputs => acceptanceLedger({
    thresholdHundredths: Math.round(defaults.threshold * 100),
    wrongCost: inputs.wrongCost, deferCost: inputs.deferCost, scoreHundredths: inputs.scoreHundredths,
  });
  const baseline = baselineFor(draft);
  const calculateInputs = inputs => {
    const proposed = acceptanceLedger(inputs);
    const comparison = acceptanceComparison(baselineFor(inputs), proposed);
    return {
      outcome: comparison.outcome,
      
      value: proposed.cost,
      proposed,
      baseline: baselineFor(inputs),
      comparison,
      explain: `${comparison.explain} Total cost moves by `
        + `${comparison.difference > 0 ? '+' : ''}${comparison.difference}.`,
    };
  };

  const rail = acceptanceRailGeometry({ ledger: baseline });
  const proposed = shown?.calculation.proposed ?? null;

  return <Investigation investigationKey="acceptance" title="Investigation C · allocate the cases the system does not answer"
    question={'Ten constructed inspection cases, each with a confidence score and a known outcome. The active '
      + 'rule answers every case at or above its threshold and defers the rest. Move the proposed threshold, '
      + 'edit what a wrong answer and a deferral cost, or change one case\'s score, then say which way the '
      + 'total cost will move.'}
    role={{ kind: 'constructed', text: 'A constructed teaching fixture, not this wine classifier. The scores '
      + 'are invented to expose the coverage and cost arithmetic and are not calibrated probabilities of '
      + 'anything.' }}
    constructedFixture={'ten invented inspection cases, not the development or held-out rows of this study, '
      + 'so none of the four evidence roles applies to their counts'}
    note="Reset returns the threshold to .8, the costs to 10 and 2, and every case score to its original value."
    onReset={state.reset}>

    <div className="ete-controls">
      <HundredthsField label="Proposed acceptance threshold" hundredths={draft.thresholdHundredths}
        min={THRESHOLD_MIN_HUNDREDTHS} max={THRESHOLD_MAX_HUNDREDTHS}
        onChange={value => state.edit({ thresholdHundredths: value })}
        hint="A case is answered when its score is at or above the threshold. 1.01 accepts nothing." />
      <CostField label="Cost of a wrong automatic answer" value={draft.wrongCost} min={0} max={50}
        onChange={value => state.edit({ wrongCost: value })} />
      <CostField label="Cost of a deferred case" value={draft.deferCost} min={0} max={20}
        onChange={value => state.edit({ deferCost: value })}
        hint="Both rules are costed with the same numbers, so editing a cost moves the baseline too." />
    </div>

    <div className="ete-controls ete-score-edits">
      {DEFERRAL_CASES.map(item => (
        <HundredthsField key={item.id} label={`Case ${item.id} score`}
          hundredths={draft.scoreHundredths[item.id] ?? Math.round(item.confidence * 100)}
          min={SCORE_MIN_HUNDREDTHS} max={SCORE_MAX_HUNDREDTHS}
          onChange={value => state.edit({
            scoreHundredths: { ...draft.scoreHundredths, [item.id]: value },
          })} />
      ))}
    </div>
    <p className="ete-caption">
      Editing a score changes which side of the threshold a case falls on. It never changes whether that case
      was answered correctly: this is a proposed decision rule over recorded outcomes, not retraining.
    </p>

    <div className="ete-presets">
      {ACCEPTANCE_PRESETS.map(([label, update]) => (
        <button key={label} type="button"
          onClick={() => state.suggest({ ...draft, ...update })}>{label}</button>
      ))}
    </div>

    <Canvas geometry={rail} className="ete-rail"
      caption={`The active rule: threshold ${asInput(baseline.threshold)}, answering `
        + `${baseline.accepted} of ${baseline.cases.length} cases`}
      describe={'Ten tiles on a confidence rail running from .5 to 1, each labelled with its case number and '
        + 'score. A vertical line marks the active threshold. Tiles at or above it are routed to the '
        + 'automatic-answer queue and the rest to the deferred queue; both queues are listed by case number '
        + 'below the rail. Tied or nearby scores stack vertically at their exact score positions; '
        + 'height has no quantitative meaning.'}>
      <line className="ete-axis" x1={rail.inset} y1={rail.railY} x2={rail.width - rail.inset} y2={rail.railY} />
      {rail.ticks.map(tick => <g key={tick.value}>
        <line className="ete-tick" x1={tick.x} y1={rail.railY} x2={tick.x} y2={rail.railY + 4} />
        <text className="ete-small" x={tick.x} y={rail.railY + 15} textAnchor="middle">{tick.value}</text>
      </g>)}
      {!rail.thresholdBeyondRail && rail.thresholdSegments.map(segment => (
        <line key={segment.y1} className="ete-cutoff" x1={rail.thresholdX} y1={segment.y1}
          x2={rail.thresholdX} y2={segment.y2} />
      ))}
      {rail.tiles.map(tile => <g key={tile.id}
        className={`ete-tile${tile.accepted ? ' is-accepted' : ' is-deferred'}`
          + `${tile.correct ? '' : ' is-wrong'}${tile.edited ? ' is-edited' : ''}`}>
        <rect x={tile.x - 7} y={tile.y - 20} width={14} height={14} rx="2" />
        <text className="ete-small" x={tile.x} y={tile.labelY} textAnchor="middle">{tile.id}</text>
      </g>)}
      {rail.queues.map(queue => <g key={queue.key}>
        <text className="ete-small ete-strong" x={rail.inset} y={queue.y}>{queue.label}</text>
        <text className="ete-small" x={rail.inset} y={queue.y + 13}>
          {queue.ids.length ? `cases ${queue.ids.join(', ')}` : 'no cases'}
        </text>
      </g>)}
    </Canvas>

    <p className="ete-caption">Tied or nearby scores stack vertically so every case stays visible.
      Height has no quantitative meaning; the horizontal position is still the exact score.</p>

    <Legend entries={[
      ['is-accepted', 'answered automatically'],
      ['is-deferred', 'deferred to a person'],
      ['is-wrong', 'the recorded outcome was wrong'],
      ['is-cutoff', 'the active threshold'],
    ]} />

    <Table caption="The ledger of the active rule, which is the baseline your proposal is compared against"
      headings={['Quantity', 'Value']}
      rows={[
        /* No role badge. These ten cases are a constructed fixture, not this
           study's development rows, and calling their count "validation" would
           be the mislabelling the lesson exists to teach against. The whole
           investigation is declared a constructed fixture instead. */
        ['Answered', `${baseline.accepted} of ${baseline.cases.length}`],
        ['Wrong among answered', String(baseline.wrong)],
        ['Deferred', String(baseline.deferred)],
        ['Coverage', fixed(baseline.coverage, 4)],
        ['Conditional error among answered',
          baseline.conditionalError === null ? baseline.conditionalErrorNote : fixed(baseline.conditionalError, 4)],
        ['Total observed cost', String(baseline.cost)],
      ]}
      footnote={`Total cost is ${asInput(baseline.wrongCost)} per wrong automatic answer plus `
        + `${asInput(baseline.deferCost)} per deferred case. The deferred cases are work someone still does; a `
        + 'comparison that counted only the answered ones would hide that work entirely.'} />

    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      
       />

    {proposed && <div className="ete-reveal">
      <Table caption="The proposed rule beside the active one"
        headings={['Quantity', `Active · ${asInput(shown.calculation.baseline.threshold)}`,
          `Proposed · ${asInput(proposed.threshold)}`]}
        rows={[
          ['Answered', String(shown.calculation.baseline.accepted), String(proposed.accepted)],
          ['Wrong among answered', String(shown.calculation.baseline.wrong), String(proposed.wrong)],
          ['Deferred', String(shown.calculation.baseline.deferred), String(proposed.deferred)],
          ['Coverage', fixed(shown.calculation.baseline.coverage, 4), fixed(proposed.coverage, 4)],
          ['Conditional error',
            shown.calculation.baseline.conditionalError === null
              ? shown.calculation.baseline.conditionalErrorNote : fixed(shown.calculation.baseline.conditionalError, 4),
            proposed.conditionalError === null
              ? proposed.conditionalErrorNote : fixed(proposed.conditionalError, 4)],
          ['Total observed cost', String(shown.calculation.baseline.cost),
            <span key="pc" data-graded-quantity="acceptance-proposed-cost">{proposed.cost}</span>],
          ['Answered case numbers', shown.calculation.baseline.acceptedIds.join(', ') || 'none',
            proposed.acceptedIds.join(', ') || 'none'],
          ['Deferred case numbers', shown.calculation.baseline.deferredIds.join(', ') || 'none',
            proposed.deferredIds.join(', ') || 'none'],
        ]}
        footnote={'Better conditional accuracy among answered cases need not lower the total: the deferred '
          + 'cases keep costing whatever they cost. With a wrong answer priced at 2 rather than 10 the same '
          + 'threshold move reverses direction, which is worth trying before leaving this investigation.'} />
    </div>}
  </Investigation>;
}
