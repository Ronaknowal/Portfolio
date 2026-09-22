import { useState } from 'react';
import {
  dependentGames, explainPolynomial, explainTreeInstance, extrapolationFlags, forwardSelection,
  informationFromCounts, limits, maskLabel, permutationExperiment, permutationSuite,
  probabilityMosaic, treeDecision, waterfall,
} from '../../data/selection-models';
import {
  alternativeReference, backgroundIds, backgroundRows, changedInference, explainedCases, fourFieldModel,
} from '../../data/selection-data';
import {
  Field, Folded, Investigation, NumberField, LiveResult, Table, TreeDiagram, Waterfall,
  fixed, round, settled, signed, unsigned, useInvestigation,
} from './SelectionShared.jsx';
import { ScrollRegion } from './SelectionShared.jsx';
import './selection-labs.css';

const SMALL_LABELS = fourFieldModel.labels;
const SHORT_LABELS = ['alcohol', 'malic', 'flavan.', 'proline'];
const TREE = fourFieldModel.tree;

/* ============================================================ §2 · I1 */

const countPresets = {
  base: { label: 'Baseline: 3/1/1/3', counts: [[3, 1], [1, 3]] },
  perfect: { label: 'Contrast: 4/0/0/4', counts: [[4, 0], [0, 4]] },
  none: { label: 'Null for association: 2/2/2/2', counts: [[2, 2], [2, 2]] },
  scaled: { label: 'Null for the estimate: every count × 3', counts: [[9, 3], [3, 9]] },
  practice: { label: 'Practice 1: 2/0/0/6', counts: [[2, 0], [0, 6]] },
  degenerate: { label: 'A target with no uncertainty: 4/0/4/0', counts: [[4, 0], [4, 0]] },
};
const CELL_NAMES = [['X=0, Y=0', 'X=0, Y=1'], ['X=1, Y=0', 'X=1, Y=1']];

/** I1 — change a count, change the information. */
export function CountInformationLab() {
  const state = useInvestigation(countPresets.base);
  const active = state.active;
  const info = informationFromCounts(active.counts);
  const mosaic = probabilityMosaic(active.counts);
  const draftInfo = (() => {
    try { return informationFromCounts(state.draft.counts); } catch { return null; }
  })();
  const blocked = draftInfo === null
    ? 'These counts have a total of zero, so there are no observed proportions to compute. Raise at least one count.'
    : undefined;
  const calculateInputs = inputs => {
    const answer = informationFromCounts(inputs.counts);
    const outcome = answer.independentExactly
      ? 'none'
      : answer.conditionals.every(row => row.entropy === 0) ? 'all' : 'some';
    return { outcome, value: answer.mutualInformation };
  };
  const edit = (row, column, value) => state.edit({
    counts: state.draft.counts.map((entry, x) => entry.map((old, y) => ((x === row && y === column) ? value : old))),
  });
  const side = 168;
  return <Investigation
    title="Change a count, change the information"
    question="Four observed counts, a binary measurement X and a binary target Y. Change the counts and observe how much target uncertainty knowing X removes."
    note="Everything below is computed from the four counts you entered. Mutual information is in bits, it is symmetric in X and Y, and it is not an accuracy percentage."
    onReset={state.reset}>
    <div className="fs-controls is-tight">
      {[0, 1].map(row => <fieldset key={row} className="fs-count-group">
        <legend>{row === 0 ? 'X = 0' : 'X = 1'}</legend>
        {[0, 1].map(column => <NumberField key={column} label={CELL_NAMES[row][column]}
          value={state.draft.counts[row][column]} min={limits.count.minimum} max={limits.count.maximum}
          step="1" integer onChange={value => edit(row, column, value)} />)}
      </fieldset>)}
    </div>
    <div className="fs-buttons">
      {Object.entries(countPresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => state.load(preset)}>{preset.label}</button>
      ))}
    </div>
    <p className="fs-caption">
      Each count is a whole number from {limits.count.minimum} to {limits.count.maximum.toLocaleString('en-US')}; the total must be positive.
      A zero cell is kept, not treated as missing: it contributes zero by the limiting value, and no logarithm of zero is evaluated.
      The tiles below are one rectangle per cell, so the drawing does not depend on the sample size.
      {draftInfo === null && ' The values in the fields cannot be computed yet; the last valid table is still shown below.'}
    </p>
    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs} blocked={blocked}
      
      describe={`H(Y) = ${round(info.targetEntropy, 9)} bits and H(Y | X) = ${round(info.conditionalEntropy, 9)} bits, so I(X; Y) = ${round(info.mutualInformation, 9)} bits.`} />

    {state.result && <>
      <div className="fs-panels is-pair">
        <div className="fs-panel">
          <h4>The unit square, one rectangle per cell</h4>
          <ScrollRegion><svg viewBox={`0 0 ${side + 82} ${side + 40}`} role="img"
            aria-label={`A unit square split so each tile's area is that cell's share of the ${info.total} observations. ${mosaic.tiles.map(tile => `X equals ${tile.x} and Y equals ${tile.y} has ${info.counts[tile.x][tile.y]} of ${info.total}, an area of ${round(tile.area, 6)}`).join('; ')}.`}>
            {mosaic.tiles.map(tile => (
              <g key={`${tile.x}-${tile.y}`}>
                <rect className={tile.y === 0 ? 'fs-target-0' : 'fs-target-1'}
                  x={4 + side * tile.left} y={16 + side * tile.top}
                  width={Math.max(side * tile.width, 0)} height={Math.max(side * tile.height, 0)}
                  fillOpacity={tile.occupied ? 0.75 : 0} stroke="#0b0f10" strokeWidth="1.5" />
                {tile.area > 0.06 && <text className="is-small" x={4 + side * (tile.left + tile.width / 2)}
                  y={16 + side * (tile.top + tile.height / 2) + 4} textAnchor="middle" style={{ fill: '#10150f' }}>
                  {info.counts[tile.x][tile.y]}
                </text>}
              </g>
            ))}
            <text className="is-small" x="4" y="12">area = count ÷ total</text>
            <text className="is-small" x={4 + side * (mosaic.tiles[0].width / 2)} y={side + 30} textAnchor="middle">X=0</text>
            <text className="is-small" x={4 + side * (info.rowProbabilities[0] + info.rowProbabilities[1] / 2)} y={side + 30} textAnchor="middle">X=1</text>
            <rect className="fs-target-0" x={side + 12} y="22" width="10" height="10" fillOpacity="0.75" />
            <text className="is-small" x={side + 26} y="31">Y=0</text>
            <rect className="fs-target-1" x={side + 12} y="42" width="10" height="10" fillOpacity="0.75" />
            <text className="is-small" x={side + 26} y="51">Y=1</text>
          </svg></ScrollRegion>
          <p>
            Column width is p(X); the split inside a column is p(Y | X). Their product is the cell's own share, so every tile's
            area is exactly count ÷ total. An empty row takes no width and is listed in the table rather than drawn.
          </p>
        </div>
        <div className="fs-panel">
          <h4>Uncertainty before and after, in bits</h4>
          <ScrollRegion><svg viewBox="0 0 300 150" role="img"
            aria-label={`Two bars in bits. Before observing X the entropy of the target is ${round(info.targetEntropy, 6)}. After observing X the weighted conditional entropy is ${round(info.conditionalEntropy, 6)}. The difference, the mutual information, is ${round(info.mutualInformation, 6)} bits.`}>
            {[
              { name: 'H(Y), before X is known', value: info.targetEntropy, y: 30 },
              { name: 'H(Y | X), after X is known', value: info.conditionalEntropy, y: 80 },
            ].map(bar => <g key={bar.name}>
              <text className="is-small" x="6" y={bar.y - 5}>{bar.name}: {round(bar.value, 6)} bits</text>
              <rect className="fs-lane" x="6" y={bar.y} width="288" height="18" rx="2" />
              <rect x="8" y={bar.y + 2} width={Math.max(0, 284 * Math.min(bar.value, 1)).toFixed(1)} height="14" fill="#8eb9a5" />
            </g>)}
            <line className="fs-stem" x1="8" y1="118" x2={8 + Math.max(0, 284 * Math.min(info.mutualInformation, 1))} y2="118" />
            <text className="is-small" x="6" y="136">removed by knowing X: {round(info.mutualInformation, 6)} bits</text>
          </svg></ScrollRegion>
          <p>
            Both bars use one scale of zero to one bit. The green segment underneath is the difference between them, which is
            the mutual information.
          </p>
        </div>
      </div>

      <Table caption={`Each row of X, its conditional distribution and the uncertainty it leaves. Total observations: ${info.total}.`}
        headings={['row', 'count', 'p(X)', 'p(Y=0 | X)', 'p(Y=1 | X)', 'entropy of the row (bits)', 'weighted contribution']}
        rows={info.conditionals.map(row => [
          `X = ${row.index}`, row.count, round(row.weight, 6),
          row.occupied ? round(row.distribution[0], 6) : 'no observations',
          row.occupied ? round(row.distribution[1], 6) : 'no observations',
          row.occupied ? round(row.entropy, 9) : 'contributes 0',
          round(row.weight * row.entropy, 9),
        ])}
        rowClass={index => (info.conditionals[index].occupied ? undefined : 'is-muted')} />

      <p className="fs-readout" aria-live="polite">
        H(Y) = {round(info.targetEntropy, 9)} bits. H(Y | X) = {round(info.conditionalEntropy, 9)} bits.
        I(X; Y) = {round(info.mutualInformation, 9)} bits.
        {info.conditionals[0].occupied && info.conditionals[1].occupied
          && Math.abs(info.conditionals[0].entropy - info.conditionals[1].entropy) <= 1e-12
          && (info.conditionals[0].distribution[0] - 0.5) * (info.conditionals[1].distribution[0] - 0.5) < 0
          ? ' The two rows have entropy equal within 10⁻¹² even though they prefer opposite labels: entropy measures how uneven a distribution is, not which label wins.'
          : ''}
        {info.targetEntropy === 0
          ? ' This target never varies, so there was no uncertainty for X to remove. That is not the same as X being uninformative in another sample.'
          : ''}
        {info.independentExactly && info.targetEntropy > 0
          ? ' The observed cells match what independence would predict exactly, so this table reveals nothing about Y. A zero estimate on a sample is not a certificate of population independence.'
          : ''}
        {' '}Multiplying every count by a common positive integer would leave all of these numbers unchanged, because they depend
        only on the empirical proportions. It would <em>not</em> leave the amount of statistical evidence unchanged: {info.total} observations
        and {info.total * 3} observations support the same estimate very differently.
      </p>

      <Folded summary="The complete trace: counts, joint probabilities, what independence predicts, and each cell's contribution">
        <Table caption="The direct form sums p(x, y) log₂[p(x, y) ÷ (p(x) p(y))] over the occupied cells only"
          headings={['cell', 'count', 'p(x, y)', 'p(x) p(y)', 'ratio', 'contribution (bits)']}
          rows={info.cells.map(cell => [
            `X = ${cell.x}, Y = ${cell.y}`, cell.count, round(cell.joint, 9), round(cell.independent, 9),
            cell.occupied ? round(cell.joint / cell.independent, 9) : 'excluded: p = 0',
            cell.occupied ? signed(cell.contribution, 9) : 'contributes 0',
          ])}
          rowClass={index => (info.cells[index].occupied ? undefined : 'is-muted')} />
        <p className="fs-caption">
          The direct sum gives {round(info.mutualInformationDirect, 12)} bits and the entropy difference gives
          {' '}{round(info.entropyDifference, 12)} bits. {info.agrees
            ? 'They agree to within 10⁻¹², which is the identity this section rests on.'
            : 'They disagree beyond the numerical tolerance; this calls for a numerical check, not a different probability identity.'}
          {' '}Near independence, subtracting entropies or summing signed contributions can lose a tiny positive result. The main readout uses an equivalent nonnegative KL remainder, giving {round(info.mutualInformation, 12)} bits. An individual occupied cell can contribute a negative number; the weighted total is a divergence and cannot be negative.
        </p>
      </Folded>
    </>}
  </Investigation>;
}

/* ============================================================ §3 · I2 */

const subsetPresets = {
  xor: { label: 'Baseline: XOR labels 0, 1, 1, 0', labels: [0, 1, 1, 0], policy: 'strict', reversed: false },
  firstFeature: { label: 'Contrast: Y = A, labels 0, 0, 1, 1', labels: [0, 0, 1, 1], policy: 'strict', reversed: false },
  constant: { label: 'Null: every label 0', labels: [0, 0, 0, 0], policy: 'strict', reversed: false },
  forced: { label: 'Policy change: XOR with two forced additions', labels: [0, 1, 1, 0], policy: 'forceTwo', reversed: false },
  reordered: { label: 'Null: the same world displayed bottom-up', labels: [0, 1, 1, 0], policy: 'strict', reversed: true },
};
const STATE_NAMES = ['A=0, B=0', 'A=0, B=1', 'A=1, B=0', 'A=1, B=1'];

/** I2 — traverse the subset lattice. */
export function SubsetSearchLab() {
  const state = useInvestigation(subsetPresets.xor);
  const active = state.active;
  const search = forwardSelection(active.labels, active.policy);
  const world = search.world;
  const draftWorld = forwardSelection(state.draft.labels, state.draft.policy).world;
  const displayOrder = active.reversed ? [3, 2, 1, 0] : [0, 1, 2, 3];
  const calculateInputs = inputs => {
    const answer = forwardSelection(inputs.labels, inputs.policy);
    return {
      outcome: answer.finalMask === 3 ? 'pair' : answer.finalMask === 0 ? 'empty' : 'single',
      value: answer.finalScore,
    };
  };
  const place = mask => ({ 0: 170, 1: 96, 2: 244, 3: 170 }[mask]);
  const lift = mask => (mask === 0 ? 30 : mask === 3 ? 170 : 100);
  return <Investigation
    title="Traverse the subset lattice"
    question="Four fixed input states form a complete declared world. Edit the binary target label attached to each state, choose the search policy, and watch whether the search reaches both inputs."
    note="Every score here is exact over all four states, because every state is known. This is a finite lookup rule, not a held-out estimate, and reusing a label is part of the defined world rather than evidence of generalization."
    onReset={state.reset}>
    <div className="fs-controls">
      {STATE_NAMES.map((name, index) => (
        <NumberField key={name} label={`Target for ${name}`} value={state.draft.labels[index]}
          min={limits.label.minimum} max={limits.label.maximum} step="1" integer
          onChange={value => state.edit({ labels: state.draft.labels.map((old, position) => (position === index ? value : old)) })} />
      ))}
      <Field label="Search policy">
        <select value={state.draft.policy} onChange={event => state.edit({ policy: event.target.value })}>
          <option value="strict">add only on a strictly higher score</option>
          <option value="forceTwo">force two additions</option>
        </select>
      </Field>
      <Field label="Display order of the states">
        <select value={state.draft.reversed ? 'reversed' : 'forward'}
          onChange={event => state.edit({ reversed: event.target.value === 'reversed' })}>
          <option value="forward">A=0,B=0 first</option>
          <option value="reversed">A=1,B=1 first</option>
        </select>
      </Field>
    </div>
    <div className="fs-buttons">
      {Object.entries(subsetPresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => state.load(preset)}>{preset.label}</button>
      ))}
    </div>
    <p className="fs-caption">
      Each target label is 0 or 1. The four input states are fixed: they are the whole declared world, so they are not editable.
      Changing a label changes the data and every score must be recomputed; changing the display order or the policy does not
      change the data at all. The empty starting subset is
      {' '}{maskLabel(0)} with accuracy {round(draftWorld.subsets[0].accuracy, 6)}, and the two candidates it can add are A and B.
    </p>
    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      describe={`The search finished at ${maskLabel(search.finalMask)} with accuracy ${round(search.finalScore, 6)}. It ${search.stopReason}.`} />

    {state.result && <>
      <div className="fs-stage">
        <h4>The subset lattice, with every exact score</h4>
        <ScrollRegion><svg viewBox="0 0 340 214" role="img"
          aria-label={`A lattice of four subsets. ${world.subsets.map(entry => `${entry.label} scores ${round(entry.accuracy, 6)}`).join('; ')}. The accepted path visits ${search.path.map(maskLabel).join(' then ')}, and the search ${search.stopReason}.`}>
          {[[0, 1], [0, 2], [1, 3], [2, 3]].map(([from, to]) => {
            const taken = search.path.includes(from) && search.path.includes(to)
              && search.path.indexOf(to) === search.path.indexOf(from) + 1;
            return <path key={`${from}-${to}`} className={taken ? 'fs-flow' : 'fs-flow is-alt'}
              d={`M${place(from)},${lift(from) + 34} L${place(to)},${lift(to)}`} />;
          })}
          {world.subsets.map(entry => (
            <g key={entry.mask}>
              <rect className={`fs-lane${search.finalMask === entry.mask ? ' is-refit' : ''}`}
                x={place(entry.mask) - 52} y={lift(entry.mask)} width="104" height="34" rx="3" />
              <text x={place(entry.mask)} y={lift(entry.mask) + 14} textAnchor="middle">{entry.label}</text>
              <text className="is-small" x={place(entry.mask)} y={lift(entry.mask) + 27} textAnchor="middle">
                accuracy {round(entry.accuracy, 4)} · {entry.correct}/4
              </text>
            </g>
          ))}
        </svg></ScrollRegion>
        <p className="fs-caption">
          A solid edge was accepted; a dashed edge was either evaluated and rejected, or never reached at all. The gold box is
          where the search finished.
        </p>
      </div>

      <Table caption="What the search actually evaluated at each step, separately from the path it accepted"
        headings={['step', 'from', 'its score', 'candidate', 'candidate score', 'strictly higher?', 'accepted?']}
        rows={search.steps.flatMap(step => step.candidates.map(candidate => [
          step.step, maskLabel(step.from), round(step.fromScore, 6), candidate.feature, round(candidate.score, 6),
          candidate.score > step.fromScore + 1e-12 ? 'yes' : 'no',
          step.accepted && candidate.mask === step.best.mask ? 'accepted' : 'not taken',
        ]))} />

      <p className="fs-readout" aria-live="polite">
        The accepted path is {search.path.map(maskLabel).join(' → ')}, finishing at accuracy {round(search.finalScore, 6)}.
        The search {search.stopReason}.
        {search.policy === 'strict' && !search.reachedPair && world.subsets[3].accuracy > search.finalScore
          ? ` The pair ${maskLabel(3)} scores ${round(world.subsets[3].accuracy, 6)}, which this rule never evaluated, because neither single addition improved on the empty subset. The stopping rule is part of the algorithm, not an administrative detail.`
          : ''}
        {search.policy === 'forceTwo'
          ? ' Forcing two additions reaches the pair, at the cost of fitting a larger subset whether or not the first addition helped. That is a different algorithm with different assumptions, not a universally better one.'
          : ''}
        {active.reversed
          ? ' The states are displayed bottom-up here. Every score above is identical to the ascending display: reordering a complete world changes no computation.'
          : ''}
      </p>

      <Folded summary="Every subset: how its lookup predictor groups the states, and what it predicts">
        {world.subsets.map(entry => (
          <Table key={entry.mask}
            caption={`${entry.label} retains ${entry.features.length === 0 ? 'nothing' : entry.features.join(' and ')}; ties predict 0. Accuracy ${round(entry.accuracy, 6)}.`}
            headings={['group', 'states in it', 'targets 0 / 1', 'majority (tie → 0)', 'predicts']}
            rows={entry.groups.map(group => [
              group.values.length === 0 ? 'all four states' : entry.features.map((name, index) => `${name}=${group.values[index]}`).join(', '),
              group.stateIds.map(id => STATE_NAMES[id]).join(' · '),
              `${group.counts[0]} / ${group.counts[1]}`,
              group.tie ? 'tie' : `label ${group.majority}`,
              group.majority,
            ])} />
        ))}
      </Folded>

      <Table caption={`The declared world${active.reversed ? ', displayed bottom-up' : ''}, and what each subset's predictor says about each state`}
        headings={['state', 'target', ...world.subsets.map(entry => entry.label)]}
        rows={displayOrder.map(id => [
          STATE_NAMES[id], world.targets[id],
          ...world.subsets.map(entry => `${entry.predictions[id]}${entry.predictions[id] === world.targets[id] ? '' : ' ✗'}`),
        ])} />
    </>}
  </Investigation>;
}

/* ============================================================ §4 · I3 */

const permutationPresets = {
  firstOnly: {
    label: 'Baseline: the first-sensor model',
    rows: [[-1, -1], [-1, -1], [1, 1], [1, 1]], target: [-1, -1, 1, 1],
    coefficients: [1, 0], donor: [2, 3, 0, 1], mode: 'first',
  },
  secondOnly: {
    label: 'Contrast: the second-sensor model',
    rows: [[-1, -1], [-1, -1], [1, 1], [1, 1]], target: [-1, -1, 1, 1],
    coefficients: [0, 1], donor: [2, 3, 0, 1], mode: 'first',
  },
  average: {
    label: 'Contrast: the average-of-sensors model',
    rows: [[-1, -1], [-1, -1], [1, 1], [1, 1]], target: [-1, -1, 1, 1],
    coefficients: [0.5, 0.5], donor: [2, 3, 0, 1], mode: 'first',
  },
  identity: {
    label: 'Null: the identity donor map',
    rows: [[-1, -1], [-1, -1], [1, 1], [1, 1]], target: [-1, -1, 1, 1],
    coefficients: [0.5, 0.5], donor: [0, 1, 2, 3], mode: 'group',
  },
  zeroCoefficient: {
    label: 'Null: shuffle a column the model never reads',
    rows: [[-1, -1], [-1, -1], [1, 1], [1, 1]], target: [-1, -1, 1, 1],
    coefficients: [1, 0], donor: [2, 3, 0, 1], mode: 'second',
  },
  practice: {
    label: 'Practice 3: average model, donor 0, 2, 1, 3',
    rows: [[-1, -1], [-1, -1], [1, 1], [1, 1]], target: [-1, -1, 1, 1],
    coefficients: [0.5, 0.5], donor: [0, 2, 1, 3], mode: 'first',
  },
};
const MODE_COLUMNS = { first: [0], second: [1], group: [0, 1] };
const MODE_NAMES = { first: 'the first sensor only', second: 'the second sensor only', group: 'both sensors, as one group' };

/** I3 — follow the donor rows. */
export function DonorPermutationLab() {
  const state = useInvestigation(permutationPresets.firstOnly);
  const active = state.active;
  const run = permutationExperiment({
    rows: active.rows, target: active.target, coefficients: active.coefficients,
    donor: active.donor, columns: MODE_COLUMNS[active.mode],
  });
  const duplicated = new Set(state.draft.donor.filter((value, index) =>
    state.draft.donor.indexOf(value) !== index));
  const donorProblem = duplicated.size > 0
    ? 'Each source row 0, 1, 2 and 3 must appear exactly once, so this is not yet a shuffle.' : null;
  const suite = permutationSuite({
    rows: active.rows, target: active.target, donor: active.donor,
    models: [
      { name: 'first sensor only', coefficients: [1, 0] },
      { name: 'second sensor only', coefficients: [0, 1] },
      { name: 'average of sensors', coefficients: [0.5, 0.5] },
    ],
  });
  const calculateInputs = inputs => {
    const answer = permutationExperiment({
      rows: inputs.rows, target: inputs.target, coefficients: inputs.coefficients,
      donor: inputs.donor, columns: MODE_COLUMNS[inputs.mode],
    });
    return {
      outcome: Math.abs(answer.increase) <= 8 * Number.EPSILON * Math.max(answer.baseMse, answer.alteredMse) ? 'same' : answer.increase > 0 ? 'higher' : 'lower',
      value: answer.increase,
    };
  };
  const editRow = (index, column, value) => state.edit({
    rows: state.draft.rows.map((row, position) => (position === index
      ? row.map((old, spot) => (spot === column ? value : old)) : row)),
  });
  return <Investigation
    title="Follow the donor rows"
    question="Four assessed rows, two sensor columns, and a FIXED linear prediction w₁x₁ + w₂x₂. Choose which column or group to shuffle, watch the donor rows arrive and compare the squared-error loss."
    note="No fitting happens here and there is no fallback to another column. The coefficients are read, never updated: a model that never reads the second sensor cannot start reading it once the first is corrupted."
    onReset={state.reset}>
    <div className="fs-controls is-wide">
      {[0, 1, 2, 3].map(index => <fieldset key={index} className="fs-count-group">
        <legend>row {index}</legend>
        <NumberField label={`row ${index} sensor 1`} value={state.draft.rows[index][0]}
          min={limits.sensor.minimum} max={limits.sensor.maximum} step="0.5" decimals={4}
          onChange={value => editRow(index, 0, value)} />
        <NumberField label={`row ${index} sensor 2`} value={state.draft.rows[index][1]}
          min={limits.sensor.minimum} max={limits.sensor.maximum} step="0.5" decimals={4}
          onChange={value => editRow(index, 1, value)} />
        <NumberField label={`row ${index} target`} value={state.draft.target[index]}
          min={limits.target.minimum} max={limits.target.maximum} step="0.5" decimals={4}
          onChange={value => state.edit({ target: state.draft.target.map((old, position) => (position === index ? value : old)) })} />
        <Field label={`row ${index} donor`} error={duplicated.has(state.draft.donor[index]) ? donorProblem : undefined}>
          <select value={state.draft.donor[index]}
            onChange={event => state.edit({ donor: state.draft.donor.map((old, position) => (position === index ? Number(event.target.value) : old)) })}>
            {[0, 1, 2, 3].map(option => <option key={option} value={option}>from row {option}</option>)}
          </select>
        </Field>
      </fieldset>)}
    </div>
    <div className="fs-controls">
      <NumberField label="Fixed coefficient w₁" value={state.draft.coefficients[0]}
        min={limits.coefficient.minimum} max={limits.coefficient.maximum} step="0.5" decimals={4}
        onChange={value => state.edit({ coefficients: [value, state.draft.coefficients[1]] })} />
      <NumberField label="Fixed coefficient w₂" value={state.draft.coefficients[1]}
        min={limits.coefficient.minimum} max={limits.coefficient.maximum} step="0.5" decimals={4}
        onChange={value => state.edit({ coefficients: [state.draft.coefficients[0], value] })} />
      <Field label="What to permute">
        <select value={state.draft.mode} onChange={event => state.edit({ mode: event.target.value })}>
          <option value="first">the first sensor</option>
          <option value="second">the second sensor</option>
          <option value="group">both sensors as one group</option>
        </select>
      </Field>
    </div>
    <div className="fs-buttons">
      {Object.entries(permutationPresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => state.load(preset)}>{preset.label}</button>
      ))}
    </div>
    <p className="fs-caption">
      Sensor readings and targets run from {limits.sensor.minimum} to {limits.sensor.maximum}; coefficients from
      {' '}{limits.coefficient.minimum} to {limits.coefficient.maximum}. The donor ordering must be a permutation of the four rows,
      so every source row is used exactly once; fixed points are allowed, and they are why a shuffle need not change every case.
      A grouped shuffle applies the <strong>same</strong> donor map to both columns, which keeps their within-row pairing while
      disturbing their relation to the target.
      {donorProblem ? ` ${donorProblem} The last valid ordering is still shown below.` : ''}
    </p>
    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      blocked={donorProblem ? `${donorProblem} Choose a different source row for the repeated entries; the last valid ordering is still shown.` : undefined}
      
      describe={`The original MSE is ${round(run.baseMse, 9)} and the shuffled MSE is ${round(run.alteredMse, 9)}, a change of ${signed(run.increase, 9)} in squared target units.`} />

    {state.result && <>
      <div className="fs-stage">
        <h4>Where each altered value came from</h4>
        <ScrollRegion><svg viewBox="0 0 340 186" role="img"
          aria-label={`Donor arrows. ${run.perRow.map(row => `Assessed row ${row.index} takes ${run.columns.map(column => `sensor ${column + 1}`).join(' and ')} from source row ${row.donor}`).join('; ')}. ${run.fixedPoints.length > 0 ? `Rows ${run.fixedPoints.join(', ')} are fixed points and keep their own values.` : 'No row is a fixed point.'}`}>
          <text className="is-small" x="6" y="16">source rows</text>
          <text className="is-small" x="334" y="16" textAnchor="end">assessed rows, after the shuffle</text>
          {run.perRow.map(row => {
            const y = 30 + row.index * 38;
            const from = 30 + row.donor * 38;
            const fixed_ = row.donor === row.index;
            return <g key={row.index}>
              <rect className="fs-lane" x="6" y={y} width="118" height="28" rx="3" />
              <text className="is-small" x="12" y={y + 18}>row {row.index}: {row.original.map(value => round(value, 3)).join(', ')}</text>
              <path className={fixed_ ? 'fs-flow is-alt' : 'fs-flow'} d={`M126,${from + 14} C168,${from + 14} 174,${y + 14} 214,${y + 14}`} />
              <rect className={`fs-lane${row.changed.some(Boolean) ? ' is-refit' : ''}`} x="216" y={y} width="118" height="28" rx="3" />
              <text className="is-small" x="222" y={y + 18}>row {row.index}: {row.hybrid.map(value => round(value, 3)).join(', ')}</text>
            </g>;
          })}
        </svg></ScrollRegion>
        <p className="fs-caption">
          A solid arrow moves values between different rows; a dashed arrow is a fixed point, where a row donates to itself and
          nothing changes. The arrows are optional: the complete trace is in the table below.
        </p>
      </div>

      <Table caption="Every row: its donor, the values that actually changed, the fixed model's prediction and its squared error"
        headings={['row', 'donor', 'original sensors', 'after the shuffle', 'target', 'original prediction', 'shuffled prediction', 'original squared error', 'shuffled squared error']}
        rows={run.perRow.map(row => [
          row.index, `row ${row.donor}${row.donor === row.index ? ' (itself)' : ''}`,
          row.original.map(value => round(value, 4)).join(', '),
          row.hybrid.map((value, column) => (row.changed[column] ? `${round(value, 4)} ←` : round(value, 4))).join(', '),
          round(row.target, 4), round(row.basePrediction, 6), round(row.alteredPrediction, 6),
          round(row.baseSquared, 6), round(row.alteredSquared, 6),
        ])}
        rowClass={index => (run.perRow[index].changed.some(Boolean) ? 'is-selected' : 'is-muted')} />

      <p className="fs-readout" aria-live="polite">
        Original MSE {round(run.baseMse, 9)}; shuffled MSE {round(run.alteredMse, 9)}; the difference is
        {' '}<strong>{signed(run.increase, 9)}</strong> in squared target units. The output is shuffled minus original, so a negative
        result is a valid outcome and is shown rather than clipped.
        {run.fixedPoints.length > 0
          ? ` Rows ${run.fixedPoints.join(', ')} donated to themselves, so a valid shuffle left them exactly as they were.`
          : ''}
        {active.coefficients[MODE_COLUMNS[active.mode][0]] === 0 && MODE_COLUMNS[active.mode].length === 1
          ? ' This column has coefficient zero, so the prediction never reads it: shuffling it is an exact null for this fixed model. That is a fact about this equation, not about the sensor in the world.'
          : ''}
        {run.unchangedRows.length === 4
          ? ' Every altered cell happened to receive its own value back, so this shuffle changed nothing at all.'
          : ''}
      </p>

      <Table caption="The same donor ordering against three different fixed predictors. These are three separate experiments on three equations, not three answers to one question."
        headings={['fixed predictor', 'coefficients', 'shuffle sensor 1', 'shuffle sensor 2', 'shuffle both as a group']}
        rows={suite.map(model => [
          model.name, `(${model.coefficients.map(value => round(value, 4)).join(', ')})`,
          ...model.results.map(result => signed(result.increase, 6)),
        ])} />
      <p className="fs-caption">
        A predictor that never reads the second sensor cannot start reading it after the first is corrupted; its equation has no
        such branch. Removing a sensor <strong>and refitting</strong> would be a fourth experiment, and a newly trained predictor
        could then use the survivor alone. None of these rows measures importance in the world.
      </p>
    </>}
  </Investigation>;
}

/* ============================================================ §5 · I4 */

const coalitionPresets = {
  zero: { label: 'Baseline: reference (0, 0)', instance: [2, 3], gamma: 1, reference: [[0, 0]], observed: 11 },
  twoRow: { label: 'Contrast: reference (0, 0) and (1, 1)', instance: [2, 3], gamma: 1, reference: [[0, 0], [1, 1]], observed: 11 },
  noInteraction: { label: 'Null for order: γ = 0', instance: [2, 3], gamma: 0, reference: [[0, 0]], observed: 11 },
  practice: { label: 'Practice 4: x = (1, 2), γ = 2, reference (0, 0)', instance: [1, 2], gamma: 2, reference: [[0, 0]], observed: 7 },
  targetNull: { label: 'Null: change only the observed target', instance: [2, 3], gamma: 1, reference: [[0, 0]], observed: 4 },
  atReference: { label: 'The instance equals its only reference row', instance: [0, 0], gamma: 1, reference: [[0, 0]], observed: 0 },
};

/** I4 — rebuild a coalition from its donor inputs. */
export function CoalitionReferenceLab() {
  const [inspected, setInspected] = useState(3);
  const state = useInvestigation(coalitionPresets.zero);
  const active = state.active;
  const explanation = explainPolynomial({
    instance: active.instance, background: active.reference, gamma: active.gamma,
  });
  const dependent = dependentGames();
  const chart = waterfall(explanation.baseline, explanation.phi);
  const calculateInputs = inputs => {
    const answer = explainPolynomial({ instance: inputs.instance, background: inputs.reference, gamma: inputs.gamma });
    const [first, second] = answer.phi;
    return {
      outcome: Math.abs(first - second) <= 1e-12 ? 'equal' : first > second ? 'a' : 'b',
      value: answer.baseline,
    };
  };
  const setReferenceCount = count => state.edit({
    reference: Array.from({ length: count }, (_, index) => state.draft.reference[index] ?? [0, 0]),
  });
  const editReference = (index, column, value) => state.edit({
    reference: state.draft.reference.map((row, position) => (position === index
      ? row.map((old, spot) => (spot === column ? value : old)) : row)),
  });
  const maskName = mask => (mask === 0 ? 'none' : mask === 1 ? 'A' : mask === 2 ? 'B' : 'A and B');
  return <Investigation
    title="Rebuild a coalition from its donor inputs"
    question="A model f(a, b) = a + b + γab, one explained instance, and a declared reference of one to six rows. Record which feature you expect to receive the larger contribution, then watch every hybrid input that produces the coalition values."
    note="A coalition value is the average model output over actual hybrid rows: retain the instance values in S and fill every other coordinate from the same donor row. The reference is part of the question being asked, not a speed trick."
    onReset={() => { state.reset(); setInspected(3); }}>
    <div className="fs-controls">
      <NumberField label="Explained a" value={state.draft.instance[0]}
        min={limits.coordinate.minimum} max={limits.coordinate.maximum} step="0.5" decimals={4}
        onChange={value => state.edit({ instance: [value, state.draft.instance[1]] })} />
      <NumberField label="Explained b" value={state.draft.instance[1]}
        min={limits.coordinate.minimum} max={limits.coordinate.maximum} step="0.5" decimals={4}
        onChange={value => state.edit({ instance: [state.draft.instance[0], value] })} />
      <NumberField label="Interaction γ" value={state.draft.gamma}
        min={limits.gamma.minimum} max={limits.gamma.maximum} step="0.5" decimals={4}
        onChange={gamma => state.edit({ gamma })} />
      <Field label="Reference rows">
        <select value={state.draft.reference.length} onChange={event => setReferenceCount(Number(event.target.value))}>
          {[1, 2, 3, 4, 5, 6].map(count => <option key={count} value={count}>{count} row{count === 1 ? '' : 's'}, equally weighted</option>)}
        </select>
      </Field>
      <NumberField label="Observed target (never used by this game)" value={state.draft.observed}
        min={limits.target.minimum * 2} max={limits.target.maximum * 2} step="0.5" decimals={4}
        onChange={observed => state.edit({ observed })} />
    </div>
    <div className="fs-controls is-tight">
      {state.draft.reference.map((row, index) => <fieldset key={index} className="fs-count-group">
        <legend>reference row {index}</legend>
        <NumberField label={`reference ${index} a`} value={row[0]}
          min={limits.coordinate.minimum} max={limits.coordinate.maximum} step="0.5" decimals={4}
          onChange={value => editReference(index, 0, value)} />
        <NumberField label={`reference ${index} b`} value={row[1]}
          min={limits.coordinate.minimum} max={limits.coordinate.maximum} step="0.5" decimals={4}
          onChange={value => editReference(index, 1, value)} />
      </fieldset>)}
    </div>
    <div className="fs-buttons">
      {Object.entries(coalitionPresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => state.load(preset)}>{preset.label}</button>
      ))}
    </div>
    <p className="fs-caption">
      Coordinates run from {limits.coordinate.minimum} to {limits.coordinate.maximum} and γ from {limits.gamma.minimum} to
      {' '}{limits.gamma.maximum}; one to six equally weighted reference rows are allowed. These bounds constrain this lesson's
      controls only. Contributions within 10⁻¹² are treated as tied in the numerical comparison. The observed-target field is deliberately excluded from this game: it explains a model <em>output</em>, so
      changing a label cannot move any coalition value.
    </p>
    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      describe={`φ_A = ${round(explanation.phi[0], 9)} and φ_B = ${round(explanation.phi[1], 9)}, above a baseline of ${round(explanation.baseline, 9)}. Their difference is ${round(explanation.phi[0] - explanation.phi[1], 12)}; the tie tolerance is 10⁻¹².`} />

    {state.result && <>
      <div className="fs-stage">
        <h4>1. Every coalition, built from its own hybrid rows</h4>
        <div className="fs-controls">
          <Field label="Inspect one coalition's hybrid rows">
            <select value={inspected} onChange={event => setInspected(Number(event.target.value))}>
              {[0, 1, 2, 3].map(mask => <option key={mask} value={mask}>retain {maskName(mask)}</option>)}
            </select>
          </Field>
        </div>
        <Table caption={`Retaining ${maskName(inspected)}: each reference row with the retained coordinates overwritten, and what the model outputs for it`}
          headings={['reference row', 'hybrid input', 'f(hybrid)']}
          rows={explanation.masks[inspected].rows.map((row, index) => [
            `row ${index}: (${active.reference[index].map(value => round(value, 4)).join(', ')})`,
            `(${row.map(value => round(value, 4)).join(', ')})`,
            round(explanation.masks[inspected].outputs[index], 9),
          ])} />
        <p className="fs-caption">
          v(retain {maskName(inspected)}) is the mean of that last column: {round(explanation.values[inspected], 9)}.
          Missing coordinates come from the <strong>same</strong> donor row, which preserves their relation to each other while
          breaking their relation to the retained values.
        </p>
      </div>

      <Table caption="The complete game, in bitmask order: none, A, B, then A and B"
        headings={['coalition', 'v(S)', 'increment when A joins', 'increment when B joins']}
        rows={[0, 1, 2, 3].map(mask => [
          `retain ${maskName(mask)}`, round(explanation.values[mask], 9),
          (mask & 1) === 0 ? signed(explanation.values[mask | 1] - explanation.values[mask], 9) : '—',
          (mask & 2) === 0 ? signed(explanation.values[mask | 2] - explanation.values[mask], 9) : '—',
        ])} />

      <div className="fs-panels is-pair">
        <div className="fs-panel">
          <h4>2. Both arrival orders, then their average</h4>
          <Table caption="Each order's actual increments; averaging matching increments gives the allocation"
            headings={['arrival order', 'first increment', 'second increment', 'total']}
            rows={explanation.paths.paths.map(path => [
              path.order.map(player => (player === 0 ? 'A' : 'B')).join(' then '),
              signed(path.steps[0].increment, 9), signed(path.steps[1].increment, 9), round(path.total, 9),
            ])} />
          <p>
            {explanation.orderIndependent
              ? (active.gamma === 0 ? 'With no interaction term, each feature adds the same amount whenever it arrives. ' : 'For these particular coordinates and reference rows, the interaction changes neither order’s increments beyond 10⁻¹². ') + 'The orders agree within the comparison tolerance; that does not mean the contributions themselves are zero.'
              : 'The two orders give different increments, because the interaction is only available once both features are present. Averaging over every order splits it between them.'}
            {' '}The averaged increments and the weighted formula agree to within 10⁻¹²: {explanation.paths.agrees ? 'they do here.' : 'they do not, which would be a defect.'}
          </p>
        </div>
        <div className="fs-panel">
          <h4>3. The allocation, as a waterfall</h4>
          <Waterfall model={chart} names={['φ_A', 'φ_B']} baselineLabel="baseline v(∅)" totalLabel="reconstructed"
            unit="output units" digits={6}
            describe={`A waterfall starting at the baseline ${round(chart.baseline, 6)}, adding ${round(chart.segments[0].value, 6)} for feature A and ${round(chart.segments[1].value, 6)} for feature B, ending at ${round(chart.reconstruction, 6)}.`} />
          <p>
            The closing bar is the sum, not a placed rectangle: v(∅) + φ_A + φ_B = {round(settled(chart.reconstruction), 9)} against
            the actual f(x) = {round(explanation.prediction, 9)}. Nothing here says that φ_A is the effect of physically changing A.
          </p>
        </div>
      </div>

      <p className="fs-readout" aria-live="polite">
        Baseline v(∅) = {round(explanation.baseline, 9)}; φ_A = {round(explanation.phi[0], 9)}; φ_B = {round(explanation.phi[1], 9)};
        f(x) = {round(explanation.prediction, 9)}, reconstructed to within {round(Math.abs(explanation.efficiencyError), 12)}.
        The mean model output over this reference is {round(explanation.baseline, 9)}, while the model evaluated at the mean
        reference input ({explanation.meanReferenceInput.map(value => round(value, 4)).join(', ')}) is
        {' '}{round(explanation.outputOfMeanInput, 9)}. These quantities can differ for a nonlinear model, but may coincide for a particular reference distribution.
        {Math.abs(explanation.phi[0]) <= 1e-12 && Math.abs(explanation.phi[1]) <= 1e-12
          ? (active.reference.every(row => row.every((value, index) => value === active.instance[index])) ? ' This instance matches every reference row, so both contributions are zero.' : ' Both contributions are within 10⁻¹² of zero for this game; cancelling increments can produce this even when the instance differs from the reference rows.')
          : ''}
        {' '}Changing the observed target to {round(active.observed, 4)} moved none of these numbers, because this game explains a
        model output and never reads a label.
      </p>

      <Folded summary="A separate comparison: two missing-feature rules on one dependent pair">
        <p className="fs-caption">
          Let X₁ = X₂ be a fair binary variable and let f(x) = x₁. Explain x = (1, 1). The two rules answer different questions,
          and neither is a claim about altering a real instrument.
        </p>
        <Table caption="Conditional expectation against background replacement, on the same model and instance"
          headings={['coalition', 'conditional v(S)', 'replacement v(S)']}
          rows={[0, 1, 2, 3].map(mask => [
            `retain ${maskName(mask)}`,
            round(dependent.conditional.values[mask], 6), round(dependent.replacement.values[mask], 6),
          ])} />
        <Table caption="The allocations they produce"
          headings={['feature', 'conditional φ', 'replacement φ']}
          rows={[0, 1].map(index => [
            `X${index + 1}`, round(dependent.conditional.phi[index], 6), round(dependent.replacement.phi[index], 6),
          ])} />
        <p className="fs-caption">
          Under conditioning, learning either coordinate reveals both, so each receives a quarter. Under replacement, learning X₂
          while replacing X₁ still averages to one half, so X₂ receives zero. There is no violation of the dummy-player rule: X₂
          changes coalition values in the conditional game and does not change them in the replacement game. The replacement game
          also evaluates hybrid rows such as {dependent.impossibleHybrids.length > 0
            ? `(${dependent.impossibleHybrids[0].row.join(', ')})`
            : 'none'}, which the stated dependence makes impossible to observe; that is a property of the rule, not a mistake in it.
        </p>
      </Folded>
    </>}
  </Investigation>;
}

/* ============================================================ §6 · I5 */

const REFERENCES = {
  fitting: { key: 'fitting', label: 'all 100 fitting rows, equally weighted', rows: backgroundRows, ids: backgroundIds },
  cohort: { key: 'cohort', label: 'the twelve highest-index fitting rows (a class-3 cohort)', rows: alternativeReference.rows, ids: alternativeReference.ids },
};
const winePresets = {
  recorded: { label: 'Recorded: source row 104', values: explainedCases[0].input.slice(), source: 0, reference: 'fitting' },
  alcohol: { label: 'Contrast: alcohol 12.51 → 13.5', values: changedInference[0].input.slice(), source: 0, reference: 'fitting' },
  malic: { label: 'Null: malic acid 1.73 → 4.1', values: changedInference[1].input.slice(), source: 0, reference: 'fitting' },
  cohort: { label: 'Reference contrast: the class-3 cohort', values: explainedCases[0].input.slice(), source: 0, reference: 'cohort' },
};

/** I5 — a changed sample takes a different tree path. */
export function WineInferenceLab() {
  const [inspectedMask, setInspectedMask] = useState(15);
  const state = useInvestigation(winePresets.recorded);
  const active = state.active;
  const reference = REFERENCES[active.reference];
  const explanation = explainTreeInstance({
    tree: TREE, instance: active.values, background: reference.rows,
    classIndex: fourFieldModel.classIndex, keepHybrids: true,
  });
  const chart = waterfall(explanation.baseline, explanation.phi);
  const flags = extrapolationFlags(active.values, fourFieldModel.fitRanges);
  const outside = flags.filter(flag => flag.outside);
  const sourceRow = explainedCases[active.source];
  const calculateInputs = inputs => {
    const answer = explainTreeInstance({
      tree: TREE, instance: inputs.values, background: REFERENCES[inputs.reference].rows,
      classIndex: fourFieldModel.classIndex,
    });
    return {
      outcome: answer.prediction === 0 ? 'zero' : answer.prediction === 1 ? 'one' : 'between',
      value: answer.prediction,
    };
  };
  const inspected = explanation.masks[inspectedMask];
  const tally = explanation.leafTally[inspectedMask];
  const retained = SMALL_LABELS.filter((_, index) => (inspectedMask & (1 << index)) !== 0);
  return <Investigation
    title="A changed sample takes a different tree path"
    question="The saved four-field tree, one explained sample and a declared reference population. Edit the four raw measurements or change the reference, record what the class-1 probability will be, then follow the actual split comparisons and the hybrid rows behind every coalition."
    note="The tree and the reference are fixed. Editing a measurement is a model-input scenario, not retraining: no split, threshold or leaf changes, and no new data are collected."
    onReset={state.reset}>
    <div className="fs-controls is-wide">
      {SMALL_LABELS.map((label, index) => (
        <NumberField key={label} label={`${label} (raw source scale)`} value={state.draft.values[index]}
          min={fourFieldModel.limits[index].minimum} max={fourFieldModel.limits[index].maximum}
          step={String(fourFieldModel.limits[index].step)} decimals={fourFieldModel.limits[index].decimals}
          onChange={value => state.edit({ values: state.draft.values.map((old, position) => (position === index ? value : old)) })} />
      ))}
      <Field label="Explained sample">
        <select value={state.draft.source} onChange={event => {
          const source = Number(event.target.value);
          state.load({ ...state.draft, source, values: explainedCases[source].input.slice() });
        }}>
          {explainedCases.map((entry, index) => (
            <option key={entry.sourceId} value={index}>source row {entry.sourceId} (actual cultivar {entry.actualClass})</option>
          ))}
        </select>
      </Field>
      <Field label="Reference population">
        <select value={state.draft.reference} onChange={event => state.edit({ reference: event.target.value })}>
          {Object.values(REFERENCES).map(entry => <option key={entry.key} value={entry.key}>{entry.label}</option>)}
        </select>
      </Field>
    </div>
    <div className="fs-buttons">
      {Object.entries(winePresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => state.load(preset)}>{preset.label}</button>
      ))}
    </div>
    <p className="fs-caption">
      Control bounds, wider than every observed source value:
      {' '}{SMALL_LABELS.map((label, index) => `${label} ${fourFieldModel.limits[index].minimum} to ${fourFieldModel.limits[index].maximum}`).join('; ')}.
      The provider's documentation does not establish physical units for these fields, so they are labelled by their raw source
      scale. An edit outside the observed fitting range can still be inspected; it is marked as extrapolation rather than
      presented as a measured specimen.
    </p>
    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      describe={`The saved tree sends this sample to leaf ${explanation.decision.leaf}, whose class-1 probability is ${round(explanation.prediction, 9)}. The baseline over this reference is ${round(explanation.baseline, 9)}.`} />

    {state.result && <>
      <div className="fs-stage">
        <h4>1. The actual route through the saved tree</h4>
        <TreeDiagram tree={TREE} shortLabels={SHORT_LABELS} controls={fourFieldModel.limits}
          decision={explanation.decision} classIndex={fourFieldModel.classIndex}
          describe={`The saved nine-node tree. This sample takes ${explanation.decision.path.map(step => `${SMALL_LABELS[step.field]} ${step.goesLeft ? 'at or below' : 'above'} ${step.threshold}`).join(', then ')}, reaching leaf ${explanation.decision.leaf} with class-1 probability ${round(explanation.prediction, 6)} over ${TREE.samples[explanation.decision.leaf]} fitted rows.`} />
        <p className="fs-caption">
          Gold boxes are the route this sample takes. A drawn threshold is the largest value these controls can hold that still
          takes the left branch, so the rule in the diagram and the rule the tree applies agree at <strong>every</strong> value you
          can enter. The stored thresholds are longer than that and are given exactly, at full precision, in the table below.
        </p>
        <Table caption="Each split this sample actually met. scikit-learn converts the input to float32 before comparing it with the stored double threshold, and ≤ takes the left child."
          headings={['node', 'field', 'your value', 'as float32', 'exact threshold', 'goes', 'to node']}
          rows={explanation.decision.path.map(step => [
            step.node, SMALL_LABELS[step.field], round(step.raw, 6), fixed(step.cast, 9), String(step.threshold),
            step.goesLeft ? 'left (≤)' : 'right (>)', step.next,
          ])} />
        <p className="fs-caption">
          Leaf {explanation.decision.leaf} holds {TREE.samples[explanation.decision.leaf]} fitted rows with class distribution
          {' '}({TREE.value[explanation.decision.leaf].map(value => round(value, 4)).join(', ')}) over cultivars 1, 2 and 3. The
          class-1 probability is {round(explanation.prediction, 9)}. That is a leaf proportion from the fitting rows, not a
          cultivar quality score and not a log-odds value.
          {outside.length > 0
            ? ` Extrapolation: ${outside.map(flag => `${SMALL_LABELS[flag.index]} ${round(flag.value, 4)} lies ${flag.below ? 'below' : 'above'} the observed fitting range ${round(flag.low, 4)} to ${round(flag.high, 4)}`).join('; ')}. The tree still answers, because a threshold comparison always answers; the answer describes no observed specimen.`
            : ' Every value lies inside the observed fitting range for its field.'}
        </p>
      </div>

      <div className="fs-stage">
        <h4>2. One coalition, built from {reference.rows.length} actual reference rows</h4>
        <div className="fs-controls">
          <Field label="Inspect one coalition">
            <select value={inspectedMask} onChange={event => setInspectedMask(Number(event.target.value))}>
              {explanation.masks.map(entry => (
                <option key={entry.mask} value={entry.mask}>
                  retain {entry.mask === 0 ? 'nothing' : SMALL_LABELS.filter((_, index) => (entry.mask & (1 << index)) !== 0).join(' + ')}
                </option>
              ))}
            </select>
          </Field>
        </div>
        <Table caption={`The first twelve of ${reference.rows.length} hybrid rows for this coalition: the retained fields take this sample's values and every other field comes from the same donor row`}
          headings={['donor', 'alcohol', 'malic acid', 'flavanoids', 'proline', 'leaf', 'class-1 probability']}
          rows={inspected.rows.slice(0, 12).map((row, index) => {
            const decision = treeDecision(TREE, row);
            return [
              `source row ${reference.ids[index]}`,
              ...row.map((value, column) => ((inspectedMask & (1 << column)) !== 0 ? `${round(value, 4)} ←` : round(value, 4))),
              decision.leaf, round(decision.probabilities[fourFieldModel.classIndex], 4),
            ];
          })} />
        <Table caption={`Where all ${reference.rows.length} hybrid rows landed, and the average they produce`}
          headings={['leaf', 'hybrid rows reaching it', 'its class-1 probability', 'contribution to the average']}
          rows={tally.leaves.map(entry => [
            entry.leaf, entry.count, round(entry.probability, 4),
            round(entry.count * entry.probability / reference.rows.length, 6),
          ])} />
        <p className="fs-readout" aria-live="polite">
          v(retain {retained.length === 0 ? 'nothing' : retained.join(' + ')}) = {round(inspected.value, 9)}.
          {inspectedMask === 0
            ? ' This is the baseline: the reference population predicted with none of this sample\'s values retained.'
            : ''}
          {inspectedMask === 15
            ? ' Every field is retained, so every hybrid row is this sample and the average is its own prediction.'
            : ''}
        </p>
        <Folded summary={`Inspect all ${reference.rows.length} hybrid rows for this coalition`}>
          <Table caption="Every donor row, its hybrid input and the leaf that input reaches" scroll
            headings={['donor', 'alcohol', 'malic acid', 'flavanoids', 'proline', 'leaf', 'class-1 probability']}
            rows={inspected.rows.map((row, index) => {
              const decision = treeDecision(TREE, row);
              return [
                `source row ${reference.ids[index]}`,
                ...row.map(value => round(value, 4)),
                decision.leaf, round(decision.probabilities[fourFieldModel.classIndex], 4),
              ];
            })} />
        </Folded>
      </div>

      <div className="fs-stage">
        <h4>3. The allocation over this reference</h4>
        <Waterfall model={chart} names={SMALL_LABELS} baselineLabel="baseline"
          totalLabel="reconstructed" unit="probability units" digits={6}
          describe={`A waterfall from the baseline ${round(chart.baseline, 6)} through ${SMALL_LABELS.map((label, index) => `${label} ${round(chart.segments[index].value, 6)}`).join(', ')} to ${round(settled(chart.reconstruction), 6)}.`} />
        <Table caption="All sixteen coalition values, in bitmask order"
          headings={['retained fields', 'v(S)']}
          rows={explanation.masks.map(entry => [
            entry.mask === 0 ? 'nothing' : SMALL_LABELS.filter((_, index) => (entry.mask & (1 << index)) !== 0).join(' + '),
            round(entry.value, 9),
          ])}
          rowClass={index => (index === inspectedMask ? 'is-selected' : undefined)} />
        <p className="fs-readout" aria-live="polite">
          Baseline {round(explanation.baseline, 6)} over {reference.rows.length} reference rows;
          contributions {SMALL_LABELS.map((label, index) => `${label} ${signed(explanation.phi[index], 6)}`).join(', ')};
          reconstructed class-1 probability {round(settled(chart.reconstruction), 9)} against the actual
          {' '}{round(explanation.prediction, 9)}, within {round(Math.abs(explanation.efficiencyError), 12)}.
          {explanation.phi[1] === 0
            ? ' Malic acid receives exactly zero, because this saved tree never splits on it: replacing that coordinate cannot change any output. That is a verified property of this fitted tree, not a statement that malic acid has no association with cultivar.'
            : ''}
          {active.reference === 'cohort'
            ? ` This reference is the twelve highest-index fitting rows, which the source file's class ordering makes an all-class-3 cohort: source rows ${reference.ids.join(', ')}. Its baseline is ${round(explanation.baseline, 6)}. The statement has changed from a comparison with the whole fitting population to a comparison with that cohort; the model, the sample and its prediction did not change.`
            : ''}
          {' '}An individual contribution is not a probability and need not lie between zero and one; the reconstructed output does.
        </p>
      </div>

      <p className="fs-caption">
        Recorded for comparison: source row {sourceRow.sourceId} with the whole fitting reference has baseline
        {' '}{round(sourceRow.baseline, 4)}, class-1 probability {round(sourceRow.prediction, 4)} and contributions
        {' '}({sourceRow.phi.map(value => round(value, 6)).join(', ')}). This page recomputes those values from the saved tree
        rather than reading them back.
      </p>
    </>}
  </Investigation>;
}
