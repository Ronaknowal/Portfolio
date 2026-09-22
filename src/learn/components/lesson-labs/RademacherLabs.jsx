import { useState } from 'react';
import {
  Drawing, Investigation, NumberField, LiveResult, Readout, Select, SignRow, Table,
  asFraction, displayValue, exactly, fixed, movementAnswer, round, signGlyph, useStagedInvestigation,
} from './RademacherShared.jsx';
import {
  absoluteComplexity, achievedCorrelation, ballGeometry, bestResponse, confidenceAddend, convexMixture,
  distinctRows, empiricalComplexity, evaluatePredictor, featureEnergy, isPositiveSemidefinite, kernelComplexity,
  feasibilitySlack, gapOutcome, limits, linearComplexity, mapFeatures, marginBound, norm2, rampLoss,
  selectByValidation, signPatternIndex, signPatterns, thresholdRows,
} from '../../data/rademacher-models.js';
import {
  experimentSettings, fittedModels, majorityBaseline, observations, representation, roles, selection, zeroCandidate,
} from '../../data/rademacher-data.js';



const signature = draft => JSON.stringify(draft);

/* ===================================================== investigation 1 */

const START_ROWS = thresholdRows([-1, 0, 1]);


export function BestResponseLab() {
  const state = useStagedInvestigation({
    rows: START_ROWS.map(row => row.slice()),
    signs: [1, -1, 1],
    convention: 'signed',
  }, ['winner', 'average'], signature);
  const draft = state.draft;
  const columns = draft.signs.length;
  const model = empiricalComplexity(draft.rows);
  const absolute = absoluteComplexity(draft.rows);
  const patternIndex = signPatternIndex(draft.signs);
  const usingAbsolute = draft.convention === 'absolute';
  const enumeration = usingAbsolute ? absolute : model;
  const patternScores = draft.rows.map(row =>
    (usingAbsolute ? Math.abs(row.reduce((sum, value, index) => sum + draft.signs[index] * value, 0))
      : row.reduce((sum, value, index) => sum + draft.signs[index] * value, 0)) / columns);
  const patternBest = Math.max(...patternScores);
  const patternWinners = patternScores
    .map((score, index) => (displayValue(score, 6) === displayValue(patternBest, 6) ? index : -1))
    .filter(index => index >= 0);
  const complexity = usingAbsolute ? absolute.complexity : model.complexity;
  const unique = distinctRows(draft.rows);

  const winnerShown = state.ready('winner');
  const averageShown = state.ready('average');

  const setCell = (rowIndex, column, value) => state.edit({
    rows: draft.rows.map((row, index) => (index === rowIndex
      ? row.map((cell, position) => (position === column ? value : cell)) : row)),
  });
  const addRow = () => {
    if (draft.rows.length >= limits.maxLabRows) return;
    state.edit({ rows: [...draft.rows, new Array(columns).fill(1)] });
  };
  const duplicateRow = () => {
    if (draft.rows.length >= limits.maxLabRows) return;
    state.edit({ rows: [...draft.rows, draft.rows[0].slice()] });
  };
  const addMixture = () => {
    if (draft.rows.length >= limits.maxLabRows || draft.rows.length < 2) return;
    const weights = draft.rows.map((_row, index) => (index === 0 ? 0.25 : (index === draft.rows.length - 1 ? 0.75 : 0)));
    state.edit({ rows: [...draft.rows, convexMixture(draft.rows, weights)] });
  };
  const removeRow = index => {
    if (draft.rows.length <= 1) return;
    state.edit({ rows: draft.rows.filter((_row, position) => position !== index) });
  };
  const addColumn = () => {
    if (columns >= limits.maxLabColumns) return;
    state.edit({ rows: draft.rows.map(row => [...row, 1]), signs: [...draft.signs, 1] });
  };
  const removeColumn = () => {
    if (columns <= 1) return;
    state.edit({ rows: draft.rows.map(row => row.slice(0, -1)), signs: draft.signs.slice(0, -1) });
  };

  const presets = [
    ['Positive thresholds on −1, 0, 1', { rows: thresholdRows([-1, 0, 1]).map(row => row.slice()), signs: [1, -1, 1], convention: 'signed' }],
    ['Both orientations', { rows: thresholdRows([-1, 0, 1], { bothOrientations: true }).map(row => row.slice()), signs: [1, -1, 1], convention: 'signed' }],
    ['Every sign row', { rows: signPatterns(3), signs: [1, -1, 1], convention: 'signed' }],
    ['The all-positive rule alone', { rows: [[1, 1, 1]], signs: [1, -1, 1], convention: 'signed' }],
    ['Two inputs, three thresholds', { rows: thresholdRows([2, 5]).map(row => row.slice()), signs: [1, -1], convention: 'signed' }],
  ];

  return <Investigation title="Investigation 1 — choose the best row, then average. Never the other way round."
    onReset={state.reset}
    question="One sign pattern is on the table. Which of the allowed rows matches it best, and what is that row's normalised correlation? Edit the rows and sign pattern to inspect both together."
    role={{ kind: 'exact', text: 'Exact enumeration. Every one of the 2ⁿ sign patterns is evaluated; nothing here is sampled.' }}
    note="The prediction rows below are the whole class, as far as this sample can tell. Edit them, add one, duplicate one, or load a different class — then ask whether the answer should move.">

    <div className="rad-presets">
      {presets.map(([label, inputs]) => <button key={label} type="button" onClick={() => state.suggest(inputs)}>
        {label}
      </button>)}
    </div>

    <SignRow label={`Sign pattern σ (pattern ${patternIndex + 1} of ${2 ** columns})`} signs={draft.signs}
      describe="Each button flips one observation's sign between +1 and −1"
      onChange={signs => state.edit({ signs })} />

    <Table caption={`The allowed prediction rows. Every entry is editable between −${limits.entryMagnitude} and ${limits.entryMagnitude}.`}
      headings={['Row', ...draft.signs.map((_sign, index) => `x${index + 1}`), 'Remove']}
      rows={draft.rows.map((row, rowIndex) => [
        `f${rowIndex + 1}`,
        ...row.map((value, column) => <NumberField key={column} label={`f${rowIndex + 1} at x${column + 1}`}
          value={value} min={-limits.entryMagnitude} max={limits.entryMagnitude} decimals={3} hideLabel
          onChange={next => setCell(rowIndex, column, next)} />),
        <button key="remove" type="button" onClick={() => removeRow(rowIndex)}
          disabled={draft.rows.length <= 1}>Remove f{rowIndex + 1}</button>,
      ])} />

    <div className="rad-presets">
      <button type="button" onClick={addRow} disabled={draft.rows.length >= limits.maxLabRows}>Add a row of +1s</button>
      <button type="button" onClick={duplicateRow} disabled={draft.rows.length >= limits.maxLabRows}>Duplicate row f1</button>
      <button type="button" onClick={addMixture} disabled={draft.rows.length >= limits.maxLabRows || draft.rows.length < 2}>
        Add the convex average ¼f1 + ¾f{draft.rows.length}
      </button>
      <button type="button" onClick={addColumn} disabled={columns >= limits.maxLabColumns}>Add an observation</button>
      <button type="button" onClick={removeColumn} disabled={columns <= 1}>Remove an observation</button>
    </div>

    <div className="rad-controls">
      <Select label="Convention" value={draft.convention} onChange={convention => state.edit({ convention })}
        options={[['signed', 'No absolute value — this lesson\'s definition'], ['absolute', 'Absolute value inside the supremum — a different measurement']]}
        hint={usingAbsolute
          ? 'This measures the class enlarged by every row\'s negative. Constants and factors from the signed convention do not carry over.'
          : 'E_σ max_f (1/n) Σ σᵢ f(xᵢ), with no absolute value and a 1/n normalisation.'} />
    </div>

    <LiveResult
      state={state.stage('winner')}
      
      
      
      
      calculateInputs={committed => {
        const n = committed.signs.length;
        const scores = committed.rows.map(row => {
          const raw = row.reduce((sum, value, index) => sum + committed.signs[index] * value, 0) / n;
          return committed.convention === 'absolute' ? Math.abs(raw) : raw;
        });
        const best = Math.max(...scores);
        
        const winners = scores.map((score, index) =>
          (displayValue(score, 6) === displayValue(best, 6) ? index : -1)).filter(index => index >= 0);
        return {
          outcome: String(winners[0]),
          acceptable: winners.map(String),
          value: best,
          explain: winners.length > 1
            ? `Rows ${winners.map(index => `f${index + 1}`).join(', ')} all reach ${round(best, 6)}, so each is a maximizing row. `
              + 'A tie is a real feature of the class, not a rounding artefact.'
            : `Row f${winners[0] + 1} reaches ${round(best, 6)}; no other row gets closer to this pattern.`
              + (best < 0 ? ' Note that the best available correlation here is negative: a class with no good match for this pattern still has a maximum, and it is not zero.' : ''),
        };
      }}
      describe={patternWinners.length > 1 ? 'Every tied row reaches the same maximum.' : undefined} />

    {winnerShown && <Table caption="Every row's normalised correlation with the current pattern. The winner is marked."
      headings={['Row', usingAbsolute ? '|Σ σᵢ f(xᵢ)| / n' : 'Σ σᵢ f(xᵢ) / n']}
      cellClass={rowIndex => (patternWinners.includes(rowIndex) ? 'is-winner' : (patternScores[rowIndex] < 0 ? 'is-negative' : undefined))}
      rows={draft.rows.map((row, index) => [`f${index + 1}`, fixed(patternScores[index], 6)])} />}

    {winnerShown && <LiveResult
      state={state.stage('average')}
      
      
      
      
      calculateInputs={committed => {
        const committedModel = committed.convention === 'absolute'
          ? absoluteComplexity(committed.rows)
          : empiricalComplexity(committed.rows);
        const exact = committedModel.complexity;
        const n = committed.signs.length;
        const single = Math.max(...committed.rows.map(row => {
          const raw = row.reduce((sum, value, index) => sum + committed.signs[index] * value, 0) / n;
          return committed.convention === 'absolute' ? Math.abs(raw) : raw;
        }));
        
        const classified = gapOutcome(single, exact, { digits: 6, below: 'smaller', equal: 'same', above: 'larger' });
        return {
          outcome: classified.outcome,
          value: exact,
          explain: `Averaging the ${2 ** n} per-pattern maxima gives ${round(exact, 6)}`
            + (asFraction(exact) ? `, exactly ${asFraction(exact)}` : '')
            + `. Averaging each row first and then taking the maximum gives `
            + `${round(committedModel.maxAfterAveraging, 6)} — a different order of operations, even when these values coincide.`,
        };
      }} />}

    {averageShown && <>
      <Table caption={`All ${2 ** columns} sign patterns, each with its winning row and best value`}
        headings={['Pattern', 'σ', 'Winning row', 'Best value']}
        rowClass={index => (index === patternIndex ? 'is-leading' : undefined)}
        rows={enumeration.patterns.map((pattern, index) => [
          String(index + 1),
          pattern.map(sign => (sign > 0 ? '+' : '−')).join(''),
          enumeration.winners[index].map(row => `f${row + 1}`).join(' or '),
          fixed(enumeration.maxima[index], 6),
        ])} />
      <Readout items={[
        ['Empirical complexity', exactly(complexity)],
        ['Averaging first instead', exactly(enumeration.maxAfterAveraging)],
        ['Distinct rows', `${unique.rows.length} of ${draft.rows.length}`,
          unique.duplicated.length
            ? `${unique.duplicated.length} duplicate group${unique.duplicated.length === 1 ? '' : 's'} — a repeated row is another name for an option you already had, so it changes nothing`
            : 'no duplicates'],
        ['Under the other convention', exactly(usingAbsolute ? model.complexity : absolute.complexity),
          'a different class, not a different rounding'],
      ]} />
      <p className="rad-caption">A high value here is available flexibility, not a prediction that this class will
        overfit. Under the signed convention, the all-positive singleton scores 0 and may still predict the real labels terribly; capacity and
        accuracy are different questions, which is why a risk bound needs a training-loss term as well.</p>
    </>}
  </Investigation>;
}

/* ===================================================== investigation 2 */

/** Steer the signed sum: the norm ball, the supporting point, and the kernel
 *  branch that does the same calculation through inner products alone. */
export function SignedGeometryLab() {
  const state = useStagedInvestigation({
    vectors: [[1, 0], [1, 0]],
    signs: [1, -1],
    radius: 1,
    candidate: [0, 0],
    similarity: 0.9,
  }, ['pattern', 'average', 'kernel'], signature);
  const draft = state.draft;
  const geometry = ballGeometry(draft.vectors, draft.signs, draft.radius, { width: 300, height: 240 });
  const candidate = achievedCorrelation(draft.vectors, draft.signs, draft.candidate, draft.radius);
  const exact = linearComplexity(draft.vectors, draft.radius);
  const gram = [[1, draft.similarity], [draft.similarity, 1]];
  const gramValid = isPositiveSemidefinite(gram);
  const kernel = gramValid ? kernelComplexity(gram, draft.radius) : null;

  const patternShown = state.ready('pattern');
  const averageShown = state.ready('average');
  const kernelShown = state.ready('kernel');

  const setVector = (index, column, value) => state.edit({
    vectors: draft.vectors.map((vector, position) => (position === index
      ? vector.map((cell, coordinate) => (coordinate === column ? value : cell)) : vector)),
  });

  const presets = [
    ['Two copies of (1, 0)', { vectors: [[1, 0], [1, 0]], signs: [1, -1], radius: 1, candidate: [0, 0], similarity: 0.9 }],
    ['Perpendicular (1, 0) and (0, 1)', { vectors: [[1, 0], [0, 1]], signs: [1, -1], radius: 1, candidate: [0, 0], similarity: 0.9 }],
    ['The same pair, rotated 90°', { vectors: [[0, 1], [-1, 0]], signs: [1, -1], radius: 1, candidate: [0, 0], similarity: 0.9 }],
    ['Perpendicular, budget doubled', { vectors: [[1, 0], [0, 1]], signs: [1, -1], radius: 2, candidate: [0, 0], similarity: 0.9 }],
    ['Practice 3: (3, 0), (0, 4) with B = 2', { vectors: [[3, 0], [0, 4]], signs: [1, -1], radius: 2, candidate: [0, 0], similarity: 0.9 }],
  ];

  return <Investigation title="Investigation 2 — align a bounded coefficient with the noise" onReset={state.reset}
    question="With these two inputs and this sign pattern, what is the signed sum, and what is the largest correlation any coefficient inside the ball can reach? Is the maximising coefficient unique?"
    role={{ kind: 'exact', text: 'Exact. The inner optimisation has a closed form — training a classifier on artificial labels would be unnecessary here, and would solve a different problem.' }}
    note="Start with the opening setup: two identical inputs and opposite signs. It is the case where the answer is most surprising.">

    <div className="rad-presets">
      {presets.map(([label, inputs]) => <button key={label} type="button" onClick={() => state.suggest(inputs)}>{label}</button>)}
    </div>

    <div className="rad-controls is-wide">
      {draft.vectors.map((vector, index) => vector.map((value, column) => (
        <NumberField key={`${index}-${column}`} label={`x${index + 1} coordinate ${column + 1}`} value={value}
          min={-limits.vectorCoordinate} max={limits.vectorCoordinate} decimals={3}
          onChange={next => setVector(index, column, next)} />
      )))}
      <NumberField label="Budget B" value={draft.radius} min={0} max={limits.maxRadius} decimals={3}
        onChange={radius => state.edit({ radius })}
        hint="B = 0 leaves only the origin feasible, and every correlation is then exactly zero." />
    </div>

    <div className="rad-presets">
      <button type="button" disabled={draft.vectors.length >= limits.maxVectors}
        onClick={() => state.edit({ vectors: [...draft.vectors, [0.5, 0.5]], signs: [...draft.signs, 1] })}>
        Add an input vector
      </button>
      <button type="button" disabled={draft.vectors.length <= 1}
        onClick={() => state.edit({ vectors: draft.vectors.slice(0, -1), signs: draft.signs.slice(0, -1) })}>
        Remove the last input
      </button>
    </div>

    <SignRow label="Sign pattern σ" signs={draft.signs} onChange={signs => state.edit({ signs })}
      describe="Each button flips one observation's sign" />

    <LiveResult
      state={state.stage('pattern')}
      
      
      
      calculateInputs={committed => {
        const best = bestResponse(committed.vectors, committed.signs, committed.radius);
        const outcome = committed.radius === 0 ? 'origin' : (best.optimizerUnique ? 'unique' : 'every');
        return {
          outcome,
          value: best.optimum,
          explain: `The signed sum is (${best.v.map(value => round(value, 4)).join(', ')}) with length `
            + `${round(best.length, 6)}, so the optimum is B‖v‖/n = ${round(best.optimum, 6)}.`
            + (best.degenerate ? ` ${best.degenerate[0].toUpperCase()}${best.degenerate.slice(1)}.` : ''),
        };
      }} />

    {patternShown && <>
      <Drawing width={geometry.width} height={geometry.height} className="rad-ball-lab"
        title="The ball, the signed sum and the supporting point"
        describe={`Signed sum (${geometry.best.v.map(value => round(value, 3)).join(', ')}), length ${round(geometry.best.length, 4)}.`}>
        <line className="rad-grid" x1={0} y1={geometry.origin.y} x2={geometry.width} y2={geometry.origin.y} />
        <line className="rad-grid" x1={geometry.origin.x} y1={0} x2={geometry.origin.x} y2={geometry.height} />
        <circle className="rad-ball" cx={geometry.origin.x} cy={geometry.origin.y} r={geometry.radiusPixels} />
        {geometry.arrows.map(arrow => <line key={arrow.index}
          className={`rad-arrow${arrow.sign < 0 ? ' is-flipped' : ''}`}
          x1={arrow.base.x} y1={arrow.base.y} x2={arrow.tip.x} y2={arrow.tip.y} />)}
        {geometry.best.length > feasibilitySlack && <line className="rad-sum" x1={geometry.origin.x} y1={geometry.origin.y}
          x2={geometry.sum.tip.x} y2={geometry.sum.tip.y} />}
        {geometry.support && <circle className="rad-support" cx={geometry.support.point.x}
          cy={geometry.support.point.y} r={5} />}
        {!geometry.support && <text className="rad-small rad-muted" x={geometry.origin.x + 8} y={geometry.origin.y - 8}>
          v = 0
        </text>}
      </Drawing>
      <div className="rad-controls">
        {draft.candidate.map((value, column) => <NumberField key={column}
          label={`Your own coefficient w, coordinate ${column + 1}`} value={value}
          min={-limits.maxRadius} max={limits.maxRadius} decimals={3}
          onChange={next => state.edit({ candidate: draft.candidate.map((cell, position) => (position === column ? next : cell)) })} />)}
      </div>
      <Readout items={[
        ['Signed sum v', `(${geometry.best.v.map(value => round(value, 4)).join(', ')})`],
        ['Optimum for this pattern', round(geometry.best.optimum, 6)],
        ['Your coefficient', candidate.feasible
          ? `reaches ${round(candidate.achieved, 6)}, short of the optimum by ${round(candidate.shortfall, 6)}`
          : `has norm ${round(candidate.candidateNorm, 6)}, which is outside the ball of radius ${draft.radius}`],
        ['Feasible?', candidate.feasible ? 'yes' : 'no — this coefficient is not in the class, so its value is not a member\'s value'],
      ]} />
    </>}

    {patternShown && <LiveResult
      state={state.stage('average')}
      
      
      
      
      calculateInputs={committed => {
        const model = linearComplexity(committed.vectors, committed.radius);
        const classified = gapOutcome(model.energyUpper, model.complexity,
          { digits: 6, below: 'below', equal: 'equal', above: 'above' });
        const coincide = classified.outcome === 'equal';
        return {
          outcome: classified.outcome,
          value: model.complexity,
          explain: `The exact value is ${round(model.complexity, 6)} and the energy bound is `
            + `${round(model.energyUpper, 6)}. `
            + (coincide
              ? (committed.radius === 0
                ? 'The zero budget makes both quantities zero, whatever the signed-sum lengths.'
                : 'They agree to every decimal shown here: the signed sums are all the same length, or so nearly so that the difference is below the precision this page prints.')
              : 'The bound replaces an average length by a root-mean-square length, which is never smaller — and the gap is exactly the directional information it discards.'),
        };
      }} />}

    {averageShown && <>
      <Table caption={`Every sign pattern's signed sum and its optimum, at B = ${draft.radius}`}
        headings={['σ', 'signed sum v', '‖v‖₂', 'B‖v‖₂/n']}
        rows={exact.patterns.map((pattern, index) => [
          pattern.map(sign => (sign > 0 ? '+' : '−')).join(''),
          `(${exact.sums[index].map(value => round(value, 3)).join(', ')})`,
          fixed(norm2(exact.sums[index]), 6),
          fixed(exact.maxima[index], 6),
        ])} />
      <Readout items={[
        ['Exact empirical complexity', exactly(exact.complexity)],
        ['Feature-energy bound B√(Σ‖xᵢ‖²)/n', round(exact.energyUpper, 6)],
        ['Largest-row bound B maxᵢ‖xᵢ‖/√n', round(exact.maxRowUpper, 6)],
        ['Rotating both inputs together', 'a change of coordinates: every length, every signed sum length and the answer are unchanged'],
      ]} />
    </>}

    <h4 className="rad-question">The same calculation through inner products alone</h4>
    <div className="rad-controls">
      <NumberField label="Similarity r = ⟨φ(x₁), φ(x₂)⟩" value={draft.similarity} min={-1} max={1} decimals={3}
        onChange={similarity => state.edit({ similarity })}
        hint="The 2×2 matrix [[1, r], [r, 1]] is positive semidefinite exactly when |r| ≤ 1. Outside that it is not a Gram matrix and the calculation is refused." />
    </div>
    {gramValid && <LiveResult
      state={state.stage('kernel')}
      
      
      
      
      calculateInputs={committed => {
        const matrix = [[1, committed.similarity], [committed.similarity, 1]];
        const model = kernelComplexity(matrix, committed.radius);
        const classified = gapOutcome(model.traceUpper, model.complexity,
          { digits: 10, below: 'below', equal: 'equal', above: 'above' });
        return {
          outcome: classified.outcome,
          value: model.complexity,
          explain: `The four quadratic forms give ${model.maxima.map(value => round(value, 6)).join(', ')}, averaging to `
            + `${round(model.complexity, 10)}. The trace bound is ${round(model.traceUpper, 10)}, and it does not move with r at all: `
            + 'the trace only sees the diagonal.',
        };
      }} />}
    {kernelShown && kernel && <Table caption="The four sign patterns, and why r and −r give the same average"
      headings={['σ', 'σᵀKσ at r', 'σᵀKσ at −r']}
      rows={kernel.patterns.map((pattern, index) => [
        pattern.map(sign => (sign > 0 ? '+' : '−')).join(''),
        round(kernel.quadratics[index], 6),
        round(kernelComplexity([[1, -draft.similarity], [-draft.similarity, 1]], draft.radius).quadratics[index], 6),
      ])} />}
    {!gramValid && <p className="rad-note">That similarity does not give a positive semidefinite matrix, so it is not
      the Gram matrix of any feature map. The calculation is refused rather than returning a number from a square root
      of a negative quantity.</p>}
  </Investigation>;
}

/* ===================================================== investigation 3 */

/** Build a margin bound from scores you choose, and watch every term move. */
export function MarginBoundLab() {
  const [scaleBaseline, setScaleBaseline] = useState(null);
  const state = useStagedInvestigation({
    inputs: [-0.2, 0.1, 0.4, 1.2],
    labels: [1, 1, 1, 1],
    weight: 1,
    radius: 1,
    rho: 0.5,
    delta: 0.05,
    comparisons: 1,
  }, ['terms', 'scale'], signature);
  const draft = state.draft;
  const margins = draft.inputs.map((x, index) => draft.labels[index] * draft.weight * x);
  const energy = featureEnergy(draft.inputs.map(x => [x]));
  const feasible = Math.abs(draft.weight) <= draft.radius + feasibilitySlack;
  const bound = feasible
    ? marginBound({
      margins, radius: draft.radius, energy, rho: draft.rho, delta: draft.delta, comparisons: draft.comparisons,
    })
    : null;

  const termsShown = state.ready('terms');
  const scaleShown = state.ready('scale');

  const setInput = (index, value) => state.edit({
    inputs: draft.inputs.map((cell, position) => (position === index ? value : cell)),
  });
  const setLabel = (index, value) => state.edit({
    labels: draft.labels.map((cell, position) => (position === index ? value : cell)),
  });
  const addRow = () => {
    if (draft.inputs.length >= limits.maxScalarRows) return;
    state.edit({ inputs: [...draft.inputs, 0.5], labels: [...draft.labels, 1] });
  };
  const removeRow = index => {
    if (draft.inputs.length <= 3) return;
    state.edit({
      inputs: draft.inputs.filter((_value, position) => position !== index),
      labels: draft.labels.filter((_value, position) => position !== index),
    });
  };

  return <Investigation title="Investigation 3 — assemble a margin bound, term by term" onReset={() => { state.reset(); setScaleBaseline(null); }}
    question="With these four scalar inputs, this coefficient and this margin threshold, what is the mean ramp loss, and what does the whole expression come to?"
    role={{ kind: 'exact', text: 'Exact arithmetic on constructed inputs. These are four made-up scalar observations, chosen so every term can be checked by hand.' }}
    note="A four-observation sample makes the confidence term enormous. A vacuous result here is expected, and it is left visible rather than clipped away.">

    <Table caption="The observations. Each score is w·x, and each margin is y times that score."
      headings={['Row', 'Input x', 'Label y', 'Remove']}
      rows={draft.inputs.map((value, index) => [
        `obs ${index + 1}`,
        <NumberField key="x" label={`x${index + 1}`} value={value} min={-3} max={3} decimals={3}
          onChange={next => setInput(index, next)} />,
        <Select key="y" label={`y${index + 1}`} value={String(draft.labels[index])}
          onChange={next => setLabel(index, Number(next))} options={[['1', '+1'], ['-1', '−1']]} />,
        <button key="remove" type="button" onClick={() => removeRow(index)}
          disabled={draft.inputs.length <= 3}>Remove</button>,
      ])} />
    <div className="rad-presets">
      <button type="button" onClick={addRow} disabled={draft.inputs.length >= limits.maxScalarRows}>Add an observation</button>
      <button type="button" onClick={() => state.suggest({ ...draft, rho: 0.25 })}>Set ρ = .25</button>
      <button type="button" onClick={() => state.suggest({ ...draft, inputs: draft.inputs.map((value, index) => index === 0 ? 0.2 : value) })}>
        Move the first input from −.2 to .2
      </button>
      <button type="button" onClick={() => state.suggest({
        ...draft, weight: 3 * draft.weight, radius: 3 * draft.radius, rho: 3 * draft.rho,
      })} disabled={3 * draft.radius > limits.maxRadius || 3 * draft.rho > limits.maxRho}>
        Multiply w, B and ρ all by 3
      </button>
    </div>

    <div className="rad-controls is-wide">
      <NumberField label="Coefficient w" value={draft.weight} min={-limits.maxRadius} max={limits.maxRadius}
        decimals={3} onChange={weight => state.edit({ weight })}
        hint={feasible ? 'Must satisfy |w| ≤ B, or it is not a member of the declared class.' : `‖w‖ = ${round(Math.abs(draft.weight), 4)} exceeds B = ${draft.radius}.`} />
      <NumberField label="Budget B" value={draft.radius} min={0} max={limits.maxRadius} decimals={3}
        onChange={radius => state.edit({ radius })} />
      <NumberField label="Margin threshold ρ" value={draft.rho} min={limits.minRho} max={limits.maxRho} decimals={3}
        onChange={rho => state.edit({ rho })} />
      <NumberField label="Failure allowance δ" value={draft.delta} min={limits.minDelta} max={limits.maxDelta}
        decimals={3} onChange={delta => state.edit({ delta })}
        hint="δ changes only the confidence term. It does not touch a margin or the empirical complexity." />
      <NumberField label="Predeclared comparisons K" value={draft.comparisons} min={1} max={limits.maxComparisons}
        decimals={0} onChange={comparisons => state.edit({ comparisons })}
        hint="Each predeclared candidate gets allowance δ/K, so the confidence term grows with the number of things you were entitled to compare." />
    </div>

    {!feasible && <p className="rad-note">This coefficient is outside the declared ball, so it is not a member of the
      class the bound is about. The norm is {round(Math.abs(draft.weight), 6)} against a budget
      of {draft.radius}. Change one of the two before applying.</p>}

    <LiveResult
      state={state.stage('terms')}
      
      
      
      
      calculateInputs={committed => {
        if (Math.abs(committed.weight) > committed.radius + feasibilitySlack) {
          return { outcome: 'vacuous', value: null, explain: 'The coefficient is outside the declared ball, so there is no bound for it to satisfy.' };
        }
        const committedMargins = committed.inputs.map((x, index) => committed.labels[index] * committed.weight * x);
        if (scaleBaseline === null) setScaleBaseline(committed);
        const result = marginBound({
          margins: committedMargins, radius: committed.radius,
          energy: featureEnergy(committed.inputs.map(x => [x])),
          rho: committed.rho, delta: committed.delta, comparisons: committed.comparisons,
        });
        return {
          outcome: result.informative ? 'informative' : 'vacuous',
          value: result.empiricalRamp,
          explain: `${round(result.empiricalRamp, 6)} + ${round(result.complexityAddend, 6)} + `
            + `${round(result.confidence, 6)} = ${round(result.raw, 6)}. `
            + (result.informative
              ? 'That is below 1, so the statement rules something out.'
              : 'The loss already lies in [0, 1], so an upper bound of ' + round(result.raw, 6)
                + ' rules nothing out. That is a statement about this expression, not evidence that the predictor is bad.'),
        };
      }} />

    {termsShown && bound && <>
      <Table caption="Every observation's margin and its exact ramp contribution"
        headings={['Row', 'x', 'y', 'margin m = y w x', 'ramp φ_ρ(m)']}
        cellClass={(index, column) => (column === 4 && bound.ramp[index] === 1 ? 'is-negative' : undefined)}
        rows={draft.inputs.map((value, index) => [
          `obs ${index + 1}`, String(value), signGlyph(draft.labels[index]),
          fixed(margins[index], 6), fixed(bound.ramp[index], 6),
        ])} />
      <Readout items={[
        ['Mean ramp loss', round(bound.empiricalRamp, 6), `${bound.trainingErrors} at or below margin 0, ${bound.smallMargins} inside the ramp`],
        ['Complexity addend 2B·energy/ρ', round(bound.complexityAddend, 6), `energy = ${round(energy, 6)}`],
        ['Confidence 3√(ln(2K/δ)/2n)', round(bound.confidence, 6), `K = ${draft.comparisons}, δ = ${draft.delta}, n = ${bound.n}`],
        ['Raw sum', round(bound.raw, 6), bound.informative ? 'below the trivial ceiling' : 'above the trivial ceiling 1'],
        ['Trivial ceiling', '1', 'the loss lies in [0, 1], so 1 is always a valid bound'],
      ]} />
    </>}

    {scaleBaseline !== null && <LiveResult
      state={state.stage('scale')}
      
      
      
      
      
      
      calculateInputs={committed => {
        const evaluate = inputs => {
          if (Math.abs(inputs.weight) > inputs.radius + feasibilitySlack) return null;
          return marginBound({
            margins: inputs.inputs.map((x, index) => inputs.labels[index] * inputs.weight * x),
            radius: inputs.radius, energy: featureEnergy(inputs.inputs.map(x => [x])),
            rho: inputs.rho, delta: inputs.delta, comparisons: inputs.comparisons,
          }).raw;
        };
        const before = evaluate(scaleBaseline);
        const after = evaluate(committed);
        if (before === null || after === null) {
          return { outcome: 'unchanged', value: null, explain: 'One of the two states has a coefficient outside its ball, so there is nothing to compare.' };
        }
        setScaleBaseline(committed);
        return movementAnswer(before, after, { quantity: 'The raw sum' });
      }} />}

    {scaleShown && <p className="rad-caption">Scaling scores, budget and threshold together leaves every prediction,
      every ramp loss and the ratio B/ρ exactly where they were. Multiplying the scores alone cannot manufacture a
      better normalised guarantee; it just changes the units the margin is measured in.</p>}
  </Investigation>;
}

/* ===================================================== investigation 4 */

const FIT = roles.find(role => role.key === 'fit');
const VALIDATION = roles.find(role => role.key === 'validation');
const ASSESSMENT = roles.find(role => role.key === 'assessment');

const mappedRows = observations.map(row => mapFeatures(row.slice(1, 5), representation));
const allLabels = observations.map(row => row[5]);
const sliceRows = role => mappedRows.slice(role.from, role.to);
const sliceLabels = role => allLabels.slice(role.from, role.to);

/** The real workbench: five predeclared budgets, a selection rule you apply
 *  yourself, and a frozen assessment report opened separately from selection. */
export function BoundedNormLab() {
  const state = useStagedInvestigation({
    candidate: [0, 0, 0, 0, 0],
    declaredRadius: 2,
    rho: 1,
    loaded: 'none',
  }, ['selection', 'expression'], signature);
  const draft = state.draft;
  const [assessmentSeen, setAssessmentSeen] = useState(false);
  const candidateNorm = norm2(draft.candidate);
  const feasible = candidateNorm <= draft.declaredRadius + feasibilitySlack;
  const rhoIndex = experimentSettings.rhoValues.indexOf(draft.rho);

  const selectionShown = state.ready('selection');
  const expressionShown = state.ready('expression');
  // Inspection is sticky: resetting controls cannot make viewed assessment data unseen.

  const candidateFit = evaluatePredictor(sliceRows(FIT), sliceLabels(FIT), draft.candidate);
  const candidateValidation = evaluatePredictor(sliceRows(VALIDATION), sliceLabels(VALIDATION), draft.candidate);
  const candidateRamp = rampLoss(candidateFit.margins, draft.rho);
  const candidateRampMean = candidateRamp.reduce((sum, value) => sum + value, 0) / candidateRamp.length;

  const loadFitted = radius => {
    const model = fittedModels.find(entry => entry.radius === radius);
    state.edit({ candidate: model.weights.slice(), declaredRadius: radius, loaded: `fitted B = ${radius}` });
  };

  return <Investigation title="Investigation 4 — a real budget sweep, and two different best answers" onReset={() => { state.reset(); }}
    question="Apply the declared selection rule yourself: fewest validation mistakes, then smaller validation log loss, then smaller budget. Which budget does it choose — and separately, which budget has the smallest bound expression?"
    role={{ kind: 'measured', text: 'Measured on real data. 480 rows of the UCI Banknote subset, drawn without replacement from a fixed corpus — not an iid sample from a deployment population.' }}
    note="The declared validation rule selects the model below. Open its frozen assessment report separately when you want to inspect it. Reset restores controls but cannot make assessment data you have seen unseen.">

    <Table caption="The five predeclared candidates, with everything the selection rule is allowed to use"
      headings={['Budget B', 'Coefficient norm', 'Fit mistakes / 240', 'Fit log loss', 'Validation mistakes / 80', 'Validation log loss']}
      rows={fittedModels.map(model => [
        String(model.radius), fixed(model.norm, 6), String(model.fit.errors), fixed(model.fit.logLoss, 6),
        String(model.validation.errors), fixed(model.validation.logLoss, 6),
      ])} />

    <LiveResult
      state={state.stage('selection')}
      
      
      
      calculateInputs={() => {
        /* The rule is applied by a function that is not given an assessment
           field at all -- it throws if one is present. The selection cannot
           read the answer it is supposed to precede. */
        const chosen = selectByValidation(fittedModels.map(model => ({
          radius: model.radius,
          validationErrors: model.validation.errors,
          validationLogLoss: model.validation.logLoss,
        })));
        return {
          outcome: String(chosen.radius),
          value: chosen.radius,
          explain: `B = ${chosen.radius} has the fewest validation mistakes, so the later tie-breaks never come into `
            + `play. The recorded selection is B = ${selection.chosenRadius}.`,
        };
      }}
      describe="This selection uses validation only; the assessment report has no influence on it." />

    {!assessmentSeen && <button type="button" onClick={() => setAssessmentSeen(true)}>
      Inspect the frozen assessment report
    </button>}
    {assessmentSeen && <>
      <Table caption="Assessment of the frozen validation-selected study. These 80 rows were never used to fit or to choose anything."
          headings={['Budget B', 'Assessment mistakes / 80', 'Assessment log loss']}
          rowClass={index => (fittedModels[index].radius === selection.chosenRadius ? 'is-leading' : undefined)}
        rows={fittedModels.map(model => [
          String(model.radius), String(model.assessment.errors), fixed(model.assessment.logLoss, 6),
        ])} />
      <Readout items={[
        ['Selected by validation', `B = ${selection.chosenRadius}`],
        ['Its assessment mistakes', `${fittedModels.find(model => model.radius === selection.chosenRadius).assessment.errors} of 80`],
        ['Majority class from the fit rows', `${majorityBaseline.assessmentErrors} of 80`,
          `predicting ${signGlyph(majorityBaseline.classSign)} for everything`],
        ['Does error turn upward at large B?', 'no — it keeps falling across this particular sweep, and no turn was manufactured to make the story familiar'],
      ]} />
    </>}

    {selectionShown && <>
      <div className="rad-controls">
        <Select label="Margin threshold ρ" value={String(draft.rho)} onChange={value => state.edit({ rho: Number(value) })}
          options={experimentSettings.rhoValues.map(value => [String(value), `ρ = ${value}`])} />
      </div>
      <LiveResult
        state={state.stage('expression')}
        
        
        
        
        calculateInputs={committed => {
          const index = experimentSettings.rhoValues.indexOf(committed.rho);
          const smallest = fittedModels.reduce((best, model) =>
            (model.bounds[index].rawUpper < best.bounds[index].rawUpper ? model : best));
          const tied = fittedModels.filter(model =>
            displayValue(model.bounds[index].rawUpper) === displayValue(smallest.bounds[index].rawUpper));
          return {
            outcome: String(smallest.radius),
            acceptable: tied.map(model => String(model.radius)),
            value: smallest.bounds[index].rawUpper,
            explain: `B = ${smallest.radius} gives ${round(smallest.bounds[index].rawUpper, 6)}, the smallest of the five `
              + `at ρ = ${committed.rho}. Every one of them is still above 1, so none of them rules anything out — `
              + '"smallest" and "informative" are different claims. Choosing the smallest expression is also not the '
              + 'declared selection rule, which uses validation.',
          };
        }} />
    </>}

    {expressionShown && <Table caption={`The three terms at ρ = ${draft.rho}, with the trivial ceiling for comparison`}
      headings={['Budget B', 'Training ramp', '2B·energy/ρ', 'Confidence', 'Raw sum', 'Above the ceiling 1?']}
      rows={fittedModels.map(model => [
        String(model.radius), fixed(model.bounds[rhoIndex].empiricalRamp, 6),
        fixed(model.bounds[rhoIndex].complexityAddend, 6), fixed(experimentSettings.confidenceAddend, 6),
        fixed(model.bounds[rhoIndex].rawUpper, 6), model.bounds[rhoIndex].rawUpper > 1 ? 'yes' : 'no',
      ])} />}

    <h4 className="rad-question">Build your own predictor inside a declared ball</h4>
    <p className="rad-caption">The five coefficients start at zero — an unsolved candidate, not a trained one. Edit
      them and the fit and validation numbers recompute from the actual rows. Nothing is refitted: the budget in the
      bound is the class you declared, not whatever norm your coefficients happen to have.</p>
    <div className="rad-controls is-wide">
      {['variance', 'skewness', 'curtosis', 'entropy', 'intercept'].map((name, index) => (
        <NumberField key={name} label={`w for ${name}`} value={draft.candidate[index]} min={-4} max={4} decimals={3}
          onChange={next => state.edit({
            candidate: draft.candidate.map((value, position) => (position === index ? next : value)),
            loaded: 'your own edit',
          })} />
      ))}
      <Select label="Declared class budget B" value={String(draft.declaredRadius)}
        onChange={value => state.edit({ declaredRadius: Number(value) })}
        options={experimentSettings.radii.map(value => [String(value), `B = ${value}`])}
        hint="The bound is about every member of this ball, not about the particular coefficients you typed." />
    </div>
    <div className="rad-presets">
      {experimentSettings.radii.map(radius => <button key={radius} type="button" onClick={() => loadFitted(radius)}>
        Load the fitted B = {radius} coefficients
      </button>)}
      <button type="button" onClick={() => state.edit({ candidate: [0, 0, 0, 0, 0], loaded: 'none' })}>
        Back to all zeros
      </button>
    </div>
    <Readout items={[
      ['Your coefficients', draft.candidate.map(value => round(value, 3)).join(', '), `loaded: ${draft.loaded}`],
      ['Their norm', round(candidateNorm, 6),
        feasible ? `inside the declared ball B = ${draft.declaredRadius}` : `OUTSIDE the declared ball B = ${draft.declaredRadius}`],
      ['Fit mistakes', `${candidateFit.errors} of ${candidateFit.n}`],
      ['Validation mistakes', `${candidateValidation.errors} of ${candidateValidation.n}`],
      ['Mean training ramp at this ρ', round(candidateRampMean, 6)],
    ]} />
    {candidateNorm === 0 && <p className="rad-note">Every score is exactly zero, so every margin is zero and the ramp
      charges 1 on every row. Under the tie rule a zero score predicts +1, which
      makes {zeroCandidate.fit.errors} mistakes of {zeroCandidate.fit.n} on the fit
      rows and {zeroCandidate.validation.errors} of {zeroCandidate.validation.n} on
      validation.{assessmentSeen && <> It makes {zeroCandidate.assessment.errors} of {zeroCandidate.assessment.n} on assessment.</>}</p>}
    {!feasible && <p className="rad-note">These coefficients have norm {round(candidateNorm, 6)}, outside the declared
      ball. Their scores are still computed and shown, but they are not a member of the class the bound is about, so
      no bound on that class covers them.</p>}
    {assessmentSeen && <p className="rad-caption">You have opened the assessment rows. Repeated use of validation is
      tuning, and once assessment has been read, further comparison against it is tuning too. Final practical evidence
      would need data this page has not shown you.</p>}
  </Investigation>;
}
