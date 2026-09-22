/* Pure models and drawn geometry for the end-to-end supervised learning and
 * error analysis lesson.
 *
 * Three rules shaped this file, and each has cost this repository a defect.
 *
 * 1. EVERY SCORE CARRIES ITS ROLE. A number on this page is training evidence,
 *    validation evidence, a selection criterion, or a held-out estimate, and
 *    those four are not interchangeable. `scoreRecord` refuses a role outside
 *    that list, so a quantity cannot reach the page unlabelled. This is the
 *    defect this particular lesson exists to teach a learner to find, and the
 *    easiest one for the lesson itself to commit.
 *
 * 2. ANYTHING A FIGURE DRAWS GEOMETRICALLY IS COMPUTED HERE. A cutoff line, a
 *    slice membership, a bar length and a repair link are mathematical claims.
 *    Each geometry returns its own scale, ticks, inset and frame so the verifier
 *    asserts the same numbers the browser paints, rather than a second copy of
 *    the layout constants.
 *
 * 3. THE RULE DRAWN EQUALS THE RULE APPLIED. `inSlice` is the single definition
 *    of slice membership; the scatter's marks, the linked table and the graded
 *    comparison all read it. The verifier sweeps every cutoff the control admits
 *    -- 1,401 of them, on both sides -- and checks that the picture and the
 *    grading agree at every one, including the specimen that sits exactly on the
 *    default cutoff.
 *
 * Nothing here reaches for a number typed into prose: the browser data module is
 * regenerated from the served dataset by scripts/verify-endtoend-data.py.
 */
import { endToEndData } from './endtoend-data.js';

/* ============================================================ roles and scores */

/** The four things a number on this page can be evidence about. */
export const SCORE_ROLES = ['training', 'validation', 'selection', 'held-out'];

/** Short text for the badge beside a printed score. */
export const ROLE_LABELS = {
  training: 'training',
  validation: 'validation',
  selection: 'selection',
  'held-out': 'held-out',
};

export function scoreRecord(metric, value, role, note) {
  if (!SCORE_ROLES.includes(role)) {
    throw new RangeError(`"${role}" is not one of the four score roles: ${SCORE_ROLES.join(', ')}`);
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`a ${role} score must be a finite number, not ${value}`);
  }
  return note === undefined ? { metric, value, role } : { metric, value, role, note };
}

/** Re-badge a validation measurement as the criterion a choice was made with.
 *
 * The number does not change and neither does where it came from. What changes
 * is what it is being used for, and the lesson's whole argument is that those
 * are different claims. Keeping `from` makes that visible instead of hiding a
 * relabelling.
 */
export function asSelectionCriterion(record) {
  if (!record || record.role !== 'validation') {
    throw new RangeError('only a validation measurement can be read as a selection criterion');
  }
  return { metric: record.metric, value: record.value, role: 'selection', from: 'validation' };
}

/* ======================================================= metrics, from scratch */

const CLASSES = [0, 1, 2];

function requireSameLength(actual, predicted, label) {
  if (!Array.isArray(actual) || !Array.isArray(predicted) || actual.length !== predicted.length) {
    throw new RangeError(`${label}: needs two label lists of the same length`);
  }
  if (actual.length === 0) throw new RangeError(`${label}: an empty list has no score`);
}

export function confusionOf(actual, predicted) {
  requireSameLength(actual, predicted, 'confusionOf');
  return CLASSES.map(row => CLASSES.map(column =>
    actual.reduce((count, value, index) =>
      count + (value === row && predicted[index] === column ? 1 : 0), 0)));
}

export function accuracyOf(actual, predicted) {
  requireSameLength(actual, predicted, 'accuracyOf');
  return actual.reduce((count, value, index) => count + (value === predicted[index] ? 1 : 0), 0) / actual.length;
}

/** Mean of the per-class recalls. A class with no support is not averaged in,
 *  because a recall out of zero is not zero -- it does not exist. */
export function balancedAccuracyOf(actual, predicted) {
  requireSameLength(actual, predicted, 'balancedAccuracyOf');
  const recalls = [];
  for (const target of CLASSES) {
    const support = actual.reduce((count, value) => count + (value === target ? 1 : 0), 0);
    if (support === 0) continue;
    const right = actual.reduce((count, value, index) =>
      count + (value === target && predicted[index] === target ? 1 : 0), 0);
    recalls.push(right / support);
  }
  if (recalls.length === 0) throw new RangeError('balancedAccuracyOf: no class has any support');
  return recalls.reduce((sum, value) => sum + value, 0) / recalls.length;
}

/** Per-class recall with its own denominator kept beside it, because "three
 *  errors" means different things out of five and out of five hundred. */
export function classRecallTable(confusion) {
  return confusion.map((row, index) => {
    const support = row.reduce((sum, value) => sum + value, 0);
    return {
      actual: index,
      support,
      correct: row[index],
      errors: support - row[index],
      recall: support === 0 ? null : row[index] / support,
    };
  });
}

export function logLossOf(actual, probabilities) {
  requireSameLength(actual, probabilities, 'logLossOf');
  let total = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const probability = probabilities[index][actual[index]];
    if (!(probability > 0)) throw new RangeError(`logLossOf: probability ${probability} at row ${index}`);
    total += -Math.log(probability);
  }
  return total / actual.length;
}

/** What one prediction costs under log loss, in nats.
 *
 *  It lives here rather than in the lesson body because `Math` in that module is
 *  the KaTeX component: `Math.log` there resolves to a React component and
 *  yields undefined, silently. */
export function natsFor(probability) {
  if (!(probability > 0) || probability > 1) {
    throw new RangeError(`natsFor: ${probability} is not a probability above zero`);
  }
  return -Math.log(probability);
}

export function wilsonInterval(successes, trials, z) {
  if (!(trials > 0) || successes < 0 || successes > trials) {
    throw new RangeError(`wilsonInterval: ${successes} successes in ${trials} trials is not a proportion`);
  }
  const proportion = successes / trials;
  const centre = (proportion + (z * z) / (2 * trials)) / (1 + (z * z) / trials);
  const radius = (z * Math.sqrt((proportion * (1 - proportion)) / trials + (z * z) / (4 * trials * trials)))
    / (1 + (z * z) / trials);
  return { centre, radius, lower: centre - radius, upper: centre + radius };
}

/* =========================================================== the study's state */

export const candidateOrder = endToEndData.candidates.map(candidate => candidate.key);
export const eligibleByDeclaration = endToEndData.candidates
  .filter(candidate => candidate.declaredEligible).map(candidate => candidate.key);
export const candidateByKey = Object.fromEntries(
  endToEndData.candidates.map(candidate => [candidate.key, candidate]));
export const validationRows = endToEndData.validationRows;
export const validationActual = validationRows.map(row => row.actual);

/** Which way is better for each selection metric, and where the number lives. */
export const SELECTION_METRICS = [
  { key: 'validationBalancedAccuracy', label: 'Validation balanced accuracy', direction: 'higher',
    declared: true },
  { key: 'validationAccuracy', label: 'Validation accuracy', direction: 'higher', declared: false },
  { key: 'validationLogLoss', label: 'Validation log loss', direction: 'lower', declared: false },
];
export const selectionMetricByKey = Object.fromEntries(SELECTION_METRICS.map(metric => [metric.key, metric]));

/* ============================================ investigation D · freeze a choice */

/**
 * The best entry under a direction, with the tie rule written out.
 *
 * `entries` is `[key, value]` pairs ALREADY IN THE DECLARED ORDER. The winner is
 * the first entry achieving the extremum, so a tie resolves to the earlier
 * candidate in the protocol's own order rather than to whichever one a sort
 * happened to put first. Taking `sorted[0]` is not the same rule, and grading a
 * tie by that route has shipped as a defect in this repository.
 *
 * It is exported so the verifier can sweep it over score vectors that contain
 * real ties. No two candidates tie on this study's data, so a tie rule left
 * inside the reducer would be a rule nothing could exercise.
 */
export function bestByRule(entries, direction) {
  if (direction !== 'higher' && direction !== 'lower') {
    throw new RangeError(`"${direction}" is not a direction; use "higher" or "lower"`);
  }
  if (!Array.isArray(entries) || entries.length === 0) return null;
  let best = entries[0];
  for (const entry of entries.slice(1)) {
    const difference = entry[1] - best[1];
    if (direction === 'higher' ? difference > 0 : difference < 0) best = entry;
  }
  return best[0];
}

/**
 * Apply a stated selection rule to a stated set of candidates.
 *
 * The tie rule is the declared candidate order, exactly as the manuscript's
 * program states it, and it is applied by taking the FIRST candidate whose
 * score equals the best rather than by sorting. A sort is not the same rule:
 * `winners[0]` after a sort has graded a tie as a miss elsewhere in this
 * repository, because the sort was not stable in the order the protocol
 * declared.
 *
 * An empty eligible set has no winner. It is reported as having none rather
 * than defaulting to anything, because a default here would be a choice nobody
 * declared.
 */
export function selectionOutcome({ eligible, metricKey }) {
  const metric = selectionMetricByKey[metricKey];
  if (!metric) throw new RangeError(`"${metricKey}" is not one of this study's selection metrics`);
  if (!Array.isArray(eligible)) throw new RangeError('selectionOutcome: eligible must be a list of keys');
  const unknown = eligible.filter(key => !candidateByKey[key]);
  if (unknown.length) throw new RangeError(`selectionOutcome: unknown candidate ${unknown.join(', ')}`);

  const ordered = candidateOrder.filter(key => eligible.includes(key));
  if (ordered.length === 0) {
    return {
      winner: null, metric, ranking: [], tiedWith: [], eligible: ordered,
      departsFromProtocol: true,
      reason: 'No candidate is eligible, so there is nothing to choose between. A selection rule needs '
        + 'something to apply to.',
      heldOutAvailable: false,
      heldOutRefusal: 'There is no selection here, so there is nothing a held-out estimate could be an '
        + 'estimate of. A report needs a decision to report on.',
    };
  }
  const scoreOf = key => candidateByKey[key][metric.key].value;
  const best = bestByRule(ordered.map(key => [key, scoreOf(key)]), metric.direction);
  const tiedWith = ordered.filter(key => key !== best && scoreOf(key) === scoreOf(best));
  const ranking = [...ordered].sort((a, b) => (metric.direction === 'higher'
    ? scoreOf(b) - scoreOf(a) || candidateOrder.indexOf(a) - candidateOrder.indexOf(b)
    : scoreOf(a) - scoreOf(b) || candidateOrder.indexOf(a) - candidateOrder.indexOf(b)));

  const declaredSet = [...eligibleByDeclaration].sort().join(',');
  const chosenSet = [...ordered].sort().join(',');
  const departsFromProtocol = chosenSet !== declaredSet || !metric.declared;
  return {
    winner: best,
    metric,
    ranking,
    tiedWith,
    eligible: ordered,
    departsFromProtocol,
    reason: departsFromProtocol
      ? 'This is not the comparison this study declared before looking at the scores. Its result is an '
        + 'exploration of the development evidence, not the study\'s decision.'
      : 'This is the declared comparison: the three fitted candidates, ranked by validation balanced accuracy.',
    /* A held-out estimate exists for exactly one model, because exactly one
       model was ever evaluated on the test rows. Offering one for any other
       selection would mean manufacturing evidence this study does not have. */
    heldOutAvailable: best === endToEndData.heldOut.selected && !departsFromProtocol,
    heldOutRefusal: (() => {
      if (best === endToEndData.heldOut.selected && !departsFromProtocol) return null;
      if (best === endToEndData.heldOut.selected) {
        return 'The test rows were opened once, for the candidate the declared protocol selected. Reaching '
          + 'them again after changing the comparison would make them development data, and this page will '
          + 'not present them as an independent estimate of a choice made afterwards.';
      }
      return `No held-out evidence exists for ${candidateByKey[best].label}: this study evaluated only its `
        + 'declared selection on the test rows, and it will not invent a number for a different choice.';
    })(),
  };
}

/** Every setting the freeze investigation's controls admit: each subset of the
 *  four candidates against each of the three metrics. Small enough to sweep. */
export function everySelectionSetting() {
  const settings = [];
  for (let mask = 0; mask < 2 ** candidateOrder.length; mask += 1) {
    const eligible = candidateOrder.filter((_key, index) => (mask >> index) & 1);
    for (const metric of SELECTION_METRICS) settings.push({ eligible, metricKey: metric.key });
  }
  return settings;
}

/* ================================ investigation B · the development-error slice */

export const CUTOFF_MIN_HUNDREDTHS = 0;
export const CUTOFF_MAX_HUNDREDTHS = 1400;

/** The cutoff is held as an integer number of hundredths and turned into a
 *  number by one division. Accumulating .01 steps drifts; this does not, and it
 *  makes the browser and the verifier compare bit-identical operands. */
export function cutoffValue(cutoffHundredths) {
  if (!Number.isInteger(cutoffHundredths)
    || cutoffHundredths < CUTOFF_MIN_HUNDREDTHS || cutoffHundredths > CUTOFF_MAX_HUNDREDTHS) {
    throw new RangeError(`the colour-intensity cutoff runs from ${CUTOFF_MIN_HUNDREDTHS / 100} to `
      + `${CUTOFF_MAX_HUNDREDTHS / 100} in hundredths; ${cutoffHundredths} is outside it`);
  }
  return cutoffHundredths / 100;
}

/** THE definition of slice membership. The scatter's marks, the linked table and
 *  the graded comparison all call this one function, so the rule drawn is the
 *  rule applied by construction rather than by coincidence. */
export function inSlice(row, cutoffHundredths, side) {
  if (side !== 'lower' && side !== 'upper') {
    throw new RangeError(`"${side}" is not a slice side; use "lower" or "upper"`);
  }
  const cutoff = cutoffValue(cutoffHundredths);
  return side === 'lower' ? row.colorIntensity < cutoff : row.colorIntensity >= cutoff;
}

export function sliceOf(cutoffHundredths, side) {
  const rows = validationRows.filter(row => inSlice(row, cutoffHundredths, side));
  return {
    cutoff: cutoffValue(cutoffHundredths), side, rows, ids: rows.map(row => row.id), n: rows.length,
    description: side === 'lower'
      ? `colour intensity < ${cutoffValue(cutoffHundredths)}`
      : `colour intensity ≥ ${cutoffValue(cutoffHundredths)}`,
  };
}

function errorsIn(rows, modelKey) {
  if (!candidateByKey[modelKey]) throw new RangeError(`"${modelKey}" is not one of this study's candidates`);
  return rows.filter(row => row.prediction[modelKey] !== row.actual);
}

/**
 * Compare two candidates inside one slice of the development rows.
 *
 * The graded quantity is an integer difference of error counts, so there is no
 * tolerance and no rounding: `fewer`, `same` and `more` are exact. An empty
 * slice has no comparison at all -- not a zero difference, not a tie -- and is
 * reported as ungradable, because grading a superiority claim over no data
 * would be reporting a result nobody computed.
 *
 * Choosing the same candidate on both sides is a supported null case and gives
 * `same` with no changed specimen on every nonempty slice.
 */
export function sliceComparison({ reference, candidate, cutoffHundredths, side }) {
  const slice = sliceOf(cutoffHundredths, side);
  const referenceErrors = errorsIn(slice.rows, reference);
  const candidateErrors = errorsIn(slice.rows, candidate);
  const repaired = slice.rows.filter(row =>
    row.prediction[reference] !== row.actual && row.prediction[candidate] === row.actual);
  const broken = slice.rows.filter(row =>
    row.prediction[reference] === row.actual && row.prediction[candidate] !== row.actual);
  if (slice.n === 0) {
    return {
      slice, reference, candidate, outcome: null, difference: null,
      referenceErrors: null, candidateErrors: null, repairedIds: [], brokenIds: [],
      explain: '0 specimens — choose another cutoff. With nothing in the slice there is no error rate to '
        + 'report and no comparison to grade.',
    };
  }
  const difference = candidateErrors.length - referenceErrors.length;
  return {
    slice,
    reference,
    candidate,
    referenceErrors: referenceErrors.length,
    candidateErrors: candidateErrors.length,
    referenceErrorIds: referenceErrors.map(row => row.id),
    candidateErrorIds: candidateErrors.map(row => row.id),
    repairedIds: repaired.map(row => row.id),
    brokenIds: broken.map(row => row.id),
    difference,
    outcome: difference < 0 ? 'fewer' : difference > 0 ? 'more' : 'same',
    /* The identity that makes the count honest: a net change is repairs minus
       regressions, and equal counts need not mean the same specimens. */
    netIdentityHolds: difference === broken.length - repaired.length,
    explain: `${slice.n} specimen${slice.n === 1 ? '' : 's'} in ${slice.description}. `
      + `${candidateByKey[reference].label} gets ${referenceErrors.length} wrong, `
      + `${candidateByKey[candidate].label} gets ${candidateErrors.length} wrong.`,
  };
}

/** The paired change across all 36 development rows, which is what the strip
 *  figure draws and what the prose's "four repaired, one broken" names. */
export function pairedChange(reference, candidate) {
  const rows = validationRows.map(row => {
    const before = row.prediction[reference] === row.actual;
    const after = row.prediction[candidate] === row.actual;
    return {
      id: row.id,
      actual: row.actual,
      before,
      after,
      state: before === after ? (before ? 'right both times' : 'wrong both times')
        : (after ? 'repaired' : 'new error'),
    };
  });
  const repaired = rows.filter(row => row.state === 'repaired');
  const broken = rows.filter(row => row.state === 'new error');
  const referenceCorrect = rows.filter(row => row.before).length;
  const candidateCorrect = rows.filter(row => row.after).length;
  return {
    reference, candidate, rows, repaired, broken, referenceCorrect, candidateCorrect,
    net: candidateCorrect - referenceCorrect,
    /* Asserted rather than narrated: the aggregate move IS repairs minus
       regressions, and this is the identity the figure exists to make visible. */
    netIdentityHolds: candidateCorrect - referenceCorrect === repaired.length - broken.length,
    total: rows.length,
  };
}

/* ======================================= investigation C · acceptance and cost */

export const DEFERRAL_CASES = endToEndData.deferralFixture;
export const SCORE_MIN_HUNDREDTHS = 50;
export const SCORE_MAX_HUNDREDTHS = 100;
export const THRESHOLD_MIN_HUNDREDTHS = 50;
export const THRESHOLD_MAX_HUNDREDTHS = 101;

function requireInteger(value, min, max, label) {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${label} runs from ${min} to ${max} in hundredths; ${value} is outside it`);
  }
  return value / 100;
}

export function requireCost(value, max, label) {
  if (!Number.isFinite(value) || value < 0 || value > max) {
    throw new RangeError(`${label} runs from 0 to ${max}; ${value} is outside it`);
  }
  return value;
}

/**
 * The ledger for one acceptance rule over the ten constructed cases.
 *
 * Conditional error with nothing accepted is undefined, not zero and not one.
 * That is the null this fixture exists to expose, so it is returned as `null`
 * and the caller has to say so rather than print a flattering 0%.
 */
export function acceptanceLedger({ thresholdHundredths, wrongCost, deferCost, scoreHundredths }) {
  const threshold = requireInteger(thresholdHundredths, THRESHOLD_MIN_HUNDREDTHS, THRESHOLD_MAX_HUNDREDTHS,
    'the acceptance threshold');
  requireCost(wrongCost, 50, 'the cost of a wrong automatic answer');
  requireCost(deferCost, 20, 'the cost of a deferred case');
  const cases = DEFERRAL_CASES.map(item => {
    const override = scoreHundredths?.[item.id];
    const confidence = override === undefined
      ? item.confidence
      : requireInteger(override, SCORE_MIN_HUNDREDTHS, SCORE_MAX_HUNDREDTHS, `the score of case ${item.id}`);
    /* Editing a score changes the DECISION, never the recorded outcome. A
       scenario in which being more confident also makes you right is not a
       scenario about acceptance rules. */
    return { ...item, confidence, edited: override !== undefined && confidence !== item.confidence };
  });
  const accepted = cases.filter(item => item.confidence >= threshold);
  const deferred = cases.filter(item => item.confidence < threshold);
  const wrong = accepted.filter(item => !item.correct);
  return {
    threshold,
    wrongCost,
    deferCost,
    cases,
    acceptedIds: accepted.map(item => item.id),
    deferredIds: deferred.map(item => item.id),
    wrongIds: wrong.map(item => item.id),
    accepted: accepted.length,
    deferred: deferred.length,
    wrong: wrong.length,
    coverage: accepted.length / cases.length,
    conditionalError: accepted.length === 0 ? null : wrong.length / accepted.length,
    conditionalErrorNote: accepted.length === 0
      ? 'undefined (no answers) — with nothing answered there is no error rate among answers'
      : null,
    cost: wrongCost * wrong.length + deferCost * deferred.length,
  };
}

/** Compare a proposed rule with the active one. Costs are numeric, so the
 *  comparison carries a tolerance; counts are integers and do not. */
export function acceptanceComparison(baseline, proposed, tolerance = 1e-9) {
  const difference = proposed.cost - baseline.cost;
  return {
    difference,
    outcome: Math.abs(difference) <= tolerance ? 'same' : difference < 0 ? 'lower' : 'higher',
    baselineCost: baseline.cost,
    proposedCost: proposed.cost,
    coverageMoved: proposed.accepted !== baseline.accepted,
    explain: `${baseline.accepted} answered at ${baseline.threshold} against ${proposed.accepted} at `
      + `${proposed.threshold}; total cost ${baseline.cost} against ${proposed.cost}.`,
  };
}

/* ================================================================== geometry */

export function linearScale(domain, range) {
  const [fromLow, fromHigh] = domain;
  const [toLow, toHigh] = range;
  if (!(fromHigh > fromLow)) throw new RangeError(`linearScale: empty domain ${domain.join('..')}`);
  const scale = value => toLow + ((value - fromLow) / (fromHigh - fromLow)) * (toHigh - toLow);
  scale.domain = domain;
  scale.range = range;
  scale.invert = position => fromLow + ((position - toLow) / (toHigh - toLow)) * (fromHigh - fromLow);
  return scale;
}

/** Nothing is drawn nearer an edge than this, and the verifier asserts the floor
 *  BEFORE it compares any position against it. A containment check that reads
 *  the inset it is meant to be guarding passes at inset zero; that shipped. */
export const MINIMUM_INSET = 12;

/**
 * The development scatter: alcohol on x, colour intensity on y, with the cutoff
 * drawn as a horizontal line.
 *
 * Bounds are fixed across models so the same specimen sits in the same place
 * whichever candidate is selected. The model's input dimension changes; the
 * plotting coordinates do not. Flavanoids gets its own labelled strip instead of
 * moving a point.
 */
export function scatterGeometry({
  cutoffHundredths, side, reference, candidate, width = 320, height = 262,
  /* `top` is a BAND, not a margin: the y-axis title sits in it, anchored at the
     left edge, and a shallower one puts that title on top of the highest tick
     label. The browser found exactly that collision on the bar chart. */
  padding = { top: 30, right: 14, bottom: 30, left: 38 },
} = {}) {
  const xDomain = [11, 15];
  const yDomain = [0, 14];
  const x = linearScale(xDomain, [padding.left, width - padding.right]);
  const y = linearScale(yDomain, [height - padding.bottom, padding.top]);
  const cutoff = cutoffValue(cutoffHundredths);
  const points = validationRows.map(row => ({
    id: row.id,
    actual: row.actual,
    alcohol: row.alcohol,
    colorIntensity: row.colorIntensity,
    flavanoids: row.flavanoids,
    cx: x(row.alcohol),
    cy: y(row.colorIntensity),
    inSlice: inSlice(row, cutoffHundredths, side),
    referenceWrong: row.prediction[reference] !== row.actual,
    candidateWrong: row.prediction[candidate] !== row.actual,
  }));
  return {
    width, height, padding, xDomain, yDomain, inset: MINIMUM_INSET, x, y,
    /* Rendered wider than its viewBox. Two development specimens sit .05 apart
       in colour intensity, on opposite sides of the default cutoff, which is
       under a pixel at the viewBox's own width. This compact rendering does not
       reliably separate them, but the extra room stops the
       rest of the scatter being needlessly cramped. */
    maxWidth: 460,
    cutoff,
    cutoffY: y(cutoff),
    side,
    points,
    xTicks: [11, 12, 13, 14, 15].map(value => ({ value, x: x(value) })),
    yTicks: [0, 2, 4, 6, 8, 10, 12, 14].map(value => ({ value, y: y(value) })),
  };
}

/** Flavanoids as its own one-dimensional strip. The added measurement is what
 *  changes between the two linear candidates, so it gets a place to be seen
 *  without displacing a point whose coordinates must stay fixed. */
export function flavanoidStripGeometry({ selectedId = null, width = 320, height = 56, inset = 16 } = {}) {
  const values = validationRows.map(row => row.flavanoids);
  const domain = [0, Math.ceil(Math.max(...values))];
  const x = linearScale(domain, [inset, width - inset]);
  return {
    width, height, inset, domain, x,
    railY: 26,
    marks: validationRows.map(row => ({
      id: row.id, value: row.flavanoids, x: x(row.flavanoids), selected: row.id === selectedId,
    })),
    ticks: Array.from({ length: domain[1] + 1 }, (_unused, value) => ({ value, x: x(value) })),
  };
}

/**
 * Figure D: the 36 development specimens in identifier order, correct before
 * and after, linked only where the state changes.
 *
 * Category comes from identifier alignment, never from a probability rank: the
 * question is which specimen changed, and a rank would answer a different one.
 */
export function pairedStripGeometry({
  reference, candidate, width = 320, height = 120, inset = 16,
} = {}) {
  const change = pairedChange(reference, candidate);
  const usable = width - 2 * inset;
  const step = usable / change.total;
  const beforeY = 34;
  const afterY = 84;
  const markHeight = 14;
  const columns = change.rows.map((row, index) => ({
    ...row,
    x: inset + step * index + step / 2,
    left: inset + step * index + step * 0.12,
    barWidth: Math.max(step * 0.76, 1.5),
    beforeY,
    afterY,
    markHeight,
    changed: row.state === 'repaired' || row.state === 'new error',
  }));
  return {
    /* Thirty-six columns at the viewBox's own width give each specimen about
       seven pixels. The figure's whole point is that individual cases moved, so
       it is rendered wider. */
    maxWidth: 460,
    width, height, inset, step, beforeY, afterY, markHeight,
    rowLabelY: { before: beforeY - 6, after: afterY + markHeight + 11 },
    columns,
    change,
  };
}

/** A grouped bar for one metric across the candidates, with the baseline kept
 *  in the frame. Removing the baseline is how a comparison stops having an
 *  anchor, so it is included by construction rather than by choice. */
export function candidateBarGeometry({
  metricKey = 'validationBalancedAccuracy', width = 320, height = 198,
  /* `bottom` holds TWO rows: the category labels and, below them, the axis
     title. Sized for one, the title landed a few units under the nearest
     category label and read as a second line of it -- "linear_three /
     candidate". The remedy is the row's own measured space, not smaller text. */
  padding = { top: 30, right: 12, bottom: 60, left: 40 },
} = {}) {
  const metric = selectionMetricByKey[metricKey];
  if (!metric) throw new RangeError(`"${metricKey}" is not one of this study's selection metrics`);
  const values = candidateOrder.map(key => candidateByKey[key][metricKey].value);
  const top = metric.direction === 'higher' ? 1 : Math.ceil(Math.max(...values) * 10) / 10;
  const y = linearScale([0, top], [height - padding.bottom, padding.top]);
  const usable = width - padding.left - padding.right;
  const step = usable / candidateOrder.length;
  return {
    width, height, padding, metric, top, y, step, inset: MINIMUM_INSET,
    baselineIncluded: candidateOrder.includes('majority'),
    bars: candidateOrder.map((key, index) => {
      const value = candidateByKey[key][metricKey].value;
      return {
        key,
        label: candidateByKey[key].short,
        value,
        role: candidateByKey[key][metricKey].role,
        isBaseline: key === 'majority',
        x: padding.left + step * index + step * 0.18,
        barWidth: step * 0.64,
        y: y(value),
        barHeight: (height - padding.bottom) - y(value),
        labelX: padding.left + step * index + step / 2,
      };
    }),
    yTicks: [0, 0.25, 0.5, 0.75, 1].filter(value => value <= top).map(value => ({ value, y: y(value) })),
    /* A categorical axis has no quantitative ticks. It is returned as an empty
       list rather than omitted, so the frame's contract is the same for every
       geometry and a missing axis is a deliberate empty rather than an
       undefined nobody noticed. */
    xTicks: [],
  };
}

/**
 * Per-class recall for two candidates side by side.
 *
 * This figure exists to show the movement the aggregate hides: adding the
 * flavanoid measurement takes classes 1 and 2 to every specimen correct and
 * takes class 0 down. Showing only the mean would be choosing the cut that
 * flatters the result.
 */
export function recallComparisonGeometry({
  reference, candidate, width = 320, height = 204,
  padding = { top: 30, right: 12, bottom: 62, left: 40 },
} = {}) {
  const y = linearScale([0, 1], [height - padding.bottom, padding.top]);
  const usable = width - padding.left - padding.right;
  const groupStep = usable / CLASSES.length;
  const referenceTable = classRecallTable(candidateByKey[reference].validationConfusion);
  const candidateTable = classRecallTable(candidateByKey[candidate].validationConfusion);
  const groups = CLASSES.map(target => {
    const left = padding.left + groupStep * target;
    const pair = [
      { key: reference, ...referenceTable[target] },
      { key: candidate, ...candidateTable[target] },
    ].map((entry, position) => ({
      ...entry,
      x: left + groupStep * (0.16 + 0.34 * position),
      barWidth: groupStep * 0.3,
      y: y(entry.recall ?? 0),
      barHeight: (height - padding.bottom) - y(entry.recall ?? 0),
    }));
    return {
      actual: target,
      labelX: left + groupStep / 2,
      support: referenceTable[target].support,
      bars: pair,
      moved: referenceTable[target].recall !== candidateTable[target].recall,
      worsened: candidateTable[target].recall < referenceTable[target].recall,
    };
  });
  return {
    width, height, padding, y, groupStep, inset: MINIMUM_INSET, groups,
    yTicks: [0, 0.25, 0.5, 0.75, 1].map(value => ({ value, y: y(value) })),
    xTicks: [],
    worsenedClasses: groups.filter(group => group.worsened).map(group => group.actual),
  };
}

/**
 * Figure A: which specimens may reach which fitted object, and where each
 * label goes.
 *
 * Positions are computed rather than typed so the verifier can assert that no
 * arrow starts or ends outside its own lane, and that no arrow runs from a
 * label into a transform or from the test lane into the selection decision --
 * the two edges whose absence is the whole point of the picture.
 */
export function informationLaneGeometry({ width = 320, height = 212, inset = 14 } = {}) {
  /* Every string that reaches the drawing is SHORT, and the sentences that
     explain each lane live in the HTML table beside it.
     "Changes the model's fitted state" is thirty-one characters; inside a
     ninety-six-unit box at nine units of monospace it ran two thirds of the way
     across the figure and landed on top of the pipeline's own labels. SVG text
     does not wrap, so the remedy is fewer words in the drawing rather than
     smaller ones. */
  const laneHeight = 34;
  const laneGap = 12;
  const laneX = inset;
  const laneWidth = 84;
  const pipelineX = laneX + laneWidth + 22;
  const pipelineWidth = 104;
  const sinkWidth = 70;
  const sinkX = width - inset - sinkWidth;
  const lanes = [
    { key: 'train', title: 'Train', rows: endToEndData.contract.trainRows,
      allowed: 'Changes the model\'s fitted state' },
    { key: 'validation', title: 'Validation', rows: endToEndData.contract.validationRows,
      allowed: 'Changes the researcher\'s choice' },
    { key: 'test', title: 'Test', rows: endToEndData.contract.testRows,
      allowed: 'Estimates a choice already made' },
  ].map((lane, index) => {
    const y = inset + 26 + index * (laneHeight + laneGap);
    return {
      ...lane, x: laneX, y, width: laneWidth, height: laneHeight, midY: y + laneHeight / 2,
      labelY: y + 21,
      /* The only text drawn inside the lane. The rest is in the table. */
      label: `${lane.title} ${lane.rows}`,
    };
  });
  const pipeline = {
    x: pipelineX, y: lanes[0].y, width: pipelineWidth,
    height: lanes[2].y + laneHeight - lanes[0].y,
    midY: (lanes[0].y + lanes[2].y + laneHeight) / 2,
    centreX: pipelineX + pipelineWidth / 2,
    titleY: lanes[0].y - 5,
    parts: [
      { label: 'mean and scale', y: lanes[0].y + 26 },
      { label: 'coefficients', y: lanes[0].y + 50 },
      { label: 'transform, predict', y: lanes[0].y + 80 },
    ],
  };
  const sinks = [
    { key: 'selection', label: 'selection', y: lanes[1].midY - 12, height: 24, x: sinkX, width: sinkWidth },
    { key: 'report', label: 'final report', y: lanes[2].midY - 12, height: 24, x: sinkX, width: sinkWidth },
  ];
  /* `from` and `to` are named objects, not coordinates, so an assertion about
     which edges exist reads as a statement about the teaching claim. */
  const arrows = [
    { key: 'train-fit', from: 'train', to: 'pipeline', kind: 'fits',
      label: 'fits', x1: laneX + laneWidth, y1: lanes[0].midY, x2: pipelineX, y2: lanes[0].midY },
    { key: 'validation-apply', from: 'validation', to: 'pipeline', kind: 'applies',
      label: 'applies', x1: laneX + laneWidth, y1: lanes[1].midY, x2: pipelineX, y2: lanes[1].midY },
    { key: 'test-apply', from: 'test', to: 'pipeline', kind: 'applies',
      label: 'applies', x1: laneX + laneWidth, y1: lanes[2].midY, x2: pipelineX, y2: lanes[2].midY },
    { key: 'validation-selection', from: 'pipeline', to: 'selection', kind: 'informs',
      label: 'scores inform the choice',
      x1: pipelineX + pipelineWidth, y1: lanes[1].midY, x2: sinkX, y2: lanes[1].midY },
    { key: 'test-report', from: 'pipeline', to: 'report', kind: 'reports',
      label: 'reports once',
      x1: pipelineX + pipelineWidth, y1: lanes[2].midY, x2: sinkX, y2: lanes[2].midY },
    { key: 'leak', from: 'report', to: 'selection', kind: 'counterexample',
      label: 'reading this and changing the model is the mistake',
      x1: sinkX + sinkWidth / 2, y1: lanes[2].midY - 12,
      x2: sinkX + sinkWidth / 2, y2: lanes[1].midY + 12 },
  ];
  return {
    width, height, inset, lanes, pipeline, sinks, arrows,
    maxWidth: 460,
    gateOnTest: { x: laneX + laneWidth + 4, y: lanes[2].midY - 9, size: 18 },
    forbiddenEdges: [
      { from: 'label', to: 'pipeline', why: 'a target never reaches a transform' },
      { from: 'test', to: 'selection', why: 'the test lane never reaches the choice' },
    ],
  };
}

/** Investigation C: ten tiles on a score rail, routed to two queues. */
export function acceptanceRailGeometry({
  ledger, width = 320, height = 150, inset = 16,
} = {}) {
  const x = linearScale([0.5, 1], [inset + 10, width - inset - 10]);
  // Keep the confidence coordinate exact. Put nearby tiles on separate rows so
  // editing scores cannot hide a case behind another one, including exact ties.
  const rowEnds = [];
  const levels = new Map();
  [...ledger.cases].sort((a, b) => a.confidence - b.confidence || a.id - b.id).forEach(item => {
    const at = x(item.confidence);
    let level = rowEnds.findIndex(end => at - end >= 18);
    if (level < 0) level = rowEnds.length;
    rowEnds[level] = at;
    levels.set(item.id, level);
  });
  const extraHeight = Math.max(0, rowEnds.length - 1) * 18;
  const railY = 40 + extraHeight;
  /* The tiles occupy this band, and the threshold rule is drawn AROUND it rather
     than through it.
     A case sitting exactly at the threshold has its tile at exactly the
     threshold's x, so a single full-height rule runs straight down that tile's
     own number -- and the case on the boundary is the most interesting one on
     the rail. The tile rectangle is opaque and painted after the line, so the
     crossing is invisible today; that is paint order doing the work of layout,
     and it stops being true the moment the drawing order changes. Two segments
     put the rule at the true threshold and leave the label alone whatever the
     order. */
  const tileBand = { top: 20, bottom: railY - 6 };
  return {
    width, height: height + extraHeight, inset, x, railY, tileBand,
    thresholdX: x(Math.min(ledger.threshold, 1)),
    thresholdBeyondRail: ledger.threshold > 1,
    /* The lower segment stops just past the rail. Run any further and it reaches
       the axis tick label directly beneath it -- the threshold sits at a round
       score, which is exactly where a tick label is, so this is not an edge
       case but the common one. */
    thresholdSegments: [
      { y1: 4, y2: tileBand.top - 2 },
      { y1: tileBand.bottom + 2, y2: railY + 3 },
    ],
    tickLabelY: railY + 15,
    /* The y the tile's own number is drawn at. Exported so the component draws
       this value and the verifier asserts against this value -- rather than the
       verifier re-deriving the offset and agreeing with itself. */
    tileLabelY: railY - 9,
    ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1].map(value => ({ value, x: x(value) })),
    tiles: ledger.cases.map(item => ({
      id: item.id,
      confidence: item.confidence,
      correct: item.correct,
      edited: item.edited,
      accepted: ledger.acceptedIds.includes(item.id),
      x: x(item.confidence),
      y: railY - levels.get(item.id) * 18,
      labelY: railY - levels.get(item.id) * 18 - 9,
    })),
    queues: [
      { key: 'accepted', label: 'Automatic answer', ids: ledger.acceptedIds, y: 84 + extraHeight },
      { key: 'deferred', label: 'Deferred to a person', ids: ledger.deferredIds, y: 116 + extraHeight },
    ],
  };
}

/* ================================================================ formatting */

const minus = text => String(text).replace('-', '−');

/** The canonical form for a COMPUTED value: every decimal place, always. The
 *  browser verifier looks for exactly this form, so it must never be used for a
 *  value the learner typed. */
export const fixed = (value, digits = 6) =>
  (Number.isFinite(value) ? minus(value.toFixed(digits)) : '—');

/** A short form for an INPUT the learner set. Deliberately never six decimals. */
export const asInput = value => (Number.isInteger(value) ? String(value) : minus(String(value)));

export const countText = (part, whole) => `${part}/${whole}`;
