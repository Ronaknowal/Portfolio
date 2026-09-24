/** Pure models for ML Problem Formulation, Baselines & Data Leakage.
 *
 * Nothing in this file renders. Every quantity a figure draws and every
 * quantity an investigation grades is computed here, so one independent suite
 * (scripts/verify-formulation-models.mjs) can assert the same numbers and the
 * same coordinates the browser paints.
 *
 * Three rules shaped it.
 *
 *   1. THE RULE DRAWN IS THE RULE APPLIED. `recordEligibility` returns, for one
 *      record, the three separate predicates the manuscript states -- entity,
 *      event age, availability -- and `latestKnown` selects using nothing else.
 *      The timeline figure draws its markers from the same record objects. A
 *      figure that decided eligibility a second way could disagree with the
 *      grading beside it, which is exactly the defect class this topic is most
 *      exposed to.
 *
 *   2. AN INFORMATION CUTOFF IS PART OF EVERY SIGNATURE. There is no function
 *      here that takes a set of records and returns "the newest value". Every
 *      lookup takes a cutoff and a maximum age, because a lesson about leakage
 *      must not contain a convenience function that commits it.
 *
 *   3. AVERAGE PRECISION AND LOG LOSS ARE IMPLEMENTED, NOT IMPORTED. The
 *      recorded measurements come from scikit-learn; these implementations are
 *      written from the definitions so the verifier has a genuinely second
 *      route to them rather than a second call.
 */

/** A refusal, not a substitute. Returning a default for a bad argument is how a
 *  wrong number reaches a page with nothing marking it. */
function demand(condition, message) {
  if (!condition) throw new RangeError(message);
}

export function checkFinite(value, label) {
  demand(typeof value === 'number' && Number.isFinite(value), `${label} must be a finite number, got ${value}`);
  return value;
}

function checkInteger(value, label) {
  demand(Number.isInteger(value), `${label} must be an integer, got ${value}`);
  return value;
}

/* ============================================================ §3 three clocks
 *
 * Event time: when the measurement happened.
 * Available-at time: when the serving system could read this particular value.
 * Prediction cutoff: the moment we must reconstruct what was knowable.
 *
 * The whole point of the section is that these are three different numbers, so
 * they are three separate fields and three separate predicates. */

/** A stable identity for one historical record. Two versions of the same event
 *  are different records and must never overwrite one another, so the version
 *  is part of the key. */
export function recordKey(record) {
  return `${record.entity}·event${record.event}·v${record.version}`;
}

/** The readable identity, for the tables, the prediction options and the
 *  reveal — anywhere the text reflows. */
export function recordLabel(record) {
  return `${record.entity.replace('sensor_', '')} · event ${record.event} · v${record.version}`;
}

/** The COMPACT identity, for the drawing only.
 *
 * The readable form is sixteen characters, about 82 viewBox units at this
 * figure's text size, and it overflowed the lane gutter and ran under the
 * arrival marker of the record it names — the curve sampler caught the diamond
 * crossing its own label. SVG coordinates are for geometry and short labels;
 * the full identity is two lines below in a table that reflows. */
export function recordShortLabel(record) {
  return `${record.entity.replace('sensor_', '')}·e${record.event}·v${record.version}`;
}

/** Nominal width of one character of the smallest SVG text, in viewBox units.
 *  Measured against the real glyphs on the rendered page by the browser
 *  verifier; this is the offline proxy that keeps a label inside its gutter. */
export const LABEL_CHARACTER_WIDTH = 5.1;

export function validateRecord(record, label = 'record') {
  demand(typeof record === 'object' && record !== null, `${label} must be an object`);
  demand(typeof record.entity === 'string' && record.entity.length > 0, `${label}.entity must be a name`);
  checkInteger(record.event, `${label}.event`);
  checkInteger(record.available, `${label}.available`);
  checkInteger(record.version, `${label}.version`);
  checkFinite(record.value, `${label}.value`);
  demand(record.available >= record.event,
    `${label}: a value cannot become available before the event it measures (event ${record.event}, `
    + `available ${record.available})`);
  return record;
}

/**
 * The three predicates, each reported separately so the reveal can say WHICH
 * one rejected a record rather than only that it was rejected.
 *
 * Interval boundaries are inclusive on both sides, as the manuscript states:
 * `cutoff - maximumAge <= event <= cutoff` and `available <= cutoff`.
 */
export function recordEligibility(record, { entity, cutoff, maximumAge }) {
  validateRecord(record);
  checkInteger(cutoff, 'cutoff');
  checkInteger(maximumAge, 'maximumAge');
  demand(maximumAge >= 0, `maximumAge must not be negative, got ${maximumAge}`);
  const entityMatch = record.entity === entity;
  const notInFuture = record.event <= cutoff;
  const withinAge = record.event >= cutoff - maximumAge;
  const known = record.available <= cutoff;
  return {
    record,
    key: recordKey(record),
    entityMatch,
    notInFuture,
    withinAge,
    known,
    eligible: entityMatch && notInFuture && withinAge && known,
    /* Named reasons, in the order the manuscript applies them. A record can
       fail more than one; all failures are reported, because "it is too old"
       and "it has not arrived" are different lessons. */
    reasons: [
      entityMatch ? null : `belongs to ${record.entity}, not ${entity}`,
      notInFuture ? null : `event ${record.event} is after the cutoff ${cutoff}`,
      withinAge ? null : `event ${record.event} is older than the ${maximumAge}-unit age limit`,
      known ? null : `arrives at ${record.available}, after the cutoff ${cutoff}`,
    ].filter(Boolean),
  };
}

/**
 * The manuscript's "latest eligible calibration" policy, and nothing else.
 *
 * Among eligible records, choose the newest event; among that event's eligible
 * versions, the latest available; ties on both broken by the integer version.
 * Returns null when nothing qualifies -- a missing calibration is a result, not
 * an invitation to substitute a future value.
 */
export function latestKnown({ records, entity, cutoff, maximumAge }) {
  demand(Array.isArray(records) && records.length > 0, 'latestKnown needs at least one record');
  const assessed = records.map(record => recordEligibility(record, { entity, cutoff, maximumAge }));
  const eligible = assessed.filter(entry => entry.eligible);
  let selected = null;
  for (const entry of eligible) {
    if (selected === null) { selected = entry; continue; }
    const a = entry.record;
    const b = selected.record;
    const better = a.event !== b.event ? a.event > b.event
      : a.available !== b.available ? a.available > b.available
        : a.version > b.version;
    if (better) selected = entry;
  }
  return {
    assessed,
    eligible,
    rejected: assessed.filter(entry => !entry.eligible),
    selected,
    selectedKey: selected ? selected.key : null,
    value: selected ? selected.record.value : null,
    entity,
    cutoff,
    maximumAge,
  };
}

/* ------------------------------------------------- the timeline's geometry */

export function linearScale({ domain, range }) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  demand(d1 !== d0, 'a scale needs a domain of nonzero width');
  const scale = value => r0 + ((checkFinite(value, 'scale input') - d0) / (d1 - d0)) * (r1 - r0);
  scale.domain = domain;
  scale.range = range;
  scale.invert = position => d0 + ((position - r0) / (r1 - r0)) * (d1 - d0);
  return scale;
}

export const TIMELINE_DOMAIN = [0, 12];

/**
 * Lanes for the availability timeline.
 *
 * One lane per record, in the order given, each carrying an event marker, an
 * arrival marker and the segment joining them. The cutoff is a vertical line
 * across every lane; the admissible age window is a band behind them. Both are
 * clipped to the drawn domain for painting while their true values are
 * reported, because clamping a coordinate and then reporting the clamped number
 * is how a picture starts disagreeing with its own model.
 *
 * `labelRow` staggers coincident time labels onto a second row instead of
 * letting two three-character labels merge into one unreadable string.
 */
/** The nominal ascent of one line of this lesson's smallest SVG text, in
 *  viewBox units. A baseline closer to the top edge than this puts the glyphs
 *  outside the drawing, which is how the cutoff label left its own viewBox. */
export const LABEL_ASCENT = 9;

export function timelineGeometry({
  records, entity, cutoff, maximumAge,
  width = 300, laneHeight = 21, labelWidth = 58, inset = 14, topPadding = 26,
}) {
  demand(Array.isArray(records) && records.length > 0, 'timelineGeometry needs records');
  demand(labelWidth >= 40, `the lane label column is ${labelWidth} units, too narrow for an entity label`);
  demand(inset >= 10, `the timeline inset is ${inset}, below the 10 units the end tick labels need`);
  /* The cutoff label sits above the lanes, so the top padding has to hold a
     whole line of text. At 20 units it did not: the label's baseline landed at
     y=5 and its glyphs rose past the top edge of the viewBox. The browser
     layout inspector caught it; this floor is what stops it coming back. */
  demand(topPadding - 16 - LABEL_ASCENT >= 0,
    `the timeline's top padding is ${topPadding}, which leaves the cutoff label's glyphs `
    + `${LABEL_ASCENT - (topPadding - 16)} units above the top of the viewBox`);
  const [start, end] = TIMELINE_DOMAIN;
  const scale = linearScale({ domain: TIMELINE_DOMAIN, range: [labelWidth, width - inset] });
  const clamp = value => Math.min(Math.max(value, start), end);
  const cutoffLabelHalfWidth = ('cutoff'.length * LABEL_CHARACTER_WIDTH) / 2;
  const windowStart = cutoff - maximumAge;
  const lanes = records.map((record, index) => {
    validateRecord(record);
    const y = topPadding + index * laneHeight;
    const drawn = recordShortLabel(record);
    /* The gutter has to hold the label it is given, with a gap before the
       ruler starts. Without this the longest identity ran under the arrival
       marker of its own record. */
    demand(drawn.length * LABEL_CHARACTER_WIDTH + 4 <= labelWidth,
      `the drawn lane label "${drawn}" needs about `
      + `${(drawn.length * LABEL_CHARACTER_WIDTH + 4).toFixed(0)} units and the gutter is ${labelWidth}`);
    return {
      key: recordKey(record),
      label: recordLabel(record),
      drawnLabel: drawn,
      drawnLabelWidth: drawn.length * LABEL_CHARACTER_WIDTH,
      record,
      index,
      y,
      markY: y + laneHeight / 2,
      eventX: scale(clamp(record.event)),
      availableX: scale(clamp(record.available)),
      /* No per-record clipping flags. Two were computed here and read by
         nothing, and they could not have been true anyway: every time control
         is bounded to 0-12 and the drawn domain IS 0-12, so no event or
         arrival can fall outside it. Dead apparatus reads like coverage. The
         one clipping case that IS reachable -- an age window reaching back
         past the ruler -- is reported by `ageWindow.startsBeforeTheRuler`,
         which the figure now renders. */
      isOtherEntity: record.entity !== entity,
      delay: record.available - record.event,
    };
  });
  const ticks = Array.from({ length: end - start + 1 }, (_unused, step) => start + step)
    .filter(value => value % 2 === 0)
    .map(value => ({ value, x: scale(value) }));
  const height = topPadding + lanes.length * laneHeight + 26;
  return {
    width,
    height,
    inset,
    labelWidth,
    laneHeight,
    topPadding,
    scale,
    lanes,
    ticks,
    axisY: topPadding + lanes.length * laneHeight + 4,
    tickY: topPadding + lanes.length * laneHeight + 8,
    tickLabelY: topPadding + lanes.length * laneHeight + 19,
    cutoff: {
      value: cutoff,
      x: scale(clamp(cutoff)),
      /* Both drawn here rather than in the component, so the verifier asserts
         the same numbers the browser paints. */
      labelY: topPadding - 16,
      lineTopY: topPadding - 13,
      /* The label is CENTRED on the line, so at cutoff 12 its right half ran
         past the right edge of the viewBox and the browser clipped the last
         glyph. The line stays where the data puts it; only the label slides
         far enough to stay inside. */
      labelX: Math.min(
        Math.max(scale(clamp(cutoff)), cutoffLabelHalfWidth),
        width - cutoffLabelHalfWidth),
      labelHalfWidth: cutoffLabelHalfWidth,
    },
    ageWindow: {
      from: windowStart,
      to: cutoff,
      /* The extent actually drawn, which is clamped to the ruler. */
      drawnFrom: clamp(windowStart),
      x: scale(clamp(windowStart)),
      width: Math.max(scale(clamp(cutoff)) - scale(clamp(windowStart)), 0),
      /* An age limit wider than the drawn ruler reaches back past its left
         edge, and the band then understates the window. This flag was computed
         and read by nothing, so the caption could say "-10 to 2" beside a band
         that spanned 0 to 2 with nothing saying it had been truncated. The
         figure's extent IS its claim; the caption now reports the truncation
         and the verifier asserts that it does. */
      startsBeforeTheRuler: windowStart < start,
    },
    domain: TIMELINE_DOMAIN,
  };
}

/** Every pair of drawn lane labels or tick labels whose boxes would overlap.
 *  Zero is required; the browser verifier measures the real glyphs. */
export function timelineLabelCollisions(geometry, perCharacter = 4.8) {
  const boxes = geometry.ticks.map(tick => ({
    text: String(tick.value),
    left: tick.x - (String(tick.value).length * perCharacter) / 2,
    right: tick.x + (String(tick.value).length * perCharacter) / 2,
  }));
  const clashes = [];
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      if (boxes[i].right > boxes[j].left && boxes[j].right > boxes[i].left) {
        clashes.push([boxes[i].text, boxes[j].text]);
      }
    }
  }
  return clashes;
}

/* ================================================== §4-§5 baselines and scores
 *
 * Average precision and log loss are written out from their definitions. The
 * recorded page numbers came from scikit-learn; these are the second route. */

/**
 * Average precision, as the sum over distinct score thresholds of the recall
 * increment times the precision at that threshold.
 *
 * Tied scores are one threshold, which is the whole reason a constant score
 * yields exactly the positive fraction: there is one threshold, recall goes
 * from 0 to 1 in one step, and the precision there is the prevalence.
 */
export function averagePrecision(targets, scores) {
  demand(Array.isArray(targets) && Array.isArray(scores), 'averagePrecision needs two arrays');
  demand(targets.length === scores.length, 'targets and scores must have the same length');
  demand(targets.length > 0, 'averagePrecision needs at least one case');
  const positives = targets.reduce((sum, value) => sum + value, 0);
  demand(positives > 0, 'average precision is undefined when no case is positive');
  const order = scores.map((score, index) => index)
    .sort((left, right) => (scores[right] - scores[left]) || (left - right));
  let truePositives = 0;
  let seen = 0;
  let previousRecall = 0;
  let total = 0;
  let position = 0;
  while (position < order.length) {
    let last = position;
    while (last + 1 < order.length && scores[order[last + 1]] === scores[order[position]]) last += 1;
    for (let step = position; step <= last; step += 1) {
      truePositives += targets[order[step]];
      seen += 1;
    }
    const recall = truePositives / positives;
    const precision = truePositives / seen;
    total += (recall - previousRecall) * precision;
    previousRecall = recall;
    position = last + 1;
  }
  return total;
}

/** Mean negative log likelihood of the observed binary outcomes. */
export function logLoss(targets, probabilities) {
  demand(targets.length === probabilities.length && targets.length > 0, 'logLoss needs matching nonempty arrays');
  let total = 0;
  for (let index = 0; index < targets.length; index += 1) {
    const p = checkFinite(probabilities[index], `probability ${index}`);
    demand(p > 0 && p < 1, `log loss needs probabilities strictly inside (0, 1); got ${p} at ${index}`);
    total += targets[index] === 1 ? -Math.log(p) : -Math.log(1 - p);
  }
  return total / targets.length;
}

/** Correct decisions and the four confusion cells at a stated threshold.
 *  The comparison is `>= threshold`, as the displayed program writes it. */
export function confusionAtThreshold(targets, probabilities, threshold) {
  demand(targets.length === probabilities.length && targets.length > 0, 'confusion needs matching arrays');
  checkFinite(threshold, 'threshold');
  let trueNegative = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  let truePositive = 0;
  targets.forEach((target, index) => {
    const predicted = probabilities[index] >= threshold ? 1 : 0;
    if (target === 1 && predicted === 1) truePositive += 1;
    else if (target === 1) falseNegative += 1;
    else if (predicted === 1) falsePositive += 1;
    else trueNegative += 1;
  });
  return {
    threshold,
    matrix: [[trueNegative, falsePositive], [falseNegative, truePositive]],
    correct: trueNegative + truePositive,
    total: targets.length,
  };
}

/**
 * The lesson's one ranking rule: descending probability, then ascending source
 * row index. It is stated once here so the figure, the investigation, the
 * practice answer and the verifier cannot drift apart, and so that a tie is
 * resolved by a documented rule rather than by whatever order an array arrived
 * in.
 */
export function rankByScore({ ids, scores }) {
  demand(Array.isArray(ids) && Array.isArray(scores), 'rankByScore needs two arrays');
  demand(ids.length === scores.length && ids.length > 0, 'rankByScore needs matching nonempty arrays');
  demand(new Set(ids).size === ids.length, 'rankByScore needs distinct case identifiers');
  return ids.map((id, index) => index)
    .sort((left, right) => (scores[right] - scores[left]) || (ids[left] - ids[right]))
    .map(index => ids[index]);
}

/**
 * What a capacity does to a ranking: which identities are selected, how many
 * of them turned out positive, and the two fractions that answer different
 * questions.
 *
 * `order` is a full permutation of the evaluated identities. Capacity above
 * the evaluated set is refused rather than silently clipped, because a
 * precision denominator that is not the capacity asked for is a different
 * quantity from the one the label promises.
 */
export function selectionMetrics({ order, capacity, targetById, totalPositives }) {
  demand(Array.isArray(order) && order.length > 0, 'selectionMetrics needs an order');
  checkInteger(capacity, 'capacity');
  demand(capacity >= 1 && capacity <= order.length,
    `capacity ${capacity} is outside 1 to ${order.length}, the number of evaluated cases`);
  demand(new Set(order).size === order.length, 'the same case cannot appear twice in one action set');
  checkInteger(totalPositives, 'totalPositives');
  demand(totalPositives > 0, 'recall needs at least one positive outcome in the evaluated set');
  const selected = order.slice(0, capacity);
  selected.forEach(id => demand(Object.prototype.hasOwnProperty.call(targetById, id),
    `case ${id} has no recorded outcome; it is not in the evaluated set`));
  const found = selected.reduce((sum, id) => sum + targetById[id], 0);
  return {
    capacity,
    selected,
    positivesFound: found,
    precision: found / capacity,
    recall: found / totalPositives,
    totalPositives,
    missed: totalPositives - found,
  };
}

/** Replace one selected identity with one unselected identity, keeping the
 *  capacity and every other position fixed. This edits the action set, never a
 *  score, a label or a fitted model. */
export function swapSelection({ order, capacity, removeId, addId }) {
  demand(order.includes(removeId), `${removeId} is not in the ranking`);
  demand(order.includes(addId), `${addId} is not in the ranking`);
  demand(removeId !== addId, 'a swap needs two different cases');
  const removeAt = order.indexOf(removeId);
  const addAt = order.indexOf(addId);
  demand(removeAt < capacity, `${removeId} is not currently selected, so it cannot be swapped out`);
  demand(addAt >= capacity, `${addId} is already selected, so it cannot be swapped in`);
  const next = [...order];
  next[removeAt] = addId;
  next[addAt] = removeId;
  return next;
}

/** Reverse the selected block without changing which identities it contains.
 *  The membership null: every set metric must be unmoved by this. */
export function reverseSelected({ order, capacity }) {
  demand(capacity >= 1 && capacity <= order.length, 'capacity out of range');
  return [...order.slice(0, capacity).reverse(), ...order.slice(capacity)];
}

/**
 * The full permutation a policy state describes: the model's ranking, then each
 * recorded exchange in the order it was made, then an optional reversal of the
 * selected block.
 *
 * The whole permutation is rebuilt from the base ranking every time rather than
 * mutated in place, so the state that is graded is a pure function of the
 * inputs the learner committed. An edited order held in a mutable variable is
 * how a lab starts grading against a state nobody recorded.
 */
export function policyOrder({ baseOrder, capacity, swaps = [], reversed = false }) {
  let order = baseOrder;
  for (const [removeId, addId] of swaps) {
    order = swapSelection({ order, capacity, removeId, addId });
  }
  return reversed ? reverseSelected({ order, capacity }) : order;
}

/** Which identities entered and which left, between two action sets. */
export function membershipChange(beforeSelected, afterSelected) {
  const before = new Set(beforeSelected);
  const after = new Set(afterSelected);
  return {
    entered: afterSelected.filter(id => !before.has(id)),
    left: beforeSelected.filter(id => !after.has(id)),
    held: afterSelected.filter(id => before.has(id)).length,
  };
}

/**
 * The smallest allowance a graded numeric answer may carry, given the number of
 * decimals the page prints that answer to.
 *
 * A learner reads the value the page shows and types it back. Printing at `d`
 * decimals rounds, so the typed value can differ from the true one by up to
 * half a unit in the last printed place. An allowance tighter than that tells a
 * learner who copied the page's own number that their answer is outside it.
 *
 * This shipped. Investigation 2 graded precision at 1e-9 while printing it at
 * six decimals, so at capacity 3 the page rendered "You wrote 0.666667 …; the
 * calculation gives 0.666667, outside 1.000 × 10⁻⁹ of it" -- two identical
 * strings declared different -- and roughly four fifths of the capacities the
 * control reaches produce a fraction that does not terminate in six places.
 * It is the second time this exact defect has appeared in this effort, so the
 * fix is the property rather than the site: `Prediction` derives its allowance
 * from the precision it prints and can never grade tighter than that.
 */
export function displayTolerance(digits) {
  checkInteger(digits, 'digits');
  demand(digits >= 0 && digits <= 12, `digits must be between 0 and 12, got ${digits}`);
  return 0.5 * 10 ** -digits;
}

/** The allowance a grader actually applies: never tighter than the precision
 *  the page prints, however tight the lab asked for. */
export function gradingAllowance({ declared, digits }) {
  checkFinite(declared, 'declared tolerance');
  demand(declared >= 0, `a declared tolerance must not be negative, got ${declared}`);
  return Math.max(declared, displayTolerance(digits));
}

export const unchangedTolerance = 1e-12;

/** Which way a quantity moved, with an explicitly inclusive tolerance so that
 *  an exact null is reported as unchanged rather than as a tiny move. */
export function movementOf(before, after, tolerance = unchangedTolerance) {
  checkFinite(before, 'before');
  checkFinite(after, 'after');
  const difference = after - before;
  if (Math.abs(difference) <= tolerance) return { outcome: 'unchanged', difference: 0, rawDifference: difference };
  return { outcome: difference > 0 ? 'rises' : 'falls', difference, rawDifference: difference };
}

/* ====================================== §4 the cost threshold and its picture */

/** The probability above which acting has the smaller expected loss, under the
 *  manuscript's stated assumptions: zero cost for a correct action, no capacity
 *  coupling, and a probability valid for the case at hand. */
export function costThreshold(falsePositiveCost, falseNegativeCost) {
  checkFinite(falsePositiveCost, 'falsePositiveCost');
  checkFinite(falseNegativeCost, 'falseNegativeCost');
  demand(falsePositiveCost > 0 && falseNegativeCost > 0, 'both costs must be positive for this comparison');
  return falsePositiveCost / (falsePositiveCost + falseNegativeCost);
}

export function expectedLosses(probability, falsePositiveCost, falseNegativeCost) {
  checkFinite(probability, 'probability');
  demand(probability >= 0 && probability <= 1, `probability ${probability} is outside [0, 1]`);
  const acting = falsePositiveCost * (1 - probability);
  const waiting = falseNegativeCost * probability;
  return {
    probability,
    acting,
    waiting,
    preferred: acting < waiting ? 'act' : acting > waiting ? 'wait' : 'indifferent',
  };
}

/** Two straight lines and their crossing, drawn from the same expressions the
 *  prose derives. The crossing is placed by the closed form and then checked
 *  against the drawn lines, so a picture that disagreed with the formula would
 *  fail rather than mislead. */
export function thresholdPlotGeometry({
  falsePositiveCost, falseNegativeCost, width = 290, height = 150,
  padding = { top: 22, right: 12, bottom: 30, left: 34 },
}) {
  const threshold = costThreshold(falsePositiveCost, falseNegativeCost);
  const maximum = Math.max(falsePositiveCost, falseNegativeCost);
  const x = linearScale({ domain: [0, 1], range: [padding.left, width - padding.right] });
  const y = linearScale({ domain: [0, maximum], range: [height - padding.bottom, padding.top] });
  const actingLine = { from: { x: x(0), y: y(falsePositiveCost) }, to: { x: x(1), y: y(0) } };
  const waitingLine = { from: { x: x(0), y: y(0) }, to: { x: x(1), y: y(falseNegativeCost) } };
  const crossing = {
    probability: threshold,
    loss: falsePositiveCost * (1 - threshold),
    x: x(threshold),
    y: y(falsePositiveCost * (1 - threshold)),
  };
  return {
    width,
    height,
    padding,
    x,
    y,
    threshold,
    maximum,
    actingLine,
    waitingLine,
    crossing,
    falsePositiveCost,
    falseNegativeCost,
    xTicks: [0, 0.25, 0.5, 0.75, 1].map(value => ({ value, x: x(value) })),
    yTicks: [0, maximum / 2, maximum].map(value => ({ value, y: y(value) })),
  };
}

/* ========================================== §8 the newsvendor decision and F6 */

/** Expected cost of one stocking level against a two-point demand distribution
 *  with equal probabilities, under the manuscript's simplified cost model. */
export function stockingCost({ stock, demands, probabilities, underageCost, overageCost }) {
  demand(demands.length === probabilities.length, 'each demand needs a probability');
  const mass = probabilities.reduce((sum, value) => sum + value, 0);
  demand(Math.abs(mass - 1) < 1e-12, `the demand probabilities sum to ${mass}, not 1`);
  return demands.reduce((total, value, index) => {
    const shortfall = Math.max(value - stock, 0);
    const surplus = Math.max(stock - value, 0);
    return total + probabilities[index] * (underageCost * shortfall + overageCost * surplus);
  }, 0);
}

export function criticalQuantile(underageCost, overageCost) {
  demand(underageCost > 0 && overageCost > 0, 'both costs must be positive');
  return underageCost / (underageCost + overageCost);
}

/** The smallest demand level whose cumulative probability reaches the critical
 *  fraction: the discrete analogue of the quantile the derivation names. */
export function optimalStock({ demands, probabilities, underageCost, overageCost }) {
  const level = criticalQuantile(underageCost, overageCost);
  let cumulative = 0;
  for (let index = 0; index < demands.length; index += 1) {
    cumulative += probabilities[index];
    if (cumulative >= level - 1e-12) return { stock: demands[index], level, cumulative };
  }
  return { stock: demands[demands.length - 1], level, cumulative };
}

export const upliftGroups = [
  { id: 'A', calledProbability: 0.8, notCalledProbability: 0.75 },
  { id: 'B', calledProbability: 0.45, notCalledProbability: 0.1 },
];

/** Paired bars on one shared 0-1 axis, with the difference bracket between
 *  them. A ranking reversal is only visible if both bars share a scale, so the
 *  scale is built once and used for every bar. */
export function upliftGeometry({
  groups = upliftGroups, width = 290, barHeight = 13, rowGap = 34,
  padding = { top: 18, right: 46, bottom: 26, left: 30 },
}) {
  const scale = linearScale({ domain: [0, 1], range: [padding.left, width - padding.right] });
  const rows = groups.map((group, index) => {
    const top = padding.top + index * rowGap;
    const increment = group.calledProbability - group.notCalledProbability;
    return {
      id: group.id,
      top,
      called: {
        y: top,
        width: Math.max(scale(group.calledProbability) - scale(0), 0.6),
        endX: scale(group.calledProbability),
        value: group.calledProbability,
      },
      notCalled: {
        y: top + barHeight + 2,
        width: Math.max(scale(group.notCalledProbability) - scale(0), 0.6),
        endX: scale(group.notCalledProbability),
        value: group.notCalledProbability,
      },
      increment,
      bracket: {
        fromX: scale(Math.min(group.calledProbability, group.notCalledProbability)),
        toX: scale(Math.max(group.calledProbability, group.notCalledProbability)),
        y: top + barHeight + barHeight / 2 + 1,
      },
    };
  });
  const byCalled = [...groups].sort((left, right) => right.calledProbability - left.calledProbability)[0].id;
  const byIncrement = [...groups].sort((left, right) =>
    (right.calledProbability - right.notCalledProbability) - (left.calledProbability - left.notCalledProbability))[0].id;
  return {
    width,
    height: padding.top + groups.length * rowGap + padding.bottom,
    padding,
    barHeight,
    scale,
    rows,
    leaderByCalledProbability: byCalled,
    leaderByIncrement: byIncrement,
    /* Derived, never typed: if the two leaders ever stopped differing, the
       figure's whole claim would be false and the assertion would say so. */
    reverses: byCalled !== byIncrement,
    xTicks: [0, 0.25, 0.5, 0.75, 1].map(value => ({ value, x: scale(value) })),
  };
}

/* ======================================================= §1 the decision flow */

export const flowStages = [
  {
    id: 'records',
    title: 'available records',
    badge: 'as of the cutoff',
    detail: 'Everything the serving system could actually read by the moment the list had to be finalised.',
  },
  {
    id: 'prediction',
    title: 'prediction',
    badge: 'p = .32',
    detail: 'One estimated probability for one opportunity. The value shown is illustrative, chosen to make '
      + 'the next box do some work; it is not taken from the bank records.',
  },
  {
    id: 'rule',
    title: 'eligibility and capacity rule',
    badge: 'take the top 50',
    detail: 'The decision rule that turns estimates into an action set. p = .32 is selected or not depending '
      + 'on how the other eligible opportunities scored, so the estimate alone does not settle it.',
  },
  {
    id: 'action',
    title: 'action',
    badge: 'calls placed',
    detail: 'What actually happens to a customer. This is the object the business outcome is about.',
  },
  {
    id: 'outcome',
    title: 'observed outcome',
    badge: 'recorded later',
    detail: 'A recorded subscription, once the observation window has closed and the answer exists.',
  },
];

/**
 * Five stages in one column, each arrow from the bottom of one box to the top
 * of the next, plus the feedback arrow that returns MATURE outcomes to later
 * training.
 *
 * Stacked rather than laid out horizontally: five boxes with readable titles
 * across a 300-unit viewBox would give each one sixty units, and the manuscript
 * asks for the relationship rather than for nine contract fields squeezed into
 * a strip. The same drawing then needs no separate mobile composition.
 */
export function flowGeometry({
  stages = flowStages, width = 290, boxHeight = 34, gap = 20, padding = { top: 6, left: 8, right: 54 },
}) {
  const boxWidth = width - padding.left - padding.right;
  const boxes = stages.map((stage, index) => ({
    ...stage,
    index,
    x: padding.left,
    y: padding.top + index * (boxHeight + gap),
    width: boxWidth,
    height: boxHeight,
    centreX: padding.left + boxWidth / 2,
  }));
  const arrows = boxes.slice(0, -1).map((box, index) => ({
    from: { x: box.centreX, y: box.y + boxHeight },
    to: { x: box.centreX, y: boxes[index + 1].y },
    fromId: box.id,
    toId: boxes[index + 1].id,
  }));
  const first = boxes[0];
  const last = boxes[boxes.length - 1];
  const channel = padding.left + boxWidth + 20;
  const feedback = {
    fromId: last.id,
    toId: first.id,
    points: [
      { x: last.x + last.width, y: last.y + boxHeight / 2 },
      { x: channel, y: last.y + boxHeight / 2 },
      { x: channel, y: first.y + boxHeight / 2 },
      { x: first.x + first.width, y: first.y + boxHeight / 2 },
    ],
    note: 'only after the outcome exists',
  };
  return {
    width,
    height: padding.top + stages.length * boxHeight + (stages.length - 1) * gap + 8,
    padding,
    boxes,
    arrows,
    feedback,
    /* The claim the figure is making, stated as data so it can be asserted:
       no arrow runs from the outcome into a prediction for the SAME case. */
    arrowsFromOutcomeIntoPrediction: arrows.filter(arrow => arrow.fromId === 'outcome').length,
  };
}

/* ============================================ §2 one entity, many rows (F2) */

export const parcelFixture = {
  parcels: ['P1', 'P2', 'P3'],
  hours: [0, 1, 2],
};

/**
 * Nine observations in three identity lanes, allocated two ways.
 *
 * View A alternates rows between training and validation, so every parcel
 * identity appears on both sides. View B holds out whole identities. The
 * allocation is computed, and the overlap each one produces is derived from the
 * allocation rather than asserted in a caption.
 */
export function splitLaneGeometry({
  fixture = parcelFixture, width = 290, laneHeight = 26, cellGap = 6,
  padding = { top: 18, left: 34, right: 10, bottom: 8 },
}) {
  const { parcels, hours } = fixture;
  const cellWidth = (width - padding.left - padding.right - (hours.length - 1) * cellGap) / hours.length;
  const allocations = {
    rowAlternating: (parcelIndex, hourIndex) => ((parcelIndex + hourIndex) % 2 === 0 ? 'train' : 'validation'),
    byIdentity: parcelIndex => (parcelIndex < parcels.length - 1 ? 'train' : 'validation'),
  };
  const build = name => {
    const cells = [];
    parcels.forEach((parcel, parcelIndex) => {
      hours.forEach((hour, hourIndex) => {
        cells.push({
          id: `${parcel}-h${hour}`,
          parcel,
          hour,
          parcelIndex,
          hourIndex,
          side: allocations[name](parcelIndex, hourIndex),
          x: padding.left + hourIndex * (cellWidth + cellGap),
          y: padding.top + parcelIndex * laneHeight,
          width: cellWidth,
          height: laneHeight - cellGap,
        });
      });
    });
    const trainParcels = new Set(cells.filter(cell => cell.side === 'train').map(cell => cell.parcel));
    const validationParcels = new Set(cells.filter(cell => cell.side === 'validation').map(cell => cell.parcel));
    const shared = [...trainParcels].filter(parcel => validationParcels.has(parcel));
    return {
      name,
      cells,
      rows: cells.length,
      trainRows: cells.filter(cell => cell.side === 'train').length,
      validationRows: cells.filter(cell => cell.side === 'validation').length,
      distinctTrainParcels: trainParcels.size,
      distinctValidationParcels: validationParcels.size,
      sharedParcels: shared,
      /* The number the caption is about. Derived from the allocation, so a
         changed allocation changes the claim instead of contradicting it. */
      identityOverlap: shared.length,
    };
  };
  return {
    width,
    height: padding.top + parcels.length * laneHeight + padding.bottom,
    padding,
    cellWidth,
    laneHeight,
    parcels,
    hours,
    views: [build('rowAlternating'), build('byIdentity')],
  };
}

/* ====================================== §§5-6 the fitting boundary (F3) */

/**
 * Proportional bars for the nested partitions.
 *
 * `scaleTotal` is the row count that fills the whole width; `startRow` is where
 * this bar begins on that same scale. Both bars therefore share one scale, so
 * the second row visibly sits inside the development segment of the first
 * instead of being a second bar at a second scale that happens to look similar.
 */
export function partitionGeometry({ groups, scaleTotal, startRow = 0, width = 290, barHeight = 17 }) {
  demand(Array.isArray(groups) && groups.length > 0, 'a partition needs groups');
  checkInteger(scaleTotal, 'scaleTotal');
  checkInteger(startRow, 'startRow');
  const sum = groups.reduce((value, group) => value + group.rows, 0);
  demand(startRow + sum <= scaleTotal,
    `${startRow} + ${sum} rows overflow the ${scaleTotal}-row scale this bar is drawn on`);
  let cursor = startRow;
  const segments = groups.map(group => {
    checkInteger(group.rows, `${group.id}.rows`);
    const x = (cursor / scaleTotal) * width;
    cursor += group.rows;
    return {
      ...group,
      x,
      width: (group.rows / scaleTotal) * width,
      endX: (cursor / scaleTotal) * width,
      fraction: group.rows / scaleTotal,
    };
  });
  return { width, height: barHeight, barHeight, scaleTotal, startRow, rows: sum, segments };
}

/** The two bars of the fitting-boundary figure, on one shared scale. */
export function nestedPartitionGeometry({ partition, width = 290, barHeight = 17 }) {
  const { sourceRows, developmentRows, reservedRows, trainRows, validationRows } = partition;
  demand(developmentRows + reservedRows === sourceRows,
    `${developmentRows} development and ${reservedRows} reserved rows do not make ${sourceRows}`);
  demand(trainRows + validationRows === developmentRows,
    `${trainRows} training and ${validationRows} validation rows do not make ${developmentRows}`);
  return {
    width,
    barHeight,
    scaleTotal: sourceRows,
    first: partitionGeometry({
      scaleTotal: sourceRows,
      width,
      barHeight,
      groups: [
        { id: 'development', rows: developmentRows, label: 'development', scored: true },
        { id: 'reserved', rows: reservedRows, label: 'reserved', scored: false },
      ],
    }),
    second: partitionGeometry({
      scaleTotal: sourceRows,
      width,
      barHeight,
      groups: [
        { id: 'fit', rows: trainRows, label: 'fit', scored: true },
        { id: 'validation', rows: validationRows, label: 'validation', scored: true },
      ],
    }),
    /* The claim the figure makes, derived rather than captioned: exactly one
       segment receives no score. */
    unscoredSegments: ['reserved'],
  };
}

export const AVAILABILITY_FEATURES = [
  { id: 'customer', label: 'age, job, marital, education', side: 'before', note: 'recorded customer attributes' },
  { id: 'history', label: 'previous, poutcome, pdays', side: 'before', note: 'earlier campaign history' },
  { id: 'credit', label: 'default, housing, loan', side: 'before', note: 'recorded credit indicators' },
  { id: 'duration', label: 'duration', side: 'after', note: 'final length of the call being predicted' },
];

/**
 * Feature families placed relative to the call. Everything on the left of the
 * cutoff is a proposed pre-call input whose availability the manuscript says
 * must still be verified; `duration` sits on the right, after the call ends.
 */
export function availabilityAxisGeometry({
  families = AVAILABILITY_FEATURES, width = 290, rowHeight = 20,
  padding = { top: 20, left: 10, right: 10, bottom: 24 },
}) {
  const cutoffX = padding.left + (width - padding.left - padding.right) * 0.62;
  const rows = families.map((family, index) => {
    const y = padding.top + index * rowHeight;
    const before = family.side === 'before';
    return {
      ...family,
      index,
      y,
      markY: y + rowHeight / 2,
      /* Before-cutoff families end at the cutoff; the after-cutoff family
         starts there. The bar's extent is the claim. */
      x: before ? padding.left : cutoffX,
      width: before ? cutoffX - padding.left : width - padding.right - cutoffX,
      crossesCutoff: !before,
    };
  });
  return {
    width,
    height: padding.top + families.length * rowHeight + padding.bottom,
    padding,
    rowHeight,
    cutoffX,
    rows,
    beforeCount: rows.filter(row => row.side === 'before').length,
    afterCount: rows.filter(row => row.side === 'after').length,
  };
}

/* ================================================= §5 measured results (F4) */

/**
 * Horizontal bars on one axis for one metric at a time, with every metric
 * given its own domain and direction. Average precision and a count of correct
 * decisions never share a numeric axis here, because placing .46 beside 738
 * on one scale is a picture of nothing.
 */
export const metricAxes = {
  averagePrecision: { label: 'average precision', domain: [0, 0.5], better: 'higher', digits: 4 },
  logLoss: { label: 'log loss', domain: [0, 0.4], better: 'lower', digits: 4 },
  /* Named for what it DRAWS. It was labelled "positives among the selected 50"
     -- a count with a maximum of 50 -- while the axis ran 0 to 1 and the bars
     were annotated .12, .4 and .52. The same figure's table column showed the
     counts 6, 20 and 26 under the same words, so a reader switching metric saw
     6 redrawn as 0.12 under an identical name. The key is still precisionAt50
     and no number moved; only the label was wrong.
     `digits` is read by the figure, so changing it changes the drawing. */
  precisionAt50: {
    label: 'share of the selected 50 that subscribed', domain: [0, 1], better: 'higher', digits: 2,
  },
};

export function metricBarGeometry({
  rows, metric, width = 290, barHeight = 15, rowGap = 26,
  /* The left padding is a LABEL GUTTER, not a margin. Putting each bar's name
     above its bar instead ran the name across the grid lines and, at 320 px,
     into the tick label under it. A gutter costs width once; a collision costs
     a reader the figure. */
  padding = { top: 14, left: 62, right: 52, bottom: 24 },
}) {
  const axis = metricAxes[metric];
  demand(Boolean(axis), `no axis is declared for the metric ${metric}`);
  const scale = linearScale({ domain: axis.domain, range: [padding.left, width - padding.right] });
  const bars = rows.map((row, index) => {
    const value = checkFinite(row[metric], `${row.id}.${metric}`);
    demand(value >= axis.domain[0] && value <= axis.domain[1],
      `${row.id}'s ${metric} is ${value}, outside the drawn axis ${axis.domain[0]} to ${axis.domain[1]}`);
    const y = padding.top + index * rowGap;
    return {
      ...row,
      index,
      value,
      y,
      x: padding.left,
      width: Math.max(scale(value) - scale(axis.domain[0]), 0.6),
      endX: scale(value),
      labelX: padding.left - 5,
      labelY: y + barHeight - 3,
      valueX: scale(value) + 5,
    };
  });
  return {
    width,
    height: padding.top + rows.length * rowGap + padding.bottom,
    padding,
    barHeight,
    scale,
    axis,
    metric,
    bars,
    xTicks: [0, 0.25, 0.5, 0.75, 1]
      .map(fraction => axis.domain[0] + fraction * (axis.domain[1] - axis.domain[0]))
      .map(value => ({ value, x: scale(value) })),
  };
}

/** A strip of `capacity` cells, the first `positives` of them marked. These are
 *  aggregate counts and are labelled as such: the cells are not in ranked
 *  order and do not name individual cases. */
export function selectedStripGeometry({ capacity, positives, width = 290, cellHeight = 10, gap = 1.4 }) {
  checkInteger(capacity, 'capacity');
  checkInteger(positives, 'positives');
  demand(positives >= 0 && positives <= capacity, `${positives} positives cannot occur among ${capacity} cells`);
  const cellWidth = (width - (capacity - 1) * gap) / capacity;
  demand(cellWidth > 0.8, `${capacity} cells leave ${cellWidth.toFixed(2)} units each, too narrow to read`);
  return {
    width,
    height: cellHeight,
    cellHeight,
    capacity,
    positives,
    cells: Array.from({ length: capacity }, (_unused, index) => ({
      index,
      x: index * (cellWidth + gap),
      width: cellWidth,
      positive: index < positives,
    })),
  };
}

/* ============================================ §6 where a shortcut enters (F5) */

/* Node text is stored already split into short lines. A node is 86 units wide
   in the default layout, which is about fifteen monospace characters; a
   twenty-five character label placed as one string runs straight through its
   own box and across the arrow beside it. */
export const lineagePaths = [
  {
    id: 'target',
    title: 'the outcome reaches the feature',
    role: 'target and temporal leakage',
    nodes: [['recorded', 'outcome'], ['status or', 'aggregate field'], ['input used', 'at the cutoff']],
    repair: 'Define the feature at the actual cutoff, and recollect the as-known versions if the history was '
      + 'overwritten.',
    kind: 'temporal',
  },
  {
    id: 'preprocessing',
    title: 'the evaluation labels reach the transformation',
    role: 'preprocessing leakage',
    nodes: [['all labels,', 'both sides'], ['feature', 'selector'], ['fitted', 'representation']],
    repair: 'Fit selection, imputation and scaling inside each training fold, never on the whole table.',
    kind: 'preprocessing',
  },
  {
    id: 'selection',
    title: 'the score reaches the choice of procedure',
    role: 'selection leakage',
    nodes: [['held-out', 'score'], ['repeated', 'variant choice'], ['reported', '“final” score']],
    repair: 'Select on development evidence, then obtain a fresh evaluation for the procedure you selected.',
    kind: 'selection',
  },
];

/** Three rows of nodes and arrows. Node widths are computed from the row's own
 *  node count so a three-node path and a two-node path both fill the width
 *  rather than one of them running off the edge. */
export function lineageGeometry({
  paths = lineagePaths, width = 290, nodeHeight = 28, rowGap = 44, arrowGap = 14,
  padding = { top: 6, left: 4, right: 4 },
}) {
  const usable = width - padding.left - padding.right;
  const rows = paths.map((path, index) => {
    const count = path.nodes.length;
    const nodeWidth = (usable - (count - 1) * arrowGap) / count;
    demand(nodeWidth > 40, `a lineage node would be ${nodeWidth.toFixed(1)} units wide, too narrow for its label`);
    const longest = Math.max(...path.nodes.flat().map(line => line.length));
    demand(longest * 4.4 < nodeWidth,
      `the lineage label "${path.nodes.flat().find(line => line.length === longest)}" needs about `
      + `${(longest * 4.4).toFixed(0)} units and its node is ${nodeWidth.toFixed(0)}`);
    const y = padding.top + index * rowGap;
    const nodes = path.nodes.map((lines, position) => ({
      lines,
      position,
      x: padding.left + position * (nodeWidth + arrowGap),
      y,
      width: nodeWidth,
      height: nodeHeight,
      centreX: padding.left + position * (nodeWidth + arrowGap) + nodeWidth / 2,
    }));
    return {
      ...path,
      index,
      y,
      nodes,
      arrows: nodes.slice(0, -1).map((node, position) => ({
        from: { x: node.x + nodeWidth, y: y + nodeHeight / 2 },
        to: { x: nodes[position + 1].x, y: y + nodeHeight / 2 },
      })),
    };
  });
  return {
    width,
    height: padding.top + paths.length * rowGap,
    padding,
    nodeHeight,
    rows,
  };
}

/* ======================================================== investigation state */

export const timelineFixture = {
  records: [
    { entity: 'sensor_A', event: 1, available: 1, version: 1, value: 10 },
    { entity: 'sensor_A', event: 4, available: 8, version: 1, value: 20 },
    { entity: 'sensor_A', event: 1, available: 6, version: 2, value: 12 },
    { entity: 'sensor_B', event: 4, available: 4, version: 1, value: 99 },
  ],
  entity: 'sensor_A',
  cutoff: 5,
  maximumAge: 5,
};

/** The practice-1 history, which is a different fixture with a different
 *  answer. It lives here so the practice text and the verifier read the same
 *  numbers. */
export const practiceTimelineFixture = {
  records: [
    { entity: 'sensor_A', event: 2, available: 2, version: 1, value: 8 },
    { entity: 'sensor_A', event: 4, available: 7, version: 1, value: 11 },
    { entity: 'sensor_A', event: 2, available: 5, version: 2, value: 9 },
  ],
  entity: 'sensor_A',
  cutoff: 6,
  maximumAge: 5,
};

/** Move one record's arrival without touching its event time. The note this
 *  lesson answers asks for exactly this operation, so it is a named function
 *  rather than an inline object spread in a button handler. */
/**
 * Move one record's arrival without touching its event time.
 *
 * `entity` names the record's OWN entity and defaults to the fixture's queried
 * one. It used to have no parameter at all and matched on the queried entity
 * alone, so a preset named for sensor A's delayed record silently matched
 * sensor B's when the selector was on B — moving a record whose arrival was
 * already 4 and leaving A's at 8. The control carrying this section's whole
 * point became a no-op one click away. A mutation should name the record it
 * mutates, not the question being asked about it.
 */
export function withArrival(fixture, { entity = fixture.entity, event, version, available }) {
  const matches = record => record.entity === entity && record.event === event
    && record.version === version;
  demand(fixture.records.some(matches),
    `no record ${entity} event ${event} version ${version} exists to move`);
  return {
    ...fixture,
    records: fixture.records.map(record => (matches(record) ? { ...record, available } : record)),
  };
}

export function withValue(fixture, { entity, event, version, value }) {
  return {
    ...fixture,
    records: fixture.records.map(record => (record.entity === entity
      && record.event === event && record.version === version
      ? { ...record, value }
      : record)),
  };
}

/** Translate every value by a common amount. The selected identity must not
 *  move; only the selected number may. */
export function translateValues(fixture, amount) {
  return { ...fixture, records: fixture.records.map(record => ({ ...record, value: record.value + amount })) };
}

export const supportFixture = {
  requests: 12,
  urgent: 3,
  policies: [
    { id: 'A', escalated: 0, urgentEscalated: 0 },
    { id: 'B', escalated: 4, urgentEscalated: 3 },
  ],
  unnecessaryCost: 2,
  missedCost: 7,
  capacity: 3,
};

/** Practice 3, computed rather than stated: correct decisions and total cost
 *  for one escalation policy. */
export function policyOutcome({ requests, urgent, policy, unnecessaryCost, missedCost, capacity }) {
  checkInteger(policy.escalated, 'escalated');
  checkInteger(policy.urgentEscalated, 'urgentEscalated');
  demand(policy.urgentEscalated <= urgent, 'a policy cannot escalate more urgent requests than exist');
  demand(policy.escalated <= requests, 'a policy cannot escalate more requests than exist');
  demand(policy.urgentEscalated <= policy.escalated, 'escalated urgent requests are a subset of escalations');
  const unnecessary = policy.escalated - policy.urgentEscalated;
  const missed = urgent - policy.urgentEscalated;
  return {
    id: policy.id,
    escalated: policy.escalated,
    unnecessary,
    missed,
    correct: requests - unnecessary - missed,
    total: requests,
    cost: unnecessary * unnecessaryCost + missed * missedCost,
    withinCapacity: policy.escalated <= capacity,
    capacity,
  };
}

/** The prevalence example of §4: predicting the majority class everywhere. */
export function majorityBaseline({ negatives, positives }) {
  const total = negatives + positives;
  demand(total > 0, 'a prevalence needs at least one case');
  return {
    total,
    accuracy: negatives / total,
    positiveRecall: 0,
    prevalence: positives / total,
  };
}
