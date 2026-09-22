/* PAC learning and VC dimension: the whole computational surface of the lesson.
 *
 * Nothing in this file renders. Every number the page prints, every band a
 * figure shades and every coordinate a diagram draws is produced here, so the
 * verifier can assert the same values the browser paints. A drawn bound curve,
 * a shattering witness line and a confidence band are mathematical claims, and
 * a claim that only exists inside a component is a claim nobody checked.
 *
 * Three kinds of quantity live here and are kept apart on purpose, because the
 * lesson's whole argument depends on not confusing them:
 *
 *   * EXACT combinatorial or rational results -- pattern enumerations, the
 *     four-input occupancy probabilities, the binary-fraction sine witnesses.
 *     These use integer or BigInt arithmetic and have no tolerance.
 *   * COMPUTED THEOREM EXPRESSIONS -- the finite-family radius, the explicit VC
 *     radius, the classical realizable sufficient size, the two-strip bound.
 *     These are formulas evaluated at stated arguments, not measurements.
 *   * CONSTRUCTED TEACHING FIXTURES -- the twenty-five candidate error counts
 *     of figure 3, the four schematic draws of figure 1. These are declared as
 *     constructed wherever they appear and are never described as measured.
 *
 * Measured data (the banknote development curves) and the retained simulation
 * summaries live in `pac-data.js`, which is regenerated from the frozen packet
 * by `scripts/verify-pac-data.py`. They are deliberately not in this file.
 */

/* =============================================================== guard rails */

/** A refusal, not a substitute. Returning a default for a bad argument is how a
 *  wrong answer gets drawn with full confidence. */
function demand(condition, message) {
  if (!condition) throw new RangeError(message);
}

export function checkFinite(value, label) {
  demand(typeof value === 'number' && Number.isFinite(value), `${label} must be a finite number, got ${value}`);
  return value;
}

/* ================================================== exact rational arithmetic
 *
 * The four-input world's failure probabilities and the sine construction are
 * exact. A float would make 1/16777216 look like a rounding artefact and would
 * make the equality `sum of all masses = 1` a tolerance question. BigInt keeps
 * both exact at the sizes this lesson uses (denominators up to 4^24).
 */

function absBig(value) { return value < 0n ? -value : value; }

function gcdBig(a, b) {
  let left = absBig(a);
  let right = absBig(b);
  while (right) { const next = left % right; left = right; right = next; }
  return left;
}

/** An exact rational. `numerator` and `denominator` are BigInt and normalised. */
export function fraction(numerator, denominator = 1n) {
  const top = BigInt(numerator);
  const bottom = BigInt(denominator);
  demand(bottom !== 0n, 'a fraction cannot have denominator zero');
  const sign = bottom < 0n ? -1n : 1n;
  const divisor = gcdBig(top, bottom) || 1n;
  return { numerator: (sign * top) / divisor, denominator: (sign * bottom) / divisor };
}

export const fractionAdd = (a, b) =>
  fraction(a.numerator * b.denominator + b.numerator * a.denominator, a.denominator * b.denominator);
export const fractionMultiply = (a, b) => fraction(a.numerator * b.numerator, a.denominator * b.denominator);
/** Sign of a − b: −1, 0 or 1. */
export const fractionCompare = (a, b) => {
  const left = a.numerator * b.denominator;
  const right = b.numerator * a.denominator;
  return left < right ? -1 : left > right ? 1 : 0;
};
export const fractionToNumber = value => Number(value.numerator) / Number(value.denominator);
export const fractionText = value => (value.denominator === 1n
  ? String(value.numerator)
  : `${value.numerator}/${value.denominator}`);
export const ZERO = fraction(0n);
export const ONE = fraction(1n);

/* ==================================================== §5-§6 pattern counting */

/** Every label pattern one closed interval can realize on n ordered points,
 *  including the empty positive region. Enumerated, not counted by formula. */
export function intervalPatterns(n) {
  demand(Number.isInteger(n) && n >= 0 && n <= 16, `interval patterns need an integer 0 <= n <= 16, got ${n}`);
  const seen = new Map();
  const remember = pattern => seen.set(pattern.join(''), pattern);
  remember(Array.from({ length: n }, () => 0));
  for (let left = 0; left < n; left += 1) {
    for (let right = left; right < n; right += 1) {
      remember(Array.from({ length: n }, (_unused, index) => (index >= left && index <= right ? 1 : 0)));
    }
  }
  return [...seen.keys()].sort().map(key => seen.get(key));
}

/** Every pattern an increasing threshold h_a(x) = 1[x >= a] can realize. */
export function thresholdPatterns(n) {
  demand(Number.isInteger(n) && n >= 0 && n <= 16, `threshold patterns need an integer 0 <= n <= 16, got ${n}`);
  return Array.from({ length: n + 1 }, (_unused, boundary) =>
    Array.from({ length: n }, (_ignored, index) => (index >= boundary ? 1 : 0)));
}

/** The closed forms. Kept separate from the enumerations above so the two can
 *  be compared; they are not allowed to share an implementation. */
export function patternCountFormula(family, n) {
  demand(Number.isInteger(n) && n >= 0, `n must be a non-negative integer, got ${n}`);
  if (family === 'threshold') return n + 1;
  if (family === 'interval') return 1 + (n * (n + 1)) / 2;
  throw new RangeError(`unknown family ${family}`);
}

/** Binomial coefficient by exact integer arithmetic. */
export function binomial(n, k) {
  demand(Number.isInteger(n) && Number.isInteger(k) && n >= 0, `binomial needs non-negative integers, got ${n}, ${k}`);
  if (k < 0 || k > n) return 0;
  let value = 1n;
  const upper = BigInt(n);
  for (let step = 0n; step < BigInt(k); step += 1n) {
    value = (value * (upper - step)) / (step + 1n);
  }
  return Number(value);
}

/** Sauer's bound: the sum of binomials up to min(d, n). An upper bound on the
 *  growth function, never a claim that a class attains it. */
export function sauerSum(n, d) {
  demand(Number.isInteger(d) && d >= 0, `sauer needs an integer d >= 0, got ${d}`);
  let total = 0;
  for (let index = 0; index <= Math.min(d, n); index += 1) total += binomial(n, index);
  return total;
}

export function growthRow(n) {
  return {
    n,
    thresholds: thresholdPatterns(n).length,
    intervals: intervalPatterns(n).length,
    allBinary: 2 ** n,
    sauerD2: sauerSum(n, 2),
  };
}

export const growthTable = Array.from({ length: 10 }, (_unused, index) => growthRow(index + 1));

/* ============================================ §5 witnesses and obstructions */

/** Sort points by coordinate while keeping their identities. Coincident
 *  coordinates are refused here rather than resolved by an arbitrary tie rule:
 *  two points at the same place with different labels is a different lesson. */
export function sortWithIdentity(points) {
  demand(Array.isArray(points) && points.length >= 1, 'at least one point is needed');
  points.forEach((point, index) => checkFinite(point.x, `point ${index} coordinate`));
  const ordered = [...points].sort((left, right) => left.x - right.x);
  for (let index = 1; index < ordered.length; index += 1) {
    demand(ordered[index].x !== ordered[index - 1].x,
      `points ${ordered[index - 1].id} and ${ordered[index].id} share the coordinate ${ordered[index].x}`);
  }
  return ordered;
}

/**
 * Can the requested ID-attached labels be realized, and if not, where exactly
 * does the order forbid it?
 *
 * The verdict and the drawing both come from this one function: the figure
 * shades the run this returns and the lab grades the flag this returns, so the
 * rule drawn is the rule applied. `witness` is the actual rule, not a hint.
 */
export function requestVerdict({ points, family }) {
  demand(family === 'interval' || family === 'threshold', `unknown family ${family}`);
  const ordered = sortWithIdentity(points);
  ordered.forEach(point => demand(point.label === 0 || point.label === 1,
    `point ${point.id} must be labelled 0 or 1, got ${point.label}`));
  const pattern = ordered.map(point => point.label);
  const positives = ordered.filter(point => point.label === 1);
  const order = ordered.map(point => point.id);

  if (family === 'threshold') {
    // Feasible exactly when the ones form a suffix of the sorted order.
    let obstruction = null;
    for (let index = 0; index < ordered.length && !obstruction; index += 1) {
      if (ordered[index].label !== 1) continue;
      for (let later = index + 1; later < ordered.length && !obstruction; later += 1) {
        if (ordered[later].label === 0) obstruction = { positive: ordered[index].id, negative: ordered[later].id };
      }
    }
    if (obstruction) {
      return {
        family, order, pattern, feasible: false, witness: null, obstruction,
        explain: `${obstruction.positive} is positive and sits left of the negative ${obstruction.negative}. `
          + 'An increasing threshold labels a suffix positive, so nothing negative can lie to the right of a positive.',
      };
    }
    const firstPositive = ordered.findIndex(point => point.label === 1);
    const threshold = firstPositive === -1
      ? ordered[ordered.length - 1].x + 1
      : ordered[firstPositive].x;
    return {
      family, order, pattern, feasible: true, obstruction: null,
      witness: { kind: 'threshold', threshold, empty: firstPositive === -1 },
      explain: firstPositive === -1
        ? `A threshold above every point, here ${threshold}, labels them all negative.`
        : `The threshold ${threshold} sits at ${ordered[firstPositive].id}, the leftmost positive point.`,
    };
  }

  if (!positives.length) {
    return {
      family, order, pattern, feasible: true, obstruction: null,
      witness: { kind: 'interval', interval: null, empty: true },
      explain: 'The empty positive region is a member of this class, and it labels every point negative.',
    };
  }
  const left = positives[0].x;
  const right = positives[positives.length - 1].x;
  const intruder = ordered.find(point => point.label === 0 && point.x > left && point.x < right);
  if (intruder) {
    return {
      family, order, pattern, feasible: false, witness: null,
      obstruction: { negative: intruder.id, betweenLeft: positives[0].id, betweenRight: positives[positives.length - 1].id },
      explain: `Any interval containing ${positives[0].id} and ${positives[positives.length - 1].id} also contains `
        + `everything between them, and ${intruder.id} lies between them with label 0. `
        + 'The positive labels would have to form one unbroken run in coordinate order.',
    };
  }
  return {
    family, order, pattern, feasible: true, obstruction: null,
    witness: { kind: 'interval', interval: [left, right], empty: false },
    explain: `The tight interval [${left}, ${right}] runs from ${positives[0].id} to `
      + `${positives[positives.length - 1].id} and contains no negative point.`,
  };
}

/** What the witness predicts at each point, so the reveal can show predicted
 *  against requested rather than asserting agreement. */
export function witnessPredictions(witness, points) {
  if (!witness) return null;
  if (witness.kind === 'threshold') {
    return points.map(point => ({ id: point.id, predicted: point.x >= witness.threshold ? 1 : 0 }));
  }
  if (witness.empty || witness.interval === null) return points.map(point => ({ id: point.id, predicted: 0 }));
  const [left, right] = witness.interval;
  return points.map(point => ({ id: point.id, predicted: point.x >= left && point.x <= right ? 1 : 0 }));
}

/* ============================================================ §3-§7 bounds */

/** Two-sided uniform radius over K rules fixed before the evaluation sample.
 *  Hoeffding at δ/K each, then a union bound. K = 1 is the single-rule case. */
export function finiteRadius(k, n, delta) {
  demand(Number.isInteger(k) && k >= 1, `K must be an integer at least 1, got ${k}`);
  demand(Number.isInteger(n) && n >= 1, `n must be an integer at least 1, got ${n}`);
  demand(Number.isFinite(delta) && delta > 0 && delta < 1, `delta must lie in (0, 1), got ${delta}`);
  return Math.sqrt(Math.log((2 * k) / delta) / (2 * n));
}

/** The conservative explicit uniform-convergence radius this lesson names, with
 *  its stated domain n >= d >= 1. Other constants exist; this is the one the
 *  page's curves and the program both use, so the two cannot disagree. */
export function vcRadius(d, n, delta) {
  demand(Number.isInteger(d) && d >= 1, `this displayed form needs an integer d >= 1, got ${d}`);
  demand(Number.isInteger(n) && n >= d, `this displayed form needs n >= d, got n = ${n} and d = ${d}`);
  demand(Number.isFinite(delta) && delta > 0 && delta < 1, `delta must lie in (0, 1), got ${delta}`);
  return Math.sqrt((32 * (d * Math.log((Math.E * n) / d) + Math.log(8 / delta))) / n);
}

/** Blumer, Ehrenfeucht, Haussler and Warmuth's classical sufficient size for
 *  the realizable case. Its logarithms are base 2, as in that paper. */
export function realizableVcSampleBound(d, epsilon, delta) {
  demand(Number.isInteger(d) && d >= 1, `d must be an integer at least 1, got ${d}`);
  demand(Number.isFinite(epsilon) && epsilon > 0 && epsilon < 1, `epsilon must lie in (0, 1), got ${epsilon}`);
  demand(Number.isFinite(delta) && delta > 0 && delta < 1, `delta must lie in (0, 1), got ${delta}`);
  return Math.ceil(Math.max((4 / epsilon) * Math.log2(2 / delta), ((8 * d) / epsilon) * Math.log2(13 / epsilon)));
}

/** The finite realizable condition n >= (ln K + ln(1/delta)) / epsilon. */
export function finiteRealizableSampleBound(k, epsilon, delta) {
  demand(Number.isInteger(k) && k >= 1, `K must be an integer at least 1, got ${k}`);
  demand(Number.isFinite(epsilon) && epsilon > 0 && epsilon < 1, `epsilon must lie in (0, 1), got ${epsilon}`);
  demand(Number.isFinite(delta) && delta > 0 && delta < 1, `delta must lie in (0, 1), got ${delta}`);
  return Math.ceil((Math.log(k) + Math.log(1 / delta)) / epsilon);
}

/** K e^{-n epsilon}: the union-bound failure probability for a consistent
 *  learner over K predeclared rules. `raw` is kept alongside `clipped` because
 *  a value above 1 is the interesting case -- the bound is valid and says
 *  nothing, which is different from the bound being wrong. */
export function finiteClassFailureBound(k, n, epsilon) {
  demand(Number.isInteger(k) && k >= 1, `K must be an integer at least 1, got ${k}`);
  demand(Number.isInteger(n) && n >= 0, `n must be a non-negative integer, got ${n}`);
  demand(Number.isFinite(epsilon) && epsilon > 0 && epsilon < 1, `epsilon must lie in (0, 1), got ${epsilon}`);
  const raw = k * Math.exp(-n * epsilon);
  return { raw, clipped: Math.min(1, raw), vacuous: raw >= 1 };
}

/** The algorithm-specific two-strip bound for the tight interval learner on a
 *  uniform input distribution. Sufficient, not necessary. */
export function twoStripBound(epsilon, n) {
  demand(Number.isFinite(epsilon) && epsilon > 0 && epsilon < 1, `epsilon must lie in (0, 1), got ${epsilon}`);
  demand(Number.isInteger(n) && n >= 0, `n must be a non-negative integer, got ${n}`);
  const raw = 2 * (1 - epsilon / 2) ** n;
  return { raw, clipped: Math.min(1, raw), vacuous: raw >= 1 };
}

/* =========================================== §3 the exact four-input world */

export const finiteWorldInputs = [0, 1, 2, 3];

/** All sixteen binary rules on the four inputs, in lexicographic order. */
export const finiteWorldHypotheses = Array.from({ length: 16 }, (_unused, code) =>
  [0, 1, 2, 3].map(index => (code >> (3 - index)) & 1));

const asMask = sample => sample.reduce((mask, point) => mask | (1 << point), 0);

/** The lexicographically first rule consistent with the observed labels.
 *
 *  Implemented by walking the sixteen rules in order, which is the definition.
 *  The closed form -- observed bits from the target, zero elsewhere -- is a
 *  separate claim, checked against this by the verifier rather than assumed. */
export function selectedHypothesis(target, sample) {
  demand(Array.isArray(target) && target.length === 4, 'the target is four bits');
  target.forEach(bit => demand(bit === 0 || bit === 1, `target bits are 0 or 1, got ${bit}`));
  sample.forEach(point => demand(Number.isInteger(point) && point >= 0 && point <= 3,
    `observations name an input 0 to 3, got ${point}`));
  const mask = asMask(sample);
  const found = finiteWorldHypotheses.find(rule =>
    [0, 1, 2, 3].every(index => !(mask & (1 << index)) || rule[index] === target[index]));
  demand(found, 'the target itself is always consistent, so a consistent rule must exist');
  return found;
}

export function finiteWorldRisk(rule, target) {
  const wrong = [0, 1, 2, 3].filter(index => rule[index] !== target[index]);
  return { wrong, risk: wrong.length / 4, exact: fraction(BigInt(wrong.length), 4n) };
}

/** One run of the finite world: what was observed, what was selected, what it
 *  costs, and whether it met the target. Strict `>` is failure, so risk exactly
 *  equal to epsilon succeeds; that boundary is the point of the epsilon = .25
 *  fixture and is never softened by a tolerance. */
export function finiteWorldRun({ target, sample, epsilon }) {
  demand(Number.isFinite(epsilon) && epsilon > 0 && epsilon <= 1, `epsilon must lie in (0, 1], got ${epsilon}`);
  const rule = selectedHypothesis(target, sample);
  const { wrong, risk, exact } = finiteWorldRisk(rule, target);
  const seen = [...new Set(sample)].sort((left, right) => left - right);
  return {
    target: [...target],
    sample: [...sample],
    seen,
    unseen: finiteWorldInputs.filter(point => !seen.includes(point)),
    rule,
    ruleText: rule.join(''),
    wrong,
    risk,
    riskExact: exact,
    empiricalRisk: 0,
    epsilon,
    meetsTarget: risk <= epsilon,
    repeatedObservations: sample.length - seen.length,
  };
}

/**
 * Exact failure probability after n independent draws.
 *
 * Occupancy dynamic programme over the sixteen "which inputs have been seen"
 * masks, in exact rational arithmetic: mass starts on the empty mask and each
 * draw sends a quarter of every mask's mass to the mask with one more bit set.
 * The masses sum to exactly one, and that is asserted rather than assumed.
 */
export function finiteWorldProbability({ target, n, epsilon }) {
  demand(Number.isInteger(n) && n >= 0 && n <= 40, `n must be an integer 0 to 40, got ${n}`);
  demand(Number.isFinite(epsilon) && epsilon > 0 && epsilon <= 1, `epsilon must lie in (0, 1], got ${epsilon}`);
  let masses = Array.from({ length: 16 }, () => ZERO);
  masses[0] = ONE;
  const quarter = fraction(1n, 4n);
  for (let step = 0; step < n; step += 1) {
    const next = Array.from({ length: 16 }, () => ZERO);
    masses.forEach((mass, mask) => {
      if (mass.numerator === 0n) return;
      for (let point = 0; point < 4; point += 1) {
        const target2 = mask | (1 << point);
        next[target2] = fractionAdd(next[target2], fractionMultiply(mass, quarter));
      }
    });
    masses = next;
  }
  const states = [];
  let failure = ZERO;
  let total = ZERO;
  masses.forEach((mass, mask) => {
    total = fractionAdd(total, mass);
    const rule = [0, 1, 2, 3].map(index => ((mask & (1 << index)) ? target[index] : 0));
    const { risk, exact } = finiteWorldRisk(rule, target);
    if (risk > epsilon) failure = fractionAdd(failure, mass);
    if (mass.numerator !== 0n) {
      states.push({
        seen: [0, 1, 2, 3].filter(index => mask & (1 << index)),
        probability: fractionText(mass),
        probabilityValue: fractionToNumber(mass),
        rule,
        ruleText: rule.join(''),
        risk,
        riskExact: fractionText(exact),
        fails: risk > epsilon,
      });
    }
  });
  demand(fractionCompare(total, ONE) === 0, `the occupancy masses sum to ${fractionText(total)}, not 1`);
  const bound = finiteClassFailureBound(16, n, epsilon);
  return {
    n,
    epsilon,
    target: [...target],
    failureExact: fractionText(failure),
    failure: fractionToNumber(failure),
    states,
    bound,
  };
}

/**
 * Figure 2: which candidate rules survive each observation.
 *
 * Every one of the sixteen rules is listed with its own error region against
 * the target, and each observation crosses out the rules that disagree with the
 * label it revealed. The selected rule is the first survivor in lexicographic
 * order -- which is only meaningful once the whole surviving set is visible, so
 * the figure marks it as a consequence of the elimination rather than as a
 * separate rule of its own.
 */
export function eliminationTrace({ target, sample }) {
  demand(Array.isArray(sample), 'an observation sequence is required');
  const steps = [];
  for (let taken = 0; taken <= sample.length; taken += 1) {
    const observed = sample.slice(0, taken);
    const mask = asMask(observed);
    const rules = finiteWorldHypotheses.map(rule => {
      const consistent = [0, 1, 2, 3].every(index => !(mask & (1 << index)) || rule[index] === target[index]);
      const { wrong, risk } = finiteWorldRisk(rule, target);
      return { rule, ruleText: rule.join(''), consistent, wrong, risk };
    });
    const survivors = rules.filter(entry => entry.consistent);
    demand(survivors.length >= 1, 'the target is always consistent, so at least one rule must survive');
    steps.push({
      taken,
      observed,
      lastObservation: taken ? sample[taken - 1] : null,
      rules,
      survivors: survivors.length,
      selected: survivors[0],
      badSurvivors: survivors.filter(entry => entry.risk > 0.25).length,
    });
  }
  return { target: [...target], sample: [...sample], steps };
}

/* ================================== §8 the interval world with a known risk */

/** Labels generated by the target interval. Closed on both sides. */
export function labelsFromTarget(points, target) {
  const [a, b] = target;
  checkFinite(a, 'target left endpoint');
  checkFinite(b, 'target right endpoint');
  demand(a <= b, `the target interval needs a <= b, got [${a}, ${b}]`);
  return points.map(x => (x >= a && x <= b ? 1 : 0));
}

/** The smallest closed interval containing every observed positive input, or
 *  the empty positive region when no positive was observed. The empty default
 *  matters: inventing a central interval would not be consistent with an
 *  all-negative sample. */
export function fitTightInterval(points, labels) {
  demand(points.length === labels.length, 'points and labels must have the same length');
  points.forEach((x, index) => checkFinite(x, `sample point ${index}`));
  const positives = points.filter((_x, index) => labels[index] === 1);
  if (!positives.length) return { interval: null, empty: true };
  return { interval: [Math.min(...positives), Math.max(...positives)], empty: false };
}

/** Population risk under a uniform input distribution on [0, 1]: the length of
 *  the symmetric difference. Length equals probability only because the input
 *  distribution is uniform; that condition travels with the figure. */
export function intervalRisk(interval, target) {
  const [a, b] = target;
  demand(a <= b, `the target interval needs a <= b, got [${a}, ${b}]`);
  if (interval === null) return b - a;
  const [left, right] = interval;
  demand(left <= right, `a fitted interval needs left <= right, got [${left}, ${right}]`);
  const intersection = Math.max(0, Math.min(b, right) - Math.max(a, left));
  return (b - a) + (right - left) - 2 * intersection;
}

/** The pieces of the target the fit misses and the pieces it invents, as drawn
 *  segments. Their total length is the risk, and that identity is asserted. */
export function riskSegments(interval, target) {
  const [a, b] = target;
  const segments = [];
  if (interval === null) {
    if (b > a) segments.push({ kind: 'missed', from: a, to: b });
  } else {
    const [left, right] = interval;
    const low = Math.max(a, left);
    const high = Math.min(b, right);
    if (high > low) {
      if (a < low) segments.push({ kind: 'missed', from: a, to: low });
      if (high < b) segments.push({ kind: 'missed', from: high, to: b });
      if (left < a) segments.push({ kind: 'invented', from: left, to: a });
      if (b < right) segments.push({ kind: 'invented', from: b, to: right });
    } else {
      if (b > a) segments.push({ kind: 'missed', from: a, to: b });
      if (right > left) segments.push({ kind: 'invented', from: left, to: right });
    }
  }
  return segments;
}

/** One deterministic experiment in the interval world. */
export function intervalExperiment({ points, target, epsilon }) {
  demand(Array.isArray(points) && points.length >= 1, 'at least one observation is needed');
  const labels = labelsFromTarget(points, target);
  const fitted = fitTightInterval(points, labels);
  const risk = intervalRisk(fitted.interval, target);
  const segments = riskSegments(fitted.interval, target);
  const segmentTotal = segments.reduce((sum, segment) => sum + (segment.to - segment.from), 0);
  return {
    points: [...points],
    labels,
    target: [...target],
    interval: fitted.interval,
    empty: fitted.empty,
    empiricalRisk: 0,
    risk,
    segments,
    segmentTotal,
    epsilon,
    // Typed coordinates have at most three decimals. Roundoff in subtracting
    // their lengths must not turn mathematical equality into a failed event.
    meetsTarget: risk <= epsilon + 1e-12,
    positives: points.filter((_x, index) => labels[index] === 1),
    strips: epsilon <= target[1] - target[0] && target[1] > target[0] ? [
      { kind: 'coverage', from: target[0], to: target[0] + epsilon / 2 },
      { kind: 'coverage', from: target[1] - epsilon / 2, to: target[1] },
    ] : [],
  };
}

/** The same world with labels the learner supplies by hand. Consistency is now
 *  a question rather than a guarantee, and an inconsistent request is a valid
 *  result rather than an error. */
export function manualIntervalRequest({ points, labels }) {
  demand(points.length === labels.length, 'points and labels must have the same length');
  const named = points.map((x, index) => ({ id: `x${index + 1}`, x, label: labels[index] }));
  const distinct = new Map();
  for (const point of named) {
    checkFinite(point.x, `${point.id} coordinate`);
    demand(point.label === 0 || point.label === 1, `${point.id} must be labelled 0 or 1`);
    const previous = distinct.get(point.x);
    if (previous && previous.label !== point.label) return {
      family: 'interval', order: named.map(entry => entry.id), pattern: [...labels],
      feasible: false, witness: null, points: named, predictions: null,
      obstruction: { conflicting: [previous.id, point.id], coordinate: point.x },
      explain: `${previous.id} and ${point.id} request different labels at the same input ${point.x}. `
        + 'No deterministic interval rule can give that input both labels.',
    };
    if (!previous) distinct.set(point.x, point);
  }
  const verdict = requestVerdict({ points: [...distinct.values()], family: 'interval' });
  return { ...verdict, points: named, predictions: witnessPredictions(verdict.witness, named) };
}

/* ====================================== a declared, reproducible browser RNG */

/** mulberry32. Declared by name and seed because it is NOT NumPy's generator:
 *  the retained seed-41 experiment in `pac-data.js` cannot be reproduced here
 *  and the page never claims it can. This only generates sample coordinates a
 *  learner then commits a prediction about. */
export function seededUniformSample(seed, n) {
  demand(Number.isInteger(seed) && seed >= 0 && seed <= 999999, `seed must be an integer 0 to 999999, got ${seed}`);
  demand(Number.isInteger(n) && n >= 1 && n <= 24, `this control draws 1 to 24 points, got ${n}`);
  let state = (seed + 0x6d2b79f5) >>> 0;
  const draws = [];
  for (let index = 0; index < n; index += 1) {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    draws.push(Number((((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296).toFixed(3)));
  }
  return draws;
}

/* ================================= §10 one real parameter, infinite patterns */

/**
 * The binary-fraction sine construction, exactly.
 *
 * For labels y_1..y_n, form r whose first n fractional bits are 1 - y_i and
 * whose next two bits are 01, then take theta = 2 pi r. At x = 2^{i-1} the
 * fractional part of r x puts the i-th chosen bit first, so the cycle lands
 * strictly inside (0, 1/2) when that bit is 0 and strictly inside (1/2, 1)
 * when it is 1. The appended bits are what keep it strictly inside.
 */
export function sineWitness(labels) {
  demand(Array.isArray(labels) && labels.length >= 1 && labels.length <= 10,
    `the construction is shown for 1 to 10 labels, got ${labels.length}`);
  labels.forEach(bit => demand(bit === 0 || bit === 1, `labels are 0 or 1, got ${bit}`));
  const n = labels.length;
  let r = fraction(1n, 2n ** BigInt(n + 2));
  labels.forEach((bit, index) => {
    r = fractionAdd(r, fraction(BigInt(1 - bit), 2n ** BigInt(index + 1)));
  });
  const points = Array.from({ length: n }, (_unused, index) => 2 ** index);
  const cycles = points.map(x => {
    const scaled = fractionMultiply(r, fraction(BigInt(x)));
    const whole = scaled.numerator / scaled.denominator;
    return fraction(scaled.numerator - whole * scaled.denominator, scaled.denominator);
  });
  const half = fraction(1n, 2n);
  const predicted = cycles.map(cycle =>
    (fractionCompare(cycle, ZERO) > 0 && fractionCompare(cycle, half) < 0 ? 1 : 0));
  demand(predicted.every((bit, index) => bit === labels[index]),
    `the construction failed to realize ${labels.join('')}`);
  return {
    labels: [...labels],
    points,
    r,
    rText: fractionText(r),
    rBits: `.${labels.map(bit => 1 - bit).join('')}01`,
    theta: 2 * Math.PI * fractionToNumber(r),
    cycles: cycles.map(fractionText),
    cycleValues: cycles.map(fractionToNumber),
    predicted,
  };
}

/* ========================================= §5 half-plane witnesses, as drawn */

/** Which side of w . x + b = 0 a point falls on, and by how much. */
export function signedMargin(point, w, b) {
  checkFinite(point[0], 'point x');
  checkFinite(point[1], 'point y');
  return w[0] * point[0] + w[1] * point[1] + b;
}

export function halfPlanePredictions(points, w, b) {
  return points.map(point => (signedMargin(point, w, b) >= 0 ? 1 : 0));
}

/**
 * The segment of the line w . x + b = 0 inside a box, or null when the rule is
 * a constant sign and draws no line at all.
 *
 * The constant-sign case is not a skip. The all-positive and all-negative
 * witnesses genuinely have w = (0, 0), and a figure that silently drew nothing
 * for them without saying so would look like a missing line.
 */
export function lineSegmentInBox(w, b, box) {
  const [w1, w2] = w;
  checkFinite(w1, 'w1');
  checkFinite(w2, 'w2');
  checkFinite(b, 'b');
  const { minX, maxX, minY, maxY } = box;
  demand(minX < maxX && minY < maxY, 'the box must be non-degenerate');
  if (w1 === 0 && w2 === 0) {
    return { kind: 'constant', sign: b >= 0 ? 1 : 0, points: null };
  }
  const hits = [];
  const add = (x, y) => {
    if (x < minX - 1e-9 || x > maxX + 1e-9 || y < minY - 1e-9 || y > maxY + 1e-9) return;
    if (hits.some(hit => Math.abs(hit[0] - x) < 1e-9 && Math.abs(hit[1] - y) < 1e-9)) return;
    hits.push([x, y]);
  };
  if (w2 !== 0) {
    add(minX, -(w1 * minX + b) / w2);
    add(maxX, -(w1 * maxX + b) / w2);
  }
  if (w1 !== 0) {
    add(-(w2 * minY + b) / w1, minY);
    add(-(w2 * maxY + b) / w1, maxY);
  }
  if (hits.length < 2) return { kind: 'outside', sign: null, points: null };
  hits.sort((left, right) => (left[0] - right[0]) || (left[1] - right[1]));
  return { kind: 'line', sign: null, points: [hits[0], hits[hits.length - 1]] };
}

/**
 * One drawn half-plane panel.
 *
 * The x and y scales are forced to the same number of units per pixel. This is
 * a geometry figure: a stretched axis would turn a perpendicular separator into
 * a slanted one and an equilateral configuration into a scalene one, and the
 * argument the figure makes is about angles and separation.
 */
export function planePanelGeometry({ points, labels, witness, size = 92, margin = 9 }) {
  demand(Array.isArray(points) && points.length >= 2, 'a panel needs at least two points');
  const xs = points.map(point => point[0]);
  const ys = points.map(point => point[1]);
  const low = Math.min(Math.min(...xs), Math.min(...ys));
  const high = Math.max(Math.max(...xs), Math.max(...ys));
  const pad = Math.max(0.55, (high - low) * 0.28);
  const box = { minX: low - pad, maxX: high + pad, minY: low - pad, maxY: high + pad };
  const x = linearScale({ domain: [box.minX, box.maxX], range: [margin, size - margin] });
  const y = linearScale({ domain: [box.minY, box.maxY], range: [size - margin, margin] });
  const unitsPerPixelX = (box.maxX - box.minX) / (size - 2 * margin);
  const unitsPerPixelY = (box.maxY - box.minY) / (size - 2 * margin);
  demand(Math.abs(unitsPerPixelX - unitsPerPixelY) < 1e-12, 'the panel axes must share one physical scale');
  const separator = witness ? lineSegmentInBox(witness.w, witness.b, box) : null;
  return {
    size,
    margin,
    box,
    x,
    y,
    unitsPerPixel: unitsPerPixelX,
    marks: points.map((point, index) => ({
      point,
      label: labels ? labels[index] : null,
      cx: x(point[0]),
      cy: y(point[1]),
      margin: witness ? signedMargin(point, witness.w, witness.b) : null,
    })),
    separator: separator && separator.kind === 'line'
      ? { kind: 'line', from: [x(separator.points[0][0]), y(separator.points[0][1])],
        to: [x(separator.points[1][0]), y(separator.points[1][1])] }
      : separator,
    agrees: witness && labels
      ? halfPlanePredictions(points, witness.w, witness.b).every((bit, index) => bit === labels[index])
      : null,
  };
}

/** The two diagonals of a four-point configuration, for the alternating
 *  labeling whose positive and negative pairs cross. */
export function diagonalGeometry({ points, labels, panel }) {
  const positive = points.map((point, index) => ({ point, index })).filter(entry => labels[entry.index] === 1);
  const negative = points.map((point, index) => ({ point, index })).filter(entry => labels[entry.index] === 0);
  demand(positive.length === 2 && negative.length === 2, 'a crossing diagonal needs two points on each side');
  const project = entry => [panel.x(entry.point[0]), panel.y(entry.point[1])];
  return {
    positive: positive.map(project),
    negative: negative.map(project),
    crosses: segmentsCross(positive[0].point, positive[1].point, negative[0].point, negative[1].point),
  };
}

function orientation(origin, first, second) {
  const value = (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0]);
  return value > 1e-12 ? 1 : value < -1e-12 ? -1 : 0;
}

/** Do the two closed segments meet? Used only to state, in the figure's own
 *  caption, that the two diagonals really do cross. */
export function segmentsCross(a, b, c, d) {
  const first = orientation(a, b, c);
  const second = orientation(a, b, d);
  const third = orientation(c, d, a);
  const fourth = orientation(c, d, b);
  return first !== second && third !== fourth && first !== 0 && second !== 0 && third !== 0 && fourth !== 0;
}

/** The convex combination that traps an interior point: the weights putting the
 *  interior point at the centroid-style average of the surrounding triangle. */
export function interiorCombination({ triangle, interior }) {
  demand(triangle.length === 3, 'three surrounding points are needed');
  const [p, q, r] = triangle;
  const determinant = (q[1] - r[1]) * (p[0] - r[0]) + (r[0] - q[0]) * (p[1] - r[1]);
  demand(Math.abs(determinant) > 1e-12, 'the three surrounding points must not be collinear');
  const first = ((q[1] - r[1]) * (interior[0] - r[0]) + (r[0] - q[0]) * (interior[1] - r[1])) / determinant;
  const second = ((r[1] - p[1]) * (interior[0] - r[0]) + (p[0] - r[0]) * (interior[1] - r[1])) / determinant;
  const third = 1 - first - second;
  const weights = [first, second, third];
  return {
    weights,
    inside: weights.every(weight => weight > -1e-12),
    reconstructed: [
      weights[0] * p[0] + weights[1] * q[0] + weights[2] * r[0],
      weights[0] * p[1] + weights[1] * q[1] + weights[2] * r[1],
    ],
  };
}

/* ================================================== scales and plot geometry
 *
 * Every figure's coordinates come from these. A chart that positions its own
 * marks by hand is a chart whose positions nobody checked.
 */

export function linearScale({ domain, range }) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  checkFinite(d0, 'domain start');
  checkFinite(d1, 'domain end');
  demand(d0 !== d1, 'a scale needs a non-degenerate domain');
  const scale = value => r0 + ((r1 - r0) * (checkFinite(value, 'scaled value') - d0)) / (d1 - d0);
  scale.domain = [d0, d1];
  scale.range = [r0, r1];
  scale.invert = position => d0 + ((d1 - d0) * (position - r0)) / (r1 - r0);
  return scale;
}

export function logScale({ domain, range }) {
  const [d0, d1] = domain;
  demand(d0 > 0 && d1 > 0, `a log scale needs a positive domain, got [${d0}, ${d1}]`);
  const inner = linearScale({ domain: [Math.log10(d0), Math.log10(d1)], range });
  const scale = value => {
    demand(checkFinite(value, 'scaled value') > 0, `a log scale cannot place ${value}`);
    return inner(Math.log10(value));
  };
  scale.domain = [d0, d1];
  scale.range = range;
  scale.invert = position => 10 ** inner.invert(position);
  return scale;
}

/* ---- figure 1: two levels of randomness, a declared schematic */

export const schematicDraws = [
  { id: 'S1', populationError: 0.06 },
  { id: 'S2', populationError: 0.09 },
  { id: 'S3', populationError: 0.17 },
  { id: 'S4', populationError: 0.04 },
];

export function meterGeometry({ draws = schematicDraws, epsilon = 0.1, width = 150, height = 13 } = {}) {
  demand(Number.isFinite(epsilon) && epsilon > 0 && epsilon < 1, `epsilon must lie in (0, 1), got ${epsilon}`);
  const scale = linearScale({ domain: [0, 0.25], range: [0, width] });
  return {
    width,
    height,
    epsilon,
    cutoffX: scale(epsilon),
    axisMaximum: 0.25,
    rows: draws.map(draw => ({
      ...draw,
      barWidth: scale(draw.populationError),
      exceeds: draw.populationError > epsilon,
    })),
    exceedingDraws: draws.filter(draw => draw.populationError > epsilon).length,
    totalDraws: draws.length,
  };
}

/* ---- figure 3: twenty-five predeclared candidates and one protected event */

/** Constructed error counts out of 500 evaluation examples. Declared, not
 *  measured: no model was fitted and no data were collected for this figure.
 *  They are counts so that every displayed error is an exact multiple of 1/500
 *  and the arithmetic on the page is reproducible. */
export const candidateErrorCounts = [
  131, 136, 149, 152, 158, 160, 163, 167, 171, 174, 176, 179, 182, 186, 189,
  193, 197, 201, 206, 212, 219, 227, 236, 248, 263,
];

export function candidateBandGeometry({
  counts = candidateErrorCounts, n = 500, delta = 0.05, width = 260, rowHeight = 11,
} = {}) {
  demand(counts.length >= 2, 'a family needs at least two candidates to make selection interesting');
  counts.forEach(count => demand(Number.isInteger(count) && count >= 0 && count <= n,
    `an error count must be an integer 0 to ${n}, got ${count}`));
  const k = counts.length;
  const radius = finiteRadius(k, n, delta);
  const singleRadius = finiteRadius(1, n, delta);
  const rows = counts.map((count, index) => {
    const empirical = count / n;
    return {
      index,
      name: `h${index + 1}`,
      errorCount: count,
      empirical,
      low: empirical - radius,
      high: empirical + radius,
    };
  });
  const best = rows.reduce((lowest, row) => (row.empirical < lowest.empirical ? row : lowest));
  const runnerUp = rows.filter(row => row !== best).reduce((lowest, row) =>
    (row.empirical < lowest.empirical ? row : lowest));
  const domainLow = Math.min(...rows.map(row => row.low));
  const domainHigh = Math.max(...rows.map(row => row.high));
  const scale = linearScale({ domain: [Math.max(0, domainLow - 0.02), Math.min(1, domainHigh + 0.02)], range: [0, width] });
  return {
    k,
    n,
    delta,
    radius,
    singleRadius,
    shrunkRadius: finiteRadius(3, n, delta),
    width,
    rowHeight,
    height: rows.length * rowHeight + 4,
    scale,
    domain: scale.domain,
    rows: rows.map((row, index) => ({
      ...row,
      y: index * rowHeight + rowHeight / 2,
      centreX: scale(row.empirical),
      lowX: scale(row.low),
      highX: scale(row.high),
      selected: row === best,
    })),
    best,
    runnerUp,
    separation: runnerUp.empirical - best.empirical,
    rankingCertified: runnerUp.empirical - best.empirical > 2 * radius,
  };
}

/* ---- figure 5: growth counts on a logarithmic axis */

export function growthPlotGeometry({ rows = growthTable, width = 290, height = 178,
  padding = { left: 40, right: 26, top: 22, bottom: 30 } } = {}) {
  const x = linearScale({ domain: [rows[0].n, rows[rows.length - 1].n], range: [padding.left, width - padding.right] });
  const maximum = Math.max(...rows.map(row => row.allBinary));
  const y = logScale({ domain: [1, maximum], range: [height - padding.bottom, padding.top] });
  const series = ['thresholds', 'intervals', 'sauerD2', 'allBinary'].map(key => ({
    key,
    points: rows.map(row => ({ n: row.n, value: row[key], x: x(row.n), y: y(row[key]) })),
  }));
  return {
    width, height, padding, x, y, rows, series,
    yTicks: [1, 10, 100, 1000].filter(value => value <= maximum).map(value => ({ value, y: y(value) })),
    xTicks: rows.filter(row => row.n === 1 || row.n % 2 === 0).map(row => ({ value: row.n, x: x(row.n) })),
  };
}

/* ---- figure 7: the explicit bound curves, with the vacuous region shown */

export const boundGrid = { dimensions: [1, 2, 10], sizes: [100, 1000, 10000, 100000], delta: 0.05 };

export function boundPlotGeometry({
  dimensions = boundGrid.dimensions, delta = boundGrid.delta,
  width = 290, height = 190, padding = { left: 42, right: 26, top: 22, bottom: 32 },
} = {}) {
  const sizes = [];
  for (let exponent = 2; exponent <= 5; exponent += 0.25) sizes.push(Math.round(10 ** exponent));
  const x = logScale({ domain: [100, 100000], range: [padding.left, width - padding.right] });
  const values = dimensions.flatMap(d => sizes.map(n => vcRadius(d, n, delta)));
  const top = Math.max(...values);
  const y = linearScale({ domain: [0, Math.ceil(top * 2) / 2], range: [height - padding.bottom, padding.top] });
  return {
    width, height, padding, x, y, delta, sizes,
    vacuousFrom: y(1),
    vacuousTo: y(y.domain[1]),
    series: dimensions.map(d => ({
      d,
      points: sizes.map(n => ({ n, value: vcRadius(d, n, delta), x: x(n), y: y(vcRadius(d, n, delta)) })),
      crossesOne: sizes.some(n => vcRadius(d, n, delta) <= 1),
    })),
    xTicks: [100, 1000, 10000, 100000].map(value => ({ value, x: x(value) })),
    yTicks: [0, 1, 2, 3].filter(value => value <= y.domain[1]).map(value => ({ value, y: y(value) })),
    table: dimensions.flatMap(d => boundGrid.sizes.map(n => ({ d, n, delta, radius: vcRadius(d, n, delta) }))),
  };
}

export const realizableGrid = { dimensions: [1, 2, 10], epsilons: [0.2, 0.1, 0.05], delta: 0.05 };

export function realizableSufficientTable({
  dimensions = realizableGrid.dimensions, epsilons = realizableGrid.epsilons, delta = realizableGrid.delta,
} = {}) {
  return dimensions.flatMap(d => epsilons.map(epsilon => ({
    d, epsilon, delta, sufficientN: realizableVcSampleBound(d, epsilon, delta),
  })));
}

/* ---- figure 8: boundary strips on the unit interval */

/* The unit line is INSET from the viewBox edges. Placing 0 at x = 0 put the
 * centred "0" tick label half outside the SVG, and an observation at .01 or .99
 * went the same way; the layout inspector found eight such labels on one line
 * and ten on another. The inset is the widest half-label plus a little. */
/**
 * The unit line, as stacked rows sharing one horizontal scale.
 *
 * The first version straddled a single axis: target band above it, fitted band
 * below it, disagreement strips across it and the coverage strips under the
 * tick labels. Rendered, that was four overlapping ribbons eight pixels apart
 * and the tick labels sat on top of the strips -- the missed pieces, which are
 * the entire point of the figure, could not be told from the bands they were
 * drawn over. The rows are now separated and ordered by meaning: what was
 * observed, what the target is, where a sample would have to land, where the
 * fit is wrong, and what was fitted, over one shared axis.
 *
 * The line is also INSET from the viewBox edges. Placing 0 at x = 0 put the
 * centred "0" tick label half outside the SVG, and an observation at .01 went
 * the same way.
 */
export function stripGeometry({ experiment, width = 300, inset = 16 }) {
  demand(inset >= 0 && 2 * inset < width, 'the inset must leave a line to draw on');
  /* Gaps of four to five units between rows, not one. The first separated
     version stacked them flush and the target, coverage and disagreement rows
     still read as one striped block at rendered size. */
  const rows = {
    pointLabelY: 20,
    /* Two rows for the coordinate labels. Observations at .31 and .35, or .65
       and .69, are eleven units apart on a 268-unit line while their labels are
       about twenty wide, so on one row they merged into "0.31|35". The shared
       layout inspector did not catch it: its overlap allowance is two pixels
       and these just cleared it. */
    pointLabelRowGap: 9,
    markY: 33,
    targetY: 43,
    targetHeight: 11,
    stripY: 58,
    stripHeight: 8,
    segmentY: 71,
    segmentHeight: 11,
    fittedY: 87,
    fittedHeight: 11,
    axisY: 109,
    tickY: 113,
    tickLabelY: 123,
  };
  const height = 131;
  const scale = linearScale({ domain: [0, 1], range: [inset, width - inset] });
  const band = ([from, to]) => ({ from, to, x: scale(from), width: scale(to) - scale(from) });
  return {
    width,
    height,
    rows,
    axisY: rows.axisY,
    inset,
    scale,
    target: band(experiment.target),
    fitted: experiment.interval ? band(experiment.interval) : null,
    segments: experiment.segments.map(segment => ({ ...segment, ...band([segment.from, segment.to]) })),
    strips: experiment.strips.map(strip => ({ ...strip, ...band([strip.from, strip.to]) })),
    points: assignLabelRows(experiment.points.map((x, index) => ({
      x, label: experiment.labels[index], position: scale(x),
    }))),
    ticks: [0, 0.25, 0.5, 0.75, 1].map(value => ({ value, x: scale(value) })),
  };
}

/**
 * Place each observation's coordinate label on one of two rows, or on neither.
 *
 * Greedy, left to right: a label goes on the highest row where it clears the
 * last label already placed there. A label that clears neither is suppressed
 * rather than drawn on top of its neighbour -- the sample stays fully readable
 * in the fields above and in the accessible description, and a collided label
 * is worse than an absent one.
 */
function assignLabelRows(points, perCharacter = 4.8, padding = 2) {
  const ordered = [...points].sort((left, right) => left.position - right.position);
  const rightEdge = [-Infinity, -Infinity];
  ordered.forEach(point => {
    const width = String(point.x).length * perCharacter + padding;
    point.labelWidth = width;
    point.labelRow = null;
    for (let row = 0; row < rightEdge.length; row += 1) {
      if (point.position - width / 2 >= rightEdge[row]) {
        point.labelRow = row;
        rightEdge[row] = point.position + width / 2;
        break;
      }
    }
  });
  return points;
}

/** Every pair of drawn coordinate labels that would overlap. Zero is required;
 *  the verifier asserts it over dense fixtures, not only the worked one. */
export function labelCollisions(drawing) {
  const drawn = drawing.points.filter(point => point.labelRow !== null);
  const found = [];
  for (let i = 0; i < drawn.length; i += 1) {
    for (let j = i + 1; j < drawn.length; j += 1) {
      if (drawn[i].labelRow !== drawn[j].labelRow) continue;
      const gap = Math.abs(drawn[i].position - drawn[j].position);
      if (gap < (drawn[i].labelWidth + drawn[j].labelWidth) / 2) {
        found.push({ a: drawn[i].x, b: drawn[j].x, row: drawn[i].labelRow, gap });
      }
    }
  }
  return found;
}

/* ---- figure 10: the measured development curves */

/** Turn the fifteen recorded rows into two labelled error curves per procedure.
 *  The rows come from `pac-data.js`; the geometry is computed here so it can be
 *  asserted against the counts. */
export function learningCurveGeometry({ rows, sizes, width = 290, height = 185,
  padding = { left: 44, right: 26, top: 22, bottom: 32 } }) {
  demand(Array.isArray(rows) && rows.length >= 1, 'the measured rows are required');
  const x = logScale({ domain: [sizes[0], sizes[sizes.length - 1]], range: [padding.left, width - padding.right] });
  const errors = rows.flatMap(row => [
    1 - row.trainCorrect / row.n, 1 - row.developmentCorrect / row.developmentN,
  ]);
  /* The top comes from EVERY procedure's errors, so switching procedures does
     not silently rescale the axis and make two panels look alike when they are
     not. It is not floored at .3: that floor pushed every RBF SVC point into
     the bottom third of the frame, where a fall from .10 to .0125 was a flat
     line. */
  const top = Math.ceil(Math.max(...errors) * 40) / 40;
  const y = linearScale({ domain: [0, top], range: [height - padding.bottom, padding.top] });
  const models = [...new Set(rows.map(row => row.model))];
  return {
    width, height, padding, x, y, sizes, models,
    series: models.flatMap(model => ['train', 'development'].map(kind => ({
      model,
      kind,
      points: rows.filter(row => row.model === model).map(row => {
        const value = kind === 'train'
          ? 1 - row.trainCorrect / row.n
          : 1 - row.developmentCorrect / row.developmentN;
        return { n: row.n, value, x: x(row.n), y: y(value), row };
      }),
    }))),
    xTicks: sizes.map(value => ({ value, x: x(value) })),
    yTicks: [0, 0.05, 0.1, 0.15, 0.2, 0.25].filter(value => value <= top).map(value => ({ value, y: y(value) })),
  };
}

/* ---- figure 9: the retained simulation summary */

/**
 * Risk quantiles across the retained draws.
 *
 * The y axis is LOGARITHMIC. On a linear axis running to the n = 10 spread of
 * about .4, the bars at n = 50, 100 and 200 -- which span .007 to .097, .004 to
 * .047 and .002 to .024 -- collapsed into slivers a pixel or two tall, so the
 * figure's own claim, that the bar spans the 5th to 95th percentile, could not
 * be read off it at three of its five sizes. Every plotted risk is strictly
 * positive, so a log axis is available and is what makes the decay visible.
 */
export function simulationPlotGeometry({ rows, width = 290, height = 175,
  padding = { left: 44, right: 26, top: 22, bottom: 32 } }) {
  demand(Array.isArray(rows) && rows.length >= 2, 'the retained summary rows are required');
  const sizes = rows.map(row => row.n);
  const x = logScale({ domain: [sizes[0], sizes[sizes.length - 1]], range: [padding.left, width - padding.right] });
  const lowest = Math.min(...rows.map(row => row.riskQuantiles[0]));
  const highest = Math.max(...rows.map(row => row.riskQuantiles[2]));
  demand(lowest > 0, 'a logarithmic risk axis needs every plotted quantile to be strictly positive');
  const y = logScale({
    domain: [10 ** Math.floor(Math.log10(lowest)), 10 ** Math.ceil(Math.log10(highest))],
    range: [height - padding.bottom, padding.top],
  });
  return {
    width, height, padding, x, y, rows, logarithmic: true,
    epsilonY: y(rows[0].epsilon),
    bars: rows.map(row => ({
      n: row.n,
      x: x(row.n),
      lowY: y(row.riskQuantiles[0]),
      medianY: y(row.riskQuantiles[1]),
      highY: y(row.riskQuantiles[2]),
      meanY: y(row.meanTrueError),
      failureCount: row.failureCount,
      repetitions: row.repetitions,
    })),
    xTicks: sizes.map(value => ({ value, x: x(value) })),
    yTicks: [0.001, 0.01, 0.1, 1]
      .filter(value => value >= y.domain[0] && value <= y.domain[1])
      .map(value => ({ value, y: y(value) })),
  };
}

/* ---- figure 11: bits, shifts and semicircles */

export function sinePlotGeometry({ labels, radius = 34 }) {
  const witness = sineWitness(labels);
  return {
    witness,
    radius,
    dials: witness.cycleValues.map((cycle, index) => {
      const angle = 2 * Math.PI * cycle;
      return {
        index,
        x: witness.points[index],
        cycle,
        cycleText: witness.cycles[index],
        positive: witness.predicted[index] === 1,
        markX: radius * Math.cos(angle),
        markY: -radius * Math.sin(angle),
      };
    }),
  };
}

/* ---- figure 6: the ghost sample and pattern collapse */

export const ghostSampleFixture = {
  training: [0.12, 0.31, 0.44, 0.68],
  ghost: [0.22, 0.55, 0.77, 0.91],
};

/** Restrict the interval class to the combined inputs and collapse the
 *  parameter settings that produce the same binary row. The count that results
 *  is the growth function on those inputs, and the figure says so. */
export function ghostCollapse({ training, ghost } = ghostSampleFixture) {
  const combined = [...training.map(x => ({ x, from: 'training' })), ...ghost.map(x => ({ x, from: 'ghost' }))]
    .sort((left, right) => left.x - right.x);
  const patterns = intervalPatterns(combined.length);
  return {
    training: [...training],
    ghost: [...ghost],
    combined,
    combinedSize: combined.length,
    patternCount: patterns.length,
    patternCountFormula: patternCountFormula('interval', combined.length),
    allBinary: 2 ** combined.length,
    trainingOnlyPatternCount: intervalPatterns(training.length).length,
    sample: patterns.slice(0, 6),
  };
}

/* ========================================================== named fixtures */

export const fixtures = {
  finiteWorld: { target: [0, 0, 1, 1], sample: [0, 1], epsilon: 0.25 },
  finiteWorldContrast: { target: [0, 0, 1, 1], sample: [0, 1, 2], epsilon: 0.25 },
  finiteWorldComplete: { target: [0, 0, 1, 1], sample: [0, 1, 2, 3], epsilon: 0.25 },
  finiteWorldRepeatNull: { target: [0, 0, 1, 1], sample: [0, 1, 0], epsilon: 0.25 },
  finiteWorldZeroTarget: { target: [0, 0, 0, 0], sample: [0, 1], epsilon: 0.25 },
  finiteWorldProbabilitySizes: [1, 2, 4, 8, 16, 24],

  request: {
    family: 'interval',
    points: [{ id: 'A', x: 0.2, label: 1 }, { id: 'B', x: 0.5, label: 0 }, { id: 'C', x: 0.8, label: 1 }],
  },
  requestRepaired: {
    family: 'interval',
    points: [{ id: 'A', x: 0.2, label: 1 }, { id: 'B', x: 0.5, label: 1 }, { id: 'C', x: 0.8, label: 1 }],
  },
  requestOrderNull: {
    family: 'interval',
    points: [{ id: 'A', x: 0.1, label: 1 }, { id: 'B', x: 0.4, label: 0 }, { id: 'C', x: 0.9, label: 1 }],
  },
  requestTranslated: {
    family: 'interval',
    points: [{ id: 'A', x: 2.2, label: 1 }, { id: 'B', x: 2.5, label: 0 }, { id: 'C', x: 2.8, label: 1 }],
  },
  requestCrossed: {
    family: 'interval',
    points: [{ id: 'A', x: 0.2, label: 1 }, { id: 'B', x: 0.95, label: 0 }, { id: 'C', x: 0.8, label: 1 }],
  },
  requestThreshold: {
    family: 'threshold',
    points: [{ id: 'A', x: 0.2, label: 1 }, { id: 'B', x: 0.5, label: 0 }],
  },

  interval: { points: [0.1, 0.2, 0.35, 0.55, 0.65, 0.9], target: [0.3, 0.7], epsilon: 0.1 },
  intervalCloserEdges: { points: [0.1, 0.2, 0.31, 0.35, 0.55, 0.65, 0.69, 0.9], target: [0.3, 0.7], epsilon: 0.1 },
  intervalNegativeNull: { points: [0.01, 0.1, 0.2, 0.35, 0.55, 0.65, 0.9, 0.99], target: [0.3, 0.7], epsilon: 0.1 },
  intervalNoPositives: { points: [0.1, 0.2, 0.8, 0.9], target: [0.3, 0.7], epsilon: 0.1 },
  intervalPractice: { points: [0.05, 0.3, 0.4, 0.7, 0.95], target: [0.25, 0.75], epsilon: 0.1 },
  intervalPracticePositive: { points: [0.05, 0.26, 0.3, 0.4, 0.7, 0.95], target: [0.25, 0.75], epsilon: 0.1 },
  intervalPracticeNegativeNull: { points: [0.05, 0.3, 0.4, 0.7, 0.95, 0.99], target: [0.25, 0.75], epsilon: 0.1 },
  manualContradiction: { points: [0.2, 0.5, 0.8], labels: [1, 0, 1] },

  halfPlanes: {
    triangle: [[0, 0], [1, 0], [0, 1]],
    square: [[0, 0], [1, 0], [1, 1], [0, 1]],
    collinear: [[0, 0], [1, 0], [2, 0]],
    interior: [[0, 0], [2, 0], [0, 2], [0.5, 0.5]],
  },

  sineLabels: [1, 0, 1, 1],
  finiteFamily: { k: 25, n: 500, delta: 0.05 },
  practiceFamily: { k: 12, n: 800, delta: 0.02 },
  finiteRealizable: { k: 32, epsilon: 0.05, delta: 0.01 },
};

/** The section-4 numbers, computed once so the prose, the figure and the
 *  verifier cannot drift apart. */
export const finiteFamilyRadii = {
  selection: finiteRadius(fixtures.finiteFamily.k, fixtures.finiteFamily.n, fixtures.finiteFamily.delta),
  single: finiteRadius(1, fixtures.finiteFamily.n, fixtures.finiteFamily.delta),
  practice: finiteRadius(fixtures.practiceFamily.k, fixtures.practiceFamily.n, fixtures.practiceFamily.delta),
  quadrupledSample: finiteRadius(fixtures.finiteFamily.k, 4 * fixtures.finiteFamily.n, fixtures.finiteFamily.delta),
  sufficientRealizable: finiteRealizableSampleBound(
    fixtures.finiteRealizable.k, fixtures.finiteRealizable.epsilon, fixtures.finiteRealizable.delta),
};

/* ======================================================= grading primitives
 *
 * Both investigations that compare two states use these, and both are exercised
 * at their degenerate inputs by the verifier. The rule is stated once: a
 * quantity is unchanged exactly when the two values differ by no more than the
 * tolerance, and "unchanged" is reported if and only if that holds -- never
 * inferred from the inputs looking similar.
 */

export const unchangedTolerance = 1e-12;

export function movementOf(before, after, tolerance = unchangedTolerance) {
  checkFinite(before, 'the earlier value');
  checkFinite(after, 'the later value');
  const difference = after - before;
  if (Math.abs(difference) <= tolerance) return { outcome: 'unchanged', difference: 0, rawDifference: difference };
  return { outcome: difference > 0 ? 'rises' : 'falls', difference, rawDifference: difference };
}
