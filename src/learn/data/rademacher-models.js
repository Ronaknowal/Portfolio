/* Pure models for the Rademacher-complexity lesson.
 *
 * Nothing here renders. Every number the page prints and every coordinate the
 * page draws comes from this file, so `scripts/verify-rademacher-models.mjs`
 * can assert the same values the browser paints.
 *
 * Three distinctions run through the whole module and are kept apart by name,
 * because collapsing them is exactly how a generalisation bound gets
 * overstated:
 *
 *   `empiricalComplexity`  the exact average over sign patterns for ONE fixed
 *                          sample. Random only in the signs.
 *   `expectedComplexity`   an average over fresh samples as well. This module
 *                          never computes one from data; it appears only as a
 *                          recorded finite-population demonstration.
 *   `boundExpression`      empirical risk + complexity term + confidence term.
 *                          Holds with probability at least 1 - delta over the
 *                          SAMPLE draw, for every member of the class at once.
 *                          It is never a deterministic promise, and this module
 *                          reports `informative: false` whenever it exceeds the
 *                          trivial ceiling 1.
 *
 * Convention, fixed everywhere: E_sigma sup_f (1/n) sum_i sigma_i f(x_i), with
 * NO absolute value and a 1/n normalisation. `absoluteComplexity` exists only
 * so the lesson can show what the other convention measures; it is never mixed
 * into a bound.
 */

/** Bounds the browser enforces. The exact enumerators accept up to twelve
 *  observations, matching the packet program; the investigations hold
 *  themselves to eight columns so a pattern sweep stays at 256 rows. */
export const limits = {
  maxExactObservations: 12,
  maxLabColumns: 8,
  maxLabRows: 24,
  maxScalarRows: 20,
  maxVectors: 8,
  maxRealRows: 480,
  entryMagnitude: 2,
  vectorCoordinate: 3,
  maxRadius: 4,
  minRho: 0.1,
  maxRho: 3,
  minDelta: 0.01,
  maxDelta: 0.2,
  maxComparisons: 20,
};

/* ------------------------------------------------------------ small helpers */

const isFiniteNumber = value => typeof value === 'number' && Number.isFinite(value);

function requireFinite(values, label) {
  if (!Array.isArray(values) || values.length === 0) throw new RangeError(`${label}: need a non-empty list`);
  values.forEach(value => {
    if (!isFiniteNumber(value)) throw new RangeError(`${label}: ${value} is not a finite number`);
  });
  return values;
}

function requireMatrix(rows, label) {
  if (!Array.isArray(rows) || rows.length === 0) throw new RangeError(`${label}: need at least one row`);
  const width = Array.isArray(rows[0]) ? rows[0].length : -1;
  if (width <= 0) throw new RangeError(`${label}: rows must be non-empty arrays`);
  rows.forEach((row, index) => {
    if (!Array.isArray(row) || row.length !== width) {
      throw new RangeError(`${label}: row ${index} has a different length`);
    }
    requireFinite(row, `${label} row ${index}`);
  });
  return rows;
}

function requireRadius(radius, label) {
  if (!isFiniteNumber(radius) || radius < 0) throw new RangeError(`${label}: radius must be finite and >= 0`);
  return radius;
}

export const norm2 = vector => Math.sqrt(requireFinite(vector, 'norm2').reduce((sum, v) => sum + v * v, 0));
export const dot = (a, b) => {
  if (a.length !== b.length) throw new RangeError('dot: lengths differ');
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
};

/** A key that makes two numerically identical rows compare equal. Rounded to
 *  twelve decimals so that .1+.2 and .3 are one row, not two. */
const rowKey = row => row.map(value => (Object.is(value, -0) ? 0 : Number(value.toFixed(12)))).join('|');

/** Lexicographic order, so a set of rows has one canonical presentation. */
const byLexicographicOrder = (a, b) => {
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return a.length - b.length;
};

/** Distinct rows, in canonical order, with the duplicate groups named. */
export function distinctRows(rows) {
  requireMatrix(rows, 'distinctRows');
  const groups = new Map();
  rows.forEach((row, index) => {
    const key = rowKey(row);
    if (!groups.has(key)) groups.set(key, { row: row.slice(), indices: [] });
    groups.get(key).indices.push(index);
  });
  const unique = [...groups.values()].sort((a, b) => byLexicographicOrder(a.row, b.row));
  return {
    rows: unique.map(entry => entry.row),
    groups: unique.map(entry => entry.indices),
    duplicated: unique.filter(entry => entry.indices.length > 1).map(entry => entry.indices),
  };
}

/* ------------------------------------------------- exact sign-pattern sweeps */

/**
 * All 2^n sign patterns, in LEXICOGRAPHIC order: the first observation's sign
 * varies slowest, so pattern 0 is all -1 and the last is all +1.
 *
 * The order is not cosmetic. The content packet enumerated its patterns with
 * `itertools.product([-1, 1], repeat=n)`, which is this order, and the packet's
 * recorded per-pattern correlations, maxima and winning-row indices are stored
 * as parallel arrays. Enumerating with the first observation varying fastest
 * would misalign every one of those comparisons while still producing the right
 * average, which is exactly the kind of agreement that hides a defect.
 */
export function signPatterns(n) {
  if (!Number.isInteger(n) || n < 1 || n > limits.maxExactObservations) {
    throw new RangeError(`signPatterns: exact enumeration supports 1 to ${limits.maxExactObservations} observations`);
  }
  const total = 2 ** n;
  const patterns = new Array(total);
  for (let code = 0; code < total; code += 1) {
    const pattern = new Array(n);
    for (let index = 0; index < n; index += 1) pattern[index] = (code >> (n - 1 - index)) & 1 ? 1 : -1;
    patterns[code] = pattern;
  }
  return patterns;
}

/** Which row of `signPatterns(n)` a given sign vector is. */
export function signPatternIndex(signs) {
  const n = signs.length;
  return signs.reduce((code, sign, index) => code + (sign > 0 ? 2 ** (n - 1 - index) : 0), 0);
}

/**
 * The empirical Rademacher complexity of a finite set of prediction vectors,
 * by exact enumeration.
 *
 * The order of operations is the whole point of section 1, so both orders are
 * returned: `complexity` takes the maximum for each sign pattern and then
 * averages, and `maxAfterAveraging` averages each row first and then takes the
 * maximum. For a sign set closed under negation the second is always zero.
 *
 * `maxima` may be negative. A per-pattern maximum initialised to zero would
 * silently floor a class with no non-negative option, which is a wrong answer
 * rather than a rounding difference.
 */
export function empiricalComplexity(values) {
  const rows = requireMatrix(values, 'empiricalComplexity');
  const n = rows[0].length;
  const patterns = signPatterns(n);
  const correlations = patterns.map(pattern => rows.map(row => dot(pattern, row) / n));
  const maxima = correlations.map(scores => Math.max(...scores));
  const winners = correlations.map((scores, index) => scores
    .map((score, rowIndex) => (Math.abs(score - maxima[index]) <= 1e-12 ? rowIndex : -1))
    .filter(rowIndex => rowIndex >= 0));
  const rowAverages = rows.map((_row, rowIndex) =>
    correlations.reduce((sum, scores) => sum + scores[rowIndex], 0) / patterns.length);
  const unique = distinctRows(rows);
  return {
    rows,
    n,
    patterns,
    correlations,
    maxima,
    winners,
    firstWinner: winners.map(list => list[0]),
    rowAverages,
    complexity: maxima.reduce((sum, value) => sum + value, 0) / patterns.length,
    maxAfterAveraging: Math.max(...rowAverages),
    distinctRowCount: unique.rows.length,
    duplicateGroups: unique.duplicated,
  };
}

/**
 * The same quantity under the absolute-value convention, which is a DIFFERENT
 * measurement: it evaluates the class enlarged by every member's negative. The
 * lesson uses it once, to show that a convention is not typography.
 */
export function absoluteComplexity(values) {
  const rows = requireMatrix(values, 'absoluteComplexity');
  const n = rows[0].length;
  const patterns = signPatterns(n);
  const correlations = patterns.map(pattern => rows.map(row => Math.abs(dot(pattern, row)) / n));
  const maxima = correlations.map(scores => Math.max(...scores));
  const winners = correlations.map((scores, index) => scores
    .map((score, rowIndex) => (Math.abs(score - maxima[index]) <= 1e-12 ? rowIndex : -1))
    .filter(rowIndex => rowIndex >= 0));
  const rowAverages = rows.map((_row, rowIndex) =>
    correlations.reduce((sum, scores) => sum + scores[rowIndex], 0) / patterns.length);
  return {
    patterns,
    correlations,
    maxima,
    winners,
    rowAverages,
    maxAfterAveraging: Math.max(...rowAverages),
    complexity: maxima.reduce((sum, value) => sum + value, 0) / patterns.length,
  };
}

/**
 * Every distinct restriction to the sample of the positive-threshold class
 * h_t(x) = +1 iff x >= t.
 *
 * Both extreme cutoffs are included: restricting the cutoffs to observed values
 * loses the all-negative row, which is a real member of the infinite class and
 * changes the answer. Equal inputs are grouped, because a threshold cannot
 * separate two identical observations.
 */
export function thresholdRows(inputs, { bothOrientations = false } = {}) {
  const x = requireFinite(inputs, 'thresholdRows');
  const cuts = [Number.NEGATIVE_INFINITY, ...[...new Set(x)].sort((a, b) => a - b), Number.POSITIVE_INFINITY];
  const rows = cuts.map(cut => x.map(value => (value >= cut ? 1 : -1)));
  const all = bothOrientations ? [...rows, ...rows.map(row => row.map(value => -value))] : rows;
  return distinctRows(all).rows;
}

/**
 * The loss class of a binary hypothesis table under the mistake indicator,
 * (1 - y h)/2. Exact for h in {-1, +1} and y in {-1, +1}; the manuscript's
 * factor-one-half identity is a statement about this transformation, not about
 * hard-thresholding a real-valued score.
 */
export function mistakeRows(rows, labels) {
  requireMatrix(rows, 'mistakeRows');
  requireFinite(labels, 'mistakeRows labels');
  if (labels.length !== rows[0].length) throw new RangeError('mistakeRows: one label per observation');
  labels.forEach(label => {
    if (label !== 1 && label !== -1) throw new RangeError('mistakeRows: labels must be -1 or +1');
  });
  rows.forEach(row => row.forEach(value => {
    if (value !== 1 && value !== -1) throw new RangeError('mistakeRows: the identity needs sign-valued predictions');
  }));
  return rows.map(row => row.map((value, index) => (1 - labels[index] * value) / 2));
}

/** A convex average of existing prediction vectors. Weights must be
 *  non-negative and sum to one, or it is not a convex combination. */
export function convexMixture(rows, weights) {
  requireMatrix(rows, 'convexMixture');
  requireFinite(weights, 'convexMixture weights');
  if (weights.length !== rows.length) throw new RangeError('convexMixture: one weight per row');
  if (weights.some(weight => weight < 0)) throw new RangeError('convexMixture: weights must be non-negative');
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (Math.abs(total - 1) > 1e-9) throw new RangeError('convexMixture: weights must sum to 1');
  return rows[0].map((_value, column) => rows.reduce((sum, row, index) => sum + weights[index] * row[column], 0));
}

/* ------------------------------------------------------- Euclidean geometry */

/** sum_i sigma_i x_i, the vector every best response points along. */
export function signedSum(vectors, signs) {
  requireMatrix(vectors, 'signedSum');
  if (!Array.isArray(signs) || signs.length !== vectors.length) {
    throw new RangeError('signedSum: one sign per observation');
  }
  signs.forEach(sign => {
    if (sign !== 1 && sign !== -1) throw new RangeError('signedSum: signs must be -1 or +1');
  });
  return vectors[0].map((_value, column) =>
    vectors.reduce((sum, row, index) => sum + signs[index] * row[column], 0));
}

/**
 * The exact inner optimisation for one sign pattern over the ball ||w||_2 <= B.
 *
 * The optimiser is B v / ||v||. When v = 0 the objective is identically zero
 * and EVERY feasible w is optimal; when B = 0 the only feasible w is the
 * origin. Those two degenerate cases are reported rather than papered over,
 * because "the maximiser is unique" is one of the questions the lab grades.
 */
export function bestResponse(vectors, signs, radius) {
  requireRadius(radius, 'bestResponse');
  const v = signedSum(vectors, signs);
  const length = norm2(v);
  const n = vectors.length;
  const zeroSum = length <= 1e-15;
  return {
    v,
    length,
    n,
    radius,
    optimum: (radius * length) / n,
    optimizer: radius === 0 ? v.map(() => 0) : (zeroSum ? null : v.map(value => (radius * value) / length)),
    optimizerUnique: radius === 0 || !zeroSum,
    degenerate: radius === 0 ? 'the budget is zero, so the origin is the unique feasible coefficient and maximizer'
      : (zeroSum ? 'the signed sum is zero, so every feasible coefficient attains the same value 0' : null),
  };
}

/** The value a LEARNER-PROPOSED coefficient actually achieves, and how far
 *  short of the supremum it falls. Infeasible proposals are refused with their
 *  own norm, never clamped into the ball. */
export function achievedCorrelation(vectors, signs, candidate, radius) {
  requireFinite(candidate, 'achievedCorrelation');
  requireRadius(radius, 'achievedCorrelation');
  if (candidate.length !== vectors[0].length) throw new RangeError('achievedCorrelation: wrong coefficient length');
  const best = bestResponse(vectors, signs, radius);
  const candidateNorm = norm2(candidate);
  return {
    ...best,
    candidate,
    candidateNorm,
    feasible: candidateNorm <= radius + 1e-12,
    achieved: dot(candidate, best.v) / vectors.length,
    shortfall: best.optimum - dot(candidate, best.v) / vectors.length,
  };
}

/**
 * The exact empirical complexity of the Euclidean ball class, by enumeration,
 * with both upper bounds the lesson derives.
 *
 * `energyUpper` uses the observed feature energy; `maxRowUpper` uses the
 * largest observed row norm. Neither is a bound on future observations, and
 * the exact value is what the lesson contrasts them with.
 */
export function linearComplexity(vectors, radius = 1) {
  const rows = requireMatrix(vectors, 'linearComplexity');
  requireRadius(radius, 'linearComplexity');
  const n = rows.length;
  const patterns = signPatterns(n);
  const sums = patterns.map(pattern => signedSum(rows, pattern));
  const maxima = sums.map(v => (radius * norm2(v)) / n);
  const energy = Math.sqrt(rows.reduce((sum, row) => sum + row.reduce((inner, value) => inner + value * value, 0), 0));
  const maxRowNorm = Math.max(...rows.map(norm2));
  return {
    vectors: rows,
    radius,
    n,
    patterns,
    sums,
    maxima,
    complexity: maxima.reduce((sum, value) => sum + value, 0) / patterns.length,
    energy,
    energyUpper: (radius * energy) / n,
    maxRowNorm,
    maxRowUpper: (radius * maxRowNorm) / Math.sqrt(n),
  };
}

/**
 * The same calculation in a feature space given only by inner products.
 *
 * A Gram matrix must be symmetric and positive semidefinite or it is not a
 * Gram matrix, and a tiny negative eigenvalue from roundoff is tolerated while
 * a genuinely indefinite matrix is refused. The quadratic form is clamped at
 * zero before the square root for the same reason.
 */
export function kernelComplexity(gram, radius = 1) {
  const rows = requireMatrix(gram, 'kernelComplexity');
  requireRadius(radius, 'kernelComplexity');
  const n = rows.length;
  if (rows[0].length !== n) throw new RangeError('kernelComplexity: the Gram matrix must be square');
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      if (Math.abs(rows[i][j] - rows[j][i]) > 1e-12) throw new RangeError('kernelComplexity: not symmetric');
    }
  }
  if (!isPositiveSemidefinite(rows)) throw new RangeError('kernelComplexity: not positive semidefinite');
  const patterns = signPatterns(n);
  const quadratics = patterns.map(pattern =>
    pattern.reduce((sum, si, i) => sum + si * pattern.reduce((inner, sj, j) => inner + rows[i][j] * sj, 0), 0));
  const maxima = quadratics.map(value => (radius * Math.sqrt(Math.max(value, 0))) / n);
  const trace = rows.reduce((sum, row, index) => sum + row[index], 0);
  return {
    gram: rows,
    radius,
    n,
    patterns,
    quadratics,
    maxima,
    complexity: maxima.reduce((sum, value) => sum + value, 0) / patterns.length,
    trace,
    traceUpper: (radius * Math.sqrt(trace)) / n,
  };
}

/** Symmetric positive semidefiniteness by leading-minor style elimination,
 *  which needs no eigensolver and is exact enough for the 2x2 and 3x3 matrices
 *  the lesson lets a learner enter. A tolerance absorbs roundoff. */
export function isPositiveSemidefinite(matrix, tolerance = 1e-10) {
  const n = matrix.length;
  const a = matrix.map(row => row.slice());
  for (let k = 0; k < n; k += 1) {
    if (a[k][k] < -tolerance) return false;
    if (a[k][k] <= tolerance) {
      /* A zero pivot is allowed only if the REMAINING submatrix's row is zero.
       *
       * Checking the whole row and column instead was a real defect: the
       * elimination only clears entries below the diagonal, so a[k][j] for
       * j < k is left holding the original value, and the perfectly good Gram
       * matrix [[1, 1], [1, 1]] -- two identical feature vectors -- was
       * rejected as indefinite. The test is about the submatrix that is still
       * to be reduced, and the matrix stays symmetric there. */
      for (let j = k; j < n; j += 1) if (Math.abs(a[k][j]) > 1e-7) return false;
      continue;
    }
    for (let i = k + 1; i < n; i += 1) {
      const factor = a[i][k] / a[k][k];
      for (let j = k; j < n; j += 1) a[i][j] -= factor * a[k][j];
    }
  }
  return true;
}

/**
 * The l1 best response: the whole budget goes on one coordinate of the signed
 * sum, the one with the largest absolute value, signed to help.
 *
 * Ties are reported in full. A diagram that draws one vertex of the diamond
 * while two are optimal is drawing a rule the calculation does not apply.
 */
export function l1BestResponse(vectors, signs, radius) {
  requireRadius(radius, 'l1BestResponse');
  const v = signedSum(vectors, signs);
  const magnitudes = v.map(Math.abs);
  const largest = Math.max(...magnitudes);
  const winners = magnitudes
    .map((value, index) => (Math.abs(value - largest) <= 1e-12 ? index : -1))
    .filter(index => index >= 0);
  const coordinate = winners[0];
  const optimizer = v.map((value, index) =>
    (index === coordinate && largest > 1e-15 ? radius * Math.sign(value) : 0));
  return {
    v,
    magnitudes,
    infinityNorm: largest,
    coordinate,
    tiedCoordinates: winners,
    optimum: (radius * largest) / vectors.length,
    optimizer,
    degenerate: largest <= 1e-15 ? 'every coordinate of the signed sum is zero' : null,
  };
}

/* ---------------------------------------------------------- counting bounds */

/**
 * Massart's finite-class bound, A sqrt(2 ln M) / n, for M distinct prediction
 * vectors of Euclidean length at most A.
 *
 * M = 1 gives exactly zero without dividing by anything, and A = 0 likewise.
 * Counting duplicate copies only inflates M: the caller passes the DISTINCT
 * count, and `countedDuplicates` records what deduplicating removed.
 */
export function massartBound(distinctCount, radius, n) {
  if (!Number.isInteger(distinctCount) || distinctCount < 1) throw new RangeError('massartBound: need M >= 1');
  if (!isFiniteNumber(radius) || radius < 0) throw new RangeError('massartBound: need A >= 0');
  if (!Number.isInteger(n) || n < 1) throw new RangeError('massartBound: need n >= 1');
  return (radius * Math.sqrt(2 * Math.log(distinctCount))) / n;
}

/** The sign-valued specialisation, with A = sqrt(n): sqrt(2 ln M / n). */
export const massartSignBound = (distinctCount, n) => massartBound(distinctCount, Math.sqrt(n), n);

/**
 * Sauer's lemma. `exactCount` is the true sum of binomial coefficients;
 * `simplified` is (e n / d)^d, which is only valid for 1 <= d <= n. Outside
 * that range the honest answer is 2^n, and `usable` says which is in force.
 */
export function sauerCount(vcDimension, n) {
  if (!Number.isInteger(vcDimension) || vcDimension < 0) throw new RangeError('sauerCount: need d >= 0');
  if (!Number.isInteger(n) || n < 1) throw new RangeError('sauerCount: need n >= 1');
  if (vcDimension === 0) return { exactCount: 1, simplified: null, usable: 'exact', bound: 0 };
  if (vcDimension > n) {
    return { exactCount: 2 ** n, simplified: null, usable: 'all-labelings', bound: massartSignBound(2 ** n, n) };
  }
  let exactCount = 0;
  let binomial = 1;
  for (let j = 0; j <= vcDimension; j += 1) {
    if (j > 0) binomial = (binomial * (n - j + 1)) / j;
    exactCount += binomial;
  }
  const simplified = ((Math.E * n) / vcDimension) ** vcDimension;
  return {
    exactCount,
    simplified,
    usable: 'simplified',
    bound: Math.sqrt((2 * vcDimension * Math.log((Math.E * n) / vcDimension)) / n),
  };
}

/* --------------------------------------------------------- losses and ramps */

export const logisticMarginLoss = margin => Math.log1p(Math.exp(-margin));
export const logisticMarginSlope = margin => -1 / (1 + Math.exp(margin));
export const hingeMarginLoss = margin => Math.max(0, 1 - margin);
export const sigmoid = score => 1 / (1 + Math.exp(-score));
export const sigmoidSlope = score => sigmoid(score) * (1 - sigmoid(score));

/**
 * The ramp phi_rho: 1 at or below margin 0, 0 at or above rho, linear between.
 *
 * This is the rule the figure draws AND the rule the lab applies. Both knots
 * are closed on the value they take from the flat side: phi(0) = 1 and
 * phi(rho) = 0 exactly, with no floating-point slack deciding a tie.
 */
export function rampLoss(margins, rho) {
  requireFinite(margins, 'rampLoss');
  if (!isFiniteNumber(rho) || rho <= 0) throw new RangeError('rampLoss: rho must be finite and positive');
  return margins.map(margin => {
    if (margin <= 0) return 1;
    if (margin >= rho) return 0;
    return 1 - margin / rho;
  });
}

/** The classification rule the whole lesson applies, stated once: a score of
 *  exactly zero predicts +1. The ramp charges 1 there, so it upper-bounds this
 *  rule's mistake indicator including the tie. */
export const predictSign = score => (score >= 0 ? 1 : -1);

/** 3 sqrt(ln(2K/delta) / (2n)) -- the empirical-complexity theorem's third
 *  term, with a union bound over K predeclared comparisons already inside it. */
export function confidenceAddend(comparisons, delta, n) {
  if (!Number.isInteger(comparisons) || comparisons < 1) throw new RangeError('confidenceAddend: need K >= 1');
  if (!isFiniteNumber(delta) || delta <= 0 || delta >= 1) throw new RangeError('confidenceAddend: need 0 < delta < 1');
  if (!Number.isInteger(n) || n < 1) throw new RangeError('confidenceAddend: need n >= 1');
  return 3 * Math.sqrt(Math.log((2 * comparisons) / delta) / (2 * n));
}

/** sqrt(sum ||x_i||^2) / n, the feature-energy factor. It depends on the
 *  inputs, so anything that edits an input must recompute it. */
export function featureEnergy(rows) {
  const matrix = requireMatrix(rows, 'featureEnergy');
  return Math.sqrt(matrix.reduce((sum, row) => sum + row.reduce((inner, value) => inner + value * value, 0), 0))
    / matrix.length;
}

/**
 * The margin bound of section 6, assembled from parts that are each visible.
 *
 * `informative` is false whenever the raw sum reaches the trivial ceiling 1.
 * The raw sum is always reported: clipping it away hides the reason the
 * expression says nothing.
 */
export function marginBound({ margins, radius, energy, rho, delta, comparisons }) {
  const ramp = rampLoss(margins, rho);
  requireRadius(radius, 'marginBound');
  if (!isFiniteNumber(energy) || energy < 0) throw new RangeError('marginBound: energy must be finite and >= 0');
  const n = margins.length;
  const empiricalRamp = ramp.reduce((sum, value) => sum + value, 0) / n;
  const complexityAddend = (2 * radius * energy) / rho;
  const confidence = confidenceAddend(comparisons, delta, n);
  const raw = empiricalRamp + complexityAddend + confidence;
  return {
    n,
    ramp,
    empiricalRamp,
    complexityAddend,
    confidence,
    raw,
    ceiling: 1,
    clipped: Math.min(1, raw),
    informative: raw < 1,
    trainingErrors: margins.filter(margin => margin <= 0).length,
    smallMargins: margins.filter(margin => margin > 0 && margin < rho).length,
  };
}

/**
 * Hoeffding's one-sided correction for a Monte-Carlo average of per-draw
 * values that lie in [0, Q].
 *
 * The lesson never runs a sign simulation in the browser: the estimates it
 * shows were produced once, with a recorded seed, by the packet program. This
 * function recomputes the DETERMINISTIC half of that table, which is what an
 * independent check can confirm.
 */
export function hoeffdingCorrection(perDrawUpper, draws, eta) {
  if (!isFiniteNumber(perDrawUpper) || perDrawUpper < 0) throw new RangeError('hoeffdingCorrection: need Q >= 0');
  if (!Number.isInteger(draws) || draws < 1) throw new RangeError('hoeffdingCorrection: need T >= 1');
  if (!isFiniteNumber(eta) || eta <= 0 || eta >= 1) throw new RangeError('hoeffdingCorrection: need 0 < eta < 1');
  return perDrawUpper * Math.sqrt(Math.log(1 / eta) / (2 * draws));
}

/** The endpoint a predeclared draw count buys: the estimate plus its
 *  correction, never worse than the deterministic ceiling Q. */
export function monteCarloEndpoint({ estimate, perDrawUpper, draws, eta }) {
  const correction = hoeffdingCorrection(perDrawUpper, draws, eta);
  return {
    estimate,
    correction,
    perDrawUpper,
    endpoint: Math.min(perDrawUpper, estimate + correction),
    cappedByCeiling: estimate + correction > perDrawUpper,
  };
}

/* ------------------------------------------------------- the real predictor */

/**
 * The frozen representation: standardise with the parameters fitted on the
 * first eighty rows, divide by three, clip to [-1, 1], then append the
 * constant intercept coordinate.
 *
 * Clipping is part of the model, not a display convenience: it is what makes
 * every mapped row have norm at most sqrt(5), which is what the bound uses.
 */
export function mapFeatures(row, { mean, scale, clipStandardDeviations = 3 }) {
  requireFinite(row, 'mapFeatures');
  if (row.length !== mean.length || row.length !== scale.length) {
    throw new RangeError('mapFeatures: the representation has a different number of coordinates');
  }
  scale.forEach(value => {
    if (!(value > 0)) throw new RangeError('mapFeatures: a fitted scale must be positive');
  });
  return [
    ...row.map((value, index) =>
      Math.min(1, Math.max(-1, (value - mean[index]) / scale[index] / clipStandardDeviations))),
    1,
  ];
}

/** Scores, margins, mistakes and mean logistic loss for one coefficient vector.
 *  `errors` uses `predictSign`, so a zero score counts as a +1 prediction. */
export function evaluatePredictor(rows, labels, weights) {
  requireMatrix(rows, 'evaluatePredictor');
  requireFinite(weights, 'evaluatePredictor weights');
  if (labels.length !== rows.length) throw new RangeError('evaluatePredictor: one label per row');
  if (rows[0].length !== weights.length) throw new RangeError('evaluatePredictor: coefficient length mismatch');
  const scores = rows.map(row => dot(row, weights));
  const margins = scores.map((score, index) => labels[index] * score);
  const predictions = scores.map(predictSign);
  const errors = predictions.reduce((count, prediction, index) => count + (prediction !== labels[index] ? 1 : 0), 0);
  return {
    n: rows.length,
    scores,
    margins,
    predictions,
    errors,
    errorRate: errors / rows.length,
    logLoss: margins.reduce((sum, margin) => sum + logisticMarginLoss(margin), 0) / rows.length,
    norm: norm2(weights),
  };
}

/** The declared selection rule, applied to whatever candidates it is given:
 *  fewest validation errors, then smaller validation log loss, then smaller
 *  budget. It never reads an assessment field; it cannot, because it is not
 *  passed one. */
export function selectByValidation(candidates) {
  if (!Array.isArray(candidates) || candidates.length === 0) throw new RangeError('selectByValidation: no candidates');
  candidates.forEach(candidate => {
    if ('assessment' in candidate) throw new RangeError('selectByValidation: assessment must not be visible here');
  });
  return candidates.reduce((best, candidate) => {
    if (candidate.validationErrors !== best.validationErrors) {
      return candidate.validationErrors < best.validationErrors ? candidate : best;
    }
    if (Math.abs(candidate.validationLogLoss - best.validationLogLoss) > 1e-15) {
      return candidate.validationLogLoss < best.validationLogLoss ? candidate : best;
    }
    return candidate.radius < best.radius ? candidate : best;
  });
}

/* ------------------------------------------------------------- did it move? */

/** The number of decimals every investigation displays and grades at. One
 *  constant, so a verdict and a printed value cannot disagree. */
export const displayDigits = 6;

/** The printed form of a quantity, and the only form a comparison may use. */
export const displayValue = (value, digits = displayDigits) =>
  (Number.isFinite(value) ? Number(value.toFixed(digits)) : null);

/**
 * Did the graded quantity move?
 *
 * The comparison is made on the DISPLAYED values, not the raw doubles. That is
 * deliberate: a verdict reading "changed" beside two identical printed numbers
 * is the same defect as a verdict reading "unchanged" beside two different
 * ones, and comparing what is shown makes "the verdict says unchanged exactly
 * when the displayed output did not move" true by construction rather than by
 * a lucky choice of tolerance. `belowDisplayPrecision` reports the case where
 * the raw values differ but the printed ones do not, so a lab can say so
 * instead of pretending the difference is zero.
 */
/** The slack used for FEASIBILITY questions — is this coefficient inside its
 *  ball? — which are about representability, not about grading a comparison.
 *  Exported as a name so that no numeric tolerance literal needs to appear in
 *  the lab sources at all, which is what lets the hygiene checker ban them
 *  outright rather than by a list of exceptions. */
export const feasibilitySlack = 1e-12;

/**
 * A three-way comparison graded at the precision the page actually prints.
 *
 * This exists because the numeric-field fix did not generalise. The value
 * field's allowance was set to half a unit in the last printed place, but the
 * CATEGORICAL comparison beside it kept a hand-written 1e-12, so a learner one
 * keystroke from a shipped preset could be told their category was wrong while
 * the same verdict printed both "different" quantities as 0.707107 and told
 * them their number was right. Every graded comparison in this lesson now
 * routes through here, and `digits` is the same value the operands are printed
 * with, so "graded equal" and "printed the same" are one statement.
 */
export function gapOutcome(reference, candidate, { digits = displayDigits, below, equal, above }) {
  const comparison = compareOutcome(reference, candidate, digits);
  if (comparison.outcome === 'unchanged') return { outcome: equal, comparison };
  return { outcome: comparison.outcome === 'decreased' ? below : above, comparison };
}

export function compareOutcome(before, after, digits = displayDigits) {
  if (!isFiniteNumber(before) || !isFiniteNumber(after)) {
    throw new RangeError('compareOutcome: both quantities must be finite numbers');
  }
  const shownBefore = displayValue(before, digits);
  const shownAfter = displayValue(after, digits);
  const moved = shownBefore !== shownAfter;
  return {
    outcome: moved ? (shownAfter > shownBefore ? 'increased' : 'decreased') : 'unchanged',
    displayedBefore: shownBefore,
    displayedAfter: shownAfter,
    delta: after - before,
    belowDisplayPrecision: !moved && before !== after,
  };
}

/* =========================================================== drawn geometry */
/* Everything below returns coordinates. A drawn curve, band or supporting
 * point is a mathematical claim, so each of these is asserted by the model
 * verifier against the quantity it is supposed to depict -- not merely against
 * itself. Every viewBox is in the same units as the rendered width, so the
 * text inside it is sized in real pixels. */

/** A number line for the three ordered inputs and one movable cutoff. */
export function numberLineLayout(inputs, { width = 320, left = 34, right = 22, y = 46 } = {}) {
  const x = requireFinite(inputs, 'numberLineLayout');
  const low = Math.min(...x);
  const high = Math.max(...x);
  const span = high - low || 1;
  const usable = width - left - right;
  const place = value => left + (usable * (value - low)) / span;
  return {
    width,
    y,
    left,
    right,
    domain: [low, high],
    place,
    points: x.map((value, index) => ({ value, index, x: place(value) })),
  };
}

/**
 * The signed-vector picture: input arrows, the signed sum, the ball of radius
 * B and the supporting point B v / ||v||.
 *
 * Both axes use ONE scale. The picture is about a Euclidean length, so
 * stretching one axis to fill the box would make the circle an ellipse and the
 * supporting point stop being the farthest point of the ball along v.
 */
export function ballGeometry(vectors, signs, radius, {
  width = 320, height = 260, padding = 26, sharedExtent = null,
} = {}) {
  const best = bestResponse(vectors, signs, radius);
  const arrows = vectors.map((vector, index) => ({
    index,
    sign: signs[index],
    vector,
    signed: vector.map(value => signs[index] * value),
  }));
  /* `sharedExtent` is how a SMALL-MULTIPLE figure makes its panels comparable.
   *
   * Autoscaling each panel independently is right for a single drawing and
   * wrong for three panels captioned "Same lengths": figure 8's first panel
   * has signed sum (2, 0) and the others (1, ±1), so the same unit vector and
   * the same radius-1 ball were drawn at half the size in panel 1 — in a
   * figure whose entire claim is that the lengths and the budget are
   * identical. Worse, the quantity that moved the scale is not drawn in that
   * figure, so a reader had no way to discover why. A caller showing panels
   * side by side passes one extent for all of them; `figureExtent` computes
   * it. */
  const autoExtent = Math.max(
    radius,
    ...arrows.map(arrow => Math.max(Math.abs(arrow.signed[0]), Math.abs(arrow.signed[1]))),
    Math.abs(best.v[0]), Math.abs(best.v[1]),
    1e-6,
  );
  const extent = sharedExtent ?? autoExtent;
  const usable = Math.min(width, height) / 2 - padding;
  const unit = usable / extent;
  const origin = { x: width / 2, y: height / 2 };
  const project = point => ({ x: origin.x + unit * point[0], y: origin.y - unit * point[1] });
  /* Where the supporting point's label goes.
   *
   * The obvious placement -- a few pixels right and up -- puts the text on top
   * of the signed-sum arrow, because the supporting point lies ON that arrow by
   * construction: it is the point of the ball farthest along v. The label is
   * therefore offset PERPENDICULAR to v, on whichever side stays inside the
   * box. The verifier asserts the perpendicular distance, so this cannot
   * silently regress. */
  const supportLabel = (() => {
    if (!best.optimizer || best.v.length !== 2) return null;
    const point = project(best.optimizer);
    const length = norm2(best.v) || 1;
    const perpendicular = { x: -best.v[1] / length, y: best.v[0] / length };
    const offset = 16;
    const candidates = [
      { x: point.x + offset * perpendicular.x, y: point.y - offset * perpendicular.y },
      { x: point.x - offset * perpendicular.x, y: point.y + offset * perpendicular.y },
    ];
    const inside = place => place.x > 26 && place.x < width - 26 && place.y > 14 && place.y < height - 8;
    const chosen = candidates.find(inside) ?? candidates[0];
    return { ...chosen, anchor: chosen.x < point.x ? 'end' : 'start', offset };
  })();
  return {
    width,
    height,
    origin,
    unit,
    extent,
    radiusPixels: radius * unit,
    project,
    autoExtent,
    sharedExtent,
    /* Which drawn arrows land on top of each other. Two identical inputs are
       two coincident lines with two coincident arrowheads, so the panel that
       exists to show the duplicate case shows one arrow; a caller can print a
       multiplicity beside it rather than leave the reader to count nothing. */
    coincidentGroups: (() => {
      const groups = new Map();
      arrows.forEach(arrow => {
        const key = arrow.signed.map(value => Number(value.toFixed(9))).join('|');
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(arrow.index);
      });
      return [...groups.values()].filter(group => group.length > 1);
    })(),
    arrows: arrows.map(arrow => ({ ...arrow, tip: project(arrow.signed), base: origin })),
    sum: { vector: best.v, tip: project(best.v) },
    support: best.optimizer ? { vector: best.optimizer, point: project(best.optimizer) } : null,
    supportLabel,
    best,
  };
}

/**
 * One extent for a whole small-multiple figure: the largest any of its panels
 * would have chosen for itself. Passing this to every panel makes a length
 * drawn in one panel mean the same as the same length drawn in another, which
 * is the precondition for captioning them "same lengths".
 */
export function figureExtent(panels) {
  if (!Array.isArray(panels) || panels.length === 0) throw new RangeError('figureExtent: no panels');
  return Math.max(...panels.map(({ vectors, signs, radius }) =>
    ballGeometry(vectors, signs, radius).autoExtent));
}

/** The l1 diamond |w_1| + |w_2| <= B, as a closed polygon in the same
 *  projection the ball uses. */
export function diamondPoints(radius, geometry) {
  return [[radius, 0], [0, radius], [-radius, 0], [0, -radius]].map(geometry.project);
}

/**
 * Sampled points of one loss curve, for drawing.
 *
 * The sampler is handed the function itself, so the drawn polyline and the
 * applied function cannot drift apart: there is only one definition.
 */
export function curvePoints(fn, { from, to, samples = 96, width = 320, height = 150, padding = 30, valueRange }) {
  if (!(samples >= 2)) throw new RangeError('curvePoints: need at least two samples');
  if (!(to > from)) throw new RangeError('curvePoints: need to > from');
  const values = [];
  for (let step = 0; step <= samples; step += 1) {
    const input = from + ((to - from) * step) / samples;
    values.push([input, fn(input)]);
  }
  const lows = valueRange ? valueRange[0] : Math.min(...values.map(entry => entry[1]));
  const highs = valueRange ? valueRange[1] : Math.max(...values.map(entry => entry[1]));
  const span = highs - lows || 1;
  const plotWidth = width - padding - 12;
  const plotHeight = height - padding - 14;
  const project = ([input, value]) => ({
    x: padding + (plotWidth * (input - from)) / (to - from),
    y: 14 + plotHeight - (plotHeight * (value - lows)) / span,
  });
  // The frame is returned rather than recomputed by each caller. Hand-written
  // axis coordinates in a component were how an axis line once sat fourteen
  // pixels below the curve it was supposed to carry.
  const frame = { left: padding, right: width - 12, top: 14, bottom: 14 + plotHeight };
  return {
    width,
    height,
    padding,
    from,
    to,
    valueRange: [lows, highs],
    values,
    project,
    frame,
    points: values.map(project),
    polyline: values.map(project).map(point => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' '),
  };
}

/**
 * The drawn ramp curve.
 *
 * This exists so that the curve a figure PAINTS and the function a lab APPLIES
 * are the same definition rather than two copies of it. The verifier samples
 * this function and compares it against `rampLoss` over the whole enterable
 * threshold grid; if a figure sampled its own lambda instead, that comparison
 * would be comparing `rampLoss` with itself and could never fail.
 */
export function rampCurve(rho, { from = -0.6, to = 1.5, samples = 240, width = 300, height = 170 } = {}) {
  if (!isFiniteNumber(rho) || rho <= 0) throw new RangeError('rampCurve: rho must be finite and positive');
  return curvePoints(margin => rampLoss([margin], rho)[0], { from, to, samples, width, height, valueRange: [0, 1] });
}

/**
 * Concentric boxes for a chain of nested classes, outermost first.
 *
 * Each box is strictly inside the one before it on all four sides, and each has
 * room for its own label line. The verifier asserts containment rather than
 * trusting the arithmetic, because a nesting picture whose boxes cross is
 * drawing the opposite of what it claims.
 */
export function nestedBoxLayout(count, { width = 320, inset = 10, labelHeight = 17, padding = 5 } = {}) {
  if (!Number.isInteger(count) || count < 1) throw new RangeError('nestedBoxLayout: need at least one box');
  const height = 2 * padding + count * labelHeight + (count - 1) * inset;
  const boxes = [];
  for (let level = 0; level < count; level += 1) {
    boxes.push({
      level,
      x: padding + level * inset,
      y: padding + level * inset,
      width: width - 2 * (padding + level * inset),
      height: height - 2 * (padding + level * inset),
      labelX: padding + level * inset + 7,
      labelY: padding + level * inset + labelHeight - 4,
    });
  }
  return { width, height, boxes };
}

/**
 * A histogram of training margins with an explicit bin edge at 0 and at rho,
 * so the two quantities the ramp cares about are boundaries rather than
 * something a bin can straddle.
 */
export function marginHistogram(margins, { rho, bins = 18, from, to } = {}) {
  requireFinite(margins, 'marginHistogram');
  if (!isFiniteNumber(rho) || rho <= 0) throw new RangeError('marginHistogram: rho must be positive');
  const low = from ?? Math.min(...margins, -rho);
  const high = to ?? Math.max(...margins, rho);
  const edges = [];
  for (let index = 0; index <= bins; index += 1) edges.push(low + ((high - low) * index) / bins);
  // Force exact edges at 0 and rho by snapping the nearest interior edge.
  [0, rho].forEach(target => {
    if (target <= low || target >= high) return;
    let nearest = 1;
    for (let index = 1; index < bins; index += 1) {
      if (Math.abs(edges[index] - target) < Math.abs(edges[nearest] - target)) nearest = index;
    }
    edges[nearest] = target;
  });
  edges.sort((a, b) => a - b);
  const counts = new Array(bins).fill(0);
  margins.forEach(margin => {
    let index = edges.findIndex((edge, position) => position < bins && margin >= edge && margin < edges[position + 1]);
    if (margin >= edges[bins]) index = bins - 1;
    if (margin < edges[0]) index = 0;
    counts[index] += 1;
  });
  return {
    edges,
    counts,
    total: margins.length,
    belowZero: margins.filter(margin => margin <= 0).length,
    insideRamp: margins.filter(margin => margin > 0 && margin < rho).length,
    aboveRho: margins.filter(margin => margin >= rho).length,
  };
}

/**
 * The three bound terms as one stacked bar, plus the trivial ceiling 1 drawn on
 * the SAME scale. If the stack is taller than the ceiling line, the reader can
 * see the bound is vacuous; clipping the stack at 1 would hide exactly that.
 */
export function stackedBoundLayout(rows, {
  width = 320, rowHeight = 30, gap = 10, left = 46, right = 12, labelGutter = 46,
} = {}) {
  if (!Array.isArray(rows) || rows.length === 0) throw new RangeError('stackedBoundLayout: no rows');
  const maximum = Math.max(1, ...rows.map(row => row.empiricalRamp + row.complexityAddend + row.confidence));
  /* The gutter is reserved for the total printed at the end of each bar. The
   * first version scaled the bars across the full width and then placed that
   * number four pixels past the longest bar, which put it outside the SVG --
   * two totals were drawn off the edge of the drawing. The space the label
   * needs is part of the layout, not an afterthought. */
  const usable = width - left - right - labelGutter;
  const scale = usable / maximum;
  return {
    width,
    height: rows.length * (rowHeight + gap) + gap + 18,
    left,
    scale,
    maximum,
    labelGutter,
    ceilingX: left + scale,
    rows: rows.map((row, index) => {
      const y = gap + index * (rowHeight + gap) + 18;
      const segments = [
        { key: 'ramp', value: row.empiricalRamp },
        { key: 'complexity', value: row.complexityAddend },
        { key: 'confidence', value: row.confidence },
      ];
      let cursor = left;
      const total = segments.reduce((sum, segment) => sum + segment.value, 0);
      return {
        ...row,
        y,
        height: rowHeight,
        total,
        labelX: left + total * scale + 5,
        segments: segments.map(segment => {
          const x = cursor;
          cursor += segment.value * scale;
          return { ...segment, x, width: segment.value * scale };
        }),
      };
    }),
  };
}

/**
 * The Monte-Carlo figure: for each recorded draw count, the estimate, the
 * one-sided correction as a bar reaching to its endpoint, and the exact value
 * as a separate reference line.
 *
 * These are FOUR separate statements, each with its own failure allowance eta.
 * Drawing them as one confidence band would assert something none of them says,
 * so the layout returns four independent intervals and one reference.
 */
export function monteCarloLayout(rows, exact, { width = 320, rowHeight = 26, gap = 12, left = 52, right = 16 } = {}) {
  if (!Array.isArray(rows) || rows.length === 0) throw new RangeError('monteCarloLayout: no rows');
  const maximum = Math.max(exact, ...rows.map(row => row.endpoint)) * 1.12;
  const usable = width - left - right;
  const place = value => left + (usable * value) / maximum;
  return {
    width,
    height: rows.length * (rowHeight + gap) + gap + 26,
    left,
    maximum,
    place,
    exact,
    exactX: place(exact),
    rows: rows.map((row, index) => ({
      ...row,
      y: gap + 20 + index * (rowHeight + gap),
      height: rowHeight,
      estimateX: place(row.estimate),
      endpointX: place(row.endpoint),
    })),
  };
}

/** The four data roles as one proportional strip. Widths are the actual row
 *  counts, so the picture cannot disagree with the table beside it. */
export function rolesStrip(roles, { width = 320, height = 26 } = {}) {
  const total = roles.reduce((sum, role) => sum + role.count, 0);
  if (!(total > 0)) throw new RangeError('rolesStrip: empty allocation');
  let cursor = 0;
  return {
    width,
    height,
    total,
    parts: roles.map(role => {
      const x = cursor;
      const partWidth = (width * role.count) / total;
      cursor += partWidth;
      return { ...role, x, width: partWidth, share: role.count / total };
    }),
  };
}

/**
 * The ghost-sample figure: n paired columns, each pair with its own sign, and
 * the difference the sign multiplies.
 *
 * The figure illustrates the swap OPERATION on fixed rows. It is not a claim
 * that any particular pair of drawn numbers has the same supremum after
 * swapping; that statement is about the distribution of iid pairs.
 */
export function ghostPairLayout(pairs, { width = 320, top = 30, laneGap = 54, columnGap = 6, left = 34 } = {}) {
  if (!Array.isArray(pairs) || pairs.length === 0) throw new RangeError('ghostPairLayout: no pairs');
  const usable = width - left - 12;
  const columnWidth = (usable - columnGap * (pairs.length - 1)) / pairs.length;
  return {
    width,
    height: top + 2 * laneGap + 40,
    columnWidth,
    columns: pairs.map((pair, index) => {
      const x = left + index * (columnWidth + columnGap);
      return {
        index,
        x,
        width: columnWidth,
        sign: pair.sign,
        sample: pair.sample,
        ghost: pair.ghost,
        // The swap exchanges which value sits in which lane; the signed
        // difference is what the proof's sum actually contains.
        topValue: pair.sign === 1 ? pair.ghost : pair.sample,
        bottomValue: pair.sign === 1 ? pair.sample : pair.ghost,
        difference: pair.sign * (pair.ghost - pair.sample),
        topY: top,
        bottomY: top + laneGap,
      };
    }),
  };
}

/** The convex-hull picture of section 9: base prediction vectors, an arbitrary
 *  mixture, and the projection of each onto one sign direction. */
export function hullLayout(basePoints, weights, direction, { width = 320, height = 220, padding = 30 } = {}) {
  requireMatrix(basePoints, 'hullLayout');
  const mixture = convexMixture(basePoints, weights);
  const all = [...basePoints, mixture];
  const extent = Math.max(...all.map(point => Math.max(Math.abs(point[0]), Math.abs(point[1]))), 1e-6);
  const unit = (Math.min(width, height) / 2 - padding) / extent;
  const origin = { x: width / 2, y: height / 2 };
  const project = point => ({ x: origin.x + unit * point[0], y: origin.y - unit * point[1] });
  const projections = all.map(point => dot(point, direction) / norm2(direction));
  /* Label placement.
   *
   * A vertex label nudged by a fixed offset lands on one of the hull's own
   * edges -- the browser curve sampler caught exactly that. Each vertex label
   * is instead pushed radially OUTWARD from the polygon's centroid, which puts
   * it outside the polygon for any convex hull, and the verifier proves that
   * with a point-in-polygon test rather than trusting the arithmetic.
   *
   * The interior mixture point gets no in-plot label at all. Pushing it inward
   * cannot be made safe: for some weightings the triangle's inradius is smaller
   * than the offset, so the label crosses the opposite edge. It is named in a
   * legend strip below the plot, where nothing is drawn. */
  const projected = basePoints.map(project);
  const centroid = {
    x: projected.reduce((sum, point) => sum + point.x, 0) / projected.length,
    y: projected.reduce((sum, point) => sum + point.y, 0) / projected.length,
  };
  const push = (point, distance) => {
    const dx = point.x - centroid.x;
    const dy = point.y - centroid.y;
    const length = Math.hypot(dx, dy) || 1;
    return {
      x: point.x + (distance * dx) / length,
      y: point.y + (distance * dy) / length,
      anchor: dx >= 0 ? 'start' : 'end',
    };
  };
  return {
    width,
    height,
    origin,
    unit,
    project,
    basePoints,
    mixture,
    direction,
    points: all.map(project),
    centroid,
    vertexLabels: projected.map(point => push(point, 17)),
    projections,
    mixtureProjection: projections[projections.length - 1],
    bestBaseProjection: Math.max(...projections.slice(0, -1)),
    /* Which base vector attains the best projection, and how far along the
       direction the drawn ray should run. Both are needed to DRAW the
       comparison the figure claims, rather than only print it. */
    bestBaseIndex: projections.slice(0, -1).indexOf(Math.max(...projections.slice(0, -1))),
    reach: Math.max(...projections, 0) * 1.15 + 0.1,
  };
}
