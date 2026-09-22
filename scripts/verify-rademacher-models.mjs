// Bounded independent checks of the Rademacher browser models against the
// content packet's recorded calculations, the manuscript's stated values,
// analytic identities and a second route for every quantity.
//
// Every assertion here is written so that it can fail: nothing is compared with
// itself, no loop is allowed to inspect an empty subject set, and every group
// asserts how many subjects it actually saw.
//
// The second routes, by name, because "checked" means nothing without them:
//
//   * every empirical complexity is recomputed in EXACT BigInt rational
//     arithmetic -- integer dot products, an integer maximum per pattern, an
//     integer sum, and one final division -- so a floating-point association
//     cannot be what makes the answer right;
//   * the threshold restrictions are rebuilt from a different characterisation
//     (a sign row is a positive-threshold restriction exactly when it is
//     non-decreasing along the sorted inputs and constant on ties) rather than
//     by sweeping cutoffs;
//   * the Euclidean ball is recomputed through the KERNEL path, using the Gram
//     matrix X Xᵀ, which shares no code with the vector path;
//   * the two-point kernel has a closed form, B(sqrt(2+2r) + sqrt(2-2r))/4,
//     which is checked against the enumeration at every enterable r;
//   * Massart's bound is recomputed by evaluating the exponential-moment
//     expression ln(M)/lambda + lambda A^2/2 at the optimising lambda;
//   * Sauer's count is recomputed by Pascal's triangle rather than by the
//     multiplicative binomial recurrence;
//   * the real experiment's margins, mistake counts, ramp means and bound
//     components are recomputed from the served CSV rather than read from the
//     module, and compared against the packet's own recorded values.
//
// Two properties are swept over an entire enterable grid rather than sampled:
// that the ramp curve a figure DRAWS is the ramp function a lab APPLIES, at
// every margin and every threshold; and that a "did it move?" verdict says
// unchanged exactly when the displayed values are equal.
//
// Run: node scripts/verify-rademacher-models.mjs [--no-evidence]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  absoluteComplexity, achievedCorrelation, ballGeometry, bestResponse, compareOutcome, confidenceAddend,
  convexMixture, curvePoints, diamondPoints, displayValue, distinctRows, dot, empiricalComplexity,
  evaluatePredictor, featureEnergy, ghostPairLayout, hingeMarginLoss, hoeffdingCorrection, hullLayout,
  isPositiveSemidefinite, kernelComplexity, l1BestResponse, limits, linearComplexity, logisticMarginLoss,
  logisticMarginSlope, mapFeatures, marginBound, marginHistogram, massartBound, massartSignBound, mistakeRows,
  figureExtent, gapOutcome, monteCarloEndpoint, monteCarloLayout, nestedBoxLayout, norm2, numberLineLayout,
  predictSign, rampLoss, rolesStrip,
  rampCurve, sauerCount, selectByValidation, sigmoid, sigmoidSlope, signPatterns, signedSum,
  stackedBoundLayout, thresholdRows,
} from '../src/learn/data/rademacher-models.js';
import {
  duplicateFeatureGroups, experimentSettings, fittedModels, majorityBaseline, observations, provenance,
  recordedFixtures, recordedMonteCarlo, representation, roles, selection, zeroCandidate,
} from '../src/learn/data/rademacher-data.js';
import { rademacherExcerpts, rademacherPrograms } from '../src/learn/data/rademacher-examples.js';

const packetDirectory = 'docs/teaching/drafts/rademacher-complexity-generalization-bounds';
const checked = JSON.parse(fs.readFileSync(`${packetDirectory}/checked-results.json`, 'utf8'));
const experiment = JSON.parse(fs.readFileSync(`${packetDirectory}/experiment-results.json`, 'utf8'));
const manuscript = fs.readFileSync(`${packetDirectory}/lesson.md`, 'utf8');

/* A provisional FAILING record, written before any assertion runs.
 *
 * This file is all top-level code, so a thrown assertion ends the process
 * before the real evidence is written — leaving the PREVIOUS run's passing
 * record on disk beside a tree that now fails. Stamping an incomplete record
 * first means a crash leaves an artefact that says so, and the successful path
 * overwrites it at the end. The independent review found the same class in its
 * Python form (S1); this closes it here rather than waiting to be told. */
const evidencePath = 'docs/teaching/evidence/rademacher-models.json';
const writesEvidence = !process.argv.includes('--no-evidence');
if (writesEvidence) {
  fs.mkdirSync('docs/teaching/evidence', { recursive: true });
  fs.writeFileSync(evidencePath, JSON.stringify({
    checkedAt: new Date().toISOString(),
    verifier: 'scripts/verify-rademacher-models.mjs',
    status: 'incomplete — this run started and did not reach its end',
    passed: false,
  }, null, 2) + '\n');
}

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) => {
  assert(typeof actual === 'number' && Number.isFinite(actual), `${label}: ${actual} is not a finite number`);
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} versus ${expected}`);
};
const vector = (actual, expected, label, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length, `${label}: length ${actual.length} versus ${expected.length}`);
  actual.forEach((value, index) => close(value, expected[index], `${label}[${index}]`, tolerance));
};
/** A loop must have something to look at, or it asserts nothing. */
const nonEmpty = (collection, expected, label) => {
  assert(collection.length > 0, `${label}: the subject set is empty, so nothing was checked`);
  if (expected !== undefined) {
    assert.equal(collection.length, expected, `${label}: saw ${collection.length}, expected ${expected}`);
  }
};
/** A call that must be refused. A silent default here would be a wrong answer. */
const refuses = (call, label) => {
  assert.throws(call, RangeError, `${label}: should have been refused`);
  record('refused input');
};

/* ===================== second route 1 · exact rational enumeration in BigInt */

/**
 * The empirical complexity as an exact rational number.
 *
 * Every entry is scaled to an integer, the dot products and the per-pattern
 * maximum are integer operations, and the sum is a BigInt. Only the final
 * division touches a float. This shares no arithmetic with the module's
 * floating-point path, so agreement is agreement between methods.
 */
function exactComplexity(rows, scale = 1000n) {
  const n = rows[0].length;
  const scaled = rows.map(row => row.map(value => {
    const lifted = BigInt(Math.round(value * Number(scale)));
    assert(Math.abs(Number(lifted) / Number(scale) - value) < 1e-12,
      `exactComplexity: ${value} is not representable at 1/${scale}`);
    return lifted;
  }));
  let total = 0n;
  const patterns = 2 ** n;
  for (let code = 0; code < patterns; code += 1) {
    let best = null;
    for (const row of scaled) {
      let sum = 0n;
      for (let index = 0; index < n; index += 1) {
        sum += ((code >> index) & 1 ? 1n : -1n) * row[index];
      }
      if (best === null || sum > best) best = sum;
    }
    total += best;
  }
  return Number(total) / (Number(scale) * n * patterns);
}

/* ============ second route 2 · threshold rows by a different characterisation */

/**
 * Every sign row that some positive threshold produces, found by testing all
 * 2^n sign rows against the defining property instead of by sweeping cutoffs:
 * sorted by input, the row must be a block of −1 followed by a block of +1,
 * and two equal inputs must receive the same sign.
 */
function thresholdRowsByProperty(inputs) {
  const order = inputs.map((value, index) => index).sort((a, b) => inputs[a] - inputs[b]);
  const rows = [];
  for (const pattern of signPatterns(inputs.length)) {
    const sorted = order.map(index => pattern[index]);
    let monotone = true;
    for (let index = 1; index < sorted.length; index += 1) {
      if (sorted[index] < sorted[index - 1]) monotone = false;
      if (inputs[order[index]] === inputs[order[index - 1]] && sorted[index] !== sorted[index - 1]) monotone = false;
    }
    if (monotone) rows.push(pattern);
  }
  return rows;
}

/** Ray casting. Used to prove a drawn label is outside (or inside) a polygon,
 *  rather than trusting the offset that put it there. */
function insidePolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const intersects = (polygon[i].y > point.y) !== (polygon[j].y > point.y)
      && point.x < ((polygon[j].x - polygon[i].x) * (point.y - polygon[i].y))
        / (polygon[j].y - polygon[i].y) + polygon[i].x;
    if (intersects) inside = !inside;
  }
  return inside;
}

/** Shortest distance from a point to a polygon's boundary. */
function distanceToPolygon(point, polygon) {
  let best = Infinity;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const dx = polygon[i].x - polygon[j].x;
    const dy = polygon[i].y - polygon[j].y;
    const lengthSquared = dx * dx + dy * dy;
    const t = lengthSquared === 0 ? 0
      : Math.max(0, Math.min(1, ((point.x - polygon[j].x) * dx + (point.y - polygon[j].y) * dy) / lengthSquared));
    best = Math.min(best, Math.hypot(point.x - (polygon[j].x + t * dx), point.y - (polygon[j].y + t * dy)));
  }
  return best;
}

let hullLabelChecks = 0;

const lexicographic = (a, b) => {
  for (let index = 0; index < a.length; index += 1) if (a[index] !== b[index]) return a[index] - b[index];
  return 0;
};

/* ============================================ §1 · the small finite classes */

const THREE = [-1, 0, 1];
const thresholds = thresholdRows(THREE);
assert.deepEqual(thresholds, [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]],
  'three ordered inputs give exactly these four rows, in canonical order');
assert.deepEqual(thresholds, thresholdRowsByProperty(THREE).sort(lexicographic),
  'and the same four rows arrive from the monotone-block characterisation');
record('threshold restriction');

// Both extreme cutoffs matter: dropping them is a wrong answer, not a rounding.
const withoutExtremes = thresholds.filter(row => row.some(value => value === 1) && row.some(value => value === -1));
nonEmpty(withoutExtremes, 2, 'interior threshold rows');
assert(empiricalComplexity(withoutExtremes).complexity < empiricalComplexity(thresholds).complexity,
  'dropping the two constant rows changes the answer, which is why they are included');
record('threshold restriction');

const cases = [
  { name: 'singleton', rows: [[1, 1, 1]], expected: 0, packet: checked.singleton },
  { name: 'constants', rows: [[-1, -1, -1], [1, 1, 1]], expected: 1 / 2, packet: checked.constants },
  { name: 'thresholds', rows: thresholds, expected: 2 / 3, packet: checked.thresholds },
  {
    name: 'two orientations',
    rows: thresholdRows(THREE, { bothOrientations: true }),
    expected: 5 / 6,
    packet: checked.two_orientations,
  },
  { name: 'full cube', rows: signPatterns(3), expected: 1, packet: checked.all_labels },
];
nonEmpty(cases, 5, 'finite classes');
for (const entry of cases) {
  const model = empiricalComplexity(entry.rows);
  close(model.complexity, entry.expected, `${entry.name}: the stated exact value`);
  close(model.complexity, exactComplexity(entry.rows), `${entry.name}: exact BigInt enumeration agrees`, 0);
  close(model.complexity, entry.packet.complexity, `${entry.name}: the packet's recorded value`);
  assert.equal(model.rows.length, entry.packet.values.length, `${entry.name}: the same number of rows`);
  assert.deepEqual(model.patterns.map(pattern => [...pattern]), entry.packet.signs,
    `${entry.name}: the same sign patterns, in the same order`);
  model.correlations.forEach((scores, index) =>
    vector(scores, entry.packet.correlations[index], `${entry.name}: correlations for pattern ${index}`));
  vector(model.maxima, entry.packet.maxima, `${entry.name}: per-pattern maxima`);
  assert.deepEqual(model.firstWinner, entry.packet.best_indices, `${entry.name}: the winning row index per pattern`);
  close(model.maxAfterAveraging, entry.packet.max_after_averaging, `${entry.name}: the wrong-order diagnostic`);
  record('finite class');
}

// The wrong order is exactly zero for every one of these, because each sign set
// is closed under negation. That is a property, not a coincidence of rounding.
for (const entry of cases) {
  /* Exactly zero as a theorem; a few units in the last place as a sum of eight
     doubles. The exactness is asserted in the BigInt route above, where the
     arithmetic is integral; here the claim is that no row has a systematically
     nonzero average correlation with fair signs. */
  close(empiricalComplexity(entry.rows).maxAfterAveraging, 0,
    `${entry.name}: averaging first gives zero to within floating-point dust`, 1e-15);
  record('order of operations');
}
assert(empiricalComplexity(thresholds).complexity > empiricalComplexity(thresholds).maxAfterAveraging,
  'and the two orders genuinely differ, so the comparison is not vacuous');
record('order of operations');

// Nesting. Each class contains the previous one, so its value cannot fall.
for (let index = 1; index < cases.length; index += 1) {
  const smaller = new Set(cases[index - 1].rows.map(row => row.join('|')));
  const larger = new Set(cases[index].rows.map(row => row.join('|')));
  [...smaller].forEach(key => assert(larger.has(key),
    `${cases[index].name} contains every row of ${cases[index - 1].name}`));
  assert(empiricalComplexity(cases[index].rows).complexity
    >= empiricalComplexity(cases[index - 1].rows).complexity - 1e-15,
  `${cases[index].name} cannot be smaller than ${cases[index - 1].name}`);
  record('nesting');
}

/* ================================================ §2 · nulls and properties */

const base = empiricalComplexity(thresholds).complexity;
const nulls = [
  { name: 'duplicate the first row', rows: [...thresholds, thresholds[0]] },
  { name: 'duplicate every row', rows: [...thresholds, ...thresholds] },
  { name: 'reverse the row order', rows: [...thresholds].reverse() },
  { name: 'add a fixed 7 to every output', rows: thresholds.map(row => row.map(value => value + 7)) },
  { name: 'subtract a fixed 0.5 from every output', rows: thresholds.map(row => row.map(value => value - 0.5)) },
  {
    name: 'add a convex average',
    rows: [...thresholds, convexMixture(thresholds, [0.25, 0, 0, 0.75])],
  },
  {
    name: 'add a different convex average',
    rows: [...thresholds, convexMixture(thresholds, [0.1, 0.2, 0.3, 0.4])],
  },
];
nonEmpty(nulls, 7, 'exact nulls');
for (const entry of nulls) {
  /* The float path can differ by one unit in the last place -- adding a fixed 7
     to every output and taking it back out again is not bit-exact in binary.
     The EXACT claim is made in the rational route, where it must hold to the
     last digit; the float claim is that the displayed answer cannot move. */
  close(empiricalComplexity(entry.rows).complexity, base, `${entry.name} is a null in floating point`, 1e-15);
  close(exactComplexity(entry.rows), exactComplexity(thresholds),
    `${entry.name} is an EXACT null in rational arithmetic`, 0);
  assert.equal(displayValue(empiricalComplexity(entry.rows).complexity), displayValue(base),
    `${entry.name}: and the displayed value does not move, which is what the lab's verdict compares`);
  record('exact null');
}
// A permutation of the observation columns is a null only when the signs travel
// with them; the quantity averages over all sign patterns, so it is a null here
// unconditionally, and the check says which statement it is making.
const permuted = thresholds.map(row => [row[2], row[0], row[1]]);
close(empiricalComplexity(permuted).complexity, base, 'permuting the observation columns is an exact null', 0);
record('exact null');

for (const factor of [0.5, 2, 3, 0.25]) {
  close(empiricalComplexity(thresholds.map(row => row.map(value => factor * value))).complexity, factor * base,
    `scaling every output by ${factor} scales the answer by ${factor}`);
  record('scaling');
}
// Negating every row gives the MIRROR class, whose value is the same because
// the sign distribution is symmetric. Exact in the rational route.
const mirrored = thresholds.map(row => row.map(value => -value));
close(exactComplexity(mirrored), exactComplexity(thresholds),
  'negating every row is an exact null, because the sign distribution is symmetric', 0);
close(empiricalComplexity(mirrored).complexity, base, 'and the float path agrees to within dust', 1e-15);
record('scaling');

// The absolute-value convention is a DIFFERENT measurement, and it equals the
// signed value of the class closed under negation. That identity is the reason
// the lesson can say what the other convention measures.
close(absoluteComplexity([[1, 1, 1]]).complexity, checked.abs_changes_singleton,
  'the absolute convention on the singleton matches the packet');
close(absoluteComplexity([[1, 1, 1]]).complexity, empiricalComplexity([[1, 1, 1], [-1, -1, -1]]).complexity,
  'and equals the signed value of the class with the rule and its negative', 0);
record('convention');
for (const entry of cases) {
  const symmetric = [...entry.rows, ...entry.rows.map(row => row.map(value => -value))];
  close(absoluteComplexity(entry.rows).complexity, empiricalComplexity(distinctRows(symmetric).rows).complexity,
    `${entry.name}: the absolute convention measures the negation-closed class`, 0);
  record('convention');
}

// Duplicate detection: the count of DISTINCT rows is what a bound may use.
const withDuplicates = [...thresholds, thresholds[0], thresholds[2], thresholds[0]];
const grouped = distinctRows(withDuplicates);
assert.equal(grouped.rows.length, 4, 'four distinct rows survive');
assert.equal(grouped.duplicated.length, 2, 'two duplicate groups are named');
assert.deepEqual(grouped.duplicated.map(group => group.length).sort(), [2, 3],
  'and their sizes are reported, so a bound can use the distinct count');
record('duplicates');

/* ============================================ §3 · the loss class identity */

const labelSets = [[1, -1, 1], [1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, -1], [-1, 1, 1]];
nonEmpty(labelSets, 8, 'label vectors');
for (const labels of labelSets) {
  const losses = mistakeRows(thresholds, labels);
  close(empiricalComplexity(losses).complexity, base / 2,
    `for labels ${labels.join(',')} the mistake class is exactly half the predictor class`, 0);
  losses.forEach(row => row.forEach(value =>
    assert(value === 0 || value === 1, 'a mistake loss is 0 or 1, exactly')));
  record('loss class');
}
close(empiricalComplexity(mistakeRows(thresholds, [1, -1, 1])).complexity, checked.classification_loss.complexity,
  'and it matches the packet');
record('loss class');
refuses(() => mistakeRows(thresholds, [1, 0, 1]), 'a label of 0');
refuses(() => mistakeRows([[0.5, 0.5, 0.5]], [1, 1, 1]), 'a non-sign hypothesis under the exact identity');

/* ================================== §5 · Euclidean geometry, by two routes */

/** The Euclidean ball through the KERNEL path: ||sum sigma_i x_i||^2 is the
 *  quadratic form of the Gram matrix X Xᵀ. Different code, same theorem. */
function complexityThroughGram(vectors, radius) {
  const gram = vectors.map(a => vectors.map(b => dot(a, b)));
  return kernelComplexity(gram, radius).complexity;
}

const geometryCases = [
  { name: 'parallel', vectors: [[1, 0], [1, 0]], radius: 1, expected: 0.5, packet: checked.geometry.parallel },
  {
    name: 'orthogonal', vectors: [[1, 0], [0, 1]], radius: 1, expected: Math.SQRT1_2,
    packet: checked.geometry.orthogonal,
  },
  {
    name: 'rotated', vectors: [[0, 1], [-1, 0]], radius: 1, expected: Math.SQRT1_2,
    packet: checked.geometry.rotated,
  },
  {
    name: 'doubled budget', vectors: [[1, 0], [0, 1]], radius: 2, expected: Math.SQRT2,
    packet: checked.geometry.doubled_budget,
  },
  { name: 'zero sample', vectors: [[0]], radius: 1, expected: 0, packet: checked.geometry.zero_sample },
  { name: 'appended nonzero', vectors: [[0], [1]], radius: 1, expected: 0.5, packet: checked.geometry.added_nonzero },
  { name: 'practice 3', vectors: [[3, 0], [0, 4]], radius: 2, expected: 5, packet: checked.practice.linear_3_4 },
];
nonEmpty(geometryCases, 7, 'geometry cases');
for (const entry of geometryCases) {
  const model = linearComplexity(entry.vectors, entry.radius);
  close(model.complexity, entry.expected, `${entry.name}: the stated exact value`);
  close(model.complexity, entry.packet.complexity, `${entry.name}: the packet's value`);
  close(model.complexity, complexityThroughGram(entry.vectors, entry.radius),
    `${entry.name}: the kernel route agrees with the vector route`);
  close(model.energyUpper, entry.packet.energy_upper, `${entry.name}: the energy bound`);
  assert(model.complexity <= model.energyUpper + 1e-12,
    `${entry.name}: the exact value never exceeds the energy bound`);
  model.sums.forEach((sum, index) => vector(sum, entry.packet.signed_sums[index],
    `${entry.name}: signed sum ${index}`));
  vector(model.maxima, entry.packet.maxima, `${entry.name}: per-pattern optima`);
  record('Euclidean geometry');
}

// The exact value and the bound coincide exactly when every signed sum has the
// same length. That is the statement the lesson makes, so it is asserted as an
// iff rather than shown on one example.
let coincidenceCases = 0;
for (const entry of geometryCases) {
  const model = linearComplexity(entry.vectors, entry.radius);
  const lengths = model.sums.map(norm2);
  const allEqual = lengths.every(value => Math.abs(value - lengths[0]) <= 1e-12);
  const coincides = Math.abs(model.complexity - model.energyUpper) <= 1e-12;
  assert.equal(allEqual, coincides,
    `${entry.name}: the bound is tight exactly when every signed sum has the same length`);
  coincidenceCases += 1;
}
assert.equal(coincidenceCases, 7, 'the coincidence property was checked on every geometry case');
record('Euclidean geometry');

// Rotation is a change of coordinates and must leave the answer alone, at every
// angle, not at one.
let rotationCases = 0;
for (let step = 0; step < 24; step += 1) {
  const angle = (step * Math.PI) / 12;
  const rotate = ([x, y]) => [x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle)];
  for (const entry of geometryCases.filter(item => item.vectors[0].length === 2)) {
    close(linearComplexity(entry.vectors.map(rotate), entry.radius).complexity,
      linearComplexity(entry.vectors, entry.radius).complexity,
      `${entry.name}: rotation by ${step}/24 turn is a null`, 1e-12);
    rotationCases += 1;
  }
}
assert(rotationCases >= 120, `only ${rotationCases} rotation nulls ran`);
record('rotation null');

// The best response, at every sign pattern of every case, including degenerate.
let responseCases = 0;
for (const entry of geometryCases) {
  for (const pattern of signPatterns(entry.vectors.length)) {
    const response = bestResponse(entry.vectors, pattern, entry.radius);
    const v = signedSum(entry.vectors, pattern);
    vector(response.v, v, `${entry.name}: the signed sum`);
    close(response.optimum, (entry.radius * norm2(v)) / entry.vectors.length,
      `${entry.name}: the optimum is B||v||/n`);
    if (response.optimizer) {
      close(norm2(response.optimizer), entry.radius, `${entry.name}: the optimiser sits on the boundary`);
      close(dot(response.optimizer, v) / entry.vectors.length, response.optimum,
        `${entry.name}: and it attains the optimum`);
      // No feasible coefficient beats it. Swept over a grid of the ball.
      for (let angle = 0; angle < 16 && v.length === 2; angle += 1) {
        const theta = (angle * Math.PI) / 8;
        const candidate = [entry.radius * Math.cos(theta), entry.radius * Math.sin(theta)];
        assert(dot(candidate, v) / entry.vectors.length <= response.optimum + 1e-9,
          `${entry.name}: no boundary coefficient beats the supporting point`);
      }
    } else {
      close(response.optimum, 0, `${entry.name}: a zero signed sum gives optimum zero`, 0);
      assert.equal(response.optimizerUnique, false, `${entry.name}: and no unique maximiser`);
      assert(typeof response.degenerate === 'string', `${entry.name}: the degenerate case names itself`);
    }
    responseCases += 1;
  }
}
assert.equal(responseCases, 26, `expected 26 best-response cases, saw ${responseCases}`);
record('best response');

// Budget zero: only the origin is feasible, every value is zero, and the model
// says the maximiser is not unique-by-direction but forced by the budget.
for (const entry of geometryCases) {
  for (const pattern of signPatterns(entry.vectors.length)) {
    const response = bestResponse(entry.vectors, pattern, 0);
    close(response.optimum, 0, 'a zero budget gives a zero optimum', 0);
    assert.equal(response.optimizerUnique, true, 'the singleton origin is the unique maximizer');
    assert.deepEqual(response.optimizer, response.v.map(() => 0), 'the unique optimizer is the origin even when the signed sum is zero');
  }
  record('degenerate budget');
}

// A learner's own coefficient: feasibility is reported, never clamped.
const proposals = [[0, 0], [1, 0], [0.5, 0.5], [3, 3], [-1, 0], [0.7071, 0.7071]];
nonEmpty(proposals, 6, 'candidate coefficients');
for (const candidate of proposals) {
  const result = achievedCorrelation([[1, 0], [0, 1]], [1, 1], candidate, 1);
  assert.equal(result.feasible, norm2(candidate) <= 1 + 1e-12,
    `feasibility of (${candidate}) is reported from its own norm`);
  close(result.achieved, dot(candidate, result.v) / 2, 'the achieved value is the actual dot product');
  assert(result.shortfall >= -1e-12 || !result.feasible,
    'a feasible candidate never exceeds the supremum');
  record('candidate coefficient');
}

/* ================================================= §5 · kernel, closed form */

/** The closed form for [[1, r], [r, 1]]: two of the four patterns give
 *  2 + 2r and two give 2 - 2r, so the average is B(sqrt(2+2r) + sqrt(2-2r))/4. */
const twoPointKernelClosedForm = (r, radius) =>
  (radius * (Math.sqrt(2 + 2 * r) + Math.sqrt(2 - 2 * r))) / 4;

let kernelCases = 0;
for (let step = 0; step <= 200; step += 1) {
  const r = -1 + step / 100;
  for (const radius of [0, 0.5, 1, 2, 4]) {
    const model = kernelComplexity([[1, r], [r, 1]], radius);
    close(model.complexity, twoPointKernelClosedForm(r, radius),
      `kernel r=${r}, B=${radius}: the closed form agrees with the enumeration`, 1e-12);
    close(model.traceUpper, (radius * Math.SQRT2) / 2, `kernel r=${r}, B=${radius}: the trace bound is constant in r`);
    assert(model.complexity <= model.traceUpper + 1e-12, `kernel r=${r}: the exact value is under the trace bound`);
    // r and -r give the same sign-averaged answer, though the individual
    // quadratic forms interchange.
    const mirrored = kernelComplexity([[1, -r], [-r, 1]], radius);
    close(mirrored.complexity, model.complexity, `kernel r=${r}: replacing r by -r is a null`, 1e-12);
    kernelCases += 1;
  }
}
assert(kernelCases >= 1000, `only ${kernelCases} kernel cases ran`);
record('kernel');

for (const [key, entry] of Object.entries(checked.kernels)) {
  const r = Number(key);
  close(kernelComplexity([[1, r], [r, 1]], 1).complexity, entry.complexity, `the packet's kernel value at r=${key}`);
  close(kernelComplexity([[1, r], [r, 1]], 1).traceUpper, entry.trace_upper, `and its trace bound at r=${key}`);
  record('kernel');
}
vector(recordedFixtures.kernelComplexities,
  recordedFixtures.kernelSimilarities.map(r => kernelComplexity([[1, r], [r, 1]], 1).complexity),
  'the generated module\'s kernel values are the model\'s values');
record('kernel');

// A matrix that is not positive semidefinite is not a Gram matrix, and is
// refused rather than given a value from the square root of a negative number.
refuses(() => kernelComplexity([[1, 2], [2, 1]], 1), 'an indefinite matrix');
refuses(() => kernelComplexity([[1, 0.5], [0.4, 1]], 1), 'an asymmetric matrix');
refuses(() => kernelComplexity([[1, 0, 0], [0, 1, 0]], 1), 'a non-square matrix');
assert.equal(isPositiveSemidefinite([[1, 1], [1, 1]]), true, 'a singular PSD matrix is accepted');
assert.equal(isPositiveSemidefinite([[0, 0], [0, 0]]), true, 'the zero matrix is accepted');
assert.equal(isPositiveSemidefinite([[0, 1], [1, 0]]), false, 'a zero pivot with a nonzero row is refused');
assert.equal(isPositiveSemidefinite([[-1e-14, 0], [0, 1]]), true, 'a tiny negative from roundoff is tolerated');
assert.equal(isPositiveSemidefinite([[-0.001, 0], [0, 1]]), false, 'a genuinely negative direction is not');
record('kernel refusal');

/* ====================================================== §5 · l1 geometry */

let l1Cases = 0;
for (const first of [[1, 0], [0.9, 0.4], [0.3, 0.8], [-0.5, 0.5], [0, 0]]) {
  for (const second of [[1, 0], [0.3, 0.8], [0, 1], [0, 0]]) {
    for (const pattern of signPatterns(2)) {
      const result = l1BestResponse([first, second], pattern, 1);
      const v = signedSum([first, second], pattern);
      close(result.optimum, Math.max(...v.map(Math.abs)) / 2, 'the l1 optimum is B||v||_inf/n');
      /* The l1 ball of radius B sits INSIDE the l2 ball of radius B, because
         ||w||_2 <= ||w||_1. So its optimum is at most the Euclidean one, which
         is the same statement as ||v||_inf <= ||v||_2. */
      assert(result.optimum <= bestResponse([first, second], pattern, 1).optimum + 1e-12,
        'the l1 ball of the same radius sits inside the l2 ball, so its optimum is no larger');
      close(result.optimum, (Math.max(...v.map(Math.abs))) / 2, 'and it is exactly B||v||_inf/n');
      // The named vertex is genuinely optimal, and the ties are complete.
      const values = v.map((_value, index) => Math.abs(v[index]));
      const best = Math.max(...values);
      assert.deepEqual(result.tiedCoordinates,
        values.map((value, index) => (Math.abs(value - best) <= 1e-12 ? index : -1)).filter(index => index >= 0),
        'every tied coordinate is named, not just the first');
      if (result.infinityNorm > 1e-15) {
        close(dot(result.optimizer, v) / 2, result.optimum, 'and the named vertex attains the optimum');
      }
      l1Cases += 1;
    }
  }
}
assert(l1Cases >= 80, `only ${l1Cases} l1 cases ran`);
record('l1 geometry');

/* ============================================== §5 · Massart and Sauer */

/** Massart by the exponential-moment route: minimise ln(M)/lambda +
 *  lambda A^2/2 over lambda, then divide by n. */
function massartByMomentGeneratingFunction(count, radius, n) {
  if (count === 1 || radius === 0) return 0;
  const lambda = Math.sqrt(2 * Math.log(count)) / radius;
  return (Math.log(count) / lambda + (lambda * radius * radius) / 2) / n;
}

let massartCases = 0;
for (const count of [1, 2, 3, 4, 8, 16, 100, 1000, 1008]) {
  for (const n of [1, 3, 10, 100, 200, 240]) {
    close(massartSignBound(count, n), massartByMomentGeneratingFunction(count, Math.sqrt(n), n),
      `Massart M=${count}, n=${n}: the exponential-moment route agrees`, 1e-12);
    if (count > 1) {
      close(massartSignBound(count, n), Math.sqrt((2 * Math.log(count)) / n),
        `Massart M=${count}, n=${n}: the familiar sign-valued form`);
    }
    massartCases += 1;
  }
}
assert(massartCases >= 54, `only ${massartCases} Massart cases ran`);
close(massartSignBound(1, 100), 0, 'a singleton class has bound exactly zero, with no division by zero', 0);
close(massartBound(5, 0, 10), 0, 'and a zero radius likewise', 0);
close(massartSignBound(16, 100), checked.massart.M16_n100, 'the packet\'s M=16, n=100 value');
close(massartSignBound(8, 200), checked.practice.massart_M8_n200, 'the packet\'s M=8, n=200 value');
close(Number(massartSignBound(16, 100).toFixed(6)), 0.235482, 'which the manuscript prints as .235482');
close(Number(massartSignBound(8, 200).toFixed(6)), 0.144203, 'and .144203');
assert(massartSignBound(1008, 200) > massartSignBound(8, 200),
  'counting duplicate copies inside the logarithm gives a looser bound, never a tighter one');
refuses(() => massartBound(0, 1, 10), 'a class with no members');
refuses(() => massartBound(5, -1, 10), 'a negative radius');
refuses(() => massartBound(5, 1, 0), 'a sample of size zero');
record('Massart');

/** Sauer's count by Pascal's triangle, not by the multiplicative recurrence. */
function sauerByPascal(d, n) {
  const rows = [[1]];
  for (let index = 1; index <= n; index += 1) {
    const previous = rows[index - 1];
    const row = [1];
    for (let position = 1; position < index; position += 1) row.push(previous[position - 1] + previous[position]);
    row.push(1);
    rows.push(row);
  }
  let total = 0;
  for (let j = 0; j <= d; j += 1) total += rows[n][j];
  return total;
}

let sauerCases = 0;
for (let d = 0; d <= 8; d += 1) {
  for (const n of [1, 2, 5, 10, 20, 40]) {
    const result = sauerCount(d, n);
    if (d === 0) {
      assert.equal(result.exactCount, 1, 'a VC dimension of zero gives one prediction pattern');
      close(result.bound, 0, 'and complexity bound exactly zero', 0);
    } else if (d > n) {
      assert.equal(result.exactCount, 2 ** n, 'beyond d = n the honest count is every labelling');
      assert.equal(result.usable, 'all-labelings', 'and the simplified expression is withheld');
      assert.equal(result.simplified, null, 'explicitly, rather than being computed out of range');
    } else {
      close(result.exactCount, sauerByPascal(d, n), `Sauer d=${d}, n=${n}: Pascal's triangle agrees`, 1e-12);
      assert(result.exactCount <= result.simplified + 1e-6,
        `Sauer d=${d}, n=${n}: the exact count is under (en/d)^d`);
      close(result.bound, Math.sqrt((2 * d * Math.log((Math.E * n) / d)) / n),
        `Sauer d=${d}, n=${n}: the combined bound`);
    }
    sauerCases += 1;
  }
}
assert(sauerCases >= 54, `only ${sauerCases} Sauer cases ran`);
refuses(() => sauerCount(-1, 10), 'a negative VC dimension');
refuses(() => sauerCount(1, 0), 'a sample of size zero');
record('Sauer');

/* ========================================== §6 · losses, slopes and ramps */

// The Lipschitz constants the lesson's table states, checked as the actual
// supremum of the slope over a dense grid of the stated domain.
let slopeSamples = 0;
let largestLogisticSlope = 0;
let largestSigmoidSlope = 0;
for (let step = -8000; step <= 8000; step += 1) {
  const value = step / 1000;
  largestLogisticSlope = Math.max(largestLogisticSlope, Math.abs(logisticMarginSlope(value)));
  largestSigmoidSlope = Math.max(largestSigmoidSlope, sigmoidSlope(value));
  // The slope function really is the derivative of the loss.
  const h = 1e-6;
  close((logisticMarginLoss(value + h) - logisticMarginLoss(value - h)) / (2 * h), logisticMarginSlope(value),
    `the stated logistic slope is its derivative at m=${value}`, 1e-4);
  close((sigmoid(value + h) - sigmoid(value - h)) / (2 * h), sigmoidSlope(value),
    `the stated sigmoid slope is its derivative at s=${value}`, 1e-4);
  slopeSamples += 1;
}
assert(slopeSamples >= 16000, `only ${slopeSamples} slope samples ran`);
assert(largestLogisticSlope < 1 && largestLogisticSlope > 0.999,
  `the logistic margin loss approaches slope 1 without reaching it; observed ${largestLogisticSlope}`);
close(largestSigmoidSlope, 0.25, 'the sigmoid probability map is exactly 1/4-Lipschitz', 1e-9);
assert(largestLogisticSlope > 3 * largestSigmoidSlope,
  'so the two constants are genuinely different, which is the point of the table');
record('Lipschitz constants');

// The hinge is 1-Lipschitz and unbounded below; both are stated in the table.
assert(hingeMarginLoss(-10) === 11, 'the hinge is unbounded as the margin falls');
assert(hingeMarginLoss(5) === 0, 'and flat above margin 1');
record('Lipschitz constants');

/* --- the rule a figure DRAWS is the rule a lab APPLIES, over the whole grid */

let rampGridCases = 0;
let rampKnotCases = 0;
for (let rhoStep = 1; rhoStep <= 30; rhoStep += 1) {
  const rho = rhoStep / 10;
  // Sample the drawn curve exactly as the figure does, then compare it with the
  // applied function at the same inputs. This is the whole enterable rho grid,
  // not one example.
  const drawn = rampCurve(rho, { width: 300, height: 170 });
  drawn.values.forEach(([input, value]) => {
    close(value, rampLoss([input], rho)[0], `rho=${rho}: the drawn curve equals the applied ramp at m=${input}`, 0);
    rampGridCases += 1;
  });
  // The two knots are exact, with no floating-point slack deciding a tie.
  assert.equal(rampLoss([0], rho)[0], 1, `rho=${rho}: a zero margin is charged in full`);
  assert.equal(rampLoss([rho], rho)[0], 0, `rho=${rho}: a margin at rho is charged nothing`);
  assert.equal(rampLoss([-0], rho)[0], 1, `rho=${rho}: negative zero is charged in full too`);
  rampKnotCases += 3;
  // And the ramp upper-bounds the mistake indicator under the lesson's tie rule
  // everywhere, which is the claim the theorem uses.
  for (let step = -300; step <= 300; step += 1) {
    const margin = step / 100;
    const mistake = margin <= 0 ? 1 : 0;
    assert(rampLoss([margin], rho)[0] >= mistake - 1e-15,
      `rho=${rho}: the ramp upper-bounds the mistake indicator at m=${margin}`);
    assert(rampLoss([margin], rho)[0] >= 0 && rampLoss([margin], rho)[0] <= 1,
      `rho=${rho}: the ramp stays in [0, 1] at m=${margin}`);
  }
}
assert(rampGridCases >= 7000, `only ${rampGridCases} drawn-versus-applied ramp comparisons ran`);
assert.equal(rampKnotCases, 90, 'the knots were checked at every rho on the grid');
record('ramp');
refuses(() => rampLoss([0.5], 0), 'a zero margin threshold');
refuses(() => rampLoss([0.5], -1), 'a negative margin threshold');

vector(rampLoss(recordedFixtures.scalarMargins, 0.5), checked.margins.ramp, 'the packet\'s default ramp');
close(rampLoss(recordedFixtures.scalarMargins, 0.5).reduce((sum, value) => sum + value, 0) / 4,
  checked.margins.ramp_mean, 'and its mean');
vector(rampLoss(recordedFixtures.scalarMargins, 0.25), checked.scalar_margin_fixture.smaller_rho_ramp,
  'the packet\'s rho = .25 ramp');
vector(rampLoss([0.2, 0.1, 0.4, 1.2], 0.5), checked.scalar_margin_fixture.edited_first_ramp,
  'the packet\'s edited-first-input ramp');
vector(rampLoss([-0.1, 0.2, 0.8], 0.4), checked.practice.ramp, 'and practice 5\'s ramp');
record('ramp');

// The scale null: multiplying scores, budget and threshold by one factor moves
// nothing. Checked at every factor on a grid, not at one.
let scaleNulls = 0;
for (let step = 1; step <= 30; step += 1) {
  const factor = step / 10;
  /* The null is: hold the INPUTS fixed and multiply the coefficient, the budget
     and the threshold by one factor. The feature energy is a property of the
     inputs, so it does not move; the margins scale, and the ramp sees them at a
     threshold that scaled with them. Scaling the inputs as well would be a
     different experiment, and not a null. */
  const inputs = recordedFixtures.scalarMargins;
  const energy = featureEnergy(inputs.map(value => [value]));
  const first = marginBound({
    margins: inputs.map(value => 1 * value), radius: 1, energy, rho: 0.5, delta: 0.05, comparisons: 1,
  });
  const second = marginBound({
    margins: inputs.map(value => factor * value), radius: factor, energy,
    rho: factor * 0.5, delta: 0.05, comparisons: 1,
  });
  vector(second.ramp, first.ramp, `scaling by ${factor} leaves every ramp loss alone`, 1e-12);
  close(second.complexityAddend, first.complexityAddend, `and the complexity addend`, 1e-12);
  close(second.raw, first.raw, `and the whole expression`, 1e-12);
  scaleNulls += 1;
}
assert.equal(scaleNulls, 30, 'the scale null was checked across the whole factor grid');
record('scale null');

close(featureEnergy(recordedFixtures.scalarMargins.map(value => [value])), checked.scalar_margin_fixture.energy,
  'the scalar fixture\'s feature energy');
close(2 * featureEnergy(recordedFixtures.scalarMargins.map(value => [value])) / 0.5,
  checked.scalar_margin_fixture.default_complexity_addend, 'and its complexity addend');
record('margin bound');

// The confidence term, and what K costs.
close(confidenceAddend(10, 0.05, 240), experiment.confidence_addend, 'the experiment\'s confidence term');
close(confidenceAddend(10, 0.05, 240), 3 * Math.sqrt(Math.log(400) / 480), 'by the closed form');
for (const comparisons of [1, 2, 5, 10, 20]) {
  for (const delta of [0.01, 0.05, 0.1, 0.2]) {
    for (const n of [4, 10, 240]) {
      const value = confidenceAddend(comparisons, delta, n);
      close(value, 3 * Math.sqrt(Math.log((2 * comparisons) / delta) / (2 * n)),
        `confidence K=${comparisons}, delta=${delta}, n=${n}`);
      assert(value > 0, 'the confidence term is strictly positive');
      if (comparisons > 1) {
        assert(value > confidenceAddend(1, delta, n),
          'predeclaring more comparisons costs more, it does not cost less');
      }
      record('confidence');
    }
  }
}
refuses(() => confidenceAddend(0, 0.05, 10), 'zero predeclared comparisons');
refuses(() => confidenceAddend(1, 0, 10), 'a zero failure allowance');
refuses(() => confidenceAddend(1, 1, 10), 'a failure allowance of one');
refuses(() => confidenceAddend(1, 0.05, 0), 'an empty sample');

// `informative` is exactly "raw < 1", swept over a grid that crosses 1.
let informativeCases = 0;
for (let ramp = 0; ramp <= 10; ramp += 1) {
  for (let addend = 0; addend <= 10; addend += 1) {
    const margins = [1, 1, 1, 1].map(() => 10);
    const result = marginBound({
      margins: [-1, ramp / 10 - 0.5, 1, 2], radius: addend / 10, energy: 1, rho: 1, delta: 0.05, comparisons: 1,
    });
    assert.equal(result.informative, result.raw < 1,
      'informative is exactly the statement that the raw sum is below the trivial ceiling');
    close(result.clipped, Math.min(1, result.raw), 'and the clipped value is the minimum with 1');
    close(result.raw, result.empiricalRamp + result.complexityAddend + result.confidence,
      'the raw sum is the sum of its three displayed parts', 1e-12);
    assert(margins.length === 4, 'the fixture is the size the loop assumes');
    informativeCases += 1;
  }
}
assert.equal(informativeCases, 121, 'the vacuity flag was swept over the whole grid');
record('vacuity flag');

/* ========================== the movement verdict, over its whole input grid */

/**
 * The property gotcha 9 asks for, asserted rather than exemplified: the verdict
 * says "unchanged" exactly when the DISPLAYED values are equal. Swept over a
 * grid that deliberately includes pairs differing only below the displayed
 * precision, exact ties, and sign changes.
 */
let movementCases = 0;
let belowPrecisionSeen = 0;
let unchangedSeen = 0;
let movedSeen = 0;
for (let a = -20; a <= 20; a += 1) {
  for (const offset of [0, 1e-12, 1e-9, 1e-7, 5e-7, 1e-6, 1e-4, 0.1, -1e-7, -0.1]) {
    const before = a / 7;
    const after = before + offset;
    const result = compareOutcome(before, after);
    const displayedEqual = displayValue(before) === displayValue(after);
    assert.equal(result.outcome === 'unchanged', displayedEqual,
      `the verdict says unchanged exactly when the displayed values are equal (${before} -> ${after})`);
    if (result.outcome !== 'unchanged') {
      assert.equal(result.outcome, displayValue(after) > displayValue(before) ? 'increased' : 'decreased',
        'and names the direction from the displayed values too');
      movedSeen += 1;
    } else {
      unchangedSeen += 1;
    }
    assert.equal(result.belowDisplayPrecision, result.outcome === 'unchanged' && before !== after,
      'a difference too small to display is reported as such, not called an exact null');
    if (result.belowDisplayPrecision) belowPrecisionSeen += 1;
    movementCases += 1;
  }
}
assert(movementCases >= 400, `only ${movementCases} movement cases ran`);
assert(unchangedSeen > 0 && movedSeen > 0,
  'both verdicts were exercised, so the equivalence is not vacuously true on one side');
assert(belowPrecisionSeen > 0,
  'the below-display-precision case was exercised, which is the case a naive tolerance gets wrong');
record('movement verdict');

/* ============ every graded comparison is graded at the printed precision */

/**
 * The property behind blocking finding B1, asserted over the whole enterable
 * grid rather than at the site where it was found.
 *
 * `gapOutcome` must return the `equal` label exactly when the two operands
 * PRINT the same at the precision that stage displays. The reviewer found 615
 * reachable three-decimal inputs whose exact complexity and energy bound print
 * identically at six decimals and yet graded as `below` under the old
 * hand-written 1e-12. This sweep walks the same space and requires zero.
 */
let gapCases = 0;
let gapTies = 0;
let gapStrict = 0;
for (let first = -3; first <= 3; first += 0.5) {
  for (let second = -3; second <= 3; second += 0.5) {
    for (const nudge of [0, 0.001, 0.002, 0.01, 0.1]) {
      for (const radius of [0.25, 1, 4]) {
        const vectors = [[1, first === 0 ? 0.001 : first], [nudge, second === 0 ? 1 : second]];
        const model = linearComplexity(vectors, radius);
        const classified = gapOutcome(model.energyUpper, model.complexity,
          { digits: 6, below: 'below', equal: 'equal', above: 'above' });
        const printsTheSame = displayValue(model.energyUpper, 6) === displayValue(model.complexity, 6);
        assert.equal(classified.outcome === 'equal', printsTheSame,
          `gapOutcome must say "equal" exactly when the operands print the same: `
          + `${model.complexity} vs ${model.energyUpper} at ${JSON.stringify(vectors)}, B=${radius}`);
        // And the exact quantity never exceeds its own upper bound, so "above"
        // must be unreachable here -- if it ever fires, the bound is wrong.
        assert.notEqual(classified.outcome, 'above', 'the exact value never exceeds the energy bound');
        if (printsTheSame) gapTies += 1; else gapStrict += 1;
        gapCases += 1;
      }
    }
  }
}
assert(gapCases >= 2000, `only ${gapCases} display-precision grading cases ran`);
assert(gapTies > 0 && gapStrict > 0,
  'both sides of the equivalence were exercised, so it is not vacuously true on one side');
record('display-precision grading');

// The reviewer's exact reachable case, pinned by value.
{
  const model = linearComplexity([[1, 0], [0.001, 1]], 1);
  assert.equal(displayValue(model.complexity, 6), displayValue(model.energyUpper, 6),
    'the reviewer\'s reachable input still prints both quantities identically');
  assert(Math.abs(model.energyUpper - model.complexity) > 1e-9,
    'and they still differ in exact arithmetic, so this is the hard case and not a trivial tie');
  assert.equal(gapOutcome(model.energyUpper, model.complexity,
    { digits: 6, below: 'below', equal: 'equal', above: 'above' }).outcome, 'equal',
  'B1: one keystroke from a shipped preset must not grade a visible tie as a miss');
  record('display-precision grading');
}

// The kernel stage prints ten decimals, so it is graded at ten.
for (let step = 0; step <= 1000; step += 1) {
  const r = step / 1000;
  const model = kernelComplexity([[1, r], [r, 1]], 1);
  const classified = gapOutcome(model.traceUpper, model.complexity,
    { digits: 10, below: 'below', equal: 'equal', above: 'above' });
  assert.equal(classified.outcome === 'equal',
    displayValue(model.traceUpper, 10) === displayValue(model.complexity, 10),
    `the kernel stage grades at the ten decimals it prints, at r=${r}`);
  gapCases += 1;
}
record('display-precision grading');

/* =============================== §7 · the recorded Monte-Carlo corrections */

nonEmpty(recordedMonteCarlo.table, 4, 'recorded Monte-Carlo rows');
recordedMonteCarlo.table.forEach((row, index) => {
  const packet = checked.monte_carlo[index];
  assert.equal(row.draws, packet.draws, `Monte-Carlo row ${index}: the draw count`);
  assert.equal(row.seed, packet.seed, `Monte-Carlo row ${index}: the seed is recorded`);
  close(row.estimate, packet.estimate, `Monte-Carlo row ${index}: the estimate`);
  close(row.perDrawUpper, packet.per_draw_upper, `Monte-Carlo row ${index}: the per-draw ceiling`);
  close(row.correction, hoeffdingCorrection(row.perDrawUpper, row.draws, row.eta),
    `Monte-Carlo row ${index}: the correction recomputed from the definition`);
  close(row.correction, packet.one_sided_correction, `Monte-Carlo row ${index}: and the packet's correction`);
  const endpoint = monteCarloEndpoint({
    estimate: row.estimate, perDrawUpper: row.perDrawUpper, draws: row.draws, eta: row.eta,
  });
  close(endpoint.endpoint, packet.upper, `Monte-Carlo row ${index}: the endpoint`);
  close(endpoint.endpoint, row.endpoint, `Monte-Carlo row ${index}: as carried in the module`);
  assert(row.endpoint >= recordedMonteCarlo.exact,
    `Monte-Carlo row ${index}: this endpoint covers the exact value, which is what the figure claims`);
  assert(typeof row.standardError === 'number' && row.standardError > 0,
    `Monte-Carlo row ${index}: a Monte-Carlo estimate reports its own uncertainty`);
  record('Monte Carlo');
});
// The teaching point: the correction shrinks monotonically, the estimate does not.
for (let index = 1; index < recordedMonteCarlo.table.length; index += 1) {
  assert(recordedMonteCarlo.table[index].correction < recordedMonteCarlo.table[index - 1].correction,
    'more draws always shrink the deterministic correction');
}
const estimates = recordedMonteCarlo.table.map(row => row.estimate);
assert(estimates.some((value, index) => index > 0
  && Math.abs(value - recordedMonteCarlo.exact) > Math.abs(estimates[index - 1] - recordedMonteCarlo.exact)),
'and at least one estimate moves AWAY from the exact value, which is what the figure is for');
record('Monte Carlo');
// The fixture really has exact complexity .5, so the reference line is right.
close(linearComplexity(recordedMonteCarlo.fixture, recordedMonteCarlo.radius).complexity,
  recordedMonteCarlo.exact, 'the Monte-Carlo figure\'s reference line is the fixture\'s exact value');
close(recordedMonteCarlo.table[0].perDrawUpper,
  (recordedMonteCarlo.radius * recordedMonteCarlo.fixture.reduce((sum, row) => sum + norm2(row), 0))
  / recordedMonteCarlo.fixture.length, 'and Q is B times the mean row norm');
record('Monte Carlo');
refuses(() => hoeffdingCorrection(1, 0, 0.05), 'zero draws');
refuses(() => hoeffdingCorrection(1, 10, 0), 'a zero failure allowance');
refuses(() => hoeffdingCorrection(-1, 10, 0.05), 'a negative range');

// The unit-ball estimate on the real fit rows, and why the analytic bound wins.
close(recordedMonteCarlo.unitBall.correction,
  hoeffdingCorrection(recordedMonteCarlo.unitBall.perDrawUpper, recordedMonteCarlo.unitBall.draws,
    recordedMonteCarlo.unitBall.eta), 'the unit-ball correction');
assert(recordedMonteCarlo.unitBall.endpoint > experimentSettings.energyFactor,
  'the corrected Monte-Carlo endpoint is worse than the analytic energy bound, so the lesson uses the analytic one');
record('Monte Carlo');

/* ======================================= §8 · the real experiment, replayed */

const csv = fs.readFileSync('public/learn-assets/rademacher/banknote-subset.csv', 'utf8').trim().split(/\r?\n/);
const header = csv[0].split(',');
assert.deepEqual(header, ['source_row', 'split', 'variance', 'skewness', 'curtosis', 'entropy', 'class'],
  'the served CSV has the schema the module was generated from');
const csvRows = csv.slice(1).map(line => line.split(','));
assert.equal(csvRows.length, 480, 'and 480 data rows');
assert.equal(observations.length, 480, 'which the module carries in full');
csvRows.forEach((row, index) => {
  assert.equal(Number(row[0]), observations[index][0], `row ${index}: the source id came from the CSV`);
  for (let column = 0; column < 4; column += 1) {
    close(Number(row[2 + column]), observations[index][1 + column], `row ${index}: feature ${column}`, 0);
  }
  assert.equal(2 * Number(row[6]) - 1, observations[index][5], `row ${index}: the label is the mapped class code`);
});
record('served data');
assert.equal(crypto.createHash('sha256')
  .update(fs.readFileSync('public/learn-assets/rademacher/banknote-subset.csv')).digest('hex'),
provenance.sha256, 'the served file matches the hash the page prints');
assert.equal(fs.statSync('public/learn-assets/rademacher/banknote-subset.csv').size, provenance.bytes,
  'and the byte count');
record('served data');

// The frozen representation, recomputed from the first eighty rows.
const rawFeatures = observations.map(row => row.slice(1, 5));
const labels = observations.map(row => row[5]);
const mean = [0, 1, 2, 3].map(column =>
  rawFeatures.slice(0, 80).reduce((sum, row) => sum + row[column], 0) / 80);
const scale = [0, 1, 2, 3].map(column => Math.sqrt(
  rawFeatures.slice(0, 80).reduce((sum, row) => sum + (row[column] - mean[column]) ** 2, 0) / 80));
vector(mean, representation.mean, 'the representation mean, recomputed from the served rows', 1e-12);
vector(scale, representation.scale, 'and its scale', 1e-12);
vector(mean, experiment.representation.mean, 'which is what the packet recorded', 1e-9);
record('representation');

const mapped = rawFeatures.map(row => mapFeatures(row, representation));
assert(mapped.every(row => row.length === 5), 'every mapped row has five coordinates');
assert(mapped.every(row => row[4] === 1), 'the fifth is the constant intercept');
assert(mapped.every(row => row.slice(0, 4).every(value => value >= -1 && value <= 1)),
  'every mapped feature coordinate is clipped into [-1, 1]');
const largestNorm = Math.max(...mapped.map(norm2));
assert(largestNorm <= Math.sqrt(5) + 1e-12,
  `every mapped row has norm at most sqrt(5); the largest is ${largestNorm}`);
close(experimentSettings.globalRowNormUpper, Math.sqrt(5), 'and the module states that bound');
record('representation');
refuses(() => mapFeatures([1, 2, 3], representation), 'a row of the wrong length');
refuses(() => mapFeatures([1, 2, 3, 4], { mean, scale: [1, 0, 1, 1] }), 'a zero fitted scale');

const fitRows = mapped.slice(80, 320);
const fitLabels = labels.slice(80, 320);
close(featureEnergy(fitRows), experimentSettings.energyFactor, 'the feature-energy factor, recomputed');
close(featureEnergy(fitRows), experiment.energy_factor, 'and the packet\'s value');
record('representation');

nonEmpty(fittedModels, 5, 'fitted candidates');
fittedModels.forEach((model, index) => {
  const packet = experiment.models[index];
  assert.equal(model.radius, packet.radius, `candidate ${index}: the budget`);
  vector(model.weights, packet.weights, `candidate ${index}: the coefficients`, 1e-15);
  close(norm2(model.weights), model.norm, `candidate ${index}: the module's norm is the coefficients' norm`, 1e-12);
  assert(norm2(model.weights) <= model.radius + 1e-12, `candidate ${index}: the coefficients are feasible`);
  for (const [name, role] of [['fit', roles[1]], ['validation', roles[2]], ['assessment', roles[3]]]) {
    const evaluation = evaluatePredictor(mapped.slice(role.from, role.to), labels.slice(role.from, role.to),
      model.weights);
    assert.equal(evaluation.errors, model[name].errors,
      `candidate ${index}: ${name} mistakes recomputed from the served rows`);
    assert.equal(evaluation.errors, packet[name].errors, `candidate ${index}: and the packet's ${name} count`);
    close(evaluation.logLoss, model[name].logLoss, `candidate ${index}: ${name} log loss`, 1e-12);
    vector(evaluation.margins, packet[name].margins, `candidate ${index}: every ${name} margin`, 1e-12);
    assert.deepEqual(evaluation.predictions, packet[name].predictions,
      `candidate ${index}: every ${name} prediction, under the score>=0 tie rule`);
    record('real predictor');
  }
  model.bounds.forEach((bound, boundIndex) => {
    const packetBound = packet.bounds[boundIndex];
    const margins = evaluatePredictor(fitRows, fitLabels, model.weights).margins;
    const recomputed = marginBound({
      margins, radius: model.radius, energy: featureEnergy(fitRows), rho: bound.rho,
      delta: experimentSettings.delta, comparisons: experimentSettings.comparisons,
    });
    close(recomputed.empiricalRamp, bound.empiricalRamp, `candidate ${index}, rho=${bound.rho}: the ramp mean`, 1e-12);
    close(recomputed.empiricalRamp, packetBound.empirical_ramp, 'and the packet\'s ramp mean', 1e-12);
    close(recomputed.complexityAddend, bound.complexityAddend, 'the complexity addend', 1e-12);
    close(recomputed.raw, bound.rawUpper, 'and the raw sum', 1e-12);
    close(recomputed.raw, packetBound.raw_upper, 'which is the packet\'s raw sum', 1e-12);
    assert.equal(recomputed.informative, false,
      `candidate ${index}, rho=${bound.rho}: the expression is above the trivial ceiling, as the lesson states`);
    assert(bound.rawUpper > 1, 'so the page must not present it as a certificate');
    record('real bound');
  });
});

// The declared selection rule, and what it is not allowed to see.
const chosen = fittedModels.reduce((best, model) => {
  if (model.validation.errors !== best.validation.errors) {
    return model.validation.errors < best.validation.errors ? model : best;
  }
  if (model.validation.logLoss !== best.validation.logLoss) {
    return model.validation.logLoss < best.validation.logLoss ? model : best;
  }
  return model.radius < best.radius ? model : best;
});
assert.equal(chosen.radius, selection.chosenRadius, 'the recorded selection is what the rule gives');
assert.equal(chosen.radius, experiment.selection.chosen_radius, 'and what the packet recorded');
// The rule the lab applies is the same one, and it is structurally incapable of
// reading the answer it is supposed to precede: a candidate carrying an
// assessment field is refused rather than quietly ignored.
assert.equal(selectByValidation(fittedModels.map(model => ({
  radius: model.radius,
  validationErrors: model.validation.errors,
  validationLogLoss: model.validation.logLoss,
}))).radius, selection.chosenRadius, 'the exported selection rule gives the recorded budget');
refuses(() => selectByValidation([{
  radius: 1, validationErrors: 0, validationLogLoss: 0, assessment: { errors: 0 },
}]), 'a candidate carrying an assessment field');
refuses(() => selectByValidation([]), 'a selection with no candidates');
// Its three tie-breaks, exercised at exact ties rather than described.
assert.equal(selectByValidation([
  { radius: 4, validationErrors: 3, validationLogLoss: 0.5 },
  { radius: 2, validationErrors: 3, validationLogLoss: 0.4 },
]).radius, 2, 'an exact tie on mistakes is broken by log loss');
assert.equal(selectByValidation([
  { radius: 4, validationErrors: 3, validationLogLoss: 0.5 },
  { radius: 2, validationErrors: 3, validationLogLoss: 0.5 },
]).radius, 2, 'an exact tie on both is broken by the smaller budget');
assert.equal(selectByValidation([
  { radius: 4, validationErrors: 2, validationLogLoss: 9 },
  { radius: 2, validationErrors: 3, validationLogLoss: 0 },
]).radius, 4, 'and mistakes outrank log loss, not the other way round');
record('selection rule');

// The smallest expression is a different candidate at each rho, and neither is
// the selected one at rho = 1 -- the lesson's central "these are two different
// questions" claim.
const smallestAtOne = fittedModels.reduce((best, model) =>
  (model.bounds[1].rawUpper < best.bounds[1].rawUpper ? model : best));
const smallestAtHalf = fittedModels.reduce((best, model) =>
  (model.bounds[0].rawUpper < best.bounds[0].rawUpper ? model : best));
assert.equal(smallestAtOne.radius, 2, 'the smallest rho=1 expression belongs to B=2');
assert.equal(smallestAtHalf.radius, 1, 'and the smallest rho=.5 expression to B=1');
assert.notEqual(smallestAtOne.radius, selection.chosenRadius,
  'neither of which is the budget the validation rule selects');
record('selection rule');

// Assessment error falls across the whole sweep. The lesson says no turn upward
// was manufactured, so that is asserted rather than described.
for (let index = 1; index < fittedModels.length; index += 1) {
  assert(fittedModels[index].assessment.errors <= fittedModels[index - 1].assessment.errors,
    'assessment error does not turn upward anywhere in this sweep');
}
assert(fittedModels[4].assessment.errors < fittedModels[0].assessment.errors,
  'and it genuinely falls, so the monotonicity claim is not vacuous');
record('measured sweep');

// The majority baseline and the zero candidate, both recomputed.
const fitSum = fitLabels.reduce((sum, value) => sum + value, 0);
assert.equal(fitSum >= 0 ? 1 : -1, majorityBaseline.classSign, 'the majority sign comes from the fit rows');
assert.equal(labels.slice(400, 480).filter(value => value !== majorityBaseline.classSign).length,
  majorityBaseline.assessmentErrors, 'and its assessment mistakes');
for (const [name, role] of [['fit', roles[1]], ['validation', roles[2]], ['assessment', roles[3]]]) {
  const evaluation = evaluatePredictor(mapped.slice(role.from, role.to), labels.slice(role.from, role.to),
    [0, 0, 0, 0, 0]);
  assert.equal(evaluation.errors, zeroCandidate[name].errors,
    `the zero candidate's ${name} mistakes, under the score>=0 tie rule`);
  assert(evaluation.margins.every(margin => margin === 0), 'every margin is exactly zero');
  assert(evaluation.predictions.every(prediction => prediction === 1), 'and every prediction is +1');
  close(rampLoss(evaluation.margins, 1).reduce((sum, value) => sum + value, 0) / evaluation.n,
    zeroCandidate[name].ramp, `and its ramp loss on ${name} is exactly 1`, 0);
  record('degenerate predictor');
}

// The tie rule itself, over the whole enterable score range.
let tieCases = 0;
for (let step = -1000; step <= 1000; step += 1) {
  const score = step / 1000;
  assert.equal(predictSign(score), score >= 0 ? 1 : -1, `the tie rule at score ${score}`);
  tieCases += 1;
}
assert.equal(predictSign(0), 1, 'a score of exactly zero predicts +1');
assert.equal(predictSign(-0), 1, 'and negative zero does too, because -0 >= 0');
assert(tieCases >= 2000, `only ${tieCases} tie-rule cases ran`);
record('tie rule');

// The duplicate-feature disclosure the page prints.
nonEmpty(duplicateFeatureGroups, 4, 'duplicate feature groups');
const seen = new Map();
const foundDuplicates = [];
rawFeatures.forEach((row, index) => {
  const key = row.join('|');
  if (seen.has(key)) foundDuplicates.push([seen.get(key), index]);
  else seen.set(key, index);
});
assert.equal(seen.size, 476, '476 distinct feature vectors among 480 rows');
assert.equal(foundDuplicates.length, 4, 'four repeated pairs, recomputed from the served rows');
foundDuplicates.forEach(([first, second], index) => {
  assert.deepEqual([observations[first][0], observations[second][0]], duplicateFeatureGroups[index].sourceRows,
    `duplicate group ${index}: the source ids the page prints`);
});
assert(!duplicateFeatureGroups.some(group => group.roles.includes('assessment')),
  'no repeated feature group involves assessment, as the page states');
assert.equal(duplicateFeatureGroups.filter(group =>
  group.roles[0] !== group.roles[1] && group.roles.includes('fit') && group.roles.includes('validation')).length, 2,
'two of them cross fitting and validation, as the page states');
record('duplicate disclosure');

// The role allocation is disjoint and covers every row.
const covered = new Set();
roles.forEach(role => {
  for (let index = role.from; index < role.to; index += 1) {
    assert(!covered.has(index), `row ${index} is claimed by two roles`);
    covered.add(index);
  }
  assert.equal(role.to - role.from, role.count, `${role.key}: the declared count matches its span`);
});
assert.equal(covered.size, 480, 'every retained row has exactly one role');
record('roles');

/* ============================================= drawn geometry is asserted */

let geometryChecks = 0;
for (const width of [320, 300, 260, 200, 150]) {
  // The number line: points in order, inside the drawing, ends inside the axis.
  const line = numberLineLayout(THREE, { width });
  line.points.forEach((point, index) => {
    assert(point.x >= 0 && point.x <= width, `number line at ${width}: point ${index} is inside the drawing`);
    if (index > 0) {
      assert(point.x > line.points[index - 1].x, 'and the inputs are placed in increasing order');
    }
    geometryChecks += 1;
  });

  // The ball: the supporting point is on the circle and along v.
  for (const entry of geometryCases.filter(item => item.vectors[0].length === 2 && item.radius > 0)) {
    for (const pattern of signPatterns(entry.vectors.length)) {
      const geometry = ballGeometry(entry.vectors, pattern, entry.radius, { width, height: width });
      const centre = geometry.origin;
      if (geometry.support) {
        const distance = Math.hypot(geometry.support.point.x - centre.x, geometry.support.point.y - centre.y);
        close(distance, geometry.radiusPixels,
          `${entry.name} at ${width}: the supporting point is drawn ON the circle`, 1e-9);
        // And it is drawn along the drawn sum, not merely at the right distance.
        const sumAngle = Math.atan2(centre.y - geometry.sum.tip.y, geometry.sum.tip.x - centre.x);
        const supportAngle = Math.atan2(centre.y - geometry.support.point.y, geometry.support.point.x - centre.x);
        close(Math.cos(sumAngle - supportAngle), 1,
          `${entry.name} at ${width}: and along the drawn signed sum`, 1e-9);
      }
      /* The supporting point's LABEL must not sit on the signed-sum arrow.
         The point lies on that arrow by construction, so a label offset along
         the arrow's own direction is a label with a line through it -- which is
         what the browser inspector found. The perpendicular distance from the
         origin-to-v segment is asserted here so the repair cannot regress. */
      if (geometry.supportLabel) {
        const v = geometry.sum.tip;
        const o = geometry.origin;
        const dx = v.x - o.x;
        const dy = v.y - o.y;
        const length = Math.hypot(dx, dy);
        if (length > 1e-6) {
          const distance = Math.abs(dx * (o.y - geometry.supportLabel.y) - (o.x - geometry.supportLabel.x) * dy)
            / length;
          assert(distance >= 12,
            `${entry.name} at ${width}: the supporting-point label sits ${distance.toFixed(1)}px from the `
            + 'signed-sum arrow, so the arrow is drawn through it');
        }
        assert(geometry.supportLabel.x > 0 && geometry.supportLabel.x < width
          && geometry.supportLabel.y > 0 && geometry.supportLabel.y < width,
        `${entry.name} at ${width}: the supporting-point label left the viewBox`);
        geometryChecks += 1;
      }
      // Both axes share one scale, so the circle is a circle.
      const horizontal = geometry.project([1, 0]).x - centre.x;
      const vertical = centre.y - geometry.project([0, 1]).y;
      close(horizontal, vertical, `${entry.name} at ${width}: the two axes use one scale`, 1e-9);
      // Nothing is drawn outside the box.
      [geometry.sum.tip, ...geometry.arrows.map(arrow => arrow.tip)].forEach(point => {
        assert(point.x >= -1e-9 && point.x <= width + 1e-9 && point.y >= -1e-9 && point.y <= width + 1e-9,
          `${entry.name} at ${width}: a drawn tip left the viewBox`);
      });
      geometryChecks += 1;
    }
  }

  // The same properties over a GENERATED grid of samples, not only the named
  // ones: a supporting point that is right on five hand-picked fixtures and
  // wrong elsewhere is the defect this sweep exists to find.
  for (let firstX = -3; firstX <= 3; firstX += 1.5) {
    for (let secondY = -3; secondY <= 3; secondY += 1.5) {
      for (const radius of [0.5, 2]) {
        const vectors = [[firstX, 0.7], [0.4, secondY]];
        for (const pattern of signPatterns(2)) {
          const geometry = ballGeometry(vectors, pattern, radius, { width, height: width });
          if (geometry.support) {
            close(Math.hypot(geometry.support.point.x - geometry.origin.x,
              geometry.support.point.y - geometry.origin.y), geometry.radiusPixels,
            `generated sample at ${width}: the supporting point is on the circle`, 1e-9);
            close(norm2(geometry.support.vector), radius,
              'and its model-space length is the budget', 1e-9);
            close(dot(geometry.support.vector, geometry.best.v) / 2, geometry.best.optimum,
              'and it attains the optimum', 1e-9);
          }
          assert(geometry.radiusPixels <= Math.min(width, width) / 2,
            `generated sample at ${width}: the ball fits inside the drawing`);
          geometryChecks += 1;
        }
      }
    }
  }
}
assert(geometryChecks >= 400, `only ${geometryChecks} layout checks ran`);
record('drawn geometry');

// The diamond really is the l1 ball, and its vertices are where the l1 optimum
// puts its coefficient.
for (const radius of [0.5, 1, 2]) {
  const geometry = ballGeometry([[0.9, 0.4], [0.3, 0.8]], [1, 1], radius, { width: 300, height: 240 });
  const diamond = diamondPoints(radius, geometry);
  assert.equal(diamond.length, 4, 'the diamond has four vertices');
  diamond.forEach(point => {
    const dx = (point.x - geometry.origin.x) / geometry.unit;
    const dy = (geometry.origin.y - point.y) / geometry.unit;
    close(Math.abs(dx) + Math.abs(dy), radius, 'every drawn vertex satisfies |w1| + |w2| = B', 1e-9);
  });
  record('drawn geometry');
}

/* A small-multiple figure's panels must share one scale.
 *
 * Nothing asserted this, which is how figure 8 came to draw the same unit
 * vector at 26.5px in one panel and 53px in the next under the caption "Same
 * lengths". The check is stated as the general property: given a shared
 * extent, every panel's `unit` is identical, and a length drawn in one panel
 * is the same number of pixels as the same length in another. */
{
  const panels = [
    { vectors: [[1, 0], [1, 0]], signs: [1, 1], radius: 1 },
    { vectors: [[1, 0], [0, 1]], signs: [1, 1], radius: 1 },
    { vectors: [[0, 1], [-1, 0]], signs: [1, 1], radius: 1 },
  ];
  const shared = figureExtent(panels);
  const drawn = panels.map(panel =>
    ballGeometry(panel.vectors, panel.signs, panel.radius, { width: 150, height: 150, padding: 22, sharedExtent: shared }));
  nonEmpty(drawn, 3, 'figure 8 panels');
  // Autoscaling really would have disagreed, or this check proves nothing.
  const autoUnits = panels.map(panel =>
    ballGeometry(panel.vectors, panel.signs, panel.radius, { width: 150, height: 150, padding: 22 }).unit);
  assert(Math.max(...autoUnits) / Math.min(...autoUnits) > 1.9,
    'without a shared extent the panels genuinely disagree by about a factor of two, so this guard is not vacuous');
  drawn.forEach((geometry, index) => {
    close(geometry.unit, drawn[0].unit, `figure 8 panel ${index + 1} uses the shared scale`, 0);
    close(geometry.radiusPixels, drawn[0].radiusPixels,
      `figure 8 panel ${index + 1} draws the same budget ball at the same radius`, 0);
    // The same unit input vector must be the same drawn length in every panel.
    const unitLength = Math.hypot(geometry.project([1, 0]).x - geometry.origin.x,
      geometry.project([1, 0]).y - geometry.origin.y);
    close(unitLength, Math.hypot(drawn[0].project([1, 0]).x - drawn[0].origin.x,
      drawn[0].project([1, 0]).y - drawn[0].origin.y),
    `figure 8 panel ${index + 1} draws a unit vector at the same length as panel 1`, 0);
    record('small-multiple scale');
  });
  // And the duplicate panel reports its coincident pair, so the figure can say
  // there are two arrows where a reader can only see one.
  assert.equal(drawn[0].coincidentGroups.length, 1, 'panel 1 has one group of coincident inputs');
  assert.equal(drawn[0].coincidentGroups[0].length, 2, 'and that group holds both of them');
  assert.equal(drawn[1].coincidentGroups.length, 0, 'the perpendicular panel has none');
  record('small-multiple scale');
}
refuses(() => figureExtent([]), 'a small-multiple figure with no panels');

// Nested boxes really nest.
for (const count of [2, 3, 5, 8]) {
  const layout = nestedBoxLayout(count, { width: 320 });
  assert.equal(layout.boxes.length, count, 'one box per class');
  for (let index = 1; index < count; index += 1) {
    const outer = layout.boxes[index - 1];
    const inner = layout.boxes[index];
    assert(inner.x > outer.x && inner.y > outer.y, `box ${index} starts strictly inside box ${index - 1}`);
    assert(inner.x + inner.width < outer.x + outer.width, 'and ends strictly inside it horizontally');
    assert(inner.y + inner.height < outer.y + outer.height, 'and vertically');
    assert(inner.width > 0 && inner.height > 0, 'and has a positive area');
    assert(inner.labelY > outer.labelY, 'and its label sits below the one outside it');
  }
  record('drawn geometry');
}

// Curve sampling: increasing in x, inside the frame, and the polyline has a
// point for every sample.
for (const [name, fn, from, to, range] of [
  ['logistic', logisticMarginLoss, -3, 4, [0, 4.05]],
  ['hinge', hingeMarginLoss, -3, 4, [0, 4.05]],
  ['sigmoid', sigmoid, -4, 4, [0, 1]],
]) {
  const curve = curvePoints(fn, { from, to, width: 300, height: 150, valueRange: range });
  assert.equal(curve.points.length, 97, `${name}: one point per sample, plus the endpoint`);
  assert.equal(curve.polyline.split(' ').length, 97, 'and the polyline carries them all');
  curve.points.forEach((point, index) => {
    if (index > 0) assert(point.x > curve.points[index - 1].x, `${name}: x increases along the curve`);
    assert(point.y >= curve.frame.top - 1e-9 && point.y <= curve.frame.bottom + 1e-9,
      `${name}: the curve stays inside its own frame at sample ${index}`);
    assert(point.x >= curve.frame.left - 1e-9 && point.x <= curve.frame.right + 1e-9,
      `${name}: and inside it horizontally`);
  });
  // The plotted value really is the function's value.
  curve.values.forEach(([input, value]) => close(value, fn(input), `${name}: the plotted value is f(x)`, 0));
  record('drawn curve');
}
refuses(() => curvePoints(sigmoid, { from: 1, to: 1, width: 300, height: 150 }), 'an empty plotting window');
refuses(() => curvePoints(sigmoid, { from: 0, to: 1, samples: 1, width: 300, height: 150 }), 'a single sample');

// The margin histogram: edges exactly at 0 and rho, counts summing to n, and
// the three regions agreeing with the ramp's own classification.
for (const model of fittedModels) {
  const margins = evaluatePredictor(fitRows, fitLabels, model.weights).margins;
  for (const rho of experimentSettings.rhoValues) {
    const histogram = marginHistogram(margins, { rho, bins: 18 });
    assert(histogram.edges.includes(0), `B=${model.radius}, rho=${rho}: there is a bin edge exactly at 0`);
    assert(histogram.edges.includes(rho), 'and one exactly at rho');
    assert.equal(histogram.counts.reduce((sum, value) => sum + value, 0), margins.length,
      'every margin lands in exactly one bin');
    assert.equal(histogram.belowZero, margins.filter(margin => margin <= 0).length,
      'the below-zero count is the ramp\'s own full-charge count');
    assert.equal(histogram.belowZero + histogram.insideRamp + histogram.aboveRho, margins.length,
      'and the three regions partition the sample');
    const ramp = rampLoss(margins, rho);
    assert.equal(histogram.belowZero, ramp.filter(value => value === 1).length,
      'so the drawn regions and the applied ramp agree on which observations are charged in full');
    assert.equal(histogram.aboveRho, ramp.filter(value => value === 0).length, 'and on which are charged nothing');
    record('margin histogram');
  }
}
refuses(() => marginHistogram([1, 2], { rho: 0 }), 'a zero margin threshold in the histogram');

// The stacked bound bars: widths proportional to the terms, the ceiling at 1 on
// the same scale, and a stack longer than the ceiling when the bound is vacuous.
for (const rhoIndex of [0, 1]) {
  const rows = fittedModels.map(model => ({
    radius: model.radius,
    empiricalRamp: model.bounds[rhoIndex].empiricalRamp,
    complexityAddend: model.bounds[rhoIndex].complexityAddend,
    confidence: experimentSettings.confidenceAddend,
  }));
  const layout = stackedBoundLayout(rows, { width: 320 });
  layout.rows.forEach(row => {
    row.segments.forEach(segment => {
      close(segment.width, segment.value * layout.scale, 'each segment is as long as the term it encodes', 1e-9);
      assert(segment.width >= 0, 'and no segment has negative length');
    });
    // The segments abut without gaps or overlaps.
    for (let index = 1; index < row.segments.length; index += 1) {
      close(row.segments[index].x, row.segments[index - 1].x + row.segments[index - 1].width,
        'segments abut exactly', 1e-9);
    }
    close(row.total, row.empiricalRamp + row.complexityAddend + row.confidence, 'the stack total is the raw sum', 1e-12);
    assert(layout.left + row.total * layout.scale > layout.ceilingX,
      'a vacuous bound is drawn LONGER than the ceiling, rather than clipped at it');
    /* The total printed at the end of each bar has to fit inside the drawing.
       Two of these were rendered outside the SVG before the gutter existed, so
       the space the label needs is asserted, not assumed. Six characters of a
       9px monospace figure is about 34px. */
    assert(row.labelX + 34 <= layout.width,
      `the total printed after the B=${row.radius} bar (at x=${row.labelX.toFixed(1)}) would run outside the `
      + `${layout.width}px drawing`);
    assert(row.labelX > layout.left, 'and it is printed after the bar, not before it');
    record('stacked bars');
  });
  close(layout.ceilingX - layout.left, layout.scale, 'the ceiling line sits at exactly 1 on the bars\' own scale', 1e-9);
}
refuses(() => stackedBoundLayout([], {}), 'an empty bound chart');

// The Monte-Carlo layout: four separate intervals, each from its estimate to
// its endpoint, with the exact value as a separate reference.
const mcLayout = monteCarloLayout(recordedMonteCarlo.table, recordedMonteCarlo.exact, { width: 320 });
assert.equal(mcLayout.rows.length, 4, 'four independent intervals');
mcLayout.rows.forEach((row, index) => {
  assert(row.endpointX > row.estimateX, `interval ${index} runs from its estimate up to its endpoint`);
  close(row.estimateX, mcLayout.place(row.estimate), 'the estimate is placed at its own value', 1e-9);
  close(row.endpointX, mcLayout.place(row.endpoint), 'and the endpoint at its own value', 1e-9);
  assert(row.endpointX >= mcLayout.exactX - 1e-9,
    `interval ${index} reaches the exact value, which is what the figure claims`);
  if (index > 0) assert(row.y > mcLayout.rows[index - 1].y, 'and the rows do not overlap');
  record('Monte-Carlo layout');
});
close(mcLayout.exactX, mcLayout.place(recordedMonteCarlo.exact), 'the reference line is at the exact value', 1e-9);
refuses(() => monteCarloLayout([], 0.5, {}), 'an empty Monte-Carlo chart');

// The roles strip: widths in proportion to the counts, abutting, filling the box.
const strip = rolesStrip(roles, { width: 320 });
assert.equal(strip.total, 480, 'the strip covers every retained row');
let cursor = 0;
strip.parts.forEach((part, index) => {
  close(part.x, cursor, `role ${index} starts where the previous one ended`, 1e-9);
  close(part.width, (320 * part.count) / 480, 'and is as wide as its share of the rows', 1e-9);
  cursor += part.width;
  record('roles strip');
});
close(cursor, 320, 'and together they fill the drawing exactly', 1e-9);
refuses(() => rolesStrip([{ key: 'a', count: 0 }], {}), 'an empty allocation');

// The ghost-pair layout: columns inside the box, differences signed correctly.
const ghost = ghostPairLayout([
  { sample: 0.2, ghost: 0.7, sign: 1 }, { sample: 0.9, ghost: 0.4, sign: -1 },
  { sample: 0.5, ghost: 0.5, sign: 1 }, { sample: 0.1, ghost: 0.8, sign: -1 },
], { width: 320 });
assert.equal(ghost.columns.length, 4, 'four pairs');
ghost.columns.forEach((column, index) => {
  assert(column.x >= 0 && column.x + column.width <= 320 + 1e-9, `pair ${index} is inside the drawing`);
  close(column.difference, column.sign * (column.ghost - column.sample), 'the signed difference is sigma(g\' - g)', 0);
  // Swapping exchanges the lanes, and nothing else.
  assert.equal(column.topValue, column.sign === 1 ? column.ghost : column.sample, 'the swap exchanges the lanes');
  assert.equal(column.bottomValue, column.sign === 1 ? column.sample : column.ghost, 'both of them');
  if (index > 0) {
    assert(column.x >= ghost.columns[index - 1].x + ghost.columns[index - 1].width - 1e-9,
      'and the columns do not overlap');
  }
  record('ghost layout');
});
// The equal pair is a genuine zero difference under either sign.
assert.equal(ghost.columns[2].difference, 0, 'an equal pair contributes exactly zero, under either sign');
refuses(() => ghostPairLayout([], {}), 'an empty pair chart');

// The convex hull: the mixture never out-projects the best base vector, at
// every weighting on a grid and every direction on a grid.
let hullCases = 0;
for (let weightStep = 0; weightStep <= 20; weightStep += 1) {
  const first = weightStep / 20;
  const weights = [first, (1 - first) * 0.6, (1 - first) * 0.4];
  for (let angleStep = 0; angleStep < 12; angleStep += 1) {
    const angle = (angleStep * Math.PI) / 6;
    const layout = hullLayout([[1, 0.2], [0.3, 1], [-0.8, 0.5]], weights,
      [Math.cos(angle), Math.sin(angle)], { width: 300, height: 230 });
    assert(layout.mixtureProjection <= layout.bestBaseProjection + 1e-12,
      `a convex average never out-projects the best base vector (weights ${weights}, angle ${angleStep})`);
    // And the mixture really is the weighted average.
    vector(layout.mixture, convexMixture([[1, 0.2], [0.3, 1], [-0.8, 0.5]], weights),
      'the drawn mixture is the convex combination', 1e-12);
    /* Label placement, tested by point-in-polygon rather than by trusting the
       offset arithmetic. Every vertex name must sit OUTSIDE the hull, and the
       mixture name inside it and clear of every edge -- the browser sampler
       found both of these being crossed by the hull outline. */
    const polygon = [[1, 0.2], [0.3, 1], [-0.8, 0.5]].map(layout.project);
    layout.vertexLabels.forEach((label, index) => {
      assert(!insidePolygon(label, polygon),
        `hull vertex label ${index + 1} sits inside the polygon, so an edge is drawn through it`);
      hullLabelChecks += 1;
    });
    /* The mixture point deliberately has no in-plot label; it is named in the
       legend strip. Asserted here so a future edit that reintroduces one has to
       confront the reason it was removed. */
    assert(layout.mixtureLabel === undefined,
      'the mixture point must not carry an in-plot label: no fixed inward offset clears every edge');
    hullLabelChecks += 1;
    hullCases += 1;
  }
}
assert(hullCases >= 240, `only ${hullCases} hull cases ran`);
assert(gapCases >= 3000, `only ${gapCases} display-precision grading cases ran`);
assert(hullLabelChecks >= 960, `only ${hullLabelChecks} hull label placements were checked`);
record('convex hull');
refuses(() => convexMixture([[1, 0], [0, 1]], [0.5, 0.6]), 'weights that do not sum to one');
refuses(() => convexMixture([[1, 0], [0, 1]], [1.5, -0.5]), 'a negative weight');

/* ========================================= input refusal across the board */

refuses(() => signPatterns(0), 'zero observations');
refuses(() => signPatterns(13), 'more observations than exact enumeration supports');
refuses(() => signPatterns(2.5), 'a fractional number of observations');
refuses(() => empiricalComplexity([]), 'an empty class');
refuses(() => empiricalComplexity([[1, 2], [3]]), 'rows of different lengths');
refuses(() => empiricalComplexity([[1, Number.NaN]]), 'a non-finite entry');
refuses(() => empiricalComplexity([[1, Number.POSITIVE_INFINITY]]), 'an infinite entry');
refuses(() => linearComplexity([[1, 0]], -1), 'a negative budget');
refuses(() => linearComplexity([], 1), 'an empty sample');
refuses(() => signedSum([[1, 0], [0, 1]], [1, 0]), 'a sign that is not plus or minus one');
refuses(() => signedSum([[1, 0], [0, 1]], [1]), 'fewer signs than observations');
refuses(() => thresholdRows([]), 'no inputs at all');
refuses(() => thresholdRows([1, Number.NaN]), 'a non-finite input');
refuses(() => evaluatePredictor([[1, 1]], [1], [1]), 'a coefficient of the wrong length');
refuses(() => evaluatePredictor([[1, 1]], [1, 1], [1, 1]), 'more labels than rows');
refuses(() => marginBound({ margins: [1], radius: -1, energy: 1, rho: 1, delta: 0.05, comparisons: 1 }),
  'a negative budget in the bound');
refuses(() => marginBound({ margins: [1], radius: 1, energy: -1, rho: 1, delta: 0.05, comparisons: 1 }),
  'a negative feature energy');
refuses(() => compareOutcome(Number.NaN, 1), 'a comparison against a non-number');

/* ======================================= the page's own stated numbers */

// Values the lesson body prints as prose, each recomputed here. A number in the
// manuscript that this file does not reproduce is a number nobody checked.
const statedInManuscript = [
  ['the threshold class', empiricalComplexity(thresholds).complexity, 0.6666667, '.6666667'],
  ['its mistake class', empiricalComplexity(mistakeRows(thresholds, [1, -1, 1])).complexity, 0.3333333, '.3333333'],
  ['two duplicate unit vectors', linearComplexity([[1, 0], [1, 0]], 1).complexity, 0.5, '(1+0+0+1)/4=.5'],
  ['two perpendicular ones', linearComplexity([[1, 0], [0, 1]], 1).complexity, 0.707107, '1/√2≈.707107'],
  ['their shared energy bound', linearComplexity([[1, 0], [1, 0]], 1).energyUpper, 0.707107,
    'the same energy upper bound .707107'],
  ['the r = 0 kernel', kernelComplexity([[1, 0], [0, 1]], 1).complexity, 0.707107, 'r=0 gives .707107'],
  ['the r = .9 kernel', kernelComplexity([[1, 0.9], [0.9, 1]], 1).complexity, 0.599143, 'r=.9 gives .599143'],
  ['the r = 1 kernel', kernelComplexity([[1, 1], [1, 1]], 1).complexity, 0.5, 'r=1 gives .5'],
  ['sixteen rules on a hundred observations', massartSignBound(16, 100), 0.235482, 'upper bound .235482'],
  ['eight rules on two hundred', massartSignBound(8, 200), 0.144203, '≈.144203'],
  ['the four-margin ramp', rampLoss([-0.2, 0.1, 0.4, 1.2], 0.5).reduce((s, v) => s + v, 0) / 4, 0.5,
    'whose mean is .5'],
  ["practice 5's ramp", rampLoss([-0.1, 0.2, 0.8], 0.4).reduce((s, v) => s + v, 0) / 3, 0.5, 'with mean .5'],
  ['practice 1', empiricalComplexity(thresholdRows([2, 5])).complexity, 0.75, 'Their average is 3/4'],
  ['practice 3', linearComplexity([[3, 0], [0, 4]], 2).complexity, 5, 'so the average is 5'],
  ['the real feature energy', experimentSettings.energyFactor, 0.0799496, '.0799496'],
  ['the real confidence term', experimentSettings.confidenceAddend, 0.335172, '.335172'],
  ['the worked B = 2 bound', fittedModels[3].bounds[1].rawUpper, 1.129848, '1.129848'],
  ['its training ramp', fittedModels[3].bounds[1].empiricalRamp, 0.474878, '.474878'],
  ['the unit-ball Monte-Carlo estimate', recordedMonteCarlo.unitBall.estimate, 0.0720833, '.0720833'],
  ['its corrected endpoint', recordedMonteCarlo.unitBall.endpoint, 0.105220, '.105220'],
];
nonEmpty(statedInManuscript, 20, 'stated values');
for (const [label, actual, expected, manuscriptText] of statedInManuscript) {
  const digits = String(expected).split('.')[1]?.length ?? 0;
  close(Number(actual.toFixed(digits)), expected, `${label}: the model reproduces the manuscript's value`, 0);
  /* The manuscript is frozen, so the phrase it uses is quoted here verbatim. A
     value this file "checks" that the manuscript does not actually contain
     would be a number invented by the implementation. */
  assert(manuscript.includes(manuscriptText),
    `${label}: the frozen manuscript really says "${manuscriptText}"`);
  record('stated value');
}
// The manuscript's five-row experiment table, read out of the frozen text.
const tableRows = [...manuscript.matchAll(/^\| (\.25|\.5|1|2|4) \| (\d+) \| (\d+) \| (\d+) \| ([\d.]+) \|$/gm)];
nonEmpty(tableRows, 5, 'manuscript experiment rows');
tableRows.forEach(match => {
  const radius = Number(match[1].startsWith('.') ? `0${match[1]}` : match[1]);
  const model = fittedModels.find(entry => entry.radius === radius);
  assert(model, `the manuscript's B = ${match[1]} row has a model`);
  assert.equal(model.fit.errors, Number(match[2]), `B = ${match[1]}: the manuscript's training mistakes`);
  assert.equal(model.validation.errors, Number(match[3]), `B = ${match[1]}: its validation mistakes`);
  assert.equal(model.assessment.errors, Number(match[4]), `B = ${match[1]}: its assessment mistakes`);
  close(Number(model.bounds[1].rawUpper.toFixed(6)), Number(match[5]), `B = ${match[1]}: its rho = 1 expression`, 0);
  record('manuscript table');
});

/* ================================ the displayed programs and their outputs */

nonEmpty(Object.keys(rademacherPrograms), 2, 'displayed programs');
const printedSummary = JSON.parse(rademacherPrograms.runCalculations.expected);
close(printedSummary.thresholds, empiricalComplexity(thresholds).complexity,
  'the program printed the threshold value this module computes');
close(printedSummary.classification_loss, empiricalComplexity(mistakeRows(thresholds, [1, -1, 1])).complexity,
  'and the mistake-class value');
close(printedSummary.singleton, 0, 'and exactly zero for the singleton', 0);
close(printedSummary.all_labels, 1, 'and exactly one for the full cube', 0);
const printedExperiment = JSON.parse(rademacherPrograms.runExperiment.expected);
assert.equal(printedExperiment.selected, selection.chosenRadius, 'the program printed the selected budget');
close(printedExperiment.energy_factor, experimentSettings.energyFactor, 'and the energy factor');
printedExperiment.rows.forEach(row => {
  const model = fittedModels.find(entry => entry.radius === row.B);
  assert.equal(model.fit.errors, row.train_errors, `B=${row.B}: the printed training mistakes`);
  assert.equal(model.assessment.errors, row.assessment_errors, `B=${row.B}: the printed assessment mistakes`);
  record('program output');
});
nonEmpty(Object.keys(rademacherExcerpts), 3, 'displayed excerpts');
for (const [key, excerpt] of Object.entries(rademacherExcerpts)) {
  assert(excerpt.code.trim().startsWith('def '), `${key}: the excerpt is a function definition`);
  assert(!excerpt.code.includes('print('), `${key}: displayed teaching code prints no disclaimer`);
  record('displayed excerpt');
}

/* ===================================================== evidence and floors */

const sources = [
  'src/learn/data/rademacher-models.js',
  'src/learn/data/rademacher-data.js',
  'src/learn/data/rademacher-examples.js',
  'src/learn/components/lesson-labs/RademacherShared.jsx',
  'src/learn/components/lesson-labs/RademacherLabs.jsx',
  'src/learn/components/lesson-labs/RademacherFigures.jsx',
  'src/learn/components/lesson-labs/rademacher-labs.css',
  'src/learn/data/topics/rademacher-complexity-generalization-bounds.jsx',
  'src/learn/data/curriculum/blueprints/rademacher-complexity-generalization-bounds.js',
  'public/learn-assets/rademacher/banknote-subset.csv',
  'public/learn-assets/rademacher/ATTRIBUTION.txt',
  'public/learn-assets/rademacher/complexity_calculations.py',
  'public/learn-assets/rademacher/bounded_norm_experiment.py',
  'scripts/verify-rademacher-data.py',
  'scripts/verify-rademacher-examples.py',
  'scripts/verify-rademacher-sources.py',
  'scripts/verify-rademacher-browser.cjs',
  'scripts/falsify-rademacher.mjs',
];
// Unfiltered on purpose: silently dropping a declared source that no longer
// exists and still writing `passed: true` is how a deleted file passes.
const missingSources = sources.filter(file => !fs.existsSync(file));
assert.deepEqual(missingSources, [], `declared source files are missing: ${missingSources.join(', ')}`);

const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
/* A counter that is reported but never floored is decoration: deleting every
 * `record()` call still printed PASS with a smaller number. These floors sit
 * below the current values but far above zero, so a wholesale loss of coverage
 * fails the run instead of quietly shrinking the headline. */
assert(total >= 300, `only ${total} grouped checks ran; the suite has lost coverage`);
assert(Object.keys(counts).length >= 30, `only ${Object.keys(counts).length} groups ran`);
assert(kernelCases >= 1000, `only ${kernelCases} kernel cases ran`);
assert(rampGridCases >= 7000, `only ${rampGridCases} drawn-versus-applied ramp comparisons ran`);
assert(movementCases >= 400, `only ${movementCases} movement-verdict cases ran`);
assert(slopeSamples >= 16000, `only ${slopeSamples} slope samples ran`);
assert(geometryChecks >= 400, `only ${geometryChecks} drawn-geometry checks ran`);
assert(hullCases >= 240, `only ${hullCases} convex-hull cases ran`);

const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])),
  packetHashes: {
    'checked-results.json': hash(`${packetDirectory}/checked-results.json`),
    'experiment-results.json': hash(`${packetDirectory}/experiment-results.json`),
    'lesson.md': hash(`${packetDirectory}/lesson.md`),
  },
  verifierHash: hash('scripts/verify-rademacher-models.mjs'),
  counts,
  totalGroupedChecks: total,
  sweeps: {
    kernelSimilarityAndBudgetPairs: kernelCases,
    drawnVersusAppliedRampComparisons: rampGridCases,
    rampKnotChecks: rampKnotCases,
    movementVerdictCases: movementCases,
    lossSlopeSamples: slopeSamples,
    rotationNulls: rotationCases,
    bestResponseCases: responseCases,
    l1Cases,
    massartCases,
    sauerCases,
    drawnGeometryChecks: geometryChecks,
    convexHullCases: hullCases,
    displayPrecisionGradingCases: gapCases,
    tieRuleCases: tieCases,
    vacuityGridCases: informativeCases,
    scaleNulls,
  },
  secondRoutes: [
    'exact BigInt rational enumeration for every finite-class complexity',
    'threshold restrictions rebuilt from the monotone-block characterisation instead of a cutoff sweep',
    'the Euclidean ball recomputed through the Gram-matrix kernel path',
    'the two-point kernel closed form B(sqrt(2+2r) + sqrt(2-2r))/4 against enumeration at 201 similarities',
    "Massart's bound by the exponential-moment expression at the optimising lambda",
    "Sauer's count by Pascal's triangle rather than the multiplicative recurrence",
    'the real experiment replayed from the served CSV rather than read from the module',
    'numeric differentiation of each loss against its stated slope function',
  ],
  scope: 'Browser Rademacher models against the content packet\'s checked-results.json and experiment-results.json, '
    + 'and against every number the frozen manuscript states. The five finite classes are enumerated twice -- once in '
    + 'floating point and once in exact BigInt rational arithmetic -- with every sign pattern, correlation, winner '
    + 'index and per-pattern maximum compared against the packet. The threshold restrictions are rebuilt from a '
    + 'second characterisation. Seven exact nulls, four scalings, the negation symmetry and the absolute-value '
    + 'convention are asserted as identities rather than shown on one example, and the absolute convention is proved '
    + 'to equal the signed value of the negation-closed class on every one of the five. The mistake-class factor of '
    + 'one half is checked at all eight label vectors. Euclidean geometry is recomputed through the kernel path, and '
    + 'the tightness of the energy bound is asserted as an iff against the equal-signed-sum condition. The two-point '
    + 'kernel is checked against its closed form at every similarity on a 201-point grid and five budgets, with r and '
    + '-r shown to be a null. The drawn ramp curve is compared with the applied ramp function at every sample of '
    + 'every threshold on the whole enterable grid, and the movement verdict is proved equivalent to displayed-value '
    + 'equality over a grid that includes differences below the displayed precision. The real experiment is replayed '
    + 'from the served CSV: the representation, every margin, every prediction under the score>=0 tie rule, every '
    + 'mistake count, log loss, ramp mean and bound component, the selection rule, the majority baseline, the zero '
    + 'candidate and the duplicate-feature disclosure. Every drawn coordinate -- supporting points, diamond vertices, '
    + 'nested boxes, curve frames, histogram edges, stacked bars, Monte-Carlo intervals, the roles strip, ghost '
    + 'columns and the convex hull -- is asserted against the quantity it depicts, at five widths where width '
    + 'matters.',
  limitations: [
    'The Monte-Carlo estimates themselves come from a recorded seeded NumPy run; this file checks the '
      + 'deterministic half (the per-draw ceiling, the Hoeffding correction, the endpoint and its coverage of the '
      + 'exact value) and that each row reports its own uncertainty. The estimates are re-derived in '
      + 'scripts/verify-rademacher-data.py.',
    'The five fitted coefficient vectors are taken as given here and checked to be feasible and to reproduce every '
      + 'downstream number; that they are the constrained MINIMUM is checked by an independent projected-gradient '
      + 'solve in scripts/verify-rademacher-data.py.',
    'Displayed program execution is a separate script, as is source hygiene.',
    'Rendering, interaction, visual layout and independent review are separate steps and are not claimed here.',
  ],
  passed: true,
};
/* `--no-evidence` lets an independent reviewer re-run this without writing to
 * the record they are reviewing. Reaching this line means every assertion above
 * passed, so this overwrites the provisional failing record stamped at the top
 * of the run. */
if (writesEvidence) {
  fs.mkdirSync('docs/teaching/evidence', { recursive: true });
  fs.writeFileSync(evidencePath, JSON.stringify(evidence, null, 2) + '\n');
}
console.log(`PASS: ${total} grouped Rademacher model checks across ${Object.keys(counts).length} groups, including `
  + `${kernelCases.toLocaleString('en-US')} kernel cases against a closed form, `
  + `${rampGridCases.toLocaleString('en-US')} drawn-versus-applied ramp comparisons across the whole threshold grid, `
  + `${movementCases.toLocaleString('en-US')} movement verdicts proved equivalent to displayed-value equality, `
  + `${slopeSamples.toLocaleString('en-US')} numerically differentiated loss samples, `
  + `${geometryChecks.toLocaleString('en-US')} drawn-geometry checks and `
  + `${hullCases.toLocaleString('en-US')} convex-hull cases.`);
